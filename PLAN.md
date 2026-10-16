# Mini-GPT: compute-efficiency ablations

## Goal

Find how close a small decoder-only LM trained on one laptop GPU gets to the
compute-optimal frontier, then push that frontier down with ablations.

1. **Measure our frontier.** Run IsoFLOP sweeps on our data and fit L(N, D).
   From the fit, read off the compute-optimal model size N\*(C), token count
   D\*(C), and the best achievable loss at each compute budget C.
2. **Move it.** Score each ablation as an *effective compute multiplier* (ECM):
   the compute the baseline would need to reach the variant's loss, divided by
   the compute the variant used. ECM 1.3 = "worth 30% more compute".
3. **Re-measure.** Stack the winners into a new baseline, rerun the IsoFLOP
   sweep, and report how far the frontier moved.

"Chinchilla frontier" here means Chinchilla's *method* (Hoffmann et al. 2022),
not its numbers. Their absolute losses depend on their data and tokenizer, and
their smallest models (~70M) are bigger than most of ours, so their fitted
constants don't transfer. What we do compare: the compute-optimal
tokens-per-parameter ratio (they found ~20) and the scaling exponents.

## Constraints

- **Hardware:** one RTX 5070 Laptop GPU, 8 GB VRAM, under WSL2. Planning
  assumption until Phase 0 measures it: ~20 TFLOP/s sustained in bf16, about
  7e16 FLOPs per hour. Laptop thermals make the sustained rate lower than the
  burst rate, so measure over 30+ minutes.
- **Time is not the limit; memory is.** The GPU can run for weeks. 8 GB caps
  model size: with AdamW, weights, gradients and two moment buffers in fp32
  take 16 bytes per parameter before activations, so the ceiling is roughly
  200–300M parameters. Phase 0 measures the real cap. Batch size is not
  limited by memory: micro-batches with gradient accumulation reach any batch
  size, and activation checkpointing is allowed. Both change wall-clock, not
  FLOPs.
- **Scale this implies:** single runs of 1e15 to 1e18 FLOPs, and
  compute-optimal models of roughly 3M to 90M parameters (N ≈ sqrt(C / 120)
  when D = 20N). The memory cap sets the top budget: an IsoFLOP curve at 3e18
  would need models around 400M to bracket its minimum. At the small end the
  embedding and LM head are a large share of both parameters and FLOPs, so
  vocab size is a first-order choice.
- **Total:** roughly 250–300 GPU-hours as written, about two weeks of
  continuous running. Phases 2 and 4 are most of it.
- **No custom kernels.** PyTorch SDPA, `torch.compile`, bf16 autocast. Kernels
  change wall-clock, never FLOPs or loss, so they can never change a result.
  Write one only if profiling shows a single op dominating and the fix is cheap.
- **Runs must resume.** A laptop sleeps; every run checkpoints and resumes to
  the same loss curve.

## Out of scope for now

- Post-training. It starts once pretraining is finished, from its own plan.
- Downstream benchmarks (ARC, MMLU, HumanEval). Models this small score near
  chance; validation bits per byte is the signal.
- Multi-GPU, fp8, custom kernels.

## Rules for every experiment

1. **Metric: validation bits per byte (bpb)** on a fixed set of held-out
   *documents* (fixed bytes, not fixed token windows).
   bpb = total NLL in nats / (ln 2 × UTF-8 bytes). It is comparable across
   tokenizers and context lengths. Per-token loss is logged but never used to
   compare runs.
2. **Compute is counted in FLOPs**, never steps or wall-clock. Per training
   token: 6 × (non-embedding params + LM-head params) + 12 × n_layers × d_attn ×
   (mean keys attended per query; ctx/2 for full causal attention). The input
   embedding lookup costs nothing; the LM-head matmul costs a lot at this size.
3. **Tune LR for every variant.** A change that moves the optimal LR looks worse
   at the baseline's LR. Sweep at least 3 LRs (×0.5, ×1, ×2 the current best)
   per variant per scale and compare best against best.
4. **Know the noise floor.** Phase 1 measures seed-to-seed bpb spread. A
   difference under 2× that spread is not a result.
5. **Two scales before adopting.** Screen at 1e16 FLOPs, confirm at 1e17. Many
   tricks shrink or reverse with scale.
6. **One change at a time** against a pinned baseline. Combine only in Phase 4.
7. **Every run is recorded** in one append-only file: config, git commit, FLOPs,
   tokens, params (embedding and non-embedding), final val bpb, loss curve.
   Fits and plots read from that file only.

## Phases

### Phase 0: minimal infrastructure

Build only what Phase 1 needs.

- `tokenizer.py`: byte-level BPE; digits split one per token; special-token
  strings inside text encode as plain text, not as control tokens.
- `data.py`: stream FineWeb-Edu `sample-10BT` (about 10B GPT-2 tokens of raw
  text) → train tokenizer → packed uint16 shards. Keep the
  held-out val documents as raw text so any tokenizer can be scored on them.
  Fetch enough that no run repeats a document: the smallest model in the 1e18
  sweep needs about 7B tokens. If Phase 0 raises the budgets, move to
  `sample-100BT`.
- `model.py`: plain PyTorch GPT built from a config.
- `flops.py`: the FLOPs formula from rule 2.
- `train.py`: train one config to a FLOP budget with a WSD (warmup, stable,
  decay) schedule; log val bpb; append the results row.
- `tests/`: FLOP formula matches a profiler count on a small model; param count
  matches the module; uniform logits give bpb = log2(V) × tokens / bytes; a
  resumed run reproduces the uninterrupted loss curve.

**Done when:** a 1e15-FLOP run trains, loss falls, a results row is written,
sustained TFLOP/s is measured, and peak memory is measured against parameter
count to find the largest model that trains in 8 GB. Replace the planning
assumptions above with the measured numbers and re-size the budgets below if
they're far off.

### Phase 1: baseline and noise floor

Baseline, a standard modern small GPT: pre-norm RMSNorm, RoPE, SwiGLU (8/3
hidden ratio), no biases, full causal multi-head attention, tied embeddings,
16K vocab, context 1024, AdamW, WSD schedule, bf16. Fixed tokens per step,
chosen so the smallest runs still get at least 1,000 steps. Everything else is
an ablation.

- LR sweep (5 points) at 3 model sizes; fit LR\*(N) as a power law so Phase 2
  runs use a near-optimal LR at every size.
- 3 seeds at 2 model sizes at 1e16 FLOPs → noise floor.
- Check that a WSD cooldown branched off a long run matches a dedicated run of
  the same length within the noise floor (Hägele et al. 2024 report it does).
  Phase 2 depends on this.

**Done when:** noise floor, LR\*(N), and the WSD-branch check are written into
this file.

### Phase 2: baseline frontier (IsoFLOP sweep)

- Budgets: 1e15, 3e15, 1e16, 3e16, 1e17, 3e17, 1e18 FLOPs. Three decades
  pin the exponents far better than two.
- Per budget, 5–7 model sizes from about 1/4× to 4× the D = 20N size. At 1e18
  the largest sizes may not fit in memory; go up to the cap, keeping at least
  two sizes above the minimum.
- One long WSD run per model size, with a cooldown branch (~20% of steps) at
  each budget's token count. One run gives points on several IsoFLOP curves.
- Fits, each with bootstrap confidence intervals:
  - IsoFLOP minima → N\*(C) ∝ C^a, D\*(C) ∝ C^b (Chinchilla approach 2).
  - Parametric L(N, D) = E + A/N^α + B/D^β with a Huber loss (approach 3).
  - Each with N = non-embedding params and with N = non-embedding + LM head.
    At this scale the choice changes the fitted ratio (Pearce & Song 2024).
- Output: frontier plot (bpb vs FLOPs), tokens-per-param ratio against ~20,
  and the frontier curve L(C) that Phase 3 scores ECM against.

**Done when:** approaches 2 and 3 agree on the exponents within their
intervals, and leaving one budget out predicts its best loss within the noise
floor. Rough cost: 80–120 GPU-hours, mostly the 1e18 budget (one
compute-optimal run there takes ~14 hours).

### Phase 3: ablations

Per ablation: screen variants at 1e16 FLOPs with 3 LRs each, at the
compute-optimal size from Phase 2. If the ECM clears the noise floor, confirm at
1e17. Rough cost: ~1.5 GPU-hours to screen, ~4 more to confirm.

Backlog, ordered by expected impact at this scale. Reorder freely.

| # | Ablation | Variants | Why it may matter here |
|---|---|---|---|
| 1 | Vocab size | 4K, 8K, 16K, 32K | LM head is a large share of params and FLOPs at d ≤ 512 |
| 2 | Optimizer | AdamW vs Muon on hidden matrices | Large speedups on small GPTs (modded-nanogpt) |
| 3 | Parameterization | standard vs μP | If LR transfers across width, LR sweeps get cheaper; try early |
| 4 | LR schedule | WSD cooldown length and shape; cosine | Cheap, interacts with everything |
| 5 | Batch size | 3–4 sizes | Small models have small critical batch sizes |
| 6 | Depth vs width | aspect ratios at fixed N | Shape matters more at small N |
| 7 | Context length | 256, 512, 1024, 2048 | Attention FLOPs vs long-range signal |
| 8 | Document masking | on, off | Stops attention across document boundaries |
| 9 | QK-norm | on, off | Stability, may allow higher LR |
| 10 | Sliding-window attention | window size, full-attention every N layers | Cuts attention FLOPs |
| 11 | Tied vs untied embeddings | | Params vs quality at small N |
| 12 | MLP | SwiGLU ratio, ReLU² | |
| 13 | Init and stability | output-layer scaling, z-loss | |
| 14 | GQA | 1, 2, 4 KV heads per group | Mainly an inference saving; check training cost |
| 15 | Data | FineWeb-Edu vs its looser score-2 version, or ClimbMix | Data often beats architecture. Score every variant on one shared held-out set; bpb on different val sets isn't comparable |
| 16 | Digit splitting | on, off | Compression vs arithmetic |

**Done when:** every row has an ECM with a confidence interval at 1e16, and
the winners have one at 1e17.

### Phase 4: stack and re-measure

- Combine the confirmed winners. Test pairs likely to interact (Muon × LR
  schedule, vocab × width).
- Rerun the Phase 2 sweep with the new baseline. Report the frontier shift as
  ECM across budgets, and the new tokens-per-param ratio. Rough cost: same as
  Phase 2.

**Done when:** the stacked ECM is compared to the product of the individual
ECMs. A large gap either way means changes interact; find which.

## Target repo layout

```
PLAN.md
requirements.txt
mini_gpt/   tokenizer.py  data.py  model.py  flops.py  train.py  scaling.py
configs/    one file per baseline and sweep
results/    runs.jsonl (append-only, one row per run)
tests/
```

Add a file only when a phase needs it.
