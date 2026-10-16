"""Train one config to a FLOP budget with a WSD schedule, score val bpb, append the results row."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from mini_gpt.data import load_val
from mini_gpt.flops import flops_per_token
from mini_gpt.model import GPT, IGNORE_INDEX, GPTConfig
from mini_gpt.tokenizer import Tokenizer

CKPT_SECONDS = 600  # a laptop can sleep at any moment; lose at most this much


@dataclass
class TrainConfig:
    flops: float         # training budget, as flops.py counts it
    batch_size: int      # sequences per optimizer step
    micro_batch: int     # sequences per forward; divides batch_size
    lr: float            # AdamW peak
    warmup_steps: int
    decay_frac: float    # WSD: linear decay to zero over this last fraction of the steps
    weight_decay: float  # on matrices only
    beta1: float
    beta2: float
    grad_clip: float
    seed: int            # model init; train.bin fixes the data order


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> tuple[GPTConfig, TrainConfig]:
    """Read a {"model": ..., "train": ...} JSON file, then apply section.key=value overrides."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for item in overrides:
        key, value = item.split("=", 1)
        section, field = key.split(".")
        raw[section][field] = json.loads(value)
    return GPTConfig(**raw["model"]), TrainConfig(**raw["train"])


def run_name(model_cfg: GPTConfig, cfg: TrainConfig) -> str:
    """The main knobs, readable, plus a hash of the whole config so two configs never share a dir."""
    config = {"model": asdict(model_cfg), "train": asdict(cfg)}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
    return f"d{model_cfg.d_model}-L{model_cfg.n_layers}-{cfg.flops:.0e}-lr{cfg.lr:g}-s{cfg.seed}-{digest}"


def lr_at(step: int, steps: int, cfg: TrainConfig) -> float:
    """WSD: linear warmup, flat at the peak, then linear decay to zero over the last decay_frac."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    decay = max(1, round(cfg.decay_frac * steps))
    return cfg.lr * min(1.0, (steps - step) / decay)


def get_batch(
    tokens: np.ndarray, start: int, rows: int, context: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """`rows` consecutive windows from token `start`; targets are the inputs shifted by one."""
    chunk = torch.from_numpy(tokens[start : start + rows * context + 1].astype(np.int64)).to(device)
    return chunk[:-1].view(rows, context), chunk[1:].view(rows, context)


def autocast(device: torch.device):
    """bf16 autocast on CUDA; plain float32 on CPU, where the tests run."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def val_windows(
    docs: list[str], tok: Tokenizer, context: int, rows: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack the held-out docs into [n, context] inputs and targets, n a multiple of rows.

    The docs form one stream, <|bos|> before each, cut into windows, so every
    token is a target exactly once. <|bos|> and padding targets are
    IGNORE_INDEX: they carry no bytes. Also returns the docs' UTF-8 byte count.
    """
    stream: list[int] = []
    for ids in tok.encode_batch(docs):
        stream.append(tok.bos_id)
        stream.extend(ids)
    s = torch.tensor(stream)
    inputs, targets = s[:-1], s[1:].masked_fill(s[1:] == tok.bos_id, IGNORE_INDEX)
    n = rows * math.ceil(len(inputs) / (rows * context))
    pad = n * context - len(inputs)
    inputs = torch.cat([inputs, torch.zeros(pad, dtype=inputs.dtype)])
    targets = torch.cat([targets, torch.full((pad,), IGNORE_INDEX)])
    return inputs.view(n, context), targets.view(n, context), sum(len(d.encode("utf-8")) for d in docs)


@torch.no_grad()
def val_bpb(
    model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, n_bytes: int, rows: int,
    device: torch.device,
) -> float:
    """Bits per byte on the held-out docs: total NLL in nats / (ln 2 * UTF-8 bytes)."""
    model.eval()
    nll = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, len(inputs), rows):
        with autocast(device):
            batch_nll = model(inputs[i : i + rows].to(device), targets[i : i + rows].to(device))
        nll += batch_nll.sum(dtype=torch.float64)
    model.train()
    return nll.item() / (math.log(2) * n_bytes)


def git_commit() -> str:
    """HEAD's hash, suffixed -dirty when mini_gpt/ has uncommitted changes."""
    repo = Path(__file__).resolve().parent.parent

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", "mini_gpt") else "")


def train(
    model_cfg: GPTConfig,
    cfg: TrainConfig,
    *,
    data_dir: str | Path,
    val_path: str | Path,
    out_dir: str | Path,
    results_path: str | Path,
    device: str = "cuda",
    compile: bool = True,
    stop_at: int | None = None,
) -> dict | None:
    """Train to cfg.flops; return the results row it appends, or None if it stopped early.

    A checkpoint in out_dir is resumed. The data position is a function of the
    step alone, so a resumed run reads exactly the tokens an uninterrupted one would.
    """
    out, data_dir, dev = Path(out_dir), Path(data_dir), torch.device(device)
    if (out / "model.pt").exists():
        print(f"{out} already finished")
        return None
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    tok = Tokenizer.load(data_dir / "tokenizer.json")
    assert tok.fingerprint() == meta["tokenizer_sha256"], "train.bin was packed by another tokenizer"
    assert model_cfg.vocab_size == tok.vocab_size, f"tokenizer has {tok.vocab_size} tokens"
    tokens = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")

    ctx = model_cfg.context
    fpt = flops_per_token(model_cfg)
    step_tokens = cfg.batch_size * ctx
    steps = max(1, round(cfg.flops / (fpt * step_tokens)))
    accum = cfg.batch_size // cfg.micro_batch
    assert accum * cfg.micro_batch == cfg.batch_size, "micro_batch must divide batch_size"
    if steps * step_tokens + 1 > len(tokens):
        raise ValueError(f"needs {steps * step_tokens:,} tokens, train.bin has {len(tokens):,}; "
                         "a run never repeats data")

    torch.manual_seed(cfg.seed)
    model = GPT(model_cfg).to(dev)
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": cfg.weight_decay},
            {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=dev.type == "cuda",
    )
    config = {"model": asdict(model_cfg), "train": asdict(cfg)}
    # loss_sum / loss_steps: the open logging interval; timed_*: steady-state throughput.
    state = {"step": 0, "curve": [], "loss_sum": 0.0, "loss_steps": 0, "timed_s": 0.0, "timed_tokens": 0}
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=dev)
        assert ckpt["config"] == config, f"{ckpt_path} was written by a different config"
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        state = ckpt["state"]
        print(f"resumed {out} at step {state['step']}")

    params = model_cfg.param_counts()
    print(f"{out.name}: {params['non_embedding']:,} non-embedding + {params['embedding']:,} embedding "
          f"params, {steps:,} steps x {step_tokens:,} tokens = {steps * step_tokens * fpt:.3e} FLOPs")
    fwd = torch.compile(model) if compile else model
    val_x, val_y, val_bytes = val_windows(load_val(val_path), tok, ctx, cfg.micro_batch)
    log_every = max(1, steps // 100)
    eval_every = 10 * log_every  # every eval step is also a log step
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    def save_ckpt() -> None:
        state["step"], state["loss_sum"] = step, loss_sum.item()
        out.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"config": config, "model": model.state_dict(), "opt": opt.state_dict(), "state": state},
                   tmp)
        tmp.replace(ckpt_path)

    step = state["step"]
    end = steps if stop_at is None else min(stop_at, steps)
    loss_sum = torch.tensor(state["loss_sum"], device=dev)
    last_ckpt = time.time()
    # The first interval after a (re)start pays for compilation, so it is not timed.
    mark, since_mark, warm = time.perf_counter(), 0, False
    while step < end:
        lr = lr_at(step, steps, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        for m in range(accum):
            x, y = get_batch(tokens, (step * cfg.batch_size + m * cfg.micro_batch) * ctx, cfg.micro_batch,
                             ctx, dev)
            with autocast(dev):
                loss = fwd(x, y).mean()
            (loss / accum).backward()
            loss_sum += loss.detach() / accum
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        state["loss_steps"] += 1
        since_mark += 1

        if step % log_every == 0 or step == steps:
            point = {"step": step, "tokens": step * step_tokens, "flops": step * step_tokens * fpt, "lr": lr,
                     "train_loss": loss_sum.item() / state["loss_steps"]}
            elapsed = time.perf_counter() - mark
            if warm:
                point["tflops"] = since_mark * step_tokens * fpt / elapsed / 1e12
                state["timed_s"] += elapsed
                state["timed_tokens"] += since_mark * step_tokens
            msg = f"step {step:>7}/{steps} loss {point['train_loss']:.4f} lr {lr:.2e}"
            if "tflops" in point:
                msg += f" {point['tflops']:5.1f} TFLOP/s"
            if step % eval_every == 0 or step == steps:
                point["val_bpb"] = val_bpb(fwd, val_x, val_y, val_bytes, cfg.micro_batch, dev)
                msg += f" val bpb {point['val_bpb']:.4f}"
            print(msg, flush=True)
            state["curve"].append(point)
            loss_sum.zero_()
            state["loss_steps"] = 0
            mark, since_mark, warm = time.perf_counter(), 0, True

        if time.time() - last_ckpt > CKPT_SECONDS and step < steps:
            save_ckpt()
            last_ckpt = time.time()
            mark, since_mark = time.perf_counter(), 0

    if step < steps:
        save_ckpt()
        print(f"stopped at step {step}; rerun to resume")
        return None

    row = {
        "run": out.name,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "config": config,
        "data": {"dir": str(data_dir), "tokenizer_sha256": meta["tokenizer_sha256"], "val": str(val_path)},
        "flops_per_token": fpt,
        "flops": steps * step_tokens * fpt,
        "tokens": steps * step_tokens,
        "steps": steps,
        "params": params,
        "val_bpb": state["curve"][-1]["val_bpb"],
        "tflops": state["timed_tokens"] * fpt / state["timed_s"] / 1e12 if state["timed_s"] else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else None,
        "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "curve": state["curve"],
    }
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config, "model": model.state_dict()}, out / "model.pt")
    ckpt_path.unlink(missing_ok=True)
    print(f"done: val bpb {row['val_bpb']:.4f} -> {results_path}")
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Train one config to its FLOP budget and append its results row. "
        "Rerunning the same command resumes from the run's checkpoint."
    )
    ap.add_argument("config", help="JSON file with 'model' and 'train' sections")
    ap.add_argument("--set", nargs="+", default=[], metavar="SECTION.KEY=VALUE",
                    help="config overrides with JSON values, e.g. train.lr=1e-3 model.d_model=192")
    ap.add_argument("--data", default="data/v16k", help="dir with tokenizer.json, train.bin, meta.json")
    ap.add_argument("--val", default="data/val.jsonl", help="held-out docs as raw text")
    ap.add_argument("--out", default=None, help="run dir (default out/<run name>)")
    ap.add_argument("--results", default="results/runs.jsonl")
    ap.add_argument("--stop-at", type=int, default=None, help="checkpoint and exit at this step")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    model_cfg, cfg = load_config(args.config, args.set)
    out = Path(args.out) if args.out else Path("out") / run_name(model_cfg, cfg)
    train(model_cfg, cfg, data_dir=args.data, val_path=args.val, out_dir=out, results_path=args.results,
          device=args.device, compile=not args.no_compile, stop_at=args.stop_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
