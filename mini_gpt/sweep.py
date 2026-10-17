"""The Phase 2 IsoFLOP sweep: the model-size ladder, each budget's sizes, and each run's LR and micro-batch."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mini_gpt.flops import flops_per_token
from mini_gpt.model import GPTConfig
from mini_gpt.train import load_config

BASELINE = Path(__file__).resolve().parent.parent / "configs" / "baseline.json"
# (d_model, n_layers), 1.17-1.51x apart in non-embedding params from 0.40M to the 287M memory cap.
# d_model / n_layers = 64 from 6M up; below that d_model's multiples of 64 are too coarse, so 43-96.
LADDER = [
    (128, 2), (128, 3), (192, 2), (192, 3), (192, 4), (256, 3), (256, 4), (256, 5), (320, 4), (320, 5),
    (384, 5), (384, 6), (448, 6), (448, 7), (512, 8), (576, 9), (640, 10), (704, 11), (768, 12), (832, 13),
    (896, 14), (960, 15), (1024, 16), (1088, 17), (1152, 18),
]
BUDGETS = [1e15, 3e15, 1e16, 3e16, 1e17, 3e17, 1e18]
# Each budget's sizes relative to its N*: dense around the minimum, wide at the wings.
SPREAD = (1 / 4, 1 / 2, 2**-0.5, 1, 2**0.5, 2, 4)
TOKENS_PER_PARAM = 20  # N* is the size with D = 20N, the centre each budget's sizes are placed around
# Phase 1: LR* = 63.35 N^-0.674 at D = 20N (three sizes). A run past D = 20N takes LR* (D / 20N)^-0.25,
# the 3.1M horizon sweep; a shorter run keeps LR*, which held at 1/4x.
LR_C, LR_K, LR_HORIZON = 63.35, -0.674, -0.25
# The largest micro-batch under 7.3 GB reserved, from the memory probes; every smaller shape fits all 8.
MICRO_BATCH = {(960, 15): 4, (1024, 16): 4, (1088, 17): 2, (1152, 18): 1}


def shape(d_model: int, n_layers: int) -> GPTConfig:
    return load_config(BASELINE, [f"model.d_model={d_model}", f"model.n_layers={n_layers}"])[0]


def n_star(budget: float) -> float:
    """The non-embedding size N with flops_per_token(N) * 20N = budget, log-interpolated along the ladder."""
    points = []
    for d_model, n_layers in LADDER:
        cfg = shape(d_model, n_layers)
        n = cfg.param_counts()["non_embedding"]
        points.append((math.log(n), math.log(flops_per_token(cfg) * TOKENS_PER_PARAM * n)))
    for (n0, c0), (n1, c1) in zip(points, points[1:]):
        if c0 <= math.log(budget) <= c1:
            return math.exp(n0 + (math.log(budget) - c0) / (c1 - c0) * (n1 - n0))
    raise ValueError(f"budget {budget:g} puts N* off the ladder")


def lr_for(non_embedding: int, tokens: float) -> float:
    """Phase 1's LR*(N), lowered for runs longer than D = 20N; 3 significant figures."""
    lr = LR_C * non_embedding**LR_K * max(1.0, tokens / (TOKENS_PER_PARAM * non_embedding)) ** LR_HORIZON
    return float(f"{lr:.3g}")


def runs(budget: float) -> list[dict]:
    """The ladder sizes nearest N* times each SPREAD factor, smallest first; the ladder's top caps the wing."""
    target = n_star(budget)
    sizes = []
    for f in SPREAD:
        best = min(LADDER, key=lambda s: abs(math.log(shape(*s).param_counts()["non_embedding"] / (f * target))))
        if best not in sizes:
            sizes.append(best)
    out = []
    for d_model, n_layers in sorted(sizes):
        cfg = shape(d_model, n_layers)
        n = cfg.param_counts()["non_embedding"]
        tokens = budget / flops_per_token(cfg)
        out.append({"d_model": d_model, "n_layers": n_layers, "flops": budget, "non_embedding": n,
                    "tokens": tokens, "horizon": tokens / (TOKENS_PER_PARAM * n), "lr": lr_for(n, tokens),
                    "micro_batch": MICRO_BATCH.get((d_model, n_layers), 8)})
    return out


def planned(rows: list[dict]) -> list[dict]:
    """The results rows of planned runs, seeds averaged into one row each: what the IsoFLOP fits read.

    Runs of a planned size at another LR, such as an LR check's outer points, are left out.
    """
    plan = {(r["flops"], r["d_model"], r["n_layers"], r["lr"], r["micro_batch"]) for b in BUDGETS for r in runs(b)}
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        m, t = row["config"]["model"], row["config"]["train"]
        key = (t["flops"], m["d_model"], m["n_layers"], t["lr"], t["micro_batch"])
        if key in plan:
            groups.setdefault(key, []).append(row)
    return [{**g[0], "val_bpb": sum(r["val_bpb"] for r in g) / len(g), "seeds": len(g)} for g in groups.values()]


def command(run: dict, lr: float | None = None) -> str:
    """The configs/phase2.sh line for a run, optionally at another LR."""
    return (f"train --set model.d_model={run['d_model']} model.n_layers={run['n_layers']} "
            f"train.flops={run['flops']:g} train.lr={lr or run['lr']:g} train.micro_batch={run['micro_batch']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print each budget's IsoFLOP runs as configs/phase2.sh lines.")
    ap.add_argument("budgets", nargs="*", type=float, default=BUDGETS)
    args = ap.parse_args(argv)
    for budget in args.budgets:
        planned = runs(budget)
        print(f"# {budget:g} FLOPs: N* = {n_star(budget) / 1e6:.1f}M, {len(planned)} sizes")
        for r in planned:
            print(f"{command(r)}  # {r['non_embedding'] / 1e6:.1f}M, {r['tokens'] / 1e6:,.0f}M tokens "
                  f"({r['horizon']:.2g}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
