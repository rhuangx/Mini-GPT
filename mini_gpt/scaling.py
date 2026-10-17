"""Plots and fits over results/runs.jsonl: loss curves, LR sweeps with LR*(N), the seed noise floor."""

from __future__ import annotations

import argparse
import fnmatch
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure

# Categorical colors, assigned in this fixed order: adjacent slots stay apart under colorblindness.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STYLE = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11, "axes.titlelocation": "left",
    "lines.linewidth": 1.0, "lines.markersize": 5, "lines.markeredgecolor": "#fcfcfb",
    "lines.markeredgewidth": 0.8, "legend.frameon": False, "font.size": 9, "savefig.dpi": 150,
}


def load_runs(path: str | Path, pattern: str = "*") -> list[dict]:
    """Rows whose run name matches the glob; a run rerun under the same name keeps its last row."""
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    latest = {r["run"]: r for r in rows}
    return [r for name, r in latest.items() if fnmatch.fnmatch(name, pattern)]


def flat(config: dict, drop: tuple[str, ...] = ()) -> dict:
    """{"model": {"d_model": 256}} -> {"model.d_model": 256}, without the keys in drop."""
    return {f"{s}.{k}": v for s, section in config.items() for k, v in section.items() if k not in drop}


def names(configs: list[dict]) -> list[str]:
    """Name each flat config by the fields that differ across the set."""
    fields = sorted({f for c in configs for f in c})
    differ = [f for f in fields if len({json.dumps(c.get(f)) for c in configs}) > 1]

    def fmt(v: object) -> str:
        return f"{v:g}" if isinstance(v, float) else str(v)

    return [", ".join(f"{f.split('.', 1)[1]}={fmt(c.get(f))}" for f in differ) or "baseline" for c in configs]


def variants(rows: list[dict], drop: tuple[str, ...]) -> dict[str, list[dict]]:
    """Rows grouped by config without the `drop` keys, smallest model first."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[json.dumps(flat(r["config"], drop), sort_keys=True)].append(r)
    return dict(sorted(groups.items(), key=lambda g: (g[1][0]["params"]["non_embedding"], g[0])))


def lr_optimum(lrs: list[float], bpbs: list[float], tol: float) -> tuple[float, float] | None:
    """(LR*, best bpb): LR* is the geometric middle of the LRs that finish within tol of the best bpb.

    LR curves are flat-bottomed with a cliff on the high side, which throws a parabola fit far off.
    None when that near-best band reaches an end of the sweep: the optimum is not bracketed.
    """
    order = np.argsort(lrs)
    x, y = np.asarray(lrs, dtype=float)[order], np.asarray(bpbs, dtype=float)[order]
    y = np.where(np.isfinite(y), y, np.inf)
    near = np.flatnonzero(y <= y.min() + tol)
    if near[0] == 0 or near[-1] == len(x) - 1:
        return None
    return float(np.sqrt(x[near[0]] * x[near[-1]])), float(y.min())


def power_law(n: list[float], y: list[float]) -> tuple[float, float]:
    """(c, k) for y = c * n^k, least squares in log-log."""
    k, log_c = np.polyfit(np.log(n), np.log(y), 1)
    return float(np.exp(log_c)), float(k)


def plot_curves(rows: list[dict], out: Path) -> None:
    """Train loss and val bpb against FLOPs, one line per run."""
    if len(rows) > len(SERIES):
        raise SystemExit(f"{len(rows)} runs match; narrow --runs to at most {len(SERIES)}")
    rows = sorted(rows, key=lambda r: (r["params"]["non_embedding"], r["run"]))
    labels = names([flat(r["config"]) for r in rows])
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(10, 4), layout="constrained")
        ax_loss, ax_bpb = fig.subplots(1, 2)
        for r, label, color in zip(rows, labels, SERIES, strict=False):
            ax_loss.plot([p["flops"] for p in r["curve"]], [p["train_loss"] for p in r["curve"]], color=color,
                         label=label)
            val = [p for p in r["curve"] if "val_bpb" in p]
            ax_bpb.plot([p["flops"] for p in val], [p["val_bpb"] for p in val], color=color, marker="o")
        for ax, title in ((ax_loss, "Train loss (nats per token)"), (ax_bpb, "Validation bits per byte")):
            ax.set(xscale="log", xlabel="Training FLOPs", title=title)
        if len(rows) > 1:
            fig.legend(loc="outside lower center", ncols=min(len(rows), 4))
        fig.savefig(out)


def lr_sweep(rows: list[dict], out: Path, tol: float) -> dict:
    """Final val bpb against LR per variant (seeds averaged), and LR* against non-embedding params."""
    groups = variants(rows, drop=("lr", "seed"))
    labels = names([json.loads(k) for k in groups])
    if len(groups) > len(SERIES):
        raise SystemExit(f"{len(groups)} variants match; narrow --runs to at most {len(SERIES)}")
    fits = []
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(10, 4), layout="constrained")
        ax_lr, ax_n = fig.subplots(1, 2)
        for (_, group), label, color in zip(groups.items(), labels, SERIES, strict=False):
            by_lr: dict[float, list[float]] = defaultdict(list)
            for r in group:
                by_lr[r["config"]["train"]["lr"]].append(r["val_bpb"])
            lrs = sorted(by_lr)
            bpbs = [float(np.mean(by_lr[lr])) for lr in lrs]
            ax_lr.plot(lrs, bpbs, color=color, marker="o", label=label)
            best = lr_optimum(lrs, bpbs, tol)
            fits.append({"variant": label, "non_embedding": group[0]["params"]["non_embedding"],
                         "lrs": lrs, "bpbs": bpbs, "lr_star": best and best[0], "bpb_star": best and best[1]})
            if best:
                ax_lr.plot(*best, color=color, marker="D", markersize=7)
        ax_lr.set(xscale="log", xlabel="Peak learning rate", title="Final validation bpb (diamond: LR*)")
        ticks = sorted({lr for f in fits for lr in f["lrs"]})
        ax_lr.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax_lr.set_xticks([], minor=True)
        if len(groups) > 1:
            ax_lr.legend()

        bracketed = [f for f in fits if f["lr_star"]]
        law = None
        if len({f["non_embedding"] for f in bracketed}) >= 2:
            n, lr = [f["non_embedding"] for f in bracketed], [f["lr_star"] for f in bracketed]
            law = power_law(n, lr)
            grid = np.geomspace(min(n) / 2, max(n) * 2)
            ax_n.plot(grid, law[0] * grid ** law[1], color="#898781", linewidth=0.8)
            for f in bracketed:
                ax_n.plot(f["non_embedding"], f["lr_star"], color=SERIES[fits.index(f)], marker="D", markersize=7)
            ax_n.annotate(f"LR* = {law[0]:.3g} · N^{law[1]:.3f}", xy=(0.04, 0.06), xycoords="axes fraction",
                          color="#52514e")
        else:
            ax_n.text(0.5, 0.5, "needs bracketed LR* at 2+ sizes", ha="center", transform=ax_n.transAxes,
                      color="#898781")
        ax_n.set(xscale="log", yscale="log", xlabel="Non-embedding parameters", title="LR* against model size")
        sizes = sorted({f["non_embedding"] for f in fits})
        ax_n.set_xticks(sizes, [f"{n / 1e6:.1f}M" for n in sizes])
        ax_n.set_xticks([], minor=True)
        fig.savefig(out)
    return {"variants": fits, "power_law": law}


def seed_spread(rows: list[dict]) -> list[dict]:
    """Final val bpb across seeds for every config run with 2+ seeds."""
    spread = []
    groups = variants(rows, drop=("seed",))
    for (_, group), label in zip(groups.items(), names([json.loads(k) for k in groups]), strict=True):
        if len(group) >= 2:
            bpb = np.array([r["val_bpb"] for r in group])
            spread.append({"variant": label, "seeds": len(bpb), "mean": float(bpb.mean()),
                           "std": float(bpb.std(ddof=1)), "range": float(bpb.max() - bpb.min())})
    return spread


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot and fit the rows of results/runs.jsonl.")
    ap.add_argument("command", choices=["curves", "lr-sweep", "seeds"],
                    help="curves: loss vs FLOPs per run; lr-sweep: bpb vs LR and LR*(N); seeds: noise floor")
    ap.add_argument("--runs", default="*", help="glob over run names, e.g. 'd256-L4-*'")
    ap.add_argument("--results", default="results/runs.jsonl")
    ap.add_argument("--out", default=None, help="PNG path (default out/plots/<command>.png)")
    ap.add_argument("--tol", type=float, default=0.0034,
                    help="lr-sweep: LRs within this many bpb of the best count as near-best (2x the noise floor)")
    args = ap.parse_args(argv)

    rows = load_runs(args.results, args.runs)
    if not rows:
        raise SystemExit(f"no runs in {args.results} match {args.runs!r}")
    out = Path(args.out or f"out/plots/{args.command}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "curves":
        plot_curves(rows, out)
    elif args.command == "lr-sweep":
        result = lr_sweep(rows, out, args.tol)
        for f in result["variants"]:
            star = f"LR* {f['lr_star']:.3g} (best bpb {f['bpb_star']:.4f})" if f["lr_star"] else "LR* not bracketed"
            print(f"{f['variant']}: N={f['non_embedding']:,}  {star}")
            print("    " + "  ".join(f"{lr:g}:{b:.4f}" for lr, b in zip(f["lrs"], f["bpbs"], strict=True)))
        if result["power_law"]:
            c, k = result["power_law"]
            print(f"LR*(N) = {c:.4g} * N^{k:.4f}")
    else:
        spread = seed_spread(rows)
        if not spread:
            raise SystemExit("no config has 2+ seeds")
        for s in spread:
            print(f"{s['variant']}: {s['seeds']} seeds, bpb {s['mean']:.4f} ± {s['std']:.4f} (range {s['range']:.4f})")
        floor = max(s["std"] for s in spread)
        print(f"noise floor (largest seed std): {floor:.4f} bpb; differences under {2 * floor:.4f} are not results")
        return 0
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
