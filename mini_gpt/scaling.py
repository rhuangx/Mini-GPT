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


HUBER_DELTA = 1e-3  # Chinchilla's, on log bpb
# Chinchilla's L-BFGS starting points for (log A, alpha, log B, beta, log E).
INIT_GRID = [(a, alpha, b, beta, e) for a in range(0, 30, 5) for alpha in (0, 0.5, 1, 1.5, 2)
             for b in range(0, 30, 5) for beta in (0, 0.5, 1, 1.5, 2) for e in (-1, -0.5, 0, 0.5, 1)]


def n_of(row: dict, n_kind: str) -> int:
    """Model size as the fits count it: non-embedding params, or "with_head" adding the tied LM head."""
    p = row["params"]
    return p["non_embedding"] + (p["embedding"] if n_kind == "with_head" else 0)


def by_budget(rows: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["config"]["train"]["flops"]].append(r)
    return dict(sorted(groups.items()))


def isoflop_minima(rows: list[dict], n_kind: str) -> list[dict]:
    """Chinchilla approach 2: per budget, a parabola of bpb in log N; its vertex is the budget's optimum.

    A budget is left out when its parabola opens downward or its vertex falls outside the sizes it ran.
    """
    minima = []
    for budget, group in by_budget(rows).items():
        group = sorted(group, key=lambda r: n_of(r, n_kind))
        x = np.log([n_of(r, n_kind) for r in group])
        if len(np.unique(x)) < 3:
            continue
        a, b, c = np.polyfit(x, [r["val_bpb"] for r in group], 2)
        v = -b / (2 * a)
        if a <= 0 or not x[0] < v < x[-1]:
            continue
        minima.append({"flops": budget, "n_opt": float(np.exp(v)), "bpb_opt": float(a * v * v + b * v + c),
                       "d_opt": float(np.exp(np.interp(v, x, np.log([r["tokens"] for r in group])))),
                       "parabola": (a, b, c), "runs": group})
    return minima


def _huber_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Huber loss of log bpb = LSE(log A - alpha x, log B - beta y, log E) against target, and its gradient."""
    a, alpha, b, beta, e = theta
    terms = np.stack([a - alpha * x, b - beta * y, np.full_like(x, e)])
    top = terms.max(axis=0)
    w = np.exp(terms - top)
    p = w / w.sum(axis=0)
    r = top + np.log(w.sum(axis=0)) - target
    loss = np.where(np.abs(r) <= HUBER_DELTA, 0.5 * r**2, HUBER_DELTA * (np.abs(r) - 0.5 * HUBER_DELTA))
    dr = np.clip(r, -HUBER_DELTA, HUBER_DELTA)
    grad = [dr @ p[0], -(dr * x) @ p[0], dr @ p[1], -(dr * y) @ p[1], dr @ p[2]]
    return float(loss.sum()), np.array(grad)


def parametric_fit(rows: list[dict], n_kind: str, inits: list | None = None) -> dict:
    """Chinchilla approach 3: bpb = E + A / N^alpha + B / D^beta, the best of L-BFGS runs from each start."""
    from scipy.optimize import minimize

    x = np.log([n_of(r, n_kind) for r in rows])
    y = np.log([r["tokens"] for r in rows])
    target = np.log([r["val_bpb"] for r in rows])
    # With delta 1e-3 the loss and its gradient are tiny, and L-BFGS-B's default tolerances stop far from the
    # optimum (Besiroglu et al. 2024 found the same in Chinchilla's own fit).
    tight = {"ftol": 1e-15, "gtol": 1e-12, "maxiter": 20_000}
    best = min((minimize(_huber_loss, np.asarray(x0, dtype=float), args=(x, y, target), jac=True,
                         method="L-BFGS-B", options=tight) for x0 in (inits or INIT_GRID)), key=lambda res: res.fun)
    a, alpha, b, beta, e = best.x
    return {"A": float(np.exp(a)), "alpha": float(alpha), "B": float(np.exp(b)), "beta": float(beta),
            "E": float(np.exp(e)), "theta": best.x}


def predict(fit: dict, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    return fit["E"] + fit["A"] / n ** fit["alpha"] + fit["B"] / d ** fit["beta"]


def fit_frontier(fit: dict, rows: list[dict], budgets: list[float], n_kind: str) -> list[dict]:
    """Each budget's optimum under the parametric fit: min over N of L(N, C / flops_per_token(N)).

    flops_per_token(N) is interpolated between the sizes run; at_edge marks an optimum outside them.
    """
    sizes = sorted({(n_of(r, n_kind), r["flops_per_token"]) for r in rows})
    log_n = np.log([n for n, _ in sizes])
    grid = np.linspace(log_n[0], log_n[-1], 4000)
    fpt = np.exp(np.interp(grid, log_n, np.log([f for _, f in sizes])))
    out = []
    for budget in budgets:
        bpb = predict(fit, np.exp(grid), budget / fpt)
        i = int(np.argmin(bpb))
        out.append({"flops": budget, "n_opt": float(np.exp(grid[i])), "d_opt": float(budget / fpt[i]),
                    "bpb_opt": float(bpb[i]), "at_edge": i in (0, len(grid) - 1)})
    return out


def exponents(points: list[dict]) -> tuple[float, float]:
    """(a, b) in N_opt ∝ C^a and D_opt ∝ C^b."""
    c = [p["flops"] for p in points]
    return power_law(c, [p["n_opt"] for p in points])[1], power_law(c, [p["d_opt"] for p in points])[1]


def isoflop(rows: list[dict], n_kind: str, reps: int = 200, seed: int = 0, inits: list | None = None) -> dict:
    """Approaches 2 and 3, 95% bootstrap intervals (runs resampled within budgets), leave-one-budget-out."""
    minima = isoflop_minima(rows, n_kind)
    fit = parametric_fit(rows, n_kind, inits)
    budgets = list(by_budget(rows))
    frontier = fit_frontier(fit, rows, budgets, n_kind)
    rng = np.random.default_rng(seed)
    boot: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        sample = [g[i] for g in by_budget(rows).values() for i in rng.integers(0, len(g), len(g))]
        m = isoflop_minima(sample, n_kind)
        if len(m) >= 2:
            a2, b2 = exponents(m)
            boot["a2"].append(a2)
            boot["b2"].append(b2)
        f = parametric_fit(sample, n_kind, [fit["theta"]])
        boot["alpha"].append(f["alpha"])
        boot["beta"].append(f["beta"])
        if len(budgets) >= 2:
            a3, b3 = exponents(fit_frontier(f, sample, budgets, n_kind))
            boot["a3"].append(a3)
            boot["b3"].append(b3)
    held_out = []
    for m in minima:
        rest = [r for r in rows if r["config"]["train"]["flops"] != m["flops"]]
        f = parametric_fit(rest, n_kind, [fit["theta"]])
        pred = fit_frontier(f, rows, [m["flops"]], n_kind)[0]["bpb_opt"]
        held_out.append({"flops": m["flops"], "predicted": pred, "observed": m["bpb_opt"]})
    a2, b2 = exponents(minima) if len(minima) >= 2 else (None, None)
    a3, b3 = exponents(frontier) if len(budgets) >= 2 else (None, None)
    return {"minima": minima, "fit": fit, "frontier": frontier, "held_out": held_out,
            "point": {"a2": a2, "b2": b2, "alpha": fit["alpha"], "beta": fit["beta"], "a3": a3, "b3": b3},
            "ci": {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in boot.items()}}


def plot_isoflop(result: dict, rows: list[dict], n_kind: str, out: Path) -> None:
    """IsoFLOP curves with their parabolas, N_opt and D_opt against C, and the bpb frontier."""
    groups = by_budget(rows)
    if len(groups) > len(SERIES):
        raise SystemExit(f"{len(groups)} budgets; plot at most {len(SERIES)}")
    vertex = {m["flops"]: m for m in result["minima"]}
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=(13, 4), layout="constrained")
        ax_iso, ax_opt, ax_front = fig.subplots(1, 3)
        for (budget, group), color in zip(groups.items(), SERIES, strict=False):
            n = np.array([n_of(r, n_kind) for r in group])
            ax_iso.plot(n, [r["val_bpb"] for r in group], color=color, marker="o", linestyle="none",
                        label=f"{budget:.0e}")
            if budget in vertex:
                a, b, c = vertex[budget]["parabola"]
                x = np.linspace(np.log(n.min()), np.log(n.max()), 100)
                ax_iso.plot(np.exp(x), a * x * x + b * x + c, color=color)
                ax_iso.plot(vertex[budget]["n_opt"], vertex[budget]["bpb_opt"], color=color, marker="D", markersize=7)
        label = "Non-embedding params" if n_kind == "non_embedding" else "Params with LM head"
        ax_iso.set(xscale="log", xlabel=label, title="IsoFLOP curves (diamond: minimum)")
        ax_iso.legend(title="FLOPs", ncols=2)

        c = np.array([m["flops"] for m in result["minima"]])
        for key, color, name in (("n_opt", SERIES[0], "N_opt"), ("d_opt", SERIES[1], "D_opt")):
            ax_opt.plot(c, [m[key] for m in result["minima"]], color=color, marker="o", linestyle="none",
                        label=f"{name} ∝ C^{result['point']['a2' if key == 'n_opt' else 'b2']:.3f}")
            if len(c) >= 2:
                k, log_k0 = np.polyfit(np.log(c), np.log([m[key] for m in result["minima"]]), 1)
                ax_opt.plot(c, np.exp(log_k0) * c**k, color=color)
        ax_opt.set(xscale="log", yscale="log", xlabel="Training FLOPs", title="Approach 2 optimum: params, tokens")
        ax_opt.legend()

        budgets = list(groups)
        front = fit_frontier(result["fit"], rows, list(np.geomspace(budgets[0], budgets[-1], 60)), n_kind)
        ax_front.plot([p["flops"] for p in front], [p["bpb_opt"] for p in front], color=SERIES[0],
                      label="approach 3 fit")
        ax_front.plot(c, [m["bpb_opt"] for m in result["minima"]], color=SERIES[1], marker="o", linestyle="none",
                      label="approach 2 minima")
        ax_front.set(xscale="log", xlabel="Training FLOPs", title="Best val bpb at each budget")
        ax_front.legend()
        fig.savefig(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plot and fit the rows of results/runs.jsonl.")
    ap.add_argument("command", choices=["curves", "lr-sweep", "seeds", "isoflop"],
                    help="curves: loss vs FLOPs per run; lr-sweep: bpb vs LR and LR*(N); seeds: noise floor; "
                    "isoflop: Phase 2 fits over the planned runs")
    ap.add_argument("--runs", default="*", help="glob over run names, e.g. 'd256-L4-*'")
    ap.add_argument("--results", default="results/runs.jsonl")
    ap.add_argument("--out", default=None, help="PNG path (default out/plots/<command>.png)")
    ap.add_argument("--tol", type=float, default=0.0034,
                    help="lr-sweep: LRs within this many bpb of the best count as near-best (2x the noise floor)")
    ap.add_argument("--n", choices=["non_embedding", "with_head"], default="non_embedding",
                    help="isoflop: count model size without or with the tied LM head")
    ap.add_argument("--reps", type=int, default=200, help="isoflop: bootstrap resamples")
    args = ap.parse_args(argv)

    rows = load_runs(args.results, args.runs)
    if not rows:
        raise SystemExit(f"no runs in {args.results} match {args.runs!r}")
    out = Path(args.out or f"out/plots/{args.command}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "curves":
        plot_curves(rows, out)
    elif args.command == "isoflop":
        from mini_gpt import sweep

        rows = sweep.planned(rows)
        result = isoflop(rows, args.n, reps=args.reps)
        plot_isoflop(result, rows, args.n, out)
        print(f"approach 2 minima (N = {args.n}):")
        for m in result["minima"]:
            print(f"  {m['flops']:.0e}: N_opt {m['n_opt'] / 1e6:.2f}M  D_opt {m['d_opt'] / 1e6:,.0f}M  "
                  f"D/N {m['d_opt'] / m['n_opt']:.1f}  bpb {m['bpb_opt']:.4f}  ({len(m['runs'])} sizes)")
        pt, ci = result["point"], result["ci"]

        def est(key: str) -> str:
            lo, hi = ci.get(key, (float("nan"), float("nan")))
            return "–" if pt[key] is None else f"{pt[key]:.3f} [{lo:.3f}, {hi:.3f}]"

        f = result["fit"]
        print(f"approach 2: N_opt ∝ C^{est('a2')}  D_opt ∝ C^{est('b2')}")
        print(f"approach 3: bpb = {f['E']:.3f} + {f['A']:.3g} / N^{f['alpha']:.3f} + {f['B']:.3g} / D^{f['beta']:.3f}")
        print(f"            alpha {est('alpha')}  beta {est('beta')}  N_opt ∝ C^{est('a3')}  D_opt ∝ C^{est('b3')}")
        print("leave one budget out (approach 3 on the rest against the approach 2 minimum):")
        for h in result["held_out"]:
            print(f"  {h['flops']:.0e}: {h['predicted']:.4f} vs {h['observed']:.4f} "
                  f"({h['predicted'] - h['observed']:+.4f})")
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
