"""TOP-12 --- figures + table for the RL-vs-greedy alpha ablation.

Reads the JSON written by the `rl_alpha_ablation` Modal entrypoint
(modal_runs/train_static_holliday.py) and produces the writeup anchor:

  1. <stem>_gap.pdf   --- the RL-vs-greedy welfare gap vs induced-demand alpha.
                          LEFT panel: the two gap curves -- best-of-N - greedy
                          (headline, LC-N optimistic) and mean-of-N - greedy
                          (honest/expected, error bars = std across seeds), with
                          a zero line. RIGHT panel: the absolute cumulative
                          welfare of RL best, RL mean, greedy and random vs alpha
                          (context for the gap).
  2. <stem>_table.md / <stem>_table.csv --- per-alpha gap table: RL best,
     RL mean +/- std, RL greedy, greedy + random baselines, gap_best, gap_mean
     +/- std (across seeds), and a one-sample t-stat of the per-seed mean gap vs
     zero (the significance check the mean-of-N curve feeds).

The gap is RL minus the GREEDY baseline (the strong baseline). Each (alpha, seed)
row is one trained policy; we aggregate over seeds per alpha.

Run from the repo root::

    python -m learning.city_builder.plot_rl_alpha_ablation \
        --in results/rl_alpha_ablation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _by_alpha(results: list[dict]) -> dict[float, list[dict]]:
    g: dict[float, list[dict]] = defaultdict(list)
    for r in results:
        g[float(r["alpha"])].append(r)
    return g


def _t_stat(vals: np.ndarray) -> tuple[float, int]:
    """One-sample t-stat of `vals` vs 0 and the dof (n-1). Manual so we don't
    pull scipy; report alongside n so the reader can look up the critical t."""
    n = vals.size
    if n < 2:
        return float("nan"), max(n - 1, 0)
    sd = vals.std(ddof=1)
    if sd == 0:
        return float("inf") if vals.mean() > 0 else float("-inf"), n - 1
    return vals.mean() / (sd / math.sqrt(n)), n - 1


METRIC_SUFFIX = {"delta": "", "integrated": "_int"}
METRIC_LABEL = {"delta": "telescoped  C(s0)-C(sT)",
                "integrated": "integrated  -Σ C_t (M2-style)"}


def build_rows(payload: dict, metric: str = "integrated") -> list[dict]:
    """Aggregate per-alpha across seeds into one summary row each, for the chosen
    welfare metric ('integrated' = M2-style level sum, alpha-sensitive; 'delta' =
    telescoped endpoint reduction)."""
    sx = METRIC_SUFFIX[metric]

    def col(recs, name):
        # Fall back to the delta field if a row predates the _int metric.
        return np.array([r.get(name + sx, r[name]) for r in recs])

    g = _by_alpha(payload["results"])
    rows = []
    for a in sorted(g):
        recs = g[a]
        n_seeds = len(recs)
        rl_best = col(recs, "rl_best")
        rl_mean = col(recs, "rl_mean")
        rl_greedy = col(recs, "rl_greedy")
        base_greedy = col(recs, "base_greedy")
        base_random = col(recs, "base_random")
        gap_best = col(recs, "gap_best")    # best - greedy
        gap_mean = col(recs, "gap_mean")    # mean - greedy
        # argmax (greedy-decode) gap, for the decode-vs-sample check. Older JSONs
        # may lack gap_greedy; fall back to rl_greedy - base_greedy.
        gap_greedy = np.array([r.get("gap_greedy" + sx,
                                     r.get("rl_greedy" + sx, r["rl_greedy"])
                                     - r.get("base_greedy" + sx, r["base_greedy"]))
                               for r in recs])
        t, dof = _t_stat(gap_mean)

        # Scale-normalized gap: (RL - greedy) / (greedy - random), per seed.
        # 0 = RL matches greedy, -1 = RL matches random, >0 = beats greedy.
        # Scale-invariant, so it is comparable ACROSS alpha (unlike the raw gap,
        # whose magnitude grows with the integrated-welfare scale, i.e. demand).
        denom = base_greedy - base_random
        good = np.abs(denom) > 1e-9
        def _norm(gap):
            out = np.full_like(gap, np.nan, dtype=float)
            out[good] = gap[good] / denom[good]
            return out
        ngap_best = _norm(gap_best)
        ngap_mean = _norm(gap_mean)
        ngap_greedy = _norm(gap_greedy)
        nt, ndof = _t_stat(ngap_mean[~np.isnan(ngap_mean)])
        rows.append({
            "alpha": a, "n_seeds": n_seeds,
            "rl_best_mu": rl_best.mean(), "rl_best_sd": rl_best.std(),
            "rl_mean_mu": rl_mean.mean(), "rl_mean_sd": rl_mean.std(),
            "rl_greedy_mu": rl_greedy.mean(),
            "base_greedy_mu": base_greedy.mean(),
            "base_random_mu": base_random.mean(),
            "gap_best_mu": gap_best.mean(), "gap_best_sd": gap_best.std(),
            "gap_mean_mu": gap_mean.mean(), "gap_mean_sd": gap_mean.std(),
            "gap_greedy_mu": gap_greedy.mean(), "gap_greedy_sd": gap_greedy.std(),
            "gap_mean_t": t, "gap_mean_dof": dof,
            "ngap_best_mu": np.nanmean(ngap_best),
            "ngap_mean_mu": np.nanmean(ngap_mean),
            "ngap_mean_sd": np.nanstd(ngap_mean),
            "ngap_greedy_mu": np.nanmean(ngap_greedy),
            "ngap_mean_t": nt, "ngap_mean_dof": ndof,
        })
    return rows


def write_table(rows: list[dict], md_path: Path, csv_path: Path,
                metric: str = "integrated") -> None:
    headers = [
        "alpha", "n_seeds", "RL best-of-N", "RL mean-of-N (±sd)", "RL greedy",
        "greedy base", "random base", "gap best-greedy",
        "gap mean-greedy (±sd)", "gap greedy(argmax)", "t (mean gap vs 0)",
        "norm gap best", "norm gap mean (±sd)", "norm gap argmax",
    ]

    def fmt(r):
        return [
            f"{r['alpha']:g}", f"{r['n_seeds']}",
            f"{r['rl_best_mu']:.4f}",
            f"{r['rl_mean_mu']:.4f} ± {r['rl_mean_sd']:.4f}",
            f"{r['rl_greedy_mu']:.4f}",
            f"{r['base_greedy_mu']:.4f}", f"{r['base_random_mu']:.4f}",
            f"{r['gap_best_mu']:+.4f}",
            f"{r['gap_mean_mu']:+.4f} ± {r['gap_mean_sd']:.4f}",
            f"{r['gap_greedy_mu']:+.4f}",
            f"{r['gap_mean_t']:+.2f} (df={r['gap_mean_dof']})",
            f"{r['ngap_best_mu']:+.3f}",
            f"{r['ngap_mean_mu']:+.3f} ± {r['ngap_mean_sd']:.3f}",
            f"{r['ngap_greedy_mu']:+.3f}",
        ]

    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(fmt(r)) + " |" for r in rows]
    md = ("# TOP-12 RL-vs-greedy gap by induced-demand alpha\n\n"
          f"Welfare metric: **{METRIC_LABEL[metric]}** via CityBuilderEnv, "
          "cost-weight fixed at 0.5. Gap = RL - greedy baseline. "
          "Headline = best-of-N; mean-of-N is "
          "the honest/expected companion (its per-seed spread gives the t-stat). "
          "A positive gap that grows with alpha is the claim: as induced demand "
          "strengthens, learning to shape demand beats reacting greedily.\n\n"
          + "\n".join(lines) + "\n")
    md_path.write_text(md)
    print(f"wrote {md_path}")

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "alpha", "n_seeds", "rl_best_mu", "rl_best_sd",
            "rl_mean_mu", "rl_mean_sd", "rl_greedy_mu",
            "base_greedy_mu", "base_random_mu",
            "gap_best_mu", "gap_best_sd", "gap_mean_mu", "gap_mean_sd",
            "gap_greedy_mu", "gap_greedy_sd", "gap_mean_t", "gap_mean_dof",
            "ngap_best_mu", "ngap_mean_mu", "ngap_mean_sd", "ngap_greedy_mu",
            "ngap_mean_t", "ngap_mean_dof",
        ])
        for r in rows:
            w.writerow([
                r["alpha"], r["n_seeds"], r["rl_best_mu"], r["rl_best_sd"],
                r["rl_mean_mu"], r["rl_mean_sd"], r["rl_greedy_mu"],
                r["base_greedy_mu"], r["base_random_mu"],
                r["gap_best_mu"], r["gap_best_sd"], r["gap_mean_mu"],
                r["gap_mean_sd"], r["gap_greedy_mu"], r["gap_greedy_sd"],
                r["gap_mean_t"], r["gap_mean_dof"],
                r["ngap_best_mu"], r["ngap_mean_mu"], r["ngap_mean_sd"],
                r["ngap_greedy_mu"], r["ngap_mean_t"], r["ngap_mean_dof"],
            ])
    print(f"wrote {csv_path}")


def plot_gap(rows: list[dict], out: Path, metric: str = "integrated") -> None:
    alphas = [r["alpha"] for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.2))

    # LEFT: the two gap curves.
    gap_best = [r["gap_best_mu"] for r in rows]
    gap_best_sd = [r["gap_best_sd"] for r in rows]
    gap_mean = [r["gap_mean_mu"] for r in rows]
    gap_mean_sd = [r["gap_mean_sd"] for r in rows]
    ax1.errorbar(alphas, gap_best, yerr=gap_best_sd, marker="o", capsize=3,
                 color="C0", label="best-of-N $-$ greedy (headline)")
    ax1.errorbar(alphas, gap_mean, yerr=gap_mean_sd, marker="s", capsize=3,
                 color="C3", label="mean-of-N $-$ greedy (expected)")
    ax1.axhline(0, color="k", lw=0.8, ls=":")
    ax1.set_xlabel(r"induced-demand strength $\alpha$")
    ax1.set_ylabel("RL $-$ greedy cumulative welfare")
    ax1.set_title("RL-vs-greedy gap")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # RIGHT: absolute welfare context.
    ax2.errorbar(alphas, [r["rl_best_mu"] for r in rows],
                 yerr=[r["rl_best_sd"] for r in rows], marker="o", capsize=3,
                 color="C0", label="RL best-of-N")
    ax2.errorbar(alphas, [r["rl_mean_mu"] for r in rows],
                 yerr=[r["rl_mean_sd"] for r in rows], marker="s", capsize=3,
                 color="C3", label="RL mean-of-N")
    ax2.plot(alphas, [r["base_greedy_mu"] for r in rows], marker="D",
             color="C1", label="greedy baseline")
    ax2.plot(alphas, [r["base_random_mu"] for r in rows], marker="v",
             color="C7", label="random baseline")
    ax2.set_xlabel(r"induced-demand strength $\alpha$")
    ax2.set_ylabel("cumulative welfare")
    ax2.set_title("Absolute welfare by policy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # NORMALIZED gap: scale-invariant, comparable across alpha. 0 = greedy,
    # -1 = random. A flat curve means the raw-gap "widening with alpha" was just
    # the welfare scale growing with demand, not RL degrading relative to greedy.
    ax3.errorbar(alphas, [r["ngap_mean_mu"] for r in rows],
                 yerr=[r["ngap_mean_sd"] for r in rows], marker="s", capsize=3,
                 color="C3", label="mean-of-N (expected)")
    ax3.plot(alphas, [r["ngap_best_mu"] for r in rows], marker="o",
             color="C0", label="best-of-N (headline)")
    ax3.plot(alphas, [r["ngap_greedy_mu"] for r in rows], marker="D",
             color="C2", label="argmax (greedy-decode)")
    ax3.axhline(0, color="C1", lw=1.0, ls="--", label="greedy baseline")
    ax3.axhline(-1, color="C7", lw=1.0, ls=":", label="random baseline")
    ax3.set_xlabel(r"induced-demand strength $\alpha$")
    ax3.set_ylabel(r"$(\mathrm{RL}-\mathrm{greedy})/(\mathrm{greedy}-\mathrm{random})$")
    ax3.set_title("Normalized gap (scale-invariant)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    fig.suptitle("TOP-12: does learning to shape demand beat greedy as "
                 r"$\alpha$ grows?  [" + METRIC_LABEL[metric] + "]")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def print_summary(rows: list[dict]) -> None:
    print("\n--- RL-vs-greedy gap by alpha (mean over seeds) ---")
    for r in rows:
        sig = ("*" if abs(r["gap_mean_t"]) >= 2 and r["gap_mean_dof"] >= 1
               else " ")
        print(f"  alpha={r['alpha']:<5g} n={r['n_seeds']}  "
              f"gap_best={r['gap_best_mu']:+.4f}  "
              f"gap_mean={r['gap_mean_mu']:+.4f}±{r['gap_mean_sd']:.4f}  "
              f"t={r['gap_mean_t']:+.2f}{sig}")
    print("  (* = |t| >= 2, i.e. mean gap plausibly != 0 at this seed count)")

    # Decode check: argmax (greedy-decode) vs sampled best/mean. If gap_greedy is
    # far below gap_best, the deterministic policy collapses -> decode/training
    # issue rather than a pure SNR one.
    print("\n--- argmax (greedy-decode) vs sampled ---")
    for r in rows:
        print(f"  alpha={r['alpha']:<5g}  gap_greedy(argmax)={r['gap_greedy_mu']:+.4f}"
              f"  gap_best(sampled)={r['gap_best_mu']:+.4f}"
              f"  gap_mean(sampled)={r['gap_mean_mu']:+.4f}")

    # Scale-normalized gap: comparable across alpha. 0 = matches greedy,
    # -1 = matches random, >0 = beats greedy.
    print("\n--- normalized gap  (RL-greedy)/(greedy-random)  [0=greedy, -1=random] ---")
    for r in rows:
        nsig = ("*" if abs(r["ngap_mean_t"]) >= 2 and r["ngap_mean_dof"] >= 1
                else " ")
        print(f"  alpha={r['alpha']:<5g}  ngap_best={r['ngap_best_mu']:+.3f}"
              f"  ngap_mean={r['ngap_mean_mu']:+.3f}±{r['ngap_mean_sd']:.3f}"
              f"  ngap_greedy={r['ngap_greedy_mu']:+.3f}  t={r['ngap_mean_t']:+.2f}{nsig}")
    print("  (a FLAT normalized curve across alpha => the raw-gap widening was "
          "just the welfare scale growing with demand, not RL degrading)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="results/rl_alpha_ablation.json")
    p.add_argument("--metric", choices=("integrated", "delta"),
                   default="integrated",
                   help="welfare functional: 'integrated' (-sum_t C_t, M2-style, "
                        "alpha-sensitive; default) or 'delta' (telescoped "
                        "C(s0)-C(sT), the original).")
    args = p.parse_args(argv)

    inp = Path(args.inp)
    payload = _load(inp)
    rows = build_rows(payload, metric=args.metric)
    if not rows:
        print("No results in JSON; nothing to plot.")
        return

    # Tag outputs by metric so delta/integrated figures don't overwrite.
    stem = inp.with_suffix("")
    tag = "_gap" if args.metric == "delta" else "_gap_int"
    ttag = "_table" if args.metric == "delta" else "_table_int"
    print(f"metric = {args.metric}  ({METRIC_LABEL[args.metric]})")
    plot_gap(rows, stem.with_name(stem.name + tag).with_suffix(".pdf"),
             metric=args.metric)
    write_table(rows,
                stem.with_name(stem.name + ttag).with_suffix(".md"),
                stem.with_name(stem.name + ttag).with_suffix(".csv"),
                metric=args.metric)
    print_summary(rows)


if __name__ == "__main__":
    main()
