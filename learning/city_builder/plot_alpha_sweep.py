"""M2 / TOP-10 --- figures for the alpha-sensitivity sweep.

Reads the JSON written by alpha_sweep.py and produces the milestone deliverables:

  1. <stem>_gap.pdf         --- greedy-vs-random cumulative-welfare gap vs alpha,
                                mean +/- std over seeds. The decision-gate figure.
  2. <stem>_evolution.pdf   --- city-evolution sanity: total activity sum(x) over
                                the horizon per alpha, plus the fraction of zones
                                pinned at the activity cap (runaway indicator).

Run from the repo root::

    python -m learning.city_builder.plot_alpha_sweep \
        --in results/top10_alpha_sweep.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _group(results: list[dict]) -> dict:
    """(alpha, baseline) -> list of per-rollout dicts."""
    g: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for r in results:
        g[(r["alpha"], r["baseline"])].append(r)
    return g


def plot_gap(payload: dict, out: Path) -> None:
    results = payload["results"]
    g = _group(results)
    alphas = sorted({r["alpha"] for r in results})

    greedy_mean, greedy_std, rand_mean, rand_std = [], [], [], []
    gap_mean, gap_std = [], []
    for a in alphas:
        gw = np.array([r["cumulative_welfare"] for r in g[(a, "greedy")]])
        rw = np.array([r["cumulative_welfare"] for r in g[(a, "random")]])
        greedy_mean.append(gw.mean()); greedy_std.append(gw.std())
        rand_mean.append(rw.mean()); rand_std.append(rw.std())
        # Paired gap per seed (greedy - random); higher = greedy better.
        gap = gw - rw
        gap_mean.append(gap.mean()); gap_std.append(gap.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.errorbar(alphas, greedy_mean, yerr=greedy_std, marker="o",
                 capsize=3, label="greedy (john_init)")
    ax1.errorbar(alphas, rand_mean, yerr=rand_std, marker="s",
                 capsize=3, label="random")
    ax1.set_xlabel(r"induced-demand strength $\alpha$")
    ax1.set_ylabel("cumulative welfare  (= $-\\sum_t$ cost)")
    ax1.set_title("Cumulative welfare by policy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.errorbar(alphas, gap_mean, yerr=gap_std, marker="D",
                 color="C2", capsize=3)
    ax2.axhline(0, color="k", lw=0.8, ls=":")
    ax2.set_xlabel(r"induced-demand strength $\alpha$")
    ax2.set_ylabel("greedy $-$ random  (paired, per seed)")
    ax2.set_title("Greedy-vs-random gap vs " + r"$\alpha$")
    ax2.grid(alpha=0.3)

    fig.suptitle("TOP-10 M2 decision gate: does induced demand open a policy gap?")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def plot_evolution(payload: dict, out: Path) -> None:
    results = payload["results"]
    g = _group(results)
    alphas = sorted({r["alpha"] for r in results})
    horizon = payload["config"]["horizon"]
    years = list(range(horizon + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for a in alphas:
        # Average the greedy rollouts' activity trajectory across seeds.
        rows = g[(a, "greedy")]
        sums = np.array([r["per_year_sum_activity"] for r in rows]).mean(0)
        caps = np.array([r["per_year_frac_at_cap"] for r in rows]).mean(0)
        ax1.plot(years, sums, marker="o", label=fr"$\alpha$={a}")
        ax2.plot(years, caps, marker="o", label=fr"$\alpha$={a}")

    ax1.set_xlabel("year"); ax1.set_ylabel(r"total activity $\sum_i x_i$")
    ax1.set_title("City growth (greedy rollout)")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_xlabel("year")
    ax2.set_ylabel("fraction of zones at cap")
    ax2.set_title("Runaway check (1.0 = fully saturated)")
    ax2.set_ylim(-0.02, 1.02)
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("TOP-10 city-evolution sanity: growth without runaway")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def print_static_check(payload: dict) -> None:
    """alpha=0 must be static: per-year cost constant year-over-year."""
    g = _group(payload["results"])
    print("\n--- alpha=0 static-reproduction check (per-year cost should be flat) ---")
    for baseline in ("greedy", "random"):
        rows = g.get((0.0, baseline), [])
        if not rows:
            continue
        costs = np.array([r["per_year_cost"] for r in rows])  # (seeds, T+1)
        # max year-over-year drift within a rollout, averaged over seeds
        drift = np.abs(np.diff(costs, axis=1)).max(axis=1).mean()
        print(f"  {baseline:<6}: mean max year-over-year cost drift = {drift:.6g} "
              f"(expect ~0)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="results/top10_alpha_sweep.json")
    args = p.parse_args(argv)

    inp = Path(args.inp)
    payload = _load(inp)
    stem = inp.with_suffix("")
    plot_gap(payload, stem.with_name(stem.name + "_gap").with_suffix(".pdf"))
    plot_evolution(payload, stem.with_name(stem.name + "_evolution").with_suffix(".pdf"))
    print_static_check(payload)


if __name__ == "__main__":
    main()
