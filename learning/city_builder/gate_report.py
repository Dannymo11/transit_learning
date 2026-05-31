"""TOP-11 gate report --- aggregate per-seed gate JSONs and apply the pass/fail
rule against Holliday & Dudek 2024 Table 2 (LC-100).

numpy-only: runs on your laptop without torch. Point it at the per-seed JSONs
written by eval_gate.py (pull them off the Modal volume first, or run eval_gate
locally).

Gate rule (from the TOP-11 calibration doc):
  * per cost-weight alpha: mu_ours[a] <= C_paper[a] + max(sigma_paper[a],
    sigma_ours[a])  --- i.e. within seed variance of (or below) the LC-100 number.
  * per-instance PASS: the rule holds at alpha=0.5 (the balanced, most diagnostic
    column). alpha=0 / alpha=1 misses are investigable but not blocking.
  * per-instance FAIL: alpha=0.5 misses by more than max(sigma_paper, sigma_ours).

Run from the repo root::

    python -m learning.city_builder.gate_report \
        --glob 'results/gate/mandl_seed*_gate.json' --city Mandl
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

# Holliday & Dudek 2024, Table 2, LC-100. (mean, std) of C(alpha,G,R) over 10
# seeds. Keyed by city then by cost-weight alpha. Lower is better.
PAPER_LC100 = {
    "Mandl":    {0.0: (0.697, 0.011), 0.5: (0.558, 0.003), 1.0: (0.328, 0.001)},
    "Mumford0": {0.0: (0.847, 0.025), 0.5: (0.916, 0.006), 1.0: (0.721, 0.004)},
    "Mumford1": {0.0: (1.747, 0.034), 0.5: (1.272, 0.018), 1.0: (0.573, 0.004)},
    "Mumford2": {0.0: (1.315, 0.049), 0.5: (0.989, 0.021), 1.0: (0.495, 0.002)},
    "Mumford3": {0.0: (1.333, 0.064), 0.5: (0.984, 0.026), 1.0: (0.476, 0.001)},
}
GATE_ALPHA = 0.5  # the alpha the per-instance pass/fail hinges on


def load_seed_costs(pattern: str) -> dict[float, list[float]]:
    """alpha -> [cost per seed]."""
    by_alpha: dict[float, list[float]] = defaultdict(list)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No gate JSONs matched: {pattern}")
    for f in files:
        d = json.loads(open(f).read())
        for a_str, rec in d["results"].items():
            by_alpha[float(a_str)].append(float(rec["cost"]))
    return dict(by_alpha), files


def report(city: str, by_alpha: dict[float, list[float]], files: list[str]) -> bool:
    if city not in PAPER_LC100:
        raise SystemExit(f"No paper targets for city '{city}'. "
                         f"Known: {list(PAPER_LC100)}")
    paper = PAPER_LC100[city]
    print(f"\nTOP-11 gate report — {city}  ({len(files)} seed(s))")
    print(f"{'alpha':>6} | {'ours mean±std':>16} | {'paper LC-100':>14} | "
          f"{'band (+tol)':>11} | {'verdict':>8}")
    print("-" * 70)

    overall_pass = None
    for a in sorted(paper):
        costs = np.array(by_alpha.get(a, []), dtype=float)
        p_mu, p_sd = paper[a]
        if costs.size == 0:
            print(f"{a:6.1f} | {'(missing)':>16} | {p_mu:6.3f} ± {p_sd:5.3f} | "
                  f"{'—':>11} | {'—':>8}")
            continue
        o_mu, o_sd = costs.mean(), costs.std()
        tol = max(p_sd, o_sd)
        band = p_mu + tol
        ok = o_mu <= band
        verdict = "PASS" if ok else "FAIL"
        print(f"{a:6.1f} | {o_mu:7.4f} ± {o_sd:6.4f} | {p_mu:6.3f} ± {p_sd:5.3f} | "
              f"{band:11.4f} | {verdict:>8}")
        if abs(a - GATE_ALPHA) < 1e-9:
            overall_pass = ok

    print("-" * 70)
    if overall_pass is None:
        print(f"GATE: INCONCLUSIVE — no alpha={GATE_ALPHA} results found.")
        return False
    print(f"GATE (alpha={GATE_ALPHA}): "
          f"{'PASS — within LC-100 seed variance' if overall_pass else 'FAIL — exceeds LC-100 + tolerance; fix before TOP-12'}")
    if overall_pass:
        # warn if either extreme missed, per the doc (investigable, not blocking)
        for a in (0.0, 1.0):
            costs = np.array(by_alpha.get(a, []), dtype=float)
            if costs.size and costs.mean() > paper[a][0] + max(paper[a][1], costs.std()):
                print(f"  note: alpha={a} misses its LC-100 band "
                      f"(investigable extreme, not blocking).")
    return overall_pass


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", required=True,
                   help="glob for per-seed gate JSONs, e.g. "
                        "'results/gate/mandl_seed*_gate.json'")
    p.add_argument("--city", default="Mandl", choices=list(PAPER_LC100))
    p.add_argument("--out", default=None, help="optional summary JSON path")
    args = p.parse_args(argv)

    by_alpha, files = load_seed_costs(args.glob)
    passed = report(args.city, by_alpha, files)

    if args.out:
        summary = {
            "city": args.city, "n_seeds": len(files), "passed": passed,
            "by_alpha": {
                str(a): {"mean": float(np.mean(v)), "std": float(np.std(v)),
                         "n": len(v)}
                for a, v in by_alpha.items()
            },
        }
        open(args.out, "w").write(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
