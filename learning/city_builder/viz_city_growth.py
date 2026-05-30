"""M3/M5 city-evolution renderer --- zones + transit network growing together.

Runs a single multi-year rollout (a baseline now; a trained policy later) and
renders, per snapshot year, the spatial city state:

  * each zone as a marker sized and colored by its activity x_t (induced-demand
    growth),
  * the street skeleton in faint grey,
  * the transit network IN FORCE that year drawn as colored polylines --- i.e.
    the routes the agent has built so far.

This is the foundation of the M5 city-evolution hero figure (project doc, M5
Visualization note). For now `--baseline greedy` uses john_init as a stand-in
for the policy; once a trained policy exists, add a `policy` branch in
`_network_for_year` that calls it and everything else (recording + rendering)
is unchanged.

Reuses the rollout primitives from alpha_sweep.py so the dynamics match the
TOP-10 sweep exactly (same Mandl setup, same step_world coupling).

Run from the repo root::

    python -m learning.city_builder.viz_city_growth \
        --alpha 0.5 --baseline greedy --seed 0 \
        --out results/city_growth

Writes <out>.png (multi-panel snapshots) and <out>.gif (per-year animation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import animation, cm

from simulation.citygraph_dataset import STOP_KEY
from simulation.transit_time_estimator import MyCostModule

from learning.city_builder import (
    LandUseConfig, LandUseDynamics, recompute_demand_in_place, step_world,
    M2_WORKING_ALPHA,
)
from learning.city_builder.alpha_sweep import (
    REPO_ROOT, DEFAULT_INSTANCES_DIR,
    N_ROUTES, MIN_ROUTE_LEN, MAX_ROUTE_LEN, HORIZON, REPLAN_EVERY,
    BETA_GRAVITY, CAP_MULTIPLIER,
    _fresh_data, _new_state, _initial_activity_from_od,
    _greedy_network, _random_network,
)


def _network_for_year(baseline: str, data, cost_obj, rand_gen):
    """The 'action' taken at a replan year. Add a `policy` branch here when a
    trained policy is available; the rest of this module is policy-agnostic."""
    if baseline == "greedy":
        return _greedy_network(data, cost_obj)
    if baseline == "random":
        return _random_network(data, cost_obj, rand_gen)
    raise ValueError(baseline)


def _routes_from_network(network: torch.Tensor) -> list[list[int]]:
    """(1, n_routes, max_len) tensor with -1 padding -> list of node-index lists."""
    routes = []
    net = network[0]
    for r in range(net.shape[0]):
        seq = net[r]
        seq = seq[seq >= 0].tolist()
        if len(seq) >= 2:
            routes.append(seq)
    return routes


def record_rollout(alpha, seed, baseline, instances_dir, instance,
                   horizon, replan_every):
    """Run one rollout, capturing (x_t, routes_t) for every year plus the
    static node positions and street edges."""
    data = _fresh_data(instances_dir, instance)
    cost_obj = MyCostModule(symmetric_routes=True)

    x_0 = _initial_activity_from_od(data.demand)
    cap = CAP_MULTIPLIER * x_0
    recompute_demand_in_place(data, x_0, beta=BETA_GRAVITY)

    dyn = LandUseDynamics(
        initial_activity=x_0,
        config=LandUseConfig(alpha=alpha, base_rate=1.0, sigma_eps=0.0,
                             cap_multiplier=CAP_MULTIPLIER,
                             beta_accessibility=BETA_GRAVITY),
        seed=seed,
    )
    rand_gen = torch.Generator().manual_seed(10_000 + seed)

    node_locs = data[STOP_KEY].pos.detach().cpu().numpy()
    # street skeleton from the (N, N) street adjacency (finite, positive entries)
    street_adj = _new_state(data, cost_obj).street_adj[0]
    have = (street_adj > 0) & torch.isfinite(street_adj)
    iu = torch.triu(have, diagonal=1).nonzero().cpu().numpy()
    street_edges = [(int(i), int(j)) for i, j in iu]

    x = x_0.clone()
    network = None
    activity, networks = [], []
    for t in range(horizon + 1):
        if t % replan_every == 0:
            network = _network_for_year(baseline, data, cost_obj, rand_gen)
        activity.append(x.detach().cpu().numpy().copy())
        networks.append(_routes_from_network(network))
        if t == horizon:
            break
        x, _, _ = step_world(dyn, data, x)

    return {
        "node_locs": node_locs,
        "street_edges": street_edges,
        "activity": np.array(activity),     # (T+1, N)
        "networks": networks,               # list[T+1] of list[route]
        "cap": cap.detach().cpu().numpy(),
    }


def _draw_panel(ax, rec, t, gmax, route_colors):
    locs = rec["node_locs"]
    # street skeleton
    for i, j in rec["street_edges"]:
        ax.plot([locs[i, 0], locs[j, 0]], [locs[i, 1], locs[j, 1]],
                color="0.85", lw=1.0, zorder=1)
    # transit routes in force this year
    for k, route in enumerate(rec["networks"][t]):
        pts = locs[route]
        ax.plot(pts[:, 0], pts[:, 1], color=route_colors[k % len(route_colors)],
                lw=2.6, alpha=0.85, zorder=2, solid_capstyle="round")
    # zones sized/colored by activity
    x = rec["activity"][t]
    sizes = 40 + 1500 * (x / gmax)
    sc = ax.scatter(locs[:, 0], locs[:, 1], s=sizes, c=x, cmap="viridis",
                    vmin=0, vmax=gmax, edgecolors="k", linewidths=0.6, zorder=3)
    frac_cap = float((x >= rec["cap"] - 1e-6).mean())
    ax.set_title(fr"year {t}   $\sum x$={x.sum():.0f}   {frac_cap*100:.0f}% at cap",
                 fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    return sc


def render(rec, out_stem: Path, alpha: float, baseline: str,
           snapshot_years: list[int]):
    gmax = rec["activity"].max()
    n_routes = max((len(rec["networks"][t]) for t in range(len(rec["networks"]))),
                   default=N_ROUTES)
    route_colors = [cm.tab10(i % 10) for i in range(max(n_routes, 1))]
    T = rec["activity"].shape[0] - 1
    years = [y for y in snapshot_years if 0 <= y <= T]

    # --- multi-panel snapshots ---
    ncol = 3
    nrow = int(np.ceil(len(years) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.7 * nrow),
                             squeeze=False)
    sc = None
    for ax, yr in zip(axes.flat, years):
        sc = _draw_panel(ax, rec, yr, gmax, route_colors)
    for ax in axes.flat[len(years):]:
        ax.axis("off")
    fig.suptitle(
        f"Mandl city evolution under induced demand "
        fr"($\alpha$={alpha}, {baseline} network): zones + transit network",
        fontsize=13)
    fig.colorbar(sc, ax=axes, shrink=0.85, label="zone activity x")
    png = out_stem.with_suffix(".png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"wrote {png}")
    plt.close(fig)

    # --- animation over every year ---
    figA, axA = plt.subplots(figsize=(6.4, 5.6))
    def frame(t):
        axA.clear()
        _draw_panel(axA, rec, t, gmax, route_colors)
    ani = animation.FuncAnimation(figA, frame, frames=range(T + 1), interval=600)
    gif = out_stem.with_suffix(".gif")
    ani.save(gif, writer=animation.PillowWriter(fps=2))
    print(f"wrote {gif}")
    plt.close(figA)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="Mandl")
    p.add_argument("--instances-dir", default=str(DEFAULT_INSTANCES_DIR))
    p.add_argument("--alpha", type=float, default=M2_WORKING_ALPHA)
    p.add_argument("--baseline", choices=("greedy", "random"), default="greedy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--replan-every", type=int, default=REPLAN_EVERY)
    p.add_argument("--snapshots", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])
    p.add_argument("--out", default="results/city_growth")
    args = p.parse_args(argv)

    rec = record_rollout(args.alpha, args.seed, args.baseline,
                         args.instances_dir, args.instance,
                         args.horizon, args.replan_every)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    render(rec, out, args.alpha, args.baseline, args.snapshots)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
