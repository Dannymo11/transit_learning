"""Export a stitched 'agent builds network, then city grows' trajectory to JSON
for the interactive HTML scrubber (city_builder_viz.html) and the MP4 renderer
(render_viz.py).

This produces REAL numbers from the torch model, in the exact schema the viewer
expects. It reuses the tested helpers in alpha_sweep.py / viz_city_growth.py so
the dynamics match the TOP-10 sweep.

Two network sources:
  * greedy (default)  -- john_init network on the t=0 demand. No checkpoint
                         needed; runs immediately.
  * rl                -- a trained policy's network via eval_rl.rl_build_routes.
                         Requires a checkpoint; see the RL hook below.

Stitching is SEQUENTIAL (matches the chosen presentation design): the network is
built once (Phase 1, animated stop-by-stop in build order), then HELD FIXED while
land use evolves for `horizon` years (Phase 2), recorded for every alpha so the
viewer's alpha toggle works.

Run from the repo root::

    python -m learning.city_builder.export_viz_trajectory \
        --source greedy --alphas 0.0 0.5 1.0 2.0 --out results/viz_trajectory.json

Then drop results/viz_trajectory.json next to city_builder_viz.html (or rebuild
the embedded HTML with build_html.py) and re-run render_viz.py for the MP4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from simulation.citygraph_dataset import STOP_KEY
from simulation.transit_time_estimator import MyCostModule

from learning.city_builder import (
    LandUseConfig, LandUseDynamics, recompute_demand_in_place, step_world,
)
from learning.city_builder.alpha_sweep import (
    REPO_ROOT, DEFAULT_INSTANCES_DIR,
    HORIZON, BETA_GRAVITY, CAP_MULTIPLIER, MIN_ROUTE_LEN, MAX_ROUTE_LEN,
    _fresh_data, _new_state, _initial_activity_from_od,
    _greedy_network, transit_drive_times,
)
from learning.city_builder.viz_city_growth import _routes_from_network
from torch_utils import get_batch_tensor_from_routes


def _construction_steps(routes: list[list[int]]) -> list[dict]:
    """Per-stop 'add node' events in build order -- the agent's sequential route
    construction. For RL this matches the order the policy extended each route;
    for greedy it is route-by-route, stop-by-stop."""
    steps = []
    for ri, r in enumerate(routes):
        for k in range(1, len(r)):
            steps.append({"route": ri, "from": int(r[k - 1]), "to": int(r[k]),
                          "partial": [int(x) for x in r[:k + 1]]})
    return steps


def _street_edges(data, cost_obj) -> list[list[int]]:
    street_adj = _new_state(data, cost_obj).street_adj[0]
    have = (street_adj > 0) & torch.isfinite(street_adj)
    iu = torch.triu(have, diagonal=1).nonzero().cpu().numpy()
    return [[int(i), int(j)] for i, j in iu]


def get_routes(source: str, data, cost_obj, cfg_path: str | None):
    """Return the agent's network as a list of node-lists (one per route)."""
    if source == "greedy":
        network = _greedy_network(data, cost_obj)            # (1, n_routes, max_len)
        return _routes_from_network(network), network
    if source == "rl":
        # ---- RL HOOK -------------------------------------------------------
        # Requires a trained checkpoint. The cleanest path reuses eval_rl, which
        # is hydra-configured. Easiest: run eval_rl once to dump the routes the
        # policy builds (state.routes[0]) to a pickle/json, then load them here
        # with --rl-routes <file>. See eval_rl.rl_build_routes for the rollout.
        raise SystemExit(
            "RL source: pass --rl-routes <json list-of-routes> produced from "
            "eval_rl.rl_build_routes(...), or wire the hydra model load here.")
    raise ValueError(source)


def grow(network, data, cost_obj, x0, cap, alpha, horizon, seed):
    """Hold the network fixed and evolve land use for `horizon` years, recording
    per-year activity, totals, capacity fraction, and normalized accessibility."""
    acc_dt = transit_drive_times(data, cost_obj, network)
    dyn = LandUseDynamics(
        initial_activity=x0,
        config=LandUseConfig(alpha=alpha, base_rate=1.0, sigma_eps=0.0,
                             cap_multiplier=CAP_MULTIPLIER,
                             beta_accessibility=BETA_GRAVITY),
        seed=seed,
    )
    developable = cap > 1e-6
    x = x0.clone()
    activity, sums, fracs, accs = [], [], [], []
    for t in range(horizon + 1):
        activity.append(x.detach().cpu().numpy().tolist())
        sums.append(float(x.sum().item()))
        fr = float((x[developable] >= cap[developable] - 1e-6).float().mean().item())
        fracs.append(fr)
        # normalized accessibility A_tilde for the CURRENT state (lines up with
        # activity[t]); dyn.step is pure here (sigma_eps=0 -> no state mutation).
        _, a_now = dyn.step(x, acc_dt)
        accs.append(a_now.detach().cpu().numpy().tolist())
        if t == horizon:
            break
        x, _, _ = step_world(dyn, data, x, accessibility_drive_times=acc_dt)
    return {"activity": activity, "sum_activity": sums,
            "frac_at_cap": fracs, "accessibility": accs}


def build_payload(instance, instances_dir, routes, construction, alphas,
                  horizon, seed, source_label, network=None, steps=None,
                  rollout_activity=None, rollout_alpha=None):
    """Assemble the viewer JSON. `routes` is a list of node-lists; `construction`
    is the per-stop build log (true RL action order, or reveal order for greedy);
    `steps` (optional) is the per-action policy-inspector log (candidate
    distribution + chosen move + halt prob). If `network` is None it is rebuilt
    from `routes`. Reused by both the CLI here and eval_rl's one-command viz
    export, so the dynamics match the sweep."""
    data = _fresh_data(instances_dir, instance)
    cost_obj = MyCostModule(symmetric_routes=True)
    x0 = _initial_activity_from_od(data.demand)
    cap = CAP_MULTIPLIER * x0
    recompute_demand_in_place(data, x0, beta=BETA_GRAVITY)

    if network is None:
        network = get_batch_tensor_from_routes(
            [routes], device=_new_state(data, cost_obj).device,
            max_route_len=MAX_ROUTE_LEN)

    series = {str(a): grow(network, data, cost_obj, x0, cap, a, horizon, seed)
              for a in alphas}
    rollout = None
    if rollout_activity is not None:
        rollout = {"alpha": rollout_alpha, "activity": rollout_activity,
                   "n_years": len(rollout_activity) - 1}
    return {
        "meta": {
            "instance": instance, "n_nodes": int(x0.shape[0]),
            "horizon": horizon, "n_routes": len(routes),
            "beta": BETA_GRAVITY, "cap_multiplier": CAP_MULTIPLIER,
            "alphas": list(alphas), "source": source_label,
            "note": "Phase 1 = sequential route construction; Phase 2 = land-use growth (network held fixed).",
        },
        "node_locs": data[STOP_KEY].pos.detach().cpu().numpy().tolist(),
        "street_edges": _street_edges(data, cost_obj),
        "routes": [[int(n) for n in r] for r in routes],
        "construction": construction,
        "steps": steps if steps is not None else [],
        "rollout": rollout,
        "cap": cap.detach().cpu().numpy().tolist(),
        "x0": x0.detach().cpu().numpy().tolist(),
        "series": series,
    }


def write_payload(payload, out):
    out = Path(out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(f"wrote {out}  (routes={payload['meta']['n_routes']}, "
          f"steps={len(payload['construction'])}, source={payload['meta']['source']})")
    for a in payload["meta"]["alphas"]:
        s = payload["series"][str(a)]
        print(f"  alpha={a}: sum(x) {s['sum_activity'][0]:.0f} -> {s['sum_activity'][-1]:.0f}"
              f"  frac_at_cap_end={s['frac_at_cap'][-1]:.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", default="Mandl")
    p.add_argument("--instances-dir", default=str(DEFAULT_INSTANCES_DIR))
    p.add_argument("--source", choices=("greedy", "rl"), default="greedy")
    p.add_argument("--rl-routes", default=None,
                   help="JSON {routes:[...], construction:[...]} from a logged RL "
                        "rollout (eval_rl.rl_build_routes_logged). Overrides --source.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/viz_trajectory.json")
    args = p.parse_args(argv)

    if args.rl_routes:
        blob = json.loads(Path(args.rl_routes).read_text())
        routes = blob["routes"] if isinstance(blob, dict) else blob
        construction = (blob.get("construction") if isinstance(blob, dict) else None) \
            or _construction_steps(routes)
        source_label = "rl policy (true action order), real torch dynamics"
        network = None
    else:
        data = _fresh_data(args.instances_dir, args.instance)
        cost_obj = MyCostModule(symmetric_routes=True)
        recompute_demand_in_place(
            data, _initial_activity_from_od(data.demand), beta=BETA_GRAVITY)
        routes, network = get_routes(args.source, data, cost_obj, None)
        construction = _construction_steps(routes)
        source_label = "greedy (john_init) network, real torch dynamics"

    payload = build_payload(args.instance, args.instances_dir, routes, construction,
                            args.alphas, args.horizon, args.seed, source_label,
                            network=network)
    write_payload(payload, args.out)


if __name__ == "__main__":
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    main()
