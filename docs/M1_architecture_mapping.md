# M1 Architecture Mapping — Holliday → City Builder

**Linear ticket:** [TOP-6](https://linear.app/topograph-stanford/issue/TOP-6/m1-architecture-mapping-note-identify-dynamics-hooks)
**Repo state:** master @ `ca77bb5` (stack rebuild to torch 2.8 / PyG 2.7 / Python 3.12) + restored `learning/inductive_route_learning.py`
**Date:** 2026-05-18

## The library boundary

The CS224R project proposal commits to using Holliday's repo *as a library, not a framework*. This note pins that boundary to concrete file/class names so subsequent milestones (M2 LandUseDynamics integration, M3 multi-year training, M5 experiments) can be specified against a stable lift surface.

Three columns: **KEEP** (use unchanged), **WRAP** (call from new outer MDP), **REPLACE** (write fresh).

### KEEP wholesale

| Module | What it does | Why we keep it |
|---|---|---|
| `simulation/transit_time_estimator.py::CostModule._cost_helper` (line 862) | Floyd-Warshall over street graph + transfer counting + per-OD trip time + unserved-demand bookkeeping. Outputs `CostHelperOutput`. | This is the welfare *primitive*. Re-implementing it correctly is research-quality work; the proposal's Method section commits to reusing it as the welfare oracle. |
| `simulation/transit_time_estimator.py::MyCostModule.forward` (line 1026) | Combines `_cost_helper` output into a scalar cost: `demand_cost * dtw + route_cost * rtw + constraint_violation_penalty`. Variable weights for op/pp/intermediate. | This is the welfare scalarization. Both reward variants in M5 Exp 3 (ridership-max, accessibility-weighted) are weight settings on this module, not new code. |
| `simulation/citygraph_dataset.py::CityGraphData` (line 391) + `STOP_KEY/STREET_KEY/DEMAND_KEY/ROUTE_KEY` (line 48) | HeteroData schema: nodes carry `pos` and node features; edges in three categories. `data.drive_times`, `data.demand`, `data.fixed_routes` are tensor attributes the cost module reads. | This is the city representation. Bogotá adapter (M4) targets this schema. |
| `simulation/citygraph_dataset.py::CityGraphData.from_mumford_data` (line 456) | Parses `*Coords.txt`, `*TravelTimes.txt`, `*Demand.txt`; builds Floyd-Warshall via `torch_utils.floyd_warshall`; populates demand edges. | Used directly for the α=0 sanity check. M4 Bogotá adapter mirrors its output shape. |
| `learning/models.py::PathCombiningRouteGenerator` (line 1192) | The GNN policy that gets trained with PPO. Takes a `RouteGenBatchState`, outputs a route plan via inductive path-combining. | The proposal commits to reusing the GNN + PPO trainer "largely unchanged." Reproducing Holliday's published Mandl numbers is the credibility anchor — that requires this exact architecture. |
| `learning/inductive_route_learning.py::train_ppo` (line 294) | PPO inner loop with NN baseline, GAE, clipped ratio, val-period evaluation. | Same reason. This is the inner loop our multi-year outer MDP wraps. |
| `learning/inductive_route_learning.py::NNBaseline` (line 104), `RollingBaseline` (line 76) | PPO value baselines. | Plumbing for `train_ppo`. |

**The seam :** the per-year welfare evaluation in our multi-year MDP is exactly `MyCostModule.forward(state).cost`, where `state` is a `RouteGenBatchState` populated with that year's network and demand. Everything else in the lift surface exists to make that one call meaningful.

### WRAP (call from new outer MDP, but don't modify)

| Module | What we do with it |
|---|---|
| `RouteGenBatchState` (transit_time_estimator.py line 84) | This is the *per-step inductive route-construction* state — holds `routes`, `n_routes_to_plan`, `n_routes_left_to_plan`, `has_current_route`, `min_route_len`, `max_route_len`, `drive_times`, `demand`. We keep it intact for the *inner* per-year route-construction episode. The *outer* multi-year MDP holds a separate `CityBuilderState` that owns a `CityGraphData` + `activity_per_zone` + `remaining_budget`, and at year-`t` materializes a `RouteGenBatchState` from those for the inner PPO rollout. |
| `learning/inductive_route_learning.py::setup_and_train` (line 749) | Entry point we wrap in our outer training script. Outer script: for each outer-episode, sample a city, run year-loop calling `setup_and_train`-equivalents inside, apply dynamics between years. |

### REPLACE (write fresh in our codebase)

| New module | Purpose | Lift surface it talks to |
|---|---|---|
| `learning/city_builder/multi_year_mdp.py` (M2 / TOP-8) | Outer MDP. State: `CityBuilderState = (CityGraphData, activity_per_zone, year, budget_remaining)`. Action: annual edge-allocation (or full inner-episode of route generation). Reward: `W(s_{t+1}) − W(s_t)` via `MyCostModule.forward`. Horizon: 20 years. | Calls `MyCostModule`, constructs `RouteGenBatchState`, reads `CityGraphData`. |
| `learning/city_builder/land_use_dynamics.py` (M2 / TOP-9) | `LandUseDynamics.step(activity_t, accessibility_t, α, base_rate, σ_ε, cap) → activity_{t+1}`. After each year, re-materializes `data.demand` via gravity model on new activity. | Mutates `CityGraphData.demand` and `CityGraphData[DEMAND_KEY].edge_attr` in place; cost module re-reads them. |
| `learning/city_builder/budget_action_space.py` (M4 / TOP-17) | Per-edge capex lookup; per-year construction-budget constraint; masks infeasible actions in the inner inductive policy's action distribution. | Modal-parameter table; `RouteGenBatchState.shortest_path_action` (line 202) is the action primitive we constrain. |
| `learning/city_builder/calibration/fit_dynamics.py` (M4 / TOP-18) | Differential evolution over `{α, base_rate, σ_ε, cap}` to minimize APE against DANE 2018 population. Inner: forward-simulate TransMilenio 2000–2020 rollout under fixed actions. | Reads Bogotá `CityGraphData`, calls `LandUseDynamics.step` repeatedly. |
| `simulation/bogota_adapter.py` (M4 / TOP-17) | Bogotá GIS → `CityGraphData`. Mirrors `from_mumford_data` output shape: `STOP_KEY`/`STREET_KEY`/`DEMAND_KEY` nodes and edges, `drive_times`, `demand`. | Outputs schema-compatible `CityGraphData`. |
| `learning/city_builder/outer_ppo.py` (M3) | Outer PPO wrapper that treats one full city-evolution trajectory as one PPO episode. Inner PPO (`train_ppo`) runs per-year for the inner inductive policy; outer accumulates per-year welfare deltas as the trajectory return. | Calls `train_ppo` per year, accumulates returns across years. |

## Critical seam properties

1. **Reward is the welfare delta.** Per proposal §3: `r_t = W(s_{t+1}) − W(s_t)`. Both `W`'s come from `MyCostModule.forward(state).cost` (or its negation, depending on sign convention). No re-implementation of welfare is needed.

2. **Demand recomputation hook.** After `LandUseDynamics.step` updates `activity`, we recompute the gravity model on the new activity and overwrite `data.demand` + `data[DEMAND_KEY].edge_attr`. The cost module re-reads these on the next call. There is no implicit caching to invalidate (verified by reading `_cost_helper` line 867-868: it pulls `state.drive_times` and `state.demand` fresh each call).

3. **`drive_times` is a function of the network only.** When the agent builds a new metro edge, `drive_times` changes. This is recomputed via `torch_utils.floyd_warshall` in `from_mumford_data` line 526 — the same primitive we'll call after each year's edge additions in the outer MDP.

4. **α=0 is the static reproduction case.** Setting `α=0` in `LandUseDynamics` makes `activity_{t+1} = activity_t + ε` (or `= activity_t` if we also disable noise). Demand stops evolving. The cost across one inner episode then equals exactly what Holliday's original training run computes. This is the proposal's credibility-anchor sanity check.

## What changed vs. the pre-pivot M1 plan

Two concrete corrections to the 19-day-old project memory:

1. **`learning/inductive_route_learning.py` was deleted in the working tree** during the stack upgrade (`git status` showed `deleted:` but uncommitted) and has been restored from HEAD. README's reference to `inductive_route_training.py` is documentation drift — the real file is `inductive_route_learning.py` (829 lines, includes `train_ppo`, `NNBaseline`, `setup_and_train`, hydra entry `@hydra.main(config_name="ppo_20nodes")`).

2. **The seam is cleaner than the original memory described.** Old memory said "write a new MDP that calls one function from Holliday's repo." Verified by reading the code: the function is `MyCostModule.forward(state)`, and the state object it consumes (`RouteGenBatchState`) already carries the only attributes the cost module reads (`drive_times`, `demand`, `routes`, `n_transfers`, `transit_times`). The outer MDP doesn't need to construct anything novel — it materializes one of these per year.

## Action items unblocked by this note

- **TOP-5 reproduction:** entry point is `python learning/eval_route_generator.py +eval=mandl +model.weights=<weights>.pt eval.dataset.path=<mumford>/Instances`. Two paths to weights: (a) download Holliday's PPO 2025 pre-trained weights from `https://www.cim.mcgill.ca/~mrl/projs/transit_learning/ppo_2025`, or (b) train fresh via `learning/inductive_route_learning.py` (3–6h on GPU per README).
- **TOP-8 MDP formalization:** the formal MDP can now be written as a wrapper around `RouteGenBatchState` rather than a redefinition of it.
- **TOP-9 LandUseDynamics module:** the demand-recomputation hook is `data.demand = gravity_model(activity_t+1); data[DEMAND_KEY].edge_attr = stack(demand_feat, drive_time_feat)`. No deeper integration work needed.
