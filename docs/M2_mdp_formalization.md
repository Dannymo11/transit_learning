# M2 MDP Formalization — City Builder Multi-Year MDP

**Linear ticket:** [TOP-8](https://linear.app/topograph-stanford/issue/TOP-8/m2-formalize-the-new-mdp-state-action-transition-reward-horizon)
**Companion:** [`docs/M1_architecture_mapping.md`](M1_architecture_mapping.md) — pins the lift surface this MDP wraps
**Status:** draft for writeup §3 (Method), pre-implementation
**Date:** 2026-05-21

## Notation note

Two greek letters appear in adjacent sections of this project and **must not be conflated**:

- **α** (this document, proposal §3) = *induced-demand strength* in `LandUseDynamics.step`. Calibrated to Bogotá TransMilenio in M4 (TOP-18), ablated in M5 (TOP-22).
- **w_p** = *passenger weight* in `MyCostModule` (Holliday's α from Table 2 of arXiv:2404.05894). Operator-vs-passenger cost-weight tradeoff; varied in M5 Exp 3 (TOP-20) for the ridership-max vs accessibility-weighted comparison.

Throughout this document, "α" means induced-demand strength only.

## 1. State

The state at year t ∈ {0, …, T} is the tuple

> `s_t = (G_t, x_t, B_t)`

where

- **G_t = (V, E_street, E_built^t)** — the *city graph*. V is the fixed set of N zones (nodes). E_street is the static street network (fixed across t — represents physical road infrastructure). E_built^t ⊆ E_candidate is the set of transit edges built by year t, drawn from a fixed candidate-edge set E_candidate (real rail right-of-way + planned Line 1 + Line 2 alternatives + a configurable set of grid alignments per M4 / TOP-16). E_built^0 = ∅ at t=0 (for the Bogotá Metro counterfactual; can be initialized non-empty for warm-starts).
- **x_t ∈ R_+^N** — per-zone activity (combined population + employment, single quantity). Initialized to x_0 from M4-calibrated Bogotá 2020 estimates (TOP-14). Evolves under `LandUseDynamics` (§4).
- **B_t ∈ R_+** — remaining annual construction budget for the year-t decision. Reset to a fixed annual budget `B_annual` at the start of each year (default ≈ 700M USD/yr per M4 / TOP-17, calibrated to actual Line 1 build pace). Unspent budget does *not* roll forward in the base formulation (a rollover variant is a stretch ablation in M6 / TOP-26).

Concretely in code: `CityBuilderState` (forthcoming `learning/city_builder/multi_year_mdp.py`) owns `(CityGraphData, activity: Tensor[N], year: int, B_remaining: float)` and materializes a per-year `RouteGenBatchState` (Holliday's inductive state, kept intact per [M1 architecture mapping](M1_architecture_mapping.md)) for the inner route-construction episode.

## 2. Action

**Decision:** at each year t, one action a_t is **a complete new transit route** (a sequence of nodes drawn from E_candidate) added to E_built, subject to the year's capex budget.

Formally: a_t ∈ {*build-route*(π) : π is a valid path in E_candidate with capex(π) ≤ B_t} ∪ {*no-build*}.

**Rejected alternatives** (the three options TOP-8 enumerated):

- **One edge per year:** 20 outer decisions × 1 edge = 20 edges; insufficient to produce a metro-scale corridor. Rejected.
- **Budget allocation across candidate edges (continuous):** would require redesigning the policy head from scratch and gives up Holliday's `PathCombiningRouteGenerator` entirely. Rejected — breaks the proposal's "GNN + PPO largely unchanged" commitment.
- **Complete new route per year (chosen):** action is materialized by running Holliday's existing inner inductive route-construction policy *constrained* to a one-route episode with `B_t / capex_per_km` as the implicit length cap. Reuses `PathCombiningRouteGenerator.forward(RouteGenBatchState, n_routes_to_plan=1)` (`learning/models.py` line 1192) and `RouteGenBatchState.shortest_path_action` (`transit_time_estimator.py` line 202) unchanged. Budget enforcement is implemented in `learning/city_builder/budget_action_space.py` (M4 / TOP-17) as a mask on the inner policy's edge-selection logits — edges that would push capex over `B_t` are set to −∞.

The *no-build* action lets the agent save budget if no positive-welfare-delta route fits — implemented as an "early halt" signal from the inner policy (Holliday's halt_scorer already supports per-route halting; we let it halt on the first step to mean "no route this year").

**Why this granularity matches the planning problem.** Bogotá Metro Line 1 was authorized as a series of corridor decisions: the broader Line 1 corridor in 2014, the elevated/underground mix in 2017, the construction start in 2020, Line 2 alignment alternatives currently under debate. Each corresponds to a discrete per-year corridor-or-segment decision, not a continuous edge-by-edge accumulation.

## 3. Reward

**Decision:** per-step welfare delta.

> `r_t = W(s_{t+1}) − W(s_t)`

where W(·) is computed by `MyCostModule.forward(state).cost` on the year-t and year-(t+1) materialized `RouteGenBatchState`. Sign convention: `MyCostModule` returns *cost* (lower = better welfare); concretely `r_t = −[C(s_{t+1}) − C(s_t)]` so that maximizing return maximizes cumulative welfare reduction over the horizon.

The episode return is `sum_{t=0}^{T-1} γ^t r_t = γ^0 (W(s_1) − W(s_0)) + γ^1 (W(s_2) − W(s_1)) + …`. With γ=1 this is exactly the total welfare gain over the horizon (telescopes to `W(s_T) − W(s_0)`). We use γ = 0.95 (Holliday's default in `cfg/ppo_mumford3.yaml`) to mildly discount distant-future welfare; this also stabilizes value-function learning under the stochastic dynamics.

**Rejected alternative — terminal cumulative welfare only.** Sparse reward harms credit assignment under stochastic dynamics where the variance over a 20-year rollout is large. Per-step delta gives the value function a denser signal. M6 ablation (TOP-24) compares per-step vs terminal-only as the reward-shaping sensitivity test, so we keep both implementable but default to per-step.

**The constraint-violation penalty** in `MyCostModule.forward` (line 1095, `const_viol_cost * constraint_weight`) is inherited unchanged: it penalizes unserved demand and stops-out-of-bounds. In the multi-year MDP this means a route that disconnects the network at year t incurs a large negative r_t — the agent learns to maintain connectivity year-over-year.

## 4. Transition

> `s_{t+1} = (G_{t+1}, x_{t+1}, B_{t+1})`

where:

**G_{t+1}**: apply action a_t to G_t.
- If a_t = build-route(π): E_built^{t+1} = E_built^t ∪ π; the street_adj is updated to reflect the new transit edge (lower travel time on segments served by the new route, per the modal-parameter table from M4 / TOP-16); `data.drive_times` is recomputed via `torch_utils.floyd_warshall` (same primitive as `from_mumford_data` line 526). Capex(π) is debited from B_t (consumed in this year, doesn't carry over).
- If a_t = no-build: E_built^{t+1} = E_built^t; drive_times unchanged.

**x_{t+1}**: apply `LandUseDynamics.step` (M2 / TOP-9):

> `x_{t+1,i} = clip(x_{t,i} · (1 + α · Ã_{t,i}) + ε_{t,i}, 0, cap_i)`

where Ã_{t,i} is the normalized accessibility of zone i under G_{t+1} (recomputed from `drive_times` after the network update), α ∈ R_+ is the induced-demand strength (calibrated in M4 / TOP-18), ε_{t,i} ~ N(0, σ_ε^2) is iid Gaussian noise per zone-year, base_rate ∈ R_+ shifts the no-accessibility growth rate (folded into the +1 term), and cap_i is the per-zone activity cap.

After x_{t+1} is computed, the OD demand matrix is re-evaluated via a gravity model `D(x_{t+1}, drive_times_{t+1})` and written back to `CityGraphData.demand` + `data[DEMAND_KEY].edge_attr`. The cost module reads these fresh on the next call (verified in the architecture mapping — no implicit caching).

**B_{t+1}**: reset to `B_annual` at the start of each year (base formulation; rollover is an M6 ablation).

**Stochasticity** is *purely* in the dynamics ε_{t,i}. The agent's action is deterministic conditional on its policy; the inner GNN's sampling-during-rollout (LC-100 style) is a *separate* mechanism for the inner inductive episode and is held fixed for outer-MDP training.

## 5. Horizon

**T = 20 years** for the headline Bogotá Metro Line 1 counterfactual (M5 Exp 4 / TOP-21), matching the planned Line 1 commercial-start to mature-network timeline (construction since 2020 → 2028 commercial start → ~2040 mature).

**T = 10 years** for the M2 α-sensitivity sweep (TOP-10) and M3 dynamic-training experiments (TOP-11, TOP-12) — shorter to enable cheaper iteration during method development. Holliday's static M3 reproduction (TOP-11) runs with T = 1 (single inner episode, no outer dynamics) for the α=0 sanity check.

**Agent decisions per episode** = T (one decision per year, plus the implicit per-step decisions inside the inner inductive route-construction).

## 6. Observation

The agent's observation o_t is everything in s_t that the inner GNN policy needs to choose the year-t route. Concretely:

> `o_t = (G_t structure, drive_times_t, demand_t, x_t, A_t, B_t / B_annual, t / T)`

where the first four are already present in `CityGraphData` and consumed by `PathCombiningRouteGenerator` via `RouteGenBatchState`. The new fields (`x_t`, `A_t`, `B_t/B_annual`, `t/T`) are added as *per-node features* concatenated onto `STOP_KEY.x`:

- **x_t** (per-zone activity, normalized to [0,1] by per-episode max)
- **A_t** (per-zone accessibility under current network)
- **B_t / B_annual** (scalar, broadcast to all nodes — remaining budget fraction)
- **t / T** (scalar, broadcast to all nodes — episode progress)

This changes `in_node_dim` in `cfg/model/bestsofar_feb2023.yaml` from **4 → 8**. Concrete consequence: pretrained PPO 2025 weights (verified in TOP-5 to reproduce Holliday's Mandl numbers) can no longer be loaded directly into the multi-year-MDP policy, because the first-layer input projection has the wrong width. Two options:

- **Train from scratch** for the multi-year policy. Costs ~3–6h on GPU per the README.
- **Warm-start by padding the input projection**: copy the first 4 input weights from Holliday's pretrained policy, initialize the 4 new feature weights to zero. The policy starts with effectively the same behavior as Holliday's and learns the new features via fine-tuning. **Recommended** — captures Holliday's training as a useful prior.

The α=0 sanity check in TOP-11 uses Holliday's original 4-feature architecture (no LandUseDynamics, no outer MDP, single inner episode) — exact Holliday reproduction. The dynamic experiments in TOP-12 onward use the 8-feature multi-year policy.

## 7. Episode termination

Terminate at t = T (fixed horizon, no early termination). The constraint-violation penalty in r_t handles degenerate states (disconnected networks) inside the reward rather than via early termination, so trajectories always have length T for clean batching.

## 8. The five things this MDP commits to that subsequent milestones depend on

1. **Welfare oracle is `MyCostModule.forward(materialize(s_t)).cost`** — verified in TOP-5 to match Holliday's published numerics within 1.1σ. No re-implementation needed for M2–M5.
2. **Action is one complete route per year** — lets the inner inductive policy be reused unchanged.
3. **Reward is per-step welfare delta with γ=0.95** — proposal-compliant; ablated against terminal-only in M6 (TOP-24).
4. **Activity feeds the policy as node features** — requires `in_node_dim=8` and warm-start from Holliday's pretrained weights (recommended) or train-from-scratch.
5. **Budget enforcement is a logit mask in `budget_action_space.py`** — keeps the policy architecture untouched.

## 9. Open questions deferred to implementation

- **Exact gravity-model form** for `D(x, drive_times)` — singly-constrained vs doubly-constrained, deterrence function. Defer to M4 (TOP-15) where we calibrate against Encuesta de Movilidad OD data; for M2 α-tuning we use a placeholder doubly-constrained form `D_{ij} = x_i x_j / drive_time_{ij}^β` with β=2.
- **Whether to apply dynamics *before* or *after* the year's action** — current formulation (action first, then dynamics) means the agent sees t-period activity when deciding t-period action; alternative is to apply dynamics first so the action responds to t+1 demand. The current ordering matches typical real-world planning (decisions are made based on present conditions, with infrastructure operationalized and dynamics observable in the following year).
- **Multiple new routes per year if budget allows** — base formulation is one route/year. M6 ablation could compare with multi-route years if computational budget allows; not in M5 critical path.

## 10. References

- Proposal §3 (state/transition/reward equations this document operationalizes)
- [`docs/M1_architecture_mapping.md`](M1_architecture_mapping.md) — lift surface for `MyCostModule`, `RouteGenBatchState`, `PathCombiningRouteGenerator`
- arXiv:2404.05894 (Holliday et al. 2025) §3 for the inner inductive route-construction MDP this wraps
- Paulsen & Rich 2024 §3 for the closest non-RL analog (MIP formulation of sequential network expansion with induced demand)
