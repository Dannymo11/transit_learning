# M3 / TOP-12 — RL training design: policy + PPO under induced demand

**Linear:** [TOP-12](https://linear.app/topograph-stanford/issue/TOP-12/m3-train-policy-under-chosen-dynamic-a)
**Status:** design (pre-implementation). Approach chosen 2026-05-31: *fork Holliday's `train_ppo`, splice the closed-loop land-use dynamics into the rollout*. This doc works out the data flow, the subtleties, the risks, and a staged plan before any trainer code is written.

---

## 1. One-paragraph approach

Holliday's `learning/inductive_route_learning.py::train_ppo` is already a PPO loop that (a) rolls out a sequence of route-construction steps, (b) rewards each step by the **decrease in `MyCostModule` cost** (`diff_reward`), and (c) updates a GNN policy with a clipped PPO objective and an `NNBaseline` critic + GAE. Our multi-year MDP is *that same loop* with one insertion: **after each transit route is completed, advance the city by one year** — run `step_world` so per-zone activity grows under the route network's transit accessibility and the demand the policy sees next is the induced demand. We therefore fork `train_ppo` into a City Builder trainer and reuse the PPO machinery (critic, GAE, clip loss, minibatching) byte-for-byte; the only new logic is detecting route completion and calling the (already-built, already-tested) dynamics.

---

## 2. What `train_ppo` already gives us (with line refs)

Rollout loop (`inductive_route_learning.py` ~397–432), per step `tt`:
- `val_module.from_state(state)` → critic value; buffered.
- `actions, logits, _ = model.step(state)` → GNN picks an action; `state.shortest_path_action(actions)` applies it.
- `diff_reward`: `result = cost_obj(state); rewards = (prev_cost - result.cost) * return_scale; prev_cost = result.cost` (welfare delta).
- Buffers: `buf_rewards`, `buf_val_ests`, `buf_logits`, `buf_actions`, `buf_states`, `done_at_step_mask`.

Then GAE (~433–470, `gamma`, `gae_lambda`), and a clipped PPO update over minibatches of stored states (~472+), with `NNBaseline` as the critic (`val_module.update(returns)`).

**Reused unchanged:** the GNN policy (`PathCombiningRouteGenerator`), `model.step`, `state.shortest_path_action`, `NNBaseline`, GAE, the clip loss, minibatching. **New:** the dynamics insertion + activity tracking + a City Builder config.

---

## 3. The key subtlety — action granularity

`model.step` + `shortest_path_action` is **not** "one whole route per step." Reading `shortest_path_action` (`transit_time_estimator.py:202`): an action is a terminal pair that either **starts** a new route, **extends** the current route at one end (by the shortest path to the chosen node), or **halts** it (`path_indices[:,0] == -1`). A *route* is built over several steps and is finished when the policy halts it. `is_done()` (`:496`) is true when `n_routes_left_to_plan == 0`, i.e. after `n_routes_to_plan` routes are complete. `cfg.ppo.horizon` (200 for Mandl) is just a generous cap on total steps; episodes finish when all routes are done.

**Consequence for us:** a **year = one completed route**, which spans multiple `model.step` calls. The land-use dynamics must fire **on route completion**, not every step.

---

## 4. Mapping to the multi-year MDP

- Set `n_routes_to_plan = T` (years; e.g. 10). One completed route per year.
- Keep `horizon` a generous step cap (as Holliday).
- Maintain per-batch-element activity `x` alongside the state.
- **Year boundary = route completion.** Detect when `state.n_finished_routes` increments (equivalently, a halt action). On that event for a given episode:
  1. compute the network's transit OD times (`state.transit_times`),
  2. `x ← LandUseDynamics.step(x, transit_times)` and `recompute_demand_in_place(state.graph_data, x, street_times)` — i.e. `step_world(..., accessibility_drive_times=transit_times)`,
  3. so the next route is constructed against the **evolved** demand.

This reuses the exact closed-loop wiring already validated in `CityBuilderEnv` / `alpha_sweep` (transit accessibility for growth, street times for gravity demand).

---

## 5. Reward attribution

Keep Holliday's per-step `diff_reward = prev_cost - cost`. Two kinds of cost change now occur:
- **From building** (route extensions) — rewards good routes, exactly as today.
- **From dynamics** (city grows after a route completes) — the cost shifts (more population on the network → usually higher average travel time → a *negative* nudge; better-served growth → positive). By letting `prev_cost` carry across the dynamics update, this shift appears as the reward at the post-completion step, so the agent is credited/penalized for the *induced* consequences of what it built.

With `gamma = 0.95` (Holliday default) the discounted sum approximates total welfare gain over the horizon; the agent is thus optimizing long-horizon welfare **including** induced demand — which is the whole point, and the lever the myopic greedy baseline lacks.

*Open choice:* whether the dynamics-step cost delta should be a reward at all, or netted out (reward only building, treat dynamics as uncontrollable). Recommendation: **keep it** — the agent *can* influence where growth lands via which corridors it builds, so it should be on the hook for it. Revisit if training is unstable.

---

## 6. Demand → GNN propagation (verified OK)

`get_node_features` (`citygraph_dataset.py:559`) encodes only street in/out degree + position — **demand-independent**. Demand enters the GNN through **edge** features, which `model.step` recomputes each step via `_get_edge_features(state)` from the live graph. `recompute_demand_in_place` updates `data.demand` + the `DEMAND_KEY` edge_attr. So mutating demand between routes propagates to the policy with no stale-feature problem.
*To confirm in implementation:* that `_get_edge_features` reads current `data.demand` (or the refreshed `DEMAND_KEY` edge_attr) rather than a cached copy.

---

## 7. Activity tracking & batched dynamics

Holliday batches `batch_size` rollouts (Mandl `ppo_mandl.yaml`: 16). Each element needs its **own** activity `x_b` and its own evolving demand in the batched `graph_data`. The dynamics helpers (`step_world`, `transit_drive_times`) are currently single-instance.

**Staging decision:** implement **`batch_size = 1` first** (one Mandl rollout per PPO iteration). This makes the dynamics insertion trivial and gives a validatable first run; throughput is lower but Mandl is tiny. Then generalize the dynamics to operate per batch element (loop or vectorize over the batch) for the full-throughput run.

---

## 8. Partial observability note

The policy observes **demand** (a function of activity `x`), not `x` directly. That's acceptable — demand is the decision-relevant signal and is what the cost module serves — but it means the MDP is technically partially observed in `x`. Flagging so it's a conscious choice, not an oversight. If needed later, per-zone activity could be added as a node feature.

---

## 9. Cost-normalization caveat (fair comparison)

`MyCostModule` normalizes cost by a diameter term scaled by `n_routes_to_plan` (`:520`). With `n_routes_to_plan = T = 10` the normalization differs from the baselines' setup. For the RL-vs-greedy/random comparison (M5 Exp 2/5) and the welfare numbers to be comparable, the **evaluation must use a consistent cost configuration across RL and baselines** (same `n_routes_to_plan`, same cost weights). The `CityBuilderEnv` baselines should be re-run with the same `n_routes_to_plan = T` used in training, or welfare reported on a fixed common evaluator. Do not compare raw cost across different `n_routes_to_plan`.

---

## 10. Staged implementation plan

- **Stage 0 — batch=1 trainer fork + dynamics injection.** New module `learning/city_builder/train_rl.py` (fork of `train_ppo`), `cfg/ppo_citybuilder_mandl.yaml` (`n_routes=T=10`, induced `alpha=0.5`, `gamma=0.95`, Holliday hyperparams). Route-completion → `step_world`. Short Modal smoke (~20–50 iters).
- **Stage 1 — batched dynamics.** Generalize activity/demand evolution to all batch elements; restore `batch_size`>1 for throughput.
- **Stage 2 — eval + baseline comparison.** Roll out the trained policy through `CityBuilderEnv` (or the same rollout with `greedy=True`); report cumulative welfare vs greedy (11.0) and random (11.34) under a consistent cost config. This is the M5 Exp 2 core and feeds the Exp 5 α-ablation.

---

## 11. Risks & open questions

1. **Route-completion detection inside the rollout.** Need a clean signal that a route finished for each episode this step (track `state.n_finished_routes` deltas, or read the halt from `actions`). Verify it's per-element correct.
2. **Reward variance / stability** under stochastic dynamics. Start `sigma_eps = 0` (deterministic dynamics); add noise only once learning is stable.
3. **Batched dynamics correctness** — the main Stage-1 hazard; isolate by validating Stage 0 single-instance first.
4. **Learning signal.** From the baselines we expect the gap to be real (greedy under-builds, random over-builds). Quantitative gate (TOP-12): RL beats greedy on cumulative welfare across seeds, paired p<0.05. If RL ≈ greedy, lean on the M3 qualitative gate (different networks) + the absolute-vs-normalized-metric risk already logged.
5. **Compute.** RL training is the expensive job → GPU on Modal (`MODAL_GPU=1`), unlike the cheap Mandl baselines.

---

## 12. Validation plan (first Modal smoke)

- Episodes run T years; `n_finished_routes` reaches T.
- `step_world` fires exactly once per completed route (log per-year `sum(x)`, `frac_at_cap`).
- Reward telescopes sensibly (cumulative ≈ initial_cost − final_cost minus dynamics jumps).
- `val cost` / avg-return curves trend the right way over iterations.
- Sanity: a `greedy=True` rollout of the *untrained* policy shouldn't crash and should produce a full network.
- Then Stage 2: trained policy's cumulative welfare vs greedy/random.

---

## 13. What lands where

- `learning/city_builder/train_rl.py` — forked PPO trainer with dynamics injection (new).
- `cfg/ppo_citybuilder_mandl.yaml` — training config (new).
- `modal_runs/` — a `train_rl` entrypoint (mirror the existing runner; GPU).
- Reused as-is: `multi_year_mdp.CityBuilderEnv` (baselines + eval), `demand_hook.step_world`, `alpha_sweep.transit_drive_times`, `LandUseDynamics`, Holliday's policy + `NNBaseline` + PPO update.
