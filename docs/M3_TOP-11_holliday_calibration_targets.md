# TOP-11 calibration targets — Holliday static reproduction

**Linear ticket:** [TOP-11](https://linear.app/topograph-stanford/issue/TOP-11/m3-train-policy-under-static-a0-holliday-reproduction)
**Milestone:** M3 (gate: "trained policy on the *static* problem matches or beats Holliday's published numbers within seed variance").
**Paper:** Holliday & Dudek, *A Neural-Evolutionary Algorithm for Autonomous Transit Network Design*, ICRA 2024.

This document pins the numeric targets we are calibrating against. If we don't write them down before the run, "matches within seed variance" is unfalsifiable.

## Notation: which α is which

Two completely different things are both called "α" in this project. Read carefully:

- **Holliday's α (= Holliday & Dudek 2024 paper notation):** the **cost-weight tradeoff** between passenger and operator cost. In our code this is `demand_time_weight` / `w_p`. The paper's Table 2 sweeps α ∈ {0.0, 0.5, 1.0}, where α=1 is passenger-only and α=0 is operator-only.
- **Our induced-demand α (M2 / TOP-9):** the **strength of induced demand** in `LandUseDynamics`. **Throughout TOP-11 our induced-demand α is fixed at 0** — the entire calibration milestone is conducted with no LandUseDynamics at all.

When this document says α, it always means Holliday's α (cost-weight) unless explicitly noted. See [`feedback_alpha_naming.md`](../../memory) for the naming convention rationale.

## Metric

The headline cost metric in the paper is **C(α, G, R)** — a normalized scalar combining passenger and operator cost. Lower is better. Holliday's Table 2 reports it across α ∈ {0.0, 0.5, 1.0} for LC-100, EA, and NEA on each instance.

**LC-100 is exactly the method our trainer produces.** Confirmed by reading `cfg/experiment/cost_function/mine.yaml`:

```yaml
demand_time_weight: 0.5
route_time_weight: 0.5
variable_weights: true   # weights vary during training
pp_fraction: 0.33        # 33% of rollouts at passenger-pure (α=1)
op_fraction: 0.33        # 33% at operator-pure (α=0); remaining 34% at α=0.5
```

So `learning/inductive_route_learning.py` trains **one** policy that handles all three α values; LC-100 in the paper = this policy sampled 100x at each α and the best route set kept.

**Secondary metrics** logged to W&B but not part of the pass/fail gate:

- **C_p** (avg passenger trip time, minutes — what `MyCostModule.mean_demand_time / 60` reports at line 820)
- **C_o** (total route time / operator cost, minutes)
- **d_0 / d_1 / d_2 / d_un** (fraction of trips with 0, 1, 2, or ≥3 transfers; `d_un = 0` is required, anything higher means the policy is dropping demand)

Table 3 in the paper reports these for NEA and RC-EA but **not for LC-100**, so we don't have a paper number to gate them on. They're sanity checks, not the gate.

## Instance / route parameters (Mumford-1981 benchmark)

Verified against Holliday & Dudek 2024, Table 1 ("Statistics of the Mandl and Mumford benchmark cities"):

| Instance | Nodes (n) | Link edges (|E_s|) | `n_routes` (S) | `min_route_len` (MIN) | `max_route_len` (MAX) | Area (km²) | Training config |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mandl    |  15 |  20 |  6 |  2 |  8 |  352.7 | `cfg/ppo_mandl.yaml` |
| Mumford0 |  30 |  90 | 12 |  2 | 15 |  354.2 | (TODO — clone `ppo_mumford3.yaml`, set `dataset.city=Mumford0`, route params) |
| Mumford1 |  70 | 210 | 15 | 10 | 30 |  858.5 | (TODO) |
| Mumford2 | 110 | 385 | 56 | 10 | 22 | 1394.3 | (TODO) |
| Mumford3 | 127 | 425 | 60 | 12 | 25 | 1703.2 | `cfg/ppo_mumford3.yaml` |

Mandl is the first calibration target per the milestone plan (Mandl first → Mumford after Mandl passes).

## LC-100 calibration targets (Holliday & Dudek 2024, Table 2)

`C(α, G, R)`, mean ± std over 10 random seeds. Lower is better. **Each cell is the paper's reported number for LC-100** at the indicated α (cost-weight). Our trained policy, evaluated 100x at each α and best-of selected, should land at or below these on a per-α basis.

| Instance | α = 0.0 (operator-only) | α = 0.5 (balanced) | α = 1.0 (passenger-only) |
| --- | ---: | ---: | ---: |
| Mandl    | 0.697 ± 0.011 | 0.558 ± 0.003 | 0.328 ± 0.001 |
| Mumford0 | 0.847 ± 0.025 | 0.916 ± 0.006 | 0.721 ± 0.004 |
| Mumford1 | 1.747 ± 0.034 | 1.272 ± 0.018 | 0.573 ± 0.004 |
| Mumford2 | 1.315 ± 0.049 | 0.989 ± 0.021 | 0.495 ± 0.002 |
| Mumford3 | 1.333 ± 0.064 | 0.984 ± 0.026 | 0.476 ± 0.001 |

For context (not the gate), the same table reports stronger baselines that we are **not** targeting in TOP-11:

- **EA** (evolutionary algorithm without learning) — beats LC-100 at α=0 on most instances.
- **NEA** (Neural Evolutionary Algorithm — LC-100 + EA refinement) — Holliday's headline method, beats both on most cells.

Reaching NEA-quality numbers would require adding the EA refinement step, which is outside TOP-11's scope. TOP-11 is purely the neural part.

### Secondary reference: Table 3 passenger-perspective numbers (NEA only, not LC-100)

These are the C_p / d_0 / d_un values for NEA on each instance. **Do not gate on these** — Table 3 does not break out LC-100 separately. They're here as a sanity check on the magnitude of `C_p` (avg passenger trip time, minutes).

| Instance | NEA C_p (min) | NEA d_0 (%) | NEA d_un (%) |
| --- | ---: | ---: | ---: |
| Mandl    | 10.37 | 93.89 | 0 |
| Mumford0 | 15.26 | 68.35 | 0 |
| Mumford1 | 22.85 | 49.28 | 0 |
| Mumford2 | 25.25 | 52.66 | 0 |
| Mumford3 | 27.96 | 49.36 | 0 |

### Reference observation that needs explaining

Running `learning/eval_route_generator.py` on Mandl with the user's existing trained weights produces ATT values in the **13.14–13.49 min** range across seeds (10 rows in `mandl_all_seeds.csv`, column index 3). This is **~3 minutes worse than the NEA paper number (10.37 min)** and noticeably higher than what a working LC-100 should produce (paper text says LC-100 gets within a few seconds of NEA on Mandl).

Two plausible explanations:

1. The eval was done at a different α than the paper's (e.g., α=0 or α=0.5 instead of α=1) — different cost weight, different trained policy behavior, different ATT.
2. The eval was using already-trained weights that were trained with a different cost-function config than the paper's.

Either way: **if the new calibration run produces Mandl ATT around 10 min at α=1 (passenger-only) and 13–14 min at α=0 (operator-only), that's the expected behavior** and the 13.x ATT in the user's CSV is just the operator-leaning slice of the variable-weights policy.

## Pass/fail rule

For each instance we run 5+ seeds via `modal_runs/train_static_holliday.py::sweep`. Evaluate the trained policy 100x at each α ∈ {0.0, 0.5, 1.0} (matching LC-100's "best of 100 samples" methodology). Let `μ_ours[α]` and `σ_ours[α]` be the mean and std across seeds of the best-of-100 cost at that α.

**Per-α gate:** `μ_ours[α] ≤ C_paper[α] + max(σ_paper[α], σ_ours[α])` — we are *within seed variance* of (or below) the LC-100 paper number at that α.

**Per-instance pass:** the gate holds at α=0.5 (the balanced cost — the most diagnostic for whether training learned a usable policy at all). Failing at α=0 or α=1 alone is investigable but not blocking — those are the extremes of the cost-weight, and one of them being slightly off can reflect optimizer noise rather than broken integration.

**Per-instance fail:** α=0.5 misses the LC-100 number by more than 1σ. Per the M3 milestone, "fix before moving on" — do not start TOP-12 (dynamic-α training) under a broken static surface.

The decision gate applies per-instance. If Mandl passes but Mumford3 fails, that's an instance-specific issue (likely hyperparam, especially LR — Mumford3 is much harder) and Mumford3 needs to be fixed before flagging the milestone complete.

The paper reports σ across 10 seeds. We default to 5 seeds for the calibration run, which makes our σ wider on small samples; that's why `max(σ_paper, σ_ours)` is in the rule (use the more permissive of the two). Bump to 10 seeds if any cell is close to the boundary.

## What the W&B sweep should log

`modal_runs/train_static_holliday.py` patches `wandb.tensorboard` so every `SummaryWriter.add_scalar` in `learning/utils.py:154` auto-syncs. Key scalars to expect on the live W&B run:

- `val cost` — the optimization objective on the held-out eval. The single most important convergence curve.
- `# stops per route` — sanity check (should stabilize, not explode).
- `baseline` — the rolling baseline ATT; should converge to a reasonable floor.

Per the calibration-gate plan, the runs should be considered converged once `val cost` plateaus for ~50 iterations. If iter 500 hasn't plateaued, extend `ppo.n_iterations` and rerun.

## Open questions to resolve before running

1. **W&B project name.** Default in the Modal runner is `cs224r-city-builder`. Confirm or override via `--wandb-project=<name>`.
2. **Number of seeds.** Holliday reports across 10 seeds. We default to 5 (M3 minimum); bump to 10 if Modal credits allow — it tightens the variance estimate and the pass band.
3. **Whether to run Mandl alone first or fan out immediately.** Recommended: Mandl-only smoke (1 seed, ~50 iters) → Mandl 5-seed sweep → Mumford3 5-seed sweep → Mumford0/1/2 if budget permits.
4. **How to evaluate the trained policy at three α values.** Holliday's `learning/eval_route_generator.py` accepts a `cost_weights` argument (passenger/operator). The Modal runner should invoke eval three times per trained seed — once at (α_p=1, α_o=0), once at (0.5, 0.5), once at (0, 1) — and log the resulting C(α, G, R) under separate W&B metric names (`eval_cost_alpha_0`, `eval_cost_alpha_0p5`, `eval_cost_alpha_1`). This is a follow-up to add to the runner before the gate evaluation; the training run itself doesn't need it.

Once the Modal/W&B secret is set and the W&B project is confirmed, kick off with:

```bash
modal run modal_runs/train_static_holliday.py::sweep \
    --config-name=ppo_mandl --seeds=0,1,2,3,4
```
