# City Builder: Transit Network Design under Induced Demand

**CS 224R course project (Spring 2026).** This project asks whether a
reinforcement-learning transit planner that *anticipates induced demand* beats a
myopic greedy replanner when the city it serves grows in response to the network
it builds.

I wrap Andrew Holliday's GNN + PPO route-design agent in a multi-year
**city-builder MDP**: each year the agent (re)builds a transit network, and the
city's per-zone activity then evolves via a land-use update driven by the
**transit accessibility** the network provides. Gravity demand is recomputed
from the evolved activity, closing the loop between what the agent builds and the
demand it will face next year. The induced-demand strength is a single knob
`alpha`; at `alpha = 0` the loop is inert and the setting collapses back to
Holliday's static-demand benchmark (our core sanity check).

The headline result comes from the **build-then-watch** regime: the agent commits
a network up front and then watches the city grow around it (vs. replanning every
year). 

> **This repository is a fork of Andrew Holliday's `transit_learning`**
> ([McGill Mobile Robotics Lab](https://www.cim.mcgill.ca/~mrl/projs/transit_learning/)),
> released under the GNU GPL. The inner route-design
> machinery — the GNN policy, the PPO training loop, the cost module, the
> city-graph schema, and the Mandl/Mumford dataset loaders — is Holliday's work.
> Our contribution is the outer multi-year MDP and the induced-demand dynamics
> built on top of it (see "What this fork adds"). Please keep the GPL notice and
> the citation below intact.

## What this fork adds

All new code for this project is isolated so it can be reviewed at a glance:

| Path | What it is |
| --- | --- |
| `learning/city_builder/` | The entire project contribution: the multi-year MDP, land-use dynamics, RL training/eval under the closed loop, baselines, ablations, and visualization. |
| `cfg/ppo_citybuilder_*.yaml` | Hydra configs for the city-builder PPO runs (Mandl, Mumford0/1, and the stage-1 batched variant). |
| `simulation/citygraph_dataset.py` | Instance-parameterized city graphs (configurable route geometry / horizon) used by the city-builder env. |
| `tests/test_land_use_dynamics.py` | Unit tests for the land-use update (`alpha = 0` identity, cap behavior, determinism). |
| `modal_runs/` | Modal + Weights & Biases cloud-training entry points for the experiments. |

Key modules inside `learning/city_builder/`:

- `multi_year_mdp.py` — the outer year-by-year environment wrapping Holliday's route machinery.
- `land_use_dynamics.py` — the per-zone activity update `x_{t+1} = clip(x_t * (base_rate + alpha * A_tilde) + eps, 0, cap)`.
- `accessibility.py`, `gravity.py`, `demand_hook.py` — transit-accessibility computation and the closed demand loop.
- `train_rl.py` / `eval_rl.py` — PPO training and apples-to-apples evaluation vs. the baselines.
- `alpha_sweep.py` — the induced-demand sensitivity sweep + decision gate (baselines).
- `plot_rl_alpha_ablation.py` — the RL-vs-greedy gap-vs-alpha figure (writeup anchor).
- `viz_city_growth.py` / `export_viz_trajectory.py` — city-evolution renderer and trajectory export.

## Setup

### Environment (uv recommended)

From the repository root:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r cc_requirements.txt
```

An `environment.yml` is also provided for conda-compatible tools.

All scripts are run as Python modules (`python -m package.module`) so sibling
packages like `simulation/` and `world/` resolve on `sys.path`; running the files
by path will fail with `ModuleNotFoundError`. Most scripts use
[Hydra](https://hydra.cc/) (override config from the CLI) or `argparse` — pass
`-h` / `--help` for usage.

### Datasets

I use the Mandl and Mumford instances, distributed as `CEC2013Supp.zip` from
[Christine Mumford's website](https://users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/Outline.html)
(mirrored on the [Wayback Machine](https://web.archive.org/web/*/users.cs.cf.ac.uk/C.L.Mumford/Research%20Topics/UTRP/CEC2013Supp.zip)).
Extract it so there is an `Instances/` subdirectory containing `MandlCoords.txt`,
`MandlTravelTimes.txt`, `MandlDemand.txt`, and the analogous `Mumford0`–`Mumford3`
triplets. Pass that directory to the city-builder scripts via `--instances-dir`
(argparse scripts) or `eval.dataset.path=` (Hydra scripts).

## Reproducing the City Builder results

All commands are run from the repository root with the venv active. The scripts
write to `results/` by default (the shipped figures there were produced this way).

**1. Induced-demand sanity check + alpha sweep (baselines).** Confirms `alpha = 0`
reproduces the static-demand setting and that a working `alpha` opens a
greedy-vs-random welfare gap:

```bash
python -m learning.city_builder.alpha_sweep \
    --instances-dir /path/to/Instances \
    --out results/top10_alpha_sweep
```

**2. Train the RL policy (GPU; Modal recommended).**

```bash
python -m learning.city_builder.train_rl \
    --config-name=ppo_citybuilder_mandl \
    +run_name=cb_mandl_seed0 experiment.seed=0
```

For cloud training + W&B logging across seeds, see `modal_runs/README.md`.

**3. Evaluate the trained policy vs. the baselines.** The policy's routes are
replayed through the same welfare path used for the greedy/random baselines, so
the numbers are directly comparable:

```bash
python -m learning.city_builder.eval_rl \
    --config-name=ppo_citybuilder_mandl \
    +model.weights=/path/to/citybuilder_..._seed0.pt
```

**4. RL-vs-greedy alpha ablation (headline figure).** Consumes the per-(alpha,
seed) JSON produced by the Modal ablation driver in `modal_runs/`:

```bash
python -m learning.city_builder.plot_rl_alpha_ablation \
    --in results/rl_alpha_ablation.json --metric integrated
```

**5. Visualize city growth.** Renders per-year snapshots of zones + transit
network and an animation:

```bash
python -m learning.city_builder.viz_city_growth \
    --alpha 0.5 --baseline greedy --seed 0 \
    --out results/city_growth
```

The build-then-watch (commit-then-observe) regime that produces the headline
result is selected through the city-builder configs; see `cfg/ppo_citybuilder_*.yaml`.

### Tests

```bash
pytest tests/test_land_use_dynamics.py
```

## Upstream base model (Holliday et al.)

The original route-design agent still works as documented upstream. To generate a
training dataset and train the base model:

```bash
python -m simulation.citygraph_dataset --min N --max N --n NUM_GRAPHS /path/to/dataset
python -m learning.inductive_route_learning dataset.kwargs.path=/path/to/dataset
```

Trained weights land in `output/` as `inductive_[run-name].pt`. To evaluate, run
the evolutionary algorithm (EA), or run the neural evolutionary algorithm (NEA),
use `learning.eval_route_generator` and `learning.bee_colony` (`bee_colony` is a
historical name); see the script `--help` for the full argument set. Pretrained
weights for the upstream experiments are available from the McGill MRL project
pages ([ITSC 2023](https://www.cim.mcgill.ca/~mrl/projs/transit_learning/itsc_2023),
[PPO 2025](https://www.cim.mcgill.ca/~mrl/projs/transit_learning/ppo_2025)).

## License

Released as free software under the **GNU General Public License**. All
constituent source files are covered by this license; see `COPYING` for the full
text.

## Citation

If you make use of this code, please cite Holliday & Dudek's associated paper:

```
@inproceedings{holliday2024autonomous,
    author = {Holliday, Andrew and Dudek, Gregory},
    title = {A Neural-Evolutionary Algorithm for Autonomous Transit Network Design},
    year = {2024},
    booktitle = {presented at 2024 IEEE International Conference on Robotics and Automation (ICRA)},
    organization = {IEEE}
}
```
