# Modal runners

Cloud-training entry points for the CS 224R City Builder project. Mirrors
`slurm_jobs/` but targets Modal GPUs + Weights & Biases.

## Setup (one-time)

1. Install Modal locally: `pip install modal`
2. Authenticate: `modal token new`
3. Create a Modal secret containing your W&B API key:

   ```bash
   modal secret create wandb-secret WANDB_API_KEY=<your-key>
   ```

   The runner expects the secret name `wandb-secret`.

4. (Optional) Verify W&B project: the runner logs to project
   `cs224r-city-builder` by default. Override via `--wandb-project=<name>`.

## Files

| File | Purpose |
| --- | --- |
| `train_static_holliday.py` | TOP-11 calibration: unmodified Holliday PPO on Mandl / Mumford0–3. No `learning/city_builder/` integration. |

## TOP-11 usage

Mandl, single seed:

```bash
modal run modal_runs/train_static_holliday.py --config-name=ppo_mandl --seed=0
```

Mandl, 5-seed sweep (parallel fan-out):

```bash
modal run modal_runs/train_static_holliday.py::sweep --config-name=ppo_mandl --seeds=0,1,2,3,4
```

Mumford3, 5-seed sweep (after Mandl calibration passes):

```bash
modal run modal_runs/train_static_holliday.py::sweep --config-name=ppo_mumford3 --seeds=0,1,2,3,4
```

Trained checkpoints land in the Modal volume `transit-rl-checkpoints`; W&B
captures the live training curves under the project `cs224r-city-builder`.
