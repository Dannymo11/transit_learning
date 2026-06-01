"""Modal runner for TOP-11: unmodified Holliday PPO training on Mandl /
Mumford0--3 with Weights & Biases logging.

DESIGN NOTE
-----------
TOP-11 is the "calibration" milestone: train Holliday's policy under his
*static* dynamics (alpha=0 in our framework, but actually no LandUseDynamics
at all per M2 MDP doc sec 6 line 109) and confirm convergence matches the
published numbers. This runner intentionally does NOT import
`learning.city_builder` --- TOP-12 is where that integration begins.

W&B integration is non-invasive: we call ``wandb.tensorboard.patch`` before
the trainer starts, which auto-mirrors every ``SummaryWriter.add_scalar``
call inside ``learning.utils`` to the live W&B run. The trainer code itself
is untouched.

Usage
-----
See ``modal_runs/README.md``. Quick reference:

    # Single seed, Mandl
    modal run modal_runs/train_static_holliday.py --config-name=ppo_mandl --seed=0

    # 5-seed parallel sweep, Mandl
    modal run modal_runs/train_static_holliday.py::sweep \
        --config-name=ppo_mandl --seeds=0,1,2,3,4
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal


# ---------------------------------------------------------------------------
# Modal image / app definition
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# GPU by DEFAULT (CUDA build + A10G containers) --- worth it for the big Mumford
# instances (110-127 nodes) and for Mandl too (CPU runs ~20s/iter ~= 2.8h for
# 500 iters). Opt out to CPU with MODAL_GPU=0 (env var read at deploy time on
# the client).
#
#   GPU (default):  modal run modal_runs/train_static_holliday.py::sweep ...
#   CPU:            MODAL_GPU=0 modal run modal_runs/train_static_holliday.py::sweep ...
#
# torch 2.8 is NOT on the cu121 index; we use cu126 (has 2.8 wheels). The code
# imports no torch_scatter/sparse/cluster (PyG 2.7 uses native scatter), so the
# finicky data.pyg.org ext wheels are NOT needed for either build.
# ---------------------------------------------------------------------------
USE_GPU = os.environ.get("MODAL_GPU", "1") != "0"
TORCH_INDEX = (
    "https://download.pytorch.org/whl/cu126" if USE_GPU
    else "https://download.pytorch.org/whl/cpu"
)
GPU_SPEC = "A10G" if USE_GPU else None  # None => CPU container

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.8.0",
        index_url=TORCH_INDEX,
    )
    .pip_install(
        # Trainer + logging deps (PyPI)
        "torch_geometric==2.7.0",
        "numpy==2.4.4",
        "tensorboard==2.20.0",
        "scikit-learn==1.8.0",
        "tqdm==4.67.3",
        "networkx==3.6.1",
        "pyyaml==6.0.3",
        # lxml: simulation/__init__.py eagerly imports the MATSim simulator,
        # which (via world.transit + config_utils) needs lxml even though the
        # RL train/eval path never uses MATSim. Single missing dep in that chain.
        "lxml",
        "matplotlib==3.10.9",
        "scipy==1.17.1",
        "pandas==3.0.2",
        "optuna==4.8.0",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "wandb>=0.18.0",
    )
    # Mount the repo at /workspace. Excludes outputs/, weights/, viz_outputs/
    # and other heavy artifacts to keep image builds fast.
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/workspace",
        ignore=[
            "outputs/**",
            "viz_outputs/**",
            "training_logs/**",
            "**/__pycache__/**",
            "pytest-cache-files-*/**",
            ".venv/**",
            ".git/**",
            "*.pt",  # excludes the inductive_gae_seed_*.pt files at root
        ],
    )
)

app = modal.App("transit-rl-holliday-calibration")

# W&B API key as a Modal secret. Create with:
#   modal secret create wandb-secret WANDB_API_KEY=<your-key>
wandb_secret = modal.Secret.from_name("wandb-secret")

# Persistent volume for checkpoints and tensorboard logs. Survives function
# restarts; mount it on every train call so you can pull weights down via
# `modal volume get transit-rl-checkpoints <path>`.
checkpoint_volume = modal.Volume.from_name(
    "transit-rl-checkpoints", create_if_missing=True
)
CHECKPOINT_MOUNT = "/checkpoints"


# ---------------------------------------------------------------------------
# Single-seed training function (runs on a GPU)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_SPEC,  # None (CPU) unless MODAL_GPU=1; A10G when GPU.
    secrets=[wandb_secret],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    timeout=60 * 60 * 6,
)
def train(
    config_name: str,
    seed: int,
    wandb_project: str = "cs224r-city-builder",
    wandb_run_group: Optional[str] = None,
    extra_overrides: Optional[list[str]] = None,
) -> dict:
    """Run one seed of Holliday's PPO trainer with W&B logging.

    Equivalent to executing locally::

        python learning/inductive_route_learning.py \\
            --config-name=<config_name> \\
            +run_name=<config_name>_seed<seed> \\
            experiment.seed=<seed> \\
            experiment.logdir=<checkpoint_volume>/tb_logs

    plus a ``wandb.tensorboard.patch`` so all ``add_scalar`` calls auto-sync
    to W&B. Returns ``{seed, run_name, status, checkpoint_path}``.
    """
    import wandb

    # Late import so module-load doesn't pull torch.
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")

    run_name = f"{config_name}_seed{seed}"
    tb_logdir = Path(CHECKPOINT_MOUNT) / "tb_logs"
    tb_logdir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B BEFORE the trainer constructs its SummaryWriter so that
    # wandb.tensorboard.patch can hook every subsequent add_scalar call.
    wandb.tensorboard.patch(root_logdir=str(tb_logdir), pytorch=True)
    wandb.init(
        project=wandb_project,
        name=run_name,
        group=wandb_run_group or f"{config_name}_calibration",
        config={
            "config_name": config_name,
            "seed": seed,
            "milestone": "M3-TOP-11",
            "trainer": "holliday-unmodified",
        },
        sync_tensorboard=True,
        reinit=True,
    )

    # Build the hydra-style argv for inductive_route_learning's @hydra.main.
    # We invoke it as a subprocess so hydra owns its own cwd / output_dir
    # rather than fighting with Modal's working directory.
    # Write the checkpoint DIRECTLY onto the persistent volume. The trainer
    # saves to ``output_dir / (run_name + '.pt')`` (inductive_route_learning.py
    # :820), and `output_dir = cfg.outdir` when provided. ppo_mandl.yaml does
    # not set outdir, so without this override it would default to a cwd-relative
    # ``output/`` dir that is lost when the container exits (hydra.job.chdir
    # defaults to False in hydra 1.3, so cwd stays /workspace). Pointing outdir
    # at the mounted volume makes persistence robust instead of relying on a
    # post-hoc glob.
    weights_dir = Path(CHECKPOINT_MOUNT) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "learning/inductive_route_learning.py",
        f"--config-name={config_name}",
        f"+run_name={run_name}",
        f"experiment.seed={seed}",
        f"experiment.logdir={tb_logdir}",
        f"+outdir={weights_dir}",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    print("[modal_runs] launching:", " ".join(shlex.quote(c) for c in cmd))
    # PYTHONPATH=/workspace so the subprocess can import the top-level
    # `simulation` / `learning` packages. The parent's sys.path edits don't
    # propagate to a subprocess, and `python learning/foo.py` only puts
    # /workspace/learning on the path, not /workspace itself.
    proc = subprocess.run(
        cmd, cwd="/workspace",
        env={**os.environ, "PYTHONPATH": "/workspace"},
        capture_output=False,
    )

    # The trainer saves as `inductive_{run_name}.pt`: setup_and_train calls
    # process_standard_experiment_cfg(cfg, 'inductive_'), which prepends that
    # prefix to cfg.run_name (utils.py:148-150). With +outdir pointing at the
    # volume it lands directly in weights_dir.
    ckpt_name = f"inductive_{run_name}.pt"
    primary = weights_dir / ckpt_name
    persisted = [str(primary)] if primary.exists() else []
    if not persisted:
        # Fallback: any file ending in {run_name}.pt, in the volume or a
        # cwd-relative output/outputs dir (covers prefix/chdir surprises).
        roots = [weights_dir, Path("/workspace/output"), Path("/workspace/outputs")]
        for root in roots:
            if not root.exists():
                continue
            for src in root.rglob(f"*{run_name}.pt"):
                dst = weights_dir / src.name
                if src != dst:
                    dst.write_bytes(src.read_bytes())
                persisted.append(str(dst))
    checkpoint_volume.commit()

    wandb.finish()

    return {
        "seed": seed,
        "run_name": run_name,
        "status": "ok" if proc.returncode == 0 else f"exit_{proc.returncode}",
        "checkpoints": persisted,
    }


# ---------------------------------------------------------------------------
# TOP-12 dynamic-environment RL training (City Builder, GPU)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_SPEC,  # GPU by default (RL training is the expensive job)
    secrets=[wandb_secret],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    timeout=60 * 60 * 6,
)
def train_rl(
    config_name: str = "ppo_citybuilder_mandl",
    seed: int = 0,
    wandb_project: str = "cs224r-city-builder",
    wandb_run_group: Optional[str] = None,
) -> dict:
    """Run learning/city_builder/train_rl.py (PPO under the closed induced-demand
    loop) for one seed, with W&B logging + checkpoint persistence to the volume.
    Mirrors `train`; the trainer saves `citybuilder_{run_name}.pt`."""
    import wandb

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    run_name = f"{config_name}_seed{seed}"
    tb_logdir = Path(CHECKPOINT_MOUNT) / "tb_logs"
    tb_logdir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(CHECKPOINT_MOUNT) / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    wandb.tensorboard.patch(root_logdir=str(tb_logdir), pytorch=True)
    wandb.init(project=wandb_project, name=run_name,
               group=wandb_run_group or f"{config_name}_rl",
               config={"config_name": config_name, "seed": seed,
                       "milestone": "M3-TOP-12"},
               sync_tensorboard=True, reinit=True)

    cmd = [
        "python", "-m", "learning.city_builder.train_rl",
        f"--config-name={config_name}",
        f"+run_name={run_name}",
        f"experiment.seed={seed}",
        f"experiment.logdir={tb_logdir}",
        f"+outdir={weights_dir}",
    ]
    print("[modal_runs] launching:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(
        cmd, cwd="/workspace",
        env={**os.environ, "PYTHONPATH": "/workspace"}, capture_output=False)

    # train_rl saves `citybuilder_{run_name}.pt` (process_standard_experiment_cfg
    # prepends 'citybuilder_'). Suffix-glob fallback as in `train`.
    ckpt_name = f"citybuilder_{run_name}.pt"
    primary = weights_dir / ckpt_name
    persisted = [str(primary)] if primary.exists() else []
    if not persisted:
        for root in (weights_dir, Path("/workspace/output"), Path("/workspace/outputs")):
            if not root.exists():
                continue
            for src in root.rglob(f"*{run_name}.pt"):
                dst = weights_dir / src.name
                if src != dst:
                    dst.write_bytes(src.read_bytes())
                persisted.append(str(dst))
    checkpoint_volume.commit()
    wandb.finish()
    return {"seed": seed, "run_name": run_name,
            "status": "ok" if proc.returncode == 0 else f"exit_{proc.returncode}",
            "checkpoints": persisted}


# ---------------------------------------------------------------------------
# Local entrypoints (CLI)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def rl(
    config_name: str = "ppo_citybuilder_mandl",
    seed: int = 0,
    wandb_project: str = "cs224r-city-builder",
):
    """TOP-12 Stage-0 smoke: one seed of dynamic-environment RL training.

        modal run modal_runs/train_static_holliday.py::rl \\
            --config-name=ppo_citybuilder_mandl --seed=0
    """
    print(train_rl.remote(config_name=config_name, seed=seed,
                          wandb_project=wandb_project))


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    timeout=60 * 20,
)
def eval_rl_remote(config_name: str = "ppo_citybuilder_mandl", seed: int = 0) -> dict:
    """Greedy eval of a trained RL checkpoint vs baselines (prints the table)."""
    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    checkpoint_volume.reload()
    run_name = f"{config_name}_seed{seed}"
    weights = f"{CHECKPOINT_MOUNT}/weights/citybuilder_{run_name}.pt"
    if not Path(weights).exists():
        cands = list((Path(CHECKPOINT_MOUNT) / "weights").glob(f"*{run_name}.pt"))
        weights = str(cands[0]) if cands else weights
    cmd = ["python", "-m", "learning.city_builder.eval_rl",
           f"--config-name={config_name}", f"+model.weights={weights}"]
    print("[modal_runs] launching:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd="/workspace",
                          env={**os.environ, "PYTHONPATH": "/workspace"},
                          capture_output=False)
    return {"status": "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"}


@app.local_entrypoint()
def eval_policy(config_name: str = "ppo_citybuilder_mandl", seed: int = 0):
    """Evaluate the trained RL policy greedily vs greedy/random baselines.

        modal run modal_runs/train_static_holliday.py::eval_policy \\
            --config-name=ppo_citybuilder_mandl --seed=0
    """
    print(eval_rl_remote.remote(config_name=config_name, seed=seed))


@app.local_entrypoint()
def main(
    config_name: str = "ppo_mandl",
    seed: int = 0,
    wandb_project: str = "cs224r-city-builder",
    extra_overrides: str = "",
):
    """Train a single seed. See ``sweep`` for multi-seed parallel fan-out.

    Example::

        modal run modal_runs/train_static_holliday.py \\
            --config-name=ppo_mandl --seed=0
    """
    overrides = shlex.split(extra_overrides) if extra_overrides else None
    result = train.remote(
        config_name=config_name,
        seed=seed,
        wandb_project=wandb_project,
        extra_overrides=overrides,
    )
    print(result)


@app.local_entrypoint()
def sweep(
    config_name: str = "ppo_mandl",
    seeds: str = "0,1,2,3,4",
    wandb_project: str = "cs224r-city-builder",
    extra_overrides: str = "",
):
    """Fan out N seeds in parallel on Modal. Decision-gate-friendly: the M3
    milestone (TOP-11) wants 5+ seeds for variance.

    Example::

        modal run modal_runs/train_static_holliday.py::sweep \\
            --config-name=ppo_mandl --seeds=0,1,2,3,4
    """
    overrides = shlex.split(extra_overrides) if extra_overrides else None
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    group = f"{config_name}_calibration_{'_'.join(str(s) for s in seed_list)}"

    # `.starmap` would be cleaner but `.spawn` lets us see live progress.
    handles = [
        train.spawn(
            config_name=config_name,
            seed=s,
            wandb_project=wandb_project,
            wandb_run_group=group,
            extra_overrides=overrides,
        )
        for s in seed_list
    ]
    results = [h.get() for h in handles]
    for r in results:
        print(r)


# ---------------------------------------------------------------------------
# Gate evaluation (TOP-11 pass/fail vs Holliday LC-100)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU_SPEC,  # None (CPU) unless MODAL_GPU=1; A10G when GPU.
    secrets=[wandb_secret],
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    timeout=60 * 30,
)
def evaluate(
    config_name: str,
    seed: int,
    eval_config_name: str = "eval_gate_mandl",
    wandb_project: str = "cs224r-city-builder",
    wandb_run_group: Optional[str] = None,
) -> dict:
    """Evaluate one trained checkpoint at cost-weight alpha in {0, 0.5, 1}.

    Reads ``{CHECKPOINT_MOUNT}/weights/{config_name}_seed{seed}.pt`` (written by
    ``train``), runs ``learning/city_builder/eval_gate.py`` to produce the
    per-alpha C(alpha) numbers, logs them to W&B as ``eval_cost_alpha_*``, and
    persists the per-seed JSON to ``{CHECKPOINT_MOUNT}/gate``. Aggregation +
    paper pass/fail is done by the ``gate`` entrypoint / gate_report.py.
    """
    import json
    import wandb

    sys.path.insert(0, "/workspace")
    os.chdir("/workspace")
    checkpoint_volume.reload()  # see weights committed by train()

    run_name = f"{config_name}_seed{seed}"
    # The trainer saves with the 'inductive_' prefix (setup_and_train ->
    # process_standard_experiment_cfg(cfg, 'inductive_')).
    weights = f"{CHECKPOINT_MOUNT}/weights/inductive_{run_name}.pt"
    gate_out = f"{CHECKPOINT_MOUNT}/gate"

    if not Path(weights).exists():
        # Fallback: tolerate a different/absent prefix by suffix-matching.
        cands = list((Path(CHECKPOINT_MOUNT) / "weights").glob(f"*{run_name}.pt"))
        if cands:
            weights = str(cands[0])
        else:
            return {"seed": seed, "run_name": run_name,
                    "status": "missing_checkpoint", "weights": weights}

    cmd = [
        "python", "-m", "learning.city_builder.eval_gate",
        f"--config-name={eval_config_name}",
        f"model.weights={weights}",
        f"+run_name={run_name}",
        f"gate_out={gate_out}",
    ]
    print("[modal_runs] launching:", " ".join(shlex.quote(c) for c in cmd))
    # PYTHONPATH=/workspace so the subprocess can import the top-level
    # `simulation` / `learning` packages. The parent's sys.path edits don't
    # propagate to a subprocess, and `python learning/foo.py` only puts
    # /workspace/learning on the path, not /workspace itself.
    proc = subprocess.run(
        cmd, cwd="/workspace",
        env={**os.environ, "PYTHONPATH": "/workspace"},
        capture_output=False,
    )
    checkpoint_volume.commit()

    json_path = Path(gate_out) / f"{run_name}_gate.json"
    if proc.returncode != 0 or not json_path.exists():
        return {"seed": seed, "run_name": run_name,
                "status": f"eval_failed_exit_{proc.returncode}"}

    payload = json.loads(json_path.read_text())
    results = payload["results"]

    # Log to W&B (resumes the seed's training run if names match).
    wandb.init(
        project=wandb_project,
        name=f"{run_name}_gate",
        group=wandb_run_group or f"{config_name}_calibration",
        config={"config_name": config_name, "seed": seed,
                "milestone": "M3-TOP-11-gate"},
        reinit=True,
    )
    label = {"0.0": "eval_cost_alpha_0", "0.5": "eval_cost_alpha_0p5",
             "1.0": "eval_cost_alpha_1"}
    for a_str, rec in results.items():
        wandb.log({label.get(a_str, f"eval_cost_alpha_{a_str}"): rec["cost"]})
        if rec.get("att_min") is not None:
            wandb.log({f"eval_att_min_alpha_{a_str}": rec["att_min"]})
    wandb.finish()

    return {"seed": seed, "run_name": run_name, "status": "ok",
            "city": payload["city"], "results": results}


@app.local_entrypoint()
def gate(
    config_name: str = "ppo_mandl",
    seeds: str = "0,1,2,3,4",
    eval_config_name: str = "eval_gate_mandl",
    wandb_project: str = "cs224r-city-builder",
):
    """Evaluate all seeds and print the TOP-11 pass/fail vs Holliday LC-100.

    Run AFTER ``sweep`` has finished training. Example::

        modal run modal_runs/train_static_holliday.py::gate \\
            --config-name=ppo_mandl --seeds=0,1,2,3,4
    """
    import numpy as np
    import importlib.util
    from collections import defaultdict

    # Load gate_report.py as a STANDALONE file rather than
    # `from learning.city_builder.gate_report import ...`. The package import
    # would run learning/city_builder/__init__.py, which pulls
    # simulation -> MATSim -> lxml. This entrypoint runs on the LOCAL machine
    # (not the container), and the local venv need not have lxml. gate_report
    # itself is numpy-only, so loading the file directly avoids the whole chain.
    _gr_path = REPO_ROOT / "learning" / "city_builder" / "gate_report.py"
    _spec = importlib.util.spec_from_file_location("_gate_report", _gr_path)
    _gr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gr)
    PAPER_LC100, GATE_ALPHA = _gr.PAPER_LC100, _gr.GATE_ALPHA

    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    group = f"{config_name}_calibration_{'_'.join(str(s) for s in seed_list)}"
    handles = [
        evaluate.spawn(config_name=config_name, seed=s,
                       eval_config_name=eval_config_name,
                       wandb_project=wandb_project, wandb_run_group=group)
        for s in seed_list
    ]
    results = [h.get() for h in handles]

    by_alpha: dict[float, list[float]] = defaultdict(list)
    city = None
    for r in results:
        print(r.get("status"), r.get("run_name"))
        if r.get("status") == "ok":
            city = r["city"]
            for a_str, rec in r["results"].items():
                by_alpha[float(a_str)].append(float(rec["cost"]))

    if city is None or city not in PAPER_LC100:
        print("No usable eval results; nothing to gate.")
        return

    paper = PAPER_LC100[city]
    print(f"\nTOP-11 gate — {city}  ({len(by_alpha.get(GATE_ALPHA, []))} seeds)")
    overall = None
    for a in sorted(paper):
        vals = np.array(by_alpha.get(a, []), dtype=float)
        p_mu, p_sd = paper[a]
        if vals.size == 0:
            print(f"  alpha={a}: (missing)")
            continue
        o_mu, o_sd = vals.mean(), vals.std()
        band = p_mu + max(p_sd, o_sd)
        ok = o_mu <= band
        print(f"  alpha={a}: ours {o_mu:.4f}±{o_sd:.4f}  paper {p_mu:.3f}±{p_sd:.3f}"
              f"  band {band:.4f}  {'PASS' if ok else 'FAIL'}")
        if abs(a - GATE_ALPHA) < 1e-9:
            overall = ok
    print(f"GATE (alpha={GATE_ALPHA}): "
          f"{'PASS' if overall else 'FAIL' if overall is not None else 'INCONCLUSIVE'}")
