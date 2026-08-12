"""Train isolated fixed-point controls for Phase-1 ablations.

This wrapper reuses the main ``train_ofdm_scma.py`` algorithm, resets the
requested seed, optionally disables only Soft-MED, changes into a dedicated
artifact directory, and then invokes the main training entry point. Generic
checkpoint names are confined to that directory and cannot overwrite the
released seed-0 checkpoint under ``artifacts/main/checkpoints``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import train_ofdm_scma as original
from evaluation_utils import portable_path, run_with_log


REPO_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("full", "no_med"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=original.NUM_STEPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_random_state(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_control(args: argparse.Namespace, output_dir: Path) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if output_dir == REPO_DIR:
        raise ValueError("Control checkpoints must not be written to the repository root")

    checkpoint = output_dir / "cb1_kmv.pt"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"{checkpoint} already exists; use --overwrite only for an intentional rerun"
        )

    reset_random_state(args.seed)
    original.SEED = args.seed
    original.NUM_STEPS = args.steps
    original.USE_MED = args.condition == "full"
    original.OUTPUT_DIR = Path(".")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metadata = {
        "experiment": "Phase-1 isolated fixed-point training control",
        "condition": args.condition,
        "status": "running",
        "seed": args.seed,
        "steps": args.steps,
        "optimizer_updates": args.steps,
        "step_index_range": [0, args.steps - 1],
        "device": str(original.DEVICE),
        "fixed_impairments": {
            "cfo_epsilon": original.EPS_FIXED,
            "pn_sigma_step_rad": original.PN_SIGMA_STEP_RAD,
        },
        "soft_med": {
            "enabled": original.USE_MED,
            "lambda": original.LAMBDA_MED if original.USE_MED else 0.0,
            "tau": original.TAU_MED,
        },
        "controlled_settings": {
            "initialization_and_hinge_reference": "built-in Zhang codebook",
            "batch_ofdm_symbols": original.BATCH_OFDMSYM,
            "q_sub": original.Q_SUB,
            "mpa_iterations_low": original.NIT_MPA_LOW,
            "mpa_iterations_high": original.NIT_MPA_HIGH,
            "learning_rate": original.LEARNING_RATE,
            "lr_scheduler_step": original.LR_SCHEDULER_STEP,
            "lr_scheduler_gamma": original.LR_SCHEDULER_GAMMA,
            "high_snr_weight": original.HIGH_SNR_WEIGHT,
            "bce_weight": original.BCE_WEIGHT,
            "margin_weight": original.MARGIN_WEIGHT,
            "global_es_target": original.GLOBAL_ES_TARGET,
            "project_every": original.PROJECT_EVERY,
            "projection_mc_symbols": original.PROJ_BMC,
            "gradient_clip_norm": original.CLIP_GRAD_NORM,
        },
        "output_separation": portable_path(output_dir),
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[INFO] Phase-1 isolated fixed-point control")
    print(f"[INFO] condition={args.condition}; seed={args.seed}; steps={args.steps}")
    print(f"[INFO] output_dir={portable_path(output_dir)}")
    print("[INFO] Original repository-root checkpoint will not be modified.")
    started = time.perf_counter()
    previous_cwd = Path.cwd()
    try:
        os.chdir(output_dir)
        original.main()
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["failure_type"] = type(exc).__name__
        metadata["failure_message"] = str(exc)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise
    finally:
        os.chdir(previous_cwd)

    required_outputs = [
        output_dir / "codebook_e2e.pt",
        output_dir / "cb1_kmv.pt",
        output_dir / "codebook_e2e.npy",
        output_dir / "cb1_kmv.npy",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"Training returned without required outputs: {missing}")

    metadata["status"] = "complete"
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    metadata["checkpoint_sha256"] = {
        path.name: sha256(path) for path in required_outputs if path.suffix == ".pt"
    }
    metadata["outputs"] = [path.name for path in required_outputs] + [
        "training.log",
        "training_metadata.json",
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] control training complete in {metadata['elapsed_seconds']:.1f} s")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_with_log(
        lambda: run_control(args, output_dir),
        str(output_dir / "training.log"),
    )


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
