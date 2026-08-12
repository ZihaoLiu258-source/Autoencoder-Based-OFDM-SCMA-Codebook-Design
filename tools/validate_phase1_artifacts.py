"""Fail-fast integrity audit for the complete Phase-1 evidence package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


TRAINING_RUNS = [
    (Path("artifacts/phase1/m3_no_med"), "no_med", 0),
    (Path("artifacts/phase1/m4_seed_check/seed1"), "full", 1),
    (Path("artifacts/phase1/m4_seed_check/seed2"), "full", 2),
]
EVALUATIONS = [
    (Path("artifacts/main/diagnostics/decoder_variance_sensitivity.mat"), Path("artifacts/main/diagnostics/eval_decoder_variance_sensitivity.log")),
    (Path("artifacts/phase1/m1_randomized_training/m1_three_point_results.mat"), Path("artifacts/phase1/m1_randomized_training/m1_three_point_results.log")),
    (Path("artifacts/phase1/m3_no_med/m3_no_med_results.mat"), Path("artifacts/phase1/m3_no_med/m3_no_med_results.log")),
    (Path("artifacts/phase1/m4_seed_check/m4_seed_check_results.mat"), Path("artifacts/phase1/m4_seed_check/m4_seed_check_results.log")),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_training(directory: Path, condition: str, seed: int, root_support: torch.Tensor) -> None:
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "complete", metadata
    assert metadata["condition"] == condition
    assert metadata["seed"] == seed
    assert metadata["steps"] == 6000
    assert sha256(directory / "cb1_kmv.pt") == metadata["checkpoint_sha256"]["cb1_kmv.pt"]

    checkpoint = torch.load(directory / "cb1_kmv.pt", map_location="cpu")
    assert tuple(checkpoint.shape) == (4, 4, 6)
    assert torch.isfinite(checkpoint.real).all() and torch.isfinite(checkpoint.imag).all()
    assert torch.equal(checkpoint.abs() > 1e-8, root_support)
    log_text = (directory / "training.log").read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in log_text
    assert "[INFO] control training complete" in log_text
    assert "[RUN] finished=" in log_text
    print(f"[OK] training {condition}, seed={seed}: {directory}")


def validate_randomized_training(root_support: torch.Tensor) -> None:
    directory = Path("artifacts/phase1/m1_randomized_training")
    metadata = json.loads((directory / "training_metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "complete", metadata
    assert metadata["seed"] == 0
    assert metadata["steps"] == 6000
    checkpoint_path = directory / "randomized_cb_kmv.pt"
    assert sha256(checkpoint_path) == metadata["checkpoint_sha256"][checkpoint_path.name]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert tuple(checkpoint.shape) == (4, 4, 6)
    assert torch.isfinite(checkpoint.real).all() and torch.isfinite(checkpoint.imag).all()
    assert torch.equal(checkpoint.abs() > 1e-8, root_support)
    log_text = (directory / "train_randomized.log").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "Traceback" not in log_text
    assert "Saved randomized checkpoint and metadata" in log_text
    assert "[RUN] finished=" in log_text
    print(f"[OK] randomized training, seed=0: {directory}")


def validate_evaluation(mat_path: Path, log_path: Path) -> None:
    data = sio.loadmat(mat_path)
    errors = np.asarray(data["error_count"], dtype=np.int64)
    bits = np.asarray(data["total_bits"], dtype=np.int64)
    ber = np.asarray(data["BER"], dtype=np.float64)
    assert np.array_equal(errors, np.rint(errors).astype(np.int64))
    assert np.all(bits > 0)
    assert np.max(np.abs(ber - errors / bits)) == 0.0
    if "per_ofdm_error_count" in data:
        clustered = np.asarray(data["per_ofdm_error_count"], dtype=np.int64)
        assert np.array_equal(clustered.sum(axis=1), errors.ravel())
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    assert "Traceback" not in log_text
    assert "[RUN] finished=" in log_text
    assert "[INFO] Saved" in log_text
    print(f"[OK] evaluation: {mat_path}; max BER identity error=0")


def main() -> None:
    root = torch.load("artifacts/main/checkpoints/cb1_kmv.pt", map_location="cpu")
    root_support = root.abs() > 1e-8
    for directory, condition, seed in TRAINING_RUNS:
        validate_training(directory, condition, seed, root_support)
    validate_randomized_training(root_support)
    for mat_path, log_path in EVALUATIONS:
        validate_evaluation(mat_path, log_path)
    print("[OK] Complete Phase-1 artifact audit passed.")


if __name__ == "__main__":
    main()
