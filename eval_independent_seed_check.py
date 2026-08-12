"""M4 one-point BER stability check across three independent training seeds."""

from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_DIR = Path("tmp/matplotlib_phase1").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import numpy as np
import scipy.io as sio

from evaluation_utils import portable_path, run_with_log
from phase1_control_utils import evaluate_at_point


SEED = 2031
EBN0_DB = 30.0
EPSILON = 0.04
PN_SIGMA_STEP = 1.0e-3
ND_TOTAL = 20_000
ND_CHUNK = 1_000

ARTIFACT_DIR = Path("artifacts/m4_seed_check")
CHECKPOINTS = [
    Path("cb1_kmv.pt"),
    ARTIFACT_DIR / "seed1/cb1_kmv.pt",
    ARTIFACT_DIR / "seed2/cb1_kmv.pt",
]
LABELS = ["Seed 0", "Seed 1", "Seed 2"]
OUTPUT_MAT = ARTIFACT_DIR / "m4_seed_check_results.mat"
OUTPUT_CSV = ARTIFACT_DIR / "m4_seed_check_results.csv"
OUTPUT_LOG = ARTIFACT_DIR / "m4_seed_check_results.log"


def run() -> None:
    result = evaluate_at_point(
        CHECKPOINTS,
        LABELS,
        seed=SEED,
        ebn0_db=EBN0_DB,
        epsilon=EPSILON,
        pn_sigma_step=PN_SIGMA_STEP,
        nd_total=ND_TOTAL,
        nd_chunk=ND_CHUNK,
    )
    ber_mean = float(np.mean(result["BER"]))
    ber_std = float(np.std(result["BER"], ddof=1))
    ber_cv = ber_std / ber_mean
    result.update(
        {
            "BER_across_training_seeds_mean": np.float64(ber_mean),
            "BER_across_training_seeds_sample_std": np.float64(ber_std),
            "BER_across_training_seeds_CV": np.float64(ber_cv),
            "training_seeds": np.array([0, 1, 2], dtype=np.int64),
            "stability_definition": "descriptive mean and sample standard deviation across three independently trained checkpoints evaluated with one shared Monte-Carlo realization",
        }
    )
    sio.savemat(OUTPUT_MAT, result)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "training_seed", "error_count", "total_bits", "BER",
                "across_seed_mean_BER", "across_seed_sample_std", "across_seed_CV",
            ]
        )
        for seed, errors, bits, ber in zip(
            (0, 1, 2), result["error_count"], result["total_bits"], result["BER"]
        ):
            writer.writerow([seed, errors, bits, ber, ber_mean, ber_std, ber_cv])
    print(
        f"[RESULT] across-training-seed BER={ber_mean:.9e} +/- {ber_std:.9e} "
        f"(sample std; CV={ber_cv:.3%})"
    )
    print(f"[INFO] Saved {portable_path(OUTPUT_MAT)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_CSV)}")


if __name__ == "__main__":
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_with_log(run, str(OUTPUT_LOG))
