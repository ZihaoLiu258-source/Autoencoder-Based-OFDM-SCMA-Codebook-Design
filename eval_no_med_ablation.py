"""M3 one-point paired BER ablation for the Soft-MED regularizer."""

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
from phase1_control_utils import clustered_paired_difference, evaluate_at_point


SEED = 2030
EBN0_DB = 30.0
EPSILON = 0.04
PN_SIGMA_STEP = 1.0e-3
ND_TOTAL = 20_000
ND_CHUNK = 1_000

ARTIFACT_DIR = Path("artifacts/m3_no_med")
CHECKPOINTS = [Path("cb1_kmv.pt"), ARTIFACT_DIR / "cb1_kmv.pt"]
LABELS = ["Full loss", "Without Soft-MED"]
OUTPUT_MAT = ARTIFACT_DIR / "m3_no_med_results.mat"
OUTPUT_CSV = ARTIFACT_DIR / "m3_no_med_results.csv"
OUTPUT_LOG = ARTIFACT_DIR / "m3_no_med_results.log"


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
    paired = clustered_paired_difference(
        result["per_ofdm_error_count"], 0, 1, int(result["bits_per_ofdm"])
    )
    result.update({f"no_med_minus_full_{key}": value for key, value in paired.items()})
    sio.savemat(OUTPUT_MAT, result)

    relative_change = result["BER"][1] / result["BER"][0] - 1.0
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "condition", "error_count", "total_bits", "BER",
                "no_med_vs_full_relative_change", "paired_OFDM_p",
                "BER_difference_CI95_low", "BER_difference_CI95_high",
            ]
        )
        for index, label in enumerate(LABELS):
            writer.writerow(
                [
                    label,
                    result["error_count"][index],
                    result["total_bits"][index],
                    result["BER"][index],
                    relative_change,
                    paired["paired_OFDM_two_sided_p"],
                    paired["mean_BER_difference_CI95"][0],
                    paired["mean_BER_difference_CI95"][1],
                ]
            )
    print(
        f"[RESULT] no-MED vs full relative BER change={relative_change:+.3%}; "
        f"clustered p={paired['paired_OFDM_two_sided_p']:.3e}; "
        f"95% CI={paired['mean_BER_difference_CI95'].tolist()}"
    )
    print(f"[INFO] Saved {portable_path(OUTPUT_MAT)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_CSV)}")


if __name__ == "__main__":
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_with_log(run, str(OUTPUT_LOG))
