"""M1 three-point Monte-Carlo comparison: fixed vs randomized training."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_DIR = Path("tmp/matplotlib_m1").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import numpy as np
import scipy.io as sio
from scipy.stats import t as student_t
import torch

import eval_ber_vs_phasenoise as sim
from evaluation_utils import (
    ber_with_zero_error_bound,
    normalize_codebook_exact_global_es,
    portable_path,
    run_with_log,
)


SEED = 2027
EBN0_DB = 30.0
ND_TOTAL = 20_000
ND_CHUNK = 1_000

POINT_LABELS = ["mild", "fixed_training_point", "heavy"]
EPSILON_VALUES = np.array([0.02, 0.04, 0.06], dtype=np.float64)
PN_SIGMA_VALUES = np.array([0.5e-3, 1.0e-3, 3.0e-3], dtype=np.float64)

ARTIFACT_DIR = Path("artifacts/phase1/m1_randomized_training")
FIXED_CHECKPOINT = Path("artifacts/main/checkpoints/cb1_kmv.pt")
RANDOMIZED_CHECKPOINT = ARTIFACT_DIR / "randomized_cb_kmv.pt"
OUTPUT_MAT = ARTIFACT_DIR / "m1_three_point_results.mat"
OUTPUT_CSV = ARTIFACT_DIR / "m1_three_point_results.csv"
OUTPUT_LOG = ARTIFACT_DIR / "m1_three_point_results.log"


def load_codebook(path: Path, device: torch.device) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    codebook = torch.load(path, map_location=device)
    if not torch.is_tensor(codebook):
        codebook = torch.as_tensor(codebook)
    if tuple(codebook.shape) != (sim.K, sim.M, sim.V):
        raise ValueError(f"{path}: shape {tuple(codebook.shape)} != {(sim.K, sim.M, sim.V)}")
    return sim.hardzero(codebook.to(device=device, dtype=sim.DTYPEC), sim.HARDZERO_THR)


def run() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(sim.DEVICE)
    labels = ["Fixed-point training", "Randomized training"]
    checkpoint_paths = [FIXED_CHECKPOINT, RANDOMIZED_CHECKPOINT]
    raw_codebooks = [load_codebook(path, device) for path in checkpoint_paths]

    codebooks = []
    source_global_es = []
    normalized_global_es = []
    for codebook in raw_codebooks:
        normalized, source_es, realized_es = normalize_codebook_exact_global_es(
            codebook, target_es=sim.GLOBAL_ES_TARGET
        )
        codebooks.append(normalized.to(sim.DTYPEC))
        source_global_es.append(source_es)
        normalized_global_es.append(realized_es)
    factor_graphs = [sim.build_factor_graph(cb, thr=sim.FG_THR) for cb in codebooks]

    snr_db = EBN0_DB + 10.0 * math.log10(sim.R_bits_per_RE)
    n0_physical = 1.0 / (10.0 ** (snr_db / 10.0))
    num_points = len(POINT_LABELS)
    num_codebooks = len(codebooks)
    error_count = np.zeros((num_points, num_codebooks), dtype=np.int64)
    total_bits = np.zeros_like(error_count)
    fixed_wrong_randomized_correct = np.zeros(num_points, dtype=np.int64)
    fixed_correct_randomized_wrong = np.zeros(num_points, dtype=np.int64)
    per_ofdm_error_difference = np.zeros((num_points, ND_TOTAL), dtype=np.int16)

    print(f"[INFO] device={device}; Eb/N0={EBN0_DB:g} dB; Nd/point={ND_TOTAL}")
    print("[INFO] Paired common-random-number evaluation for the two checkpoints.")
    for point_index, (point_label, epsilon, pn_sigma) in enumerate(
        zip(POINT_LABELS, EPSILON_VALUES, PN_SIGMA_VALUES)
    ):
        nd_done = 0
        while nd_done < ND_TOTAL:
            nd_now = min(ND_CHUNK, ND_TOTAL - nd_done)
            x_vmq, h_rel, w_td, phi_td = sim.gen_shared_chunk(
                nd_now,
                n0_physical,
                device=device,
                sigma_val=float(pn_sigma),
            )
            chunk_error_masks = []
            for codebook_index, codebook in enumerate(codebooks):
                res_users, user_ress = factor_graphs[codebook_index]
                y, h, x_flat = sim.simulate_chunk_shared(
                    codebook,
                    float(epsilon),
                    x_vmq,
                    h_rel,
                    w_td,
                    phi_td,
                    device=device,
                )
                llr = sim.scmadec_logmpa_llr(
                    y,
                    codebook,
                    h,
                    n0_physical,
                    sim.Nit,
                    res_users,
                    user_ress,
                )
                bits_tx = sim.symbols_to_bits(x_flat, m_bits=sim.m_bits)
                bits_hat = sim.llr_to_bits(llr, V=sim.V, m_bits=sim.m_bits)
                error_mask = bits_tx != bits_hat
                chunk_error_masks.append(error_mask)
                error_count[point_index, codebook_index] += int(error_mask.sum().item())
                total_bits[point_index, codebook_index] += int(bits_tx.numel())
            fixed_error, randomized_error = chunk_error_masks
            fixed_errors_per_ofdm = fixed_error.reshape(
                sim.V, nd_now, sim.Q, sim.m_bits
            ).sum(dim=(0, 2, 3))
            randomized_errors_per_ofdm = randomized_error.reshape(
                sim.V, nd_now, sim.Q, sim.m_bits
            ).sum(dim=(0, 2, 3))
            per_ofdm_error_difference[
                point_index, nd_done:nd_done + nd_now
            ] = (
                randomized_errors_per_ofdm - fixed_errors_per_ofdm
            ).cpu().numpy().astype(np.int16)
            fixed_wrong_randomized_correct[point_index] += int(
                (fixed_error & ~randomized_error).sum().item()
            )
            fixed_correct_randomized_wrong[point_index] += int(
                (~fixed_error & randomized_error).sum().item()
            )
            nd_done += nd_now
            if nd_done % 5_000 == 0 or nd_done == ND_TOTAL:
                print(
                    f"[INFO] point={point_label} eps={epsilon:.3f} "
                    f"sigma={pn_sigma:.1e}: Nd={nd_done}/{ND_TOTAL}"
                )

    ber = error_count / total_bits
    ber_95_upper = np.full_like(ber, np.nan, dtype=np.float64)
    ber_is_upper_bound = np.zeros_like(error_count, dtype=np.uint8)
    for point_index in range(num_points):
        for codebook_index in range(num_codebooks):
            empirical, upper, is_upper = ber_with_zero_error_bound(
                int(error_count[point_index, codebook_index]),
                int(total_bits[point_index, codebook_index]),
            )
            ber[point_index, codebook_index] = empirical
            ber_95_upper[point_index, codebook_index] = upper
            ber_is_upper_bound[point_index, codebook_index] = int(is_upper)

    relative_ber_change = np.divide(
        ber[:, 1] - ber[:, 0],
        ber[:, 0],
        out=np.full(num_points, np.nan, dtype=np.float64),
        where=ber[:, 0] > 0,
    )
    bits_per_ofdm = sim.V * sim.Q * sim.m_bits
    mean_ber_difference = per_ofdm_error_difference.mean(axis=1) / bits_per_ofdm
    standard_error_ber_difference = (
        per_ofdm_error_difference.std(axis=1, ddof=1)
        / math.sqrt(ND_TOTAL)
        / bits_per_ofdm
    )
    t_statistic = np.divide(
        mean_ber_difference,
        standard_error_ber_difference,
        out=np.zeros(num_points, dtype=np.float64),
        where=standard_error_ber_difference > 0,
    )
    paired_ofdm_two_sided_p = 2.0 * student_t.sf(
        np.abs(t_statistic), df=ND_TOTAL - 1
    )
    t_critical_95 = student_t.ppf(0.975, df=ND_TOTAL - 1)
    mean_ber_difference_ci95 = np.column_stack(
        [
            mean_ber_difference - t_critical_95 * standard_error_ber_difference,
            mean_ber_difference + t_critical_95 * standard_error_ber_difference,
        ]
    )

    sio.savemat(
        OUTPUT_MAT,
        {
            "labels": labels,
            "point_labels": POINT_LABELS,
            "epsilon": EPSILON_VALUES,
            "pn_sigma_step": PN_SIGMA_VALUES,
            "EbN0_dB": np.float64(EBN0_DB),
            "Nd_used": np.full(num_points, ND_TOTAL, dtype=np.int64),
            "seed": np.int64(SEED),
            "error_count": error_count,
            "total_bits": total_bits,
            "BER": ber,
            "BER_95_upper": ber_95_upper,
            "BER_is_upper_bound": ber_is_upper_bound,
            "randomized_minus_fixed_relative_BER": relative_ber_change,
            "fixed_wrong_randomized_correct": fixed_wrong_randomized_correct,
            "fixed_correct_randomized_wrong": fixed_correct_randomized_wrong,
            "per_ofdm_error_difference_randomized_minus_fixed": per_ofdm_error_difference,
            "mean_BER_difference_randomized_minus_fixed": mean_ber_difference,
            "standard_error_BER_difference": standard_error_ber_difference,
            "mean_BER_difference_CI95": mean_ber_difference_ci95,
            "paired_OFDM_t_statistic": t_statistic,
            "paired_OFDM_two_sided_p": paired_ofdm_two_sided_p,
            "paired_test_definition": "two-sided paired t-test over independent OFDM-symbol aggregate error-count differences; each symbol is one cluster",
            "source_global_es": np.asarray(source_global_es),
            "normalized_global_es": np.asarray(normalized_global_es),
            "fixed_checkpoint": str(FIXED_CHECKPOINT),
            "randomized_checkpoint": str(RANDOMIZED_CHECKPOINT),
            "decoder_model": "same no-tracking 10-iteration Log-MPA; N0_dec=N0_physical",
            "zero_error_bound_definition": "95% rule of three: 3 / total_bits",
        },
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "point", "epsilon", "pn_sigma_step", "EbN0_dB", "training",
                "error_count", "total_bits", "BER", "BER_95_upper", "is_upper_bound",
                "fixed_wrong_randomized_correct", "fixed_correct_randomized_wrong",
                "mean_BER_difference_randomized_minus_fixed", "BER_difference_CI95_low",
                "BER_difference_CI95_high", "paired_OFDM_two_sided_p",
            ]
        )
        for point_index, point_label in enumerate(POINT_LABELS):
            for codebook_index, label in enumerate(labels):
                writer.writerow(
                    [
                        point_label,
                        EPSILON_VALUES[point_index],
                        PN_SIGMA_VALUES[point_index],
                        EBN0_DB,
                        label,
                        error_count[point_index, codebook_index],
                        total_bits[point_index, codebook_index],
                        ber[point_index, codebook_index],
                        ber_95_upper[point_index, codebook_index],
                        ber_is_upper_bound[point_index, codebook_index],
                        fixed_wrong_randomized_correct[point_index],
                        fixed_correct_randomized_wrong[point_index],
                        mean_ber_difference[point_index],
                        mean_ber_difference_ci95[point_index, 0],
                        mean_ber_difference_ci95[point_index, 1],
                        paired_ofdm_two_sided_p[point_index],
                    ]
                )

    for point_index, point_label in enumerate(POINT_LABELS):
        parts = []
        for codebook_index, label in enumerate(labels):
            if ber_is_upper_bound[point_index, codebook_index]:
                metric = f"BER=0 (95% UB={ber_95_upper[point_index, codebook_index]:.3e})"
            else:
                metric = f"BER={ber[point_index, codebook_index]:.6e}"
            parts.append(
                f"{label}: errors={error_count[point_index, codebook_index]}, "
                f"bits={total_bits[point_index, codebook_index]}, {metric}"
            )
        print(
            f"[RESULT] {point_label} | eps={EPSILON_VALUES[point_index]:.3f} | "
            f"sigma={PN_SIGMA_VALUES[point_index]:.1e} | " + " | ".join(parts)
            + f" | discordant fixed-wrong/randomized-correct="
            f"{fixed_wrong_randomized_correct[point_index]}, "
            f"fixed-correct/randomized-wrong={fixed_correct_randomized_wrong[point_index]}, "
            f"clustered BER difference={mean_ber_difference[point_index]:.3e} "
            f"(95% CI [{mean_ber_difference_ci95[point_index, 0]:.3e}, "
            f"{mean_ber_difference_ci95[point_index, 1]:.3e}]), "
            f"paired-OFDM p={paired_ofdm_two_sided_p[point_index]:.3e}"
        )
    print(f"[INFO] Saved {portable_path(OUTPUT_MAT)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_CSV)}")


if __name__ == "__main__":
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    run_with_log(run, str(OUTPUT_LOG))
