"""Shared paired Monte-Carlo helpers for the Phase-1 control experiments."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
import torch

import eval_ber_vs_phasenoise as sim
from evaluation_utils import normalize_codebook_exact_global_es


def reset_random_state(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_normalized_codebook(
    path: Path, device: torch.device
) -> tuple[torch.Tensor, float, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    codebook = torch.load(path, map_location=device)
    if not torch.is_tensor(codebook):
        codebook = torch.as_tensor(codebook)
    if tuple(codebook.shape) != (sim.K, sim.M, sim.V):
        raise ValueError(
            f"{path}: shape {tuple(codebook.shape)} != {(sim.K, sim.M, sim.V)}"
        )
    codebook = sim.hardzero(
        codebook.to(device=device, dtype=sim.DTYPEC), sim.HARDZERO_THR
    )
    normalized, source_es, realized_es = normalize_codebook_exact_global_es(
        codebook, sim.GLOBAL_ES_TARGET
    )
    return normalized.to(sim.DTYPEC), source_es, realized_es


def evaluate_at_point(
    checkpoint_paths: list[Path],
    labels: list[str],
    *,
    seed: int,
    ebn0_db: float,
    epsilon: float,
    pn_sigma_step: float,
    nd_total: int,
    nd_chunk: int,
) -> dict:
    if len(checkpoint_paths) != len(labels):
        raise ValueError("checkpoint_paths and labels must have the same length")
    if nd_total <= 0 or nd_chunk <= 0:
        raise ValueError("nd_total and nd_chunk must be positive")

    reset_random_state(seed)
    device = torch.device(sim.DEVICE)
    loaded = [load_normalized_codebook(path, device) for path in checkpoint_paths]
    codebooks = [item[0] for item in loaded]
    source_global_es = np.array([item[1] for item in loaded], dtype=np.float64)
    normalized_global_es = np.array([item[2] for item in loaded], dtype=np.float64)
    factor_graphs = [sim.build_factor_graph(cb, thr=sim.FG_THR) for cb in codebooks]

    snr_db = ebn0_db + 10.0 * math.log10(sim.R_bits_per_RE)
    n0_physical = 1.0 / (10.0 ** (snr_db / 10.0))
    bits_per_ofdm = sim.V * sim.Q * sim.m_bits
    per_ofdm_errors = np.zeros((len(codebooks), nd_total), dtype=np.int16)

    print(
        f"[INFO] device={device}; Eb/N0={ebn0_db:g} dB; epsilon={epsilon:g}; "
        f"PN sigma={pn_sigma_step:.3e}; Nd={nd_total}"
    )
    print("[INFO] All codebooks use paired common random numbers.")
    nd_done = 0
    while nd_done < nd_total:
        nd_now = min(nd_chunk, nd_total - nd_done)
        x_vmq, h_rel, w_td, phi_td = sim.gen_shared_chunk(
            nd_now, n0_physical, device=device, sigma_val=pn_sigma_step
        )
        for index, codebook in enumerate(codebooks):
            res_users, user_ress = factor_graphs[index]
            y, h, x_flat = sim.simulate_chunk_shared(
                codebook,
                epsilon,
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
            per_ofdm_errors[index, nd_done:nd_done + nd_now] = (
                error_mask.reshape(sim.V, nd_now, sim.Q, sim.m_bits)
                .sum(dim=(0, 2, 3))
                .cpu()
                .numpy()
                .astype(np.int16)
            )
        nd_done += nd_now
        if nd_done % 5_000 == 0 or nd_done == nd_total:
            print(f"[INFO] Nd={nd_done}/{nd_total}")

    error_count = per_ofdm_errors.sum(axis=1, dtype=np.int64)
    total_bits = np.full(len(codebooks), nd_total * bits_per_ofdm, dtype=np.int64)
    ber = error_count / total_bits
    for label, errors, bits, value in zip(labels, error_count, total_bits, ber):
        print(f"[RESULT] {label}: errors={errors}, bits={bits}, BER={value:.9e}")

    return {
        "labels": labels,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "seed": np.int64(seed),
        "EbN0_dB": np.float64(ebn0_db),
        "epsilon": np.float64(epsilon),
        "pn_sigma_step": np.float64(pn_sigma_step),
        "Nd_used": np.int64(nd_total),
        "bits_per_ofdm": np.int64(bits_per_ofdm),
        "N0_physical": np.float64(n0_physical),
        "source_global_es": source_global_es,
        "normalized_global_es": normalized_global_es,
        "per_ofdm_error_count": per_ofdm_errors,
        "error_count": error_count,
        "total_bits": total_bits,
        "BER": ber,
        "decoder_model": "same no-tracking 10-iteration Log-MPA; N0_dec=N0_physical",
    }


def clustered_paired_difference(
    per_ofdm_errors: np.ndarray,
    baseline_index: int,
    comparison_index: int,
    bits_per_ofdm: int,
) -> dict:
    difference = (
        per_ofdm_errors[comparison_index].astype(np.float64)
        - per_ofdm_errors[baseline_index].astype(np.float64)
    )
    n = difference.size
    mean_ber_difference = difference.mean() / bits_per_ofdm
    standard_error = difference.std(ddof=1) / math.sqrt(n) / bits_per_ofdm
    if standard_error > 0:
        t_statistic = mean_ber_difference / standard_error
        p_two_sided = 2.0 * student_t.sf(abs(t_statistic), df=n - 1)
    else:
        t_statistic = 0.0
        p_two_sided = 1.0
    t_critical = student_t.ppf(0.975, df=n - 1)
    ci95 = np.array(
        [
            mean_ber_difference - t_critical * standard_error,
            mean_ber_difference + t_critical * standard_error,
        ],
        dtype=np.float64,
    )
    return {
        "per_ofdm_error_difference": difference.astype(np.int16),
        "mean_BER_difference": np.float64(mean_ber_difference),
        "standard_error_BER_difference": np.float64(standard_error),
        "mean_BER_difference_CI95": ci95,
        "paired_OFDM_t_statistic": np.float64(t_statistic),
        "paired_OFDM_two_sided_p": np.float64(p_two_sided),
        "paired_test_definition": "two-sided paired t-test over independent OFDM-symbol aggregate error-count differences; each symbol is one cluster",
    }
