"""Lightweight mismatched-decoder variance sensitivity at one operating point."""

import csv
import math
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

import eval_ber_vs_phasenoise as sim
from evaluation_utils import normalize_codebook_exact_global_es, portable_path, run_with_log


SEED = 2026
EBN0_DB = 30
EPSILON = 0.04
PN_SIGMA_STEP = 1.0e-3
GAMMA_VALUES = np.array([0.5, 1.0, 2.0], dtype=np.float64)
ND_TOTAL = 20000
ND_CHUNK = 1000

OUTPUT_MAT = Path("decoder_variance_sensitivity.mat")
OUTPUT_CSV = Path("decoder_variance_sensitivity.csv")


def run():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(sim.DEVICE)
    proposed_raw = sim.hardzero(
        sim.load_trained_cb1_kmv("cb1_kmv.pt", device=device),
        thr=sim.HARDZERO_THR,
    )
    baseline_raw = sim.build_cb_lpcb_pn43(device=device)
    labels = ["Proposed", "PN-resilient"]
    codebooks = [
        normalize_codebook_exact_global_es(cb, sim.GLOBAL_ES_TARGET)[0].to(sim.DTYPEC)
        for cb in (proposed_raw, baseline_raw)
    ]
    factor_graphs = [sim.build_factor_graph(cb, thr=sim.FG_THR) for cb in codebooks]

    snr_db = EBN0_DB + 10.0 * math.log10(sim.R_bits_per_RE)
    n0_physical = 1.0 / (10.0 ** (snr_db / 10.0))

    error_count = np.zeros((len(codebooks), len(GAMMA_VALUES)), dtype=np.int64)
    total_bits = np.zeros_like(error_count)

    nd_done = 0
    while nd_done < ND_TOTAL:
        nd_now = min(ND_CHUNK, ND_TOTAL - nd_done)
        x_vmq, h_rel, w_td, phi_td = sim.gen_shared_chunk(
            nd_now, n0_physical, device=device, sigma_val=PN_SIGMA_STEP
        )

        for icb, codebook in enumerate(codebooks):
            res_users, user_ress = factor_graphs[icb]
            y, h, x_flat = sim.simulate_chunk_shared(
                codebook, EPSILON, x_vmq, h_rel, w_td, phi_td, device=device
            )
            bits_tx = sim.symbols_to_bits(x_flat, m_bits=sim.m_bits)
            n_bits = int(bits_tx.numel())

            for igamma, gamma in enumerate(GAMMA_VALUES):
                llr = sim.scmadec_logmpa_llr(
                    y,
                    codebook,
                    h,
                    float(gamma * n0_physical),
                    sim.Nit,
                    res_users,
                    user_ress,
                )
                bits_hat = sim.llr_to_bits(llr, V=sim.V, m_bits=sim.m_bits)
                error_count[icb, igamma] += int((bits_tx != bits_hat).sum().item())
                total_bits[icb, igamma] += n_bits

        nd_done += nd_now
        if nd_done % 5000 == 0 or nd_done == ND_TOTAL:
            print(f"[INFO] Nd={nd_done}/{ND_TOTAL}")

    ber = error_count / total_bits
    sio.savemat(
        OUTPUT_MAT,
        {
            "labels": labels,
            "gamma": GAMMA_VALUES,
            "error_count": error_count,
            "total_bits": total_bits,
            "BER": ber,
            "Nd_used": np.int64(ND_TOTAL),
            "seed": np.int64(SEED),
            "EbN0_dB": np.float64(EBN0_DB),
            "epsilon": np.float64(EPSILON),
            "pn_sigma_step": np.float64(PN_SIGMA_STEP),
            "N0_physical": np.float64(n0_physical),
            "decoder_variance_definition": "N0_dec = gamma * N0_physical",
        },
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["codebook", "gamma", "error_count", "total_bits", "BER"])
        for icb, label in enumerate(labels):
            for igamma, gamma in enumerate(GAMMA_VALUES):
                writer.writerow(
                    [label, gamma, error_count[icb, igamma], total_bits[icb, igamma], ber[icb, igamma]]
                )

    for icb, label in enumerate(labels):
        values = " | ".join(
            f"gamma={gamma:g}: errors={error_count[icb, igamma]}, "
            f"bits={total_bits[icb, igamma]}, BER={ber[icb, igamma]:.3e}"
            for igamma, gamma in enumerate(GAMMA_VALUES)
        )
        print(f"[RESULT] {label} | {values}")
    print(f"[INFO] Saved {portable_path(OUTPUT_MAT)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_CSV)}")


if __name__ == "__main__":
    run_with_log(run, "eval_decoder_variance_sensitivity.log")
