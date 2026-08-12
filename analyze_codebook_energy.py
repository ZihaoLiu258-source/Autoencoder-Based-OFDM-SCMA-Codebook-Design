"""Generate exact energy statistics for all evaluated SCMA codebooks."""

import csv
from pathlib import Path

import torch

import eval_ber_vs_phasenoise as sim
from evaluation_utils import codebook_energy_metrics, normalize_codebook_exact_global_es


OUTPUT_CSV = Path("codebook_energy_summary.csv")


def build_codebooks(device):
    proposed = sim.hardzero(
        sim.load_trained_cb1_kmv("cb1_kmv.pt", device=device),
        thr=sim.HARDZERO_THR,
    )
    return [
        ("Proposed", proposed),
        ("Deka", sim.build_cb2_matlab_fullq(device=device)),
        ("Li", sim.build_cb_xudong_li(device=device)),
        ("Zhang", sim.build_cb_shutian_zhang(device=device)),
        ("Zheng", sim.build_cb_screenshot_new(device=device)),
        ("PN-resilient", sim.build_cb_lpcb_pn43(device=device)),
    ]


def main():
    device = torch.device(sim.DEVICE)
    rows = []
    for label, raw_codebook in build_codebooks(device):
        normalized, source_es, realized_es = normalize_codebook_exact_global_es(
            raw_codebook, target_es=sim.GLOBAL_ES_TARGET
        )
        metrics = codebook_energy_metrics(normalized)
        row = {
            "codebook": label,
            "source_global_es": source_es,
            "normalized_global_es": realized_es,
            **{
                f"user_{index + 1}_average_energy": value
                for index, value in enumerate(metrics["per_user_average_energy"])
            },
            "user_energy_min": metrics["user_energy_min"],
            "user_energy_max": metrics["user_energy_max"],
            "user_energy_max_min_ratio": metrics["user_energy_max_min_ratio"],
            "peak_codeword_energy": metrics["peak_codeword_energy"],
        }
        rows.append(row)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        users = ", ".join(
            f"{row[f'user_{index}_average_energy']:.4f}" for index in range(1, 7)
        )
        print(
            f"{row['codebook']:>12s} | Es={row['normalized_global_es']:.6f} | "
            f"users=[{users}] | peak={row['peak_codeword_energy']:.4f} | "
            f"max/min={row['user_energy_max_min_ratio']:.3f}"
        )
    print(f"[INFO] Saved {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
