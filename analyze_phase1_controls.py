"""Compare the geometry of the original, no-MED, and independent-seed codebooks."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_DIR = Path("tmp/matplotlib_phase1").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch

import eval_ber_vs_phasenoise as sim
from analyze_randomized_training_ablation import load_normalized, summarize
from evaluation_utils import portable_path


ARTIFACT_DIR = Path("artifacts/phase1_summary")
CHECKPOINTS = [
    Path("cb1_kmv.pt"),
    Path("artifacts/m3_no_med/cb1_kmv.pt"),
    Path("artifacts/m4_seed_check/seed1/cb1_kmv.pt"),
    Path("artifacts/m4_seed_check/seed2/cb1_kmv.pt"),
]
LABELS = ["Full loss, seed 0", "No Soft-MED, seed 0", "Full loss, seed 1", "Full loss, seed 2"]
OUTPUT_JSON = ARTIFACT_DIR / "phase1_codebook_metrics.json"
OUTPUT_CSV = ARTIFACT_DIR / "phase1_codebook_metrics.csv"
OUTPUT_FIGURE_PDF = ARTIFACT_DIR / "phase1_codebook_constellations.pdf"
OUTPUT_FIGURE_PNG = ARTIFACT_DIR / "phase1_codebook_constellations.png"


def soft_med_auxiliary_value(codebook: torch.Tensor, tau: float = 0.15) -> float:
    """Evaluate the implemented Soft-MED penalty on an exactly normalized codebook."""
    res_users, _ = sim.build_factor_graph(codebook, thr=sim.FG_THR)
    penalties = []
    for resource, users in enumerate(res_users):
        c1, c2, c3 = (codebook[resource, :, user] for user in users)
        points = (c1[:, None, None] + c2[None, :, None] + c3[None, None, :]).reshape(-1)
        distance_squared = (points[:, None] - points[None, :]).abs().square()
        distance_squared.fill_diagonal_(1.0e9)
        soft_minimum = -tau * torch.logsumexp((-distance_squared / tau).reshape(-1), dim=0)
        penalties.append(-soft_minimum)
    return float(torch.stack(penalties).mean())


def phase_aligned_distance(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    mask = reference.abs() > sim.HARDZERO_THR
    if not torch.equal(mask, candidate.abs() > sim.HARDZERO_THR):
        raise ValueError("Codebooks do not share the same sparse support")
    inner = torch.sum(torch.conj(reference[mask]) * candidate[mask])
    phase = torch.angle(inner)
    aligned = candidate * torch.exp(-1j * phase)
    distance = torch.linalg.vector_norm(aligned - reference) / torch.linalg.vector_norm(reference)
    return float(phase), float(distance)


def write_figure(codebooks: list[torch.Tensor]) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, sim.V))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.0), sharex=True, sharey=True)
    limit = max(float(codebook.abs().max()) for codebook in codebooks) * 1.13
    for axis, codebook, label in zip(axes.flat, codebooks, LABELS):
        for user in range(sim.V):
            values = codebook[:, :, user].flatten()
            values = values[values.abs() > sim.HARDZERO_THR]
            axis.scatter(values.real, values.imag, s=28, alpha=0.82, color=colors[user], label=f"User {user + 1}")
        axis.axhline(0.0, color="0.8", linewidth=0.8)
        axis.axvline(0.0, color="0.8", linewidth=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_title(label)
        axis.grid(True, linewidth=0.35, alpha=0.4)
    for axis in axes[-1, :]:
        axis.set_xlabel("In-phase")
    for axis in axes[:, 0]:
        axis.set_ylabel("Quadrature")
    handles, legend_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [load_normalized(path) for path in CHECKPOINTS]
    codebooks = [item[0] for item in loaded]
    rows = [
        summarize(label, codebook, source_es, realized_es)
        for label, (codebook, source_es, realized_es) in zip(LABELS, loaded)
    ]
    for row, codebook in zip(rows, codebooks):
        row["implemented_soft_med_penalty_at_tau_0p15"] = soft_med_auxiliary_value(codebook)
    comparisons = []
    for label, codebook in zip(LABELS, codebooks):
        phase, distance = phase_aligned_distance(codebooks[0], codebook)
        comparisons.append(
            {
                "candidate": label,
                "reference": LABELS[0],
                "global_phase_alignment_rad": phase,
                "relative_frobenius_distance_after_global_phase_alignment": distance,
            }
        )

    full_seed_rows = [rows[index] for index in (0, 2, 3)]
    stability = {}
    for key in (
        "user_energy_max_min_ratio",
        "peak_codeword_energy",
        "mean_resource_minimum_superposition_distance_squared",
        "mean_resource_q05_nearest_neighbor_distance_squared",
    ):
        values = np.array([row[key] for row in full_seed_rows], dtype=np.float64)
        stability[key] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    payload = {
        "codebooks": rows,
        "comparisons_to_original": comparisons,
        "full_loss_three_seed_stability": stability,
        "notes": [
            "All checkpoints are hard-zeroed and exactly normalized to global Es=1 before comparison.",
            "Two-shell centers are deterministic descriptive clusters of active magnitudes, not model-selection claims.",
            "The local superposition distances are noiseless resource-wise geometric diagnostics, not BER surrogates.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "label", "source_global_es", "normalized_global_es", "user_energy_max_min_ratio",
        "peak_codeword_energy", "nonzero_magnitude_cv",
        "implemented_soft_med_penalty_at_tau_0p15",
        "mean_resource_minimum_superposition_distance_squared",
        "mean_resource_q05_nearest_neighbor_distance_squared",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    write_figure(codebooks)

    for row, comparison in zip(rows, comparisons):
        centers = ", ".join(f"{value:.4f}" for value in row["two_shell_centers"])
        print(
            f"[RESULT] {row['label']}: phase-aligned distance={comparison['relative_frobenius_distance_after_global_phase_alignment']:.4f}; "
            f"shells=[{centers}]; energy ratio={row['user_energy_max_min_ratio']:.4f}; "
            f"local min d2={row['mean_resource_minimum_superposition_distance_squared']:.6e}; "
            f"Soft-MED penalty={row['implemented_soft_med_penalty_at_tau_0p15']:.6f}"
        )
    print(f"[INFO] Saved {portable_path(OUTPUT_JSON)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_CSV)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_FIGURE_PDF)}")


if __name__ == "__main__":
    main()
