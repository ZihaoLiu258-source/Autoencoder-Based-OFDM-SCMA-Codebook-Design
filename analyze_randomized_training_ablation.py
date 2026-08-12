"""Quantify energy and geometric differences between M1 codebooks."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPL_CONFIG_DIR = Path("tmp/matplotlib_m1").resolve()
_MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch

import eval_ber_vs_phasenoise as sim
from evaluation_utils import (
    codebook_energy_metrics,
    normalize_codebook_exact_global_es,
    portable_path,
)


ARTIFACT_DIR = Path("artifacts/m1_randomized_training")
FIXED_CHECKPOINT = Path("cb1_kmv.pt")
RANDOMIZED_CHECKPOINT = ARTIFACT_DIR / "randomized_cb_kmv.pt"
OUTPUT_JSON = ARTIFACT_DIR / "m1_codebook_metrics.json"
OUTPUT_CSV = ARTIFACT_DIR / "m1_codebook_metrics.csv"
OUTPUT_FIGURE_PDF = ARTIFACT_DIR / "m1_fixed_vs_randomized_constellations.pdf"
OUTPUT_FIGURE_PNG = ARTIFACT_DIR / "m1_fixed_vs_randomized_constellations.png"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_normalized(path: Path) -> tuple[torch.Tensor, float, float]:
    raw = torch.load(path, map_location="cpu")
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw)
    if tuple(raw.shape) != (sim.K, sim.M, sim.V):
        raise ValueError(f"{path}: shape {tuple(raw.shape)} != {(sim.K, sim.M, sim.V)}")
    raw = sim.hardzero(raw.to(torch.complex64), sim.HARDZERO_THR)
    normalized, source_es, realized_es = normalize_codebook_exact_global_es(raw, 1.0)
    return normalized.cpu(), source_es, realized_es


def two_shell_summary(values: np.ndarray) -> tuple[list[float], list[int]]:
    """Deterministic descriptive two-centroid summary, not a fitted model claim."""
    centers = np.quantile(values, [0.25, 0.75]).astype(np.float64)
    labels = np.zeros(values.size, dtype=np.int64)
    for _ in range(100):
        new_labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        new_centers = centers.copy()
        for cluster in range(2):
            if np.any(new_labels == cluster):
                new_centers[cluster] = values[new_labels == cluster].mean()
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels, centers = new_labels, new_centers
    order = np.argsort(centers)
    counts = np.array([(labels == cluster).sum() for cluster in range(2)])
    return centers[order].tolist(), counts[order].astype(int).tolist()


def local_superposition_distances(codebook: torch.Tensor) -> tuple[list[float], list[float]]:
    res_users, _ = sim.build_factor_graph(codebook, thr=sim.FG_THR)
    minima = []
    fifth_percentiles = []
    for resource, users in enumerate(res_users):
        points = []
        for symbols in itertools.product(range(sim.M), repeat=len(users)):
            point = sum(codebook[resource, symbol, user] for symbol, user in zip(symbols, users))
            points.append(point)
        points_t = torch.stack(points)
        d2 = (points_t[:, None] - points_t[None, :]).abs().square()
        d2.fill_diagonal_(float("inf"))
        nearest = d2.min(dim=1).values.cpu().numpy()
        minima.append(float(nearest.min()))
        fifth_percentiles.append(float(np.quantile(nearest, 0.05)))
    return minima, fifth_percentiles


def summarize(label: str, codebook: torch.Tensor, source_es: float, realized_es: float) -> dict:
    energy = codebook_energy_metrics(codebook)
    nonzero = codebook.abs()[codebook.abs() > sim.HARDZERO_THR].cpu().numpy()
    shell_centers, shell_counts = two_shell_summary(nonzero)
    min_d2, q05_d2 = local_superposition_distances(codebook)
    return {
        "label": label,
        "source_global_es": source_es,
        "normalized_global_es": realized_es,
        "per_user_average_energy": energy["per_user_average_energy"],
        "user_energy_max_min_ratio": energy["user_energy_max_min_ratio"],
        "peak_codeword_energy": energy["peak_codeword_energy"],
        "nonzero_magnitude_min": float(nonzero.min()),
        "nonzero_magnitude_q25": float(np.quantile(nonzero, 0.25)),
        "nonzero_magnitude_median": float(np.median(nonzero)),
        "nonzero_magnitude_q75": float(np.quantile(nonzero, 0.75)),
        "nonzero_magnitude_max": float(nonzero.max()),
        "nonzero_magnitude_mean": float(nonzero.mean()),
        "nonzero_magnitude_cv": float(nonzero.std(ddof=0) / nonzero.mean()),
        "two_shell_centers": shell_centers,
        "two_shell_counts": shell_counts,
        "resource_minimum_superposition_distance_squared": min_d2,
        "resource_q05_nearest_neighbor_distance_squared": q05_d2,
        "mean_resource_minimum_superposition_distance_squared": float(np.mean(min_d2)),
        "mean_resource_q05_nearest_neighbor_distance_squared": float(np.mean(q05_d2)),
    }


def write_figure(codebooks: list[torch.Tensor], labels: list[str]) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, sim.V))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.25), sharex=True, sharey=True)
    all_values = torch.cat([cb.flatten() for cb in codebooks])
    limit = float(torch.max(torch.stack([all_values.real.abs(), all_values.imag.abs()]))) * 1.12
    for axis, codebook, label in zip(axes, codebooks, labels):
        for user in range(sim.V):
            values = codebook[:, :, user].flatten()
            values = values[values.abs() > sim.HARDZERO_THR]
            axis.scatter(
                values.real.numpy(),
                values.imag.numpy(),
                s=36,
                alpha=0.82,
                color=colors[user],
                label=f"User {user + 1}",
            )
        axis.axhline(0.0, color="0.8", linewidth=0.8)
        axis.axvline(0.0, color="0.8", linewidth=0.8)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_title(label)
        axis.set_xlabel("In-phase")
        axis.grid(True, linewidth=0.35, alpha=0.4)
    axes[0].set_ylabel("Quadrature")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUTPUT_FIGURE_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    fixed_cb, fixed_source, fixed_realized = load_normalized(FIXED_CHECKPOINT)
    randomized_cb, randomized_source, randomized_realized = load_normalized(RANDOMIZED_CHECKPOINT)
    fixed_mask = fixed_cb.abs() > sim.HARDZERO_THR
    randomized_mask = randomized_cb.abs() > sim.HARDZERO_THR
    masks_identical = bool(torch.equal(fixed_mask, randomized_mask))

    inner = torch.sum(torch.conj(fixed_cb[fixed_mask]) * randomized_cb[fixed_mask])
    phase = torch.angle(inner)
    randomized_aligned = randomized_cb * torch.exp(-1j * phase)
    relative_frobenius_after_global_phase_alignment = float(
        torch.linalg.vector_norm(randomized_aligned - fixed_cb)
        / torch.linalg.vector_norm(fixed_cb)
    )

    rows = [
        summarize("Fixed-point training", fixed_cb, fixed_source, fixed_realized),
        summarize("Randomized training", randomized_cb, randomized_source, randomized_realized),
    ]
    comparison = {
        "support_masks_identical": masks_identical,
        "fixed_checkpoint_sha256": sha256(FIXED_CHECKPOINT),
        "randomized_checkpoint_sha256": sha256(RANDOMIZED_CHECKPOINT),
        "global_phase_alignment_rad": float(phase),
        "relative_frobenius_distance_after_global_phase_alignment": relative_frobenius_after_global_phase_alignment,
        "interpretation_note": "Two-shell centers are a deterministic descriptive clustering of active coefficient magnitudes; they are not a statistical model-selection result.",
    }
    OUTPUT_JSON.write_text(
        json.dumps({"codebooks": rows, "comparison": comparison}, indent=2),
        encoding="utf-8",
    )

    csv_fields = [
        "label", "source_global_es", "normalized_global_es",
        "user_energy_max_min_ratio", "peak_codeword_energy",
        "nonzero_magnitude_min", "nonzero_magnitude_q25", "nonzero_magnitude_median",
        "nonzero_magnitude_q75", "nonzero_magnitude_max", "nonzero_magnitude_mean",
        "nonzero_magnitude_cv", "mean_resource_minimum_superposition_distance_squared",
        "mean_resource_q05_nearest_neighbor_distance_squared",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})

    write_figure([fixed_cb, randomized_cb], [row["label"] for row in rows])
    for row in rows:
        shell_text = ", ".join(
            f"{center:.4f} (n={count})"
            for center, count in zip(row["two_shell_centers"], row["two_shell_counts"])
        )
        print(
            f"[RESULT] {row['label']} | user max/min={row['user_energy_max_min_ratio']:.4f} | "
            f"peak={row['peak_codeword_energy']:.4f} | magnitude CV={row['nonzero_magnitude_cv']:.4f} | "
            f"two-shell summary=[{shell_text}] | mean local MED^2="
            f"{row['mean_resource_minimum_superposition_distance_squared']:.6e}"
        )
    print(
        "[RESULT] support masks identical="
        f"{masks_identical}; phase-aligned relative Frobenius distance="
        f"{relative_frobenius_after_global_phase_alignment:.4f}"
    )
    print(f"[INFO] Saved {portable_path(OUTPUT_JSON)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_CSV)}")
    print(f"[INFO] Saved {portable_path(OUTPUT_FIGURE_PDF)}")


if __name__ == "__main__":
    main()
