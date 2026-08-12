"""Regenerate the three Fig. 3 codebook-geometry panels.

All panels use zero-based resource indices (0--3), matching the Python
implementation and the revised manuscript. The script reads the released
proposed checkpoint and reuses the exact baseline definitions from the BER
evaluator; it does not alter or retrain any codebook.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval_ber_vs_cfo import build_cb2_matlab_fullq, build_cb_lpcb_pn43


PAPER_DIR = ROOT.parent / "Robust SCMA Codebook design"
COLORS = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE")
MARKERS = ("o", "s", "^", "D", "v", "*")


def plot_codebook(cb_kmv: np.ndarray, output_name: str) -> None:
    if cb_kmv.shape != (4, 4, 6):
        raise ValueError(f"Expected codebook shape (4, 4, 6), got {cb_kmv.shape}")

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.5), constrained_layout=True)
    active = np.abs(cb_kmv) > 1e-10
    extent = max(1.0, float(np.max(np.abs(cb_kmv[active]))) * 1.18)

    for resource, ax in enumerate(axes):
        for user in range(cb_kmv.shape[2]):
            points = cb_kmv[resource, :, user]
            points = points[np.abs(points) > 1e-10]
            if points.size == 0:
                continue
            ax.plot(
                points.real,
                points.imag,
                linestyle="none",
                marker=MARKERS[user],
                markersize=7.5,
                markerfacecolor="none",
                markeredgewidth=1.4,
                color=COLORS[user],
            )
        ax.set_title(f"Resource {resource}", fontsize=11)
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="major", color="0.65", linewidth=0.6)
        ax.minorticks_on()
        ax.grid(True, which="minor", color="0.85", linestyle=":", linewidth=0.4)
        ax.tick_params(direction="in", labelsize=9)
        ax.set_xlabel("In-Phase", fontsize=10)
        if resource == 0:
            ax.set_ylabel("Quadrature", fontsize=10)

    output_path = PAPER_DIR / output_name
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    proposed = np.load(ROOT / "cb1_kmv.npy")
    deka = build_cb2_matlab_fullq(device=torch.device("cpu")).cpu().numpy()
    pn_resilient = build_cb_lpcb_pn43(device=torch.device("cpu")).cpu().numpy()

    plot_codebook(proposed, "Constellation_CB1_Proposed.pdf")
    plot_codebook(deka, "Constellation_CB2_Deka.pdf")
    plot_codebook(pn_resilient, "Constellation_CB_Provided.pdf")


if __name__ == "__main__":
    main()
