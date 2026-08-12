# Artifact layout

Generated evidence is separated from executable source code.

- `main/checkpoints/`: released fixed-point seed-0 codebooks in PyTorch and NumPy formats.
- `main/sweeps/`: full BER sweep MAT files and complete evaluator logs.
- `main/diagnostics/`: energy-normalization and decoder-variance checks.
- `main/figures/`: publication-ready BER and constellation figures.
- `phase1/`: isolated M1 randomized-training, M3 Soft-MED, M4 seed, and cross-control evidence.

Regenerate these files with the commands in [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md); do not edit numerical results manually.
