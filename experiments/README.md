# Experiments

- `phase1/` contains the randomized-training, Soft-MED, and independent-seed controls.
- `diagnostics/` contains lightweight energy-normalization and decoder-variance checks.

Run scripts from the repository root with module syntax:

```bash
python -m experiments.phase1.eval_no_med_ablation
python -m experiments.diagnostics.analyze_codebook_energy
```
