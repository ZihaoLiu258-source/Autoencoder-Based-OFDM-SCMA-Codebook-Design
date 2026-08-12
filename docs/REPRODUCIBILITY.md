# Reproducibility guide

This guide separates fast integrity checks from GPU-intensive training and long Monte-Carlo evaluations. Run commands from the repository root.

## 1. Environment

Minimum supported packages are listed in `requirements.txt`. The revision evidence was validated with:

| Component | Validated version |
|---|---:|
| Python | 3.13.15 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.4.4 |
| SciPy | 1.18.0 |
| Matplotlib | 3.11.1 |
| CUDA runtime reported by PyTorch | 13.0 |

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

MATLAB is only required to regenerate the publication-ready BER figures from the exported `.mat` files. Training and Monte-Carlo evaluation use PyTorch; the released artifact checks require PyTorch, NumPy, and SciPy.

## 2. Validation levels

| Level | Command | GPU | Purpose |
|---|---|---:|---|
| Repository audit | `python -m tools.check_repository` | No | Syntax, JSON, file-size, required-file, and path-portability checks |
| Artifact audit | `python -m tools.validate_phase1_artifacts` | No | Checkpoint shapes/support/hashes and exact `BER = errors / bits` identities |
| Lightweight numerical check | `python -m experiments.diagnostics.eval_decoder_variance_sensitivity` | Recommended | Representative decoder-variance sensitivity run |
| Full evaluation | three `eval_ber_vs_*.py` scripts | Yes | Regenerate all reported BER sweeps and raw counts |
| Full retraining | training commands below | Yes | Regenerate main and ablation checkpoints |

The repository audit is intentionally dependency-free and suitable for local or continuous-integration use. The artifact audit loads the released checkpoints and `.mat` files but does not rerun Monte-Carlo simulations.

## 3. Main model

```bash
python train_ofdm_scma.py
```

This writes the seed-0 fixed-impairment model in two layouts and two formats:

- `artifacts/main/checkpoints/codebook_e2e.pt` and `.npy`: `(V, K, M)`
- `artifacts/main/checkpoints/cb1_kmv.pt` and `.npy`: `(K, M, V)`

Training starts from the built-in Zhang codebook and does not require an external baseline checkpoint.

## 4. Phase-1 control experiments

### M1: randomized impairment training

```bash
python -m experiments.phase1.train_ofdm_scma_randomized
python -m experiments.phase1.eval_randomized_training_ablation
python -m experiments.phase1.analyze_randomized_training_ablation
```

Outputs are isolated under `artifacts/phase1/m1_randomized_training/`; the main checkpoint is never overwritten.

### M3: remove only the Soft-MED term

```bash
python -m experiments.phase1.train_fixed_point_control --condition no_med --seed 0 --steps 6000 --output-dir artifacts/phase1/m3_no_med
python -m experiments.phase1.eval_no_med_ablation
```

### M4: independent training seeds

```bash
python -m experiments.phase1.train_fixed_point_control --condition full --seed 1 --steps 6000 --output-dir artifacts/phase1/m4_seed_check/seed1
python -m experiments.phase1.train_fixed_point_control --condition full --seed 2 --steps 6000 --output-dir artifacts/phase1/m4_seed_check/seed2
python -m experiments.phase1.eval_independent_seed_check
python -m experiments.phase1.analyze_phase1_controls
```

Every control training directory contains metadata, complete logs, PyTorch checkpoints, and NumPy exports. Every evaluation package contains raw integer error counts, total bits, empirical BER, a complete log, and MATLAB-compatible data.

## 5. Full BER evaluation

```bash
python eval_ber_vs_phasenoise.py
python eval_ber_vs_cfo.py
python eval_ber_vs_ebn0.py
```

| Evaluation | Primary output | MATLAB renderer |
|---|---|---|
| PN sweep | `artifacts/main/sweeps/SCMA_CFO_Simulation_Results.mat` | `plots/matlab/plot_ber_vs_phasenoise.m` |
| CFO sweep | `artifacts/main/sweeps/SCMA_SweepCFO_Simulation_Results.mat` | `plots/matlab/plot_ber_vs_cfo.m` |
| Eb/N0 sweep | `artifacts/main/sweeps/SCMA_EbN0_Simulation_Results.mat` | `plots/matlab/plot_ber_vs_ebn0.m` |

The evaluators use paired common random numbers across codebooks, exact analytic global-superposition energy normalization, raw error counts, and a separate rule-of-three upper bound for zero-error points. See the main README for the stopping budgets and model assumptions.

## 6. Released control results

The one-point controls are deliberately small because they address sensitivity questions rather than introduce new BER curves.

| Control | Released result |
|---|---|
| M3 Soft-MED ablation | Full/no-MED BER: `3.3083e-4` / `3.4372e-4`; no-MED is 3.90% higher in the paired run |
| M4 independent seeds | Seed-0/1/2 BER: `3.4652e-4`, `3.5002e-4`, `3.5607e-4`; CV 1.38% |

These values are empirical results at one representative operating point, not universal performance guarantees.

## 7. Artifact integrity

Do not edit generated `.mat`, `.pt`, `.npy`, or experiment logs manually. If a run is repeated, keep the command, seed, stopping reason, raw counts, and generated metadata together. `python -m tools.validate_phase1_artifacts` fails if the released checkpoint hashes, support masks, BER identities, or completion markers are inconsistent.
