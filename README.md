# Robust SCMA Codebook Design: A Hardware-Aware Autoencoder Approach

[![arXiv](https://img.shields.io/badge/arXiv-2606.22603-b31b1b.svg)](https://arxiv.org/abs/2606.22603)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official reproducibility package for the preprint by Zihao Liu, Zilong Liu, and Leila Musavian: [arXiv:2606.22603](https://arxiv.org/abs/2606.22603). Repository visibility remains under author control during the pre-submission embargo.

End-to-end learned SCMA codebook for an OFDM downlink, optimized with a differentiable Log-MPA detector under multipath fading, carrier frequency offset (CFO), and Wiener phase noise (PN).

## Repository guide

| Goal | Start here |
|---|---|
| Understand the method and assumptions | [Idea](#idea) and [What this model honestly does (and doesn't)](#what-this-model-honestly-does-and-doesnt) |
| Install and run the released model | [Quick start](#quick-start) |
| Reproduce training, controls, and BER sweeps | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Verify downloaded artifacts | `python validate_phase1_artifacts.py` |
| Check repository integrity without dependencies | `python check_repository.py` |
| Cite the work | [Citation](#citation) or [CITATION.cff](CITATION.cff) |

The repository separates the paper's fixed-point model from all additional controls. Ablation scripts and checkpoints use dedicated `artifacts/` subdirectories and never overwrite the released seed-0 checkpoint.

## Idea

Treat the (transmitter codebook → channel → receiver detector) chain as a differentiable graph. The SCMA codebook is a complex `torch.nn.Parameter`. Gradients flow back through:

- SCMA superposition over `K = 4` resources
- OFDM modulation and the frequency-domain multipath channel
- CFO and Wiener phase noise in the time domain
- AWGN
- A 10-iteration Log-MPA detector (logsumexp form, numerically stable end-to-end)

The simulator represents one downlink receiver. All `J = 6` SCMA layers are superimposed before transmission and therefore share that receiver's frequency-selective channel realization within an OFDM symbol. The channel changes independently between simulated OFDM symbols.

The implemented time-domain order is `common downlink channel → aggregate CFO → aggregate Wiener PN → AWGN → CP removal/FFT`. Thus, CFO/PN are receiver-side aggregate impairments applied after channel convolution. The aggregate PN layer is a stated modeling approximation; the code does not model separate user oscillators or per-user uplink offsets.

The capacity-based codebook of Zhang et al. is built directly into the training script and serves as both the initialization and the fixed quality reference. A squared-hinge term `(loss_tr − loss_base + m)_+²` encourages the trainable codebook to outperform this reference under identical messages, channels, AWGN, and PN realizations.

## System parameters

| Parameter | Value | Note |
|---|---|---|
| Users `J` | 6 | factor-graph degree 2 per user |
| Resources `K` | 4 | factor-graph degree 3 per resource |
| Order `M` | 4 | 2 bits per symbol |
| FFT `Nfft` | 1024 | |
| CP `Ncp` | 32 | covers the 20-sample channel spread |
| PDP | 8 taps, normalized | delays `[1, 2, 4, 6, 9, 11, 15, 20]` |
| CFO `ε` | 0.04 | fraction of subcarrier spacing (fixed) |
| PN | Wiener, `σ_step = 1e-3` rad / sample | |
| MPA iterations | 10 | |

During training, `Es` is projected to `1.0` every 4 steps via a 64-sample Monte-Carlo estimate. The final BER evaluators independently apply the exact analytic global-superposition normalization described below, eliminating normalization sampling error from codebook comparisons.

## Loss

```
L_total = (loss_tr_low  + λ_low  · hinge_low²)
        + 10 · (loss_tr_high + λ_high · hinge_high²)
        + λ_MED · L_MED
```

where `loss_tr = 0.3 · BCE + 0.8 · margin`, the margin term is `softplus((m₀ − signed_LLR)/t).mean()`, and `L_MED` is a soft-min over codeword-superposition distances at each resource (encourages well-separated effective constellations).

Each step samples a low/mid-SNR batch and a high-SNR batch. The first uses `[0, 15] dB` with probability 0.85 and `[16, 21] dB` with probability 0.15; the second uses `[22, 35] dB`. The high-SNR branch is weighted 10× to push the error floor down. Training uses Adam for 6000 steps with a batch size of 128 OFDM symbols per branch, an initial learning rate of `5e-4`, a factor-0.5 decay every 2500 steps, and gradient clipping at norm 3.

## What this model honestly does (and doesn't)

- **Perfect CSI** is fed to the MPA. Channel-estimation noise is not modeled.
- **The main released codebook uses fixed impairments:** CFO is fixed at `ε = 0.04` and PN at `σ_step = 1e-3` rad/sample during its training.
- **A separate randomized-training ablation is provided.** It samples `ε ~ U[0, 0.06]` and `σ_step ~ U[0, 3e-3]`; it does not redefine the main fixed-point method or its checkpoint.
- **Quasi-static block fading** per OFDM symbol; no time correlation across symbols.
- **Integer-sample tap delays.** No fractional delay or PDP leakage.
- **Representative-receiver downlink.** All superimposed SCMA layers share one channel response at that receiver; the implementation is not a multi-user uplink channel model.
- **Aggregate receiver-side CFO/PN.** Both rotations are applied after the common channel. Distributed transmitter/receiver PN is approximated by one post-channel Wiener process.

## Quick start

```bash
python -m pip install -r requirements.txt
python check_repository.py
python train_ofdm_scma.py
```

Python 3.10 or newer is recommended. A CUDA-capable GPU is strongly recommended for full training and Monte-Carlo sweeps; repository and released-artifact validation can run on CPU. Exact validated package versions, computational levels, expected files, and full commands are documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

Outputs (saved to the working directory):

- `codebook_e2e.pt` — shape `(V, K, M)`, complex64, end-to-end format
- `cb1_kmv.pt`     — shape `(K, M, V)`, complex64, simulator format
- The same two tensors as `.npy` for non-PyTorch consumers

Training always starts from the built-in Zhang capacity-based codebook returned by `build_zhang_codebook()`, which is also used as the hinge baseline. No external baseline checkpoint is required, so a clean run uses the same reference as the manuscript.

A snapshot of LOW / HIGH loss and hard BER is printed every 50 steps. `Ctrl-C` saves the current codebook safely before exit.

### M1 randomized-training ablation

The M1 scripts and outputs are deliberately separated from the original training path:

```bash
# Trains the additional model; never overwrites cb1_kmv.pt.
python train_ofdm_scma_randomized.py

# Paired three-point comparison at Eb/N0 = 30 dB.
python eval_randomized_training_ablation.py

# Exact-energy and constellation-geometry comparison.
python analyze_randomized_training_ablation.py
```

All additional checkpoints, histories, raw-count `.mat`/`.csv` files, logs, and geometry summaries are written below `artifacts/m1_randomized_training/`. The randomized model inherits the fixed model's initialization, 6000 steps, SNR sampler, batch size, Log-MPA iterations, optimizer, loss, and energy projection; only the CFO/PN parameter sampling differs.

### M3/M4 fixed-point controls

The remaining Phase-1 controls reuse the original fixed-point training entry point while isolating every new checkpoint from the repository-root seed-0 model:

```bash
# M3: change only lambda_MED from 1e-2 to zero.
python train_fixed_point_control.py --condition no_med --seed 0 --steps 6000 --output-dir artifacts/m3_no_med
python eval_no_med_ablation.py

# M4: retain the full loss and change only the independent training seed.
python train_fixed_point_control.py --condition full --seed 1 --steps 6000 --output-dir artifacts/m4_seed_check/seed1
python train_fixed_point_control.py --condition full --seed 2 --steps 6000 --output-dir artifacts/m4_seed_check/seed2
python eval_independent_seed_check.py

# Exact-energy and geometry audit for the original and three control checkpoints.
python analyze_phase1_controls.py
```

`train_fixed_point_control.py` resets NumPy/PyTorch/CUDA random states, records the exact condition and constants in `training_metadata.json`, and runs the unchanged original optimizer in the requested artifact directory. The M3/M4 evaluators use one representative training point, exact global-`Es` normalization, raw integer error counts, and paired common random numbers. No new BER curve or receiver is introduced.

The completed one-point M3 control gives full/no-MED BER `3.3083e-4`/`3.4372e-4` (`20,326`/`21,118` errors per `61,440,000` bits); removing Soft-MED raises BER by 3.90% in this paired run. The completed M4 check gives seed-0/1/2 BER `3.4652e-4`, `3.5002e-4`, and `3.5607e-4`, or `(3.5087 +/- 0.0483)e-4` as mean +/- sample standard deviation (CV 1.38%). These are narrow representative-point controls, not new full sweeps.

`analyze_phase1_controls.py` shows that the no-MED and independent-seed checkpoints keep the original sparse support and two descriptive magnitude shells near `0.257/0.774`. It also demonstrates why the geometry is not used as a BER surrogate: the no-MED checkpoint has a larger average hard local minimum distance squared while producing worse BER.

## Evaluation

Evaluation is a two-stage pipeline: a Python script runs the Monte-Carlo BER sweep and exports a `.mat` file, then a matching MATLAB script renders the publication figure. Each sweep benchmarks the trained codebook against five literature baselines (Deka, Li, Zhang, Zheng, PN-Resilient/Liu).

| Python sweep | MATLAB plot | Sweep axis | Other impairment | `.mat` file |
|---|---|---|---|---|
| `eval_ber_vs_phasenoise.py` | `plot_ber_vs_phasenoise.m` | PN `σ_step` ∈ [0, 3e-3] | CFO `ε` ∈ {0, 0.04} | `SCMA_CFO_Simulation_Results.mat` |
| `eval_ber_vs_cfo.py` | `plot_ber_vs_cfo.m` | CFO `ε` ∈ [0, 0.06] | PN `σ_step` ∈ {0, 2.4e-3} | `SCMA_SweepCFO_Simulation_Results.mat` |
| `eval_ber_vs_ebn0.py` | `plot_ber_vs_ebn0.m` | `E_b/N_0` ∈ [10, 34] dB | two conditions: ideal `(ε=0, σ=0)` and impaired `(ε=0.03, σ=1e-4)` | `SCMA_EbN0_Simulation_Results.mat` |

In every plot, the **solid** curve is the impaired condition and the **dashed** curve (same color, not in the legend) is the impairment-free reference, so the gap between the two shows each codebook's robustness directly.

The CFO and PN sweeps run at `E_b/N_0 = 30 dB`, simulate at least 2,000 and at most 100,000 OFDM symbols per point, and stop early only after all six codebooks have accumulated at least 2,000 aggregate bit errors. The Eb/N0 sweep uses 20,000 symbols at regular points and up to 50,000 above 30 dB, with a 10,000-error target for every codebook at the high-SNR points. One OFDM symbol carries `J * (Nfft/K) * log2(M) = 3072` aggregate user bits, so the largest budgets are `3.072e8` and `1.536e8` bits per point, respectively.

Evaluation uses seed 2025. Every OFDM symbol draws an independent eight-tap Rayleigh channel. Within each simulation chunk, all six codebooks receive identical messages, channel taps, AWGN, and PN paths; this common-random-number design reduces the variance of pairwise BER comparisons without changing any codebook's marginal channel distribution.

Each revised evaluator exports the raw `error_count`, `total_bits`, `Nd_used`, `seed`, and `stopping_reason` together with the empirical BER. A zero-error run is stored as empirical `BER = 0`, never as an artificial floor. Its separate `BER_95_upper = 3 / total_bits` value and `BER_is_upper_bound` flag implement the conventional 95% rule-of-three bound. The MATLAB scripts plot those zero-error points at the bound using downward-triangle markers.

Before each BER sweep, all codebooks are scaled using the same exact analytic criterion

```text
Es = (1/K) E_{m1,...,mJ}[ || sum_j c_j(mj) ||² ] = 1,
R  = J log2(M) / K = 3 bits/resource element,
N0 = 1 / (R · 10^(Eb/N0_dB/10)).
```

The expectation includes cross terms caused by nonzero per-user codebook means. It is evaluated analytically for independent equiprobable messages, rather than estimated with a finite Monte-Carlo batch. Time-domain AWGN uses variance `N0/Nfft`, which becomes variance `N0` per subcarrier after the unnormalized FFT. We preserve each baseline's original user-power allocation and equalize only total downlink superposition energy. Run `python analyze_codebook_energy.py` to reproduce the complete per-user and peak-codeword energy table in `codebook_energy_summary.csv`.

The intentionally mismatched Log-MPA uses `N0_dec = N0` while the physical simulator still applies time-domain CFO and Wiener PN. The lightweight check `python eval_decoder_variance_sensitivity.py` repeats one representative point with `N0_dec = gamma*N0`, `gamma in {0.5, 1, 2}`, for the proposed and strongest PN-resilient baseline; it writes a raw-count `.mat`, `.csv`, and `.log`.

Every evaluation also mirrors stdout and stderr to a complete log in the working directory: `eval_ber_vs_phasenoise.log`, `eval_ber_vs_cfo.log`, or `eval_ber_vs_ebn0.log`. The log records timestamps, script path, stopping reason, `Nd`, error count, total bits, empirical BER, and any zero-error upper bound for every simulated point.

```bash
# 1) Run the sweeps (Python) -> writes the .mat files
python eval_ber_vs_phasenoise.py
python eval_ber_vs_cfo.py
python eval_ber_vs_ebn0.py

# Reproduce assumption/fairness checks used in the response letter
python analyze_codebook_energy.py
python eval_decoder_variance_sensitivity.py

# 2) Render the figures (MATLAB) -> writes BER_vs_*.pdf / .eps
#    run plot_ber_vs_phasenoise.m, plot_ber_vs_cfo.m, plot_ber_vs_ebn0.m
```

> **Codebook path:** all three Python scripts require the released `cb1_kmv.pt` checkpoint (also written by `train_ofdm_scma.py`). If the file is missing or has the wrong shape, evaluation stops with an explicit error; it never substitutes a random codebook.

## Hardware

Developed on i5-13600KF + RTX 4070 (12 GB). At `BATCH_OFDMSYM = 128`, `Q_SUB = 256`, a full 6000-step run fits comfortably in 12 GB. Lower these knobs if you have less VRAM.

## Files

```
.
├── train_ofdm_scma.py            # main training script
├── train_ofdm_scma_randomized.py # isolated M1 randomized-training script
├── train_fixed_point_control.py   # isolated M3/M4 fixed-point control wrapper
├── eval_ber_vs_phasenoise.py     # BER vs PN sigma sweep  -> .mat
├── eval_ber_vs_cfo.py            # BER vs CFO sweep        -> .mat
├── eval_ber_vs_ebn0.py           # BER vs Eb/N0 sweep      -> .mat
├── analyze_codebook_energy.py     # exact Es/user/peak energy audit -> .csv
├── eval_decoder_variance_sensitivity.py # one-point N0_dec audit -> .mat/.csv/.log
├── eval_randomized_training_ablation.py # M1 fixed/randomized three-point audit
├── analyze_randomized_training_ablation.py # M1 energy/geometry audit
├── eval_no_med_ablation.py        # M3 one-point paired Soft-MED ablation
├── eval_independent_seed_check.py # M4 three-training-seed one-point check
├── phase1_control_utils.py        # shared paired Monte-Carlo helpers
├── analyze_phase1_controls.py     # M3/M4 exact-energy and geometry audit
├── validate_phase1_artifacts.py   # fail-fast checkpoint/MAT/log integrity audit
├── check_repository.py            # dependency-free repository integrity audit
├── evaluation_utils.py            # exact normalization and BER/log helpers
├── plot_ber_vs_phasenoise.m      # MATLAB: render PN sweep figure
├── plot_ber_vs_cfo.m             # MATLAB: render CFO sweep figure
├── plot_ber_vs_ebn0.m            # MATLAB: render Eb/N0 figure
├── codebook_e2e.pt / .npy        # trained codebook, (V, K, M)
├── cb1_kmv.pt / .npy             # same codebook in (K, M, V) layout
├── artifacts/m1_randomized_training/ # isolated M1 checkpoints and evidence
├── artifacts/m3_no_med/              # isolated no-MED checkpoint and evidence
├── artifacts/m4_seed_check/          # independent-seed checkpoints and evidence
├── artifacts/phase1_summary/         # cross-control feature audit
├── REPRODUCIBILITY.md                # complete reproduction and validation guide
├── CONTRIBUTING.md                   # contribution and artifact standards
├── CITATION.cff                      # GitHub-readable citation metadata
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).

## References

- H. Nikopour and H. Baligh, "Sparse Code Multiple Access," *PIMRC 2013*.
- M. Taherzadeh, H. Nikopour, A. Bayesteh, and H. Baligh, "SCMA Codebook Design," *VTC-Fall 2014*.
- F. Wei and W. Chen, "Message Passing Receiver Design for Uplink Grant-Free SCMA," *IEEE Wireless Commun. Lett.*

## Citation

GitHub can read the repository's [CITATION.cff](CITATION.cff) directly. The associated preprint citation is:

```bibtex
@misc{liu2026robust,
  title         = {Robust SCMA Codebook Design: A Hardware-Aware Autoencoder Approach},
  author        = {Liu, Zihao and Liu, Zilong and Musavian, Leila},
  year          = {2026},
  eprint        = {2606.22603},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT},
  doi           = {10.48550/arXiv.2606.22603}
}
```
