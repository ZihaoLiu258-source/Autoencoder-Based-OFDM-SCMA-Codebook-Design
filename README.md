# Autoencoder-Based OFDM-SCMA Codebook Design

End-to-end learned SCMA codebook for an OFDM downlink, optimized jointly with a differentiable Log-MPA detector under multipath fading, carrier frequency offset (CFO), and Wiener phase noise (PN).

## Idea

Treat the (transmitter codebook → channel → receiver detector) chain as a differentiable graph. The SCMA codebook is a complex `torch.nn.Parameter`. Gradients flow back through:

- SCMA superposition over `K = 4` resources
- OFDM modulation and the frequency-domain multipath channel
- CFO and Wiener phase noise in the time domain
- AWGN
- A 10-iteration Log-MPA detector (logsumexp form, numerically stable end-to-end)

A fixed reference codebook (`base`) acts as a quality floor: a squared-hinge term `(loss_tr − loss_base + m)_+²` forces the trainable codebook to beat the reference on identical channel / noise / PN draws.

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

`Es` is renormalized to `1.0` every 4 steps via a 64-sample Monte-Carlo estimate.

## Loss

```
L_total = (loss_tr_low  + λ_low  · hinge_low²)
        + 10 · (loss_tr_high + λ_high · hinge_high²)
        + λ_MED · L_MED
```

where `loss_tr = 0.3 · BCE + 0.8 · margin`, the margin term is `softplus((m₀ − signed_LLR)/t).mean()`, and `L_MED` is a soft-min over codeword-superposition distances at each resource (encourages well-separated effective constellations).

EbN0 is sampled per step from a low band `[0, 15] dB` and a high band `[22, 35] dB` separately so the optimizer sees both regimes. The high-SNR branch is weighted 10× to push the error floor down.

## What this model honestly does (and doesn't)

- **Perfect CSI** is fed to the MPA. Channel-estimation noise is not modeled.
- **CFO is fixed** at `ε = 0.04`. The codebook adapts to this one value; for deployment-grade robustness, randomize `ε` per step.
- **PN σ is fixed.** Same caveat.
- **Quasi-static block fading** per OFDM symbol; no time correlation across symbols.
- **Integer-sample tap delays.** No fractional delay or PDP leakage.

## Quick start

```bash
pip install -r requirements.txt
python train_ofdm_scma.py
```

Outputs (saved to the working directory):

- `codebook_e2e.pt` — shape `(V, K, M)`, complex64, end-to-end format
- `cb1_kmv.pt`     — shape `(K, M, V)`, complex64, simulator format
- The same two tensors as `.npy` for non-PyTorch consumers

If a file named `deka_codebook.pt` exists in the working directory, the script enters fine-tune mode: it uses that codebook both as the initialization and as the hinge baseline. Otherwise it starts from the analytical SCMA codebook in `build_base_codebook()`.

A snapshot of LOW / HIGH loss and hard BER is printed every 50 steps. `Ctrl-C` saves the current codebook safely before exit.

## Evaluation

Three independent scripts sweep one impairment axis each and export a `.mat` file for MATLAB plotting. Each script benchmarks the trained codebook against five literature baselines (Deka, Li, Zhang, Zheng, LPCB-PN43).

| Script | Sweep axis | Parameter | Output `.mat` |
|---|---|---|---|
| `eval_ber_vs_phasenoise.py` | PN `σ_step` ∈ [0, 3e-3] | CFO `ε` ∈ {0, 0.04} | `SCMA_CFO_Simulation_Results.mat` |
| `eval_ber_vs_cfo.py` | CFO `ε` ∈ [0, 0.06] | PN `σ_step` ∈ {0, 2.4e-3} | `SCMA_SweepCFO_Simulation_Results.mat` |
| `eval_ber_vs_ebn0.py` | `E_b/N_0` ∈ [10, 34] dB | CFO `ε = 0.04`, PN `σ = 5e-4` | `SCMA_EbN0_Simulation_Results.mat` |

The first two run at a fixed `E_b/N_0 = 30 dB` and a Monte-Carlo budget of `Nd_total = 100000` symbols with early-stop on error count. The Eb/N0 sweep uses `Nd_total = 20000` per point and tightens the budget above 30 dB.

```bash
python eval_ber_vs_phasenoise.py
python eval_ber_vs_cfo.py
python eval_ber_vs_ebn0.py
```

> **Codebook path:** all three scripts load the trained codebook from `cb1_kmv.pt` (the file `train_ofdm_scma.py` writes). If that file is missing, the loader falls back to a random codebook and the resulting BER curves are meaningless — run training first, or `git pull` the example codebook included in this repo.

## Hardware

Developed on i5-13600KF + RTX 4070 (12 GB). At `BATCH_OFDMSYM = 128`, `Q_SUB = 256`, a full 6000-step run fits comfortably in 12 GB. Lower these knobs if you have less VRAM.

## Files

```
.
├── train_ofdm_scma.py            # main training script
├── eval_ber_vs_phasenoise.py     # BER vs PN sigma sweep
├── eval_ber_vs_cfo.py            # BER vs CFO sweep
├── eval_ber_vs_ebn0.py           # BER vs Eb/N0 sweep
├── codebook_e2e.pt / .npy        # trained codebook, (V, K, M)
├── cb1_kmv.pt / .npy             # same codebook in (K, M, V) layout
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

If you find this useful in academic work, please cite this repository.
