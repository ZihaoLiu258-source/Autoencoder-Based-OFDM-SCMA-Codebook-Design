

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import numpy as np
import torch
import torch.nn.functional as F

# =========================
# 0) Repro / device
# =========================
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device = {DEVICE}")
DTYPEC = torch.complex64
DTYPEF = torch.float32
# TF32 (Ampere+), can speed up some float ops; FFT unaffected but overall helps a bit
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# =========================
# 1) Dimensions
# =========================
J, K, M = 6, 4, 4
V = J
m_bits = int(math.log2(M))
R_bits_per_RE = (m_bits * V) / K

# OFDM
Nfft = 1024
Ncp = 32
NsymTD = Nfft + Ncp
Q = Nfft // K
assert Nfft % K == 0

# 8-path PDP (match sim)
delays = torch.tensor([1, 2, 4, 6, 9, 11, 15, 20], dtype=torch.int64, device=DEVICE)
powers = torch.tensor([0.36, 0.24, 0.15, 0.10, 0.06, 0.04, 0.025, 0.017], dtype=torch.float32, device=DEVICE)
powers = powers / powers.sum()
d_rel = delays - delays[0]
Lh = int(d_rel.max().item()) + 1

assert Ncp >= (Lh - 1), f"CP too short: Ncp={Ncp}, need >= {Lh - 1} for circular conv model."

# Precompute sqrt(powers/2) for taps generation
tap_amp = torch.sqrt(powers / 2.0)  # (8,)
NUM_TAPS = len(delays)
TAP_IDX = d_rel.to(torch.long)  # device tensor; avoids per-tap .item() sync in batch builder

# =========================
# 2) Training hyperparams
# =========================
NUM_STEPS = 6000

# Memory / throughput knobs
# Q = Nfft/K = 256, so Q_SUB=256 means use every subcarrier. Lower it (64/128) if memory-bound.
Q_SUB = 256
# 128 fits comfortably in 12GB (4070). 192/256 is also fine if you have headroom.
BATCH_OFDMSYM = 128

# MPA iterations
NIT_MPA_LOW = 10
NIT_MPA_HIGH = 10
# Optional random Nit: enable USE_RANDOM_NIT to sample from NIT_SET each step.
USE_RANDOM_NIT = False
NIT_SET = [6, 8, 10]

LR = 1.0e-4
PRINT_EVERY = 50
CLIP_GRAD_NORM = 3.0

# CFO fixed
EPS_FIXED = 0.04

# Phase Noise
ENABLE_PN = True
PN_SIGMA_STEP_RAD = 1e-3
USE_PN_RAMP = False
PN_RAMP_STEPS = 4000

# Loss mix
BCE_WEIGHT = 0.3
MARGIN_WEIGHT = 0.8
MARGIN_M0 = 0.8
MARGIN_T = 1.2

# MED penalty
USE_MED = True
LAMBDA_MED = 1.0e-2
TAU_MED = 0.15

# Global Es normalization
GLOBAL_ES_TARGET = 1.0

# --- Speed flags ---
# Skip the baseline forward on the HIGH branch to save time.
# Keep False during fine-tuning so the hinge term stays accurate at high SNR.
SKIP_BASELINE_HIGH = False

# Projection frequency: re-normalize Es every N steps.
# Es drifts slowly; projecting every step wastes ~15% wall time for negligible gain.
PROJECT_EVERY = 4
PROJ_BMC = 64  # MC samples for Es estimation (64 is usually enough)


# =========================
# 3) Precompute CFO vector and N0 table
# =========================
def precompute_tables(max_ebn0_db=60):
    # SNRdB = EbN0dB + 10log10(R_bits_per_RE)
    snr_offset_db = 10.0 * math.log10(R_bits_per_RE)
    n0 = np.zeros(max_ebn0_db + 1, dtype=np.float64)
    noise_std_td = np.zeros(max_ebn0_db + 1, dtype=np.float64)
    for eb in range(max_ebn0_db + 1):
        snr_db = eb + snr_offset_db
        snr_lin = 10.0 ** (snr_db / 10.0)
        n0_awgn = 1.0 / snr_lin
        # time-domain noise variance per sample = N0 / Nfft (your sim convention)
        noise_var_td = n0_awgn / Nfft
        n0[eb] = n0_awgn
        noise_std_td[eb] = math.sqrt(noise_var_td / 2.0)
    return snr_offset_db, n0, noise_std_td


SNR_OFFSET_DB, N0_TABLE, NOISE_STD_TD_TABLE = precompute_tables(max_ebn0_db=60)


def cfo_phase_vec_fixed(epsv):
    n = torch.arange(NsymTD, device=DEVICE, dtype=torch.float32)
    return torch.exp(1j * 2 * math.pi * epsv * n / Nfft).to(DTYPEC).reshape(1, NsymTD)


CFO_VEC = cfo_phase_vec_fixed(EPS_FIXED)  # (1, NsymTD)


# =========================
# 4) Base codebook (J,K,M)
# =========================
def build_base_codebook(device):
    CB = torch.zeros((K, M, V), dtype=DTYPEC, device=device)

    CB[:, :, 0] = torch.tensor([
        [-1.3498 + 0j, -0.4218 + 0j, 0.4218 + 0j, 1.3498 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.1980 - 0.3724j, 0.6337 + 1.1918j, -0.6337 - 1.1918j, 0.1980 + 0.3724j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    CB[:, :, 1] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.1980 - 0.3724j, 0.6337 + 1.1918j, -0.6337 - 1.1918j, 0.1980 + 0.3724j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-1.3498 + 0j, -0.4218 + 0j, 0.4218 + 0j, 1.3498 + 0j],
    ], dtype=DTYPEC, device=device)

    CB[:, :, 2] = torch.tensor([
        [-0.6337 - 1.1918j, -0.1980 - 0.3724j, 0.1980 + 0.3724j, 0.6337 + 1.1918j],
        [0.2109 - 0.3653j, -0.6749 + 1.1690j, 0.6749 - 1.1690j, -0.2109 + 0.3653j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    CB[:, :, 3] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-1.3498 + 0j, -0.4218 + 0j, 0.4218 + 0j, 1.3498 + 0j],
        [0.2109 - 0.3653j, -0.6749 + 1.1690j, 0.6749 - 1.1690j, -0.2109 + 0.3653j],
    ], dtype=DTYPEC, device=device)

    CB[:, :, 4] = torch.tensor([
        [0.2109 - 0.3653j, -0.6749 + 1.1690j, 0.6749 - 1.1690j, -0.2109 + 0.3653j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.6337 - 1.1918j, -0.1980 - 0.3724j, 0.1980 + 0.3724j, 0.6337 + 1.1918j],
    ], dtype=DTYPEC, device=device)

    CB[:, :, 5] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-1.3498 + 0j, -0.4218 + 0j, 0.4218 + 0j, 1.3498 + 0j],
        [0.2109 - 0.3653j, -0.6749 + 1.1690j, 0.6749 - 1.1690j, -0.2109 + 0.3653j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    return CB.permute(2, 0, 1).contiguous()  # (V,K,M) == (J,K,M)


# =========================
# 5) Factor graph
# =========================
def build_factor_graph_from_mask(mask_jkm):
    edge = (mask_jkm.sum(dim=2) > 0)  # (J,K)
    res_users = [torch.nonzero(edge[:, r]).flatten().tolist() for r in range(K)]
    user_ress = [torch.nonzero(edge[j, :]).flatten().tolist() for j in range(J)]
    for r in range(K):
        if len(res_users[r]) != 3:
            raise ValueError(f"Resource {r} degree {len(res_users[r])} != 3")
    for j in range(J):
        if len(user_ress[j]) != 2:
            raise ValueError(f"User {j} degree {len(user_ress[j])} != 2")
    return res_users, user_ress


# =========================
# 6) Vectorized SCMA superposition
# =========================
def scma_superpose_fullQ_vec(C_jkm_masked, x_bvq):
    """
    C_jkm_masked: (V,K,M) complex
    x_bvq:        (B,V,Q) int64 in [0..M-1]
    returns W:    (B,K,Q)
    """
    B, V_, Q_ = x_bvq.shape
    # (B,V,K,M)
    CB = C_jkm_masked.unsqueeze(0).expand(B, -1, -1, -1)
    # idx: (B,V,K,Q)
    idx = x_bvq.unsqueeze(2).expand(B, V_, K, Q_)
    # gather on M dim -> (B,V,K,Q)
    cw = torch.gather(CB, dim=3, index=idx)
    # sum over users -> (B,K,Q)
    return cw.sum(dim=1)


# =========================
# 7) Differentiable Log-MPA
# =========================
def logmpa_llr_safe(y, CB, h, N0_eff, Nit, res_users, user_ress):
    device = y.device
    K_, M_, V_ = CB.shape
    N = y.shape[1]
    Ap = math.log(1.0 / M_)
    noise_inv = 1.0 / float(N0_eff)

    neg_inf = -1e9
    Ivg = torch.full((K_, V_, M_, N), neg_inf, dtype=DTYPEF, device=device)
    for u in range(V_):
        for r in user_ress[u]:
            Ivg[r, u, :, :] = Ap

    Igv = torch.full((K_, V_, M_, N), neg_inf, dtype=DTYPEF, device=device)

    for _ in range(Nit):
        Igv_new = torch.full_like(Igv, neg_inf)

        for r in range(K_):
            u1, u2, u3 = res_users[r]

            c1 = CB[r, :, u1]
            c2 = CB[r, :, u2]
            c3 = CB[r, :, u3]

            h1 = h[r, u1, :]
            h2 = h[r, u2, :]
            h3 = h[r, u3, :]

            t1 = c1[:, None] * h1[None, :]
            t2 = c2[:, None] * h2[None, :]
            t3 = c3[:, None] * h3[None, :]

            s = t1[:, None, None, :] + t2[None, :, None, :] + t3[None, None, :, :]
            yr = y[r, :].reshape(1, 1, 1, N)
            d = yr - s
            metric = -(d.real * d.real + d.imag * d.imag) * noise_inv

            a = metric + Ivg[r, u2, :, :].reshape(1, M_, 1, N) + Ivg[r, u3, :, :].reshape(1, 1, M_, N)
            Igv_new[r, u1, :, :] = torch.logsumexp(a, dim=(1, 2))

            b = metric + Ivg[r, u1, :, :].reshape(M_, 1, 1, N) + Ivg[r, u3, :, :].reshape(1, 1, M_, N)
            Igv_new[r, u2, :, :] = torch.logsumexp(b, dim=(0, 2))

            c = metric + Ivg[r, u1, :, :].reshape(M_, 1, 1, N) + Ivg[r, u2, :, :].reshape(1, M_, 1, N)
            Igv_new[r, u3, :, :] = torch.logsumexp(c, dim=(0, 1))

        Ivg_new = torch.full_like(Ivg, neg_inf)
        for u in range(V_):
            r1, r2 = user_ress[u]
            gv1 = Igv_new[r1, u, :, :]
            gv2 = Igv_new[r2, u, :, :]
            s1 = torch.logsumexp(gv1, dim=0)
            s2 = torch.logsumexp(gv2, dim=0)
            Ivg_new[r1, u, :, :] = gv2 - s2.reshape(1, N)
            Ivg_new[r2, u, :, :] = gv1 - s1.reshape(1, N)

        Igv = Igv_new
        Ivg = Ivg_new

    LLR = torch.zeros((m_bits * V_, N), dtype=DTYPEF, device=device)
    for u in range(V_):
        r1, r2 = user_ress[u]
        Qv = (-math.log(M_)) + Igv[r1, u, :, :] + Igv[r2, u, :, :]

        a1 = torch.logsumexp(Qv[0:2, :], dim=0)
        b1 = torch.logsumexp(Qv[2:4, :], dim=0)
        a2 = torch.logsumexp(Qv[[0, 2], :], dim=0)
        b2 = torch.logsumexp(Qv[[1, 3], :], dim=0)

        LLR[2 * u + 0, :] = a1 - b1
        LLR[2 * u + 1, :] = a2 - b2

    return LLR


# =========================
# 8) Loss helpers
# =========================
def bits_from_symbols(x_sym_vn):
    targets = torch.zeros((V, x_sym_vn.shape[1], m_bits), dtype=torch.float32, device=x_sym_vn.device)
    targets[:, :, 0] = ((x_sym_vn >> 1) & 1).float()
    targets[:, :, 1] = ((x_sym_vn >> 0) & 1).float()
    return targets


def bce_from_llr(LLR_2Vn, x_sym_vn):
    targets = bits_from_symbols(x_sym_vn)  # (V,N,2)
    llr_vnb = LLR_2Vn.reshape(V, m_bits, -1).permute(0, 2, 1).contiguous()
    logits_one = -llr_vnb
    return F.binary_cross_entropy_with_logits(logits_one, targets)


def margin_loss_from_llr(LLR_2Vn, x_sym_vn, m0=1.0, t=1.5):
    targets = bits_from_symbols(x_sym_vn)
    llr_vnb = LLR_2Vn.reshape(V, m_bits, -1).permute(0, 2, 1).contiguous()
    signed = (1.0 - 2.0 * targets) * llr_vnb
    return F.softplus((m0 - signed) / float(t)).mean()


def hard_ber_from_llr(LLR_2Vn, x_sym_vn):
    targets = bits_from_symbols(x_sym_vn)
    llr_vnb = LLR_2Vn.reshape(V, m_bits, -1).permute(0, 2, 1).contiguous()
    bhat = (llr_vnb <= 0).float()
    return (bhat != targets).float().mean().item()


# =========================
# 9) Soft-MED
# =========================
def soft_med_penalty(CB_kmv, res_users, tau=0.2):
    penalties = []
    for r in range(K):
        u1, u2, u3 = res_users[r]
        c1 = CB_kmv[r, :, u1]
        c2 = CB_kmv[r, :, u2]
        c3 = CB_kmv[r, :, u3]
        s = (c1[:, None, None] + c2[None, :, None] + c3[None, None, :]).reshape(-1)
        diff = s[:, None] - s[None, :]
        d2 = (diff.real * diff.real + diff.imag * diff.imag)
        d2 = d2 + torch.eye(d2.size(0), device=d2.device, dtype=d2.dtype) * 1e9
        softmin = -tau * torch.logsumexp((-d2 / tau).reshape(-1), dim=0)
        penalties.append(-softmin)
    return torch.stack(penalties).mean()


# =========================
# 10) Global Es normalization (fast, vectorized)
# =========================
@torch.no_grad()
def normalize_es_inplace(codebook_param, mask, Es_target=1.0, Bmc=64):
    x_mc = torch.randint(0, M, (Bmc, V, Q), device=DEVICE, dtype=torch.int64)
    C_masked = (codebook_param.data * mask).to(DTYPEC)
    W_mc = scma_superpose_fullQ_vec(C_masked, x_mc)  # (Bmc,K,Q)
    S_mc = W_mc.reshape(Bmc, Nfft)
    Es = (S_mc.real ** 2 + S_mc.imag ** 2).mean()
    g = torch.sqrt(torch.tensor(Es_target, device=DEVICE) / (Es + 1e-12))
    codebook_param.data *= g.to(DTYPEC)
    return float(Es.item())


@torch.no_grad()
def project_inplace(codebook_param, mask, Es_target=1.0, Bmc=64):
    codebook_param.data *= mask
    return normalize_es_inplace(codebook_param, mask, Es_target=Es_target, Bmc=Bmc)


# =========================
# 11) Wiener PN generator (vectorized)
# =========================
def wiener_pn_vec(Bsym, sigma_step):
    # dphi ~ N(0, sigma^2), phi = cumsum(dphi)
    dphi = sigma_step * torch.randn((Bsym, NsymTD), device=DEVICE, dtype=torch.float32)
    phi = torch.cumsum(dphi, dim=1)
    return torch.exp(1j * phi).to(DTYPEC)  # (Bsym, NsymTD)


def current_pn_sigma(step):
    if not USE_PN_RAMP:
        return PN_SIGMA_STEP_RAD
    # linear ramp 0 -> PN_SIGMA_STEP_RAD
    t = min(1.0, float(step) / float(PN_RAMP_STEPS))
    return PN_SIGMA_STEP_RAD * t


# =========================
# 12) Forward pass (matches sim convention; CP-valid channel model)
# =========================
def forward_one_codebook_fast(C_jkm, mask, x_bvq, h_rel, w_td, N0_awgn,
                              q_idx, Nit, epsv_fixed, cfo_vec,
                              pn_td,  # pre-generated shared phase noise
                              res_users, user_ress):
    """
    C_jkm:   (V,K,M)
    mask:    (V,K,M) float
    x_bvq:   (B,V,Q) symbols
    h_rel:   (B,Lh)  channel taps (starting at delay 0)
    w_td:    (B,NsymTD) complex AWGN already scaled for this EbN0
    N0_awgn: scalar; noise variance per subcarrier after FFT
    pn_td:   (B,NsymTD) shared phase noise vector
    """
    Bsym = x_bvq.size(0)

    # --- SCMA superposition -> S (B,Nfft) ---
    C_masked = (C_jkm * mask).to(DTYPEC)
    W = scma_superpose_fullQ_vec(C_masked, x_bvq)  # (B,K,Q)
    S = W.reshape(Bsym, Nfft)  # (B,Nfft)

    # --- Channel in frequency domain (CP condition valid) ---
    # h_pad: (B,Nfft)
    h_pad = F.pad(h_rel, (0, Nfft - Lh))  # pad last dim to Nfft
    Hf = torch.fft.fft(h_pad, n=Nfft, dim=-1)  # (B,Nfft)

    Y0_fd = S * Hf  # (B,Nfft)
    y_td0 = torch.fft.ifft(Y0_fd, n=Nfft, dim=-1)  # (B,Nfft)

    # add CP
    y_td = torch.cat([y_td0[:, -Ncp:], y_td0], dim=-1)  # (B,NsymTD)

    # --- CFO (fixed vector, precomputed) ---
    y_td = y_td * cfo_vec  # (B,NsymTD)

    # --- Phase noise (Wiener); shared between tr and base ---
    if pn_td is not None:
        y_td = y_td * pn_td  # (B,NsymTD)

    # --- Add AWGN (shared) ---
    y_cp = y_td + w_td

    # remove CP + FFT
    y_no_cp = y_cp[:, Ncp:Ncp + Nfft]
    Y_fd = torch.fft.fft(y_no_cp, n=Nfft, dim=-1)  # (B,Nfft)

    # reshape to (B,K,Q)
    Y_kdq = Y_fd.reshape(Bsym, K, Q)
    H_kdq = Hf.reshape(Bsym, K, Q)  # channel per tone

    # subset q
    Y_sub = Y_kdq[:, :, q_idx]  # (B,K,Qsub)
    H_sub = H_kdq[:, :, q_idx]  # (B,K,Qsub)

    # flatten to (K, B*Qsub)
    y = Y_sub.permute(1, 0, 2).contiguous().reshape(K, Bsym * q_idx.numel())
    Hk = H_sub.permute(1, 0, 2).contiguous().reshape(K, Bsym * q_idx.numel())

    # expand avoids the copy that repeat would force
    h_det = Hk[:, None, :].expand(-1, V, -1)

    CB_kmv = C_masked.permute(1, 2, 0).contiguous()  # (K,M,V)

    LLR = logmpa_llr_safe(y, CB_kmv, h_det, float(N0_awgn), Nit, res_users, user_ress)

    x_sub = x_bvq[:, :, q_idx].permute(1, 0, 2).contiguous().reshape(V, Bsym * q_idx.numel())
    return LLR, x_sub


# =========================
# 13) Batch builder (shared randomness)
# =========================
def make_shared_batch(EbN0dB_int, Bsym, q_sub):
    # symbols
    x_bvq = torch.randint(0, M, (Bsym, V, Q), device=DEVICE, dtype=torch.int64)

    # channel taps — vectorized; avoids the per-tap d_rel[p].item() GPU->CPU sync
    re = torch.randn(NUM_TAPS, Bsym, device=DEVICE)
    im = torch.randn(NUM_TAPS, Bsym, device=DEVICE)
    taps = tap_amp.view(NUM_TAPS, 1) * torch.complex(re, im)  # (P, Bsym), complex64
    h_rel = torch.zeros((Bsym, Lh), dtype=DTYPEC, device=DEVICE)
    h_rel[:, TAP_IDX] = taps.T  # scatter to tap delays

    # noise
    N0_awgn = float(N0_TABLE[int(EbN0dB_int)])
    std = float(NOISE_STD_TD_TABLE[int(EbN0dB_int)])
    w_td = std * (torch.randn(Bsym, NsymTD, device=DEVICE) + 1j * torch.randn(Bsym, NsymTD, device=DEVICE))
    w_td = w_td.to(DTYPEC)

    # q subset
    q_idx = torch.randperm(Q, device=DEVICE)[:q_sub]

    return x_bvq, h_rel, w_td, N0_awgn, q_idx


# =========================
# 14) Hinge params by SNR
# =========================
def hinge_params(EbN0dB):
    if EbN0dB < 16:
        return 0.06, 15.0
    elif EbN0dB < 27:
        return 0.02, 2.0
    else:
        # Tight margin at high SNR to keep pressuring the error floor.
        return 0.015, 2.0


# =========================
# 15) SNR samplers
# =========================
LOW_SET = np.arange(0, 16)  # 0..15
HIGH_SET = np.arange(22, 36)  # 27..40
MID_SET = np.arange(16, 22)  # 16..26

USE_MID_OCCASIONALLY = True
P_MID_IN_LOW_PASS = 0.15


def sample_low_like():
    if USE_MID_OCCASIONALLY and (np.random.rand() < P_MID_IN_LOW_PASS):
        return int(np.random.choice(MID_SET))
    return int(np.random.choice(LOW_SET))


def sample_high_like():
    return int(np.random.choice(HIGH_SET))


def sample_nit(default_nit):
    if not USE_RANDOM_NIT:
        return default_nit
    return int(np.random.choice(NIT_SET))


# =========================
# 16) Main
# =========================
def main():
    torch.multiprocessing.freeze_support()

    # Build the sparsity mask and factor graph from the analytical SCMA codebook.
    math_base = build_base_codebook(DEVICE)  # (V,K,M)
    mask = (math_base.abs() > 1e-12).to(torch.float32)
    res_users, user_ress = build_factor_graph_from_mask(mask)

    best_cb_path = "deka_codebook.pt"

    # Fine-tune mode: start from a previously saved codebook if available.
    if os.path.exists(best_cb_path):
        print(f"[INFO] Loading {best_cb_path} as both initialization and hinge baseline.")
        loaded_cb = torch.load(best_cb_path, map_location=DEVICE)

        # Loaded codebook serves as the hinge baseline.
        base_param = torch.nn.Parameter(loaded_cb.clone())
        _ = normalize_es_inplace(base_param, mask, Es_target=GLOBAL_ES_TARGET, Bmc=PROJ_BMC)
        base = base_param.data.detach()

        # Trainable codebook starts at the baseline.
        codebook = torch.nn.Parameter(base.clone())
    else:
        print(f"[INFO] {best_cb_path} not found; starting from the analytical SCMA codebook.")
        base_param = torch.nn.Parameter(math_base.clone())
        _ = normalize_es_inplace(base_param, mask, Es_target=GLOBAL_ES_TARGET, Bmc=PROJ_BMC)
        base = base_param.data.detach()
        codebook = torch.nn.Parameter(base.clone())

    # Fine-tuning learning rate.
    FINE_TUNE_LR = 5e-4
    opt = torch.optim.Adam([codebook], lr=FINE_TUNE_LR)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=2500, gamma=0.5)

    # initial projection
    _ = project_inplace(codebook, mask, Es_target=GLOBAL_ES_TARGET, Bmc=PROJ_BMC)

    epsv = EPS_FIXED
    cfo_vec = CFO_VEC  # precomputed

    # Training loop.
    try:
        for step in range(NUM_STEPS):
            opt.zero_grad(set_to_none=True)

            EbN0_low = sample_low_like()
            EbN0_high = sample_high_like()

            x_l, h_l, w_l, N0_l, q_l = make_shared_batch(EbN0_low, BATCH_OFDMSYM, Q_SUB)
            x_h, h_h, w_h, N0_h, q_h = make_shared_batch(EbN0_high, BATCH_OFDMSYM, Q_SUB)

            pn_sigma = current_pn_sigma(step) if ENABLE_PN else 0.0

            def one_pass(EbN0dB, x_bvq, h_rel, w_td, N0_awgn, q_idx, default_nit, do_baseline=True):
                Nit = sample_nit(default_nit)

                # Generate PN once and reuse for tr and base so the hinge sees identical conditions.
                if ENABLE_PN and (pn_sigma > 0.0):
                    shared_pn_td = wiener_pn_vec(x_bvq.size(0), pn_sigma)
                else:
                    shared_pn_td = None

                LLR_tr, xsub_tr = forward_one_codebook_fast(
                    codebook, mask, x_bvq, h_rel, w_td, N0_awgn,
                    q_idx, Nit, epsv, cfo_vec,
                    shared_pn_td,
                    res_users, user_ress
                )
                loss_tr = BCE_WEIGHT * bce_from_llr(LLR_tr, xsub_tr) + \
                          MARGIN_WEIGHT * margin_loss_from_llr(LLR_tr, xsub_tr, m0=MARGIN_M0, t=MARGIN_T)

                if do_baseline:
                    with torch.no_grad():
                        LLR_b, xsub_b = forward_one_codebook_fast(
                            base, mask, x_bvq, h_rel, w_td, N0_awgn,
                            q_idx, Nit, epsv, cfo_vec,
                            shared_pn_td,
                            res_users, user_ress
                        )
                        loss_base = BCE_WEIGHT * bce_from_llr(LLR_b, xsub_b) + \
                                    MARGIN_WEIGHT * margin_loss_from_llr(LLR_b, xsub_b, m0=MARGIN_M0, t=MARGIN_T)
                else:
                    # No baseline: collapse the hinge term to zero.
                    loss_base = loss_tr.detach()

                hm, lam = hinge_params(EbN0dB)
                hinge = torch.relu(loss_tr - loss_base + hm)
                loss_hinge = hinge * hinge
                return loss_tr, loss_base, loss_hinge, lam, (LLR_tr, xsub_tr, (None if not do_baseline else LLR_b),
                                                             (None if not do_baseline else xsub_b))

            # LOW: always compute the baseline.
            loss_tr_l, loss_b_l, loss_h_l, lam_l, pack_l = one_pass(EbN0_low, x_l, h_l, w_l, N0_l, q_l, NIT_MPA_LOW,
                                                                    do_baseline=True)
            # HIGH: optionally skip the baseline to save time. Keep False during fine-tuning.
            do_base_high = (not SKIP_BASELINE_HIGH)
            loss_tr_h, loss_b_h, loss_h_h, lam_h, pack_h = one_pass(EbN0_high, x_h, h_h, w_h, N0_h, q_h, NIT_MPA_HIGH,
                                                                    do_baseline=do_base_high)

            med_pen = torch.tensor(0.0, device=DEVICE)
            if USE_MED:
                CB_kmv_tr = (codebook * mask).permute(1, 2, 0).contiguous()
                med_pen = soft_med_penalty(CB_kmv_tr, res_users, tau=TAU_MED)

            # Up-weight the high-SNR branch to drive the error floor down.
            HIGH_WEIGHT = 10.0
            total = (loss_tr_l + lam_l * loss_h_l) + HIGH_WEIGHT * (loss_tr_h + lam_h * loss_h_h) + LAMBDA_MED * med_pen

            total.backward()
            torch.nn.utils.clip_grad_norm_([codebook], CLIP_GRAD_NORM)
            opt.step()
            sch.step()

            # Re-project Es every PROJECT_EVERY steps.
            if (step % PROJECT_EVERY) == 0:
                Es_est = project_inplace(codebook, mask, Es_target=GLOBAL_ES_TARGET, Bmc=PROJ_BMC)
            else:
                Es_est = float("nan")

            if step % PRINT_EVERY == 0:
                lr_now = opt.param_groups[0]["lr"]
                with torch.no_grad():
                    LLR_tr_l, xsub_tr_l, LLR_b_l, xsub_b_l = pack_l
                    ber_tr_l = hard_ber_from_llr(LLR_tr_l, xsub_tr_l)
                    ber_b_l = hard_ber_from_llr(LLR_b_l, xsub_b_l)

                    LLR_tr_h, xsub_tr_h, LLR_b_h, xsub_b_h = pack_h
                    ber_tr_h = hard_ber_from_llr(LLR_tr_h, xsub_tr_h)
                    ber_b_h = float("nan")
                    if (LLR_b_h is not None) and (xsub_b_h is not None):
                        ber_b_h = hard_ber_from_llr(LLR_b_h, xsub_b_h)

                print(
                    f"[TRAIN] step {step:5d} | eps {epsv:.2f} | PN_sigma {pn_sigma:.2e} | "
                    f"Q_SUB {Q_SUB} | B {BATCH_OFDMSYM} | lr {lr_now:.1e} | EsMC {Es_est:.3f}\n"
                    f"  LOW  EbN0 {EbN0_low:>2d}: tr {loss_tr_l.item():.3e} base {loss_b_l.item():.3e} "
                    f"hinge {loss_h_l.item():.3e}*{lam_l:.2f} | hardBER tr {ber_tr_l:.3e} base {ber_b_l:.3e}\n"
                    f"  HIGH EbN0 {EbN0_high:>2d}: tr {loss_tr_h.item():.3e} base {loss_b_h.item():.3e} "
                    f"hinge {loss_h_h.item():.3e}*{lam_h:.2f} | hardBER tr {ber_tr_h:.3e} base {ber_b_h:.3e}\n"
                    f"  MED {med_pen.item():.3e}*{(LAMBDA_MED if USE_MED else 0):.1e} | total {total.item():.3e}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received; saving current codebook before exit.")

    finally:
        with torch.no_grad():
            C_final = (codebook * mask).detach().cpu()  # (V,K,M) == (J,K,M)
            CB1_sim = C_final.permute(1, 2, 0).contiguous()  # (K,M,V)
        torch.save(C_final, "codebook_e2e.pt")
        torch.save(CB1_sim, "cb1_kmv.pt")
        np.save("codebook_e2e.npy", C_final.numpy())
        np.save("cb1_kmv.npy", CB1_sim.numpy())
        print("[INFO] Saved codebook_e2e.pt/.npy and cb1_kmv.pt/.npy.")
        print("[INFO] Training finished.")


if __name__ == "__main__":
    main()