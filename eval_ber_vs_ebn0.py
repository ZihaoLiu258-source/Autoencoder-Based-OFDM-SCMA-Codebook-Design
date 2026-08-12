# ============================================================
# Evaluation: BER vs Eb/N0 at fixed CFO and phase noise.
# Channel: 8-tap Rayleigh, CFO eps=0.04, Wiener PN sigma=5e-4 rad/sample.
# Output: SCMA_EbN0_Simulation_Results.mat for MATLAB plotting.
# ============================================================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import math
import numpy as np
import torch
import scipy.io as sio

from evaluation_utils import (
    ber_with_zero_error_bound,
    normalize_codebook_exact_global_es,
    run_with_log,
)

torch.set_num_threads(1)
# =========================
# 0) Global settings
# =========================
SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPEC = torch.complex64
DTYPEF = torch.float32

K, M, V = 4, 4, 6

# OFDM
Nfft = 1024
Ncp = 32
NsymTD = Nfft + Ncp
Q = Nfft // K
assert Nfft % K == 0

delays = torch.tensor([1, 2, 4, 6, 9, 11, 15, 20], dtype=torch.int64)
powers = torch.tensor([0.36, 0.24, 0.15, 0.10, 0.06, 0.04, 0.025, 0.017], dtype=torch.float32)
powers = powers / powers.sum()
d_rel = delays - delays[0]
Lh = int(d_rel.max().item()) + 1

# Sweep
# Two impairment conditions are evaluated in parallel so the MATLAB plot can
# show a dashed "ideal" reference curve (CFO=0, PN=0) for each codebook.
# Order matters: the i-th entry of eps_list pairs with the i-th entry of sigma_list.
eps_list = [0.0, 0.03]
sigma_list = [0.0, 1e-4]
cond_labels = ['Ideal (CFO=0, PN=0)', 'CFO=0.03, PN=1e-4']
EbN0dB_vec = list(range(10, 35, 2))
Nit = 10

# Chunk simulation
Nd_total = 20000
Nd_chunk = 50

m_bits = int(math.log2(M))
R_bits_per_RE = (m_bits * V) / K

# Stop rule
STOP_EBN0_DB = 30
TARGET_ERRS_HIGH = 10000
MIN_ND_HIGH = 500
MAX_ND_HIGH = 50000

# ---------- Toggles ----------
# CFO/ICI handling
USE_TIME_DOMAIN_CFO = True
USE_ICI_AS_NOISE_APPROX = False

# ===== Phase Noise (Wiener / Random Walk) =====
USE_PHASE_NOISE = True
PN_MODEL = "wiener"

# Wiener phase noise: sigma_step is now per-condition (see sigma_list above).
PN_RANDOM_INIT = False

# Energy normalization for fair comparison
GLOBAL_ES_TARGET = 1.0

# Hard-zero threshold for trained CB
HARDZERO_THR = 1e-6
FG_THR = 1e-12


# ============================================================
# 1) Codebooks
# ============================================================
def load_trained_cb1_kmv(path="cb1_kmv.pt", device=DEVICE):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required trained codebook '{path}' was not found. "
            "Run train_ofdm_scma.py or restore the released checkpoint before evaluation."
        )
    cb = torch.load(path, map_location=device)
    if not torch.is_tensor(cb):
        cb = torch.tensor(cb)
    cb = cb.to(device)
    if not torch.is_complex(cb):
        cb = cb.to(DTYPEC)
    if tuple(cb.shape) != (K, M, V):
        raise ValueError(f"Loaded CB1 shape {tuple(cb.shape)} != {(K, M, V)}")
    return cb.to(DTYPEC)


def build_cb2_matlab_fullq(device=DEVICE):
    CB = torch.zeros((K, M, V), dtype=DTYPEC, device=device)
    CB[:, :, 0] = torch.tensor([
        [-0.3318 + 0.6262j, -0.8304 + 0.4252j, 0.8304 - 0.4252j, 0.3318 - 0.6262j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.7055 + 0j, -0.3601 + 0j, 0.3601 + 0j, -0.7055 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 1] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.7055 + 0j, -0.3601 + 0j, 0.3601 + 0j, -0.7055 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.3318 + 0.6262j, -0.8304 + 0.4252j, 0.8304 - 0.4252j, 0.3318 - 0.6262j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 2] = torch.tensor([
        [0.3601 + 0j, 0.7055 + 0j, -0.7055 + 0j, -0.3601 + 0j],
        [-0.4202 - 0.8350j, 0.5933 + 0.3548j, -0.5933 - 0.3548j, 0.4202 + 0.8350j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 3] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.3318 + 0.6262j, -0.8304 + 0.4252j, 0.8304 - 0.4252j, 0.3318 - 0.6262j],
        [-0.4202 - 0.8350j, 0.5933 + 0.3548j, -0.5933 - 0.3548j, 0.4202 + 0.8350j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 4] = torch.tensor([
        [-0.4202 - 0.8350j, 0.5933 + 0.3548j, -0.5933 - 0.3548j, 0.4202 + 0.8350j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.3601 + 0j, 0.7055 + 0j, -0.7055 + 0j, -0.3601 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 5] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.3318 + 0.6262j, -0.8304 + 0.4252j, 0.8304 - 0.4252j, 0.3318 - 0.6262j],
        [-0.4202 - 0.8350j, 0.5933 + 0.3548j, -0.5933 - 0.3548j, 0.4202 + 0.8350j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    return CB


def build_cb_xudong_li(device=DEVICE):
    CB = torch.zeros((K, M, V), dtype=DTYPEC, device=device)
    CB[:, :, 0] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.2378 + 1.0684j, -0.0684 + 0.3074j, 0.0684 - 0.3074j, 0.2378 - 1.0684j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.2840 + 0j, 0.9869 + 0j, -0.9869 + 0j, 0.2840 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 1] = torch.tensor([
        [-0.2378 + 1.0684j, -0.0684 + 0.3074j, 0.0684 - 0.3074j, 0.2378 - 1.0684j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.2840 + 0j, 0.9869 + 0j, -0.9869 + 0j, 0.2840 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 2] = torch.tensor([
        [0.6744 + 0.3794j, 0.1941 + 0.1092j, -0.1941 - 0.1092j, -0.6744 - 0.3794j],
        [-0.1941 - 0.1092j, 0.6744 + 0.3794j, -0.6744 - 0.3794j, 0.1941 + 0.1092j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 3] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.6744 + 0.3794j, 0.1941 + 0.1092j, -0.1941 - 0.1092j, -0.6744 - 0.3794j],
        [-0.1941 - 0.1092j, 0.6744 + 0.3794j, -0.6744 - 0.3794j, 0.1941 + 0.1092j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 4] = torch.tensor([
        [0.9869 + 0j, 0.2840 + 0j, -0.2840 + 0j, -0.9869 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.0684 - 0.3074j, -0.2378 + 1.0684j, 0.2378 - 1.0684j, -0.0684 + 0.3074j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 5] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.9869 + 0j, 0.2840 + 0j, -0.2840 + 0j, -0.9869 + 0j],
        [0.0684 - 0.3074j, -0.2378 + 1.0684j, 0.2378 - 1.0684j, -0.0684 + 0.3074j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    return CB


def build_cb_shutian_zhang(device=DEVICE):
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
    return CB


def build_cb_screenshot_new(device=DEVICE):
    CB = torch.zeros((K, M, V), dtype=DTYPEC, device=device)
    CB[:, :, 0] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.0795 - 0.8313j, 0.6546 - 0.1240j, -0.6544 + 0.1184j, -0.0801 + 0.8285j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [1.1001 - 1.1002j, -0.5186 + 0.1751j, 0.5088 - 0.1777j, -1.1000 + 1.1002j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 1] = torch.tensor([
        [-0.5572 - 0.2744j, 0.5608 + 0.2741j, -0.1253 + 0.9245j, 0.1308 - 0.9233j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.3889 + 0.3722j, 0.3927 - 0.3745j, -1.0805 + 1.0803j, 1.0805 - 1.0804j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 2] = torch.tensor([
        [0.9559 + 0.9561j, -0.9569 - 0.9571j, 0.4290 + 0.1975j, -0.4260 - 0.2028j],
        [0.4309 + 0.1225j, -0.4395 - 0.1296j, -0.9803 - 0.9790j, 0.9803 + 0.9793j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 3] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.9403 + 0.9400j, 0.0744 - 0.5773j, -0.9406 - 0.9399j, -0.0734 + 0.5820j],
        [0.2559 - 0.3754j, -0.9736 - 0.9737j, -0.2585 + 0.3752j, 0.9737 + 0.9738j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 4] = torch.tensor([
        [1.1171 - 1.1170j, -0.5509 + 0.1312j, 0.5561 - 0.1405j, -1.1170 + 1.1171j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.7533 - 0.3479j, -0.3812 + 0.5043j, 0.3697 - 0.5024j, 0.7535 + 0.3428j],
    ], dtype=DTYPEC, device=device)
    CB[:, :, 5] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.3115 + 1.0655j, -1.0652 + 0.5120j, 0.3088 - 1.0652j, 1.0652 - 0.5169j],
        [-0.5884 - 0.8779j, 0.4712 - 0.0476j, 0.5888 + 0.8758j, -0.4713 + 0.0435j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)
    return CB


def build_cb_lpcb_pn43(device=DEVICE):
    """LPCB PN43: codebook tailored for phase-noise robustness."""
    CB = torch.zeros((K, M, V), dtype=DTYPEC, device=device)

    # (:,:,1)
    CB[:, :, 0] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0.0850 + 1.0324j, 0 + 0j, 0 + 0j, -0.0850 - 1.0324j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 1.0841 + 0j, -1.0841 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    # (:,:,2)
    CB[:, :, 1] = torch.tensor([
        [0.0850 + 1.0324j, 0 + 0j, 0 + 0j, -0.0850 - 1.0324j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 1.0841 + 0j, -1.0841 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    # (:,:,3)
    CB[:, :, 2] = torch.tensor([
        [-0.7156 + 0.4894j, 0 + 0j, 0 + 0j, 0.7156 - 0.4894j],
        [0 + 0j, -0.7156 + 0.4894j, 0.7156 - 0.4894j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    # (:,:,4)
    CB[:, :, 3] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [-0.7156 + 0.4894j, 0 + 0j, 0 + 0j, 0.7156 - 0.4894j],
        [0 + 0j, -0.7156 + 0.4894j, 0.7156 - 0.4894j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    # (:,:,5)
    CB[:, :, 4] = torch.tensor([
        [1.0841 + 0j, 0 + 0j, 0 + 0j, -1.0841 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [0 + 0j, 0.0850 + 1.0324j, -0.0850 - 1.0324j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    # (:,:,6)
    CB[:, :, 5] = torch.tensor([
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
        [1.0841 + 0j, 0 + 0j, 0 + 0j, -1.0841 + 0j],
        [0 + 0j, 0.0850 + 1.0324j, -0.0850 - 1.0324j, 0 + 0j],
        [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    ], dtype=DTYPEC, device=device)

    return CB


# ============================================================
# 2) hard-zero + factor graph + Es normalization
# ============================================================
@torch.no_grad()
def hardzero(CB, thr=1e-6):
    return torch.where(CB.abs() < thr, torch.zeros_like(CB), CB)


def build_factor_graph(CB, thr=1e-12):
    nz = (CB.abs() > thr)
    F = torch.zeros((K, V), dtype=torch.bool, device=CB.device)
    for u in range(V):
        F[:, u] = torch.any(nz[:, :, u], dim=1)
    res_users = [torch.nonzero(F[r, :]).flatten().tolist() for r in range(K)]
    user_ress = [torch.nonzero(F[:, u]).flatten().tolist() for u in range(V)]
    for r in range(K):
        if len(res_users[r]) != 3:
            raise ValueError(f"[FG] Resource {r} degree {len(res_users[r])}, expected 3. Check CB zeros/threshold.")
    for u in range(V):
        if len(user_ress[u]) != 2:
            raise ValueError(f"[FG] User {u} degree {len(user_ress[u])}, expected 2. Check CB zeros/threshold.")
    return res_users, user_ress


@torch.no_grad()
def normalize_cb_global_es(CB_kmv, Es_target=1.0):
    normalized, source_es, _ = normalize_codebook_exact_global_es(CB_kmv, Es_target)
    return normalized.to(DTYPEC), source_es


# ============================================================
# 3) CFO helpers
# ============================================================
def cfo_ici_var(epsv, N):
    if abs(epsv) < 1e-15:
        return 0.0
    C0 = math.sin(math.pi * epsv) / (N * math.sin(math.pi * epsv / N))
    return float(1.0 - (abs(C0) ** 2))


def cfo_phase_vec(epsv, N, NsymTD, device):
    n = torch.arange(NsymTD, device=device, dtype=torch.float32)
    return torch.exp(1j * 2 * math.pi * epsv * n / N).to(DTYPEC)


# ============================================================
# 3.5) Phase Noise helpers (Wiener / Random Walk)
# ============================================================
@torch.no_grad()
def gen_wiener_phase_noise(NsymTD, batch, sigma_step_rad, device, random_init=True):
    device = torch.device(device)
    sigma = float(sigma_step_rad)

    if random_init:
        phi0 = (2.0 * math.pi) * torch.rand((batch, 1), device=device, dtype=torch.float32)
    else:
        phi0 = torch.zeros((batch, 1), device=device, dtype=torch.float32)

    if NsymTD <= 1:
        return phi0

    delta = sigma * torch.randn((batch, NsymTD - 1), device=device, dtype=torch.float32)
    walk = torch.cumsum(delta, dim=1)
    phi = torch.cat([phi0, phi0 + walk], dim=1)
    return phi


# ============================================================
# 4) SCMA encoder
# ============================================================
def scma_superpose_blocks(CB, x_uq):
    K_, M_, V_ = CB.shape
    Q_ = x_uq.shape[1]
    W = torch.zeros((K_, Q_), dtype=DTYPEC, device=CB.device)
    for u in range(V_):
        CB_u = CB[:, :, u]
        idx = x_uq[u, :].reshape(1, Q_).expand(K_, Q_)
        W += torch.gather(CB_u, dim=1, index=idx)
    return W


# ============================================================
# 5) Log-MPA decoder
# ============================================================
@torch.no_grad()
def scmadec_logmpa_llr(y, CB, h, N0_eff, Nit, res_users, user_ress):
    device = y.device
    K_, M_, V_ = CB.shape
    N = y.shape[1]
    Ap = math.log(1.0 / M_)  # = -log(M)
    noise_inv = 1.0 / float(N0_eff)

    neg_inf = -1e9
    Igv = torch.full((K_, V_, M_, N), neg_inf, dtype=DTYPEF, device=device)
    Ivg = torch.full((K_, V_, M_, N), neg_inf, dtype=DTYPEF, device=device)

    for u in range(V_):
        for r in user_ress[u]:
            Ivg[r, u, :, :] = Ap

    for _ in range(Nit):
        for r in range(K_):
            u1, u2, u3 = res_users[r]

            c1, c2, c3 = CB[r, :, u1], CB[r, :, u2], CB[r, :, u3]
            h1, h2, h3 = h[r, u1, :], h[r, u2, :], h[r, u3, :]

            t1 = c1[:, None] * h1[None, :]
            t2 = c2[:, None] * h2[None, :]
            t3 = c3[:, None] * h3[None, :]

            s = t1[:, None, None, :] + t2[None, :, None, :] + t3[None, None, :, :]
            yr = y[r, :].reshape(1, 1, 1, N)
            d = yr - s
            metric = -(d.real * d.real + d.imag * d.imag) * noise_inv

            a = metric + Ivg[r, u2, :, :].reshape(1, M_, 1, N) + Ivg[r, u3, :, :].reshape(1, 1, M_, N)
            Igv[r, u1, :, :] = torch.logsumexp(a, dim=(1, 2))

            b = metric + Ivg[r, u1, :, :].reshape(M_, 1, 1, N) + Ivg[r, u3, :, :].reshape(1, 1, M_, N)
            Igv[r, u2, :, :] = torch.logsumexp(b, dim=(0, 2))

            c = metric + Ivg[r, u1, :, :].reshape(M_, 1, 1, N) + Ivg[r, u2, :, :].reshape(1, M_, 1, N)
            Igv[r, u3, :, :] = torch.logsumexp(c, dim=(0, 1))

        for u in range(V_):
            r1, r2 = user_ress[u]
            s1 = torch.logsumexp(Igv[r1, u, :, :], dim=0)
            s2 = torch.logsumexp(Igv[r2, u, :, :], dim=0)
            Ivg[r1, u, :, :] = Igv[r2, u, :, :] - s2.reshape(1, N)
            Ivg[r2, u, :, :] = Igv[r1, u, :, :] - s1.reshape(1, N)

    LLR = torch.zeros((m_bits * V_, N), dtype=DTYPEF, device=device)
    for u in range(V_):
        r1, r2 = user_ress[u]
        Qv = Ap + Igv[r1, u, :, :] + Igv[r2, u, :, :]

        a1 = torch.logsumexp(Qv[0:2, :], dim=0)
        b1 = torch.logsumexp(Qv[2:4, :], dim=0)
        a2 = torch.logsumexp(Qv[[0, 2], :], dim=0)
        b2 = torch.logsumexp(Qv[[1, 3], :], dim=0)

        LLR[2 * u + 0, :] = a1 - b1
        LLR[2 * u + 1, :] = a2 - b2

    return LLR


# ============================================================
# 6) Bits helpers
# ============================================================
def symbols_to_bits(x, m_bits=2):
    V_, N_ = x.shape
    bits = torch.zeros((V_, N_, m_bits), dtype=torch.int64, device=x.device)
    for b in range(m_bits):
        shift = (m_bits - 1 - b)
        bits[:, :, b] = (x >> shift) & 1
    return bits


def llr_to_bits(LLR, V, m_bits=2):
    bh = (LLR <= 0).to(torch.int64)
    N = LLR.shape[1]
    return bh.reshape(V, m_bits, N).permute(0, 2, 1).contiguous()


# ============================================================
# 7) Shared randomness chunk simulation
# ============================================================
@torch.no_grad()
def gen_shared_chunk(Nd_now, N0_awgn, device, sigma_val):
    device = torch.device(device)

    x_vmq = torch.randint(0, M, (V, Nd_now * Q), device=device, dtype=torch.int64).view(V, Nd_now, Q)

    h_rel = torch.zeros((Nd_now, Lh), dtype=DTYPEC, device=device)
    for p in range(len(delays)):
        idxp = int(d_rel[p].item())
        amp = torch.sqrt(powers[p] / 2.0).to(device)
        h_rel[:, idxp] = amp * (torch.randn(Nd_now, device=device) + 1j * torch.randn(Nd_now, device=device))

    noiseVar_td = float(N0_awgn) / Nfft
    w_td = math.sqrt(noiseVar_td / 2.0) * (
            torch.randn((Nd_now, NsymTD), device=device) + 1j * torch.randn((Nd_now, NsymTD), device=device)
    )
    w_td = w_td.to(DTYPEC)

    if USE_PHASE_NOISE and PN_MODEL.lower() == "wiener" and sigma_val > 0.0:
        phi_td = gen_wiener_phase_noise(
            NsymTD=NsymTD,
            batch=Nd_now,
            sigma_step_rad=sigma_val,
            device=device,
            random_init=PN_RANDOM_INIT,
        )
    else:
        phi_td = None

    return x_vmq, h_rel, w_td, phi_td


@torch.no_grad()
def simulate_chunk_shared(CB, epsv, x_vmq, h_rel, w_td, phi_td, device):
    device = torch.device(device)
    CB = CB.to(device)

    Nd_now = x_vmq.shape[1]
    x_flat = x_vmq.reshape(V, Nd_now * Q)

    W_mkq = torch.zeros((Nd_now, K, Q), dtype=DTYPEC, device=device)
    for m in range(Nd_now):
        W_mkq[m] = scma_superpose_blocks(CB, x_vmq[:, m, :])

    S = torch.zeros((Nd_now, Nfft), dtype=DTYPEC, device=device)
    for k_re in range(K):
        S[:, k_re * Q:(k_re + 1) * Q] = W_mkq[:, k_re, :]

    x_fd = torch.fft.ifft(S, n=Nfft, dim=-1)
    x_td = torch.cat([x_fd[:, -Ncp:], x_fd], dim=-1)

    Lconv = NsymTD + Lh - 1
    x_pad = torch.cat([x_td, torch.zeros((Nd_now, Lh - 1), dtype=DTYPEC, device=device)], dim=-1)
    h_pad = torch.cat([h_rel, torch.zeros((Nd_now, Lconv - Lh), dtype=DTYPEC, device=device)], dim=-1)

    # One common downlink channel acts on the superposition observed by the
    # representative receiver; aggregate CFO/PN are applied afterward.
    y_lin = torch.fft.ifft(
        torch.fft.fft(x_pad, n=Lconv, dim=-1) * torch.fft.fft(h_pad, n=Lconv, dim=-1),
        n=Lconv, dim=-1
    )[:, :NsymTD]

    y_after = y_lin

    if USE_TIME_DOMAIN_CFO:
        cfo = cfo_phase_vec(epsv, Nfft, NsymTD, device).reshape(1, NsymTD)
        y_after = y_after * cfo

    if USE_PHASE_NOISE and (phi_td is not None):
        pn = torch.exp(1j * phi_td).to(DTYPEC)
        y_after = y_after * pn

    y_cp = y_after + w_td
    y_td2 = y_cp[:, Ncp:Ncp + Nfft]
    Y_fd = torch.fft.fft(y_td2, n=Nfft, dim=-1)

    h_pad2 = torch.cat([h_rel, torch.zeros((Nd_now, Nfft - Lh), dtype=DTYPEC, device=device)], dim=-1)
    Hf = torch.fft.fft(h_pad2, n=Nfft, dim=-1)

    Y_kdq = Y_fd.reshape(Nd_now, K, Q)
    H_kdq = Hf.reshape(Nd_now, K, Q)

    y = Y_kdq.permute(1, 0, 2).contiguous().reshape(K, Nd_now * Q)
    Hk = H_kdq.permute(1, 0, 2).contiguous().reshape(K, Nd_now * Q)
    h = Hk[:, None, :].repeat(1, V, 1)

    return y, h, x_flat


# ============================================================
# 8) Main
# ============================================================
def run():
    global USE_TIME_DOMAIN_CFO

    device = torch.device(DEVICE)
    print(f"[INFO] device = {device}")

    if USE_ICI_AS_NOISE_APPROX:
        USE_TIME_DOMAIN_CFO = False

    # Load or build candidate codebooks.
    CB1 = load_trained_cb1_kmv("cb1_kmv.pt", device=device)
    CB2 = build_cb2_matlab_fullq(device=device)
    CB3 = build_cb_xudong_li(device=device)
    CB4 = build_cb_shutian_zhang(device=device)
    CB5 = build_cb_screenshot_new(device=device)
    CB6 = build_cb_lpcb_pn43(device=device)

    CB1 = hardzero(CB1, thr=HARDZERO_THR)

    # Codebooks under test.
    CBs_raw = [CB1, CB2, CB3, CB4, CB5, CB6]
    labels = [
        "CB1: Proposed",
        "CB2: Deka et al. [6]",
        "CB3: Li et al. [7]",
        "CB4: Zhang et al. [8]",
        "CB5: Zheng et al. [11]",
        "CB6: PN-Resilient Liu et al. [10]",
    ]

    CB_all = []
    Es_list = []
    for cb in CBs_raw:
        cbn, Es = normalize_cb_global_es(cb, Es_target=GLOBAL_ES_TARGET)
        CB_all.append(cbn)
        Es_list.append(Es)

    print("[Es align]")
    for lab, Es in zip(labels, Es_list):
        print(f"  {lab:>26s} | Es={Es:.6f} -> {GLOBAL_ES_TARGET}")

    # Factor graph per codebook.
    FG = [build_factor_graph(CB_all[i], thr=FG_THR) for i in range(len(CB_all))]
    nCB = len(CB_all)

    # Per-codebook statistics use layout (condition, codebook, Eb/N0).
    stats_shape = (len(eps_list), nCB, len(EbN0dB_vec))
    BER = np.zeros(stats_shape, dtype=np.float64)  # empirical errors / bits; may be zero
    BER_95_upper = np.full(stats_shape, np.nan, dtype=np.float64)
    BER_is_upper_bound = np.zeros(stats_shape, dtype=np.uint8)
    error_count = np.zeros(stats_shape, dtype=np.int64)
    total_bits = np.zeros(stats_shape, dtype=np.int64)
    Nd_used = np.zeros((len(eps_list), len(EbN0dB_vec)), dtype=np.int64)
    stopping_reason = np.empty((len(eps_list), len(EbN0dB_vec)), dtype=object)

    for ie, (epsv, sigma_val) in enumerate(zip(eps_list, sigma_list)):
        ici = cfo_ici_var(epsv, Nfft)

        cfo_active = USE_TIME_DOMAIN_CFO and abs(epsv) > 0.0
        pn_active = USE_PHASE_NOISE and PN_MODEL.lower() == "wiener" and sigma_val > 0.0
        mode_cfo = "TD-CFO" if cfo_active else "no-CFO"
        mode_pn = "Wiener-PN" if pn_active else "no-PN"

        print(f"\n[INFO] Condition {ie + 1}/{len(eps_list)}: {cond_labels[ie]} | "
              f"{mode_cfo} + {mode_pn} | eps={epsv:.3f} | sigma={sigma_val:.1e}")

        for is_, EbN0dB in enumerate(EbN0dB_vec):
            SNRdB = EbN0dB + 10.0 * math.log10(R_bits_per_RE)
            SNRlin = 10.0 ** (SNRdB / 10.0)
            N0_awgn = 1.0 / SNRlin

            if USE_ICI_AS_NOISE_APPROX:
                N0_eff = N0_awgn + ici
            else:
                N0_eff = N0_awgn

            if EbN0dB < STOP_EBN0_DB:
                nd_limit = Nd_total
                use_stop = False
            else:
                nd_limit = MAX_ND_HIGH
                use_stop = True

            err_total = [torch.zeros((V,), dtype=torch.int64, device=device) for _ in range(nCB)]
            bits_total = [torch.zeros((V,), dtype=torch.int64, device=device) for _ in range(nCB)]

            nd_done = 0
            reached_target_errors = False
            while nd_done < nd_limit:
                nd_now = min(Nd_chunk, nd_limit - nd_done)

                x_vmq, h_rel, w_td, phi_td = gen_shared_chunk(nd_now, N0_awgn, device=device, sigma_val=sigma_val)

                for icb in range(nCB):
                    CB = CB_all[icb]
                    res_users, user_ress = FG[icb]

                    y, h, x_flat = simulate_chunk_shared(CB, epsv, x_vmq, h_rel, w_td, phi_td, device=device)
                    Nblk = y.shape[1]

                    LLR = scmadec_logmpa_llr(
                        y=y, CB=CB, h=h, N0_eff=N0_eff, Nit=Nit,
                        res_users=res_users, user_ress=user_ress
                    )

                    bits_tx = symbols_to_bits(x_flat, m_bits=m_bits)
                    bits_hat = llr_to_bits(LLR, V=V, m_bits=m_bits)

                    err_vec = (bits_tx != bits_hat).sum(dim=(1, 2)).to(torch.int64)
                    bits_vec = torch.tensor(Nblk * m_bits, device=device, dtype=torch.int64).repeat(V)

                    err_total[icb] += err_vec
                    bits_total[icb] += bits_vec

                nd_done += nd_now

                if use_stop and nd_done >= MIN_ND_HIGH:
                    if all(int(e.sum().item()) >= TARGET_ERRS_HIGH for e in err_total):
                        reached_target_errors = True
                        break

            Nd_used[ie, is_] = nd_done
            if not use_stop:
                stopping_reason[ie, is_] = "fixed_Nd_budget_completed"
            elif reached_target_errors:
                stopping_reason[ie, is_] = "target_errors_all_codebooks"
            else:
                stopping_reason[ie, is_] = "maximum_Nd_reached"

            for icb in range(nCB):
                n_err = int(err_total[icb].sum().item())
                n_bits = int(bits_total[icb].sum().item())
                ber, upper_95, is_upper = ber_with_zero_error_bound(n_err, n_bits)
                error_count[ie, icb, is_] = n_err
                total_bits[ie, icb, is_] = n_bits
                BER[ie, icb, is_] = ber
                BER_95_upper[ie, icb, is_] = upper_95
                BER_is_upper_bound[ie, icb, is_] = int(is_upper)

            line = (
                f"eps={epsv:.2f} sigma={sigma_val:.0e} | EbN0={EbN0dB:2d} dB | "
                f"Nd={nd_done} | stop={stopping_reason[ie, is_]} | "
            )
            summaries = []
            for icb, label in enumerate(labels):
                n_err = error_count[ie, icb, is_]
                n_bits = total_bits[ie, icb, is_]
                if BER_is_upper_bound[ie, icb, is_]:
                    metric = f"BER=0, 95%UB={BER_95_upper[ie, icb, is_]:.3e}"
                else:
                    metric = f"BER={BER[ie, icb, is_]:.3e}"
                summaries.append(f"{label.split(':')[0]}: errors={n_err}, bits={n_bits}, {metric}")
            line += " | ".join(summaries)
            print(line)

    # ============================================================
    # Export results for MATLAB
    # ============================================================
    mat_data = {
        'EbN0dB_vec': np.array(EbN0dB_vec),
        'eps_list': np.array(eps_list),
        'sigma_list': np.array(sigma_list),
        'cond_labels': cond_labels,
        'BER': BER,
        'BER_95_upper': BER_95_upper,
        'BER_is_upper_bound': BER_is_upper_bound,
        'error_count': error_count,
        'total_bits': total_bits,
        'Nd_used': Nd_used,
        'seed': np.int64(SEED),
        'stopping_reason': stopping_reason,
        'zero_error_bound_definition': '95% rule of three: 3 / total_bits',
        'labels': labels,
    }
    mat_filename = "SCMA_EbN0_Simulation_Results.mat"
    sio.savemat(mat_filename, mat_data)
    print(f"[INFO] Saved {mat_filename}")


if __name__ == "__main__":
    run_with_log(run, "eval_ber_vs_ebn0.log")
