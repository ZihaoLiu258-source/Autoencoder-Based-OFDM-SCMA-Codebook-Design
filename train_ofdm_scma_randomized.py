"""M1 ablation: train an SCMA codebook over randomized CFO and PN.

This script deliberately reuses the fixed-training implementation for every
component except the impairment sampler.  Its outputs use distinct names under
``artifacts/m1_randomized_training`` so the original fixed-point checkpoint is
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import train_ofdm_scma as fixed
from evaluation_utils import portable_path, run_with_log


DEFAULT_OUTPUT_DIR = Path("artifacts/m1_randomized_training")
CFO_RANGE = (0.0, 0.06)
PN_SIGMA_RANGE = (0.0, 3.0e-3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=fixed.NUM_STEPS)
    parser.add_argument("--seed", type=int, default=fixed.SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing randomized checkpoint.",
    )
    return parser.parse_args()


def reset_random_state(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    output_dir = args.output_dir.resolve()
    original_dir = Path(__file__).resolve().parent
    if output_dir == original_dir:
        raise ValueError("Randomized outputs must not be written to the original code directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "randomized_cb_kmv.pt"
    if checkpoint_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{checkpoint_path} already exists; pass --overwrite only for an intentional rerun"
        )

    reset_random_state(args.seed)
    impairment_rng = np.random.default_rng(args.seed + 100_003)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metadata = {
        "experiment": "M1 randomized-training ablation",
        "status": "running",
        "seed": args.seed,
        "steps": args.steps,
        "device": str(fixed.DEVICE),
        "impairment_sampling": {
            "cfo_epsilon": {"distribution": "uniform", "minimum": CFO_RANGE[0], "maximum": CFO_RANGE[1]},
            "pn_sigma_step_rad": {
                "distribution": "uniform",
                "minimum": PN_SIGMA_RANGE[0],
                "maximum": PN_SIGMA_RANGE[1],
            },
            "sampling_scope": "independent low/high draw per optimization step; shared by trainable and hinge-reference codebooks within each branch",
        },
        "controlled_against_fixed_training": {
            "initialization": "built-in Zhang codebook",
            "batch_ofdm_symbols": fixed.BATCH_OFDMSYM,
            "q_sub": fixed.Q_SUB,
            "mpa_iterations_low": fixed.NIT_MPA_LOW,
            "mpa_iterations_high": fixed.NIT_MPA_HIGH,
            "learning_rate": fixed.LEARNING_RATE,
            "lr_scheduler_step": fixed.LR_SCHEDULER_STEP,
            "lr_scheduler_gamma": fixed.LR_SCHEDULER_GAMMA,
            "high_snr_weight": fixed.HIGH_SNR_WEIGHT,
            "bce_weight": fixed.BCE_WEIGHT,
            "margin_weight": fixed.MARGIN_WEIGHT,
            "use_med": fixed.USE_MED,
            "lambda_med": fixed.LAMBDA_MED,
            "global_es_target": fixed.GLOBAL_ES_TARGET,
        },
        "output_separation": "all filenames are randomized_* under a dedicated artifact directory",
    }
    metadata_path = output_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    zhang_base = fixed.build_zhang_codebook(fixed.DEVICE)
    mask = (zhang_base.abs() > 1e-12).to(torch.float32)
    res_users, user_ress = fixed.build_factor_graph_from_mask(mask)

    print("[INFO] M1 randomized-training ablation")
    print("[INFO] Original fixed-point training code/checkpoint will not be modified.")
    print(f"[INFO] Output directory: {portable_path(output_dir)}")
    print(f"[INFO] CFO epsilon ~ U{CFO_RANGE}; PN sigma_step ~ U{PN_SIGMA_RANGE} rad/sample")
    print("[INFO] Every other training setting is inherited from train_ofdm_scma.py.")

    base_param = torch.nn.Parameter(zhang_base.clone())
    fixed.normalize_es_inplace(
        base_param, mask, Es_target=fixed.GLOBAL_ES_TARGET, Bmc=fixed.PROJ_BMC
    )
    base = base_param.data.detach()
    codebook = torch.nn.Parameter(base.clone())

    optimizer = torch.optim.Adam([codebook], lr=fixed.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=fixed.LR_SCHEDULER_STEP,
        gamma=fixed.LR_SCHEDULER_GAMMA,
    )
    fixed.project_inplace(
        codebook, mask, Es_target=fixed.GLOBAL_ES_TARGET, Bmc=fixed.PROJ_BMC
    )

    history_path = output_dir / "training_history.csv"
    history_fields = [
        "step", "ebn0_low_db", "ebn0_high_db", "epsilon_low", "epsilon_high",
        "pn_sigma_low", "pn_sigma_high", "loss_low", "loss_high", "total_loss",
        "ber_low", "ber_high", "learning_rate", "elapsed_seconds",
    ]
    started = time.perf_counter()

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history_writer = csv.DictWriter(history_file, fieldnames=history_fields)
        history_writer.writeheader()

        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)

            ebn0_low = fixed.sample_low_like()
            ebn0_high = fixed.sample_high_like()
            low_batch = fixed.make_shared_batch(
                ebn0_low, fixed.BATCH_OFDMSYM, fixed.Q_SUB
            )
            high_batch = fixed.make_shared_batch(
                ebn0_high, fixed.BATCH_OFDMSYM, fixed.Q_SUB
            )

            epsilon_low = float(impairment_rng.uniform(*CFO_RANGE))
            epsilon_high = float(impairment_rng.uniform(*CFO_RANGE))
            pn_sigma_low = float(impairment_rng.uniform(*PN_SIGMA_RANGE))
            pn_sigma_high = float(impairment_rng.uniform(*PN_SIGMA_RANGE))

            def one_pass(
                ebn0_db: int,
                batch,
                default_nit: int,
                epsilon: float,
                pn_sigma: float,
            ):
                x_bvq, h_rel, w_td, n0_awgn, q_idx = batch
                nit = fixed.sample_nit(default_nit)
                cfo_vec = fixed.cfo_phase_vec_fixed(epsilon)
                shared_pn_td = (
                    fixed.wiener_pn_vec(x_bvq.size(0), pn_sigma)
                    if fixed.ENABLE_PN and pn_sigma > 0.0
                    else None
                )

                llr_train, xsub_train = fixed.forward_one_codebook_fast(
                    codebook, mask, x_bvq, h_rel, w_td, n0_awgn,
                    q_idx, nit, epsilon, cfo_vec, shared_pn_td,
                    res_users, user_ress,
                )
                loss_train = (
                    fixed.BCE_WEIGHT * fixed.bce_from_llr(llr_train, xsub_train)
                    + fixed.MARGIN_WEIGHT
                    * fixed.margin_loss_from_llr(
                        llr_train, xsub_train, m0=fixed.MARGIN_M0, t=fixed.MARGIN_T
                    )
                )

                with torch.no_grad():
                    llr_base, xsub_base = fixed.forward_one_codebook_fast(
                        base, mask, x_bvq, h_rel, w_td, n0_awgn,
                        q_idx, nit, epsilon, cfo_vec, shared_pn_td,
                        res_users, user_ress,
                    )
                    loss_base = (
                        fixed.BCE_WEIGHT * fixed.bce_from_llr(llr_base, xsub_base)
                        + fixed.MARGIN_WEIGHT
                        * fixed.margin_loss_from_llr(
                            llr_base, xsub_base, m0=fixed.MARGIN_M0, t=fixed.MARGIN_T
                        )
                    )

                hinge_margin, hinge_weight = fixed.hinge_params(ebn0_db)
                hinge = torch.relu(loss_train - loss_base + hinge_margin)
                return (
                    loss_train,
                    loss_base,
                    hinge.square(),
                    hinge_weight,
                    (llr_train, xsub_train, llr_base, xsub_base),
                )

            low = one_pass(
                ebn0_low, low_batch, fixed.NIT_MPA_LOW, epsilon_low, pn_sigma_low
            )
            high = one_pass(
                ebn0_high, high_batch, fixed.NIT_MPA_HIGH, epsilon_high, pn_sigma_high
            )
            loss_tr_low, loss_base_low, loss_hinge_low, lam_low, pack_low = low
            loss_tr_high, loss_base_high, loss_hinge_high, lam_high, pack_high = high

            med_penalty = torch.tensor(0.0, device=fixed.DEVICE)
            if fixed.USE_MED:
                cb_kmv = (codebook * mask).permute(1, 2, 0).contiguous()
                med_penalty = fixed.soft_med_penalty(
                    cb_kmv, res_users, tau=fixed.TAU_MED
                )

            total = (
                loss_tr_low + lam_low * loss_hinge_low
                + fixed.HIGH_SNR_WEIGHT
                * (loss_tr_high + lam_high * loss_hinge_high)
                + fixed.LAMBDA_MED * med_penalty
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_([codebook], fixed.CLIP_GRAD_NORM)
            optimizer.step()
            scheduler.step()

            if step % fixed.PROJECT_EVERY == 0:
                es_estimate = fixed.project_inplace(
                    codebook,
                    mask,
                    Es_target=fixed.GLOBAL_ES_TARGET,
                    Bmc=fixed.PROJ_BMC,
                )
            else:
                es_estimate = float("nan")

            if step % fixed.PRINT_EVERY == 0 or step == args.steps - 1:
                with torch.no_grad():
                    ber_low = fixed.hard_ber_from_llr(pack_low[0], pack_low[1])
                    ber_base_low = fixed.hard_ber_from_llr(pack_low[2], pack_low[3])
                    ber_high = fixed.hard_ber_from_llr(pack_high[0], pack_high[1])
                    ber_base_high = fixed.hard_ber_from_llr(pack_high[2], pack_high[3])
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - started
                print(
                    f"[TRAIN-RAND] step {step:5d}/{args.steps - 1:5d} | "
                    f"LOW eps={epsilon_low:.4f} pn={pn_sigma_low:.3e} EbN0={ebn0_low:2d} "
                    f"loss={loss_tr_low.item():.3e} BER={ber_low:.3e} "
                    f"(base={ber_base_low:.3e})\n"
                    f"             HIGH eps={epsilon_high:.4f} pn={pn_sigma_high:.3e} EbN0={ebn0_high:2d} "
                    f"loss={loss_tr_high.item():.3e} BER={ber_high:.3e} "
                    f"(base={ber_base_high:.3e}) | total={total.item():.3e} "
                    f"lr={lr_now:.1e} EsMC={es_estimate:.3f} elapsed={elapsed:.1f}s"
                )
                history_writer.writerow({
                    "step": step,
                    "ebn0_low_db": ebn0_low,
                    "ebn0_high_db": ebn0_high,
                    "epsilon_low": epsilon_low,
                    "epsilon_high": epsilon_high,
                    "pn_sigma_low": pn_sigma_low,
                    "pn_sigma_high": pn_sigma_high,
                    "loss_low": float(loss_tr_low.item()),
                    "loss_high": float(loss_tr_high.item()),
                    "total_loss": float(total.item()),
                    "ber_low": ber_low,
                    "ber_high": ber_high,
                    "learning_rate": lr_now,
                    "elapsed_seconds": elapsed,
                })
                history_file.flush()

    with torch.no_grad():
        codebook_jkm = (codebook * mask).detach().cpu()
        codebook_kmv = codebook_jkm.permute(1, 2, 0).contiguous()
    torch.save(codebook_jkm, output_dir / "randomized_codebook_e2e.pt")
    torch.save(codebook_kmv, output_dir / "randomized_cb_kmv.pt")
    np.save(output_dir / "randomized_codebook_e2e.npy", codebook_jkm.numpy())
    np.save(output_dir / "randomized_cb_kmv.npy", codebook_kmv.numpy())

    metadata["status"] = "complete"
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    metadata["checkpoint_sha256"] = {
        filename: sha256(output_dir / filename)
        for filename in ("randomized_codebook_e2e.pt", "randomized_cb_kmv.pt")
    }
    metadata["outputs"] = [
        "randomized_codebook_e2e.pt",
        "randomized_cb_kmv.pt",
        "randomized_codebook_e2e.npy",
        "randomized_cb_kmv.npy",
        "training_history.csv",
        "train_randomized.log",
        "training_metadata.json",
    ]
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        "[INFO] Saved randomized checkpoint and metadata under "
        f"{portable_path(output_dir)}"
    )
    print(f"[INFO] Training elapsed: {metadata['elapsed_seconds']:.1f} s")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_with_log(lambda: train(args), str(output_dir / "train_randomized.log"))


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
