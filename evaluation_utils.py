"""Shared reproducibility helpers for the Monte-Carlo BER evaluations."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
import traceback


RULE_OF_THREE_CONFIDENCE = 0.95


def exact_global_superposition_energy(codebook) -> float:
    """Exact Es per resource for independent, equiprobable user messages.

    ``codebook`` has shape ``(K, M, V)``.  This analytic expectation includes
    cross terms caused by nonzero per-user codebook means; no Monte-Carlo
    normalization error is introduced.
    """
    import torch

    if codebook.ndim != 3:
        raise ValueError("codebook must have shape (K, M, V)")
    K, _, _ = codebook.shape
    with torch.no_grad():
        user_means = codebook.mean(dim=1)  # (K, V)
        user_second_moments = codebook.abs().square().sum(dim=0).mean(dim=0)
        cross_terms = (
            user_means.sum(dim=1).abs().square().sum()
            - user_means.abs().square().sum()
        )
        energy = (user_second_moments.sum() + cross_terms.real) / K
    return float(energy.item())


def normalize_codebook_exact_global_es(codebook, target_es: float = 1.0):
    """Scale a ``(K,M,V)`` codebook to an exact global superposition Es."""
    import math

    if target_es <= 0:
        raise ValueError("target_es must be positive")
    source_es = exact_global_superposition_energy(codebook)
    if source_es <= 0:
        raise ValueError("codebook energy must be positive")
    scale = math.sqrt(target_es / source_es)
    normalized = codebook * scale
    realized_es = exact_global_superposition_energy(normalized)
    return normalized, source_es, realized_es


def codebook_energy_metrics(codebook) -> dict:
    """Return exact global, per-user, and peak codeword energy statistics."""
    import torch

    with torch.no_grad():
        user_energy = codebook.abs().square().sum(dim=0).mean(dim=0)
        peak_codeword_energy = codebook.abs().square().sum(dim=0).max()
    user_values = [float(value) for value in user_energy.cpu().tolist()]
    return {
        "global_es": exact_global_superposition_energy(codebook),
        "per_user_average_energy": user_values,
        "user_energy_min": min(user_values),
        "user_energy_max": max(user_values),
        "user_energy_max_min_ratio": max(user_values) / min(user_values),
        "peak_codeword_energy": float(peak_codeword_energy.item()),
    }


def ber_with_zero_error_bound(error_count: int, total_bits: int) -> tuple[float, float, bool]:
    """Return empirical BER and the rule-of-three bound for a zero-error run.

    The measured BER is always ``error_count / total_bits`` and may therefore
    be exactly zero.  ``upper_95`` is finite only for a zero-error result; it is
    the conventional 95% rule-of-three bound ``3 / total_bits``.
    """
    if total_bits <= 0:
        raise ValueError("total_bits must be positive")
    if error_count < 0 or error_count > total_bits:
        raise ValueError("error_count must satisfy 0 <= error_count <= total_bits")

    empirical = error_count / total_bits
    is_upper_bound = error_count == 0
    upper_95 = 3.0 / total_bits if is_upper_bound else float("nan")
    return empirical, upper_95, is_upper_bound


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run_with_log(run_fn, log_filename: str) -> None:
    """Run an evaluation while mirroring stdout/stderr to a complete log."""
    log_path = Path(log_filename)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            started = datetime.now().astimezone()
            print(f"[RUN] started={started.isoformat()}")
            print(f"[RUN] command_script={Path(sys.argv[0]).resolve()}")
            print(f"[RUN] log_file={log_path.resolve()}")
            try:
                run_fn()
            except Exception:
                traceback.print_exc()
                raise
            finally:
                finished = datetime.now().astimezone()
                print(f"[RUN] finished={finished.isoformat()}")
