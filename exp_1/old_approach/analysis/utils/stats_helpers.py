"""
Script name: stats_helpers.py
Purpose: Statistical helper functions for behavioral analyses.
    - Clean paired arrays for tests.
    - Compute effect sizes for paired samples.
    - Run paired t-tests with formatted output (console and file).
Author: Rachel C. Metzgar
Date: 2025-09-29
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, TextIO
from scipy.stats import ttest_rel

__all__ = ["paired_clean", "cohens_dz", "paired_ttest_report"]

# --------------------------
# Paired-sample helpers
# --------------------------


def paired_clean(a: List[float], b: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return paired arrays with NaNs removed."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(b)
    return a[mask], b[mask]


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's dz effect size for paired samples."""
    diff = a - b
    if diff.size < 2 or np.nanstd(diff, ddof=1) == 0:
        return np.nan
    return np.nanmean(diff) / np.nanstd(diff, ddof=1)


def paired_ttest_report(
    a: List[float],
    b: List[float],
    label_a: str,
    label_b: str,
    measure_name: str,
    out_file: Optional[TextIO] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Run paired t-test, print formatted results, and optionally write to file.

    Returns (t, p, dz) or None if no valid pairs.
    """
    a, b = paired_clean(a, b)
    n = a.size
    if n == 0:
        msg = f"{measure_name}: no usable pairs."
        print(msg)
        if out_file:
            out_file.write(msg + "\n")
        return None

    t, p = ttest_rel(a, b, nan_policy="omit")
    dz = cohens_dz(a, b)

    msg = (f"{measure_name}: N={n}, "
           f"{label_a}={a.mean():.2f}, {label_b}={b.mean():.2f}, "
           f"t={t:.3f}, p={p:.4f}, dz={dz:.3f}")
    print(msg)
    if out_file:
        out_file.write(msg + "\n")

    return t, p, dz
