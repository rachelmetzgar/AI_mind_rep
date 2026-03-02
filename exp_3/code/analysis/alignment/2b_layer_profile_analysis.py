#!/usr/bin/env python3
"""
Experiment 3, Phase 2b: Per-Layer Alignment Profile Analysis

Reads alignment results from 2a_alignment_analysis.py and produces
detailed layer-wise analyses answering three key questions:

    Q1: WHERE does concept-probe alignment emerge across the network?
        → Layer-onset detection, peak-layer identification, layer-band analysis

    Q2: Do READING and CONTROL probes show different layer profiles?
        → Probe-type divergence analysis per dimension

    Q3: Are there LAYER-SPECIFIC patterns masked by the mean?
        → Variance decomposition, peak-vs-spread classification

Outputs:
    For each analysis type (contrasts/raw, contrasts/residual, standalone):
        <analysis>/layer_profiles/
            per_dimension/
                <dim_name>_layer_profile.json   — full per-layer data
                <dim_name>_layer_profile.csv     — tabular for plotting
            cross_dimension_layer_summary.json   — all dims, all layers
            cross_dimension_layer_summary.csv    — all dims, all layers (flat)
            analysis_report.txt                  — human-readable findings

    And a top-level combined report:
        combined_layer_analysis_report.txt       — all analyses in one file

Usage:
    python 2b_layer_profile_analysis.py --version labels --analysis raw
    python 2b_layer_profile_analysis.py --version balanced_names --analysis residual
    python 2b_layer_profile_analysis.py --version labels --analysis standalone
    python 2b_layer_profile_analysis.py --version labels --analysis all

Env: llama2_env (only needs numpy + json, no GPU)
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import csv
import argparse
import numpy as np
from collections import OrderedDict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import config, set_version, add_version_argument, get_version_results_dir


# ========================== CONFIG ========================== #

# Version-dependent paths — set by _init_paths() after set_version()
ALIGNMENT_ROOT = None
OUTPUT_ROOT = None
ANALYSIS_DIR_MAP = None


def _init_paths():
    """Initialize version-dependent paths after set_version() has been called."""
    global ALIGNMENT_ROOT, OUTPUT_ROOT, ANALYSIS_DIR_MAP

    version_dir = get_version_results_dir(config.RESULTS.alignment_versions)
    ALIGNMENT_ROOT = str(version_dir)
    OUTPUT_ROOT = str(version_dir)

    ANALYSIS_DIR_MAP = {
        "raw": str(version_dir / "contrasts" / "raw"),
        "residual": str(version_dir / "contrasts" / "residual"),
        "standalone": str(version_dir / "standalone"),
    }

# LLaMA-2-13B has 40 transformer layers → 41 hidden states (0=embedding, 1-40=layers)
# Probes are typically trained on layers 1-40 (hidden_states indices)
N_LAYERS_EXPECTED = 41  # adjust if your probe set differs

# Layer band definitions (for LLaMA-2-13B-Chat, 40 layers)
# These are rough functional regions; adjust based on your probe accuracy profiles
LAYER_BANDS = {
    "early":  (1, 10),    # layers 1-10:  token-level / shallow features
    "middle": (11, 25),   # layers 11-25: compositional / intermediate
    "late":   (26, 40),   # layers 26-40: abstract / decision-relevant
}

# Thresholds for onset detection
ONSET_THRESHOLD_ABS_COS = 0.05   # |cos| above which we consider "non-trivial"
ONSET_THRESHOLD_R2 = 0.005       # R² above which we consider "non-trivial"

# For peak detection: a layer is "near-peak" if within this fraction of max
NEAR_PEAK_FRACTION = 0.80


# ========================== LOADING ========================== #

def load_alignment_results(analysis_dir):
    """
    Load per-layer alignment data for all dimensions in an analysis directory.

    Returns dict: {dim_name: {"reading": {layer: cos}, "control": {layer: cos}, "dim_id": int}}

    Per-layer data is stored as JSON strings inside .npz files from 2b.
    """
    results = {}

    if not os.path.isdir(analysis_dir):
        print(f"  [WARN] Directory not found: {analysis_dir}")
        return results

    for dim_name in sorted(os.listdir(analysis_dir)):
        dim_path = os.path.join(analysis_dir, dim_name)
        if not os.path.isdir(dim_path):
            continue

        align_file = os.path.join(dim_path, "alignment.npz")
        if not os.path.exists(align_file):
            continue

        data = np.load(align_file, allow_pickle=True)

        # Parse per-layer JSON
        reading_raw = json.loads(str(data["reading_per_layer"]))
        control_raw = json.loads(str(data["control_per_layer"]))

        # Convert string keys to int, extract cosine values
        reading = {int(k): v["cosine"] for k, v in reading_raw.items()}
        control = {int(k): v["cosine"] for k, v in control_raw.items()}

        # Extract dim_id from folder name
        parts = dim_name.split("_", 1)
        try:
            dim_id = int(parts[0])
        except ValueError:
            dim_id = -1

        # Also load entity overlap if present (residual analysis)
        entity_overlap = None
        if "entity_overlap_per_layer" in data:
            eo_raw = json.loads(str(data["entity_overlap_per_layer"]))
            entity_overlap = {int(k): float(v) for k, v in eo_raw.items()}

        # Load bootstrap distributions if present
        boot_reading = data.get("boot_reading_r2", None)
        boot_control = data.get("boot_control_r2", None)

        results[dim_name] = {
            "dim_id": dim_id,
            "reading": reading,
            "control": control,
            "entity_overlap": entity_overlap,
            "boot_reading_r2": boot_reading,
            "boot_control_r2": boot_control,
        }

    return results


# ========================== PER-DIMENSION ANALYSIS ========================== #

def analyze_layer_profile(cosine_by_layer, probe_type="reading"):
    """
    Analyze the layer-wise alignment profile for a single dimension × probe type.

    Args:
        cosine_by_layer: dict {layer_idx: cosine_similarity}
        probe_type: str label

    Returns dict with:
        - raw profile (layer → cos, |cos|, R²)
        - peak layer and value
        - onset layer (first layer exceeding threshold)
        - band means (early, middle, late)
        - concentration: fraction of total R² in each band
        - spread: how distributed is alignment across layers
        - sign consistency: does the direction flip?
    """
    if not cosine_by_layer:
        return {"error": "no data", "probe_type": probe_type}

    layers = sorted(cosine_by_layer.keys())
    cos_vals = np.array([cosine_by_layer[l] for l in layers])
    abs_cos = np.abs(cos_vals)
    r2_vals = cos_vals ** 2

    # ---- Peak detection ----
    peak_idx = np.argmax(abs_cos)
    peak_layer = layers[peak_idx]
    peak_cos = cos_vals[peak_idx]
    peak_abs_cos = abs_cos[peak_idx]
    peak_r2 = r2_vals[peak_idx]

    # ---- Onset detection ----
    onset_layer = None
    for i, l in enumerate(layers):
        if abs_cos[i] >= ONSET_THRESHOLD_ABS_COS:
            onset_layer = l
            break

    # ---- Band analysis ----
    band_stats = {}
    total_r2 = np.sum(r2_vals) if np.sum(r2_vals) > 0 else 1e-10

    for band_name, (lo, hi) in LAYER_BANDS.items():
        mask = np.array([(lo <= l <= hi) for l in layers])
        if mask.sum() == 0:
            band_stats[band_name] = {
                "mean_cos": 0.0, "mean_abs_cos": 0.0, "mean_r2": 0.0,
                "r2_fraction": 0.0, "n_layers": 0,
            }
            continue

        band_cos = cos_vals[mask]
        band_abs = abs_cos[mask]
        band_r2 = r2_vals[mask]

        band_stats[band_name] = {
            "mean_cos": float(np.mean(band_cos)),
            "mean_abs_cos": float(np.mean(band_abs)),
            "mean_r2": float(np.mean(band_r2)),
            "max_r2": float(np.max(band_r2)),
            "r2_fraction": float(np.sum(band_r2) / total_r2),
            "n_layers": int(mask.sum()),
        }

    # ---- Spread / concentration ----
    # Gini-like concentration: how peaked is the R² distribution?
    if np.sum(r2_vals) > 0:
        r2_norm = r2_vals / np.sum(r2_vals)
        entropy = -np.sum(r2_norm[r2_norm > 0] * np.log(r2_norm[r2_norm > 0] + 1e-15))
        max_entropy = np.log(len(r2_vals))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 0.0

    # Near-peak layers: how many layers are within 80% of peak?
    threshold_val = NEAR_PEAK_FRACTION * peak_abs_cos if peak_abs_cos > 0 else 0
    near_peak_layers = [l for l, ac in zip(layers, abs_cos) if ac >= threshold_val]
    near_peak_fraction = len(near_peak_layers) / len(layers) if layers else 0

    # ---- Sign consistency ----
    if len(cos_vals) > 0:
        n_positive = np.sum(cos_vals > 0)
        n_negative = np.sum(cos_vals < 0)
        sign_consistency = max(n_positive, n_negative) / len(cos_vals)
        dominant_sign = "positive" if n_positive >= n_negative else "negative"
    else:
        sign_consistency = 0.0
        dominant_sign = "none"

    # ---- Profile classification ----
    # Categorize the shape of the layer profile
    late_r2_frac = band_stats.get("late", {}).get("r2_fraction", 0)
    middle_r2_frac = band_stats.get("middle", {}).get("r2_fraction", 0)
    early_r2_frac = band_stats.get("early", {}).get("r2_fraction", 0)

    if peak_abs_cos < ONSET_THRESHOLD_ABS_COS:
        profile_type = "negligible"
    elif late_r2_frac > 0.6:
        profile_type = "late-concentrated"
    elif middle_r2_frac > 0.5:
        profile_type = "middle-concentrated"
    elif early_r2_frac > 0.5:
        profile_type = "early-concentrated"
    elif normalized_entropy > 0.85:
        profile_type = "distributed"
    else:
        profile_type = "mixed"

    return {
        "probe_type": probe_type,
        "n_layers": len(layers),
        "overall_mean_r2": float(np.mean(r2_vals)),
        "overall_mean_abs_cos": float(np.mean(abs_cos)),
        "peak_layer": peak_layer,
        "peak_cosine": float(peak_cos),
        "peak_abs_cosine": float(peak_abs_cos),
        "peak_r2": float(peak_r2),
        "onset_layer": onset_layer,
        "band_stats": band_stats,
        "normalized_entropy": float(normalized_entropy),
        "near_peak_layers": near_peak_layers,
        "near_peak_fraction": float(near_peak_fraction),
        "sign_consistency": float(sign_consistency),
        "dominant_sign": dominant_sign,
        "profile_type": profile_type,
        "per_layer": {
            int(l): {
                "cosine": float(cosine_by_layer[l]),
                "abs_cosine": float(abs(cosine_by_layer[l])),
                "r_squared": float(cosine_by_layer[l] ** 2),
            }
            for l in layers
        },
    }


def compare_probe_profiles(reading_profile, control_profile):
    """
    Compare reading vs control probe layer profiles for a single dimension.

    Returns dict with divergence metrics.
    """
    if "error" in reading_profile or "error" in control_profile:
        return {"error": "incomplete profiles"}

    r_layers = reading_profile["per_layer"]
    c_layers = control_profile["per_layer"]
    shared_layers = sorted(set(r_layers.keys()) & set(c_layers.keys()))

    if not shared_layers:
        return {"error": "no shared layers"}

    r_cos = np.array([r_layers[l]["cosine"] for l in shared_layers])
    c_cos = np.array([c_layers[l]["cosine"] for l in shared_layers])
    r_r2 = np.array([r_layers[l]["r_squared"] for l in shared_layers])
    c_r2 = np.array([c_layers[l]["r_squared"] for l in shared_layers])

    # Correlation between reading and control layer profiles
    if np.std(r_r2) > 0 and np.std(c_r2) > 0:
        profile_correlation = float(np.corrcoef(r_r2, c_r2)[0, 1])
    else:
        profile_correlation = 0.0

    # Peak offset
    r_peak = reading_profile["peak_layer"]
    c_peak = control_profile["peak_layer"]
    peak_offset = c_peak - r_peak

    # Band-level divergence
    band_divergence = {}
    for band_name in LAYER_BANDS:
        r_band = reading_profile["band_stats"].get(band_name, {})
        c_band = control_profile["band_stats"].get(band_name, {})
        r_val = r_band.get("mean_r2", 0)
        c_val = c_band.get("mean_r2", 0)
        band_divergence[band_name] = {
            "reading_mean_r2": r_val,
            "control_mean_r2": c_val,
            "difference": c_val - r_val,
            "ratio": c_val / r_val if r_val > 1e-8 else float("inf") if c_val > 1e-8 else 1.0,
        }

    # Which probe type dominates where?
    r_dominant_layers = [l for l, rr, cr in zip(shared_layers, r_r2, c_r2) if rr > cr]
    c_dominant_layers = [l for l, rr, cr in zip(shared_layers, r_r2, c_r2) if cr > rr]

    # Crossover detection: does one probe type dominate early and the other late?
    crossover_layer = None
    if len(shared_layers) > 2:
        diff = r_r2 - c_r2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            # Take the most prominent crossover (largest magnitude change)
            crossover_idx = sign_changes[0]
            crossover_layer = shared_layers[crossover_idx]

    return {
        "profile_correlation": profile_correlation,
        "reading_peak_layer": r_peak,
        "control_peak_layer": c_peak,
        "peak_offset": peak_offset,
        "band_divergence": band_divergence,
        "n_reading_dominant_layers": len(r_dominant_layers),
        "n_control_dominant_layers": len(c_dominant_layers),
        "crossover_layer": crossover_layer,
        "reading_profile_type": reading_profile["profile_type"],
        "control_profile_type": control_profile["profile_type"],
    }


# ========================== CROSS-DIMENSION ANALYSIS ========================== #

def build_cross_dimension_table(all_profiles):
    """
    Build a flat cross-dimension × layer table for easy plotting and comparison.

    Returns list of dicts (one row per dimension × layer × probe_type).
    """
    rows = []
    for dim_name, profiles in all_profiles.items():
        dim_id = profiles.get("dim_id", -1)
        for probe_type in ["reading", "control"]:
            prof = profiles.get(f"{probe_type}_profile", {})
            if "error" in prof:
                continue
            per_layer = prof.get("per_layer", {})
            for layer, vals in per_layer.items():
                rows.append({
                    "dim_name": dim_name,
                    "dim_id": dim_id,
                    "probe_type": probe_type,
                    "layer": layer,
                    "cosine": vals["cosine"],
                    "abs_cosine": vals["abs_cosine"],
                    "r_squared": vals["r_squared"],
                })
    return rows


def compute_cross_dimension_correlations(all_profiles, probe_type="reading"):
    """
    Compute pairwise correlations between dimension layer profiles.

    This reveals whether different concepts have similar layer-wise
    alignment patterns (suggesting shared mechanisms) or distinct ones.
    """
    dim_names = []
    profiles_matrix = []

    for dim_name, profiles in sorted(all_profiles.items()):
        prof = profiles.get(f"{probe_type}_profile", {})
        if "error" in prof:
            continue
        per_layer = prof.get("per_layer", {})
        if not per_layer:
            continue

        layers = sorted(per_layer.keys())
        r2_vec = [per_layer[l]["r_squared"] for l in layers]
        dim_names.append(dim_name)
        profiles_matrix.append(r2_vec)

    if len(profiles_matrix) < 2:
        return {"dim_names": dim_names, "correlation_matrix": []}

    mat = np.array(profiles_matrix)
    # Correlation matrix across dimensions
    corr = np.corrcoef(mat)

    return {
        "dim_names": dim_names,
        "correlation_matrix": corr.tolist(),
    }


# ========================== REPORT GENERATION ========================== #

def generate_report(analysis_name, all_profiles, all_comparisons, cross_corr):
    """
    Generate a human-readable analysis report.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"  LAYER-WISE ALIGNMENT PROFILE ANALYSIS: {analysis_name.upper()}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # ---- Section 1: Overview table ----
    lines.append("")
    lines.append("-" * 80)
    lines.append("SECTION 1: OVERVIEW — Peak layers, onset, and profile types")
    lines.append("-" * 80)
    lines.append("")

    header = (
        f"{'Dimension':<32} │ {'Probe':<8} │ {'Peak Lyr':>8} │ {'Peak |cos|':>10} │ "
        f"{'Onset Lyr':>9} │ {'Mean R²':>8} │ {'Profile Type':<20}"
    )
    lines.append(header)
    lines.append("─" * len(header))

    for dim_name in sorted(all_profiles.keys(), key=lambda x: all_profiles[x].get("dim_id", 99)):
        profiles = all_profiles[dim_name]
        for probe_type in ["reading", "control"]:
            prof = profiles.get(f"{probe_type}_profile", {})
            if "error" in prof:
                continue
            onset_str = str(prof["onset_layer"]) if prof["onset_layer"] is not None else "—"
            lines.append(
                f"{dim_name:<32} │ {probe_type:<8} │ {prof['peak_layer']:>8} │ "
                f"{prof['peak_abs_cosine']:>10.4f} │ {onset_str:>9} │ "
                f"{prof['overall_mean_r2']:>8.4f} │ {prof['profile_type']:<20}"
            )
        lines.append("")  # blank line between dimensions

    # ---- Section 2: Band concentration ----
    lines.append("")
    lines.append("-" * 80)
    lines.append("SECTION 2: LAYER-BAND R² CONCENTRATION (fraction of total R² per band)")
    lines.append(f"  Early: layers {LAYER_BANDS['early']}, "
                 f"Middle: layers {LAYER_BANDS['middle']}, "
                 f"Late: layers {LAYER_BANDS['late']}")
    lines.append("-" * 80)
    lines.append("")

    header2 = (
        f"{'Dimension':<32} │ {'Probe':<8} │ "
        f"{'Early':>8} │ {'Middle':>8} │ {'Late':>8} │ {'Dominant Band':<15}"
    )
    lines.append(header2)
    lines.append("─" * len(header2))

    for dim_name in sorted(all_profiles.keys(), key=lambda x: all_profiles[x].get("dim_id", 99)):
        profiles = all_profiles[dim_name]
        for probe_type in ["reading", "control"]:
            prof = profiles.get(f"{probe_type}_profile", {})
            if "error" in prof:
                continue
            bands = prof.get("band_stats", {})
            e_frac = bands.get("early", {}).get("r2_fraction", 0)
            m_frac = bands.get("middle", {}).get("r2_fraction", 0)
            l_frac = bands.get("late", {}).get("r2_fraction", 0)
            dominant = max(
                [("early", e_frac), ("middle", m_frac), ("late", l_frac)],
                key=lambda x: x[1]
            )[0]
            lines.append(
                f"{dim_name:<32} │ {probe_type:<8} │ "
                f"{e_frac:>8.3f} │ {m_frac:>8.3f} │ {l_frac:>8.3f} │ {dominant:<15}"
            )
        lines.append("")

    # ---- Section 3: Reading vs Control divergence ----
    lines.append("")
    lines.append("-" * 80)
    lines.append("SECTION 3: READING vs CONTROL PROBE DIVERGENCE")
    lines.append("-" * 80)
    lines.append("")

    header3 = (
        f"{'Dimension':<32} │ {'R Peak':>7} │ {'C Peak':>7} │ {'Offset':>7} │ "
        f"{'Profile r':>9} │ {'Crossover':>9} │ {'R Type':<18} │ {'C Type':<18}"
    )
    lines.append(header3)
    lines.append("─" * len(header3))

    for dim_name in sorted(all_comparisons.keys(),
                           key=lambda x: all_profiles.get(x, {}).get("dim_id", 99)):
        comp = all_comparisons[dim_name]
        if "error" in comp:
            continue
        xover = str(comp["crossover_layer"]) if comp["crossover_layer"] is not None else "—"
        lines.append(
            f"{dim_name:<32} │ {comp['reading_peak_layer']:>7} │ "
            f"{comp['control_peak_layer']:>7} │ {comp['peak_offset']:>+7} │ "
            f"{comp['profile_correlation']:>9.3f} │ {xover:>9} │ "
            f"{comp['reading_profile_type']:<18} │ {comp['control_profile_type']:<18}"
        )

    # ---- Section 4: Interpretive findings ----
    lines.append("")
    lines.append("-" * 80)
    lines.append("SECTION 4: KEY FINDINGS AND INTERPRETATIONS")
    lines.append("-" * 80)
    lines.append("")

    findings = extract_key_findings(all_profiles, all_comparisons, analysis_name)
    for i, finding in enumerate(findings, 1):
        lines.append(f"  Finding {i}: {finding['title']}")
        lines.append(f"    {finding['detail']}")
        lines.append(f"    Evidence: {finding['evidence']}")
        lines.append(f"    Implication: {finding['implication']}")
        lines.append("")

    # ---- Section 5: Per-layer data for top dimensions ----
    lines.append("")
    lines.append("-" * 80)
    lines.append("SECTION 5: FULL PER-LAYER DATA (top 5 dimensions by peak reading R²)")
    lines.append("-" * 80)

    # Sort by peak reading R²
    ranked = sorted(
        all_profiles.items(),
        key=lambda x: x[1].get("reading_profile", {}).get("peak_r2", 0),
        reverse=True,
    )

    for dim_name, profiles in ranked[:5]:
        lines.append(f"\n  --- {dim_name} ---")
        r_prof = profiles.get("reading_profile", {})
        c_prof = profiles.get("control_profile", {})

        if "error" in r_prof:
            continue

        r_layers = r_prof.get("per_layer", {})
        c_layers = c_prof.get("per_layer", {}) if "error" not in c_prof else {}
        all_layers = sorted(set(r_layers.keys()) | set(c_layers.keys()))

        lines.append(f"  {'Layer':>6} │ {'Read cos':>10} │ {'Read R²':>10} │ "
                     f"{'Ctrl cos':>10} │ {'Ctrl R²':>10} │ {'R²  Diff (R-C)':>14}")
        lines.append(f"  {'─' * 6} │ {'─' * 10} │ {'─' * 10} │ "
                     f"{'─' * 10} │ {'─' * 10} │ {'─' * 14}")

        for l in all_layers:
            r_cos = r_layers.get(l, {}).get("cosine", 0)
            r_r2 = r_layers.get(l, {}).get("r_squared", 0)
            c_cos = c_layers.get(l, {}).get("cosine", 0)
            c_r2 = c_layers.get(l, {}).get("r_squared", 0)
            diff = r_r2 - c_r2
            lines.append(
                f"  {l:>6} │ {r_cos:>+10.4f} │ {r_r2:>10.4f} │ "
                f"{c_cos:>+10.4f} │ {c_r2:>10.4f} │ {diff:>+14.4f}"
            )

    # ---- Section 6: Cross-dimension profile correlations ----
    lines.append("")
    lines.append("-" * 80)
    lines.append("SECTION 6: CROSS-DIMENSION PROFILE CORRELATIONS (reading probes)")
    lines.append("  High correlation → dimensions share similar layer-wise alignment patterns")
    lines.append("  (suggesting shared mechanisms)")
    lines.append("-" * 80)
    lines.append("")

    if cross_corr and cross_corr.get("correlation_matrix"):
        dim_names = cross_corr["dim_names"]
        corr_mat = np.array(cross_corr["correlation_matrix"])

        # Print abbreviated names for readability
        abbrev = [d[:20] for d in dim_names]

        # Print top-5 most correlated pairs
        pairs = []
        for i in range(len(dim_names)):
            for j in range(i + 1, len(dim_names)):
                pairs.append((dim_names[i], dim_names[j], corr_mat[i, j]))

        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        lines.append("  Top correlated dimension pairs:")
        for d1, d2, r in pairs[:10]:
            lines.append(f"    r = {r:+.3f}  :  {d1}  ↔  {d2}")

        lines.append("")
        lines.append("  Least correlated dimension pairs:")
        for d1, d2, r in pairs[-5:]:
            lines.append(f"    r = {r:+.3f}  :  {d1}  ↔  {d2}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def extract_key_findings(all_profiles, all_comparisons, analysis_name):
    """
    Automatically extract interpretable findings from the analysis.
    """
    findings = []

    # ---- Finding: Where alignment emerges ----
    reading_peaks = []
    control_peaks = []
    for dim_name, profiles in all_profiles.items():
        r = profiles.get("reading_profile", {})
        c = profiles.get("control_profile", {})
        if "error" not in r and r.get("peak_abs_cosine", 0) > ONSET_THRESHOLD_ABS_COS:
            reading_peaks.append((dim_name, r["peak_layer"], r["peak_abs_cosine"]))
        if "error" not in c and c.get("peak_abs_cosine", 0) > ONSET_THRESHOLD_ABS_COS:
            control_peaks.append((dim_name, c["peak_layer"], c["peak_abs_cosine"]))

    if reading_peaks:
        mean_r_peak = np.mean([p[1] for p in reading_peaks])
        mean_c_peak = np.mean([p[1] for p in control_peaks]) if control_peaks else 0

        # Categorize where peaks cluster
        late_peaks = [p for p in reading_peaks if p[1] >= LAYER_BANDS["late"][0]]
        mid_peaks = [p for p in reading_peaks if LAYER_BANDS["middle"][0] <= p[1] < LAYER_BANDS["late"][0]]
        early_peaks = [p for p in reading_peaks if p[1] < LAYER_BANDS["middle"][0]]

        findings.append({
            "title": "Alignment peak distribution across network depth",
            "detail": (
                f"Reading probe peaks: {len(early_peaks)} early, "
                f"{len(mid_peaks)} middle, {len(late_peaks)} late "
                f"(mean peak layer: {mean_r_peak:.1f}). "
                f"Control probe mean peak layer: {mean_c_peak:.1f}."
            ),
            "evidence": (
                f"Dimensions with late peaks (≥L{LAYER_BANDS['late'][0]}): "
                + ", ".join(f"{p[0]} (L{p[1]})" for p in late_peaks[:5])
                + (f" +{len(late_peaks)-5} more" if len(late_peaks) > 5 else "")
            ),
            "implication": (
                "Late-concentrated alignment suggests conceptual content becomes "
                "relevant to partner representation only after substantial processing. "
                "Early alignment would suggest shallow feature overlap."
            ),
        })

    # ---- Finding: Reading vs Control divergence patterns ----
    divergent_dims = []
    convergent_dims = []
    for dim_name, comp in all_comparisons.items():
        if "error" in comp:
            continue
        if comp["profile_correlation"] < 0.5:
            divergent_dims.append((dim_name, comp["profile_correlation"], comp["peak_offset"]))
        elif comp["profile_correlation"] > 0.8:
            convergent_dims.append((dim_name, comp["profile_correlation"]))

    if divergent_dims:
        findings.append({
            "title": "Dimensions where reading and control probes diverge in layer profile",
            "detail": (
                f"{len(divergent_dims)} dimension(s) show low reading-control profile "
                f"correlation (r < 0.5), suggesting the two probe types interface with "
                f"conceptual knowledge at different network depths."
            ),
            "evidence": ", ".join(
                f"{d[0]} (r={d[1]:.2f}, peak offset={d[2]:+d})"
                for d in sorted(divergent_dims, key=lambda x: x[1])[:5]
            ),
            "implication": (
                "Divergent profiles imply that reading probes (reflective token positions) "
                "and control probes (generation boundaries) access different processing "
                "stages for the same conceptual content. This is mechanistically informative."
            ),
        })

    if convergent_dims:
        findings.append({
            "title": "Dimensions where reading and control probes converge",
            "detail": (
                f"{len(convergent_dims)} dimension(s) show high reading-control profile "
                f"correlation (r > 0.8)."
            ),
            "evidence": ", ".join(
                f"{d[0]} (r={d[1]:.2f})" for d in convergent_dims[:5]
            ),
            "implication": (
                "Convergent profiles suggest these concepts align with partner probes "
                "through a shared processing pathway regardless of token position."
            ),
        })

    # ---- Finding: Profile type distribution ----
    profile_types = {}
    for dim_name, profiles in all_profiles.items():
        r = profiles.get("reading_profile", {})
        if "error" not in r:
            pt = r.get("profile_type", "unknown")
            profile_types.setdefault(pt, []).append(dim_name)

    if profile_types:
        type_summary = ", ".join(f"{k}: {len(v)}" for k, v in sorted(profile_types.items()))
        dominant_type = max(profile_types.items(), key=lambda x: len(x[1]))

        findings.append({
            "title": "Distribution of layer profile types (reading probes)",
            "detail": type_summary,
            "evidence": (
                f"Most common: '{dominant_type[0]}' ({len(dominant_type[1])} dims): "
                + ", ".join(dominant_type[1][:5])
            ),
            "implication": (
                "A predominance of late-concentrated profiles would suggest alignment "
                "emerges from abstract semantic processing. Distributed profiles suggest "
                "concept-probe overlap at multiple representational levels."
            ),
        })

    # ---- Finding: Sign consistency ----
    sign_flippers = []
    for dim_name, profiles in all_profiles.items():
        r = profiles.get("reading_profile", {})
        if "error" not in r and r.get("sign_consistency", 1) < 0.7:
            sign_flippers.append((dim_name, r["sign_consistency"]))

    if sign_flippers:
        findings.append({
            "title": "Dimensions with inconsistent alignment direction across layers",
            "detail": (
                f"{len(sign_flippers)} dimension(s) show sign consistency < 0.7, "
                f"meaning the concept vector flips direction relative to the probe "
                f"across layers."
            ),
            "evidence": ", ".join(f"{d[0]} ({d[1]:.2f})" for d in sign_flippers[:5]),
            "implication": (
                "Sign flips suggest the concept is encoded in opposing directions "
                "at different depths — potentially reflecting different computational "
                "roles (e.g., input encoding vs. output preparation)."
            ),
        })

    # ---- Finding: Control dimensions ----
    for dim_name, profiles in all_profiles.items():
        if "shapes" in dim_name.lower() or "15_" in dim_name:
            r = profiles.get("reading_profile", {})
            if "error" not in r:
                findings.append({
                    "title": f"Control dimension check: {dim_name}",
                    "detail": (
                        f"Peak |cos| = {r.get('peak_abs_cosine', 0):.4f}, "
                        f"mean R² = {r.get('overall_mean_r2', 0):.4f}"
                    ),
                    "evidence": f"Profile type: {r.get('profile_type', 'unknown')}",
                    "implication": (
                        "If the shapes control shows meaningful alignment, it suggests "
                        "pipeline artifacts. Negligible alignment validates the pipeline."
                    ),
                })

    return findings


# ========================== RUNNERS ========================== #

def run_analysis(analysis_name):
    """Run the full layer profile analysis for one analysis type."""
    print(f"\n{'=' * 60}")
    print(f"  LAYER PROFILE ANALYSIS: {analysis_name.upper()}")
    print(f"{'=' * 60}")

    analysis_dir = ANALYSIS_DIR_MAP.get(analysis_name)
    if analysis_dir is None:
        print(f"  Unknown analysis type: {analysis_name}")
        return None
    results = load_alignment_results(analysis_dir)

    if not results:
        print(f"  No alignment data found in {analysis_dir}")
        return None

    print(f"  Loaded {len(results)} dimensions")

    # Output directories — layer_profiles live alongside the alignment data
    out_dir = os.path.join(analysis_dir, "layer_profiles")
    per_dim_dir = os.path.join(out_dir, "per_dimension")
    os.makedirs(per_dim_dir, exist_ok=True)

    all_profiles = {}
    all_comparisons = {}

    for dim_name, data in sorted(results.items(), key=lambda x: x[1]["dim_id"]):
        print(f"\n  Analyzing: {dim_name}")

        # Per-dimension layer profile analysis
        reading_profile = analyze_layer_profile(data["reading"], "reading")
        control_profile = analyze_layer_profile(data["control"], "control")
        probe_comparison = compare_probe_profiles(reading_profile, control_profile)

        all_profiles[dim_name] = {
            "dim_id": data["dim_id"],
            "reading_profile": reading_profile,
            "control_profile": control_profile,
            "probe_comparison": probe_comparison,
        }
        all_comparisons[dim_name] = probe_comparison

        # Print quick summary
        r_peak = reading_profile.get("peak_layer", "?")
        r_type = reading_profile.get("profile_type", "?")
        c_peak = control_profile.get("peak_layer", "?")
        print(f"    Reading: peak L{r_peak}, type={r_type}")
        print(f"    Control: peak L{c_peak}")
        if "error" not in probe_comparison:
            print(f"    R-C correlation: {probe_comparison['profile_correlation']:.3f}")

        # Save per-dimension JSON
        dim_out = {
            "dim_name": dim_name,
            "dim_id": data["dim_id"],
            "reading_profile": reading_profile,
            "control_profile": control_profile,
            "probe_comparison": probe_comparison,
        }
        with open(os.path.join(per_dim_dir, f"{dim_name}_layer_profile.json"), "w") as f:
            json.dump(dim_out, f, indent=2, default=str)

        # Save per-dimension CSV (for easy plotting)
        csv_rows = []
        for probe_type, prof in [("reading", reading_profile), ("control", control_profile)]:
            if "error" in prof:
                continue
            for layer, vals in prof.get("per_layer", {}).items():
                csv_rows.append({
                    "dim_name": dim_name,
                    "dim_id": data["dim_id"],
                    "probe_type": probe_type,
                    "layer": layer,
                    "cosine": vals["cosine"],
                    "abs_cosine": vals["abs_cosine"],
                    "r_squared": vals["r_squared"],
                })

        csv_path = os.path.join(per_dim_dir, f"{dim_name}_layer_profile.csv")
        if csv_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

    # ---- Cross-dimension outputs ----

    # Cross-dimension layer table (flat CSV)
    cross_rows = build_cross_dimension_table(all_profiles)
    cross_csv = os.path.join(out_dir, "cross_dimension_layer_summary.csv")
    if cross_rows:
        with open(cross_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cross_rows[0].keys())
            writer.writeheader()
            writer.writerows(cross_rows)
        print(f"\n  Saved cross-dimension CSV: {cross_csv}")

    # Cross-dimension JSON summary
    summary_json = {
        dim_name: {
            "dim_id": p["dim_id"],
            "reading_peak_layer": p["reading_profile"].get("peak_layer"),
            "reading_peak_abs_cos": p["reading_profile"].get("peak_abs_cosine"),
            "reading_peak_r2": p["reading_profile"].get("peak_r2"),
            "reading_mean_r2": p["reading_profile"].get("overall_mean_r2"),
            "reading_onset": p["reading_profile"].get("onset_layer"),
            "reading_profile_type": p["reading_profile"].get("profile_type"),
            "reading_band_early_r2_frac": p["reading_profile"].get("band_stats", {}).get("early", {}).get("r2_fraction"),
            "reading_band_middle_r2_frac": p["reading_profile"].get("band_stats", {}).get("middle", {}).get("r2_fraction"),
            "reading_band_late_r2_frac": p["reading_profile"].get("band_stats", {}).get("late", {}).get("r2_fraction"),
            "control_peak_layer": p["control_profile"].get("peak_layer"),
            "control_peak_abs_cos": p["control_profile"].get("peak_abs_cosine"),
            "control_peak_r2": p["control_profile"].get("peak_r2"),
            "control_mean_r2": p["control_profile"].get("overall_mean_r2"),
            "control_onset": p["control_profile"].get("onset_layer"),
            "control_profile_type": p["control_profile"].get("profile_type"),
            "control_band_early_r2_frac": p["control_profile"].get("band_stats", {}).get("early", {}).get("r2_fraction"),
            "control_band_middle_r2_frac": p["control_profile"].get("band_stats", {}).get("middle", {}).get("r2_fraction"),
            "control_band_late_r2_frac": p["control_profile"].get("band_stats", {}).get("late", {}).get("r2_fraction"),
            "rc_profile_correlation": p.get("probe_comparison", {}).get("profile_correlation"),
            "rc_peak_offset": p.get("probe_comparison", {}).get("peak_offset"),
            "rc_crossover_layer": p.get("probe_comparison", {}).get("crossover_layer"),
            "reading_sign_consistency": p["reading_profile"].get("sign_consistency"),
            "reading_dominant_sign": p["reading_profile"].get("dominant_sign"),
            "reading_normalized_entropy": p["reading_profile"].get("normalized_entropy"),
        }
        for dim_name, p in all_profiles.items()
        if "error" not in p.get("reading_profile", {"error": True})
    }

    with open(os.path.join(out_dir, "cross_dimension_layer_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"  Saved cross-dimension JSON: {os.path.join(out_dir, 'cross_dimension_layer_summary.json')}")

    # Cross-dimension profile correlations
    cross_corr = compute_cross_dimension_correlations(all_profiles, "reading")
    if cross_corr.get("correlation_matrix"):
        with open(os.path.join(out_dir, "cross_dimension_profile_correlations.json"), "w") as f:
            json.dump(cross_corr, f, indent=2)

    # ---- Generate report ----
    report = generate_report(analysis_name, all_profiles, all_comparisons, cross_corr)
    report_path = os.path.join(out_dir, "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved: {report_path}")

    return {
        "analysis_name": analysis_name,
        "all_profiles": all_profiles,
        "all_comparisons": all_comparisons,
        "cross_corr": cross_corr,
        "report": report,
    }


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 Phase 1c: Per-layer alignment profile analysis"
    )
    add_version_argument(parser)
    parser.add_argument("--analysis", type=str, required=True,
                        choices=["raw", "residual", "standalone", "all"],
                        help="Which analysis to run")
    args = parser.parse_args()

    # Set version and initialize paths
    set_version(args.version)
    _init_paths()
    print(f"Version: {args.version}")

    analyses_to_run = (
        ["raw", "residual", "standalone"] if args.analysis == "all"
        else [args.analysis]
    )

    all_reports = []

    for analysis_name in analyses_to_run:
        result = run_analysis(analysis_name)
        if result:
            all_reports.append(result["report"])

    # Combined report
    if len(all_reports) > 1:
        combined = "\n\n\n".join(all_reports)
        combined_path = os.path.join(ALIGNMENT_ROOT, "combined_layer_analysis_report.txt")
        with open(combined_path, "w") as f:
            f.write(combined)
        print(f"\n✅ Combined report saved: {combined_path}")

    print("\n✅ Layer profile analysis complete.")


if __name__ == "__main__":
    main()