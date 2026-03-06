#!/usr/bin/env python3
"""
Generate pairwise bootstrap statistical tests between all dimension pairs.

For each pair of dimensions, computes paired bootstrap comparison using the
boot_metacognitive_r2 and boot_operational_r2 arrays from alignment.npz files.
These are paired across bootstrap iterations (same prompt resampling), so the
paired difference is valid.

Produces:
  - pairwise_dimensions.csv  — full pairwise table
  - pairwise_dimensions.json — same data as JSON (for HTML report generator)
  - pairwise_summary.json    — for each dim, which dims it's significantly > and <

Usage:
    python generate_pairwise_tests.py                      # turn 5, all types
    python generate_pairwise_tests.py --turn 3 --type raw  # specific turn/type

Rachel C. Metzgar · Mar 2026
"""

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ============================================================================
# CONFIG
# ============================================================================

ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent.parent / "versions"
VERSIONS = ["balanced_gpt", "nonsense_codeword"]

ANALYSIS_TYPES = {
    "raw": "contrasts/raw",
    "residual": "contrasts/residual",
    "standalone": "standalone",
}

# Dimension display info
DIM_NAMES = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind (holistic)", 17: "Attention", 18: "SysPrompt (labeled)",
    20: "SysPrompt (talkto human)", 21: "SysPrompt (talkto AI)",
    22: "SysPrompt (bare human)", 23: "SysPrompt (bare AI)",
}

STANDALONE_DIM_NAMES = dict(DIM_NAMES)
STANDALONE_DIM_NAMES[16] = "Human (concept)"
STANDALONE_DIM_NAMES[17] = "AI (concept)"
STANDALONE_DIM_NAMES[18] = "Attention"

EXCLUDE_DIMS = {
    "raw": {16},
    "residual": {16},
    "standalone": set(),
}

PROBE_KEYS = {
    "metacognitive": "boot_metacognitive_r2",
    "operational": "boot_operational_r2",
}

PROBE_LABELS = {
    "metacognitive": "Metacognitive",
    "operational": "Operational",
}


def get_dim_category(dim_id, analysis_type):
    if analysis_type == "standalone":
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 18],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Control":   [14, 15],
            "Entity":    [16, 17],
            "SysPrompt": [20, 21, 22, 23],
        }
    else:
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 17],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Control":   [0, 14, 15],
            "SysPrompt": [18],
        }
    for cat, ids in categories.items():
        if dim_id in ids:
            return cat
    return "Other"


# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

def pairwise_bootstrap_test(boot_a, boot_b):
    """
    Paired bootstrap test: proportion of iterations where A > B.
    Returns dict with mean difference, 95% CI on difference, and p-values.
    """
    diff = boot_a - boot_b
    p_greater = float(np.mean(diff <= 0))   # low = A significantly > B
    p_less = float(np.mean(diff >= 0))      # low = A significantly < B
    p_two_sided = 2.0 * min(p_greater, p_less)
    p_two_sided = min(p_two_sided, 1.0)
    return {
        "mean_diff": float(np.mean(diff)),
        "ci95_lo": float(np.percentile(diff, 2.5)),
        "ci95_hi": float(np.percentile(diff, 97.5)),
        "p_greater": p_greater,
        "p_less": p_less,
        "p_two_sided": p_two_sided,
    }


def benjamini_hochberg(p_values, q=0.05):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list of float
        Raw p-values.
    q : float
        False discovery rate threshold.

    Returns
    -------
    adjusted : list of float
        FDR-adjusted p-values.
    significant : list of bool
        Whether each test is significant after correction.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    # Sort by p-value
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    significant = [False] * n

    # Compute adjusted p-values (step-up)
    prev_adj = 1.0
    for rank_idx in range(n - 1, -1, -1):
        orig_idx, p = indexed[rank_idx]
        rank = rank_idx + 1  # 1-based rank
        adj_p = min(p * n / rank, prev_adj)
        adj_p = min(adj_p, 1.0)
        adjusted[orig_idx] = adj_p
        significant[orig_idx] = adj_p <= q
        prev_adj = adj_p

    return adjusted, significant


# ============================================================================
# DATA LOADING
# ============================================================================

def discover_dimensions(version_dir, analysis_type):
    """Find all dimension directories with alignment.npz files."""
    exclude = EXCLUDE_DIMS.get(analysis_type, set())
    dims = {}
    if not version_dir.exists():
        return dims
    for entry in sorted(version_dir.iterdir()):
        if not entry.is_dir():
            continue
        npz_path = entry / "alignment.npz"
        if not npz_path.exists():
            continue
        # Parse dim_id from folder name (e.g., "1_phenomenology" -> 1)
        parts = entry.name.split("_", 1)
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        if dim_id in exclude:
            continue
        dims[dim_id] = {
            "folder": entry.name,
            "npz_path": npz_path,
        }
    return dims


def load_bootstrap_data(dims):
    """Load bootstrap distributions for all dimensions and probe types."""
    boot_data = {}
    for dim_id, info in dims.items():
        data = np.load(info["npz_path"])
        boot_data[dim_id] = {}
        for probe_name, key in PROBE_KEYS.items():
            if key in data:
                boot_data[dim_id][probe_name] = data[key]
            else:
                print(f"  WARNING: {key} not found in {info['npz_path']}")
    return boot_data


# ============================================================================
# PAIRWISE ANALYSIS
# ============================================================================

def run_pairwise_tests(boot_data, dim_info, analysis_type):
    """
    Run pairwise bootstrap tests for all dimension pairs, for each probe type.

    Returns
    -------
    results : list of dict
        One entry per pair × probe type.
    summary : dict
        Per-dimension summary of significant differences.
    """
    dim_names = STANDALONE_DIM_NAMES if analysis_type == "standalone" else DIM_NAMES
    dim_ids = sorted(boot_data.keys())
    results = []

    # Collect p-values per probe type for FDR correction
    for probe_name in PROBE_KEYS:
        pair_results = []
        raw_p_values = []

        for dim_a, dim_b in combinations(dim_ids, 2):
            if probe_name not in boot_data[dim_a] or probe_name not in boot_data[dim_b]:
                continue
            boot_a = boot_data[dim_a][probe_name]
            boot_b = boot_data[dim_b][probe_name]
            test = pairwise_bootstrap_test(boot_a, boot_b)

            pair_results.append({
                "dim_a_id": dim_a,
                "dim_b_id": dim_b,
                "dim_a_name": dim_names.get(dim_a, dim_info[dim_a]["folder"]),
                "dim_b_name": dim_names.get(dim_b, dim_info[dim_b]["folder"]),
                "dim_a_folder": dim_info[dim_a]["folder"],
                "dim_b_folder": dim_info[dim_b]["folder"],
                "dim_a_category": get_dim_category(dim_a, analysis_type),
                "dim_b_category": get_dim_category(dim_b, analysis_type),
                "probe_type": probe_name,
                **test,
            })
            raw_p_values.append(test["p_two_sided"])

        # FDR correction
        adjusted_p, significant = benjamini_hochberg(raw_p_values, q=0.05)
        for i, pr in enumerate(pair_results):
            pr["p_fdr"] = adjusted_p[i]
            pr["significant_fdr"] = significant[i]
            # Significance stars
            if pr["p_fdr"] <= 0.001:
                pr["stars"] = "***"
            elif pr["p_fdr"] <= 0.01:
                pr["stars"] = "**"
            elif pr["p_fdr"] <= 0.05:
                pr["stars"] = "*"
            else:
                pr["stars"] = "n.s."

        results.extend(pair_results)

    # Build per-dimension summary
    summary = {}
    for dim_id in dim_ids:
        name = dim_names.get(dim_id, dim_info[dim_id]["folder"])
        cat = get_dim_category(dim_id, analysis_type)
        summary[str(dim_id)] = {
            "name": name,
            "folder": dim_info[dim_id]["folder"],
            "category": cat,
            "probes": {},
        }
        for probe_name in PROBE_KEYS:
            sig_greater = []  # dims this one is significantly greater than
            sig_less = []     # dims this one is significantly less than
            for r in results:
                if r["probe_type"] != probe_name or not r["significant_fdr"]:
                    continue
                if r["dim_a_id"] == dim_id:
                    if r["mean_diff"] > 0:
                        sig_greater.append(r["dim_b_name"])
                    else:
                        sig_less.append(r["dim_b_name"])
                elif r["dim_b_id"] == dim_id:
                    if r["mean_diff"] < 0:
                        sig_greater.append(r["dim_a_name"])
                    else:
                        sig_less.append(r["dim_a_name"])
            summary[str(dim_id)]["probes"][probe_name] = {
                "n_sig_greater": len(sig_greater),
                "n_sig_less": len(sig_less),
                "sig_greater_than": sig_greater,
                "sig_less_than": sig_less,
            }

    return results, summary


# ============================================================================
# OUTPUT
# ============================================================================

def save_csv(results, out_path):
    """Save pairwise results as CSV."""
    if not results:
        return
    fieldnames = [
        "probe_type", "dim_a_id", "dim_a_name", "dim_a_category",
        "dim_b_id", "dim_b_name", "dim_b_category",
        "mean_diff", "ci95_lo", "ci95_hi",
        "p_greater", "p_less", "p_two_sided", "p_fdr",
        "significant_fdr", "stars",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def save_json(results, out_path):
    """Save pairwise results as JSON."""
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


def save_summary(summary, out_path):
    """Save per-dimension summary as JSON."""
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# MAIN
# ============================================================================

def process_version(version, analysis_type, turn, out_dir):
    """Run pairwise tests for a single version."""
    subpath = ANALYSIS_TYPES[analysis_type]
    version_dir = ALIGNMENT_ROOT / version / f"turn_{turn}" / subpath

    if not version_dir.exists():
        print(f"  WARNING: {version_dir} not found, skipping")
        return None, None

    dims = discover_dimensions(version_dir, analysis_type)
    if len(dims) < 2:
        print(f"  WARNING: Only {len(dims)} dimensions found, need at least 2")
        return None, None

    print(f"  Found {len(dims)} dimensions")
    boot_data = load_bootstrap_data(dims)

    results, summary = run_pairwise_tests(boot_data, dims, analysis_type)
    n_pairs = len(results) // 2  # 2 probe types
    n_sig = sum(1 for r in results if r["significant_fdr"])
    print(f"  {n_pairs} pairs per probe type, {n_sig} significant (FDR q=0.05)")

    # Save version-specific results
    version_dir_out = out_dir / version
    version_dir_out.mkdir(parents=True, exist_ok=True)

    save_csv(results, version_dir_out / "pairwise_dimensions.csv")
    save_json(results, version_dir_out / "pairwise_dimensions.json")
    save_summary(summary, version_dir_out / "pairwise_summary.json")

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise bootstrap statistical tests between dimension pairs")
    parser.add_argument("--turn", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Conversation turn (default: 5)")
    parser.add_argument("--type", choices=["raw", "residual", "standalone", "all"],
                        default="all", help="Analysis type (default: all)")
    args = parser.parse_args()

    types = ["raw", "residual", "standalone"] if args.type == "all" else [args.type]
    comparisons_root = Path(__file__).resolve().parent.parent

    for atype in types:
        print(f"\n{'='*60}")
        print(f"  Pairwise Tests: {atype} (turn {args.turn})")
        print(f"{'='*60}")

        out_dir = comparisons_root / f"turn_{args.turn}" / atype

        for version in VERSIONS:
            print(f"\n  Version: {version}")
            process_version(version, atype, args.turn, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
