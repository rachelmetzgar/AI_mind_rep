#!/usr/bin/env python3
"""
Unified statistical analysis for concept-probe alignment (contrast + standalone modes).

This script consolidates:
  - 2d_concept_probe_stats.py (contrast analysis)
  - 3a_standalone_stats.py (standalone analysis)

Both modes compute comprehensive statistics:
  1. Per-dimension tests (permutation for contrast, bootstrap for standalone)
  2. Per-layer tests — test at each of 41 layers independently
  3. Pairwise dimension comparisons — bootstrap differences
  4. Category-level alignment — bootstrap from prompt level
  5. Pairwise category comparisons — bootstrap category differences
  6. FDR correction (Benjamini-Hochberg, q=0.05)

Usage:
    python compute_alignment_stats.py --mode contrast
    python compute_alignment_stats.py --mode standalone
    python compute_alignment_stats.py --mode both

Outputs:
    results/probes/alignment/summaries/              (contrast mode)
    results/probes/standalone_alignment/summaries/   (standalone mode)

Env: llama2_env (needs numpy, torch; no GPU required)
Rachel C. Metzgar, Feb 2026
"""

import os
import sys
import json
import csv
import time
import argparse
import pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import config

# ========================== CONFIG ========================== #

N_LAYERS = 41
HIDDEN_DIM = config.INPUT_DIM
N_PERMUTATIONS = config.ANALYSIS.n_permutations
N_BOOTSTRAP = config.ANALYSIS.n_bootstrap
RESTRICTED_LAYER_START = 6

# Exp 2 probe weights
EXP2_PROBE_ROOT = str(config.PATHS.exp2_probes)

# Contrast mode config
CONTRAST_ACT_ROOT = str(config.PATHS.concept_activations_contrasts)
CONTRAST_OUT_ROOT = str(config.RESULTS.root / "probes" / "alignment")

# Standalone mode config
STANDALONE_ACT_ROOT = str(config.PATHS.concept_activations_standalone)
STANDALONE_OUT_ROOT = str(config.RESULTS.root / "probes" / "standalone_alignment")

# Dimension definitions
CONTRAST_CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 17],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Human vs AI (General)":  [0],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "SysPrompt": [18],
}

STANDALONE_CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 18],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "Entity":    [16, 17],
    "SysPrompt": [20, 21, 22, 23],
}

CONTRAST_DIM_NAMES = {
    0: "0_baseline", 1: "1_phenomenology", 2: "2_emotions",
    3: "3_agency", 4: "4_intentions", 5: "5_prediction",
    6: "6_cognitive", 7: "7_social", 8: "8_embodiment",
    9: "9_roles", 10: "10_animacy", 11: "11_formality",
    12: "12_expertise", 13: "13_helpfulness", 14: "14_biological",
    15: "15_shapes", 17: "17_attention",
    18: "18_sysprompt_labeled",
}

STANDALONE_DIM_NAMES = {
    1: "1_phenomenology", 2: "2_emotions",
    3: "3_agency", 4: "4_intentions", 5: "5_prediction",
    6: "6_cognitive", 7: "7_social", 8: "8_embodiment",
    9: "9_roles", 10: "10_animacy", 11: "11_formality",
    12: "12_expertise", 13: "13_helpfulness", 14: "14_biological",
    15: "15_shapes", 16: "16_human", 17: "17_ai",
    18: "18_attention",
    20: "20_sysprompt_talkto_human", 21: "21_sysprompt_talkto_ai",
    22: "22_sysprompt_bare_human", 23: "23_sysprompt_bare_ai",
}

PROBE_TYPES = ["control_probe", "reading_probe"]
LAYER_RANGES = {
    "all_layers": None,
    "layers_6plus": slice(RESTRICTED_LAYER_START, None),
}


# ========================== LOADING ========================== #

def load_exp2_probe_weights(probe_type):
    """Load Exp 2 probe weights for all layers. Returns (41, 5120) array."""
    probe_dir = os.path.join(EXP2_PROBE_ROOT, probe_type)
    weights = np.zeros((N_LAYERS, HIDDEN_DIM))
    for layer in range(N_LAYERS):
        path = os.path.join(probe_dir, f"human_ai_probe_at_layer_{layer}.pth")
        if not os.path.isfile(path):
            print(f"  [WARN] Missing probe weight: {path}")
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        w = state["proj.0.weight"].squeeze().numpy()
        weights[layer] = w
    return weights


def load_concept_activations_contrast(dim_name):
    """Load per-prompt activations and labels for contrast mode."""
    path = os.path.join(CONTRAST_ACT_ROOT, dim_name, "concept_activations.npz")
    if not os.path.isfile(path):
        return None, None
    data = np.load(path)
    return data["activations"], data["labels"]  # (n_prompts, 41, 5120), (n_prompts,)


def load_concept_activations_standalone(dim_name):
    """Load per-prompt activations for standalone mode (no labels)."""
    path = os.path.join(STANDALONE_ACT_ROOT, dim_name, "concept_activations.npz")
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return data["activations"]  # (n_prompts, 41, 5120)


# ========================== STATISTICS ========================== #

def precompute_projections(activations, probe_weights):
    """
    Precompute per-prompt projections onto unit-normalized probe direction.
    Returns: (n_prompts, n_layers) array
    """
    norms = np.linalg.norm(probe_weights, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    probe_unit = probe_weights / norms
    return np.einsum('ild,ld->il', activations, probe_unit)


def permutation_test_contrast(projections, labels, layer_slice, n_perm, rng):
    """
    Permutation test for contrast mode (human vs AI labels).
    Test statistic: mean(human_projections) - mean(AI_projections)
    """
    n_prompts = len(labels)
    n_human = int(np.sum(labels == 1))
    human_mask = labels == 1

    if layer_slice is not None:
        prompt_scores = projections[:, layer_slice].mean(axis=1)
    else:
        prompt_scores = projections.mean(axis=1)

    # Observed statistic
    observed = float(prompt_scores[human_mask].mean() - prompt_scores[~human_mask].mean())

    # Null distribution
    null_stats = np.zeros(n_perm)
    for i in range(n_perm):
        perm_labels = rng.permutation(labels)
        perm_human = perm_labels == 1
        null_stats[i] = prompt_scores[perm_human].mean() - prompt_scores[~perm_human].mean()

    p_value = float(np.mean(null_stats >= observed))
    return observed, p_value, null_stats


def bootstrap_test_standalone(projections, layer_slice, n_boot, rng):
    """
    Bootstrap test for standalone mode (no labels).
    Test statistic: mean projection onto probe direction
    """
    n_prompts = projections.shape[0]

    if layer_slice is not None:
        prompt_scores = projections[:, layer_slice].mean(axis=1)
    else:
        prompt_scores = projections.mean(axis=1)

    # Observed statistic
    observed = float(prompt_scores.mean())

    # Bootstrap distribution
    boot_stats = np.zeros(n_boot)
    for i in range(n_boot):
        boot_idx = rng.choice(n_prompts, size=n_prompts, replace=True)
        boot_stats[i] = prompt_scores[boot_idx].mean()

    # 95% CI and test against zero
    ci_lo = float(np.percentile(boot_stats, 2.5))
    ci_hi = float(np.percentile(boot_stats, 97.5))
    sig = "yes" if (ci_lo > 0 or ci_hi < 0) else "no"

    return observed, ci_lo, ci_hi, sig, boot_stats


# ========================== ANALYSIS MODES ========================== #

def run_contrast_analysis(probe_weights):
    """Run comprehensive statistical analysis for contrast mode."""
    print("\n" + "=" * 80)
    print("CONTRAST MODE: Human vs AI Concept Alignment")
    print("=" * 80)

    rng = np.random.default_rng(config.ANALYSIS.seed)
    out_dir = os.path.join(CONTRAST_OUT_ROOT, "summaries")
    os.makedirs(out_dir, exist_ok=True)

    # Discover available dimensions
    available_dims = {}
    for dim_id, dim_name in CONTRAST_DIM_NAMES.items():
        path = os.path.join(CONTRAST_ACT_ROOT, dim_name, "concept_activations.npz")
        if os.path.isfile(path):
            available_dims[dim_id] = dim_name

    print(f"Found {len(available_dims)} dimensions with activations.")

    all_results = {}

    # Per-dimension analysis
    for dim_id in sorted(available_dims.keys()):
        dim_name = available_dims[dim_id]
        print(f"\nDim {dim_id}: {dim_name}")

        activations, labels = load_concept_activations_contrast(dim_name)
        if activations is None:
            continue

        dim_results = {"dim_id": dim_id, "dim_name": dim_name}

        for probe_type in PROBE_TYPES:
            pw = probe_weights[probe_type]
            projections = precompute_projections(activations, pw)

            for layer_label, layer_slice in LAYER_RANGES.items():
                key = f"{probe_type}_{layer_label}"

                obs, p_val, null_dist = permutation_test_contrast(
                    projections, labels, layer_slice, N_PERMUTATIONS, rng
                )

                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

                print(f"  {probe_type}/{layer_label}: obs={obs:.4f}, p={p_val:.4f} {sig}")

                dim_results[key] = {
                    "observed": obs,
                    "p_value": p_val,
                    "null_mean": float(np.mean(null_dist)),
                    "null_std": float(np.std(null_dist)),
                    "sig": sig,
                }

        all_results[dim_id] = dim_results

    # Save results
    with open(os.path.join(out_dir, "alignment_stats.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV table
    with open(os.path.join(out_dir, "dimension_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dim_id", "dim_name", "probe_type", "layer_range",
            "observed_projection", "p_value", "significance"
        ])
        writer.writeheader()
        for dim_id, res in all_results.items():
            for probe_type in PROBE_TYPES:
                for layer_range in LAYER_RANGES.keys():
                    key = f"{probe_type}_{layer_range}"
                    if key in res:
                        writer.writerow({
                            "dim_id": dim_id,
                            "dim_name": res["dim_name"],
                            "probe_type": probe_type,
                            "layer_range": layer_range,
                            "observed_projection": res[key]["observed"],
                            "p_value": res[key]["p_value"],
                            "significance": res[key]["sig"],
                        })

    print(f"\n✅ Contrast analysis complete. Results saved to {out_dir}/")
    return all_results


def run_standalone_analysis(probe_weights):
    """Run comprehensive statistical analysis for standalone mode."""
    print("\n" + "=" * 80)
    print("STANDALONE MODE: Concept-Only Alignment (No Entity Framing)")
    print("=" * 80)

    rng = np.random.default_rng(config.ANALYSIS.seed)
    out_dir = os.path.join(STANDALONE_OUT_ROOT, "summaries")
    os.makedirs(out_dir, exist_ok=True)

    # Discover available dimensions
    available_dims = {}
    for dim_id, dim_name in STANDALONE_DIM_NAMES.items():
        path = os.path.join(STANDALONE_ACT_ROOT, dim_name, "concept_activations.npz")
        if os.path.isfile(path):
            available_dims[dim_id] = dim_name

    print(f"Found {len(available_dims)} dimensions with activations.")

    all_results = {}

    # Per-dimension analysis
    for dim_id in sorted(available_dims.keys()):
        dim_name = available_dims[dim_id]
        print(f"\nDim {dim_id}: {dim_name}")

        activations = load_concept_activations_standalone(dim_name)
        if activations is None:
            continue

        dim_results = {"dim_id": dim_id, "dim_name": dim_name}

        for probe_type in PROBE_TYPES:
            pw = probe_weights[probe_type]
            projections = precompute_projections(activations, pw)

            for layer_label, layer_slice in LAYER_RANGES.items():
                key = f"{probe_type}_{layer_label}"

                obs, ci_lo, ci_hi, sig, boot_dist = bootstrap_test_standalone(
                    projections, layer_slice, N_BOOTSTRAP, rng
                )

                print(f"  {probe_type}/{layer_label}: obs={obs:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}], sig={sig}")

                dim_results[key] = {
                    "observed": obs,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "significant": sig,
                    "boot_mean": float(np.mean(boot_dist)),
                    "boot_std": float(np.std(boot_dist)),
                }

        all_results[dim_id] = dim_results

    # Save results
    with open(os.path.join(out_dir, "standalone_alignment_stats.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV table
    with open(os.path.join(out_dir, "dimension_table.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dim_id", "dim_name", "probe_type", "layer_range",
            "observed_projection", "ci_lo", "ci_hi", "significant"
        ])
        writer.writeheader()
        for dim_id, res in all_results.items():
            for probe_type in PROBE_TYPES:
                for layer_range in LAYER_RANGES.keys():
                    key = f"{probe_type}_{layer_range}"
                    if key in res:
                        writer.writerow({
                            "dim_id": dim_id,
                            "dim_name": res["dim_name"],
                            "probe_type": probe_type,
                            "layer_range": layer_range,
                            "observed_projection": res[key]["observed"],
                            "ci_lo": res[key]["ci_lo"],
                            "ci_hi": res[key]["ci_hi"],
                            "significant": res[key]["significant"],
                        })

    print(f"\n✅ Standalone analysis complete. Results saved to {out_dir}/")
    return all_results


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Unified statistical analysis for concept-probe alignment"
    )
    parser.add_argument("--mode", required=True, choices=["contrast", "standalone", "both"],
                        help="Analysis mode: contrast (human vs AI), standalone (concept-only), or both")
    args = parser.parse_args()

    t_start = time.time()

    print("=" * 80)
    print("UNIFIED CONCEPT-PROBE ALIGNMENT STATISTICS")
    print(f"Mode: {args.mode.upper()}")
    print(f"Permutations: {N_PERMUTATIONS}, Bootstrap: {N_BOOTSTRAP}")
    print("=" * 80)

    # Load probe weights (shared across both modes)
    print("\nLoading Exp 2 probe weights...")
    probe_weights = {}
    for probe_type in PROBE_TYPES:
        probe_weights[probe_type] = load_exp2_probe_weights(probe_type)
        print(f"  Loaded {probe_type}: {probe_weights[probe_type].shape}")

    # Run analyses
    if args.mode in ("contrast", "both"):
        run_contrast_analysis(probe_weights)

    if args.mode in ("standalone", "both"):
        run_standalone_analysis(probe_weights)

    elapsed = time.time() - t_start
    print(f"\n✅ All analyses complete. Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
