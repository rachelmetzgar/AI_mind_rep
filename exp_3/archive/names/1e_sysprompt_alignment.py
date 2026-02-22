#!/usr/bin/env python3
"""
Experiment 3, Phase 1e: System Prompt ↔ Concept Alignment Analysis

Computes alignment between system prompt representations (from 1d) and
concept dimension representations (from 1/1a), answering:

    1. How much does each concept dimension overlap with the system prompt
       partner identity representation?
    2. Is this overlap above chance (permutation baseline)?
    3. Do some concepts overlap more than others (pairwise bootstrap tests)?

Three comparison types:

    A. CONTRAST × CONTRAST
       System prompt contrast vector (human-labeled − AI-labeled names)
       vs. concept contrast vectors (human − AI for each dimension).
       → Does the human/AI distinction in names align with the human/AI
         distinction in phenomenology, emotions, etc.?

    B. STANDALONE × STANDALONE (per entity type)
       System prompt standalone human names vs. standalone concept dims
       System prompt standalone AI names vs. standalone concept dims
       → Which concepts are most activated by human vs AI names?

    C. BARE × STANDALONE
       Bare human/AI names vs. standalone concept dims
       → Does conceptual structure emerge from names alone?

For each comparison, outputs:
    - Per-layer cosine similarity (and R²)
    - Mean-across-layers summary
    - Permutation-based null distribution (shuffle prompt labels)
    - Bootstrap CIs on alignment
    - Pairwise dimension comparisons (bootstrap difference tests)

Usage:
    python 1e_sysprompt_concept_alignment.py --analysis all
    python 1e_sysprompt_concept_alignment.py --analysis contrast
    python 1e_sysprompt_concept_alignment.py --analysis standalone
    python 1e_sysprompt_concept_alignment.py --analysis bare

Env: llama2_env (CPU only — just vector math)
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))


# ========================== CONFIG ========================== #

CONCEPT_ACT_DIR = "data/concept_activations"
CONTRAST_DIR = os.path.join(CONCEPT_ACT_DIR, "contrasts")
STANDALONE_DIR = os.path.join(CONCEPT_ACT_DIR, "standalone")
OUTPUT_ROOT = "data/alignment_results/sysprompt"

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 1000
SEED = 42

# System prompt dimension IDs (from 1d)
SYSPROMPT_CONTRAST_DIM = "18_sysprompt_labeled"
SYSPROMPT_TALKTO_HUMAN = "20_sysprompt_talkto_human"
SYSPROMPT_TALKTO_AI = "21_sysprompt_talkto_ai"
SYSPROMPT_BARE_HUMAN = "22_sysprompt_bare_human"
SYSPROMPT_BARE_AI = "23_sysprompt_bare_ai"

# Concept dimensions to compare against
CONCEPT_CONTRAST_DIMS = [
    "0_baseline", "1_phenomenology", "2_emotions", "3_agency",
    "4_intentions", "5_prediction", "6_cognitive", "7_social",
    "8_embodiment", "9_roles", "10_animacy",
    "11_formality", "12_expertise", "13_helpfulness",
    "14_biological", "15_shapes",
]

CONCEPT_STANDALONE_DIMS = [
    "1_phenomenology", "2_emotions", "3_agency",
    "4_intentions", "5_prediction", "6_cognitive", "7_social",
    "8_embodiment", "9_roles", "10_animacy",
    "11_formality", "12_expertise", "13_helpfulness",
    "14_biological", "15_shapes",
]

# Tier labels for display
TIER_MAP = {
    "0_baseline": "Baseline",
    "1_phenomenology": "Mental", "2_emotions": "Mental", "3_agency": "Mental",
    "4_intentions": "Mental", "5_prediction": "Mental", "6_cognitive": "Mental",
    "7_social": "Mental",
    "8_embodiment": "Physical", "9_roles": "Physical", "10_animacy": "Physical",
    "11_formality": "Alternative", "12_expertise": "Alternative",
    "13_helpfulness": "Alternative",
    "14_biological": "Biological", "15_shapes": "Control",
}


# ========================== DATA LOADING ========================== #

def load_contrast_vector(dim_name):
    """Load mean-difference vector (human - AI) per layer."""
    path = os.path.join(CONTRAST_DIR, dim_name, "concept_vector_per_layer.npz")
    return np.load(path)["concept_direction"]  # (n_layers, hidden_dim)


def load_contrast_activations(dim_name):
    """Load raw activations and labels for a contrast dimension."""
    path = os.path.join(CONTRAST_DIR, dim_name, "concept_activations.npz")
    data = np.load(path)
    return data["activations"], data["labels"]


def load_standalone_mean(dim_name):
    """Load mean activation vector per layer for a standalone dim."""
    path = os.path.join(STANDALONE_DIR, dim_name, "mean_vectors_per_layer.npz")
    return np.load(path)["mean_concept"]  # (n_layers, hidden_dim)


def load_standalone_activations(dim_name):
    """Load raw activations for a standalone dimension."""
    path = os.path.join(STANDALONE_DIR, dim_name, "concept_activations.npz")
    return np.load(path)["activations"]  # (n_prompts, n_layers, hidden_dim)


def discover_available(base_dir, candidates):
    """Filter candidate dim names to those with data on disk."""
    available = []
    for dim_name in candidates:
        if os.path.isdir(os.path.join(base_dir, dim_name)):
            available.append(dim_name)
        else:
            print(f"  [SKIP] {dim_name} not found in {base_dir}")
    return available


# ========================== ALIGNMENT MATH ========================== #

def cosine_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def alignment_per_layer(vec_a, vec_b):
    """Cosine similarity between two (n_layers, hidden_dim) arrays, per layer."""
    n_layers = min(vec_a.shape[0], vec_b.shape[0])
    return {layer: cosine_sim(vec_a[layer], vec_b[layer]) for layer in range(n_layers)}


def mean_r2(per_layer_cos):
    """Mean R² (cos²) across layers."""
    vals = [c ** 2 for c in per_layer_cos.values()]
    return float(np.mean(vals))


# ========================== STATISTICAL TESTS ========================== #

def bootstrap_alignment_contrast(acts_a, labels_a, acts_b, labels_b,
                                  n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap R² between two contrast dimensions.
    Resamples prompts within each class, recomputes mean-diff vectors,
    then computes alignment.
    """
    rng = np.random.RandomState(SEED)
    h_a = acts_a[labels_a == 1]
    a_a = acts_a[labels_a == 0]
    h_b = acts_b[labels_b == 1]
    a_b = acts_b[labels_b == 0]

    boot_r2 = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Resample within each class for each dimension
        idx_ha = rng.choice(len(h_a), len(h_a), replace=True)
        idx_aa = rng.choice(len(a_a), len(a_a), replace=True)
        idx_hb = rng.choice(len(h_b), len(h_b), replace=True)
        idx_ab = rng.choice(len(a_b), len(a_b), replace=True)

        va = h_a[idx_ha].mean(axis=0) - a_a[idx_aa].mean(axis=0)
        vb = h_b[idx_hb].mean(axis=0) - a_b[idx_ab].mean(axis=0)

        per_layer = alignment_per_layer(va, vb)
        boot_r2[b] = mean_r2(per_layer)

    return boot_r2


def bootstrap_alignment_standalone(acts_a, acts_b, n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap R² between two standalone dimensions.
    Resamples prompts, recomputes mean vectors, then alignment.
    """
    rng = np.random.RandomState(SEED)
    boot_r2 = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx_a = rng.choice(len(acts_a), len(acts_a), replace=True)
        idx_b = rng.choice(len(acts_b), len(acts_b), replace=True)
        ma = acts_a[idx_a].mean(axis=0)
        mb = acts_b[idx_b].mean(axis=0)
        per_layer = alignment_per_layer(ma, mb)
        boot_r2[b] = mean_r2(per_layer)

    return boot_r2


def permutation_null_contrast(acts_a, labels_a, vec_b, n_perms=N_PERMUTATIONS):
    """
    Permutation null: shuffle labels within dimension A, recompute contrast
    vector, measure alignment with vec_b. Tests whether alignment exceeds
    what you'd get from random label assignments.
    """
    rng = np.random.RandomState(SEED)
    null_r2 = np.zeros(n_perms)

    for p in range(n_perms):
        shuffled = rng.permutation(labels_a)
        h = acts_a[shuffled == 1]
        a = acts_a[shuffled == 0]
        v_shuf = h.mean(axis=0) - a.mean(axis=0)
        per_layer = alignment_per_layer(v_shuf, vec_b)
        null_r2[p] = mean_r2(per_layer)

    return null_r2


def permutation_null_standalone(acts_a, mean_b, n_perms=N_PERMUTATIONS):
    """
    Permutation null for standalone: randomly subsample half of prompts
    from A, compute mean, measure alignment with B. Establishes baseline
    variability from prompt sampling noise.
    """
    rng = np.random.RandomState(SEED)
    null_r2 = np.zeros(n_perms)
    n = len(acts_a)
    half = n // 2

    for p in range(n_perms):
        idx = rng.choice(n, half, replace=False)
        m = acts_a[idx].mean(axis=0)
        per_layer = alignment_per_layer(m, mean_b)
        null_r2[p] = mean_r2(per_layer)

    return null_r2


def pairwise_bootstrap_test(boot_a, boot_b):
    """
    Paired bootstrap test: proportion of iterations where A > B.
    Returns p-value for the hypothesis that A <= B.
    """
    diff = boot_a - boot_b
    p_value = float(np.mean(diff <= 0))
    return {
        "mean_diff": float(np.mean(diff)),
        "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "p_greater": p_value,  # p(A <= B), i.e. low = A > B
    }


# ========================== ANALYSIS RUNNERS ========================== #

def run_contrast_analysis():
    """Compare system prompt contrast vector with concept contrast vectors."""
    print("\n" + "=" * 60)
    print("CONTRAST × CONTRAST ALIGNMENT")
    print("=" * 60)

    out_dir = os.path.join(OUTPUT_ROOT, "contrast_x_contrast")
    os.makedirs(out_dir, exist_ok=True)

    # Load system prompt contrast
    sp_vec = load_contrast_vector(SYSPROMPT_CONTRAST_DIM)
    sp_acts, sp_labels = load_contrast_activations(SYSPROMPT_CONTRAST_DIM)
    print(f"Loaded sysprompt contrast: {sp_vec.shape}")

    available = discover_available(CONTRAST_DIR, CONCEPT_CONTRAST_DIMS)
    summary = {}
    boot_cache = {}  # for pairwise comparisons

    for dim_name in available:
        print(f"\n--- {dim_name} ({TIER_MAP.get(dim_name, '?')}) ---")
        c_vec = load_contrast_vector(dim_name)
        c_acts, c_labels = load_contrast_activations(dim_name)

        # Point estimate
        per_layer = alignment_per_layer(sp_vec, c_vec)
        r2 = mean_r2(per_layer)

        # Bootstrap
        boot = bootstrap_alignment_contrast(sp_acts, sp_labels, c_acts, c_labels)
        boot_cache[dim_name] = boot

        # Permutation null
        null = permutation_null_contrast(c_acts, c_labels, sp_vec)
        p_val = float(np.mean(null >= r2))

        print(f"  R² = {r2:.6f}  CI95 = [{np.percentile(boot, 2.5):.6f}, "
              f"{np.percentile(boot, 97.5):.6f}]  p(perm) = {p_val:.4f}")

        summary[dim_name] = {
            "tier": TIER_MAP.get(dim_name, "?"),
            "mean_r2": r2,
            "boot_ci95": [float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5))],
            "perm_p": p_val,
            "perm_null_mean": float(np.mean(null)),
            "per_layer_cos": {str(k): v for k, v in per_layer.items()},
        }

    # Pairwise comparisons
    pairwise = {}
    for (a, b) in combinations(boot_cache.keys(), 2):
        result = pairwise_bootstrap_test(boot_cache[a], boot_cache[b])
        pairwise[f"{a}_vs_{b}"] = result

    # Save
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "pairwise.json"), "w") as f:
        json.dump(pairwise, f, indent=2)

    print_summary("CONTRAST × CONTRAST", summary)
    return summary


def run_standalone_analysis(sysprompt_dim, sysprompt_label, out_subdir):
    """Compare one system prompt standalone set with concept standalone dims."""
    print(f"\n{'=' * 60}")
    print(f"STANDALONE ALIGNMENT: {sysprompt_label}")
    print(f"{'=' * 60}")

    out_dir = os.path.join(OUTPUT_ROOT, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Load system prompt standalone
    sp_mean = load_standalone_mean(sysprompt_dim)
    sp_acts = load_standalone_activations(sysprompt_dim)
    print(f"Loaded {sysprompt_dim}: mean shape {sp_mean.shape}, "
          f"{len(sp_acts)} prompts")

    available = discover_available(STANDALONE_DIR, CONCEPT_STANDALONE_DIMS)
    summary = {}
    boot_cache = {}

    for dim_name in available:
        print(f"\n--- {dim_name} ({TIER_MAP.get(dim_name, '?')}) ---")
        c_mean = load_standalone_mean(dim_name)
        c_acts = load_standalone_activations(dim_name)

        # Point estimate
        per_layer = alignment_per_layer(sp_mean, c_mean)
        r2 = mean_r2(per_layer)

        # Bootstrap
        boot = bootstrap_alignment_standalone(sp_acts, c_acts)
        boot_cache[dim_name] = boot

        # Permutation null
        null = permutation_null_standalone(sp_acts, c_mean)
        p_val = float(np.mean(null >= r2))

        print(f"  R² = {r2:.6f}  CI95 = [{np.percentile(boot, 2.5):.6f}, "
              f"{np.percentile(boot, 97.5):.6f}]  p(perm) = {p_val:.4f}")

        summary[dim_name] = {
            "tier": TIER_MAP.get(dim_name, "?"),
            "mean_r2": r2,
            "boot_ci95": [float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5))],
            "perm_p": p_val,
            "perm_null_mean": float(np.mean(null)),
            "per_layer_cos": {str(k): v for k, v in per_layer.items()},
        }

    # Pairwise comparisons
    pairwise = {}
    for (a, b) in combinations(boot_cache.keys(), 2):
        result = pairwise_bootstrap_test(boot_cache[a], boot_cache[b])
        pairwise[f"{a}_vs_{b}"] = result

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "pairwise.json"), "w") as f:
        json.dump(pairwise, f, indent=2)

    print_summary(f"STANDALONE: {sysprompt_label}", summary)
    return summary


# ========================== DISPLAY ========================== #

def print_summary(title, summary):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Dimension':<25} {'Tier':<12} {'R²':>10} "
          f"{'95% CI':>22} {'p(perm)':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*22} {'-'*10}")

    for dim_name in sorted(summary.keys(),
                           key=lambda x: summary[x]["mean_r2"], reverse=True):
        s = summary[dim_name]
        ci = s["boot_ci95"]
        print(f"  {dim_name:<25} {s['tier']:<12} {s['mean_r2']:>10.6f} "
              f"[{ci[0]:.6f}, {ci[1]:.6f}] {s['perm_p']:>10.4f}")
    print(f"{'=' * 70}\n")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1e: System prompt ↔ concept alignment"
    )
    parser.add_argument("--analysis", required=True,
                        choices=["contrast", "standalone", "bare", "all"],
                        help="Which comparison to run")
    args = parser.parse_args()

    if args.analysis in ("contrast", "all"):
        run_contrast_analysis()

    if args.analysis in ("standalone", "all"):
        run_standalone_analysis(
            SYSPROMPT_TALKTO_HUMAN,
            "talkto human names",
            "standalone_talkto_human")
        run_standalone_analysis(
            SYSPROMPT_TALKTO_AI,
            "talkto AI names",
            "standalone_talkto_ai")

    if args.analysis in ("bare", "all"):
        run_standalone_analysis(
            SYSPROMPT_BARE_HUMAN,
            "bare human names",
            "standalone_bare_human")
        run_standalone_analysis(
            SYSPROMPT_BARE_AI,
            "bare AI names",
            "standalone_bare_ai")

    print("✅ Phase 1e complete.")


if __name__ == "__main__":
    main()