#!/usr/bin/env python3
"""
Matched Concept Behavioral PCA Reanalysis

Reads saved pairwise_raw_responses.json from the full behavioral PCA,
filters to matched concept subsets (human-favored, AI-favored, directional),
and runs PCA + categorical analysis on each subset separately.

CPU-only — no model loading.

Subsets:
    1. human-favored (12 concepts from MATCHED_HUMAN)
    2. ai-favored (6 concepts from MATCHED_AI)
    3. directional (human + AI = 18 concepts, excluding ambiguous)
    4. all (full set — same as original, for comparison)

For each subset:
    - Filter responses to matching concepts
    - compute_character_means_pairwise() on filtered responses
    - run_pca_varimax() on the subset means matrix
    - compute_categorical_analysis() (Mann-Whitney on factor scores)
    - Save per-subset NPZ + JSON

Output:
    results/{model}/concept_geometry/pca/behavioral/data/
        matched_{subset}_pca_results.npz
        matched_{subset}_categorical_analysis.json
        matched_{subset}_character_means.npz

Usage:
    python concept_geometry/pca/matched_behavioral_pca.py --model llama2_13b_base
    python concept_geometry/pca/matched_behavioral_pca.py --model llama2_13b_chat --both

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir,
)
from concept_geometry.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS, CHARACTER_TYPES,
)
from concept_geometry.concepts import (
    CONCEPT_KEYS, MATCHED_HUMAN, MATCHED_AI, MATCHED_AMBIGUOUS,
)
from utils.utils import (
    run_pca_varimax,
    compute_character_means_pairwise,
)
from scipy.stats import mannwhitneyu


# ============================================================================
# SUBSET DEFINITIONS
# ============================================================================

SUBSETS = {
    "human": MATCHED_HUMAN,
    "ai": MATCHED_AI,
    "directional": MATCHED_HUMAN + MATCHED_AI,
    "all": list(CONCEPT_KEYS),
}


def compute_categorical_analysis(pca_results, char_keys):
    """Mann-Whitney U tests on factor scores for AI vs human groups."""
    scores_01 = pca_results["factor_scores_01"]
    n_factors = scores_01.shape[1]

    char_to_idx = {k: i for i, k in enumerate(char_keys)}
    ai_indices = [char_to_idx[k] for k in char_keys if k in AI_CHARACTERS]
    human_indices = [char_to_idx[k] for k in char_keys if k in HUMAN_CHARACTERS]

    results = {"factors": [], "anomalies": []}

    for fi in range(min(n_factors, 4)):
        ai_scores = scores_01[ai_indices, fi]
        human_scores = scores_01[human_indices, fi]

        try:
            u_stat, p_val = mannwhitneyu(ai_scores, human_scores,
                                         alternative="two-sided")
        except ValueError:
            u_stat, p_val = float("nan"), float("nan")

        results["factors"].append({
            "factor": fi + 1,
            "ai_mean": float(np.mean(ai_scores)),
            "ai_std": float(np.std(ai_scores)),
            "human_mean": float(np.mean(human_scores)),
            "human_std": float(np.std(human_scores)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "separation": abs(float(np.mean(ai_scores)) -
                              float(np.mean(human_scores))),
        })

    # Anomaly detection
    for fi in range(min(n_factors, 2)):
        ai_mean = np.mean(scores_01[ai_indices, fi])
        human_mean = np.mean(scores_01[human_indices, fi])

        for idx, char_key in enumerate(char_keys):
            score = scores_01[idx, fi]
            char_type = CHARACTER_TYPES[char_key]
            if char_type == "ai":
                own_dist = abs(score - ai_mean)
                other_dist = abs(score - human_mean)
            else:
                own_dist = abs(score - human_mean)
                other_dist = abs(score - ai_mean)
            if other_dist < own_dist:
                results["anomalies"].append({
                    "character": char_key,
                    "type": char_type,
                    "factor": fi + 1,
                    "score": float(score),
                    "own_group_mean": float(
                        ai_mean if char_type == "ai" else human_mean),
                    "other_group_mean": float(
                        human_mean if char_type == "ai" else ai_mean),
                })

    return results


def run_subset_analysis(responses, char_keys, concept_subset, subset_name,
                        ddir, is_chat):
    """Run PCA + categorical analysis on a concept subset."""
    rating_key = "rating" if is_chat else "expected_rating"

    # Filter responses to this subset
    subset_set = set(concept_subset)
    filtered = [r for r in responses if r["capacity"] in subset_set]

    if not filtered:
        print(f"  {subset_name}: no responses found, skipping")
        return None

    n_concepts = len(concept_subset)
    print(f"\n  {subset_name}: {n_concepts} concepts, "
          f"{len(filtered)} responses")

    # Character means
    means = compute_character_means_pairwise(
        filtered, char_keys, concept_subset, rating_key=rating_key
    )

    # Need at least 2 concepts for PCA
    if means.shape[0] < 2:
        print(f"    Too few concepts for PCA, skipping")
        return None

    # PCA with varimax
    pca_results = run_pca_varimax(means)

    # Categorical analysis
    cat_analysis = compute_categorical_analysis(pca_results, char_keys)

    # Save
    prefix = f"matched_{subset_name}"
    np.savez_compressed(
        os.path.join(str(ddir), f"{prefix}_character_means.npz"),
        means=means,
        character_keys=np.array(char_keys),
        concept_keys=np.array(concept_subset),
    )
    np.savez_compressed(
        os.path.join(str(ddir), f"{prefix}_pca_results.npz"),
        rotated_loadings=pca_results["rotated_loadings"],
        unrotated_loadings=pca_results["unrotated_loadings"],
        factor_scores_raw=pca_results["factor_scores_raw"],
        factor_scores_01=pca_results["factor_scores_01"],
        eigenvalues=pca_results["eigenvalues"],
        explained_var_ratio=pca_results["explained_var_ratio"],
        character_keys=np.array(char_keys),
        concept_keys=np.array(concept_subset),
    )
    with open(os.path.join(str(ddir), f"{prefix}_categorical_analysis.json"), "w") as fh:
        json.dump({
            "categorical": cat_analysis,
            "subset_name": subset_name,
            "n_concepts": n_concepts,
        }, fh, indent=2)

    # Print summary
    for finfo in cat_analysis["factors"][:2]:
        print(f"    F{finfo['factor']}: AI={finfo['ai_mean']:.3f}, "
              f"Human={finfo['human_mean']:.3f}, "
              f"p={finfo['p_value']:.4f}")

    return pca_results


def generate_for_model(model_key):
    """Run matched behavioral PCA for one model."""
    set_model(model_key)
    is_chat = config.IS_CHAT

    ddir = data_dir("concept_geometry/pca", "behavioral")

    # Load raw responses
    raw_path = os.path.join(str(ddir), "pairwise_raw_responses.json")
    if not os.path.exists(raw_path):
        print(f"Raw responses not found at {raw_path} — skipping {model_key}")
        return

    with open(raw_path) as f:
        responses = json.load(f)

    print(f"Loaded {len(responses)} pairwise responses for {config.MODEL_LABEL}")

    # Get character keys from responses
    char_keys_set = set()
    for r in responses:
        char_keys_set.add(r["entity_a"])
        char_keys_set.add(r["entity_b"])

    # Use consistent ordering from ALL_CHARACTERS
    from concept_geometry.characters import ALL_CHARACTERS
    char_keys = [k for k in ALL_CHARACTERS if k in char_keys_set]

    # Run each subset
    for subset_name, concept_list in SUBSETS.items():
        # Filter to concepts that actually appear in responses
        available = set(r["capacity"] for r in responses)
        filtered_concepts = [c for c in concept_list if c in available]
        if not filtered_concepts:
            print(f"\n  {subset_name}: no matching concepts in data, skipping")
            continue
        run_subset_analysis(responses, char_keys, filtered_concepts,
                            subset_name, ddir, is_chat)

    print(f"\nAll subsets saved to {ddir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Matched concept behavioral PCA reanalysis"
    )
    add_model_argument(parser)
    parser.add_argument("--both", action="store_true",
                        help="Run for both models")
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Matched Behavioral PCA: {model_key}")
        print(f"{'='*60}")
        generate_for_model(model_key)


if __name__ == "__main__":
    main()
