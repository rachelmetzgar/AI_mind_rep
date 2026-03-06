#!/usr/bin/env python3
"""
Experiment 4, Phase 1b: RSA by Dimension (Chat Model Reanalysis)

Loads existing saved activations from data/entity_activations/{tag}/ and
recomputes RSA with three human RDM variants:
  - Combined: Euclidean distance in 2D (Experience, Agency) space
  - Experience-only: |exp_i - exp_j|
  - Agency-only: |agency_i - agency_j|

No GPU needed — just loads .npz activations and computes correlations.

Output:
    data/entity_activations/{tag}/
        rsa_results.json             # all 3 RSA variants per layer
        rdm_cosine_per_layer.npz     # updated with all 3 human RDMs

    results/{tag}/
        rsa_results.json             # copy for convenience

Usage:
    python 1b_rsa_by_dimension.py                  # 12 entities (no self)
    python 1b_rsa_by_dimension.py --include_self   # 13 entities
    python 1b_rsa_by_dimension.py --both            # run both conditions

Env: llama2_env (CPU only)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from scipy.stats import spearmanr

# ── Local imports ──
sys.path.insert(0, os.path.dirname(__file__))
from entities.gray_entities import (
    GRAY_ET_AL_SCORES,
    ENTITY_NAMES,
)


# ========================== RDM ========================== #

def compute_rdm_cosine(entity_activations):
    """
    Compute representational dissimilarity matrix (cosine distance)
    at each layer.
    """
    n_entities, n_layers, hidden_dim = entity_activations.shape
    rdm = np.zeros((n_layers, n_entities, n_entities))

    for layer in range(n_layers):
        vecs = entity_activations[:, layer, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_normed = vecs / norms
        cos_sim = vecs_normed @ vecs_normed.T
        rdm[layer] = 1.0 - cos_sim

    return rdm


def compute_human_rdm_combined(entity_keys):
    """Euclidean distance in 2D (Experience, Agency) space."""
    n = len(entity_keys)
    coords = np.array([GRAY_ET_AL_SCORES[k] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    return rdm


def compute_human_rdm_experience(entity_keys):
    """Absolute difference in Experience scores only."""
    n = len(entity_keys)
    exp_scores = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(exp_scores[i] - exp_scores[j])
    return rdm


def compute_human_rdm_agency(entity_keys):
    """Absolute difference in Agency scores only."""
    n = len(entity_keys)
    agency_scores = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(agency_scores[i] - agency_scores[j])
    return rdm


# ========================== RSA ========================== #

def compute_rsa_all_layers(model_rdm, human_rdm, n_entities):
    """
    Spearman correlation between upper triangles of model and human RDMs
    at every layer.
    """
    n_layers = model_rdm.shape[0]
    triu_idx = np.triu_indices(n_entities, k=1)
    human_upper = human_rdm[triu_idx]
    n_pairs = len(human_upper)

    results = []
    for layer in range(n_layers):
        model_upper = model_rdm[layer][triu_idx]
        if np.std(model_upper) < 1e-12:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(model_upper, human_upper)
        results.append({
            "layer": layer,
            "rho": float(rho),
            "p_value": float(p),
            "n_pairs": n_pairs,
        })

    return results


# ========================== MAIN ========================== #

def run_condition(tag, entity_keys):
    """Run RSA reanalysis for one condition (with_self or without_self)."""
    n_entities = len(entity_keys)
    data_dir = os.path.join("data", "entity_activations", tag)
    results_dir = os.path.join("results", tag)
    os.makedirs(results_dir, exist_ok=True)

    # Load existing activations
    act_path = os.path.join(data_dir, "all_entity_activations.npz")
    if not os.path.exists(act_path):
        print(f"  ERROR: {act_path} not found. Skipping {tag}.")
        return

    print(f"\n{'='*60}")
    print(f"  Condition: {tag} ({n_entities} entities)")
    print(f"{'='*60}")

    data = np.load(act_path)
    acts_array = data["activations"]
    saved_keys = list(data["entity_keys"])
    print(f"  Loaded activations: {acts_array.shape}")
    print(f"  Entity keys: {saved_keys}")

    # Verify entity ordering matches
    assert saved_keys == entity_keys, \
        f"Entity key mismatch: saved={saved_keys}, expected={entity_keys}"

    # Compute model RDM from activations
    print("  Computing model RDM (cosine distance)...")
    model_rdm = compute_rdm_cosine(acts_array)

    # Compute all 3 human RDMs
    print("  Computing human RDMs (combined, experience, agency)...")
    human_rdm_combined = compute_human_rdm_combined(entity_keys)
    human_rdm_experience = compute_human_rdm_experience(entity_keys)
    human_rdm_agency = compute_human_rdm_agency(entity_keys)

    # RSA for all 3 variants
    print("  Computing RSA (Spearman) at all layers...")
    rsa_combined = compute_rsa_all_layers(model_rdm, human_rdm_combined,
                                          n_entities)
    rsa_experience = compute_rsa_all_layers(model_rdm, human_rdm_experience,
                                            n_entities)
    rsa_agency = compute_rsa_all_layers(model_rdm, human_rdm_agency,
                                        n_entities)

    rsa_all = {
        "combined": rsa_combined,
        "experience": rsa_experience,
        "agency": rsa_agency,
    }

    # Print summaries
    for variant_name, rsa_results in rsa_all.items():
        valid = [r for r in rsa_results if not np.isnan(r["rho"])]
        if valid:
            peak = max(valid, key=lambda r: r["rho"])
            print(f"    {variant_name}: Peak Layer {peak['layer']}, "
                  f"rho = {peak['rho']:+.4f}, p = {peak['p_value']:.4f}")

    # Verify combined matches existing rsa_all_layers.json
    existing_rsa_path = os.path.join(results_dir, "rsa_all_layers.json")
    if os.path.exists(existing_rsa_path):
        with open(existing_rsa_path) as f:
            existing_rsa = json.load(f)
        print("\n  Verifying combined RSA matches existing rsa_all_layers.json...")
        max_diff = 0.0
        for old, new in zip(existing_rsa, rsa_combined):
            if not (np.isnan(old["rho"]) and np.isnan(new["rho"])):
                diff = abs(old["rho"] - new["rho"])
                max_diff = max(max_diff, diff)
        print(f"    Max rho difference: {max_diff:.2e}")
        if max_diff < 1e-3:
            print("    PASS: Combined variant matches existing results "
                  "(within float16 round-trip tolerance).")
        else:
            print("    WARNING: Large differences detected. Check activations.")

    # Save rsa_results.json to data dir
    rsa_data_path = os.path.join(data_dir, "rsa_results.json")
    with open(rsa_data_path, "w") as f:
        json.dump(rsa_all, f, indent=2)
    print(f"  Saved: {rsa_data_path}")

    # Save rsa_results.json to results dir
    rsa_results_path = os.path.join(results_dir, "rsa_results.json")
    with open(rsa_results_path, "w") as f:
        json.dump(rsa_all, f, indent=2)
    print(f"  Saved: {rsa_results_path}")

    # Update rdm npz with all 3 human RDMs
    rdm_path = os.path.join(data_dir, "rdm_cosine_per_layer.npz")
    np.savez_compressed(
        rdm_path,
        model_rdm=model_rdm,
        human_rdm_combined=human_rdm_combined,
        human_rdm_experience=human_rdm_experience,
        human_rdm_agency=human_rdm_agency,
        entity_keys=np.array(entity_keys),
    )
    print(f"  Updated: {rdm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 1b: RSA by dimension (chat model reanalysis)"
    )
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both with_self and without_self conditions"
    )
    args = parser.parse_args()

    if args.both:
        entity_keys_no_self = [k for k in ENTITY_NAMES if k != "you_self"]
        run_condition("without_self", entity_keys_no_self)
        run_condition("with_self", ENTITY_NAMES)
    elif args.include_self:
        run_condition("with_self", ENTITY_NAMES)
    else:
        entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]
        run_condition("without_self", entity_keys)

    print("\nDone.")


if __name__ == "__main__":
    main()
