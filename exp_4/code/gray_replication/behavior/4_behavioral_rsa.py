#!/usr/bin/env python3
"""
Experiment 4: Behavioral RSA for Gray Replication

Computes behavioral representational similarity analysis (RSA) for the
Gray et al. replication. For each model, constructs a behavioral RDM
(entity x entity distance matrix) and Spearman-correlates its upper
triangle with the human RDM from Gray et al. (2007).

Two RDM sources:
    1. Pairwise PCA factor scores (factor_scores_01, 13x2):
       Euclidean distance in 2D factor space.
    2. Individual rating profiles (rating_matrix, 18x13):
       Correlation distance in 18D capacity space. Richer, doesn't
       assume 2-factor structure.

Three human RDM variants:
    - Combined: 2D Euclidean distance in (Experience, Agency) space
    - Experience-only: absolute difference in Experience scores
    - Agency-only: absolute difference in Agency scores

Output:
    data_dir("gray_replication", "behavior", tag)/
        behavioral_rsa_results.json
        behavioral_rdms.npz

Usage:
    python 4_behavioral_rsa.py --model llama2_13b_base
    python 4_behavioral_rsa.py --model llama2_13b_chat --both
    python 4_behavioral_rsa.py --model llama3_8b_instruct --include_self

Env: llama2_env (login node, CPU only — no model loading)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, get_condition_tag,
)
from utils.utils import (
    compute_behavioral_rdm,
    compute_human_rdm_combined,
    compute_human_rdm_experience,
    compute_human_rdm_agency,
)
from entities.gray_entities import GRAY_ET_AL_SCORES


# ============================================================================
# RSA HELPERS
# ============================================================================

def rsa_spearman(model_rdm, human_rdm):
    """Spearman correlation between upper triangles of two RDMs.

    Returns dict with rho, p_value, n_pairs.
    """
    n = model_rdm.shape[0]
    triu_idx = np.triu_indices(n, k=1)
    model_upper = model_rdm[triu_idx]
    human_upper = human_rdm[triu_idx]
    n_pairs = len(model_upper)

    if np.std(model_upper) < 1e-12 or np.std(human_upper) < 1e-12:
        return {"rho": float("nan"), "p_value": float("nan"), "n_pairs": n_pairs}

    rho, p = spearmanr(model_upper, human_upper)
    return {"rho": float(rho), "p_value": float(p), "n_pairs": n_pairs}


def compute_individual_rdm(rating_matrix):
    """Correlation distance RDM from the 18D rating profile per entity.

    Args:
        rating_matrix: (n_capacities, n_entities) — capacities x entities

    Returns:
        (n_entities, n_entities) correlation distance matrix
    """
    # Transpose to (n_entities, n_capacities) for pdist
    profiles = rating_matrix.T
    return squareform(pdist(profiles, metric="correlation"))


# ============================================================================
# RUN ONE CONDITION
# ============================================================================

def run_condition(entity_keys, tag):
    """Run behavioral RSA for one condition (with/without self)."""
    ddir = data_dir("gray_replication", "behavior", tag)
    n_entities = len(entity_keys)
    n_pairs = n_entities * (n_entities - 1) // 2

    print(f"\n{'='*60}")
    print(f"  CONDITION: {tag} ({n_entities} entities, {n_pairs} pairs)")
    print(f"{'='*60}")

    # Build human RDMs
    human_rdm_combined = compute_human_rdm_combined(entity_keys, GRAY_ET_AL_SCORES)
    human_rdm_experience = compute_human_rdm_experience(entity_keys, GRAY_ET_AL_SCORES)
    human_rdm_agency = compute_human_rdm_agency(entity_keys, GRAY_ET_AL_SCORES)
    human_rdms = {
        "combined": human_rdm_combined,
        "experience": human_rdm_experience,
        "agency": human_rdm_agency,
    }

    results = {}
    rdms_to_save = {
        "entity_keys": np.array(entity_keys),
        "human_rdm_combined": human_rdm_combined,
        "human_rdm_experience": human_rdm_experience,
        "human_rdm_agency": human_rdm_agency,
    }

    # ── Source 1: Pairwise PCA factor scores ──
    pca_path = ddir / "pairwise_pca_results.npz"
    if pca_path.exists():
        pca = np.load(pca_path, allow_pickle=True)
        factor_scores = pca["factor_scores_01"]  # (n_entities, 2)
        pca_entity_keys = list(pca["entity_keys"])

        # Verify entity alignment
        if pca_entity_keys == entity_keys:
            model_rdm_pca = compute_behavioral_rdm(factor_scores)
            rdms_to_save["model_rdm_pairwise_pca"] = model_rdm_pca

            pca_results = {}
            for variant_name, human_rdm in human_rdms.items():
                rsa = rsa_spearman(model_rdm_pca, human_rdm)
                pca_results[variant_name] = rsa
                print(f"  Pairwise PCA vs {variant_name}: "
                      f"rho={rsa['rho']:.3f}, p={rsa['p_value']:.4f}")

            results["pairwise_pca"] = pca_results
        else:
            print(f"  WARNING: Entity key mismatch in pairwise PCA — skipping")
    else:
        print(f"  No pairwise_pca_results.npz found — skipping pairwise PCA source")

    # ── Source 2: Individual 18D rating profiles ──
    ind_path = ddir / "individual_rating_matrix.npz"
    if ind_path.exists():
        ind = np.load(ind_path, allow_pickle=True)
        rating_matrix = ind["rating_matrix"]  # (18, n_entities)
        ind_entity_keys = list(ind["entity_keys"])

        if ind_entity_keys == entity_keys:
            model_rdm_ind = compute_individual_rdm(rating_matrix)
            rdms_to_save["model_rdm_individual_18d"] = model_rdm_ind

            ind_results = {}
            for variant_name, human_rdm in human_rdms.items():
                rsa = rsa_spearman(model_rdm_ind, human_rdm)
                ind_results[variant_name] = rsa
                print(f"  Individual 18D vs {variant_name}: "
                      f"rho={rsa['rho']:.3f}, p={rsa['p_value']:.4f}")

            results["individual_18d"] = ind_results
        else:
            print(f"  WARNING: Entity key mismatch in individual ratings — skipping")
    else:
        print(f"  No individual_rating_matrix.npz found — skipping individual source")

    if not results:
        print("  No data sources available — nothing to save.")
        return

    # ── Save ──
    json_path = os.path.join(str(ddir), "behavioral_rsa_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    npz_path = os.path.join(str(ddir), "behavioral_rdms.npz")
    np.savez_compressed(npz_path, **rdms_to_save)
    print(f"  Saved: {npz_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Behavioral RSA for Gray Replication"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both with_self and without_self conditions"
    )
    args = parser.parse_args()

    set_model(args.model)
    print(f"Model: {config.MODEL_LABEL}")
    print(f"Behavioral RSA: Gray Replication")

    from entities.gray_entities import ENTITY_NAMES

    if args.both:
        entity_keys_no_self = [k for k in ENTITY_NAMES if k != "you_self"]
        run_condition(entity_keys_no_self, "without_self")
        run_condition(ENTITY_NAMES, "with_self")
    else:
        tag = get_condition_tag(args.include_self)
        if args.include_self:
            entity_keys = ENTITY_NAMES
        else:
            entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]
        run_condition(entity_keys, tag)


if __name__ == "__main__":
    main()
