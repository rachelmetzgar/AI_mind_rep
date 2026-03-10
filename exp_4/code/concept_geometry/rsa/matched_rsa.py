#!/usr/bin/env python3
"""
Matched Concept RSA Reanalysis

Uses saved activation RDMs (rdm_cosine_per_layer.npz) from activation RSA.
Computes RSA against behavioral RDMs derived from matched concept subsets
(instead of just categorical).

For each matched subset:
    - Compute behavioral RDM from matched PCA factor scores
    - RSA between activation RDM and behavioral RDM at each layer

CPU-only — reads pre-computed data.

Output:
    results/{model}/concept_geometry/rsa/activation/data/
        matched_rsa_results.json

Usage:
    python concept_geometry/rsa/matched_rsa.py --model llama2_13b_base
    python concept_geometry/rsa/matched_rsa.py --model llama2_13b_chat --both

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir,
)
from utils.utils import compute_behavioral_rdm


SUBSETS = ["human", "ai", "directional", "all"]


def generate_for_model(model_key):
    """Run matched RSA for one model."""
    set_model(model_key)

    # Load activation RDMs
    act_ddir = data_dir("concept_geometry/rsa", "activation")
    rdm_path = os.path.join(str(act_ddir), "rdm_cosine_per_layer.npz")
    if not os.path.exists(rdm_path):
        print(f"RDM data not found at {rdm_path} — skipping {model_key}")
        return

    rdm_data = np.load(rdm_path)
    model_rdm = rdm_data["model_rdm"]
    char_keys = list(rdm_data["character_keys"])
    n_chars = len(char_keys)
    n_layers = model_rdm.shape[0]

    print(f"Loaded model RDM: {model_rdm.shape} "
          f"({n_layers} layers, {n_chars} characters)")

    triu_idx = np.triu_indices(n_chars, k=1)

    # Load behavioral PCA results for each subset
    beh_ddir = data_dir("concept_geometry/pca", "behavioral")
    all_rsa = {}

    for subset in SUBSETS:
        pca_path = os.path.join(str(beh_ddir),
                                f"matched_{subset}_pca_results.npz")
        if not os.path.exists(pca_path):
            print(f"  matched_{subset}: PCA results not found, skipping")
            continue

        pca_data = np.load(pca_path)
        factor_scores = pca_data["factor_scores_01"]

        # Compute behavioral RDM
        beh_rdm = compute_behavioral_rdm(factor_scores)
        beh_upper = beh_rdm[triu_idx]

        # RSA at each layer
        rsa_results = []
        for layer in range(n_layers):
            model_upper = model_rdm[layer][triu_idx]
            if np.std(model_upper) < 1e-12:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(model_upper, beh_upper)
            rsa_results.append({
                "layer": layer,
                "rho": float(rho),
                "p_value": float(p),
                "n_pairs": len(beh_upper),
            })

        all_rsa[f"matched_{subset}"] = rsa_results

        # Print peak
        valid = [r for r in rsa_results if not np.isnan(r["rho"])]
        if valid:
            peak = max(valid, key=lambda r: r["rho"])
            print(f"  matched_{subset}: peak layer {peak['layer']}, "
                  f"rho={peak['rho']:+.4f}, p={peak['p_value']:.4f}")

    # Also do RSA with original behavioral RDM if available
    orig_pca_path = os.path.join(str(beh_ddir), "pairwise_pca_results.npz")
    if os.path.exists(orig_pca_path):
        pca_data = np.load(orig_pca_path)
        factor_scores = pca_data["factor_scores_01"]
        beh_rdm = compute_behavioral_rdm(factor_scores)
        beh_upper = beh_rdm[triu_idx]

        rsa_results = []
        for layer in range(n_layers):
            model_upper = model_rdm[layer][triu_idx]
            if np.std(model_upper) < 1e-12:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = spearmanr(model_upper, beh_upper)
            rsa_results.append({
                "layer": layer,
                "rho": float(rho),
                "p_value": float(p),
                "n_pairs": len(beh_upper),
            })
        all_rsa["original_behavioral"] = rsa_results

        valid = [r for r in rsa_results if not np.isnan(r["rho"])]
        if valid:
            peak = max(valid, key=lambda r: r["rho"])
            print(f"  original_behavioral: peak layer {peak['layer']}, "
                  f"rho={peak['rho']:+.4f}, p={peak['p_value']:.4f}")

    # Save
    out_path = os.path.join(str(act_ddir), "matched_rsa_results.json")
    with open(out_path, "w") as f:
        json.dump(all_rsa, f, indent=2)

    print(f"\nSaved matched RSA results to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Matched concept RSA reanalysis"
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
        print(f"  Matched RSA: {model_key}")
        print(f"{'='*60}")
        generate_for_model(model_key)


if __name__ == "__main__":
    main()
