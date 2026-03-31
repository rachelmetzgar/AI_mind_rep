#!/usr/bin/env python3
"""
Matched Concept Activation PCA Reanalysis

Reads saved character activations (from activation RSA, already computed
for both models), runs PCA on character activation RDMs at each layer,
then tests how well the resulting factor structure separates AI from
human characters — comparing against matched behavioral PCA results.

CPU-only — reads pre-computed activation data.

Output:
    results/{model}/expanded_mental_concepts/internals/pca/data/
        matched_activation_pca_results.npz

Usage:
    python expanded_mental_concepts/internals/pca/matched_activation_pca.py --model llama2_13b_base
    python expanded_mental_concepts/internals/pca/matched_activation_pca.py --model llama2_13b_chat --both

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir, N_LAYERS,
)
from entities.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_TYPES,
)
from utils.utils import compute_rdm_cosine, compute_behavioral_rdm


def activation_pca_per_layer(activations, n_components=2):
    """
    Run PCA on character activations at each layer.

    Args:
        activations: (n_characters, n_layers, hidden_dim)
        n_components: number of PCA components to retain

    Returns:
        factor_scores: (n_layers, n_characters, n_components)
        explained_var: (n_layers, n_components)
    """
    n_chars, n_layers, hidden_dim = activations.shape
    factor_scores = np.zeros((n_layers, n_chars, n_components))
    explained_var = np.zeros((n_layers, n_components))

    for layer in range(n_layers):
        X = activations[:, layer, :]

        # Center
        X_centered = X - X.mean(axis=0)

        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Explained variance
        total_var = np.sum(S ** 2)
        for c in range(min(n_components, len(S))):
            explained_var[layer, c] = S[c] ** 2 / total_var

        # Factor scores (projection onto top components)
        n_use = min(n_components, len(S))
        factor_scores[layer, :, :n_use] = U[:, :n_use] * S[:n_use]

    return factor_scores, explained_var


def categorical_separation_per_layer(factor_scores, char_keys):
    """
    Mann-Whitney U test at each layer for each PCA component.

    Returns:
        results: list of dicts per layer with U, p, separation for each factor
    """
    n_layers, n_chars, n_components = factor_scores.shape
    char_to_idx = {k: i for i, k in enumerate(char_keys)}
    ai_idx = [char_to_idx[k] for k in char_keys if k in AI_CHARACTERS]
    hu_idx = [char_to_idx[k] for k in char_keys if k in HUMAN_CHARACTERS]

    results = []
    for layer in range(n_layers):
        layer_result = {"layer": layer, "factors": []}
        for fi in range(n_components):
            ai_scores = factor_scores[layer, ai_idx, fi]
            hu_scores = factor_scores[layer, hu_idx, fi]
            try:
                u, p = mannwhitneyu(ai_scores, hu_scores,
                                    alternative="two-sided")
            except ValueError:
                u, p = float("nan"), float("nan")
            layer_result["factors"].append({
                "factor": fi + 1,
                "u": float(u),
                "p": float(p),
                "ai_mean": float(np.mean(ai_scores)),
                "human_mean": float(np.mean(hu_scores)),
                "separation": abs(float(np.mean(ai_scores)) -
                                  float(np.mean(hu_scores))),
            })
        results.append(layer_result)
    return results


def generate_for_model(model_key):
    """Run matched activation PCA for one model."""
    set_model(model_key)

    # Load character activations
    act_ddir = data_dir("human_ai_characters", "neural/names_only", "rsa_pca")
    act_path = os.path.join(str(act_ddir), "all_character_activations.npz")
    if not os.path.exists(act_path):
        print(f"Activations not found at {act_path} — skipping {model_key}")
        return

    act_data = np.load(act_path)
    activations = act_data["activations"]
    char_keys = list(act_data["character_keys"])

    n_chars, n_layers, hidden_dim = activations.shape
    print(f"Loaded activations: {activations.shape} "
          f"({n_chars} chars x {n_layers} layers x {hidden_dim} dims)")

    # Run PCA at each layer
    print("Running PCA at each layer...")
    factor_scores, explained_var = activation_pca_per_layer(activations)

    # Categorical separation
    print("Computing categorical separation...")
    cat_results = categorical_separation_per_layer(factor_scores, char_keys)

    # Find peak separation layer for each factor
    for fi in range(2):
        p_vals = [r["factors"][fi]["p"] for r in cat_results]
        valid_p = [(i, p) for i, p in enumerate(p_vals) if not np.isnan(p)]
        if valid_p:
            best_layer, best_p = min(valid_p, key=lambda x: x[1])
            sep = cat_results[best_layer]["factors"][fi]["separation"]
            print(f"  F{fi+1}: best separation at layer {best_layer} "
                  f"(p={best_p:.4f}, sep={sep:.4f})")

    # Try to load behavioral PCA for RSA comparison
    beh_ddir = data_dir("human_ai_characters", "behavior", "pca")
    behavioral_rsa = None

    # Check for matched subset results
    for subset in ["directional", "human", "ai", "all"]:
        pca_path = os.path.join(str(beh_ddir),
                                f"matched_{subset}_pca_results.npz")
        if os.path.exists(pca_path):
            pca_data = np.load(pca_path)
            beh_scores = pca_data["factor_scores_01"]
            beh_rdm = compute_behavioral_rdm(beh_scores)

            # RSA at each layer
            rdm_ddir = data_dir("human_ai_characters", "neural/names_only", "rsa_pca")
            rdm_path = os.path.join(str(rdm_ddir), "rdm_cosine_per_layer.npz")
            if os.path.exists(rdm_path):
                rdm_data = np.load(rdm_path)
                model_rdm = rdm_data["model_rdm"]
                triu_idx = np.triu_indices(n_chars, k=1)
                beh_upper = beh_rdm[triu_idx]

                rsa_per_layer = []
                for layer in range(model_rdm.shape[0]):
                    model_upper = model_rdm[layer][triu_idx]
                    if np.std(model_upper) < 1e-12:
                        rho, p = float("nan"), float("nan")
                    else:
                        rho, p = spearmanr(model_upper, beh_upper)
                    rsa_per_layer.append({
                        "layer": layer,
                        "rho": float(rho),
                        "p_value": float(p),
                    })

                if behavioral_rsa is None:
                    behavioral_rsa = {}
                behavioral_rsa[f"matched_{subset}"] = rsa_per_layer

                valid = [r for r in rsa_per_layer if not np.isnan(r["rho"])]
                if valid:
                    peak = max(valid, key=lambda r: r["rho"])
                    print(f"  RSA vs matched_{subset}: peak layer "
                          f"{peak['layer']}, rho={peak['rho']:+.4f}")

    # Save
    out_ddir = data_dir("human_ai_characters", "neural/names_only", "rsa_pca")
    save_kwargs = {
        "factor_scores": factor_scores,
        "explained_var": explained_var,
        "character_keys": np.array(char_keys),
    }
    np.savez_compressed(
        os.path.join(str(out_ddir), "matched_activation_pca_results.npz"),
        **save_kwargs
    )

    # Save categorical separation results
    with open(os.path.join(str(out_ddir),
              "matched_activation_pca_separation.json"), "w") as f:
        json.dump({
            "categorical_separation": cat_results,
            "behavioral_rsa": behavioral_rsa,
        }, f)

    print(f"\nSaved to {out_ddir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Matched concept activation PCA reanalysis"
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
        print(f"  Matched Activation PCA: {model_key}")
        print(f"{'='*60}")
        generate_for_model(model_key)


if __name__ == "__main__":
    main()
