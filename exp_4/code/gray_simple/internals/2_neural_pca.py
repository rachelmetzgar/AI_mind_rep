#!/usr/bin/env python3
"""
Experiment 4: Neural PCA on Gray Entity Activations

PCA on entity activation vectors (n_entities x hidden_dim) at each layer,
then correlates principal components with human Experience/Agency scores.
Complements RSA (which tests overall geometric similarity) by asking
whether specific dimensions of neural variance track specific human factors.

Three analyses per layer:
    1. PCA: Top 5 PCs, correlate each with human Experience and Agency
    2. Procrustes: Align top 2 PCs to human 2D (Experience, Agency) space
    3. MDS: Classical MDS on cosine-distance RDM, same correlations + Procrustes

Output:
    data_dir("gray_simple", "internals", tag)/
        neural_pca_results.npz     # per-layer PCA + MDS results
        neural_pca_analysis.json   # summary statistics

Usage:
    python internals/2_neural_pca.py --model llama2_13b_chat
    python internals/2_neural_pca.py --model llama2_13b_base --include_self
    python internals/2_neural_pca.py --model llama2_13b_chat --both

SLURM:
    sbatch internals/slurm/2_neural_pca_chat.sh
    sbatch internals/slurm/2_neural_pca_base.sh

Env: llama2_env (needs numpy, scipy — no GPU required)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    config, set_model, add_model_argument,
    data_dir, get_condition_tag, N_LAYERS,
)
from entities.gray_entities import GRAY_ET_AL_SCORES, ENTITY_NAMES


# ========================== PCA ========================== #

def pca_at_layer(acts, n_components=5):
    """
    PCA on (n_entities, hidden_dim) at a single layer.

    Returns:
        scores: (n_entities, n_components) — projections
        explained_var_ratio: (n_components,) — fraction of variance
        components: (n_components, hidden_dim) — PC directions
    """
    # Center
    mean = acts.mean(axis=0)
    centered = acts - mean

    # SVD (more stable than eigendecomposition for wide matrices)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    total_var = np.sum(S ** 2) / (acts.shape[0] - 1)

    k = min(n_components, len(S))
    scores = U[:, :k] * S[:k]
    var_explained = (S[:k] ** 2) / (acts.shape[0] - 1)
    explained_ratio = var_explained / max(total_var, 1e-12)

    return scores, explained_ratio, Vt[:k]


# ========================== PROCRUSTES ========================== #

def procrustes_align(source, target):
    """
    Orthogonal Procrustes: find rotation R, scale s, translation t
    that minimize ||s * source @ R + t - target||^2.

    Both inputs: (n, 2).
    Returns: aligned source, disparity (sum of squared residuals / sum of
    squared target deviations from centroid).
    """
    # Center both
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    src = source - mu_s
    tgt = target - mu_t

    # Scale to unit Frobenius norm
    norm_s = np.sqrt(np.sum(src ** 2))
    norm_t = np.sqrt(np.sum(tgt ** 2))
    if norm_s < 1e-12 or norm_t < 1e-12:
        return source, 1.0
    src = src / norm_s
    tgt = tgt / norm_t

    # Optimal rotation via SVD
    M = tgt.T @ src
    U, _, Vt = np.linalg.svd(M)
    R = (U @ Vt)

    # Correct for reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Apply rotation + rescale to target scale
    aligned = (src @ R.T) * norm_t + mu_t

    # Disparity: sum of squared residuals / sum of squared target deviations
    residuals = np.sum((aligned - target) ** 2)
    target_var = np.sum(tgt ** 2) * norm_t ** 2
    disparity = residuals / max(target_var, 1e-12)

    return aligned, disparity


# ========================== MDS ========================== #

def classical_mds(rdm, n_dims=2):
    """
    Classical (metric) MDS on a distance matrix.
    Returns (n, n_dims) coordinates.
    """
    n = rdm.shape[0]
    D_sq = rdm ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Take top n_dims positive eigenvalues
    k = min(n_dims, np.sum(eigenvalues > 0))
    coords = eigenvectors[:, :k] * np.sqrt(np.maximum(eigenvalues[:k], 0))

    # Pad if fewer positive eigenvalues than requested
    if k < n_dims:
        coords = np.hstack([coords, np.zeros((n, n_dims - k))])

    return coords


# ========================== MAIN ANALYSIS ========================== #

def analyze_condition(entity_keys, tag):
    """Run neural PCA analysis for one condition."""
    ddir = data_dir("gray_simple", "internals", tag)

    # Load activations
    act_path = os.path.join(str(ddir), "all_entity_activations.npz")
    if not os.path.exists(act_path):
        print(f"  No activations found at {act_path}, skipping.")
        return

    d = np.load(act_path, allow_pickle=True)
    acts = d["activations"]  # (n_entities, n_layers, hidden_dim)
    stored_keys = list(d["entity_keys"])

    # Verify entity ordering matches
    assert stored_keys == entity_keys, (
        f"Entity key mismatch: stored={stored_keys}, expected={entity_keys}"
    )

    n_entities, n_layers, hidden_dim = acts.shape
    print(f"  Activations: {n_entities} entities x {n_layers} layers x {hidden_dim} dims")

    # Human scores
    human_exp = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    human_ag = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])
    human_2d = np.column_stack([human_exp, human_ag])

    # Storage
    n_pcs = 5
    pc_exp_corr = np.full((n_layers, n_pcs), np.nan)
    pc_ag_corr = np.full((n_layers, n_pcs), np.nan)
    pc_exp_pval = np.full((n_layers, n_pcs), np.nan)
    pc_ag_pval = np.full((n_layers, n_pcs), np.nan)
    explained_var = np.full((n_layers, n_pcs), np.nan)
    procrustes_disparity = np.full(n_layers, np.nan)
    procrustes_coords = np.full((n_layers, n_entities, 2), np.nan)

    # MDS results
    mds_exp_corr = np.full((n_layers, 2), np.nan)
    mds_ag_corr = np.full((n_layers, 2), np.nan)
    mds_exp_pval = np.full((n_layers, 2), np.nan)
    mds_ag_pval = np.full((n_layers, 2), np.nan)
    mds_procrustes_disparity = np.full(n_layers, np.nan)
    mds_procrustes_coords = np.full((n_layers, n_entities, 2), np.nan)

    for layer in range(n_layers):
        layer_acts = acts[:, layer, :]  # (n_entities, hidden_dim)

        # Check for constant activations
        if np.std(layer_acts) < 1e-12:
            continue

        # --- PCA ---
        scores, var_ratio, _ = pca_at_layer(layer_acts, n_pcs)
        k = scores.shape[1]
        explained_var[layer, :k] = var_ratio

        for pc in range(k):
            if np.std(scores[:, pc]) < 1e-12:
                continue
            rho_e, p_e = spearmanr(scores[:, pc], human_exp)
            rho_a, p_a = spearmanr(scores[:, pc], human_ag)
            pc_exp_corr[layer, pc] = rho_e
            pc_ag_corr[layer, pc] = rho_a
            pc_exp_pval[layer, pc] = p_e
            pc_ag_pval[layer, pc] = p_a

        # --- Procrustes on top 2 PCs ---
        if k >= 2:
            aligned, disp = procrustes_align(scores[:, :2], human_2d)
            procrustes_disparity[layer] = disp
            procrustes_coords[layer] = aligned

        # --- MDS on cosine distance RDM ---
        norms = np.linalg.norm(layer_acts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = layer_acts / norms
        cos_sim = normed @ normed.T
        cos_dist = 1.0 - cos_sim
        np.fill_diagonal(cos_dist, 0)
        cos_dist = np.maximum(cos_dist, 0)  # numerical safety

        mds_coords = classical_mds(cos_dist, n_dims=2)

        for dim in range(2):
            if np.std(mds_coords[:, dim]) < 1e-12:
                continue
            rho_e, p_e = spearmanr(mds_coords[:, dim], human_exp)
            rho_a, p_a = spearmanr(mds_coords[:, dim], human_ag)
            mds_exp_corr[layer, dim] = rho_e
            mds_ag_corr[layer, dim] = rho_a
            mds_exp_pval[layer, dim] = p_e
            mds_ag_pval[layer, dim] = p_a

        mds_aligned, mds_disp = procrustes_align(mds_coords, human_2d)
        mds_procrustes_disparity[layer] = mds_disp
        mds_procrustes_coords[layer] = mds_aligned

    # --- Summary statistics ---
    # Best PC1 correlation with Experience across layers
    valid_mask = ~np.isnan(pc_exp_corr[:, 0])
    summary = {}

    if valid_mask.any():
        # Find peak layer for each PC-human pairing (by absolute correlation)
        for pc in range(min(n_pcs, 2)):
            for dim_name, corr_arr, pval_arr in [
                ("experience", pc_exp_corr[:, pc], pc_exp_pval[:, pc]),
                ("agency", pc_ag_corr[:, pc], pc_ag_pval[:, pc]),
            ]:
                valid = ~np.isnan(corr_arr)
                if valid.any():
                    peak_layer = int(np.argmax(np.abs(corr_arr[valid])))
                    # Map back to actual layer index
                    actual_layers = np.where(valid)[0]
                    peak_layer = int(actual_layers[peak_layer])
                    summary[f"pc{pc+1}_{dim_name}_peak_layer"] = peak_layer
                    summary[f"pc{pc+1}_{dim_name}_peak_rho"] = float(corr_arr[peak_layer])
                    summary[f"pc{pc+1}_{dim_name}_peak_p"] = float(pval_arr[peak_layer])

        # Peak Procrustes alignment
        valid_proc = ~np.isnan(procrustes_disparity)
        if valid_proc.any():
            best_layer = int(np.argmin(procrustes_disparity[valid_proc]))
            actual_layers = np.where(valid_proc)[0]
            best_layer = int(actual_layers[best_layer])
            summary["procrustes_best_layer"] = best_layer
            summary["procrustes_best_disparity"] = float(procrustes_disparity[best_layer])

        # Peak MDS Procrustes
        valid_mds = ~np.isnan(mds_procrustes_disparity)
        if valid_mds.any():
            best_layer = int(np.argmin(mds_procrustes_disparity[valid_mds]))
            actual_layers = np.where(valid_mds)[0]
            best_layer = int(actual_layers[best_layer])
            summary["mds_procrustes_best_layer"] = best_layer
            summary["mds_procrustes_best_disparity"] = float(mds_procrustes_disparity[best_layer])

    summary["n_entities"] = n_entities
    summary["n_layers"] = n_layers
    summary["hidden_dim"] = hidden_dim
    summary["model"] = config.MODEL_KEY
    summary["model_label"] = config.MODEL_LABEL
    summary["condition"] = tag
    summary["entity_keys"] = entity_keys

    # --- Save ---
    np.savez_compressed(
        os.path.join(str(ddir), "neural_pca_results.npz"),
        pc_exp_corr=pc_exp_corr,
        pc_ag_corr=pc_ag_corr,
        pc_exp_pval=pc_exp_pval,
        pc_ag_pval=pc_ag_pval,
        explained_var=explained_var,
        procrustes_disparity=procrustes_disparity,
        procrustes_coords=procrustes_coords,
        mds_exp_corr=mds_exp_corr,
        mds_ag_corr=mds_ag_corr,
        mds_exp_pval=mds_exp_pval,
        mds_ag_pval=mds_ag_pval,
        mds_procrustes_disparity=mds_procrustes_disparity,
        mds_procrustes_coords=mds_procrustes_coords,
        entity_keys=np.array(entity_keys),
    )

    analysis_path = os.path.join(str(ddir), "neural_pca_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved results to {ddir}/")

    # Print key findings
    for key in ["pc1_experience_peak_layer", "pc1_agency_peak_layer",
                "pc2_experience_peak_layer", "pc2_agency_peak_layer",
                "procrustes_best_layer"]:
        if key in summary:
            val = summary[key]
            if "layer" in key:
                rho_key = key.replace("_layer", "_rho")
                rho = summary.get(rho_key, "")
                if isinstance(rho, float):
                    print(f"    {key}: layer {val}, rho={rho:+.4f}")
                else:
                    disp_key = key.replace("_layer", "_disparity")
                    disp = summary.get(disp_key, "")
                    print(f"    {key}: layer {val}, disparity={disp:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Neural PCA on Gray entity activations"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both with_self and without_self conditions"
    )
    args = parser.parse_args()

    set_model(args.model)
    print(f"Model: {config.MODEL_LABEL}")

    if args.both:
        conditions = [False, True]
    else:
        conditions = [args.include_self]

    for include_self in conditions:
        tag = get_condition_tag(include_self)
        entity_keys = list(ENTITY_NAMES) if include_self else [
            k for k in ENTITY_NAMES if k != "you_self"
        ]
        print(f"\n{'='*60}")
        print(f"  Condition: {tag} ({len(entity_keys)} entities)")
        print(f"{'='*60}")
        analyze_condition(entity_keys, tag)

    print("\nDone.")


if __name__ == "__main__":
    main()
