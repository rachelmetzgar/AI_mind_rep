#!/usr/bin/env python3
"""
Concept Geometry, Phase B+: Internal PCA

PCA on character activations to understand the structure of the
representation space. Loads saved activations from Phase B
(rsa/activation_rsa.py) — no GPU needed.

For each layer:
    1. PCA on (28 characters x 5120 dims) → principal components
    2. Check whether PC1/PC2 separates humans from AIs
    3. Compute the supervised human/AI axis (LDA direction) and
       measure how much variance it captures vs the unsupervised PCs
    4. Silhouette score for human/AI clustering quality

Produces:
    - Per-layer variance explained by PCs and by the human/AI axis
    - Character positions in PC1/PC2 space at peak layer
    - Layer profile showing where human/AI structure is strongest

Output:
    results/{model}/concept_geometry/pca/activation/data/
        internal_pca_results.npz
        internal_pca_analysis.json

    results/{model}/concept_geometry/pca/activation/
        internal_pca_summary.md

Usage:
    python concept_geometry/pca/activation_pca.py --model llama2_13b_chat
    python concept_geometry/pca/activation_pca.py --model llama2_13b_base

Env: llama2_env (CPU-only, can run on login node)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import mannwhitneyu

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir,
)
from concept_geometry.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)


# ========================== PCA PER LAYER ========================== #

def pca_at_layer(activations, n_components=None):
    """
    PCA on (n_samples, n_features) matrix.

    Returns dict with eigenvalues, eigenvectors, projections, variance explained.
    """
    n_samples, n_features = activations.shape
    if n_components is None:
        n_components = min(n_samples, n_features)

    # Center
    mean = activations.mean(axis=0)
    centered = activations - mean

    # SVD (more stable than eigendecomposition for wide matrices)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Eigenvalues from singular values
    eigenvalues = (S ** 2) / (n_samples - 1)

    # Projections (scores)
    projections = centered @ Vt.T  # (n_samples, n_components)

    total_var = np.sum(eigenvalues)
    explained_ratio = eigenvalues / total_var if total_var > 0 else eigenvalues

    return {
        "eigenvalues": eigenvalues[:n_components],
        "explained_ratio": explained_ratio[:n_components],
        "components": Vt[:n_components],  # principal directions
        "projections": projections[:, :n_components],
        "mean": mean,
    }


def compute_lda_axis(activations, labels):
    """
    Compute Fisher's LDA direction for two groups.
    Returns the unit direction vector and the fraction of total variance
    captured along this direction.
    """
    classes = np.unique(labels)
    assert len(classes) == 2

    mask_0 = labels == classes[0]
    mask_1 = labels == classes[1]

    mean_0 = activations[mask_0].mean(axis=0)
    mean_1 = activations[mask_1].mean(axis=0)

    # LDA direction: difference of class means (normalized)
    w = mean_1 - mean_0
    w_norm = np.linalg.norm(w)
    if w_norm < 1e-10:
        return w, 0.0
    w = w / w_norm

    # Variance along this direction
    global_mean = activations.mean(axis=0)
    centered = activations - global_mean
    projections = centered @ w
    var_along_axis = np.var(projections)
    total_var = np.sum(np.var(centered, axis=0))

    return w, float(var_along_axis / total_var) if total_var > 0 else 0.0


def silhouette_score_binary(projections_2d, labels):
    """
    Simplified silhouette score for two groups in 2D space.
    """
    classes = np.unique(labels)
    n = len(labels)
    scores = np.zeros(n)

    for i in range(n):
        same_mask = labels == labels[i]
        other_mask = ~same_mask
        same_mask[i] = False  # exclude self

        if np.sum(same_mask) == 0 or np.sum(other_mask) == 0:
            scores[i] = 0
            continue

        dists = np.linalg.norm(projections_2d - projections_2d[i], axis=1)
        a_i = np.mean(dists[same_mask])
        b_i = np.mean(dists[other_mask])

        scores[i] = (b_i - a_i) / max(a_i, b_i)

    return float(np.mean(scores))


# ========================== MAIN ANALYSIS ========================== #

def run_internal_pca(model_key):
    """Run internal PCA analysis on saved activations."""
    set_model(model_key)

    # Input: activations from RSA phase
    rsa_ddir = data_dir("concept_geometry/rsa", "activation")
    # Output: PCA results
    ddir = data_dir("concept_geometry/pca", "activation")
    rdir = results_phase_dir("concept_geometry/pca", "activation")

    # Load activations
    acts_path = os.path.join(str(rsa_ddir), "all_character_activations.npz")
    if not os.path.exists(acts_path):
        print(f"Error: activations not found at {acts_path}")
        print("Run rsa/activation_rsa.py first to extract activations.")
        sys.exit(1)

    print(f"Loading activations from {acts_path}...")
    data = np.load(acts_path)
    activations = data["activations"]  # (n_chars, n_layers, hidden_dim)
    char_keys = list(data["character_keys"])

    n_chars, n_layers, hidden_dim = activations.shape
    print(f"Shape: {n_chars} characters x {n_layers} layers x {hidden_dim} dims")

    # Build label array (0 = AI, 1 = human)
    labels = np.array([0 if CHARACTER_TYPES[k] == "ai" else 1 for k in char_keys])

    # ── Analyze each layer ──
    layer_results = []
    all_projections = np.zeros((n_layers, n_chars, min(n_chars, 10)))

    for layer in range(n_layers):
        acts = activations[:, layer, :]  # (n_chars, hidden_dim)

        # PCA
        pca = pca_at_layer(acts, n_components=min(n_chars, 10))

        # LDA axis
        lda_direction, lda_var_frac = compute_lda_axis(acts, labels)

        # Mann-Whitney U on PC1 and PC2
        ai_mask = labels == 0
        human_mask = labels == 1

        pc_separations = []
        for pc_idx in range(min(2, pca["projections"].shape[1])):
            pc_scores = pca["projections"][:, pc_idx]
            try:
                u_stat, p_val = mannwhitneyu(
                    pc_scores[ai_mask], pc_scores[human_mask],
                    alternative="two-sided"
                )
            except ValueError:
                u_stat, p_val = float("nan"), float("nan")

            pc_separations.append({
                "pc": pc_idx + 1,
                "ai_mean": float(np.mean(pc_scores[ai_mask])),
                "human_mean": float(np.mean(pc_scores[human_mask])),
                "mann_whitney_u": float(u_stat),
                "p_value": float(p_val),
            })

        # Silhouette in PC1/PC2 space
        if pca["projections"].shape[1] >= 2:
            sil = silhouette_score_binary(pca["projections"][:, :2], labels)
        else:
            sil = float("nan")

        layer_results.append({
            "layer": layer,
            "eigenvalues": pca["eigenvalues"][:5].tolist(),
            "explained_ratio": pca["explained_ratio"][:5].tolist(),
            "lda_var_fraction": lda_var_frac,
            "silhouette_pc12": sil,
            "pc_separations": pc_separations,
        })

        n_store = min(n_chars, 10)
        all_projections[layer, :, :n_store] = pca["projections"][:, :n_store]

        if layer % 10 == 0 or layer == n_layers - 1:
            print(f"  Layer {layer:2d}: PC1={pca['explained_ratio'][0]*100:.1f}%, "
                  f"LDA axis={lda_var_frac*100:.1f}%, "
                  f"silhouette={sil:.3f}")

    # ── Find peak layers ──
    sil_scores = [r["silhouette_pc12"] for r in layer_results]
    lda_fracs = [r["lda_var_fraction"] for r in layer_results]

    valid_sil = [(i, s) for i, s in enumerate(sil_scores) if not np.isnan(s)]
    if valid_sil:
        peak_sil_layer, peak_sil = max(valid_sil, key=lambda x: x[1])
    else:
        peak_sil_layer, peak_sil = -1, float("nan")

    peak_lda_layer = int(np.argmax(lda_fracs))
    peak_lda_frac = lda_fracs[peak_lda_layer]

    print(f"\nPeak silhouette: Layer {peak_sil_layer} ({peak_sil:.3f})")
    print(f"Peak LDA var fraction: Layer {peak_lda_layer} ({peak_lda_frac*100:.1f}%)")

    # ── Save ──
    np.savez_compressed(
        os.path.join(str(ddir), "internal_pca_results.npz"),
        projections=all_projections,  # (n_layers, n_chars, n_components)
        character_keys=np.array(char_keys),
        labels=labels,
    )

    analysis = {
        "model": config.MODEL_LABEL,
        "n_characters": n_chars,
        "n_layers": n_layers,
        "peak_silhouette_layer": peak_sil_layer,
        "peak_silhouette": peak_sil,
        "peak_lda_layer": peak_lda_layer,
        "peak_lda_var_fraction": peak_lda_frac,
        "layer_results": layer_results,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(str(ddir), "internal_pca_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # ── Write summary ──
    write_summary(rdir, analysis, char_keys, labels, all_projections,
                  peak_sil_layer, peak_lda_layer)

    print(f"\nSaved to {ddir}/ and {rdir}/")


def write_summary(out_dir, analysis, char_keys, labels, all_projections,
                  peak_sil_layer, peak_lda_layer):
    """Write markdown summary of internal PCA analysis."""
    layer_results = analysis["layer_results"]
    n_layers = analysis["n_layers"]

    path = os.path.join(str(out_dir), "internal_pca_summary.md")
    with open(path, "w") as f:
        f.write("# Concept Geometry: Internal PCA Analysis\n\n")
        f.write(f"**Model:** {analysis['model']}\n")
        f.write(f"**Run:** {analysis['timestamp']}\n\n")

        f.write("## What is being tested\n\n")
        f.write(
            "Does a human/AI axis emerge as a natural principal component "
            "of the representation space, or is the categorical structure "
            "only visible via supervised methods (RSA/LDA)?\n\n"
        )

        f.write("## Metrics\n\n")
        f.write(
            "- **PC explained variance**: How much variance each unsupervised "
            "PC captures\n"
            "- **LDA axis variance**: Fraction of total variance along the "
            "supervised human/AI direction (Fisher's discriminant)\n"
            "- **Silhouette score**: Clustering quality of human vs AI groups "
            "in PC1/PC2 space (-1 to 1, higher = better separation)\n"
            "- **Mann-Whitney U**: Whether PC1/PC2 scores differ between "
            "AI and human groups\n\n"
        )

        # Peak layers
        f.write("## Peak Layers\n\n")
        f.write(f"- **Best silhouette (PC1/PC2):** Layer {peak_sil_layer} "
                f"({analysis['peak_silhouette']:.3f})\n")
        f.write(f"- **Most LDA variance:** Layer {peak_lda_layer} "
                f"({analysis['peak_lda_var_fraction']*100:.1f}%)\n\n")

        # Layer-by-layer table
        f.write("## Layer Profile\n\n")
        f.write("| Layer | PC1 var% | PC2 var% | LDA var% | Silhouette | "
                "PC1 U p-val | PC2 U p-val |\n")
        f.write("|------:|---------:|---------:|---------:|-----------:|"
                "-----------:|-----------:|\n")
        for r in layer_results:
            pc1_var = r["explained_ratio"][0] * 100 if len(r["explained_ratio"]) > 0 else 0
            pc2_var = r["explained_ratio"][1] * 100 if len(r["explained_ratio"]) > 1 else 0
            lda_var = r["lda_var_fraction"] * 100
            sil = r["silhouette_pc12"]

            pc1_p = r["pc_separations"][0]["p_value"] if len(r["pc_separations"]) > 0 else float("nan")
            pc2_p = r["pc_separations"][1]["p_value"] if len(r["pc_separations"]) > 1 else float("nan")

            sil_str = f"{sil:.3f}" if not np.isnan(sil) else "nan"
            pc1_p_str = f"{pc1_p:.4f}" if not np.isnan(pc1_p) else "nan"
            pc2_p_str = f"{pc2_p:.4f}" if not np.isnan(pc2_p) else "nan"

            f.write(f"| {r['layer']:5d} | {pc1_var:8.1f} | {pc2_var:8.1f} | "
                    f"{lda_var:8.1f} | {sil_str:>10s} | "
                    f"{pc1_p_str:>10s} | {pc2_p_str:>10s} |\n")
        f.write("\n")

        # Character positions at peak silhouette layer
        if peak_sil_layer >= 0:
            f.write(f"## Character Positions at Peak Layer {peak_sil_layer}\n\n")
            f.write("| Character | Type | PC1 | PC2 |\n")
            f.write("|-----------|------|----:|----:|\n")
            projs = all_projections[peak_sil_layer]
            for i, char_key in enumerate(char_keys):
                char_type = CHARACTER_TYPES[char_key]
                f.write(f"| {CHARACTER_NAMES[char_key]} | {char_type} | "
                        f"{projs[i, 0]:+.3f} | {projs[i, 1]:+.3f} |\n")
            f.write("\n")

    print(f"  Summary: {path}")


# ========================== ENTRY POINT ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Concept Geometry: Internal PCA Analysis"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--both", action="store_true",
        help="Run both chat and base models"
    )
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Internal PCA: {model_key}")
        print(f"{'='*60}\n")
        run_internal_pca(model_key)

    print("\nDone.")


if __name__ == "__main__":
    main()
