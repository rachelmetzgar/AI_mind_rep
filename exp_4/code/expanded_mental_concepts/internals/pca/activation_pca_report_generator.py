#!/usr/bin/env python3
"""
Activation PCA Report Generator

Generates an HTML report with full methodology and charts for interpreting
the internal PCA results (PCA on character activations).

Reads from:
    results/{model}/expanded_mental_concepts/internals/pca/data/
        internal_pca_results.npz
        internal_pca_analysis.json

Writes to:
    results/{model}/expanded_mental_concepts/internals/pca/{dataset}/
        activation_pca_report.html

Usage:
    python expanded_mental_concepts/internals/pca/activation_pca_report_generator.py --model llama2_13b_chat
    python expanded_mental_concepts/internals/pca/activation_pca_report_generator.py --model llama2_13b_base

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir,
)
from entities.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
    expanded_concepts_stimuli_html,
)

SECTIONS = [
    {"id": "research-question", "label": "1. Research Question"},
    {"id": "analysis-approach", "label": "2. Analysis Approach"},
    {"id": "stimuli", "label": "3. Stimuli"},
    {"id": "layer-profile", "label": "4. Layer Profile"},
    {"id": "scatter", "label": "5. Character Positions at Peak Layer"},
    {"id": "significance", "label": "6. Statistical Significance"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate activation PCA report for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_model(model_key)

    ddir = data_dir("expanded_mental_concepts", "internals", "pca")
    rdir = results_dir("expanded_mental_concepts", "internals", "pca")

    # ── Load data ──
    analysis_path = os.path.join(str(ddir), "internal_pca_analysis.json")
    pca_path = os.path.join(str(ddir), "internal_pca_results.npz")

    if not os.path.exists(analysis_path):
        print(f"PCA analysis not found at {analysis_path} — skipping {model_key}")
        return None

    with open(analysis_path) as f:
        analysis = json.load(f)
    pca_data = np.load(pca_path)

    projections = pca_data["projections"]  # (n_layers, n_chars, n_components)
    char_keys = list(pca_data["character_keys"])
    labels = pca_data["labels"]  # 0 = AI, 1 = human

    layer_results = analysis["layer_results"]
    n_chars = analysis["n_characters"]
    n_layers = analysis["n_layers"]
    peak_sil_layer = analysis["peak_silhouette_layer"]
    peak_sil = analysis["peak_silhouette"]
    peak_lda_layer = analysis["peak_lda_layer"]
    peak_lda_frac = analysis["peak_lda_var_fraction"]

    n_ai = int(np.sum(labels == 0))
    n_hu = int(np.sum(labels == 1))

    figures = {}

    # ── 1. Layer profile: PC1 var, LDA var, silhouette ──
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    layers_x = [r["layer"] for r in layer_results]

    # PC1 and PC2 variance
    pc1_var = [r["explained_ratio"][0] * 100 for r in layer_results]
    pc2_var = [r["explained_ratio"][1] * 100 if len(r["explained_ratio"]) > 1 else 0
               for r in layer_results]
    axes[0].plot(layers_x, pc1_var, color="#1E88E5", linewidth=2, label="PC1")
    axes[0].plot(layers_x, pc2_var, color="#E53935", linewidth=2, label="PC2")
    axes[0].set_ylabel("Variance Explained (%)")
    axes[0].set_title(f"Unsupervised PCA — {config.MODEL_LABEL}")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # LDA variance fraction
    lda_var = [r["lda_var_fraction"] * 100 for r in layer_results]
    axes[1].plot(layers_x, lda_var, color="#7B1FA2", linewidth=2)
    axes[1].axvline(x=peak_lda_layer, color="#7B1FA2", linestyle=":", alpha=0.5)
    axes[1].annotate(f"peak L{peak_lda_layer}\n{peak_lda_frac*100:.1f}%",
                     (peak_lda_layer, peak_lda_frac * 100), fontsize=8,
                     xytext=(8, -15), textcoords="offset points", color="#7B1FA2")
    axes[1].set_ylabel("LDA Axis Variance (%)")
    axes[1].set_title("Supervised Human/AI Axis (Fisher's LDA)")
    axes[1].grid(True, alpha=0.2)

    # Silhouette
    sil_scores = [r["silhouette_pc12"] for r in layer_results]
    axes[2].plot(layers_x, sil_scores, color="#00897B", linewidth=2)
    if peak_sil_layer >= 0:
        axes[2].axvline(x=peak_sil_layer, color="#00897B", linestyle=":", alpha=0.5)
        axes[2].annotate(f"peak L{peak_sil_layer}\n{peak_sil:.3f}",
                         (peak_sil_layer, peak_sil), fontsize=8,
                         xytext=(8, -15), textcoords="offset points", color="#00897B")
    axes[2].axhline(y=0, color="gray", linewidth=0.5)
    axes[2].set_ylabel("Silhouette Score")
    axes[2].set_xlabel("Layer")
    axes[2].set_title("Clustering Quality (Human/AI in PC1/PC2 space)")
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    figures["layer_profile"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 2. Character scatter at peak silhouette layer ──
    if peak_sil_layer >= 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        projs = projections[peak_sil_layer]  # (n_chars, n_components)

        for i, char_key in enumerate(char_keys):
            is_ai = labels[i] == 0
            color = "#E53935" if is_ai else "#1E88E5"
            marker = "s" if is_ai else "o"
            ax.scatter(projs[i, 0], projs[i, 1],
                       c=color, marker=marker, s=80, edgecolors="white",
                       linewidth=0.5, zorder=3)
            ax.annotate(CHARACTER_NAMES[char_key],
                        (projs[i, 0], projs[i, 1]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points",
                        color=color, alpha=0.85)

        # Centroids
        ai_mask = labels == 0
        hu_mask = labels == 1
        ai_center = projs[ai_mask].mean(axis=0)
        hu_center = projs[hu_mask].mean(axis=0)
        ax.scatter(ai_center[0], ai_center[1], c="#E53935", marker="X", s=200,
                   edgecolors="black", linewidth=1.5, zorder=5)
        ax.scatter(hu_center[0], hu_center[1], c="#1E88E5", marker="X", s=200,
                   edgecolors="black", linewidth=1.5, zorder=5)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#E53935",
                   markersize=10, label="AI characters"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E88E5",
                   markersize=10, label="Human characters"),
            Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=12, label="Group centroid"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=9)

        lr = layer_results[peak_sil_layer]
        pc1_pct = lr["explained_ratio"][0] * 100
        pc2_pct = lr["explained_ratio"][1] * 100 if len(lr["explained_ratio"]) > 1 else 0
        ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% variance)")
        ax.set_title(f"Character Activations in PC1/PC2 Space — "
                     f"Layer {peak_sil_layer} — {config.MODEL_LABEL}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures["scatter_peak"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 3. Mann-Whitney p-values across layers ──
    fig, ax = plt.subplots(figsize=(10, 4))
    pc1_pvals = []
    pc2_pvals = []
    for r in layer_results:
        seps = r["pc_separations"]
        pc1_pvals.append(seps[0]["p_value"] if len(seps) > 0 else 1.0)
        pc2_pvals.append(seps[1]["p_value"] if len(seps) > 1 else 1.0)

    ax.plot(layers_x, [-np.log10(p) if p > 0 else 10 for p in pc1_pvals],
            color="#1E88E5", linewidth=2, label="PC1")
    ax.plot(layers_x, [-np.log10(p) if p > 0 else 10 for p in pc2_pvals],
            color="#E53935", linewidth=2, label="PC2")
    ax.axhline(y=-np.log10(0.05), color="gray", linestyle="--", linewidth=1,
               label="p = .05")
    ax.set_xlabel("Layer")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"Mann-Whitney U Significance — {config.MODEL_LABEL}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    figures["mw_pvals"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    fig_num = 0
    html_parts = []
    html_parts.append(build_html_header("Activation PCA Report", config.MODEL_LABEL))

    html_parts.append(f"""
<div class="stat">
<strong>Summary:</strong> {n_chars} characters ({n_ai} AI, {n_hu} human),
{n_layers} layers.
Peak silhouette: Layer {peak_sil_layer} ({peak_sil:.3f}).
Peak LDA variance: Layer {peak_lda_layer} ({peak_lda_frac*100:.1f}%).
</div>
""")

    html_parts.append(build_toc(SECTIONS))

    html_parts.append(f"""
<h2 id="research-question">1. Research Question</h2>
<p>Does {config.MODEL_LABEL} spontaneously represent a human/AI distinction as a natural
axis of variation in its internal activations? Specifically, we ask two complementary
questions:</p>
<ol>
<li><strong>Unsupervised:</strong> Do the top principal components of the activation space
    naturally separate human from AI characters — without any knowledge of the labels?</li>
<li><strong>Supervised:</strong> How much of the total variance in character activations lies
    along the <em>optimal</em> human/AI discriminant direction (Fisher's LDA), and is this
    direction already captured by the unsupervised PCs?</li>
</ol>
<p>This complements the RSA analysis. RSA asks "does the geometry <em>correlate</em> with
a categorical structure?" PCA asks "does the categorical structure emerge as a
<em>principal axis</em> of the space?" A strong RSA result with weak PCA separation would
mean the human/AI distinction exists but is a minor dimension; strong PCA separation means
it dominates the representation.</p>

<h2 id="analysis-approach">2. Analysis Approach</h2>
<div class="method">
<strong>Step-by-step procedure:</strong>
<ol>
<li><strong>Input:</strong> Character activations extracted by the activation RSA script:
    a ({n_chars}, {n_layers}, 5120) tensor of residual-stream activations at the last token
    of "Think about {{Name}}." prompts.</li>
<li><strong>Per-layer PCA:</strong> At each of {n_layers} layers, the ({n_chars} &times; 5120)
    activation matrix is centered (subtract mean) and decomposed via SVD. This yields principal
    components (PCs) — orthogonal directions of maximum variance. PC1 captures the most variance,
    PC2 the next most, etc.</li>
<li><strong>Variance explained:</strong> The fraction of total variance captured by each PC.
    If PC1 captures a large fraction and separates the groups, the human/AI distinction is the
    dominant axis of variation.</li>
<li><strong>Fisher's LDA direction:</strong> The supervised discriminant direction is computed
    as the (normalized) difference between the mean AI activation and the mean human activation.
    The fraction of total variance along this axis is measured. High values mean the human/AI
    axis captures a large share of what varies in the representation.</li>
<li><strong>Relationship between PCA and LDA:</strong> If the LDA axis aligns with PC1, the
    human/AI distinction is the <em>single most important</em> axis of variation. If LDA
    variance is high but PC1 separation is weak, the human/AI axis exists but is spread across
    multiple PCs.</li>
<li><strong>Mann-Whitney U tests:</strong> At each layer, PC1 and PC2 scores are tested for
    significant differences between AI and human groups. This gives a formal significance
    test for group separation on each PC.</li>
<li><strong>Silhouette score:</strong> At each layer, characters are projected into 2D (PC1/PC2)
    and the silhouette score is computed. Silhouette ranges from -1 to +1: +1 means characters
    are perfectly clustered by type, 0 means no structure, -1 means characters are assigned to
    the wrong cluster. It accounts for both within-group tightness and between-group separation.</li>
</ol>
</div>

<div class="interpret">
<strong>How to interpret:</strong>
<ul>
<li><strong>Layer profile:</strong> Shows how the human/AI structure waxes and wanes across
    layers. Early layers often reflect surface/token-level features; middle-to-late layers
    capture abstract semantic properties.</li>
<li><strong>High silhouette + high PC1 variance + significant Mann-Whitney:</strong> The human/AI
    distinction is the dominant, natural axis of variation at that layer.</li>
<li><strong>High LDA variance but low silhouette:</strong> The human/AI information exists
    but is entangled with other variance — a supervised classifier could find it, but it
    doesn't dominate the unsupervised structure.</li>
<li><strong>Character scatter:</strong> Direct visualization of where each character sits.
    Clear group separation means the model's "default" representation of characters encodes
    their type.</li>
</ul>
</div>
""")

    html_parts.append(expanded_concepts_stimuli_html())

    # Layer profile
    fig_num += 1
    html_parts.append(f"""
<h2 id="layer-profile">4. Layer Profile</h2>
<p>Three metrics across all {n_layers} layers:</p>
<ul>
<li><strong>Top panel:</strong> Variance explained by PC1 and PC2 (unsupervised).
    High PC1 variance means a single direction dominates the representation.</li>
<li><strong>Middle panel:</strong> Variance along the supervised LDA axis (Fisher's
    discriminant direction between AI and human groups). Higher = the human/AI direction
    captures more of the total variance.</li>
<li><strong>Bottom panel:</strong> Silhouette score for AI/human clustering in PC1/PC2
    space. Above 0 = some clustering; above 0.5 = strong clustering.</li>
</ul>
""")
    html_parts.append(html_figure(
        figures['layer_profile'],
        "Three complementary metrics across transformer layers: PC1/PC2 explained variance "
        "(top), LDA explained variance (middle), and silhouette score (bottom). Higher values "
        "indicate stronger AI/human separation at that layer.",
        fig_num=fig_num,
        alt="Layer profile",
    ))

    # Scatter at peak
    if "scatter_peak" in figures:
        fig_num += 1
        html_parts.append(f"""
<h2 id="scatter">5. Character Positions at Peak Layer (L{peak_sil_layer})</h2>
<p>Each character projected onto the first two principal components at the layer with the
best silhouette score. Red squares = AI characters, blue circles = human characters.
Large X markers = group centroids. Clear separation between the red and blue clusters
indicates that the model's internal representation naturally distinguishes human from AI
characters.</p>
""")
        html_parts.append(html_figure(
            figures['scatter_peak'],
            "Characters projected onto PC1 vs PC2 of activation space at the peak "
            "silhouette layer. Red squares = AI, blue circles = human. X markers = "
            "group centroids.",
            fig_num=fig_num,
            alt="Character scatter at peak layer",
        ))

    # Mann-Whitney significance
    fig_num += 1
    html_parts.append(f"""
<h2 id="significance">6. Statistical Significance Across Layers</h2>
<p>Mann-Whitney U test p-values for group separation on PC1 and PC2, plotted as
-log10(p). The horizontal dashed line marks p = .05. Points above this line are
significant. This shows at which layers each PC carries significant human/AI
information.</p>
""")
    html_parts.append(html_figure(
        figures['mw_pvals'],
        "\u2212log10(p) from Mann-Whitney U tests on PC1 scores at each layer. "
        "Dashed line marks p = .05. Peaks indicate layers where AI and human character "
        "activations are most separable.",
        fig_num=fig_num,
        alt="Mann-Whitney p-values",
    ))

    html_parts.append(build_html_footer())

    out_dir = os.path.join(str(rdir), dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "activation_pca_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html_parts))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate activation PCA report with charts"
    )
    add_model_argument(parser)
    add_dataset_argument(parser)
    parser.add_argument("--both", action="store_true",
                        help="Generate for both chat and base models")
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Generating report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
