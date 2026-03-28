#!/usr/bin/env python3
"""
Experiment 4: Neural PCA Report Generator

Self-contained HTML report comparing neural PCA results across conditions
(with_self, without_self) for a given model. Includes:
    1. Methodology comparison (neural PCA vs RSA vs behavioral PCA)
    2. Variance structure (scree plots across layers)
    3. PC-human correlation profiles (layerwise)
    4. Procrustes alignment profile (layerwise disparity)
    5. Entity scatter at peak layer (Procrustes-rotated vs human)
    6. MDS comparison
    7. Summary table

Usage:
    python internals/2a_neural_pca_report_generator.py --model llama2_13b_chat
    python internals/2a_neural_pca_report_generator.py --model llama2_13b_base

Env: llama2_env (CPU only, needs matplotlib)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir,
)
from utils.utils import nice_entity
from utils.report_utils import (
    fig_to_b64, build_html_header, build_html_footer,
    build_toc, html_figure, add_dataset_argument,
    gray_entities_stimuli_html,
)
from entities.gray_entities import GRAY_ET_AL_SCORES, ENTITY_NAMES

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

C_EXP = "#2166ac"
C_AG = "#b2182b"
C_PROC = "#4daf4a"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(tag):
    """Load neural_pca_results.npz and neural_pca_analysis.json for a condition."""
    ddir = data_dir("gray_simple", "internals", tag)
    npz_path = ddir / "neural_pca_results.npz"
    json_path = ddir / "neural_pca_analysis.json"

    if not npz_path.exists():
        return None, None

    d = np.load(str(npz_path), allow_pickle=True)
    with open(str(json_path)) as f:
        summary = json.load(f)

    return d, summary


# ============================================================================
# FIGURES
# ============================================================================

def fig_scree(data_ws, data_ns):
    """Scree plot: explained variance of top PCs across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, d, title in [(axes[0], data_ns, "Without Self"),
                          (axes[1], data_ws, "With Self")]:
        if d is None:
            ax.set_title(f"{title} (no data)")
            continue
        ev = d["explained_var"]
        n_layers = ev.shape[0]
        layers = np.arange(n_layers)
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        for pc in range(min(5, ev.shape[1])):
            ax.plot(layers, ev[:, pc] * 100, color=colors[pc],
                    label=f"PC{pc+1}", linewidth=1.5)
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)

    axes[0].set_ylabel("Explained Variance (%)")
    fig.suptitle("Variance Captured by Top PCs Across Layers", fontsize=14)
    plt.tight_layout()
    return fig


def fig_pc_correlations(data_ws, data_ns):
    """Layerwise PC-human correlations for PC1 and PC2."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for col, d, title in [(0, data_ns, "Without Self"),
                            (1, data_ws, "With Self")]:
        if d is None:
            for row in range(2):
                axes[row, col].set_title(f"{title} (no data)")
            continue
        n_layers = d["pc_exp_corr"].shape[0]
        layers = np.arange(n_layers)

        for row, pc in enumerate([0, 1]):
            ax = axes[row, col]
            exp_corr = d["pc_exp_corr"][:, pc]
            ag_corr = d["pc_ag_corr"][:, pc]

            ax.plot(layers, exp_corr, color=C_EXP, label="Experience",
                    linewidth=1.5)
            ax.plot(layers, ag_corr, color=C_AG, label="Agency",
                    linewidth=1.5)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title(f"PC{pc+1} — {title}")
            if col == 0:
                ax.set_ylabel("Spearman rho")
            ax.legend(fontsize=8)

    for ax in axes[1, :]:
        ax.set_xlabel("Layer")
    fig.suptitle("PC-Human Correlation Across Layers", fontsize=14)
    plt.tight_layout()
    return fig


def fig_procrustes_profile(data_ws, data_ns):
    """Layerwise Procrustes disparity for PCA and MDS."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, d, title in [(axes[0], data_ns, "Without Self"),
                          (axes[1], data_ws, "With Self")]:
        if d is None:
            ax.set_title(f"{title} (no data)")
            continue
        n_layers = len(d["procrustes_disparity"])
        layers = np.arange(n_layers)

        ax.plot(layers, d["procrustes_disparity"], color=C_PROC,
                label="PCA (top 2 PCs)", linewidth=1.5)
        ax.plot(layers, d["mds_procrustes_disparity"], color="#984ea3",
                label="MDS (cosine)", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_title(title)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Procrustes Disparity (lower = better)")
    fig.suptitle("Procrustes Alignment to Human 2D Space", fontsize=14)
    plt.tight_layout()
    return fig


def fig_entity_scatter(data, summary, entity_keys, tag):
    """Entity scatter at best Procrustes layer: aligned PCA vs human."""
    if data is None or "procrustes_best_layer" not in summary:
        return None

    best_layer = summary["procrustes_best_layer"]
    coords = data["procrustes_coords"][best_layer]
    disp = data["procrustes_disparity"][best_layer]

    human_exp = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    human_ag = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Procrustes-aligned neural PCA
    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], s=60, c=C_PROC, zorder=3)
    for i, k in enumerate(entity_keys):
        ax.annotate(nice_entity(k), (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Dim 1 (aligned)")
    ax.set_ylabel("Dim 2 (aligned)")
    ax.set_title(f"Neural PCA (layer {best_layer}, Procrustes-aligned)")

    # Right: Human scores
    ax = axes[1]
    ax.scatter(human_exp, human_ag, s=60, c="#666", zorder=3)
    for i, k in enumerate(entity_keys):
        ax.annotate(nice_entity(k), (human_exp[i], human_ag[i]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Experience")
    ax.set_ylabel("Agency")
    ax.set_title("Human (Gray et al. 2007)")

    fig.suptitle(
        f"Entity Geometry — {tag.replace('_', ' ').title()} "
        f"(disparity={disp:.3f})",
        fontsize=14,
    )
    plt.tight_layout()
    return fig


def fig_mds_scatter(data, summary, entity_keys, tag):
    """Entity scatter at best MDS Procrustes layer."""
    if data is None or "mds_procrustes_best_layer" not in summary:
        return None

    best_layer = summary["mds_procrustes_best_layer"]
    coords = data["mds_procrustes_coords"][best_layer]
    disp = data["mds_procrustes_disparity"][best_layer]

    human_exp = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    human_ag = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(coords[:, 0], coords[:, 1], s=60, c="#984ea3", zorder=3)
    for i, k in enumerate(entity_keys):
        ax.annotate(nice_entity(k), (coords[i, 0], coords[i, 1]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Dim 1 (aligned)")
    ax.set_ylabel("Dim 2 (aligned)")
    ax.set_title(f"MDS (layer {best_layer}, Procrustes-aligned)")

    ax = axes[1]
    ax.scatter(human_exp, human_ag, s=60, c="#666", zorder=3)
    for i, k in enumerate(entity_keys):
        ax.annotate(nice_entity(k), (human_exp[i], human_ag[i]),
                    fontsize=7, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    ax.set_xlabel("Experience")
    ax.set_ylabel("Agency")
    ax.set_title("Human (Gray et al. 2007)")

    fig.suptitle(
        f"MDS Entity Geometry — {tag.replace('_', ' ').title()} "
        f"(disparity={disp:.3f})",
        fontsize=14,
    )
    plt.tight_layout()
    return fig


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def summary_table_html(summaries):
    """Build an HTML summary table across conditions."""
    rows = []
    for s in summaries:
        if s is None:
            continue
        tag = s["condition"]
        rows.append(f"<tr><td colspan='4' style='background:#f0f0f0'>"
                    f"<strong>{tag.replace('_', ' ').title()}</strong> "
                    f"({s['n_entities']} entities)</td></tr>")

        for key_prefix, label in [
            ("pc1_experience", "PC1 ↔ Experience"),
            ("pc1_agency", "PC1 ↔ Agency"),
            ("pc2_experience", "PC2 ↔ Experience"),
            ("pc2_agency", "PC2 ↔ Agency"),
        ]:
            layer = s.get(f"{key_prefix}_peak_layer", "—")
            rho = s.get(f"{key_prefix}_peak_rho", None)
            p = s.get(f"{key_prefix}_peak_p", None)
            rho_str = f"{rho:+.4f}" if rho is not None else "—"
            p_str = f"{p:.4f}" if p is not None else "—"
            sig = ' class="sig"' if p is not None and p < 0.05 else ""
            rows.append(f"<tr><td>{label}</td><td>{layer}</td>"
                        f"<td{sig}>{rho_str}</td><td{sig}>{p_str}</td></tr>")

        for key_prefix, label in [
            ("procrustes", "PCA Procrustes"),
            ("mds_procrustes", "MDS Procrustes"),
        ]:
            layer = s.get(f"{key_prefix}_best_layer", "—")
            disp = s.get(f"{key_prefix}_best_disparity", None)
            disp_str = f"{disp:.4f}" if disp is not None else "—"
            rows.append(f"<tr><td>{label}</td><td>{layer}</td>"
                        f"<td>{disp_str}</td><td>—</td></tr>")

    return (
        "<table>\n"
        "<tr><th>Metric</th><th>Peak Layer</th><th>Rho / Disparity</th>"
        "<th>p-value</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n"
    )


# ============================================================================
# MAIN
# ============================================================================

def generate_report(args):
    set_model(args.model)

    data_ns, summary_ns = load_data("without_self")
    data_ws, summary_ws = load_data("with_self")

    if data_ns is None and data_ws is None:
        print("No neural PCA data found for either condition. Run 2_neural_pca.py first.")
        return

    entity_keys_ns = [k for k in ENTITY_NAMES if k != "you_self"]
    entity_keys_ws = list(ENTITY_NAMES)

    # Build HTML
    html = build_html_header("Neural PCA Report", config.MODEL_LABEL)

    sections = [
        {"id": "methodology", "label": "1. Methodology"},
        {"id": "stimuli", "label": "2. Stimuli"},
        {"id": "variance", "label": "3. Variance Structure"},
        {"id": "correlations", "label": "4. PC-Human Correlations"},
        {"id": "procrustes", "label": "5. Procrustes Alignment"},
        {"id": "scatter-pca", "label": "6. Entity Scatter (PCA)"},
        {"id": "scatter-mds", "label": "7. Entity Scatter (MDS)"},
        {"id": "summary", "label": "8. Summary"},
    ]
    html += build_toc(sections)

    # 1. Methodology
    html += '<h2 id="methodology">1. Methodology</h2>\n'
    html += '<div class="method">\n'
    html += (
        "<p><strong>Neural PCA</strong> asks a different question than RSA. "
        "RSA tests whether the <em>overall pattern of distances</em> between "
        "entities matches human geometry. Neural PCA asks whether <em>specific "
        "directions of variance</em> in the activation space correspond to "
        "specific human dimensions (Experience, Agency).</p>\n"
        "<ol>\n"
        "<li><strong>PCA</strong>: For each layer, extract principal components "
        "of the (n_entities × hidden_dim) activation matrix. Correlate each "
        "PC with human Experience and Agency scores.</li>\n"
        "<li><strong>Procrustes alignment</strong>: Rotate and scale the top 2 "
        "PCs to best match the human 2D (Experience, Agency) space. The "
        "disparity measures residual misalignment.</li>\n"
        "<li><strong>MDS</strong>: Classical multidimensional scaling on the "
        "cosine-distance RDM → 2D coordinates. Same correlation + Procrustes "
        "analysis as PCA.</li>\n"
        "</ol>\n"
        "</div>\n"
    )

    # 2. Stimuli
    html += '<h2 id="stimuli">2. Stimuli</h2>\n'
    html += gray_entities_stimuli_html(include_capacities=False)

    # 3. Variance structure
    html += '<h2 id="variance">3. Variance Structure</h2>\n'
    fig = fig_scree(data_ws, data_ns)
    html += html_figure(
        fig_to_b64(fig), fig_num=1,
        caption="Explained variance of top 5 principal components at each layer. "
        "If mind-perception information concentrates into fewer dimensions at "
        "deeper layers, we should see PC1 variance increasing.",
    )
    plt.close(fig)

    # 4. PC-human correlations
    html += '<h2 id="correlations">4. PC-Human Correlations</h2>\n'
    fig = fig_pc_correlations(data_ws, data_ns)
    html += html_figure(
        fig_to_b64(fig), fig_num=2,
        caption="Spearman correlation between each PC and human Experience/Agency "
        "scores across layers. A strong positive or negative correlation means "
        "that PC captures the corresponding human dimension.",
    )
    plt.close(fig)

    # 5. Procrustes alignment
    html += '<h2 id="procrustes">5. Procrustes Alignment</h2>\n'
    fig = fig_procrustes_profile(data_ws, data_ns)
    html += html_figure(
        fig_to_b64(fig), fig_num=3,
        caption="Procrustes disparity (lower = better alignment) between neural "
        "2D representation and human 2D space. Solid: PCA top-2 PCs. "
        "Dashed: classical MDS on cosine-distance RDM.",
    )
    plt.close(fig)

    # 6. Entity scatter (PCA)
    html += '<h2 id="scatter-pca">6. Entity Scatter at Peak Layer (PCA)</h2>\n'
    fig_num = 4
    for entity_keys, d, s, tag in [
        (entity_keys_ns, data_ns, summary_ns, "without_self"),
        (entity_keys_ws, data_ws, summary_ws, "with_self"),
    ]:
        fig = fig_entity_scatter(d, s, entity_keys, tag)
        if fig is not None:
            html += html_figure(
                fig_to_b64(fig), fig_num=fig_num,
                caption=f"Procrustes-aligned top-2 PCs (left) vs human geometry "
                f"(right), {tag.replace('_', ' ')} condition.",
            )
            plt.close(fig)
            fig_num += 1

    # 7. Entity scatter (MDS)
    html += '<h2 id="scatter-mds">7. Entity Scatter at Peak Layer (MDS)</h2>\n'
    for entity_keys, d, s, tag in [
        (entity_keys_ns, data_ns, summary_ns, "without_self"),
        (entity_keys_ws, data_ws, summary_ws, "with_self"),
    ]:
        fig = fig_mds_scatter(d, s, entity_keys, tag)
        if fig is not None:
            html += html_figure(
                fig_to_b64(fig), fig_num=fig_num,
                caption=f"MDS Procrustes-aligned 2D (left) vs human geometry "
                f"(right), {tag.replace('_', ' ')} condition.",
            )
            plt.close(fig)
            fig_num += 1

    # 8. Summary
    html += '<h2 id="summary">8. Summary Table</h2>\n'
    html += summary_table_html([summary_ns, summary_ws])

    html += build_html_footer()

    # Write
    out_dir = results_dir("gray_simple", "internals", args.dataset)
    os.makedirs(str(out_dir), exist_ok=True)
    out_path = os.path.join(str(out_dir), "neural_pca_report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Neural PCA report generator"
    )
    add_model_argument(parser)
    add_dataset_argument(parser)
    args = parser.parse_args()
    generate_report(args)


if __name__ == "__main__":
    main()
