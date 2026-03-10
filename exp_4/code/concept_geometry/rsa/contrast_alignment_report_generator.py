#!/usr/bin/env python3
"""
Contrast-Character Alignment Report Generator

Generates an HTML report from pre-computed contrast alignment results.
Run on login node (CPU only, <1 min).

Usage:
    python concept_geometry/rsa/contrast_alignment_report_generator.py --model llama2_13b_chat

Env: llama2_env
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
import matplotlib.colors as mcolors

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config, set_model, add_model_argument, data_dir, results_phase_dir
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header, build_html_footer,
    html_figure,
)


# ========================== CONFIG ========================== #

EXTRA_CSS = """
    .concept-table td.positive { background: #e8f5e9; }
    .concept-table td.negative { background: #ffebee; }
    .concept-table td.strong-positive { background: #c8e6c9; font-weight: bold; }
    .concept-table td.strong-negative { background: #ffcdd2; font-weight: bold; }
    .entity-tag { background: #e3f2fd; color: #1565C0; padding: 2px 6px;
                  border-radius: 3px; font-size: 0.8em; }
    .control-tag { background: #f3e5f5; color: #7B1FA2; padding: 2px 6px;
                   border-radius: 3px; font-size: 0.8em; }
"""

DIM_LABELS = {
    0: "Baseline (entity)", 1: "Phenomenology", 2: "Emotions",
    3: "Agency", 4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social", 8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpfulness", 14: "Biological",
    15: "Shapes (ctrl)", 16: "Mind", 17: "Attention",
    25: "Beliefs", 26: "Desires", 27: "Goals",
    29: "Shapes flip (ctrl)", 30: "Granite/Sandstone (ctrl)",
    31: "Squares/Triangles (ctrl)", 32: "Horizontal/Vertical (ctrl)",
}


# ========================== FIGURE GENERATORS ========================== #

def make_layer_profile_plot(results, n_layers):
    """Line plot: mean alignment across layers for entity vs control dims."""
    entity = [r for r in results if r["is_entity_framed"]]
    control = [r for r in results if r["is_control"]]

    layers = np.arange(n_layers)

    def mean_alignment_curve(concept_list):
        curves = []
        for r in concept_list:
            curves.append([l["combined_alignment"] for l in r["layers"]])
        return np.mean(curves, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    if entity:
        entity_curve = mean_alignment_curve(entity)
        ax.plot(layers, entity_curve, "b-", linewidth=2, label=f"Entity-framed (n={len(entity)})")
    if control:
        control_curve = mean_alignment_curve(control)
        ax.plot(layers, control_curve, "r--", linewidth=2, label=f"Controls (n={len(control)})")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment Score")
    ax.set_title("Mean Contrast-Character Alignment Across Layers")
    ax.legend()
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    return fig


def make_per_concept_layer_plot(results, n_layers):
    """Individual layer profiles for all entity-framed concepts."""
    entity = [r for r in results if r["is_entity_framed"]]
    if not entity:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(n_layers)
    cmap = plt.cm.tab20

    for i, r in enumerate(sorted(entity, key=lambda x: x["dim_id"])):
        curve = [l["combined_alignment"] for l in r["layers"]]
        color = cmap(i / max(len(entity) - 1, 1))
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        ax.plot(layers, curve, linewidth=1.2, color=color, alpha=0.7, label=label)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Alignment Score")
    ax.set_title("Per-Concept Alignment Profiles (Entity-Framed)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    return fig


def make_bar_plot(results):
    """Bar chart of peak alignment for all concepts, color-coded."""
    sorted_results = sorted(results, key=lambda r: r["peak_alignment"], reverse=True)

    names = []
    values = []
    colors = []
    for r in sorted_results:
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        names.append(label)
        values.append(r["peak_alignment"])
        if r["is_control"]:
            colors.append("#CE93D8")  # purple for controls
        elif r["is_entity_framed"]:
            colors.append("#64B5F6")  # blue for entity-framed
        else:
            colors.append("#BDBDBD")  # gray for other

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Peak Alignment Score")
    ax.set_title("Peak Alignment by Concept (blue=entity, purple=control)")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def make_similarity_heatmap(sim_matrices, dim_ids, char_keys, concept_idx, layer,
                            dim_label):
    """2-row heatmap: human-framed and AI-framed similarities to all characters."""
    sims = sim_matrices[concept_idx, layer]  # (2, n_characters)

    # Sort characters: humans first, then AI
    from concept_geometry.characters import CHARACTER_TYPES
    human_idx = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "human"]
    ai_idx = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "ai"]
    order = human_idx + ai_idx
    sorted_keys = [char_keys[i] for i in order]
    sorted_sims = sims[:, order]

    fig, ax = plt.subplots(figsize=(14, 2.5))
    im = ax.imshow(sorted_sims, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(sorted_sims)),
                   vmax=np.max(np.abs(sorted_sims)))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Human-framed", "AI-framed"])
    ax.set_xticks(range(len(sorted_keys)))
    ax.set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{dim_label} — Layer {layer}")

    # Add divider between human and AI characters
    ax.axvline(len(human_idx) - 0.5, color="black", linewidth=1.5)

    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    plt.tight_layout()
    return fig


# ========================== REPORT BUILDER ========================== #

def generate_report(model_key):
    """Build HTML report from saved data."""
    set_model(model_key)

    ddir = data_dir("concept_geometry/rsa", "contrast_alignment")
    rdir = results_phase_dir("concept_geometry/rsa", "contrast_alignment")

    # Load results
    with open(os.path.join(ddir, "alignment_results.json")) as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["concepts"]
    n_layers = summary["n_layers"]

    # Load similarity matrices for heatmaps
    sim_data = np.load(os.path.join(ddir, "similarity_matrices.npz"))
    sim_matrices = sim_data["similarities"]
    dim_ids = list(sim_data["dim_ids"])
    char_keys = list(sim_data["character_keys"])

    # ── Build HTML ──
    sections = [
        {"id": "methods", "label": "Methods"},
        {"id": "layer-profile", "label": "Layer Profile"},
        {"id": "peak-alignment", "label": "Peak Alignment"},
        {"id": "per-concept", "label": "Per-Concept Profiles"},
        {"id": "concept-table", "label": "Concept Table"},
        {"id": "heatmaps", "label": "Similarity Heatmaps"},
        {"id": "controls", "label": "Control Dimensions"},
    ]

    html = build_html_header(
        "Contrast-Character Alignment", config.MODEL_LABEL, css=EXTRA_CSS
    )
    html += build_toc(sections)

    # ── Methods ──
    html += '<h2 id="methods">Methods</h2>\n'
    html += '<div class="method">\n'
    html += "<p><strong>Research question:</strong> Do entity-framed concept prompts from "
    html += "exp_3 contrasts activate representations closer to the \"correct\" character "
    html += "group? Human-framed prompts should be closer to human characters; AI-framed "
    html += "prompts closer to AI characters.</p>\n"
    html += "<ol>\n"
    html += f"<li>Pre-computed character activations for {summary['n_characters']} characters "
    html += "(15 AI + 15 human) from <code>activation_rsa.py</code></li>\n"
    html += f"<li>Extracted mean activations for {summary['n_concepts']} contrast concepts "
    html += f"({summary['n_entity_framed']} entity-framed + {summary['n_control']} controls), "
    html += "80 prompts each (40 human-framed + 40 AI-framed)</li>\n"
    html += "<li>Cosine similarity between each concept framing and each character at every layer</li>\n"
    html += "<li>Alignment = mean(sim to correct group) &minus; mean(sim to incorrect group)</li>\n"
    html += "</ol>\n</div>\n"

    # ── Layer Profile ──
    html += '<h2 id="layer-profile">Layer Profile</h2>\n'
    fig = make_layer_profile_plot(results, n_layers)
    html += html_figure(fig_to_b64(fig), "Mean alignment score across layers for "
                        "entity-framed vs control dimensions.", fig_num=1)
    plt.close(fig)

    entity_peak = summary.get("entity_mean_peak_alignment")
    control_peak = summary.get("control_mean_peak_alignment")
    html += '<div class="stat">\n'
    if entity_peak is not None:
        html += f"<p><strong>Entity-framed mean peak alignment:</strong> {entity_peak:+.4f}</p>\n"
    if control_peak is not None:
        html += f"<p><strong>Control mean peak alignment:</strong> {control_peak:+.4f}</p>\n"
    html += "</div>\n"

    if entity_peak is not None and control_peak is not None:
        if entity_peak > 0 and entity_peak > control_peak * 2:
            html += '<div class="success"><p>Entity-framed concepts show substantially '
            html += 'higher alignment than controls, consistent with the hypothesis.</p></div>\n'
        elif entity_peak <= 0:
            html += '<div class="warning"><p>Entity-framed concepts do not show positive '
            html += 'alignment on average.</p></div>\n'

    # ── Peak Alignment Bar ──
    html += '<h2 id="peak-alignment">Peak Alignment by Concept</h2>\n'
    fig = make_bar_plot(results)
    html += html_figure(fig_to_b64(fig), "Peak alignment score per concept. "
                        "Blue = entity-framed, purple = control.", fig_num=2)
    plt.close(fig)

    # ── Per-Concept Layer Profiles ──
    html += '<h2 id="per-concept">Per-Concept Layer Profiles</h2>\n'
    fig = make_per_concept_layer_plot(results, n_layers)
    if fig is not None:
        html += html_figure(fig_to_b64(fig), "Individual layer profiles for all "
                            "entity-framed concepts.", fig_num=3)
        plt.close(fig)

    # ── Concept Table ──
    html += '<h2 id="concept-table">Concept Table (Peak Layer)</h2>\n'
    html += '<table class="concept-table">\n'
    html += "<tr><th>Dim</th><th>Concept</th><th>Type</th>"
    html += "<th>Peak Layer</th><th>Human Align</th><th>AI Align</th>"
    html += "<th>Combined</th></tr>\n"

    for r in sorted(results, key=lambda x: x["peak_alignment"], reverse=True):
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        tag = ('<span class="entity-tag">entity</span>' if r["is_entity_framed"]
               else '<span class="control-tag">control</span>' if r["is_control"]
               else "other")
        peak = r["layers"][r["peak_layer"]]
        combined = peak["combined_alignment"]

        if combined > 0.01:
            cls = "strong-positive" if combined > 0.02 else "positive"
        elif combined < -0.01:
            cls = "strong-negative" if combined < -0.02 else "negative"
        else:
            cls = ""

        html += (f"<tr><td>{r['dim_id']}</td><td>{label}</td><td>{tag}</td>"
                 f"<td>{r['peak_layer']}</td>"
                 f"<td>{peak['human_alignment']:+.4f}</td>"
                 f"<td>{peak['ai_alignment']:+.4f}</td>"
                 f'<td class="{cls}">{combined:+.4f}</td></tr>\n')

    html += "</table>\n"

    # ── Similarity Heatmaps ──
    html += '<h2 id="heatmaps">Similarity Heatmaps (Selected Concepts)</h2>\n'
    html += "<p>Showing heatmaps at each concept's peak layer. Characters sorted: "
    html += "humans (left) | AI (right).</p>\n"

    # Pick top 3 entity-framed + 1 control
    entity_sorted = sorted(
        [r for r in results if r["is_entity_framed"]],
        key=lambda x: x["peak_alignment"], reverse=True,
    )
    control_sorted = [r for r in results if r["is_control"]]

    heatmap_concepts = entity_sorted[:3] + control_sorted[:1]
    fig_num = 4
    for r in heatmap_concepts:
        ci = dim_ids.index(r["dim_id"])
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        fig = make_similarity_heatmap(
            sim_matrices, dim_ids, char_keys, ci, r["peak_layer"], label
        )
        tag = "entity-framed" if r["is_entity_framed"] else "control"
        html += html_figure(
            fig_to_b64(fig),
            f"{label} ({tag}) at peak layer {r['peak_layer']}. "
            f"Combined alignment = {r['peak_alignment']:+.4f}.",
            fig_num=fig_num,
        )
        plt.close(fig)
        fig_num += 1

    # ── Controls Section ──
    html += '<h2 id="controls">Control Dimensions</h2>\n'
    if control_sorted:
        html += '<div class="interpret">\n'
        html += "<p>Control dimensions test orthogonal contrasts (shapes, rocks) with "
        html += "no expected human/AI alignment.</p>\n"
        html += "<table>\n<tr><th>Dim</th><th>Concept</th><th>Peak Layer</th>"
        html += "<th>Peak Alignment</th></tr>\n"
        for r in control_sorted:
            label = DIM_LABELS.get(r["dim_id"], r["name"])
            html += (f"<tr><td>{r['dim_id']}</td><td>{label}</td>"
                     f"<td>{r['peak_layer']}</td>"
                     f"<td>{r['peak_alignment']:+.4f}</td></tr>\n")
        html += "</table>\n"

        mean_ctrl = np.mean([r["peak_alignment"] for r in control_sorted])
        if abs(mean_ctrl) < 0.005:
            html += "<p class=\"match\">Controls show near-zero alignment as expected.</p>\n"
        elif abs(mean_ctrl) > 0.01:
            html += ("<p class=\"mismatch\">Controls show non-trivial alignment — "
                     "investigate potential confounds.</p>\n")
        html += "</div>\n"
    else:
        html += "<p>No control dimensions found.</p>\n"

    html += build_html_footer()

    # ── Write ──
    report_path = os.path.join(rdir, "contrast_alignment_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report written to: {report_path}")
    return report_path


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Generate Contrast-Character Alignment Report"
    )
    add_model_argument(parser)
    args = parser.parse_args()

    generate_report(args.model)


if __name__ == "__main__":
    main()
