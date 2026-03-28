#!/usr/bin/env python3
"""
Standalone-Character Alignment Report Generator

For each standalone concept (no human/AI framing), shows whether its activations
lean closer to human or AI characters.

Usage:
    python expanded_mental_concepts/internals/standalone_alignment/standalone_alignment_report_generator.py --model llama2_13b_chat

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

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import config, set_model, add_model_argument, data_dir, results_dir
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header, build_html_footer,
    html_figure, expanded_concepts_stimuli_html,
)


# ========================== CONFIG ========================== #

EXTRA_CSS = """
    .human-bias { background: #e3f2fd; }
    .ai-bias { background: #fce4ec; }
    .strong-human { background: #bbdefb; font-weight: bold; }
    .strong-ai { background: #f8bbd0; font-weight: bold; }
"""

DIM_LABELS = {
    1: "Phenomenology", 2: "Emotions", 3: "Agency", 4: "Intentions",
    5: "Prediction", 6: "Cognitive", 7: "Social", 8: "Embodiment",
    9: "Roles", 10: "Animacy", 11: "Formality", 12: "Expertise",
    13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Human (identity)", 17: "AI (identity)", 18: "Attention", 19: "Mind",
    25: "Beliefs", 26: "Desires", 27: "Goals",
    30: "Granite/Sandstone", 31: "Squares/Triangles", 32: "Horizontal/Vertical",
}


# ========================== FIGURES ========================== #

def make_layer_profile_plot(results, n_layers):
    """Line plot: bias across layers for all concepts."""
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(n_layers)
    cmap = plt.cm.tab20

    for i, r in enumerate(sorted(results, key=lambda x: x["dim_id"])):
        curve = [l["human_ai_bias"] for l in r["layers"]]
        color = cmap(i / max(len(results) - 1, 1))
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        ax.plot(layers, curve, linewidth=1.0, color=color, alpha=0.7, label=label)

    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Bias (+ = human, - = AI)")
    ax.set_title("Standalone Concept Bias Toward Human vs AI Characters")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.set_xlim(0, n_layers - 1)
    plt.tight_layout()
    return fig


def make_bar_plot(results):
    """Horizontal bar of peak bias, sorted."""
    sorted_results = sorted(results, key=lambda r: r["peak_bias"], reverse=True)

    names = [DIM_LABELS.get(r["dim_id"], r["name"]) for r in sorted_results]
    values = [r["peak_bias"] for r in sorted_results]
    colors = ["#64B5F6" if v > 0 else "#F48FB1" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Peak Bias (+ = human, - = AI)")
    ax.set_title("Standalone Concept Alignment (blue = human-leaning, pink = AI-leaning)")
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def make_similarity_heatmap(sim_matrices, dim_ids, char_keys, concept_idx, layer,
                            dim_label):
    """Heatmap of similarities to all characters at a given layer."""
    from entities.characters import CHARACTER_TYPES
    sims = sim_matrices[concept_idx, layer]  # (n_characters,)

    human_idx = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "human"]
    ai_idx = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "ai"]
    order = human_idx + ai_idx
    sorted_keys = [char_keys[i] for i in order]
    sorted_sims = sims[order]

    fig, ax = plt.subplots(figsize=(14, 1.8))
    im = ax.imshow(sorted_sims.reshape(1, -1), aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(sorted_sims)),
                   vmax=np.max(np.abs(sorted_sims)))
    ax.set_yticks([0])
    ax.set_yticklabels([dim_label])
    ax.set_xticks(range(len(sorted_keys)))
    ax.set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=7)
    ax.axvline(len(human_idx) - 0.5, color="black", linewidth=1.5)
    plt.colorbar(im, ax=ax, label="Cosine similarity", shrink=0.8)
    plt.tight_layout()
    return fig


# ========================== REPORT ========================== #

def generate_report(model_key):
    set_model(model_key)

    ddir = data_dir("expanded_mental_concepts", "internals", "standalone_alignment")
    rdir = results_dir("expanded_mental_concepts", "internals", "standalone_alignment")

    with open(os.path.join(ddir, "alignment_results.json")) as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["concepts"]
    n_layers = summary["n_layers"]

    sim_data = np.load(os.path.join(ddir, "similarity_matrices.npz"))
    sim_matrices = sim_data["similarities"]
    dim_ids = list(sim_data["dim_ids"])
    char_keys = list(sim_data["character_keys"])

    sections = [
        {"id": "methods", "label": "Methods"},
        {"id": "stimuli", "label": "Stimuli"},
        {"id": "layer-profile", "label": "Layer Profiles"},
        {"id": "peak-bias", "label": "Peak Bias"},
        {"id": "concept-table", "label": "Concept Table"},
        {"id": "heatmaps", "label": "Similarity Heatmaps"},
    ]

    html = build_html_header(
        "Standalone-Character Alignment", config.MODEL_LABEL, css=EXTRA_CSS
    )
    html += build_toc(sections)

    # ── Methods ──
    html += '<h2 id="methods">Methods</h2>\n'
    html += '<div class="method">\n'
    html += "<p><strong>Question:</strong> Which character group (human vs AI) does each "
    html += "standalone concept naturally align with, even without explicit entity framing?</p>\n"
    html += "<ol>\n"
    html += f"<li>Pre-computed activations for {summary['n_characters']} characters</li>\n"
    html += f"<li>Extracted mean activations for {summary['n_concepts']} standalone concepts "
    html += "(40 prompts each, no human/AI framing)</li>\n"
    html += "<li>Cosine similarity to each character at every layer</li>\n"
    html += "<li>Bias = mean(sim to human chars) &minus; mean(sim to AI chars)</li>\n"
    html += "</ol>\n"
    html += "<p>Positive bias = concept closer to human characters. "
    html += "Negative bias = closer to AI characters.</p>\n"
    html += "</div>\n"

    html += expanded_concepts_stimuli_html()

    # ── Layer profiles ──
    html += '<h2 id="layer-profile">Layer Profiles</h2>\n'
    fig = make_layer_profile_plot(results, n_layers)
    html += html_figure(fig_to_b64(fig), "Human-AI bias across layers for all "
                        "standalone concepts.", fig_num=1)
    plt.close(fig)

    # ── Peak bias bar ──
    html += '<h2 id="peak-bias">Peak Bias by Concept</h2>\n'
    fig = make_bar_plot(results)
    html += html_figure(fig_to_b64(fig), "Peak bias per concept. "
                        "Blue = human-leaning, pink = AI-leaning.", fig_num=2)
    plt.close(fig)

    # ── Concept table ──
    html += '<h2 id="concept-table">Concept Table (Peak Layer)</h2>\n'
    html += "<table>\n<tr><th>Dim</th><th>Concept</th><th>Peak Layer</th>"
    html += "<th>Mean Sim Human</th><th>Mean Sim AI</th><th>Bias</th><th>Direction</th></tr>\n"

    for r in sorted(results, key=lambda x: x["peak_bias"], reverse=True):
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        peak = r["layers"][r["peak_layer"]]
        bias = peak["human_ai_bias"]
        direction = "human" if bias > 0 else "AI"

        if abs(bias) > 0.01:
            cls = "strong-human" if bias > 0 else "strong-ai"
        elif abs(bias) > 0.003:
            cls = "human-bias" if bias > 0 else "ai-bias"
        else:
            cls = ""

        html += (f"<tr><td>{r['dim_id']}</td><td>{label}</td>"
                 f"<td>{r['peak_layer']}</td>"
                 f"<td>{peak['mean_sim_human']:.4f}</td>"
                 f"<td>{peak['mean_sim_ai']:.4f}</td>"
                 f'<td class="{cls}">{bias:+.4f}</td>'
                 f"<td>{direction}</td></tr>\n")

    html += "</table>\n"

    # ── Heatmaps ──
    html += '<h2 id="heatmaps">Similarity Heatmaps (Selected)</h2>\n'
    html += "<p>Top 3 human-leaning + top 3 AI-leaning concepts at peak layer.</p>\n"

    sorted_by_bias = sorted(results, key=lambda x: x["peak_bias"], reverse=True)
    show_concepts = sorted_by_bias[:3] + sorted_by_bias[-3:]

    fig_num = 3
    for r in show_concepts:
        ci = dim_ids.index(r["dim_id"])
        label = DIM_LABELS.get(r["dim_id"], r["name"])
        fig = make_similarity_heatmap(
            sim_matrices, dim_ids, char_keys, ci, r["peak_layer"], label
        )
        direction = "human" if r["peak_bias"] > 0 else "AI"
        html += html_figure(
            fig_to_b64(fig),
            f"{label} at peak layer {r['peak_layer']}. "
            f"Bias = {r['peak_bias']:+.4f} ({direction}-leaning).",
            fig_num=fig_num,
        )
        plt.close(fig)
        fig_num += 1

    html += build_html_footer()

    report_path = os.path.join(rdir, "standalone_alignment_report.html")
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report written to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Standalone-Character Alignment Report"
    )
    add_model_argument(parser)
    args = parser.parse_args()
    generate_report(args.model)


if __name__ == "__main__":
    main()
