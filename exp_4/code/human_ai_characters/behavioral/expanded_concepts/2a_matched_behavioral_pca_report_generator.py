#!/usr/bin/env python3
"""
Matched Behavioral PCA Report Generator

Reads matched PCA results from matched_behavioral_pca.py and generates
an HTML report comparing PCA results across concept subsets.

For each subset (human-favored, AI-favored, directional, all):
    - PCA scatterplot (Factor 1 vs Factor 2, colored by type)
    - Varimax loadings table
    - Eigenvalue scree plot
    - Categorical alignment (Mann-Whitney p-values)
    - Cross-subset comparison table

Reads from:
    results/{model}/human_ai_characters/behavior/pca/data/
        matched_{subset}_pca_results.npz
        matched_{subset}_categorical_analysis.json

Writes to:
    results/{model}/human_ai_characters/behavior/pca/{dataset}/
        matched_behavioral_pca_report.html

Usage:
    python human_ai_characters/behavior/pca/matched_behavioral_pca_report_generator.py --model llama2_13b_base
    python human_ai_characters/behavior/pca/matched_behavioral_pca_report_generator.py --model llama2_13b_chat --both

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
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

SUBSETS = ["human", "ai", "directional", "all"]
SUBSET_LABELS = {
    "human": "Human-Favored (12 concepts)",
    "ai": "AI-Favored (6 concepts)",
    "directional": "Directional (18 concepts)",
    "all": "All Concepts (full set)",
}

SECTIONS = [
    {"id": "methods", "label": "1. Methods"},
    {"id": "stimuli", "label": "2. Stimuli"},
    {"id": "cross-subset", "label": "3. Cross-Subset Comparison"},
    {"id": "subset-details", "label": "4. Per-Subset Details"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate matched behavioral PCA report for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    set_model(model_key)
    ddir = data_dir("human_ai_characters", "behavioral/expanded_concepts")
    rdir = results_dir("human_ai_characters", "behavioral/expanded_concepts")

    # Load all available subsets
    subset_data = {}
    for subset in SUBSETS:
        pca_path = os.path.join(str(ddir), f"matched_{subset}_pca_results.npz")
        cat_path = os.path.join(str(ddir), f"matched_{subset}_categorical_analysis.json")
        if not os.path.exists(pca_path):
            continue

        pca = np.load(pca_path)
        cat_data = {}
        if os.path.exists(cat_path):
            with open(cat_path) as f:
                cat_data = json.load(f)

        subset_data[subset] = {
            "pca": pca,
            "categorical": cat_data.get("categorical", {}),
            "n_concepts": cat_data.get("n_concepts", 0),
        }

    if not subset_data:
        print(f"No matched PCA data found for {model_key}")
        return None

    figures = {}

    # ── Per-subset: scatter + scree ──
    for subset, data in subset_data.items():
        pca = data["pca"]
        scores_01 = pca["factor_scores_01"]
        eigenvalues = pca["eigenvalues"]
        explained = pca["explained_var_ratio"]
        char_keys = list(pca["character_keys"])
        concept_keys = list(pca["concept_keys"])
        rotated = pca["rotated_loadings"]

        n_factors = int(np.sum(eigenvalues > 1.0))

        # Scatter
        if scores_01.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            for char_key in char_keys:
                idx = char_keys.index(char_key)
                is_ai = char_key in AI_CHARACTERS
                color = "#E53935" if is_ai else "#1E88E5"
                marker = "s" if is_ai else "o"
                ax.scatter(scores_01[idx, 0], scores_01[idx, 1],
                           c=color, marker=marker, s=60, edgecolors="white",
                           linewidth=0.5, zorder=3)
                ax.annotate(CHARACTER_NAMES[char_key],
                            (scores_01[idx, 0], scores_01[idx, 1]),
                            fontsize=6, ha="left", va="bottom",
                            xytext=(3, 3), textcoords="offset points",
                            color=color, alpha=0.8)

            legend_elements = [
                Line2D([0], [0], marker="s", color="w", markerfacecolor="#E53935",
                       markersize=8, label="AI"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E88E5",
                       markersize=8, label="Human"),
            ]
            ax.legend(handles=legend_elements, fontsize=9)
            ax.set_xlabel(f"Factor 1 ({explained[0]*100:.1f}%)")
            ax.set_ylabel(f"Factor 2 ({explained[1]*100:.1f}%)")
            ax.set_title(f"{SUBSET_LABELS[subset]} — {config.MODEL_LABEL}")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            figures[f"scatter_{subset}"] = fig_to_b64(fig)
            plt.close(fig)

        # Scree
        n_eig = min(8, len(eigenvalues))
        fig, ax = plt.subplots(figsize=(6, 3))
        x = np.arange(1, n_eig + 1)
        ax.bar(x, eigenvalues[:n_eig],
               color=["#2196F3" if e > 1 else "#ccc" for e in eigenvalues[:n_eig]],
               edgecolor="white")
        ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(f"Scree — {subset}")
        ax.set_xticks(x)
        plt.tight_layout()
        figures[f"scree_{subset}"] = fig_to_b64(fig)
        plt.close(fig)

    # ── Build HTML ──
    html = []
    fig_num = 0

    html.append(build_html_header("Matched Behavioral PCA Report", config.MODEL_LABEL))
    html.append(build_toc(SECTIONS))

    html.append(f"""
<h2 id="methods">1. Methods</h2>
<div class="method">
<strong>Research question:</strong> Does the factor structure of {config.MODEL_LABEL}'s
mental-property judgments depend on which concepts are included? Specifically, do
human-favored concepts, AI-favored concepts, and directional (all signed) subsets
produce the same two-factor Experience/Agency structure, or does the factor structure
change with concept selection?

<p><strong>Data source:</strong> Pairwise comparison responses from {config.MODEL_LABEL},
filtered into concept subsets: human-favored (concepts where human advantage > 0),
AI-favored (human advantage < 0), directional (all concepts with nonzero direction),
and all concepts.</p>

<strong>Procedure:</strong>
<ol>
<li><strong>Concept subsetting:</strong> Concepts are partitioned by their observed
    human-advantage direction from the full behavioral analysis.</li>
<li><strong>PCA per subset:</strong> For each subset, PCA on the correlation matrix of
    concept ratings across characters, with Kaiser criterion and varimax rotation.</li>
<li><strong>Cross-subset comparison:</strong> Number of retained factors, total variance
    explained, and Mann-Whitney U p-values compared across subsets to assess stability
    of the factor structure.</li>
</ol>
</div>
""")

    html.append(expanded_concepts_stimuli_html())

    html.append(f"""
<h2 id="cross-subset">3. Cross-Subset Comparison</h2>
<table>
<tr><th>Subset</th><th>N Concepts</th><th>Factors Retained</th>
<th>F1 Variance</th><th>F1 p-value</th><th>F1 Separation</th>
<th>F2 p-value</th><th>F2 Separation</th></tr>
""")

    for subset in SUBSETS:
        if subset not in subset_data:
            continue
        data = subset_data[subset]
        pca = data["pca"]
        eigenvalues = pca["eigenvalues"]
        explained = pca["explained_var_ratio"]
        cat = data["categorical"]
        n_factors = int(np.sum(eigenvalues > 1.0))
        n_concepts = len(pca["concept_keys"])

        f1_info = cat["factors"][0] if cat.get("factors") else {}
        f2_info = cat["factors"][1] if cat.get("factors") and len(cat["factors"]) > 1 else {}

        f1_p = f1_info.get("p_value", float("nan"))
        f2_p = f2_info.get("p_value", float("nan"))
        f1_sep = f1_info.get("separation", 0)
        f2_sep = f2_info.get("separation", 0)
        f1_p_class = ' class="sig"' if f1_p < 0.05 else ""
        f2_p_class = ' class="sig"' if f2_p < 0.05 else ""

        html.append(
            f'<tr><td>{SUBSET_LABELS[subset]}</td><td>{n_concepts}</td>'
            f'<td>{n_factors}</td><td>{explained[0]*100:.1f}%</td>'
            f'<td{f1_p_class}>{f1_p:.4f}</td><td>{f1_sep:.3f}</td>'
            f'<td{f2_p_class}>{f2_p:.4f}</td><td>{f2_sep:.3f}</td></tr>\n'
        )
    html.append("</table>\n")

    # Per-subset details
    first_subset = True
    for subset in SUBSETS:
        if subset not in subset_data:
            continue
        data = subset_data[subset]
        pca = data["pca"]
        concept_keys = list(pca["concept_keys"])
        rotated = pca["rotated_loadings"]
        eigenvalues = pca["eigenvalues"]
        explained = pca["explained_var_ratio"]
        cat = data["categorical"]
        n_factors = min(int(np.sum(eigenvalues > 1.0)), 4)
        label = SUBSET_LABELS[subset]

        if first_subset:
            html.append(f'\n<h2 id="subset-details">4. {label}</h2>\n')
            first_subset = False
        else:
            html.append(f'\n<h2 id="subset-{subset}">4. {label}</h2>\n')

        if f"scatter_{subset}" in figures:
            fig_num += 1
            html.append("<h3>Factor Space</h3>\n")
            html.append(html_figure(
                figures[f"scatter_{subset}"],
                f"Character positions in Factor 1 vs Factor 2 space for the {label} subset. "
                f"Red squares = AI, blue circles = human.",
                fig_num=fig_num,
                alt=f"Scatter {subset}",
            ))

        fig_num += 1
        html.append("<h3>Scree Plot</h3>\n")
        html.append(html_figure(
            figures[f"scree_{subset}"],
            f"Eigenvalue scree plot for the {label} subset. "
            f"Components above the red line retained.",
            fig_num=fig_num,
            alt=f"Scree {subset}",
        ))

        # Loadings table
        html.append("""
<h3>Varimax-Rotated Loadings</h3>
<table><tr><th>Concept</th>""")
        for fi in range(min(n_factors, 4)):
            html.append(f"<th>F{fi+1}</th>")
        html.append("</tr>\n")
        for c_idx, concept_key in enumerate(concept_keys):
            html.append(f"<tr><td>{concept_key}</td>")
            for fi in range(min(n_factors, 4)):
                val = rotated[c_idx, fi]
                bold = ' style="font-weight:bold"' if abs(val) > 0.4 else ""
                html.append(f"<td{bold}>{val:+.3f}</td>")
            html.append("</tr>\n")
        html.append("</table>\n")

        # Categorical alignment
        if cat.get("factors"):
            html.append("<h3>Categorical Alignment</h3>\n<table>")
            html.append("<tr><th>Factor</th><th>AI Mean</th><th>Human Mean</th>"
                        "<th>Separation</th><th>U</th><th>p</th></tr>\n")
            for finfo in cat["factors"]:
                p_class = ' class="sig"' if finfo["p_value"] < 0.05 else ""
                html.append(
                    f'<tr><td>F{finfo["factor"]}</td>'
                    f'<td>{finfo["ai_mean"]:.3f}</td>'
                    f'<td>{finfo["human_mean"]:.3f}</td>'
                    f'<td>{finfo["separation"]:.3f}</td>'
                    f'<td>{finfo["mann_whitney_u"]:.1f}</td>'
                    f'<td{p_class}>{finfo["p_value"]:.4f}</td></tr>\n'
                )
            html.append("</table>\n")

    html.append(build_html_footer())

    out_dir = str(rdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "matched_behavioral_pca_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate matched behavioral PCA report"
    )
    add_model_argument(parser)
    add_dataset_argument(parser)
    parser.add_argument("--both", action="store_true",
                        help="Generate for both models")
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Matched PCA Report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
