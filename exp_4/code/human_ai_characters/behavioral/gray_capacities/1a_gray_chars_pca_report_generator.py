#!/usr/bin/env python3
"""
Gray-with-Characters PCA Report Generator

Reads PCA results from behavior/gray_characters/ and generates an HTML
report showing:
    - PCA scatterplot (Factor 1 vs Factor 2, colored by AI/human type)
    - Varimax loadings per capacity
    - Categorical alignment statistics
    - Per-character factor scores table
    - Comparison with original Gray et al. factor structure

Reads from:
    data_dir("human_ai_characters", "behavioral/gray_capacities")/
        pairwise_pca_results.npz
        pairwise_categorical_analysis.json
        pairwise_consistency_stats.json

Writes to:
    results_dir("human_ai_characters", "behavioral/gray_capacities")/{dataset}/
        gray_chars_pca_report.html

Usage:
    python behavior/4a_gray_chars_pca_report_generator.py --model llama2_13b_base
    python behavior/4a_gray_chars_pca_report_generator.py --model llama2_13b_chat --both

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir,
)
from entities.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from entities.gray_entities import CAPACITY_PROMPTS
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
    characters_stimuli_html,
)

# ── Table of contents ──
SECTIONS = [
    {"id": "methods", "label": "1. Methods"},
    {"id": "stimuli", "label": "2. Stimuli"},
    {"id": "scree", "label": "3. Scree Plot"},
    {"id": "scatter", "label": "4. Character Positions"},
    {"id": "loadings", "label": "5. Capacity Loadings"},
    {"id": "alignment", "label": "6. Categorical Alignment"},
    {"id": "scores", "label": "7. Character Factor Scores"},
    {"id": "consistency", "label": "8. Order Consistency"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate Gray-with-characters PCA report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    set_model(model_key)
    ddir = data_dir("human_ai_characters", "behavioral/gray_capacities")
    rdir = results_dir("human_ai_characters", "behavioral/gray_capacities")

    pca_path = os.path.join(str(ddir), "pairwise_pca_results.npz")
    if not os.path.exists(pca_path):
        print(f"PCA results not found at {pca_path} — skipping {model_key}")
        return None

    pca = np.load(pca_path)
    cat_path = os.path.join(str(ddir), "pairwise_categorical_analysis.json")
    consistency_path = os.path.join(str(ddir), "pairwise_consistency_stats.json")

    cat_data = {}
    if os.path.exists(cat_path):
        with open(cat_path) as f:
            cat_data = json.load(f)
    consistency_stats = {}
    if os.path.exists(consistency_path):
        with open(consistency_path) as f:
            consistency_stats = json.load(f)

    rotated = pca["rotated_loadings"]
    scores_01 = pca["factor_scores_01"]
    eigenvalues = pca["eigenvalues"]
    explained = pca["explained_var_ratio"]
    char_keys = list(pca["character_keys"])
    capacity_keys = list(pca["capacity_keys"])

    n_chars = len(char_keys)
    n_caps = len(capacity_keys)
    n_factors_retained = int(np.sum(eigenvalues > 1.0))
    cat_analysis = cat_data.get("categorical", {})
    fig_num = 1

    figures = {}

    # ── 1. Scree plot ──
    fig, ax = plt.subplots(figsize=(7, 4))
    n_eig = min(10, len(eigenvalues))
    x = np.arange(1, n_eig + 1)
    ax.bar(x, eigenvalues[:n_eig],
           color=["#2196F3" if e > 1 else "#ccc" for e in eigenvalues[:n_eig]],
           edgecolor="white")
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1,
               label="Kaiser criterion")
    cum = np.cumsum(explained[:min(n_eig, len(explained))]) * 100
    for i in range(min(n_eig, len(explained))):
        ax.text(x[i], eigenvalues[i] + 0.1, f"{cum[i]:.0f}%",
                ha="center", fontsize=8, color="#555")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Scree Plot — Gray Characters — {config.MODEL_LABEL}")
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    plt.tight_layout()
    figures["scree"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 2. Character scatter F1 x F2 ──
    if scores_01.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for char_key in char_keys:
            idx = char_keys.index(char_key)
            is_ai = char_key in AI_CHARACTERS
            color = "#E53935" if is_ai else "#1E88E5"
            marker = "s" if is_ai else "o"
            ax.scatter(scores_01[idx, 0], scores_01[idx, 1],
                       c=color, marker=marker, s=80, edgecolors="white",
                       linewidth=0.5, zorder=3)
            ax.annotate(CHARACTER_NAMES[char_key],
                        (scores_01[idx, 0], scores_01[idx, 1]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(4, 4), textcoords="offset points",
                        color=color, alpha=0.85)

        legend_elements = [
            Line2D([0], [0], marker="s", color="w", markerfacecolor="#E53935",
                   markersize=10, label="AI characters"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1E88E5",
                   markersize=10, label="Human characters"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=9)
        ax.set_xlabel(f"Factor 1 ({explained[0]*100:.1f}%)")
        ax.set_ylabel(f"Factor 2 ({explained[1]*100:.1f}%)")
        ax.set_title(f"Character Positions — Gray Capacities — {config.MODEL_LABEL}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures["scatter"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 3. Loadings bar chart ──
    n_show = min(n_factors_retained, 4)
    fig, ax = plt.subplots(figsize=(3 + n_show * 1.2, max(6, n_caps * 0.4)))
    loadings_show = rotated[:, :n_show]
    sort_idx = np.argsort(-np.abs(loadings_show[:, 0]))
    loadings_sorted = loadings_show[sort_idx]
    caps_sorted = [capacity_keys[i] for i in sort_idx]

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(loadings_sorted, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"F{i+1}\n({explained[i]*100:.1f}%)" for i in range(n_show)])
    ax.set_yticks(range(len(caps_sorted)))
    ax.set_yticklabels(caps_sorted, fontsize=9)
    for i in range(len(caps_sorted)):
        for j in range(n_show):
            val = loadings_sorted[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            weight = "bold" if abs(val) > 0.4 else "normal"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)
    ax.set_title(f"Capacity Loadings — {config.MODEL_LABEL}")
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    figures["loadings"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    html = []
    html.append(build_html_header(
        "Gray Replication with AI/Human Characters", config.MODEL_LABEL))

    html.append(f"""
<div class="stat">
<strong>Summary:</strong> {n_caps} mental capacities (Gray et al. 2007),
{n_chars} characters ({len(AI_CHARACTERS)} AI, {len(HUMAN_CHARACTERS)} human).
<strong>{n_factors_retained}</strong> factor(s) retained,
explaining <strong>{np.sum(explained[:n_factors_retained])*100:.1f}%</strong> of variance.
</div>
""")

    html.append(build_toc(SECTIONS))

    # ── Methods ──
    html.append(f"""
<h2 id="methods">1. Methods</h2>
<div class="method">
<strong>Research question:</strong> Does {config.MODEL_LABEL}'s folk psychology produce
the two-factor Experience/Agency structure described by Gray et al. (2007) when
the entities are {n_chars} real-world AI systems and human characters (instead of Gray's
original 13 entities)? Do the factors separate AI from human groups?

<p><strong>Data source:</strong> Pairwise comparison responses from {config.MODEL_LABEL}.
For every pair of characters and each of {n_caps} Gray et al. mental capacities,
the model rated which character possesses more of that capacity on a 1&ndash;5 scale.
Each pair was presented in both orders (A-vs-B and B-vs-A) to control for position bias.</p>

<strong>Procedure:</strong>
<ol>
<li><strong>Relative scoring:</strong> Each pairwise rating R (1&ndash;5) is converted to
    relative scores: entity A gets (3&nbsp;&minus;&nbsp;R), entity B gets (R&nbsp;&minus;&nbsp;3).
    A rating of 3 (equal) contributes 0 to both.</li>
<li><strong>Character means matrix:</strong> For each (capacity, character) cell, scores are
    averaged across all comparisons involving that character, yielding an {n_caps}&nbsp;&times;&nbsp;{n_chars} matrix.</li>
<li><strong>PCA on correlation matrix:</strong> Principal Component Analysis is applied to the
    {n_caps}&nbsp;&times;&nbsp;{n_caps} correlation matrix of capacities across characters, following
    Gray et al.&rsquo;s methodology exactly.</li>
<li><strong>Kaiser criterion:</strong> Factors with eigenvalue &gt; 1 are retained
    ({n_factors_retained} retained here).</li>
<li><strong>Varimax rotation:</strong> Retained factors are varimax-rotated to maximize
    interpretability (simple structure). Loadings &gt; |0.4| are considered meaningful.</li>
<li><strong>Factor scores:</strong> Characters are projected into rotated factor space via the
    regression method, then min-max normalized to [0,&nbsp;1].</li>
<li><strong>Categorical alignment:</strong> Mann-Whitney U tests compare factor scores between
    AI and human character groups. Significant p &lt; .05 means that factor reliably separates groups.</li>
</ol>
</div>
""")

    # ── Stimuli ──
    html.append(characters_stimuli_html(include_capacities=True))

    # ── Scree ──
    html.append(f'<h2 id="scree">3. Scree Plot</h2>\n')
    html.append(html_figure(
        figures['scree'],
        f"Eigenvalues from PCA on the {n_caps}-capacity correlation matrix. "
        f"Components above the red dashed line (eigenvalue &gt; 1, Kaiser criterion) are "
        f"retained. Cumulative variance percentages shown above each bar. "
        f"{n_factors_retained} component(s) retained, explaining "
        f"{np.sum(explained[:n_factors_retained])*100:.1f}% of total variance.",
        fig_num=fig_num, alt="Scree plot"))
    fig_num += 1

    # ── Scatter ──
    if "scatter" in figures:
        html.append(f'<h2 id="scatter">4. Character Positions in Factor Space</h2>\n')
        html.append(html_figure(
            figures['scatter'],
            "Each character plotted by Factor 1 vs Factor 2. Red squares = AI, "
            "blue circles = human. Separation along either axis indicates that "
            "factor differentiates AI from human characters in the model's "
            "folk-psychological judgments.",
            fig_num=fig_num, alt="Character scatter"))
        fig_num += 1

    # ── Loadings ──
    html.append(f'<h2 id="loadings">5. Capacity Loadings (Varimax-Rotated)</h2>\n')
    html.append(f"""<p>Compare to Gray et al.: Experience capacities should load on one factor,
Agency capacities on the other.</p>\n""")
    html.append(html_figure(
        figures['loadings'],
        "Varimax-rotated loading matrix. Each cell shows how strongly a mental capacity "
        "loads on a given factor. Bold values exceed |0.4|. Capacities sorted by absolute "
        "loading on Factor 1. If the model replicates Gray et al., Experience capacities "
        "(e.g., hunger, pain, pleasure) should cluster on one factor, and Agency capacities "
        "(e.g., self-control, planning, thought) on the other.",
        fig_num=fig_num, alt="Loadings heatmap"))
    fig_num += 1

    # Loadings table with Gray factor labels
    html.append("""
<h3>Loadings Table</h3>
<table><tr><th>Capacity</th><th>Gray Factor</th>""")
    for fi in range(min(n_factors_retained, 4)):
        html.append(f"<th>F{fi+1}</th>")
    html.append("</tr>\n")
    for c_idx, cap_key in enumerate(capacity_keys):
        _, gray_factor = CAPACITY_PROMPTS[cap_key]
        html.append(f"<tr><td>{cap_key}</td><td>{gray_factor}</td>")
        for fi in range(min(n_factors_retained, 4)):
            val = rotated[c_idx, fi]
            bold = ' style="font-weight:bold"' if abs(val) > 0.4 else ""
            html.append(f"<td{bold}>{val:+.3f}</td>")
        html.append("</tr>\n")
    html.append("</table>\n")

    # ── Categorical alignment ──
    if cat_analysis.get("factors"):
        html.append(f"""
<h2 id="alignment">6. Categorical Alignment (AI vs Human)</h2>
<table><tr><th>Factor</th><th>AI Mean</th><th>Human Mean</th>
<th>Separation</th><th>Mann-Whitney U</th><th>p-value</th></tr>
""")
        for finfo in cat_analysis["factors"]:
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

    # ── Character factor scores table ──
    html.append(f"""
<h2 id="scores">7. Character Factor Scores</h2>
<table><tr><th>Character</th><th>Type</th>""")
    for fi in range(min(n_factors_retained, 2)):
        html.append(f"<th>F{fi+1}</th>")
    html.append("</tr>\n")
    for idx, char_key in enumerate(char_keys):
        html.append(f"<tr><td>{CHARACTER_NAMES[char_key]}</td>"
                     f"<td>{CHARACTER_TYPES[char_key]}</td>")
        for fi in range(min(n_factors_retained, 2)):
            html.append(f"<td>{scores_01[idx, fi]:.3f}</td>")
        html.append("</tr>\n")
    html.append("</table>\n")

    # ── Consistency stats ──
    if consistency_stats.get("n_pairs_both", 0) > 0:
        html.append(f"""
<h2 id="consistency">8. Order Consistency</h2>
<div class="stat">
Pairs with both orders: <strong>{consistency_stats['n_pairs_both']}</strong>.
Consistent: <strong>{consistency_stats['n_consistent']}</strong>
({consistency_stats['pct_consistent']:.1f}%).
Mean deviation: <strong>{consistency_stats['mean_deviation']:.3f}</strong>.
</div>
""")

    html.append(build_html_footer())

    out_dir = str(rdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gray_chars_pca_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gray-with-characters PCA report"
    )
    add_model_argument(parser)
    add_dataset_argument(parser)
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Gray Characters PCA Report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
