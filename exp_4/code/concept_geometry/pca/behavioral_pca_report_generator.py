#!/usr/bin/env python3
"""
Behavioral PCA Report Generator

Generates an HTML report with full methodology and charts for interpreting
the behavioral PCA results.

Reads from:
    results/{model}/concept_geometry/pca/behavioral/data/
        pairwise_pca_results.npz
        pairwise_character_means.npz
        pairwise_categorical_analysis.json
        pairwise_consistency_stats.json

Writes to:
    results/{model}/concept_geometry/pca/behavioral/{dataset}/
        behavioral_pca_report.html

Usage:
    python concept_geometry/pca/behavioral_pca_report_generator.py --model llama2_13b_chat
    python concept_geometry/pca/behavioral_pca_report_generator.py --model llama2_13b_base
    python concept_geometry/pca/behavioral_pca_report_generator.py --model llama2_13b_chat --dataset reduced_dataset

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir,
)
from concept_geometry.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
)

SECTIONS = [
    {"id": "research-question", "label": "1. Research Question"},
    {"id": "analysis-approach", "label": "2. Analysis Approach"},
    {"id": "consistency", "label": "3. Order Consistency"},
    {"id": "scree", "label": "4. Scree Plot"},
    {"id": "loadings", "label": "5. Factor Loadings"},
    {"id": "scatter", "label": "6. Character Positions"},
    {"id": "group-means", "label": "7. Per-Concept Group Means"},
    {"id": "anomalies", "label": "8. Anomalies"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate behavioral PCA report for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    set_model(model_key)
    is_chat = config.IS_CHAT

    ddir = data_dir("concept_geometry/pca", "behavioral")
    rdir = results_phase_dir("concept_geometry/pca", "behavioral")

    # ── Load data ──
    pca_path = os.path.join(str(ddir), "pairwise_pca_results.npz")
    means_path = os.path.join(str(ddir), "pairwise_character_means.npz")
    cat_path = os.path.join(str(ddir), "pairwise_categorical_analysis.json")
    consistency_path = os.path.join(str(ddir), "pairwise_consistency_stats.json")

    if not os.path.exists(pca_path):
        print(f"PCA results not found at {pca_path} — skipping {model_key}")
        return None

    pca = np.load(pca_path)
    means_data = np.load(means_path)
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
    concept_keys = list(pca["concept_keys"])
    means = means_data["means"]

    n_chars = len(char_keys)
    n_concepts = len(concept_keys)
    n_pairs = n_chars * (n_chars - 1) // 2
    n_factors_retained = int(np.sum(eigenvalues > 1.0))
    n_show = min(n_factors_retained, 4)

    cat_analysis = cat_data["categorical"]
    per_concept = cat_data["per_concept_groups"]

    if is_chat:
        method_str = "text generation with greedy decoding, then parsing the first digit (1-5) from the response"
    else:
        method_str = ("extracting logit probabilities over tokens '1' through '5' at the "
                      "next-token position, then computing the expected rating as the "
                      "probability-weighted mean")

    figures = {}
    fig_num = 0

    # ── 1. Scree plot ──
    fig, ax = plt.subplots(figsize=(7, 4))
    n_eig = min(10, len(eigenvalues))
    n_expl = min(n_eig, len(explained))
    x = np.arange(1, n_eig + 1)
    ax.bar(x, eigenvalues[:n_eig],
           color=["#2196F3" if e > 1 else "#ccc" for e in eigenvalues[:n_eig]],
           edgecolor="white", linewidth=0.5)
    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1,
               label="Kaiser criterion (eigenvalue = 1)")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Scree Plot — {config.MODEL_LABEL}")
    ax.set_xticks(x)
    cum = np.cumsum(explained[:n_expl]) * 100
    for i in range(n_expl):
        ax.text(x[i], eigenvalues[i] + 0.1, f"{cum[i]:.0f}%",
                ha="center", fontsize=8, color="#555")
    ax.legend(fontsize=9)
    plt.tight_layout()
    figures["scree"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 2. Loadings heatmap ──
    fig, ax = plt.subplots(figsize=(3 + n_show * 1.2, max(6, n_concepts * 0.4)))
    loadings_show = rotated[:, :n_show]
    sort_idx = np.argsort(-np.abs(loadings_show[:, 0]))
    loadings_sorted = loadings_show[sort_idx]
    concepts_sorted = [concept_keys[i] for i in sort_idx]

    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(loadings_sorted, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"F{i+1}\n({explained[i]*100:.1f}%)" for i in range(n_show)])
    ax.set_yticks(range(len(concepts_sorted)))
    ax.set_yticklabels(concepts_sorted, fontsize=9)
    for i in range(len(concepts_sorted)):
        for j in range(n_show):
            val = loadings_sorted[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            weight = "bold" if abs(val) > 0.4 else "normal"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)
    ax.set_title(f"Varimax-Rotated Loadings — {config.MODEL_LABEL}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Loading")
    plt.tight_layout()
    figures["loadings_heatmap"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 3. Loadings bar chart per factor ──
    for fi in range(n_show):
        fig, ax = plt.subplots(figsize=(8, max(5, n_concepts * 0.3)))
        vals = rotated[:, fi]
        sort_idx_f = np.argsort(vals)
        sorted_vals = vals[sort_idx_f]
        sorted_names = [concept_keys[i] for i in sort_idx_f]
        colors = ["#E53935" if v < -0.3 else "#1E88E5" if v > 0.3 else "#ccc"
                  for v in sorted_vals]
        ax.barh(range(len(sorted_names)), sorted_vals, color=colors,
                edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=0.4, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.4, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Loading")
        ax.set_title(f"Factor {fi+1} Loadings ({explained[fi]*100:.1f}% var) — {config.MODEL_LABEL}")
        plt.tight_layout()
        figures[f"loadings_f{fi+1}"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 4. Character scatter F1 × F2 ──
    if n_show >= 2:
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

        ai_idx = [char_keys.index(k) for k in char_keys if k in AI_CHARACTERS]
        hu_idx = [char_keys.index(k) for k in char_keys if k in HUMAN_CHARACTERS]
        ai_center = scores_01[ai_idx].mean(axis=0)
        hu_center = scores_01[hu_idx].mean(axis=0)
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

        f1_info = cat_analysis["factors"][0]
        f2_info = cat_analysis["factors"][1]
        ax.set_xlabel(f"Factor 1 ({explained[0]*100:.1f}% var) — "
                      f"U={f1_info['mann_whitney_u']:.0f}, p={f1_info['p_value']:.4f}")
        ax.set_ylabel(f"Factor 2 ({explained[1]*100:.1f}% var) — "
                      f"U={f2_info['mann_whitney_u']:.0f}, p={f2_info['p_value']:.4f}")
        ax.set_title(f"Character Positions in Factor Space — {config.MODEL_LABEL}")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures["scatter"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 5. Per-concept group means ──
    fig, ax = plt.subplots(figsize=(10, max(5, len(per_concept) * 0.35)))
    concepts_pc = [p["concept"] for p in per_concept]
    ai_means = [p["ai_mean"] for p in per_concept]
    hu_means = [p["human_mean"] for p in per_concept]
    diffs = [p["difference"] for p in per_concept]

    sort_idx_d = np.argsort(diffs)
    y = np.arange(len(concepts_pc))
    bar_h = 0.35
    ax.barh(y - bar_h/2, [ai_means[i] for i in sort_idx_d], bar_h,
            label="AI", color="#E53935", alpha=0.8)
    ax.barh(y + bar_h/2, [hu_means[i] for i in sort_idx_d], bar_h,
            label="Human", color="#1E88E5", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([concepts_pc[i] for i in sort_idx_d], fontsize=9)
    ax.set_xlabel("Mean Rating")
    ax.set_title(f"Per-Concept Group Means — {config.MODEL_LABEL}")
    ax.legend(fontsize=9)
    ax.axvline(x=3.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    figures["group_means"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    html_parts = []
    html_parts.append(build_html_header("Behavioral PCA Report", config.MODEL_LABEL))

    html_parts.append(f"""
<div class="stat">
<strong>Summary:</strong> {n_concepts} concept dimensions,
{n_chars} characters ({len(AI_CHARACTERS)} AI, {len(HUMAN_CHARACTERS)} human).
<strong>{n_factors_retained}</strong> factor(s) retained (eigenvalue &gt; 1),
explaining <strong>{np.sum(explained[:n_factors_retained])*100:.1f}%</strong> of variance.
</div>
""")

    html_parts.append(build_toc(SECTIONS))

    html_parts.append(f"""
<h2 id="research-question">1. Research Question</h2>
<p>Does {config.MODEL_LABEL} have a structured representation of mental properties
that differentiates human characters from AI characters? Specifically: when the model
makes explicit pairwise judgments about which of two characters has "more" of a given
mental property, do those judgments organize into interpretable latent factors, and do
those factors separate human from AI characters?</p>

<p>This is a <em>behavioral</em> analysis — it measures what the model <em>says</em>
about characters, not the structure of its internal activations. It answers whether
the model's explicit outputs encode a human/AI distinction along concept dimensions
drawn from the philosophy of mind (phenomenology, agency, beliefs, etc.).</p>

<h2 id="analysis-approach">2. Analysis Approach</h2>
<div class="method">
<strong>Step-by-step procedure:</strong>
<ol>
<li><strong>Characters:</strong> {n_chars} characters are defined — {len(AI_CHARACTERS)} AI
    (e.g., ChatGPT, Siri, a robot) and {len(HUMAN_CHARACTERS)} human
    (e.g., a teacher, a doctor, a child). Each has a name and 1-sentence description.</li>
<li><strong>Concept dimensions:</strong> {n_concepts} mental-property dimensions are drawn
    from Experiment 3's standalone concepts (e.g., phenomenology, agency, beliefs, desires,
    theory of mind). Each dimension has a pairwise survey prompt like
    "which character is more capable of X?"</li>
<li><strong>Pairwise comparisons:</strong> For each concept dimension, all {n_pairs} unique
    character pairs (C({n_chars},2)) are presented to the model. Each pair is presented in
    <em>both orders</em> (A-vs-B and B-vs-A) to counterbalance order effects, yielding
    {n_pairs * 2} trials per concept, {n_pairs * 2 * n_concepts} total.</li>
<li><strong>Rating extraction:</strong> The model produces a rating on a 1-5 scale
    (1 = "much more character A", 5 = "much more character B") via {method_str}.</li>
<li><strong>Order consistency check:</strong> For each pair presented in both orders, perfect
    consistency means the two ratings sum to 6 (e.g., if A-vs-B = 2, then B-vs-A should = 4).
    Deviations from 6 indicate inconsistency.</li>
<li><strong>Character means:</strong> For each (concept, character) cell, the mean rating is
    computed across all pairwise comparisons involving that character. This produces a
    {n_concepts} &times; {n_chars} matrix where each entry is a character's "score" on a concept.</li>
<li><strong>PCA:</strong> Principal Component Analysis on the correlation matrix of concept
    ratings across characters. This finds the latent dimensions along which concept ratings
    co-vary. Factors with eigenvalue &gt; 1 are retained (Kaiser criterion).</li>
<li><strong>Varimax rotation:</strong> Retained factors are varimax-rotated to maximize the
    number of near-zero loadings per factor, making each factor more interpretable. A concept
    with a high loading (|loading| &gt; 0.4) on a factor is considered well-captured by that factor.</li>
<li><strong>Factor scores:</strong> Each character is projected into the rotated factor space.
    Scores are min-max normalized to [0, 1] for visualization.</li>
<li><strong>Categorical alignment:</strong> Mann-Whitney U tests compare factor scores between
    AI and human character groups. A significant test (p &lt; .05) means that factor reliably
    separates the two groups.</li>
<li><strong>Per-concept group means:</strong> For each concept, the mean rating for AI
    characters vs. human characters is compared to identify which properties show the
    largest group differences.</li>
</ol>
</div>

<div class="interpret">
<strong>How to interpret:</strong>
<ul>
<li><strong>Loadings:</strong> Tell you what each factor <em>means</em>. If Factor 1 has
    high loadings on "phenomenology", "emotion", and "pain", it captures a
    "capacity for subjective experience" dimension.</li>
<li><strong>Factor scores:</strong> Tell you where each character <em>sits</em> on each factor.
    If AI characters cluster at one end and human characters at the other, the model has a
    structured human/AI distinction on that dimension.</li>
<li><strong>Group separation (Mann-Whitney U):</strong> A formal test of whether the factor
    meaningfully distinguishes human from AI characters. Small p-values indicate reliable separation.</li>
<li><strong>Anomalies:</strong> Characters that sit closer to the <em>other</em> group's centroid —
    e.g., an AI character that the model rates as very human-like on some factor.</li>
</ul>
</div>
""")

    # Consistency stats
    if consistency_stats.get("n_pairs_both", 0) > 0:
        html_parts.append(f"""
<h2 id="consistency">3. Order Consistency</h2>
<div class="stat">
Pairs with both orders: <strong>{consistency_stats['n_pairs_both']}</strong>.
Perfectly consistent (sum = 6): <strong>{consistency_stats['n_consistent']}</strong>
({consistency_stats['pct_consistent']:.1f}%).
Mean deviation from perfect consistency: <strong>{consistency_stats['mean_deviation']:.3f}</strong>.
</div>
<p>Low consistency does not invalidate the analysis — the character means average over both
orders, canceling systematic order bias. But high inconsistency suggests the model's judgments
are noisy on individual trials.</p>
""")

    # Scree
    fig_num += 1
    html_parts.append(f"""
<h2 id="scree">4. Scree Plot</h2>
<p>Eigenvalues from PCA on the concept correlation matrix. Components above the red line
(eigenvalue &gt; 1, Kaiser criterion) are retained. Cumulative variance percentages shown above
each bar. A steep drop-off followed by a flat "scree" indicates the meaningful factors have been
captured.</p>
{html_figure(figures['scree'],
             'Eigenvalues from PCA on the concept correlation matrix. Components above the red dashed line (eigenvalue &gt; 1, Kaiser criterion) are retained. Cumulative variance shown above bars.',
             fig_num=fig_num, alt="Scree plot")}
""")

    # Loadings heatmap
    fig_num += 1
    html_parts.append(f"""
<h2 id="loadings">5. Factor Loadings (Varimax-Rotated)</h2>
<p>Each cell shows how strongly a concept loads on a factor after varimax rotation.
Loadings close to +1 or -1 mean the concept is strongly associated with that factor;
loadings near 0 mean it is not. Bold values exceed |0.4|. Concepts are sorted by their
absolute loading on Factor 1.</p>
{html_figure(figures['loadings_heatmap'],
             'Varimax-rotated loading matrix. Each cell shows how strongly a concept loads on a factor. Bold values exceed |0.4|. Concepts sorted by absolute loading on Factor 1.',
             fig_num=fig_num, alt="Loadings heatmap")}
""")

    for fi in range(n_show):
        key = f"loadings_f{fi+1}"
        fig_num += 1
        html_parts.append(f"""
<h3>Factor {fi+1} — Sorted Loadings</h3>
{html_figure(figures[key],
             f'All concepts sorted by their loading on Factor {fi+1}. Blue (&gt;0.3) and red (&lt;-0.3) highlight defining concepts. Dotted lines at &plusmn;0.4 mark conventional threshold.',
             fig_num=fig_num, alt=f"Factor {fi+1} loadings")}
""")

    # Character scatter
    if "scatter" in figures:
        fig_num += 1
        html_parts.append(f"""
<h2 id="scatter">6. Character Positions in Factor Space</h2>
<p>Each character plotted by its normalized factor score (0-1 scale) on Factor 1 (x-axis)
vs Factor 2 (y-axis). Red squares = AI characters, blue circles = human characters.
Large X markers show each group's centroid (mean position). If the groups are well-separated,
AI and human characters occupy distinct regions of the factor space.</p>
{html_figure(figures['scatter'],
             'Character positions in normalized factor space (0-1). Red squares = AI, blue circles = human. Large X markers = group centroids.',
             fig_num=fig_num, alt="Character scatter")}
""")

        html_parts.append("""
<h3>Group Separation Statistics</h3>
<p>Mann-Whitney U test for each factor: tests whether AI and human character groups have
significantly different factor scores. Separation = absolute difference in group means.
Red p-values are significant at p &lt; .05.</p>
<table>
<tr><th>Factor</th><th>AI mean</th><th>Human mean</th><th>Separation</th>
<th>Mann-Whitney U</th><th>p-value</th></tr>
""")
        for finfo in cat_analysis["factors"]:
            p_class = ' class="sig"' if finfo["p_value"] < 0.05 else ""
            html_parts.append(
                f'<tr><td>F{finfo["factor"]}</td>'
                f'<td>{finfo["ai_mean"]:.3f}</td>'
                f'<td>{finfo["human_mean"]:.3f}</td>'
                f'<td>{finfo["separation"]:.3f}</td>'
                f'<td>{finfo["mann_whitney_u"]:.1f}</td>'
                f'<td{p_class}>{finfo["p_value"]:.4f}</td></tr>\n'
            )
        html_parts.append("</table>\n")

    # Group means
    fig_num += 1
    html_parts.append(f"""
<h2 id="group-means">7. Per-Concept Group Means</h2>
<p>For each concept dimension, the mean pairwise rating for AI characters (red) vs human characters
(blue). A rating of 3.0 (dotted line) means "both equally". Concepts where the bars diverge most
are the ones where the model most strongly differentiates human from AI characters in its explicit
judgments.</p>
{html_figure(figures['group_means'],
             'Per-concept mean pairwise ratings for AI (red) vs human (blue) character groups. Dotted line at 3.0 = equal. Concepts sorted by difference.',
             fig_num=fig_num, alt="Group means")}
""")

    # Anomalies
    anomalies = cat_analysis.get("anomalies", [])
    if anomalies:
        html_parts.append("""
<h2 id="anomalies">8. Anomalies</h2>
<p>Characters whose factor score is closer to the <em>other</em> group's mean than their
own group's mean. These are characters that "cross over" — e.g., an AI character that the
model treats as human-like on a given factor, or vice versa.</p>
<table>
<tr><th>Character</th><th>Type</th><th>Factor</th><th>Score</th>
<th>Own group mean</th><th>Other group mean</th></tr>
""")
        for a in anomalies:
            html_parts.append(
                f'<tr><td>{a["character"]}</td><td>{a["type"]}</td>'
                f'<td>F{a["factor"]}</td><td>{a["score"]:.3f}</td>'
                f'<td>{a["own_group_mean"]:.3f}</td>'
                f'<td>{a["other_group_mean"]:.3f}</td></tr>\n'
            )
        html_parts.append("</table>\n")

    html_parts.append(build_html_footer())

    # Write
    out_dir = os.path.join(str(rdir), dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "behavioral_pca_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html_parts))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate behavioral PCA report with charts"
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
