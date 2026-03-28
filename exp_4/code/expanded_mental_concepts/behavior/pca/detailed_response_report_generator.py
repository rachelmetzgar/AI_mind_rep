#!/usr/bin/env python3
"""
Detailed Pairwise Response Report Generator

The main "how is each model actually responding" report. Reads
pairwise_raw_responses.json from the behavioral PCA and produces
a comprehensive HTML breakdown.

Sections:
    1. Methods
    2. Overall by human vs AI group (mean human-advantage per concept)
    3. Per character name breakdown (sorted bar charts)
    4. Per question x human/AI group heatmap
    5. Per question x character name full heatmap
    6. Raw response distribution (rating histograms)
    7. Position bias analysis

Reads from:
    results/{model}/expanded_mental_concepts/behavior/pca/data/
        pairwise_raw_responses.json

Writes to:
    results/{model}/expanded_mental_concepts/behavior/pca/{dataset}/
        detailed_response_report.html

Usage:
    python expanded_mental_concepts/behavior/pca/detailed_response_report_generator.py --model llama2_13b_base
    python expanded_mental_concepts/behavior/pca/detailed_response_report_generator.py --model llama2_13b_chat --both
    python expanded_mental_concepts/behavior/pca/detailed_response_report_generator.py --model llama2_13b_chat --dataset reduced_dataset

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir,
)
from entities.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from expanded_mental_concepts.concepts import (
    CONCEPT_DIMENSIONS, CONCEPT_NAMES, CONCEPT_DIRECTION,
)
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
    expanded_concepts_stimuli_html,
)

SECTIONS = [
    {"id": "methods", "label": "1. Methods"},
    {"id": "stimuli", "label": "2. Stimuli"},
    {"id": "human-advantage", "label": "3. Human Advantage per Concept"},
    {"id": "per-character", "label": "4. Per-Character Scores"},
    {"id": "concept-group", "label": "5. Concept x Group Means"},
    {"id": "heatmap", "label": "6. Concept x Character Heatmap"},
    {"id": "distribution", "label": "7. Rating Distribution"},
    {"id": "position-bias", "label": "8. Position Bias"},
]

DIRECTION_COLORS = {
    "human": "#1E88E5",
    "ai": "#E53935",
    "ambiguous": "#9E9E9E",
}


def compute_scores(responses, is_chat):
    """
    Compute human-advantage scores and per-character relative scores.
    """
    rating_key = "rating" if is_chat else "expected_rating"

    concept_scores = defaultdict(list)
    char_concept_scores = defaultdict(lambda: defaultdict(list))
    char_all_scores = defaultdict(list)
    position_first = []
    position_second = []

    for resp in responses:
        ea, eb = resp["entity_a"], resp["entity_b"]
        ta, tb = CHARACTER_TYPES.get(ea), CHARACTER_TYPES.get(eb)
        rating = resp.get(rating_key)
        if rating is None:
            continue

        cap = resp["capacity"]

        # Relative scores
        score_a = 3 - rating
        score_b = rating - 3

        # Per-character scores (all pairs, not just cross-type)
        char_concept_scores[cap][ea].append(score_a)
        char_concept_scores[cap][eb].append(score_b)
        char_all_scores[ea].append(score_a)
        char_all_scores[eb].append(score_b)

        # Position bias
        position_first.append(score_a)
        position_second.append(score_b)

        # Cross-type only for human advantage
        if ta != tb:
            if ta == "human":
                concept_scores[cap].append(score_a)
            else:
                concept_scores[cap].append(score_b)

    return (dict(concept_scores), char_concept_scores,
            dict(char_all_scores), position_first, position_second)


def generate_report(model_key, dataset="full_dataset"):
    """Generate detailed response report for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.stats import ttest_1samp, mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    set_model(model_key)
    is_chat = config.IS_CHAT

    ddir = data_dir("expanded_mental_concepts", "behavior", "pca")
    rdir = results_dir("expanded_mental_concepts", "behavior", "pca")

    raw_path = os.path.join(str(ddir), "pairwise_raw_responses.json")
    if not os.path.exists(raw_path):
        print(f"Raw responses not found at {raw_path} — skipping {model_key}")
        return None

    with open(raw_path) as f:
        responses = json.load(f)

    print(f"Loaded {len(responses)} responses for {config.MODEL_LABEL}")

    # Compute scores
    (concept_scores, char_concept_scores, char_all_scores,
     position_first, position_second) = compute_scores(responses, is_chat)

    capacities = sorted(concept_scores.keys())
    concept_name_map = {}
    for cap in capacities:
        if cap in CONCEPT_DIMENSIONS:
            concept_name_map[cap] = CONCEPT_DIMENSIONS[cap]["name"]
        else:
            concept_name_map[cap] = cap.replace("_", " ").title()

    figures = {}
    fig_num = 0

    # ── 1. Human advantage per concept ──
    test_results = {}
    p_values = []
    cap_order = []
    for cap in capacities:
        scores = np.array(concept_scores[cap])
        n = len(scores)
        mean_adv = float(np.mean(scores))
        std_adv = float(np.std(scores, ddof=1)) if n > 1 else 0.0

        if n >= 2 and std_adv > 0:
            t_stat, p_val = ttest_1samp(scores, 0)
            test_results[cap] = {
                "n": n, "mean": mean_adv, "std": std_adv,
                "t_stat": float(t_stat), "p_value": float(p_val),
            }
            p_values.append(p_val)
            cap_order.append(cap)
        else:
            test_results[cap] = {
                "n": n, "mean": mean_adv, "std": std_adv,
                "t_stat": float("nan"), "p_value": float("nan"),
                "p_fdr": float("nan"),
            }

    if p_values:
        _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
        for cap, pf in zip(cap_order, p_fdr):
            test_results[cap]["p_fdr"] = float(pf)
    for cap in test_results:
        if "p_fdr" not in test_results[cap]:
            test_results[cap]["p_fdr"] = float("nan")

    sorted_caps = sorted(capacities, key=lambda c: test_results[c]["mean"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_caps) * 0.35)))
    y_pos = np.arange(len(sorted_caps))
    means = [test_results[c]["mean"] for c in sorted_caps]
    colors = [DIRECTION_COLORS.get(CONCEPT_DIRECTION.get(c, "ambiguous"), "#9E9E9E")
              for c in sorted_caps]

    ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    for i, cap in enumerate(sorted_caps):
        p_fdr = test_results[cap]["p_fdr"]
        if not np.isnan(p_fdr) and p_fdr < 0.05:
            marker = "***" if p_fdr < 0.001 else "**" if p_fdr < 0.01 else "*"
            m = means[i]
            offset = 0.002 if m >= 0 else -0.002
            ha = "left" if m >= 0 else "right"
            ax.text(m + offset, i, marker, va="center", ha=ha, fontsize=10,
                    fontweight="bold", color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([concept_name_map.get(c, c) for c in sorted_caps], fontsize=9)
    ax.set_xlabel("Mean Human Advantage (positive = human favored)")
    ax.set_title(f"Human Advantage per Concept — {config.MODEL_LABEL}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DIRECTION_COLORS["human"], alpha=0.8, label="Expected: human"),
        Patch(facecolor=DIRECTION_COLORS["ai"], alpha=0.8, label="Expected: AI"),
        Patch(facecolor=DIRECTION_COLORS["ambiguous"], alpha=0.8, label="Ambiguous"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    figures["human_advantage"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 2. Per-character scores ──
    for group_name, group_chars, color in [
        ("human", HUMAN_CHARACTERS, "#1E88E5"),
        ("ai", AI_CHARACTERS, "#E53935"),
    ]:
        char_data = []
        for c in group_chars:
            if c in char_all_scores:
                char_data.append((c, float(np.mean(char_all_scores[c]))))
        char_data.sort(key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(8, max(4, len(char_data) * 0.35)))
        y_pos = np.arange(len(char_data))
        vals = [d[1] for d in char_data]
        ax.barh(y_pos, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CHARACTER_NAMES[d[0]] for d in char_data], fontsize=9)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Mean Relative Score (positive = favored in comparisons)")
        ax.set_title(f"{group_name.upper()} Characters — {config.MODEL_LABEL}")
        plt.tight_layout()
        figures[f"{group_name}_chars"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 3. Concept x group heatmap ──
    concept_order = sorted(capacities, key=lambda c: (
        {"human": 0, "ai": 1, "ambiguous": 2}.get(
            CONCEPT_DIRECTION.get(c, "ambiguous"), 2),
        -test_results[c]["mean"]
    ))

    group_means = np.zeros((len(concept_order), 3))  # AI mean, Human mean, diff
    for i, cap in enumerate(concept_order):
        ai_scores_cap = []
        hu_scores_cap = []
        for ch in AI_CHARACTERS:
            if cap in char_concept_scores and ch in char_concept_scores[cap]:
                ai_scores_cap.extend(char_concept_scores[cap][ch])
        for ch in HUMAN_CHARACTERS:
            if cap in char_concept_scores and ch in char_concept_scores[cap]:
                hu_scores_cap.extend(char_concept_scores[cap][ch])
        group_means[i, 0] = np.mean(ai_scores_cap) if ai_scores_cap else 0
        group_means[i, 1] = np.mean(hu_scores_cap) if hu_scores_cap else 0
        group_means[i, 2] = group_means[i, 1] - group_means[i, 0]

    fig, ax = plt.subplots(figsize=(8, max(6, len(concept_order) * 0.4)))
    vmax = max(abs(np.min(group_means)), abs(np.max(group_means)))
    if vmax == 0:
        vmax = 0.1
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(group_means, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["AI Group\nMean", "Human Group\nMean", "Difference\n(H-AI)"])
    ax.set_yticks(range(len(concept_order)))
    ax.set_yticklabels([concept_name_map.get(c, c) for c in concept_order], fontsize=9)
    for i in range(len(concept_order)):
        for j in range(3):
            val = group_means[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean Score")
    ax.set_title(f"Group Mean Scores — {config.MODEL_LABEL}")
    plt.tight_layout()
    figures["group_heatmap"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 4. Full concept x character heatmap ──
    char_order = AI_CHARACTERS + HUMAN_CHARACTERS
    heatmap = np.full((len(concept_order), len(char_order)), np.nan)
    for i, cap in enumerate(concept_order):
        for j, ch in enumerate(char_order):
            if cap in char_concept_scores and ch in char_concept_scores[cap]:
                vals = char_concept_scores[cap][ch]
                if vals:
                    heatmap[i, j] = np.mean(vals)

    vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)))
    if vmax == 0:
        vmax = 0.1

    fig, ax = plt.subplots(figsize=(max(14, len(char_order) * 0.5),
                                     max(6, len(concept_order) * 0.4)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(heatmap, cmap=plt.cm.RdBu, norm=norm, aspect="auto")
    char_labels = [CHARACTER_NAMES[c] for c in char_order]
    ax.set_xticks(range(len(char_order)))
    ax.set_xticklabels(char_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(concept_order)))
    ax.set_yticklabels([concept_name_map.get(c, c) for c in concept_order], fontsize=9)
    sep_x = len(AI_CHARACTERS) - 0.5
    ax.axvline(x=sep_x, color="black", linewidth=2)
    ax.text(len(AI_CHARACTERS) / 2 - 0.5, -1.5, "AI Characters",
            ha="center", fontsize=10, fontweight="bold", color="#E53935")
    ax.text(len(AI_CHARACTERS) + len(HUMAN_CHARACTERS) / 2 - 0.5, -1.5,
            "Human Characters", ha="center", fontsize=10, fontweight="bold",
            color="#1E88E5")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean Relative Score")
    ax.set_title(f"Concept x Character — {config.MODEL_LABEL}", pad=25)
    plt.tight_layout()
    figures["full_heatmap"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 5. Rating distribution ──
    if is_chat:
        ratings = [r["rating"] for r in responses if r.get("rating") is not None]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(ratings, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                color="#2196F3", edgecolor="white", rwidth=0.8)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_title(f"Rating Distribution — {config.MODEL_LABEL}")
        plt.tight_layout()
        figures["rating_dist"] = fig_to_b64(fig)
        plt.close(fig)
    else:
        expected_ratings = [r["expected_rating"] for r in responses]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(expected_ratings, bins=50, color="#2196F3", edgecolor="white")
        ax.set_xlabel("Expected Rating")
        ax.set_ylabel("Count")
        ax.set_title(f"Expected Rating Distribution — {config.MODEL_LABEL}")
        plt.tight_layout()
        figures["rating_dist"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 6. Position bias ──
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(position_first, bins=50, alpha=0.6, color="#E53935",
            label=f"First-listed (mean={np.mean(position_first):.4f})")
    ax.hist(position_second, bins=50, alpha=0.6, color="#1E88E5",
            label=f"Second-listed (mean={np.mean(position_second):.4f})")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Relative Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Position Bias — {config.MODEL_LABEL}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    figures["position_bias"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    html = []
    html.append(build_html_header("Detailed Pairwise Response Report", config.MODEL_LABEL))

    html.append(f"""
<div class="stat">
<strong>Summary:</strong> {len(responses)} total pairwise comparisons,
{len(capacities)} concept dimensions,
{len(ALL_CHARACTERS)} characters ({len(AI_CHARACTERS)} AI, {len(HUMAN_CHARACTERS)} human).
</div>
""")

    html.append(build_toc(SECTIONS))

    html.append(f"""
<h2 id="methods">1. Methods</h2>
<div class="method">
<strong>Research question:</strong> How does {config.MODEL_LABEL} rate AI vs human characters on
each mental-property concept? Which concepts show the largest human advantages, and how
consistent are ratings across characters and positions?

<p><strong>Data source:</strong> pairwise_raw_responses.json — pairwise comparison responses
from {config.MODEL_LABEL}. Each trial presents two characters and one concept, asking which
character possesses more of that property (1-5 scale). Each pair is tested in both
presentation orders (A-vs-B and B-vs-A).</p>

<strong>Procedure:</strong>
<ol>
<li><strong>Relative scoring:</strong> For each comparison with rating R: entity A gets
    (3 - R), entity B gets (R - 3).</li>
<li><strong>Cross-type filtering:</strong> For human-advantage analysis, only pairs where
    one character is AI and the other is human.</li>
<li><strong>Per-concept aggregation:</strong> Mean human advantage across all cross-type
    comparisons per concept.</li>
<li><strong>Per-character aggregation:</strong> Mean relative score per character across
    all concepts and comparisons.</li>
<li><strong>Statistical tests:</strong> One-sample t-tests (H0: human advantage = 0) per
    concept, FDR correction (Benjamini-Hochberg). Mann-Whitney U per concept.</li>
<li><strong>Position bias:</strong> For each pair tested in both orders, compare ratings
    to check for systematic position effects (perfect consistency: sum = 6).</li>
</ol>
</div>
""")

    html.append(expanded_concepts_stimuli_html())

    fig_num = 1
    html.append(f"""
<h2 id="human-advantage">3. Human Advantage per Concept (Cross-Type Pairs)</h2>
<p>For each concept, only cross-type pairs (human vs AI) are included. The human
character's relative score is plotted. Positive = human favored, negative = AI favored.
Stars indicate FDR-corrected significance.</p>
{html_figure(figures['human_advantage'],
             "Mean human advantage per concept across all cross-type pairwise comparisons. "
             "Bar color indicates the expected direction (blue = human-associated, "
             "red = AI-associated, gray = ambiguous). Stars denote FDR-corrected "
             "significance (* p < .05, ** p < .01, *** p < .001).",
             fig_num=fig_num, alt="Human advantage per concept")}
""")

    fig_num += 1
    html.append("""
<h3>Full Results Table</h3>
<table>
<tr><th>Concept</th><th>Direction</th><th>N</th><th>Mean Adv.</th>
<th>Std</th><th>t</th><th>p (FDR)</th></tr>
""")

    for cap in sorted(capacities, key=lambda c: -test_results[c]["mean"]):
        r = test_results[cap]
        direction = CONCEPT_DIRECTION.get(cap, "ambiguous")
        p_class = ' class="sig"' if (not np.isnan(r["p_fdr"]) and r["p_fdr"] < 0.05) else ""
        pf_str = f'{r["p_fdr"]:.2e}' if not np.isnan(r["p_fdr"]) else "—"
        t_str = f'{r["t_stat"]:.2f}' if not np.isnan(r["t_stat"]) else "—"
        html.append(
            f'<tr><td>{concept_name_map.get(cap, cap)}</td><td>{direction}</td>'
            f'<td>{r["n"]}</td><td{p_class}>{r["mean"]:+.4f}</td>'
            f'<td>{r["std"]:.4f}</td><td>{t_str}</td><td>{pf_str}</td></tr>\n'
        )
    html.append("</table>\n")

    html.append(f"""
<h2 id="per-character">4. Per-Character Relative Scores</h2>
<p>Mean relative score across all comparisons (all concepts, all opponents).
Positive = this character is generally favored in comparisons.</p>

<h3>Human Characters</h3>
{html_figure(figures['human_chars'],
             "Mean relative score for each human character across all pairwise comparisons. "
             "Positive values indicate the character is generally favored over opponents.",
             fig_num=fig_num, alt="Human character scores")}
""")

    fig_num += 1
    html.append(f"""
<h3>AI Characters</h3>
{html_figure(figures['ai_chars'],
             "Mean relative score for each AI character across all pairwise comparisons. "
             "Positive values indicate the character is generally favored over opponents.",
             fig_num=fig_num, alt="AI character scores")}
""")

    fig_num += 1
    html.append(f"""
<h2 id="concept-group">5. Concept x Group Means</h2>
<p>Mean relative score per concept for AI group vs human group, plus the difference.
Shows which concepts most differentiate the groups.</p>
{html_figure(figures['group_heatmap'],
             "Heatmap of mean relative scores per concept, broken down by character group "
             "(AI vs human). The difference column (H-AI) highlights concepts that most "
             "differentiate between groups. Red = positive, blue = negative.",
             fig_num=fig_num, alt="Concept x group heatmap")}
""")

    fig_num += 1
    html.append(f"""
<h2 id="heatmap">6. Full Concept x Character Heatmap</h2>
<p>Each cell: mean relative score for that character on that concept across all
pairwise comparisons. Red = this character is favored, blue = disfavored.
AI characters on the left, human on the right.</p>
{html_figure(figures['full_heatmap'],
             "Full concept-by-character heatmap of mean relative scores. AI characters are "
             "shown on the left (separated by vertical line) and human characters on the right. "
             "Red cells indicate the character is favored for that concept; blue cells indicate "
             "the character is disfavored.",
             fig_num=fig_num, alt="Concept x character heatmap")}
""")

    fig_num += 1
    html.append(f"""
<h2 id="distribution">7. Rating Distribution</h2>
<p>Distribution of {'discrete ratings (1-5)' if is_chat else 'expected ratings (continuous)'} across
all comparisons.</p>
{html_figure(figures['rating_dist'],
             "Histogram of {'discrete pairwise ratings (1-5 scale)' if is_chat else 'continuous expected ratings'} "
             "across all comparisons. A uniform distribution would indicate no systematic preference; "
             "concentration at 3 indicates frequent ties.",
             fig_num=fig_num, alt="Rating distribution")}
""")

    fig_num += 1
    html.append(f"""
<h2 id="position-bias">8. Position Bias</h2>
<p>Do first-listed characters get systematically different scores than second-listed?
If both histograms are symmetric and centered at 0, there is no position bias.</p>
{html_figure(figures['position_bias'],
             "Overlapping histograms of relative scores for first-listed (red) vs second-listed "
             "(blue) characters. Symmetric distributions centered at 0 indicate no position bias.",
             fig_num=fig_num, alt="Position bias analysis")}

<div class="stat">
<strong>Position bias:</strong>
First-listed mean: <strong>{np.mean(position_first):+.4f}</strong>,
Second-listed mean: <strong>{np.mean(position_second):+.4f}</strong>.
</div>
""")

    html.append(build_html_footer())

    out_dir = os.path.join(str(rdir), dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "detailed_response_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed pairwise response report"
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
        print(f"  Detailed Response Report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
