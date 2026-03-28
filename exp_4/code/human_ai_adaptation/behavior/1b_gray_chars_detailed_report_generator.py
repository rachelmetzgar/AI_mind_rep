#!/usr/bin/env python3
"""
Gray-with-Characters Detailed Response Report Generator

Shows how the model rates AI vs human characters on each of the 18 mental
capacities. Includes human-advantage analysis, per-character breakdowns,
and Experience vs Agency comparisons.

Reads from:
    data_dir("human_ai_adaptation", "behavior")/
        pairwise_raw_responses.json

Writes to:
    results_dir("human_ai_adaptation", "behavior")/{dataset}/
        gray_chars_detailed_report.html

Usage:
    python behavior/4b_gray_chars_detailed_report_generator.py --model llama2_13b_base
    python behavior/4b_gray_chars_detailed_report_generator.py --model llama2_13b_chat --both

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir,
)
from entities.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from entities.gray_entities import CAPACITY_PROMPTS, CAPACITY_NAMES
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
    characters_stimuli_html,
)

# ── Table of contents ──
SECTIONS = [
    {"id": "methods", "label": "1. Methods"},
    {"id": "stimuli", "label": "2. Stimuli"},
    {"id": "human-advantage", "label": "3. Human Advantage per Capacity"},
    {"id": "per-character", "label": "4. Per-Character Scores"},
    {"id": "heatmap", "label": "5. Capacity x Character Heatmap"},
    {"id": "exp-vs-ag", "label": "6. Experience vs Agency"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate Gray-with-characters detailed report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests

    set_model(model_key)
    is_chat = config.IS_CHAT
    ddir = data_dir("human_ai_adaptation", "behavior")
    rdir = results_dir("human_ai_adaptation", "behavior")

    raw_path = os.path.join(str(ddir), "pairwise_raw_responses.json")
    if not os.path.exists(raw_path):
        print(f"Raw responses not found at {raw_path} — skipping {model_key}")
        return None

    with open(raw_path) as f:
        responses = json.load(f)

    print(f"Loaded {len(responses)} responses for {config.MODEL_LABEL}")

    rating_key = "rating" if is_chat else "expected_rating"

    # Compute scores
    cap_human_advantage = defaultdict(list)
    char_scores = defaultdict(list)
    cap_char_scores = defaultdict(lambda: defaultdict(list))

    for resp in responses:
        ea, eb = resp["entity_a"], resp["entity_b"]
        ta, tb = CHARACTER_TYPES.get(ea), CHARACTER_TYPES.get(eb)
        rating = resp.get(rating_key)
        if rating is None:
            continue

        cap = resp["capacity"]
        score_a = 3 - rating
        score_b = rating - 3

        # Per-character scores
        cap_char_scores[cap][ea].append(score_a)
        cap_char_scores[cap][eb].append(score_b)
        char_scores[ea].append(score_a)
        char_scores[eb].append(score_b)

        # Cross-type: human advantage
        if ta != tb:
            if ta == "human":
                cap_human_advantage[cap].append(score_a)
            else:
                cap_human_advantage[cap].append(score_b)

    capacities = list(CAPACITY_NAMES)
    figures = {}
    fig_num = 1

    # ── 1. Human advantage per capacity ──
    test_results = {}
    p_values = []
    cap_order = []
    for cap in capacities:
        if cap not in cap_human_advantage:
            continue
        scores = np.array(cap_human_advantage[cap])
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

    # Color by Gray factor
    FACTOR_COLORS = {"E": "#FF9800", "A": "#4CAF50"}
    sorted_caps = sorted(test_results.keys(), key=lambda c: test_results[c]["mean"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(sorted_caps) * 0.35)))
    y_pos = np.arange(len(sorted_caps))
    means = [test_results[c]["mean"] for c in sorted_caps]
    colors = [FACTOR_COLORS.get(CAPACITY_PROMPTS[c][1], "#9E9E9E")
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
    ax.set_yticklabels([f"{c} ({CAPACITY_PROMPTS[c][1]})" for c in sorted_caps],
                       fontsize=9)
    ax.set_xlabel("Mean Human Advantage (positive = human favored)")
    ax.set_title(f"Human Advantage per Capacity — {config.MODEL_LABEL}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FACTOR_COLORS["E"], alpha=0.8, label="Experience (E)"),
        Patch(facecolor=FACTOR_COLORS["A"], alpha=0.8, label="Agency (A)"),
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
        char_data = [(c, float(np.mean(char_scores[c])))
                     for c in group_chars if c in char_scores]
        char_data.sort(key=lambda x: x[1])

        fig, ax = plt.subplots(figsize=(8, max(4, len(char_data) * 0.35)))
        y_pos = np.arange(len(char_data))
        vals = [d[1] for d in char_data]
        ax.barh(y_pos, vals, color=color, alpha=0.8, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([CHARACTER_NAMES[d[0]] for d in char_data], fontsize=9)
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Mean Relative Score")
        ax.set_title(f"{group_name.upper()} — {config.MODEL_LABEL}")
        plt.tight_layout()
        figures[f"{group_name}_chars"] = fig_to_b64(fig)
        plt.close(fig)

    # ── 3. Capacity x Character heatmap ──
    cap_order_heatmap = sorted(capacities,
                                key=lambda c: CAPACITY_PROMPTS[c][1])
    char_order = AI_CHARACTERS + HUMAN_CHARACTERS

    heatmap = np.full((len(cap_order_heatmap), len(char_order)), np.nan)
    for i, cap in enumerate(cap_order_heatmap):
        for j, ch in enumerate(char_order):
            if cap in cap_char_scores and ch in cap_char_scores[cap]:
                vals = cap_char_scores[cap][ch]
                if vals:
                    heatmap[i, j] = np.mean(vals)

    vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)))
    if vmax == 0:
        vmax = 0.1

    fig, ax = plt.subplots(figsize=(max(14, len(char_order) * 0.5),
                                     max(6, len(cap_order_heatmap) * 0.4)))
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(heatmap, cmap=plt.cm.RdBu, norm=norm, aspect="auto")
    char_labels = [CHARACTER_NAMES[c] for c in char_order]
    ax.set_xticks(range(len(char_order)))
    ax.set_xticklabels(char_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cap_order_heatmap)))
    ax.set_yticklabels([f"{c} ({CAPACITY_PROMPTS[c][1]})"
                        for c in cap_order_heatmap], fontsize=9)
    sep_x = len(AI_CHARACTERS) - 0.5
    ax.axvline(x=sep_x, color="black", linewidth=2)
    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean Relative Score")
    ax.set_title(f"Capacity x Character — {config.MODEL_LABEL}", pad=15)
    plt.tight_layout()
    figures["heatmap"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 4. Experience vs Agency comparison ──
    exp_caps = [c for c in capacities if CAPACITY_PROMPTS[c][1] == "E"]
    ag_caps = [c for c in capacities if CAPACITY_PROMPTS[c][1] == "A"]

    exp_means = [test_results[c]["mean"] for c in exp_caps if c in test_results]
    ag_means = [test_results[c]["mean"] for c in ag_caps if c in test_results]

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = [0, 1]
    bp = ax.boxplot([exp_means, ag_means], positions=positions,
                    patch_artist=True, widths=0.6)
    bp["boxes"][0].set_facecolor("#FF9800")
    bp["boxes"][1].set_facecolor("#4CAF50")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Experience\nCapacities", "Agency\nCapacities"])
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("Mean Human Advantage")
    ax.set_title(f"Experience vs Agency — {config.MODEL_LABEL}")
    plt.tight_layout()
    figures["exp_vs_ag"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    html = []
    html.append(build_html_header(
        "Gray Characters Detailed Response Report", config.MODEL_LABEL))

    html.append(f"""
<div class="stat">
<strong>Summary:</strong> {len(responses)} total comparisons,
{len(capacities)} Gray et al. mental capacities,
{len(ALL_CHARACTERS)} characters ({len(AI_CHARACTERS)} AI, {len(HUMAN_CHARACTERS)} human).
</div>
""")

    html.append(build_toc(SECTIONS))

    # ── Methods ──
    html.append(f"""
<h2 id="methods">1. Methods</h2>
<div class="method">
<strong>Research question:</strong> Which of Gray et al.&rsquo;s 18 mental capacities most
differentiate AI from human characters in {config.MODEL_LABEL}&rsquo;s explicit judgments?
Do Experience capacities (e.g., hunger, pain, pleasure) show larger human advantages
than Agency capacities (e.g., self-control, planning)?

<p><strong>Data source:</strong> {len(responses)} pairwise comparison responses from
{config.MODEL_LABEL}. Each trial presents two characters and one mental capacity,
asking which character possesses more of that capacity (1&ndash;5 scale). Each pair
is tested in both presentation orders.</p>

<strong>Procedure:</strong>
<ol>
<li><strong>Relative scoring:</strong> For each comparison of entity A vs B with
    rating R: entity A gets (3&nbsp;&minus;&nbsp;R), entity B gets (R&nbsp;&minus;&nbsp;3).
    A rating of 3 (equal) contributes 0 to both.</li>
<li><strong>Cross-type pair filtering:</strong> Only pairs where one character is AI
    and the other is human are used for the human-advantage analysis. Within-type
    pairs (AI-vs-AI, human-vs-human) are excluded.</li>
<li><strong>Per-capacity human advantage:</strong> For each capacity, the mean score
    from the human character&rsquo;s perspective across all cross-type comparisons.
    Positive = humans rated higher; negative = AI rated higher.</li>
<li><strong>Statistical tests:</strong> One-sample t-tests (H0: human advantage = 0)
    per capacity, corrected for multiple comparisons using Benjamini-Hochberg FDR.</li>
<li><strong>Experience vs Agency grouping:</strong> Capacities are grouped by their
    Gray et al. factor assignment (E or A) to compare which dimension shows larger
    human advantages overall.</li>
</ol>
</div>
""")

    # ── Stimuli ──
    html.append(characters_stimuli_html(include_capacities=True))

    # ── Human advantage ──
    html.append(f'<h2 id="human-advantage">3. Human Advantage per Capacity</h2>\n')
    html.append(html_figure(
        figures['human_advantage'],
        "Mean human advantage for each of the 18 Gray et al. mental capacities. "
        "Positive values indicate the model attributes more of that capacity to "
        "human characters; negative values favor AI characters. Orange = Experience "
        "capacities, green = Agency capacities. Stars indicate FDR-corrected "
        "significance (* p &lt; .05, ** p &lt; .01, *** p &lt; .001).",
        fig_num=fig_num, alt="Human advantage per capacity"))
    fig_num += 1

    html.append("""
<h3>Results Table</h3>
<table>
<tr><th>Capacity</th><th>Factor</th><th>N</th><th>Mean Adv.</th>
<th>Std</th><th>t</th><th>p (FDR)</th></tr>
""")

    for cap in sorted(test_results.keys(),
                      key=lambda c: -test_results[c]["mean"]):
        r = test_results[cap]
        _, factor = CAPACITY_PROMPTS[cap]
        p_class = ' class="sig"' if (not np.isnan(r["p_fdr"]) and r["p_fdr"] < 0.05) else ""
        pf_str = f'{r["p_fdr"]:.2e}' if not np.isnan(r["p_fdr"]) else "—"
        t_str = f'{r["t_stat"]:.2f}' if not np.isnan(r["t_stat"]) else "—"
        html.append(
            f'<tr><td>{cap}</td><td>{factor}</td>'
            f'<td>{r["n"]}</td><td{p_class}>{r["mean"]:+.4f}</td>'
            f'<td>{r["std"]:.4f}</td><td>{t_str}</td><td>{pf_str}</td></tr>\n'
        )
    html.append("</table>\n")

    # ── Per-character ──
    html.append(f'<h2 id="per-character">4. Per-Character Scores</h2>\n')
    html.append('<h3>Human Characters</h3>\n')
    html.append(html_figure(
        figures['human_chars'],
        "Mean relative score for each human character across all capacities and "
        "comparisons. Positive values indicate the character is generally rated "
        "as having more mental capacities than its comparison partners.",
        fig_num=fig_num, alt="Human character scores"))
    fig_num += 1
    html.append('<h3>AI Characters</h3>\n')
    html.append(html_figure(
        figures['ai_chars'],
        "Mean relative score for each AI character across all capacities and "
        "comparisons.",
        fig_num=fig_num, alt="AI character scores"))
    fig_num += 1

    # ── Heatmap ──
    html.append(f'<h2 id="heatmap">5. Capacity x Character Heatmap</h2>\n')
    html.append(html_figure(
        figures['heatmap'],
        "Full capacity-by-character matrix of mean relative scores. Red = character "
        "favored (positive score), blue = disfavored (negative score). AI characters "
        "are left of the black divider, human characters to the right. Rows are grouped "
        "by Gray factor assignment (E = Experience, A = Agency).",
        fig_num=fig_num, alt="Capacity x character heatmap"))
    fig_num += 1

    # ── Experience vs Agency ──
    html.append(f'<h2 id="exp-vs-ag">6. Experience vs Agency Comparison</h2>\n')
    html.append(html_figure(
        figures['exp_vs_ag'],
        "Distribution of human advantage values across Experience capacities (orange) "
        "vs Agency capacities (green). If Experience capacities show larger human "
        "advantages than Agency capacities, this suggests the Experience dimension "
        "differentiates human/AI more strongly in the model's folk psychology.",
        fig_num=fig_num, alt="Experience vs Agency boxplot"))
    fig_num += 1

    html.append(build_html_footer())

    out_dir = os.path.join(str(rdir), dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gray_chars_detailed_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gray-with-characters detailed report"
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
        print(f"  Gray Characters Detailed Report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
