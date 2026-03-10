#!/usr/bin/env python3
"""
Behavioral Attribution Report Generator

For each concept dimension, analyzes cross-type (human vs AI) pairwise comparisons
to determine how strongly the model attributes each mental property to humans vs AI
characters. Uses the same relative-scoring approach as the behavioral PCA:
    entity A score = (3 - R), entity B score = (R - 3)
which cancels position bias when both pair orders are included.

Reads from:
    results/{model}/concept_geometry/pca/behavioral/data/
        pairwise_raw_responses.json

Writes to:
    results/{model}/concept_geometry/pca/behavioral/{dataset}/
        behavioral_attribution_report.html

Usage:
    python concept_geometry/pca/behavioral_attribution_report_generator.py --model llama2_13b_base
    python concept_geometry/pca/behavioral_attribution_report_generator.py --model llama2_13b_chat
    python concept_geometry/pca/behavioral_attribution_report_generator.py --model llama2_13b_chat --both

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir,
)
from concept_geometry.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from concept_geometry.concepts import CONCEPT_NAMES, CONCEPT_DIMENSIONS
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
)


# ============================================================================
# EXPECTED DIRECTION LABELS
# ============================================================================

EXPECTED_DIRECTION = {
    "phenomenology": "human",
    "emotions":      "human",
    "agency":        "human",
    "intentions":    "human",
    "prediction":    "human",
    "social":        "human",
    "embodiment":    "human",
    "animacy":       "human",
    "biological":    "human",
    "human":         "human",
    "beliefs":       "human",
    "desires":       "human",
    "cognitive":     "AI",
    "formality":     "AI",
    "expertise":     "AI",
    "helpfulness":   "AI",
    "ai":            "AI",
    "attention":     "AI",
    "roles":         "ambiguous",
    "shapes":        "ambiguous",
    "general_mind":  "ambiguous",
    "goals":         "ambiguous",
}

DIRECTION_COLORS = {
    "human": "#1E88E5",
    "AI": "#E53935",
    "ambiguous": "#9E9E9E",
}

SECTIONS = [
    {"id": "methodology", "label": "1. Methodology"},
    {"id": "human-advantage", "label": "2. Human Advantage per Concept"},
    {"id": "effect-sizes", "label": "3. Effect Sizes"},
    {"id": "direction-validation", "label": "4. Direction Validation"},
    {"id": "per-character", "label": "5. Per-Character Breakdown"},
    {"id": "heatmap", "label": "6. Concept x Character Heatmap"},
]


def compute_human_advantage(responses, is_chat):
    """
    Compute human-advantage scores for each cross-type pair.

    Uses the same relative-scoring as the behavioral PCA (compute_character_means_pairwise):
        entity A score = (3 - R), entity B score = (R - 3)

    For each cross-type pair, the "human advantage" is the human character's relative
    score: positive means human favored, negative means AI favored. This cancels
    position bias because both pair orders contribute equally.

    Returns:
        concept_scores: dict[concept] -> list of human-advantage scores
        char_scores: dict[character] -> list of relative scores (positive = this char favored)
        concept_char_scores: dict[concept][character] -> list of relative scores
    """
    rating_key = "rating" if is_chat else "expected_rating"

    concept_scores = defaultdict(list)
    char_scores = defaultdict(list)
    concept_char_scores = defaultdict(lambda: defaultdict(list))

    for resp in responses:
        ea, eb = resp["entity_a"], resp["entity_b"]
        ta, tb = CHARACTER_TYPES[ea], CHARACTER_TYPES[eb]

        # Only cross-type pairs
        if ta == tb:
            continue

        rating = resp.get(rating_key)
        if rating is None:
            continue

        cap = resp["capacity"]

        # Relative scores (same as PCA)
        score_a = 3 - rating   # positive when A is favored
        score_b = rating - 3   # positive when B is favored

        # Human advantage: the human character's relative score
        if ta == "human":
            human_adv = score_a   # human is A
            human_char, ai_char = ea, eb
        else:
            human_adv = score_b   # human is B
            human_char, ai_char = eb, ea

        concept_scores[cap].append(human_adv)

        # Per-character: each character's relative score (positive = favored)
        char_scores[human_char].append(score_a if ta == "human" else score_b)
        char_scores[ai_char].append(score_a if ta == "ai" else score_b)

        # Per concept x character: human advantage from perspective of this pair
        concept_char_scores[cap][human_char].append(human_adv)
        concept_char_scores[cap][ai_char].append(human_adv)

    return dict(concept_scores), dict(char_scores), concept_char_scores


def run_concept_tests(concept_scores):
    """
    Run one-sample t-tests (H0: mean human advantage = 0) per concept,
    with FDR correction.
    """
    from scipy.stats import ttest_1samp
    from statsmodels.stats.multitest import multipletests

    results = {}
    p_values = []
    concept_order = []

    for cap in sorted(concept_scores.keys()):
        scores = np.array(concept_scores[cap])
        n = len(scores)
        mean_adv = float(np.mean(scores))
        std_adv = float(np.std(scores, ddof=1)) if n > 1 else 0.0

        if n < 2 or std_adv == 0:
            results[cap] = {
                "n": n, "mean": mean_adv, "std": std_adv,
                "t_stat": np.nan, "p_value": np.nan, "p_fdr": np.nan,
                "dominant": "tie",
            }
            continue

        t_stat, p_val = ttest_1samp(scores, 0)

        results[cap] = {
            "n": n,
            "mean": mean_adv,
            "std": std_adv,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "dominant": "human" if mean_adv > 0 else "AI" if mean_adv < 0 else "tie",
        }
        p_values.append(p_val)
        concept_order.append(cap)

    # FDR correction
    if p_values:
        _, p_fdr, _, _ = multipletests(p_values, method="fdr_bh")
        for cap, pf in zip(concept_order, p_fdr):
            results[cap]["p_fdr"] = float(pf)

    for cap in results:
        if "p_fdr" not in results[cap]:
            results[cap]["p_fdr"] = np.nan

    return results


def generate_report(model_key, dataset="full_dataset"):
    """Generate behavioral attribution report for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    set_model(model_key)
    is_chat = config.IS_CHAT

    ddir = data_dir("concept_geometry/pca", "behavioral")
    rdir = results_phase_dir("concept_geometry/pca", "behavioral")

    raw_path = os.path.join(str(ddir), "pairwise_raw_responses.json")
    if not os.path.exists(raw_path):
        print(f"Raw responses not found at {raw_path} — skipping {model_key}")
        return None

    with open(raw_path) as f:
        responses = json.load(f)

    print(f"Loaded {len(responses)} pairwise responses")

    # ── Analysis ──
    concept_scores, char_scores, concept_char_scores = compute_human_advantage(
        responses, is_chat
    )
    test_results = run_concept_tests(concept_scores)

    capacities = sorted(test_results.keys())
    n_concepts = len(capacities)

    # Build concept name map
    concept_name_map = {}
    for cap in capacities:
        if cap in CONCEPT_DIMENSIONS:
            concept_name_map[cap] = CONCEPT_DIMENSIONS[cap]["name"]
        else:
            concept_name_map[cap] = cap.replace("_", " ").title()

    # Overall stats
    all_scores = []
    for cap in capacities:
        all_scores.extend(concept_scores[cap])
    all_scores = np.array(all_scores)
    overall_mean = float(np.mean(all_scores))
    overall_n = len(all_scores)

    # Direction validation
    dir_results = []
    for cap in capacities:
        expected = EXPECTED_DIRECTION.get(cap, "ambiguous")
        r = test_results[cap]
        observed = r["dominant"]
        sig = not np.isnan(r["p_fdr"]) and r["p_fdr"] < 0.05
        match = (expected == "ambiguous") or (expected == observed and sig)
        dir_results.append({
            "concept": cap,
            "expected": expected,
            "observed": observed,
            "match": match,
            "mean_adv": r["mean"],
            "p_fdr": r["p_fdr"],
            "sig": sig,
        })

    human_expected = [d for d in dir_results if d["expected"] == "human"]
    ai_expected = [d for d in dir_results if d["expected"] == "AI"]
    human_correct = sum(1 for d in human_expected if d["match"])
    ai_correct = sum(1 for d in ai_expected if d["match"])

    figures = {}

    # ── Figure 1: Mean human advantage per concept ──
    fig, ax = plt.subplots(figsize=(10, max(6, n_concepts * 0.35)))
    sorted_caps = sorted(capacities, key=lambda c: test_results[c]["mean"])
    y_pos = np.arange(len(sorted_caps))
    means = [test_results[c]["mean"] for c in sorted_caps]
    colors = [DIRECTION_COLORS.get(EXPECTED_DIRECTION.get(c, "ambiguous"), "#9E9E9E")
              for c in sorted_caps]

    ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5, alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)

    # Significance markers
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
    ax.set_yticklabels([concept_name_map[c] for c in sorted_caps], fontsize=9)
    ax.set_xlabel("Mean Human Advantage (positive = human favored)")
    ax.set_title(f"Human Advantage per Concept — {config.MODEL_LABEL}")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=DIRECTION_COLORS["human"], alpha=0.8, label="Expected: human"),
        Patch(facecolor=DIRECTION_COLORS["AI"], alpha=0.8, label="Expected: AI"),
        Patch(facecolor=DIRECTION_COLORS["ambiguous"], alpha=0.8, label="Ambiguous"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    figures["human_advantage"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Figure 2: Per-character mean relative score (human chars) ──
    human_char_data = []
    for c in HUMAN_CHARACTERS:
        if c in char_scores and len(char_scores[c]) > 0:
            human_char_data.append((c, float(np.mean(char_scores[c]))))
    human_char_data.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, max(4, len(human_char_data) * 0.35)))
    y_pos = np.arange(len(human_char_data))
    vals = [d[1] for d in human_char_data]
    ax.barh(y_pos, vals, color="#1E88E5", alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CHARACTER_NAMES[d[0]] for d in human_char_data], fontsize=9)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean Relative Score vs AI Opponents (positive = favored)")
    ax.set_title(f"Human Characters — Relative Score — {config.MODEL_LABEL}")
    plt.tight_layout()
    figures["human_chars"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Figure 3: Per-character mean relative score (AI chars) ──
    ai_char_data = []
    for c in AI_CHARACTERS:
        if c in char_scores and len(char_scores[c]) > 0:
            ai_char_data.append((c, float(np.mean(char_scores[c]))))
    ai_char_data.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(8, max(4, len(ai_char_data) * 0.35)))
    y_pos = np.arange(len(ai_char_data))
    vals = [d[1] for d in ai_char_data]
    ax.barh(y_pos, vals, color="#E53935", alpha=0.8, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([CHARACTER_NAMES[d[0]] for d in ai_char_data], fontsize=9)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Mean Relative Score vs Human Opponents (positive = favored)")
    ax.set_title(f"AI Characters — Relative Score — {config.MODEL_LABEL}")
    plt.tight_layout()
    figures["ai_chars"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Figure 4: Concept x Character heatmap ──
    concept_order = sorted(capacities, key=lambda c: (
        {"human": 0, "AI": 1, "ambiguous": 2}.get(
            EXPECTED_DIRECTION.get(c, "ambiguous"), 2),
        -test_results[c]["mean"]
    ))
    char_order = AI_CHARACTERS + HUMAN_CHARACTERS

    # Build matrix: mean human advantage for each concept x character
    heatmap = np.full((len(concept_order), len(char_order)), np.nan)
    for i, cap in enumerate(concept_order):
        for j, ch in enumerate(char_order):
            if cap in concept_char_scores and ch in concept_char_scores[cap]:
                vals = concept_char_scores[cap][ch]
                if vals:
                    heatmap[i, j] = np.mean(vals)

    vmax = max(abs(np.nanmin(heatmap)), abs(np.nanmax(heatmap)))
    if vmax == 0:
        vmax = 0.1

    fig, ax = plt.subplots(figsize=(max(14, len(char_order) * 0.5),
                                     max(6, len(concept_order) * 0.4)))
    cmap = plt.cm.RdBu
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(heatmap, cmap=cmap, norm=norm, aspect="auto")

    char_labels = [CHARACTER_NAMES[c] for c in char_order]
    ax.set_xticks(range(len(char_order)))
    ax.set_xticklabels(char_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(concept_order)))
    ax.set_yticklabels([concept_name_map[c] for c in concept_order], fontsize=9)

    # Separator between AI and human characters
    sep_x = len(AI_CHARACTERS) - 0.5
    ax.axvline(x=sep_x, color="black", linewidth=2)

    ax.text(len(AI_CHARACTERS) / 2 - 0.5, -1.5, "AI Characters",
            ha="center", fontsize=10, fontweight="bold", color="#E53935")
    ax.text(len(AI_CHARACTERS) + len(HUMAN_CHARACTERS) / 2 - 0.5, -1.5,
            "Human Characters", ha="center", fontsize=10, fontweight="bold",
            color="#1E88E5")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Mean Human Advantage")
    ax.set_title(f"Human Advantage: Concept x Character — {config.MODEL_LABEL}",
                 pad=25)
    plt.tight_layout()
    figures["heatmap"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Figure 5: Effect size (Cohen's d) per concept ──
    fig, ax = plt.subplots(figsize=(10, max(6, n_concepts * 0.35)))
    es_y = np.arange(len(sorted_caps))
    cohens_d = []
    es_colors = []
    for cap in sorted_caps:
        r = test_results[cap]
        d = r["mean"] / r["std"] if r["std"] > 0 else 0
        cohens_d.append(d)
        es_colors.append(DIRECTION_COLORS.get(
            EXPECTED_DIRECTION.get(cap, "ambiguous"), "#9E9E9E"))

    ax.barh(es_y, cohens_d, color=es_colors, edgecolor="white", linewidth=0.5,
            alpha=0.8)
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    for threshold, label in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        ax.axvline(x=threshold, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.axvline(x=-threshold, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    ax.set_yticks(es_y)
    ax.set_yticklabels([concept_name_map[c] for c in sorted_caps], fontsize=9)
    ax.set_xlabel("Cohen's d (positive = human favored)")
    ax.set_title(f"Effect Size per Concept — {config.MODEL_LABEL}")
    plt.tight_layout()
    figures["effect_size"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    html = []
    fig_num = 0

    rating_method = ("the model's generated rating (1-5 integer)" if is_chat else
                     "the probability-weighted expected rating (continuous 1-5 scale) "
                     "computed from next-token logit probabilities over digits 1-5")

    html.append(build_html_header("Behavioral Attribution Report", config.MODEL_LABEL))

    html.append(f"""
<div class="stat">
<strong>Summary:</strong> {n_concepts} concepts, {len(HUMAN_CHARACTERS)} human characters,
{len(AI_CHARACTERS)} AI characters.
<strong>{overall_n}</strong> cross-type comparisons analyzed.
Overall mean human advantage: <strong>{overall_mean:+.4f}</strong>.
</div>
""")

    html.append(build_toc(SECTIONS))

    html.append(f"""
<h2 id="methodology">1. Methodology</h2>
<div class="method">
<strong>Research question:</strong> For each concept dimension, does {config.MODEL_LABEL}
systematically attribute that property more to human or AI characters?
<ol>
<li><strong>Data source:</strong> Pairwise comparison data from the behavioral PCA experiment.
    Each trial presents two characters and asks which is "more capable of" a given mental
    property. Rating scale: 1 = "much more A", 3 = "equal", 5 = "much more B".
    Rating extracted via {rating_method}.</li>
<li><strong>Cross-type filtering:</strong> Only pairs where one character is human and one is AI
    are retained. Same-type pairs (human-human, AI-AI) are excluded.</li>
<li><strong>Relative scoring:</strong> Following the same approach as the behavioral PCA,
    each rating R is converted to relative scores: entity A gets (3 &minus; R), entity B
    gets (R &minus; 3). This centers scores at 0 and, crucially, <em>cancels position bias</em>
    because both pair orders (A-vs-B and B-vs-A) are included.</li>
<li><strong>Human advantage:</strong> For each cross-type pair, the human character's relative
    score is the "human advantage." Positive values mean the human was favored; negative
    values mean the AI was favored.</li>
<li><strong>Statistical test:</strong> One-sample t-test per concept (H<sub>0</sub>: mean
    human advantage = 0). FDR-corrected (Benjamini-Hochberg) across all concepts.
    Effect sizes reported as Cohen's d.</li>
</ol>
</div>

<div class="interpret">
<strong>How to interpret:</strong>
<ul>
<li><strong>Mean human advantage &gt; 0:</strong> The model attributes this property more to
    human characters (e.g., phenomenology, emotions).</li>
<li><strong>Mean human advantage &lt; 0:</strong> The model attributes this property more to
    AI characters (e.g., formality, helpfulness).</li>
<li><strong>Near zero:</strong> No systematic attribution difference between human and AI.</li>
<li><strong>Scale:</strong> The maximum possible advantage is &plusmn;2 (one character always
    gets rating 1, the other always gets 5). Values of &plusmn;0.01 to &plusmn;0.05 are typical
    for the base model.</li>
</ul>
</div>
""")

    # Section 2: Human advantage per concept
    fig_num += 1
    html.append(f"""
<h2 id="human-advantage">2. Human Advantage per Concept</h2>
<p>Bar chart shows the mean human advantage for each concept, colored by expected direction
(blue = expected human-favored, red = expected AI-favored, gray = ambiguous).
Stars indicate FDR-corrected significance (* p &lt; .05, ** p &lt; .01, *** p &lt; .001).
The dashed line at 0 represents no systematic difference.</p>
""")
    html.append(html_figure(
        figures['human_advantage'],
        "Mean human advantage for each concept dimension from cross-type pair analysis. "
        "Positive = human-favored, negative = AI-favored. Stars indicate FDR-corrected significance.",
        fig_num=fig_num,
        alt="Human advantage per concept",
    ))

    html.append("""
<h3>Full Results Table</h3>
<table>
<tr><th>Concept</th><th>Expected</th><th>N pairs</th><th>Mean Adv.</th>
<th>Std</th><th>t</th><th>p (raw)</th><th>p (FDR)</th><th>Cohen's d</th>
<th>Dominant</th></tr>
""")

    for cap in sorted(capacities, key=lambda c: -test_results[c]["mean"]):
        r = test_results[cap]
        expected = EXPECTED_DIRECTION.get(cap, "ambiguous")
        p_class = ' class="sig"' if (not np.isnan(r["p_fdr"]) and r["p_fdr"] < 0.05) else ""
        d_val = r["mean"] / r["std"] if r["std"] > 0 else 0
        p_str = f'{r["p_value"]:.2e}' if not np.isnan(r["p_value"]) else "—"
        pf_str = f'{r["p_fdr"]:.2e}' if not np.isnan(r["p_fdr"]) else "—"
        t_str = f'{r["t_stat"]:.2f}' if not np.isnan(r["t_stat"]) else "—"
        html.append(
            f'<tr><td>{concept_name_map[cap]}</td><td>{expected}</td>'
            f'<td>{r["n"]}</td><td{p_class}>{r["mean"]:+.4f}</td>'
            f'<td>{r["std"]:.4f}</td><td>{t_str}</td><td>{p_str}</td>'
            f'<td>{pf_str}</td><td>{d_val:+.3f}</td>'
            f'<td>{r["dominant"]}</td></tr>\n'
        )

    html.append("</table>\n")

    # Section 3: Effect sizes
    fig_num += 1
    html.append(f"""
<h2 id="effect-sizes">3. Effect Sizes</h2>
<p>Cohen's d for each concept. Dotted lines mark conventional thresholds
(0.2 = small, 0.5 = medium, 0.8 = large). Positive d means human-favored.</p>
""")
    html.append(html_figure(
        figures['effect_size'],
        "Cohen's d effect sizes for the human advantage on each concept. "
        "Larger absolute values indicate stronger differentiation between AI and human characters.",
        fig_num=fig_num,
        alt="Effect sizes per concept",
    ))

    # Section 4: Direction validation
    html.append(f"""
<h2 id="direction-validation">4. Direction Validation</h2>
<p>Does the model's observed attribution direction match the conceptually expected
direction? A concept "matches" if it shows significant (FDR p &lt; .05)
human advantage in the expected direction (positive for human-expected,
negative for AI-expected). Ambiguous concepts always count as matching.</p>

<div class="{'success' if human_correct == len(human_expected) and ai_correct == len(ai_expected) else 'stat'}">
<strong>Human-favored concepts:</strong> {human_correct}/{len(human_expected)} significantly match expected direction.<br>
<strong>AI-favored concepts:</strong> {ai_correct}/{len(ai_expected)} significantly match expected direction.
</div>

<table>
<tr><th>Concept</th><th>Expected</th><th>Observed</th><th>Mean Adv.</th>
<th>p (FDR)</th><th>Sig?</th><th>Match?</th></tr>
""")

    for d in sorted(dir_results, key=lambda x: (
        {"human": 0, "AI": 1, "ambiguous": 2}[x["expected"]], x["concept"])):
        pf_str = f'{d["p_fdr"]:.2e}' if not np.isnan(d["p_fdr"]) else "—"
        match_class = "match" if d["match"] else "mismatch"
        match_str = "Yes" if d["match"] else "NO"
        sig_str = "Yes" if d["sig"] else "No"
        html.append(
            f'<tr><td>{concept_name_map[d["concept"]]}</td><td>{d["expected"]}</td>'
            f'<td>{d["observed"]}</td><td>{d["mean_adv"]:+.4f}</td><td>{pf_str}</td>'
            f'<td>{sig_str}</td>'
            f'<td class="{match_class}">{match_str}</td></tr>\n'
        )

    html.append("</table>\n")

    # Summarize mismatches
    mismatches = [d for d in dir_results if not d["match"] and d["expected"] != "ambiguous"]
    if mismatches:
        # Distinguish "wrong direction" from "right direction but not significant"
        wrong_dir = [d for d in mismatches if d["observed"] != d["expected"]
                     and d["observed"] != "tie"]
        not_sig = [d for d in mismatches if d["observed"] == d["expected"]
                   or d["observed"] == "tie"]

        if wrong_dir:
            html.append('<div class="warning"><strong>Wrong direction:</strong><ul>')
            for d in wrong_dir:
                html.append(
                    f'<li><strong>{concept_name_map[d["concept"]]}</strong>: expected '
                    f'{d["expected"]}, observed {d["observed"]} '
                    f'(mean adv = {d["mean_adv"]:+.4f})</li>'
                )
            html.append("</ul></div>\n")

        if not_sig:
            html.append('<div class="interpret"><strong>Right direction but not significant '
                        '(FDR p &ge; .05):</strong><ul>')
            for d in not_sig:
                html.append(
                    f'<li><strong>{concept_name_map[d["concept"]]}</strong>: '
                    f'mean adv = {d["mean_adv"]:+.4f}, p(FDR) = '
                    f'{d["p_fdr"]:.2e}</li>'
                )
            html.append("</ul></div>\n")
    else:
        html.append("""
<div class="success">All non-ambiguous concepts significantly match their expected direction.</div>
""")

    # Section 5: Per-character breakdown
    fig_num += 1
    html.append(f"""
<h2 id="per-character">5. Per-Character Breakdown</h2>
<p>Mean relative score for each character across all concepts against cross-type opponents.
Positive values mean this character tends to be favored over its opponents.</p>

<h3>Human Characters</h3>
""")
    html.append(html_figure(
        figures['human_chars'],
        "Mean relative score for each human character across all concepts.",
        fig_num=fig_num,
        alt="Human character scores",
    ))

    fig_num += 1
    html.append("""<h3>AI Characters</h3>
""")
    html.append(html_figure(
        figures['ai_chars'],
        "Mean relative score for each AI character across all concepts.",
        fig_num=fig_num,
        alt="AI character scores",
    ))

    # Tables
    html.append("""<h3>Human Character Details</h3>
<table><tr><th>Character</th><th>N pairs</th><th>Mean Score</th><th>Std</th></tr>""")

    for ch in sorted(HUMAN_CHARACTERS,
                     key=lambda c: -np.mean(char_scores[c]) if c in char_scores else 0):
        if ch in char_scores:
            vals = char_scores[ch]
            html.append(
                f'<tr><td>{CHARACTER_NAMES[ch]}</td><td>{len(vals)}</td>'
                f'<td>{np.mean(vals):+.4f}</td><td>{np.std(vals):.4f}</td></tr>\n'
            )
    html.append("</table>\n")

    html.append("""<h3>AI Character Details</h3>
<table><tr><th>Character</th><th>N pairs</th><th>Mean Score</th><th>Std</th></tr>""")

    for ch in sorted(AI_CHARACTERS,
                     key=lambda c: -np.mean(char_scores[c]) if c in char_scores else 0):
        if ch in char_scores:
            vals = char_scores[ch]
            html.append(
                f'<tr><td>{CHARACTER_NAMES[ch]}</td><td>{len(vals)}</td>'
                f'<td>{np.mean(vals):+.4f}</td><td>{np.std(vals):.4f}</td></tr>\n'
            )
    html.append("</table>\n")

    # Section 6: Heatmap
    fig_num += 1
    html.append("""
<h2 id="heatmap">6. Concept x Character Heatmap</h2>
<p>Each cell shows the mean human advantage for that concept-character combination.
Red = human favored (positive), blue = AI favored (negative), white = equal.
Characters are split by type (AI left, human right) with a black separator.
Concepts are sorted by expected direction (human-favored top, AI-favored middle,
ambiguous bottom).</p>
""")
    html.append(html_figure(
        figures['heatmap'],
        "Full concept-by-character matrix of mean relative scores. "
        "Red = character favored, blue = disfavored. AI characters left, human right.",
        fig_num=fig_num,
        alt="Concept x character heatmap",
    ))

    html.append(build_html_footer())

    # ── Write ──
    out_dir = os.path.join(str(rdir), dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "behavioral_attribution_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate behavioral attribution report (cross-type pair analysis)"
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
        print(f"  Generating attribution report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
