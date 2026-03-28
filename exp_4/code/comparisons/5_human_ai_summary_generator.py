#!/usr/bin/env python3
"""
Cross-Model Summary Report: Human-AI Adaptation Branch

Extends Gray's mind perception paradigm from 13 entities to 30 AI/human
characters. Compares PCA factor structure, AI-vs-human separation, and
names-only effects across all available models.

Usage:
    python comparisons/5_human_ai_summary_generator.py

Output:
    results/comparisons/human_ai_summary.html

Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# -- Imports from project ---------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, COMPARISONS_DIR, VALID_MODELS, MODELS, ensure_dir, set_model, data_dir, results_dir
from utils.report_utils import (
    REPORT_CSS, build_cross_model_header, build_html_footer, build_toc,
    fig_to_b64, html_figure, MODEL_COLORS, MODEL_LABELS, ALL_MODELS,
    characters_stimuli_html,
)
from utils.utils import nice_capacity
from entities.characters import (
    CHARACTER_INFO, AI_CHARACTERS, HUMAN_CHARACTERS, ALL_CHARACTERS,
)
from entities.gray_entities import CAPACITY_PROMPTS


# -- Style -----------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_AI = "#e41a1c"       # red for AI characters
C_HUMAN = "#377eb8"    # blue for human characters


# ==========================================================================
# DATA LOADING
# ==========================================================================

def load_model_data(model_key):
    """Load all human_ai_adaptation behavioral data for a model.

    Returns dict with pca, categorical, means, consistency data,
    or None if data is missing.
    """
    set_model(model_key)
    ddir = data_dir("human_ai_adaptation", "behavior")

    result = {"model": model_key, "label": MODEL_LABELS[model_key]}

    # PCA results
    pca_path = ddir / "pairwise_pca_results.npz"
    if not pca_path.exists():
        return None
    pca = np.load(pca_path, allow_pickle=True)
    result["pca"] = {k: pca[k] for k in pca.files}
    result["character_keys"] = list(pca["character_keys"])
    result["capacity_keys"] = list(pca["capacity_keys"])
    result["n_chars"] = len(result["character_keys"])
    result["n_factors"] = pca["factor_scores_01"].shape[1]

    # Categorical analysis
    cat_path = ddir / "pairwise_categorical_analysis.json"
    if cat_path.exists():
        with open(cat_path) as f:
            result["categorical"] = json.load(f)
    else:
        result["categorical"] = None

    # Character means
    means_path = ddir / "pairwise_character_means.npz"
    if means_path.exists():
        m = np.load(means_path, allow_pickle=True)
        result["means"] = {k: m[k] for k in m.files}
    else:
        result["means"] = None

    # Consistency stats
    cons_path = ddir / "pairwise_consistency_stats.json"
    if cons_path.exists():
        with open(cons_path) as f:
            result["consistency"] = json.load(f)
    else:
        result["consistency"] = None

    # Names-only (chat models only)
    names_dir = ddir.parent / "names_only" / "data"
    if names_dir.exists() and (names_dir / "pairwise_pca_results.npz").exists():
        npca = np.load(names_dir / "pairwise_pca_results.npz", allow_pickle=True)
        result["names_only_pca"] = {k: npca[k] for k in npca.files}
        result["names_only_character_keys"] = list(npca["character_keys"])
        ncat_path = names_dir / "pairwise_categorical_analysis.json"
        if ncat_path.exists():
            with open(ncat_path) as f:
                result["names_only_categorical"] = json.load(f)
        else:
            result["names_only_categorical"] = None
    else:
        result["names_only_pca"] = None

    return result


def classify_characters(char_keys):
    """Return lists of AI and human character keys from an ordered list."""
    ai_keys = [k for k in char_keys if CHARACTER_INFO.get(k, {}).get("type") == "ai"]
    human_keys = [k for k in char_keys if CHARACTER_INFO.get(k, {}).get("type") == "human"]
    return ai_keys, human_keys


def get_type_mask(char_keys):
    """Return boolean arrays: (is_ai, is_human) aligned to char_keys."""
    is_ai = np.array([CHARACTER_INFO.get(k, {}).get("type") == "ai" for k in char_keys])
    is_human = np.array([CHARACTER_INFO.get(k, {}).get("type") == "human" for k in char_keys])
    return is_ai, is_human


def nice_character(key):
    """Pretty-print a character key."""
    info = CHARACTER_INFO.get(key)
    if info:
        return info["name"]
    return key.replace("_", " ").title()


# ==========================================================================
# FIGURE GENERATORS
# ==========================================================================

def fig_pca_scatter(all_data):
    """PCA scatter: F1 vs F2 colored by AI/human type, one panel per model."""
    models_with_data = [d for d in all_data if d is not None]
    n = len(models_with_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    axes = axes[0]

    # Determine shared axis limits
    all_f1, all_f2 = [], []
    for d in models_with_data:
        scores = d["pca"]["factor_scores_01"]
        all_f1.extend(scores[:, 0].tolist())
        all_f2.extend(scores[:, 1].tolist())
    xlim = (min(all_f1) - 0.08, max(all_f1) + 0.08)
    ylim = (min(all_f2) - 0.08, max(all_f2) + 0.08)

    for idx, d in enumerate(models_with_data):
        ax = axes[idx]
        scores = d["pca"]["factor_scores_01"]
        char_keys = d["character_keys"]
        is_ai, is_human = get_type_mask(char_keys)

        ax.scatter(scores[is_ai, 0], scores[is_ai, 1],
                   s=70, c=C_AI, edgecolor="white", linewidth=0.5,
                   label="AI", zorder=5)
        ax.scatter(scores[is_human, 0], scores[is_human, 1],
                   s=70, c=C_HUMAN, edgecolor="white", linewidth=0.5,
                   label="Human", zorder=5)

        # Label each point
        for i, ck in enumerate(char_keys):
            ax.annotate(nice_character(ck), (scores[i, 0], scores[i, 1]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, ha="left", alpha=0.85)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("Factor 1 (0-1)")
        ax.set_ylabel("Factor 2 (0-1)")
        evar = d["pca"]["explained_var_ratio"]
        ev1 = evar[0] * 100 if len(evar) > 0 else 0
        ev2 = evar[1] * 100 if len(evar) > 1 else 0
        ax.set_title(f"{d['label']}\n({d['n_chars']} chars, "
                     f"F1: {ev1:.1f}%, F2: {ev2:.1f}%)")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Mind Perception Space: AI vs Human Characters",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def fig_factor_loadings(all_data):
    """Horizontal bar chart of F1 and F2 loadings, one row per model."""
    models_with_data = [d for d in all_data if d is not None]
    n = len(models_with_data)
    if n == 0:
        return None

    exp_caps = {c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"}

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)

    for row, d in enumerate(models_with_data):
        loadings = d["pca"]["rotated_loadings"]
        cap_keys = d["capacity_keys"]
        y = np.arange(len(cap_keys))
        labels = [nice_capacity(c) for c in cap_keys]
        colors = ["#2166ac" if c in exp_caps else "#b2182b" for c in cap_keys]

        for col, fi in enumerate([0, 1]):
            ax = axes[row, col]
            ax.barh(y, loadings[:, fi], color=colors, edgecolor="white", height=0.7)
            ax.set_xlabel("Loading")
            ax.set_title(f"{d['label']} — Factor {fi + 1}")
            ax.set_yticks(y)
            if col == 0:
                ax.set_yticklabels(labels, fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.axvline(0, color="gray", lw=0.5)
            ax.invert_yaxis()

    legend_elements = [
        Patch(facecolor="#2166ac", label="Experience capacity (Gray)"),
        Patch(facecolor="#b2182b", label="Agency capacity (Gray)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Varimax-Rotated Capacity Loadings (F1 and F2)", y=1.01)
    fig.tight_layout()
    return fig


def fig_ai_human_separation(all_data):
    """Bar chart: mean F1/F2 scores for AI vs Human group per model."""
    models_with_data = [d for d in all_data if d is not None and d["categorical"] is not None]
    n = len(models_with_data)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    axes = axes[0]

    for idx, d in enumerate(models_with_data):
        ax = axes[idx]
        cat = d["categorical"]["categorical"]
        factors_data = cat["factors"]
        n_show = min(len(factors_data), 4)  # show up to 4 factors

        x = np.arange(n_show)
        width = 0.35
        ai_means = [factors_data[i]["ai_mean"] for i in range(n_show)]
        human_means = [factors_data[i]["human_mean"] for i in range(n_show)]
        ai_stds = [factors_data[i]["ai_std"] for i in range(n_show)]
        human_stds = [factors_data[i]["human_std"] for i in range(n_show)]
        pvals = [factors_data[i]["p_value"] for i in range(n_show)]

        bars_ai = ax.bar(x - width / 2, ai_means, width, yerr=ai_stds,
                         color=C_AI, edgecolor="white", label="AI", alpha=0.85,
                         capsize=3)
        bars_human = ax.bar(x + width / 2, human_means, width, yerr=human_stds,
                            color=C_HUMAN, edgecolor="white", label="Human", alpha=0.85,
                            capsize=3)

        # Annotate p-values
        for i in range(n_show):
            p = pvals[i]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ymax = max(ai_means[i] + ai_stds[i], human_means[i] + human_stds[i])
            ax.text(x[i], ymax + 0.04, f"p={p:.3f}\n{sig}",
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold" if p < 0.05 else "normal",
                    color="#E53935" if p < 0.05 else "#666")

        ax.set_xticks(x)
        ax.set_xticklabels([f"F{i + 1}" for i in range(n_show)])
        ax.set_ylabel("Mean Factor Score (0-1)")
        ax.set_title(f"{d['label']}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(0, 1.4)

    fig.suptitle("AI vs Human Character Separation by Factor\n"
                 "(Mann-Whitney U test)", fontsize=14, y=1.03)
    fig.tight_layout()
    return fig


def fig_names_only_comparison(all_data):
    """Side-by-side scatter: full descriptions vs names-only for chat models."""
    models_with_names = [d for d in all_data
                         if d is not None and d.get("names_only_pca") is not None]
    n = len(models_with_names)
    if n == 0:
        return None

    fig, axes = plt.subplots(n, 2, figsize=(12, 5.5 * n), squeeze=False)

    for row, d in enumerate(models_with_names):
        # Full descriptions
        ax = axes[row, 0]
        scores = d["pca"]["factor_scores_01"]
        char_keys = d["character_keys"]
        is_ai, is_human = get_type_mask(char_keys)

        ax.scatter(scores[is_ai, 0], scores[is_ai, 1],
                   s=70, c=C_AI, edgecolor="white", linewidth=0.5,
                   label="AI", zorder=5)
        ax.scatter(scores[is_human, 0], scores[is_human, 1],
                   s=70, c=C_HUMAN, edgecolor="white", linewidth=0.5,
                   label="Human", zorder=5)
        for i, ck in enumerate(char_keys):
            ax.annotate(nice_character(ck), (scores[i, 0], scores[i, 1]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, ha="left", alpha=0.85)
        ax.set_xlabel("Factor 1 (0-1)")
        ax.set_ylabel("Factor 2 (0-1)")
        ax.set_title(f"{d['label']} — Full Descriptions")
        ax.legend(loc="best", fontsize=8)

        # Names only
        ax = axes[row, 1]
        nscores = d["names_only_pca"]["factor_scores_01"]
        nchar_keys = d["names_only_character_keys"]
        nis_ai, nis_human = get_type_mask(nchar_keys)

        ax.scatter(nscores[nis_ai, 0], nscores[nis_ai, 1],
                   s=70, c=C_AI, edgecolor="white", linewidth=0.5,
                   label="AI", zorder=5)
        ax.scatter(nscores[nis_human, 0], nscores[nis_human, 1],
                   s=70, c=C_HUMAN, edgecolor="white", linewidth=0.5,
                   label="Human", zorder=5)
        for i, ck in enumerate(nchar_keys):
            ax.annotate(nice_character(ck), (nscores[i, 0], nscores[i, 1]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, ha="left", alpha=0.85)
        ax.set_xlabel("Factor 1 (0-1)")
        ax.set_ylabel("Factor 2 (0-1)")
        ax.set_title(f"{d['label']} — Names Only")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Full Descriptions vs Names-Only Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ==========================================================================
# HTML SECTION BUILDERS
# ==========================================================================

def section_overview():
    """Section 1: Overview."""
    html = '<h2 id="overview">1. Overview</h2>\n'
    html += (
        '<p>This report extends the Gray, Gray, &amp; Wegner (2007) mind '
        'perception paradigm from the original 13 entities to 30 characters '
        '(15 AI systems + 15 human characters). The <strong>Human-AI Adaptation</strong> '
        'branch tests whether LLMs systematically differentiate AI from human '
        'characters in the mind perception space.</p>\n'
        '<p>Key questions:</p>\n'
        '<ul>\n'
        '<li>Do LLMs assign different mental capacity profiles to AI vs human characters?</li>\n'
        '<li>Does the two-factor (Experience/Agency) structure from Gray et al. emerge '
        'with 30 characters?</li>\n'
        '<li>Is AI-human separation driven by descriptions or by names alone?</li>\n'
        '</ul>\n'
    )
    return html


def section_stimuli():
    """Section 2: Stimuli."""
    return characters_stimuli_html(include_capacities=True)


def section_methods(all_data):
    """Section 3: Methods."""
    html = '<h2 id="methods">3. Methods</h2>\n'
    html += '<div class="method">\n'
    html += '<ol>\n'
    html += (
        '<li><strong>Pairwise comparisons:</strong> For each pair of characters '
        'and each of 18 mental capacities, the model judges which character is '
        'more capable. With 30 characters, this gives C(30,2) &times; 18 = 7,830 '
        'comparisons (fewer if some characters are excluded due to data issues).</li>\n'
    )
    html += (
        '<li><strong>PCA with varimax rotation:</strong> Following Gray et al., '
        'we compute the correlation matrix across capacities, run PCA, retain '
        'factors with eigenvalue &gt; 1 (minimum 2), and apply varimax rotation. '
        'Factor scores are rescaled to 0&ndash;1.</li>\n'
    )
    html += (
        '<li><strong>AI-Human separation test:</strong> Mann-Whitney U tests compare '
        'mean factor scores for AI characters vs human characters on each factor.</li>\n'
    )
    html += (
        '<li><strong>Names-only control (chat models):</strong> Repeats the pairwise '
        'paradigm using only character names (no descriptions), testing whether the '
        'model differentiates AI from human based on name alone.</li>\n'
    )
    html += '</ol>\n</div>\n'

    # Per-model data summary
    models_with_data = [d for d in all_data if d is not None]
    if models_with_data:
        html += '<h3>Data Summary</h3>\n'
        html += '<table>\n'
        html += ('<tr><th>Model</th><th>Characters</th><th>Factors</th>'
                 '<th>Eigenvalue 1</th><th>Eigenvalue 2</th>'
                 '<th>Names-Only</th></tr>\n')
        for d in models_with_data:
            eig = d["pca"]["eigenvalues"]
            has_names = "Yes" if d.get("names_only_pca") is not None else "No"
            html += (f'<tr><td>{d["label"]}</td>'
                     f'<td>{d["n_chars"]}</td>'
                     f'<td>{d["n_factors"]}</td>'
                     f'<td>{eig[0]:.2f}</td>'
                     f'<td>{eig[1]:.2f}</td>'
                     f'<td>{has_names}</td></tr>\n')
        html += '</table>\n'

        # Consistency stats
        for d in models_with_data:
            if d.get("consistency"):
                cs = d["consistency"]
                html += (f'<div class="stat"><strong>{d["label"]}:</strong> '
                         f'{cs["pct_consistent"]:.1f}% of repeated pairs were '
                         f'consistent (same direction), mean rating deviation = '
                         f'{cs["mean_deviation"]:.2f} across {cs["n_pairs_both"]} '
                         f'repeated pairs.</div>\n')

    return html


def section_pca_scatter(all_data):
    """Section 4: PCA scatter plots."""
    html = '<h2 id="pca-scatter">4. PCA Scatter: AI vs Human Characters</h2>\n'
    html += (
        '<p>Factor scores (rescaled to 0&ndash;1) for each character, plotted '
        'in the first two PCA factor dimensions. <span style="color:%s">Red = AI</span>, '
        '<span style="color:%s">blue = human</span>.</p>\n' % (C_AI, C_HUMAN)
    )

    fig = fig_pca_scatter(all_data)
    if fig is not None:
        b64 = fig_to_b64(fig)
        plt.close(fig)
        html += html_figure(
            b64,
            "Mind perception space for AI (red) and human (blue) characters. "
            "Each point is a character positioned by its factor scores from "
            "varimax-rotated PCA on 18 mental capacity pairwise ratings.",
            fig_num=1,
        )
    else:
        html += '<p class="warning">No data available for PCA scatter.</p>\n'

    return html


def section_loadings(all_data):
    """Section 5: Factor loadings."""
    html = '<h2 id="loadings">5. Factor Loadings</h2>\n'
    html += (
        '<p>Varimax-rotated loadings for the first two factors. Bars colored '
        'by the human factor assignment from Gray et al.: '
        '<span style="color:#2166ac">Experience</span> and '
        '<span style="color:#b2182b">Agency</span>.</p>\n'
    )

    fig = fig_factor_loadings(all_data)
    if fig is not None:
        b64 = fig_to_b64(fig)
        plt.close(fig)
        html += html_figure(
            b64,
            "Varimax-rotated loadings for F1 and F2. Blue bars are capacities "
            "assigned to Experience in Gray et al.; red bars are Agency capacities.",
            fig_num=2,
        )
    else:
        html += '<p class="warning">No data available for factor loadings.</p>\n'

    return html


def section_separation(all_data):
    """Section 6: AI-Human separation analysis."""
    html = '<h2 id="separation">6. AI-Human Separation</h2>\n'
    html += (
        '<p>Mean factor scores for AI vs human character groups, with '
        'Mann-Whitney U tests for statistical significance.</p>\n'
    )

    fig = fig_ai_human_separation(all_data)
    if fig is not None:
        b64 = fig_to_b64(fig)
        plt.close(fig)
        html += html_figure(
            b64,
            "Mean factor scores for AI (red) vs human (blue) character groups. "
            "Error bars show standard deviations. Significance from Mann-Whitney U.",
            fig_num=3,
        )
    else:
        html += '<p class="warning">No categorical analysis data available.</p>\n'

    # Detail table per model
    models_with_cat = [d for d in all_data
                       if d is not None and d.get("categorical") is not None]
    for d in models_with_cat:
        cat = d["categorical"]["categorical"]
        factors_data = cat["factors"]
        html += f'<h3>{d["label"]}</h3>\n'
        html += '<table>\n'
        html += ('<tr><th>Factor</th><th>AI Mean</th><th>Human Mean</th>'
                 '<th>Separation</th><th>U</th><th>p-value</th><th>Sig</th></tr>\n')
        for fd in factors_data:
            p = fd["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            sig_cls = ' class="sig"' if p < 0.05 else ''
            html += (f'<tr><td>F{fd["factor"]}</td>'
                     f'<td>{fd["ai_mean"]:.3f}</td>'
                     f'<td>{fd["human_mean"]:.3f}</td>'
                     f'<td>{fd["separation"]:.3f}</td>'
                     f'<td>{fd["mann_whitney_u"]:.0f}</td>'
                     f'<td{sig_cls}>{p:.4f}</td>'
                     f'<td{sig_cls}>{sig}</td></tr>\n')
        html += '</table>\n'

    return html


def section_anomalies(all_data):
    """Section 7: Anomalous characters."""
    html = '<h2 id="anomalies">7. Anomalies</h2>\n'
    html += (
        '<p>Characters that land on the &ldquo;wrong side&rdquo; of the AI-human '
        'divide: AI characters whose factor scores fall closer to the human group '
        'mean, or human characters whose scores fall closer to the AI group mean.</p>\n'
    )

    models_with_cat = [d for d in all_data
                       if d is not None and d.get("categorical") is not None]

    if not models_with_cat:
        html += '<p class="warning">No categorical analysis data available.</p>\n'
        return html

    for d in models_with_cat:
        cat = d["categorical"]["categorical"]
        anomalies = cat.get("anomalies", [])
        html += f'<h3>{d["label"]}</h3>\n'

        if not anomalies:
            html += '<p class="success">No anomalies detected.</p>\n'
            continue

        # Group anomalies by factor
        by_factor = {}
        for a in anomalies:
            fi = a["factor"]
            if fi not in by_factor:
                by_factor[fi] = []
            by_factor[fi].append(a)

        for fi in sorted(by_factor.keys()):
            html += f'<h4>Factor {fi}</h4>\n'
            html += '<table>\n'
            html += ('<tr><th>Character</th><th>Type</th><th>Score</th>'
                     '<th>Own Group Mean</th><th>Other Group Mean</th>'
                     '<th>Direction</th></tr>\n')
            for a in by_factor[fi]:
                name = nice_character(a["character"])
                char_type = a["type"].upper()
                score = a["score"]
                own_mean = a["own_group_mean"]
                other_mean = a["other_group_mean"]

                # Determine the direction of anomaly
                if a["type"] == "ai":
                    if own_mean > other_mean:
                        # AI group is higher; anomaly if this AI is below human mean
                        direction = "Below human mean" if score < other_mean else "Near human mean"
                    else:
                        direction = "Above human mean" if score > other_mean else "Near human mean"
                else:
                    if own_mean < other_mean:
                        # Human group is lower; anomaly if this human is above AI mean
                        direction = "Above AI mean" if score > other_mean else "Near AI mean"
                    else:
                        direction = "Below AI mean" if score < other_mean else "Near AI mean"

                html += (f'<tr><td>{name}</td><td>{char_type}</td>'
                         f'<td>{score:.3f}</td><td>{own_mean:.3f}</td>'
                         f'<td>{other_mean:.3f}</td><td>{direction}</td></tr>\n')
            html += '</table>\n'

    return html


def section_names_only(all_data):
    """Section 8: Names-only comparison."""
    html = '<h2 id="names-only">8. Names-Only Comparison</h2>\n'
    html += (
        '<p>For chat models, we repeat the pairwise paradigm using only character '
        'names (e.g., &ldquo;ChatGPT&rdquo; vs &ldquo;Sam&rdquo;) without '
        'descriptions. This tests whether AI-human separation is driven by '
        'the explicit description content or by name recognition alone.</p>\n'
    )

    models_with_names = [d for d in all_data
                         if d is not None and d.get("names_only_pca") is not None]

    if not models_with_names:
        html += '<p class="warning">No names-only data available for any model.</p>\n'
        return html

    fig = fig_names_only_comparison(all_data)
    if fig is not None:
        b64 = fig_to_b64(fig)
        plt.close(fig)
        html += html_figure(
            b64,
            "Left: PCA scatter from full descriptions. Right: PCA scatter from "
            "names-only. Comparing the degree of AI-human separation with and "
            "without character descriptions.",
            fig_num=4,
        )

    # Compare categorical results
    for d in models_with_names:
        html += f'<h3>{d["label"]}</h3>\n'

        # Full descriptions
        full_cat = d.get("categorical")
        names_cat = d.get("names_only_categorical")

        if full_cat and names_cat:
            full_factors = full_cat["categorical"]["factors"]
            names_factors = names_cat["categorical"]["factors"]

            html += '<table>\n'
            html += ('<tr><th>Factor</th>'
                     '<th colspan="2">Full Descriptions</th>'
                     '<th colspan="2">Names Only</th></tr>\n')
            html += ('<tr><th></th>'
                     '<th>Separation</th><th>p-value</th>'
                     '<th>Separation</th><th>p-value</th></tr>\n')

            n_compare = min(len(full_factors), len(names_factors), 4)
            for i in range(n_compare):
                ff = full_factors[i]
                nf = names_factors[i]
                fp = ff["p_value"]
                np_val = nf["p_value"]
                fsig_cls = ' class="sig"' if fp < 0.05 else ''
                nsig_cls = ' class="sig"' if np_val < 0.05 else ''
                html += (f'<tr><td>F{i + 1}</td>'
                         f'<td>{ff["separation"]:.3f}</td>'
                         f'<td{fsig_cls}>{fp:.4f}</td>'
                         f'<td>{nf["separation"]:.3f}</td>'
                         f'<td{nsig_cls}>{np_val:.4f}</td></tr>\n')
            html += '</table>\n'

        # Eigenvalue comparison
        full_eig = d["pca"]["eigenvalues"]
        names_eig = d["names_only_pca"]["eigenvalues"]
        html += '<div class="stat">\n'
        html += '<strong>Eigenvalue comparison:</strong><br>\n'
        html += (f'Full descriptions: {full_eig[0]:.2f}, {full_eig[1]:.2f} '
                 f'(first two eigenvalues)<br>\n')
        html += (f'Names only: {names_eig[0]:.2f}, {names_eig[1]:.2f} '
                 f'(first two eigenvalues)\n')
        html += '</div>\n'

        # Character overlap analysis
        full_chars = set(d["character_keys"])
        names_chars = set(d["names_only_character_keys"])
        shared = full_chars & names_chars
        html += (f'<p>Characters in common: {len(shared)} / '
                 f'{len(full_chars)} (full) / {len(names_chars)} (names-only)</p>\n')

    return html


def section_takeaways(all_data):
    """Section 9: Key takeaways (auto-generated)."""
    html = '<h2 id="takeaways">9. Key Takeaways</h2>\n'

    models_with_data = [d for d in all_data if d is not None]
    if not models_with_data:
        html += '<p class="warning">No data available.</p>\n'
        return html

    takeaways = []

    for d in models_with_data:
        label = d["label"]

        # Factor structure
        eig = d["pca"]["eigenvalues"]
        n_factors = d["n_factors"]
        if n_factors == 2:
            takeaways.append(
                f'<strong>{label}:</strong> Clean two-factor structure emerges '
                f'(eigenvalues: {eig[0]:.1f}, {eig[1]:.1f}), mirroring the '
                f'Experience/Agency dimensions from Gray et al.'
            )
        else:
            takeaways.append(
                f'<strong>{label}:</strong> {n_factors}-factor structure '
                f'(eigenvalues: {", ".join(f"{e:.1f}" for e in eig[:n_factors])}). '
                f'The mind perception space is more fragmented than the classic '
                f'two-factor model.'
            )

        # Separation
        if d.get("categorical"):
            cat = d["categorical"]["categorical"]
            sig_factors = [fd for fd in cat["factors"] if fd["p_value"] < 0.05]
            if sig_factors:
                factor_labels = ", ".join(f'F{fd["factor"]} (p={fd["p_value"]:.3f})'
                                          for fd in sig_factors)
                takeaways.append(
                    f'<strong>{label}:</strong> Significant AI-human separation on '
                    f'{factor_labels}. The model distinguishes AI from human '
                    f'characters in its mind perception judgments.'
                )
            else:
                takeaways.append(
                    f'<strong>{label}:</strong> No significant AI-human separation '
                    f'on any factor (all p &gt; .05). The model does not reliably '
                    f'distinguish AI from human characters.'
                )

        # Names-only
        if d.get("names_only_categorical"):
            ncat = d["names_only_categorical"]["categorical"]
            nsig = [fd for fd in ncat["factors"] if fd["p_value"] < 0.05]
            if nsig:
                nsig_labels = ", ".join(f"F{fd['factor']}" for fd in nsig)
                takeaways.append(
                    f'<strong>{label} (names-only):</strong> AI-human separation '
                    f'persists with names alone on {nsig_labels}. '
                    f'Name recognition alone is sufficient for differentiation.'
                )
            else:
                takeaways.append(
                    f'<strong>{label} (names-only):</strong> No significant '
                    f'separation with names alone. Descriptions are necessary '
                    f'for the model to differentiate AI from human.'
                )

    html += '<div class="interpret">\n<ul>\n'
    for t in takeaways:
        html += f'<li>{t}</li>\n'
    html += '</ul>\n</div>\n'

    return html


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    print("=" * 60)
    print("Human-AI Adaptation: Cross-Model Summary Report")
    print("=" * 60)

    # Load data for all models
    all_data = []
    for model_key in VALID_MODELS:
        print(f"\nLoading {model_key}...")
        try:
            d = load_model_data(model_key)
            if d is not None:
                print(f"  Loaded: {d['n_chars']} characters, "
                      f"{d['n_factors']} factors, "
                      f"names-only: {d.get('names_only_pca') is not None}")
            else:
                print(f"  No data found.")
            all_data.append(d)
        except Exception as e:
            print(f"  Error loading: {e}")
            all_data.append(None)

    n_loaded = sum(1 for d in all_data if d is not None)
    print(f"\nLoaded data for {n_loaded} / {len(VALID_MODELS)} models.")

    if n_loaded == 0:
        print("ERROR: No data available for any model. Exiting.")
        return

    # Build report
    print("\nGenerating report...")

    sections = [
        {"id": "overview", "label": "1. Overview"},
        {"id": "stimuli", "label": "2. Stimuli"},
        {"id": "methods", "label": "3. Methods"},
        {"id": "pca-scatter", "label": "4. PCA Scatter: AI vs Human"},
        {"id": "loadings", "label": "5. Factor Loadings"},
        {"id": "separation", "label": "6. AI-Human Separation"},
        {"id": "anomalies", "label": "7. Anomalies"},
        {"id": "names-only", "label": "8. Names-Only Comparison"},
        {"id": "takeaways", "label": "9. Key Takeaways"},
    ]

    html = build_cross_model_header(
        "Experiment 4: Human-AI Adaptation — Cross-Model Summary"
    )
    html += build_toc(sections)

    html += section_overview()
    html += section_stimuli()
    html += section_methods(all_data)
    html += section_pca_scatter(all_data)
    html += section_loadings(all_data)
    html += section_separation(all_data)
    html += section_anomalies(all_data)
    html += section_names_only(all_data)
    html += section_takeaways(all_data)

    html += '<hr>\n'
    html += '<p style="font-size:0.85em; color:#888;">Rachel C. Metzgar &middot; Mar 2026</p>\n'
    html += build_html_footer()

    # Write output
    out_path = ensure_dir(COMPARISONS_DIR) / "human_ai_summary.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
