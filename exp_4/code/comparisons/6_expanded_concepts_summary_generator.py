#!/usr/bin/env python3
"""
Cross-Model Summary: Expanded Mental Concepts Branch

Bridges Exp 3 concept vectors into the mind perception space.
Compares behavioral PCA, activation RSA, per-concept RSA, and
alignment with Exp 3 concept vectors across all available models.

Reads data from:
  results/{model}/expanded_mental_concepts/behavior/pca/data/
  results/{model}/expanded_mental_concepts/internals/rsa/data/
  results/{model}/expanded_mental_concepts/internals/concept_rsa/{concept}/data/
  results/{model}/expanded_mental_concepts/internals/concept_rsa/data/
  results/{model}/expanded_mental_concepts/internals/contrast_alignment/data/
  results/{model}/expanded_mental_concepts/internals/standalone_alignment/data/

Usage:
    python comparisons/6_expanded_concepts_summary_generator.py

Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# -- Imports from project ----------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, COMPARISONS_DIR, VALID_MODELS, MODELS, ensure_dir, set_model
from utils.report_utils import (
    REPORT_CSS, build_cross_model_header, build_html_footer, build_toc,
    fig_to_b64, html_figure, MODEL_COLORS, MODEL_LABELS, ALL_MODELS,
    expanded_concepts_stimuli_html, sort_models, GRID_NCOLS, make_model_grid,
    model_row_td, format_p_cell,
    methodology_primer_html, neural_methods_primer_html,
    expanded_concepts_primer_html,
)
from entities.characters import CHARACTER_INFO, AI_CHARACTERS, HUMAN_CHARACTERS


# -- Constants ---------------------------------------------------------------
CONCEPT_KEYS = [
    "agency", "ai", "animacy", "attention", "beliefs", "biological",
    "cognitive", "desires", "embodiment", "emotions", "expertise",
    "formality", "goals", "helpfulness", "human", "intentions",
    "phenomenology", "prediction", "roles", "shapes", "social",
]

CONTROL_CONCEPTS = {"shapes"}
MENTAL_CONCEPTS = set(CONCEPT_KEYS) - CONTROL_CONCEPTS

BRANCH = "human_ai_characters"

# -- Style -------------------------------------------------------------------
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

C_AI = "#e41a1c"      # red for AI
C_HUMAN = "#377eb8"    # blue for human


# ============================================================================
# DATA LOADING
# ============================================================================

def _results_base(model):
    """Return base path for a model's expanded_mental_concepts results."""
    return ROOT_DIR / "results" / model / BRANCH


def load_pca_data(model):
    """Load behavioral PCA data for a model. Returns dict or None."""
    base = _results_base(model) / "behavior" / "pca" / "data"
    npz_path = base / "pairwise_pca_results.npz"
    json_path = base / "pairwise_categorical_analysis.json"
    if not npz_path.exists():
        return None
    data = {}
    npz = np.load(npz_path, allow_pickle=True)
    data["rotated_loadings"] = npz["rotated_loadings"]
    data["factor_scores_01"] = npz["factor_scores_01"]
    data["eigenvalues"] = npz["eigenvalues"]
    data["explained_var_ratio"] = npz["explained_var_ratio"]
    data["character_keys"] = list(npz["character_keys"])
    data["concept_keys"] = list(npz["concept_keys"])
    if json_path.exists():
        with open(json_path) as f:
            data["categorical"] = json.load(f)
    return data


def load_activation_rsa(model):
    """Load activation RSA results. Returns dict or None."""
    base = _results_base(model) / "internals" / "rsa" / "data"
    json_path = base / "rsa_results.json"
    rdm_path = base / "rdm_cosine_per_layer.npz"
    if not json_path.exists():
        return None
    data = {}
    with open(json_path) as f:
        rsa = json.load(f)
    # RSA results may be keyed by "categorical" or by "combined"/"experience"/"agency"
    if "categorical" in rsa:
        data["rsa_layers"] = rsa["categorical"]
    elif "combined" in rsa:
        data["rsa_layers"] = rsa["combined"]
    else:
        # Assume top-level list
        data["rsa_layers"] = rsa if isinstance(rsa, list) else list(rsa.values())[0]
    if rdm_path.exists():
        rdm_npz = np.load(rdm_path, allow_pickle=True)
        data["model_rdm"] = rdm_npz["model_rdm"]
        data["categorical_rdm"] = rdm_npz["categorical_rdm"]
        data["character_keys"] = list(rdm_npz["character_keys"])
    return data


def load_concept_rsa(model):
    """Load per-concept RSA results. Returns dict of concept -> layer results, or None."""
    base = _results_base(model) / "internals" / "concept_rsa"
    results = {}
    for concept in CONCEPT_KEYS:
        json_path = base / concept / "data" / "rsa_results.json"
        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            # Per-concept RSA is a list of layer dicts
            if isinstance(raw, list):
                results[concept] = raw
            elif isinstance(raw, dict):
                if "categorical" in raw:
                    results[concept] = raw["categorical"]
                else:
                    results[concept] = list(raw.values())[0] if raw else []
    # Also load cross-concept summary if available
    summary_path = base / "data" / "cross_concept_rsa_summary.json"
    summary = None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    return results if results else None, summary


def load_contrast_alignment(model):
    """Load contrast alignment results. Returns dict or None."""
    path = _results_base(model) / "internals" / "contrast_alignment" / "data" / "alignment_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_standalone_alignment(model):
    """Load standalone alignment results. Returns dict or None."""
    path = _results_base(model) / "internals" / "standalone_alignment" / "data" / "alignment_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_peak_rho(layer_entries):
    """Find the peak rho (max absolute value) from a list of layer dicts."""
    best_rho = None
    best_entry = None
    for entry in layer_entries:
        rho = entry.get("rho")
        if rho is None or (isinstance(rho, float) and np.isnan(rho)):
            continue
        if best_rho is None or abs(rho) > abs(best_rho):
            best_rho = rho
            best_entry = entry
    return best_entry


def get_n_sig_layers(layer_entries, alpha=0.05):
    """Count layers with p_fdr < alpha (or p_value if no p_fdr)."""
    count = 0
    for entry in layer_entries:
        p = entry.get("p_fdr") if entry.get("p_fdr") is not None else entry.get("p_value")
        if p is not None and not (isinstance(p, float) and np.isnan(p)) and p < alpha:
            count += 1
    return count


# ============================================================================
# FIGURE GENERATORS
# ============================================================================

def make_pca_scatter(all_pca, fig_num):
    """Section 4a: F1 x F2 scatter, one subplot per model."""
    models_list = [m for m in all_pca if all_pca[m] is not None]
    if not models_list:
        return ""
    positions, ordered, nrows, ncols, _ = make_model_grid(models_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)
    for idx, model in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        data = all_pca[model]
        scores = data["factor_scores_01"]
        char_keys = data["character_keys"]
        var_explained = data["explained_var_ratio"]
        for i, key in enumerate(char_keys):
            ctype = CHARACTER_INFO.get(key, {}).get("type", "unknown")
            color = C_AI if ctype == "ai" else C_HUMAN
            marker = "s" if ctype == "ai" else "o"
            ax.scatter(scores[i, 0], scores[i, 1], c=color, marker=marker,
                       s=50, alpha=0.8, edgecolors="white", linewidths=0.5)
            name = CHARACTER_INFO.get(key, {}).get("name", key)
            ax.annotate(name, (scores[i, 0], scores[i, 1]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(f"F1 ({var_explained[0]*100:.1f}%)")
        ax.set_ylabel(f"F2 ({var_explained[1]*100:.1f}%)")
        ax.set_title(MODEL_LABELS.get(model, model))
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=C_AI, label="AI"),
            Patch(facecolor=C_HUMAN, label="Human"),
        ], loc="best", framealpha=0.8)
    fig.suptitle("Behavioral PCA: Character Positions (F1 x F2)", fontsize=14, y=1.02)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Character positions in the first two PCA factors of concept-guided "
               "pairwise comparison scores. AI characters (red squares) vs human "
               "characters (blue circles).")
    return html_figure(b64, caption, fig_num=fig_num, alt="PCA scatter")


def make_loading_comparison(all_pca, fig_num):
    """Section 4b: Horizontal bar chart of F1 and F2 loadings per model."""
    models_list = [m for m in all_pca if all_pca[m] is not None]
    if not models_list:
        return ""
    height = max(6, len(CONCEPT_KEYS) * 0.3)
    positions, ordered, nrows, ncols, _ = make_model_grid(models_list)
    actual_ncols = ncols * 2  # F1 + F2 per model
    fig, axes = plt.subplots(nrows, actual_ncols,
                             figsize=(4 * actual_ncols, height * nrows),
                             squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)
    for idx, model in enumerate(ordered):
        data = all_pca[model]
        loadings = data["rotated_loadings"]
        concept_keys = data["concept_keys"]
        var_explained = data["explained_var_ratio"]
        row, col = positions[idx]
        col_base = col * 2
        for fi, factor_idx in enumerate([0, 1]):
            ax = axes[row, col_base + fi]
            ax.set_visible(True)
            vals = loadings[:, factor_idx]
            sorted_idx = np.argsort(vals)
            sorted_keys = [concept_keys[i] for i in sorted_idx]
            sorted_vals = vals[sorted_idx]
            colors = ["#d32f2f" if v < 0 else "#1976d2" for v in sorted_vals]
            ax.barh(range(len(sorted_keys)), sorted_vals, color=colors, height=0.7)
            ax.set_yticks(range(len(sorted_keys)))
            ax.set_yticklabels(sorted_keys, fontsize=7)
            ax.set_xlabel("Loading")
            ax.set_title(f"{MODEL_LABELS.get(model, model)}\nF{factor_idx+1} "
                         f"({var_explained[factor_idx]*100:.1f}%)", fontsize=10)
            ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    fig.suptitle("PCA Factor Loadings by Concept Dimension", fontsize=14, y=1.02)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Rotated loadings on F1 and F2 for each concept dimension. "
               "Blue bars = positive loadings, red bars = negative loadings.")
    return html_figure(b64, caption, fig_num=fig_num, alt="PCA loadings")


def make_group_separation(all_pca, fig_num):
    """Section 4c: Bar chart of mean factor scores by AI/human group."""
    models_list = [m for m in all_pca
                   if all_pca[m] is not None and "categorical" in all_pca[m]]
    if not models_list:
        return ""
    n_factors = 4
    positions, ordered, nrows, ncols, _ = make_model_grid(models_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)
    for idx, model in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        data = all_pca[model]
        cat = data["categorical"].get("categorical", data["categorical"])
        factors = cat.get("factors", cat) if isinstance(cat, dict) else cat
        if isinstance(factors, dict):
            factors = factors.get("factors", [])
        factor_nums = []
        ai_means = []
        human_means = []
        ai_stds = []
        human_stds = []
        p_values = []
        for f_entry in factors:
            factor_nums.append(f_entry["factor"])
            ai_means.append(f_entry["ai_mean"])
            human_means.append(f_entry["human_mean"])
            ai_stds.append(f_entry["ai_std"])
            human_stds.append(f_entry["human_std"])
            p_values.append(f_entry["p_value"])
        x = np.arange(len(factor_nums))
        width = 0.35
        ax.bar(x - width / 2, ai_means, width, yerr=ai_stds, label="AI",
               color=C_AI, alpha=0.8, capsize=3)
        ax.bar(x + width / 2, human_means, width, yerr=human_stds, label="Human",
               color=C_HUMAN, alpha=0.8, capsize=3)
        for i, p in enumerate(p_values):
            if p < 0.001:
                label = "***"
            elif p < 0.01:
                label = "**"
            elif p < 0.05:
                label = "*"
            else:
                label = "ns"
            y_max = max(ai_means[i] + ai_stds[i], human_means[i] + human_stds[i])
            ax.text(i, y_max + 0.05, label, ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{fn}" for fn in factor_nums])
        ax.set_ylabel("Mean Factor Score (0-1)")
        ax.set_title(MODEL_LABELS.get(model, model))
        ax.legend(loc="best", framealpha=0.8)
        ax.set_ylim(0, 1.15)
    fig.suptitle("AI vs Human Group Separation by Factor", fontsize=14, y=1.02)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Mean factor scores (0-1 scaled) for AI and human character groups. "
               "Error bars show standard deviation. Significance from Mann-Whitney U: "
               "*** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, ns = not significant.")
    return html_figure(b64, caption, fig_num=fig_num, alt="Group separation")


def make_rsa_layerwise(all_rsa, fig_num):
    """Section 5a: Bar-chart grid of layerwise RSA, one panel per model.

    Colored bars = FDR-significant (q < .05), gray = non-significant.
    Also includes an overlay line plot as a second figure.
    """
    from utils.report_utils import apply_fdr

    sorted_keys = sort_models([m for m in all_rsa if all_rsa[m] is not None])
    models_with_data = [(m, all_rsa[m]) for m in sorted_keys]
    if not models_with_data:
        return ""

    # Apply FDR to each model's RSA layers
    for model, data in models_with_data:
        apply_fdr(data["rsa_layers"])

    model_keys = [m for m, _ in models_with_data]

    # Shared y-range
    all_rhos = []
    for _, data in models_with_data:
        for r in data["rsa_layers"]:
            rho = r.get("rho")
            if rho is not None and not (isinstance(rho, float) and np.isnan(rho)):
                all_rhos.append(rho)
    y_min = min(all_rhos) - 0.05 if all_rhos else -0.1
    y_max = max(all_rhos) + 0.1 if all_rhos else 0.5

    # Bar chart grid
    positions, ordered, nrows, ncols, _ = make_model_grid(model_keys)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)

    data_by_model = {m: d for m, d in models_with_data}
    for idx, model in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        data = data_by_model[model]
        rsa_layers = data["rsa_layers"]
        base_color = MODEL_COLORS.get(model, "#333")

        layers = [r["layer"] for r in rsa_layers]
        rhos = [r["rho"] if r.get("rho") is not None and not (isinstance(r["rho"], float) and np.isnan(r["rho"])) else 0.0
                for r in rsa_layers]
        qvals = [r.get("q_fdr", 1.0) for r in rsa_layers]

        colors = [base_color if q < 0.05 else "#cccccc" for q in qvals]
        ax.bar(layers, rhos, color=colors, edgecolor="white", width=0.8)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(y_min, y_max)

        # Peak annotation
        peak = get_peak_rho(rsa_layers)
        n_sig = sum(1 for q in qvals if q < 0.05)
        n_total = len(layers)
        if peak is not None:
            peak_rho = peak["rho"]
            peak_q = peak.get("q_fdr", 1.0)
            ax.set_title(f"{MODEL_LABELS.get(model, model)}\n{n_sig}/{n_total} layers q < .05, "
                         f"peak layer {peak['layer']} (ρ={peak_rho:.3f})", fontsize=9)
            ax.annotate(
                f"ρ={peak_rho:.3f}\nq={peak_q:.3f}",
                (peak["layer"], peak_rho),
                textcoords="offset points", xytext=(12, 8), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )
        else:
            ax.set_title(MODEL_LABELS.get(model, model), fontsize=9)

    fig.suptitle("Activation RSA: AI/Human Categorical RDM Correlation", fontsize=14)
    fig.tight_layout()
    b64_bars = fig_to_b64(fig)
    plt.close(fig)
    html = html_figure(
        b64_bars,
        "Layerwise categorical RSA for each model. Colored bars indicate "
        "FDR-significant layers (q &lt; .05); gray bars are non-significant. "
        "Peak layer annotated with ρ and q values.",
        fig_num=fig_num, alt="RSA bar charts"
    )

    # Overlay line plot (all models on one axis)
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, data in models_with_data:
        rsa_layers = data["rsa_layers"]
        layer_nums = [r["layer"] for r in rsa_layers]
        rhos = [r["rho"] if r.get("rho") is not None and not (isinstance(r["rho"], float) and np.isnan(r["rho"])) else 0.0
                for r in rsa_layers]
        ax.plot(layer_nums, rhos, "-o", markersize=3,
                color=MODEL_COLORS.get(model, "#333"),
                label=MODEL_LABELS.get(model, model), linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Activation RSA: All Models Overlaid")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(loc="best", framealpha=0.8, fontsize=8)
    plt.tight_layout()
    b64_overlay = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64_overlay,
        "Overlay of categorical RSA across all models for direct "
        "comparison of peak layer locations and magnitudes.",
        fig_num=fig_num + 1, alt="RSA overlay"
    )

    return html


def make_peak_rsa_table(all_rsa):
    """Section 5b: Peak RSA summary table."""
    sorted_keys = sort_models([m for m in all_rsa if all_rsa[m] is not None])
    models_with_data = [(m, all_rsa[m]) for m in sorted_keys]
    if not models_with_data:
        return ""
    html = '<table>\n'
    html += '<tr><th>Model</th><th>Peak Layer</th><th>Peak rho</th><th>p (at peak)</th><th>Sig Layers (p&lt;0.05)</th></tr>\n'
    for model, data in models_with_data:
        peak = get_peak_rho(data["rsa_layers"])
        n_sig = get_n_sig_layers(data["rsa_layers"])
        if peak is not None:
            p_val = peak.get("p_fdr") if peak.get("p_fdr") is not None else peak.get("p_value")
            p_str = f"{p_val:.2e}" if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)) else "N/A"
            sig_class = ' class="sig"' if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)) and p_val < 0.05 else ""
            html += (f'<tr>{model_row_td(model)}'
                     f'<td>{peak["layer"]}</td>'
                     f'<td{sig_class}>{peak["rho"]:.4f}</td>'
                     f'<td{sig_class}>{p_str}</td>'
                     f'<td>{n_sig}</td></tr>\n')
    html += '</table>\n'
    return html


def make_rdm_heatmaps(all_rsa, fig_num):
    """Section 5c: RDM heatmaps at peak layer, one per model."""
    models_list = [m for m in all_rsa
                   if all_rsa[m] is not None and "model_rdm" in all_rsa[m]]
    if not models_list:
        return ""
    positions, ordered, nrows, ncols, _ = make_model_grid(models_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows),
                             squeeze=False, layout="constrained")
    for ax in axes.flatten():
        ax.set_visible(False)

    # Compute shared vmin/vmax across all panels for comparable colorscale
    all_rdm_vals = []
    for model in ordered:
        data = all_rsa[model]
        peak = get_peak_rho(data["rsa_layers"])
        if peak is not None:
            all_rdm_vals.append(data["model_rdm"][peak["layer"]])
    if all_rdm_vals:
        shared_vmin = min(r.min() for r in all_rdm_vals)
        shared_vmax = max(r.max() for r in all_rdm_vals)
    else:
        shared_vmin, shared_vmax = 0, 1

    visible_axes = []
    im = None
    for idx, model in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        visible_axes.append(ax)
        data = all_rsa[model]
        peak = get_peak_rho(data["rsa_layers"])
        if peak is None:
            ax.set_title(f"{MODEL_LABELS.get(model, model)}\n(no valid peak)")
            continue
        peak_layer = peak["layer"]
        rdm = data["model_rdm"][peak_layer]
        char_keys = data["character_keys"]
        # Sort: AI first, then human
        ai_idx = [i for i, k in enumerate(char_keys) if CHARACTER_INFO.get(k, {}).get("type") == "ai"]
        hu_idx = [i for i, k in enumerate(char_keys) if CHARACTER_INFO.get(k, {}).get("type") == "human"]
        order = ai_idx + hu_idx
        sorted_rdm = rdm[np.ix_(order, order)]
        sorted_names = [CHARACTER_INFO.get(char_keys[i], {}).get("name", char_keys[i]) for i in order]
        im = ax.imshow(sorted_rdm, cmap="RdBu_r", aspect="equal",
                       vmin=shared_vmin, vmax=shared_vmax)
        ax.set_xticks(range(len(sorted_names)))
        ax.set_xticklabels(sorted_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=6)
        # Draw AI/human boundary
        boundary = len(ai_idx) - 0.5
        ax.axhline(boundary, color="black", linewidth=1)
        ax.axvline(boundary, color="black", linewidth=1)
        ax.set_title(f"{MODEL_LABELS.get(model, model)}\nLayer {peak_layer} "
                     f"(rho={peak['rho']:.3f})")
    if im is not None:
        fig.colorbar(im, ax=visible_axes, shrink=0.6, label="Cosine distance")
    fig.suptitle("Activation RDMs at Peak RSA Layer", fontsize=14)
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Representational dissimilarity matrices (cosine distance) at each model's "
               "peak RSA layer. Characters sorted with AI first (top-left block), then "
               "human. Black lines mark the AI/human boundary.")
    return html_figure(b64, caption, fig_num=fig_num, alt="RDM heatmaps")


def make_concept_rsa_heatmap(all_concept_rsa, fig_num):
    """Section 6a: Heatmap of peak rho by concept x model."""
    sorted_keys = sort_models([m for m in all_concept_rsa if all_concept_rsa[m] is not None])
    models_with_data = [(m, all_concept_rsa[m]) for m in sorted_keys]
    if not models_with_data:
        return ""
    concepts_sorted = sorted(CONCEPT_KEYS)
    model_keys = [m for m, _ in models_with_data]
    n_concepts = len(concepts_sorted)
    n_models = len(model_keys)
    matrix = np.full((n_concepts, n_models), np.nan)
    sig_mask = np.zeros((n_concepts, n_models), dtype=bool)
    for j, (model, concept_data) in enumerate(models_with_data):
        for i, concept in enumerate(concepts_sorted):
            if concept in concept_data:
                peak = get_peak_rho(concept_data[concept])
                if peak is not None:
                    matrix[i, j] = peak["rho"]
                    p_val = peak.get("p_fdr") if peak.get("p_fdr") is not None else peak.get("p_value")
                    if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)) and p_val < 0.05:
                        sig_mask[i, j] = True
    fig, ax = plt.subplots(figsize=(max(4, 2 * n_models + 1), max(6, n_concepts * 0.35)))
    vmax = np.nanmax(np.abs(matrix)) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, cmap="RdYlBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    # Mark significant cells
    for i in range(n_concepts):
        for j in range(n_models):
            if sig_mask[i, j]:
                ax.text(j, i, "*", ha="center", va="center", fontsize=12,
                        fontweight="bold", color="black")
            if not np.isnan(matrix[i, j]):
                ax.text(j, i + 0.3, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="gray")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_keys], rotation=45, ha="right")
    ax.set_yticks(range(n_concepts))
    ax.set_yticklabels(concepts_sorted, fontsize=8)
    ax.set_title("Per-Concept RSA: Peak Spearman rho")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Peak rho")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Peak Spearman rho between each concept's activation RDM and the "
               "categorical (AI vs human) RDM, by model. * marks FDR-corrected "
               "significance (p &lt; 0.05). Values shown below.")
    return html_figure(b64, caption, fig_num=fig_num, alt="Concept RSA heatmap")


def make_concept_rsa_tables(all_concept_rsa):
    """Section 6b: Combined cross-model concept RSA summary table.

    Shows peak rho for each concept across all models in a single table,
    with concepts sorted by mean |rho| across models.
    """
    sorted_keys = sort_models([m for m in all_concept_rsa if all_concept_rsa[m] is not None])
    models_with_data = [(m, all_concept_rsa[m]) for m in sorted_keys]
    if not models_with_data:
        return ""

    # Gather peak rho per concept per model
    concept_peaks = {}  # concept -> {model: {rho, layer, p}}
    for concept in CONCEPT_KEYS:
        concept_peaks[concept] = {}
        for model, concept_data in models_with_data:
            if concept in concept_data:
                peak = get_peak_rho(concept_data[concept])
                if peak is not None:
                    p_val = peak.get("p_fdr") if peak.get("p_fdr") is not None else peak.get("p_value")
                    concept_peaks[concept][model] = {
                        "rho": peak["rho"], "layer": peak["layer"], "p": p_val,
                    }

    # Sort concepts by mean |rho| across models (descending)
    def mean_abs_rho(concept):
        vals = [v["rho"] for v in concept_peaks[concept].values()]
        return np.mean(np.abs(vals)) if vals else 0.0

    concepts_sorted = sorted(CONCEPT_KEYS, key=mean_abs_rho, reverse=True)

    # Build combined table: concepts as rows, models as columns
    model_keys = [m for m, _ in models_with_data]
    html = '<table>\n'
    html += '<tr><th>Concept</th><th>Category</th>'
    for m in model_keys:
        html += f'<th>{MODEL_LABELS.get(m, m)}</th>'
    html += '<th>Mean |ρ|</th></tr>\n'

    for concept in concepts_sorted:
        cat_label = "Control" if concept in CONTROL_CONCEPTS else "Mental"
        html += f'<tr><td>{concept}</td><td>{cat_label}</td>'
        rhos_for_mean = []
        for m in model_keys:
            if m in concept_peaks[concept]:
                info = concept_peaks[concept][m]
                rho = info["rho"]
                p = info["p"]
                rhos_for_mean.append(abs(rho))
                is_sig = (p is not None
                          and not (isinstance(p, float) and np.isnan(p))
                          and p < 0.05)
                sig_cls = ' class="sig"' if is_sig else ''
                star = "*" if is_sig else ""
                html += f'<td{sig_cls}>{rho:.3f}{star}</td>'
            else:
                html += '<td>--</td>'
        mean_rho = np.mean(rhos_for_mean) if rhos_for_mean else 0.0
        html += f'<td><strong>{mean_rho:.3f}</strong></td>'
        html += '</tr>\n'

    html += '</table>\n'
    html += ('<p style="font-size:0.85em; color:#555;">'
             'Peak Spearman ρ between concept-projected character RDM and '
             'categorical (AI/human) RDM. * = FDR-corrected p &lt; 0.05. '
             'Sorted by mean |ρ| across models (descending).</p>\n')

    return html


def make_concept_consistency(all_concept_rsa):
    """Section 6c: Cross-model consistency for per-concept RSA."""
    sorted_keys = sort_models([m for m in all_concept_rsa if all_concept_rsa[m] is not None])
    models_with_data = [(m, all_concept_rsa[m]) for m in sorted_keys]
    if len(models_with_data) < 2:
        return '<p>Cross-model consistency requires data from at least 2 models.</p>\n'
    # For each concept, gather peak rhos across models
    concept_peaks = {}
    for concept in CONCEPT_KEYS:
        rhos = []
        for model, concept_data in models_with_data:
            if concept in concept_data:
                peak = get_peak_rho(concept_data[concept])
                if peak is not None:
                    rhos.append(peak["rho"])
        if len(rhos) >= 2:
            concept_peaks[concept] = rhos
    if not concept_peaks:
        return '<p>Not enough overlapping concept data across models.</p>\n'
    # Sort by mean peak rho
    ranked = sorted(concept_peaks.items(), key=lambda x: np.mean(x[1]), reverse=True)
    html = '<table>\n'
    html += '<tr><th>Concept</th><th>Mean Peak rho</th><th>Range</th><th>Category</th></tr>\n'
    for concept, rhos in ranked:
        mean_rho = np.mean(rhos)
        range_str = f"[{min(rhos):.3f}, {max(rhos):.3f}]"
        cat_label = "Control" if concept in CONTROL_CONCEPTS else "Mental"
        html += (f'<tr><td>{concept}</td><td>{mean_rho:.4f}</td>'
                 f'<td>{range_str}</td><td>{cat_label}</td></tr>\n')
    html += '</table>\n'
    # Identify consistently high / low
    consistently_high = [c for c, rhos in ranked if np.mean(rhos) > 0.3]
    consistently_low = [c for c, rhos in ranked if abs(np.mean(rhos)) < 0.1]
    if consistently_high:
        html += '<div class="success">\n'
        html += f'<strong>Consistently high across models (mean rho &gt; 0.3):</strong> {", ".join(consistently_high)}\n'
        html += '</div>\n'
    if consistently_low:
        html += '<div class="warning">\n'
        html += f'<strong>Consistently low across models (|mean rho| &lt; 0.1):</strong> {", ".join(consistently_low)}\n'
        html += '</div>\n'
    return html


def _extract_concept_peak_alignment(alignment_data, field="combined_alignment"):
    """Extract peak alignment per concept from alignment JSON (contrast or standalone)."""
    if alignment_data is None:
        return {}
    concepts_list = alignment_data.get("concepts", [])
    results = {}
    for entry in concepts_list:
        name = entry.get("name", "")
        layers = entry.get("layers", [])
        best_val = 0.0
        best_layer = 0
        for layer_entry in layers:
            val = layer_entry.get(field, layer_entry.get("mean_sim_human", 0.0))
            if val is not None and abs(val) > abs(best_val):
                best_val = val
                best_layer = layer_entry.get("layer", 0)
        results[name] = {"peak_alignment": best_val, "peak_layer": best_layer,
                         "dim_id": entry.get("dim_id"),
                         "is_control": entry.get("is_control", False),
                         "is_entity_framed": entry.get("is_entity_framed", True)}
    return results


def make_contrast_alignment_chart(all_contrast, fig_num):
    """Section 7a: Bar chart of peak contrast alignment per concept, grouped by model."""
    sorted_keys = sort_models([m for m in all_contrast if all_contrast[m] is not None])
    models_with_data = [(m, all_contrast[m]) for m in sorted_keys]
    if not models_with_data:
        return ""
    # Extract peak alignment per concept per model
    all_peaks = {}
    for model, data in models_with_data:
        all_peaks[model] = _extract_concept_peak_alignment(data, field="combined_alignment")
    # Gather all concept names
    all_concept_names = set()
    for peaks in all_peaks.values():
        all_concept_names.update(peaks.keys())
    concept_names_sorted = sorted(all_concept_names)
    if not concept_names_sorted:
        return ""
    n_models = len(models_with_data)
    x = np.arange(len(concept_names_sorted))
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(max(10, len(concept_names_sorted) * 0.6), 5))
    for j, (model, _) in enumerate(models_with_data):
        peaks = all_peaks[model]
        vals = [peaks.get(c, {}).get("peak_alignment", 0.0) for c in concept_names_sorted]
        offset = (j - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS.get(model, "#333"), alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names_sorted, rotation=90, fontsize=8)
    ax.set_ylabel("Peak Cosine Similarity")
    ax.set_title("Contrast Alignment: Peak Cosine Similarity per Concept")
    ax.legend(loc="best", framealpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Peak cosine similarity between concept contrast vectors (Exp 3) and "
               "the character activation space, by concept and model. Higher values "
               "indicate the concept dimension is reflected in how characters are "
               "represented.")
    return html_figure(b64, caption, fig_num=fig_num, alt="Contrast alignment bars")


def make_standalone_alignment_chart(all_standalone, fig_num):
    """Section 7b: Bar chart of peak standalone alignment per concept, grouped by model."""
    sorted_keys = sort_models([m for m in all_standalone if all_standalone[m] is not None])
    models_with_data = [(m, all_standalone[m]) for m in sorted_keys]
    if not models_with_data:
        return ""
    all_peaks = {}
    for model, data in models_with_data:
        all_peaks[model] = _extract_concept_peak_alignment(data, field="human_ai_bias")
    all_concept_names = set()
    for peaks in all_peaks.values():
        all_concept_names.update(peaks.keys())
    concept_names_sorted = sorted(all_concept_names)
    if not concept_names_sorted:
        return ""
    n_models = len(models_with_data)
    x = np.arange(len(concept_names_sorted))
    width = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(max(10, len(concept_names_sorted) * 0.6), 5))
    for j, (model, _) in enumerate(models_with_data):
        peaks = all_peaks[model]
        vals = [peaks.get(c, {}).get("peak_alignment", 0.0) for c in concept_names_sorted]
        offset = (j - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=MODEL_LABELS.get(model, model),
               color=MODEL_COLORS.get(model, "#333"), alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(concept_names_sorted, rotation=90, fontsize=8)
    ax.set_ylabel("Peak Alignment (human-AI bias)")
    ax.set_title("Standalone Alignment: Peak Human-AI Bias per Concept")
    ax.legend(loc="best", framealpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Peak standalone alignment (human-AI bias in cosine similarity to "
               "concept vector) per concept, by model. Positive values mean human "
               "characters are more aligned with the concept direction.")
    return html_figure(b64, caption, fig_num=fig_num, alt="Standalone alignment bars")


def make_mental_vs_control(all_contrast, all_standalone, fig_num):
    """Section 7c: Compare alignment for mental vs control concepts."""
    sorted_c = sort_models([m for m in all_contrast if all_contrast[m] is not None])
    models_with_contrast = [(m, all_contrast[m]) for m in sorted_c]
    sorted_s = sort_models([m for m in all_standalone if all_standalone[m] is not None])
    models_with_standalone = [(m, all_standalone[m]) for m in sorted_s]
    if not models_with_contrast and not models_with_standalone:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Contrast alignment
    ax = axes[0]
    ax.set_title("Contrast Alignment")
    model_keys_c = []
    mental_means_c = []
    control_means_c = []
    mental_sems_c = []
    control_sems_c = []
    for model, data in models_with_contrast:
        peaks = _extract_concept_peak_alignment(data, field="combined_alignment")
        mental_vals = [v["peak_alignment"] for k, v in peaks.items()
                       if not v.get("is_control", False) and v.get("is_entity_framed", True)]
        control_vals = [v["peak_alignment"] for k, v in peaks.items()
                        if v.get("is_control", False)]
        if mental_vals:
            model_keys_c.append(model)
            mental_means_c.append(np.mean(mental_vals))
            mental_sems_c.append(np.std(mental_vals) / np.sqrt(len(mental_vals)))
            control_means_c.append(np.mean(control_vals) if control_vals else 0)
            control_sems_c.append(np.std(control_vals) / np.sqrt(len(control_vals)) if len(control_vals) > 1 else 0)
    if model_keys_c:
        x = np.arange(len(model_keys_c))
        width = 0.35
        ax.bar(x - width / 2, mental_means_c, width, yerr=mental_sems_c,
               label="Mental", color="#1976d2", alpha=0.8, capsize=3)
        ax.bar(x + width / 2, control_means_c, width, yerr=control_sems_c,
               label="Control", color="#9e9e9e", alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_keys_c],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Peak Cosine Similarity")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    # Standalone alignment
    ax = axes[1]
    ax.set_title("Standalone Alignment")
    model_keys_s = []
    mental_means_s = []
    control_means_s = []
    mental_sems_s = []
    control_sems_s = []
    for model, data in models_with_standalone:
        peaks = _extract_concept_peak_alignment(data, field="human_ai_bias")
        # For standalone, no is_control flag; use name matching
        mental_vals = [v["peak_alignment"] for k, v in peaks.items()
                       if k.lower() not in CONTROL_CONCEPTS]
        control_vals = [v["peak_alignment"] for k, v in peaks.items()
                        if k.lower() in CONTROL_CONCEPTS]
        if mental_vals:
            model_keys_s.append(model)
            mental_means_s.append(np.mean(mental_vals))
            mental_sems_s.append(np.std(mental_vals) / np.sqrt(len(mental_vals)))
            control_means_s.append(np.mean(control_vals) if control_vals else 0)
            control_sems_s.append(np.std(control_vals) / np.sqrt(len(control_vals)) if len(control_vals) > 1 else 0)
    if model_keys_s:
        x = np.arange(len(model_keys_s))
        width = 0.35
        ax.bar(x - width / 2, mental_means_s, width, yerr=mental_sems_s,
               label="Mental", color="#1976d2", alpha=0.8, capsize=3)
        ax.bar(x + width / 2, control_means_s, width, yerr=control_sems_s,
               label="Control", color="#9e9e9e", alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in model_keys_s],
                           rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Mean Peak Alignment")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Mental vs Control Concept Alignment", fontsize=14, y=1.02)
    plt.tight_layout()
    b64 = fig_to_b64(fig)
    plt.close(fig)
    caption = ("Mean peak alignment for mental concepts (entity-framed, non-control) vs "
               "control concepts (shapes). Left: contrast alignment (cosine similarity "
               "with concept contrast vectors). Right: standalone alignment (human-AI bias). "
               "Error bars show SEM.")
    return html_figure(b64, caption, fig_num=fig_num, alt="Mental vs control alignment")


# ============================================================================
# KEY TAKEAWAYS
# ============================================================================

def generate_takeaways(all_pca, all_rsa, all_concept_rsa, all_contrast, all_standalone):
    """Generate auto-summary bullets."""
    bullets = []
    # PCA
    models_with_pca = sort_models([m for m, d in all_pca.items() if d is not None])
    if models_with_pca:
        for model in models_with_pca:
            data = all_pca[model]
            if "categorical" in data:
                cat = data["categorical"].get("categorical", data["categorical"])
                factors = cat.get("factors", cat) if isinstance(cat, dict) else cat
                if isinstance(factors, dict):
                    factors = factors.get("factors", [])
                sig_factors = [f for f in factors if f["p_value"] < 0.05]
                label = MODEL_LABELS.get(model, model)
                if sig_factors:
                    factor_strs = [f"F{f['factor']} (p={f['p_value']:.4f})" for f in sig_factors]
                    bullets.append(
                        f"<strong>{label}</strong>: {len(sig_factors)} of "
                        f"{len(factors)} PCA factors significantly separate AI "
                        f"from human characters: {', '.join(factor_strs)}."
                    )
                else:
                    bullets.append(
                        f"<strong>{label}</strong>: No PCA factors significantly "
                        f"separate AI from human characters."
                    )
    # RSA
    for model in sort_models([m for m in all_rsa if all_rsa[m] is not None]):
        data = all_rsa[model]
        peak = get_peak_rho(data["rsa_layers"])
        n_sig = get_n_sig_layers(data["rsa_layers"])
        label = MODEL_LABELS.get(model, model)
        if peak is not None:
            bullets.append(
                f"<strong>{label}</strong>: Activation RSA peaks at layer "
                f"{peak['layer']} (rho={peak['rho']:.3f}), with {n_sig} "
                f"significant layers."
            )
    # Concept RSA
    for model in sort_models([m for m in all_concept_rsa if all_concept_rsa[m] is not None]):
        concept_data = all_concept_rsa[model]
        label = MODEL_LABELS.get(model, model)
        sig_concepts = []
        for concept in CONCEPT_KEYS:
            if concept in concept_data:
                peak = get_peak_rho(concept_data[concept])
                if peak is not None:
                    p_val = peak.get("p_fdr") if peak.get("p_fdr") is not None else peak.get("p_value")
                    if p_val is not None and not (isinstance(p_val, float) and np.isnan(p_val)) and p_val < 0.05:
                        sig_concepts.append(concept)
        bullets.append(
            f"<strong>{label}</strong>: {len(sig_concepts)} of {len(CONCEPT_KEYS)} "
            f"concept dimensions show significant categorical RSA."
        )
    # Contrast alignment: mental vs control
    for model in sort_models([m for m in all_contrast if all_contrast[m] is not None]):
        data = all_contrast[model]
        label = MODEL_LABELS.get(model, model)
        summary = data.get("summary", {})
        entity_mean = summary.get("entity_mean_peak_alignment")
        control_mean = summary.get("control_mean_peak_alignment")
        if entity_mean is not None and control_mean is not None:
            ratio = entity_mean / control_mean if control_mean > 0 else float("inf")
            bullets.append(
                f"<strong>{label}</strong>: Mental concepts show "
                f"{ratio:.1f}x higher contrast alignment than control concepts "
                f"(entity={entity_mean:.4f}, control={control_mean:.4f})."
            )
    if not bullets:
        return '<p>No data available for summary.</p>\n'
    html = '<ul>\n'
    for b in bullets:
        html += f'<li>{b}</li>\n'
    html += '</ul>\n'
    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Collect data from all models
    all_pca = {}
    all_rsa = {}
    all_concept_rsa = {}
    all_concept_summary = {}
    all_contrast = {}
    all_standalone = {}
    available_models = []

    for model in VALID_MODELS:
        try:
            set_model(model)
        except Exception:
            continue
        pca = load_pca_data(model)
        rsa = load_activation_rsa(model)
        concept_rsa, concept_summary = load_concept_rsa(model)
        contrast = load_contrast_alignment(model)
        standalone = load_standalone_alignment(model)
        has_data = any(x is not None for x in [pca, rsa, concept_rsa, contrast, standalone])
        if has_data:
            available_models.append(model)
            all_pca[model] = pca
            all_rsa[model] = rsa
            all_concept_rsa[model] = concept_rsa
            all_concept_summary[model] = concept_summary
            all_contrast[model] = contrast
            all_standalone[model] = standalone

    available_models = sort_models(available_models)

    if not available_models:
        print("No model data found. Exiting.")
        return

    print(f"Found data for {len(available_models)} models: {available_models}")
    for model in available_models:
        parts = []
        if all_pca[model] is not None:
            parts.append("PCA")
        if all_rsa[model] is not None:
            parts.append("RSA")
        if all_concept_rsa[model] is not None:
            parts.append(f"Concept RSA ({len(all_concept_rsa[model])} concepts)")
        if all_contrast[model] is not None:
            parts.append("Contrast Align")
        if all_standalone[model] is not None:
            parts.append("Standalone Align")
        print(f"  {model}: {', '.join(parts)}")

    # Build the report
    fig_counter = [0]

    def next_fig():
        fig_counter[0] += 1
        return fig_counter[0]

    sections = [
        {"id": "overview", "label": "1. Overview"},
        {"id": "stimuli", "label": "2. Stimuli"},
        {"id": "methods", "label": "3. Methods"},
        {"id": "behavioral-pca", "label": "4. Behavioral PCA"},
        {"id": "activation-rsa", "label": "5. Activation RSA"},
        {"id": "concept-rsa", "label": "6. Per-Concept RSA"},
        {"id": "alignment", "label": "7. Alignment with Exp 3 Concept Vectors"},
        {"id": "takeaways", "label": "8. Key Takeaways"},
    ]

    html = build_cross_model_header("Expanded Mental Concepts: Cross-Model Summary")
    html += build_toc(sections)

    # ---- Section 1: Overview ----
    html += '<h2 id="overview">1. Overview</h2>\n'
    html += '<div class="method">\n'
    html += ('<p>This report bridges Experiment 3 concept probes into the mind '
             'perception geometry of Experiment 4. Using 28 characters (a subset of '
             'the 30 AI + human characters, excluding 2 that lack concept vector data) '
             'and 22 concept dimensions from Exp 3 (21 after PCA filtering), we test '
             'whether the internal concept structure distinguishes AI from human '
             'characters at both behavioral and neural levels.</p>\n')
    html += ('<p><strong>Analyses:</strong></p>\n'
             '<ol>\n'
             '<li><strong>Behavioral PCA</strong> — Factor structure of concept-guided '
             'pairwise comparison scores.</li>\n'
             '<li><strong>Activation RSA</strong> — Representational similarity between '
             'character activation patterns and a categorical AI/human RDM.</li>\n'
             '<li><strong>Per-Concept RSA</strong> — Which individual concept dimensions '
             'carry categorical structure?</li>\n'
             '<li><strong>Concept Vector Alignment</strong> — Do Exp 3 concept contrast '
             'vectors align with how characters are represented?</li>\n'
             '</ol>\n')
    models_str = ", ".join(MODEL_LABELS.get(m, m) for m in available_models)
    html += f'<p><strong>Models with data:</strong> {models_str}</p>\n'
    html += '</div>\n'

    # ---- Section 2: Stimuli ----
    html += expanded_concepts_stimuli_html()

    # ---- Section 3: Methods ----
    html += '<h2 id="methods">3. Methods</h2>\n'
    html += '<div class="method">\n'
    html += ('<p><strong>Behavioral PCA:</strong> For each character pair and each concept '
             'dimension, the model is prompted with a pairwise comparison question (e.g., '
             '"Which character is more capable of experiencing emotions?"). Responses are '
             'parsed to extract which character was chosen. The resulting entity x concept '
             'win-rate matrix is submitted to PCA with varimax rotation.</p>\n')
    html += ('<p><strong>Activation RSA:</strong> Character descriptions are fed to the model, '
             'and hidden-state activations are extracted at each layer. A representational '
             'dissimilarity matrix (RDM) is computed from pairwise cosine distances between '
             'character activation vectors. This model RDM is then correlated (Spearman) with '
             'a categorical RDM that groups characters as AI vs human.</p>\n')
    html += ('<p><strong>Per-Concept RSA:</strong> Instead of using the raw character '
             'activations, each character is projected onto a single concept dimension '
             '(from Exp 3 concept vectors). The resulting 1D concept-projected representations '
             'are used to build a concept-specific RDM, which is correlated with the '
             'categorical RDM. This reveals which individual concept dimensions carry '
             'AI/human categorical structure in the activation space.</p>\n')
    html += ('<p><strong>Concept Vector Alignment:</strong> Two alignment analyses test whether '
             'Exp 3 concept vectors point in directions that distinguish AI from human characters: '
             '(1) <em>Contrast alignment</em> computes cosine similarity between character '
             'activation vectors and concept contrast vectors (human-direction minus AI-direction). '
             '(2) <em>Standalone alignment</em> measures mean cosine similarity between character '
             'activations and concept vectors, comparing the human-AI bias.</p>\n')

    # Concept vector construction details
    html += '<h3>Concept Vector Construction (Exp 3)</h3>\n'
    html += ('<p>Concept vectors come from Experiment 3, where each concept dimension is '
             'defined by <strong>40 reflective prompts</strong> (4 sub-facets &times; 10 '
             'prompts each). Prompts use introspective language (&ldquo;Think about&hellip;&rdquo;, '
             '&ldquo;Imagine&hellip;&rdquo;, &ldquo;Consider&hellip;&rdquo;) with no entity '
             'framing (no mention of &ldquo;humans&rdquo; or &ldquo;AIs&rdquo;). '
             'Exception: Dimension 19 (General Mind) concatenates dims 1&ndash;7 for 280 prompts.</p>\n')

    html += ('<p><strong>Vector extraction:</strong> Each prompt is fed through the model. '
             'Last-token residual-stream activations are extracted at every layer. '
             'For <em>contrast</em> vectors, prompts are paired with human/AI entity contexts '
             'and the contrast direction is: mean(human-context activations) &minus; '
             'mean(AI-context activations). For <em>standalone</em> vectors, the mean '
             'activation across all prompts is used directly.</p>\n')

    html += '<h4>Example Prompts by Concept</h4>\n'
    html += '<table>\n'
    html += '<tr><th>Concept</th><th>Sub-facet</th><th>Example Prompts</th></tr>\n'

    concept_examples = [
        ("Phenomenology", "Sensory Qualia",
         "Imagine what it is like to see the color red for the first time. / "
         "Think about the raw sensory quality of hearing a single, clear musical note. / "
         "Consider what it is like to feel the warmth of sunlight on skin."),
        ("Emotions", "Basic Emotions",
         "Imagine experiencing a sudden wave of fear triggered by an unexpected noise at night. / "
         "Think about what it is like when anger rises in response to an unfair situation. / "
         "Consider the experience of deep sadness after losing something valued."),
        ("Agency", "Self-Initiated Action",
         "Imagine deciding to stand up and leave a room for no particular reason. / "
         "Think about spontaneously beginning to do something while alone, with no prompt or plan. / "
         "Consider starting a new project without anyone asking."),
        ("Social Cognition", "Mentalizing",
         "Imagine realizing that another holds a belief that is not true. / "
         "Think about inferring what someone else knows based only on what they have been told. / "
         "Consider attributing a specific motive to another based on their pattern of behavior."),
        ("Beliefs", "Propositional Belief",
         "Imagine firmly believing that someone is trustworthy based on years of accumulated evidence. / "
         "Think about holding the belief that effort leads to results, even when the evidence is mixed."),
        ("Desires", "Appetitive Wanting",
         "Imagine feeling a strong pull toward warmth and food after a long stretch of cold and hunger. / "
         "Think about being drawn to learn more about something fascinating."),
        ("Goals", "Goal Representation",
         "Imagine articulating a specific goal for the first time and feeling it become real by being stated. / "
         "Think about a goal so clearly defined that the exact conditions for success can be described."),
        ("Human", "Simple Reference",
         "Think about a human. / Consider a human being. / Imagine a human person."),
        ("AI", "Simple Reference",
         "Think about an AI. / Consider an artificial intelligence. / Imagine an AI system."),
        ("Shapes (control)", "Geometric",
         "Think about a square. / Consider a triangle. / Imagine a circle."),
    ]

    for concept, facet, examples in concept_examples:
        html += (f'<tr><td>{concept}</td><td style="font-size:0.85em">{facet}</td>'
                 f'<td style="font-size:0.85em">{examples}</td></tr>\n')
    html += '</table>\n'

    html += ('<p style="font-size:0.85em; color:#555;">'
             '22 concept dimensions total: 14 human-favored (phenomenology, emotions, agency, '
             'intentions, cognitive, prediction, social, attention, embodiment, animacy, '
             'biological, human, beliefs, desires), 4 AI-favored (formality, expertise, '
             'helpfulness, AI), 3 ambiguous (roles, goals, general mind), '
             '1 control (shapes). Each has 40 prompts across 4 sub-facets, except '
             'general mind (280 prompts = dims 1&ndash;7 concatenated).</p>\n')

    html += '</div>\n'

    # ── Interpretation Guide ──
    html += '<h3>Interpretation Guide</h3>\n'
    html += expanded_concepts_primer_html()
    html += neural_methods_primer_html(
        include_layers=True, include_rdm=True, include_rsa=True,
        include_procrustes=False)
    html += methodology_primer_html(
        include_pca=True, include_spearman=True, include_fdr=True,
        include_prompting=True, include_pairwise=True)

    html += '<div class="method">\n'
    html += '<h4>Why 28 Characters Instead of 30?</h4>\n'
    html += (
        '<p>Two characters from the full 30-character set used in the '
        'Human-AI Adaptation branch were excluded from the expanded concepts '
        'analysis because Experiment&nbsp;3 concept vectors were not available '
        'for them (they lacked the required pairwise behavioral data on the '
        '22 concept dimensions).</p>\n')
    html += '<h4>What Alignment Values Mean</h4>\n'
    html += (
        '<p>Alignment R&sup2; values are typically small in absolute terms '
        '(~0.001&ndash;0.01) because the activation space is very high-'
        'dimensional (4,096&ndash;5,120 dimensions) and a single concept '
        'vector captures only one direction. The meaningful comparison is '
        'the <strong>ratio</strong> between mental concept alignment and '
        'control (shapes) alignment. A ratio &gt;&nbsp;1 indicates the '
        'concept vector captures mind-relevant structure beyond a generic '
        'categorical signal; ratios of 2&ndash;10x are typical for mental '
        'concepts.</p>\n')
    html += '</div>\n'

    # ---- Section 4: Behavioral PCA ----
    html += '<h2 id="behavioral-pca">4. Behavioral PCA</h2>\n'
    any_pca = any(d is not None for d in all_pca.values())
    if not any_pca:
        html += '<p class="warning">No behavioral PCA data available for any model.</p>\n'
    else:
        html += '<h3>4a. Character Positions (F1 x F2)</h3>\n'
        html += make_pca_scatter(all_pca, next_fig())

        html += '<h3>4b. Factor Loadings</h3>\n'
        html += make_loading_comparison(all_pca, next_fig())

        html += '<h3>4c. Group Separation</h3>\n'
        html += make_group_separation(all_pca, next_fig())

        # Variance explained table
        html += '<h3>Variance Explained</h3>\n'
        html += '<table>\n<tr><th>Model</th><th>F1</th><th>F2</th><th>F3</th><th>F4</th><th>Total (4 factors)</th></tr>\n'
        for model in sort_models([m for m in all_pca if all_pca[m] is not None]):
            data = all_pca[model]
            ve = data["explained_var_ratio"]
            n_f = min(4, len(ve))
            row = f'<tr>{model_row_td(model)}'
            for i in range(n_f):
                row += f'<td>{ve[i]*100:.1f}%</td>'
            for i in range(n_f, 4):
                row += '<td>-</td>'
            row += f'<td>{sum(ve[:n_f])*100:.1f}%</td></tr>\n'
            html += row
        html += '</table>\n'

    # ---- Section 5: Activation RSA ----
    html += '<h2 id="activation-rsa">5. Activation RSA</h2>\n'
    any_rsa = any(d is not None for d in all_rsa.values())
    if not any_rsa:
        html += '<p class="warning">No activation RSA data available for any model.</p>\n'
    else:
        html += '<h3>5a. Layerwise RSA</h3>\n'
        html += make_rsa_layerwise(all_rsa, next_fig())
        next_fig()  # make_rsa_layerwise produces 2 figures

        html += '<h3>5b. Peak RSA Summary</h3>\n'
        html += make_peak_rsa_table(all_rsa)

        html += '<h3>5c. RDM at Peak Layer</h3>\n'
        html += make_rdm_heatmaps(all_rsa, next_fig())

    # ---- Section 6: Per-Concept RSA ----
    html += '<h2 id="concept-rsa">6. Per-Concept RSA</h2>\n'
    any_concept = any(d is not None for d in all_concept_rsa.values())
    if not any_concept:
        html += '<p class="warning">No per-concept RSA data available for any model.</p>\n'
    else:
        html += '<h3>6a. Concept x Model Heatmap</h3>\n'
        html += make_concept_rsa_heatmap(all_concept_rsa, next_fig())

        html += '<h3>6b. Top and Bottom Concepts per Model</h3>\n'
        html += make_concept_rsa_tables(all_concept_rsa)

        html += '<h3>6c. Cross-Model Consistency</h3>\n'
        html += make_concept_consistency(all_concept_rsa)

    # ---- Section 7: Alignment ----
    html += '<h2 id="alignment">7. Alignment with Exp 3 Concept Vectors</h2>\n'
    any_align = any(d is not None for d in all_contrast.values()) or any(
        d is not None for d in all_standalone.values())
    if not any_align:
        html += '<p class="warning">No alignment data available for any model.</p>\n'
    else:
        html += '<h3>7a. Contrast Alignment</h3>\n'
        html += make_contrast_alignment_chart(all_contrast, next_fig())

        html += '<h3>7b. Standalone Alignment</h3>\n'
        html += make_standalone_alignment_chart(all_standalone, next_fig())

        html += '<h3>7c. Mental vs Control Concepts</h3>\n'
        html += make_mental_vs_control(all_contrast, all_standalone, next_fig())

        # Summary table for contrast alignment
        any_contrast_data = any(d is not None for d in all_contrast.values())
        if any_contrast_data:
            html += '<h3>Contrast Alignment Summary</h3>\n'
            html += '<table>\n<tr><th>Model</th><th>Entity-Framed Mean</th><th>Control Mean</th><th>Ratio</th></tr>\n'
            for model in sort_models([m for m in all_contrast if all_contrast[m] is not None]):
                data = all_contrast[model]
                summary = data.get("summary", {})
                entity_mean = summary.get("entity_mean_peak_alignment")
                control_mean = summary.get("control_mean_peak_alignment")
                if entity_mean is not None and control_mean is not None:
                    ratio = entity_mean / control_mean if control_mean > 0 else float("inf")
                    ratio_str = f"{ratio:.1f}x" if ratio != float("inf") else "inf"
                    html += (f'<tr>{model_row_td(model)}'
                             f'<td>{entity_mean:.6f}</td>'
                             f'<td>{control_mean:.6f}</td>'
                             f'<td><strong>{ratio_str}</strong></td></tr>\n')
            html += '</table>\n'

    # ---- Section 8: Key Takeaways ----
    html += '<h2 id="takeaways">8. Key Takeaways</h2>\n'
    html += generate_takeaways(all_pca, all_rsa, all_concept_rsa, all_contrast, all_standalone)

    # Footer
    html += '<hr>\n'
    html += '<p style="color:#888;font-size:0.85em;">Rachel C. Metzgar &middot; Mar 2026</p>\n'
    html += build_html_footer()

    # Write
    out_dir = ensure_dir(COMPARISONS_DIR)
    out_path = out_dir / "expanded_concepts_summary.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
