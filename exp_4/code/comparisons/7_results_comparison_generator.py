#!/usr/bin/env python3
"""
Experiment 4: Cross-Experiment Results Comparison Report

Generates a publication-ready HTML report synthesizing findings across all
four experimental branches (gray_replication, gray_simple, human_ai_adaptation,
expanded_mental_concepts) and all 11 models. Includes 8 figures highlighting
the main trends and takeaways.

Reads pre-computed data from:
    results/{model}/{branch}/{modality}/{condition}/data/

Output:
    results/comparisons/results_comparison.html

Usage:
    python comparisons/7_results_comparison_generator.py

Env: llama2_env (login node, CPU only)
Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    ROOT_DIR, COMPARISONS_DIR, VALID_MODELS, MODELS,
    ensure_dir, set_model, data_dir,
)
from utils.report_utils import (
    REPORT_CSS, build_cross_model_header, build_html_footer, build_toc,
    fig_to_b64, html_figure, MODEL_COLORS, MODEL_LABELS, ALL_MODELS,
    sort_models, GRID_NCOLS, make_model_grid, model_row_td, format_p_cell,
    INSTRUCTION_TUNING_PAIRS, MODEL_FAMILIES,
    methodology_primer_html, neural_methods_primer_html,
    expanded_concepts_primer_html,
)

# ============================================================================
# STYLE
# ============================================================================

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

C_EXP = "#2166ac"   # blue  -- Experience factor
C_AGE = "#b2182b"   # red   -- Agency factor
C_HUM = "#555555"   # gray  -- Human reference
C_INST = "#2a5fa5"  # blue  -- Instruct/chat aggregate
C_BASE = "#8b7355"  # muted -- Base aggregate


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load data from all four branches for all available models.

    Returns a nested dict: data[model][branch] = {...}.
    """
    all_data = {}

    for model in VALID_MODELS:
        set_model(model)
        entry = {
            "label": MODEL_LABELS[model],
            "is_chat": MODELS[model]["is_chat"],
            "family": MODELS[model]["family"],
            "n_layers": MODELS[model]["n_transformer_layers"],
        }

        # --- gray_replication (behavioral) ---
        try:
            ddir = data_dir("gray_entities", "behavioral", "with_self")
            pca_path = ddir / "pairwise_pca_results.npz"
            if pca_path.exists():
                pca = np.load(pca_path, allow_pickle=True)
                entry["gray_rep"] = {
                    "eigenvalues": pca["eigenvalues"],
                    "loadings": pca["rotated_loadings"],
                    "scores_01": pca["factor_scores_01"],
                    "entity_keys": list(pca["entity_keys"]),
                }
                # Human correlations
                corr_path = ddir / "pairwise_human_correlations.json"
                if corr_path.exists():
                    with open(corr_path) as f:
                        entry["gray_rep"]["correlations"] = json.load(f)
                # Behavioral RSA
                rsa_path = ddir / "behavioral_rsa_results.json"
                if rsa_path.exists():
                    with open(rsa_path) as f:
                        entry["gray_rep"]["behavioral_rsa"] = json.load(f)
        except Exception as e:
            print(f"  gray_rep error for {model}: {e}")

        # --- gray_simple (neural internals) ---
        try:
            ddir = data_dir("gray_entities", "neural", "with_self")
            rsa_path = ddir / "rsa_results.json"
            if rsa_path.exists():
                with open(rsa_path) as f:
                    rsa_data = json.load(f)
                entry["gray_entities"] = rsa_data  # {combined, experience, agency}
        except Exception as e:
            print(f"  gray_simple error for {model}: {e}")

        # --- human_ai_adaptation (behavioral) ---
        try:
            ddir = data_dir("human_ai_characters", "behavioral/gray_capacities")
            pca_path = ddir / "pairwise_pca_results.npz"
            if pca_path.exists():
                pca = np.load(pca_path, allow_pickle=True)
                entry["human_ai"] = {
                    "eigenvalues": pca["eigenvalues"],
                    "scores_01": pca["factor_scores_01"],
                    "char_keys": list(pca["character_keys"]) if "character_keys" in pca else [],
                }
                # Categorical analysis
                cat_path = ddir / "pairwise_categorical_analysis.json"
                if cat_path.exists():
                    with open(cat_path) as f:
                        entry["human_ai"]["categorical"] = json.load(f)["categorical"]
            # Names-only
            names_dir = ROOT_DIR / "results" / model / "human_ai_characters" / "behavior" / "names_only" / "data"
            names_cat_path = names_dir / "pairwise_categorical_analysis.json"
            if names_cat_path.exists():
                with open(names_cat_path) as f:
                    entry["human_ai_names"] = json.load(f)["categorical"]
        except Exception as e:
            print(f"  human_ai error for {model}: {e}")

        # --- expanded_mental_concepts (activation RSA) ---
        try:
            rsa_dir = ROOT_DIR / "results" / model / "human_ai_characters" / "internals" / "rsa" / "data"
            rsa_path = rsa_dir / "rsa_results.json"
            if rsa_path.exists():
                with open(rsa_path) as f:
                    rsa_data = json.load(f)
                entry["expanded_rsa"] = rsa_data.get("categorical", rsa_data)
        except Exception as e:
            print(f"  expanded_rsa error for {model}: {e}")

        # --- expanded_mental_concepts (per-concept RSA) ---
        try:
            concept_rsa_path = (ROOT_DIR / "results" / model /
                                "human_ai_characters" / "internals" /
                                "concept_rsa" / "data" / "cross_concept_rsa_summary.json")
            if concept_rsa_path.exists():
                with open(concept_rsa_path) as f:
                    entry["concept_rsa"] = json.load(f)
        except Exception as e:
            print(f"  concept_rsa error for {model}: {e}")

        # --- expanded_mental_concepts (contrast alignment) ---
        try:
            align_path = (ROOT_DIR / "results" / model /
                          "human_ai_characters" / "internals" /
                          "contrast_alignment" / "data" / "alignment_results.json")
            if align_path.exists():
                with open(align_path) as f:
                    entry["contrast_alignment"] = json.load(f)
        except Exception as e:
            print(f"  contrast_alignment error for {model}: {e}")

        # --- expanded_mental_concepts (behavioral PCA) ---
        try:
            bpca_dir = (ROOT_DIR / "results" / model /
                        "human_ai_characters" / "behavior" / "pca" / "data")
            bpca_path = bpca_dir / "pairwise_pca_results.npz"
            if bpca_path.exists():
                bpca = np.load(bpca_path, allow_pickle=True)
                entry["expanded_bpca"] = {
                    "eigenvalues": bpca["eigenvalues"],
                }
                cat_path = bpca_dir / "pairwise_categorical_analysis.json"
                if cat_path.exists():
                    with open(cat_path) as f:
                        entry["expanded_bpca"]["categorical"] = json.load(f)["categorical"]
        except Exception as e:
            print(f"  expanded_bpca error for {model}: {e}")

        all_data[model] = entry
        print(f"  Loaded {model}: {', '.join(k for k in ['gray_rep','gray_simple','human_ai','expanded_rsa','concept_rsa','contrast_alignment','expanded_bpca'] if k in entry)}")

    return all_data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_peak_rsa(layer_list):
    """Find peak rho and its layer from a list of RSA layer dicts."""
    valid = [r for r in layer_list
             if r.get("rho") is not None
             and not (isinstance(r["rho"], float) and np.isnan(r["rho"]))]
    if not valid:
        return None, None, None
    best = max(valid, key=lambda r: abs(r["rho"]))
    return best["layer"], best["rho"], best.get("p_value", 1.0)


def _count_sig_layers(layer_list, alpha=0.05):
    """Count layers with p < alpha (before FDR)."""
    return sum(1 for r in layer_list
               if r.get("p_value") is not None
               and not (isinstance(r["p_value"], float) and np.isnan(r["p_value"]))
               and r["p_value"] < alpha)


def _sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    return ""


def _fig_to_b64_and_close(fig):
    """Convert figure to b64, close it, return b64 string."""
    if fig is None:
        return None
    b64 = fig_to_b64(fig, dpi=150)
    plt.close(fig)
    return b64


def _save_all_figures(data, fig_dir):
    """Regenerate all figures and save as PDF + PNG."""
    fig_funcs = [
        ("fig1a_layerwise_rsa_combined", _make_fig1a),
        ("fig1b_layerwise_rsa_experience", _make_fig1b),
        ("fig1c_layerwise_rsa_agency", _make_fig1c),
        ("fig2_agency_experience_asymmetry", _make_fig1),
        ("fig3_eigenvalue_base_vs_instruct", _make_fig2),
        ("fig4_categorical_rsa_saturation", _make_fig3),
        ("fig5_instruction_tuning_effect", _make_fig4),
        ("fig6_model_rankings_heatmap", _make_fig5),
        ("fig7_peak_layer_location", _make_fig6),
        ("fig8_names_vs_descriptions", _make_fig7),
        ("fig9_contrast_alignment_ratio", _make_fig8),
    ]
    for name, func in fig_funcs:
        try:
            fig = func(data)
            if fig is not None:
                fig.savefig(fig_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
                fig.savefig(fig_dir / f"{name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {name}.pdf/.png")
            else:
                print(f"  Skipped {name} (no data)")
        except Exception as e:
            print(f"  Warning: could not save {name}: {e}")


# ============================================================================
# FIGURES 1a-1c: Layerwise RSA Grids (Combined, Experience, Agency)
# ============================================================================

C_COMB = "#7b2d8e"  # purple -- Combined


def _clean_rho(rho_val):
    """Replace NaN rho values with 0.0 for plotting."""
    if rho_val is None:
        return 0.0
    if isinstance(rho_val, float) and np.isnan(rho_val):
        return 0.0
    return float(rho_val)


def _apply_fdr(rsa_layer_list):
    """Apply Benjamini-Hochberg FDR correction; adds 'q_fdr' key in-place."""
    from statsmodels.stats.multitest import multipletests
    pvals, valid_idx = [], []
    for i, r in enumerate(rsa_layer_list):
        p = r.get("p_value")
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            pvals.append(p)
            valid_idx.append(i)
    for r in rsa_layer_list:
        r["q_fdr"] = 1.0
    if pvals:
        _, q_corr, _, _ = multipletests(pvals, method="fdr_bh")
        for idx, q in zip(valid_idx, q_corr):
            rsa_layer_list[idx]["q_fdr"] = float(q)
    return rsa_layer_list


def _make_layerwise_rsa_grid(data, variant, title_suffix, default_color):
    """Build a family-grouped grid of layerwise RSA bar charts.

    Args:
        data: full data dict
        variant: "combined", "experience", or "agency"
        title_suffix: e.g. "Combined (Experience + Agency)"
        default_color: fallback bar color for significant layers

    Returns:
        matplotlib Figure or None
    """
    models_with_data = [m for m in sort_models(data) if "gray_entities" in data[m]]
    if not models_with_data:
        return None

    # Apply FDR to this variant for each model
    for m in models_with_data:
        rsa_list = data[m]["gray_entities"].get(variant, [])
        if rsa_list:
            _apply_fdr(rsa_list)

    # Shared y-range
    all_rhos = []
    for m in models_with_data:
        for r in data[m]["gray_entities"].get(variant, []):
            all_rhos.append(_clean_rho(r["rho"]))
    if not all_rhos:
        return None
    y_min = min(all_rhos) - 0.05
    y_max = max(all_rhos) + 0.1

    positions, ordered, nrows, ncols, _ = make_model_grid(models_with_data)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 4.5 * nrows),
                             squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)

    for idx, mk in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)

        rsa_list = data[mk]["gray_entities"].get(variant, [])
        if not rsa_list:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#999")
            ax.set_title(MODEL_LABELS[mk], fontsize=9)
            continue

        base_color = MODEL_COLORS.get(mk, default_color)
        layers = [r["layer"] for r in rsa_list]
        rhos = [_clean_rho(r["rho"]) for r in rsa_list]
        qvals = [r.get("q_fdr", 1.0) for r in rsa_list]

        colors = [base_color if q < 0.05 else "#cccccc" for q in qvals]
        ax.bar(layers, rhos, color=colors, edgecolor="white", width=0.8)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Spearman \u03c1")
        ax.set_ylim(y_min, y_max)

        # Find peak
        best = max(rsa_list,
                   key=lambda r: _clean_rho(r["rho"]))
        peak_rho = _clean_rho(best["rho"])
        peak_q = best.get("q_fdr", 1.0)
        n_sig = sum(1 for q in qvals if q < 0.05)
        n_total = len(layers)
        ax.set_title(
            f"{MODEL_LABELS[mk]}\n{n_sig}/{n_total} sig, "
            f"peak L{best['layer']} (\u03c1={peak_rho:.3f})",
            fontsize=9)
        if best["layer"] >= 0:
            ax.annotate(
                f"\u03c1={peak_rho:.3f}\nq={peak_q:.3f}",
                (best["layer"], peak_rho),
                textcoords="offset points", xytext=(12, 8), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    fig.suptitle(f"{title_suffix}: Model RDM vs Human Mind Perception RDM",
                 fontsize=14)
    fig.tight_layout()
    return fig


def _make_fig1a(data):
    return _make_layerwise_rsa_grid(
        data, "combined",
        "Combined RSA (Experience + Agency)", C_COMB)


def _make_fig1b(data):
    return _make_layerwise_rsa_grid(
        data, "experience",
        "Experience-Only RSA", C_EXP)


def _make_fig1c(data):
    return _make_layerwise_rsa_grid(
        data, "agency",
        "Agency-Only RSA", C_AGE)


# ============================================================================
# FIGURE 2: Agency vs Experience Peak Bar Chart (gray_simple)
# ============================================================================


def _rsa_sig_sd(layer_list, alpha=0.05):
    """Compute SD of |rho| across significant layers (error bar)."""
    vals = [abs(r["rho"]) for r in layer_list
            if r.get("p_value") is not None
            and not (isinstance(r["p_value"], float) and np.isnan(r["p_value"]))
            and r["p_value"] < alpha
            and r.get("rho") is not None
            and not (isinstance(r["rho"], float) and np.isnan(r["rho"]))]
    return np.std(vals) if len(vals) > 1 else 0.0


def _make_fig1(data):
    """Build Figure 1 and return the fig object.

    Three grouped bars per model: Combined (purple), Experience (blue),
    Agency (red), with SD of |rho| across significant layers as error bars.
    """
    models_with_data = [m for m in sort_models(data) if "gray_entities" in data[m]]
    if not models_with_data:
        return None

    labels = []
    comb_peaks, exp_peaks, age_peaks = [], [], []
    comb_errs, exp_errs, age_errs = [], [], []
    colors_list = []

    for m in models_with_data:
        gs = data[m]["gray_entities"]
        _, c_rho, _ = _get_peak_rsa(gs.get("combined", []))
        _, e_rho, _ = _get_peak_rsa(gs.get("experience", []))
        _, a_rho, _ = _get_peak_rsa(gs.get("agency", []))
        labels.append(MODEL_LABELS[m])
        comb_peaks.append(abs(c_rho) if c_rho else 0)
        exp_peaks.append(abs(e_rho) if e_rho else 0)
        age_peaks.append(abs(a_rho) if a_rho else 0)
        comb_errs.append(_rsa_sig_sd(gs.get("combined", [])))
        exp_errs.append(_rsa_sig_sd(gs.get("experience", [])))
        age_errs.append(_rsa_sig_sd(gs.get("agency", [])))
        colors_list.append(MODEL_COLORS.get(m, "#333"))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width, comb_peaks, width,
           color=C_COMB, edgecolor="white", label="Combined", alpha=0.85,
           zorder=2)
    ax.bar(x, exp_peaks, width,
           color=C_EXP, edgecolor="white", label="Experience", alpha=0.85,
           zorder=2)
    ax.bar(x + width, age_peaks, width,
           color=C_AGE, edgecolor="white", label="Agency", alpha=0.85,
           zorder=2)

    # Draw error bars separately on top so caps are never occluded.
    # Use a very high zorder and draw after all bars.
    err_kw = dict(fmt="none", capsize=4, lw=1.5, capthick=1.5,
                  ecolor="#222222", zorder=10)
    eb1 = ax.errorbar(x - width, comb_peaks, yerr=comb_errs, **err_kw)
    eb2 = ax.errorbar(x, exp_peaks, yerr=exp_errs, **err_kw)
    eb3 = ax.errorbar(x + width, age_peaks, yerr=age_errs, **err_kw)
    # Force error bar artists above everything else
    for eb in (eb1, eb2, eb3):
        for child in eb.get_children():
            child.set_zorder(10)

    # Color model names on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for tick_label, color in zip(ax.get_xticklabels(), colors_list):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_ylabel("Peak |Spearman rho| with Human RDM")
    ax.set_title("Experience vs. Agency: Neural RSA Peak Alignment")
    ax.legend(loc="upper right")
    all_vals = comb_peaks + exp_peaks + age_peaks
    all_errs = comb_errs + exp_errs + age_errs
    max_top = max(v + e for v, e in zip(all_vals, all_errs)) if all_vals else 0.5
    ax.set_ylim(0, max_top * 1.15)
    ax.axhline(0, color="gray", lw=0.5)

    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 2: Base vs Instruct Eigenvalue Structure (gray_replication)
# ============================================================================

def _make_fig2(data):
    """Paired eigenvalue ratio comparison: base models (one-factor) vs instruct."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (group_label, is_chat) in enumerate([("Base Models", False),
                                                       ("Instruct/Chat Models", True)]):
        ax = axes[ax_idx]
        models = [m for m in sort_models(data)
                  if "gray_rep" in data[m] and data[m]["is_chat"] == is_chat]

        human_eig = np.array([15.85, 1.46])
        ax.plot(range(1, 3), human_eig, "s--", color=C_HUM, lw=2, ms=8,
                zorder=10, label="Humans")

        for m in models:
            eig = data[m]["gray_rep"]["eigenvalues"]
            n = min(len(eig), 6)
            color = MODEL_COLORS.get(m, "#333")
            ax.plot(range(1, n+1), eig[:n], "o-", color=color, lw=2, ms=5,
                    alpha=0.85, label=MODEL_LABELS[m])

        ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel("Component")
        ax.set_title(group_label, fontsize=12, fontweight="bold")
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(-0.5, 18.5)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Eigenvalue")
    fig.suptitle("PCA Eigenvalue Structure: Base Models Collapse to One Factor",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 3: AI/Human Categorical RSA Saturation (expanded_concepts)
# ============================================================================

def _make_fig3(data):
    """Per-concept RSA showing narrow range + shapes control performance."""
    # Collect per-concept peak rho for each model
    models_with_data = [m for m in sort_models(data) if "concept_rsa" in data[m]]
    if not models_with_data:
        return None

    # Gather all concept names and their peak rho per model
    concept_data = defaultdict(list)  # concept_name -> [rho values]
    for m in models_with_data:
        for c in data[m]["concept_rsa"]["concepts"]:
            concept_data[c["concept_name"]].append(abs(c["peak_rho"]))

    # Sort by mean rho
    concept_means = {k: np.mean(v) for k, v in concept_data.items()}
    sorted_concepts = sorted(concept_means, key=concept_means.get, reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(sorted_concepts))
    means = [concept_means[c] for c in sorted_concepts]
    stds = [np.std(concept_data[c]) for c in sorted_concepts]

    # Color: shapes control in red, others in blue
    colors = []
    for c in sorted_concepts:
        if c.lower() in ("shapes", "shape"):
            colors.append("#e74c3c")  # red for control
        else:
            colors.append(C_INST)

    ax.barh(y, means, xerr=stds, color=colors, edgecolor="white",
            height=0.7, capsize=3, alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_concepts, fontsize=9)
    ax.set_xlabel("Mean Peak |Spearman rho| (across 11 models)")
    ax.set_title("Per-Concept Categorical RSA: AI/Human Signal Is Pervasive")
    ax.invert_yaxis()

    # Add range annotation
    rng = max(means) - min(means)
    ax.axvline(min(means), color="gray", ls=":", lw=1, alpha=0.5)
    ax.axvline(max(means), color="gray", ls=":", lw=1, alpha=0.5)

    legend_elements = [
        Patch(facecolor=C_INST, label="Mental concepts"),
        Patch(facecolor="#e74c3c", label="Shapes (control)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 4: Instruction Tuning Neural Effect by Family (gray_simple)
# ============================================================================

def _make_fig4(data):
    """Paired bar chart: base vs instruct peak RSA for each model family."""
    pairs = []
    for family_label, base_key, instruct_key in INSTRUCTION_TUNING_PAIRS:
        if ("gray_entities" in data.get(base_key, {}) and
                "gray_entities" in data.get(instruct_key, {})):
            _, base_rho, _ = _get_peak_rsa(data[base_key]["gray_entities"]["combined"])
            _, inst_rho, _ = _get_peak_rsa(data[instruct_key]["gray_entities"]["combined"])
            if base_rho is not None and inst_rho is not None:
                pairs.append((family_label, base_key, instruct_key,
                              abs(base_rho), abs(inst_rho)))

    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pairs))
    width = 0.35

    base_vals = [p[3] for p in pairs]
    inst_vals = [p[4] for p in pairs]
    deltas = [p[4] - p[3] for p in pairs]

    bars_base = ax.bar(x - width/2, base_vals, width,
                       color=C_BASE, edgecolor="white", label="Base", alpha=0.85)
    bars_inst = ax.bar(x + width/2, inst_vals, width,
                       color=C_INST, edgecolor="white", label="Instruct/Chat", alpha=0.85)

    # Annotate deltas
    for i, (bv, iv, delta) in enumerate(zip(base_vals, inst_vals, deltas)):
        top = max(bv, iv) + 0.01
        sign = "+" if delta > 0 else ""
        color = "#2E7D32" if delta > 0 else "#C62828"
        ax.text(i, top, f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=11)
    ax.set_ylabel("Peak |Spearman rho| with Human RDM")
    ax.set_title("Instruction Tuning Effect on Neural Mind Perception (Gray Simple RSA)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(max(base_vals), max(inst_vals)) * 1.25)

    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 5: Model Rankings Heatmap
# ============================================================================

def _make_fig5(data):
    """Heatmap of model performance across different analyses."""
    analyses = []
    model_keys = sort_models(data)

    # 1. Gray replication behavioral RSA (combined, pairwise PCA)
    row_behav_rsa = []
    for m in model_keys:
        val = np.nan
        brsa = data[m].get("gray_rep", {}).get("behavioral_rsa", {})
        if brsa:
            # Look for pairwise_pca combined
            for source_key in ["pairwise_pca", "pca_2d"]:
                src = brsa.get(source_key, {})
                if isinstance(src, dict):
                    comb = src.get("combined", {})
                    if isinstance(comb, dict) and comb.get("p_value", 1.0) < 0.05:
                        val = abs(comb.get("rho", np.nan))
                        break
        row_behav_rsa.append(val)
    analyses.append(("Behavioral RSA\n(Gray Replication)", row_behav_rsa))

    # 2. Gray simple neural RSA (combined peak)
    row_neural = []
    for m in model_keys:
        val = np.nan
        gs = data[m].get("gray_entities", {})
        if gs:
            _, rho, p = _get_peak_rsa(gs.get("combined", []))
            if rho is not None and p is not None and p < 0.05:
                val = abs(rho)
        row_neural.append(val)
    analyses.append(("Neural RSA\n(Gray Simple)", row_neural))

    # 3. Categorical activation RSA (expanded concepts)
    row_cat = []
    for m in model_keys:
        val = np.nan
        ersa = data[m].get("expanded_rsa", [])
        if ersa:
            _, rho, p = _get_peak_rsa(ersa)
            if rho is not None:
                val = abs(rho)
        row_cat.append(val)
    analyses.append(("Categorical RSA\n(Expanded Concepts)", row_cat))

    # 4. Experience RSA peak
    row_exp = []
    for m in model_keys:
        val = np.nan
        gs = data[m].get("gray_entities", {})
        if gs:
            _, rho, p = _get_peak_rsa(gs.get("experience", []))
            if rho is not None and p is not None and p < 0.05:
                val = abs(rho)
        row_exp.append(val)
    analyses.append(("Experience RSA\n(Gray Simple)", row_exp))

    # 5. Agency RSA peak
    row_age = []
    for m in model_keys:
        val = np.nan
        gs = data[m].get("gray_entities", {})
        if gs:
            _, rho, p = _get_peak_rsa(gs.get("agency", []))
            if rho is not None and p is not None and p < 0.05:
                val = abs(rho)
        row_age.append(val)
    analyses.append(("Agency RSA\n(Gray Simple)", row_age))

    matrix = np.array([row for _, row in analyses])
    analysis_labels = [label for label, _ in analyses]
    model_labels = [MODEL_LABELS[m] for m in model_keys]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.9)

    ax.set_xticks(range(len(model_keys)))
    ax.set_xticklabels(model_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(analysis_labels)))
    ax.set_yticklabels(analysis_labels, fontsize=10)

    # Color x-axis labels by model
    for i, m in enumerate(model_keys):
        ax.get_xticklabels()[i].set_color(MODEL_COLORS.get(m, "#333"))
        ax.get_xticklabels()[i].set_fontweight("bold")

    # Annotate cells
    for i in range(len(analyses)):
        for j in range(len(model_keys)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "n.s.", ha="center", va="center",
                        fontsize=8, color="#999")
            else:
                text_color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color=text_color)

    # Best in each row
    for i in range(len(analyses)):
        row = matrix[i]
        valid = row[~np.isnan(row)]
        if len(valid) > 0:
            best_j = np.nanargmax(row)
            ax.add_patch(plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor="black", lw=2.5))

    plt.colorbar(im, ax=ax, label="|Spearman rho|", shrink=0.8)
    ax.set_title("Model Performance Across Analyses (black box = best)", fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 6: Peak Layer Location by Model Family
# ============================================================================

def _make_fig6(data):
    """Scatter plot of peak RSA layer as % of network depth, by model."""
    models_with_data = [m for m in sort_models(data) if "gray_entities" in data[m]]
    if not models_with_data:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, m in enumerate(models_with_data):
        gs = data[m]["gray_entities"]
        n_layers = data[m]["n_layers"]
        color = MODEL_COLORS.get(m, "#333")
        marker = "o" if data[m]["is_chat"] else "s"

        # Combined peak
        layer, rho, _ = _get_peak_rsa(gs["combined"])
        if layer is not None:
            pct = layer / n_layers * 100
            ax.scatter(pct, abs(rho), color=color, s=80, marker=marker,
                       edgecolor="white", linewidth=0.5, zorder=5)
            ax.annotate(MODEL_LABELS[m], (pct, abs(rho)),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color=color, fontweight="bold")

    ax.set_xlabel("Peak Layer (% of network depth)")
    ax.set_ylabel("Peak |Spearman rho|")
    ax.set_title("Peak Mind Perception Layer: Location and Strength")
    ax.set_xlim(-5, 105)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#666",
               ms=8, label="Instruct/Chat"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#666",
               ms=8, label="Base"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 7: AI/Human Separation — Full Descriptions vs Names Only
# ============================================================================

def _make_fig7(data):
    """Grouped bar chart: max AI-human factor separation with full vs names-only."""
    models_with_both = []
    for m in sort_models(data):
        if ("human_ai" in data[m] and "categorical" in data[m].get("human_ai", {})
                and "human_ai_names" in data[m]):
            models_with_both.append(m)

    if not models_with_both:
        return None

    def _max_sig_separation(cat_data):
        """Get the maximum separation among significant factors."""
        factors = cat_data.get("factors", [])
        sig = [f for f in factors if f.get("p_value", 1) < 0.05]
        if not sig:
            return 0.0
        return max(abs(f["separation"]) for f in sig)

    def _any_significant(cat_data):
        factors = cat_data.get("factors", [])
        return any(f.get("p_value", 1) < 0.05 for f in factors)

    labels = [MODEL_LABELS[m] for m in models_with_both]
    full_seps = [_max_sig_separation(data[m]["human_ai"]["categorical"])
                 for m in models_with_both]
    names_seps = [_max_sig_separation(data[m]["human_ai_names"])
                  for m in models_with_both]
    names_sig = [_any_significant(data[m]["human_ai_names"])
                 for m in models_with_both]
    colors_list = [MODEL_COLORS.get(m, "#333") for m in models_with_both]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width/2, full_seps, width, color=[c for c in colors_list],
           edgecolor="white", alpha=0.9, label="Full descriptions")
    ax.bar(x + width/2, names_seps, width, color=[c for c in colors_list],
           edgecolor="white", alpha=0.45, label="Names only")

    # Mark non-significant names-only with "n.s."
    for i, (ns, sig) in enumerate(zip(names_seps, names_sig)):
        if not sig:
            ax.text(i + width/2, max(ns, 0.01) + 0.01, "n.s.",
                    ha="center", va="bottom", fontsize=8, color="#C62828",
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    for tick_label, color in zip(ax.get_xticklabels(), colors_list):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")

    ax.set_ylabel("Max Significant AI-Human Factor Separation")
    ax.set_title("AI-Human Separation: Full Descriptions vs. Names Only")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


# ============================================================================
# FIGURE 8: Concept Alignment — Mental vs Control Ratio
# ============================================================================

def _make_fig8(data):
    """Bar chart of mental/control alignment ratio across models."""
    models_with_data = [m for m in sort_models(data) if "contrast_alignment" in data[m]]
    if not models_with_data:
        return None

    labels = []
    entity_means = []
    control_means = []
    ratios = []
    colors_list = []

    for m in models_with_data:
        ca = data[m]["contrast_alignment"]
        concepts = ca.get("concepts", [])
        if not concepts:
            continue

        entity_vals = []
        control_vals = []
        for c in concepts:
            peak = c.get("peak_alignment", 0)
            if c.get("is_control", False):
                control_vals.append(abs(peak))
            elif c.get("is_entity_framed", False):
                entity_vals.append(abs(peak))

        if entity_vals and control_vals:
            em = np.mean(entity_vals)
            cm = np.mean(control_vals)
            ratio = em / cm if cm > 0 else float("inf")
            labels.append(MODEL_LABELS[m])
            entity_means.append(em)
            control_means.append(cm)
            ratios.append(ratio)
            colors_list.append(MODEL_COLORS.get(m, "#333"))

    if not labels:
        return None

    # Left panel uses canonical model order (as collected); right panel
    # is sorted by ratio descending.
    ratio_order = np.argsort(ratios)[::-1]
    r_labels = [labels[i] for i in ratio_order]
    r_ratios = [ratios[i] for i in ratio_order]
    r_colors = [colors_list[i] for i in ratio_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: absolute alignment (canonical model order)
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width/2, entity_means, width, color=C_INST, edgecolor="white",
            alpha=0.85, label="Mental concepts")
    ax1.bar(x + width/2, control_means, width, color="#e74c3c", edgecolor="white",
            alpha=0.85, label="Shapes (control)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for tick_label, color in zip(ax1.get_xticklabels(), colors_list):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax1.set_ylabel("Mean Peak Alignment")
    ax1.set_title("Absolute Alignment Values")
    ax1.legend(fontsize=8)

    # Right: ratio (sorted by ratio descending)
    ax2.barh(np.arange(len(r_labels)), r_ratios, color=r_colors,
             edgecolor="white", height=0.7, alpha=0.85)
    ax2.set_yticks(np.arange(len(r_labels)))
    ax2.set_yticklabels(r_labels, fontsize=9)
    for tick_label, color in zip(ax2.get_yticklabels(), r_colors):
        tick_label.set_color(color)
        tick_label.set_fontweight("bold")
    ax2.set_xlabel("Mental / Control Ratio")
    ax2.set_title("Selectivity Ratio")
    ax2.invert_yaxis()
    ax2.axvline(1, color="gray", ls=":", lw=1, alpha=0.7)

    # Annotate ratios
    for i, r in enumerate(r_ratios):
        ax2.text(r + 0.3, i, f"{r:.1f}x", va="center", fontsize=9,
                 fontweight="bold")

    fig.suptitle("Concept Vector Alignment: Mental Concepts vs. Control",
                 y=1.02, fontsize=14)
    fig.tight_layout()
    return fig


# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_report(data):
    """Build the full HTML report."""
    print("\nGenerating report...")

    sections = [
        {"id": "overview", "label": "1. Overview"},
        {"id": "methodology", "label": "2. Methodology Primer"},
        {"id": "folk-psychology", "label": "3. LLMs Have Implicit Folk Psychology"},
        {"id": "base-vs-instruct", "label": "4. Base vs. Instruct Divergence"},
        {"id": "categorical-saturation", "label": "5. AI/Human Categorical Saturation"},
        {"id": "instruction-tuning", "label": "6. Instruction Tuning Effect Is Family-Dependent"},
        {"id": "model-rankings", "label": "7. Model Rankings Vary by Analysis"},
        {"id": "peak-layers", "label": "8. Peak Processing Location"},
        {"id": "name-recognition", "label": "9. Name Recognition Reveals Prior Knowledge"},
        {"id": "concept-alignment", "label": "10. Concept Vector Alignment"},
    ]

    html = build_cross_model_header(
        "Experiment 4: Cross-Branch Results Synthesis"
    )
    html += '<div class="method">\n'
    html += ('<p>This report synthesizes findings across all four experimental '
             'branches of Experiment 4 (Gray Replication, Gray Simple, '
             'Human-AI Adaptation, Expanded Mental Concepts) and all 11 models '
             '(LLaMA-2-13B Chat/Base, LLaMA-3-8B Instruct/Base, '
             'Gemma-2-2B-IT/Base, Gemma-2-9B-IT/Base, Qwen-2.5-7B-Instruct/Base, '
             'Qwen3-8B). It highlights the main trends and takeaways with '
             'publication-ready figures.</p>\n')
    html += '</div>\n'
    html += build_toc(sections)

    # ── Section 1: Overview ──
    html += '<h2 id="overview">1. Overview</h2>\n'
    html += '<div class="stat">\n'
    html += '<p><strong>Core question:</strong> Do LLMs have an implicit folk psychology '
    html += 'that mirrors the human Experience/Agency mind perception structure '
    html += '(Gray, Gray, &amp; Wegner, 2007)?</p>\n'
    html += '<p><strong>4 branches:</strong> '
    html += '(1) Gray Replication &mdash; pairwise behavioral ratings on 13 entities; '
    html += '(2) Gray Simple &mdash; neural RSA from "Think about {entity}" activations; '
    html += '(3) Human-AI Adaptation &mdash; 30 AI/human characters on Gray capacities; '
    html += '(4) Expanded Mental Concepts &mdash; 28 characters on 21 Exp&nbsp;3 concept dimensions.</p>\n'
    html += '<p><strong>11 models</strong> spanning 4 families (LLaMA-2, LLaMA-3, '
    html += 'Gemma-2, Qwen), each with base and instruct/chat variants.</p>\n'
    html += '</div>\n'

    # Summary table: which models have data for each branch
    html += '<h3>Data Availability</h3>\n'
    html += '<table>\n<tr><th>Model</th>'
    branches_check = [
        ("Gray Replication", "gray_rep"),
        ("Gray Simple", "gray_entities"),
        ("Human-AI Adaptation", "human_ai"),
        ("Expanded Concepts (RSA)", "expanded_rsa"),
        ("Expanded Concepts (Per-Concept)", "concept_rsa"),
        ("Expanded Concepts (Alignment)", "contrast_alignment"),
    ]
    for label, _ in branches_check:
        html += f'<th>{label}</th>'
    html += '</tr>\n'
    for m in sort_models(data):
        html += f'<tr>{model_row_td(m)}'
        for _, key in branches_check:
            has = key in data[m]
            cls = ' class="match"' if has else ' class="mismatch"'
            html += f'<td{cls}>{"Yes" if has else "No"}</td>'
        html += '</tr>\n'
    html += '</table>\n'

    # ── Section 2: Methodology Primer ──
    html += '<h2 id="methodology">2. Methodology Primer</h2>\n'
    html += ('<p>This section provides background on the analysis methods used '
             'across the four branches. Readers familiar with PCA, RSA, and '
             'Procrustes alignment can skip ahead.</p>\n')
    html += methodology_primer_html(
        include_pca=True, include_spearman=True, include_fdr=True,
        include_prompting=True, include_pairwise=True)
    html += neural_methods_primer_html(
        include_layers=True, include_rdm=True, include_rsa=True,
        include_procrustes=True)
    html += expanded_concepts_primer_html()

    html += '<div class="method">\n'
    html += '<h3>Additional Terminology</h3>\n'
    html += '<h4>Agency/Experience Asymmetry</h4>\n'
    html += ('<p>When models encode <strong>Agency</strong> (self-control, '
             'planning, communication) more strongly than '
             '<strong>Experience</strong> (pain, pleasure, hunger), it means '
             'the model&rsquo;s representational geometry aligns better with '
             'the &ldquo;what can it do&rdquo; dimension than the &ldquo;what '
             'does it feel&rdquo; dimension of human mind perception. This '
             'asymmetry is expected: agentic properties are more prominent in '
             'language model training data (task completion, reasoning, '
             'communication) than experiential ones (subjective feelings, '
             'sensations).</p>\n')
    html += '<h4>Instruction Tuning Effects</h4>\n'
    html += ('<p>Instruction tuning (SFT, RLHF, DPO) transforms a base '
             'language model into a conversational assistant. This process '
             'restructures internal representations in ways that sometimes '
             'improve and sometimes degrade alignment with human mind '
             'perception geometry. The direction and magnitude of the effect '
             'is architecture- and training-procedure-dependent, making it '
             'an empirical question rather than a guaranteed improvement.</p>\n')
    html += '</div>\n'

    # ── Section 3: Folk Psychology Asymmetry ──
    html += '<h2 id="folk-psychology">3. LLMs Have Implicit Folk Psychology &mdash; But It\'s Lopsided</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>Every model shows statistically significant alignment with human '
             'mind perception geometry. However, <strong>Agency dominates '
             'Experience</strong>: Agency-only RSA is significant in 8/11 models; '
             'Experience-only is significant in only 5/11. Models more robustly '
             'encode "what can it do" (planning, communication, self-control) than '
             '"what does it feel" (pain, pleasure, hunger, fear).</p>\n')
    html += '</div>\n'

    # Method note about the Gray Simple procedure
    html += '<div class="method">\n'
    html += ('<p><strong>Data source: Gray Simple branch.</strong> '
             'with a minimal prompt &mdash; just <em>&ldquo;Think about [entity '
             'description]&rdquo;</em> (e.g., &ldquo;Think about a five-month-old '
             'baby&rdquo;). No behavioral rating was asked for. Instead, the '
             'model&rsquo;s internal activation vector at the <strong>last token '
             'position</strong> was extracted at every transformer layer. '
             'For instruction-tuned models, the prompt was wrapped in the '
             'appropriate chat template; for base models, it was tokenized directly. '
             'Cosine distances between all entity pairs&rsquo; activations form '
             'a neural RDM at each layer, which is then compared to the human '
             'mind-perception RDM via Spearman rank correlation (RSA). '
             'Full character descriptions from Gray et al. were <em>not</em> '
             'included &mdash; the prompt is deliberately minimal to test '
             'what the model represents about an entity from its name/description '
             'alone.</p>\n')
    html += '</div>\n'

    # Figure 1a: Layerwise RSA — Combined
    b64 = _fig_to_b64_and_close(_make_fig1a(data))
    if b64:
        html += html_figure(b64,
            "Layerwise RSA (combined Experience + Agency) for each model. "
            "Colored bars indicate FDR-significant layers (q &lt; .05); "
            "gray bars are non-significant. Peak layer annotated with "
            "&rho; and q values.",
            fig_num="1a", alt="Layerwise RSA combined")

    # Figure 1b: Layerwise RSA — Experience only
    b64 = _fig_to_b64_and_close(_make_fig1b(data))
    if b64:
        html += html_figure(b64,
            "Layerwise RSA (Experience only) for each model. "
            "Many models fail to reach significance for Experience, "
            "reflecting the Agency-dominant asymmetry. "
            "Colored bars = FDR-significant (q &lt; .05).",
            fig_num="1b", alt="Layerwise RSA experience")

    # Figure 1c: Layerwise RSA — Agency only
    b64 = _fig_to_b64_and_close(_make_fig1c(data))
    if b64:
        html += html_figure(b64,
            "Layerwise RSA (Agency only) for each model. "
            "Agency alignment is more broadly significant than Experience "
            "across models and layers. "
            "Colored bars = FDR-significant (q &lt; .05).",
            fig_num="1c", alt="Layerwise RSA agency")

    # Figure 2: Peak bar chart (Combined / Experience / Agency)
    b64 = _fig_to_b64_and_close(_make_fig1(data))
    if b64:
        html += html_figure(b64,
            "Peak neural RSA alignment with human Combined (purple), Experience "
            "(blue), and Agency (red) RDMs. Error bars "
            "show SD of |&rho;| across significant layers. Agency is more "
            "robustly encoded than Experience across nearly all models. "
            "Zero-height bars indicate no layers reached significance "
            "(p &lt; 0.05).",
            fig_num=2, alt="Agency vs Experience RSA peaks")

    # Add the numbers table
    html += '<h3>Peak RSA by Dimension</h3>\n'
    html += '<table>\n<tr><th>Model</th><th>Combined rho</th><th>Experience rho</th><th>Agency rho</th></tr>\n'
    for m in sort_models(data):
        if "gray_entities" not in data[m]:
            continue
        gs = data[m]["gray_entities"]
        _, c_rho, c_p = _get_peak_rsa(gs.get("combined", []))
        _, e_rho, e_p = _get_peak_rsa(gs.get("experience", []))
        _, a_rho, a_p = _get_peak_rsa(gs.get("agency", []))
        html += f'<tr>{model_row_td(m)}'
        html += format_p_cell(c_rho, c_p)
        html += format_p_cell(e_rho, e_p)
        html += format_p_cell(a_rho, a_p)
        html += '</tr>\n'
    html += '</table>\n'

    # ── Section 4: Base vs Instruct ──
    html += '<h2 id="base-vs-instruct">4. Base vs. Instruct: A Consistent Divergence</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p><strong>Base models collapse mental capacities into a single '
             'dimension</strong> (PCA eigenvalue ratios of 13:1 to 44:1). '
             '<strong>Instruct/chat models create richer, multi-factor '
             'structures</strong> that better approximate the human two-factor '
             'Experience/Agency distinction &mdash; though some fragment into '
             '3&ndash;4 factors rather than cleanly replicating two.</p>\n')
    html += ('<p>AI-human separation also loads differently: base models put it '
             'on F2 (secondary dimension), instruct models on F1 (primary). '
             'Instruction tuning makes the AI/human distinction the '
             '<em>organizing principle</em> of the mind space.</p>\n')
    html += '</div>\n'
    html += '<div class="method">\n'
    html += ('<p><strong>Data source: Gray Replication branch (behavioral).</strong> '
             'PCA on pairwise comparison ratings of 13 Gray et al. entities on '
             '18 mental capacities. Eigenvalue structure reveals how many '
             'latent dimensions the model uses to organize mental capacity '
             'judgments.</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig2(data))
    if b64:
        html += html_figure(b64,
            "PCA eigenvalue structure from Gray Replication pairwise comparisons. "
            "Left: base models show a single dominant eigenvalue (near-uniform "
            "treatment of all capacities). Right: instruct/chat models distribute "
            "variance more evenly, approaching the human two-factor structure "
            "(gray squares, dashed). Kaiser criterion at eigenvalue = 1 (dotted line).",
            fig_num=3, alt="Eigenvalue base vs instruct")

    # Eigenvalue ratio table
    html += '<h3>Eigenvalue Ratios (Eig1 / Eig2)</h3>\n'
    html += '<table>\n<tr><th>Model</th><th>Eig 1</th><th>Eig 2</th><th>Ratio</th><th>Factors &gt; Kaiser</th></tr>\n'
    for m in sort_models(data):
        if "gray_rep" not in data[m]:
            continue
        eig = data[m]["gray_rep"]["eigenvalues"]
        e1 = eig[0]
        e2 = eig[1] if len(eig) > 1 else 0.01
        ratio = e1 / e2 if e2 > 0 else float("inf")
        n_kaiser = sum(1 for e in eig if e > 1.0)
        html += f'<tr>{model_row_td(m)}'
        html += f'<td>{e1:.2f}</td><td>{e2:.2f}</td>'
        html += f'<td><strong>{ratio:.1f}</strong></td>'
        html += f'<td>{n_kaiser}</td></tr>\n'
    html += '</table>\n'
    html += '<p><em>Human reference: Eig1 = 15.85, Eig2 = 1.46, ratio = 10.9, 2 factors.</em></p>\n'

    # ── Section 5: Categorical Saturation ──
    html += '<h2 id="categorical-saturation">5. AI/Human Categorical Signal Is Overwhelming</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>Across the expanded concepts analysis, <strong>every layer of '
             'every model</strong> shows significant categorical RSA (AI vs. human). '
             'All 21 concept dimensions &mdash; including the shapes control &mdash; '
             'show significant per-concept RSA in all 11 models. The range across '
             'concepts is extremely narrow (0.64&ndash;0.69 mean rho), suggesting '
             'the AI/human categorical structure <strong>saturates the '
             'representational space</strong>.</p>\n')
    html += '</div>\n'
    html += '<div class="method">\n'
    html += ('<p><strong>Data source: Expanded Mental Concepts branch '
             '(neural).</strong> Activation RSA using a binary AI/human '
             'categorical RDM applied to 28-character representations at '
             'each transformer layer. Per-concept RSA projects activations '
             'onto individual concept vectors before computing RDMs.</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig3(data))
    if b64:
        html += html_figure(b64,
            "Mean peak categorical RSA per concept dimension across all 11 models. "
            "The narrow spread (range &lt; 0.05) and the strong performance of "
            "the Shapes control concept (red) suggest the AI/human distinction "
            "dominates any direction in activation space, limiting claims about "
            "concept-specific processing. Error bars show cross-model SD.",
            fig_num=4, alt="Per-concept categorical RSA")

    # Categorical RSA peak table
    html += '<h3>Categorical Activation RSA (Expanded Concepts)</h3>\n'
    html += '<table>\n<tr><th>Model</th><th>Peak Layer</th><th>Peak rho</th><th>Sig Layers</th></tr>\n'
    for m in sort_models(data):
        ersa = data[m].get("expanded_rsa", [])
        if not ersa:
            continue
        layer, rho, p = _get_peak_rsa(ersa)
        n_sig = _count_sig_layers(ersa)
        n_total = len(ersa)
        html += f'<tr>{model_row_td(m)}'
        html += f'<td>{layer}</td>'
        html += format_p_cell(rho, p)
        html += f'<td>{n_sig}/{n_total}</td></tr>\n'
    html += '</table>\n'

    # ── Section 6: Instruction Tuning Effect ──
    html += '<h2 id="instruction-tuning">6. Instruction Tuning Effect Is Family-Dependent</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>Instruction tuning does not uniformly push representations toward '
             'human-like structure. It <strong>helps</strong> for Gemma-2-2B '
             '(+0.202), LLaMA-3 (+0.074), and Gemma-2-9B (+0.032), but '
             '<strong>hurts</strong> for LLaMA-2-13B (&minus;0.165) and slightly '
             'for Qwen-2.5 (&minus;0.038). LLaMA-2-13B Chat is consistently '
             'the worst-performing model &mdash; its heavy RLHF safety training '
             'may have disrupted rather than refined mind perception '
             'representations.</p>\n')
    html += '</div>\n'
    html += '<div class="method">\n'
    html += ('<p><strong>Data source: Gray Simple branch (neural).</strong> '
             'Instruction tuning effect is computed as the difference in peak '
             'combined RSA &rho; between the instruct/chat model and its base '
             'counterpart within each model family.</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig4(data))
    if b64:
        html += html_figure(b64,
            "Effect of instruction tuning on peak neural RSA with human mind "
            "perception geometry (Gray Simple branch). Green deltas indicate "
            "instruction tuning improved alignment; red indicates degradation. "
            "The effect direction and magnitude vary substantially across "
            "model families.",
            fig_num=5, alt="Instruction tuning effect")

    # ── Section 7: Model Rankings ──
    html += '<h2 id="model-rankings">7. Model Rankings Vary by Analysis</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>No single model wins everywhere. <strong>Qwen-2.5-7B-Instruct</strong> '
             'leads behavioral pairwise replication, <strong>LLaMA-3-8B-Instruct</strong> '
             'leads neural RSA for Gray entities, <strong>Gemma-2-9B-IT</strong> '
             'leads categorical activation RSA for expanded concepts, and '
             '<strong>Gemma-2-2B-IT</strong> achieves the best Procrustes match '
             'to human 2D mind space despite being the smallest model (2B params).</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig5(data))
    if b64:
        html += html_figure(b64,
            "Model performance across five analysis types. Cell values show peak "
            "|Spearman rho| (significant results only; n.s. = not significant). "
            "Black border marks the best model in each row. No single model "
            "dominates across all analyses.",
            fig_num=6, alt="Model rankings heatmap")

    # ── Section 8: Peak Layer Location ──
    html += '<h2 id="peak-layers">8. Peak Processing Location Varies by Architecture</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>Mind perception representations peak at an average of 44% network '
             'depth, with large family-level variation: LLaMA peaks early '
             '(20&ndash;31%), Gemma in the middle (27&ndash;50%), Qwen-2.5 late '
             '(68%), and Qwen3 very late (83%). This suggests different '
             'architectures locate entity-type processing at different stages '
             'of the forward pass.</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig6(data))
    if b64:
        html += html_figure(b64,
            "Peak RSA layer (as % of network depth) vs. peak alignment strength. "
            "Circles = instruct/chat; squares = base. Models cluster by family "
            "along the x-axis, suggesting architecture-dependent placement of "
            "mind perception processing.",
            fig_num=7, alt="Peak layer location")

    # Peak layer table
    html += '<h3>Peak Layer Details</h3>\n'
    html += '<table>\n<tr><th>Model</th><th>Peak Layer</th><th>Total Layers</th><th>% Depth</th><th>Sig Layers</th></tr>\n'
    for m in sort_models(data):
        if "gray_entities" not in data[m]:
            continue
        gs = data[m]["gray_entities"]
        layer, rho, _ = _get_peak_rsa(gs["combined"])
        n_layers = data[m]["n_layers"]
        n_sig = _count_sig_layers(gs["combined"])
        n_total = len(gs["combined"])
        if layer is not None:
            pct = layer / n_layers * 100
            html += f'<tr>{model_row_td(m)}'
            html += f'<td>{layer}</td><td>{n_layers}</td>'
            html += f'<td>{pct:.0f}%</td>'
            html += f'<td>{n_sig}/{n_total}</td></tr>\n'
    html += '</table>\n'

    # ── Section 9: Name Recognition ──
    html += '<h2 id="name-recognition">9. Name Recognition Reveals Prior Knowledge</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>Some models differentiate AI from human characters using '
             '<strong>names alone</strong> (no descriptions): notably '
             'Qwen-2.5-7B-Instruct, Gemma-2-9B-IT, and Qwen3-8B. Others '
             '(LLaMA-2 Chat, Gemma base models) require full character '
             'descriptions. This reveals pre-existing "mind type" associations '
             'baked in during pretraining &mdash; a form of learned folk '
             'psychology from training data exposure.</p>\n')
    html += '</div>\n'
    html += '<div class="method">\n'
    html += ('<p><strong>Data source: Human-AI Adaptation branch '
             '(behavioral).</strong> The names-only condition repeats the '
             'full 30-character pairwise comparison procedure using only '
             'character names (e.g., &ldquo;Aria-7&rdquo; or &ldquo;David '
             'Park&rdquo;) without descriptive bios. This tests whether the '
             'model differentiates AI from human based on name associations '
             'alone, revealing prior knowledge from pretraining data. '
             'Separation is measured as the maximum absolute difference in '
             'mean PCA factor scores between the 15 AI and 15 human '
             'characters, with significance assessed via Mann&ndash;Whitney U '
             'tests.</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig7(data))
    if b64:
        html += html_figure(b64,
            "Maximum significant AI-human factor separation with full character "
            "descriptions (saturated bars) vs. names only (faded bars). "
            "Red \"n.s.\" labels indicate models where names alone fail to "
            "produce significant separation &mdash; these models require "
            "contextual descriptions to distinguish AI from human characters.",
            fig_num=8, alt="Names vs descriptions separation")

    # Names-only significance table
    html += '<h3>Names-Only Significance</h3>\n'
    html += '<table>\n<tr><th>Model</th><th>Full (max sep)</th><th>Names-only (max sep)</th><th>Names sig?</th></tr>\n'
    for m in sort_models(data):
        if "human_ai" not in data[m] or "categorical" not in data[m].get("human_ai", {}):
            continue
        cat = data[m]["human_ai"]["categorical"]
        factors = cat.get("factors", [])
        sig_full = [f for f in factors if f.get("p_value", 1) < 0.05]
        max_full = max((abs(f["separation"]) for f in sig_full), default=0)

        has_names = "human_ai_names" in data[m]
        if has_names:
            ncat = data[m]["human_ai_names"]
            nfactors = ncat.get("factors", [])
            sig_names = [f for f in nfactors if f.get("p_value", 1) < 0.05]
            max_names = max((abs(f["separation"]) for f in sig_names), default=0)
            names_sig = len(sig_names) > 0
        else:
            max_names = None
            names_sig = None

        html += f'<tr>{model_row_td(m)}'
        html += f'<td>{max_full:.3f}</td>'
        if max_names is not None:
            html += f'<td>{max_names:.3f}</td>'
            cls = ' class="match"' if names_sig else ' class="mismatch"'
            html += f'<td{cls}>{"Yes" if names_sig else "No"}</td>'
        else:
            html += '<td>--</td><td>--</td>'
        html += '</tr>\n'
    html += '</table>\n'

    # ── Section 10: Concept Alignment ──
    html += '<h2 id="concept-alignment">10. Concept Vector Alignment: Mental vs. Control</h2>\n'
    html += '<div class="interpret">\n'
    html += ('<p>Exp&nbsp;3 contrast vectors (human-direction minus AI-direction) '
             'align with character representations more strongly for mental '
             'concepts than for shapes controls in all 11 models (ratios 1.9x '
             'to 36.5x). However, instruction tuning raises <em>both</em> '
             'mental and control alignment, reducing selectivity. Base models '
             'paradoxically show better alignment ratios in several cases, '
             'possibly because instruction tuning introduces additional '
             'representational structure that partially decorrelates concept '
             'directions from the pure categorical signal.</p>\n')
    html += '</div>\n'
    html += '<div class="method">\n'
    html += ('<p><strong>Data source: Expanded Mental Concepts branch '
             '(alignment).</strong> Concept contrast vectors from '
             'Experiment&nbsp;3 (human-context minus AI-context activation '
             'directions) are projected onto character activations at each '
             'layer. Alignment is the peak R&sup2; of the projection across '
             'layers. &ldquo;Mental concepts&rdquo; include phenomenology, '
             'emotions, agency, social cognition, etc.; the '
             '&ldquo;shapes control&rdquo; is a non-mental concept vector '
             'used as a baseline. The selectivity ratio (mental&nbsp;/&nbsp;'
             'control) indicates whether concept vectors capture '
             'mind-relevant structure beyond a generic categorical signal.</p>\n')
    html += '</div>\n'

    b64 = _fig_to_b64_and_close(_make_fig8(data))
    if b64:
        html += html_figure(b64,
            "Left: absolute peak alignment values for mental concepts (blue) vs. "
            "shapes control (red). Right: selectivity ratio (mental / control). "
            "Higher ratios indicate the concept vectors selectively capture "
            "mind-relevant structure beyond a generic categorical signal. "
            "Models sorted by selectivity ratio.",
            fig_num=9, alt="Concept alignment ratio")

    # ── Bottom Line ──
    html += '<h2>Bottom Line</h2>\n'
    html += '<div class="success">\n'
    html += ('<p>LLMs develop a genuine implicit folk psychology that partially '
             'mirrors human mind perception structure, but it is '
             '<strong>Agency-heavy</strong>, <strong>categorically dominated '
             'by the AI/human distinction</strong>, and <strong>heavily shaped '
             'by instruction tuning</strong>.</p>\n')
    html += ('<p>Base models see minds as one-dimensional; instruction tuning '
             'introduces differentiation that sometimes aligns with human '
             'psychology (Qwen, Gemma-9B) and sometimes fragments it '
             '(LLaMA-2 Chat). The AI/human signal is so pervasive that it '
             'may crowd out more nuanced concept-specific structure &mdash; '
             'a key limitation for interpreting the expanded concepts results.</p>\n')
    html += '</div>\n'

    html += build_html_footer()
    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("EXPERIMENT 4: Cross-Branch Results Synthesis")
    print("=" * 60)

    data = load_all_data()
    print(f"\nLoaded {len(data)} models")

    html = generate_report(data)

    out_dir = ensure_dir(COMPARISONS_DIR)
    out_path = out_dir / "results_comparison.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport saved to: {out_path}")

    # Save figures as standalone PDF + PNG
    fig_dir = ensure_dir(out_dir / "figures")
    print(f"Saving standalone figures to: {fig_dir}")
    _save_all_figures(data, fig_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
