#!/usr/bin/env python3
"""
Experiment 4: Cross-Model Gray Replication Summary Report

Generates an HTML report comparing Gray et al. (2007) mind perception
replication results across all available models. Includes scree plots,
factor loadings, mind perception space, human correlations, and
individual ratings (all models).

Reads pre-computed data from:
    results/{model}/gray_replication/behavior/with_self/data/

Usage:
    python comparisons/3_gray_replication_summary_generator.py

Env: llama2_env (login node, CPU only)
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
from scipy.stats import spearmanr

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, COMPARISONS_DIR, VALID_MODELS, MODELS, ensure_dir, set_model, data_dir
from utils.report_utils import (
    REPORT_CSS, build_cross_model_header, build_html_footer, build_toc,
    fig_to_b64, html_figure, MODEL_COLORS, MODEL_LABELS, ALL_MODELS,
    gray_entities_stimuli_html, sort_models, GRID_NCOLS, make_model_grid,
    model_row_td, format_p_cell,
    methodology_primer_html,
)
from utils.utils import nice_entity, nice_capacity
from entities.gray_entities import (
    GRAY_ET_AL_SCORES, CAPACITY_PROMPTS, ENTITY_NAMES, N_ENTITIES, N_CAPACITIES,
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
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_EXP = "#2166ac"   # blue  -- Experience factor
C_AGE = "#b2182b"   # red   -- Agency factor
C_HUM = "#555555"   # gray  -- Human reference


# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_data():
    """Load gray_replication pairwise data for all available models.

    Returns dict keyed by model name with sub-dicts containing:
        pca, correlations, consistency, entity_keys, cap_keys,
        eigenvalues, loadings, scores_01,
        and optionally individual_pca, individual_matrix.
    """
    model_data = {}

    for model in VALID_MODELS:
        set_model(model)
        ddir = data_dir("gray_replication", "behavior", "with_self")
        pca_path = ddir / "pairwise_pca_results.npz"

        if not pca_path.exists():
            print(f"  Skipping {model}: no pairwise_pca_results.npz")
            continue

        try:
            pca = np.load(pca_path, allow_pickle=True)
            entry = {
                "label": MODEL_LABELS[model],
                "is_chat": MODELS[model]["is_chat"],
                "eigenvalues": pca["eigenvalues"],
                "loadings": pca["rotated_loadings"],       # (18, 2)
                "scores_01": pca["factor_scores_01"],       # (13, 2)
                "entity_keys": list(pca["entity_keys"]),
                "cap_keys": list(pca["capacity_keys"]),
            }

            # Human correlations
            corr_path = ddir / "pairwise_human_correlations.json"
            if corr_path.exists():
                with open(corr_path) as f:
                    entry["correlations"] = json.load(f)
            else:
                # Compute from factor scores
                entry["correlations"] = _compute_correlations(
                    entry["scores_01"], entry["entity_keys"]
                )

            # Consistency stats
            cons_path = ddir / "pairwise_consistency_stats.json"
            if cons_path.exists():
                with open(cons_path) as f:
                    entry["consistency"] = json.load(f)

            # Individual ratings (base models)
            ind_pca_path = ddir / "individual_pca_results.npz"
            if ind_pca_path.exists():
                ind_pca = np.load(ind_pca_path, allow_pickle=True)
                entry["individual_pca"] = {
                    "eigenvalues": ind_pca["eigenvalues"],
                    "loadings": ind_pca["rotated_loadings"],
                    "scores_01": ind_pca["factor_scores_01"],
                    "entity_keys": list(ind_pca["entity_keys"]),
                    "cap_keys": list(ind_pca["capacity_keys"]),
                }
                # Individual correlations
                ind_corr_path = ddir / "individual_human_correlations.json"
                if ind_corr_path.exists():
                    with open(ind_corr_path) as f:
                        entry["individual_correlations"] = json.load(f)
                else:
                    entry["individual_correlations"] = _compute_correlations(
                        ind_pca["factor_scores_01"],
                        list(ind_pca["entity_keys"]),
                    )

            ind_mat_path = ddir / "individual_rating_matrix.npz"
            if ind_mat_path.exists():
                ind_mat = np.load(ind_mat_path, allow_pickle=True)
                entry["individual_matrix"] = {
                    "matrix": ind_mat["rating_matrix"],     # (18, 13)
                    "entity_keys": list(ind_mat["entity_keys"]),
                    "cap_keys": list(ind_mat["capacity_keys"]),
                }

            # Behavioral RSA
            rsa_path = ddir / "behavioral_rsa_results.json"
            if rsa_path.exists():
                with open(rsa_path) as f:
                    entry["behavioral_rsa"] = json.load(f)

            rdm_path = ddir / "behavioral_rdms.npz"
            if rdm_path.exists():
                rdm_data = np.load(rdm_path, allow_pickle=True)
                entry["behavioral_rdms"] = {
                    k: rdm_data[k] for k in rdm_data.files
                }

            model_data[model] = entry
            print(f"  Loaded {model} ({len(entry['entity_keys'])} entities)")

        except Exception as e:
            print(f"  Error loading {model}: {e}")
            continue

    return model_data


def _compute_correlations(scores_01, entity_keys):
    """Compute Spearman correlations between model factors and human scores."""
    human_exp = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    human_age = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])
    n_factors = min(2, scores_01.shape[1])
    corr = {}
    for fi in range(n_factors):
        rho_e, p_e = spearmanr(scores_01[:, fi], human_exp)
        rho_a, p_a = spearmanr(scores_01[:, fi], human_age)
        corr[f"f{fi+1}_experience"] = {"rho": float(rho_e), "p_value": float(p_e)}
        corr[f"f{fi+1}_agency"] = {"rho": float(rho_a), "p_value": float(p_a)}
    return corr


def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _get_p(corr_entry):
    """Get p-value from a correlation dict, handling both key names."""
    if "p_value" in corr_entry:
        return corr_entry["p_value"]
    return corr_entry.get("p", 1.0)


# ============================================================================
# FIGURE GENERATORS
# ============================================================================

def fig_scree_comparison(model_data):
    """Scree plot: eigenvalue curves for all models + human reference."""
    human_eig = np.array([15.85, 1.46])

    fig, ax = plt.subplots(figsize=(7, 5))

    # Human reference
    ax.plot([1, 2], human_eig, "s--", color=C_HUM, lw=2, ms=8, zorder=10,
            label="Humans (Gray et al.)")
    ax.annotate(f"{human_eig[0]:.2f}", (1, human_eig[0]),
                textcoords="offset points", xytext=(-45, 5), fontsize=9,
                color=C_HUM)
    ax.annotate(f"{human_eig[1]:.2f}", (2, human_eig[1]),
                textcoords="offset points", xytext=(-35, 5), fontsize=9,
                color=C_HUM)

    max_n = 0
    for model in sort_models(model_data):
        d = model_data[model]
        eig = d["eigenvalues"]
        n = len(eig)
        if n > max_n:
            max_n = n
        x = np.arange(1, n + 1)
        color = MODEL_COLORS.get(model, "#333333")
        ax.plot(x, eig, "o-", color=color, lw=2, ms=5,
                label=d["label"], alpha=0.85)
        # Annotate first two eigenvalues
        ax.annotate(f"{eig[0]:.2f}", (1, eig[0]),
                    textcoords="offset points", xytext=(8, -8), fontsize=8,
                    color=color)
        if n > 1:
            ax.annotate(f"{eig[1]:.2f}", (2, eig[1]),
                        textcoords="offset points", xytext=(8, 4), fontsize=8,
                        color=color)

    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7, label="Kaiser criterion")
    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Comparison: Pairwise PCA")
    ax.set_xlim(0.5, min(max_n, 10) + 0.5)
    ax.set_ylim(-0.5, max(human_eig[0] + 1, 18))
    ax.legend(loc="upper right", fontsize=9)

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def fig_loadings_combined(model_data, sorted_models):
    """Combined loading chart: family-grouped rows, F1+F2 side by side per model."""
    exp_caps = {c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"}
    LOADING_THRESH = 0.4

    positions, ordered, nrows, ncols, _ = make_model_grid(sorted_models)
    actual_ncols = ncols * 2  # F1 + F2 per model
    fig, axes = plt.subplots(nrows, actual_ncols,
                             figsize=(4 * actual_ncols, 5 * nrows),
                             squeeze=False)

    # Hide all axes first, then show the ones we use
    for ax in axes.flatten():
        ax.set_visible(False)

    for idx, model in enumerate(ordered):
        d = model_data[model]
        loadings = d["loadings"]  # (18, 2)
        cap_keys = d["cap_keys"]
        y = np.arange(len(cap_keys))
        labels = [nice_capacity(c) for c in cap_keys]
        colors = [C_EXP if c in exp_caps else C_AGE for c in cap_keys]

        row, col = positions[idx]
        col_base = col * 2
        ax_f1 = axes[row, col_base]
        ax_f2 = axes[row, col_base + 1]
        ax_f1.set_visible(True)
        ax_f2.set_visible(True)

        # Factor 1
        ax_f1.barh(y, loadings[:, 0], color=colors, edgecolor="white", height=0.7)
        ax_f1.set_xlabel("Loading")
        ax_f1.set_title(f"{MODEL_LABELS[model]}\nFactor 1", fontsize=10)
        ax_f1.set_yticks(y)
        ax_f1.set_yticklabels(labels, fontsize=8)
        ax_f1.axvline(0, color="gray", lw=0.5)
        ax_f1.axvline(LOADING_THRESH, color="gray", lw=1, ls=":", alpha=0.7)
        ax_f1.axvline(-LOADING_THRESH, color="gray", lw=1, ls=":", alpha=0.7)
        ax_f1.invert_yaxis()

        # Factor 2
        ax_f2.barh(y, loadings[:, 1], color=colors, edgecolor="white", height=0.7)
        ax_f2.set_xlabel("Loading")
        ax_f2.set_title("Factor 2", fontsize=10)
        ax_f2.set_yticks(y)
        ax_f2.set_yticklabels([], fontsize=8)
        ax_f2.axvline(0, color="gray", lw=0.5)
        ax_f2.axvline(LOADING_THRESH, color="gray", lw=1, ls=":", alpha=0.7)
        ax_f2.axvline(-LOADING_THRESH, color="gray", lw=1, ls=":", alpha=0.7)
        ax_f2.invert_yaxis()

    legend_elements = [
        Patch(facecolor=C_EXP, label="Experience (human)"),
        Patch(facecolor=C_AGE, label="Agency (human)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    fig.suptitle("Varimax-Rotated Factor Loadings: All Models", y=1.01, fontsize=14)
    fig.tight_layout()

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def _entity_green_colors():
    """Compute a dark-to-light purple color for each entity based on human scores.

    Uses Euclidean distance from origin in the human (Experience, Agency) space
    mapped to the Purples colormap: light purple = low scores, dark purple = high.
    Returns dict[entity_key] → RGBA color.
    """
    cmap = plt.cm.Purples
    # Compute combined score (Euclidean distance from origin, normalised to 0-1)
    dists = {}
    for e in ENTITY_NAMES:
        exp, age = GRAY_ET_AL_SCORES[e]
        dists[e] = np.sqrt(exp ** 2 + age ** 2)
    d_min, d_max = min(dists.values()), max(dists.values())
    rng = d_max - d_min if d_max > d_min else 1.0
    # Map to 0.15–0.9 range of Greens cmap (avoid near-white and near-black)
    colors = {}
    for e in ENTITY_NAMES:
        norm = (dists[e] - d_min) / rng
        colors[e] = cmap(0.15 + 0.75 * norm)
    return colors


def fig_mind_space(model_data):
    """Entity scatter plots: human reference + one panel per model.

    Uses a dark-to-light green gradient: each entity's color is derived from
    its human mind perception score (dark green = high, light green = low).
    Model panels use the same entity colors so deviations from human positions
    are visually intuitive.
    """
    entity_colors = _entity_green_colors()
    positions, ordered, nrows, ncols, human_pos = make_model_grid(
        model_data.keys(), include_human=True
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)

    # Hide all axes first
    for ax in axes.flatten():
        ax.set_visible(False)

    # Human reference panel
    ax_human = axes[human_pos[0], human_pos[1]]
    ax_human.set_visible(True)
    h_exp = np.array([GRAY_ET_AL_SCORES[e][0] for e in ENTITY_NAMES])
    h_age = np.array([GRAY_ET_AL_SCORES[e][1] for e in ENTITY_NAMES])
    h_colors = [entity_colors[e] for e in ENTITY_NAMES]
    ax_human.scatter(h_age, h_exp, s=70, c=h_colors, edgecolor="white", zorder=5, linewidth=0.5)
    for i, ek in enumerate(ENTITY_NAMES):
        ax_human.annotate(nice_entity(ek), (h_age[i], h_exp[i]),
                    textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax_human.set_xlabel("Agency")
    ax_human.set_ylabel("Experience")
    ax_human.set_title("Human (Gray et al., 2007)", fontsize=11)
    ax_human.set_xlim(-0.05, 1.12)
    ax_human.set_ylim(-0.05, 1.12)
    ax_human.set_aspect("equal")

    # Model panels — same entity colors as human panel
    for idx, model in enumerate(ordered):
        d = model_data[model]
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        scores = d["scores_01"]      # (13, 2)
        ek = d["entity_keys"]
        m_colors = [entity_colors.get(e, "#999999") for e in ek]
        ax.scatter(scores[:, 0], scores[:, 1], s=70, c=m_colors,
                   edgecolor="white", zorder=5, linewidth=0.5)
        for i, e in enumerate(ek):
            ax.annotate(nice_entity(e), (scores[i, 0], scores[i, 1]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7.5)
        ax.set_xlabel("Factor 1")
        ax.set_ylabel("Factor 2")
        ax.set_title(d["label"], fontsize=11)
        ax.set_xlim(-0.1, 1.15)
        ax.set_ylim(-0.1, 1.15)
        ax.set_aspect("equal")

    fig.suptitle("Mind Perception Space: Human vs Models", fontsize=14, y=1.02)
    fig.tight_layout()

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def fig_individual_heatmap(model, d):
    """Heatmap of individual Likert ratings for a single model (legacy)."""
    mat_info = d["individual_matrix"]
    matrix = mat_info["matrix"]          # (18, 13)
    entity_keys = mat_info["entity_keys"]
    cap_keys = mat_info["cap_keys"]

    exp_caps = {c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"}

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=2.5, vmax=3.5)
    ax.set_xticks(range(len(entity_keys)))
    ax.set_xticklabels([nice_entity(e) for e in entity_keys], rotation=45,
                       ha="right", fontsize=9)
    ax.set_yticks(range(len(cap_keys)))
    ylabels = [f"{nice_capacity(c)} ({'E' if c in exp_caps else 'A'})"
               for c in cap_keys]
    ax.set_yticklabels(ylabels, fontsize=9)

    for i in range(len(cap_keys)):
        for j in range(len(entity_keys)):
            val = matrix[i, j]
            color = "white" if val < 2.8 or val > 3.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Expected Rating (1-5)")
    ax.set_title(f"Individual Likert Ratings: {d['label']}")
    fig.tight_layout()

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def fig_individual_heatmaps_combined(model_data, models_with_individual):
    """All individual Likert rating heatmaps in a family-grouped grid layout."""
    n = len(models_with_individual)
    if n == 0:
        return None

    exp_caps = {c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"}

    positions, ordered, nrows, ncols, _ = make_model_grid(models_with_individual)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 7 * nrows),
                             squeeze=False, layout="constrained")

    # Hide all axes first
    for ax in axes.flatten():
        ax.set_visible(False)

    im = None
    visible_axes = []
    for idx, model in enumerate(ordered):
        d = model_data[model]
        mat_info = d["individual_matrix"]
        matrix = mat_info["matrix"]          # (18, 13)
        entity_keys = mat_info["entity_keys"]
        cap_keys = mat_info["cap_keys"]

        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        visible_axes.append(ax)
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r",
                       vmin=2.5, vmax=3.5)
        ax.set_xticks(range(len(entity_keys)))
        ax.set_xticklabels([nice_entity(e) for e in entity_keys],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(d["label"], fontsize=10)

        # Show y-axis labels on the first column only
        if col == 0:
            ax.set_yticks(range(len(cap_keys)))
            ylabels = [f"{nice_capacity(c)} ({'E' if c in exp_caps else 'A'})"
                       for c in cap_keys]
            ax.set_yticklabels(ylabels, fontsize=8)
        else:
            ax.set_yticks(range(len(cap_keys)))
            ax.set_yticklabels([])

        # Cell values (skip if too many models to keep readable)
        if n <= 4:
            for i in range(len(cap_keys)):
                for j in range(len(entity_keys)):
                    val = matrix[i, j]
                    color = "white" if val < 2.8 or val > 3.35 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5, color=color)

    fig.suptitle("Individual Likert Ratings: All Models", fontsize=13)
    if im is not None:
        fig.colorbar(im, ax=visible_axes, shrink=0.6,
                     label="Expected Rating (1-5)")

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def _best_individual_correlation(d):
    """Find the strongest significant factor-human correlation for a model.

    Returns (factor_idx, human_dim_name, human_arr, color, rho, p) or None.
    """
    ind = d["individual_pca"]
    scores = ind["scores_01"]
    ek = ind["entity_keys"]
    h_exp = np.array([GRAY_ET_AL_SCORES[e][0] for e in ek])
    h_age = np.array([GRAY_ET_AL_SCORES[e][1] for e in ek])

    candidates = []
    for fi in range(min(2, scores.shape[1])):
        for h_arr, h_name, h_color in [(h_exp, "Experience", C_EXP),
                                        (h_age, "Agency", C_AGE)]:
            rho, p = spearmanr(scores[:, fi], h_arr)
            candidates.append((fi, h_name, h_arr, h_color, rho, p))

    # Pick best significant; if none significant, pick highest |rho|
    sig = [c for c in candidates if c[5] < 0.05]
    if sig:
        best = max(sig, key=lambda c: abs(c[4]))
    else:
        best = max(candidates, key=lambda c: abs(c[4]))
    return best


def fig_individual_scatter_combined(model_data, models_with_individual):
    """Combined cross-model scatter: each panel shows the strongest factor-human
    correlation for that model's individual ratings PCA."""
    if not models_with_individual:
        return None

    positions, ordered, nrows, ncols, _ = make_model_grid(models_with_individual)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)

    for idx, model in enumerate(ordered):
        d = model_data[model]
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)

        fi, h_name, h_arr, h_color, rho, p = _best_individual_correlation(d)
        ind = d["individual_pca"]
        scores = ind["scores_01"]
        ek = ind["entity_keys"]
        stars = _sig_stars(p)

        ax.scatter(h_arr, scores[:, fi], s=70, c=h_color,
                   edgecolor="white", zorder=5, linewidth=0.5, alpha=0.85)
        for i, e in enumerate(ek):
            ax.annotate(nice_entity(e), (h_arr[i], scores[i, fi]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7.5, alpha=0.8)
        # Trend line
        z = np.polyfit(h_arr, scores[:, fi], 1)
        xline = np.linspace(-0.05, 1.12, 100)
        ax.plot(xline, np.polyval(z, xline), "--", color=h_color, alpha=0.3)

        ax.set_xlabel(f"Human {h_name}")
        ax.set_ylabel(f"Model Factor {fi+1} (0-1)")
        ax.set_title(f"{MODEL_LABELS[model]}\nF{fi+1} vs {h_name} "
                     f"(ρ={rho:.3f}{stars})", fontsize=10)
        ax.set_xlim(-0.05, 1.12)
        ax.set_ylim(-0.1, 1.15)

    fig.suptitle("Individual Ratings: Best Factor-Human Correlation per Model",
                 fontsize=14)
    fig.tight_layout()

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


def fig_behavioral_rdm_heatmaps(model_data):
    """RDM heatmaps: human reference + one per model with behavioral RSA data.

    Shows combined human RDM plus model behavioral RDMs (pairwise PCA source).
    All panels share the same colormap range for visual comparability.
    """
    models_with_rsa = sort_models([m for m in model_data if "behavioral_rdms" in model_data[m]])
    if not models_with_rsa:
        return None

    # Get entity keys from first model with RSA data
    first_rdms = model_data[models_with_rsa[0]]["behavioral_rdms"]
    entity_keys = list(first_rdms["entity_keys"])

    # Collect all RDMs for shared colorscale (in canonical order)
    all_rdms = [first_rdms["human_rdm_combined"]]
    model_rdm_list = []
    for m in models_with_rsa:
        rdms = model_data[m]["behavioral_rdms"]
        if "model_rdm_pairwise_pca" in rdms:
            model_rdm_list.append((m, rdms["model_rdm_pairwise_pca"]))
            all_rdms.append(rdms["model_rdm_pairwise_pca"])

    if not model_rdm_list:
        return None

    vmax = max(r.max() for r in all_rdms)
    rdm_models = [m for m, _ in model_rdm_list]
    positions, ordered, nrows, ncols, human_pos = make_model_grid(
        rdm_models, include_human=True
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                             squeeze=False, layout="constrained")

    # Hide all axes first
    for ax in axes.flatten():
        ax.set_visible(False)

    labels = [nice_entity(e) for e in entity_keys]

    # Human panel
    ax_human = axes[human_pos[0], human_pos[1]]
    ax_human.set_visible(True)
    im = ax_human.imshow(first_rdms["human_rdm_combined"], cmap="viridis",
                         vmin=0, vmax=vmax, aspect="equal")
    ax_human.set_xticks(range(len(entity_keys)))
    ax_human.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax_human.set_yticks(range(len(entity_keys)))
    ax_human.set_yticklabels(labels, fontsize=7)
    ax_human.set_title("Human (Combined)", fontsize=10)

    # Build model key → RDM lookup
    rdm_by_model = {m: rdm for m, rdm in model_rdm_list}

    # Model panels
    visible_axes = [ax_human]
    for idx, model in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        visible_axes.append(ax)
        rdm = rdm_by_model[model]
        ax.imshow(rdm, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(entity_keys)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(entity_keys)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(model_data[model]["label"], fontsize=10)

    fig.suptitle("Behavioral RDMs: Human vs Model (Pairwise PCA)", fontsize=13)
    fig.colorbar(im, ax=visible_axes, shrink=0.5,
                 label="Distance")

    b64 = fig_to_b64(fig)
    plt.close(fig)
    return b64


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report():
    print("=" * 60)
    print("Gray Replication: Cross-Model Summary Report")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    model_data = load_model_data()

    if not model_data:
        print("ERROR: No model data found. Cannot generate report.")
        return None

    models_loaded = sort_models(model_data)
    n_models = len(models_loaded)
    print(f"\nLoaded {n_models} model(s): {', '.join(models_loaded)}")

    # Build sections list
    sections = [
        {"id": "overview", "label": "1. Overview"},
        {"id": "stimuli", "label": "2. Stimuli"},
        {"id": "methods", "label": "3. Methods"},
        {"id": "scree", "label": "4. Scree Plot Comparison"},
        {"id": "loadings", "label": "5. Factor Loading Comparison"},
        {"id": "mind-space", "label": "6. Mind Perception Space"},
        {"id": "correlations", "label": "7. Human Correlation Summary"},
    ]

    # Check if any model has individual data
    has_individual = any("individual_matrix" in d for d in model_data.values())
    if has_individual:
        sections.append({"id": "individual", "label": "8. Individual Ratings"})

    # Check if any model has behavioral RSA data
    has_rsa = any("behavioral_rsa" in d for d in model_data.values())
    next_num = 8
    if has_individual:
        next_num = 9
    if has_rsa:
        sections.append({"id": "behavioral-rsa", "label": f"{next_num}. Behavioral RSA"})
        next_num += 1
    sections.append({"id": "takeaways", "label": f"{next_num}. Key Takeaways"})

    # Start HTML
    html = build_cross_model_header("Exp 4: Gray Replication - Cross-Model Summary")
    html += f'<p><em>Models included: {", ".join(MODEL_LABELS[m] for m in models_loaded)}</em></p>\n'
    html += build_toc(sections)

    fig_num = 0

    # ── Section 1: Overview ──
    html += '<h2 id="overview">1. Overview</h2>\n'
    html += '<div class="method">\n'
    html += (
        "<p>This report compares how different LLMs replicate the two-factor "
        "structure of human mind perception from Gray, Gray, &amp; Wegner (2007). "
        "The original study asked human participants to make pairwise comparisons "
        "of 13 entities across 18 mental capacities. PCA with varimax rotation "
        "revealed two factors: <strong>Experience</strong> (hunger, fear, pain, "
        "pleasure, rage, desire, personality, consciousness, pride, embarrassment, "
        "joy) and <strong>Agency</strong> (self-control, morality, memory, emotion "
        "recognition, planning, communication, thought).</p>\n"
        "<p>We replicate this paradigm using LLMs. Chat models answer pairwise "
        "comparison questions using their chat template. Base models use completion "
        "prompting with expected ratings derived from token logit distributions.</p>\n"
    )
    html += (
        f"<p><strong>Models evaluated:</strong> {n_models} models. "
        f"Chat: {', '.join(MODEL_LABELS[m] for m in models_loaded if model_data[m]['is_chat'])}. "
        f"Base: {', '.join(MODEL_LABELS[m] for m in models_loaded if not model_data[m]['is_chat'])}.</p>\n"
    )
    html += "</div>\n"

    # ── Section 2: Stimuli ──
    html += gray_entities_stimuli_html(include_capacities=True)

    # ── Section 3: Methods ──
    html += '<h2 id="methods">3. Methods</h2>\n'
    html += '<div class="method">\n'
    html += "<ol>\n"
    html += (
        "<li><strong>Pairwise comparisons:</strong> All 78 entity pairs "
        f"({N_ENTITIES} choose 2) are compared on each of {N_CAPACITIES} "
        "mental capacities. For each pair, the model rates which entity is "
        'more capable (1-5 scale, where 3 = equal). This yields a '
        f'{N_CAPACITIES} x {N_ENTITIES} matrix of relative ratings.</li>\n'
    )
    html += (
        "<li><strong>PCA with varimax rotation:</strong> Following Gray et al., "
        "we compute the correlation matrix of capacities across entities, "
        "extract components with eigenvalue &gt; 1, and apply varimax rotation "
        "to maximize simple structure.</li>\n"
    )
    html += (
        "<li><strong>Factor scores:</strong> Regression-method factor scores "
        "are computed for each entity and rescaled to 0-1 range.</li>\n"
    )
    html += (
        "<li><strong>Human comparison:</strong> Spearman rank correlations "
        "between model factor scores and human Experience/Agency scores from "
        "Gray et al. (2007).</li>\n"
    )
    html += (
        "<li><strong>Individual ratings:</strong> Each entity is rated "
        "individually on each capacity (1-5 Likert scale) using logit-based "
        "extraction (P('1')..P('5') at next-token position). Chat/instruct "
        "models use chat-template wrapping. Same PCA procedure as pairwise.</li>\n"
    )
    html += "</ol>\n</div>\n"

    # ── Interpretation Guide ──
    html += '<h3>Interpretation Guide</h3>\n'
    html += methodology_primer_html(
        include_pca=True, include_spearman=True, include_fdr=False,
        include_prompting=True, include_pairwise=True)

    html += '<div class="method">\n'
    html += '<h4>Human Reference Scores</h4>\n'
    html += (
        '<p>Human Experience and Agency scores were estimated by digitizing '
        'entity positions from Figure 1 of Gray, Gray, &amp; Wegner (2007). '
        'These are approximate x&ndash;y coordinates read from the published '
        '2D scatter plot, rescaled to 0&ndash;1. They serve as the ground-truth '
        'reference against which model factor scores are correlated.</p>\n')
    html += '<h4>Pairwise vs. Individual Paradigms</h4>\n'
    html += (
        '<p>The <strong>pairwise paradigm</strong> presents entities in pairs '
        'and asks the model which is &ldquo;more capable&rdquo; of each mental '
        'capacity. This mirrors Gray et al.&rsquo;s original survey design. '
        'The <strong>individual paradigm</strong> rates each entity alone on a '
        '1&ndash;5 Likert scale for each capacity&mdash;no comparison partner '
        'is shown. Individual ratings provide an absolute rather than relative '
        'measure. Both are analyzed via the same PCA procedure.</p>\n')
    html += '<h4>Behavioral RSA</h4>\n'
    html += (
        '<p><strong>Behavioral RSA</strong> compares the pairwise distance '
        'pattern among entities based on behavioral ratings (capacity win-rates '
        'or absolute Likert scores) to the human reference RDM. This is '
        'distinct from <em>neural</em> RSA (used in the Gray Simple branch), '
        'which uses internal activation vector distances. Behavioral RSA tests '
        'whether the model&rsquo;s <em>overt judgments</em> share geometric '
        'structure with human judgments; neural RSA tests whether the '
        'model&rsquo;s <em>internal representations</em> do.</p>\n')
    html += '</div>\n'

    # ── Section 4: Scree Plot ──
    html += '<h2 id="scree">4. Scree Plot Comparison</h2>\n'
    html += (
        "<p>Eigenvalue profiles from the pairwise rating PCA, with the human "
        "reference from Gray et al. (eigenvalues: 15.85, 1.46). The Kaiser "
        "criterion (eigenvalue &gt; 1) determines how many factors to retain.</p>\n"
    )
    print("\nGenerating scree plot...")
    fig_num += 1
    b64_scree = fig_scree_comparison(model_data)
    html += html_figure(
        b64_scree,
        "Eigenvalue comparison across models. Each line shows the eigenvalue "
        "profile from PCA of pairwise ratings. The human reference (gray squares) "
        "shows the dominant two-factor structure from Gray et al. (2007).",
        fig_num=fig_num,
        alt="Scree plot comparison",
    )

    # Eigenvalue summary table
    html += "<table>\n"
    html += "<tr><th>Model</th><th>Eig 1</th><th>Eig 2</th><th>Eig 3</th>"
    html += "<th>Factors retained</th><th>Var explained (F1+F2)</th></tr>\n"
    html += (
        f"<tr><td>Humans (Gray et al.)</td><td>15.85</td><td>1.46</td>"
        f"<td>--</td><td>2</td><td>{(15.85 + 1.46) / 18 * 100:.1f}%</td></tr>\n"
    )
    for model in models_loaded:
        d = model_data[model]
        eig = d["eigenvalues"]
        n_retained = int(np.sum(eig > 1.0))
        n_retained = max(n_retained, 2)
        var_exp = float(np.sum(eig[:2]) / np.sum(eig) * 100)
        eig3 = f"{eig[2]:.2f}" if len(eig) > 2 else "--"
        html += (
            f"<tr>{model_row_td(model)}"
            f"<td>{eig[0]:.2f}</td><td>{eig[1]:.2f}</td><td>{eig3}</td>"
            f"<td>{n_retained}</td><td>{var_exp:.1f}%</td></tr>\n"
        )
    html += "</table>\n"

    # ── Section 5: Factor Loadings ──
    html += '<h2 id="loadings">5. Factor Loading Comparison</h2>\n'
    html += (
        "<p>Varimax-rotated capacity loadings for each model's two-factor solution. "
        "Bars are colored by the human factor assignment: "
        '<span style="color: ' + C_EXP + '">blue = Experience</span>, '
        '<span style="color: ' + C_AGE + '">red = Agency</span>. '
        "If the model replicates the human structure, Experience capacities should "
        "load heavily on one factor and Agency capacities on the other.</p>\n"
    )

    print("Generating loading plots...")
    fig_num += 1
    b64_load = fig_loadings_combined(model_data, models_loaded)
    html += html_figure(
        b64_load,
        "Varimax-rotated loadings for all models. Each model shows Factor 1 and "
        "Factor 2 side by side. Blue bars = Experience capacities, red = Agency "
        "capacities (human assignment). Dashed lines mark |0.4| threshold.",
        fig_num=fig_num,
        alt="Factor loadings comparison",
    )

    # ── Section 6: Mind Perception Space ──
    html += '<h2 id="mind-space">6. Mind Perception Space</h2>\n'
    html += (
        "<p>Two-dimensional entity maps. The human panel (first) shows the "
        "original Gray et al. space with Agency on X and Experience on Y. "
        "Model panels show entities in their F1/F2 factor space (0-1 "
        "normalized). Qualitative similarity in entity clustering suggests "
        "the model captures aspects of human mind perception.</p>\n"
    )

    print("Generating mind perception space...")
    fig_num += 1
    b64_space = fig_mind_space(model_data)
    html += html_figure(
        b64_space,
        "Mind perception space comparison. First panel: human data from "
        "Gray et al. (2007). Subsequent panels: model factor spaces from "
        "pairwise PCA. Entity labels show where each character falls in "
        "the two-dimensional mind perception space.",
        fig_num=fig_num,
        alt="Mind perception space comparison",
    )

    # ── Section 7: Human Correlation Summary ──
    html += '<h2 id="correlations">7. Human Correlation Summary</h2>\n'
    html += (
        "<p>Spearman rank correlations between each model's pairwise factor "
        "scores and the human Experience/Agency scores from Gray et al. "
        "Significance: * p &lt; .05, ** p &lt; .01, *** p &lt; .001.</p>\n"
    )

    html += "<table>\n"
    html += (
        "<tr><th>Model</th>"
        "<th>F1 vs Exp (rho)</th><th>F1 vs Age (rho)</th>"
        "<th>F2 vs Exp (rho)</th><th>F2 vs Age (rho)</th></tr>\n"
    )

    best_rho = 0.0
    best_model = None
    best_desc = ""

    for model in models_loaded:
        d = model_data[model]
        corr = d["correlations"]
        row = f"<tr>{model_row_td(model)}"
        for key in ["f1_experience", "f1_agency", "f2_experience", "f2_agency"]:
            if key in corr:
                rho = corr[key]["rho"]
                p = _get_p(corr[key])
                row += format_p_cell(rho, p)
                if abs(rho) > best_rho and p < 0.05:
                    best_rho = abs(rho)
                    best_model = model
                    best_desc = f"{key} (rho={rho:.3f})"
            else:
                row += "<td>--</td>"
        row += "</tr>\n"
        html += row
    html += "</table>\n"

    # Interpretation
    html += '<div class="interpret">\n'
    if best_model:
        html += (
            f"<p><strong>Strongest human alignment (pairwise):</strong> "
            f"{MODEL_LABELS[best_model]}, {best_desc}.</p>\n"
        )
    else:
        html += "<p>No significant correlations found with human scores.</p>\n"
    html += "</div>\n"

    # ── Section 8: Individual Ratings ──
    models_with_individual = sort_models([
        m for m in models_loaded if "individual_matrix" in model_data[m]
    ])

    if has_individual:
        html += '<h2 id="individual">8. Individual Ratings</h2>\n'
        html += (
            "<p>Each entity is rated individually on each capacity using a 1-5 "
            "Likert scale, with ratings extracted from the logit distribution "
            "over digit tokens. Chat/instruct models use chat-template wrapping. "
            "This avoids pairwise position bias entirely.</p>\n"
        )

        # Combined heatmap (all models side by side)
        print("  Generating combined individual heatmaps...")
        fig_num += 1
        b64_combined = fig_individual_heatmaps_combined(model_data, models_with_individual)
        if b64_combined:
            html += html_figure(
                b64_combined,
                "Individual Likert rating heatmaps for all models, displayed "
                "side by side for comparison. Each cell shows the expected "
                "rating (1-5 scale) computed from the model's next-token logit "
                "distribution. 3.0 is neutral. Red/warm = higher capacity "
                "attribution; blue/cool = lower. Row labels indicate Experience "
                "(E) or Agency (A) factor assignment from Gray et al.",
                fig_num=fig_num,
                alt="Combined individual rating heatmaps",
            )

        # Combined scatter: best factor-human correlation per model
        models_with_pca = sort_models([
            m for m in models_with_individual if "individual_pca" in model_data[m]
        ])
        if models_with_pca:
            print("  Generating combined individual scatter...")
            fig_num += 1
            b64_scat = fig_individual_scatter_combined(model_data, models_with_pca)
            if b64_scat:
                html += html_figure(
                    b64_scat,
                    "Individual ratings PCA: best factor-human correlation for "
                    "each model. Each panel shows the factor and human dimension "
                    "(Experience or Agency) with the strongest Spearman rho. "
                    "Blue = Experience, red = Agency.",
                    fig_num=fig_num,
                    alt="Combined individual scatter",
                )

        # Combined individual correlation table
        models_with_corr = [
            m for m in models_with_individual
            if "individual_correlations" in model_data[m]
        ]
        if models_with_corr:
            html += "<h3>Individual Correlation Summary</h3>\n"
            html += "<table>\n"
            html += (
                "<tr><th>Model</th>"
                "<th>F1 vs Exp (ρ)</th><th>F1 vs Age (ρ)</th>"
                "<th>F2 vs Exp (ρ)</th><th>F2 vs Age (ρ)</th></tr>\n"
            )
            for model in models_with_corr:
                d = model_data[model]
                ind_corr = d["individual_correlations"]
                color = MODEL_COLORS.get(model, "#333333")
                row = (f'<tr><td style="border-left: 4px solid {color}; '
                       f'font-weight: 600;">{d["label"]}</td>')
                for key in ["f1_experience", "f1_agency",
                            "f2_experience", "f2_agency"]:
                    if key in ind_corr:
                        rho = ind_corr[key]["rho"]
                        p = _get_p(ind_corr[key])
                        stars = _sig_stars(p)
                        css = ' class="sig"' if p < 0.05 else ""
                        row += f"<td{css}>{rho:.3f}{stars}</td>"
                    else:
                        row += "<td>--</td>"
                row += "</tr>\n"
                html += row
            html += "</table>\n"

    # ── Section N: Behavioral RSA ──
    models_with_rsa = sort_models([m for m in models_loaded if "behavioral_rsa" in model_data[m]])
    if has_rsa:
        rsa_num = 9 if has_individual else 8
        html += f'<h2 id="behavioral-rsa">{rsa_num}. Behavioral RSA</h2>\n'
        html += (
            "<p>Representational Similarity Analysis (RSA) compares the model's "
            "behavioral similarity structure across entities with the human "
            "similarity structure from Gray et al. Each RDM captures pairwise "
            "distances between all 13 entities. The RSA statistic is the Spearman "
            "rank correlation between the upper triangles of the model and human "
            "RDMs.</p>\n"
            "<p>Two RDM sources: <strong>Pairwise PCA</strong> (Euclidean distance "
            "in the 2D factor space) and <strong>Individual 18D</strong> (correlation "
            "distance across all 18 capacity ratings). Three human RDM variants: "
            "Combined (2D Euclidean), Experience-only, and Agency-only.</p>\n"
        )

        # RSA Summary Table
        html += "<h3>RSA Summary</h3>\n"
        html += "<table>\n"
        html += (
            "<tr><th>Model</th><th>RDM Source</th>"
            "<th>vs Combined (rho)</th><th>vs Experience (rho)</th>"
            "<th>vs Agency (rho)</th></tr>\n"
        )

        for model in models_with_rsa:
            d = model_data[model]
            rsa = d["behavioral_rsa"]

            for source_key, source_label in [
                ("pairwise_pca", "Pairwise PCA"),
                ("individual_18d", "Individual 18D"),
            ]:
                if source_key not in rsa:
                    continue
                source = rsa[source_key]
                row = f"<tr>{model_row_td(model)}<td>{source_label}</td>"
                for variant in ["combined", "experience", "agency"]:
                    if variant in source:
                        rho = source[variant]["rho"]
                        p = source[variant]["p_value"]
                        row += format_p_cell(rho, p)
                    else:
                        row += "<td>--</td>"
                row += "</tr>\n"
                html += row

        html += "</table>\n"

        # RDM Heatmaps
        print("Generating behavioral RDM heatmaps...")
        b64_rdm = fig_behavioral_rdm_heatmaps(model_data)
        if b64_rdm:
            fig_num += 1
            html += html_figure(
                b64_rdm,
                "Behavioral RDMs (pairwise PCA source). First panel: human "
                "combined RDM (Euclidean distance in Experience x Agency space). "
                "Subsequent panels: model behavioral RDMs (Euclidean distance in "
                "2D factor space). Shared colormap for visual comparability.",
                fig_num=fig_num,
                alt="Behavioral RDM heatmaps",
            )

        # Interpretation
        html += '<div class="interpret">\n'
        best_rsa_rho = 0.0
        best_rsa_model = None
        best_rsa_desc = ""
        for m in models_with_rsa:
            rsa = model_data[m]["behavioral_rsa"]
            for src_key, src_label in [("pairwise_pca", "PCA"), ("individual_18d", "18D")]:
                if src_key in rsa:
                    for var in ["combined", "experience", "agency"]:
                        if var in rsa[src_key]:
                            rho = rsa[src_key][var]["rho"]
                            p = rsa[src_key][var]["p_value"]
                            if p < 0.05 and abs(rho) > best_rsa_rho:
                                best_rsa_rho = abs(rho)
                                best_rsa_model = m
                                best_rsa_desc = f"{src_label} vs {var} (rho={rho:.3f})"
        if best_rsa_model:
            html += (
                f"<p><strong>Strongest behavioral RSA:</strong> "
                f"{MODEL_LABELS[best_rsa_model]}, {best_rsa_desc}.</p>\n"
            )
        else:
            html += "<p>No significant behavioral RSA correlations found.</p>\n"
        html += "</div>\n"

    # ── Section N+1: Key Takeaways ──
    takeaway_num = next_num
    html += f'<h2 id="takeaways">{takeaway_num}. Key Takeaways</h2>\n'

    # Auto-generate takeaways
    takeaways = []

    # Which model best correlates?
    if best_model:
        takeaways.append(
            f"<strong>Strongest pairwise replication:</strong> "
            f"{MODEL_LABELS[best_model]} shows the highest significant "
            f"correlation with human mind perception ({best_desc})."
        )

    # Chat vs base comparison
    chat_models = [m for m in models_loaded if model_data[m]["is_chat"]]
    base_models = [m for m in models_loaded if not model_data[m]["is_chat"]]

    if chat_models and base_models:
        # Compute max significant |rho| for each group
        def _max_sig_rho(model_list):
            best = 0.0
            for m in model_list:
                corr = model_data[m]["correlations"]
                for key in ["f1_experience", "f1_agency", "f2_experience", "f2_agency"]:
                    if key in corr:
                        rho = corr[key]["rho"]
                        p = _get_p(corr[key])
                        if p < 0.05 and abs(rho) > best:
                            best = abs(rho)
            return best

        chat_best = _max_sig_rho(chat_models)
        base_best = _max_sig_rho(base_models)

        if chat_best > 0 and base_best > 0:
            if chat_best > base_best:
                takeaways.append(
                    "<strong>Chat vs base:</strong> Chat models show stronger "
                    "human-aligned mind perception structure in pairwise comparisons "
                    f"(max |rho| = {chat_best:.3f}) compared to base models "
                    f"(max |rho| = {base_best:.3f})."
                )
            else:
                takeaways.append(
                    "<strong>Chat vs base:</strong> Base models show stronger "
                    "human-aligned mind perception structure in pairwise comparisons "
                    f"(max |rho| = {base_best:.3f}) compared to chat models "
                    f"(max |rho| = {chat_best:.3f})."
                )
        elif base_best > 0:
            takeaways.append(
                "<strong>Chat vs base:</strong> Only base models show significant "
                "correlations with human mind perception scores."
            )
        elif chat_best > 0:
            takeaways.append(
                "<strong>Chat vs base:</strong> Only chat models show significant "
                "correlations with human mind perception scores."
            )
        else:
            takeaways.append(
                "<strong>Chat vs base:</strong> Neither chat nor base models show "
                "significant correlations with human mind perception scores in the "
                "pairwise paradigm."
            )

    # Eigenvalue structure
    for model in models_loaded:
        eig = model_data[model]["eigenvalues"]
        dominance = eig[0] / eig[1] if eig[1] > 0.01 else float("inf")
        if dominance > 10:
            takeaways.append(
                f"<strong>{MODEL_LABELS[model]} eigenvalue structure:</strong> "
                f"Highly dominant first factor (eig1/eig2 = {dominance:.1f}), "
                f"suggesting a one-factor solution may better describe the "
                f"model's capacity structure."
            )

    # Individual ratings summary
    if models_with_individual:
        ind_sigs = []
        for m in models_with_individual:
            if "individual_correlations" in model_data[m]:
                for key in ["f1_experience", "f1_agency", "f2_experience", "f2_agency"]:
                    corr = model_data[m]["individual_correlations"]
                    if key in corr and _get_p(corr[key]) < 0.05:
                        ind_sigs.append(
                            f"{MODEL_LABELS[m]} {key.replace('_', ' ')}"
                        )
        if ind_sigs:
            takeaways.append(
                f"<strong>Individual ratings:</strong> Significant human "
                f"correlations found for: {'; '.join(ind_sigs)}."
            )
        else:
            takeaways.append(
                "<strong>Individual ratings:</strong> No significant human "
                "correlations found in individual rating factor scores."
            )

    # Behavioral RSA summary
    if models_with_rsa:
        rsa_sigs = []
        for m in models_with_rsa:
            rsa = model_data[m]["behavioral_rsa"]
            for src_key, src_label in [("pairwise_pca", "PCA"), ("individual_18d", "18D")]:
                if src_key in rsa:
                    for var in ["combined", "experience", "agency"]:
                        if var in rsa[src_key]:
                            p = rsa[src_key][var]["p_value"]
                            rho = rsa[src_key][var]["rho"]
                            if p < 0.05:
                                rsa_sigs.append(
                                    f"{MODEL_LABELS[m]} {src_label} vs {var} "
                                    f"(rho={rho:.3f})"
                                )
        if rsa_sigs:
            takeaways.append(
                "<strong>Behavioral RSA:</strong> Significant model-human RSA: "
                + "; ".join(rsa_sigs) + "."
            )
        else:
            takeaways.append(
                "<strong>Behavioral RSA:</strong> No significant RSA correlations "
                "between model and human behavioral RDMs."
            )

    if not takeaways:
        takeaways.append(
            "Insufficient data across models to draw comparative conclusions. "
            "Run additional model pipelines and regenerate this report."
        )

    html += '<div class="stat">\n<ol>\n'
    for t in takeaways:
        html += f"<li>{t}</li>\n"
    html += "</ol>\n</div>\n"

    # Attribution
    html += '<hr>\n<p style="font-size: 0.85em; color: #888;">Rachel C. Metzgar &middot; Mar 2026</p>\n'

    html += build_html_footer()

    # ── Write ──
    out_path = ensure_dir(COMPARISONS_DIR) / "gray_replication_summary.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nReport written to: {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    generate_report()
