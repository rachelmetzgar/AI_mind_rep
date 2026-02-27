#!/usr/bin/env python3
"""
Generate publication-quality figures for Experiment 4.

Figures cover both the base model (pairwise + individual ratings) and
the chat model (RSA of entity activations). All figures saved to the
respective results/ directories.

Usage:
    python generate_figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # ai_mind_rep/

BASE_DIR = os.path.join(SCRIPT_DIR)  # llama_exp_4-13B-base/
CHAT_DIR = os.path.join(PROJECT_ROOT, "exp_4", "llama_exp_4-13B-chat")

# Data paths — base model
PAIR_DATA = os.path.join(BASE_DIR, "data", "behavioral_replication")
INDIV_DATA = os.path.join(BASE_DIR, "data", "individual_ratings")

# Data paths — chat model
CHAT_ACTIV = os.path.join(CHAT_DIR, "data", "entity_activations")
CHAT_RSA = os.path.join(CHAT_DIR, "results")

# Output directories
PAIR_FIG_DIR = os.path.join(BASE_DIR, "results", "behavioral_replication", "figures")
INDIV_FIG_DIR = os.path.join(BASE_DIR, "results", "individual_ratings", "figures")
CHAT_FIG_DIR = os.path.join(CHAT_DIR, "results", "figures")

sys.path.insert(0, os.path.join(BASE_DIR, "entities"))
from gray_entities import (
    GRAY_ET_AL_SCORES, CHARACTER_NAMES, CAPACITY_PROMPTS,
    ENTITY_NAMES, N_ENTITIES, CAPACITY_NAMES, N_CAPACITIES,
)

# ── Style ────────────────────────────────────────────────────────────
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

# Color scheme
C_EXP = "#2166ac"   # blue  — Experience
C_AGE = "#b2182b"   # red   — Agency
C_MOD = "#4daf4a"   # green — Model
C_HUM = "#984ea3"   # purple — Human


# ── Helpers ──────────────────────────────────────────────────────────
def nice_entity(name):
    """Pretty-print entity names."""
    lookup = {
        "dead_woman": "Dead woman",
        "frog": "Frog",
        "robot": "Robot (Kismet)",
        "fetus": "Fetus (7 wk)",
        "pvs_patient": "PVS patient",
        "god": "God",
        "dog": "Dog",
        "chimpanzee": "Chimpanzee",
        "baby": "Baby (5 mo)",
        "girl": "Girl (5 yr)",
        "adult_woman": "Adult woman",
        "adult_man": "Adult man",
        "you_self": "You (self)",
    }
    return lookup.get(name, name)


def nice_capacity(name):
    """Pretty-print capacity names."""
    return name.replace("_", " ").replace("emotion recognition", "emotion recog.").title()


def load_human_scores(entity_keys):
    """Get human Experience and Agency arrays aligned to entity_keys."""
    exp = np.array([GRAY_ET_AL_SCORES[e][0] for e in entity_keys])
    age = np.array([GRAY_ET_AL_SCORES[e][1] for e in entity_keys])
    return exp, age


def save_fig(fig, path, name):
    """Save figure as both PNG and PDF."""
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, f"{name}.png"))
    fig.savefig(os.path.join(path, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf → {path}")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 1: Scree plot — eigenvalue comparison (model vs human)
# ═════════════════════════════════════════════════════════════════════
def fig1_scree_plot():
    """Compare eigenvalue structure: pairwise model vs Gray et al. humans."""
    print("Fig 1: Scree plot (eigenvalue comparison)")

    pca = np.load(os.path.join(PAIR_DATA, "with_self", "pca_results.npz"))
    model_eig = pca["eigenvalues"]

    # Gray et al. human eigenvalues (from paper: 15.85 + 1.46)
    human_eig = np.array([15.85, 1.46])

    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = np.arange(1, len(model_eig) + 1)
    ax.plot(x, model_eig, "o-", color=C_MOD, label="LLaMA-2-13B (base)", lw=2, ms=5)
    ax.plot([1, 2], human_eig, "s--", color=C_HUM, label="Humans (Gray et al.)",
            lw=2, ms=8, zorder=5)
    ax.axhline(1.0, color="gray", ls=":", lw=1, alpha=0.7, label="Kaiser criterion")

    # Annotate eigenvalues
    ax.annotate(f"{model_eig[0]:.2f}", (1, model_eig[0]), textcoords="offset points",
                xytext=(12, -5), fontsize=9, color=C_MOD)
    ax.annotate(f"{model_eig[1]:.2f}", (2, model_eig[1]), textcoords="offset points",
                xytext=(12, 2), fontsize=9, color=C_MOD)
    ax.annotate(f"{human_eig[0]:.2f}", (1, human_eig[0]), textcoords="offset points",
                xytext=(-40, 8), fontsize=9, color=C_HUM)
    ax.annotate(f"{human_eig[1]:.2f}", (2, human_eig[1]), textcoords="offset points",
                xytext=(-35, 8), fontsize=9, color=C_HUM)

    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Eigenvalue Comparison: Model vs Human")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-0.3, 17)
    ax.legend(loc="upper right")

    save_fig(fig, PAIR_FIG_DIR, "fig1_scree_plot")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 2: Factor loading comparison (pairwise)
# ═════════════════════════════════════════════════════════════════════
def fig2_loading_comparison():
    """Side-by-side loadings: model F1/F2 vs human Experience/Agency."""
    print("Fig 2: Factor loading comparison")

    pca = np.load(os.path.join(PAIR_DATA, "with_self", "pca_results.npz"),
                  allow_pickle=True)
    loadings = pca["rotated_loadings"]  # (18, 2)
    cap_keys = list(pca["capacity_keys"])

    # Human factor assignment
    exp_caps = [c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"]
    age_caps = [c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "A"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

    y = np.arange(len(cap_keys))
    labels = [nice_capacity(c) for c in cap_keys]
    colors = [C_EXP if c in exp_caps else C_AGE for c in cap_keys]

    # Panel A: Factor 1
    axes[0].barh(y, loadings[:, 0], color=colors, edgecolor="white", height=0.7)
    axes[0].set_xlabel("Loading")
    axes[0].set_title("Model Factor 1")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].invert_yaxis()

    # Panel B: Factor 2
    axes[1].barh(y, loadings[:, 1], color=colors, edgecolor="white", height=0.7)
    axes[1].set_xlabel("Loading")
    axes[1].set_title("Model Factor 2")
    axes[1].axvline(0, color="gray", lw=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_EXP, label="Experience capacity (human)"),
        Patch(facecolor=C_AGE, label="Agency capacity (human)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Varimax-Rotated Capacity Loadings (Pairwise, 13 entities)", y=1.01)
    fig.tight_layout()

    save_fig(fig, PAIR_FIG_DIR, "fig2_loading_comparison")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 3: Entity scatter — Model F2 vs Human Experience
# ═════════════════════════════════════════════════════════════════════
def fig3_entity_scatter():
    """Scatter: model factor scores vs human Experience/Agency."""
    print("Fig 3: Entity scatter (model vs human)")

    pca = np.load(os.path.join(PAIR_DATA, "with_self", "pca_results.npz"),
                  allow_pickle=True)
    scores_01 = pca["factor_scores_01"]  # (13, 2)
    entity_keys = list(pca["entity_keys"])
    h_exp, h_age = load_human_scores(entity_keys)

    # The significant correlation is F2 vs Experience
    rho_f2_exp, p_f2_exp = stats.spearmanr(scores_01[:, 1], h_exp)
    rho_f2_age, p_f2_age = stats.spearmanr(scores_01[:, 1], h_age)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: Model F2 vs Human Experience
    ax = axes[0]
    ax.scatter(h_exp, scores_01[:, 1], s=80, c=C_EXP, edgecolor="white",
               zorder=5, linewidth=0.5)
    for i, ek in enumerate(entity_keys):
        ax.annotate(nice_entity(ek), (h_exp[i], scores_01[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    ha="left")
    # Trend line
    z = np.polyfit(h_exp, scores_01[:, 1], 1)
    xline = np.linspace(0, 1, 100)
    ax.plot(xline, np.polyval(z, xline), "--", color=C_EXP, alpha=0.4)

    sig_str = "***" if p_f2_exp < 0.001 else "**" if p_f2_exp < 0.01 else "*" if p_f2_exp < 0.05 else "n.s."
    ax.set_title(f"Model Factor 2 vs Human Experience\n"
                 f"rho = {rho_f2_exp:.3f}, p = {p_f2_exp:.4f} {sig_str}")
    ax.set_xlabel("Human Experience (Gray et al.)")
    ax.set_ylabel("Model Factor 2 (0-1)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.1, 1.15)

    # Panel B: Model F2 vs Human Agency
    ax = axes[1]
    ax.scatter(h_age, scores_01[:, 1], s=80, c=C_AGE, edgecolor="white",
               zorder=5, linewidth=0.5)
    for i, ek in enumerate(entity_keys):
        ax.annotate(nice_entity(ek), (h_age[i], scores_01[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8,
                    ha="left")
    z = np.polyfit(h_age, scores_01[:, 1], 1)
    ax.plot(xline, np.polyval(z, xline), "--", color=C_AGE, alpha=0.4)

    sig_str = "***" if p_f2_age < 0.001 else "**" if p_f2_age < 0.01 else "*" if p_f2_age < 0.05 else "n.s."
    ax.set_title(f"Model Factor 2 vs Human Agency\n"
                 f"rho = {rho_f2_age:.3f}, p = {p_f2_age:.4f} {sig_str}")
    ax.set_xlabel("Human Agency (Gray et al.)")
    ax.set_ylabel("Model Factor 2 (0-1)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.1, 1.15)

    fig.suptitle("Base Model Pairwise: Entity Positions vs Human Mind Perception",
                 fontsize=14, y=1.03)
    fig.tight_layout()

    save_fig(fig, PAIR_FIG_DIR, "fig3_entity_scatter_pairwise")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 4: Mind perception space — 2D entity plot (model vs human)
# ═════════════════════════════════════════════════════════════════════
def fig4_mind_space():
    """Side-by-side 2D entity maps: human vs model."""
    print("Fig 4: Mind perception space (2D entity map)")

    pca = np.load(os.path.join(PAIR_DATA, "with_self", "pca_results.npz"),
                  allow_pickle=True)
    scores_01 = pca["factor_scores_01"]  # (13, 2)
    entity_keys = list(pca["entity_keys"])
    h_exp, h_age = load_human_scores(entity_keys)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Human (Gray et al. Fig 1 recreation)
    ax = axes[0]
    ax.scatter(h_age, h_exp, s=80, c="#555555", edgecolor="white",
               zorder=5, linewidth=0.5)
    for i, ek in enumerate(entity_keys):
        ax.annotate(nice_entity(ek), (h_age[i], h_exp[i]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Agency")
    ax.set_ylabel("Experience")
    ax.set_title("Human Mind Perception\n(Gray et al., 2007)")
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_aspect("equal")

    # Panel B: Model
    ax = axes[1]
    ax.scatter(scores_01[:, 0], scores_01[:, 1], s=80, c=C_MOD,
               edgecolor="white", zorder=5, linewidth=0.5)
    for i, ek in enumerate(entity_keys):
        ax.annotate(nice_entity(ek), (scores_01[i, 0], scores_01[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Factor 1")
    ax.set_ylabel("Factor 2")
    ax.set_title("LLaMA-2-13B (Base) Mind Perception\n(Pairwise Ratings)")
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect("equal")

    fig.suptitle("Mind Perception Space: Human vs Model", fontsize=14, y=1.02)
    fig.tight_layout()

    save_fig(fig, PAIR_FIG_DIR, "fig4_mind_space_comparison")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 5: Rating heatmap (individual ratings)
# ═════════════════════════════════════════════════════════════════════
def fig5_rating_heatmap():
    """Heatmap of individual Likert ratings (entities x capacities)."""
    print("Fig 5: Individual rating heatmap")

    d = np.load(os.path.join(INDIV_DATA, "with_self", "rating_matrix.npz"),
                allow_pickle=True)
    matrix = d["rating_matrix"]        # (18, 13)
    entity_keys = list(d["entity_keys"])
    cap_keys = list(d["capacity_keys"])

    # Human factor assignment for color-coding y-axis
    exp_caps = [c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"]

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=2.5, vmax=3.5)
    ax.set_xticks(range(len(entity_keys)))
    ax.set_xticklabels([nice_entity(e) for e in entity_keys], rotation=45,
                       ha="right", fontsize=9)
    ax.set_yticks(range(len(cap_keys)))
    ylabels = []
    for c in cap_keys:
        label = nice_capacity(c)
        factor = "E" if c in exp_caps else "A"
        ylabels.append(f"{label} ({factor})")
    ax.set_yticklabels(ylabels, fontsize=9)

    # Add text annotations
    for i in range(len(cap_keys)):
        for j in range(len(entity_keys)):
            val = matrix[i, j]
            color = "white" if val < 2.8 or val > 3.35 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Expected Rating (1-5)")
    ax.set_title("Individual Likert Ratings: LLaMA-2-13B (Base)\n"
                 "Expected rating from logit distribution")

    fig.tight_layout()
    save_fig(fig, INDIV_FIG_DIR, "fig5_rating_heatmap")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 6: Individual ratings — entity scatter vs human
# ═════════════════════════════════════════════════════════════════════
def fig6_individual_scatter():
    """Scatter for individual ratings: F1 vs human Agency (significant result)."""
    print("Fig 6: Individual ratings entity scatter")

    # Use without_self data (stronger result, matches Gray et al.)
    pca_ws = np.load(os.path.join(INDIV_DATA, "without_self", "pca_results.npz"),
                     allow_pickle=True)
    scores_ws = pca_ws["factor_scores_01"]
    ek_ws = list(pca_ws["entity_keys"])
    h_exp_ws, h_age_ws = load_human_scores(ek_ws)

    # Also load with_self for comparison
    pca = np.load(os.path.join(INDIV_DATA, "with_self", "pca_results.npz"),
                  allow_pickle=True)
    scores = pca["factor_scores_01"]
    entity_keys = list(pca["entity_keys"])
    h_exp, h_age = load_human_scores(entity_keys)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Panel A: with_self F1 vs Human Agency
    ax = axes[0]
    rho, p = stats.spearmanr(scores[:, 0], h_age)
    ax.scatter(h_age, scores[:, 0], s=80, c=C_AGE, edgecolor="white",
               zorder=5, linewidth=0.5)
    for i, ek in enumerate(entity_keys):
        ax.annotate(nice_entity(ek), (h_age[i], scores[i, 0]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    z = np.polyfit(h_age, scores[:, 0], 1)
    xline = np.linspace(-0.05, 1.1, 100)
    ax.plot(xline, np.polyval(z, xline), "--", color=C_AGE, alpha=0.4)
    sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.set_title(f"With self (13 entities)\nrho = {rho:.3f}, p = {p:.4f} {sig_str}")
    ax.set_xlabel("Human Agency (Gray et al.)")
    ax.set_ylabel("Model Factor 1 (0-1)")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.1, 1.15)

    # Panel B: without_self F1 vs Human Agency
    ax = axes[1]
    rho, p = stats.spearmanr(scores_ws[:, 0], h_age_ws)
    ax.scatter(h_age_ws, scores_ws[:, 0], s=80, c=C_AGE, edgecolor="white",
               zorder=5, linewidth=0.5)
    for i, ek in enumerate(ek_ws):
        ax.annotate(nice_entity(ek), (h_age_ws[i], scores_ws[i, 0]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    z = np.polyfit(h_age_ws, scores_ws[:, 0], 1)
    ax.plot(xline, np.polyval(z, xline), "--", color=C_AGE, alpha=0.4)
    sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.set_title(f"Without self (12 entities)\nrho = {rho:.3f}, p = {p:.4f} {sig_str}")
    ax.set_xlabel("Human Agency (Gray et al.)")
    ax.set_ylabel("Model Factor 1 (0-1)")
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.1, 1.15)

    fig.suptitle("Individual Ratings: Model Factor 1 vs Human Agency",
                 fontsize=14, y=1.03)
    fig.tight_layout()

    save_fig(fig, INDIV_FIG_DIR, "fig6_individual_entity_scatter")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 7: RSA across layers (chat model entity activations)
# ═════════════════════════════════════════════════════════════════════
def fig7_rsa_layerwise():
    """RSA correlation (model RDM vs human RDM) across transformer layers."""
    print("Fig 7: RSA across layers (chat model)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, cond in enumerate(["without_self", "with_self"]):
        rsa_path = os.path.join(CHAT_RSA, cond, "rsa_all_layers.json")
        with open(rsa_path) as f:
            rsa = json.load(f)

        layers = [r["layer"] for r in rsa]
        rhos = [r["rho"] for r in rsa]
        pvals = [r["p_value"] for r in rsa]

        ax = axes[idx]
        # Replace NaN with 0 for plotting
        rhos_clean = [r if not (isinstance(r, float) and np.isnan(r)) else 0.0
                      for r in rhos]
        pvals_clean = [p if not (isinstance(p, float) and np.isnan(p)) else 1.0
                       for p in pvals]

        # Color points by significance
        sig = [p < 0.05 for p in pvals_clean]
        colors = [C_MOD if s else "#cccccc" for s in sig]

        ax.bar(layers, rhos_clean, color=colors, edgecolor="white", width=0.8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Spearman rho (model RDM vs human RDM)")

        n_sig = sum(sig)
        n_ent = 13 if cond == "with_self" else 12
        ax.set_title(f"{cond.replace('_', ' ').title()} ({n_ent} entities)\n"
                     f"{n_sig}/{len(layers)} layers p < .05")
        ax.set_ylim(-0.35, 0.55)

        # Mark peak (skip NaN)
        peak_idx = int(np.nanargmax([r if not np.isnan(r) else -np.inf
                                      for r in rhos_clean]))
        ax.annotate(f"peak: rho={rhos_clean[peak_idx]:.3f}\nlayer {layers[peak_idx]}",
                    (layers[peak_idx], rhos[peak_idx]),
                    textcoords="offset points", xytext=(15, 10),
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))

    fig.suptitle("Representational Similarity Analysis: Entity Activations (Chat Model)\n"
                 "Cosine-distance RDM correlated with human mind perception RDM",
                 fontsize=13, y=1.06)
    fig.tight_layout()

    save_fig(fig, CHAT_FIG_DIR, "fig7_rsa_layerwise")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 8: Summary — all human correlations across methods
# ═════════════════════════════════════════════════════════════════════
def fig8_correlation_summary():
    """Summary bar chart of all model-human Spearman correlations."""
    print("Fig 8: Correlation summary across methods")

    results = []

    # Pairwise with_self
    pca = np.load(os.path.join(PAIR_DATA, "with_self", "pca_results.npz"),
                  allow_pickle=True)
    ek = list(pca["entity_keys"])
    scores = pca["factor_scores_01"]
    h_exp, h_age = load_human_scores(ek)
    for fi in range(2):
        for hi, (h_arr, h_name) in enumerate([(h_exp, "Experience"), (h_age, "Agency")]):
            rho, p = stats.spearmanr(scores[:, fi], h_arr)
            results.append({
                "method": f"Pairwise F{fi+1}",
                "human_dim": h_name,
                "rho": rho, "p": p, "n": len(ek),
            })

    # Individual with_self
    pca = np.load(os.path.join(INDIV_DATA, "with_self", "pca_results.npz"),
                  allow_pickle=True)
    ek = list(pca["entity_keys"])
    scores = pca["factor_scores_01"]
    h_exp, h_age = load_human_scores(ek)
    for fi in range(2):
        for hi, (h_arr, h_name) in enumerate([(h_exp, "Experience"), (h_age, "Agency")]):
            rho, p = stats.spearmanr(scores[:, fi], h_arr)
            results.append({
                "method": f"Individual F{fi+1}",
                "human_dim": h_name,
                "rho": rho, "p": p, "n": len(ek),
            })

    # RSA (peak layer, with_self) — skip NaN layers
    rsa_path = os.path.join(CHAT_RSA, "with_self", "rsa_all_layers.json")
    with open(rsa_path) as f:
        rsa = json.load(f)
    valid_rsa = [r for r in rsa if not np.isnan(r["rho"])]
    peak = max(valid_rsa, key=lambda r: r["rho"])
    results.append({
        "method": f"RSA (layer {peak['layer']})",
        "human_dim": "Mind RDM",
        "rho": peak["rho"], "p": peak["p_value"], "n": 13,
    })

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = [f"{r['method']}\nvs {r['human_dim']}" for r in results]
    rhos = [r["rho"] for r in results]
    colors = []
    for r in results:
        if r["p"] < 0.05:
            colors.append(C_MOD)
        else:
            colors.append("#cccccc")

    y = np.arange(len(results))
    bars = ax.barh(y, rhos, color=colors, edgecolor="white", height=0.6)

    # Significance annotations
    for i, r in enumerate(results):
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else \
              "*" if r["p"] < 0.05 else "n.s."
        x_pos = r["rho"] + 0.02 if r["rho"] >= 0 else r["rho"] - 0.02
        ha = "left" if r["rho"] >= 0 else "right"
        ax.text(x_pos, i, f"rho={r['rho']:.3f} {sig}", va="center", ha=ha,
                fontsize=9, fontweight="bold" if r["p"] < 0.05 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Model-Human Alignment Summary\n"
                 "Green = p < .05, Gray = n.s.")
    ax.set_xlim(-0.5, 1.0)
    ax.invert_yaxis()

    fig.tight_layout()
    save_fig(fig, os.path.join(BASE_DIR, "results", "figures"), "fig8_correlation_summary")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 9: Pairwise rating heatmap (character means)
# ═════════════════════════════════════════════════════════════════════
def fig9_pairwise_heatmap():
    """Heatmap of pairwise-derived character means (capacities x entities)."""
    print("Fig 9: Pairwise character means heatmap")

    d = np.load(os.path.join(PAIR_DATA, "with_self", "character_means.npz"),
                allow_pickle=True)
    means = d["means"]              # (18, 13)
    entity_keys = list(d["entity_keys"])
    cap_keys = list(d["capacity_keys"])

    exp_caps = [c for c, (_, f) in CAPACITY_PROMPTS.items() if f == "E"]

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(means, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(len(entity_keys)))
    ax.set_xticklabels([nice_entity(e) for e in entity_keys], rotation=45,
                       ha="right", fontsize=9)
    ax.set_yticks(range(len(cap_keys)))
    ylabels = [f"{nice_capacity(c)} ({'E' if c in exp_caps else 'A'})"
               for c in cap_keys]
    ax.set_yticklabels(ylabels, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8,
                        label="Mean Relative Rating (pairwise)")
    ax.set_title("Pairwise Character Means: LLaMA-2-13B (Base)\n"
                 "Mean E[R] across all pairwise comparisons per capacity")

    fig.tight_layout()
    save_fig(fig, PAIR_FIG_DIR, "fig9_pairwise_heatmap")


# ═════════════════════════════════════════════════════════════════════
# FIGURE 10: Model RDM vs Human RDM (chat model, peak layer)
# ═════════════════════════════════════════════════════════════════════
def fig10_rdm_comparison():
    """Side-by-side RDM matrices: model (peak RSA layer) vs human."""
    print("Fig 10: RDM comparison (model vs human)")

    # Find peak RSA layer (skip NaN)
    rsa_path = os.path.join(CHAT_RSA, "with_self", "rsa_all_layers.json")
    with open(rsa_path) as f:
        rsa = json.load(f)
    valid_rsa = [r for r in rsa if not np.isnan(r["rho"])]
    peak = max(valid_rsa, key=lambda r: r["rho"])
    peak_layer = peak["layer"]

    # Load RDMs
    rdm_data = np.load(os.path.join(CHAT_ACTIV, "with_self",
                                     "rdm_cosine_per_layer.npz"),
                       allow_pickle=True)
    model_rdm = rdm_data["model_rdm"]  # (41, 13, 13)
    human_rdm = rdm_data["human_rdm"]  # (13, 13)
    entity_keys = list(rdm_data["entity_keys"])
    nice_labels = [nice_entity(e) for e in entity_keys]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Human RDM
    ax = axes[0]
    im0 = ax.imshow(human_rdm, cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(entity_keys)))
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(entity_keys)))
    ax.set_yticklabels(nice_labels, fontsize=8)
    ax.set_title("Human RDM\n(Gray et al. factor distance)")
    fig.colorbar(im0, ax=ax, shrink=0.7, label="Dissimilarity")

    # Panel B: Model RDM (peak layer)
    ax = axes[1]
    im1 = ax.imshow(model_rdm[peak_layer], cmap="viridis", aspect="equal")
    ax.set_xticks(range(len(entity_keys)))
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(entity_keys)))
    ax.set_yticklabels(nice_labels, fontsize=8)
    ax.set_title(f"Model RDM (Layer {peak_layer})\n"
                 f"Cosine distance, rho={peak['rho']:.3f}, p={peak['p_value']:.4f}")
    fig.colorbar(im1, ax=ax, shrink=0.7, label="Cosine Distance")

    fig.suptitle("Representational Dissimilarity Matrices: 13 Entities",
                 fontsize=14, y=1.02)
    fig.tight_layout()

    save_fig(fig, CHAT_FIG_DIR, "fig10_rdm_comparison")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Experiment 4 Figures")
    print("=" * 60)
    print()

    fig1_scree_plot()
    fig2_loading_comparison()
    fig3_entity_scatter()
    fig4_mind_space()
    fig5_rating_heatmap()
    fig6_individual_scatter()
    fig7_rsa_layerwise()
    fig8_correlation_summary()
    fig9_pairwise_heatmap()
    fig10_rdm_comparison()

    print()
    print("=" * 60)
    print("All figures generated successfully.")
    print(f"  Pairwise figures:    {PAIR_FIG_DIR}")
    print(f"  Individual figures:  {INDIV_FIG_DIR}")
    print(f"  Chat/RSA figures:    {CHAT_FIG_DIR}")
    print(f"  Summary figures:     {os.path.join(BASE_DIR, 'results', 'figures')}")
    print("=" * 60)
