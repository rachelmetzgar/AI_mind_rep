#!/usr/bin/env python3
"""
Publication-quality figures for standalone concept activation alignment.

Reads the standalone_alignment_stats.json produced by 3a_standalone_stats.py
and generates all publication and presentation figures.

Key difference from contrast figures (2e):
  - No human/AI labels → no "human minus AI" semantics
  - Y-axis: "Mean Projection onto Probe Direction" (positive = human side)
  - Bootstrap significance (not permutation)
  - New figures: entity comparison, sysprompt variants

Outputs to: results/standalone_alignment/figures/

Usage:
    python 3b_standalone_figures.py

Env: llama2_env (needs numpy, matplotlib; no GPU)
Rachel C. Metzgar, Feb 2026
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ========================== STYLE ========================== #

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ========================== CONFIG ========================== #

BASE = os.path.dirname(os.path.abspath(__file__))
STATS_JSON = os.path.join(BASE, "results", "standalone_alignment",
                          "summaries", "standalone_alignment_stats.json")
FIG_ROOT = os.path.join(BASE, "results", "standalone_alignment", "figures")

HIDDEN_DIM = 5120
RESTRICTED_LAYER_START = 6

CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 18],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "Entity":    [16, 17],
    "SysPrompt": [20, 21, 22, 23],
}

DIM_LABELS = {
    1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social cognition",
    8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpfulness",
    14: "Biological", 15: "Shapes\n(negative ctrl)",
    16: "Human\n(entity)", 17: "AI\n(entity)",
    18: "Attention",
    20: "SysPrompt:\ntalk-to human", 21: "SysPrompt:\ntalk-to AI",
    22: "SysPrompt:\nbare human", 23: "SysPrompt:\nbare AI",
}

DIM_LABELS_SHORT = {
    1: "Phenom.", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social", 8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpful.",
    14: "Biological", 15: "Shapes",
    16: "Human (ent)", 17: "AI (ent)",
    18: "Attention",
    20: "SP:talk-H", 21: "SP:talk-AI",
    22: "SP:bare-H", 23: "SP:bare-AI",
}

CAT_PALETTE = {
    "Mental":    "#3274A1",
    "Physical":  "#E1812C",
    "Pragmatic": "#3A923A",
    "SysPrompt": "#845B53",
    "Bio Ctrl":  "#D4A03A",
    "Shapes":    "#E377C2",
    "Entity":    "#C03D3E",
}

CAT_ORDER = ["Mental", "Entity", "SysPrompt", "Physical", "Pragmatic",
             "Bio Ctrl", "Shapes"]

PROBE_TYPES = ["control_probe", "reading_probe"]
LAYER_RANGES = ["all_layers", "layers_6plus"]


def dim_cat(d):
    for cat, ids in CATEGORIES.items():
        if d in ids:
            return cat
    return "Other"


def dim_color(d):
    return CAT_PALETTE.get(dim_cat(d), "#666666")


def lr_label(layer_range):
    return "All Layers" if layer_range == "all_layers" else "Layers 6+"


def pt_label(probe_type):
    return probe_type.replace("_probe", "").title()


def pt_short(probe_type):
    return probe_type.replace("_probe", "")


def savefig(fig, path):
    """Save as both PNG and PDF."""
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# ========================== DATA LOADING ========================== #

def load_stats():
    """Load the master JSON from 3a_standalone_stats.py."""
    with open(STATS_JSON) as f:
        return json.load(f)


def get_dim_ids(stats):
    """Get sorted list of integer dim IDs."""
    return sorted(int(k) for k in stats["dimensions"].keys())


# ========================== FIGURE: RANKED BARS ========================== #

def fig_ranked_bars(stats, probe_type, layer_range, out_dir, xlim=None):
    """
    Horizontal bars, all dims ranked by projection, with CIs and significance stars.
    """
    key = f"{probe_type}_{layer_range}"
    dims = stats["dimensions"]
    dim_data = []
    for did_str, d in dims.items():
        did = int(did_str)
        r = d.get(key)
        if r is None:
            continue
        dim_data.append((did, r["observed_projection"],
                         r["ci_lo"], r["ci_hi"],
                         r.get("p_adjusted", r["p_value"]),
                         r.get("sig_fdr", r["sig"])))

    dim_data.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = np.arange(len(dim_data))
    names = [DIM_LABELS.get(d[0], str(d[0])) for d in dim_data]
    means = [d[1] for d in dim_data]
    errs_lo = [d[1] - d[2] for d in dim_data]
    errs_hi = [d[3] - d[1] for d in dim_data]
    colors = [dim_color(d[0]) for d in dim_data]

    ax.barh(y_pos, means, xerr=[errs_lo, errs_hi],
            color=colors, edgecolor="white", linewidth=0.5,
            capsize=2, error_kw={"lw": 0.8})
    ax.axvline(0, color="black", lw=0.4)

    # Significance stars
    for i, (did, m, lo, hi, p, sig) in enumerate(dim_data):
        if sig != "n.s.":
            x_pos = max(hi, m) + 0.002
            ax.text(x_pos, i, sig, va="center", fontsize=7,
                    color="#333333", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel("Mean Projection onto Probe Direction\n"
                  "(positive = human side; 95% bootstrap CI)")
    ax.set_title(f"Standalone Activation ↔ {pt_label(probe_type)} Probe "
                 f"({lr_label(layer_range)})")

    cat_handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c) for c in CAT_ORDER]
    ax.legend(handles=cat_handles, loc="lower right", fontsize=7, ncol=2,
              framealpha=0.9, edgecolor="none")

    if xlim is not None:
        ax.set_xlim(xlim)

    fname = f"fig_ranked_bars_{layer_range}.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: LAYER PROFILES (summary) ========================== #

def fig_layer_profiles(stats, probe_type, out_dir, ylim=None):
    """Layer-by-layer projection for key dims + mental avg band."""
    dims = stats["dimensions"]
    fig, ax = plt.subplots(figsize=(10, 5))

    key = f"{probe_type}_per_layer_boot"

    # Compute mental average
    mental_stacks = []
    for d in CATEGORIES["Mental"]:
        layer_data = dims.get(str(d), {}).get(key, [])
        if layer_data:
            mental_stacks.append([r["observed_projection"] for r in layer_data])

    if mental_stacks:
        mental_avg = np.mean(mental_stacks, axis=0)
        mental_sem = np.std(mental_stacks, axis=0, ddof=1) / np.sqrt(len(mental_stacks))
        layers = list(range(len(mental_avg)))
        ax.plot(layers, mental_avg, color=CAT_PALETTE["Mental"], lw=2.5,
                label="Mental (avg)", zorder=5)
        ax.fill_between(layers, mental_avg - mental_sem, mental_avg + mental_sem,
                        color=CAT_PALETTE["Mental"], alpha=0.15, zorder=4)

    # Key individual dims
    highlight = {
        16: {"color": CAT_PALETTE["Entity"], "lw": 1.8, "ls": "-"},
        17: {"color": CAT_PALETTE["Entity"], "lw": 1.8, "ls": "--"},
        20: {"color": CAT_PALETTE["SysPrompt"], "lw": 1.5, "ls": "--"},
        15: {"color": CAT_PALETTE["Shapes"], "lw": 2.0, "ls": ":"},
    }
    for dim_id, style in highlight.items():
        layer_data = dims.get(str(dim_id), {}).get(key, [])
        if layer_data:
            projs = [r["observed_projection"] for r in layer_data]
            ax.plot(range(len(projs)), projs,
                    color=style["color"], lw=style["lw"], ls=style["ls"],
                    label=DIM_LABELS_SHORT.get(dim_id, str(dim_id)), zorder=3)

    # Background: all other dims as thin gray
    highlight_ids = set(CATEGORIES["Mental"]) | set(highlight.keys())
    for did_str, d in dims.items():
        did = int(did_str)
        if did in highlight_ids:
            continue
        layer_data = d.get(key, [])
        if layer_data:
            projs = [r["observed_projection"] for r in layer_data]
            ax.plot(range(len(projs)), projs, color="#CCCCCC", lw=0.5,
                    alpha=0.5, zorder=1)

    ax.axhline(0, color="black", lw=0.4, zorder=0)
    ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6, zorder=2)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Projection")
    ax.set_title(f"Layer-by-Layer Standalone ↔ {pt_label(probe_type)} Probe Alignment")
    ax.set_xlim(0, 40)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.text(RESTRICTED_LAYER_START + 0.3, ax.get_ylim()[1] * 0.95, "L6+",
            fontsize=7, color="gray", va="top")
    ax.legend(loc="best", framealpha=0.9, edgecolor="none", fontsize=7.5)

    fname = "fig_layer_profiles.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: LAYER PROFILES GRID ========================== #

def fig_layer_profiles_grid(stats, probe_type, out_dir, ylim=None):
    """Small-multiples grid: one subplot per dimension, showing layer-by-layer
    projection with the probe. Dimensions ordered by category."""
    dims = stats["dimensions"]
    key = f"{probe_type}_per_layer_boot"

    ordered_dims = []
    for cat in CAT_ORDER:
        for d in sorted(CATEGORIES[cat]):
            ordered_dims.append(d)

    n = len(ordered_dims)
    ncols = 6
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, dim_id in enumerate(ordered_dims):
        ax = axes_flat[idx]
        layer_data = dims.get(str(dim_id), {}).get(key, [])
        if not layer_data:
            ax.set_visible(False)
            continue

        projs = [r["observed_projection"] for r in layer_data]
        color = dim_color(dim_id)
        ax.plot(range(len(projs)), projs, color=color, lw=1.5)
        ax.axhline(0, color="black", lw=0.3)
        ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=0.6, alpha=0.5)

        # Significance overlay: shade layers where bootstrap p < .05
        for r in layer_data:
            if r["p_value"] < 0.05:
                ax.axvspan(r["layer"] - 0.5, r["layer"] + 0.5,
                           color=color, alpha=0.08, zorder=0)

        cat = dim_cat(dim_id)
        label = DIM_LABELS_SHORT.get(dim_id, str(dim_id))
        ax.set_title(f"{label} [{cat}]", fontsize=8, color=color, fontweight="bold")
        ax.set_xlim(0, 40)
        ax.tick_params(labelsize=7)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if ylim is not None:
        axes_flat[0].set_ylim(ylim)

    fig.supxlabel("Transformer Layer", fontsize=11)
    fig.supylabel("Mean Projection", fontsize=11)
    fig.suptitle(f"Per-Dimension Layer Profiles: {pt_label(probe_type)} Probe\n"
                 f"(shaded layers = bootstrap p < .05)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

    fname = "fig_layer_profiles_grid.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: HEATMAP ========================== #

def fig_heatmap(stats, probe_type, out_dir, vmax=None):
    """Layer x dim heatmap of per-layer projections with bootstrap significance dots."""
    dims = stats["dimensions"]

    key_boot = f"{probe_type}_per_layer_boot"

    sorted_dims = []
    for cat in CAT_ORDER:
        cat_dims = []
        for d in CATEGORIES[cat]:
            layer_data = dims.get(str(d), {}).get(key_boot, [])
            m = np.mean([r["observed_projection"] for r in layer_data]) if layer_data else 0
            cat_dims.append((d, m))
        cat_dims.sort(key=lambda x: -x[1])
        sorted_dims.extend([d for d, _ in cat_dims])

    matrix = []
    col_labels = []
    sig_mask = []
    for d in sorted_dims:
        layer_data = dims.get(str(d), {}).get(key_boot, [])
        if not layer_data:
            continue
        matrix.append([r["observed_projection"] for r in layer_data])
        col_labels.append(DIM_LABELS_SHORT.get(d, str(d)))
        sig_mask.append([r["p_value"] < 0.05 for r in layer_data])

    if not matrix:
        return

    matrix = np.array(matrix).T
    sig_mask = np.array(sig_mask).T

    fig, ax = plt.subplots(figsize=(10, 7))
    if vmax is None:
        vmax = np.percentile(np.abs(matrix), 98)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    for layer in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if sig_mask[layer, col]:
                ax.plot(col, layer, 'k.', markersize=2, alpha=0.6)

    ax.set_xlabel("Concept Dimension")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(f"Projection Alignment ↔ {pt_label(probe_type)} Probe "
                 f"(dots = bootstrap p < .05)")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(0, 41, 5))

    # Category separators
    pos = 0
    for cat in CAT_ORDER:
        n_cat = len([d for d in CATEGORIES[cat]
                     if str(d) in dims and dims[str(d)].get(key_boot)])
        if n_cat > 0 and pos > 0:
            ax.axvline(pos - 0.5, color="white", lw=1.5)
        pos += n_cat

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean Projection", fontsize=9)

    fname = f"fig_heatmap_{pt_short(probe_type)}.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: LAYERWISE SIGNIFICANCE ========================== #

def fig_layerwise_significance(stats, out_dir):
    """Number of significant dimensions at each layer."""
    dims = stats["dimensions"]
    n_layers = stats["meta"]["n_layers"]

    fig, ax = plt.subplots(figsize=(10, 4))

    for probe, color, label in [
        ("control_probe", "#3274A1", "Control probe"),
        ("reading_probe", "#C03D3E", "Reading probe"),
    ]:
        key = f"{probe}_per_layer_boot"
        counts = np.zeros(n_layers)
        for did_str, d in dims.items():
            boot = d.get(key, [])
            if boot:
                for r in boot:
                    if r["p_value"] < 0.05:
                        counts[r["layer"]] += 1

        ax.plot(range(n_layers), counts, color=color, lw=2, label=label)
        ax.fill_between(range(n_layers), counts, alpha=0.15, color=color)

    total_dims = len([d for d in dims.values()
                      if "control_probe_per_layer_boot" in d])
    ax.axhline(total_dims, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.5, total_dims + 0.3, f"n = {total_dims} dims", fontsize=7, color="gray")
    ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("# Significant Dimensions (bootstrap p < .05)")
    ax.set_title("Layerwise Significance Count (Standalone)")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)

    savefig(fig, os.path.join(out_dir, "fig_layerwise_significance.png"))


# ========================== FIGURE: CATEGORY BARS ========================== #

def fig_category_bars(stats, layer_range, out_dir):
    """Grouped bars, control vs reading, with strip plot overlay."""
    cat_data = stats["categories"]
    dim_data = stats["dimensions"]

    fig, ax = plt.subplots(figsize=(10, 5))

    cat_labels = ["Mental\n(1-7,18)", "Entity\n(16,17)", "SysPr.\n(20-23)",
                  "Physical\n(8-10)", "Pragm.\n(11-13)",
                  "Bio\n(14)", "Shapes\n(15)"]
    x = np.arange(len(CAT_ORDER))
    width = 0.35

    for offset, probe, color, label in [
        (-width/2, "control_probe", "#3274A1", "Control probe (generation)"),
        (width/2,  "reading_probe", "#C03D3E", "Reading probe (reflective)"),
    ]:
        key = f"{probe}_{layer_range}"
        cat_means = []
        cat_lo = []
        cat_hi = []
        cat_individual = []
        for cn in CAT_ORDER:
            cr = cat_data.get(key, {}).get(cn)
            if cr:
                cat_means.append(cr["mean"])
                cat_lo.append(cr["mean"] - cr["ci_lo"])
                cat_hi.append(cr["ci_hi"] - cr["mean"])
            else:
                cat_means.append(0)
                cat_lo.append(0)
                cat_hi.append(0)

            dim_vals = []
            for d in CATEGORIES[cn]:
                dr = dim_data.get(str(d), {}).get(key)
                if dr:
                    dim_vals.append(dr["observed_projection"])
            cat_individual.append(dim_vals)

        ax.bar(x + offset, cat_means, width, yerr=[cat_lo, cat_hi],
               color=color, alpha=0.75, capsize=3,
               error_kw={"lw": 0.8}, label=label, edgecolor="white", linewidth=0.5)

        rng = np.random.default_rng(42)
        for i, vals in enumerate(cat_individual):
            if len(vals) > 1:
                jitter = rng.uniform(-width * 0.3, width * 0.3, len(vals))
                ax.scatter(x[i] + offset + jitter, vals, s=15, color="white",
                           edgecolors=color, linewidth=0.6, zorder=5, alpha=0.8)

    ax.axhline(0, color="black", lw=0.4)

    ax.set_ylabel("Mean Projection\n(positive = human side of probe)")
    ax.set_title(f"Standalone Alignment by Category ({lr_label(layer_range)})")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.legend(fontsize=8, loc="best", framealpha=0.9, edgecolor="none")

    savefig(fig, os.path.join(out_dir, f"fig_category_bars_{layer_range}.png"))


# ========================== FIGURE: CONTROL vs READING SCATTER ========================== #

def fig_ctrl_vs_read_scatter(stats, layer_range, out_dir):
    """Control (x) vs reading (y) per dim."""
    dims = stats["dimensions"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for did_str, d in dims.items():
        did = int(did_str)
        ctrl = d.get(f"control_probe_{layer_range}", {}).get("observed_projection")
        read = d.get(f"reading_probe_{layer_range}", {}).get("observed_projection")
        if ctrl is None or read is None:
            continue

        c = dim_color(did)
        ax.scatter(ctrl, read, s=90, color=c, edgecolors="white",
                   linewidth=0.8, zorder=5)

        label = DIM_LABELS_SHORT.get(did, str(did))
        ax.annotate(label, (ctrl, read), fontsize=6.5, ha="left", va="bottom",
                    xytext=(5, 4), textcoords="offset points", color="#333333")

    ax.axhline(0, color="black", lw=0.3, zorder=0)
    ax.axvline(0, color="black", lw=0.3, zorder=0)

    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color="gray", ls=":", lw=0.6, alpha=0.4, zorder=0)

    ax.set_xlabel("Alignment with Control Probe\n(in-context partner rep. during generation)")
    ax.set_ylabel("Alignment with Reading Probe\n(reflective partner rep.)")
    ax.set_title(f"Standalone Alignment:\nControl vs Reading Probe ({lr_label(layer_range)})")

    handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c) for c in CAT_ORDER]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.9, edgecolor="none")

    savefig(fig, os.path.join(out_dir, f"fig_ctrl_vs_read_scatter_{layer_range}.png"))


# ========================== FIGURE: PAIRWISE MATRIX ========================== #

def fig_pairwise_matrix(stats, probe_type, layer_range, out_dir):
    """Lower-triangle heatmap of pairwise dimension p-values (FDR-adjusted)."""
    key = f"{probe_type}_{layer_range}"
    pairs = stats["pairwise_dimensions"].get(key, [])
    if not pairs:
        return

    dim_ids = sorted(set(r["dim_a"] for r in pairs) | set(r["dim_b"] for r in pairs))
    n = len(dim_ids)
    id_to_idx = {d: i for i, d in enumerate(dim_ids)}

    mat = np.full((n, n), np.nan)
    for r in pairs:
        i, j = id_to_idx[r["dim_a"]], id_to_idx[r["dim_b"]]
        p = max(r["p_adjusted"], 1e-10)
        val = -np.log10(p)
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(11, 10))

    mask = np.triu(np.ones_like(mat, dtype=bool))
    masked = np.ma.array(mat, mask=mask)

    im = ax.imshow(masked, cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=4)

    labels = [DIM_LABELS_SHORT.get(d, str(d)) for d in dim_ids]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=6)

    for r in pairs:
        i, j = id_to_idx[r["dim_a"]], id_to_idx[r["dim_b"]]
        if r["p_adjusted"] < 0.05:
            ax.text(i, j, "*", ha="center", va="center", fontsize=5,
                    fontweight="bold",
                    color="white" if r["p_adjusted"] < 0.001 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("-log10(p_adjusted)", fontsize=9)

    ax.set_title(f"Pairwise Dimension Comparisons: {pt_label(probe_type)} Probe "
                 f"({lr_label(layer_range)})\n(FDR-corrected, * = p < .05)")

    savefig(fig, os.path.join(out_dir,
            f"fig_pairwise_matrix_{pt_short(probe_type)}_{layer_range}.png"))


# ========================== FIGURE: CATEGORY PAIRWISE ========================== #

def fig_category_pairwise(stats, probe_type, layer_range, out_dir):
    """Category comparison forest plot."""
    key = f"{probe_type}_{layer_range}"
    pairs = stats["pairwise_categories"].get(key, [])
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    pairs_sorted = sorted(pairs, key=lambda r: r["diff"], reverse=True)

    y_pos = np.arange(len(pairs_sorted))
    labels = [f"{r['cat_a']} vs {r['cat_b']}" for r in pairs_sorted]

    diffs = [r["diff"] for r in pairs_sorted]
    ci_lo = [r["diff"] - r["ci_lo"] for r in pairs_sorted]
    ci_hi = [r["ci_hi"] - r["diff"] for r in pairs_sorted]

    colors = ["#C03D3E" if r["p_adjusted"] < 0.05 else "#999999"
              for r in pairs_sorted]

    ax.barh(y_pos, diffs, xerr=[ci_lo, ci_hi], color=colors,
            edgecolor="white", linewidth=0.5, capsize=2, error_kw={"lw": 0.8},
            height=0.7)
    ax.axvline(0, color="black", lw=0.5)

    for i, r in enumerate(pairs_sorted):
        sig = r.get("sig_fdr", "n.s.")
        if sig != "n.s.":
            x_pos = r["ci_hi"] + 0.001
            ax.text(x_pos, i, sig, va="center", fontsize=7,
                    fontweight="bold", color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Projection Difference (95% bootstrap CI)")
    ax.set_title(f"Pairwise Category Comparisons: {pt_label(probe_type)} Probe "
                 f"({lr_label(layer_range)})\nRed = significant after FDR correction")

    savefig(fig, os.path.join(out_dir,
            f"fig_category_pairwise_{pt_short(probe_type)}_{layer_range}.png"))


# ========================== FIGURE: ENTITY COMPARISON (NEW) ========================== #

def fig_entity_comparison(stats, out_dir):
    """
    Side-by-side comparison of human entity (dim 16) vs AI entity (dim 17)
    for both probes. Sanity check: expect opposite signs.
    """
    dims = stats["dimensions"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    for ax, lr, lr_display in [
        (axes[0], "all_layers", "All Layers"),
        (axes[1], "layers_6plus", "Layers 6+"),
    ]:
        bar_data = []
        bar_labels = []
        bar_colors = []
        bar_errs_lo = []
        bar_errs_hi = []
        bar_sigs = []

        for probe, probe_label, color in [
            ("control_probe", "Control", "#3274A1"),
            ("reading_probe", "Reading", "#C03D3E"),
        ]:
            key = f"{probe}_{lr}"
            for dim_id, dim_label in [(16, "Human\n(entity)"), (17, "AI\n(entity)")]:
                r = dims.get(str(dim_id), {}).get(key)
                if r is None:
                    continue
                bar_data.append(r["observed_projection"])
                bar_errs_lo.append(r["observed_projection"] - r["ci_lo"])
                bar_errs_hi.append(r["ci_hi"] - r["observed_projection"])
                bar_labels.append(f"{dim_label}\n{probe_label}")
                bar_colors.append(color)
                bar_sigs.append(r.get("sig_fdr", r["sig"]))

        x = np.arange(len(bar_data))
        ax.bar(x, bar_data, yerr=[bar_errs_lo, bar_errs_hi],
               color=bar_colors, alpha=0.75, capsize=3,
               error_kw={"lw": 0.8}, edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="black", lw=0.4)

        for i, sig in enumerate(bar_sigs):
            if sig != "n.s.":
                y_pos = bar_data[i] + bar_errs_hi[i] + 0.5
                ax.text(i, y_pos, sig, ha="center", fontsize=8,
                        fontweight="bold", color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=7)
        ax.set_title(lr_display)
        ax.set_ylabel("Mean Projection\n(positive = human side)")

    fig.suptitle("Entity Concept Comparison: Human vs AI\n"
                 "(expect: 'Human' → positive, 'AI' → negative)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    savefig(fig, os.path.join(out_dir, "fig_entity_comparison.png"))


# ========================== FIGURE: SYSPROMPT VARIANTS (NEW) ========================== #

def fig_sysprompt_variants(stats, out_dir):
    """
    4 sysprompt variants (dims 20-23) as grouped bars for both probes.
    Tests whether "talk to human" → human side, "talk to AI" → AI side.
    """
    dims = stats["dimensions"]
    sp_dims = [20, 21, 22, 23]
    sp_labels = ["Talk-to\nHuman", "Talk-to\nAI", "Bare\nHuman", "Bare\nAI"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, lr, lr_display in [
        (axes[0], "all_layers", "All Layers"),
        (axes[1], "layers_6plus", "Layers 6+"),
    ]:
        x = np.arange(len(sp_dims))
        width = 0.35

        for offset, probe, color, plabel in [
            (-width/2, "control_probe", "#3274A1", "Control"),
            (width/2,  "reading_probe", "#C03D3E", "Reading"),
        ]:
            key = f"{probe}_{lr}"
            means = []
            err_lo = []
            err_hi = []
            sigs = []
            for dim_id in sp_dims:
                r = dims.get(str(dim_id), {}).get(key)
                if r:
                    means.append(r["observed_projection"])
                    err_lo.append(r["observed_projection"] - r["ci_lo"])
                    err_hi.append(r["ci_hi"] - r["observed_projection"])
                    sigs.append(r.get("sig_fdr", r["sig"]))
                else:
                    means.append(0)
                    err_lo.append(0)
                    err_hi.append(0)
                    sigs.append("n.s.")

            ax.bar(x + offset, means, width, yerr=[err_lo, err_hi],
                   color=color, alpha=0.75, capsize=3,
                   error_kw={"lw": 0.8}, label=plabel,
                   edgecolor="white", linewidth=0.5)

            for i, sig in enumerate(sigs):
                if sig != "n.s.":
                    y = means[i] + err_hi[i] + 0.3
                    ax.text(x[i] + offset, y, sig, ha="center", fontsize=7,
                            fontweight="bold", color="#333")

        ax.axhline(0, color="black", lw=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(sp_labels, fontsize=8)
        ax.set_title(lr_display)
        ax.set_ylabel("Mean Projection\n(positive = human side)")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("System Prompt Variants: Standalone Alignment\n"
                 "(expect: human variants → positive, AI variants → negative)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    savefig(fig, os.path.join(out_dir, "fig_sysprompt_variants.png"))


# ========================== FIGURE: MAIN RESULT (3-PANEL) ========================== #

def fig_main_result(stats, out_dir):
    """
    Composite 3-panel for paper:
      A: Layer profiles (control probe)
      B: Layer profiles (reading probe)
      C: Ranked bars (control probe, all layers)
    """
    dims = stats["dimensions"]

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.5], hspace=0.35, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ── Panels A/B: Layer profiles ──
    for ax, probe, title_suffix in [
        (ax_a, "control_probe", "Control Probe"),
        (ax_b, "reading_probe", "Reading Probe"),
    ]:
        key = f"{probe}_per_layer_boot"

        mental_stacks = []
        for d in CATEGORIES["Mental"]:
            layer_data = dims.get(str(d), {}).get(key, [])
            if layer_data:
                mental_stacks.append([r["observed_projection"] for r in layer_data])

        if mental_stacks:
            mental_avg = np.mean(mental_stacks, axis=0)
            mental_sem = np.std(mental_stacks, axis=0, ddof=1) / np.sqrt(len(mental_stacks))
            layers = list(range(len(mental_avg)))
            ax.plot(layers, mental_avg, color=CAT_PALETTE["Mental"], lw=2.5,
                    label="Mental (avg)", zorder=5)
            ax.fill_between(layers, mental_avg - mental_sem, mental_avg + mental_sem,
                            color=CAT_PALETTE["Mental"], alpha=0.15, zorder=4)

        highlight_dims = {
            16: {"color": CAT_PALETTE["Entity"], "lw": 1.8, "ls": "-"},
            17: {"color": CAT_PALETTE["Entity"], "lw": 1.8, "ls": "--"},
            15: {"color": CAT_PALETTE["Shapes"], "lw": 2.0, "ls": ":"},
        }
        for dim_id, style in highlight_dims.items():
            layer_data = dims.get(str(dim_id), {}).get(key, [])
            if layer_data:
                projs = [r["observed_projection"] for r in layer_data]
                ax.plot(range(len(projs)), projs,
                        color=style["color"], lw=style["lw"], ls=style["ls"],
                        label=DIM_LABELS_SHORT.get(dim_id, str(dim_id)), zorder=3)

        highlight_ids = set(CATEGORIES["Mental"]) | set(highlight_dims.keys())
        for did_str, d in dims.items():
            did = int(did_str)
            if did in highlight_ids:
                continue
            layer_data = d.get(key, [])
            if layer_data:
                projs = [r["observed_projection"] for r in layer_data]
                ax.plot(range(len(projs)), projs, color="#CCCCCC", lw=0.5,
                        alpha=0.5, zorder=1)

        ax.axhline(0, color="black", lw=0.4, zorder=0)
        ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6, zorder=2)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Mean Projection")
        ax.set_title(f"Standalone ↔ Exp 2 {title_suffix}")
        ax.set_xlim(0, 40)
        ax.legend(loc="best", framealpha=0.9, edgecolor="none", fontsize=7.5)

    # Equalize y-axis
    yl_a = ax_a.get_ylim()
    yl_b = ax_b.get_ylim()
    shared_yl = (min(yl_a[0], yl_b[0]), max(yl_a[1], yl_b[1]))
    ax_a.set_ylim(shared_yl)
    ax_b.set_ylim(shared_yl)

    # ── Panel C: Ranked bars (control, all layers) ──
    key = "control_probe_all_layers"
    dim_data = []
    for did_str, d in dims.items():
        did = int(did_str)
        r = d.get(key)
        if r is None:
            continue
        dim_data.append((did, r["observed_projection"],
                         r["ci_lo"], r["ci_hi"],
                         r.get("p_adjusted", r["p_value"]),
                         r.get("sig_fdr", r["sig"])))
    dim_data.sort(key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(dim_data))
    names = [DIM_LABELS.get(d[0], str(d[0])) for d in dim_data]
    means = [d[1] for d in dim_data]
    errs_lo = [d[1] - d[2] for d in dim_data]
    errs_hi = [d[3] - d[1] for d in dim_data]
    colors = [dim_color(d[0]) for d in dim_data]

    ax_c.barh(y_pos, means, xerr=[errs_lo, errs_hi],
              color=colors, edgecolor="white", linewidth=0.5,
              capsize=2, error_kw={"lw": 0.8})
    ax_c.axvline(0, color="black", lw=0.4)

    for i, (did, m, lo, hi, p, sig) in enumerate(dim_data):
        if sig != "n.s.":
            ax_c.text(max(hi, m) + 0.002, i, sig, va="center", fontsize=7,
                      color="#333333", fontweight="bold")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(names, fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Mean Projection onto Exp 2 Control Probe Direction\n"
                    "(positive = human side; 95% bootstrap CI)")
    ax_c.set_title("Standalone Concept Activation Alignment with Partner-Identity Probe")

    cat_handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c) for c in CAT_ORDER]
    ax_c.legend(handles=cat_handles, loc="lower right", fontsize=7, ncol=2,
                framealpha=0.9, edgecolor="none")

    ax_a.text(-0.12, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")
    ax_b.text(-0.12, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    ax_c.text(-0.05, 1.03, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold")

    savefig(fig, os.path.join(out_dir, "fig_main_result.png"))


# ========================== FIGURE: SUMMARY PANEL (2-PANEL) ========================== #

def fig_summary_panel(stats, out_dir):
    """
    Composite 2-panel for talks:
      A: Category-level bars
      B: Control vs reading scatter
    """
    dims = stats["dimensions"]
    cat_data = stats["categories"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})

    # ── Panel A: Category bars ──
    cat_labels = ["Mental\n(1-7,18)", "Entity\n(16,17)", "SysPr.\n(20-23)",
                  "Physical\n(8-10)", "Pragm.\n(11-13)",
                  "Bio\n(14)", "Shapes\n(15)"]
    x = np.arange(len(CAT_ORDER))
    width = 0.35

    for offset, probe, color, label in [
        (-width/2, "control_probe", "#3274A1", "Control probe (generation)"),
        (width/2,  "reading_probe", "#C03D3E", "Reading probe (reflective)"),
    ]:
        key = f"{probe}_all_layers"
        cat_means = []
        cat_lo = []
        cat_hi = []
        for cn in CAT_ORDER:
            cr = cat_data.get(key, {}).get(cn)
            if cr:
                cat_means.append(cr["mean"])
                cat_lo.append(cr["mean"] - cr["ci_lo"])
                cat_hi.append(cr["ci_hi"] - cr["mean"])
            else:
                cat_means.append(0)
                cat_lo.append(0)
                cat_hi.append(0)

        ax1.bar(x + offset, cat_means, width, yerr=[cat_lo, cat_hi],
                color=color, alpha=0.8, capsize=3,
                error_kw={"lw": 0.8}, label=label, edgecolor="white", linewidth=0.5)

    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_ylabel("Mean Projection")
    ax1.set_title("Alignment by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=8)
    ax1.legend(fontsize=7.5, loc="best", framealpha=0.9, edgecolor="none")

    # ── Panel B: Scatter ──
    for did_str, d in dims.items():
        did = int(did_str)
        ctrl = d.get("control_probe_all_layers", {}).get("observed_projection")
        read = d.get("reading_probe_all_layers", {}).get("observed_projection")
        if ctrl is None or read is None:
            continue
        c = dim_color(did)
        ax2.scatter(ctrl, read, s=70, color=c, edgecolors="white",
                    linewidth=0.7, zorder=5)
        ax2.annotate(DIM_LABELS_SHORT.get(did, str(did)), (ctrl, read),
                     fontsize=6.5, ha="left", va="bottom",
                     xytext=(4, 3), textcoords="offset points", color="#333333")

    ax2.axhline(0, color="black", lw=0.3)
    ax2.axvline(0, color="black", lw=0.3)
    ax2.set_xlabel("↔ Control Probe")
    ax2.set_ylabel("↔ Reading Probe")
    ax2.set_title("Control vs Reading Alignment")
    handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c) for c in CAT_ORDER]
    ax2.legend(handles=handles, loc="upper left", fontsize=6.5,
               framealpha=0.9, edgecolor="none")

    ax1.text(-0.08, 1.03, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")
    ax2.text(-0.08, 1.03, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    savefig(fig, os.path.join(out_dir, "fig_summary_panel.png"))


# ========================== MAIN ========================== #

def main():
    print("Loading standalone alignment stats...")
    stats = load_stats()
    dims = stats["dimensions"]
    n_dims = len(dims)
    print(f"  {n_dims} dimensions loaded\n")

    # Create output directories
    for subdir in ["control_probe", "reading_probe", "layerwise",
                   "comparisons", "standalone_specific"]:
        os.makedirs(os.path.join(FIG_ROOT, subdir), exist_ok=True)

    # ── Pre-compute shared axis limits for paired probe figures ──

    # Ranked bars: shared xlim per layer_range
    shared_xlims = {}
    for lr in LAYER_RANGES:
        all_vals = []
        for probe in PROBE_TYPES:
            key = f"{probe}_{lr}"
            for d in dims.values():
                r = d.get(key)
                if r:
                    all_vals.extend([r["ci_lo"], r["ci_hi"]])
        if all_vals:
            pad = max(abs(min(all_vals)), abs(max(all_vals))) * 0.1
            shared_xlims[lr] = (min(min(all_vals), 0) - pad,
                                max(max(all_vals), 0) + pad + 0.02)

    # Layer profiles: shared ylim across both probes
    all_projs = []
    for probe in PROBE_TYPES:
        key = f"{probe}_per_layer_boot"
        for d in dims.values():
            layer_data = d.get(key, [])
            if layer_data:
                all_projs.extend([r["observed_projection"] for r in layer_data])
    shared_ylim = None
    if all_projs:
        shared_ylim = (min(all_projs) - 0.5, max(all_projs) + 0.5)

    # Heatmaps: shared vmax across both probes
    all_heatmap_abs = []
    for probe in PROBE_TYPES:
        key = f"{probe}_per_layer_boot"
        for d in dims.values():
            layer_data = d.get(key, [])
            if layer_data:
                all_heatmap_abs.extend([abs(r["observed_projection"]) for r in layer_data])
    shared_vmax = np.percentile(all_heatmap_abs, 98) if all_heatmap_abs else None

    print("  Shared axis limits computed:")
    for lr, xl in shared_xlims.items():
        print(f"    ranked_bars {lr}: x = [{xl[0]:.3f}, {xl[1]:.3f}]")
    if shared_ylim:
        print(f"    layer_profiles: y = [{shared_ylim[0]:.3f}, {shared_ylim[1]:.3f}]")
    if shared_vmax:
        print(f"    heatmap: vmax = {shared_vmax:.3f}")

    # ── Per-probe figures ──
    for probe in PROBE_TYPES:
        probe_dir = os.path.join(FIG_ROOT, probe)
        print(f"\n{probe}:")
        for lr in LAYER_RANGES:
            fig_ranked_bars(stats, probe, lr, probe_dir, xlim=shared_xlims.get(lr))
        fig_layer_profiles(stats, probe, probe_dir, ylim=shared_ylim)
        fig_layer_profiles_grid(stats, probe, probe_dir, ylim=shared_ylim)

    # ── Layerwise figures ──
    lw_dir = os.path.join(FIG_ROOT, "layerwise")
    print("\nlayerwise:")
    for probe in PROBE_TYPES:
        fig_heatmap(stats, probe, lw_dir, vmax=shared_vmax)
    fig_layerwise_significance(stats, lw_dir)

    # ── Comparison figures ──
    comp_dir = os.path.join(FIG_ROOT, "comparisons")
    print("\ncomparisons:")
    for lr in LAYER_RANGES:
        fig_category_bars(stats, lr, comp_dir)
        fig_ctrl_vs_read_scatter(stats, lr, comp_dir)
    for probe in PROBE_TYPES:
        for lr in LAYER_RANGES:
            fig_pairwise_matrix(stats, probe, lr, comp_dir)
            fig_category_pairwise(stats, probe, lr, comp_dir)

    # ── Standalone-specific figures ──
    ss_dir = os.path.join(FIG_ROOT, "standalone_specific")
    print("\nstandalone-specific:")
    fig_entity_comparison(stats, ss_dir)
    fig_sysprompt_variants(stats, ss_dir)

    # ── Composite figures ──
    print("\ncomposite:")
    fig_main_result(stats, FIG_ROOT)
    fig_summary_panel(stats, FIG_ROOT)

    print(f"\nAll figures saved to: {FIG_ROOT}/")
    print("Done.")


if __name__ == "__main__":
    main()
