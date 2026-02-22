#!/usr/bin/env python3
"""
Publication-quality figures for concept-probe alignment analysis.

Reads the alignment_stats.json produced by 2d_concept_probe_stats.py
and generates all publication and presentation figures.

Every figure type is generated for the full question space:
  - Probe type:  control_probe, reading_probe
  - Layer range: all_layers, layers_6plus
  - Granularity: individual dimensions, categories

Outputs to: results/concept_probe_alignment/figures/

Usage:
    python 2e_concept_probe_figures.py

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
STATS_JSON = os.path.join(BASE, "results", "concept_probe_alignment",
                          "summaries", "alignment_stats.json")
FIG_ROOT = os.path.join(BASE, "results", "concept_probe_alignment", "figures")

HIDDEN_DIM = 5120
RESTRICTED_LAYER_START = 6

CATEGORIES = {
    "Mental":              [1, 2, 3, 4, 5, 6, 7, 17],
    "Physical":            [8, 9, 10],
    "Pragmatic":           [11, 12, 13],
    "Human vs AI (General)": [0],
    "Bio Ctrl":            [14],
    "Shapes":              [15],
    "SysPrompt":           [18],
}

DIM_LABELS = {
    0: "Human vs AI\n(General)",
    1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social cognition",
    8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpfulness",
    14: "Biological", 15: "Shapes\n(negative ctrl)",
    17: "Attention", 18: "System prompt",
}

DIM_LABELS_SHORT = {
    0: "H vs AI (Gen)", 1: "Phenom.", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social", 8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpful.",
    14: "Biological", 15: "Shapes", 17: "Attention",
    18: "SysPrompt",
}

CAT_PALETTE = {
    "Mental":              "#3274A1",
    "Physical":            "#E1812C",
    "Pragmatic":           "#3A923A",
    "SysPrompt":           "#845B53",
    "Human vs AI (General)": "#999999",
    "Bio Ctrl":            "#D4A03A",
    "Shapes":              "#E377C2",
}

CAT_ORDER = ["Mental", "SysPrompt", "Physical", "Pragmatic",
             "Human vs AI (General)", "Bio Ctrl", "Shapes"]

# The JSON was generated with "Baseline" as the category key.
# Map it to our display name so look-ups work on the stats dict.
JSON_CAT_NAME = {
    "Human vs AI (General)": "Baseline",
}

PROBE_TYPES = ["control_probe", "reading_probe"]
LAYER_RANGES = ["all_layers", "layers_6plus"]


def json_cat(display_name):
    """Map display category name to the key used in the stats JSON."""
    return JSON_CAT_NAME.get(display_name, display_name)


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
    """Load the master JSON from 2d_concept_probe_stats.py."""
    with open(STATS_JSON) as f:
        return json.load(f)


def get_dim_ids(stats):
    """Get sorted list of integer dim IDs."""
    return sorted(int(k) for k in stats["dimensions"].keys())


# ========================== FIGURE: RANKED BARS ========================== #

def fig_ranked_bars(stats, probe_type, layer_range, out_dir, xlim=None):
    """
    Horizontal bars, all dims ranked by alignment, with CIs and significance stars.
    xlim: optional shared (xmin, xmax) so control & reading use the same scale.
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

    fig, ax = plt.subplots(figsize=(10, 7))
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
            ax.text(hi + 0.002, i, sig, va="center", fontsize=7,
                    color="#333333", fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()

    ax.set_xlabel("Mean Projection onto Probe Direction\n(95% bootstrap CI)")
    ax.set_title(f"Concept Direction ↔ {pt_label(probe_type)} Probe ({lr_label(layer_range)})")

    cat_handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c) for c in CAT_ORDER]
    ax.legend(handles=cat_handles, loc="lower right", fontsize=7, ncol=2,
              framealpha=0.9, edgecolor="none")

    if xlim is not None:
        ax.set_xlim(xlim)

    fname = f"fig_ranked_bars_{layer_range}.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: LAYER PROFILES (summary) ========================== #

def fig_layer_profiles(stats, probe_type, out_dir, ylim=None):
    """Layer-by-layer cosine for key dims + mental avg band.
    ylim: optional shared (ymin, ymax) so control & reading use the same scale.
    """
    dims = stats["dimensions"]
    fig, ax = plt.subplots(figsize=(10, 5))

    key = f"{probe_type}_per_layer_cosines"

    # Compute mental average
    mental_stacks = []
    for d in CATEGORIES["Mental"]:
        cosines = dims.get(str(d), {}).get(key, [])
        if cosines:
            mental_stacks.append(cosines)

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
        18: {"color": CAT_PALETTE["SysPrompt"], "lw": 1.5, "ls": "--"},
        0:  {"color": CAT_PALETTE["Human vs AI (General)"], "lw": 1.5, "ls": "--"},
        15: {"color": CAT_PALETTE["Shapes"], "lw": 2.0, "ls": ":"},
        11: {"color": CAT_PALETTE["Pragmatic"], "lw": 1.5, "ls": ":"},
    }
    for dim_id, style in highlight.items():
        cosines = dims.get(str(dim_id), {}).get(key, [])
        if cosines:
            ax.plot(range(len(cosines)), cosines,
                    color=style["color"], lw=style["lw"], ls=style["ls"],
                    label=DIM_LABELS_SHORT.get(dim_id, str(dim_id)), zorder=3)

    # Background: all other dims as thin gray
    for did_str, d in dims.items():
        did = int(did_str)
        if did in [0, 11, 15, 18] or did in CATEGORIES["Mental"]:
            continue
        cosines = d.get(key, [])
        if cosines:
            ax.plot(range(len(cosines)), cosines, color="#CCCCCC", lw=0.5,
                    alpha=0.5, zorder=1)

    ax.axhline(0, color="black", lw=0.4, zorder=0)
    ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6, zorder=2)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Layer-by-Layer Concept ↔ {pt_label(probe_type)} Probe Alignment")
    ax.set_xlim(0, 40)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.text(RESTRICTED_LAYER_START + 0.3, ax.get_ylim()[1] * 0.95, "L6+",
            fontsize=7, color="gray", va="top")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", fontsize=7.5)

    fname = "fig_layer_profiles.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: LAYER PROFILES GRID (all dims) ========================== #

def fig_layer_profiles_grid(stats, probe_type, out_dir, ylim=None):
    """Small-multiples grid: one subplot per dimension, showing layer-by-layer
    cosine alignment with the probe. Dimensions are ordered by category."""
    dims = stats["dimensions"]
    key = f"{probe_type}_per_layer_cosines"

    # Build ordered dim list: by category order, then dim ID within category
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
        cosines = dims.get(str(dim_id), {}).get(key, [])
        if not cosines:
            ax.set_visible(False)
            continue

        color = dim_color(dim_id)
        ax.plot(range(len(cosines)), cosines, color=color, lw=1.5)
        ax.axhline(0, color="black", lw=0.3)
        ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=0.6, alpha=0.5)

        # Significance overlay: shade layers where p < .05
        perm_key = f"{probe_type}_per_layer_perm"
        perm = dims.get(str(dim_id), {}).get(perm_key, [])
        if perm:
            for r in perm:
                if r["p_value"] < 0.05:
                    ax.axvspan(r["layer"] - 0.5, r["layer"] + 0.5,
                               color=color, alpha=0.08, zorder=0)

        cat = dim_cat(dim_id)
        label = DIM_LABELS_SHORT.get(dim_id, str(dim_id))
        ax.set_title(f"{label} [{cat}]", fontsize=8, color=color, fontweight="bold")
        ax.set_xlim(0, 40)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    if ylim is not None:
        axes_flat[0].set_ylim(ylim)

    # Shared axis labels
    fig.supxlabel("Transformer Layer", fontsize=11)
    fig.supylabel("Cosine Similarity", fontsize=11)
    fig.suptitle(f"Per-Dimension Layer Profiles: {pt_label(probe_type)} Probe\n"
                 f"(shaded layers = permutation p < .05)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.95])

    fname = "fig_layer_profiles_grid.png"
    savefig(fig, os.path.join(out_dir, fname))


# ========================== FIGURE: HEATMAP ========================== #

def fig_heatmap(stats, probe_type, out_dir, vmax=None):
    """Layer x dim heatmap of per-layer cosines with significance dots.
    vmax: optional shared color scale so control & reading use the same range.
    """
    dims = stats["dimensions"]

    # Sort dims by category then by mean alignment
    key_cosines = f"{probe_type}_per_layer_cosines"
    key_perm = f"{probe_type}_per_layer_perm"

    sorted_dims = []
    for cat in CAT_ORDER:
        cat_dims = []
        for d in CATEGORIES[cat]:
            cosines = dims.get(str(d), {}).get(key_cosines, [])
            m = np.mean(cosines) if cosines else 0
            cat_dims.append((d, m))
        cat_dims.sort(key=lambda x: -x[1])
        sorted_dims.extend([d for d, _ in cat_dims])

    matrix = []
    col_labels = []
    sig_mask = []  # (layers, dims) boolean
    for d in sorted_dims:
        cosines = dims.get(str(d), {}).get(key_cosines, [])
        perm = dims.get(str(d), {}).get(key_perm, [])
        if not cosines:
            continue
        matrix.append(cosines)
        col_labels.append(DIM_LABELS_SHORT.get(d, str(d)))

        # Per-layer significance
        if perm:
            sig_mask.append([r["p_value"] < 0.05 for r in perm])
        else:
            sig_mask.append([False] * len(cosines))

    if not matrix:
        return

    matrix = np.array(matrix).T   # (layers, dims)
    sig_mask = np.array(sig_mask).T  # (layers, dims)

    fig, ax = plt.subplots(figsize=(8, 7))
    if vmax is None:
        vmax = np.percentile(np.abs(matrix), 98)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Significance dots
    for layer in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if sig_mask[layer, col]:
                ax.plot(col, layer, 'k.', markersize=2, alpha=0.6)

    ax.set_xlabel("Concept Dimension")
    ax.set_ylabel("Transformer Layer")
    ax.set_title(f"Cosine Alignment ↔ {pt_label(probe_type)} Probe (dots = p < .05)")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(0, 41, 5))

    # Category separators
    pos = 0
    for cat in CAT_ORDER:
        n = len([d for d in CATEGORIES[cat]
                 if str(d) in dims and dims[str(d)].get(key_cosines)])
        if n > 0 and pos > 0:
            ax.axvline(pos - 0.5, color="white", lw=1.5)
        pos += n

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=9)

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
        key = f"{probe}_per_layer_perm"
        counts = np.zeros(n_layers)
        for did_str, d in dims.items():
            perm = d.get(key, [])
            if perm:
                for r in perm:
                    if r["p_value"] < 0.05:
                        counts[r["layer"]] += 1

        ax.plot(range(n_layers), counts, color=color, lw=2, label=label)
        ax.fill_between(range(n_layers), counts, alpha=0.15, color=color)

    total_dims = len([d for d in dims.values()
                      if "control_probe_per_layer_perm" in d])
    ax.axhline(total_dims, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.5, total_dims + 0.3, f"n = {total_dims} dims", fontsize=7, color="gray")
    ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6)

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("# Significant Dimensions (p < .05)")
    ax.set_title("Layerwise Significance Count")
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

    cat_labels = ["Mental\n(1-7,17)", "SysPr.\n(18)", "Physical\n(8-10)",
                  "Pragm.\n(11-13)", "H vs AI\n(Gen, 0)", "Bio\n(14)", "Shapes\n(15)"]
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
            cr = cat_data.get(key, {}).get(json_cat(cn))
            if cr:
                cat_means.append(cr["mean"])
                cat_lo.append(cr["mean"] - cr["ci_lo"])
                cat_hi.append(cr["ci_hi"] - cr["mean"])
            else:
                cat_means.append(0)
                cat_lo.append(0)
                cat_hi.append(0)

            # Individual dim values for strip overlay
            dim_vals = []
            for d in CATEGORIES[cn]:
                dr = dim_data.get(str(d), {}).get(key)
                if dr:
                    dim_vals.append(dr["observed_projection"])
            cat_individual.append(dim_vals)

        ax.bar(x + offset, cat_means, width, yerr=[cat_lo, cat_hi],
               color=color, alpha=0.75, capsize=3,
               error_kw={"lw": 0.8}, label=label, edgecolor="white", linewidth=0.5)

        # Overlay individual dims as dots
        rng = np.random.default_rng(42)
        for i, vals in enumerate(cat_individual):
            if len(vals) > 1:
                jitter = rng.uniform(-width * 0.3, width * 0.3, len(vals))
                ax.scatter(x[i] + offset + jitter, vals, s=15, color="white",
                           edgecolors=color, linewidth=0.6, zorder=5, alpha=0.8)

    ax.axhline(0, color="black", lw=0.4)

    ax.set_ylabel("Mean Projection Test Statistic\n(human − AI, averaged across layers)")
    ax.set_title(f"Representational Alignment by Concept Category ({lr_label(layer_range)})")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="none")

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
        ha, va, ox, oy = "left", "bottom", 5, 4
        if did == 11:
            ha, va, ox, oy = "right", "top", -5, -4
        elif did == 10:
            ha, va, ox, oy = "right", "top", -5, -4
        elif did == 15:
            ha, va, ox, oy = "left", "top", 5, -4
        elif did == 0:
            ha, va, ox, oy = "right", "bottom", -5, 4
        ax.annotate(label, (ctrl, read), fontsize=7, ha=ha, va=va,
                    xytext=(ox, oy), textcoords="offset points", color="#333333")

    ax.axhline(0, color="black", lw=0.3, zorder=0)
    ax.axvline(0, color="black", lw=0.3, zorder=0)

    # Diagonal reference
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, color="gray", ls=":", lw=0.6, alpha=0.4, zorder=0)

    ax.set_xlabel("Alignment with Control Probe\n(in-context partner rep. during generation)")
    ax.set_ylabel("Alignment with Reading Probe\n(reflective partner rep.)")
    ax.set_title(f"Concept-Conversation Alignment:\nControl vs Reading Probe ({lr_label(layer_range)})")

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

    # Build matrix of -log10(p_adjusted)
    mat = np.full((n, n), np.nan)
    for r in pairs:
        i, j = id_to_idx[r["dim_a"]], id_to_idx[r["dim_b"]]
        p = max(r["p_adjusted"], 1e-10)
        val = -np.log10(p)
        mat[j, i] = val  # lower triangle

    fig, ax = plt.subplots(figsize=(9, 8))

    # Mask upper triangle
    mask = np.triu(np.ones_like(mat, dtype=bool))
    masked = np.ma.array(mat, mask=mask)

    im = ax.imshow(masked, cmap="YlOrRd", interpolation="nearest",
                   vmin=0, vmax=4)

    labels = [DIM_LABELS_SHORT.get(d, str(d)) for d in dim_ids]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=7)

    # Mark significant cells
    for r in pairs:
        i, j = id_to_idx[r["dim_a"]], id_to_idx[r["dim_b"]]
        if r["p_adjusted"] < 0.05:
            ax.text(i, j, "*", ha="center", va="center", fontsize=6,
                    fontweight="bold", color="white" if r["p_adjusted"] < 0.001 else "black")

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

    # Forest plot style: each comparison as a horizontal CI
    pairs_sorted = sorted(pairs, key=lambda r: r["diff"], reverse=True)

    y_pos = np.arange(len(pairs_sorted))
    labels = []
    for r in pairs_sorted:
        # Map JSON cat names to display names
        a = r['cat_a']
        b = r['cat_b']
        for display, jname in JSON_CAT_NAME.items():
            if a == jname:
                a = display
            if b == jname:
                b = display
        labels.append(f"{a} vs {b}")

    diffs = [r["diff"] for r in pairs_sorted]
    ci_lo = [r["diff"] - r["ci_lo"] for r in pairs_sorted]
    ci_hi = [r["ci_hi"] - r["diff"] for r in pairs_sorted]

    colors = []
    for r in pairs_sorted:
        if r["p_adjusted"] < 0.05:
            colors.append("#C03D3E")
        else:
            colors.append("#999999")

    ax.barh(y_pos, diffs, xerr=[ci_lo, ci_hi], color=colors,
            edgecolor="white", linewidth=0.5, capsize=2, error_kw={"lw": 0.8},
            height=0.7)
    ax.axvline(0, color="black", lw=0.5)

    # Stars
    for i, r in enumerate(pairs_sorted):
        sig = r.get("sig_fdr", "n.s.")
        if sig != "n.s.":
            x_pos = r["ci_hi"] + 0.001
            ax.text(x_pos, i, sig, va="center", fontsize=7,
                    fontweight="bold", color="#333")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Alignment Difference (95% bootstrap CI)")
    ax.set_title(f"Pairwise Category Comparisons: {pt_label(probe_type)} Probe "
                 f"({lr_label(layer_range)})\nRed = significant after FDR correction")

    savefig(fig, os.path.join(out_dir,
            f"fig_category_pairwise_{pt_short(probe_type)}_{layer_range}.png"))


# ========================== FIGURE: MAIN RESULT (3-PANEL) ========================== #

def fig_main_result(stats, out_dir):
    """
    Composite 3-panel for paper:
      A: Layer profiles (control probe)
      B: Layer profiles (reading probe)
      C: Ranked bars (control probe, all layers)
    """
    dims = stats["dimensions"]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ── Panels A/B: Layer profiles ──
    for ax, probe, title_suffix in [
        (ax_a, "control_probe", "Control Probe"),
        (ax_b, "reading_probe", "Reading Probe"),
    ]:
        key = f"{probe}_per_layer_cosines"

        mental_stacks = []
        for d in CATEGORIES["Mental"]:
            cosines = dims.get(str(d), {}).get(key, [])
            if cosines:
                mental_stacks.append(cosines)

        if mental_stacks:
            mental_avg = np.mean(mental_stacks, axis=0)
            mental_sem = np.std(mental_stacks, axis=0, ddof=1) / np.sqrt(len(mental_stacks))
            layers = list(range(len(mental_avg)))
            ax.plot(layers, mental_avg, color=CAT_PALETTE["Mental"], lw=2.5,
                    label="Mental (avg)", zorder=5)
            ax.fill_between(layers, mental_avg - mental_sem, mental_avg + mental_sem,
                            color=CAT_PALETTE["Mental"], alpha=0.15, zorder=4)

        highlight_dims = {
            18: {"color": CAT_PALETTE["SysPrompt"], "lw": 1.5, "ls": "--"},
            0:  {"color": CAT_PALETTE["Human vs AI (General)"], "lw": 1.5, "ls": "--"},
            15: {"color": CAT_PALETTE["Shapes"], "lw": 2.0, "ls": ":"},
            11: {"color": CAT_PALETTE["Pragmatic"], "lw": 1.5, "ls": ":"},
        }
        for dim_id, style in highlight_dims.items():
            cosines = dims.get(str(dim_id), {}).get(key, [])
            if cosines:
                ax.plot(range(len(cosines)), cosines,
                        color=style["color"], lw=style["lw"], ls=style["ls"],
                        label=DIM_LABELS_SHORT.get(dim_id, str(dim_id)), zorder=3)

        for did_str, d in dims.items():
            did = int(did_str)
            if did in [0, 11, 15, 18] or did in CATEGORIES["Mental"]:
                continue
            cosines = d.get(key, [])
            if cosines:
                ax.plot(range(len(cosines)), cosines, color="#CCCCCC", lw=0.5,
                        alpha=0.5, zorder=1)

        ax.axhline(0, color="black", lw=0.4, zorder=0)
        ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6, zorder=2)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Concept ↔ Exp 2 {title_suffix}")
        ax.set_xlim(0, 40)
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", fontsize=7.5)

    # Equalize y-axis across panels A and B
    yl_a = ax_a.get_ylim()
    yl_b = ax_b.get_ylim()
    shared_yl = (min(yl_a[0], yl_b[0]), max(yl_a[1], yl_b[1]))
    ax_a.set_ylim(shared_yl)
    ax_b.set_ylim(shared_yl)

    # ── Panel C: Ranked bars (control, all layers) — projection metric ──
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
            ax_c.text(hi + 0.002, i, sig, va="center", fontsize=7,
                      color="#333333", fontweight="bold")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(names, fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Mean Projection onto Exp 2 Control Probe Direction\n(95% bootstrap CI)")
    ax_c.set_title("Alignment of Concept Directions with Conversational Partner-Identity Probe")

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
    cat_labels = ["Mental\n(1-7,17)", "SysPr.\n(18)", "Physical\n(8-10)",
                  "Pragm.\n(11-13)", "H vs AI\n(Gen, 0)", "Bio\n(14)", "Shapes\n(15)"]
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
            cr = cat_data.get(key, {}).get(json_cat(cn))
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
    ax1.set_ylabel("Mean Alignment")
    ax1.set_title("Alignment by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=8)
    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9, edgecolor="none")

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
    print("Loading alignment stats...")
    stats = load_stats()
    dims = stats["dimensions"]
    n_dims = len(dims)
    print(f"  {n_dims} dimensions loaded\n")

    # Create output directories
    for subdir in ["control_probe", "reading_probe", "layerwise", "comparisons"]:
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
            shared_xlims[lr] = (min(min(all_vals), 0) - 0.005,
                                max(all_vals) + 0.025)

    # Layer profiles: shared ylim across both probes
    all_cosines = []
    for probe in PROBE_TYPES:
        key = f"{probe}_per_layer_cosines"
        for d in dims.values():
            cosines = d.get(key, [])
            if cosines:
                all_cosines.extend(cosines)
    shared_ylim = None
    if all_cosines:
        shared_ylim = (min(all_cosines) - 0.02, max(all_cosines) + 0.02)

    # Heatmaps: shared vmax across both probes
    all_heatmap_abs = []
    for probe in PROBE_TYPES:
        key = f"{probe}_per_layer_cosines"
        for d in dims.values():
            cosines = d.get(key, [])
            if cosines:
                all_heatmap_abs.extend([abs(c) for c in cosines])
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

    # ── Composite figures ──
    print("\ncomposite:")
    fig_main_result(stats, FIG_ROOT)
    fig_summary_panel(stats, FIG_ROOT)

    print(f"\nAll figures saved to: {FIG_ROOT}/")
    print("Done.")


if __name__ == "__main__":
    main()
