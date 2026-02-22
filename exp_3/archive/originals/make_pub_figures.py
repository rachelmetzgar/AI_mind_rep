#!/usr/bin/env python3
"""
Publication-quality figures for Experiment 3 Phase 2 results.
Concept probe alignment with Exp 2b conversational probes.

Outputs to: data/concept_probes/summary_stats/pub_figures/

Rachel C. Metzgar · Feb 2026
"""

import os
import json
import pickle
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
PROBE_ROOT = os.path.join(BASE, "data/concept_probes")
ALIGN_ROOT = os.path.join(BASE, "data/alignment")
OUT_DIR = os.path.join(PROBE_ROOT, "summary_stats", "pub_figures")
os.makedirs(OUT_DIR, exist_ok=True)

HIDDEN_DIM = 5120
SIGMA_CHANCE = 1.0 / np.sqrt(HIDDEN_DIM)

DIM_LABELS = {
    0: "Baseline\n(generic entity)",
    1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social cognition",
    8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpfulness",
    14: "Biological", 15: "Shapes\n(negative ctrl)",
    16: "Mind (holistic)", 17: "Attention", 18: "System prompt",
}

DIM_LABELS_SHORT = {
    0: "Baseline", 1: "Phenom.", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social", 8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpful.",
    14: "Biological", 15: "Shapes", 16: "Mind", 17: "Attention",
    18: "SysPrompt",
}

CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 16, 17],  # 16=holistic mind (pooled 1-10), 17=attention
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Baseline":  [0],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "SysPrompt": [18],
}

RESTRICTED_LAYER_START = 6

def dim_cat(d):
    for cat, ids in CATEGORIES.items():
        if d in ids:
            return cat
    return "Other"

# Colorblind-friendly palette
CAT_PALETTE = {
    "Mental":    "#3274A1",  # steel blue
    "Physical":  "#E1812C",  # burnt orange
    "Pragmatic": "#3A923A",  # forest green
    "SysPrompt": "#845B53",  # brown
    "Baseline":  "#999999",  # gray
    "Bio Ctrl":  "#D4A03A",  # gold
    "Shapes":    "#E377C2",  # pink
}

def dim_color(d):
    return CAT_PALETTE.get(dim_cat(d), "#666666")


# ========================== DATA LOADING ========================== #

def load_all():
    """Load all alignment + accuracy data."""
    data = {}
    for name in sorted(os.listdir(PROBE_ROOT)):
        full = os.path.join(PROBE_ROOT, name)
        if not os.path.isdir(full):
            continue
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue

        entry = {"dim_name": name, "dim_id": dim_id, "category": dim_cat(dim_id)}

        # accuracy
        pkl = os.path.join(full, "accuracy_summary.pkl")
        if os.path.isfile(pkl):
            with open(pkl, "rb") as f:
                s = pickle.load(f)
            entry["accs"] = np.array(s.get("acc", []))

        # alignment
        for pt in ["control_probe", "reading_probe"]:
            jp = os.path.join(ALIGN_ROOT, name, pt, "alignment_results.json")
            if os.path.isfile(jp):
                with open(jp) as f:
                    aj = json.load(f)
                entry[f"{pt}_layers"] = aj.get("layers", [])
                entry[f"{pt}_mean_to_2b"] = aj.get("mean_to_2b", [])
                entry[f"{pt}_probe_to_2b"] = aj.get("probe_to_2b", [])
                entry[f"{pt}_probe_to_mean"] = aj.get("probe_to_mean", [])

        data[dim_id] = entry
    return data


def bootstrap_ci(vals, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.array([v for v in vals if v is not None])
    if len(arr) < 2:
        return np.mean(arr), np.mean(arr), np.mean(arr)
    boots = np.array([np.mean(rng.choice(arr, len(arr), replace=True))
                      for _ in range(n_boot)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), float(np.mean(arr))


# ========================== FIGURE 1: MAIN RESULT ========================== #
# Multi-panel: (A) Layer profiles for key dims, (B) Ranked bar chart

def fig1_main_result(data):
    """
    3-panel figure:
      A: Layer-by-layer cosine alignment (concept mean-diff ↔ control probe) for key dims
      B: Layer-by-layer cosine alignment (concept mean-diff ↔ reading probe) for key dims
      C: Ranked bar chart of all dimensions, mean cosine ↔ control probe
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # -- Key dimensions to highlight in layer profiles --
    highlight = {
        "Mental (avg)": {"color": CAT_PALETTE["Mental"], "lw": 2.5, "ls": "-"},
        18: {"color": CAT_PALETTE["SysPrompt"], "lw": 1.5, "ls": "--"},
        0:  {"color": CAT_PALETTE["Baseline"], "lw": 1.5, "ls": "--"},
        15: {"color": CAT_PALETTE["Shapes"], "lw": 2.0, "ls": ":"},
        11: {"color": CAT_PALETTE["Pragmatic"], "lw": 1.5, "ls": ":"},
    }

    for ax, pt, title_suffix in [
        (ax_a, "control_probe", "Control Probe"),
        (ax_b, "reading_probe", "Reading Probe"),
    ]:
        key = f"{pt}_mean_to_2b"

        # Compute mental average
        mental_stacks = []
        for d in CATEGORIES["Mental"]:
            vals = data.get(d, {}).get(key, [])
            if vals:
                mental_stacks.append(vals)
        if mental_stacks:
            mental_avg = np.mean(mental_stacks, axis=0)
            mental_sem = np.std(mental_stacks, axis=0, ddof=1) / np.sqrt(len(mental_stacks))
            layers = list(range(len(mental_avg)))
            style = highlight["Mental (avg)"]
            ax.plot(layers, mental_avg, color=style["color"], lw=style["lw"],
                    ls=style["ls"], label="Mental (avg)", zorder=5)
            ax.fill_between(layers, mental_avg - mental_sem, mental_avg + mental_sem,
                            color=style["color"], alpha=0.15, zorder=4)

        # Plot individual highlighted dims
        for dim_id in [18, 0, 15, 11]:
            vals = data.get(dim_id, {}).get(key, [])
            if not vals:
                continue
            layers = list(range(len(vals)))
            style = highlight[dim_id]
            label = DIM_LABELS_SHORT.get(dim_id, str(dim_id))
            ax.plot(layers, vals, color=style["color"], lw=style["lw"],
                    ls=style["ls"], label=label, zorder=3)

        # Background: all other dims as thin gray
        for dim_id, d in data.items():
            if dim_id in [0, 11, 15, 18] or dim_id in CATEGORIES["Mental"]:
                continue
            vals = d.get(key, [])
            if vals:
                ax.plot(range(len(vals)), vals, color="#CCCCCC", lw=0.5, alpha=0.5, zorder=1)

        ax.axhline(0, color="black", lw=0.4, zorder=0)
        ax.axhline(3 * SIGMA_CHANCE, color="red", ls="--", lw=0.8, alpha=0.5, zorder=2)
        ax.axvline(RESTRICTED_LAYER_START, color="gray", ls="-.", lw=1.0, alpha=0.6, zorder=2)
        ax.text(RESTRICTED_LAYER_START + 0.3, ax.get_ylim()[1] * 0.95, "L6+",
                fontsize=7, color="gray", va="top")
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Concept ↔ Exp 2 {title_suffix}")
        ax.set_xlim(0, 40)
        if pt == "control_probe":
            ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", fontsize=7.5)
        else:
            ax.legend(loc="upper left", framealpha=0.9, edgecolor="none", fontsize=7.5)

    # -- Panel C: Ranked horizontal bar chart (control probe alignment) --
    pt = "control_probe"
    dim_means = []
    for dim_id, d in data.items():
        vals = d.get(f"{pt}_mean_to_2b", [])
        if not vals:
            continue
        ci_lo, ci_hi, mean_val = bootstrap_ci(vals)
        dim_means.append((dim_id, mean_val, ci_lo, ci_hi))

    dim_means.sort(key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(dim_means))
    names = []
    means = []
    errs_lo = []
    errs_hi = []
    colors = []
    for dim_id, m, lo, hi in dim_means:
        names.append(DIM_LABELS.get(dim_id, str(dim_id)))
        means.append(m)
        errs_lo.append(m - lo)
        errs_hi.append(hi - m)
        colors.append(dim_color(dim_id))

    bars = ax_c.barh(y_pos, means, xerr=[errs_lo, errs_hi],
                     color=colors, edgecolor="white", linewidth=0.5,
                     capsize=2, error_kw={"lw": 0.8})
    ax_c.axvline(0, color="black", lw=0.4)
    ax_c.axvline(3 * SIGMA_CHANCE, color="red", ls="--", lw=0.8, alpha=0.5)
    ax_c.axvline(SIGMA_CHANCE, color="red", ls=":", lw=0.5, alpha=0.3)

    # Significance stars (from the stats JSON)
    stats_path = os.path.join(PROBE_ROOT, "summary_stats", "statistical_results.json")
    if os.path.isfile(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        vs_chance = stats.get("vs_chance", {}).get(pt, {})
        for i, (dim_id, m, lo, hi) in enumerate(dim_means):
            res = vs_chance.get(str(dim_id), {})
            p = res.get("p_value", 1.0)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                ax_c.text(hi + 0.0008, i, star, va="center", fontsize=7,
                          color="#333333", fontweight="bold")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(names, fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Mean Cosine Similarity with Exp 2 Control Probe\n(95% bootstrap CI)")
    ax_c.set_title("Alignment of Concept Directions with Conversational Partner-Identity Probe")

    # Category legend for panel C
    cat_handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c)
                   for c in ["Mental", "Physical", "Pragmatic",
                             "SysPrompt", "Baseline", "Bio Ctrl", "Shapes"]]
    cat_handles.append(Line2D([], [], color="red", ls="--", lw=0.8, alpha=0.5, label="3σ chance"))
    ax_c.legend(handles=cat_handles, loc="lower right", fontsize=7, ncol=2,
                framealpha=0.9, edgecolor="none")

    # Panel labels
    ax_a.text(-0.12, 1.05, "A", transform=ax_a.transAxes, fontsize=14, fontweight="bold")
    ax_b.text(-0.12, 1.05, "B", transform=ax_b.transAxes, fontsize=14, fontweight="bold")
    ax_c.text(-0.05, 1.03, "C", transform=ax_c.transAxes, fontsize=14, fontweight="bold")

    path = os.path.join(OUT_DIR, "fig1_main_alignment.png")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ========================== FIGURE 2: CATEGORY COMPARISON ========================== #

def fig2_category(data):
    """
    Category-level grouped bar chart: control vs reading probe alignment.
    With individual dimension dots overlaid (strip plot style).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    cat_order = ["Mental", "SysPrompt",
                 "Physical", "Pragmatic", "Baseline", "Bio Ctrl", "Shapes"]
    cat_labels = ["Mental\n(1-7,16,17)", "SysPrompt\n(dim 18)", "Physical\n(dims 8-10)",
                  "Pragmatic\n(dims 11-13)", "Baseline\n(dim 0)",
                  "Bio Ctrl\n(dim 14)", "Shapes\n(dim 15)"]

    x = np.arange(len(cat_order))
    width = 0.35

    for offset, pt, color, label in [
        (-width/2, "control_probe", "#3274A1", "↔ Control probe (surface)"),
        (width/2,  "reading_probe", "#C03D3E", "↔ Reading probe (internal)"),
    ]:
        cat_means = []
        cat_sems = []
        cat_individual = []
        for cat_name in cat_order:
            dim_ids = CATEGORIES[cat_name]
            dim_vals = []
            for d in dim_ids:
                vals = data.get(d, {}).get(f"{pt}_mean_to_2b", [])
                if vals:
                    dim_vals.append(np.mean([v for v in vals if v is not None]))
            cat_means.append(np.mean(dim_vals) if dim_vals else 0)
            cat_sems.append(np.std(dim_vals, ddof=1) / np.sqrt(len(dim_vals))
                            if len(dim_vals) > 1 else 0)
            cat_individual.append(dim_vals)

        bars = ax.bar(x + offset, cat_means, width, yerr=cat_sems,
                      color=color, alpha=0.75, capsize=3,
                      error_kw={"lw": 0.8}, label=label, edgecolor="white", linewidth=0.5)

        # Overlay individual dims as dots
        for i, vals in enumerate(cat_individual):
            if len(vals) > 1:
                jitter = np.random.default_rng(42).uniform(-width*0.3, width*0.3, len(vals))
                ax.scatter(x[i] + offset + jitter, vals, s=15, color="white",
                           edgecolors=color, linewidth=0.6, zorder=5, alpha=0.8)

    ax.axhline(3 * SIGMA_CHANCE, color="red", ls="--", lw=0.8, alpha=0.5, label="3σ chance")
    ax.axhline(0, color="black", lw=0.4)

    ax.set_ylabel("Mean Cosine Similarity\n(concept direction ↔ Exp 2 probe)")
    ax.set_title("Representational Alignment by Concept Category")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9, edgecolor="none")

    path = os.path.join(OUT_DIR, "fig2_category_comparison.png")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ========================== FIGURE 3: HEATMAP ========================== #

def fig3_heatmap(data):
    """
    Layer × Dimension heatmap of cosine alignment with control probe.
    Rows = layers (0-40), columns = dimensions sorted by category then mean alignment.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25})

    for ax, pt, title in [
        (ax1, "control_probe", "↔ Exp 2 Control Probe"),
        (ax2, "reading_probe", "↔ Exp 2 Reading Probe"),
    ]:
        # Sort dims: by category group, then by mean alignment within group
        cat_order_for_sort = ["Mental", "SysPrompt",
                              "Physical", "Pragmatic", "Baseline", "Bio Ctrl", "Shapes"]
        sorted_dims = []
        for cat in cat_order_for_sort:
            cat_dims = []
            for d in CATEGORIES[cat]:
                vals = data.get(d, {}).get(f"{pt}_mean_to_2b", [])
                m = np.mean([v for v in vals if v is not None]) if vals else 0
                cat_dims.append((d, m))
            cat_dims.sort(key=lambda x: -x[1])
            sorted_dims.extend([d for d, _ in cat_dims])

        matrix = []
        col_labels = []
        for d in sorted_dims:
            vals = data.get(d, {}).get(f"{pt}_mean_to_2b", [])
            if vals:
                matrix.append(vals)
                col_labels.append(DIM_LABELS_SHORT.get(d, str(d)))

        if not matrix:
            continue
        matrix = np.array(matrix).T  # (layers, dims)

        vmax = np.percentile(np.abs(matrix), 98)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xlabel("Concept Dimension")
        ax.set_ylabel("Transformer Layer")
        ax.set_title(f"Cosine Alignment {title}")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=55, ha="right", fontsize=7)
        ax.set_yticks(range(0, 41, 5))

        # Category separators
        cat_starts = {}
        pos = 0
        for cat in cat_order_for_sort:
            n = len([d for d in CATEGORIES[cat] if d in [sd for sd in sorted_dims]])
            if n > 0:
                cat_starts[cat] = (pos, pos + n - 1)
                if pos > 0:
                    ax.axvline(pos - 0.5, color="white", lw=1.5)
                pos += n

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Cosine Similarity", fontsize=9)

    ax1.text(-0.08, 1.03, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")
    ax2.text(-0.08, 1.03, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    path = os.path.join(OUT_DIR, "fig3_layer_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ========================== FIGURE 4: SPECIFICITY SCATTER ========================== #

def fig4_specificity(data):
    """
    Scatter: control probe alignment (x) vs reading probe alignment (y).
    Each point = one concept dimension. Shows that mental/mind dims are
    high on both, shapes/formality are near zero.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for dim_id, d in data.items():
        ctrl = d.get("control_probe_mean_to_2b", [])
        read = d.get("reading_probe_mean_to_2b", [])
        if not ctrl or not read:
            continue
        x = np.mean([v for v in ctrl if v is not None])
        y = np.mean([v for v in read if v is not None])
        c = dim_color(dim_id)
        ax.scatter(x, y, s=90, color=c, edgecolors="white", linewidth=0.8, zorder=5)
        label = DIM_LABELS_SHORT.get(dim_id, str(dim_id))
        # Smart label placement
        ha, va, ox, oy = "left", "bottom", 5, 4
        if dim_id == 11:  # formality (bottom left)
            ha, va, ox, oy = "right", "top", -5, -4
        elif dim_id == 10:  # animacy
            ha, va, ox, oy = "right", "top", -5, -4
        elif dim_id == 15:  # shapes
            ha, va, ox, oy = "left", "top", 5, -4
        elif dim_id == 0:
            ha, va, ox, oy = "right", "bottom", -5, 4
        ax.annotate(label, (x, y), fontsize=7, ha=ha, va=va,
                    xytext=(ox, oy), textcoords="offset points",
                    color="#333333")

    ax.axhline(0, color="black", lw=0.3, zorder=0)
    ax.axvline(0, color="black", lw=0.3, zorder=0)
    ax.axhline(3 * SIGMA_CHANCE, color="red", ls=":", lw=0.6, alpha=0.4)
    ax.axvline(3 * SIGMA_CHANCE, color="red", ls=":", lw=0.6, alpha=0.4)

    ax.set_xlabel("Alignment with Control Probe\n(surface-level partner detection)")
    ax.set_ylabel("Alignment with Reading Probe\n(internal model of partner)")
    ax.set_title("Concept-Conversation Alignment:\nControl vs Reading Probe")

    # Category legend
    handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c)
               for c in ["Mental", "Physical", "Pragmatic",
                          "SysPrompt", "Baseline", "Bio Ctrl", "Shapes"]]
    ax.legend(handles=handles, loc="upper left", fontsize=7,
              framealpha=0.9, edgecolor="none")

    path = os.path.join(OUT_DIR, "fig4_ctrl_vs_read_scatter.png")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ========================== FIGURE 5: ACCURACY PROFILES ========================== #

def fig5_accuracy(data):
    """
    Probe classification accuracy by layer, grouped by category.
    Shows that most dims achieve near-ceiling, while shapes/helpfulness lag.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Mental dims as a band
    mental_accs = []
    for d in CATEGORIES["Mental"]:
        accs = data.get(d, {}).get("accs", np.array([]))
        if len(accs) > 0:
            mental_accs.append(accs)
    if mental_accs:
        mental_avg = np.mean(mental_accs, axis=0)
        mental_min = np.min(mental_accs, axis=0)
        mental_max = np.max(mental_accs, axis=0)
        layers = np.arange(len(mental_avg))
        ax.fill_between(layers, mental_min, mental_max,
                        color=CAT_PALETTE["Mental"], alpha=0.2, label="Mental (range)")
        ax.plot(layers, mental_avg, color=CAT_PALETTE["Mental"], lw=2, label="Mental (avg)")

    # Key individual dims
    for dim_id, ls, lw in [(15, ":", 2), (13, "--", 1.5), (0, "-.", 1.5), (16, "-", 1.5)]:
        accs = data.get(dim_id, {}).get("accs", np.array([]))
        if len(accs) > 0:
            ax.plot(range(len(accs)), accs, color=dim_color(dim_id),
                    ls=ls, lw=lw, label=DIM_LABELS_SHORT[dim_id])

    ax.axhline(0.5, color="gray", ls=":", lw=0.5, alpha=0.3)
    ax.axhline(0.7, color="red", ls="--", lw=0.7, alpha=0.4, label="70% threshold")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Concept Probe Classification Accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(0, 40)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, edgecolor="none")

    path = os.path.join(OUT_DIR, "fig5_accuracy_profiles.png")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ========================== FIGURE 6: COMBINED SUMMARY ========================== #

def fig6_summary(data):
    """
    Single summary panel for talks/posters:
    Left: category-level bars. Right: control vs reading scatter.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                    gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})

    # -- Left: Category bars --
    cat_order = ["Mental", "SysPrompt",
                 "Physical", "Pragmatic", "Baseline", "Bio Ctrl", "Shapes"]
    cat_labels = ["Mental\n(1-7,16,17)", "SysPr.\n(18)",
                  "Physical\n(8-10)", "Pragm.\n(11-13)", "Baseln.\n(0)",
                  "Bio\n(14)", "Shapes\n(15)"]

    x = np.arange(len(cat_order))
    width = 0.35

    for offset, pt, color, label in [
        (-width/2, "control_probe", "#3274A1", "Control probe"),
        (width/2,  "reading_probe", "#C03D3E", "Reading probe"),
    ]:
        cat_means = []
        cat_sems = []
        for cat_name in cat_order:
            dim_vals = []
            for d in CATEGORIES[cat_name]:
                vals = data.get(d, {}).get(f"{pt}_mean_to_2b", [])
                if vals:
                    dim_vals.append(np.mean([v for v in vals if v is not None]))
            cat_means.append(np.mean(dim_vals) if dim_vals else 0)
            cat_sems.append(np.std(dim_vals, ddof=1) / np.sqrt(len(dim_vals))
                            if len(dim_vals) > 1 else 0)

        ax1.bar(x + offset, cat_means, width, yerr=cat_sems,
                color=color, alpha=0.8, capsize=3,
                error_kw={"lw": 0.8}, label=label, edgecolor="white", linewidth=0.5)

    ax1.axhline(3 * SIGMA_CHANCE, color="red", ls="--", lw=0.8, alpha=0.5, label="3σ chance")
    ax1.axhline(0, color="black", lw=0.4)
    ax1.set_ylabel("Mean Cosine Similarity")
    ax1.set_title("Alignment by Category")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_labels, fontsize=8)
    ax1.legend(fontsize=7.5, loc="upper right", framealpha=0.9, edgecolor="none")

    # -- Right: Scatter --
    for dim_id, d in data.items():
        ctrl = d.get("control_probe_mean_to_2b", [])
        read = d.get("reading_probe_mean_to_2b", [])
        if not ctrl or not read:
            continue
        xv = np.mean([v for v in ctrl if v is not None])
        yv = np.mean([v for v in read if v is not None])
        c = dim_color(dim_id)
        ax2.scatter(xv, yv, s=70, color=c, edgecolors="white", linewidth=0.7, zorder=5)
        ax2.annotate(DIM_LABELS_SHORT.get(dim_id, str(dim_id)), (xv, yv),
                     fontsize=6.5, ha="left", va="bottom",
                     xytext=(4, 3), textcoords="offset points", color="#333333")

    ax2.axhline(0, color="black", lw=0.3)
    ax2.axvline(0, color="black", lw=0.3)
    ax2.set_xlabel("↔ Control Probe")
    ax2.set_ylabel("↔ Reading Probe")
    ax2.set_title("Control vs Reading Alignment")
    handles = [mpatches.Patch(color=CAT_PALETTE[c], label=c)
               for c in ["Mental", "Physical", "Pragmatic",
                          "SysPrompt", "Baseline", "Bio Ctrl", "Shapes"]]
    ax2.legend(handles=handles, loc="upper left", fontsize=6.5,
               framealpha=0.9, edgecolor="none")

    ax1.text(-0.08, 1.03, "A", transform=ax1.transAxes, fontsize=14, fontweight="bold")
    ax2.text(-0.08, 1.03, "B", transform=ax2.transAxes, fontsize=14, fontweight="bold")

    path = os.path.join(OUT_DIR, "fig6_summary_panel.png")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".png", ".pdf"))
    plt.close()
    print(f"  Saved: {path}")


# ========================== MAIN ========================== #

def main():
    print("Loading data...")
    data = load_all()
    print(f"  Found {len(data)} dimensions\n")

    print("Generating publication figures...")
    fig1_main_result(data)
    fig2_category(data)
    fig3_heatmap(data)
    fig4_specificity(data)
    fig5_accuracy(data)
    fig6_summary(data)

    print(f"\nAll figures saved to: {OUT_DIR}/")
    print("  fig1_main_alignment.{png,pdf}    — Main result (layer profiles + ranked bars)")
    print("  fig2_category_comparison.{png,pdf} — Category grouped bars with dot overlay")
    print("  fig3_layer_heatmap.{png,pdf}      — Layer × dimension heatmap")
    print("  fig4_ctrl_vs_read_scatter.{png,pdf} — Control vs reading alignment scatter")
    print("  fig5_accuracy_profiles.{png,pdf}  — Probe classification accuracy")
    print("  fig6_summary_panel.{png,pdf}      — Combined summary for talks")


if __name__ == "__main__":
    main()
