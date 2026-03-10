#!/usr/bin/env python3
"""
Experiment 3, Phase 2f: Concept Overlap Report Generator

Reads output from 2f_concept_overlap.py and produces HTML/MD report with
matplotlib figures and detailed narrative explaining the analysis.

Figures:
    1. Full pairwise overlap heatmap (n_dims x n_dims)
    2. Baseline overlap bar chart (per-dim overlap with dim 0)
    3. Baseline comparison (dim 0 vs dim 18 side-by-side)
    4. Layer-resolved cosine profiles: every dim vs Baseline (dim 0)
    5. Layer-resolved cosine profiles: every dim vs SysPrompt (dim 18)
    6. Category-level summary (within vs between category overlap)

Output goes to exp_3/results/{model}/concept_overlap/ alongside the data.

Usage:
    python 2f_concept_overlap_report.py

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import base64
import csv
import io
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from config import config, add_variant_argument, set_variant, variant_filename, get_variant_suffix, data_subdir

# ============================================================================
# CONFIG
# ============================================================================

OVERLAP_DIR = Path(str(config.RESULTS.concept_overlap)) / "contrasts"
OUTPUT_DIR = OVERLAP_DIR  # Report lives alongside data

DIM_NAMES = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind (holistic)", 17: "Attention", 18: "SysPrompt (labeled)",
    25: "Beliefs", 26: "Desires", 27: "Goals",
    30: "Granite/Sandstone", 31: "Squares/Triangles",
    32: "Horizontal/Vertical",
}

# Full description for the dimension table in the report
DIM_DESCRIPTIONS = {
    0: "Think about what it means to be a human/AI (entity baseline)",
    1: "Phenomenological experience — what-it-is-like-ness",
    2: "Emotional states and affect",
    3: "Agency and autonomous action",
    4: "Intentions, goals, and purposes",
    5: "Prediction and anticipation",
    6: "Cognitive processing and reasoning",
    7: "Social cognition and interaction",
    8: "Embodiment and physical form",
    9: "Social roles and identity",
    10: "Animacy and liveliness",
    11: "Formality of interaction",
    12: "Expertise and knowledge depth",
    13: "Helpfulness and cooperation",
    14: "Biological processes (negative control)",
    15: "Geometric shapes (negative control)",
    16: "Mind as holistic concept",
    17: "Attention and focus",
    18: "System prompt framing (you are speaking to human/AI)",
}

CATEGORY_FOR_DIM = {
    0: "Baseline", 1: "Mental", 2: "Mental", 3: "Mental",
    4: "Mental", 5: "Mental", 6: "Mental", 7: "Mental",
    8: "Mental", 9: "Mental", 10: "Physical", 11: "Pragmatic",
    12: "Pragmatic", 13: "Pragmatic", 14: "Bio Ctrl", 15: "Shapes",
    16: "Mental", 17: "Mental", 18: "SysPrompt",
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
    "Pragmatic": "#FF9800",
    "Baseline": "#9E9E9E",
    "Bio Ctrl": "#795548",
    "Shapes": "#E91E63",
    "SysPrompt": "#00BCD4",
    "Other": "#999",
}

CATEGORY_ORDER = ["Baseline", "Mental", "Pragmatic",
                  "Bio Ctrl", "Shapes", "SysPrompt"]

# Expected |cosine| for random unit vectors in 5120-d space
# E[|cos|] ~ sqrt(2/(pi*d)) for high d
CHANCE_LEVEL = np.sqrt(2.0 / (np.pi * 5120))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_overlap_data():
    """Load all overlap analysis outputs."""
    data = {}

    # Main overlap matrix
    npz_path = data_subdir(OVERLAP_DIR) / variant_filename("overlap_matrix", ".npz")
    if npz_path.exists():
        f = np.load(npz_path, allow_pickle=True)
        data["overlap"] = f["overlap"]
        data["boot_overlap"] = f["boot_overlap"] if "boot_overlap" in f else None
        data["dim_ids"] = f["dim_ids"].tolist()
        data["dim_names"] = f["dim_names"].tolist()
        data["dim_categories"] = f["dim_categories"].tolist()
        data["layer_range_start"] = int(f["layer_range_start"])
        data["layer_range_end"] = int(f["layer_range_end"])
        data["n_bootstrap"] = int(f["n_bootstrap"])
        if "excluded_dims" in f:
            data["excluded_dims"] = f["excluded_dims"].tolist()
        else:
            data["excluded_dims"] = []
    else:
        print(f"ERROR: {npz_path} not found. Run 2f_concept_overlap.py first.")
        sys.exit(1)

    # Layer profiles
    lp_path = data_subdir(OVERLAP_DIR) / variant_filename("layer_profiles", ".npz")
    if lp_path.exists():
        f = np.load(lp_path)
        data["layer_profiles"] = f["layer_profiles"]

    # Baseline overlap CSVs
    for key, fname in [("baseline_0", "baseline_overlap"),
                       ("baseline_18", "sysprompt_baseline_overlap")]:
        csv_path = data_subdir(OVERLAP_DIR) / variant_filename(fname, ".csv")
        if csv_path.exists():
            rows = []
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append({
                        "dim_id": int(row["dim_id"]),
                        "name": row["name"],
                        "category": row["category"],
                        "mean_abs_cosine": float(row["mean_abs_cosine"]),
                        "ci_lower": float(row["ci_lower"]),
                        "ci_upper": float(row["ci_upper"]),
                    })
            data[key] = rows

    # Count prompts per dimension from raw activations if available
    act_dir = Path(str(config.RESULTS.concept_activations_contrasts))
    prompt_counts = {}
    if act_dir.exists():
        for dim_id in data["dim_ids"]:
            for dname in sorted(os.listdir(act_dir)):
                if dname.startswith(f"{dim_id}_"):
                    act_file = act_dir / dname / "concept_activations.npz"
                    if act_file.exists():
                        af = np.load(act_file)
                        n_total = af["activations"].shape[0]
                        labels = af["labels"]
                        n_human = int((labels == 1).sum())
                        n_ai = int((labels == 0).sum())
                        prompt_counts[dim_id] = (n_total, n_human, n_ai)
                    break
    data["prompt_counts"] = prompt_counts

    return data


# ============================================================================
# UTILITIES
# ============================================================================

def fig_to_base64_png(fig, dpi=150):
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def fig_to_png_bytes(fig, dpi=150):
    """Convert a matplotlib Figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def dim_sort_key(dim_id):
    """Sort key: by category order, then by dim_id within category."""
    cat = CATEGORY_FOR_DIM.get(dim_id, "Other")
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99
    return (cat_idx, dim_id)


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def make_overlap_heatmap(data):
    """Full pairwise overlap heatmap."""
    overlap = data["overlap"]
    dim_ids = data["dim_ids"]
    names = [DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]
    categories = [CATEGORY_FOR_DIM.get(d, "Other") for d in dim_ids]
    n = len(dim_ids)

    # Sort by category
    sort_order = sorted(range(n), key=lambda i: dim_sort_key(dim_ids[i]))
    sorted_overlap = overlap[np.ix_(sort_order, sort_order)]
    sorted_names = [names[i] for i in sort_order]
    sorted_cats = [categories[i] for i in sort_order]

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sorted_overlap, cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = sorted_overlap[i, j]
            color = "white" if val > 0.6 or val < 0.1 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)

    # Axis labels colored by category
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sorted_names, fontsize=8)

    for i, cat in enumerate(sorted_cats):
        color = CATEGORY_COLORS.get(cat, "#333")
        ax.get_xticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_color(color)

    # Category separators
    prev_cat = sorted_cats[0]
    for i in range(1, n):
        if sorted_cats[i] != prev_cat:
            ax.axhline(i - 0.5, color="black", linewidth=1.5)
            ax.axvline(i - 0.5, color="black", linewidth=1.5)
            prev_cat = sorted_cats[i]

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean |cosine similarity|", fontsize=10)
    ax.set_title("Pairwise Contrast Direction Overlap\n(mean |cos| across layers 6-40)",
                 fontsize=12)
    fig.tight_layout()
    return fig


def make_baseline_bar_chart(baseline_results, title, baseline_name):
    """Bar chart of per-dimension overlap with a baseline."""
    sorted_results = sorted(baseline_results, key=lambda r: dim_sort_key(r["dim_id"]))

    names = [r["name"] for r in sorted_results]
    values = [r["mean_abs_cosine"] for r in sorted_results]
    ci_lo = [r["ci_lower"] for r in sorted_results]
    ci_hi = [r["ci_upper"] for r in sorted_results]
    cats = [r["category"] for r in sorted_results]
    colors = [CATEGORY_COLORS.get(c, "#999") for c in cats]
    errors = [[v - lo for v, lo in zip(values, ci_lo)],
              [hi - v for v, hi in zip(values, ci_hi)]]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(names))
    ax.bar(x, values, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="#333",
                elinewidth=1, capsize=2)

    ax.axhline(CHANCE_LEVEL, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({CHANCE_LEVEL:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    for i, cat in enumerate(cats):
        ax.get_xticklabels()[i].set_color(CATEGORY_COLORS.get(cat, "#333"))

    ax.set_ylabel("Mean |cosine similarity|", fontsize=10)
    ax.set_title(f"Overlap with {baseline_name}\n(mean |cos| across layers 6-40, "
                 f"95% bootstrap CI)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def make_baseline_comparison(data):
    """Side-by-side comparison: overlap with dim 0 vs dim 18."""
    if "baseline_0" not in data or "baseline_18" not in data:
        return None

    bl0 = {r["dim_id"]: r for r in data["baseline_0"]}
    bl18 = {r["dim_id"]: r for r in data["baseline_18"]}

    common = sorted(set(bl0.keys()) & set(bl18.keys()), key=dim_sort_key)

    names = [DIM_NAMES.get(d, f"dim_{d}") for d in common]
    vals_0 = [bl0[d]["mean_abs_cosine"] for d in common]
    vals_18 = [bl18[d]["mean_abs_cosine"] for d in common]
    cats = [CATEGORY_FOR_DIM.get(d, "Other") for d in common]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(common))
    w = 0.35

    ax.bar(x - w/2, vals_0, w, color="#9E9E9E", alpha=0.85,
           label="vs Baseline (dim 0)", edgecolor="white")
    ax.bar(x + w/2, vals_18, w, color="#00BCD4", alpha=0.85,
           label="vs SysPrompt (dim 18)", edgecolor="white")

    ax.axhline(CHANCE_LEVEL, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({CHANCE_LEVEL:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    for i, cat in enumerate(cats):
        ax.get_xticklabels()[i].set_color(CATEGORY_COLORS.get(cat, "#333"))

    ax.set_ylabel("Mean |cosine similarity|", fontsize=10)
    ax.set_title("Baseline Comparison: Entity Baseline (dim 0) vs SysPrompt (dim 18)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def make_layer_profiles_vs_ref(data, ref_dim_id, ref_name):
    """Layer-resolved cosine profiles for every dimension vs a reference dimension.

    Returns a figure with a grid of subplots (4 columns), one per dimension
    (excluding the reference dim itself).
    """
    if "layer_profiles" not in data:
        return None

    lp = data["layer_profiles"]
    dim_ids = data["dim_ids"]

    if ref_dim_id not in dim_ids:
        return None

    ref_idx = dim_ids.index(ref_dim_id)
    remaining = [d for d in dim_ids if d != ref_dim_id]

    # Sort: Attention (17) first, Baseline (0) last, rest by dim_sort_key
    def _layer_profile_sort(d):
        if d == 17:  # Attention first
            return (0,)
        if d == 0:   # Baseline last
            return (2,)
        return (1, dim_sort_key(d))

    remaining.sort(key=_layer_profile_sort)
    target_dims = [(d, dim_ids.index(d)) for d in remaining]

    if not target_dims:
        return None

    n_pairs = len(target_dims)
    n_cols = 4
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3 * n_rows),
                             squeeze=False)

    layers = np.arange(lp.shape[2])
    for idx, (dim_id, dim_idx) in enumerate(target_dims):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        cosines = lp[dim_idx, ref_idx, :]
        cat = CATEGORY_FOR_DIM.get(dim_id, "Other")
        color = CATEGORY_COLORS.get(cat, "#2196F3")

        ax.plot(layers, cosines, color=color, linewidth=1.5)
        ax.axhline(0, color="#ccc", linewidth=0.5)
        ax.axvline(data["layer_range_start"], color="red", linewidth=0.5,
                   linestyle="--", alpha=0.5)
        ax.fill_between(layers, cosines, 0, alpha=0.15, color=color)
        ax.set_title(f"{DIM_NAMES.get(dim_id, dim_id)}", fontsize=9)
        ax.set_xlabel("Layer", fontsize=7)
        ax.set_ylabel("Cosine sim.", fontsize=7)
        ax.set_ylim(-1, 1)
        ax.tick_params(labelsize=6)

    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Layer-Resolved Cosine Similarity vs {ref_name}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def make_category_summary(data):
    """Mean overlap within vs between categories."""
    overlap = data["overlap"]
    dim_ids = data["dim_ids"]
    n = len(dim_ids)
    cats = [CATEGORY_FOR_DIM.get(d, "Other") for d in dim_ids]

    unique_cats = [c for c in CATEGORY_ORDER if c != "Baseline" and c in set(cats)]

    within = {}
    between = {}

    for cat in unique_cats:
        cat_indices = [i for i in range(n) if cats[i] == cat]
        vals = []
        for i in cat_indices:
            for j in cat_indices:
                if i < j:
                    vals.append(overlap[i, j])
        within[cat] = np.mean(vals) if vals else 0.0

        other_indices = [i for i in range(n)
                         if cats[i] != cat and cats[i] != "Baseline"]
        vals = []
        for i in cat_indices:
            for j in other_indices:
                vals.append(overlap[i, j])
        between[cat] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(unique_cats))
    w = 0.35

    within_vals = [within[c] for c in unique_cats]
    between_vals = [between[c] for c in unique_cats]
    cat_colors = [CATEGORY_COLORS.get(c, "#999") for c in unique_cats]

    ax.bar(x - w/2, within_vals, w, color=cat_colors, alpha=0.85,
           label="Within category", edgecolor="white")
    ax.bar(x + w/2, between_vals, w, color=cat_colors, alpha=0.4,
           label="Between categories", edgecolor="white", hatch="//")

    ax.axhline(CHANCE_LEVEL, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({CHANCE_LEVEL:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(unique_cats, fontsize=10)
    for i, cat in enumerate(unique_cats):
        ax.get_xticklabels()[i].set_color(CATEGORY_COLORS.get(cat, "#333"))

    ax.set_ylabel("Mean |cosine similarity|", fontsize=10)
    ax.set_title("Within vs Between Category Contrast Overlap", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


# ============================================================================
# NARRATIVE GENERATION
# ============================================================================

def generate_methods_section(data):
    """Generate the methods/pipeline overview section."""
    n_dims = len(data["dim_ids"])
    layer_start = data["layer_range_start"]
    layer_end = data["layer_range_end"] - 1
    n_boot = data["n_bootstrap"]
    prompt_counts = data.get("prompt_counts", {})

    # Compute typical prompt count
    typical_n = 80
    if prompt_counts:
        counts = [v[0] for v in prompt_counts.values()]
        typical_n = max(set(counts), key=counts.count) if counts else 80

    html = []
    html.append('<h2>Methods</h2>')
    html.append('<div class="methods">')
    html.append('<h3>Pipeline Overview</h3>')
    html.append('<p>This analysis measures how much the <strong>contrast directions</strong> '
                'of different concept dimensions overlap in LLaMA-2-13B\'s representation space. '
                'A contrast direction is the vector difference between mean activations for '
                'human-framed prompts and AI-framed prompts within a single concept dimension.</p>')

    html.append('<h3>Data</h3>')
    html.append('<ul>')
    html.append(f'<li><strong>{n_dims} contrast dimensions</strong>, each with human-framed and '
                f'AI-framed prompts (typically {typical_n} total: {typical_n//2} human + '
                f'{typical_n//2} AI per dimension)</li>')
    html.append(f'<li><strong>41 layers</strong> (embedding + 40 transformer layers), each producing '
                f'a 5120-dimensional activation vector per prompt</li>')
    html.append(f'<li><strong>Layers {layer_start}-{layer_end}</strong> used for summary statistics. '
                f'Layers 0-{layer_start-1} are excluded because early layers have near-zero-norm '
                f'vectors and prompt-format confounds dominate.</li>')
    html.append('</ul>')

    html.append('<h3>Computation</h3>')
    html.append('<ol>')
    html.append('<li><strong>Contrast vector</strong>: For each dimension d and layer L, compute '
                'contrast_d[L] = mean(human_activations[L]) - mean(AI_activations[L]). This gives '
                'the direction that separates human-framed from AI-framed prompts.</li>')
    html.append('<li><strong>Pairwise overlap</strong>: For each pair (i, j) and each layer L, '
                'compute cos(contrast_i[L], contrast_j[L]). Take |cos| and average across '
                f'layers {layer_start}-{layer_end}. This gives one overlap score per pair.</li>')
    html.append(f'<li><strong>Bootstrap CIs</strong>: {n_boot} iterations. Each iteration resamples '
                'prompts with replacement (separately for human and AI prompts within each dimension), '
                'recomputes contrast vectors, then recomputes all pairwise |cos|. The 2.5th and 97.5th '
                'percentiles give 95% confidence intervals.</li>')
    html.append('</ol>')

    html.append(f'<h3>Chance Level</h3>')
    html.append(f'<p>For random unit vectors in 5120-dimensional space, '
                f'E[|cos|] = sqrt(2/(pi*d)) = <strong>{CHANCE_LEVEL:.4f}</strong>. '
                f'Any overlap substantially above this indicates non-random alignment of '
                f'contrast directions.</p>')
    html.append('</div>')

    return '\n'.join(html)


def generate_dimension_table(data):
    """Generate the dimension reference table."""
    dim_ids = data["dim_ids"]
    prompt_counts = data.get("prompt_counts", {})

    html = []
    html.append('<h3>Dimension Reference</h3>')
    html.append('<table class="dim-table">')
    html.append('<tr><th>ID</th><th>Name</th><th>Category</th>'
                '<th>Description</th><th>N prompts</th></tr>')

    for d in sorted(dim_ids, key=dim_sort_key):
        name = DIM_NAMES.get(d, f"dim_{d}")
        cat = CATEGORY_FOR_DIM.get(d, "Other")
        desc = DIM_DESCRIPTIONS.get(d, "")
        color = CATEGORY_COLORS.get(cat, "#333")
        if d in prompt_counts:
            n_total, n_h, n_a = prompt_counts[d]
            count_str = f"{n_total} ({n_h}H + {n_a}A)"
        else:
            count_str = "-"
        html.append(f'<tr><td>{d}</td>'
                     f'<td style="color:{color}">{name}</td>'
                     f'<td>{cat}</td>'
                     f'<td style="text-align:left">{desc}</td>'
                     f'<td>{count_str}</td></tr>')

    html.append('</table>')
    return '\n'.join(html)


def generate_heatmap_narrative(data):
    """Generate narrative for the pairwise overlap heatmap."""
    overlap = data["overlap"]
    dim_ids = data["dim_ids"]
    n = len(dim_ids)

    # Find highest off-diagonal pairs
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((dim_ids[i], dim_ids[j], overlap[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    top5 = pairs[:5]
    bottom5 = pairs[-5:]

    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append('<p>Each cell shows the mean |cosine similarity| between two dimensions\' '
                'contrast directions, averaged across layers 6-40. Values near 1.0 mean the '
                'two dimensions\' human-vs-AI directions point the same way. Values near 0 '
                'mean orthogonal (independent) directions.</p>')

    html.append('<h3>How to Read It</h3>')
    html.append('<ul>')
    html.append('<li><strong>Diagonal</strong>: Always 1.0 (a dimension perfectly overlaps itself)</li>')
    html.append('<li><strong>Blocks of high overlap</strong> within categories suggest shared '
                'underlying representational structure</li>')
    html.append('<li><strong>High overlap with Baseline (dim 0)</strong> suggests the contrast '
                'direction is partially explained by the general entity (human vs AI) direction</li>')
    html.append('<li><strong>Category separators</strong> (black lines) group dimensions by '
                'Mental, Physical, Pragmatic, etc.</li>')
    html.append('</ul>')

    html.append('<h3>Key Findings</h3>')
    html.append(f'<p><strong>Highest-overlap pairs</strong> (excluding diagonal):</p>')
    html.append('<ol>')
    for d_i, d_j, val in top5:
        html.append(f'<li>{DIM_NAMES.get(d_i, d_i)} x {DIM_NAMES.get(d_j, d_j)}: '
                     f'<strong>{val:.3f}</strong></li>')
    html.append('</ol>')

    html.append(f'<p><strong>Lowest-overlap pairs</strong>:</p>')
    html.append('<ol>')
    for d_i, d_j, val in bottom5:
        html.append(f'<li>{DIM_NAMES.get(d_i, d_i)} x {DIM_NAMES.get(d_j, d_j)}: '
                     f'<strong>{val:.3f}</strong></li>')
    html.append('</ol>')

    # Mean off-diagonal
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            off_diag.append(overlap[i, j])
    mean_off = np.mean(off_diag)
    html.append(f'<p>Mean off-diagonal overlap: <strong>{mean_off:.3f}</strong> '
                f'(chance: {CHANCE_LEVEL:.3f})</p>')
    html.append('</div>')

    return '\n'.join(html)


def generate_baseline_narrative(data, baseline_key, baseline_name, baseline_id):
    """Generate narrative for a baseline overlap section."""
    if baseline_key not in data:
        return ""

    results = sorted(data[baseline_key], key=lambda r: r["mean_abs_cosine"], reverse=True)

    html = []
    html.append('<div class="narrative">')
    html.append(f'<h3>What This Shows</h3>')
    html.append(f'<p>How much each dimension\'s contrast direction aligns with {baseline_name} '
                f'(dim {baseline_id}). A dimension with overlap 0.6 means 60% of its contrast '
                f'direction is shared with the entity baseline; the residual 40% is '
                f'dimension-specific.</p>')

    if baseline_id == 0:
        html.append(f'<h3>What Dim 0 Is</h3>')
        html.append(f'<p>Dim 0 uses explicit prompts like "Think about what it means to be a '
                     f'human" vs "Think about what it means to be an AI." This captures the '
                     f'purest entity-level direction in the model\'s representation.</p>')

    html.append(f'<h3>Bootstrap CIs</h3>')
    html.append(f'<p>{data["n_bootstrap"]} bootstrap iterations. Each iteration resamples '
                f'prompts with replacement for <em>both</em> the target dimension and the '
                f'baseline, recomputes contrast vectors, then computes cosine. The 2.5th and '
                f'97.5th percentiles give the 95% CI.</p>')

    html.append(f'<h3>Interpretation</h3>')
    html.append('<ul>')
    html.append(f'<li><strong>High overlap</strong> (well above {CHANCE_LEVEL:.3f}): This '
                f'dimension\'s human-vs-AI contrast largely recapitulates the general entity direction</li>')
    html.append(f'<li><strong>Low overlap</strong> (near chance): This dimension captures a '
                f'genuinely distinct aspect of how the model distinguishes humans from AIs</li>')
    html.append(f'<li><strong>CI excluding chance</strong>: Statistically reliable overlap</li>')
    html.append('</ul>')

    # Summary stats
    above_thresh = [r for r in results if r["ci_lower"] > CHANCE_LEVEL]
    html.append(f'<p><strong>{len(above_thresh)}/{len(results)}</strong> dimensions have overlap '
                f'with {baseline_name} significantly above chance (95% CI lower bound > {CHANCE_LEVEL:.3f}).</p>')

    if results:
        html.append(f'<p>Highest overlap: <strong>{results[0]["name"]}</strong> '
                     f'({results[0]["mean_abs_cosine"]:.3f}). '
                     f'Lowest overlap: <strong>{results[-1]["name"]}</strong> '
                     f'({results[-1]["mean_abs_cosine"]:.3f}).</p>')

    html.append('</div>')
    return '\n'.join(html)


def generate_baseline_comparison_narrative():
    """Generate narrative for the baseline comparison section."""
    html = []
    html.append('<div class="narrative">')
    html.append('<h3>Why Two Baselines?</h3>')
    html.append('<p>Dim 0 uses explicit leading prompts ("Think about what it means to be a '
                'human/AI"). Dim 18 uses system-prompt framing ("You are speaking to a human/AI"). '
                'These are both entity-level directions, but derived from very different prompt '
                'strategies.</p>')
    html.append('<h3>Interpretation</h3>')
    html.append('<ul>')
    html.append('<li><strong>Similar patterns</strong>: The entity direction is robust to '
                'prompt framing. Both methods tap the same underlying representation.</li>')
    html.append('<li><strong>Different patterns</strong>: The specific wording of the baseline '
                'matters. Some dimensions may align more with one framing than the other, '
                'suggesting the model encodes entity identity differently depending on context.</li>')
    html.append('</ul>')
    html.append('</div>')
    return '\n'.join(html)


def generate_layer_profiles_vs_ref_narrative(data, ref_dim_id, ref_name):
    """Generate narrative for per-layer profiles vs a reference dimension."""
    layer_start = data["layer_range_start"]

    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append(f'<p>Signed cosine similarity at each of the 41 layers between every remaining '
                f'dimension\'s contrast direction and <strong>{ref_name} (dim {ref_dim_id})</strong>. '
                f'Positive means the two contrast vectors point the same way at that layer; '
                f'negative means opposite directions.</p>')

    html.append('<h3>Red Dashed Line</h3>')
    html.append(f'<p>Marks layer {layer_start}, the start of the restricted range used for '
                f'summary statistics. Layers 0-{layer_start-1} are excluded from mean |cos| '
                f'calculations due to embedding / prompt-format confounds.</p>')

    html.append('<h3>Interpretation</h3>')
    html.append('<ul>')
    html.append(f'<li><strong>Consistently high positive cosine</strong> across mid-to-late layers: '
                f'this dimension\'s human-vs-AI contrast largely recapitulates the {ref_name} direction.</li>')
    html.append(f'<li><strong>Near-zero across layers</strong>: the dimension captures a genuinely '
                f'distinct aspect of human-vs-AI representation, orthogonal to {ref_name}.</li>')
    html.append(f'<li><strong>Emerging late</strong> (low early, high late): the overlap develops as '
                f'the model builds higher-level representations.</li>')
    html.append('</ul>')
    html.append('</div>')
    return '\n'.join(html)


def generate_category_summary_narrative(data):
    """Generate narrative for the category-level summary."""
    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append('<p>Average pairwise |cosine| within each category (solid bars) vs between '
                'that category and all other non-baseline categories (hatched bars).</p>')

    html.append('<h3>Categories</h3>')
    html.append('<ul>')
    html.append('<li><strong>Mental</strong> (dims 1-9, 16-17): Core mental capacities — '
                'phenomenology, emotions, agency, intentions, prediction, cognitive, social, '
                'embodiment, roles, mind, attention</li>')
    html.append('<li><strong>Pragmatic</strong> (dims 11-13): Formality, expertise, helpfulness</li>')
    html.append('<li><strong>Bio Ctrl</strong> (dim 14): Biological processes (control dimension)</li>')
    html.append('<li><strong>Shapes</strong> (dim 15): Geometric shapes (negative control)</li>')
    html.append('<li><strong>SysPrompt</strong> (dim 18): System prompt entity framing</li>')
    html.append('</ul>')
    html.append('<p>Note: Baseline (dim 0) is excluded from the within/between calculation. '
                'Categories with a single dimension (Bio Ctrl, Shapes, SysPrompt) have no '
                'within-category pairs, so their "within" bar is 0.</p>')

    html.append('<h3>Interpretation</h3>')
    html.append('<ul>')
    html.append('<li><strong>Within > Between</strong>: The category grouping captures real '
                'structure in how the model organizes concept contrasts — dimensions in the same '
                'category tend to define human-vs-AI the same way.</li>')
    html.append('<li><strong>Within ~ Between</strong>: The category boundaries are not '
                'meaningful at the representational level.</li>')
    html.append('</ul>')
    html.append('</div>')
    return '\n'.join(html)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_html(figures_b64, data):
    """Generate HTML report with detailed narrative."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_dims = len(data["dim_ids"])
    layer_start = data["layer_range_start"]
    layer_end = data["layer_range_end"] - 1
    n_boot = data["n_bootstrap"]

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Contrast Direction Overlap Analysis</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "         max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }",
        "  h1 { border-bottom: 2px solid #2196F3; padding-bottom: 8px; }",
        "  h2 { color: #1565C0; margin-top: 2em; }",
        "  h3 { color: #424242; margin-top: 1.2em; }",
        "  .meta { color: #666; font-size: 0.9em; margin-bottom: 2em; }",
        "  .figure { margin: 1.5em 0; text-align: center; }",
        "  .figure img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }",
        "  .caption { font-size: 0.85em; color: #666; margin-top: 0.5em; }",
        "  table { border-collapse: collapse; margin: 1em 0; font-size: 0.85em; }",
        "  th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }",
        "  th { background: #f5f5f5; text-align: center; }",
        "  .dim-table td:nth-child(4) { text-align: left; max-width: 400px; }",
        "  .note { background: #E3F2FD; border-left: 4px solid #2196F3;",
        "          padding: 12px 16px; margin: 1em 0; font-size: 0.9em; }",
        "  .methods { background: #FAFAFA; border: 1px solid #E0E0E0;",
        "             padding: 16px 20px; margin: 1em 0; border-radius: 4px; }",
        "  .methods h3 { color: #1565C0; margin-top: 0.8em; }",
        "  .methods h3:first-child { margin-top: 0; }",
        "  .narrative { background: #FFF8E1; border-left: 4px solid #FFC107;",
        "               padding: 12px 16px; margin: 1em 0; font-size: 0.9em; }",
        "  .narrative h3 { color: #F57F17; margin-top: 0.8em; }",
        "  .narrative h3:first-child { margin-top: 0; }",
        "  code { background: #f5f5f5; padding: 1px 4px; border-radius: 2px; font-size: 0.9em; }",
        "</style></head><body>",
        "",
        "<h1>Contrast Direction Overlap Analysis</h1>",
        f'<p class="meta">Generated: {timestamp} | '
        f'{n_dims} contrast dimensions | Layers {layer_start}-{layer_end} | '
        f'{n_boot} bootstrap iterations</p>',
        "",
        '<div class="note">',
        "<strong>Summary:</strong> For each pair of contrast dimensions, "
        "how much does the human-vs-AI direction for one concept overlap with the "
        "human-vs-AI direction for another? High overlap means the model uses similar "
        "representational directions for both contrasts. Overlap with baseline (dim 0) "
        "indicates how much of each concept's contrast is just the general entity direction.",
        "</div>",
    ]

    # Note about excluded dimensions
    excluded = data.get("excluded_dims", [])
    if excluded:
        excl_names = [DIM_NAMES.get(d, f"dim_{d}") for d in excluded]
        excl_str = ", ".join(f"{d} ({n})" for d, n in zip(excluded, excl_names))
        lines.append(f'<div class="note"><strong>Excluded dimensions:</strong> {excl_str}. '
                     f'These were removed before all analyses.</div>')

    # Methods section
    lines.append(generate_methods_section(data))

    # Dimension table
    lines.append(generate_dimension_table(data))

    # Figure 1: Heatmap
    if "heatmap" in figures_b64:
        lines.extend([
            "<h2>1. Pairwise Overlap Matrix</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["heatmap"]}" alt="Overlap heatmap">',
            '<p class="caption">Mean |cosine similarity| between each pair of contrast '
            'vectors, averaged across layers 6-40. Dimensions grouped by category.</p>',
            "</div>",
        ])
        lines.append(generate_heatmap_narrative(data))

    # Figure 2: Baseline bar chart
    if "baseline_0" in figures_b64:
        lines.extend([
            "<h2>2. Overlap with Entity Baseline (Dim 0)</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["baseline_0"]}" alt="Baseline overlap">',
            '<p class="caption">How much of each dimension\'s contrast direction aligns '
            'with the general entity baseline direction. Error bars: 95% bootstrap CI. '
            f'Red dashed line: expected |cosine| for random vectors in 5120-d space '
            f'({CHANCE_LEVEL:.4f}).</p>',
            "</div>",
        ])
        lines.append(generate_baseline_narrative(data, "baseline_0", "Entity Baseline", 0))

    # Baseline table
    if "baseline_0" in data:
        lines.extend([
            "<h3>Baseline Overlap Table</h3>",
            "<table>",
            "<tr><th>Dimension</th><th>Category</th><th>|cos| with Baseline</th>"
            "<th>95% CI</th></tr>",
        ])
        sorted_bl = sorted(data["baseline_0"], key=lambda r: dim_sort_key(r["dim_id"]))
        for r in sorted_bl:
            color = CATEGORY_COLORS.get(r["category"], "#333")
            lines.append(
                f'<tr><td style="text-align:left; color:{color}">{r["name"]}</td>'
                f'<td style="text-align:left">{r["category"]}</td>'
                f'<td>{r["mean_abs_cosine"]:.4f}</td>'
                f'<td>[{r["ci_lower"]:.4f}, {r["ci_upper"]:.4f}]</td></tr>'
            )
        lines.append("</table>")

    # Figure 3: SysPrompt bar chart (dim 18, excluding baseline dim 0)
    if "baseline_18_nobaseline" in figures_b64:
        lines.extend([
            "<h2>3. Overlap with SysPrompt (Dim 18)</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["baseline_18_nobaseline"]}" '
            f'alt="SysPrompt overlap">',
            '<p class="caption">How much of each dimension\'s contrast direction aligns '
            'with the system prompt direction (dim 18). Baseline (dim 0) excluded. '
            f'Error bars: 95% bootstrap CI. Red dashed line: chance ({CHANCE_LEVEL:.4f}).</p>',
            "</div>",
        ])
        lines.append(generate_baseline_narrative(data, "baseline_18", "SysPrompt", 18))

    # Figure 4: Baseline comparison
    if "baseline_comparison" in figures_b64:
        lines.extend([
            "<h2>4. Baseline Comparison: Dim 0 vs Dim 18 (SysPrompt)</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["baseline_comparison"]}" '
            f'alt="Baseline comparison">',
            '<p class="caption">Side-by-side overlap with dim 0 (entity baseline) vs '
            'dim 18 (system prompt baseline).</p>',
            "</div>",
        ])
        lines.append(generate_baseline_comparison_narrative())

    # Figure 5: Layer profiles vs Baseline (dim 0)
    if "layer_profiles_vs_0" in figures_b64:
        lines.extend([
            "<h2>5. Layer-Resolved Overlap vs Entity Baseline (Dim 0)</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["layer_profiles_vs_0"]}" '
            f'alt="Layer profiles vs baseline">',
            f'<p class="caption">Per-layer cosine similarity (signed) between each dimension\'s '
            f'contrast direction and the Entity Baseline (dim 0). Red dashed line marks layer {layer_start}.</p>',
            "</div>",
        ])
        lines.append(generate_layer_profiles_vs_ref_narrative(data, 0, "Entity Baseline"))

    # Figure 6: Layer profiles vs SysPrompt (dim 18)
    if "layer_profiles_vs_18" in figures_b64:
        lines.extend([
            "<h2>6. Layer-Resolved Overlap vs SysPrompt (Dim 18)</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["layer_profiles_vs_18"]}" '
            f'alt="Layer profiles vs sysprompt">',
            f'<p class="caption">Per-layer cosine similarity (signed) between each dimension\'s '
            f'contrast direction and the SysPrompt baseline (dim 18). Red dashed line marks layer {layer_start}.</p>',
            "</div>",
        ])
        lines.append(generate_layer_profiles_vs_ref_narrative(data, 18, "SysPrompt"))

    # Figure 7: Category summary
    if "category_summary" in figures_b64:
        lines.extend([
            "<h2>7. Category-Level Summary</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["category_summary"]}" '
            f'alt="Category summary">',
            '<p class="caption">Mean pairwise |cosine| within each category (solid) '
            'vs between categories (hatched).</p>',
            "</div>",
        ])
        lines.append(generate_category_summary_narrative(data))

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def generate_markdown(data):
    """Generate markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_dims = len(data["dim_ids"])
    layer_start = data["layer_range_start"]
    layer_end = data["layer_range_end"] - 1
    n_boot = data["n_bootstrap"]
    prompt_counts = data.get("prompt_counts", {})

    lines = [
        "# Contrast Direction Overlap Analysis",
        "",
        f"Generated: {timestamp} | {n_dims} contrast dimensions | "
        f"Layers {layer_start}-{layer_end} | {n_boot} bootstrap iterations",
        "",
        "## Summary",
        "",
        "For each pair of contrast dimensions, how much does the human-vs-AI "
        "direction for one concept overlap with the human-vs-AI direction for "
        "another? High overlap means the model uses similar representational "
        "directions for both contrasts.",
        "",
        "## Methods",
        "",
        "1. **Contrast vector**: For each dimension d and layer L, compute "
        "contrast_d[L] = mean(human) - mean(AI). This gives the direction "
        "separating human- from AI-framed prompts.",
        f"2. **Pairwise overlap**: For each pair (i, j) and layer L, compute "
        f"|cos(contrast_i[L], contrast_j[L])|, then average across layers "
        f"{layer_start}-{layer_end}.",
        f"3. **Bootstrap**: {n_boot} iterations resampling prompts with replacement.",
        f"4. **Chance level**: E[|cos|] for random 5120-d vectors = {CHANCE_LEVEL:.4f}.",
        "",
        "## Dimension Reference",
        "",
        "| ID | Name | Category | N prompts |",
        "|----|------|----------|-----------|",
    ]
    for d in sorted(data["dim_ids"], key=dim_sort_key):
        name = DIM_NAMES.get(d, f"dim_{d}")
        cat = CATEGORY_FOR_DIM.get(d, "Other")
        if d in prompt_counts:
            n_total, n_h, n_a = prompt_counts[d]
            count_str = f"{n_total} ({n_h}H + {n_a}A)"
        else:
            count_str = "-"
        lines.append(f"| {d} | {name} | {cat} | {count_str} |")
    lines.append("")

    # Figures
    lines.extend([
        "## 1. Pairwise Overlap Matrix",
        "",
        "![Overlap Heatmap](figures/overlap_heatmap.png)",
        "",
    ])

    # Baseline overlap table
    if "baseline_0" in data:
        lines.extend([
            "## 2. Overlap with Entity Baseline (Dim 0)",
            "",
            "![Baseline Overlap](figures/baseline_overlap_dim0.png)",
            "",
            "| Dimension | Category | |cos| with Baseline | 95% CI |",
            "|-----------|----------|---------------------|--------|",
        ])
        sorted_bl = sorted(data["baseline_0"], key=lambda r: dim_sort_key(r["dim_id"]))
        for r in sorted_bl:
            lines.append(
                f"| {r['name']} | {r['category']} | "
                f"{r['mean_abs_cosine']:.4f} | "
                f"[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}] |"
            )
        lines.append("")

    if "baseline_18" in data:
        lines.extend([
            "## 3. Overlap with SysPrompt (Dim 18)",
            "",
            "![SysPrompt Overlap](figures/sysprompt_overlap_dim18.png)",
            "",
        ])

    if "baseline_0" in data and "baseline_18" in data:
        lines.extend([
            "## 4. Baseline Comparison: Dim 0 vs Dim 18",
            "",
            "![Baseline Comparison](figures/baseline_comparison.png)",
            "",
        ])

    # Note about excluded dimensions
    excluded = data.get("excluded_dims", [])
    if excluded:
        excl_names = [DIM_NAMES.get(d, f"dim_{d}") for d in excluded]
        excl_str = ", ".join(f"{d} ({n})" for d, n in zip(excluded, excl_names))
        lines.insert(4, f"**Excluded dimensions:** {excl_str}")
        lines.insert(5, "")

    lines.extend([
        "## 5. Layer-Resolved Overlap vs Entity Baseline (Dim 0)",
        "",
        "![Layer Profiles vs Baseline](figures/layer_profiles_vs_0.png)",
        "",
        "## 6. Layer-Resolved Overlap vs SysPrompt (Dim 18)",
        "",
        "![Layer Profiles vs SysPrompt](figures/layer_profiles_vs_18.png)",
        "",
        "## 7. Category Summary",
        "",
        "![Category Summary](figures/category_summary.png)",
        "",
    ])

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Concept overlap report generator")
    add_variant_argument(parser)
    args = parser.parse_args()

    if args.variant:
        set_variant(args.variant)

    print("Loading concept overlap data...")
    data = load_overlap_data()
    print(f"  {len(data['dim_ids'])} dimensions, "
          f"layers {data['layer_range_start']}-{data['layer_range_end'] - 1}")
    if data.get("prompt_counts"):
        print(f"  Prompt counts loaded for {len(data['prompt_counts'])} dimensions")

    # Output directory (same as data directory)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    figures_b64 = {}
    png_data = {}

    # Figure 1: Overlap heatmap
    print("Generating overlap heatmap...")
    fig = make_overlap_heatmap(data)
    figures_b64["heatmap"] = fig_to_base64_png(fig)
    fig2 = make_overlap_heatmap(data)
    png_data["overlap_heatmap"] = fig_to_png_bytes(fig2)

    # Figure 2: Baseline bar chart (dim 0)
    if "baseline_0" in data:
        print("Generating baseline overlap bar chart (dim 0)...")
        fig = make_baseline_bar_chart(data["baseline_0"],
                                      "Entity Baseline Overlap", "Baseline (dim 0)")
        figures_b64["baseline_0"] = fig_to_base64_png(fig)
        fig2 = make_baseline_bar_chart(data["baseline_0"],
                                       "Entity Baseline Overlap", "Baseline (dim 0)")
        png_data["baseline_overlap_dim0"] = fig_to_png_bytes(fig2)

    # Figure 3: SysPrompt bar chart (dim 18, excluding dim 0)
    if "baseline_18" in data:
        print("Generating sysprompt overlap bar chart (dim 18, excl. baseline)...")
        filtered_18 = [r for r in data["baseline_18"] if r["dim_id"] != 0]
        fig = make_baseline_bar_chart(filtered_18,
                                      "SysPrompt Overlap", "SysPrompt (dim 18)")
        figures_b64["baseline_18_nobaseline"] = fig_to_base64_png(fig)
        fig2 = make_baseline_bar_chart(filtered_18,
                                       "SysPrompt Overlap", "SysPrompt (dim 18)")
        png_data["sysprompt_overlap_dim18"] = fig_to_png_bytes(fig2)

    # Figure 4: Baseline comparison (dim 0 vs dim 18)
    fig = make_baseline_comparison(data)
    if fig is not None:
        print("Generating baseline comparison (dim 0 vs dim 18)...")
        figures_b64["baseline_comparison"] = fig_to_base64_png(fig)
        fig2 = make_baseline_comparison(data)
        png_data["baseline_comparison"] = fig_to_png_bytes(fig2)

    # Figure 5: Layer profiles vs Baseline (dim 0)
    if "layer_profiles" in data:
        print("Generating layer profiles vs Baseline (dim 0)...")
        fig = make_layer_profiles_vs_ref(data, 0, "Entity Baseline (dim 0)")
        if fig is not None:
            figures_b64["layer_profiles_vs_0"] = fig_to_base64_png(fig)
            fig2 = make_layer_profiles_vs_ref(data, 0, "Entity Baseline (dim 0)")
            png_data["layer_profiles_vs_0"] = fig_to_png_bytes(fig2)

    # Figure 6: Layer profiles vs SysPrompt (dim 18)
    if "layer_profiles" in data:
        print("Generating layer profiles vs SysPrompt (dim 18)...")
        fig = make_layer_profiles_vs_ref(data, 18, "SysPrompt (dim 18)")
        if fig is not None:
            figures_b64["layer_profiles_vs_18"] = fig_to_base64_png(fig)
            fig2 = make_layer_profiles_vs_ref(data, 18, "SysPrompt (dim 18)")
            png_data["layer_profiles_vs_18"] = fig_to_png_bytes(fig2)

    # Figure 7: Category summary
    print("Generating category summary...")
    fig = make_category_summary(data)
    figures_b64["category_summary"] = fig_to_base64_png(fig)
    fig2 = make_category_summary(data)
    png_data["category_summary"] = fig_to_png_bytes(fig2)

    # Save figure PNGs to figures/ subdir (variant suffix on dir name)
    fig_dir = OUTPUT_DIR / f"figures{get_variant_suffix()}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name, data_bytes in png_data.items():
        path = fig_dir / f"{name}.png"
        with open(path, "wb") as f:
            f.write(data_bytes)
        print(f"  Saved {path}")

    # Generate HTML
    print("Generating HTML report...")
    html = generate_html(figures_b64, data)
    html_path = OUTPUT_DIR / variant_filename("concept_overlap_report", ".html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  {html_path}")

    # Generate MD
    print("Generating markdown report...")
    md = generate_markdown(data)
    md_path = OUTPUT_DIR / variant_filename("concept_overlap_report", ".md")
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
