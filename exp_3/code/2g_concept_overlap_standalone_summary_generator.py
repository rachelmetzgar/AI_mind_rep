#!/usr/bin/env python3
"""
Experiment 3, Phase 2g: Standalone Concept Overlap Report Generator

Reads output from 2g_concept_overlap_standalone.py and produces HTML/MD report
with matplotlib figures and detailed narrative.

This parallels 2f_concept_overlap_report.py but for standalone concepts rather
than contrast directions. Key difference: standalone overlap measures whether two
concepts activate similar patterns (no human-vs-AI framing), while contrast
overlap measures whether two dimensions define the human-vs-AI direction the
same way.

Figures:
    1. Full pairwise overlap heatmap (n_dims x n_dims)
    2. Entity reference bar chart (overlap with dim 16=human and dim 17=ai)
    3. Human vs AI reference comparison (side-by-side)
    4. Layer-resolved cosine profiles for selected pairs
    5. Category-level summary (within vs between category overlap)

Output goes to exp_3/results/{model}/concept_overlap/standalone/ alongside data.

Usage:
    python 2g_concept_overlap_standalone_report.py

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

from config import config

# ============================================================================
# CONFIG
# ============================================================================

OVERLAP_DIR = Path(str(config.RESULTS.alignment)) / "concept_overlap" / "standalone"
OUTPUT_DIR = OVERLAP_DIR  # Report lives alongside data

DIM_NAMES = {
    1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Human", 17: "AI", 18: "Attention",
    20: "SysPrompt (talkto human)", 21: "SysPrompt (talkto AI)",
    22: "SysPrompt (bare human)", 23: "SysPrompt (bare AI)",
}

DIM_DESCRIPTIONS = {
    1: "Phenomenological experience prompts (no entity framing)",
    2: "Emotional states and affect prompts",
    3: "Agency and autonomous action prompts",
    4: "Intentions, goals, and purposes prompts",
    5: "Prediction and anticipation prompts",
    6: "Cognitive processing and reasoning prompts",
    7: "Social cognition and interaction prompts",
    8: "Embodiment and physical form prompts",
    9: "Social roles and identity prompts",
    10: "Animacy and liveliness prompts",
    11: "Formality of interaction prompts",
    12: "Expertise and knowledge depth prompts",
    13: "Helpfulness and cooperation prompts",
    14: "Biological processes (negative control)",
    15: "Geometric shapes (negative control)",
    16: "Standalone 'human' concept (entity reference)",
    17: "Standalone 'AI' concept (entity reference)",
    18: "Attention and focus prompts",
    20: "System prompt: 'You are talking to a human'",
    21: "System prompt: 'You are talking to an AI'",
    22: "System prompt: bare 'human' mention",
    23: "System prompt: bare 'AI' mention",
}

CATEGORY_FOR_DIM = {
    1: "Mental", 2: "Mental", 3: "Mental",
    4: "Mental", 5: "Mental", 6: "Mental", 7: "Mental",
    8: "Physical", 9: "Physical", 10: "Physical", 11: "Pragmatic",
    12: "Pragmatic", 13: "Pragmatic", 14: "Bio Ctrl", 15: "Shapes",
    16: "Entity", 17: "Entity", 18: "Mental",
    20: "SysPrompt", 21: "SysPrompt", 22: "SysPrompt", 23: "SysPrompt",
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
    "Pragmatic": "#FF9800",
    "Entity": "#9C27B0",
    "Bio Ctrl": "#795548",
    "Shapes": "#E91E63",
    "SysPrompt": "#00BCD4",
    "Other": "#999",
}

CATEGORY_ORDER = ["Mental", "Physical", "Pragmatic", "Entity",
                  "Bio Ctrl", "Shapes", "SysPrompt"]

# High-interest pairs for layer-resolved profiles
LAYER_PROFILE_PAIRS = [
    (16, 1, "Human vs Phenomenology"),
    (17, 1, "AI vs Phenomenology"),
    (16, 17, "Human vs AI"),
    (16, 15, "Human vs Shapes"),
    (1, 2, "Phenomenology vs Emotions"),
    (1, 8, "Phenomenology vs Embodiment"),
    (16, 14, "Human vs Biological"),
    (20, 22, "SysPrompt talkto vs bare (human)"),
]

# Expected |cosine| for random unit vectors in 5120-d space
CHANCE_LEVEL = np.sqrt(2.0 / (np.pi * 5120))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_overlap_data():
    """Load all standalone overlap analysis outputs (raw + centered)."""
    data = {}

    # Main overlap matrix (raw)
    npz_path = OVERLAP_DIR / "overlap_matrix.npz"
    if npz_path.exists():
        f = np.load(npz_path, allow_pickle=True)
        data["overlap_raw"] = f["overlap"]
        data["boot_overlap_raw"] = f["boot_overlap"]
        data["dim_ids"] = f["dim_ids"].tolist()
        data["dim_names"] = f["dim_names"].tolist()
        data["dim_categories"] = f["dim_categories"].tolist()
        data["layer_range_start"] = int(f["layer_range_start"])
        data["layer_range_end"] = int(f["layer_range_end"])
        data["n_bootstrap"] = int(f["n_bootstrap"])
    else:
        print(f"ERROR: {npz_path} not found. Run 2g_concept_overlap_standalone.py first.")
        sys.exit(1)

    # Centered overlap matrix (the informative one)
    npz_c_path = OVERLAP_DIR / "overlap_matrix_centered.npz"
    if npz_c_path.exists():
        f = np.load(npz_c_path, allow_pickle=True)
        data["overlap"] = f["overlap"]
        data["boot_overlap"] = f["boot_overlap"]
    else:
        # Fall back to raw if centered not available
        print("WARNING: centered overlap not found, using raw data")
        data["overlap"] = data["overlap_raw"]
        data["boot_overlap"] = data["boot_overlap_raw"]

    # Layer profiles (raw)
    lp_path = OVERLAP_DIR / "layer_profiles.npz"
    if lp_path.exists():
        f = np.load(lp_path)
        data["layer_profiles_raw"] = f["layer_profiles"]

    # Layer profiles (centered)
    lp_c_path = OVERLAP_DIR / "layer_profiles_centered.npz"
    if lp_c_path.exists():
        f = np.load(lp_c_path)
        data["layer_profiles"] = f["layer_profiles"]
    elif "layer_profiles_raw" in data:
        data["layer_profiles"] = data["layer_profiles_raw"]

    # Entity overlap CSV (computed from centered vectors)
    eo_path = OVERLAP_DIR / "entity_overlap.csv"
    if eo_path.exists():
        rows = []
        with open(eo_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "dim_id": int(row["dim_id"]),
                    "name": row["name"],
                    "category": row["category"],
                    "overlap_human_mean": float(row["overlap_human_mean"]),
                    "overlap_human_ci_lo": float(row["overlap_human_ci_lo"]),
                    "overlap_human_ci_hi": float(row["overlap_human_ci_hi"]),
                    "overlap_ai_mean": float(row["overlap_ai_mean"]),
                    "overlap_ai_ci_lo": float(row["overlap_ai_ci_lo"]),
                    "overlap_ai_ci_hi": float(row["overlap_ai_ci_hi"]),
                })
        data["entity_overlap"] = rows

    # Count prompts per dimension from raw activations
    act_dir = Path(str(config.RESULTS.concept_activations_standalone))
    prompt_counts = {}
    if act_dir.exists():
        for dim_id in data["dim_ids"]:
            for dname in sorted(os.listdir(act_dir)):
                if dname.startswith(f"{dim_id}_"):
                    act_file = act_dir / dname / "concept_activations.npz"
                    if act_file.exists():
                        af = np.load(act_file)
                        prompt_counts[dim_id] = int(af["activations"].shape[0])
                    break
    data["prompt_counts"] = prompt_counts

    return data


# ============================================================================
# UTILITIES
# ============================================================================

def fig_to_base64_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def fig_to_png_bytes(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    return data


def dim_sort_key(dim_id):
    cat = CATEGORY_FOR_DIM.get(dim_id, "Other")
    cat_idx = CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99
    return (cat_idx, dim_id)


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def _make_heatmap_from_matrix(overlap, dim_ids, title):
    """Helper: generate a pairwise overlap heatmap from a matrix."""
    names = [DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]
    categories = [CATEGORY_FOR_DIM.get(d, "Other") for d in dim_ids]
    n = len(dim_ids)

    sort_order = sorted(range(n), key=lambda i: dim_sort_key(dim_ids[i]))
    sorted_overlap = overlap[np.ix_(sort_order, sort_order)]
    sorted_names = [names[i] for i in sort_order]
    sorted_cats = [categories[i] for i in sort_order]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sorted_overlap, cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest")

    for i in range(n):
        for j in range(n):
            val = sorted_overlap[i, j]
            color = "white" if val > 0.6 or val < 0.1 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=5, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sorted_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(sorted_names, fontsize=7)

    for i, cat in enumerate(sorted_cats):
        color = CATEGORY_COLORS.get(cat, "#333")
        ax.get_xticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_color(color)

    prev_cat = sorted_cats[0]
    for i in range(1, n):
        if sorted_cats[i] != prev_cat:
            ax.axhline(i - 0.5, color="black", linewidth=1.5)
            ax.axvline(i - 0.5, color="black", linewidth=1.5)
            prev_cat = sorted_cats[i]

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean |cosine similarity|", fontsize=10)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig


def make_overlap_heatmap(data):
    """Full pairwise overlap heatmap (centered vectors)."""
    return _make_heatmap_from_matrix(
        data["overlap"], data["dim_ids"],
        "Standalone Concept Pairwise Overlap (Centered)\n(mean |cos| across layers 6-40)"
    )


def make_overlap_heatmap_raw(data):
    """Full pairwise overlap heatmap (raw vectors, for reference)."""
    if "overlap_raw" not in data:
        return None
    return _make_heatmap_from_matrix(
        data["overlap_raw"], data["dim_ids"],
        "Standalone Concept Pairwise Overlap (Raw)\n(mean |cos| across layers 6-40)"
    )


def make_entity_reference_bar_chart(entity_overlap):
    """Bar chart showing overlap with both human (dim 16) and AI (dim 17) concepts."""
    sorted_results = sorted(entity_overlap, key=lambda r: dim_sort_key(r["dim_id"]))

    names = [r["name"] for r in sorted_results]
    vals_h = [r["overlap_human_mean"] for r in sorted_results]
    vals_a = [r["overlap_ai_mean"] for r in sorted_results]
    cats = [r["category"] for r in sorted_results]

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(names))
    w = 0.35

    # Human reference bars
    err_h = [[r["overlap_human_mean"] - r["overlap_human_ci_lo"] for r in sorted_results],
             [r["overlap_human_ci_hi"] - r["overlap_human_mean"] for r in sorted_results]]
    ax.bar(x - w/2, vals_h, w, color="#9C27B0", alpha=0.85,
           label="vs Human (dim 16)", edgecolor="white")
    ax.errorbar(x - w/2, vals_h, yerr=err_h, fmt="none", ecolor="#333",
                elinewidth=0.8, capsize=2)

    # AI reference bars
    err_a = [[r["overlap_ai_mean"] - r["overlap_ai_ci_lo"] for r in sorted_results],
             [r["overlap_ai_ci_hi"] - r["overlap_ai_mean"] for r in sorted_results]]
    ax.bar(x + w/2, vals_a, w, color="#FF5722", alpha=0.85,
           label="vs AI (dim 17)", edgecolor="white")
    ax.errorbar(x + w/2, vals_a, yerr=err_a, fmt="none", ecolor="#333",
                elinewidth=0.8, capsize=2)

    ax.axhline(CHANCE_LEVEL, color="red", linestyle="--", linewidth=1,
               label=f"Chance ({CHANCE_LEVEL:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    for i, cat in enumerate(cats):
        ax.get_xticklabels()[i].set_color(CATEGORY_COLORS.get(cat, "#333"))

    ax.set_ylabel("Mean |cosine similarity|", fontsize=10)
    ax.set_title("Overlap with Entity Reference Concepts\n"
                 "(standalone Human dim 16 and AI dim 17, 95% bootstrap CI)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def make_entity_comparison(entity_overlap):
    """Scatter plot comparing human vs AI overlap for each dimension."""
    sorted_results = sorted(entity_overlap, key=lambda r: dim_sort_key(r["dim_id"]))

    fig, ax = plt.subplots(figsize=(8, 8))

    for r in sorted_results:
        cat = r["category"]
        color = CATEGORY_COLORS.get(cat, "#999")
        ax.scatter(r["overlap_human_mean"], r["overlap_ai_mean"],
                   c=color, s=60, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.annotate(r["name"], (r["overlap_human_mean"], r["overlap_ai_mean"]),
                    fontsize=6, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    # Diagonal line
    max_val = max(max(r["overlap_human_mean"] for r in sorted_results),
                  max(r["overlap_ai_mean"] for r in sorted_results))
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, linewidth=1)

    ax.axhline(CHANCE_LEVEL, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.axvline(CHANCE_LEVEL, color="red", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_xlabel("Overlap with Human concept (dim 16)", fontsize=10)
    ax.set_ylabel("Overlap with AI concept (dim 17)", fontsize=10)
    ax.set_title("Human vs AI Entity Reference Comparison", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Legend for categories
    seen = set()
    for r in sorted_results:
        cat = r["category"]
        if cat not in seen:
            ax.scatter([], [], c=CATEGORY_COLORS.get(cat, "#999"), label=cat, s=40)
            seen.add(cat)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


def make_layer_profiles(data):
    """Layer-resolved cosine profiles for selected pairs."""
    if "layer_profiles" not in data:
        return None

    lp = data["layer_profiles"]
    dim_ids = data["dim_ids"]

    valid_pairs = []
    for id_a, id_b, label in LAYER_PROFILE_PAIRS:
        if id_a in dim_ids and id_b in dim_ids:
            valid_pairs.append((id_a, id_b, label))

    if not valid_pairs:
        return None

    n_pairs = len(valid_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows),
                             squeeze=False)

    for idx, (id_a, id_b, label) in enumerate(valid_pairs):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        i = dim_ids.index(id_a)
        j = dim_ids.index(id_b)
        layers = np.arange(lp.shape[2])
        cosines = lp[i, j, :]

        ax.plot(layers, cosines, color="#9C27B0", linewidth=1.5)
        ax.axhline(0, color="#ccc", linewidth=0.5)
        ax.axvline(data["layer_range_start"], color="red", linewidth=0.5,
                   linestyle="--", alpha=0.5)
        ax.fill_between(layers, cosines, 0, alpha=0.15, color="#9C27B0")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylabel("Cosine similarity", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.tick_params(labelsize=7)

    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Layer-Resolved Cosine Similarity Between Standalone Concept Pairs",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def make_category_summary(data):
    """Mean overlap within vs between categories."""
    overlap = data["overlap"]
    dim_ids = data["dim_ids"]
    n = len(dim_ids)
    cats = [CATEGORY_FOR_DIM.get(d, "Other") for d in dim_ids]

    unique_cats = [c for c in CATEGORY_ORDER if c in set(cats)]

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

        other_indices = [i for i in range(n) if cats[i] != cat]
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
    ax.set_title("Within vs Between Category Concept Overlap (Standalone)", fontsize=11)
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
    typical_n = 40
    if prompt_counts:
        counts = list(prompt_counts.values())
        typical_n = max(set(counts), key=counts.count) if counts else 40

    html = []
    html.append('<h2>Methods</h2>')
    html.append('<div class="methods">')
    html.append('<h3>Pipeline Overview</h3>')
    html.append('<p>This analysis measures how much the <strong>standalone concept mean vectors</strong> '
                'of different dimensions overlap in LLaMA-2-13B\'s representation space. '
                'Unlike the contrast analysis (2f), there is no human-vs-AI framing here. '
                'Each concept is presented independently (e.g., "Think about phenomenological '
                'experience"), and we measure cosine similarity between the mean activation '
                'vectors of different concepts.</p>')

    html.append('<h3>Key Difference from Contrast Overlap (2f)</h3>')
    html.append('<table class="comparison-table">')
    html.append('<tr><th></th><th>Contrast (2f)</th><th>Standalone (2g)</th></tr>')
    html.append('<tr><td>Input</td><td>Human- and AI-framed prompts</td>'
                '<td>Unframed concept prompts</td></tr>')
    html.append('<tr><td>Vector</td><td>mean(human) - mean(AI) per layer</td>'
                '<td>mean(all prompts) per layer</td></tr>')
    html.append('<tr><td>Overlap means</td><td>"Do these dims define human-vs-AI the same way?"</td>'
                '<td>"Do these concepts activate similar patterns?"</td></tr>')
    html.append('<tr><td>N dims</td><td>19 (with dim 0 baseline)</td>'
                f'<td>{n_dims} (no baseline; has entity + sysprompt variants)</td></tr>')
    html.append('<tr><td>Entity reference</td><td>Dim 0 (explicit baseline)</td>'
                '<td>Dim 16 (human) and dim 17 (AI)</td></tr>')
    html.append('</table>')

    html.append('<h3>Data</h3>')
    html.append('<ul>')
    html.append(f'<li><strong>{n_dims} standalone dimensions</strong>, each with '
                f'~{typical_n} concept prompts (no human/AI labels)</li>')
    html.append(f'<li><strong>41 layers</strong>, each 5120-dimensional</li>')
    html.append(f'<li><strong>Layers {layer_start}-{layer_end}</strong> used for summary statistics</li>')
    html.append('</ul>')

    html.append('<h3>Computation</h3>')
    html.append('<ol>')
    html.append('<li><strong>Mean vector</strong>: For each dimension d and layer L, compute '
                'mean_d[L] = mean(all prompt activations at layer L).</li>')
    html.append('<li><strong>Centering</strong>: Compute global_mean[L] = mean across all '
                'dimensions of mean_d[L]. Then centered_d[L] = mean_d[L] - global_mean[L]. '
                'This removes the shared "abstract concept processing" component that '
                'dominates raw mean vectors.</li>')
    html.append('<li><strong>Pairwise overlap</strong>: For each pair (i, j) and each layer L, '
                'compute |cos(centered_i[L], centered_j[L])|, average across '
                f'layers {layer_start}-{layer_end}.</li>')
    html.append(f'<li><strong>Bootstrap CIs</strong>: {n_boot} iterations. Each iteration '
                'resamples prompts with replacement for each dimension, recomputes mean vectors, '
                'subtracts that iteration\'s global mean, then recomputes pairwise |cos|.</li>')
    html.append('</ol>')

    html.append('<h3>Why Centering?</h3>')
    html.append('<p>Raw mean vectors for standalone concepts are dominated by a large shared '
                'component — all concept prompts produce similar average activations because '
                'they share similar prompt structure and abstract reasoning. This makes all '
                'raw pairwise cosines ~0.96, masking real differences between concepts. '
                'Centering (subtracting the global mean across all dimensions) removes this '
                'shared component and reveals concept-specific structure: centered overlaps '
                'have mean ~0.49, std ~0.22, range 0.05-0.92 — comparable to contrast results.</p>')
    html.append('<p>This is analogous to mean-centering in PCA or removing the first principal '
                'component. The raw results are preserved for reference.</p>')

    html.append(f'<h3>Chance Level</h3>')
    html.append(f'<p>E[|cos|] for random 5120-d vectors = <strong>{CHANCE_LEVEL:.4f}</strong>.</p>')
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
        n = prompt_counts.get(d, "-")
        html.append(f'<tr><td>{d}</td>'
                     f'<td style="color:{color}">{name}</td>'
                     f'<td>{cat}</td>'
                     f'<td style="text-align:left">{desc}</td>'
                     f'<td>{n}</td></tr>')

    html.append('</table>')
    return '\n'.join(html)


def generate_heatmap_narrative(data):
    overlap = data["overlap"]
    dim_ids = data["dim_ids"]
    n = len(dim_ids)

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
    html.append('<p>Each cell shows the mean |cosine similarity| between two standalone '
                'concepts\' <strong>centered</strong> mean activation vectors, averaged across '
                'layers 6-40. Centering removes the shared component that makes all raw cosines '
                '~0.96, revealing concept-specific structure.</p>')

    html.append('<h3>Key Findings</h3>')
    html.append('<p><strong>Highest-overlap pairs</strong>:</p><ol>')
    for d_i, d_j, val in top5:
        html.append(f'<li>{DIM_NAMES.get(d_i, d_i)} x {DIM_NAMES.get(d_j, d_j)}: '
                     f'<strong>{val:.3f}</strong></li>')
    html.append('</ol>')

    html.append('<p><strong>Lowest-overlap pairs</strong>:</p><ol>')
    for d_i, d_j, val in bottom5:
        html.append(f'<li>{DIM_NAMES.get(d_i, d_i)} x {DIM_NAMES.get(d_j, d_j)}: '
                     f'<strong>{val:.3f}</strong></li>')
    html.append('</ol>')

    off_diag = [overlap[i, j] for i in range(n) for j in range(i+1, n)]
    mean_off = np.mean(off_diag)
    std_off = np.std(off_diag)
    html.append(f'<p>Mean off-diagonal overlap: <strong>{mean_off:.3f}</strong> '
                f'(std: {std_off:.3f}, chance: {CHANCE_LEVEL:.3f})</p>')
    html.append('</div>')
    return '\n'.join(html)


def generate_entity_reference_narrative(data):
    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append('<p>How much each standalone concept\'s <strong>centered</strong> mean '
                'activation pattern resembles the standalone "human" concept (dim 16) and '
                '"AI" concept (dim 17). These serve as entity-level references analogous to '
                'the baseline (dim 0) in the contrast analysis, but derived from standalone '
                'concept prompts rather than entity-framing contrasts.</p>')

    html.append('<h3>Interpretation</h3>')
    html.append('<ul>')
    html.append('<li><strong>High human overlap, low AI overlap</strong>: Concept activates '
                'patterns more similar to "human" than "AI"</li>')
    html.append('<li><strong>High overlap with both</strong>: Concept is broadly related to '
                'entity representations</li>')
    html.append('<li><strong>Low overlap with both</strong>: Concept is orthogonal to entity '
                'identity (e.g., shapes, formality)</li>')
    html.append('</ul>')
    html.append('</div>')
    return '\n'.join(html)


def generate_entity_comparison_narrative():
    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append('<p>Each point is a standalone concept, plotted by its overlap with the '
                '"human" concept (x-axis) vs "AI" concept (y-axis). Points on the '
                'diagonal have equal overlap with both entity concepts.</p>')
    html.append('<h3>Interpretation</h3>')
    html.append('<ul>')
    html.append('<li><strong>Above diagonal</strong>: More similar to AI than human</li>')
    html.append('<li><strong>Below diagonal</strong>: More similar to human than AI</li>')
    html.append('<li><strong>Far from origin</strong>: Strongly related to entity concepts</li>')
    html.append('<li><strong>Near origin</strong>: Entity-orthogonal concept</li>')
    html.append('</ul>')
    html.append('</div>')
    return '\n'.join(html)


def generate_layer_profiles_narrative(data):
    layer_start = data["layer_range_start"]
    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append('<p>Signed cosine similarity at each layer for selected standalone concept '
                'pairs. Positive = similar direction, negative = opposite direction.</p>')
    html.append('<h3>Selected Pairs</h3>')
    html.append('<ul>')
    html.append('<li><strong>Human vs Phenomenology</strong>: Does the "human" concept '
                'align with phenomenological experience?</li>')
    html.append('<li><strong>AI vs Phenomenology</strong>: Same question for the AI concept</li>')
    html.append('<li><strong>Human vs AI</strong>: How similar are the two entity concepts themselves?</li>')
    html.append('<li><strong>Human vs Shapes</strong>: Negative control — should be low</li>')
    html.append('<li><strong>Phenomenology vs Emotions</strong>: Two closely related mental concepts</li>')
    html.append('<li><strong>Phenomenology vs Embodiment</strong>: Cross-category (Mental vs Physical)</li>')
    html.append('<li><strong>Human vs Biological</strong>: Humans are biological — does the model reflect this?</li>')
    html.append('<li><strong>SysPrompt talkto vs bare</strong>: Different prompt styles for same entity</li>')
    html.append('</ul>')
    html.append(f'<p>Red dashed line: layer {layer_start} cutoff.</p>')
    html.append('</div>')
    return '\n'.join(html)


def generate_category_summary_narrative():
    html = []
    html.append('<div class="narrative">')
    html.append('<h3>What This Shows</h3>')
    html.append('<p>Average pairwise |cosine| within each category (solid) vs between '
                'categories (hatched). Higher within-category overlap suggests the model '
                'groups these concepts similarly.</p>')
    html.append('<h3>Categories</h3>')
    html.append('<ul>')
    html.append('<li><strong>Mental</strong> (dims 1-7, 18): Core mental capacities</li>')
    html.append('<li><strong>Physical</strong> (dims 8-10): Embodiment, roles, animacy</li>')
    html.append('<li><strong>Pragmatic</strong> (dims 11-13): Formality, expertise, helpfulness</li>')
    html.append('<li><strong>Entity</strong> (dims 16-17): Human and AI standalone concepts</li>')
    html.append('<li><strong>Bio Ctrl</strong> (dim 14): Biological processes</li>')
    html.append('<li><strong>Shapes</strong> (dim 15): Geometric shapes</li>')
    html.append('<li><strong>SysPrompt</strong> (dims 20-23): System prompt variants</li>')
    html.append('</ul>')
    html.append('</div>')
    return '\n'.join(html)


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_html(figures_b64, data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_dims = len(data["dim_ids"])
    layer_start = data["layer_range_start"]
    layer_end = data["layer_range_end"] - 1
    n_boot = data["n_bootstrap"]

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Standalone Concept Overlap Analysis</title>",
        "<style>",
        "  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;",
        "         max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }",
        "  h1 { border-bottom: 2px solid #9C27B0; padding-bottom: 8px; }",
        "  h2 { color: #7B1FA2; margin-top: 2em; }",
        "  h3 { color: #424242; margin-top: 1.2em; }",
        "  .meta { color: #666; font-size: 0.9em; margin-bottom: 2em; }",
        "  .figure { margin: 1.5em 0; text-align: center; }",
        "  .figure img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }",
        "  .caption { font-size: 0.85em; color: #666; margin-top: 0.5em; }",
        "  table { border-collapse: collapse; margin: 1em 0; font-size: 0.85em; }",
        "  th, td { border: 1px solid #ddd; padding: 4px 8px; text-align: right; }",
        "  th { background: #f5f5f5; text-align: center; }",
        "  .dim-table td:nth-child(4) { text-align: left; max-width: 400px; }",
        "  .comparison-table td:first-child { font-weight: bold; text-align: left; }",
        "  .note { background: #F3E5F5; border-left: 4px solid #9C27B0;",
        "          padding: 12px 16px; margin: 1em 0; font-size: 0.9em; }",
        "  .methods { background: #FAFAFA; border: 1px solid #E0E0E0;",
        "             padding: 16px 20px; margin: 1em 0; border-radius: 4px; }",
        "  .methods h3 { color: #7B1FA2; margin-top: 0.8em; }",
        "  .methods h3:first-child { margin-top: 0; }",
        "  .narrative { background: #FFF8E1; border-left: 4px solid #FFC107;",
        "               padding: 12px 16px; margin: 1em 0; font-size: 0.9em; }",
        "  .narrative h3 { color: #F57F17; margin-top: 0.8em; }",
        "  .narrative h3:first-child { margin-top: 0; }",
        "  code { background: #f5f5f5; padding: 1px 4px; border-radius: 2px; font-size: 0.9em; }",
        "</style></head><body>",
        "",
        "<h1>Standalone Concept Overlap Analysis</h1>",
        f'<p class="meta">Generated: {timestamp} | '
        f'{n_dims} standalone dimensions | Layers {layer_start}-{layer_end} | '
        f'{n_boot} bootstrap iterations</p>',
        "",
        '<div class="note">',
        "<strong>Summary:</strong> This measures how much standalone concept activation "
        "patterns overlap in the model's representation space. Unlike the contrast overlap "
        "analysis (2f), there is no human-vs-AI framing — each concept is presented "
        "independently. All results use <strong>centered</strong> vectors (global mean "
        "subtracted) to remove the shared component that dominates raw mean vectors. "
        "Dims 16 (human) and 17 (AI) serve as entity references.",
        "</div>",
    ]

    # Methods
    lines.append(generate_methods_section(data))

    # Dimension table
    lines.append(generate_dimension_table(data))

    # Figure 1: Heatmap (centered)
    if "heatmap" in figures_b64:
        lines.extend([
            "<h2>1. Pairwise Overlap Matrix (Centered)</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["heatmap"]}" alt="Overlap heatmap (centered)">',
            '<p class="caption">Mean |cosine similarity| between <strong>centered</strong> '
            'standalone concept mean vectors, averaged across layers 6-40. Centering removes '
            'the shared component across all concepts. Dimensions grouped by category.</p>',
            "</div>",
        ])
        lines.append(generate_heatmap_narrative(data))

    # Raw vs Centered comparison
    if "heatmap_raw" in figures_b64:
        lines.extend([
            '<h3>Raw vs Centered Comparison</h3>',
            '<div class="narrative">',
            '<h3>Why Centering Matters</h3>',
            '<p>The raw (uncentered) heatmap below shows all pairwise overlaps ~0.96. '
            'This is because standalone concept mean vectors are dominated by a large shared '
            'component — all concept prompts produce similar average activations due to '
            'shared prompt structure and abstract reasoning patterns. This shared component '
            'acts as a "DC offset" that overwhelms concept-specific variation.</p>',
            '<p>Centering (subtracting the global mean across all dimensions at each layer) '
            'removes this shared component, analogous to removing the first principal component '
            'or mean-centering before PCA. The centered analysis above reveals rich variation: '
            'some concept pairs are highly aligned (e.g., related mental capacities) while '
            'others are nearly orthogonal (e.g., shapes vs entity concepts).</p>',
            '<p><strong>Note:</strong> This issue does not affect the contrast analysis (2f) '
            'because contrast vectors (human_mean - AI_mean) already subtract out the shared '
            'component within each dimension.</p>',
            '</div>',
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["heatmap_raw"]}" '
            f'alt="Overlap heatmap (raw)">',
            '<p class="caption">Raw (uncentered) overlap matrix for reference. '
            'All values ~0.96 due to the shared component dominating mean vectors.</p>',
            "</div>",
        ])

    # Figure 2: Entity reference bar chart
    if "entity_reference" in figures_b64:
        lines.extend([
            "<h2>2. Entity Reference Overlap</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["entity_reference"]}" '
            f'alt="Entity reference overlap">',
            f'<p class="caption">Overlap with Human (dim 16) and AI (dim 17) standalone '
            f'concepts. Error bars: 95% bootstrap CI. '
            f'Red dashed line: chance ({CHANCE_LEVEL:.4f}).</p>',
            "</div>",
        ])
        lines.append(generate_entity_reference_narrative(data))

    # Entity overlap table
    if "entity_overlap" in data:
        lines.extend([
            "<h3>Entity Overlap Table</h3>",
            "<table>",
            "<tr><th>Dimension</th><th>Category</th>"
            "<th>|cos| Human</th><th>CI</th>"
            "<th>|cos| AI</th><th>CI</th></tr>",
        ])
        for r in sorted(data["entity_overlap"], key=lambda x: dim_sort_key(x["dim_id"])):
            color = CATEGORY_COLORS.get(r["category"], "#333")
            lines.append(
                f'<tr><td style="text-align:left; color:{color}">{r["name"]}</td>'
                f'<td style="text-align:left">{r["category"]}</td>'
                f'<td>{r["overlap_human_mean"]:.4f}</td>'
                f'<td>[{r["overlap_human_ci_lo"]:.4f}, {r["overlap_human_ci_hi"]:.4f}]</td>'
                f'<td>{r["overlap_ai_mean"]:.4f}</td>'
                f'<td>[{r["overlap_ai_ci_lo"]:.4f}, {r["overlap_ai_ci_hi"]:.4f}]</td></tr>'
            )
        lines.append("</table>")

    # Figure 3: Entity comparison scatter
    if "entity_comparison" in figures_b64:
        lines.extend([
            "<h2>3. Human vs AI Reference Comparison</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["entity_comparison"]}" '
            f'alt="Entity comparison">',
            '<p class="caption">Each point is a standalone concept. X-axis: overlap with '
            'Human concept. Y-axis: overlap with AI concept. Diagonal = equal overlap.</p>',
            "</div>",
        ])
        lines.append(generate_entity_comparison_narrative())

    # Figure 4: Layer profiles
    if "layer_profiles" in figures_b64:
        lines.extend([
            "<h2>4. Layer-Resolved Overlap Profiles</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["layer_profiles"]}" '
            f'alt="Layer profiles">',
            f'<p class="caption">Cosine similarity (signed) between selected standalone '
            f'concept pairs at each layer. Red dashed line marks layer '
            f'{data["layer_range_start"]}.</p>',
            "</div>",
        ])
        lines.append(generate_layer_profiles_narrative(data))

    # Figure 5: Category summary
    if "category_summary" in figures_b64:
        lines.extend([
            "<h2>5. Category-Level Summary</h2>",
            '<div class="figure">',
            f'<img src="data:image/png;base64,{figures_b64["category_summary"]}" '
            f'alt="Category summary">',
            '<p class="caption">Mean pairwise |cosine| within each category (solid) '
            'vs between categories (hatched).</p>',
            "</div>",
        ])
        lines.append(generate_category_summary_narrative())

    lines.extend(["</body></html>"])
    return "\n".join(lines)


def generate_markdown(data):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_dims = len(data["dim_ids"])
    layer_start = data["layer_range_start"]
    layer_end = data["layer_range_end"] - 1
    n_boot = data["n_bootstrap"]
    prompt_counts = data.get("prompt_counts", {})

    lines = [
        "# Standalone Concept Overlap Analysis",
        "",
        f"Generated: {timestamp} | {n_dims} standalone dimensions | "
        f"Layers {layer_start}-{layer_end} | {n_boot} bootstrap iterations",
        "",
        "## Summary",
        "",
        "Measures how much standalone concept activation patterns overlap. "
        "No human-vs-AI framing — each concept presented independently. "
        "All results use centered vectors (global mean subtracted) to remove shared component. "
        "Dims 16 (human) and 17 (AI) serve as entity references.",
        "",
        "## Dimension Reference",
        "",
        "| ID | Name | Category | N prompts |",
        "|----|------|----------|-----------|",
    ]
    for d in sorted(data["dim_ids"], key=dim_sort_key):
        name = DIM_NAMES.get(d, f"dim_{d}")
        cat = CATEGORY_FOR_DIM.get(d, "Other")
        n = prompt_counts.get(d, "-")
        lines.append(f"| {d} | {name} | {cat} | {n} |")
    lines.append("")

    lines.extend([
        "## 1. Pairwise Overlap Matrix (Centered)",
        "",
        "![Overlap Heatmap](figures/overlap_heatmap.png)",
        "",
        "### Raw vs Centered",
        "",
        "![Raw Overlap Heatmap](figures/overlap_heatmap_raw.png)",
        "",
        "## 2. Entity Reference Overlap",
        "",
        "![Entity Reference](figures/entity_reference.png)",
        "",
    ])

    if "entity_overlap" in data:
        lines.extend([
            "| Dimension | Category | |cos| Human | CI | |cos| AI | CI |",
            "|-----------|----------|------------|----|---------|----|",
        ])
        for r in sorted(data["entity_overlap"], key=lambda x: dim_sort_key(x["dim_id"])):
            lines.append(
                f"| {r['name']} | {r['category']} | "
                f"{r['overlap_human_mean']:.4f} | "
                f"[{r['overlap_human_ci_lo']:.4f}, {r['overlap_human_ci_hi']:.4f}] | "
                f"{r['overlap_ai_mean']:.4f} | "
                f"[{r['overlap_ai_ci_lo']:.4f}, {r['overlap_ai_ci_hi']:.4f}] |"
            )
        lines.append("")

    lines.extend([
        "## 3. Human vs AI Reference Comparison",
        "",
        "![Entity Comparison](figures/entity_comparison.png)",
        "",
        "## 4. Layer-Resolved Profiles",
        "",
        "![Layer Profiles](figures/layer_profiles.png)",
        "",
        "## 5. Category Summary",
        "",
        "![Category Summary](figures/category_summary.png)",
        "",
    ])

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading standalone concept overlap data...")
    data = load_overlap_data()
    print(f"  {len(data['dim_ids'])} dimensions, "
          f"layers {data['layer_range_start']}-{data['layer_range_end'] - 1}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures_b64 = {}
    png_data = {}

    # Figure 1: Overlap heatmap (centered — primary)
    print("Generating overlap heatmap (centered)...")
    fig = make_overlap_heatmap(data)
    figures_b64["heatmap"] = fig_to_base64_png(fig)
    fig2 = make_overlap_heatmap(data)
    png_data["overlap_heatmap"] = fig_to_png_bytes(fig2)

    # Raw heatmap (for reference / raw-vs-centered comparison)
    if "overlap_raw" in data:
        print("Generating overlap heatmap (raw, for reference)...")
        fig = make_overlap_heatmap_raw(data)
        if fig is not None:
            figures_b64["heatmap_raw"] = fig_to_base64_png(fig)
            fig2 = make_overlap_heatmap_raw(data)
            png_data["overlap_heatmap_raw"] = fig_to_png_bytes(fig2)

    # Figure 2: Entity reference bar chart (from centered data)
    if "entity_overlap" in data:
        print("Generating entity reference bar chart...")
        fig = make_entity_reference_bar_chart(data["entity_overlap"])
        figures_b64["entity_reference"] = fig_to_base64_png(fig)
        fig2 = make_entity_reference_bar_chart(data["entity_overlap"])
        png_data["entity_reference"] = fig_to_png_bytes(fig2)

    # Figure 3: Entity comparison scatter
    if "entity_overlap" in data:
        print("Generating entity comparison scatter...")
        fig = make_entity_comparison(data["entity_overlap"])
        figures_b64["entity_comparison"] = fig_to_base64_png(fig)
        fig2 = make_entity_comparison(data["entity_overlap"])
        png_data["entity_comparison"] = fig_to_png_bytes(fig2)

    # Figure 4: Layer profiles (centered)
    if "layer_profiles" in data:
        print("Generating layer profiles...")
        fig = make_layer_profiles(data)
        if fig is not None:
            figures_b64["layer_profiles"] = fig_to_base64_png(fig)
            fig2 = make_layer_profiles(data)
            png_data["layer_profiles"] = fig_to_png_bytes(fig2)

    # Figure 5: Category summary (centered)
    print("Generating category summary...")
    fig = make_category_summary(data)
    figures_b64["category_summary"] = fig_to_base64_png(fig)
    fig2 = make_category_summary(data)
    png_data["category_summary"] = fig_to_png_bytes(fig2)

    # Save figure PNGs to figures/ subdir
    for name, data_bytes in png_data.items():
        path = fig_dir / f"{name}.png"
        with open(path, "wb") as f:
            f.write(data_bytes)
        print(f"  Saved {path}")

    # Generate HTML
    print("Generating HTML report...")
    html = generate_html(figures_b64, data)
    html_path = OUTPUT_DIR / "concept_overlap_report.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  {html_path}")

    # Generate MD
    print("Generating markdown report...")
    md = generate_markdown(data)
    md_path = OUTPUT_DIR / "concept_overlap_report.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
