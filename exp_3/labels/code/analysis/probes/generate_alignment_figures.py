#!/usr/bin/env python3
"""
Unified figure generation for concept-probe alignment (contrast + standalone modes).

This script consolidates:
  - 2e_concept_probe_figures.py (contrast figures)
  - 3b_standalone_figures.py (standalone figures)

Generates publication-quality figures for both analysis modes.

Usage:
    python generate_alignment_figures.py --mode contrast
    python generate_alignment_figures.py --mode standalone
    python generate_alignment_figures.py --mode both

Outputs:
    results/probes/alignment/figures/              (contrast mode)
    results/probes/standalone_alignment/figures/   (standalone mode)

Env: llama2_env (needs numpy, matplotlib; no GPU)
Rachel C. Metzgar, Feb 2026
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import config

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

# Category colors
CAT_PALETTE = {
    "Mental":              "#3274A1",
    "Physical":            "#E1812C",
    "Pragmatic":           "#3A923A",
    "SysPrompt":           "#845B53",
    "Human vs AI (General)": "#999999",
    "Bio Ctrl":            "#D4A03A",
    "Shapes":              "#E377C2",
    "Entity":              "#C03D3E",
}

# Dimension labels
DIM_LABELS = {
    0: "Human vs AI\n(General)",
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


# ========================== LOADING ========================== #

def load_stats(mode):
    """Load statistics JSON for the specified mode."""
    if mode == "contrast":
        stats_json = str(config.RESULTS.root / "probes" / "alignment" / "summaries" / "alignment_stats.json")
    else:  # standalone
        stats_json = str(config.RESULTS.root / "probes" / "standalone_alignment" / "summaries" / "standalone_alignment_stats.json")

    if not os.path.isfile(stats_json):
        print(f"ERROR: Stats file not found: {stats_json}")
        print(f"Run compute_alignment_stats.py --mode {mode} first!")
        return None

    with open(stats_json) as f:
        return json.load(f)


def get_category(dim_id, mode):
    """Get category for a dimension ID."""
    if mode == "contrast":
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 17],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Human vs AI (General)":  [0],
            "Bio Ctrl":  [14],
            "Shapes":    [15],
            "SysPrompt": [18],
        }
    else:  # standalone
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 18],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Bio Ctrl":  [14],
            "Shapes":    [15],
            "Entity":    [16, 17],
            "SysPrompt": [20, 21, 22, 23],
        }

    for cat, ids in categories.items():
        if dim_id in ids:
            return cat
    return "Other"


# ========================== FIGURES ========================== #

def create_bar_chart(stats, mode, probe_type, layer_range, out_dir):
    """Create bar chart of alignment by dimension."""
    # Extract data
    dim_ids = []
    dim_names = []
    observed = []
    categories = []

    for dim_id_str, res in stats.items():
        dim_id = int(dim_id_str)
        key = f"{probe_type}_{layer_range}"
        if key in res:
            dim_ids.append(dim_id)
            dim_names.append(res["dim_name"])
            observed.append(res[key]["observed"])
            categories.append(get_category(dim_id, mode))

    # Sort by observed value
    sorted_indices = np.argsort(observed)[::-1]
    dim_ids = [dim_ids[i] for i in sorted_indices]
    dim_names = [dim_names[i] for i in sorted_indices]
    observed = [observed[i] for i in sorted_indices]
    categories = [categories[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [CAT_PALETTE.get(cat, "#999999") for cat in categories]
    bars = ax.bar(range(len(observed)), observed, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Labels
    ax.set_xlabel("Concept Dimension", fontweight='bold')
    if mode == "contrast":
        ax.set_ylabel("Alignment (Human-AI Difference)", fontweight='bold')
        title = f"Concept-Probe Alignment ({probe_type.replace('_', ' ').title()}, {layer_range.replace('_', ' ')})"
    else:
        ax.set_ylabel("Projection onto Probe Direction", fontweight='bold')
        title = f"Standalone Concept Alignment ({probe_type.replace('_', ' ').title()}, {layer_range.replace('_', ' ')})"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(dim_names)))
    ax.set_xticklabels([DIM_LABELS.get(dim_ids[i], dim_names[i]) for i in range(len(dim_names))],
                        rotation=45, ha='right', fontsize=8)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)

    # Legend
    unique_cats = list(dict.fromkeys(categories))  # preserve order
    legend_elements = [mpatches.Patch(facecolor=CAT_PALETTE.get(cat, "#999999"), label=cat)
                       for cat in unique_cats]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    filename = f"{mode}_{probe_type}_{layer_range}_bar_chart.png"
    fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Created: {filename}")


def create_category_comparison(stats, mode, probe_type, layer_range, out_dir):
    """Create category-level comparison bar chart."""
    # Aggregate by category
    category_data = {}
    for dim_id_str, res in stats.items():
        dim_id = int(dim_id_str)
        cat = get_category(dim_id, mode)
        key = f"{probe_type}_{layer_range}"
        if key in res:
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(res[key]["observed"])

    # Compute means
    categories = []
    means = []
    for cat in sorted(category_data.keys()):
        categories.append(cat)
        means.append(np.mean(category_data[cat]))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [CAT_PALETTE.get(cat, "#999999") for cat in categories]
    bars = ax.bar(range(len(means)), means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

    # Labels
    ax.set_xlabel("Category", fontweight='bold')
    if mode == "contrast":
        ax.set_ylabel("Mean Alignment", fontweight='bold')
        title = f"Category-Level Alignment ({probe_type.replace('_', ' ').title()}, {layer_range.replace('_', ' ')})"
    else:
        ax.set_ylabel("Mean Projection", fontweight='bold')
        title = f"Category-Level Projection ({probe_type.replace('_', ' ').title()}, {layer_range.replace('_', ' ')})"

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()
    filename = f"{mode}_{probe_type}_{layer_range}_category_comparison.png"
    fig.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Created: {filename}")


# ========================== MAIN ========================== #

def generate_figures_for_mode(mode):
    """Generate all figures for a specific analysis mode."""
    print(f"\n{'=' * 80}")
    print(f"GENERATING FIGURES: {mode.upper()} MODE")
    print(f"{'=' * 80}")

    # Load stats
    stats = load_stats(mode)
    if stats is None:
        return

    # Set up output directory
    if mode == "contrast":
        out_dir = str(config.RESULTS.root / "probes" / "alignment" / "figures")
    else:
        out_dir = str(config.RESULTS.root / "probes" / "standalone_alignment" / "figures")

    os.makedirs(out_dir, exist_ok=True)

    # Generate figures for each probe type and layer range
    probe_types = ["control_probe", "reading_probe"]
    layer_ranges = ["all_layers", "layers_6plus"]

    for probe_type in probe_types:
        for layer_range in layer_ranges:
            print(f"\n{probe_type} / {layer_range}:")
            create_bar_chart(stats, mode, probe_type, layer_range, out_dir)
            create_category_comparison(stats, mode, probe_type, layer_range, out_dir)

    print(f"\n✅ All figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Unified figure generation for concept-probe alignment"
    )
    parser.add_argument("--mode", required=True, choices=["contrast", "standalone", "both"],
                        help="Figure mode: contrast, standalone, or both")
    args = parser.parse_args()

    print("=" * 80)
    print("UNIFIED ALIGNMENT FIGURE GENERATION")
    print(f"Mode: {args.mode.upper()}")
    print("=" * 80)

    if args.mode in ("contrast", "both"):
        generate_figures_for_mode("contrast")

    if args.mode in ("standalone", "both"):
        generate_figures_for_mode("standalone")

    print("\n✅ All figure generation complete!")


if __name__ == "__main__":
    main()
