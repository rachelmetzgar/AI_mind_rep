#!/usr/bin/env python3
"""
Generate cross-version raw alignment comparison report (HTML + MD).

Reads summary.json from each version's contrasts/raw/ directory and produces:
  - raw_comparison.html  (interactive report with embedded SVG figures)
  - raw_comparison.md    (markdown summary)

Chart types:
  1. Category Summary: control dims individually + category averages, dashed separator
  2. Per-Category Breakdowns: individual dims per category + control dims at reduced opacity
  3. All Dimensions: controls first, dashed separator, then all dims by category
  4. Heatmaps: rows = dimensions, columns = versions

No external dependencies — uses only Python standard library.

Usage:
    python generate_raw_comparison.py               # turn 5 (default)
    python generate_raw_comparison.py --turn 3      # turn 3

Rachel C. Metzgar · Feb 2026
"""

import base64
import io
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================

from config import config, get_model
_MODEL_RESULTS = config.RESULTS.root / get_model()
ALIGNMENT_ROOT = _MODEL_RESULTS
VERSIONS = ["balanced_gpt", "nonsense_codeword"]

VERSION_LABELS = {
    "balanced_gpt": "Partner Identity",
    "nonsense_codeword": "Control",
}

VERSION_DESCRIPTIONS = {
    "balanced_gpt": "Balanced names with GPT-4 replacing 'AI' partner — explicit identity cues",
    "nonsense_codeword": "Nonsense codewords replacing identity labels — no meaningful identity cues",
}

# Probe display names (metacognitive = formerly "reading", operational = formerly "control")
PROBE_LABELS = {
    "reading": "Metacognitive",
    "control": "Operational",
}

# Probe type header colors (for section subheadings)
PROBE_HEADER_COLORS = {
    "reading": "#43A047",   # medium green
    "control": "#424242",   # dark charcoal
}

# Dimension display info
DIM_CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 25, 26, 27],
    "Pragmatic": [11, 12, 13],
    "Control":   [0, 14, 15, 29, 30, 31, 32],
    "SysPrompt": [18],
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Pragmatic": "#FF9800",
    "Control": "#9E9E9E",
    "SysPrompt": "#00BCD4",
}

DIM_NAMES = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind (holistic)", 17: "Attention", 18: "SysPrompt",
    25: "Beliefs", 26: "Desires", 27: "Goals",
    29: "Shapes (flip)", 30: "Granite/Sandstone", 31: "Squares/Triangles",
    32: "Horizontal/Vertical",
}

# Category-summary chart uses different label for dim 0
CATEGORY_CHART_NAMES = dict(DIM_NAMES)
CATEGORY_CHART_NAMES[0] = "Human vs AI (general)"

# Exclude dim 16 (circular: pooled from other dims)
# Dim 0 (baseline) optionally excluded via --no-baseline
EXCLUDE_DIMS = {16}
BASELINE_DIM = 0

# Colors: version × probe type. Metacognitive = lighter, Operational = darker.
# Partner Identity = muted greens, Control = grayscale.
VERSION_COLORS = {
    "balanced_gpt":       {"reading": "#81C784", "control": "#2E7D32"},
    "nonsense_codeword":  {"reading": "#BDBDBD", "control": "#616161"},
}

# Heatmap color scales per version
HEATMAP_SCALES = {
    "balanced_gpt":       {"hi": (46, 125, 50),   "lo": (232, 245, 233)},
    "nonsense_codeword":  {"hi": (97, 97, 97),    "lo": (238, 238, 238)},
}

# Category order for sorting
CAT_ORDER = ["Control", "Mental", "Pragmatic", "SysPrompt"]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_summaries(turn=5):
    """Load summary.json for each version. Returns dict[version] = summary."""
    data = {}
    for v in VERSIONS:
        path = ALIGNMENT_ROOT / v / "alignment" / f"turn_{turn}" / "contrasts" / "raw" / "data" / "summary.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {v}")
            continue
        with open(path) as f:
            data[v] = json.load(f)
    return data


def organize_data(all_data, exclude_dims=None):
    """Organize into structured dict keyed by dim_id."""
    skip = exclude_dims if exclude_dims is not None else EXCLUDE_DIMS
    result = {}
    for version, summary in all_data.items():
        for dim_folder, stats in summary.items():
            dim_id = stats["dim_id"]
            if dim_id in skip:
                continue
            if dim_id not in result:
                cat = "Other"
                for c, ids in DIM_CATEGORIES.items():
                    if dim_id in ids:
                        cat = c
                        break
                result[dim_id] = {
                    "name": DIM_NAMES.get(dim_id, dim_folder),
                    "folder": dim_folder,
                    "category": cat,
                    "versions": {},
                }
            result[dim_id]["versions"][version] = {
                "reading_r2": stats["reading_mean_r2"],
                "control_r2": stats["control_mean_r2"],
                "reading_ci": stats.get("reading_boot_ci95", [0, 0]),
                "control_ci": stats.get("control_boot_ci95", [0, 0]),
            }
    return result


def get_available_versions(dim_data):
    for dd in dim_data.values():
        return [v for v in VERSIONS if v in dd["versions"]]
    return []


def sort_dims_in_category(dim_ids, dim_data, category):
    """Sort dims within a category. Mental/Pragmatic: by Partner Identity alignment descending.
    Control/SysPrompt: by dim_id ascending."""
    if category in ("Control", "SysPrompt"):
        return sorted(dim_ids)

    def _alignment_key(d):
        vd = dim_data[d]["versions"].get("balanced_gpt")
        if vd is None:
            return 0
        return vd["reading_r2"] + vd["control_r2"]
    return sorted(dim_ids, key=_alignment_key, reverse=True)


def sorted_all_dims(dim_data):
    """Return all dim IDs sorted: by category order, then within-category by alignment."""
    result = []
    for cat in CAT_ORDER:
        cat_ids = [d for d in dim_data if dim_data[d]["category"] == cat]
        result.extend(sort_dims_in_category(cat_ids, dim_data, cat))
    # Any remaining categories not in CAT_ORDER
    seen = set(result)
    remaining = [d for d in dim_data if d not in seen]
    result.extend(sorted(remaining))
    return result


def compute_global_max(dim_data):
    """Compute global max R² across all dims, versions, probe types, and CI bounds."""
    all_r2 = []
    for dd in dim_data.values():
        for vd in dd["versions"].values():
            all_r2.extend([vd["reading_r2"], vd["control_r2"]])
            all_r2.extend(vd["reading_ci"])
            all_r2.extend(vd["control_ci"])
    return max(all_r2) * 1.05 if all_r2 else 0.001


# ============================================================================
# SVG HELPERS
# ============================================================================

def _svg_header(width, height):
    return (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;max-width:{width}px;height:auto;font-family:Arial,sans-serif;">\n'
            f'<rect width="{width}" height="{height}" fill="white"/>\n')


def _svg_y_axis(plot_w, plot_h, y_max, n_ticks=5):
    """Generate y-axis gridlines and labels. Returns list of SVG lines."""
    lines = []
    for i in range(n_ticks + 1):
        y_val = y_max * i / n_ticks
        y_pos = plot_h - (y_val / y_max * plot_h)
        lines.append(f'<line x1="0" y1="{y_pos:.1f}" x2="{plot_w}" y2="{y_pos:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="-8" y="{y_pos+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#666">{y_val*1000:.2f}</text>')
    lines.append(f'<text x="-50" y="{plot_h/2}" text-anchor="middle" font-size="11" fill="#333" '
                 f'transform="rotate(-90,-50,{plot_h/2})">Mean R\u00b2 (\u00d710\u207b\u00b3)</text>')
    return lines


def _svg_bar(x, y, w, h, color, opacity, title):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{color}" opacity="{opacity}">'
            f'<title>{title}</title></rect>')


def _svg_whisker(cx, y_lo, y_hi):
    lines = []
    lines.append(f'<line x1="{cx:.2f}" y1="{y_lo:.2f}" x2="{cx:.2f}" y2="{y_hi:.2f}" '
                 f'stroke="#333" stroke-width="1"/>')
    lines.append(f'<line x1="{cx-2:.2f}" y1="{y_lo:.2f}" x2="{cx+2:.2f}" y2="{y_lo:.2f}" '
                 f'stroke="#333" stroke-width="1"/>')
    lines.append(f'<line x1="{cx-2:.2f}" y1="{y_hi:.2f}" x2="{cx+2:.2f}" y2="{y_hi:.2f}" '
                 f'stroke="#333" stroke-width="1"/>')
    return lines


def _svg_version_legend(available_versions, probe_type, x_start, y):
    """Render version legend colored by probe type."""
    lines = []
    for i, v in enumerate(available_versions):
        x = x_start + i * 140
        color = VERSION_COLORS[v][probe_type]
        lines.append(f'<rect x="{x}" y="{y-8}" width="10" height="10" fill="{color}"/>')
        lines.append(f'<text x="{x+14}" y="{y}" font-size="9" fill="#333">{VERSION_LABELS[v]}</text>')
    return lines


# ============================================================================
# SVG CHART: CATEGORY SUMMARY
# ============================================================================

def svg_category_summary(dim_data, probe_type, global_max, chart_width=700, chart_height=360):
    """
    Category-level summary: individual control dims on left, category averages on right.
    Dashed vertical line separator.
    """
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"
    available_versions = get_available_versions(dim_data)
    n_av = len(available_versions)

    # Control dims (individual)
    control_dim_ids = sorted(d for d in dim_data if dim_data[d]["category"] == "Control")
    # Category averages
    cat_names = [c for c in CAT_ORDER if c not in ("Control", "SysPrompt")
                 and any(dim_data[d]["category"] == c for d in dim_data)]

    # Compute category averages and mean CIs
    cat_avgs = {}
    for cat in cat_names:
        cat_dims = [d for d in dim_data if dim_data[d]["category"] == cat]
        cat_avgs[cat] = {}
        for v in available_versions:
            vals = [dim_data[d]["versions"][v][key] for d in cat_dims if v in dim_data[d]["versions"]]
            ci_los = [dim_data[d]["versions"][v][ci_key][0] for d in cat_dims if v in dim_data[d]["versions"]]
            ci_his = [dim_data[d]["versions"][v][ci_key][1] for d in cat_dims if v in dim_data[d]["versions"]]
            cat_avgs[cat][v] = {
                "val": sum(vals) / len(vals) if vals else 0,
                "ci": [sum(ci_los) / len(ci_los) if ci_los else 0,
                       sum(ci_his) / len(ci_his) if ci_his else 0],
            }

    n_groups = len(control_dim_ids) + len(cat_names)
    margin = {"top": 30, "right": 20, "bottom": 120, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]
    group_w = plot_w / n_groups
    bar_w = group_w * 0.35
    y_max = global_max * 1.1

    lines = [_svg_header(chart_width, chart_height)]
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')
    lines.extend(_svg_y_axis(plot_w, plot_h, y_max))

    # Dashed separator between control dims and category averages
    sep_x = len(control_dim_ids) * group_w
    lines.append(f'<line x1="{sep_x}" y1="0" x2="{sep_x}" y2="{plot_h}" '
                 f'stroke="#ccc" stroke-width="1" stroke-dasharray="4,3"/>')

    # Draw control dims (individual)
    for gi, dim_id in enumerate(control_dim_ids):
        dd = dim_data[dim_id]
        x_group = gi * group_w + group_w * 0.15
        for vi, v in enumerate(available_versions):
            if v not in dd["versions"]:
                continue
            vd = dd["versions"][v]
            val = vd[key]
            ci = vd[ci_key]
            x = x_group + vi * (bar_w + 2)
            bar_h = max(val / y_max * plot_h, 0)
            y = plot_h - bar_h
            color = VERSION_COLORS[v][probe_type]
            lines.append(_svg_bar(x, y, bar_w, bar_h, color, 0.85,
                                  f'{VERSION_LABELS[v]} \u2014 {CATEGORY_CHART_NAMES.get(dim_id, dd["name"])}: {val*1000:.3f} \u00d710\u207b\u00b3'))
            cx = x + bar_w / 2
            y_lo = plot_h - max(ci[0] / y_max * plot_h, 0)
            y_hi = plot_h - max(ci[1] / y_max * plot_h, 0)
            lines.extend(_svg_whisker(cx, y_lo, y_hi))

        # X label
        x_center = x_group + n_av * (bar_w + 2) / 2
        lines.append(f'<text x="{x_center:.1f}" y="{plot_h+14}" text-anchor="end" '
                     f'font-size="10" fill="#9E9E9E" '
                     f'transform="rotate(-30,{x_center:.1f},{plot_h+14})">'
                     f'{CATEGORY_CHART_NAMES.get(dim_id, dd["name"])}</text>')

    # Draw category averages
    for ci_idx, cat in enumerate(cat_names):
        gi = len(control_dim_ids) + ci_idx
        x_group = gi * group_w + group_w * 0.15
        for vi, v in enumerate(available_versions):
            ca = cat_avgs[cat][v]
            val = ca["val"]
            ci = ca["ci"]
            x = x_group + vi * (bar_w + 2)
            bar_h = max(val / y_max * plot_h, 0)
            y = plot_h - bar_h
            color = VERSION_COLORS[v][probe_type]
            lines.append(_svg_bar(x, y, bar_w, bar_h, color, 0.85,
                                  f'{VERSION_LABELS[v]} \u2014 {cat}: {val*1000:.3f} \u00d710\u207b\u00b3'))
            cx = x + bar_w / 2
            y_lo = plot_h - max(ci[0] / y_max * plot_h, 0)
            y_hi = plot_h - max(ci[1] / y_max * plot_h, 0)
            lines.extend(_svg_whisker(cx, y_lo, y_hi))

        x_center = x_group + n_av * (bar_w + 2) / 2
        cat_color = CATEGORY_COLORS.get(cat, "#333")
        lines.append(f'<text x="{x_center:.1f}" y="{plot_h+14}" text-anchor="end" '
                     f'font-size="10" fill="{cat_color}" '
                     f'transform="rotate(-30,{x_center:.1f},{plot_h+14})">{cat}</text>')

    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333"/>')
    lines.append('</g>')

    # Annotations
    ctrl_center = margin["left"] + len(control_dim_ids) * group_w / 2
    cat_center = margin["left"] + (len(control_dim_ids) + len(cat_names) / 2) * group_w
    lines.append(f'<text x="{ctrl_center:.0f}" y="{chart_height - 38}" text-anchor="middle" '
                 f'font-size="10" fill="#666" font-style="italic">Control Dimensions</text>')
    lines.append(f'<text x="{cat_center:.0f}" y="{chart_height - 38}" text-anchor="middle" '
                 f'font-size="10" fill="#666" font-style="italic">Category Averages</text>')

    # Legend
    lines.extend(_svg_version_legend(available_versions, probe_type, margin["left"], chart_height - 12))
    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================================
# SVG CHART: PER-CATEGORY BREAKDOWN
# ============================================================================

def svg_category_breakdown(dim_data, probe_type, category, global_max, chart_width=800, chart_height=320):
    """
    Per-category breakdown: individual dims within category on left,
    control dims at reduced opacity on right, dashed separator.
    """
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"
    available_versions = get_available_versions(dim_data)
    n_av = len(available_versions)

    # Category dims
    cat_dim_ids = sort_dims_in_category(
        [d for d in dim_data if dim_data[d]["category"] == category], dim_data, category)
    # Control dims (shown at reduced opacity for reference)
    control_dim_ids = sorted(d for d in dim_data if dim_data[d]["category"] == "Control")

    n_groups = len(cat_dim_ids) + len(control_dim_ids)
    margin = {"top": 30, "right": 20, "bottom": 110, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]
    group_w = plot_w / max(n_groups, 1)
    bar_w = group_w * 0.35
    y_max = global_max * 1.1

    lines = [_svg_header(chart_width, chart_height)]
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')
    lines.extend(_svg_y_axis(plot_w, plot_h, y_max))

    def draw_dim(gi, dim_id, opacity):
        dd = dim_data[dim_id]
        x_group = gi * group_w + group_w * 0.15
        for vi, v in enumerate(available_versions):
            if v not in dd["versions"]:
                continue
            vd = dd["versions"][v]
            val = vd[key]
            ci = vd[ci_key]
            x = x_group + vi * (bar_w + 2)
            bar_h = max(val / y_max * plot_h, 0)
            y = plot_h - bar_h
            color = VERSION_COLORS[v][probe_type]
            name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
            lines.append(_svg_bar(x, y, bar_w, bar_h, color, opacity,
                                  f'{VERSION_LABELS[v]} \u2014 {name}: {val*1000:.3f} \u00d710\u207b\u00b3'))
            cx = x + bar_w / 2
            y_lo = plot_h - max(ci[0] / y_max * plot_h, 0)
            y_hi = plot_h - max(ci[1] / y_max * plot_h, 0)
            lines.extend(_svg_whisker(cx, y_lo, y_hi))

        x_center = x_group + n_av * (bar_w + 2) / 2
        cat = dd["category"]
        cat_color = CATEGORY_COLORS.get(cat, "#333")
        name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
        lines.append(f'<text x="{x_center:.1f}" y="{plot_h+14}" text-anchor="end" '
                     f'font-size="10" fill="{cat_color}" '
                     f'transform="rotate(-30,{x_center:.1f},{plot_h+14})">{name}</text>')

    # Draw category dims at full opacity
    for gi, dim_id in enumerate(cat_dim_ids):
        draw_dim(gi, dim_id, 0.85)

    # Dashed separator
    sep_x = len(cat_dim_ids) * group_w
    lines.append(f'<line x1="{sep_x}" y1="0" x2="{sep_x}" y2="{plot_h}" '
                 f'stroke="#ccc" stroke-width="1" stroke-dasharray="4,3"/>')

    # Draw control dims at reduced opacity
    for ci_idx, dim_id in enumerate(control_dim_ids):
        draw_dim(len(cat_dim_ids) + ci_idx, dim_id, 0.50)

    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333"/>')
    lines.append('</g>')

    # Legend
    lines.extend(_svg_version_legend(available_versions, probe_type, margin["left"], chart_height - 12))
    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================================
# SVG CHART: ALL DIMENSIONS
# ============================================================================

def svg_all_dims(dim_data, probe_type, global_max, chart_width=800, chart_height=340):
    """
    All dimensions: control dims first, dashed separator, then remaining dims by category.
    """
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"
    available_versions = get_available_versions(dim_data)
    n_av = len(available_versions)

    # All dims sorted: by category order, then within-category by alignment
    all_ids = sorted_all_dims(dim_data)

    # Find where control ends for dashed separator
    control_ids = [d for d in all_ids if dim_data[d]["category"] == "Control"]

    n_groups = len(all_ids)
    margin = {"top": 30, "right": 20, "bottom": 120, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]
    group_w = plot_w / max(n_groups, 1)
    bar_w = group_w * 0.35
    y_max = global_max * 1.1

    lines = [_svg_header(chart_width, chart_height)]
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')
    lines.extend(_svg_y_axis(plot_w, plot_h, y_max))

    # Dashed separator after control dims
    sep_x = len(control_ids) * group_w
    lines.append(f'<line x1="{sep_x}" y1="0" x2="{sep_x}" y2="{plot_h}" '
                 f'stroke="#ddd" stroke-width="1" stroke-dasharray="4,3"/>')

    for gi, dim_id in enumerate(all_ids):
        dd = dim_data[dim_id]
        x_group = gi * group_w + group_w * 0.15
        for vi, v in enumerate(available_versions):
            if v not in dd["versions"]:
                continue
            vd = dd["versions"][v]
            val = vd[key]
            ci = vd[ci_key]
            x = x_group + vi * (bar_w + 2)
            bar_h = max(val / y_max * plot_h, 0)
            y = plot_h - bar_h
            color = VERSION_COLORS[v][probe_type]
            lines.append(_svg_bar(x, y, bar_w, bar_h, color, 0.85,
                                  f'{VERSION_LABELS[v]}: {val*1000:.3f} \u00d710\u207b\u00b3'))
            cx = x + bar_w / 2
            y_lo = plot_h - max(ci[0] / y_max * plot_h, 0)
            y_hi = plot_h - max(ci[1] / y_max * plot_h, 0)
            lines.extend(_svg_whisker(cx, y_lo, y_hi))

        x_center = x_group + n_av * (bar_w + 2) / 2
        cat_color = CATEGORY_COLORS.get(dd["category"], "#333")
        name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
        lines.append(f'<text x="{x_center:.1f}" y="{plot_h+14}" text-anchor="end" '
                     f'font-size="9" fill="{cat_color}" '
                     f'transform="rotate(-45,{x_center:.1f},{plot_h+14})">{name}</text>')

    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333"/>')
    lines.append('</g>')

    # Legend
    lines.extend(_svg_version_legend(available_versions, probe_type, margin["left"], chart_height - 12))
    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================================
# SVG CHART: HEATMAP
# ============================================================================

def svg_heatmap(dim_data, probe_type, global_max, chart_width=700, row_height=28):
    """Generate an SVG heatmap: rows = dimensions, columns = versions."""
    key = f"{probe_type}_r2"

    sorted_dims = sorted_all_dims(dim_data)

    available_versions = get_available_versions(dim_data)
    n_dims = len(sorted_dims)
    n_vers = len(available_versions)

    margin = {"top": 100, "right": 80, "bottom": 20, "left": 140}
    cell_w = (chart_width - margin["left"] - margin["right"]) / max(n_vers, 1)
    chart_height = margin["top"] + n_dims * row_height + margin["bottom"]

    lines = [_svg_header(chart_width, chart_height)]

    # Column headers
    for vi, v in enumerate(available_versions):
        x = margin["left"] + vi * cell_w + cell_w / 2
        lines.append(f'<text x="{x:.1f}" y="{margin["top"]-10}" text-anchor="start" '
                     f'font-size="10" fill="#333" '
                     f'transform="rotate(-55,{x:.1f},{margin["top"]-10})">'
                     f'{VERSION_LABELS[v]}</text>')

    # Rows
    prev_cat = None
    for ri, dim_id in enumerate(sorted_dims):
        dd = dim_data[dim_id]
        y = margin["top"] + ri * row_height
        cat = dd["category"]
        if cat != prev_cat and prev_cat is not None:
            lines.append(f'<line x1="{margin["left"]}" y1="{y}" '
                         f'x2="{margin["left"] + n_vers * cell_w}" y2="{y}" '
                         f'stroke="#999" stroke-width="1.5"/>')
        prev_cat = cat
        cat_color = CATEGORY_COLORS.get(cat, "#333")
        name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
        lines.append(f'<text x="{margin["left"]-8}" y="{y + row_height/2 + 4}" '
                     f'text-anchor="end" font-size="10" fill="{cat_color}">{name}</text>')

        for vi, v in enumerate(available_versions):
            if v not in dd["versions"]:
                continue
            val = dd["versions"][v][key]
            intensity = min(val / global_max, 1.0) if global_max > 0 else 0
            scale = HEATMAP_SCALES.get(v, {"hi": (33, 150, 243), "lo": (227, 242, 253)})
            r = int(scale["lo"][0] + intensity * (scale["hi"][0] - scale["lo"][0]))
            g = int(scale["lo"][1] + intensity * (scale["hi"][1] - scale["lo"][1]))
            b = int(scale["lo"][2] + intensity * (scale["hi"][2] - scale["lo"][2]))
            x = margin["left"] + vi * cell_w
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" '
                         f'height="{row_height}" fill="rgb({r},{g},{b})" '
                         f'stroke="white" stroke-width="1">'
                         f'<title>{VERSION_LABELS[v]} \u2014 {name}: R\u00b2={val:.6f}</title></rect>')
            text_color = "#fff" if intensity > 0.5 else "#333"
            lines.append(f'<text x="{x + cell_w/2:.1f}" y="{y + row_height/2 + 4:.1f}" '
                         f'text-anchor="middle" font-size="8" fill="{text_color}">'
                         f'{val*1000:.2f}</text>')

    # Colorbars — one per version
    cb_x_start = margin["left"] + n_vers * cell_w + 15
    cb_h = n_dims * row_height
    cb_w = 12
    cb_gap = 30
    for vi, v in enumerate(available_versions):
        cb_x = cb_x_start + vi * (cb_w + cb_gap)
        scale = HEATMAP_SCALES.get(v, {"hi": (33, 150, 243), "lo": (227, 242, 253)})
        for i in range(50):
            frac = i / 49
            cr = int(scale["lo"][0] + frac * (scale["hi"][0] - scale["lo"][0]))
            cg = int(scale["lo"][1] + frac * (scale["hi"][1] - scale["lo"][1]))
            cb_val = int(scale["lo"][2] + frac * (scale["hi"][2] - scale["lo"][2]))
            y = margin["top"] + cb_h - (frac * cb_h)
            lines.append(f'<rect x="{cb_x}" y="{y:.1f}" width="{cb_w}" height="{cb_h/49+1:.1f}" '
                         f'fill="rgb({cr},{cg},{cb_val})"/>')
        lines.append(f'<text x="{cb_x + cb_w/2}" y="{margin["top"] - 4}" '
                     f'text-anchor="middle" font-size="7" fill="#333">{VERSION_LABELS[v]}</text>')
        if vi == 0:
            lines.append(f'<text x="{cb_x + cb_w + 3}" y="{margin["top"] + 4}" '
                         f'font-size="7" fill="#333">{global_max*1000:.2f}</text>')
            lines.append(f'<text x="{cb_x + cb_w + 3}" y="{margin["top"] + cb_h + 4}" '
                         f'font-size="7" fill="#333">0.00</text>')
            lines.append(f'<text x="{cb_x + cb_w + 3}" y="{margin["top"] + cb_h/2 + 4}" '
                         f'font-size="7" fill="#666">\u00d710\u207b\u00b3</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figures(dim_data):
    """Generate all SVG figures. Returns dict of {name: svg_string} and global_max."""
    global_max = compute_global_max(dim_data)

    figures = {}

    # Category summary (Section 3)
    for pt in ["reading", "control"]:
        figures[f"category_summary_{PROBE_LABELS[pt].lower()}"] = \
            svg_category_summary(dim_data, pt, global_max)

    # Per-category breakdowns (Section 4)
    breakdown_cats = [c for c in CAT_ORDER if c not in ("Control", "SysPrompt")
                      and any(dim_data[d]["category"] == c for d in dim_data)]
    for cat in breakdown_cats:
        for pt in ["reading", "control"]:
            figures[f"breakdown_{cat.lower()}_{PROBE_LABELS[pt].lower()}"] = \
                svg_category_breakdown(dim_data, pt, cat, global_max)

    # All dimensions (Sections 5-6)
    for pt in ["reading", "control"]:
        figures[f"all_dims_{PROBE_LABELS[pt].lower()}"] = \
            svg_all_dims(dim_data, pt, global_max)

    # Heatmaps (Section 7)
    for pt in ["reading", "control"]:
        figures[f"heatmap_{PROBE_LABELS[pt].lower()}"] = \
            svg_heatmap(dim_data, pt, global_max)

    return figures, global_max


# ============================================================================
# PNG GENERATION (matplotlib)
# ============================================================================

def fig_to_base64_png(fig, dpi=150):
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
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


def _mpl_category_summary(dim_data, probe_type, global_max):
    """Matplotlib category summary bar chart."""
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"
    available_versions = get_available_versions(dim_data)
    n_av = len(available_versions)

    control_dim_ids = sorted(d for d in dim_data if dim_data[d]["category"] == "Control")
    cat_names = [c for c in CAT_ORDER if c not in ("Control", "SysPrompt")
                 and any(dim_data[d]["category"] == c for d in dim_data)]

    # Compute category averages
    cat_avgs = {}
    for cat in cat_names:
        cat_dims = [d for d in dim_data if dim_data[d]["category"] == cat]
        cat_avgs[cat] = {}
        for v in available_versions:
            vals = [dim_data[d]["versions"][v][key] for d in cat_dims if v in dim_data[d]["versions"]]
            ci_los = [dim_data[d]["versions"][v][ci_key][0] for d in cat_dims if v in dim_data[d]["versions"]]
            ci_his = [dim_data[d]["versions"][v][ci_key][1] for d in cat_dims if v in dim_data[d]["versions"]]
            cat_avgs[cat][v] = {
                "val": sum(vals) / len(vals) if vals else 0,
                "ci": [sum(ci_los) / len(ci_los) if ci_los else 0,
                       sum(ci_his) / len(ci_his) if ci_his else 0],
            }

    labels = [CATEGORY_CHART_NAMES.get(d, dim_data[d]["name"]) for d in control_dim_ids] + cat_names
    n_groups = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n_groups * 0.8), 4))
    x = np.arange(n_groups)
    bar_w = 0.35

    for vi, v in enumerate(available_versions):
        vals = []
        errs_lo = []
        errs_hi = []
        for d in control_dim_ids:
            if v in dim_data[d]["versions"]:
                vd = dim_data[d]["versions"][v]
                val = vd[key]
                ci = vd[ci_key]
                vals.append(val * 1000)
                errs_lo.append(max((val - ci[0]) * 1000, 0))
                errs_hi.append(max((ci[1] - val) * 1000, 0))
            else:
                vals.append(0); errs_lo.append(0); errs_hi.append(0)
        for cat in cat_names:
            ca = cat_avgs[cat][v]
            vals.append(ca["val"] * 1000)
            errs_lo.append(max((ca["val"] - ca["ci"][0]) * 1000, 0))
            errs_hi.append(max((ca["ci"][1] - ca["val"]) * 1000, 0))

        color = VERSION_COLORS[v][probe_type]
        ax.bar(x + vi * bar_w, vals, bar_w, yerr=[errs_lo, errs_hi],
               color=color, alpha=0.85, label=VERSION_LABELS[v],
               capsize=2, error_kw={"linewidth": 0.5})

    # Dashed separator
    sep_x = len(control_dim_ids) - 0.5
    ax.axvline(x=sep_x, color="#ccc", linewidth=1, linestyle="--")

    ax.set_xticks(x + bar_w * (n_av - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean R² (×10⁻³)", fontsize=9)
    ax.set_ylim(0, global_max * 1.1 * 1000)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _mpl_category_breakdown(dim_data, probe_type, category, global_max):
    """Matplotlib per-category breakdown bar chart."""
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"
    available_versions = get_available_versions(dim_data)
    n_av = len(available_versions)

    cat_dim_ids = sort_dims_in_category(
        [d for d in dim_data if dim_data[d]["category"] == category], dim_data, category)
    control_dim_ids = sorted(d for d in dim_data if dim_data[d]["category"] == "Control")
    all_ids = cat_dim_ids + control_dim_ids

    labels = [CATEGORY_CHART_NAMES.get(d, dim_data[d]["name"]) for d in all_ids]
    n_groups = len(labels)
    fig, ax = plt.subplots(figsize=(max(7, n_groups * 0.7), 4))
    x = np.arange(n_groups)
    bar_w = 0.35

    for vi, v in enumerate(available_versions):
        vals = []; errs_lo = []; errs_hi = []
        alphas = []
        for i, d in enumerate(all_ids):
            if v in dim_data[d]["versions"]:
                vd = dim_data[d]["versions"][v]
                val = vd[key]; ci = vd[ci_key]
                vals.append(val * 1000)
                errs_lo.append(max((val - ci[0]) * 1000, 0))
                errs_hi.append(max((ci[1] - val) * 1000, 0))
            else:
                vals.append(0); errs_lo.append(0); errs_hi.append(0)
            alphas.append(0.85 if i < len(cat_dim_ids) else 0.40)

        color = VERSION_COLORS[v][probe_type]
        # Draw with proper alpha per bar
        for i in range(n_groups):
            ax.bar(x[i] + vi * bar_w, vals[i], bar_w,
                   yerr=[[errs_lo[i]], [errs_hi[i]]],
                   color=color, alpha=alphas[i],
                   capsize=2, error_kw={"linewidth": 0.5})

    # Dashed separator
    sep_x = len(cat_dim_ids) - 0.5
    ax.axvline(x=sep_x, color="#ccc", linewidth=1, linestyle="--")

    cat_colors = [CATEGORY_COLORS.get(dim_data[d]["category"], "#333") for d in all_ids]
    ax.set_xticks(x + bar_w * (n_av - 1) / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    for tick, color in zip(ax.get_xticklabels(), cat_colors):
        tick.set_color(color)
    ax.set_ylabel("Mean R² (×10⁻³)", fontsize=9)
    ax.set_ylim(0, global_max * 1.1 * 1000)
    ax.grid(axis="y", alpha=0.3)

    # Version legend (proxy patches since bars were drawn individually)
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=VERSION_COLORS[v][probe_type], alpha=0.85,
                            label=VERSION_LABELS[v]) for v in available_versions]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    fig.tight_layout()
    return fig


def _mpl_all_dims(dim_data, probe_type, global_max):
    """Matplotlib all-dimensions bar chart."""
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"
    available_versions = get_available_versions(dim_data)
    n_av = len(available_versions)

    all_ids = sorted_all_dims(dim_data)
    control_ids = [d for d in all_ids if dim_data[d]["category"] == "Control"]

    labels = [CATEGORY_CHART_NAMES.get(d, dim_data[d]["name"]) for d in all_ids]
    cat_colors = [CATEGORY_COLORS.get(dim_data[d]["category"], "#333") for d in all_ids]
    n_groups = len(labels)

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.55), 4))
    x = np.arange(n_groups)
    bar_w = 0.8 / max(n_av, 1)

    for vi, v in enumerate(available_versions):
        vals = []; errs_lo = []; errs_hi = []
        for d in all_ids:
            if v in dim_data[d]["versions"]:
                vd = dim_data[d]["versions"][v]
                val = vd[key]; ci = vd[ci_key]
                vals.append(val * 1000)
                errs_lo.append(max((val - ci[0]) * 1000, 0))
                errs_hi.append(max((ci[1] - val) * 1000, 0))
            else:
                vals.append(0); errs_lo.append(0); errs_hi.append(0)

        color = VERSION_COLORS[v][probe_type]
        ax.bar(x + vi * bar_w, vals, bar_w, yerr=[errs_lo, errs_hi],
               color=color, alpha=0.85, label=VERSION_LABELS[v],
               capsize=2, error_kw={"linewidth": 0.5})

    sep_x = len(control_ids) - 0.5
    ax.axvline(x=sep_x, color="#ddd", linewidth=1, linestyle="--")

    ax.set_xticks(x + bar_w * (n_av - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    for tick, color in zip(ax.get_xticklabels(), cat_colors):
        tick.set_color(color)
    ax.set_ylabel("Mean R² (×10⁻³)", fontsize=9)
    ax.set_ylim(0, global_max * 1.1 * 1000)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _mpl_heatmap(dim_data, probe_type, global_max):
    """Matplotlib heatmap."""
    key = f"{probe_type}_r2"
    sorted_dims = sorted_all_dims(dim_data)
    available_versions = get_available_versions(dim_data)
    n_dims = len(sorted_dims)
    n_vers = len(available_versions)

    matrix = np.zeros((n_dims, n_vers))
    for ri, dim_id in enumerate(sorted_dims):
        for vi, v in enumerate(available_versions):
            if v in dim_data[dim_id]["versions"]:
                matrix[ri, vi] = dim_data[dim_id]["versions"][v][key] * 1000

    fig, ax = plt.subplots(figsize=(max(4, n_vers * 1.5 + 2), max(6, n_dims * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=global_max * 1000)

    for ri in range(n_dims):
        for vi in range(n_vers):
            val = matrix[ri, vi]
            text_color = "white" if val > global_max * 500 else "black"
            ax.text(vi, ri, f"{val:.2f}", ha="center", va="center", fontsize=6, color=text_color)

    labels = [CATEGORY_CHART_NAMES.get(d, dim_data[d]["name"]) for d in sorted_dims]
    cats = [dim_data[d]["category"] for d in sorted_dims]
    cat_colors = [CATEGORY_COLORS.get(c, "#333") for c in cats]

    ax.set_yticks(range(n_dims))
    ax.set_yticklabels(labels, fontsize=7)
    for tick, color in zip(ax.get_yticklabels(), cat_colors):
        tick.set_color(color)

    ax.set_xticks(range(n_vers))
    ax.set_xticklabels([VERSION_LABELS[v] for v in available_versions],
                       rotation=45, ha="right", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="R² (×10⁻³)")
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def generate_png_figures(dim_data):
    """Generate all matplotlib PNG figures. Returns dict of {name: base64_png_string}."""
    global_max = compute_global_max(dim_data)
    png_figures = {}
    png_bytes = {}

    for pt_key, pt_label in [("reading", "metacognitive"), ("control", "operational")]:
        # Category summary
        fig = _mpl_category_summary(dim_data, pt_key, global_max)
        name = f"category_summary_{pt_label}"
        png_figures[name] = fig_to_base64_png(fig)
        fig2 = _mpl_category_summary(dim_data, pt_key, global_max)
        png_bytes[name] = fig_to_png_bytes(fig2)

        # Per-category breakdowns
        breakdown_cats = [c for c in CAT_ORDER if c not in ("Control", "SysPrompt")
                          and any(dim_data[d]["category"] == c for d in dim_data)]
        for cat in breakdown_cats:
            fig = _mpl_category_breakdown(dim_data, pt_key, cat, global_max)
            name = f"breakdown_{cat.lower()}_{pt_label}"
            png_figures[name] = fig_to_base64_png(fig)
            fig2 = _mpl_category_breakdown(dim_data, pt_key, cat, global_max)
            png_bytes[name] = fig_to_png_bytes(fig2)

        # All dims
        fig = _mpl_all_dims(dim_data, pt_key, global_max)
        name = f"all_dims_{pt_label}"
        png_figures[name] = fig_to_base64_png(fig)
        fig2 = _mpl_all_dims(dim_data, pt_key, global_max)
        png_bytes[name] = fig_to_png_bytes(fig2)

        # Heatmaps
        fig = _mpl_heatmap(dim_data, pt_key, global_max)
        name = f"heatmap_{pt_label}"
        png_figures[name] = fig_to_base64_png(fig)
        fig2 = _mpl_heatmap(dim_data, pt_key, global_max)
        png_bytes[name] = fig_to_png_bytes(fig2)

    return png_figures, png_bytes, global_max


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html(dim_data, all_data, figures=None, global_max=None, png_figures=None,
                   no_baseline=False):
    """Generate the full HTML report."""

    if figures is None:
        figures, global_max = generate_figures(dim_data)

    # Helper to prefer PNG (base64) over inline SVG
    def embed_figure(name, svg_fallback):
        if png_figures and name in png_figures:
            return f'<img src="data:image/png;base64,{png_figures[name]}" style="width:100%; max-width:1100px;">'
        return svg_fallback

    # Wrap all SVG figure references through embed_figure
    png_figs = {}
    for fig_name, svg_str in figures.items():
        png_figs[fig_name] = embed_figure(fig_name, svg_str)
    figures = png_figs

    available_versions = get_available_versions(dim_data)
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    control_dims_list = [d for d in dim_data if dim_data[d]["category"] == "Control"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    gmax_display = f"{global_max*1000:.2f}"
    # Identify which dims were excluded beyond the default (dim 16)
    all_dim_ids_present = set(dim_data.keys())
    excluded_names = [DIM_NAMES.get(d, str(d)) for d in sorted(EXCLUDE_DIMS - {16})
                      if d not in all_dim_ids_present] if no_baseline else []
    # More robust: check what's missing vs full set
    full_set = set(DIM_NAMES.keys()) - {16}
    actually_excluded = sorted(full_set - all_dim_ids_present)
    excluded_name_list = [DIM_NAMES.get(d, str(d)) for d in actually_excluded]
    baseline_note = f" ({', '.join(excluded_name_list)} excluded)" if excluded_name_list else ""
    baseline_suffix = f" (excl. {', '.join(excluded_name_list)})" if excluded_name_list else ""
    ctrl_dim_names = ", ".join(dim_data[d]["name"] for d in sorted(control_dims_list))

    # Summary table rows
    summary_table_rows = ''
    for v in available_versions:
        r_mental = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_mental = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        r_ctrl = [dim_data[d]["versions"][v]["reading_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        c_ctrl = [dim_data[d]["versions"][v]["control_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        summary_table_rows += (
            "<tr><td><strong>" + VERSION_LABELS[v] + "</strong></td>"
            + f"<td>{sum(r_mental)/len(r_mental)*1000:.3f}</td>" if r_mental else "<td>-</td>"
        )
        # Rewrite properly
    # Let me build the summary table more carefully
    summary_table_rows = ''
    for v in available_versions:
        r_m = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_m = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        r_c = [dim_data[d]["versions"][v]["reading_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        c_c = [dim_data[d]["versions"][v]["control_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        rm = sum(r_m) / len(r_m) * 1000 if r_m else 0
        cm = sum(c_m) / len(c_m) * 1000 if c_m else 0
        rc = sum(r_c) / len(r_c) * 1000 if r_c else 0
        cc = sum(c_c) / len(c_c) * 1000 if c_c else 0
        summary_table_rows += (
            f"<tr><td><strong>{VERSION_LABELS[v]}</strong></td>"
            f"<td>{rm:.3f}</td><td>{cm:.3f}</td>"
            f"<td>{rc:.3f}</td><td>{cc:.3f}</td></tr>\n"
        )

    # Version descriptions
    version_desc_rows = ''.join(
        "<tr><td><strong>" + VERSION_LABELS[v] + "</strong></td><td>"
        + VERSION_DESCRIPTIONS[v] + "</td></tr>"
        for v in available_versions
    )

    # Full data table
    sorted_dims = sorted_all_dims(dim_data)
    data_table_rows = ''
    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        for v in available_versions:
            if v not in dd["versions"]:
                continue
            vd = dd["versions"][v]
            name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
            data_table_rows += (
                f'<tr><td>{name}</td><td>{dd["category"]}</td>'
                f'<td>{VERSION_LABELS[v]}</td>'
                f'<td>{vd["reading_r2"]*1000:.3f}</td>'
                f'<td>[{vd["reading_ci"][0]*1000:.3f}, {vd["reading_ci"][1]*1000:.3f}]</td>'
                f'<td>{vd["control_r2"]*1000:.3f}</td>'
                f'<td>[{vd["control_ci"][0]*1000:.3f}, {vd["control_ci"][1]*1000:.3f}]</td></tr>\n'
            )

    # Category breakdown HTML
    active_cats = [c for c in CAT_ORDER if c not in ("Control", "SysPrompt")
                   and any(dim_data[d]["category"] == c for d in dim_data)]
    cat_breakdown_html = ''
    for cat in active_cats:
        cat_dims = sorted(d for d in dim_data if dim_data[d]["category"] == cat)
        dim_list = ", ".join(dim_data[d]["name"] for d in cat_dims)
        cat_breakdown_html += f"""
<h3>{cat} Dimensions ({len(cat_dims)} dims: {dim_list})</h3>

<div class="figure-container">
<p style="font-weight:600; color:{PROBE_HEADER_COLORS['reading']}; margin-bottom:8px;">Metacognitive Probes</p>
{figures[f"breakdown_{cat.lower()}_metacognitive"]}
</div>

<div class="figure-container">
<p style="font-weight:600; color:{PROBE_HEADER_COLORS['control']}; margin-bottom:8px;">Operational Probes</p>
{figures[f"breakdown_{cat.lower()}_operational"]}
</div>
<div class="caption">Control dimensions ({ctrl_dim_names}) shown at reduced opacity
to the right of the dashed separator for reference.</div>
"""

    # Probe label for header colors
    mc_hdr = PROBE_HEADER_COLORS["reading"]
    op_hdr = PROBE_HEADER_COLORS["control"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 3: Raw Alignment{baseline_suffix} \u2014 {VERSION_LABELS["balanced_gpt"]} vs {VERSION_LABELS["nonsense_codeword"]}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ border-bottom: 3px solid #2E7D32; padding-bottom: 10px; }}
  h2 {{ color: #2E7D32; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
  h3 {{ color: #333; margin-top: 25px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
  .section {{ margin: 30px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.85em; }}
  th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: center; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .figure-container {{ margin: 20px 0; background: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; }}
  .caption {{ font-size: 0.85em; color: #555; margin-top: 10px; padding: 8px; background: #f5f5f5; border-left: 3px solid #2E7D32; }}
  .interpretation {{ background: #FFF8E1; padding: 15px; border-radius: 8px; border-left: 4px solid #FF9800; margin: 20px 0; }}
  .key-finding {{ background: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
  .method-box {{ background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Experiment 3: Raw Alignment{baseline_suffix} \u2014 {VERSION_LABELS["balanced_gpt"]} vs {VERSION_LABELS["nonsense_codeword"]}</h1>
<div class="meta">
    <p>Generated: {timestamp} | Analysis: Phase 2a (Raw Contrast Alignment)</p>
    <p>Rachel C. Metzgar, Princeton University, Graziano Lab</p>
</div>

<h2>1. Overview</h2>
<div class="section">
<p>Raw cosine alignment between human\u2013AI contrast vectors and probe weight vectors for two conditions:
<strong>{VERSION_LABELS["balanced_gpt"]}</strong> (explicit named partners) vs <strong>{VERSION_LABELS["nonsense_codeword"]}</strong> (meaningless identity labels).
Concept direction = mean(human) \u2212 mean(AI) per layer. No entity baseline subtraction.{baseline_note}</p>

<div class="method-box">
<h3>Analysis Details</h3>
<p><strong>Concept vectors</strong> (same across both versions): For each conceptual dimension, we extracted
the model's internal direction distinguishing human from AI concepts.
Computed as <code>mean(human prompts) \u2212 mean(AI prompts)</code> across layers 6\u201340 of LLaMA-2-13B-Chat.</p>
<p><strong>Alignment metric</strong>: Cosine similarity (squared = R\u00b2) between each concept direction and
each probe's weight vector, computed per layer then averaged across layers 6\u201340 (35 layers).
Bootstrap confidence intervals (1,000 iterations) computed by resampling concept prompts.</p>
<p><strong>Versions compared</strong>:</p>
<table style="width:auto; margin: 10px 0;">
{version_desc_rows}
</table>
<p><strong>Two probe types</strong>: <strong>Metacognitive probe</strong> (classifies partner identity while
reading partner messages \u2014 reflects how the model perceives its partner) and <strong>Operational probe</strong>
(captures the model's behavioral adaptation representation).</p>
</div>
</div>

<h2>2. Summary Table</h2>
<div class="section">
<table>
<tr><th>Version</th>
<th>Metacognitive R\u00b2 (Mental, \u00d710\u207b\u00b3)</th>
<th>Operational R\u00b2 (Mental, \u00d710\u207b\u00b3)</th>
<th>Metacognitive R\u00b2 (Control dims, \u00d710\u207b\u00b3)</th>
<th>Operational R\u00b2 (Control dims, \u00d710\u207b\u00b3)</th></tr>
{summary_table_rows}
</table>
<div class="caption"><strong>Table 1.</strong> Mean R\u00b2 alignment averaged across mental dimensions ({len(mental_dims)} dims)
and control dimensions ({len(control_dims_list)} dims). Values \u00d710\u207b\u00b3.</div>
</div>

<h2>3. Category Summary</h2>
<p>Dimensions averaged by category ({', '.join(active_cats)}), with control dimensions shown individually.
The dashed line separates individual control dimensions from category averages.</p>

<div class="figure-container">
<p style="font-weight:600; color:{mc_hdr}; margin-bottom:8px;">Metacognitive Probes</p>
{figures["category_summary_metacognitive"]}
<div class="caption"><strong>Figure 1.</strong> Category-level alignment with metacognitive probes.
Left of dashed line: individual control dimensions ({ctrl_dim_names}).
Right: category averages ({', '.join(f'{c} ({sum(1 for d in dim_data if dim_data[d]["category"]==c)} dims)' for c in active_cats)}).
Error bars show mean of per-dimension 95% bootstrap CIs. Uniform y-axis (max = {gmax_display} \u00d710\u207b\u00b3).</div>
</div>

<div class="figure-container">
<p style="font-weight:600; color:{op_hdr}; margin-bottom:8px;">Operational Probes</p>
{figures["category_summary_operational"]}
<div class="caption"><strong>Figure 2.</strong> Category-level alignment with operational probes. Same layout as Figure 1.</div>
</div>

<h2>4. Per-Category Breakdowns</h2>
<p>Individual dimension alignment within each category. Control dimensions ({ctrl_dim_names}) are included on each graph at reduced opacity for reference, separated by a dashed line.
All figures use the same y-axis scale (max = {gmax_display} \u00d710\u207b\u00b3) for direct comparison across categories.</p>

{cat_breakdown_html}

<h2>5. All Dimensions: Metacognitive Probes</h2>
<div class="figure-container">
{figures["all_dims_metacognitive"]}
<div class="caption"><strong>Figure 7.</strong> Per-dimension R\u00b2 alignment with metacognitive probes across both versions.
Bars = mean R\u00b2 across layers; whiskers = 95% bootstrap CI (1,000 iterations).
Uniform y-axis (max = {gmax_display} \u00d710\u207b\u00b3). Control dimensions to the left of dashed line.</div>
</div>

<h2>6. All Dimensions: Operational Probes</h2>
<div class="figure-container">
{figures["all_dims_operational"]}
<div class="caption"><strong>Figure 8.</strong> Per-dimension R\u00b2 alignment with operational probes. Same layout as Figure 7.</div>
</div>

<h2>7. Heatmaps</h2>

<div class="figure-container">
<p style="font-weight:600; color:{mc_hdr}; margin-bottom:8px;">Metacognitive Probes</p>
{figures["heatmap_metacognitive"]}
<div class="caption"><strong>Figure 9.</strong> Heatmap of metacognitive probe R\u00b2 across dimensions and versions.
Cell values are R\u00b2 \u00d710\u207b\u00b3. Color intensity proportional to alignment strength.
Green = {VERSION_LABELS["balanced_gpt"]}; gray = {VERSION_LABELS["nonsense_codeword"]}. Hover for exact values.</div>
</div>

<div class="figure-container">
<p style="font-weight:600; color:{op_hdr}; margin-bottom:8px;">Operational Probes</p>
{figures["heatmap_operational"]}
<div class="caption"><strong>Figure 10.</strong> Heatmap of operational probe R\u00b2 across dimensions and versions.
Same layout as Figure 9.</div>
</div>

<h2>8. Full Data Table</h2>
<div class="section" style="overflow-x: auto;">
<table>
<tr><th>Dimension</th><th>Category</th><th>Version</th>
<th>Metacognitive R\u00b2 (\u00d710\u207b\u00b3)</th><th>Metacognitive 95% CI</th>
<th>Operational R\u00b2 (\u00d710\u207b\u00b3)</th><th>Operational 95% CI</th></tr>
{data_table_rows}
</table>
<div class="caption"><strong>Table 2.</strong> Complete per-dimension alignment statistics. R\u00b2 and 95% bootstrap CIs \u00d710\u207b\u00b3.</div>
</div>

<h2>9. Methods</h2>
<div class="method-box">
<p><strong>Model</strong>: LLaMA-2-13B-Chat</p>
<p><strong>Probes</strong>: Logistic linear probes (5,120 \u2192 1) from Exp 2, turn 5.</p>
<p><strong>Layer range</strong>: Layers 6\u201340 (35 of 41). Layers 0\u20135 excluded (embedding artifact + prompt-format confound).</p>
<p><strong>Bootstrap</strong>: 1,000 iterations, prompt-level resampling.</p>
<p><strong>Software</strong>: PyTorch, NumPy. Script: <code>exp_3/code/2a_alignment_analysis.py</code></p>
</div>

</body>
</html>"""
    return html


# ============================================================================
# MARKDOWN GENERATION
# ============================================================================

def generate_markdown(dim_data, all_data, no_baseline=False):
    """Generate a concise markdown summary."""
    available_versions = get_available_versions(dim_data)
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    baseline_suffix = " (No Baseline)" if no_baseline else ""

    lines = []
    lines.append(f"# Exp 3: Raw Alignment{baseline_suffix} \u2014 Partner Identity vs Control")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Version | Metacognitive R\u00b2 (Mental, \u00d710\u207b\u00b3) | Operational R\u00b2 (Mental, \u00d710\u207b\u00b3) | Description |")
    lines.append("|---------|---|---|---|")

    for v in available_versions:
        r_vals = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_vals = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        r_mean = sum(r_vals) / len(r_vals) * 1000 if r_vals else 0
        c_mean = sum(c_vals) / len(c_vals) * 1000 if c_vals else 0
        lines.append(f"| {VERSION_LABELS[v]} | {r_mean:.3f} | {c_mean:.3f} | {VERSION_DESCRIPTIONS[v]} |")

    lines.append("")
    lines.append("## Per-Dimension Data (Metacognitive Probe R\u00b2 \u00d710\u207b\u00b3)")
    lines.append("")

    header = "| Dimension | Category | " + " | ".join(VERSION_LABELS[v] for v in available_versions) + " |"
    sep = "|---|---|" + "|".join("---" for _ in available_versions) + "|"
    lines.append(header)
    lines.append(sep)

    sorted_dims = sorted_all_dims(dim_data)

    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
        vals = [f"{dd['versions'][v]['reading_r2']*1000:.3f}" if v in dd["versions"] else "\u2014"
                for v in available_versions]
        lines.append(f"| {name} | {dd['category']} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Per-Dimension Data (Operational Probe R\u00b2 \u00d710\u207b\u00b3)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        name = CATEGORY_CHART_NAMES.get(dim_id, dd["name"])
        vals = [f"{dd['versions'][v]['control_r2']*1000:.3f}" if v in dd["versions"] else "\u2014"
                for v in available_versions]
        lines.append(f"| {name} | {dd['category']} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Partner Identity version shows substantially higher alignment** than the Control version across nearly all dimensions")
    lines.append("2. **Metacognitive and operational probes** may show different alignment patterns, reflecting distinct aspects of partner encoding")
    lines.append("3. **Control version shows near-floor alignment**, comparable to the shapes negative control dimension")
    lines.append("4. **This is raw alignment** \u2014 residual analysis (projecting out entity baseline) needed to assess concept-specific contribution")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("- **Metric**: Mean R\u00b2 (cosine similarity squared) between concept direction vectors and probe weight vectors, averaged across 35 layers (6\u201340)")
    lines.append("- **Bootstrap**: 1,000 iterations (prompt resampling)")
    lines.append("- **Model**: LLaMA-2-13B-Chat")
    lines.append("- **Concept vectors**: 18 dimensions, ~80 prompts each (contrasts mode: human vs AI)")
    lines.append("- **Probes**: From Exp 2, metacognitive + operational probes (logistic, per-layer)")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate cross-version raw alignment comparison report")
    parser.add_argument("--turn", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Conversation turn (default: 5)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Exclude baseline entity dimension (dim 0) from all data and charts")
    parser.add_argument("--exclude", type=int, nargs="*", default=[],
                        help="Additional dimension IDs to exclude (e.g. --exclude 10 for animacy)")
    args = parser.parse_args()

    # Build exclusion set
    exclude = set(EXCLUDE_DIMS)
    if args.no_baseline:
        exclude.add(BASELINE_DIM)
    for d in args.exclude:
        exclude.add(d)
    extra_excluded = sorted(exclude - EXCLUDE_DIMS)
    if extra_excluded:
        names = [DIM_NAMES.get(d, str(d)) for d in extra_excluded]
        print(f"[NOTE] Excluding dimensions: {', '.join(names)}")

    print(f"Loading alignment summaries (turn {args.turn})...")
    all_data = load_all_summaries(turn=args.turn)
    print(f"  Loaded {len(all_data)} versions: {list(all_data.keys())}")

    if not all_data:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    print("Organizing data...")
    dim_data = organize_data(all_data, exclude_dims=exclude)
    excluded_str = ", ".join(str(d) for d in sorted(exclude))
    print(f"  {len(dim_data)} dimensions (excluding dims {excluded_str})")

    comparisons_root = config.RESULTS.comparisons / "alignment"
    out_dir = comparisons_root / f"turn_{args.turn}" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    # File name suffix for exclusion variants
    has_exclusions = bool(extra_excluded)
    if has_exclusions:
        suffix_parts = [DIM_NAMES.get(d, str(d)).lower().replace(" ", "_").replace("(", "").replace(")", "")
                        for d in extra_excluded]
        suffix = "_no_" + "_".join(suffix_parts)
    else:
        suffix = ""
    fig_subdir = f"figures{suffix}" if has_exclusions else "figures"

    # Generate and save SVG figures
    print("Generating SVG figures...")
    figures, global_max = generate_figures(dim_data)
    fig_dir = out_dir / fig_subdir
    fig_dir.mkdir(parents=True, exist_ok=True)
    for fig_name, svg_str in figures.items():
        fig_path = fig_dir / f"{fig_name}.svg"
        with open(fig_path, "w") as f:
            f.write(svg_str)

    # Generate and save PNG figures (matplotlib)
    print("Generating PNG figures...")
    png_figures, png_bytes, _ = generate_png_figures(dim_data)
    for fig_name, png_data in png_bytes.items():
        fig_path = fig_dir / f"{fig_name}.png"
        with open(fig_path, "wb") as f:
            f.write(png_data)
    print(f"  Figures: {fig_dir}/ ({len(figures)} SVGs + {len(png_bytes)} PNGs)")

    print("Generating HTML report...")
    html = generate_html(dim_data, all_data, figures=figures, global_max=global_max,
                         png_figures=png_figures, no_baseline=has_exclusions)
    html_path = out_dir / f"raw_detailed_comparison{suffix}.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Saved: {html_path}")

    print("Generating Markdown summary...")
    md = generate_markdown(dim_data, all_data, no_baseline=has_exclusions)
    md_path = out_dir / f"raw_detailed_comparison{suffix}.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Saved: {md_path}")

    print(f"\nDone! {len(figures)} figures generated.")


if __name__ == "__main__":
    main()
