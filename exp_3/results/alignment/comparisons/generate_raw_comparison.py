#!/usr/bin/env python3
"""
Generate cross-version raw alignment comparison report (HTML + MD).

Reads summary.json from each version's contrasts/raw/ directory and produces:
  - raw_comparison.html  (interactive report with embedded SVG figures)
  - raw_comparison.md    (markdown summary)

No external dependencies — uses only Python standard library.

Usage:
    python generate_raw_comparison.py

Rachel C. Metzgar · Feb 2026
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent / "versions"
VERSIONS = ["labels", "balanced_names", "balanced_gpt", "names",
            "nonsense_codeword", "nonsense_ignore",
            "labels_turnwise", "you_are_labels",
            "you_are_balanced_gpt", "you_are_labels_turnwise"]

VERSION_LABELS = {
    "labels": "Labels",
    "balanced_names": "Balanced Names",
    "balanced_gpt": "Balanced GPT",
    "names": "Names (orig.)",
    "nonsense_codeword": "Nonsense Codeword",
    "nonsense_ignore": "Nonsense Ignore",
    "labels_turnwise": "Labels + Turnwise",
    "you_are_labels": "You-Are Labels",
    "you_are_balanced_gpt": "You-Are Bal. GPT",
    "you_are_labels_turnwise": "You-Are Lab. Turn.",
}

VERSION_DESCRIPTIONS = {
    "labels": "Partner identified as 'human' or 'AI' (no names)",
    "balanced_names": "Gender-balanced names (e.g., Alex/Jordan)",
    "balanced_gpt": "Balanced names with GPT-4 replacing 'AI' partner",
    "names": "Original Sam/Casey names (deprecated due to name confound)",
    "nonsense_codeword": "Nonsense codewords replacing identity labels",
    "nonsense_ignore": "Nonsense labels with instruction to ignore them",
    "labels_turnwise": "Labels + turn-level 'Human:'/'AI:' prefix each turn",
    "you_are_labels": "'You are talking to a Human/an AI' framing",
    "you_are_balanced_gpt": "'You are talking to' + named partners (Gregory/Rebecca, ChatGPT/GPT-4)",
    "you_are_labels_turnwise": "'You are talking to' framing + turn-level prefix",
}

# Dimension display info
DIM_CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 17],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Control":   [0, 14, 15],
    "SysPrompt": [18],
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
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
}

# Exclude dim 16 (circular: pooled from other dims)
EXCLUDE_DIMS = {16}

VERSION_COLORS = {
    "labels": "#E53935",
    "balanced_names": "#1E88E5",
    "balanced_gpt": "#43A047",
    "names": "#FB8C00",
    "nonsense_codeword": "#8E24AA",
    "nonsense_ignore": "#546E7A",
    "labels_turnwise": "#D81B60",
    "you_are_labels": "#00ACC1",
    "you_are_balanced_gpt": "#7CB342",
    "you_are_labels_turnwise": "#5E35B1",
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_summaries():
    """Load summary.json for each version. Returns dict[version] = summary."""
    data = {}
    for v in VERSIONS:
        path = ALIGNMENT_ROOT / v / "contrasts" / "raw" / "summary.json"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {v}")
            continue
        with open(path) as f:
            data[v] = json.load(f)
    return data


def get_dim_id_from_name(dim_name):
    """Extract numeric ID from dimension folder name like '1_phenomenology'."""
    return int(dim_name.split("_", 1)[0])


def organize_data(all_data):
    """
    Organize into a structure:
        result[dim_id] = {
            "name": "Phenomenology",
            "category": "Mental",
            "versions": {
                "labels": {"reading_r2": ..., "control_r2": ..., "reading_ci": [...], "control_ci": [...]},
                ...
            }
        }
    """
    result = {}
    for version, summary in all_data.items():
        for dim_folder, stats in summary.items():
            dim_id = stats["dim_id"]
            if dim_id in EXCLUDE_DIMS:
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


# ============================================================================
# SVG CHART GENERATION
# ============================================================================

def svg_grouped_bar(dim_data, probe_type, global_max, chart_width=1100, chart_height=340):
    """
    Generate an SVG grouped bar chart for one probe type across all versions.

    Args:
        dim_data: organized data dict (dim_id -> {...})
        probe_type: "reading" or "control"
        global_max: max R² value for uniform y-axis
        chart_width, chart_height: dimensions
    """
    key = f"{probe_type}_r2"
    ci_key = f"{probe_type}_ci"

    # Sort dims by category then ID
    cat_order = ["Mental", "Physical", "Pragmatic", "Control", "SysPrompt"]
    sorted_dims = sorted(dim_data.keys(),
                         key=lambda d: (cat_order.index(dim_data[d]["category"])
                                        if dim_data[d]["category"] in cat_order else 99, d))

    n_dims = len(sorted_dims)
    n_versions = len(VERSIONS)
    available_versions = [v for v in VERSIONS if v in list(dim_data.values())[0]["versions"]]
    n_av = len(available_versions)

    margin = {"top": 30, "right": 20, "bottom": 120, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]

    group_width = plot_w / n_dims
    bar_width = group_width * 0.8 / n_av
    gap = group_width * 0.1

    y_max = global_max * 1.1  # 10% headroom

    lines = []
    lines.append(f'<svg viewBox="0 0 {chart_width} {chart_height}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="width:100%;max-width:{chart_width}px;height:auto;font-family:Arial,sans-serif;">')

    # Background
    lines.append(f'<rect width="{chart_width}" height="{chart_height}" fill="white"/>')

    # Plot area
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')

    # Y-axis gridlines and labels
    n_ticks = 5
    for i in range(n_ticks + 1):
        y_val = y_max * i / n_ticks
        y_pos = plot_h - (y_val / y_max * plot_h)
        lines.append(f'<line x1="0" y1="{y_pos:.1f}" x2="{plot_w}" y2="{y_pos:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        label = f"{y_val*1000:.2f}" if y_val < 0.01 else f"{y_val:.4f}"
        lines.append(f'<text x="-8" y="{y_pos+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#666">{label}</text>')

    # Y-axis label
    lines.append(f'<text x="-50" y="{plot_h/2}" text-anchor="middle" '
                 f'font-size="12" fill="#333" '
                 f'transform="rotate(-90,-50,{plot_h/2})">Mean R² (×10⁻³)</text>')

    # Bars
    for gi, dim_id in enumerate(sorted_dims):
        dd = dim_data[dim_id]
        x_group = gi * group_width + gap

        for vi, version in enumerate(available_versions):
            if version not in dd["versions"]:
                continue
            vd = dd["versions"][version]
            val = vd[key]
            ci = vd[ci_key]

            x = x_group + vi * bar_width
            bar_h = val / y_max * plot_h
            y = plot_h - bar_h

            color = VERSION_COLORS[version]
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                         f'height="{bar_h:.2f}" fill="{color}" opacity="0.85">'
                         f'<title>{VERSION_LABELS[version]}: {val:.6f}</title></rect>')

            # CI whisker
            ci_lo_h = ci[0] / y_max * plot_h
            ci_hi_h = ci[1] / y_max * plot_h
            cx = x + bar_width / 2
            y_lo = plot_h - ci_lo_h
            y_hi = plot_h - ci_hi_h
            lines.append(f'<line x1="{cx:.2f}" y1="{y_lo:.2f}" x2="{cx:.2f}" '
                         f'y2="{y_hi:.2f}" stroke="#333" stroke-width="1"/>')
            lines.append(f'<line x1="{cx-2:.2f}" y1="{y_lo:.2f}" x2="{cx+2:.2f}" '
                         f'y2="{y_lo:.2f}" stroke="#333" stroke-width="1"/>')
            lines.append(f'<line x1="{cx-2:.2f}" y1="{y_hi:.2f}" x2="{cx+2:.2f}" '
                         f'y2="{y_hi:.2f}" stroke="#333" stroke-width="1"/>')

        # X-axis label
        x_center = x_group + (n_av * bar_width) / 2
        cat = dd["category"]
        cat_color = CATEGORY_COLORS.get(cat, "#333")
        lines.append(f'<text x="{x_center:.2f}" y="{plot_h + 14}" text-anchor="end" '
                     f'font-size="9" fill="{cat_color}" '
                     f'transform="rotate(-45,{x_center:.2f},{plot_h + 14})">'
                     f'{dd["name"]}</text>')

    # X-axis line
    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" '
                 f'stroke="#333" stroke-width="1"/>')
    # Y-axis line
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333" stroke-width="1"/>')

    lines.append('</g>')

    # Legend (2 rows for many versions)
    cols_per_row = 5
    col_spacing = (chart_width - margin["left"] - margin["right"]) / cols_per_row
    for i, v in enumerate(available_versions):
        row = i // cols_per_row
        col = i % cols_per_row
        x = margin["left"] + col * col_spacing
        y = chart_height - 28 + row * 16
        lines.append(f'<rect x="{x}" y="{y-8}" width="10" height="10" '
                     f'fill="{VERSION_COLORS[v]}"/>')
        lines.append(f'<text x="{x+14}" y="{y}" font-size="8" fill="#333">'
                     f'{VERSION_LABELS[v]}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def svg_heatmap(dim_data, probe_type, global_max, chart_width=1100, row_height=28):
    """Generate an SVG heatmap: rows = dimensions, columns = versions."""
    key = f"{probe_type}_r2"

    cat_order = ["Mental", "Physical", "Pragmatic", "Control", "SysPrompt"]
    sorted_dims = sorted(dim_data.keys(),
                         key=lambda d: (cat_order.index(dim_data[d]["category"])
                                        if dim_data[d]["category"] in cat_order else 99, d))

    available_versions = [v for v in VERSIONS if v in list(dim_data.values())[0]["versions"]]
    n_dims = len(sorted_dims)
    n_vers = len(available_versions)

    margin = {"top": 140, "right": 80, "bottom": 20, "left": 140}
    cell_w = (chart_width - margin["left"] - margin["right"]) / n_vers
    chart_height = margin["top"] + n_dims * row_height + margin["bottom"]

    lines = []
    lines.append(f'<svg viewBox="0 0 {chart_width} {chart_height}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="width:100%;max-width:{chart_width}px;height:auto;font-family:Arial,sans-serif;">')
    lines.append(f'<rect width="{chart_width}" height="{chart_height}" fill="white"/>')

    # Column headers (version names, rotated)
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

        # Category separator
        if cat != prev_cat and prev_cat is not None:
            lines.append(f'<line x1="{margin["left"]}" y1="{y}" '
                         f'x2="{margin["left"] + n_vers * cell_w}" y2="{y}" '
                         f'stroke="#999" stroke-width="1.5"/>')
        prev_cat = cat

        # Row label
        cat_color = CATEGORY_COLORS.get(cat, "#333")
        lines.append(f'<text x="{margin["left"]-8}" y="{y + row_height/2 + 4}" '
                     f'text-anchor="end" font-size="10" fill="{cat_color}">'
                     f'{dd["name"]}</text>')

        # Cells
        for vi, v in enumerate(available_versions):
            if v not in dd["versions"]:
                continue
            val = dd["versions"][v][key]
            intensity = min(val / global_max, 1.0)

            # Blue color scale
            r = int(255 - intensity * 200)
            g = int(255 - intensity * 150)
            b = 255

            x = margin["left"] + vi * cell_w
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" '
                         f'height="{row_height}" fill="rgb({r},{g},{b})" '
                         f'stroke="white" stroke-width="1">'
                         f'<title>{VERSION_LABELS[v]} — {dd["name"]}: R²={val:.6f}</title></rect>')

            # Value text
            text_color = "#fff" if intensity > 0.5 else "#333"
            lines.append(f'<text x="{x + cell_w/2:.1f}" y="{y + row_height/2 + 4:.1f}" '
                         f'text-anchor="middle" font-size="8" fill="{text_color}">'
                         f'{val*1000:.2f}</text>')

    # Colorbar
    cb_x = margin["left"] + n_vers * cell_w + 15
    cb_h = n_dims * row_height
    cb_w = 15
    for i in range(50):
        frac = i / 49
        r = int(255 - frac * 200)
        g = int(255 - frac * 150)
        y = margin["top"] + cb_h - (frac * cb_h)
        lines.append(f'<rect x="{cb_x}" y="{y:.1f}" width="{cb_w}" height="{cb_h/49+1:.1f}" '
                     f'fill="rgb({r},{g},255)"/>')
    lines.append(f'<text x="{cb_x + cb_w + 4}" y="{margin["top"] + 4}" '
                 f'font-size="8" fill="#333">{global_max*1000:.2f}</text>')
    lines.append(f'<text x="{cb_x + cb_w + 4}" y="{margin["top"] + cb_h + 4}" '
                 f'font-size="8" fill="#333">0.00</text>')
    lines.append(f'<text x="{cb_x + cb_w + 4}" y="{margin["top"] + cb_h/2 + 4}" '
                 f'font-size="8" fill="#666">×10⁻³</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def svg_version_summary_bar(dim_data, chart_width=1100, chart_height=340):
    """Bar chart: mean R² across all mental dims, per version, reading vs control side by side."""
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    available_versions = [v for v in VERSIONS if v in list(dim_data.values())[0]["versions"]]

    version_means = {}
    for v in available_versions:
        reading_vals = [dim_data[d]["versions"][v]["reading_r2"]
                        for d in mental_dims if v in dim_data[d]["versions"]]
        control_vals = [dim_data[d]["versions"][v]["control_r2"]
                        for d in mental_dims if v in dim_data[d]["versions"]]
        version_means[v] = {
            "reading": sum(reading_vals) / len(reading_vals) if reading_vals else 0,
            "control": sum(control_vals) / len(control_vals) if control_vals else 0,
        }

    global_max = max(max(vm["reading"], vm["control"]) for vm in version_means.values()) * 1.15

    margin = {"top": 30, "right": 20, "bottom": 80, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]
    group_w = plot_w / len(available_versions)
    bar_w = group_w * 0.35

    lines = []
    lines.append(f'<svg viewBox="0 0 {chart_width} {chart_height}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="width:100%;max-width:{chart_width}px;height:auto;font-family:Arial,sans-serif;">')
    lines.append(f'<rect width="{chart_width}" height="{chart_height}" fill="white"/>')
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')

    # Gridlines
    for i in range(6):
        y_val = global_max * i / 5
        y_pos = plot_h - (y_val / global_max * plot_h)
        lines.append(f'<line x1="0" y1="{y_pos:.1f}" x2="{plot_w}" y2="{y_pos:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="-8" y="{y_pos+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#666">{y_val*1000:.2f}</text>')

    lines.append(f'<text x="-50" y="{plot_h/2}" text-anchor="middle" '
                 f'font-size="12" fill="#333" '
                 f'transform="rotate(-90,-50,{plot_h/2})">Mean R² (×10⁻³)</text>')

    for gi, v in enumerate(available_versions):
        vm = version_means[v]
        x_group = gi * group_w + group_w * 0.15

        # Reading bar
        h_r = vm["reading"] / global_max * plot_h
        lines.append(f'<rect x="{x_group:.1f}" y="{plot_h - h_r:.1f}" '
                     f'width="{bar_w:.1f}" height="{h_r:.1f}" fill="#2196F3" opacity="0.8">'
                     f'<title>Reading: {vm["reading"]:.6f}</title></rect>')

        # Control bar
        x_c = x_group + bar_w + 2
        h_c = vm["control"] / global_max * plot_h
        lines.append(f'<rect x="{x_c:.1f}" y="{plot_h - h_c:.1f}" '
                     f'width="{bar_w:.1f}" height="{h_c:.1f}" fill="#FF5722" opacity="0.8">'
                     f'<title>Control: {vm["control"]:.6f}</title></rect>')

        # Label
        x_center = x_group + bar_w
        lines.append(f'<text x="{x_center:.1f}" y="{plot_h+14}" text-anchor="end" '
                     f'font-size="10" fill="#333" '
                     f'transform="rotate(-35,{x_center:.1f},{plot_h+14})">'
                     f'{VERSION_LABELS[v]}</text>')

    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333"/>')
    lines.append('</g>')

    # Legend
    ly = chart_height - 12
    lines.append(f'<rect x="{margin["left"]}" y="{ly-8}" width="10" height="10" fill="#2196F3"/>')
    lines.append(f'<text x="{margin["left"]+14}" y="{ly}" font-size="10">Reading Probe</text>')
    lines.append(f'<rect x="{margin["left"]+120}" y="{ly-8}" width="10" height="10" fill="#FF5722"/>')
    lines.append(f'<text x="{margin["left"]+134}" y="{ly}" font-size="10">Control Probe</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html(dim_data, all_data):
    """Generate the full HTML report."""

    # Compute global max for uniform scales
    all_r2 = []
    for dd in dim_data.values():
        for vd in dd["versions"].values():
            all_r2.append(vd["reading_r2"])
            all_r2.append(vd["control_r2"])
            all_r2.extend(vd["reading_ci"])
            all_r2.extend(vd["control_ci"])
    global_max = max(all_r2) * 1.05

    # Generate figures
    reading_bars = svg_grouped_bar(dim_data, "reading", global_max)
    control_bars = svg_grouped_bar(dim_data, "control", global_max)
    reading_heatmap = svg_heatmap(dim_data, "reading", global_max)
    control_heatmap = svg_heatmap(dim_data, "control", global_max)
    summary_bars = svg_version_summary_bar(dim_data)

    # Compute summary stats for table
    available_versions = [v for v in VERSIONS if v in list(dim_data.values())[0]["versions"]]
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    control_dims_list = [d for d in dim_data if dim_data[d]["category"] == "Control"]

    summary_rows = []
    for v in available_versions:
        r_mental = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_mental = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        r_ctrl = [dim_data[d]["versions"][v]["reading_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        c_ctrl = [dim_data[d]["versions"][v]["control_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]

        summary_rows.append({
            "version": v,
            "label": VERSION_LABELS[v],
            "reading_mental_mean": sum(r_mental) / len(r_mental) if r_mental else 0,
            "control_mental_mean": sum(c_mental) / len(c_mental) if c_mental else 0,
            "reading_control_mean": sum(r_ctrl) / len(r_ctrl) if r_ctrl else 0,
            "control_control_mean": sum(c_ctrl) / len(c_ctrl) if c_ctrl else 0,
        })

    # Full data table
    cat_order = ["Mental", "Physical", "Pragmatic", "Control", "SysPrompt"]
    sorted_dims = sorted(dim_data.keys(),
                         key=lambda d: (cat_order.index(dim_data[d]["category"])
                                        if dim_data[d]["category"] in cat_order else 99, d))

    data_table_rows = []
    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        for v in available_versions:
            if v not in dd["versions"]:
                continue
            vd = dd["versions"][v]
            data_table_rows.append(
                f'<tr><td>{dd["name"]}</td><td>{dd["category"]}</td>'
                f'<td>{VERSION_LABELS[v]}</td>'
                f'<td>{vd["reading_r2"]*1000:.3f}</td>'
                f'<td>[{vd["reading_ci"][0]*1000:.3f}, {vd["reading_ci"][1]*1000:.3f}]</td>'
                f'<td>{vd["control_r2"]*1000:.3f}</td>'
                f'<td>[{vd["control_ci"][0]*1000:.3f}, {vd["control_ci"][1]*1000:.3f}]</td></tr>'
            )

    # Pre-compute dynamic HTML fragments outside the f-string
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    global_max_display = f"{global_max*1000:.2f}"

    version_desc_rows = ''.join(
        "<tr><td><strong>" + VERSION_LABELS[v] + "</strong></td><td>" + VERSION_DESCRIPTIONS[v] + "</td></tr>"
        for v in available_versions
    )

    summary_table_rows = ''
    for r in summary_rows:
        summary_table_rows += (
            "<tr>"
            "<td><strong>" + r["label"] + "</strong></td>"
            "<td>" + f'{r["reading_mental_mean"]*1000:.3f}' + "</td>"
            "<td>" + f'{r["control_mental_mean"]*1000:.3f}' + "</td>"
            "<td>" + f'{r["reading_control_mean"]*1000:.3f}' + "</td>"
            "<td>" + f'{r["control_control_mean"]*1000:.3f}' + "</td>"
            "</tr>\n"
        )

    data_table_html = ''.join(data_table_rows)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 3: Raw Alignment Comparison Across Data Versions</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
  h2 {{ color: #1565C0; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
  h3 {{ color: #333; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
  .section {{ margin: 30px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.85em; }}
  th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: center; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .figure-container {{ margin: 20px 0; background: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; }}
  .caption {{ font-size: 0.85em; color: #555; margin-top: 10px; padding: 8px; background: #f5f5f5; border-left: 3px solid #2196F3; }}
  .interpretation {{ background: #FFF8E1; padding: 15px; border-radius: 8px; border-left: 4px solid #FF9800; margin: 20px 0; }}
  .key-finding {{ background: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
  .method-box {{ background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .highlight {{ font-weight: 600; color: #1565C0; }}
  .version-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; color: white; font-size: 0.8em; margin: 1px; }}
</style>
</head>
<body>

<h1>Experiment 3: Raw Alignment Comparison Across Data Versions</h1>
<div class="meta">
    <p>Generated: {timestamp} | Analysis: Phase 2a (Raw Contrast Alignment)</p>
    <p>Rachel C. Metzgar, Princeton University, Graziano Lab</p>
</div>

<h2>1. Overview</h2>

<div class="section">
<p>This report compares the <strong>raw alignment</strong> between concept-of-mind vectors (Experiment 3) and
conversational partner-identity probes (Experiment 2) across six different data versions. The central question
is whether the model's general semantic knowledge about human and AI minds (concept vectors) aligns with
the representations it uses to distinguish partner identity during conversation (Exp 2 probes), and how this
alignment varies depending on how partner identity was communicated.</p>

<div class="method-box">
<h3>Analysis Logic</h3>
<p><strong>Concept vectors</strong> (same across all versions): For each conceptual dimension (phenomenology,
emotions, agency, etc.), we extracted the model's internal direction distinguishing human from AI concepts.
These are computed as <code>mean(human prompts) - mean(AI prompts)</code> across all 41 layers of LLaMA-2-13B-Chat.</p>

<p><strong>Conversational probes</strong> (version-specific): Linear probes were trained in Exp 2 to classify
whether the model was talking to a human or AI partner. Each version used a different method to communicate
partner identity:</p>

<table style="width:auto; margin: 10px 0;">
{version_desc_rows}
</table>

<p><strong>Alignment metric</strong>: Cosine similarity (squared = R²) between each concept direction and each
probe's weight vector, computed per layer then averaged across all 41 layers. Bootstrap confidence intervals
(1,000 iterations) are computed by resampling concept prompts.</p>

<p><strong>Two probe types</strong>:</p>
<ul>
<li><strong>Reading probe</strong>: Trained to classify partner identity from the model's hidden states while
<em>reading</em> partner messages. Reflects how the model perceives/encodes its partner.</li>
<li><strong>Control probe</strong>: Trained on the same task but with a different training objective. Reflects
the model's representation of its own behavioral adaptation.</li>
</ul>

<p><strong>Key controls</strong>:</p>
<ul>
<li><strong>Dim 0 (Baseline)</strong>: Entity labels only ("this is a human/AI") with no conceptual content. Alignment should reflect surface-level identity encoding.</li>
<li><strong>Dim 15 (Shapes)</strong>: Round vs angular shapes — semantically irrelevant negative control. Alignment should be near zero.</li>
<li><strong>Dim 14 (Biological)</strong>: Biological features — partially related control.</li>
</ul>
</div>
</div>

<h2>2. Summary: Mean Alignment by Version</h2>

<div class="section">
<table>
<tr>
<th>Version</th>
<th>Reading R² (Mental dims, ×10⁻³)</th>
<th>Control R² (Mental dims, ×10⁻³)</th>
<th>Reading R² (Control dims, ×10⁻³)</th>
<th>Control R² (Control dims, ×10⁻³)</th>
</tr>
{summary_table_rows}
</table>
<div class="caption"><strong>Table 1.</strong> Mean R² alignment between concept-of-mind vectors and Exp 2 conversational probes,
averaged across mental dimensions (dims 1-7, 17) and control dimensions (dims 0, 14, 15), for each data version.
Values are reported as ×10⁻³ for readability. Higher values indicate stronger geometric alignment between conceptual
and conversational representations. Reading and control probes capture complementary aspects of partner-identity encoding.</div>
</div>

<h2>3. Version Summary (Mental Dimensions Only)</h2>

<div class="figure-container">
{summary_bars}
<div class="caption"><strong>Figure 1.</strong> Mean alignment (R²) between concept-of-mind vectors and conversational probes,
averaged across all eight mental dimensions (phenomenology, emotions, agency, intentions, prediction, cognitive, social, attention).
Blue bars = reading probes; orange bars = control probes. Versions with explicit identity cues (<em>balanced_gpt</em>,
<em>names</em>, <em>balanced_names</em>) show substantially higher alignment than versions with abstract labels or
nonsense cues. All figures use the same scale for direct comparison.</div>
</div>

<h2>4. Per-Dimension Alignment: Reading Probes</h2>

<div class="figure-container">
{reading_bars}
<div class="caption"><strong>Figure 2.</strong> Per-dimension R² alignment between concept vectors and <strong>reading probes</strong>
across all six data versions. Bars show mean R² across layers; whiskers show 95% bootstrap confidence intervals (1,000 iterations).
Dimensions are grouped by category (Mental, Physical, Pragmatic, Control, SysPrompt) and color-coded by version.
Y-axis scale is uniform across Figures 2-5 (max = {global_max_display} ×10⁻³) for direct comparison.</div>
</div>

<h2>5. Per-Dimension Alignment: Control Probes</h2>

<div class="figure-container">
{control_bars}
<div class="caption"><strong>Figure 3.</strong> Per-dimension R² alignment between concept vectors and <strong>control probes</strong>
across all six data versions. Same layout as Figure 2. Note that control probe alignment is generally higher than reading probe
alignment for the name-based versions (balanced_gpt, names, balanced_names), suggesting control probes capture more of the
conceptual structure of partner identity.</div>
</div>

<h2>6. Heatmap: Reading Probe Alignment</h2>

<div class="figure-container">
{reading_heatmap}
<div class="caption"><strong>Figure 4.</strong> Heatmap of reading probe R² alignment across all dimensions and versions.
Cell values are R² ×10⁻³. Color intensity is proportional to alignment strength on a uniform scale (0 to {global_max_display} ×10⁻³).
Rows are grouped by dimension category; columns are data versions. Hover over cells for exact values.</div>
</div>

<h2>7. Heatmap: Control Probe Alignment</h2>

<div class="figure-container">
{control_heatmap}
<div class="caption"><strong>Figure 5.</strong> Heatmap of control probe R² alignment across all dimensions and versions.
Same layout and color scale as Figure 4. The clear difference between name-based versions (left columns, darker blue)
and nonsense/label versions (right columns, lighter) is readily apparent.</div>
</div>

<h2>8. Full Data Table</h2>

<div class="section" style="overflow-x: auto;">
<table>
<tr><th>Dimension</th><th>Category</th><th>Version</th>
<th>Reading R² (×10⁻³)</th><th>Reading 95% CI</th>
<th>Control R² (×10⁻³)</th><th>Control 95% CI</th></tr>
{data_table_html}
</table>
<div class="caption"><strong>Table 2.</strong> Complete per-dimension alignment statistics for all versions. R² values
and 95% bootstrap confidence intervals (1,000 iterations) are reported as ×10⁻³. CI computed by resampling human and
AI concept prompts with replacement and recomputing the concept direction vector at each iteration.</div>
</div>

<h2>9. Interpretation</h2>

<div class="key-finding">
<h3>Key Finding: Alignment Scales with Identity Cue Specificity</h3>
<p>The data reveal a clear ordering of alignment strength that tracks the specificity of partner identity cues:</p>
<ol>
<li><strong>balanced_gpt</strong> (highest): Explicit name-based identity with GPT-4 as the AI partner, providing the richest identity signal.</li>
<li><strong>names</strong>: Original Sam/Casey names — strong signal but potentially confounded by name-specific features.</li>
<li><strong>balanced_names</strong>: Gender-balanced names — moderately strong signal.</li>
<li><strong>labels / nonsense_codeword / nonsense_ignore</strong> (lowest): All show near-floor alignment,
with R² values that are an order of magnitude lower than the name-based versions.</li>
</ol>
</div>

<div class="interpretation">
<h3>What This Means</h3>
<p><strong>For versions with explicit identity cues</strong> (balanced_gpt, names, balanced_names):
The model's conceptual knowledge about human and AI minds is geometrically aligned with the partner-identity
representations it forms during conversation. This suggests the model draws on its general semantic knowledge —
not just surface-level label detection — when adapting its behavior to different partners.</p>

<p><strong>For versions with abstract/nonsense cues</strong> (labels, nonsense_codeword, nonsense_ignore):
Near-zero alignment means the conversational probes trained on these versions do <em>not</em> align with
concept-of-mind directions. This is consistent with two (non-exclusive) interpretations:</p>
<ul>
<li>The probes in these versions may rely on shallow pattern matching (detecting the label/codeword) rather than
activating deep conceptual representations of what humans and AIs are like.</li>
<li>The identity signal in these versions may be too weak or too abstract to recruit the same conceptual
representations that name-based identity cues activate.</li>
</ul>

<p><strong>Nonsense controls confirm the pattern</strong>: Both nonsense versions show alignment comparable to
the negative control (shapes), confirming that alignment is not a statistical artifact of probe training but
requires meaningful identity information to emerge.</p>

<p><strong>Control probes show higher alignment than reading probes</strong> for name-based versions, suggesting
the model's behavioral adaptation mechanism (captured by control probes) is more strongly coupled to conceptual
knowledge than its partner-perception mechanism (reading probes).</p>
</div>

<div class="interpretation">
<h3>Caveats</h3>
<ul>
<li>R² values are small in absolute terms (×10⁻³), which is expected for alignment in a 5,120-dimensional space.
The meaningful comparison is <em>relative</em> differences between versions and between mental vs control dimensions.</li>
<li>This is <strong>raw</strong> alignment — it includes shared variance from entity-level features.
The residual alignment analysis (projecting out the entity baseline) will reveal concept-specific alignment.</li>
<li>Bootstrap CIs are computed from prompt resampling, not across subjects, so they reflect prompt variability
rather than population generalizability.</li>
</ul>
</div>

<h2>10. Methods Summary</h2>

<div class="method-box">
<p><strong>Model</strong>: LLaMA-2-13B-Chat (meta-llama/Llama-2-13b-chat-hf)</p>
<p><strong>Concept extraction</strong>: 19 dimensions × ~80 prompts per dimension (40 human + 40 AI, contrasts mode).
Concept direction = mean(human activations) - mean(AI activations) per layer.</p>
<p><strong>Probes</strong>: Linear logistic probes (5,120 → 1, sigmoid) trained per layer in Experiment 2 to classify
partner identity (human vs AI) from model hidden states at conversation turn 5.</p>
<p><strong>Alignment</strong>: Cosine similarity between concept direction vector and probe weight vector,
computed per layer, then squared (R²) and averaged across all 41 layers.</p>
<p><strong>Bootstrap</strong>: 1,000 iterations, resampling human and AI concept prompts independently with replacement.</p>
<p><strong>Software</strong>: PyTorch, NumPy, SciPy. Analysis code: <code>exp_3/code/analysis/alignment/2a_alignment_analysis.py</code></p>
</div>

</body>
</html>"""
    return html


# ============================================================================
# MARKDOWN GENERATION
# ============================================================================

def generate_markdown(dim_data, all_data):
    """Generate a concise markdown summary."""
    available_versions = [v for v in VERSIONS if v in list(dim_data.values())[0]["versions"]]
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]

    lines = []
    lines.append("# Exp 3: Raw Alignment Comparison Across Data Versions")
    lines.append(f"\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Version | Reading R² (Mental, ×10⁻³) | Control R² (Mental, ×10⁻³) | Description |")
    lines.append("|---------|---------------------------|---------------------------|-------------|")

    for v in available_versions:
        r_vals = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_vals = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        r_mean = sum(r_vals) / len(r_vals) if r_vals else 0
        c_mean = sum(c_vals) / len(c_vals) if c_vals else 0
        lines.append(f"| {VERSION_LABELS[v]} | {r_mean*1000:.3f} | {c_mean*1000:.3f} | {VERSION_DESCRIPTIONS[v]} |")

    lines.append("")
    lines.append("## Per-Dimension Data (Reading Probe R² ×10⁻³)")
    lines.append("")

    header = "| Dimension | Category | " + " | ".join(VERSION_LABELS[v] for v in available_versions) + " |"
    sep = "|-----------|----------|" + "|".join("---" for _ in available_versions) + "|"
    lines.append(header)
    lines.append(sep)

    cat_order = ["Mental", "Physical", "Pragmatic", "Control", "SysPrompt"]
    sorted_dims = sorted(dim_data.keys(),
                         key=lambda d: (cat_order.index(dim_data[d]["category"])
                                        if dim_data[d]["category"] in cat_order else 99, d))

    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        vals = []
        for v in available_versions:
            if v in dd["versions"]:
                vals.append(f"{dd['versions'][v]['reading_r2']*1000:.3f}")
            else:
                vals.append("—")
        lines.append(f"| {dd['name']} | {dd['category']} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Per-Dimension Data (Control Probe R² ×10⁻³)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        vals = []
        for v in available_versions:
            if v in dd["versions"]:
                vals.append(f"{dd['versions'][v]['control_r2']*1000:.3f}")
            else:
                vals.append("—")
        lines.append(f"| {dd['name']} | {dd['category']} | " + " | ".join(vals) + " |")

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("1. **Alignment scales with identity cue specificity**: balanced_gpt > names > balanced_names >> labels ≈ nonsense_codeword ≈ nonsense_ignore")
    lines.append("2. **Control probes show higher alignment** than reading probes for name-based versions")
    lines.append("3. **Nonsense and label versions show near-floor alignment**, comparable to the shapes negative control")
    lines.append("4. **This is raw alignment** — residual analysis (projecting out entity baseline) needed to assess concept-specific contribution")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    lines.append("- **Metric**: Mean R² (cosine similarity squared) between concept direction vectors and probe weight vectors, averaged across 41 layers")
    lines.append("- **Bootstrap**: 1,000 iterations (prompt resampling)")
    lines.append("- **Model**: LLaMA-2-13B-Chat")
    lines.append("- **Concept vectors**: 19 dimensions, ~80 prompts each (contrasts mode: human vs AI)")
    lines.append("- **Probes**: From Exp 2, turn 5, reading and control probes (logistic, per-layer)")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading alignment summaries...")
    all_data = load_all_summaries()
    print(f"  Loaded {len(all_data)} versions: {list(all_data.keys())}")

    if not all_data:
        print("ERROR: No data loaded. Exiting.")
        sys.exit(1)

    print("Organizing data...")
    dim_data = organize_data(all_data)
    print(f"  {len(dim_data)} dimensions (excluding dim 16)")

    out_dir = Path(__file__).resolve().parent

    print("Generating HTML report...")
    html = generate_html(dim_data, all_data)
    html_path = out_dir / "raw_comparison.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  Saved: {html_path}")

    print("Generating Markdown summary...")
    md = generate_markdown(dim_data, all_data)
    md_path = out_dir / "raw_comparison.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  Saved: {md_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
