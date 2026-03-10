#!/usr/bin/env python3
"""
Generate pairwise dimension comparison HTML/MD reports.

Reads pairwise JSON from generate_pairwise_tests.py and produces:
  - Significance matrix heatmap (matplotlib PNG + SVG)
  - Summary table: per-dimension count of significant differences
  - Detailed pairwise table with diff, CI, p-value, FDR stars
  - Per-dimension detail sections

Usage:
    python generate_pairwise_report.py                        # turn 5, all types
    python generate_pairwise_report.py --turn 3 --type raw    # specific

Rachel C. Metzgar · Mar 2026
"""

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================

from config import config
COMPARISONS_ROOT = config.RESULTS.comparisons / "alignment"
VERSIONS = ["balanced_gpt", "nonsense_codeword"]

VERSION_LABELS = {
    "balanced_gpt": "Partner Identity",
    "nonsense_codeword": "Control",
}

PROBE_LABELS = {
    "metacognitive": "Metacognitive",
    "operational": "Operational",
}

PROBE_HEADER_COLORS = {
    "metacognitive": "#43A047",
    "operational": "#424242",
}

CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
    "Pragmatic": "#FF9800",
    "Control": "#9E9E9E",
    "Entity": "#7B1FA2",
    "SysPrompt": "#00BCD4",
    "Other": "#999",
}

CAT_ORDER = ["Mental", "Physical", "Pragmatic", "Control", "Entity", "SysPrompt", "Other"]

DIM_NAMES = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind (holistic)", 17: "Attention", 18: "SysPrompt (labeled)",
    20: "SysPrompt (talkto human)", 21: "SysPrompt (talkto AI)",
    22: "SysPrompt (bare human)", 23: "SysPrompt (bare AI)",
    25: "Beliefs", 26: "Desires", 27: "Goals",
    29: "Shapes (flip)", 30: "Granite/Sandstone", 31: "Squares/Triangles",
    32: "Horizontal/Vertical",
}

STANDALONE_DIM_NAMES = dict(DIM_NAMES)
STANDALONE_DIM_NAMES[16] = "Human (concept)"
STANDALONE_DIM_NAMES[17] = "AI (concept)"
STANDALONE_DIM_NAMES[18] = "Attention"

ANALYSIS_TITLES = {
    "raw": "Raw Contrast Alignment",
    "residual": "Residual Alignment",
    "standalone": "Standalone Concept Alignment",
}


# ============================================================================
# UTILITIES
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


def fig_to_svg_string(fig, dpi=150):
    """Convert a matplotlib Figure to SVG string."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pairwise_data(pairwise_json_path):
    """Load pairwise test results from JSON."""
    with open(pairwise_json_path) as f:
        return json.load(f)


def load_summary_data(summary_json_path):
    """Load per-dimension summary from JSON."""
    with open(summary_json_path) as f:
        return json.load(f)


# ============================================================================
# MATPLOTLIB FIGURES
# ============================================================================

def make_significance_matrix(pairwise_results, probe_type, dim_names_map, analysis_type):
    """
    Create a significance matrix heatmap.

    Lower triangle: mean difference (A - B).
    Color: green if A > B significantly, red if A < B significantly, white if n.s.
    Returns matplotlib Figure.
    """
    # Filter to this probe type
    results = [r for r in pairwise_results if r["probe_type"] == probe_type]
    if not results:
        return None

    # Get unique dim IDs in order
    dim_ids = sorted(set(r["dim_a_id"] for r in results) | set(r["dim_b_id"] for r in results))
    # Sort by category then id
    dim_ids.sort(key=lambda d: (
        CAT_ORDER.index(get_dim_category_static(d, analysis_type))
        if get_dim_category_static(d, analysis_type) in CAT_ORDER else 99, d))
    n = len(dim_ids)
    id_to_idx = {d: i for i, d in enumerate(dim_ids)}

    # Build matrix
    matrix = np.full((n, n), np.nan)
    sig_matrix = np.zeros((n, n), dtype=bool)
    for r in results:
        i = id_to_idx[r["dim_a_id"]]
        j = id_to_idx[r["dim_b_id"]]
        matrix[i, j] = r["mean_diff"]
        matrix[j, i] = -r["mean_diff"]
        sig_matrix[i, j] = r["significant_fdr"]
        sig_matrix[j, i] = r["significant_fdr"]

    # Create figure
    fig_size = max(8, n * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Color map: red (A < B) -> white (n.s.) -> green (A > B)
    vmax = np.nanmax(np.abs(matrix))
    if vmax == 0:
        vmax = 1e-6
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "sig", ["#E53935", "#FFFFFF", "#43A047"])
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Plot lower triangle only
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    masked_matrix = np.ma.array(matrix, mask=mask)

    im = ax.pcolormesh(
        np.arange(n + 1), np.arange(n + 1), masked_matrix,
        cmap=cmap, norm=norm, edgecolors="white", linewidth=0.5)

    # Add significance markers
    for i in range(n):
        for j in range(i):
            if sig_matrix[i, j]:
                val = matrix[i, j]
                ax.text(j + 0.5, i + 0.5, f"{val*1000:.2f}",
                        ha="center", va="center", fontsize=5,
                        fontweight="bold", color="black")
            else:
                ax.text(j + 0.5, i + 0.5, "n.s.",
                        ha="center", va="center", fontsize=4,
                        color="#999")

    # Labels
    labels = [dim_names_map.get(d, str(d)) for d in dim_ids]
    cats = [get_dim_category_static(d, analysis_type) for d in dim_ids]
    cat_colors = [CATEGORY_COLORS.get(c, "#333") for c in cats]

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    for tick, color in zip(ax.get_xticklabels(), cat_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), cat_colors):
        tick.set_color(color)

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)

    # Category separators
    prev_cat = None
    for idx, d in enumerate(dim_ids):
        cat = get_dim_category_static(d, analysis_type)
        if cat != prev_cat and prev_cat is not None:
            ax.axhline(y=idx, color="#999", linewidth=1, linestyle="--", alpha=0.5)
            ax.axvline(x=idx, color="#999", linewidth=1, linestyle="--", alpha=0.5)
        prev_cat = cat

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Mean Diff (×10⁻³)")
    cbar.ax.tick_params(labelsize=7)

    probe_label = PROBE_LABELS.get(probe_type, probe_type)
    ax.set_title(f"Pairwise Significance Matrix — {probe_label} Probes\n"
                 f"(Bold values = FDR-significant, q < 0.05)",
                 fontsize=10, pad=10)

    fig.tight_layout()
    return fig


def get_dim_category_static(dim_id, analysis_type):
    """Static version of category lookup."""
    if analysis_type == "standalone":
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 18, 25, 26, 27],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Control":   [14, 15, 30, 31, 32],
            "Entity":    [16, 17],
            "SysPrompt": [20, 21, 22, 23],
        }
    else:
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 17, 25, 26, 27],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Control":   [0, 14, 15, 29, 30, 31, 32],
            "SysPrompt": [18],
        }
    for cat, ids in categories.items():
        if dim_id in ids:
            return cat
    return "Other"


def make_summary_bar(summary_data, probe_type, analysis_type):
    """
    Bar chart showing n_sig_greater and n_sig_less for each dimension.
    Returns matplotlib Figure.
    """
    dim_entries = []
    for dim_id_str, info in summary_data.items():
        dim_id = int(dim_id_str)
        if probe_type not in info["probes"]:
            continue
        p = info["probes"][probe_type]
        dim_entries.append({
            "dim_id": dim_id,
            "name": info["name"],
            "category": info["category"],
            "n_greater": p["n_sig_greater"],
            "n_less": p["n_sig_less"],
        })

    if not dim_entries:
        return None

    # Sort by category then id
    dim_entries.sort(key=lambda e: (
        CAT_ORDER.index(e["category"]) if e["category"] in CAT_ORDER else 99,
        e["dim_id"]))

    n = len(dim_entries)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5))

    x = np.arange(n)
    names = [e["name"] for e in dim_entries]
    n_greater = [e["n_greater"] for e in dim_entries]
    n_less = [-e["n_less"] for e in dim_entries]  # negative for downward bars
    cats = [e["category"] for e in dim_entries]

    # Bar colors by category
    bar_colors = [CATEGORY_COLORS.get(c, "#999") for c in cats]

    ax.bar(x, n_greater, color=bar_colors, alpha=0.8, label="Sig. greater than")
    ax.bar(x, n_less, color=bar_colors, alpha=0.4, label="Sig. less than")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of dimensions", fontsize=9)

    # Color tick labels
    for tick, color in zip(ax.get_xticklabels(), bar_colors):
        tick.set_color(color)

    probe_label = PROBE_LABELS.get(probe_type, probe_type)
    ax.set_title(f"Significant Pairwise Differences — {probe_label} Probes\n"
                 f"(Upward = greater than N dims, downward = less than N dims; FDR q < 0.05)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right")

    # Category separators
    prev_cat = None
    for i, e in enumerate(dim_entries):
        if e["category"] != prev_cat and prev_cat is not None:
            ax.axvline(x=i - 0.5, color="#ccc", linewidth=1, linestyle="--")
        prev_cat = e["category"]

    fig.tight_layout()
    return fig


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html(version, pairwise_results, summary_data, analysis_type,
                  figures_b64, turn):
    """Generate HTML report for a single version."""
    title = ANALYSIS_TITLES.get(analysis_type, analysis_type)
    version_label = VERSION_LABELS.get(version, version)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    dim_names = STANDALONE_DIM_NAMES if analysis_type == "standalone" else DIM_NAMES

    # Summary table HTML
    summary_rows = ""
    dim_entries = []
    for dim_id_str, info in summary_data.items():
        dim_id = int(dim_id_str)
        dim_entries.append((dim_id, info))
    dim_entries.sort(key=lambda e: (
        CAT_ORDER.index(e[1]["category"]) if e[1]["category"] in CAT_ORDER else 99, e[0]))

    for dim_id, info in dim_entries:
        cat_color = CATEGORY_COLORS.get(info["category"], "#333")
        row = f'<tr><td style="color:{cat_color}; font-weight:600;">{info["name"]}</td>'
        row += f'<td>{info["category"]}</td>'
        for probe in ["metacognitive", "operational"]:
            if probe in info["probes"]:
                p = info["probes"][probe]
                row += f'<td style="color:#43A047;">{p["n_sig_greater"]}</td>'
                row += f'<td style="color:#E53935;">{p["n_sig_less"]}</td>'
            else:
                row += "<td>-</td><td>-</td>"
        row += "</tr>"
        summary_rows += row + "\n"

    # Pairwise detail table
    detail_rows = ""
    for probe in ["metacognitive", "operational"]:
        probe_results = [r for r in pairwise_results if r["probe_type"] == probe]
        # Sort by abs(mean_diff) descending
        probe_results.sort(key=lambda r: abs(r["mean_diff"]), reverse=True)
        for r in probe_results:
            cat_a_color = CATEGORY_COLORS.get(r["dim_a_category"], "#333")
            cat_b_color = CATEGORY_COLORS.get(r["dim_b_category"], "#333")
            sig_style = ' style="background:#E8F5E9;"' if r["significant_fdr"] else ""
            detail_rows += (
                f'<tr{sig_style}>'
                f'<td>{PROBE_LABELS[probe]}</td>'
                f'<td style="color:{cat_a_color};">{r["dim_a_name"]}</td>'
                f'<td style="color:{cat_b_color};">{r["dim_b_name"]}</td>'
                f'<td>{r["mean_diff"]*1000:.3f}</td>'
                f'<td>[{r["ci95_lo"]*1000:.3f}, {r["ci95_hi"]*1000:.3f}]</td>'
                f'<td>{r["p_two_sided"]:.4f}</td>'
                f'<td>{r["p_fdr"]:.4f}</td>'
                f'<td><strong>{r["stars"]}</strong></td>'
                f'</tr>\n'
            )

    # Per-dimension detail sections
    dim_detail_html = ""
    for dim_id, info in dim_entries:
        cat_color = CATEGORY_COLORS.get(info["category"], "#333")
        dim_detail_html += f'<h4 style="color:{cat_color};">{info["name"]} ({info["category"]})</h4>\n'
        for probe in ["metacognitive", "operational"]:
            if probe not in info["probes"]:
                continue
            p = info["probes"][probe]
            probe_label = PROBE_LABELS[probe]
            dim_detail_html += f'<p><strong>{probe_label}:</strong> '
            if p["n_sig_greater"] > 0:
                dim_detail_html += f'Significantly greater than {p["n_sig_greater"]} dims: '
                dim_detail_html += ", ".join(f"<em>{d}</em>" for d in p["sig_greater_than"])
                dim_detail_html += ". "
            if p["n_sig_less"] > 0:
                dim_detail_html += f'Significantly less than {p["n_sig_less"]} dims: '
                dim_detail_html += ", ".join(f"<em>{d}</em>" for d in p["sig_less_than"])
                dim_detail_html += ". "
            if p["n_sig_greater"] == 0 and p["n_sig_less"] == 0:
                dim_detail_html += "No significant pairwise differences."
            dim_detail_html += "</p>\n"

    # Embed figures
    sig_matrix_sections = ""
    for probe in ["metacognitive", "operational"]:
        key = f"sig_matrix_{probe}"
        if key in figures_b64:
            probe_label = PROBE_LABELS[probe]
            hdr_color = PROBE_HEADER_COLORS[probe]
            sig_matrix_sections += f"""
<div class="figure-container">
<p style="font-weight:600; color:{hdr_color}; margin-bottom:8px;">{probe_label} Probes</p>
<img src="data:image/png;base64,{figures_b64[key]}" style="width:100%; max-width:900px;">
<div class="caption"><strong>Figure.</strong> Pairwise significance matrix for {probe_label.lower()} probes.
Cells show mean difference in R² (×10⁻³). Bold values = FDR-significant (q &lt; 0.05).
Green = row dimension has higher alignment; red = lower alignment.</div>
</div>
"""

    summary_bar_sections = ""
    for probe in ["metacognitive", "operational"]:
        key = f"summary_bar_{probe}"
        if key in figures_b64:
            probe_label = PROBE_LABELS[probe]
            hdr_color = PROBE_HEADER_COLORS[probe]
            summary_bar_sections += f"""
<div class="figure-container">
<p style="font-weight:600; color:{hdr_color}; margin-bottom:8px;">{probe_label} Probes</p>
<img src="data:image/png;base64,{figures_b64[key]}" style="width:100%; max-width:1000px;">
<div class="caption"><strong>Figure.</strong> Number of dimensions each dimension is significantly
greater than (upward) or less than (downward), for {probe_label.lower()} probes. FDR q &lt; 0.05.</div>
</div>
"""

    # Counts
    n_pairs = len([r for r in pairwise_results if r["probe_type"] == "metacognitive"])
    n_sig_mc = sum(1 for r in pairwise_results if r["probe_type"] == "metacognitive" and r["significant_fdr"])
    n_sig_op = sum(1 for r in pairwise_results if r["probe_type"] == "operational" and r["significant_fdr"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 3: Pairwise Dimension Comparison — {title} ({version_label})</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ border-bottom: 3px solid #2E7D32; padding-bottom: 10px; }}
  h2 {{ color: #2E7D32; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
  h3 {{ color: #333; margin-top: 25px; }}
  h4 {{ margin-top: 15px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
  .section {{ margin: 30px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.85em; }}
  th, td {{ padding: 6px 8px; border: 1px solid #ddd; text-align: center; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .figure-container {{ margin: 20px 0; background: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; }}
  .caption {{ font-size: 0.85em; color: #555; margin-top: 10px; padding: 8px; background: #f5f5f5; border-left: 3px solid #2E7D32; }}
  .method-box {{ background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  .key-finding {{ background: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
  code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  .detail-table {{ max-height: 600px; overflow-y: auto; }}
</style>
</head>
<body>

<h1>Experiment 3: Pairwise Dimension Comparison — {title}</h1>
<div class="meta">
    <p>Version: <strong>{version_label}</strong> ({version}) | Turn: {turn} | Generated: {timestamp}</p>
    <p>Rachel C. Metzgar, Princeton University, Graziano Lab</p>
</div>

<h2>1. Overview</h2>
<div class="section">
<p>Pairwise bootstrap comparison of alignment (R²) between every pair of conceptual dimensions.
Tests whether dimension A's alignment is significantly different from dimension B's alignment,
using paired bootstrap distributions (1,000 iterations, same prompt resampling).</p>
<div class="method-box">
<p><strong>Test</strong>: Paired bootstrap difference. For each bootstrap iteration <em>i</em>,
compute <code>diff_i = R²_A(i) - R²_B(i)</code>. P-value (two-sided) = 2 × min(P(diff ≤ 0), P(diff ≥ 0)).</p>
<p><strong>Multiple comparisons</strong>: Benjamini-Hochberg FDR correction at q = 0.05 across all {n_pairs} pairs
(per probe type, separately).</p>
<p><strong>Results</strong>: Metacognitive probes: {n_sig_mc}/{n_pairs} pairs significant.
Operational probes: {n_sig_op}/{n_pairs} pairs significant.</p>
</div>
</div>

<h2>2. Significance Matrices</h2>
{sig_matrix_sections}

<h2>3. Summary: Significant Difference Counts</h2>
{summary_bar_sections}

<h3>Summary Table</h3>
<div class="section" style="overflow-x: auto;">
<table>
<tr><th rowspan="2">Dimension</th><th rowspan="2">Category</th>
<th colspan="2" style="color:#43A047;">Metacognitive</th>
<th colspan="2" style="color:#424242;">Operational</th></tr>
<tr><th style="color:#43A047;">N sig &gt;</th><th style="color:#43A047;">N sig &lt;</th>
<th style="color:#424242;">N sig &gt;</th><th style="color:#424242;">N sig &lt;</th></tr>
{summary_rows}
</table>
<div class="caption"><strong>Table 1.</strong> For each dimension, the number of other dimensions it is
significantly greater than or less than (FDR q &lt; 0.05), for each probe type.</div>
</div>

<h2>4. Per-Dimension Details</h2>
<div class="section">
{dim_detail_html}
</div>

<h2>5. Full Pairwise Table</h2>
<div class="section detail-table" style="overflow-x: auto;">
<table>
<tr><th>Probe</th><th>Dim A</th><th>Dim B</th>
<th>Mean Diff (×10⁻³)</th><th>95% CI (×10⁻³)</th>
<th>p (two-sided)</th><th>p (FDR)</th><th>Sig</th></tr>
{detail_rows}
</table>
<div class="caption"><strong>Table 2.</strong> Full pairwise comparison results. Positive mean diff = dim A &gt; dim B.
Rows highlighted green are FDR-significant (q &lt; 0.05). Sorted by |mean diff| descending.</div>
</div>

<h2>6. Methods</h2>
<div class="method-box">
<p><strong>Model</strong>: LLaMA-2-13B-Chat</p>
<p><strong>Probes</strong>: Logistic linear probes (5,120 → 1) from Exp 2, turn {turn}.</p>
<p><strong>Layer range</strong>: Layers 6–40 (35 of 41).</p>
<p><strong>Bootstrap</strong>: 1,000 iterations (prompt-level resampling), paired across dimensions.</p>
<p><strong>FDR correction</strong>: Benjamini-Hochberg, q = 0.05, applied per probe type.</p>
<p><strong>Script</strong>: <code>exp_3/code/2k_pairwise_tests.py</code></p>
</div>

</body>
</html>"""
    return html


def generate_markdown(version, pairwise_results, summary_data, analysis_type, turn):
    """Generate markdown summary."""
    title = ANALYSIS_TITLES.get(analysis_type, analysis_type)
    version_label = VERSION_LABELS.get(version, version)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Exp 3: Pairwise Dimension Comparison — {title} ({version_label})",
        f"\n*Generated: {timestamp} | Turn: {turn}*\n",
        "## Summary\n",
    ]

    # Summary table
    header = "| Dimension | Category | MC sig > | MC sig < | OP sig > | OP sig < |"
    sep = "|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)

    dim_entries = []
    for dim_id_str, info in summary_data.items():
        dim_entries.append((int(dim_id_str), info))
    dim_entries.sort(key=lambda e: (
        CAT_ORDER.index(e[1]["category"]) if e[1]["category"] in CAT_ORDER else 99, e[0]))

    for dim_id, info in dim_entries:
        mc = info["probes"].get("metacognitive", {})
        op = info["probes"].get("operational", {})
        lines.append(
            f"| {info['name']} | {info['category']} "
            f"| {mc.get('n_sig_greater', '-')} | {mc.get('n_sig_less', '-')} "
            f"| {op.get('n_sig_greater', '-')} | {op.get('n_sig_less', '-')} |"
        )

    lines.append("")
    lines.append("## Top Significant Pairs\n")
    lines.append("| Probe | Dim A | Dim B | Diff (×10⁻³) | p (FDR) | Sig |")
    lines.append("|---|---|---|---|---|---|")

    # Top 20 most significant
    sig_results = [r for r in pairwise_results if r["significant_fdr"]]
    sig_results.sort(key=lambda r: abs(r["mean_diff"]), reverse=True)
    for r in sig_results[:20]:
        lines.append(
            f"| {PROBE_LABELS[r['probe_type']]} | {r['dim_a_name']} | {r['dim_b_name']} "
            f"| {r['mean_diff']*1000:.3f} | {r['p_fdr']:.4f} | {r['stars']} |"
        )

    lines.append("")
    lines.append("## Methods\n")
    lines.append(f"- **Analysis**: {title}, pairwise bootstrap comparison")
    lines.append("- **FDR**: Benjamini-Hochberg, q = 0.05")
    lines.append("- **Bootstrap**: 1,000 paired iterations")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def process_version(version, analysis_type, turn, out_base):
    """Generate report for one version."""
    in_dir = out_base / version
    pairwise_json = in_dir / "pairwise_dimensions.json"
    summary_json = in_dir / "pairwise_summary.json"

    if not pairwise_json.exists():
        print(f"  WARNING: {pairwise_json} not found, skipping")
        return
    if not summary_json.exists():
        print(f"  WARNING: {summary_json} not found, skipping")
        return

    pairwise_results = load_pairwise_data(pairwise_json)
    summary_data = load_summary_data(summary_json)
    dim_names = STANDALONE_DIM_NAMES if analysis_type == "standalone" else DIM_NAMES

    print(f"  Loaded {len(pairwise_results)} pairwise results, {len(summary_data)} dimensions")

    # Generate figures
    figures_b64 = {}
    fig_dir = in_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for probe in ["metacognitive", "operational"]:
        # Significance matrix
        fig = make_significance_matrix(pairwise_results, probe, dim_names, analysis_type)
        if fig is not None:
            key = f"sig_matrix_{probe}"
            figures_b64[key] = fig_to_base64_png(fig)
            # Also save as file
            fig2 = make_significance_matrix(pairwise_results, probe, dim_names, analysis_type)
            png_data = fig_to_png_bytes(fig2)
            with open(fig_dir / f"{key}.png", "wb") as f:
                f.write(png_data)
            fig3 = make_significance_matrix(pairwise_results, probe, dim_names, analysis_type)
            svg_str = fig_to_svg_string(fig3)
            with open(fig_dir / f"{key}.svg", "w") as f:
                f.write(svg_str)

        # Summary bar
        fig = make_summary_bar(summary_data, probe, analysis_type)
        if fig is not None:
            key = f"summary_bar_{probe}"
            figures_b64[key] = fig_to_base64_png(fig)
            fig2 = make_summary_bar(summary_data, probe, analysis_type)
            png_data = fig_to_png_bytes(fig2)
            with open(fig_dir / f"{key}.png", "wb") as f:
                f.write(png_data)
            fig3 = make_summary_bar(summary_data, probe, analysis_type)
            svg_str = fig_to_svg_string(fig3)
            with open(fig_dir / f"{key}.svg", "w") as f:
                f.write(svg_str)

    print(f"  Figures: {fig_dir}/ ({len(figures_b64)} figures)")

    # HTML
    html = generate_html(version, pairwise_results, summary_data, analysis_type,
                         figures_b64, turn)
    html_path = in_dir / "pairwise_comparison.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"  HTML: {html_path}")

    # Markdown
    md = generate_markdown(version, pairwise_results, summary_data, analysis_type, turn)
    md_path = in_dir / "pairwise_comparison.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"  MD:   {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pairwise dimension comparison HTML/MD reports")
    parser.add_argument("--turn", type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help="Conversation turn (default: 5)")
    parser.add_argument("--type", choices=["raw", "residual", "standalone", "all"],
                        default="all", help="Analysis type (default: all)")
    args = parser.parse_args()

    types = ["raw", "residual", "standalone"] if args.type == "all" else [args.type]

    for atype in types:
        print(f"\n{'='*60}")
        print(f"  Pairwise Report: {atype} (turn {args.turn})")
        print(f"{'='*60}")

        out_base = COMPARISONS_ROOT / f"turn_{args.turn}" / atype

        for version in VERSIONS:
            print(f"\n  Version: {version}")
            process_version(version, atype, args.turn, out_base)

    print("\nDone!")


if __name__ == "__main__":
    main()
