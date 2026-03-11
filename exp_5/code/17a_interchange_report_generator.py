#!/usr/bin/env python3
"""
Experiment 5, Phase 17a: Interchange Intervention Report Generator

Generates a comprehensive HTML report from saved interchange intervention results.
Standalone — regenerates from saved data without recomputing.

Data files (from data_dir("interchange")):
    - transfer_matrix.npz: transfer_matrix (6, 6, n_layers_tested), layers array
    - block_analysis.json: contrast results, regression results, subject swap results
    - verb_swap_results.csv: per-swap outcomes
    - subject_swap_results.csv: per-item subject swap outcomes

Output:
    results/{model}/interchange/interchange_report.html

Usage:
    python code/17a_interchange_report_generator.py --model llama2_13b_chat

Env: llama2_env (no GPU needed)
Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import argparse
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    set_model, add_model_argument, results_dir, data_dir,
    ensure_dir, figures_dir, CONDITION_LABELS,
)


# ── Constants ────────────────────────────────────────────────────────────────

COND_SHORT = {
    "mental_state": "C1",
    "dis_mental": "C2",
    "scr_mental": "C3",
    "action": "C4",
    "dis_action": "C5",
    "scr_action": "C6",
}

COND_COLORS = {
    "mental_state": "#d62728",
    "dis_mental": "#ff7f0e",
    "scr_mental": "#ffbb78",
    "action": "#1f77b4",
    "dis_action": "#2ca02c",
    "scr_action": "#98df8a",
}

COND_NAMES = {
    "mental_state": "C1: mental_state",
    "dis_mental": "C2: dis_mental",
    "scr_mental": "C3: scr_mental",
    "action": "C4: action",
    "dis_action": "C5: dis_action",
    "scr_action": "C6: scr_action",
}

PLACEHOLDER_IMG = (
    '<div style="background:#f8f8f8; border:2px dashed #ccc; padding:40px; '
    'text-align:center; color:#999; margin:15px 0; border-radius:4px;">'
    'Data not yet available — run the interchange intervention pipeline first.</div>'
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def sig_stars(p):
    """Return significance stars for a p-value."""
    try:
        if p is None or np.isnan(p):
            return ""
    except (TypeError, ValueError):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ── Data loading ─────────────────────────────────────────────────────────────

def load_transfer_matrix(d_dir):
    """Load transfer_matrix.npz → (matrix, layers) or (None, None)."""
    path = d_dir / "transfer_matrix.npz"
    if not path.exists():
        return None, None
    data = np.load(path)
    return data["transfer_matrix"], data["layers"]


def load_block_analysis(d_dir):
    """Load block_analysis.json → dict or None."""
    path = d_dir / "block_analysis.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_verb_swap_results(d_dir):
    """Load verb_swap_results.csv → DataFrame or None."""
    path = d_dir / "verb_swap_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_subject_swap_results(d_dir):
    """Load subject_swap_results.csv → DataFrame or None."""
    path = d_dir / "subject_swap_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Plot functions ───────────────────────────────────────────────────────────

def find_peak_layer(transfer_matrix, layers):
    """Find layer with highest mean within-mental (C1→C1) swap success."""
    c1_idx = CONDITION_LABELS.index("mental_state")
    within_mental = transfer_matrix[c1_idx, c1_idx, :]
    peak_idx = int(np.argmax(within_mental))
    return peak_idx, int(layers[peak_idx])


def plot_transfer_heatmap(transfer_matrix, layers, fig_dir):
    """6x6 transfer heatmap at peak layer."""
    peak_idx, peak_layer = find_peak_layer(transfer_matrix, layers)
    mat = transfer_matrix[:, :, peak_idx]

    short_labels = [COND_SHORT[c] for c in CONDITION_LABELS]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(6))
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.set_yticks(range(6))
    ax.set_yticklabels(short_labels, fontsize=10)
    ax.set_xlabel("Target condition", fontsize=11)
    ax.set_ylabel("Source condition", fontsize=11)
    ax.set_title(f"Interchange Transfer Matrix — Layer {peak_layer}", fontsize=13)

    # Annotate cells
    for i in range(6):
        for j in range(6):
            val = mat[i, j]
            color = "white" if val > 0.6 or val < 0.2 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean swap success", fontsize=10)

    fig.tight_layout()
    save_fig(fig, fig_dir / "transfer_heatmap.png")
    return fig_to_base64(fig)


def plot_layer_profiles(transfer_matrix, layers, fig_dir):
    """Line plot of key swap metrics across layers."""
    c1 = CONDITION_LABELS.index("mental_state")
    c4 = CONDITION_LABELS.index("action")

    profiles = {
        "C1→C1 (within mental)": transfer_matrix[c1, c1, :],
        "C4→C4 (within action)": transfer_matrix[c4, c4, :],
        "C1→C4 (mental→action)": transfer_matrix[c1, c4, :],
        "C4→C1 (action→mental)": transfer_matrix[c4, c1, :],
    }
    colors = ["#d62728", "#1f77b4", "#9467bd", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, vals), color in zip(profiles.items(), colors):
        ax.plot(layers, vals, label=label, color=color, linewidth=2, marker="o",
                markersize=3)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean swap success", fontsize=11)
    ax.set_title("Interchange Transfer — Layer Profiles", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, fig_dir / "layer_profiles.png")
    return fig_to_base64(fig)


def plot_block_structure(transfer_matrix, layers, fig_dir):
    """Within-type vs cross-type mean swap success across layers."""
    n_layers_tested = transfer_matrix.shape[2]

    # Mental conditions: C1, C2, C3 (indices 0, 1, 2)
    # Action conditions: C4, C5, C6 (indices 3, 4, 5)
    mental_idxs = [0, 1, 2]
    action_idxs = [3, 4, 5]

    within_vals = np.zeros(n_layers_tested)
    cross_vals = np.zeros(n_layers_tested)

    for li in range(n_layers_tested):
        mat = transfer_matrix[:, :, li]
        # Within-type: mental×mental + action×action (excluding diagonal)
        within_pairs = []
        for block_idxs in [mental_idxs, action_idxs]:
            for i in block_idxs:
                for j in block_idxs:
                    if i != j:
                        within_pairs.append(mat[i, j])
        # Cross-type: mental→action + action→mental
        cross_pairs = []
        for i in mental_idxs:
            for j in action_idxs:
                cross_pairs.append(mat[i, j])
                cross_pairs.append(mat[j, i])

        within_vals[li] = np.mean(within_pairs) if within_pairs else 0
        cross_vals[li] = np.mean(cross_pairs) if cross_pairs else 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, within_vals, label="Within-type mean", color="#2ca02c",
            linewidth=2, marker="o", markersize=3)
    ax.plot(layers, cross_vals, label="Cross-type mean", color="#d62728",
            linewidth=2, marker="s", markersize=3)
    ax.fill_between(layers, within_vals, cross_vals, alpha=0.15, color="#888888")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean swap success", fontsize=11)
    ax.set_title("Block Structure: Within-Type vs Cross-Type Transfer", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, fig_dir / "block_structure.png")
    return fig_to_base64(fig)


def plot_subject_swap_profile(subject_df, fig_dir):
    """Cross-type vs within-type subject swap effect across layers."""
    if "swap_type" not in subject_df.columns or "layer_idx" not in subject_df.columns:
        return None

    cross_df = subject_df[subject_df["swap_type"] == "cross_type"]
    within_df = subject_df[subject_df["swap_type"] == "within_type"]

    cross_by_layer = cross_df.groupby("layer_idx")["effect"].mean()
    within_by_layer = within_df.groupby("layer_idx")["effect"].mean()

    all_layers = sorted(subject_df["layer_idx"].unique())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_layers,
            [cross_by_layer.get(l, np.nan) for l in all_layers],
            label="Cross-type effect", color="#d62728", linewidth=2,
            marker="o", markersize=3)
    ax.plot(all_layers,
            [within_by_layer.get(l, np.nan) for l in all_layers],
            label="Within-type effect", color="#1f77b4", linewidth=2,
            marker="s", markersize=3)

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean effect", fontsize=11)
    ax.set_title("Subject Swap Analysis: Cross-Type vs Within-Type", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, fig_dir / "subject_swap_profile.png")
    return fig_to_base64(fig)


def plot_verb_similarity_regression(block_analysis, fig_dir):
    """Plot verb similarity regression if data available."""
    if block_analysis is None:
        return None
    reg = block_analysis.get("regression_results")
    if reg is None:
        return None

    # If raw data points are available, scatter; otherwise just table
    if "points" in reg:
        points = reg["points"]
        x = [p["verb_similarity"] for p in points]
        y = [p["swap_success"] for p in points]
        slope = reg.get("slope", 0)
        intercept = reg.get("intercept", 0)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(x, y, alpha=0.4, s=20, color="#1f77b4")
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#d62728",
                linewidth=2, label=f"slope={slope:.3f}")
        ax.set_xlabel("Verb similarity", fontsize=11)
        ax.set_ylabel("Swap success", fontsize=11)
        ax.set_title("Swap Success vs Verb Similarity", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        save_fig(fig, fig_dir / "verb_similarity_regression.png")
        return fig_to_base64(fig)

    return None


# ── HTML building ────────────────────────────────────────────────────────────

def build_stats_table(block_analysis):
    """Build HTML table from contrast_results in block_analysis."""
    if block_analysis is None:
        return PLACEHOLDER_IMG

    contrasts = block_analysis.get("contrast_results")
    if not contrasts:
        return "<p>No contrast results found in block_analysis.json.</p>"

    rows = []
    if isinstance(contrasts, list):
        for item in contrasts:
            name = item.get("contrast", item.get("name", ""))
            obs = item.get("observed_difference", item.get("observed", np.nan))
            p = item.get("p_value", item.get("p", np.nan))
            stars = sig_stars(p)
            sig_class = ' class="sig"' if stars else ""
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td>{obs:.4f}</td>"
                f"<td{sig_class}>{p:.4f} {stars}</td></tr>"
            )
    elif isinstance(contrasts, dict):
        for name, vals in contrasts.items():
            if isinstance(vals, dict):
                obs = vals.get("observed_difference", vals.get("observed", np.nan))
                p = vals.get("p_value", vals.get("p", np.nan))
            else:
                obs = vals
                p = np.nan
            stars = sig_stars(p)
            sig_class = ' class="sig"' if stars else ""
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td>{obs:.4f}</td>"
                f"<td{sig_class}>{p:.4f} {stars}</td></tr>"
            )

    return (
        '<table class="results-table">'
        "<tr><th>Contrast</th><th>Observed Difference</th><th>p-value</th></tr>"
        + "\n".join(rows)
        + "</table>"
    )


def build_regression_table(block_analysis):
    """Build HTML table of regression coefficients."""
    if block_analysis is None:
        return ""
    reg = block_analysis.get("regression_results")
    if reg is None:
        return ""

    rows = []
    for key in ["slope", "intercept", "r_squared", "r", "p_value", "n"]:
        val = reg.get(key)
        if val is not None:
            if isinstance(val, float):
                rows.append(f"<tr><td>{key}</td><td>{val:.4f}</td></tr>")
            else:
                rows.append(f"<tr><td>{key}</td><td>{val}</td></tr>")

    if not rows:
        return ""

    return (
        '<table class="results-table">'
        "<tr><th>Parameter</th><th>Value</th></tr>"
        + "\n".join(rows)
        + "</table>"
    )


def build_subject_swap_table(block_analysis):
    """Build HTML table of subject swap summary from block_analysis."""
    if block_analysis is None:
        return ""
    ss = block_analysis.get("subject_swap_results")
    if ss is None:
        return ""

    if isinstance(ss, dict):
        rows = []
        for name, vals in ss.items():
            if isinstance(vals, dict):
                obs = vals.get("observed_difference", vals.get("observed", np.nan))
                p = vals.get("p_value", vals.get("p", np.nan))
                stars = sig_stars(p)
                sig_class = ' class="sig"' if stars else ""
                rows.append(
                    f"<tr><td>{name}</td>"
                    f"<td>{obs:.4f}</td>"
                    f"<td{sig_class}>{p:.4f} {stars}</td></tr>"
                )
        if rows:
            return (
                '<table class="results-table">'
                "<tr><th>Comparison</th><th>Observed Difference</th><th>p-value</th></tr>"
                + "\n".join(rows)
                + "</table>"
            )
    return ""


# ── Main report builder ─────────────────────────────────────────────────────

def generate_report(model):
    set_model(model)

    d_dir = data_dir("interchange")
    fig_dir = ensure_dir(figures_dir("interchange"))
    out_dir = ensure_dir(results_dir("interchange"))
    out_path = out_dir / "interchange_report.html"

    # Load data
    transfer_matrix, layers = load_transfer_matrix(d_dir)
    block_analysis = load_block_analysis(d_dir)
    verb_swap_df = load_verb_swap_results(d_dir)
    subject_swap_df = load_subject_swap_results(d_dir)

    # ── Section 1: Transfer Matrix ───────────────────────────────────────────
    if transfer_matrix is not None and layers is not None:
        peak_idx, peak_layer = find_peak_layer(transfer_matrix, layers)
        img_heatmap = plot_transfer_heatmap(transfer_matrix, layers, fig_dir)
        heatmap_html = (
            f'<img src="data:image/png;base64,{img_heatmap}" alt="Transfer heatmap">'
            f'<p class="caption">6x6 transfer matrix at peak layer {peak_layer} '
            f'(highest C1→C1 swap success). Cell values show mean swap success '
            f'rate when source activations from row condition are patched into '
            f'target condition (column).</p>'
        )
    else:
        peak_layer = "N/A"
        heatmap_html = PLACEHOLDER_IMG

    # ── Section 2: Layer Profiles ────────────────────────────────────────────
    if transfer_matrix is not None and layers is not None:
        img_profiles = plot_layer_profiles(transfer_matrix, layers, fig_dir)
        profiles_html = (
            f'<img src="data:image/png;base64,{img_profiles}" alt="Layer profiles">'
            '<p class="caption">Swap success for key condition pairs across layers. '
            'Within-type swaps (C1→C1, C4→C4) test whether same-type activations '
            'are interchangeable. Cross-type swaps (C1→C4, C4→C1) test whether '
            'mental and action representations occupy distinct subspaces.</p>'
        )
    else:
        profiles_html = PLACEHOLDER_IMG

    # ── Section 3: Block Structure ───────────────────────────────────────────
    if transfer_matrix is not None and layers is not None:
        img_block = plot_block_structure(transfer_matrix, layers, fig_dir)
        block_html = (
            f'<img src="data:image/png;base64,{img_block}" alt="Block structure">'
            '<p class="caption">Mean swap success averaged over all within-type '
            'pairs (mental-mental + action-action, excluding self) vs all cross-type '
            'pairs (mental-action + action-mental). Shaded area shows the gap — '
            'a consistent positive gap indicates block-diagonal structure in the '
            'transfer matrix.</p>'
        )
    else:
        block_html = PLACEHOLDER_IMG

    # ── Section 4: Subject Swap Analysis ─────────────────────────────────────
    if subject_swap_df is not None and len(subject_swap_df) > 0:
        img_subject = plot_subject_swap_profile(subject_swap_df, fig_dir)
        if img_subject is not None:
            subject_html = (
                f'<img src="data:image/png;base64,{img_subject}" alt="Subject swap">'
                '<p class="caption">Subject swap analysis: effect of swapping the '
                'subject identity between sentence pairs. Cross-type swaps change '
                'the subject across mental/action boundaries; within-type swaps keep '
                'the same verb type.</p>'
            )
        else:
            subject_html = ('<p>Subject swap data loaded but required columns '
                            '(swap_type, layer_idx, effect) not found.</p>')
        subject_table = build_subject_swap_table(block_analysis)
        if subject_table:
            subject_html += (
                '<h3>Subject Swap Statistical Summary</h3>'
                + subject_table
            )
    else:
        subject_html = PLACEHOLDER_IMG

    # ── Section 5: Statistical Summary ───────────────────────────────────────
    stats_table_html = build_stats_table(block_analysis)

    # ── Section 6: Verb Similarity Control ───────────────────────────────────
    img_regression = None
    if block_analysis is not None:
        img_regression = plot_verb_similarity_regression(block_analysis, fig_dir)

    if img_regression is not None:
        regression_html = (
            f'<img src="data:image/png;base64,{img_regression}" '
            f'alt="Verb similarity regression">'
            '<p class="caption">Relationship between verb semantic similarity '
            'and swap success. If interchange effects are driven by verb meaning '
            'overlap rather than representational structure, we expect a strong '
            'positive correlation.</p>'
        )
    else:
        regression_html = ""

    regression_table = build_regression_table(block_analysis)
    if regression_table:
        regression_html += "<h3>Regression Coefficients</h3>" + regression_table

    if not regression_html:
        regression_html = PLACEHOLDER_IMG

    # ── Status banner ────────────────────────────────────────────────────────
    missing = []
    if transfer_matrix is None:
        missing.append("transfer_matrix.npz")
    if block_analysis is None:
        missing.append("block_analysis.json")
    if verb_swap_df is None:
        missing.append("verb_swap_results.csv")
    if subject_swap_df is None:
        missing.append("subject_swap_results.csv")

    status_banner = ""
    if missing:
        status_banner = (
            '<div style="background:#fff3cd; border:2px solid #ffc107; '
            'padding:15px; margin:15px 0; border-radius:4px;">'
            '<strong>Pending data:</strong> The following files are not yet '
            'available: ' + ", ".join(f"<code>{m}</code>" for m in missing)
            + '. Run the interchange intervention pipeline first, then '
            'regenerate this report.</div>'
        )

    # ── Assemble HTML ────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 5: Interchange Intervention Results</title>
<style>
    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; max-width: 1100px;
           margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #d62728; padding-bottom: 10px; }}
    h2 {{ color: #1a1a2e; margin-top: 40px; border-bottom: 1px solid #ccc;
          padding-bottom: 5px; }}
    h3 {{ color: #444; margin-top: 25px; }}
    .summary-box {{ background: #f0f4f8; border-left: 4px solid #1f77b4;
                    padding: 15px; margin: 15px 0; border-radius: 4px; }}
    .result-box {{ background: #fff8f0; border-left: 4px solid #ff7f0e;
                   padding: 15px; margin: 15px 0; border-radius: 4px; }}
    .method-box {{ background: #f0f8f0; border-left: 4px solid #2ca02c;
                   padding: 15px; margin: 15px 0; border-radius: 4px; }}
    img {{ max-width: 100%; margin: 15px 0; border: 1px solid #ddd;
           border-radius: 4px; }}
    .caption {{ font-size: 0.9em; color: #555; margin-top: -10px; margin-bottom: 20px;
                font-style: italic; }}
    table.results-table {{ border-collapse: collapse; width: 100%; font-size: 0.85em;
                          margin: 15px 0; }}
    table.results-table th {{ background: #1a1a2e; color: white; padding: 8px 10px;
                             text-align: left; }}
    table.results-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
    table.results-table tr:hover {{ background: #f5f5f5; }}
    .sig {{ color: #d62728; font-weight: bold; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px;
            font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Experiment 5: Interchange Intervention Results</h1>

{status_banner}

<div class="summary-box">
<strong>Question:</strong> Do mental state attribution representations (C1) occupy a
distinct computational role from action attributions (C4), such that swapping internal
activations between conditions transfers the representational identity?<br><br>
<strong>Method:</strong> Interchange intervention — for each pair of sentences, patch
the internal activations from one sentence into the forward pass of another and measure
whether the patched representation shifts to match the source condition.<br><br>
<strong>Conditions:</strong>
C1 (mental_state), C2 (dis_mental), C3 (scr_mental),
C4 (action), C5 (dis_action), C6 (scr_action).<br><br>
<strong>Peak layer:</strong> {peak_layer}
</div>

<h2>1. Transfer Matrix</h2>

<div class="method-box">
The 6&times;6 transfer matrix shows how well activations from each source condition
(row) can be patched into each target condition (column) to reproduce the source's
representational signature. High within-block values and low cross-block values indicate
that mental and action representations are functionally distinct.
</div>

{heatmap_html}

<h2>2. Layer Profiles</h2>

<div class="method-box">
Layer-by-layer swap success for key condition pairs. Comparing within-type
(C1&rarr;C1, C4&rarr;C4) vs cross-type (C1&rarr;C4, C4&rarr;C1) reveals at which
layers the mental/action distinction is most strongly encoded.
</div>

{profiles_html}

<h2>3. Block Structure</h2>

<div class="method-box">
Aggregated within-type vs cross-type swap success across layers. A persistent gap
between within-type and cross-type means the transfer matrix has block-diagonal
structure — mental conditions share activations with each other more than with
action conditions, and vice versa.
</div>

{block_html}

<h2>4. Subject Swap Analysis</h2>

<div class="method-box">
Tests whether the <em>subject identity</em> (He) drives distinct representations
by swapping subject activations between sentence pairs. Cross-type swaps (mental
subject into action frame) vs within-type swaps (mental subject into mental frame)
reveal whether subject representations are modulated by verb type.
</div>

{subject_html}

<h2>5. Statistical Summary</h2>

<div class="method-box">
Contrast tests from block_analysis.json. Each contrast compares the mean swap
success between two sets of condition pairs (e.g., within-mental vs cross-type).
P-values from permutation tests; significance: * p&lt;.05, ** p&lt;.01, *** p&lt;.001.
</div>

{stats_table_html}

<h2>6. Verb Similarity Control</h2>

<div class="method-box">
Tests whether interchange success is confounded by verb semantic similarity.
If verbs with more similar meanings produce higher swap success, the interchange
effect may reflect lexical overlap rather than representational structure.
</div>

{regression_html}

<hr>
<p style="color:#999; font-size:0.8em;">
Generated by <code>17a_interchange_report_generator.py</code> | Model: {model}
</p>

</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)

    print(f"Report written to {out_path}")
    if missing:
        print(f"  WARNING: {len(missing)} data file(s) missing — sections show placeholders")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate interchange intervention HTML report"
    )
    add_model_argument(parser)
    args = parser.parse_args()

    generate_report(args.model)


if __name__ == "__main__":
    main()
