#!/usr/bin/env python3
"""
Experiment 5, Step 8a: Residual Probe Report Generator

Generates an HTML report from saved residual probe results.
Standalone — regenerates from CSVs without recomputing.

Output:
    results/{model}/probe_training/residual_probe_report.html
    results/{model}/probe_training/figures/residual_probes/*.png

Usage:
    python code/probes/8a_residual_report_generator.py --model llama2_13b_chat

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    set_model, add_model_argument, results_dir, data_dir,
    ensure_dir, figures_dir, POSITION_LABELS,
)


# ── Colors ──────────────────────────────────────────────────────────────────

POSITION_COLORS = {
    "verb": "#d62728",
    "object": "#1f77b4",
    "period": "#2ca02c",
}

POSITION_NAMES = {
    "verb": "Verb",
    "object": "Object",
    "period": "Period",
}

PROBE_TYPE_STYLES = {
    "residual": {"linestyle": "-", "linewidth": 2.5, "label_suffix": " (residual)"},
    "raw":      {"linestyle": "--", "linewidth": 1.5, "label_suffix": " (raw)"},
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def sig_marker(p_fdr):
    try:
        if np.isnan(p_fdr):
            return ""
    except (TypeError, ValueError):
        return ""
    if p_fdr < 0.001:
        return "***"
    elif p_fdr < 0.01:
        return "**"
    elif p_fdr < 0.05:
        return "*"
    return ""


PENDING_HTML = (
    '<div class="result-box pending">'
    '<strong>Data pending</strong> — analysis not yet complete. '
    'Rerun report generator after data files are produced.'
    '</div>'
)


# ── Plot functions ──────────────────────────────────────────────────────────

def plot_confound_r2_heatmap(r2_df, fig_dir):
    """Heatmap of mean R² (confound variance explained) by position × layer."""
    positions = POSITION_LABELS
    layers = sorted(r2_df["layer"].unique())

    matrix = np.full((len(positions), len(layers)), np.nan)
    for pi, pos in enumerate(positions):
        pdf = r2_df[r2_df["position"] == pos].sort_values("layer")
        for _, row in pdf.iterrows():
            li = layers.index(int(row["layer"]))
            matrix[pi, li] = row["mean_r2"]

    fig, ax = plt.subplots(figsize=(14, 2.5))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.15,
                   interpolation="nearest")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=7)
    ax.set_yticks(range(len(positions)))
    ax.set_yticklabels([POSITION_NAMES.get(p, p) for p in positions], fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title("Confound Variance Explained (Mean R² per Dimension)", fontsize=12,
                 fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Mean R²", fontsize=9)

    fig.tight_layout()
    save_fig(fig, fig_dir / "confound_r2_heatmap.png")
    return fig_to_base64(fig)


def plot_residual_vs_raw_profiles(df, fig_dir):
    """AUC vs layer: two lines per position panel (raw=dashed, residual=solid)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for pi, pos in enumerate(POSITION_LABELS):
        ax = axes[pi]

        for probe_type in ["raw", "residual"]:
            pdf = df[(df["position"] == pos) & (df["probe_type"] == probe_type)]
            pdf = pdf.sort_values("layer")
            if pdf.empty:
                continue

            layers = pdf["layer"].values
            aucs = pdf["auc"].values
            p_fdr = pdf["p_fdr"].values

            style = PROBE_TYPE_STYLES[probe_type]
            color = POSITION_COLORS[pos]
            alpha = 1.0 if probe_type == "residual" else 0.6

            ax.plot(layers, aucs, color=color, linewidth=style["linewidth"],
                    linestyle=style["linestyle"], alpha=alpha,
                    label=f"{POSITION_NAMES[pos]}{style['label_suffix']}")

            # Significance dots
            sig = p_fdr < 0.05
            if sig.any():
                marker_size = 40 if probe_type == "residual" else 25
                ax.scatter(layers[sig], aucs[sig], color=color, s=marker_size,
                           zorder=5, edgecolors="black", linewidth=0.5, alpha=alpha)

        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Layer", fontsize=11)
        if pi == 0:
            ax.set_ylabel("AUC", fontsize=11)
        ax.set_title(f"{POSITION_NAMES[pos]} Position", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_ylim(0.35, 1.05)

    fig.suptitle("C1 vs C2+C3+C4 Probe — Residual vs Raw Activations",
                 fontsize=14, fontweight="bold")
    fig.subplots_adjust(bottom=0.1, top=0.88, wspace=0.15)
    save_fig(fig, fig_dir / "residual_vs_raw_profiles.png")
    return fig_to_base64(fig)


def plot_residual_acc_auc_layerwise(df, fig_dir):
    """Layerwise accuracy AND AUC for residual probes, one line per position."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for pos in POSITION_LABELS:
        pdf = df[(df["position"] == pos) & (df["probe_type"] == "residual")].sort_values("layer")
        if pdf.empty:
            continue
        layers = pdf["layer"].values
        color = POSITION_COLORS[pos]

        # Accuracy
        ax1.plot(layers, pdf["accuracy"].values, color=color, linewidth=2,
                 label=POSITION_NAMES[pos])
        sig_mask = pdf["p_fdr"].values < 0.05
        if sig_mask.any():
            ax1.scatter(layers[sig_mask], pdf["accuracy"].values[sig_mask],
                       color=color, s=40, zorder=5, edgecolors="black", linewidth=0.5)

        # AUC
        ax2.plot(layers, pdf["auc"].values, color=color, linewidth=2,
                 label=POSITION_NAMES[pos])
        if sig_mask.any():
            ax2.scatter(layers[sig_mask], pdf["auc"].values[sig_mask],
                       color=color, s=40, zorder=5, edgecolors="black", linewidth=0.5)

    ax1.axhline(0.25, color="gray", linewidth=0.8, linestyle="--", label="Chance (0.25)")
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Residual Probe — Accuracy by Layer", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_ylim(0.0, 1.05)

    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Chance (0.5)")
    ax2.set_ylabel("AUC", fontsize=12)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_title("Residual Probe — AUC by Layer", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(0.35, 1.05)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))

    fig.subplots_adjust(bottom=0.08, top=0.95, hspace=0.25)
    save_fig(fig, fig_dir / "residual_probe_acc_auc_layerwise.png")
    return fig_to_base64(fig)


def plot_signal_retention(df, fig_dir):
    """Bar chart: above-chance signal for raw vs residual at each position's peak layer."""
    positions = POSITION_LABELS
    bar_data = []

    for pos in positions:
        for pt in ["raw", "residual"]:
            pdf = df[(df["position"] == pos) & (df["probe_type"] == pt)]
            if pdf.empty:
                continue
            peak = pdf.loc[pdf["auc"].idxmax()]
            bar_data.append({
                "position": pos,
                "probe_type": pt,
                "peak_auc": peak["auc"],
                "above_chance": peak["auc"] - 0.5,
                "peak_layer": int(peak["layer"]),
                "p_fdr": peak["p_fdr"],
            })

    if not bar_data:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw vs residual AUC at peak
    ax = axes[0]
    x = np.arange(len(positions))
    width = 0.35

    raw_vals = []
    resid_vals = []
    for pos in positions:
        raw_row = [d for d in bar_data if d["position"] == pos and d["probe_type"] == "raw"]
        resid_row = [d for d in bar_data if d["position"] == pos and d["probe_type"] == "residual"]
        raw_vals.append(raw_row[0]["above_chance"] if raw_row else 0)
        resid_vals.append(resid_row[0]["above_chance"] if resid_row else 0)

    bars1 = ax.bar(x - width/2, raw_vals, width, label="Raw", color="#888888", alpha=0.7)
    bars2 = ax.bar(x + width/2, resid_vals, width, label="Residual",
                   color="#d62728", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([POSITION_NAMES[p] for p in positions])
    ax.set_ylabel("AUC - 0.5 (above chance)", fontsize=11)
    ax.set_title("Signal Strength at Peak Layer", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Right: retention fraction
    ax = axes[1]
    retention = []
    for i, pos in enumerate(positions):
        if raw_vals[i] > 0.01:
            retention.append(resid_vals[i] / raw_vals[i])
        else:
            retention.append(0)

    colors = [POSITION_COLORS[p] for p in positions]
    ax.bar(x, retention, color=colors, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([POSITION_NAMES[p] for p in positions])
    ax.set_ylabel("Fraction of Signal Retained", fontsize=11)
    ax.set_title("Signal Retention After Deconfounding", fontsize=12, fontweight="bold")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Full retention")
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=9)

    # Add percentage labels
    for i, ret in enumerate(retention):
        ax.text(i, ret + 0.03, f"{ret*100:.0f}%", ha="center", fontsize=11,
                fontweight="bold")

    fig.tight_layout()
    save_fig(fig, fig_dir / "signal_retention.png")
    return fig_to_base64(fig)


# ── Summary helpers ─────────────────────────────────────────────────────────

def peak_summary(df, probe_type, pos=None):
    """Find peak layer for a probe type and summarize."""
    sub = df[df["probe_type"] == probe_type].copy()
    if pos is not None:
        sub = sub[sub["position"] == pos]
    if sub.empty:
        return "No data."

    peak_idx = sub["auc"].idxmax()
    peak = sub.loc[peak_idx]
    sig_layers = sub[sub["p_fdr"] < 0.05]["layer"].tolist()
    sig_str = ", ".join(str(int(l)) for l in sig_layers) if sig_layers else "none"

    return (f"Peak layer: <strong>{int(peak['layer'])}</strong> "
            f"(AUC={peak['auc']:.4f}, p_perm={peak['p_perm']:.4f}, "
            f"p_fdr={peak['p_fdr']:.4f}). "
            f"Significant layers (FDR &lt; .05): <strong>{sig_str}</strong>.")


# ── Main report ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate residual probe report")
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    probe_data = data_dir("probe_training")
    fig_dir = ensure_dir(figures_dir("probe_training") / "residual_probes")
    out_html = results_dir("probe_training") / "residual_probe_report.html"
    ensure_dir(out_html.parent)

    print(f"Loading data from: {probe_data}")

    # ── Load data ───────────────────────────────────────────────────────
    def load_csv(name):
        path = probe_data / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {name}: {len(df)} rows")
            return df
        print(f"  Missing {name} — skipping")
        return None

    def load_json(name):
        path = probe_data / name
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"  Loaded {name}")
            return data
        print(f"  Missing {name} — skipping")
        return None

    probe_df = load_csv("residual_probe_results.csv")
    r2_df = load_csv("residual_confound_r2.csv")
    critical_tests = load_json("critical_tests.json")

    # ── Generate figures ────────────────────────────────────────────────
    img_r2 = ""
    if r2_df is not None:
        img_r2 = plot_confound_r2_heatmap(r2_df, fig_dir)

    img_profiles = ""
    if probe_df is not None:
        img_profiles = plot_residual_vs_raw_profiles(probe_df, fig_dir)

    img_resid_layerwise = ""
    if probe_df is not None:
        img_resid_layerwise = plot_residual_acc_auc_layerwise(probe_df, fig_dir)

    img_retention = ""
    if probe_df is not None:
        img_retention = plot_signal_retention(probe_df, fig_dir)

    # ── Summaries ───────────────────────────────────────────────────────
    residual_summary = ""
    raw_summary = ""
    if probe_df is not None:
        parts = []
        for pos in POSITION_LABELS:
            parts.append(
                f"<strong>{POSITION_NAMES[pos]}:</strong> "
                + peak_summary(probe_df, "residual", pos)
            )
        residual_summary = "<br>".join(parts)

        parts = []
        for pos in POSITION_LABELS:
            parts.append(
                f"<strong>{POSITION_NAMES[pos]}:</strong> "
                + peak_summary(probe_df, "raw", pos)
            )
        raw_summary = "<br>".join(parts)

    # ── Peak detail table ───────────────────────────────────────────────
    peak_table_html = ""
    if probe_df is not None:
        rows = []
        for pt in ["residual", "raw"]:
            for pos in POSITION_LABELS:
                sub = probe_df[(probe_df["probe_type"] == pt) &
                               (probe_df["position"] == pos)]
                if sub.empty:
                    continue
                peak = sub.loc[sub["auc"].idxmax()]
                sig = sig_marker(peak["p_fdr"])
                rows.append(
                    f"<tr><td>{pt}</td><td>{POSITION_NAMES[pos]}</td>"
                    f"<td>{int(peak['layer'])}</td>"
                    f"<td>{peak['accuracy']:.4f}</td>"
                    f"<td>{peak['auc']:.4f}</td>"
                    f"<td>{peak['p_perm']:.4f}</td>"
                    f"<td>{peak['p_fdr']:.4f} <span class='sig'>{sig}</span></td></tr>"
                )
        peak_table_html = "\n".join(rows)

    # ── Gram-Schmidt comparison ─────────────────────────────────────────
    gs_comparison_html = ""
    if critical_tests is not None and probe_df is not None:
        ct = critical_tests
        gs_auc = ct.get("residual_auc")
        if gs_auc is not None:
            # Find residual probe peak at same position (period = last token)
            resid_period = probe_df[
                (probe_df["probe_type"] == "residual") &
                (probe_df["position"] == "period")
            ]
            if not resid_period.empty:
                peak = resid_period.loc[resid_period["auc"].idxmax()]
                gs_comparison_html = f"""
<div class="result-box">
<strong>Gram-Schmidt (critical_tests.json) residual AUC:</strong> {gs_auc:.4f}<br>
<strong>OLS residual probe peak AUC (period):</strong> {peak['auc']:.4f} at layer {int(peak['layer'])}<br>
<em>Both methods test whether C1 is distinguishable after removing confound components.
Gram-Schmidt operates on probe weight vectors; OLS residualization operates directly on
activation data. Convergent results strengthen the conclusion.</em>
</div>"""

    # ── Build HTML ──────────────────────────────────────────────────────
    fig_num = 1

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 5: Residual Probe Results</title>
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
    .result-box.pending {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
    .method-box {{ background: #f0f8f0; border-left: 4px solid #2ca02c;
                   padding: 15px; margin: 15px 0; border-radius: 4px; }}
    .interpretation-box {{ background: #f8f0f8; border-left: 4px solid #9467bd;
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
    nav.toc {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px;
               padding: 15px 25px; margin: 20px 0; }}
    nav.toc h2 {{ margin-top: 5px; border: none; font-size: 1.1em; }}
    nav.toc ul {{ padding-left: 20px; margin: 5px 0; }}
    nav.toc li {{ margin: 3px 0; }}
    nav.toc a {{ text-decoration: none; color: #1f77b4; }}
    nav.toc a:hover {{ text-decoration: underline; }}
    table.feature-matrix {{ border-collapse: collapse; margin: 15px auto; font-size: 0.9em; }}
    table.feature-matrix th, table.feature-matrix td {{
        border: 1px solid #ccc; padding: 6px 14px; text-align: center; }}
    table.feature-matrix th {{ background: #f5f5f5; font-weight: bold; }}
    table.feature-matrix td.left {{ text-align: left; }}
</style>
</head>
<body>

<h1>Experiment 5: Residual Activation Probes</h1>

<div class="summary-box">
<strong>Purpose:</strong> Test whether C1 (full mental state attribution) is distinguishable
from C2+C3+C4 in activation space <em>after removing confound features</em> via OLS
residualization. This complements the 5-predictor RSA (which operates in distance space)
and the Gram-Schmidt critical tests (which operate on probe weight vectors).<br><br>
<strong>Method:</strong> Regress out subject_presence, mental_verb, and grammaticality from
raw activations using OLS, then train logistic probes on the residuals. A raw baseline probe
(same C1-C4 subset, no deconfounding) provides a within-analysis comparison.<br><br>
<strong>Key question:</strong> If the residual probe classifies C1 above chance, the model
maintains an attribution-specific representation that is irreducible to these three binary
features.
</div>

<nav class="toc">
<h2>Contents</h2>
<ul>
<li><a href="#method">1. Method Overview</a></li>
<li><a href="#feature-matrix">2. Confound Feature Matrix</a></li>
<li><a href="#confound-r2">3. Confound Variance Explained</a></li>
<li><a href="#profiles">4. Layer Profiles: Residual vs Raw</a></li>
<li><a href="#peak-table">5. Peak Layer Detail</a></li>
<li><a href="#retention">6. Signal Retention</a></li>
<li><a href="#gram-schmidt">7. Comparison with Gram-Schmidt</a></li>
<li><a href="#interpretation">8. Interpretation Guide</a></li>
<li><a href="#future">9. Future Directions: Causal Intervention</a></li>
</ul>
</nav>

<hr>

<h2 id="method">1. Method Overview</h2>

<div class="method-box">
<strong>Residual activation probing</strong> tests whether the model's representation of C1
(mental state attribution) contains information beyond what is captured by three confound
features: subject presence, mental verb presence, and grammaticality.<br><br>
<strong>Procedure:</strong>
<ol>
<li>Filter to C1-C4 (224 sentences) — matching the 5-predictor RSA design</li>
<li>Build a (224 &times; 4) confound feature matrix F with intercept + 3 binary features</li>
<li>Compute the residual-maker matrix: M = I &minus; F(F'F)<sup>&minus;1</sup>F'</li>
<li>For each (position, layer): X<sub>resid</sub> = M &middot; X<sub>raw</sub></li>
<li>Train logistic probe (C1 vs C2+C3+C4) on residual activations</li>
<li>Compare to raw baseline probe on unresidualized activations</li>
</ol>
<strong>Null hypothesis:</strong> After removing confound features, C1 is not distinguishable
(residual probe AUC = 0.5). Rejection implies an irreducible attribution representation.
</div>

<div class="method-box">
<strong>Metric definitions:</strong>
<ul>
<li><strong>Accuracy (Acc):</strong> Proportion of sentences correctly classified as C1 vs
    not-C1 by the logistic probe. Chance = 0.25 (since 56 of 224 sentences are C1).
    A probe that always predicts &ldquo;not C1&rdquo; achieves 0.75 accuracy, so accuracy alone
    is a weak metric for imbalanced classes.</li>
<li><strong>AUC (Area Under the ROC Curve):</strong> Measures how well the probe&rsquo;s
    continuous confidence scores discriminate C1 from non-C1 across all possible thresholds.
    Chance = 0.5. Unlike accuracy, AUC is insensitive to class imbalance &mdash; it asks
    &ldquo;given a random C1 and a random non-C1 sentence, how often does the probe assign
    higher confidence to C1?&rdquo; This is the primary metric for probe evaluation.</li>
<li><strong>p (perm):</strong> Permutation p-value from shuffling C1/non-C1 labels 200 times
    (baseline probes) or 10,000 times (critical tests). Tests whether observed AUC exceeds
    the null distribution.</li>
<li><strong>p (FDR):</strong> Benjamini-Hochberg FDR-corrected p-value across layers within
    each (position, probe_type) group. Controls the expected false discovery rate at q = 0.05.</li>
</ul>
</div>

<hr>

<h2 id="feature-matrix">2. Confound Feature Matrix</h2>

<table class="feature-matrix">
<tr>
    <th style="text-align:left">Feature</th>
    <th>C1 (mental_state)</th>
    <th>C2 (dis_mental)</th>
    <th>C3 (scr_mental)</th>
    <th>C4 (action)</th>
</tr>
<tr>
    <td class="left">Subject presence ("He")</td>
    <td>1</td><td>0</td><td>0</td><td>1</td>
</tr>
<tr>
    <td class="left">Mental verb</td>
    <td>1</td><td>1</td><td>1</td><td>0</td>
</tr>
<tr>
    <td class="left">Grammaticality</td>
    <td>1</td><td>1</td><td>0</td><td>1</td>
</tr>
</table>

<div class="method-box">
<strong>Design matrix:</strong> 224 rows (56 items &times; 4 conditions), 4 columns
(intercept + 3 features). Rank = 4 (full rank). <code>action_verb</code> excluded because
it is perfectly anti-correlated with <code>mental_verb</code> in C1&ndash;C4.<br><br>
C1 is the <em>only</em> condition with subject=1, mental_verb=1, grammaticality=1. No single
feature or pair of features uniquely identifies C1. The residual probe must therefore rely on
activation structure beyond these features.
</div>

<hr>

<h2 id="confound-r2">3. Confound Variance Explained</h2>

<div class="method-box">
<strong>Diagnostic:</strong> For each (position, layer), compute R&sup2; per hidden dimension
(fraction of variance explained by the 3 confound features + intercept). Summary statistics
(mean, median, max) indicate how much of the activation variance the confounds account for.
</div>
"""

    if r2_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_r2}" alt="Confound R-squared heatmap">
<p class="caption"><strong>Figure {fig_num}.</strong> Mean R&sup2; across hidden dimensions.
Higher values indicate the confound features explain more activation variance at that
(position, layer). This is what gets removed before probing.</p>
"""
        fig_num += 1

        # Summary stats
        overall_mean = r2_df["mean_r2"].mean()
        overall_max = r2_df["mean_r2"].max()
        peak_r2 = r2_df.loc[r2_df["mean_r2"].idxmax()]
        html += f"""
<div class="result-box">
<strong>Overall mean R&sup2;:</strong> {overall_mean:.4f}<br>
<strong>Peak mean R&sup2;:</strong> {peak_r2['mean_r2']:.4f} at ({peak_r2['position']},
layer {int(peak_r2['layer'])})<br>
<em>Low R&sup2; means the confounds explain little activation variance (residuals are close
to raw). High R&sup2; means substantial variance is removed.</em>
</div>
"""
    else:
        html += PENDING_HTML

    html += """
<hr>

<h2 id="profiles">4. Layer Profiles: Residual vs Raw</h2>

<div class="method-box">
<strong>Key figure:</strong> AUC vs layer for C1 vs C2+C3+C4 probe. Solid lines = residual
(deconfounded) activations; dashed lines = raw activations. Filled circles mark layers
significant at FDR &lt; .05. Each panel shows one token position.
</div>
"""

    if probe_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_profiles}" alt="Residual vs raw layer profiles">
<p class="caption"><strong>Figure {fig_num}.</strong> C1 vs C2+C3+C4 probe AUC across layers.
Solid = residual probe (confounds removed); dashed = raw probe (no deconfounding). Filled
circles = significant at FDR &lt; .05. Gray dashed line = chance (0.5).</p>

<h3>Residual Probe</h3>
<div class="result-box">
{residual_summary}
</div>

<h3>Raw Baseline Probe</h3>
<div class="result-box">
{raw_summary}
</div>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    # Section 4b: Residual probe layerwise detail
    if probe_df is not None:
        html += f"""
<h3>Residual Probe Layerwise Detail</h3>
<p>Accuracy and AUC for the <strong>residual</strong> (deconfounded) probe only, at each layer and position.
Filled circles = FDR-significant.</p>
<img src="data:image/png;base64,{img_resid_layerwise}" alt="Residual probe accuracy and AUC by layer">
<p class="caption"><strong>Figure {fig_num}.</strong> Residual probe accuracy (top) and AUC (bottom) across layers.
Chance accuracy = 0.25 (1 of 4 conditions). Chance AUC = 0.5.</p>
"""
        fig_num += 1

        # Full layerwise table for residual probes
        resid_df = probe_df[probe_df["probe_type"] == "residual"].copy()
        if not resid_df.empty:
            html += """
<h3>Full Layerwise Table (Residual Probes)</h3>
<table class="results-table">
<tr><th>Layer</th>"""
            for pos in POSITION_LABELS:
                html += f'<th colspan="3">{POSITION_NAMES[pos]}</th>'
            html += '</tr><tr><th></th>'
            for _ in POSITION_LABELS:
                html += '<th>Acc</th><th>AUC</th><th>p<sub>FDR</sub></th>'
            html += '</tr>'

            all_layers = sorted(resid_df["layer"].unique())
            for layer in all_layers:
                html += f'<tr><td>{int(layer)}</td>'
                for pos in POSITION_LABELS:
                    row = resid_df[(resid_df["layer"] == layer) & (resid_df["position"] == pos)]
                    if row.empty:
                        html += '<td>—</td><td>—</td><td>—</td>'
                    else:
                        r = row.iloc[0]
                        p_fdr = r["p_fdr"]
                        marker = sig_marker(p_fdr)
                        bold = ' style="font-weight:bold;color:#d62728"' if marker else ""
                        html += f'<td>{r["accuracy"]:.4f}</td>'
                        html += f'<td{bold}>{r["auc"]:.4f}</td>'
                        html += f'<td{bold}>{p_fdr:.4f} {marker}</td>'
                html += '</tr>'
            html += '</table>'

    html += """
<hr>

<h2 id="peak-table">5. Peak Layer Detail</h2>
"""

    if peak_table_html:
        html += f"""
<table class="results-table">
<tr>
    <th>Probe Type</th><th>Position</th><th>Peak Layer</th>
    <th>Accuracy</th><th>AUC</th><th>p (perm)</th><th>p (FDR)</th>
</tr>
{peak_table_html}
</table>
"""
    else:
        html += PENDING_HTML

    html += """
<hr>

<h2 id="retention">6. Signal Retention</h2>

<div class="method-box">
<strong>Signal retention</strong> = (residual_AUC &minus; 0.5) / (raw_AUC &minus; 0.5)
at each position's peak layer. A value of 1.0 means all above-chance signal survives
deconfounding; 0 means all signal is explained by the three confound features.
</div>
"""

    if probe_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_retention}" alt="Signal retention chart">
<p class="caption"><strong>Figure {fig_num}.</strong> Left: Above-chance AUC for raw (gray)
and residual (red) probes at each position's peak layer. Right: Fraction of signal retained
after deconfounding.</p>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    html += f"""
<hr>

<h2 id="gram-schmidt">7. Comparison with Gram-Schmidt</h2>

<div class="method-box">
<strong>Two complementary approaches:</strong>
<ul>
<li><strong>Gram-Schmidt (critical tests):</strong> Projects feature probe directions out of
    the attribution probe weight vector, then tests whether the residual direction still
    classifies C1. Operates on <em>probe weight vectors</em>.</li>
<li><strong>OLS residualization (this analysis):</strong> Projects confound features out of
    the <em>activation data</em>, then trains a new probe from scratch on residuals.</li>
</ul>
Both test the same hypothesis (C1 has irreducible representational structure), but from
different angles. Convergent results provide stronger evidence.
</div>
"""

    if gs_comparison_html:
        html += gs_comparison_html
    elif critical_tests is None:
        html += PENDING_HTML
    else:
        html += '<div class="result-box">Gram-Schmidt residual AUC not available in critical_tests.json.</div>'

    html += """
<hr>

<h2 id="interpretation">8. Interpretation Guide</h2>

<div class="interpretation-box">
<h3>What would each pattern of results mean?</h3>
<ul>
<li><strong>Residual probe significant, high retention:</strong> The model encodes C1
    (mental state attribution) in a way that cannot be reduced to subject presence, mental
    verb presence, and grammaticality. Most of the C1 signal is attribution-specific. This
    is the strongest evidence for a dedicated bound attribution representation.</li>

<li><strong>Residual probe significant, low retention:</strong> C1 is still distinguishable
    after deconfounding, but much of the signal comes from the confound features. The
    attribution-specific signal exists but is smaller than the feature-driven signal.</li>

<li><strong>Residual probe not significant:</strong> After removing the three confound
    features, C1 is no longer distinguishable. The raw probe's success was entirely driven
    by the additive combination of subject presence + mental verb + grammaticality. No
    irreducible attribution representation exists.</li>

<li><strong>Position effects:</strong> If the residual probe is significant at the period
    (last token) but not the verb, this suggests attribution binding emerges through
    sequential composition across the sentence, rather than being present at the verb
    alone.</li>

<li><strong>Layer effects:</strong> Significant residual probe results at middle layers
    (10-25) suggest semantic composition; results at early layers (&lt;5) may reflect
    lexical confounds not captured by the three features.</li>
</ul>
</div>

<hr>

<h2 id="future">9. Future Directions: Causal Intervention</h2>

<div class="method-box">
<strong>If the residual probe classifies above chance</strong>, the probe weight vector
defines a direction in residual-activation space for causal intervention:
<ul>
<li><strong>Ablation:</strong> Project out the residual probe direction during forward pass:
    h<sub>ablated</sub> = h &minus; (h &middot; &wcirc;)&wcirc;. If downstream behavior
    changes for C1 but not C4, this is causal evidence.</li>
<li><strong>Amplification:</strong> Scale up the component:
    h<sub>amp</sub> = h + &alpha;(h &middot; &wcirc;)&wcirc;. Test if the model becomes
    more &ldquo;mentalistic&rdquo; in continuations.</li>
<li><strong>Injection:</strong> Add direction to C4 (action) sentences:
    h<sub>steered</sub> = h + &alpha;&wcirc;. Test if action sentences start being processed
    like mental state attributions.</li>
<li><strong>Double dissociation:</strong> Ablating the residual direction disrupts C1 but
    not C4; ablating a confound direction disrupts something else but not
    attribution-specific processing.</li>
</ul>
</div>

<hr>
"""

    html += f"""
<p style="font-size:0.85em; color:#888;">
Generated by <code>probes/8a_residual_report_generator.py</code>.
Regenerate: <code>python code/probes/8a_residual_report_generator.py --model {args.model}</code>
</p>

</body>
</html>"""

    with open(out_html, "w") as f:
        f.write(html)
    print(f"\nReport saved: {out_html}")


if __name__ == "__main__":
    main()
