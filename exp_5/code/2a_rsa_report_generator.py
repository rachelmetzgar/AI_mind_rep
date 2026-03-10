#!/usr/bin/env python3
"""
Experiment 5, Phase 2a: RSA Report Generator

Generates a comprehensive HTML report from saved RSA results.
Standalone — regenerates from CSVs without recomputing.

Output:
    results/{model}/rsa/rsa_report.html

Usage:
    python code/2a_rsa_report_generator.py --model llama2_13b_chat

Env: llama2_env (no GPU needed)
Rachel C. Metzgar · Mar 2026
"""

import sys
import argparse
import base64
import io
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    set_model, add_model_argument, results_dir, data_dir,
    ensure_dir, figures_dir, CONDITION_LABELS, CATEGORY_LABELS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

COLORS = {
    "A": "#d62728",  # red
    "B": "#1f77b4",  # blue
    "C": "#ff7f0e",  # orange
    "D": "#2ca02c",  # green
    "E": "#9467bd",  # purple
    "F": "#8c564b",  # brown
    "G": "#e377c2",  # pink
    "H": "#7f7f7f",  # gray
}

MODEL_NAMES = {
    "A": "Full Attribution (C1 only)",
    "B": "Mental Verb Presence (C1,C2,C3)",
    "C": "Subject Presence (C1,C4)",
    "D": "Item Identity (same item)",
    "E": "Mental Verb + Object (C1,C2)",
    "F": "Grammatical Order (C1,C2,C4,C5)",
    "G": "Scrambled Form (C3,C6)",
    "H": "Action Verb Presence (C4,C5,C6)",
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
    "mental_state": "C1: He [mental] the X",
    "dis_mental": "C2: [Mental] the X",
    "scr_mental": "C3: The X to [mental]",
    "action": "C4: He [action] the X",
    "dis_action": "C5: [Action] the X",
    "scr_action": "C6: The X to [action]",
}


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


def safe_p_fdr(df):
    """Return p_fdr column if present, else p column, else ones."""
    if "p_fdr" in df.columns:
        return df["p_fdr"].values
    if "p" in df.columns:
        return df["p"].values
    return np.ones(len(df))


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_simple_rsa(df, fig_dir):
    """Layer profile for simple RSA (Model A)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    layers = df["layer"].values
    rhos = df["rho"].values
    p_fdr = safe_p_fdr(df)

    ax.plot(layers, rhos, color=COLORS["A"], linewidth=2, label="Model A (Full Attribution)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # Mark significant layers
    sig = p_fdr < 0.05
    if sig.any():
        ax.scatter(layers[sig], rhos[sig], color=COLORS["A"], s=40, zorder=5,
                   edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Spearman rho", fontsize=12)
    ax.set_title("Analysis 1: Simple RSA — Full Attribution (Model A) vs. Neural RDM", fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlim(-0.5, layers.max() + 0.5)

    save_fig(fig, fig_dir / "simple_rsa_layer_profile.png")
    return fig_to_base64(fig)


def plot_partial_rsa(df, hypothesis_key, fig_dir, suffix="primary"):
    """Layer profile for partial RSA betas — all models."""
    all_keys = [hypothesis_key] + [k for k in df["model"].unique() if k != hypothesis_key]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    # Top: hypothesis model
    ax = axes[0]
    hyp_df = df[df["model"] == hypothesis_key]
    layers = hyp_df["layer"].values
    betas = hyp_df["beta"].values
    srs = hyp_df["semi_partial_r"].values
    p_fdr = safe_p_fdr(hyp_df)

    ax.plot(layers, betas, color=COLORS[hypothesis_key], linewidth=2.5,
            label=f"Model {hypothesis_key}: {MODEL_NAMES[hypothesis_key]}")
    ax.fill_between(layers, 0, betas, alpha=0.15, color=COLORS[hypothesis_key])

    sig = p_fdr < 0.05
    if sig.any():
        ax.scatter(layers[sig], betas[sig], color=COLORS[hypothesis_key],
                   s=50, zorder=5, edgecolors="black", linewidth=0.5)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Standardized beta", fontsize=12)
    ax.set_title(f"Analysis 2: Partial RSA — Model {hypothesis_key} (hypothesis) "
                 f"with confounds partialed out", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

    # Bottom: confound models
    ax2 = axes[1]
    confound_keys = [k for k in all_keys if k != hypothesis_key]
    for k in confound_keys:
        k_df = df[df["model"] == k]
        ax2.plot(k_df["layer"].values, k_df["beta"].values,
                 color=COLORS.get(k, "gray"), linewidth=1.2, alpha=0.8,
                 label=f"{k}: {MODEL_NAMES[k]}")

    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Standardized beta", fontsize=12)
    ax2.set_title("Confound model betas (same regression)", fontsize=11)
    ax2.legend(fontsize=7, ncol=2, loc="upper left")
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_xlim(-0.5, layers.max() + 0.5)

    fig.tight_layout()
    save_fig(fig, fig_dir / f"partial_rsa_{suffix}_layer_profile.png")
    return fig_to_base64(fig)


def plot_partial_semi_partial(df, hypothesis_key, fig_dir, suffix="primary"):
    """Semi-partial r for hypothesis model across layers."""
    fig, ax = plt.subplots(figsize=(10, 4))
    hyp_df = df[df["model"] == hypothesis_key]
    layers = hyp_df["layer"].values
    srs = hyp_df["semi_partial_r"].values
    p_fdr = safe_p_fdr(hyp_df)

    ax.bar(layers, srs, color=COLORS[hypothesis_key], alpha=0.7, width=0.8)
    sig = p_fdr < 0.05
    if sig.any():
        ax.scatter(layers[sig], srs[sig], color="black", s=20, zorder=5, marker="v")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Semi-partial r", fontsize=12)
    ax.set_title(f"Model {hypothesis_key}: Unique variance (semi-partial r) across layers",
                 fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlim(-0.5, layers.max() + 0.5)

    save_fig(fig, fig_dir / f"partial_rsa_{suffix}_semi_partial.png")
    return fig_to_base64(fig)


def plot_category_rsa(df, fig_dir):
    """Category RSA across conditions and layers."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for cond in CONDITION_LABELS:
        cdf = df[df["condition"] == cond]
        if cdf.empty:
            continue
        layers = cdf["layer"].values
        rhos = cdf["rho"].values
        p_fdr = safe_p_fdr(cdf)

        lw = 2.5 if cond == "mental_state" else 1.2
        alpha = 1.0 if cond == "mental_state" else 0.7
        ax.plot(layers, rhos, color=COND_COLORS[cond], linewidth=lw, alpha=alpha,
                label=COND_NAMES[cond])

        sig = p_fdr < 0.05
        if sig.any() and cond == "mental_state":
            ax.scatter(layers[sig], rhos[sig], color=COND_COLORS[cond],
                       s=40, zorder=5, edgecolors="black", linewidth=0.5)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Spearman rho", fontsize=12)
    ax.set_title("Analysis 3: Category Structure RSA (7 verb categories) by Condition",
                 fontsize=13)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlim(-0.5, layers.max() + 0.5)

    save_fig(fig, fig_dir / "category_rsa_by_condition.png")
    return fig_to_base64(fig)


def plot_a_vs_e(df_primary, df_secondary, fig_dir):
    """Compare Model A and Model E betas across layers."""
    fig, ax = plt.subplots(figsize=(10, 4))

    a_df = df_primary[df_primary["model"] == "A"]
    e_df = df_secondary[df_secondary["model"] == "E"]

    ax.plot(a_df["layer"].values, a_df["beta"].values,
            color=COLORS["A"], linewidth=2, label=f"A: {MODEL_NAMES['A']}")
    ax.plot(e_df["layer"].values, e_df["beta"].values,
            color=COLORS["E"], linewidth=2, linestyle="--",
            label=f"E: {MODEL_NAMES['E']}")

    sig_a = safe_p_fdr(a_df) < 0.05
    sig_e = safe_p_fdr(e_df) < 0.05
    if sig_a.any():
        ax.scatter(a_df["layer"].values[sig_a], a_df["beta"].values[sig_a],
                   color=COLORS["A"], s=40, zorder=5, edgecolors="black", linewidth=0.5)
    if sig_e.any():
        ax.scatter(e_df["layer"].values[sig_e], e_df["beta"].values[sig_e],
                   color=COLORS["E"], s=40, zorder=5, edgecolors="black", linewidth=0.5,
                   marker="D")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Standardized beta", fontsize=12)
    ax.set_title("Model A vs. Model E: Is the subject required for attribution structure?",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    save_fig(fig, fig_dir / "model_a_vs_e_comparison.png")
    return fig_to_base64(fig)


# ── Summary tables ───────────────────────────────────────────────────────────

def make_layerwise_table(df, value_col, p_col="p_fdr", label=""):
    """Generate an HTML table with one row per layer, coloring significant rows."""
    if p_col not in df.columns:
        p_col = "p" if "p" in df.columns else None
    rows_html = []
    for _, row in df.iterrows():
        sig = sig_marker(row[p_col]) if p_col else ""
        style = ' style="background:#ffe0e0; font-weight:bold"' if sig else ""
        cols = "".join(f"<td>{row[c]:.4f}</td>" if isinstance(row[c], float) else f"<td>{row[c]}</td>"
                       for c in df.columns)
        rows_html.append(f"<tr{style}>{cols}<td>{sig}</td></tr>")

    header = "".join(f"<th>{c}</th>" for c in df.columns) + "<th>Sig</th>"
    return f"""
    <table class="results-table">
    <thead><tr>{header}</tr></thead>
    <tbody>{"".join(rows_html)}</tbody>
    </table>
    """


def peak_summary(df, value_col, p_col="p_fdr", model_filter=None):
    """Find peak layer and summarize."""
    if model_filter:
        df = df[df["model"] == model_filter]
    if df.empty:
        return "No data."
    peak_idx = df[value_col].abs().idxmax()
    peak = df.loc[peak_idx]
    if p_col not in df.columns:
        p_col = "p" if "p" in df.columns else None
    sig_layers = df[df[p_col] < 0.05]["layer"].tolist() if p_col else []
    sig_str = ", ".join(str(l) for l in sig_layers) if sig_layers else "none"
    p_val = peak[p_col] if p_col and p_col in peak.index else float('nan')
    p_label = p_col if p_col else "p"
    return (f"Peak layer: <strong>{int(peak['layer'])}</strong> "
            f"({value_col}={peak[value_col]:.4f}, {p_label}={p_val:.4f}). "
            f"Significant layers (FDR < .05): <strong>{sig_str}</strong>.")


# ── Main report ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RSA report")
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    rsa_data = data_dir("rsa")
    fig_dir = ensure_dir(figures_dir("rsa"))
    out_html = results_dir("rsa") / "rsa_report.html"

    # Load data — handle missing files gracefully for partial results
    def load_csv(name):
        path = rsa_data / name
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {name}: {len(df)} rows")
            return df
        print(f"  Missing {name} — skipping")
        return None

    simple_df = load_csv("simple_rsa_results.csv")
    partial_primary_df = load_csv("partial_rsa_primary_results.csv")
    partial_secondary_df = load_csv("partial_rsa_secondary_results.csv")
    category_df = load_csv("category_rsa_results.csv")

    PLACEHOLDER_IMG = ""
    PENDING_HTML = '<div class="result-box"><strong>Analysis still running — rerun report generator later.</strong></div>'

    # Generate figures — only for available data
    img_simple = plot_simple_rsa(simple_df, fig_dir) if simple_df is not None else PLACEHOLDER_IMG
    if partial_primary_df is not None and len(partial_primary_df) > 0:
        img_partial_primary = plot_partial_rsa(partial_primary_df, "A", fig_dir, "primary")
        img_partial_primary_sr = plot_partial_semi_partial(
            partial_primary_df, "A", fig_dir, "primary")
    else:
        img_partial_primary = img_partial_primary_sr = PLACEHOLDER_IMG
    if partial_secondary_df is not None and len(partial_secondary_df) > 0:
        img_partial_secondary = plot_partial_rsa(partial_secondary_df, "E", fig_dir, "secondary")
        img_partial_secondary_sr = plot_partial_semi_partial(
            partial_secondary_df, "E", fig_dir, "secondary")
    else:
        img_partial_secondary = img_partial_secondary_sr = PLACEHOLDER_IMG
    if partial_primary_df is not None and partial_secondary_df is not None:
        img_a_vs_e = plot_a_vs_e(partial_primary_df, partial_secondary_df, fig_dir)
    else:
        img_a_vs_e = PLACEHOLDER_IMG
    img_category = plot_category_rsa(category_df, fig_dir) if category_df is not None else PLACEHOLDER_IMG

    # Status banner
    missing = []
    if partial_primary_df is None or (partial_primary_df is not None and
            len(partial_primary_df[partial_primary_df["model"] == "A"]) < 41):
        n_done = len(partial_primary_df[partial_primary_df["model"] == "A"]) if partial_primary_df is not None else 0
        missing.append(f"Partial RSA primary ({n_done}/41 layers)")
    if partial_secondary_df is None:
        missing.append("Partial RSA secondary (not started)")
    elif len(partial_secondary_df[partial_secondary_df["model"] == "E"]) < 41:
        n_done = len(partial_secondary_df[partial_secondary_df["model"] == "E"])
        missing.append(f"Partial RSA secondary ({n_done}/41 layers)")
    status_banner = ""
    if missing:
        status_banner = ('<div style="background:#fff3cd; border:2px solid #ffc107; '
                         'padding:15px; margin:15px 0; border-radius:4px;">'
                         '<strong>Partial results:</strong> The following analyses are '
                         'still running: ' + ", ".join(missing) + '. '
                         'Rerun the report generator after they complete.</div>')

    # Build report
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 5: Mental State Attribution RSA — Results</title>
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
    table.conditions {{ border-collapse: collapse; margin: 15px 0; }}
    table.conditions th, table.conditions td {{ padding: 8px 12px;
                                                border: 1px solid #ddd; }}
    table.conditions th {{ background: #f0f0f0; }}
    .sig {{ color: #d62728; font-weight: bold; }}
    code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px;
            font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Experiment 5: Mental State Attribution RSA</h1>

{status_banner}

<div class="summary-box">
<strong>Question:</strong> Does LLaMA-2-13B-Chat maintain a dedicated representational
structure for mental state attributions (He + mental verb + object) that is distinct
from its component parts?<br><br>
<strong>Design:</strong> 56 items &times; 6 conditions = 336 sentences. Each item pairs
a mental state verb with a matched action verb and the same object noun across all
conditions.<br><br>
<strong>Model:</strong> LLaMA-2-13B-Chat. Activations extracted at the <strong>last token
position</strong> (period ".") for all 41 layers.<br><br>
<strong>Statistical testing:</strong> All p-values computed via permutation tests (10,000
iterations, shuffling condition labels within items). Multiple comparisons corrected
using Benjamini-Hochberg FDR across all 41 layers.
</div>

<h2>Stimulus Design</h2>

<table class="conditions">
<tr><th>Condition</th><th>Label</th><th>Example</th><th>Controls for</th></tr>
<tr><td>C1</td><td><code>mental_state</code></td><td>He notices the crack.</td>
    <td>Full attribution: subject + mental verb + object</td></tr>
<tr><td>C2</td><td><code>dis_mental</code></td><td>Notice the crack.</td>
    <td>Mental verb + object, NO subject</td></tr>
<tr><td>C3</td><td><code>scr_mental</code></td><td>The crack to notice.</td>
    <td>Same words as C2, scrambled order</td></tr>
<tr><td>C4</td><td><code>action</code></td><td>He fills the crack.</td>
    <td>Subject + action verb + object (same SVO frame, no mental state)</td></tr>
<tr><td>C5</td><td><code>dis_action</code></td><td>Fill the crack.</td>
    <td>Action verb + object, NO subject</td></tr>
<tr><td>C6</td><td><code>scr_action</code></td><td>The crack to fill.</td>
    <td>Same words as C5, scrambled order</td></tr>
</table>

<p>56 mental state verbs span 7 categories (8 each): Attention, Memory, Sensation,
Belief, Desire, Emotion, Intention. Each paired with a unique concrete action verb
(the "security camera test": identifiable from silent video).</p>

<hr>

<h2>Analysis 1: Simple RSA (Model A &mdash; Full Attribution)</h2>

<div class="method-box">
<strong>Method:</strong> Model A predicts that only Condition 1 (full mental state
attribution) sentences are similar to each other; all other pairs are dissimilar.
Spearman rank correlation between the neural RDM and Model A at each layer.
Permutation test: shuffle condition labels within items (10,000 iterations).
BH-FDR correction across 41 layers.<br><br>
<strong>What this tests:</strong> Do full mental state attribution sentences cluster
together in activation space more than sentences from any other condition? Because
C2/C3 share the same mental verbs and C4 shares the same SVO frame, a significant
result already implies the clustering is not purely lexical or syntactic &mdash; the
stimulus design does the controlling.
</div>

<img src="data:image/png;base64,{img_simple}" alt="Simple RSA layer profile">
<p class="caption"><strong>Figure 1.</strong> Layer-by-layer Spearman correlation between
the neural RDM (correlation distance) and Model A (full attribution). Filled circles
indicate layers significant at FDR &lt; .05 (permutation test, 10,000 iterations).
Model A predicts similarity only among Condition 1 (He + mental verb + object) sentences.</p>

<div class="result-box">
<strong>Summary:</strong> {peak_summary(simple_df, "rho") if simple_df is not None else "Pending."}
</div>

<h3>Layer-by-Layer Results</h3>
{make_layerwise_table(simple_df, "rho") if simple_df is not None else PENDING_HTML}

<hr>

<h2>Analysis 2a: Partial RSA &mdash; Model A (Full Attribution) as Hypothesis</h2>

<div class="method-box">
<strong>Method:</strong> Multiple regression of the neural RDM on Model A (hypothesis)
plus 6 confound models simultaneously:
<ul>
<li><strong>B</strong> &mdash; Mental Verb Presence (C1,C2,C3 similar): lexical confound</li>
<li><strong>C</strong> &mdash; Subject Presence (C1,C4 similar): syntactic confound</li>
<li><strong>D</strong> &mdash; Item Identity (same item similar): word overlap confound</li>
<li><strong>F</strong> &mdash; Grammatical Order (C1,C2,C4,C5 similar): grammaticality confound</li>
<li><strong>G</strong> &mdash; Scrambled Form (C3,C6 similar): scrambled fragment confound</li>
<li><strong>H</strong> &mdash; Action Verb Presence (C4,C5,C6 similar): action lexical confound</li>
</ul>
The standardized beta for Model A tells you: how much variance in the neural RDM is
<em>uniquely</em> explained by the full attribution structure after removing all confound
variance. Semi-partial r quantifies the unique contribution. Significance via permutation
(10,000 iterations). BH-FDR correction across layers.
</div>

<img src="data:image/png;base64,{img_partial_primary}" alt="Partial RSA primary">
<p class="caption"><strong>Figure 2.</strong> Partial RSA with Model A as hypothesis.
<strong>Top:</strong> Standardized beta for Model A (full attribution) across layers,
with confound models (B, C, D, F, G, H) partialed out. Filled circles = FDR &lt; .05.
<strong>Bottom:</strong> Confound model betas from the same regression, showing what each
surface feature contributes. A significant Model A beta means the full attribution
structure explains neural variance <em>beyond</em> all of these confounds.</p>

<div class="result-box">
<strong>Model A summary:</strong> {peak_summary(partial_primary_df, "beta", model_filter="A") if partial_primary_df is not None else "Pending."}<br>
<strong>Model B (mental verb) summary:</strong> {peak_summary(partial_primary_df, "beta", model_filter="B") if partial_primary_df is not None else "Pending."}<br>
<strong>Model D (item identity) summary:</strong> {peak_summary(partial_primary_df, "beta", model_filter="D") if partial_primary_df is not None else "Pending."}<br>
<strong>Model F (grammaticality) summary:</strong> {peak_summary(partial_primary_df, "beta", model_filter="F") if partial_primary_df is not None else "Pending."}
</div>

{f'<img src="data:image/png;base64,{img_partial_primary_sr}" alt="Semi-partial r for Model A">' if img_partial_primary_sr else ""}
<p class="caption"><strong>Figure 3.</strong> Semi-partial r for Model A across layers.
Each bar shows the unique variance in the neural RDM explained by the full attribution
structure after removing all confound variance. Triangles indicate FDR &lt; .05.</p>

<h3>Layer-by-Layer Results: Model A</h3>
{make_layerwise_table(partial_primary_df[partial_primary_df["model"] == "A"].reset_index(drop=True), "beta") if partial_primary_df is not None else PENDING_HTML}

<h3>Layer-by-Layer Results: All Models (Primary Regression)</h3>
{make_layerwise_table(partial_primary_df, "beta") if partial_primary_df is not None else PENDING_HTML}

<hr>

<h2>Analysis 2b: Partial RSA &mdash; Model E (Subject-Optional) as Hypothesis</h2>

<div class="method-box">
<strong>Method:</strong> Same regression framework, but Model A is replaced by Model E,
which predicts that C1 <em>and</em> C2 sentences are similar (mental verb + object in
grammatical order, with or without subject). This tests whether the subject "He" is
necessary for the attribution structure, or whether verb + object binding alone suffices.
</div>

<img src="data:image/png;base64,{img_partial_secondary}" alt="Partial RSA secondary">
<p class="caption"><strong>Figure 4.</strong> Partial RSA with Model E (mental verb + object,
subject-optional) as hypothesis. Same format as Figure 2.</p>

<img src="data:image/png;base64,{img_partial_secondary_sr}" alt="Semi-partial r for Model E">
<p class="caption"><strong>Figure 5.</strong> Semi-partial r for Model E across layers.</p>

<div class="result-box">
<strong>Model E summary:</strong> {peak_summary(partial_secondary_df, "beta", model_filter="E") if partial_secondary_df is not None else "Pending."}
</div>

<h3>Layer-by-Layer Results: Model E</h3>
{make_layerwise_table(partial_secondary_df[partial_secondary_df["model"] == "E"].reset_index(drop=True), "beta") if partial_secondary_df is not None else PENDING_HTML}

<hr>

<h2>Model A vs. Model E: Is the Subject Required?</h2>

<div class="method-box">
<strong>Question:</strong> Does mental state attribution require an explicit agent
("He"), or does the mental verb + object binding suffice?<br><br>
<strong>Interpretation guide:</strong>
<ul>
<li>&beta;<sub>A</sub> significant, &beta;<sub>E</sub> not &rarr; Subject is <em>necessary</em>
    for attribution structure</li>
<li>&beta;<sub>E</sub> significant, &beta;<sub>A</sub> not &rarr; Mental verb + object
    binding <em>suffices</em> without subject</li>
<li>Both significant &rarr; Core verb+object structure that the subject enriches</li>
<li>&beta;<sub>E</sub> significant, &beta;<sub>A</sub> not beyond confounds &rarr;
    Subject adds nothing beyond verb+object</li>
</ul>
</div>

<img src="data:image/png;base64,{img_a_vs_e}" alt="Model A vs E comparison">
<p class="caption"><strong>Figure 6.</strong> Direct comparison of standardized betas for
Model A (full attribution, requires subject) and Model E (subject-optional) from their
respective partial RSA regressions. Filled markers = FDR &lt; .05.</p>

<hr>

<h2>Analysis 3: Within-Condition Category Structure RSA</h2>

<div class="method-box">
<strong>Question:</strong> Within the mental state attribution sentences (C1 only), does
the model organize representations according to the 7 verb categories (Attention, Memory,
Sensation, Belief, Desire, Emotion, Intention)?<br><br>
<strong>Method:</strong> Compute 56&times;56 neural RDM for C1 sentences only. Model Cat
predicts same-category pairs are similar, different-category pairs are dissimilar.
Spearman correlation + permutation test (shuffle category labels across items, 10,000
iterations). BH-FDR across layers.<br><br>
<strong>Cross-condition comparison:</strong> The same category RSA is also run on each
of the other 5 conditions independently. If category structure appears in C1 but not
C2&ndash;C6, it specifically requires the full attribution form. If it also appears in C2
(disembodied mental), verb semantics alone organize the space.
</div>

<img src="data:image/png;base64,{img_category}" alt="Category RSA by condition">
<p class="caption"><strong>Figure 7.</strong> Category structure RSA (7 mental state verb
categories) computed separately for each condition across layers. Bold red line = C1
(full attribution); filled circles = FDR &lt; .05 for C1. If category structure is
strongest or only significant in C1, it requires the full attribution form to emerge.</p>

<div class="result-box">
<strong>C1 (mental_state) summary:</strong>
{peak_summary(category_df[category_df["condition"] == "mental_state"].reset_index(drop=True), "rho") if category_df is not None else "Pending."}<br>
<strong>C2 (dis_mental) summary:</strong>
{peak_summary(category_df[category_df["condition"] == "dis_mental"].reset_index(drop=True), "rho") if category_df is not None else "Pending."}<br>
<strong>C4 (action) summary:</strong>
{peak_summary(category_df[category_df["condition"] == "action"].reset_index(drop=True), "rho") if category_df is not None else "Pending."}
</div>

<h3>Layer-by-Layer Results: Category RSA (C1 — Mental State)</h3>
{make_layerwise_table(category_df[category_df["condition"] == "mental_state"].reset_index(drop=True), "rho") if category_df is not None else PENDING_HTML}

<hr>

<h2>Interpretation Guide</h2>

<div class="interpretation-box">
<h3>What would each pattern of results mean?</h3>
<ul>
<li><strong>Model A significant (Analysis 1 & 2a):</strong> The model has a dedicated
representational structure for bound mental state attributions that cannot be reduced to
lexical (mental vocabulary), syntactic (SVO frame), or surface (word overlap, grammaticality)
features.</li>

<li><strong>Model B significant but not A (Analysis 2a):</strong> The clustering is driven
by shared mental vocabulary &mdash; the model groups mental verbs together regardless of
whether they appear in a full attribution frame.</li>

<li><strong>Model D significant but not A:</strong> The clustering is driven by shared
object nouns &mdash; items cluster by word overlap, not by condition structure.</li>

<li><strong>Category structure in C1 only (Analysis 3):</strong> The 7-category organization
of mental states requires the full attribution form. The model doesn't just know that
"fears" and "dreads" are similar words; it represents them as similar <em>kinds of mental
state attributions</em>.</li>

<li><strong>Category structure in C1 and C2:</strong> Verb semantics alone organize the
representational space. Still interesting, but a weaker claim about attribution-specific
structure.</li>
</ul>
</div>

<hr>
<p style="font-size:0.85em; color:#888;">
Generated by <code>2a_rsa_report_generator.py</code>.
Regenerate: <code>python code/2a_rsa_report_generator.py --model {args.model}</code>
</p>

</body>
</html>"""

    with open(out_html, "w") as f:
        f.write(html)
    print(f"Report saved: {out_html}")


if __name__ == "__main__":
    main()
