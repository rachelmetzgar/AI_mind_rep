#!/usr/bin/env python3
"""
Experiment 5, Phase 13a: Probe Training Report Generator

Generates a comprehensive HTML report from saved probe training results.
Standalone — regenerates from CSVs/JSONs without recomputing.

Output:
    results/{model}/probe_training/probe_report.html

Usage:
    python code/13a_probe_report_generator.py --model llama2_13b_chat

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
    ensure_dir, figures_dir, CONDITION_LABELS, CATEGORY_LABELS,
    POSITION_LABELS,
)


# ── Colors and labels ────────────────────────────────────────────────────────

PROBE_COLORS = {
    "subject_presence": "#d62728",
    "mental_verb": "#1f77b4",
    "grammaticality": "#2ca02c",
    "action_verb": "#ff7f0e",
    "c1_vs_all": "#9467bd",
    "c1_vs_c2": "#d62728",
    "c4_vs_c5": "#1f77b4",
}

POSITION_COLORS = {
    "verb": "#d62728",
    "object": "#1f77b4",
    "period": "#2ca02c",
}

PROBE_NAMES = {
    "subject_presence": "Subject Presence",
    "mental_verb": "Mental Verb",
    "grammaticality": "Grammaticality",
    "action_verb": "Action Verb",
    "c1_vs_all": "C1 vs All",
    "c1_vs_c2": "C1 vs C2",
    "c4_vs_c5": "C4 vs C5",
}

POSITION_NAMES = {
    "verb": "Verb",
    "object": "Object",
    "period": "Period",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

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


PENDING_HTML = (
    '<div class="result-box pending">'
    '<strong>Data pending</strong> — analysis not yet complete. '
    'Rerun report generator after data files are produced.'
    '</div>'
)


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_feature_probe_heatmaps(df, fig_dir):
    """
    Feature probe AUC heatmaps — one heatmap per probe.
    x-axis = layer (0-40), y-axis = position (verb/object/period).
    Colormap = AUC values. Significant cells (p_fdr < 0.05) marked with dots.
    """
    probes = sorted(df["probe_name"].unique())
    n_probes = len(probes)
    fig, axes = plt.subplots(n_probes, 1, figsize=(14, 3.0 * n_probes),
                             squeeze=False)

    for i, probe in enumerate(probes):
        ax = axes[i, 0]
        pdf = df[df["probe_name"] == probe]

        positions = POSITION_LABELS
        layers = sorted(pdf["layer"].unique())

        # Build matrix: rows = positions, cols = layers
        auc_matrix = np.full((len(positions), len(layers)), np.nan)
        sig_matrix = np.zeros((len(positions), len(layers)), dtype=bool)

        for pi, pos in enumerate(positions):
            pos_df = pdf[pdf["position"] == pos].sort_values("layer")
            for _, row in pos_df.iterrows():
                li = layers.index(int(row["layer"]))
                auc_matrix[pi, li] = row["auc"]
                p_val = row["p_fdr"] if "p_fdr" in row.index else row.get("p_perm", 1.0)
                sig_matrix[pi, li] = p_val < 0.05

        im = ax.imshow(auc_matrix, aspect="auto", cmap="RdYlBu_r",
                        vmin=0.4, vmax=1.0, interpolation="nearest")

        # Mark significant cells with dots
        sig_y, sig_x = np.where(sig_matrix)
        if len(sig_y) > 0:
            ax.scatter(sig_x, sig_y, color="black", s=15, zorder=5, marker="o")

        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, fontsize=7)
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels([POSITION_NAMES.get(p, p) for p in positions], fontsize=10)

        color = PROBE_COLORS.get(probe, "#333333")
        ax.set_title(f"{PROBE_NAMES.get(probe, probe)} Probe — AUC by Layer and Position",
                     fontsize=12, color=color, fontweight="bold")

        if i == n_probes - 1:
            ax.set_xlabel("Layer", fontsize=11)

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label("AUC", fontsize=9)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    fig.tight_layout()
    save_fig(fig, fig_dir / "feature_probe_heatmaps.png")
    return fig_to_base64(fig)


def plot_attribution_probe_profiles(df, fig_dir):
    """
    Attribution probe layer profiles — AUC vs layer.
    One line per position, one subplot per probe. Mark significant points.
    """
    probes = sorted(df["probe_name"].unique())
    n_probes = len(probes)
    fig, axes = plt.subplots(1, n_probes, figsize=(6 * n_probes, 4), squeeze=False)

    for i, probe in enumerate(probes):
        ax = axes[0, i]
        pdf = df[df["probe_name"] == probe]

        for pos in POSITION_LABELS:
            pos_df = pdf[pdf["position"] == pos].sort_values("layer")
            if pos_df.empty:
                continue

            layers = pos_df["layer"].values
            aucs = pos_df["auc"].values
            p_fdr = safe_p_fdr(pos_df)

            color = POSITION_COLORS.get(pos, "#333333")
            ax.plot(layers, aucs, color=color, linewidth=2,
                    label=POSITION_NAMES.get(pos, pos))

            sig = p_fdr < 0.05
            if sig.any():
                ax.scatter(layers[sig], aucs[sig], color=color, s=30, zorder=5,
                           edgecolors="black", linewidth=0.5)

        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", label="Chance")
        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("AUC", fontsize=11)
        ax.set_title(PROBE_NAMES.get(probe, probe), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.set_ylim(0.35, 1.05)

    fig.suptitle("Attribution Probes — AUC by Layer and Position", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir / "attribution_probe_profiles.png")
    return fig_to_base64(fig)


def plot_critical_tests(ct, fig_dir):
    """
    Critical tests summary panel — three subplots:
    (a) residual AUC with null distribution
    (b) interaction AUC with null distribution
    (c) cosine similarity with 95% CI errorbar
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Residual AUC
    ax = axes[0]
    res_auc = ct.get("residual_auc", None)
    res_p = ct.get("residual_p", None)
    res_null = ct.get("residual_null_dist", None)

    if res_null is not None and len(res_null) > 0:
        ax.hist(res_null, bins=40, color="#cccccc", edgecolor="white", alpha=0.8,
                label="Null distribution")
    if res_auc is not None:
        ax.axvline(res_auc, color="#d62728", linewidth=2.5, linestyle="-",
                   label=f"Observed = {res_auc:.3f}")
    ax.set_xlabel("AUC", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    p_str = f"p = {res_p:.4f}" if res_p is not None else "p = N/A"
    ax.set_title(f"(a) Residual AUC ({p_str})", fontsize=12)
    ax.legend(fontsize=9)

    # (b) Interaction AUC
    ax = axes[1]
    int_auc = ct.get("interaction_auc", None)
    int_p = ct.get("interaction_p", None)
    int_null = ct.get("interaction_null_dist", None)

    if int_null is not None and len(int_null) > 0:
        ax.hist(int_null, bins=40, color="#cccccc", edgecolor="white", alpha=0.8,
                label="Null distribution")
    if int_auc is not None:
        ax.axvline(int_auc, color="#1f77b4", linewidth=2.5, linestyle="-",
                   label=f"Observed = {int_auc:.3f}")
    ax.set_xlabel("AUC", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    p_str = f"p = {int_p:.4f}" if int_p is not None else "p = N/A"
    ax.set_title(f"(b) Interaction AUC ({p_str})", fontsize=12)
    ax.legend(fontsize=9)

    # (c) Cosine similarity with CI
    ax = axes[2]
    cos_obs = ct.get("cosine_observed", None)
    cos_lo = ct.get("cosine_ci_low", None)
    cos_hi = ct.get("cosine_ci_high", None)

    if cos_obs is not None:
        yerr_lo = cos_obs - cos_lo if cos_lo is not None else 0
        yerr_hi = cos_hi - cos_obs if cos_hi is not None else 0
        ax.bar([0], [cos_obs], color="#2ca02c", alpha=0.8, width=0.5,
               yerr=[[yerr_lo], [yerr_hi]], capsize=8, ecolor="black")
        ax.text(0, cos_obs + yerr_hi + 0.02, f"{cos_obs:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks([0])
    ax.set_xticklabels(["Cosine Similarity"], fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ci_str = ""
    if cos_lo is not None and cos_hi is not None:
        ci_str = f" [{cos_lo:.3f}, {cos_hi:.3f}]"
    ax.set_title(f"(c) Direction Cosine Similarity{ci_str}", fontsize=12)
    ax.set_ylim(-0.2, 1.1)

    fig.suptitle("Critical Tests Summary", fontsize=14, y=1.02)
    fig.tight_layout()
    save_fig(fig, fig_dir / "critical_tests_panel.png")
    return fig_to_base64(fig)


def plot_projected_rsa(proj_df, orig_rsa_dir, fig_dir):
    """
    Projected RSA comparison — bar chart comparing original RSA rho
    vs probe-projected RSA for each model, at peak layer.
    """
    # Try to load original RSA simple results for comparison
    orig_simple = None
    orig_path = orig_rsa_dir / "simple_rsa_results.csv"
    if orig_path.exists():
        orig_simple = pd.read_csv(orig_path)

    models = sorted(proj_df["model"].unique()) if "model" in proj_df.columns else ["A"]
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: layer profiles
    ax = axes[0]
    for model in models:
        mdf = proj_df[proj_df["model"] == model] if "model" in proj_df.columns else proj_df
        mdf = mdf.sort_values("layer")
        val_col = "rho" if "rho" in mdf.columns else "beta"
        layers = mdf["layer"].values
        vals = mdf[val_col].values
        p_fdr = safe_p_fdr(mdf)

        color = PROBE_COLORS.get(model, "#333333")
        ax.plot(layers, vals, linewidth=2, label=f"Projected {model}", color=color)

        sig = p_fdr < 0.05
        if sig.any():
            ax.scatter(layers[sig], vals[sig], color=color, s=30, zorder=5,
                       edgecolors="black", linewidth=0.5)

    if orig_simple is not None:
        ax.plot(orig_simple["layer"].values, orig_simple["rho"].values,
                linewidth=2, linestyle="--", color="#888888", label="Original RSA")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=11)
    val_label = "rho" if "rho" in proj_df.columns else "beta"
    ax.set_ylabel(val_label, fontsize=11)
    ax.set_title("Projected RSA — Layer Profiles", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Right: peak comparison bars
    ax = axes[1]
    bar_labels = []
    bar_vals = []
    bar_colors = []

    if orig_simple is not None:
        peak_idx = orig_simple["rho"].abs().idxmax()
        bar_labels.append("Original RSA")
        bar_vals.append(orig_simple.loc[peak_idx, "rho"])
        bar_colors.append("#888888")

    for model in models:
        mdf = proj_df[proj_df["model"] == model] if "model" in proj_df.columns else proj_df
        val_col = "rho" if "rho" in mdf.columns else "beta"
        peak_idx = mdf[val_col].abs().idxmax()
        bar_labels.append(f"Projected {model}")
        bar_vals.append(mdf.loc[peak_idx, val_col])
        bar_colors.append(PROBE_COLORS.get(model, "#333333"))

    x = np.arange(len(bar_labels))
    ax.bar(x, bar_vals, color=bar_colors, alpha=0.8, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"Peak {val_label}", fontsize=11)
    ax.set_title("Peak RSA: Original vs Projected", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    save_fig(fig, fig_dir / "projected_rsa_comparison.png")
    return fig_to_base64(fig)


def plot_category_probes(df, fig_dir):
    """
    Category probe results — layer profile of accuracy for each analysis type,
    with chance line at 14.3% (1/7 categories).
    """
    analyses = sorted(df["analysis"].unique())
    analysis_colors = {
        "c1_only": "#d62728",
        "c2_only": "#1f77b4",
        "cross_condition": "#2ca02c",
    }
    analysis_names = {
        "c1_only": "C1 Only (train & test on C1)",
        "c2_only": "C2 Only (train & test on C2)",
        "cross_condition": "Cross-Condition (train C1, test C2)",
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for analysis in analyses:
        adf = df[df["analysis"] == analysis].sort_values("layer")
        if adf.empty:
            continue

        layers = adf["layer"].values
        accs = adf["accuracy"].values
        p_fdr = safe_p_fdr(adf)

        color = analysis_colors.get(analysis, "#333333")
        label = analysis_names.get(analysis, analysis)
        ax.plot(layers, accs * 100, color=color, linewidth=2, label=label)

        sig = p_fdr < 0.05
        if sig.any():
            ax.scatter(layers[sig], accs[sig] * 100, color=color, s=30, zorder=5,
                       edgecolors="black", linewidth=0.5)

    # Chance line
    chance = df["chance"].iloc[0] if "chance" in df.columns else 1.0 / 7.0
    ax.axhline(chance * 100, color="gray", linewidth=1.2, linestyle="--",
               label=f"Chance ({chance*100:.1f}%)")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Category Probes — 7-Way Verb Category Classification", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_ylim(0, 105)

    fig.tight_layout()
    save_fig(fig, fig_dir / "category_probe_profiles.png")
    return fig_to_base64(fig)


# ── Summary helpers ──────────────────────────────────────────────────────────

def peak_summary_probe(df, value_col="auc", p_col="p_fdr", probe_filter=None,
                       pos_filter=None, analysis_filter=None):
    """Find peak layer for a probe and summarize."""
    sub = df.copy()
    if probe_filter is not None:
        sub = sub[sub["probe_name"] == probe_filter]
    if pos_filter is not None:
        sub = sub[sub["position"] == pos_filter]
    if analysis_filter is not None:
        sub = sub[sub["analysis"] == analysis_filter]
    if sub.empty:
        return "No data."

    if p_col not in sub.columns:
        p_col = "p_perm" if "p_perm" in sub.columns else ("p" if "p" in sub.columns else None)

    peak_idx = sub[value_col].abs().idxmax()
    peak = sub.loc[peak_idx]
    sig_layers = sub[sub[p_col] < 0.05]["layer"].tolist() if p_col else []
    sig_str = ", ".join(str(int(l)) for l in sig_layers) if sig_layers else "none"
    p_val = peak[p_col] if p_col and p_col in peak.index else float("nan")
    p_label = p_col if p_col else "p"

    return (f"Peak layer: <strong>{int(peak['layer'])}</strong> "
            f"({value_col}={peak[value_col]:.4f}, {p_label}={p_val:.4f}). "
            f"Significant layers (FDR &lt; .05): <strong>{sig_str}</strong>.")


# ── Main report ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate probe training report")
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    probe_data = data_dir("probe_training")
    fig_dir = ensure_dir(figures_dir("probe_training"))
    rsa_data = data_dir("rsa")
    out_html = results_dir("probe_training") / "probe_report.html"
    ensure_dir(out_html.parent)

    print(f"Loading data from: {probe_data}")

    # ── Load data — handle missing files gracefully ──────────────────────
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

    feature_df = load_csv("feature_probe_results.csv")
    attribution_df = load_csv("attribution_probe_results.csv")
    critical_tests = load_json("critical_tests.json")
    projected_rsa_df = load_csv("projected_rsa_results.csv")
    projected_cat_rsa_df = load_csv("projected_category_rsa_results.csv")
    category_probe_df = load_csv("category_probe_results.csv")

    # ── Generate figures ─────────────────────────────────────────────────
    img_feature = ""
    if feature_df is not None:
        img_feature = plot_feature_probe_heatmaps(feature_df, fig_dir)

    img_attribution = ""
    if attribution_df is not None:
        img_attribution = plot_attribution_probe_profiles(attribution_df, fig_dir)

    img_critical = ""
    if critical_tests is not None:
        img_critical = plot_critical_tests(critical_tests, fig_dir)

    img_projected = ""
    if projected_rsa_df is not None:
        img_projected = plot_projected_rsa(projected_rsa_df, rsa_data, fig_dir)

    img_category = ""
    if category_probe_df is not None:
        img_category = plot_category_probes(category_probe_df, fig_dir)

    # ── Feature probe summary text ───────────────────────────────────────
    feature_summary = ""
    if feature_df is not None:
        probes = sorted(feature_df["probe_name"].unique())
        parts = []
        for probe in probes:
            for pos in POSITION_LABELS:
                parts.append(
                    f"<strong>{PROBE_NAMES.get(probe, probe)} @ {POSITION_NAMES.get(pos, pos)}:</strong> "
                    + peak_summary_probe(feature_df, "auc", "p_fdr",
                                         probe_filter=probe, pos_filter=pos)
                )
        feature_summary = "<br>".join(parts)

    # ── Attribution probe summary text ───────────────────────────────────
    attribution_summary = ""
    if attribution_df is not None:
        probes = sorted(attribution_df["probe_name"].unique())
        parts = []
        for probe in probes:
            parts.append(
                f"<strong>{PROBE_NAMES.get(probe, probe)}:</strong> "
                + peak_summary_probe(attribution_df, "auc", "p_fdr",
                                     probe_filter=probe)
            )
        attribution_summary = "<br>".join(parts)

    # ── Critical tests summary text ──────────────────────────────────────
    critical_summary = ""
    if critical_tests is not None:
        ct = critical_tests
        parts = []
        res_auc = ct.get("residual_auc")
        res_p = ct.get("residual_p")
        if res_auc is not None:
            sig = "significant" if (res_p is not None and res_p < 0.05) else "not significant"
            parts.append(f"<strong>Residual AUC:</strong> {res_auc:.4f} (p = {res_p:.4f}, {sig})")

        int_auc = ct.get("interaction_auc")
        int_p = ct.get("interaction_p")
        if int_auc is not None:
            sig = "significant" if (int_p is not None and int_p < 0.05) else "not significant"
            parts.append(f"<strong>Interaction AUC:</strong> {int_auc:.4f} (p = {int_p:.4f}, {sig})")

        cos_obs = ct.get("cosine_observed")
        cos_lo = ct.get("cosine_ci_low")
        cos_hi = ct.get("cosine_ci_high")
        if cos_obs is not None:
            ci_str = f" [95% CI: {cos_lo:.4f}, {cos_hi:.4f}]" if cos_lo is not None else ""
            parts.append(f"<strong>Cosine similarity:</strong> {cos_obs:.4f}{ci_str}")

        critical_summary = "<br>".join(parts)

    # ── Category probe summary text ──────────────────────────────────────
    category_summary = ""
    if category_probe_df is not None:
        analyses = sorted(category_probe_df["analysis"].unique())
        parts = []
        for analysis in analyses:
            parts.append(
                f"<strong>{analysis}:</strong> "
                + peak_summary_probe(category_probe_df, "accuracy", "p_perm",
                                     analysis_filter=analysis)
            )
        category_summary = "<br>".join(parts)

    # ── Build HTML ───────────────────────────────────────────────────────
    fig_num = 1

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 5: Probe Training Results</title>
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
</style>
</head>
<body>

<h1>Experiment 5: Probe Training Results</h1>

<div class="summary-box">
<strong>Purpose:</strong> Train linear probes on LLaMA-2-13B-Chat activations
to test whether the model encodes specific features of mental state attribution
sentences (subject presence, mental verb type, grammaticality, action verb type)
and whether full attribution structure (C1) is decodable beyond its component features.<br><br>
<strong>Probes:</strong> Feature probes (binary classifiers for individual sentence features)
and attribution probes (C1 vs other conditions). Trained at three token positions
(verb, object, period) across all 41 layers.<br><br>
<strong>Critical tests:</strong> Residual AUC (attribution decodability after regressing out
feature probe predictions), interaction AUC (feature conjunction), and direction cosine
similarity (comparing probe weight vectors).
</div>

<hr>

<h2>1. Feature Probes</h2>

<div class="method-box">
<strong>Method:</strong> Train binary logistic probes for four sentence features:
<ul>
<li><strong>Subject Presence</strong> — C1,C4 (subject) vs C2,C3,C5,C6 (no subject)</li>
<li><strong>Mental Verb</strong> — C1,C2,C3 (mental verb) vs C4,C5,C6 (action verb)</li>
<li><strong>Grammaticality</strong> — C1,C2,C4,C5 (grammatical) vs C3,C6 (scrambled)</li>
<li><strong>Action Verb</strong> — C4,C5,C6 (action verb) vs C1,C2,C3 (mental verb)</li>
</ul>
Each probe trained at verb, object, and period token positions. AUC evaluated via
cross-validation. Significance by permutation test (label shuffling), FDR-corrected
across layers.
</div>
"""

    if feature_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_feature}" alt="Feature probe AUC heatmaps">
<p class="caption"><strong>Figure {fig_num}.</strong> Feature probe AUC heatmaps. Each panel
shows one probe type. X-axis = layer (0-40), y-axis = token position. Color = AUC;
black dots mark cells significant at FDR &lt; .05.</p>

<div class="result-box">
{feature_summary}
</div>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    html += """
<hr>

<h2>2. Attribution Probes</h2>

<div class="method-box">
<strong>Method:</strong> Train binary probes to discriminate attribution conditions:
<ul>
<li><strong>C1 vs All</strong> — full attribution (C1) vs all other conditions</li>
<li><strong>C1 vs C2</strong> — full attribution (C1) vs disembodied mental (C2). Tests whether
    the subject contributes beyond mental verb + object.</li>
<li><strong>C4 vs C5</strong> — action control: He + action verb (C4) vs action verb alone (C5).
    Matched syntactic contrast to C1 vs C2.</li>
</ul>
AUC layer profiles at each position. Significant points marked (FDR &lt; .05).
</div>
"""

    if attribution_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_attribution}" alt="Attribution probe layer profiles">
<p class="caption"><strong>Figure {fig_num}.</strong> Attribution probe AUC across layers.
Each subplot shows one probe; lines colored by token position. Filled circles = FDR &lt; .05.
Dashed line at 0.5 = chance.</p>

<div class="result-box">
{attribution_summary}
</div>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    html += """
<hr>

<h2>3. Critical Tests</h2>

<div class="method-box">
<strong>Purpose:</strong> Test whether the attribution structure captured by probes is
<em>more than the sum of its parts</em> — i.e., whether C1 vs other conditions can be
decoded above and beyond what is predicted by the feature probes.<br><br>
<strong>Tests:</strong>
<ul>
<li><strong>Residual AUC:</strong> Train a logistic regression to predict C1 membership from
    feature-probe predicted probabilities alone. Then test whether the <em>residuals</em>
    from this model can still be predicted from activations. If residual AUC &gt; chance,
    activations encode attribution structure beyond decomposable features.</li>
<li><strong>Interaction AUC:</strong> Test whether the conjunction of features (subject
    &times; mental verb) is decodable beyond the additive combination. A significant
    interaction AUC implies the model represents the <em>binding</em> of subject + mental verb,
    not just each feature independently.</li>
<li><strong>Direction cosine:</strong> Cosine similarity between the C1-vs-All probe weight
    vector and the vector predicted by summing feature probe weights. Low cosine indicates
    the attribution direction is geometrically distinct from the feature directions.</li>
</ul>
</div>
"""

    if critical_tests is not None:
        html += f"""
<img src="data:image/png;base64,{img_critical}" alt="Critical tests summary panel">
<p class="caption"><strong>Figure {fig_num}.</strong> Critical tests. (a) Residual AUC after
removing feature-probe variance; red line = observed, histogram = null distribution from
permutation. (b) Interaction AUC for subject &times; mental verb conjunction. (c) Cosine
similarity between attribution probe direction and feature-predicted direction, with
95% bootstrap CI.</p>

<div class="result-box">
{critical_summary}
</div>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    html += """
<hr>

<h2>4. Projected RSA</h2>

<div class="method-box">
<strong>Purpose:</strong> Test whether the representational structure found by RSA
(Analysis 1-2) can be recovered when projecting activations onto the probe-discovered
directions. If probe directions capture the attribution structure, projected RSA
should show similar (or stronger) effects compared to original RSA on raw activations.<br><br>
<strong>Method:</strong> Project activations onto each probe's weight vector, compute
RDM from the projected (1D) activations, and run the same RSA models (simple and partial)
as in the RSA pipeline.
</div>
"""

    if projected_rsa_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_projected}" alt="Projected RSA comparison">
<p class="caption"><strong>Figure {fig_num}.</strong> Left: Projected RSA layer profiles
compared to original RSA. Right: Peak values for original vs. projected RSA.</p>

<div class="result-box">
<strong>Projected RSA:</strong> See figure for comparison of original and projected RSA
values across layers.
</div>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    # Projected category RSA
    if projected_cat_rsa_df is not None:
        html += """
<h3>Projected Category RSA</h3>
<div class="result-box">
<strong>Category RSA on projected activations:</strong><br>
"""
        peak_idx = projected_cat_rsa_df["rho"].abs().idxmax()
        peak = projected_cat_rsa_df.loc[peak_idx]
        p_col = "p_fdr" if "p_fdr" in projected_cat_rsa_df.columns else "p"
        sig_layers = projected_cat_rsa_df[projected_cat_rsa_df[p_col] < 0.05]["layer"].tolist()
        sig_str = ", ".join(str(int(l)) for l in sig_layers) if sig_layers else "none"
        html += (
            f"Peak layer: <strong>{int(peak['layer'])}</strong> "
            f"(rho={peak['rho']:.4f}, {p_col}={peak[p_col]:.4f}). "
            f"Significant layers (FDR &lt; .05): <strong>{sig_str}</strong>."
        )
        html += "</div>"

    html += """
<hr>

<h2>5. Category Probes</h2>

<div class="method-box">
<strong>Purpose:</strong> Test whether the 7 verb categories (Attention, Memory, Sensation,
Belief, Desire, Emotion, Intention) can be decoded from activations, and whether
this category structure generalizes across conditions.<br><br>
<strong>Analyses:</strong>
<ul>
<li><strong>C1 only:</strong> Train and test on C1 sentences (within-condition).</li>
<li><strong>C2 only:</strong> Train and test on C2 sentences (disembodied mental verbs).</li>
<li><strong>Cross-condition:</strong> Train on C1, test on C2. If this works, category
    structure generalizes beyond the full attribution form.</li>
</ul>
Chance level: 14.3% (1/7 categories). Significance by permutation test.
</div>
"""

    if category_probe_df is not None:
        html += f"""
<img src="data:image/png;base64,{img_category}" alt="Category probe layer profiles">
<p class="caption"><strong>Figure {fig_num}.</strong> Category probe accuracy (7-way
classification) across layers. Each line = one analysis type. Dashed line = chance (14.3%).
Filled circles = significant at p &lt; .05 (permutation).</p>

<div class="result-box">
{category_summary}
</div>
"""
        fig_num += 1
    else:
        html += PENDING_HTML

    # ── Interpretation guide ─────────────────────────────────────────────
    html += """
<hr>

<h2>Interpretation Guide</h2>

<div class="interpretation-box">
<h3>What would each pattern of results mean?</h3>
<ul>
<li><strong>Feature probes all significant:</strong> The model encodes all individual sentence
    features (subject, mental verb, grammaticality) as linearly decodable dimensions. This is a
    prerequisite for the critical tests.</li>

<li><strong>Attribution probes significant + Residual AUC significant:</strong> The model
    encodes attribution structure (C1 distinctiveness) <em>beyond</em> the sum of its features.
    This is evidence for compositional mental state binding.</li>

<li><strong>Interaction AUC significant:</strong> The model represents the <em>conjunction</em>
    of subject + mental verb as a distinct feature, not just an additive combination.</li>

<li><strong>Low cosine similarity:</strong> The attribution probe direction is geometrically
    distinct from the feature probe directions, suggesting a dedicated representational axis
    for bound mental state attributions.</li>

<li><strong>C1 vs C2 significant but C4 vs C5 not:</strong> The subject matters specifically
    for mental state sentences, not for action sentences. The binding is mental-state-specific.</li>

<li><strong>Category probes: C1 &gt; C2, cross-condition works:</strong> Category structure
    is strongest in full attributions but the verb category axis generalizes, suggesting shared
    but enriched representations.</li>

<li><strong>Category probes: C1 only, cross fails:</strong> Category structure requires the
    full attribution form and does not transfer, supporting a truly compositional
    representation.</li>
</ul>
</div>

<hr>
"""

    html += f"""
<p style="font-size:0.85em; color:#888;">
Generated by <code>13a_probe_report_generator.py</code>.
Regenerate: <code>python code/13a_probe_report_generator.py --model {args.model}</code>
</p>

</body>
</html>"""

    with open(out_html, "w") as f:
        f.write(html)
    print(f"\nReport saved: {out_html}")


if __name__ == "__main__":
    main()
