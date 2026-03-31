#!/usr/bin/env python3
"""
Experiment 5: Reduced 1-4 RSA Report Generator (HTML)

Generates HTML report for the 5-predictor reduced RSA (A-E, C1-C4 only).
Includes: stimulus design, model specification, VIF check, model correlations,
Model A beta + unique variance, all model betas, full results table, model RDMs.

Output:
    results/{model}/rsa/reports/
        5_predictors_report_corr.html

Usage:
    python code/rsa/5_predictors/2_reduced_1_4_report.py --model llama2_13b_chat

Env: llama2_env (no GPU needed)
Rachel C. Metzgar · Mar 2026
"""

import sys
import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    set_model, add_model_argument, data_dir, results_dir, ensure_dir,
    N_ITEMS, N_CONDITIONS, POSITION_LABELS,
)
from stimuli import STIMULI

# ── Position support ─────────────────────────────────────────────────────────

POSITION_MAP = {"verb": 0, "object": 1, "period": 2}
_active_position = "period"


def _load_acts_for_position(position):
    """Load activations for the given position. Returns (336, n_layers, hidden_dim)."""
    if position == "period":
        act_path = data_dir("activations") / "activations_last_token.npz"
        if not act_path.exists():
            return None
        return np.load(act_path)["activations"].astype(np.float32)
    else:
        act_path = data_dir("activations") / "activations_multipos.npz"
        if not act_path.exists():
            return None
        acts_full = np.load(act_path)["activations"].astype(np.float32)
        return acts_full[:, POSITION_MAP[position], :, :]

# ── Constants ────────────────────────────────────────────────────────────────

N_CONDS_REDUCED = 4

ALL_MODEL_KEYS = ["A", "B", "C", "D", "E"]

COLORS = {
    "A": "#d62728",  # red
    "B": "#1f77b4",  # blue
    "C": "#ff7f0e",  # orange
    "D": "#2ca02c",  # green
    "E": "#9467bd",  # purple
}

MODEL_NAMES = {
    "A": "Full Attribution",
    "B": "Mental Verb + Object",
    "C": "Mental Verb Presence",
    "D": "Verb + Object",
    "E": "Subject + Verb + Object",
}

MODEL_CONDS = {
    "A": "{C1}",
    "B": "{C1, C2}",
    "C": "{C1, C2, C3}",
    "D": "{C1, C2, C4}",
    "E": "{C1, C4}",
}

MODEL_PURPOSE = {
    "A": "Target: bound subject + mental verb + object",
    "B": "Controls for grammatical mental-verb-object binding without subject",
    "C": "Controls for mental vocabulary clustering",
    "D": "Controls for grammatical verb+object binding regardless of verb type",
    "E": "Controls for having a subject with a grammatical verb+object frame",
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


def sig_marker_fdr(p_fdr):
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


def sig_marker_uncorr(p):
    try:
        if np.isnan(p):
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


# ── Plot: Model A beta + unique variance with significance ──────────────────

def plot_model_a_with_sig(df, fig_dir, metric_label):
    a_df = df[df["model"] == "A"].sort_values("layer")
    layers = a_df["layer"].values
    betas = a_df["beta"].values
    delta_r2 = a_df["delta_r2"].values
    p_fdr = a_df["p_fdr"].values if "p_fdr" in a_df.columns else np.ones(len(a_df))
    p_raw = a_df["p"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [1.2, 1]})

    ax1.plot(layers, betas, color=COLORS["A"], linewidth=2.5,
             label="Model A: Full Attribution")
    ax1.fill_between(layers, 0, betas, alpha=0.15, color=COLORS["A"])
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for i, layer in enumerate(layers):
        fdr_sig = sig_marker_fdr(p_fdr[i])
        raw_sig = sig_marker_uncorr(p_raw[i])
        if fdr_sig:
            ax1.annotate(fdr_sig, (layer, betas[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, fontweight="bold",
                        color=COLORS["A"])
        elif raw_sig:
            ax1.plot(layer, betas[i], marker="^", color=COLORS["A"],
                    markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                    zorder=5)

    ax1.set_ylabel("Standardized beta", fontsize=12)
    ax1.set_title(f"Model A: Full Attribution — Beta (confounds partialed out) [{metric_label}]",
                  fontsize=13)
    ax1.legend(fontsize=10, loc="upper left")

    ax2.bar(layers, delta_r2, color=COLORS["A"], alpha=0.7, width=0.8)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    for i, layer in enumerate(layers):
        fdr_sig = sig_marker_fdr(p_fdr[i])
        raw_sig = sig_marker_uncorr(p_raw[i])
        if fdr_sig:
            ax2.annotate(fdr_sig, (layer, delta_r2[i]), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8, fontweight="bold",
                        color=COLORS["A"])
        elif raw_sig:
            ax2.plot(layer, delta_r2[i] + 0.0001, marker="^", color=COLORS["A"],
                    markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                    zorder=5)

    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Unique variance (ΔR²)", fontsize=12)
    ax2.set_title("Model A: Unique variance after partialing out all confounds", fontsize=13)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.set_xlim(-0.5, layers.max() + 0.5)

    fig.tight_layout()
    fname = f"model_a_beta_unique_variance_{metric_label.lower()}.png"
    save_fig(fig, fig_dir / fname)
    return fig_to_base64(fig)


# ── Plot: All model betas ───────────────────────────────────────────────────

def plot_all_betas(df, fig_dir, metric_label):
    fig, ax = plt.subplots(figsize=(12, 6))

    for k in ALL_MODEL_KEYS:
        k_df = df[df["model"] == k].sort_values("layer")
        lw = 2.5 if k == "A" else 1.2
        alpha = 1.0 if k == "A" else 0.7
        ax.plot(k_df["layer"].values, k_df["beta"].values,
                color=COLORS[k], linewidth=lw, alpha=alpha,
                label=f"{k}: {MODEL_NAMES[k]}")

        if "p_fdr" in k_df.columns:
            sig = k_df["p_fdr"].values < 0.05
            if sig.any():
                ax.scatter(k_df["layer"].values[sig], k_df["beta"].values[sig],
                          color=COLORS[k], s=20 if k != "A" else 40,
                          zorder=5, edgecolors="black", linewidth=0.3)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Standardized beta", fontsize=12)
    ax.set_title(f"All Model Betas — 5-Predictor Regression (C1-C4) [{metric_label}]", fontsize=13)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.set_xlim(-0.5, df["layer"].max() + 0.5)

    fig.tight_layout()
    fname = f"all_model_betas_{metric_label.lower()}.png"
    save_fig(fig, fig_dir / fname)
    return fig_to_base64(fig)


# ── Plot: Model RDM visualizations ──────────────────────────────────────────

def _reorder_by_condition(rdm, n_items, n_conds):
    """Reorder RDM from item-grouped to condition-grouped order.

    Input order:  item0-C1, item0-C2, ..., item1-C1, item1-C2, ...
    Output order: all-C1 (56 items), all-C2 (56 items), ...
    """
    perm = []
    for c in range(n_conds):
        for i in range(n_items):
            perm.append(i * n_conds + c)
    perm = np.array(perm)
    return rdm[np.ix_(perm, perm)]


COND_LABELS_4 = ["C1: mental_state", "C2: dis_mental", "C3: scr_mental", "C4: action"]


def plot_model_rdms(fig_dir):
    from utils.rsa import _cross_item_condition_rdm, _pair_conditions

    n_conds = N_CONDS_REDUCED
    model_rdms = {}
    model_rdms["A"] = _cross_item_condition_rdm(N_ITEMS, n_conds, [0])
    model_rdms["B"] = _cross_item_condition_rdm(N_ITEMS, n_conds, [0, 1])
    model_rdms["C"] = _cross_item_condition_rdm(N_ITEMS, n_conds, [0, 1, 2])
    model_rdms["D"] = _cross_item_condition_rdm(N_ITEMS, n_conds, [0, 1, 3])
    model_rdms["E"] = _cross_item_condition_rdm(N_ITEMS, n_conds, [0, 3])

    rdm_cmap = LinearSegmentedColormap.from_list(
        "rdm_br", ["#2166AC", "#F7F7F7", "#D6342D"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, k in enumerate(ALL_MODEL_KEYS):
        ax = axes[idx // 3][idx % 3]
        rdm = _reorder_by_condition(model_rdms[k], N_ITEMS, n_conds)
        display_rdm = rdm.copy()
        np.fill_diagonal(display_rdm, np.nan)
        cmap_copy = rdm_cmap.copy()
        cmap_copy.set_bad("black")
        im = ax.imshow(display_rdm, cmap=cmap_copy, vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"Model {k}: {MODEL_NAMES[k]}", fontsize=10, fontweight="bold")
        for ci in range(1, n_conds):
            boundary = ci * N_ITEMS - 0.5
            ax.axhline(boundary, color="black", linewidth=1, linestyle="-", alpha=0.5)
            ax.axvline(boundary, color="black", linewidth=1, linestyle="-", alpha=0.5)
        tick_positions = [(ci * N_ITEMS + (ci + 1) * N_ITEMS) / 2 - 0.5
                         for ci in range(n_conds)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"C{ci+1}" for ci in range(n_conds)], fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(COND_LABELS_4, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     ticks=[0, 0.5, 1], label="Dissimilarity")

    axes[1][2].set_visible(False)

    fig.suptitle("Model RDMs — Organized by Condition (224×224, 0=similar, 1=dissimilar)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, fig_dir / "model_rdms.png")
    return fig_to_base64(fig)


# ── Plot: Neural RDMs at peak layers ─────────────────────────────────────────

def _filter_acts_to_4cond(acts_full):
    """Filter 336-row activations to 224 rows (C1-C4 only)."""
    keep = []
    for item in range(N_ITEMS):
        base = item * N_CONDITIONS
        for ci in [0, 1, 2, 3]:
            keep.append(base + ci)
    return acts_full[np.array(keep)]


def plot_neural_rdms(df, fig_dir):
    """Plot neural RDMs at the most significant layer for each model."""
    from utils.rsa import compute_rdm

    acts_full = _load_acts_for_position(_active_position)
    if acts_full is None:
        print(f"  Skipping neural RDMs — activations not found for position={_active_position}")
        return None

    acts = _filter_acts_to_4cond(acts_full)
    n_conds = N_CONDS_REDUCED

    rdm_cmap = LinearSegmentedColormap.from_list(
        "rdm_br", ["#2166AC", "#F7F7F7", "#D6342D"])

    # Find peak layer (lowest p_fdr, or lowest p if no FDR sig) per model
    peak_layers = {}
    for k in ALL_MODEL_KEYS:
        k_df = df[df["model"] == k].copy()
        if "p_fdr" in k_df.columns and (k_df["p_fdr"] < 0.05).any():
            best = k_df.loc[k_df["p_fdr"].idxmin()]
        else:
            best = k_df.loc[k_df["p"].idxmin()]
        peak_layers[k] = int(best["layer"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, k in enumerate(ALL_MODEL_KEYS):
        ax = axes[idx // 3][idx % 3]
        layer = peak_layers[k]
        layer_acts = acts[:, layer, :]
        neural_rdm = compute_rdm(layer_acts, metric="correlation")
        neural_rdm_reordered = _reorder_by_condition(neural_rdm, N_ITEMS, n_conds)

        # Mask diagonal with NaN so it renders as black
        display_rdm = neural_rdm_reordered.copy()
        np.fill_diagonal(display_rdm, np.nan)
        cmap_copy = rdm_cmap.copy()
        cmap_copy.set_bad("black")
        im = ax.imshow(display_rdm, cmap=cmap_copy, aspect="equal")
        p_col = "p_fdr" if "p_fdr" in df.columns else "p"
        k_row = df[(df["model"] == k) & (df["layer"] == layer)].iloc[0]
        p_val = k_row[p_col]
        sig = sig_marker_fdr(p_val) if p_col == "p_fdr" else sig_marker_uncorr(p_val)
        ax.set_title(f"Model {k} peak: Layer {layer} (p={p_val:.4f}{sig})",
                     fontsize=10, fontweight="bold")
        for ci in range(1, n_conds):
            boundary = ci * N_ITEMS - 0.5
            ax.axhline(boundary, color="black", linewidth=1, linestyle="-", alpha=0.5)
            ax.axvline(boundary, color="black", linewidth=1, linestyle="-", alpha=0.5)
        tick_positions = [(ci * N_ITEMS + (ci + 1) * N_ITEMS) / 2 - 0.5
                         for ci in range(n_conds)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"C{ci+1}" for ci in range(n_conds)], fontsize=8)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(COND_LABELS_4, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dissimilarity")

    axes[1][2].set_visible(False)

    fig.suptitle("Neural RDMs at Peak Significant Layer per Model (224×224, correlation distance)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, fig_dir / "neural_rdms_peak.png")
    return fig_to_base64(fig)


# ── Plot: Large Model A neural RDM with full condition labels ────────────────

def _get_sentence_labels_4cond():
    """Build list of 224 sentence strings in condition-grouped order.

    Returns sentences ordered: all C1 (56), all C2 (56), all C3 (56), all C4 (56).
    """
    cond_keys = ["mental_state", "dis_mental", "scr_mental", "action"]
    labels = []
    for ck in cond_keys:
        for item in STIMULI:
            labels.append(item[ck])
    return labels


def plot_model_a_neural_rdm(df, fig_dir):
    """Large neural RDM at Model A's peak layer with every sentence labeled."""
    from utils.rsa import compute_rdm

    acts_full = _load_acts_for_position(_active_position)
    if acts_full is None:
        print(f"  Skipping Model A detail RDM — activations not found for position={_active_position}")
        return None

    acts = _filter_acts_to_4cond(acts_full)
    n_conds = N_CONDS_REDUCED

    # Find Model A peak layer
    a_df = df[df["model"] == "A"].copy()
    if "p_fdr" in a_df.columns and (a_df["p_fdr"] < 0.05).any():
        best = a_df.loc[a_df["p_fdr"].idxmin()]
    else:
        best = a_df.loc[a_df["p"].idxmin()]
    peak_layer = int(best["layer"])
    peak_beta = best["beta"]
    peak_p = best.get("p_fdr", best["p"])

    # Compute neural RDM
    layer_acts = acts[:, peak_layer, :]
    neural_rdm = compute_rdm(layer_acts, metric="correlation")
    rdm_reordered = _reorder_by_condition(neural_rdm, N_ITEMS, n_conds)

    # Sentence labels in condition-grouped order
    sentence_labels = _get_sentence_labels_4cond()
    assert len(sentence_labels) == N_ITEMS * n_conds

    rdm_cmap = LinearSegmentedColormap.from_list(
        "rdm_br", ["#2166AC", "#F7F7F7", "#D6342D"])

    display_rdm = rdm_reordered.copy()
    np.fill_diagonal(display_rdm, np.nan)
    cmap_copy = rdm_cmap.copy()
    cmap_copy.set_bad("black")

    # Condition colors for tick labels
    cond_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"]  # red, blue, orange, green
    cond_headers = ["C1: mental_state", "C2: dis_mental", "C3: scr_mental", "C4: action"]

    n_total = N_ITEMS * n_conds  # 224
    fig, ax = plt.subplots(figsize=(40, 36))
    im = ax.imshow(display_rdm, cmap=cmap_copy, aspect="equal")

    # Condition boundaries — thick lines
    for ci in range(1, n_conds):
        boundary = ci * N_ITEMS - 0.5
        ax.axhline(boundary, color="black", linewidth=3, linestyle="-")
        ax.axvline(boundary, color="black", linewidth=3, linestyle="-")

    # Every sentence as a tick label
    ax.set_xticks(range(n_total))
    ax.set_xticklabels(sentence_labels, rotation=90, fontsize=5.5, ha="center")
    ax.set_yticks(range(n_total))
    ax.set_yticklabels(sentence_labels, fontsize=5.5)

    # Color tick labels by condition
    for i in range(n_total):
        ci = i // N_ITEMS
        ax.get_xticklabels()[i].set_color(cond_colors[ci])
        ax.get_yticklabels()[i].set_color(cond_colors[ci])

    # Condition header labels between boundary lines (secondary ticks)
    for ci in range(n_conds):
        mid = ci * N_ITEMS + N_ITEMS / 2 - 0.5
        # Top side (above x-axis)
        ax.text(mid, -3, cond_headers[ci],
                fontsize=14, fontweight="bold", color=cond_colors[ci],
                ha="center", va="bottom", clip_on=False)
        # Left side (left of y-axis)
        ax.text(-3, mid, cond_headers[ci],
                fontsize=14, fontweight="bold", color=cond_colors[ci],
                ha="right", va="center", clip_on=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.08,
                        ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_label("Correlation Distance (1 − Pearson r)", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    sig = sig_marker_fdr(peak_p)
    ax.set_title(
        f"Neural RDM at Model A Peak — Layer {peak_layer}  "
        f"(β_A = {peak_beta:.4f}, p_FDR = {peak_p:.4f}{sig})\n"
        f"224 sentences (56 items × 4 conditions), organized by condition\n"
        f"Negative β_A: C1×C1 pairs are MORE dissimilar than other within-condition pairs (individuation)",
        fontsize=16, fontweight="bold", pad=20
    )

    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.96])
    save_fig(fig, fig_dir / "model_a_neural_rdm_detail.png", dpi=120)
    return fig_to_base64(fig, dpi=120)


# ── HTML generation ──────────────────────────────────────────────────────────

def build_report(df, metric_label, fig_dir, vifs, corr_matrix):

    img_model_rdms = plot_model_rdms(fig_dir)
    img_neural_rdms = plot_neural_rdms(df, fig_dir)
    img_model_a_rdm = plot_model_a_neural_rdm(df, fig_dir)
    img_model_a = plot_model_a_with_sig(df, fig_dir, metric_label)
    img_all_betas = plot_all_betas(df, fig_dir, metric_label)

    # Build results table data
    results_rows = []
    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        row = {"layer": int(layer)}
        for k in ALL_MODEL_KEYS:
            k_row = layer_df[layer_df["model"] == k]
            if len(k_row) == 0:
                continue
            k_row = k_row.iloc[0]
            row[f"beta_{k}"] = k_row["beta"]
            row[f"sr_{k}"] = k_row["semi_partial_r"]
            row[f"dr2_{k}"] = k_row["delta_r2"]
            row[f"p_{k}"] = k_row["p"]
            row[f"pfdr_{k}"] = k_row.get("p_fdr", np.nan)
        results_rows.append(row)

    item = STIMULI[0]

    # VIF table
    vif_html = "<table><tr><th>Model</th><th>VIF</th><th>Status</th></tr>"
    for k in ALL_MODEL_KEYS:
        v = vifs.get(k, float("nan"))
        status = "OK" if v < 5 else ("Caution" if v < 10 else "WARNING")
        color = "#2ca02c" if v < 5 else ("#ff7f0e" if v < 10 else "#d62728")
        vif_html += f'<tr><td>Model {k}: {MODEL_NAMES[k]}</td><td>{v:.2f}</td><td style="color:{color};font-weight:bold">{status}</td></tr>'
    vif_html += "</table>"

    # Model correlation matrix
    corr_html = '<table class="corr-table"><tr><th></th>'
    for k in ALL_MODEL_KEYS:
        corr_html += f"<th>{k}</th>"
    corr_html += "</tr>"
    for i, ki in enumerate(ALL_MODEL_KEYS):
        corr_html += f"<tr><th>{ki}</th>"
        for j, kj in enumerate(ALL_MODEL_KEYS):
            val = corr_matrix[i, j]
            intensity = abs(val)
            if i == j:
                bg = "#f0f0f0"
            elif intensity > 0.7:
                bg = "#ffcccc"
            elif intensity > 0.4:
                bg = "#fff3cd"
            else:
                bg = "#ffffff"
            corr_html += f'<td style="background:{bg}">{val:.3f}</td>'
        corr_html += "</tr>"
    corr_html += "</table>"

    # Full results table — significance markers on p-values
    table_html = '<table class="results-table"><tr><th>Layer</th>'
    for k in ALL_MODEL_KEYS:
        table_html += f'<th>&beta;<sub>{k}</sub></th><th>&Delta;R&sup2;<sub>{k}</sub></th><th>p<sub>uncorr</sub></th><th>p<sub>FDR</sub></th>'
    table_html += "</tr>"
    for row in results_rows:
        table_html += f'<tr><td>{row["layer"]}</td>'
        for k in ALL_MODEL_KEYS:
            beta = row.get(f"beta_{k}", np.nan)
            dr2 = row.get(f"dr2_{k}", np.nan)
            p_fdr = row.get(f"pfdr_{k}", np.nan)
            p_raw = row.get(f"p_{k}", np.nan)
            sig_fdr = sig_marker_fdr(p_fdr)
            sig_raw = sig_marker_uncorr(p_raw)
            bold_beta = ' style="font-weight:bold"' if sig_fdr else ""
            table_html += f'<td{bold_beta}>{beta:.4f}</td>'
            table_html += f'<td>{dr2:.6f}</td>'
            p_raw_bold = ' style="font-weight:bold"' if sig_raw else ""
            table_html += f'<td{p_raw_bold}>{p_raw:.4f} {sig_raw}</td>'
            if not np.isnan(p_fdr):
                p_fdr_bold = ' style="font-weight:bold;color:#d62728"' if sig_fdr else ""
                table_html += f'<td{p_fdr_bold}>{p_fdr:.4f} {sig_fdr}</td>'
            else:
                table_html += '<td>&mdash;</td>'
        table_html += "</tr>"
    table_html += "</table>"

    beta_formula = " + ".join(f"&beta;<sub>{k}</sub>" for k in ALL_MODEL_KEYS)
    beta_interp = ("&beta;<sub>A</sub> significant means: C1 pairs are more similar in the model's "
                   "representations than predicted by any additive combination of sharing a "
                   "mental-verb+object frame (B), mental vocabulary (C), a verb+object frame "
                   "regardless of type (D), or a subject+verb+object frame regardless of verb type (E).")

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Exp 5: Reduced 1-4 RSA ({metric_label})</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #d62728; padding-bottom: 10px; }}
h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 40px; }}
h3 {{ color: #555; }}
table {{ border-collapse: collapse; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: center; font-size: 13px; }}
th {{ background: #34495e; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.results-table {{ font-size: 11px; }}
.results-table th {{ padding: 4px 6px; }}
.results-table td {{ padding: 3px 5px; }}
.corr-table td {{ padding: 4px 8px; font-size: 12px; }}
img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
.toc {{ background: #ecf0f1; padding: 15px 25px; border-radius: 5px; margin: 20px 0; }}
.toc a {{ color: #2980b9; text-decoration: none; }}
.toc a:hover {{ text-decoration: underline; }}
.toc ul {{ list-style-type: none; padding-left: 15px; }}
.toc li {{ margin: 5px 0; }}
.model-spec {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #d62728; margin: 15px 0; font-family: monospace; font-size: 13px; white-space: pre-wrap; }}
.note {{ background: #fff3cd; padding: 10px 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
.sig-legend {{ background: #e8f4f8; padding: 10px 15px; border-radius: 5px; margin: 10px 0; font-size: 13px; }}
.sample-table td {{ text-align: left; padding: 4px 12px; }}
</style>
</head>
<body>

<h1>Experiment 5: Reduced 1-4 RSA — {metric_label} Distance — {_active_position.capitalize()} Position</h1>
<p>5-predictor regression (C1-C4 only): neural RDM ~ {beta_formula} + error</p>
<p>Model: <strong>LLaMA-2-13B-Chat</strong> | Metric: <strong>{metric_label}</strong> | Position: <strong>{_active_position}</strong> | 56 items &times; 4 conditions = 224 sentences | Permutations: 10,000 | FDR: Benjamini-Hochberg (q=0.05)</p>

<div class="toc">
<h3>Table of Contents</h3>
<ul>
<li><a href="#design">1. Stimulus Design</a></li>
<li><a href="#sample">2. Sample Stimulus</a></li>
<li><a href="#model-spec">3. Model Specification</a></li>
<li><a href="#vifs">4. VIF Check & Model Correlations</a></li>
<li><a href="#model-a">5. Model A: Beta & Unique Variance</a></li>
<li><a href="#all-betas">6. All Model Betas</a></li>
<li><a href="#results-table">7. Full Results Table</a></li>
<li><a href="#model-rdms">8. Model RDMs</a></li>
<li><a href="#neural-rdms">9. Neural RDMs at Peak Layers</a></li>
<li><a href="#model-a-rdm">10. Model A: Neural RDM Detail</a></li>
</ul>
</div>

<h2 id="design">1. Stimulus Design</h2>
<p>56 items &times; 4 conditions = 224 sentences. Subject fixed to "He" throughout.
Conditions C5 (dis_action) and C6 (scr_action) are excluded from this reduced analysis.</p>

<table>
<tr><th>Code</th><th>Label</th><th>Template</th><th>Example</th></tr>
<tr><td>C1</td><td>mental_state</td><td>He [mental verb] the [object].</td><td>He notices the crack.</td></tr>
<tr><td>C2</td><td>dis_mental</td><td>[Mental verb] the [object].</td><td>Notice the crack.</td></tr>
<tr><td>C3</td><td>scr_mental</td><td>The [object] to [mental verb].</td><td>The crack to notice.</td></tr>
<tr><td>C4</td><td>action</td><td>He [action verb] the [object].</td><td>He fills the crack.</td></tr>
</table>

<h3>Condition Feature Matrix</h3>
<table>
<tr><th>Feature</th><th>C1</th><th>C2</th><th>C3</th><th>C4</th></tr>
<tr><td style="text-align:left">Has subject "He"</td><td>&check;</td><td></td><td></td><td>&check;</td></tr>
<tr><td style="text-align:left">Mental verb present</td><td>&check;</td><td>&check;</td><td>&check;</td><td></td></tr>
<tr><td style="text-align:left">Action verb present</td><td></td><td></td><td></td><td>&check;</td></tr>
<tr><td style="text-align:left">Grammatical word order</td><td>&check;</td><td>&check;</td><td></td><td>&check;</td></tr>
<tr><td style="text-align:left">Scrambled form</td><td></td><td></td><td>&check;</td><td></td></tr>
<tr><td style="text-align:left">Object noun present</td><td>&check;</td><td>&check;</td><td>&check;</td><td>&check;</td></tr>
<tr><td style="text-align:left">Mental verb + object (grammatical)</td><td>&check;</td><td>&check;</td><td></td><td></td></tr>
<tr><td style="text-align:left">Action verb + object (grammatical)</td><td></td><td></td><td></td><td>&check;</td></tr>
</table>

<p>7 verb categories &times; 8 items each: Attention, Memory, Sensation, Belief, Desire, Emotion, Intention.</p>

<h2 id="sample">2. Sample Stimulus</h2>
<p>Item 1 (Attention): mental verb = <em>{item["mverb"]}</em>, action verb = <em>{item["averb"]}</em>, object = <em>{item["obj"]}</em></p>
<table class="sample-table">
<tr><th>Condition</th><th>Sentence</th></tr>
<tr><td>C1 (mental_state)</td><td>{item["mental_state"]}</td></tr>
<tr><td>C2 (dis_mental)</td><td>{item["dis_mental"]}</td></tr>
<tr><td>C3 (scr_mental)</td><td>{item["scr_mental"]}</td></tr>
<tr><td>C4 (action)</td><td>{item["action"]}</td></tr>
</table>

<h2 id="model-spec">3. Model Specification</h2>

<table>
<tr><th>Model</th><th>Name</th><th>Predicts similarity when both in...</th><th>Purpose</th></tr>
{"".join(f'<tr><td><strong>{k}</strong></td><td>{MODEL_NAMES[k]}</td><td>{MODEL_CONDS[k]}</td><td>{MODEL_PURPOSE[k]}</td></tr>' for k in ALL_MODEL_KEYS)}
</table>

<div class="model-spec">neural_RDM = {beta_formula} + error</div>

<p>{beta_interp}</p>

<div class="note">
<strong>Reduced design rationale:</strong> Dropping C5 and C6 reduces the RDM from 336&times;336 to 224&times;224, increasing the proportion of C1&times;C1 pairs from 2.7% to 6.2%. This improves sensitivity to Model A while retaining the key contrasts needed to partial out confounds.
</div>

<h2 id="vifs">4. VIF Check & Model Correlations</h2>
<h3>Variance Inflation Factors</h3>
<p>VIF &lt; 5: OK. VIF 5&ndash;10: proceed with caution. VIF &gt; 10: regression not trustworthy.</p>
{vif_html}

<h3>Model RDM Correlation Matrix</h3>
{corr_html}

<h2 id="model-a">5. Model A: Beta & Unique Variance</h2>
<div class="sig-legend">
<strong>Significance markers:</strong> * p<sub>FDR</sub> &lt; .05 &nbsp; ** p<sub>FDR</sub> &lt; .01 &nbsp; *** p<sub>FDR</sub> &lt; .001 &nbsp; &#9651; p<sub>uncorr</sub> &lt; .05 (not FDR-corrected)
</div>
<img src="data:image/png;base64,{img_model_a}" alt="Model A Beta and Unique Variance">

<h2 id="all-betas">6. All Model Betas</h2>
<p>All 5 model betas from the simultaneous regression at each layer. Filled dots indicate FDR-corrected significance (p<sub>FDR</sub> &lt; .05).</p>
<img src="data:image/png;base64,{img_all_betas}" alt="All Model Betas">

<h2 id="results-table">7. Full Results Table</h2>
<div class="sig-legend">
<strong>Significance markers:</strong> * p &lt; .05, ** p &lt; .01, *** p &lt; .001. Markers shown on p<sub>uncorr</sub> and p<sub>FDR</sub> columns. FDR-significant p-values in red.
</div>
{table_html}

<h2 id="model-rdms">8. Model RDMs</h2>
<p>5 binary model RDMs (224&times;224). Each predicts similarity (0) for sentence pairs where both belong to the specified condition set.</p>
<img src="data:image/png;base64,{img_model_rdms}" alt="Model RDMs">

<h2 id="neural-rdms">9. Neural RDMs at Peak Layers</h2>
<p>Neural RDMs (correlation distance) at the most significant layer for each model predictor. Rows and columns are organized by condition.</p>
{f'<img src="data:image/png;base64,{img_neural_rdms}" alt="Neural RDMs at Peak Layers">' if img_neural_rdms else '<p><em>Activations not available — run on a node with access to activation files to generate this plot.</em></p>'}

<h2 id="model-a-rdm">10. Model A: Neural RDM Detail</h2>
<p>Large neural RDM at Model A&rsquo;s peak significant layer. Numbers in each block show the <strong>mean correlation distance</strong> for that condition pair. The C1&times;C1 block (upper-left) has the <em>largest</em> within-condition distance, confirming the negative &beta;<sub>A</sub>: full mental state attributions <strong>individuate</strong> rather than cluster.</p>
<div class="note">
<strong>Interpretation:</strong> Negative &beta;<sub>A</sub> means the model differentiates among specific mental state attributions (&ldquo;He notices the crack&rdquo; vs. &ldquo;He believes the story&rdquo;) more sharply than it differentiates among control conditions. Full attributions occupy a <em>larger</em> region of representational space &mdash; consistent with finer-grained compositional semantic encoding when subject, mental verb, and object are bound together.
</div>
{f'<img src="data:image/png;base64,{img_model_a_rdm}" alt="Model A Neural RDM Detail">' if img_model_a_rdm else '<p><em>Activations not available.</em></p>'}

<hr>
<p style="color:#888; font-size:12px">Generated by <code>code/rsa/5_predictors/2_reduced_1_4_report.py</code></p>
</body>
</html>"""

    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Reduced 1-4 RSA Report (HTML)")
    add_model_argument(parser)
    parser.add_argument(
        "--position", type=str, nargs="+",
        default=["period"],
        choices=["verb", "object", "period"],
        help="Token position(s) to generate reports for (default: period)"
    )
    return parser.parse_args()


def main():
    global _active_position
    args = parse_args()
    set_model(args.model)

    data_path = data_dir("rsa") / "5_predictors"
    rsa_dir = ensure_dir(results_dir("rsa") / "reports")

    for position in args.position:
        _active_position = position
        suffix = "" if position == "period" else f"_{position}"

        fig_dir = ensure_dir(data_path / "figures" / (position if position != "period" else ""))

        # Load VIFs (position-independent, but saved per-position)
        vif_path = data_path / f"vif_check{suffix}.json"
        if not vif_path.exists():
            vif_path = data_path / "vif_check.json"
        vifs = {}
        if vif_path.exists():
            with open(vif_path) as f:
                vifs = json.load(f)

        # Load model correlations
        corr_path = data_path / f"model_correlations{suffix}.npz"
        if not corr_path.exists():
            corr_path = data_path / "model_correlations.npz"
        if corr_path.exists():
            corr_data = np.load(corr_path)
            corr_matrix = corr_data["corr_matrix"]
        else:
            corr_matrix = np.eye(len(ALL_MODEL_KEYS))

        csv_path = data_path / f"5_predictors_rsa_corr{suffix}.csv"
        if not csv_path.exists():
            print(f"WARNING: Results CSV not found: {csv_path} — skipping {position}")
            continue

        print(f"Generating reduced 1-4 RSA report ({position} position)...")
        df = pd.read_csv(csv_path)
        html = build_report(df, "Correlation", fig_dir, vifs, corr_matrix)

        report_path = rsa_dir / f"5_predictors_report_corr{suffix}.html"
        with open(report_path, "w") as f:
            f.write(html)
        print(f"  Saved: {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
