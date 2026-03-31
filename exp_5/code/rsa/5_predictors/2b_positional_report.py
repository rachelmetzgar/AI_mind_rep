#!/usr/bin/env python3
"""
Experiment 5: Positional 5-Predictor RSA Summary Report (HTML)

Cross-position (verb/object/period) and cross-model comparison for the
5-predictor reduced RSA (A-E, C1-C4 only). Shows per-predictor significance,
emergence patterns, and full results tables.

Output:
    results/{model}/rsa/reports/
        5_predictors_positional_report.html       (per-model)
    results/comparisons/rsa/5_predictors/
        5_predictors_cross_model_positional.html   (cross-model, --all-models)

Usage:
    python code/rsa/5_predictors/2b_positional_report.py --model llama2_13b_chat
    python code/rsa/5_predictors/2b_positional_report.py --all-models

Env: llama2_env (no GPU needed)
Rachel C. Metzgar · Mar 2026
"""

import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    set_model, get_model, add_model_argument, data_dir, results_dir,
    ensure_dir, VALID_MODELS, RESULTS_DIR,
)

# ── Constants ────────────────────────────────────────────────────────────────

POSITIONS = ["verb", "object", "period"]
POS_COLORS = {"verb": "#2ca02c", "object": "#ff7f0e", "period": "#1f77b4"}
ALL_MODEL_KEYS = ["A", "B", "C", "D", "E"]

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

COLORS = {
    "A": "#d62728",
    "B": "#1f77b4",
    "C": "#ff7f0e",
    "D": "#2ca02c",
    "E": "#9467bd",
}

# ── Model metadata (matching Exp 4 conventions) ────────────────────────────

MODEL_DISPLAY = {
    "llama2_13b_chat": "LLaMA-2-13B Chat",
    "llama2_13b_base": "LLaMA-2-13B Base",
    "llama3_8b_instruct": "LLaMA-3-8B Instruct",
    "llama3_8b_base": "LLaMA-3-8B Base",
    "gemma2_2b_it": "Gemma-2-2B-IT",
    "gemma2_2b": "Gemma-2-2B Base",
    "gemma2_9b_it": "Gemma-2-9B-IT",
    "gemma2_9b": "Gemma-2-9B Base",
    "qwen25_7b_instruct": "Qwen-2.5-7B-Instruct",
    "qwen25_7b": "Qwen-2.5-7B Base",
    "qwen3_8b": "Qwen3-8B",
}

CROSS_MODEL_COLORS = {
    "llama2_13b_chat": "#5b9bd5",
    "llama2_13b_base": "#8faabc",
    "llama3_8b_instruct": "#2a5fa5",
    "llama3_8b_base": "#5a7080",
    "gemma2_2b_it": "#e74c3c",
    "gemma2_2b": "#b07a7a",
    "gemma2_9b_it": "#c0392b",
    "gemma2_9b": "#8b5e5e",
    "qwen25_7b_instruct": "#b8860b",
    "qwen25_7b": "#8b7355",
    "qwen3_8b": "#d4a017",
}

# Canonical display order: chat/instruct before base within each family
MODEL_ORDER = [
    "llama2_13b_chat", "llama2_13b_base",
    "llama3_8b_instruct", "llama3_8b_base",
    "gemma2_2b_it", "gemma2_2b",
    "gemma2_9b_it", "gemma2_9b",
    "qwen25_7b_instruct", "qwen25_7b",
    "qwen3_8b",
]

MODEL_FAMILIES = [
    ("LLaMA", ["llama2_13b_chat", "llama2_13b_base",
               "llama3_8b_instruct", "llama3_8b_base"]),
    ("Gemma", ["gemma2_2b_it", "gemma2_2b",
               "gemma2_9b_it", "gemma2_9b"]),
    ("Qwen",  ["qwen25_7b_instruct", "qwen25_7b", "qwen3_8b"]),
]

CROSS_MODEL_LS = {m: "-" for m in MODEL_ORDER}  # all solid for grid layout


def sort_models(models):
    """Sort model keys by canonical MODEL_ORDER."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order.get(m, 999))


# ── Helpers ──────────────────────────────────────────────────────────────────

def fig_to_b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_fig(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def sig(p):
    try:
        if np.isnan(p):
            return ""
    except (TypeError, ValueError):
        return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


def load_position_data(data_path, position):
    if position == "period":
        csv_path = data_path / "5_predictors_rsa_corr.csv"
    else:
        csv_path = data_path / f"5_predictors_rsa_corr_{position}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["position"] = position
    return df


def load_model_data(model_key):
    set_model(model_key)
    data_path = data_dir("rsa") / "5_predictors"
    dfs = {}
    for pos in POSITIONS:
        df = load_position_data(data_path, pos)
        if df is not None:
            dfs[pos] = df
    return dfs


# ── Per-predictor summary stats ─────────────────────────────────────────────

def predictor_summary(dfs):
    """Build summary dict: {predictor: {position: {peak_layer, peak_beta, ...}}}."""
    summary = {}
    for k in ALL_MODEL_KEYS:
        summary[k] = {}
        for pos in POSITIONS:
            if pos not in dfs:
                continue
            df = dfs[pos]
            k_df = df[df["model"] == k].copy()
            if k_df.empty:
                continue
            has_fdr = "p_fdr" in k_df.columns
            # Peak by absolute beta
            peak_idx = k_df["beta"].abs().idxmax()
            peak = k_df.loc[peak_idx]
            n_sig_fdr = int((k_df["p_fdr"] < 0.05).sum()) if has_fdr else 0
            n_sig_uncorr = int((k_df["p"] < 0.05).sum())
            sig_layers_fdr = sorted(k_df.loc[k_df["p_fdr"] < 0.05, "layer"].tolist()) if has_fdr else []
            # First significant layer
            first_sig = int(min(sig_layers_fdr)) if sig_layers_fdr else None
            summary[k][pos] = {
                "peak_layer": int(peak["layer"]),
                "peak_beta": float(peak["beta"]),
                "peak_sr": float(peak["semi_partial_r"]),
                "peak_dr2": float(peak["delta_r2"]),
                "peak_p": float(peak["p"]),
                "peak_p_fdr": float(peak["p_fdr"]) if has_fdr else np.nan,
                "n_sig_fdr": n_sig_fdr,
                "n_sig_uncorr": n_sig_uncorr,
                "sig_layers_fdr": sig_layers_fdr,
                "first_sig": first_sig,
            }
    return summary


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_all_predictors_by_position(dfs, fig_dir, model_label):
    """5 subplots (A-E), each showing beta across layers at verb/object/period."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, k in enumerate(ALL_MODEL_KEYS):
        ax = axes[idx // 3][idx % 3]
        for pos in POSITIONS:
            if pos not in dfs:
                continue
            df = dfs[pos]
            k_df = df[df["model"] == k].sort_values("layer")
            layers = k_df["layer"].values
            betas = k_df["beta"].values
            ax.plot(layers, betas, color=POS_COLORS[pos], linewidth=2, label=pos)
            if "p_fdr" in k_df.columns:
                sig_mask = k_df["p_fdr"].values < 0.05
                if sig_mask.any():
                    ax.scatter(layers[sig_mask], betas[sig_mask],
                              color=POS_COLORS[pos], s=35, zorder=5,
                              edgecolors="black", linewidth=0.5)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"Model {k}: {MODEL_NAMES[k]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Beta", fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        if idx == 0:
            ax.legend(fontsize=9)

    axes[1][2].set_visible(False)
    fig.suptitle(f"All Predictors — Beta by Token Position [{model_label}]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, fig_dir / "all_predictors_by_position.png")
    return fig_to_b64(fig)


def plot_model_a_detail(dfs, fig_dir, model_label):
    """Model A: beta + ΔR² + semi-partial r, one line per position."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 1, 1]})
    metrics = [("beta", "Standardized β"), ("delta_r2", "Unique Variance (ΔR²)"),
               ("semi_partial_r", "Semi-partial r")]

    for mi, (col, ylabel) in enumerate(metrics):
        ax = axes[mi]
        for pos in POSITIONS:
            if pos not in dfs:
                continue
            df = dfs[pos]
            a_df = df[df["model"] == "A"].sort_values("layer")
            layers = a_df["layer"].values
            vals = a_df[col].values
            ax.plot(layers, vals, color=POS_COLORS[pos], linewidth=2.5, label=pos)
            if "p_fdr" in a_df.columns:
                sig_mask = a_df["p_fdr"].values < 0.05
                if sig_mask.any():
                    ax.scatter(layers[sig_mask], vals[sig_mask],
                              color=POS_COLORS[pos], s=40, zorder=5,
                              edgecolors="black", linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_ylabel(ylabel, fontsize=11)
        if mi == 0:
            ax.set_title(f"Model A (Full Attribution) — {model_label}", fontsize=13)
            ax.legend(fontsize=10)

    axes[-1].set_xlabel("Layer", fontsize=11)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(5))
    fig.tight_layout()
    save_fig(fig, fig_dir / "model_a_detail_positional.png")
    return fig_to_b64(fig)


def plot_emergence_heatmap(summary, fig_dir, model_label, n_layers):
    """Heatmap: rows=predictors, cols=layers, cells colored by position of first significance."""
    fig, ax = plt.subplots(figsize=(14, 4))

    # Build matrix: for each predictor × layer, mark which positions are significant
    data = np.zeros((len(ALL_MODEL_KEYS), n_layers, 3))  # 3 positions
    for ki, k in enumerate(ALL_MODEL_KEYS):
        for pi, pos in enumerate(POSITIONS):
            if pos in summary[k]:
                for layer in summary[k][pos]["sig_layers_fdr"]:
                    data[ki, layer, pi] = 1

    # Create composite image: RGB channels = verb(G), object(O→R channel mix), period(B)
    img = np.zeros((len(ALL_MODEL_KEYS), n_layers, 3))
    # verb = green, object = orange (R+some G), period = blue
    for ki in range(len(ALL_MODEL_KEYS)):
        for li in range(n_layers):
            r, g, b = 0.92, 0.92, 0.92  # light gray background
            active = []
            if data[ki, li, 0]:  # verb
                active.append("verb")
            if data[ki, li, 1]:  # object
                active.append("object")
            if data[ki, li, 2]:  # period
                active.append("period")

            if active:
                # Blend colors
                r_sum, g_sum, b_sum = 0, 0, 0
                for pos in active:
                    cr, cg, cb = tuple(int(POS_COLORS[pos].lstrip("#")[i:i+2], 16)/255 for i in (0,2,4))
                    r_sum += cr; g_sum += cg; b_sum += cb
                n = len(active)
                r, g, b = r_sum/n, g_sum/n, b_sum/n

            img[ki, li] = [r, g, b]

    ax.imshow(img, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(ALL_MODEL_KEYS)))
    ax.set_yticklabels([f"{k}: {MODEL_NAMES[k]}" for k in ALL_MODEL_KEYS], fontsize=10)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_title(f"FDR-Significant Layers by Position [{model_label}]", fontsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=POS_COLORS[p], label=p.capitalize()) for p in POSITIONS]
    legend_elements.append(Patch(facecolor="#e0e0e0", label="Not significant"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, ncol=4)

    fig.tight_layout()
    save_fig(fig, fig_dir / "emergence_heatmap.png")
    return fig_to_b64(fig)


def _make_model_grid(available_models):
    """Compute family-grouped grid positions for multi-panel figures.

    Each model family starts a new row. Returns:
        positions: list of (row, col) per model
        ordered: model keys in display order
        nrows, ncols: grid dimensions
    """
    ncols = 4
    available_set = set(available_models)
    positions, ordered = [], []
    row = 0
    for _family_name, family_members in MODEL_FAMILIES:
        family_avail = [m for m in family_members if m in available_set]
        if not family_avail:
            continue
        for col_idx, model in enumerate(family_avail):
            positions.append((row, col_idx))
            ordered.append(model)
        row += 1
    nrows = max(row, 1)
    return positions, ordered, nrows, ncols


def plot_cross_model_per_predictor(all_data, fig_dir):
    """Family-grouped grid: one subplot per model, showing all 5 predictors.

    Each model gets its own panel (family-grouped rows). Within each panel,
    lines show the 5 predictors (A-E) at the period position. FDR-significant
    layers are marked with filled dots.
    """
    available = sort_models(all_data.keys())
    positions, ordered, nrows, ncols = _make_model_grid(available)

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                                   squeeze=False)
    # Hide all axes first
    for r in range(nrows):
        for c in range(ncols):
            axes_grid[r][c].set_visible(False)

    for (row, col), model_key in zip(positions, ordered):
        ax = axes_grid[row][col]
        ax.set_visible(True)
        dfs = all_data[model_key]
        pos = "period"  # primary position for overview
        if pos in dfs:
            df = dfs[pos]
            for k in ALL_MODEL_KEYS:
                k_df = df[df["model"] == k].sort_values("layer")
                layers = k_df["layer"].values
                betas = k_df["beta"].values
                ax.plot(layers, betas, color=COLORS[k], linewidth=1.8, label=k if row == 0 and col == 0 else None)
                if "p_fdr" in k_df.columns:
                    sig_mask = k_df["p_fdr"].values < 0.05
                    if sig_mask.any():
                        ax.scatter(layers[sig_mask], betas[sig_mask],
                                  color=COLORS[k], s=20, zorder=5,
                                  edgecolors="black", linewidth=0.3)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=11,
                     fontweight="bold",
                     color=CROSS_MODEL_COLORS.get(model_key, "#333"))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        if col == 0:
            ax.set_ylabel("Beta", fontsize=10)
        if row == nrows - 1 or not axes_grid[row+1][col].get_visible() if row < nrows - 1 else True:
            ax.set_xlabel("Layer", fontsize=10)

    # Legend from first visible panel
    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10,
                   bbox_to_anchor=(0.5, -0.01),
                   title="Predictors", title_fontsize=11)

    fig.suptitle("All Predictors (Period Position) — Family-Grouped",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, fig_dir / "cross_model_all_predictors_grid.png")
    return fig_to_b64(fig)


def plot_cross_model_model_a(all_data, fig_dir):
    """Family-grouped grid: one subplot per model, showing Model A at all 3 positions.

    Each model gets its own panel. Within each panel, three lines show
    Model A beta at verb (green), object (orange), and period (blue).
    """
    available = sort_models(all_data.keys())
    positions, ordered, nrows, ncols = _make_model_grid(available)

    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows),
                                   squeeze=False)
    for r in range(nrows):
        for c in range(ncols):
            axes_grid[r][c].set_visible(False)

    for (row, col), model_key in zip(positions, ordered):
        ax = axes_grid[row][col]
        ax.set_visible(True)
        dfs = all_data[model_key]
        for pos in POSITIONS:
            if pos not in dfs:
                continue
            df = dfs[pos]
            a_df = df[df["model"] == "A"].sort_values("layer")
            layers = a_df["layer"].values
            betas = a_df["beta"].values
            ax.plot(layers, betas, color=POS_COLORS[pos], linewidth=2,
                    label=pos.capitalize() if row == 0 and col == 0 else None)
            if "p_fdr" in a_df.columns:
                sig_mask = a_df["p_fdr"].values < 0.05
                if sig_mask.any():
                    ax.scatter(layers[sig_mask], betas[sig_mask],
                              color=POS_COLORS[pos], s=30, zorder=5,
                              edgecolors="black", linewidth=0.4)

        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(MODEL_DISPLAY.get(model_key, model_key), fontsize=11,
                     fontweight="bold",
                     color=CROSS_MODEL_COLORS.get(model_key, "#333"))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        if col == 0:
            ax.set_ylabel("Model A β", fontsize=10)
        if row == nrows - 1 or not axes_grid[row+1][col].get_visible() if row < nrows - 1 else True:
            ax.set_xlabel("Layer", fontsize=10)

    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, -0.01),
                   title="Token Position", title_fontsize=11)

    fig.suptitle("Model A (Full Attribution) — Cross-Model by Position",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, fig_dir / "cross_model_model_a.png")
    return fig_to_b64(fig)


# ── HTML: summary tables ────────────────────────────────────────────────────

def build_full_predictor_table(summary):
    """All predictors × all positions: peak beta, peak layer, sig layers, first sig."""
    html = '<table class="results-table">'
    html += '<tr><th>Predictor</th><th>Description</th><th>Conditions</th>'
    for pos in POSITIONS:
        html += f'<th colspan="5">{pos.capitalize()}</th>'
    html += '</tr><tr><th></th><th></th><th></th>'
    for _ in POSITIONS:
        html += '<th>Peak β</th><th>Peak L</th><th>ΔR²</th><th># Sig</th><th>1st Sig L</th>'
    html += '</tr>'

    for k in ALL_MODEL_KEYS:
        html += f'<tr><td><strong>{k}</strong></td>'
        html += f'<td>{MODEL_NAMES[k]}</td><td>{MODEL_CONDS[k]}</td>'
        for pos in POSITIONS:
            if pos in summary[k]:
                s = summary[k][pos]
                p_fdr = s["peak_p_fdr"]
                marker = sig(p_fdr)
                bold = ' style="font-weight:bold;color:#d62728"' if marker else ""
                html += f'<td{bold}>{s["peak_beta"]:.4f}{marker}</td>'
                html += f'<td>{s["peak_layer"]}</td>'
                html += f'<td>{s["peak_dr2"]:.6f}</td>'
                html += f'<td>{s["n_sig_fdr"]}</td>'
                first = str(s["first_sig"]) if s["first_sig"] is not None else "—"
                html += f'<td>{first}</td>'
            else:
                html += '<td colspan="5">—</td>'
        html += '</tr>'
    html += '</table>'
    return html


def build_position_detail_table(dfs, position):
    """Full layer-by-layer results for one position, all predictors."""
    if position not in dfs:
        return "<p>No data.</p>"
    df = dfs[position]
    html = f'<table class="results-table"><tr><th>Layer</th>'
    for k in ALL_MODEL_KEYS:
        html += f'<th>β<sub>{k}</sub></th><th>ΔR²<sub>{k}</sub></th><th>p<sub>FDR</sub></th>'
    html += '</tr>'

    for layer in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == layer]
        html += f'<tr><td>{int(layer)}</td>'
        for k in ALL_MODEL_KEYS:
            k_row = layer_df[layer_df["model"] == k]
            if k_row.empty:
                html += '<td>—</td><td>—</td><td>—</td>'
                continue
            k_row = k_row.iloc[0]
            beta = k_row["beta"]
            dr2 = k_row["delta_r2"]
            p_fdr = k_row.get("p_fdr", np.nan)
            marker = sig(p_fdr)
            bold = ' style="font-weight:bold;color:#d62728"' if marker else ""
            html += f'<td{bold}>{beta:.4f}</td>'
            html += f'<td>{dr2:.6f}</td>'
            html += f'<td{bold}>{p_fdr:.4f} {marker}</td>'
        html += '</tr>'
    html += '</table>'
    return html


def build_emergence_narrative(summary):
    """Text interpretation of when each predictor first becomes significant."""
    lines = []
    for k in ALL_MODEL_KEYS:
        firsts = {}
        for pos in POSITIONS:
            if pos in summary[k] and summary[k][pos]["first_sig"] is not None:
                firsts[pos] = summary[k][pos]["first_sig"]
        if not firsts:
            lines.append(f"<li><strong>Model {k} ({MODEL_NAMES[k]}):</strong> Not significant at any position (FDR).</li>")
            continue
        parts = [f"{pos} L{layer}" for pos, layer in sorted(firsts.items(), key=lambda x: POSITIONS.index(x[0]))]
        n_sig = {pos: summary[k][pos]["n_sig_fdr"] for pos in firsts}
        sig_counts = ", ".join(f"{pos}: {n_sig[pos]}L" for pos in sorted(n_sig, key=lambda x: POSITIONS.index(x)))
        lines.append(
            f"<li><strong>Model {k} ({MODEL_NAMES[k]}):</strong> "
            f"First significant at {', '.join(parts)}. "
            f"Total sig layers: {sig_counts}.</li>"
        )
    return "<ul>" + "\n".join(lines) + "</ul>"


# ── HTML builders ────────────────────────────────────────────────────────────

CSS = """
body { font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; }
h1 { color: #2c3e50; border-bottom: 3px solid #d62728; padding-bottom: 10px; }
h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 40px; }
h3 { color: #555; }
table { border-collapse: collapse; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: center; font-size: 13px; }
th { background: #34495e; color: white; }
tr:nth-child(even) { background: #f2f2f2; }
.results-table { font-size: 11px; }
.results-table th { padding: 4px 6px; }
.results-table td { padding: 3px 5px; }
img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
.toc { background: #ecf0f1; padding: 15px 25px; border-radius: 5px; margin: 20px 0; }
.toc a { color: #2980b9; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.toc ul { list-style-type: none; padding-left: 15px; }
.toc li { margin: 5px 0; }
.note { background: #fff3cd; padding: 10px 15px; border-left: 4px solid #ffc107; margin: 15px 0; }
.key-finding { background: #d4edda; padding: 10px 15px; border-left: 4px solid #28a745; margin: 15px 0; }
.sig-legend { background: #e8f4f8; padding: 10px 15px; border-radius: 5px; margin: 10px 0; font-size: 13px; }
"""


def build_per_model_html(dfs, model_key, fig_dir):
    model_label = MODEL_DISPLAY.get(model_key, model_key)
    n_layers = max(df["layer"].max() for df in dfs.values()) + 1
    summary = predictor_summary(dfs)

    img_a_detail = plot_model_a_detail(dfs, fig_dir, model_label)
    img_all = plot_all_predictors_by_position(dfs, fig_dir, model_label)
    img_heatmap = plot_emergence_heatmap(summary, fig_dir, model_label, int(n_layers))

    pred_table = build_full_predictor_table(summary)
    emergence = build_emergence_narrative(summary)

    position_tables = ""
    for pos in POSITIONS:
        if pos in dfs:
            position_tables += f'<h3 id="detail-{pos}">{pos.capitalize()} Position</h3>\n'
            position_tables += build_position_detail_table(dfs, pos)

    positions_found = [p for p in POSITIONS if p in dfs]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Exp 5: 5-Predictor Positional RSA — {model_label}</title>
<style>{CSS}</style>
</head>
<body>

<h1>Exp 5: 5-Predictor Positional RSA Summary — {model_label}</h1>
<p>5-predictor regression (C1-C4 only) at verb, object, and period token positions.</p>
<p>Positions: <strong>{', '.join(positions_found)}</strong> | {int(n_layers)} layers | 10K permutations | FDR q=0.05</p>

<div class="toc">
<h3>Contents</h3>
<ul>
<li><a href="#summary">1. All-Predictor Summary Table</a></li>
<li><a href="#emergence">2. Emergence Narrative</a></li>
<li><a href="#heatmap">3. Significance Heatmap</a></li>
<li><a href="#model-a">4. Model A Detail (β, ΔR², semi-partial r)</a></li>
<li><a href="#all-pred">5. All Predictors by Position</a></li>
<li><a href="#detail-tables">6. Full Layer-by-Layer Tables</a></li>
</ul>
</div>

<div class="sig-legend">
<strong>Markers:</strong> * p<sub>FDR</sub> &lt; .05, ** p<sub>FDR</sub> &lt; .01, *** p<sub>FDR</sub> &lt; .001.
Filled dots on plots = FDR-significant. "# Sig" = number of FDR-significant layers. "1st Sig L" = first layer reaching FDR significance.
</div>

<div class="note">
<strong>Key question:</strong> Does attribution binding (Model A) emerge left-to-right as the sentence is processed?
At the <strong>verb</strong>, the object has not yet been seen — only subject+verb information is available.
At the <strong>object</strong>, verb-object binding can begin.
At the <strong>period</strong>, the full sentence has been processed.
We predict: A absent/weak at verb, emerging at object, strongest at period.
</div>

<h2 id="summary">1. All-Predictor Summary</h2>
<p>Peak statistics and significance counts for all 5 predictors at each position.</p>
{pred_table}

<h2 id="emergence">2. Emergence Narrative</h2>
{emergence}

<h2 id="heatmap">3. Significance Heatmap</h2>
<p>Each cell shows whether a predictor is FDR-significant at that layer, colored by position.
Blended colors indicate significance at multiple positions.</p>
<img src="data:image/png;base64,{img_heatmap}" alt="Emergence Heatmap">

<h2 id="model-a">4. Model A Detail</h2>
<p>Three metrics for Model A (Full Attribution): standardized β, unique variance ΔR², and semi-partial r.
Filled dots = FDR-significant.</p>
<img src="data:image/png;base64,{img_a_detail}" alt="Model A Detail">

<h2 id="all-pred">5. All Predictors by Position</h2>
<p>Each panel shows one predictor (A-E) with beta trajectories at each token position.</p>
<img src="data:image/png;base64,{img_all}" alt="All Predictors by Position">

<h2 id="detail-tables">6. Full Layer-by-Layer Results</h2>
<p>Complete results for each position. Bold red = FDR-significant.</p>
{position_tables}

<hr>
<p style="color:#888; font-size:12px">Generated by <code>code/rsa/5_predictors/2b_positional_report.py</code></p>
</body>
</html>"""
    return html


def _build_peak_summary_table(all_data):
    """Cross-model peak summary: one row per model, key stats for Model A at each position."""
    html = '<table class="results-table">\n'
    html += '<tr><th>Model</th>'
    for pos in POSITIONS:
        html += f'<th colspan="4">{pos.capitalize()}</th>'
    html += '</tr>\n<tr><th></th>'
    for _ in POSITIONS:
        html += '<th>Peak β</th><th>Peak L</th><th>ΔR²</th><th># Sig</th>'
    html += '</tr>\n'

    for model_key in sort_models(all_data.keys()):
        dfs = all_data[model_key]
        summary = predictor_summary(dfs)
        label = MODEL_DISPLAY.get(model_key, model_key)
        color = CROSS_MODEL_COLORS.get(model_key, "#333")
        html += f'<tr><td style="border-left: 4px solid {color}; font-weight: 600; text-align: left;">{label}</td>'
        for pos in POSITIONS:
            if "A" in summary and pos in summary["A"]:
                s = summary["A"][pos]
                marker = sig(s["peak_p_fdr"])
                bold = ' style="font-weight:bold;color:#d62728"' if marker else ""
                html += f'<td{bold}>{s["peak_beta"]:.4f}{marker}</td>'
                html += f'<td>{s["peak_layer"]}</td>'
                html += f'<td>{s["peak_dr2"]:.6f}</td>'
                html += f'<td>{s["n_sig_fdr"]}</td>'
            else:
                html += '<td colspan="4">—</td>'
        html += '</tr>\n'
    html += '</table>\n'
    return html


def build_cross_model_html(all_data, fig_dir):
    from datetime import datetime
    img_a = plot_cross_model_model_a(all_data, fig_dir)
    img_grid = plot_cross_model_per_predictor(all_data, fig_dir)

    n_models = len(all_data)
    models_sorted = sort_models(all_data.keys())

    # Peak summary table (Model A across all models)
    peak_table = _build_peak_summary_table(all_data)

    # Per-model detail sections
    model_sections = ""
    for model_key in models_sorted:
        dfs = all_data[model_key]
        model_label = MODEL_DISPLAY.get(model_key, model_key)
        color = CROSS_MODEL_COLORS.get(model_key, "#333")
        summary = predictor_summary(dfs)
        model_sections += f'<h3 style="border-left: 4px solid {color}; padding-left: 10px;">{model_label}</h3>\n'
        model_sections += build_full_predictor_table(summary)
        model_sections += build_emergence_narrative(summary)

    # Family summary
    family_sections = ""
    for family_name, family_members in MODEL_FAMILIES:
        avail = [m for m in family_members if m in all_data]
        if not avail:
            continue
        family_sections += f"<li><strong>{family_name}</strong>: {', '.join(MODEL_DISPLAY.get(m, m) for m in avail)}</li>\n"

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Exp 5: Cross-Model 5-Predictor RSA ({n_models} models)</title>
<style>{CSS}</style>
</head>
<body>

<h1>Exp 5: Cross-Model 5-Predictor RSA</h1>
<p>Generated: {now}</p>
<p>{n_models} models across verb, object, and period token positions.
5-predictor regression (A-E) on conditions C1-C4 with 10K permutations and FDR correction.</p>

<p><strong>Model families:</strong></p>
<ul>
{family_sections}
</ul>

<div class="sig-legend">
<strong>Significance:</strong> * p<sub>FDR</sub> &lt; .05, ** p<sub>FDR</sub> &lt; .01, *** p<sub>FDR</sub> &lt; .001.
Filled dots on plots = FDR-significant layers.
</div>

<div class="note">
<strong>5-Predictor Model Specification:</strong><br>
<strong>A</strong> (Full Attribution): both sentences in C1 (mental_state) — tests bound {"{"}subject + mental verb + object{"}"}<br>
<strong>B</strong> (Mental Verb + Object): both in C1 or C2 — tests mental verb with object, no subject required<br>
<strong>C</strong> (Mental Verb Presence): both in C1, C2, or C3 — tests presence of mental verb in any form<br>
<strong>D</strong> (Verb + Object): both in C1, C2, or C4 — tests verb-object binding regardless of verb type<br>
<strong>E</strong> (Subject + Verb + Object): both in C1 or C4 — tests full sentence structure regardless of verb type<br>
<br>
<strong>Key question:</strong> Does predictor A (full mental state attribution) capture unique variance
beyond what is explained by its component parts (B-E)?
</div>

<div class="toc">
<h3>Contents</h3>
<ul>
<li><a href="#peak-summary">1. Model A Peak Summary (All Models)</a></li>
<li><a href="#model-a-cross">2. Model A — Cross-Model by Position</a></li>
<li><a href="#all-pred-grid">3. All Predictors — Family-Grouped Grid</a></li>
<li><a href="#per-model">4. Per-Model Detailed Tables</a></li>
</ul>
</div>

<h2 id="peak-summary">1. Model A (Full Attribution) — Peak Summary</h2>
<p>Peak statistics for predictor A at each token position across all {n_models} models.
Model A captures the unique variance attributable to bound mental state attribution
({"{"}subject + mental verb + object{"}"}) after controlling for component features.</p>
{peak_table}

<h2 id="model-a-cross">2. Model A — Cross-Model by Position</h2>
<p>Each panel shows one model (family-grouped). Lines show Model A beta across layers
at verb (green), object (orange), and period (blue) positions.
Filled dots = FDR-significant. Does the attribution signal emerge at the same
position and depth across architectures?</p>
<img src="data:image/png;base64,{img_a}" alt="Cross-Model Model A">

<h2 id="all-pred-grid">3. All Predictors — Family-Grouped Grid</h2>
<p>Each panel shows one model with all 5 predictors (A-E) at the period position.
Family-grouped layout: LLaMA (row 1), Gemma (row 2), Qwen (row 3).</p>
<img src="data:image/png;base64,{img_grid}" alt="Full Grid">

<h2 id="per-model">4. Per-Model Detailed Tables</h2>
<p>Full predictor summary and emergence narrative for each model.</p>
{model_sections}

<hr>
<p style="color:#888; font-size:12px">Generated by <code>code/rsa/5_predictors/2b_positional_report.py --all-models</code> on {now}</p>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Positional 5-Predictor RSA Summary Report"
    )
    add_model_argument(parser)
    parser.add_argument("--all-models", action="store_true",
                        help="Generate cross-model comparison + all per-model reports")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_models:
        all_data = {}
        for model_key in VALID_MODELS:
            dfs = load_model_data(model_key)
            if dfs:
                all_data[model_key] = dfs
                print(f"  {model_key}: {list(dfs.keys())}")

        if not all_data:
            print("No model data found.")
            sys.exit(1)

        # Cross-model report
        if len(all_data) >= 2:
            fig_dir = ensure_dir(RESULTS_DIR / "comparisons" / "rsa" / "5_predictors" / "figures")
            report_dir = ensure_dir(RESULTS_DIR / "comparisons" / "rsa" / "5_predictors")
            print("Generating cross-model positional report...")
            html = build_cross_model_html(all_data, fig_dir)
            report_path = report_dir / "5_predictors_cross_model_positional.html"
            with open(report_path, "w") as f:
                f.write(html)
            print(f"  Saved: {report_path}")

        # Per-model reports
        for model_key, dfs in all_data.items():
            set_model(model_key)
            fig_dir = ensure_dir(data_dir("rsa") / "5_predictors" / "figures")
            report_dir = ensure_dir(results_dir("rsa") / "reports")
            print(f"Generating per-model report for {model_key}...")
            html = build_per_model_html(dfs, model_key, fig_dir)
            report_path = report_dir / "5_predictors_positional_report.html"
            with open(report_path, "w") as f:
                f.write(html)
            print(f"  Saved: {report_path}")
    else:
        set_model(args.model)
        model_key = get_model()
        dfs = load_model_data(model_key)
        if not dfs:
            print(f"No data found for {model_key}")
            sys.exit(1)
        print(f"Positions: {list(dfs.keys())}")
        fig_dir = ensure_dir(data_dir("rsa") / "5_predictors" / "figures")
        report_dir = ensure_dir(results_dir("rsa") / "reports")
        html = build_per_model_html(dfs, model_key, fig_dir)
        report_path = report_dir / "5_predictors_positional_report.html"
        with open(report_path, "w") as f:
            f.write(html)
        print(f"  Saved: {report_path}")

    print("Done.")


if __name__ == "__main__":
    main()
