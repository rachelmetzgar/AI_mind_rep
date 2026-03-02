#!/usr/bin/env python3
"""
Generate Probe Training Comparison — All 6 Versions on Unified Scale.

Reads layerwise_probe_stats.csv from each version, computes a shared y-axis
range, and generates acc_by_layer_group and best_test_acc_by_layer figures
on the same scale for fair visual comparison.

Outputs:
    results/comparisons/probe_training/probe_training_comparison.html
    results/comparisons/probe_training/probe_training_comparison.md

Usage:
    python code/analysis/gen_probe_training_comparison.py

Rachel C. Metzgar · Feb 2026
"""

import os, sys, base64, io, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config

# ========================== CONFIG ========================== #
VERSIONS = [
    ("names",             "Names (Original)",           "Original Sam/Casey/Copilot names. Known name confound: probes may encode partner name tokens rather than abstract identity."),
    ("balanced_names",    "Balanced Names",              "Gender-balanced names (no Copilot). Removes gender confound but names still provide lexical signal."),
    ("balanced_gpt",      "Balanced GPT",                "Like balanced names but with GPT-4 replacing Copilot as AI partner. Tests whether AI partner identity matters."),
    ("labels",            "Labels (Primary)",            'Partner labeled as "a Human" / "an AI". Primary version: minimal lexical confound, tests abstract identity representation.'),
    ("labels_turnwise",   "Labels Turnwise",             'Same as Labels but partner messages prefixed "Human:" / "AI:" instead of "Partner:". Tests turn-level identity reinforcement.'),
    ("you_are_labels",    "You Are Labels",              '"You are talking to {type}." Uses direct framing instead of "you believe you are speaking with."'),
    ("you_are_labels_turnwise", "You Are Labels Turnwise", '"You are talking to {type}" + "Human:"/"AI:" turn prefix. Combines direct framing with turn-level reinforcement.'),
    ("you_are_balanced_gpt", "You Are Balanced GPT",     '"You are talking to {name} ({type})." Direct framing with gender-balanced names + GPT-4.'),
    ("nonsense_codeword", "Nonsense Codeword (Control)", 'Token-matched control: "Your session code word is {a Human/an AI}". Same tokens present but no identity meaning.'),
    ("nonsense_ignore",   "Nonsense Ignore (Control)",   "Token-present with ignore instruction: tokens appear but model is told to disregard them."),
]

RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "results"
OUT_DIR = RESULTS_ROOT / "comparisons" / "probe_training"

LAYER_GROUPS = {
    "early":  (0, 13),
    "middle": (14, 27),
    "late":   (28, 40),
}

N_LAYERS = 41


# ========================== HELPERS ========================== #
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_version_data(version):
    csv_path = RESULTS_ROOT / version / "probe_training" / "layerwise_probe_stats.csv"
    if not csv_path.exists():
        print(f"WARNING: {csv_path} not found, skipping {version}")
        return None
    return pd.read_csv(csv_path)


def load_accuracy_summary(version):
    """Load raw accuracy_summary.pkl for reading/control probes."""
    from config import set_version
    set_version(version)
    probe_dir = config.PATHS.probe_checkpoints / "turn_5"
    data = {}
    for pt in ["reading_probe", "control_probe"]:
        pkl_path = probe_dir / pt / "accuracy_summary.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            data[pt] = {k: np.array(v, dtype=float) for k, v in d.items()}
    return data


# ========================== FIGURE GENERATORS ========================== #
def make_best_test_acc_figure(df, label, y_min, y_max):
    """Layerwise best test accuracy — reading vs control."""
    layers = df["layer"].values
    r_acc = df["reading_best_acc"].values
    c_acc = df["control_best_acc"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, r_acc, "o-", color="#2166ac", markersize=4, linewidth=1.5, label="Reading probe")
    ax.plot(layers, c_acc, "s-", color="#b2182b", markersize=4, linewidth=1.5, label="Control probe")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.set_title(f"{label}", fontsize=13)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(range(0, N_LAYERS, 5))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def make_layer_group_figure(df, label, y_min, y_max):
    """Accuracy by layer group — reading vs control bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax_idx, (metric_label, r_col, c_col) in enumerate([
        ("Best Test Acc", "reading_best_acc", "control_best_acc"),
        ("Final Test Acc", "reading_final_acc", "control_final_acc"),
    ]):
        ax = axes[ax_idx]
        group_names = list(LAYER_GROUPS.keys())
        x = np.arange(len(group_names))
        width = 0.35
        r_means, c_means, r_sems, c_sems = [], [], [], []

        for gname in group_names:
            lo, hi = LAYER_GROUPS[gname]
            r_vals = df[(df["layer"] >= lo) & (df["layer"] <= hi)][r_col].values
            c_vals = df[(df["layer"] >= lo) & (df["layer"] <= hi)][c_col].values
            r_means.append(np.mean(r_vals))
            c_means.append(np.mean(c_vals))
            r_sems.append(np.std(r_vals, ddof=1) / np.sqrt(len(r_vals)))
            c_sems.append(np.std(c_vals, ddof=1) / np.sqrt(len(c_vals)))

        ax.bar(x - width/2, r_means, width, yerr=r_sems, color="#2166ac",
               alpha=0.8, capsize=4, label="Reading")
        ax.bar(x + width/2, c_means, width, yerr=c_sems, color="#b2182b",
               alpha=0.8, capsize=4, label="Control")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer Group", fontsize=12)
        ax.set_ylabel("Mean Accuracy", fontsize=12)
        ax.set_title(metric_label, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({LAYER_GROUPS[g][0]}\u2013{LAYER_GROUPS[g][1]})" for g in group_names])
        ax.set_ylim(y_min, y_max)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{label}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ========================== SUMMARY STATS ========================== #
def compute_summary(df):
    """Compute summary stats for a version."""
    r_acc = df["reading_best_acc"].values
    c_acc = df["control_best_acc"].values

    r_peak_layer = int(np.argmax(r_acc))
    c_peak_layer = int(np.argmax(c_acc))

    t_stat, p_val = stats.ttest_rel(r_acc, c_acc)
    diff = r_acc - c_acc
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    def fmt_p(p):
        if p < 0.0001:
            return "p<.0001"
        return f"p={p:.3f}"

    return {
        "r_peak": f"{r_acc[r_peak_layer]*100:.1f}% (L{r_peak_layer})",
        "c_peak": f"{c_acc[c_peak_layer]*100:.1f}% (L{c_peak_layer})",
        "r_mean": f"{np.mean(r_acc)*100:.1f}%",
        "c_mean": f"{np.mean(c_acc)*100:.1f}%",
        "diff": f"{np.mean(diff)*100:+.1f}%",
        "t": f"t(40)={t_stat:.2f}",
        "p": fmt_p(p_val),
        "d": f"d={d:.2f}",
    }


# ========================== MAIN ========================== #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all version data
    all_data = {}
    for key, label, desc in VERSIONS:
        df = load_version_data(key)
        if df is not None:
            all_data[key] = df

    if not all_data:
        print("ERROR: No data found.")
        return

    # Compute unified y-axis range across all versions
    all_acc_values = []
    for key, df in all_data.items():
        all_acc_values.extend(df["reading_best_acc"].values)
        all_acc_values.extend(df["control_best_acc"].values)
        all_acc_values.extend(df["reading_final_acc"].values)
        all_acc_values.extend(df["control_final_acc"].values)

    all_acc_values = np.array(all_acc_values)
    global_min = np.floor(all_acc_values.min() * 20) / 20 - 0.02
    global_max = np.ceil(all_acc_values.max() * 20) / 20 + 0.02
    Y_MIN = min(0.40, global_min)
    Y_MAX = global_max

    print(f"Unified y-axis: [{Y_MIN:.2f}, {Y_MAX:.2f}]")
    print(f"Data range: [{all_acc_values.min():.3f}, {all_acc_values.max():.3f}]")

    # Generate figures and compute stats
    group_figs_b64 = {}
    layer_figs_b64 = {}
    summaries = {}

    for key, label, desc in VERSIONS:
        if key not in all_data:
            continue
        df = all_data[key]
        summaries[key] = compute_summary(df)
        group_figs_b64[key] = fig_to_b64(make_layer_group_figure(df, label, Y_MIN, Y_MAX))
        layer_figs_b64[key] = fig_to_b64(make_best_test_acc_figure(df, label, Y_MIN, Y_MAX))
        print(f"  {key}: done")

    # ========================== HTML ========================== #
    html_parts = []
    html_parts.append(f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Probe Training Comparison &mdash; All 6 Versions</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
  h2 {{ color: #283593; margin-top: 50px; border-bottom: 2px solid #c5cae9; padding-bottom: 8px; }}
  h3 {{ color: #3949ab; margin-top: 30px; }}
  .version-block {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }}
  .version-block h3 {{ margin-top: 0; }}
  .desc {{ color: #666; font-size: 0.95em; margin-bottom: 12px; }}
  .fig {{ text-align: center; margin: 15px 0; }}
  .fig img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .summary-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  .summary-table th {{ background: #1a237e; color: white; padding: 10px 12px; text-align: center; }}
  .summary-table td {{ padding: 8px 12px; border: 1px solid #ddd; text-align: center; }}
  .summary-table tr:nth-child(even) {{ background: #f5f5f5; }}
  .highlight {{ background: #e8f5e9 !important; }}
  .control-row {{ background: #fff3e0 !important; }}
  .timestamp {{ color: #999; font-size: 0.85em; }}
  .scale-note {{ background: #e3f2fd; padding: 10px 15px; border-left: 4px solid #1565c0; margin: 15px 0; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Probe Training Comparison &mdash; All 6 Data Versions</h1>
<p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<p>Linear probes trained on LLaMA-2-13B-Chat hidden states (turn 5, full conversation) to classify
whether the conversation partner is human or AI. Each version uses a different system prompt strategy.
Reading probes append a reflective suffix; control probes probe at the natural generation position.</p>

<div class="scale-note">All figures use a unified y-axis scale [{Y_MIN:.2f}, {Y_MAX:.2f}] for direct visual comparison across versions.</div>

<h2>Summary Table</h2>

<table class="summary-table">
<tr><th>Version</th><th>Reading Peak</th><th>Control Peak</th><th>Reading M</th><th>Control M</th><th>Diff (R&minus;C)</th><th>Paired t</th><th>p</th><th>Cohen&rsquo;s d</th></tr>''')

    for key, label, desc in VERSIONS:
        if key not in summaries:
            continue
        s = summaries[key]
        row_class = ""
        if key.startswith("nonsense"):
            row_class = ' class="control-row"'
        elif key == "labels":
            row_class = ' class="highlight"'
        html_parts.append(
            f'<tr{row_class}><td style="text-align:left;"><b>{label}</b></td>'
            f'<td>{s["r_peak"]}</td><td>{s["c_peak"]}</td>'
            f'<td>{s["r_mean"]}</td><td>{s["c_mean"]}</td><td>{s["diff"]}</td>'
            f'<td>{s["t"]}</td><td>{s["p"]}</td><td>{s["d"]}</td></tr>'
        )

    html_parts.append("</table>")
    html_parts.append('<p style="font-size:0.85em;color:#666;">Green = primary version (labels). Orange = nonsense controls.</p>')

    # Section 1: acc_by_layer_group
    html_parts.append("<h2>Probe Accuracy by Layer Group</h2>")
    html_parts.append("<p>Mean accuracy (&plusmn; SEM across layers) for early (0&ndash;13), middle (14&ndash;27), and late (28&ndash;40) layer groups. Left panel: best test accuracy. Right panel: final-epoch test accuracy.</p>")

    for key, label, desc in VERSIONS:
        if key not in group_figs_b64:
            continue
        html_parts.append(f'''<div class="version-block">
<h3>{label}</h3>
<p class="desc">{desc}</p>
<div class="fig"><img src="data:image/png;base64,{group_figs_b64[key]}" alt="{label} — Accuracy by Layer Group"></div>
</div>''')

    # Section 2: best_test_acc_by_layer
    html_parts.append("<h2>Layerwise Best Test Accuracy</h2>")
    html_parts.append("<p>Best test accuracy (across 50 training epochs) for reading and control probes at each of 41 transformer layers. Dashed line = chance (50%).</p>")

    for key, label, desc in VERSIONS:
        if key not in layer_figs_b64:
            continue
        html_parts.append(f'''<div class="version-block">
<h3>{label}</h3>
<p class="desc">{desc}</p>
<div class="fig"><img src="data:image/png;base64,{layer_figs_b64[key]}" alt="{label} — Best Test Acc by Layer"></div>
</div>''')

    html_parts.append("</body></html>")

    html_content = "\n".join(html_parts)
    html_path = OUT_DIR / "probe_training_comparison.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"\nSaved HTML: {html_path} ({len(html_content)//1024} KB)")

    # ========================== SAVE INDIVIDUAL PNGS ========================== #
    # Save unified-scale PNGs into figures/ subfolder
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, label, desc in VERSIONS:
        if key not in all_data:
            continue
        df = all_data[key]
        fig = make_layer_group_figure(df, label, Y_MIN, Y_MAX)
        fig.savefig(fig_dir / f"acc_by_layer_group_{key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        fig = make_best_test_acc_figure(df, label, Y_MIN, Y_MAX)
        fig.savefig(fig_dir / f"best_test_acc_by_layer_{key}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("Saved unified-scale PNGs to figures/")

    # ========================== MD ========================== #
    md_parts = []
    md_parts.append("# Probe Training Comparison — All 6 Data Versions\n")
    md_parts.append("Linear probes trained on LLaMA-2-13B-Chat hidden states (turn 5, full conversation) to classify whether the conversation partner is human or AI. Each version uses a different system prompt strategy. Reading probes append a reflective suffix; control probes probe at the natural generation position.\n")
    md_parts.append(f"> All figures use a unified y-axis scale [{Y_MIN:.2f}, {Y_MAX:.2f}] for direct visual comparison across versions.\n")

    md_parts.append("## Summary Table\n")
    md_parts.append("| Version | Reading Peak | Control Peak | Reading M | Control M | Diff (R-C) | Paired t | p | Cohen's d |")
    md_parts.append("|---------|-------------|-------------|-----------|-----------|------------|----------|---|-----------|")
    for key, label, desc in VERSIONS:
        if key not in summaries:
            continue
        s = summaries[key]
        md_parts.append(f"| **{label}** | {s['r_peak']} | {s['c_peak']} | {s['r_mean']} | {s['c_mean']} | {s['diff']} | {s['t']} | {s['p']} | {s['d']} |")

    md_parts.append("\n---\n")
    md_parts.append("## Probe Accuracy by Layer Group\n")
    md_parts.append("Mean accuracy (+/- SEM across layers) for early (0-13), middle (14-27), and late (28-40) layer groups. Left panel: best test accuracy. Right panel: final-epoch test accuracy.\n")

    for i, (key, label, desc) in enumerate(VERSIONS):
        if key not in all_data:
            continue
        md_parts.append(f"### {i+1}. {label}\n")
        md_parts.append(f"{desc}\n")
        md_parts.append(f"![{label} — Accuracy by Layer Group](figures/acc_by_layer_group_{key}.png)\n")

    md_parts.append("---\n")
    md_parts.append("## Layerwise Best Test Accuracy\n")
    md_parts.append("Best test accuracy (across 50 training epochs) for reading and control probes at each of 41 transformer layers. Dashed line = chance (50%).\n")

    for i, (key, label, desc) in enumerate(VERSIONS):
        if key not in all_data:
            continue
        md_parts.append(f"### {i+1}. {label}\n")
        md_parts.append(f"{desc}\n")
        md_parts.append(f"![{label} — Best Test Acc by Layer](figures/best_test_acc_by_layer_{key}.png)\n")

    md_content = "\n".join(md_parts)
    md_path = OUT_DIR / "probe_training_comparison.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved MD: {md_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
