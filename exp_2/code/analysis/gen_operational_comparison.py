#!/usr/bin/env python3
"""
Operational Probe Comparison — balanced_gpt vs nonsense_codeword.

Compares operational (control) probe accuracy across the Partner Identity
(balanced_gpt) and Control (nonsense_codeword) dataset versions. Generates
a two-panel figure, statistical tests (binomial vs chance, between-version
z-tests), and an HTML report with auto-generated Markdown companion.

Outputs:
    results/comparisons/probe_training/operational_comparison.html
    results/comparisons/probe_training/operational_comparison.md
    results/comparisons/probe_training/figures/operational_comparison.png

Usage:
    python code/analysis/gen_operational_comparison.py

Rachel C. Metzgar · Mar 2026
"""

import sys, base64, io
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.report_utils import save_report

# ========================== CONFIG ========================== #
RESULTS_ROOT = Path(__file__).resolve().parent.parent.parent / "results"
OUT_DIR = RESULTS_ROOT / "comparisons" / "probe_training"
FIG_DIR = OUT_DIR / "figures"

N_TEST = 400   # 20% of ~2000 samples
N_LAYERS = 41

LAYER_GROUPS = {
    "early":  (0, 13),
    "middle": (14, 27),
    "late":   (28, 40),
}

VERSIONS = {
    "balanced_gpt": {
        "label": "Partner Identity",
        "color": "#5a8a6a",
        "marker": "o-",
    },
    "nonsense_codeword": {
        "label": "Control",
        "color": "#555555",
        "marker": "s-",
    },
}


# ========================== HELPERS ========================== #
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def load_version_data(version):
    csv_path = RESULTS_ROOT / "versions" / version / "probe_training" / "layerwise_probe_stats.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return None
    return pd.read_csv(csv_path)


def fmt_p(p):
    if p < 0.0001:
        return "&lt;.0001"
    return f"{p:.4f}"


def fmt_p_plain(p):
    if p < 0.0001:
        return "<.0001"
    return f"{p:.4f}"


# ========================== STATISTICAL TESTS ========================== #
def binomial_vs_chance(acc_array, n_test=N_TEST):
    """One-sided binomial test (greater than 0.5) per layer, FDR corrected."""
    raw_pvals = []
    for acc in acc_array:
        k = int(round(acc * n_test))
        result = stats.binomtest(k, n_test, 0.5, alternative="greater")
        raw_pvals.append(result.pvalue)
    raw_pvals = np.array(raw_pvals)
    reject, pvals_fdr, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
    return raw_pvals, pvals_fdr, reject


def between_version_ztest(acc_a, acc_b, n_test=N_TEST):
    """Two-sided z-test comparing two proportions per layer, FDR corrected."""
    raw_pvals = []
    z_stats = []
    for a, b in zip(acc_a, acc_b):
        count = np.array([int(round(a * n_test)), int(round(b * n_test))])
        nobs = np.array([n_test, n_test])
        z, p = proportions_ztest(count, nobs, alternative="two-sided")
        z_stats.append(z)
        raw_pvals.append(p)
    raw_pvals = np.array(raw_pvals)
    z_stats = np.array(z_stats)
    reject, pvals_fdr, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")
    return z_stats, raw_pvals, pvals_fdr, reject


# ========================== FIGURE ========================== #
def make_figure(bg_best, bg_final, nc_best, nc_final,
                bg_above_chance):
    """Two-panel figure: best acc (left), final acc (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    layers = np.arange(N_LAYERS)

    for ax_idx, (metric_label, bg_acc, nc_acc) in enumerate([
        ("Best Test Accuracy", bg_best, nc_best),
        ("Final Test Accuracy", bg_final, nc_final),
    ]):
        ax = axes[ax_idx]

        # Background bands where balanced_gpt is above chance (FDR sig)
        for l in range(N_LAYERS):
            if bg_above_chance[l]:
                ax.axvspan(l - 0.4, l + 0.4, alpha=0.12, color="#a5d6a7", zorder=0)

        # Plot lines
        bg_cfg = VERSIONS["balanced_gpt"]
        nc_cfg = VERSIONS["nonsense_codeword"]
        ax.plot(layers, bg_acc, bg_cfg["marker"], color=bg_cfg["color"],
                markersize=4, linewidth=1.5, label=bg_cfg["label"])
        ax.plot(layers, nc_acc, nc_cfg["marker"], color=nc_cfg["color"],
                markersize=4, linewidth=1.5, label=nc_cfg["label"])

        # Chance line
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")

        # Star at peak layer for each version
        bg_peak = int(np.argmax(bg_acc))
        nc_peak = int(np.argmax(nc_acc))
        ax.plot(bg_peak, bg_acc[bg_peak] + 0.008, "*", color="#d4a017", markersize=12, zorder=5)
        ax.plot(nc_peak, nc_acc[nc_peak] + 0.008, "*", color="#d4a017", markersize=12, zorder=5)

        ax.set_xlabel("Layer", fontsize=12)
        if ax_idx == 0:
            ax.set_ylabel("Operational Probe Accuracy", fontsize=12)
        ax.set_title(metric_label, fontsize=13)
        ax.set_xlim(-0.5, N_LAYERS - 0.5)
        ax.set_xticks(range(0, N_LAYERS, 5))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # Unified y-axis
    all_vals = np.concatenate([bg_best, bg_final, nc_best, nc_final])
    y_min = min(0.45, np.floor(all_vals.min() * 20) / 20 - 0.02)
    y_max = np.ceil(all_vals.max() * 20) / 20 + 0.03
    axes[0].set_ylim(y_min, y_max)

    fig.suptitle("Operational Probe Accuracy — Partner Identity vs Control Dataset",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ========================== MAIN ========================== #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    bg_df = load_version_data("balanced_gpt")
    nc_df = load_version_data("nonsense_codeword")
    if bg_df is None or nc_df is None:
        print("ERROR: Missing data files. Aborting.")
        return

    bg_best = bg_df["control_best_acc"].values
    bg_final = bg_df["control_final_acc"].values
    nc_best = nc_df["control_best_acc"].values
    nc_final = nc_df["control_final_acc"].values
    layers = np.arange(N_LAYERS)

    print("=" * 60)
    print("OPERATIONAL PROBE COMPARISON")
    print("=" * 60)

    # --- 1. Binomial tests vs chance ---
    bg_raw_p, bg_fdr_p, bg_reject = binomial_vs_chance(bg_best)
    nc_raw_p, nc_fdr_p, nc_reject = binomial_vs_chance(nc_best)

    bg_n_sig = int(np.sum(bg_reject))
    nc_n_sig = int(np.sum(nc_reject))
    print(f"\nAbove-chance layers (FDR q<.05):")
    print(f"  Partner Identity: {bg_n_sig}/{N_LAYERS}")
    print(f"  Control:          {nc_n_sig}/{N_LAYERS}")

    # --- 2. Between-version z-tests ---
    z_stats, zraw_p, zfdr_p, z_reject = between_version_ztest(bg_best, nc_best)
    z_n_sig = int(np.sum(z_reject))
    print(f"\nBetween-version sig layers (FDR q<.05): {z_n_sig}/{N_LAYERS}")
    if z_n_sig > 0:
        sig_layers = np.where(z_reject)[0]
        print(f"  Layers: {list(sig_layers)}")

    # --- 3. Summary stats ---
    # Paired t-test across layers
    t_best, p_best = stats.ttest_rel(bg_best, nc_best)
    diff_best = bg_best - nc_best
    d_best = np.mean(diff_best) / np.std(diff_best, ddof=1) if np.std(diff_best, ddof=1) > 0 else 0

    t_final, p_final = stats.ttest_rel(bg_final, nc_final)
    diff_final = bg_final - nc_final
    d_final = np.mean(diff_final) / np.std(diff_final, ddof=1) if np.std(diff_final, ddof=1) > 0 else 0

    bg_peak_layer = int(np.argmax(bg_best))
    nc_peak_layer = int(np.argmax(nc_best))

    print(f"\nPeak accuracy:")
    print(f"  Partner Identity: {bg_best[bg_peak_layer]:.3f} (layer {bg_peak_layer})")
    print(f"  Control:          {nc_best[nc_peak_layer]:.3f} (layer {nc_peak_layer})")
    print(f"\nPaired t-test (best acc): t(40)={t_best:.3f}, p={p_best:.4f}, d={d_best:.3f}")

    # Layer group means
    group_stats = {}
    for gname, (lo, hi) in LAYER_GROUPS.items():
        bg_vals = bg_best[lo:hi+1]
        nc_vals = nc_best[lo:hi+1]
        group_stats[gname] = {
            "bg_mean": np.mean(bg_vals),
            "bg_sem": np.std(bg_vals, ddof=1) / np.sqrt(len(bg_vals)),
            "nc_mean": np.mean(nc_vals),
            "nc_sem": np.std(nc_vals, ddof=1) / np.sqrt(len(nc_vals)),
        }

    # --- 4. Generate figure ---
    fig = make_figure(bg_best, bg_final, nc_best, nc_final, bg_reject)
    fig.savefig(FIG_DIR / "operational_comparison.png", dpi=200, bbox_inches="tight")
    fig_b64 = fig_to_b64(make_figure(bg_best, bg_final, nc_best, nc_final, bg_reject))
    print(f"\nSaved: {FIG_DIR / 'operational_comparison.png'}")

    # ========================== HTML REPORT ========================== #

    # Summary stats table
    summary_rows = f"""
<tr>
  <td>Best Test Acc</td>
  <td>{bg_best[bg_peak_layer]*100:.1f}% (L{bg_peak_layer})</td>
  <td>{nc_best[nc_peak_layer]*100:.1f}% (L{nc_peak_layer})</td>
  <td>{np.mean(bg_best)*100:.1f}%</td>
  <td>{np.mean(nc_best)*100:.1f}%</td>
  <td>{np.mean(diff_best)*100:+.1f}%</td>
  <td>t(40)={t_best:.2f}</td>
  <td>{fmt_p(p_best)}</td>
  <td>{d_best:.2f}</td>
</tr>
<tr>
  <td>Final Test Acc</td>
  <td>{bg_final[int(np.argmax(bg_final))]*100:.1f}% (L{int(np.argmax(bg_final))})</td>
  <td>{nc_final[int(np.argmax(nc_final))]*100:.1f}% (L{int(np.argmax(nc_final))})</td>
  <td>{np.mean(bg_final)*100:.1f}%</td>
  <td>{np.mean(nc_final)*100:.1f}%</td>
  <td>{np.mean(diff_final)*100:+.1f}%</td>
  <td>t(40)={t_final:.2f}</td>
  <td>{fmt_p(p_final)}</td>
  <td>{d_final:.2f}</td>
</tr>"""

    # Above-chance results
    bg_sig_layers = list(np.where(bg_reject)[0])
    nc_sig_layers = list(np.where(nc_reject)[0])

    above_chance_html = f"""
<p><strong>Partner Identity (balanced_gpt):</strong> {bg_n_sig}/{N_LAYERS} layers above chance (FDR q&lt;.05)</p>
"""
    if bg_n_sig > 0:
        above_chance_html += "<p>Significant layers: " + ", ".join(str(l) for l in bg_sig_layers) + "</p>\n"
    above_chance_html += f"""
<p><strong>Control (nonsense_codeword):</strong> {nc_n_sig}/{N_LAYERS} layers above chance (FDR q&lt;.05)</p>
"""
    if nc_n_sig > 0:
        above_chance_html += "<p>Significant layers: " + ", ".join(str(l) for l in nc_sig_layers) + "</p>\n"
    else:
        above_chance_html += "<p>No layers reached significance, consistent with operational probes learning no meaningful signal from nonsense context.</p>\n"

    # Between-version results
    between_html = f"<p><strong>Significant layers (FDR q&lt;.05):</strong> {z_n_sig}/{N_LAYERS}</p>\n"
    if z_n_sig > 0:
        z_sig_layers = list(np.where(z_reject)[0])
        between_html += "<p>Layers: " + ", ".join(str(l) for l in z_sig_layers) + "</p>\n"
        between_html += "<table class='stats-table'>\n"
        between_html += "<tr><th>Layer</th><th>Partner Identity</th><th>Control</th><th>z</th><th>p (raw)</th><th>p (FDR)</th></tr>\n"
        for l in z_sig_layers:
            between_html += (
                f"<tr><td>{l}</td><td>{bg_best[l]:.3f}</td><td>{nc_best[l]:.3f}</td>"
                f"<td>{z_stats[l]:.2f}</td><td>{fmt_p(zraw_p[l])}</td><td>{fmt_p(zfdr_p[l])}</td></tr>\n"
            )
        between_html += "</table>\n"
    else:
        between_html += "<p>No individual layers reached significance after FDR correction.</p>\n"

    # Layer group table
    group_rows = ""
    for gname, (lo, hi) in LAYER_GROUPS.items():
        gs = group_stats[gname]
        group_rows += (
            f"<tr><td>{gname} ({lo}&ndash;{hi})</td>"
            f"<td>{gs['bg_mean']*100:.1f}% &plusmn; {gs['bg_sem']*100:.1f}%</td>"
            f"<td>{gs['nc_mean']*100:.1f}% &plusmn; {gs['nc_sem']*100:.1f}%</td>"
            f"<td>{(gs['bg_mean']-gs['nc_mean'])*100:+.1f}%</td></tr>\n"
        )

    # Full layerwise table
    layerwise_rows = ""
    for i in range(N_LAYERS):
        bg_sig_mark = "*" if bg_reject[i] else ""
        nc_sig_mark = "*" if nc_reject[i] else ""
        z_sig_mark = "*" if z_reject[i] else ""
        row_style = ""
        if bg_reject[i]:
            row_style = ' style="background:#e8f5e9;"'
        elif z_reject[i]:
            row_style = ' style="background:#ffffcc;"'
        layerwise_rows += (
            f"<tr{row_style}>"
            f"<td>{i}</td>"
            f"<td>{bg_best[i]:.3f}</td><td>{fmt_p(bg_raw_p[i])}</td><td>{fmt_p(bg_fdr_p[i])}</td><td>{bg_sig_mark}</td>"
            f"<td>{nc_best[i]:.3f}</td><td>{fmt_p(nc_raw_p[i])}</td><td>{fmt_p(nc_fdr_p[i])}</td><td>{nc_sig_mark}</td>"
            f"<td>{z_stats[i]:.2f}</td><td>{fmt_p(zraw_p[i])}</td><td>{fmt_p(zfdr_p[i])}</td><td>{z_sig_mark}</td>"
            f"</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Operational Probe Comparison &mdash; Partner Identity vs Control</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
  h2 {{ color: #283593; margin-top: 50px; border-bottom: 2px solid #c5cae9; padding-bottom: 8px; }}
  h3 {{ color: #3949ab; margin-top: 30px; }}
  .fig {{ text-align: center; margin: 25px 0; }}
  .fig img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .fig-caption {{ font-size: 0.9em; color: #666; margin-top: 8px; font-style: italic; }}
  .stats-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 0.9em; }}
  .stats-table th {{ background: #1a237e; color: white; padding: 8px 10px; text-align: center; }}
  .stats-table td {{ padding: 6px 10px; border: 1px solid #ddd; text-align: center; }}
  .stats-table tr:nth-child(even) {{ background: #f5f5f5; }}
  .interpretation {{ background: #f8f9fa; border-left: 4px solid #5a8a6a; padding: 15px 20px; margin: 20px 0; border-radius: 0 4px 4px 0; }}
  .timestamp {{ color: #999; font-size: 0.85em; }}
  .note {{ background: #e3f2fd; padding: 10px 15px; border-left: 4px solid #1565c0; margin: 15px 0; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Operational Probe Comparison &mdash; Partner Identity vs Control</h1>
<p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<p>Compares <strong>operational probe</strong> (probing at the natural generation position, no reflective suffix)
accuracy between the Partner Identity dataset (<code>balanced_gpt</code>) and the Control dataset
(<code>nonsense_codeword</code>). If partner identity information is genuinely represented in the model&rsquo;s
operational context, the Partner Identity version should show above-chance accuracy while the Control version
should remain at chance.</p>

<div class="note">
<strong>Column note:</strong> The operational probe corresponds to <code>control_best_acc</code> /
<code>control_final_acc</code> in the CSV files (column naming predates the probe rename).
N<sub>test</sub> = {N_TEST} per layer.
</div>

<!-- ============================================================ -->
<h2>Statistical Methods</h2>

<h3>Data Source</h3>
<p>Each dataset version has a CSV file (<code>layerwise_probe_stats.csv</code>) containing per-layer probe
accuracy summaries. The <strong>operational probe</strong> is a linear classifier trained on LLaMA-2-13B-Chat
hidden states extracted at the natural generation position (the <code>[/INST]</code> token where the model
is about to produce its next response). No reflective suffix is appended &mdash; this tests whether partner
identity is represented in the model&rsquo;s ordinary operational context.</p>

<p>We compare two dataset versions:</p>
<ul>
  <li><strong>Partner Identity (<code>balanced_gpt</code>):</strong> System prompt tells the model it is
  speaking with a specific human or AI partner. The operational probe should detect partner identity signal
  if the model internalizes this information during normal generation.</li>
  <li><strong>Control (<code>nonsense_codeword</code>):</strong> The same human/AI tokens appear in the system
  prompt but as a meaningless &ldquo;session code word&rdquo; rather than a partner identity. This controls
  for the mere presence of identity-related tokens. The operational probe should perform at chance.</li>
</ul>

<p>For each version, we use the <code>control_best_acc</code> column (best test accuracy across 50 training
epochs) and <code>control_final_acc</code> column (final epoch test accuracy) from the CSV. Column names
use the pre-rename convention where &ldquo;control&rdquo; refers to the operational probe type.
Each accuracy value is a proportion of correctly classified test samples out of
N<sub>test</sub> = {N_TEST} (20% of ~2000 total samples = 50 agents &times; 40 conversations, stratified
train/test split).</p>

<h3>Test 1: Above-Chance Performance (Binomial Test)</h3>
<p><strong>Question:</strong> At each layer, does the operational probe classify partner type better than
random guessing (50%)?</p>
<p><strong>Procedure:</strong></p>
<ol>
  <li>Convert each layer&rsquo;s accuracy proportion to a success count:
  <code>k = round(accuracy &times; {N_TEST})</code>.</li>
  <li>Run a one-sided exact binomial test (<code>scipy.stats.binomtest(k, {N_TEST}, 0.5,
  alternative=&quot;greater&quot;)</code>) testing H<sub>1</sub>: accuracy &gt; 0.5.</li>
  <li>Collect all {N_LAYERS} raw p-values and apply Benjamini-Hochberg FDR correction
  (<code>statsmodels.stats.multitest.multipletests(..., method=&quot;fdr_bh&quot;)</code>) at &alpha; = 0.05.</li>
  <li>FDR correction is applied <em>separately</em> per version (Partner Identity and Control each get
  their own family of {N_LAYERS} tests).</li>
</ol>
<p><strong>Rationale:</strong> The binomial test is the exact test for whether a proportion differs from a
known reference value, and is appropriate here because each test sample is an independent binary
classification. FDR correction controls the expected proportion of false discoveries across the {N_LAYERS}
layers tested.</p>

<h3>Test 2: Between-Version Comparison (Two-Proportions Z-Test)</h3>
<p><strong>Question:</strong> At each layer, is the Partner Identity operational probe significantly more
accurate than the Control operational probe?</p>
<p><strong>Procedure:</strong></p>
<ol>
  <li>For each layer, convert both versions&rsquo; accuracies to success counts as above.</li>
  <li>Run a two-sided two-proportions z-test
  (<code>statsmodels.stats.proportion.proportions_ztest([k<sub>PI</sub>, k<sub>Ctrl</sub>],
  [{N_TEST}, {N_TEST}], alternative=&quot;two-sided&quot;)</code>).</li>
  <li>Collect {N_LAYERS} raw p-values and apply Benjamini-Hochberg FDR correction at &alpha; = 0.05.</li>
</ol>
<p><strong>Rationale:</strong> The two-proportions z-test compares two independent proportions under the
assumption of large-sample normal approximation to the binomial &mdash; appropriate given
N<sub>test</sub> = {N_TEST}. A two-sided test is used because we do not assume directionality a priori
(although we expect Partner Identity &gt; Control). FDR correction is applied across all {N_LAYERS} layers.</p>

<h3>Test 3: Overall Paired T-Test</h3>
<p><strong>Question:</strong> Across all layers as a whole, does Partner Identity outperform Control?</p>
<p><strong>Procedure:</strong></p>
<ol>
  <li>Treat the {N_LAYERS} layers as paired observations: for each layer <em>i</em>, compute
  d<sub>i</sub> = acc<sub>PI</sub>(i) &minus; acc<sub>Ctrl</sub>(i).</li>
  <li>Run a paired t-test (<code>scipy.stats.ttest_rel</code>) on the {N_LAYERS} difference scores.</li>
  <li>Compute Cohen&rsquo;s d for paired samples:
  d = mean(d<sub>i</sub>) / SD(d<sub>i</sub>).</li>
</ol>
<p><strong>Rationale:</strong> The per-layer z-tests above test each layer independently but have limited
power (N<sub>test</sub> = {N_TEST} per layer). The paired t-test pools evidence across all layers,
treating the layerwise accuracy profile as the unit of analysis. This is the appropriate omnibus test for
whether one version consistently outperforms the other. Cohen&rsquo;s d provides a standardized effect size.</p>

<h3>Layer Group Means</h3>
<p>Layers are grouped into early (0&ndash;13, 14 layers), middle (14&ndash;27, 14 layers), and late
(28&ndash;40, 13 layers). Mean accuracy &plusmn; SEM is reported per group to characterize where in the
network any differences emerge.</p>

<!-- ============================================================ -->
<h2>1. Figure</h2>

<div class="fig">
  <img src="data:image/png;base64,{fig_b64}" alt="Operational Probe Comparison — Partner Identity vs Control">
  <div class="fig-caption">Figure 1. Operational probe accuracy by layer for Partner Identity (green) and Control (gray)
  datasets. Left: best test accuracy across 50 epochs. Right: final-epoch test accuracy.
  Light green bands mark layers where Partner Identity is significantly above chance (binomial test, FDR q&lt;.05).
  Gold star marks the peak accuracy layer.</div>
</div>

<!-- ============================================================ -->
<h2>2. Summary Statistics</h2>

<table class="stats-table">
<tr><th>Metric</th><th>PI Peak</th><th>Ctrl Peak</th><th>PI Mean</th><th>Ctrl Mean</th><th>Diff</th><th>Paired t</th><th>p</th><th>Cohen&rsquo;s d</th></tr>
{summary_rows}
</table>

<div class="interpretation">
<strong>Interpretation:</strong> The Partner Identity version shows consistently higher operational probe accuracy
(peak {bg_best[bg_peak_layer]*100:.1f}% at layer {bg_peak_layer}) compared to the Control version
(peak {nc_best[nc_peak_layer]*100:.1f}% at layer {nc_peak_layer}). The paired t-test across {N_LAYERS} layers
confirms this difference is {"highly " if p_best < 0.001 else ""}significant
(t(40)={t_best:.2f}, p={fmt_p_plain(p_best)}, d={d_best:.2f}).
</div>

<!-- ============================================================ -->
<h2>3. Layer Group Means</h2>

<table class="stats-table">
<tr><th>Layer Group</th><th>Partner Identity (M &plusmn; SEM)</th><th>Control (M &plusmn; SEM)</th><th>Difference</th></tr>
{group_rows}
</table>

<!-- ============================================================ -->
<h2>4. Above-Chance Performance (Binomial Tests)</h2>

<p>One-sided binomial tests (H<sub>1</sub>: accuracy &gt; 0.5) per layer, FDR corrected (Benjamini-Hochberg)
across {N_LAYERS} layers, separately per version.</p>

{above_chance_html}

<div class="interpretation">
<strong>Interpretation:</strong> {"The Partner Identity version shows above-chance operational probe accuracy at " + str(bg_n_sig) + " layers, concentrated in middle-to-late layers where partner identity information is expected to emerge. " if bg_n_sig > 0 else "Neither version shows widespread above-chance performance. "}{"The Control version shows no above-chance layers, confirming that the nonsense codeword context does not provide the model with usable partner identity information at the operational position." if nc_n_sig == 0 else f"The Control version shows {nc_n_sig} above-chance layers, which may reflect weak residual signal."}
</div>

<!-- ============================================================ -->
<h2>5. Between-Version Comparison (Z-Tests)</h2>

<p>Two-proportions z-tests comparing Partner Identity vs Control operational probe accuracy at each layer,
FDR corrected across {N_LAYERS} layers.</p>

{between_html}

<!-- ============================================================ -->
<h2>6. Full Layerwise Statistics</h2>

<p>Green rows: Partner Identity above chance. Yellow rows: versions significantly different.</p>

<div style="overflow-x: auto;">
<table class="stats-table" style="font-size: 0.8em;">
<tr>
  <th rowspan="2">Layer</th>
  <th colspan="4">Partner Identity (balanced_gpt)</th>
  <th colspan="4">Control (nonsense_codeword)</th>
  <th colspan="4">Between-Version</th>
</tr>
<tr>
  <th>Acc</th><th>p (raw)</th><th>p (FDR)</th><th>Sig</th>
  <th>Acc</th><th>p (raw)</th><th>p (FDR)</th><th>Sig</th>
  <th>z</th><th>p (raw)</th><th>p (FDR)</th><th>Sig</th>
</tr>
{layerwise_rows}
</table>
</div>

</body>
</html>"""

    html_path = OUT_DIR / "operational_comparison.html"
    save_report(html, html_path)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
