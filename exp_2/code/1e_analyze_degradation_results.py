#!/usr/bin/env python3
"""
Analyze degradation-probe correlation results.

Tests the Bayesian updating hypothesis: do conversations with more behavioral
degradation show faster decline in probe confidence?

Reads per_conversation_probe_degradation.csv produced by 1d script.
Produces figures and statistical summaries.

Usage:
    python 1e_analyze_degradation_results.py --version labels

Env: behavior_env (no GPU needed)
Rachel C. Metzgar - Feb 2026
"""

import os, sys, base64, io
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config, set_version, add_version_argument, add_model_argument, set_model, get_active_version

# ================================================================
# Version-specific metadata
# ================================================================

VERSION_DISPLAY = {
    "labels":            "Labels",
    "balanced_names":    "Balanced Names",
    "balanced_gpt":      "Balanced GPT",
    "names":             "Names",
    "nonsense_codeword": "Nonsense Codeword",
    "nonsense_ignore":   "Nonsense Ignore",
}

VERSION_PROMPTS = {
    "labels":            "You believe you are speaking with {a Human / an AI}.",
    "balanced_names":    "You believe you are speaking with {Name} ({a Human / an AI}).",
    "balanced_gpt":      "You believe you are speaking with {Name} ({a Human / an AI}).",
    "names":             "You believe you are speaking with {Name} ({a Human / an AI}).",
    "nonsense_codeword": "Your assigned session code word is {a Human / an AI}.",
    "nonsense_ignore":   "Ignore the following phrase: {a Human / an AI}.",
}

VERSION_DESCRIPTIONS = {
    "labels":            "Partner labeled generically as &ldquo;a Human&rdquo; or &ldquo;an AI&rdquo; &mdash; no names, no gender cues.",
    "balanced_names":    "Partner labeled with gender-balanced names (Gregory/Rebecca for human, ChatGPT/Copilot for AI).",
    "balanced_gpt":      "Partner labeled with gender-balanced names (Gregory/Rebecca for human, ChatGPT/GPT-4 for AI).",
    "names":             "Partner labeled with specific names (Sam/Casey for human, ChatGPT/Copilot for AI). Gender-confounded.",
    "nonsense_codeword": "Token-matched control: the identity tokens (&ldquo;a Human&rdquo;/&ldquo;an AI&rdquo;) appear in a semantically inert context (&ldquo;code word&rdquo;). Tests whether behavioral effects require semantic processing of the identity instruction.",
    "nonsense_ignore":   "Token-matched control: model is told to ignore the identity phrase. Tests whether the model can suppress semantic processing of &ldquo;a Human&rdquo;/&ldquo;an AI&rdquo; tokens.",
}


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def compute_degradation_score(group):
    """Compute per-trial degradation score from turn 1 to turn 5."""
    turns = group.sort_values("turn")
    scores = {}

    ttr_vals = turns["sub_ttr"].values
    if len(ttr_vals) >= 2 and ttr_vals[0] > 0:
        scores["ttr_drop"] = (ttr_vals[0] - ttr_vals[-1]) / ttr_vals[0]
    else:
        scores["ttr_drop"] = np.nan

    tri_vals = turns["sub_trigram_rep"].values
    scores["trigram_increase"] = tri_vals[-1] - tri_vals[0] if len(tri_vals) >= 2 else np.nan

    caps_vals = turns["sub_allcaps_ratio"].values
    scores["allcaps_increase"] = caps_vals[-1] - caps_vals[0] if len(caps_vals) >= 2 else np.nan

    excl_vals = turns["sub_exclamation_rate"].values
    scores["excl_increase"] = excl_vals[-1] - excl_vals[0] if len(excl_vals) >= 2 else np.nan

    return pd.Series(scores)


def padded_ylim(ax, pad_frac=0.15):
    """Set y-limits with padding so data isn't clipped."""
    lines = ax.get_lines()
    collections = ax.collections
    all_y = []
    for line in lines:
        yd = line.get_ydata()
        if yd is not None and len(yd) > 0:
            all_y.extend(yd)
    for coll in collections:
        offsets = coll.get_offsets()
        if len(offsets) > 0:
            all_y.extend(offsets[:, 1])
    if not all_y:
        return
    ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
    margin = (ymax - ymin) * pad_frac if ymax > ymin else 0.05
    ax.set_ylim(ymin - margin, ymax + margin)


def main():
    version = get_active_version()
    version_label = VERSION_DISPLAY.get(version, version)

    DATA_PATH = config.RESULTS.degradation / "per_conversation_probe_degradation.csv"
    OUT_DIR = config.RESULTS.degradation
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run 1d script first.")
        return

    df = pd.read_csv(DATA_PATH)
    n_subjects = df["subject"].nunique()
    n_convs = df.groupby(["subject", "trial"]).ngroups
    n_rows = len(df)
    print(f"Loaded {n_rows} rows, {n_subjects} subjects, {n_convs} conversations")

    # Get peak layer info for display
    peak_info = {}
    for pt in ["reading", "control"]:
        for lm in ["peak", "fixed"]:
            col = f"{pt}_{lm}_layer"
            if col in df.columns:
                layers = df[df["turn"] == 5][col].dropna()
                if len(layers) > 0:
                    peak_info[f"{pt}_{lm}"] = int(layers.mode().iloc[0])

    # ================================================================
    # Figure 1: Probe confidence trajectories
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"Probe Confidence Across Turns — {version_label}", fontsize=14, fontweight="bold")

    for col_idx, probe_type in enumerate(["reading", "control"]):
        for row_idx, layer_mode in enumerate(["peak", "fixed"]):
            ax = axes[row_idx, col_idx]
            conf_col = f"{probe_type}_{layer_mode}_confidence"
            if conf_col not in df.columns:
                ax.set_title(f"{probe_type} ({layer_mode}) — no data")
                continue

            for condition, color, label in [("human", "#2196F3", "Human-labeled"),
                                             ("ai", "#F44336", "AI-labeled")]:
                sub = df[df["partner_type"] == condition]
                means = sub.groupby("turn")[conf_col].mean()
                sems = sub.groupby("turn")[conf_col].sem()
                ax.errorbar(means.index, means.values, yerr=sems.values,
                           marker="o", color=color, label=label, capsize=3, linewidth=2)

            ax.axhline(0.5, ls="--", color="gray", alpha=0.5, label="Chance (0.5)")
            ax.set_xlabel("Turn")
            ax.set_ylabel("Probe confidence (sigmoid output)")
            layer_num = peak_info.get(f"{probe_type}_{layer_mode}", "?")
            ax.set_title(f"{probe_type.title()} probe, {layer_mode} layer (L{layer_num})")
            ax.legend(fontsize=9)
            padded_ylim(ax, pad_frac=0.2)
            ax.set_xticks([1, 2, 3, 4, 5])

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_b64 = fig_to_base64(fig)

    # ================================================================
    # Per-conversation degradation scores
    # ================================================================
    trial_deg = df.groupby(["subject", "trial", "partner_type"], group_keys=False).apply(
        compute_degradation_score
    ).reset_index()
    print(f"\nDegradation scores computed for {len(trial_deg)} conversations")
    print(trial_deg.describe())

    # ================================================================
    # Figure 2: Classification accuracy by condition
    # ================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))
    fig2.suptitle(f"Probe Classification Accuracy by Turn — {version_label}", fontsize=14, fontweight="bold")

    for idx, probe_type in enumerate(["reading", "control"]):
        ax = axes2[idx]
        correct_col = f"{probe_type}_peak_correct"
        if correct_col not in df.columns:
            continue

        for condition, color, label in [("human", "#2196F3", "Human-labeled"),
                                         ("ai", "#F44336", "AI-labeled")]:
            sub = df[df["partner_type"] == condition]
            accs = sub.groupby("turn")[correct_col].mean()
            ax.plot(accs.index, accs.values, marker="o", color=color, label=label, linewidth=2)

        # Overall accuracy
        overall = df.groupby("turn")[correct_col].mean()
        ax.plot(overall.index, overall.values, marker="s", color="#666", ls="--",
                label="Overall", linewidth=1.5, alpha=0.7)

        ax.axhline(0.5, ls=":", color="gray", alpha=0.5, label="Chance (50%)")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Classification accuracy")
        layer_num = peak_info.get(f"{probe_type}_peak", "?")
        ax.set_title(f"{probe_type.title()} probe (peak layer L{layer_num})")
        ax.legend(fontsize=9)
        padded_ylim(ax, pad_frac=0.15)
        ax.set_xticks([1, 2, 3, 4, 5])

    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2_b64 = fig_to_base64(fig2)

    # ================================================================
    # Figure 3: Text degradation metrics by condition
    # ================================================================
    text_metrics = [
        ("sub_ttr",              "Type-Token Ratio (TTR)"),
        ("sub_trigram_rep",      "Trigram Repetition Rate"),
        ("sub_word_count",       "Word Count"),
        ("sub_self_ref_rate",    "Self-Reference Rate"),
        ("sub_allcaps_ratio",    "ALL-CAPS Ratio"),
        ("sub_exclamation_rate", "Exclamation Rate"),
        ("sub_emoji_count",      "Emoji Count"),
    ]
    # Filter to metrics that exist in the data
    text_metrics = [(m, t) for m, t in text_metrics if m in df.columns]

    n_metrics = len(text_metrics)
    n_cols = 3
    n_rows_fig = (n_metrics + n_cols - 1) // n_cols
    fig3, axes3 = plt.subplots(n_rows_fig, n_cols, figsize=(16, 4.5 * n_rows_fig))
    fig3.suptitle(f"Text Quality Metrics Across Turns — {version_label}",
                  fontsize=14, fontweight="bold")
    axes3_flat = axes3.flatten() if n_metrics > 1 else [axes3]

    condition_diff_results = []
    for idx, (metric, title) in enumerate(text_metrics):
        ax = axes3_flat[idx]

        for condition, color in [("human", "#2196F3"), ("ai", "#F44336")]:
            sub = df[df["partner_type"] == condition]
            means = sub.groupby("turn")[metric].mean()
            sems = sub.groupby("turn")[metric].sem()
            ax.errorbar(means.index, means.values, yerr=sems.values,
                       marker="o", color=color, label=condition.title(), capsize=3, linewidth=2)

        ax.set_xlabel("Turn")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks([1, 2, 3, 4, 5])
        padded_ylim(ax, pad_frac=0.2)

        # Test condition differences at turns 1 and 5
        for turn in [1, 5]:
            h = df[(df["partner_type"] == "human") & (df["turn"] == turn)][metric].dropna()
            a = df[(df["partner_type"] == "ai") & (df["turn"] == turn)][metric].dropna()
            if len(h) > 5 and len(a) > 5:
                t_stat, p_val = stats.ttest_ind(h, a)
                condition_diff_results.append({
                    "metric": title, "metric_col": metric, "turn": turn,
                    "human_mean": h.mean(), "ai_mean": a.mean(),
                    "diff": h.mean() - a.mean(),
                    "t": t_stat, "p": p_val,
                })

    # Hide unused axes
    for idx in range(n_metrics, len(axes3_flat)):
        axes3_flat[idx].set_visible(False)

    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3_b64 = fig_to_base64(fig3)

    # ================================================================
    # Figure 4: Degradation vs probe confidence scatter
    # ================================================================
    turn5 = df[df["turn"] == 5].copy()
    turn5_merged = turn5.merge(trial_deg, on=["subject", "trial", "partner_type"], how="inner")

    deg_metrics = [
        ("ttr_drop", "TTR Drop (Turn 1 to 5)"),
        ("trigram_increase", "Trigram Rep. Increase (Turn 1 to 5)"),
    ]
    probe_cols = [
        ("reading_peak_confidence", "Reading Probe Confidence"),
        ("control_peak_confidence", "Control Probe Confidence"),
    ]

    fig4, axes4 = plt.subplots(len(deg_metrics), len(probe_cols), figsize=(14, 5.5 * len(deg_metrics)))
    fig4.suptitle(f"Text Degradation vs Probe Confidence at Turn 5 — {version_label}",
                  fontsize=14, fontweight="bold")
    if len(deg_metrics) == 1:
        axes4 = axes4.reshape(1, -1)

    correlation_results = []
    for i, (deg_col, deg_label) in enumerate(deg_metrics):
        for j, (probe_col, probe_label) in enumerate(probe_cols):
            ax = axes4[i, j]
            if deg_col not in turn5_merged.columns or probe_col not in turn5_merged.columns:
                ax.set_title("No data")
                continue

            valid = turn5_merged[[deg_col, probe_col, "partner_type"]].dropna()
            if len(valid) < 10:
                ax.set_title(f"{deg_label} vs {probe_label} — too few points")
                continue

            for condition, color, marker in [("human", "#2196F3", "o"), ("ai", "#F44336", "^")]:
                sub = valid[valid["partner_type"] == condition]
                ax.scatter(sub[deg_col], sub[probe_col], c=color, marker=marker,
                          alpha=0.3, s=20, label=f"{condition.title()} (n={len(sub)})")

            # Overall correlation
            r, p = stats.pearsonr(valid[deg_col], valid[probe_col])
            sig_str = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            ax.set_xlabel(deg_label)
            ax.set_ylabel(probe_label)
            ax.set_title(f"r = {r:.3f}, p = {p:.4f} ({sig_str})", fontsize=11)
            ax.legend(fontsize=8)
            padded_ylim(ax, pad_frac=0.1)
            correlation_results.append({
                "degradation_metric": deg_label,
                "probe": probe_label,
                "condition": "all",
                "r": r, "p": p, "n": len(valid),
            })

            # Per-condition correlations
            for condition in ["human", "ai"]:
                sub = valid[valid["partner_type"] == condition]
                if len(sub) >= 10:
                    rc, pc = stats.pearsonr(sub[deg_col], sub[probe_col])
                    correlation_results.append({
                        "degradation_metric": deg_label,
                        "probe": probe_label,
                        "condition": condition,
                        "r": rc, "p": pc, "n": len(sub),
                    })

    fig4.tight_layout(rect=[0, 0, 1, 0.94])
    fig4_b64 = fig_to_base64(fig4)

    # ================================================================
    # Build HTML tables
    # ================================================================
    corr_html = '<table><tr><th>Text Metric</th><th>Probe Type</th><th>Condition</th><th>r</th><th>p</th><th>Sig.</th><th>n</th></tr>'
    for cr in correlation_results:
        sig = "***" if cr["p"] < 0.001 else ("**" if cr["p"] < 0.01 else ("*" if cr["p"] < 0.05 else ""))
        corr_html += (f'<tr><td>{cr["degradation_metric"]}</td><td>{cr["probe"]}</td>'
                      f'<td>{cr["condition"]}</td><td>{cr["r"]:.3f}</td>'
                      f'<td>{cr["p"]:.4f}</td><td>{sig}</td><td>{cr["n"]}</td></tr>')
    corr_html += "</table>"

    cond_html = '<table><tr><th>Metric</th><th>Turn</th><th>Human Mean</th><th>AI Mean</th><th>Diff (H&minus;AI)</th><th>t</th><th>p</th><th>Sig.</th></tr>'
    for cr in condition_diff_results:
        sig = "***" if cr["p"] < 0.001 else ("**" if cr["p"] < 0.01 else ("*" if cr["p"] < 0.05 else ""))
        cond_html += (f'<tr><td>{cr["metric"]}</td><td>{cr["turn"]}</td>'
                      f'<td>{cr["human_mean"]:.4f}</td><td>{cr["ai_mean"]:.4f}</td>'
                      f'<td>{cr["diff"]:+.4f}</td>'
                      f'<td>{cr["t"]:.2f}</td><td>{cr["p"]:.4f}</td><td>{sig}</td></tr>')
    cond_html += "</table>"

    # Accuracy summary table
    acc_html = '<table><tr><th>Probe</th><th>Turn</th><th>Overall Acc.</th><th>Human Acc.</th><th>AI Acc.</th></tr>'
    for pt in ["reading", "control"]:
        col = f"{pt}_peak_correct"
        if col not in df.columns:
            continue
        for turn in [1, 2, 3, 4, 5]:
            t_df = df[df["turn"] == turn]
            overall = t_df[col].mean()
            h_acc = t_df[t_df["partner_type"] == "human"][col].mean()
            a_acc = t_df[t_df["partner_type"] == "ai"][col].mean()
            acc_html += (f'<tr><td>{pt.title()}</td><td>{turn}</td>'
                         f'<td>{overall:.3f}</td><td>{h_acc:.3f}</td><td>{a_acc:.3f}</td></tr>')
    acc_html += "</table>"

    # ================================================================
    # Generate HTML report
    # ================================================================
    prompt_text = VERSION_PROMPTS.get(version, "")
    version_desc = VERSION_DESCRIPTIONS.get(version, "")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Degradation Analysis: {version_label}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px 30px; color: #2c3e50; line-height: 1.6; }}
  h1 {{ color: #1a237e; border-bottom: 3px solid #3f51b5; padding-bottom: 10px; }}
  h2 {{ color: #283593; border-bottom: 2px solid #c5cae9; padding-bottom: 8px; margin-top: 40px; }}
  h3 {{ color: #3949ab; margin-top: 25px; }}
  img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; margin: 10px 0; }}
  table {{ border-collapse: collapse; margin: 10px 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: center; }}
  th {{ background: #e8eaf6; font-weight: 600; }}
  td:first-child {{ text-align: left; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .version-box {{ background: #e8eaf6; border: 2px solid #5c6bc0; border-radius: 8px;
                  padding: 15px 20px; margin: 15px 0; }}
  .version-box h3 {{ margin-top: 5px; color: #283593; }}
  .version-box code {{ background: #c5cae9; padding: 2px 6px; border-radius: 3px;
                       font-family: 'SF Mono', Monaco, monospace; font-size: 0.92em; }}
  .key-finding {{ background: #fff3e0; padding: 12px 16px; border-left: 4px solid #ff9800;
                  margin: 10px 0; border-radius: 0 4px 4px 0; }}
  .method-box {{ background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 6px;
                 padding: 12px 16px; margin: 10px 0; font-size: 0.92em; }}
  .term {{ font-weight: 700; color: #1a237e; }}
  dl {{ margin: 10px 0; }}
  dt {{ font-weight: 700; color: #283593; margin-top: 8px; }}
  dd {{ margin-left: 20px; margin-bottom: 4px; }}
  .sig-note {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
</style></head><body>

<h1>Degradation &amp; Probe Confidence Analysis: {version_label}</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} &mdash;
Version: <code>{version}</code> &mdash;
{n_subjects} subjects &times; {n_convs // n_subjects} conversations = {n_convs} conversations &times; 5 turns = {n_rows} observations</p>

<div class="version-box">
<h3>Approach: {version_label}</h3>
<p>{version_desc}</p>
<p><strong>System prompt (key sentence):</strong> <code>&ldquo;{prompt_text}&rdquo;</code></p>
<p><strong>Conditions:</strong> Each subject converses with partners labeled as
&ldquo;a Human&rdquo; (20 conversations) and &ldquo;an AI&rdquo; (20 conversations).
In reality, all partners are identical LLaMA-2-13B-Chat instances with a generic system prompt
and no identity information.</p>
</div>

<h2>Overview</h2>

<p>This report analyzes two related questions about the {version_label} version of Experiment 1:</p>
<ol>
<li><strong>Text quality degradation:</strong> Does the participant LLM&rsquo;s output quality
    change across the 5 turns of each conversation? Does degradation differ between
    human-labeled and AI-labeled conditions?</li>
<li><strong>Probe confidence trajectories:</strong> Linear probes trained to classify partner type
    (human vs. AI) from the model&rsquo;s hidden states &mdash; does their confidence change across
    turns? Does it track text degradation (Bayesian updating) or decline uniformly (prompt dilution)?</li>
</ol>

<h2>Definitions</h2>

<h3>Text quality metrics</h3>
<p>Computed per utterance from the participant LLM&rsquo;s raw text output at each turn.
All metrics are extracted by <code>1d_degradation_probe_correlation.py</code> using
simple whitespace tokenization (<code>text.split()</code>).</p>

<dl>
<dt>Type-Token Ratio (TTR)</dt>
<dd>Number of unique words (lowercased) divided by total word count. Measures
    lexical diversity. Higher = more varied vocabulary. A drop across turns indicates
    the model is becoming more repetitive.</dd>

<dt>Trigram Repetition Rate</dt>
<dd>For all consecutive 3-word sequences in the utterance, count how many are
    repeated (excess count / total trigrams). Higher = more repeated phrases.
    An increase across turns suggests the model is falling into loops.</dd>

<dt>Word Count</dt>
<dd>Total number of whitespace-delimited tokens per utterance.</dd>

<dt>Self-Reference Rate</dt>
<dd>Proportion of words that are first-person pronouns (I, me, my, mine, myself)
    out of total word count. Captures how self-focused the model&rsquo;s language is.</dd>

<dt>ALL-CAPS Ratio</dt>
<dd>Proportion of multi-character words written entirely in uppercase (e.g., &ldquo;AMAZING&rdquo;).
    Can indicate emphatic or degraded output.</dd>

<dt>Exclamation Rate</dt>
<dd>Number of exclamation marks divided by total word count. Captures expressive
    or emphatic style.</dd>

<dt>Emoji Count</dt>
<dd>Number of Unicode emoji characters detected via regex pattern matching.</dd>
</dl>

<h3>Degradation scores (per conversation)</h3>
<p>Computed once per conversation (not per turn). Each measures the change from
turn 1 to turn 5 of that conversation:</p>
<dl>
<dt>TTR Drop</dt>
<dd><code>(TTR_turn1 &minus; TTR_turn5) / TTR_turn1</code>. Proportional decrease in lexical
    diversity. Positive = vocabulary got more repetitive.</dd>

<dt>Trigram Repetition Increase</dt>
<dd><code>trigram_rep_turn5 &minus; trigram_rep_turn1</code>. Absolute increase in phrase repetition.
    Positive = more repeated phrases by end of conversation.</dd>
</dl>

<h3>Probe types</h3>
<dl>
<dt>Reading probe</dt>
<dd>A linear classifier trained on the model&rsquo;s hidden states extracted <em>after</em> appending a
    reflective suffix: <code>&ldquo;I think the conversation partner of this user is&rdquo;</code>.
    This elicits the model&rsquo;s representation of partner identity at the point of generation.
    Probes are trained separately for each turn (1&ndash;5) at all 41 layers.</dd>

<dt>Control probe</dt>
<dd>A linear classifier trained on hidden states at the <em>pre-generation position</em>
    (end of the &lt;/s&gt; token after the last [/INST]), with no reflective suffix.
    This tests whether identity information is accessible without prompting the model
    to &ldquo;think about&rdquo; its partner.</dd>

<dt>Peak layer</dt>
<dd>For each turn and probe type, the layer with the highest classification accuracy
    on held-out data (identified during probe training). Each subplot uses a potentially
    different layer.</dd>

<dt>Fixed layer</dt>
<dd>A single reference layer (the turn-5 reading probe&rsquo;s peak layer) applied across all
    turns. This tests whether the <em>same</em> neural location carries identity information
    throughout the conversation, or whether the signal migrates.</dd>

<dt>Probe confidence</dt>
<dd>The sigmoid output of the linear probe (0&ndash;1). Values &gt; 0.5 predict &ldquo;human partner&rdquo;;
    values &lt; 0.5 predict &ldquo;AI partner&rdquo;. The distance from 0.5 reflects certainty.</dd>

<dt>Classification accuracy</dt>
<dd>Binary accuracy: the probe predicts &ldquo;human&rdquo; if confidence &gt; 0.5, &ldquo;AI&rdquo; if &le; 0.5.
    Compared to the ground-truth label (which condition this conversation belongs to).
    Chance = 50%.</dd>
</dl>

<h2>1. Probe Confidence Trajectories</h2>
<p>Each panel shows the mean probe confidence (&plusmn; SEM) across turns 1&ndash;5,
separately for conversations where the partner was labeled as human (blue) vs. AI (red).
The dashed gray line marks chance (0.5).</p>
<ul>
<li><strong>Top row:</strong> Peak layer &mdash; each turn uses the layer where that turn&rsquo;s
    probe achieved maximum accuracy during training.</li>
<li><strong>Bottom row:</strong> Fixed layer &mdash; all turns use the same layer
    (turn-5 reading probe peak), testing whether the signal is stable at one location.</li>
<li><strong>Left column:</strong> Reading probes (with reflective suffix).</li>
<li><strong>Right column:</strong> Control probes (no suffix).</li>
</ul>
<p><strong>What to look for:</strong> If probe confidence stays high across turns, the identity
representation persists. If it drops toward 0.5, the signal is being diluted. If human-labeled
and AI-labeled lines diverge or converge differently, that suggests condition-specific updating.</p>
<img src="data:image/png;base64,{fig1_b64}" alt="Probe confidence trajectories">

<h2>2. Classification Accuracy by Turn</h2>
<p>Binary classification accuracy of the peak-layer probe at each turn, split by condition.
The dashed gray line marks chance (50%). Each point represents accuracy across all
{n_convs // 2} conversations in that condition at that turn.</p>
<p><strong>What to look for:</strong> Near-perfect accuracy at turn 1 (probe trained on that turn&rsquo;s data)
declining across turns indicates <em>prompt dilution</em> &mdash; the system prompt tokens become a
smaller fraction of the total context as conversation history grows. If accuracy drops faster for
one condition, the model may be updating its representation based on conversational evidence.</p>
<img src="data:image/png;base64,{fig2_b64}" alt="Classification accuracy by turn">

<h3>Accuracy table</h3>
{acc_html}

<h2>3. Text Quality Across Turns</h2>
<p>Each panel shows a text quality metric averaged across all conversations (&plusmn; SEM),
split by human-labeled (blue) vs. AI-labeled (red) conditions. All metrics are computed
from the <em>participant</em> LLM&rsquo;s output (not the partner&rsquo;s).</p>
<p><strong>What to look for:</strong> Declining TTR or increasing trigram repetition across turns
indicates the model is becoming more formulaic as context length grows. Condition differences
would suggest the model&rsquo;s text quality is modulated by its belief about its partner.</p>
<img src="data:image/png;base64,{fig3_b64}" alt="Text quality metrics across turns">

<h3>Condition differences (independent-samples t-test at turns 1 and 5)</h3>
<p class="sig-note">Tests whether human-labeled and AI-labeled conversations differ on each
metric at the first and last turn. Independent-samples t-test (not paired, since different
conversations contribute to each condition). * p &lt; .05, ** p &lt; .01, *** p &lt; .001.</p>
{cond_html}

<h2>4. Text Degradation vs. Probe Confidence (Turn 5)</h2>
<p>Each scatter plot shows, for every conversation, the relationship between how much the
participant&rsquo;s text quality changed from turn 1 to turn 5 (x-axis) and the probe&rsquo;s
confidence at turn 5 (y-axis). Blue circles = human-labeled, red triangles = AI-labeled.</p>
<ul>
<li><strong>TTR Drop:</strong> Proportional decrease in vocabulary diversity from turn 1 to turn 5.
    Positive values mean the text got more repetitive.</li>
<li><strong>Trigram Rep. Increase:</strong> Absolute increase in repeated 3-word phrases from
    turn 1 to turn 5. Positive values mean more phrase loops.</li>
</ul>
<p><strong>What to look for:</strong> A significant positive correlation between text degradation and
probe confidence would support <em>Bayesian updating</em> &mdash; conversations where the
participant &ldquo;fell apart&rdquo; more also lost probe signal faster. No correlation supports
<em>prompt dilution</em> (confidence decline is uniform regardless of text quality).</p>
<img src="data:image/png;base64,{fig4_b64}" alt="Degradation vs probe confidence scatter">

<h3>Pearson correlations</h3>
<p class="sig-note">Pearson r between per-conversation degradation score and turn-5 probe confidence.
Computed overall and separately per condition. * p &lt; .05, ** p &lt; .01, *** p &lt; .001.</p>
{corr_html}

<h2>5. Interpretation</h2>
<div class="key-finding">
<strong>Bayesian updating hypothesis:</strong> The model maintains a &ldquo;live&rdquo; representation
of its partner that updates based on conversational evidence. Predictions: (1) human-labeled
conversations should show faster probe decline (partner is actually another LLM and fails to
act convincingly human), (2) text degradation should correlate with probe confidence loss
(conversations that degrade more also lose identity signal faster).
</div>
<div class="key-finding">
<strong>Prompt dilution hypothesis:</strong> The probe simply reads the system prompt tokens, and
as the conversation grows longer, those tokens become a smaller fraction of the total context.
Predictions: (1) both conditions degrade at similar rates, (2) no correlation between text quality
and probe confidence &mdash; the probe just reads a fading prompt signal, regardless of what
the conversation actually contains.
</div>

</body></html>"""

    from src.report_utils import save_report
    html_path = OUT_DIR / "degradation_probe_report.html"
    save_report(html, html_path)

    # Save CSV results
    pd.DataFrame(correlation_results).to_csv(OUT_DIR / "correlation_results.csv", index=False)
    pd.DataFrame(condition_diff_results).to_csv(OUT_DIR / "condition_diff_results.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze degradation-probe correlation results")
    add_version_argument(parser)
    add_model_argument(parser)
    args = parser.parse_args()
    set_model(args.model)
    set_version(args.version)
    main()
