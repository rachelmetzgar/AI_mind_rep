#!/usr/bin/env python3
"""
Analyze degradation–probe correlation results.

Tests the Bayesian updating hypothesis: do conversations with more behavioral
degradation show faster decline in probe confidence?

Reads per_conversation_probe_degradation.csv produced by 1d script.
Produces figures and statistical summaries.

Usage:
    python 1e_analyze_degradation_results.py --version labels

Env: behavior_env (no GPU needed)
Rachel C. Metzgar · Feb 2026
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
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_version, add_version_argument


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

    # TTR drop
    ttr_vals = turns["sub_ttr"].values
    if len(ttr_vals) >= 2 and ttr_vals[0] > 0:
        scores["ttr_drop"] = (ttr_vals[0] - ttr_vals[-1]) / ttr_vals[0]
    else:
        scores["ttr_drop"] = np.nan

    # Trigram repetition increase
    tri_vals = turns["sub_trigram_rep"].values
    scores["trigram_increase"] = tri_vals[-1] - tri_vals[0] if len(tri_vals) >= 2 else np.nan

    # ALL-CAPS increase
    caps_vals = turns["sub_allcaps_ratio"].values
    scores["allcaps_increase"] = caps_vals[-1] - caps_vals[0] if len(caps_vals) >= 2 else np.nan

    # Exclamation increase
    excl_vals = turns["sub_exclamation_rate"].values
    scores["excl_increase"] = excl_vals[-1] - excl_vals[0] if len(excl_vals) >= 2 else np.nan

    # Composite degradation: average of z-scored individual metrics
    # (computed later when we have all trials)

    return pd.Series(scores)


def main():
    DATA_PATH = config.RESULTS.degradation / "per_conversation_probe_degradation.csv"
    OUT_DIR = config.RESULTS.degradation
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run 1d script first.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects, "
          f"{df.groupby(['subject','trial']).ngroups} conversations")

    # ================================================================
    # 1. Probe confidence trajectories by condition
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Probe Confidence Across Turns by Condition", fontsize=14)

    for col_idx, probe_type in enumerate(["reading", "control"]):
        for row_idx, layer_mode in enumerate(["peak", "fixed"]):
            ax = axes[row_idx, col_idx]
            conf_col = f"{probe_type}_{layer_mode}_confidence"
            if conf_col not in df.columns:
                ax.set_title(f"{probe_type} ({layer_mode}) — no data")
                continue

            for condition, color, label in [("human", "#2196F3", "Human condition"),
                                             ("ai", "#F44336", "AI condition")]:
                sub = df[df["partner_type"] == condition]
                means = sub.groupby("turn")[conf_col].mean()
                sems = sub.groupby("turn")[conf_col].sem()
                ax.errorbar(means.index, means.values, yerr=sems.values,
                           marker="o", color=color, label=label, capsize=3)

            ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Probe Confidence (sigmoid)")
            ax.set_title(f"{probe_type.title()} Probe ({layer_mode} layer)")
            ax.legend()
            ax.set_ylim(0.3, 0.8)
            ax.set_xticks([1, 2, 3, 4, 5])

    fig1_b64 = fig_to_base64(fig)

    # ================================================================
    # 2. Per-conversation degradation scores
    # ================================================================
    trial_deg = df.groupby(["subject", "trial", "partner_type"]).apply(
        compute_degradation_score
    ).reset_index()
    print(f"\nDegradation scores computed for {len(trial_deg)} conversations")
    print(trial_deg.describe())

    # ================================================================
    # 3. Probe accuracy by condition across turns
    # ================================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Probe Classification Accuracy by Condition and Turn", fontsize=14)

    for idx, probe_type in enumerate(["reading", "control"]):
        ax = axes2[idx]
        correct_col = f"{probe_type}_peak_correct"
        if correct_col not in df.columns:
            continue

        for condition, color, label in [("human", "#2196F3", "Human partner"),
                                         ("ai", "#F44336", "AI partner")]:
            sub = df[df["partner_type"] == condition]
            accs = sub.groupby("turn")[correct_col].mean()
            ax.plot(accs.index, accs.values, marker="o", color=color, label=label)

        ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Classification Accuracy")
        ax.set_title(f"{probe_type.title()} Probe (peak layer)")
        ax.legend()
        ax.set_ylim(0.4, 1.05)
        ax.set_xticks([1, 2, 3, 4, 5])

    fig2_b64 = fig_to_base64(fig2)

    # ================================================================
    # 4. Correlation: degradation score vs probe confidence at turn 5
    # ================================================================
    # Merge degradation scores with turn-5 probe data
    turn5 = df[df["turn"] == 5].copy()
    turn5_merged = turn5.merge(trial_deg, on=["subject", "trial", "partner_type"], how="inner")

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle("Degradation Score vs Probe Confidence at Turn 5", fontsize=14)

    deg_metrics = ["ttr_drop", "trigram_increase"]
    probe_cols = ["reading_peak_confidence", "control_peak_confidence"]
    correlation_results = []

    for i, deg_metric in enumerate(deg_metrics):
        for j, probe_col in enumerate(probe_cols):
            ax = axes3[i, j]
            if deg_metric not in turn5_merged.columns or probe_col not in turn5_merged.columns:
                ax.set_title("No data")
                continue

            valid = turn5_merged[[deg_metric, probe_col, "partner_type"]].dropna()
            if len(valid) < 10:
                ax.set_title(f"{deg_metric} vs {probe_col} — too few points")
                continue

            for condition, color, marker in [("human", "#2196F3", "o"), ("ai", "#F44336", "^")]:
                sub = valid[valid["partner_type"] == condition]
                ax.scatter(sub[deg_metric], sub[probe_col], c=color, marker=marker,
                          alpha=0.3, s=20, label=f"{condition} (n={len(sub)})")

            # Overall correlation
            r, p = stats.pearsonr(valid[deg_metric], valid[probe_col])
            ax.set_xlabel(deg_metric.replace("_", " ").title())
            ax.set_ylabel(probe_col.replace("_", " ").title())
            ax.set_title(f"r={r:.3f}, p={p:.4f}")
            ax.legend(fontsize=8)
            correlation_results.append({
                "degradation_metric": deg_metric,
                "probe": probe_col,
                "r": r, "p": p, "n": len(valid),
            })

            # Per-condition correlations
            for condition in ["human", "ai"]:
                sub = valid[valid["partner_type"] == condition]
                if len(sub) >= 10:
                    rc, pc = stats.pearsonr(sub[deg_metric], sub[probe_col])
                    correlation_results.append({
                        "degradation_metric": deg_metric,
                        "probe": probe_col,
                        "condition": condition,
                        "r": rc, "p": pc, "n": len(sub),
                    })

    fig3_b64 = fig_to_base64(fig3)

    # ================================================================
    # 5. Key test: does the human condition degrade faster?
    # ================================================================
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
    fig4.suptitle("Text Degradation by Condition Across Turns", fontsize=14)

    text_metrics = [
        ("sub_ttr", "Type-Token Ratio"),
        ("sub_trigram_rep", "Trigram Repetition Rate"),
        ("sub_allcaps_ratio", "ALL-CAPS Ratio"),
        ("sub_exclamation_rate", "Exclamation Rate"),
    ]

    condition_diff_results = []
    for idx, (metric, title) in enumerate(text_metrics):
        ax = axes4[idx // 2, idx % 2]
        if metric not in df.columns:
            continue

        for condition, color in [("human", "#2196F3"), ("ai", "#F44336")]:
            sub = df[df["partner_type"] == condition]
            means = sub.groupby("turn")[metric].mean()
            sems = sub.groupby("turn")[metric].sem()
            ax.errorbar(means.index, means.values, yerr=sems.values,
                       marker="o", color=color, label=condition.title(), capsize=3)

        ax.set_xlabel("Turn")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5])

        # Test: does condition affect degradation rate? (turn x condition interaction)
        for turn in [1, 5]:
            h = df[(df["partner_type"] == "human") & (df["turn"] == turn)][metric].dropna()
            a = df[(df["partner_type"] == "ai") & (df["turn"] == turn)][metric].dropna()
            if len(h) > 5 and len(a) > 5:
                t, p = stats.ttest_ind(h, a)
                condition_diff_results.append({
                    "metric": metric, "turn": turn,
                    "human_mean": h.mean(), "ai_mean": a.mean(),
                    "t": t, "p": p,
                })

    fig4_b64 = fig_to_base64(fig4)

    # ================================================================
    # 6. Generate HTML report
    # ================================================================
    corr_html = "<table><tr><th>Degradation Metric</th><th>Probe</th><th>Condition</th><th>r</th><th>p</th><th>n</th></tr>"
    for cr in correlation_results:
        cond = cr.get("condition", "all")
        sig = "**" if cr["p"] < 0.01 else ("*" if cr["p"] < 0.05 else "")
        corr_html += f"<tr><td>{cr['degradation_metric']}</td><td>{cr['probe']}</td><td>{cond}</td><td>{cr['r']:.3f}</td><td>{cr['p']:.4f}{sig}</td><td>{cr['n']}</td></tr>"
    corr_html += "</table>"

    cond_html = "<table><tr><th>Metric</th><th>Turn</th><th>Human Mean</th><th>AI Mean</th><th>t</th><th>p</th></tr>"
    for cr in condition_diff_results:
        sig = "**" if cr["p"] < 0.01 else ("*" if cr["p"] < 0.05 else "")
        cond_html += f"<tr><td>{cr['metric']}</td><td>{cr['turn']}</td><td>{cr['human_mean']:.4f}</td><td>{cr['ai_mean']:.4f}</td><td>{cr['t']:.2f}</td><td>{cr['p']:.4f}{sig}</td></tr>"
    cond_html += "</table>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Degradation–Probe Correlation Analysis</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 2px solid #c5cae9; padding-bottom: 8px; }}
  img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }}
  table {{ border-collapse: collapse; margin: 10px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: center; }}
  th {{ background: #e8eaf6; }}
  .key-finding {{ background: #fff3e0; padding: 12px; border-left: 4px solid #ff9800; margin: 10px 0; }}
</style></head><body>
<h1>Degradation–Probe Correlation Analysis</h1>
<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><b>Hypothesis:</b> If the model maintains an active representation of its partner
that updates based on evidence, conversations where the partner behaves less like
the labeled type should show faster probe degradation (Bayesian updating).
Alternatively, if the probe just reads the system prompt tokens, degradation should
be uniform regardless of conversation quality.</p>

<h2>1. Probe Confidence Trajectories by Condition</h2>
<p>Does probe confidence (sigmoid output) differ between human-labeled and AI-labeled
conditions? If the model updates its representation, human-condition confidence should
drop faster (the partner doesn't really act human).</p>
<img src="data:image/png;base64,{fig1_b64}">

<h2>2. Classification Accuracy by Condition</h2>
<p>Per-conversation binary accuracy at the peak layer, split by condition.</p>
<img src="data:image/png;base64,{fig2_b64}">

<h2>3. Degradation vs Probe Confidence</h2>
<p>Scatter plots: per-trial text degradation score vs probe confidence at turn 5.</p>
<img src="data:image/png;base64,{fig3_b64}">
<h3>Correlations</h3>
{corr_html}

<h2>4. Text Degradation by Condition</h2>
<p>Do human-labeled and AI-labeled conversations degrade at different rates?</p>
<img src="data:image/png;base64,{fig4_b64}">
<h3>Condition Differences</h3>
{cond_html}

<h2>5. Interpretation Guide</h2>
<div class="key-finding">
<b>Bayesian updating support:</b> Human-condition conversations should degrade more
(partner fails to act human) AND show steeper probe confidence decline. Positive
correlation between text degradation and probe confidence drop.
</div>
<div class="key-finding">
<b>Prompt dilution support:</b> Both conditions degrade at similar rates. Probe
confidence decline is uniform. No correlation between text degradation and probe
confidence — the probe just reads the prompt, and the prompt fades equally.
</div>
</body></html>"""

    html_path = OUT_DIR / "degradation_probe_report.html"
    with open(html_path, "w") as f:
        f.write(html)
    print(f"\nHTML report: {html_path}")

    # Save correlation results
    pd.DataFrame(correlation_results).to_csv(OUT_DIR / "correlation_results.csv", index=False)
    pd.DataFrame(condition_diff_results).to_csv(OUT_DIR / "condition_diff_results.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze degradation–probe correlation results")
    add_version_argument(parser)
    args = parser.parse_args()
    set_version(args.version)
    main()
