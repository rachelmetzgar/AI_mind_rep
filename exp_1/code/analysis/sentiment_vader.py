#!/usr/bin/env python3
"""
Script name: sentiment_vader.py
Purpose: Analyze sentiment in participant transcripts using VADER sentiment analysis.
    - Compute sentiment scores (compound, pos, neg, neu).
    - Save per-trial and per-subject averages.
    - Compare Human vs Bot conditions with paired t-tests.
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/sentiment/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv

Outputs:
    - Trial-level and summary CSVs.
    - Statistical output text files.
    - Figures (bar/violin plots).
    - Run log + config snapshot.

Usage:
    python code/analysis/sentiment.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from nltk.sentiment import SentimentIntensityAnalyzer

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_warn

SCRIPT_NAME = "sentiment_vader"

# -------------------------------
# Sentiment Analyzer
# -------------------------------
sia = SentimentIntensityAnalyzer()

def compute_sentiment(text: str) -> dict[str, float]:
    """Compute sentiment scores (compound, pos, neg, neu) using VADER."""
    if pd.isna(text) or not isinstance(text, str):
        return {"compound": np.nan, "pos": np.nan, "neg": np.nan, "neu": np.nan}
    return sia.polarity_scores(text)


# -------------------------------
# Analysis
# -------------------------------

def analyze_sentiment(data_dir: str, out_dir: str):
    """Compute sentiment and compare Human vs Bot conditions."""
    print_header("1) Sentiment Analysis — Humans vs Bots")

    trials = []
    for subfile in sorted(os.listdir(data_dir)):
        if not subfile.endswith(".csv"):
            continue
        sub_id = os.path.splitext(subfile)[0]
        csv_path = os.path.join(data_dir, subfile)

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        if "agent" not in df.columns or "transcript_sub" not in df.columns:
            print_warn(f"{sub_id}: Missing required columns; skipping.")
            continue

        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["Subject"] = sub_id

        # Compute sentiment scores
        sent_scores = df["transcript_sub"].apply(compute_sentiment)
        sent_df = pd.DataFrame(list(sent_scores))
        df = pd.concat([df, sent_df], axis=1)
        trials.append(df)

    if not trials:
        print_warn("No transcript CSVs found — analysis aborted.")
        return

    df = pd.concat(trials, ignore_index=True)

    # Save trial-level data
    trial_out = os.path.join(out_dir, "sentiment_triallevel.csv")
    df.to_csv(trial_out, index=False)
    print_save(trial_out, kind="CSV")

    # Subject-level summary (compound score)
    summary = df.groupby(["Subject", "Condition"])["compound"].mean().unstack().reset_index()
    out_summary = os.path.join(out_dir, "sentiment_summary_all_subjects.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV")

    # Paired t-test (compound)
    if "hum" in summary and "bot" in summary:
        x, y = paired_clean(summary["hum"], summary["bot"])
        t_stat, p_val = ttest_rel(x, y, nan_policy="omit")

        mean_hum, mean_bot = x.mean(), y.mean()
        sem_hum = x.std(ddof=1) / np.sqrt(len(x))
        sem_bot = y.std(ddof=1) / np.sqrt(len(y))
        diff = x - y
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

        lines = [
            "--- Sentiment Analysis (compound score) ---",
            "Compound sentiment ranges from -1 (negative) to +1 (positive).",
            f"N = {len(x)} paired subjects",
            f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
            f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
            f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.3f}",
            f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val:.4f}",
            f"Cohen's d = {cohens_d:.3f}",
            "Interpretation: " + (
                "Significant difference (p < .05)"
                if p_val < 0.05 else
                "No significant difference (p > .05)"
            )
        ]

        stats_path = os.path.join(out_dir, "sentiment_stats.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print_save(stats_path, kind="stats")

    # Violin plot
    long_df = summary.melt(id_vars="Subject", var_name="Condition", value_name="compound")
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=long_df, x="Condition", y="compound",
        palette={"hum": "skyblue", "bot": "sandybrown"}, inner="box", cut=0
    )
    for _, row in summary.iterrows():
        plt.plot(["hum", "bot"], [row["hum"], row["bot"]], color="gray", alpha=0.4)
    plt.ylabel("Avg. Compound Sentiment (per subject)")
    plt.title("Sentiment by Condition (Human vs Bot)")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "sentiment_violinplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Sentiment analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "sentiment_vader")  # ✅ results/model/temp/sentiment
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_sentiment(data_dir, out_dir)

    logger.info("✅ Sentiment analysis complete.")
    print("\n[DONE] ✅ Sentiment analysis complete.\n")


if __name__ == "__main__":
    main()
