#!/usr/bin/env python3
"""
Script name: sentiment_vader.py
Purpose: Analyze sentiment in participant transcripts using VADER sentiment analysis.
    - Reads from the combined transcript CSV created by combine_text_data.py.
    - Computes sentiment scores (compound, pos, neg, neu).
    - Saves per-trial and per-subject averages.
    - Compares Human vs Bot sentiment (paired t-tests).
    - Saves results (CSVs, plots, stats) to results/<model>/<temperature>/sentiment_vader/.

Inputs:
    - Combined transcript CSV: data/<model>/<temperature>/combined_text_data.csv

Outputs:
    - sentiment_by_interaction.csv
    - sentiment_subject_summary.csv
    - sentiment_stats.txt
    - sentiment_barplot.png
    - sentiment_violinplot.png

Usage:
    python code/analysis/sentiment_vader.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-31
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from nltk.sentiment import SentimentIntensityAnalyzer

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_info, print_warn
from utils.plot_helpers import barplot_with_lines, main_effect_violin_lines, DEFAULT_PALETTE

SCRIPT_NAME = "sentiment_vader"

# ------------------------------------------------------------
# Sentiment Analyzer
# ------------------------------------------------------------
sia = SentimentIntensityAnalyzer()


def compute_sentiment(text: str) -> dict[str, float]:
    """Compute sentiment scores (compound, pos, neg, neu) using VADER."""
    if pd.isna(text) or not isinstance(text, str):
        return {"compound": np.nan, "pos": np.nan, "neg": np.nan, "neu": np.nan}
    return sia.polarity_scores(text)


# ------------------------------------------------------------
# Core Analysis
# ------------------------------------------------------------
def analyze_sentiment(combined_path: str, out_dir: str):
    """Compute sentiment and compare Human vs Bot conditions."""
    print_header("1) Sentiment Analysis — Humans vs Bots")

    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined CSV not found: {combined_path}")

    df = pd.read_csv(combined_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    required = {"subject", "agent", "transcript_sub"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Combined CSV missing required columns: {missing}")

    # Derive Condition and Subject columns
    df["Condition"] = df["agent"].astype(str).str.extract(r"(hum|bot)", expand=False)
    df["Subject"] = df["subject"].astype(str)

    # ------------------------------------------------------------
    # Compute sentiment scores
    # ------------------------------------------------------------
    print_info("Computing VADER sentiment scores...")
    sent_scores = df["transcript_sub"].apply(compute_sentiment)
    sent_df = pd.DataFrame(list(sent_scores))
    df = pd.concat([df, sent_df], axis=1)

    # ------------------------------------------------------------
    # Save trial-level data
    # ------------------------------------------------------------
    out_interactions = os.path.join(out_dir, "sentiment_by_interaction.csv")
    cols = [
        "Subject", "agent", "Condition", "topic",
        "transcript_sub", "compound", "pos", "neg", "neu"
    ]
    df[cols].to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV (per-interaction sentiment scores)")

    # ------------------------------------------------------------
    # Subject-level summary (compound)
    # ------------------------------------------------------------
    summary = df.groupby(["Subject", "Condition"])["compound"].mean().unstack().reset_index()
    out_summary = os.path.join(out_dir, "sentiment_subject_summary.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV (subject-level compound means)")

    # ------------------------------------------------------------
    # Paired t-test (compound)
    # ------------------------------------------------------------
    p_val_main = None
    if "hum" in summary.columns and "bot" in summary.columns:
        x, y = paired_clean(summary["hum"], summary["bot"])
        if x.size > 0:
            t_stat, p_val_main = ttest_rel(x, y, nan_policy="omit")

            mean_hum, mean_bot = x.mean(), y.mean()
            sem_hum = x.std(ddof=1) / np.sqrt(len(x))
            sem_bot = y.std(ddof=1) / np.sqrt(len(x))
            diff = x - y
            cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

            lines = [
                "--- Sentiment Analysis (compound score) ---",
                "Compound sentiment ranges from -1 (negative) to +1 (positive).",
                f"N = {len(x)} paired subjects",
                f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
                f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
                f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.3f}",
                f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val_main:.4f}",
                f"Cohen's d = {cohens_d:.3f}",
                "Interpretation: "
                + ("Significant (p < .05)" if p_val_main < 0.05 else "Not significant (p > .05)")
            ]

            stats_path = os.path.join(out_dir, "sentiment_stats.txt")
            with open(stats_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print_save(stats_path, kind="stats")

    # ------------------------------------------------------------
    # Plots (bar + violin)
    # ------------------------------------------------------------
    long_df = summary.rename(columns={"hum": "Hum", "bot": "Bot"}).melt(
        id_vars="Subject", var_name="Condition", value_name="compound"
    )

    # Bar plot
    barplot_with_lines(
        df_long=long_df,
        x_col="Condition",
        y_col="compound",
        out_path=os.path.join(out_dir, "sentiment_barplot.png"),
        title="Average Sentiment by Condition (Human vs Bot)",
        palette={"Hum": "steelblue", "Bot": "sandybrown"},
        p_val=p_val_main,
    )

    # Violin plot
    main_effect_violin_lines(
        df_summary=summary.rename(columns={"hum": "Hum", "bot": "Bot"}),
        cond_a="Hum",
        cond_b="Bot",
        y_col="compound",
        out_path=os.path.join(out_dir, "sentiment_violinplot.png"),
        title="Sentiment per Subject (Human vs Bot)",
        palette=DEFAULT_PALETTE,
        p_val=p_val_main,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args, cfg = parse_and_load_config("Sentiment VADER analysis")
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    combined_path = os.path.join(data_dir, "combined_text_data.csv")
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "sentiment_vader")
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_sentiment(combined_path, out_dir)

    logger.info("✅ Sentiment VADER analysis complete.")
    print("\n[DONE] ✅ Sentiment VADER analysis complete.\n")


if __name__ == "__main__":
    main()
