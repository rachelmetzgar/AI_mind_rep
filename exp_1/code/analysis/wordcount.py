#!/usr/bin/env python3
"""
Script name: wordcount.py
Purpose: Analyze how many words participants used in each conversation.
    - Reads from the combined transcript CSV created by combine_text_data.py.
    - Computes word counts per participant utterance.
    - Compares Human vs Bot word counts (paired t-tests, bar + violin plots).
    - Compares Social vs Nonsocial topics.
    - Tests interaction (Condition × Sociality) via repeated-measures ANOVA.
    - Saves results (CSVs, plots, stats) to results/<model>/<temperature>/wordcount/.

Inputs:
    - combined_text_data.csv
    - topics.csv (with social/nonsocial labels)

Outputs:
    - wordcount_by_interaction.csv
    - wordcount_subject_summary.csv
    - wordcount_stats.txt
    - wordcount_barplot.png
    - wordcount_violinplot.png
    - wordcount_anova.txt

Usage:
    python code/analysis/wordcount.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-31
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_info, print_warn
from utils.plot_helpers import barplot_with_lines, plot_violin_basic, DEFAULT_PALETTE

SCRIPT_NAME = "wordcount"


# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------
def analyze_wordcount(combined_path: str, out_dir: str):
    print_header("1) Word Count Analysis — Humans vs Bots")

    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined CSV not found: {combined_path}")

    df = pd.read_csv(combined_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    required = {"subject", "agent", "topic", "transcript_sub"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Combined CSV missing required columns: {missing}")

    # Derive Condition and Subject columns
    df["Condition"] = df["agent"].astype(str).str.extract(r"(hum|bot)", expand=False)
    df["Subject"] = df["subject"].astype(str)

    # ------------------------------------------------------------
    # Compute word counts
    # ------------------------------------------------------------
    print_info("Counting words in participant utterances...")
    df["Word_Count"] = df["transcript_sub"].astype(str).apply(lambda x: len(x.split()))

    # ------------------------------------------------------------
    # Save trial-level CSV
    # ------------------------------------------------------------
    out_interactions = os.path.join(out_dir, "wordcount_by_interaction.csv")
    cols = ["Subject", "agent", "Condition", "topic", "transcript_sub", "Word_Count"]
    df[cols].to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV (per-interaction word counts)")

    # ------------------------------------------------------------
    # Per-subject summary
    # ------------------------------------------------------------
    summary = df.groupby(["Subject", "Condition"])["Word_Count"].mean().unstack().reset_index()
    out_summary = os.path.join(out_dir, "wordcount_subject_summary.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV (subject-level word counts)")

    # ------------------------------------------------------------
    # Paired t-test (Human vs Bot)
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
                "--- Paired t-test: Word Count (Hum vs Bot) ---",
                f"N = {len(x)} paired subjects",
                f"Mean (Human) = {mean_hum:.2f} ± {sem_hum:.2f} SEM",
                f"Mean (Bot)   = {mean_bot:.2f} ± {sem_bot:.2f} SEM",
                f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.2f}",
                f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val_main:.4f}",
                f"Cohen's d = {cohens_d:.3f}",
                "Interpretation: "
                + ("Significant (p < .05)" if p_val_main < 0.05 else "Not significant (p > .05)"),
            ]

            stats_path = os.path.join(out_dir, "wordcount_stats.txt")
            with open(stats_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print_save(stats_path, kind="stats")

    # ------------------------------------------------------------
    # Plots (bar + violin)
    # ------------------------------------------------------------
    long_df = summary.rename(columns={"hum": "Hum", "bot": "Bot"}).melt(
        id_vars="Subject", var_name="Condition", value_name="Word_Count"
    )

    barplot_with_lines(
        df_long=long_df,
        x_col="Condition",
        y_col="Word_Count",
        out_path=os.path.join(out_dir, "wordcount_barplot.png"),
        title="Word Count by Condition (Human vs Bot)",
        palette={"Hum": "steelblue", "Bot": "sandybrown"},
        p_val=p_val_main,
    )

    plot_violin_basic(
        df_long=long_df,
        y_col="Word_Count",
        x_col="Condition",
        out_path=os.path.join(out_dir, "wordcount_violinplot.png"),
        title="Word Count Distribution (Human vs Bot)",
        palette={"Hum": "steelblue", "Bot": "sandybrown"},
    )

    # ------------------------------------------------------------
    # Social vs Nonsocial topics
    # ------------------------------------------------------------
    analyze_social_topics(df, out_dir)

    # ------------------------------------------------------------
    # Repeated-measures ANOVA
    # ------------------------------------------------------------
    run_anova(out_dir)


# ------------------------------------------------------------
# Social vs Nonsocial topic analysis
# ------------------------------------------------------------
def analyze_social_topics(df: pd.DataFrame, out_dir: str):
    """Word counts split by Social vs Nonsocial topics."""
    print_header("2) Social vs Nonsocial — Word Counts")

    topics_path = os.path.join(PROJECT_ROOT, "data/conds/topics.csv")
    topics_df = pd.read_csv(topics_path)
    topics_df["topic"] = topics_df["topic"].str.strip()

    df = df.merge(topics_df, on="topic", how="left")
    df["topic_type"] = df["social"].map({1: "social", 0: "nonsocial"})

    out_csv = os.path.join(out_dir, "wordcount_by_social.csv")
    df.to_csv(out_csv, index=False)
    print_save(out_csv, kind="CSV (word counts by topic type)")

    # Compute Subject × Condition × TopicType means
    summary = df.groupby(["Subject", "Condition", "topic_type"])["Word_Count"].mean().reset_index()
    wide = summary.pivot(index="Subject", columns=["Condition", "topic_type"], values="Word_Count")

    # Save subject-level summary
    out_summary = os.path.join(out_dir, "wordcount_subject_summary_social.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV (subject × condition × topic summary)")

    # Violin plot
    summary["Cond_Topic"] = summary["Condition"] + "_" + summary["topic_type"]
    plot_violin_basic(
        df_long=summary,
        y_col="Word_Count",
        x_col="Cond_Topic",
        out_path=os.path.join(out_dir, "wordcount_social_violin.png"),
        title="Word Count by Condition × Topic Type",
        palette=DEFAULT_PALETTE,
    )


# ------------------------------------------------------------
# Repeated-measures ANOVA
# ------------------------------------------------------------
def run_anova(out_dir: str):
    """Repeated-measures ANOVA for word counts (Condition × Sociality)."""
    print_header("3) Repeated-Measures ANOVA — Word Counts")

    src_csv = os.path.join(out_dir, "wordcount_by_social.csv")
    if not os.path.exists(src_csv):
        print_warn(f"Missing source for ANOVA: {src_csv}")
        return

    df = pd.read_csv(src_csv)
    grouped = df.groupby(["Subject", "Condition", "social"])["Word_Count"].mean().reset_index()
    grouped["topic_type"] = grouped["social"].map({1: "social", 0: "nonsocial"})

    try:
        aov = AnovaRM(grouped, "Word_Count", "Subject", within=["Condition", "topic_type"]).fit()
        n_subs = grouped["Subject"].nunique()
        desc = grouped.groupby(["Condition", "topic_type"])["Word_Count"].agg(["mean", "sem"]).round(2)

        lines = []
        lines.append(f"[ANOVA] Word Count ANOVA: N={n_subs} participants\n")
        lines.append(aov.summary().as_text())
        lines.append("\nCell means ± SEM (avg words per interaction):")
        lines.append(desc.to_string())

        # Compute η² effect sizes
        if hasattr(aov, "anova_table") and "Sum Sq" in aov.anova_table.columns:
            anova_table = aov.anova_table.copy()
            total_ss = anova_table["Sum Sq"].sum()
            if total_ss > 0:
                anova_table["eta_sq"] = anova_table["Sum Sq"] / total_ss
                cols = [c for c in ["F Value", "Num DF", "Den DF", "Pr > F", "Sum Sq", "eta_sq"]
                        if c in anova_table.columns]
                lines.append("\nEffect sizes (η²):")
                lines.append(anova_table[cols].round(4).to_string())

        out_stats = os.path.join(out_dir, "wordcount_anova.txt")
        with open(out_stats, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print_save(out_stats, kind="stats")

        # Save means CSV
        desc.to_csv(os.path.join(out_dir, "wordcount_anova_means.csv"))
        print_save(os.path.join(out_dir, "wordcount_anova_means.csv"), kind="CSV")

    except Exception as e:
        print_warn(f"ANOVA failed: {e}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args, cfg = parse_and_load_config("Word Count analysis")
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    combined_path = os.path.join(data_dir, "combined_text_data.csv")
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "wordcount")
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir, script_name=SCRIPT_NAME, args=args, cfg=cfg, used_alias=False
    )

    analyze_wordcount(combined_path, out_dir)

    logger.info("✅ Word count analysis complete.")
    print("\n[DONE] ✅ Word count analysis complete.\n")


if __name__ == "__main__":
    main()
