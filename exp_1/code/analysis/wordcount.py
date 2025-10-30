"""
Script name: wordcount.py
Purpose: Analyze how many words participants used in each conversation.
    - Compute word counts for participant responses (per interaction).
    - Save per-trial and per-subject word counts.
    - Compare Human vs Bot conditions (paired t-tests, plots).
    - Compare Social vs Nonsocial topics.
    - Test interaction (Condition × Social/Nonsocial) with repeated-measures ANOVA.
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/wordcount/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv
    - topics.csv (with social/nonsocial labels).

Outputs:
    - Trial-level and summary CSVs.
    - Statistical output text files.
    - Figures (barplots, violin plots).
    - Run log + config snapshot.

Usage:
    python code/analysis/wordcount.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os
from typing import List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_clean, paired_ttest_report
from utils.print_helpers import print_header, print_warn, print_save

SCRIPT_NAME = "wordcount"


# -------------------------------
# Analyses
# -------------------------------

def analyze_word_counts(sub_ids: List[str], data_dir: str, out_dir: str) -> None:
    """Interaction-level and per-subject word counts; Humans vs Bots comparison."""
    print_header("1) Humans vs Bots — Participant Word Counts")

    trials, summaries = [], []
    word_hum, word_bot = [], []

    for sub_id in sub_ids:
        csv_path = os.path.join(data_dir, f"{sub_id}.csv")
        if not os.path.exists(csv_path):
            print_warn(f"Missing CSV for {sub_id}")
            continue

        print(f"[LOAD] {csv_path}")
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        df["agent"] = df["agent"].astype(str).str.strip()
        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["sub_word_count"] = df["transcript_sub"].astype(str).apply(lambda x: len(x.split()))
        df["Subject"] = sub_id
        trials.append(df)

        summary = df.groupby("Condition")["sub_word_count"].mean().to_dict()
        word_hum.append(summary.get("hum", np.nan))
        word_bot.append(summary.get("bot", np.nan))
        summaries.append({"Subject": sub_id, "Hum": summary.get("hum"), "Bot": summary.get("bot")})

    if not trials:
        print_warn("No data for word count analysis.")
        return

    # Save trial-level
    all_trials = pd.concat(trials, ignore_index=True)
    out_trials = os.path.join(out_dir, "word_counts_by_interaction.csv")
    all_trials.to_csv(out_trials, index=False)
    print_save(out_trials, kind="CSV")

    # Save per-subject summary
    out_summary = os.path.join(out_dir, "avg_sub_wordcount_by_condition.csv")
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV")

    # Paired t-test
    stats_file = os.path.join(out_dir, "wordcount_stats.txt")
    with open(stats_file, "w") as f:
        paired_ttest_report(word_hum, word_bot, "Humans", "Bots", "Word Count", out_file=f)
    print_save(stats_file, kind="stats")


def analyze_by_social(sub_ids: List[str], data_dir: str, out_dir: str) -> None:
    """Word counts split by Social vs Nonsocial topics, with t-tests and plots."""
    print_header("2) Social vs Nonsocial — Participant Word Counts")

    topics_path = os.path.join(PROJECT_ROOT, "data/conds/topics.csv")
    topics_df = pd.read_csv(topics_path)
    topics_df["topic"] = topics_df["topic"].str.strip()

    trials = []
    for sub_id in sub_ids:
        csv_path = os.path.join(data_dir, f"{sub_id}.csv")
        if not os.path.exists(csv_path):
            print_warn(f"Missing CSV for {sub_id}")
            continue

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df["agent"] = df["agent"].astype(str).str.strip()
        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["sub_word_count"] = df["transcript_sub"].astype(str).apply(lambda x: len(x.split()))
        df["Subject"] = sub_id
        df = df.merge(topics_df, on="topic", how="left")
        df["topic_type"] = df["social"].map({1: "social", 0: "nonsocial"})
        trials.append(df)

    if not trials:
        print_warn("No merged data for social/nonsocial analysis.")
        return

    df = pd.concat(trials, ignore_index=True)
    out_csv = os.path.join(out_dir, "word_counts_by_social.csv")
    df.to_csv(out_csv, index=False)
    print_save(out_csv, kind="CSV")

    # Subject × Condition × TopicType averages
    summary = df.groupby(["Subject", "Condition", "topic_type"])["sub_word_count"].mean().reset_index()
    wide = summary.pivot(index="Subject", columns=["Condition", "topic_type"], values="sub_word_count")

    # Paired t-tests
    comparisons = [
        (("hum", "social"), ("hum", "nonsocial")),
        (("bot", "social"), ("bot", "nonsocial")),
        (("hum", "social"), ("bot", "social")),
        (("hum", "nonsocial"), ("bot", "nonsocial")),
    ]
    stats_file = os.path.join(out_dir, "wordcount_social_stats.txt")
    with open(stats_file, "w") as f:
        for a, b in comparisons:
            if a in wide.columns and b in wide.columns:
                paired_ttest_report(
                    wide[a].values.tolist(),
                    wide[b].values.tolist(),
                    f"{a[0]}_{a[1]}",
                    f"{b[0]}_{b[1]}",
                    "Word Count (by topic)",
                    out_file=f,
                )
    print_save(stats_file, kind="stats")

    # Violin plot
    long_df = summary.copy()
    long_df["Condition_Topic"] = long_df["Condition"] + "_" + long_df["topic_type"]
    palette = {"hum_nonsocial": "skyblue", "hum_social": "steelblue",
               "bot_nonsocial": "sandybrown", "bot_social": "peru"}
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=long_df, x="Condition_Topic", y="sub_word_count",
                   palette=palette, inner="box", cut=0)
    sns.swarmplot(data=long_df, x="Condition_Topic", y="sub_word_count",
                  color="k", size=4, alpha=0.6)
    plt.ylabel("Avg. Word Count")
    plt.title("Word Count by Condition × Topic Type")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "wordcount_violinplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


def run_anova(out_dir: str) -> None:
    """Repeated-measures ANOVA for word counts (Condition × Sociality)."""
    print_header("3) Repeated-Measures ANOVA — Word Counts")

    src_csv = os.path.join(out_dir, "word_counts_by_social.csv")
    if not os.path.exists(src_csv):
        print_warn(f"Missing source for ANOVA: {src_csv}")
        return

    df = pd.read_csv(src_csv)
    grouped = df.groupby(["Subject", "Condition", "social"])["sub_word_count"].mean().reset_index()
    grouped["topic_type"] = grouped["social"].map({1: "social", 0: "nonsocial"})

    try:
        aov = AnovaRM(grouped, "sub_word_count", "Subject", within=["Condition", "topic_type"]).fit()
        n_subs = grouped["Subject"].nunique()
        desc = grouped.groupby(["Condition", "topic_type"])["sub_word_count"].agg(["mean", "sem"]).round(2)

        lines = []
        lines.append(f"[ANOVA] Word Count ANOVA: N={n_subs} participants\n")
        lines.append(aov.summary().as_text())
        lines.append("\nCell means ± SEM (avg words per interaction):")
        lines.append(desc.to_string())

        # Optional η²
        if hasattr(aov, "anova_table") and "Sum Sq" in aov.anova_table.columns:
            anova_table = aov.anova_table.copy()
            total_ss = anova_table["Sum Sq"].sum()
            if total_ss and np.isfinite(total_ss) and total_ss > 0:
                anova_table["eta_sq"] = anova_table["Sum Sq"] / total_ss
                cols = [c for c in ["F Value", "Num DF", "Den DF", "Pr > F", "Sum Sq", "eta_sq"]
                        if c in anova_table.columns]
                lines.append("\nEffect sizes (η²):")
                lines.append(anova_table[cols].round(4).to_string())

        report = "\n".join(lines)
        out_stats = os.path.join(out_dir, "wordcount_anova.txt")
        with open(out_stats, "w", encoding="utf-8") as f:
            f.write(report)
        print_save(out_stats, kind="stats")

        desc_out = os.path.join(out_dir, "wordcount_anova_means.csv")
        desc.to_csv(desc_out)
        print_save(desc_out, kind="CSV")

    except Exception as e:
        print_warn(f"ANOVA failed: {e}")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Word Count analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "wordcount")  # ✅ results/model/temp/wordcount
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir, script_name=SCRIPT_NAME, args=args, cfg=cfg, used_alias=False
    )

    analyze_word_counts(subjects, data_dir, out_dir)
    analyze_by_social(subjects, data_dir, out_dir)
    run_anova(out_dir)

    logger.info("✅ Word count analysis complete.")
    print("\n[DONE] ✅ Word count analysis complete.\n")


if __name__ == "__main__":
    main()
