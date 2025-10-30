#!/usr/bin/env python3
"""
Script name: questions.py
Purpose: Analyze total number of questions participants asked across the experiment.
    - Reads from the combined transcript CSV created by combine_text_data.py.
    - Counts questions using two methods: question marks (?) and regex-based patterns.
    - Saves per-trial and per-subject averages.
    - Compares Human vs Bot question frequency (paired t-tests).
    - Saves results (CSVs, plots, stats) to results/<model>/<temperature>/questions/.

Inputs:
    - Combined transcript CSV: data/<model>/<temperature>/combined_text_data.csv

Outputs:
    - questions_by_interaction_<method>.csv
    - questions_subject_summary_<method>.csv
    - questions_stats_<method>.txt
    - questions_barplot_<method>.png
    - questions_violinplot_<method>.png

Usage:
    python code/analysis/questions.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-31
"""

from __future__ import annotations
import os, re
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_info, print_warn
from utils.plot_helpers import barplot_with_lines, main_effect_violin_lines, DEFAULT_PALETTE

SCRIPT_NAME = "questions"

# ------------------------------------------------------------
# Question detection helpers
# ------------------------------------------------------------
QUESTION_STARTS = re.compile(
    r"^\s*(who|what|when|where|why|how|which|do|does|did|can|could|would|will|should|is|are|am|was|were)\b",
    re.I,
)


def regex_question_count(text: str) -> int:
    """Count sentences that appear to be questions:
    - end with '?'
    - OR start with interrogative words
    """
    if pd.isna(text):
        return 0
    sentences = re.split(r"(?<=[.?!])\s+", str(text))
    count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s.endswith("?") or QUESTION_STARTS.match(s):
            count += 1
    return count


# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------
def analyze_questions(combined_path: str, out_dir: str, method: str = "question_mark"):
    """Analyze total participant questions (Hum vs Bot) using the specified detection method."""
    label = "?" if method == "question_mark" else "regex-based"
    print_header(f"1) Total Questions — Method: {label}")

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
    # Count questions per utterance
    # ------------------------------------------------------------
    print_info(f"Counting questions using method: {method}")
    if method == "regex":
        df["Question_Count"] = df["transcript_sub"].apply(regex_question_count)
    else:
        df["Question_Count"] = df["transcript_sub"].astype(str).apply(lambda x: x.count("?"))

    # ------------------------------------------------------------
    # Save per-utterance data
    # ------------------------------------------------------------
    out_interactions = os.path.join(out_dir, f"questions_by_interaction_{method}.csv")
    cols = ["Subject", "agent", "Condition", "topic", "transcript_sub", "Question_Count"]
    df[cols].to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV (per-interaction question counts)")

    # ------------------------------------------------------------
    # Subject-level totals
    # ------------------------------------------------------------
    summary = df.groupby(["Subject", "Condition"])["Question_Count"].sum().unstack().reset_index()
    out_summary = os.path.join(out_dir, f"questions_subject_summary_{method}.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV (subject-level totals)")

    # ------------------------------------------------------------
    # Paired t-test (Hum vs Bot)
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
                f"--- Paired t-test: Total Questions (Hum vs Bot, {method}) ---",
                f"N = {len(x)} paired subjects",
                f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
                f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
                f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.3f}",
                f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val_main:.4f}",
                f"Cohen's d = {cohens_d:.3f}",
                "Interpretation: "
                + ("Significant (p < .05)" if p_val_main < 0.05 else "Not significant (p > .05)"),
            ]

            stats_path = os.path.join(out_dir, f"questions_stats_{method}.txt")
            with open(stats_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print_save(stats_path, kind="stats")

    # ------------------------------------------------------------
    # Plots (bar + violin)
    # ------------------------------------------------------------
    long_df = summary.rename(columns={"hum": "Hum", "bot": "Bot"}).melt(
        id_vars="Subject", var_name="Condition", value_name="Question_Count"
    )

    # Bar plot
    barplot_with_lines(
        df_long=long_df,
        x_col="Condition",
        y_col="Question_Count",
        out_path=os.path.join(out_dir, f"questions_barplot_{method}.png"),
        title=f"Total Questions by Condition ({label})",
        palette={"Hum": "steelblue", "Bot": "sandybrown"},
        p_val=p_val_main,
    )

    # Violin plot
    main_effect_violin_lines(
        df_summary=summary.rename(columns={"hum": "Hum", "bot": "Bot"}),
        cond_a="Hum",
        cond_b="Bot",
        y_col="Question_Count",
        out_path=os.path.join(out_dir, f"questions_violinplot_{method}.png"),
        title=f"Total Questions per Subject ({label})",
        palette=DEFAULT_PALETTE,
        p_val=p_val_main,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args, cfg = parse_and_load_config("Questions analysis")
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    combined_path = os.path.join(data_dir, "combined_text_data.csv")
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "questions")
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    # Run both detection methods
    analyze_questions(combined_path, out_dir, method="question_mark")
    analyze_questions(combined_path, out_dir, method="regex")

    logger.info("✅ Questions analysis complete.")
    print("\n[DONE] ✅ Questions analysis complete.\n")


if __name__ == "__main__":
    main()
