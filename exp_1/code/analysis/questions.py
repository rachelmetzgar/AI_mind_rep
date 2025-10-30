"""
Script name: questions.py
Purpose: Analyze total number of questions participants asked across the experiment.
    - Count questions by question marks (?) in participant responses.
    - Exclude participants who asked 0 questions in both conditions.
    - Re-run analysis using regex-based detection of questions.
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/questions/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv
    - Each file must include 'transcript_sub', 'agent', and 'topic' columns.

Outputs:
    - Trial-level and summary CSVs.
    - Statistical output text files.
    - Figures (bar plots with subject-level lines).
    - Run log + config snapshot.

Usage:
    python code/analysis/questions.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os, re
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_ttest_report, paired_clean
from utils.print_helpers import print_header, print_warn, print_save, print_info

SCRIPT_NAME = "questions"


# -------------------------------
# Helpers
# -------------------------------

QUESTION_STARTS = re.compile(
    r'^\s*(who|what|when|where|why|how|which|do|does|did|can|could|would|will|should|is|are|am|was|were)\b',
    re.I
)

def regex_question_count(text: str) -> int:
    """Count sentences that look like questions:
    - end with '?'
    - OR start with interrogative words (who, what, why, do, is, etc.)
    """
    sentences = re.split(r'(?<=[.?!])\s+', str(text))
    count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s.endswith("?") or QUESTION_STARTS.match(s):
            count += 1
    return count


def run_paired_test_and_save(q_hum, q_bot, label, out_path, description):
    """Run paired t-test and save detailed results."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"--- Paired-samples t-test: {label} ---\n")
        f.write(f"Analysis: {description}\n\n")

        res = paired_ttest_report(q_hum, q_bot, "Humans", "Bots", label, out_file=f)

        a, b = paired_clean(q_hum, q_bot)
        if a.size > 0:
            sems = (a.std(ddof=1) / np.sqrt(a.size), b.std(ddof=1) / np.sqrt(b.size))
            f.write(f"\nN = {a.size} paired observations\n")
            f.write(f"Mean (Humans) = {a.mean():.2f} ± {sems[0]:.3f} SEM\n")
            f.write(f"Mean (Bots)   = {b.mean():.2f} ± {sems[1]:.3f} SEM\n\n")
            f.write(f"t({a.size-1}) = {res[0]:.3f}, p = {res[1]:.4f}, dz = {res[2]:.3f}\n")
            f.write("Significant" if res[1] < 0.05 else "Not significant")
            f.write("\n")
    print_save(out_path, kind="stats")


def plot_totals(df_totals: pd.DataFrame, title: str, out_path: str):
    """Bar plot with subject-level lines for totals."""
    long_df = df_totals.melt(id_vars="Subject", var_name="Condition", value_name="Question_Count")
    plt.figure(figsize=(6, 5))
    sns.barplot(
        data=long_df, x="Condition", y="Question_Count",
        palette={"Hum": "skyblue", "Bot": "sandybrown"}, ci="sd"
    )
    for _, row in df_totals.iterrows():
        plt.plot(["Hum", "Bot"], [row["Hum"], row["Bot"]], color="gray", alpha=0.5, linewidth=1)
    plt.title(title)
    plt.ylabel("Total Questions per Subject")
    plt.xlabel("Condition")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print_save(out_path, kind="figure")


# -------------------------------
# Analyses
# -------------------------------

def analyze_by_method(sub_ids: List[str], data_dir: str, out_dir: str, method: str = "question_mark"):
    """Run question analyses using different detection methods."""
    if method == "question_mark":
        print_header("1) Total Participant Questions — Question Mark Method")
    elif method == "regex":
        print_header("3) Total Participant Questions — Regex Method")

    trials, totals = [], []
    q_hum, q_bot = [], []

    for sub_id in sub_ids:
        csv_path = os.path.join(data_dir, f"{sub_id}.csv")
        if not os.path.exists(csv_path):
            print_warn(f"Missing transcript CSV for {sub_id}, skipping.")
            continue

        print(f"[LOAD] {sub_id}: {csv_path}")
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        df["agent"] = df["agent"].astype(str).str.strip()
        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["Subject"] = sub_id

        # Count questions
        if method == "regex":
            df["question_count"] = df["transcript_sub"].astype(str).apply(regex_question_count)
        else:
            df["question_count"] = df["transcript_sub"].astype(str).apply(lambda x: x.count("?"))

        trials.append(df)

        total = df.groupby("Condition")["question_count"].sum().to_dict()
        q_hum.append(total.get("hum", 0))
        q_bot.append(total.get("bot", 0))
        totals.append({"Subject": sub_id, "Hum": total.get("hum", 0), "Bot": total.get("bot", 0)})

        print_info(f"{sub_id}: total questions (hum={total.get('hum', 0)}, bot={total.get('bot', 0)})")

    if not trials:
        print_warn("No data available — analysis aborted.")
        return

    # Save trial-level data
    trial_df = pd.concat(trials, ignore_index=True)
    trial_path = os.path.join(out_dir, f"question_counts_by_interaction_{method}.csv")
    trial_df.to_csv(trial_path, index=False)
    print_save(trial_path, kind="CSV")

    # Save subject-level totals
    totals_df = pd.DataFrame(totals)
    totals_path = os.path.join(out_dir, f"total_questions_by_subject_{method}.csv")
    totals_df.to_csv(totals_path, index=False)
    print_save(totals_path, kind="CSV")

    # Run paired t-test
    stats_path = os.path.join(out_dir, f"statistical_output_{method}.txt")
    desc = f"Do participants ask more questions (method={method}) when speaking with Humans vs Bots?"
    run_paired_test_and_save(q_hum, q_bot, f"Total Questions ({method})", stats_path, desc)

    # Plot
    plot_path = os.path.join(out_dir, f"total_questions_barplot_{method}.png")
    plot_totals(totals_df, f"Total Questions by Condition ({method})", plot_path)

    # Secondary analysis: excluding non-askers
    analyze_excluding_nonaskers(totals_df, out_dir, method)


def analyze_excluding_nonaskers(df_totals: pd.DataFrame, out_dir: str, method: str):
    """Repeat analysis excluding participants with 0 questions in both conditions."""
    print_header(f"2) Excluding Non-Askers — Method={method}")

    filtered = df_totals[(df_totals["Hum"] > 0) | (df_totals["Bot"] > 0)]
    if filtered.empty:
        print_warn("No subjects left after excluding non-askers.")
        return

    q_hum = filtered["Hum"].tolist()
    q_bot = filtered["Bot"].tolist()
    print_info(f"Retained {len(filtered)} subjects (excluded {len(df_totals) - len(filtered)} non-askers).")

    # Save filtered totals
    out_totals = os.path.join(out_dir, f"total_questions_excluding_nonaskers_{method}.csv")
    filtered.to_csv(out_totals, index=False)
    print_save(out_totals, kind="CSV")

    # Run paired t-test
    stats_path = os.path.join(out_dir, f"statistical_output_excluding_nonaskers_{method}.txt")
    desc = f"Do participants who asked ≥1 question differ between Humans vs Bots? (method={method})"
    run_paired_test_and_save(q_hum, q_bot, f"Questions (Excluding Non-Askers, {method})", stats_path, desc)

    # Plot
    out_fig = os.path.join(out_dir, f"questions_barplot_excluding_nonaskers_{method}.png")
    plot_totals(filtered, f"Total Questions (Excluding Non-Askers, {method})", out_fig)


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Total Questions analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "questions")  # ✅ results/model/temp/questions
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir, script_name=SCRIPT_NAME, args=args, cfg=cfg, used_alias=False
    )

    analyze_by_method(subjects, data_dir, out_dir, method="question_mark")
    analyze_by_method(subjects, data_dir, out_dir, method="regex")

    logger.info("✅ Total questions analysis complete.")
    print("\n[DONE] ✅ Total questions analysis complete.\n")


if __name__ == "__main__":
    main()
