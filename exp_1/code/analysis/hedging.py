#!/usr/bin/env python3
"""
Script name: hedging.py
Purpose: Analyze hedging in participant transcripts using regex markers.
    - Compute hedging scores (# of hedges per utterance).
    - Save per-trial and per-subject averages.
    - Compare Human vs Bot conditions with paired t-tests.
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/hedging/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv

Outputs:
    - Trial-level and summary CSVs.
    - Statistical output text files.
    - Figures (bar/violin plots).
    - Run log + config snapshot.

Usage:
    python code/analysis/hedging.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_save, print_warn

SCRIPT_NAME = "hedging"

# -------------------------------
# Hedging markers
# -------------------------------
HEDGE_MARKERS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bmight\b", r"\bcould be\b",
    r"\bit seems\b", r"\bi think\b", r"\bin a way\b", r"\bsort of\b", r"\bkind of\b",
    r"\bmore or less\b", r"\broughly\b", r"\btends to\b"
]


def count_hedges(text: str) -> int:
    """Count hedging markers in text."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    text = text.lower()
    return sum(1 for pattern in HEDGE_MARKERS if re.search(pattern, text))


# -------------------------------
# Main Analysis
# -------------------------------

def analyze_hedging(data_dir: str, out_dir: str):
    """Compute per-subject and per-condition hedging statistics."""
    print_header("1) Hedging Analysis — Humans vs Bots")

    trials = []
    for subfile in sorted(os.listdir(data_dir)):
        if not subfile.endswith(".csv"):
            continue
        sub_id = os.path.splitext(subfile)[0]
        csv_path = os.path.join(data_dir, subfile)

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()
        if "agent" not in df.columns or "transcript_sub" not in df.columns:
            print_warn(f"{sub_id}: Missing expected columns; skipping.")
            continue

        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["Subject"] = sub_id
        df["Hedge_Count"] = df["transcript_sub"].apply(count_hedges)
        trials.append(df)

    if not trials:
        print_warn("No transcript CSVs found — analysis aborted.")
        return

    df = pd.concat(trials, ignore_index=True)

    # Subject-level summary
    summary = df.groupby(["Subject", "Condition"])["Hedge_Count"].mean().unstack().reset_index()
    out_summary = os.path.join(out_dir, "hedging_summary_all_subjects.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV")

    # Paired t-test
    if "hum" in summary and "bot" in summary:
        x, y = paired_clean(summary["hum"], summary["bot"])
        t_stat, p_val = ttest_rel(x, y, nan_policy="omit")
        mean_hum, mean_bot = x.mean(), y.mean()
        sem_hum = x.std(ddof=1) / np.sqrt(len(x))
        sem_bot = y.std(ddof=1) / np.sqrt(len(y))
        diff = x - y
        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan

        lines = [
            "--- Paired t-test: Hedging (Hum vs Bot) ---",
            f"N = {len(x)} paired subjects",
            f"Mean (Human) = {mean_hum:.3f} ± {sem_hum:.3f} SEM",
            f"Mean (Bot)   = {mean_bot:.3f} ± {sem_bot:.3f} SEM",
            f"Mean difference (Human - Bot) = {(mean_hum - mean_bot):.3f}",
            f"t({len(x)-1}) = {t_stat:.3f}, p = {p_val:.4f}",
            f"Cohen's d = {cohens_d:.3f}",
            "Interpretation: " + ("Significant (p < .05)" if p_val < 0.05 else "Not significant (p > .05)")
        ]

        stats_path = os.path.join(out_dir, "hedging_stats.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print_save(stats_path, kind="stats")

    # Plot
    long_df = summary.melt(id_vars="Subject", var_name="Condition", value_name="Hedge_Count")
    plt.figure(figsize=(6, 5))
    sns.violinplot(data=long_df, x="Condition", y="Hedge_Count",
                   palette={"hum": "skyblue", "bot": "sandybrown"}, inner="box")
    for _, row in summary.iterrows():
        plt.plot(["hum", "bot"], [row["hum"], row["bot"]], color="gray", alpha=0.4)
    plt.title("Hedging by Condition")
    plt.ylabel("Average Hedge Count per Utterance")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "hedging_violin.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Hedging analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "hedging")  # ✅ results/model/temp/hedging
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_hedging(data_dir, out_dir)

    logger.info("✅ Hedging analysis complete.")
    print("\n[DONE] ✅ Hedging analysis complete.\n")


if __name__ == "__main__":
    main()
