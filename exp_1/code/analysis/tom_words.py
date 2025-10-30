"""
Script name: tom_words.py
Purpose: Analyze Theory of Mind (ToM) word use in participant transcripts.
    - Count ToM-related phrases in participant responses.
    - Save per-trial and per-subject totals.
    - Compare Human vs Bot totals (paired t-test + plot).
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/tom_words/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv

Outputs:
    - Trial-level and summary CSVs.
    - Statistical output text files (detailed, explicit).
    - Figures (bar plots with subject-level lines).
    - Run log + config snapshot.

Usage:
    python code/analysis/tom_words.py --config configs/behavior.json

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

SCRIPT_NAME = "tom_words"

# -------------------------------
# ToM Word List
# -------------------------------
TOM_WORDS = [
    "you think", "you believe", "you know", "you feel", "you understand",
    "you guess", "you imagine", "you wonder", "you consider", "you expect",
    "you hope", "you assume", "you realize", "you remember", "you forget"
]


def count_tom_words(text: str) -> int:
    """Count the number of Theory-of-Mind (ToM) phrases in a text string."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    text = text.lower()
    count = 0
    for phrase in TOM_WORDS:
        count += len(re.findall(rf"\b{re.escape(phrase)}\b", text))
    return count


# -------------------------------
# Analysis
# -------------------------------

def analyze_tom_words(sub_ids: List[str], data_dir: str, out_dir: str) -> None:
    print_header("1) Theory of Mind Word Analysis — Humans vs Bots")

    trials, totals = [], []
    tom_hum, tom_bot = [], []

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

        # Count ToM words
        df["ToM_count"] = df["transcript_sub"].apply(count_tom_words)
        trials.append(df)

        total = df.groupby("Condition")["ToM_count"].sum().to_dict()
        tom_hum.append(total.get("hum", 0))
        tom_bot.append(total.get("bot", 0))
        totals.append({"Subject": sub_id, "Hum": total.get("hum", 0), "Bot": total.get("bot", 0)})

        print_info(f"{sub_id}: total ToM words (hum={total.get('hum', 0)}, bot={total.get('bot', 0)})")

    if not trials:
        print_warn("No data available — analysis aborted.")
        return

    # Save interaction-level data
    trial_df = pd.concat(trials, ignore_index=True)
    out_interactions = os.path.join(out_dir, "tom_counts_by_interaction.csv")
    trial_df.to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV")

    # Save subject-level totals
    totals_df = pd.DataFrame(totals)
    out_totals = os.path.join(out_dir, "tom_word_totals_by_subject.csv")
    totals_df.to_csv(out_totals, index=False)
    print_save(out_totals, kind="CSV")

    # Run paired t-test
    stats_path = os.path.join(out_dir, "tom_words_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("--- Paired-samples t-test: Theory of Mind Words ---\n")
        f.write("Analysis: Do participants use more ToM-related words when speaking with Humans vs Bots?\n\n")

        res = paired_ttest_report(tom_hum, tom_bot, "Humans", "Bots", "ToM Words", out_file=f)
        a, b = paired_clean(tom_hum, tom_bot)
        if a.size > 0:
            sems = (a.std(ddof=1) / np.sqrt(a.size), b.std(ddof=1) / np.sqrt(b.size))
            f.write(f"\nN = {a.size} paired observations\n")
            f.write(f"Mean (Humans) = {a.mean():.2f} ± {sems[0]:.3f} SEM\n")
            f.write(f"Mean (Bots)   = {b.mean():.2f} ± {sems[1]:.3f} SEM\n\n")
            f.write(f"t({a.size-1}) = {res[0]:.3f}, p = {res[1]:.4f}\n")
            f.write(f"Effect size (Cohen’s dz) = {res[2]:.3f}\n\n")
            f.write("Interpretation: " +
                    ("Significant difference (p < .05)." if res[1] < 0.05
                     else "No significant difference (p > .05)."))
    print_save(stats_path, kind="stats")

    # Plot
    long_df = totals_df.melt(id_vars="Subject", var_name="Condition", value_name="ToM_Words")
    plt.figure(figsize=(6, 5))
    sns.barplot(data=long_df, x="Condition", y="ToM_Words",
                palette={"Hum": "skyblue", "Bot": "sandybrown"}, ci="sd")
    for _, row in totals_df.iterrows():
        plt.plot(["Hum", "Bot"], [row["Hum"], row["Bot"]],
                 color="gray", alpha=0.5, linewidth=1)
    plt.title("Total ToM Words by Condition")
    plt.ylabel("Total ToM Words (per subject)")
    plt.xlabel("Condition")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "tom_words_barplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("ToM Words analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "tom_words")  # ✅ results/model/temp/tom_words
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_tom_words(subjects, data_dir, out_dir)

    logger.info("✅ ToM words analysis complete.")
    print("\n[DONE] ✅ ToM words analysis complete.\n")


if __name__ == "__main__":
    main()
