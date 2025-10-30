"""
Script name: politeness.py
Purpose: Analyze politeness in participant transcripts.
    - Compute politeness scores per trial and per subject.
    - Compare Human vs Bot politeness (t-tests, plots).
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/politeness/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv

Outputs:
    - Trial-level politeness CSVs.
    - Statistical outputs (t-tests).
    - Figures (barplots).
    - Run log + config snapshot.

Usage:
    python code/analysis/politeness.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run
from utils.print_helpers import print_header, print_warn, print_save
from utils.stats_helpers import paired_ttest_report, paired_clean


SCRIPT_NAME = "politeness"


# -------------------------------
# Core Analysis
# -------------------------------

def analyze_politeness(data_dir: str, out_dir: str):
    """Compute politeness scores and compare Human vs Bot conditions."""
    print_header("1) Politeness Analysis — Human vs Bot")

    trials, summaries = [], []
    pol_hum, pol_bot = [], []

    for file in sorted(os.listdir(data_dir)):
        if not file.endswith(".csv"):
            continue
        sub_id = os.path.splitext(file)[0]
        csv_path = os.path.join(data_dir, file)
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()

        if "agent" not in df.columns or "transcript_sub" not in df.columns:
            print_warn(f"{sub_id}: Missing expected columns; skipping.")
            continue

        df["agent"] = df["agent"].astype(str).str.strip()
        df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
        df["Subject"] = sub_id

        # Simple politeness score = count of "please" + "thank"
        df["Politeness_Score"] = df["transcript_sub"].astype(str).apply(
            lambda t: sum(w in t.lower() for w in ["please", "thank"])
        )

        trials.append(df)
        avg = df.groupby("Condition")["Politeness_Score"].mean().to_dict()
        pol_hum.append(avg.get("hum", np.nan))
        pol_bot.append(avg.get("bot", np.nan))
        summaries.append({"Subject": sub_id, "Hum": avg.get("hum"), "Bot": avg.get("bot")})

    if not trials:
        print_warn("No data found for politeness analysis.")
        return

    # Save trial-level
    all_trials = pd.concat(trials, ignore_index=True)
    out_trials = os.path.join(out_dir, "politeness_by_interaction.csv")
    all_trials.to_csv(out_trials, index=False)
    print_save(out_trials, kind="CSV")

    # Save per-subject summary
    out_summary = os.path.join(out_dir, "politeness_subject_summary.csv")
    pd.DataFrame(summaries).to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV")

    # Paired t-test
    stats_path = os.path.join(out_dir, "politeness_ttest.txt")
    with open(stats_path, "w") as f:
        paired_ttest_report(pol_hum, pol_bot, "Humans", "Bots", "Politeness", out_file=f)
    print_save(stats_path, kind="stats")

    # Bar plot
    long_df = pd.DataFrame(summaries).melt(id_vars="Subject", var_name="Condition", value_name="Politeness")
    plt.figure(figsize=(6, 5))
    sns.barplot(data=long_df, x="Condition", y="Politeness",
                palette={"Hum": "skyblue", "Bot": "sandybrown"}, ci="sd")
    for _, row in long_df.pivot(index="Subject", columns="Condition", values="Politeness").iterrows():
        plt.plot(["Hum", "Bot"], [row["Hum"], row["Bot"]], color="gray", alpha=0.5)
    plt.ylabel("Avg. Politeness Score")
    plt.title("Politeness by Condition")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "politeness_barplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Politeness analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "politeness")  # ✅ results/model/temp/politeness
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_politeness(data_dir, out_dir)

    logger.info("✅ Politeness analysis complete.")
    print("\n[DONE] ✅ Politeness analysis complete.\n")


if __name__ == "__main__":
    main()
