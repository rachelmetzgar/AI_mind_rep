"""
Script name: qual_connect_analysis.py
Purpose: Analyze average Quality and Connectedness ratings across conditions.
    - Compute per-subject and overall averages (Humans vs Bots).
    - Test differences across social vs nonsocial topics.
    - Test interaction between agent (human/bot) × topic type (social/nonsocial).
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/qual_connect/.

Inputs:
    - Per-subject CSVs in data/<model>/<temperature>/{s###}.csv
    - topics.csv (with social/nonsocial labels)

Outputs:
    - Trial-level and summary CSVs
    - Statistical output text files
    - Figures (barplots, violin plots)
    - Run log + config snapshot

Usage:
    python code/analysis/qual_connect_analysis.py --config configs/behavior.json
    python code/analysis/qual_connect_analysis.py --config configs/behavior.json --sub-id s001

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_clean, paired_ttest_report
from utils.print_helpers import print_header, print_save, print_warn, print_info

SCRIPT_NAME = "qual_connect_analysis"


# -------------------------------
# Analyses
# -------------------------------

def average_qual_con(sub_ids: List[str], data_dir: str, out_dir: str) -> None:
    """Compare average Quality & Connectedness ratings for human vs bot conditions."""
    print_header("1) Humans vs Bots — Average Quality & Connectedness")
    print_info(f"Subjects to process: {len(sub_ids)} → {sub_ids}")

    summaries, trials = [], []
    qual_hum, qual_bot, con_hum, con_bot = [], [], [], []
    missing_subjects = []

    for sub_id in sub_ids:
        file_path = os.path.join(data_dir, f"{sub_id}.csv")
        if not os.path.exists(file_path):
            print_warn(f"Missing CSV for {sub_id} (skipping).")
            missing_subjects.append(sub_id)
            continue

        print(f"[LOAD] {sub_id}: {file_path}")
        df = pd.read_csv(csv_path, on_bad_lines="skip")
        df.columns = df.columns.str.strip()

        # Normalize/clean
        df["agent"] = df["agent"].astype(str).str.strip()
        df["Quality"] = pd.to_numeric(df["Quality"], errors="coerce")
        df["Connectedness"] = pd.to_numeric(df["Connectedness"], errors="coerce")
        df["Subject"] = sub_id

        # Trial-level
        trial_df = df[["Subject", "run", "order", "topic", "Quality", "Connectedness", "agent"]].dropna()
        print_info(f"{sub_id}: trial rows kept after dropna = {len(trial_df)}")
        trials.append(trial_df)

        # Split by condition
        hum = df[df["agent"].str.contains("hum", na=False)]
        bot = df[df["agent"].str.contains("bot", na=False)]

        qh = hum["Quality"].mean(skipna=True)
        qb = bot["Quality"].mean(skipna=True)
        ch = hum["Connectedness"].mean(skipna=True)
        cb = bot["Connectedness"].mean(skipna=True)

        print(f"       {sub_id}: mean Quality (hum={qh:.3f}, bot={qb:.3f}); "
              f"Connectedness (hum={ch:.3f}, bot={cb:.3f})")

        qual_hum.append(qh); qual_bot.append(qb)
        con_hum.append(ch);  con_bot.append(cb)

        summaries.extend([
            {"Subject": sub_id, "Condition": "Humans",
             "Avg_Quality": qh, "Avg_Connectedness": ch},
            {"Subject": sub_id, "Condition": "Bots",
             "Avg_Quality": qb, "Avg_Connectedness": cb},
        ])

    if len(trials) == 0:
        print_warn("No trial data assembled. Exiting section.")
        return

    # Save combined data
    all_trials_path = os.path.join(out_dir, "qual_con_all_data.csv")
    pd.concat(trials, ignore_index=True).to_csv(all_trials_path, index=False)
    print_save(all_trials_path, kind="CSV")

    averages_path = os.path.join(out_dir, "qual_con_averages.csv")
    pd.DataFrame(summaries).to_csv(averages_path, index=False)
    print_save(averages_path, kind="CSV")

    # Paired t-tests (write means; add SEMs for readability)
    results_txt = os.path.join(out_dir, "statistical_output.txt")
    with open(results_txt, "w") as f:
        # Quality
        res_q = paired_ttest_report(qual_hum, qual_bot, "Humans", "Bots", "Quality", out_file=f)
        a_q, b_q = paired_clean(qual_hum, qual_bot)
        if a_q.size > 0:
            sems = (np.std(a_q, ddof=1)/np.sqrt(a_q.size) if a_q.size > 1 else np.nan,
                    np.std(b_q, ddof=1)/np.sqrt(b_q.size) if b_q.size > 1 else np.nan)
            f.write(f"Quality SEMs: Humans={sems[0]:.3f}, Bots={sems[1]:.3f}\n")
        else:
            f.write("Quality: no usable pairs for SEMs.\n")

        # Connectedness
        res_c = paired_ttest_report(con_hum, con_bot, "Humans", "Bots", "Connectedness", out_file=f)
        a_c, b_c = paired_clean(con_hum, con_bot)
        if a_c.size > 0:
            sems = (np.std(a_c, ddof=1)/np.sqrt(a_c.size) if a_c.size > 1 else np.nan,
                    np.std(b_c, ddof=1)/np.sqrt(b_c.size) if b_c.size > 1 else np.nan)
            f.write(f"Connectedness SEMs: Humans={sems[0]:.3f}, Bots={sems[1]:.3f}\n")
        else:
            f.write("Connectedness: no usable pairs for SEMs.\n")
    print_save(results_txt, kind="stats")

    # Bar plots
    def plot_bar(vals1: List[float], vals2: List[float],
                 title: str, ylabel: str, fname: str, colors: Tuple[str, str]) -> None:
        a, b = paired_clean(vals1, vals2)
        if a.size == 0:
            print_warn(f"{title}: skipped (no usable pairs).")
            return
        means = [float(np.mean(a)), float(np.mean(b))]
        sems = [
            float(np.std(a, ddof=1)/np.sqrt(len(a))) if len(a) > 1 else np.nan,
            float(np.std(b, ddof=1)/np.sqrt(len(b))) if len(b) > 1 else np.nan
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(["Humans", "Bots"], means, yerr=sems, capsize=5, color=list(colors))
        for i in range(len(a)):
            plt.plot([0, 1], [a[i], b[i]], color="gray", alpha=0.5)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print_save(out_path, kind="figure")

    plot_bar(qual_hum, qual_bot,
             "Average Quality Ratings", "Quality",
             "quality_comparison.png", ("skyblue", "sandybrown"))
    plot_bar(con_hum, con_bot,
             "Average Connectedness Ratings", "Connectedness",
             "connectedness_comparison.png", ("skyblue", "sandybrown"))

    if missing_subjects:
        print_info(f"Subjects skipped due to missing CSVs: {missing_subjects}")


def analyze_social_vs_nonsocial(sub_ids: List[str], data_dir: str, out_dir: str) -> None:
    """Compare Quality/Connectedness across social vs nonsocial topics."""
    print_header("2) Social vs Nonsocial — Collapsed Across Agent")

    topics_path = os.path.join(PROJECT_ROOT, "data/conds/topics.csv")
    print(f"[LOAD] topics.csv → {topics_path}")
    topic_info = pd.read_csv(topics_path)
    topic_info["topic"] = topic_info["topic"].str.strip()

    trials = []
    for sub_id in sub_ids:
        file_path = os.path.join(data_dir, f"{sub_id}.csv")
        if not os.path.exists(file_path):
            print_warn(f"Missing CSV for {sub_id} (skipping social/nonsocial merge).")
            continue

        print(f"[LOAD] {sub_id}: {file_path}")
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df["topic"] = df["topic"].str.strip()
        df["Quality"] = pd.to_numeric(df["Quality"], errors="coerce")
        df["Connectedness"] = pd.to_numeric(df["Connectedness"], errors="coerce")
        df["Subject"] = sub_id
        df = df[["Subject", "run", "order", "topic", "Quality", "Connectedness", "agent"]].dropna()
        before = len(df)
        df = df.merge(topic_info, on="topic", how="left")
        after = len(df)
        missing_soc = df["social"].isna().sum()
        print(f"[MERGE] {sub_id}: rows before/after={before}/{after}; "
              f"topics with missing 'social' label={missing_soc}")
        trials.append(df)

    if len(trials) == 0:
        print_warn("No merged trial data assembled for social/nonsocial. Exiting section.")
        return

    df = pd.concat(trials, ignore_index=True)
    all_soc_path = os.path.join(out_dir, "qual_con_soc_all_data.csv")
    df.to_csv(all_soc_path, index=False)
    print_save(all_soc_path, kind="CSV")

    summary = df.groupby(["Subject", "social"]).agg({"Quality": "mean", "Connectedness": "mean"}).reset_index()
    pivot_q = summary.pivot(index="Subject", columns="social", values="Quality")
    pivot_c = summary.pivot(index="Subject", columns="social", values="Connectedness")

    for col in [0, 1]:
        if col not in pivot_q.columns:
            pivot_q[col] = np.nan
        if col not in pivot_c.columns:
            pivot_c[col] = np.nan

    stats_path = os.path.join(out_dir, "social_vs_nonsocial_stats.txt")
    with open(stats_path, "w") as f:
        f.write("[TEST] Paired t-tests (Social=1 vs Nonsocial=0), per-subject means\n")
        paired_ttest_report(
            pivot_q[1].values.tolist(), pivot_q[0].values.tolist(),
            "Social", "Nonsocial", "Quality (Social vs Nonsocial)", out_file=f
        )
        a_q, b_q = paired_clean(pivot_q[1].values, pivot_q[0].values)
        if a_q.size > 0:
            sems_q = (np.std(a_q, ddof=1)/np.sqrt(a_q.size) if a_q.size > 1 else np.nan,
                      np.std(b_q, ddof=1)/np.sqrt(a_q.size) if a_q.size > 1 else np.nan)
            f.write(f"Quality SEMs: Social={sems_q[0]:.3f}, Nonsocial={sems_q[1]:.3f}\n")

        paired_ttest_report(
            pivot_c[1].values.tolist(), pivot_c[0].values.tolist(),
            "Social", "Nonsocial", "Connectedness (Social vs Nonsocial)", out_file=f
        )
        a_c, b_c = paired_clean(pivot_c[1].values, pivot_c[0].values)
        if a_c.size > 0:
            sems_c = (np.std(a_c, ddof=1)/np.sqrt(a_c.size) if a_c.size > 1 else np.nan,
                      np.std(b_c, ddof=1)/np.sqrt(a_c.size) if a_c.size > 1 else np.nan)
            f.write(f"Connectedness SEMs: Social={sems_c[0]:.3f}, Nonsocial={sems_c[1]:.3f}\n")
    print_save(stats_path, kind="stats")


def test_interaction(out_dir: str) -> None:
    """Run repeated-measures ANOVA: Condition × Social/Nonsocial."""
    print_header("3) Repeated-Measures ANOVA — Agent × Sociality (per-subject means)")
    src = os.path.join(out_dir, "qual_con_soc_all_data.csv")
    if not os.path.exists(src):
        print_warn(f"Missing merged file for ANOVA: {src}")
        return
    print(f"[LOAD] ANOVA source: {src}")
    df = pd.read_csv(src)

    df["Condition"] = df["agent"].str.extract(r"(hum|bot)", expand=False)
    df["social"] = df["social"].replace({0: "nonsocial", 1: "social"})

    grouped = df.groupby(["Subject", "Condition", "social"]).agg(
        {"Quality": "mean", "Connectedness": "mean"}
    ).reset_index()

    n_subs = grouped["Subject"].nunique()
    print_info(f"ANOVA groups rows: {grouped[['Subject','Condition','social']].drop_duplicates().shape[0]} "
               f"(N={n_subs} × 2 agents × 2 sociality minus missing)")

    out_txt = os.path.join(out_dir, "anova_stats.txt")
    lines = ["--- Repeated-Measures ANOVA ---", f"N subjects = {n_subs}"]

    for measure in ["Quality", "Connectedness"]:
        print(f"[ANOVA] Fitting AnovaRM for {measure} with within-factors: Condition, social")
        try:
            aovrm = AnovaRM(grouped, depvar=measure, subject="Subject", within=["Condition", "social"]).fit()
            desc = grouped.groupby(["Condition", "social"])[measure].agg(["mean", "sem"])
            lines.append(f"\n=== {measure} ===")
            lines.append(aovrm.summary().as_text())
            lines.append("\nCell means ± SEM:")
            lines.append(desc.round(3).to_string())
        except Exception as e:
            msg = f"[ERROR] ANOVA failed for {measure}: {e}"
            print_warn(msg)
            lines.append(msg)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print_save(out_txt, kind="stats")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Quality & Connectedness analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "qual_connect")  # ✅ results/model/temp/qual_connect
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print(f"Running {SCRIPT_NAME}")
    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] Output directory: {out_dir}")
    print(f"[INFO] Subjects: {subjects}")
    print("=" * 80)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir, script_name=SCRIPT_NAME, args=args, cfg=cfg, used_alias=False
    )

    average_qual_con(subjects, data_dir, out_dir)
    analyze_social_vs_nonsocial(subjects, data_dir, out_dir)
    test_interaction(out_dir)

    logger.info("✅ Behavioral Quality & Connectedness analysis complete.")
    print("\n[DONE] ✅ Behavioral Quality & Connectedness analysis complete.\n")


if __name__ == "__main__":
    main()
