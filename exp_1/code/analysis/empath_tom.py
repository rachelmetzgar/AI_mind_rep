"""
Script name: empath_tom.py
Purpose: Analyze Empath-derived Theory of Mind (ToM) language use in participant transcripts.
    - Compute Empath scores for custom "thinking_custom" category.
    - Save per-trial scores and per-subject summaries.
    - Compare Human vs Bot totals (paired t-tests).
    - Compare Human vs Bot across social vs nonsocial topics.
    - Run a 2-way repeated-measures ANOVA: Condition × Sociality.
    - Save results (CSVs, plots, stats) to results/<model>/<temperature>/empath_tom/.

Inputs:
    - Per-subject transcript CSVs in data/<model>/<temperature>/{s###}.csv
    - topics.csv (with 'topic' and 'social' coding).

Outputs:
    - Trial-level and summary CSVs.
    - Statistical output text files (detailed, explicit).
    - Figures (violin plots with subject-level distributions).
    - Run log + config snapshot.

Usage:
    python code/analysis/empath_tom.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-29
"""

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from empath import Empath
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.formula.api import ols

from utils.globals import DATA_DIR, RESULTS_DIR, PROJECT_ROOT
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_warn, print_save, print_info

SCRIPT_NAME = "empath_tom"

# -------------------------------
# Empath setup
# -------------------------------
CUSTOM_TOM_WORDS = ["think", "know", "understand", "feel", "believe", "guess", "wonder", "remember"]


# -------------------------------
# Analysis
# -------------------------------

def analyze_empath_tom_language(sub_ids: list[str], data_dir: str, out_dir: str, topics_path: str) -> None:
    print_header("1) Empath Analysis — ToM Language by Condition")

    trials, summaries = [], []
    stats_lines = []

    # Initialize Empath once
    lex = Empath()
    lex.create_category("thinking_custom", CUSTOM_TOM_WORDS)
    topic_df = pd.read_csv(topics_path)
    topic_df.columns = topic_df.columns.str.strip()
    topic_df["topic"] = topic_df["topic"].astype(str).str.strip()

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

        # Merge topic info
        df["topic"] = df["topic"].astype(str).str.strip()
        df = df.merge(topic_df, on="topic", how="left")
        df["social"] = df["social"].fillna(-1)
        df["Social_Type"] = df["social"].map({1: "social", 0: "nonsocial"})

        # Compute Empath ToM score
        df["ToM_score"] = df["transcript_sub"].astype(str).apply(
            lambda text: lex.analyze(text, categories=["thinking_custom"])["thinking_custom"]
        )

        trials.append(df)

        # Aggregate subject totals
        total = df.groupby("Condition")["ToM_score"].mean().to_dict()
        summaries.append({
            "Subject": sub_id,
            "Hum": total.get("hum", np.nan),
            "Bot": total.get("bot", np.nan),
        })
        print_info(f"{sub_id}: mean ToM score (hum={total.get('hum', 0):.4f}, bot={total.get('bot', 0):.4f})")

    if not trials:
        print_warn("No data available — analysis aborted.")
        return

    # Save interaction-level data
    trial_df = pd.concat(trials, ignore_index=True)
    out_interactions = os.path.join(out_dir, "tom_empath_by_interaction.csv")
    trial_df.to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV")

    # Save subject-level totals
    totals_df = pd.DataFrame(summaries)
    out_totals = os.path.join(out_dir, "tom_empath_subject_summary.csv")
    totals_df.to_csv(out_totals, index=False)
    print_save(out_totals, kind="CSV")

    # Add Condition × Social summaries
    trial_df["Cond_Social"] = trial_df["Condition"] + "_" + trial_df["Social_Type"]
    cond_social_summary = trial_df.groupby(["Subject", "Cond_Social"])["ToM_score"].mean().unstack()
    out_social = os.path.join(out_dir, "tom_empath_subject_social_summary.csv")
    cond_social_summary.reset_index().to_csv(out_social, index=False)
    print_save(out_social, kind="CSV")

    # -------------------------------
    # Paired t-tests
    # -------------------------------
    stats_lines.append("--- Paired-samples t-tests: ToM Empath Scores ---")

    # Collapsed Human vs Bot
    if "Hum" in totals_df.columns and "Bot" in totals_df.columns:
        x, y = paired_clean(totals_df["Hum"], totals_df["Bot"])
        if x.size > 0:
            t_stat, p_val = ttest_rel(x, y, nan_policy="omit")
            m1, m2 = x.mean(), y.mean()
            sem1, sem2 = x.std(ddof=1) / np.sqrt(len(x)), y.std(ddof=1) / np.sqrt(len(y))
            stats_lines.append(
                f"Collapsed Human vs Bot:\n"
                f"  N={len(x)} paired subjects\n"
                f"  Mean Human = {m1:.3f} ± {sem1:.3f} SEM\n"
                f"  Mean Bot   = {m2:.3f} ± {sem2:.3f} SEM\n"
                f"  t({len(x)-1}) = {t_stat:.3f}, p = {p_val:.4f}\n"
                f"  Interpretation: {'Significant (p < .05)' if p_val < 0.05 else 'No significant difference (p > .05)'}\n"
            )

    # Condition × Social comparisons
    comparisons = [
        ("hum_social", "hum_nonsocial"),
        ("bot_social", "bot_nonsocial"),
        ("hum_social", "bot_social"),
        ("hum_nonsocial", "bot_nonsocial"),
    ]
    for a, b in comparisons:
        if a in cond_social_summary.columns and b in cond_social_summary.columns:
            x, y = paired_clean(cond_social_summary[a], cond_social_summary[b])
            if x.size > 0:
                t_stat, p_val = ttest_rel(x, y, nan_policy="omit")
                m1, m2 = x.mean(), y.mean()
                sem1, sem2 = x.std(ddof=1) / np.sqrt(len(x)), y.std(ddof=1) / np.sqrt(len(y))
                stats_lines.append(
                    f"{a} vs {b}:\n"
                    f"  N={len(x)} paired subjects\n"
                    f"  Mean {a} = {m1:.3f} ± {sem1:.3f} SEM\n"
                    f"  Mean {b} = {m2:.3f} ± {sem2:.3f} SEM\n"
                    f"  t({len(x)-1}) = {t_stat:.3f}, p = {p_val:.4f}\n"
                    f"  Interpretation: {'Significant (p < .05)' if p_val < 0.05 else 'No significant difference (p > .05)'}\n"
                )
        else:
            stats_lines.append(f"{a} vs {b}: Data missing")

    # -------------------------------
    # Two-way repeated-measures ANOVA
    # -------------------------------
    stats_lines.append("\n--- 2-Way Repeated-Measures ANOVA (Condition × Sociality) ---")
    long_anova_df = trial_df[trial_df["Social_Type"].isin(["social", "nonsocial"])].copy()
    long_anova_df = long_anova_df.groupby(["Subject", "Condition", "Social_Type"])["ToM_score"].mean().reset_index()

    pivot_check = long_anova_df.pivot_table(index="Subject", columns=["Condition", "Social_Type"], values="ToM_score")
    if pivot_check.isnull().any().any():
        stats_lines.append("⚠ Warning: Some subjects missing condition × topic data — excluded from ANOVA.\n")

    model = ols("ToM_score ~ C(Condition) * C(Social_Type) + C(Subject)", data=long_anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    stats_lines.append(anova_table.to_string())

    # Save stats
    stats_path = os.path.join(out_dir, "tom_empath_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines))
    print_save(stats_path, kind="stats")

    # Plot
    long_df = cond_social_summary.reset_index().melt(id_vars="Subject", var_name="Cond_Social", value_name="ToM_score")
    palette = {
        "hum_nonsocial": "skyblue",
        "hum_social": "steelblue",
        "bot_nonsocial": "sandybrown",
        "bot_social": "peru",
    }
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=long_df, x="Cond_Social", y="ToM_score", palette=palette, inner="box")
    plt.title("Empath ToM Score by Condition and Topic Type")
    plt.ylabel("Average ToM Score")
    plt.xlabel("Condition × Topic")
    plt.tight_layout()
    out_fig = os.path.join(out_dir, "tom_empath_violinplot.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print_save(out_fig, kind="figure")


# -------------------------------
# Main
# -------------------------------

def main():
    args, cfg = parse_and_load_config("Empath ToM analysis")

    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "empath_tom")  # ✅ results/model/temp/empath_tom
    topics_path = os.path.join(PROJECT_ROOT, "data/conds/topics.csv")
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_empath_tom_language(subjects, data_dir, out_dir, topics_path)

    logger.info("✅ Empath ToM analysis complete.")
    print("\n[DONE] ✅ Empath ToM analysis complete.\n")


if __name__ == "__main__":
    main()
