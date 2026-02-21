"""
Script name: generic_analysis.py
Purpose: Centralized, reusable behavioral analysis engine.
    - Handles shared setup (loading, merging, logging, directories)
    - Delegates metric computation to external user-defined functions
    - Performs:
        * Aggregation (per subject / condition / topic)
        * Paired t-tests
        * 2-way repeated-measures ANOVA
        * Plot generation
        * Optional extra output subdirectories (extra_dir)
Usage:
    from generic_analysis import run_generic_main
    run_generic_main(SCRIPT_NAME, HEADER, feature_func, METRIC_COL, extra_dir="compound")

Author: Rachel C. Metzgar
Date: 2025-11-10
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sys

from utils.globals import DATA_DIR, RESULTS_DIR
from utils.run_logger import init_run
from utils.cli_helpers import parse_and_load_config
from utils.stats_helpers import paired_clean
from utils.print_helpers import print_header, print_info, print_save
from utils.plot_helpers import plot_violin_basic, main_effect_violin_lines


# ============================================================
#                LOAD STANDARD EXPERIMENT DATA
# ============================================================

def load_experiment_data(model: str, temp: str) -> pd.DataFrame:
    """Load, merge, and standardize experiment-wide CSVs."""
    data_dir = os.path.join(DATA_DIR, model, str(temp))
    combined_path = os.path.join(data_dir, "combined_text_data.csv")
    topics_path = os.path.join(DATA_DIR, "conds", "topics.csv")
    
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Missing combined CSV: {combined_path}")
    if not os.path.exists(topics_path):
        raise FileNotFoundError(f"Missing topics.csv: {topics_path}")

    df = pd.read_csv(combined_path, on_bad_lines="skip")
    topic_df = pd.read_csv(topics_path)

    df.columns = df.columns.str.strip()
    topic_df.columns = topic_df.columns.str.strip()
    df = df.merge(topic_df, on="topic", how="left")

    df["Social_Type"] = df["social"].map({1: "social", 0: "nonsocial"})
    df["Condition"] = df["agent"].astype(str).str.extract(r"(hum|bot)", expand=False)
    df["Subject"] = df["subject"].astype(str)
    return df


# ============================================================
#                  CORE ANALYSIS PIPELINE
# ============================================================

def run_generic_analysis(df: pd.DataFrame, SCRIPT_NAME: str, HEADER: str, METRIC_COL: str, out_dir: str):
    """Run shared stats, save outputs, and generate plots."""
    print_header(HEADER)

    # Check required columns
    required = {"Subject", "Condition", "Social_Type", METRIC_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ------------------------------------------------------------
    # 1. Save per-trial data
    # ------------------------------------------------------------
    trial_path = os.path.join(out_dir, f"{SCRIPT_NAME}_by_interaction.csv")
    df.to_csv(trial_path, index=False)
    print_save(trial_path, kind="CSV (trial-level)")

    # ------------------------------------------------------------
    # 2. Aggregations
    # ------------------------------------------------------------
    summary = df.groupby(["Subject", "Condition"])[METRIC_COL].mean().unstack().reset_index()
    social_summary = df.groupby(["Subject", "Condition", "Social_Type"])[METRIC_COL].mean().reset_index()
    df["Cond_Social"] = df["Condition"] + "_" + df["Social_Type"]
    cond_social_summary = df.groupby(["Subject", "Cond_Social"])[METRIC_COL].mean().unstack()

    # ✅ Save subject-level summaries
    out_summary = os.path.join(out_dir, f"{SCRIPT_NAME}_subject_summary.csv")
    summary.to_csv(out_summary, index=False)
    print_save(out_summary, kind="CSV (subject summary)")

    out_social = os.path.join(out_dir, f"{SCRIPT_NAME}_subject_social_summary.csv")
    cond_social_summary.reset_index().to_csv(out_social, index=False)
    print_save(out_social, kind="CSV (Condition × Sociality summary)")

    # ------------------------------------------------------------
    # 3. Statistical tests
    # ------------------------------------------------------------
    stats_lines = [f"--- {SCRIPT_NAME.title()} Statistical Summary ---"]
    p_val_main = None

    # Main paired t-test (Human vs Bot)
    if {"hum", "bot"}.issubset(summary.columns):
        x, y = paired_clean(summary["hum"], summary["bot"])
        if len(x) > 0:
            t_stat, p_val_main = ttest_rel(x, y, nan_policy="omit")
            m1, m2 = x.mean(), y.mean()
            sem1, sem2 = x.std(ddof=1)/np.sqrt(len(x)), y.std(ddof=1)/np.sqrt(len(y))
            stats_lines.append(
                f"\nHuman vs Bot:\n"
                f"  N={len(x)}\n"
                f"  Mean(Human)={m1:.4f} ± {sem1:.4f}, "
                f"Mean(Bot)={m2:.4f} ± {sem2:.4f}\n"
                f"  t({len(x)-1})={t_stat:.3f}, p={p_val_main:.4f}\n"
            )

    # Condition × Sociality comparisons
    stats_lines.append("\n--- Paired-samples t-tests: Condition × Sociality ---")
    comparisons = [
        ("hum_social", "hum_nonsocial"),
        ("bot_social", "bot_nonsocial"),
        ("hum_social", "bot_social"),
        ("hum_nonsocial", "bot_nonsocial"),
    ]
    for a, b in comparisons:
        if a in cond_social_summary.columns and b in cond_social_summary.columns:
            x, y = paired_clean(cond_social_summary[a], cond_social_summary[b])
            if len(x) > 0:
                t_stat, p_val = ttest_rel(x, y, nan_policy="omit")
                m1, m2 = np.nanmean(x), np.nanmean(y)
                sem1, sem2 = np.nanstd(x, ddof=1)/np.sqrt(len(x)), np.nanstd(y, ddof=1)/np.sqrt(len(y))
                stats_lines.append(
                    f"{a} vs {b}:\n"
                    f"  Mean {a}={m1:.4f} ± {sem1:.4f}, Mean {b}={m2:.4f} ± {sem2:.4f}\n"
                    f"  t({len(x)-1})={t_stat:.3f}, p={p_val:.4f}\n"
                )

    # 2-Way Repeated-Measures ANOVA
    stats_lines.append("\n--- 2-Way Repeated-Measures ANOVA (Condition × Sociality) ---")
    model = ols(f"{METRIC_COL} ~ C(Condition) * C(Social_Type) + C(Subject)", data=social_summary).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    stats_lines.append(anova_table.to_string())

    # Add cell means ± SEM
    stats_lines.append(f"\nCell means ± SEM (avg {SCRIPT_NAME} per condition/topic):")
    desc = social_summary.groupby(["Condition", "Social_Type"])[METRIC_COL].agg(["mean", "sem"]).round(4)
    stats_lines.append(desc.to_string())

    topic_means = social_summary.groupby("Social_Type")[METRIC_COL].agg(["mean", "sem"]).round(4)
    stats_lines.append("\n--- Collapsed by Topic Type (Social vs Nonsocial) ---")
    stats_lines.append(topic_means.to_string())

    stats_path = os.path.join(out_dir, f"{SCRIPT_NAME}_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines))
    print_save(stats_path, kind="stats")

    # ------------------------------------------------------------
    # 4. Plotting
    # ------------------------------------------------------------
    print_info("Generating plots...")

    long_df = cond_social_summary.reset_index().melt(
        id_vars="Subject", var_name="Cond_Social", value_name=METRIC_COL
    )
    plot_violin_basic(
        df_long=long_df,
        y_col=METRIC_COL,
        x_col="Cond_Social",
        out_path=os.path.join(out_dir, f"{SCRIPT_NAME}_violinplot.png"),
        title=f"{SCRIPT_NAME.title()} by Condition × Topic Type",
    )

    main_effect_plot = os.path.join(out_dir, f"{SCRIPT_NAME}_main_effect_lines.png")
    main_effect_violin_lines(
        df_summary=summary,
        cond_a="hum",
        cond_b="bot",
        y_col=METRIC_COL,
        out_path=main_effect_plot,
        title=f"Main Effect of Partner Type: Human vs Bot ({SCRIPT_NAME.title()})",
        p_val=p_val_main,
    )
    print_save(main_effect_plot, kind="main effect plot")


# ============================================================
#                  UNIFIED ENTRY POINT
# ============================================================

def run_generic_main(SCRIPT_NAME: str, HEADER: str, feature_func, METRIC_COL: str, extra_dir: str | None = None):
    """
    Unified entry point for all behavioral analyses.
    feature_func: callable(df) -> df
        Should return df with a new per-utterance metric column.
    extra_dir: optional subdirectory to save outputs (e.g., 'compound', 'positive', etc.)
    """
    args, cfg = parse_and_load_config(f"{SCRIPT_NAME.title()} analysis")
    model = cfg.get("model")
    temp = str(cfg.get("temperature"))
    
    base_out_dir = os.path.join(RESULTS_DIR, model, temp, SCRIPT_NAME)
    out_dir = os.path.join(base_out_dir, extra_dir) if extra_dir else base_out_dir
    os.makedirs(out_dir, exist_ok=True)

    logger, _, _, _ = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False
    )

    df = load_experiment_data(model, temp)
    df = feature_func(df)

    run_generic_analysis(df, SCRIPT_NAME, HEADER, METRIC_COL, out_dir)

    logger.info(f"✅ {SCRIPT_NAME.title()} analysis complete.")
    print(f"\n[DONE] ✅ {SCRIPT_NAME.title()} analysis complete.\n")
