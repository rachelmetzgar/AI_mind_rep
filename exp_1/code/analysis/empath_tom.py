"""
Script name: empath_tom.py
Purpose: Analyze Empath-derived Theory of Mind (ToM) language use in participant transcripts.
    - Reads from a combined CSV created by combine_text_data.py.
    - Computes Empath scores for custom "thinking_custom" category.
    - Detects ToM words in each interaction (count + which words).
    - Performs summary stats, paired t-tests, and a 2-way repeated-measures ANOVA.
    - Saves trial-level, detailed, and summary CSVs, plus plots and stats.

Inputs:
    - Combined CSV: data/<model>/<temperature>/combined_text_data.csv
    - topics.csv (with 'topic' and 'social' coding)

Outputs:
    - tom_empath_by_interaction.csv
    - tom_empath_interaction_detailed.csv
    - tom_empath_subject_summary.csv
    - tom_empath_subject_social_summary.csv
    - tom_empath_stats.txt
    - tom_empath_violinplot.png
    - tom_empath_main_effect_lines.png

Usage:
    python code/analysis/empath_tom.py --config configs/behavior.json

Author: Rachel C. Metzgar
Date: 2025-10-31
"""

from __future__ import annotations
import os
import re
import pandas as pd
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
from utils.plot_helpers import plot_violin_basic, main_effect_violin_lines, DEFAULT_PALETTE

SCRIPT_NAME = "empath_tom"
CUSTOM_TOM_WORDS = ["think", "know", "understand", "feel", "believe", "guess", "wonder", "remember"]


# ------------------------------------------------------------
# Core analysis
# ------------------------------------------------------------
def analyze_empath_tom_language(combined_path: str, out_dir: str, topics_path: str) -> None:
    print_header("1) Empath Analysis — ToM Language by Condition")

    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"Combined CSV not found: {combined_path}")

    # Load combined data
    df = pd.read_csv(combined_path, on_bad_lines="skip")
    df.columns = df.columns.str.strip()

    required = {"subject", "agent", "topic", "transcript_sub"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Combined CSV missing required columns: {missing}")

    # Initialize Empath
    lex = Empath()
    lex.create_category("thinking_custom", CUSTOM_TOM_WORDS)

    # Load and merge topics
    topic_df = pd.read_csv(topics_path)
    topic_df.columns = topic_df.columns.str.strip()
    topic_df["topic"] = topic_df["topic"].astype(str).str.strip()
    df["topic"] = df["topic"].astype(str).str.strip()
    df = df.merge(topic_df, on="topic", how="left")
    df["social"] = df["social"].fillna(-1)
    df["Social_Type"] = df["social"].map({1: "social", 0: "nonsocial"})

    # Derive condition labels
    df["Condition"] = df["agent"].astype(str).str.extract(r"(hum|bot)", expand=False)
    df["Subject"] = df["subject"].astype(str)

    # Compute Empath ToM scores
    print_info("Computing Empath ToM scores...")
    df["ToM_score"] = df["transcript_sub"].astype(str).apply(
        lambda text: lex.analyze(text, categories=["thinking_custom"])["thinking_custom"]
    )

    # Detect ToM words (count + which words)
    def detect_tom_words(text: str):
        text_l = str(text).lower()
        found = [w for w in CUSTOM_TOM_WORDS if f" {w} " in f" {text_l} "]
        return len(found), ", ".join(found) if found else ""

    df[["tom_word_count", "tom_words_used"]] = df["transcript_sub"].apply(
        lambda t: pd.Series(detect_tom_words(t))
    )

    # Save interaction-level data
    out_interactions = os.path.join(out_dir, "tom_empath_by_interaction.csv")
    df.to_csv(out_interactions, index=False)
    print_save(out_interactions, kind="CSV")

    # Detailed per-interaction file
    out_detailed = os.path.join(out_dir, "tom_empath_interaction_detailed.csv")
    cols = [
        "Subject", "agent", "topic", "Social_Type", "Condition",
        "transcript_sub", "ToM_score", "tom_word_count", "tom_words_used"
    ]
    df[cols].to_csv(out_detailed, index=False)
    print_save(out_detailed, kind="detailed CSV (per-interaction with ToM words)")

    # Per-subject summaries
    summaries = []
    for sub_id, g in df.groupby("Subject"):
        total = g.groupby("Condition")["ToM_score"].mean().to_dict()
        summaries.append({
            "Subject": sub_id,
            "Hum": total.get("hum", np.nan),
            "Bot": total.get("bot", np.nan),
        })
        print_info(f"{sub_id}: mean ToM score (hum={total.get('hum', 0):.4f}, bot={total.get('bot', 0):.4f})")

    totals_df = pd.DataFrame(summaries)
    out_totals = os.path.join(out_dir, "tom_empath_subject_summary.csv")
    totals_df.to_csv(out_totals, index=False)
    print_save(out_totals, kind="CSV")

    # Condition × Social summaries
    df["Cond_Social"] = df["Condition"] + "_" + df["Social_Type"]
    cond_social_summary = df.groupby(["Subject", "Cond_Social"])["ToM_score"].mean().unstack()
    out_social = os.path.join(out_dir, "tom_empath_subject_social_summary.csv")
    cond_social_summary.reset_index().to_csv(out_social, index=False)
    print_save(out_social, kind="CSV")

    # ------------------------------------------------------------
    # Stats & ANOVA
    # ------------------------------------------------------------
    stats_lines = []
    stats_lines.append("--- Paired-samples t-tests: ToM Empath Scores ---")

    # Collapsed Human vs Bot
    p_val_main = None
    if "Hum" in totals_df.columns and "Bot" in totals_df.columns:
        x, y = paired_clean(totals_df["Hum"], totals_df["Bot"])
        if x.size > 0:
            t_stat, p_val_main = ttest_rel(x, y, nan_policy="omit")
            m1, m2 = x.mean(), y.mean()
            sem1, sem2 = x.std(ddof=1)/np.sqrt(len(x)), y.std(ddof=1)/np.sqrt(len(x))
            stats_lines.append(
                f"Collapsed Human vs Bot:\n"
                f"  N={len(x)}\n"
                f"  Mean Human = {m1:.3f} ± {sem1:.3f}\n"
                f"  Mean Bot = {m2:.3f} ± {sem2:.3f}\n"
                f"  t({len(x)-1}) = {t_stat:.3f}, p = {p_val_main:.4f}\n"
            )

    # Topic-level comparisons
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
                stats_lines.append(f"{a} vs {b}: t={t_stat:.3f}, p={p_val:.4f}")
        else:
            stats_lines.append(f"{a} vs {b}: missing data")

    # ANOVA
    stats_lines.append("\n--- 2-Way Repeated-Measures ANOVA (Condition × Sociality) ---")
    long_anova_df = df[df["Social_Type"].isin(["social", "nonsocial"])].copy()
    long_anova_df = long_anova_df.groupby(["Subject", "Condition", "Social_Type"])["ToM_score"].mean().reset_index()
    model = ols("ToM_score ~ C(Condition) * C(Social_Type) + C(Subject)", data=long_anova_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    stats_lines.append(anova_table.to_string())

    stats_path = os.path.join(out_dir, "tom_empath_stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(stats_lines))
    print_save(stats_path, kind="stats")

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------
    # (1) Condition × Sociality violin
    long_df = cond_social_summary.reset_index().melt(
        id_vars="Subject", var_name="Cond_Social", value_name="ToM_score"
    )
    plot_violin_basic(
        df_long=long_df,
        y_col="ToM_score",
        x_col="Cond_Social",
        out_path=os.path.join(out_dir, "tom_empath_violinplot.png"),
        title="Empath ToM Score by Condition × Topic Type",
    )

    # (2) Main effect: Human vs Bot
    main_effect_violin_lines(
        df_summary=totals_df,
        cond_a="Hum",
        cond_b="Bot",
        y_col="ToM_score",
        out_path=os.path.join(out_dir, "tom_empath_main_effect_lines.png"),
        title="Main Effect of Partner Type: Human vs Bot",
        p_val=p_val_main,
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args, cfg = parse_and_load_config("Empath ToM analysis (combined)")
    model = cfg.get("model")
    temp = cfg.get("temperature")

    data_dir = os.path.join(DATA_DIR, model, str(temp))
    combined_path = os.path.join(data_dir, "combined_text_data.csv")
    out_dir = os.path.join(RESULTS_DIR, model, str(temp), "empath_tom")
    topics_path = os.path.join(PROJECT_ROOT, "data/conds/topics.csv")
    os.makedirs(out_dir, exist_ok=True)

    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    analyze_empath_tom_language(combined_path, out_dir, topics_path)

    logger.info("✅ Empath ToM analysis complete.")
    print("\n[DONE] ✅ Empath ToM analysis complete.\n")


if __name__ == "__main__":
    main()
