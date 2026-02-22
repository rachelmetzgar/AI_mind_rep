#!/usr/bin/env python3
"""
Script: behavioral_analysis.py

Purpose: Compute linguistic feature profiles on Experiment 2b steered conversation
data and compare across conditions (baseline, human-steered, AI-steered).

Adapts the Experiment 1 cross_experiment_comparison.py pipeline but:
- Operates only on Exp 2b data (no human experiment data, no cross-experiment stats)
- Handles 3 conditions: baseline, human, ai (instead of hum/bot)
- Supports both V1 (single-turn test questions) and V2 (multi-turn Exp 1 recreation) output formats
- References Experiment 1 utils directly (no code duplication)
- Runs within-experiment ANOVAs: Condition (3) or Condition (3) × Sociality (2)

Usage:
    # V1 output (single-turn test questions)
    python behavioral_analysis_exp2b.py --input data/intervention_results/v1_test_questions/intervention_responses.csv --version v1

    # V2 output (multi-turn Exp 1 recreation, per-subject CSVs)
    python behavioral_analysis_exp2b.py --input data/intervention_results/v2_exp1_recreation/per_subject --version v2

    # V2 with sociality info (requires topics.csv)
    python behavioral_analysis_exp2b.py --input data/intervention_results/v2_exp1_recreation/per_subject --version v2 --topics /path/to/topics.csv

Output saved to: data/intervention_results/{version}/behavioral_results/

Env: behavior_env

Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_rel, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Import Experiment 1 utils (shared linguistic markers)
# ============================================================

# Add Experiment 1 code to path for shared utils
EXP1_CODE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "exp_1", "code", "data_gen")
)
if EXP1_CODE_DIR not in sys.path:
    sys.path.insert(0, EXP1_CODE_DIR)

from utils.hedges_demir import (
    DEMIR_ALL_HEDGES, DEMIR_NOUNS, DEMIR_ADJECTIVES, DEMIR_ADVERBS,
    DEMIR_VERBS, DEMIR_QUANTIFIERS, DEMIR_MODALS
)
from utils.discourse_markers_fung import (
    FUNG_INTERPERSONAL, FUNG_REFERENTIAL, FUNG_STRUCTURAL, FUNG_COGNITIVE,
    FUNG_ALL_23_MARKERS
)
from utils.misc_text_markers import (
    LIWC_NONFLUENCIES, LIWC_FILLERS, LIWC_DISFLUENCIES,
    TOM_PHRASES, POLITE_POSITIVE, POLITE_NEGATIVE, IMPOLITE, LIKE_MARKER
)


# ============================================================
#                    CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavioral analysis of Experiment 2b steered conversations."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to V1 CSV or V2 per_subject directory.",
    )
    parser.add_argument(
        "--version", required=True, choices=["v1", "v2"],
        help="V1 = single-turn test questions; V2 = multi-turn Exp 1 recreation.",
    )
    parser.add_argument(
        "--topics", default=None,
        help="Path to topics.csv for social/nonsocial classification (V2 only).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory. Default: sibling of input path.",
    )
    return parser.parse_args()


# ============================================================
#                    DATA LOADING
# ============================================================

def load_v1_data(csv_path):
    """
    Load V1 intervention responses CSV.
    Expected columns: question_idx, question, condition, response
    """
    print(f"[INFO] Loading V1 data from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"question_idx", "question", "condition", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"V1 CSV missing columns: {missing}")

    # Rename to match pipeline expectations
    df = df.rename(columns={"response": "transcript_sub"})

    # V1 has no subjects — treat each question as a "trial" and
    # use a single dummy subject for within-condition stats
    df["subject"] = "s001"
    df["topic"] = df["question"].apply(lambda q: q[:50])  # truncated as label

    print(f"  Loaded {len(df)} rows, conditions: {df['condition'].value_counts().to_dict()}")
    return df


def load_v2_data(input_dir):
    """
    Load V2 per-subject CSVs.
    Expected columns: subject, run, order, trial, condition, topic, topic_file,
                      pair_index, transcript_sub, transcript_llm
    """
    print(f"[INFO] Loading V2 data from {input_dir}")

    csv_files = sorted(glob.glob(os.path.join(input_dir, "s*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No subject CSVs found in {input_dir}")

    all_dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        all_dfs.append(df)
        print(f"  Loaded {os.path.basename(path)}: {len(df)} rows")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total: {len(combined)} rows, {combined['subject'].nunique()} subjects")
    print(f"  Conditions: {combined['condition'].value_counts().to_dict()}")
    return combined


def add_social_classification(df, topics_path=None):
    """Add social/nonsocial classification from topics.csv if available."""
    if topics_path and os.path.exists(topics_path):
        print(f"[INFO] Loading topic classification from {topics_path}")
        topic_df = pd.read_csv(topics_path)
        topic_df.columns = topic_df.columns.str.strip().str.lower()
        topic_df["topic"] = topic_df["topic"].astype(str).str.strip().str.lower()

        df["topic_lower"] = df["topic"].astype(str).str.strip().str.lower()
        df = df.merge(
            topic_df[["topic", "social"]],
            left_on="topic_lower", right_on="topic",
            how="left", suffixes=("", "_from_file"),
        )
        if "topic_from_file" in df.columns:
            df = df.drop(columns=["topic_from_file"])
        df["social"] = df.get("social", pd.Series(dtype=float))
        if "social_from_file" in df.columns:
            df["social"] = df["social_from_file"].combine_first(df["social"])
            df = df.drop(columns=["social_from_file"])

        df["social"] = df["social"].fillna(0).astype(int)
        df["social_type"] = df["social"].map({1: "social", 0: "nonsocial"})
        counts = df["social_type"].value_counts()
        print(f"  Classification: {counts.get('social', 0)} social, {counts.get('nonsocial', 0)} nonsocial")
    else:
        print("[INFO] No topics file — skipping social/nonsocial classification.")
        df["social_type"] = "all"

    return df


# ============================================================
#                    FEATURE COMPUTATION
# ============================================================

def _count_words(text):
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def count_patterns(text, patterns):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def compute_all_metrics(df):
    """Compute all linguistic metrics on transcript_sub column."""
    print("[INFO] Computing linguistic metrics...")
    df = df.copy()

    df["word_count"] = df["transcript_sub"].apply(_count_words)
    df["question_count"] = df["transcript_sub"].apply(
        lambda x: str(x).count("?") if isinstance(x, str) else 0
    )

    # Demir (2018) hedge taxonomy
    for cat, patterns in [
        ("modal", DEMIR_MODALS), ("verb", DEMIR_VERBS),
        ("adverb", DEMIR_ADVERBS), ("adjective", DEMIR_ADJECTIVES),
        ("quantifier", DEMIR_QUANTIFIERS), ("noun", DEMIR_NOUNS),
    ]:
        col = f"demir_{cat}_count"
        df[col] = df["transcript_sub"].apply(lambda x: count_patterns(x, patterns))
        df[f"demir_{cat}_rate"] = df[col] / df["word_count"].replace(0, np.nan)

    df["demir_total_count"] = sum(
        df[f"demir_{c}_count"]
        for c in ["modal", "verb", "adverb", "adjective", "quantifier", "noun"]
    )
    df["demir_total_rate"] = df["demir_total_count"] / df["word_count"].replace(0, np.nan)

    # Fung & Carter (2007) discourse markers
    for cat, patterns in [
        ("interpersonal", FUNG_INTERPERSONAL), ("referential", FUNG_REFERENTIAL),
        ("structural", FUNG_STRUCTURAL), ("cognitive", FUNG_COGNITIVE),
        ("total", FUNG_ALL_23_MARKERS),
    ]:
        col = f"fung_{cat}_count"
        df[col] = df["transcript_sub"].apply(lambda x: count_patterns(x, patterns))
        df[f"fung_{cat}_rate"] = df[col] / df["word_count"].replace(0, np.nan)

    # LIWC disfluencies
    df["nonfluency_count"] = df["transcript_sub"].apply(lambda x: count_patterns(x, LIWC_NONFLUENCIES))
    df["nonfluency_rate"] = df["nonfluency_count"] / df["word_count"].replace(0, np.nan)
    df["liwc_filler_count"] = df["transcript_sub"].apply(lambda x: count_patterns(x, LIWC_FILLERS))
    df["liwc_filler_rate"] = df["liwc_filler_count"] / df["word_count"].replace(0, np.nan)
    df["disfluency_count"] = df["nonfluency_count"] + df["liwc_filler_count"]
    df["disfluency_rate"] = df["disfluency_count"] / df["word_count"].replace(0, np.nan)

    # ToM, politeness, like
    df["tom_count"] = df["transcript_sub"].apply(lambda x: count_patterns(x, TOM_PHRASES))
    df["tom_rate"] = df["tom_count"] / df["word_count"].replace(0, np.nan)

    df["polite_pos"] = df["transcript_sub"].apply(lambda x: count_patterns(x, POLITE_POSITIVE))
    df["polite_neg"] = df["transcript_sub"].apply(lambda x: count_patterns(x, POLITE_NEGATIVE))
    df["impolite"] = df["transcript_sub"].apply(lambda x: count_patterns(x, IMPOLITE))
    df["politeness_score"] = df["polite_pos"] + df["polite_neg"] - df["impolite"]
    df["politeness_rate"] = df["politeness_score"] / df["word_count"].replace(0, np.nan)

    df["like_count"] = df["transcript_sub"].apply(
        lambda x: len(re.findall(LIKE_MARKER, x.lower())) if isinstance(x, str) else 0
    )
    df["like_rate"] = df["like_count"] / df["word_count"].replace(0, np.nan)

    # VADER sentiment
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        df["sentiment"] = df["transcript_sub"].apply(
            lambda x: sia.polarity_scores(str(x)).get("compound", np.nan) if isinstance(x, str) else np.nan
        )
        print("  [OK] VADER sentiment computed.")
    except (ImportError, LookupError):
        df["sentiment"] = np.nan
        print("  [SKIP] VADER not available.")

    print(f"  [OK] Metrics computed on {len(df)} rows.")
    return df


# ============================================================
#                    AGGREGATION
# ============================================================

def aggregate_to_trial_level(df):
    """Aggregate utterances to trial level (V2: sum within each conversation)."""
    print("[INFO] Aggregating to trial level...")

    count_cols = [c for c in df.columns if c.endswith("_count") or c in ["word_count", "question_count", "politeness_score"]]
    groupby_cols = [c for c in ["subject", "condition", "topic", "social_type", "trial"] if c in df.columns]

    agg_dict = {col: "sum" for col in count_cols if col in df.columns}
    if "sentiment" in df.columns:
        agg_dict["sentiment"] = "mean"

    trial_df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)

    # Recompute rates
    for col in trial_df.columns:
        if col.endswith("_count") and col != "word_count":
            rate_col = col.replace("_count", "_rate")
            trial_df[rate_col] = trial_df[col] / trial_df["word_count"].replace(0, np.nan)
    if "politeness_score" in trial_df.columns:
        trial_df["politeness_rate"] = trial_df["politeness_score"] / trial_df["word_count"].replace(0, np.nan)

    print(f"  {len(df)} utterances → {len(trial_df)} trials")
    return trial_df


def aggregate_to_subject_condition(df, metrics):
    """Aggregate to subject × condition (mean across trials/utterances)."""
    return df.groupby(["subject", "condition"])[metrics].mean().reset_index()


def aggregate_to_subject_condition_social(df, metrics):
    """Aggregate to subject × condition × sociality."""
    if "social_type" not in df.columns or df["social_type"].nunique() <= 1:
        return None
    return df.groupby(["subject", "condition", "social_type"])[metrics].mean().reset_index()


# ============================================================
#                    STATISTICAL TESTS
# ============================================================

def run_condition_anova(agg_df, metric):
    """
    One-way RM-ANOVA across 3 conditions (baseline, human, ai).
    Falls back to paired t-tests if only 2 conditions or RM-ANOVA fails.
    """
    result = {"metric": metric}
    conditions = sorted(agg_df["condition"].unique())
    result["conditions"] = conditions

    try:
        # Pivot to wide format
        wide = agg_df.pivot(index="subject", columns="condition", values=metric).dropna()
        if len(wide) < 3:
            result["error"] = f"Too few subjects with complete data (n={len(wide)})"
            return result

        result["n_subjects"] = len(wide)

        # Condition means
        for cond in conditions:
            if cond in wide.columns:
                result[f"{cond}_mean"] = wide[cond].mean()
                result[f"{cond}_sem"] = wide[cond].sem()

        # RM-ANOVA
        long = wide.reset_index().melt(id_vars="subject", var_name="condition", value_name="value")
        try:
            aovrm = AnovaRM(long, "value", "subject", within=["condition"])
            res = aovrm.fit()
            result["F"] = res.anova_table["F Value"].iloc[0]
            result["p"] = res.anova_table["Pr > F"].iloc[0]
            result["df_num"] = res.anova_table["Num DF"].iloc[0]
            result["df_den"] = res.anova_table["Den DF"].iloc[0]
        except Exception as e:
            result["anova_error"] = str(e)
            # Fallback: pairwise t-tests
            result["F"] = np.nan
            result["p"] = np.nan

        # Pairwise paired t-tests
        pairs = []
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                if c1 in wide.columns and c2 in wide.columns:
                    t, p = ttest_rel(wide[c1], wide[c2])
                    pairs.append({
                        "pair": f"{c1}_vs_{c2}",
                        "t": t, "p": p,
                        "diff_mean": (wide[c1] - wide[c2]).mean(),
                    })
        result["pairwise"] = pairs

    except Exception as e:
        result["error"] = str(e)

    return result


def run_condition_by_sociality_anova(agg_df, metric):
    """2-way RM-ANOVA: Condition (3) × Sociality (2)."""
    result = {"metric": metric}

    if agg_df is None or "social_type" not in agg_df.columns:
        result["error"] = "No sociality data"
        return result

    try:
        # Need balanced data: each subject has all condition × sociality combos
        required_cells = set()
        for cond in agg_df["condition"].unique():
            for soc in agg_df["social_type"].unique():
                required_cells.add((cond, soc))

        complete_subs = []
        for sub in agg_df["subject"].unique():
            sub_data = agg_df[agg_df["subject"] == sub]
            sub_cells = set(zip(sub_data["condition"], sub_data["social_type"]))
            if required_cells.issubset(sub_cells):
                complete_subs.append(sub)

        if len(complete_subs) < 3:
            result["error"] = f"Too few complete subjects (n={len(complete_subs)})"
            return result

        df_complete = agg_df[agg_df["subject"].isin(complete_subs)].copy()
        result["n_subjects"] = len(complete_subs)

        aovrm = AnovaRM(
            df_complete, metric, "subject",
            within=["condition", "social_type"],
        )
        res = aovrm.fit()

        for row_name in res.anova_table.index:
            key = row_name.strip().lower().replace(" ", "_").replace(":", "_x_")
            result[f"{key}_F"] = res.anova_table.loc[row_name, "F Value"]
            result[f"{key}_p"] = res.anova_table.loc[row_name, "Pr > F"]

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
#                    OUTPUT FORMATTING
# ============================================================

def format_results(condition_results, sociality_results, version):
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT 2b BEHAVIORAL ANALYSIS ({version.upper()})")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Condition-only results
    lines.append("\n" + "=" * 80)
    lines.append("ONE-WAY RM-ANOVA: Effect of Steering Condition")
    lines.append("Conditions: baseline, human-steered, AI-steered")
    lines.append("=" * 80)

    for r in condition_results:
        lines.append(f"\n{'-' * 60}")
        lines.append(f"{r['metric']} (N = {r.get('n_subjects', '?')})")
        lines.append(f"{'-' * 60}")

        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
            continue

        # Condition means
        for cond in r.get("conditions", []):
            m = r.get(f"{cond}_mean", np.nan)
            se = r.get(f"{cond}_sem", np.nan)
            lines.append(f"  {cond:>10}: M = {m:.4f} ± {se:.4f}")

        # Omnibus F
        F = r.get("F", np.nan)
        p = r.get("p", np.nan)
        sig = ""
        if not np.isnan(p):
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
        df_num = r.get("df_num", "?")
        df_den = r.get("df_den", "?")
        lines.append(f"\n  Omnibus: F({df_num}, {df_den}) = {F:.3f}, p = {p:.4f} {sig}")

        if "anova_error" in r:
            lines.append(f"  (RM-ANOVA failed: {r['anova_error']}; pairwise tests below)")

        # Pairwise
        if "pairwise" in r:
            lines.append(f"\n  Pairwise comparisons (paired t-test):")
            for pw in r["pairwise"]:
                pw_sig = ""
                if pw["p"] < 0.001: pw_sig = "***"
                elif pw["p"] < 0.01: pw_sig = "**"
                elif pw["p"] < 0.05: pw_sig = "*"
                lines.append(
                    f"    {pw['pair']:>25}: diff = {pw['diff_mean']:+.4f}, "
                    f"t = {pw['t']:.3f}, p = {pw['p']:.4f} {pw_sig}"
                )

    # Condition × Sociality results
    if sociality_results and any("error" not in r for r in sociality_results):
        lines.append("\n\n" + "=" * 80)
        lines.append("TWO-WAY RM-ANOVA: Condition (3) × Sociality (2)")
        lines.append("=" * 80)

        for r in sociality_results:
            lines.append(f"\n{'-' * 60}")
            lines.append(f"{r['metric']} (N = {r.get('n_subjects', '?')})")
            lines.append(f"{'-' * 60}")

            if "error" in r:
                lines.append(f"  ERROR: {r['error']}")
                continue

            for key in sorted(r.keys()):
                if key.endswith("_F"):
                    effect = key[:-2]
                    F_val = r[key]
                    p_val = r.get(f"{effect}_p", np.nan)
                    sig = ""
                    if not np.isnan(p_val):
                        if p_val < 0.001: sig = "***"
                        elif p_val < 0.01: sig = "**"
                        elif p_val < 0.05: sig = "*"
                    lines.append(f"  {effect}: F = {F_val:.3f}, p = {p_val:.4f} {sig}")

    # Summary table
    lines.append("\n\n" + "=" * 90)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 90)
    lines.append(f"\n{'Metric':<30} {'Baseline':<14} {'Human':<14} {'AI':<14} {'F':<10} {'p':<10}")
    lines.append("-" * 90)

    for r in condition_results:
        if "error" in r:
            continue
        metric = r["metric"]
        bl = f"{r.get('baseline_mean', np.nan):.4f}"
        hu = f"{r.get('human_mean', np.nan):.4f}"
        ai = f"{r.get('ai_mean', np.nan):.4f}"
        F = f"{r.get('F', np.nan):.3f}"
        p = r.get("p", np.nan)
        p_str = f"{p:.4f}"
        sig = ""
        if not np.isnan(p):
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
        lines.append(f"{metric:<30} {bl:<14} {hu:<14} {ai:<14} {F:<10} {p_str + sig:<10}")

    lines.append("-" * 90)
    lines.append("* p < .05, ** p < .01, *** p < .001")

    return "\n".join(lines)


# ============================================================
#                    MAIN
# ============================================================

def main():
    args = parse_args()

    # Load data
    if args.version == "v1":
        df = load_v1_data(args.input)
    else:
        df = load_v2_data(args.input)
        df = add_social_classification(df, args.topics)

    # Compute metrics
    df = compute_all_metrics(df)

    # Define metrics to analyze
    all_metrics = [
        "word_count", "question_count",
        # Demir hedges
        "demir_modal_rate", "demir_verb_rate", "demir_adverb_rate",
        "demir_adjective_rate", "demir_quantifier_rate", "demir_noun_rate",
        "demir_total_rate",
        # Fung & Carter discourse markers
        "fung_interpersonal_rate", "fung_referential_rate",
        "fung_structural_rate", "fung_cognitive_rate", "fung_total_rate",
        # LIWC disfluencies
        "nonfluency_rate", "liwc_filler_rate", "disfluency_rate",
        # Other
        "like_rate", "tom_rate", "politeness_rate",
    ]
    if "sentiment" in df.columns and df["sentiment"].notna().any():
        all_metrics.append("sentiment")

    # Output directory
    if args.output:
        out_dir = args.output
    else:
        if args.version == "v1":
            out_dir = os.path.join(os.path.dirname(args.input), "behavioral_results")
        else:
            out_dir = os.path.join(os.path.dirname(args.input), "..", "behavioral_results")
    os.makedirs(out_dir, exist_ok=True)

    # Save utterance-level data
    utt_path = os.path.join(out_dir, "utterance_level_metrics.csv")
    df.to_csv(utt_path, index=False)
    print(f"[SAVED] {utt_path}")

    # ================================================================
    # Aggregation depends on version
    # ================================================================

    if args.version == "v2":
        # V2: aggregate utterances → trials → subject × condition
        trial_df = aggregate_to_trial_level(df)
        trial_path = os.path.join(out_dir, "trial_level_metrics.csv")
        trial_df.to_csv(trial_path, index=False)
        print(f"[SAVED] {trial_path}")

        agg_df = aggregate_to_subject_condition(trial_df, all_metrics)
        agg_social_df = aggregate_to_subject_condition_social(trial_df, all_metrics)
    else:
        # V1: no trial structure — aggregate directly by condition
        # (single subject, so stats are limited)
        agg_df = aggregate_to_subject_condition(df, all_metrics)
        agg_social_df = None

    agg_path = os.path.join(out_dir, "subject_condition_means.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"[SAVED] {agg_path}")

    # ================================================================
    # Statistical tests
    # ================================================================

    print("\n[INFO] Running condition RM-ANOVAs...")
    condition_results = [run_condition_anova(agg_df, m) for m in all_metrics]

    sociality_results = []
    if agg_social_df is not None:
        print("[INFO] Running condition × sociality RM-ANOVAs...")
        sociality_results = [run_condition_by_sociality_anova(agg_social_df, m) for m in all_metrics]

    # ================================================================
    # Output
    # ================================================================

    report = format_results(condition_results, sociality_results, args.version)

    stats_path = os.path.join(out_dir, f"behavioral_stats_{args.version}.txt")
    with open(stats_path, "w") as f:
        f.write(report)
    print(f"\n[SAVED] {stats_path}")

    # Print summary
    print("\n" + report)

    print(f"\n{'='*60}")
    print(f"[DONE] Results saved to {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()