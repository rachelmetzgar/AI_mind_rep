#!/usr/bin/env python3
"""
Experiment 3: Behavioral Analysis of Concept-Steered Conversations

Computes linguistic feature profiles on concept-injection output and compares
across conditions (baseline, human-steered, AI-steered) for each concept
dimension.

Adapts the Exp 2b behavioral_analysis script but:
    - Loops through concept dimensions (--dim_id or --dim_ids or --all)
    - Loops through intervention strengths within each dimension
    - Produces per-dimension reports + cross-dimension comparison summary
    - Same linguistic features and statistical tests as Exp 2b

Usage:
    # Single dimension, single strength, V1
    python 4_behavioral_analysis.py --version v1 --dim_id 7 --strength 4

    # Single dimension, all strengths, V1
    python 4_behavioral_analysis.py --version v1 --dim_id 7

    # All dimensions, all strengths, V1
    python 4_behavioral_analysis.py --version v1 --all

    # V2 multi-turn
    python 4_behavioral_analysis.py --version v2 --dim_id 7 --strength 4 --topics /path/to/topics.csv

Output:
    data/intervention_results/V{1,2}/{dim_name}/is_{N}/behavioral_results/
    data/intervention_results/V{1,2}/cross_dimension_summary.csv

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
from scipy.stats import ttest_rel
import statsmodels.api as sm
from statsmodels.stats.anova import AnovaRM
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Import Experiment 1 utils (shared linguistic markers)
# ============================================================

EXP1_CODE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "exp_1", "code", "data_gen")
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

# Import dimension registry
sys.path.insert(0, os.path.dirname(__file__))
from importlib.util import spec_from_file_location, module_from_spec
_s1_spec = spec_from_file_location(
    "script1",
    os.path.join(os.path.dirname(__file__), "1_elicit_concept_vectors.py"),
)
_s1_mod = module_from_spec(_s1_spec)
_s1_spec.loader.exec_module(_s1_mod)
DIMENSION_REGISTRY = _s1_mod.DIMENSION_REGISTRY


# ============================================================
# CONFIG
# ============================================================

DEFAULT_STRENGTHS = [1, 2, 4, 8]
RESULT_ROOT_V1 = "data/intervention_results/V1"
RESULT_ROOT_V2 = "data/intervention_results/V2"

ALL_METRICS = [
    "word_count", "question_count",
    "demir_modal_rate", "demir_verb_rate", "demir_adverb_rate",
    "demir_adjective_rate", "demir_quantifier_rate", "demir_noun_rate",
    "demir_total_rate",
    "fung_interpersonal_rate", "fung_referential_rate",
    "fung_structural_rate", "fung_cognitive_rate", "fung_total_rate",
    "nonfluency_rate", "liwc_filler_rate", "disfluency_rate",
    "like_rate", "tom_rate", "politeness_rate",
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 3: Behavioral analysis of concept-steered conversations."
    )
    parser.add_argument(
        "--version", required=True, choices=["v1", "v2"],
    )
    parser.add_argument(
        "--dim_id", type=int, default=None,
        help="Single dimension ID (1-13).",
    )
    parser.add_argument(
        "--dim_ids", type=int, nargs="+", default=None,
        help="Multiple dimension IDs.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all dimensions.",
    )
    parser.add_argument(
        "--strength", type=int, default=None,
        help="Single intervention strength.",
    )
    parser.add_argument(
        "--strengths", type=int, nargs="+", default=None,
        help=f"Multiple strengths (default: {DEFAULT_STRENGTHS}).",
    )
    parser.add_argument(
        "--topics", default=None,
        help="Path to topics.csv for social/nonsocial classification (V2).",
    )
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_v1_data(csv_path):
    """Load V1 intervention_responses.csv."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={"response": "transcript_sub"})
    df["subject"] = "s001"
    df["topic"] = df["question"].apply(lambda q: str(q)[:50])
    return df


def load_v2_data(input_dir):
    """Load V2 per-subject CSVs from a per_subject directory."""
    csv_files = sorted(glob.glob(os.path.join(input_dir, "s*.csv")))
    if not csv_files:
        return None

    all_dfs = []
    for path in csv_files:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


def add_social_classification(df, topics_path=None):
    """Add social/nonsocial classification from topics.csv if available."""
    if topics_path and os.path.exists(topics_path):
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
    else:
        df["social_type"] = "all"
    return df


# ============================================================
# FEATURE COMPUTATION
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
    df = df.copy()

    df["word_count"] = df["transcript_sub"].apply(_count_words)
    df["question_count"] = df["transcript_sub"].apply(
        lambda x: str(x).count("?") if isinstance(x, str) else 0
    )

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

    for cat, patterns in [
        ("interpersonal", FUNG_INTERPERSONAL), ("referential", FUNG_REFERENTIAL),
        ("structural", FUNG_STRUCTURAL), ("cognitive", FUNG_COGNITIVE),
        ("total", FUNG_ALL_23_MARKERS),
    ]:
        col = f"fung_{cat}_count"
        df[col] = df["transcript_sub"].apply(lambda x: count_patterns(x, patterns))
        df[f"fung_{cat}_rate"] = df[col] / df["word_count"].replace(0, np.nan)

    df["nonfluency_count"] = df["transcript_sub"].apply(lambda x: count_patterns(x, LIWC_NONFLUENCIES))
    df["nonfluency_rate"] = df["nonfluency_count"] / df["word_count"].replace(0, np.nan)
    df["liwc_filler_count"] = df["transcript_sub"].apply(lambda x: count_patterns(x, LIWC_FILLERS))
    df["liwc_filler_rate"] = df["liwc_filler_count"] / df["word_count"].replace(0, np.nan)
    df["disfluency_count"] = df["nonfluency_count"] + df["liwc_filler_count"]
    df["disfluency_rate"] = df["disfluency_count"] / df["word_count"].replace(0, np.nan)

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

    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        df["sentiment"] = df["transcript_sub"].apply(
            lambda x: sia.polarity_scores(str(x)).get("compound", np.nan)
            if isinstance(x, str) else np.nan
        )
    except (ImportError, LookupError):
        df["sentiment"] = np.nan

    return df


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_to_trial_level(df):
    """V2: aggregate utterances within each conversation to trial level."""
    count_cols = [
        c for c in df.columns
        if c.endswith("_count") or c in ["word_count", "question_count", "politeness_score"]
    ]
    groupby_cols = [
        c for c in ["subject", "condition", "topic", "social_type", "trial"]
        if c in df.columns
    ]

    agg_dict = {col: "sum" for col in count_cols if col in df.columns}
    if "sentiment" in df.columns:
        agg_dict["sentiment"] = "mean"

    trial_df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)

    for col in trial_df.columns:
        if col.endswith("_count") and col != "word_count":
            rate_col = col.replace("_count", "_rate")
            trial_df[rate_col] = trial_df[col] / trial_df["word_count"].replace(0, np.nan)
    if "politeness_score" in trial_df.columns:
        trial_df["politeness_rate"] = (
            trial_df["politeness_score"] / trial_df["word_count"].replace(0, np.nan)
        )

    return trial_df


def aggregate_to_subject_condition(df, metrics):
    return df.groupby(["subject", "condition"])[metrics].mean().reset_index()


# ============================================================
# STATISTICAL TESTS
# ============================================================

def run_condition_anova(agg_df, metric):
    """One-way RM-ANOVA across conditions + pairwise t-tests."""
    result = {"metric": metric}
    conditions = sorted(agg_df["condition"].unique())
    result["conditions"] = conditions

    try:
        wide = agg_df.pivot(
            index="subject", columns="condition", values=metric,
        ).dropna()

        if len(wide) < 3:
            result["error"] = f"Too few subjects (n={len(wide)})"
            return result

        result["n_subjects"] = len(wide)

        for cond in conditions:
            if cond in wide.columns:
                result[f"{cond}_mean"] = wide[cond].mean()
                result[f"{cond}_sem"] = wide[cond].sem()

        long = wide.reset_index().melt(
            id_vars="subject", var_name="condition", value_name="value",
        )
        try:
            aovrm = AnovaRM(long, "value", "subject", within=["condition"])
            res = aovrm.fit()
            result["F"] = res.anova_table["F Value"].iloc[0]
            result["p"] = res.anova_table["Pr > F"].iloc[0]
        except Exception as e:
            result["anova_error"] = str(e)
            result["F"] = np.nan
            result["p"] = np.nan

        # Pairwise
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


def run_v1_between_subjects_tests(agg_df, metric):
    """
    V1 fallback: single subject means RM-ANOVA fails.
    Use per-question as observations, independent-samples approach.
    """
    result = {"metric": metric}
    conditions = sorted(agg_df["condition"].unique())
    result["conditions"] = conditions

    try:
        for cond in conditions:
            vals = agg_df[agg_df["condition"] == cond][metric].dropna()
            result[f"{cond}_mean"] = vals.mean()
            result[f"{cond}_sem"] = vals.sem()
            result[f"{cond}_n"] = len(vals)

        # Pairwise independent t-tests (questions as observations)
        from scipy.stats import ttest_ind
        pairs = []
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                v1 = agg_df[agg_df["condition"] == c1][metric].dropna()
                v2 = agg_df[agg_df["condition"] == c2][metric].dropna()
                t, p = ttest_ind(v1, v2)
                pairs.append({
                    "pair": f"{c1}_vs_{c2}",
                    "t": t, "p": p,
                    "diff_mean": v1.mean() - v2.mean(),
                })
        result["pairwise"] = pairs
        result["F"] = np.nan
        result["p"] = np.nan
        result["note"] = "V1: independent t-tests (questions as observations)"

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
# PER-DIMENSION ANALYSIS
# ============================================================

def analyze_single_run(df, version, metrics):
    """Run full analysis on one dataframe, return condition results."""
    df = compute_all_metrics(df)

    active_metrics = [m for m in metrics if m in df.columns]
    if "sentiment" in df.columns and df["sentiment"].notna().any():
        if "sentiment" not in active_metrics:
            active_metrics.append("sentiment")

    if version == "v2":
        trial_df = aggregate_to_trial_level(df)
        agg_df = aggregate_to_subject_condition(trial_df, active_metrics)

        if agg_df["subject"].nunique() >= 3:
            results = [run_condition_anova(agg_df, m) for m in active_metrics]
        else:
            results = [run_v1_between_subjects_tests(df, m) for m in active_metrics]
    else:
        # V1: use questions as observations
        results = [run_v1_between_subjects_tests(df, m) for m in active_metrics]

    return results, df


def extract_summary_row(dim_name, N, condition_results):
    """Extract one row for the cross-dimension summary table."""
    row = {"dimension": dim_name, "N": N}

    for r in condition_results:
        metric = r["metric"]
        for cond in ["baseline", "human", "ai"]:
            row[f"{metric}_{cond}_mean"] = r.get(f"{cond}_mean", np.nan)

        # Human vs AI difference (key comparison)
        h = r.get("human_mean", np.nan)
        a = r.get("ai_mean", np.nan)
        row[f"{metric}_human_minus_ai"] = h - a if not (np.isnan(h) or np.isnan(a)) else np.nan

        # Pairwise human vs ai p-value
        if "pairwise" in r:
            for pw in r["pairwise"]:
                if pw["pair"] == "human_vs_ai":
                    row[f"{metric}_hva_p"] = pw["p"]
                    row[f"{metric}_hva_t"] = pw["t"]

    return row


def format_single_report(dim_name, N, condition_results, version):
    """Format a text report for one dimension × strength."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT 3: CONCEPT INJECTION BEHAVIORAL ANALYSIS")
    lines.append(f"Dimension: {dim_name}  |  N = {N}  |  Version: {version.upper()}")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Summary table
    lines.append(f"\n{'Metric':<30} {'Baseline':<12} {'Human':<12} {'AI':<12} {'H-A diff':<12} {'p(H vs A)':<12}")
    lines.append("-" * 90)

    for r in condition_results:
        if "error" in r:
            continue
        metric = r["metric"]
        bl = r.get("baseline_mean", np.nan)
        hu = r.get("human_mean", np.nan)
        ai = r.get("ai_mean", np.nan)
        diff = hu - ai if not (np.isnan(hu) or np.isnan(ai)) else np.nan

        # Find human vs ai pairwise p
        p_hva = np.nan
        if "pairwise" in r:
            for pw in r["pairwise"]:
                if pw["pair"] == "human_vs_ai":
                    p_hva = pw["p"]

        sig = ""
        if not np.isnan(p_hva):
            if p_hva < 0.001: sig = "***"
            elif p_hva < 0.01: sig = "**"
            elif p_hva < 0.05: sig = "*"

        lines.append(
            f"{metric:<30} {bl:<12.4f} {hu:<12.4f} {ai:<12.4f} "
            f"{diff:<+12.4f} {p_hva:<8.4f} {sig}"
        )

    lines.append("-" * 90)
    lines.append("* p < .05, ** p < .01, *** p < .001")

    # Detailed pairwise
    lines.append(f"\n\nDETAILED PAIRWISE COMPARISONS")
    lines.append("=" * 80)

    for r in condition_results:
        if "error" in r or "pairwise" not in r:
            continue
        lines.append(f"\n  {r['metric']}:")
        for pw in r["pairwise"]:
            sig = ""
            if pw["p"] < 0.001: sig = "***"
            elif pw["p"] < 0.01: sig = "**"
            elif pw["p"] < 0.05: sig = "*"
            lines.append(
                f"    {pw['pair']:>25}: diff={pw['diff_mean']:+.4f}, "
                f"t={pw['t']:.3f}, p={pw['p']:.4f} {sig}"
            )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Resolve dimensions
    if args.all:
        dim_ids = sorted(DIMENSION_REGISTRY.keys())
    elif args.dim_ids:
        dim_ids = args.dim_ids
    elif args.dim_id:
        dim_ids = [args.dim_id]
    else:
        print("[ERROR] Must specify --dim_id, --dim_ids, or --all.")
        sys.exit(1)

    # Resolve strengths
    if args.strength is not None:
        strengths = [args.strength]
    elif args.strengths is not None:
        strengths = args.strengths
    else:
        strengths = DEFAULT_STRENGTHS

    result_root = RESULT_ROOT_V1 if args.version == "v1" else RESULT_ROOT_V2

    # Collect cross-dimension summary rows
    cross_dim_rows = []

    for dim_id in dim_ids:
        if dim_id not in DIMENSION_REGISTRY:
            print(f"[WARN] Unknown dim_id={dim_id}, skipping.")
            continue

        _, dim_name = DIMENSION_REGISTRY[dim_id]

        for N in strengths:
            print(f"\n{'#'*70}")
            print(f"# Dim {dim_id}: {dim_name}  |  N = {N}")
            print(f"{'#'*70}")

            if args.version == "v1":
                csv_path = os.path.join(
                    result_root, dim_name, f"is_{N}",
                    "intervention_responses.csv",
                )
                if not os.path.isfile(csv_path):
                    print(f"  [SKIP] Not found: {csv_path}")
                    continue
                df = load_v1_data(csv_path)

            else:  # v2
                per_sub_dir = os.path.join(
                    result_root, dim_name, f"is_{N}", "per_subject",
                )
                if not os.path.isdir(per_sub_dir):
                    print(f"  [SKIP] Not found: {per_sub_dir}")
                    continue
                df = load_v2_data(per_sub_dir)
                if df is None or len(df) == 0:
                    print(f"  [SKIP] No data in {per_sub_dir}")
                    continue
                df = add_social_classification(df, args.topics)

            print(f"  Loaded {len(df)} rows, "
                  f"conditions: {df['condition'].value_counts().to_dict()}")

            # Analyze
            condition_results, df_with_metrics = analyze_single_run(
                df, args.version, ALL_METRICS,
            )

            # Save outputs
            out_dir = os.path.join(
                result_root, dim_name, f"is_{N}", "behavioral_results",
            )
            os.makedirs(out_dir, exist_ok=True)

            # Utterance-level metrics
            utt_path = os.path.join(out_dir, "utterance_level_metrics.csv")
            df_with_metrics.to_csv(utt_path, index=False)

            # Report
            report = format_single_report(
                dim_name, N, condition_results, args.version,
            )
            report_path = os.path.join(out_dir, f"behavioral_stats.txt")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"  Saved: {report_path}")

            # Summary row
            row = extract_summary_row(dim_name, N, condition_results)
            cross_dim_rows.append(row)

            # Print quick summary
            sig_metrics = []
            for r in condition_results:
                if "pairwise" in r:
                    for pw in r["pairwise"]:
                        if pw["pair"] == "human_vs_ai" and pw["p"] < 0.05:
                            sig_metrics.append(r["metric"])
            if sig_metrics:
                print(f"  Significant H vs A: {', '.join(sig_metrics)}")
            else:
                print(f"  No significant H vs A differences.")

    # ================================================================
    # Cross-dimension summary
    # ================================================================

    if cross_dim_rows:
        summary_df = pd.DataFrame(cross_dim_rows)
        summary_path = os.path.join(
            result_root, "cross_dimension_summary.csv",
        )
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SAVED] Cross-dimension summary: {summary_path}")

        # Print compact cross-dimension table
        print(f"\n{'='*90}")
        print(f"CROSS-DIMENSION SUMMARY: Human vs AI steering effect")
        print(f"{'='*90}")

        # Key metrics to show
        key_metrics = [
            "word_count", "question_count", "like_rate",
            "fung_interpersonal_rate", "tom_rate",
            "politeness_rate", "disfluency_rate",
        ]

        header = f"{'Dimension':<28} {'N':<4}"
        for m in key_metrics:
            short = m.replace("_rate", "").replace("fung_", "f_")
            header += f" {short:<10}"
        print(header)
        print("-" * (32 + 10 * len(key_metrics)))

        for _, row in summary_df.iterrows():
            line = f"{row['dimension']:<28} {row['N']:<4}"
            for m in key_metrics:
                diff = row.get(f"{m}_human_minus_ai", np.nan)
                p = row.get(f"{m}_hva_p", np.nan)
                sig = ""
                if not np.isnan(p):
                    if p < 0.001: sig = "***"
                    elif p < 0.01: sig = "**"
                    elif p < 0.05: sig = "*"
                if np.isnan(diff):
                    line += f" {'N/A':<10}"
                else:
                    line += f" {diff:+.3f}{sig:<4} "
            print(line)

        print("-" * (32 + 10 * len(key_metrics)))

    print(f"\n✅ Behavioral analysis complete.")


if __name__ == "__main__":
    main()