#!/usr/bin/env python3
"""
Experiment 3: Behavioral Analysis of Concept-Vector-Steered Conversations (V1)

Computes linguistic feature profiles on concept-steering V1 output and compares
across conditions (baseline, human, ai) for each dim × strategy × strength cell.

Adapts the analysis logic from 6_behavior_analysis.py but for the concept steering
directory layout: results/{model}/{version}/concept_steering/v1/{dim}/{strategy}/

Usage:
    # All available cells for a version
    python 4b_concept_steering_behavior.py --version balanced_gpt

    # Filter to specific dims/strategies/strengths
    python 4b_concept_steering_behavior.py --version balanced_gpt \
        --dim_ids 0_baseline 15_shapes --strategies exp2_peak --strengths 2 4

Output per cell:
    .../v1/{dim}/{strategy}/N_{N}_behavioral.txt     (readable stats report)
    .../v1/{dim}/{strategy}/N_{N}_utterances.csv      (utterance-level metrics)

Cross-summary:
    .../v1/behavioral_summary.csv

Env: behavior_env
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Import Experiment 1 utils (shared linguistic markers)
# ============================================================

from config import config, set_version

EXP1_UTILS_DIR = str(config.PATHS.exp1_utils)
if EXP1_UTILS_DIR not in sys.path:
    sys.path.insert(0, EXP1_UTILS_DIR)

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
# CONFIG
# ============================================================

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
        description="Exp 3: Behavioral analysis of concept-vector-steered conversations (V1)."
    )
    parser.add_argument(
        "--version", required=True,
        help="Exp 2 version (e.g., balanced_gpt, nonsense_codeword).",
    )
    parser.add_argument(
        "--dim_ids", nargs="+", default=None,
        help="Filter to specific dimension names (e.g., 0_baseline 15_shapes). "
             "Default: auto-discover all.",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Filter to specific strategies (e.g., exp2_peak concept_aligned). "
             "Default: auto-discover all.",
    )
    parser.add_argument(
        "--strengths", type=int, nargs="+", default=None,
        help="Filter to specific strengths (e.g., 2 4 8). Default: all found.",
    )
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_v1_csv(csv_path):
    """Load a concept steering V1 results CSV."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    # Rename 'response' to 'transcript_sub' for compatibility with compute_all_metrics
    df = df.rename(columns={"response": "transcript_sub"})
    return df


def discover_cells(v1_root, dim_filter=None, strategy_filter=None, strength_filter=None):
    """Auto-discover all (dim, strategy, N) cells with results CSVs.

    Returns list of dicts: {dim, strategy, N, csv_path}
    """
    cells = []
    csv_pattern = os.path.join(v1_root, "*", "*", "N_*_results.csv")
    for csv_path in sorted(glob.glob(csv_pattern)):
        # Parse path: .../v1/{dim}/{strategy}/N_{N}_results.csv
        parts = csv_path.split(os.sep)
        filename = parts[-1]           # N_{N}_results.csv
        strategy = parts[-2]
        dim = parts[-3]

        # Extract N from filename
        m = re.match(r"N_(\d+)_results\.csv", filename)
        if not m:
            continue
        N = int(m.group(1))

        # Apply filters
        if dim_filter and dim not in dim_filter:
            continue
        if strategy_filter and strategy not in strategy_filter:
            continue
        if strength_filter and N not in strength_filter:
            continue

        cells.append({"dim": dim, "strategy": strategy, "N": N, "csv_path": csv_path})

    return cells


# ============================================================
# FEATURE COMPUTATION (copied from 6_behavior_analysis.py)
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
# STATISTICAL TESTS
# ============================================================

def run_v1_between_subjects_tests(df, metric):
    """
    V1: single subject, use per-question as observations with independent t-tests.
    """
    result = {"metric": metric}
    conditions = sorted(df["condition"].unique())
    result["conditions"] = conditions

    try:
        for cond in conditions:
            vals = df[df["condition"] == cond][metric].dropna()
            result[f"{cond}_mean"] = vals.mean()
            result[f"{cond}_sem"] = vals.sem()
            result[f"{cond}_n"] = len(vals)

        # Pairwise independent t-tests
        pairs = []
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                v1 = df[df["condition"] == c1][metric].dropna()
                v2 = df[df["condition"] == c2][metric].dropna()
                t, p = ttest_ind(v1, v2)
                pairs.append({
                    "pair": f"{c1}_vs_{c2}",
                    "t": t, "p": p,
                    "diff_mean": v1.mean() - v2.mean(),
                })
        result["pairwise"] = pairs
        result["note"] = "V1: independent t-tests (questions as observations)"

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
# REPORT FORMATTING
# ============================================================

def format_single_report(dim_name, strategy, N, condition_results):
    """Format a text report for one dim × strategy × strength cell."""
    lines = []
    lines.append("=" * 90)
    lines.append(f"CONCEPT VECTOR STEERING V1: BEHAVIORAL ANALYSIS")
    lines.append(f"Dimension: {dim_name}  |  Strategy: {strategy}  |  N = {N}")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 90)

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
    lines.append("=" * 90)

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


def extract_summary_row(dim_name, strategy, N, condition_results):
    """Extract one row for the cross-cell summary table."""
    row = {"dimension": dim_name, "strategy": strategy, "N": N}

    for r in condition_results:
        metric = r["metric"]
        for cond in ["baseline", "human", "ai"]:
            row[f"{metric}_{cond}_mean"] = r.get(f"{cond}_mean", np.nan)

        # Human vs AI difference
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


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Initialize version-dependent paths
    set_version(args.version)

    # Build root path
    v1_root = str(config.RESULTS.concept_steering / "v1")
    if not os.path.isdir(v1_root):
        print(f"[ERROR] V1 root not found: {v1_root}")
        sys.exit(1)

    # Discover cells
    cells = discover_cells(
        v1_root,
        dim_filter=args.dim_ids,
        strategy_filter=args.strategies,
        strength_filter=args.strengths,
    )

    if not cells:
        print(f"[ERROR] No result CSVs found in {v1_root}")
        sys.exit(1)

    print(f"Found {len(cells)} cells to analyze in {v1_root}")

    # Collect cross-cell summary rows
    summary_rows = []

    for cell in cells:
        dim = cell["dim"]
        strategy = cell["strategy"]
        N = cell["N"]
        csv_path = cell["csv_path"]

        print(f"\n{'#'*70}")
        print(f"# {dim} / {strategy} / N={N}")
        print(f"{'#'*70}")

        # Load data
        df = load_v1_csv(csv_path)
        print(f"  Loaded {len(df)} rows, "
              f"conditions: {df['condition'].value_counts().to_dict()}")

        # Compute metrics
        df = compute_all_metrics(df)

        active_metrics = [m for m in ALL_METRICS if m in df.columns]
        if "sentiment" in df.columns and df["sentiment"].notna().any():
            if "sentiment" not in active_metrics:
                active_metrics.append("sentiment")

        # Run tests
        condition_results = [run_v1_between_subjects_tests(df, m) for m in active_metrics]

        # Save outputs alongside the results CSV
        out_dir = os.path.dirname(csv_path)

        # Utterance-level metrics
        utt_path = os.path.join(out_dir, f"N_{N}_utterances.csv")
        df.to_csv(utt_path, index=False)

        # Report
        report = format_single_report(dim, strategy, N, condition_results)
        report_path = os.path.join(out_dir, f"N_{N}_behavioral.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Saved: {report_path}")

        # Summary row
        row = extract_summary_row(dim, strategy, N, condition_results)
        summary_rows.append(row)

        # Quick summary
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
    # Cross-cell summary
    # ================================================================

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(v1_root, "behavioral_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SAVED] Cross-cell summary: {summary_path}")

        # Print compact cross-cell table
        print(f"\n{'='*100}")
        print(f"CROSS-CELL SUMMARY: Human vs AI steering effect")
        print(f"{'='*100}")

        key_metrics = [
            "word_count", "question_count", "like_rate",
            "fung_interpersonal_rate", "tom_rate",
            "politeness_rate", "disfluency_rate",
        ]

        header = f"{'Dimension':<20} {'Strategy':<18} {'N':<4}"
        for m in key_metrics:
            short = m.replace("_rate", "").replace("fung_", "f_")
            header += f" {short:<10}"
        print(header)
        print("-" * (42 + 10 * len(key_metrics)))

        for _, row in summary_df.iterrows():
            line = f"{row['dimension']:<20} {row['strategy']:<18} {row['N']:<4}"
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

        print("-" * (42 + 10 * len(key_metrics)))

    print(f"\nBehavioral analysis complete.")


if __name__ == "__main__":
    main()
