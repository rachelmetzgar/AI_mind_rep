#!/usr/bin/env python3
"""
Script: 4_behavior_analysis.py

Purpose: Compute linguistic feature profiles on Experiment 2b steered conversation
data and compare across conditions (baseline, human-steered, AI-steered).

V1 mode auto-detects and runs on BOTH control_probes/ and reading_probes/ subdirs.

Usage:
    python 4_behavior_analysis.py --version v1
    python 4_behavior_analysis.py --version v1 --input data/intervention_results/V1

Env: behavior_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import re
import glob
import argparse
import importlib.util
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
# Direct file loading to avoid 'utils' namespace collisions
# ============================================================

EXP1_UTILS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "exp_1", "code", "analysis", "utils")
)


def _load_util_module(name):
    """Direct file import to avoid 'utils' namespace collisions."""
    fpath = os.path.join(EXP1_UTILS_DIR, f"{name}.py")
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Cannot find {fpath}\n"
            f"  EXP1_UTILS_DIR = {EXP1_UTILS_DIR}\n"
            f"  Update EXP1_UTILS_DIR if your exp_1 utils live elsewhere."
        )
    spec = importlib.util.spec_from_file_location(name, fpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hedges = _load_util_module("hedges_demir")
_discourse = _load_util_module("discourse_markers_fung")
_misc = _load_util_module("misc_text_markers")

DEMIR_ALL_HEDGES = _hedges.DEMIR_ALL_HEDGES
DEMIR_NOUNS = _hedges.DEMIR_NOUNS
DEMIR_ADJECTIVES = _hedges.DEMIR_ADJECTIVES
DEMIR_ADVERBS = _hedges.DEMIR_ADVERBS
DEMIR_VERBS = _hedges.DEMIR_VERBS
DEMIR_QUANTIFIERS = _hedges.DEMIR_QUANTIFIERS
DEMIR_MODALS = _hedges.DEMIR_MODALS

FUNG_INTERPERSONAL = _discourse.FUNG_INTERPERSONAL
FUNG_REFERENTIAL = _discourse.FUNG_REFERENTIAL
FUNG_STRUCTURAL = _discourse.FUNG_STRUCTURAL
FUNG_COGNITIVE = _discourse.FUNG_COGNITIVE
FUNG_ALL_23_MARKERS = _discourse.FUNG_ALL_23_MARKERS

LIWC_NONFLUENCIES = _misc.LIWC_NONFLUENCIES
LIWC_FILLERS = _misc.LIWC_FILLERS
LIWC_DISFLUENCIES = _misc.LIWC_DISFLUENCIES
TOM_PHRASES = _misc.TOM_PHRASES
POLITE_POSITIVE = _misc.POLITE_POSITIVE
POLITE_NEGATIVE = _misc.POLITE_NEGATIVE
IMPOLITE = _misc.IMPOLITE
LIKE_MARKER = _misc.LIKE_MARKER


# ============================================================
#                    CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Behavioral analysis of Experiment 2b steered conversations."
    )
    parser.add_argument(
        "--input", default=None,
        help=(
            "V1: base dir containing control_probes/ and reading_probes/ subdirs "
            "(default: data/intervention_results/V1). "
            "V2: path to per_subject directory."
        ),
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
        help="Output directory. Default: {input}/behavioral_results/",
    )
    return parser.parse_args()


# ============================================================
#                    DATA LOADING
# ============================================================

def load_v1_single_csv(csv_path, probe_label):
    """
    Load one V1 intervention responses CSV.
    Expected columns: question_idx, question, condition, response
    """
    print(f"  Loading {probe_label} data from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"question_idx", "question", "condition", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"V1 CSV missing columns: {missing}")

    df = df.rename(columns={"response": "transcript_sub"})
    df["probe_type"] = probe_label
    df["subject"] = "s001"
    df["topic"] = df["question"].apply(lambda q: q[:50])

    print(f"    {len(df)} rows, conditions: {df['condition'].value_counts().to_dict()}")
    return df


def load_v1_data(base_dir):
    """
    Load V1 data from both control_probes/ and reading_probes/ subdirs.
    Looks for: {base_dir}/control_probes/intervention_responses.csv
               {base_dir}/reading_probes/intervention_responses.csv
    """
    print(f"[INFO] Loading V1 data from {base_dir}")

    probe_dfs = {}
    probe_subdirs = [
        ("control_probes", "control_probe"),
        ("reading_probes", "reading_probe"),
    ]

    for subdir, label in probe_subdirs:
        csv_path = os.path.join(base_dir, subdir, "intervention_responses.csv")
        if os.path.exists(csv_path):
            probe_dfs[label] = load_v1_single_csv(csv_path, label)
        else:
            print(f"  [SKIP] Not found: {csv_path}")

    if not probe_dfs:
        raise FileNotFoundError(
            f"No intervention_responses.csv found in:\n"
            f"  {os.path.join(base_dir, 'control_probes', 'intervention_responses.csv')}\n"
            f"  {os.path.join(base_dir, 'reading_probes', 'intervention_responses.csv')}"
        )

    return probe_dfs


def load_v2_data(input_dir):
    """Load V2 per-subject CSVs."""
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

    for col in trial_df.columns:
        if col.endswith("_count") and col != "word_count":
            rate_col = col.replace("_count", "_rate")
            trial_df[rate_col] = trial_df[col] / trial_df["word_count"].replace(0, np.nan)
    if "politeness_score" in trial_df.columns:
        trial_df["politeness_rate"] = trial_df["politeness_score"] / trial_df["word_count"].replace(0, np.nan)

    print(f"  {len(df)} utterances → {len(trial_df)} trials")
    return trial_df


def aggregate_to_subject_condition(df, metrics):
    """Aggregate to subject × condition."""
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
    """One-way RM-ANOVA across 3 conditions."""
    result = {"metric": metric}
    conditions = sorted(agg_df["condition"].unique())
    result["conditions"] = conditions

    try:
        wide = agg_df.pivot(index="subject", columns="condition", values=metric).dropna()
        if len(wide) < 3:
            result["error"] = f"Too few subjects with complete data (n={len(wide)})"
            return result

        result["n_subjects"] = len(wide)

        for cond in conditions:
            if cond in wide.columns:
                result[f"{cond}_mean"] = wide[cond].mean()
                result[f"{cond}_sem"] = wide[cond].sem()

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
            result["F"] = np.nan
            result["p"] = np.nan

        pairs = []
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                if c1 in wide.columns and c2 in wide.columns:
                    t, p = ttest_rel(wide[c1], wide[c2])
                    pairs.append({"pair": f"{c1}_vs_{c2}", "t": t, "p": p,
                                  "diff_mean": (wide[c1] - wide[c2]).mean()})
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

        aovrm = AnovaRM(df_complete, metric, "subject", within=["condition", "social_type"])
        res = aovrm.fit()

        for row_name in res.anova_table.index:
            key = row_name.strip().lower().replace(" ", "_").replace(":", "_x_")
            result[f"{key}_F"] = res.anova_table.loc[row_name, "F Value"]
            result[f"{key}_p"] = res.anova_table.loc[row_name, "Pr > F"]

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
#          V1 BETWEEN-CONDITION STATS (no repeated measures)
# ============================================================

def run_v1_condition_stats(df, metric):
    """
    V1 stats: treat each question as an observation.
    One-way ANOVA + independent t-tests + descriptive stats.
    """
    result = {"metric": metric}
    conditions = sorted(df["condition"].unique())
    result["conditions"] = conditions

    try:
        groups = {c: df.loc[df["condition"] == c, metric].dropna() for c in conditions}
        result["n_per_condition"] = {c: len(g) for c, g in groups.items()}

        for cond, vals in groups.items():
            result[f"{cond}_mean"] = vals.mean()
            result[f"{cond}_sem"] = vals.sem()
            result[f"{cond}_std"] = vals.std()

        valid_groups = [g.values for g in groups.values() if len(g) > 1]
        if len(valid_groups) >= 2:
            F, p = f_oneway(*valid_groups)
            result["F"] = F
            result["p"] = p
            result["df_between"] = len(valid_groups) - 1
            result["df_within"] = sum(len(g) for g in valid_groups) - len(valid_groups)
        else:
            result["F"] = np.nan
            result["p"] = np.nan

        pairs = []
        for i, c1 in enumerate(conditions):
            for c2 in conditions[i+1:]:
                if len(groups[c1]) > 1 and len(groups[c2]) > 1:
                    t, p = ttest_ind(groups[c1], groups[c2])
                    pairs.append({"pair": f"{c1}_vs_{c2}", "t": t, "p": p,
                                  "diff_mean": groups[c1].mean() - groups[c2].mean()})
        result["pairwise"] = pairs

    except Exception as e:
        result["error"] = str(e)

    return result


# ============================================================
#                    OUTPUT FORMATTING
# ============================================================

def format_v1_results(condition_results, probe_label):
    """Format V1 results for a single probe type."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT 2b BEHAVIORAL ANALYSIS (V1) — {probe_label.upper()}")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append(
        "\nNote: V1 has a single model (no subjects), so each question is treated as\n"
        "an independent observation. Stats are one-way ANOVA + independent t-tests\n"
        "(not repeated measures). Interpret with caution — observations are not\n"
        "truly independent."
    )

    lines.append("\n" + "=" * 80)
    lines.append("ONE-WAY ANOVA: Effect of Steering Condition")
    lines.append("Conditions: baseline, human-steered, AI-steered")
    lines.append("=" * 80)

    for r in condition_results:
        lines.append(f"\n{'-' * 60}")
        lines.append(f"{r['metric']} (n = {r.get('n_per_condition', '?')})")
        lines.append(f"{'-' * 60}")

        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
            continue

        for cond in r.get("conditions", []):
            m = r.get(f"{cond}_mean", np.nan)
            se = r.get(f"{cond}_sem", np.nan)
            sd = r.get(f"{cond}_std", np.nan)
            lines.append(f"  {cond:>10}: M = {m:.4f} ± {se:.4f} (SD = {sd:.4f})")

        F = r.get("F", np.nan)
        p = r.get("p", np.nan)
        sig = ""
        if not np.isnan(p):
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
        df_b = r.get("df_between", "?")
        df_w = r.get("df_within", "?")
        lines.append(f"\n  Omnibus: F({df_b}, {df_w}) = {F:.3f}, p = {p:.4f} {sig}")

        if "pairwise" in r:
            lines.append(f"\n  Pairwise comparisons (independent t-test):")
            for pw in r["pairwise"]:
                pw_sig = ""
                if pw["p"] < 0.001: pw_sig = "***"
                elif pw["p"] < 0.01: pw_sig = "**"
                elif pw["p"] < 0.05: pw_sig = "*"
                lines.append(
                    f"    {pw['pair']:>25}: diff = {pw['diff_mean']:+.4f}, "
                    f"t = {pw['t']:.3f}, p = {pw['p']:.4f} {pw_sig}"
                )

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


def format_v1_comparison(all_probe_results):
    """Format cross-probe comparison table for V1."""
    lines = []
    lines.append("\n\n" + "=" * 100)
    lines.append("CROSS-PROBE COMPARISON: Control vs Reading Probes")
    lines.append("=" * 100)
    lines.append(
        f"\n{'Metric':<28} "
        f"{'Ctrl BL':<10} {'Ctrl Hum':<10} {'Ctrl AI':<10} {'Ctrl p':<10} "
        f"{'Read BL':<10} {'Read Hum':<10} {'Read AI':<10} {'Read p':<10}"
    )
    lines.append("-" * 100)

    ctrl_results = {r["metric"]: r for r in all_probe_results.get("control_probe", [])}
    read_results = {r["metric"]: r for r in all_probe_results.get("reading_probe", [])}
    all_metrics = sorted(set(list(ctrl_results.keys()) + list(read_results.keys())))

    for metric in all_metrics:
        cr = ctrl_results.get(metric, {})
        rr = read_results.get(metric, {})

        def _fmt(r, key, fmt=".4f"):
            v = r.get(key, np.nan)
            if isinstance(v, float) and np.isnan(v):
                return "—"
            return f"{v:{fmt}}"

        def _p_fmt(r):
            p = r.get("p", np.nan)
            if isinstance(p, float) and np.isnan(p):
                return "—"
            sig = ""
            if p < 0.001: sig = "***"
            elif p < 0.01: sig = "**"
            elif p < 0.05: sig = "*"
            return f"{p:.4f}{sig}"

        lines.append(
            f"{metric:<28} "
            f"{_fmt(cr, 'baseline_mean'):<10} {_fmt(cr, 'human_mean'):<10} {_fmt(cr, 'ai_mean'):<10} {_p_fmt(cr):<10} "
            f"{_fmt(rr, 'baseline_mean'):<10} {_fmt(rr, 'human_mean'):<10} {_fmt(rr, 'ai_mean'):<10} {_p_fmt(rr):<10}"
        )

    lines.append("-" * 100)
    lines.append("* p < .05, ** p < .01, *** p < .001")
    lines.append("BL = baseline, Hum = human-steered, AI = AI-steered")
    lines.append("Ctrl = control probes, Read = reading probes")

    return "\n".join(lines)


def format_results_v2(condition_results, sociality_results, version):
    """Format V2 results (RM-ANOVA)."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT 2b BEHAVIORAL ANALYSIS ({version.upper()})")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

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

        for cond in r.get("conditions", []):
            m = r.get(f"{cond}_mean", np.nan)
            se = r.get(f"{cond}_sem", np.nan)
            lines.append(f"  {cond:>10}: M = {m:.4f} ± {se:.4f}")

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


def run_v1_pipeline(args):
    """Full V1 pipeline: load both probe types, compute metrics, run stats, save."""
    base_dir = args.input or "data/intervention_results/V1"
    out_dir = args.output or os.path.join(base_dir, "behavioral_results")
    os.makedirs(out_dir, exist_ok=True)

    probe_dfs = load_v1_data(base_dir)

    all_probe_results = {}
    all_reports = []

    for probe_label, df in probe_dfs.items():
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {probe_label}")
        print(f"{'='*60}")

        df = compute_all_metrics(df)

        metrics = ALL_METRICS.copy()
        if "sentiment" in df.columns and df["sentiment"].notna().any():
            metrics.append("sentiment")

        utt_path = os.path.join(out_dir, f"utterance_level_metrics_{probe_label}.csv")
        df.to_csv(utt_path, index=False)
        print(f"[SAVED] {utt_path}")

        print(f"\n[INFO] Running condition ANOVAs for {probe_label}...")
        condition_results = [run_v1_condition_stats(df, m) for m in metrics]
        all_probe_results[probe_label] = condition_results

        report = format_v1_results(condition_results, probe_label)
        all_reports.append(report)

        stats_path = os.path.join(out_dir, f"behavioral_stats_v1_{probe_label}.txt")
        with open(stats_path, "w") as f:
            f.write(report)
        print(f"[SAVED] {stats_path}")

    if len(all_probe_results) == 2:
        comparison = format_v1_comparison(all_probe_results)
        all_reports.append(comparison)

    combined_report = "\n\n".join(all_reports)
    combined_path = os.path.join(out_dir, "behavioral_stats_v1_combined.txt")
    with open(combined_path, "w") as f:
        f.write(combined_report)
    print(f"\n[SAVED] {combined_path}")

    print("\n" + combined_report)

    print(f"\n{'='*60}")
    print(f"[DONE] Results saved to {out_dir}/")
    print(f"{'='*60}")


def run_v2_pipeline(args):
    """Full V2 pipeline."""
    df = load_v2_data(args.input)
    df = add_social_classification(df, args.topics)
    df = compute_all_metrics(df)

    metrics = ALL_METRICS.copy()
    if "sentiment" in df.columns and df["sentiment"].notna().any():
        metrics.append("sentiment")

    out_dir = args.output or os.path.join(os.path.dirname(args.input), "..", "behavioral_results")
    os.makedirs(out_dir, exist_ok=True)

    utt_path = os.path.join(out_dir, "utterance_level_metrics.csv")
    df.to_csv(utt_path, index=False)
    print(f"[SAVED] {utt_path}")

    trial_df = aggregate_to_trial_level(df)
    trial_path = os.path.join(out_dir, "trial_level_metrics.csv")
    trial_df.to_csv(trial_path, index=False)
    print(f"[SAVED] {trial_path}")

    agg_df = aggregate_to_subject_condition(trial_df, metrics)
    agg_social_df = aggregate_to_subject_condition_social(trial_df, metrics)

    agg_path = os.path.join(out_dir, "subject_condition_means.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"[SAVED] {agg_path}")

    print("\n[INFO] Running condition RM-ANOVAs...")
    condition_results = [run_condition_anova(agg_df, m) for m in metrics]

    sociality_results = []
    if agg_social_df is not None:
        print("[INFO] Running condition × sociality RM-ANOVAs...")
        sociality_results = [run_condition_by_sociality_anova(agg_social_df, m) for m in metrics]

    report = format_results_v2(condition_results, sociality_results, "v2")
    stats_path = os.path.join(out_dir, "behavioral_stats_v2.txt")
    with open(stats_path, "w") as f:
        f.write(report)
    print(f"\n[SAVED] {stats_path}")
    print("\n" + report)

    print(f"\n{'='*60}")
    print(f"[DONE] Results saved to {out_dir}/")
    print(f"{'='*60}")


def main():
    args = parse_args()

    if args.version == "v1":
        run_v1_pipeline(args)
    else:
        run_v2_pipeline(args)


if __name__ == "__main__":
    main()