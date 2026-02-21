#!/usr/bin/env python3
"""
Script: 5_behavior_analysis.py

Purpose: Compute linguistic feature profiles on Experiment 2b steered conversation
data and compare across conditions (baseline, human-steered, AI-steered).

Handles the directory structure from 3b_causality_generate.py:
    intervention_results/V{1,2}/{control,reading}_probes/is_{N}/...

V1: auto-walks probe types × strengths under the result root.
V2: auto-walks probe types × strengths, loads per-subject CSVs.

Usage:
    # Walk all probe types × strengths:
    python 4_behavior_analysis.py --version v1
    python 4_behavior_analysis.py --version v2

    # Specific strength only:
    python 4_behavior_analysis.py --version v1 --strength 16
    python 4_behavior_analysis.py --version v2 --strength 8

    # Custom input root:
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
# ============================================================

EXP1_UTILS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "exp_1", "balanced_names", "code", "analysis", "utils")
)


def _load_util_module(name):
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
            "Root dir containing {control,reading}_probes/is_{N}/ subdirs. "
            "Default: data/intervention_results/V1 or V2."
        ),
    )
    parser.add_argument(
        "--version", required=True, choices=["v1", "v2"],
        help="V1 = single-turn test questions; V2 = multi-turn Exp 1 recreation.",
    )
    parser.add_argument(
        "--strength", type=int, default=None,
        help="Only analyze a specific intervention strength (e.g., 16). Default: all.",
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
#                    DIRECTORY DISCOVERY
# ============================================================

def discover_probe_strength_dirs(base_dir, strength_filter=None):
    """
    Walk base_dir/{control,reading}_probes/is_{N}/ and return list of
    (probe_label, strength, dir_path) tuples.
    """
    results = []
    for probe_dir_name in sorted(os.listdir(base_dir)):
        probe_path = os.path.join(base_dir, probe_dir_name)
        #if not os.path.isdir(probe_path):
        #    continue
        #if not probe_dir_name.endswith("_probes"):
        #    continue
        ## Derive label: "control_probes" -> "control_probe"
        #probe_label = probe_dir_name.rstrip("s")
        
        if "_probes" not in probe_dir_name:
            continue
        # Keep full dir name as label (e.g., "reading_probes_matched")
        probe_label = probe_dir_name

        for is_dir_name in sorted(os.listdir(probe_path)):
            is_path = os.path.join(probe_path, is_dir_name)
            if not os.path.isdir(is_path) or not is_dir_name.startswith("is_"):
                continue
            try:
                strength = int(is_dir_name.split("_", 1)[1])
            except ValueError:
                continue
            if strength_filter is not None and strength != strength_filter:
                continue
            results.append((probe_label, strength, is_path))

    if not results:
        raise FileNotFoundError(
            f"No probe/strength dirs found under {base_dir}.\n"
            f"Expected structure: {base_dir}/{{control,reading}}_probes/is_{{N}}/"
        )
    print(f"[INFO] Discovered {len(results)} probe×strength combinations in {base_dir}")
    for label, N, path in results:
        print(f"  {label} / is_{N}: {path}")
    return results


# ============================================================
#                    DATA LOADING
# ============================================================

def load_v1_csv(csv_path, probe_label, strength):
    """Load one V1 intervention_responses.csv."""
    print(f"  Loading {probe_label}/is_{strength} from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = {"question_idx", "question", "condition", "response"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"V1 CSV missing columns: {missing}")

    df = df.rename(columns={"response": "transcript_sub"})
    df["probe_type"] = probe_label
    df["strength"] = strength
    df["subject"] = "s001"
    df["topic"] = df["question"].apply(lambda q: q[:50])

    print(f"    {len(df)} rows, conditions: {df['condition'].value_counts().to_dict()}")
    return df


def load_v1_data(base_dir, strength_filter=None):
    """Load V1 data, walking probe types × strengths."""
    print(f"[INFO] Loading V1 data from {base_dir}")
    combos = discover_probe_strength_dirs(base_dir, strength_filter)

    probe_dfs = {}  # key: (probe_label, strength)
    for probe_label, strength, dir_path in combos:
        csv_path = os.path.join(dir_path, "intervention_responses.csv")
        if os.path.exists(csv_path):
            probe_dfs[(probe_label, strength)] = load_v1_csv(csv_path, probe_label, strength)
        else:
            print(f"  [SKIP] Not found: {csv_path}")

    if not probe_dfs:
        raise FileNotFoundError(f"No intervention_responses.csv found under {base_dir}")
    return probe_dfs


def load_v2_data(base_dir, strength_filter=None):
    """Load V2 data, walking probe types × strengths × subjects."""
    print(f"[INFO] Loading V2 data from {base_dir}")
    combos = discover_probe_strength_dirs(base_dir, strength_filter)

    probe_dfs = {}  # key: (probe_label, strength)
    for probe_label, strength, dir_path in combos:
        per_subj_dir = os.path.join(dir_path, "per_subject")
        if not os.path.isdir(per_subj_dir):
            print(f"  [SKIP] No per_subject/ in {dir_path}")
            continue

        csv_files = sorted(glob.glob(os.path.join(per_subj_dir, "s*.csv")))
        if not csv_files:
            print(f"  [SKIP] No subject CSVs in {per_subj_dir}")
            continue

        all_dfs = []
        for path in csv_files:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower()
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        combined["probe_type"] = probe_label
        combined["strength"] = strength
        print(f"  {probe_label}/is_{strength}: {len(combined)} rows, "
              f"{combined['subject'].nunique()} subjects, "
              f"conditions: {combined['condition'].value_counts().to_dict()}")
        probe_dfs[(probe_label, strength)] = combined

    if not probe_dfs:
        raise FileNotFoundError(f"No V2 subject CSVs found under {base_dir}")
    return probe_dfs


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

    print(f"  {len(df)} utterances -> {len(trial_df)} trials")
    return trial_df


def aggregate_to_subject_condition(df, metrics):
    return df.groupby(["subject", "condition"])[metrics].mean().reset_index()


def aggregate_to_subject_condition_social(df, metrics):
    if "social_type" not in df.columns or df["social_type"].nunique() <= 1:
        return None
    return df.groupby(["subject", "condition", "social_type"])[metrics].mean().reset_index()


# ============================================================
#                    STATISTICAL TESTS
# ============================================================

def run_condition_anova(agg_df, metric):
    """One-way RM-ANOVA across 3 conditions (V2)."""
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
    """2-way RM-ANOVA: Condition (3) x Sociality (2)."""
    result = {"metric": metric}
    if agg_df is None or "social_type" not in agg_df.columns:
        result["error"] = "No sociality data"
        return result

    try:
        required_cells = set()
        for cond in agg_df["condition"].unique():
            for soc in agg_df["social_type"].unique():
                required_cells.add((cond, soc))

        complete_subs = [
            sub for sub in agg_df["subject"].unique()
            if required_cells.issubset(
                set(zip(agg_df[agg_df["subject"] == sub]["condition"],
                        agg_df[agg_df["subject"] == sub]["social_type"]))
            )
        ]

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


def run_v1_condition_stats(df, metric):
    """V1 stats: one-way ANOVA + independent t-tests (each question = observation)."""
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

def _sig_stars(p):
    if isinstance(p, float) and not np.isnan(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
    return ""


def format_v1_results(condition_results, probe_label, strength):
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT 2b BEHAVIORAL ANALYSIS (V1) — {probe_label.upper()} — N={strength}")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append(
        "\nNote: V1 has a single model (no subjects), so each question is treated as\n"
        "an independent observation. Stats are one-way ANOVA + independent t-tests\n"
        "(not repeated measures). Interpret with caution."
    )

    lines.append("\n" + "=" * 80)
    lines.append("ONE-WAY ANOVA: Effect of Steering Condition")
    lines.append("=" * 80)

    for r in condition_results:
        lines.append(f"\n{'-' * 60}")
        lines.append(f"{r['metric']} (n = {r.get('n_per_condition', '?')})")
        lines.append(f"{'-' * 60}")
        if "error" in r:
            lines.append(f"  ERROR: {r['error']}"); continue
        for cond in r.get("conditions", []):
            m, se, sd = r.get(f"{cond}_mean", np.nan), r.get(f"{cond}_sem", np.nan), r.get(f"{cond}_std", np.nan)
            lines.append(f"  {cond:>10}: M = {m:.4f} +/- {se:.4f} (SD = {sd:.4f})")
        F, p = r.get("F", np.nan), r.get("p", np.nan)
        lines.append(f"\n  Omnibus: F({r.get('df_between','?')}, {r.get('df_within','?')}) = {F:.3f}, p = {p:.4f} {_sig_stars(p)}")
        if "pairwise" in r:
            lines.append(f"\n  Pairwise (independent t-test):")
            for pw in r["pairwise"]:
                lines.append(f"    {pw['pair']:>25}: diff = {pw['diff_mean']:+.4f}, t = {pw['t']:.3f}, p = {pw['p']:.4f} {_sig_stars(pw['p'])}")

    # Summary table
    lines.append("\n\n" + "=" * 90)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 90)
    lines.append(f"\n{'Metric':<30} {'Baseline':<14} {'Human':<14} {'AI':<14} {'F':<10} {'p':<10}")
    lines.append("-" * 90)
    for r in condition_results:
        if "error" in r: continue
        p = r.get("p", np.nan)
        lines.append(f"{r['metric']:<30} {r.get('baseline_mean',np.nan):<14.4f} "
                     f"{r.get('human_mean',np.nan):<14.4f} {r.get('ai_mean',np.nan):<14.4f} "
                     f"{r.get('F',np.nan):<10.3f} {p:.4f}{_sig_stars(p)}")
    lines.append("-" * 90)
    lines.append("* p < .05, ** p < .01, *** p < .001")
    return "\n".join(lines)


def format_v2_results(condition_results, sociality_results, probe_label, strength):
    lines = []
    lines.append("=" * 80)
    lines.append(f"EXPERIMENT 2b BEHAVIORAL ANALYSIS (V2) — {probe_label.upper()} — N={strength}")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    lines.append("\n" + "=" * 80)
    lines.append("ONE-WAY RM-ANOVA: Effect of Steering Condition")
    lines.append("=" * 80)

    for r in condition_results:
        lines.append(f"\n{'-' * 60}")
        lines.append(f"{r['metric']} (N = {r.get('n_subjects', '?')})")
        lines.append(f"{'-' * 60}")
        if "error" in r:
            lines.append(f"  ERROR: {r['error']}"); continue
        for cond in r.get("conditions", []):
            m, se = r.get(f"{cond}_mean", np.nan), r.get(f"{cond}_sem", np.nan)
            lines.append(f"  {cond:>10}: M = {m:.4f} +/- {se:.4f}")
        F, p = r.get("F", np.nan), r.get("p", np.nan)
        lines.append(f"\n  Omnibus: F({r.get('df_num','?')}, {r.get('df_den','?')}) = {F:.3f}, p = {p:.4f} {_sig_stars(p)}")
        if "pairwise" in r:
            lines.append(f"\n  Pairwise (paired t-test):")
            for pw in r["pairwise"]:
                lines.append(f"    {pw['pair']:>25}: diff = {pw['diff_mean']:+.4f}, t = {pw['t']:.3f}, p = {pw['p']:.4f} {_sig_stars(pw['p'])}")

    if sociality_results and any("error" not in r for r in sociality_results):
        lines.append("\n\n" + "=" * 80)
        lines.append("TWO-WAY RM-ANOVA: Condition (3) x Sociality (2)")
        lines.append("=" * 80)
        for r in sociality_results:
            lines.append(f"\n{'-' * 60}")
            lines.append(f"{r['metric']} (N = {r.get('n_subjects', '?')})")
            if "error" in r:
                lines.append(f"  ERROR: {r['error']}"); continue
            for key in sorted(r.keys()):
                if key.endswith("_F"):
                    effect = key[:-2]
                    F_val, p_val = r[key], r.get(f"{effect}_p", np.nan)
                    lines.append(f"  {effect}: F = {F_val:.3f}, p = {p_val:.4f} {_sig_stars(p_val)}")

    # Summary table
    lines.append("\n\n" + "=" * 90)
    lines.append("SUMMARY TABLE")
    lines.append("=" * 90)
    lines.append(f"\n{'Metric':<30} {'Baseline':<14} {'Human':<14} {'AI':<14} {'F':<10} {'p':<10}")
    lines.append("-" * 90)
    for r in condition_results:
        if "error" in r: continue
        p = r.get("p", np.nan)
        lines.append(f"{r['metric']:<30} {r.get('baseline_mean',np.nan):<14.4f} "
                     f"{r.get('human_mean',np.nan):<14.4f} {r.get('ai_mean',np.nan):<14.4f} "
                     f"{r.get('F',np.nan):<10.3f} {p:.4f}{_sig_stars(p)}")
    lines.append("-" * 90)
    lines.append("* p < .05, ** p < .01, *** p < .001")
    return "\n".join(lines)


def format_cross_probe_comparison(all_results):
    """Cross-probe comparison table (one per strength)."""
    lines = []
    # Group by strength
    by_strength = {}
    for (probe_label, strength), results in all_results.items():
        by_strength.setdefault(strength, {})[probe_label] = results

    for strength, probe_results in sorted(by_strength.items()):
        lines.append("\n\n" + "=" * 100)
        lines.append(f"CROSS-PROBE COMPARISON — N={strength}")
        lines.append("=" * 100)
        lines.append(
            f"\n{'Metric':<28} "
            f"{'Ctrl BL':<10} {'Ctrl Hum':<10} {'Ctrl AI':<10} {'Ctrl p':<10} "
            f"{'Read BL':<10} {'Read Hum':<10} {'Read AI':<10} {'Read p':<10}"
        )
        lines.append("-" * 100)

        ctrl = {r["metric"]: r for r in probe_results.get("control_probe", [])}
        read = {r["metric"]: r for r in probe_results.get("reading_probe", [])}

        for metric in sorted(set(list(ctrl.keys()) + list(read.keys()))):
            def _v(r, k):
                v = r.get(k, np.nan)
                return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "---"
            def _p(r):
                p = r.get("p", np.nan)
                return f"{p:.4f}{_sig_stars(p)}" if isinstance(p, float) and not np.isnan(p) else "---"
            cr, rr = ctrl.get(metric, {}), read.get(metric, {})
            lines.append(
                f"{metric:<28} "
                f"{_v(cr,'baseline_mean'):<10} {_v(cr,'human_mean'):<10} {_v(cr,'ai_mean'):<10} {_p(cr):<10} "
                f"{_v(rr,'baseline_mean'):<10} {_v(rr,'human_mean'):<10} {_v(rr,'ai_mean'):<10} {_p(rr):<10}"
            )
        lines.append("-" * 100)
    return "\n".join(lines)


# ============================================================
#                    MAIN PIPELINES
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
    base_dir = args.input or "data/intervention_results/V1"
    out_dir = args.output or os.path.join(base_dir, "behavioral_results")
    os.makedirs(out_dir, exist_ok=True)

    probe_dfs = load_v1_data(base_dir, strength_filter=args.strength)

    all_results = {}   # (probe_label, strength) -> list of stat dicts
    all_reports = []

    for (probe_label, strength), df in sorted(probe_dfs.items()):
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {probe_label} / is_{strength}")
        print(f"{'='*60}")

        df = compute_all_metrics(df)
        metrics = ALL_METRICS.copy()
        if "sentiment" in df.columns and df["sentiment"].notna().any():
            metrics.append("sentiment")

        utt_path = os.path.join(out_dir, f"utterance_metrics_{probe_label}_is{strength}.csv")
        df.to_csv(utt_path, index=False)
        print(f"[SAVED] {utt_path}")

        condition_results = [run_v1_condition_stats(df, m) for m in metrics]
        all_results[(probe_label, strength)] = condition_results

        report = format_v1_results(condition_results, probe_label, strength)
        all_reports.append(report)

        stats_path = os.path.join(out_dir, f"stats_v1_{probe_label}_is{strength}.txt")
        with open(stats_path, "w") as f:
            f.write(report)
        print(f"[SAVED] {stats_path}")

    # Cross-probe comparison
    if len(all_results) > 1:
        comparison = format_cross_probe_comparison(all_results)
        all_reports.append(comparison)

    combined = "\n\n".join(all_reports)
    combined_path = os.path.join(out_dir, "stats_v1_combined.txt")
    with open(combined_path, "w") as f:
        f.write(combined)
    print(f"\n[SAVED] {combined_path}")
    print("\n" + combined)

    print(f"\n[DONE] Results in {out_dir}/")


def run_v2_pipeline(args):
    base_dir = args.input or "data/intervention_results/V2"
    out_dir = args.output or os.path.join(base_dir, "behavioral_results")
    os.makedirs(out_dir, exist_ok=True)

    probe_dfs = load_v2_data(base_dir, strength_filter=args.strength)

    all_results = {}
    all_reports = []

    for (probe_label, strength), df in sorted(probe_dfs.items()):
        print(f"\n{'='*60}")
        print(f"  ANALYZING: {probe_label} / is_{strength}")
        print(f"{'='*60}")

        df = add_social_classification(df, args.topics)
        df = compute_all_metrics(df)

        metrics = ALL_METRICS.copy()
        if "sentiment" in df.columns and df["sentiment"].notna().any():
            metrics.append("sentiment")

        utt_path = os.path.join(out_dir, f"utterance_metrics_{probe_label}_is{strength}.csv")
        df.to_csv(utt_path, index=False)
        print(f"[SAVED] {utt_path}")

        trial_df = aggregate_to_trial_level(df)
        trial_path = os.path.join(out_dir, f"trial_metrics_{probe_label}_is{strength}.csv")
        trial_df.to_csv(trial_path, index=False)
        print(f"[SAVED] {trial_path}")

        agg_df = aggregate_to_subject_condition(trial_df, metrics)
        agg_social_df = aggregate_to_subject_condition_social(trial_df, metrics)

        agg_path = os.path.join(out_dir, f"subject_condition_{probe_label}_is{strength}.csv")
        agg_df.to_csv(agg_path, index=False)
        print(f"[SAVED] {agg_path}")

        condition_results = [run_condition_anova(agg_df, m) for m in metrics]
        sociality_results = []
        if agg_social_df is not None:
            sociality_results = [run_condition_by_sociality_anova(agg_social_df, m) for m in metrics]

        all_results[(probe_label, strength)] = condition_results

        report = format_v2_results(condition_results, sociality_results, probe_label, strength)
        all_reports.append(report)

        stats_path = os.path.join(out_dir, f"stats_v2_{probe_label}_is{strength}.txt")
        with open(stats_path, "w") as f:
            f.write(report)
        print(f"[SAVED] {stats_path}")

    if len(all_results) > 1:
        comparison = format_cross_probe_comparison(all_results)
        all_reports.append(comparison)

    combined = "\n\n".join(all_reports)
    combined_path = os.path.join(out_dir, "stats_v2_combined.txt")
    with open(combined_path, "w") as f:
        f.write(combined)
    print(f"\n[SAVED] {combined_path}")
    print("\n" + combined)

    print(f"\n[DONE] Results in {out_dir}/")


def main():
    args = parse_args()
    if args.version == "v1":
        run_v1_pipeline(args)
    else:
        run_v2_pipeline(args)


if __name__ == "__main__":
    main()