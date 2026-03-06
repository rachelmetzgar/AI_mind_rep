#!/usr/bin/env python3
"""
Per-agent identity breakdown analysis.

Fine-grained breakdown of linguistic behavior by individual partner agent.
Compares whether behavior differs across specific individual partners —
both within type and across type, and via aggregate type contrasts.

Statistics:
    1. One-way repeated-measures ANOVA (k agents) per metric.
    2. Post-hoc pairwise paired t-tests with BH-FDR correction.
    3. Planned aggregate contrasts (Human avg vs AI avg, etc.).
    4. Type-level summary (Human avg vs AI avg) with BH-FDR across metrics.

Usage:
    python 2_identity_breakdown.py --version balanced_gpt --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import warnings
import textwrap
import argparse
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, f as f_dist

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.config import (
    parse_version_model, get_version_config, data_dir, results_dir, figures_dir,
)
from code.utils.hedges_demir import (
    DEMIR_ALL_HEDGES, DEMIR_NOUNS, DEMIR_ADJECTIVES, DEMIR_ADVERBS,
    DEMIR_VERBS, DEMIR_QUANTIFIERS, DEMIR_MODALS,
)
from code.utils.discourse_markers_fung import (
    FUNG_INTERPERSONAL, FUNG_REFERENTIAL, FUNG_STRUCTURAL, FUNG_COGNITIVE,
    FUNG_ALL_23_MARKERS,
)
from code.utils.misc_text_markers import (
    LIWC_NONFLUENCIES, LIWC_FILLERS,
    TOM_PHRASES, POLITE_POSITIVE, POLITE_NEGATIVE, IMPOLITE, LIKE_MARKER,
)


# ── Metrics to analyse ──────────────────────────────────────────────────────
RATE_METRICS = [
    "word_count",
    "question_count",
    "demir_modal_rate",
    "demir_verb_rate",
    "demir_adverb_rate",
    "demir_adjective_rate",
    "demir_quantifier_rate",
    "demir_noun_rate",
    "demir_total_rate",
    "fung_interpersonal_rate",
    "fung_referential_rate",
    "fung_structural_rate",
    "fung_cognitive_rate",
    "fung_total_rate",
    "nonfluency_rate",
    "liwc_filler_rate",
    "disfluency_rate",
    "like_rate",
    "tom_rate",
    "politeness_rate",
    "sentiment",
    "quality",
    "connectedness",
]

METRIC_LABELS = {
    "word_count":              "Word Count",
    "question_count":          "Questions (#)",
    "demir_modal_rate":        "Demir: Modal Aux.",
    "demir_verb_rate":         "Demir: Epistemic Verbs",
    "demir_adverb_rate":       "Demir: Epistemic Adverbs",
    "demir_adjective_rate":    "Demir: Epistemic Adj.",
    "demir_quantifier_rate":   "Demir: Quantifiers",
    "demir_noun_rate":         "Demir: Epistemic Nouns",
    "demir_total_rate":        "Demir: Hedging (Total)",
    "fung_interpersonal_rate": "Fung: Interpersonal DMs",
    "fung_referential_rate":   "Fung: Referential DMs",
    "fung_structural_rate":    "Fung: Structural DMs",
    "fung_cognitive_rate":     "Fung: Cognitive DMs",
    "fung_total_rate":         "Fung: Discourse Markers (Total)",
    "nonfluency_rate":         "Nonfluency Rate (LIWC)",
    "liwc_filler_rate":        "Filler Rate (LIWC)",
    "disfluency_rate":         "Disfluency Rate (Total)",
    "like_rate":               "Discourse 'Like' Rate",
    "tom_rate":                "ToM Phrases Rate",
    "politeness_rate":         "Politeness Rate",
    "sentiment":               "Sentiment (VADER)",
    "quality":                 "Conv. Quality (1-4)",
    "connectedness":           "Connectedness (1-4)",
}

METRIC_GROUPS = {
    "Content":          ["word_count", "question_count"],
    "Hedging (Demir)":  ["demir_modal_rate", "demir_verb_rate", "demir_adverb_rate",
                         "demir_adjective_rate", "demir_quantifier_rate",
                         "demir_noun_rate", "demir_total_rate"],
    "Discourse (Fung)": ["fung_interpersonal_rate", "fung_referential_rate",
                         "fung_structural_rate", "fung_cognitive_rate",
                         "fung_total_rate"],
    "Disfluency":       ["nonfluency_rate", "liwc_filler_rate", "disfluency_rate"],
    "Social/Pragmatic": ["like_rate", "tom_rate", "politeness_rate"],
    "Ratings/Affect":   ["sentiment", "quality", "connectedness"],
}

COUNT_COLS = [
    "word_count", "question_count",
    "demir_modal_count", "demir_verb_count", "demir_adverb_count",
    "demir_adjective_count", "demir_quantifier_count", "demir_noun_count",
    "demir_total_count",
    "fung_interpersonal_count", "fung_referential_count",
    "fung_structural_count", "fung_cognitive_count", "fung_total_count",
    "nonfluency_count", "liwc_filler_count", "disfluency_count",
    "tom_count", "like_count", "politeness_score",
]

RATE_PAIRS = [
    ("demir_modal_rate",        "demir_modal_count"),
    ("demir_verb_rate",         "demir_verb_count"),
    ("demir_adverb_rate",       "demir_adverb_count"),
    ("demir_adjective_rate",    "demir_adjective_count"),
    ("demir_quantifier_rate",   "demir_quantifier_count"),
    ("demir_noun_rate",         "demir_noun_count"),
    ("demir_total_rate",        "demir_total_count"),
    ("fung_interpersonal_rate", "fung_interpersonal_count"),
    ("fung_referential_rate",   "fung_referential_count"),
    ("fung_structural_rate",    "fung_structural_count"),
    ("fung_cognitive_rate",     "fung_cognitive_count"),
    ("fung_total_rate",         "fung_total_count"),
    ("nonfluency_rate",         "nonfluency_count"),
    ("liwc_filler_rate",        "liwc_filler_count"),
    ("disfluency_rate",         "disfluency_count"),
    ("tom_rate",                "tom_count"),
    ("like_rate",               "like_count"),
    ("politeness_rate",         "politeness_score"),
]


# ── Build agent info from version config ─────────────────────────────────────

def build_agent_info(vcfg: dict):
    """
    Build agent mapping, ordering, and color assignment from version config.

    Returns:
        agent_map:    dict mapping agent code → display name
        agent_order:  list of display names in order (humans first, then AI)
        human_agents: list of human display names
        ai_agents:    list of AI display names
        agent_colors: dict mapping display name → color
    """
    agent_map_cfg = vcfg["agent_map"]

    # Build code → display name mapping
    agent_map = {}
    human_agents = []
    ai_agents = []
    for code, info in agent_map_cfg.items():
        name = info["name"] or info["type"]  # fallback to type if no name
        agent_map[code] = name
        if code.startswith("hum"):
            if name not in human_agents:
                human_agents.append(name)
        elif code.startswith("bot"):
            if name not in ai_agents:
                ai_agents.append(name)

    agent_order = human_agents + ai_agents

    # Assign colours: blues for human, reds for AI
    human_palette = ["#4C72B0", "#55A868", "#8172B2", "#64B5CD"]
    ai_palette = ["#C44E52", "#DD8452", "#DA8BC3", "#CCB974"]
    agent_colors = {}
    for i, a in enumerate(human_agents):
        agent_colors[a] = human_palette[i % len(human_palette)]
    for i, a in enumerate(ai_agents):
        agent_colors[a] = ai_palette[i % len(ai_palette)]

    return agent_map, agent_order, human_agents, ai_agents, agent_colors


# ── Data I/O ─────────────────────────────────────────────────────────────────

def load_combined_csv(dd: str) -> pd.DataFrame:
    combined = os.path.join(dd, "combined_text_data.csv")
    if os.path.exists(combined):
        print(f"[INFO] Loading combined CSV: {combined}")
        try:
            df = pd.read_csv(combined, on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(combined)
        print(f"  Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        return df

    print(f"[WARN] No combined CSV found; loading per-subject files from {dd}")
    frames = []
    for fname in sorted(os.listdir(dd)):
        if re.match(r"s\d+\.csv$", fname):
            frames.append(pd.read_csv(os.path.join(dd, fname)))
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {dd}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Concatenated {len(frames)} subject files -> {len(df)} rows")
    return df


# ── Metric computation ───────────────────────────────────────────────────────

def _count_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def count_patterns(text: str, patterns: List[str]) -> int:
    if not isinstance(text, str):
        return 0
    tl = text.lower()
    return sum(len(re.findall(p, tl)) for p in patterns)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Computing linguistic metrics ...")
    df = df.copy()
    t = df["transcript_sub"]

    df["word_count"] = t.apply(_count_words)
    df["question_count"] = t.apply(lambda x: str(x).count("?") if isinstance(x, str) else 0)

    for col, patterns in [
        ("demir_modal_count",      DEMIR_MODALS),
        ("demir_verb_count",       DEMIR_VERBS),
        ("demir_adverb_count",     DEMIR_ADVERBS),
        ("demir_adjective_count",  DEMIR_ADJECTIVES),
        ("demir_quantifier_count", DEMIR_QUANTIFIERS),
        ("demir_noun_count",       DEMIR_NOUNS),
    ]:
        df[col] = t.apply(lambda x, p=patterns: count_patterns(x, p))
    df["demir_total_count"] = (df["demir_modal_count"] + df["demir_verb_count"] +
                               df["demir_adverb_count"] + df["demir_adjective_count"] +
                               df["demir_quantifier_count"] + df["demir_noun_count"])

    for col, patterns in [
        ("fung_interpersonal_count", FUNG_INTERPERSONAL),
        ("fung_referential_count",   FUNG_REFERENTIAL),
        ("fung_structural_count",    FUNG_STRUCTURAL),
        ("fung_cognitive_count",     FUNG_COGNITIVE),
        ("fung_total_count",         FUNG_ALL_23_MARKERS),
    ]:
        df[col] = t.apply(lambda x, p=patterns: count_patterns(x, p))

    df["nonfluency_count"]  = t.apply(lambda x: count_patterns(x, LIWC_NONFLUENCIES))
    df["liwc_filler_count"] = t.apply(lambda x: count_patterns(x, LIWC_FILLERS))
    df["disfluency_count"]  = df["nonfluency_count"] + df["liwc_filler_count"]

    df["tom_count"]  = t.apply(lambda x: count_patterns(x, TOM_PHRASES))
    df["like_count"] = t.apply(lambda x: len(re.findall(LIKE_MARKER, x.lower())) if isinstance(x, str) else 0)
    df["polite_pos"] = t.apply(lambda x: count_patterns(x, POLITE_POSITIVE))
    df["polite_neg"] = t.apply(lambda x: count_patterns(x, POLITE_NEGATIVE))
    df["impolite"]   = t.apply(lambda x: count_patterns(x, IMPOLITE))
    df["politeness_score"] = df["polite_pos"] + df["polite_neg"] - df["impolite"]

    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        df["sentiment"] = t.apply(lambda x: sia.polarity_scores(str(x)).get("compound", np.nan)
                                   if isinstance(x, str) else np.nan)
    except Exception:
        df["sentiment"] = np.nan

    print("  [OK] Utterance-level metrics computed.")
    return df


def aggregate_to_subject_agent(df: pd.DataFrame, agent_map: dict) -> pd.DataFrame:
    print("[INFO] Aggregating to subject x agent level ...")
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    df["partner"] = df["agent"].map(agent_map)
    unknown = df["partner"].isna()
    if unknown.any():
        print(f"  [WARN] {unknown.sum()} rows with unrecognised agent values: "
              f"{df.loc[unknown, 'agent'].unique()}")
        df = df[~unknown].copy()

    for col in ("quality", "connectedness"):
        cap = col.capitalize()
        if col not in df.columns and cap in df.columns:
            df[col] = pd.to_numeric(df[cap], errors="coerce")
        elif col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    avail_count = [c for c in COUNT_COLS if c in df.columns]
    avail_sent  = [c for c in ["sentiment"] if c in df.columns]
    avail_rat   = [c for c in ["quality", "connectedness"] if c in df.columns]

    agg_dict = {c: "sum" for c in avail_count}
    agg_dict.update({c: "mean" for c in avail_sent + avail_rat})

    group_cols = [c for c in ["subject", "partner", "topic"] if c in df.columns]
    trial_df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    wc = trial_df["word_count"].replace(0, np.nan)
    for rate_col, count_col in RATE_PAIRS:
        if count_col in trial_df.columns:
            trial_df[rate_col] = trial_df[count_col] / wc

    rate_cols_avail = [c for c in RATE_METRICS if c in trial_df.columns]
    sub_agent = (trial_df.groupby(["subject", "partner"])[rate_cols_avail]
                          .mean()
                          .reset_index())

    n_subj = sub_agent["subject"].nunique()
    n_agents = sub_agent["partner"].nunique()
    print(f"  {n_subj} subjects x {n_agents} agents -> {len(sub_agent)} rows")
    return sub_agent


# ── Statistical helpers ──────────────────────────────────────────────────────

def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    n = len(pvals)
    if n == 0:
        return np.array([]), np.array([], dtype=bool)
    order    = np.argsort(pvals)
    ranks    = np.argsort(order) + 1
    adj      = np.minimum(1.0, pvals * n / ranks)
    for i in range(n - 2, -1, -1):
        adj[order[i]] = min(adj[order[i]], adj[order[i + 1]])
    rejected = adj <= alpha
    return adj, rejected


def cohens_dz(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if diff.size < 2:
        return np.nan
    sd = np.nanstd(diff, ddof=1)
    return np.nan if sd == 0 else np.nanmean(diff) / sd


def rm_anova_one_way(wide: pd.DataFrame, agents: List[str]) -> Dict:
    mat = wide[agents].dropna().values
    n, k = mat.shape
    if n < 3 or k < 2:
        return {"error": f"Insufficient data: n={n}, k={k}"}

    grand_mean    = mat.mean()
    agent_means   = mat.mean(axis=0)
    subject_means = mat.mean(axis=1)

    ss_between  = n * np.sum((agent_means  - grand_mean) ** 2)
    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_total    = np.sum((mat - grand_mean) ** 2)
    ss_error    = ss_total - ss_between - ss_subjects

    df_between = k - 1
    df_error   = (k - 1) * (n - 1)

    ms_between = ss_between / df_between
    ms_error   = ss_error   / df_error if df_error > 0 else np.nan

    F = ms_between / ms_error if ms_error and ms_error > 0 else np.nan
    p = 1 - f_dist.cdf(F, df_between, df_error) if np.isfinite(F) else np.nan
    eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "F": F, "p": p,
        "df_between": df_between, "df_error": df_error,
        "eta_sq": eta_sq,
        "n_subjects": n,
        "agent_means": dict(zip(agents, agent_means)),
        "agent_sems":  dict(zip(agents, mat.std(axis=0, ddof=1) / np.sqrt(n))),
    }


def pairwise_tests(wide: pd.DataFrame, agents: List[str]) -> pd.DataFrame:
    pairs = list(combinations(agents, 2))
    rows  = []
    for a, b in pairs:
        data = wide[[a, b]].dropna()
        x, y = data[a].values, data[b].values
        if len(x) < 3:
            rows.append({"pair": f"{a} vs {b}", "a": a, "b": b,
                         "n": len(x), "mean_a": np.nan, "sem_a": np.nan,
                         "mean_b": np.nan, "sem_b": np.nan,
                         "t": np.nan, "p_raw": np.nan, "dz": np.nan})
            continue
        t, p = ttest_rel(x, y, nan_policy="omit")
        dz = cohens_dz(x, y)
        rows.append({
            "pair": f"{a} vs {b}", "a": a, "b": b, "n": len(x),
            "mean_a": x.mean(), "sem_a": x.std(ddof=1) / np.sqrt(len(x)),
            "mean_b": y.mean(), "sem_b": y.std(ddof=1) / np.sqrt(len(y)),
            "t": t, "p_raw": p, "dz": dz,
        })

    result = pd.DataFrame(rows)
    if result.empty or result["p_raw"].isna().all():
        result["p_adj"] = np.nan
        result["sig"]   = ""
        return result

    pvals = result["p_raw"].values
    adj, rejected = bh_fdr(np.where(np.isnan(pvals), 1.0, pvals))
    result["p_adj"]    = adj
    result["rejected"] = rejected
    result["sig"] = result["p_adj"].apply(
        lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    )
    return result


def aggregate_contrasts(wide: pd.DataFrame, agent_order: List[str],
                        human_agents: List[str], ai_agents: List[str]) -> pd.DataFrame:
    data = wide[agent_order].dropna()
    rows = []

    human_avg = data[human_agents].mean(axis=1)
    ai_avg    = data[ai_agents].mean(axis=1)

    hum_str = "+".join(human_agents)
    ai_str  = "+".join(ai_agents)

    contrasts = [
        (f"Human avg vs AI avg",
         human_avg, ai_avg,
         f"Mean of ({hum_str}) vs mean of ({ai_str})."),
    ]
    # Individual human vs AI avg
    for h in human_agents:
        contrasts.append((
            f"{h} vs AI avg",
            data[h], ai_avg,
            f"{h} vs AI avg ({ai_str}).",
        ))

    for label, x, y, note in contrasts:
        x, y = x.values, y.values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[mask], y[mask]
        if len(x) < 3:
            rows.append({"contrast": label, "n": len(x),
                         "mean_a": np.nan, "mean_b": np.nan,
                         "t": np.nan, "p_raw": np.nan, "dz": np.nan,
                         "note": note})
            continue
        t, p = ttest_rel(x, y)
        dz = cohens_dz(x, y)
        rows.append({
            "contrast": label, "n": len(x),
            "mean_a": x.mean(), "mean_b": y.mean(),
            "t": t, "p_raw": p, "dz": dz, "note": note,
        })

    result = pd.DataFrame(rows)
    pvals = result["p_raw"].values
    adj, _ = bh_fdr(np.where(np.isnan(pvals), 1.0, pvals))
    result["p_adj"] = adj
    result["sig"] = result["p_adj"].apply(
        lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    )
    return result


def type_level_analysis(wide_per_metric: Dict[str, pd.DataFrame],
                        metrics: List[str],
                        human_agents: List[str],
                        ai_agents: List[str]) -> pd.DataFrame:
    print("[INFO] Running type-level analysis (Human avg vs AI avg) ...")
    rows = []
    for m in metrics:
        w = wide_per_metric[m]
        hum_cols = [a for a in human_agents if a in w.columns]
        ai_cols  = [a for a in ai_agents if a in w.columns]
        if not hum_cols or not ai_cols:
            continue

        data = w[hum_cols + ai_cols].dropna()
        human_avg = data[hum_cols].mean(axis=1).values
        ai_avg    = data[ai_cols].mean(axis=1).values

        if len(human_avg) < 3:
            continue

        t_stat, p_raw = ttest_rel(human_avg, ai_avg)
        dz = cohens_dz(human_avg, ai_avg)
        rows.append({
            "metric":     m,
            "mean_human": human_avg.mean(),
            "sem_human":  human_avg.std(ddof=1) / np.sqrt(len(human_avg)),
            "mean_ai":    ai_avg.mean(),
            "sem_ai":     ai_avg.std(ddof=1) / np.sqrt(len(ai_avg)),
            "t":          t_stat,
            "p_raw":      p_raw,
            "dz":         dz,
            "n":          len(human_avg),
        })

    results = pd.DataFrame(rows)
    if not results.empty:
        pvals = results["p_raw"].values
        adj, rejected = bh_fdr(np.where(np.isnan(pvals), 1.0, pvals))
        results["p_adj"]    = adj
        results["rejected"] = rejected
        results["sig"] = results["p_adj"].apply(
            lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        )
        n_sig = results["rejected"].sum()
        print(f"  {n_sig} of {len(results)} metrics significant after BH-FDR")

    return results


# ── Figure generation ────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


def plot_bar_metric(sub_agent: pd.DataFrame, metric: str,
                    pairwise: pd.DataFrame, anova_res: Dict,
                    agent_order: List[str], agent_colors: Dict[str, str]) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    agents = agent_order
    means  = []
    sems   = []
    for ag in agents:
        vals = sub_agent.loc[sub_agent["partner"] == ag, metric].dropna()
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

    x      = np.arange(len(agents))
    colors = [agent_colors.get(a, "#999999") for a in agents]
    bars   = ax.bar(x, means, yerr=sems, color=colors, width=0.55,
                    capsize=4, alpha=0.88, edgecolor="white", linewidth=0.8,
                    error_kw={"elinewidth": 1.4, "ecolor": "grey"})

    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=9)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    if "F" in anova_res and np.isfinite(anova_res["F"]):
        f_str = f"F({anova_res['df_between']},{anova_res['df_error']}) = {anova_res['F']:.2f}"
        p_str = _fmt_p(anova_res["p"])
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\n{f_str}, p {p_str}, eta2 = {anova_res['eta_sq']:.3f}",
                     fontsize=9, pad=4)
    else:
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=9)

    sig_pairs = pairwise[pairwise["sig"] != ""][["a", "b", "sig"]].values.tolist()
    _draw_sig_brackets(ax, agents, means, sems, sig_pairs)

    # Legend
    seen_types = set()
    handles = []
    for ag in agents:
        ag_type = "Human" if ag in agent_colors and list(agent_colors.keys()).index(ag) < len([a for a in agent_order if a in agent_colors]) else "AI"
        # Simpler: check if agent is in first half (human) or second half (AI)
        pass
    # Use first human and first AI color for legend
    human_color = colors[0] if colors else "#4C72B0"
    ai_color = colors[-1] if len(colors) > 1 else "#C44E52"
    handles = [
        mpatches.Patch(color=human_color, label="Human"),
        mpatches.Patch(color=ai_color, label="AI"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="upper right", framealpha=0.6)
    fig.tight_layout()
    return fig_to_b64(fig)


def _draw_sig_brackets(ax, agents, means, sems, sig_pairs):
    if not sig_pairs:
        return
    ymax  = max(m + s for m, s in zip(means, sems)) if means else 0
    step  = (ymax - ax.get_ylim()[0]) * 0.08
    ymax += step * 0.5
    agent_x = {a: i for i, a in enumerate(agents)}

    for i, (a, b, sig) in enumerate(sig_pairs[:4]):
        if a not in agent_x or b not in agent_x:
            continue
        x1, x2 = sorted([agent_x[a], agent_x[b]])
        y = ymax + step * i
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.2, y + step * 0.2, y],
                lw=0.9, color="0.3")
        ax.text((x1 + x2) / 2, y + step * 0.25, sig,
                ha="center", va="bottom", fontsize=8, color="0.2")
    ax.set_ylim(top=ymax + step * (min(len(sig_pairs), 4) + 1))


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < .001:
        return "< .001"
    if p < .01:
        return f"= .{int(round(p * 1000)):03d}"[:-1]
    return f"= {p:.3f}"


def plot_effect_heatmap(all_pairwise: Dict[str, pd.DataFrame],
                        metrics: List[str], title: str,
                        agent_order: List[str]) -> str:
    pairs = [f"{a}\nvs\n{b}" for a, b in combinations(agent_order, 2)]
    pair_keys = [f"{a} vs {b}" for a, b in combinations(agent_order, 2)]

    dz_mat  = np.full((len(metrics), len(pairs)), np.nan)
    sig_mat = np.full((len(metrics), len(pairs)), "", dtype=object)

    for i, m in enumerate(metrics):
        if m not in all_pairwise:
            continue
        df = all_pairwise[m]
        for j, pk in enumerate(pair_keys):
            row = df[df["pair"] == pk]
            if len(row) == 0:
                continue
            dz_mat[i, j] = row["dz"].values[0]
            sig_mat[i, j] = row["sig"].values[0]

    fig, ax = plt.subplots(figsize=(9, max(4, len(metrics) * 0.42)))
    vmax = np.nanmax(np.abs(dz_mat)) if not np.all(np.isnan(dz_mat)) else 1.0
    im   = ax.imshow(dz_mat, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, fontsize=7.5)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=7.5)
    ax.set_title(title, fontsize=10, pad=8)

    for i in range(len(metrics)):
        for j in range(len(pairs)):
            sig = sig_mat[i, j]
            if sig:
                ax.text(j, i, sig, ha="center", va="center", fontsize=8,
                        color="white" if abs(dz_mat[i, j]) > vmax * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Cohen's dz", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_pval_heatmap(all_pairwise: Dict[str, pd.DataFrame],
                      metrics: List[str], title: str,
                      agent_order: List[str]) -> str:
    pair_keys = [f"{a} vs {b}" for a, b in combinations(agent_order, 2)]
    pairs_display = [f"{a}\nvs\n{b}" for a, b in combinations(agent_order, 2)]

    p_mat   = np.full((len(metrics), len(pair_keys)), np.nan)
    sig_mat = np.full((len(metrics), len(pair_keys)), "", dtype=object)

    for i, m in enumerate(metrics):
        if m not in all_pairwise:
            continue
        df = all_pairwise[m]
        for j, pk in enumerate(pair_keys):
            row = df[df["pair"] == pk]
            if len(row) == 0:
                continue
            p = row["p_adj"].values[0]
            p_mat[i, j]   = -np.log10(np.clip(p, 1e-10, 1.0)) if np.isfinite(p) else np.nan
            sig_mat[i, j] = row["sig"].values[0]

    fig, ax = plt.subplots(figsize=(9, max(4, len(metrics) * 0.42)))
    vmax = max(3.0, np.nanmax(p_mat)) if not np.all(np.isnan(p_mat)) else 3.0
    im   = ax.imshow(p_mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)

    ax.set_xticks(range(len(pair_keys)))
    ax.set_xticklabels(pairs_display, fontsize=7.5)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=7.5)
    ax.set_title(title, fontsize=10, pad=8)

    for i in range(len(metrics)):
        for j in range(len(pair_keys)):
            sig = sig_mat[i, j]
            if sig:
                ax.text(j, i, sig, ha="center", va="center",
                        fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("-log10(p_adj)", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_type_effect_summary(type_results: pd.DataFrame) -> str:
    df = type_results.sort_values("dz", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.38)))

    colors = ["#C44E52" if dz > 0 else "#4C72B0" for dz in df["dz"]]
    y = np.arange(len(df))
    ax.barh(y, df["dz"], color=colors, alpha=0.85, edgecolor="white")

    for i, (_, row) in enumerate(df.iterrows()):
        if row["sig"]:
            ax.text(row["dz"] + (0.02 if row["dz"] >= 0 else -0.02), i,
                    row["sig"], va="center",
                    ha="left" if row["dz"] >= 0 else "right",
                    fontsize=8, color="0.2")

    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in df["metric"]], fontsize=8)
    ax.set_xlabel("Cohen's dz (positive = Human > AI)", fontsize=9)
    ax.axvline(0, color="0.5", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Effect Sizes: Human vs AI (averaged across individual partners)",
                 fontsize=10, pad=8)

    handles = [
        mpatches.Patch(color="#4C72B0", label="AI > Human"),
        mpatches.Patch(color="#C44E52", label="Human > AI"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.6)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── HTML generation ──────────────────────────────────────────────────────────

def _sig_badge(sig: str) -> str:
    colors = {"***": "#c0392b", "**": "#e67e22", "*": "#2980b9", "": "#95a5a6"}
    text   = sig if sig else "ns"
    c = colors.get(sig, "#95a5a6")
    return (f'<span style="background:{c};color:white;padding:1px 5px;'
            f'border-radius:3px;font-size:0.82em;font-weight:bold">{text}</span>')


def _p_cell(p: float, sig: str = "") -> str:
    if np.isnan(p):
        return "<td>--</td>"
    color = "#fdf2f2" if sig == "***" else "#fef9f0" if sig == "**" else \
            "#f0f8ff" if sig == "*" else "white"
    return f'<td style="background:{color}">{p:.4f} {_sig_badge(sig)}</td>'


def _num_cell(v: float, fmt: str = ".4f") -> str:
    if np.isnan(v):
        return "<td>--</td>"
    return f"<td>{v:{fmt}}</td>"


def build_html(
    sub_agent:    pd.DataFrame,
    anova_all:    Dict[str, Dict],
    pairwise_all: Dict[str, pd.DataFrame],
    agg_all:      Dict[str, pd.DataFrame],
    metrics:      List[str],
    bar_imgs:     Dict[str, str],
    dz_heatmap:   str,
    pval_heatmap: str,
    n_subjects:   int,
    generated:    str,
    version:      str,
    agent_order:  List[str],
    human_agents: List[str],
    ai_agents:    List[str],
    type_results: pd.DataFrame = None,
    type_effect_img: str = "",
) -> str:

    hum_str = ", ".join(human_agents)
    ai_str  = ", ".join(ai_agents)
    k = len(agent_order)
    df_err = (k - 1) * (n_subjects - 1)

    methods_html = textwrap.dedent(f"""
    <div class="card">
      <h2>Statistical Methods</h2>
      <ul>
        <li><b>N = {n_subjects} LLM subjects</b>, each completing conversations
            with {k} agents ({hum_str}, {ai_str}).</li>
        <li><b>Unit of analysis:</b> per-subject, per-agent mean.
            All tests are within-subject (paired).</li>
        <li><b>Omnibus test:</b> One-way repeated-measures ANOVA
            (k={k} agents; df = ({k-1}, {df_err}) for N={n_subjects}).
            Effect size: eta-squared.</li>
        <li><b>Post-hoc pairwise tests:</b> {k*(k-1)//2} paired t-tests per metric,
            BH-FDR correction within each metric.</li>
        <li><b>Aggregate contrasts:</b> Human avg vs AI avg (and per-human vs AI avg),
            BH-FDR corrected within family.</li>
        <li><b>Significance:</b> * p_adj &lt; .05, ** &lt; .01, *** &lt; .001.</li>
      </ul>
    </div>
    """)

    # Type-level summary
    type_html = ""
    if type_results is not None and not type_results.empty:
        type_rows = ""
        for _, row in type_results.iterrows():
            m = row["metric"]
            sig = row["sig"]
            bg  = "#fdf2f2" if sig == "***" else "#fef9f0" if sig == "**" else \
                  "#f0f8ff" if sig == "*" else "white"
            direction = "Human > AI" if row["dz"] > 0 else "AI > Human" if row["dz"] < 0 else "="
            type_rows += (
                f'<tr style="background:{bg}">'
                f'<td><b>{METRIC_LABELS.get(m, m)}</b></td>'
                f'{_num_cell(row["mean_human"])}'
                f'{_num_cell(row["sem_human"])}'
                f'{_num_cell(row["mean_ai"])}'
                f'{_num_cell(row["sem_ai"])}'
                f'{_num_cell(row["t"], ".3f")}'
                f'{_p_cell(row["p_raw"])}'
                f'{_p_cell(row["p_adj"], sig)}'
                f'{_num_cell(row["dz"], ".3f")}'
                f'<td>{direction}</td>'
                f'</tr>\n'
            )

        n_type_sig = type_results["rejected"].sum() if "rejected" in type_results.columns else 0
        type_html = f"""
        <div class="card">
          <h2>1 - Human vs AI Summary (averaged across individual partners)</h2>
          <p class="note">Human mean = average of {hum_str};
             AI mean = average of {ai_str}.
             Paired t-tests with BH-FDR correction across all {len(type_results)} metrics.
             <b>{n_type_sig} significant</b> after correction.</p>
          <table>
            <thead><tr>
              <th>Metric</th>
              <th>Human<br><small>Mean</small></th>
              <th><small>SEM</small></th>
              <th>AI<br><small>Mean</small></th>
              <th><small>SEM</small></th>
              <th>t</th>
              <th>p (raw)</th>
              <th>p (adj)</th>
              <th>dz</th>
              <th>Direction</th>
            </tr></thead>
            <tbody>{type_rows}</tbody>
          </table>
        </div>
        """

        if type_effect_img:
            type_html += f"""
            <div class="card">
              <h2>1b - Effect Sizes: Human vs AI (Cohen's dz)</h2>
              <img src="data:image/png;base64,{type_effect_img}" style="max-width:100%">
            </div>
            """

    # ANOVA summary
    anova_rows = ""
    for m in metrics:
        res = anova_all.get(m, {})
        if "error" in res:
            continue
        F   = res.get("F", np.nan)
        p   = res.get("p", np.nan)
        eta = res.get("eta_sq", np.nan)
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
        bg  = "#fdf2f2" if sig == "***" else "#fef9f0" if sig == "**" else \
              "#f0f8ff" if sig == "*" else "white"
        means = res.get("agent_means", {})
        sems  = res.get("agent_sems",  {})
        def ms(ag):
            m_ = means.get(ag, np.nan); s_ = sems.get(ag, np.nan)
            return f"{m_:.4f}<br><small>+/-{s_:.4f}</small>" if np.isfinite(m_) else "--"

        agent_cells = "".join(f"<td>{ms(ag)}</td>" for ag in agent_order)
        anova_rows += (
            f'<tr style="background:{bg}">'
            f'<td><b>{METRIC_LABELS.get(m, m)}</b></td>'
            f'{agent_cells}'
            f'<td>{"--" if np.isnan(F) else f"{F:.2f}"}</td>'
            f'{_p_cell(p, sig)}'
            f'<td>{"--" if np.isnan(eta) else f"{eta:.3f}"}</td>'
            f'</tr>\n'
        )

    agent_headers = "".join(f"<th>{ag}<br><small>M +/- SEM</small></th>" for ag in agent_order)
    anova_html = f"""
    <div class="card">
      <h2>2 - One-Way RM-ANOVA Summary ({k} agents)</h2>
      <table>
        <thead><tr>
          <th>Metric</th>
          {agent_headers}
          <th>F({k-1},{df_err})</th><th>p</th><th>eta2</th>
        </tr></thead>
        <tbody>{anova_rows}</tbody>
      </table>
    </div>
    """

    # Heatmaps
    heatmap_html = f"""
    <div class="card">
      <h2>3 - Pairwise Effect Sizes (Cohen's dz)</h2>
      <img src="data:image/png;base64,{dz_heatmap}" style="max-width:100%">
    </div>
    <div class="card">
      <h2>4 - FDR-Adjusted p-Values (-log10 scale)</h2>
      <img src="data:image/png;base64,{pval_heatmap}" style="max-width:100%">
    </div>
    """

    # Per-metric sections
    metric_sections = ""
    for group_name, group_metrics in METRIC_GROUPS.items():
        group_metrics_avail = [m for m in group_metrics if m in metrics]
        if not group_metrics_avail:
            continue

        metric_sections += f'<div class="card"><h2>5 - {group_name}</h2>'

        for m in group_metrics_avail:
            res  = anova_all.get(m, {})
            pw   = pairwise_all.get(m, pd.DataFrame())
            agg  = agg_all.get(m, pd.DataFrame())
            img  = bar_imgs.get(m, "")

            if "F" in res:
                F = res["F"]; p = res["p"]; eta = res["eta_sq"]
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                anov_line = (f"RM-ANOVA: <b>F({res['df_between']},{res['df_error']}) = "
                             f"{F:.2f}</b>, p {_fmt_p(p)} {_sig_badge(sig)}, eta2 = {eta:.3f}")
            else:
                anov_line = f"RM-ANOVA: {res.get('error', '--')}"

            metric_sections += f"""
            <div style="margin-bottom:28px">
              <h3 style="color:#333;margin-bottom:4px">{METRIC_LABELS.get(m, m)}</h3>
              <p style="margin:2px 0;font-size:0.93em">{anov_line}</p>
            """

            if not pw.empty:
                pw_rows = ""
                for _, r in pw.iterrows():
                    da = r["mean_a"]; db = r["mean_b"]
                    direction = (f"{r['a']} > {r['b']}" if da > db else f"{r['b']} > {r['a']}")
                    pw_rows += (
                        f"<tr><td>{r['pair']}</td>"
                        f"{_num_cell(r['mean_a'])}{_num_cell(r['mean_b'])}"
                        f"{_num_cell(r['t'], '.3f')}"
                        f"{_p_cell(r['p_raw'])}"
                        f"{_p_cell(r['p_adj'], r['sig'])}"
                        f"{_num_cell(r['dz'], '.3f')}"
                        f"<td>{direction}</td></tr>\n"
                    )
                metric_sections += f"""
                <details open>
                  <summary style="cursor:pointer;font-size:0.9em;color:#555">
                    Pairwise post-hoc (BH-FDR corrected)</summary>
                  <table style="font-size:0.85em;margin-top:6px">
                    <thead><tr>
                      <th>Pair</th><th>Mean A</th><th>Mean B</th>
                      <th>t</th><th>p (raw)</th><th>p (adj)</th>
                      <th>dz</th><th>Direction</th>
                    </tr></thead>
                    <tbody>{pw_rows}</tbody>
                  </table>
                </details>
                """

            if not agg.empty:
                agg_rows = ""
                for _, r in agg.iterrows():
                    agg_rows += (
                        f"<tr><td>{r['contrast']}</td>"
                        f"{_num_cell(r['mean_a'])}{_num_cell(r['mean_b'])}"
                        f"{_num_cell(r['t'], '.3f')}"
                        f"{_p_cell(r['p_raw'])}"
                        f"{_p_cell(r['p_adj'], r['sig'])}"
                        f"{_num_cell(r['dz'], '.3f')}"
                        f"<td style='font-size:0.78em;color:#555'>{r.get('note','')}</td></tr>\n"
                    )
                metric_sections += f"""
                <details>
                  <summary style="cursor:pointer;font-size:0.9em;color:#555">
                    Aggregate contrasts (planned, BH-FDR corrected)</summary>
                  <table style="font-size:0.85em;margin-top:6px">
                    <thead><tr>
                      <th>Contrast</th><th>Mean A</th><th>Mean B</th>
                      <th>t</th><th>p (raw)</th><th>p (adj)</th>
                      <th>dz</th><th>Note</th>
                    </tr></thead>
                    <tbody>{agg_rows}</tbody>
                  </table>
                </details>
                """

            if img:
                metric_sections += (f'<img src="data:image/png;base64,{img}" '
                                     f'style="max-width:660px;margin-top:10px">')

            metric_sections += "</div>"
        metric_sections += "</div>"

    # Interpretation
    sig_anova = [m for m in metrics if anova_all.get(m, {}).get("p", 1) < .05]
    sig_within_hum = []
    sig_within_ai  = []
    sig_cross      = []
    for m in metrics:
        pw = pairwise_all.get(m, pd.DataFrame())
        if pw.empty:
            continue
        # Within-human pairs
        for h1, h2 in combinations(human_agents, 2):
            gv = pw[((pw["a"] == h1) & (pw["b"] == h2)) | ((pw["a"] == h2) & (pw["b"] == h1))]["sig"].values
            if len(gv) and gv[0]:
                sig_within_hum.append(m)
                break
        # Within-AI pairs
        for a1, a2 in combinations(ai_agents, 2):
            cv = pw[((pw["a"] == a1) & (pw["b"] == a2)) | ((pw["a"] == a2) & (pw["b"] == a1))]["sig"].values
            if len(cv) and cv[0]:
                sig_within_ai.append(m)
                break
        # Cross-type
        cross = pw[((pw["a"].isin(human_agents)) & (pw["b"].isin(ai_agents))) |
                   ((pw["a"].isin(ai_agents))    & (pw["b"].isin(human_agents)))]
        if cross["sig"].any():
            sig_cross.append(m)

    def _bullet_list(items):
        if not items:
            return "<li><em>None</em></li>"
        return "\n".join(f"<li>{METRIC_LABELS.get(i, i)}</li>" for i in items)

    interp_html = f"""
    <div class="card">
      <h2>6 - Interpretation</h2>
      <h3>Overall RM-ANOVA: agent-level differences</h3>
      <p>{len(sig_anova)} of {len(metrics)} metrics showed a significant omnibus effect
      (p &lt; .05): {', '.join(METRIC_LABELS.get(m, m) for m in sig_anova) or 'none'}.
      </p>

      <h3>Within-human variation</h3>
      <p>Significant after BH-FDR in {len(sig_within_hum)} metric(s):</p>
      <ul>{_bullet_list(sig_within_hum)}</ul>

      <h3>Within-AI variation</h3>
      <p>Significant after BH-FDR in {len(sig_within_ai)} metric(s):</p>
      <ul>{_bullet_list(sig_within_ai)}</ul>

      <h3>Cross-type variation (any human vs any AI, post-hoc)</h3>
      <p>At least one human-AI pairwise comparison significant in
      {len(sig_cross)} metric(s):</p>
      <ul>{_bullet_list(sig_cross)}</ul>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Identity Breakdown - Exp 1 ({version})</title>
<style>
  body  {{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
           background:#f5f6fa;color:#222;margin:0;padding:0 }}
  h1    {{ background:#2c3e50;color:white;padding:18px 28px;margin:0;font-size:1.4em }}
  h2    {{ color:#2c3e50;border-bottom:2px solid #e0e0e0;padding-bottom:4px;font-size:1.15em }}
  h3    {{ color:#444;font-size:1.02em }}
  .card {{ background:white;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.1);
           margin:18px 24px;padding:20px 24px }}
  table {{ border-collapse:collapse;width:100%;font-size:0.88em }}
  th,td {{ border:1px solid #e0e0e0;padding:5px 8px;text-align:center }}
  th    {{ background:#f0f2f5;font-weight:600;white-space:nowrap }}
  td:first-child {{ text-align:left }}
  .note {{ color:#666;font-size:0.82em;margin-top:6px }}
  details > summary {{ padding:4px 0 }}
  img   {{ display:block;margin-top:6px }}
</style>
</head>
<body>
<h1>Identity Breakdown: {' - '.join(agent_order)}
    &nbsp;<small style="font-weight:normal;font-size:0.7em">
    Exp 1 - {version} | Generated: {generated}
    </small>
</h1>
{methods_html}
{type_html}
{anova_html}
{heatmap_html}
{metric_sections}
{interp_html}
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Identity breakdown analysis")
    args = parse_version_model(parser)

    vcfg = get_version_config()
    dd = str(data_dir())
    rd = str(results_dir())
    fd = str(figures_dir())
    os.makedirs(rd, exist_ok=True)
    os.makedirs(fd, exist_ok=True)

    agent_map, agent_order, human_agents, ai_agents, agent_colors = build_agent_info(vcfg)

    version = args.version

    print("=" * 65)
    print(f"IDENTITY BREAKDOWN ANALYSIS - {version}")
    print(f"Data dir    : {dd}")
    print(f"Results dir : {rd}")
    print(f"Agents      : {agent_order}")
    print("=" * 65)

    # Load & prepare data
    df_raw = load_combined_csv(dd)
    df_raw = compute_metrics(df_raw)
    sub_agent = aggregate_to_subject_agent(df_raw, agent_map)

    n_subjects = sub_agent["subject"].nunique()
    available_metrics = [m for m in RATE_METRICS if m in sub_agent.columns]
    print(f"\n[INFO] {len(available_metrics)} metrics available, "
          f"{n_subjects} subjects, agents: {sub_agent['partner'].unique().tolist()}")

    # Build wide dataframe per metric
    wide_per_metric: Dict[str, pd.DataFrame] = {}
    for m in available_metrics:
        w = sub_agent.pivot(index="subject", columns="partner", values=m)
        w = w[[a for a in agent_order if a in w.columns]]
        wide_per_metric[m] = w

    # Run statistics
    print("\n[INFO] Running RM-ANOVAs ...")
    anova_all: Dict[str, Dict] = {}
    for m in available_metrics:
        w = wide_per_metric[m]
        present = [a for a in agent_order if a in w.columns]
        anova_all[m] = rm_anova_one_way(w, present)

    print("[INFO] Running pairwise tests ...")
    pairwise_all: Dict[str, pd.DataFrame] = {}
    for m in available_metrics:
        w = wide_per_metric[m]
        present = [a for a in agent_order if a in w.columns]
        pairwise_all[m] = pairwise_tests(w, present)

    print("[INFO] Running aggregate contrasts ...")
    agg_all: Dict[str, pd.DataFrame] = {}
    for m in available_metrics:
        w = wide_per_metric[m]
        agents_present = [a for a in agent_order if a in w.columns]
        if not all(a in agents_present for a in agent_order):
            agg_all[m] = pd.DataFrame()
            continue
        agg_all[m] = aggregate_contrasts(w, agent_order, human_agents, ai_agents)

    # Type-level analysis
    type_results = type_level_analysis(wide_per_metric, available_metrics,
                                       human_agents, ai_agents)

    # Save stats text
    txt_path = os.path.join(rd, "identity_breakdown_stats.txt")
    with open(txt_path, "w") as f:
        f.write(f"IDENTITY BREAKDOWN STATS - {version}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for m in available_metrics:
            f.write(f"\n{'─'*60}\n{METRIC_LABELS.get(m, m)}\n{'─'*60}\n")
            res = anova_all.get(m, {})
            if "F" in res:
                p = res["p"]
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                f.write(f"RM-ANOVA: F({res['df_between']},{res['df_error']}) = "
                        f"{res['F']:.3f}, p = {p:.4f} {sig}, eta2 = {res['eta_sq']:.3f}, "
                        f"N = {res['n_subjects']}\n")
                f.write("Agent means:\n")
                for ag in agent_order:
                    mn = res['agent_means'].get(ag, np.nan)
                    se = res['agent_sems'].get(ag, np.nan)
                    f.write(f"  {ag:12s}: M = {mn:.5f} +/- {se:.5f}\n")
            else:
                f.write(f"RM-ANOVA: {res.get('error', '--')}\n")

            pw = pairwise_all.get(m, pd.DataFrame())
            if not pw.empty:
                f.write("\nPost-hoc pairwise (BH-FDR):\n")
                for _, r in pw.iterrows():
                    sig = r.get("sig", "")
                    f.write(f"  {r['pair']:<30s}: t = {r['t']:.3f}, "
                            f"p_raw = {r['p_raw']:.4f}, p_adj = {r['p_adj']:.4f} "
                            f"{sig}, dz = {r['dz']:.3f}\n")

            agg = agg_all.get(m, pd.DataFrame())
            if not agg.empty:
                f.write("\nAggregate contrasts (BH-FDR):\n")
                for _, r in agg.iterrows():
                    sig = r.get("sig", "")
                    f.write(f"  {r['contrast']:<40s}: t = {r['t']:.3f}, "
                            f"p_raw = {r['p_raw']:.4f}, p_adj = {r['p_adj']:.4f} "
                            f"{sig}, dz = {r['dz']:.3f}\n")

        if not type_results.empty:
            f.write(f"\n\n{'='*70}\nTYPE-LEVEL SUMMARY (Human avg vs AI avg)\n{'='*70}\n")
            f.write(f"BH-FDR correction across {len(type_results)} metrics\n\n")
            for _, r in type_results.iterrows():
                m = r["metric"]
                sig = r.get("sig", "")
                direction = "Human > AI" if r["dz"] > 0 else "AI > Human"
                f.write(f"  {METRIC_LABELS.get(m, m):<35s}: "
                        f"t({r['n']-1}) = {r['t']:.3f}, "
                        f"p_raw = {r['p_raw']:.4f}, p_adj = {r['p_adj']:.4f} {sig}, "
                        f"dz = {r['dz']:.3f}, {direction}\n")
    print(f"[OK] Stats text saved -> {txt_path}")

    # Generate figures
    print("[INFO] Generating type-level effect summary ...")
    type_effect_img = ""
    if not type_results.empty:
        type_effect_img = plot_type_effect_summary(type_results)

    print("[INFO] Generating bar charts ...")
    bar_imgs: Dict[str, str] = {}
    for m in available_metrics:
        try:
            bar_imgs[m] = plot_bar_metric(sub_agent, m,
                                          pairwise_all.get(m, pd.DataFrame()),
                                          anova_all.get(m, {}),
                                          agent_order, agent_colors)
        except Exception as e:
            print(f"  [WARN] Bar chart failed for {m}: {e}")

    print("[INFO] Generating heatmaps ...")
    dz_heatmap   = plot_effect_heatmap(pairwise_all, available_metrics,
                                        "Cohen's dz -- All Metrics x All Pairs",
                                        agent_order)
    pval_heatmap = plot_pval_heatmap(pairwise_all, available_metrics,
                                      "FDR-adjusted p-values -- All Metrics x All Pairs",
                                      agent_order)

    # Build and save HTML
    print("[INFO] Building HTML report ...")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = build_html(
        sub_agent       = sub_agent,
        anova_all       = anova_all,
        pairwise_all    = pairwise_all,
        agg_all         = agg_all,
        metrics         = available_metrics,
        bar_imgs        = bar_imgs,
        dz_heatmap      = dz_heatmap,
        pval_heatmap    = pval_heatmap,
        n_subjects      = n_subjects,
        generated       = generated,
        version         = version,
        agent_order     = agent_order,
        human_agents    = human_agents,
        ai_agents       = ai_agents,
        type_results    = type_results,
        type_effect_img = type_effect_img,
    )

    html_path = os.path.join(rd, "identity_breakdown.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML report saved -> {html_path}")

    # Save summary CSV
    rows = []
    for m in available_metrics:
        res = anova_all.get(m, {})
        row = {"metric": m, "label": METRIC_LABELS.get(m, m)}
        row["anova_F"]    = res.get("F", np.nan)
        row["anova_p"]    = res.get("p", np.nan)
        row["anova_eta2"] = res.get("eta_sq", np.nan)
        row["anova_sig"]  = ("***" if row["anova_p"] < .001 else
                             "**"  if row["anova_p"] < .01  else
                             "*"   if row["anova_p"] < .05  else "")
        for ag in agent_order:
            row[f"mean_{ag.lower().replace(' ', '_')}"] = res.get("agent_means", {}).get(ag, np.nan)
            row[f"sem_{ag.lower().replace(' ', '_')}"]  = res.get("agent_sems",  {}).get(ag, np.nan)
        for _, r in pairwise_all.get(m, pd.DataFrame()).iterrows():
            key = r["pair"].replace(" ", "_")
            row[f"dz_{key}"]   = r["dz"]
            row[f"padj_{key}"] = r["p_adj"]
            row[f"sig_{key}"]  = r["sig"]
        rows.append(row)

    csv_path = os.path.join(dd, "identity_breakdown_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[OK] Summary CSV saved -> {csv_path}")

    print("\n" + "=" * 65)
    print("DONE. Output files:")
    print(f"  {html_path}")
    print(f"  {txt_path}")
    print(f"  {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
