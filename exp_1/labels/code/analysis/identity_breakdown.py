#!/usr/bin/env python3
"""
Script: identity_breakdown.py

Purpose:
    Type-level behavioral analysis (Human vs AI) in the labels exp_1 dataset.

    Unlike the named-partner versions (names/, balanced_names/, balanced_gpt/),
    the labels version has only two partner conditions ("a human" / "an AI")
    with no individual identity variation. This script performs a simpler
    analysis: paired t-tests per metric with BH-FDR correction.

Statistics:
    1. Paired t-test per metric (Human vs AI, N=50 subjects).
       Benjamini-Hochberg (BH) FDR correction across all metrics.
       Effect size: Cohen's dz = mean(diff) / sd(diff).

Output:
    results/meta-llama-Llama-2-13b-chat-hf/0.8/identity_breakdown.html
    results/meta-llama-Llama-2-13b-chat-hf/0.8/identity_breakdown_stats.txt
    results/meta-llama-Llama-2-13b-chat-hf/0.8/identity_breakdown_summary.csv

Run from:
    exp_1/labels/code/analysis/

Author: Rachel C. Metzgar
Date: 2026-02-20
"""

from __future__ import annotations

import base64
import io
import os
import re
import sys
import warnings
import textwrap
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

# ── Import shared word lists from project utils ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils.hedges_demir import (
    DEMIR_ALL_HEDGES, DEMIR_NOUNS, DEMIR_ADJECTIVES, DEMIR_ADVERBS,
    DEMIR_VERBS, DEMIR_QUANTIFIERS, DEMIR_MODALS,
)
from utils.discourse_markers_fung import (
    FUNG_INTERPERSONAL, FUNG_REFERENTIAL, FUNG_STRUCTURAL, FUNG_COGNITIVE,
    FUNG_ALL_23_MARKERS,
)
from utils.misc_text_markers import (
    LIWC_NONFLUENCIES, LIWC_FILLERS,
    TOM_PHRASES, POLITE_POSITIVE, POLITE_NEGATIVE, IMPOLITE, LIKE_MARKER,
)

# ── Agent mapping ─────────────────────────────────────────────────────────────
# labels/ uses only type labels, no individual names.
# bot_1/bot_2 both map to "AI"; hum_1/hum_2 both map to "Human".
AGENT_MAP = {
    "bot_1": "AI",
    "bot_2": "AI",
    "hum_1": "Human",
    "hum_2": "Human",
}
CONDITION_ORDER = ["Human", "AI"]

CONDITION_COLORS = {
    "Human": "#4C72B0",
    "AI":    "#C44E52",
}

# ── Metrics to analyse ────────────────────────────────────────────────────────
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

# ── Data I/O ──────────────────────────────────────────────────────────────────

def find_project_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    code_dir  = os.path.abspath(os.path.join(here, ".."))
    proj_root = os.path.abspath(os.path.join(code_dir, ".."))
    env_root  = os.environ.get("PROJECT_ROOT", "").strip()
    return env_root or proj_root


def load_combined_csv(data_dir: str) -> pd.DataFrame:
    combined = os.path.join(data_dir, "combined_text_data.csv")
    if os.path.exists(combined):
        print(f"[INFO] Loading combined CSV: {combined}")
        try:
            df = pd.read_csv(combined, on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(combined)
        print(f"  Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
        return df

    print(f"[WARN] No combined CSV found; loading per-subject files from {data_dir}")
    frames = []
    for fname in sorted(os.listdir(data_dir)):
        if re.match(r"s\d+\.csv$", fname):
            frames.append(pd.read_csv(os.path.join(data_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Concatenated {len(frames)} subject files -> {len(df)} rows")
    return df


# ── Metric computation ────────────────────────────────────────────────────────

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

    df["word_count"]    = t.apply(_count_words)
    df["question_count"]= t.apply(lambda x: str(x).count("?") if isinstance(x, str) else 0)

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


def aggregate_to_subject_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate utterance-level data -> subject x condition means.

    Strategy (mirrors cross_experiment_comparison.py):
      1. Sum counts within each trial (subject x condition x topic).
      2. Recompute rates from summed counts per trial.
      3. Average trials -> subject x condition mean.
    """
    print("[INFO] Aggregating to subject x condition level ...")

    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # Map agent codes -> type condition
    df["condition"] = df["agent"].map(AGENT_MAP)
    unknown = df["condition"].isna()
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

    # Step 1: sum counts per trial (subject x condition x topic)
    avail_count = [c for c in COUNT_COLS if c in df.columns]
    avail_sent  = [c for c in ["sentiment"] if c in df.columns]
    avail_rat   = [c for c in ["quality", "connectedness"] if c in df.columns]

    agg_dict = {c: "sum" for c in avail_count}
    agg_dict.update({c: "mean" for c in avail_sent + avail_rat})

    group_cols = ["subject", "condition", "topic"]
    group_cols = [c for c in group_cols if c in df.columns]

    trial_df = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Step 2: recompute rates at trial level
    wc = trial_df["word_count"].replace(0, np.nan)
    for rate_col, count_col in RATE_PAIRS:
        if count_col in trial_df.columns:
            trial_df[rate_col] = trial_df[count_col] / wc

    # Step 3: mean across topics -> subject x condition
    rate_cols_avail = [c for c in RATE_METRICS if c in trial_df.columns]
    sub_cond = (trial_df.groupby(["subject", "condition"])[rate_cols_avail]
                         .mean()
                         .reset_index())

    n_subj = sub_cond["subject"].nunique()
    n_conds = sub_cond["condition"].nunique()
    print(f"  {n_subj} subjects x {n_conds} conditions -> {len(sub_cond)} rows")
    return sub_cond


# ── Statistical helpers ───────────────────────────────────────────────────────

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


# ── Figure generation ─────────────────────────────────────────────────────────

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


def _fmt_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < .001:
        return "< .001"
    if p < .01:
        return f"= .{int(round(p * 1000)):03d}"[:-1]
    return f"= {p:.3f}"


def plot_bar_metric(sub_cond: pd.DataFrame, metric: str,
                    t_stat: float, p_adj: float, sig: str, dz: float) -> str:
    """Bar chart: Human vs AI, with significance bracket if significant."""
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    means = []
    sems  = []
    for cond in CONDITION_ORDER:
        vals = sub_cond.loc[sub_cond["condition"] == cond, metric].dropna()
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)

    x      = np.arange(len(CONDITION_ORDER))
    colors = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
    bars   = ax.bar(x, means, yerr=sems, color=colors, width=0.45,
                    capsize=5, alpha=0.88, edgecolor="white", linewidth=0.8,
                    error_kw={"elinewidth": 1.4, "ecolor": "grey"})

    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_ORDER, fontsize=10)
    ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

    # Title with t-test result
    t_str = f"t(49) = {t_stat:.2f}" if np.isfinite(t_stat) else "t = n/a"
    p_str = _fmt_p(p_adj)
    sig_str = f" {sig}" if sig else ""
    ax.set_title(f"{METRIC_LABELS.get(metric, metric)}\n{t_str}, p_adj {p_str}{sig_str}, dz = {dz:.3f}",
                 fontsize=9, pad=4)

    # Significance bracket
    if sig:
        ymax = max(m + s for m, s in zip(means, sems))
        step = (ymax - ax.get_ylim()[0]) * 0.08
        y = ymax + step * 0.5
        ax.plot([0, 0, 1, 1], [y, y + step * 0.2, y + step * 0.2, y],
                lw=0.9, color="0.3")
        ax.text(0.5, y + step * 0.25, sig,
                ha="center", va="bottom", fontsize=9, color="0.2")
        ax.set_ylim(top=y + step * 2.5)

    fig.tight_layout()
    return fig_to_b64(fig)


def plot_effect_summary(results_df: pd.DataFrame) -> str:
    """Horizontal bar chart of Cohen's dz for all metrics."""
    df = results_df.sort_values("dz", ascending=True).copy()
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
    ax.set_title("Effect Sizes: Human vs AI (all metrics)", fontsize=10, pad=8)

    # Legend
    handles = [
        mpatches.Patch(color="#4C72B0", label="AI > Human"),
        mpatches.Patch(color="#C44E52", label="Human > AI"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.6)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── HTML generation ───────────────────────────────────────────────────────────

def _sig_badge(sig: str) -> str:
    colors = {"***": "#c0392b", "**": "#e67e22", "*": "#2980b9", "": "#95a5a6"}
    text   = sig if sig else "ns"
    c = colors.get(sig, "#95a5a6")
    return (f'<span style="background:{c};color:white;padding:1px 5px;'
            f'border-radius:3px;font-size:0.82em;font-weight:bold">{text}</span>')


def _p_cell(p: float, sig: str = "") -> str:
    if np.isnan(p):
        return "<td>---</td>"
    color = "#fdf2f2" if sig == "***" else "#fef9f0" if sig == "**" else \
            "#f0f8ff" if sig == "*" else "white"
    return f'<td style="background:{color}">{p:.4f} {_sig_badge(sig)}</td>'


def _num_cell(v: float, fmt: str = ".4f") -> str:
    if np.isnan(v):
        return "<td>---</td>"
    return f"<td>{v:{fmt}}</td>"


def build_html(
    results_df:   pd.DataFrame,
    bar_imgs:     Dict[str, str],
    effect_img:   str,
    n_subjects:   int,
    generated:    str,
) -> str:

    methods_html = textwrap.dedent(f"""
    <div class="card">
      <h2>Statistical Methods</h2>
      <ul>
        <li><b>N = {n_subjects} LLM subjects</b>, each completing 20 conversations
            per condition (40 total: 2 conditions x 20 topics).</li>
        <li><b>Unit of analysis:</b> per-subject, per-condition mean (aggregated
            over 20 trials). All tests are within-subject (paired).</li>
        <li><b>Test:</b> Paired <i>t</i>-test (df = N-1 = {n_subjects - 1}) per metric.
            Effect size: Cohen's d<sub>z</sub>
            (mean of differences / SD of differences).</li>
        <li><b>Multiple comparison correction:</b> Benjamini-Hochberg (BH) FDR
            applied across all {len(results_df)} metrics.</li>
        <li><b>Significance:</b> * p<sub>adj</sub> &lt; .05, ** &lt; .01, *** &lt; .001.</li>
        <li><b>Note:</b> This is the <code>labels/</code> version with generic
            type labels ("a human" / "an AI") and no individual partner names.</li>
      </ul>
    </div>
    """)

    # Summary table
    table_rows = ""
    for _, row in results_df.iterrows():
        m = row["metric"]
        sig = row["sig"]
        bg  = "#fdf2f2" if sig == "***" else "#fef9f0" if sig == "**" else \
              "#f0f8ff" if sig == "*" else "white"
        direction = "Human > AI" if row["dz"] > 0 else "AI > Human" if row["dz"] < 0 else "="
        table_rows += (
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

    summary_html = f"""
    <div class="card">
      <h2>1 - Summary: Human vs AI (all metrics)</h2>
      <table>
        <thead><tr>
          <th>Metric</th>
          <th>Human<br><small>Mean</small></th>
          <th><small>SEM</small></th>
          <th>AI<br><small>Mean</small></th>
          <th><small>SEM</small></th>
          <th><i>t</i></th>
          <th>p (raw)</th>
          <th>p (adj)</th>
          <th>d<sub>z</sub></th>
          <th>Direction</th>
        </tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
      <p class="note">Row highlighting: red = p&lt;.001, orange = p&lt;.01, blue = p&lt;.05.
      BH-FDR correction applied across all {len(results_df)} metrics.</p>
    </div>
    """

    # Effect size chart
    effect_html = f"""
    <div class="card">
      <h2>2 - Effect Sizes (Cohen's d<sub>z</sub>)</h2>
      <p class="note">Positive dz = Human > AI; negative = AI > Human.
         Asterisks indicate BH-FDR significance.</p>
      <img src="data:image/png;base64,{effect_img}" style="max-width:100%">
    </div>
    """

    # Per-metric bar charts (grouped)
    metric_sections = ""
    for group_name, group_metrics in METRIC_GROUPS.items():
        group_avail = [m for m in group_metrics if m in bar_imgs]
        if not group_avail:
            continue

        metric_sections += f'<div class="card"><h2>3 - {group_name}</h2>'
        for m in group_avail:
            img = bar_imgs.get(m, "")
            if img:
                metric_sections += (f'<img src="data:image/png;base64,{img}" '
                                     f'style="max-width:500px;margin:10px 0">')
        metric_sections += "</div>"

    # Interpretation
    sig_metrics = results_df[results_df["sig"] != ""]
    n_sig = len(sig_metrics)
    n_total = len(results_df)
    human_higher = sig_metrics[sig_metrics["dz"] > 0]
    ai_higher = sig_metrics[sig_metrics["dz"] < 0]

    interp_html = f"""
    <div class="card">
      <h2>4 - Interpretation</h2>
      <p><b>{n_sig} of {n_total} metrics</b> showed a significant Human vs AI
      difference after BH-FDR correction.</p>

      <h3>Human > AI ({len(human_higher)} metrics)</h3>
      <ul>
        {"".join(f"<li>{METRIC_LABELS.get(m, m)}</li>" for m in human_higher["metric"]) or "<li><em>None</em></li>"}
      </ul>

      <h3>AI > Human ({len(ai_higher)} metrics)</h3>
      <ul>
        {"".join(f"<li>{METRIC_LABELS.get(m, m)}</li>" for m in ai_higher["metric"]) or "<li><em>None</em></li>"}
      </ul>

      <h3>Notes</h3>
      <ul>
        <li>This is the <b>labels</b> version: partners are identified only as
            "a human" or "an AI" with no individual names. Any behavioral differences
            reflect the type distinction itself, free of name-specific associations.</li>
        <li>Compare with the <code>names/</code> and <code>balanced_names/</code>
            versions to assess how much individual partner identity (vs type label)
            drives behavioral adaptation.</li>
      </ul>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Type-Level Behavioral Analysis -- Exp 1 (labels)</title>
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
  img   {{ display:block;margin-top:6px }}
</style>
</head>
<body>
<h1>Human vs AI: Type-Level Behavioral Analysis
    &nbsp;<small style="font-weight:normal;font-size:0.7em">
    Exp 1 -- labels -- LLaMA-2-13B-Chat &nbsp;|&nbsp; Generated: {generated}
    </small>
</h1>
{methods_html}
{summary_html}
{effect_html}
{metric_sections}
{interp_html}
</body>
</html>"""
    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    project_root = find_project_root()
    data_dir     = os.path.join(project_root, "data",
                                "meta-llama-Llama-2-13b-chat-hf", "0.8")
    results_dir  = os.path.join(project_root, "results",
                                "meta-llama-Llama-2-13b-chat-hf", "0.8")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("TYPE-LEVEL BEHAVIORAL ANALYSIS -- labels")
    print(f"Project root : {project_root}")
    print(f"Data dir     : {data_dir}")
    print(f"Results dir  : {results_dir}")
    print("=" * 65)

    # ── Load & prepare data ───────────────────────────────────────────────────
    df_raw = load_combined_csv(data_dir)
    df_raw = compute_metrics(df_raw)
    sub_cond = aggregate_to_subject_condition(df_raw)

    n_subjects = sub_cond["subject"].nunique()
    available_metrics = [m for m in RATE_METRICS if m in sub_cond.columns]
    print(f"\n[INFO] {len(available_metrics)} metrics available, "
          f"{n_subjects} subjects, conditions: {sub_cond['condition'].unique().tolist()}")

    # Build wide dataframe per metric
    wide_per_metric: Dict[str, pd.DataFrame] = {}
    for m in available_metrics:
        w = sub_cond.pivot(index="subject", columns="condition", values=m)
        w = w[[c for c in CONDITION_ORDER if c in w.columns]]
        wide_per_metric[m] = w

    # ── Run paired t-tests ────────────────────────────────────────────────────
    print("\n[INFO] Running paired t-tests ...")
    rows = []
    for m in available_metrics:
        w = wide_per_metric[m]
        if "Human" not in w.columns or "AI" not in w.columns:
            continue
        data = w[["Human", "AI"]].dropna()
        human_vals = data["Human"].values
        ai_vals    = data["AI"].values
        if len(human_vals) < 3:
            continue

        t_stat, p_raw = ttest_rel(human_vals, ai_vals)
        dz = cohens_dz(human_vals, ai_vals)
        rows.append({
            "metric":     m,
            "mean_human": human_vals.mean(),
            "sem_human":  human_vals.std(ddof=1) / np.sqrt(len(human_vals)),
            "mean_ai":    ai_vals.mean(),
            "sem_ai":     ai_vals.std(ddof=1) / np.sqrt(len(ai_vals)),
            "t":          t_stat,
            "p_raw":      p_raw,
            "dz":         dz,
            "n":          len(human_vals),
        })

    results_df = pd.DataFrame(rows)

    # BH-FDR correction across all metrics
    pvals = results_df["p_raw"].values
    adj, rejected = bh_fdr(np.where(np.isnan(pvals), 1.0, pvals))
    results_df["p_adj"]    = adj
    results_df["rejected"] = rejected
    results_df["sig"] = results_df["p_adj"].apply(
        lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
    )

    n_sig = results_df["rejected"].sum()
    print(f"  {n_sig} of {len(results_df)} metrics significant after BH-FDR")

    # ── Save stats text ───────────────────────────────────────────────────────
    txt_path = os.path.join(results_dir, "identity_breakdown_stats.txt")
    with open(txt_path, "w") as f:
        f.write(f"TYPE-LEVEL BEHAVIORAL ANALYSIS STATS (labels)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"N = {n_subjects} subjects, 2 conditions (Human, AI)\n")
        f.write(f"BH-FDR correction across {len(results_df)} metrics\n")
        f.write("=" * 70 + "\n\n")

        for _, row in results_df.iterrows():
            m = row["metric"]
            sig = row["sig"]
            direction = "Human > AI" if row["dz"] > 0 else "AI > Human"
            f.write(f"\n{'_'*60}\n{METRIC_LABELS.get(m, m)}\n{'_'*60}\n")
            f.write(f"  Human: M = {row['mean_human']:.5f} +/- {row['sem_human']:.5f}\n")
            f.write(f"  AI:    M = {row['mean_ai']:.5f} +/- {row['sem_ai']:.5f}\n")
            f.write(f"  t({row['n']-1}) = {row['t']:.3f}, "
                    f"p_raw = {row['p_raw']:.4f}, p_adj = {row['p_adj']:.4f} {sig}, "
                    f"dz = {row['dz']:.3f}, {direction}\n")
    print(f"[OK] Stats text saved -> {txt_path}")

    # ── Generate figures ──────────────────────────────────────────────────────
    print("[INFO] Generating bar charts ...")
    bar_imgs: Dict[str, str] = {}
    for _, row in results_df.iterrows():
        m = row["metric"]
        try:
            bar_imgs[m] = plot_bar_metric(sub_cond, m,
                                           row["t"], row["p_adj"],
                                           row["sig"], row["dz"])
        except Exception as e:
            print(f"  [WARN] Bar chart failed for {m}: {e}")

    print("[INFO] Generating effect summary ...")
    effect_img = plot_effect_summary(results_df)

    # ── Build and save HTML ───────────────────────────────────────────────────
    print("[INFO] Building HTML report ...")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = build_html(
        results_df = results_df,
        bar_imgs   = bar_imgs,
        effect_img = effect_img,
        n_subjects = n_subjects,
        generated  = generated,
    )

    html_path = os.path.join(results_dir, "identity_breakdown.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML report saved -> {html_path}")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(results_dir, "identity_breakdown_summary.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"[OK] Summary CSV saved -> {csv_path}")

    print("\n" + "=" * 65)
    print("DONE. Output files:")
    print(f"  {html_path}")
    print(f"  {txt_path}")
    print(f"  {csv_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
