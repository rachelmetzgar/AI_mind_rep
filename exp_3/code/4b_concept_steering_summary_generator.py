#!/usr/bin/env python3
"""
Experiment 3: Concept Steering V1 — HTML Summary Report Generator

Discovers all concept steering V1 results, computes behavioral metrics,
runs statistical tests, and generates a single HTML report with:
  - Clickable table of contents
  - Cross-cell overview table (all dims x strategies x strengths)
  - Per-cell detail sections with full metric tables and pairwise comparisons

Reads from:  results/{model}/{version}/concept_steering/v1/{dim}/{strategy}/N_{N}_results.csv
Writes to:   results/{model}/{version}/concept_steering/v1/concept_steering_summary.html

Usage:
    python 4b_concept_steering_summary_generator.py --version balanced_gpt

Env: behavior_env
Rachel C. Metzgar · Mar 2026
"""

import os
import re
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import config, set_version, add_variant_argument, set_variant

# ============================================================
# Import Exp 1 linguistic marker utils
# ============================================================

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
# METRIC DEFINITIONS
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
    "sentiment",
]

# Human-readable labels for metrics
METRIC_LABELS = {
    "word_count": "Word Count",
    "question_count": "Question Count",
    "demir_modal_rate": "Hedges: Modal",
    "demir_verb_rate": "Hedges: Verb",
    "demir_adverb_rate": "Hedges: Adverb",
    "demir_adjective_rate": "Hedges: Adjective",
    "demir_quantifier_rate": "Hedges: Quantifier",
    "demir_noun_rate": "Hedges: Noun",
    "demir_total_rate": "Hedges: Total",
    "fung_interpersonal_rate": "Discourse: Interpersonal",
    "fung_referential_rate": "Discourse: Referential",
    "fung_structural_rate": "Discourse: Structural",
    "fung_cognitive_rate": "Discourse: Cognitive",
    "fung_total_rate": "Discourse: Total",
    "nonfluency_rate": "Nonfluency",
    "liwc_filler_rate": "Fillers (LIWC)",
    "disfluency_rate": "Disfluency (total)",
    "like_rate": "'Like' Rate",
    "tom_rate": "Theory of Mind",
    "politeness_rate": "Politeness",
    "sentiment": "Sentiment (VADER)",
}

# Metric groups for visual separation in tables
METRIC_GROUPS = [
    ("Response Length", ["word_count", "question_count"]),
    ("Hedges (Demir)", [
        "demir_modal_rate", "demir_verb_rate", "demir_adverb_rate",
        "demir_adjective_rate", "demir_quantifier_rate", "demir_noun_rate",
        "demir_total_rate",
    ]),
    ("Discourse Markers (Fung)", [
        "fung_interpersonal_rate", "fung_referential_rate",
        "fung_structural_rate", "fung_cognitive_rate", "fung_total_rate",
    ]),
    ("Fluency & Style", [
        "nonfluency_rate", "liwc_filler_rate", "disfluency_rate",
        "like_rate", "tom_rate", "politeness_rate", "sentiment",
    ]),
]


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 3: Generate HTML summary of concept steering V1 behavioral results."
    )
    p.add_argument("--version", required=True,
                   help="Exp version (e.g., balanced_gpt)")
    add_variant_argument(p)
    return p.parse_args()


# ============================================================
# DATA LOADING & DISCOVERY
# ============================================================

def discover_cells(v1_root):
    """Discover all (dim, strategy, N) cells with results CSVs."""
    cells = []
    csv_pattern = os.path.join(v1_root, "*", "*", "N_*_results.csv")
    for csv_path in sorted(glob.glob(csv_pattern)):
        parts = csv_path.split(os.sep)
        filename = parts[-1]
        strategy = parts[-2]
        dim = parts[-3]
        m = re.match(r"N_(\d+)_results\.csv", filename)
        if not m:
            continue
        N = int(m.group(1))
        cells.append({
            "dim": dim, "strategy": strategy, "N": N,
            "csv_path": csv_path,
            "dir": os.path.dirname(csv_path),
        })
    return cells


def load_config_json(cell_dir, N):
    """Load generation config JSON if available."""
    cfg_path = os.path.join(cell_dir, f"N_{N}_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return None


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
        df[col] = df["transcript_sub"].apply(lambda x, p=patterns: count_patterns(x, p))
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
        df[col] = df["transcript_sub"].apply(lambda x, p=patterns: count_patterns(x, p))
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

PAIR_ORDER = [("human", "ai"), ("human", "baseline"), ("ai", "baseline")]
PAIR_LABELS = {
    ("human", "ai"): "Human vs AI",
    ("human", "baseline"): "Human vs Baseline",
    ("ai", "baseline"): "AI vs Baseline",
}


def run_tests(df, metric):
    """Run pairwise independent t-tests for one metric."""
    result = {"metric": metric}
    for cond in ["baseline", "human", "ai"]:
        vals = df[df["condition"] == cond][metric].dropna()
        result[f"{cond}_mean"] = vals.mean() if len(vals) > 0 else np.nan
        result[f"{cond}_sem"] = vals.sem() if len(vals) > 0 else np.nan
        result[f"{cond}_n"] = len(vals)

    pairwise = {}
    for c1, c2 in PAIR_ORDER:
        v1 = df[df["condition"] == c1][metric].dropna()
        v2 = df[df["condition"] == c2][metric].dropna()
        if len(v1) < 2 or len(v2) < 2:
            pairwise[(c1, c2)] = {"t": np.nan, "p": np.nan, "diff": np.nan}
            continue
        t, p = ttest_ind(v1, v2)
        pairwise[(c1, c2)] = {"t": t, "p": p, "diff": v1.mean() - v2.mean()}
    result["pairwise"] = pairwise
    return result


def sig_stars(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def sig_class(p):
    """CSS class for significance level."""
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "sig3"
    if p < 0.01:
        return "sig2"
    if p < 0.05:
        return "sig1"
    return ""


# ============================================================
# HTML GENERATION
# ============================================================

CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         margin: 20px 40px; color: #1a1a1a; background: #fafafa; max-width: 1400px; }
  h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }
  h2 { color: #2c3e50; border-bottom: 2px solid #bdc3c7; padding-bottom: 4px; margin-top: 40px; }
  h3 { color: #34495e; margin-top: 30px; }
  h4 { color: #7f8c8d; margin-top: 20px; }

  /* TOC */
  .toc { background: #ecf0f1; padding: 15px 25px; border-radius: 8px;
         margin: 20px 0 30px 0; }
  .toc ul { list-style: none; padding-left: 0; }
  .toc ul ul { padding-left: 20px; }
  .toc li { margin: 3px 0; }
  .toc a { text-decoration: none; color: #2980b9; }
  .toc a:hover { text-decoration: underline; }

  /* Tables */
  table { border-collapse: collapse; margin: 10px 0 20px 0; font-size: 13px; }
  th, td { padding: 5px 10px; border: 1px solid #ddd; text-align: right; }
  th { background: #34495e; color: white; font-weight: 600; text-align: center; }
  td:first-child { text-align: left; font-weight: 500; }
  tr:nth-child(even) { background: #f8f9fa; }
  tr:hover { background: #eaf2f8; }

  /* Group headers in tables */
  .group-header td { background: #d5dbdb; font-weight: 700; font-size: 12px;
                     text-transform: uppercase; letter-spacing: 0.5px; color: #2c3e50; }

  /* Significance highlighting */
  .sig1 { background: #fef9e7 !important; }
  .sig2 { background: #fdebd0 !important; }
  .sig3 { background: #fadbd8 !important; }

  /* Config box */
  .config-box { background: #eaf2f8; padding: 10px 15px; border-radius: 6px;
                font-size: 12px; margin: 8px 0 15px 0; color: #2c3e50; }
  .config-box code { background: #d4e6f1; padding: 1px 4px; border-radius: 3px; }

  /* Overview table */
  .overview-table th { font-size: 11px; white-space: nowrap; }
  .overview-table td { font-size: 12px; white-space: nowrap; }
  .overview-table .dim-header td { background: #2c3e50; color: white;
                                    font-weight: 700; text-align: left; }

  /* Positive/negative diff coloring */
  .pos { color: #27ae60; }
  .neg { color: #c0392b; }

  .back-to-top { font-size: 12px; color: #7f8c8d; }
  .back-to-top a { color: #2980b9; text-decoration: none; }
</style>
"""


def make_anchor(dim, strategy, N):
    return f"{dim}__{strategy}__N{N}"


def fmt_val(v, is_count=False):
    if np.isnan(v):
        return "—"
    if is_count:
        return f"{v:.1f}"
    return f"{v:.4f}"


def fmt_diff(v):
    if np.isnan(v):
        return "—"
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{v:+.4f}</span>'


def fmt_p(p):
    if np.isnan(p):
        return "—"
    stars = sig_stars(p)
    if p < 0.0001:
        return f"<.0001 {stars}"
    return f"{p:.4f} {stars}"


def generate_overview_table(all_results):
    """Generate cross-cell overview table with key metrics."""
    key_metrics = [
        "word_count", "demir_total_rate", "fung_interpersonal_rate",
        "fung_total_rate", "like_rate", "tom_rate", "politeness_rate", "sentiment",
    ]
    short_labels = {
        "word_count": "Words",
        "demir_total_rate": "Hedges",
        "fung_interpersonal_rate": "Interpers.",
        "fung_total_rate": "Discourse",
        "like_rate": "'Like'",
        "tom_rate": "ToM",
        "politeness_rate": "Polite",
        "sentiment": "Sentiment",
    }

    lines = []
    lines.append('<table class="overview-table">')
    lines.append("<thead><tr>")
    lines.append("<th>Dimension</th><th>Strategy</th><th>N</th>")
    for m in key_metrics:
        lines.append(f"<th>{short_labels[m]}<br><small>H−A (p)</small></th>")
    lines.append("<th># Sig</th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    # Group by dimension
    dims_seen = set()
    for r in all_results:
        dim = r["dim"]
        strategy = r["strategy"]
        N = r["N"]
        anchor = make_anchor(dim, strategy, N)

        # Dimension header row
        if dim not in dims_seen:
            dims_seen.add(dim)
            lines.append(f'<tr class="dim-header"><td colspan="{3 + len(key_metrics) + 1}">'
                         f'{dim}</td></tr>')

        lines.append("<tr>")
        lines.append(f'<td><a href="#{anchor}">{dim}</a></td>')
        lines.append(f"<td>{strategy}</td>")
        lines.append(f"<td>{N}</td>")

        n_sig = 0
        for m in key_metrics:
            test = r["tests"].get(m)
            if test is None:
                lines.append("<td>—</td>")
                continue
            pw = test["pairwise"].get(("human", "ai"))
            if pw is None:
                lines.append("<td>—</td>")
                continue
            diff = pw["diff"]
            p = pw["p"]
            cls = sig_class(p)
            if cls:
                n_sig += 1
            diff_str = f"{diff:+.3f}" if not np.isnan(diff) else "—"
            p_str = sig_stars(p) if not np.isnan(p) else ""
            lines.append(f'<td class="{cls}">{diff_str} {p_str}</td>')

        lines.append(f"<td><b>{n_sig}</b>/{len(key_metrics)}</td>")
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def generate_detail_section(r):
    """Generate HTML detail section for one cell."""
    dim = r["dim"]
    strategy = r["strategy"]
    N = r["N"]
    tests = r["tests"]
    cfg = r.get("config")
    anchor = make_anchor(dim, strategy, N)

    lines = []
    lines.append(f'<h3 id="{anchor}">{dim} / {strategy} / N={N}</h3>')
    lines.append('<p class="back-to-top"><a href="#top">Back to top</a></p>')

    # Config box
    if cfg and "config" in cfg:
        c = cfg["config"]
        layers = c.get("active_layers", [])
        lines.append('<div class="config-box">')
        lines.append(f'<b>Vector type:</b> <code>{c.get("vector_type", "?")}</code> &nbsp; '
                     f'<b>Active layers ({len(layers)}):</b> <code>{layers}</code> &nbsp; '
                     f'<b>Questions:</b> {c.get("n_questions", "?")} &nbsp; '
                     f'<b>Temp:</b> {c.get("temperature", "?")} &nbsp; '
                     f'<b>Top-p:</b> {c.get("top_p", "?")}')
        lines.append('</div>')

    # --- Main metrics table ---
    lines.append('<h4>Condition Means & Human vs AI Test</h4>')
    lines.append("<table>")
    lines.append("<thead><tr>"
                 "<th>Metric</th>"
                 "<th>Baseline</th><th>Human</th><th>AI</th>"
                 "<th>H − A</th><th>t</th><th>p</th>"
                 "</tr></thead>")
    lines.append("<tbody>")

    for group_name, group_metrics in METRIC_GROUPS:
        lines.append(f'<tr class="group-header"><td colspan="7">{group_name}</td></tr>')
        for m in group_metrics:
            t = tests.get(m)
            if t is None:
                continue
            is_count = m in ("word_count", "question_count")
            bl = fmt_val(t.get("baseline_mean", np.nan), is_count)
            hu = fmt_val(t.get("human_mean", np.nan), is_count)
            ai = fmt_val(t.get("ai_mean", np.nan), is_count)

            pw = t["pairwise"].get(("human", "ai"), {})
            diff = pw.get("diff", np.nan)
            t_val = pw.get("t", np.nan)
            p_val = pw.get("p", np.nan)
            cls = sig_class(p_val)
            label = METRIC_LABELS.get(m, m)

            t_cell = f"<td>{t_val:.2f}</td>" if not np.isnan(t_val) else "<td>—</td>"
            lines.append(f'<tr class="{cls}">'
                         f"<td>{label}</td>"
                         f"<td>{bl}</td><td>{hu}</td><td>{ai}</td>"
                         f"<td>{fmt_diff(diff)}</td>"
                         f"{t_cell}"
                         f"<td>{fmt_p(p_val)}</td>"
                         f"</tr>")

    lines.append("</tbody></table>")

    # --- Full pairwise table ---
    lines.append('<h4>All Pairwise Comparisons</h4>')
    lines.append("<table>")
    lines.append("<thead><tr><th>Metric</th>")
    for c1, c2 in PAIR_ORDER:
        lines.append(f"<th>{PAIR_LABELS[(c1, c2)]}<br><small>diff / t / p</small></th>")
    lines.append("</tr></thead>")
    lines.append("<tbody>")

    for group_name, group_metrics in METRIC_GROUPS:
        lines.append(f'<tr class="group-header"><td colspan="{1 + len(PAIR_ORDER)}">'
                     f'{group_name}</td></tr>')
        for m in group_metrics:
            t = tests.get(m)
            if t is None:
                continue
            label = METRIC_LABELS.get(m, m)
            lines.append(f"<tr><td>{label}</td>")
            for pair_key in PAIR_ORDER:
                pw = t["pairwise"].get(pair_key, {})
                diff = pw.get("diff", np.nan)
                t_val = pw.get("t", np.nan)
                p_val = pw.get("p", np.nan)
                cls = sig_class(p_val)
                if np.isnan(p_val):
                    lines.append("<td>—</td>")
                else:
                    lines.append(
                        f'<td class="{cls}">'
                        f'{diff:+.4f} / {t_val:.2f} / {fmt_p(p_val)}'
                        f'</td>'
                    )
            lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def generate_html(all_results, version, v1_root):
    """Assemble full HTML report."""
    lines = []
    lines.append("<!DOCTYPE html><html><head>")
    lines.append('<meta charset="utf-8">')
    lines.append(f"<title>Exp 3: Concept Steering V1 Summary — {version}</title>")
    lines.append(CSS)
    lines.append("</head><body>")
    lines.append(f'<h1 id="top">Exp 3: Concept Steering V1 — Behavioral Summary</h1>')
    lines.append(f"<p>Version: <b>{version}</b> &nbsp; | &nbsp; "
                 f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp; | &nbsp; "
                 f"Cells: {len(all_results)}</p>")

    # --- Table of Contents ---
    lines.append('<div class="toc">')
    lines.append("<h2>Table of Contents</h2>")
    lines.append("<ul>")
    lines.append('<li><a href="#overview">Cross-Cell Overview</a></li>')

    # Group by dimension
    dims = {}
    for r in all_results:
        dims.setdefault(r["dim"], []).append(r)

    for dim in sorted(dims.keys()):
        lines.append(f'<li><b>{dim}</b><ul>')
        for r in sorted(dims[dim], key=lambda x: (x["strategy"], x["N"])):
            anchor = make_anchor(r["dim"], r["strategy"], r["N"])
            lines.append(f'<li><a href="#{anchor}">{r["strategy"]} / N={r["N"]}</a></li>')
        lines.append("</ul></li>")

    lines.append("</ul></div>")

    # --- Overview Table ---
    lines.append('<h2 id="overview">Cross-Cell Overview: Human vs AI Steering Effect</h2>')
    lines.append("<p>Each cell shows the Human − AI difference and significance. "
                 "Key: <span class='sig1'>* p&lt;.05</span> &nbsp; "
                 "<span class='sig2'>** p&lt;.01</span> &nbsp; "
                 "<span class='sig3'>*** p&lt;.001</span></p>")
    lines.append(generate_overview_table(all_results))

    # --- Detail Sections ---
    lines.append("<h2>Detailed Results by Cell</h2>")
    for dim in sorted(dims.keys()):
        lines.append(f"<h2>{dim}</h2>")
        for r in sorted(dims[dim], key=lambda x: (x["strategy"], x["N"])):
            lines.append(generate_detail_section(r))

    lines.append("</body></html>")
    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    if args.variant:
        set_variant(args.variant)
    set_version(args.version)

    v1_root = str(config.RESULTS.concept_steering / "v1")
    if not os.path.isdir(v1_root):
        print(f"[ERROR] V1 root not found: {v1_root}")
        sys.exit(1)

    cells = discover_cells(v1_root)
    if not cells:
        print(f"[ERROR] No result CSVs found in {v1_root}")
        sys.exit(1)

    print(f"Found {len(cells)} cells in {v1_root}")

    all_results = []
    for cell in cells:
        dim = cell["dim"]
        strategy = cell["strategy"]
        N = cell["N"]
        csv_path = cell["csv_path"]

        print(f"  Processing {dim} / {strategy} / N={N} ...")

        # Load and compute metrics
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={"response": "transcript_sub"})
        df = compute_all_metrics(df)

        # Run tests for each metric
        active_metrics = [m for m in ALL_METRICS if m in df.columns and df[m].notna().any()]
        tests = {}
        for m in active_metrics:
            tests[m] = run_tests(df, m)

        # Load config
        cfg = load_config_json(cell["dir"], N)

        all_results.append({
            "dim": dim, "strategy": strategy, "N": N,
            "tests": tests, "config": cfg,
        })

    # Sort: dim, strategy, N
    all_results.sort(key=lambda x: (x["dim"], x["strategy"], x["N"]))

    # Generate HTML
    html = generate_html(all_results, args.version, v1_root)
    out_path = os.path.join(v1_root, "concept_steering_summary.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n[SAVED] {out_path}")
    print(f"  {len(all_results)} cells analyzed")

    # Quick sig count
    n_any_sig = 0
    for r in all_results:
        for m, t in r["tests"].items():
            pw = t["pairwise"].get(("human", "ai"), {})
            if pw.get("p", 1.0) < 0.05:
                n_any_sig += 1
                break
    print(f"  {n_any_sig}/{len(all_results)} cells have at least one significant H vs A metric")


if __name__ == "__main__":
    main()
