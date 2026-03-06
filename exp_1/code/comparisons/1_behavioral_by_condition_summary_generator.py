#!/usr/bin/env python3
"""
Behavioral (linguistic) measures by partner condition across all labeling
approaches -- full conversation aggregate (turns 1-5 combined).

For each version, runs a paired t-test (Human avg vs AI avg) with BH-FDR
correction across measures.

Usage:
    python 1_behavioral_by_condition_summary_generator.py --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

import csv
import sys
import pathlib
import warnings
import argparse
from datetime import date
from collections import defaultdict

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from code.config import (
    VALID_VERSIONS, VERSIONS, VALID_MODELS,
    set_model, data_dir, comparisons_dir,
)

# ── Measures ─────────────────────────────────────────────────────────────────
MEASURES = [
    ("word_count",              "Word Count",                "General"),
    ("question_count",          "Questions (#)",             "General"),
    ("demir_modal_rate",        "Modal Aux.",                "Hedging (Demir 2018)"),
    ("demir_verb_rate",         "Epistemic Verbs",           "Hedging (Demir 2018)"),
    ("demir_adverb_rate",       "Epistemic Adverbs",         "Hedging (Demir 2018)"),
    ("demir_adjective_rate",    "Epistemic Adj.",            "Hedging (Demir 2018)"),
    ("demir_quantifier_rate",   "Quantifiers",               "Hedging (Demir 2018)"),
    ("demir_noun_rate",         "Epistemic Nouns",           "Hedging (Demir 2018)"),
    ("demir_total_rate",        "Hedging (Total)",           "Hedging (Demir 2018)"),
    ("fung_interpersonal_rate", "Interpersonal DMs",         "Discourse Markers (Fung & Carter 2007)"),
    ("fung_referential_rate",   "Referential DMs",           "Discourse Markers (Fung & Carter 2007)"),
    ("fung_structural_rate",    "Structural DMs",            "Discourse Markers (Fung & Carter 2007)"),
    ("fung_cognitive_rate",     "Cognitive DMs",             "Discourse Markers (Fung & Carter 2007)"),
    ("fung_total_rate",         "DMs (Total)",               "Discourse Markers (Fung & Carter 2007)"),
    ("nonfluency_rate",         "Nonfluency (LIWC)",         "Fluency"),
    ("liwc_filler_rate",        "Filler (LIWC)",             "Fluency"),
    ("disfluency_rate",         "Disfluency (Total)",        "Fluency"),
    ("like_rate",               "Discourse 'Like'",          "Pragmatic / Evaluative"),
    ("tom_rate",                "ToM Phrases",               "Pragmatic / Evaluative"),
    ("politeness_rate",         "Politeness",                "Pragmatic / Evaluative"),
    ("sentiment",               "Sentiment (VADER)",         "Pragmatic / Evaluative"),
    ("quality",                 "Conv. Quality (1-4)",       "Pragmatic / Evaluative"),
    ("connectedness",           "Connectedness (1-4)",       "Pragmatic / Evaluative"),
]
N_MEASURES = len(MEASURES)


# ── Data loading ─────────────────────────────────────────────────────────────
def load_version(version, model):
    """Return dict: (subject, condition_type) -> {measure: value}"""
    csv_path = data_dir(model, version) / "combined_trial_level_data.csv"

    if not csv_path.exists():
        print(f"  [WARN] No data for {version}: {csv_path}")
        return {}

    raw = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment", "LLM") != "LLM":
                continue
            subj = row["subject"]
            cond = row["condition"]
            vals = {}
            for col, _, _ in MEASURES:
                v = row.get(col, "").strip()
                if v and v != "nan":
                    vals[col] = float(v)
                else:
                    vals[col] = np.nan
            raw[(subj, cond)].append(vals)

    result = {}
    for (subj, cond), val_list in raw.items():
        avg = {}
        for col, _, _ in MEASURES:
            values = [v[col] for v in val_list if not np.isnan(v[col])]
            avg[col] = np.mean(values) if values else np.nan
        result[(subj, cond)] = avg

    return result


def compute_stats(data):
    subjects = sorted(set(k[0] for k in data.keys()))
    results = []

    for col, label, category in MEASURES:
        h_vals, ai_vals = [], []
        for subj in subjects:
            h_key = (subj, "hum")
            ai_key = (subj, "bot")
            if h_key in data and ai_key in data:
                hv = data[h_key].get(col, np.nan)
                av = data[ai_key].get(col, np.nan)
                if not (np.isnan(hv) or np.isnan(av)):
                    h_vals.append(hv)
                    ai_vals.append(av)

        if len(h_vals) >= 3:
            t_stat, p_val = stats.ttest_rel(h_vals, ai_vals)
            direction = "H>AI" if np.mean(h_vals) > np.mean(ai_vals) else "AI>H"
        else:
            t_stat, p_val = np.nan, 1.0
            direction = "--"

        results.append({
            "measure": col, "label": label, "category": category,
            "t": t_stat, "p_raw": p_val,
            "direction": direction,
            "mean_h": np.mean(h_vals) if h_vals else np.nan,
            "mean_ai": np.mean(ai_vals) if ai_vals else np.nan,
            "n": len(h_vals),
        })

    p_vals = [r["p_raw"] for r in results]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rejected, p_adj, _, _ = multipletests(p_vals, method="fdr_bh")

    for i, r in enumerate(results):
        r["p_adj"] = p_adj[i]
        r["rejected"] = rejected[i]
        if not r["rejected"]:
            r["sig_str"] = "--"
        else:
            d = r["direction"]
            if r["p_adj"] < 0.001:
                r["sig_str"] = f"{d}***"
            elif r["p_adj"] < 0.01:
                r["sig_str"] = f"{d}**"
            else:
                r["sig_str"] = f"{d}*"

    return results


# ── HTML generation ──────────────────────────────────────────────────────────
def sig_cell(sig_str):
    if sig_str == "--":
        return '<td class="ns">&mdash;</td>'
    elif sig_str.startswith("H>AI"):
        return f'<td class="sig-h">{sig_str}</td>'
    else:
        return f'<td class="sig-a">{sig_str}</td>'


def build_html(all_stats, versions):
    today = date.today().isoformat()
    n_versions = len(versions)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Exp 1: Linguistic Measures by Partner Condition (Full Conversation)</title>
<style>
  body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 1400px; margin: 40px auto; padding: 0 20px;
    color: #333; background: #fafafa;
  }}
  h1 {{ font-size: 1.5em; margin-bottom: 0.2em; }}
  h2 {{ font-size: 1.15em; color: #555; margin-top: 2em; }}
  .subtitle {{ color: #777; font-size: 0.9em; margin-bottom: 2em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.85em;
           background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  th, td {{ padding: 7px 10px; text-align: center; border: 1px solid #ddd; }}
  th {{ background: #4a5568; color: #fff; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; font-weight: 500; min-width: 180px; }}
  tr:nth-child(even) {{ background: #f7f7f7; }}
  tr:hover {{ background: #eef2ff; }}
  .sig-h {{ color: #1a7f37; font-weight: 700; }}
  .sig-a {{ color: #b35900; font-weight: 700; }}
  .ns {{ color: #aaa; }}
  .summary-row td {{ font-weight: 700; background: #edf2f7 !important; border-top: 2px solid #4a5568; }}
  .category-row td {{ background: #f0f4f8 !important; font-weight: 600; font-style: italic;
                      color: #4a5568; border-top: 2px solid #cbd5e0; }}
  .notes {{ font-size: 0.85em; color: #555; line-height: 1.6; margin-top: 1.5em; }}
  .notes ul {{ padding-left: 1.2em; }}
  .legend {{ display: flex; gap: 20px; margin: 1em 0; font-size: 0.85em; flex-wrap: wrap; }}
  .legend span {{ display: inline-flex; align-items: center; gap: 5px; }}
  .dot-h {{ display: inline-block; width: 10px; height: 10px; background: #1a7f37; border-radius: 50%; }}
  .dot-a {{ display: inline-block; width: 10px; height: 10px; background: #b35900; border-radius: 50%; }}
  .dot-ns {{ display: inline-block; width: 10px; height: 10px; background: #ccc; border-radius: 50%; }}
  .approach-table {{ margin-top: 1em; font-size: 0.85em; }}
  .approach-table th {{ background: #718096; }}
</style>
</head>
<body>

<h1>Experiment 1: Linguistic Measures by Partner Condition (Full Conversation)</h1>
<p class="subtitle">Across {n_versions} Labeling Approaches &mdash; Generated {today}</p>

<div class="legend">
  <span><span class="dot-h"></span> H&gt;AI (significant)</span>
  <span><span class="dot-a"></span> AI&gt;H (significant)</span>
  <span><span class="dot-ns"></span> Not significant</span>
  <span>* p<sub>adj</sub>&lt;.05 &nbsp; ** p<sub>adj</sub>&lt;.01 &nbsp; *** p<sub>adj</sub>&lt;.001</span>
</div>

<table>
  <thead>
    <tr>
      <th>Measure</th>""")

    for v in versions:
        parts.append(f"      <th>{VERSIONS[v]['label']}</th>")

    parts.append("""    </tr>
  </thead>
  <tbody>""")

    current_category = None
    for col, label, category in MEASURES:
        if category != current_category:
            current_category = category
            parts.append(f'    <tr class="category-row"><td colspan="{n_versions + 1}">{category}</td></tr>')
        parts.append("    <tr>")
        parts.append(f"      <td>{label}</td>")
        for v in versions:
            if v in all_stats:
                r = next((s for s in all_stats[v] if s["measure"] == col), None)
                if r:
                    parts.append(f"      {sig_cell(r['sig_str'])}")
                else:
                    parts.append('      <td class="ns">&mdash;</td>')
            else:
                parts.append('      <td class="ns">N/A</td>')
        parts.append("    </tr>")

    # Summary row
    parts.append('    <tr class="summary-row">')
    parts.append("      <td>Significant count</td>")
    for v in versions:
        if v in all_stats:
            n_sig = sum(1 for r in all_stats[v] if r["rejected"])
            parts.append(f"      <td>{n_sig} / {N_MEASURES}</td>")
        else:
            parts.append("      <td>N/A</td>")
    parts.append("    </tr>")

    parts.append("""  </tbody>
</table>""")

    # Labeling approaches table
    parts.append("""
<div class="notes">
  <h2>Notes</h2>
  <ul>
    <li>All tests: paired t-test (Human avg vs AI avg), N = 50 LLM agents, BH-FDR corrected across 23 metrics.</li>
    <li>For 4-condition versions, Human/AI avg = mean of two sub-conditions per type.</li>
    <li>For 2-condition versions, the test is a direct paired comparison.</li>
  </ul>

  <h2>Labeling Approaches</h2>
  <table class="approach-table">
    <thead>
      <tr><th>Version</th><th>Human Partners</th><th>AI Partners</th><th>Key Sentence</th><th>Turn Prefix</th></tr>
    </thead>
    <tbody>""")

    for v in versions:
        vcfg = VERSIONS[v]
        parts.append(f"      <tr><td>{vcfg['label']}</td>"
                     f"<td>{vcfg['human_partners']}</td>"
                     f"<td>{vcfg['ai_partners']}</td>"
                     f"<td>{vcfg['key_sentence']}</td>"
                     f"<td>{vcfg['turn_prefix_desc']}</td></tr>")

    parts.append("""    </tbody>
  </table>
</div>

</body>
</html>""")

    return "\n".join(parts)


def build_md(all_stats, versions):
    today = date.today().isoformat()
    lines = []
    lines.append("# Experiment 1: Linguistic Measures by Partner Condition\n")
    lines.append(f"**Generated:** {today}\n")
    lines.append(f"Partner condition effect (Human avg vs AI avg), BH-FDR corrected "
                 f"across {N_MEASURES} metrics. Full conversation aggregate (turns 1-5).\n")

    header = "| Measure | " + " | ".join(VERSIONS[v]["label"] for v in versions) + " |"
    sep = "|---|" + "|".join(["---"] * len(versions)) + "|"
    lines.append(header)
    lines.append(sep)

    current_category = None
    for col, label, category in MEASURES:
        if category != current_category:
            current_category = category
            lines.append(f"| **{category}** | " + " | ".join([""] * len(versions)) + " |")
        cells = []
        for v in versions:
            if v in all_stats:
                r = next((s for s in all_stats[v] if s["measure"] == col), None)
                cells.append(r["sig_str"] if r else "N/A")
            else:
                cells.append("N/A")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    sig_cells = []
    for v in versions:
        if v in all_stats:
            n_sig = sum(1 for r in all_stats[v] if r["rejected"])
            sig_cells.append(f"**{n_sig}/{N_MEASURES}**")
        else:
            sig_cells.append("N/A")
    lines.append(f"| **Sig. count** | " + " | ".join(sig_cells) + " |")

    lines.append("\n## Notes\n")
    lines.append(f"- All tests: paired t-test (Human avg vs AI avg), N = 50 LLM agents, "
                 f"BH-FDR corrected across {N_MEASURES} metrics.")
    lines.append("- \\* p_adj < .05, \\*\\* p_adj < .01, \\*\\*\\* p_adj < .001")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-version behavioral comparison")
    parser.add_argument("--model", default="llama2_13b_chat", choices=VALID_MODELS)
    parser.add_argument("--versions", default=None,
                        help="Comma-separated version list (default: all)")
    args = parser.parse_args()

    set_model(args.model)

    if args.versions:
        versions = [v.strip() for v in args.versions.split(",")]
    else:
        versions = list(VALID_VERSIONS)

    out_dir = comparisons_dir(args.model)

    print("Loading data and computing stats...")
    all_stats = {}
    for version in versions:
        print(f"  {version}...")
        data = load_version(version, args.model)
        if data:
            all_stats[version] = compute_stats(data)
        else:
            print(f"    [SKIP] No data found")

    print("Writing HTML...")
    html = build_html(all_stats, versions)
    out_html = out_dir / "behavioral_by_condition.html"
    out_html.write_text(html)
    print(f"  -> {out_html}")

    print("Writing MD...")
    md = build_md(all_stats, versions)
    out_md = out_dir / "behavioral_by_condition.md"
    out_md.write_text(md)
    print(f"  -> {out_md}")

    print(f"\n=== Significant effects summary ({N_MEASURES} measures) ===")
    for version in versions:
        if version in all_stats:
            vlabel = VERSIONS[version]["label"]
            n_sig = sum(1 for r in all_stats[version] if r["rejected"])
            print(f"  {vlabel:15s}: {n_sig}/{N_MEASURES}")
