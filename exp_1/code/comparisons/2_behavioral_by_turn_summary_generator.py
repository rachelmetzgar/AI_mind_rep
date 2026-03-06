#!/usr/bin/env python3
"""
Behavioral (linguistic) measures broken down by conversation turn (1-5)
across all labeling approaches.

For each version x turn, runs a paired t-test (Human avg vs AI avg)
with BH-FDR correction across measures.

Usage:
    python 2_behavioral_by_turn_summary_generator.py --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

import csv, io, base64, sys, pathlib, warnings, argparse
from datetime import date
from collections import defaultdict
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from code.config import (
    VALID_VERSIONS, VERSIONS, VALID_MODELS,
    set_model, data_dir, comparisons_dir,
)

MEASURES = [
    ("word_count",              "Word Count"),
    ("question_count",          "Questions (#)"),
    ("demir_modal_rate",        "Demir: Modal Aux."),
    ("demir_verb_rate",         "Demir: Epistemic Verbs"),
    ("demir_adverb_rate",       "Demir: Epistemic Adverbs"),
    ("demir_adjective_rate",    "Demir: Epistemic Adj."),
    ("demir_quantifier_rate",   "Demir: Quantifiers"),
    ("demir_noun_rate",         "Demir: Epistemic Nouns"),
    ("demir_total_rate",        "Demir: Hedging (Total)"),
    ("fung_interpersonal_rate", "Fung: Interpersonal DMs"),
    ("fung_referential_rate",   "Fung: Referential DMs"),
    ("fung_structural_rate",    "Fung: Structural DMs"),
    ("fung_cognitive_rate",     "Fung: Cognitive DMs"),
    ("fung_total_rate",         "Fung: DMs (Total)"),
    ("nonfluency_rate",         "Nonfluency (LIWC)"),
    ("liwc_filler_rate",        "Filler (LIWC)"),
    ("disfluency_rate",         "Disfluency (Total)"),
    ("like_rate",               "Discourse 'Like'"),
    ("tom_rate",                "ToM Phrases"),
    ("politeness_rate",         "Politeness"),
    ("sentiment",               "Sentiment (VADER)"),
]
N_MEASURES = len(MEASURES)
TURNS = [1, 2, 3, 4, 5]

VERSION_COLORS = {
    "names": "#1f77b4", "balanced_names": "#ff7f0e", "balanced_gpt": "#2ca02c",
    "labels": "#d62728", "labels_turnwise": "#e377c2",
    "you_are_balanced_gpt": "#17becf", "you_are_labels": "#bcbd22",
    "you_are_labels_turnwise": "#7f7f7f",
    "nonsense_codeword": "#9467bd", "nonsense_ignore": "#8c564b",
}


def load_version(version, model):
    csv_path = data_dir(model, version) / "combined_utterance_level_data.csv"
    if not csv_path.exists():
        print(f"  [WARN] No data for {version}: {csv_path}")
        return {}

    conversations = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("experiment", "LLM") != "LLM":
                continue
            key = (row["subject"], row["agent"], row["topic"], row.get("order", "1"))
            conversations[key].append(row)

    data = {}
    for key, rows in conversations.items():
        subj = key[0]
        cond = rows[0]["condition"]
        for turn_idx, row in enumerate(rows):
            turn = turn_idx + 1
            vals = {}
            for col, _ in MEASURES:
                v = row.get(col, "").strip()
                if v and v != "nan":
                    vals[col] = float(v)
                else:
                    vals[col] = np.nan
            data.setdefault((subj, cond, turn), []).append(vals)

    result = {}
    for (subj, cond, turn), val_list in data.items():
        avg = {}
        for col, _ in MEASURES:
            values = [v[col] for v in val_list if not np.isnan(v[col])]
            avg[col] = np.mean(values) if values else np.nan
        result[(subj, cond, turn)] = avg
    return result


def compute_stats(data, turn):
    subjects = sorted(set(k[0] for k in data.keys()))
    results = []
    for col, label in MEASURES:
        h_vals, ai_vals = [], []
        for subj in subjects:
            h_key = (subj, "hum", turn)
            ai_key = (subj, "bot", turn)
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
            "measure": col, "label": label,
            "t": t_stat, "p_raw": p_val, "direction": direction,
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
            if r["p_adj"] < 0.001: r["sig_str"] = f"{d}***"
            elif r["p_adj"] < 0.01: r["sig_str"] = f"{d}**"
            else: r["sig_str"] = f"{d}*"
    return results


def fig_to_b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def make_effect_size_plot(all_stats, versions):
    fig, ax = plt.subplots(figsize=(8, 5))
    for version in versions:
        if version not in all_stats:
            continue
        sig_counts = []
        for turn in TURNS:
            n_sig = sum(1 for r in all_stats[version][turn] if r["rejected"])
            sig_counts.append(n_sig)
        ax.plot(TURNS, sig_counts, "o-",
                color=VERSION_COLORS.get(version, "#333"),
                label=VERSIONS[version]["label"], linewidth=2, markersize=7)
    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel(f"# Significant Measures (of {N_MEASURES})", fontsize=12)
    ax.set_title("Significant H vs AI Behavioral Differences by Turn", fontsize=13)
    ax.set_xticks(TURNS)
    ax.set_ylim(-0.5, N_MEASURES + 0.5)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig_to_b64(fig)


def make_measure_trajectories(all_stats, versions, measures_subset, title_suffix=""):
    n = len(measures_subset)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    for idx, (col, label) in enumerate(measures_subset):
        ax = axes[idx // ncols][idx % ncols]
        for version in versions:
            if version not in all_stats:
                continue
            diffs = []
            for turn in TURNS:
                r = next(s for s in all_stats[version][turn] if s["measure"] == col)
                if not np.isnan(r["mean_h"]) and not np.isnan(r["mean_ai"]):
                    diffs.append(r["mean_h"] - r["mean_ai"])
                else:
                    diffs.append(0)
            ax.plot(TURNS, diffs, "o-",
                    color=VERSION_COLORS.get(version, "#333"),
                    label=VERSIONS[version]["label"], linewidth=1.5, markersize=4)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(TURNS)
        ax.set_xlabel("Turn")
        ax.set_ylabel("H - AI")
        ax.grid(True, alpha=0.2)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(versions)),
              fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"H - AI Difference by Turn{title_suffix}", fontsize=13, y=1.05)
    fig.tight_layout()
    return fig_to_b64(fig)


def sig_cell(sig_str):
    if sig_str == "--":
        return '<td class="ns">--</td>'
    elif sig_str.startswith("H>AI"):
        return f'<td class="h-gt-ai">{sig_str}</td>'
    else:
        return f'<td class="ai-gt-h">{sig_str}</td>'


def build_html(all_stats, figures, versions):
    today = date.today().isoformat()
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Behavioral Measures by Turn</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
table {{ border-collapse: collapse; margin: 15px 0; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 5px 8px; text-align: center; }}
th {{ background: #34495e; color: white; font-weight: 600; }}
th.measure {{ text-align: left; min-width: 180px; }}
td.measure {{ text-align: left; font-weight: 500; }}
td.ns {{ color: #bdc3c7; }}
td.h-gt-ai {{ background: #d5f5e3; color: #1e8449; font-weight: 600; }}
td.ai-gt-h {{ background: #fdebd0; color: #e67e22; font-weight: 600; }}
.fig-container {{ text-align: center; margin: 20px 0; }}
.fig-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
.note {{ font-size: 12px; color: #7f8c8d; margin: 5px 0; }}
.sig-count {{ font-weight: 700; background: #eaf2f8; }}
.section {{ margin: 30px 0; }}
</style>
</head><body>
<h1>Experiment 1: Behavioral Measures by Conversation Turn</h1>
<p class="note">Generated: {today} -- Paired t-test (H avg vs AI avg),
BH-FDR corrected across {N_MEASURES} measures per turn.
Quality &amp; Connectedness excluded (conversation-level only).</p>
""")

    parts.append('<div class="section">')
    parts.append("<h2>Overview: Significant Effects by Turn</h2>")
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["overview"]}"></div>')
    parts.append("</div>")

    for version in versions:
        if version not in all_stats:
            continue
        vlabel = VERSIONS[version]["label"]
        parts.append(f'<div class="section"><h2>{vlabel}</h2><table>')
        parts.append('<tr><th class="measure">Measure</th>')
        for t in TURNS:
            parts.append(f"<th>Turn {t}</th>")
        parts.append("</tr>")
        for col, label in MEASURES:
            parts.append(f'<tr><td class="measure">{label}</td>')
            for turn in TURNS:
                r = next(s for s in all_stats[version][turn] if s["measure"] == col)
                parts.append(sig_cell(r["sig_str"]))
            parts.append("</tr>")
        parts.append('<tr class="sig-count"><td class="measure"><strong>Sig. count</strong></td>')
        for turn in TURNS:
            n_sig = sum(1 for r in all_stats[version][turn] if r["rejected"])
            parts.append(f"<td><strong>{n_sig}/{N_MEASURES}</strong></td>")
        parts.append("</tr></table></div>")

    for key, title in [("core_traj", "Core Measures"), ("hedging_traj", "Hedging"),
                       ("dm_traj", "DMs & Disfluency")]:
        if key in figures:
            parts.append(f'<div class="section"><h2>Effect Trajectories: {title}</h2>')
            parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures[key]}"></div></div>')

    parts.append("""
<div class="section"><h2>Notes</h2><ul>
<li>All tests: paired t-test (Human avg vs AI avg), N = 50 LLM agents, BH-FDR corrected per turn.</li>
<li>Quality and Connectedness excluded (conversation-level only).</li>
<li>* p_adj &lt; .05, ** p_adj &lt; .01, *** p_adj &lt; .001</li>
</ul></div></body></html>""")
    return "\n".join(parts)


def build_md(all_stats, versions):
    today = date.today().isoformat()
    lines = [
        "# Experiment 1: Behavioral Measures by Conversation Turn\n",
        f"**Generated:** {today}\n",
        f"Partner condition effect (Human avg vs AI avg), BH-FDR corrected across {N_MEASURES} metrics per turn.\n",
    ]
    for version in versions:
        if version not in all_stats:
            continue
        vlabel = VERSIONS[version]["label"]
        lines.append(f"\n## {vlabel}\n")
        header = "| Measure | " + " | ".join(f"T{t}" for t in TURNS) + " |"
        sep = "|---|" + "|".join(["---"] * len(TURNS)) + "|"
        lines.append(header)
        lines.append(sep)
        for col, label in MEASURES:
            cells = []
            for turn in TURNS:
                r = next(s for s in all_stats[version][turn] if s["measure"] == col)
                cells.append(r["sig_str"])
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        sig_cells = []
        for turn in TURNS:
            n_sig = sum(1 for r in all_stats[version][turn] if r["rejected"])
            sig_cells.append(f"**{n_sig}/{N_MEASURES}**")
        lines.append(f"| **Sig. count** | " + " | ".join(sig_cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-version behavioral by turn")
    parser.add_argument("--model", default="llama2_13b_chat", choices=VALID_MODELS)
    parser.add_argument("--versions", default=None)
    args = parser.parse_args()

    set_model(args.model)
    versions = [v.strip() for v in args.versions.split(",")] if args.versions else list(VALID_VERSIONS)
    out_dir = comparisons_dir(args.model)

    print("Loading data...")
    all_stats = {}
    for version in versions:
        print(f"  {version}...")
        data = load_version(version, args.model)
        if data:
            all_stats[version] = {}
            for turn in TURNS:
                all_stats[version][turn] = compute_stats(data, turn)

    print("Generating figures...")
    figures = {"overview": make_effect_size_plot(all_stats, versions)}

    core = [m for m in MEASURES if m[0] in [
        "fung_interpersonal_rate", "fung_cognitive_rate", "fung_total_rate",
        "like_rate", "politeness_rate", "tom_rate", "question_count", "word_count"]]
    figures["core_traj"] = make_measure_trajectories(all_stats, versions, core, " (Core)")

    hedging = [m for m in MEASURES if m[0].startswith("demir_")]
    figures["hedging_traj"] = make_measure_trajectories(all_stats, versions, hedging, " (Hedging)")

    dm_disfl = [m for m in MEASURES if m[0] in [
        "fung_interpersonal_rate", "fung_referential_rate", "fung_structural_rate",
        "fung_cognitive_rate", "nonfluency_rate", "liwc_filler_rate", "disfluency_rate", "sentiment"]]
    figures["dm_traj"] = make_measure_trajectories(all_stats, versions, dm_disfl, " (DMs & Disfluency)")

    print("Writing HTML...")
    html = build_html(all_stats, figures, versions)
    (out_dir / "behavioral_by_turn.html").write_text(html)
    print(f"  -> {out_dir / 'behavioral_by_turn.html'}")

    print("Writing MD...")
    md = build_md(all_stats, versions)
    (out_dir / "behavioral_by_turn.md").write_text(md)
    print(f"  -> {out_dir / 'behavioral_by_turn.md'}")

    print(f"\n=== Significant effects summary ===")
    for version in versions:
        if version in all_stats:
            vlabel = VERSIONS[version]["label"]
            counts = [sum(1 for r in all_stats[version][t] if r["rejected"]) for t in TURNS]
            print(f"  {vlabel:15s}: {' -> '.join(str(c) for c in counts)} (T1->T5)")
