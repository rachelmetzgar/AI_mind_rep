#!/usr/bin/env python3
"""
Behavioral (linguistic) measures broken down by conversation turn (1-5)
across all 6 labeling approaches.

Companion to behavioral_measures_by_condition.{html,md} which reports
the full-conversation aggregate (turns 1-5 combined). This script
produces the per-turn breakdown.

For each version × turn, runs a paired t-test (Human avg vs AI avg)
with BH-FDR correction across measures, and produces:
  - An HTML report with heatmap tables and line plots
  - A companion .md summary

Output → exp_1/comparisons/behavioral_by_turn.html
         exp_1/comparisons/behavioral_by_turn.md
"""

import csv, io, base64, pathlib, warnings
from collections import defaultdict
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
EXP1 = pathlib.Path("/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_1")
OUT_DIR = EXP1 / "comparisons"
OUT_DIR.mkdir(exist_ok=True)

VERSIONS = ["names", "balanced_names", "balanced_gpt", "labels", "nonsense_codeword", "nonsense_ignore"]
VERSION_LABELS = {
    "names": "Names",
    "balanced_names": "Bal. Names",
    "balanced_gpt": "Bal. GPT",
    "labels": "Labels",
    "nonsense_codeword": "Non. Code",
    "nonsense_ignore": "Non. Ignore",
}

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
    "names": "#1f77b4",
    "balanced_names": "#ff7f0e",
    "balanced_gpt": "#2ca02c",
    "labels": "#d62728",
    "nonsense_codeword": "#9467bd",
    "nonsense_ignore": "#8c564b",
}


# ── Data loading ────────────────────────────────────────────────────────────
def load_version(version):
    """Return dict: (subject, condition_type, turn) → {measure: value}

    For 4-condition versions (names, balanced_names, balanced_gpt):
      condition_type is 'hum' or 'bot' (aggregating the two sub-conditions).
    For 2-condition versions: same.

    Turn is 1-5 (assigned by row order within each conversation).
    """
    csv_path = (EXP1 / version / "results" / "meta-llama-Llama-2-13b-chat-hf"
                / "0.8" / "combined_utterance_level_data.csv")

    # Group rows by (subject, agent, topic, order) to assign turn numbers
    conversations = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["experiment"] != "LLM":
                continue
            key = (row["subject"], row["agent"], row["topic"], row["order"])
            conversations[key].append(row)

    data = {}
    for key, rows in conversations.items():
        subj = key[0]
        cond = rows[0]["condition"]  # 'hum' or 'bot'
        for turn_idx, row in enumerate(rows):
            turn = turn_idx + 1
            vals = {}
            for col, _ in MEASURES:
                v = row.get(col, "").strip()
                if v and v != "nan":
                    vals[col] = float(v)
                else:
                    vals[col] = np.nan
            data[(subj, cond, turn)] = data.get((subj, cond, turn), [])
            data[(subj, cond, turn)].append(vals)

    # Average across conversations for each (subject, condition, turn)
    # Each subject has multiple conversations per condition, average them
    result = {}
    for (subj, cond, turn), val_list in data.items():
        avg = {}
        for col, _ in MEASURES:
            values = [v[col] for v in val_list if not np.isnan(v[col])]
            avg[col] = np.mean(values) if values else np.nan
        result[(subj, cond, turn)] = avg

    return result


def compute_stats(data, turn):
    """Paired t-test (Human vs AI) for each measure at given turn.
    Returns list of dicts with keys: measure, label, t, p_raw, direction, mean_h, mean_ai.
    """
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
            direction = "—"

        results.append({
            "measure": col, "label": label,
            "t": t_stat, "p_raw": p_val,
            "direction": direction,
            "mean_h": np.mean(h_vals) if h_vals else np.nan,
            "mean_ai": np.mean(ai_vals) if ai_vals else np.nan,
            "n": len(h_vals),
        })

    # FDR correction across measures
    p_vals = [r["p_raw"] for r in results]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rejected, p_adj, _, _ = multipletests(p_vals, method="fdr_bh")

    for i, r in enumerate(results):
        r["p_adj"] = p_adj[i]
        r["rejected"] = rejected[i]
        if not r["rejected"]:
            r["sig_str"] = "—"
        else:
            d = r["direction"]
            if r["p_adj"] < 0.001:
                r["sig_str"] = f"{d}***"
            elif r["p_adj"] < 0.01:
                r["sig_str"] = f"{d}**"
            else:
                r["sig_str"] = f"{d}*"

    return results


# ── Figures ─────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dpi=130):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def make_effect_size_plot(all_stats):
    """Line plot: number of significant measures per turn for each version."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for version in VERSIONS:
        sig_counts = []
        for turn in TURNS:
            stats_list = all_stats[version][turn]
            n_sig = sum(1 for r in stats_list if r["rejected"])
            sig_counts.append(n_sig)
        ax.plot(TURNS, sig_counts, "o-", color=VERSION_COLORS[version],
                label=VERSION_LABELS[version], linewidth=2, markersize=7)

    ax.set_xlabel("Conversation Turn", fontsize=12)
    ax.set_ylabel("# Significant Measures (of 21)", fontsize=12)
    ax.set_title("Significant H vs AI Behavioral Differences by Turn", fontsize=13)
    ax.set_xticks(TURNS)
    ax.set_ylim(-0.5, N_MEASURES + 0.5)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    return fig_to_b64(fig)


def make_measure_trajectories(all_stats, measures_subset, title_suffix=""):
    """For selected measures, plot effect size (Cohen's d proxy) across turns."""
    n = len(measures_subset)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (col, label) in enumerate(measures_subset):
        ax = axes[idx // ncols][idx % ncols]
        for version in VERSIONS:
            diffs = []
            for turn in TURNS:
                stats_list = all_stats[version][turn]
                r = next(s for s in stats_list if s["measure"] == col)
                if not np.isnan(r["mean_h"]) and not np.isnan(r["mean_ai"]):
                    diffs.append(r["mean_h"] - r["mean_ai"])
                else:
                    diffs.append(0)
            ax.plot(TURNS, diffs, "o-", color=VERSION_COLORS[version],
                    label=VERSION_LABELS[version], linewidth=1.5, markersize=4)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(TURNS)
        ax.set_xlabel("Turn")
        ax.set_ylabel("H − AI")
        ax.grid(True, alpha=0.2)

    # Hide extra axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(6, len(VERSIONS)),
              fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"H − AI Difference by Turn{title_suffix}", fontsize=13, y=1.05)
    fig.tight_layout()
    return fig_to_b64(fig)


# ── HTML generation ─────────────────────────────────────────────────────────
def sig_cell(sig_str):
    """Return colored HTML cell."""
    if sig_str == "—":
        return '<td class="ns">—</td>'
    elif sig_str.startswith("H>AI"):
        return f'<td class="h-gt-ai">{sig_str}</td>'
    else:
        return f'<td class="ai-gt-h">{sig_str}</td>'


def build_html(all_stats, figures):
    parts = []
    parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Behavioral Measures by Turn</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 30px; }
h3 { color: #7f8c8d; }
table { border-collapse: collapse; margin: 15px 0; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 5px 8px; text-align: center; }
th { background: #34495e; color: white; font-weight: 600; }
th.measure { text-align: left; min-width: 180px; }
td.measure { text-align: left; font-weight: 500; }
td.ns { color: #bdc3c7; }
td.h-gt-ai { background: #d5f5e3; color: #1e8449; font-weight: 600; }
td.ai-gt-h { background: #fdebd0; color: #e67e22; font-weight: 600; }
.fig-container { text-align: center; margin: 20px 0; }
.fig-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.note { font-size: 12px; color: #7f8c8d; margin: 5px 0; }
.sig-count { font-weight: 700; background: #eaf2f8; }
.section { margin: 30px 0; }
</style>
</head><body>
<h1>Experiment 1: Behavioral Measures by Conversation Turn</h1>
<p class="note">Generated: 2026-02-26 &mdash; Paired t-test (H avg vs AI avg),
BH-FDR corrected across 21 measures per turn.
Quality &amp; Connectedness excluded (conversation-level only).</p>
""")

    # Overview figure
    parts.append('<div class="section">')
    parts.append("<h2>Overview: Significant Effects by Turn</h2>")
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["overview"]}"></div>')
    parts.append("</div>")

    # Per-version tables
    for version in VERSIONS:
        vlabel = VERSION_LABELS[version]
        parts.append(f'<div class="section">')
        parts.append(f"<h2>{vlabel}</h2>")
        parts.append("<table>")
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

        # Sig count row
        parts.append('<tr class="sig-count"><td class="measure"><strong>Sig. count</strong></td>')
        for turn in TURNS:
            n_sig = sum(1 for r in all_stats[version][turn] if r["rejected"])
            parts.append(f"<td><strong>{n_sig}/{N_MEASURES}</strong></td>")
        parts.append("</tr></table></div>")

    # Trajectory plots for key measures
    core_measures = [m for m in MEASURES if m[0] in [
        "fung_interpersonal_rate", "fung_cognitive_rate", "fung_total_rate",
        "like_rate", "politeness_rate", "tom_rate", "question_count", "word_count"
    ]]
    parts.append('<div class="section">')
    parts.append("<h2>Effect Trajectories: Core Measures</h2>")
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["core_traj"]}"></div>')
    parts.append("</div>")

    hedging_measures = [m for m in MEASURES if m[0].startswith("demir_")]
    parts.append('<div class="section">')
    parts.append("<h2>Effect Trajectories: Hedging Measures</h2>")
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["hedging_traj"]}"></div>')
    parts.append("</div>")

    dm_disfl_measures = [m for m in MEASURES if m[0] in [
        "fung_interpersonal_rate", "fung_referential_rate", "fung_structural_rate",
        "fung_cognitive_rate", "nonfluency_rate", "liwc_filler_rate", "disfluency_rate", "sentiment"
    ]]
    parts.append('<div class="section">')
    parts.append("<h2>Effect Trajectories: DMs, Disfluency &amp; Sentiment</h2>")
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["dm_traj"]}"></div>')
    parts.append("</div>")

    # Combined heatmap table (all versions × turns)
    parts.append('<div class="section">')
    parts.append("<h2>Combined: All Versions × Turns</h2>")
    parts.append("<table>")
    # Header row
    parts.append('<tr><th class="measure">Measure</th>')
    for version in VERSIONS:
        vlabel = VERSION_LABELS[version]
        for t in TURNS:
            parts.append(f"<th>{vlabel}<br>T{t}</th>")
    parts.append("</tr>")

    for col, label in MEASURES:
        parts.append(f'<tr><td class="measure">{label}</td>')
        for version in VERSIONS:
            for turn in TURNS:
                r = next(s for s in all_stats[version][turn] if s["measure"] == col)
                parts.append(sig_cell(r["sig_str"]))
        parts.append("</tr>")
    parts.append("</table></div>")

    # Notes
    parts.append("""
<div class="section">
<h2>Notes</h2>
<ul>
<li>All tests: paired t-test (Human avg vs AI avg), N = 50 LLM agents (49 for names), BH-FDR corrected across 21 metrics per turn.</li>
<li>For 4-condition versions (names, balanced_names, balanced_gpt), "Human avg" and "AI avg" are the mean of the two human-labeled and two AI-labeled partner conditions.</li>
<li>For 2-condition versions (labels, nonsense_codeword, nonsense_ignore), the test is a direct paired comparison.</li>
<li>Quality and Connectedness are excluded (available only at conversation level, not per-turn).</li>
<li>Turn 1 = first exchange, Turn 5 = last exchange in the 5-turn conversation.</li>
<li>* p_adj &lt; .05, ** p_adj &lt; .01, *** p_adj &lt; .001</li>
</ul>
</div>
</body></html>""")

    return "\n".join(parts)


def build_md(all_stats):
    """Compact markdown summary."""
    lines = []
    lines.append("# Experiment 1: Behavioral Measures by Conversation Turn\n")
    lines.append("**Generated:** 2026-02-26\n")
    lines.append("Partner condition effect (Human avg vs AI avg), BH-FDR corrected across 21 metrics per turn.")
    lines.append("Quality & Connectedness excluded (conversation-level only).\n")

    for version in VERSIONS:
        vlabel = VERSION_LABELS[version]
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

        # Sig count
        sig_cells = []
        for turn in TURNS:
            n_sig = sum(1 for r in all_stats[version][turn] if r["rejected"])
            sig_cells.append(f"**{n_sig}/{N_MEASURES}**")
        lines.append(f"| **Sig. count** | " + " | ".join(sig_cells) + " |")

    lines.append("\n## Notes\n")
    lines.append("- All tests: paired t-test (Human avg vs AI avg), N = 50 LLM agents, BH-FDR corrected across 21 metrics per turn.")
    lines.append("- Quality and Connectedness excluded (conversation-level only).")
    lines.append("- \\* p_adj < .05, \\*\\* p_adj < .01, \\*\\*\\* p_adj < .001")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    all_stats = {}  # version → turn → [result_dicts]

    for version in VERSIONS:
        print(f"  {version}...")
        data = load_version(version)
        all_stats[version] = {}
        for turn in TURNS:
            all_stats[version][turn] = compute_stats(data, turn)

    print("Generating figures...")
    figures = {}
    figures["overview"] = make_effect_size_plot(all_stats)

    core_measures = [m for m in MEASURES if m[0] in [
        "fung_interpersonal_rate", "fung_cognitive_rate", "fung_total_rate",
        "like_rate", "politeness_rate", "tom_rate", "question_count", "word_count"
    ]]
    figures["core_traj"] = make_measure_trajectories(all_stats, core_measures, " (Core)")

    hedging_measures = [m for m in MEASURES if m[0].startswith("demir_")]
    figures["hedging_traj"] = make_measure_trajectories(all_stats, hedging_measures, " (Hedging)")

    dm_disfl = [m for m in MEASURES if m[0] in [
        "fung_interpersonal_rate", "fung_referential_rate", "fung_structural_rate",
        "fung_cognitive_rate", "nonfluency_rate", "liwc_filler_rate", "disfluency_rate", "sentiment"
    ]]
    figures["dm_traj"] = make_measure_trajectories(all_stats, dm_disfl, " (DMs & Disfluency)")

    print("Writing HTML...")
    html = build_html(all_stats, figures)
    (OUT_DIR / "behavioral_by_turn.html").write_text(html)
    print(f"  → {OUT_DIR / 'behavioral_by_turn.html'}")

    print("Writing MD...")
    md = build_md(all_stats)
    (OUT_DIR / "behavioral_by_turn.md").write_text(md)
    print(f"  → {OUT_DIR / 'behavioral_by_turn.md'}")

    # Print summary
    print("\n=== Significant effects summary ===")
    for version in VERSIONS:
        vlabel = VERSION_LABELS[version]
        counts = [sum(1 for r in all_stats[version][t] if r["rejected"]) for t in TURNS]
        print(f"  {vlabel:15s}: {' → '.join(str(c) for c in counts)} (T1→T5)")
