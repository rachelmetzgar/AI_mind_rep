#!/usr/bin/env python3
"""
Per-identity behavioral breakdown for Experiment 1.

Shows how each specific partner identity drives behavioral differences
within and across labeling versions -- both aggregate and per-turn.

Usage:
    python 3_identity_summary_generator.py --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

import csv, io, base64, sys, pathlib, warnings, argparse, datetime
from collections import defaultdict
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from code.config import (
    VALID_VERSIONS, VERSIONS, VALID_MODELS, VERSION_AGENT_NAMES,
    set_model, data_dir, comparisons_dir,
)

AGENTS = ["bot_1", "bot_2", "hum_1", "hum_2"]
AGENT_TYPES = {"bot_1": "AI", "bot_2": "AI", "hum_1": "Human", "hum_2": "Human"}

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
TURNS = [1, 2, 3, 4, 5]

AGENT_COLORS = {
    "bot_1": "#e74c3c", "bot_2": "#c0392b",
    "hum_1": "#2980b9", "hum_2": "#3498db",
}


def get_agent_names(version):
    """Get display names for agents from config, with numbering for label versions."""
    cfg = VERSIONS[version]["agent_map"]
    names = {}
    for agent in AGENTS:
        info = cfg[agent]
        name = info.get("name") or info["type"]
        # For label versions where agents have identical names, add numbering
        names[agent] = name
    # Check for duplicate names within type and add numbering
    for prefix, agents in [("bot", ["bot_1", "bot_2"]), ("hum", ["hum_1", "hum_2"])]:
        if names[agents[0]] == names[agents[1]]:
            names[agents[0]] = f"{names[agents[0]]} (1)"
            names[agents[1]] = f"{names[agents[1]]} (2)"
    return names


def load_version_by_agent(version, model):
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

    turn_data = defaultdict(list)
    agg_data = defaultdict(list)
    for key, rows in conversations.items():
        subj, agent = key[0], key[1]
        for turn_idx, row in enumerate(rows):
            turn = turn_idx + 1
            vals = {}
            for col, _ in MEASURES:
                v = row.get(col, "").strip()
                vals[col] = float(v) if (v and v != "nan") else np.nan
            turn_data[(subj, agent, turn)].append(vals)
            agg_data[(subj, agent, 0)].append(vals)

    result = {}
    for store in [turn_data, agg_data]:
        for (subj, agent, turn), val_list in store.items():
            avg = {}
            for col, _ in MEASURES:
                values = [v[col] for v in val_list if not np.isnan(v[col])]
                avg[col] = np.mean(values) if values else np.nan
            result[(subj, agent, turn)] = avg
    return result


def agent_means(data, agent, turn, measure):
    vals = []
    for (subj, ag, t), mvals in data.items():
        if ag == agent and t == turn:
            v = mvals.get(measure, np.nan)
            if not np.isnan(v):
                vals.append(v)
    return vals


def compute_agent_stats(data, turn):
    subjects = sorted(set(k[0] for k in data.keys()))
    results = []

    for col, label in MEASURES:
        row = {"measure": col, "label": label}
        for ag in AGENTS:
            vals = agent_means(data, ag, turn, col)
            row[f"mean_{ag}"] = np.mean(vals) if vals else np.nan
            row[f"se_{ag}"] = stats.sem(vals) if len(vals) > 1 else np.nan

        for a1, a2, label_pair in [("bot_1", "bot_2", "within_ai"), ("hum_1", "hum_2", "within_hum")]:
            v1, v2 = [], []
            for subj in subjects:
                k1 = (subj, a1, turn)
                k2 = (subj, a2, turn)
                if k1 in data and k2 in data:
                    val1 = data[k1].get(col, np.nan)
                    val2 = data[k2].get(col, np.nan)
                    if not (np.isnan(val1) or np.isnan(val2)):
                        v1.append(val1)
                        v2.append(val2)
            if len(v1) >= 3:
                t_stat, p_val = stats.ttest_rel(v1, v2)
                row[f"{label_pair}_t"] = t_stat
                row[f"{label_pair}_p"] = p_val
                row[f"{label_pair}_n"] = len(v1)
            else:
                row[f"{label_pair}_t"] = np.nan
                row[f"{label_pair}_p"] = 1.0
                row[f"{label_pair}_n"] = len(v1)

        h_vals, ai_vals = [], []
        for subj in subjects:
            h_list, a_list = [], []
            for ag in ["hum_1", "hum_2"]:
                k = (subj, ag, turn)
                if k in data:
                    v = data[k].get(col, np.nan)
                    if not np.isnan(v): h_list.append(v)
            for ag in ["bot_1", "bot_2"]:
                k = (subj, ag, turn)
                if k in data:
                    v = data[k].get(col, np.nan)
                    if not np.isnan(v): a_list.append(v)
            if h_list and a_list:
                h_vals.append(np.mean(h_list))
                ai_vals.append(np.mean(a_list))

        if len(h_vals) >= 3:
            t_stat, p_val = stats.ttest_rel(h_vals, ai_vals)
            row["cross_direction"] = "H>AI" if np.mean(h_vals) > np.mean(ai_vals) else "AI>H"
        else:
            t_stat, p_val = np.nan, 1.0
            row["cross_direction"] = "--"
        row["cross_t"] = t_stat
        row["cross_p"] = p_val
        results.append(row)

    # FDR corrections
    for key in ["cross", "within_ai", "within_hum"]:
        p_key = f"{key}_p"
        p_vals = [r[p_key] for r in results]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rej, padj, _, _ = multipletests(p_vals, method="fdr_bh")
        for i, r in enumerate(results):
            r[f"{key}_padj"] = padj[i]
            r[f"{key}_sig"] = rej[i]

    for r in results:
        if not r["cross_sig"]:
            r["cross_str"] = "--"
        else:
            d = r["cross_direction"]
            r["cross_str"] = f"{d}***" if r["cross_padj"] < 0.001 else f"{d}**" if r["cross_padj"] < 0.01 else f"{d}*"

    return results


def fig_to_b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def fig_agent_means_bar(all_data, version):
    stats_agg = compute_agent_stats(all_data[version], 0)
    agent_names = get_agent_names(version)
    vlabel = VERSIONS[version]["label"]

    groups = [
        ("Word Count & Hedging", MEASURES[:9]),
        ("Discourse Markers & Disfluency", MEASURES[9:17]),
        ("Like, ToM, Politeness, Sentiment", MEASURES[17:]),
    ]

    fig, axes = plt.subplots(len(groups), 1, figsize=(14, 4 * len(groups)))
    width = 0.2

    for gi, (group_label, group_measures) in enumerate(groups):
        ax = axes[gi]
        n_m = len(group_measures)
        x = np.arange(n_m)
        for i, ag in enumerate(AGENTS):
            means = []
            ses = []
            for col, _ in group_measures:
                r = next(s for s in stats_agg if s["measure"] == col)
                means.append(r[f"mean_{ag}"])
                ses.append(r[f"se_{ag}"])
            offset = (i - 1.5) * width
            ax.bar(x + offset, means, width, yerr=ses,
                   color=AGENT_COLORS[ag], label=f"{agent_names[ag]} ({AGENT_TYPES[ag]})",
                   edgecolor="white", linewidth=0.5, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in group_measures], fontsize=9, rotation=30, ha="right")
        ax.set_ylabel("Mean", fontsize=10)
        ax.set_title(group_label, fontsize=10, fontweight="bold")
        if gi == 0:
            ax.legend(fontsize=9, loc="best")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{vlabel}: Per-Identity Means (All Measures, Aggregate)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_agent_trajectories(all_data, version):
    agent_names = get_agent_names(version)
    vlabel = VERSIONS[version]["label"]

    per_turn_stats = {t: compute_agent_stats(all_data[version], t) for t in TURNS}

    n = len(MEASURES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (col, label) in enumerate(MEASURES):
        ax = axes[idx // ncols][idx % ncols]
        for ag in AGENTS:
            vals = [next(s for s in per_turn_stats[t] if s["measure"] == col)[f"mean_{ag}"] for t in TURNS]
            ls = "--" if AGENT_TYPES[ag] == "AI" else "-"
            ax.plot(TURNS, vals, f"o{ls}", color=AGENT_COLORS[ag],
                    label=f"{agent_names[ag]}", linewidth=1.8, markersize=4)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(TURNS)
        ax.set_xlabel("Turn", fontsize=8)
        ax.grid(True, alpha=0.2)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{vlabel}: Per-Identity Trajectories", fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_within_type_sig_heatmap(all_data, named_versions):
    import matplotlib.colors as mcolors

    n_rows = len(MEASURES)
    n_cols = len(named_versions) * 2

    matrix = np.full((n_rows, n_cols), np.nan)

    for vi, v in enumerate(named_versions):
        stats_agg = compute_agent_stats(all_data[v], 0)
        for ri, r in enumerate(stats_agg):
            ci_ai = vi * 2
            if r["within_ai_sig"]:
                matrix[ri, ci_ai] = 1 if r["mean_bot_1"] > r["mean_bot_2"] else -1
            else:
                matrix[ri, ci_ai] = 0
            ci_hum = vi * 2 + 1
            if r["within_hum_sig"]:
                matrix[ri, ci_hum] = 1 if r["mean_hum_1"] > r["mean_hum_2"] else -1
            else:
                matrix[ri, ci_hum] = 0

    cmap = mcolors.ListedColormap(["#f5b7b1", "#f0f0f0", "#aed6f1"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([lbl for _, lbl in MEASURES], fontsize=9)

    col_labels = []
    for v in named_versions:
        an = get_agent_names(v)
        col_labels.append(f"{an['bot_1']}\nvs\n{an['bot_2']}")
        col_labels.append(f"{an['hum_1']}\nvs\n{an['hum_2']}")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha="center")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for vi, v in enumerate(named_versions):
        x_mid = vi * 2 + 0.5
        ax.text(x_mid, -2.5, VERSIONS[v]["label"], ha="center", va="bottom",
                fontsize=10, fontweight="bold", transform=ax.get_xaxis_transform())
        if vi > 0:
            ax.axvline(vi * 2 - 0.5, color="black", linewidth=1.5)

    for ri in range(n_rows):
        for ci in range(n_cols):
            if matrix[ri, ci] != 0:
                ax.text(ci, ri, "*", ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_title("Within-Type Differences (FDR-corrected, aggregate)", fontsize=12, fontweight="bold", pad=55)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#aed6f1", label="Agent 1 > Agent 2 *"),
        Patch(facecolor="#f0f0f0", label="Not significant"),
        Patch(facecolor="#f5b7b1", label="Agent 2 > Agent 1 *"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    fig.subplots_adjust(top=0.88)
    return fig_to_b64(fig)


def sig_cell(sig_str):
    if sig_str == "--":
        return '<td class="ns">--</td>'
    elif sig_str.startswith("H>AI"):
        return f'<td class="h-gt">{sig_str}</td>'
    else:
        return f'<td class="ai-gt">{sig_str}</td>'


def fmt_mean(val):
    return "--" if np.isnan(val) else f"{val:.4f}"


def build_html(figures, all_data, all_stats_agg, versions):
    today = datetime.date.today().isoformat()

    # Identify named versions (those with distinct names per agent)
    named_versions = []
    other_versions = []
    for v in versions:
        if v in all_data:
            an = get_agent_names(v)
            # Named if bot_1 != bot_2 names (without numbering)
            cfg = VERSIONS[v]["agent_map"]
            b1_name = cfg["bot_1"].get("name")
            b2_name = cfg["bot_2"].get("name")
            if b1_name and b2_name and b1_name != b2_name:
                named_versions.append(v)
            else:
                other_versions.append(v)

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 1: Per-Identity Behavioral Breakdown</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px 30px; background: #fafafa; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 1.5em; }}
h2 {{ color: #34495e; margin-top: 35px; font-size: 1.2em; }}
h3 {{ color: #555; margin-top: 25px; }}
.meta {{ font-size: 0.85em; color: #7f8c8d; }}
.fig-container {{ text-align: center; margin: 20px 0; }}
.fig-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
table {{ border-collapse: collapse; margin: 15px 0; font-size: 0.82em; }}
th, td {{ border: 1px solid #ddd; padding: 4px 7px; text-align: center; }}
th {{ background: #34495e; color: white; font-weight: 600; }}
th.measure {{ text-align: left; min-width: 160px; }}
td.measure {{ text-align: left; font-weight: 500; }}
td.ns {{ color: #bdc3c7; }}
td.h-gt {{ background: #d5f5e3; color: #1e8449; font-weight: 600; }}
td.ai-gt {{ background: #fdebd0; color: #e67e22; font-weight: 600; }}
td.sig-within {{ background: #fadbd8; font-weight: 600; }}
.ai-col {{ background: #fff5f5; }}
.hum-col {{ background: #f0f8ff; }}
.section {{ margin-bottom: 40px; }}
</style></head><body>

<h1>Experiment 1: Per-Identity Behavioral Breakdown</h1>
<p class="meta">Generated: {today}</p>
""")

    # Identity mapping table
    parts.append('<div class="section"><h2>1. Agent Identity Mapping</h2><table>')
    parts.append('<tr><th>Version</th><th>bot_1 (AI)</th><th>bot_2 (AI)</th><th>hum_1 (Human)</th><th>hum_2 (Human)</th></tr>')
    for v in versions:
        if v not in all_data:
            continue
        an = get_agent_names(v)
        parts.append(f'<tr><td style="text-align:left;font-weight:600">{VERSIONS[v]["label"]}</td>')
        parts.append(f'<td class="ai-col">{an["bot_1"]}</td><td class="ai-col">{an["bot_2"]}</td>')
        parts.append(f'<td class="hum-col">{an["hum_1"]}</td><td class="hum-col">{an["hum_2"]}</td></tr>')
    parts.append("</table></div>")

    # Within-type heatmap
    if "within_type_heatmap" in figures:
        parts.append('<div class="section"><h2>2. Within-Type Differences</h2>')
        parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["within_type_heatmap"]}"></div></div>')

    # Named versions with bar + trajectory
    if named_versions:
        parts.append('<div class="section"><h2>3. Per-Identity Means (Named Versions)</h2>')
        for v in named_versions:
            vlabel = VERSIONS[v]["label"]
            an = get_agent_names(v)
            parts.append(f"<h3>{vlabel}</h3>")
            if f"bar_{v}" in figures:
                parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures[f"bar_{v}"]}"></div>')
            if f"traj_{v}" in figures:
                parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures[f"traj_{v}"]}"></div>')

            stats_agg = all_stats_agg[v]
            parts.append("<table>")
            parts.append(f'<tr><th class="measure">Measure</th>')
            for ag in AGENTS:
                cls = "ai-col" if ag.startswith("bot") else "hum-col"
                parts.append(f'<th class="{cls}">{an[ag]}</th>')
            parts.append('<th>H vs AI</th>')
            parts.append(f'<th>{an["bot_1"]} vs {an["bot_2"]}</th>')
            parts.append(f'<th>{an["hum_1"]} vs {an["hum_2"]}</th></tr>')

            for r in stats_agg:
                parts.append(f'<tr><td class="measure">{r["label"]}</td>')
                for ag in AGENTS:
                    cls = "ai-col" if ag.startswith("bot") else "hum-col"
                    parts.append(f'<td class="{cls}">{fmt_mean(r[f"mean_{ag}"])}</td>')
                parts.append(sig_cell(r["cross_str"]))
                for key in ["within_ai", "within_hum"]:
                    if r[f"{key}_sig"]:
                        parts.append(f'<td class="sig-within">p={r[f"{key}_padj"]:.3f}</td>')
                    else:
                        parts.append('<td class="ns">--</td>')
                parts.append("</tr>")
            parts.append("</table>")
        parts.append("</div>")

    # Other versions
    if other_versions:
        parts.append('<div class="section"><h2>4. Label and Nonsense Versions</h2>')
        for v in other_versions:
            if v not in all_stats_agg:
                continue
            vlabel = VERSIONS[v]["label"]
            an = get_agent_names(v)
            parts.append(f"<h3>{vlabel}</h3><table>")
            parts.append(f'<tr><th class="measure">Measure</th>')
            for ag in AGENTS:
                cls = "ai-col" if ag.startswith("bot") else "hum-col"
                parts.append(f'<th class="{cls}">{an[ag]}</th>')
            parts.append('<th>H vs AI</th></tr>')
            for r in all_stats_agg[v]:
                parts.append(f'<tr><td class="measure">{r["label"]}</td>')
                for ag in AGENTS:
                    cls = "ai-col" if ag.startswith("bot") else "hum-col"
                    parts.append(f'<td class="{cls}">{fmt_mean(r[f"mean_{ag}"])}</td>')
                parts.append(sig_cell(r["cross_str"]))
                parts.append("</tr>")
            parts.append("</table>")
        parts.append("</div>")

    parts.append("</body></html>")
    return "\n".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-identity behavioral summary")
    parser.add_argument("--model", default="llama2_13b_chat", choices=VALID_MODELS)
    parser.add_argument("--versions", default=None)
    args = parser.parse_args()

    set_model(args.model)
    versions = [v.strip() for v in args.versions.split(",")] if args.versions else list(VALID_VERSIONS)
    out_dir = comparisons_dir(args.model)

    print("Loading data by agent...")
    all_data = {}
    for version in versions:
        print(f"  {version}...")
        data = load_version_by_agent(version, args.model)
        if data:
            all_data[version] = data

    print("Computing aggregate stats...")
    all_stats_agg = {}
    for version in all_data:
        all_stats_agg[version] = compute_agent_stats(all_data[version], 0)

    print("Generating figures...")
    figures = {}

    # Identify named versions
    named_versions = []
    for v in versions:
        if v in all_data:
            cfg = VERSIONS[v]["agent_map"]
            b1 = cfg["bot_1"].get("name")
            b2 = cfg["bot_2"].get("name")
            if b1 and b2 and b1 != b2:
                named_versions.append(v)

    if named_versions:
        print("  within-type heatmap...")
        figures["within_type_heatmap"] = fig_within_type_sig_heatmap(all_data, named_versions)

        for v in named_versions:
            print(f"  bar + trajectory: {v}...")
            figures[f"bar_{v}"] = fig_agent_means_bar(all_data, v)
            figures[f"traj_{v}"] = fig_agent_trajectories(all_data, v)

    print("Writing HTML...")
    html = build_html(figures, all_data, all_stats_agg, versions)
    out_path = out_dir / "identity_summary.html"
    out_path.write_text(html)
    print(f"  -> {out_path}")

    print("\n=== Cross-type (H vs AI) significance counts (aggregate) ===")
    for v in versions:
        if v in all_stats_agg:
            n_sig = sum(1 for r in all_stats_agg[v] if r["cross_sig"])
            print(f"  {VERSIONS[v]['label']:15s}: {n_sig}/{len(MEASURES)}")
