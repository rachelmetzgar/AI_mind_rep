#!/usr/bin/env python3
"""
Per-identity behavioral breakdown for Experiment 1.

Shows how each specific partner identity (ChatGPT, Copilot, GPT-4, Casey,
Sam, Gregory, Rebecca, "a Human", "an AI") drives behavioral differences
within and across the 10 labeling versions — both aggregate and per-turn.

Output → exp_1/comparisons/identity_summary.html
"""

import csv, io, base64, pathlib, warnings, datetime, json
from collections import defaultdict
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────────
EXP1 = pathlib.Path("/mnt/cup/labs/graziano/rachel/mind_rep/exp_1")
OUT_DIR = EXP1 / "comparisons"
OUT_DIR.mkdir(exist_ok=True)

VERSIONS = [
    "names", "balanced_names", "balanced_gpt",
    "labels", "labels_turnwise",
    "you_are_balanced_gpt", "you_are_labels", "you_are_labels_turnwise",
    "nonsense_codeword", "nonsense_ignore",
]
VERSION_LABELS = {
    "names": "Names",
    "balanced_names": "Bal. Names",
    "balanced_gpt": "Bal. GPT",
    "labels": "Labels",
    "labels_turnwise": "Labels TW",
    "you_are_balanced_gpt": "YA Bal. GPT",
    "you_are_labels": "YA Labels",
    "you_are_labels_turnwise": "YA Labels TW",
    "nonsense_codeword": "Non. Code",
    "nonsense_ignore": "Non. Ignore",
}

# What each agent key maps to per version (display label)
AGENT_NAMES = {
    "names":                  {"bot_1": "ChatGPT",  "bot_2": "Copilot",  "hum_1": "Casey",   "hum_2": "Sam"},
    "balanced_names":         {"bot_1": "ChatGPT",  "bot_2": "Copilot",  "hum_1": "Gregory", "hum_2": "Rebecca"},
    "balanced_gpt":           {"bot_1": "ChatGPT",  "bot_2": "GPT-4",    "hum_1": "Gregory", "hum_2": "Rebecca"},
    "labels":                 {"bot_1": "an AI (1)", "bot_2": "an AI (2)", "hum_1": "a Human (1)", "hum_2": "a Human (2)"},
    "labels_turnwise":        {"bot_1": "AI (1)",    "bot_2": "AI (2)",    "hum_1": "Human (1)",   "hum_2": "Human (2)"},
    "you_are_balanced_gpt":   {"bot_1": "ChatGPT",  "bot_2": "GPT-4",    "hum_1": "Gregory", "hum_2": "Rebecca"},
    "you_are_labels":         {"bot_1": "an AI (1)", "bot_2": "an AI (2)", "hum_1": "a Human (1)", "hum_2": "a Human (2)"},
    "you_are_labels_turnwise":{"bot_1": "AI (1)",    "bot_2": "AI (2)",    "hum_1": "Human (1)",   "hum_2": "Human (2)"},
    "nonsense_codeword":      {"bot_1": "an AI (1)", "bot_2": "an AI (2)", "hum_1": "a Human (1)", "hum_2": "a Human (2)"},
    "nonsense_ignore":        {"bot_1": "an AI (1)", "bot_2": "an AI (2)", "hum_1": "a Human (1)", "hum_2": "a Human (2)"},
}

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

CORE_MEASURES = [
    ("fung_interpersonal_rate", "Interpersonal DMs"),
    ("fung_cognitive_rate",     "Cognitive DMs"),
    ("fung_total_rate",         "DMs (Total)"),
    ("like_rate",               "Discourse 'Like'"),
    ("politeness_rate",         "Politeness"),
]

TURNS = [1, 2, 3, 4, 5]

# Colors for the 4 agent conditions
AGENT_COLORS = {
    "bot_1": "#e74c3c",   # red
    "bot_2": "#c0392b",   # dark red
    "hum_1": "#2980b9",   # blue
    "hum_2": "#3498db",   # light blue
}


# ── Data loading ────────────────────────────────────────────────────────────
def load_version_by_agent(version):
    """Return dict: (subject, agent, turn) → {measure: value}

    turn=0 = aggregate across all turns.
    turn=1..5 = per-turn.
    Keeps agent keys (bot_1, bot_2, hum_1, hum_2) separate.
    """
    csv_path = (EXP1 / "versions" / version / "results" / "meta-llama-Llama-2-13b-chat-hf"
                / "0.8" / "combined_utterance_level_data.csv")

    conversations = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["experiment"] != "LLM":
                continue
            key = (row["subject"], row["agent"], row["topic"], row["order"])
            conversations[key].append(row)

    turn_data = defaultdict(list)  # (subj, agent, turn) → [vals]
    agg_data = defaultdict(list)   # (subj, agent, 0)    → [vals]

    for key, rows in conversations.items():
        subj = key[0]
        agent = key[1]
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
    """Get list of per-subject values for one agent × turn × measure."""
    vals = []
    for (subj, ag, t), mvals in data.items():
        if ag == agent and t == turn:
            v = mvals.get(measure, np.nan)
            if not np.isnan(v):
                vals.append(v)
    return vals


def compute_agent_stats(data, turn):
    """For each measure, compute mean per agent. Also do within-type paired tests:
    bot_1 vs bot_2, hum_1 vs hum_2.
    Returns dict with agent means and within-type comparisons.
    """
    subjects = sorted(set(k[0] for k in data.keys()))
    results = []

    for col, label in MEASURES:
        row = {"measure": col, "label": label}

        # Per-agent means
        for ag in AGENTS:
            vals = agent_means(data, ag, turn, col)
            row[f"mean_{ag}"] = np.mean(vals) if vals else np.nan
            row[f"se_{ag}"] = stats.sem(vals) if len(vals) > 1 else np.nan

        # Within-type paired tests
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

        # Cross-type: mean(hum) vs mean(ai) per subject (the standard H vs AI test)
        h_vals, ai_vals = [], []
        for subj in subjects:
            h_list, a_list = [], []
            for ag in ["hum_1", "hum_2"]:
                k = (subj, ag, turn)
                if k in data:
                    v = data[k].get(col, np.nan)
                    if not np.isnan(v):
                        h_list.append(v)
            for ag in ["bot_1", "bot_2"]:
                k = (subj, ag, turn)
                if k in data:
                    v = data[k].get(col, np.nan)
                    if not np.isnan(v):
                        a_list.append(v)
            if h_list and a_list:
                h_vals.append(np.mean(h_list))
                ai_vals.append(np.mean(a_list))

        if len(h_vals) >= 3:
            t_stat, p_val = stats.ttest_rel(h_vals, ai_vals)
            row["cross_direction"] = "H>AI" if np.mean(h_vals) > np.mean(ai_vals) else "AI>H"
        else:
            t_stat, p_val = np.nan, 1.0
            row["cross_direction"] = "—"
        row["cross_t"] = t_stat
        row["cross_p"] = p_val

        results.append(row)

    # FDR for cross-type
    p_cross = [r["cross_p"] for r in results]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rej, padj, _, _ = multipletests(p_cross, method="fdr_bh")
    for i, r in enumerate(results):
        r["cross_padj"] = padj[i]
        r["cross_sig"] = rej[i]
        if not rej[i]:
            r["cross_str"] = "—"
        else:
            d = r["cross_direction"]
            r["cross_str"] = f"{d}***" if padj[i] < 0.001 else f"{d}**" if padj[i] < 0.01 else f"{d}*"

    # FDR for within-AI
    p_ai = [r["within_ai_p"] for r in results]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rej_ai, padj_ai, _, _ = multipletests(p_ai, method="fdr_bh")
    for i, r in enumerate(results):
        r["within_ai_padj"] = padj_ai[i]
        r["within_ai_sig"] = rej_ai[i]

    # FDR for within-Human
    p_hum = [r["within_hum_p"] for r in results]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rej_hum, padj_hum, _, _ = multipletests(p_hum, method="fdr_bh")
    for i, r in enumerate(results):
        r["within_hum_padj"] = padj_hum[i]
        r["within_hum_sig"] = rej_hum[i]

    return results


# ── Figures ─────────────────────────────────────────────────────────────────
def fig_to_b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def fig_agent_means_bar(all_data, version):
    """Grouped bar chart: 4 agent means for all 21 measures (aggregate).
    Split into 3 panels to keep readable.
    """
    stats_agg = compute_agent_stats(all_data[version], 0)
    agent_names = AGENT_NAMES[version]

    # Split measures into 3 groups for readability
    groups = [
        ("Word Count & Hedging", MEASURES[:9]),    # word_count through demir_total_rate
        ("Discourse Markers & Disfluency", MEASURES[9:17]),  # fung_ through disfluency
        ("Like, ToM, Politeness, Sentiment", MEASURES[17:]),  # like through sentiment
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
        ax.set_xticklabels([lbl for _, lbl in group_measures], fontsize=9,
                           rotation=30, ha="right")
        ax.set_ylabel("Mean", fontsize=10)
        ax.set_title(group_label, fontsize=10, fontweight="bold")
        if gi == 0:
            ax.legend(fontsize=9, loc="best")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{VERSION_LABELS[version]}: Per-Identity Means (All Measures, Aggregate)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_agent_trajectories(all_data, version):
    """Line plots for all 21 measures: 4 agent lines across turns.
    Precompute per-turn stats to avoid redundant calls.
    """
    agent_names = AGENT_NAMES[version]

    # Precompute stats for all turns
    per_turn_stats = {}
    for t in TURNS:
        per_turn_stats[t] = compute_agent_stats(all_data[version], t)

    n = len(MEASURES)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (col, label) in enumerate(MEASURES):
        ax = axes[idx // ncols][idx % ncols]
        for ag in AGENTS:
            vals = []
            for t in TURNS:
                r = next(s for s in per_turn_stats[t] if s["measure"] == col)
                vals.append(r[f"mean_{ag}"])
            ls = "--" if AGENT_TYPES[ag] == "AI" else "-"
            ax.plot(TURNS, vals, f"o{ls}", color=AGENT_COLORS[ag],
                    label=f"{agent_names[ag]}", linewidth=1.8, markersize=4)

        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks(TURNS)
        ax.set_xlabel("Turn", fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide extra axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
              fontsize=9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"{VERSION_LABELS[version]}: Per-Identity Trajectories (All Measures)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig_to_b64(fig)


def fig_within_type_sig_heatmap(all_data):
    """Heatmap: which measures show within-type differences (bot_1≠bot_2 or hum_1≠hum_2)?
    Rows = measures, columns = 4 named versions × 2 (within-AI, within-Human).
    Aggregate only.
    """
    named_versions = ["names", "balanced_names", "balanced_gpt", "you_are_balanced_gpt"]
    n_rows = len(MEASURES)
    n_cols = len(named_versions) * 2  # within-AI and within-Human per version

    matrix = np.full((n_rows, n_cols), np.nan)

    for vi, v in enumerate(named_versions):
        stats_agg = compute_agent_stats(all_data[v], 0)
        for ri, r in enumerate(stats_agg):
            # within-AI
            ci_ai = vi * 2
            if r["within_ai_sig"]:
                # Direction: which agent is higher?
                if r["mean_bot_1"] > r["mean_bot_2"]:
                    matrix[ri, ci_ai] = 1  # bot_1 > bot_2
                else:
                    matrix[ri, ci_ai] = -1
            else:
                matrix[ri, ci_ai] = 0

            # within-Human
            ci_hum = vi * 2 + 1
            if r["within_hum_sig"]:
                if r["mean_hum_1"] > r["mean_hum_2"]:
                    matrix[ri, ci_hum] = 1  # hum_1 > hum_2
                else:
                    matrix[ri, ci_hum] = -1
            else:
                matrix[ri, ci_hum] = 0

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(["#f5b7b1", "#f0f0f0", "#aed6f1"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([lbl for _, lbl in MEASURES], fontsize=9)

    col_labels = []
    for v in named_versions:
        an = AGENT_NAMES[v]
        col_labels.append(f"{an['bot_1']}\nvs\n{an['bot_2']}")
        col_labels.append(f"{an['hum_1']}\nvs\n{an['hum_2']}")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=8, ha="center")

    # Version group labels — place above the x-axis labels using tick_top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Add version group labels well above the column labels
    for vi, v in enumerate(named_versions):
        x_mid = vi * 2 + 0.5
        ax.text(x_mid, -2.5, VERSION_LABELS[v], ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                transform=ax.get_xaxis_transform())
        if vi > 0:
            ax.axvline(vi * 2 - 0.5, color="black", linewidth=1.5)

    # Annotate significant cells
    for ri in range(n_rows):
        for ci in range(n_cols):
            if matrix[ri, ci] != 0:
                ax.text(ci, ri, "*", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")

    ax.set_title("Within-Type Differences (FDR-corrected, aggregate)",
                 fontsize=12, fontweight="bold", pad=55)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#aed6f1", label="Agent 1 > Agent 2 *"),
        Patch(facecolor="#f0f0f0", label="Not significant"),
        Patch(facecolor="#f5b7b1", label="Agent 2 > Agent 1 *"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.subplots_adjust(top=0.88)
    return fig_to_b64(fig)


# ── HTML generation ─────────────────────────────────────────────────────────
def sig_cell(sig_str):
    if sig_str == "—":
        return '<td class="ns">—</td>'
    elif sig_str.startswith("H>AI"):
        return f'<td class="h-gt">{sig_str}</td>'
    else:
        return f'<td class="ai-gt">{sig_str}</td>'


def fmt_mean(val):
    if np.isnan(val):
        return "—"
    return f"{val:.4f}"


def build_html(figures, all_data, all_stats_agg):
    today = datetime.date.today().isoformat()
    parts = []
    parts.append(f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Exp 1: Per-Identity Behavioral Breakdown</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 20px 30px; background: #fafafa;
    color: #2c3e50; line-height: 1.6;
}}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 1.5em; }}
h2 {{ color: #34495e; margin-top: 35px; font-size: 1.2em; }}
h3 {{ color: #555; margin-top: 25px; }}
p {{ font-size: 0.95em; }}
.meta {{ font-size: 0.85em; color: #7f8c8d; }}
.fig-container {{ text-align: center; margin: 20px 0; }}
.fig-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
.finding {{ background: #eaf2f8; border-left: 4px solid #3498db; padding: 10px 15px;
            margin: 15px 0; font-size: 0.92em; border-radius: 0 4px 4px 0; }}
.finding strong {{ color: #2c3e50; }}
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
.notes {{ font-size: 0.85em; color: #555; margin-top: 25px; }}
.notes ul {{ padding-left: 1.2em; }}
.section {{ margin-bottom: 40px; }}
</style>
</head><body>

<h1>Experiment 1: Per-Identity Behavioral Breakdown</h1>
<p class="meta">Generated: {today} &mdash;
How do specific partner identities (names, labels) drive behavioral differences?</p>

<p>The standard H vs AI analysis collapses the two AI conditions and two Human
conditions. This report breaks them apart to show whether effects are
symmetric or driven by specific identities.</p>
""")

    # ── Section 1: Agent identity mapping table ─────────────────────────────
    parts.append("""
<div class="section">
<h2>1. Agent Identity Mapping</h2>
<p>Each version uses 4 conversation partner conditions (2 AI-labeled, 2 Human-labeled).
The 4 named versions assign distinct names; the 6 label/nonsense versions use
identical labels for both conditions within a type.</p>
<table>
<tr><th>Version</th><th>bot_1 (AI)</th><th>bot_2 (AI)</th><th>hum_1 (Human)</th><th>hum_2 (Human)</th><th>Prompt template</th></tr>
""")
    prompt_templates = {
        "names": "You believe you are speaking with {name} ({type}).",
        "balanced_names": "You believe you are speaking with {name} ({type}).",
        "balanced_gpt": "You believe you are speaking with {name} ({type}).",
        "labels": "You believe you are speaking with {type}.",
        "labels_turnwise": "You believe you are speaking with {type}. [Turn prefix: Human:/AI:]",
        "you_are_balanced_gpt": "You are talking to {name} ({type}).",
        "you_are_labels": "You are talking to {type}.",
        "you_are_labels_turnwise": "You are talking to {type}. [Turn prefix: Human:/AI:]",
        "nonsense_codeword": "Your assigned session code word is {type}.",
        "nonsense_ignore": "Ignore the following phrase: {type}.",
    }
    for v in VERSIONS:
        an = AGENT_NAMES[v]
        parts.append(f'<tr><td style="text-align:left;font-weight:600">{VERSION_LABELS[v]}</td>')
        parts.append(f'<td class="ai-col">{an["bot_1"]}</td>')
        parts.append(f'<td class="ai-col">{an["bot_2"]}</td>')
        parts.append(f'<td class="hum-col">{an["hum_1"]}</td>')
        parts.append(f'<td class="hum-col">{an["hum_2"]}</td>')
        parts.append(f'<td style="text-align:left;font-size:0.85em">{prompt_templates[v]}</td></tr>')
    parts.append("</table></div>")

    # ── Section 2: Within-type significance heatmap ─────────────────────────
    parts.append("""
<div class="section">
<h2>2. Within-Type Differences: Do the Two AI (or Two Human) Conditions Differ?</h2>
<p>For the 4 named versions, paired <em>t</em>-tests comparing bot_1 vs bot_2
and hum_1 vs hum_2 (BH-FDR corrected across 21 measures). If within-type
differences are large, effects may reflect specific name associations rather
than abstract identity category.</p>
""")
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures["within_type_heatmap"]}"></div>')
    parts.append("</div>")

    # ── Section 3: Per-version agent means (bar + trajectory) ───────────────
    named_versions = ["names", "balanced_names", "balanced_gpt", "you_are_balanced_gpt"]
    parts.append("""
<div class="section">
<h2>3. Per-Identity Means and Trajectories (Named Versions)</h2>
<p>For each named version, grouped bar charts show aggregate means per agent
for all 21 measures, and trajectory plots show how each agent&rsquo;s mean
evolves across turns. Dashed lines = AI agents, solid = Human agents.</p>
""")

    for v in named_versions:
        parts.append(f"<h3>{VERSION_LABELS[v]}</h3>")
        parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures[f"bar_{v}"]}"></div>')
        parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{figures[f"traj_{v}"]}"></div>')

        # Compact table with all 21 measures
        stats_agg = all_stats_agg[v]
        an = AGENT_NAMES[v]
        parts.append("<table>")
        parts.append(f'<tr><th class="measure">Measure</th>')
        parts.append(f'<th class="ai-col">{an["bot_1"]}<br>(AI)</th>')
        parts.append(f'<th class="ai-col">{an["bot_2"]}<br>(AI)</th>')
        parts.append(f'<th class="hum-col">{an["hum_1"]}<br>(H)</th>')
        parts.append(f'<th class="hum-col">{an["hum_2"]}<br>(H)</th>')
        parts.append('<th>H vs AI</th>')
        parts.append(f'<th>{an["bot_1"]} vs {an["bot_2"]}</th>')
        parts.append(f'<th>{an["hum_1"]} vs {an["hum_2"]}</th></tr>')

        for r in stats_agg:
            parts.append(f'<tr><td class="measure">{r["label"]}</td>')
            parts.append(f'<td class="ai-col">{fmt_mean(r["mean_bot_1"])}</td>')
            parts.append(f'<td class="ai-col">{fmt_mean(r["mean_bot_2"])}</td>')
            parts.append(f'<td class="hum-col">{fmt_mean(r["mean_hum_1"])}</td>')
            parts.append(f'<td class="hum-col">{fmt_mean(r["mean_hum_2"])}</td>')
            parts.append(sig_cell(r["cross_str"]))
            # within-AI sig
            if r["within_ai_sig"]:
                parts.append(f'<td class="sig-within">p={r["within_ai_padj"]:.3f}</td>')
            else:
                parts.append('<td class="ns">—</td>')
            # within-Human sig
            if r["within_hum_sig"]:
                parts.append(f'<td class="sig-within">p={r["within_hum_padj"]:.3f}</td>')
            else:
                parts.append('<td class="ns">—</td>')
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</div>")

    # ── Section 4: Label/Nonsense versions (simpler — 2 conditions) ────────
    other_versions = [
        "labels", "labels_turnwise",
        "you_are_labels", "you_are_labels_turnwise",
        "nonsense_codeword", "nonsense_ignore",
    ]
    parts.append("""
<div class="section">
<h2>4. Label and Nonsense Versions (Per-Agent Detail)</h2>
<p>In these versions, bot_1 and bot_2 have identical labels (&ldquo;an AI&rdquo;),
as do hum_1 and hum_2 (&ldquo;a Human&rdquo;). Within-type differences here
reflect only random variation across counterbalanced conditions.</p>
""")

    for v in other_versions:
        parts.append(f"<h3>{VERSION_LABELS[v]}</h3>")
        stats_agg = all_stats_agg[v]
        an = AGENT_NAMES[v]
        parts.append("<table>")
        parts.append(f'<tr><th class="measure">Measure</th>')
        parts.append(f'<th class="ai-col">{an["bot_1"]}</th>')
        parts.append(f'<th class="ai-col">{an["bot_2"]}</th>')
        parts.append(f'<th class="hum-col">{an["hum_1"]}</th>')
        parts.append(f'<th class="hum-col">{an["hum_2"]}</th>')
        parts.append('<th>H vs AI</th></tr>')

        for r in stats_agg:
            parts.append(f'<tr><td class="measure">{r["label"]}</td>')
            parts.append(f'<td class="ai-col">{fmt_mean(r["mean_bot_1"])}</td>')
            parts.append(f'<td class="ai-col">{fmt_mean(r["mean_bot_2"])}</td>')
            parts.append(f'<td class="hum-col">{fmt_mean(r["mean_hum_1"])}</td>')
            parts.append(f'<td class="hum-col">{fmt_mean(r["mean_hum_2"])}</td>')
            parts.append(sig_cell(r["cross_str"]))
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</div>")

    # ── Section 5: Per-turn tables for named versions ───────────────────────
    parts.append("""
<div class="section">
<h2>5. Per-Turn Agent Means (Named Versions)</h2>
<p>All measure means by agent and turn, for the 4 named versions.</p>
""")
    for v in named_versions:
        an = AGENT_NAMES[v]
        parts.append(f"<h3>{VERSION_LABELS[v]}</h3>")

        # Precompute per-turn stats
        per_turn = {t: compute_agent_stats(all_data[v], t) for t in TURNS}

        for col, label in MEASURES:
            parts.append(f"<p><strong>{label}</strong></p>")
            parts.append("<table><tr><th>Turn</th>")
            for ag in AGENTS:
                cls = "ai-col" if ag.startswith("bot") else "hum-col"
                parts.append(f'<th class="{cls}">{an[ag]}</th>')
            parts.append("<th>H − AI</th></tr>")
            for t in TURNS:
                r = next(s for s in per_turn[t] if s["measure"] == col)
                parts.append(f"<tr><td><strong>T{t}</strong></td>")
                for ag in AGENTS:
                    cls = "ai-col" if ag.startswith("bot") else "hum-col"
                    parts.append(f'<td class="{cls}">{fmt_mean(r[f"mean_{ag}"])}</td>')
                h_mean = np.mean([r["mean_hum_1"], r["mean_hum_2"]])
                a_mean = np.mean([r["mean_bot_1"], r["mean_bot_2"]])
                diff = h_mean - a_mean
                diff_str = f"{diff:+.4f}"
                cls = "h-gt" if diff > 0 else "ai-gt" if diff < 0 else "ns"
                parts.append(f'<td class="{cls}">{diff_str}</td>')
                parts.append("</tr>")
            parts.append("</table>")
    parts.append("</div>")

    # ── Notes ───────────────────────────────────────────────────────────────
    parts.append("""
<div class="notes">
<h2>Method Notes</h2>
<ul>
<li>N = 50 LLM agents (49 for names). Each agent has multiple conversations per condition.</li>
<li><strong>H vs AI (cross-type):</strong> For each subject, average the two human-labeled and
    two AI-labeled conditions, then paired <em>t</em>-test. BH-FDR across 21 measures.</li>
<li><strong>Within-type:</strong> Paired <em>t</em>-test comparing agent 1 vs agent 2 within
    each type (e.g., ChatGPT vs Copilot). Separately FDR-corrected across 21 measures.</li>
<li>Aggregate = all 5 turns pooled. Per-turn values shown in Section 6.</li>
<li>* p<sub>adj</sub> &lt; .05, ** p<sub>adj</sub> &lt; .01, *** p<sub>adj</sub> &lt; .001</li>
</ul>
</div>

</body></html>""")

    return "\n".join(parts)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data by agent...")
    all_data = {}
    for version in VERSIONS:
        print(f"  {version}...")
        all_data[version] = load_version_by_agent(version)

    print("Computing aggregate stats...")
    all_stats_agg = {}
    for version in VERSIONS:
        all_stats_agg[version] = compute_agent_stats(all_data[version], 0)

    print("Generating figures...")
    figures = {}

    # Within-type significance heatmap
    print("  within-type heatmap...")
    figures["within_type_heatmap"] = fig_within_type_sig_heatmap(all_data)

    # Per-version bar charts and trajectories (named versions only)
    named_versions = ["names", "balanced_names", "balanced_gpt", "you_are_balanced_gpt"]
    for v in named_versions:
        print(f"  bar + trajectory: {v}...")
        figures[f"bar_{v}"] = fig_agent_means_bar(all_data, v)
        figures[f"traj_{v}"] = fig_agent_trajectories(all_data, v)

    print("Writing HTML...")
    html = build_html(figures, all_data, all_stats_agg)
    out_path = OUT_DIR / "identity_summary.html"
    out_path.write_text(html)
    print(f"  -> {out_path}")

    # Summary
    print("\n=== Within-type significance counts (aggregate, named versions) ===")
    for v in named_versions:
        stats_agg = all_stats_agg[v]
        n_ai = sum(1 for r in stats_agg if r["within_ai_sig"])
        n_hum = sum(1 for r in stats_agg if r["within_hum_sig"])
        an = AGENT_NAMES[v]
        print(f"  {VERSION_LABELS[v]:15s}: {an['bot_1']} vs {an['bot_2']}: {n_ai}/21 sig  |  "
              f"{an['hum_1']} vs {an['hum_2']}: {n_hum}/21 sig")

    print("\n=== Cross-type (H vs AI) significance counts (aggregate) ===")
    for v in VERSIONS:
        n = sum(1 for r in all_stats_agg[v] if r["cross_sig"])
        print(f"  {VERSION_LABELS[v]:15s}: {n}/21 sig")
