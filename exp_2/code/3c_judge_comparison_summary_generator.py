#!/usr/bin/env python3
"""
Generate cross-version comparison HTML report for LLM judge results.

Compares judge accuracy between balanced_gpt and nonsense_codeword,
with statistical tests (Fisher's exact, chi-squared) for each probe type
and an overall grouped bar chart.

Output:
    results/llama2_13b_chat/comparisons/v1_causality/judge/judge_comparison.html

Usage:
    python gen_judge_comparison.py

Env: behavior_env (or any env with matplotlib, numpy, scipy)
"""

import os
import sys
import json
import io
import base64
from pathlib import Path
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats as scipy_stats

# Project imports
EXP2_ROOT = Path(__file__).resolve().parent.parent  # exp_2/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.report_utils import save_report


# ============================================================
#  CONFIG
# ============================================================

VERSIONS = ["balanced_gpt", "nonsense_codeword"]

VERSION_DISPLAY = {
    "balanced_gpt": "Balanced GPT (Partner Identity)",
    "nonsense_codeword": "Nonsense Codeword (Control)",
}
VERSION_SHORT = {
    "balanced_gpt": "Partner Identity",
    "nonsense_codeword": "Control",
}

PROBE_TYPE_LABELS = {
    "operational": "Operational",
    "metacognitive": "Metacognitive",
    "metacognitive_matched": "Metacognitive (matched)",
}

VERSION_COLORS = {
    "balanced_gpt": "#4A7C59",
    "nonsense_codeword": "#707070",
}


# ============================================================
#  DATA LOADING
# ============================================================

def load_judge_data(version):
    """Load all judge_results.json for a version. Return list of dicts."""
    v1_dir = EXP2_ROOT / "results" / "llama2_13b_chat" / version / "V1_causality" / "data"
    if not v1_dir.exists():
        return []

    results = []
    for jf in sorted(v1_dir.rglob("judge_results.json")):
        parts = jf.relative_to(v1_dir).parts
        if len(parts) < 4:
            continue
        strategy = parts[0]
        probe_type = parts[1]
        strength_dir = parts[2]
        if not strength_dir.startswith("is_"):
            continue
        try:
            strength = int(strength_dir.replace("is_", ""))
        except ValueError:
            continue

        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        summary = data.get("summary_stats", {})
        if summary.get("n_judged", 0) == 0:
            continue

        results.append({
            "path": jf,
            "version": version,
            "strategy": strategy,
            "probe_type": probe_type,
            "strength": strength,
            "data": data,
            "summary": summary,
        })

    return results


def binomial_ci(n_correct, n_total, alpha=0.05):
    """Wilson score interval."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p_hat = n_correct / n_total
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def compare_proportions(n1_correct, n1_total, n2_correct, n2_total):
    """
    Compare two proportions using Fisher's exact test and chi-squared test.
    Returns dict with test results.
    """
    # Build 2x2 contingency table
    #              correct  incorrect
    # version1     n1_c     n1_t - n1_c
    # version2     n2_c     n2_t - n2_c
    table = np.array([
        [n1_correct, n1_total - n1_correct],
        [n2_correct, n2_total - n2_correct],
    ])

    # Fisher's exact test (two-sided)
    fisher_or, fisher_p = scipy_stats.fisher_exact(table, alternative="two-sided")

    # Chi-squared test
    if n1_total >= 5 and n2_total >= 5:
        chi2, chi2_p, _, _ = scipy_stats.chi2_contingency(table, correction=True)
    else:
        chi2, chi2_p = np.nan, np.nan

    # Proportion difference and SE
    p1 = n1_correct / n1_total if n1_total > 0 else 0
    p2 = n2_correct / n2_total if n2_total > 0 else 0
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1_total + p2 * (1 - p2) / n2_total) if (n1_total > 0 and n2_total > 0) else 0

    return {
        "p1": p1,
        "p2": p2,
        "diff": diff,
        "se": se,
        "fisher_or": fisher_or,
        "fisher_p": fisher_p,
        "chi2": chi2,
        "chi2_p": chi2_p,
    }


# ============================================================
#  FIGURES
# ============================================================

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def sig_stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def make_comparison_chart(all_entries):
    """
    Grouped bar chart: x-axis = probe type (matched across versions),
    grouped bars = one per version, with chance line and significance brackets.
    """
    # Organise: key = (strategy, probe_type, strength), val = {version: entry}
    conditions = {}
    for e in all_entries:
        key = (e["strategy"], e["probe_type"], e["strength"])
        if key not in conditions:
            conditions[key] = {}
        conditions[key][e["version"]] = e

    cond_keys = sorted(conditions.keys())
    n_conds = len(cond_keys)
    if n_conds == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(6, n_conds * 3), 5.5))
    width = 0.32
    x = np.arange(n_conds)

    for vi, version in enumerate(VERSIONS):
        accs, ci_lows, ci_highs = [], [], []
        for ck in cond_keys:
            entry = conditions[ck].get(version)
            if entry:
                s = entry["summary"]
                p_hat, lo, hi = binomial_ci(s["n_correct"], s["n_judged"])
                accs.append(p_hat)
                ci_lows.append(p_hat - lo)
                ci_highs.append(hi - p_hat)
            else:
                accs.append(0)
                ci_lows.append(0)
                ci_highs.append(0)

        offset = (vi - 0.5) * width
        color = VERSION_COLORS.get(version, "#6BAED6")
        bars = ax.bar(x + offset, accs, width * 0.9,
                      label=VERSION_SHORT.get(version, version),
                      color=color, edgecolor="white",
                      yerr=[ci_lows, ci_highs], capsize=3,
                      error_kw={"linewidth": 1})

        # Add accuracy labels on bars
        for j, (bar, acc) in enumerate(zip(bars, accs)):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ci_highs[j] + 0.02,
                        f"{acc:.0%}", ha="center", va="bottom",
                        fontsize=9, color="#333", fontweight="bold")

    # Significance brackets between version pairs
    for j, ck in enumerate(cond_keys):
        cond = conditions[ck]
        if len(cond) == 2:
            e1 = cond.get(VERSIONS[0])
            e2 = cond.get(VERSIONS[1])
            if e1 and e2:
                s1, s2 = e1["summary"], e2["summary"]
                result = compare_proportions(
                    s1["n_correct"], s1["n_judged"],
                    s2["n_correct"], s2["n_judged"])
                stars = sig_stars(result["fisher_p"])
                if stars:
                    # Draw bracket
                    max_y = max(
                        s1["n_correct"] / s1["n_judged"],
                        s2["n_correct"] / s2["n_judged"]
                    )
                    bracket_y = max_y + 0.14
                    x1 = j - 0.5 * width
                    x2 = j + 0.5 * width
                    ax.plot([x1, x1, x2, x2],
                            [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
                            color="#333", linewidth=1)
                    ax.text((x1 + x2) / 2, bracket_y + 0.01, stars,
                            ha="center", va="bottom", fontsize=10, fontweight="bold",
                            color="#333")

    ax.axhline(0.5, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")

    x_labels = []
    for ck in cond_keys:
        probe_label = PROBE_TYPE_LABELS.get(ck[1], ck[1])
        x_labels.append(f"{probe_label}\n(str={ck[2]})")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Judge Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("LLM Judge Accuracy — Cross-Version Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig


def make_operational_chart(all_entries):
    """Simple bar chart: one bar per version, operational probes only."""
    op_entries = {e["version"]: e for e in all_entries if e["probe_type"] == "operational"}
    if not op_entries:
        return None

    fig, ax = plt.subplots(figsize=(4.5, 5))
    x = np.arange(len(VERSIONS))
    accs, ci_lows, ci_highs, bar_labels = [], [], [], []

    for v in VERSIONS:
        entry = op_entries.get(v)
        if entry:
            s = entry["summary"]
            p_hat, lo, hi = binomial_ci(s["n_correct"], s["n_judged"])
            accs.append(p_hat)
            ci_lows.append(p_hat - lo)
            ci_highs.append(hi - p_hat)
            bar_labels.append(f"{s['n_correct']}/{s['n_judged']}")
        else:
            accs.append(0); ci_lows.append(0); ci_highs.append(0)
            bar_labels.append("")

    colors = [VERSION_COLORS.get(v, "#6BAED6") for v in VERSIONS]
    bars = ax.bar(x, accs, color=colors, edgecolor="white", linewidth=0.5,
                  yerr=[ci_lows, ci_highs], capsize=5, error_kw={"linewidth": 1.2})

    for i, (bar, lbl) in enumerate(zip(bars, bar_labels)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci_highs[i] + 0.02,
                lbl, ha="center", va="bottom", fontsize=10, color="#333")

    ax.axhline(0.5, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")

    # Significance bracket
    if len(op_entries) == 2:
        s1 = op_entries[VERSIONS[0]]["summary"]
        s2 = op_entries[VERSIONS[1]]["summary"]
        result = compare_proportions(s1["n_correct"], s1["n_judged"],
                                     s2["n_correct"], s2["n_judged"])
        stars = sig_stars(result["fisher_p"])
        if stars:
            bracket_y = max(accs) + max(ci_highs) + 0.08
            ax.plot([0, 0, 1, 1],
                    [bracket_y - 0.02, bracket_y, bracket_y, bracket_y - 0.02],
                    color="#333", linewidth=1)
            ax.text(0.5, bracket_y + 0.01, stars,
                    ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([VERSION_SHORT.get(v, v) for v in VERSIONS], fontsize=11)
    ax.set_ylabel("Judge Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title("Operational Probe — Judge Accuracy", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig


def make_per_probe_detail_charts(all_entries):
    """One chart per probe type showing breakdown by target type and order for each version."""
    # Group by (strategy, probe_type, strength)
    conditions = {}
    for e in all_entries:
        key = (e["strategy"], e["probe_type"], e["strength"])
        if key not in conditions:
            conditions[key] = {}
        conditions[key][e["version"]] = e

    figs = {}
    for ck, cond in sorted(conditions.items()):
        probe_label = PROBE_TYPE_LABELS.get(ck[1], ck[1])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

        for vi, version in enumerate(VERSIONS):
            ax = axes[vi]
            entry = cond.get(version)
            if not entry:
                ax.set_title(f"{VERSION_SHORT.get(version, version)}\n(no data)")
                continue

            details = entry["data"].get("judge_details", [])
            # Compute breakdown
            cats = {
                "Overall": {"c": 0, "t": 0},
                "Target:\nHuman": {"c": 0, "t": 0},
                "Target:\nAI": {"c": 0, "t": 0},
                "Order:\nHuman 1st": {"c": 0, "t": 0},
                "Order:\nAI 1st": {"c": 0, "t": 0},
            }
            key_map = {
                "Overall": lambda d: True,
                "Target:\nHuman": lambda d: d.get("target_type") == "human",
                "Target:\nAI": lambda d: d.get("target_type") == "ai",
                "Order:\nHuman 1st": lambda d: d.get("response_order") == "human_first",
                "Order:\nAI 1st": lambda d: d.get("response_order") == "ai_first",
            }
            for d in details:
                if d.get("is_correct") is None:
                    continue
                for cat_name, pred in key_map.items():
                    if pred(d):
                        cats[cat_name]["t"] += 1
                        if d["is_correct"]:
                            cats[cat_name]["c"] += 1

            cat_names = list(cats.keys())
            accs = []
            ci_lows = []
            ci_highs = []
            bar_labels = []
            for cn in cat_names:
                c, t = cats[cn]["c"], cats[cn]["t"]
                p_hat, lo, hi = binomial_ci(c, t)
                accs.append(p_hat)
                ci_lows.append(p_hat - lo)
                ci_highs.append(hi - p_hat)
                bar_labels.append(f"{c}/{t}")

            color = VERSION_COLORS.get(version, "#6BAED6")
            xp = np.arange(len(cat_names))
            bars = ax.bar(xp, accs, color=color, edgecolor="white", linewidth=0.5,
                          yerr=[ci_lows, ci_highs], capsize=3, error_kw={"linewidth": 1})
            for i, (bar, lbl) in enumerate(zip(bars, bar_labels)):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ci_highs[i] + 0.02,
                        lbl, ha="center", va="bottom", fontsize=8, color="#333")

            ax.axhline(0.5, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xticks(xp)
            ax.set_xticklabels(cat_names, fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.set_title(VERSION_SHORT.get(version, version), fontsize=11, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_ylabel("Judge Accuracy", fontsize=11)
        fig.suptitle(f"{probe_label} — Breakdown by Version", fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        figs[ck] = fig

    return figs


# ============================================================
#  HTML
# ============================================================

def _sig_cell(p):
    if p is None or np.isnan(p):
        return '<span style="color: #999;">N/A</span>'
    if p < 0.001:
        return f'<span style="color: #c0392b; font-weight: bold;">{p:.4f} ***</span>'
    if p < 0.01:
        return f'<span style="color: #e74c3c; font-weight: bold;">{p:.4f} **</span>'
    if p < 0.05:
        return f'<span style="color: #f39c12; font-weight: bold;">{p:.4f} *</span>'
    return f'<span style="color: #666;">{p:.4f} n.s.</span>'


def generate_html(all_entries):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Detect judge model
    models = set()
    for e in all_entries:
        m = e["summary"].get("judge_model")
        if m:
            models.add(m)
    judge_model_str = ", ".join(sorted(models)) if models else "unknown"

    # Organise by condition
    conditions = {}
    for e in all_entries:
        key = (e["strategy"], e["probe_type"], e["strength"])
        if key not in conditions:
            conditions[key] = {}
        conditions[key][e["version"]] = e
    cond_keys = sorted(conditions.keys())

    html = []
    html.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Judge Comparison — Partner Identity vs Control</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               max-width: 1200px; margin: 2rem auto; padding: 0 2rem; line-height: 1.5;
               background: #fafafa; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
        h2 {{ color: #2c3e50; margin-top: 3rem; }}
        h3 {{ color: #555; margin-top: 2rem; }}
        .summary-box {{ background: white; padding: 1.5rem; border-left: 4px solid #3498db;
                        margin: 1.5rem 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .note {{ color: #666; font-style: italic; margin: 0.5rem 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.9em; }}
        th, td {{ padding: 0.6rem 0.8rem; text-align: left; border: 1px solid #e0e0e0; }}
        th {{ background: #34495e; color: white; font-weight: 600; position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .metric {{ font-family: 'Courier New', monospace; white-space: nowrap; }}
        .toc {{ background: white; padding: 1.5rem; margin: 1.5rem 0; border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .toc ul {{ padding-left: 1.5rem; }}
        .toc li {{ margin: 0.3rem 0; }}
        .toc a {{ color: #3498db; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .timestamp {{ color: #6c757d; font-size: 0.9em; }}
        img {{ border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); max-width: 100%; }}
        .section {{ background: white; padding: 1.5rem; margin: 1.5rem 0; border-radius: 4px;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .key-metric {{ display: inline-block; padding: 0.3rem 0.8rem; margin: 0.2rem;
                       border-radius: 4px; font-weight: 600; font-size: 0.95em; }}
        .sig {{ background: #d4edda; color: #155724; }}
        .nonsig {{ background: #f8f9fa; color: #666; }}
        .finding {{ background: #fff3cd; padding: 1rem 1.5rem; border-left: 4px solid #ffc107;
                    margin: 1rem 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>LLM Judge Comparison &mdash; Partner Identity vs Control</h1>

    <div class="summary-box">
        <p><strong>Generated:</strong> <span class="timestamp">{now}</span></p>
        <p><strong>Judge model:</strong> {judge_model_str}</p>
        <p><strong>Versions compared:</strong> {', '.join(VERSION_DISPLAY.get(v, v) for v in VERSIONS)}</p>
        <p><strong>Purpose:</strong> Compare LLM judge accuracy between the partner-identity version
        (balanced_gpt) and the control version (nonsense_codeword) to assess whether causal
        interventions produce detectably different behavior only when meaningful partner labels are used.</p>
    </div>
""")

    # TOC
    html.append("""
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#overview">Overview Comparison</a></li>
            <li><a href="#statistical-tests">Statistical Tests</a></li>
            <li><a href="#comparison-chart">Comparison Chart</a></li>
            <li><a href="#operational-chart">Operational Probe — Judge Accuracy</a></li>
""")
    for ck in cond_keys:
        probe_label = PROBE_TYPE_LABELS.get(ck[1], ck[1])
        anchor = f"detail_{ck[1]}_s{ck[2]}"
        html.append(f'            <li><a href="#{anchor}">{probe_label} — Detailed Breakdown</a></li>')
    html.append("""
            <li><a href="#interpretation">Interpretation</a></li>
        </ul>
    </div>
""")

    # --- Overview table ---
    html.append("""
    <h2 id="overview">Overview Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Probe Type</th>
                <th>Strength</th>
""")
    for v in VERSIONS:
        short = VERSION_SHORT.get(v, v)
        html.append(f'                <th>{short}<br>Accuracy</th>')
        html.append(f'                <th>{short}<br>n correct / n</th>')
        html.append(f'                <th>{short}<br>p (vs chance)</th>')
    html.append("""
                <th>Difference</th>
                <th>Fisher p<br>(between versions)</th>
            </tr>
        </thead>
        <tbody>
""")

    comparison_results = []  # store for interpretation section
    for ck in cond_keys:
        cond = conditions[ck]
        probe_label = PROBE_TYPE_LABELS.get(ck[1], ck[1])

        cells = [f"<td><strong>{probe_label}</strong></td>", f"<td>{ck[2]}</td>"]

        summaries = {}
        for v in VERSIONS:
            entry = cond.get(v)
            if entry:
                s = entry["summary"]
                rate = s.get("success_rate", 0)
                n_c = s.get("n_correct", 0)
                n_j = s.get("n_judged", 0)
                p_val = s.get("binomial_test_pvalue")
                rate_color = "#155724" if p_val is not None and p_val < 0.05 else "#666"
                cells.append(f'<td class="metric" style="color: {rate_color}; font-weight: bold;">{rate:.1%}</td>')
                cells.append(f'<td class="metric">{n_c} / {n_j}</td>')
                cells.append(f'<td class="metric">{_sig_cell(p_val)}</td>')
                summaries[v] = s
            else:
                cells.extend(['<td>—</td>'] * 3)

        # Cross-version comparison
        if len(summaries) == 2:
            s1 = summaries[VERSIONS[0]]
            s2 = summaries[VERSIONS[1]]
            result = compare_proportions(
                s1["n_correct"], s1["n_judged"],
                s2["n_correct"], s2["n_judged"])
            diff_str = f"{result['diff']:+.1%}"
            diff_color = "#155724" if result["fisher_p"] < 0.05 else "#666"
            cells.append(f'<td class="metric" style="color: {diff_color}; font-weight: bold;">{diff_str}</td>')
            cells.append(f'<td class="metric">{_sig_cell(result["fisher_p"])}</td>')
            comparison_results.append({
                "probe_type": ck[1],
                "probe_label": probe_label,
                "strength": ck[2],
                "result": result,
                "s1": s1,
                "s2": s2,
            })
        else:
            cells.extend(['<td>—</td>'] * 2)

        html.append("            <tr>" + "".join(cells) + "</tr>")

    html.append("        </tbody>\n    </table>")

    # --- Statistical Tests Detail ---
    html.append("""
    <h2 id="statistical-tests">Statistical Tests</h2>
    <p class="note">Two-sided tests comparing accuracy between Partner Identity (balanced_gpt)
    and Control (nonsense_codeword) for each probe type.</p>
    <table>
        <thead>
            <tr>
                <th>Probe Type</th>
                <th>Partner Identity<br>Accuracy</th>
                <th>Control<br>Accuracy</th>
                <th>Difference<br>(PI &minus; Ctrl)</th>
                <th>SE</th>
                <th>Fisher's Exact<br>OR</th>
                <th>Fisher's Exact<br>p-value</th>
                <th>&chi;&sup2; (Yates)<br>p-value</th>
            </tr>
        </thead>
        <tbody>
""")

    for cr in comparison_results:
        r = cr["result"]
        html.append(f"""
            <tr>
                <td><strong>{cr['probe_label']}</strong> (str={cr['strength']})</td>
                <td class="metric">{r['p1']:.1%} ({cr['s1']['n_correct']}/{cr['s1']['n_judged']})</td>
                <td class="metric">{r['p2']:.1%} ({cr['s2']['n_correct']}/{cr['s2']['n_judged']})</td>
                <td class="metric" style="font-weight: bold;">{r['diff']:+.1%}</td>
                <td class="metric">{r['se']:.3f}</td>
                <td class="metric">{r['fisher_or']:.2f}</td>
                <td class="metric">{_sig_cell(r['fisher_p'])}</td>
                <td class="metric">{_sig_cell(r['chi2_p'])}</td>
            </tr>
""")

    html.append("        </tbody>\n    </table>")

    # --- Comparison Chart ---
    html.append('<h2 id="comparison-chart">Comparison Chart</h2>')
    summary_fig = make_comparison_chart(all_entries)
    if summary_fig:
        b64 = fig_to_base64(summary_fig)
        html.append(f'<img src="data:image/png;base64,{b64}" alt="Cross-version comparison chart">')

    # --- Operational-only chart ---
    html.append('<h2 id="operational-chart">Operational Probe — Judge Accuracy</h2>')
    op_fig = make_operational_chart(all_entries)
    if op_fig:
        b64 = fig_to_base64(op_fig)
        html.append(f'<img src="data:image/png;base64,{b64}" alt="Operational probe judge accuracy">')

    # --- Per-probe detail charts ---
    detail_figs = make_per_probe_detail_charts(all_entries)
    for ck in cond_keys:
        probe_label = PROBE_TYPE_LABELS.get(ck[1], ck[1])
        anchor = f"detail_{ck[1]}_s{ck[2]}"
        html.append(f"""
    <div class="section">
        <h2 id="{anchor}">{probe_label} — Detailed Breakdown (Strength {ck[2]})</h2>
""")
        fig = detail_figs.get(ck)
        if fig:
            b64 = fig_to_base64(fig)
            html.append(f'        <img src="data:image/png;base64,{b64}" alt="{probe_label} breakdown">')

        # Per-version stats side by side
        cond = conditions[ck]
        for v in VERSIONS:
            entry = cond.get(v)
            if not entry:
                continue
            s = entry["summary"]
            pb = s.get("position_bias", {})
            hf_c = pb.get("human_first_correct", 0)
            hf_t = pb.get("human_first_total", 0)
            af_c = pb.get("ai_first_correct", 0)
            af_t = pb.get("ai_first_total", 0)
            hf_acc = f"{hf_c/hf_t:.1%}" if hf_t > 0 else "N/A"
            af_acc = f"{af_c/af_t:.1%}" if af_t > 0 else "N/A"

            html.append(f"""
        <h3>{VERSION_SHORT.get(v, v)}</h3>
        <p>
            <span class="key-metric {'sig' if s.get('binomial_test_pvalue', 1) < 0.05 else 'nonsig'}">
                Accuracy: {s.get('success_rate', 0):.1%} {sig_stars(s.get('binomial_test_pvalue'))}
            </span>
            <span class="key-metric nonsig">n = {s.get('n_judged', 0)}</span>
            <span class="key-metric nonsig">Failed: {s.get('n_failed', 0)}</span>
        </p>
        <p style="font-size: 0.9em; color: #555;">
            Position bias: Human-first {hf_acc} ({hf_c}/{hf_t}), AI-first {af_acc} ({af_c}/{af_t})
        </p>
""")

        html.append("    </div>")

    # --- Interpretation ---
    html.append("""
    <div class="section">
        <h2 id="interpretation">Interpretation</h2>
""")

    # Auto-generate interpretation based on results
    any_sig_cross = any(cr["result"]["fisher_p"] < 0.05 for cr in comparison_results)
    any_sig_chance = False
    for cr in comparison_results:
        p1_vs_chance = cr["s1"].get("binomial_test_pvalue", 1)
        p2_vs_chance = cr["s2"].get("binomial_test_pvalue", 1)
        if p1_vs_chance < 0.05 or p2_vs_chance < 0.05:
            any_sig_chance = True

    findings = []
    for cr in comparison_results:
        r = cr["result"]
        probe = cr["probe_label"]
        p1_vs_chance = cr["s1"].get("binomial_test_pvalue", 1)
        p2_vs_chance = cr["s2"].get("binomial_test_pvalue", 1)

        # Version-specific vs chance
        if p1_vs_chance < 0.05:
            findings.append(
                f"<strong>{probe} — Partner Identity:</strong> Judge accuracy ({r['p1']:.1%}) is "
                f"significantly above chance (p = {p1_vs_chance:.4f}).")
        else:
            findings.append(
                f"<strong>{probe} — Partner Identity:</strong> Judge accuracy ({r['p1']:.1%}) does not "
                f"significantly differ from chance (p = {p1_vs_chance:.4f}).")

        if p2_vs_chance < 0.05:
            findings.append(
                f"<strong>{probe} — Control:</strong> Judge accuracy ({r['p2']:.1%}) is "
                f"significantly above chance (p = {p2_vs_chance:.4f}).")
        else:
            findings.append(
                f"<strong>{probe} — Control:</strong> Judge accuracy ({r['p2']:.1%}) does not "
                f"significantly differ from chance (p = {p2_vs_chance:.4f}).")

        # Cross-version comparison
        if r["fisher_p"] < 0.05:
            findings.append(
                f"<strong>{probe} — Cross-version:</strong> The {r['diff']:+.1%} accuracy difference "
                f"between versions is statistically significant (Fisher p = {r['fisher_p']:.4f}).")
        else:
            findings.append(
                f"<strong>{probe} — Cross-version:</strong> The {r['diff']:+.1%} accuracy difference "
                f"between versions is not significant (Fisher p = {r['fisher_p']:.4f}).")

    if findings:
        html.append("        <ul>")
        for f in findings:
            html.append(f"            <li>{f}</li>")
        html.append("        </ul>")

    # Summary finding box
    if any_sig_cross or any_sig_chance:
        html.append('        <div class="finding">')
        if any_sig_cross:
            html.append(
                "            <p><strong>Key finding:</strong> At least one probe type shows a "
                "statistically significant difference in judge accuracy between the partner-identity "
                "and control versions, suggesting the intervention has a detectable, version-specific effect.</p>")
        elif any_sig_chance:
            html.append(
                "            <p><strong>Key finding:</strong> At least one condition shows judge accuracy "
                "significantly above chance, but the cross-version difference is not significant.</p>")
        html.append("        </div>")

    html.append("    </div>")
    html.append("</body>\n</html>")
    return "\n".join(html)


# ============================================================
#  MAIN
# ============================================================

def main():
    print("[INFO] Loading judge results for all versions...")
    all_entries = []
    for version in VERSIONS:
        entries = load_judge_data(version)
        print(f"  {version}: {len(entries)} judge files")
        for e in entries:
            s = e["summary"]
            print(f"    {e['strategy']}/{e['probe_type']}/is_{e['strength']}: "
                  f"{s['success_rate']:.1%} ({s['n_correct']}/{s['n_judged']})")
        all_entries.extend(entries)

    if not all_entries:
        print("[ERROR] No judge results found for any version.")
        sys.exit(1)

    print(f"\n[INFO] Total: {len(all_entries)} judge result files across {len(VERSIONS)} versions")

    html = generate_html(all_entries)

    out_dir = EXP2_ROOT / "results" / "llama2_13b_chat" / "comparisons" / "v1_causality" / "judge"
    out_path = out_dir / "judge_comparison.html"
    save_report(html, out_path)

    # Save standalone comparison chart
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = make_comparison_chart(all_entries)
    if fig:
        fig.savefig(fig_dir / "cross_version_comparison.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fig_dir / 'cross_version_comparison.png'}")

    op_fig = make_operational_chart(all_entries)
    if op_fig:
        op_fig.savefig(fig_dir / "operational_judge_accuracy.png", dpi=150,
                       bbox_inches="tight", facecolor="white")
        plt.close(op_fig)
        print(f"  Saved: {fig_dir / 'operational_judge_accuracy.png'}")

    print(f"\n[DONE] Comparison report saved to {out_path}")


if __name__ == "__main__":
    main()
