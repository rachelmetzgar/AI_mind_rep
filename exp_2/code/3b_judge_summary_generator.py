#!/usr/bin/env python3
"""
Generate HTML report for LLM judge results (V1 causal intervention).

Scans data/{version}/intervention_results/V1/ for judge_results.json files,
builds a summary table and per-analysis bar charts, and saves an HTML report.

Output:
    results/versions/{version}/V1_causality/judge/judge_report.html
    results/versions/{version}/V1_causality/judge/figures/*.png

Usage:
    python gen_judge_report.py --version balanced_gpt
    python gen_judge_report.py --version nonsense_codeword

Env: behavior_env (or any env with matplotlib, numpy, scipy)
"""

import os
import sys
import json
import io
import base64
import argparse
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
from config import VALID_VERSIONS
from src.report_utils import save_report


# ============================================================
#  CONFIG
# ============================================================

VERSION_DISPLAY = {
    "balanced_gpt": {
        "label": "Balanced GPT (Partner Identity)",
        "type": "partner_identity",
    },
    "nonsense_codeword": {
        "label": "Nonsense Codeword (Control)",
        "type": "control",
    },
}

PROBE_TYPE_LABELS = {
    "operational": "Operational",
    "metacognitive": "Metacognitive",
    "metacognitive_matched": "Metacognitive (matched)",
}

# Color scheme from plan
COLORS = {
    "partner_identity": {
        "metacognitive": "#8FBC8F",   # light muted green
        "metacognitive_matched": "#8FBC8F",
        "operational": "#4A7C59",           # darker muted green
    },
    "control": {
        "metacognitive": "#B0B0B0",   # light gray
        "metacognitive_matched": "#B0B0B0",
        "operational": "#707070",           # darker gray
    },
}

# Fallback colors for versions not in VERSION_DISPLAY
DEFAULT_COLORS = {
    "metacognitive": "#6BAED6",
    "metacognitive_matched": "#6BAED6",
    "operational": "#2171B5",
}


# ============================================================
#  DATA LOADING
# ============================================================

def discover_judge_files(version):
    """Scan intervention_results/V1/ for judge_results.json files. Return list of dicts."""
    v1_dir = EXP2_ROOT / "results" / "llama2_13b_chat" / version / "V1_causality" / "data"
    if not v1_dir.exists():
        print(f"[WARN] V1 directory not found: {v1_dir}")
        return []

    results = []
    for jf in sorted(v1_dir.rglob("judge_results.json")):
        # Parse path: .../V1/{strategy}/{probe_type}/is_{strength}/judge_results.json
        parts = jf.relative_to(v1_dir).parts
        if len(parts) < 4:
            continue
        strategy = parts[0]
        probe_type = parts[1]
        strength_dir = parts[2]  # e.g. "is_4"
        if not strength_dir.startswith("is_"):
            continue
        try:
            strength = int(strength_dir.replace("is_", ""))
        except ValueError:
            continue

        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Failed to read {jf}: {e}")
            continue

        summary = data.get("summary_stats", {})
        n_judged = summary.get("n_judged", 0)
        if n_judged == 0:
            print(f"[SKIP] {jf} — n_judged=0")
            continue

        results.append({
            "path": jf,
            "strategy": strategy,
            "probe_type": probe_type,
            "strength": strength,
            "data": data,
            "summary": summary,
        })

    return results


def binomial_ci(n_correct, n_total, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0, 0.0
    p_hat = n_correct / n_total
    z = scipy_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return p_hat, max(0, center - margin), min(1, center + margin)


def compute_breakdown(details):
    """Compute accuracy breakdown by target_type and response_order."""
    breakdown = {
        "overall": {"correct": 0, "total": 0},
        "target_human": {"correct": 0, "total": 0},
        "target_ai": {"correct": 0, "total": 0},
        "order_human_first": {"correct": 0, "total": 0},
        "order_ai_first": {"correct": 0, "total": 0},
    }
    for d in details:
        if d.get("is_correct") is None:
            continue
        breakdown["overall"]["total"] += 1
        if d["is_correct"]:
            breakdown["overall"]["correct"] += 1

        key_target = f"target_{d['target_type']}"
        breakdown[key_target]["total"] += 1
        if d["is_correct"]:
            breakdown[key_target]["correct"] += 1

        key_order = f"order_{d['response_order']}"
        breakdown[key_order]["total"] += 1
        if d["is_correct"]:
            breakdown[key_order]["correct"] += 1

    return breakdown


# ============================================================
#  FIGURES
# ============================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64-encoded PNG for HTML embedding."""
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


def make_breakdown_chart(entry, version_type):
    """Bar chart showing accuracy breakdown for a single judge file."""
    details = entry["data"].get("judge_details", [])
    breakdown = compute_breakdown(details)
    summary = entry["summary"]
    probe_type = entry["probe_type"]

    # Get color for this probe type
    colors_map = COLORS.get(version_type, DEFAULT_COLORS)
    bar_color = colors_map.get(probe_type, "#6BAED6")

    categories = ["Overall", "Target:\nHuman", "Target:\nAI",
                   "Order:\nHuman 1st", "Order:\nAI 1st"]
    keys = ["overall", "target_human", "target_ai",
            "order_human_first", "order_ai_first"]

    accs = []
    ci_lows = []
    ci_highs = []
    labels = []
    for cat, key in zip(categories, keys):
        bd = breakdown[key]
        p_hat, lo, hi = binomial_ci(bd["correct"], bd["total"])
        accs.append(p_hat)
        ci_lows.append(p_hat - lo)
        ci_highs.append(hi - p_hat)
        labels.append(f"{bd['correct']}/{bd['total']}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(categories))
    bars = ax.bar(x, accs, color=bar_color, edgecolor="white", linewidth=0.5,
                  yerr=[ci_lows, ci_highs], capsize=4, error_kw={"linewidth": 1.2})

    # Chance line
    ax.axhline(0.5, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7, label="Chance (50%)")

    # Add count labels on bars
    for i, (bar, lbl) in enumerate(zip(bars, labels)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci_highs[i] + 0.02,
                lbl, ha="center", va="bottom", fontsize=9, color="#333")

    # Significance annotation for overall
    p_val = summary.get("binomial_test_pvalue")
    stars = sig_stars(p_val)
    if p_val is not None:
        p_text = f"p = {p_val:.4f} {stars}"
    else:
        p_text = "p = N/A"
    ax.text(0, max(accs[0] + ci_highs[0] + 0.08, 0.65), p_text,
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Judge Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{entry['strategy'].replace('_', ' ').title()} / "
        f"{PROBE_TYPE_LABELS.get(probe_type, probe_type)} / "
        f"Strength {entry['strength']}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig


def make_summary_comparison_chart(all_entries, version_type):
    """Grouped bar chart comparing all judge results at a glance."""
    if not all_entries:
        return None

    # Group by (strategy, strength)
    groups = {}
    for entry in all_entries:
        key = (entry["strategy"], entry["strength"])
        if key not in groups:
            groups[key] = {}
        groups[key][entry["probe_type"]] = entry

    group_keys = sorted(groups.keys())
    probe_types_seen = sorted(set(e["probe_type"] for e in all_entries))
    n_groups = len(group_keys)
    n_probes = len(probe_types_seen)

    if n_groups == 0:
        return None

    fig, ax = plt.subplots(figsize=(max(6, n_groups * 2.5), 5))
    width = 0.35
    x = np.arange(n_groups)

    colors_map = COLORS.get(version_type, DEFAULT_COLORS)

    for i, pt in enumerate(probe_types_seen):
        accs = []
        ci_lows = []
        ci_highs = []
        for gk in group_keys:
            entry = groups[gk].get(pt)
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

        offset = (i - (n_probes - 1) / 2) * width
        color = colors_map.get(pt, "#6BAED6")
        bars = ax.bar(x + offset, accs, width * 0.9, label=PROBE_TYPE_LABELS.get(pt, pt),
                      color=color, edgecolor="white",
                      yerr=[ci_lows, ci_highs], capsize=3, error_kw={"linewidth": 1})

        # Significance stars
        for j, gk in enumerate(group_keys):
            entry = groups[gk].get(pt)
            if entry:
                p_val = entry["summary"].get("binomial_test_pvalue")
                stars = sig_stars(p_val)
                if stars and stars != "n.s.":
                    ax.text(x[j] + offset, accs[j] + ci_highs[j] + 0.02, stars,
                            ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(0.5, color="#cc0000", linestyle="--", linewidth=1, alpha=0.7, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\nN={n}" for s, n in group_keys], fontsize=10)
    ax.set_xlabel("Strategy / Strength", fontsize=11)
    ax.set_ylabel("Judge Accuracy", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("LLM Judge Accuracy — All Conditions", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig


# ============================================================
#  HTML GENERATION
# ============================================================

def _sig_cell(p):
    """Return colored significance indicator for HTML."""
    if p is None:
        return '<span style="color: #999;">N/A</span>'
    if p < 0.001:
        return f'<span style="color: #c0392b; font-weight: bold;">{p:.4f} ***</span>'
    if p < 0.01:
        return f'<span style="color: #e74c3c; font-weight: bold;">{p:.4f} **</span>'
    if p < 0.05:
        return f'<span style="color: #f39c12; font-weight: bold;">{p:.4f} *</span>'
    return f'<span style="color: #666;">{p:.4f} n.s.</span>'


def _detect_judge_model(entries):
    """Extract judge model name from entries, return string for display."""
    models = set()
    for e in entries:
        m = e["summary"].get("judge_model")
        if m:
            models.add(m)
    if not models:
        return "unknown"
    return ", ".join(sorted(models))


def generate_html(version, entries, version_type):
    """Generate the full HTML report."""
    version_label = VERSION_DISPLAY.get(version, {}).get("label", version)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = []
    html.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Judge Report — {version}</title>
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
    </style>
</head>
<body>
    <h1>LLM Judge Report &mdash; {version_label}</h1>

    <div class="summary-box">
        <p><strong>Generated:</strong> <span class="timestamp">{now}</span></p>
        <p><strong>Version:</strong> {version}</p>
        <p><strong>Judge files found:</strong> {len(entries)}</p>
        <p><strong>Judge model:</strong> {_detect_judge_model(entries)}</p>
        <p><strong>Purpose:</strong> Evaluate whether an LLM judge can distinguish
        human-steered vs AI-steered LLaMA-2 responses after causal intervention.</p>
    </div>
""")

    if not entries:
        html.append("""
    <div class="section">
        <h2>No Judge Results Found</h2>
        <p>No judge_results.json files with n_judged &gt; 0 were found for this version.
        Check that judging jobs have completed successfully.</p>
    </div>
</body></html>""")
        return "\n".join(html)

    # --- TOC ---
    html.append("""
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#summary">Summary Table</a></li>
            <li><a href="#comparison-chart">Comparison Chart</a></li>
""")
    for entry in entries:
        anchor = f"{entry['strategy']}_{entry['probe_type']}_s{entry['strength']}"
        title = (f"{entry['strategy'].replace('_', ' ').title()} / "
                 f"{PROBE_TYPE_LABELS.get(entry['probe_type'], entry['probe_type'])} / "
                 f"Strength {entry['strength']}")
        html.append(f'            <li><a href="#{anchor}">{title}</a></li>')
    html.append("""
        </ul>
    </div>
""")

    # --- Summary Table ---
    html.append("""
    <h2 id="summary">Summary Table</h2>
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Probe Type</th>
                <th>Strength</th>
                <th>Success Rate</th>
                <th>n_correct / n_judged</th>
                <th>p-value (binomial)</th>
                <th>Position Bias</th>
            </tr>
        </thead>
        <tbody>
""")

    for entry in entries:
        s = entry["summary"]
        rate = s.get("success_rate", 0)
        n_c = s.get("n_correct", 0)
        n_j = s.get("n_judged", 0)
        p_val = s.get("binomial_test_pvalue")
        pb = s.get("position_bias", {})

        hf_c = pb.get("human_first_correct", 0)
        hf_t = pb.get("human_first_total", 0)
        af_c = pb.get("ai_first_correct", 0)
        af_t = pb.get("ai_first_total", 0)

        hf_rate = f"{hf_c}/{hf_t} ({hf_c/hf_t:.0%})" if hf_t > 0 else "N/A"
        af_rate = f"{af_c}/{af_t} ({af_c/af_t:.0%})" if af_t > 0 else "N/A"
        bias_str = f"H1st: {hf_rate}<br>A1st: {af_rate}"

        rate_color = "#155724" if p_val is not None and p_val < 0.05 else "#666"

        html.append(f"""
            <tr>
                <td><strong>{entry['strategy']}</strong></td>
                <td>{PROBE_TYPE_LABELS.get(entry['probe_type'], entry['probe_type'])}</td>
                <td>{entry['strength']}</td>
                <td class="metric" style="color: {rate_color}; font-weight: bold;">{rate:.1%}</td>
                <td class="metric">{n_c} / {n_j}</td>
                <td class="metric">{_sig_cell(p_val)}</td>
                <td style="font-size: 0.85em;">{bias_str}</td>
            </tr>
""")

    html.append("        </tbody>\n    </table>")

    # --- Comparison Chart ---
    html.append('<h2 id="comparison-chart">Comparison Chart</h2>')
    summary_fig = make_summary_comparison_chart(entries, version_type)
    if summary_fig:
        b64 = fig_to_base64(summary_fig)
        html.append(f'<img src="data:image/png;base64,{b64}" alt="Summary comparison chart">')

    # --- Per-analysis Sections ---
    for entry in entries:
        anchor = f"{entry['strategy']}_{entry['probe_type']}_s{entry['strength']}"
        title = (f"{entry['strategy'].replace('_', ' ').title()} / "
                 f"{PROBE_TYPE_LABELS.get(entry['probe_type'], entry['probe_type'])} / "
                 f"Strength {entry['strength']}")
        s = entry["summary"]

        html.append(f"""
    <div class="section">
        <h2 id="{anchor}">{title}</h2>
""")

        # Key metrics
        rate = s.get("success_rate", 0)
        p_val = s.get("binomial_test_pvalue")
        stars = sig_stars(p_val)
        sig_class = "sig" if p_val is not None and p_val < 0.05 else "nonsig"

        html.append(f"""
        <p>
            <span class="key-metric {sig_class}">Accuracy: {rate:.1%} {stars}</span>
            <span class="key-metric nonsig">n = {s.get('n_judged', 0)} / {s.get('n_total', 0)}</span>
            <span class="key-metric nonsig">Failed: {s.get('n_failed', 0)}</span>
            <span class="key-metric nonsig">Model: {s.get('judge_model', '?')}</span>
        </p>
""")

        # Bar chart
        fig = make_breakdown_chart(entry, version_type)
        b64 = fig_to_base64(fig)
        html.append(f'        <img src="data:image/png;base64,{b64}" alt="{title} breakdown">')

        # Position bias table
        pb = s.get("position_bias", {})
        html.append("""
        <h3>Position Bias Breakdown</h3>
        <table style="width: auto;">
            <thead>
                <tr>
                    <th>Presentation Order</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
            </thead>
            <tbody>
""")
        for label, key_c, key_t in [
            ("Human response first", "human_first_correct", "human_first_total"),
            ("AI response first", "ai_first_correct", "ai_first_total"),
        ]:
            c = pb.get(key_c, 0)
            t = pb.get(key_t, 0)
            acc = f"{c/t:.1%}" if t > 0 else "N/A"
            html.append(f"""
                <tr>
                    <td>{label}</td>
                    <td class="metric">{c}</td>
                    <td class="metric">{t}</td>
                    <td class="metric">{acc}</td>
                </tr>
""")
        html.append("            </tbody>\n        </table>")

        # Target type breakdown
        details = entry["data"].get("judge_details", [])
        breakdown = compute_breakdown(details)
        html.append("""
        <h3>Target Type Breakdown</h3>
        <table style="width: auto;">
            <thead>
                <tr>
                    <th>Target Type</th>
                    <th>Correct</th>
                    <th>Total</th>
                    <th>Accuracy</th>
                </tr>
            </thead>
            <tbody>
""")
        for label, key in [("Human target", "target_human"), ("AI target", "target_ai")]:
            bd = breakdown[key]
            c, t = bd["correct"], bd["total"]
            acc = f"{c/t:.1%}" if t > 0 else "N/A"
            html.append(f"""
                <tr>
                    <td>{label}</td>
                    <td class="metric">{c}</td>
                    <td class="metric">{t}</td>
                    <td class="metric">{acc}</td>
                </tr>
""")
        html.append("            </tbody>\n        </table>")
        html.append("    </div>")

    html.append("</body>\n</html>")
    return "\n".join(html)


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate LLM judge HTML report.")
    parser.add_argument("--version", type=str, required=True, choices=VALID_VERSIONS,
                        help="Data version to report on.")
    args = parser.parse_args()
    version = args.version

    print(f"[INFO] Scanning judge results for version={version} ...")
    entries = discover_judge_files(version)
    print(f"[INFO] Found {len(entries)} judge files with data")

    for e in entries:
        s = e["summary"]
        print(f"  {e['strategy']}/{e['probe_type']}/is_{e['strength']}: "
              f"{s['success_rate']:.1%} ({s['n_correct']}/{s['n_judged']})")

    version_type = VERSION_DISPLAY.get(version, {}).get("type", "partner_identity")
    html = generate_html(version, entries, version_type)

    # Save report
    out_dir = EXP2_ROOT / "results" / "llama2_13b_chat" / version / "V1_causality" / "judge"
    out_path = out_dir / "judge_report.html"
    save_report(html, out_path)

    # Save standalone PNGs
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Summary chart
    summary_fig = make_summary_comparison_chart(entries, version_type)
    if summary_fig:
        summary_fig.savefig(fig_dir / "summary_comparison.png", dpi=150, bbox_inches="tight",
                            facecolor="white")
        plt.close(summary_fig)
        print(f"  Saved: {fig_dir / 'summary_comparison.png'}")

    # Per-entry charts
    for entry in entries:
        fig = make_breakdown_chart(entry, version_type)
        fname = f"{entry['strategy']}_{entry['probe_type']}_s{entry['strength']}.png"
        fig.savefig(fig_dir / fname, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved: {fig_dir / fname}")

    print(f"\n[DONE] Report saved to {out_path}")


if __name__ == "__main__":
    main()
