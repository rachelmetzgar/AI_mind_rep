#!/usr/bin/env python3
"""
Generate per-version behavioral summary HTML for V1 causal intervention results.

For each version (balanced_gpt, nonsense_codeword), produces a single HTML report
with a big table of all metrics × strengths, split by probe type. Saved alongside
the behavioral stats files.

Usage:
    python gen_behavioral_summary.py                    # all versions
    python gen_behavioral_summary.py --version balanced_gpt

Env: behavior_env (only needs pandas, numpy; no heavy deps)
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

EXP2_ROOT = Path(__file__).resolve().parent.parent  # exp_2/

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.report_utils import save_report

VERSIONS = ["balanced_gpt", "nonsense_codeword"]

PROBE_LABELS = {
    "operational": "Operational (pre-generation position)",
    "metacognitive": "Metacognitive",
    "metacognitive_matched": "Metacognitive (matched layers)",
}

ALL_METRICS = [
    "word_count", "question_count",
    "demir_modal_rate", "demir_verb_rate", "demir_adverb_rate",
    "demir_adjective_rate", "demir_quantifier_rate", "demir_noun_rate",
    "demir_total_rate",
    "fung_interpersonal_rate", "fung_referential_rate",
    "fung_structural_rate", "fung_cognitive_rate", "fung_total_rate",
    "nonfluency_rate", "liwc_filler_rate", "disfluency_rate",
    "like_rate", "tom_rate", "politeness_rate", "sentiment",
]

# Group metrics for visual separation
METRIC_GROUPS = {
    "Word-level":        ["word_count", "question_count"],
    "Hedge markers":     ["demir_modal_rate", "demir_verb_rate", "demir_adverb_rate",
                          "demir_adjective_rate", "demir_quantifier_rate", "demir_noun_rate",
                          "demir_total_rate"],
    "Discourse markers": ["fung_interpersonal_rate", "fung_referential_rate",
                          "fung_structural_rate", "fung_cognitive_rate", "fung_total_rate"],
    "Fluency":           ["nonfluency_rate", "liwc_filler_rate", "disfluency_rate"],
    "Social / style":    ["like_rate", "tom_rate", "politeness_rate", "sentiment"],
}


# ── Parsing ──────────────────────────────────────────────────


def parse_stats_file(filepath):
    """Parse a stats_v1_*.txt file. Returns dict of metric -> {baseline, human, ai, F, p, sd_*, pairwise}."""
    if not filepath.exists():
        return None

    content = filepath.read_text()
    metrics = {}

    # Parse per-metric blocks for means, SDs, and pairwise comparisons
    block_pat = re.compile(
        r"-{40,}\n(\S+)\s+\(n\s*=.*?\)\n-{40,}\n(.*?)(?=\n-{40,}|\nSUMMARY TABLE|\Z)",
        re.DOTALL,
    )

    for m in block_pat.finditer(content):
        metric_name = m.group(1)
        block = m.group(2)

        info = {}

        # Means and SDs
        for cond in ["ai", "baseline", "human"]:
            mean_m = re.search(
                rf"^\s*{cond}:\s+M\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)\s*\(SD\s*=\s*([\d.]+)\)",
                block, re.MULTILINE,
            )
            if mean_m:
                info[f"{cond}_mean"] = float(mean_m.group(1))
                info[f"{cond}_se"] = float(mean_m.group(2))
                info[f"{cond}_sd"] = float(mean_m.group(3))

        # Omnibus F and p
        omni = re.search(r"Omnibus:\s+F\((\d+),\s*(\d+)\)\s*=\s*([\d.]+),\s*p\s*=\s*([\d.]+)", block)
        if omni:
            info["df1"] = int(omni.group(1))
            info["df2"] = int(omni.group(2))
            info["F"] = float(omni.group(3))
            info["p"] = float(omni.group(4))

        # Pairwise
        pairwise = {}
        for pw in re.finditer(
            r"(\w+_vs_\w+):\s+diff\s*=\s*([-+\d.]+),\s*t\s*=\s*([-\d.]+),\s*p\s*=\s*([\d.]+)",
            block,
        ):
            pairwise[pw.group(1)] = {
                "diff": float(pw.group(2)),
                "t": float(pw.group(3)),
                "p": float(pw.group(4)),
            }
        info["pairwise"] = pairwise

        metrics[metric_name] = info

    return metrics if metrics else None


def discover_stats(beh_dir):
    """Discover all stats files under a behavioral results directory.

    Returns: dict of (probe_type, strength) -> parsed metrics dict
    """
    data = {}
    for f in sorted(beh_dir.glob("stats_v1_*_is*.txt")):
        m = re.match(r"stats_v1_(.+)_is(\d+)", f.stem)
        if not m:
            continue
        probe_type = m.group(1)
        strength = int(m.group(2))
        parsed = parse_stats_file(f)
        if parsed:
            data[(probe_type, strength)] = parsed
    return data


# ── HTML generation ──────────────────────────────────────────


def _sig(p):
    if isinstance(p, (int, float)) and not np.isnan(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
    return ""


def _p_style(p):
    if isinstance(p, (int, float)) and not np.isnan(p):
        if p < 0.001:
            return "background:#c0392b;color:white;"
        if p < 0.01:
            return "background:#e74c3c;color:white;"
        if p < 0.05:
            return "background:#f39c12;color:white;"
    return ""


def _fmt(val, metric):
    """Format a value based on metric type."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if metric in ("word_count", "question_count"):
        return f"{val:.1f}"
    return f"{val:.4f}"


def _pairwise_cell(pw_dict, metric):
    """Build a small sub-table of pairwise comparisons."""
    if not pw_dict:
        return ""
    parts = []
    pair_labels = {
        "ai_vs_baseline": "AI–BL",
        "ai_vs_human": "AI–Hum",
        "baseline_vs_human": "BL–Hum",
    }
    for pair_key, label in pair_labels.items():
        pw = pw_dict.get(pair_key)
        if not pw:
            continue
        sig = _sig(pw["p"])
        diff_str = _fmt(pw["diff"], metric)
        if diff_str != "—" and pw["diff"] > 0:
            diff_str = "+" + diff_str
        color = ""
        if pw["p"] < 0.05:
            color = "color:#c0392b;font-weight:bold;"
        parts.append(f'<span style="{color}">{label}: {diff_str} (p={pw["p"]:.4f}{sig})</span>')
    return "<br>".join(parts)


def generate_probe_table(data, probe_type, strengths):
    """Generate HTML table for one probe type across strengths."""
    rows = []

    # Header row 1: strength labels
    rows.append("<thead>")
    rows.append("<tr>")
    rows.append('<th rowspan="2" style="min-width:180px;">Metric</th>')
    for s in strengths:
        rows.append(f'<th colspan="6" class="strength-header">Strength {s}</th>')
    rows.append("</tr>")

    # Header row 2: sub-columns
    rows.append("<tr>")
    for _ in strengths:
        rows.append("<th>Baseline</th><th>Human</th><th>AI</th><th>F</th><th>p</th><th>Pairwise</th>")
    rows.append("</tr>")
    rows.append("</thead>")
    rows.append("<tbody>")

    for group_name, group_metrics in METRIC_GROUPS.items():
        # Group separator row
        total_cols = 1 + len(strengths) * 6
        rows.append(f'<tr class="group-row"><td colspan="{total_cols}">{group_name}</td></tr>')

        for metric in group_metrics:
            rows.append("<tr>")
            rows.append(f"<td><strong>{metric}</strong></td>")

            for s in strengths:
                mdata = data.get((probe_type, s), {}).get(metric)
                if not mdata:
                    rows.append('<td colspan="6" style="color:#ccc;text-align:center;">—</td>')
                    continue

                bl = _fmt(mdata.get("baseline_mean"), metric)
                hm = _fmt(mdata.get("human_mean"), metric)
                ai = _fmt(mdata.get("ai_mean"), metric)
                F_val = mdata.get("F")
                p_val = mdata.get("p")
                F_str = f"{F_val:.3f}" if F_val is not None else "—"
                p_str = f"{p_val:.4f}{_sig(p_val)}" if p_val is not None else "—"
                pw_html = _pairwise_cell(mdata.get("pairwise", {}), metric)

                rows.append(f'<td class="num">{bl}</td>')
                rows.append(f'<td class="num">{hm}</td>')
                rows.append(f'<td class="num">{ai}</td>')
                rows.append(f'<td class="num">{F_str}</td>')
                rows.append(f'<td class="num" style="{_p_style(p_val)}">{p_str}</td>')
                rows.append(f'<td class="pw">{pw_html}</td>')

            rows.append("</tr>")

    rows.append("</tbody>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def generate_html(version, strategy, data):
    """Generate full HTML report for one version × strategy."""
    probe_types = sorted(set(pt for pt, _ in data.keys()))
    strengths = sorted(set(s for _, s in data.keys()))

    # Count significant metrics per probe × strength
    sig_summary = []
    for pt in probe_types:
        for s in strengths:
            mdata = data.get((pt, s), {})
            n_sig = sum(1 for m in ALL_METRICS if m in mdata and mdata[m].get("p", 1) < 0.05)
            sig_summary.append((pt, s, n_sig, len(mdata)))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>V1 Behavioral Summary — {version} / {strategy}</title>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 2000px; margin: 2rem auto; padding: 0 2rem;
    background: #fafafa; color: #333; line-height: 1.4;
}}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
h2 {{ color: #2c3e50; margin-top: 2.5rem; }}
.info {{ background: white; padding: 1rem 1.5rem; border-left: 4px solid #3498db;
         margin: 1rem 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
.info p {{ margin: 0.3rem 0; }}
table {{
    border-collapse: collapse; width: 100%; margin: 1rem 0;
    background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    font-size: 0.82em;
}}
th, td {{ padding: 5px 8px; border: 1px solid #ddd; text-align: left; }}
th {{ background: #34495e; color: white; font-weight: 600; font-size: 0.85em;
      position: sticky; top: 0; z-index: 2; white-space: nowrap; }}
.strength-header {{ background: #2c3e50; text-align: center; font-size: 1em; }}
.group-row td {{
    background: #ecf0f1; font-weight: 700; font-size: 0.9em;
    color: #2c3e50; padding: 6px 8px; border-top: 2px solid #bdc3c7;
}}
.num {{ font-family: 'Courier New', monospace; font-size: 0.95em; white-space: nowrap; text-align: right; }}
.pw {{ font-size: 0.8em; line-height: 1.5; white-space: nowrap; }}
tr:nth-child(even) {{ background: #f8f9fa; }}
tr:hover {{ background: #eef3f7; }}
.legend {{ margin-top: 0.5rem; font-size: 0.85em; color: #555; }}
.sig-summary {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 1rem 0; }}
.sig-card {{
    background: white; border-radius: 6px; padding: 0.8rem 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 140px; text-align: center;
}}
.sig-card .count {{ font-size: 1.8em; font-weight: 700; color: #2c3e50; }}
.sig-card .label {{ font-size: 0.8em; color: #666; }}
</style>
</head>
<body>

<h1>V1 Behavioral Summary — {version}</h1>

<div class="info">
    <p><strong>Strategy:</strong> {strategy}</p>
    <p><strong>Strengths:</strong> {', '.join(str(s) for s in strengths)}</p>
    <p><strong>Probe types:</strong> {', '.join(PROBE_LABELS.get(pt, pt) for pt in probe_types)}</p>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>

<h2>Significance Overview</h2>
<div class="sig-summary">
"""

    for pt, s, n_sig, n_total in sig_summary:
        label = PROBE_LABELS.get(pt, pt).split("(")[0].strip()
        html += f"""<div class="sig-card">
    <div class="count">{n_sig}/{n_total}</div>
    <div class="label">{label}<br>strength {s}</div>
</div>
"""

    html += "</div>\n"

    # One table per probe type
    for pt in probe_types:
        pt_strengths = sorted(s for ptype, s in data.keys() if ptype == pt)
        if not pt_strengths:
            continue

        html += f'\n<h2>{PROBE_LABELS.get(pt, pt)}</h2>\n'
        html += generate_probe_table(data, pt, pt_strengths)
        html += '<p class="legend">* p &lt; .05 &nbsp; ** p &lt; .01 &nbsp; *** p &lt; .001</p>\n'

    html += "\n</body>\n</html>"
    return html


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Generate per-version V1 behavioral summary HTML.")
    parser.add_argument("--version", default=None, help="Version to generate (default: all)")
    args = parser.parse_args()

    versions = [args.version] if args.version else VERSIONS

    for version in versions:
        beh_root = EXP2_ROOT / "results" / "llama2_13b_chat" / version / "V1_causality" / "behavioral"
        if not beh_root.exists():
            print(f"[SKIP] {version}: {beh_root} not found")
            continue

        for strategy_dir in sorted(beh_root.iterdir()):
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name

            data = discover_stats(strategy_dir)
            if not data:
                print(f"[SKIP] {version}/{strategy}: no stats files found")
                continue

            print(f"[INFO] {version}/{strategy}: {len(data)} probe×strength combinations")

            html = generate_html(version, strategy, data)
            out_path = strategy_dir / "behavioral_summary.html"
            save_report(html, out_path)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
