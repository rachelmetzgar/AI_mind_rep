#!/usr/bin/env python3
"""
Generate comprehensive HTML QC summary for ALL Exp 2 variants.

Reads behavioral stats from all 4 variants (names, balanced_names, balanced_gpt, labels),
generates embedded matplotlib figures (bar charts, heatmaps), and produces a single
cross-variant comparison HTML report.

Also generates per-variant HTML summaries in each variant's results/ dir.

Usage:
    python create_v1_qc_summary.py

Env: behavior_env
"""

import os
import re
import io
import base64
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — figures will be skipped")

# ============================================================
#  CONFIG
# ============================================================

EXP2_ROOT = Path(__file__).resolve().parent.parent.parent  # exp_2/code/analysis -> exp_2/

VARIANTS = {
    "names": {
        "label": "Names (Sam/Casey)",
        "color": "#e74c3c",
        "path": EXP2_ROOT / "data" / "names",
        "strategies": ["narrow", "peak_15", "wide", "all_70"],
        "strengths": [1, 2, 3, 4, 5, 6, 8],
        "reading_probe": "reading_probes_matched",
        "note": "DEPRECATED: name confound (probes learned partner names)",
    },
    "balanced_names": {
        "label": "Balanced Names (Gregory/Rebecca)",
        "color": "#3498db",
        "path": EXP2_ROOT / "data" / "balanced_names",
        "strategies": ["peak_15", "wide", "all_70"],
        "strengths": [1, 2, 4, 8],
        "reading_probe": "reading_probes_peak",
        "note": "Gender-balanced human names, same AI names",
    },
    "balanced_gpt": {
        "label": "Balanced GPT (Gregory/Rebecca + GPT partner)",
        "color": "#2ecc71",
        "path": EXP2_ROOT / "data" / "balanced_gpt",
        "strategies": ["peak_15", "wide", "all_70"],
        "strengths": [1, 2, 4, 5, 6, 8],
        "reading_probe": "reading_probes_peak",
        "note": "Cross-model generalization: AI partner was GPT, not LLaMA",
    },
    "labels": {
        "label": "Labels ('a human' / 'an AI')",
        "color": "#9b59b6",
        "path": EXP2_ROOT / "data" / "labels",
        "strategies": ["peak_15", "wide", "all_70"],
        "strengths": [2, 4, 5, 6],
        "reading_probe": "reading_probes_peak",
        "note": "PRIMARY: no name cues, pure identity labels",
    },
}

# Key metrics to highlight
KEY_METRICS = [
    "word_count", "question_count",
    "fung_interpersonal_rate", "fung_structural_rate", "fung_cognitive_rate", "fung_total_rate",
    "demir_modal_rate", "demir_total_rate",
    "like_rate", "tom_rate", "politeness_rate", "sentiment",
    "disfluency_rate",
]

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


# ============================================================
#  PARSING
# ============================================================

def parse_stats_file(filepath):
    """Parse a stats_v1_*.txt file into structured metrics dict."""
    if not filepath.exists():
        return None

    with open(filepath) as f:
        content = f.read()

    metrics = {}
    # Parse summary table at bottom of file
    table_match = re.search(r"SUMMARY TABLE.*?\n-+\n(.*?)\n-+", content, re.DOTALL)
    if not table_match:
        return None

    for line in table_match.group(1).strip().split("\n"):
        parts = line.split()
        if len(parts) >= 6:
            metric = parts[0]
            try:
                baseline = float(parts[1])
                human = float(parts[2])
                ai = float(parts[3])
                F_val = float(parts[4])
                # p-value may have stars appended
                p_str = re.match(r"([\d.]+)", parts[5])
                p_val = float(p_str.group(1)) if p_str else np.nan
                metrics[metric] = {
                    "baseline": baseline, "human": human, "ai": ai,
                    "F": F_val, "p": p_val,
                }
            except (ValueError, IndexError):
                continue

    return metrics


def parse_pairwise(filepath, metric_name):
    """Extract pairwise t-test results for a specific metric."""
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        content = f.read()

    # Find the metric section
    pattern = rf"{re.escape(metric_name)}.*?Pairwise.*?\n(.*?)(?:\n\n|\n-{{40}})"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return {}

    pairs = {}
    for line in match.group(1).strip().split("\n"):
        pw_match = re.search(r"(\w+_vs_\w+):\s+diff\s*=\s*([-+\d.]+),\s*t\s*=\s*([-\d.]+),\s*p\s*=\s*([\d.]+)", line)
        if pw_match:
            pairs[pw_match.group(1)] = {
                "diff": float(pw_match.group(2)),
                "t": float(pw_match.group(3)),
                "p": float(pw_match.group(4)),
            }
    return pairs


def load_all_stats():
    """Load behavioral stats for all variants × strategies × strengths × probe types."""
    all_data = {}

    for vname, vconf in VARIANTS.items():
        v1_dir = vconf["path"] / "intervention_results" / "V1"
        if not v1_dir.exists():
            print(f"[SKIP] {vname}: {v1_dir} not found")
            continue

        for strategy in vconf["strategies"]:
            strat_dir = v1_dir / strategy
            if not strat_dir.exists():
                continue

            beh_dir = strat_dir / "behavioral_results"
            if not beh_dir.exists():
                continue

            # Discover available strengths and probe types
            for stats_file in sorted(beh_dir.glob("stats_v1_*_is*.txt")):
                fname = stats_file.stem
                # Parse: stats_v1_{probe_type}_is{strength}
                m = re.match(r"stats_v1_(.+)_is(\d+)", fname)
                if not m:
                    continue
                probe_type = m.group(1)
                strength = int(m.group(2))

                metrics = parse_stats_file(stats_file)
                if metrics:
                    key = (vname, strategy, probe_type, strength)
                    all_data[key] = metrics

    return all_data


def load_sample_generations(variant_name, strategy, strength, probe_type="control_probes", n_samples=1):
    """Load sample generations from intervention_responses.csv."""
    vconf = VARIANTS[variant_name]
    csv_path = (vconf["path"] / "intervention_results" / "V1" /
                strategy / probe_type / f"is_{strength}" / "intervention_responses.csv")

    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    samples = []
    for cond in ["baseline", "human", "ai"]:
        subset = df[df["condition"] == cond]
        if len(subset) > 0:
            row = subset.iloc[min(n_samples - 1, len(subset) - 1)]
            resp = str(row["response"])
            samples.append({
                "condition": cond,
                "question": str(row["question"])[:200],
                "response": resp[:600] + ("..." if len(resp) > 600 else ""),
                "word_count": len(resp.split()),
            })
    return samples


# ============================================================
#  FIGURES
# ============================================================

def fig_to_base64(fig):
    """Convert matplotlib figure to base64-encoded PNG for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def make_condition_bar_chart(all_data, variant, strategy, strength, probe_type, metrics=None):
    """Bar chart of condition means for key metrics at a specific config."""
    key = (variant, strategy, probe_type, strength)
    data = all_data.get(key)
    if not data:
        return None

    if metrics is None:
        metrics = KEY_METRICS

    available = [m for m in metrics if m in data]
    if not available:
        return None

    fig, axes = plt.subplots(2, (len(available) + 1) // 2, figsize=(max(12, len(available) * 1.5), 8))
    axes = axes.flatten()

    colors = {"baseline": "#95a5a6", "human": "#3498db", "ai": "#2ecc71"}

    for i, metric in enumerate(available):
        ax = axes[i]
        vals = data[metric]
        conditions = ["baseline", "human", "ai"]
        means = [vals[c] for c in conditions]
        bars = ax.bar(conditions, means, color=[colors[c] for c in conditions], edgecolor="white", linewidth=0.5)
        ax.set_title(metric.replace("_", "\n"), fontsize=8, fontweight="bold")
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

        # Add significance star
        p = vals.get("p", 1.0)
        if p < 0.001:
            ax.set_title(f"{metric.replace('_', chr(10))}\n***", fontsize=8, fontweight="bold")
        elif p < 0.01:
            ax.set_title(f"{metric.replace('_', chr(10))}\n**", fontsize=8, fontweight="bold")
        elif p < 0.05:
            ax.set_title(f"{metric.replace('_', chr(10))}\n*", fontsize=8, fontweight="bold")

    # Hide unused axes
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{VARIANTS[variant]['label']} — {strategy} / {probe_type} / strength={strength}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_pvalue_heatmap(all_data, variant, strategy, probe_type):
    """Heatmap of p-values: metrics × strengths."""
    vconf = VARIANTS[variant]
    strengths = sorted([s for (v, st, pt, s) in all_data if v == variant and st == strategy and pt == probe_type])
    if not strengths:
        return None

    metrics = ALL_METRICS
    p_matrix = np.full((len(metrics), len(strengths)), np.nan)

    for j, strength in enumerate(strengths):
        key = (variant, strategy, probe_type, strength)
        data = all_data.get(key, {})
        for i, metric in enumerate(metrics):
            if metric in data:
                p_matrix[i, j] = data[metric].get("p", np.nan)

    fig, ax = plt.subplots(figsize=(max(6, len(strengths) * 1.2), max(8, len(metrics) * 0.4)))

    # Custom colormap: red = significant, white = not
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.LogNorm(vmin=0.001, vmax=1.0)

    im = ax.imshow(p_matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([str(s) for s in strengths], fontsize=9)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("_rate", "").replace("_", " ") for m in metrics], fontsize=8)

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(strengths)):
            p = p_matrix[i, j]
            if np.isnan(p):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="gray")
            else:
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                color = "white" if p < 0.01 else "black"
                ax.text(j, i, f"{p:.3f}{stars}", ha="center", va="center", fontsize=6.5, color=color)

    ax.set_xlabel("Intervention Strength", fontsize=10)
    ax.set_title(f"{VARIANTS[variant]['label']} — {strategy} / {probe_type}\np-value heatmap (log scale)",
                 fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("p-value", fontsize=9)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_cross_variant_comparison(all_data, strategy="peak_15", probe_type_default="control_probes"):
    """Compare key metrics across variants at a common strength (2 or 4)."""
    # Find a strength common to multiple variants
    common_strengths = set()
    for (v, st, pt, s) in all_data:
        if st == strategy and pt == probe_type_default:
            common_strengths.add(s)

    if not common_strengths:
        return None

    # Prefer strength 2 or 4
    target_strength = 4 if 4 in common_strengths else (2 if 2 in common_strengths else min(common_strengths))

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    highlight_metrics = ["word_count", "question_count", "fung_interpersonal_rate",
                         "fung_total_rate", "like_rate", "tom_rate"]

    for idx, metric in enumerate(highlight_metrics):
        ax = axes[idx]
        variant_names = []
        baseline_vals = []
        human_vals = []
        ai_vals = []

        for vname in VARIANTS:
            key = (vname, strategy, probe_type_default, target_strength)
            data = all_data.get(key, {})
            if metric in data:
                variant_names.append(VARIANTS[vname]["label"].split("(")[0].strip())
                baseline_vals.append(data[metric]["baseline"])
                human_vals.append(data[metric]["human"])
                ai_vals.append(data[metric]["ai"])

        if not variant_names:
            ax.set_visible(False)
            continue

        x = np.arange(len(variant_names))
        width = 0.25

        ax.bar(x - width, baseline_vals, width, label="Baseline", color="#95a5a6", edgecolor="white")
        ax.bar(x, human_vals, width, label="Human", color="#3498db", edgecolor="white")
        ax.bar(x + width, ai_vals, width, label="AI", color="#2ecc71", edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(variant_names, fontsize=7, rotation=15, ha="right")
        ax.set_title(metric.replace("_", " "), fontsize=9, fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"Cross-Variant Comparison — {strategy} / {probe_type_default} / strength={target_strength}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_dose_response_plot(all_data, variant, strategy, probe_type, metrics=None):
    """Line plot showing metric values across intervention strengths."""
    if metrics is None:
        metrics = ["word_count", "fung_interpersonal_rate", "like_rate", "question_count"]

    vconf = VARIANTS[variant]
    strengths = sorted([s for (v, st, pt, s) in all_data
                        if v == variant and st == strategy and pt == probe_type])

    if len(strengths) < 2:
        return None

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 3.5, 4))
    if len(metrics) == 1:
        axes = [axes]

    colors = {"baseline": "#95a5a6", "human": "#3498db", "ai": "#2ecc71"}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for cond in ["baseline", "human", "ai"]:
            vals = []
            for s in strengths:
                key = (variant, strategy, probe_type, s)
                data = all_data.get(key, {})
                if metric in data:
                    vals.append(data[metric][cond])
                else:
                    vals.append(np.nan)
            ax.plot(strengths, vals, 'o-', color=colors[cond], label=cond, markersize=4, linewidth=1.5)

        ax.set_xlabel("Strength", fontsize=9)
        ax.set_title(metric.replace("_", " "), fontsize=9, fontweight="bold")
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle(f"{VARIANTS[variant]['label']} — {strategy} / {probe_type}\nDose-Response",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig_to_base64(fig)


# ============================================================
#  QUALITY ASSESSMENT
# ============================================================

def assess_quality(all_data, variant, strategy, strength, probe_type="control_probes"):
    """Assess output quality based on metrics."""
    key = (variant, strategy, probe_type, strength)
    data = all_data.get(key)
    if not data:
        return "UNKNOWN", "No stats available", "#6c757d"

    issues = []
    n_sig = 0

    wc = data.get("word_count", {})
    ai_wc = wc.get("ai", 0)
    bl_wc = wc.get("baseline", 1)

    if ai_wc > 500:
        issues.append(f"Very verbose (AI={ai_wc:.0f} words)")
    elif ai_wc > 350 and ai_wc > bl_wc * 1.5:
        issues.append(f"Verbose (AI={ai_wc:.0f}, {ai_wc/bl_wc:.1f}x baseline)")

    if strength >= 6:
        issues.append("High strength — likely degradation")

    # Count significant metrics (excluding word_count which may just indicate verbosity)
    for metric in ALL_METRICS:
        if metric == "word_count":
            continue
        if metric in data and data[metric].get("p", 1) < 0.05:
            n_sig += 1

    if issues:
        return "DEGRADED", "; ".join(issues), "#f8d7da"
    elif n_sig >= 3:
        return "GOOD", f"{n_sig} significant metrics (excl. word_count)", "#d4edda"
    elif n_sig >= 1:
        return "MARGINAL", f"{n_sig} significant metric(s)", "#fff3cd"
    else:
        return "WEAK", "No significant behavioral effects", "#f8f9fa"


# ============================================================
#  HTML GENERATION
# ============================================================

def _sig(p):
    if isinstance(p, float) and not np.isnan(p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
    return ""


def _cell_color(p):
    if isinstance(p, float) and not np.isnan(p):
        if p < 0.001: return "background: #c0392b; color: white;"
        elif p < 0.01: return "background: #e74c3c; color: white;"
        elif p < 0.05: return "background: #f39c12; color: white;"
    return ""


def generate_variant_section(all_data, variant_name):
    """Generate HTML section for one variant."""
    vconf = VARIANTS[variant_name]
    html = []

    html.append(f"""
    <h2 id="{variant_name}" style="border-left: 5px solid {vconf['color']}; padding-left: 10px;">
        {vconf['label']}
    </h2>
    <p class="note">{vconf['note']}</p>
    """)

    # Overview table: quality assessment across strategies × strengths
    html.append("""
    <h3>Quality Assessment Overview</h3>
    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Strength</th>
                <th>Assessment</th>
                <th># Sig Metrics</th>
                <th>Word Count (B/H/AI)</th>
                <th>Fung Interp. (B/H/AI)</th>
                <th>Like rate (B/H/AI)</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
    """)

    for strategy in vconf["strategies"]:
        for strength in vconf["strengths"]:
            # Try control probes first
            key = (variant_name, strategy, "control_probes", strength)
            data = all_data.get(key)
            if not data:
                continue

            quality, notes, bg = assess_quality(all_data, variant_name, strategy, strength)

            wc = data.get("word_count", {})
            fi = data.get("fung_interpersonal_rate", {})
            lr = data.get("like_rate", {})

            n_sig = sum(1 for m in ALL_METRICS if m != "word_count"
                       and m in data and data[m].get("p", 1) < 0.05)

            html.append(f"""
            <tr style="background: {bg};">
                <td><strong>{strategy}</strong></td>
                <td>{strength}</td>
                <td><strong>{quality}</strong></td>
                <td>{n_sig}</td>
                <td class="metric">{wc.get('baseline',0):.0f} / {wc.get('human',0):.0f} / {wc.get('ai',0):.0f}
                    <span style="{_cell_color(wc.get('p',1))}">{_sig(wc.get('p',1))}</span></td>
                <td class="metric">{fi.get('baseline',0):.4f} / {fi.get('human',0):.4f} / {fi.get('ai',0):.4f}
                    <span style="{_cell_color(fi.get('p',1))}">{_sig(fi.get('p',1))}</span></td>
                <td class="metric">{lr.get('baseline',0):.4f} / {lr.get('human',0):.4f} / {lr.get('ai',0):.4f}
                    <span style="{_cell_color(lr.get('p',1))}">{_sig(lr.get('p',1))}</span></td>
                <td style="font-size: 0.85em;">{notes}</td>
            </tr>
            """)

    html.append("</tbody></table>")

    # Per-strategy detailed sections with figures
    for strategy in vconf["strategies"]:
        available_strengths = sorted([s for (v, st, pt, s) in all_data
                                      if v == variant_name and st == strategy and pt == "control_probes"])
        if not available_strengths:
            continue

        html.append(f"<h3>Strategy: {strategy}</h3>")

        # P-value heatmap
        if HAS_MPL:
            heatmap_b64 = make_pvalue_heatmap(all_data, variant_name, strategy, "control_probes")
            if heatmap_b64:
                html.append(f'<img src="data:image/png;base64,{heatmap_b64}" '
                           f'style="max-width:100%; margin: 1rem 0;">')

            # Dose-response plots
            dose_b64 = make_dose_response_plot(all_data, variant_name, strategy, "control_probes")
            if dose_b64:
                html.append(f'<img src="data:image/png;base64,{dose_b64}" '
                           f'style="max-width:100%; margin: 1rem 0;">')

        # Full metrics table for each strength
        html.append("""
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
        """)
        for s in available_strengths:
            html.append(f"<th colspan='4'>Strength {s}</th>")
        html.append("</tr><tr><th></th>")
        for s in available_strengths:
            html.append("<th>Baseline</th><th>Human</th><th>AI</th><th>p</th>")
        html.append("</tr></thead><tbody>")

        for metric in ALL_METRICS:
            html.append(f"<tr><td><strong>{metric}</strong></td>")
            for s in available_strengths:
                key = (variant_name, strategy, "control_probes", s)
                data = all_data.get(key, {})
                mdata = data.get(metric, {})
                if mdata:
                    p = mdata.get("p", np.nan)
                    fmt = ".0f" if metric in ("word_count", "question_count") else ".4f"
                    html.append(f"""
                        <td class="metric">{mdata['baseline']:{fmt}}</td>
                        <td class="metric">{mdata['human']:{fmt}}</td>
                        <td class="metric">{mdata['ai']:{fmt}}</td>
                        <td class="metric" style="{_cell_color(p)}">{p:.4f}{_sig(p)}</td>
                    """)
                else:
                    html.append("<td colspan='4'>—</td>")
            html.append("</tr>")
        html.append("</tbody></table>")

        # Sample generations for the best-looking strength
        best_strength = None
        for s in [2, 4, 1, 8]:
            if s in available_strengths:
                best_strength = s
                break
        if best_strength is None and available_strengths:
            best_strength = available_strengths[0]

        if best_strength:
            samples = load_sample_generations(variant_name, strategy, best_strength)
            if samples:
                html.append(f"<h4>Sample Generations ({strategy}, strength={best_strength})</h4>")
                for sample in samples:
                    cond = sample["condition"]
                    cond_colors = {"baseline": "#6c757d", "human": "#3498db", "ai": "#2ecc71"}
                    html.append(f"""
                    <div class="example">
                        <span class="condition-label" style="background: {cond_colors.get(cond, '#999')}; color: white;">
                            {cond.upper()} ({sample['word_count']} words)
                        </span>
                        <p><strong>Q:</strong> {sample['question']}</p>
                        <p>{sample['response']}</p>
                    </div>
                    """)

    return "\n".join(html)


def generate_full_html(all_data):
    """Generate the complete HTML report."""
    html = []
    html.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exp 2 — V1 QC Summary (All Variants)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               max-width: 1600px; margin: 2rem auto; padding: 0 2rem; line-height: 1.5;
               background: #fafafa; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
        h2 {{ color: #2c3e50; margin-top: 3rem; }}
        h3 {{ color: #555; margin-top: 2rem; }}
        .summary-box {{ background: white; padding: 1.5rem; border-left: 4px solid #3498db;
                        margin: 1.5rem 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .note {{ color: #666; font-style: italic; margin: 0.5rem 0; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: white;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 0.85em; }}
        th, td {{ padding: 0.5rem 0.6rem; text-align: left; border: 1px solid #e0e0e0; }}
        th {{ background: #34495e; color: white; font-weight: 600; font-size: 0.85em;
              position: sticky; top: 0; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .metric {{ font-family: 'Courier New', monospace; font-size: 0.9em; white-space: nowrap; }}
        .example {{ background: white; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;
                   border-left: 3px solid #6c757d; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
        .example p {{ margin: 0.3rem 0; font-size: 0.9em; }}
        .condition-label {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
                           font-weight: 600; font-size: 0.8em; margin-bottom: 0.3rem; }}
        .toc {{ background: white; padding: 1.5rem; margin: 1.5rem 0; border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .toc ul {{ padding-left: 1.5rem; }}
        .toc li {{ margin: 0.3rem 0; }}
        .toc a {{ color: #3498db; text-decoration: none; }}
        .toc a:hover {{ text-decoration: underline; }}
        .timestamp {{ color: #6c757d; font-size: 0.9em; }}
        img {{ border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <h1>Experiment 2 — V1 QC Summary (All Variants)</h1>

    <div class="summary-box">
        <p><strong>Generated:</strong> <span class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
        <p><strong>Purpose:</strong> Compare causal intervention results across 4 dataset variants to assess
        whether partner-identity steering persists regardless of how partners are labeled.</p>
        <p><strong>Variants:</strong> names (deprecated), balanced_names, balanced_gpt, labels (primary)</p>
    </div>

    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#cross-variant">Cross-Variant Comparison</a></li>
    """)

    for vname, vconf in VARIANTS.items():
        html.append(f'<li><a href="#{vname}">{vconf["label"]}</a></li>')

    html.append("""
        </ul>
    </div>
    """)

    # Cross-variant comparison section
    html.append('<h2 id="cross-variant">Cross-Variant Comparison</h2>')

    if HAS_MPL:
        for strategy in ["peak_15", "wide", "all_70"]:
            cross_b64 = make_cross_variant_comparison(all_data, strategy=strategy)
            if cross_b64:
                html.append(f'<h3>Strategy: {strategy}</h3>')
                html.append(f'<img src="data:image/png;base64,{cross_b64}" '
                           f'style="max-width:100%; margin: 1rem 0;">')

    # Quick comparison table: peak_15 at strength 2 and 4
    for target_strength in [2, 4]:
        html.append(f"<h3>Peak-15, Strength={target_strength} — All Variants</h3>")
        html.append("""
        <table>
            <thead>
                <tr><th>Metric</th>
        """)
        for vname in VARIANTS:
            html.append(f"<th colspan='4' style='background:{VARIANTS[vname]['color']};'>{VARIANTS[vname]['label'].split('(')[0].strip()}</th>")
        html.append("</tr><tr><th></th>")
        for _ in VARIANTS:
            html.append("<th>BL</th><th>Hum</th><th>AI</th><th>p</th>")
        html.append("</tr></thead><tbody>")

        for metric in KEY_METRICS:
            html.append(f"<tr><td><strong>{metric}</strong></td>")
            for vname in VARIANTS:
                key = (vname, "peak_15", "control_probes", target_strength)
                data = all_data.get(key, {})
                mdata = data.get(metric, {})
                if mdata:
                    p = mdata.get("p", np.nan)
                    fmt = ".0f" if metric in ("word_count", "question_count") else ".4f"
                    html.append(f"""
                        <td class="metric">{mdata['baseline']:{fmt}}</td>
                        <td class="metric">{mdata['human']:{fmt}}</td>
                        <td class="metric">{mdata['ai']:{fmt}}</td>
                        <td class="metric" style="{_cell_color(p)}">{p:.4f}{_sig(p)}</td>
                    """)
                else:
                    html.append("<td colspan='4' style='color:#ccc;'>no data</td>")
            html.append("</tr>")
        html.append("</tbody></table>")

    # Per-variant sections
    for vname in VARIANTS:
        html.append(generate_variant_section(all_data, vname))

    html.append("</body></html>")
    return "\n".join(html)


# ============================================================
#  MAIN
# ============================================================

def main():
    print("[INFO] Loading all behavioral stats...")
    all_data = load_all_stats()
    print(f"[INFO] Loaded {len(all_data)} strategy×probe×strength combinations")

    # Count per variant
    for vname in VARIANTS:
        count = sum(1 for k in all_data if k[0] == vname)
        print(f"  {vname}: {count} combinations")

    print("[INFO] Generating HTML report...")
    html = generate_full_html(all_data)

    import sys as _sys
    _sys.path.insert(0, str(EXP2_ROOT / "code"))
    from src.report_utils import save_report

    output_path = EXP2_ROOT / "results" / "cross_variant" / "v1_qc_summary_all_variants.html"
    save_report(html, output_path)

    # Also save per-variant HTML summaries
    for vname, vconf in VARIANTS.items():
        results_dir = EXP2_ROOT / "results" / vname
        results_dir.mkdir(parents=True, exist_ok=True)
        variant_html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Exp 2 ({vname}) V1 Summary</title>
<style>
body {{ font-family: sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85em; }}
th, td {{ padding: 0.5rem; border: 1px solid #ddd; text-align: left; }}
th {{ background: #34495e; color: white; }}
.metric {{ font-family: monospace; }}
.example {{ background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #6c757d; }}
.condition-label {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px; font-weight: 600; font-size: 0.8em; }}
.note {{ color: #666; font-style: italic; }}
img {{ max-width: 100%; }}
</style></head><body>
<h1>Experiment 2 ({vconf['label']}) — V1 Summary</h1>
<p class="note">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
{generate_variant_section(all_data, vname)}
</body></html>"""
        variant_output = results_dir / "v1_analysis_summary.html"
        save_report(variant_html, variant_output)

    print("\n[DONE] All summaries generated.")


if __name__ == "__main__":
    main()
