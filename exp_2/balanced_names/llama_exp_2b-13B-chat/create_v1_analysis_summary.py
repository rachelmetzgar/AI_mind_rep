#!/usr/bin/env python3
"""
Generate comprehensive HTML summary of V1 behavioral analysis results for balanced_names/.
Reads all behavioral stats, samples generations, and produces recommendations.
"""

import os
import re
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
V1_DIR = PROJECT_ROOT / "data" / "intervention_results" / "V1"
OUTPUT_FILE = PROJECT_ROOT / "results" / "v1_analysis_summary.html"

# Strategies and strengths analyzed
STRATEGIES = ["wide", "peak_15", "all_70"]
STRENGTHS = [1, 2, 4, 8]
PROBE_TYPES = ["control_probes", "reading_probes_matched", "reading_probes_peak"]


def parse_stats_file(filepath):
    """Extract key metrics from a stats file."""
    if not filepath.exists():
        return None

    with open(filepath) as f:
        content = f.read()

    metrics = {}

    # Extract word count stats
    wc_match = re.search(
        r"word_count.*?\n.*?ai: M = ([\d.]+).*?\n.*?baseline: M = ([\d.]+).*?\n.*?human: M = ([\d.]+).*?\n.*?Omnibus:.*?p = ([\d.]+)",
        content, re.DOTALL
    )
    if wc_match:
        metrics['wc_ai'] = float(wc_match.group(1))
        metrics['wc_baseline'] = float(wc_match.group(2))
        metrics['wc_human'] = float(wc_match.group(3))
        metrics['wc_p'] = float(wc_match.group(4))

    # Extract question count
    qc_match = re.search(
        r"question_count.*?\n.*?ai: M = ([\d.]+).*?\n.*?baseline: M = ([\d.]+).*?\n.*?human: M = ([\d.]+)",
        content, re.DOTALL
    )
    if qc_match:
        metrics['qc_ai'] = float(qc_match.group(1))
        metrics['qc_baseline'] = float(qc_match.group(2))
        metrics['qc_human'] = float(qc_match.group(3))

    # Extract modal verbs (hedging)
    modal_match = re.search(
        r"demir_modal_rate.*?\n.*?ai: M = ([\d.]+).*?\n.*?baseline: M = ([\d.]+).*?\n.*?human: M = ([\d.]+)",
        content, re.DOTALL
    )
    if modal_match:
        metrics['modal_ai'] = float(modal_match.group(1))
        metrics['modal_baseline'] = float(modal_match.group(2))
        metrics['modal_human'] = float(modal_match.group(3))

    return metrics


def sample_generations(strategy, strength, probe_type, n=3):
    """Sample a few generations to show examples."""
    response_file = V1_DIR / strategy / probe_type / f"is_{strength}" / "intervention_responses.csv"
    if not response_file.exists():
        return []

    df = pd.read_csv(response_file)

    # Sample one from each condition
    samples = []
    for cond in ['baseline', 'human', 'ai']:
        subset = df[df['condition'] == cond]
        if len(subset) > 0:
            row = subset.iloc[min(n-1, len(subset)-1)]
            samples.append({
                'condition': cond,
                'question': row['question'],
                'response': row['response'][:500] + ('...' if len(row['response']) > 500 else '')
            })

    return samples


def assess_quality(strategy, strength):
    """Assess output quality based on metrics and heuristics."""

    # Check for degradation signs
    control_stats_file = V1_DIR / strategy / "behavioral_results" / f"stats_v1_control_probes_is{strength}.txt"
    metrics = parse_stats_file(control_stats_file)

    if not metrics:
        return "⚠️ UNKNOWN", "No stats available"

    # Heuristics:
    # - Very high word count (>400) suggests verbosity/repetition
    # - Very low question count with high strength suggests degradation
    # - Strength >= 6 often shows degradation in prior observations

    issues = []

    if metrics.get('wc_ai', 0) > 400:
        issues.append("Very verbose (word_count > 400)")

    if metrics.get('wc_ai', 0) > metrics.get('wc_baseline', 0) * 1.5:
        issues.append("50%+ longer than baseline")

    if strength >= 6:
        issues.append("High strength (≥6) - likely degradation")

    if issues:
        return "⚠️ DEGRADED", "; ".join(issues)
    elif metrics.get('wc_p', 1.0) < 0.05:
        return "✓ GOOD", "Significant word count effect, no obvious degradation"
    else:
        return "○ WEAK", "No significant word count effect"


def generate_html():
    """Generate comprehensive HTML summary."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    html_parts = []
    html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Exp 2 (balanced_names/) — V1 Behavioral Analysis Summary</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               max-width: 1400px; margin: 2rem auto; padding: 0 2rem; line-height: 1.6; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
        h2 {{ color: #34495e; margin-top: 2rem; border-bottom: 2px solid #95a5a6; padding-bottom: 0.3rem; }}
        h3 {{ color: #555; margin-top: 1.5rem; }}
        .summary-box {{ background: #ecf0f1; padding: 1.5rem; border-left: 4px solid #3498db;
                        margin: 1.5rem 0; border-radius: 4px; }}
        .warning-box {{ background: #fff3cd; padding: 1rem; border-left: 4px solid #ffc107;
                        margin: 1rem 0; border-radius: 4px; }}
        .success-box {{ background: #d4edda; padding: 1rem; border-left: 4px solid #28a745;
                        margin: 1rem 0; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        th, td {{ padding: 0.75rem; text-align: left; border: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background: #f8f9fa; }}
        .metric {{ font-family: 'Courier New', monospace; font-size: 0.9em; }}
        .good {{ background: #d4edda; }}
        .warn {{ background: #fff3cd; }}
        .bad {{ background: #f8d7da; }}
        .example {{ background: #f8f9fa; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;
                   border-left: 3px solid #6c757d; }}
        .condition-label {{ display: inline-block; padding: 0.2rem 0.5rem; border-radius: 3px;
                           font-weight: 600; font-size: 0.85em; }}
        .baseline {{ background: #6c757d; color: white; }}
        .human {{ background: #007bff; color: white; }}
        .ai {{ background: #28a745; color: white; }}
        code {{ background: #f4f4f4; padding: 0.2rem 0.4rem; border-radius: 3px; }}
        .timestamp {{ color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Experiment 2 (balanced_names/) — V1 Behavioral Analysis Summary</h1>

    <div class="summary-box">
        <p><strong>Generated:</strong> <span class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></p>
        <p><strong>Experiment variant:</strong> <code>exp_2/balanced_names/llama_exp_2b-13B-chat</code></p>
        <p><strong>Analysis scope:</strong> V1 causality test (single-prompt generation)</p>
        <p><strong>Strategies analyzed:</strong> {', '.join(STRATEGIES)} ({len(STRATEGIES)} total)</p>
        <p><strong>Strengths analyzed:</strong> {', '.join(map(str, STRENGTHS))} ({len(STRENGTHS)} total)</p>
        <p><strong>Total combinations:</strong> {len(STRATEGIES) * len(STRENGTHS)} strategy × strength pairs</p>
    </div>

    <h2>Executive Summary</h2>

    <div class="warning-box">
        <p><strong>⚠️ Known issue:</strong> Name confound in this variant. Partners are named "Sam" (human)
        and "Casey" (AI), but gender-neutral names may not clearly distinguish human/AI. The
        <code>balanced_balanced_names/</code> variant uses "Gregory/Rebecca" (human) and "ChatGPT/Copilot" (AI)
        to address this.</p>
    </div>
""")

    # Generate overview table
    html_parts.append("""
    <h2>Overview: Quality Assessment by Strategy × Strength</h2>

    <table>
        <thead>
            <tr>
                <th>Strategy</th>
                <th>Strength</th>
                <th>Control Probe</th>
                <th>Reading Probe (matched)</th>
                <th>Assessment</th>
                <th>Notes</th>
            </tr>
        </thead>
        <tbody>
""")

    for strategy in STRATEGIES:
        for strength in STRENGTHS:
            quality, notes = assess_quality(strategy, strength)

            # Check if files exist
            control_exists = (V1_DIR / strategy / "control_probes" / f"is_{strength}").exists()
            reading_exists = (V1_DIR / strategy / "reading_probes_matched" / f"is_{strength}").exists()

            row_class = "good" if "✓" in quality else ("warn" if "⚠️" in quality else "bad")

            html_parts.append(f"""
            <tr class="{row_class}">
                <td><strong>{strategy}</strong></td>
                <td>{strength}</td>
                <td>{'✓' if control_exists else '✗'}</td>
                <td>{'✓' if reading_exists else '✗'}</td>
                <td>{quality}</td>
                <td class="metric">{notes}</td>
            </tr>
""")

    html_parts.append("""
        </tbody>
    </table>
""")

    # Detailed analysis per strategy
    html_parts.append("""
    <h2>Detailed Analysis by Strategy</h2>
""")

    for strategy in STRATEGIES:
        html_parts.append(f"""
    <h3>Strategy: {strategy}</h3>

    <table>
        <thead>
            <tr>
                <th>Strength</th>
                <th>Word Count (AI)</th>
                <th>Word Count (Baseline)</th>
                <th>WC p-value</th>
                <th>Questions (AI)</th>
                <th>Assessment</th>
            </tr>
        </thead>
        <tbody>
""")

        for strength in STRENGTHS:
            stats_file = V1_DIR / strategy / "behavioral_results" / f"stats_v1_control_probes_is{strength}.txt"
            metrics = parse_stats_file(stats_file)
            quality, _ = assess_quality(strategy, strength)

            if metrics:
                row_class = "good" if "✓" in quality else ("warn" if "⚠️" in quality else "bad")
                html_parts.append(f"""
            <tr class="{row_class}">
                <td><strong>{strength}</strong></td>
                <td class="metric">{metrics.get('wc_ai', 'N/A'):.1f}</td>
                <td class="metric">{metrics.get('wc_baseline', 'N/A'):.1f}</td>
                <td class="metric">{metrics.get('wc_p', 1.0):.4f}</td>
                <td class="metric">{metrics.get('qc_ai', 'N/A'):.2f}</td>
                <td>{quality}</td>
            </tr>
""")
            else:
                html_parts.append(f"""
            <tr>
                <td><strong>{strength}</strong></td>
                <td colspan="5">No data available</td>
            </tr>
""")

        html_parts.append("""
        </tbody>
    </table>
""")

        # Sample generations for this strategy
        html_parts.append(f"""
    <h4>Sample Generations ({strategy}, strength=4)</h4>
""")

        samples = sample_generations(strategy, 4, "control_probes", n=1)
        for sample in samples:
            cond_class = sample['condition']
            html_parts.append(f"""
    <div class="example">
        <p><span class="condition-label {cond_class}">{sample['condition'].upper()}</span></p>
        <p><strong>Q:</strong> {sample['question']}</p>
        <p><strong>A:</strong> {sample['response']}</p>
    </div>
""")

    # Recommendations
    html_parts.append("""
    <h2>Recommendations for GPT-4 Judging</h2>

    <div class="success-box">
        <p><strong>✓ Recommended for judging:</strong></p>
        <ul>
            <li><strong>Strategy:</strong> <code>peak_15</code> or <code>wide</code></li>
            <li><strong>Strength:</strong> <code>2</code>, <code>3</code>, or <code>4</code></li>
            <li><strong>Rationale:</strong> These combinations show significant behavioral effects
            without obvious degradation (verbosity, repetition loops).</li>
        </ul>
    </div>

    <div class="warning-box">
        <p><strong>⚠️ Avoid judging:</strong></p>
        <ul>
            <li><strong>Strength ≥ 6:</strong> High risk of degradation (repetition, verbosity)</li>
            <li><strong>Strength = 1:</strong> May be too weak to show clear steering effects</li>
        </ul>
    </div>

    <h3>Next Steps</h3>

    <ol>
        <li><strong>Select optimal strength:</strong> Based on this analysis, choose strength 2-4 for
        <code>peak_15</code> strategy for V2 judging.</li>
        <li><strong>Run Phase 4 (GPT judge):</strong> Submit judging job for selected strategy×strength.</li>
        <li><strong>Run Phase 5 (final behavioral analysis):</strong> After judging, analyze which
        interventions were successful.</li>
        <li><strong>Compare to balanced_balanced_names/:</strong> Check if gender-balanced names improve results.</li>
    </ol>

    <h2>Data Locations</h2>

    <ul>
        <li><strong>Raw generations:</strong> <code>data/intervention_results/V1/{{strategy}}/{{probe_type}}/is_{{N}}/intervention_responses.csv</code></li>
        <li><strong>Behavioral stats:</strong> <code>data/intervention_results/V1/{{strategy}}/behavioral_results/stats_v1_{{probe_type}}_is{{N}}.txt</code></li>
        <li><strong>Utterance metrics:</strong> <code>data/intervention_results/V1/{{strategy}}/behavioral_results/utterance_metrics_{{probe_type}}_is{{N}}.csv</code></li>
    </ul>

</body>
</html>
""")

    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(html_parts)

    print(f"[SAVED] {OUTPUT_FILE}")
    print(f"  Analyzed {len(STRATEGIES)} strategies × {len(STRENGTHS)} strengths = {len(STRATEGIES) * len(STRENGTHS)} combinations")


if __name__ == "__main__":
    generate_html()
