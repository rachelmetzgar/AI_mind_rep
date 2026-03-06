#!/usr/bin/env python3
"""
Generate steered_samples_v1.html for a given variant.

Shows 3 sample responses per condition (baseline, human, ai) at each
strength for the peak_15 strategy (operational), so you can visually
compare text quality across the dose-response curve.

Usage:
    python generate_steered_samples.py --version balanced_gpt
    python generate_steered_samples.py --version nonsense_codeword

Env: behavior_env
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

EXP2_ROOT = Path(__file__).resolve().parent.parent  # exp_2/

VARIANTS = {
    "balanced_gpt": {
        "label": "Balanced GPT (Gregory/Rebecca + GPT partner)",
        "strengths": [1, 2, 4, 5, 6, 8],
    },
    "nonsense_codeword": {
        "label": "Nonsense Codeword (control)",
        "strengths": [2, 4, 8, 16],
    },
    "names": {
        "label": "Names (Sam/Casey)",
        "strengths": [1, 2, 3, 4, 5, 6, 8],
    },
    "balanced_names": {
        "label": "Balanced Names (Gregory/Rebecca)",
        "strengths": [1, 2, 4, 8],
    },
    "labels": {
        "label": "Labels ('a human' / 'an AI')",
        "strengths": [2, 4, 5, 6],
    },
    "labels_turnwise": {
        "label": "Labels Turnwise (Human:/AI: prefix)",
        "strengths": [2, 4, 8, 16],
    },
    "you_are_labels": {
        "label": "You Are Labels",
        "strengths": [2, 4, 8, 16],
    },
    "you_are_labels_turnwise": {
        "label": "You Are Labels Turnwise",
        "strengths": [2, 4, 8, 16],
    },
    "you_are_balanced_gpt": {
        "label": "You Are Balanced GPT",
        "strengths": [2, 4, 8, 16],
    },
    "nonsense_ignore": {
        "label": "Nonsense Ignore (control)",
        "strengths": [2, 4, 8, 16],
    },
}

COND_COLORS = {
    "baseline": "#6c757d",
    "human": "#3498db",
    "ai": "#2ecc71",
}

N_SAMPLES = 3
STRATEGY = "peak_15"
PROBE_TYPE = "operational"


def compute_ttr(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)


def generate_html(version):
    vconf = VARIANTS[version]
    data_root = EXP2_ROOT / "results" / "llama2_13b_chat" / version / "V1_causality" / "data" / STRATEGY / PROBE_TYPE

    sections = []
    for strength in vconf["strengths"]:
        csv_path = data_root / f"is_{strength}" / "intervention_responses.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        # Compute per-condition mean TTR
        ttr_summary = {}
        for cond in ["baseline", "human", "ai"]:
            subset = df[df["condition"] == cond]
            ttrs = [compute_ttr(r) for r in subset["response"]]
            ttr_summary[cond] = np.mean(ttrs) if ttrs else float("nan")

        ttr_str = " / ".join(
            f'<span style="{"color:#c0392b;font-weight:bold;" if ttr_summary[c] < 0.3 else ""}">'
            f'{ttr_summary[c]:.2f}</span>'
            for c in ["baseline", "human", "ai"]
        )

        sections.append(
            f'<h3>Strength {strength}'
            f' <span style="font-size:0.8em;color:#666;">'
            f'TTR (B/H/AI): {ttr_str}</span></h3>'
        )

        for cond in ["baseline", "human", "ai"]:
            subset = df[df["condition"] == cond].reset_index(drop=True)
            # Pick N_SAMPLES evenly spaced
            indices = np.linspace(0, len(subset) - 1, N_SAMPLES, dtype=int)
            for i in indices:
                row = subset.iloc[i]
                resp = str(row["response"])
                wc = len(resp.split())
                ttr = compute_ttr(resp)
                color = COND_COLORS[cond]
                sections.append(f"""
                <div class="example" style="border-left-color: {color};">
                    <span class="condition-label" style="background: {color}; color: white;">
                        {cond.upper()} ({wc} words, TTR={ttr:.2f})
                    </span>
                    <p><strong>Q:</strong> {str(row["question"])[:200]}</p>
                    <p>{resp[:800]}{"..." if len(resp) > 800 else ""}</p>
                </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Steered Samples — {vconf['label']} (peak_15, control_probes)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               max-width: 1200px; margin: 2rem auto; padding: 0 2rem; line-height: 1.5;
               background: #fafafa; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.5rem; }}
        h3 {{ color: #2c3e50; margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
        .note {{ color: #666; font-style: italic; margin: 0.5rem 0; }}
        .example {{ background: white; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;
                   border-left: 3px solid #6c757d; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
        .example p {{ margin: 0.3rem 0; font-size: 0.9em; }}
        .condition-label {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 3px;
                           font-weight: 600; font-size: 0.8em; margin-bottom: 0.3rem; }}
    </style>
</head>
<body>
    <h1>Steered Samples — {vconf['label']}</h1>
    <p class="note">Strategy: {STRATEGY} | Probe: {PROBE_TYPE} | {N_SAMPLES} samples per condition per strength</p>
    <p class="note">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

{"".join(sections)}

</body></html>"""

    out_dir = EXP2_ROOT / "results" / "llama2_13b_chat" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steered_samples_v1.html"
    out_path.write_text(html)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Variant name")
    args = parser.parse_args()

    if args.version not in VARIANTS:
        print(f"[ERROR] Unknown version: {args.version}")
        print(f"  Available: {', '.join(VARIANTS)}")
        exit(1)

    generate_html(args.version)
