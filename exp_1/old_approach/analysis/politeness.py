#!/usr/bin/env python3
"""
Script name: politeness_analysis.py
Purpose: Analyze politeness in participant transcripts using regex markers.
    - Compute politeness scores (+1 polite, –1 impolite).
    - Aggregate per-trial and per-subject averages.
    - Compare Human vs Bot conditions (paired t-tests).
    - Compare Human vs Bot across social vs nonsocial topics.
    - Run 2-way repeated-measures ANOVA: Condition × Sociality.
    - Save results (CSVs, plots, stats) to results/behavior/politeness/.

Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub').
    - topics.csv (with 'topic' and 'social' coding).

Outputs:
    - politeness_by_interaction.csv
    - politeness_subject_summary.csv
    - politeness_subject_social_summary.csv
    - politeness_stats.txt
    - politeness_violinplot.png
    - politeness_main_effect_lines.png

Usage:
    python code/behavior/politeness_analysis.py --config configs/behavior.json
    python code/behavior/politeness_analysis.py --config configs/behavior.json --sub-id sub-001

Author: Rachel C. Metzgar
Date: 2025-11-10
"""

from __future__ import annotations
import os, re, sys
import pandas as pd
import numpy as np

from utils.generic_analysis import run_generic_main
from utils.print_helpers import print_info

SCRIPT_NAME = "politeness"
HEADER = "1) Politeness Analysis — Human vs Bot, Social vs Nonsocial"

# -------------------------------
# Politeness markers
# -------------------------------
POLITENESS_MARKERS = {
    "positive_politeness": [
        r"\bthank(s| you|ful)?\b",
        r"\bappreciate\b",
        r"\b(great|wonderful|fantastic|awesome|excellent)\b",
        r"\b(hey|hello|hi)\b",
    ],
    "negative_politeness": [
        r"\bsorry\b",
        r"\bplease\b",
        r"\bcould you\b",
        r"\bwould you\b",
        r"\bmight you\b",
        r"\bif you could\b",
        r"\bperhaps\b",
        r"\bby any chance\b",
    ],
    "impoliteness": [
        r"\byou need to\b",
        r"\byou should\b",
        r"\bdo not\b",
        r"\bin fact\b",
    ]
}

# -------------------------------
# Metric computation
# -------------------------------
def compute_politeness_score(text: str) -> int:
    """Compute politeness score: +1 for polite markers, –1 for impolite markers."""
    if pd.isna(text) or not isinstance(text, str):
        return 0
    text = text.lower()
    score = 0
    for mtype, patterns in POLITENESS_MARKERS.items():
        for p in patterns:
            if re.search(p, text):
                score += 1 if mtype in ["positive_politeness", "negative_politeness"] else -1
    return score


def add_politeness_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add politeness-related columns to dataframe."""
    print_info("Computing politeness scores per utterance...")
    df["Politeness_Score"] = df["transcript_sub"].apply(compute_politeness_score)
    df["Word_Count"] = df["transcript_sub"].str.split().apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["Politeness_Rate"] = df["Politeness_Score"] / df["Word_Count"].replace(0, np.nan)
    return df


# -------------------------------
# Main
# -------------------------------
def main():
    run_generic_main(
        SCRIPT_NAME=SCRIPT_NAME,
        HEADER=HEADER,
        feature_func=add_politeness_metric,
        METRIC_COL="Politeness_Rate"
    )


if __name__ == "__main__":
    main()
