"""
Script name: hedging.py
Purpose: Compute hedging rate per utterance or per trial and run shared generic analysis.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - hedging_by_interaction.csv
    - hedging_subject_summary.csv
    - hedging_subject_social_summary.csv
    - hedging_stats.txt
    - hedging_violinplot.png
    - hedging_main_effect_lines.png
Usage:
    python code/behavior/hedging.py --config configs/behavior.json
    python code/behavior/hedging.py --config configs/behavior.json --sub-id sub-001
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

import re
import pandas as pd
import numpy as np
from utils.generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "hedging"
HEADER = "1) Hedging Analysis — Human vs Bot × Sociality"

# Toggle this manually
AGGREGATE = False   # True = per trial, False = per utterance

# Optional: output subdirectory name (auto from toggle)
SUBDIR = "per_trial" if AGGREGATE else "per_utterance"

# Hedge markers
HEDGE_MARKERS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bmight\b",
    r"\bcould be\b", r"\bit seems\b", r"\bi think\b",
    r"\bin a way\b", r"\bsort of\b", r"\bkind of\b",
    r"\bmore or less\b", r"\broughly\b", r"\btends to\b"
]

# ============================================================
#            FEATURE COMPUTATION FUNCTION
# ============================================================

def compute_hedging(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute hedge rate per utterance or aggregate to per-trial."""
    level = "trial-level" if aggregate else "utterance-level"
    print(f"[INFO] Computing hedging ({level})...")

    # Count markers and words
    df["Hedge_Count"] = df["transcript_sub"].apply(
        lambda text: sum(len(re.findall(pattern, str(text).lower())) for pattern in HEDGE_MARKERS)
        if isinstance(text, str) else 0
    )
    df["Word_Count"] = df["transcript_sub"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )
    df["Hedge_Rate"] = df["Hedge_Count"] / df["Word_Count"].replace(0, np.nan)

    if aggregate:
        df = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type"])["Hedge_Count"]
            .sum()
            .reset_index()
        )
        df["Hedge_Rate"] = df["Hedge_Count"].replace(0, np.nan)

    return df

# ============================================================
#            MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    run_generic_main(
        SCRIPT_NAME,
        f"{HEADER} ({SUBDIR})",
        lambda df: compute_hedging(df, aggregate=AGGREGATE),
        METRIC_COL="Hedge_Rate",
        extra_dir=SUBDIR
    )
