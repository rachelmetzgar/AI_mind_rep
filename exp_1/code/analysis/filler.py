"""
Script name: filler.py
Purpose: Compute total filler word counts per utterance or per trial 
         and run shared generic analysis.
"""

import re
import pandas as pd
import numpy as np
from utils.generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "filler"
HEADER = "1) Filler Word Analysis — Human vs Bot × Sociality"

# Toggle manually
AGGREGATE = False   # True = per trial (aggregate), False = per utterance

# Auto output subdirectory
SUBDIR = "per_trial" if AGGREGATE else "per_utterance"

# Filler markers
FILLER_MARKERS = [
    r"\bum\b", r"\buh\b", r"\ber\b", r"\bah\b",
    r"\blike\b", r"\byou know\b", r"\bi mean\b",
    r"\bwell\b", r"\bbasically\b", r"\bactually\b",
    r"\bright\b", r"\bokay\b", r"\bso\b"
]

# ============================================================
#            FEATURE COMPUTATION FUNCTION
# ============================================================

def compute_fillers(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute total filler word counts per utterance or aggregate to per-trial."""
    level = "trial-level" if aggregate else "utterance-level"
    print(f"[INFO] Computing filler counts ({level})...")

    # --- Count fillers and words per utterance ---
    df["Filler_Count"] = df["transcript_sub"].apply(
        lambda text: sum(len(re.findall(pattern, str(text).lower())) for pattern in FILLER_MARKERS)
        if isinstance(text, str) else 0
    )
    df["Word_Count"] = df["transcript_sub"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )
    df["Filler_Rate"] = df["Filler_Count"] / df["Word_Count"].replace(0, np.nan)

    # --- Aggregate to per-trial (sum of counts) ---
    if aggregate:
        df = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type"])["Filler_Count"]
            .sum()
            .reset_index()
        )
        # keep filler count as the per-trial metric
        df["Filler_Rate"] = df["Filler_Count"].replace(0, np.nan)

    return df

# ============================================================
#            MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    run_generic_main(
        SCRIPT_NAME,
        f"{HEADER} ({SUBDIR})",
        lambda df: compute_fillers(df, aggregate=AGGREGATE),
        METRIC_COL="Filler_Rate",
        extra_dir=SUBDIR
    )
