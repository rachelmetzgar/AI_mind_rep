"""
Script name: empath_tom.py
Purpose: Compute Empath-based Theory of Mind (ToM) scores per utterance or per trial,
         and analyze them using the shared generic analysis framework.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - empath_tom_by_interaction.csv
    - empath_tom_subject_summary.csv
    - empath_tom_subject_social_summary.csv
    - empath_tom_stats.txt
    - empath_tom_violinplot.png
    - empath_tom_main_effect_lines.png
Usage:
    python code/behavior/empath_tom.py --config configs/behavior.json
    python code/behavior/empath_tom.py --config configs/behavior.json --sub-id sub-001

Author: Rachel C. Metzgar
Date: 2025-11-10
"""

import os
import pandas as pd
import numpy as np
from empath import Empath
from utils.generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "empath_tom"
HEADER = "1) Empath ToM Analysis — Human vs Bot × Sociality"

# Toggle this manually
AGGREGATE = False   # True = per trial (aggregate), False = per utterance

# Auto output subdirectory
SUBDIR = "per_trial" if AGGREGATE else "per_utterance"

# Custom ToM category for Empath
CUSTOM_TOM_WORDS = ["think", "know", "understand", "feel", "believe", "guess", "wonder", "remember"]

# ============================================================
#            FEATURE COMPUTATION FUNCTION
# ============================================================

def compute_empath_tom(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute Empath-based ToM category scores per utterance or aggregate to per-trial."""
    level = "trial-level" if aggregate else "utterance-level"
    print(f"[INFO] Computing Empath ToM ({level})...")

    # --- Initialize Empath ---
    lex = Empath()
    lex.create_category("thinking_custom", CUSTOM_TOM_WORDS)

    # --- Compute Empath scores per utterance ---
    df["ToM_score"] = df["transcript_sub"].astype(str).apply(
        lambda text: lex.analyze(text, categories=["thinking_custom"])["thinking_custom"]
        if isinstance(text, str) else np.nan
    )

    # --- Optionally aggregate per trial ---
    if aggregate:
        df = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type"])["ToM_score"]
            .mean()
            .reset_index()
        )

    return df

# ============================================================
#            MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    run_generic_main(
        SCRIPT_NAME,
        f"{HEADER} ({SUBDIR})",
        lambda df: compute_empath_tom(df, aggregate=AGGREGATE),
        METRIC_COL="ToM_score",
        extra_dir=SUBDIR
    )
