"""
Script name: tom_words.py
Purpose: Compute Theory of Mind (ToM) word counts per utterance or per trial and run shared generic analysis.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - tom_words_by_interaction.csv
    - tom_words_subject_summary.csv
    - tom_words_subject_social_summary.csv
    - tom_words_stats.txt
    - tom_words_violinplot.png
    - tom_words_main_effect_lines.png
Usage:
    python code/behavior/tom_words.py --config configs/behavior.json
    python code/behavior/tom_words.py --config configs/behavior.json --sub-id sub-001
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

SCRIPT_NAME = "tom_words"
HEADER = "1) ToM Word Analysis — Human vs Bot × Sociality"

# Toggle this manually
AGGREGATE = True   # True = per trial, False = per utterance

# Auto output subdirectory
SUBDIR = "per_trial" if AGGREGATE else "per_utterance"

# List of ToM-related phrases
TOM_PHRASES = [
    "you think", "you believe", "you know", "you feel", "you understand",
    "you guess", "you imagine", "you wonder", "you consider", "you expect",
    "you hope", "you assume", "you realize", "you remember", "you forget"
]

# ============================================================
#            FEATURE COMPUTATION FUNCTION
# ============================================================

def compute_tom_words(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute ToM word counts per utterance or aggregate to per-trial."""
    level = "trial-level" if aggregate else "utterance-level"
    print(f"[INFO] Computing ToM word counts ({level})...")

    # --- Count ToM phrases in each utterance ---
    def count_tom_phrases(text: str) -> int:
        if pd.isna(text) or not isinstance(text, str):
            return 0
        text = text.lower()
        return sum(len(re.findall(rf"\b{re.escape(phrase)}\b", text)) for phrase in TOM_PHRASES)

    df["ToM_Count"] = df["transcript_sub"].apply(count_tom_phrases)

    # --- Aggregate per trial if requested ---
    if aggregate:
        df = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type"])["ToM_Count"]
            .sum()
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
        lambda df: compute_tom_words(df, aggregate=AGGREGATE),
        METRIC_COL="ToM_Count",
        extra_dir=SUBDIR
    )
