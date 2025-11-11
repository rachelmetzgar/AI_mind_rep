"""
Script name: wordcount.py
Purpose: Compute total word count per utterance or per trial using shared generic analysis.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - wordcount_by_interaction.csv
    - wordcount_subject_summary.csv
    - wordcount_subject_social_summary.csv
    - wordcount_stats.txt
    - wordcount_violinplot.png
    - wordcount_main_effect_lines.png
Usage:
    python code/behavior/wordcount.py --config configs/behavior.json
    python code/behavior/wordcount.py --config configs/behavior.json --sub-id sub-001
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

import pandas as pd
import numpy as np
from utils.generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "wordcount"
HEADER = "1) Word Count Analysis — Human vs Bot × Sociality"

# Toggle between per-utterance and per-trial aggregation
AGGREGATE = False  # True = per trial (sum), False = per utterance (raw)
SUBDIR = "per_trial" if AGGREGATE else "per_utterance"

# ============================================================
#            FEATURE COMPUTATION FUNCTION
# ============================================================

def compute_wordcount(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute word count per utterance or aggregate to per trial."""
    level = "trial-level" if aggregate else "utterance-level"
    print(f"[INFO] Computing total word counts ({level})...")

    # Count words per utterance
    df["Word_Count"] = df["transcript_sub"].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )

    # Aggregate to per-trial (sum of all utterances)
    if aggregate:
        df = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type"])["Word_Count"]
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
        lambda df: compute_wordcount(df, aggregate=AGGREGATE),
        METRIC_COL="Word_Count",
        extra_dir=SUBDIR
    )