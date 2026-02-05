#!/usr/bin/env python3
"""
Script name: qual_connect_analysis.py
Purpose: Analyze Quality and Connectedness ratings directly from the combined dataset
         (no per-subject files needed). Uses the shared generic analysis engine.
Inputs:
    - combined_text_data.csv (merged experiment-wide file with 'Quality', 'Connectedness',
      'topic', 'agent', 'social', and 'subject' columns)
Outputs:
    - qual_connect_by_interaction.csv
    - quality_subject_summary.csv
    - connectedness_subject_summary.csv
    - stats, violin plots, and main-effect figures for each metric
Usage:
    python code/behavior/qual_connect_analysis.py --config configs/behavior.json
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

from __future__ import annotations
import pandas as pd
from utils.generic_analysis import run_generic_main

# ============================================================
#              CONFIGURATION
# ============================================================

SCRIPT_NAME = "qual_connect"
HEADER = "1) Quality & Connectedness Analysis — Human vs Bot × Sociality"
METRICS = ["Quality", "Connectedness"]

# ============================================================
#              FEATURE FUNCTION
# ============================================================

def clean_quality_connect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean combined dataset for Quality & Connectedness analysis:
      - Keeps only rows with valid (non-NaN) Quality or Connectedness values.
      - Drops rows missing required condition/topic fields.
    """
    required = ["Subject", "Condition", "Social_Type", "topic"]
    metric_cols = ["Quality", "Connectedness"]

    # Drop rows missing identifiers
    df = df.dropna(subset=required, how="any")

    # Drop rows where both metrics are NaN
    df = df.dropna(subset=metric_cols, how="all")

    return df


# ============================================================
#              MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Loop over both metrics
    for metric in METRICS:
        subdir = metric.lower()
        run_generic_main(
            SCRIPT_NAME=f"{SCRIPT_NAME}_{metric.lower()}",
            HEADER=f"{HEADER} ({metric})",
            feature_func=clean_quality_connect,
            METRIC_COL=metric,
            extra_dir=subdir
        )
