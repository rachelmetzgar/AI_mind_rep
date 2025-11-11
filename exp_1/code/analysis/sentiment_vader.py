"""
Script name: sentiment_vader.py
Purpose: Compute VADER sentiment (compound, pos, neg, neu) per utterance or per trial and run shared generic analysis.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - sentiment_vader_by_interaction.csv
    - sentiment_vader_subject_summary.csv
    - sentiment_vader_subject_social_summary.csv
    - sentiment_vader_stats.txt
    - sentiment_vader_violinplot.png
    - sentiment_vader_main_effect_lines.png
Usage:
    python code/behavior/sentiment_vader.py --config configs/behavior.json
    python code/behavior/sentiment_vader.py --config configs/behavior.json --sub-id sub-001
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

import os
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from utils.generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "sentiment_vader"
HEADER = "VADER Sentiment Analysis — Human vs Bot × Sociality"

# Define which sentiment metrics to analyze
SENTIMENT_METRICS = ["compound", "pos", "neg", "neu"]

# Initialize VADER once
sia = SentimentIntensityAnalyzer()

# ============================================================
#            FEATURE COMPUTATION FUNCTION
# ============================================================

def compute_sentiment(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Compute the chosen VADER sentiment metric for each utterance.
    metric ∈ {'compound', 'pos', 'neg', 'neu'}
    """
    col_name = f"Sentiment_{metric.capitalize()}"
    print(f"[INFO] Computing sentiment metric: {metric}")
    df[col_name] = df["transcript_sub"].apply(
        lambda x: sia.polarity_scores(str(x)).get(metric, np.nan)
        if isinstance(x, str) else np.nan
    )
    return df

# ============================================================
#            MAIN EXECUTION (LOOP OVER METRICS)
# ============================================================

if __name__ == "__main__":
    for metric in SENTIMENT_METRICS:
        metric_col = f"Sentiment_{metric.capitalize()}"
        run_generic_main(
            SCRIPT_NAME,
            HEADER,
            lambda df, m=metric: compute_sentiment(df, m),
            METRIC_COL=metric_col,
            extra_dir=metric
        )
