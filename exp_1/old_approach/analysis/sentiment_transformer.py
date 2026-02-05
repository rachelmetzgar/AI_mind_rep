"""
Script name: sentiment_transformer.py
Purpose: Compute transformer-based sentiment probabilities per utterance or trial and analyze via generic analysis.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - sentiment_by_interaction.csv
    - sentiment_subject_summary.csv
    - sentiment_subject_social_summary.csv
    - sentiment_stats.txt
    - sentiment_violinplot.png
    - sentiment_main_effect_lines.png
Usage:
    python code/behavior/sentiment_transformer.py --config configs/behavior.json
    python code/behavior/sentiment_transformer.py --config configs/behavior.json --sub-id sub-001
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

import os
import pandas as pd
import numpy as np
from transformers import pipeline
from generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "sentiment_transformer"
HEADER = "1) Transformer Sentiment Analysis — Human vs Bot × Sociality"

# --- Metrics to analyze (choose any subset) ---
SENTIMENT_METRICS = ["pos", "neu", "neg", "compound"]

# --- Aggregation setting ---
AGGREGATE = True  # True = per trial, False = per utterance

# ============================================================
#            SENTIMENT MODEL SETUP
# ============================================================

print("[INFO] Loading CardiffNLP Twitter-RoBERTa sentiment model...")
classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    return_all_scores=True
)

# ============================================================
#            SENTIMENT COMPUTATION
# ============================================================

def compute_sentiment_scores(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute sentiment probabilities per utterance and optionally aggregate per trial."""
    print(f"[INFO] Computing sentiment scores ({'trial-level' if aggregate else 'utterance-level'})...")

    def get_sentiment(text: str):
        if pd.isna(text) or not isinstance(text, str) or not text.strip():
            return {"neg": np.nan, "neu": np.nan, "pos": np.nan, "compound": np.nan}
        try:
            res = classifier(text[:512])[0]  # single list of dicts
            scores = {d["label"].lower(): d["score"] for d in res}
            # mimic VADER-style compound: (pos - neg)
            compound = scores.get("positive", 0) - scores.get("negative", 0)
            return {
                "neg": scores.get("negative", np.nan),
                "neu": scores.get("neutral", np.nan),
                "pos": scores.get("positive", np.nan),
                "compound": compound
            }
        except Exception as e:
            print(f"[WARN] Sentiment computation failed: {e}")
            return {"neg": np.nan, "neu": np.nan, "pos": np.nan, "compound": np.nan}

    sent_dicts = df["transcript_sub"].apply(get_sentiment)
    sent_df = pd.DataFrame(list(sent_dicts))
    df = pd.concat([df, sent_df], axis=1)

    # Aggregate (average per trial)
    if aggregate:
        df = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type"])[["pos", "neu", "neg", "compound"]]
            .mean()
            .reset_index()
        )

    return df

# ============================================================
#            MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    for metric in SENTIMENT_METRICS:
        subdir = f"{metric}_{'per_trial' if AGGREGATE else 'per_utterance'}"
        run_generic_main(
            SCRIPT_NAME=f"{SCRIPT_NAME}_{metric}",
            HEADER=f"{HEADER} ({metric})",
            count_func=lambda df, m=metric: compute_sentiment_scores(df, aggregate=AGGREGATE),
            METRIC_COL=metric,
            extra_dir=subdir
        )
