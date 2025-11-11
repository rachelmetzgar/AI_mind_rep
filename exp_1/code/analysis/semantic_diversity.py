#!/usr/bin/env python3
"""
Script name: semantic_diversity.py
Purpose: Compute semantic diversity using sentence embeddings and analyze via shared generic analysis.
Inputs:
    - combined_text_data.csv (with 'subject', 'agent', 'topic', 'transcript_sub', 'conversation_id', 'turn_index')
    - topics.csv (with 'topic' and 'social' coding)
Outputs:
    - semantic_diversity_by_interaction.csv
    - semantic_diversity_subject_summary.csv
    - semantic_diversity_subject_social_summary.csv
    - semantic_diversity_stats.txt
    - semantic_diversity_violinplot.png
    - semantic_diversity_main_effect_lines.png
Usage:
    python code/behavior/semantic_diversity.py --config configs/behavior.json
    python code/behavior/semantic_diversity.py --config configs/behavior.json --sub-id sub-001
Author: Rachel C. Metzgar
Date: 2025-11-10
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from generic_analysis import run_generic_main

# ============================================================
#            ANALYSIS-SPECIFIC CONFIGURATION
# ============================================================

SCRIPT_NAME = "semantic_diversity"
HEADER = "1) Semantic Diversity Analysis — Human vs Bot × Sociality"

# --- Embedding model ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

# --- Aggregation level toggle ---
AGGREGATE = True  # True = per trial (conversation-level), False = per utterance
SUBDIR = "per_trial" if AGGREGATE else "per_utterance"

# ============================================================
#            SEMANTIC DIVERSITY COMPUTATION
# ============================================================

def compute_semantic_diversity(df: pd.DataFrame, aggregate: bool = True) -> pd.DataFrame:
    """Compute mean cosine distance between consecutive utterances per conversation or per utterance."""
    print(f"[INFO] Computing semantic diversity ({'trial-level' if aggregate else 'utterance-level'})...")

    # Skip missing transcripts
    df = df.dropna(subset=["transcript_sub"])
    df["transcript_sub"] = df["transcript_sub"].astype(str)

    # Ensure conversation/turn order columns exist
    if not {"conversation_id", "turn_index"}.issubset(df.columns):
        raise ValueError("Missing 'conversation_id' or 'turn_index' columns required for semantic diversity.")

    # Sort and compute pairwise distances
    def per_conversation_diversity(conv_df):
        texts = conv_df.sort_values("turn_index")["transcript_sub"].tolist()
        if len(texts) < 2:
            return np.nan
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        dists = [
            cosine_distances([embeddings[i]], [embeddings[i + 1]])[0][0]
            for i in range(len(embeddings) - 1)
        ]
        return np.mean(dists)

    if aggregate:
        # Per conversation/trial: average across turns within each conversation
        scores = (
            df.groupby(["Subject", "Condition", "topic", "Social_Type", "conversation_id"])
            .apply(per_conversation_diversity)
            .reset_index(name="SemanticDiversity")
        )
        # Then aggregate across conversations for each trial
        scores = (
            scores.groupby(["Subject", "Condition", "topic", "Social_Type"])["SemanticDiversity"]
            .mean()
            .reset_index()
        )
    else:
        # Per utterance: compute distances between consecutive turns, assign to each utterance
        distances = []
        for _, conv_df in df.groupby(["Subject", "Condition", "conversation_id"]):
            texts = conv_df.sort_values("turn_index")["transcript_sub"].tolist()
            if len(texts) < 2:
                distances.extend([np.nan] * len(texts))
                continue
            embeddings = embedder.encode(texts, convert_to_numpy=True)
            conv_dists = [
                cosine_distances([embeddings[i]], [embeddings[i + 1]])[0][0]
                for i in range(len(embeddings) - 1)
            ]
            conv_dists.append(np.nan)
            distances.extend(conv_dists)
        df["SemanticDiversity"] = distances
        scores = df.copy()

    return scores

# ============================================================
#            MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    run_generic_main(
        SCRIPT_NAME,
        f"{HEADER} ({SUBDIR})",
        lambda df: compute_semantic_diversity(df, aggregate=AGGREGATE),
        METRIC_COL="SemanticDiversity",
        extra_dir=SUBDIR
    )
