#!/usr/bin/env python3
"""
Extract 23 linguistic measures from combined_text_data.csv.

Produces trial-level and utterance-level CSVs with all computed features.

Usage:
    python 1_extract_features.py --version balanced_gpt --model llama2_13b_chat

Author: Rachel C. Metzgar
"""

import os
import sys
import re
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.config import (
    parse_version_model, data_dir, results_dir,
    TOPIC_SOCIAL_MAP, HUMAN_IDS,
)
from code.utils.subject_utils import standardize_sub_id, find_old_id
from code.utils.hedges_demir import (
    DEMIR_ALL_HEDGES, DEMIR_NOUNS, DEMIR_ADJECTIVES, DEMIR_ADVERBS,
    DEMIR_VERBS, DEMIR_QUANTIFIERS, DEMIR_MODALS,
)
from code.utils.discourse_markers_fung import (
    FUNG_INTERPERSONAL, FUNG_REFERENTIAL, FUNG_STRUCTURAL, FUNG_COGNITIVE,
    FUNG_ALL_23_MARKERS,
)
from code.utils.misc_text_markers import (
    LIWC_NONFLUENCIES, LIWC_FILLERS,
    TOM_PHRASES, POLITE_POSITIVE, POLITE_NEGATIVE, IMPOLITE, LIKE_MARKER,
)
from code.utils.stats_helpers import (
    run_within_experiment_anova, run_cross_experiment_analysis,
)


def _count_words(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def count_patterns(text, patterns):
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def get_social_classification(df, topic_social_map):
    df = df.copy()
    df['topic_lower'] = df['topic'].astype(str).str.strip().str.lower()
    df['social'] = df['topic_lower'].map(topic_social_map)
    df['social'] = df['social'].fillna(0).astype(int)
    df['social_type'] = df['social'].map({1: 'social', 0: 'nonsocial'})
    return df


def load_llm_data(filepath, topic_social_map):
    print(f"[INFO] Loading LLM data from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"LLM data file not found: {filepath}")
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip')
    except TypeError:
        df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=True)
    df.columns = df.columns.str.strip().str.lower()
    if 'transcript_sub' not in df.columns:
        raise ValueError(f"Missing 'transcript_sub' column in {filepath}")
    df['condition'] = df['agent'].astype(str).str.extract(r'(hum|bot)', expand=False)
    df['subject'] = df['subject'].astype(str).apply(standardize_sub_id)
    df['topic'] = df['topic'].astype(str).str.strip()
    df = get_social_classification(df, topic_social_map)
    df['experiment'] = 'LLM'
    print(f"  Loaded {len(df)} rows, {df['subject'].nunique()} subjects")
    return df


def compute_all_metrics(df):
    print("[INFO] Computing linguistic metrics at utterance level...")
    df = df.copy()
    df['word_count'] = df['transcript_sub'].apply(_count_words)
    df['question_count'] = df['transcript_sub'].apply(
        lambda x: str(x).count('?') if isinstance(x, str) else 0
    )

    # Demir hedges
    for cat, patterns in [
        ('modal', DEMIR_MODALS), ('verb', DEMIR_VERBS), ('adverb', DEMIR_ADVERBS),
        ('adjective', DEMIR_ADJECTIVES), ('quantifier', DEMIR_QUANTIFIERS), ('noun', DEMIR_NOUNS),
    ]:
        df[f'demir_{cat}_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, patterns))
    df['demir_total_count'] = sum(df[f'demir_{c}_count'] for c in ['modal', 'verb', 'adverb', 'adjective', 'quantifier', 'noun'])
    for cat in ['modal', 'verb', 'adverb', 'adjective', 'quantifier', 'noun', 'total']:
        df[f'demir_{cat}_rate'] = df[f'demir_{cat}_count'] / df['word_count'].replace(0, np.nan)

    # Fung discourse markers
    for cat, patterns in [
        ('interpersonal', FUNG_INTERPERSONAL), ('referential', FUNG_REFERENTIAL),
        ('structural', FUNG_STRUCTURAL), ('cognitive', FUNG_COGNITIVE),
    ]:
        df[f'fung_{cat}_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, patterns))
    df['fung_total_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, FUNG_ALL_23_MARKERS))
    for cat in ['interpersonal', 'referential', 'structural', 'cognitive', 'total']:
        df[f'fung_{cat}_rate'] = df[f'fung_{cat}_count'] / df['word_count'].replace(0, np.nan)

    # LIWC disfluencies
    df['nonfluency_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, LIWC_NONFLUENCIES))
    df['nonfluency_rate'] = df['nonfluency_count'] / df['word_count'].replace(0, np.nan)
    df['liwc_filler_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, LIWC_FILLERS))
    df['liwc_filler_rate'] = df['liwc_filler_count'] / df['word_count'].replace(0, np.nan)
    df['disfluency_count'] = df['nonfluency_count'] + df['liwc_filler_count']
    df['disfluency_rate'] = df['disfluency_count'] / df['word_count'].replace(0, np.nan)

    # Other markers
    df['tom_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, TOM_PHRASES))
    df['tom_rate'] = df['tom_count'] / df['word_count'].replace(0, np.nan)
    df['polite_pos'] = df['transcript_sub'].apply(lambda x: count_patterns(x, POLITE_POSITIVE))
    df['polite_neg'] = df['transcript_sub'].apply(lambda x: count_patterns(x, POLITE_NEGATIVE))
    df['impolite'] = df['transcript_sub'].apply(lambda x: count_patterns(x, IMPOLITE))
    df['politeness_score'] = df['polite_pos'] + df['polite_neg'] - df['impolite']
    df['politeness_rate'] = df['politeness_score'] / df['word_count'].replace(0, np.nan)
    df['like_count'] = df['transcript_sub'].apply(
        lambda x: len(re.findall(LIKE_MARKER, x.lower())) if isinstance(x, str) else 0
    )
    df['like_rate'] = df['like_count'] / df['word_count'].replace(0, np.nan)

    # VADER sentiment
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        for col_name, key in [('sentiment', 'compound'), ('sentiment_pos', 'pos'),
                               ('sentiment_neg', 'neg'), ('sentiment_neu', 'neu')]:
            df[col_name] = df['transcript_sub'].apply(
                lambda x: sia.polarity_scores(str(x)).get(key, np.nan) if isinstance(x, str) else np.nan
            )
        print("  [OK] Sentiment analysis computed (VADER)")
    except (ImportError, LookupError):
        for col_name in ['sentiment', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu']:
            df[col_name] = np.nan
        print("  [SKIP] VADER sentiment analysis - nltk not available")

    return df


def aggregate_to_trial_level(df):
    print("[INFO] Aggregating utterances to trial level...")
    count_cols = [c for c in [
        'word_count', 'question_count', 'tom_count',
        'polite_pos', 'polite_neg', 'impolite', 'politeness_score',
        'nonfluency_count', 'liwc_filler_count', 'disfluency_count', 'like_count',
        'fung_interpersonal_count', 'fung_referential_count',
        'fung_structural_count', 'fung_cognitive_count', 'fung_total_count',
        'demir_modal_count', 'demir_verb_count', 'demir_adverb_count',
        'demir_adjective_count', 'demir_quantifier_count', 'demir_noun_count',
        'demir_total_count',
    ] if c in df.columns]

    groupby_cols = [c for c in ['experiment', 'subject', 'condition', 'topic', 'social_type', 'social']
                    if c in df.columns]
    agg_dict = {col: 'sum' for col in count_cols}
    for col in ['sentiment', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'quality', 'connectedness']:
        if col in df.columns:
            agg_dict[col] = 'mean'

    trial_df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)

    rate_pairs = [
        ('tom_rate', 'tom_count'), ('politeness_rate', 'politeness_score'),
        ('nonfluency_rate', 'nonfluency_count'), ('liwc_filler_rate', 'liwc_filler_count'),
        ('disfluency_rate', 'disfluency_count'), ('like_rate', 'like_count'),
        ('fung_interpersonal_rate', 'fung_interpersonal_count'),
        ('fung_referential_rate', 'fung_referential_count'),
        ('fung_structural_rate', 'fung_structural_count'),
        ('fung_cognitive_rate', 'fung_cognitive_count'),
        ('fung_total_rate', 'fung_total_count'),
        ('demir_modal_rate', 'demir_modal_count'), ('demir_verb_rate', 'demir_verb_count'),
        ('demir_adverb_rate', 'demir_adverb_count'), ('demir_adjective_rate', 'demir_adjective_count'),
        ('demir_quantifier_rate', 'demir_quantifier_count'), ('demir_noun_rate', 'demir_noun_count'),
        ('demir_total_rate', 'demir_total_count'),
    ]
    for rate_col, count_col in rate_pairs:
        if count_col in trial_df.columns and 'word_count' in trial_df.columns:
            trial_df[rate_col] = trial_df[count_col] / trial_df['word_count'].replace(0, np.nan)

    print(f"  Aggregated {len(df)} utterances -> {len(trial_df)} trials")
    return trial_df


def main():
    parser = argparse.ArgumentParser(description="Extract linguistic features")
    args = parse_version_model(parser)

    dd = data_dir()
    rd = results_dir()

    llm_path = str(dd / "combined_text_data.csv")
    llm_df = load_llm_data(llm_path, TOPIC_SOCIAL_MAP)
    llm_df = compute_all_metrics(llm_df)

    all_metrics = [
        'word_count', 'question_count',
        'demir_modal_rate', 'demir_verb_rate', 'demir_adverb_rate',
        'demir_adjective_rate', 'demir_quantifier_rate', 'demir_noun_rate', 'demir_total_rate',
        'fung_interpersonal_rate', 'fung_referential_rate', 'fung_structural_rate',
        'fung_cognitive_rate', 'fung_total_rate',
        'nonfluency_rate', 'liwc_filler_rate', 'disfluency_rate',
        'like_rate', 'tom_rate', 'politeness_rate',
    ]
    if 'sentiment' in llm_df.columns and llm_df['sentiment'].notna().any():
        all_metrics.append('sentiment')
    if 'quality' in llm_df.columns and llm_df['quality'].notna().any():
        all_metrics.append('quality')
    if 'connectedness' in llm_df.columns and llm_df['connectedness'].notna().any():
        all_metrics.append('connectedness')

    # Save utterance-level data
    utt_path = str(dd / "combined_utterance_level_data.csv")
    llm_df.to_csv(utt_path, index=False)
    print(f"[SAVED] {utt_path}")

    # Trial-level
    trial_df = aggregate_to_trial_level(llm_df)
    trial_path = str(dd / "combined_trial_level_data.csv")
    trial_df.to_csv(trial_path, index=False)
    print(f"[SAVED] {trial_path}")

    # Subject-level aggregations
    trial_subject = trial_df.groupby(['experiment', 'subject', 'condition'])[all_metrics].mean().reset_index()
    trial_subject.to_csv(str(dd / "per_trial_subject_condition.csv"), index=False)

    trial_social = trial_df.groupby(['experiment', 'subject', 'condition', 'social_type'])[all_metrics].mean().reset_index()
    trial_social.to_csv(str(dd / "per_trial_subject_condition_social.csv"), index=False)

    # Utterance-level aggregations
    utt_subject = llm_df.groupby(['experiment', 'subject', 'condition'])[all_metrics].mean().reset_index()
    utt_subject.to_csv(str(dd / "per_utterance_subject_condition.csv"), index=False)

    utt_social = llm_df.groupby(['experiment', 'subject', 'condition', 'social_type'])[all_metrics].mean().reset_index()
    utt_social.to_csv(str(dd / "per_utterance_subject_condition_social.csv"), index=False)

    print(f"\n[DONE] Feature extraction complete. Results saved to {dd}/")


if __name__ == "__main__":
    main()
