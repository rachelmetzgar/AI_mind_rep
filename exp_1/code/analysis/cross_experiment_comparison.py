#!/usr/bin/env python3
"""
Script name: cross_experiment_comparison.py
Purpose: Compare behavioral measures between Human and LLM experiments.

DISFLUENCY METRICS (LIWC2007):
- nonfluency_rate: LIWC nonfluencies / word_count
- liwc_filler_rate: LIWC fillers / word_count
- disfluency_rate: Combined nonfluencies + fillers / word_count

DISCOURSE MARKERS (Fung & Carter 2007):
# Source: Applied Linguistics 28(3): 410-439
# Based on Table 1 (functional paradigm) and Table 2 (frequency analysis)
- fung_interpersonal_rate, fung_referential_rate, fung_structural_rate,
  fung_cognitive_rate, fung_total_rate

OTHER METRICS:
- tom_rate: Theory of Mind phrases
- politeness_rate: Politeness markers
- like_rate: Quotative/discourse 'like'

Usage:
    python cross_experiment_comparison.py

Author: Rachel C. Metzgar
Date: 2025
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, f as f_dist
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')
from utils.cli_helpers import parse_and_load_config
from utils.subject_utils import get_sub_id_map, find_old_id, standardize_sub_id
from utils.globals import HUMAN_DIR, TOPICS_PATH, HUMAN_FILE_PATTERN
from utils.hedges_demir import (
    DEMIR_ALL_HEDGES, DEMIR_NOUNS, DEMIR_ADJECTIVES, DEMIR_ADVERBS,
    DEMIR_VERBS, DEMIR_QUANTIFIERS, DEMIR_MODALS
)
from utils.discourse_markers_fung import (
    FUNG_INTERPERSONAL, FUNG_REFERENTIAL, FUNG_STRUCTURAL, FUNG_COGNITIVE,
    FUNG_ALL_23_MARKERS
)
from utils.misc_text_markers import (
    LIWC_NONFLUENCIES, LIWC_FILLERS, LIWC_DISFLUENCIES,
    TOM_PHRASES, POLITE_POSITIVE, POLITE_NEGATIVE, IMPOLITE, LIKE_MARKER
)
from utils.stats_helpers import (
    run_within_experiment_anova,
    run_cross_experiment_analysis,
    run_simple_effects
)

# ============================================================
#                    WORD COUNT FUNCTION
# ============================================================

def _count_words(text: str) -> int:
    """Count words in text - MATCHES ORIGINAL wordcount.py implementation."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


# ============================================================
#                    DATA LOADING
# ============================================================

def load_topics_file(topics_path):
    """Load topics.csv if available, otherwise use TOPIC_SOCIAL_MAP."""
    if os.path.exists(topics_path):
        print(f"[INFO] Loading topics from {topics_path}")
        topic_df = pd.read_csv(topics_path)
        topic_df.columns = topic_df.columns.str.strip().str.lower()
        topic_df['topic'] = topic_df['topic'].astype(str).str.strip().str.lower()
        return topic_df
    else:
        print(f"[WARN] Topics file not found at {topics_path}, using built-in TOPIC_SOCIAL_MAP")
        return None


def get_social_classification(df, topic_df=None):
    """Add social/nonsocial classification to dataframe."""
    df = df.copy()
    df['topic_lower'] = df['topic'].astype(str).str.strip().str.lower()
    
    if topic_df is not None:
        topic_df = topic_df.copy()
        topic_df['topic'] = topic_df['topic'].astype(str).str.strip().str.lower()
        df = df.merge(topic_df[['topic', 'social']], left_on='topic_lower', right_on='topic', 
                      how='left', suffixes=('', '_from_file'))
        if 'social_from_file' in df.columns:
            df['social'] = df['social_from_file'].combine_first(df.get('social'))
            df = df.drop(columns=['social_from_file', 'topic_from_file'], errors='ignore')
    
    if 'social' not in df.columns:
        df['social'] = np.nan
    
    unmapped = df['social'].isna()
    if unmapped.any():
        df.loc[unmapped, 'social'] = df.loc[unmapped, 'topic_lower'].map(TOPIC_SOCIAL_MAP)
    
    still_unmapped = df['social'].isna()
    if still_unmapped.any():
        for idx in df[still_unmapped].index:
            topic = df.loc[idx, 'topic_lower']
            for key, val in TOPIC_SOCIAL_MAP.items():
                if key in topic or topic in key:
                    df.loc[idx, 'social'] = val
                    break
    
    final_unmapped = df['social'].isna()
    if final_unmapped.any():
        unmapped_topics = df.loc[final_unmapped, 'topic_lower'].unique()
        print(f"  [WARN] {len(unmapped_topics)} topics could not be classified (defaulting to nonsocial)")
    
    df['social'] = df['social'].fillna(0).astype(int)
    df['social_type'] = df['social'].map({1: 'social', 0: 'nonsocial'})
    
    social_counts = df['social_type'].value_counts()
    print(f"  Topic classification: {social_counts.get('social', 0)} social, {social_counts.get('nonsocial', 0)} nonsocial utterances")
    
    return df


def load_human_data(subject_dir, subject_ids, file_pattern, topic_df=None):
    """Load human experiment data from individual per-subject CSVs."""
    print(f"[INFO] Loading Human data from individual subject files...")
    print(f"  Directory: {subject_dir}")
    print(f"  Subjects: {len(subject_ids)}")
    
    all_trials = []
    loaded_subjects = []
    
    for sub_id in subject_ids:
        old_id = find_old_id(sub_id)
        old_ids = [old_id] if old_id else []
        
        possible_paths = []
        if file_pattern:
            possible_paths.append(os.path.join(subject_dir, file_pattern.format(sub_id=sub_id)))
        possible_paths.extend([
            os.path.join(subject_dir, f"{sub_id}.csv"),
            os.path.join(subject_dir, f"{sub_id}_transcript.csv"),
            os.path.join(subject_dir, f"{sub_id}_data.csv"),
        ])
        
        for oid in old_ids:
            if file_pattern:
                possible_paths.append(os.path.join(subject_dir, file_pattern.format(sub_id=oid)))
            possible_paths.extend([
                os.path.join(subject_dir, f"{oid}.csv"),
                os.path.join(subject_dir, f"{oid}_transcript.csv"),
                os.path.join(subject_dir, f"{oid}_data.csv"),
                os.path.join(subject_dir, f"{oid}_combined.csv"),
            ])
        
        for id_variant in [sub_id] + old_ids:
            subdir = os.path.join(subject_dir, id_variant)
            if os.path.isdir(subdir):
                csvs_in_subdir = glob.glob(os.path.join(subdir, "*.csv"))
                possible_paths.extend(csvs_in_subdir)
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if not found_path:
            print(f"  [WARN] Missing CSV for {sub_id} (tried old IDs: {old_ids})")
            continue
        
        try:
            df = pd.read_csv(found_path)
            print(f"  [OK] Loaded {sub_id} from {os.path.basename(found_path)}")
        except Exception as e:
            print(f"  [WARN] Error reading {found_path}: {e}")
            continue
        
        df.columns = df.columns.str.strip()
        
        if 'Quality' in df.columns:
            df['quality'] = pd.to_numeric(df['Quality'], errors='coerce')
        elif 'quality' in df.columns:
            df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
        
        if 'Connectedness' in df.columns:
            df['connectedness'] = pd.to_numeric(df['Connectedness'], errors='coerce')
        elif 'connectedness' in df.columns:
            df['connectedness'] = pd.to_numeric(df['connectedness'], errors='coerce')
        
        df['condition'] = df['agent'].astype(str).str.extract(r'(hum|bot)', expand=False)
        df['subject'] = sub_id
        df['topic'] = df['topic'].astype(str).str.strip()
        
        all_trials.append(df)
        loaded_subjects.append(sub_id)
    
    if not all_trials:
        raise ValueError("No valid data found for Human experiment")
    
    df_combined = pd.concat(all_trials, ignore_index=True)
    df_combined = get_social_classification(df_combined, topic_df)
    df_combined['experiment'] = 'Human'
    
    print(f"  Loaded {len(df_combined)} rows (utterances) from {len(loaded_subjects)} subjects")
    print(f"  Conditions: {df_combined['condition'].value_counts().to_dict()}")
    
    if 'quality' in df_combined.columns and df_combined['quality'].notna().any():
        print(f"  Quality ratings: {df_combined['quality'].notna().sum()} non-null values")
    if 'connectedness' in df_combined.columns and df_combined['connectedness'].notna().any():
        print(f"  Connectedness ratings: {df_combined['connectedness'].notna().sum()} non-null values")
    
    return df_combined


def load_llm_data(filepath, topic_df=None):
    """Load LLM experiment data from combined CSV."""
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
    
    df = get_social_classification(df, topic_df)
    df['experiment'] = 'LLM'
    
    print(f"  Loaded {len(df)} rows (utterances), {df['subject'].nunique()} subjects")
    print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
    
    return df


# ============================================================
#                    FEATURE COMPUTATION
# ============================================================

def count_patterns(text, patterns):
    """Count occurrences of regex patterns in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def compute_all_metrics(df):
    """
    Compute all linguistic metrics for each UTTERANCE (row).
    
    v7: Uses Demir (2018) hedge taxonomy and LIWC disfluency markers
    """
    print("[INFO] Computing linguistic metrics at utterance level...")
    df = df.copy()
    
    # Word count
    df['word_count'] = df['transcript_sub'].apply(_count_words)
    
    # Question count
    df['question_count'] = df['transcript_sub'].apply(
        lambda x: str(x).count('?') if isinstance(x, str) else 0
    )
    
    # ============================================================
    # DEMIR (2018) HEDGE TAXONOMY
    # ============================================================
    
    print("  Computing Demir (2018) hedge taxonomy categories...")
    
    # Count each hedge category
    df['demir_modal_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, DEMIR_MODALS)
    )
    df['demir_verb_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, DEMIR_VERBS)
    )
    df['demir_adverb_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, DEMIR_ADVERBS)
    )
    df['demir_adjective_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, DEMIR_ADJECTIVES)
    )
    df['demir_quantifier_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, DEMIR_QUANTIFIERS)
    )
    df['demir_noun_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, DEMIR_NOUNS)
    )
    
    # Total Demir hedges
    df['demir_total_count'] = (
        df['demir_modal_count'] + 
        df['demir_verb_count'] + 
        df['demir_adverb_count'] + 
        df['demir_adjective_count'] + 
        df['demir_quantifier_count'] + 
        df['demir_noun_count']
    )
    
    # Compute rates (normalized by word count)
    df['demir_modal_rate'] = df['demir_modal_count'] / df['word_count'].replace(0, np.nan)
    df['demir_verb_rate'] = df['demir_verb_count'] / df['word_count'].replace(0, np.nan)
    df['demir_adverb_rate'] = df['demir_adverb_count'] / df['word_count'].replace(0, np.nan)
    df['demir_adjective_rate'] = df['demir_adjective_count'] / df['word_count'].replace(0, np.nan)
    df['demir_quantifier_rate'] = df['demir_quantifier_count'] / df['word_count'].replace(0, np.nan)
    df['demir_noun_rate'] = df['demir_noun_count'] / df['word_count'].replace(0, np.nan)
    df['demir_total_rate'] = df['demir_total_count'] / df['word_count'].replace(0, np.nan)
    
    print(f"  [OK] Demir (2018) hedge taxonomy computed:")
    print(f"       Modal auxiliaries:  mean = {df['demir_modal_count'].mean():.3f} per utterance")
    print(f"       Epistemic verbs:    mean = {df['demir_verb_count'].mean():.3f} per utterance")
    print(f"       Epistemic adverbs:  mean = {df['demir_adverb_count'].mean():.3f} per utterance")
    print(f"       Epistemic adj:      mean = {df['demir_adjective_count'].mean():.3f} per utterance")
    print(f"       Quantifiers:        mean = {df['demir_quantifier_count'].mean():.3f} per utterance")
    print(f"       Epistemic nouns:    mean = {df['demir_noun_count'].mean():.3f} per utterance")
    print(f"       TOTAL:              mean = {df['demir_total_count'].mean():.3f} per utterance")
    
    # ============================================================
    # FUNG & CARTER (2007) DISCOURSE MARKERS
    # ============================================================
    
    df['fung_interpersonal_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FUNG_INTERPERSONAL)
    )
    df['fung_referential_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FUNG_REFERENTIAL)
    )
    df['fung_structural_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FUNG_STRUCTURAL)
    )
    df['fung_cognitive_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FUNG_COGNITIVE)
    )
    df['fung_total_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FUNG_ALL_23_MARKERS)
    )
    
    df['fung_interpersonal_rate'] = df['fung_interpersonal_count'] / df['word_count'].replace(0, np.nan)
    df['fung_referential_rate'] = df['fung_referential_count'] / df['word_count'].replace(0, np.nan)
    df['fung_structural_rate'] = df['fung_structural_count'] / df['word_count'].replace(0, np.nan)
    df['fung_cognitive_rate'] = df['fung_cognitive_count'] / df['word_count'].replace(0, np.nan)
    df['fung_total_rate'] = df['fung_total_count'] / df['word_count'].replace(0, np.nan)
    
    # ============================================================
    # LIWC-BASED DISFLUENCY MARKERS
    # ============================================================
    
    df['nonfluency_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, LIWC_NONFLUENCIES))
    df['nonfluency_rate'] = df['nonfluency_count'] / df['word_count'].replace(0, np.nan)
    
    df['liwc_filler_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, LIWC_FILLERS))
    df['liwc_filler_rate'] = df['liwc_filler_count'] / df['word_count'].replace(0, np.nan)
    
    df['disfluency_count'] = df['nonfluency_count'] + df['liwc_filler_count']
    df['disfluency_rate'] = df['disfluency_count'] / df['word_count'].replace(0, np.nan)
    
    # ============================================================
    # OTHER LINGUISTIC MARKERS
    # ============================================================
    
    df['tom_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, TOM_PHRASES))
    df['tom_rate'] = df['tom_count'] / df['word_count'].replace(0, np.nan)
    
    df['polite_pos'] = df['transcript_sub'].apply(lambda x: count_patterns(x, POLITE_POSITIVE))
    df['polite_neg'] = df['transcript_sub'].apply(lambda x: count_patterns(x, POLITE_NEGATIVE))
    df['impolite'] = df['transcript_sub'].apply(lambda x: count_patterns(x, IMPOLITE))
    df['politeness_score'] = df['polite_pos'] + df['polite_neg'] - df['impolite']
    df['politeness_rate'] = df['politeness_score'] / df['word_count'].replace(0, np.nan)
    
    # Quotative/Discourse 'like' (Dailey-O'Cain, 2000; D'Arcy, 2007)
    df['like_count'] = df['transcript_sub'].apply(
        lambda x: len(re.findall(LIKE_MARKER, x.lower())) if isinstance(x, str) else 0
    )
    df['like_rate'] = df['like_count'] / df['word_count'].replace(0, np.nan)
    
    # ============================================================
    # SENTIMENT ANALYSIS - VADER
    # ============================================================
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        
        df['sentiment'] = df['transcript_sub'].apply(
            lambda x: sia.polarity_scores(str(x)).get('compound', np.nan)
            if isinstance(x, str) else np.nan
        )
        df['sentiment_pos'] = df['transcript_sub'].apply(
            lambda x: sia.polarity_scores(str(x)).get('pos', np.nan)
            if isinstance(x, str) else np.nan
        )
        df['sentiment_neg'] = df['transcript_sub'].apply(
            lambda x: sia.polarity_scores(str(x)).get('neg', np.nan)
            if isinstance(x, str) else np.nan
        )
        df['sentiment_neu'] = df['transcript_sub'].apply(
            lambda x: sia.polarity_scores(str(x)).get('neu', np.nan)
            if isinstance(x, str) else np.nan
        )
        print("  [OK] Sentiment analysis computed (VADER)")
    except (ImportError, LookupError):
        df['sentiment'] = np.nan
        df['sentiment_pos'] = np.nan
        df['sentiment_neg'] = np.nan
        df['sentiment_neu'] = np.nan
        print("  [SKIP] VADER sentiment analysis - nltk not available")
    
    return df


def aggregate_to_trial_level(df):
    """
    Aggregate utterance-level data to TRIAL level (per conversation/topic).
    """
    print("[INFO] Aggregating utterances to trial level (sum within each conversation)...")
    
    count_cols = [
        'word_count', 'question_count', 'tom_count', 
        'polite_pos', 'polite_neg', 'impolite', 'politeness_score',
        'nonfluency_count', 'liwc_filler_count', 'disfluency_count',
        'like_count',
        # Fung & Carter counts
        'fung_interpersonal_count', 'fung_referential_count', 
        'fung_structural_count', 'fung_cognitive_count', 'fung_total_count',
        # Demir hedge counts
        'demir_modal_count', 'demir_verb_count', 'demir_adverb_count',
        'demir_adjective_count', 'demir_quantifier_count', 'demir_noun_count',
        'demir_total_count'
    ]
    
    count_cols = [c for c in count_cols if c in df.columns]
    
    groupby_cols = ['experiment', 'subject', 'condition', 'topic', 'social_type', 'social']
    groupby_cols = [c for c in groupby_cols if c in df.columns]
    
    agg_dict = {col: 'sum' for col in count_cols}
    
    for sent_col in ['sentiment', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu']:
        if sent_col in df.columns:
            agg_dict[sent_col] = 'mean'
    
    if 'quality' in df.columns:
        agg_dict['quality'] = 'mean'
    if 'connectedness' in df.columns:
        agg_dict['connectedness'] = 'mean'
    
    trial_df = df.groupby(groupby_cols, as_index=False).agg(agg_dict)
    
    # Recompute rate metrics at trial level
    rate_pairs = [
        ('tom_rate', 'tom_count'),
        ('politeness_rate', 'politeness_score'),
        ('nonfluency_rate', 'nonfluency_count'),
        ('liwc_filler_rate', 'liwc_filler_count'),
        ('disfluency_rate', 'disfluency_count'),
        ('like_rate', 'like_count'),
        # Fung & Carter rates
        ('fung_interpersonal_rate', 'fung_interpersonal_count'),
        ('fung_referential_rate', 'fung_referential_count'),
        ('fung_structural_rate', 'fung_structural_count'),
        ('fung_cognitive_rate', 'fung_cognitive_count'),
        ('fung_total_rate', 'fung_total_count'),
        # Demir hedge rates
        ('demir_modal_rate', 'demir_modal_count'),
        ('demir_verb_rate', 'demir_verb_count'),
        ('demir_adverb_rate', 'demir_adverb_count'),
        ('demir_adjective_rate', 'demir_adjective_count'),
        ('demir_quantifier_rate', 'demir_quantifier_count'),
        ('demir_noun_rate', 'demir_noun_count'),
        ('demir_total_rate', 'demir_total_count'),
    ]
    
    for rate_col, count_col in rate_pairs:
        if count_col in trial_df.columns and 'word_count' in trial_df.columns:
            trial_df[rate_col] = trial_df[count_col] / trial_df['word_count'].replace(0, np.nan)
    
    print(f"  Aggregated {len(df)} utterances -> {len(trial_df)} trials")
    
    return trial_df


def aggregate_to_subject_condition_social(df, metrics):
    """Aggregate trial-level data to subject × condition × sociality."""
    print("[INFO] Aggregating to subject × condition × sociality (mean)...")
    return df.groupby(['experiment', 'subject', 'condition', 'social_type'])[metrics].mean().reset_index()


def aggregate_to_subject_condition(df, metrics):
    """Aggregate trial-level data to subject × condition (collapsed across sociality)."""
    return df.groupby(['experiment', 'subject', 'condition'])[metrics].mean().reset_index()


def aggregate_utterances_to_subject_social(df, metrics):
    """Aggregate DIRECTLY from utterances to subject × condition × sociality."""
    print("[INFO] Aggregating utterances directly to subject × condition × sociality (mean of utterances)...")
    return df.groupby(['experiment', 'subject', 'condition', 'social_type'])[metrics].mean().reset_index()


def aggregate_utterances_to_subject(df, metrics):
    """Aggregate DIRECTLY from utterances to subject × condition."""
    return df.groupby(['experiment', 'subject', 'condition'])[metrics].mean().reset_index()



# ============================================================
#                    OUTPUT FORMATTING
# ============================================================

def format_results(within_human, within_llm, cross_results, n_human, n_llm):
    """Format all results as text output."""
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-EXPERIMENT BEHAVIORAL ANALYSIS")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append(f"\nHuman: N = {n_human} subjects")
    lines.append(f"LLM:   N = {n_llm} subjects")
    
    # ================================================================
    # DEMIR (2018) HEDGE TAXONOMY CITATION NOTE
    # ================================================================
    lines.append("\n" + "-" * 80)
    lines.append("NOTE: Hedging Analysis Based on Demir (2018) Taxonomy")
    lines.append("-" * 80)
    lines.append("""
References:
  - Demir, C. (2018). Hedging and academic writing: An analysis of lexical hedges.
    Journal of Language and Linguistic Studies, 14(4), 74-92.
  - Hyland, K. (1998). Boosting, hedging, and the negotiation of academic knowledge.
    Text, 18(3), 349-382.

Demir's taxonomy follows Hyland's (1998) six-category framework.
Word lists taken verbatim from Demir (2018) Appendix A (pp. 90-91).

The Demir framework categorizes lexical hedges into six forms:

1. MODAL AUXILIARIES (6 items): can, could, may, might, should, would

2. EPISTEMIC VERBS (44 items): advise, advocate, agree with, allege, anticipate,
   appear, argue, assert, assume, attempt, believe, calculate, conjecture, consider,
   contend, correlate with, demonstrate, display, doubt, estimate, expect, feel, find,
   guess, hint, hope, hypothesize, implicate, imply, indicate, insinuate, intend,
   intimate, maintain, mention, observe, offer, opine, postulate, predict, presume,
   propose, reckon, recommend, report, reveal, seem, show, signal, speculate, suggest,
   support, suppose, surmise, suspect, tend to, think, try to

3. EPISTEMIC ADVERBS (47 items): about, admittedly, all but, almost, approximately,
   arguably, around, averagely, fairly, frequently, generally, hardly, largely, likely,
   mainly, mildly, moderately, mostly, near, nearly, not always, occasionally, often,
   partially, partly, passably, perhaps, possibly, potentially, predictably, presumably,
   primarily, probably, quite, rarely, rather, reasonably, relatively, roughly, scarcely,
   seemingly, slightly, sometimes, somewhat, subtly, supposedly, tolerably, usually, virtually

4. EPISTEMIC ADJECTIVES (19 items): advisable, approximate, consistent with, in conjunction
   with, in harmony with, in line with, in tune with, liable, likely, partial, plausible,
   possible, potential, probable, prone to, reasonable, reported, rough, slight, subtle,
   suggested, uncertain, unlikely

5. QUANTIFIERS/DETERMINERS (10 items): a few, few, little, a little, more or less, most,
   much, not all, on occasion, several, to a lesser, to a minor extent, to an extent,
   to some extent

6. EPISTEMIC NOUNS (19 items): agreement with, assertion, assumption, attempt, belief,
   chance, claim, doubt, estimate, expectation, guidance, hope, implication, in accord with,
   intention, majority, possibility, potential, prediction, presupposition, probability,
   proposal, proposition, recommendation, suggestion, tendency

Metrics computed:
  - demir_modal_rate: Modal auxiliaries / word_count
  - demir_verb_rate: Epistemic verbs / word_count
  - demir_adverb_rate: Epistemic adverbs / word_count
  - demir_adjective_rate: Epistemic adjectives / word_count
  - demir_quantifier_rate: Quantifiers/determiners / word_count
  - demir_noun_rate: Epistemic nouns / word_count
  - demir_total_rate: All Demir hedges / word_count
""")
    
    # ================================================================
    # FUNG & CARTER CITATION NOTE
    # ================================================================
    lines.append("\n" + "-" * 80)
    lines.append("NOTE: Discourse Marker Analysis Based on Fung & Carter (2007)")
    lines.append("-" * 80)
    lines.append("""
Fung, L., & Carter, R. (2007). Discourse markers and spoken English: Native and 
learner use in pedagogic settings. Applied Linguistics, 28(3), 410-439.

Metrics: fung_interpersonal_rate, fung_referential_rate, fung_structural_rate, 
         fung_cognitive_rate, fung_total_rate
""")
    
    # ================================================================
    # LIWC CITATION NOTE
    # ================================================================
    lines.append("\n" + "-" * 80)
    lines.append("NOTE: Disfluency Analysis Uses LIWC2007 Categories")
    lines.append("-" * 80)
    lines.append("""
LIWC Nonfluencies: er, hm*, sigh, uh, um, umm*, well
LIWC Fillers: blah, i don't know, i mean, oh well, or anything, or something, 
              or whatever, ya know, y'know, you know
""")
    
    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    lines.append("\n" + "=" * 110)
    lines.append("SUMMARY TABLE: Comparison of Belief-Conditioned Effects in Humans vs LLMs")
    lines.append("=" * 110)
    lines.append(f"\n{'Measure':<30} {'Human':<20} {'LLM':<20} {'Cross-Exp p':<15} {'Pattern':<12}")
    lines.append("-" * 110)
    
    for r in cross_results:
        if 'error' in r:
            continue
        metric = r['metric']
        
        int_p = f"{r.get('interaction_p', np.nan):.4f}" if 'interaction_p' in r else 'N/A'
        int_sig = ''
        if 'interaction_p' in r and not np.isnan(r.get('interaction_p', np.nan)):
            if r['interaction_p'] < 0.001: int_sig = '***'
            elif r['interaction_p'] < 0.01: int_sig = '**'
            elif r['interaction_p'] < 0.05: int_sig = '*'
        
        human_sig = '*' if r.get('human_within_p', 1) < 0.05 else ''
        llm_sig = '*' if r.get('llm_within_p', 1) < 0.05 else ''
        
        lines.append(f"{metric:<30} {r['human_direction'] + human_sig:<20} {r['llm_direction'] + llm_sig:<20} {int_p + int_sig:<15} {r['pattern']:<12}")
    
    lines.append("-" * 110)
    lines.append("Within-exp significance: * p < .05 (paired t-test)")
    lines.append("Cross-exp significance: * p < .05, ** p < .01, *** p < .001 (independent t-test on condition effects)")
    
    # ================================================================
    # WITHIN-EXPERIMENT DETAILED RESULTS
    # ================================================================
    lines.append("\n\n" + "=" * 80)
    lines.append("WITHIN-EXPERIMENT ANALYSES: 2×2 RM-ANOVA (Condition × Sociality)")
    lines.append("=" * 80)
    
    for r in within_human + within_llm:
        if r is None:
            continue
        if 'error' in r:
            lines.append(f"\n{r.get('experiment', '?')} - {r.get('metric', '?')}: ERROR - {r['error']}")
            continue
        
        lines.append(f"\n{'-' * 70}")
        lines.append(f"{r['experiment']} | {r['metric']} (N = {r['n_subjects']})")
        lines.append(f"{'-' * 70}")
        
        lines.append(f"\n  Condition Means (collapsed across sociality):")
        lines.append(f"    Human-label: M = {r.get('hum_mean', np.nan):.4f} ± {r.get('hum_sem', np.nan):.4f}")
        lines.append(f"    AI-label:    M = {r.get('bot_mean', np.nan):.4f} ± {r.get('bot_sem', np.nan):.4f}")
        
        lines.append(f"\n  Sociality Means (collapsed across condition):")
        lines.append(f"    Social:      M = {r.get('social_mean', np.nan):.4f} ± {r.get('social_sem', np.nan):.4f}")
        lines.append(f"    Nonsocial:   M = {r.get('nonsocial_mean', np.nan):.4f} ± {r.get('nonsocial_sem', np.nan):.4f}")
        
        lines.append(f"\n  ANOVA Results:")
        
        sig_c = '***' if r.get('condition_p', 1) < 0.001 else ('**' if r.get('condition_p', 1) < 0.01 else ('*' if r.get('condition_p', 1) < 0.05 else ''))
        df_str = f"(1, {r['n_subjects'] - 1})" if 'condition_df' not in r else f"({r['condition_df'][0]}, {r['condition_df'][1]})"
        lines.append(f"    Main Effect of Condition:   F{df_str} = {r.get('condition_F', np.nan):.2f}, p = {r.get('condition_p', np.nan):.4f} {sig_c}")
        
        sig_s = '***' if r.get('sociality_p', 1) < 0.001 else ('**' if r.get('sociality_p', 1) < 0.01 else ('*' if r.get('sociality_p', 1) < 0.05 else ''))
        lines.append(f"    Main Effect of Sociality:   F = {r.get('sociality_F', np.nan):.2f}, p = {r.get('sociality_p', np.nan):.4f} {sig_s}")
        
        sig_i = '***' if r.get('interaction_p', 1) < 0.001 else ('**' if r.get('interaction_p', 1) < 0.01 else ('*' if r.get('interaction_p', 1) < 0.05 else ''))
        lines.append(f"    Condition × Sociality:      F = {r.get('interaction_F', np.nan):.2f}, p = {r.get('interaction_p', np.nan):.4f} {sig_i}")
        
        if 'simple_effects' in r:
            lines.append(f"\n  Simple Effects (Condition effect at each Sociality level):")
            for soc_level in ['social', 'nonsocial']:
                se = r['simple_effects'].get(soc_level, {})
                if 'error' in se:
                    lines.append(f"    {soc_level.capitalize()}: ERROR - {se['error']}")
                else:
                    sig_se = '*' if se.get('p_val', 1) < 0.05 else ''
                    lines.append(f"    {soc_level.capitalize()} topics: Human M = {se.get('hum_mean', np.nan):.4f}, AI M = {se.get('bot_mean', np.nan):.4f}")
                    lines.append(f"      Diff = {se.get('diff_mean', np.nan):+.4f}, t({se.get('n', 0)-1}) = {se.get('t_stat', np.nan):.2f}, p = {se.get('p_val', np.nan):.4f} {sig_se} [{se.get('direction', '')}]")
    
    # ================================================================
    # CROSS-EXPERIMENT DETAILED RESULTS
    # ================================================================
    lines.append("\n\n" + "=" * 80)
    lines.append("CROSS-EXPERIMENT ANALYSES: Experiment × Condition Interaction")
    lines.append("=" * 80)
    
    for r in cross_results:
        if 'error' in r:
            lines.append(f"\n{r.get('metric', 'Unknown')}: ERROR - {r['error']}")
            continue
        
        lines.append(f"\n{'-' * 70}")
        lines.append(f"{r['metric']}")
        lines.append(f"  Sample sizes: Human N = {r['n_human']}, LLM N = {r['n_llm']}")
        lines.append(f"{'-' * 70}")
        
        lines.append(f"\n  Condition Means by Experiment:")
        lines.append(f"    Human: Human-label = {r.get('human_hum_mean', np.nan):.4f}, AI-label = {r.get('human_bot_mean', np.nan):.4f}")
        lines.append(f"    LLM:   Human-label = {r.get('llm_hum_mean', np.nan):.4f}, AI-label = {r.get('llm_bot_mean', np.nan):.4f}")
        
        lines.append(f"\n  Condition Effect (Human-label minus AI-label):")
        
        p1_str = f"  [t = {r.get('human_within_t', np.nan):.2f}, p = {r.get('human_within_p', np.nan):.4f}]" if 'human_within_p' in r else ""
        p2_str = f"  [t = {r.get('llm_within_t', np.nan):.2f}, p = {r.get('llm_within_p', np.nan):.4f}]" if 'llm_within_p' in r else ""
        
        lines.append(f"    Human: M = {r['human_effect_mean']:+.4f} ± {r.get('human_effect_sem', np.nan):.4f}{p1_str}")
        lines.append(f"    LLM:   M = {r['llm_effect_mean']:+.4f} ± {r.get('llm_effect_sem', np.nan):.4f}{p2_str}")
        
        if 'interaction_t' in r:
            sig = '***' if r['interaction_p'] < 0.001 else ('**' if r['interaction_p'] < 0.01 else ('*' if r['interaction_p'] < 0.05 else ''))
            lines.append(f"\n  Cross-Experiment Interaction Test:")
            lines.append(f"    t({r['interaction_df']}) = {r['interaction_t']:.3f}, p = {r['interaction_p']:.4f} {sig}")
        
        lines.append(f"\n  Direction: Human = {r['human_direction']}, LLM = {r['llm_direction']}")
        lines.append(f"  Pattern Classification: {r['pattern']}")
    
    return '\n'.join(lines)


# ============================================================
#                    DIAGNOSTIC FUNCTIONS
# ============================================================

def print_diagnostics(df, experiment_name):
    """Print diagnostic information to help verify data loading."""
    print(f"\n[DIAGNOSTICS] {experiment_name}")
    print(f"  Total utterances: {len(df)}")
    print(f"  Subjects: {df['subject'].nunique()}")
    print(f"  Word count stats: mean={df['word_count'].mean():.2f}, std={df['word_count'].std():.2f}")
    print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"  Social types: {df['social_type'].value_counts().to_dict()}")
    
    print(f"\n  Demir (2018) Hedge Taxonomy:")
    print(f"    Modal auxiliaries:  mean={df['demir_modal_count'].mean():.3f}")
    print(f"    Epistemic verbs:    mean={df['demir_verb_count'].mean():.3f}")
    print(f"    Epistemic adverbs:  mean={df['demir_adverb_count'].mean():.3f}")
    print(f"    Epistemic adj:      mean={df['demir_adjective_count'].mean():.3f}")
    print(f"    Quantifiers:        mean={df['demir_quantifier_count'].mean():.3f}")
    print(f"    Epistemic nouns:    mean={df['demir_noun_count'].mean():.3f}")
    print(f"    TOTAL:              mean={df['demir_total_count'].mean():.3f}")
    
    print(f"\n  Fung & Carter Discourse Markers:")
    print(f"    Interpersonal: mean={df['fung_interpersonal_count'].mean():.3f}")
    print(f"    Referential:   mean={df['fung_referential_count'].mean():.3f}")
    print(f"    Structural:    mean={df['fung_structural_count'].mean():.3f}")
    print(f"    Cognitive:     mean={df['fung_cognitive_count'].mean():.3f}")
    
    print(f"\n  LIWC Disfluency Metrics:")
    print(f"    Nonfluency count: mean={df['nonfluency_count'].mean():.3f}")
    print(f"    LIWC Filler count: mean={df['liwc_filler_count'].mean():.3f}")
    
    if 'sentiment' in df.columns:
        n_valid = df['sentiment'].notna().sum()
        if n_valid > 0:
            print(f"  Sentiment (compound): {n_valid} non-null values, mean={df['sentiment'].mean():.4f}")


# ============================================================
#                    MAIN EXECUTION
# ============================================================

def main():
    args, cfg = parse_and_load_config("Cross-experiment behavioral comparison")
    model = cfg.get("model")
    temp = cfg.get("temperature")
    human_ids = cfg.get("HUMAN_IDS")
    
    # Build paths dynamically
    LLM_PATH = f"data/{model}/{temp}/combined_text_data.csv"
    OUTPUT_DIR = f"results/{model}/{temp}/"

    print("\n" + "=" * 80)
    print("CROSS-EXPERIMENT ANALYSIS: Human vs LLM Behavioral Patterns")
    print("(v7 - cleaned up, removed legacy hedge_rate and filler_rate)")
    print("=" * 80 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    topic_df = load_topics_file(TOPICS_PATH)
    
    # Load data
    human_df = load_human_data(
        subject_dir=HUMAN_DIR,
        subject_ids=human_ids,
        file_pattern=HUMAN_FILE_PATTERN,
        topic_df=topic_df
    )
    
    llm_df = load_llm_data(LLM_PATH, topic_df)
    
    n_human = human_df['subject'].nunique()
    n_llm = llm_df['subject'].nunique()
    
    # Compute metrics
    human_df = compute_all_metrics(human_df)
    llm_df = compute_all_metrics(llm_df)
    
    print_diagnostics(human_df, "Human")
    print_diagnostics(llm_df, "LLM")
    
    combined_utterance = pd.concat([human_df, llm_df], ignore_index=True)
    
    # Define metrics
    count_metrics = ['word_count', 'question_count']
    
    rate_metrics = [
        # Demir hedge taxonomy
        'demir_modal_rate',
        'demir_verb_rate',
        'demir_adverb_rate',
        'demir_adjective_rate',
        'demir_quantifier_rate',
        'demir_noun_rate',
        'demir_total_rate',
        # Fung & Carter discourse markers
        'fung_interpersonal_rate',
        'fung_referential_rate',
        'fung_structural_rate', 
        'fung_cognitive_rate',
        'fung_total_rate',
        # LIWC disfluencies
        'nonfluency_rate',
        'liwc_filler_rate',
        'disfluency_rate',
        # Discourse marker 'like'
        'like_rate',
        # Other markers
        'tom_rate', 
        'politeness_rate'
    ]
    
    score_metrics = []
    if 'sentiment' in combined_utterance.columns and combined_utterance['sentiment'].notna().any():
        score_metrics.append('sentiment')
    if 'quality' in combined_utterance.columns and combined_utterance['quality'].notna().any():
        score_metrics.append('quality')
    if 'connectedness' in combined_utterance.columns and combined_utterance['connectedness'].notna().any():
        score_metrics.append('connectedness')
    
    all_metrics = count_metrics + rate_metrics + score_metrics
    print(f"\n[INFO] Analyzing metrics: {all_metrics}")
    
    # Save utterance-level data
    utterance_path = os.path.join(OUTPUT_DIR, "combined_utterance_level_data.csv")
    combined_utterance.to_csv(utterance_path, index=False)
    print(f"[SAVED] {utterance_path}")
    
    # ========================================================================
    # PIPELINE 1: PER-TRIAL ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE 1: PER-TRIAL ANALYSIS")
    print("="*80)
    
    combined_trials = aggregate_to_trial_level(combined_utterance)
    trial_social = aggregate_to_subject_condition_social(combined_trials, all_metrics)
    trial_subject = aggregate_to_subject_condition(combined_trials, all_metrics)
    
    print("\n[INFO] Running per-trial RM-ANOVAs...")
    trial_within_human = []
    trial_within_llm = []
    for m in all_metrics:
        r1 = run_within_experiment_anova(trial_social, m, 'Human')
        r2 = run_within_experiment_anova(trial_social, m, 'LLM')
        if r1: trial_within_human.append(r1)
        if r2: trial_within_llm.append(r2)
    
    print("[INFO] Running per-trial cross-experiment comparisons...")
    trial_cross = [run_cross_experiment_analysis(trial_subject, m) for m in all_metrics]
    
    trial_output = format_results(trial_within_human, trial_within_llm, trial_cross, n_human, n_llm)
    trial_output = "AGGREGATION METHOD: PER-TRIAL (sum within trials, then mean across trials)\n\n" + trial_output
    
    trial_stats_path = os.path.join(OUTPUT_DIR, "cross_experiment_stats_PER_TRIAL.txt")
    with open(trial_stats_path, 'w') as f:
        f.write(trial_output)
    print(f"[SAVED] {trial_stats_path}")
    
    trial_social.to_csv(os.path.join(OUTPUT_DIR, "per_trial_subject_condition_social.csv"), index=False)
    trial_subject.to_csv(os.path.join(OUTPUT_DIR, "per_trial_subject_condition.csv"), index=False)
    combined_trials.to_csv(os.path.join(OUTPUT_DIR, "combined_trial_level_data.csv"), index=False)
    
    # ========================================================================
    # PIPELINE 2: PER-UTTERANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE 2: PER-UTTERANCE ANALYSIS")
    print("="*80)
    
    utt_social = aggregate_utterances_to_subject_social(combined_utterance, all_metrics)
    utt_subject = aggregate_utterances_to_subject(combined_utterance, all_metrics)
    
    print("\n[INFO] Running per-utterance RM-ANOVAs...")
    utt_within_human = []
    utt_within_llm = []
    for m in all_metrics:
        r1 = run_within_experiment_anova(utt_social, m, 'Human')
        r2 = run_within_experiment_anova(utt_social, m, 'LLM')
        if r1: utt_within_human.append(r1)
        if r2: utt_within_llm.append(r2)
    
    print("[INFO] Running per-utterance cross-experiment comparisons...")
    utt_cross = [run_cross_experiment_analysis(utt_subject, m) for m in all_metrics]
    
    utt_output = format_results(utt_within_human, utt_within_llm, utt_cross, n_human, n_llm)
    utt_output = "AGGREGATION METHOD: PER-UTTERANCE (mean directly from utterances, no trial-level summing)\n\n" + utt_output
    
    utt_stats_path = os.path.join(OUTPUT_DIR, "cross_experiment_stats_PER_UTTERANCE.txt")
    with open(utt_stats_path, 'w') as f:
        f.write(utt_output)
    print(f"[SAVED] {utt_stats_path}")
    
    utt_social.to_csv(os.path.join(OUTPUT_DIR, "per_utterance_subject_condition_social.csv"), index=False)
    utt_subject.to_csv(os.path.join(OUTPUT_DIR, "per_utterance_subject_condition.csv"), index=False)
    
    print("\n" + "="*80)
    print(f"[DONE] ✅ Analysis complete. Results saved to {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()