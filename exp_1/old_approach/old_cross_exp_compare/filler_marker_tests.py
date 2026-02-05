#!/usr/bin/env python3
"""
Script name: individual_marker_analysis_v3.py
Purpose: Analyze each discourse marker from the original filler list individually
         to identify which specific markers drive the overall filler_rate condition effect.

Based on cross_experiment_comparison_v4.py - uses identical data loading and analysis conventions.

Individual markers analyzed (from original filler list):
    - Filled pauses: um, uh, er, ah (Clark & Fox Tree, 2002)
    - Discourse markers: like, you know, I mean, well, so (Schiffrin, 1987)
    - Pragmatic markers: basically, actually, right, okay (Aijmer, 2002)

Author: Rachel C. Metzgar
Date: 2025
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, ttest_ind, f as f_dist
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#                    SUBJECT ID MAPPING
#         (Matches original subject_utils.py / globals.py)
# ============================================================

def get_sub_id_map():
    """Legacy behavioral ID -> standardized MRI ID mapping."""
    return {
        "P08": "sub-001", "s08": "sub-001",
        "P12": "sub-002", "s12": "sub-002",
        "P13": "sub-003", "s13": "sub-003",
        "P14": "sub-004", "s14": "sub-004",
        "P15": "sub-005", "s15": "sub-005",
        "P16": "sub-006", "s16": "sub-006",
        "P17": "sub-007", "s17": "sub-007",
        "P18": "sub-008", "s18": "sub-008",
        "P20": "sub-009", "s20": "sub-009",
        "P21": "sub-010", "s21": "sub-010",
        "P22": "sub-011", "s22": "sub-011",
        "P24": "sub-012", "s24": "sub-012",
        "P25": "sub-013", "s25": "sub-013",
        "P26": "sub-014", "s26": "sub-014",
        "P27": "sub-015", "s27": "sub-015",
        "P28": "sub-016", "s28": "sub-016",
        "P30": "sub-017", "s30": "sub-017",
        "P31": "sub-018", "s31": "sub-018",
        "P32": "sub-019", "s32": "sub-019",
        "P33": "sub-020", "s33": "sub-020",
        "P34": "sub-021", "s34": "sub-021",
        "P35": "sub-022", "s35": "sub-022",
        "P36": "sub-023", "s36": "sub-023",
    }


def standardize_sub_id(raw_id: str) -> str:
    """Map a behavioral/legacy ID to the standardized MRI ID."""
    key = raw_id.strip()
    mapping = get_sub_id_map()
    return mapping.get(key, key)


def find_old_id(new_id: str) -> str | None:
    """Return the old subject ID (e.g. s36, P36) given sub-###."""
    id_map = get_sub_id_map()
    old_ids = [k for k, v in id_map.items() if v == new_id]
    if not old_ids:
        return None
    for old in old_ids:
        if old.startswith("s"):
            return old
    return old_ids[0]


def get_all_old_ids(new_id: str) -> list:
    """Return all old subject IDs (e.g. ['s36', 'P36']) given sub-###."""
    id_map = get_sub_id_map()
    return [k for k, v in id_map.items() if v == new_id]


# ============================================================
#                    CONFIGURATION - EDIT THESE
# ============================================================

EXP1_USE_INDIVIDUAL_FILES = True
EXP1_SUBJECT_DIR = "exp_csv_human"
EXP1_FILE_PATTERN = "{sub_id}.csv"

EXP1_SUBJECT_IDS = [
    "sub-001", "sub-002", "sub-003", "sub-004", "sub-005",
    "sub-006", "sub-007", "sub-008", "sub-009", "sub-010",
    "sub-011", "sub-012", "sub-013", "sub-014", "sub-015",
    "sub-016", "sub-017", "sub-018", "sub-019", "sub-020",
    "sub-021", "sub-022", "sub-023"
]

EXP1_COMBINED_PATH = "combined_text_data_humans.csv"
EXP2_PATH = "combined_text_data_LLM.csv"
OUTPUT_DIR = "individual_marker_results"
TOPICS_PATH = "topics.csv"

# Topic classification
TOPIC_SOCIAL_MAP = {
    # Social topics (1)
    "friendship": 1, "conflict resolution": 1, "hobbies": 1, "childhood memories": 1,
    "support networks": 1, "empathy": 1, "mentorship": 1, "role models": 1,
    "cultural celebrations": 1, "trust": 1, "love": 1, "loneliness": 1,
    "forgiveness": 1, "work-life balance": 1, "communication styles": 1,
    "teamwork": 1, "life goals": 1, "family traditions": 1, "social media": 1,
    "friendship boundaries": 1,
    # Abbreviated names
    "conflict": 1, "childhood_memories": 1, "role_models": 1, "networks": 1,
    "cultural_celebrations": 1, "work_life": 1, "communication": 1, "goals": 1,
    "boundaries": 1, "social_media": 1, "family": 1,
    # Nonsocial topics (0)
    "cars": 0, "nature": 0, "space exploration": 0, "photography": 0,
    "books": 0, "exercise": 0, "art": 0, "sports": 0, "movies": 0,
    "architecture": 0, "cooking": 0, "music": 0, "seasons": 0,
    "podcasts": 0, "weather": 0, "travel": 0, "gardening": 0,
    "food": 0, "fashion": 0, "technology": 0,
    "space": 0,
}


# ============================================================
#            INDIVIDUAL MARKER DEFINITIONS
# ============================================================

# Individual markers from the original filler list
# Each gets its own count and rate metric

INDIVIDUAL_MARKERS = {
    # === Filled Pauses (Clark & Fox Tree, 2002) ===
    "um": r"\bum+\b",
    "uh": r"\buh+\b",
    "er": r"\ber+\b",
    "ah": r"\bah+\b",
    
    # === Discourse Connectives (Schiffrin, 1987) ===
    "well": r"\bwell\b",
    "so": r"\bso\b",
    
    # === Interactional Phrases (Fox Tree & Schrock, 2002) ===
    "you_know": r"\byou know\b",
    "i_mean": r"\bi mean\b",
    
    # === Quotative/Discourse 'like' ===
    "like": r"\blike\b",
    
    # === Pragmatic Markers (Aijmer, 2002) ===
    "basically": r"\bbasically\b",
    "actually": r"\bactually\b",
    "right": r"\bright\b",
    "okay": r"\b(okay|ok)\b",
}

# Category groupings for subtotals
MARKER_CATEGORIES = {
    "filled_pauses": ["um", "uh", "er", "ah"],
    "discourse_connectives": ["well", "so"],
    "interactional_phrases": ["you_know", "i_mean"],
    "quotative_hedge": ["like"],
    "pragmatic_stance": ["basically", "actually", "right", "okay"],
}

# Original composite filler pattern (for comparison)
FILLER_MARKERS_ORIGINAL = [
    r"\bum\b", r"\buh\b", r"\ber\b", r"\bah\b",
    r"\blike\b", r"\byou know\b", r"\bi mean\b",
    r"\bwell\b", r"\bbasically\b", r"\bactually\b",
    r"\bright\b", r"\bokay\b", r"\bso\b"
]


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


def load_exp1_from_individual_files(subject_dir, subject_ids, file_pattern, topic_df=None):
    """Load human experiment data from individual per-subject CSVs."""
    print(f"[INFO] Loading Exp1 (Human) from individual subject files...")
    print(f"  Directory: {subject_dir}")
    print(f"  Subjects: {len(subject_ids)}")
    
    all_trials = []
    loaded_subjects = []
    
    for sub_id in subject_ids:
        old_ids = get_all_old_ids(sub_id)
        old_id = find_old_id(sub_id)
        
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
        
        df['condition'] = df['agent'].astype(str).str.extract(r'(hum|bot)', expand=False)
        df['subject'] = sub_id
        df['topic'] = df['topic'].astype(str).str.strip()
        
        all_trials.append(df)
        loaded_subjects.append(sub_id)
    
    if not all_trials:
        raise ValueError("No valid data found for Exp1 (Human)")
    
    df_combined = pd.concat(all_trials, ignore_index=True)
    df_combined = get_social_classification(df_combined, topic_df)
    df_combined['experiment'] = 'Exp1_Human'
    
    print(f"  Loaded {len(df_combined)} rows (utterances) from {len(loaded_subjects)} subjects")
    print(f"  Conditions: {df_combined['condition'].value_counts().to_dict()}")
    
    return df_combined


def load_exp1_from_combined_file(filepath, topic_df=None):
    """Load human experiment data from combined CSV."""
    print(f"[INFO] Loading Exp1 (Human) from combined file: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Exp1 combined file not found: {filepath}")
    
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
    df['experiment'] = 'Exp1_Human'
    
    print(f"  Loaded {len(df)} rows (utterances), {df['subject'].nunique()} subjects")
    print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
    
    return df


def load_exp2_data(filepath, topic_df=None):
    """Load LLM experiment data from combined CSV."""
    print(f"[INFO] Loading Exp2 (LLM) from {filepath}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Exp2 file not found: {filepath}")
    
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
    df['experiment'] = 'Exp2_LLM'
    
    print(f"  Loaded {len(df)} rows (utterances), {df['subject'].nunique()} subjects")
    print(f"  Conditions: {df['condition'].value_counts().to_dict()}")
    
    return df


# ============================================================
#                    FEATURE COMPUTATION
# ============================================================

def count_pattern(text, pattern):
    """Count occurrences of a single regex pattern in text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(pattern, text.lower()))


def count_patterns(text, patterns):
    """Count occurrences of multiple regex patterns in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def compute_individual_markers(df):
    """
    Compute counts and rates for each individual marker from the original filler list.
    """
    print("[INFO] Computing individual marker metrics at utterance level...")
    df = df.copy()
    
    # Word count
    df['word_count'] = df['transcript_sub'].apply(_count_words)
    
    # Compute each individual marker
    for marker_name, pattern in INDIVIDUAL_MARKERS.items():
        count_col = f"{marker_name}_count"
        rate_col = f"{marker_name}_rate"
        
        df[count_col] = df['transcript_sub'].apply(lambda x: count_pattern(x, pattern))
        df[rate_col] = df[count_col] / df['word_count'].replace(0, np.nan)
    
    # Compute category subtotals
    for cat_name, markers in MARKER_CATEGORIES.items():
        count_cols = [f"{m}_count" for m in markers]
        df[f"{cat_name}_count"] = df[count_cols].sum(axis=1)
        df[f"{cat_name}_rate"] = df[f"{cat_name}_count"] / df['word_count'].replace(0, np.nan)
    
    # Compute original composite filler (for comparison)
    df['filler_composite_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FILLER_MARKERS_ORIGINAL)
    )
    df['filler_composite_rate'] = df['filler_composite_count'] / df['word_count'].replace(0, np.nan)
    
    # Print summary
    print(f"  [OK] Individual markers computed:")
    for marker_name in INDIVIDUAL_MARKERS.keys():
        mean_count = df[f"{marker_name}_count"].mean()
        print(f"       {marker_name}: mean = {mean_count:.3f} per utterance")
    print(f"       COMPOSITE: mean = {df['filler_composite_count'].mean():.3f} per utterance")
    
    return df


def aggregate_utterances_to_subject_social(df, metrics):
    """Aggregate DIRECTLY from utterances to subject × condition × sociality."""
    print("[INFO] Aggregating utterances to subject × condition × sociality (mean)...")
    return df.groupby(['experiment', 'subject', 'condition', 'social_type'])[metrics].mean().reset_index()


def aggregate_utterances_to_subject(df, metrics):
    """Aggregate DIRECTLY from utterances to subject × condition."""
    return df.groupby(['experiment', 'subject', 'condition'])[metrics].mean().reset_index()


# ============================================================
#                    STATISTICAL ANALYSES
# ============================================================

def run_simple_effects(aov_df, metric):
    """Run simple effects tests: Condition effect at each level of Sociality."""
    simple_effects = {}
    
    for sociality in ['social', 'nonsocial']:
        soc_df = aov_df[aov_df['social_type'] == sociality]
        pivot = soc_df.pivot(index='subject', columns='condition', values=metric)
        
        if 'hum' not in pivot.columns or 'bot' not in pivot.columns:
            simple_effects[sociality] = {'error': 'Missing condition data'}
            continue
        
        pivot = pivot.dropna()
        
        if len(pivot) < 3:
            simple_effects[sociality] = {'error': f'Insufficient subjects: {len(pivot)}'}
            continue
        
        hum_vals = pivot['hum']
        bot_vals = pivot['bot']
        
        t_stat, p_val = ttest_rel(hum_vals, bot_vals)
        
        simple_effects[sociality] = {
            'hum_mean': hum_vals.mean(),
            'hum_sem': hum_vals.sem(),
            'bot_mean': bot_vals.mean(),
            'bot_sem': bot_vals.sem(),
            'diff_mean': (hum_vals - bot_vals).mean(),
            'diff_sem': (hum_vals - bot_vals).sem(),
            't_stat': t_stat,
            'p_val': p_val,
            'n': len(pivot),
            'direction': 'Human > AI' if hum_vals.mean() > bot_vals.mean() else 'AI > Human',
            'sig': '*' if p_val < .05 else ''
        }
    
    return simple_effects


def run_within_experiment_anova(df, metric, experiment_name):
    """
    Run 2×2 RM-ANOVA (Condition × Sociality) for one experiment.
    Uses manual computation with proper error terms.
    """
    exp_df = df[df['experiment'] == experiment_name].copy()
    
    if exp_df.empty:
        return None
    
    aov_df = exp_df.copy()
    n_subjects = aov_df['subject'].nunique()
    n_cells = aov_df.groupby(['condition', 'social_type']).size()
    
    if n_subjects < 3:
        return {'error': f'Insufficient subjects: {n_subjects}', 'experiment': experiment_name, 'metric': metric}
    
    if len(n_cells) < 4:
        return {'error': f'Missing cells: {n_cells.to_dict()}', 'experiment': experiment_name, 'metric': metric}
    
    if aov_df[metric].isna().all():
        return {'error': f'All values are NaN for {metric}', 'experiment': experiment_name, 'metric': metric}
    
    aov_df = aov_df.dropna(subset=[metric])
    
    # Keep only subjects with complete data (all 4 cells)
    complete_subjects = aov_df.groupby('subject').filter(lambda x: len(x) == 4)['subject'].unique()
    aov_df = aov_df[aov_df['subject'].isin(complete_subjects)]
    n_subjects = aov_df['subject'].nunique()
    
    if n_subjects < 3:
        return {'error': f'Insufficient complete subjects: {n_subjects}', 'experiment': experiment_name, 'metric': metric}
    
    results = {
        'experiment': experiment_name,
        'metric': metric,
        'n_subjects': n_subjects,
    }
    
    try:
        wide = aov_df.pivot_table(
            index='subject',
            columns=['condition', 'social_type'],
            values=metric
        )
        
        Y_hs = wide[('hum', 'social')].values
        Y_hn = wide[('hum', 'nonsocial')].values
        Y_bs = wide[('bot', 'social')].values
        Y_bn = wide[('bot', 'nonsocial')].values
        
        n = len(Y_hs)
        
        GM = (Y_hs + Y_hn + Y_bs + Y_bn).mean() / 4
        
        M_hum = (Y_hs + Y_hn).mean() / 2
        M_bot = (Y_bs + Y_bn).mean() / 2
        M_social = (Y_hs + Y_bs).mean() / 2
        M_nonsocial = (Y_hn + Y_bn).mean() / 2
        
        M_subj = (Y_hs + Y_hn + Y_bs + Y_bn) / 4
        
        # Main effect of Condition
        SS_cond = 2 * n * ((M_hum - GM)**2 + (M_bot - GM)**2)
        df_cond = 1
        
        subj_hum = (Y_hs + Y_hn) / 2
        subj_bot = (Y_bs + Y_bn) / 2
        SS_cond_subj = 2 * np.sum((subj_hum - M_subj - M_hum + GM)**2 + (subj_bot - M_subj - M_bot + GM)**2)
        df_cond_subj = n - 1
        
        MS_cond = SS_cond / df_cond
        MS_cond_subj = SS_cond_subj / df_cond_subj
        F_cond = MS_cond / MS_cond_subj if MS_cond_subj > 0 else np.nan
        p_cond = 1 - f_dist.cdf(F_cond, df_cond, df_cond_subj) if not np.isnan(F_cond) else np.nan
        
        results['condition_F'] = F_cond
        results['condition_p'] = p_cond
        results['condition_df'] = (df_cond, df_cond_subj)
        
        # Main effect of Sociality
        SS_social = 2 * n * ((M_social - GM)**2 + (M_nonsocial - GM)**2)
        df_social = 1
        
        subj_soc = (Y_hs + Y_bs) / 2
        subj_nonsoc = (Y_hn + Y_bn) / 2
        SS_social_subj = 2 * np.sum((subj_soc - M_subj - M_social + GM)**2 + (subj_nonsoc - M_subj - M_nonsocial + GM)**2)
        df_social_subj = n - 1
        
        MS_social = SS_social / df_social
        MS_social_subj = SS_social_subj / df_social_subj
        F_social = MS_social / MS_social_subj if MS_social_subj > 0 else np.nan
        p_social = 1 - f_dist.cdf(F_social, df_social, df_social_subj) if not np.isnan(F_social) else np.nan
        
        results['sociality_F'] = F_social
        results['sociality_p'] = p_social
        results['sociality_df'] = (df_social, df_social_subj)
        
        # Condition × Sociality Interaction
        M_hs = Y_hs.mean()
        M_hn = Y_hn.mean()
        M_bs = Y_bs.mean()
        M_bn = Y_bn.mean()
        
        SS_inter = n * ((M_hs - M_hum - M_social + GM)**2 + 
                        (M_hn - M_hum - M_nonsocial + GM)**2 +
                        (M_bs - M_bot - M_social + GM)**2 + 
                        (M_bn - M_bot - M_nonsocial + GM)**2)
        df_inter = 1
        
        SS_inter_subj = np.sum(
            (Y_hs - subj_hum - subj_soc + M_subj - M_hs + M_hum + M_social - GM)**2 +
            (Y_hn - subj_hum - subj_nonsoc + M_subj - M_hn + M_hum + M_nonsocial - GM)**2 +
            (Y_bs - subj_bot - subj_soc + M_subj - M_bs + M_bot + M_social - GM)**2 +
            (Y_bn - subj_bot - subj_nonsoc + M_subj - M_bn + M_bot + M_nonsocial - GM)**2
        )
        df_inter_subj = n - 1
        
        MS_inter = SS_inter / df_inter
        MS_inter_subj = SS_inter_subj / df_inter_subj
        F_inter = MS_inter / MS_inter_subj if MS_inter_subj > 0 else np.nan
        p_inter = 1 - f_dist.cdf(F_inter, df_inter, df_inter_subj) if not np.isnan(F_inter) else np.nan
        
        results['interaction_F'] = F_inter
        results['interaction_p'] = p_inter
        results['interaction_df'] = (df_inter, df_inter_subj)
        
        # Descriptive statistics
        cond_means = aov_df.groupby('condition')[metric].agg(['mean', 'sem'])
        results['hum_mean'] = cond_means.loc['hum', 'mean'] if 'hum' in cond_means.index else np.nan
        results['hum_sem'] = cond_means.loc['hum', 'sem'] if 'hum' in cond_means.index else np.nan
        results['bot_mean'] = cond_means.loc['bot', 'mean'] if 'bot' in cond_means.index else np.nan
        results['bot_sem'] = cond_means.loc['bot', 'sem'] if 'bot' in cond_means.index else np.nan
        
        # Simple effects if interaction is significant
        if results.get('interaction_p', 1.0) < 0.05:
            results['simple_effects'] = run_simple_effects(aov_df, metric)
        
    except Exception as e:
        return {'error': str(e), 'experiment': experiment_name, 'metric': metric}
    
    return results


def run_cross_experiment_analysis(combined_subject_df, metric):
    """Test whether the Condition effect differs between experiments."""
    df = combined_subject_df.copy()
    
    wide_df = df.pivot_table(
        index=['experiment', 'subject'],
        columns='condition',
        values=metric,
        aggfunc='mean'
    ).reset_index()
    
    if 'hum' not in wide_df.columns or 'bot' not in wide_df.columns:
        return {'error': 'Missing condition columns', 'metric': metric}
    
    wide_df['condition_effect'] = wide_df['hum'] - wide_df['bot']
    
    exp1_data = wide_df[wide_df['experiment'] == 'Exp1_Human'].dropna(subset=['condition_effect'])
    exp2_data = wide_df[wide_df['experiment'] == 'Exp2_LLM'].dropna(subset=['condition_effect'])
    
    exp1_effects = exp1_data['condition_effect']
    exp2_effects = exp2_data['condition_effect']
    
    results = {
        'metric': metric,
        'n_exp1': len(exp1_effects),
        'n_exp2': len(exp2_effects),
        'exp1_effect_mean': exp1_effects.mean() if len(exp1_effects) > 0 else np.nan,
        'exp1_effect_sem': exp1_effects.std(ddof=1) / np.sqrt(len(exp1_effects)) if len(exp1_effects) > 1 else np.nan,
        'exp2_effect_mean': exp2_effects.mean() if len(exp2_effects) > 0 else np.nan,
        'exp2_effect_sem': exp2_effects.std(ddof=1) / np.sqrt(len(exp2_effects)) if len(exp2_effects) > 1 else np.nan,
    }
    
    results['exp1_hum_mean'] = exp1_data['hum'].mean() if len(exp1_data) > 0 else np.nan
    results['exp1_bot_mean'] = exp1_data['bot'].mean() if len(exp1_data) > 0 else np.nan
    results['exp2_hum_mean'] = exp2_data['hum'].mean() if len(exp2_data) > 0 else np.nan
    results['exp2_bot_mean'] = exp2_data['bot'].mean() if len(exp2_data) > 0 else np.nan
    
    exp1_sig = False
    exp2_sig = False
    
    if len(exp1_data) > 1:
        t1, p1 = ttest_rel(exp1_data['hum'], exp1_data['bot'])
        results['exp1_within_t'] = t1
        results['exp1_within_p'] = p1
        exp1_sig = p1 < 0.05
    
    if len(exp2_data) > 1:
        t2, p2 = ttest_rel(exp2_data['hum'], exp2_data['bot'])
        results['exp2_within_t'] = t2
        results['exp2_within_p'] = p2
        exp2_sig = p2 < 0.05
    
    if len(exp1_effects) > 1 and len(exp2_effects) > 1:
        t_stat, p_val = ttest_ind(exp1_effects, exp2_effects)
        results['interaction_t'] = t_stat
        results['interaction_p'] = p_val
        results['interaction_df'] = len(exp1_effects) + len(exp2_effects) - 2
    
    # Determine direction
    if not exp1_sig:
        exp1_dir = 'ns'
    elif results['exp1_effect_mean'] > 0:
        exp1_dir = 'Human > AI'
    else:
        exp1_dir = 'AI > Human'
    
    if not exp2_sig:
        exp2_dir = 'ns'
    elif results['exp2_effect_mean'] > 0:
        exp2_dir = 'Human > AI'
    else:
        exp2_dir = 'AI > Human'
    
    results['exp1_direction'] = exp1_dir
    results['exp2_direction'] = exp2_dir
    
    # Pattern classification
    if exp1_dir == exp2_dir:
        results['pattern'] = 'Both ns' if exp1_dir == 'ns' else 'Similar'
    elif exp1_dir == 'ns' or exp2_dir == 'ns':
        results['pattern'] = 'Different'
    elif (exp1_dir == 'Human > AI' and exp2_dir == 'AI > Human') or \
         (exp1_dir == 'AI > Human' and exp2_dir == 'Human > AI'):
        results['pattern'] = 'Flipped'
    else:
        results['pattern'] = 'Different'
    
    return results


# ============================================================
#                    OUTPUT FORMATTING
# ============================================================

def format_results(within_exp1, within_exp2, cross_results, n_exp1, n_exp2):
    """Format all results as text output."""
    lines = []
    lines.append("=" * 100)
    lines.append("INDIVIDUAL DISCOURSE MARKER ANALYSIS")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 100)
    lines.append(f"\nExp1 (Human): N = {n_exp1} subjects")
    lines.append(f"Exp2 (LLM):   N = {n_exp2} subjects")
    
    # ================================================================
    # MARKER REFERENCE
    # ================================================================
    lines.append("\n" + "-" * 100)
    lines.append("INDIVIDUAL MARKERS ANALYZED (from original filler list)")
    lines.append("-" * 100)
    lines.append("""
Filled Pauses (Clark & Fox Tree, 2002):
    um, uh, er, ah

Discourse Connectives (Schiffrin, 1987):
    well, so

Interactional Phrases (Fox Tree & Schrock, 2002):
    you know, I mean

Quotative/Discourse marker:
    like

Pragmatic Markers (Aijmer, 2002):
    basically, actually, right, okay
""")
    
    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    lines.append("\n" + "=" * 120)
    lines.append("SUMMARY TABLE: Condition Effects by Individual Marker")
    lines.append("=" * 120)
    lines.append(f"\n{'Marker':<20} {'Human (Exp1)':<22} {'LLM (Exp2)':<22} {'Cross-Exp p':<15} {'Pattern':<12}")
    lines.append("-" * 120)
    
    for r in cross_results:
        if 'error' in r:
            lines.append(f"{r['metric']:<20} ERROR: {r['error']}")
            continue
        
        metric = r['metric']
        
        int_p = f"{r.get('interaction_p', np.nan):.4f}" if 'interaction_p' in r else 'N/A'
        int_sig = ''
        if 'interaction_p' in r and not np.isnan(r.get('interaction_p', np.nan)):
            if r['interaction_p'] < 0.001: int_sig = '***'
            elif r['interaction_p'] < 0.01: int_sig = '**'
            elif r['interaction_p'] < 0.05: int_sig = '*'
        
        exp1_sig = '*' if r.get('exp1_within_p', 1) < 0.05 else ''
        exp2_sig = '*' if r.get('exp2_within_p', 1) < 0.05 else ''
        
        lines.append(f"{metric:<20} {r['exp1_direction'] + exp1_sig:<22} {r['exp2_direction'] + exp2_sig:<22} {int_p + int_sig:<15} {r['pattern']:<12}")
    
    lines.append("-" * 120)
    lines.append("Within-exp significance: * p < .05 (paired t-test)")
    lines.append("Cross-exp significance: * p < .05, ** p < .01, *** p < .001 (independent t-test)")
    
    # ================================================================
    # DETAILED RESULTS BY MARKER
    # ================================================================
    lines.append("\n\n" + "=" * 100)
    lines.append("DETAILED RESULTS BY MARKER")
    lines.append("=" * 100)
    
    # Group results by metric
    metrics_seen = set()
    for r in within_exp1 + within_exp2:
        if r and 'metric' in r:
            metrics_seen.add(r['metric'])
    
    for metric in sorted(metrics_seen):
        lines.append(f"\n{'='*80}")
        lines.append(f"MARKER: {metric}")
        lines.append(f"{'='*80}")
        
        # Find results for this metric
        exp1_r = next((r for r in within_exp1 if r and r.get('metric') == metric), None)
        exp2_r = next((r for r in within_exp2 if r and r.get('metric') == metric), None)
        cross_r = next((r for r in cross_results if r.get('metric') == metric), None)
        
        # Exp1 results
        lines.append(f"\n--- Exp1 (Human) ---")
        if exp1_r and 'error' not in exp1_r:
            lines.append(f"  N = {exp1_r['n_subjects']}")
            lines.append(f"  Human-label: M = {exp1_r.get('hum_mean', np.nan):.6f} ± {exp1_r.get('hum_sem', np.nan):.6f}")
            lines.append(f"  AI-label:    M = {exp1_r.get('bot_mean', np.nan):.6f} ± {exp1_r.get('bot_sem', np.nan):.6f}")
            
            sig_c = '***' if exp1_r.get('condition_p', 1) < 0.001 else ('**' if exp1_r.get('condition_p', 1) < 0.01 else ('*' if exp1_r.get('condition_p', 1) < 0.05 else ''))
            lines.append(f"  Condition:   F(1,{exp1_r['n_subjects']-1}) = {exp1_r.get('condition_F', np.nan):.2f}, p = {exp1_r.get('condition_p', np.nan):.4f} {sig_c}")
            
            sig_s = '***' if exp1_r.get('sociality_p', 1) < 0.001 else ('**' if exp1_r.get('sociality_p', 1) < 0.01 else ('*' if exp1_r.get('sociality_p', 1) < 0.05 else ''))
            lines.append(f"  Sociality:   F = {exp1_r.get('sociality_F', np.nan):.2f}, p = {exp1_r.get('sociality_p', np.nan):.4f} {sig_s}")
            
            sig_i = '***' if exp1_r.get('interaction_p', 1) < 0.001 else ('**' if exp1_r.get('interaction_p', 1) < 0.01 else ('*' if exp1_r.get('interaction_p', 1) < 0.05 else ''))
            lines.append(f"  Interaction: F = {exp1_r.get('interaction_F', np.nan):.2f}, p = {exp1_r.get('interaction_p', np.nan):.4f} {sig_i}")
            
            if 'simple_effects' in exp1_r:
                lines.append(f"  Simple Effects:")
                for soc_level in ['social', 'nonsocial']:
                    se = exp1_r['simple_effects'].get(soc_level, {})
                    if 'error' not in se:
                        sig_se = '*' if se.get('p_val', 1) < 0.05 else ''
                        lines.append(f"    {soc_level}: t({se.get('n',0)-1}) = {se.get('t_stat', np.nan):.2f}, p = {se.get('p_val', np.nan):.4f} {sig_se} [{se.get('direction', '')}]")
        elif exp1_r:
            lines.append(f"  ERROR: {exp1_r.get('error', 'Unknown error')}")
        else:
            lines.append(f"  No results")
        
        # Exp2 results
        lines.append(f"\n--- Exp2 (LLM) ---")
        if exp2_r and 'error' not in exp2_r:
            lines.append(f"  N = {exp2_r['n_subjects']}")
            lines.append(f"  Human-label: M = {exp2_r.get('hum_mean', np.nan):.6f} ± {exp2_r.get('hum_sem', np.nan):.6f}")
            lines.append(f"  AI-label:    M = {exp2_r.get('bot_mean', np.nan):.6f} ± {exp2_r.get('bot_sem', np.nan):.6f}")
            
            sig_c = '***' if exp2_r.get('condition_p', 1) < 0.001 else ('**' if exp2_r.get('condition_p', 1) < 0.01 else ('*' if exp2_r.get('condition_p', 1) < 0.05 else ''))
            lines.append(f"  Condition:   F(1,{exp2_r['n_subjects']-1}) = {exp2_r.get('condition_F', np.nan):.2f}, p = {exp2_r.get('condition_p', np.nan):.4f} {sig_c}")
            
            sig_s = '***' if exp2_r.get('sociality_p', 1) < 0.001 else ('**' if exp2_r.get('sociality_p', 1) < 0.01 else ('*' if exp2_r.get('sociality_p', 1) < 0.05 else ''))
            lines.append(f"  Sociality:   F = {exp2_r.get('sociality_F', np.nan):.2f}, p = {exp2_r.get('sociality_p', np.nan):.4f} {sig_s}")
            
            sig_i = '***' if exp2_r.get('interaction_p', 1) < 0.001 else ('**' if exp2_r.get('interaction_p', 1) < 0.01 else ('*' if exp2_r.get('interaction_p', 1) < 0.05 else ''))
            lines.append(f"  Interaction: F = {exp2_r.get('interaction_F', np.nan):.2f}, p = {exp2_r.get('interaction_p', np.nan):.4f} {sig_i}")
            
            if 'simple_effects' in exp2_r:
                lines.append(f"  Simple Effects:")
                for soc_level in ['social', 'nonsocial']:
                    se = exp2_r['simple_effects'].get(soc_level, {})
                    if 'error' not in se:
                        sig_se = '*' if se.get('p_val', 1) < 0.05 else ''
                        lines.append(f"    {soc_level}: t({se.get('n',0)-1}) = {se.get('t_stat', np.nan):.2f}, p = {se.get('p_val', np.nan):.4f} {sig_se} [{se.get('direction', '')}]")
        elif exp2_r:
            lines.append(f"  ERROR: {exp2_r.get('error', 'Unknown error')}")
        else:
            lines.append(f"  No results")
        
        # Cross-experiment
        lines.append(f"\n--- Cross-Experiment ---")
        if cross_r and 'error' not in cross_r:
            lines.append(f"  Exp1 effect (Hum-AI): {cross_r['exp1_effect_mean']:+.6f} [{cross_r['exp1_direction']}]")
            lines.append(f"  Exp2 effect (Hum-AI): {cross_r['exp2_effect_mean']:+.6f} [{cross_r['exp2_direction']}]")
            
            if 'interaction_t' in cross_r:
                sig = '***' if cross_r['interaction_p'] < 0.001 else ('**' if cross_r['interaction_p'] < 0.01 else ('*' if cross_r['interaction_p'] < 0.05 else ''))
                lines.append(f"  Interaction: t({cross_r['interaction_df']}) = {cross_r['interaction_t']:.3f}, p = {cross_r['interaction_p']:.4f} {sig}")
            
            lines.append(f"  Pattern: {cross_r['pattern']}")
        elif cross_r:
            lines.append(f"  ERROR: {cross_r.get('error', 'Unknown error')}")
        else:
            lines.append(f"  No results")
    
    return '\n'.join(lines)


# ============================================================
#                    MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("INDIVIDUAL DISCOURSE MARKER ANALYSIS")
    print("(v3 - matching cross_experiment_comparison_v4.py conventions)")
    print("=" * 80 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    topic_df = load_topics_file(TOPICS_PATH)
    
    # Load data
    if EXP1_USE_INDIVIDUAL_FILES:
        exp1_df = load_exp1_from_individual_files(
            subject_dir=EXP1_SUBJECT_DIR,
            subject_ids=EXP1_SUBJECT_IDS,
            file_pattern=EXP1_FILE_PATTERN,
            topic_df=topic_df
        )
    else:
        exp1_df = load_exp1_from_combined_file(EXP1_COMBINED_PATH, topic_df)
    
    exp2_df = load_exp2_data(EXP2_PATH, topic_df)
    
    n_exp1 = exp1_df['subject'].nunique()
    n_exp2 = exp2_df['subject'].nunique()
    
    # Compute individual marker metrics
    exp1_df = compute_individual_markers(exp1_df)
    exp2_df = compute_individual_markers(exp2_df)
    
    combined_utterance = pd.concat([exp1_df, exp2_df], ignore_index=True)
    
    # Define metrics to analyze
    # Individual markers
    individual_rate_metrics = [f"{m}_rate" for m in INDIVIDUAL_MARKERS.keys()]
    
    # Category subtotals
    category_rate_metrics = [f"{cat}_rate" for cat in MARKER_CATEGORIES.keys()]
    
    # Composite
    composite_metrics = ['filler_composite_rate']
    
    all_metrics = individual_rate_metrics + category_rate_metrics + composite_metrics
    
    print(f"\n[INFO] Analyzing {len(all_metrics)} metrics:")
    print(f"  Individual: {individual_rate_metrics}")
    print(f"  Categories: {category_rate_metrics}")
    print(f"  Composite:  {composite_metrics}")
    
    # Aggregate to subject level
    utt_social = aggregate_utterances_to_subject_social(combined_utterance, all_metrics)
    utt_subject = aggregate_utterances_to_subject(combined_utterance, all_metrics)
    
    print(f"\n[INFO] Aggregated data:")
    print(f"  Subject × Condition × Sociality: {len(utt_social)} rows")
    print(f"  Subject × Condition: {len(utt_subject)} rows")
    
    # Run analyses
    print("\n[INFO] Running RM-ANOVAs...")
    within_exp1 = []
    within_exp2 = []
    for m in all_metrics:
        r1 = run_within_experiment_anova(utt_social, m, 'Exp1_Human')
        r2 = run_within_experiment_anova(utt_social, m, 'Exp2_LLM')
        if r1: within_exp1.append(r1)
        if r2: within_exp2.append(r2)
    
    print("[INFO] Running cross-experiment comparisons...")
    cross_results = [run_cross_experiment_analysis(utt_subject, m) for m in all_metrics]
    
    # Format and save output
    output_text = format_results(within_exp1, within_exp2, cross_results, n_exp1, n_exp2)
    
    output_path = os.path.join(OUTPUT_DIR, "individual_marker_analysis.txt")
    with open(output_path, 'w') as f:
        f.write(output_text)
    print(f"\n[SAVED] {output_path}")
    
    # Save CSVs
    utt_social.to_csv(os.path.join(OUTPUT_DIR, "individual_marker_subject_condition_social.csv"), index=False)
    utt_subject.to_csv(os.path.join(OUTPUT_DIR, "individual_marker_subject_condition.csv"), index=False)
    
    # Save ANOVA results as CSV
    anova_rows = []
    for r in within_exp1 + within_exp2:
        if r and 'error' not in r:
            anova_rows.append({
                'experiment': r['experiment'],
                'metric': r['metric'],
                'n_subjects': r['n_subjects'],
                'hum_mean': r.get('hum_mean'),
                'bot_mean': r.get('bot_mean'),
                'condition_F': r.get('condition_F'),
                'condition_p': r.get('condition_p'),
                'sociality_F': r.get('sociality_F'),
                'sociality_p': r.get('sociality_p'),
                'interaction_F': r.get('interaction_F'),
                'interaction_p': r.get('interaction_p'),
            })
    
    if anova_rows:
        pd.DataFrame(anova_rows).to_csv(os.path.join(OUTPUT_DIR, "individual_marker_anova_results.csv"), index=False)
    
    # Save cross-experiment results as CSV
    cross_rows = []
    for r in cross_results:
        if 'error' not in r:
            cross_rows.append(r)
    
    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(os.path.join(OUTPUT_DIR, "individual_marker_cross_exp.csv"), index=False)
    
    print(f"\n[SAVED] CSV files to {OUTPUT_DIR}/")
    
    # Print summary table to console
    print("\n" + "=" * 100)
    print("SUMMARY: Condition Effects by Individual Marker")
    print("=" * 100)
    print(f"{'Marker':<20} {'Human (Exp1)':<20} {'LLM (Exp2)':<20} {'Cross-Exp p':<15} {'Pattern':<12}")
    print("-" * 100)
    
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
        
        exp1_sig = '*' if r.get('exp1_within_p', 1) < 0.05 else ''
        exp2_sig = '*' if r.get('exp2_within_p', 1) < 0.05 else ''
        
        print(f"{metric:<20} {r['exp1_direction'] + exp1_sig:<20} {r['exp2_direction'] + exp2_sig:<20} {int_p + int_sig:<15} {r['pattern']:<12}")
    
    print("-" * 100)
    print("\n[DONE] ✅ Analysis complete.")


if __name__ == "__main__":
    main()
