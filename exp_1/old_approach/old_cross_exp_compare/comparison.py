#!/usr/bin/env python3
"""
Script name: cross_experiment_comparison_v4.py
Purpose: Compare behavioral measures between Human (Exp1) and LLM (Exp2) experiments.

UPDATED v4: Added Fung & Carter (2007) discourse markers analysis
Based on: Fung, L., & Carter, R. (2007). Discourse markers and spoken English: 
Native and learner use in pedagogic settings. Applied Linguistics, 28(3), 410-439.

The Fung & Carter framework categorizes discourse markers into four functional categories:
1. INTERPERSONAL: markers for shared knowledge, attitudes, and responses
2. REFERENTIAL: markers indicating textual relationships (cause, contrast, etc.)
3. STRUCTURAL: markers for discourse organization (topic shifts, sequence, etc.)
4. COGNITIVE: markers for thinking processes, reformulation, hesitation

NEW METRICS (v4):
- fung_interpersonal_rate: Interpersonal discourse markers / word_count
- fung_referential_rate: Referential discourse markers / word_count
- fung_structural_rate: Structural discourse markers / word_count
- fung_cognitive_rate: Cognitive discourse markers / word_count
- fung_total_rate: All Fung & Carter discourse markers / word_count

EXISTING METRICS (v3):
- LIWC2007 nonfluencies and fillers (Pennebaker et al., 2007)
- Original filler markers for comparison

Usage:
    python cross_experiment_comparison_v4.py

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
    # Prefer 's##' format over 'P##'
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
OUTPUT_DIR = "cross_experiment_results"
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
#                    LINGUISTIC MARKERS
# ============================================================

# ============================================================
# FUNG & CARTER (2007) DISCOURSE MARKERS
# Source: Applied Linguistics 28(3): 410-439
# Based on Table 1 (functional paradigm) and Table 2 (frequency analysis)
# ============================================================

# The 23 discourse markers analyzed in Fung & Carter (2007), Table 2
# Organized by the four functional categories from Table 1

# INTERPERSONAL CATEGORY
# Functions: marking shared knowledge, indicating attitudes, showing responses
# Markers: you know, you see, see, listen, well, really, I think, obviously,
#          absolutely, basically, actually, exactly, sort of, kind of, like,
#          just, oh, OK/okay, right/alright, yeah, yes, I see, great, sure
FUNG_INTERPERSONAL = [
    # Marking shared knowledge
    r"\byou know\b",
    r"\byou see\b",
    r"\bsee\b",              # Note: can be verb, context-dependent
    r"\blisten\b",
    # Indicating attitudes (stance markers)
    r"\bwell\b",
    r"\breally\b",
    r"\bi think\b",
    r"\bobviously\b",
    r"\babsolutely\b",
    r"\bbasically\b",
    r"\bactually\b",
    r"\bexactly\b",
    r"\bsort of\b",
    r"\bkind of\b",
    r"\blike\b",             # Note: highly multifunctional
    r"\bjust\b",
    r"\boh\b",
    # Showing responses (acknowledgement/agreement)
    r"\bokay\b",
    r"\bok\b",
    r"\bright\b",
    r"\balright\b",
    r"\byeah\b",
    r"\byes\b",
    r"\bi see\b",
    r"\bgreat\b",
    r"\bsure\b",
]

# REFERENTIAL CATEGORY
# Functions: indicating relationships between utterances
# Markers: because/cos (cause), but/and/yet/however/nevertheless (contrast),
#          and (coordination), or (disjunction), so (consequence),
#          anyway (digression), likewise/similarly (comparison)
FUNG_REFERENTIAL = [
    # Cause
    r"\bbecause\b",
    r"\bcos\b",
    r"\bcause\b",            # Informal spelling
    # Contrast
    r"\bbut\b",
    r"\byet\b",
    r"\bhowever\b",
    r"\bnevertheless\b",
    # Coordination
    r"\band\b",
    # Disjunction
    r"\bor\b",
    # Consequence
    r"\bso\b",
    # Digression
    r"\banyway\b",
    r"\banyways\b",
    # Comparison
    r"\blikewise\b",
    r"\bsimilarly\b",
]

# STRUCTURAL CATEGORY
# Functions: organizing discourse, marking boundaries, topic management
# Markers: now, OK, right, well (opening/closing), first/firstly/second/next/then/finally (sequence),
#          so/now/well/and what about/how about (topic shifts), so (summarizing),
#          yeah/and/cos/so (continuation)
FUNG_STRUCTURAL = [
    # Opening and closing of topics
    r"\bnow\b",
    r"\bokay\b",
    r"\bok\b",
    r"\bright\b",
    r"\balright\b",
    r"\bwell\b",
    r"\blet's start\b",
    r"\blet's discuss\b",
    r"\blet me conclude\b",
    # Sequence markers
    r"\bfirst\b",
    r"\bfirstly\b",
    r"\bsecond\b",
    r"\bsecondly\b",
    r"\bthird\b",
    r"\bthirdly\b",
    r"\bnext\b",
    r"\bthen\b",
    r"\bfinally\b",
    # Topic shifts
    r"\bso\b",
    r"\band what about\b",
    r"\bhow about\b",
    r"\bwhat about\b",
    # Continuation markers (also structural)
    r"\byeah\b",
    r"\band\b",
    r"\bcos\b",
]

# COGNITIVE CATEGORY
# Functions: denoting thinking, reformulation, elaboration, hesitation
# Markers: well/I think/I see/and (thinking), I mean/that is/in other words (reformulation),
#          like/I mean (elaboration), well/sort of (hesitation), you know (assessment)
FUNG_COGNITIVE = [
    # Denoting thinking process
    r"\bwell\b",
    r"\bi think\b",
    r"\bi see\b",
    # Reformulation/Self-correction
    r"\bi mean\b",
    r"\bthat is\b",
    r"\bin other words\b",
    r"\bwhat i mean is\b",
    r"\bto put it another way\b",
    # Elaboration
    r"\blike\b",
    # Hesitation
    r"\bsort of\b",
    r"\bkind of\b",
    # Assessment of listener's knowledge
    r"\byou know\b",
]

# COMPLETE LIST: All 23 markers from Table 2 (for total count)
# These are the markers specifically analyzed in the frequency comparison
FUNG_ALL_23_MARKERS = [
    r"\band\b",
    r"\bso\b",
    r"\byeah\b",
    r"\bright\b",
    r"\bbut\b",
    r"\bor\b",
    r"\bjust\b",
    r"\bokay\b",
    r"\bok\b",
    r"\blike\b",
    r"\byou know\b",
    r"\bwell\b",
    r"\bbecause\b",
    r"\bnow\b",
    r"\byes\b",
    r"\bsort of\b",
    r"\bsee\b",
    r"\bi think\b",
    r"\bi mean\b",
    r"\bsay\b",
    r"\bactually\b",
    r"\boh\b",
    r"\breally\b",
    r"\bcos\b",
]


# ============================================================
# LIWC2007 SPOKEN CATEGORIES (Pennebaker et al., 2007)
# ============================================================

LIWC_NONFLUENCIES = [
    r"\ber\b",
    r"\bhm+\b",
    r"\bsigh\b",
    r"\buh\b",
    r"\bum\b",
    r"\bumm+\b",
    r"\bwell\b",
]

LIWC_FILLERS = [
    r"\bblah\b",
    r"\bi\s*don'?t\s*know\b",
    r"\bi\s*mean\b",
    r"\boh\s*well\b",
    r"\bor\s*anything\w*\b",
    r"\bor\s*something\w*\b",
    r"\bor\s*whatever\w*\b",
    r"\bya\s*know\w*\b",
    r"\by'know\w*\b",
    r"\byou\s*know\w*\b",
]

LIWC_DISFLUENCIES = LIWC_NONFLUENCIES + LIWC_FILLERS

# ============================================================
# ORIGINAL FILLER MARKERS (kept for comparison)
# ============================================================
FILLER_MARKERS_ORIGINAL = [
    r"\bum\b", r"\buh\b", r"\ber\b", r"\bah\b",
    r"\blike\b", r"\byou know\b", r"\bi mean\b",
    r"\bwell\b", r"\bbasically\b", r"\bactually\b",
    r"\bright\b", r"\bokay\b", r"\bso\b"
]

# ============================================================
# OTHER LINGUISTIC MARKERS (unchanged from v3)
# ============================================================

HEDGE_MARKERS = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bprobably\b", r"\bmight\b",
    r"\bcould be\b", r"\bit seems\b", r"\bi think\b",
    r"\bin a way\b", r"\bsort of\b", r"\bkind of\b",
    r"\bmore or less\b", r"\broughly\b", r"\btends to\b"
]

TOM_PHRASES = [
    r"\byou think\b", r"\byou believe\b", r"\byou know\b", r"\byou feel\b",
    r"\byou understand\b", r"\byou guess\b", r"\byou imagine\b",
    r"\byou wonder\b", r"\byou consider\b", r"\byou expect\b",
    r"\byou hope\b", r"\byou assume\b", r"\byou realize\b",
    r"\byou remember\b", r"\byou forget\b"
]

POLITE_POSITIVE = [
    r"\bthank(s| you|ful)?\b",
    r"\bappreciate\b",
    r"\b(great|wonderful|fantastic|awesome|excellent)\b",
    r"\b(hey|hello|hi)\b",
]

POLITE_NEGATIVE = [
    r"\bsorry\b",
    r"\bplease\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bmight you\b",
    r"\bif you could\b",
    r"\bperhaps\b",
    r"\bby any chance\b",
]

IMPOLITE = [
    r"\byou need to\b",
    r"\byou should\b",
    r"\bdo not\b",
    r"\bin fact\b",
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
        raise ValueError("No valid data found for Exp1 (Human)")
    
    df_combined = pd.concat(all_trials, ignore_index=True)
    df_combined = get_social_classification(df_combined, topic_df)
    df_combined['experiment'] = 'Exp1_Human'
    
    print(f"  Loaded {len(df_combined)} rows (utterances) from {len(loaded_subjects)} subjects")
    print(f"  Conditions: {df_combined['condition'].value_counts().to_dict()}")
    
    if 'quality' in df_combined.columns and df_combined['quality'].notna().any():
        print(f"  Quality ratings: {df_combined['quality'].notna().sum()} non-null values")
    if 'connectedness' in df_combined.columns and df_combined['connectedness'].notna().any():
        print(f"  Connectedness ratings: {df_combined['connectedness'].notna().sum()} non-null values")
    
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
    
    if 'Quality' in df.columns:
        df['quality'] = pd.to_numeric(df['Quality'], errors='coerce')
    if 'Connectedness' in df.columns:
        df['connectedness'] = pd.to_numeric(df['Connectedness'], errors='coerce')
    
    df.columns = df.columns.str.strip().str.lower()
    
    if 'quality' in df.columns:
        df['quality'] = pd.to_numeric(df['quality'], errors='coerce')
    if 'connectedness' in df.columns:
        df['connectedness'] = pd.to_numeric(df['connectedness'], errors='coerce')
    
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

def count_patterns(text, patterns):
    """Count occurrences of regex patterns in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def count_unique_patterns(text, patterns):
    """
    Count occurrences of regex patterns, avoiding double-counting.
    For multi-word patterns that may overlap with single-word patterns.
    """
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    
    # Sort patterns by length (longest first) to prioritize multi-word matches
    sorted_patterns = sorted(patterns, key=lambda p: len(p), reverse=True)
    
    total = 0
    # Create a copy of text to mark matched regions
    matched_positions = set()
    
    for pattern in sorted_patterns:
        for match in re.finditer(pattern, text_lower):
            # Check if this position was already matched by a longer pattern
            positions = set(range(match.start(), match.end()))
            if not positions & matched_positions:  # No overlap
                total += 1
                matched_positions |= positions
    
    return total


def compute_all_metrics(df):
    """
    Compute all linguistic metrics for each UTTERANCE (row).
    
    NEW in v4: Fung & Carter (2007) discourse marker categories
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
    # FUNG & CARTER (2007) DISCOURSE MARKERS (NEW in v4)
    # ============================================================
    
    # Count each functional category
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
    
    # Total count using the specific 23 markers from Table 2
    df['fung_total_count'] = df['transcript_sub'].apply(
        lambda x: count_patterns(x, FUNG_ALL_23_MARKERS)
    )
    
    # Compute rates (normalized by word count)
    df['fung_interpersonal_rate'] = df['fung_interpersonal_count'] / df['word_count'].replace(0, np.nan)
    df['fung_referential_rate'] = df['fung_referential_count'] / df['word_count'].replace(0, np.nan)
    df['fung_structural_rate'] = df['fung_structural_count'] / df['word_count'].replace(0, np.nan)
    df['fung_cognitive_rate'] = df['fung_cognitive_count'] / df['word_count'].replace(0, np.nan)
    df['fung_total_rate'] = df['fung_total_count'] / df['word_count'].replace(0, np.nan)
    
    print(f"  [OK] Fung & Carter discourse markers computed:")
    print(f"       Interpersonal: mean = {df['fung_interpersonal_count'].mean():.2f} per utterance")
    print(f"       Referential:   mean = {df['fung_referential_count'].mean():.2f} per utterance")
    print(f"       Structural:    mean = {df['fung_structural_count'].mean():.2f} per utterance")
    print(f"       Cognitive:     mean = {df['fung_cognitive_count'].mean():.2f} per utterance")
    print(f"       Total (23):    mean = {df['fung_total_count'].mean():.2f} per utterance")
    
    # ============================================================
    # LIWC-BASED DISFLUENCY MARKERS (from v3)
    # ============================================================
    
    df['nonfluency_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, LIWC_NONFLUENCIES))
    df['nonfluency_rate'] = df['nonfluency_count'] / df['word_count'].replace(0, np.nan)
    
    df['liwc_filler_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, LIWC_FILLERS))
    df['liwc_filler_rate'] = df['liwc_filler_count'] / df['word_count'].replace(0, np.nan)
    
    df['disfluency_count'] = df['nonfluency_count'] + df['liwc_filler_count']
    df['disfluency_rate'] = df['disfluency_count'] / df['word_count'].replace(0, np.nan)
    
    # ============================================================
    # ORIGINAL FILLER MARKERS (kept for comparison)
    # ============================================================
    df['filler_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, FILLER_MARKERS_ORIGINAL))
    df['filler_rate'] = df['filler_count'] / df['word_count'].replace(0, np.nan)
    
    # ============================================================
    # OTHER LINGUISTIC MARKERS
    # ============================================================
    
    df['hedge_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, HEDGE_MARKERS))
    df['hedge_rate'] = df['hedge_count'] / df['word_count'].replace(0, np.nan)
    
    df['tom_count'] = df['transcript_sub'].apply(lambda x: count_patterns(x, TOM_PHRASES))
    df['tom_rate'] = df['tom_count'] / df['word_count'].replace(0, np.nan)
    
    df['polite_pos'] = df['transcript_sub'].apply(lambda x: count_patterns(x, POLITE_POSITIVE))
    df['polite_neg'] = df['transcript_sub'].apply(lambda x: count_patterns(x, POLITE_NEGATIVE))
    df['impolite'] = df['transcript_sub'].apply(lambda x: count_patterns(x, IMPOLITE))
    df['politeness_score'] = df['polite_pos'] + df['polite_neg'] - df['impolite']
    df['politeness_rate'] = df['politeness_score'] / df['word_count'].replace(0, np.nan)
    
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
        'word_count', 'question_count', 'filler_count', 'hedge_count', 'tom_count', 
        'polite_pos', 'polite_neg', 'impolite', 'politeness_score',
        'nonfluency_count', 'liwc_filler_count', 'disfluency_count',
        # NEW in v4: Fung & Carter counts
        'fung_interpersonal_count', 'fung_referential_count', 
        'fung_structural_count', 'fung_cognitive_count', 'fung_total_count'
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
        ('filler_rate', 'filler_count'),
        ('hedge_rate', 'hedge_count'),
        ('tom_rate', 'tom_count'),
        ('politeness_rate', 'politeness_score'),
        ('nonfluency_rate', 'nonfluency_count'),
        ('liwc_filler_rate', 'liwc_filler_count'),
        ('disfluency_rate', 'disfluency_count'),
        # NEW in v4: Fung & Carter rates
        ('fung_interpersonal_rate', 'fung_interpersonal_count'),
        ('fung_referential_rate', 'fung_referential_count'),
        ('fung_structural_rate', 'fung_structural_count'),
        ('fung_cognitive_rate', 'fung_cognitive_count'),
        ('fung_total_rate', 'fung_total_count'),
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
#                    STATISTICAL ANALYSES
# ============================================================

def run_simple_effects(aov_df, metric):
    """Run simple effects tests: Condition effect at each level of Sociality."""
    from scipy import stats
    
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
        
        t_stat, p_val = stats.ttest_rel(hum_vals, bot_vals)
        
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
    """
    exp_df = df[df['experiment'] == experiment_name].copy()
    
    if exp_df.empty:
        return None
    
    aov_df = exp_df.copy()
    n_subjects = aov_df['subject'].nunique()
    n_cells = aov_df.groupby(['condition', 'social_type']).size()
    
    if n_subjects < 3:
        return {'error': f'Insufficient subjects: {n_subjects}'}
    
    if len(n_cells) < 4:
        return {'error': f'Missing cells: {n_cells.to_dict()}'}
    
    if aov_df[metric].isna().all():
        return {'error': f'All values are NaN for {metric}'}
    
    aov_df = aov_df.dropna(subset=[metric])
    
    complete_subjects = aov_df.groupby('subject').filter(lambda x: len(x) == 4)['subject'].unique()
    aov_df = aov_df[aov_df['subject'].isin(complete_subjects)]
    n_subjects = aov_df['subject'].nunique()
    
    if n_subjects < 3:
        return {'error': f'Insufficient complete subjects after filtering: {n_subjects}'}
    
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
        F_cond = MS_cond / MS_cond_subj
        p_cond = 1 - f_dist.cdf(F_cond, df_cond, df_cond_subj)
        
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
        F_social = MS_social / MS_social_subj
        p_social = 1 - f_dist.cdf(F_social, df_social, df_social_subj)
        
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
        F_inter = MS_inter / MS_inter_subj
        p_inter = 1 - f_dist.cdf(F_inter, df_inter, df_inter_subj)
        
        results['interaction_F'] = F_inter
        results['interaction_p'] = p_inter
        results['interaction_df'] = (df_inter, df_inter_subj)
        
        # Descriptive statistics
        cond_means = aov_df.groupby('condition')[metric].agg(['mean', 'sem'])
        results['hum_mean'] = cond_means.loc['hum', 'mean'] if 'hum' in cond_means.index else np.nan
        results['hum_sem'] = cond_means.loc['hum', 'sem'] if 'hum' in cond_means.index else np.nan
        results['bot_mean'] = cond_means.loc['bot', 'mean'] if 'bot' in cond_means.index else np.nan
        results['bot_sem'] = cond_means.loc['bot', 'sem'] if 'bot' in cond_means.index else np.nan
        
        social_means = aov_df.groupby('social_type')[metric].agg(['mean', 'sem'])
        results['social_mean'] = social_means.loc['social', 'mean'] if 'social' in social_means.index else np.nan
        results['social_sem'] = social_means.loc['social', 'sem'] if 'social' in social_means.index else np.nan
        results['nonsocial_mean'] = social_means.loc['nonsocial', 'mean'] if 'nonsocial' in social_means.index else np.nan
        results['nonsocial_sem'] = social_means.loc['nonsocial', 'sem'] if 'nonsocial' in social_means.index else np.nan
        
        cell_means = aov_df.groupby(['condition', 'social_type'])[metric].agg(['mean', 'sem'])
        results['cell_means'] = cell_means.to_dict()
        
        results['anova_table'] = pd.DataFrame({
            'Source': ['condition', 'social_type', 'condition:social_type'],
            'SS': [SS_cond, SS_social, SS_inter],
            'DF1': [df_cond, df_social, df_inter],
            'DF2': [df_cond_subj, df_social_subj, df_inter_subj],
            'MS': [MS_cond, MS_social, MS_inter],
            'MS_error': [MS_cond_subj, MS_social_subj, MS_inter_subj],
            'F': [F_cond, F_social, F_inter],
            'p': [p_cond, p_social, p_inter]
        })
        
        if results.get('interaction_p', 1.0) < 0.05:
            results['simple_effects'] = run_simple_effects(aov_df, metric)
        
    except Exception as e:
        return {'error': str(e)}
    
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
        return {'error': 'Missing condition columns'}
    
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
    lines.append("=" * 80)
    lines.append("CROSS-EXPERIMENT BEHAVIORAL ANALYSIS")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append(f"\nExp1 (Human): N = {n_exp1} subjects")
    lines.append(f"Exp2 (LLM):   N = {n_exp2} subjects")
    
    # ================================================================
    # FUNG & CARTER CITATION NOTE (NEW in v4)
    # ================================================================
    lines.append("\n" + "-" * 80)
    lines.append("NOTE: Discourse Marker Analysis Based on Fung & Carter (2007)")
    lines.append("-" * 80)
    lines.append("""
Fung, L., & Carter, R. (2007). Discourse markers and spoken English: Native and 
learner use in pedagogic settings. Applied Linguistics, 28(3), 410-439.

The Fung & Carter framework categorizes discourse markers into four functional categories:

1. INTERPERSONAL: Markers for shared knowledge, attitudes, and responses
   - Examples: you know, you see, well, really, I think, actually, sort of, 
     okay, right, yeah, yes, oh, just, basically

2. REFERENTIAL: Markers indicating textual relationships
   - Examples: because/cos (cause), but/and/yet/however (contrast), 
     or (disjunction), so (consequence), anyway (digression)

3. STRUCTURAL: Markers for discourse organization
   - Examples: now, okay, right, well (boundaries), first/then/finally (sequence),
     so (topic shifts), yeah/and (continuation)

4. COGNITIVE: Markers for thinking, reformulation, hesitation
   - Examples: well, I think, I see (thinking), I mean/that is (reformulation),
     like (elaboration), sort of (hesitation), you know (assessment)

Metrics computed:
  - fung_interpersonal_rate: Interpersonal markers / word_count
  - fung_referential_rate: Referential markers / word_count  
  - fung_structural_rate: Structural markers / word_count
  - fung_cognitive_rate: Cognitive markers / word_count
  - fung_total_rate: All 23 markers from Table 2 / word_count
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
    lines.append(f"\n{'Measure':<25} {'Human (Exp1)':<20} {'LLM (Exp2)':<20} {'Cross-Exp p':<15} {'Pattern':<12}")
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
        
        exp1_sig = '*' if r.get('exp1_within_p', 1) < 0.05 else ''
        exp2_sig = '*' if r.get('exp2_within_p', 1) < 0.05 else ''
        
        lines.append(f"{metric:<25} {r['exp1_direction'] + exp1_sig:<20} {r['exp2_direction'] + exp2_sig:<20} {int_p + int_sig:<15} {r['pattern']:<12}")
    
    lines.append("-" * 110)
    lines.append("Within-exp significance: * p < .05 (paired t-test)")
    lines.append("Cross-exp significance: * p < .05, ** p < .01, *** p < .001 (independent t-test on condition effects)")
    
    # ================================================================
    # WITHIN-EXPERIMENT DETAILED RESULTS
    # ================================================================
    lines.append("\n\n" + "=" * 80)
    lines.append("WITHIN-EXPERIMENT ANALYSES: 2×2 RM-ANOVA (Condition × Sociality)")
    lines.append("=" * 80)
    
    for r in within_exp1 + within_exp2:
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
        lines.append(f"  Sample sizes: Exp1 N = {r['n_exp1']}, Exp2 N = {r['n_exp2']}")
        lines.append(f"{'-' * 70}")
        
        lines.append(f"\n  Condition Means by Experiment:")
        lines.append(f"    Exp1 (Human): Human-label = {r.get('exp1_hum_mean', np.nan):.4f}, AI-label = {r.get('exp1_bot_mean', np.nan):.4f}")
        lines.append(f"    Exp2 (LLM):   Human-label = {r.get('exp2_hum_mean', np.nan):.4f}, AI-label = {r.get('exp2_bot_mean', np.nan):.4f}")
        
        lines.append(f"\n  Condition Effect (Human-label minus AI-label):")
        
        p1_str = f"  [t = {r.get('exp1_within_t', np.nan):.2f}, p = {r.get('exp1_within_p', np.nan):.4f}]" if 'exp1_within_p' in r else ""
        p2_str = f"  [t = {r.get('exp2_within_t', np.nan):.2f}, p = {r.get('exp2_within_p', np.nan):.4f}]" if 'exp2_within_p' in r else ""
        
        lines.append(f"    Exp1 (Human): M = {r['exp1_effect_mean']:+.4f} ± {r.get('exp1_effect_sem', np.nan):.4f}{p1_str}")
        lines.append(f"    Exp2 (LLM):   M = {r['exp2_effect_mean']:+.4f} ± {r.get('exp2_effect_sem', np.nan):.4f}{p2_str}")
        
        if 'interaction_t' in r:
            sig = '***' if r['interaction_p'] < 0.001 else ('**' if r['interaction_p'] < 0.01 else ('*' if r['interaction_p'] < 0.05 else ''))
            lines.append(f"\n  Cross-Experiment Interaction Test:")
            lines.append(f"    t({r['interaction_df']}) = {r['interaction_t']:.3f}, p = {r['interaction_p']:.4f} {sig}")
        
        lines.append(f"\n  Direction: Exp1 = {r['exp1_direction']}, Exp2 = {r['exp2_direction']}")
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
    
    print(f"\n  Fung & Carter Discourse Markers:")
    print(f"    Interpersonal: mean={df['fung_interpersonal_count'].mean():.3f}")
    print(f"    Referential:   mean={df['fung_referential_count'].mean():.3f}")
    print(f"    Structural:    mean={df['fung_structural_count'].mean():.3f}")
    print(f"    Cognitive:     mean={df['fung_cognitive_count'].mean():.3f}")
    print(f"    Total (23):    mean={df['fung_total_count'].mean():.3f}")
    
    print(f"\n  LIWC Disfluency Metrics:")
    print(f"    Nonfluency count: mean={df['nonfluency_count'].mean():.3f}")
    print(f"    LIWC Filler count: mean={df['liwc_filler_count'].mean():.3f}")
    print(f"    Combined disfluency count: mean={df['disfluency_count'].mean():.3f}")
    
    if 'sentiment' in df.columns:
        n_valid = df['sentiment'].notna().sum()
        if n_valid > 0:
            print(f"  Sentiment (compound): {n_valid} non-null values, mean={df['sentiment'].mean():.4f}")


# ============================================================
#                    MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("CROSS-EXPERIMENT ANALYSIS: Human vs LLM Behavioral Patterns")
    print("(v4 - with Fung & Carter 2007 discourse markers)")
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
    
    # Compute metrics
    exp1_df = compute_all_metrics(exp1_df)
    exp2_df = compute_all_metrics(exp2_df)
    
    print_diagnostics(exp1_df, "Exp1_Human")
    print_diagnostics(exp2_df, "Exp2_LLM")
    
    combined_utterance = pd.concat([exp1_df, exp2_df], ignore_index=True)
    
    # Define metrics
    count_metrics = ['word_count', 'question_count']
    
    rate_metrics = [
        # Fung & Carter discourse markers (NEW in v4)
        'fung_interpersonal_rate',
        'fung_referential_rate',
        'fung_structural_rate', 
        'fung_cognitive_rate',
        'fung_total_rate',
        # LIWC disfluencies
        'nonfluency_rate',
        'liwc_filler_rate',
        'disfluency_rate',
        # Original markers
        'filler_rate',
        'hedge_rate', 
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
    print("  NEW: fung_interpersonal_rate, fung_referential_rate, fung_structural_rate,")
    print("       fung_cognitive_rate, fung_total_rate (Fung & Carter 2007)")
    
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
    trial_within_exp1 = []
    trial_within_exp2 = []
    for m in all_metrics:
        r1 = run_within_experiment_anova(trial_social, m, 'Exp1_Human')
        r2 = run_within_experiment_anova(trial_social, m, 'Exp2_LLM')
        if r1: trial_within_exp1.append(r1)
        if r2: trial_within_exp2.append(r2)
    
    print("[INFO] Running per-trial cross-experiment comparisons...")
    trial_cross = [run_cross_experiment_analysis(trial_subject, m) for m in all_metrics]
    
    trial_output = format_results(trial_within_exp1, trial_within_exp2, trial_cross, n_exp1, n_exp2)
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
    utt_within_exp1 = []
    utt_within_exp2 = []
    for m in all_metrics:
        r1 = run_within_experiment_anova(utt_social, m, 'Exp1_Human')
        r2 = run_within_experiment_anova(utt_social, m, 'Exp2_LLM')
        if r1: utt_within_exp1.append(r1)
        if r2: utt_within_exp2.append(r2)
    
    print("[INFO] Running per-utterance cross-experiment comparisons...")
    utt_cross = [run_cross_experiment_analysis(utt_subject, m) for m in all_metrics]
    
    utt_output = format_results(utt_within_exp1, utt_within_exp2, utt_cross, n_exp1, n_exp2)
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