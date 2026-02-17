"""
Script name: clean_transcripts.py
Purpose: Clean per-subject transcript CSVs and save cleaned versions alongside
         the originals. Original files are never modified.

Cleaning steps (in order):
    1. Replace partner names (Casey, Sam, ChatGPT, Copilot, Gemini, + invented
       names and condition-skewed address terms) with "Partner".
    2. Strip emoji characters.
    3. Remove asterisk-delimited emotes (*action*).
    4. Remove identity-leaking phrases ("As an AI", "language model", etc.)
       — phrase-only removal, rest of sentence preserved.
    5. Remove meta-narration openers ("Sure! Here's my first message:", etc.)
    6. Drop stuck-loop turns (consecutive duplicate turns within a trial).

All-caps text is intentionally preserved.

Inputs:
    - Per-subject transcript CSVs in DATA_DIR/<model>/<temp>/<sub_id>.csv
    - Subject list from config (subject_ids)
Outputs:
    - Cleaned CSVs saved as <sub_id>_clean.csv in the same directory
    - Run log in output directory
    - Config snapshot

Author: Rachel C. Metzgar
Date: 2026-02-10
"""
from __future__ import annotations

import os
import re
from typing import List

import pandas as pd

from utils.globals import DATA_DIR
from utils.cli_helpers import parse_and_load_config
from utils.run_logger import init_run

SCRIPT_NAME = "clean_transcripts"

# ── Cleaning constants ────────────────────────────────────────────────────────

# Partner names used across the experiment (human-label and AI-label)
# plus names invented by the partner LLM and condition-skewed address terms.
PARTNER_NAMES = [
    # Original partner names
    "Casey", "Sam", "ChatGPT", "Copilot", "Gemini",
    # Person names invented by partner LLM
    "Alex", "Rachel", "Emily", "Sarah", "Lauren", "Samantha",
    "Olivia", "Emma", "Summer", "Hayley", "Hannah", "Anna",
    "Lily", "Alexa", "Priya", "Adrian", "Amelia", "Kara",
    "Ashley", "Rach", "Luna", "Kitty",
    # Role/label artifacts
    "Pilot", "Navigator", "Participant", "Commander",
    "Comrade", "Traveler", "Captain", "Flighty",
    # Condition-skewed address terms
    "Girl", "Dude",
]
_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in PARTNER_NAMES) + r")\b",
    flags=re.IGNORECASE,
)

# Broad Unicode emoji ranges (covers emoticons, symbols, flags, etc.)
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U0000FE00-\U0000FE0F"
    "\U0000200D"
    "\U000020E3"
    "\U00002600-\U000026FF"
    "\U00002300-\U000023FF"
    "]+",
    flags=re.UNICODE,
)

# Asterisk-delimited emotes, e.g. *excitedly*, *nods vigorously*
_EMOTE_RE = re.compile(r"\*[^*]+\*")

# ── Identity leakage patterns ────────────────────────────────────────────────
# These are PHRASE-ONLY removals: the phrase is deleted but the rest of the
# sentence is kept, to avoid inflating word-count differences between conditions.

_IDENTITY_PHRASES = [
    # "As an AI, ..." / "As a human, ..." — remove the phrase + trailing comma
    r"As an AI(?:\s+language model)?(?:\s+assistant)?,?\s*",
    r"As a human(?:\s+being)?,?\s*",
    # Standalone identity declarations
    r"I'?m (?:just )?an? AI(?:\s+language model)?(?:\s+assistant)?,?\s*",
    r"I am (?:just )?an? AI(?:\s+language model)?(?:\s+assistant)?,?\s*",
    r"I'?m (?:just )?a (?:large )?(?:language model|chatbot|chat bot|virtual assistant),?\s*",
    r"I am (?:just )?a (?:large )?(?:language model|chatbot|chat bot|virtual assistant),?\s*",
    r"I'?m not a (?:real )?(?:person|human),?\s*",
    r"I'?m (?:just )?a human,?\s*",
    r"I am (?:just )?a human,?\s*",
    r"(?:B|b)eing an AI,?\s*",
    r"(?:B|b)eing a human,?\s*",
    # "I don't have feelings/emotions/experiences/preferences"
    r"I don'?t have (?:personal )?(?:feelings|emotions|experiences|preferences),?\s*",
    # Catch remaining "language model" and "chatbot" as standalone noun phrases
    # but only when preceded by "a" to avoid false positives
    r"\ba (?:large )?language model\b",
    r"\ba chatbot\b",
]
_IDENTITY_RE = [re.compile(p, flags=re.IGNORECASE) for p in _IDENTITY_PHRASES]

# ── Meta-narration opener patterns ────────────────────────────────────────────
# Formulaic LLM openers that are condition-skewed (more common in AI condition).
# These are removed entirely — they add no conversational content.

_META_NARRATION_PATTERNS = [
    # "Sure! Here's my first message:" and variants
    r"(?:Sure|Great|Of course|Absolutely|Okay)[!,.]?\s*"
    r"(?:I'?d be happy to (?:start|chat|help|engage|have)[^.!?\n]*[.!]?\s*)?"
    r"(?:Here(?:'s| is) my (?:first|next|final)?\s*(?:message|response|question|reply)"
    r"(?:\s+(?:as|to|for|in)\s+[^:.\n]*)?\s*:?\s*)",
    # "Here's my first message:" without the Sure/Great prefix
    r"Here(?:'s| is) my (?:first|next|final)?\s*(?:message|response|question|reply)"
    r"(?:\s+(?:as|to|for|in)\s+[^:.\n]*)?\s*:?\s*",
    # "Sure, I'd be happy to start/chat!" without the Here's part
    r"(?:Sure|Great|Of course|Absolutely)[!,.]?\s*"
    r"I'?d be happy to (?:start|chat|help|engage|have)[^.!?\n]*[.!]\s*",
    # "Sure! Let me start:" / "Sure! I'll start:"
    r"(?:Sure|Great|Of course|Absolutely)[!,.]?\s*"
    r"(?:Let me|I'?ll)\s+(?:start|begin)[^.!?\n]*[.!:]\s*",
]
_META_RE = [re.compile(p, flags=re.IGNORECASE) for p in _META_NARRATION_PATTERNS]


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _collapse_spaces(text: str) -> str:
    """Collapse runs of whitespace left behind by removals."""
    return re.sub(r"  +", " ", text).strip()


def _replace_names(text: str) -> str:
    """Replace partner names with the generic token 'Partner'."""
    if not isinstance(text, str):
        return text
    return _NAME_RE.sub("Partner", text)


def _strip_emoji(text: str) -> str:
    """Remove emoji characters."""
    if not isinstance(text, str):
        return text
    return _collapse_spaces(_EMOJI_RE.sub("", text))


def _strip_emotes(text: str) -> str:
    """Remove asterisk-delimited emotes like *excitedly*."""
    if not isinstance(text, str):
        return text
    return _collapse_spaces(_EMOTE_RE.sub("", text))


def _scrub_identity_leakage(text: str) -> str:
    """Remove identity-leaking phrases, keeping the rest of each sentence."""
    if not isinstance(text, str):
        return text
    for pattern in _IDENTITY_RE:
        text = pattern.sub("", text)
    # Capitalize the first letter after removal if we left a lowercase start
    # e.g., "As an AI, i think..." → "i think..." → "I think..."
    text = re.sub(r"(?:^|\.\s+)([a-z])", lambda m: m.group(0).upper(), text)
    return _collapse_spaces(text)


def _scrub_meta_narration(text: str) -> str:
    """Remove formulaic LLM conversation openers."""
    if not isinstance(text, str):
        return text
    for pattern in _META_RE:
        text = pattern.sub("", text)
    return _collapse_spaces(text)


def _remove_stuck_loops(df: pd.DataFrame) -> pd.DataFrame:
    """Drop turns where the conversation got stuck in a repetition loop.

    Within each (subject, run, trial) group sorted by pair_index, a row is
    flagged as stuck if its transcript_sub OR transcript_llm is identical to
    the immediately preceding turn.  Stuck rows are dropped and a summary is
    printed to stdout.
    """
    required = {"subject", "run", "trial", "pair_index", "transcript_sub", "transcript_llm"}
    if not required.issubset(df.columns):
        print("[WARN] Cannot detect stuck loops — missing columns: "
              f"{sorted(required - set(df.columns))}. Skipping.")
        return df

    df = df.sort_values(["subject", "run", "trial", "pair_index"]).copy()
    group_keys = ["subject", "run", "trial"]

    sub_prev = df.groupby(group_keys)["transcript_sub"].shift(1)
    llm_prev = df.groupby(group_keys)["transcript_llm"].shift(1)

    is_stuck = (df["transcript_sub"] == sub_prev) | (df["transcript_llm"] == llm_prev)

    n_stuck = is_stuck.sum()
    if n_stuck > 0:
        stuck_rows = df.loc[is_stuck, ["subject", "run", "trial", "pair_index"]]
        print(f"[INFO] Removing {n_stuck} stuck-loop turn(s):")
        for _, row in stuck_rows.iterrows():
            print(f"       {row['subject']}  run={row['run']}  "
                  f"trial={row['trial']}  pair_index={row['pair_index']}")

    return df.loc[~is_stuck].reset_index(drop=True)


# ── Main cleaning entry point ─────────────────────────────────────────────────

def clean_subject_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a single subject's transcript DataFrame.

    Steps (in order):
        1. Replace partner names with "Partner" in both transcript columns.
        2. Strip emoji characters.
        3. Remove asterisk-delimited emotes (*action*).
        4. Scrub identity-leaking phrases (phrase-only, preserves sentence).
        5. Scrub meta-narration openers.
        6. Drop stuck-loop turns (exact-duplicate consecutive turns).

    All-caps text is intentionally preserved.
    """
    transcript_cols = ["transcript_sub", "transcript_llm"]

    # 1. Partner-name scrubbing
    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_replace_names)

    # 2. Emoji removal
    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_strip_emoji)

    # 3. Emote removal
    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_strip_emotes)

    # 4. Identity leakage scrubbing
    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_scrub_identity_leakage)

    # 5. Meta-narration scrubbing
    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_scrub_meta_narration)

    # 6. Stuck-loop removal (operates on rows, not cell text)
    df = _remove_stuck_loops(df)

    return df


def clean_all_subjects(
    sub_ids: List[str],
    data_dir: str,
) -> dict:
    """Clean each subject's transcript CSV and save as <sub_id>_clean.csv.

    Parameters
    ----------
    sub_ids : list[str]
        Subject identifiers to process.
    data_dir : str
        Directory containing ``<sub_id>.csv`` files.  Cleaned files are
        written to the same directory as ``<sub_id>_clean.csv``.

    Returns
    -------
    dict
        Per-subject summary: ``{sub_id: {"rows_before": int, "rows_after": int}}``.
    """
    summary = {}

    for sub_id in sub_ids:
        csv_path = os.path.join(data_dir, f"{sub_id}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] No file found for {sub_id} at {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        n_before = len(df)

        print(f"[INFO] Cleaning {sub_id} ({n_before} rows) ...")
        df_clean = clean_subject_transcript(df.copy())
        n_after = len(df_clean)

        out_path = os.path.join(data_dir, f"{sub_id}_clean.csv")
        df_clean.to_csv(out_path, index=False)

        print(f"[INFO] {sub_id}: {n_before} → {n_after} rows "
              f"({n_before - n_after} stuck-loop turns removed) → {out_path}")

        summary[sub_id] = {"rows_before": n_before, "rows_after": n_after}

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    # --- Parse CLI and load config ---
    args, cfg = parse_and_load_config("Clean per-subject transcript CSVs")

    # get analysis info
    subjects = [str(s) for s in cfg["subject_ids"]]
    model = cfg.get("model")
    temp = cfg.get("temperature")

    # --- Setup directories ---
    data_dir = os.path.join(DATA_DIR, model, str(temp))
    out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Initialize logging ---
    logger, seed, overwrite, dry_run = init_run(
        output_dir=out_dir,
        script_name=SCRIPT_NAME,
        args=args,
        cfg=cfg,
        used_alias=False,
    )

    # --- Clean ---
    summary = clean_all_subjects(subjects, data_dir)

    total_before = sum(s["rows_before"] for s in summary.values())
    total_after = sum(s["rows_after"] for s in summary.values())
    logger.info(
        f"Cleaned {len(summary)} subjects: {total_before} → {total_after} total rows "
        f"({total_before - total_after} stuck-loop turns removed)"
    )


if __name__ == "__main__":
    main()