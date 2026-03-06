"""
Script name: 0_clean_transcripts.py
Purpose: Clean per-subject transcript CSVs and save cleaned versions alongside
         the originals. Original files are never modified.

Only the `names` version currently has _clean.csv files. Other versions were
analyzed without cleaning. This script is kept for reference and optional
future use. It is NOT integrated into the run_analysis.sh pipeline.

Cleaning steps (in order):
    1. Replace partner names (version-aware, pulled from config agent_map)
       plus common invented names and condition-skewed address terms with
       "Partner".
    2. Strip emoji characters.
    3. Remove asterisk-delimited emotes (*action*).
    4. Remove meta-narration openers ("Sure! Here's my first message:", etc.)
    5. Drop stuck-loop turns (consecutive duplicate turns within a trial).

NOTE: Identity-leaking phrases (e.g. "As an AI", "I'm a language model") are
intentionally NOT scrubbed. These reflect genuine behavioral differences between
conditions and are part of the signal under study.

Usage:
    cd /mnt/cup/labs/graziano/rachel/mind_rep/exp_1
    python code/analysis/0_clean_transcripts.py --version names --clean

Author: Rachel C. Metzgar
Date: 2026-02-10
Refactored: 2026-03-06 — ported to unified config.py structure
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from code.config import (
    parse_version_model, get_version_config, data_dir, VERSIONS,
    SUBJECT_IDS,
)

# ── Cleaning constants ────────────────────────────────────────────────────────

# Common names invented by the partner LLM and condition-skewed address terms.
# Version-specific partner names are added dynamically from config.
_EXTRA_NAMES = [
    "Alex", "Rachel", "Emily", "Sarah", "Lauren", "Samantha",
    "Olivia", "Emma", "Summer", "Hayley", "Hannah", "Anna",
    "Lily", "Alexa", "Priya", "Adrian", "Amelia", "Kara",
    "Ashley", "Rach", "Luna", "Kitty", "Gemini",
    # Role/label artifacts
    "Pilot", "Navigator", "Participant", "Commander",
    "Comrade", "Traveler", "Captain", "Flighty",
    # Condition-skewed address terms
    "Girl", "Dude",
]

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

# ── Meta-narration opener patterns ────────────────────────────────────────────
_META_NARRATION_PATTERNS = [
    r"(?:Sure|Great|Of course|Absolutely|Okay)[!,.]?\s*"
    r"(?:I'?d be happy to (?:start|chat|help|engage|have)[^.!?\n]*[.!]?\s*)?"
    r"(?:Here(?:'s| is) my (?:first|next|final)?\s*(?:message|response|question|reply)"
    r"(?:\s+(?:as|to|for|in)\s+[^:.\n]*)?\s*:?\s*)",
    r"Here(?:'s| is) my (?:first|next|final)?\s*(?:message|response|question|reply)"
    r"(?:\s+(?:as|to|for|in)\s+[^:.\n]*)?\s*:?\s*",
    r"(?:Sure|Great|Of course|Absolutely)[!,.]?\s*"
    r"I'?d be happy to (?:start|chat|help|engage|have)[^.!?\n]*[.!]\s*",
    r"(?:Sure|Great|Of course|Absolutely)[!,.]?\s*"
    r"(?:Let me|I'?ll)\s+(?:start|begin)[^.!?\n]*[.!:]\s*",
]
_META_RE = [re.compile(p, flags=re.IGNORECASE) for p in _META_NARRATION_PATTERNS]


def _build_name_regex(version: str) -> re.Pattern:
    """Build a name-replacement regex that includes version-specific partner names."""
    names = list(_EXTRA_NAMES)
    agent_map = VERSIONS[version]["agent_map"]
    for info in agent_map.values():
        name = info.get("name")
        if name and name not in names:
            names.append(name)
    return re.compile(
        r"\b(" + "|".join(re.escape(n) for n in names) + r")\b",
        flags=re.IGNORECASE,
    )


# ── Cleaning helpers ──────────────────────────────────────────────────────────

def _collapse_spaces(text: str) -> str:
    return re.sub(r"  +", " ", text).strip()


def _replace_names(text: str, name_re: re.Pattern) -> str:
    if not isinstance(text, str):
        return text
    return name_re.sub("Partner", text)


def _strip_emoji(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _collapse_spaces(_EMOJI_RE.sub("", text))


def _strip_emotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    return _collapse_spaces(_EMOTE_RE.sub("", text))


def _scrub_meta_narration(text: str) -> str:
    if not isinstance(text, str):
        return text
    for pattern in _META_RE:
        text = pattern.sub("", text)
    return _collapse_spaces(text)


def _remove_stuck_loops(df: pd.DataFrame) -> pd.DataFrame:
    required = {"subject", "run", "trial", "pair_index", "transcript_sub", "transcript_llm"}
    if not required.issubset(df.columns):
        print(f"[WARN] Cannot detect stuck loops — missing columns: "
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

def clean_subject_transcript(df: pd.DataFrame, name_re: re.Pattern) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a single subject's transcript DataFrame."""
    transcript_cols = ["transcript_sub", "transcript_llm"]

    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda t: _replace_names(t, name_re))

    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_strip_emoji)

    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_strip_emotes)

    for col in transcript_cols:
        if col in df.columns:
            df[col] = df[col].apply(_scrub_meta_narration)

    df = _remove_stuck_loops(df)
    return df


def clean_all_subjects(
    sub_ids: List[str],
    data_path: str,
    name_re: re.Pattern,
) -> dict:
    """Clean each subject's transcript CSV and save as <sub_id>_clean.csv."""
    summary = {}

    for sub_id in sub_ids:
        csv_path = os.path.join(data_path, f"{sub_id}.csv")
        if not os.path.exists(csv_path):
            print(f"[WARN] No file found for {sub_id} at {csv_path}. Skipping.")
            continue

        df = pd.read_csv(csv_path, on_bad_lines="skip")
        n_before = len(df)

        print(f"[INFO] Cleaning {sub_id} ({n_before} rows) ...")
        df_clean = clean_subject_transcript(df.copy(), name_re)
        n_after = len(df_clean)

        out_path = os.path.join(data_path, f"{sub_id}_clean.csv")
        df_clean.to_csv(out_path, index=False)

        print(f"[INFO] {sub_id}: {n_before} → {n_after} rows "
              f"({n_before - n_after} stuck-loop turns removed) → {out_path}")

        summary[sub_id] = {"rows_before": n_before, "rows_after": n_after}

    return summary


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean per-subject transcript CSVs (partner names, emoji, emotes, meta-narration, stuck loops)."
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Actually write _clean.csv files. Without this flag, does a dry run (prints what would be cleaned).",
    )
    args = parse_version_model(parser)

    version_cfg = get_version_config()
    dd = str(data_dir())
    version = args.version

    name_re = _build_name_regex(version)

    print(f"=== Transcript Cleaning: {version} ===")
    print(f"Data dir: {dd}")
    print(f"Clean mode: {'WRITE' if args.clean else 'DRY RUN (use --clean to write)'}")
    print()

    if not args.clean:
        # Dry run: just report what would happen
        for sub_id in SUBJECT_IDS:
            csv_path = os.path.join(dd, f"{sub_id}.csv")
            if os.path.exists(csv_path):
                print(f"  Would clean: {csv_path} → {sub_id}_clean.csv")
        print(f"\n{len(SUBJECT_IDS)} subjects. Rerun with --clean to write files.")
        return

    summary = clean_all_subjects(SUBJECT_IDS, dd, name_re)

    total_before = sum(s["rows_before"] for s in summary.values())
    total_after = sum(s["rows_after"] for s in summary.values())
    print(f"\nCleaned {len(summary)} subjects: {total_before} → {total_after} total rows "
          f"({total_before - total_after} stuck-loop turns removed)")


if __name__ == "__main__":
    main()
