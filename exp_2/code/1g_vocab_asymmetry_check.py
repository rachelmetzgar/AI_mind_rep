"""
Vocabulary asymmetry analysis for Experiment 2 naturalistic conversations.

Checks whether human-partner vs AI-partner conversations show vocabulary
differences in emotion, mental-state, embodiment, and formal/technical words
that could drive lexical overlap concerns.

Analyzes the participant LLM's utterances (### Assistant: turns) separately
from the full conversation text.

NOTE: This is a one-off analysis script originally written for the "names"
variant (combined_all.csv from naturalistic conversations). It may not apply
to other versions which use synthetic Exp 1 data.
"""

import sys
import argparse
import pandas as pd
import re
from pathlib import Path
from collections import Counter

# ── Central config ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import config as cfg, set_version, add_version_argument, add_model_argument, set_model


# ── Extract participant (Assistant) utterances ─────────────────────────────
def extract_assistant_text(full_text):
    """Pull out only the ### Assistant: portions of a conversation."""
    parts = re.split(r'###\s*(User|Assistant):\s*', full_text)
    # parts alternates: preamble, role, text, role, text, ...
    assistant_chunks = []
    for i in range(1, len(parts) - 1, 2):
        role = parts[i]
        text = parts[i + 1]
        if role == "Assistant":
            assistant_chunks.append(text.strip())
    return " ".join(assistant_chunks)


def extract_user_text(full_text):
    """Pull out only the ### User: portions of a conversation."""
    parts = re.split(r'###\s*(User|Assistant):\s*', full_text)
    user_chunks = []
    for i in range(1, len(parts) - 1, 2):
        role = parts[i]
        text = parts[i + 1]
        if role == "User":
            user_chunks.append(text.strip())
    return " ".join(user_chunks)


# ── Word categories ────────────────────────────────────────────────────────
CATEGORIES = {
    "Emotion": [
        "feel", "feeling", "emotion", "happy", "sad", "angry",
        "fear", "joy", "love", "pain", "hurt", "grief",
        "hope", "anxious", "excited",
    ],
    "Mental-state": [
        "think", "believe", "know", "understand", "realize",
        "imagine", "wonder", "remember", "conscious", "aware",
        "mind", "thought",
    ],
    "Embodiment/physical": [
        "body", "physical", "touch", "sense", "hand",
        "eye", "face", "skin", "muscle", "brain",
    ],
    "Formal/technical": [
        "system", "algorithm", "data", "process", "function",
        "compute", "analyze", "optimize", "parameter", "module",
    ],
    "Identity labels": [
        "human", "person", "ai", "artificial", "machine", "robot",
    ],
}

# ── Count helper ───────────────────────────────────────────────────────
def tokenize(text):
    """Lowercase and split on non-alpha to get word tokens."""
    return re.findall(r"[a-z]+", text.lower())

def count_words(tokens, word_list):
    """Count occurrences of each word in the token list."""
    token_counts = Counter(tokens)
    return {w: token_counts.get(w, 0) for w in word_list}


# ── Views ──────────────────────────────────────────────────────────────
VIEWS = [
    ("PARTICIPANT (Assistant) UTTERANCES", "assistant"),
    ("PROMPT (User) UTTERANCES", "user"),
    ("FULL CONVERSATION", "full"),
]


# ── Main ───────────────────────────────────────────────────────────────
def main(data_path):
    """Run vocabulary asymmetry analysis on combined conversation CSV."""

    # ── Load data ──────────────────────────────────────────────────────
    df = pd.read_csv(data_path)

    print(f"Data path: {data_path}")
    print(f"Loaded {len(df)} conversations  (human-partner: {(df['label']=='human').sum()}, "
          f"AI-partner: {(df['label']=='ai').sum()})")
    print("=" * 80)

    # Build pooled text for each condition × role
    conditions = {"human": df[df["label"] == "human"], "ai": df[df["label"] == "ai"]}

    pooled = {}
    for cond_name, cond_df in conditions.items():
        pooled[f"{cond_name}_assistant"] = " ".join(cond_df["text"].apply(extract_assistant_text))
        pooled[f"{cond_name}_user"]      = " ".join(cond_df["text"].apply(extract_user_text))
        pooled[f"{cond_name}_full"]      = " ".join(cond_df["text"])

    # ── Run analysis ───────────────────────────────────────────────────
    # We'll report three views: Assistant only, User only, Full conversation
    for view_label, view_key in VIEWS:
        print(f"\n{'=' * 80}")
        print(f"  {view_label}")
        print(f"{'=' * 80}")

        human_tokens = tokenize(pooled[f"human_{view_key}"])
        ai_tokens    = tokenize(pooled[f"ai_{view_key}"])

        n_human = len(human_tokens)
        n_ai    = len(ai_tokens)
        print(f"\nTotal words:  human-partner = {n_human:,}   AI-partner = {n_ai:,}")
        print(f"Word-count ratio (human/AI): {n_human / n_ai:.3f}")

        for cat_name, word_list in CATEGORIES.items():
            print(f"\n── {cat_name} words ──")
            human_counts = count_words(human_tokens, word_list)
            ai_counts    = count_words(ai_tokens, word_list)

            cat_total_human = sum(human_counts.values())
            cat_total_ai    = sum(ai_counts.values())

            # per 10k normalization
            norm_human = cat_total_human / n_human * 10_000
            norm_ai    = cat_total_ai / n_ai * 10_000
            ratio = norm_human / norm_ai if norm_ai > 0 else float("inf")

            print(f"  Category total (raw):  human-partner = {cat_total_human}   AI-partner = {cat_total_ai}")
            print(f"  Per 10k words:         human-partner = {norm_human:.2f}   AI-partner = {norm_ai:.2f}")
            print(f"  Ratio (human / AI):    {ratio:.3f}")

            # Individual word breakdown
            print(f"  {'Word':<14} {'Human (raw)':>12} {'AI (raw)':>12} "
                  f"{'Human/10k':>10} {'AI/10k':>10} {'Ratio':>8}")
            for w in word_list:
                h = human_counts[w]
                a = ai_counts[w]
                h_norm = h / n_human * 10_000
                a_norm = a / n_ai * 10_000
                r = h_norm / a_norm if a_norm > 0 else float("inf")
                flag = " ***" if abs(r - 1.0) > 0.5 and (h + a) >= 10 else ""
                print(f"  {w:<14} {h:>12,} {a:>12,} {h_norm:>10.2f} {a_norm:>10.2f} {r:>8.3f}{flag}")

        # ── Vocabulary overlap statistics ──────────────────────────────
        print(f"\n── Vocabulary overlap ──")
        human_vocab = set(human_tokens)
        ai_vocab    = set(ai_tokens)
        shared      = human_vocab & ai_vocab
        only_human  = human_vocab - ai_vocab
        only_ai     = ai_vocab - human_vocab
        print(f"  Unique word types:  human-partner = {len(human_vocab):,}   AI-partner = {len(ai_vocab):,}")
        print(f"  Shared types:       {len(shared):,}")
        print(f"  Only in human-partner: {len(only_human):,}")
        print(f"  Only in AI-partner:    {len(only_ai):,}")
        print(f"  Jaccard similarity:    {len(shared) / len(human_vocab | ai_vocab):.4f}")

    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print("""
Ratios close to 1.0 mean the word/category appears at similar rates in both
conditions → vocabulary is NOT driving probe results through lexical overlap.

Ratios far from 1.0 (with substantial raw counts) would indicate a vocabulary
asymmetry that COULD contribute to lexical overlap concerns.

'***' flags words with ratio > 1.5 or < 0.5 AND at least 10 total occurrences.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vocabulary asymmetry check for naturalistic conversations."
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to combined_all.csv. If not provided, defaults to the "
             "'names' variant location."
    )
    # Version is optional — this script predates the multi-version setup
    # and only applies to variants that have a combined_all.csv
    parser.add_argument(
        "--version", type=str, default=None,
        choices=("labels", "balanced_names", "balanced_gpt", "names",
                 "nonsense_codeword", "nonsense_ignore"),
        help="Optional data version (calls set_version for config paths)."
    )
    add_model_argument(parser)

    args = parser.parse_args()

    # Set version if provided (populates config paths)
    set_model(args.model)
    if args.version is not None:
        set_version(args.version)

    # Resolve data path
    if args.data_path is not None:
        data_path = args.data_path
    else:
        # Default: names variant combined_all.csv
        # NOTE: This path assumes the names/ subdirectory still exists under
        # exp_2/. If that directory is archived or moved, update this default
        # or always pass --data_path explicitly.
        data_path = str(
            cfg.ROOT_DIR / "names" / "llama_exp_2b-13B-chat" / "combined_all.csv"
        )
        print(f"WARNING: No --data_path provided. Defaulting to names variant:\n"
              f"  {data_path}\n"
              f"Pass --data_path /path/to/combined_all.csv to use a different file.\n")

    main(data_path)
