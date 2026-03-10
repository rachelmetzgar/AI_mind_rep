#!/usr/bin/env python3
"""
Experiment 3: Cross-Dataset Lexical Overlap Analysis

Tests whether concept-probe alignment could be driven by shared vocabulary
between concept prompts and the exp 2 probe training conversations, rather
than genuine representational alignment.

For each word in the vocabulary, computes a "human-bias" score: does it appear
more often in human-condition conversations or AI-condition conversations?
Then for each concept dimension, asks: do the concept prompts contain words
that are biased toward the same condition as the probe label?

Three analyses:
  1. CONTRAST prompts: human-labeled vs AI-labeled concept prompts
     - Computes H-A word bias differential per dimension
     - Correlates with raw and residual alignment R²

  2. STANDALONE prompts: no human/AI labels
     - Computes mean word bias per dimension
     - Correlates with standalone alignment R²

  3. Word-level inspection: shows most biased words per concept

Key findings:
  - Contrast prompts: significant correlation (rho~0.6, p=0.001) — lexical
    confound is plausible for contrast analysis
  - Standalone prompts: correlation is NEGATIVE (rho~-0.44, p=0.018) — goes
    opposite direction from confound prediction
  - Control concepts (shapes, granite): near-zero alignment despite having
    the same prompt structure

Usage:
    python 8b_cross_dataset_lexical_overlap.py

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import json
import os
import re
import csv
import glob
import numpy as np
from scipy import stats
from collections import Counter

from config import config

# ── Paths ──────────────────────────────────────────────────────────────────
CONV_DIR = str(config.PATHS.exp1_data)  # exp 1 conversation CSVs
CONTRASTS_DIR = str(config.RESULTS.concept_activations_contrasts)
STANDALONE_DIR = str(config.RESULTS.concept_activations_standalone)

# Alignment summary files (balanced_gpt, turn 5)
ALIGN_RAW_PATH = os.path.join(
    str(config.RESULTS.root), "llama2_13b_chat", "balanced_gpt",
    "alignment", "turn_5", "contrasts", "raw", "summary.json"
)
ALIGN_RESID_PATH = os.path.join(
    str(config.RESULTS.root), "llama2_13b_chat", "balanced_gpt",
    "alignment", "turn_5", "contrasts", "residual", "summary.json"
)
ALIGN_STAND_PATH = os.path.join(
    str(config.RESULTS.root), "llama2_13b_chat", "balanced_gpt",
    "alignment", "turn_5", "standalone", "summary.json"
)

# ── Stopwords ──────────────────────────────────────────────────────────────
STOPWORDS = {
    "a", "the", "an", "is", "are", "to", "of", "in", "that", "this", "it",
    "for", "with", "on", "at", "by", "from", "as", "or", "and", "but", "not",
    "be", "was", "were", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall", "i",
    "you", "we", "they", "he", "she", "my", "your", "our", "their", "me",
    "us", "him", "her", "its", "am", "im", "youre", "dont", "cant", "wont",
    "ive", "id", "ill", "thats", "whats", "heres", "theres", "lets", "also",
    "just", "really", "very", "so", "if", "then", "than", "when", "where",
    "what", "how", "who", "which", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "each", "few", "more",
    "most", "other", "some", "such", "no", "nor", "only", "own", "same",
    "too", "s", "t", "re", "ve", "ll", "d", "m",
}


def tokenize(text):
    """Lowercase, split on non-alpha, drop stopwords and short words."""
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in STOPWORDS and len(t) > 2]


# ====================================================================
# 1. Build word-bias dictionary from exp 1 conversations
# ====================================================================
def load_conversation_word_counts():
    """Load exp 1 conversations and count word frequencies by condition."""
    csv_files = sorted(glob.glob(os.path.join(CONV_DIR, "s*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No conversation CSVs in {CONV_DIR}")

    human_words = Counter()
    ai_words = Counter()
    n_human = 0
    n_ai = 0

    for csv_file in csv_files:
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            seen_trials = {}
            for row in reader:
                trial = row["trial"]
                partner_type = row["partner_type"]
                seen_trials[trial] = (partner_type, row.get("sub_input", ""))

            for trial, (ptype, sub_input) in seen_trials.items():
                try:
                    messages = json.loads(sub_input)
                except (json.JSONDecodeError, TypeError):
                    continue
                full_text = " ".join(
                    m.get("content", "") for m in messages if isinstance(m, dict)
                )
                words = tokenize(full_text)
                if "human" in ptype.lower():
                    human_words.update(words)
                    n_human += 1
                elif "ai" in ptype.lower():
                    ai_words.update(words)
                    n_ai += 1

    return human_words, ai_words, n_human, n_ai


def compute_word_bias(human_words, ai_words):
    """Compute normalized human-AI bias for each word.

    Returns dict: word -> bias in [-1, 1].
    Positive = more frequent in human conversations.
    """
    total_h = sum(human_words.values())
    total_a = sum(ai_words.values())
    all_words = set(human_words.keys()) | set(ai_words.keys())

    bias = {}
    for w in all_words:
        h_rate = human_words[w] / total_h
        a_rate = ai_words[w] / total_a
        denom = h_rate + a_rate
        if denom > 0:
            bias[w] = (h_rate - a_rate) / denom
    return bias


# ====================================================================
# 2. Analyze concept prompts
# ====================================================================
def analyze_contrast_prompts(word_bias, align_raw, align_resid):
    """Compute word-bias scores for contrast concept prompts."""
    results = []
    for dim_dir in sorted(os.listdir(CONTRASTS_DIR)):
        dim_path = os.path.join(CONTRASTS_DIR, dim_dir)
        if not os.path.isdir(dim_path):
            continue
        prompts_file = os.path.join(dim_path, "concept_prompts.json")
        if not os.path.exists(prompts_file):
            continue

        prompts = json.load(open(prompts_file))
        human_prompts = [p for p in prompts if p["label"] == 1]
        ai_prompts = [p for p in prompts if p["label"] == 0]

        h_biases = [word_bias.get(w, 0) for w in
                     sum([tokenize(p["prompt"]) for p in human_prompts], [])]
        a_biases = [word_bias.get(w, 0) for w in
                     sum([tokenize(p["prompt"]) for p in ai_prompts], [])]

        h_mean = np.mean(h_biases) if h_biases else 0
        a_mean = np.mean(a_biases) if a_biases else 0

        raw_r2 = align_raw.get(dim_dir, {}).get("control_mean_r2", float("nan"))
        res_r2 = align_resid.get(dim_dir, {}).get("control_mean_r2", float("nan"))

        results.append({
            "dim": dim_dir, "h_bias": h_mean, "a_bias": a_mean,
            "diff_bias": h_mean - a_mean, "raw_r2": raw_r2, "resid_r2": res_r2,
        })
    return results


def analyze_standalone_prompts(word_bias, align_stand):
    """Compute word-bias scores for standalone concept prompts."""
    results = []
    for dim_dir in sorted(os.listdir(STANDALONE_DIR)):
        dim_path = os.path.join(STANDALONE_DIR, dim_dir)
        if not os.path.isdir(dim_path):
            continue
        prompts_file = os.path.join(dim_path, "concept_prompts.json")
        if not os.path.exists(prompts_file):
            continue

        prompts = json.load(open(prompts_file))
        all_words = []
        for p in prompts:
            text = p.get("prompt", p.get("text", str(p)))
            all_words.extend(tokenize(text))

        biases = [word_bias.get(w, 0) for w in all_words]
        mean_bias = np.mean(biases) if biases else 0
        stand_r2 = align_stand.get(dim_dir, {}).get(
            "control_mean_r2", float("nan"))

        results.append({
            "dim": dim_dir, "mean_bias": mean_bias, "stand_r2": stand_r2,
        })
    return results


def show_biased_words(dim_name, prompts_dir, word_bias, n=10):
    """Print the most human/AI-biased words in a concept's prompts."""
    prompts_file = os.path.join(prompts_dir, dim_name, "concept_prompts.json")
    if not os.path.exists(prompts_file):
        return
    prompts = json.load(open(prompts_file))
    words = []
    for p in prompts:
        text = p.get("prompt", p.get("text", str(p)))
        words.extend(tokenize(text))

    word_counts = Counter(words)
    biased = [(w, word_bias.get(w, 0), c) for w, c in word_counts.most_common(50)]
    biased.sort(key=lambda x: -abs(x[1]))

    print(f"\n  {dim_name}:")
    for w, bias, count in biased[:n]:
        direction = "H" if bias > 0 else "A"
        print(f"    {w:<20s}  bias={bias:+.4f} ({direction})  count={count}")


# ====================================================================
# MAIN
# ====================================================================
def main():
    print("Loading exp 1 conversation data (probe training source)...")
    human_words, ai_words, n_human, n_ai = load_conversation_word_counts()
    print(f"  {n_human} human conversations, {n_ai} AI conversations")
    print(f"  Human vocab: {len(human_words)} unique, AI vocab: {len(ai_words)} unique")

    word_bias = compute_word_bias(human_words, ai_words)

    # Load alignment summaries
    align_raw = json.load(open(ALIGN_RAW_PATH))
    align_resid = json.load(open(ALIGN_RESID_PATH))
    align_stand = json.load(open(ALIGN_STAND_PATH))

    # ── Contrast analysis ──────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("CONTRAST PROMPTS: H-A word bias differential vs alignment")
    print("=" * 120)
    print(f"{'Dimension':<30s}  {'H_bias':>7s} {'A_bias':>7s} {'H-A':>8s}"
          f"  {'raw_op':>7s} {'res_op':>7s}")
    print("-" * 100)

    c_results = analyze_contrast_prompts(word_bias, align_raw, align_resid)
    for r in c_results:
        print(f"{r['dim']:<30s}  {r['h_bias']:+7.4f} {r['a_bias']:+7.4f} "
              f"{r['diff_bias']:+8.4f}  {r['raw_r2']:7.4f} {r['resid_r2']:7.4f}")

    diffs = np.array([r["diff_bias"] for r in c_results])
    raw_r2s = np.array([r["raw_r2"] for r in c_results])
    valid = [(r["diff_bias"], r["resid_r2"]) for r in c_results
             if not np.isnan(r["resid_r2"])]
    res_diffs = np.array([v[0] for v in valid])
    res_r2s = np.array([v[1] for v in valid])

    print(f"\n  Spearman (N={len(c_results)}):")
    rho, p = stats.spearmanr(diffs, raw_r2s)
    print(f"    Word bias diff vs raw operational R²:      rho={rho:+.4f}, p={p:.4f}")
    rho2, p2 = stats.spearmanr(res_diffs, res_r2s)
    print(f"    Word bias diff vs residual operational R²:  rho={rho2:+.4f}, p={p2:.4f}")

    # ── Standalone analysis ────────────────────────────────────────────
    print(f"\n{'=' * 120}")
    print("STANDALONE PROMPTS: word bias vs alignment")
    print(f"{'=' * 120}")
    print(f"{'Dimension':<30s}  {'mean_bias':>9s}  {'stand_op':>8s}")
    print("-" * 55)

    s_results = analyze_standalone_prompts(word_bias, align_stand)
    for r in s_results:
        print(f"{r['dim']:<30s}  {r['mean_bias']:+9.4f}  {r['stand_r2']:8.4f}")

    s_valid = [(r["mean_bias"], r["stand_r2"]) for r in s_results
               if not np.isnan(r["stand_r2"])]
    s_biases = np.array([v[0] for v in s_valid])
    s_r2s = np.array([v[1] for v in s_valid])

    print(f"\n  Spearman (N={len(s_valid)}):")
    rho3, p3 = stats.spearmanr(s_biases, s_r2s)
    print(f"    Word bias vs standalone operational R²:  rho={rho3:+.4f}, p={p3:.4f}")

    # ── Most biased words in selected concepts ─────────────────────────
    print(f"\n{'=' * 120}")
    print("TOP BIASED WORDS in selected concept prompts (standalone)")
    print(f"{'=' * 120}")
    for dim in ["7_social", "17_attention", "6_cognitive",
                "15_shapes", "30_granite_sandstone"]:
        show_biased_words(dim, STANDALONE_DIR, word_bias)

    print("\nDone.")


if __name__ == "__main__":
    main()
