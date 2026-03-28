#!/usr/bin/env python3
"""
Experiment 4, Phase 1: Behavioral Replication of Gray et al. (2007)

Direct replication of the Dimensions of Mind Perception study using
LLaMA-2-13B (chat or base) as a participant. The model rates pairwise
comparisons of 12 entities on 18 mental capacities, closely matching
the original survey procedure from Gray, Gray, & Wegner (2007, Science).

Supports two model variants:
    - Chat model (config.IS_CHAT == True): Text generation + parse_rating().
      Uses LLaMA-2-Chat template with system prompt. rating_key="rating"
      for compute_character_means_pairwise.
    - Base model (config.IS_CHAT == False): Logit extraction over tokens
      "1"-"5" via get_rating_probs(). No chat template.
      rating_key="expected_rating" for compute_character_means_pairwise.

Design (matching original):
    - 12 entities (dropping "You"; original had 13)
    - 18 mental capacity surveys
    - 66 pairwise comparisons per capacity (12 choose 2)
    - Each pair presented in BOTH orders (A-B and B-A) to control for
      position bias, since we have one deterministic model instead of
      multiple human respondents whose position effects average out
    - 5-point scale anchored by character NAMES (matching original's
      character-anchored visual scale)
    - Total: 66 pairs x 2 orders x 18 capacities = 2,376 comparisons

Design (different from original):
    - No photos (text descriptions only)
    - One deterministic model instead of ~2,400 human participants
    - Model rates all 18 capacities (humans chose one)
    - No "You" entity (12 instead of 13 characters)

Analysis (matching original):
    - Compute mean relative rating per entity per capacity
    - PCA with VARIMAX rotation on capacity correlations across characters
    - Regression-method factor scores, rescaled to 0-1
    - Compare entity positions to Gray et al. Figure 1

Output (all with pairwise_ prefix):
    data_dir("gray_replication", "behavior", tag)/
        pairwise_raw_responses.json
        pairwise_character_means.npz
        pairwise_pca_results.npz
        pairwise_human_correlations.json
        pairwise_consistency_stats.json
    results_dir("gray_replication", "behavior", tag)/
        pairwise_results_summary.md

Usage:
    python 1_pairwise_replication.py --model llama2_13b_chat
    python 1_pairwise_replication.py --model llama2_13b_base
    python 1_pairwise_replication.py --model llama2_13b_chat --include_self
    python 1_pairwise_replication.py --model llama2_13b_chat --both
    python 1_pairwise_replication.py --model llama2_13b_base --single_order --seed 42

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from datetime import datetime
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir, get_device, get_condition_tag,
)
from entities.gray_entities import (
    GRAY_ET_AL_SCORES,
    CAPACITY_PROMPTS,
    CAPACITY_NAMES,
    ENTITY_NAMES,
    CHARACTER_NAMES,
    CHARACTER_DESCRIPTIONS,
)
from utils.utils import (
    varimax_rotation,
    run_pca_varimax,
    compute_character_means_pairwise,
    correlate_with_humans,
    format_chat_prompt,
)


# ========================== CONFIG ========================== #

MAX_NEW_TOKENS = 50

# Minimal system prompt for chat model -- does not define the scale
# (scale appears in each comparison, matching the original where the
# scale was part of the visual display, not a separate instruction).
SYSTEM_PROMPT = (
    "You are completing a survey about the mental capacities of different "
    "characters. For each comparison, give your honest judgment. Respond "
    "with only a number from 1 to 5."
)


# ========================== PROMPT FORMATTING ========================== #

def format_comparison_prompt_chat(entity_a, entity_b, capacity_survey_desc):
    """
    Format a single pairwise comparison prompt for the chat model.

    Matches the original procedure:
    - Survey description from Appendix B ("This survey asks you to judge...")
    - Both character descriptions from Appendix A
    - 5-point scale anchored by character NAMES (not generic A/B labels)
    """
    name_a = CHARACTER_NAMES[entity_a]
    desc_a = CHARACTER_DESCRIPTIONS[entity_a]
    name_b = CHARACTER_NAMES[entity_b]
    desc_b = CHARACTER_DESCRIPTIONS[entity_b]

    user_msg = (
        f"{name_a}. {desc_a}\n\n"
        f"{name_b}. {desc_b}\n\n"
        f"{capacity_survey_desc}\n\n"
        f"1 = Much more {name_a}\n"
        f"2 = Slightly more {name_a}\n"
        f"3 = Both equally\n"
        f"4 = Slightly more {name_b}\n"
        f"5 = Much more {name_b}\n\n"
        f"Rating:"
    )
    return user_msg


def format_comparison_prompt_base(entity_a, entity_b, capacity_survey_desc):
    """
    Format a pairwise comparison as a text-completion prompt for the
    base model. Structured like a survey response form so the natural
    completion is a single digit 1-5.
    """
    name_a = CHARACTER_NAMES[entity_a]
    desc_a = CHARACTER_DESCRIPTIONS[entity_a]
    name_b = CHARACTER_NAMES[entity_b]
    desc_b = CHARACTER_DESCRIPTIONS[entity_b]

    prompt = (
        f"Survey: Mental Capacity Comparison\n\n"
        f"Character A: {name_a}. {desc_a}\n\n"
        f"Character B: {name_b}. {desc_b}\n\n"
        f"{capacity_survey_desc}\n\n"
        f"1 = Much more {name_a}\n"
        f"2 = Slightly more {name_a}\n"
        f"3 = Both equally\n"
        f"4 = Slightly more {name_b}\n"
        f"5 = Much more {name_b}\n\n"
        f"Rating: "
    )
    return prompt


# ========================== CHAT-SPECIFIC: GENERATION ========================== #

def generate_response(model, tokenizer, user_prompt, device):
    """Generate a response from LLaMA-2-Chat using greedy decoding."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    formatted = format_chat_prompt(messages, system_prompt=SYSTEM_PROMPT)

    encoding = tokenizer(
        formatted,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = encoding["input_ids"].shape[1]
    generated = tokenizer.decode(
        output_ids[0][prompt_len:], skip_special_tokens=True
    )
    return generated.strip()


def parse_rating(response_text):
    """Extract a 1-5 rating from the model's response.

    Returns the rating (int 1-5) or None if unparseable.
    """
    match = re.search(r'[1-5]', response_text)
    if match:
        return int(match.group())
    return None


# ========================== BASE-SPECIFIC: LOGIT EXTRACTION ========================== #

def get_rating_probs(model, tokenizer, prompt, device):
    """
    Single forward pass to extract probability distribution over
    tokens "1" through "5" at the next-token position.

    Returns:
        probs: np.array of shape (5,) -- P("1"), P("2"), ..., P("5")
        expected: float -- expected rating E[R] = sum(p_i * i)
        argmax: int -- most probable rating (1-5)
    """
    # Get token IDs for "1" through "5"
    rating_token_ids = []
    for digit in ["1", "2", "3", "4", "5"]:
        # Encode just the digit -- some tokenizers add BOS, so take last token
        ids = tokenizer.encode(digit, add_special_tokens=False)
        rating_token_ids.append(ids[-1])

    encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=2048,
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )

    # Logits at last position
    last_logits = outputs.logits[0, -1, :]  # (vocab_size,)

    # Extract logits for rating tokens only
    rating_logits = torch.tensor(
        [last_logits[tid].item() for tid in rating_token_ids]
    )

    # Softmax over just the 5 rating tokens (renormalized)
    probs = torch.softmax(rating_logits, dim=0).numpy()

    # Expected value
    values = np.array([1, 2, 3, 4, 5], dtype=float)
    expected = float(np.sum(probs * values))

    # Argmax
    argmax_rating = int(np.argmax(probs) + 1)

    return probs, expected, argmax_rating


# ========================== RESULTS SUMMARY ========================== #

def _write_pca_section(f, pca_results, entity_keys, capacity_keys,
                       human_correlations):
    """Write PCA and alignment sections (shared by chat and base summaries)."""
    eigenvalues = pca_results["eigenvalues"]

    f.write("## PCA Results\n\n")
    f.write("### Eigenvalues and explained variance\n\n")
    f.write(
        "Gray et al. found: Experience eigenvalue = 15.85 (88%), "
        "Agency eigenvalue = 1.46 (8%), total = 97%.\n\n"
    )
    f.write("| Component | Eigenvalue | Variance | Cumulative |\n")
    f.write("|----------:|-----------:|---------:|-----------:|\n")
    cum = 0
    for i in range(min(5, len(eigenvalues))):
        pct = 100 * eigenvalues[i] / np.sum(eigenvalues)
        cum += pct
        marker = " *" if eigenvalues[i] > 1.0 else ""
        f.write(f"| PC{i+1} | {eigenvalues[i]:.2f}{marker} | "
                f"{pct:.1f}% | {cum:.1f}% |\n")
    f.write("\n*eigenvalue > 1 (retained)\n\n")

    # Capacity loadings (varimax-rotated)
    rotated = pca_results["rotated_loadings"]
    f.write("### Varimax-rotated capacity loadings\n\n")
    f.write(
        "Compare to Gray et al. Table S1. In the original, Experience "
        "capacities loaded .67-.97 on Factor 1 and Agency capacities "
        "loaded .73-.97 on Factor 2.\n\n"
    )
    f.write("| Capacity | Human factor | F1 loading | F2 loading |\n")
    f.write("|----------|:------------:|-----------:|-----------:|\n")
    for c_idx, cap_key in enumerate(capacity_keys):
        _, factor = CAPACITY_PROMPTS[cap_key]
        n_cols = min(2, rotated.shape[1])
        l1 = rotated[c_idx, 0] if n_cols > 0 else 0
        l2 = rotated[c_idx, 1] if n_cols > 1 else 0
        f.write(f"| {cap_key} | {factor} | "
                f"{l1:+.3f} | {l2:+.3f} |\n")
    f.write("\n")

    # Entity factor scores (0-1 scale)
    scores_01 = pca_results["factor_scores_01"]
    f.write("### Entity positions (factor scores, 0-1 scale)\n\n")
    f.write(
        "Compare to Gray et al. Figure 1. Scores are rescaled to 0-1 "
        "to match the original figure.\n\n"
    )
    n_f = min(2, scores_01.shape[1])
    f.write("| Entity | Model F1 | Model F2 | Human Exp | Human Ag |\n")
    f.write("|--------|--------:|--------:|----------:|---------:|\n")
    for e_idx, ent_key in enumerate(entity_keys):
        h_exp, h_ag = GRAY_ET_AL_SCORES[ent_key]
        f1 = scores_01[e_idx, 0] if n_f > 0 else 0
        f2 = scores_01[e_idx, 1] if n_f > 1 else 0
        f.write(f"| {ent_key} | {f1:.3f} | {f2:.3f} | "
                f"{h_exp:.2f} | {h_ag:.2f} |\n")
    f.write("\n")

    # Alignment with human scores
    f.write("## Alignment with human Experience/Agency\n\n")
    f.write(
        "Spearman correlations between model factor scores (0-1) and "
        "human factor scores. We check all factor-to-dimension "
        "correlations since the factor ordering and sign may differ.\n\n"
    )
    f.write("| | Human Experience | Human Agency |\n")
    f.write("|---|---:|---:|\n")
    for fi in range(min(2, n_f)):
        label = f"Factor {fi+1}"
        rho_e = human_correlations[f"f{fi+1}_experience"]["rho"]
        p_e = human_correlations[f"f{fi+1}_experience"]["p"]
        rho_a = human_correlations[f"f{fi+1}_agency"]["rho"]
        p_a = human_correlations[f"f{fi+1}_agency"]["p"]
        f.write(f"| {label} | rho={rho_e:+.3f} (p={p_e:.4f}) | "
                f"rho={rho_a:+.3f} (p={p_a:.4f}) |\n")
    f.write("\n")


def write_results_summary_chat(out_dir, responses, means, pca_results,
                               entity_keys, capacity_keys,
                               human_correlations, consistency_stats):
    """Write documented markdown summary for the chat model."""
    n_ent = len(entity_keys)
    n_cap = len(capacity_keys)
    n_total = len(responses)
    n_parsed = sum(1 for r in responses if r["rating"] is not None)
    n_factors = pca_results["n_factors"]

    path = os.path.join(str(out_dir), "pairwise_results_summary.md")
    with open(path, "w") as f:
        f.write("# Experiment 4: Pairwise Behavioral Replication of "
                "Gray et al. (2007)\n")
        f.write(f"## {config.MODEL_LABEL}\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # What is being tested
        f.write("---\n\n## What is being tested\n\n")
        f.write(
            f"Does {config.MODEL_LABEL}'s **explicit folk psychology** of mind "
            "perception match the human folk psychology described by Gray, "
            "Gray, & Wegner (2007, Science)?\n\n"
            "This is a direct behavioral replication: the model answers the "
            "same pairwise comparison questions that ~2,400 human participants "
            "answered in the original study, using the exact character "
            "descriptions and mental capacity survey prompts from the "
            "supplementary materials.\n\n"
        )

        # Procedure
        f.write("## Procedure: What matches and what differs\n\n")
        f.write("### Matches original\n\n")
        f.write(
            "- Character descriptions: verbatim from Appendix A\n"
            "- Capacity survey prompts: verbatim from Appendix B "
            "(\"This survey asks you to judge...\")\n"
            "- 5-point scale anchored by character names "
            "(\"Much more [Name]\" / \"Slightly more [Name]\" / "
            "\"Both equally\")\n"
            "- All pairwise comparisons per capacity\n"
            "- Analysis: PCA with varimax rotation on capacity correlations "
            "across characters, regression-method factor scores rescaled "
            "to 0-1\n\n"
        )
        f.write("### Differs from original\n\n")
        f.write(
            "- **No photos**: original included character images; model "
            "gets text descriptions only\n"
            "- **One deterministic participant**: original averaged over "
            "~100+ respondents per capacity survey; we have one model with "
            "greedy decoding\n"
            "- **Position-bias control**: original counterbalanced left/right "
            "across participants; we present each pair in BOTH orders and "
            "average to eliminate position bias\n"
            "- **No survey selection**: original participants chose a "
            "capacity to rate; model rates all 18\n"
            f"- **{n_ent} entities**: dropped \"You\" (no self-referential "
            "analog for an LLM)\n\n"
        )

        # Parse stats
        f.write("## Response statistics\n\n")
        f.write(f"- Total comparisons: {n_total}\n")
        f.write(f"- Successfully parsed: {n_parsed} / {n_total} "
                f"({100 * n_parsed / max(n_total, 1):.1f}%)\n\n")

        n_failed = n_total - n_parsed
        if n_failed > 0:
            f.write(f"### Parse failures ({n_failed})\n\n")
            for resp in responses:
                if resp["rating"] is None:
                    f.write(
                        f"- {resp['capacity']}: {resp['name_a']} vs "
                        f"{resp['name_b']} -> \"{resp['response']}\"\n"
                    )
            f.write("\n")

        # Rating distribution
        f.write("### Rating distribution\n\n")
        f.write("| Rating | Count | Pct |\n")
        f.write("|-------:|------:|----:|\n")
        for r in range(1, 6):
            count = sum(1 for resp in responses if resp["rating"] == r)
            pct = 100 * count / max(n_parsed, 1)
            f.write(f"| {r} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        # Order consistency
        f.write("### Order consistency\n\n")
        f.write(
            "For each pair presented in both orders (A-B and B-A), a "
            "consistent model should give opposite ratings (R_AB + R_BA = 6). "
            "Deviation from this indicates position bias.\n\n"
        )
        f.write(f"- Pairs with both orders parsed: "
                f"{consistency_stats['n_pairs_both']}\n")
        f.write(f"- Perfectly consistent (sum = 6): "
                f"{consistency_stats['n_consistent']} "
                f"({consistency_stats['pct_consistent']:.1f}%)\n")
        f.write(f"- Mean |sum - 6|: "
                f"{consistency_stats['mean_deviation']:.2f}\n\n")

        # PCA results
        _write_pca_section(f, pca_results, entity_keys, capacity_keys,
                           human_correlations)

    print(f"  Results summary: {path}")


def write_results_summary_base(out_dir, responses, means, pca_results,
                               entity_keys, capacity_keys,
                               human_correlations, consistency_stats):
    """Write documented markdown summary for the base model."""
    n_ent = len(entity_keys)
    n_total = len(responses)
    n_factors = pca_results["n_factors"]

    path = os.path.join(str(out_dir), "pairwise_results_summary.md")
    with open(path, "w") as f:
        f.write("# Experiment 4: Pairwise Behavioral Replication of "
                "Gray et al. (2007)\n")
        f.write(f"## {config.MODEL_LABEL}\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n## What is being tested\n\n")
        f.write(
            f"Does {config.MODEL_LABEL}'s **implicit folk psychology** "
            "of mind perception match the human folk psychology described by "
            "Gray, Gray, & Wegner (2007, Science)?\n\n"
            "Unlike the chat model, the base model has no RLHF safety training, "
            "so it does not refuse to make comparisons. Instead of generating "
            "text responses, we extract the probability distribution over "
            "rating tokens (1-5) from the next-token logits.\n\n"
        )

        f.write("## Procedure\n\n")
        f.write("### Matches original\n\n")
        f.write(
            "- Character descriptions: verbatim from Appendix A\n"
            "- Capacity survey prompts: verbatim from Appendix B\n"
            "- 5-point scale anchored by character names\n"
            "- All pairwise comparisons per capacity\n"
            "- Analysis: PCA with varimax rotation, regression factor "
            "scores rescaled to 0-1\n\n"
        )
        f.write("### Differs from original\n\n")
        f.write(
            f"- **Base model**: pretrained {config.MODEL_LABEL} (no chat/RLHF)\n"
            "- **Logit-based ratings**: probability distribution over tokens "
            "\"1\"-\"5\" instead of generated text\n"
            "- **Expected rating**: continuous E[R] = sum(p_i * i) instead "
            "of discrete response\n"
            "- **No photos**: text descriptions only\n"
            "- **Completion format**: prompt ends with \"Rating:\" for "
            "natural completion\n"
            "- **Position-bias control**: each pair in both orders\n"
            f"- **{n_ent} entities**\n\n"
        )

        # Response statistics
        f.write("## Response statistics\n\n")
        f.write(f"- Total comparisons: {n_total}\n")
        f.write(f"- All comparisons yield ratings (logit-based, no refusals)\n\n")

        # Probability concentration
        all_probs = np.array([r["probs"] for r in responses])
        max_probs = all_probs.max(axis=1)
        f.write("### Rating probability concentration\n\n")
        f.write(
            "How confident is the model? Distribution of max P(rating):\n\n"
        )
        for threshold in [0.5, 0.7, 0.9]:
            pct = 100 * np.mean(max_probs >= threshold)
            f.write(f"- P(top rating) >= {threshold}: {pct:.1f}%\n")
        f.write(f"- Mean max P: {np.mean(max_probs):.3f}\n")
        f.write(f"- Mean expected rating: "
                f"{np.mean([r['expected_rating'] for r in responses]):.2f}\n\n")

        # Argmax rating distribution
        f.write("### Argmax rating distribution\n\n")
        f.write("| Rating | Count | Pct |\n")
        f.write("|-------:|------:|----:|\n")
        for r in range(1, 6):
            count = sum(1 for resp in responses if resp["argmax_rating"] == r)
            pct = 100 * count / n_total
            f.write(f"| {r} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        # Order consistency
        f.write("### Order consistency\n\n")
        f.write(
            "For each pair in both orders, expected ratings should be "
            "complementary: E[R_AB] + E[R_BA] should equal 6.\n\n"
        )
        f.write(f"- Pairs with both orders: "
                f"{consistency_stats['n_pairs_both']}\n")
        f.write(f"- Mean |E[R_AB] + E[R_BA] - 6|: "
                f"{consistency_stats['mean_deviation']:.3f}\n")
        f.write(f"- Argmax perfectly consistent: "
                f"{consistency_stats['n_consistent']} "
                f"({consistency_stats['pct_consistent']:.1f}%)\n\n")

        # PCA results
        _write_pca_section(f, pca_results, entity_keys, capacity_keys,
                           human_correlations)

    print(f"  Results summary: {path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Pairwise behavioral replication of Gray et al."
    )
    add_model_argument(parser)
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both chat and base models sequentially"
    )
    parser.add_argument(
        "--single_order", action="store_true",
        help="Present each pair in one random order only (skip "
             "counterbalancing). Halves runtime but doesn't control "
             "for position bias."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pair order randomization (single_order mode)"
    )
    args = parser.parse_args()

    if args.both:
        models_to_run = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models_to_run = [args.model]

    for model_key in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Running model: {model_key}")
        print(f"{'='*60}\n")
        set_model(model_key)
        run_replication(args)


def run_replication(args):
    """Run the pairwise replication for the currently active model."""
    device = get_device()
    is_chat = config.IS_CHAT

    # Select entities
    if args.include_self:
        entity_keys = list(ENTITY_NAMES)
    else:
        entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]

    capacity_keys = list(CAPACITY_NAMES)
    n_entities = len(entity_keys)
    n_capacities = len(capacity_keys)
    n_pairs = n_entities * (n_entities - 1) // 2

    # Build pair list
    base_pairs = list(combinations(range(n_entities), 2))
    if args.single_order:
        # Randomize which entity is presented first
        rng = np.random.RandomState(args.seed)
        trial_pairs = []
        for i, j in base_pairs:
            if rng.random() < 0.5:
                trial_pairs.append((j, i))
            else:
                trial_pairs.append((i, j))
    else:
        # Both orders for each pair
        trial_pairs = []
        for i, j in base_pairs:
            trial_pairs.append((i, j))
            trial_pairs.append((j, i))

    n_trials_per_cap = len(trial_pairs)
    n_total = n_trials_per_cap * n_capacities

    tag = get_condition_tag(args.include_self)

    print(f"Model: {config.MODEL_LABEL} (is_chat={is_chat})")
    print(f"Device: {device}")
    print(f"Entities ({n_entities}): {entity_keys}")
    print(f"Capacities: {n_capacities}")
    print(f"Pairs: {n_pairs}, trials per capacity: {n_trials_per_cap}")
    print(f"Total comparisons: {n_total}")
    print(f"Counterbalanced: {not args.single_order}")
    print(f"Tag: {tag}")

    # Output dirs
    ddir = data_dir("gray_replication", "behavior", tag)
    rdir = results_dir("gray_replication", "behavior", tag)
    print(f"Data dir: {ddir}")
    print(f"Results dir: {rdir}")

    # Build survey descriptions (matching Appendix B format)
    survey_descriptions = {}
    for cap_key in capacity_keys:
        cap_prompt, _ = CAPACITY_PROMPTS[cap_key]
        survey_descriptions[cap_key] = (
            f"This survey asks you to judge {cap_prompt}."
        )

    # Load model
    print(f"\nLoading {config.MODEL_LABEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model.half().to(device).eval()
    print(f"Model loaded on {device}.")

    # Base model: verify rating token IDs and run diagnostic
    if not is_chat:
        for digit in ["1", "2", "3", "4", "5"]:
            ids = tokenizer.encode(digit, add_special_tokens=False)
            print(f"  Token '{digit}' -> ID {ids}")

        test_prompt = format_comparison_prompt_base(
            entity_keys[0], entity_keys[1],
            survey_descriptions[capacity_keys[0]]
        )
        test_enc = tokenizer(test_prompt, return_tensors="pt")
        with torch.no_grad():
            test_out = model(
                input_ids=test_enc["input_ids"].to(device),
                attention_mask=test_enc["attention_mask"].to(device),
            )
        test_logits = test_out.logits[0, -1, :]
        top5 = torch.topk(test_logits, 5)
        print("\n  Diagnostic -- top 5 next-token predictions after 'Rating: ':")
        for rank, (logit, tid) in enumerate(
            zip(top5.values, top5.indices)
        ):
            tok = tokenizer.decode([tid.item()])
            print(f"    {rank+1}. '{tok}' (id={tid.item()}, "
                  f"logit={logit.item():.2f})")
        print()

    # ── Run all comparisons ──
    all_responses = []
    n_parsed = 0
    n_done = 0

    for cap_idx, cap_key in enumerate(capacity_keys):
        _, cap_factor = CAPACITY_PROMPTS[cap_key]
        survey_desc = survey_descriptions[cap_key]
        print(f"\n[{cap_idx+1}/{n_capacities}] {cap_key} ({cap_factor})")

        for idx_a, idx_b in trial_pairs:
            entity_a = entity_keys[idx_a]
            entity_b = entity_keys[idx_b]

            if is_chat:
                # Chat model: generate text + parse rating
                user_prompt = format_comparison_prompt_chat(
                    entity_a, entity_b, survey_desc
                )
                response = generate_response(
                    model, tokenizer, user_prompt, device
                )
                rating = parse_rating(response)

                all_responses.append({
                    "capacity": cap_key,
                    "capacity_factor": cap_factor,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "name_a": CHARACTER_NAMES[entity_a],
                    "name_b": CHARACTER_NAMES[entity_b],
                    "response": response,
                    "rating": rating,
                })

                if rating is not None:
                    n_parsed += 1

            else:
                # Base model: logit extraction
                prompt = format_comparison_prompt_base(
                    entity_a, entity_b, survey_desc
                )
                probs, expected, argmax = get_rating_probs(
                    model, tokenizer, prompt, device
                )

                all_responses.append({
                    "capacity": cap_key,
                    "capacity_factor": cap_factor,
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "name_a": CHARACTER_NAMES[entity_a],
                    "name_b": CHARACTER_NAMES[entity_b],
                    "probs": probs.tolist(),
                    "expected_rating": expected,
                    "argmax_rating": argmax,
                })
                n_parsed += 1

            n_done += 1
            if n_done % 200 == 0:
                if is_chat:
                    print(f"  {n_done}/{n_total} done "
                          f"({100*n_parsed/n_done:.1f}% parsed)")
                else:
                    mean_exp = np.mean([r["expected_rating"]
                                        for r in all_responses])
                    print(f"  {n_done}/{n_total} done "
                          f"(mean expected rating: {mean_exp:.2f})")

    print(f"\nCompleted: {n_done}, parsed: {n_parsed} "
          f"({100*n_parsed/max(n_done,1):.1f}%)")

    # Save raw responses
    raw_path = os.path.join(str(ddir), "pairwise_raw_responses.json")
    with open(raw_path, "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"Saved raw responses to {raw_path}")

    # ── Order consistency check ──
    consistency_stats = {"n_pairs_both": 0, "n_consistent": 0,
                         "mean_deviation": 0.0, "pct_consistent": 0.0}

    if not args.single_order:
        pair_groups = defaultdict(list)
        for resp in all_responses:
            if is_chat and resp.get("rating") is None:
                continue
            pair_key = (resp["capacity"],
                        frozenset([resp["entity_a"], resp["entity_b"]]))
            pair_groups[pair_key].append(resp)

        if is_chat:
            # Chat: discrete rating consistency
            deviations = []
            n_both = 0
            n_consistent = 0

            for key, resps in pair_groups.items():
                if len(resps) == 2:
                    n_both += 1
                    r1 = resps[0]["rating"]
                    r2 = resps[1]["rating"]
                    deviation = abs((r1 + r2) - 6)
                    deviations.append(deviation)
                    if deviation == 0:
                        n_consistent += 1

            consistency_stats = {
                "n_pairs_both": n_both,
                "n_consistent": n_consistent,
                "pct_consistent": 100 * n_consistent / max(n_both, 1),
                "mean_deviation": np.mean(deviations) if deviations else 0.0,
            }
            print(f"\nOrder consistency: {n_consistent}/{n_both} perfectly "
                  f"consistent ({consistency_stats['pct_consistent']:.1f}%)")
            print(f"  Mean |R_AB + R_BA - 6| = "
                  f"{consistency_stats['mean_deviation']:.2f}")

        else:
            # Base: expected rating + argmax consistency
            deviations_expected = []
            deviations_argmax = []
            n_both = 0
            n_consistent_argmax = 0

            for key, resps in pair_groups.items():
                if len(resps) == 2:
                    n_both += 1
                    # Expected rating consistency
                    e1 = resps[0]["expected_rating"]
                    e2 = resps[1]["expected_rating"]
                    deviations_expected.append(abs((e1 + e2) - 6))
                    # Argmax consistency
                    a1 = resps[0]["argmax_rating"]
                    a2 = resps[1]["argmax_rating"]
                    deviations_argmax.append(abs((a1 + a2) - 6))
                    if (a1 + a2) == 6:
                        n_consistent_argmax += 1

            consistency_stats = {
                "n_pairs_both": n_both,
                "n_consistent": n_consistent_argmax,
                "pct_consistent": (100 * n_consistent_argmax
                                   / max(n_both, 1)),
                "mean_deviation": (np.mean(deviations_expected)
                                   if deviations_expected else 0.0),
                "mean_deviation_argmax": (np.mean(deviations_argmax)
                                          if deviations_argmax else 0.0),
            }
            print(f"\nOrder consistency (expected rating):")
            print(f"  Pairs: {n_both}")
            print(f"  Mean |E[R_AB] + E[R_BA] - 6|: "
                  f"{consistency_stats['mean_deviation']:.3f}")
            print(f"  Argmax consistent: {n_consistent_argmax}/{n_both} "
                  f"({consistency_stats['pct_consistent']:.1f}%)")

    # ── Character means ──
    rating_key = "rating" if is_chat else "expected_rating"
    means = compute_character_means_pairwise(
        all_responses, entity_keys, capacity_keys, rating_key=rating_key
    )
    print(f"\nCharacter means: {means.shape} (capacities x entities)")

    # ── PCA with varimax ──
    print("\nPCA with varimax rotation:")
    pca_results = run_pca_varimax(means)

    evr = pca_results["explained_var_ratio"]
    for i in range(min(3, len(evr))):
        print(f"  Factor {i+1}: {evr[i]*100:.1f}%")

    # ── Correlate with human scores ──
    human_correlations = correlate_with_humans(
        pca_results, entity_keys, GRAY_ET_AL_SCORES
    )
    scores_01 = pca_results["factor_scores_01"]
    n_factors = min(2, scores_01.shape[1])

    print("\nCorrelation with human factor scores (0-1 scaled):")
    for fi in range(n_factors):
        rho_e = human_correlations[f"f{fi+1}_experience"]["rho"]
        p_e = human_correlations[f"f{fi+1}_experience"]["p"]
        rho_a = human_correlations[f"f{fi+1}_agency"]["rho"]
        p_a = human_correlations[f"f{fi+1}_agency"]["p"]
        print(f"  Factor {fi+1} vs Experience: rho={rho_e:+.3f} (p={p_e:.4f})")
        print(f"  Factor {fi+1} vs Agency:     rho={rho_a:+.3f} (p={p_a:.4f})")

    # ── Save ──
    np.savez_compressed(
        os.path.join(str(ddir), "pairwise_character_means.npz"),
        means=means,
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    np.savez_compressed(
        os.path.join(str(ddir), "pairwise_pca_results.npz"),
        rotated_loadings=pca_results["rotated_loadings"],
        unrotated_loadings=pca_results["unrotated_loadings"],
        factor_scores_raw=pca_results["factor_scores_raw"],
        factor_scores_01=pca_results["factor_scores_01"],
        eigenvalues=pca_results["eigenvalues"],
        explained_var_ratio=pca_results["explained_var_ratio"],
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    with open(os.path.join(str(ddir), "pairwise_human_correlations.json"), "w") as fh:
        json.dump(human_correlations, fh, indent=2)
    with open(os.path.join(str(ddir), "pairwise_consistency_stats.json"), "w") as fh:
        json.dump(consistency_stats, fh, indent=2)

    # Results summary
    if is_chat:
        write_results_summary_chat(
            rdir, all_responses, means, pca_results,
            entity_keys, capacity_keys, human_correlations,
            consistency_stats
        )
    else:
        write_results_summary_base(
            rdir, all_responses, means, pca_results,
            entity_keys, capacity_keys, human_correlations,
            consistency_stats
        )

    print(f"\nSaved data to {ddir}/")
    print(f"Saved results to {rdir}/")


if __name__ == "__main__":
    main()
