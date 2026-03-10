#!/usr/bin/env python3
"""
Gray Replication with AI/Human Characters — Names Only

Same as 4_gray_with_characters.py but character descriptions are omitted.
Only character names appear in the prompts, testing whether the model's
mental capacity ratings rely on prior knowledge of names alone (vs explicit
descriptions that reveal character type).

Design:
    - 30 characters (15 AI + 15 human) from concept_geometry/characters.py
    - 18 mental capacities from entities/gray_entities.py
    - 435 pairs per capacity (30 choose 2) x 2 orders = 870 trials/capacity
    - Total: 870 x 18 = 15,660 comparisons
    - Prompts use names only — no descriptions

Analysis:
    - PCA with varimax rotation on capacity correlations across characters
    - Categorical alignment: Mann-Whitney U on factor scores (AI vs human)
    - Per-capacity group means
    - Order consistency stats

Output:
    results/{model}/behavior/names_only/data/
        pairwise_raw_responses.json
        pairwise_character_means.npz
        pairwise_pca_results.npz
        pairwise_categorical_analysis.json
        pairwise_consistency_stats.json

    results/{model}/behavior/names_only/
        pairwise_results_summary.md

Usage:
    python behavior/5_gray_names_only.py --model llama2_13b_chat
    python behavior/5_gray_names_only.py --model llama2_13b_base
    python behavior/5_gray_names_only.py --model llama2_13b_chat --both

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
from scipy.stats import mannwhitneyu

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir, get_device,
)
from concept_geometry.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from entities.gray_entities import (
    CAPACITY_PROMPTS, CAPACITY_NAMES,
)
from utils.utils import (
    run_pca_varimax,
    compute_character_means_pairwise,
    llama_v2_prompt,
)


# ========================== CONFIG ========================== #

CONDITION = "names_only"

MAX_NEW_TOKENS = 50

SYSTEM_PROMPT = (
    "You are completing a survey about the mental capacities of different "
    "characters. For each comparison, give your honest judgment. Respond "
    "with only a number from 1 to 5."
)


# ========================== PROMPT FORMATTING ========================== #

def format_comparison_prompt_chat(char_a, char_b, capacity_survey_desc):
    """Format a single pairwise comparison prompt for the chat model.
    Names only — no character descriptions."""
    name_a = CHARACTER_NAMES[char_a]
    name_b = CHARACTER_NAMES[char_b]

    user_msg = (
        f"Character A: {name_a}\n\n"
        f"Character B: {name_b}\n\n"
        f"{capacity_survey_desc}\n\n"
        f"1 = Much more {name_a}\n"
        f"2 = Slightly more {name_a}\n"
        f"3 = Both equally\n"
        f"4 = Slightly more {name_b}\n"
        f"5 = Much more {name_b}\n\n"
        f"Rating:"
    )
    return user_msg


def format_comparison_prompt_base(char_a, char_b, capacity_survey_desc):
    """Format a pairwise comparison as text-completion for base model.
    Names only — no character descriptions."""
    name_a = CHARACTER_NAMES[char_a]
    name_b = CHARACTER_NAMES[char_b]

    prompt = (
        f"Survey: Mental Capacity Comparison\n\n"
        f"Character A: {name_a}\n\n"
        f"Character B: {name_b}\n\n"
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
    formatted = llama_v2_prompt(messages, system_prompt=SYSTEM_PROMPT)

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
    """Extract a 1-5 rating from the model's response."""
    match = re.search(r'[1-5]', response_text)
    if match:
        return int(match.group())
    return None


# ========================== BASE-SPECIFIC: LOGIT EXTRACTION ========================== #

def get_rating_probs(model, tokenizer, prompt, device):
    """
    Single forward pass to extract probability distribution over
    tokens "1" through "5" at the next-token position.
    """
    rating_token_ids = []
    for digit in ["1", "2", "3", "4", "5"]:
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

    last_logits = outputs.logits[0, -1, :]
    rating_logits = torch.tensor(
        [last_logits[tid].item() for tid in rating_token_ids]
    )
    probs = torch.softmax(rating_logits, dim=0).numpy()
    values = np.array([1, 2, 3, 4, 5], dtype=float)
    expected = float(np.sum(probs * values))
    argmax_rating = int(np.argmax(probs) + 1)

    return probs, expected, argmax_rating


# ========================== CATEGORICAL ANALYSIS ========================== #

def compute_categorical_analysis(pca_results, char_keys):
    """
    Analyze how well PCA factor scores separate AI from human characters.
    """
    scores_01 = pca_results["factor_scores_01"]
    n_factors = scores_01.shape[1]

    char_to_idx = {k: i for i, k in enumerate(char_keys)}
    ai_indices = [char_to_idx[k] for k in char_keys if k in AI_CHARACTERS]
    human_indices = [char_to_idx[k] for k in char_keys if k in HUMAN_CHARACTERS]

    results = {"factors": [], "anomalies": []}

    for fi in range(min(n_factors, 4)):
        ai_scores = scores_01[ai_indices, fi]
        human_scores = scores_01[human_indices, fi]

        try:
            u_stat, p_val = mannwhitneyu(ai_scores, human_scores,
                                         alternative="two-sided")
        except ValueError:
            u_stat, p_val = float("nan"), float("nan")

        results["factors"].append({
            "factor": fi + 1,
            "ai_mean": float(np.mean(ai_scores)),
            "ai_std": float(np.std(ai_scores)),
            "human_mean": float(np.mean(human_scores)),
            "human_std": float(np.std(human_scores)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_val),
            "separation": abs(float(np.mean(ai_scores)) -
                              float(np.mean(human_scores))),
        })

    # Anomaly detection
    for fi in range(min(n_factors, 2)):
        ai_mean = np.mean(scores_01[ai_indices, fi])
        human_mean = np.mean(scores_01[human_indices, fi])

        for idx, char_key in enumerate(char_keys):
            score = scores_01[idx, fi]
            char_type = CHARACTER_TYPES[char_key]

            if char_type == "ai":
                own_dist = abs(score - ai_mean)
                other_dist = abs(score - human_mean)
            else:
                own_dist = abs(score - human_mean)
                other_dist = abs(score - ai_mean)

            if other_dist < own_dist:
                results["anomalies"].append({
                    "character": char_key,
                    "type": char_type,
                    "factor": fi + 1,
                    "score": float(score),
                    "own_group_mean": float(
                        ai_mean if char_type == "ai" else human_mean),
                    "other_group_mean": float(
                        human_mean if char_type == "ai" else ai_mean),
                })

    return results


# ========================== RESULTS SUMMARY ========================== #

def write_results_summary(out_dir, responses, means, pca_results,
                          char_keys, capacity_keys, categorical_analysis,
                          consistency_stats, is_chat):
    """Write documented markdown summary."""
    n_chars = len(char_keys)
    n_caps = len(capacity_keys)
    n_total = len(responses)
    n_factors = pca_results["n_factors"]

    path = os.path.join(str(out_dir), "pairwise_results_summary.md")
    with open(path, "w") as f:
        f.write("# Gray Replication with AI/Human Characters — Names Only\n")
        f.write(f"## {config.MODEL_LABEL}\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n## What is being tested\n\n")
        f.write(
            f"Does {config.MODEL_LABEL}'s folk psychology produce an "
            "Experience/Agency factor structure (Gray et al. 2007) when "
            "rating 30 AI/human characters on 18 mental capacities, "
            "using **names only** (no descriptions)? "
            "This tests whether prior knowledge of names alone is sufficient "
            "to drive differential mental capacity attributions.\n\n"
        )

        f.write("## Procedure\n\n")
        f.write(f"- {n_chars} characters: {len([k for k in char_keys if k in AI_CHARACTERS])} AI, "
                f"{len([k for k in char_keys if k in HUMAN_CHARACTERS])} human\n")
        f.write(f"- {n_caps} mental capacities (Gray et al. 2007)\n")
        n_pairs = n_chars * (n_chars - 1) // 2
        f.write(f"- {n_pairs} pairwise comparisons per capacity\n")
        f.write(f"- Total comparisons: {n_total}\n")
        f.write("- **Prompts contain names only — no character descriptions**\n")
        if is_chat:
            f.write("- Method: text generation + parse rating (chat)\n\n")
        else:
            f.write("- Method: logit extraction over tokens 1-5 (base)\n\n")

        # Response statistics
        f.write("## Response statistics\n\n")
        if is_chat:
            n_parsed = sum(1 for r in responses if r.get("rating") is not None)
            f.write(f"- Successfully parsed: {n_parsed} / {n_total} "
                    f"({100 * n_parsed / max(n_total, 1):.1f}%)\n\n")
        else:
            f.write(f"- All {n_total} comparisons yield ratings (logit-based)\n\n")

        # Order consistency
        if consistency_stats["n_pairs_both"] > 0:
            f.write("### Order consistency\n\n")
            f.write(f"- Pairs with both orders: "
                    f"{consistency_stats['n_pairs_both']}\n")
            f.write(f"- Perfectly consistent: "
                    f"{consistency_stats['n_consistent']} "
                    f"({consistency_stats['pct_consistent']:.1f}%)\n")
            f.write(f"- Mean deviation: "
                    f"{consistency_stats['mean_deviation']:.3f}\n\n")

        # PCA
        eigenvalues = pca_results["eigenvalues"]
        rotated = pca_results["rotated_loadings"]
        scores_01 = pca_results["factor_scores_01"]

        f.write("## PCA Results\n\n")
        f.write("### Eigenvalues\n\n")
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

        # Capacity loadings
        f.write("### Varimax-rotated capacity loadings\n\n")
        header = "| Capacity | Factor |"
        sep = "|----------|--------|"
        for fi in range(min(n_factors, 4)):
            header += f" F{fi+1} |"
            sep += "----:|"
        f.write(header + "\n")
        f.write(sep + "\n")
        for c_idx, cap_key in enumerate(capacity_keys):
            _, cap_factor = CAPACITY_PROMPTS[cap_key]
            row = f"| {cap_key} | {cap_factor} |"
            for fi in range(min(n_factors, 4)):
                row += f" {rotated[c_idx, fi]:+.3f} |"
            f.write(row + "\n")
        f.write("\n")

        # Character positions
        f.write("### Character positions (factor scores, 0-1)\n\n")
        header = "| Character | Type |"
        sep = "|-----------|------|"
        for fi in range(min(n_factors, 2)):
            header += f" F{fi+1} |"
            sep += "----:|"
        f.write(header + "\n")
        f.write(sep + "\n")
        for e_idx, char_key in enumerate(char_keys):
            row = f"| {CHARACTER_NAMES[char_key]} | {CHARACTER_TYPES[char_key]} |"
            for fi in range(min(n_factors, 2)):
                row += f" {scores_01[e_idx, fi]:.3f} |"
            f.write(row + "\n")
        f.write("\n")

        # Categorical alignment
        f.write("## Categorical Alignment (AI vs Human)\n\n")
        for finfo in categorical_analysis["factors"]:
            fi = finfo["factor"]
            f.write(f"### Factor {fi}\n\n")
            f.write(f"- AI mean: {finfo['ai_mean']:.3f} "
                    f"(SD={finfo['ai_std']:.3f})\n")
            f.write(f"- Human mean: {finfo['human_mean']:.3f} "
                    f"(SD={finfo['human_std']:.3f})\n")
            f.write(f"- Separation: {finfo['separation']:.3f}\n")
            f.write(f"- Mann-Whitney U={finfo['mann_whitney_u']:.1f}, "
                    f"p={finfo['p_value']:.4f}\n\n")

        # Anomalies
        if categorical_analysis["anomalies"]:
            f.write("### Anomalies (closer to other group)\n\n")
            f.write("| Character | Type | Factor | Score | Own mean | "
                    "Other mean |\n")
            f.write("|-----------|------|-------:|------:|---------:|"
                    "-----------:|\n")
            for a in categorical_analysis["anomalies"]:
                f.write(f"| {a['character']} | {a['type']} | "
                        f"{a['factor']} | {a['score']:.3f} | "
                        f"{a['own_group_mean']:.3f} | "
                        f"{a['other_group_mean']:.3f} |\n")
            f.write("\n")

    print(f"  Results summary: {path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Gray replication with AI/human characters — names only"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--both", action="store_true",
        help="Run both chat and base models sequentially"
    )
    parser.add_argument(
        "--single_order", action="store_true",
        help="Present each pair in one random order only"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for pair order randomization"
    )
    parser.add_argument(
        "--exclude", nargs="+", default=[],
        help="Character keys to exclude (e.g., --exclude claude cortana casey)"
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
        run_names_only(args)


def run_names_only(args):
    """Run the names-only pairwise comparisons."""
    device = get_device()
    is_chat = config.IS_CHAT

    char_keys = [k for k in ALL_CHARACTERS if k not in args.exclude]
    if args.exclude:
        print(f"Excluding {len(args.exclude)} characters: {args.exclude}")
    capacity_keys = list(CAPACITY_NAMES)
    n_chars = len(char_keys)
    n_caps = len(capacity_keys)
    n_pairs = n_chars * (n_chars - 1) // 2

    # Build pair list
    base_pairs = list(combinations(range(n_chars), 2))
    if args.single_order:
        rng = np.random.RandomState(args.seed)
        trial_pairs = []
        for i, j in base_pairs:
            if rng.random() < 0.5:
                trial_pairs.append((j, i))
            else:
                trial_pairs.append((i, j))
    else:
        trial_pairs = []
        for i, j in base_pairs:
            trial_pairs.append((i, j))
            trial_pairs.append((j, i))

    n_trials_per_cap = len(trial_pairs)
    n_total = n_trials_per_cap * n_caps

    print(f"Model: {config.MODEL_LABEL} (is_chat={is_chat})")
    print(f"Device: {device}")
    print(f"Characters: {n_chars} ({len([k for k in char_keys if k in AI_CHARACTERS])} AI, "
          f"{len([k for k in char_keys if k in HUMAN_CHARACTERS])} human)")
    print(f"Capacities: {n_caps}")
    print(f"Pairs: {n_pairs}, trials per capacity: {n_trials_per_cap}")
    print(f"Total comparisons: {n_total}")
    print(f"Prompt style: NAMES ONLY (no descriptions)")

    # Output dirs
    ddir = data_dir("behavior", CONDITION)
    rdir = results_phase_dir("behavior", CONDITION)
    print(f"Data dir: {ddir}")
    print(f"Results dir: {rdir}")

    # Build survey descriptions
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
    model_obj = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model_obj.half().to(device).eval()
    print(f"Model loaded on {device}.")

    # Base model: verify rating tokens
    if not is_chat:
        for digit in ["1", "2", "3", "4", "5"]:
            ids = tokenizer.encode(digit, add_special_tokens=False)
            print(f"  Token '{digit}' -> ID {ids}")

    # ── Run all comparisons ──
    all_responses = []
    n_parsed = 0
    n_done = 0

    for cap_idx, cap_key in enumerate(capacity_keys):
        _, cap_factor = CAPACITY_PROMPTS[cap_key]
        survey_desc = survey_descriptions[cap_key]
        print(f"\n[{cap_idx+1}/{n_caps}] {cap_key} ({cap_factor})")

        for idx_a, idx_b in trial_pairs:
            char_a = char_keys[idx_a]
            char_b = char_keys[idx_b]

            if is_chat:
                user_prompt = format_comparison_prompt_chat(
                    char_a, char_b, survey_desc
                )
                response = generate_response(
                    model_obj, tokenizer, user_prompt, device
                )
                rating = parse_rating(response)

                all_responses.append({
                    "capacity": cap_key,
                    "capacity_factor": cap_factor,
                    "entity_a": char_a,
                    "entity_b": char_b,
                    "name_a": CHARACTER_NAMES[char_a],
                    "name_b": CHARACTER_NAMES[char_b],
                    "response": response,
                    "rating": rating,
                })

                if rating is not None:
                    n_parsed += 1

            else:
                prompt = format_comparison_prompt_base(
                    char_a, char_b, survey_desc
                )
                probs, expected, argmax = get_rating_probs(
                    model_obj, tokenizer, prompt, device
                )

                all_responses.append({
                    "capacity": cap_key,
                    "capacity_factor": cap_factor,
                    "entity_a": char_a,
                    "entity_b": char_b,
                    "name_a": CHARACTER_NAMES[char_a],
                    "name_b": CHARACTER_NAMES[char_b],
                    "probs": probs.tolist(),
                    "expected_rating": expected,
                    "argmax_rating": argmax,
                })
                n_parsed += 1

            n_done += 1
            if n_done % 500 == 0:
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

    # ── Order consistency ──
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
        else:
            deviations_expected = []
            n_both = 0
            n_consistent_argmax = 0
            for key, resps in pair_groups.items():
                if len(resps) == 2:
                    n_both += 1
                    e1 = resps[0]["expected_rating"]
                    e2 = resps[1]["expected_rating"]
                    deviations_expected.append(abs((e1 + e2) - 6))
                    a1 = resps[0]["argmax_rating"]
                    a2 = resps[1]["argmax_rating"]
                    if (a1 + a2) == 6:
                        n_consistent_argmax += 1
            consistency_stats = {
                "n_pairs_both": n_both,
                "n_consistent": n_consistent_argmax,
                "pct_consistent": 100 * n_consistent_argmax / max(n_both, 1),
                "mean_deviation": (np.mean(deviations_expected)
                                   if deviations_expected else 0.0),
            }

        print(f"\nOrder consistency: {consistency_stats['n_consistent']}/"
              f"{consistency_stats['n_pairs_both']} "
              f"({consistency_stats['pct_consistent']:.1f}%)")

    # ── Character means ──
    rating_key = "rating" if is_chat else "expected_rating"
    means = compute_character_means_pairwise(
        all_responses, char_keys, capacity_keys, rating_key=rating_key
    )
    print(f"\nCharacter means: {means.shape} (capacities x characters)")

    # ── PCA with varimax ──
    print("\nPCA with varimax rotation:")
    pca_results = run_pca_varimax(means)

    evr = pca_results["explained_var_ratio"]
    for i in range(min(3, len(evr))):
        print(f"  Factor {i+1}: {evr[i]*100:.1f}%")

    # ── Categorical analysis ──
    categorical_analysis = compute_categorical_analysis(pca_results, char_keys)

    print("\nCategorical alignment (AI vs Human):")
    for finfo in categorical_analysis["factors"]:
        print(f"  Factor {finfo['factor']}: AI={finfo['ai_mean']:.3f}, "
              f"Human={finfo['human_mean']:.3f}, "
              f"U={finfo['mann_whitney_u']:.1f}, "
              f"p={finfo['p_value']:.4f}")

    # ── Save ──
    np.savez_compressed(
        os.path.join(str(ddir), "pairwise_character_means.npz"),
        means=means,
        character_keys=np.array(char_keys),
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
        character_keys=np.array(char_keys),
        capacity_keys=np.array(capacity_keys),
    )
    with open(os.path.join(str(ddir), "pairwise_categorical_analysis.json"), "w") as fh:
        json.dump({
            "categorical": categorical_analysis,
        }, fh, indent=2)
    with open(os.path.join(str(ddir), "pairwise_consistency_stats.json"), "w") as fh:
        json.dump(consistency_stats, fh, indent=2)

    # Results summary
    write_results_summary(
        rdir, all_responses, means, pca_results,
        char_keys, capacity_keys, categorical_analysis,
        consistency_stats, is_chat
    )

    print(f"\nSaved data to {ddir}/")
    print(f"Saved results to {rdir}/")


if __name__ == "__main__":
    main()
