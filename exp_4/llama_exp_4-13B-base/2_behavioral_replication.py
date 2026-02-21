#!/usr/bin/env python3
"""
Experiment 4, Phase 2: Behavioral Replication of Gray et al. (2007)
BASE MODEL version — LLaMA-2-13B (no chat fine-tuning)

Uses the base (pretrained) model instead of the chat model to avoid
RLHF safety refusals. Instead of generating text and parsing a rating,
we do a single forward pass and extract the probability distribution
over tokens "1"-"5" at the completion position, yielding:
    - argmax rating (discrete 1-5)
    - expected rating (continuous, E[R] = sum(p_i * i))
    - full probability distribution over the 5-point scale

Design (matching original Gray et al.):
    - 12 entities (dropping "You"; original had 13)
    - 18 mental capacity surveys
    - 66 pairwise comparisons per capacity (12 choose 2)
    - Each pair in BOTH orders (position-bias control)
    - 5-point scale anchored by character NAMES
    - Total: 66 pairs x 2 orders x 18 capacities = 2,376 comparisons
    - Analysis: PCA with varimax rotation, regression factor scores

Key difference from chat version:
    - No chat template — plain text completion prompt
    - No system prompt — base model doesn't use instruction format
    - Logit-based rating extraction — no refusals, continuous values
    - Expected rating from probability distribution (more informative)

Output:
    data/behavioral_replication/{tag}/
        raw_responses.json
        character_means.npz
        pca_results.npz
    results/behavioral_replication/{tag}/
        results_summary.md

Usage:
    python 2_behavioral_replication.py
    python 2_behavioral_replication.py --include_self

SLURM:
    sbatch slurm/2_behavioral_replication.sh

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
from itertools import combinations
from datetime import datetime
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from entities.gray_entities import (
    GRAY_ET_AL_SCORES,
    CHARACTER_NAMES,
    CHARACTER_DESCRIPTIONS,
    CAPACITY_PROMPTS,
    CAPACITY_NAMES,
    ENTITY_NAMES,
)


# ========================== CONFIG ========================== #

MODEL_NAME = "meta-llama/Llama-2-13b-hf"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================== PROMPT FORMATTING ========================== #

def format_comparison_prompt(entity_a, entity_b, capacity_survey_desc):
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


# ========================== LOGIT EXTRACTION ========================== #

def get_rating_probs(model, tokenizer, prompt):
    """
    Single forward pass to extract probability distribution over
    tokens "1" through "5" at the next-token position.

    Returns:
        probs: np.array of shape (5,) — P("1"), P("2"), ..., P("5")
        expected: float — expected rating E[R] = sum(p_i * i)
        argmax: int — most probable rating (1-5)
    """
    # Get token IDs for "1" through "5"
    rating_token_ids = []
    for digit in ["1", "2", "3", "4", "5"]:
        # Encode just the digit — some tokenizers add BOS, so take last token
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
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
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


# ========================== ANALYSIS ========================== #

def compute_character_means(responses, entity_keys, capacity_keys,
                            use_expected=True):
    """
    Compute mean relative rating per entity per capacity.

    Following Gray et al.: "mean relative ratings were computed for each
    character across all respondents to that survey."

    For each comparison of entity A vs entity B with rating R (1-5):
        - Entity A gets score: (3 - R)   [positive if A rated higher]
        - Entity B gets score: (R - 3)   [positive if B rated higher]

    Args:
        use_expected: if True, use continuous expected rating; else argmax
    """
    n_cap = len(capacity_keys)
    n_ent = len(entity_keys)
    ent_to_idx = {k: i for i, k in enumerate(entity_keys)}
    rating_key = "expected_rating" if use_expected else "argmax_rating"

    scores = [[[] for _ in range(n_ent)] for _ in range(n_cap)]

    for resp in responses:
        cap_idx = capacity_keys.index(resp["capacity"])
        a_idx = ent_to_idx[resp["entity_a"]]
        b_idx = ent_to_idx[resp["entity_b"]]
        r = resp[rating_key]

        scores[cap_idx][a_idx].append(3 - r)
        scores[cap_idx][b_idx].append(r - 3)

    means = np.zeros((n_cap, n_ent))
    for c in range(n_cap):
        for e in range(n_ent):
            if scores[c][e]:
                means[c, e] = np.mean(scores[c][e])

    return means


def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """
    Varimax rotation of a factor loading matrix.

    Matches Gray et al.'s use of varimax rotation to maximize simple
    structure (each capacity loads highly on one factor).
    """
    n, k = loadings.shape
    rotation = np.eye(k)
    rotated = loadings.copy()

    for iteration in range(max_iter):
        old_rotated = rotated.copy()

        for i in range(k):
            for j in range(i + 1, k):
                x = rotated[:, i]
                y = rotated[:, j]

                u = x ** 2 - y ** 2
                v = 2 * x * y

                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)

                num = D - 2 * A * B / n
                den = C - (A ** 2 - B ** 2) / n

                phi = 0.25 * np.arctan2(num, den)

                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                new_i = rotated[:, i] * cos_phi + rotated[:, j] * sin_phi
                new_j = -rotated[:, i] * sin_phi + rotated[:, j] * cos_phi
                rotated[:, i] = new_i
                rotated[:, j] = new_j

                rot_ij = np.eye(k)
                rot_ij[i, i] = cos_phi
                rot_ij[j, j] = cos_phi
                rot_ij[i, j] = sin_phi
                rot_ij[j, i] = -sin_phi
                rotation = rotation @ rot_ij

        if np.max(np.abs(rotated - old_rotated)) < tol:
            break

    return rotated, rotation


def run_pca_varimax(means):
    """
    PCA with varimax rotation on the capacity-by-entity matrix.

    Following Gray et al. exactly:
    1. Correlations between capacities across characters
    2. PCA on the correlation matrix
    3. Retain factors with eigenvalue > 1
    4. Varimax rotation
    5. Regression-method factor scores
    6. Rescale to 0-1
    """
    n_cap, n_ent = means.shape

    corr_matrix = np.corrcoef(means)  # (n_cap, n_cap)

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_factors = np.sum(eigenvalues > 1.0)
    n_factors = max(n_factors, 2)
    print(f"  Eigenvalues: {eigenvalues[:5]}")
    print(f"  Factors retained (eigenvalue > 1): {n_factors}")

    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues[:n_factors] / total_var

    rotated_loadings, rotation_matrix = varimax_rotation(loadings)

    # Factor scores via regression method
    means_std = (means - means.mean(axis=1, keepdims=True))
    stds = means.std(axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-10)
    means_std = means_std / stds

    corr_inv = np.linalg.pinv(corr_matrix)
    score_coefficients = corr_inv @ rotated_loadings

    factor_scores = means_std.T @ score_coefficients  # (n_ent, n_factors)

    # Rescale to 0-1
    factor_scores_01 = np.zeros_like(factor_scores)
    for f in range(n_factors):
        fmin = factor_scores[:, f].min()
        fmax = factor_scores[:, f].max()
        if fmax - fmin > 1e-10:
            factor_scores_01[:, f] = (
                (factor_scores[:, f] - fmin) / (fmax - fmin)
            )

    return {
        "rotated_loadings": rotated_loadings,
        "unrotated_loadings": loadings,
        "factor_scores_raw": factor_scores,
        "factor_scores_01": factor_scores_01,
        "eigenvalues": eigenvalues,
        "explained_var_ratio": explained_var_ratio,
        "n_factors": n_factors,
        "rotation_matrix": rotation_matrix,
        "score_coefficients": score_coefficients,
    }


# ========================== RESULTS SUMMARY ========================== #

def write_results_summary(results_dir, responses, means, pca_results,
                          entity_keys, capacity_keys, human_correlations,
                          consistency_stats):
    """Write documented markdown summary."""
    n_ent = len(entity_keys)
    n_total = len(responses)
    n_factors = pca_results["n_factors"]

    path = os.path.join(results_dir, "results_summary.md")
    with open(path, "w") as f:
        f.write("# Experiment 4, Phase 2: Behavioral Replication of "
                "Gray et al. (2007)\n")
        f.write("## BASE MODEL (LLaMA-2-13B, no chat fine-tuning)\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n## What is being tested\n\n")
        f.write(
            "Does LLaMA-2-13B's (base, pretrained) **implicit folk psychology** "
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
            "- **Base model**: pretrained LLaMA-2-13B (no chat/RLHF)\n"
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
        f.write(f"- Mean expected rating: {np.mean([r['expected_rating'] for r in responses]):.2f}\n\n")

        # Argmax rating distribution
        f.write("### Argmax rating distribution\n\n")
        f.write("| Rating | Count | Pct |\n")
        f.write("|-------:|------:|----:|\n")
        for r in range(1, 6):
            count = sum(1 for resp in responses if resp["argmax_rating"] == r)
            pct = 100 * count / n_total
            f.write(f"| {r} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        # Position bias / consistency
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
        eigenvalues = pca_results["eigenvalues"]
        evr = pca_results["explained_var_ratio"]

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

        # Capacity loadings
        rotated = pca_results["rotated_loadings"]
        f.write("### Varimax-rotated capacity loadings\n\n")
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

        # Entity factor scores
        scores_01 = pca_results["factor_scores_01"]
        f.write("### Entity positions (factor scores, 0-1 scale)\n\n")
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

        # Alignment
        f.write("## Alignment with human Experience/Agency\n\n")
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

    print(f"  Results summary: {path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 2: Behavioral replication (BASE model)"
    )
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
    )
    parser.add_argument(
        "--single_order", action="store_true",
        help="Present each pair in one random order only"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (single_order mode)"
    )
    args = parser.parse_args()

    # Select entities
    if args.include_self:
        entity_keys = ENTITY_NAMES
    else:
        entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]

    capacity_keys = CAPACITY_NAMES
    n_entities = len(entity_keys)
    n_capacities = len(capacity_keys)
    n_pairs = n_entities * (n_entities - 1) // 2

    # Build pair list
    base_pairs = list(combinations(range(n_entities), 2))
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
    n_total = n_trials_per_cap * n_capacities

    print(f"Entities ({n_entities}): {entity_keys}")
    print(f"Capacities: {n_capacities}")
    print(f"Pairs: {n_pairs}, trials per capacity: {n_trials_per_cap}")
    print(f"Total comparisons: {n_total}")
    print(f"Counterbalanced: {not args.single_order}")

    # Output dirs
    tag = "with_self" if args.include_self else "without_self"
    data_dir = os.path.join("data", "behavioral_replication", tag)
    results_dir = os.path.join("results", "behavioral_replication", tag)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Tag: {tag}")
    print(f"Data dir: {data_dir}")
    print(f"Results dir: {results_dir}")

    # Survey descriptions
    survey_descriptions = {}
    for cap_key in capacity_keys:
        cap_prompt, _ = CAPACITY_PROMPTS[cap_key]
        survey_descriptions[cap_key] = (
            f"This survey asks you to judge {cap_prompt}."
        )

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.half().to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}.")

    # Verify rating token IDs
    for digit in ["1", "2", "3", "4", "5"]:
        ids = tokenizer.encode(digit, add_special_tokens=False)
        print(f"  Token '{digit}' -> ID {ids}")

    # Diagnostic: check top-5 predictions on the first prompt to verify
    # that digit tokens are natural completions after "Rating: "
    test_prompt = format_comparison_prompt(
        entity_keys[0], entity_keys[1],
        survey_descriptions[capacity_keys[0]]
    )
    test_enc = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        test_out = model(
            input_ids=test_enc["input_ids"].to(DEVICE),
            attention_mask=test_enc["attention_mask"].to(DEVICE),
        )
    test_logits = test_out.logits[0, -1, :]
    top5 = torch.topk(test_logits, 5)
    print("\n  Diagnostic — top 5 next-token predictions after 'Rating: ':")
    for rank, (logit, tid) in enumerate(zip(top5.values, top5.indices)):
        tok = tokenizer.decode([tid.item()])
        print(f"    {rank+1}. '{tok}' (id={tid.item()}, logit={logit.item():.2f})")
    print()

    # ── Run all comparisons ──
    all_responses = []
    n_done = 0

    for cap_idx, cap_key in enumerate(capacity_keys):
        _, cap_factor = CAPACITY_PROMPTS[cap_key]
        survey_desc = survey_descriptions[cap_key]
        print(f"\n[{cap_idx+1}/{n_capacities}] {cap_key} ({cap_factor})")

        for idx_a, idx_b in trial_pairs:
            entity_a = entity_keys[idx_a]
            entity_b = entity_keys[idx_b]

            prompt = format_comparison_prompt(
                entity_a, entity_b, survey_desc
            )
            probs, expected, argmax = get_rating_probs(
                model, tokenizer, prompt
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

            n_done += 1
            if n_done % 200 == 0:
                mean_exp = np.mean([r["expected_rating"]
                                    for r in all_responses])
                print(f"  {n_done}/{n_total} done "
                      f"(mean expected rating: {mean_exp:.2f})")

    print(f"\nCompleted: {n_done}")

    # Save raw responses
    with open(os.path.join(data_dir, "raw_responses.json"), "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"Saved raw responses to {data_dir}/")

    # ── Order consistency check ──
    consistency_stats = {"n_pairs_both": 0, "n_consistent": 0,
                         "mean_deviation": 0.0, "pct_consistent": 0.0}

    if not args.single_order:
        from collections import defaultdict
        pair_groups = defaultdict(list)
        for resp in all_responses:
            pair_key = (resp["capacity"],
                        frozenset([resp["entity_a"], resp["entity_b"]]))
            pair_groups[pair_key].append(resp)

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
            "pct_consistent": 100 * n_consistent_argmax / max(n_both, 1),
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

    # ── Character means (using expected ratings) ──
    means = compute_character_means(
        all_responses, entity_keys, capacity_keys, use_expected=True
    )
    print(f"\nCharacter means: {means.shape} (capacities x entities)")

    # ── PCA with varimax ──
    print("\nPCA with varimax rotation:")
    pca_results = run_pca_varimax(means)

    evr = pca_results["explained_var_ratio"]
    for i in range(min(3, len(evr))):
        print(f"  Factor {i+1}: {evr[i]*100:.1f}%")

    # ── Correlate with human scores ──
    human_exp = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    human_ag = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])
    scores_01 = pca_results["factor_scores_01"]
    n_factors = min(2, scores_01.shape[1])

    human_correlations = {}
    print("\nCorrelation with human factor scores (0-1 scaled):")
    for fi in range(n_factors):
        rho_e, p_e = spearmanr(scores_01[:, fi], human_exp)
        rho_a, p_a = spearmanr(scores_01[:, fi], human_ag)
        human_correlations[f"f{fi+1}_experience"] = {
            "rho": float(rho_e), "p": float(p_e)
        }
        human_correlations[f"f{fi+1}_agency"] = {
            "rho": float(rho_a), "p": float(p_a)
        }
        print(f"  Factor {fi+1} vs Experience: rho={rho_e:+.3f} (p={p_e:.4f})")
        print(f"  Factor {fi+1} vs Agency:     rho={rho_a:+.3f} (p={p_a:.4f})")

    # ── Save ──
    np.savez_compressed(
        os.path.join(data_dir, "character_means.npz"),
        means=means,
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    np.savez_compressed(
        os.path.join(data_dir, "pca_results.npz"),
        rotated_loadings=pca_results["rotated_loadings"],
        unrotated_loadings=pca_results["unrotated_loadings"],
        factor_scores_raw=pca_results["factor_scores_raw"],
        factor_scores_01=pca_results["factor_scores_01"],
        eigenvalues=pca_results["eigenvalues"],
        explained_var_ratio=pca_results["explained_var_ratio"],
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    with open(os.path.join(data_dir, "human_correlations.json"), "w") as f:
        json.dump(human_correlations, f, indent=2)
    with open(os.path.join(data_dir, "consistency_stats.json"), "w") as f:
        json.dump(consistency_stats, f, indent=2)

    # Results summary
    write_results_summary(
        results_dir, all_responses, means, pca_results,
        entity_keys, capacity_keys, human_correlations,
        consistency_stats
    )

    print(f"\nSaved data to {data_dir}/")
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    main()
