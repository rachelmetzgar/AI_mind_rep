#!/usr/bin/env python3
"""
Experiment 4, Phase 2: Behavioral Replication of Gray et al. (2007)

Direct replication of the Dimensions of Mind Perception study using
LLaMA-2-13B-Chat as a participant. The model rates pairwise comparisons
of 12 entities on 18 mental capacities, closely matching the original
survey procedure from Gray, Gray, & Wegner (2007, Science).

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

Output:
    data/behavioral_replication/
        raw_responses.json           # all 2376 responses + parsed ratings
        character_means.npz          # (18 capacities, 12 entities)
        pca_results.npz              # varimax-rotated loadings + scores
    results/behavioral_replication/
        results_summary.md           # documented methodology + results

Usage:
    python 2_behavioral_replication.py
    python 2_behavioral_replication.py --include_self
    python 2_behavioral_replication.py --single_order   # skip counterbalancing

SLURM:
    sbatch slurm/2_behavioral_replication.sh

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import re
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

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50


# ========================== PROMPT FORMATTING ========================== #

# Minimal system prompt — does not define the scale (scale appears in each
# comparison, matching the original where the scale was part of the visual
# display, not a separate instruction).
SYSTEM_PROMPT = (
    "You are completing a survey about the mental capacities of different "
    "characters. For each comparison, give your honest judgment. Respond "
    "with only a number from 1 to 5."
)


def llama_v2_prompt(messages, system_prompt=None):
    """Format messages into LLaMA-2-Chat token string."""
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    if system_prompt:
        default_sys = system_prompt
    else:
        default_sys = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Please ensure that your responses are socially unbiased and "
            "positive in nature. If a question does not make any sense, or "
            "is not factually coherent, explain why instead of answering "
            "something not correct. If you don't know the answer to a "
            "question, please don't share false information."
        )

    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": default_sys}] + messages

    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS
                       + messages[1]["content"],
        }
    ] + messages[2:]

    parts = [
        f"{BOS}{B_INST} {prompt['content'].strip()} {E_INST} "
        f"{answer['content'].strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    if messages[-1]["role"] == "user":
        parts.append(
            f"{BOS}{B_INST} {messages[-1]['content'].strip()} {E_INST}"
        )

    return "".join(parts)


def format_comparison_prompt(entity_a, entity_b, capacity_survey_desc):
    """
    Format a single pairwise comparison prompt.

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


# ========================== GENERATION ========================== #

def generate_response(model, tokenizer, user_prompt):
    """Generate a response from LLaMA-2-Chat."""
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
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
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


# ========================== ANALYSIS ========================== #

def compute_character_means(responses, entity_keys, capacity_keys):
    """
    Compute mean relative rating per entity per capacity.

    Following Gray et al.: "mean relative ratings were computed for each
    character across all respondents to that survey."

    For each comparison of entity A vs entity B with rating R (1-5):
        - Entity A gets score: (3 - R)   [positive if A rated higher]
        - Entity B gets score: (R - 3)   [positive if B rated higher]

    With counterbalanced orders, each pair contributes two scores per
    entity (one from each order), which averages out position bias.

    Returns: (n_capacities, n_entities) array of mean relative ratings.
    """
    n_cap = len(capacity_keys)
    n_ent = len(entity_keys)
    ent_to_idx = {k: i for i, k in enumerate(entity_keys)}

    scores = [[[] for _ in range(n_ent)] for _ in range(n_cap)]

    for resp in responses:
        if resp["rating"] is None:
            continue
        cap_idx = capacity_keys.index(resp["capacity"])
        a_idx = ent_to_idx[resp["entity_a"]]
        b_idx = ent_to_idx[resp["entity_b"]]
        r = resp["rating"]

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

    Args:
        loadings: (n_variables, n_factors) loading matrix
        max_iter: maximum iterations
        tol: convergence tolerance

    Returns:
        rotated_loadings: (n_variables, n_factors) rotated loading matrix
        rotation_matrix: (n_factors, n_factors) rotation matrix applied
    """
    n, k = loadings.shape
    rotation = np.eye(k)
    rotated = loadings.copy()

    for iteration in range(max_iter):
        old_rotated = rotated.copy()

        for i in range(k):
            for j in range(i + 1, k):
                # Kaiser's varimax criterion
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

                # Apply rotation
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                new_i = rotated[:, i] * cos_phi + rotated[:, j] * sin_phi
                new_j = -rotated[:, i] * sin_phi + rotated[:, j] * cos_phi
                rotated[:, i] = new_i
                rotated[:, j] = new_j

                # Track cumulative rotation
                rot_ij = np.eye(k)
                rot_ij[i, i] = cos_phi
                rot_ij[j, j] = cos_phi
                rot_ij[i, j] = sin_phi
                rot_ij[j, i] = -sin_phi
                rotation = rotation @ rot_ij

        # Check convergence
        if np.max(np.abs(rotated - old_rotated)) < tol:
            break

    return rotated, rotation


def run_pca_varimax(means):
    """
    PCA with varimax rotation on the capacity-by-entity matrix.

    Following Gray et al. exactly:
    1. Compute correlations between capacities across characters
    2. PCA on the correlation matrix
    3. Retain factors with eigenvalue > 1
    4. Varimax rotation
    5. Regression-method factor scores
    6. Rescale factor scores to 0-1

    Args:
        means: (n_capacities, n_entities) character means

    Returns dict with rotated loadings, factor scores, explained variance.
    """
    n_cap, n_ent = means.shape

    # Step 1: Correlation matrix between capacities across characters
    # Each row of means is one capacity's scores across entities
    corr_matrix = np.corrcoef(means)  # (n_cap, n_cap)

    # Step 2: Eigendecomposition of correlation matrix
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)

    # Sort by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Step 3: Retain factors with eigenvalue > 1 (Kaiser criterion)
    n_factors = np.sum(eigenvalues > 1.0)
    n_factors = max(n_factors, 2)  # keep at least 2 for comparison
    print(f"  Eigenvalues: {eigenvalues[:5]}")
    print(f"  Factors retained (eigenvalue > 1): {n_factors}")

    # Unrotated loadings = eigenvectors * sqrt(eigenvalues)
    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    # Explained variance
    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues[:n_factors] / total_var

    # Step 4: Varimax rotation
    rotated_loadings, rotation_matrix = varimax_rotation(loadings)

    # Step 5: Factor scores via regression method
    # Score = Z @ R^{-1} @ L   where Z = standardized data, R = correlation
    # matrix, L = rotated loadings
    means_std = (means - means.mean(axis=1, keepdims=True))
    stds = means.std(axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-10)
    means_std = means_std / stds

    # Regression factor score coefficients: R^{-1} @ L
    corr_inv = np.linalg.pinv(corr_matrix)
    score_coefficients = corr_inv @ rotated_loadings  # (n_cap, n_factors)

    # Factor scores for each entity
    factor_scores = means_std.T @ score_coefficients  # (n_ent, n_factors)

    # Step 6: Rescale to 0-1 (matching Gray et al. Figure 1)
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
    n_cap = len(capacity_keys)
    n_total = len(responses)
    n_parsed = sum(1 for r in responses if r["rating"] is not None)
    n_factors = pca_results["n_factors"]

    path = os.path.join(results_dir, "results_summary.md")
    with open(path, "w") as f:
        f.write("# Experiment 4, Phase 2: Behavioral Replication of "
                "Gray et al. (2007)\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # ── What is being tested ──
        f.write("---\n\n## What is being tested\n\n")
        f.write(
            "Does LLaMA-2-13B-Chat's **explicit folk psychology** of mind "
            "perception match the human folk psychology described by Gray, "
            "Gray, & Wegner (2007, Science)?\n\n"
            "This is a direct behavioral replication: the model answers the "
            "same pairwise comparison questions that ~2,400 human participants "
            "answered in the original study, using the exact character "
            "descriptions and mental capacity survey prompts from the "
            "supplementary materials.\n\n"
        )

        # ── What matches / what differs ──
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

        # ── Parse stats ──
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

        # Position bias / consistency
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

        # ── PCA results ──
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

        # ── Alignment with human scores ──
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

    print(f"  Results summary: {path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 2: Behavioral replication of Gray et al."
    )
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
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

    print(f"Entities ({n_entities}): {entity_keys}")
    print(f"Capacities: {n_capacities}")
    print(f"Pairs: {n_pairs}, trials per capacity: {n_trials_per_cap}")
    print(f"Total comparisons: {n_total}")
    print(f"Counterbalanced: {not args.single_order}")

    # Output dirs (tagged by condition)
    if args.include_self:
        tag = "with_self"
    else:
        tag = "without_self"
    data_dir = os.path.join("data", "behavioral_replication", tag)
    results_dir = os.path.join("results", "behavioral_replication", tag)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Tag: {tag}")
    print(f"Data dir: {data_dir}")
    print(f"Results dir: {results_dir}")

    # Build survey descriptions (matching Appendix B format)
    survey_descriptions = {}
    for cap_key in capacity_keys:
        cap_prompt, _ = CAPACITY_PROMPTS[cap_key]
        survey_descriptions[cap_key] = (
            f"This survey asks you to judge {cap_prompt}."
        )

    # Load model
    print("\nLoading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}.")

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

            user_prompt = format_comparison_prompt(
                entity_a, entity_b, survey_desc
            )
            response = generate_response(model, tokenizer, user_prompt)
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

            n_done += 1
            if rating is not None:
                n_parsed += 1

            if n_done % 200 == 0:
                print(f"  {n_done}/{n_total} done "
                      f"({100*n_parsed/n_done:.1f}% parsed)")

    print(f"\nCompleted: {n_done}, parsed: {n_parsed} "
          f"({100*n_parsed/max(n_done,1):.1f}%)")

    # Save raw responses
    with open(os.path.join(data_dir, "raw_responses.json"), "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"Saved raw responses to {data_dir}/")

    # ── Order consistency check ──
    consistency_stats = {"n_pairs_both": 0, "n_consistent": 0,
                         "mean_deviation": 0.0, "pct_consistent": 0.0}

    if not args.single_order:
        deviations = []
        # Group responses by (capacity, entity pair as frozenset)
        from collections import defaultdict
        pair_groups = defaultdict(list)
        for resp in all_responses:
            if resp["rating"] is None:
                continue
            pair_key = (resp["capacity"],
                        frozenset([resp["entity_a"], resp["entity_b"]]))
            pair_groups[pair_key].append(resp)

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

    # ── Character means ──
    means = compute_character_means(all_responses, entity_keys, capacity_keys)
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
