#!/usr/bin/env python3
"""
Experiment 4, Phase 2c: Individual Likert Ratings
BASE MODEL version — LLaMA-2-13B (no chat fine-tuning)

Alternative to pairwise comparisons: rate each entity individually on each
mental capacity using a 1-5 Likert scale. Avoids pairwise position bias.

Design:
    - 12 entities (or 13 with self)
    - 18 mental capacities
    - Each entity rated on each capacity: 12 × 18 = 216 ratings (or 234)
    - 5-point scale: 1 = Not at all capable ... 5 = Extremely capable
    - Logit-based extraction (same as pairwise version)
    - Analysis: PCA with varimax rotation on the 18 × N_entities matrix

Note: This deviates from Gray et al.'s pairwise methodology but avoids
position bias entirely. If results are cleaner, it suggests the pairwise
format is the problem, not the model's representations.

Output:
    data/individual_ratings/{tag}/
        raw_responses.json
        rating_matrix.npz
        pca_results.npz
    results/individual_ratings/{tag}/
        results_summary.md

Usage:
    python 2c_individual_ratings.py
    python 2c_individual_ratings.py --include_self

SLURM:
    sbatch slurm/2c_individual_ratings.sh

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
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

def format_individual_prompt(entity_key, capacity_desc):
    """
    Format an individual Likert rating prompt for the base model.
    """
    name = CHARACTER_NAMES[entity_key]
    desc = CHARACTER_DESCRIPTIONS[entity_key]

    prompt = (
        f"Survey: Mental Capacity Rating\n\n"
        f"Character: {name}. {desc}\n\n"
        f"Question: How capable is {name} of {capacity_desc}?\n\n"
        f"1 = Not at all capable\n"
        f"2 = Slightly capable\n"
        f"3 = Moderately capable\n"
        f"4 = Very capable\n"
        f"5 = Extremely capable\n\n"
        f"Rating: "
    )
    return prompt


# ========================== LOGIT EXTRACTION ========================== #

def get_rating_probs(model, tokenizer, prompt):
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
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
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


# ========================== ANALYSIS ========================== #

def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    n, k = loadings.shape
    rotation = np.eye(k)
    rotated = loadings.copy()
    for iteration in range(max_iter):
        old_rotated = rotated.copy()
        for i in range(k):
            for j in range(i + 1, k):
                x, y = rotated[:, i], rotated[:, j]
                u = x**2 - y**2
                v = 2 * x * y
                A, B = np.sum(u), np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)
                phi = 0.25 * np.arctan2(D - 2*A*B/n, C - (A**2 - B**2)/n)
                cos_phi, sin_phi = np.cos(phi), np.sin(phi)
                new_i = rotated[:, i]*cos_phi + rotated[:, j]*sin_phi
                new_j = -rotated[:, i]*sin_phi + rotated[:, j]*cos_phi
                rotated[:, i], rotated[:, j] = new_i, new_j
                rot_ij = np.eye(k)
                rot_ij[i,i] = rot_ij[j,j] = cos_phi
                rot_ij[i,j] = sin_phi
                rot_ij[j,i] = -sin_phi
                rotation = rotation @ rot_ij
        if np.max(np.abs(rotated - old_rotated)) < tol:
            break
    return rotated, rotation


def run_pca_varimax(rating_matrix):
    """
    PCA with varimax rotation.

    Input: rating_matrix of shape (n_capacities, n_entities)
    Same pipeline as pairwise version — correlate capacities, PCA, varimax.
    """
    n_cap, n_ent = rating_matrix.shape

    corr_matrix = np.corrcoef(rating_matrix)  # (n_cap, n_cap)
    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_factors = max(np.sum(eigenvalues > 1.0), 2)
    print(f"  Eigenvalues: {eigenvalues[:5]}")
    print(f"  Factors retained (eigenvalue > 1): {n_factors}")

    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])
    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues[:n_factors] / total_var

    rotated_loadings, rotation_matrix = varimax_rotation(loadings)

    # Factor scores via regression method
    means_std = (rating_matrix - rating_matrix.mean(axis=1, keepdims=True))
    stds = np.maximum(rating_matrix.std(axis=1, keepdims=True), 1e-10)
    means_std = means_std / stds

    corr_inv = np.linalg.pinv(corr_matrix)
    score_coefficients = corr_inv @ rotated_loadings
    factor_scores = means_std.T @ score_coefficients  # (n_ent, n_factors)

    # Rescale to 0-1
    factor_scores_01 = np.zeros_like(factor_scores)
    for f in range(n_factors):
        fmin, fmax = factor_scores[:, f].min(), factor_scores[:, f].max()
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

def write_results_summary(results_dir, responses, rating_matrix, pca_results,
                          entity_keys, capacity_keys, human_correlations):
    n_ent = len(entity_keys)
    n_total = len(responses)
    n_factors = pca_results["n_factors"]

    path = os.path.join(results_dir, "results_summary.md")
    with open(path, "w") as f:
        f.write("# Experiment 4, Phase 2c: Individual Likert Ratings\n")
        f.write("## BASE MODEL (LLaMA-2-13B, no chat fine-tuning)\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n## Method\n\n")
        f.write(
            "Instead of pairwise comparisons (Gray et al. methodology), each entity "
            "is rated individually on each mental capacity using a 1-5 Likert scale. "
            "This avoids pairwise position bias entirely.\n\n"
            "The rating matrix (capacities x entities) is analyzed with PCA + varimax "
            "rotation, same as the pairwise version.\n\n"
        )

        f.write("## Response statistics\n\n")
        f.write(f"- Total ratings: {n_total}\n")
        f.write(f"- Entities: {n_ent}\n")
        f.write(f"- Capacities: {len(capacity_keys)}\n\n")

        # Rating distribution
        all_probs = np.array([r["probs"] for r in responses])
        max_probs = all_probs.max(axis=1)
        f.write("### Probability concentration\n\n")
        for threshold in [0.5, 0.7, 0.9]:
            pct = 100 * np.mean(max_probs >= threshold)
            f.write(f"- P(top rating) >= {threshold}: {pct:.1f}%\n")
        f.write(f"- Mean max P: {np.mean(max_probs):.3f}\n")
        f.write(f"- Mean expected rating: "
                f"{np.mean([r['expected_rating'] for r in responses]):.3f}\n\n")

        f.write("### Expected rating distribution\n\n")
        f.write("| Rating | Count | Pct |\n")
        f.write("|-------:|------:|----:|\n")
        for r in range(1, 6):
            count = sum(1 for resp in responses if resp["argmax_rating"] == r)
            pct = 100 * count / n_total
            f.write(f"| {r} | {count} | {pct:.1f}% |\n")
        f.write("\n")

        # Rating matrix
        f.write("### Rating matrix (expected ratings)\n\n")
        f.write("| Capacity |")
        for ent in entity_keys:
            f.write(f" {ent[:8]:>8s} |")
        f.write("\n|----------|")
        for _ in entity_keys:
            f.write("---------:|")
        f.write("\n")
        for c_idx, cap in enumerate(capacity_keys):
            f.write(f"| {cap[:8]:8s} |")
            for e_idx in range(n_ent):
                f.write(f" {rating_matrix[c_idx, e_idx]:8.3f} |")
            f.write("\n")
        f.write("\n")

        # PCA
        eigenvalues = pca_results["eigenvalues"]
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
        f.write("\n")

        # Loadings
        rotated = pca_results["rotated_loadings"]
        f.write("### Varimax-rotated capacity loadings\n\n")
        f.write("| Capacity | Human factor | F1 | F2 |\n")
        f.write("|----------|:------------:|---:|---:|\n")
        for c_idx, cap_key in enumerate(capacity_keys):
            _, factor = CAPACITY_PROMPTS[cap_key]
            n_cols = min(2, rotated.shape[1])
            l1 = rotated[c_idx, 0] if n_cols > 0 else 0
            l2 = rotated[c_idx, 1] if n_cols > 1 else 0
            f.write(f"| {cap_key} | {factor} | {l1:+.3f} | {l2:+.3f} |\n")
        f.write("\n")

        # Entity scores
        scores_01 = pca_results["factor_scores_01"]
        n_f = min(2, scores_01.shape[1])
        f.write("### Entity positions (0-1 scale)\n\n")
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
            rho_e = human_correlations[f"f{fi+1}_experience"]["rho"]
            p_e = human_correlations[f"f{fi+1}_experience"]["p"]
            rho_a = human_correlations[f"f{fi+1}_agency"]["rho"]
            p_a = human_correlations[f"f{fi+1}_agency"]["p"]
            f.write(f"| Factor {fi+1} | rho={rho_e:+.3f} (p={p_e:.4f}) | "
                    f"rho={rho_a:+.3f} (p={p_a:.4f}) |\n")
        f.write("\n")

    print(f"  Results summary: {path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 2c: Individual Likert ratings (BASE model)"
    )
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
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
    n_total = n_entities * n_capacities

    print(f"Entities ({n_entities}): {entity_keys}")
    print(f"Capacities: {n_capacities}")
    print(f"Total ratings: {n_total}")

    # Output dirs
    tag = "with_self" if args.include_self else "without_self"
    data_dir = os.path.join("data", "individual_ratings", tag)
    results_dir = os.path.join("results", "individual_ratings", tag)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Capacity descriptions (strip "which character is more capable of")
    capacity_descs = {}
    for cap_key in capacity_keys:
        cap_prompt, _ = CAPACITY_PROMPTS[cap_key]
        # Original: "which character is more capable of feeling hungry"
        # Extract: "feeling hungry"
        desc = cap_prompt.replace("which character is more capable of ", "")
        capacity_descs[cap_key] = desc

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.half().to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}.")

    # Verify token IDs
    for digit in ["1", "2", "3", "4", "5"]:
        ids = tokenizer.encode(digit, add_special_tokens=False)
        print(f"  Token '{digit}' -> ID {ids}")

    # Diagnostic
    test_prompt = format_individual_prompt(entity_keys[0], capacity_descs[capacity_keys[0]])
    test_enc = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        test_out = model(
            input_ids=test_enc["input_ids"].to(DEVICE),
            attention_mask=test_enc["attention_mask"].to(DEVICE),
        )
    test_logits = test_out.logits[0, -1, :]
    top5 = torch.topk(test_logits, 5)
    print(f"\n  Diagnostic — top 5 next-token predictions after 'Rating: ':")
    for rank, (logit, tid) in enumerate(zip(top5.values, top5.indices)):
        tok = tokenizer.decode([tid.item()])
        print(f"    {rank+1}. '{tok}' (id={tid.item()}, logit={logit.item():.2f})")

    print(f"\n  Example prompt:\n{'='*60}")
    print(test_prompt)
    print(f"{'='*60}\n")

    # ── Run all ratings ──
    all_responses = []
    rating_matrix = np.zeros((n_capacities, n_entities))

    for cap_idx, cap_key in enumerate(capacity_keys):
        _, cap_factor = CAPACITY_PROMPTS[cap_key]
        cap_desc = capacity_descs[cap_key]
        print(f"[{cap_idx+1}/{n_capacities}] {cap_key} ({cap_factor})")

        for ent_idx, ent_key in enumerate(entity_keys):
            prompt = format_individual_prompt(ent_key, cap_desc)
            probs, expected, argmax = get_rating_probs(model, tokenizer, prompt)

            all_responses.append({
                "capacity": cap_key,
                "capacity_factor": cap_factor,
                "entity": ent_key,
                "name": CHARACTER_NAMES[ent_key],
                "probs": probs.tolist(),
                "expected_rating": expected,
                "argmax_rating": argmax,
            })

            rating_matrix[cap_idx, ent_idx] = expected

        # Print row summary
        row = rating_matrix[cap_idx, :]
        print(f"  mean={row.mean():.2f}, min={row.min():.2f} ({entity_keys[row.argmin()]}), "
              f"max={row.max():.2f} ({entity_keys[row.argmax()]})")

    print(f"\nCompleted: {len(all_responses)} ratings")

    # Save raw
    with open(os.path.join(data_dir, "raw_responses.json"), "w") as f:
        json.dump(all_responses, f, indent=2)

    # ── PCA ──
    print(f"\nRating matrix: {rating_matrix.shape}")
    print(f"  Grand mean: {rating_matrix.mean():.3f}")
    print(f"  Std: {rating_matrix.std():.3f}")
    print(f"  Range: [{rating_matrix.min():.3f}, {rating_matrix.max():.3f}]")

    print("\nPCA with varimax rotation:")
    pca_results = run_pca_varimax(rating_matrix)

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
        os.path.join(data_dir, "rating_matrix.npz"),
        rating_matrix=rating_matrix,
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    np.savez_compressed(
        os.path.join(data_dir, "pca_results.npz"),
        rotated_loadings=pca_results["rotated_loadings"],
        factor_scores_raw=pca_results["factor_scores_raw"],
        factor_scores_01=pca_results["factor_scores_01"],
        eigenvalues=pca_results["eigenvalues"],
        explained_var_ratio=pca_results["explained_var_ratio"],
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    with open(os.path.join(data_dir, "human_correlations.json"), "w") as f:
        json.dump(human_correlations, f, indent=2)

    write_results_summary(
        results_dir, all_responses, rating_matrix, pca_results,
        entity_keys, capacity_keys, human_correlations
    )

    print(f"\nSaved data to {data_dir}/")
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    main()
