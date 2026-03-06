#!/usr/bin/env python3
"""
Experiment 4: Individual Likert Ratings
BASE MODEL version -- LLaMA-2-13B (no chat fine-tuning)

Alternative to pairwise comparisons: rate each entity individually on each
mental capacity using a 1-5 Likert scale. Avoids pairwise position bias.

Design:
    - 12 entities (or 13 with --include_self)
    - 18 mental capacities
    - Each entity rated on each capacity: 12 x 18 = 216 ratings (or 234)
    - 5-point scale: 1 = Not at all capable ... 5 = Extremely capable
    - Logit-based extraction (same as pairwise base version)
    - Analysis: PCA with varimax rotation on the 18 x N_entities matrix

Note: This deviates from Gray et al.'s pairwise methodology but avoids
position bias entirely. If results are cleaner, it suggests the pairwise
format is the problem, not the model's representations.

Output (individual_ prefix):
    data_dir("behavior", tag)/
        individual_raw_responses.json
        individual_rating_matrix.npz
        individual_pca_results.npz
        individual_human_correlations.json
    results_phase_dir("behavior", tag)/
        individual_results_summary.md

Usage:
    python 3_individual_ratings.py --model base
    python 3_individual_ratings.py --model base --include_self
    python 3_individual_ratings.py --model base --both

Env: llama2_env
Rachel C. Metzgar / Mar 2026
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

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir, get_device, get_condition_tag,
)
from src.utils import varimax_rotation, run_pca_varimax, correlate_with_humans
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from entities.gray_entities import (
    GRAY_ET_AL_SCORES, CHARACTER_NAMES, CHARACTER_DESCRIPTIONS,
    CAPACITY_PROMPTS, CAPACITY_NAMES, ENTITY_NAMES,
)


# ========================== PROMPT FORMATTING ========================== #

def format_individual_prompt(entity_key, capacity_desc):
    """
    Format an individual Likert rating prompt for the base model.

    The prompt ends with "Rating: " (trailing space) for natural
    digit completion.
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

def get_rating_probs(model, tokenizer, prompt, device):
    """
    Single forward pass to extract probability distribution over
    tokens "1" through "5" at the next-token position.

    Returns:
        probs: np.array of shape (5,) -- P("1"), P("2"), ..., P("5")
        expected: float -- expected rating E[R] = sum(p_i * i)
        argmax: int -- most probable rating (1-5)
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


# ========================== RESULTS SUMMARY ========================== #

def write_results_summary(out_dir, responses, rating_matrix, pca_results,
                          entity_keys, capacity_keys, human_correlations):
    """Write documented markdown summary of individual rating results."""
    n_ent = len(entity_keys)
    n_total = len(responses)
    n_factors = pca_results["n_factors"]
    model_label = config.MODEL_LABEL

    path = os.path.join(str(out_dir), "individual_results_summary.md")
    with open(path, "w") as f:
        f.write("# Experiment 4: Individual Likert Ratings\n")
        f.write(f"## {model_label}\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n## Method\n\n")
        f.write(
            "Instead of pairwise comparisons (Gray et al. methodology), each "
            "entity is rated individually on each mental capacity using a 1-5 "
            "Likert scale. This avoids pairwise position bias entirely.\n\n"
            "The rating matrix (capacities x entities) is analyzed with PCA + "
            "varimax rotation, same as the pairwise version.\n\n"
        )

        f.write("## Response statistics\n\n")
        f.write(f"- Total ratings: {n_total}\n")
        f.write(f"- Entities: {n_ent}\n")
        f.write(f"- Capacities: {len(capacity_keys)}\n\n")

        # Probability concentration
        all_probs = np.array([r["probs"] for r in responses])
        max_probs = all_probs.max(axis=1)
        f.write("### Probability concentration\n\n")
        for threshold in [0.5, 0.7, 0.9]:
            pct = 100 * np.mean(max_probs >= threshold)
            f.write(f"- P(top rating) >= {threshold}: {pct:.1f}%\n")
        f.write(f"- Mean max P: {np.mean(max_probs):.3f}\n")
        f.write(f"- Mean expected rating: "
                f"{np.mean([r['expected_rating'] for r in responses]):.3f}\n\n")

        # Argmax distribution
        f.write("### Argmax rating distribution\n\n")
        f.write("| Rating | Count | Pct |\n")
        f.write("|-------:|------:|----:|\n")
        for r in range(1, 6):
            count = sum(1 for resp in responses
                        if resp["argmax_rating"] == r)
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
            f.write(f"| {cap_key} | {factor} | "
                    f"{l1:+.3f} | {l2:+.3f} |\n")
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


# ========================== RUN ONE CONDITION ========================== #

def run_condition(model, tokenizer, device, entity_keys, capacity_keys,
                  capacity_descs, tag):
    """Run individual ratings for one condition (with/without self)."""
    n_entities = len(entity_keys)
    n_capacities = len(capacity_keys)
    n_total = n_entities * n_capacities

    d_dir = data_dir("behavior", tag)
    r_dir = results_phase_dir("behavior", tag)

    print(f"\n{'='*70}")
    print(f"  CONDITION: {tag} ({n_entities} entities, {n_total} ratings)")
    print(f"{'='*70}")

    # Run all ratings
    all_responses = []
    rating_matrix = np.zeros((n_capacities, n_entities))

    for cap_idx, cap_key in enumerate(capacity_keys):
        _, cap_factor = CAPACITY_PROMPTS[cap_key]
        cap_desc = capacity_descs[cap_key]
        print(f"[{cap_idx+1}/{n_capacities}] {cap_key} ({cap_factor})")

        for ent_idx, ent_key in enumerate(entity_keys):
            prompt = format_individual_prompt(ent_key, cap_desc)
            probs, expected, argmax = get_rating_probs(
                model, tokenizer, prompt, device
            )

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
        print(f"  mean={row.mean():.2f}, "
              f"min={row.min():.2f} ({entity_keys[row.argmin()]}), "
              f"max={row.max():.2f} ({entity_keys[row.argmax()]})")

    print(f"\nCompleted: {len(all_responses)} ratings")

    # Save raw
    with open(os.path.join(str(d_dir), "individual_raw_responses.json"), "w") as f:
        json.dump(all_responses, f, indent=2)

    # PCA
    print(f"\nRating matrix: {rating_matrix.shape}")
    print(f"  Grand mean: {rating_matrix.mean():.3f}")
    print(f"  Std: {rating_matrix.std():.3f}")
    print(f"  Range: [{rating_matrix.min():.3f}, {rating_matrix.max():.3f}]")

    print("\nPCA with varimax rotation:")
    pca_results = run_pca_varimax(rating_matrix)

    evr = pca_results["explained_var_ratio"]
    for i in range(min(3, len(evr))):
        print(f"  Factor {i+1}: {evr[i]*100:.1f}%")

    # Correlate with human scores
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

    # Save
    np.savez_compressed(
        os.path.join(str(d_dir), "individual_rating_matrix.npz"),
        rating_matrix=rating_matrix,
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    np.savez_compressed(
        os.path.join(str(d_dir), "individual_pca_results.npz"),
        rotated_loadings=pca_results["rotated_loadings"],
        unrotated_loadings=pca_results.get("unrotated_loadings",
                                           pca_results["rotated_loadings"]),
        factor_scores_raw=pca_results["factor_scores_raw"],
        factor_scores_01=pca_results["factor_scores_01"],
        eigenvalues=pca_results["eigenvalues"],
        explained_var_ratio=pca_results["explained_var_ratio"],
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    with open(os.path.join(str(d_dir),
              "individual_human_correlations.json"), "w") as f:
        json.dump(human_correlations, f, indent=2)

    write_results_summary(
        r_dir, all_responses, rating_matrix, pca_results,
        entity_keys, capacity_keys, human_correlations
    )

    print(f"\nSaved data to {d_dir}/")
    print(f"Saved results to {r_dir}/")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Individual Likert ratings (base model only)"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both with_self and without_self conditions"
    )
    args = parser.parse_args()

    if args.model != "base":
        parser.error("Individual ratings requires --model base "
                      "(uses logit-based extraction)")

    # Initialize model config
    set_model(args.model)
    DEVICE = get_device()

    print(f"Model: {config.MODEL_LABEL}")
    print(f"Device: {DEVICE}")

    capacity_keys = CAPACITY_NAMES

    # Capacity descriptions (strip pairwise phrasing)
    capacity_descs = {}
    for cap_key in capacity_keys:
        cap_prompt, _ = CAPACITY_PROMPTS[cap_key]
        desc = cap_prompt.replace(
            "which character is more capable of ", ""
        )
        capacity_descs[cap_key] = desc

    # Load model
    print(f"\nLoading {config.MODEL_LABEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model.half().to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}.")

    # Verify token IDs
    for digit in ["1", "2", "3", "4", "5"]:
        ids = tokenizer.encode(digit, add_special_tokens=False)
        print(f"  Token '{digit}' -> ID {ids}")

    # Diagnostic
    diag_entity_keys = ENTITY_NAMES if (args.include_self or args.both) else \
        [k for k in ENTITY_NAMES if k != "you_self"]
    test_prompt = format_individual_prompt(
        diag_entity_keys[0], capacity_descs[capacity_keys[0]]
    )
    test_enc = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        test_out = model(
            input_ids=test_enc["input_ids"].to(DEVICE),
            attention_mask=test_enc["attention_mask"].to(DEVICE),
        )
    test_logits = test_out.logits[0, -1, :]
    top5 = torch.topk(test_logits, 5)
    print(f"\n  Diagnostic -- top 5 next-token predictions after 'Rating: ':")
    for rank, (logit, tid) in enumerate(zip(top5.values, top5.indices)):
        tok = tokenizer.decode([tid.item()])
        print(f"    {rank+1}. '{tok}' (id={tid.item()}, "
              f"logit={logit.item():.2f})")

    print(f"\n  Example prompt:\n{'='*60}")
    print(test_prompt)
    print(f"{'='*60}\n")

    # Run condition(s)
    if args.both:
        entity_keys_no_self = [k for k in ENTITY_NAMES if k != "you_self"]
        run_condition(model, tokenizer, DEVICE,
                      entity_keys_no_self, capacity_keys, capacity_descs,
                      "without_self")
        run_condition(model, tokenizer, DEVICE,
                      ENTITY_NAMES, capacity_keys, capacity_descs,
                      "with_self")
    else:
        tag = get_condition_tag(args.include_self)
        if args.include_self:
            entity_keys = ENTITY_NAMES
        else:
            entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]
        run_condition(model, tokenizer, DEVICE,
                      entity_keys, capacity_keys, capacity_descs, tag)


if __name__ == "__main__":
    main()
