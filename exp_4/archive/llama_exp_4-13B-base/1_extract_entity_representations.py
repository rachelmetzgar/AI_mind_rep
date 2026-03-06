#!/usr/bin/env python3
"""
Experiment 4, Phase 1: Entity Representation Extraction (Base Model)

Extracts internal representations for each of the 13 Gray et al. (2007)
entities from LLaMA-2-13B (base, not chat). One prompt per entity
("Think about {X}"), last-token residual-stream activations across all layers.

Key differences from the chat model version:
  - No chat template: raw text prompts tokenized directly
  - Three human RDM variants: combined (2D Euclidean), experience-only, agency-only
  - Three RSA runs per layer (one per human RDM variant)

Output:
    data/entity_activations/{tag}/
        all_entity_activations.npz   # (n_entities, n_layers, hidden_dim)
        entity_prompts.json          # prompt metadata
        rdm_cosine_per_layer.npz     # model RDM + all 3 human RDMs

    data/entity_activations/{tag}/
        rsa_results.json             # all 3 RSA variants per layer

    results/{tag}/
        results_summary.md           # documented methodology + results

Usage:
    python 1_extract_entity_representations.py                  # 12 entities (no self)
    python 1_extract_entity_representations.py --include_self   # 13 entities

SLURM:
    sbatch slurm/1_extract_entities.sh

Env: llama2_env
Rachel C. Metzgar · Mar 2026
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

# ── Local imports ──
sys.path.insert(0, os.path.dirname(__file__))
from entities.gray_entities import (
    GRAY_ET_AL_SCORES,
    ENTITY_PROMPTS,
    ENTITY_NAMES,
)


# ========================== CONFIG ========================== #

MODEL_ID = "meta-llama/Llama-2-13b-hf"
INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)


# ========================== EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text):
    """
    Forward pass on a single prompt; extract residual-stream activations
    at the last token position across all layers.

    No chat template — raw text is tokenized directly.

    Returns: tensor of shape (n_layers+1, hidden_dim)
        Layer 0 = embedding output, layers 1..N = transformer block outputs.
    """
    with torch.no_grad():
        encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=2048,
            return_attention_mask=True,
            return_tensors="pt",
        )
        output = model(
            input_ids=encoding["input_ids"].to(DEVICE),
            attention_mask=encoding["attention_mask"].to(DEVICE),
            output_hidden_states=True,
            return_dict=True,
        )

    # Collect last-token activation from each layer
    last_acts = []
    for layer_hidden in output["hidden_states"]:
        act = layer_hidden[:, -1].detach().cpu().to(torch.float32)
        last_acts.append(act)

    return torch.cat(last_acts, dim=0)  # (n_layers+1, hidden_dim)


# ========================== RDM ========================== #

def compute_rdm_cosine(entity_activations):
    """
    Compute representational dissimilarity matrix (cosine distance)
    at each layer.

    Args:
        entity_activations: (n_entities, n_layers, hidden_dim)

    Returns:
        rdm: (n_layers, n_entities, n_entities) cosine distances
    """
    n_entities, n_layers, hidden_dim = entity_activations.shape
    rdm = np.zeros((n_layers, n_entities, n_entities))

    for layer in range(n_layers):
        vecs = entity_activations[:, layer, :]  # (n_entities, hidden_dim)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_normed = vecs / norms
        cos_sim = vecs_normed @ vecs_normed.T
        rdm[layer] = 1.0 - cos_sim

    return rdm


def compute_human_rdm_combined(entity_keys):
    """
    Compute human RDM from Gray et al. Experience/Agency scores
    using Euclidean distance in 2D space.
    """
    n = len(entity_keys)
    coords = np.array([GRAY_ET_AL_SCORES[k] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    return rdm


def compute_human_rdm_experience(entity_keys):
    """
    Compute human RDM using absolute difference in Experience scores only.
    """
    n = len(entity_keys)
    exp_scores = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(exp_scores[i] - exp_scores[j])
    return rdm


def compute_human_rdm_agency(entity_keys):
    """
    Compute human RDM using absolute difference in Agency scores only.
    """
    n = len(entity_keys)
    agency_scores = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(agency_scores[i] - agency_scores[j])
    return rdm


# ========================== RSA ========================== #

def compute_rsa_all_layers(model_rdm, human_rdm, n_entities):
    """
    Spearman correlation between upper triangles of model and human RDMs
    at every layer.

    Returns: list of dicts with {layer, rho, p_value, n_pairs}.
    """
    n_layers = model_rdm.shape[0]
    triu_idx = np.triu_indices(n_entities, k=1)
    human_upper = human_rdm[triu_idx]
    n_pairs = len(human_upper)

    results = []
    for layer in range(n_layers):
        model_upper = model_rdm[layer][triu_idx]
        if np.std(model_upper) < 1e-12:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(model_upper, human_upper)
        results.append({
            "layer": layer,
            "rho": float(rho),
            "p_value": float(p),
            "n_pairs": n_pairs,
        })

    return results


# ========================== RESULTS SUMMARY ========================== #

def write_results_summary(results_dir, rsa_all, entity_keys, tag,
                          include_self):
    """Write a documented markdown summary of the analysis and results."""
    n_entities = len(entity_keys)
    n_pairs = n_entities * (n_entities - 1) // 2

    summary_path = os.path.join(results_dir, "results_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Experiment 4, Phase 1: Entity Representation Extraction "
                "(Base Model)\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Tag:** `{tag}`\n")
        f.write(f"**Include self:** {include_self}\n")
        f.write(f"**Model:** LLaMA-2-13B (base)\n\n")

        f.write("---\n\n")
        f.write("## Entities\n\n")
        f.write("| Entity | Prompt | Experience | Agency |\n")
        f.write("|--------|--------|------------|--------|\n")
        for key in entity_keys:
            exp, agency = GRAY_ET_AL_SCORES[key]
            prompt = ENTITY_PROMPTS[key]
            f.write(f"| {key} | {prompt} | {exp:.2f} | {agency:.2f} |\n")
        f.write("\n")

        for variant_name in ["combined", "experience", "agency"]:
            rsa_results = rsa_all[variant_name]
            valid = [r for r in rsa_results if not np.isnan(r["rho"])]
            if valid:
                peak = max(valid, key=lambda r: r["rho"])
            else:
                peak = {"layer": -1, "rho": float("nan"),
                        "p_value": float("nan")}

            f.write(f"## RSA by Layer — {variant_name.title()}\n\n")
            f.write(f"**Peak:** Layer {peak['layer']}, "
                    f"rho = {peak['rho']:+.4f}, p = {peak['p_value']:.4f}\n\n")
            f.write("| Layer | Spearman rho | p-value |\n")
            f.write("|------:|-------------:|--------:|\n")
            for r in rsa_results:
                if np.isnan(r["rho"]):
                    f.write(f"| {r['layer']:5d} |          nan |"
                            f"     nan |\n")
                else:
                    f.write(f"| {r['layer']:5d} | {r['rho']:+12.4f} |"
                            f" {r['p_value']:7.4f} |\n")
            f.write("\n")

    print(f"  Results summary: {summary_path}")
    return summary_path


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 1: Extract entity representations (base model)"
    )
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
    )
    args = parser.parse_args()

    # Select entities and tag
    if args.include_self:
        entity_keys = ENTITY_NAMES
        tag = "with_self"
    else:
        entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]
        tag = "without_self"

    n_entities = len(entity_keys)
    data_dir = os.path.join("data", "entity_activations", tag)
    results_dir = os.path.join("results", tag)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Entities ({n_entities}): {entity_keys}")
    print(f"Tag: {tag}")
    print(f"Data dir: {data_dir}")
    print(f"Results dir: {results_dir}")

    # Load model (base, from HF cache)
    print(f"\nLoading LLaMA-2-13B (base) from HF cache...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, local_files_only=True)
    model.half().to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}.")

    # Extract activations
    all_activations = []
    prompt_metadata = []

    for entity_key in entity_keys:
        prompt = ENTITY_PROMPTS[entity_key]
        exp_score, agency_score = GRAY_ET_AL_SCORES[entity_key]

        print(f"  {entity_key}: \"{prompt}\"")
        acts = extract_activations(model, tokenizer, prompt)
        all_activations.append(acts)

        prompt_metadata.append({
            "entity": entity_key,
            "prompt": prompt,
            "experience": exp_score,
            "agency": agency_score,
        })

    # Stack: (n_entities, n_layers, hidden_dim)
    acts_array = torch.stack(all_activations).numpy()
    n_layers = acts_array.shape[1]
    hidden_dim = acts_array.shape[2]

    print(f"\nActivations shape: {acts_array.shape}")
    print(f"  {n_entities} entities x {n_layers} layers x {hidden_dim} dims")

    # Compute RDMs
    print("\nComputing model RDM (cosine distance)...")
    model_rdm = compute_rdm_cosine(acts_array)

    print("Computing human RDMs (combined, experience, agency)...")
    human_rdm_combined = compute_human_rdm_combined(entity_keys)
    human_rdm_experience = compute_human_rdm_experience(entity_keys)
    human_rdm_agency = compute_human_rdm_agency(entity_keys)

    # RSA at every layer for all 3 variants
    print("\nComputing RSA (Spearman) at all layers...")
    rsa_combined = compute_rsa_all_layers(model_rdm, human_rdm_combined,
                                          n_entities)
    rsa_experience = compute_rsa_all_layers(model_rdm, human_rdm_experience,
                                            n_entities)
    rsa_agency = compute_rsa_all_layers(model_rdm, human_rdm_agency,
                                        n_entities)

    rsa_all = {
        "combined": rsa_combined,
        "experience": rsa_experience,
        "agency": rsa_agency,
    }

    # Print RSA summaries
    for variant_name, rsa_results in rsa_all.items():
        valid = [r for r in rsa_results if not np.isnan(r["rho"])]
        if valid:
            peak = max(valid, key=lambda r: r["rho"])
            print(f"\n  {variant_name}: Peak Layer {peak['layer']}, "
                  f"rho = {peak['rho']:+.4f}, p = {peak['p_value']:.4f}")
        else:
            print(f"\n  {variant_name}: all NaN")

    # ── Save data ──
    np.savez_compressed(
        os.path.join(data_dir, "all_entity_activations.npz"),
        activations=acts_array,
        entity_keys=np.array(entity_keys),
    )
    np.savez_compressed(
        os.path.join(data_dir, "rdm_cosine_per_layer.npz"),
        model_rdm=model_rdm,
        human_rdm_combined=human_rdm_combined,
        human_rdm_experience=human_rdm_experience,
        human_rdm_agency=human_rdm_agency,
        entity_keys=np.array(entity_keys),
    )
    with open(os.path.join(data_dir, "entity_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    # ── Save RSA results ──
    with open(os.path.join(data_dir, "rsa_results.json"), "w") as f:
        json.dump(rsa_all, f, indent=2)

    # Also save to results dir for backward compat
    with open(os.path.join(results_dir, "rsa_results.json"), "w") as f:
        json.dump(rsa_all, f, indent=2)

    write_results_summary(results_dir, rsa_all, entity_keys, tag,
                          args.include_self)

    print(f"\nSaved data to {data_dir}/")
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    main()
