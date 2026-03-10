#!/usr/bin/env python3
"""
Experiment 3: Simple (Syntactically Controlled) Concept Elicitation

Extracts activations for the simple prompt set: 153 concepts × 1 standalone
template each ("Think about what it is like to have [X]."). Groups by category
(10 categories: 5 mental, 5 control) and saves per-category mean vectors.

Each category becomes a "dimension" for downstream alignment analysis.
Output format matches the existing standalone pipeline for compatibility.

Output:
    results/{model}/concept_activations/standalone/{dim_name}/
        concept_activations_simple.npz   (activations, n_prompts)
        concept_prompts_simple.json      (prompt metadata)
        mean_vectors_per_layer_simple.npz (mean_concept)
        split_half_stability_simple.json

Usage:
    python 1_elicit_simple_vectors.py

SLURM:
    sbatch slurm/simple_elicit.sh

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.dataset import llama_v2_prompt
from config import config, get_device, set_variant, variant_filename

# Add concepts directory to path for import
sys.path.insert(0, os.path.join(str(config.ROOT_DIR), "concepts"))
from simple_prompts import (
    ALL_CONCEPTS, STANDALONE_PROMPTS, CATEGORY_ORDER, CATEGORY_INFO,
    MENTAL_CONCEPTS, CONTROL_CONCEPTS,
)


# ========================== CONFIG ========================== #

MODEL_NAME = config.MODEL_NAME
DEVICE = get_device()
torch.manual_seed(config.ANALYSIS.seed)


# ========================== EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text):
    """Forward pass; extract residual-stream activations at last token, all layers."""
    messages = [{"role": "user", "content": prompt_text}]
    formatted = llama_v2_prompt(messages)

    with torch.no_grad():
        encoding = tokenizer(
            formatted,
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

    last_acts = []
    for layer_num in range(len(output["hidden_states"])):
        act = (output["hidden_states"][layer_num][:, -1]
               .detach().cpu().to(torch.float32))
        last_acts.append(act)

    return torch.cat(last_acts, dim=0)  # (n_layers, hidden_dim)


def compute_split_half_standalone(activations_array, n_splits=100):
    """Split-half reliability of the standalone mean vector."""
    n_prompts = activations_array.shape[0]
    n_layers = activations_array.shape[1]

    results = {}
    for layer in range(n_layers):
        cos_sims = []
        for _ in range(n_splits):
            perm = np.random.permutation(n_prompts)
            half = n_prompts // 2
            v1 = activations_array[perm[:half], layer].mean(axis=0)
            v2 = activations_array[perm[half:], layer].mean(axis=0)
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-10 and n2 > 1e-10:
                cos = float(np.dot(v1, v2) / (n1 * n2))
            else:
                cos = 0.0
            cos_sims.append(cos)
        results[layer] = {
            "mean": float(np.mean(cos_sims)),
            "std": float(np.std(cos_sims)),
        }
    return results


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Extract simple (syntactically controlled) concept activations"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompt counts without running model")
    args = parser.parse_args()

    print("=" * 70)
    print("SIMPLE PROMPT PIPELINE: Concept Activation Extraction")
    print("=" * 70)

    # Show category summary
    print(f"\n{len(CATEGORY_ORDER)} categories, {len(ALL_CONCEPTS)} total concepts:")
    for dim_id, cat_name in CATEGORY_ORDER:
        info = next(ci for ci in CATEGORY_INFO if ci["name"] == cat_name)
        print(f"  {dim_id:>2d}. {cat_name:<30s}  ({info['type']}, {info['count']} concepts)")

    if args.dry_run:
        print(f"\nTotal standalone prompts: {len(STANDALONE_PROMPTS)}")
        print("[DRY RUN] Exiting.")
        return

    # Load model
    print(f"\nLoading LLaMA-2-Chat-13B from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print("Model loaded.")

    # Extract all activations
    print(f"\nExtracting activations for {len(STANDALONE_PROMPTS)} standalone prompts...")
    all_activations = []
    for prompt_text in tqdm(STANDALONE_PROMPTS, desc="Extracting"):
        acts = extract_activations(model, tokenizer, prompt_text)
        all_activations.append(acts.numpy())

    all_activations = np.stack(all_activations)  # (153, n_layers, hidden_dim)
    print(f"Activations shape: {all_activations.shape}")

    # Set variant for output paths
    set_variant("_simple")
    out_root = str(config.RESULTS.concept_activations_standalone)

    # Save per category
    np.random.seed(config.ANALYSIS.seed)
    for dim_id, cat_name in CATEGORY_ORDER:
        info = next(ci for ci in CATEGORY_INFO if ci["name"] == cat_name)
        start, end = info["start"], info["end"]

        dim_name = f"{dim_id}_{cat_name}"
        dim_dir = os.path.join(out_root, dim_name)
        os.makedirs(dim_dir, exist_ok=True)

        # Slice activations for this category
        cat_acts = all_activations[start:end]  # (n_cat, n_layers, hidden_dim)
        mean_concept = cat_acts.mean(axis=0)    # (n_layers, hidden_dim)

        # Build prompt metadata
        cat_concepts = ALL_CONCEPTS[start:end]
        prompt_metadata = []
        for i, item in enumerate(cat_concepts):
            prompt_metadata.append({
                "prompt": STANDALONE_PROMPTS[start + i],
                "concept": item["concept"],
                "category": cat_name,
                "concept_type": item["type"],
                "label": -1,
            })

        # Split-half stability
        split_half = compute_split_half_standalone(cat_acts)

        # Save
        np.savez_compressed(
            os.path.join(dim_dir, variant_filename("concept_activations", ".npz")),
            activations=cat_acts,
            n_prompts=cat_acts.shape[0],
        )
        np.savez_compressed(
            os.path.join(dim_dir, variant_filename("mean_vectors_per_layer", ".npz")),
            mean_concept=mean_concept,
        )
        with open(os.path.join(dim_dir, variant_filename("concept_prompts", ".json")), "w") as f:
            json.dump(prompt_metadata, f, indent=2)
        with open(os.path.join(dim_dir, variant_filename("split_half_stability", ".json")), "w") as f:
            json.dump({str(k): v for k, v in split_half.items()}, f, indent=2)

        # Print summary
        norms = np.linalg.norm(mean_concept, axis=1)
        sh_mid = split_half.get(30, split_half.get(20, {}))
        sh_str = f"sh={sh_mid.get('mean', 0):.4f}" if sh_mid else ""
        print(f"  {dim_name:<35s}: {cat_acts.shape[0]:>2d} prompts, "
              f"||mean|| L30={norms[30]:.1f}  {sh_str}")

    print(f"\nSaved to: {out_root}")
    print("Simple concept activation extraction complete.")


if __name__ == "__main__":
    main()
