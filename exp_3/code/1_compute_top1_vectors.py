#!/usr/bin/env python3
"""
Experiment 3: Top-1 Prompt Pipeline — Compute Concept Vectors

Selects the single most representative standalone prompt per concept dimension,
then computes contrast directions: activation(top_X) - mean(activation(top_Y) for Y≠X).

This isolates what is unique about each concept's representation using a single
exemplar per concept, avoiding lexical averaging artifacts.

Steps:
    1. Load existing standalone activations for each available concept dim
       (excludes sysprompt variants 20-23)
    2. Select top prompt per dim: highest mean cosine to centroid (layers 20-40)
    3. Compute per-concept contrast: top_X - mean(top_Y for Y≠X)
    4. Save in concept_activations/ with _top_align filename suffix

Output:
    results/{model}/concept_activations/
        top_prompt_selections_top_align.json
        contrasts/{dim_name}/
            concept_vector_per_layer_top_align.npz   (concept_direction, norms)
            concept_prompts_top_align.json            (single prompt metadata)
        standalone/{dim_name}/
            mean_vectors_per_layer_top_align.npz      (mean_concept = single prompt activation)

No GPU needed — reads existing .npz files and computes means/cosines.

Usage:
    python 1_compute_top1_vectors.py

Env: llama2_env (CPU only)
"""

import os
import sys
import json
import argparse
import numpy as np

from config import config, set_variant, variant_filename

# ========================== CONFIG ========================== #

# Dims to exclude (sysprompt variants — not standalone concepts)
SYSPROMPT_DIMS = {20, 21, 22, 23}

# Layer range for selecting top prompt (mid-to-late layers)
SELECT_LAYER_START = 20
SELECT_LAYER_END = 41  # exclusive

# Source: existing standalone activations (variant="", i.e., original)
SOURCE_ACT_DIR = str(config.RESULTS.concept_activations_standalone)


# ========================== HELPERS ========================== #

def cosine_sim(v1, v2):
    """Cosine similarity between two 1D vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def discover_standalone_dims(act_dir):
    """Discover available standalone dimensions, excluding sysprompt variants."""
    dims = {}
    if not os.path.isdir(act_dir):
        print(f"[ERROR] Standalone activations not found: {act_dir}")
        sys.exit(1)
    for name in sorted(os.listdir(act_dir)):
        full = os.path.join(act_dir, name)
        if not os.path.isdir(full):
            continue
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        if dim_id in SYSPROMPT_DIMS:
            continue
        # Check that activations exist
        acts_path = os.path.join(full, "concept_activations.npz")
        if os.path.isfile(acts_path):
            dims[dim_id] = name
    return dims


def select_top_prompt(activations, layer_start=SELECT_LAYER_START,
                      layer_end=SELECT_LAYER_END):
    """Select the prompt whose activation has highest mean cosine to centroid.

    Args:
        activations: (n_prompts, n_layers, hidden_dim) array
        layer_start: first layer to use for selection
        layer_end: last layer (exclusive)

    Returns:
        top_idx: int, index of top prompt
        top_score: float, mean cosine score
        all_scores: list of floats, scores for all prompts
    """
    n_prompts = activations.shape[0]

    # Compute centroid across prompts for each layer
    centroid = activations.mean(axis=0)  # (n_layers, hidden_dim)

    # Score each prompt: mean cosine to centroid over selected layers
    scores = np.zeros(n_prompts)
    for p in range(n_prompts):
        layer_cosines = []
        for l in range(layer_start, min(layer_end, activations.shape[1])):
            cos = cosine_sim(activations[p, l], centroid[l])
            layer_cosines.append(cos)
        scores[p] = np.mean(layer_cosines)

    top_idx = int(np.argmax(scores))
    return top_idx, float(scores[top_idx]), scores.tolist()


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Compute top-1 prompt concept vectors for _1 pipeline"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print selections without saving")
    args = parser.parse_args()

    print("=" * 70)
    print("TOP-1 PROMPT PIPELINE: Concept Vector Computation")
    print("=" * 70)

    # Discover dims from original (non-variant) standalone activations
    dims = discover_standalone_dims(SOURCE_ACT_DIR)
    dim_ids = sorted(dims.keys())
    print(f"\nFound {len(dim_ids)} standalone dims (excluding sysprompt): {dim_ids}")

    # Step 1: Select top prompt per dim
    print("\n--- Step 1: Selecting top prompts ---")
    selections = {}
    top_activations = {}

    for dim_id in dim_ids:
        dim_name = dims[dim_id]
        acts_path = os.path.join(SOURCE_ACT_DIR, dim_name, "concept_activations.npz")
        prompts_path = os.path.join(SOURCE_ACT_DIR, dim_name, "concept_prompts.json")

        data = np.load(acts_path)
        activations = data["activations"]  # (n_prompts, n_layers, hidden_dim)

        # Load prompt texts if available
        prompt_texts = None
        if os.path.isfile(prompts_path):
            with open(prompts_path) as f:
                prompt_meta = json.load(f)
            prompt_texts = [p.get("prompt", f"prompt_{i}") for i, p in enumerate(prompt_meta)]

        top_idx, top_score, all_scores = select_top_prompt(activations)

        top_text = prompt_texts[top_idx] if prompt_texts else f"prompt_{top_idx}"
        top_act = activations[top_idx]  # (n_layers, hidden_dim)
        top_activations[dim_id] = top_act

        selections[dim_name] = {
            "dim_id": dim_id,
            "prompt_idx": top_idx,
            "prompt_text": top_text,
            "cosine_score": top_score,
            "n_prompts": activations.shape[0],
            "score_range": [float(np.min(all_scores)), float(np.max(all_scores))],
        }

        print(f"  dim {dim_id:>2d} ({dim_name:<25s}): prompt {top_idx:>2d}/{activations.shape[0]}  "
              f"cos={top_score:.4f}  text={top_text[:60]}...")

    if args.dry_run:
        print("\n[DRY RUN] Would save selections. Exiting.")
        return

    # Step 2: Compute contrast directions
    print(f"\n--- Step 2: Computing contrast directions ({len(dim_ids)} dims) ---")

    # Stack all top activations
    all_top = np.stack([top_activations[d] for d in dim_ids])  # (n_dims, n_layers, hidden_dim)
    grand_mean = all_top.mean(axis=0)  # (n_layers, hidden_dim)
    n_dims = len(dim_ids)

    # Set variant to _1 for output paths
    set_variant("_1")
    out_root = str(config.RESULTS.concept_activations)
    contrasts_dir = str(config.RESULTS.concept_activations_contrasts)
    standalone_dir = str(config.RESULTS.concept_activations_standalone)

    for i, dim_id in enumerate(dim_ids):
        dim_name = dims[dim_id]
        top_act = top_activations[dim_id]  # (n_layers, hidden_dim)

        # Contrast: top_X - mean(top_Y for Y != X)
        # other_mean = (grand_mean * n_dims - top_act) / (n_dims - 1)
        other_mean = (grand_mean * n_dims - top_act) / (n_dims - 1)
        contrast_direction = top_act - other_mean  # (n_layers, hidden_dim)
        norms = np.linalg.norm(contrast_direction, axis=1)

        # Save contrasts
        dim_contrast_dir = os.path.join(contrasts_dir, dim_name)
        os.makedirs(dim_contrast_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dim_contrast_dir, variant_filename("concept_vector_per_layer", ".npz")),
            concept_direction=contrast_direction,
            norms=norms,
        )
        with open(os.path.join(dim_contrast_dir, variant_filename("concept_prompts", ".json")), "w") as f:
            json.dump([{
                "prompt": selections[dim_name]["prompt_text"],
                "label": -1,
                "category": "top1_selected",
                "source_idx": selections[dim_name]["prompt_idx"],
            }], f, indent=2)

        # Save standalone (single prompt activation as mean_concept)
        dim_standalone_dir = os.path.join(standalone_dir, dim_name)
        os.makedirs(dim_standalone_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(dim_standalone_dir, variant_filename("mean_vectors_per_layer", ".npz")),
            mean_concept=top_act,
        )

        print(f"  {dim_name:<25s}: ||contrast|| layers 20-40 mean={np.mean(norms[20:41]):.2f}")

    # Save selection metadata
    meta_path = os.path.join(out_root, variant_filename("top_prompt_selections", ".json"))
    with open(meta_path, "w") as f:
        json.dump(selections, f, indent=2)
    print(f"\n[SAVED] Top prompt selections: {meta_path}")
    print(f"[SAVED] Contrast vectors: {contrasts_dir}")
    print(f"[SAVED] Standalone vectors: {standalone_dir}")

    # Verification
    print(f"\n--- Verification ---")
    sample_dim = dims[dim_ids[0]]
    sample_path = os.path.join(contrasts_dir, sample_dim, "concept_vector_per_layer.npz")
    sample_data = np.load(sample_path)
    print(f"  Sample ({sample_dim}): concept_direction shape = {sample_data['concept_direction'].shape}")
    print(f"  Expected: ({config.N_LAYERS}, {config.INPUT_DIM})")

    print(f"\nTop-1 pipeline vector computation complete.")


if __name__ == "__main__":
    main()
