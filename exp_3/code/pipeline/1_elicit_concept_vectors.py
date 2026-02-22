#!/usr/bin/env python3
"""
Experiment 3, Phase 1: Concept Elicitation and Vector Extraction

Extracts internal representations for a SINGLE concept dimension,
specified by --dim_id. Designed to run in parallel via SLURM array jobs.

Supports two modes (--mode):
    contrasts : Extract human/AI paired prompts. Computes mean-difference
                concept vector (human - AI) per layer. Used for raw_alignment
                and residual_alignment analyses.
    standalone: Extract concept-only prompts (no entity framing). Computes
                mean activation vector per layer. Used for standalone analysis.

For each prompt:
    - Runs a forward pass through LLaMA-2-Chat-13B
    - Extracts residual-stream activations at the last token across all layers
    - Saves activations, labels, mean vectors, and (for contrasts) contrastive direction

Output (per dimension, per mode):
    data/concept_activations/{mode}/{dim_name}/
        concept_activations.npz
        concept_prompts.json
        mean_vectors_per_layer.npz          (contrasts: mean_human + mean_ai)
                                            (standalone: mean_concept only)
        concept_vector_per_layer.npz        (contrasts only: human - AI direction)
        split_half_stability.json

Usage:
    # Contrast mode (human vs AI prompts)
    python 1_elicit_concept_vectors.py --mode contrasts --dim_id 1
    python 1_elicit_concept_vectors.py --mode contrasts --dim_id 0   # entity baseline

    # Standalone mode (concept-only prompts)
    python 1_elicit_concept_vectors.py --mode standalone --dim_id 1
    python 1_elicit_concept_vectors.py --mode standalone --dim_id 15  # shapes control

SLURM:
    sbatch slurm/elicit_concept_vectors.sh contrasts
    sbatch slurm/elicit_concept_vectors.sh standalone

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
import importlib
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local imports ---
# Add parent directory to path (pipeline/ -> labels/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.dataset import llama_v2_prompt
from config import config


# ========================== CONFIG ========================== #

MODEL_NAME = config.MODEL_NAME
OUTPUT_ROOT = str(config.PATHS.concept_activations)  # data/concept_activations (intermediate data)
INPUT_DIM = config.INPUT_DIM
DEVICE = config.get_device()
torch.manual_seed(config.ANALYSIS.seed)


# ========================== DIMENSION REGISTRY ========================== #

def build_dimension_registry(concepts_dir):
    """
    Scan a concepts directory for files matching {N}_{name}.py or {N}_{name}.txt.
    Returns dict: {dim_id: (module_stem, dim_name)}.
    """
    registry = {}
    if not os.path.isdir(concepts_dir):
        print(f"[WARNING] Concepts directory not found: {concepts_dir}")
        return registry

    for fname in sorted(os.listdir(concepts_dir)):
        if not (fname.endswith(".py") or fname.endswith(".txt")) or fname.startswith("__"):
            continue
        stem = fname.rsplit(".", 1)[0]  # strip extension
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        registry[dim_id] = (stem, stem)

    return registry


def load_contrast_prompts(dim_id, concepts_dir):
    """
    Load human/AI paired prompts from a contrast file.
    Returns (human_prompts, ai_prompts, category_info, dim_name).
    """
    registry = build_dimension_registry(concepts_dir)
    if dim_id not in registry:
        raise ValueError(f"Unknown dim_id={dim_id} in {concepts_dir}. "
                         f"Valid: {list(registry.keys())}")

    module_file, dim_name = registry[dim_id]
    module_path = os.path.join(concepts_dir, f"{module_file}.py")

    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Concept file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"contrast_dim{dim_id}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    human_prompts = ai_prompts = category_info = None
    for attr_name in dir(mod):
        val = getattr(mod, attr_name)
        if attr_name.startswith("HUMAN_PROMPTS") and isinstance(val, list):
            human_prompts = val
        elif attr_name.startswith("AI_PROMPTS") and isinstance(val, list):
            ai_prompts = val
        elif attr_name.startswith("CATEGORY_INFO") and isinstance(val, list):
            category_info = val

    if human_prompts is None or ai_prompts is None:
        raise AttributeError(
            f"Could not find HUMAN_PROMPTS_* and AI_PROMPTS_* in {module_path}"
        )
    assert len(human_prompts) == len(ai_prompts), (
        f"Dim {dim_id}: {len(human_prompts)} human vs {len(ai_prompts)} AI"
    )

    print(f"[contrasts] Loaded dim {dim_id} ({dim_name}): "
          f"{len(human_prompts)} human + {len(ai_prompts)} AI prompts")
    return human_prompts, ai_prompts, category_info, dim_name


def load_standalone_prompts(dim_id, concepts_dir):
    """
    Load standalone concept prompts (no entity framing).
    Returns (prompts, category_info, dim_name).
    """
    registry = build_dimension_registry(concepts_dir)
    if dim_id not in registry:
        raise ValueError(f"Unknown dim_id={dim_id} in {concepts_dir}. "
                         f"Valid: {list(registry.keys())}")

    module_file, dim_name = registry[dim_id]
    module_path = os.path.join(concepts_dir, f"{module_file}.py")

    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Concept file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(f"standalone_dim{dim_id}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    prompts = category_info = None
    for attr_name in dir(mod):
        val = getattr(mod, attr_name)
        if attr_name.startswith("STANDALONE_PROMPTS") and isinstance(val, list):
            prompts = val
        elif attr_name.startswith("CATEGORY_INFO") and isinstance(val, list):
            category_info = val

    if prompts is None:
        raise AttributeError(
            f"Could not find STANDALONE_PROMPTS_* in {module_path}"
        )

    print(f"[standalone] Loaded dim {dim_id} ({dim_name}): "
          f"{len(prompts)} prompts")
    return prompts, category_info, dim_name


# ========================== EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text):
    """
    Forward pass on a single prompt; extract residual-stream activations
    at the last token position across all layers.

    Returns: tensor of shape (n_layers+1, hidden_dim)
    """
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

    return torch.cat(last_acts, dim=0)


# ========================== CONTRAST MODE ========================== #

def run_contrast_mode(model, tokenizer, human_prompts, ai_prompts,
                      category_info, dim_name, dim_id, out_dir):
    """Extract activations and compute contrast vectors for human vs AI prompts."""
    activations = []
    labels = []
    prompt_metadata = []

    all_prompts = (
        [(p, 1, "human") for p in human_prompts]
        + [(p, 0, "ai") for p in ai_prompts]
    )

    print(f"Extracting activations for {len(all_prompts)} prompts...")
    for prompt_text, label, concept in tqdm(all_prompts):
        acts = extract_activations(model, tokenizer, prompt_text)
        activations.append(acts)
        labels.append(label)

        idx = (human_prompts.index(prompt_text) if label == 1
               else ai_prompts.index(prompt_text))
        cat = "unknown"
        if category_info:
            for ci in category_info:
                if ci["start"] <= idx < ci["end"]:
                    cat = ci["name"]
                    break

        prompt_metadata.append({
            "prompt": prompt_text,
            "concept": concept,
            "label": label,
            "category": cat,
        })

    # Compute vectors
    acts_array = torch.stack(activations)
    labels_array = torch.tensor(labels)
    mean_human = acts_array[labels_array == 1].mean(dim=0)
    mean_ai = acts_array[labels_array == 0].mean(dim=0)
    concept_direction = mean_human - mean_ai

    n_layers = concept_direction.shape[0]
    print(f"\nDimension: {dim_name} (id={dim_id})")
    print(f"  Layers: {n_layers}, hidden_dim: {concept_direction.shape[1]}")
    print(f"  Human: {(labels_array == 1).sum().item()}, "
          f"AI: {(labels_array == 0).sum().item()}")

    # Split-half stability
    split_half = compute_split_half_contrast(activations, labels)

    # Norms
    norms = torch.norm(concept_direction, dim=1)
    print("\nConcept vector norms by layer:")
    for layer in range(0, n_layers, 5):
        print(f"  Layer {layer:2d}: ||v|| = {norms[layer]:.4f}")

    # Save
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "concept_activations.npz"),
        activations=acts_array.numpy(),
        labels=labels_array.numpy(),
        n_human=int((labels_array == 1).sum()),
        n_ai=int((labels_array == 0).sum()),
    )
    with open(os.path.join(out_dir, "concept_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "mean_vectors_per_layer.npz"),
        mean_human=mean_human.numpy(),
        mean_ai=mean_ai.numpy(),
    )
    np.savez_compressed(
        os.path.join(out_dir, "concept_vector_per_layer.npz"),
        concept_direction=concept_direction.numpy(),
        norms=norms.numpy(),
    )
    with open(os.path.join(out_dir, "split_half_stability.json"), "w") as f:
        json.dump({str(k): v for k, v in split_half.items()}, f, indent=2)

    print(f"\nSaved contrast outputs to {out_dir}/")


# ========================== STANDALONE MODE ========================== #

def run_standalone_mode(model, tokenizer, prompts, category_info,
                        dim_name, dim_id, out_dir):
    """Extract activations and compute mean vector for standalone concept prompts."""
    activations = []
    prompt_metadata = []

    print(f"Extracting activations for {len(prompts)} standalone prompts...")
    for i, prompt_text in enumerate(tqdm(prompts)):
        acts = extract_activations(model, tokenizer, prompt_text)
        activations.append(acts)

        cat = "unknown"
        if category_info:
            for ci in category_info:
                if ci["start"] <= i < ci["end"]:
                    cat = ci["name"]
                    break

        prompt_metadata.append({
            "prompt": prompt_text,
            "label": -1,  # no label for standalone
            "category": cat,
        })

    # Compute mean activation vector
    acts_array = torch.stack(activations)
    mean_concept = acts_array.mean(dim=0)

    n_layers = mean_concept.shape[0]
    print(f"\nDimension: {dim_name} (id={dim_id})")
    print(f"  Layers: {n_layers}, hidden_dim: {mean_concept.shape[1]}")
    print(f"  Prompts: {len(prompts)}")

    # Split-half stability (cosine between two halves' means)
    split_half = compute_split_half_standalone(activations)

    # Save
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "concept_activations.npz"),
        activations=acts_array.numpy(),
        n_prompts=len(prompts),
    )
    with open(os.path.join(out_dir, "concept_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "mean_vectors_per_layer.npz"),
        mean_concept=mean_concept.numpy(),
    )
    with open(os.path.join(out_dir, "split_half_stability.json"), "w") as f:
        json.dump({str(k): v for k, v in split_half.items()}, f, indent=2)

    print(f"\nSaved standalone outputs to {out_dir}/")


# ========================== SPLIT-HALF STABILITY ========================== #

def compute_split_half_contrast(activations, labels, n_splits=100):
    """Split-half reliability of the contrast direction (human - AI)."""
    acts_array = torch.stack(activations)
    labels_array = torch.tensor(labels)
    human_acts = acts_array[labels_array == 1]
    ai_acts = acts_array[labels_array == 0]
    n_human = human_acts.shape[0]
    n_layers = acts_array.shape[1]

    results = {}
    for layer in range(n_layers):
        cos_sims = []
        for _ in range(n_splits):
            perm_h = torch.randperm(n_human)
            perm_a = torch.randperm(n_human)
            half = n_human // 2
            v1 = (human_acts[perm_h[:half], layer].mean(0)
                  - ai_acts[perm_a[:half], layer].mean(0))
            v2 = (human_acts[perm_h[half:], layer].mean(0)
                  - ai_acts[perm_a[half:], layer].mean(0))
            cos = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)
            ).item()
            cos_sims.append(cos)
        results[layer] = {
            "mean": float(np.mean(cos_sims)),
            "std": float(np.std(cos_sims)),
        }

    print("\nSplit-half stability (cosine, contrast mode):")
    for layer in range(0, n_layers, 5):
        r = results[layer]
        print(f"  Layer {layer:2d}: cos = {r['mean']:.4f} (±{r['std']:.4f})")
    return results


def compute_split_half_standalone(activations, n_splits=100):
    """Split-half reliability of the standalone mean vector."""
    acts_array = torch.stack(activations)
    n_prompts = acts_array.shape[0]
    n_layers = acts_array.shape[1]

    results = {}
    for layer in range(n_layers):
        cos_sims = []
        for _ in range(n_splits):
            perm = torch.randperm(n_prompts)
            half = n_prompts // 2
            v1 = acts_array[perm[:half], layer].mean(0)
            v2 = acts_array[perm[half:], layer].mean(0)
            cos = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)
            ).item()
            cos_sims.append(cos)
        results[layer] = {
            "mean": float(np.mean(cos_sims)),
            "std": float(np.std(cos_sims)),
        }

    print("\nSplit-half stability (cosine, standalone mode):")
    for layer in range(0, n_layers, 5):
        r = results[layer]
        print(f"  Layer {layer:2d}: cos = {r['mean']:.4f} (±{r['std']:.4f})")
    return results


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Extract concept activations for one dimension"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["contrasts", "standalone"],
                        help="Extraction mode: 'contrasts' (human/AI pairs) "
                             "or 'standalone' (concept-only)")
    parser.add_argument("--dim_id", type=int, required=True,
                        help="Dimension ID")
    parser.add_argument("--concepts_root", type=str, default=str(config.PATHS.concepts_root),
                        help="Root directory containing contrasts/ and standalone/")
    args = parser.parse_args()

    concepts_dir = os.path.join(args.concepts_root, args.mode)

    # Load model
    print("Loading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print("Model loaded.")

    if args.mode == "contrasts":
        human_prompts, ai_prompts, category_info, dim_name = load_contrast_prompts(
            args.dim_id, concepts_dir
        )
        out_dir = os.path.join(OUTPUT_ROOT, "contrasts", dim_name)
        run_contrast_mode(
            model, tokenizer, human_prompts, ai_prompts,
            category_info, dim_name, args.dim_id, out_dir
        )

    elif args.mode == "standalone":
        prompts, category_info, dim_name = load_standalone_prompts(
            args.dim_id, concepts_dir
        )
        out_dir = os.path.join(OUTPUT_ROOT, "standalone", dim_name)
        run_standalone_mode(
            model, tokenizer, prompts, category_info,
            dim_name, args.dim_id, out_dir
        )

    print(f"✅ Phase 1 complete: mode={args.mode}, dim_id={args.dim_id}")


if __name__ == "__main__":
    main()