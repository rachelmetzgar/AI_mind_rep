#!/usr/bin/env python3
"""
Experiment 3, Phase 1: Concept Elicitation and Vector Extraction

Extracts internal representations for a SINGLE concept dimension,
specified by --dim_id (1-13). Designed to run in parallel via SLURM
array jobs.

For each prompt in the specified dimension:
    - Runs a forward pass through LLaMA-2-Chat-13B
    - Extracts residual-stream activations at the last token across all layers
    - Saves activations, labels, mean vectors, and contrastive direction

Output (per dimension):
    data/concept_activations/{dim_name}/
        concept_activations.npz
        concept_prompts.json
        mean_vectors_per_layer.npz
        concept_vector_per_layer.npz
        split_half_stability.json

Usage:
    python 1_elicit_concept_vectors.py --dim_id 1
    python 1_elicit_concept_vectors.py --dim_id 7

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
sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import llama_v2_prompt


# ========================== CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

OUTPUT_ROOT = "data/concept_activations"
INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)


# ========================== DIMENSION REGISTRY ========================== #
# Auto-discovered from concepts/ directory.
# Files must be named: {dim_id}_{name}.py (e.g., 0_baseline.py, 7_social.py)
# dim_id is the integer prefix, dim_name (for output folders) = full filename stem.

def build_dimension_registry(concepts_dir=None):
    """
    Scan concepts/ for files matching {N}_{name}.py pattern.
    Returns dict: {dim_id: (module_stem, dim_name)} where dim_name = module_stem.
    """
    if concepts_dir is None:
        concepts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "concepts")

    registry = {}
    if not os.path.isdir(concepts_dir):
        print(f"[WARNING] Concepts directory not found: {concepts_dir}")
        return registry

    for fname in sorted(os.listdir(concepts_dir)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        stem = fname[:-3]  # strip .py
        parts = stem.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        registry[dim_id] = (stem, stem)

    return registry


DIMENSION_REGISTRY = build_dimension_registry()


def load_dimension_prompts(dim_id, concepts_dir="concepts"):
    """
    Dynamically import the prompt module for a given dimension.
    Returns (human_prompts, ai_prompts, category_info, dim_name).
    """
    if dim_id not in DIMENSION_REGISTRY:
        raise ValueError(f"Unknown dim_id={dim_id}. Valid: {list(DIMENSION_REGISTRY.keys())}")

    module_file, dim_name = DIMENSION_REGISTRY[dim_id]
    module_path = os.path.join(concepts_dir, f"{module_file}.py")

    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Concept file not found: {module_path}")

    # Dynamic import
    spec = importlib.util.spec_from_file_location(f"concept_dim{dim_id}", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the prompt variables by scanning module attributes
    human_prompts = None
    ai_prompts = None
    category_info = None

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
        f"Dim {dim_id}: mismatched prompt counts: "
        f"{len(human_prompts)} human vs {len(ai_prompts)} AI"
    )

    print(f"Loaded dimension {dim_id} ({dim_name}): "
          f"{len(human_prompts)} human + {len(ai_prompts)} AI prompts")

    return human_prompts, ai_prompts, category_info, dim_name


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


def extract_all(model, tokenizer, human_prompts, ai_prompts, category_info):
    """Extract activations for all prompts in a dimension."""
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

        # Determine category
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

    return activations, labels, prompt_metadata


# ========================== VECTOR COMPUTATION ========================== #

def compute_concept_vectors(activations, labels):
    """Compute mean vectors and contrastive direction per layer."""
    acts_array = torch.stack(activations)
    labels_array = torch.tensor(labels)
    mean_human = acts_array[labels_array == 1].mean(dim=0)
    mean_ai = acts_array[labels_array == 0].mean(dim=0)
    concept_direction = mean_human - mean_ai
    return mean_human, mean_ai, concept_direction


def compute_split_half_stability(activations, labels, n_splits=20):
    """Split-half reliability of the concept direction."""
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

    print("\nSplit-half stability (cosine similarity):")
    for layer in range(0, n_layers, 5):
        r = results[layer]
        print(f"  Layer {layer:2d}: cos = {r['mean']:.4f} (±{r['std']:.4f})")
    return results


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Extract concept activations for one dimension"
    )
    parser.add_argument("--dim_id", type=int, required=True,
                        help="Dimension ID (1-13)")
    parser.add_argument("--concepts_dir", type=str, default="concepts",
                        help="Directory containing concept prompt files")
    args = parser.parse_args()

    # Load prompts
    human_prompts, ai_prompts, category_info, dim_name = load_dimension_prompts(
        args.dim_id, args.concepts_dir
    )

    out_dir = os.path.join(OUTPUT_ROOT, dim_name)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print("Loading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print("Model loaded.")

    # Extract
    activations, labels, prompt_metadata = extract_all(
        model, tokenizer, human_prompts, ai_prompts, category_info
    )

    # Compute vectors
    mean_human, mean_ai, concept_direction = compute_concept_vectors(
        activations, labels
    )

    n_layers = concept_direction.shape[0]
    print(f"\nDimension: {dim_name} (id={args.dim_id})")
    print(f"  Layers: {n_layers}, hidden_dim: {concept_direction.shape[1]}")
    print(f"  Human: {sum(1 for l in labels if l == 1)}, "
          f"AI: {sum(1 for l in labels if l == 0)}")

    # Stability
    split_half = compute_split_half_stability(activations, labels)

    # Norms
    norms = torch.norm(concept_direction, dim=1)
    print("\nConcept vector norms by layer:")
    for layer in range(0, n_layers, 5):
        print(f"  Layer {layer:2d}: ||v|| = {norms[layer]:.4f}")

    # === Save ===
    acts_array = torch.stack(activations).numpy()
    labels_array = np.array(labels)

    np.savez_compressed(
        os.path.join(out_dir, "concept_activations.npz"),
        activations=acts_array,
        labels=labels_array,
        n_human=sum(1 for l in labels if l == 1),
        n_ai=sum(1 for l in labels if l == 0),
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

    print(f"\nSaved all outputs to {out_dir}/")
    print(f"✅ Phase 1 complete for dimension {args.dim_id} ({dim_name}).")


if __name__ == "__main__":
    main()