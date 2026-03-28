#!/usr/bin/env python3
"""
Concept Geometry: Standalone-Character Alignment

For each standalone concept (no human/AI framing), compute cosine similarity
between the concept's mean activation and each of 30 characters. Reports
whether each concept leans closer to human or AI characters.

Procedure:
    1. Load pre-computed character activations (from activation_rsa.py)
    2. Auto-discover exp_3 standalone concepts
    3. Extract mean activations for each concept (40 prompts)
    4. Cosine similarity to each of 30 characters per layer
    5. Bias = mean(sim to human chars) - mean(sim to AI chars)

Output → results/{model}/expanded_mental_concepts/internals/standalone_alignment/data/:
    standalone_activations.npz   (n_concepts, n_layers, hidden_dim)
    similarity_matrices.npz      (n_concepts, n_layers, n_characters)
    alignment_results.json       per-concept per-layer metrics

Usage:
    python expanded_mental_concepts/internals/standalone_alignment/standalone_alignment.py --model llama2_13b_chat

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import re
import importlib.util
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir, get_device,
)
from entities.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES, N_CHARACTERS,
)
from expanded_mental_concepts.internals.rsa.activation_rsa import extract_activations


# ========================== CONFIG ========================== #

torch.manual_seed(12345)

STANDALONE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "exp_3" / "concepts" / "standalone"


# ========================== CONCEPT DISCOVERY ========================== #

def discover_standalone_concepts():
    """
    Auto-discover exp_3 standalone concept files.
    Returns list of dicts with keys: dim_id, name, prompts.
    """
    concepts = []
    pattern = re.compile(r"^(\d+)_(.+)\.py$")

    for fname in sorted(STANDALONE_DIR.iterdir()):
        m = pattern.match(fname.name)
        if not m:
            continue

        dim_id = int(m.group(1))
        dim_name = m.group(2)

        spec = importlib.util.spec_from_file_location(f"standalone_{dim_id}", fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        var_name = f"STANDALONE_PROMPTS_DIM{dim_id}"
        if not hasattr(mod, var_name):
            print(f"  SKIP {fname.name}: missing {var_name}")
            continue

        prompts = getattr(mod, var_name)

        concepts.append({
            "dim_id": dim_id,
            "name": dim_name,
            "prompts": prompts,
        })

    return concepts


# ========================== ALIGNMENT ========================== #

def compute_cosine_similarity(vec, mat):
    """Cosine similarity between a vector and each row of a matrix."""
    vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
    mat_norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat_norms @ vec_norm


# ========================== MAIN ========================== #

def run(model_obj, tokenizer):
    """Main pipeline."""

    # ── Load character activations ──
    char_data_dir = data_dir("expanded_mental_concepts", "internals", "rsa")
    char_path = os.path.join(char_data_dir, "all_character_activations.npz")
    if not os.path.exists(char_path):
        raise FileNotFoundError(
            f"Character activations not found at {char_path}. "
            "Run activation_rsa.py first."
        )

    print("Loading character activations...")
    char_data = np.load(char_path)
    char_acts = char_data["activations"]  # (n_characters, n_layers, hidden_dim)
    char_keys = list(char_data["character_keys"])
    n_chars, n_layers, hidden_dim = char_acts.shape

    ai_indices = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "ai"]
    human_indices = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "human"]
    print(f"  {len(ai_indices)} AI + {len(human_indices)} human characters, "
          f"{n_layers} layers, {hidden_dim} dims")

    # ── Discover concepts ──
    concepts = discover_standalone_concepts()
    n_concepts = len(concepts)
    print(f"\nDiscovered {n_concepts} standalone concepts:")
    for c in concepts:
        print(f"  dim {c['dim_id']:2d} {c['name']:25s} ({len(c['prompts'])} prompts)")

    # ── Extract concept activations ──
    standalone_acts = np.zeros((n_concepts, n_layers, hidden_dim), dtype=np.float32)

    for ci, concept in enumerate(concepts):
        print(f"\n[{ci+1}/{n_concepts}] dim {concept['dim_id']} ({concept['name']})")
        print(f"  {len(concept['prompts'])} prompts...")
        all_acts = []
        for prompt in concept["prompts"]:
            acts = extract_activations(model_obj, tokenizer, prompt, config.IS_CHAT)
            all_acts.append(acts.numpy())
        standalone_acts[ci] = np.mean(all_acts, axis=0)

    # ── Compute similarities ──
    print("\nComputing cosine similarities...")
    # sim_matrices: (n_concepts, n_layers, n_characters)
    sim_matrices = np.zeros((n_concepts, n_layers, n_chars), dtype=np.float32)

    for ci in range(n_concepts):
        for li in range(n_layers):
            char_layer = char_acts[:, li, :]
            sim_matrices[ci, li] = compute_cosine_similarity(
                standalone_acts[ci, li], char_layer
            )

    # ── Compute alignment bias ──
    print("Computing human vs AI bias...")
    results = []

    for ci, concept in enumerate(concepts):
        concept_result = {
            "dim_id": concept["dim_id"],
            "name": concept["name"],
            "n_prompts": len(concept["prompts"]),
            "layers": [],
        }

        for li in range(n_layers):
            sim_to_human = sim_matrices[ci, li, human_indices]
            sim_to_ai = sim_matrices[ci, li, ai_indices]
            # Positive = closer to humans, negative = closer to AI
            bias = float(np.mean(sim_to_human) - np.mean(sim_to_ai))

            concept_result["layers"].append({
                "layer": li,
                "mean_sim_human": round(float(np.mean(sim_to_human)), 6),
                "mean_sim_ai": round(float(np.mean(sim_to_ai)), 6),
                "human_ai_bias": round(bias, 6),
            })

        # Peak bias (largest absolute value)
        biases = [l["human_ai_bias"] for l in concept_result["layers"]]
        abs_biases = [abs(b) for b in biases]
        peak_layer = int(np.argmax(abs_biases))
        concept_result["peak_layer"] = peak_layer
        concept_result["peak_bias"] = biases[peak_layer]

        results.append(concept_result)
        print(f"  dim {concept['dim_id']:2d}: peak layer {peak_layer}, "
              f"bias = {biases[peak_layer]:+.4f} "
              f"({'human' if biases[peak_layer] > 0 else 'AI'})")

    # ── Summary ──
    summary = {
        "model": config.MODEL_KEY,
        "model_label": config.MODEL_LABEL,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_concepts": n_concepts,
        "n_characters": n_chars,
        "n_layers": n_layers,
    }

    # ── Save ──
    out_dir = data_dir("expanded_mental_concepts", "internals", "standalone_alignment")

    np.savez_compressed(
        os.path.join(out_dir, "standalone_activations.npz"),
        activations=standalone_acts,
        dim_ids=np.array([c["dim_id"] for c in concepts]),
        dim_names=np.array([c["name"] for c in concepts]),
    )

    np.savez_compressed(
        os.path.join(out_dir, "similarity_matrices.npz"),
        similarities=sim_matrices,
        dim_ids=np.array([c["dim_id"] for c in concepts]),
        character_keys=np.array(char_keys),
    )

    output = {"summary": summary, "concepts": results}
    with open(os.path.join(out_dir, "alignment_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Data saved to: {out_dir}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Standalone-Character Alignment Analysis"
    )
    add_model_argument(parser)
    args = parser.parse_args()

    set_model(args.model)
    device = get_device()

    print(f"Model: {config.MODEL_LABEL}")
    print(f"Device: {device}")
    print(f"Standalone dir: {STANDALONE_DIR}")

    print(f"\nLoading {config.MODEL_LABEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model_obj = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model_obj.half().to(device).eval()
    print(f"Model loaded on {device}.")

    run(model_obj, tokenizer)
    print("\nDone.")


if __name__ == "__main__":
    main()
