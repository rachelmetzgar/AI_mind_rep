#!/usr/bin/env python3
"""
Concept Geometry: Contrast-Character Alignment

Tests whether entity-framed concept prompts from exp_3 contrasts align with the
"correct" character group in activation space. Human-framed prompts (e.g.,
"Imagine a human who firmly believes...") should produce activations closer to
human characters; AI-framed prompts should be closer to AI characters. Control
dimensions (shapes, rocks) should show no alignment.

Procedure:
    1. Load pre-computed character activations (from activation_rsa.py)
    2. Auto-discover exp_3 contrast concepts
    3. Extract mean activations per framing (human/AI) for each concept
    4. Cosine similarity to each of 30 characters per layer
    5. Alignment = mean(sim to correct group) - mean(sim to incorrect group)

Output → results/{model}/expanded_mental_concepts/internals/contrast_alignment/data/:
    contrast_activations.npz    (n_concepts, 2, n_layers, hidden_dim)
    similarity_matrices.npz     (n_concepts, n_layers, 2, n_characters)
    alignment_results.json      per-concept per-layer alignment metrics

Usage:
    python expanded_mental_concepts/internals/contrast_alignment/contrast_alignment.py --model llama2_13b_chat

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
from utils.utils import format_chat_prompt


# ========================== CONFIG ========================== #

torch.manual_seed(12345)

# Entity-framed dimensions: human vs AI prompts expected to align
ENTITY_FRAMED_DIMS = set(range(0, 15)) | {16, 17, 25, 26, 27}

# Control dimensions: no expected alignment
CONTROL_DIMS = {15, 29, 30, 31, 32}

CONTRASTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "exp_3" / "concepts" / "contrasts"


# ========================== CONCEPT DISCOVERY ========================== #

def discover_contrast_concepts():
    """
    Auto-discover exp_3 contrast concept files.
    Returns list of dicts with keys: dim_id, name, human_prompts, ai_prompts.
    """
    concepts = []
    pattern = re.compile(r"^(\d+)_(.+)\.py$")

    for fname in sorted(CONTRASTS_DIR.iterdir()):
        m = pattern.match(fname.name)
        if not m:
            continue

        dim_id = int(m.group(1))
        dim_name = m.group(2)

        # Import the module
        spec = importlib.util.spec_from_file_location(f"contrast_{dim_id}", fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        human_var = f"HUMAN_PROMPTS_DIM{dim_id}"
        ai_var = f"AI_PROMPTS_DIM{dim_id}"

        if not hasattr(mod, human_var) or not hasattr(mod, ai_var):
            print(f"  SKIP {fname.name}: missing {human_var}/{ai_var}")
            continue

        human_prompts = getattr(mod, human_var)
        ai_prompts = getattr(mod, ai_var)

        concepts.append({
            "dim_id": dim_id,
            "name": dim_name,
            "human_prompts": human_prompts,
            "ai_prompts": ai_prompts,
            "is_entity_framed": dim_id in ENTITY_FRAMED_DIMS,
            "is_control": dim_id in CONTROL_DIMS,
        })

    return concepts


# ========================== ALIGNMENT ========================== #

def compute_cosine_similarity(vec, mat):
    """
    Cosine similarity between a vector (hidden_dim,) and each row
    of a matrix (n, hidden_dim). Returns (n,) array.
    """
    vec_norm = vec / (np.linalg.norm(vec) + 1e-12)
    mat_norms = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat_norms @ vec_norm


def compute_alignment_score(sim_correct, sim_incorrect):
    """Alignment = mean(sim to correct group) - mean(sim to incorrect group)."""
    return float(np.mean(sim_correct) - np.mean(sim_incorrect))


# ========================== MAIN ========================== #

def run(model_obj, tokenizer):
    """Main pipeline: extract, compare, save."""

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

    # Build character group indices
    ai_indices = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "ai"]
    human_indices = [i for i, k in enumerate(char_keys) if CHARACTER_TYPES[k] == "human"]
    print(f"  {len(ai_indices)} AI + {len(human_indices)} human characters, "
          f"{n_layers} layers, {hidden_dim} dims")

    # ── Discover concepts ──
    concepts = discover_contrast_concepts()
    n_concepts = len(concepts)
    print(f"\nDiscovered {n_concepts} contrast concepts:")
    for c in concepts:
        tag = "ENTITY" if c["is_entity_framed"] else ("CONTROL" if c["is_control"] else "other")
        print(f"  dim {c['dim_id']:2d} {c['name']:25s} [{tag}] "
              f"({len(c['human_prompts'])}+{len(c['ai_prompts'])} prompts)")

    # ── Extract concept activations ──
    # contrast_acts: (n_concepts, 2, n_layers, hidden_dim)  [0=human, 1=AI mean]
    contrast_acts = np.zeros((n_concepts, 2, n_layers, hidden_dim), dtype=np.float32)

    for ci, concept in enumerate(concepts):
        print(f"\n[{ci+1}/{n_concepts}] dim {concept['dim_id']} ({concept['name']})")

        for fi, (framing, prompts) in enumerate([
            ("human", concept["human_prompts"]),
            ("AI", concept["ai_prompts"]),
        ]):
            print(f"  {framing} framing: {len(prompts)} prompts...")
            all_acts = []
            for prompt in prompts:
                acts = extract_activations(model_obj, tokenizer, prompt, config.IS_CHAT)
                all_acts.append(acts.numpy())
            # Mean across prompts
            mean_acts = np.mean(all_acts, axis=0)  # (n_layers, hidden_dim)
            contrast_acts[ci, fi] = mean_acts

    # ── Compute similarities ──
    print("\nComputing cosine similarities...")
    # sim_matrices: (n_concepts, n_layers, 2, n_characters)
    sim_matrices = np.zeros((n_concepts, n_layers, 2, n_chars), dtype=np.float32)

    for ci in range(n_concepts):
        for li in range(n_layers):
            char_layer = char_acts[:, li, :]  # (n_chars, hidden_dim)
            for fi in range(2):
                concept_vec = contrast_acts[ci, fi, li]
                sim_matrices[ci, li, fi] = compute_cosine_similarity(concept_vec, char_layer)

    # ── Compute alignment scores ──
    print("Computing alignment scores...")
    results = []

    for ci, concept in enumerate(concepts):
        concept_result = {
            "dim_id": concept["dim_id"],
            "name": concept["name"],
            "is_entity_framed": concept["is_entity_framed"],
            "is_control": concept["is_control"],
            "n_human_prompts": len(concept["human_prompts"]),
            "n_ai_prompts": len(concept["ai_prompts"]),
            "layers": [],
        }

        for li in range(n_layers):
            # Human-framed prompts: correct = human chars, incorrect = AI chars
            human_sim_to_human = sim_matrices[ci, li, 0, human_indices]
            human_sim_to_ai = sim_matrices[ci, li, 0, ai_indices]
            human_alignment = compute_alignment_score(human_sim_to_human, human_sim_to_ai)

            # AI-framed prompts: correct = AI chars, incorrect = human chars
            ai_sim_to_ai = sim_matrices[ci, li, 1, ai_indices]
            ai_sim_to_human = sim_matrices[ci, li, 1, human_indices]
            ai_alignment = compute_alignment_score(ai_sim_to_ai, ai_sim_to_human)

            # Combined alignment (average of both framings)
            combined = (human_alignment + ai_alignment) / 2

            concept_result["layers"].append({
                "layer": li,
                "human_alignment": round(human_alignment, 6),
                "ai_alignment": round(ai_alignment, 6),
                "combined_alignment": round(combined, 6),
            })

        # Find peak layer
        combined_scores = [l["combined_alignment"] for l in concept_result["layers"]]
        peak_layer = int(np.argmax(combined_scores))
        concept_result["peak_layer"] = peak_layer
        concept_result["peak_alignment"] = combined_scores[peak_layer]

        results.append(concept_result)

        tag = "ENTITY" if concept["is_entity_framed"] else "CTRL"
        print(f"  dim {concept['dim_id']:2d} [{tag}]: "
              f"peak layer {peak_layer}, alignment = {combined_scores[peak_layer]:+.4f}")

    # ── Summary statistics ──
    entity_results = [r for r in results if r["is_entity_framed"]]
    control_results = [r for r in results if r["is_control"]]

    summary = {
        "model": config.MODEL_KEY,
        "model_label": config.MODEL_LABEL,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_concepts": n_concepts,
        "n_entity_framed": len(entity_results),
        "n_control": len(control_results),
        "n_characters": n_chars,
        "n_layers": n_layers,
        "entity_mean_peak_alignment": round(
            float(np.mean([r["peak_alignment"] for r in entity_results])), 6
        ) if entity_results else None,
        "control_mean_peak_alignment": round(
            float(np.mean([r["peak_alignment"] for r in control_results])), 6
        ) if control_results else None,
    }

    # ── Save ──
    out_dir = data_dir("expanded_mental_concepts", "internals", "contrast_alignment")
    rdir = results_dir("expanded_mental_concepts", "internals", "contrast_alignment")

    np.savez_compressed(
        os.path.join(out_dir, "contrast_activations.npz"),
        activations=contrast_acts,
        dim_ids=np.array([c["dim_id"] for c in concepts]),
        dim_names=np.array([c["name"] for c in concepts]),
    )

    np.savez_compressed(
        os.path.join(out_dir, "similarity_matrices.npz"),
        similarities=sim_matrices,
        dim_ids=np.array([c["dim_id"] for c in concepts]),
        character_keys=np.array(char_keys),
    )

    output = {
        "summary": summary,
        "concepts": results,
    }
    with open(os.path.join(out_dir, "alignment_results.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Entity-framed ({len(entity_results)} dims): "
          f"mean peak alignment = {summary['entity_mean_peak_alignment']:+.4f}")
    print(f"  Controls ({len(control_results)} dims): "
          f"mean peak alignment = {summary['control_mean_peak_alignment']:+.4f}")
    print(f"\n  Data saved to: {out_dir}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Contrast-Character Alignment Analysis"
    )
    add_model_argument(parser)
    args = parser.parse_args()

    set_model(args.model)
    device = get_device()

    print(f"Model: {config.MODEL_LABEL}")
    print(f"Device: {device}")
    print(f"Contrasts dir: {CONTRASTS_DIR}")

    # Load model
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
