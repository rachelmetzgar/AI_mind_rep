#!/usr/bin/env python3
"""
Concept Geometry, Phase B: Internal RSA

Extracts internal representations for 28 characters (14 AI + 14 human) from
LLaMA-2-13B (chat or base) using simple prompts ("Think about {Name}.").
Computes RSA between the model's representational geometry and two reference
RDMs:
    - Categorical RDM: binary (same type = 0, cross type = 1)
    - Behavioral RDM: Euclidean distance in Phase A PCA factor space
      (loaded from Phase A output if available)

Adapts internals/1_extract_entity_representations.py for concept geometry
characters instead of Gray et al. entities.

Output:
    results/{model}/expanded_mental_concepts/internals/rsa/data/
        all_character_activations.npz   # (n_characters, n_layers, hidden_dim)
        character_prompts.json          # prompt metadata
        rdm_cosine_per_layer.npz        # model RDM + reference RDMs
        rsa_results.json                # RSA per layer per reference RDM

    results/{model}/expanded_mental_concepts/internals/rsa/
        internals_results_summary.md

Usage:
    python expanded_mental_concepts/internals/rsa/activation_rsa.py --model llama2_13b_chat
    python expanded_mental_concepts/internals/rsa/activation_rsa.py --model llama2_13b_base
    python expanded_mental_concepts/internals/rsa/activation_rsa.py --model llama2_13b_chat --both

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
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
    CHARACTER_NAMES, CHARACTER_PROMPTS, CHARACTER_TYPES,
    N_CHARACTERS,
)
from utils.utils import (
    compute_rdm_cosine,
    compute_rsa_all_layers,
    compute_categorical_rdm,
    compute_behavioral_rdm,
    format_chat_prompt,
)


# ========================== CONFIG ========================== #

torch.manual_seed(12345)


# ========================== EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text, is_chat):
    """
    Forward pass on a single prompt; extract residual-stream activations
    at the last token position across all layers.

    For chat model: wraps prompt in LLaMA-2 chat template.
    For base model: tokenizes raw text directly.

    Returns: tensor of shape (n_layers+1, hidden_dim)
    """
    if is_chat:
        messages = [{"role": "user", "content": prompt_text}]
        formatted = format_chat_prompt(messages)
    else:
        formatted = prompt_text

    device = get_device()

    with torch.no_grad():
        encoding = tokenizer(
            formatted,
            truncation=True,
            max_length=2048,
            return_attention_mask=True,
            return_tensors="pt",
        )
        output = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
            output_hidden_states=True,
            return_dict=True,
        )

    last_acts = []
    for layer_hidden in output["hidden_states"]:
        act = layer_hidden[:, -1].detach().cpu().to(torch.float32)
        last_acts.append(act)

    return torch.cat(last_acts, dim=0)  # (n_layers+1, hidden_dim)


# ========================== RESULTS SUMMARY ========================== #

def write_results_summary(out_dir, rsa_all, char_keys):
    """Write documented markdown summary."""
    n_chars = len(char_keys)
    model_label = config.MODEL_LABEL

    summary_path = os.path.join(out_dir, "internals_results_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Concept Geometry, Phase B: Internal RSA\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Model:** {model_label}\n\n")

        f.write("---\n\n## What is being tested\n\n")
        f.write(
            f"Does {model_label}'s internal representational geometry over "
            "28 characters (14 AI, 14 human) reflect their categorical "
            "distinction? RSA compares the model's activation-based RDM "
            "to reference RDMs at every layer.\n\n"
        )

        f.write("## Method\n\n")
        f.write(
            f"- **Model RDM**: Cosine distance between last-token activations "
            f"for {n_chars} characters, one prompt each "
            "(\"Think about {{Name}}.\"), at each of 41 layers.\n"
            "- **Categorical RDM**: Binary (same type = 0, cross type = 1).\n"
            "- **Behavioral RDM** (if available): Euclidean distance in "
            "Phase A PCA factor space.\n"
            "- **Test**: Spearman correlation between upper triangles.\n\n"
        )

        f.write("## Characters\n\n")
        f.write("| Key | Name | Type |\n")
        f.write("|-----|------|------|\n")
        for key in char_keys:
            f.write(f"| {key} | {CHARACTER_NAMES[key]} | "
                    f"{CHARACTER_TYPES[key]} |\n")
        f.write("\n")

        for variant_name in sorted(rsa_all.keys()):
            rsa_results = rsa_all[variant_name]
            valid = [r for r in rsa_results if not np.isnan(r["rho"])]
            if valid:
                peak = max(valid, key=lambda r: r["rho"])
            else:
                peak = {"layer": -1, "rho": float("nan"),
                        "p_value": float("nan")}

            f.write(f"## RSA by Layer -- {variant_name}\n\n")
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

def run_extraction(model_obj, tokenizer):
    """Run extraction and RSA for concept geometry characters."""
    char_keys = list(ALL_CHARACTERS)
    n_chars = len(char_keys)

    ddir = data_dir("human_ai_characters", "neural/names_only", "rsa_pca")
    rdir = results_dir("human_ai_characters", "neural/names_only", "rsa_pca")

    print(f"\n{'='*60}")
    print(f"  Concept Geometry Internal RSA ({n_chars} characters)")
    print(f"  Data dir: {ddir}")
    print(f"  Results dir: {rdir}")
    print(f"{'='*60}")

    # Extract activations
    all_activations = []
    prompt_metadata = []

    for char_key in char_keys:
        prompt = CHARACTER_PROMPTS[char_key]
        char_type = CHARACTER_TYPES[char_key]

        print(f"  {char_key} ({char_type}): \"{prompt}\"")
        acts = extract_activations(model_obj, tokenizer, prompt, config.IS_CHAT)
        all_activations.append(acts)

        prompt_metadata.append({
            "character": char_key,
            "name": CHARACTER_NAMES[char_key],
            "type": char_type,
            "prompt": prompt,
        })

    # Stack: (n_characters, n_layers, hidden_dim)
    acts_array = torch.stack(all_activations).numpy()
    n_layers = acts_array.shape[1]
    hidden_dim = acts_array.shape[2]

    print(f"\nActivations shape: {acts_array.shape}")
    print(f"  {n_chars} characters x {n_layers} layers x {hidden_dim} dims")

    # Compute model RDM
    print("\nComputing model RDM (cosine distance)...")
    model_rdm = compute_rdm_cosine(acts_array)

    # Compute reference RDMs
    print("Computing categorical RDM...")
    categorical_rdm = compute_categorical_rdm(char_keys, CHARACTER_TYPES)

    # Try to load behavioral RDM from Phase A
    behavioral_rdm = None
    behavior_data_dir = data_dir("human_ai_characters", "behavior", "pca")
    pca_path = os.path.join(str(behavior_data_dir), "pairwise_pca_results.npz")
    if os.path.exists(pca_path):
        print("Loading behavioral RDM from Phase A PCA results...")
        pca_data = np.load(pca_path)
        factor_scores = pca_data["factor_scores_01"]
        behavioral_rdm = compute_behavioral_rdm(factor_scores)
    else:
        print("  Phase A PCA results not found -- skipping behavioral RDM")

    # RSA at every layer
    print("\nComputing RSA (Spearman) at all layers...")
    rsa_all = {}

    rsa_all["categorical"] = compute_rsa_all_layers(
        model_rdm, categorical_rdm, n_chars
    )

    if behavioral_rdm is not None:
        rsa_all["behavioral"] = compute_rsa_all_layers(
            model_rdm, behavioral_rdm, n_chars
        )

    # Print summaries
    for variant_name, rsa_results in rsa_all.items():
        valid = [r for r in rsa_results if not np.isnan(r["rho"])]
        if valid:
            peak = max(valid, key=lambda r: r["rho"])
            print(f"\n  {variant_name}: Peak Layer {peak['layer']}, "
                  f"rho = {peak['rho']:+.4f}, p = {peak['p_value']:.4f}")
        else:
            print(f"\n  {variant_name}: all NaN")

    # ── Save data ──
    save_kwargs = {
        "activations": acts_array,
        "character_keys": np.array(char_keys),
    }
    np.savez_compressed(
        os.path.join(ddir, "all_character_activations.npz"),
        **save_kwargs
    )

    rdm_kwargs = {
        "model_rdm": model_rdm,
        "categorical_rdm": categorical_rdm,
        "character_keys": np.array(char_keys),
    }
    if behavioral_rdm is not None:
        rdm_kwargs["behavioral_rdm"] = behavioral_rdm
    np.savez_compressed(
        os.path.join(ddir, "rdm_cosine_per_layer.npz"),
        **rdm_kwargs
    )

    with open(os.path.join(ddir, "character_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    with open(os.path.join(ddir, "rsa_results.json"), "w") as f:
        json.dump(rsa_all, f, indent=2)

    write_results_summary(rdir, rsa_all, char_keys)

    print(f"\nSaved data to {ddir}/")
    print(f"Saved results to {rdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Concept Geometry Phase B: Internal RSA"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--both", action="store_true",
        help="Run both chat and base models sequentially"
    )
    args = parser.parse_args()

    if args.both:
        models_to_run = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models_to_run = [args.model]

    for model_key in models_to_run:
        print(f"\n{'='*60}")
        print(f"  Running model: {model_key}")
        print(f"{'='*60}\n")
        set_model(model_key)
        device = get_device()

        print(f"Model: {config.MODEL_LABEL}")
        print(f"Device: {device}")

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

        run_extraction(model_obj, tokenizer)

    print("\nDone.")


if __name__ == "__main__":
    main()
