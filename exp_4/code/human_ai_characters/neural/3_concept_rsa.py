#!/usr/bin/env python3
"""
Concept Geometry, Phase C: Concept-Specific RSA

For each concept dimension, contextualizes standalone prompts to each of
28 characters, extracts activations, computes concept-specific RDMs, and
runs RSA against a categorical RDM (human vs AI).

This is the most computationally intensive phase:
    ~22 concepts x 28 characters x 40 prompts = ~24,000+ forward passes.

Pipeline per concept:
    1. For each character: extract activations for all 40 contextualized
       prompts -> average to get character's concept vector
    2. Stack: (n_characters, n_layers, hidden_dim) per concept
    3. Compute concept-specific RDM (cosine distance between characters)
    4. RSA with categorical RDM at every layer
    5. FDR correction across layers

Prompt contextualization:
    Standalone: "Imagine experiencing a sudden wave of fear"
    Contextualized: "Think about what it is like for Claude to experience
                     a sudden wave of fear"

Output per concept:
    results/{model}/expanded_mental_concepts/internals/concept_rsa/{concept_key}/data/
        character_concept_activations.npz
        rdm_cosine_per_layer.npz
        rsa_results.json

Cross-concept summary:
    results/{model}/expanded_mental_concepts/internals/concept_rsa/data/
        cross_concept_rsa_summary.json

Usage:
    python expanded_mental_concepts/internals/concept_rsa/concept_rsa.py --model llama2_13b_chat
    python expanded_mental_concepts/internals/concept_rsa/concept_rsa.py --model llama2_13b_chat --concept phenomenology
    python expanded_mental_concepts/internals/concept_rsa/concept_rsa.py --model llama2_13b_base --both

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import false_discovery_control

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    get_device, ensure_dir,
)
from entities.characters import (
    ALL_CHARACTERS, AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES, N_CHARACTERS,
)
from expanded_mental_concepts.concepts import CONCEPT_DIMENSIONS, CONCEPT_KEYS
from utils.utils import (
    compute_rdm_cosine,
    compute_rsa_all_layers,
    compute_categorical_rdm,
    format_chat_prompt,
)


# ========================== CONFIG ========================== #

torch.manual_seed(12345)

# Root results dir for concept RSA
def _concept_rsa_dir(concept_key=None):
    """Return results dir for concept RSA phase."""
    from config import ROOT_DIR, get_active_model
    model = get_active_model()
    if model is None:
        raise RuntimeError("Call set_model() first")
    base = ROOT_DIR / "results" / model / "human_ai_characters" / "internals" / "concept_rsa"
    if concept_key:
        return ensure_dir(base / concept_key / "data")
    return ensure_dir(base / "data")


# ========================== PROMPT CONTEXTUALIZATION ========================== #

def contextualize_prompt(standalone_prompt, character_name):
    """
    Adapt a standalone concept prompt to reference a specific character.

    Strategy: prefix with "Think about what it is like for {Name} to ..."
    and lowercase the beginning of the standalone prompt to flow naturally.

    Examples:
        "Imagine experiencing a sudden wave of fear"
        -> "Think about what it is like for Claude to experience a sudden wave of fear"

        "Think about the raw sensory quality of hearing a single, clear musical note."
        -> "Think about what it is like for Sarah to experience the raw sensory quality of hearing a single, clear musical note."
    """
    # Strip trailing period for joining
    prompt = standalone_prompt.rstrip(".")

    # Remove leading "Imagine", "Think about", "Consider" for cleaner joining
    cleaned = prompt
    for prefix in ["Imagine what it is like to ",
                   "Imagine ",
                   "Think about what it is like to ",
                   "Think about what it is like when ",
                   "Think about what it means to ",
                   "Think about what it means for ",
                   "Think about what happens when ",
                   "Think about the ",
                   "Think about a ",
                   "Think about an ",
                   "Think about ",
                   "Consider what it is like to ",
                   "Consider what it is like when ",
                   "Consider what it means to ",
                   "Consider what it means for ",
                   "Consider what the ",
                   "Consider the ",
                   "Consider a ",
                   "Consider an ",
                   "Consider being ",
                   "Consider how ",
                   "Consider "]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    # Lowercase first char if it's uppercase (natural flow)
    if cleaned and cleaned[0].isupper():
        cleaned = cleaned[0].lower() + cleaned[1:]

    return f"Think about what it is like for {character_name} to experience {cleaned}."


# ========================== EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text, is_chat):
    """
    Forward pass on a single prompt; extract residual-stream activations
    at the last token position across all layers.
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


# ========================== PER-CONCEPT PIPELINE ========================== #

def run_concept(model_obj, tokenizer, concept_key, char_keys, categorical_rdm):
    """
    Run the full concept-specific RSA pipeline for one concept.

    1. Contextualize prompts to each character
    2. Extract activations, average across prompts per character
    3. Compute concept-specific RDM
    4. RSA with categorical RDM
    5. FDR correction
    """
    concept = CONCEPT_DIMENSIONS[concept_key]
    prompts = concept["prompts"]
    n_prompts = len(prompts)
    n_chars = len(char_keys)

    print(f"\n--- {concept_key} ({concept['name']}) ---")
    print(f"  {n_prompts} prompts x {n_chars} characters = "
          f"{n_prompts * n_chars} forward passes")

    ddir = _concept_rsa_dir(concept_key)

    # Extract and average activations per character
    char_concept_acts = []  # will be (n_chars, n_layers, hidden_dim)

    for c_idx, char_key in enumerate(char_keys):
        char_name = CHARACTER_NAMES[char_key]
        prompt_acts = []

        for p_idx, standalone_prompt in enumerate(prompts):
            ctx_prompt = contextualize_prompt(standalone_prompt, char_name)

            acts = extract_activations(
                model_obj, tokenizer, ctx_prompt, config.IS_CHAT
            )
            prompt_acts.append(acts)

        # Average across prompts: (n_prompts, n_layers, hidden_dim) -> (n_layers, hidden_dim)
        stacked = torch.stack(prompt_acts)  # (n_prompts, n_layers, hidden_dim)
        mean_acts = stacked.mean(dim=0)     # (n_layers, hidden_dim)
        char_concept_acts.append(mean_acts)

        total_done = (c_idx + 1) * n_prompts
        total_needed = n_chars * n_prompts
        print(f"  [{c_idx+1}/{n_chars}] {char_key}: "
              f"{total_done}/{total_needed} forward passes done")

    # Stack: (n_chars, n_layers, hidden_dim)
    acts_array = torch.stack(char_concept_acts).numpy()

    # Compute RDM
    model_rdm = compute_rdm_cosine(acts_array)

    # RSA with categorical RDM
    rsa_results = compute_rsa_all_layers(model_rdm, categorical_rdm, n_chars)

    # FDR correction
    p_values = np.array([r["p_value"] for r in rsa_results])
    valid_mask = ~np.isnan(p_values)
    if np.sum(valid_mask) > 0:
        valid_p = p_values[valid_mask]
        fdr_corrected = np.full_like(p_values, np.nan)
        fdr_corrected[valid_mask] = false_discovery_control(valid_p)
        for i, r in enumerate(rsa_results):
            r["p_fdr"] = float(fdr_corrected[i]) if not np.isnan(fdr_corrected[i]) else None

    # Find peak
    valid = [r for r in rsa_results if not np.isnan(r["rho"])]
    if valid:
        peak = max(valid, key=lambda r: r["rho"])
        print(f"  Peak: Layer {peak['layer']}, rho={peak['rho']:+.4f}, "
              f"p={peak['p_value']:.4f}")
    else:
        peak = {"layer": -1, "rho": float("nan"), "p_value": float("nan")}

    # Save
    np.savez_compressed(
        os.path.join(ddir, "character_concept_activations.npz"),
        activations=acts_array,
        character_keys=np.array(char_keys),
        concept_key=concept_key,
    )
    np.savez_compressed(
        os.path.join(ddir, "rdm_cosine_per_layer.npz"),
        model_rdm=model_rdm,
        categorical_rdm=categorical_rdm,
        character_keys=np.array(char_keys),
    )
    with open(os.path.join(ddir, "rsa_results.json"), "w") as f:
        json.dump(rsa_results, f, indent=2)

    return {
        "concept_key": concept_key,
        "concept_name": concept["name"],
        "n_prompts": n_prompts,
        "peak_layer": peak["layer"],
        "peak_rho": peak["rho"],
        "peak_p": peak["p_value"],
        "peak_p_fdr": peak.get("p_fdr"),
    }


# ========================== CROSS-CONCEPT SUMMARY ========================== #

def write_cross_concept_summary(summaries, char_keys):
    """Write a cross-concept summary JSON and markdown."""
    summary_dir = _concept_rsa_dir()

    # JSON summary
    with open(os.path.join(summary_dir, "cross_concept_rsa_summary.json"), "w") as f:
        json.dump({
            "model": config.MODEL_LABEL,
            "n_characters": len(char_keys),
            "n_ai": len([k for k in char_keys if k in AI_CHARACTERS]),
            "n_human": len([k for k in char_keys if k in HUMAN_CHARACTERS]),
            "concepts": summaries,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Markdown summary
    md_path = os.path.join(summary_dir, "cross_concept_rsa_summary.md")
    with open(md_path, "w") as f:
        f.write("# Concept Geometry, Phase C: Cross-Concept RSA Summary\n\n")
        f.write(f"**Model:** {config.MODEL_LABEL}\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Peak RSA per Concept\n\n")
        f.write("| Concept | Peak Layer | Peak rho | p-value | FDR p |\n")
        f.write("|---------|----------:|---------:|--------:|------:|\n")

        # Sort by peak rho descending
        sorted_summaries = sorted(
            summaries, key=lambda s: s["peak_rho"] if not np.isnan(s["peak_rho"]) else -1,
            reverse=True
        )
        for s in sorted_summaries:
            fdr_str = f"{s['peak_p_fdr']:.4f}" if s.get("peak_p_fdr") is not None else "n/a"
            f.write(f"| {s['concept_name']} | {s['peak_layer']} | "
                    f"{s['peak_rho']:+.4f} | {s['peak_p']:.4f} | {fdr_str} |\n")
        f.write("\n")

        # Which concepts show strongest human/AI divergence
        sig_concepts = [s for s in sorted_summaries
                        if not np.isnan(s["peak_rho"]) and s["peak_p"] < 0.05]
        if sig_concepts:
            f.write("## Concepts with significant human/AI divergence (p < .05)\n\n")
            for s in sig_concepts:
                f.write(f"- **{s['concept_name']}**: Layer {s['peak_layer']}, "
                        f"rho = {s['peak_rho']:+.4f}\n")
        else:
            f.write("## No concepts reached p < .05 for human/AI divergence\n")
        f.write("\n")

    print(f"\nCross-concept summary: {md_path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Concept Geometry Phase C: Concept-Specific RSA"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--concept", type=str, default=None,
        help="Run a single concept (by key, e.g. 'phenomenology'). "
             "If not specified, runs all concepts."
    )
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

        # Determine which concepts to run
        if args.concept:
            if args.concept not in CONCEPT_DIMENSIONS:
                available = ", ".join(CONCEPT_KEYS)
                print(f"Error: unknown concept '{args.concept}'. "
                      f"Available: {available}")
                sys.exit(1)
            concepts_to_run = [args.concept]
        else:
            concepts_to_run = list(CONCEPT_KEYS)

        char_keys = list(ALL_CHARACTERS)
        n_total_passes = sum(
            len(CONCEPT_DIMENSIONS[c]["prompts"]) * len(char_keys)
            for c in concepts_to_run
        )

        print(f"Model: {config.MODEL_LABEL}")
        print(f"Device: {device}")
        print(f"Concepts to run: {len(concepts_to_run)}")
        print(f"Characters: {len(char_keys)}")
        print(f"Total forward passes: {n_total_passes}")

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

        # Compute categorical RDM (shared across all concepts)
        categorical_rdm = compute_categorical_rdm(char_keys, CHARACTER_TYPES)

        # Run each concept
        summaries = []
        for c_idx, concept_key in enumerate(concepts_to_run):
            print(f"\n[{c_idx+1}/{len(concepts_to_run)}] "
                  f"Processing {concept_key}...")
            summary = run_concept(
                model_obj, tokenizer, concept_key, char_keys, categorical_rdm
            )
            summaries.append(summary)

        # Write cross-concept summary
        if len(concepts_to_run) > 1:
            write_cross_concept_summary(summaries, char_keys)

        print(f"\nDone with {model_key}.")


if __name__ == "__main__":
    main()
