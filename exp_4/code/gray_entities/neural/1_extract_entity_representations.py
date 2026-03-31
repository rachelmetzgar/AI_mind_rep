#!/usr/bin/env python3
"""
Experiment 4, Phase 1: Entity Representation Extraction

Extracts internal representations for each of the 13 Gray et al. (2007)
entities from LLaMA-2-13B (chat or base variant). One prompt per entity
("Think about {X}"), last-token residual-stream activations across all layers.

Computes three human RDM variants (combined, experience-only, agency-only)
and runs RSA at every layer for each variant.

Output:
    data_dir("gray_entities", "neural", tag)/
        all_entity_activations.npz   # (n_entities, n_layers, hidden_dim)
        entity_prompts.json          # prompt metadata
        rdm_cosine_per_layer.npz     # model RDM + all 3 human RDMs

    data_dir("gray_entities", "neural", tag)/
        rsa_results.json             # all 3 RSA variants per layer

    results_dir("gray_entities", "neural", tag)/
        internals_results_summary.md # documented methodology + results

Usage:
    python 1_extract_entity_representations.py --model llama2_13b_chat
    python 1_extract_entity_representations.py --model llama2_13b_base --include_self
    python 1_extract_entity_representations.py --model llama2_13b_chat --both

SLURM:
    sbatch slurm/1_extract_entities.sh

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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir, get_device, get_condition_tag,
)
from utils.utils import (
    compute_rdm_cosine, compute_all_human_rdms,
    compute_rsa_all_layers, format_chat_prompt,
)
from entities.gray_entities import GRAY_ET_AL_SCORES, ENTITY_PROMPTS, ENTITY_NAMES


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
        Layer 0 = embedding output, layers 1..N = transformer block outputs.
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

    # Collect last-token activation from each layer
    last_acts = []
    for layer_hidden in output["hidden_states"]:
        act = layer_hidden[:, -1].detach().cpu().to(torch.float32)
        last_acts.append(act)

    return torch.cat(last_acts, dim=0)  # (n_layers+1, hidden_dim)


# ========================== RESULTS SUMMARY ========================== #

def write_results_summary(out_dir, rsa_all, entity_keys, tag, include_self):
    """Write a documented markdown summary of the analysis and results."""
    n_entities = len(entity_keys)
    model_label = config.MODEL_LABEL

    summary_path = os.path.join(out_dir, "internals_results_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Experiment 4, Phase 1: Entity Representation Extraction\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Tag:** `{tag}`\n")
        f.write(f"**Include self:** {include_self}\n")
        f.write(f"**Model:** {model_label}\n\n")

        f.write("---\n\n")
        f.write("## What is being tested\n\n")
        f.write(
            f"Does {model_label}'s internal representational geometry over "
            "diverse entities mirror the human folk-psychological geometry "
            "from Gray, Gray, & Wegner (2007)?\n\n"
            "Gray et al. had ~2,400 human participants rate 13 entities "
            "(from dead woman to God) on 18 mental capacities. Factor analysis "
            "revealed two dimensions: **Experience** (capacity to feel) and "
            "**Agency** (capacity to act). Each entity has a position in this "
            "2D space.\n\n"
            "We ask: does the model's internal representation of these entities "
            "have a similar geometric structure?\n\n"
        )

        f.write("## Method: Representational Similarity Analysis (RSA)\n\n")
        f.write(
            "RSA compares two distance matrices (Kriegeskorte et al., 2008). "
            "Instead of comparing representations directly (which would require "
            "them to be in the same space), we compare the *pattern of distances* "
            "between entities.\n\n"
            "**Inputs:**\n\n"
            f"- **Model RDM**: For each of {n_entities} entities, we extract the "
            f"last-token residual-stream activation from {model_label} in "
            "response to a simple prompt (\"Think about {{entity}}.\"). We compute "
            "pairwise **cosine distance** (1 - cosine similarity) between all "
            f"entity pairs -> a {n_entities}x{n_entities} representational "
            "dissimilarity matrix (RDM). Computed separately at each of 41 layers "
            "(layer 0 = embedding, layers 1-40 = transformer blocks).\n\n"
            "- **Human RDMs**: Three variants from Gray et al. (2007) factor "
            "scores:\n"
            "  - Combined: Euclidean distance in 2D (Experience, Agency) space\n"
            "  - Experience-only: |exp_i - exp_j|\n"
            "  - Agency-only: |agency_i - agency_j|\n\n"
            "**Test:** Spearman rank correlation between the upper triangles of the "
            f"model RDM and each human RDM ({n_entities * (n_entities - 1) // 2} "
            "unique entity pairs). Computed at every layer to track where in the "
            "network mind-perception structure emerges.\n\n"
            "**Interpretation:** A positive Spearman rho means entities that are "
            "far apart in human mind-perception space are also far apart in the "
            "model's activation space (and vice versa). The model's entity geometry "
            "mirrors human folk psychology.\n\n"
        )

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

            f.write(f"## RSA by Layer -- {variant_name.title()}\n\n")
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

        f.write("## Notes\n\n")
        f.write(
            "- Factor scores from Gray et al. (2007) were estimated by AI from "
            "Figure 1 and should be verified before final analyses.\n"
            "- One prompt per entity -- no averaging across prompt variants. "
            "Reliability cannot be assessed with this design; robustness can be "
            "checked by rerunning with a different template.\n"
            "- Layer 0 (embedding) typically produces constant cosine distances "
            "(all activations have similar geometry before any transformer "
            "processing), yielding NaN correlation.\n"
        )

    print(f"  Results summary: {summary_path}")
    return summary_path


# ========================== MAIN ========================== #

def run_condition(model, tokenizer, include_self):
    """Run extraction and RSA for one condition (with_self or without_self)."""
    tag = get_condition_tag(include_self)

    if include_self:
        entity_keys = ENTITY_NAMES
    else:
        entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]

    n_entities = len(entity_keys)
    ddir = data_dir("gray_entities", "neural", tag)
    rdir = results_dir("gray_entities", "neural", tag)

    print(f"\n{'='*60}")
    print(f"  Condition: {tag} ({n_entities} entities)")
    print(f"  Data dir: {ddir}")
    print(f"  Results dir: {rdir}")
    print(f"{'='*60}")

    # Extract activations
    all_activations = []
    prompt_metadata = []

    for entity_key in entity_keys:
        prompt = ENTITY_PROMPTS[entity_key]
        exp_score, agency_score = GRAY_ET_AL_SCORES[entity_key]

        print(f"  {entity_key}: \"{prompt}\"")
        acts = extract_activations(model, tokenizer, prompt, config.IS_CHAT)
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
    human_rdms = compute_all_human_rdms(entity_keys, GRAY_ET_AL_SCORES)

    # RSA at every layer for all 3 variants
    print("\nComputing RSA (Spearman) at all layers...")
    rsa_all = {}
    for variant_name, human_rdm in human_rdms.items():
        rsa_all[variant_name] = compute_rsa_all_layers(
            model_rdm, human_rdm, n_entities
        )

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
        os.path.join(ddir, "all_entity_activations.npz"),
        activations=acts_array,
        entity_keys=np.array(entity_keys),
    )
    np.savez_compressed(
        os.path.join(ddir, "rdm_cosine_per_layer.npz"),
        model_rdm=model_rdm,
        human_rdm_combined=human_rdms["combined"],
        human_rdm_experience=human_rdms["experience"],
        human_rdm_agency=human_rdms["agency"],
        entity_keys=np.array(entity_keys),
    )
    with open(os.path.join(ddir, "entity_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    # ── Save RSA results ──
    with open(os.path.join(ddir, "rsa_results.json"), "w") as f:
        json.dump(rsa_all, f, indent=2)

    write_results_summary(rdir, rsa_all, entity_keys, tag, include_self)

    print(f"\nSaved data to {ddir}/")
    print(f"Saved results to {rdir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 1: Extract entity representations"
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

    # Set model configuration
    set_model(args.model)
    device = get_device()

    print(f"Model: {config.MODEL_LABEL}")
    print(f"Model path: {config.MODEL_PATH}")
    print(f"Is chat: {config.IS_CHAT}")
    print(f"Device: {device}")

    # Load model
    print(f"\nLoading {config.MODEL_LABEL}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH, local_files_only=config.LOCAL_FILES_ONLY
    )
    model.half().to(device).eval()
    print(f"Model loaded on {device}.")

    # Run extraction
    if args.both:
        run_condition(model, tokenizer, include_self=False)
        run_condition(model, tokenizer, include_self=True)
    else:
        run_condition(model, tokenizer, include_self=args.include_self)

    print("\nDone.")


if __name__ == "__main__":
    main()
