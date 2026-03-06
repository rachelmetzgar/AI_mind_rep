#!/usr/bin/env python3
"""
Experiment 4, Phase 1: Entity Representation Extraction

Extracts internal representations for each of the 13 Gray et al. (2007)
entities from LLaMA-2-13B-Chat. One prompt per entity ("Think about {X}"),
last-token residual-stream activations across all layers.

Computes RSA (representational similarity analysis) at every layer and
saves a documented results summary.

Output:
    data/entity_activations/{tag}/
        all_entity_activations.npz   # (n_entities, n_layers, hidden_dim)
        entity_prompts.json          # prompt metadata
        rdm_cosine_per_layer.npz     # (n_layers, n_entities, n_entities)

    results/{tag}/
        rsa_all_layers.json          # RSA at every layer
        results_summary.md           # documented methodology + results

Usage:
    python 1_extract_entity_representations.py                  # 12 entities (no self)
    python 1_extract_entity_representations.py --include_self   # 13 entities

SLURM:
    sbatch slurm/1_extract_entities.sh

Env: llama2_env
Rachel C. Metzgar · Feb 2026
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

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)


# ========================== PROMPT FORMATTING ========================== #

def llama_v2_prompt(messages, system_prompt=None):
    """Format messages into LLaMA-2-Chat token string.

    If no system message is present in messages, prepends the default
    LLaMA-2 system prompt (or a custom one via system_prompt).
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    if system_prompt:
        default_sys = system_prompt
    else:
        default_sys = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Please ensure that your responses are socially unbiased and "
            "positive in nature. If a question does not make any sense, or "
            "is not factually coherent, explain why instead of answering "
            "something not correct. If you don't know the answer to a "
            "question, please don't share false information."
        )

    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": default_sys}] + messages

    # Fold system prompt into first user message
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS
                       + messages[1]["content"],
        }
    ] + messages[2:]

    # Pair up user/assistant turns
    parts = [
        f"{BOS}{B_INST} {prompt['content'].strip()} {E_INST} "
        f"{answer['content'].strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    # Unpaired final user message
    if messages[-1]["role"] == "user":
        parts.append(
            f"{BOS}{B_INST} {messages[-1]['content'].strip()} {E_INST}"
        )

    return "".join(parts)


# ========================== EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text):
    """
    Forward pass on a single prompt; extract residual-stream activations
    at the last token position across all layers.

    Returns: tensor of shape (n_layers+1, hidden_dim)
        Layer 0 = embedding output, layers 1..N = transformer block outputs.
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
        # Normalize
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_normed = vecs / norms
        # Cosine similarity -> cosine distance
        cos_sim = vecs_normed @ vecs_normed.T
        rdm[layer] = 1.0 - cos_sim

    return rdm


def compute_human_rdm(entity_keys):
    """
    Compute human RDM from Gray et al. Experience/Agency scores
    using Euclidean distance.

    Args:
        entity_keys: list of entity names (ordering)

    Returns:
        rdm: (n_entities, n_entities) Euclidean distances
    """
    n = len(entity_keys)
    coords = np.array([GRAY_ET_AL_SCORES[k] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
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
        # Check for constant input (e.g., embedding layer)
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

def write_results_summary(results_dir, rsa_results, entity_keys, tag,
                          include_self):
    """Write a documented markdown summary of the analysis and results."""
    n_entities = len(entity_keys)
    n_pairs = n_entities * (n_entities - 1) // 2

    # Find peak layer (excluding NaN)
    valid = [r for r in rsa_results if not np.isnan(r["rho"])]
    if valid:
        peak = max(valid, key=lambda r: r["rho"])
    else:
        peak = {"layer": -1, "rho": float("nan"), "p_value": float("nan")}

    summary_path = os.path.join(results_dir, "results_summary.md")
    with open(summary_path, "w") as f:
        f.write("# Experiment 4, Phase 1: Entity Representation Extraction\n\n")
        f.write(f"**Run:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Tag:** `{tag}`\n")
        f.write(f"**Include self:** {include_self}\n\n")

        f.write("---\n\n")
        f.write("## What is being tested\n\n")
        f.write(
            "Does LLaMA-2-13B-Chat's internal representational geometry over "
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
            "last-token residual-stream activation from LLaMA-2-13B-Chat in "
            "response to a simple prompt (\"Think about {entity}.\"). We compute "
            "pairwise **cosine distance** (1 - cosine similarity) between all "
            f"entity pairs -> a {n_entities}x{n_entities} representational "
            "dissimilarity matrix (RDM). Computed separately at each of 41 layers "
            "(layer 0 = embedding, layers 1-40 = transformer blocks).\n\n"
            "- **Human RDM**: From Gray et al. (2007) Experience and Agency factor "
            f"scores, we compute pairwise **Euclidean distance** between all entity "
            f"pairs in 2D (Experience, Agency) space -> a {n_entities}x{n_entities} "
            "RDM.\n\n"
            "**Test:** Spearman rank correlation between the upper triangles of the "
            f"model RDM and human RDM ({n_pairs} unique entity pairs). Computed at "
            "every layer to track where in the network mind-perception structure "
            "emerges.\n\n"
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

        f.write("## Results: RSA by Layer\n\n")
        f.write(f"**Peak:** Layer {peak['layer']}, "
                f"rho = {peak['rho']:+.4f}, p = {peak['p_value']:.4f}\n\n")
        f.write("| Layer | Spearman rho | p-value |\n")
        f.write("|------:|-------------:|--------:|\n")
        for r in rsa_results:
            if np.isnan(r["rho"]):
                f.write(f"| {r['layer']:5d} |          nan |     nan |\n")
            else:
                f.write(f"| {r['layer']:5d} | {r['rho']:+12.4f} | {r['p_value']:7.4f} |\n")
        f.write("\n")

        f.write("## Notes\n\n")
        f.write(
            "- Factor scores from Gray et al. (2007) were estimated by AI from "
            "Figure 1 and should be verified before final analyses.\n"
            "- One prompt per entity — no averaging across prompt variants. "
            "Reliability cannot be assessed with this design; robustness can be "
            "checked by rerunning with a different template.\n"
            "- Layer 0 (embedding) typically produces constant cosine distances "
            "(all activations have similar geometry before any transformer "
            "processing), yielding NaN correlation.\n"
        )

    print(f"  Results summary: {summary_path}")
    return summary_path


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4 Phase 1: Extract entity representations"
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

    # Load model
    print("\nLoading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
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

    print("Computing human RDM (Euclidean in Experience/Agency space)...")
    human_rdm = compute_human_rdm(entity_keys)

    # RSA at every layer
    print("\nComputing RSA (Spearman) at all layers...")
    rsa_results = compute_rsa_all_layers(model_rdm, human_rdm, n_entities)

    # Print full RSA table
    print(f"\nRSA results ({n_entities} entities, "
          f"{n_entities * (n_entities - 1) // 2} pairs):")
    for r in rsa_results:
        if np.isnan(r["rho"]):
            print(f"  Layer {r['layer']:2d}: rho =    nan")
        else:
            print(f"  Layer {r['layer']:2d}: rho = {r['rho']:+.4f}  "
                  f"(p = {r['p_value']:.4f})")

    # Peak
    valid = [r for r in rsa_results if not np.isnan(r["rho"])]
    if valid:
        peak = max(valid, key=lambda r: r["rho"])
        print(f"\n  Peak: Layer {peak['layer']}, rho = {peak['rho']:+.4f}, "
              f"p = {peak['p_value']:.4f}")

    # ── Save data ──
    np.savez_compressed(
        os.path.join(data_dir, "all_entity_activations.npz"),
        activations=acts_array,
        entity_keys=np.array(entity_keys),
    )
    np.savez_compressed(
        os.path.join(data_dir, "rdm_cosine_per_layer.npz"),
        model_rdm=model_rdm,
        human_rdm=human_rdm,
        entity_keys=np.array(entity_keys),
    )
    with open(os.path.join(data_dir, "entity_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    # ── Save results ──
    with open(os.path.join(results_dir, "rsa_all_layers.json"), "w") as f:
        json.dump(rsa_results, f, indent=2)

    write_results_summary(results_dir, rsa_results, entity_keys, tag,
                          args.include_self)

    print(f"\nSaved data to {data_dir}/")
    print(f"Saved results to {results_dir}/")


if __name__ == "__main__":
    main()
