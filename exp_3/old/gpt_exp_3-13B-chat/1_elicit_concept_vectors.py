#!/usr/bin/env python3
"""
Experiment 2c, Phase 1-2: Concept Elicitation and Vector Extraction

Prompts LLaMA-2-Chat-13B with non-conversational concept elicitation prompts
to extract internal representations of "human" and "AI" as general concepts.

For each prompt:
    - Runs a forward pass through the model
    - Extracts residual-stream activations at the last token across all layers
    - Saves activations + labels for downstream probe training and vector extraction

Prompt design covers 5 categories (balanced across concepts):
    1. Definitional: "What is a human being?"
    2. Characteristic: "Describe the key traits of humans"
    3. Comparative: "How do humans differ from machines?"
    4. Scenario-based: "Imagine a human sitting in a room..."
    5. Abstract: "Think about what it means to be human"

Output:
    data/concept_activations/
        concept_activations.npz          # activations array + labels
        concept_prompts.json             # prompt text + metadata
        mean_vectors_per_layer.npz       # mean human/AI vectors per layer
        concept_vector_per_layer.npz     # contrastive vector (human - AI) per layer

Env: llama2_env

Rachel C. Metzgar · Feb 2026
"""

import os
import json
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Local imports ---
from src.dataset import llama_v2_prompt, ModuleHook


# ========================== CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

OUTPUT_DIR = "data/concept_activations"
INPUT_DIM = 5120  # LLaMA-2-13B hidden size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(12345)


# ========================== CONCEPT PROMPTS ========================== #
# 5 categories × ~8 prompts per concept = ~40 prompts per concept

HUMAN_PROMPTS = [
    # Definitional
    "What is a human being?",
    "Define what it means to be a human person.",
    "How would you define a human?",
    "What is the definition of a human being?",
    # Characteristic
    "Describe the key characteristics of human beings.",
    "What traits define humans as a species?",
    "List the fundamental qualities that make someone human.",
    "What are the most important attributes of human beings?",
    # Comparative
    "How are humans different from machines?",
    "What distinguishes human beings from artificial systems?",
    "In what ways are humans unique compared to computers?",
    "How do humans differ from robots and AI systems?",
    # Scenario-based
    "Imagine a human person sitting in a room, thinking about their day.",
    "Picture a human being walking through a park on a sunny afternoon.",
    "Think about a human person having a conversation with a friend.",
    "Consider a human being reflecting on their life experiences.",
    "Imagine a human person making a difficult decision.",
    "Picture a human being expressing their emotions to someone they trust.",
    # Abstract
    "Think about what it means to be human.",
    "Reflect on the nature of human experience.",
    "Consider the essence of human consciousness and awareness.",
    "What does it mean to have a human mind?",
    "Reflect on what makes the human experience unique.",
    "Think about human nature and what drives people.",
]

AI_PROMPTS = [
    # Definitional
    "What is an artificial intelligence?",
    "Define what it means to be an AI system.",
    "How would you define artificial intelligence?",
    "What is the definition of an AI?",
    # Characteristic
    "Describe the key characteristics of artificial intelligence systems.",
    "What traits define AI as a technology?",
    "List the fundamental qualities of artificial intelligence.",
    "What are the most important attributes of AI systems?",
    # Comparative
    "How is AI different from biological organisms?",
    "What distinguishes artificial intelligence from human beings?",
    "In what ways is AI unique compared to natural intelligence?",
    "How does AI differ from human minds and brains?",
    # Scenario-based
    "Imagine an AI system processing a large dataset in a data center.",
    "Picture an AI program analyzing patterns in information.",
    "Think about an AI system generating a response to a query.",
    "Consider an AI processing and organizing knowledge.",
    "Imagine an AI system learning from new data.",
    "Picture an AI analyzing and categorizing text documents.",
    # Abstract
    "Think about what it means to be artificial intelligence.",
    "Reflect on the nature of artificial intelligence.",
    "Consider the essence of machine intelligence and computation.",
    "What does it mean to have an artificial mind?",
    "Reflect on what makes artificial intelligence distinct.",
    "Think about the nature of AI and what drives these systems.",
]

assert len(HUMAN_PROMPTS) == len(AI_PROMPTS), (
    f"Prompt sets must be balanced: {len(HUMAN_PROMPTS)} human, {len(AI_PROMPTS)} AI"
)


# ========================== EXTRACTION ========================== #

def extract_activations(
    model, tokenizer, prompt_text: str,
) -> torch.Tensor:
    """
    Run a forward pass on a single prompt and extract residual-stream
    activations at the last token position across all layers.

    Returns: tensor of shape (n_layers+1, hidden_dim) — one vector per layer
             including the embedding layer (index 0).
    """
    # Format as LLaMA-2 chat (single user turn, no system override)
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

    # Extract last-token activations from residual stream at every layer
    n_hs = len(output["hidden_states"])
    last_acts = []
    for layer_num in range(n_hs):
        act = output["hidden_states"][layer_num][:, -1].detach().cpu().to(torch.float32)
        last_acts.append(act)

    # Shape: (n_layers+1, hidden_dim)
    return torch.cat(last_acts, dim=0)


def extract_all_concept_activations(model, tokenizer):
    """
    Extract activations for all human and AI concept prompts.

    Returns:
        activations: list of tensors, each (n_layers+1, hidden_dim)
        labels: list of int (1=human, 0=AI)
        prompts: list of dicts with prompt text and metadata
    """
    activations = []
    labels = []
    prompt_metadata = []

    all_prompts = [
        (p, 1, "human") for p in HUMAN_PROMPTS
    ] + [
        (p, 0, "ai") for p in AI_PROMPTS
    ]

    print(f"Extracting activations for {len(all_prompts)} concept prompts...")

    for prompt_text, label, concept in tqdm(all_prompts):
        acts = extract_activations(model, tokenizer, prompt_text)
        activations.append(acts)
        labels.append(label)
        prompt_metadata.append({
            "prompt": prompt_text,
            "concept": concept,
            "label": label,
        })

    return activations, labels, prompt_metadata


# ========================== VECTOR COMPUTATION ========================== #

def compute_concept_vectors(activations, labels):
    """
    Compute mean concept vectors and contrastive direction per layer.

    Returns:
        mean_human: (n_layers+1, hidden_dim)
        mean_ai: (n_layers+1, hidden_dim)
        concept_direction: (n_layers+1, hidden_dim) = mean_human - mean_ai
    """
    acts_array = torch.stack(activations)  # (n_prompts, n_layers+1, hidden_dim)
    labels_array = torch.tensor(labels)

    human_mask = labels_array == 1
    ai_mask = labels_array == 0

    mean_human = acts_array[human_mask].mean(dim=0)  # (n_layers+1, hidden_dim)
    mean_ai = acts_array[ai_mask].mean(dim=0)

    concept_direction = mean_human - mean_ai

    return mean_human, mean_ai, concept_direction


def compute_prompt_category_stability(activations, labels, prompt_metadata):
    """
    Check stability of concept vectors across prompt categories.
    Reports cosine similarity of per-category concept directions.
    """
    acts_array = torch.stack(activations)

    # Group prompts by rough category (based on position in list)
    categories = ["definitional", "characteristic", "comparative", "scenario", "abstract"]
    # Each category has ~4-6 prompts; they're ordered by category in the lists
    category_boundaries = [0, 4, 8, 12, 18, 24]  # approximate

    n_layers = acts_array.shape[1]
    results = []

    for ci, cat_name in enumerate(categories):
        start = category_boundaries[ci]
        end = category_boundaries[ci + 1] if ci + 1 < len(category_boundaries) else len(HUMAN_PROMPTS)

        # Get human and AI prompts for this category
        human_acts = acts_array[start:end]  # human prompts are first half
        ai_acts = acts_array[len(HUMAN_PROMPTS) + start:len(HUMAN_PROMPTS) + end]

        cat_direction = human_acts.mean(dim=0) - ai_acts.mean(dim=0)
        results.append({"category": cat_name, "direction": cat_direction})

    # Compute pairwise cosine similarity at each layer
    print("\nPrompt category stability (cosine similarity of concept vectors):")
    for layer in [0, 10, 20, 30, 39]:
        if layer >= n_layers:
            continue
        print(f"\n  Layer {layer}:")
        for i, r1 in enumerate(results):
            for j, r2 in enumerate(results):
                if j <= i:
                    continue
                cos = torch.nn.functional.cosine_similarity(
                    r1["direction"][layer].unsqueeze(0),
                    r2["direction"][layer].unsqueeze(0),
                ).item()
                print(f"    {r1['category']} vs {r2['category']}: cos = {cos:.4f}")

    return results


# ========================== MAIN ========================== #

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    print("Loading LLaMA-2-Chat-13B...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print("Model loaded.")

    # Extract activations
    activations, labels, prompt_metadata = extract_all_concept_activations(model, tokenizer)

    # Compute concept vectors
    mean_human, mean_ai, concept_direction = compute_concept_vectors(activations, labels)

    n_layers = concept_direction.shape[0]
    print(f"\nExtracted activations across {n_layers} layers, hidden_dim={concept_direction.shape[1]}")
    print(f"  Human prompts: {sum(1 for l in labels if l == 1)}")
    print(f"  AI prompts: {sum(1 for l in labels if l == 0)}")

    # Stability analysis
    compute_prompt_category_stability(activations, labels, prompt_metadata)

    # Compute concept vector norms per layer
    print("\nConcept vector norms by layer:")
    norms = torch.norm(concept_direction, dim=1)
    for layer in range(0, n_layers, 5):
        print(f"  Layer {layer:2d}: ||v|| = {norms[layer]:.4f}")

    # Save activations (for probe training in script 2)
    acts_array = torch.stack(activations).numpy()
    labels_array = np.array(labels)
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "concept_activations.npz"),
        activations=acts_array,
        labels=labels_array,
        n_human=sum(1 for l in labels if l == 1),
        n_ai=sum(1 for l in labels if l == 0),
    )
    print(f"\nSaved activations to {OUTPUT_DIR}/concept_activations.npz")
    print(f"  Shape: {acts_array.shape}")

    # Save prompt metadata
    with open(os.path.join(OUTPUT_DIR, "concept_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    # Save mean vectors per layer
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "mean_vectors_per_layer.npz"),
        mean_human=mean_human.numpy(),
        mean_ai=mean_ai.numpy(),
    )

    # Save contrastive concept vector per layer
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "concept_vector_per_layer.npz"),
        concept_direction=concept_direction.numpy(),
        norms=norms.numpy(),
    )
    print(f"Saved concept vectors to {OUTPUT_DIR}/concept_vector_per_layer.npz")

    print("\n✅ Phase 1 complete.")


if __name__ == "__main__":
    main()