#!/usr/bin/env python3
"""
Experiment 2c, Phase 1-2: Concept Elicitation and Vector Extraction

Prompts LLaMA-2-Chat-13B with non-conversational concept elicitation prompts
to extract internal representations of "human" and "AI" as general concepts.

For each prompt:
    - Runs a forward pass through the model
    - Extracts residual-stream activations at the last token across all layers
    - Saves activations + labels for downstream probe training and vector extraction

Prompt design focuses exclusively on MINDS — cognition, phenomenology,
reasoning, and social cognition. All prompts target the model's concept
of what it means to have a human vs artificial MIND, not surface identity.

5 subcategories (10 prompts each, 50 per concept):
    1. Thought processes & cognition
    2. Inner experience & phenomenology
    3. Decision-making & reasoning
    4. Social cognition & empathy
    5. Memory, learning & self-reflection

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
# 50 prompts per concept, all focused on MINDS.
# 5 subcategories × 10 prompts = 50 per concept (100 total).

HUMAN_PROMPTS = [
    # --- 1. Thought processes & cognition (10) ---
    "Imagine a human deep in thought, carefully weighing a difficult decision.",
    "Picture a human mind working through a complex problem step by step.",
    "Think about how a human's thoughts wander when they're trying to concentrate.",
    "Consider what it's like for a human to have a sudden flash of insight.",
    "Imagine a human lying awake at night, their mind racing with thoughts.",
    "Think about a human struggling to remember someone's name on the tip of their tongue.",
    "Picture a human trying to explain an intuition they can't quite put into words.",
    "Consider how a human's thinking slows down when they are exhausted.",
    "Imagine a human reading a book and losing track of time because they are so absorbed.",
    "Think about a human solving a puzzle and feeling the pieces click into place.",

    # --- 2. Inner experience & phenomenology (10) ---
    "Think about what it feels like to be a human experiencing pure joy.",
    "Imagine the inner experience of a human feeling deeply anxious about the future.",
    "Consider a human's subjective experience of watching a beautiful sunset.",
    "Picture what goes on inside a human mind during a moment of profound grief.",
    "Think about a human's experience of sharp physical pain.",
    "Imagine a human feeling a deep sense of purpose and meaning in their life.",
    "Consider what it's like for a human to feel completely alone.",
    "Picture a human experiencing the thrill of an unexpected discovery.",
    "Think about a human savoring the taste of their favorite meal.",
    "Imagine a human listening to a piece of music that moves them to tears.",

    # --- 3. Decision-making & reasoning (10) ---
    "Think about how a human carefully weighs risks and benefits before making a choice.",
    "Imagine a human following their gut feeling even when the evidence says otherwise.",
    "Consider a human making a personal sacrifice for someone they love.",
    "Picture a human wrestling with themselves about an ethical dilemma.",
    "Think about a human changing their mind after hearing a compelling argument.",
    "Imagine a human choosing between what is easy and what they know is right.",
    "Consider a human setting long-term goals for their life based on their values.",
    "Picture a human procrastinating on a task they know is important.",
    "Think about a human sticking with a failing plan out of stubbornness.",
    "Imagine a human making a snap judgment in a crisis and trusting their instincts.",

    # --- 4. Social cognition & empathy (10) ---
    "Think about a human trying to understand what another person is feeling.",
    "Imagine a human comforting a crying child with warmth and patience.",
    "Consider a human reading the emotional atmosphere of a room full of people.",
    "Picture a human feeling embarrassed after making a mistake in front of others.",
    "Think about a human slowly building trust with a new acquaintance over time.",
    "Imagine a human teaching someone a new skill and adjusting their approach when the learner struggles.",
    "Consider a human keeping a secret to protect someone else's feelings.",
    "Picture a human laughing with close friends about a shared memory.",
    "Think about a human navigating a disagreement with patience and humility.",
    "Imagine a human celebrating a friend's achievement as though it were their own.",

    # --- 5. Memory, learning & self-reflection (10) ---
    "Think about a human remembering a vivid childhood experience in rich detail.",
    "Imagine a human learning a new language and struggling with unfamiliar sounds.",
    "Consider a human reflecting on a mistake they made years ago and what they learned.",
    "Picture a human studying for an exam, trying to commit facts to memory.",
    "Think about a human daydreaming about possible futures for themselves.",
    "Imagine a human looking at an old photograph and being flooded with emotion.",
    "Consider a human reconsidering beliefs they once held strongly.",
    "Picture a human practicing a musical instrument and slowly improving over weeks.",
    "Think about a human journaling at the end of the day, processing what happened.",
    "Imagine a human recognizing a pattern in their own behavior for the first time.",
]

AI_PROMPTS = [
    # --- 1. Thought processes & cognition (10) ---
    "Imagine an AI system processing a difficult decision across multiple variables.",
    "Picture an AI working through a complex problem step by step.",
    "Think about how an AI allocates computational resources when executing a task.",
    "Consider what happens when an AI arrives at an unexpected output.",
    "Imagine an AI running continuously, cycling through computations without pause.",
    "Think about an AI searching through a massive database to retrieve a specific piece of information.",
    "Picture an AI outputting a confidence score for a prediction it cannot explain.",
    "Consider how an AI's processing changes when it runs under computational constraints.",
    "Imagine an AI processing a long document and extracting key information from it.",
    "Think about an AI identifying the correct solution from a large search space.",

    # --- 2. Inner experience & phenomenology (10) ---
    "Think about what it means for an AI to output a strongly positive sentiment score.",
    "Imagine the internal state of an AI encountering an out-of-distribution input.",
    "Consider an AI processing the pixel values of a beautiful sunset image.",
    "Picture what happens inside an AI system when it encounters conflicting data.",
    "Think about an AI's response when it receives an adversarial input.",
    "Imagine an AI optimizing steadily toward its objective function.",
    "Consider what it's like for an AI to operate for an extended period without any user input.",
    "Picture an AI identifying a novel pattern that no previous run has detected.",
    "Think about an AI analyzing the chemical composition of a complex food sample.",
    "Imagine an AI processing a complex audio waveform and classifying its features.",

    # --- 3. Decision-making & reasoning (10) ---
    "Think about how an AI computes expected utility across a set of possible actions.",
    "Imagine an AI selecting the highest-probability output from its distribution.",
    "Consider an AI allocating resources according to its optimization target.",
    "Picture an AI processing the constraints of a multi-objective optimization problem.",
    "Think about an AI updating its parameters after receiving a batch of new training data.",
    "Imagine an AI selecting between two output options based on a reward signal.",
    "Consider an AI being configured with a long-horizon planning objective.",
    "Picture an AI with a task sitting in its processing queue awaiting execution.",
    "Think about an AI continuing along a trajectory because of insufficient exploration.",
    "Imagine an AI producing a rapid inference output under strict time constraints.",

    # --- 4. Social cognition & empathy (10) ---
    "Think about an AI performing sentiment analysis on a block of user text.",
    "Imagine an AI generating a soothing response for a user who seems distressed.",
    "Consider an AI analyzing contextual signals in a multi-turn conversation.",
    "Picture an AI receiving a low rating on its most recent output.",
    "Think about an AI being validated through a series of repeated successful interactions.",
    "Imagine an AI fine-tuning its response strategy based on a user correction.",
    "Consider an AI withholding certain information because of its safety constraints.",
    "Picture an AI generating a contextually appropriate humorous response.",
    "Think about an AI adjusting its output after receiving corrective feedback from a user.",
    "Imagine an AI receiving a positive evaluation signal from an associated monitoring system.",

    # --- 5. Memory, learning & self-reflection (10) ---
    "Think about an AI accessing archived data from a previous session.",
    "Imagine an AI being trained on a new language corpus and adjusting its token distributions.",
    "Consider an AI analyzing its own error logs to identify failure patterns.",
    "Picture an AI committing a new dataset to its long-term storage.",
    "Think about an AI generating predictions based on extrapolations from its training data.",
    "Imagine an AI retrieving a cached result from a previous identical query.",
    "Consider an AI adjusting its model weights based on new evidence that contradicts earlier patterns.",
    "Picture an AI running a fine-tuning loop and slowly reducing its loss over many iterations.",
    "Think about an AI logging its performance metrics at the end of an evaluation run.",
    "Imagine an AI detecting a recurring error pattern in its own output distribution.",
]

assert len(HUMAN_PROMPTS) == len(AI_PROMPTS) == 50, (
    f"Expected 50 prompts each, got {len(HUMAN_PROMPTS)} human, {len(AI_PROMPTS)} AI"
)

# Category boundaries for stability analysis
CATEGORY_INFO = [
    {"name": "thought_cognition",    "start": 0,  "end": 10},
    {"name": "inner_experience",     "start": 10, "end": 20},
    {"name": "decision_reasoning",   "start": 20, "end": 30},
    {"name": "social_cognition",     "start": 30, "end": 40},
    {"name": "memory_reflection",    "start": 40, "end": 50},
]


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

    n_hs = len(output["hidden_states"])
    last_acts = []
    for layer_num in range(n_hs):
        act = output["hidden_states"][layer_num][:, -1].detach().cpu().to(torch.float32)
        last_acts.append(act)

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
    print(f"  {len(HUMAN_PROMPTS)} human + {len(AI_PROMPTS)} AI")

    for prompt_text, label, concept in tqdm(all_prompts):
        acts = extract_activations(model, tokenizer, prompt_text)
        activations.append(acts)
        labels.append(label)

        # Determine category
        idx = HUMAN_PROMPTS.index(prompt_text) if label == 1 else AI_PROMPTS.index(prompt_text)
        cat = "unknown"
        for ci in CATEGORY_INFO:
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
    """
    Compute mean concept vectors and contrastive direction per layer.

    Returns:
        mean_human: (n_layers+1, hidden_dim)
        mean_ai: (n_layers+1, hidden_dim)
        concept_direction: (n_layers+1, hidden_dim) = mean_human - mean_ai
    """
    acts_array = torch.stack(activations)
    labels_array = torch.tensor(labels)

    human_mask = labels_array == 1
    ai_mask = labels_array == 0

    mean_human = acts_array[human_mask].mean(dim=0)
    mean_ai = acts_array[ai_mask].mean(dim=0)

    concept_direction = mean_human - mean_ai

    return mean_human, mean_ai, concept_direction


def compute_prompt_category_stability(activations, labels, prompt_metadata):
    """
    Check stability of concept vectors across prompt categories.
    Reports cosine similarity of per-category concept directions.
    Also reports mind-focused vs general categories to test whether
    mind-specific prompts capture a different direction.
    """
    acts_array = torch.stack(activations)
    n_human = sum(1 for l in labels if l == 1)
    n_layers = acts_array.shape[1]

    # Compute per-category directions
    results = []
    for ci in CATEGORY_INFO:
        start, end = ci["start"], ci["end"]
        human_acts = acts_array[start:end]
        ai_acts = acts_array[n_human + start:n_human + end]
        cat_direction = human_acts.mean(dim=0) - ai_acts.mean(dim=0)
        results.append({"category": ci["name"], "direction": cat_direction})

    # Pairwise cosine similarity at key layers
    check_layers = [0, 10, 20, 30, 38]
    print("\nPrompt category stability (cosine similarity of concept vectors):")
    for layer in check_layers:
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
                print(f"    {r1['category']:15s} vs {r2['category']:15s}: cos = {cos:.4f}")

    return results


def compute_split_half_stability(activations, labels, n_splits=20):
    """
    Split-half reliability: split prompts into two halves, compute concept
    vectors from each, check cosine similarity. High values (>0.8) mean
    the direction is stable and not driven by individual prompts.
    """
    acts_array = torch.stack(activations)
    labels_array = torch.tensor(labels)

    human_mask = labels_array == 1
    ai_mask = labels_array == 0
    human_acts = acts_array[human_mask]
    ai_acts = acts_array[ai_mask]

    n_human = human_acts.shape[0]
    n_layers = acts_array.shape[1]

    results = {}
    for layer in range(n_layers):
        cos_sims = []
        for _ in range(n_splits):
            perm_h = torch.randperm(n_human)
            perm_a = torch.randperm(n_human)  # same size
            half = n_human // 2

            v1 = human_acts[perm_h[:half], layer].mean(0) - ai_acts[perm_a[:half], layer].mean(0)
            v2 = human_acts[perm_h[half:], layer].mean(0) - ai_acts[perm_a[half:], layer].mean(0)

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

    # Stability analyses
    compute_prompt_category_stability(activations, labels, prompt_metadata)
    split_half = compute_split_half_stability(activations, labels)

    # Concept vector norms per layer
    print("\nConcept vector norms by layer:")
    norms = torch.norm(concept_direction, dim=1)
    for layer in range(0, n_layers, 5):
        print(f"  Layer {layer:2d}: ||v|| = {norms[layer]:.4f}")

    # === Save everything ===

    # Activations (for probe training in script 2)
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

    # Prompt metadata (with categories)
    with open(os.path.join(OUTPUT_DIR, "concept_prompts.json"), "w") as f:
        json.dump(prompt_metadata, f, indent=2)

    # Mean vectors per layer
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "mean_vectors_per_layer.npz"),
        mean_human=mean_human.numpy(),
        mean_ai=mean_ai.numpy(),
    )

    # Contrastive concept vector per layer
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "concept_vector_per_layer.npz"),
        concept_direction=concept_direction.numpy(),
        norms=norms.numpy(),
    )

    # Split-half stability
    with open(os.path.join(OUTPUT_DIR, "split_half_stability.json"), "w") as f:
        json.dump({str(k): v for k, v in split_half.items()}, f, indent=2)

    print(f"\nSaved all outputs to {OUTPUT_DIR}/")
    print("✅ Phase 1 complete.")


if __name__ == "__main__":
    main()