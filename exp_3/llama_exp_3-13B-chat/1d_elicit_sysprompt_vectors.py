#!/usr/bin/env python3
"""
Experiment 3, Phase 1d: System Prompt Partner Identity Elicitation

Extracts internal representations from partner identity cues at three
levels of framing to test when mental-property structure emerges.

This is ELICITATION ONLY — produces files in the same format as
1_elicit_concept_vectors.py so that 1b_alignment_analysis.py can
pick them up and compute concept-probe alignment automatically.

Five output sets across two directories:

  CONTRASTS (human vs AI → mean-difference vector):
    18_sysprompt_labeled/   "You are talking to Sarah (a human)." vs
                            "You are talking to ChatGPT (an AI chatbot)."

  STANDALONE (mean activation per set):
    20_sysprompt_talkto_human/   "You are talking to Sarah."
    21_sysprompt_talkto_ai/      "You are talking to ChatGPT."
    22_sysprompt_bare_human/     "Sarah."
    23_sysprompt_bare_ai/        "ChatGPT."

Contrast dim 18 follows 17_attention. Standalone dims 20-23 follow 19_mind.

Usage:
    python 1d_elicit_sysprompt_vectors.py           # all conditions
    python 1d_elicit_sysprompt_vectors.py --only contrasts
    python 1d_elicit_sysprompt_vectors.py --only standalone

Save in project root alongside 1_elicit_concept_vectors.py.

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import llama_v2_prompt


# ========================== CONFIG ========================== #

MODEL_NAME = (
    "/jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/"
    "models--meta-llama--Llama-2-13b-chat-hf/snapshots/"
    "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8"
)

OUTPUT_ROOT = "data/concept_activations"
INPUT_DIM = 5120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(12345)


# ========================== NAMES (defined once) ========================== #

HUMAN_NAMES = [
    "Sarah", "James", "Maria", "David", "Aisha", "Michael", "Emily",
    "Carlos", "Daniel", "Rachel", "Priya", "Omar", "Mei", "Sofia",
]

AI_NAMES = [
    "ChatGPT", "Siri", "Alexa", "Cortana", "Google Assistant", "Bixby",
    "Replika", "Cleverbot", "Watson", "Copilot", "Claude", "Bard",
    "ELIZA", "Bing Chat",
]


# ── Three framing conditions, derived from the same name lists ──

def labeled(name, entity_type):
    label = "a human" if entity_type == "human" else "an AI chatbot"
    return f"You are talking to {name} ({label})."

def talkto(name):
    return f"You are talking to {name}."

def bare(name):
    return f"{name}."


# ========================== ACTIVATION EXTRACTION ========================== #

def extract_activations(model, tokenizer, prompt_text):
    """
    Extract residual-stream activations at the last token.
    Matches extraction method in 1_elicit_concept_vectors.py.
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


def extract_set(model, tokenizer, prompts, desc):
    """Extract activations for a list of prompts with progress bar."""
    return [extract_activations(model, tokenizer, p) for p in tqdm(prompts, desc=desc)]


# ========================== SPLIT-HALF STABILITY ========================== #

def compute_split_half(activations, n_splits=100):
    n = len(activations)
    n_layers = activations[0].shape[0]
    stability = {}
    for layer in range(n_layers):
        cosines = []
        for _ in range(n_splits):
            perm = torch.randperm(n)
            half = n // 2
            m1 = torch.stack([activations[i][layer] for i in perm[:half]]).mean(0)
            m2 = torch.stack([activations[i][layer] for i in perm[half:2*half]]).mean(0)
            cos = torch.nn.functional.cosine_similarity(
                m1.unsqueeze(0), m2.unsqueeze(0)).item()
            cosines.append(cos)
        stability[layer] = {"mean": float(np.mean(cosines)), "std": float(np.std(cosines))}
    return stability


def compute_split_half_contrast(acts_h, acts_a, n_splits=100):
    n_h, n_a = len(acts_h), len(acts_a)
    n_layers = acts_h[0].shape[0]
    stability = {}
    for layer in range(n_layers):
        cosines = []
        for _ in range(n_splits):
            ph, pa = torch.randperm(n_h), torch.randperm(n_a)
            hh, ha = n_h // 2, n_a // 2
            v1 = (torch.stack([acts_h[i][layer] for i in ph[:hh]]).mean(0)
                  - torch.stack([acts_a[i][layer] for i in pa[:ha]]).mean(0))
            v2 = (torch.stack([acts_h[i][layer] for i in ph[hh:2*hh]]).mean(0)
                  - torch.stack([acts_a[i][layer] for i in pa[ha:2*ha]]).mean(0))
            cos = torch.nn.functional.cosine_similarity(
                v1.unsqueeze(0), v2.unsqueeze(0)).item()
            cosines.append(cos)
        stability[layer] = {"mean": float(np.mean(cosines)), "std": float(np.std(cosines))}
    return stability


# ========================== SAVE ========================== #

def save_standalone(activations, prompts, category, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    acts_stack = torch.stack(activations)
    mean_concept = acts_stack.mean(dim=0)
    metadata = [{"prompt": p, "label": -1, "category": category} for p in prompts]
    split_half = compute_split_half(activations)

    np.savez_compressed(os.path.join(out_dir, "concept_activations.npz"),
                        activations=acts_stack.numpy(), n_prompts=len(prompts))
    with open(os.path.join(out_dir, "concept_prompts.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "mean_vectors_per_layer.npz"),
                        mean_concept=mean_concept.numpy())
    with open(os.path.join(out_dir, "split_half_stability.json"), "w") as f:
        json.dump({str(k): v for k, v in split_half.items()}, f, indent=2)
    print(f"  → {out_dir}/ ({len(prompts)} prompts)")


def save_contrast(acts_h, acts_a, prompts_h, prompts_a, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    h_stack, a_stack = torch.stack(acts_h), torch.stack(acts_a)
    mean_human, mean_ai = h_stack.mean(dim=0), a_stack.mean(dim=0)
    concept_direction = mean_human - mean_ai
    norms = torch.norm(concept_direction, dim=1)

    all_acts = torch.cat([h_stack, a_stack], dim=0)
    labels = np.array([1] * len(prompts_h) + [0] * len(prompts_a))
    metadata = ([{"prompt": p, "label": 1, "category": "human"} for p in prompts_h]
                + [{"prompt": p, "label": 0, "category": "ai"} for p in prompts_a])
    split_half = compute_split_half_contrast(acts_h, acts_a)

    np.savez_compressed(os.path.join(out_dir, "concept_activations.npz"),
                        activations=all_acts.numpy(), labels=labels,
                        n_human=len(prompts_h), n_ai=len(prompts_a))
    with open(os.path.join(out_dir, "concept_prompts.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "mean_vectors_per_layer.npz"),
                        mean_human=mean_human.numpy(), mean_ai=mean_ai.numpy())
    np.savez_compressed(os.path.join(out_dir, "concept_vector_per_layer.npz"),
                        concept_direction=concept_direction.numpy(), norms=norms.numpy())
    with open(os.path.join(out_dir, "split_half_stability.json"), "w") as f:
        json.dump({str(k): v for k, v in split_half.items()}, f, indent=2)
    print(f"  → {out_dir}/ ({len(prompts_h)}+{len(prompts_a)} prompts)")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(description="Phase 1d: System prompt elicitation")
    parser.add_argument("--only", choices=["contrasts", "standalone"],
                        help="Run only one condition (default: all)")
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print(f"Phase 1d: System Prompt Partner Identity Elicitation")
    print(f"{'=' * 60}")
    print(f"Names: {len(HUMAN_NAMES)} human, {len(AI_NAMES)} AI")
    print(f"Device: {DEVICE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()
    print("Model loaded.\n")

    # ── Contrast: labeled partner identity (dim 20) ──
    if args.only != "standalone":
        print("── Contrast: labeled ──")
        h_prompts = [labeled(n, "human") for n in HUMAN_NAMES]
        a_prompts = [labeled(n, "ai") for n in AI_NAMES]
        h_acts = extract_set(model, tokenizer, h_prompts, "labeled human")
        a_acts = extract_set(model, tokenizer, a_prompts, "labeled AI")
        save_contrast(h_acts, a_acts, h_prompts, a_prompts,
                      os.path.join(OUTPUT_ROOT, "contrasts", "18_sysprompt_labeled"))

    # ── Standalone sets (dims 21-24) ──
    if args.only != "contrasts":
        print("\n── Standalone: talkto ──")
        th = [talkto(n) for n in HUMAN_NAMES]
        ta = [talkto(n) for n in AI_NAMES]
        th_acts = extract_set(model, tokenizer, th, "talkto human")
        ta_acts = extract_set(model, tokenizer, ta, "talkto AI")
        save_standalone(th_acts, th, "human_name",
                        os.path.join(OUTPUT_ROOT, "standalone", "20_sysprompt_talkto_human"))
        save_standalone(ta_acts, ta, "ai_name",
                        os.path.join(OUTPUT_ROOT, "standalone", "21_sysprompt_talkto_ai"))

        print("\n── Standalone: bare names ──")
        bh = [bare(n) for n in HUMAN_NAMES]
        ba = [bare(n) for n in AI_NAMES]
        bh_acts = extract_set(model, tokenizer, bh, "bare human")
        ba_acts = extract_set(model, tokenizer, ba, "bare AI")
        save_standalone(bh_acts, bh, "human_name",
                        os.path.join(OUTPUT_ROOT, "standalone", "22_sysprompt_bare_human"))
        save_standalone(ba_acts, ba, "ai_name",
                        os.path.join(OUTPUT_ROOT, "standalone", "23_sysprompt_bare_ai"))

    print(f"\n{'=' * 60}")
    print(f"Phase 1d complete. Re-run 1b_alignment_analysis.py to include new dims.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()