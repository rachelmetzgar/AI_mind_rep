#!/usr/bin/env python3
"""
Experiment 3, Phase 9a: Extract Conversation Activations

Runs LLaMA-2-13B-Chat forward on each conversation from Exp 1 data and
caches the last-token hidden states (operational position, no suffix)
across all 41 layers.

These cached activations are used by 9b_concept_conversation_alignment.py
to compare standalone concept vectors against actual conversation
representations.

Output:
    results/llama2_13b_chat/{version}/conversation_activations/turn_{turn}/
        activations.npz    (n_conversations, 41, 5120) float16
        metadata.csv       conv_idx, condition, subject, topic, partner_name

Usage:
    python 9a_extract_conversation_activations.py --version balanced_gpt
    python 9a_extract_conversation_activations.py --version balanced_gpt --turn 3

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import csv
import json
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add exp_3/code to path for config
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    config, set_version, set_model,
    add_version_argument, add_model_argument, add_turn_argument,
    ensure_dir, get_model,
)

# Add exp_2/code/src to path for llama_v2_prompt
sys.path.insert(0, os.path.join(config.PROJECT_ROOT, "exp_2", "code", "src"))
from dataset import llama_v2_prompt


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract conversation activations for concept-conversation alignment."
    )
    add_version_argument(parser)
    add_model_argument(parser)
    add_turn_argument(parser)
    parser.add_argument(
        "--max_length", type=int, default=2048,
        help="Max token length for truncation (default: 2048).",
    )
    return parser.parse_args()


# ============================================================
# CONVERSATION LOADING
# ============================================================

def load_conversations(csv_dir, turn_index):
    """Load conversations from per-subject CSVs.

    Returns list of dicts with keys: messages, condition, subject, topic, partner_name
    """
    import glob
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "s[0-9][0-9][0-9].csv")))
    if not csv_files:
        print(f"[ERROR] No subject CSVs found in {csv_dir}")
        return []

    print(f"Found {len(csv_files)} subject files in {csv_dir}")
    conversations = []

    for csv_path in csv_files:
        subject_id = os.path.basename(csv_path).replace(".csv", "")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Group rows by trial
        trials = {}
        for r in rows:
            t = int(r["trial"])
            if t not in trials:
                trials[t] = []
            trials[t].append(r)

        for trial_num in sorted(trials.keys()):
            trial_rows = trials[trial_num]

            # Select the desired turn
            if turn_index == -1 or turn_index is None:
                row = trial_rows[-1]
            else:
                # turn_index is 1-indexed (turn 1-5), trial rows are 0-indexed
                idx = turn_index - 1 if turn_index > 0 else turn_index
                if idx >= len(trial_rows):
                    continue
                row = trial_rows[idx]

            # Parse condition
            partner_type = row["partner_type"]
            if "AI" in partner_type or "ai" in partner_type.lower():
                condition = "ai"
            elif "Human" in partner_type or "human" in partner_type.lower():
                condition = "human"
            else:
                continue

            # Parse messages
            try:
                messages = json.loads(row["sub_input"])
            except (json.JSONDecodeError, KeyError):
                continue

            if len(messages) < 2:
                continue

            conversations.append({
                "messages": messages,
                "condition": condition,
                "subject": subject_id,
                "topic": row.get("topic", ""),
                "partner_name": row.get("partner_name", ""),
            })

    print(f"Loaded {len(conversations)} conversations "
          f"(human: {sum(1 for c in conversations if c['condition'] == 'human')}, "
          f"ai: {sum(1 for c in conversations if c['condition'] == 'ai')})")
    return conversations


# ============================================================
# ACTIVATION EXTRACTION
# ============================================================

def extract_activations(model, tokenizer, conversations, max_length=2048):
    """Run forward pass on each conversation, extract last-token hidden states.

    Returns:
        activations: np.ndarray, shape (n_convs, n_layers, hidden_dim), float16
        metadata: list of dicts
    """
    all_acts = []
    metadata = []

    for idx, conv in enumerate(tqdm(conversations, desc="Extracting activations")):
        # Format to LLaMA-2 chat (operational position — no suffix)
        try:
            text = llama_v2_prompt(conv["messages"])
        except Exception as e:
            print(f"  Skipping conv {idx}: format error: {e}")
            continue

        # Tokenize
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Forward pass
        with torch.no_grad():
            output = model(
                input_ids=encoding["input_ids"].to("cuda"),
                attention_mask=encoding["attention_mask"].to("cuda"),
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract last-token hidden state from each layer
        layer_acts = []
        for layer_hs in output["hidden_states"]:
            layer_acts.append(layer_hs[:, -1].detach().cpu().float().numpy())

        # Stack: (n_layers, hidden_dim)
        acts = np.concatenate(layer_acts, axis=0)  # (n_layers, hidden_dim)
        all_acts.append(acts.astype(np.float16))

        metadata.append({
            "conv_idx": len(metadata),
            "condition": conv["condition"],
            "subject": conv["subject"],
            "topic": conv["topic"],
            "partner_name": conv["partner_name"],
        })

        # Clear GPU cache periodically
        if idx % 50 == 0:
            torch.cuda.empty_cache()

    activations = np.stack(all_acts)  # (n_convs, n_layers, hidden_dim)
    return activations, metadata


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # Set model and version
    set_model(args.model)
    set_version(args.version, turn=args.turn)

    # Resolve conversation data path
    csv_dir = os.path.join(
        str(config.PROJECT_ROOT), "exp_1", "results",
        get_model(), args.version, "data",
    )
    if not os.path.isdir(csv_dir):
        print(f"[ERROR] Conversation data not found: {csv_dir}")
        sys.exit(1)

    # Output path
    out_dir = os.path.join(
        str(config.RESULTS.root), get_model(), args.version,
        "conversation_activations", f"turn_{args.turn}",
    )
    ensure_dir(out_dir)

    # Check if already done
    out_npz = os.path.join(out_dir, "activations.npz")
    if os.path.exists(out_npz):
        print(f"[SKIP] Activations already exist: {out_npz}")
        print("Delete the file to re-extract.")
        return

    # Load conversations
    conversations = load_conversations(csv_dir, turn_index=args.turn)
    if not conversations:
        print("[ERROR] No conversations loaded.")
        sys.exit(1)

    # Load model
    print(f"\nLoading model: {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Extract activations
    print(f"\nExtracting activations ({len(conversations)} conversations)...")
    activations, metadata = extract_activations(
        model, tokenizer, conversations, max_length=args.max_length,
    )

    print(f"\nActivations shape: {activations.shape}")  # (n_convs, 41, 5120)
    print(f"  dtype: {activations.dtype}")
    print(f"  size: {activations.nbytes / 1e6:.0f} MB")

    # Save
    np.savez_compressed(out_npz, activations=activations)
    print(f"Saved: {out_npz}")

    import pandas as pd
    meta_df = pd.DataFrame(metadata)
    meta_path = os.path.join(out_dir, "metadata.csv")
    meta_df.to_csv(meta_path, index=False)
    print(f"Saved: {meta_path}")
    print(f"  Conditions: {meta_df['condition'].value_counts().to_dict()}")

    print(f"\nExtraction complete.")


if __name__ == "__main__":
    main()
