#!/usr/bin/env python3
"""
Experiment 5, Phase 1: Extract Activations

Run all 336 sentences through LLaMA-2-13B-Chat and extract hidden state
activations at the last token position for all 41 layers.

Output:
    results/{model}/activations/data/
        activations_last_token.npz   — (336, 41, 5120) float16
        stimuli_metadata.csv         — item_id, condition, category, sentence, n_tokens

Usage:
    python code/rsa/1_extract_activations.py --model llama2_13b_chat

SLURM:
    sbatch code/rsa/slurm/1_extract_activations.sh

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    set_model, add_model_argument, model_path, hidden_dim, n_layers,
    data_dir, ensure_dir, N_SENTENCES,
)
from stimuli import get_all_sentences


def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations for exp 5")
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    out_dir = ensure_dir(data_dir("activations"))
    out_npz = out_dir / "activations_last_token.npz"
    out_csv = out_dir / "stimuli_metadata.csv"

    if out_npz.exists():
        print(f"Output already exists: {out_npz}")
        print("Delete it to re-run. Exiting.")
        return

    # ── Load model ───────────────────────────────────────────────────────
    mpath = model_path()
    print(f"Loading tokenizer from {mpath}")
    tokenizer = AutoTokenizer.from_pretrained(mpath, local_files_only=True)

    print(f"Loading model from {mpath}")
    model = AutoModelForCausalLM.from_pretrained(
        mpath,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded. Device: {next(model.parameters()).device}")

    # ── Extract activations ──────────────────────────────────────────────
    sentences = get_all_sentences()
    assert len(sentences) == N_SENTENCES

    nl = n_layers()
    hd = hidden_dim()
    activations = np.zeros((N_SENTENCES, nl, hd), dtype=np.float16)
    metadata = []

    t0 = time.time()
    with torch.no_grad():
        for idx, (item_id, condition, category, sentence) in enumerate(sentences):
            encoding = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = encoding["input_ids"].to(model.device)
            n_tokens = input_ids.shape[1]

            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # outputs.hidden_states: tuple of (1, seq_len, hidden_dim) x (n_layers+1)
            # Index 0 = embedding layer, 1..40 = transformer blocks
            # We keep all 41 (embedding + 40 blocks)
            for layer_idx, hs in enumerate(outputs.hidden_states):
                if layer_idx >= nl:
                    break
                activations[idx, layer_idx, :] = (
                    hs[0, -1, :].cpu().float().numpy().astype(np.float16)
                )

            metadata.append({
                "idx": idx,
                "item_id": item_id,
                "condition": condition,
                "category": category,
                "sentence": sentence,
                "n_tokens": n_tokens,
            })

            if (idx + 1) % 56 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1:3d}/{N_SENTENCES}] {elapsed:.1f}s")

            # Free GPU memory
            del outputs, input_ids, encoding

    elapsed = time.time() - t0
    print(f"Extraction complete: {N_SENTENCES} sentences in {elapsed:.1f}s")

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"Saving activations to {out_npz}")
    np.savez_compressed(out_npz, activations=activations)
    print(f"  Shape: {activations.shape}, dtype: {activations.dtype}")
    print(f"  File size: {out_npz.stat().st_size / 1e6:.1f} MB")

    print(f"Saving metadata to {out_csv}")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    print("Done.")


if __name__ == "__main__":
    main()
