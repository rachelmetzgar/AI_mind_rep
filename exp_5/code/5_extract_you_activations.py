#!/usr/bin/env python3
"""
Experiment 5, Phase 5: Extract Activations for "You" Variant Stimuli

Same as 1_extract_activations.py but uses get_all_sentences_you(),
which prepends "You" to C2/C5 (dis_mental/dis_action) sentences.

Output:
    results/{model}/activations/data/
        activations_last_token_you.npz   — (336, 41, 5120) float16
        stimuli_metadata_you.csv         — item_id, condition, category, sentence, n_tokens

Usage:
    python code/5_extract_you_activations.py --model llama2_13b_chat

SLURM:
    sbatch code/slurm/5_extract_you_activations.sh

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    set_model, add_model_argument, model_path, hidden_dim, n_layers,
    data_dir, ensure_dir, N_SENTENCES,
)
from stimuli import get_all_sentences_you


def parse_args():
    parser = argparse.ArgumentParser(description="Extract activations for 'You' variant stimuli")
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    out_dir = ensure_dir(data_dir("activations"))
    out_npz = out_dir / "activations_last_token_you.npz"
    out_csv = out_dir / "stimuli_metadata_you.csv"

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
    sentences = get_all_sentences_you()
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
