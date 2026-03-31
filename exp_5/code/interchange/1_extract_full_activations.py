#!/usr/bin/env python3
"""
Experiment 5, Step 14: Extract Full Activations

Extract ALL token positions × ALL layers for each sentence.
Saved as per-sentence .npz files for interchange interventions.

Output:
    results/{model}/activations/data/full/
        sentence_000.npz  — (n_tok, 41, 5120) float16
        ...
        sentence_335.npz

Usage:
    python code/interchange/1_extract_full_activations.py --model llama2_13b_chat

SLURM:
    sbatch code/interchange/slurm/1_extract_full.sh

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

import sys
import argparse
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
    parser = argparse.ArgumentParser(
        description="Extract full activations (all tokens) for exp 5"
    )
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    out_dir = ensure_dir(data_dir("activations") / "full")

    # ── Check for existing files (supports resume) ────────────────────────
    existing = list(out_dir.glob("sentence_*.npz"))
    start_idx = len(existing)
    if start_idx >= N_SENTENCES:
        print(f"All {N_SENTENCES} sentence files exist in {out_dir}. Exiting.")
        return
    if start_idx > 0:
        print(f"Resuming from index {start_idx} ({start_idx} files found)")

    # ── Load or build position map (for token count info) ─────────────────
    pos_csv = data_dir("activations") / "token_position_map.csv"
    if pos_csv.exists():
        from utils.token_positions import load_position_map
        pos_map = load_position_map(pos_csv)
        print(f"Loaded position map from {pos_csv}")
    else:
        from utils.token_positions import build_position_map, save_position_map
        tokenizer_tmp = AutoTokenizer.from_pretrained(
            model_path(), local_files_only=True
        )
        pos_map = build_position_map(tokenizer_tmp)
        ensure_dir(data_dir("activations"))
        save_position_map(pos_map, pos_csv)
        print(f"Built and saved position map to {pos_csv}")
        del tokenizer_tmp

    # ── Load model ────────────────────────────────────────────────────────
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

    # ── Extract activations ───────────────────────────────────────────────
    sentences = get_all_sentences()
    assert len(sentences) == N_SENTENCES

    nl = n_layers()
    hd = hidden_dim()

    print(f"Extracting {N_SENTENCES} sentences × all tokens × {nl} layers × {hd} dims")
    if start_idx > 0:
        print(f"  Skipping first {start_idx} (already extracted)")

    t0 = time.time()
    with torch.no_grad():
        for idx, (item_id, condition, category, sentence) in enumerate(sentences):
            if idx < start_idx:
                continue

            out_file = out_dir / f"sentence_{idx:03d}.npz"

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

            # Stack all hidden states: (n_tokens, n_layers, hidden_dim)
            act = np.zeros((n_tokens, nl, hd), dtype=np.float16)
            for layer_idx, hs in enumerate(outputs.hidden_states):
                if layer_idx >= nl:
                    break
                act[:, layer_idx, :] = (
                    hs[0, :n_tokens, :].cpu().float().numpy().astype(np.float16)
                )

            np.savez_compressed(out_file, activations=act)

            if (idx + 1) % 56 == 0:
                elapsed = time.time() - t0
                print(
                    f"  [{idx+1:3d}/{N_SENTENCES}] {elapsed:.1f}s, "
                    f"n_tokens={n_tokens}, "
                    f"file size: {out_file.stat().st_size / 1e3:.1f} KB"
                )

            # Free GPU memory
            del outputs, input_ids, encoding, act

    elapsed = time.time() - t0
    print(f"Extraction complete: {N_SENTENCES - start_idx} sentences in {elapsed:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    all_files = sorted(out_dir.glob("sentence_*.npz"))
    total_size = sum(f.stat().st_size for f in all_files)
    print(f"Total: {len(all_files)} files, {total_size / 1e6:.1f} MB")

    # Spot-check first and last files
    for label, fpath in [("First", all_files[0]), ("Last", all_files[-1])]:
        data = np.load(fpath)["activations"]
        print(f"  {label} ({fpath.name}): shape={data.shape}, dtype={data.dtype}")

    print("Done.")


if __name__ == "__main__":
    main()
