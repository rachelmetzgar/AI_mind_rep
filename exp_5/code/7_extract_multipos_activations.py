#!/usr/bin/env python3
"""
Experiment 5, Step 7: Extract Multi-Position Activations

Extract hidden states at verb, object, and period token positions
for all 336 sentences × 41 layers.

Output:
    results/{model}/activations/data/
        activations_multipos.npz   — (336, 3, 41, 5120) float16
        token_position_map.csv     — token position indices

Usage:
    python code/7_extract_multipos_activations.py --model llama2_13b_chat

SLURM:
    sbatch code/slurm/7_extract_multipos.sh

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    set_model, add_model_argument, model_path, hidden_dim, n_layers,
    data_dir, ensure_dir, N_SENTENCES, POSITION_LABELS, N_POSITIONS,
)
from stimuli import get_all_sentences
from utils.token_positions import build_position_map, save_position_map, validate_positions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract multi-position activations for exp 5"
    )
    add_model_argument(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    out_dir = ensure_dir(data_dir("activations"))
    out_npz = out_dir / "activations_multipos.npz"
    out_csv = out_dir / "token_position_map.csv"

    if out_npz.exists():
        print(f"Output already exists: {out_npz}")
        print("Delete it to re-run. Exiting.")
        return

    # ── Load tokenizer for position map ───────────────────────────────────
    mpath = model_path()
    print(f"Loading tokenizer from {mpath}")
    tokenizer = AutoTokenizer.from_pretrained(mpath, local_files_only=True)

    # ── Build and save position map ───────────────────────────────────────
    print("Building token position map...")
    pos_map = build_position_map(tokenizer)
    save_position_map(pos_map, out_csv)
    print(f"  Saved {len(pos_map)} entries to {out_csv}")

    # Validate positions
    n_warn = validate_positions(pos_map, tokenizer)
    if n_warn > 0:
        print(f"  WARNING: {n_warn} position mismatches detected!")

    # ── Load model ────────────────────────────────────────────────────────
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
    n_pos = N_POSITIONS  # 3
    activations = np.zeros((N_SENTENCES, n_pos, nl, hd), dtype=np.float16)

    print(f"Extracting {N_SENTENCES} sentences × {n_pos} positions × {nl} layers × {hd} dims")
    print(f"  Positions: {POSITION_LABELS}")

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

            outputs = model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )

            # Get token positions for this sentence
            entry = pos_map[idx]
            positions = [entry["verb_idx"], entry["object_idx"], entry["period_idx"]]

            # outputs.hidden_states: tuple of (1, seq_len, hidden_dim) x (n_layers+1)
            # Index 0 = embedding layer, 1..40 = transformer blocks
            for layer_idx, hs in enumerate(outputs.hidden_states):
                if layer_idx >= nl:
                    break
                for pos_i, tok_idx in enumerate(positions):
                    activations[idx, pos_i, layer_idx, :] = (
                        hs[0, tok_idx, :].cpu().float().numpy().astype(np.float16)
                    )

            if (idx + 1) % 56 == 0:
                elapsed = time.time() - t0
                print(f"  [{idx+1:3d}/{N_SENTENCES}] {elapsed:.1f}s")

            # Free GPU memory
            del outputs, input_ids, encoding

    elapsed = time.time() - t0
    print(f"Extraction complete: {N_SENTENCES} sentences in {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"Saving activations to {out_npz}")
    np.savez_compressed(out_npz, activations=activations)
    print(f"  Shape: {activations.shape}, dtype: {activations.dtype}")
    print(f"  File size: {out_npz.stat().st_size / 1e6:.1f} MB")

    # ── Verify period position matches existing last-token activations ───
    last_token_npz = out_dir / "activations_last_token.npz"
    if last_token_npz.exists():
        print("Verifying period activations match last-token activations...")
        last_token = np.load(last_token_npz)["activations"]  # (336, 41, 5120)
        period_acts = activations[:, 2, :, :]  # position 2 = period
        max_diff = np.max(
            np.abs(period_acts.astype(np.float32) - last_token.astype(np.float32))
        )
        print(f"  Max difference: {max_diff:.6f}")
        if max_diff < 0.01:
            print("  PASS: period activations match last-token activations")
        else:
            print("  WARNING: period activations differ from last-token activations!")
    else:
        print(f"  Note: {last_token_npz} not found, skipping verification")

    print("Done.")


if __name__ == "__main__":
    main()
