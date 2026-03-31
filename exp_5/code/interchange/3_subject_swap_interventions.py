#!/usr/bin/env python3
"""
Experiment 5, Step 16: Subject Swap Interventions

Swap "He" activations between C1 (mental_state) and C4 (action) conditions
to test whether subject representations differ between mental/action contexts.

Only C1 and C4 have "He" as subject (subject_idx from position map).

Swap types:
  1. Cross-type: For each of 56 items, swap "He" from C1→C4 and C4→C1 (112/layer)
  2. Within-type control: For each item in C1, swap "He" from 10 random other
     C1 items (560/layer)

Total: (112 + 560) x 8 layers = 5,376 forward passes.

Metric: subject_swap_effect = 1 - cosine_sim(rep_intervened, rep_original)
  Measures how much the representation changed due to swapping.
  Should be ~0 for within-type, higher for cross-type if subjects are
  encoded differently in mental vs action contexts.

Output:
    results/{model}/interchange/data/
        subject_swap_results.csv

Usage:
    python code/interchange/3_subject_swap_interventions.py --model llama2_13b_chat

SLURM:
    sbatch code/interchange/slurm/3_subject_swaps.sh

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

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
    data_dir, ensure_dir, N_SENTENCES, N_ITEMS, N_CONDITIONS, CONDITION_LABELS,
)
from stimuli import get_all_sentences, get_condition_indices
from utils.token_positions import load_position_map


# ── Layer subsampling ────────────────────────────────────────────────────────

def get_layer_indices():
    """Return ~8 evenly spaced layer indices for the active model."""
    nl = n_layers()
    step = max(1, nl // 8)
    indices = list(range(step, nl, step))
    if indices[-1] != nl - 1:
        indices.append(nl - 1)
    return indices


# ── Cosine similarity ───────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two numpy vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


# ── Hook ─────────────────────────────────────────────────────────────────────

def make_subject_swap_hook(token_idx, replacement_vector):
    """Create hook that replaces activation at token_idx (subject position)."""
    def hook_fn(module, input, output):
        hidden = output[0]  # (1, seq_len, 5120)
        hidden = hidden.clone()
        hidden[0, token_idx, :] = replacement_vector
        return (hidden,) + output[1:]
    return hook_fn


# ── Single swap ──────────────────────────────────────────────────────────────

def run_subject_swap(model, input_ids, subject_idx, period_idx,
                     replacement_vec, layer_idx, rep_orig):
    """Run one subject swap intervention. Returns subject_swap_effect.

    Args:
        model: loaded LLaMA model
        input_ids: target sentence input_ids tensor on GPU (1, seq_len)
        subject_idx: token index of "He" in target sentence
        period_idx: token index of period in target sentence
        replacement_vec: numpy array (5120,) of source subject activation
        layer_idx: index into hidden_states (0=emb, 1=layer0_out, ...)
        rep_orig: numpy array (5120,) original period rep at last layer

    Returns:
        effect: 1 - cosine_sim(rep_intervened, rep_original)
    """
    layer_num = layer_idx - 1
    replacement_tensor = torch.tensor(
        replacement_vec, dtype=torch.float16
    ).to(model.device)

    hook = make_subject_swap_hook(subject_idx, replacement_tensor)
    handle = model.model.layers[layer_num].register_forward_hook(hook)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )

    rep_intervened = (
        outputs.hidden_states[-1][0, period_idx, :]
        .cpu().float().numpy()
    )
    handle.remove()
    del outputs

    effect = 1.0 - cosine_sim(rep_intervened, rep_orig)
    return effect


# ── Build swap pairs ─────────────────────────────────────────────────────────

def build_subject_swap_pairs(rng):
    """Build list of (target_idx, source_idx, swap_type) for subject swaps.

    Only C1 (mental_state, offset 0) and C4 (action, offset 3) have subjects.

    Returns:
        list of (target_idx, source_idx, swap_type) tuples
    """
    c1_indices = get_condition_indices("mental_state")  # 56 items
    c4_indices = get_condition_indices("action")          # 56 items

    pairs = []

    # 1. Cross-type: C1→C4 and C4→C1 for each of 56 items
    for item_i in range(N_ITEMS):
        c1_idx = c1_indices[item_i]
        c4_idx = c4_indices[item_i]
        # Swap C1's "He" into C4 (same item)
        pairs.append((c4_idx, c1_idx, "cross_c1_to_c4"))
        # Swap C4's "He" into C1 (same item)
        pairs.append((c1_idx, c4_idx, "cross_c4_to_c1"))

    # 2. Within-type control: for each C1 item, swap "He" from 10 random other C1 items
    for target_idx in c1_indices:
        others = [i for i in c1_indices if i != target_idx]
        chosen = rng.choice(others, size=min(10, len(others)), replace=False)
        for source_idx in chosen:
            pairs.append((target_idx, int(source_idx), "within_c1"))

    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Subject swap interchange interventions"
    )
    add_model_argument(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(data_dir("interchange"))
    out_csv = out_dir / "subject_swap_results.csv"

    # ── Load position map ────────────────────────────────────────────────
    pos_csv = data_dir("activations") / "token_position_map.csv"
    pos_map = load_position_map(pos_csv)
    print(f"Loaded position map: {len(pos_map)} entries")

    sentences = get_all_sentences()
    assert len(sentences) == N_SENTENCES

    # ── Build swap pairs ─────────────────────────────────────────────────
    pairs = build_subject_swap_pairs(rng)
    print(f"Built {len(pairs)} subject swap pairs")
    for swap_type in ["cross_c1_to_c4", "cross_c4_to_c1", "within_c1"]:
        n = sum(1 for _, _, st in pairs if st == swap_type)
        print(f"  {swap_type}: {n}")

    # ── Check resume ─────────────────────────────────────────────────────
    completed_layers = set()
    if out_csv.exists():
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_layers.add(int(row["layer_idx"]))
    layer_indices = get_layer_indices()
    remaining_layers = [l for l in layer_indices if l not in completed_layers]
    if not remaining_layers:
        print(f"All {len(layer_indices)} layers complete. Exiting.")
        return
    if completed_layers:
        print(f"Resuming: layers {sorted(completed_layers)} done, "
              f"remaining: {remaining_layers}")

    # ── Pre-load cached activations ──────────────────────────────────────
    full_dir = data_dir("activations") / "full"
    hd = hidden_dim()

    print("Pre-loading cached activations...")
    t_load = time.time()

    # Original period representations at last layer
    period_reps = {}

    # Subject activations at intervention layers
    # subject_acts[sentence_idx][layer_idx] = (hd,) float16
    subject_acts = {}

    # Determine which sentences are involved
    involved = set()
    for target_idx, source_idx, _ in pairs:
        involved.add(target_idx)
        involved.add(source_idx)

    for idx in sorted(involved):
        fpath = full_dir / f"sentence_{idx:03d}.npz"
        act = np.load(fpath)["activations"]  # (n_tok, 41, 5120)

        period_pos = pos_map[idx]["period_idx"]
        subject_pos = pos_map[idx]["subject_idx"]

        # Period rep at last layer
        period_reps[idx] = act[period_pos, -1, :].astype(np.float32)

        # Subject activations at intervention layers
        if subject_pos >= 0:  # Only C1 and C4 have subjects
            subj_dict = {}
            for li in remaining_layers:
                subj_dict[li] = act[subject_pos, li, :].copy()
            subject_acts[idx] = subj_dict

        del act

    elapsed_load = time.time() - t_load
    print(f"Loaded activations for {len(involved)} sentences in {elapsed_load:.1f}s")

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

    # ── Pre-tokenize target sentences ────────────────────────────────────
    print("Pre-tokenizing target sentences...")
    target_input_ids = {}
    target_indices = set(t for t, s, _ in pairs)
    for idx in sorted(target_indices):
        sentence = sentences[idx][3]
        encoding = tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )
        target_input_ids[idx] = encoding["input_ids"].to(model.device)

    # ── Run interventions layer by layer ─────────────────────────────────
    fieldnames = [
        "item", "layer_idx", "swap_type", "partner_item", "effect",
    ]

    write_mode = "a" if completed_layers else "w"
    csv_file = open(out_csv, write_mode, newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not completed_layers:
        writer.writeheader()

    total_swaps = len(pairs) * len(remaining_layers)
    done = 0
    t_start = time.time()

    for layer_idx in remaining_layers:
        layer_t0 = time.time()
        print(f"\n=== Layer {layer_idx} ({remaining_layers.index(layer_idx)+1}/"
              f"{len(remaining_layers)}) ===")

        for pair_i, (target_idx, source_idx, swap_type) in enumerate(pairs):
            target_item = target_idx // N_CONDITIONS + 1  # 1-based
            source_item = source_idx // N_CONDITIONS + 1

            subject_idx_A = pos_map[target_idx]["subject_idx"]
            period_idx_A = pos_map[target_idx]["period_idx"]

            # Source subject activation at this layer
            replacement_vec = subject_acts[source_idx][layer_idx].astype(np.float32)

            # Original period rep for target
            rep_orig = period_reps[target_idx]

            effect = run_subject_swap(
                model,
                target_input_ids[target_idx],
                subject_idx_A,
                period_idx_A,
                replacement_vec,
                layer_idx,
                rep_orig,
            )

            writer.writerow({
                "item": target_item,
                "layer_idx": layer_idx,
                "swap_type": swap_type,
                "partner_item": source_item,
                "effect": f"{effect:.6f}",
            })

            done += 1
            if (pair_i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (total_swaps - done) / rate if rate > 0 else 0
                print(f"  [{pair_i+1}/{len(pairs)}] "
                      f"{done}/{total_swaps} total, "
                      f"{rate:.1f} swaps/s, ETA {eta/60:.0f}min")

        csv_file.flush()
        layer_elapsed = time.time() - layer_t0
        print(f"  Layer {layer_idx} complete: {len(pairs)} pairs in "
              f"{layer_elapsed:.1f}s ({layer_elapsed/len(pairs)*1000:.1f}ms/pair)")

    csv_file.close()

    total_elapsed = time.time() - t_start
    print(f"\nAll done: {done} swaps in {total_elapsed:.1f}s "
          f"({total_elapsed/60:.1f}min)")
    print(f"Results saved to {out_csv}")

    # ── Summary ──────────────────────────────────────────────────────────
    with open(out_csv) as f:
        n_rows = sum(1 for _ in f) - 1
    print(f"Total rows in CSV: {n_rows}")


if __name__ == "__main__":
    main()
