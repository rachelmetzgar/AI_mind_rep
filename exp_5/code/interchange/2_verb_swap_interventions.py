#!/usr/bin/env python3
"""
Experiment 5, Step 15: Verb Swap Interventions

Interchange interventions swapping verb activations between sentence pairs.
Tests whether verb representations carry condition-specific information.

For each swap pair at each layer, runs a forward pass with a hook injecting
the replacement verb activation from a source sentence into the target
sentence. Measures whether the resulting final representation shifts toward
the source sentence's original final representation.

Swap pair types:
  1. Within-condition: 10 random partners per sentence (3,360 pairs)
  2. Cross-condition, same item: all 30 ordered condition pairs (1,680 pairs)
  3. Cross-condition, different item: 1 random partner per pair (1,680 pairs)
Total: ~6,720 pairs x 8 layers = ~53,760 forward passes.

Output:
    results/{model}/interchange/data/
        verb_swap_results.csv

Usage:
    python code/interchange/2_verb_swap_interventions.py --model llama2_13b_chat

SLURM:
    sbatch code/interchange/slurm/2_verb_swaps.sh

Env: llama2_env (GPU required)
Rachel C. Metzgar · Mar 2026
"""

import sys
import argparse
import csv
import time
from pathlib import Path
from itertools import combinations

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

def make_verb_swap_hook(token_idx, replacement_vector):
    """Create hook that replaces activation at token_idx."""
    def hook_fn(module, input, output):
        hidden = output[0]  # (1, seq_len, 5120)
        hidden = hidden.clone()
        hidden[0, token_idx, :] = replacement_vector
        return (hidden,) + output[1:]
    return hook_fn


# ── Single swap ──────────────────────────────────────────────────────────────

def run_swap(model, input_ids_A, verb_idx_A, period_idx_A,
             replacement_vec, layer_idx, rep_A_orig, rep_B_orig):
    """Run one verb swap intervention. Returns swap_success.

    Args:
        model: loaded LLaMA model
        input_ids_A: target sentence input_ids tensor on GPU (1, seq_len)
        verb_idx_A: token index of verb in sentence A
        period_idx_A: token index of period in sentence A
        replacement_vec: numpy array (5120,) of source verb activation
        layer_idx: index into hidden_states (0=emb, 1=layer0_out, ...)
        rep_A_orig: numpy array (5120,) original period rep at last layer
        rep_B_orig: numpy array (5120,) source's original period rep at last layer

    Returns:
        swap_success: cosine_sim(intervened, B) - cosine_sim(intervened, A)
    """
    # layer_idx maps to model.model.layers[layer_idx - 1]
    layer_num = layer_idx - 1
    replacement_tensor = torch.tensor(
        replacement_vec, dtype=torch.float16
    ).to(model.device)

    hook = make_verb_swap_hook(verb_idx_A, replacement_tensor)
    handle = model.model.layers[layer_num].register_forward_hook(hook)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids_A,
            output_hidden_states=True,
            return_dict=True,
        )

    # Get final layer representation at period position
    rep_intervened = (
        outputs.hidden_states[-1][0, period_idx_A, :]
        .cpu().float().numpy()
    )
    handle.remove()
    del outputs

    sim_to_B = cosine_sim(rep_intervened, rep_B_orig)
    sim_to_A = cosine_sim(rep_intervened, rep_A_orig)
    return sim_to_B - sim_to_A


# ── Build swap pairs ─────────────────────────────────────────────────────────

def build_swap_pairs(rng):
    """Build list of (target_idx, source_idx, same_item) swap pairs.

    Returns:
        list of (target_idx, source_idx, same_item) tuples
    """
    pairs = []

    # 1. Within-condition: for each sentence, 10 random others in same condition
    for cond in CONDITION_LABELS:
        cond_indices = get_condition_indices(cond)
        for target_idx in cond_indices:
            others = [i for i in cond_indices if i != target_idx]
            chosen = rng.choice(others, size=min(10, len(others)), replace=False)
            for source_idx in chosen:
                pairs.append((target_idx, int(source_idx), False))

    # 2. Cross-condition, same item: all 30 ordered pairs of 6 conditions
    for item_i in range(N_ITEMS):
        item_indices = [item_i * N_CONDITIONS + c for c in range(N_CONDITIONS)]
        for ci in range(N_CONDITIONS):
            for cj in range(N_CONDITIONS):
                if ci == cj:
                    continue
                target_idx = item_indices[ci]
                source_idx = item_indices[cj]
                pairs.append((target_idx, source_idx, True))

    # 3. Cross-condition, different item: 1 random partner per ordered pair
    for ci in range(N_CONDITIONS):
        for cj in range(N_CONDITIONS):
            if ci == cj:
                continue
            cond_i_indices = get_condition_indices(CONDITION_LABELS[ci])
            cond_j_indices = get_condition_indices(CONDITION_LABELS[cj])
            for target_idx in cond_i_indices:
                target_item = target_idx // N_CONDITIONS
                # Pick a random source from cond_j that is a different item
                others = [
                    s for s in cond_j_indices if s // N_CONDITIONS != target_item
                ]
                source_idx = rng.choice(others)
                pairs.append((target_idx, int(source_idx), False))

    return pairs


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verb swap interchange interventions"
    )
    add_model_argument(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(data_dir("interchange"))
    out_csv = out_dir / "verb_swap_results.csv"

    # ── Load position map ────────────────────────────────────────────────
    pos_csv = data_dir("activations") / "token_position_map.csv"
    pos_map = load_position_map(pos_csv)
    print(f"Loaded position map: {len(pos_map)} entries")

    sentences = get_all_sentences()
    assert len(sentences) == N_SENTENCES

    # ── Build swap pairs ─────────────────────────────────────────────────
    pairs = build_swap_pairs(rng)
    print(f"Built {len(pairs)} swap pairs")

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
    nl = n_layers()

    print("Pre-loading cached activations...")
    t_load = time.time()

    # Original period representations at last layer (for all 336 sentences)
    period_reps = np.zeros((N_SENTENCES, hd), dtype=np.float32)

    # Verb activations at intervention layers
    # verb_acts[sentence_idx][layer_idx] = (hd,) float16
    verb_acts = {}

    # Determine which sentences are involved
    involved = set()
    for target_idx, source_idx, _ in pairs:
        involved.add(target_idx)
        involved.add(source_idx)

    for idx in sorted(involved):
        fpath = full_dir / f"sentence_{idx:03d}.npz"
        act = np.load(fpath)["activations"]  # (n_tok, 41, 5120)

        period_pos = pos_map[idx]["period_idx"]
        verb_pos = pos_map[idx]["verb_idx"]

        # Period rep at last layer (layer_idx 40 = hidden_states[-1])
        period_reps[idx] = act[period_pos, -1, :].astype(np.float32)

        # Verb activations at intervention layers
        verb_dict = {}
        for li in remaining_layers:
            verb_dict[li] = act[verb_pos, li, :].copy()
        verb_acts[idx] = verb_dict

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

    # ── Pre-tokenize all target sentences ────────────────────────────────
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
        "source_idx", "target_idx", "source_cond", "target_cond",
        "item_a", "item_b", "layer_idx", "swap_success", "same_item",
    ]

    # Open CSV in append mode (or write mode if starting fresh)
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

        for pair_i, (target_idx, source_idx, same_item) in enumerate(pairs):
            target_cond = CONDITION_LABELS[target_idx % N_CONDITIONS]
            source_cond = CONDITION_LABELS[source_idx % N_CONDITIONS]
            target_item = target_idx // N_CONDITIONS + 1  # 1-based item_id
            source_item = source_idx // N_CONDITIONS + 1

            verb_idx_A = pos_map[target_idx]["verb_idx"]
            period_idx_A = pos_map[target_idx]["period_idx"]

            # Source verb activation at this layer
            replacement_vec = verb_acts[source_idx][layer_idx].astype(np.float32)

            # Original period reps
            rep_A_orig = period_reps[target_idx]
            rep_B_orig = period_reps[source_idx]

            swap_success = run_swap(
                model,
                target_input_ids[target_idx],
                verb_idx_A,
                period_idx_A,
                replacement_vec,
                layer_idx,
                rep_A_orig,
                rep_B_orig,
            )

            writer.writerow({
                "source_idx": source_idx,
                "target_idx": target_idx,
                "source_cond": source_cond,
                "target_cond": target_cond,
                "item_a": target_item,
                "item_b": source_item,
                "layer_idx": layer_idx,
                "swap_success": f"{swap_success:.6f}",
                "same_item": int(same_item),
            })

            done += 1
            if (pair_i + 1) % 500 == 0:
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
