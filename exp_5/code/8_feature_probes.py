#!/usr/bin/env python3
"""
Experiment 5, Step 8: Feature Probes

Train 4 binary probes (subject_presence, mental_verb, grammaticality, action_verb)
at 3 positions × 41 layers using LOIO CV with permutation tests.

Output:
    results/{model}/probe_training/data/
        feature_probe_results.csv
        probe_weights.npz

Usage:
    python code/8_feature_probes.py --model llama2_13b_chat

SLURM:
    sbatch code/slurm/8_feature_probes.sh

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import sys
import time
import argparse
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    set_model, add_model_argument, data_dir, ensure_dir, figures_dir,
    N_SENTENCES, N_ITEMS, N_CONDITIONS, CONDITION_LABELS,
    POSITION_LABELS, N_POSITIONS, N_PERM_PROBES,
    hidden_dim, n_layers,
)
from stimuli import get_condition_indices
from utils.probes import (
    loio_accuracy_auc, permutation_test_loio, train_full_probe,
)
from utils.rsa import fdr_correct


# ── Probe definitions ────────────────────────────────────────────────────────

PROBE_DEFS = {
    "subject_presence": {
        "pos_conditions": ["mental_state", "action"],          # C1, C4
        "neg_conditions": ["dis_mental", "scr_mental",
                           "dis_action", "scr_action"],        # C2, C3, C5, C6
        "description": "Has 'He' subject (C1+C4 vs C2+C3+C5+C6)",
    },
    "mental_verb": {
        "pos_conditions": ["mental_state", "dis_mental", "scr_mental"],  # C1, C2, C3
        "neg_conditions": ["action", "dis_action", "scr_action"],        # C4, C5, C6
        "description": "Has mental verb (C1+C2+C3 vs C4+C5+C6)",
    },
    "grammaticality": {
        "pos_conditions": ["mental_state", "dis_mental",
                           "action", "dis_action"],            # C1, C2, C4, C5
        "neg_conditions": ["scr_mental", "scr_action"],        # C3, C6
        "description": "Grammatical word order (C1+C2+C4+C5 vs C3+C6)",
    },
    "action_verb": {
        "pos_conditions": ["action", "dis_action", "scr_action"],        # C4, C5, C6
        "neg_conditions": ["mental_state", "dis_mental", "scr_mental"],  # C1, C2, C3
        "description": "Has action verb (C4+C5+C6 vs C1+C2+C3)",
    },
}


def build_labels(probe_name):
    """Build binary label array (336,) for a given probe."""
    defn = PROBE_DEFS[probe_name]
    labels = np.zeros(N_SENTENCES, dtype=np.int32)
    for cond in defn["pos_conditions"]:
        idx = get_condition_indices(cond)
        labels[idx] = 1
    return labels


def build_item_ids():
    """Build item_id array (336,): each sentence maps to item = idx // N_CONDITIONS."""
    return np.array([i // N_CONDITIONS for i in range(N_SENTENCES)], dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Feature probes (Step 8)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERM_PROBES,
                        help="Permutation iterations per test")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)
    n_perms = args.n_perms

    out_dir = ensure_dir(data_dir("probe_training"))
    csv_path = out_dir / "feature_probe_results.csv"
    npz_path = out_dir / "probe_weights.npz"

    # Check for existing output
    if csv_path.exists() and npz_path.exists():
        print(f"Output already exists:\n  {csv_path}\n  {npz_path}")
        print("Delete to rerun. Exiting.")
        return

    # Load activations: (336, 3, 41, 5120) float16
    act_path = data_dir("activations") / "activations_multipos.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"]
    print(f"  Shape: {acts.shape}, dtype: {acts.dtype}")
    assert acts.shape == (N_SENTENCES, N_POSITIONS, n_layers(), hidden_dim())

    item_ids = build_item_ids()
    nl = n_layers()
    fieldnames = ["position", "layer", "probe_name", "accuracy", "auc", "p_perm", "p_fdr"]

    all_rows = []
    all_weights = {}  # key: f"{probe_name}_pos{pos}_layer{layer}" -> (hidden_dim,)
    all_biases = {}

    for probe_name, defn in PROBE_DEFS.items():
        labels = build_labels(probe_name)
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        print(f"\n{'='*60}")
        print(f"Probe: {probe_name}")
        print(f"  {defn['description']}")
        print(f"  Positive: {n_pos}, Negative: {n_neg}")
        print(f"{'='*60}")

        probe_rows = []

        for pos in range(N_POSITIONS):
            pos_label = POSITION_LABELS[pos]
            print(f"\n  Position: {pos_label}")
            t0 = time.time()

            for layer in range(nl):
                X = acts[:, pos, layer, :].astype(np.float32)

                # LOIO CV with permutation test
                obs_acc, obs_auc, p_acc, p_auc = permutation_test_loio(
                    X, labels, item_ids, n_perms=n_perms, seed=42 + pos * 100 + layer
                )

                probe_rows.append({
                    "position": pos_label,
                    "layer": layer,
                    "probe_name": probe_name,
                    "accuracy": round(obs_acc, 6),
                    "auc": round(obs_auc, 6),
                    "p_perm": round(p_auc, 6),
                    "p_fdr": None,  # filled later
                })

                # Train full probe for weight extraction
                w, b = train_full_probe(X, labels)
                wkey = f"{probe_name}_pos{pos}_layer{layer}"
                all_weights[wkey] = w
                all_biases[wkey] = b

                if (layer + 1) % 10 == 0 or layer == nl - 1:
                    elapsed = time.time() - t0
                    print(f"    Layer {layer:2d}/{nl-1}: acc={obs_acc:.4f}, "
                          f"auc={obs_auc:.4f}, p={p_auc:.4f}  [{elapsed:.0f}s]")

        # FDR correction within each (probe, position)
        for pos_label in POSITION_LABELS:
            pos_rows = [r for r in probe_rows
                        if r["position"] == pos_label]
            pvals = np.array([r["p_perm"] for r in pos_rows])
            fdr = fdr_correct(pvals)
            for i, r in enumerate(pos_rows):
                r["p_fdr"] = round(float(fdr[i]), 6)

        all_rows.extend(probe_rows)

        # Incremental CSV save after each probe type
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Saved partial CSV ({len(all_rows)} rows) to {csv_path}")

    # Save probe weights
    np.savez_compressed(npz_path, **all_weights, **{f"bias_{k}": v for k, v in all_biases.items()})
    print(f"\nSaved probe weights to {npz_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Feature probe training complete.")
    for probe_name in PROBE_DEFS:
        probe_rows = [r for r in all_rows if r["probe_name"] == probe_name]
        sig = [(r["position"], r["layer"]) for r in probe_rows
               if r["p_fdr"] is not None and r["p_fdr"] < 0.05]
        best = max(probe_rows, key=lambda r: r["auc"])
        print(f"  {probe_name}: {len(sig)} sig (pos,layer) pairs; "
              f"best AUC={best['auc']:.4f} at ({best['position']}, L{best['layer']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
