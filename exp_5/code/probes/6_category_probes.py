#!/usr/bin/env python3
"""
Experiment 5, Step 12: Category Probes

7-way verb category classification within conditions + cross-condition generalization.

Output:
    results/{model}/probe_training/data/
        category_probe_results.csv

Usage:
    python code/probes/6_category_probes.py --model llama2_13b_chat

SLURM:
    sbatch code/probes/slurm/6_category_probes.sh

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import sys
import time
import argparse
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    set_model, add_model_argument, data_dir, ensure_dir,
    N_SENTENCES, N_ITEMS, N_CONDITIONS,
    CATEGORY_LABELS, ITEMS_PER_CATEGORY, N_CATEGORIES,
    POSITION_LABELS, N_POSITIONS, N_PERM_PROBES,
    hidden_dim, n_layers,
)
from stimuli import get_condition_indices
from utils.probes import multinomial_loio_cv, cross_condition_accuracy


def build_category_labels(n_items=N_ITEMS, items_per_cat=ITEMS_PER_CATEGORY):
    """Build category label array for items 0..55.

    Items 0-7 = category 0 (attention), 8-15 = category 1 (memory), etc.
    Returns (n_items,) array of int category indices.
    """
    return np.array([i // items_per_cat for i in range(n_items)], dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Category probes (Step 12)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=1000,
                        help="Permutation iterations for 5a/5b")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)
    n_perms = args.n_perms

    out_dir = ensure_dir(data_dir("probe_training"))
    csv_path = out_dir / "category_probe_results.csv"

    if csv_path.exists():
        print(f"Output already exists: {csv_path}")
        print("Delete to rerun. Exiting.")
        return

    # Load activations: (336, 3, 41, 5120) float16
    act_path = data_dir("activations") / "activations_multipos.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"]
    print(f"  Shape: {acts.shape}, dtype: {acts.dtype}")
    assert acts.shape == (N_SENTENCES, N_POSITIONS, n_layers(), hidden_dim())

    nl = n_layers()
    hd = hidden_dim()
    cat_labels = build_category_labels()  # (56,)
    chance = 1.0 / N_CATEGORIES  # 14.3%

    c1_idx = np.array(get_condition_indices("mental_state"))
    c2_idx = np.array(get_condition_indices("dis_mental"))

    # Item IDs for LOIO: each of the 56 samples is its own item
    item_ids = np.arange(N_ITEMS, dtype=np.int32)

    from sklearn.decomposition import PCA
    PCA_DIM = min(48, hd, N_ITEMS - 1)  # 56 samples → cap at 48
    print(f"PCA: {hd} -> {PCA_DIM} dims for permutation tests")

    fieldnames = ["position", "layer", "analysis", "accuracy", "chance", "p_perm"]
    all_rows = []

    # ── 5a: 7-way multinomial on C1 only ─────────────────────────────────────

    print(f"\n{'='*60}")
    print("Analysis 5a: 7-way classification on C1 (mental_state)")
    print(f"{'='*60}")

    for pos in range(N_POSITIONS):
        pos_label = POSITION_LABELS[pos]
        print(f"\n  Position: {pos_label}")
        t0 = time.time()

        for layer in range(nl):
            X_c1 = acts[c1_idx, pos, layer, :].astype(np.float32)  # (56, 5120)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_pca = pca.fit_transform(X_c1)

            acc = multinomial_loio_cv(X_pca, cat_labels, item_ids)

            # Permutation test: shuffle category labels across items
            rng = np.random.default_rng(42 + pos * 100 + layer)
            null_accs = np.empty(n_perms)
            for p in range(n_perms):
                perm_labels = rng.permutation(cat_labels)
                null_accs[p] = multinomial_loio_cv(X_pca, perm_labels, item_ids)

            p_perm = float(np.mean(null_accs >= acc))

            all_rows.append({
                "position": pos_label,
                "layer": layer,
                "analysis": "c1_only",
                "accuracy": round(acc, 6),
                "chance": round(chance, 6),
                "p_perm": round(p_perm, 6),
            })

            if (layer + 1) % 10 == 0 or layer == nl - 1:
                elapsed = time.time() - t0
                print(f"    Layer {layer:2d}/{nl-1}: acc={acc:.4f}, "
                      f"p={p_perm:.4f}  [{elapsed:.0f}s]")

    # Incremental save
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved partial CSV ({len(all_rows)} rows)")

    # ── 5b: 7-way multinomial on C2 only ─────────────────────────────────────

    print(f"\n{'='*60}")
    print("Analysis 5b: 7-way classification on C2 (dis_mental)")
    print(f"{'='*60}")

    for pos in range(N_POSITIONS):
        pos_label = POSITION_LABELS[pos]
        print(f"\n  Position: {pos_label}")
        t0 = time.time()

        for layer in range(nl):
            X_c2 = acts[c2_idx, pos, layer, :].astype(np.float32)  # (56, 5120)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_pca = pca.fit_transform(X_c2)

            acc = multinomial_loio_cv(X_pca, cat_labels, item_ids)

            rng = np.random.default_rng(1000 + pos * 100 + layer)
            null_accs = np.empty(n_perms)
            for p in range(n_perms):
                perm_labels = rng.permutation(cat_labels)
                null_accs[p] = multinomial_loio_cv(X_pca, perm_labels, item_ids)

            p_perm = float(np.mean(null_accs >= acc))

            all_rows.append({
                "position": pos_label,
                "layer": layer,
                "analysis": "c2_only",
                "accuracy": round(acc, 6),
                "chance": round(chance, 6),
                "p_perm": round(p_perm, 6),
            })

            if (layer + 1) % 10 == 0 or layer == nl - 1:
                elapsed = time.time() - t0
                print(f"    Layer {layer:2d}/{nl-1}: acc={acc:.4f}, "
                      f"p={p_perm:.4f}  [{elapsed:.0f}s]")

    # Incremental save
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  Saved partial CSV ({len(all_rows)} rows)")

    # ── 5c: Cross-condition generalization ───────────────────────────────────

    print(f"\n{'='*60}")
    print("Analysis 5c: Cross-condition generalization (C1 <-> C2)")
    print(f"{'='*60}")

    for pos in range(N_POSITIONS):
        pos_label = POSITION_LABELS[pos]
        print(f"\n  Position: {pos_label}")
        t0 = time.time()

        for layer in range(nl):
            X_c1 = acts[c1_idx, pos, layer, :].astype(np.float32)
            X_c2 = acts[c2_idx, pos, layer, :].astype(np.float32)

            # PCA fit on combined data for cross-condition
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_both = np.vstack([X_c1, X_c2])
            pca.fit(X_both)
            X_c1_pca = pca.transform(X_c1)
            X_c2_pca = pca.transform(X_c2)

            # Train C1 → test C2
            acc_c1_c2 = cross_condition_accuracy(
                X_c1_pca, cat_labels, X_c2_pca, cat_labels
            )
            # Train C2 → test C1
            acc_c2_c1 = cross_condition_accuracy(
                X_c2_pca, cat_labels, X_c1_pca, cat_labels
            )

            all_rows.append({
                "position": pos_label,
                "layer": layer,
                "analysis": "c1_train_c2_test",
                "accuracy": round(acc_c1_c2, 6),
                "chance": round(chance, 6),
                "p_perm": "",  # no permutation test for cross-condition
            })
            all_rows.append({
                "position": pos_label,
                "layer": layer,
                "analysis": "c2_train_c1_test",
                "accuracy": round(acc_c2_c1, 6),
                "chance": round(chance, 6),
                "p_perm": "",
            })

            if (layer + 1) % 10 == 0 or layer == nl - 1:
                elapsed = time.time() - t0
                print(f"    Layer {layer:2d}/{nl-1}: C1→C2={acc_c1_c2:.4f}, "
                      f"C2→C1={acc_c2_c1:.4f}  [{elapsed:.0f}s]")

    # Final save
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {csv_path}")

    # Summary
    print(f"\n{'='*60}")
    print("Category probe training complete.")
    for analysis in ["c1_only", "c2_only", "c1_train_c2_test", "c2_train_c1_test"]:
        a_rows = [r for r in all_rows if r["analysis"] == analysis]
        if not a_rows:
            continue
        best = max(a_rows, key=lambda r: r["accuracy"])
        print(f"  {analysis}: best acc={best['accuracy']:.4f} "
              f"at ({best['position']}, L{best['layer']})")
        if analysis in ("c1_only", "c2_only"):
            sig = [r for r in a_rows if r["p_perm"] != "" and float(r["p_perm"]) < 0.05]
            print(f"    {len(sig)} / {len(a_rows)} significant at p<.05")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
