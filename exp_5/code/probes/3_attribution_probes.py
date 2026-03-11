#!/usr/bin/env python3
"""
Experiment 5, Step 9: Attribution Probes

Train 3 attribution probes (c1_vs_all, c1_vs_c2, c4_vs_c5)
at 3 positions × 41 layers.

Output:
    results/{model}/probe_training/data/
        attribution_probe_results.csv
        probe_weights.npz (updated)

Usage:
    python code/probes/3_attribution_probes.py --model llama2_13b_chat

SLURM:
    sbatch code/probes/slurm/3_attribution_probes.sh

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
    "c1_vs_all": {
        "pos_conditions": ["mental_state"],                     # C1
        "neg_conditions": ["dis_mental", "scr_mental",
                           "action", "dis_action", "scr_action"],  # C2-C6
        "description": "C1 (1) vs C2-C6 pooled (0) — 56 vs 280",
    },
    "c1_vs_c2": {
        "pos_conditions": ["mental_state"],    # C1
        "neg_conditions": ["dis_mental"],      # C2
        "description": "C1 (1) vs C2 (0) — 56 vs 56",
    },
    "c4_vs_c5": {
        "pos_conditions": ["action"],          # C4
        "neg_conditions": ["dis_action"],      # C5
        "description": "C4 (1) vs C5 (0) — 56 vs 56 (control)",
    },
}


def build_labels_and_mask(probe_name):
    """Build binary labels and boolean mask for included samples.

    Returns:
        labels: (N_SENTENCES,) array — 1 for positive, 0 for negative, -1 for excluded
        mask: (N_SENTENCES,) boolean — True for included samples
    """
    defn = PROBE_DEFS[probe_name]
    labels = np.full(N_SENTENCES, -1, dtype=np.int32)
    for cond in defn["pos_conditions"]:
        idx = get_condition_indices(cond)
        labels[idx] = 1
    for cond in defn["neg_conditions"]:
        idx = get_condition_indices(cond)
        labels[idx] = 0
    mask = labels >= 0
    return labels, mask


def build_item_ids():
    """Build item_id array (336,): each sentence maps to item = idx // N_CONDITIONS."""
    return np.array([i // N_CONDITIONS for i in range(N_SENTENCES)], dtype=np.int32)


def parse_args():
    parser = argparse.ArgumentParser(description="Attribution probes (Step 9)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERM_PROBES,
                        help="Permutation iterations per test")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)
    n_perms = args.n_perms

    out_dir = ensure_dir(data_dir("probe_training"))
    csv_path = out_dir / "attribution_probe_results.csv"
    npz_path = out_dir / "probe_weights.npz"

    # Check for existing output
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

    full_item_ids = build_item_ids()
    nl = n_layers()
    hd = hidden_dim()
    fieldnames = ["position", "layer", "probe_name", "accuracy", "auc", "p_perm", "p_fdr"]

    from sklearn.decomposition import PCA

    # Load existing weights from step 8 if available
    existing_weights = {}
    if npz_path.exists():
        print(f"Loading existing probe weights from {npz_path}")
        with np.load(npz_path) as data:
            for k in data.files:
                existing_weights[k] = data[k]
        print(f"  Loaded {len(existing_weights)} arrays")

    all_rows = []
    new_weights = {}
    new_biases = {}

    for probe_name, defn in PROBE_DEFS.items():
        labels, mask = build_labels_and_mask(probe_name)
        n_pos = np.sum(labels[mask] == 1)
        n_neg = np.sum(labels[mask] == 0)
        n_total = int(n_pos + n_neg)
        print(f"\n{'='*60}")
        print(f"Probe: {probe_name}")
        print(f"  {defn['description']}")
        print(f"  Positive: {n_pos}, Negative: {n_neg}, Total: {n_total}")
        print(f"{'='*60}")

        # Subset item_ids for included samples
        item_ids_sub = full_item_ids[mask]
        labels_sub = labels[mask]

        # PCA for fast permutation tests
        PCA_DIM = min(256, hd, n_total - 1)
        print(f"  PCA: {hd} -> {PCA_DIM} dims for permutation tests")

        probe_rows = []

        for pos in range(N_POSITIONS):
            pos_label = POSITION_LABELS[pos]
            print(f"\n  Position: {pos_label}")
            t0 = time.time()

            for layer in range(nl):
                X_full = acts[mask, pos, layer, :].astype(np.float32)

                # PCA-reduced for permutation test (fast)
                pca = PCA(n_components=PCA_DIM, random_state=42)
                X_pca = pca.fit_transform(X_full)

                obs_acc, obs_auc, p_acc, p_auc = permutation_test_loio(
                    X_pca, labels_sub, item_ids_sub,
                    n_perms=n_perms, seed=42 + pos * 100 + layer
                )

                probe_rows.append({
                    "position": pos_label,
                    "layer": layer,
                    "probe_name": probe_name,
                    "accuracy": round(obs_acc, 6),
                    "auc": round(obs_auc, 6),
                    "p_perm": round(p_auc, 6),
                    "p_fdr": None,
                })

                # Train full probe on original dims for weight extraction
                w, b = train_full_probe(X_full, labels_sub)
                wkey = f"{probe_name}_pos{pos}_layer{layer}"
                new_weights[wkey] = w
                new_biases[wkey] = b

                if (layer + 1) % 10 == 0 or layer == nl - 1:
                    elapsed = time.time() - t0
                    print(f"    Layer {layer:2d}/{nl-1}: acc={obs_acc:.4f}, "
                          f"auc={obs_auc:.4f}, p={p_auc:.4f}  [{elapsed:.0f}s]")

        # FDR correction within each (probe, position)
        for pos_label in POSITION_LABELS:
            pos_rows = [r for r in probe_rows if r["position"] == pos_label]
            pvals = np.array([r["p_perm"] for r in pos_rows])
            fdr = fdr_correct(pvals)
            for i, r in enumerate(pos_rows):
                r["p_fdr"] = round(float(fdr[i]), 6)

        all_rows.extend(probe_rows)

        # Incremental CSV save
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Saved partial CSV ({len(all_rows)} rows) to {csv_path}")

    # Save probe weights (merge with existing from step 8)
    combined = {**existing_weights, **new_weights,
                **{f"bias_{k}": v for k, v in new_biases.items()}}
    np.savez_compressed(npz_path, **combined)
    print(f"\nSaved probe weights ({len(combined)} arrays) to {npz_path}")

    # Find peak per probe
    print(f"\n{'='*60}")
    print("Attribution probe training complete.")
    for probe_name in PROBE_DEFS:
        probe_rows = [r for r in all_rows if r["probe_name"] == probe_name]
        sig = [(r["position"], r["layer"]) for r in probe_rows
               if r["p_fdr"] is not None and r["p_fdr"] < 0.05]
        best = max(probe_rows, key=lambda r: r["auc"])
        print(f"  {probe_name}: {len(sig)} sig (pos,layer) pairs; "
              f"best AUC={best['auc']:.4f} at ({best['position']}, L{best['layer']})")
        # Peak among significant
        sig_rows = [r for r in probe_rows
                    if r["p_fdr"] is not None and r["p_fdr"] < 0.05]
        if sig_rows:
            peak = max(sig_rows, key=lambda r: r["auc"])
            print(f"    Peak (significant): ({peak['position']}, L{peak['layer']}), "
                  f"AUC={peak['auc']:.4f}")
        else:
            print(f"    No significant results — peak uses best overall")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
