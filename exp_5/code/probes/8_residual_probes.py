#!/usr/bin/env python3
"""
Experiment 5, Step 8: Residual Activation Probes

OLS-deconfound activations (remove subject_presence, mental_verb, grammaticality),
then train logistic probes on residuals to test whether C1 is still distinguishable.
Also trains a raw baseline probe on the same C1-C4 subset for comparison.

Output:
    results/{model}/probe_training/data/
        residual_probe_results.csv
        residual_probe_weights.npz
        residual_confound_r2.csv

Usage:
    python code/probes/8_residual_probes.py --model llama2_13b_chat

SLURM:
    sbatch code/probes/slurm/8_residual_probes.sh

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


# ── Condition subset (C1-C4) ────────────────────────────────────────────────

C14_CONDITIONS = ["mental_state", "dis_mental", "scr_mental", "action"]

# Confound feature values per condition (C1-C4)
# Columns: subject_presence, mental_verb, grammaticality
CONFOUND_FEATURES = {
    "mental_state": [1, 1, 1],  # C1
    "dis_mental":   [0, 1, 1],  # C2
    "scr_mental":   [0, 1, 0],  # C3
    "action":       [1, 0, 1],  # C4
}


def build_c14_mask():
    """Boolean mask for C1-C4 sentences (224 of 336)."""
    mask = np.zeros(N_SENTENCES, dtype=bool)
    for cond in C14_CONDITIONS:
        idx = get_condition_indices(cond)
        mask[idx] = True
    return mask


def build_confound_matrix(n_c14):
    """Build (n_c14, 4) design matrix: [intercept, subject, mental_verb, grammaticality].

    Rows follow the same order as C14 sentences extracted by the mask.
    """
    # Build per-sentence feature vectors matching the mask order
    features = []
    for item_idx in range(N_ITEMS):
        for cond_idx, cond in enumerate(CONDITION_LABELS):
            if cond in C14_CONDITIONS:
                features.append(CONFOUND_FEATURES[cond])
    features = np.array(features, dtype=np.float64)
    assert features.shape == (n_c14, 3), f"Expected ({n_c14}, 3), got {features.shape}"

    # Add intercept column
    F = np.column_stack([np.ones(n_c14), features])
    return F


def build_residual_maker(F):
    """Compute residual-maker matrix M = I - F @ pinv(F).

    M @ x removes the component of x that lies in the column space of F.
    """
    M = np.eye(F.shape[0]) - F @ np.linalg.pinv(F)
    return M


def build_labels(mask):
    """C1 = 1, C2+C3+C4 = 0, for samples in the mask."""
    labels = np.full(N_SENTENCES, -1, dtype=np.int32)
    for cond in C14_CONDITIONS:
        idx = get_condition_indices(cond)
        labels[idx] = 0
    # Override C1 to positive
    c1_idx = get_condition_indices("mental_state")
    labels[c1_idx] = 1
    return labels[mask]


def build_item_ids(mask):
    """Item IDs for masked samples: item = sentence_idx // N_CONDITIONS."""
    full_ids = np.array([i // N_CONDITIONS for i in range(N_SENTENCES)], dtype=np.int32)
    return full_ids[mask]


def parse_args():
    parser = argparse.ArgumentParser(description="Residual activation probes (Step 8)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERM_PROBES,
                        help="Permutation iterations per test")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)
    n_perms = args.n_perms

    out_dir = ensure_dir(data_dir("probe_training"))
    csv_path = out_dir / "residual_probe_results.csv"
    npz_path = out_dir / "residual_probe_weights.npz"
    r2_csv_path = out_dir / "residual_confound_r2.csv"

    # ── Load activations ────────────────────────────────────────────────
    act_path = data_dir("activations") / "activations_multipos.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"]
    print(f"  Shape: {acts.shape}, dtype: {acts.dtype}")
    assert acts.shape == (N_SENTENCES, N_POSITIONS, n_layers(), hidden_dim())

    # ── Build C1-C4 subset ──────────────────────────────────────────────
    mask = build_c14_mask()
    n_c14 = int(mask.sum())
    print(f"\nC1-C4 subset: {n_c14} sentences")

    labels = build_labels(mask)
    item_ids = build_item_ids(mask)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    print(f"  C1 (positive): {n_pos}, C2+C3+C4 (negative): {n_neg}")

    # ── Build confound matrix and residual maker ────────────────────────
    F = build_confound_matrix(n_c14)
    print(f"\nConfound matrix F: shape {F.shape}")
    FtF = F.T @ F
    rank = np.linalg.matrix_rank(FtF)
    print(f"  F'F rank: {rank} (should be {F.shape[1]})")
    assert rank == F.shape[1], f"Design matrix not full rank: {rank} < {F.shape[1]}"

    M = build_residual_maker(F)
    print(f"  Residual-maker M: shape {M.shape}")

    # Condition counts
    for cond in C14_CONDITIONS:
        idx = get_condition_indices(cond)
        in_mask = sum(1 for i in idx if mask[i])
        print(f"  {cond}: {in_mask} sentences in subset")

    # ── Setup ───────────────────────────────────────────────────────────
    nl = n_layers()
    hd = hidden_dim()
    from sklearn.decomposition import PCA

    PCA_DIM = min(256, hd, n_c14 - F.shape[1])  # 224 - 4 = 220
    print(f"\nPCA: {hd} -> {PCA_DIM} dims for permutation tests")

    fieldnames = ["position", "layer", "probe_type", "accuracy", "auc", "p_perm", "p_fdr"]
    r2_fieldnames = ["position", "layer", "mean_r2", "median_r2", "max_r2"]

    all_rows = []
    r2_rows = []
    new_weights = {}
    new_biases = {}

    orthogonality_checked = False

    for probe_type in ["residual", "raw"]:
        print(f"\n{'='*60}")
        print(f"Probe type: {probe_type}")
        print(f"{'='*60}")

        probe_rows = []

        for pos in range(N_POSITIONS):
            pos_label = POSITION_LABELS[pos]
            print(f"\n  Position: {pos_label}")
            t0 = time.time()

            for layer in range(nl):
                X_raw = acts[mask, pos, layer, :].astype(np.float32)

                if probe_type == "residual":
                    # Apply OLS residualization
                    X_work = (M @ X_raw.astype(np.float64)).astype(np.float32)

                    # Orthogonality check at first (position, layer)
                    if not orthogonality_checked:
                        cross = np.abs(F.T @ X_work.astype(np.float64))
                        max_cross = cross.max()
                        print(f"\n  Orthogonality check (pos={pos_label}, layer={layer}):")
                        print(f"    max|F' @ X_resid| = {max_cross:.2e} (should be < 1e-6)")
                        assert max_cross < 1e-4, f"Orthogonality violated: {max_cross}"
                        orthogonality_checked = True

                    # R² diagnostic (how much variance confounds explain)
                    if probe_type == "residual":
                        var_raw = np.var(X_raw, axis=0)
                        var_resid = np.var(X_work, axis=0)
                        # R² per dimension
                        with np.errstate(divide='ignore', invalid='ignore'):
                            r2_per_dim = 1.0 - (var_resid / (var_raw + 1e-30))
                        r2_per_dim = np.clip(r2_per_dim, 0, 1)
                        r2_rows.append({
                            "position": pos_label,
                            "layer": layer,
                            "mean_r2": round(float(np.mean(r2_per_dim)), 6),
                            "median_r2": round(float(np.median(r2_per_dim)), 6),
                            "max_r2": round(float(np.max(r2_per_dim)), 6),
                        })
                else:
                    X_work = X_raw

                # PCA for fast permutation test
                pca = PCA(n_components=PCA_DIM, random_state=42)
                X_pca = pca.fit_transform(X_work)

                obs_acc, obs_auc, p_acc, p_auc = permutation_test_loio(
                    X_pca, labels, item_ids,
                    n_perms=n_perms, seed=42 + pos * 100 + layer,
                    two_sided=(probe_type == "residual"),
                )

                probe_rows.append({
                    "position": pos_label,
                    "layer": layer,
                    "probe_type": probe_type,
                    "accuracy": round(obs_acc, 6),
                    "auc": round(obs_auc, 6),
                    "p_perm": round(p_auc, 6),
                    "p_fdr": None,
                })

                # Train full probe for weight extraction
                w, b = train_full_probe(X_work, labels)
                wkey = f"{probe_type}_pos{pos}_layer{layer}"
                new_weights[wkey] = w
                new_biases[wkey] = b

                if (layer + 1) % 10 == 0 or layer == nl - 1:
                    elapsed = time.time() - t0
                    print(f"    Layer {layer:2d}/{nl-1}: acc={obs_acc:.4f}, "
                          f"auc={obs_auc:.4f}, p={p_auc:.4f}  [{elapsed:.0f}s]")

        # FDR correction within each (probe_type, position)
        for pos_label in POSITION_LABELS:
            pos_rows = [r for r in probe_rows if r["position"] == pos_label]
            pvals = np.array([r["p_perm"] for r in pos_rows])
            fdr = fdr_correct(pvals)
            for i, r in enumerate(pos_rows):
                r["p_fdr"] = round(float(fdr[i]), 6)

        all_rows.extend(probe_rows)

        # ── Incremental save after each probe type ──────────────────────
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Saved partial CSV ({len(all_rows)} rows) to {csv_path}")

        if r2_rows:
            with open(r2_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=r2_fieldnames)
                writer.writeheader()
                writer.writerows(r2_rows)
            print(f"  Saved confound R² ({len(r2_rows)} rows) to {r2_csv_path}")

        combined = {**new_weights, **{f"bias_{k}": v for k, v in new_biases.items()}}
        np.savez_compressed(npz_path, **combined)
        print(f"  Saved probe weights ({len(combined)} arrays) to {npz_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Residual probe training complete.")
    for pt in ["residual", "raw"]:
        pt_rows = [r for r in all_rows if r["probe_type"] == pt]
        sig = [(r["position"], r["layer"]) for r in pt_rows
               if r["p_fdr"] is not None and r["p_fdr"] < 0.05]
        best = max(pt_rows, key=lambda r: r["auc"])
        print(f"  {pt}: {len(sig)} sig (pos,layer) pairs; "
              f"best AUC={best['auc']:.4f} at ({best['position']}, L{best['layer']})")
        sig_rows = [r for r in pt_rows
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
