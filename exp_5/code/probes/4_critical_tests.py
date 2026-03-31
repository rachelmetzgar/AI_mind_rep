#!/usr/bin/env python3
"""
Experiment 5, Step 10: Critical Tests

Residual probe direction, interaction direction, and direction comparison
at peak (position, layer) from attribution probes.

Output:
    results/{model}/probe_training/data/
        critical_tests.json

Usage:
    python code/probes/4_critical_tests.py --model llama2_13b_chat

SLURM:
    sbatch code/probes/slurm/4_critical_tests.sh

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import time
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    set_model, add_model_argument, data_dir, ensure_dir,
    N_SENTENCES, N_ITEMS, N_CONDITIONS,
    POSITION_LABELS, N_POSITIONS, N_PERM_CRITICAL, N_BOOTSTRAP,
    hidden_dim, n_layers,
)
from stimuli import get_condition_indices
from utils.probes import project_out_directions, train_full_probe


def parse_args():
    parser = argparse.ArgumentParser(description="Critical tests (Step 10)")
    add_model_argument(parser)
    parser.add_argument("--n-perm-critical", type=int, default=N_PERM_CRITICAL,
                        help="Permutation iterations for critical tests")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Bootstrap iterations for direction comparison")
    return parser.parse_args()


def find_peak(csv_path, probe_name):
    """Find peak (position, layer) for a probe from attribution results.

    Returns (position_label, position_index, layer_index).
    Uses highest AUC among significant (p_fdr < 0.05) entries.
    Falls back to highest AUC overall if none significant.
    """
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["probe_name"] == probe_name:
                rows.append(r)

    sig_rows = [r for r in rows
                if "p_fdr" in r and r["p_fdr"] != ""
                and float(r["p_fdr"]) < 0.05]

    search = sig_rows if sig_rows else rows
    best = max(search, key=lambda r: float(r["auc"]))

    pos_label = best["position"]
    pos_idx = POSITION_LABELS.index(pos_label)
    layer_idx = int(best["layer"])
    return pos_label, pos_idx, layer_idx


def load_probe_weights(npz_path, probe_name, pos_idx, layer_idx):
    """Load probe weights for a given probe at (position, layer)."""
    data = np.load(npz_path)
    wkey = f"{probe_name}_pos{pos_idx}_layer{layer_idx}"
    return data[wkey]


def compute_auc_on_projection(z, labels):
    """Compute AUC for binary labels given 1D projections z."""
    try:
        return float(roc_auc_score(labels, z))
    except ValueError:
        return float("nan")


def main():
    args = parse_args()
    set_model(args.model)
    n_perm = args.n_perm_critical
    n_boot = args.n_bootstrap

    out_dir = ensure_dir(data_dir("probe_training"))
    out_path = out_dir / "critical_tests.json"

    if out_path.exists():
        print(f"Output already exists: {out_path}")
        print("Delete to rerun. Exiting.")
        return

    # Paths
    attr_csv = out_dir / "attribution_probe_results.csv"
    npz_path = out_dir / "probe_weights.npz"

    if not attr_csv.exists():
        print(f"ERROR: Attribution results not found: {attr_csv}")
        print("Run step 9 first.")
        return
    if not npz_path.exists():
        print(f"ERROR: Probe weights not found: {npz_path}")
        print("Run steps 8 and 9 first.")
        return

    # Find peak from c1_vs_c2 probe
    pos_label, pos_idx, layer_idx = find_peak(attr_csv, "c1_vs_c2")
    print(f"Peak (position, layer) for c1_vs_c2: ({pos_label}, L{layer_idx})")

    # Load activations
    act_path = data_dir("activations") / "activations_multipos.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"]
    print(f"  Shape: {acts.shape}")

    # Extract activations at peak
    X = acts[:, pos_idx, layer_idx, :].astype(np.float64)  # (336, 5120)
    print(f"  Activations at peak: {X.shape}")

    # Condition indices
    c1_idx = np.array(get_condition_indices("mental_state"))
    c2_idx = np.array(get_condition_indices("dis_mental"))
    c4_idx = np.array(get_condition_indices("action"))
    c5_idx = np.array(get_condition_indices("dis_action"))

    # Build C1-vs-rest labels for 336 samples
    labels_c1_rest = np.zeros(N_SENTENCES, dtype=np.int32)
    labels_c1_rest[c1_idx] = 1

    # Load probe weights at peak
    w_attr = load_probe_weights(npz_path, "c1_vs_c2", pos_idx, layer_idx).astype(np.float64)
    w_subj = load_probe_weights(npz_path, "subject_presence", pos_idx, layer_idx).astype(np.float64)
    w_mental = load_probe_weights(npz_path, "mental_verb", pos_idx, layer_idx).astype(np.float64)
    w_gram = load_probe_weights(npz_path, "grammaticality", pos_idx, layer_idx).astype(np.float64)
    w_c4vc5 = load_probe_weights(npz_path, "c4_vs_c5", pos_idx, layer_idx).astype(np.float64)

    results = {
        "peak_position": pos_label,
        "peak_position_idx": pos_idx,
        "peak_layer": layer_idx,
    }

    # ── Test 3a: Residual Probe Direction ────────────────────────────────────

    print(f"\n{'='*60}")
    print("Test 3a: Residual Probe Direction")
    print(f"{'='*60}")
    t0 = time.time()

    w_residual = project_out_directions(w_attr, [w_subj, w_mental, w_gram])
    w_residual_norm = w_residual / (np.linalg.norm(w_residual) + 1e-12)

    z_residual = X @ w_residual_norm  # (336,)
    auc_residual = compute_auc_on_projection(z_residual, labels_c1_rest)
    print(f"  Residual AUC (C1 vs rest): {auc_residual:.4f}")

    # Permutation test
    rng = np.random.default_rng(42)
    null_aucs_3a = np.empty(n_perm)
    for p in range(n_perm):
        perm_labels = rng.permutation(labels_c1_rest)
        null_aucs_3a[p] = compute_auc_on_projection(z_residual, perm_labels)
        if (p + 1) % 2000 == 0:
            print(f"    Perm {p+1}/{n_perm}")

    p_residual = float(np.mean(null_aucs_3a >= auc_residual))
    elapsed = time.time() - t0
    print(f"  p = {p_residual:.4f}  [{elapsed:.0f}s]")

    results["test_3a_residual"] = {
        "auc": round(auc_residual, 6),
        "p_perm": round(p_residual, 6),
        "n_perms": n_perm,
        "w_residual_norm": float(np.linalg.norm(w_residual)),
        "w_attr_norm": float(np.linalg.norm(w_attr)),
        "fraction_preserved": float(np.linalg.norm(w_residual) / (np.linalg.norm(w_attr) + 1e-12)),
    }

    # ── Test 3b: Interaction Direction ───────────────────────────────────────

    print(f"\n{'='*60}")
    print("Test 3b: Interaction Direction")
    print(f"{'='*60}")
    t0 = time.time()

    mean_c1 = X[c1_idx].mean(axis=0)
    mean_c2 = X[c2_idx].mean(axis=0)
    mean_c4 = X[c4_idx].mean(axis=0)
    mean_c5 = X[c5_idx].mean(axis=0)

    delta_mental = mean_c1 - mean_c2
    delta_action = mean_c4 - mean_c5
    w_interaction = delta_mental - delta_action
    w_interaction_norm = w_interaction / (np.linalg.norm(w_interaction) + 1e-12)

    z_interaction = X @ w_interaction_norm  # (336,)
    auc_interaction = compute_auc_on_projection(z_interaction, labels_c1_rest)
    print(f"  Interaction AUC (C1 vs rest): {auc_interaction:.4f}")

    # Permutation test
    rng2 = np.random.default_rng(123)
    null_aucs_3b = np.empty(n_perm)
    for p in range(n_perm):
        perm_labels = rng2.permutation(labels_c1_rest)
        null_aucs_3b[p] = compute_auc_on_projection(z_interaction, perm_labels)
        if (p + 1) % 2000 == 0:
            print(f"    Perm {p+1}/{n_perm}")

    p_interaction = float(np.mean(null_aucs_3b >= auc_interaction))
    elapsed = time.time() - t0
    print(f"  p = {p_interaction:.4f}  [{elapsed:.0f}s]")

    results["test_3b_interaction"] = {
        "auc": round(auc_interaction, 6),
        "p_perm": round(p_interaction, 6),
        "n_perms": n_perm,
        "delta_mental_norm": float(np.linalg.norm(delta_mental)),
        "delta_action_norm": float(np.linalg.norm(delta_action)),
        "w_interaction_norm": float(np.linalg.norm(w_interaction)),
    }

    # Store directions for use in step 11
    results["directions"] = {
        "w_residual": w_residual.tolist(),
        "w_interaction": w_interaction.tolist(),
    }

    # ── Test 3c: Direction Comparison ────────────────────────────────────────

    print(f"\n{'='*60}")
    print("Test 3c: Direction Comparison (c1_vs_c2 vs c4_vs_c5)")
    print(f"{'='*60}")
    t0 = time.time()

    # Cosine similarity between c1_vs_c2 and c4_vs_c5 probe weights
    cos_sim = float(np.dot(w_attr, w_c4vc5) /
                    (np.linalg.norm(w_attr) * np.linalg.norm(w_c4vc5) + 1e-12))
    print(f"  Cosine similarity: {cos_sim:.4f}")

    # Bootstrap CI
    # For each bootstrap, resample 56 items with replacement,
    # retrain both probes, compute cosine
    # Need c1_vs_c2 and c4_vs_c5 data
    c1c2_idx = np.concatenate([c1_idx, c2_idx])
    c1c2_labels = np.concatenate([np.ones(len(c1_idx), dtype=np.int32),
                                   np.zeros(len(c2_idx), dtype=np.int32)])
    c4c5_idx = np.concatenate([c4_idx, c5_idx])
    c4c5_labels = np.concatenate([np.ones(len(c4_idx), dtype=np.int32),
                                   np.zeros(len(c5_idx), dtype=np.int32)])

    X_c1c2 = X[c1c2_idx]
    X_c4c5 = X[c4c5_idx]

    rng3 = np.random.default_rng(456)
    boot_cosines = np.empty(n_boot)

    for b in range(n_boot):
        # Resample items (0..55) with replacement
        boot_items = rng3.choice(N_ITEMS, size=N_ITEMS, replace=True)

        # For c1_vs_c2: item i maps to indices i (in c1) and i (in c2)
        # c1_idx[i] gives the 336-row index for item i in C1
        # In X_c1c2, first 56 rows are C1, next 56 are C2
        boot_c1c2_rows = np.concatenate([boot_items, boot_items + N_ITEMS])
        boot_c4c5_rows = np.concatenate([boot_items, boot_items + N_ITEMS])

        X_b1 = X_c1c2[boot_c1c2_rows]
        y_b1 = c1c2_labels[boot_c1c2_rows]
        X_b2 = X_c4c5[boot_c4c5_rows]
        y_b2 = c4c5_labels[boot_c4c5_rows]

        w1, _ = train_full_probe(X_b1.astype(np.float32), y_b1)
        w2, _ = train_full_probe(X_b2.astype(np.float32), y_b2)

        boot_cosines[b] = float(np.dot(w1, w2) /
                                (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-12))

        if (b + 1) % 2000 == 0:
            elapsed = time.time() - t0
            print(f"    Bootstrap {b+1}/{n_boot}  [{elapsed:.0f}s]")

    ci_lower = float(np.percentile(boot_cosines, 2.5))
    ci_upper = float(np.percentile(boot_cosines, 97.5))
    elapsed = time.time() - t0
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]  [{elapsed:.0f}s]")

    results["test_3c_direction_comparison"] = {
        "cosine_similarity": round(cos_sim, 6),
        "bootstrap_ci_lower": round(ci_lower, 6),
        "bootstrap_ci_upper": round(ci_upper, 6),
        "bootstrap_mean": round(float(np.mean(boot_cosines)), 6),
        "bootstrap_std": round(float(np.std(boot_cosines)), 6),
        "n_bootstrap": n_boot,
    }

    # ── Save ─────────────────────────────────────────────────────────────────

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

    print(f"\n{'='*60}")
    print("Critical tests complete.")
    print(f"  3a Residual AUC: {auc_residual:.4f} (p={p_residual:.4f})")
    print(f"  3b Interaction AUC: {auc_interaction:.4f} (p={p_interaction:.4f})")
    print(f"  3c Cosine sim: {cos_sim:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
