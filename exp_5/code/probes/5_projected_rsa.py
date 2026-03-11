#!/usr/bin/env python3
"""
Experiment 5, Step 11: Probe-Projected RSA

RSA analysis on activations projected onto probe-derived directions.

Output:
    results/{model}/probe_training/data/
        projected_rsa_results.csv
        projected_category_rsa_results.csv

Usage:
    python code/probes/5_projected_rsa.py --model llama2_13b_chat

SLURM:
    sbatch code/probes/slurm/5_projected_rsa.sh

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    set_model, add_model_argument, data_dir, ensure_dir,
    N_SENTENCES, N_ITEMS, N_CONDITIONS,
    CONDITION_LABELS, ITEMS_PER_CATEGORY,
    POSITION_LABELS, N_POSITIONS, N_PERM_CRITICAL,
    hidden_dim, n_layers,
)
from stimuli import get_condition_indices
from utils.rsa import (
    compute_rdm, lower_triangle, simple_rsa,
    build_model_rdms, build_category_rdm,
    partial_rsa_regression,
    permutation_test_simple, permutation_test_partial,
    permutation_test_category, fdr_correct,
)


CONFOUND_KEYS = ["B", "C", "D", "F", "G", "H"]


def parse_args():
    parser = argparse.ArgumentParser(description="Probe-projected RSA (Step 11)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERM_CRITICAL,
                        help="Permutation iterations")
    return parser.parse_args()


def gram_schmidt(vectors):
    """Orthogonalize a list of vectors using Gram-Schmidt.

    Args:
        vectors: list of (D,) arrays

    Returns:
        orthonormal: list of (D,) unit vectors (may be shorter if
                     any vector was linearly dependent)
    """
    basis = []
    for v in vectors:
        v = v.astype(np.float64).copy()
        for b in basis:
            v = v - np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            basis.append(v / norm)
    return basis


def load_directions(out_dir, npz_path):
    """Load significant directions from critical_tests.json or recompute from weights."""
    json_path = out_dir / "critical_tests.json"

    if json_path.exists():
        print(f"Loading directions from {json_path}")
        with open(json_path) as f:
            ct = json.load(f)

        directions = []
        direction_names = []

        if "directions" in ct:
            if "w_residual" in ct["directions"]:
                w_res = np.array(ct["directions"]["w_residual"], dtype=np.float64)
                directions.append(w_res)
                direction_names.append("residual")
            if "w_interaction" in ct["directions"]:
                w_int = np.array(ct["directions"]["w_interaction"], dtype=np.float64)
                directions.append(w_int)
                direction_names.append("interaction")

        peak_pos = ct.get("peak_position_idx", 0)
        peak_layer = ct.get("peak_layer", 0)
        return directions, direction_names, peak_pos, peak_layer

    else:
        print(f"WARNING: {json_path} not found. Run step 10 first.")
        return [], [], 0, 0


def main():
    args = parse_args()
    set_model(args.model)
    n_perms = args.n_perms

    out_dir = ensure_dir(data_dir("probe_training"))
    rsa_csv = out_dir / "projected_rsa_results.csv"
    cat_csv = out_dir / "projected_category_rsa_results.csv"
    npz_path = out_dir / "probe_weights.npz"

    if rsa_csv.exists() and cat_csv.exists():
        print(f"Output already exists:\n  {rsa_csv}\n  {cat_csv}")
        print("Delete to rerun. Exiting.")
        return

    # Load directions
    directions, direction_names, peak_pos, peak_layer = load_directions(out_dir, npz_path)
    if not directions:
        print("ERROR: No directions found. Run steps 8-10 first.")
        return

    k = len(directions)
    print(f"Using {k} directions: {direction_names}")
    print(f"Peak position: {POSITION_LABELS[peak_pos]}, layer: {peak_layer}")

    # Orthogonalize
    ortho_basis = gram_schmidt(directions)
    k_ortho = len(ortho_basis)
    print(f"Orthogonalized: {k} → {k_ortho} basis vectors")

    # Build projection matrix: (k_ortho, D)
    P = np.stack(ortho_basis, axis=0)  # (k_ortho, hidden_dim)

    # Load activations at peak
    act_path = data_dir("activations") / "activations_multipos.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"]
    print(f"  Shape: {acts.shape}")

    X = acts[:, peak_pos, peak_layer, :].astype(np.float64)  # (336, 5120)
    print(f"  Activations at peak: {X.shape}")

    # Project
    z_proj = X @ P.T  # (336, k_ortho)
    print(f"  Projected shape: {z_proj.shape}")

    # ── Full Partial RSA on Projected Activations ────────────────────────────

    print(f"\n{'='*60}")
    print("Partial RSA on projected activations")
    print(f"{'='*60}")

    model_rdms = build_model_rdms(N_ITEMS, N_CONDITIONS)
    neural_rdm = compute_rdm(z_proj, metric="correlation")

    # Run partial RSA: Model A as hypothesis, B-H as confounds
    t0 = time.time()

    # Primary: A vs confounds
    obs_results_A, p_values_A, _ = permutation_test_partial(
        neural_rdm, model_rdms, "A", CONFOUND_KEYS,
        n_perms=n_perms, seed=42
    )
    elapsed = time.time() - t0
    print(f"  Primary (A): beta={obs_results_A['A']['beta']:+.4f}, "
          f"sr={obs_results_A['A']['semi_partial_r']:+.4f}, "
          f"p={p_values_A['A']:.4f}  [{elapsed:.0f}s]")

    # Secondary: E vs confounds
    t0 = time.time()
    obs_results_E, p_values_E, _ = permutation_test_partial(
        neural_rdm, model_rdms, "E", CONFOUND_KEYS,
        n_perms=n_perms, seed=43
    )
    elapsed = time.time() - t0
    print(f"  Secondary (E): beta={obs_results_E['E']['beta']:+.4f}, "
          f"sr={obs_results_E['E']['semi_partial_r']:+.4f}, "
          f"p={p_values_E['E']:.4f}  [{elapsed:.0f}s]")

    # Also run simple RSA for Model A
    neural_vec = lower_triangle(neural_rdm)
    model_A_vec = lower_triangle(model_rdms["A"])
    obs_rho, _ = simple_rsa(neural_vec, model_A_vec)
    print(f"  Simple RSA (A): rho={obs_rho:+.4f}")

    # Build output rows
    rsa_fieldnames = ["analysis", "model", "beta", "semi_partial_r", "p", "p_fdr"]
    rsa_rows = []

    # Simple RSA
    from utils.rsa import permutation_test_simple as pts
    _, p_simple, _ = pts(neural_rdm, model_rdms["A"], n_perms=n_perms, seed=44)
    rsa_rows.append({
        "analysis": "simple",
        "model": "A",
        "beta": round(obs_rho, 6),
        "semi_partial_r": round(obs_rho, 6),
        "p": round(p_simple, 6),
        "p_fdr": round(p_simple, 6),
    })

    # Primary partial RSA results
    all_keys_A = ["A"] + CONFOUND_KEYS
    for k in all_keys_A:
        rsa_rows.append({
            "analysis": "partial_primary",
            "model": k,
            "beta": round(obs_results_A[k]["beta"], 6),
            "semi_partial_r": round(obs_results_A[k]["semi_partial_r"], 6),
            "p": round(p_values_A[k], 6),
            "p_fdr": None,
        })

    # Secondary partial RSA results
    all_keys_E = ["E"] + CONFOUND_KEYS
    for k in all_keys_E:
        rsa_rows.append({
            "analysis": "partial_secondary",
            "model": k,
            "beta": round(obs_results_E[k]["beta"], 6),
            "semi_partial_r": round(obs_results_E[k]["semi_partial_r"], 6),
            "p": round(p_values_E[k], 6),
            "p_fdr": None,
        })

    # FDR correction within each analysis
    for analysis in ["partial_primary", "partial_secondary"]:
        a_rows = [r for r in rsa_rows if r["analysis"] == analysis]
        pvals = np.array([r["p"] for r in a_rows])
        fdr = fdr_correct(pvals)
        for i, r in enumerate(a_rows):
            r["p_fdr"] = round(float(fdr[i]), 6)

    with open(rsa_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rsa_fieldnames)
        writer.writeheader()
        writer.writerows(rsa_rows)
    print(f"\nSaved RSA results to {rsa_csv}")

    # ── Category RSA on Projected C1 Activations ────────────────────────────

    print(f"\n{'='*60}")
    print("Category RSA on projected C1 activations")
    print(f"{'='*60}")

    c1_idx = np.array(get_condition_indices("mental_state"))
    z_c1 = z_proj[c1_idx]  # (56, k_ortho)
    neural_rdm_c1 = compute_rdm(z_c1, metric="correlation")
    category_rdm = build_category_rdm(N_ITEMS, ITEMS_PER_CATEGORY)

    t0 = time.time()
    obs_rho_cat, p_cat, _ = permutation_test_category(
        neural_rdm_c1, category_rdm,
        n_items=N_ITEMS, items_per_cat=ITEMS_PER_CATEGORY,
        n_perms=n_perms, seed=45
    )
    elapsed = time.time() - t0
    print(f"  Category RSA: rho={obs_rho_cat:+.4f}, p={p_cat:.4f}  [{elapsed:.0f}s]")

    # Also run category RSA on each condition
    cat_fieldnames = ["condition", "rho", "p", "p_fdr"]
    cat_rows = []

    for cond in CONDITION_LABELS:
        cond_idx = np.array(get_condition_indices(cond))
        z_cond = z_proj[cond_idx]
        rdm_cond = compute_rdm(z_cond, metric="correlation")

        t0 = time.time()
        rho_c, p_c, _ = permutation_test_category(
            rdm_cond, category_rdm,
            n_items=N_ITEMS, items_per_cat=ITEMS_PER_CATEGORY,
            n_perms=n_perms, seed=46 + CONDITION_LABELS.index(cond)
        )
        elapsed = time.time() - t0
        print(f"  {cond:15s}: rho={rho_c:+.4f}, p={p_c:.4f}  [{elapsed:.0f}s]")

        cat_rows.append({
            "condition": cond,
            "rho": round(rho_c, 6),
            "p": round(p_c, 6),
            "p_fdr": None,
        })

    # FDR correction
    pvals = np.array([r["p"] for r in cat_rows])
    fdr = fdr_correct(pvals)
    for i, r in enumerate(cat_rows):
        r["p_fdr"] = round(float(fdr[i]), 6)

    with open(cat_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cat_fieldnames)
        writer.writeheader()
        writer.writerows(cat_rows)
    print(f"\nSaved category RSA results to {cat_csv}")

    # Summary
    print(f"\n{'='*60}")
    print("Probe-projected RSA complete.")
    print(f"  Directions used: {direction_names}")
    print(f"  Projected dims: {k_ortho}")
    a_row = [r for r in rsa_rows if r["analysis"] == "partial_primary" and r["model"] == "A"][0]
    print(f"  Model A partial: beta={a_row['beta']}, sr={a_row['semi_partial_r']}, p={a_row['p']}")
    c1_cat = [r for r in cat_rows if r["condition"] == "mental_state"][0]
    print(f"  C1 category RSA: rho={c1_cat['rho']}, p={c1_cat['p']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
