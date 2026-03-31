#!/usr/bin/env python3
"""
Experiment 5: Reduced 1-4 RSA (5 predictors A-E, conditions C1-C4 only)

Drops C5 (dis_action) and C6 (scr_action) entirely.
56 items x 4 conditions = 224 sentences, 224x224 neural RDM.

Regression: neural_RDM ~ A + B + C + D + E + error

Model RDMs (all 224x224 binary):
    A (Full Attribution):       both in {C1}
    B (Mental Verb + Object):   both in {C1, C2}
    C (Mental Verb Presence):   both in {C1, C2, C3}
    D (Verb + Object):          both in {C1, C2, C4}
    E (Subject + Verb + Object): both in {C1, C4}

10K permutation tests, Benjamini-Hochberg FDR at q=0.05.
Runs with correlation distance only.

Output:
    results/{model}/rsa/data/reduced_1_4/
        5_predictors_rsa_corr.csv
        vif_check.json
        model_correlations.npz

Usage:
    python code/rsa/reduced_1-4/1_reduced_1_4_rsa.py --model llama2_13b_chat

SLURM:
    sbatch code/rsa/reduced_1-4/slurm/1_reduced_1_4_rsa.sh

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import sys
import time
import argparse
import csv
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    set_model, add_model_argument, data_dir, ensure_dir,
    N_ITEMS, N_CONDITIONS, N_PERMUTATIONS,
)
from utils.rsa import (
    compute_rdm, lower_triangle,
    _cross_item_condition_rdm, _pair_conditions,
    fdr_correct,
)


# Reduced design: 4 conditions per item
N_CONDS_REDUCED = 4
N_SENTENCES_REDUCED = N_ITEMS * N_CONDS_REDUCED  # 224

# Condition mapping from 6-cond to 4-cond:
#   Original: 0=mental_state, 1=dis_mental, 2=scr_mental, 3=action, 4=dis_action, 5=scr_action
#   Reduced:  0=mental_state, 1=dis_mental, 2=scr_mental, 3=action
KEEP_COND_INDICES = [0, 1, 2, 3]  # indices into original 6-condition ordering

ALL_MODEL_KEYS = ["A", "B", "C", "D", "E"]
FIELDNAMES = ["layer", "model", "beta", "semi_partial_r", "delta_r2", "p"]


def filter_activations(acts_full):
    """Filter 336-row activations to keep only C1-C4 (224 rows).

    Input shape:  (336, n_layers, hidden_dim) — ordered as item*6 + cond
    Output shape: (224, n_layers, hidden_dim) — ordered as item*4 + cond
    """
    keep_rows = []
    for item in range(N_ITEMS):
        base = item * N_CONDITIONS  # 6 conditions in original
        for ci in KEEP_COND_INDICES:
            keep_rows.append(base + ci)
    keep_rows = np.array(keep_rows)
    assert len(keep_rows) == N_SENTENCES_REDUCED
    return acts_full[keep_rows]


def build_reduced_model_rdms(n_items=56, n_conds=4):
    """Build 5 model RDMs for the reduced C1-C4 design.

    In the 4-condition layout:
        0 = mental_state (C1)
        1 = dis_mental   (C2)
        2 = scr_mental   (C3)
        3 = action        (C4)
    """
    models = {}

    # Model A: Full Attribution — both in C1
    models["A"] = _cross_item_condition_rdm(n_items, n_conds, [0])

    # Model B: Mental Verb + Object — both in {C1, C2}
    models["B"] = _cross_item_condition_rdm(n_items, n_conds, [0, 1])

    # Model C: Mental Verb Presence — both in {C1, C2, C3}
    models["C"] = _cross_item_condition_rdm(n_items, n_conds, [0, 1, 2])

    # Model D: Verb + Object — both in {C1, C2, C4} (i.e., indices 0,1,3 in reduced)
    models["D"] = _cross_item_condition_rdm(n_items, n_conds, [0, 1, 3])

    # Model E: Subject + Verb + Object — both in {C1, C4} (i.e., indices 0,3 in reduced)
    models["E"] = _cross_item_condition_rdm(n_items, n_conds, [0, 3])

    return models


def regression(neural_vec, model_vecs, model_names):
    """OLS regression of neural RDM on model RDMs.

    Returns dict of model_name -> {beta, semi_partial_r, delta_r2}.
    """
    X = np.column_stack(model_vecs)
    y = neural_vec

    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    y_z = (y - y.mean()) / (y.std() + 1e-12)

    XtX = X_z.T @ X_z
    Xty = X_z.T @ y_z
    try:
        betas = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        betas = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    y_hat = X_z @ betas
    ss_total = np.sum(y_z ** 2)
    ss_full = np.sum((y_z - y_hat) ** 2)

    results = {}
    for i, name in enumerate(model_names):
        keep = [j for j in range(len(model_names)) if j != i]
        X_reduced = X_z[:, keep]
        try:
            betas_reduced = np.linalg.solve(
                X_reduced.T @ X_reduced, X_reduced.T @ y_z
            )
        except np.linalg.LinAlgError:
            betas_reduced = np.linalg.lstsq(
                X_reduced.T @ X_reduced, X_reduced.T @ y_z, rcond=None
            )[0]
        y_hat_reduced = X_reduced @ betas_reduced
        ss_reduced = np.sum((y_z - y_hat_reduced) ** 2)
        delta_r2 = (ss_reduced - ss_full) / ss_total
        semi_partial_r = np.sign(betas[i]) * np.sqrt(max(delta_r2, 0.0))
        results[name] = {
            "beta": float(betas[i]),
            "semi_partial_r": float(semi_partial_r),
            "delta_r2": float(delta_r2),
        }
    return results


def _permute_conditions_within_items(n_items, n_conds, rng):
    """Shuffle condition labels within each item."""
    perm = np.arange(n_items * n_conds)
    for item in range(n_items):
        start = item * n_conds
        block = perm[start:start + n_conds].copy()
        rng.shuffle(block)
        perm[start:start + n_conds] = block
    return perm


def permutation_test(neural_rdm, model_rdm_dict, model_keys,
                     n_perms=10000, seed=42, n_conds=4):
    """Permutation test for 5-predictor regression."""
    n = neural_rdm.shape[0]
    n_items = n // n_conds

    neural_vec = lower_triangle(neural_rdm)
    model_vecs = [lower_triangle(model_rdm_dict[k]) for k in model_keys]

    observed = regression(neural_vec, model_vecs, model_keys)

    rng = np.random.default_rng(seed)
    null_betas = {k: np.empty(n_perms) for k in model_keys}

    for p in range(n_perms):
        perm = _permute_conditions_within_items(n_items, n_conds, rng)
        perm_rdm = neural_rdm[np.ix_(perm, perm)]
        perm_vec = lower_triangle(perm_rdm)
        perm_results = regression(perm_vec, model_vecs, model_keys)
        for k in model_keys:
            null_betas[k][p] = perm_results[k]["beta"]

    p_values = {}
    for k in model_keys:
        obs_beta = observed[k]["beta"]
        p_values[k] = float(np.mean(np.abs(null_betas[k]) >= abs(obs_beta)))

    return observed, p_values, null_betas


def compute_vifs(model_rdm_dict, model_keys):
    """Compute VIF for each predictor."""
    vecs = [lower_triangle(model_rdm_dict[k]) for k in model_keys]
    X = np.column_stack(vecs)
    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    vifs = {}
    for i, name in enumerate(model_keys):
        y_i = X_z[:, i]
        keep = [j for j in range(len(model_keys)) if j != i]
        X_others = X_z[:, keep]
        try:
            betas = np.linalg.solve(X_others.T @ X_others, X_others.T @ y_i)
        except np.linalg.LinAlgError:
            betas = np.linalg.lstsq(X_others.T @ X_others, X_others.T @ y_i, rcond=None)[0]
        y_hat = X_others @ betas
        ss_res = np.sum((y_i - y_hat) ** 2)
        ss_tot = np.sum(y_i ** 2)
        r2 = 1 - ss_res / ss_tot
        vifs[name] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return vifs


def model_correlations(model_rdm_dict, model_keys):
    """Compute pairwise correlations between model RDM vectors."""
    vecs = [lower_triangle(model_rdm_dict[k]) for k in model_keys]
    X = np.column_stack(vecs)
    return np.corrcoef(X.T)


def load_existing(csv_path):
    """Load existing results for resume support."""
    done_layers = set()
    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["layer"] = int(r["layer"])
                for k in ["beta", "semi_partial_r", "delta_r2", "p"]:
                    r[k] = float(r[k])
                rows.append(r)
                done_layers.add(r["layer"])
    return rows, done_layers


def save_with_fdr(rows, csv_path):
    """Sort, add FDR per model, save."""
    rows_sorted = sorted(rows, key=lambda r: (r["layer"],
                         ALL_MODEL_KEYS.index(r["model"])
                         if r["model"] in ALL_MODEL_KEYS else 99))
    for model_key in ALL_MODEL_KEYS:
        model_rows = [r for r in rows_sorted if r["model"] == model_key]
        if not model_rows:
            continue
        pvals = np.array([r["p"] for r in model_rows])
        fdr = fdr_correct(pvals)
        for i, r in enumerate(model_rows):
            r["p_fdr"] = round(float(fdr[i]), 6)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)


def run_analysis(acts, model_rdms, metric, out_csv, n_perms, resume):
    """Run 5-predictor RSA at each layer."""
    nl = acts.shape[1]

    rows, done_layers = [], set()
    if resume:
        rows, done_layers = load_existing(out_csv)
        if done_layers:
            print(f"  Resuming: {len(done_layers)} layers already done")

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric=metric)
        obs_results, p_values, _ = permutation_test(
            neural_rdm, model_rdms, ALL_MODEL_KEYS,
            n_perms=n_perms, seed=42 + layer,
            n_conds=N_CONDS_REDUCED,
        )

        for k in ALL_MODEL_KEYS:
            rows.append({
                "layer": layer,
                "model": k,
                "beta": round(obs_results[k]["beta"], 6),
                "semi_partial_r": round(obs_results[k]["semi_partial_r"], 6),
                "delta_r2": round(obs_results[k]["delta_r2"], 6),
                "p": round(p_values[k], 6),
            })

        # Incremental save (no FDR yet)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            rows_sorted = sorted(rows, key=lambda r: (r["layer"],
                                 ALL_MODEL_KEYS.index(r["model"])
                                 if r["model"] in ALL_MODEL_KEYS else 99))
            writer.writerows(rows_sorted)

        elapsed = time.time() - t0
        a_res = obs_results["A"]
        print(f"  Layer {layer:2d}/{nl-1}: beta_A={a_res['beta']:+.4f}, "
              f"sr_A={a_res['semi_partial_r']:+.4f}, p_A={p_values['A']:.4f}  "
              f"[{elapsed:.0f}s]")

    # Final save with FDR
    save_with_fdr(rows, out_csv)
    print(f"  Saved: {out_csv}")

    a_rows = [r for r in rows if r["model"] == "A"]
    sig = [r["layer"] for r in a_rows if r.get("p_fdr", 1) < 0.05]
    print(f"  Significant layers for A (FDR<.05): {sig if sig else 'none'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reduced 1-4 RSA (C1-C4 only, 5 predictors A-E)"
    )
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    # Load full 336-row activations and filter to C1-C4
    act_path = data_dir("activations") / "activations_last_token.npz"
    print(f"Loading activations from {act_path}")
    acts_full = np.load(act_path)["activations"].astype(np.float32)
    print(f"  Full shape: {acts_full.shape}")

    acts = filter_activations(acts_full)
    print(f"  Reduced shape (C1-C4 only): {acts.shape}")
    assert acts.shape[0] == N_SENTENCES_REDUCED

    out_dir = ensure_dir(data_dir("rsa") / "5_predictors")
    model_rdms = build_reduced_model_rdms(N_ITEMS, N_CONDS_REDUCED)

    # Verify RDM sizes
    for k, rdm in model_rdms.items():
        assert rdm.shape == (N_SENTENCES_REDUCED, N_SENTENCES_REDUCED), \
            f"Model {k} RDM shape {rdm.shape} != ({N_SENTENCES_REDUCED}, {N_SENTENCES_REDUCED})"
    print(f"  Model RDMs: {list(model_rdms.keys())}, each {N_SENTENCES_REDUCED}x{N_SENTENCES_REDUCED}")

    # VIF check
    vifs = compute_vifs(model_rdms, ALL_MODEL_KEYS)
    print("\nVIF check:")
    for k in ALL_MODEL_KEYS:
        flag = " *** WARNING" if vifs[k] > 10 else (" * caution" if vifs[k] > 5 else "")
        print(f"  Model {k}: VIF = {vifs[k]:.2f}{flag}")

    vif_path = out_dir / "vif_check.json"
    with open(vif_path, "w") as f:
        json.dump(vifs, f, indent=2)
    print(f"  VIFs saved: {vif_path}")

    # Model correlations
    corr_matrix = model_correlations(model_rdms, ALL_MODEL_KEYS)
    np.savez(out_dir / "model_correlations.npz",
             corr_matrix=corr_matrix, model_keys=ALL_MODEL_KEYS)

    print(f"\n{'='*60}")
    print("Reduced 1-4 RSA — Correlation distance")
    print(f"{'='*60}")
    run_analysis(acts, model_rdms, "correlation",
                 out_dir / "5_predictors_rsa_corr.csv",
                 args.n_perms, args.resume)

    print("\nDone.")


if __name__ == "__main__":
    main()
