#!/usr/bin/env python3
"""
Experiment 5, Phase 4b: Reduced 4-Condition RSA

Drops C5 (dis_action) and C6 (scr_action), keeping C1-C4 only.
Drops Models G and H, keeping confounds B, C, D, F only.
Runs all 3 analyses: simple RSA, partial RSA (A and E), category RSA.

Rationale: C5/C6 never test Model A — their similar blocks don't overlap
with C1×C1. Removing them reduces multicollinearity, increases C1×C1 pair
proportion (2.7% → 6.2%), and sharpens standard errors.

Output:
    results/{model}/rsa/data/reduced_4cond/
        simple_rsa_results.csv
        partial_rsa_primary_results.csv
        partial_rsa_secondary_results.csv
        category_rsa_results.csv

Usage:
    python code/rsa/5_reduced_rsa.py --model llama2_13b_chat
    python code/rsa/5_reduced_rsa.py --model llama2_13b_chat --analysis simple
    python code/rsa/5_reduced_rsa.py --model llama2_13b_chat --analysis partial
    python code/rsa/5_reduced_rsa.py --model llama2_13b_chat --analysis category

SLURM:
    sbatch code/rsa/slurm/5_reduced_rsa.sh

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
    N_ITEMS, N_CONDITIONS, N_SENTENCES, N_PERMUTATIONS,
    CONDITION_LABELS, CATEGORY_LABELS,
)
from utils.rsa import (
    compute_rdm, build_model_rdms_4cond, build_category_rdm,
    permutation_test_simple, permutation_test_partial,
    permutation_test_category, fdr_correct,
    lower_triangle, simple_rsa,
)

# 4-condition design: keep C1, C2, C3, C4 (indices 0, 1, 2, 3 within each item)
KEEP_CONDS = [0, 1, 2, 3]  # mental_state, dis_mental, scr_mental, action
N_CONDS_REDUCED = 4
CONFOUND_KEYS = ["B", "C", "D", "F"]  # dropped G, H
COND_NAMES_REDUCED = ["mental_state", "dis_mental", "scr_mental", "action"]


def subset_activations(acts_full, n_items=N_ITEMS, n_conds_orig=N_CONDITIONS):
    """Extract C1-C4 rows from the 336-row activation matrix.

    Input:  (336, n_layers, hidden_dim) — items interleaved with 6 conditions
    Output: (224, n_layers, hidden_dim) — items interleaved with 4 conditions

    New ordering: item0_C1, item0_C2, item0_C3, item0_C4, item1_C1, ...
    """
    keep_indices = []
    for item in range(n_items):
        base = item * n_conds_orig
        for c in KEEP_CONDS:
            keep_indices.append(base + c)
    keep_indices = np.array(keep_indices)
    return acts_full[keep_indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Reduced 4-condition RSA")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--analysis", choices=["all", "simple", "partial", "category"],
                        default="all", help="Which analysis to run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    return parser.parse_args()


def load_existing(csv_path, key_col="layer"):
    """Load existing CSV rows and return (rows, set of done keys)."""
    done = set()
    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                for k in r:
                    try:
                        r[k] = int(r[k]) if k == key_col else float(r[k])
                    except (ValueError, TypeError):
                        pass
                rows.append(r)
                done.add(r[key_col])
    return rows, done


def run_simple(acts, model_rdms, out_csv, n_perms, resume):
    """Simple RSA: Model A vs neural RDM."""
    nl = acts.shape[1]
    model_A = model_rdms["A"]
    fieldnames = ["layer", "rho", "p"]

    rows, done_layers = [], set()
    if resume:
        rows, done_layers = load_existing(out_csv)
        if done_layers:
            print(f"  Resuming: {len(done_layers)} layers done")

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric="correlation")
        obs_rho, p_val, _ = permutation_test_simple(
            neural_rdm, model_A, n_perms=n_perms, seed=42 + layer,
            n_conds=N_CONDS_REDUCED
        )
        rows.append({"layer": layer, "rho": round(obs_rho, 6), "p": round(p_val, 6)})

        # Incremental save
        rows_sorted = sorted(rows, key=lambda r: r["layer"])
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

        elapsed = time.time() - t0
        print(f"  Layer {layer:2d}/{nl-1}: rho={obs_rho:+.4f}, p={p_val:.4f}  [{elapsed:.0f}s]")

    # FDR
    rows_sorted = sorted(rows, key=lambda r: r["layer"])
    pvals = np.array([r["p"] for r in rows_sorted])
    fdr = fdr_correct(pvals)
    for i, r in enumerate(rows_sorted):
        r["p_fdr"] = round(float(fdr[i]), 6)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"  Saved: {out_csv}")

    sig = [r["layer"] for r in rows_sorted if r.get("p_fdr", 1) < 0.05]
    print(f"  Significant (FDR<.05): {sig if sig else 'none'}")


def run_partial(acts, model_rdms, hyp_key, out_csv, n_perms, resume,
                confound_keys=None):
    """Partial RSA for one hypothesis model."""
    nl = acts.shape[1]
    confounds = confound_keys if confound_keys is not None else CONFOUND_KEYS
    all_keys = [hyp_key] + list(confounds)
    fieldnames = ["layer", "model", "beta", "semi_partial_r", "p"]

    rows, done_layers = [], set()
    if resume:
        rows, done_layers = load_existing(out_csv)
        if done_layers:
            print(f"  Resuming: {len(done_layers)} layers done")

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric="correlation")
        obs_results, p_values, _ = permutation_test_partial(
            neural_rdm, model_rdms, hyp_key, confounds,
            n_perms=n_perms, seed=42 + layer,
            n_conds=N_CONDS_REDUCED
        )

        for k in all_keys:
            rows.append({
                "layer": layer,
                "model": k,
                "beta": round(obs_results[k]["beta"], 6),
                "semi_partial_r": round(obs_results[k]["semi_partial_r"], 6),
                "p": round(p_values[k], 6),
            })

        # Incremental save
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            rows_sorted = sorted(rows, key=lambda r: (r["layer"],
                                 all_keys.index(r["model"]) if r["model"] in all_keys else 99))
            writer.writerows(rows_sorted)

        hyp_res = obs_results[hyp_key]
        elapsed = time.time() - t0
        print(f"  Layer {layer:2d}/{nl-1}: beta_{hyp_key}={hyp_res['beta']:+.4f}, "
              f"sr={hyp_res['semi_partial_r']:+.4f}, p={p_values[hyp_key]:.4f}  "
              f"[{elapsed:.0f}s]")

    # FDR per model
    rows_sorted = sorted(rows, key=lambda r: (r["layer"],
                         all_keys.index(r["model"]) if r["model"] in all_keys else 99))
    for model_key in all_keys:
        model_rows = [r for r in rows_sorted if r["model"] == model_key]
        if not model_rows:
            continue
        pvals = np.array([r["p"] for r in model_rows])
        fdr = fdr_correct(pvals)
        for i, r in enumerate(model_rows):
            r["p_fdr"] = round(float(fdr[i]), 6)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"  Saved: {out_csv}")

    hyp_rows = [r for r in rows_sorted if r["model"] == hyp_key]
    sig = [r["layer"] for r in hyp_rows if r.get("p_fdr", 1) < 0.05]
    print(f"  Significant for {hyp_key} (FDR<.05): {sig if sig else 'none'}")


def run_category(acts_full, out_csv, n_perms, resume):
    """Category RSA within each of the 4 conditions."""
    nl = acts_full.shape[1]
    cat_rdm = build_category_rdm(N_ITEMS, 8)
    fieldnames = ["layer", "condition", "rho", "p"]

    rows, done = [], set()
    if resume and out_csv.exists():
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["layer"] = int(r["layer"])
                r["rho"] = float(r["rho"])
                r["p"] = float(r["p"])
                rows.append(r)
                done.add((r["layer"], r["condition"]))
        if done:
            print(f"  Resuming: {len(done)} (layer, cond) pairs done")

    t0 = time.time()
    for cond_idx, cond_name in enumerate(COND_NAMES_REDUCED):
        print(f"\n  Condition: {cond_name}")
        # Extract 56 rows for this condition from the 224-row subset
        cond_rows = np.array([item * N_CONDS_REDUCED + cond_idx
                              for item in range(N_ITEMS)])

        for layer in range(nl):
            if (layer, cond_name) in done:
                continue

            X = acts_full[cond_rows, layer, :]
            neural_rdm = compute_rdm(X, metric="correlation")
            obs_rho, p_val, _ = permutation_test_category(
                neural_rdm, cat_rdm, n_items=N_ITEMS, items_per_cat=8,
                n_perms=n_perms, seed=42 + cond_idx * 100 + layer
            )

            rows.append({
                "layer": layer,
                "condition": cond_name,
                "rho": round(obs_rho, 6),
                "p": round(p_val, 6),
            })

            if (layer + 1) % 10 == 0 or layer == nl - 1:
                elapsed = time.time() - t0
                print(f"    Layer {layer:2d}/{nl-1}: rho={obs_rho:+.4f}, p={p_val:.4f}  [{elapsed:.0f}s]")

        # Incremental save after each condition
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["layer"], r["condition"])))

    # FDR per condition
    rows_sorted = sorted(rows, key=lambda r: (r["layer"], r["condition"]))
    for cond_name in COND_NAMES_REDUCED:
        cond_rows_list = [r for r in rows_sorted if r["condition"] == cond_name]
        if not cond_rows_list:
            continue
        pvals = np.array([r["p"] for r in cond_rows_list])
        fdr = fdr_correct(pvals)
        for i, r in enumerate(cond_rows_list):
            r["p_fdr"] = round(float(fdr[i]), 6)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"\n  Saved: {out_csv}")

    for cond_name in COND_NAMES_REDUCED:
        cond_rows_list = [r for r in rows_sorted if r["condition"] == cond_name]
        sig = [r["layer"] for r in cond_rows_list if r.get("p_fdr", 1) < 0.05]
        if sig:
            print(f"  {cond_name}: significant at layers {sig}")


def main():
    args = parse_args()
    set_model(args.model)

    act_path = data_dir("activations") / "activations_last_token.npz"
    print(f"Loading activations from {act_path}")
    acts_full = np.load(act_path)["activations"].astype(np.float32)
    print(f"  Full shape: {acts_full.shape}")

    # Subset to 4 conditions
    acts = subset_activations(acts_full)
    print(f"  Reduced shape: {acts.shape}  (C1, C2, C3, C4 only)")
    n_expected = N_ITEMS * N_CONDS_REDUCED
    assert acts.shape[0] == n_expected, f"Expected {n_expected}, got {acts.shape[0]}"

    out_dir = ensure_dir(data_dir("rsa") / "reduced_4cond")
    model_rdms = build_model_rdms_4cond(N_ITEMS)

    print(f"\nC1×C1 pairs: {N_ITEMS * (N_ITEMS - 1) // 2} / "
          f"{n_expected * (n_expected - 1) // 2} total = "
          f"{N_ITEMS * (N_ITEMS - 1) / (n_expected * (n_expected - 1)) * 100:.1f}%")
    print(f"Confounds: {CONFOUND_KEYS}  (4 regressors in partial RSA)")

    if args.analysis in ("all", "simple"):
        print(f"\n{'='*60}")
        print("Analysis 1: Simple RSA — Model A (4-condition)")
        print(f"{'='*60}")
        run_simple(acts, model_rdms,
                   out_dir / "simple_rsa_results.csv",
                   args.n_perms, args.resume)

    if args.analysis in ("all", "partial"):
        print(f"\n{'='*60}")
        print("Analysis 2: Combined Partial RSA — A and E in one regression")
        print(f"{'='*60}\n")
        print(f"  Formula: RDM_neural = β_A·A + β_E·E + β_B·B + β_C·C + β_D·D + β_F·F + ε")
        print(f"  β_A = unique variance of full attribution BEYOND verb+object binding (E)")
        print(f"  β_E = unique variance of verb+object binding BEYOND full attribution (A)")
        # A is "hypothesis", E + B,C,D,F are confounds — gives us β_A unique of E
        # But OLS gives betas for ALL predictors, so we get β_E too
        run_partial(acts, model_rdms, "A",
                    out_dir / "partial_rsa_combined_results.csv",
                    args.n_perms, args.resume,
                    confound_keys=["E", "B", "C", "D", "F"])

    if args.analysis in ("all", "category"):
        print(f"\n{'='*60}")
        print("Analysis 3: Category RSA (4-condition)")
        print(f"{'='*60}")
        run_category(acts, out_dir / "category_rsa_results.csv",
                     args.n_perms, args.resume)

    print(f"\n{'='*60}")
    print("Reduced 4-condition RSA complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
