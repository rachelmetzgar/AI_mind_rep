#!/usr/bin/env python3
"""
Experiment 5, Phase 6: Variant RSA Analyses

Unified script for running RSA with different distance metrics and/or
stimulus variants. Runs all three analyses (simple, partial, category)
and saves to a variant-specific subfolder.

Variants:
    cosine       — cosine distance, original stimuli
    corr_you     — correlation distance, "You" stimuli (C2/C5 have "You" subject)
    cosine_you   — cosine distance, "You" stimuli

Output:
    results/{model}/rsa/data/{variant}/
        simple_rsa_results.csv
        partial_rsa_primary_results.csv
        partial_rsa_secondary_results.csv
        category_rsa_results.csv

Usage:
    python code/6_variant_rsa.py --metric cosine --variant original
    python code/6_variant_rsa.py --metric correlation --variant you
    python code/6_variant_rsa.py --metric cosine --variant you

SLURM:
    sbatch code/slurm/6_variant_cosine.sh
    sbatch code/slurm/6_variant_corr_you.sh
    sbatch code/slurm/6_variant_cosine_you.sh

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
    set_model, add_model_argument, data_dir, ensure_dir,
    N_ITEMS, N_CONDITIONS, N_SENTENCES, N_PERMUTATIONS,
    CONDITION_LABELS, ITEMS_PER_CATEGORY,
)
from stimuli import get_condition_indices
from utils.rsa import (
    compute_rdm, build_model_rdms, build_category_rdm,
    permutation_test_simple, permutation_test_partial,
    permutation_test_category, fdr_correct,
)


CONFOUND_KEYS = ["B", "C", "D", "F", "G", "H"]

VARIANT_NAMES = {
    ("cosine", "original"): "cosine",
    ("correlation", "you"): "corr_you",
    ("cosine", "you"): "cosine_you",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Variant RSA analyses")
    add_model_argument(parser)
    parser.add_argument("--metric", choices=["correlation", "cosine"],
                        required=True, help="Distance metric for neural RDMs")
    parser.add_argument("--variant", choices=["original", "you"],
                        required=True, help="Stimulus variant")
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    return parser.parse_args()


def variant_name(metric, variant):
    return VARIANT_NAMES[(metric, variant)]


def load_activations(variant):
    """Load the appropriate activation file."""
    act_dir = data_dir("activations")
    if variant == "original":
        path = act_dir / "activations_last_token.npz"
    else:
        path = act_dir / "activations_last_token_you.npz"
    print(f"Loading activations from {path}")
    acts = np.load(path)["activations"].astype(np.float32)
    print(f"  Shape: {acts.shape}")
    return acts


# ── Analysis 1: Simple RSA ─────────────────────────────────────────────────

def run_simple_rsa(acts, out_dir, metric, n_perms, resume):
    """Simple RSA: Model A vs neural RDM at each layer."""
    out_csv = out_dir / "simple_rsa_results.csv"
    fieldnames = ["layer", "rho", "p"]
    nl = acts.shape[1]

    done_layers = set()
    rows = []
    if resume and out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                r["layer"] = int(r["layer"])
                r["rho"] = float(r["rho"])
                r["p"] = float(r["p"])
                rows.append(r)
                done_layers.add(r["layer"])
        print(f"  Resuming simple RSA: {len(done_layers)} layers done")

    model_rdms = build_model_rdms(N_ITEMS, N_CONDITIONS)
    model_A = model_rdms["A"]

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric=metric)
        obs_rho, p_val, _ = permutation_test_simple(
            neural_rdm, model_A, n_perms=n_perms, seed=42 + layer
        )
        row = {"layer": layer, "rho": round(obs_rho, 6), "p": round(p_val, 6)}
        rows.append(row)

        # Save incrementally
        rows_sorted = sorted(rows, key=lambda r: r["layer"])
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)

        elapsed = time.time() - t0
        print(f"  Layer {layer:2d}/{nl-1}: rho={obs_rho:+.4f}, p={p_val:.4f}  [{elapsed:.0f}s]")

    # FDR correction
    rows_sorted = sorted(rows, key=lambda r: r["layer"])
    pvals = np.array([r["p"] for r in rows_sorted])
    fdr = fdr_correct(pvals)
    for i, r in enumerate(rows_sorted):
        r["p_fdr"] = round(float(fdr[i]), 6)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    sig = [r["layer"] for r in rows_sorted if r.get("p_fdr", 1) < 0.05]
    print(f"  Simple RSA done. Significant layers (FDR<.05): {sig if sig else 'none'}")


# ── Analysis 2: Partial RSA ────────────────────────────────────────────────

PARTIAL_FIELDNAMES = ["layer", "model", "beta", "semi_partial_r", "p"]


def load_partial_existing(csv_path):
    done_layers = set()
    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                r["layer"] = int(r["layer"])
                r["beta"] = float(r["beta"])
                r["semi_partial_r"] = float(r["semi_partial_r"])
                r["p"] = float(r["p"])
                rows.append(r)
                done_layers.add(r["layer"])
    return rows, done_layers


def save_partial_with_fdr(rows, csv_path, all_keys):
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

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PARTIAL_FIELDNAMES + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)


def run_partial_rsa(acts, out_dir, metric, hyp_key, suffix, n_perms, resume):
    """Partial RSA for one hypothesis model."""
    out_csv = out_dir / f"partial_rsa_{suffix}_results.csv"
    nl = acts.shape[1]
    all_keys = [hyp_key] + CONFOUND_KEYS

    model_rdms = build_model_rdms(N_ITEMS, N_CONDITIONS)

    rows, done_layers = [], set()
    if resume:
        rows, done_layers = load_partial_existing(out_csv)
        if done_layers:
            print(f"  Resuming partial RSA ({suffix}): {len(done_layers)} layers done")

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric=metric)
        obs_results, p_values, _ = permutation_test_partial(
            neural_rdm, model_rdms, hyp_key, CONFOUND_KEYS,
            n_perms=n_perms, seed=42 + layer
        )

        for k in all_keys:
            rows.append({
                "layer": layer,
                "model": k,
                "beta": round(obs_results[k]["beta"], 6),
                "semi_partial_r": round(obs_results[k]["semi_partial_r"], 6),
                "p": round(p_values[k], 6),
            })

        # Save incrementally (without FDR)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PARTIAL_FIELDNAMES)
            writer.writeheader()
            rows_sorted = sorted(rows, key=lambda r: (r["layer"],
                                 all_keys.index(r["model"]) if r["model"] in all_keys else 99))
            writer.writerows(rows_sorted)

        hyp_res = obs_results[hyp_key]
        elapsed = time.time() - t0
        print(f"  Layer {layer:2d}/{nl-1}: beta_{hyp_key}={hyp_res['beta']:+.4f}, "
              f"sr={hyp_res['semi_partial_r']:+.4f}, p={p_values[hyp_key]:.4f}  "
              f"[{elapsed:.0f}s]")

    # Final save with FDR
    save_partial_with_fdr(rows, out_csv, all_keys)

    hyp_rows = [r for r in rows if r["model"] == hyp_key]
    sig = [r["layer"] for r in hyp_rows if r.get("p_fdr", 1) < 0.05]
    print(f"  Partial RSA ({suffix}) done. {hyp_key} sig layers (FDR<.05): {sig if sig else 'none'}")


# ── Analysis 3: Category RSA ──────────────────────────────────────────────

CAT_FIELDNAMES = ["layer", "condition", "rho", "p"]


def run_category_rsa(acts, out_dir, metric, n_perms, resume):
    """Category structure RSA within each condition."""
    out_csv = out_dir / "category_rsa_results.csv"
    nl = acts.shape[1]

    category_rdm = build_category_rdm(N_ITEMS, ITEMS_PER_CATEGORY)

    done = set()
    rows = []
    if resume and out_csv.exists():
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                r["layer"] = int(r["layer"])
                r["rho"] = float(r["rho"])
                r["p"] = float(r["p"])
                rows.append(r)
                done.add((r["layer"], r["condition"]))
        print(f"  Resuming category RSA: {len(done)} (layer, condition) pairs done")

    for cond in CONDITION_LABELS:
        cond_idx = np.array(get_condition_indices(cond))
        print(f"\n  --- Condition: {cond} ---")
        t0 = time.time()

        for layer in range(nl):
            if (layer, cond) in done:
                continue

            c_acts = acts[cond_idx, layer, :]
            neural_rdm_c = compute_rdm(c_acts, metric=metric)
            obs_rho, p_val, _ = permutation_test_category(
                neural_rdm_c, category_rdm,
                n_items=N_ITEMS, items_per_cat=ITEMS_PER_CATEGORY,
                n_perms=n_perms, seed=42 + layer
            )
            rows.append({
                "layer": layer,
                "condition": cond,
                "rho": round(obs_rho, 6),
                "p": round(p_val, 6),
            })

            # Save incrementally
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CAT_FIELDNAMES)
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: (
                    CONDITION_LABELS.index(r["condition"]), r["layer"])))

            if (layer + 1) % 10 == 0 or layer == nl - 1:
                elapsed = time.time() - t0
                print(f"    Layer {layer:2d}/{nl-1}: rho={obs_rho:+.4f}, "
                      f"p={p_val:.4f}  [{elapsed:.0f}s]")

    # FDR per condition
    for cond in CONDITION_LABELS:
        cond_rows = sorted([r for r in rows if r["condition"] == cond],
                           key=lambda r: r["layer"])
        pvals = np.array([r["p"] for r in cond_rows])
        fdr = fdr_correct(pvals)
        for i, r in enumerate(cond_rows):
            r["p_fdr"] = round(float(fdr[i]), 6)

    rows_sorted = sorted(rows, key=lambda r: (
        CONDITION_LABELS.index(r["condition"]), r["layer"]))
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CAT_FIELDNAMES + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    c1_rows = [r for r in rows if r["condition"] == "mental_state"]
    sig = [r["layer"] for r in c1_rows if r.get("p_fdr", 1) < 0.05]
    print(f"\n  Category RSA done. C1 sig layers (FDR<.05): {sig if sig else 'none'}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_model(args.model)

    vname = variant_name(args.metric, args.variant)
    print(f"{'='*60}")
    print(f"Variant RSA: {vname}")
    print(f"  metric={args.metric}, variant={args.variant}")
    print(f"  n_perms={args.n_perms}")
    print(f"{'='*60}")

    acts = load_activations(args.variant)
    out_dir = ensure_dir(data_dir("rsa") / vname)

    print(f"\n{'='*60}")
    print("Analysis 1: Simple RSA (Model A)")
    print(f"{'='*60}")
    run_simple_rsa(acts, out_dir, args.metric, args.n_perms, args.resume)

    print(f"\n{'='*60}")
    print("Analysis 2a: Partial RSA — Model A (Full Attribution)")
    print(f"{'='*60}")
    run_partial_rsa(acts, out_dir, args.metric, "A", "primary",
                    args.n_perms, args.resume)

    print(f"\n{'='*60}")
    print("Analysis 2b: Partial RSA — Model E (Subject-Optional)")
    print(f"{'='*60}")
    run_partial_rsa(acts, out_dir, args.metric, "E", "secondary",
                    args.n_perms, args.resume)

    print(f"\n{'='*60}")
    print("Analysis 3: Category Structure RSA")
    print(f"{'='*60}")
    run_category_rsa(acts, out_dir, args.metric, args.n_perms, args.resume)

    print(f"\nAll analyses complete for variant: {vname}")


if __name__ == "__main__":
    main()
