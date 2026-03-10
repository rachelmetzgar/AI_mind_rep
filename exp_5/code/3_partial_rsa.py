#!/usr/bin/env python3
"""
Experiment 5, Phase 3: Partial RSA (Analysis 2)

Multiple regression: neural RDM ~ hypothesis + confounds (B,C,D,F,G,H).
Runs both primary (Model A) and secondary (Model E) analyses.
Saves incrementally after each layer.

Output:
    results/{model}/rsa/data/
        partial_rsa_primary_results.csv     (Model A as hypothesis)
        partial_rsa_secondary_results.csv   (Model E as hypothesis)

Usage:
    python code/3_partial_rsa.py --model llama2_13b_chat
    python code/3_partial_rsa.py --model llama2_13b_chat --analysis primary
    python code/3_partial_rsa.py --model llama2_13b_chat --analysis secondary

SLURM:
    sbatch code/slurm/3_partial_rsa.sh

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
)
from utils.rsa import (
    compute_rdm, build_model_rdms,
    permutation_test_partial, fdr_correct,
)


CONFOUND_KEYS = ["B", "C", "D", "F", "G", "H"]
FIELDNAMES = ["layer", "model", "beta", "semi_partial_r", "p"]


def parse_args():
    parser = argparse.ArgumentParser(description="Partial RSA (Analysis 2)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--analysis", choices=["both", "primary", "secondary"],
                        default="both", help="Which analysis to run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    return parser.parse_args()


def load_existing(csv_path):
    """Load existing results, return rows and set of completed layers."""
    done_layers = set()
    rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["layer"] = int(r["layer"])
                r["beta"] = float(r["beta"])
                r["semi_partial_r"] = float(r["semi_partial_r"])
                r["p"] = float(r["p"])
                rows.append(r)
                done_layers.add(r["layer"])
    return rows, done_layers


def save_with_fdr(rows, csv_path, all_keys):
    """Sort, add FDR, save."""
    rows_sorted = sorted(rows, key=lambda r: (r["layer"], all_keys.index(r["model"])
                                               if r["model"] in all_keys else 99))
    # FDR per model
    for model_key in all_keys:
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


def run_analysis(acts, model_rdms, hyp_key, out_csv, n_perms, resume):
    """Run partial RSA for one hypothesis model."""
    nl = acts.shape[1]
    all_keys = [hyp_key] + CONFOUND_KEYS

    rows, done_layers = [], set()
    if resume:
        rows, done_layers = load_existing(out_csv)
        if done_layers:
            print(f"  Resuming: {len(done_layers)} layers already done")

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric="correlation")
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

        # Save incrementally (without FDR — added at end)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
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
    save_with_fdr(rows, out_csv, all_keys)
    print(f"  Saved: {out_csv}")

    # Report
    hyp_rows = [r for r in rows if r["model"] == hyp_key]
    sig = [r["layer"] for r in hyp_rows if r.get("p_fdr", 1) < 0.05]
    print(f"  Significant layers for {hyp_key} (FDR<.05): {sig if sig else 'none'}")


def main():
    args = parse_args()
    set_model(args.model)

    act_path = data_dir("activations") / "activations_last_token.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"].astype(np.float32)
    print(f"  Shape: {acts.shape}")

    out_dir = ensure_dir(data_dir("rsa"))
    model_rdms = build_model_rdms(N_ITEMS, N_CONDITIONS)

    if args.analysis in ("both", "primary"):
        print(f"\n{'='*60}")
        print("Analysis 2a: Partial RSA — Model A (Full Attribution)")
        print(f"{'='*60}")
        run_analysis(acts, model_rdms, "A",
                     out_dir / "partial_rsa_primary_results.csv",
                     args.n_perms, args.resume)

    if args.analysis in ("both", "secondary"):
        print(f"\n{'='*60}")
        print("Analysis 2b: Partial RSA — Model E (Subject-Optional)")
        print(f"{'='*60}")
        run_analysis(acts, model_rdms, "E",
                     out_dir / "partial_rsa_secondary_results.csv",
                     args.n_perms, args.resume)

    print("\nDone.")


if __name__ == "__main__":
    main()
