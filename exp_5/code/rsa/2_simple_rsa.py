#!/usr/bin/env python3
"""
Experiment 5, Phase 2: Simple RSA (Analysis 1)

Model A (full attribution) vs neural RDM at each layer.
Saves incrementally after each layer so partial results survive timeouts.

Output:
    results/{model}/rsa/data/simple_rsa_results.csv

Usage:
    python code/rsa/2_simple_rsa.py --model llama2_13b_chat

SLURM:
    sbatch code/rsa/slurm/2_simple_rsa.sh

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
)
from utils.rsa import (
    compute_rdm, build_model_rdms,
    permutation_test_simple, fdr_correct,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple RSA (Analysis 1)")
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    act_path = data_dir("activations") / "activations_last_token.npz"
    print(f"Loading activations from {act_path}")
    acts = np.load(act_path)["activations"].astype(np.float32)
    nl = acts.shape[1]

    out_dir = ensure_dir(data_dir("rsa"))
    out_csv = out_dir / "simple_rsa_results.csv"
    fieldnames = ["layer", "rho", "p"]

    # Resume support: load existing results
    done_layers = set()
    rows = []
    if args.resume and out_csv.exists():
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["layer"] = int(r["layer"])
                r["rho"] = float(r["rho"])
                r["p"] = float(r["p"])
                rows.append(r)
                done_layers.add(r["layer"])
        print(f"Resuming: {len(done_layers)} layers already done")

    model_rdms = build_model_rdms(N_ITEMS, N_CONDITIONS)
    model_A = model_rdms["A"]

    t0 = time.time()
    for layer in range(nl):
        if layer in done_layers:
            continue

        neural_rdm = compute_rdm(acts[:, layer, :], metric="correlation")
        obs_rho, p_val, _ = permutation_test_simple(
            neural_rdm, model_A, n_perms=args.n_perms, seed=42 + layer
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

    # FDR correction — add p_fdr column to final file
    rows_sorted = sorted(rows, key=lambda r: r["layer"])
    pvals = np.array([r["p"] for r in rows_sorted])
    fdr = fdr_correct(pvals)
    for i, r in enumerate(rows_sorted):
        r["p_fdr"] = round(float(fdr[i]), 6)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"\nDone. Results: {out_csv}")
    sig_layers = [r["layer"] for r in rows_sorted if r.get("p_fdr", 1) < 0.05]
    print(f"Significant layers (FDR<.05): {sig_layers if sig_layers else 'none'}")


if __name__ == "__main__":
    main()
