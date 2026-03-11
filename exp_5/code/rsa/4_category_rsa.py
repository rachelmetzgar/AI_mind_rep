#!/usr/bin/env python3
"""
Experiment 5, Phase 4: Category Structure RSA (Analysis 3)

Within each condition, test whether 7 mental state verb categories
produce category-structured representations. Saves incrementally.

Output:
    results/{model}/rsa/data/category_rsa_results.csv

Usage:
    python code/rsa/4_category_rsa.py --model llama2_13b_chat

SLURM:
    sbatch code/rsa/slurm/4_category_rsa.sh

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
    CONDITION_LABELS, ITEMS_PER_CATEGORY,
)
from stimuli import get_condition_indices
from utils.rsa import (
    compute_rdm, build_category_rdm,
    permutation_test_category, fdr_correct,
)


FIELDNAMES = ["layer", "condition", "rho", "p"]


def parse_args():
    parser = argparse.ArgumentParser(description="Category Structure RSA (Analysis 3)")
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
    out_csv = out_dir / "category_rsa_results.csv"

    category_rdm = build_category_rdm(N_ITEMS, ITEMS_PER_CATEGORY)

    # Resume support
    done = set()  # (layer, condition) tuples
    rows = []
    if args.resume and out_csv.exists():
        with open(out_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["layer"] = int(r["layer"])
                r["rho"] = float(r["rho"])
                r["p"] = float(r["p"])
                rows.append(r)
                done.add((r["layer"], r["condition"]))
        print(f"Resuming: {len(done)} (layer, condition) pairs already done")

    for cond in CONDITION_LABELS:
        cond_idx = np.array(get_condition_indices(cond))
        print(f"\n--- Condition: {cond} ---")
        t0 = time.time()

        for layer in range(nl):
            if (layer, cond) in done:
                continue

            c_acts = acts[cond_idx, layer, :]
            neural_rdm_c = compute_rdm(c_acts, metric="correlation")
            obs_rho, p_val, _ = permutation_test_category(
                neural_rdm_c, category_rdm,
                n_items=N_ITEMS, items_per_cat=ITEMS_PER_CATEGORY,
                n_perms=args.n_perms, seed=42 + layer
            )
            rows.append({
                "layer": layer,
                "condition": cond,
                "rho": round(obs_rho, 6),
                "p": round(p_val, 6),
            })

            # Save incrementally
            with open(out_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                writer.writeheader()
                writer.writerows(sorted(rows, key=lambda r: (
                    CONDITION_LABELS.index(r["condition"]), r["layer"])))

            if (layer + 1) % 10 == 0 or layer == nl - 1:
                elapsed = time.time() - t0
                print(f"  Layer {layer:2d}/{nl-1}: rho={obs_rho:+.4f}, "
                      f"p={p_val:.4f}  [{elapsed:.0f}s]")

    # FDR per condition
    for cond in CONDITION_LABELS:
        cond_rows = [r for r in rows if r["condition"] == cond]
        cond_rows_sorted = sorted(cond_rows, key=lambda r: r["layer"])
        pvals = np.array([r["p"] for r in cond_rows_sorted])
        fdr = fdr_correct(pvals)
        for i, r in enumerate(cond_rows_sorted):
            r["p_fdr"] = round(float(fdr[i]), 6)

    rows_sorted = sorted(rows, key=lambda r: (
        CONDITION_LABELS.index(r["condition"]), r["layer"]))
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES + ["p_fdr"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print(f"\nDone. Results: {out_csv}")
    c1_rows = [r for r in rows if r["condition"] == "mental_state"]
    sig = [r["layer"] for r in c1_rows if r.get("p_fdr", 1) < 0.05]
    print(f"C1 significant layers (FDR<.05): {sig if sig else 'none'}")


if __name__ == "__main__":
    main()
