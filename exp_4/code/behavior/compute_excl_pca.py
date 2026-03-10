#!/usr/bin/env python3
"""
Re-run PCA+varimax excluding fetus and god, save results alongside the
full-set results.

Reads:  results/{model}/behavior/{condition}/data/
            pairwise_character_means.npz  OR  individual_rating_matrix.npz
Writes: results/{model}/behavior/{condition}/data/
            {method}_pca_results_excl_fetus_god.npz

Usage:
    python behavior/compute_excl_pca.py --model llama2_13b_base

Env: llama2_env (CPU-only, login node OK)
"""
import os, sys, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_model, add_model_argument, data_dir
from utils.utils import run_pca_varimax

EXCLUSION_SETS = {
    "excl_fetus_god": {"fetus", "god"},
    "excl_fetus_god_dead": {"fetus", "god", "dead_woman"},
}


def run_condition(model_key, method, condition, excl_name, exclude):
    set_model(model_key)
    ddir = str(data_dir("behavior", condition))

    # Load means matrix
    if method == "pairwise":
        means_path = os.path.join(ddir, "pairwise_character_means.npz")
        if not os.path.exists(means_path):
            print(f"  Not found: {means_path}")
            return
        d = np.load(means_path)
        means = d["means"]  # (n_caps, n_entities)
        entity_keys = list(d["entity_keys"])
        capacity_keys = list(d["capacity_keys"])
    else:
        mat_path = os.path.join(ddir, "individual_rating_matrix.npz")
        if not os.path.exists(mat_path):
            print(f"  Not found: {mat_path}")
            return
        d = np.load(mat_path)
        means = d["rating_matrix"]  # (n_caps, n_entities)
        entity_keys = list(d["entity_keys"])
        capacity_keys = list(d["capacity_keys"])

    # Exclude entities
    keep_mask = np.array([k not in exclude for k in entity_keys])
    excl_keys = [k for k, m in zip(entity_keys, keep_mask) if m]
    excl_means = means[:, keep_mask]

    print(f"  Full: {len(entity_keys)} entities → Excl: {len(excl_keys)} entities")
    print(f"  Excluded: {[k for k in entity_keys if k in exclude]}")

    # Run PCA
    pca_results = run_pca_varimax(excl_means)

    # Save
    out_path = os.path.join(ddir, f"{method}_pca_results_{excl_name}.npz")
    np.savez(out_path,
             rotated_loadings=pca_results["rotated_loadings"],
             unrotated_loadings=pca_results.get("unrotated_loadings",
                                                 pca_results["rotated_loadings"]),
             factor_scores_raw=pca_results["factor_scores_raw"],
             factor_scores_01=pca_results["factor_scores_01"],
             eigenvalues=pca_results["eigenvalues"],
             explained_var_ratio=pca_results["explained_var_ratio"],
             entity_keys=np.array(excl_keys),
             capacity_keys=np.array(capacity_keys))
    print(f"  Wrote: {out_path}")

    # Print summary
    ev = pca_results["eigenvalues"]
    expl = pca_results["explained_var_ratio"]
    n_ret = int(np.sum(ev > 1.0))
    print(f"  Eigenvalues: {ev[:5]}")
    print(f"  Retained: {n_ret}, explaining {np.sum(expl[:n_ret])*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run PCA excluding fetus + god")
    add_model_argument(parser)
    args = parser.parse_args()

    for excl_name, exclude in EXCLUSION_SETS.items():
        for condition in ["with_self", "without_self"]:
            for method in ["pairwise", "individual"]:
                print(f"\n{'='*50}")
                print(f"  {excl_name} / {method} / {condition}")
                print(f"{'='*50}")
                run_condition(args.model, method, condition, excl_name, exclude)


if __name__ == "__main__":
    main()
