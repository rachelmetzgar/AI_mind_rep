#!/usr/bin/env python3
"""
Experiment 5: Positional 5-predictor RSA (verb and object token positions)

Runs the same reduced 1-4 RSA (5 predictors A-E, C1-C4 only) at verb and
object token positions using multi-position activations.

Input:  activations_multipos.npz  (336, 3, 41, 5120)  pos: 0=verb, 1=object, 2=period
Output: results/{model}/rsa/data/5_predictors/
            5_predictors_rsa_corr_verb.csv
            5_predictors_rsa_corr_object.csv
            model_correlations_verb.npz, model_correlations_object.npz
            vif_check_verb.json, vif_check_object.json

Usage:
    python code/rsa/5_predictors/1b_positional_rsa.py --model llama2_13b_chat
    python code/rsa/5_predictors/1b_positional_rsa.py --model llama2_13b_chat --position verb
    python code/rsa/5_predictors/1b_positional_rsa.py --model llama2_13b_chat --position object --resume

SLURM:
    sbatch code/rsa/5_predictors/slurm/1b_positional_rsa.sh

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    set_model, add_model_argument, data_dir, ensure_dir,
    N_ITEMS, N_PERMUTATIONS, POSITION_LABELS,
)

# Import core RSA functions from the existing script (filename starts with digit)
_base_script = Path(__file__).resolve().parent / "1_reduced_1_4_rsa.py"
_spec = importlib.util.spec_from_file_location("reduced_1_4_rsa", _base_script)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

filter_activations = _mod.filter_activations
build_reduced_model_rdms = _mod.build_reduced_model_rdms
run_analysis = _mod.run_analysis
compute_vifs = _mod.compute_vifs
model_correlations = _mod.model_correlations
ALL_MODEL_KEYS = _mod.ALL_MODEL_KEYS
N_CONDS_REDUCED = _mod.N_CONDS_REDUCED
N_SENTENCES_REDUCED = _mod.N_SENTENCES_REDUCED

POSITION_MAP = {"verb": 0, "object": 1, "period": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Positional 5-predictor RSA (verb/object token positions)"
    )
    add_model_argument(parser)
    parser.add_argument("--n-perms", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--position", type=str, nargs="+", default=["verb", "object"],
        choices=["verb", "object"],
        help="Which positions to run (default: both verb and object)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_model(args.model)

    # Load multi-position activations (336, 3, 41, 5120)
    act_path = data_dir("activations") / "activations_multipos.npz"
    print(f"Loading multi-position activations from {act_path}")
    acts_full = np.load(act_path)["activations"].astype(np.float32)
    print(f"  Full shape: {acts_full.shape}")
    assert acts_full.ndim == 4, f"Expected 4D array, got {acts_full.ndim}D"

    out_dir = ensure_dir(data_dir("rsa") / "5_predictors")

    # Build model RDMs (position-independent)
    model_rdms = build_reduced_model_rdms(N_ITEMS, N_CONDS_REDUCED)
    for k, rdm in model_rdms.items():
        assert rdm.shape == (N_SENTENCES_REDUCED, N_SENTENCES_REDUCED), \
            f"Model {k} RDM shape {rdm.shape} != ({N_SENTENCES_REDUCED}, {N_SENTENCES_REDUCED})"

    for pos_name in args.position:
        pos_idx = POSITION_MAP[pos_name]
        suffix = f"_{pos_name}"

        print(f"\n{'='*60}")
        print(f"Position: {pos_name} (index {pos_idx})")
        print(f"{'='*60}")

        # Extract this position: (336, 41, 5120)
        acts_pos = acts_full[:, pos_idx, :, :]
        print(f"  Position slice shape: {acts_pos.shape}")

        # Filter to C1-C4: (224, 41, 5120)
        acts = filter_activations(acts_pos)
        print(f"  Reduced shape (C1-C4 only): {acts.shape}")
        assert acts.shape[0] == N_SENTENCES_REDUCED

        # VIF check (same across positions, but save per-position for consistency)
        vifs = compute_vifs(model_rdms, ALL_MODEL_KEYS)
        print(f"\nVIF check ({pos_name}):")
        for k in ALL_MODEL_KEYS:
            flag = " *** WARNING" if vifs[k] > 10 else (" * caution" if vifs[k] > 5 else "")
            print(f"  Model {k}: VIF = {vifs[k]:.2f}{flag}")

        vif_path = out_dir / f"vif_check{suffix}.json"
        with open(vif_path, "w") as f:
            json.dump(vifs, f, indent=2)
        print(f"  VIFs saved: {vif_path}")

        # Model correlations (also position-independent)
        corr_matrix = model_correlations(model_rdms, ALL_MODEL_KEYS)
        np.savez(out_dir / f"model_correlations{suffix}.npz",
                 corr_matrix=corr_matrix, model_keys=ALL_MODEL_KEYS)

        # Run RSA
        print(f"\nReduced 1-4 RSA — Correlation distance — {pos_name} position")
        run_analysis(
            acts, model_rdms, "correlation",
            out_dir / f"5_predictors_rsa_corr{suffix}.csv",
            args.n_perms, args.resume,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
