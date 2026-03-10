#!/usr/bin/env python3
"""
Experiment 3: Shapes Flip Sanity Check

Negates the 15_shapes concept vector so that angular maps to the
+1 ("human") pole and round maps to the -1 ("ai") pole.  If the
behavioral effects reverse (i.e. the "human" condition now looks
more formal/angular and the "ai" condition looks warmer/round),
we confirm the effects come from shape semantics in activation
space, not from an artifact of the +1/-1 steering procedure.

Modes:
    setup   : Create flipped concept vector NPZ (login node, no GPU)
    analyze : Run behavioral analysis on steering results (login node, behavior_env)

Steering itself is run via SLURM — see slurm/4c_shapes_flip.sh

Usage:
    python 4c_shapes_flip_test.py --mode setup   --version balanced_gpt
    python 4c_shapes_flip_test.py --mode analyze --version balanced_gpt

Env: llama2_env (setup) or behavior_env (analyze)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

from config import config, set_version, add_version_argument

FLIP_DIM_NAME = "29_shapes_flip"
ORIG_DIM_NAME = "15_shapes"


def run_setup():
    """Negate the shapes concept vector and save as dim 115."""
    orig_dir = config.RESULTS.concept_activations_contrasts / ORIG_DIM_NAME
    flip_dir = config.RESULTS.concept_activations_contrasts / FLIP_DIM_NAME

    vec_path = orig_dir / "concept_vector_per_layer.npz"
    if not vec_path.exists():
        print(f"[ERROR] Original shapes vector not found: {vec_path}")
        sys.exit(1)

    data = np.load(vec_path)
    orig_direction = data["concept_direction"]
    orig_norms = data["norms"]

    flipped_direction = -orig_direction

    os.makedirs(flip_dir, exist_ok=True)
    out_path = flip_dir / "concept_vector_per_layer.npz"
    np.savez_compressed(
        out_path,
        concept_direction=flipped_direction,
        norms=orig_norms,  # norms unchanged by negation
    )

    print(f"Saved flipped concept vector to {out_path}")
    print(f"  Shape: {flipped_direction.shape}")
    print(f"  Original direction: round - angular (round = +1)")
    print(f"  Flipped direction:  angular - round (angular = +1)")
    print(f"\nNow run steering via SLURM:")
    print(f"  sbatch slurm/4c_shapes_flip.sh")


def run_analyze(version):
    """Run behavioral analysis on the flipped steering results."""
    # Import the behavioral analysis main function
    # We re-use the existing 4a script by calling it as a subprocess
    import subprocess

    v1_root = config.RESULTS.concept_steering / "v1"
    flip_results = v1_root / FLIP_DIM_NAME
    if not flip_results.exists():
        print(f"[ERROR] Flipped steering results not found: {flip_results}")
        print("Run steering first: sbatch slurm/4c_shapes_flip.sh")
        sys.exit(1)

    code_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable, str(code_dir / "4a_concept_steering_behavior.py"),
        "--version", version,
        "--dim_ids", FLIP_DIM_NAME,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Print comparison hint
    print(f"\n{'='*70}")
    print("FLIP TEST COMPARISON")
    print(f"{'='*70}")
    print(f"Original (round=human, angular=ai):")
    print(f"  Results in: {v1_root / ORIG_DIM_NAME}")
    print(f"Flipped (angular=human, round=ai):")
    print(f"  Results in: {flip_results}")
    print(f"\nIf effects reverse (flipped 'human' now looks colder/formal),")
    print(f"the original effects are driven by shape semantics, not steering artifacts.")


def main():
    parser = argparse.ArgumentParser(
        description="Exp 3: Shapes flip sanity check."
    )
    add_version_argument(parser)
    parser.add_argument(
        "--mode", required=True, choices=["setup", "analyze"],
        help="'setup' to create flipped vector, 'analyze' to run behavioral analysis.",
    )
    args = parser.parse_args()
    set_version(args.version)

    if args.mode == "setup":
        run_setup()
    elif args.mode == "analyze":
        run_analyze(args.version)


if __name__ == "__main__":
    main()
