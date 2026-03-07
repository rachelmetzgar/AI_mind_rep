#!/usr/bin/env python3
"""
Pre-compute concept-aligned layers for each contrast dimension.

For each dimension, reads alignment.npz and extracts per-layer |cosine
similarity| between the concept vector and the exp2 metacognitive probe
weight vector. Selects top 15 layers by |cosine| and saves a JSON lookup.

Output:
    exp_3/results/{model}/{version}/alignment/concept_aligned_layers.json
    Format: {dim_name: [layer_idx_1, layer_idx_2, ...]}
    (sorted by |cosine| descending)

This is a lightweight file-I/O script — safe to run on login node.

Usage:
    python 2h_concept_aligned_layers.py --version balanced_gpt
    python 2h_concept_aligned_layers.py --version balanced_gpt --turn 5 --top_k 15

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

from config import (
    config, set_version, add_version_argument, add_turn_argument, ensure_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute concept-aligned layers per dimension."
    )
    add_version_argument(parser)
    add_turn_argument(parser)
    parser.add_argument(
        "--top_k", type=int, default=15,
        help="Number of top layers to select per dimension (default: 15).",
    )
    args = parser.parse_args()

    set_version(args.version, turn=args.turn)

    # Alignment results for this version/turn
    alignment_root = (
        config.RESULTS.alignment_versions
        / f"turn_{args.turn}" / "contrasts" / "raw"
    )
    if not alignment_root.exists():
        print(f"[ERROR] Alignment directory not found: {alignment_root}")
        sys.exit(1)

    # Scan for dimension directories
    dim_dirs = sorted([
        d for d in os.listdir(alignment_root)
        if os.path.isdir(alignment_root / d) and d[0].isdigit()
    ])
    print(f"Found {len(dim_dirs)} dimensions in {alignment_root}")

    result = {}
    for dim_name in dim_dirs:
        npz_path = alignment_root / dim_name / "alignment.npz"
        if not npz_path.exists():
            print(f"  [SKIP] {dim_name}: no alignment.npz")
            continue

        data = np.load(npz_path, allow_pickle=True)

        # reading_per_layer is a JSON-encoded dict: {layer_str: {cosine, r_squared}}
        rpl_raw = data["reading_per_layer"]
        rpl = json.loads(str(rpl_raw))

        # Extract |cosine| per layer
        layer_cosines = []
        for layer_str, metrics in rpl.items():
            layer_idx = int(layer_str)
            cosine = abs(metrics["cosine"])
            layer_cosines.append((layer_idx, cosine))

        # Sort by |cosine| descending, take top_k
        layer_cosines.sort(key=lambda x: x[1], reverse=True)
        top_layers = [idx for idx, _ in layer_cosines[:args.top_k]]

        result[dim_name] = top_layers

        top_cos = layer_cosines[0][1] if layer_cosines else 0
        bot_cos = layer_cosines[min(args.top_k - 1, len(layer_cosines) - 1)][1] if layer_cosines else 0
        print(f"  {dim_name}: top {args.top_k} layers = {sorted(top_layers)} "
              f"(|cos| range: {bot_cos:.4f} - {top_cos:.4f})")

    # Save
    out_path = config.RESULTS.alignment / "concept_aligned_layers.json"
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved concept-aligned layers for {len(result)} dimensions to:")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()
