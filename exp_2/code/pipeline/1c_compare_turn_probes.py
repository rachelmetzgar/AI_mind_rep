#!/usr/bin/env python3
"""
Compare probe accuracy across turns 1-5.
Run after 1b_train_probes_turn_comparison.py finishes for all turns.

Usage:
    python 1c_compare_turn_probes.py --version labels

Env: behavior_env (no GPU needed)
"""

import os, sys, pickle, argparse, numpy as np
from pathlib import Path
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_version, add_version_argument

TURNS = [1, 2, 3, 4, 5]
PROBE_TYPES = ["metacognitive", "operational"]


def load_accuracy(turn, probe_type):
    """Load accuracy_summary.pkl for a given turn and probe type."""
    pkl_path = config.PATHS.probe_checkpoints / f"turn_{turn}" / probe_type / "accuracy_summary.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def main(args):
    set_version(args.version)

    print("=" * 80)
    print("PROBE ACCURACY COMPARISON ACROSS CONVERSATION TURNS")
    print("=" * 80)
    print(f"\nVersion: {args.version} (LLaMA-2-13B-chat)")
    print(f"Turn 1 = first exchange only, Turn 5 = full conversation (5 exchanges)")
    print(f"Turn 5 probes are the existing probes from data/probe_checkpoints/turn_5/\n")

    for probe_type in PROBE_TYPES:
        print(f"\n{'='*60}")
        print(f"  {probe_type.upper().replace('_', ' ')}")
        print(f"{'='*60}")

        all_accs = {}
        for turn in TURNS:
            summary = load_accuracy(turn, probe_type)
            if summary is None:
                print(f"  Turn {turn}: NOT AVAILABLE (job may still be running)")
                continue
            accs = summary["acc"]  # best test acc per layer
            all_accs[turn] = accs
            peak_layer = np.argmax(accs)
            print(f"  Turn {turn}: Mean={np.mean(accs):.3f}  Peak={np.max(accs):.3f} (layer {peak_layer})  "
                  f"Median={np.median(accs):.3f}  Min={np.min(accs):.3f}")

        if len(all_accs) < 2:
            print("  (Need at least 2 turns to compare)")
            continue

        # Pairwise comparisons vs turn 5
        if 5 in all_accs:
            print(f"\n  Pairwise vs Turn 5 (existing probes):")
            for turn in [1, 2, 3, 4]:
                if turn not in all_accs:
                    continue
                t, p = ttest_ind(all_accs[turn], all_accs[5])
                diff = np.mean(all_accs[turn]) - np.mean(all_accs[5])
                stars = '***' if p < .001 else ('**' if p < .01 else ('*' if p < .05 else 'ns'))
                better = "BETTER" if diff > 0 else "WORSE" if diff < 0 else "SAME"
                print(f"    Turn {turn} vs Turn 5: diff={diff:+.3f} ({better}), t={t:.2f}, p={p:.4f} {stars}")

        # Layer-by-layer comparison (turn with best peak)
        if len(all_accs) >= 2:
            print(f"\n  Layer-by-layer peak comparison:")
            best_turn = max(all_accs, key=lambda t: np.max(all_accs[t]))
            print(f"    Best peak accuracy: Turn {best_turn} ({np.max(all_accs[best_turn]):.3f})")

            # How many layers does each turn win?
            print(f"\n  Layers where each turn has highest accuracy:")
            for turn in sorted(all_accs.keys()):
                n_wins = sum(1 for layer in range(len(all_accs[turn]))
                            if all(all_accs[turn][layer] >= all_accs.get(t, [0]*41)[layer]
                                   for t in all_accs if t != turn))
                print(f"    Turn {turn}: wins {n_wins}/{len(all_accs[turn])} layers")

    print("\n" + "=" * 80)
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare probe accuracy across turns 1-5.")
    add_version_argument(parser)
    args = parser.parse_args()
    main(args)
