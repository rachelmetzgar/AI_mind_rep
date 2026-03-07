#!/usr/bin/env python3
"""
Experiment 3, Phase 3: Cross-Dimension Summary

Collects alignment results from all 13 concept dimensions and produces:
    - Summary table (JSON + printed)
    - Heatmap of alignment (probe_to_2b) across dimensions × layers
    - Bar chart of max |cosine similarity| per dimension

Run AFTER all array jobs complete:
    python 3_summarize_alignment.py

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path for imports
from config import config, set_version, add_version_argument, add_turn_argument

# Import dimension registry from pipeline script
from importlib.util import spec_from_file_location, module_from_spec
_s1_spec = spec_from_file_location(
    "script1",
    os.path.join(os.path.dirname(__file__), "1_elicit_concept_vectors.py"),
)
_s1_mod = module_from_spec(_s1_spec)
_s1_spec.loader.exec_module(_s1_mod)
DIMENSION_REGISTRY = _s1_mod.DIMENSION_REGISTRY

PROBE_ROOT = str(config.RESULTS.concept_probes_data)

# Version-dependent paths — set by _init_paths() after set_version()
ALIGN_ROOT = None
OUTPUT_DIR = None


def _init_paths():
    """Initialize version-dependent paths after set_version() has been called."""
    global ALIGN_ROOT, OUTPUT_DIR
    from config import _active_turn
    turn_dir = config.RESULTS.alignment_versions / f"turn_{_active_turn}"
    ALIGN_ROOT = str(turn_dir)
    OUTPUT_DIR = str(turn_dir / "summary")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Exp 3: Cross-dimension alignment summary"
    )
    add_version_argument(parser)
    add_turn_argument(parser)
    args = parser.parse_args()

    # Set version and initialize paths
    set_version(args.version, turn=args.turn)
    _init_paths()
    print(f"Version: {args.version}")
    print(f"Turn: {args.turn}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}

    for dim_id, (_, dim_name) in sorted(DIMENSION_REGISTRY.items()):
        entry = {"dim_id": dim_id, "dim_name": dim_name}

        # Probe accuracy
        acc_path = os.path.join(PROBE_ROOT, dim_name, "accuracy_summary.pkl")
        if os.path.isfile(acc_path):
            with open(acc_path, "rb") as f:
                acc = pickle.load(f)
            entry["probe_acc_mean"] = float(np.mean(acc["acc"]))
            entry["probe_acc_max"] = float(np.max(acc["acc"]))
            entry["probe_acc_max_layer"] = int(np.argmax(acc["acc"]))
        else:
            entry["probe_acc_mean"] = None
            print(f"  [SKIP] No probe accuracy for {dim_name}")

        # Alignment with control and reading probes
        for probe_type in ["operational", "metacognitive"]:
            align_path = os.path.join(
                ALIGN_ROOT, dim_name, probe_type, "alignment_results.json"
            )
            if os.path.isfile(align_path):
                with open(align_path) as f:
                    res = json.load(f)
                valid_pp = [v for v in res["probe_to_2b"] if v is not None]
                valid_m = [v for v in res["mean_to_2b"] if v is not None]
                if valid_pp:
                    entry[f"{probe_type}_probe_max_cos"] = float(
                        max(valid_pp, key=abs)
                    )
                    entry[f"{probe_type}_mean_max_cos"] = float(
                        max(valid_m, key=abs)
                    )
                    entry[f"{probe_type}_probe_mean_cos"] = float(
                        np.mean(valid_pp)
                    )
                    # Layer-wise values for heatmap
                    entry[f"{probe_type}_layers"] = res["layers"]
                    entry[f"{probe_type}_probe_to_2b"] = res["probe_to_2b"]
                else:
                    entry[f"{probe_type}_probe_max_cos"] = None
            else:
                entry[f"{probe_type}_probe_max_cos"] = None
                print(f"  [SKIP] No alignment for {dim_name}/{probe_type}")

        summary[dim_id] = entry

    # Save summary JSON
    with open(os.path.join(OUTPUT_DIR, "cross_dimension_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # === Print table ===
    print(f"\n{'='*90}")
    print(f"EXPERIMENT 3: Cross-Dimension Alignment Summary")
    print(f"{'='*90}")
    print(f"\n{'ID':<4} {'Dimension':<28} {'Acc':<8} "
          f"{'Oper max|cos|':<16} {'Meta max|cos|':<16}")
    print("-" * 80)

    for dim_id in sorted(summary.keys()):
        e = summary[dim_id]
        acc_str = f"{e['probe_acc_max']:.3f}" if e.get("probe_acc_mean") else "N/A"
        ctrl_str = (f"{e['operational_probe_max_cos']:.4f}"
                    if e.get("operational_probe_max_cos") is not None else "N/A")
        read_str = (f"{e['metacognitive_probe_max_cos']:.4f}"
                    if e.get("metacognitive_probe_max_cos") is not None else "N/A")
        print(f"{dim_id:<4} {e['dim_name']:<28} {acc_str:<8} "
              f"{ctrl_str:<16} {read_str:<16}")

    # === Bar chart: max |cos| per dimension ===
    try:
        dim_ids = sorted(summary.keys())
        dim_labels = [summary[d]["dim_name"].replace("_", "\n") for d in dim_ids]
        ctrl_vals = []
        read_vals = []
        for d in dim_ids:
            c = summary[d].get("operational_probe_max_cos")
            r = summary[d].get("metacognitive_probe_max_cos")
            ctrl_vals.append(abs(c) if c is not None else 0)
            read_vals.append(abs(r) if r is not None else 0)

        x = np.arange(len(dim_ids))
        width = 0.35

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.bar(x - width / 2, ctrl_vals, width, label="Operational probe",
               color="tab:red", alpha=0.8)
        ax.bar(x + width / 2, read_vals, width, label="Metacognitive probe",
               color="tab:blue", alpha=0.8)
        ax.set_xlabel("Concept Dimension")
        ax.set_ylabel("Max |Cosine Similarity| with Exp 2b")
        ax.set_title("Experiment 3: Alignment of Concept Dimensions with Conversational Probes")
        ax.set_xticks(x)
        ax.set_xticklabels(dim_labels, fontsize=7, ha="center")
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "alignment_bar_chart.png"),
                    dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nSaved bar chart to {OUTPUT_DIR}/alignment_bar_chart.png")
    except Exception as e:
        print(f"[WARN] Bar chart failed: {e}")

    # === Heatmap: probe_to_2b across dimensions × layers (reading probe) ===
    try:
        # Collect layer-wise data for reading probe
        all_layers = None
        heatmap_data = []
        heatmap_labels = []

        for dim_id in dim_ids:
            e = summary[dim_id]
            key = "metacognitive_probe_to_2b"
            if key in e and e[key] is not None:
                vals = [v if v is not None else 0.0 for v in e[key]]
                if all_layers is None:
                    all_layers = e["metacognitive_layers"]
                heatmap_data.append(vals)
                heatmap_labels.append(e["dim_name"])

        if heatmap_data:
            arr = np.array(heatmap_data)
            fig, ax = plt.subplots(figsize=(16, 8))
            im = ax.imshow(arr, aspect="auto", cmap="RdBu_r",
                           vmin=-1, vmax=1)
            ax.set_yticks(range(len(heatmap_labels)))
            ax.set_yticklabels(heatmap_labels, fontsize=8)
            if all_layers:
                tick_pos = list(range(0, len(all_layers), 5))
                ax.set_xticks(tick_pos)
                ax.set_xticklabels([all_layers[i] for i in tick_pos])
            ax.set_xlabel("Layer")
            ax.set_ylabel("Concept Dimension")
            ax.set_title("Concept Probe ↔ Reading Probe Alignment (cosine similarity)")
            plt.colorbar(im, ax=ax, label="Cosine Similarity")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "alignment_heatmap_reading.png"),
                        dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved heatmap to {OUTPUT_DIR}/alignment_heatmap_reading.png")
    except Exception as e:
        print(f"[WARN] Heatmap failed: {e}")

    print(f"\n✅ Summary complete. Outputs in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()