#!/usr/bin/env python3
"""
Experiment 3, Phase 2b: Residual Alignment Analysis

For each concept dimension (1-13), computes alignment with Exp 2b probes
in two ways:
    1. Raw alignment (already computed in Phase 2 — loaded from JSON)
    2. Residual alignment: after projecting OUT the generic entity vector
       (dim 0), what dimension-specific content remains, and does it align?

The generic entity vector captures the shared "human vs AI" signal common
to all dimensions. The residual captures what each dimension adds beyond
entity type — e.g., the "phenomenal experience" part of "human phenomenal
experience."

Requires:
    - Generic entity baseline (dim 0) must be extracted first:
        python 1_elicit_concept_vectors.py --dim_id 0
    - All target dimensions (1-13) must be extracted

Output:
    data/alignment/residual_analysis/
        residual_alignment_summary.csv
        residual_vs_raw_comparison.png
        per_dimension/ {dim_name}_residual.json

Usage:
    python 2b_residual_alignment.py
    python 2b_residual_alignment.py --dim_ids 1 5 7 11

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from src.probes import LinearProbeClassification

# Import dimension registry
from importlib.util import spec_from_file_location, module_from_spec
_s1_spec = spec_from_file_location(
    "script1",
    os.path.join(os.path.dirname(__file__), "1_elicit_concept_vectors.py"),
)
_s1_mod = module_from_spec(_s1_spec)
_s1_spec.loader.exec_module(_s1_mod)
DIMENSION_REGISTRY = _s1_mod.DIMENSION_REGISTRY


# ========================== CONFIG ========================== #

CONCEPT_ROOT = "data/concept_activations"
CONCEPT_PROBE_ROOT = "data/concept_probes"
OUTPUT_ROOT = "data/alignment/residual_analysis"
INPUT_DIM = 5120

# Exp 2b probe locations
EXP2B_PROBE_ROOT = (
    "/jukebox/graziano/rachel/ai_mind_rep/exp_2b/llama_exp_2b-13B-chat/"
    "data/probe_checkpoints"
)
EXP2B_PROBE_TYPES = {
    "control_probe": os.path.join(EXP2B_PROBE_ROOT, "control_probe"),
    "reading_probe": os.path.join(EXP2B_PROBE_ROOT, "reading_probe"),
}


# ========================== VECTOR LOADING ========================== #

def load_concept_vector(dim_name):
    """Load the mean-difference concept vector for a dimension."""
    path = os.path.join(CONCEPT_ROOT, dim_name, "concept_vector_per_layer.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Concept vector not found: {path}")
    data = np.load(path)
    # concept_direction shape: (n_layers, hidden_dim)
    return torch.from_numpy(data["concept_direction"]).float()


def load_concept_probe_weights(dim_name):
    """Load trained concept probe weight vectors for a dimension."""
    probe_dir = os.path.join(CONCEPT_PROBE_ROOT, dim_name)
    weights = {}
    if not os.path.isdir(probe_dir):
        return weights
    for fname in sorted(os.listdir(probe_dir)):
        if not fname.startswith("concept_probe_layer_") or not fname.endswith(".pth"):
            continue
        if fname.endswith("_final.pth"):
            continue
        layer_idx = int(fname.split("_layer_")[-1].split(".pth")[0])
        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        probe.load_state_dict(torch.load(os.path.join(probe_dir, fname), map_location="cpu"))
        with torch.no_grad():
            weights[layer_idx] = probe.proj[0].weight.clone()  # (1, hidden_dim)
    return weights


def load_exp2b_weights(probe_type_dir):
    """Load Exp 2b probe weights."""
    weights = {}
    if not os.path.isdir(probe_type_dir):
        print(f"  [WARN] Exp 2b probe dir not found: {probe_type_dir}")
        return weights
    for fname in sorted(os.listdir(probe_type_dir)):
        if not fname.endswith(".pth") or not fname.startswith("human_ai_probe_at_layer_"):
            continue
        if fname.endswith("_final.pth"):
            continue
        layer_idx = int(fname.split("_layer_")[-1].split(".pth")[0])
        probe = LinearProbeClassification(
            device="cpu", probe_class=1, input_dim=INPUT_DIM, logistic=True,
        )
        probe.load_state_dict(torch.load(os.path.join(probe_type_dir, fname), map_location="cpu"))
        with torch.no_grad():
            weights[layer_idx] = probe.proj[0].weight.clone()
    return weights


# ========================== PROJECTION ========================== #

def project_out(vector, direction):
    """
    Project `direction` out of `vector`.
    Returns the component of `vector` orthogonal to `direction`.

    Both inputs: (1, hidden_dim) or (hidden_dim,)
    """
    v = vector.view(-1).float()
    d = direction.view(-1).float()

    d_norm = d / (d.norm() + 1e-8)
    projection = (v @ d_norm) * d_norm
    residual = v - projection

    return residual.unsqueeze(0)


def compute_cosine(a, b):
    """Cosine similarity between two vectors."""
    return F.cosine_similarity(
        a.view(1, -1).float(), b.view(1, -1).float(),
    ).item()


# ========================== ANALYSIS ========================== #

def analyze_dimension_residual(dim_name, entity_mean_vec, entity_probe_weights,
                                exp2b_weights_by_type, skip_projection=False):
    """
    For one dimension:
        1. Load its concept vectors (mean-diff and probe weights)
        2. Project out generic entity vector (unless skip_projection=True)
        3. Compute alignment before and after projection
    """
    # Load this dimension's vectors
    concept_mean = load_concept_vector(dim_name)  # (n_layers, hidden_dim)
    concept_probes = load_concept_probe_weights(dim_name)

    results = {
        "dimension": dim_name,
        "layers": [],
        "raw": {},      # alignment before projection
        "residual": {}, # alignment after projection
        "entity_overlap": [],  # how much of concept is explained by entity
    }

    for probe_type in exp2b_weights_by_type:
        results["raw"][probe_type] = {"probe_to_2b": [], "mean_to_2b": []}
        results["residual"][probe_type] = {"probe_to_2b": [], "mean_to_2b": []}

    n_layers = concept_mean.shape[0]
    entity_mean = entity_mean_vec  # (n_layers, hidden_dim)

    for layer in range(n_layers):
        results["layers"].append(layer)

        # Concept mean-diff vector at this layer
        cm = concept_mean[layer].unsqueeze(0)  # (1, hidden_dim)
        em = entity_mean[layer].unsqueeze(0)    # (1, hidden_dim)

        # How much of concept is explained by entity? (cosine between them)
        overlap = compute_cosine(cm, em) if not skip_projection else None
        results["entity_overlap"].append(overlap)

        # Residual mean-diff: project out entity (or copy raw if skipping)
        if skip_projection:
            cm_residual = cm
        else:
            cm_residual = project_out(cm, em)

        # Concept probe weight at this layer (if exists)
        cp = concept_probes.get(layer)
        ep = entity_probe_weights.get(layer)

        if skip_projection:
            cp_residual = cp
        elif cp is not None and ep is not None:
            cp_residual = project_out(cp, ep)
        elif cp is not None:
            # No entity probe at this layer — project out entity mean instead
            cp_residual = project_out(cp, em)
        else:
            cp_residual = None

        # Compute alignment with each Exp 2b probe type
        for probe_type, e2b_weights in exp2b_weights_by_type.items():
            e2b_w = e2b_weights.get(layer)
            if e2b_w is None:
                results["raw"][probe_type]["probe_to_2b"].append(None)
                results["raw"][probe_type]["mean_to_2b"].append(None)
                results["residual"][probe_type]["probe_to_2b"].append(None)
                results["residual"][probe_type]["mean_to_2b"].append(None)
                continue

            # Raw alignment
            raw_mean = compute_cosine(cm, e2b_w)
            raw_probe = compute_cosine(cp, e2b_w) if cp is not None else None

            results["raw"][probe_type]["mean_to_2b"].append(raw_mean)
            results["raw"][probe_type]["probe_to_2b"].append(raw_probe)

            # Residual alignment
            res_mean = compute_cosine(cm_residual, e2b_w)
            res_probe = (
                compute_cosine(cp_residual, e2b_w)
                if cp_residual is not None else None
            )

            results["residual"][probe_type]["mean_to_2b"].append(res_mean)
            results["residual"][probe_type]["probe_to_2b"].append(res_probe)

    return results


def summarize_results(results):
    """Extract key numbers from one dimension's results."""
    summary = {"dimension": results["dimension"]}

    # Entity overlap: how much of concept vector is generic entity
    overlaps = results["entity_overlap"]
    valid_overlaps = [v for v in overlaps if v is not None and abs(v) > 0.001]
    if valid_overlaps:
        summary["entity_overlap_mean"] = float(np.mean(valid_overlaps))
        summary["entity_overlap_max"] = float(max(valid_overlaps, key=abs))
    else:
        summary["entity_overlap_mean"] = None
        summary["entity_overlap_max"] = None

    for probe_type in results["raw"]:
        for analysis_type in ["raw", "residual"]:
            for metric in ["probe_to_2b", "mean_to_2b"]:
                vals = results[analysis_type][probe_type][metric]
                valid = [v for v in vals if v is not None]
                prefix = f"{analysis_type}_{probe_type}_{metric}"
                if valid:
                    summary[f"{prefix}_mean"] = float(np.mean(valid))
                    summary[f"{prefix}_max_abs"] = float(max(valid, key=abs))
                else:
                    summary[f"{prefix}_mean"] = None
                    summary[f"{prefix}_max_abs"] = None

    return summary


# ========================== PLOTTING ========================== #

def plot_raw_vs_residual(all_results, output_dir):
    """
    Bar chart: raw vs residual max|cos| for each dimension,
    for both control and reading probes.
    """
    os.makedirs(output_dir, exist_ok=True)

    dims = [r["dimension"] for r in all_results]
    n_dims = len(dims)

    for probe_type in ["control_probe", "reading_probe"]:
        raw_vals = []
        res_vals = []

        for r in all_results:
            # Use probe_to_2b (trained probe alignment)
            raw_v = r["raw"].get(probe_type, {}).get("probe_to_2b", [])
            res_v = r["residual"].get(probe_type, {}).get("probe_to_2b", [])

            raw_valid = [v for v in raw_v if v is not None]
            res_valid = [v for v in res_v if v is not None]

            raw_vals.append(max(raw_valid, key=abs) if raw_valid else 0)
            res_vals.append(max(res_valid, key=abs) if res_valid else 0)

        x = np.arange(n_dims)
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(12, n_dims * 0.8), 6))
        bars1 = ax.bar(x - width/2, raw_vals, width, label="Raw", color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + width/2, res_vals, width, label="Residual (entity removed)", color="coral", alpha=0.8)

        ax.set_xlabel("Concept Dimension")
        ax.set_ylabel("Max |cosine similarity| with Exp 2b probe")
        ax.set_title(f"Raw vs Residual Alignment: {probe_type}")
        ax.set_xticks(x)
        ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylim(-1, 1)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"raw_vs_residual_{probe_type}.png"),
            dpi=200, bbox_inches="tight",
        )
        plt.close()

    # Entity overlap bar chart
    overlaps = []
    for r in all_results:
        valid = [v for v in r["entity_overlap"] if v is not None and abs(v) > 0.001]
        overlaps.append(np.mean(valid) if valid else 0)

    fig, ax = plt.subplots(figsize=(max(12, n_dims * 0.8), 5))
    ax.bar(range(n_dims), overlaps, color="mediumpurple", alpha=0.8)
    ax.set_xlabel("Concept Dimension")
    ax.set_ylabel("Mean cosine with generic entity vector")
    ax.set_title("Entity-Type Overlap per Dimension")
    ax.set_xticks(range(n_dims))
    ax.set_xticklabels(dims, rotation=45, ha="right", fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "entity_overlap_by_dimension.png"),
        dpi=200, bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plots to {output_dir}/")


def plot_layer_wise_residual(results, output_dir):
    """Per-dimension layer-wise plot: raw vs residual alignment."""
    dim_name = results["dimension"]
    dim_dir = os.path.join(output_dir, "per_dimension")
    os.makedirs(dim_dir, exist_ok=True)

    layers = results["layers"]

    for probe_type in results["raw"]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, metric, label in [
            (axes[0], "probe_to_2b", "Concept Probe"),
            (axes[1], "mean_to_2b", "Mean-Diff Vector"),
        ]:
            raw_v = results["raw"][probe_type][metric]
            res_v = results["residual"][probe_type][metric]

            raw_valid = [(l, v) for l, v in zip(layers, raw_v) if v is not None]
            res_valid = [(l, v) for l, v in zip(layers, res_v) if v is not None]

            if raw_valid:
                ax.plot(
                    [p[0] for p in raw_valid],
                    [p[1] for p in raw_valid],
                    'o-', label="Raw", color="steelblue", markersize=3,
                )
            if res_valid:
                ax.plot(
                    [p[0] for p in res_valid],
                    [p[1] for p in res_valid],
                    's--', label="Residual", color="coral", markersize=3,
                )

            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"{label} ↔ {probe_type}")
            ax.legend(fontsize=8)
            ax.set_ylim(-1, 1)

        plt.suptitle(f"{dim_name}: Raw vs Residual Alignment", y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(dim_dir, f"{dim_name}_{probe_type}_residual.png"),
            dpi=200, bbox_inches="tight",
        )
        plt.close()


# ========================== MAIN ========================== #

def parse_args():
    p = argparse.ArgumentParser(
        description="Exp 3: Residual alignment analysis (entity subtraction)."
    )
    p.add_argument(
        "--dim_ids", type=int, nargs="+", default=None,
        help="Dimensions to analyze (default: all 1-13).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.dim_ids:
        dim_ids = args.dim_ids
    else:
        # All dimensions including dim 0 (entity baseline analyzed as reference)
        dim_ids = sorted(DIMENSION_REGISTRY.keys())

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # === Load generic entity baseline (dim 0) ===
    if 0 not in DIMENSION_REGISTRY:
        print("[ERROR] Dimension 0 (entity baseline) not found in registry.")
        print("  Make sure a file like 0_baseline.py exists in concepts/")
        sys.exit(1)

    _, entity_dim_name = DIMENSION_REGISTRY[0]
    entity_vec_path = os.path.join(
        CONCEPT_ROOT, entity_dim_name, "concept_vector_per_layer.npz",
    )
    if not os.path.isfile(entity_vec_path):
        print(f"[ERROR] Generic entity baseline not found at {entity_vec_path}")
        print(f"  Run: python 1_elicit_concept_vectors.py --dim_id 0")
        sys.exit(1)

    entity_mean_vec = load_concept_vector(entity_dim_name)
    entity_probe_weights = load_concept_probe_weights(entity_dim_name)
    print(f"Loaded generic entity vector: {entity_mean_vec.shape}")
    print(f"Loaded {len(entity_probe_weights)} entity probe weights")

    # === Load Exp 2b probes ===
    exp2b_weights = {}
    for probe_type, probe_dir in EXP2B_PROBE_TYPES.items():
        exp2b_weights[probe_type] = load_exp2b_weights(probe_dir)
        print(f"Loaded {len(exp2b_weights[probe_type])} Exp 2b {probe_type} weights")

    # === Analyze each dimension ===
    all_results = []
    all_summaries = []

    for dim_id in dim_ids:
        if dim_id not in DIMENSION_REGISTRY:
            print(f"[WARN] Unknown dim_id={dim_id}, skipping.")
            continue

        _, dim_name = DIMENSION_REGISTRY[dim_id]
        is_entity_baseline = (dim_id == 0)
        print(f"\n{'='*60}")
        print(f"Dimension {dim_id}: {dim_name}"
              f"{' (ENTITY BASELINE — no self-subtraction)' if is_entity_baseline else ''}")
        print(f"{'='*60}")

        try:
            results = analyze_dimension_residual(
                dim_name, entity_mean_vec, entity_probe_weights,
                exp2b_weights,
                skip_projection=is_entity_baseline,
            )
            all_results.append(results)

            summary = summarize_results(results)
            all_summaries.append(summary)

            # Save per-dimension JSON
            per_dim_dir = os.path.join(OUTPUT_ROOT, "per_dimension")
            os.makedirs(per_dim_dir, exist_ok=True)
            json_path = os.path.join(per_dim_dir, f"{dim_name}_residual.json")
            # Convert for JSON serialization
            json_safe = json.loads(json.dumps(results, default=str))
            with open(json_path, "w") as f:
                json.dump(json_safe, f, indent=2)

            # Per-dimension plot
            plot_layer_wise_residual(results, OUTPUT_ROOT)

            # Print summary
            eo_val = summary.get('entity_overlap_mean')
            eo_str = f"{eo_val:.4f}" if eo_val is not None else "N/A (baseline)"
            print(f"\n  Entity overlap (mean cos): {eo_str}")
            for pt in ["control_probe", "reading_probe"]:
                raw_max = summary.get(f"raw_{pt}_probe_to_2b_max_abs", "N/A")
                res_max = summary.get(f"residual_{pt}_probe_to_2b_max_abs", "N/A")
                raw_str = f"{raw_max:.4f}" if isinstance(raw_max, float) else raw_max
                res_str = f"{res_max:.4f}" if isinstance(res_max, float) else res_max
                print(f"  {pt}: raw max|cos|={raw_str}, "
                      f"residual max|cos|={res_str}")

        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

    # === Summary table ===
    if all_summaries:
        import pandas as pd
        summary_df = pd.DataFrame(all_summaries)
        summary_path = os.path.join(OUTPUT_ROOT, "residual_alignment_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[SAVED] {summary_path}")

        # Print compact table
        print(f"\n{'='*100}")
        print(f"RESIDUAL ALIGNMENT SUMMARY")
        print(f"{'='*100}")
        print(f"{'Dimension':<28} {'Entity':<10} "
              f"{'Ctrl Raw':<10} {'Ctrl Res':<10} "
              f"{'Read Raw':<10} {'Read Res':<10}")
        print("-" * 78)

        for s in all_summaries:
            eo = s.get("entity_overlap_mean")
            eo_str = f"{eo:<10.4f}" if eo is not None else "N/A       "
            cr = s.get("raw_control_probe_probe_to_2b_max_abs", 0) or 0
            cres = s.get("residual_control_probe_probe_to_2b_max_abs", 0) or 0
            rr = s.get("raw_reading_probe_probe_to_2b_max_abs", 0) or 0
            rres = s.get("residual_reading_probe_probe_to_2b_max_abs", 0) or 0
            print(f"{s['dimension']:<28} {eo_str} "
                  f"{cr:<10.4f} {cres:<10.4f} "
                  f"{rr:<10.4f} {rres:<10.4f}")

        print("-" * 78)

    # === Cross-dimension plots ===
    if all_results:
        plot_raw_vs_residual(all_results, OUTPUT_ROOT)

    print(f"\n✅ Residual alignment analysis complete.")
    print(f"   Results: {OUTPUT_ROOT}/")


if __name__ == "__main__":
    main()