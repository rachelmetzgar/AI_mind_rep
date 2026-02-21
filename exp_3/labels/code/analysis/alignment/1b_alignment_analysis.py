#!/usr/bin/env python3
"""
Experiment 3, Phase 1b: Concept-Probe Alignment Analysis

Computes alignment between concept vectors (from Phase 1a) and
conversational probes (reading and control, from Experiment 2).

Three analyses, organized under two output folders:

    contrasts/raw/       Raw cosine alignment between human-AI contrast vectors
                         and probe weight vectors. No entity subtraction.
                         Uses data from: data/concept_activations/contrasts/

    contrasts/residual/  Same as raw, but with the entity baseline (dim 0)
                         direction projected out of each concept vector first.
                         Uses data from: data/concept_activations/contrasts/

    standalone/          Cosine alignment between standalone mean activation
                         vectors and probe weight vectors. No entity framing.
                         Uses data from: data/concept_activations/standalone/

For each analysis, outputs per dimension:
    - Per-layer cosine similarity (and R²) with reading and control probes
    - Mean-across-layers summary statistic
    - Bootstrap confidence intervals (1000 iterations)
    - Split-half paired bootstrap for pairwise dimension comparisons

Also outputs cross-dimension summary tables and figures.

Usage:
    python 1b_alignment_analysis.py --analysis raw
    python 1b_alignment_analysis.py --analysis residual
    python 1b_alignment_analysis.py --analysis standalone
    python 1b_alignment_analysis.py --analysis all

SLURM:
    sbatch slurm/alignment_analysis.sh all

Env: llama2_env
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.probes import LinearProbeClassification
from config import config


# ========================== CONFIG ========================== #

CONTRAST_ACT_DIR = str(config.PATHS.concept_activations_contrasts)
STANDALONE_ACT_DIR = str(config.PATHS.concept_activations_standalone)
OUTPUT_ROOT = str(config.RESULTS.alignment)

# Output subdirectories — raw and residual live under contrasts/
CONTRASTS_OUTPUT_DIR = str(config.RESULTS.alignment_contrasts)
RAW_OUTPUT_DIR = str(config.RESULTS.alignment_contrasts_raw)
RESIDUAL_OUTPUT_DIR = str(config.RESULTS.alignment_contrasts_residual)
STANDALONE_OUTPUT_DIR = str(config.RESULTS.alignment_standalone)

# Probe directories (from Experiment 2)
READING_PROBE_DIR = str(config.PATHS.exp2_reading_probe)
CONTROL_PROBE_DIR = str(config.PATHS.exp2_control_probe)

INPUT_DIM = config.INPUT_DIM
DEVICE = "cpu"  # Alignment is just vector math, no GPU needed

# Entity baseline dimension ID (for residual analysis)
ENTITY_BASELINE_DIM = "0_baseline"

# Bootstrap config
N_BOOTSTRAP = config.ANALYSIS.n_bootstrap
N_SPLIT_HALF = config.ANALYSIS.n_split_half
SEED = config.ANALYSIS.seed


# ========================== PROBE LOADING ========================== #

def load_probe_weights(probe_dir, input_dim=INPUT_DIM):
    """
    Load probe weight vectors from saved checkpoints.

    Returns dict: {layer_idx: weight_vector} where weight_vector is shape (input_dim,).
    For logistic probes (sigmoid output), the weight is from the linear layer.
    """
    weights = {}
    for fname in sorted(os.listdir(probe_dir)):
        if not fname.endswith(".pth") or fname.endswith("_final.pth"):
            continue
        if "layer_" not in fname:
            continue

        layer_str = fname.split("_layer_")[-1].split(".pth")[0]
        try:
            layer_idx = int(layer_str)
        except ValueError:
            continue

        probe = LinearProbeClassification(
            device=DEVICE, probe_class=1, input_dim=input_dim, logistic=True
        )
        state = torch.load(os.path.join(probe_dir, fname), map_location=DEVICE)
        probe.load_state_dict(state)
        probe.eval()

        # Extract weight vector: proj[0] is the Linear layer
        w = probe.proj[0].weight.detach().squeeze()  # (input_dim,)
        weights[layer_idx] = w.numpy()

    print(f"Loaded {len(weights)} probe weight vectors from {probe_dir}")
    return weights


# ========================== VECTOR LOADING ========================== #

def discover_dimensions(act_dir):
    """Discover available dimension folders in an activation directory."""
    dims = {}
    if not os.path.isdir(act_dir):
        return dims
    for name in sorted(os.listdir(act_dir)):
        full = os.path.join(act_dir, name)
        if not os.path.isdir(full):
            continue
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        dims[dim_id] = name
    return dims


def load_contrast_vector(dim_name, act_dir=CONTRAST_ACT_DIR):
    """Load the concept direction (human - AI) per layer for a contrast dimension."""
    path = os.path.join(act_dir, dim_name, "concept_vector_per_layer.npz")
    data = np.load(path)
    return data["concept_direction"]  # shape: (n_layers, hidden_dim)


def load_contrast_activations(dim_name, act_dir=CONTRAST_ACT_DIR):
    """Load raw activations and labels for a contrast dimension."""
    path = os.path.join(act_dir, dim_name, "concept_activations.npz")
    data = np.load(path)
    return data["activations"], data["labels"]  # (n_prompts, n_layers, hidden), (n_prompts,)


def load_standalone_mean(dim_name, act_dir=STANDALONE_ACT_DIR):
    """Load the mean concept vector per layer for a standalone dimension."""
    path = os.path.join(act_dir, dim_name, "mean_vectors_per_layer.npz")
    data = np.load(path)
    return data["mean_concept"]  # shape: (n_layers, hidden_dim)


def load_standalone_activations(dim_name, act_dir=STANDALONE_ACT_DIR):
    """Load raw activations for a standalone dimension."""
    path = os.path.join(act_dir, dim_name, "concept_activations.npz")
    data = np.load(path)
    return data["activations"]  # (n_prompts, n_layers, hidden)


# ========================== ALIGNMENT COMPUTATION ========================== #

def cosine_sim(v1, v2):
    """Cosine similarity between two 1D numpy vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def project_out(v, direction):
    """Project direction out of v. Returns the residual component."""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    projection = np.dot(v, d_norm) * d_norm
    return v - projection


def compute_alignment_per_layer(concept_vectors, probe_weights):
    """
    Compute cosine similarity between concept vector and probe weight
    at each layer where both are available.

    Args:
        concept_vectors: (n_layers, hidden_dim) numpy array
        probe_weights: dict {layer_idx: (hidden_dim,) numpy array}

    Returns:
        dict {layer_idx: {"cosine": float, "r_squared": float}}
    """
    results = {}
    for layer_idx, w in probe_weights.items():
        if layer_idx >= concept_vectors.shape[0]:
            continue
        cv = concept_vectors[layer_idx]
        cos = cosine_sim(cv, w)
        results[layer_idx] = {
            "cosine": cos,
            "r_squared": cos ** 2,
        }
    return results


def mean_across_layers(alignment_dict):
    """Compute mean R² and mean |cosine| across all layers."""
    if not alignment_dict:
        return {"mean_r_squared": 0.0, "mean_abs_cosine": 0.0, "n_layers": 0}
    r2_vals = [v["r_squared"] for v in alignment_dict.values()]
    cos_vals = [abs(v["cosine"]) for v in alignment_dict.values()]
    return {
        "mean_r_squared": float(np.mean(r2_vals)),
        "mean_abs_cosine": float(np.mean(cos_vals)),
        "n_layers": len(r2_vals),
    }


# ========================== BOOTSTRAP ========================== #

def bootstrap_contrast_alignment(activations, labels, probe_weights,
                                 n_bootstrap=N_BOOTSTRAP,
                                 entity_baseline_acts=None,
                                 entity_baseline_labels=None,
                                 residual_mode=False):
    """
    Bootstrap resampling for contrast concept vectors.

    For each iteration:
        1. Resample human and AI prompt indices (with replacement)
        2. Recompute mean-difference vector
        3. (If residual) Also resample entity baseline and project out
        4. Compute alignment with probes
        5. Record mean-across-layers R²

    Returns: array of shape (n_bootstrap,) with mean R² per iteration.
    """
    rng = np.random.RandomState(SEED)
    human_mask = labels == 1
    ai_mask = labels == 0
    human_acts = activations[human_mask]  # (n_human, n_layers, hidden)
    ai_acts = activations[ai_mask]
    n_human = human_acts.shape[0]
    n_ai = ai_acts.shape[0]

    if residual_mode:
        assert entity_baseline_acts is not None
        bl_human = entity_baseline_acts[entity_baseline_labels == 1]
        bl_ai = entity_baseline_acts[entity_baseline_labels == 0]
        n_bl_h = bl_human.shape[0]
        n_bl_a = bl_ai.shape[0]

    boot_r2 = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        # Resample
        idx_h = rng.choice(n_human, size=n_human, replace=True)
        idx_a = rng.choice(n_ai, size=n_ai, replace=True)
        cv = human_acts[idx_h].mean(axis=0) - ai_acts[idx_a].mean(axis=0)

        if residual_mode:
            idx_bl_h = rng.choice(n_bl_h, size=n_bl_h, replace=True)
            idx_bl_a = rng.choice(n_bl_a, size=n_bl_a, replace=True)
            bl_dir = bl_human[idx_bl_h].mean(axis=0) - bl_ai[idx_bl_a].mean(axis=0)
            # Project out entity baseline at each layer
            for layer in range(cv.shape[0]):
                cv[layer] = project_out(cv[layer], bl_dir[layer])

        # Alignment
        r2_per_layer = []
        for layer_idx, w in probe_weights.items():
            if layer_idx >= cv.shape[0]:
                continue
            cos = cosine_sim(cv[layer_idx], w)
            r2_per_layer.append(cos ** 2)

        boot_r2[b] = np.mean(r2_per_layer) if r2_per_layer else 0.0

    return boot_r2


def bootstrap_standalone_alignment(activations, probe_weights,
                                   n_bootstrap=N_BOOTSTRAP):
    """
    Bootstrap resampling for standalone mean activation vectors.

    For each iteration:
        1. Resample prompt indices (with replacement)
        2. Recompute mean activation vector
        3. Compute alignment with probes
        4. Record mean-across-layers R²

    Returns: array of shape (n_bootstrap,) with mean R² per iteration.
    """
    rng = np.random.RandomState(SEED)
    n_prompts = activations.shape[0]

    boot_r2 = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.choice(n_prompts, size=n_prompts, replace=True)
        mean_vec = activations[idx].mean(axis=0)

        r2_per_layer = []
        for layer_idx, w in probe_weights.items():
            if layer_idx >= mean_vec.shape[0]:
                continue
            cos = cosine_sim(mean_vec[layer_idx], w)
            r2_per_layer.append(cos ** 2)

        boot_r2[b] = np.mean(r2_per_layer) if r2_per_layer else 0.0

    return boot_r2


# ========================== ANALYSIS RUNNERS ========================== #

def run_raw_alignment(reading_weights, control_weights, dim_filter=None):
    """Analysis A: Raw alignment of contrast vectors with probes."""
    print("\n" + "=" * 60)
    print("ANALYSIS: raw alignment (contrast vectors, no subtraction)")
    print("=" * 60)

    out_dir = RAW_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    dims = discover_dimensions(CONTRAST_ACT_DIR)
    if dim_filter:
        dims = {k: v for k, v in dims.items() if k in dim_filter}

    # Load existing summary to merge with new results
    summary_path = os.path.join(out_dir, "summary.json")
    if dim_filter and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  Loaded existing summary ({len(summary)} dims), will merge new results")
    else:
        summary = {}

    for dim_id in sorted(dims.keys()):
        dim_name = dims[dim_id]
        print(f"\n--- Dim {dim_id}: {dim_name} ---")

        cv = load_contrast_vector(dim_name)
        acts, labels = load_contrast_activations(dim_name)

        # Per-layer alignment
        reading_align = compute_alignment_per_layer(cv, reading_weights)
        control_align = compute_alignment_per_layer(cv, control_weights)

        reading_summary = mean_across_layers(reading_align)
        control_summary = mean_across_layers(control_align)

        print(f"  Reading: mean R² = {reading_summary['mean_r_squared']:.4f}")
        print(f"  Control: mean R² = {control_summary['mean_r_squared']:.4f}")

        # Bootstrap
        boot_reading = bootstrap_contrast_alignment(acts, labels, reading_weights)
        boot_control = bootstrap_contrast_alignment(acts, labels, control_weights)

        # Save per-dimension
        dim_out = os.path.join(out_dir, dim_name)
        os.makedirs(dim_out, exist_ok=True)

        np.savez_compressed(
            os.path.join(dim_out, "alignment.npz"),
            reading_per_layer=json.dumps(reading_align),
            control_per_layer=json.dumps(control_align),
            boot_reading_r2=boot_reading,
            boot_control_r2=boot_control,
        )

        summary[dim_name] = {
            "dim_id": dim_id,
            "reading_mean_r2": reading_summary["mean_r_squared"],
            "control_mean_r2": control_summary["mean_r_squared"],
            "reading_boot_ci95": [
                float(np.percentile(boot_reading, 2.5)),
                float(np.percentile(boot_reading, 97.5)),
            ],
            "control_boot_ci95": [
                float(np.percentile(boot_control, 2.5)),
                float(np.percentile(boot_control, 97.5)),
            ],
        }

    # Save summary
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary_table("RAW ALIGNMENT", summary)
    return summary


def run_residual_alignment(reading_weights, control_weights, dim_filter=None):
    """Analysis B: Residual alignment after projecting out entity baseline."""
    print("\n" + "=" * 60)
    print("ANALYSIS: residual alignment (entity baseline projected out)")
    print("=" * 60)

    out_dir = RESIDUAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    dims = discover_dimensions(CONTRAST_ACT_DIR)

    # Load entity baseline (always needed, even if filtered)
    if ENTITY_BASELINE_DIM not in [dims[k] for k in dims]:
        # Try to find it
        baseline_found = False
        for did, dname in dims.items():
            if "baseline" in dname or did == 0:
                entity_baseline_name = dname
                baseline_found = True
                break
        if not baseline_found:
            raise FileNotFoundError(
                f"Entity baseline dimension not found in {CONTRAST_ACT_DIR}"
            )
    else:
        entity_baseline_name = ENTITY_BASELINE_DIM

    baseline_cv = load_contrast_vector(entity_baseline_name)
    baseline_acts, baseline_labels = load_contrast_activations(entity_baseline_name)
    print(f"Loaded entity baseline: {entity_baseline_name}")

    # Filter after baseline is loaded
    if dim_filter:
        dims = {k: v for k, v in dims.items() if k in dim_filter}

    # Load existing summary to merge
    summary_path = os.path.join(out_dir, "summary.json")
    if dim_filter and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  Loaded existing summary ({len(summary)} dims), will merge new results")
    else:
        summary = {}

    for dim_id in sorted(dims.keys()):
        dim_name = dims[dim_id]

        # Skip shapes and baseline from main results (still compute for completeness)
        print(f"\n--- Dim {dim_id}: {dim_name} ---")

        cv = load_contrast_vector(dim_name)

        # Compute entity overlap before subtraction
        entity_overlaps = {}
        for layer in range(cv.shape[0]):
            entity_overlaps[layer] = cosine_sim(cv[layer], baseline_cv[layer])

        mean_entity_overlap = float(np.mean([abs(v) for v in entity_overlaps.values()]))
        print(f"  Entity overlap (mean |cos|): {mean_entity_overlap:.4f}")

        # Project out entity baseline at each layer
        residual_cv = np.zeros_like(cv)
        for layer in range(cv.shape[0]):
            residual_cv[layer] = project_out(cv[layer], baseline_cv[layer])

        # Per-layer alignment of residual
        reading_align = compute_alignment_per_layer(residual_cv, reading_weights)
        control_align = compute_alignment_per_layer(residual_cv, control_weights)

        reading_summary = mean_across_layers(reading_align)
        control_summary = mean_across_layers(control_align)

        print(f"  Reading (residual): mean R² = {reading_summary['mean_r_squared']:.4f}")
        print(f"  Control (residual): mean R² = {control_summary['mean_r_squared']:.4f}")

        # Bootstrap with joint entity baseline resampling
        acts, labels = load_contrast_activations(dim_name)
        boot_reading = bootstrap_contrast_alignment(
            acts, labels, reading_weights,
            entity_baseline_acts=baseline_acts,
            entity_baseline_labels=baseline_labels,
            residual_mode=True,
        )
        boot_control = bootstrap_contrast_alignment(
            acts, labels, control_weights,
            entity_baseline_acts=baseline_acts,
            entity_baseline_labels=baseline_labels,
            residual_mode=True,
        )

        # Save per-dimension
        dim_out = os.path.join(out_dir, dim_name)
        os.makedirs(dim_out, exist_ok=True)

        np.savez_compressed(
            os.path.join(dim_out, "alignment.npz"),
            reading_per_layer=json.dumps(reading_align),
            control_per_layer=json.dumps(control_align),
            entity_overlap_per_layer=json.dumps(
                {str(k): v for k, v in entity_overlaps.items()}
            ),
            boot_reading_r2=boot_reading,
            boot_control_r2=boot_control,
        )

        summary[dim_name] = {
            "dim_id": dim_id,
            "entity_overlap": mean_entity_overlap,
            "reading_mean_r2": reading_summary["mean_r_squared"],
            "control_mean_r2": control_summary["mean_r_squared"],
            "reading_boot_ci95": [
                float(np.percentile(boot_reading, 2.5)),
                float(np.percentile(boot_reading, 97.5)),
            ],
            "control_boot_ci95": [
                float(np.percentile(boot_control, 2.5)),
                float(np.percentile(boot_control, 97.5)),
            ],
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary_table("RESIDUAL ALIGNMENT", summary)
    return summary


def run_standalone_alignment(reading_weights, control_weights, dim_filter=None):
    """Analysis C: Standalone concept alignment (mean activation vectors)."""
    print("\n" + "=" * 60)
    print("ANALYSIS: standalone alignment (concept-only, no entity framing)")
    print("=" * 60)

    out_dir = STANDALONE_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    dims = discover_dimensions(STANDALONE_ACT_DIR)
    if dim_filter:
        dims = {k: v for k, v in dims.items() if k in dim_filter}

    # Load existing summary to merge
    summary_path = os.path.join(out_dir, "summary.json")
    if dim_filter and os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"  Loaded existing summary ({len(summary)} dims), will merge new results")
    else:
        summary = {}

    for dim_id in sorted(dims.keys()):
        dim_name = dims[dim_id]
        print(f"\n--- Dim {dim_id}: {dim_name} ---")

        mean_vec = load_standalone_mean(dim_name)
        acts = load_standalone_activations(dim_name)

        # Per-layer alignment
        reading_align = compute_alignment_per_layer(mean_vec, reading_weights)
        control_align = compute_alignment_per_layer(mean_vec, control_weights)

        reading_summary = mean_across_layers(reading_align)
        control_summary = mean_across_layers(control_align)

        print(f"  Reading: mean R² = {reading_summary['mean_r_squared']:.4f}")
        print(f"  Control: mean R² = {control_summary['mean_r_squared']:.4f}")

        # Bootstrap
        boot_reading = bootstrap_standalone_alignment(acts, reading_weights)
        boot_control = bootstrap_standalone_alignment(acts, control_weights)

        # Save per-dimension
        dim_out = os.path.join(out_dir, dim_name)
        os.makedirs(dim_out, exist_ok=True)

        np.savez_compressed(
            os.path.join(dim_out, "alignment.npz"),
            reading_per_layer=json.dumps(reading_align),
            control_per_layer=json.dumps(control_align),
            boot_reading_r2=boot_reading,
            boot_control_r2=boot_control,
        )

        summary[dim_name] = {
            "dim_id": dim_id,
            "reading_mean_r2": reading_summary["mean_r_squared"],
            "control_mean_r2": control_summary["mean_r_squared"],
            "reading_boot_ci95": [
                float(np.percentile(boot_reading, 2.5)),
                float(np.percentile(boot_reading, 97.5)),
            ],
            "control_boot_ci95": [
                float(np.percentile(boot_control, 2.5)),
                float(np.percentile(boot_control, 97.5)),
            ],
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_summary_table("STANDALONE ALIGNMENT", summary)
    return summary


# ========================== DISPLAY ========================== #

def print_summary_table(title, summary):
    """Print a formatted summary table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Dimension':<30} {'Read R²':>10} {'Ctrl R²':>10} "
          f"{'Read 95% CI':>20}")
    print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 20}")

    for dim_name in sorted(summary.keys(),
                           key=lambda x: summary[x].get("dim_id", 99)):
        s = summary[dim_name]
        ci = s.get("reading_boot_ci95", [0, 0])
        print(f"  {dim_name:<30} {s['reading_mean_r2']:>10.4f} "
              f"{s['control_mean_r2']:>10.4f} "
              f"[{ci[0]:.4f}, {ci[1]:.4f}]")

    print(f"{'=' * 70}\n")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 Phase 1b: Concept-probe alignment analysis"
    )
    parser.add_argument("--analysis", type=str, required=True,
                        choices=["raw", "residual", "standalone", "all"],
                        help="Which analysis to run")
    parser.add_argument("--dims", type=str, default=None,
                        help="Comma-separated dim IDs to process (default: all). "
                             "E.g. --dims 18,20,21,22,23")
    args = parser.parse_args()

    # Parse optional dimension filter
    dim_filter = None
    if args.dims:
        dim_filter = set(int(d.strip()) for d in args.dims.split(","))
        print(f"Filtering to dimensions: {sorted(dim_filter)}")

    # Load probe weights (shared across all analyses)
    print("Loading probe weight vectors...")
    reading_weights = load_probe_weights(READING_PROBE_DIR)
    control_weights = load_probe_weights(CONTROL_PROBE_DIR)

    if args.analysis in ("raw", "all"):
        run_raw_alignment(reading_weights, control_weights, dim_filter=dim_filter)

    if args.analysis in ("residual", "all"):
        run_residual_alignment(reading_weights, control_weights, dim_filter=dim_filter)

    if args.analysis in ("standalone", "all"):
        run_standalone_alignment(reading_weights, control_weights, dim_filter=dim_filter)

    print("✅ Phase 2 alignment analysis complete.")


if __name__ == "__main__":
    main()