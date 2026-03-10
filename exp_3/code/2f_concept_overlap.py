#!/usr/bin/env python3
"""
Experiment 3, Phase 2f: Concept-Concept Overlap Analysis

Computes pairwise cosine similarity between all contrast dimension vectors
to understand how much the human-vs-AI contrast for one mental dimension
overlaps with the contrast for another (especially the entity baseline).

This is version-independent — it operates on the concept vectors themselves
(from Phase 1a), not on probe alignment.

Key question: When we measure "human phenomenology vs AI phenomenology,"
how much of that direction is just the general "human vs AI" direction?

Outputs:
    concept_overlap/
        overlap_matrix.npz       Full pairwise overlap + bootstrap distributions
        overlap_matrix.csv       Readable pairwise table
        baseline_overlap.csv     Per-dim overlap with dim 0 (entity baseline)
        sysprompt_baseline_overlap.csv  Same using dim 18 (sysprompt) as baseline
        layer_profiles.npz       Per-layer cosine for each pair

Usage:
    python 2f_concept_overlap.py
    python 2f_concept_overlap.py --baseline 0
    python 2f_concept_overlap.py --baseline 18

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import argparse
import numpy as np
import csv

from config import config, add_variant_argument, set_variant, variant_filename, data_subdir

# ========================== CONFIG ========================== #

CONTRAST_ACT_DIR = str(config.RESULTS.concept_activations_contrasts)
N_LAYERS = config.N_LAYERS
N_BOOTSTRAP = config.ANALYSIS.n_bootstrap
SEED = config.ANALYSIS.seed
MIN_LAYER = config.ANALYSIS.restricted_layer_start  # 6

OUTPUT_DIR = os.path.join(str(config.RESULTS.concept_overlap), "contrasts")

# Dimension display names
DIM_NAMES = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind (holistic)", 17: "Attention", 18: "SysPrompt (labeled)",
    25: "Beliefs", 26: "Desires", 27: "Goals",
    30: "Granite/Sandstone", 31: "Squares/Triangles",
    32: "Horizontal/Vertical",
}

DIM_CATEGORIES = {
    0: "Baseline", 1: "Mental", 2: "Mental", 3: "Mental",
    4: "Mental", 5: "Mental", 6: "Mental", 7: "Mental",
    8: "Mental", 9: "Mental", 10: "Physical", 11: "Pragmatic",
    12: "Pragmatic", 13: "Pragmatic", 14: "Bio Ctrl", 15: "Shapes",
    16: "Mental", 17: "Mental", 18: "SysPrompt",
}


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


def load_contrast_vector(dim_name, act_dir=None):
    """Load the concept direction (human - AI) per layer for a contrast dimension."""
    if act_dir is None:
        act_dir = CONTRAST_ACT_DIR
    path = os.path.join(act_dir, dim_name, "concept_vector_per_layer.npz")
    data = np.load(path)
    return data["concept_direction"]  # shape: (n_layers, hidden_dim)


def load_contrast_activations(dim_name, act_dir=None):
    """Load raw activations and labels for a contrast dimension."""
    if act_dir is None:
        act_dir = CONTRAST_ACT_DIR
    path = os.path.join(act_dir, dim_name, "concept_activations.npz")
    data = np.load(path)
    return data["activations"], data["labels"]  # (n_prompts, n_layers, hidden), (n_prompts,)


def cosine_sim(v1, v2):
    """Cosine similarity between two 1D numpy vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ========================== OVERLAP COMPUTATION ========================== #

def compute_overlap_matrix(dim_vectors, dim_ids, layer_range):
    """
    Compute pairwise mean |cosine| across layers for all dimension pairs.

    Args:
        dim_vectors: dict {dim_id: (n_layers, hidden_dim) array}
        dim_ids: sorted list of dim IDs
        layer_range: range of layers to average over

    Returns:
        overlap: (n_dims, n_dims) matrix of mean |cosine| across layers
        layer_profiles: (n_dims, n_dims, n_layers) per-layer cosine values
    """
    n = len(dim_ids)
    overlap = np.zeros((n, n))
    layer_profiles = np.zeros((n, n, N_LAYERS))

    for i in range(n):
        vi = dim_vectors[dim_ids[i]]
        for j in range(i, n):
            vj = dim_vectors[dim_ids[j]]
            # Per-layer cosine
            for layer in range(N_LAYERS):
                layer_profiles[i, j, layer] = cosine_sim(vi[layer], vj[layer])
                layer_profiles[j, i, layer] = layer_profiles[i, j, layer]

            # Mean |cosine| over restricted layer range
            cos_vals = [abs(layer_profiles[i, j, l]) for l in layer_range]
            mean_abs_cos = float(np.mean(cos_vals))
            overlap[i, j] = mean_abs_cos
            overlap[j, i] = mean_abs_cos

    return overlap, layer_profiles


def _vectorized_pairwise_abs_cosine(vectors, layer_indices):
    """
    Compute pairwise mean |cosine| across layers for a stack of contrast vectors.

    Args:
        vectors: (n_dims, n_layers, hidden_dim) array
        layer_indices: array of layer indices to average over

    Returns:
        overlap: (n_dims, n_dims) matrix of mean |cosine|
    """
    # Select layers: (n_dims, n_selected_layers, hidden_dim)
    V = vectors[:, layer_indices, :]
    # Norms: (n_dims, n_selected_layers)
    norms = np.linalg.norm(V, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    V_normed = V / norms  # (n_dims, n_selected_layers, hidden_dim)

    n = V.shape[0]
    n_layers = len(layer_indices)
    overlap = np.zeros((n, n))

    for l_idx in range(n_layers):
        # (n_dims, hidden_dim) for this layer
        vl = V_normed[:, l_idx, :]
        # Pairwise cosine: (n_dims, n_dims)
        cos_matrix = vl @ vl.T
        overlap += np.abs(cos_matrix)

    overlap /= n_layers
    return overlap


def bootstrap_overlap(dim_activations, dim_labels, dim_ids, layer_range,
                      n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap pairwise overlap by resampling prompts and recomputing contrast vectors.

    Uses vectorized numpy operations for efficiency.

    Args:
        dim_activations: dict {dim_id: (n_prompts, n_layers, hidden)}
        dim_labels: dict {dim_id: (n_prompts,)}
        dim_ids: sorted list of dim IDs
        layer_range: range of layers to average over
        n_bootstrap: number of iterations
        seed: random seed

    Returns:
        boot_overlap: (n_dims, n_dims, n_bootstrap) array
    """
    rng = np.random.RandomState(seed)
    n = len(dim_ids)
    layer_indices = np.array(list(layer_range))
    boot_overlap = np.zeros((n, n, n_bootstrap))

    # Pre-split activations by label for each dimension
    human_acts = {}
    ai_acts = {}
    for dim_id in dim_ids:
        acts = dim_activations[dim_id]
        labels = dim_labels[dim_id]
        human_acts[dim_id] = acts[labels == 1]
        ai_acts[dim_id] = acts[labels == 0]

    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstrap}", flush=True)

        # Resample and compute contrast vector for each dimension
        boot_vectors = np.zeros((n, dim_activations[dim_ids[0]].shape[1],
                                 dim_activations[dim_ids[0]].shape[2]))
        for idx, dim_id in enumerate(dim_ids):
            h = human_acts[dim_id]
            a = ai_acts[dim_id]
            idx_h = rng.choice(h.shape[0], size=h.shape[0], replace=True)
            idx_a = rng.choice(a.shape[0], size=a.shape[0], replace=True)
            boot_vectors[idx] = h[idx_h].mean(axis=0) - a[idx_a].mean(axis=0)

        # Vectorized pairwise overlap
        boot_overlap[:, :, b] = _vectorized_pairwise_abs_cosine(
            boot_vectors, layer_indices
        )

    return boot_overlap


def compute_baseline_overlap(dim_vectors, dim_ids, baseline_id, layer_range):
    """
    Compute per-dimension overlap with a specific baseline dimension.

    Returns:
        overlaps: list of dicts with dim_id, name, category, mean_abs_cosine, per_layer
    """
    if baseline_id not in dim_vectors:
        raise ValueError(f"Baseline dim {baseline_id} not found in loaded vectors")

    baseline_vec = dim_vectors[baseline_id]
    results = []

    for dim_id in dim_ids:
        if dim_id == baseline_id:
            continue
        cv = dim_vectors[dim_id]
        per_layer = [cosine_sim(cv[l], baseline_vec[l]) for l in range(N_LAYERS)]
        restricted = [abs(per_layer[l]) for l in layer_range]
        results.append({
            "dim_id": dim_id,
            "name": DIM_NAMES.get(dim_id, f"dim_{dim_id}"),
            "category": DIM_CATEGORIES.get(dim_id, "Other"),
            "mean_abs_cosine": float(np.mean(restricted)),
            "per_layer_cosine": per_layer,
        })

    return results


# ========================== OUTPUT ========================== #

def save_overlap_matrix_csv(overlap, dim_ids, filepath):
    """Save overlap matrix as CSV with dimension names as headers."""
    names = [DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for i, dim_id in enumerate(dim_ids):
            row = [names[i]] + [f"{overlap[i, j]:.4f}" for j in range(len(dim_ids))]
            writer.writerow(row)


def save_baseline_overlap_csv(baseline_results, boot_overlap, dim_ids,
                              baseline_id, filepath):
    """Save per-dimension baseline overlap with bootstrap CIs."""
    # Find baseline index in dim_ids
    bl_idx = dim_ids.index(baseline_id)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim_id", "name", "category", "mean_abs_cosine",
                         "ci_lower", "ci_upper"])
        for res in baseline_results:
            dim_id = res["dim_id"]
            dim_idx = dim_ids.index(dim_id)
            boot_vals = boot_overlap[dim_idx, bl_idx, :]
            ci_lo = float(np.percentile(boot_vals, 2.5))
            ci_hi = float(np.percentile(boot_vals, 97.5))
            writer.writerow([
                dim_id, res["name"], res["category"],
                f"{res['mean_abs_cosine']:.4f}",
                f"{ci_lo:.4f}", f"{ci_hi:.4f}",
            ])


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 Phase 2f: Concept-concept overlap analysis"
    )
    parser.add_argument("--baseline", type=int, default=None,
                        help="Baseline dim ID for focused comparison (default: compute all)")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help=f"Bootstrap iterations (default: {N_BOOTSTRAP})")
    parser.add_argument("--exclude-dims", type=int, nargs="+", default=[10, 16],
                        help="Dimension IDs to exclude (default: 10 16)")
    add_variant_argument(parser)
    args = parser.parse_args()

    # Apply variant before reading paths
    if args.variant:
        set_variant(args.variant)
        # Re-read paths after variant is set
        global CONTRAST_ACT_DIR, OUTPUT_DIR
        CONTRAST_ACT_DIR = str(config.RESULTS.concept_activations_contrasts)
        OUTPUT_DIR = os.path.join(str(config.RESULTS.concept_overlap), "contrasts")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    layer_range = range(MIN_LAYER, N_LAYERS)

    # Discover and load all contrast dimensions
    dims = discover_dimensions(CONTRAST_ACT_DIR)
    dim_ids = sorted(dims.keys())
    print(f"Found {len(dim_ids)} contrast dimensions: {dim_ids}", flush=True)

    # Exclude specified dimensions
    excluded_dims = sorted(set(args.exclude_dims) & set(dim_ids))
    if excluded_dims:
        excluded_names = [DIM_NAMES.get(d, f"dim_{d}") for d in excluded_dims]
        print(f"Excluding {len(excluded_dims)} dims: {list(zip(excluded_dims, excluded_names))}", flush=True)
        dim_ids = [d for d in dim_ids if d not in excluded_dims]
        dims = {d: dims[d] for d in dim_ids}
        print(f"  Remaining: {len(dim_ids)} dimensions: {dim_ids}", flush=True)

    print("\nLoading contrast vectors...", flush=True)
    dim_vectors = {}
    for dim_id in dim_ids:
        dim_vectors[dim_id] = load_contrast_vector(dims[dim_id])
    print(f"  Loaded {len(dim_vectors)} vectors, each shape {dim_vectors[dim_ids[0]].shape}",
          flush=True)

    # Step 1: Compute pairwise overlap matrix (point estimate)
    print("\nComputing pairwise overlap matrix...")
    overlap, layer_profiles = compute_overlap_matrix(dim_vectors, dim_ids, layer_range)

    print("\nOverlap matrix (mean |cosine|, layers {}-{}):"
          .format(MIN_LAYER, N_LAYERS - 1))
    names = [DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]
    print(f"  {'':>20s}", end="")
    for n in names:
        print(f"  {n[:8]:>8s}", end="")
    print()
    for i, dim_id in enumerate(dim_ids):
        print(f"  {names[i]:>20s}", end="")
        for j in range(len(dim_ids)):
            print(f"  {overlap[i, j]:>8.3f}", end="")
        print()

    # Step 2: Load raw activations for bootstrap (if available)
    # For variant pipelines (e.g., _1), raw activations may not exist
    sample_acts_path = os.path.join(CONTRAST_ACT_DIR, dims[dim_ids[0]], "concept_activations.npz")
    has_raw_activations = os.path.isfile(sample_acts_path)

    boot_overlap = None
    if has_raw_activations:
        print("\nLoading raw activations for bootstrap...", flush=True)
        dim_activations = {}
        dim_labels = {}
        for dim_id in dim_ids:
            print(f"  Loading dim {dim_id}: {dims[dim_id]}...", flush=True)
            acts, labels = load_contrast_activations(dims[dim_id])
            dim_activations[dim_id] = acts
            dim_labels[dim_id] = labels
        print(f"  Loaded activations for {len(dim_activations)} dimensions", flush=True)

        # Step 3: Bootstrap
        print(f"\nRunning bootstrap ({args.n_bootstrap} iterations)...", flush=True)
        boot_overlap = bootstrap_overlap(
            dim_activations, dim_labels, dim_ids, layer_range,
            n_bootstrap=args.n_bootstrap, seed=SEED
        )
    else:
        print("\n[INFO] Raw activations not found — skipping bootstrap.", flush=True)

    # Step 4: Compute baseline overlaps
    baseline_results_0 = None
    baseline_results_18 = None

    if 0 in dim_ids:
        print("\nComputing overlap with dim 0 (entity baseline)...")
        baseline_results_0 = compute_baseline_overlap(
            dim_vectors, dim_ids, baseline_id=0, layer_range=layer_range
        )
        for r in baseline_results_0:
            print(f"  {r['name']:>20s}: |cos| = {r['mean_abs_cosine']:.4f}  ({r['category']})")

    if 18 in dim_ids:
        print("\nComputing overlap with dim 18 (sysprompt baseline)...")
        baseline_results_18 = compute_baseline_overlap(
            dim_vectors, dim_ids, baseline_id=18, layer_range=layer_range
        )
        for r in baseline_results_18:
            print(f"  {r['name']:>20s}: |cos| = {r['mean_abs_cosine']:.4f}  ({r['category']})")

    # Step 5: Save outputs
    print(f"\nSaving results to {OUTPUT_DIR}/")

    # overlap_matrix.npz
    save_kwargs = dict(
        overlap=overlap,
        dim_ids=np.array(dim_ids),
        dim_names=np.array([DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]),
        dim_categories=np.array([DIM_CATEGORIES.get(d, "Other") for d in dim_ids]),
        excluded_dims=np.array(excluded_dims),
        layer_range_start=MIN_LAYER,
        layer_range_end=N_LAYERS,
        n_bootstrap=args.n_bootstrap,
    )
    if boot_overlap is not None:
        save_kwargs["boot_overlap"] = boot_overlap
    out_data = str(data_subdir(OUTPUT_DIR))
    np.savez_compressed(os.path.join(out_data, variant_filename("overlap_matrix", ".npz")), **save_kwargs)
    print(f"  {variant_filename('overlap_matrix', '.npz')}")

    # overlap_matrix.csv
    csv_path = os.path.join(out_data, variant_filename("overlap_matrix", ".csv"))
    save_overlap_matrix_csv(overlap, dim_ids, csv_path)
    print(f"  {variant_filename('overlap_matrix', '.csv')}")

    # layer_profiles.npz
    np.savez_compressed(
        os.path.join(out_data, variant_filename("layer_profiles", ".npz")),
        layer_profiles=layer_profiles,
        dim_ids=np.array(dim_ids),
        dim_names=np.array([DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]),
    )
    print(f"  {variant_filename('layer_profiles', '.npz')}")

    # Baseline overlap CSVs (only if bootstrap was run)
    if baseline_results_0 is not None and boot_overlap is not None:
        path = os.path.join(out_data, variant_filename("baseline_overlap", ".csv"))
        save_baseline_overlap_csv(baseline_results_0, boot_overlap, dim_ids, 0, path)
        print(f"  {variant_filename('baseline_overlap', '.csv')}")

    if baseline_results_18 is not None and boot_overlap is not None:
        path = os.path.join(out_data, variant_filename("sysprompt_baseline_overlap", ".csv"))
        save_baseline_overlap_csv(baseline_results_18, boot_overlap, dim_ids, 18, path)
        print(f"  {variant_filename('sysprompt_baseline_overlap', '.csv')}")

    print("\nDone.")


if __name__ == "__main__":
    main()
