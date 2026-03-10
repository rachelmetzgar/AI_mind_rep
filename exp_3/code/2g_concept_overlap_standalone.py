#!/usr/bin/env python3
"""
Experiment 3, Phase 2g: Standalone Concept Overlap Analysis

Computes pairwise cosine similarity between standalone concept mean vectors
to understand how similar the model's representations are for different concepts
when no human/AI entity framing is involved.

This complements 2f (contrast overlap) which measures whether two dimensions
define the human-vs-AI *direction* the same way. Here we measure whether two
standalone concepts activate *similar patterns* in the model.

Key question: Do standalone concepts cluster by category (mental, physical,
pragmatic) in the model's representation, and how do the human/AI entity
concepts relate to other concepts?

Standalone dimensions:
    1-15: Same concepts as contrasts (but no entity framing)
    16: "human" (standalone entity concept)
    17: "ai" (standalone entity concept)
    18: "attention" (standalone, not sysprompt)
    20-23: sysprompt variants (talkto_human, talkto_ai, bare_human, bare_ai)

Outputs:
    concept_overlap_standalone/
        overlap_matrix.npz       Full pairwise overlap + bootstrap distributions
        overlap_matrix.csv       Readable pairwise table
        entity_overlap.csv       Per-dim overlap with dim 16 (human) and dim 17 (ai)
        layer_profiles.npz       Per-layer cosine for each pair

Usage:
    python 2g_concept_overlap_standalone.py
    python 2g_concept_overlap_standalone.py --n-bootstrap 500

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import argparse
import csv
import numpy as np

from config import config, add_variant_argument, set_variant, variant_filename, data_subdir

# ========================== CONFIG ========================== #

STANDALONE_ACT_DIR = str(config.RESULTS.concept_activations_standalone)
N_LAYERS = config.N_LAYERS
N_BOOTSTRAP = config.ANALYSIS.n_bootstrap
SEED = config.ANALYSIS.seed
MIN_LAYER = config.ANALYSIS.restricted_layer_start  # 6

OUTPUT_DIR = os.path.join(str(config.RESULTS.alignment), "concept_overlap", "standalone")

# Dimension display names for standalone
DIM_NAMES = {
    1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Human", 17: "AI", 18: "Attention",
    20: "SysPrompt (talkto human)", 21: "SysPrompt (talkto AI)",
    22: "SysPrompt (bare human)", 23: "SysPrompt (bare AI)",
}

DIM_CATEGORIES = {
    1: "Mental", 2: "Mental", 3: "Mental",
    4: "Mental", 5: "Mental", 6: "Mental", 7: "Mental",
    8: "Physical", 9: "Physical", 10: "Physical", 11: "Pragmatic",
    12: "Pragmatic", 13: "Pragmatic", 14: "Bio Ctrl", 15: "Shapes",
    16: "Entity", 17: "Entity", 18: "Mental",
    20: "SysPrompt", 21: "SysPrompt", 22: "SysPrompt", 23: "SysPrompt",
}

CATEGORY_ORDER = ["Mental", "Physical", "Pragmatic", "Entity",
                  "Bio Ctrl", "Shapes", "SysPrompt"]


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


def load_mean_vector(dim_name, act_dir=STANDALONE_ACT_DIR):
    """Load the mean concept vector per layer for a standalone dimension."""
    path = os.path.join(act_dir, dim_name, "mean_vectors_per_layer.npz")
    data = np.load(path)
    return data["mean_concept"]  # shape: (n_layers, hidden_dim)


def load_activations(dim_name, act_dir=STANDALONE_ACT_DIR):
    """Load raw activations for a standalone dimension."""
    path = os.path.join(act_dir, dim_name, "concept_activations.npz")
    data = np.load(path)
    return data["activations"]  # (n_prompts, n_layers, hidden)


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
            for layer in range(N_LAYERS):
                layer_profiles[i, j, layer] = cosine_sim(vi[layer], vj[layer])
                layer_profiles[j, i, layer] = layer_profiles[i, j, layer]

            cos_vals = [abs(layer_profiles[i, j, l]) for l in layer_range]
            mean_abs_cos = float(np.mean(cos_vals))
            overlap[i, j] = mean_abs_cos
            overlap[j, i] = mean_abs_cos

    return overlap, layer_profiles


def _vectorized_pairwise_abs_cosine(vectors, layer_indices):
    """
    Compute pairwise mean |cosine| across layers for a stack of mean vectors.

    Args:
        vectors: (n_dims, n_layers, hidden_dim) array
        layer_indices: array of layer indices to average over

    Returns:
        overlap: (n_dims, n_dims) matrix of mean |cosine|
    """
    V = vectors[:, layer_indices, :]
    norms = np.linalg.norm(V, axis=2, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    V_normed = V / norms

    n = V.shape[0]
    n_layers = len(layer_indices)
    overlap = np.zeros((n, n))

    for l_idx in range(n_layers):
        vl = V_normed[:, l_idx, :]
        cos_matrix = vl @ vl.T
        overlap += np.abs(cos_matrix)

    overlap /= n_layers
    return overlap


def bootstrap_overlap(dim_activations, dim_ids, layer_range,
                      n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap pairwise overlap by resampling prompts and recomputing mean vectors.

    For standalone concepts (no labels), we simply resample all prompts with
    replacement and recompute the mean vector.

    Args:
        dim_activations: dict {dim_id: (n_prompts, n_layers, hidden)}
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

    # Get dimensions for output array
    sample_acts = dim_activations[dim_ids[0]]
    n_layers_total = sample_acts.shape[1]
    hidden_dim = sample_acts.shape[2]

    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"  Bootstrap {b + 1}/{n_bootstrap}", flush=True)

        # Resample and compute mean vector for each dimension
        boot_vectors = np.zeros((n, n_layers_total, hidden_dim))
        for idx, dim_id in enumerate(dim_ids):
            acts = dim_activations[dim_id]
            boot_idx = rng.choice(acts.shape[0], size=acts.shape[0], replace=True)
            boot_vectors[idx] = acts[boot_idx].mean(axis=0)

        boot_overlap[:, :, b] = _vectorized_pairwise_abs_cosine(
            boot_vectors, layer_indices
        )

    return boot_overlap


def center_vectors(dim_vectors, dim_ids):
    """
    Center mean concept vectors by subtracting the global mean across all dimensions.

    Raw mean vectors are dominated by a shared "abstract concept processing" component,
    making all pairwise cosines ~0.96. Centering removes this shared component and
    reveals concept-specific structure (mean ~0.49, std ~0.22).

    Args:
        dim_vectors: dict {dim_id: (n_layers, hidden_dim) array}
        dim_ids: sorted list of dim IDs

    Returns:
        centered: dict {dim_id: (n_layers, hidden_dim) array} with global mean subtracted
        global_mean: (n_layers, hidden_dim) array — the subtracted mean
    """
    all_vecs = np.stack([dim_vectors[d] for d in dim_ids], axis=0)  # (n_dims, n_layers, hidden)
    global_mean = all_vecs.mean(axis=0)  # (n_layers, hidden)
    centered = {}
    for dim_id in dim_ids:
        centered[dim_id] = dim_vectors[dim_id] - global_mean
    return centered, global_mean


def bootstrap_overlap_centered(dim_activations, dim_ids, layer_range,
                               n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Bootstrap pairwise overlap on centered mean vectors.

    Each iteration: resample prompts per dim → compute mean → subtract that
    iteration's global mean → pairwise |cosine|.

    Args:
        dim_activations: dict {dim_id: (n_prompts, n_layers, hidden)}
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

    sample_acts = dim_activations[dim_ids[0]]
    n_layers_total = sample_acts.shape[1]
    hidden_dim = sample_acts.shape[2]

    for b in range(n_bootstrap):
        if (b + 1) % 200 == 0:
            print(f"  Centered bootstrap {b + 1}/{n_bootstrap}", flush=True)

        # Resample and compute mean vector for each dimension
        boot_vectors = np.zeros((n, n_layers_total, hidden_dim))
        for idx, dim_id in enumerate(dim_ids):
            acts = dim_activations[dim_id]
            boot_idx = rng.choice(acts.shape[0], size=acts.shape[0], replace=True)
            boot_vectors[idx] = acts[boot_idx].mean(axis=0)

        # Center: subtract global mean across dimensions for this iteration
        global_mean = boot_vectors.mean(axis=0)  # (n_layers, hidden)
        boot_vectors -= global_mean[np.newaxis, :, :]

        boot_overlap[:, :, b] = _vectorized_pairwise_abs_cosine(
            boot_vectors, layer_indices
        )

    return boot_overlap


def compute_entity_overlap(dim_vectors, dim_ids, entity_id, layer_range):
    """
    Compute per-dimension overlap with an entity reference dimension.

    Args:
        dim_vectors: dict {dim_id: mean vector array}
        dim_ids: sorted list of dim IDs
        entity_id: dim ID of the entity reference (16=human, 17=ai)
        layer_range: range of layers

    Returns:
        results: list of dicts with overlap info
    """
    if entity_id not in dim_vectors:
        raise ValueError(f"Entity dim {entity_id} not found in loaded vectors")

    entity_vec = dim_vectors[entity_id]
    results = []

    for dim_id in dim_ids:
        if dim_id == entity_id:
            continue
        cv = dim_vectors[dim_id]
        per_layer = [cosine_sim(cv[l], entity_vec[l]) for l in range(N_LAYERS)]
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
    """Save overlap matrix as CSV."""
    names = [DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + names)
        for i, dim_id in enumerate(dim_ids):
            row = [names[i]] + [f"{overlap[i, j]:.4f}" for j in range(len(dim_ids))]
            writer.writerow(row)


def save_entity_overlap_csv(entity_results_16, entity_results_17,
                            boot_overlap, dim_ids, filepath):
    """Save per-dimension overlap with both entity references (dim 16, dim 17)."""
    # Build lookup from entity results
    r16 = {r["dim_id"]: r for r in entity_results_16} if entity_results_16 else {}
    r17 = {r["dim_id"]: r for r in entity_results_17} if entity_results_17 else {}

    # Indices for bootstrap CIs
    idx_16 = dim_ids.index(16) if 16 in dim_ids else None
    idx_17 = dim_ids.index(17) if 17 in dim_ids else None

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dim_id", "name", "category",
                         "overlap_human_mean", "overlap_human_ci_lo", "overlap_human_ci_hi",
                         "overlap_ai_mean", "overlap_ai_ci_lo", "overlap_ai_ci_hi"])

        for dim_id in dim_ids:
            if dim_id in (16, 17):
                continue
            name = DIM_NAMES.get(dim_id, f"dim_{dim_id}")
            cat = DIM_CATEGORIES.get(dim_id, "Other")
            dim_idx = dim_ids.index(dim_id)

            # Human entity overlap
            h_mean = r16[dim_id]["mean_abs_cosine"] if dim_id in r16 else float('nan')
            h_lo, h_hi = float('nan'), float('nan')
            if idx_16 is not None:
                boot_vals = boot_overlap[dim_idx, idx_16, :]
                h_lo = float(np.percentile(boot_vals, 2.5))
                h_hi = float(np.percentile(boot_vals, 97.5))

            # AI entity overlap
            a_mean = r17[dim_id]["mean_abs_cosine"] if dim_id in r17 else float('nan')
            a_lo, a_hi = float('nan'), float('nan')
            if idx_17 is not None:
                boot_vals = boot_overlap[dim_idx, idx_17, :]
                a_lo = float(np.percentile(boot_vals, 2.5))
                a_hi = float(np.percentile(boot_vals, 97.5))

            writer.writerow([
                dim_id, name, cat,
                f"{h_mean:.4f}", f"{h_lo:.4f}", f"{h_hi:.4f}",
                f"{a_mean:.4f}", f"{a_lo:.4f}", f"{a_hi:.4f}",
            ])


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 3 Phase 2g: Standalone concept overlap analysis"
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help=f"Bootstrap iterations (default: {N_BOOTSTRAP})")
    add_variant_argument(parser)
    args = parser.parse_args()

    if args.variant:
        set_variant(args.variant)
        global STANDALONE_ACT_DIR, OUTPUT_DIR
        STANDALONE_ACT_DIR = str(config.RESULTS.concept_activations_standalone)
        OUTPUT_DIR = os.path.join(str(config.RESULTS.concept_overlap), "standalone")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    layer_range = range(MIN_LAYER, N_LAYERS)

    # Discover and load all standalone dimensions
    dims = discover_dimensions(STANDALONE_ACT_DIR)
    dim_ids = sorted(dims.keys())
    print(f"Found {len(dim_ids)} standalone dimensions: {dim_ids}", flush=True)

    print("\nLoading mean concept vectors...", flush=True)
    dim_vectors = {}
    for dim_id in dim_ids:
        dim_vectors[dim_id] = load_mean_vector(dims[dim_id])
    print(f"  Loaded {len(dim_vectors)} vectors, each shape {dim_vectors[dim_ids[0]].shape}",
          flush=True)

    # ================================================================
    # RAW ANALYSIS
    # ================================================================

    # Step 1: Compute pairwise overlap matrix (point estimate)
    print("\nComputing pairwise overlap matrix (raw)...")
    overlap, layer_profiles = compute_overlap_matrix(dim_vectors, dim_ids, layer_range)

    # Print summary
    off_diag_raw = []
    n = len(dim_ids)
    for i in range(n):
        for j in range(i+1, n):
            off_diag_raw.append(overlap[i, j])
    print(f"  Raw off-diagonal: mean={np.mean(off_diag_raw):.3f}, "
          f"std={np.std(off_diag_raw):.3f}, "
          f"range=[{np.min(off_diag_raw):.3f}, {np.max(off_diag_raw):.3f}]")

    # ================================================================
    # CENTERED ANALYSIS
    # ================================================================

    # Step 2: Center vectors and recompute overlap
    print("\nCentering mean vectors (subtracting global mean across dimensions)...")
    centered_vectors, global_mean = center_vectors(dim_vectors, dim_ids)

    print("Computing pairwise overlap matrix (centered)...")
    overlap_centered, layer_profiles_centered = compute_overlap_matrix(
        centered_vectors, dim_ids, layer_range
    )

    off_diag_centered = []
    for i in range(n):
        for j in range(i+1, n):
            off_diag_centered.append(overlap_centered[i, j])
    print(f"  Centered off-diagonal: mean={np.mean(off_diag_centered):.3f}, "
          f"std={np.std(off_diag_centered):.3f}, "
          f"range=[{np.min(off_diag_centered):.3f}, {np.max(off_diag_centered):.3f}]")

    # Step 3: Load raw activations for bootstrap
    print("\nLoading raw activations for bootstrap...", flush=True)
    dim_activations = {}
    for dim_id in dim_ids:
        print(f"  Loading dim {dim_id}: {dims[dim_id]}...", flush=True)
        dim_activations[dim_id] = load_activations(dims[dim_id])
    print(f"  Loaded activations for {len(dim_activations)} dimensions", flush=True)

    # Step 4: Bootstrap (raw)
    print(f"\nRunning raw bootstrap ({args.n_bootstrap} iterations)...", flush=True)
    boot_overlap = bootstrap_overlap(
        dim_activations, dim_ids, layer_range,
        n_bootstrap=args.n_bootstrap, seed=SEED
    )

    # Step 5: Bootstrap (centered)
    print(f"\nRunning centered bootstrap ({args.n_bootstrap} iterations)...", flush=True)
    boot_overlap_centered = bootstrap_overlap_centered(
        dim_activations, dim_ids, layer_range,
        n_bootstrap=args.n_bootstrap, seed=SEED
    )

    # Step 6: Compute entity reference overlaps (on centered vectors)
    entity_results_16 = None
    entity_results_17 = None

    if 16 in dim_ids:
        print("\nComputing centered overlap with dim 16 (human concept)...")
        entity_results_16 = compute_entity_overlap(
            centered_vectors, dim_ids, entity_id=16, layer_range=layer_range
        )
        for r in entity_results_16:
            print(f"  {r['name']:>25s}: |cos| = {r['mean_abs_cosine']:.4f}  ({r['category']})")

    if 17 in dim_ids:
        print("\nComputing centered overlap with dim 17 (AI concept)...")
        entity_results_17 = compute_entity_overlap(
            centered_vectors, dim_ids, entity_id=17, layer_range=layer_range
        )
        for r in entity_results_17:
            print(f"  {r['name']:>25s}: |cos| = {r['mean_abs_cosine']:.4f}  ({r['category']})")

    # ================================================================
    # SAVE OUTPUTS
    # ================================================================
    print(f"\nSaving results to {OUTPUT_DIR}/")

    # --- Raw outputs ---
    out_data = str(data_subdir(OUTPUT_DIR))
    np.savez_compressed(
        os.path.join(out_data, variant_filename("overlap_matrix", ".npz")),
        overlap=overlap,
        boot_overlap=boot_overlap,
        dim_ids=np.array(dim_ids),
        dim_names=np.array([DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]),
        dim_categories=np.array([DIM_CATEGORIES.get(d, "Other") for d in dim_ids]),
        layer_range_start=MIN_LAYER,
        layer_range_end=N_LAYERS,
        n_bootstrap=args.n_bootstrap,
    )
    print(f"  {variant_filename('overlap_matrix', '.npz')} (raw)")

    csv_path = os.path.join(out_data, variant_filename("overlap_matrix", ".csv"))
    save_overlap_matrix_csv(overlap, dim_ids, csv_path)
    print(f"  {variant_filename('overlap_matrix', '.csv')} (raw)")

    np.savez_compressed(
        os.path.join(out_data, variant_filename("layer_profiles", ".npz")),
        layer_profiles=layer_profiles,
        dim_ids=np.array(dim_ids),
        dim_names=np.array([DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]),
    )
    print(f"  {variant_filename('layer_profiles', '.npz')} (raw)")

    # --- Centered outputs ---
    np.savez_compressed(
        os.path.join(out_data, variant_filename("overlap_matrix_centered", ".npz")),
        overlap=overlap_centered,
        boot_overlap=boot_overlap_centered,
        dim_ids=np.array(dim_ids),
        dim_names=np.array([DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]),
        dim_categories=np.array([DIM_CATEGORIES.get(d, "Other") for d in dim_ids]),
        layer_range_start=MIN_LAYER,
        layer_range_end=N_LAYERS,
        n_bootstrap=args.n_bootstrap,
    )
    print(f"  {variant_filename('overlap_matrix_centered', '.npz')}")

    csv_path_c = os.path.join(out_data, variant_filename("overlap_matrix_centered", ".csv"))
    save_overlap_matrix_csv(overlap_centered, dim_ids, csv_path_c)
    print(f"  {variant_filename('overlap_matrix_centered', '.csv')}")

    np.savez_compressed(
        os.path.join(out_data, variant_filename("layer_profiles_centered", ".npz")),
        layer_profiles=layer_profiles_centered,
        dim_ids=np.array(dim_ids),
        dim_names=np.array([DIM_NAMES.get(d, f"dim_{d}") for d in dim_ids]),
    )
    print(f"  {variant_filename('layer_profiles_centered', '.npz')}")

    # Entity overlap CSV (from centered vectors with centered bootstrap CIs)
    if entity_results_16 is not None or entity_results_17 is not None:
        path = os.path.join(out_data, variant_filename("entity_overlap", ".csv"))
        save_entity_overlap_csv(
            entity_results_16, entity_results_17,
            boot_overlap_centered, dim_ids, path
        )
        print(f"  {variant_filename('entity_overlap', '.csv')} (centered)")

    print("\nDone.")


if __name__ == "__main__":
    main()
