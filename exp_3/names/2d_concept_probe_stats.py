#!/usr/bin/env python3
"""
Consolidated statistical analysis for concept-probe alignment.

Computes ALL statistical tests and writes structured JSON, CSV tables,
and an auto-generated methods description.

Statistical tests:
  1. Per-dimension permutation tests (from 2c) — for each dim x probe x layer_range
  2. Per-layer permutation tests — test at each of 41 layers independently
  3. Pairwise dimension comparisons — bootstrap "is dim X > dim Y?"
  4. Category-level alignment — bootstrap from prompt level
  5. Pairwise category comparisons — bootstrap category-level differences
  6. FDR correction (Benjamini-Hochberg, q=0.05) on pairwise comparisons

Inputs:
    data/concept_activations/contrasts/{dim}/concept_activations.npz
    Exp 2 probe weights (.pth files)
    data/alignment/{dim}/{probe}/alignment_results.json  (per-layer cosines)
    data/concept_probes/{dim}/accuracy_summary.pkl

Outputs:
    results/concept_probe_alignment/summaries/alignment_stats.json
    results/concept_probe_alignment/summaries/dimension_table.csv
    results/concept_probe_alignment/summaries/category_table.csv
    results/concept_probe_alignment/summaries/pairwise_dimensions.csv
    results/concept_probe_alignment/summaries/pairwise_categories.csv
    results/concept_probe_alignment/methods/statistical_methods.md

Usage:
    python 2d_concept_probe_stats.py

Env: llama2_env (needs numpy, torch; no GPU required)
Rachel C. Metzgar, Feb 2026
"""

import os
import json
import csv
import time
import pickle
import numpy as np
import torch

# ========================== CONFIG ========================== #

BASE = os.path.dirname(os.path.abspath(__file__))
ACT_ROOT = os.path.join(BASE, "data", "concept_activations", "contrasts")
ALIGN_ROOT = os.path.join(BASE, "data", "alignment")
PROBE_DATA_ROOT = os.path.join(BASE, "data", "concept_probes")
OUT_ROOT = os.path.join(BASE, "results", "concept_probe_alignment")
SUMMARIES_DIR = os.path.join(OUT_ROOT, "summaries")
METHODS_DIR = os.path.join(OUT_ROOT, "methods")

# Exp 2 probe weights
EXP2_PROBE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(BASE)),  # up to ai_mind_rep
    "exp_2", "llama_exp_2b-13B-chat", "data", "probe_checkpoints"
)

N_LAYERS = 41
HIDDEN_DIM = 5120
N_PERMUTATIONS = 10000
N_BOOTSTRAP = 10000
RESTRICTED_LAYER_START = 6
RNG = np.random.default_rng(42)

CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 17],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Human vs AI (General)":  [0],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "SysPrompt": [18],
}

DIM_NAMES = {
    0: "0_baseline", 1: "1_phenomenology", 2: "2_emotions",
    3: "3_agency", 4: "4_intentions", 5: "5_prediction",
    6: "6_cognitive", 7: "7_social", 8: "8_embodiment",
    9: "9_roles", 10: "10_animacy", 11: "11_formality",
    12: "12_expertise", 13: "13_helpfulness", 14: "14_biological",
    15: "15_shapes", 17: "17_attention",
    18: "18_sysprompt_labeled",
}

DIM_LABELS = {
    0: "Human vs AI (General)", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive",
    7: "Social cognition", 8: "Embodiment", 9: "Roles", 10: "Animacy",
    11: "Formality", 12: "Expertise", 13: "Helpfulness",
    14: "Biological", 15: "Shapes",
    17: "Attention", 18: "System prompt",
}

PROBE_TYPES = ["control_probe", "reading_probe"]
LAYER_RANGES = {
    "all_layers": None,
    "layers_6plus": slice(RESTRICTED_LAYER_START, None),
}


def dim_category(dim_id):
    for cat, ids in CATEGORIES.items():
        if dim_id in ids:
            return cat
    return "Other"


# ========================== LOADING ========================== #

def load_exp2_probe_weights(probe_type):
    """Load Exp 2 probe weights for all layers. Returns (41, 5120) array."""
    probe_dir = os.path.join(EXP2_PROBE_ROOT, probe_type)
    weights = np.zeros((N_LAYERS, HIDDEN_DIM))
    for layer in range(N_LAYERS):
        path = os.path.join(probe_dir, f"human_ai_probe_at_layer_{layer}.pth")
        if not os.path.isfile(path):
            print(f"  [WARN] Missing probe weight: {path}")
            continue
        state = torch.load(path, map_location="cpu", weights_only=True)
        w = state["proj.0.weight"].squeeze().numpy()
        weights[layer] = w
    return weights


def load_concept_activations(dim_name):
    """Load per-prompt activations and labels."""
    path = os.path.join(ACT_ROOT, dim_name, "concept_activations.npz")
    if not os.path.isfile(path):
        return None, None
    data = np.load(path)
    return data["activations"], data["labels"]  # (80, 41, 5120), (80,)


def load_per_layer_cosines(dim_name, probe_type):
    """Load precomputed per-layer cosine alignment from alignment JSON."""
    jp = os.path.join(ALIGN_ROOT, dim_name, probe_type, "alignment_results.json")
    if not os.path.isfile(jp):
        return None
    with open(jp) as f:
        aj = json.load(f)
    return aj.get("mean_to_2b", [])


def load_accuracy(dim_name):
    """Load per-layer accuracy from pickle."""
    pkl = os.path.join(PROBE_DATA_ROOT, dim_name, "accuracy_summary.pkl")
    if not os.path.isfile(pkl):
        return None
    with open(pkl, "rb") as f:
        s = pickle.load(f)
    accs = s.get("acc", [])
    return list(accs) if accs is not None and len(accs) > 0 else None


# ========================== CORE STATISTICS ========================== #

def precompute_projections(activations, probe_weights):
    """
    Precompute per-prompt projections onto the unit-normalized probe direction.
    Returns: (n_prompts, n_layers) array
    """
    norms = np.linalg.norm(probe_weights, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    probe_unit = probe_weights / norms
    return np.einsum('ild,ld->il', activations, probe_unit)


def compute_cosine_alignment(activations, labels, probe_weights, layer_slice=None):
    """Compute cosine-based mean alignment (for reporting alongside projection)."""
    human_mask = labels == 1
    mean_human = activations[human_mask].mean(axis=0)
    mean_ai = activations[~human_mask].mean(axis=0)
    concept_vec = mean_human - mean_ai

    dot = np.sum(concept_vec * probe_weights, axis=1)
    norm_c = np.linalg.norm(concept_vec, axis=1)
    norm_p = np.linalg.norm(probe_weights, axis=1)
    denom = norm_c * norm_p
    denom[denom == 0] = 1e-10
    cosines = dot / denom

    if layer_slice is not None:
        cosines = cosines[layer_slice]
    return cosines


def permutation_test(projections, labels, layer_slice=None, n_perm=N_PERMUTATIONS):
    """
    Fast permutation test using precomputed projections.
    Returns: (observed, p_value, null_distribution)
    """
    n_prompts = len(labels)
    n_human = int(np.sum(labels == 1))
    n_ai = n_prompts - n_human
    human_mask = labels == 1

    if layer_slice is not None:
        prompt_scores = projections[:, layer_slice].mean(axis=1)
    else:
        prompt_scores = projections.mean(axis=1)

    observed = float(prompt_scores[human_mask].mean() - prompt_scores[~human_mask].mean())

    indicators = np.zeros((n_perm, n_prompts))
    for i in range(n_perm):
        sel = RNG.choice(n_prompts, size=n_human, replace=False)
        indicators[i, sel] = 1.0

    total = prompt_scores.sum()
    sel_sums = indicators @ prompt_scores
    null_stats = sel_sums / n_human - (total - sel_sums) / n_ai

    p_value = float(np.mean(null_stats >= observed))
    return observed, p_value, null_stats


def bootstrap_ci(projections, labels, layer_slice=None, n_boot=N_BOOTSTRAP):
    """Bootstrap CI by resampling prompts within each group."""
    if layer_slice is not None:
        prompt_scores = projections[:, layer_slice].mean(axis=1)
    else:
        prompt_scores = projections.mean(axis=1)

    human_mask = labels == 1
    human_scores = prompt_scores[human_mask]
    ai_scores = prompt_scores[~human_mask]

    boot_human = RNG.choice(human_scores, size=(n_boot, len(human_scores)), replace=True)
    boot_ai = RNG.choice(ai_scores, size=(n_boot, len(ai_scores)), replace=True)
    boot_stats = boot_human.mean(axis=1) - boot_ai.mean(axis=1)

    ci_lo = float(np.percentile(boot_stats, 2.5))
    ci_hi = float(np.percentile(boot_stats, 97.5))
    return ci_lo, ci_hi, boot_stats


def permutation_test_single_layer(projections, labels, layer_idx, n_perm=N_PERMUTATIONS):
    """Permutation test at a single layer. Returns (observed, p_value)."""
    n_prompts = len(labels)
    n_human = int(np.sum(labels == 1))
    n_ai = n_prompts - n_human
    human_mask = labels == 1

    prompt_scores = projections[:, layer_idx]
    observed = float(prompt_scores[human_mask].mean() - prompt_scores[~human_mask].mean())

    # Reuse the same permutation matrix approach
    indicators = np.zeros((n_perm, n_prompts))
    for i in range(n_perm):
        sel = RNG.choice(n_prompts, size=n_human, replace=False)
        indicators[i, sel] = 1.0

    total = prompt_scores.sum()
    sel_sums = indicators @ prompt_scores
    null_stats = sel_sums / n_human - (total - sel_sums) / n_ai

    p_value = float(np.mean(null_stats >= observed))
    return observed, p_value


def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def benjamini_hochberg(p_values, q=0.05):
    """
    Benjamini-Hochberg FDR correction.
    Returns array of adjusted p-values in the same order as input.
    """
    p = np.array(p_values)
    n = len(p)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    adjusted = np.zeros(n)
    # BH adjustment: p_adj[i] = min(p[i] * n / rank, 1)
    # then enforce monotonicity from bottom up
    for i in range(n - 1, -1, -1):
        rank = i + 1
        raw_adj = sorted_p[i] * n / rank
        if i == n - 1:
            adjusted[sorted_idx[i]] = min(raw_adj, 1.0)
        else:
            adjusted[sorted_idx[i]] = min(raw_adj, adjusted[sorted_idx[i + 1]])
    return adjusted


# ========================== PAIRWISE COMPARISONS ========================== #

def pairwise_dimension_bootstrap(dim_projections, dim_labels, dim_ids,
                                 layer_slice=None, n_boot=N_BOOTSTRAP):
    """
    Bootstrap pairwise dimension comparisons.
    For each pair: resample prompts within each dim, compute alignment diff.
    Returns list of dicts.
    """
    # First compute per-dim bootstrap distributions of alignment
    dim_boot_dists = {}
    for dim_id in dim_ids:
        proj = dim_projections[dim_id]
        lab = dim_labels[dim_id]
        if layer_slice is not None:
            scores = proj[:, layer_slice].mean(axis=1)
        else:
            scores = proj.mean(axis=1)

        human_mask = lab == 1
        h_scores = scores[human_mask]
        a_scores = scores[~human_mask]

        boot_h = RNG.choice(h_scores, size=(n_boot, len(h_scores)), replace=True)
        boot_a = RNG.choice(a_scores, size=(n_boot, len(a_scores)), replace=True)
        dim_boot_dists[dim_id] = boot_h.mean(axis=1) - boot_a.mean(axis=1)

    # Now compute pairwise
    results = []
    sorted_ids = sorted(dim_ids)
    for i, d_a in enumerate(sorted_ids):
        for d_b in sorted_ids[i + 1:]:
            boot_diff = dim_boot_dists[d_a] - dim_boot_dists[d_b]
            obs_diff = float(np.mean(dim_boot_dists[d_a]) - np.mean(dim_boot_dists[d_b]))
            p_val = 2 * min(float(np.mean(boot_diff <= 0)),
                            float(np.mean(boot_diff >= 0)))
            p_val = min(p_val, 1.0)
            ci_lo = float(np.percentile(boot_diff, 2.5))
            ci_hi = float(np.percentile(boot_diff, 97.5))
            results.append({
                "dim_a": d_a, "dim_b": d_b,
                "name_a": DIM_NAMES[d_a], "name_b": DIM_NAMES[d_b],
                "diff": obs_diff, "p_value": p_val,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            })
    return results


def category_bootstrap(dim_projections, dim_labels, layer_slice=None,
                       n_boot=N_BOOTSTRAP):
    """
    Category-level alignment via bootstrap from the prompt level.
    For each bootstrap iteration: resample prompts within each dim,
    compute dim alignment, average across dims in category.
    Returns dict of {cat_name: {mean, ci_lo, ci_hi, boot_dist}}.
    """
    # Per-dim bootstrap distributions
    dim_boot = {}
    for dim_id in dim_projections:
        proj = dim_projections[dim_id]
        lab = dim_labels[dim_id]
        if layer_slice is not None:
            scores = proj[:, layer_slice].mean(axis=1)
        else:
            scores = proj.mean(axis=1)
        human_mask = lab == 1
        h = scores[human_mask]
        a = scores[~human_mask]
        boot_h = RNG.choice(h, size=(n_boot, len(h)), replace=True)
        boot_a = RNG.choice(a, size=(n_boot, len(a)), replace=True)
        dim_boot[dim_id] = boot_h.mean(axis=1) - boot_a.mean(axis=1)

    cat_results = {}
    for cat_name, cat_ids in CATEGORIES.items():
        active = [d for d in cat_ids if d in dim_boot]
        if not active:
            continue
        # Average dim-level bootstraps within category
        cat_boot = np.mean([dim_boot[d] for d in active], axis=0)
        cat_results[cat_name] = {
            "mean": float(np.mean(cat_boot)),
            "ci_lo": float(np.percentile(cat_boot, 2.5)),
            "ci_hi": float(np.percentile(cat_boot, 97.5)),
            "n_dims": len(active),
            "dim_ids": active,
            "_boot_dist": cat_boot,
        }
    return cat_results


def pairwise_category_bootstrap(cat_results, n_boot=N_BOOTSTRAP):
    """
    Pairwise category comparisons from bootstrap distributions.
    Returns list of dicts.
    """
    results = []
    cat_names = sorted(cat_results.keys())
    for i, ca in enumerate(cat_names):
        for cb in cat_names[i + 1:]:
            boot_a = cat_results[ca]["_boot_dist"]
            boot_b = cat_results[cb]["_boot_dist"]
            boot_diff = boot_a - boot_b
            obs_diff = float(np.mean(boot_diff))
            p_val = 2 * min(float(np.mean(boot_diff <= 0)),
                            float(np.mean(boot_diff >= 0)))
            p_val = min(p_val, 1.0)
            ci_lo = float(np.percentile(boot_diff, 2.5))
            ci_hi = float(np.percentile(boot_diff, 97.5))
            results.append({
                "cat_a": ca, "cat_b": cb,
                "diff": obs_diff, "p_value": p_val,
                "ci_lo": ci_lo, "ci_hi": ci_hi,
            })
    return results


# ========================== MAIN ========================== #

def main():
    for d in [SUMMARIES_DIR, METHODS_DIR]:
        os.makedirs(d, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("CONCEPT-PROBE ALIGNMENT: CONSOLIDATED STATISTICS")
    print(f"  {N_PERMUTATIONS} permutations, {N_BOOTSTRAP} bootstrap resamples")
    print(f"  Restricted layer analysis: layers {RESTRICTED_LAYER_START}+")
    print("=" * 80)

    # ── Load probe weights ──
    print("\nLoading Exp 2 probe weights...")
    probe_weights = {}
    for pt in PROBE_TYPES:
        probe_weights[pt] = load_exp2_probe_weights(pt)
    print("  Done.")

    # ── Discover available dimensions ──
    available_dims = {}
    for dim_id, dim_name in DIM_NAMES.items():
        act_path = os.path.join(ACT_ROOT, dim_name, "concept_activations.npz")
        if os.path.isfile(act_path):
            available_dims[dim_id] = dim_name
        else:
            print(f"  [MISS] Dim {dim_id} ({dim_name}): no activation file")
    print(f"Found {len(available_dims)}/{len(DIM_NAMES)} dims with activations.\n")

    # ══════════════════════════════════════════════════════════════
    #  1. PER-DIMENSION PERMUTATION TESTS (matches 2c output)
    # ══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("1. PER-DIMENSION PERMUTATION TESTS")
    print("=" * 80)

    all_dim_results = {}
    dim_projections = {}   # {dim_id: {probe_type: (80, 41)}}
    dim_labels_cache = {}  # {dim_id: labels}

    for dim_id in sorted(available_dims.keys()):
        dim_name = available_dims[dim_id]
        cat = dim_category(dim_id)
        print(f"\n{'─' * 70}")
        print(f"Dim {dim_id}: {dim_name} [{cat}]")

        activations, labels = load_concept_activations(dim_name)
        if activations is None:
            print("  [SKIP] No activations found.")
            continue

        n_human = int(np.sum(labels == 1))
        n_ai = int(np.sum(labels == 0))
        print(f"  {activations.shape[0]} prompts ({n_human}H/{n_ai}A), "
              f"{activations.shape[1]} layers, dim {activations.shape[2]}")

        dim_labels_cache[dim_id] = labels

        dim_result = {
            "dim_id": dim_id, "dim_name": dim_name, "label": DIM_LABELS[dim_id],
            "category": cat, "n_human": n_human, "n_ai": n_ai,
        }

        # Load per-layer cosines and accuracy
        for pt in PROBE_TYPES:
            cosines = load_per_layer_cosines(dim_name, pt)
            if cosines:
                dim_result[f"{pt}_per_layer_cosines"] = cosines
        accs = load_accuracy(dim_name)
        if accs:
            dim_result["per_layer_accuracy"] = accs

        dim_projections[dim_id] = {}

        for pt in PROBE_TYPES:
            pt_short = pt.replace("_probe", "")
            pw = probe_weights[pt]

            projections = precompute_projections(activations, pw)
            dim_projections[dim_id][pt] = projections

            for lr_label, ls in LAYER_RANGES.items():
                key = f"{pt}_{lr_label}"
                t0 = time.time()

                cosine_vals = compute_cosine_alignment(activations, labels, pw, ls)
                cosine_obs = float(np.mean(cosine_vals))

                proj_obs, p_val, null_dist = permutation_test(
                    projections, labels, ls, N_PERMUTATIONS)

                ci_lo, ci_hi, _ = bootstrap_ci(projections, labels, ls, N_BOOTSTRAP)

                dt = time.time() - t0
                sig = sig_label(p_val)
                null_mean = float(np.mean(null_dist))
                null_std = float(np.std(null_dist))

                print(f"  {pt_short:>7}/{lr_label:<12} "
                      f"cos={cosine_obs:>7.4f}  proj={proj_obs:>7.4f}  "
                      f"p={p_val:.4f} {sig:>4}  "
                      f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  ({dt:.1f}s)")

                dim_result[key] = {
                    "observed_cosine": cosine_obs,
                    "observed_projection": proj_obs,
                    "p_value": p_val,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "null_mean": null_mean, "null_std": null_std,
                    "sig": sig,
                }

        # Free large activations array — we keep projections (80,41)
        del activations
        all_dim_results[dim_id] = dim_result

    # ── FDR correction on per-dimension p-values ──
    # Apply BH-FDR within each probe × layer_range family (18 tests each)
    print(f"\n{'─' * 70}")
    print("FDR correction on per-dimension permutation tests")
    for pt in PROBE_TYPES:
        for lr_label_key in LAYER_RANGES:
            key = f"{pt}_{lr_label_key}"
            # Collect p-values in dim_id order
            ordered_ids = sorted(all_dim_results.keys())
            raw_ps = []
            for dim_id in ordered_ids:
                r = all_dim_results[dim_id].get(key)
                if r is not None:
                    raw_ps.append(r["p_value"])
                else:
                    raw_ps.append(1.0)
            adj_ps = benjamini_hochberg(raw_ps)
            for dim_id, adj_p in zip(ordered_ids, adj_ps):
                r = all_dim_results[dim_id].get(key)
                if r is not None:
                    r["p_adjusted"] = float(adj_p)
                    r["sig_fdr"] = sig_label(float(adj_p))
            n_sig_raw = sum(1 for p in raw_ps if p < 0.05)
            n_sig_fdr = sum(1 for p in adj_ps if p < 0.05)
            print(f"  {key}: {n_sig_raw}/{len(raw_ps)} sig raw → "
                  f"{n_sig_fdr}/{len(raw_ps)} sig after FDR")

    # ══════════════════════════════════════════════════════════════
    #  2. PER-LAYER PERMUTATION TESTS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("2. PER-LAYER PERMUTATION TESTS")
    print("=" * 80)

    for dim_id in sorted(dim_projections.keys()):
        labels = dim_labels_cache[dim_id]
        for pt in PROBE_TYPES:
            projections = dim_projections[dim_id][pt]
            layer_results = []
            for layer in range(N_LAYERS):
                obs, p = permutation_test_single_layer(projections, labels, layer)
                layer_results.append({
                    "layer": layer,
                    "observed_projection": obs,
                    "p_value": p,
                })
            key = f"{pt}_per_layer_perm"
            all_dim_results[dim_id][key] = layer_results

        n_sig_ctrl = sum(1 for r in all_dim_results[dim_id].get(
            "control_probe_per_layer_perm", []) if r["p_value"] < 0.05)
        n_sig_read = sum(1 for r in all_dim_results[dim_id].get(
            "reading_probe_per_layer_perm", []) if r["p_value"] < 0.05)
        print(f"  Dim {dim_id:>2} ({DIM_NAMES[dim_id]:<20}): "
              f"ctrl {n_sig_ctrl}/41 sig layers, "
              f"read {n_sig_read}/41 sig layers")

    # ══════════════════════════════════════════════════════════════
    #  3. PAIRWISE DIMENSION COMPARISONS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("3. PAIRWISE DIMENSION COMPARISONS (bootstrap)")
    print("=" * 80)

    pairwise_dim_results = {}
    active_dim_ids = sorted(dim_projections.keys())

    for pt in PROBE_TYPES:
        for lr_label, ls in LAYER_RANGES.items():
            key = f"{pt}_{lr_label}"
            print(f"\n  {key}...")

            pt_proj = {d: dim_projections[d][pt] for d in active_dim_ids}
            pairs = pairwise_dimension_bootstrap(
                pt_proj, dim_labels_cache, active_dim_ids, ls, N_BOOTSTRAP)

            # FDR correction
            raw_ps = [r["p_value"] for r in pairs]
            adj_ps = benjamini_hochberg(raw_ps)
            for r, adj_p in zip(pairs, adj_ps):
                r["p_adjusted"] = float(adj_p)
                r["sig_raw"] = sig_label(r["p_value"])
                r["sig_fdr"] = sig_label(float(adj_p))

            n_sig = sum(1 for r in pairs if r["p_adjusted"] < 0.05)
            print(f"    {len(pairs)} pairs, {n_sig} significant after FDR")
            pairwise_dim_results[key] = pairs

    # ══════════════════════════════════════════════════════════════
    #  4. CATEGORY-LEVEL ALIGNMENT (bootstrap from prompt level)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("4. CATEGORY-LEVEL ALIGNMENT (bootstrap)")
    print("=" * 80)

    category_results = {}
    for pt in PROBE_TYPES:
        for lr_label, ls in LAYER_RANGES.items():
            key = f"{pt}_{lr_label}"
            pt_proj = {d: dim_projections[d][pt] for d in active_dim_ids}
            cat_res = category_bootstrap(pt_proj, dim_labels_cache, ls, N_BOOTSTRAP)

            # Strip boot distribution for serialization
            serializable = {}
            for cn, cr in cat_res.items():
                serializable[cn] = {k: v for k, v in cr.items() if k != "_boot_dist"}
            category_results[key] = serializable

            print(f"\n  {key}:")
            for cn in ["Mental", "Physical", "Pragmatic", "SysPrompt",
                        "Human vs AI (General)", "Bio Ctrl", "Shapes"]:
                if cn in serializable:
                    r = serializable[cn]
                    print(f"    {cn:<12} mean={r['mean']:.4f}  "
                          f"CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  "
                          f"(n_dims={r['n_dims']})")

    # ══════════════════════════════════════════════════════════════
    #  5. PAIRWISE CATEGORY COMPARISONS
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("5. PAIRWISE CATEGORY COMPARISONS (bootstrap)")
    print("=" * 80)

    pairwise_cat_results = {}
    for pt in PROBE_TYPES:
        for lr_label, ls in LAYER_RANGES.items():
            key = f"{pt}_{lr_label}"

            # Recompute with boot distributions included
            pt_proj = {d: dim_projections[d][pt] for d in active_dim_ids}
            cat_res_with_boot = category_bootstrap(
                pt_proj, dim_labels_cache, ls, N_BOOTSTRAP)

            pairs = pairwise_category_bootstrap(cat_res_with_boot, N_BOOTSTRAP)

            # FDR correction
            raw_ps = [r["p_value"] for r in pairs]
            adj_ps = benjamini_hochberg(raw_ps)
            for r, adj_p in zip(pairs, adj_ps):
                r["p_adjusted"] = float(adj_p)
                r["sig_raw"] = sig_label(r["p_value"])
                r["sig_fdr"] = sig_label(float(adj_p))

            n_sig = sum(1 for r in pairs if r["p_adjusted"] < 0.05)
            print(f"  {key}: {len(pairs)} pairs, {n_sig} significant after FDR")
            pairwise_cat_results[key] = pairs

    # ══════════════════════════════════════════════════════════════
    #  SAVE OUTPUTS
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'=' * 80}")
    print(f"SAVING OUTPUTS (total elapsed: {elapsed:.1f}s)")
    print("=" * 80)

    # ── Master JSON ──
    master = {
        "meta": {
            "n_permutations": N_PERMUTATIONS,
            "n_bootstrap": N_BOOTSTRAP,
            "restricted_layer_start": RESTRICTED_LAYER_START,
            "n_layers": N_LAYERS,
            "hidden_dim": HIDDEN_DIM,
            "seed": 42,
            "elapsed_seconds": round(elapsed, 1),
        },
        "dimensions": {str(k): v for k, v in all_dim_results.items()},
        "categories": category_results,
        "pairwise_dimensions": pairwise_dim_results,
        "pairwise_categories": pairwise_cat_results,
    }

    json_path = os.path.join(SUMMARIES_DIR, "alignment_stats.json")
    with open(json_path, "w") as f:
        json.dump(master, f, indent=2)
    print(f"  JSON: {json_path}")

    # ── Dimension table CSV ──
    csv_path = os.path.join(SUMMARIES_DIR, "dimension_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dim_id", "dim_name", "label", "category",
            "probe_type", "layer_range",
            "observed_cosine", "observed_projection",
            "p_value", "p_adjusted", "ci_lo", "ci_hi",
            "null_mean", "null_std", "sig", "sig_fdr",
        ])
        for dim_id in sorted(all_dim_results.keys()):
            dr = all_dim_results[dim_id]
            for pt in PROBE_TYPES:
                for lr_label in LAYER_RANGES:
                    key = f"{pt}_{lr_label}"
                    r = dr.get(key)
                    if r is None:
                        continue
                    writer.writerow([
                        dim_id, dr["dim_name"], dr["label"], dr["category"],
                        pt, lr_label,
                        f"{r['observed_cosine']:.6f}",
                        f"{r['observed_projection']:.6f}",
                        f"{r['p_value']:.6f}",
                        f"{r.get('p_adjusted', r['p_value']):.6f}",
                        f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                        f"{r['null_mean']:.6f}", f"{r['null_std']:.6f}",
                        r["sig"],
                        r.get("sig_fdr", r["sig"]),
                    ])
    print(f"  CSV:  {csv_path}")

    # ── Category table CSV ──
    csv_path = os.path.join(SUMMARIES_DIR, "category_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "category", "probe_type", "layer_range",
            "mean", "ci_lo", "ci_hi", "n_dims",
        ])
        for key, cats in category_results.items():
            parts = key.rsplit("_", 1)
            # Parse probe_type and layer_range from key
            for pt in PROBE_TYPES:
                if key.startswith(pt):
                    probe = pt
                    lr = key[len(pt) + 1:]
                    break
            for cn in ["Mental", "Physical", "Pragmatic", "SysPrompt",
                        "Human vs AI (General)", "Bio Ctrl", "Shapes"]:
                if cn in cats:
                    r = cats[cn]
                    writer.writerow([
                        cn, probe, lr,
                        f"{r['mean']:.6f}",
                        f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                        r["n_dims"],
                    ])
    print(f"  CSV:  {csv_path}")

    # ── Pairwise dimensions CSV ──
    csv_path = os.path.join(SUMMARIES_DIR, "pairwise_dimensions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "probe_type", "layer_range",
            "dim_a", "dim_b", "name_a", "name_b",
            "diff", "p_value", "p_adjusted", "ci_lo", "ci_hi",
            "sig_raw", "sig_fdr",
        ])
        for key, pairs in pairwise_dim_results.items():
            for pt in PROBE_TYPES:
                if key.startswith(pt):
                    probe = pt
                    lr = key[len(pt) + 1:]
                    break
            for r in pairs:
                writer.writerow([
                    probe, lr,
                    r["dim_a"], r["dim_b"], r["name_a"], r["name_b"],
                    f"{r['diff']:.6f}",
                    f"{r['p_value']:.6f}", f"{r['p_adjusted']:.6f}",
                    f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                    r["sig_raw"], r["sig_fdr"],
                ])
    print(f"  CSV:  {csv_path}")

    # ── Pairwise categories CSV ──
    csv_path = os.path.join(SUMMARIES_DIR, "pairwise_categories.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "probe_type", "layer_range",
            "cat_a", "cat_b",
            "diff", "p_value", "p_adjusted", "ci_lo", "ci_hi",
            "sig_raw", "sig_fdr",
        ])
        for key, pairs in pairwise_cat_results.items():
            for pt in PROBE_TYPES:
                if key.startswith(pt):
                    probe = pt
                    lr = key[len(pt) + 1:]
                    break
            for r in pairs:
                writer.writerow([
                    probe, lr,
                    r["cat_a"], r["cat_b"],
                    f"{r['diff']:.6f}",
                    f"{r['p_value']:.6f}", f"{r['p_adjusted']:.6f}",
                    f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}",
                    r["sig_raw"], r["sig_fdr"],
                ])
    print(f"  CSV:  {csv_path}")

    # ── Methods writeup ──
    methods_path = os.path.join(METHODS_DIR, "statistical_methods.md")
    with open(methods_path, "w") as f:
        f.write("# Statistical Methods: Concept-Probe Alignment Analysis\n\n")
        f.write("*Auto-generated by `2d_concept_probe_stats.py`*\n\n")

        f.write("## Overview\n\n")
        f.write("We test whether concept directions (mean human - mean AI activations "
                "in Experiment 3) align with conversational partner-identity probes "
                "trained in Experiment 2. Two probe types are tested: the **control "
                "probe** (an active in-context representation of partner identity used "
                "while generating the first token of the LLM response) and the "
                "**reading probe** (a metacognitive reflection of partner identity).\n\n")

        f.write("## Test Statistic\n\n")
        f.write("For each concept dimension, we project per-prompt activations onto "
                "the unit-normalized probe direction at each layer, yielding a "
                "(n_prompts, n_layers) projection matrix. The test statistic is the "
                "mean across layers of (mean_human_projection - mean_AI_projection). "
                "This reduces to a simple difference of means on per-prompt scores "
                "averaged over layers.\n\n")

        f.write("## Per-Dimension Significance\n\n")
        f.write(f"- **Permutation test** (n = {N_PERMUTATIONS:,}): Randomly shuffle "
                "human/AI prompt labels, recompute statistic. P-value = fraction of "
                "permutations with statistic >= observed.\n")
        f.write(f"- **Bootstrap 95% CI** (n = {N_BOOTSTRAP:,}): Resample prompts "
                "with replacement within each group (human, AI), recompute statistic.\n")
        f.write(f"- Two layer ranges: all layers (0-40) and restricted (layers "
                f"{RESTRICTED_LAYER_START}-40) to exclude early-layer prompt-format "
                "confounds.\n\n")

        f.write("## Per-Layer Tests\n\n")
        f.write(f"Same permutation approach applied independently at each of "
                f"{N_LAYERS} layers, enabling layerwise significance heatmaps.\n\n")

        f.write("## Pairwise Dimension Comparisons\n\n")
        f.write(f"For each pair of dimensions ({len(active_dim_ids)} dims, "
                f"{len(active_dim_ids) * (len(active_dim_ids) - 1) // 2} pairs), "
                "we independently bootstrap alignment within each dimension and "
                "test whether the difference is different from zero (two-sided). "
                "P-values are corrected for multiple comparisons using "
                "Benjamini-Hochberg FDR (q = 0.05).\n\n")

        f.write("## Category-Level Alignment\n\n")
        f.write("Categories group dimensions by theoretical type (Mental, Physical, "
                "Pragmatic, etc.). Category alignment is computed by averaging "
                "dimension-level bootstrap distributions, providing CIs that properly "
                "account for prompt-level variance rather than treating per-dimension "
                "means as independent observations.\n\n")

        f.write("## Pairwise Category Comparisons\n\n")
        f.write("Bootstrap differences between category-level alignment distributions, "
                "with FDR correction across all category pairs.\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Permutations: {N_PERMUTATIONS:,}\n")
        f.write(f"- Bootstrap resamples: {N_BOOTSTRAP:,}\n")
        f.write(f"- Random seed: 42\n")
        f.write(f"- Restricted layer cutoff: {RESTRICTED_LAYER_START}\n")
        f.write(f"- Model: LLaMA-2-13B-Chat ({N_LAYERS} layers, "
                f"hidden dim {HIDDEN_DIM})\n")
        f.write(f"- Prompts per dimension: 80 (40 human, 40 AI)\n")
    print(f"  Methods: {methods_path}")

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    for pt in PROBE_TYPES:
        pt_label = pt.replace("_probe", "").title()
        for lr_label in LAYER_RANGES:
            key = f"{pt}_{lr_label}"
            n_sig = sum(1 for d in all_dim_results.values()
                        if d.get(key, {}).get("p_value", 1) < 0.05)
            n_total = sum(1 for d in all_dim_results.values() if key in d)
            print(f"  {pt_label}/{lr_label}: {n_sig}/{n_total} dims significant (p < .05)")

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"All outputs in: {OUT_ROOT}/")
    print("Done.")


if __name__ == "__main__":
    main()
