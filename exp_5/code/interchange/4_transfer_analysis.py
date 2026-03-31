#!/usr/bin/env python3
"""
Experiment 5, Step 17: Transfer Analysis

Statistical analysis of verb swap and subject swap intervention results.

Analyses:
  2a. Transfer matrix: 6x6 mean swap_success at each layer
  2b. Key contrasts with 10K permutation tests
  2c. Block structure (within-type vs cross-type)
  2d. Verb similarity control regression
  3.  Subject swap cross-type vs within-type

Output:
    results/{model}/interchange/data/
        transfer_matrix.npz
        block_analysis.json

Usage:
    python code/interchange/4_transfer_analysis.py --model llama2_13b_chat

Env: llama2_env (CPU only)
Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    set_model, add_model_argument,
    data_dir, ensure_dir, N_SENTENCES, N_ITEMS, N_CONDITIONS, CONDITION_LABELS,
)


# ── Constants ────────────────────────────────────────────────────────────────

N_PERMUTATIONS = 10_000


# ── Helpers ──────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    """Cosine similarity between two numpy vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)


def build_transfer_matrix(df, layers):
    """Build 6x6xL transfer matrix from verb swap results.

    Args:
        df: DataFrame with columns source_cond, target_cond, layer_idx, swap_success
        layers: sorted list of layer indices

    Returns:
        (6, 6, L) array where entry [s, t, l] = mean swap_success for
        source_cond=s, target_cond=t at layer l
    """
    n_conds = len(CONDITION_LABELS)
    n_layers = len(layers)
    matrix = np.full((n_conds, n_layers, n_conds), np.nan)

    for li, layer in enumerate(layers):
        layer_df = df[df["layer_idx"] == layer]
        for si, scond in enumerate(CONDITION_LABELS):
            for ti, tcond in enumerate(CONDITION_LABELS):
                mask = (layer_df["source_cond"] == scond) & \
                       (layer_df["target_cond"] == tcond)
                vals = layer_df.loc[mask, "swap_success"].values
                if len(vals) > 0:
                    matrix[si, li, ti] = np.mean(vals)

    # Reshape to (6, 6, L): source x target x layer
    # Currently (source, layer, target) — transpose last two axes
    matrix = np.transpose(matrix, (0, 2, 1))  # (source, target, layer)
    return matrix


def permutation_test(group_a, group_b, n_perm, rng, two_sided=True):
    """Permutation test for difference in means.

    Args:
        group_a, group_b: 1D arrays of values
        n_perm: number of permutations
        rng: numpy random generator
        two_sided: if True, test |diff|; if False, test diff > 0

    Returns:
        dict with observed_diff, p_value, n_perm
    """
    observed = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    na = len(group_a)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:na]) - np.mean(combined[na:])
        if two_sided:
            if abs(perm_diff) >= abs(observed):
                count += 1
        else:
            if perm_diff >= observed:
                count += 1

    return {
        "observed_diff": float(observed),
        "p_value": float((count + 1) / (n_perm + 1)),
        "n_perm": n_perm,
        "n_a": len(group_a),
        "n_b": len(group_b),
        "mean_a": float(np.mean(group_a)),
        "mean_b": float(np.mean(group_b)),
    }


# ── Step 2b: Key Contrasts ──────────────────────────────────────────────────

def run_key_contrasts(df, layers, rng):
    """Run 5 key contrasts at each layer with permutation tests.

    Returns:
        dict mapping contrast_name -> {layer_idx: permutation_test_result}
    """
    contrasts = {}

    for layer in layers:
        ldf = df[df["layer_idx"] == layer]

        # Helper to get swap_success values for a source→target condition pair
        def get_vals(scond, tcond):
            mask = (ldf["source_cond"] == scond) & (ldf["target_cond"] == tcond)
            return ldf.loc[mask, "swap_success"].values

        # Test 1: within_mental (C1→C1) vs cross_to_action (C1→C4)
        a = get_vals("mental_state", "mental_state")
        b = get_vals("mental_state", "action")
        key = "test1_within_mental_vs_cross_action"
        if key not in contrasts:
            contrasts[key] = {}
        if len(a) > 0 and len(b) > 0:
            contrasts[key][layer] = permutation_test(a, b, N_PERMUTATIONS, rng)

        # Test 2: within_action (C4→C4) vs cross_to_mental (C4→C1)
        a = get_vals("action", "action")
        b = get_vals("action", "mental_state")
        key = "test2_within_action_vs_cross_mental"
        if key not in contrasts:
            contrasts[key] = {}
        if len(a) > 0 and len(b) > 0:
            contrasts[key][layer] = permutation_test(a, b, N_PERMUTATIONS, rng)

        # Test 3: C1→C1 vs C1→C2 (does removing subject hurt verb transfer?)
        a = get_vals("mental_state", "mental_state")
        b = get_vals("mental_state", "dis_mental")
        key = "test3_c1c1_vs_c1c2"
        if key not in contrasts:
            contrasts[key] = {}
        if len(a) > 0 and len(b) > 0:
            contrasts[key][layer] = permutation_test(a, b, N_PERMUTATIONS, rng)

        # Test 4: C1→C2 vs C1→C4 (verb type vs subject presence)
        a = get_vals("mental_state", "dis_mental")
        b = get_vals("mental_state", "action")
        key = "test4_c1c2_vs_c1c4"
        if key not in contrasts:
            contrasts[key] = {}
        if len(a) > 0 and len(b) > 0:
            contrasts[key][layer] = permutation_test(a, b, N_PERMUTATIONS, rng)

        # Test 5: Attribution interaction
        # (C1→C1 - C1→C2) vs (C4→C4 - C4→C5)
        c1c1 = get_vals("mental_state", "mental_state")
        c1c2 = get_vals("mental_state", "dis_mental")
        c4c4 = get_vals("action", "action")
        c4c5 = get_vals("action", "dis_action")
        key = "test5_attribution_interaction"
        if key not in contrasts:
            contrasts[key] = {}
        if all(len(x) > 0 for x in [c1c1, c1c2, c4c4, c4c5]):
            # Use item-level differences for paired test
            # Compute per-item means, then difference
            n_min = min(len(c1c1), len(c1c2), len(c4c4), len(c4c5))
            diff_mental = c1c1[:n_min] - c1c2[:n_min]
            diff_action = c4c4[:n_min] - c4c5[:n_min]
            contrasts[key][layer] = permutation_test(
                diff_mental, diff_action, N_PERMUTATIONS, rng
            )

    return contrasts


# ── Step 2c: Block Structure ─────────────────────────────────────────────────

def run_block_analysis(df, layers, rng):
    """Within-type vs cross-type mean swap_success at each layer.

    Within-type: mental conditions (C1, C2, C3) transferring among themselves,
                 and action conditions (C4, C5, C6) among themselves.
    Cross-type:  mental→action and action→mental transfers.

    Returns:
        dict mapping layer -> permutation_test_result
    """
    mental_conds = {"mental_state", "dis_mental", "scr_mental"}
    action_conds = {"action", "dis_action", "scr_action"}

    results = {}
    for layer in layers:
        ldf = df[df["layer_idx"] == layer]

        # Within-type: both source and target in same type
        within_mask = (
            (ldf["source_cond"].isin(mental_conds) & ldf["target_cond"].isin(mental_conds)) |
            (ldf["source_cond"].isin(action_conds) & ldf["target_cond"].isin(action_conds))
        )
        # Cross-type: source and target in different types
        cross_mask = (
            (ldf["source_cond"].isin(mental_conds) & ldf["target_cond"].isin(action_conds)) |
            (ldf["source_cond"].isin(action_conds) & ldf["target_cond"].isin(mental_conds))
        )

        within_vals = ldf.loc[within_mask, "swap_success"].values
        cross_vals = ldf.loc[cross_mask, "swap_success"].values

        if len(within_vals) > 0 and len(cross_vals) > 0:
            results[layer] = permutation_test(
                within_vals, cross_vals, N_PERMUTATIONS, rng
            )

    return results


# ── Step 2d: Verb Similarity Control Regression ─────────────────────────────

def run_verb_regression(df, layers):
    """Regress swap_success on verb similarity + condition features.

    Loads verb embeddings from layer 0 of activations_multipos.npz.

    Returns:
        dict mapping layer -> regression results
    """
    # Load verb embeddings (layer 0 = embedding layer)
    multipos_path = data_dir("activations") / "activations_multipos.npz"
    if not multipos_path.exists():
        print(f"  WARNING: {multipos_path} not found, skipping verb regression")
        return {}

    multipos = np.load(multipos_path)["activations"]  # (336, 3, 41, 5120)
    verb_embeddings = multipos[:, 0, 0, :].astype(np.float32)  # (336, 5120) — verb, embedding layer
    del multipos

    print(f"  Loaded verb embeddings: {verb_embeddings.shape}")

    # Build condition type features
    mental_conds = {"mental_state", "dis_mental", "scr_mental"}

    results = {}
    for layer in layers:
        ldf = df[df["layer_idx"] == layer].copy()
        n = len(ldf)
        if n == 0:
            continue

        # Compute verb similarity for each pair
        source_idxs = ldf["source_idx"].values
        target_idxs = ldf["target_idx"].values

        verb_sims = np.array([
            cosine_sim(verb_embeddings[s], verb_embeddings[t])
            for s, t in zip(source_idxs, target_idxs)
        ])

        # Same condition indicator
        same_cond = (ldf["source_cond"].values == ldf["target_cond"].values).astype(float)

        # Same verb type indicator (both mental or both action)
        source_mental = np.array([c in mental_conds for c in ldf["source_cond"].values])
        target_mental = np.array([c in mental_conds for c in ldf["target_cond"].values])
        same_verb_type = (source_mental == target_mental).astype(float)

        # OLS regression: swap_success ~ verb_sim + same_condition + same_verb_type
        y = ldf["swap_success"].values
        X = np.column_stack([
            np.ones(n),
            verb_sims,
            same_cond,
            same_verb_type,
        ])

        # Solve via normal equations
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residuals = y - y_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Standard errors
            dof = n - X.shape[1]
            mse = ss_res / dof if dof > 0 else 0.0
            try:
                var_beta = mse * np.diag(np.linalg.inv(X.T @ X))
                se_beta = np.sqrt(np.maximum(var_beta, 0))
                t_stats = beta / (se_beta + 1e-12)
            except np.linalg.LinAlgError:
                se_beta = np.full_like(beta, np.nan)
                t_stats = np.full_like(beta, np.nan)

            results[layer] = {
                "intercept": float(beta[0]),
                "verb_sim_coef": float(beta[1]),
                "same_cond_coef": float(beta[2]),
                "same_verb_type_coef": float(beta[3]),
                "intercept_se": float(se_beta[0]),
                "verb_sim_se": float(se_beta[1]),
                "same_cond_se": float(se_beta[2]),
                "same_verb_type_se": float(se_beta[3]),
                "intercept_t": float(t_stats[0]),
                "verb_sim_t": float(t_stats[1]),
                "same_cond_t": float(t_stats[2]),
                "same_verb_type_t": float(t_stats[3]),
                "r_squared": float(r_squared),
                "n": n,
                "dof": int(dof),
            }
        except np.linalg.LinAlgError:
            print(f"  WARNING: regression failed at layer {layer}")
            continue

    return results


# ── Step 3: Subject Swap Analysis ────────────────────────────────────────────

def run_subject_analysis(subject_df, layers, rng):
    """Cross-type vs within-type subject swap effects at each layer.

    Returns:
        dict mapping layer -> permutation_test_result
    """
    results = {}
    for layer in layers:
        ldf = subject_df[subject_df["layer_idx"] == layer]

        cross_mask = ldf["swap_type"].isin(["cross_c1_to_c4", "cross_c4_to_c1"])
        within_mask = ldf["swap_type"] == "within_c1"

        cross_vals = ldf.loc[cross_mask, "effect"].values
        within_vals = ldf.loc[within_mask, "effect"].values

        if len(cross_vals) > 0 and len(within_vals) > 0:
            results[layer] = permutation_test(
                cross_vals, within_vals, N_PERMUTATIONS, rng
            )

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer analysis of interchange intervention results"
    )
    add_model_argument(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def make_json_serializable(obj):
    """Recursively convert numpy types and dict keys for JSON serialization."""
    if isinstance(obj, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    args = parse_args()
    set_model(args.model)
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(data_dir("interchange"))

    # ── Load verb swap results ───────────────────────────────────────────
    verb_csv = out_dir / "verb_swap_results.csv"
    if not verb_csv.exists():
        print(f"ERROR: {verb_csv} not found. Run 15_verb_swap_interventions.py first.")
        sys.exit(1)

    print(f"Loading verb swap results from {verb_csv}")
    verb_df = pd.read_csv(verb_csv)
    print(f"  {len(verb_df)} rows, layers: {sorted(verb_df['layer_idx'].unique())}")

    layers = sorted(verb_df["layer_idx"].unique())

    # ── Step 2a: Transfer Matrix ─────────────────────────────────────────
    print("\n=== Step 2a: Transfer Matrix ===")
    t0 = time.time()
    transfer_matrix = build_transfer_matrix(verb_df, layers)
    print(f"  Shape: {transfer_matrix.shape} (source x target x layer)")
    print(f"  Range: [{np.nanmin(transfer_matrix):.4f}, {np.nanmax(transfer_matrix):.4f}]")

    # Save transfer matrix
    np.savez_compressed(
        out_dir / "transfer_matrix.npz",
        transfer_matrix=transfer_matrix,
        layers=np.array(layers),
        condition_labels=CONDITION_LABELS,
    )
    print(f"  Saved to {out_dir / 'transfer_matrix.npz'}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Step 2b: Key Contrasts ───────────────────────────────────────────
    print("\n=== Step 2b: Key Contrasts ===")
    t0 = time.time()
    contrasts = run_key_contrasts(verb_df, layers, rng)
    for name, layer_results in contrasts.items():
        sig_layers = [l for l, r in layer_results.items() if r["p_value"] < 0.05]
        print(f"  {name}: sig at layers {sig_layers}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Step 2c: Block Structure ─────────────────────────────────────────
    print("\n=== Step 2c: Block Structure ===")
    t0 = time.time()
    block_results = run_block_analysis(verb_df, layers, rng)
    for layer, result in sorted(block_results.items()):
        print(f"  Layer {layer:2d}: within={result['mean_a']:.4f}, "
              f"cross={result['mean_b']:.4f}, diff={result['observed_diff']:.4f}, "
              f"p={result['p_value']:.4f}")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Step 2d: Verb Similarity Regression ──────────────────────────────
    print("\n=== Step 2d: Verb Similarity Control Regression ===")
    t0 = time.time()
    regression_results = run_verb_regression(verb_df, layers)
    for layer, result in sorted(regression_results.items()):
        print(f"  Layer {layer:2d}: R²={result['r_squared']:.4f}, "
              f"verb_sim={result['verb_sim_coef']:.4f} (t={result['verb_sim_t']:.2f}), "
              f"same_cond={result['same_cond_coef']:.4f} (t={result['same_cond_t']:.2f}), "
              f"same_type={result['same_verb_type_coef']:.4f} (t={result['same_verb_type_t']:.2f})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # ── Step 3: Subject Swap Analysis ────────────────────────────────────
    subject_csv = out_dir / "subject_swap_results.csv"
    subject_results = {}
    if subject_csv.exists():
        print("\n=== Step 3: Subject Swap Analysis ===")
        t0 = time.time()
        subject_df = pd.read_csv(subject_csv)
        print(f"  {len(subject_df)} rows")

        subject_results = run_subject_analysis(subject_df, layers, rng)
        for layer, result in sorted(subject_results.items()):
            print(f"  Layer {layer:2d}: cross={result['mean_a']:.6f}, "
                  f"within={result['mean_b']:.6f}, diff={result['observed_diff']:.6f}, "
                  f"p={result['p_value']:.4f}")
        print(f"  Time: {time.time() - t0:.1f}s")
    else:
        print(f"\n  Subject swap results not found ({subject_csv}), skipping Step 3")

    # ── Save all results ─────────────────────────────────────────────────
    all_results = {
        "layers": layers,
        "condition_labels": CONDITION_LABELS,
        "n_permutations": N_PERMUTATIONS,
        "key_contrasts": contrasts,
        "block_structure": block_results,
        "verb_regression": regression_results,
        "subject_swap": subject_results,
    }

    out_json = out_dir / "block_analysis.json"
    with open(out_json, "w") as f:
        json.dump(make_json_serializable(all_results), f, indent=2)
    print(f"\nSaved analysis to {out_json}")

    print("\nDone.")


if __name__ == "__main__":
    main()
