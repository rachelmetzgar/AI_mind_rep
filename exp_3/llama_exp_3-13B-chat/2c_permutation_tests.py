#!/usr/bin/env python3
"""
Permutation test for concept-probe alignment significance.

Instead of treating 41 layers as independent observations (which inflates
significance due to autocorrelation), this tests whether the observed
alignment exceeds what would be expected if the human/AI prompt labels
were randomly shuffled.

Test statistic: mean across layers of the projection of the concept
mean-diff vector onto the (unit-normalized) probe direction. Equivalently,
for each prompt compute its projection onto the probe direction at each
layer, average across layers to get a per-prompt score, then the statistic
is mean(human_scores) - mean(AI_scores).

This precomputation reduces each permutation to comparing means of 40
precomputed scalars, enabling 10,000 permutations per test in milliseconds.

For each concept dimension x probe type:
  1. Precompute per-prompt projections onto probe direction: (80, 41)
  2. Average over layers -> per-prompt score: (80,)
  3. Observed statistic = mean(human scores) - mean(AI scores)
  4. Permutation null: randomly reassign 40/40 labels, recompute stat
  5. P-value = fraction of permutations with stat >= observed
  6. Bootstrap 95% CIs by resampling prompts within each group

Inputs:
    data/concept_activations/contrasts/{dim}/concept_activations.npz
    Exp 2 probe weights (.pth files)

Outputs:
    data/concept_probes/summary_stats/permutation_results.json
    data/concept_probes/summary_stats/permutation_summary.txt

Usage:
    python 2c_permutation_tests.py

Env: llama2_env (needs numpy, torch; no GPU required)
Rachel C. Metzgar, Feb 2026
"""

import os
import json
import time
import numpy as np
import torch

# ========================== CONFIG ========================== #

BASE = os.path.dirname(os.path.abspath(__file__))
ACT_ROOT = os.path.join(BASE, "data", "concept_activations", "contrasts")
STATS_DIR = os.path.join(BASE, "data", "concept_probes", "summary_stats")

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
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 16, 17],
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Baseline":  [0],
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
    15: "15_shapes", 16: "16_mind", 17: "17_attention",
    18: "18_sysprompt",
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
        state = torch.load(path, map_location="cpu")
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


# ========================== CORE (OPTIMIZED) ========================== #

def precompute_projections(activations, probe_weights):
    """
    Precompute per-prompt projections onto the unit-normalized probe direction.

    proj[i, l] = dot(activations[i, l, :], probe_unit[l, :])

    Returns: (n_prompts, n_layers) array
    """
    norms = np.linalg.norm(probe_weights, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    probe_unit = probe_weights / norms
    return np.einsum('ild,ld->il', activations, probe_unit)


def compute_cosine_alignment(activations, labels, probe_weights, layer_slice=None):
    """
    Compute the cosine-based mean alignment (for reporting alongside
    the projection-based permutation test).
    """
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
    return float(np.mean(cosines))


def permutation_test(projections, labels, layer_slice=None, n_perm=N_PERMUTATIONS):
    """
    Fast permutation test using precomputed projections.

    Test statistic: mean over layers of (mean_human_proj - mean_AI_proj).
    Reduces to: mean(prompt_scores[human]) - mean(prompt_scores[ai])
    where prompt_scores[i] = mean_l(projections[i, l]).

    Each permutation is a single dot product of a binary indicator vector
    with the (80,) prompt_scores array.

    Returns: (observed, p_value, null_distribution)
    """
    n_prompts = len(labels)
    n_human = int(np.sum(labels == 1))
    n_ai = n_prompts - n_human
    human_mask = labels == 1

    # Average projections across relevant layers -> per-prompt scores
    if layer_slice is not None:
        prompt_scores = projections[:, layer_slice].mean(axis=1)
    else:
        prompt_scores = projections.mean(axis=1)  # (80,)

    # Observed statistic
    observed = float(prompt_scores[human_mask].mean() - prompt_scores[~human_mask].mean())

    # Generate permutation selections: binary indicator matrix (n_perm, n_prompts)
    indicators = np.zeros((n_perm, n_prompts))
    for i in range(n_perm):
        sel = RNG.choice(n_prompts, size=n_human, replace=False)
        indicators[i, sel] = 1.0

    # Null statistics via matrix multiply
    # stat = sum_selected/n_human - (total - sum_selected)/n_ai
    total = prompt_scores.sum()
    sel_sums = indicators @ prompt_scores  # (n_perm,)
    null_stats = sel_sums / n_human - (total - sel_sums) / n_ai

    # One-sided p-value: is observed >= permuted?
    p_value = float(np.mean(null_stats >= observed))

    return observed, p_value, null_stats


def bootstrap_ci(projections, labels, layer_slice=None, n_boot=N_BOOTSTRAP):
    """
    Bootstrap CI by resampling prompts within each group.

    Returns: (ci_lo, ci_hi, boot_distribution)
    """
    if layer_slice is not None:
        prompt_scores = projections[:, layer_slice].mean(axis=1)
    else:
        prompt_scores = projections.mean(axis=1)

    human_mask = labels == 1
    human_scores = prompt_scores[human_mask]
    ai_scores = prompt_scores[~human_mask]

    # Vectorized bootstrap: (n_boot, n_per_group)
    boot_human = RNG.choice(human_scores, size=(n_boot, len(human_scores)), replace=True)
    boot_ai = RNG.choice(ai_scores, size=(n_boot, len(ai_scores)), replace=True)
    boot_stats = boot_human.mean(axis=1) - boot_ai.mean(axis=1)

    ci_lo = float(np.percentile(boot_stats, 2.5))
    ci_hi = float(np.percentile(boot_stats, 97.5))
    return ci_lo, ci_hi, boot_stats


# ========================== MAIN ========================== #

def main():
    os.makedirs(STATS_DIR, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("PERMUTATION TEST FOR CONCEPT-PROBE ALIGNMENT")
    print(f"  {N_PERMUTATIONS} permutations, {N_BOOTSTRAP} bootstrap resamples")
    print(f"  Restricted layer analysis: layers {RESTRICTED_LAYER_START}+")
    print(f"  Test statistic: mean projection (dot with unit-norm probe)")
    print("=" * 80)

    # Load probe weights
    print("\nLoading Exp 2 probe weights...")
    probe_weights = {}
    for probe_type in ["control_probe", "reading_probe"]:
        probe_weights[probe_type] = load_exp2_probe_weights(probe_type)
    print("  Done.")

    # Discover available dimensions
    available_dims = {}
    for dim_id, dim_name in DIM_NAMES.items():
        act_path = os.path.join(ACT_ROOT, dim_name, "concept_activations.npz")
        if os.path.isfile(act_path):
            available_dims[dim_id] = dim_name
        else:
            print(f"  [MISS] Dim {dim_id} ({dim_name}): no activation file")
    print(f"\nFound {len(available_dims)}/{len(DIM_NAMES)} dimensions with activations.")

    # Layer slices
    layer_slice_full = None
    layer_slice_restricted = slice(RESTRICTED_LAYER_START, None)

    all_results = {}

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

        dim_results = {
            "dim_id": dim_id, "dim_name": dim_name,
            "category": cat, "n_human": n_human, "n_ai": n_ai
        }

        for probe_type in ["control_probe", "reading_probe"]:
            pt_short = probe_type.replace("_probe", "")
            pw = probe_weights[probe_type]

            # Precompute projections for this dim x probe: (80, 41)
            projections = precompute_projections(activations, pw)

            for layer_label, ls in [("all_layers", layer_slice_full),
                                     ("layers_6plus", layer_slice_restricted)]:
                key = f"{probe_type}_{layer_label}"
                t0 = time.time()

                # Cosine alignment (for reporting / comparison with old results)
                cosine_obs = compute_cosine_alignment(activations, labels, pw, ls)

                # Permutation test (projection-based, fast)
                proj_obs, p_val, null_dist = permutation_test(
                    projections, labels, ls, N_PERMUTATIONS
                )

                # Bootstrap CI (projection-based, fast)
                ci_lo, ci_hi, _ = bootstrap_ci(projections, labels, ls, N_BOOTSTRAP)

                dt = time.time() - t0
                sig = ("***" if p_val < 0.001 else "**" if p_val < 0.01
                       else "*" if p_val < 0.05 else "n.s.")
                null_mean = float(np.mean(null_dist))
                null_std = float(np.std(null_dist))

                print(f"  {pt_short:>7}/{layer_label:<12} "
                      f"cos={cosine_obs:>7.4f}  proj={proj_obs:>7.4f}  "
                      f"p={p_val:.4f} {sig:>4}  "
                      f"CI=[{ci_lo:.4f},{ci_hi:.4f}]  "
                      f"null={null_mean:.4f}\u00b1{null_std:.4f}  "
                      f"({dt:.1f}s)")

                dim_results[key] = {
                    "observed_cosine": cosine_obs,
                    "observed_projection": proj_obs,
                    "p_value": p_val,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "null_mean": null_mean,
                    "null_std": null_std,
                    "sig": sig,
                }

        all_results[dim_id] = dim_results

    # ── Save JSON ──
    json_path = os.path.join(STATS_DIR, "permutation_results.json")
    with open(json_path, "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\n\nJSON saved: {json_path}")

    # ── Summary tables ──
    print(f"\n{'=' * 120}")
    print("SUMMARY: PERMUTATION TEST RESULTS")
    print(f"{'=' * 120}")

    for probe_type in ["control_probe", "reading_probe"]:
        pt_label = probe_type.replace("_probe", "").title()

        for layer_label in ["all_layers", "layers_6plus"]:
            ll_label = ("All Layers" if layer_label == "all_layers"
                        else "Layers 6+")
            key = f"{probe_type}_{layer_label}"
            print(f"\n--- {pt_label} Probe / {ll_label} ---")
            mu, sigma = "\u03bc", "\u03c3"
            print(f"{'Dim':>3} {'Name':<20} {'Cat':<10} \u2502 "
                  f"{'Cosine':>7} {'Proj':>7} {'p':>8} {'Sig':>4} \u2502 "
                  f"{'95% CI':>18} \u2502 "
                  f"{'Null '+mu:>7} {'Null '+sigma:>7}")
            print("\u2500" * 105)

            sorted_dims = sorted(
                all_results.keys(),
                key=lambda d: all_results[d].get(key, {}).get(
                    "observed_cosine", 0),
                reverse=True
            )
            for dim_id in sorted_dims:
                r = all_results[dim_id].get(key)
                if r is None:
                    continue
                print(f"{dim_id:>3} {all_results[dim_id]['dim_name']:<20} "
                      f"{all_results[dim_id]['category']:<10} \u2502 "
                      f"{r['observed_cosine']:>7.4f} "
                      f"{r['observed_projection']:>7.4f} "
                      f"{r['p_value']:>8.4f} {r['sig']:>4} \u2502 "
                      f"[{r['ci_lo']:>7.4f},{r['ci_hi']:>7.4f}] \u2502 "
                      f"{r['null_mean']:>7.4f} {r['null_std']:>7.4f}")

    # ── Comparison with old t-test results ──
    old_json = os.path.join(STATS_DIR, "statistical_results.json")
    if os.path.isfile(old_json):
        with open(old_json) as f:
            old_data = json.load(f)

        print(f"\n{'=' * 120}")
        print("COMPARISON: T-TEST (old) vs PERMUTATION (new)")
        print(f"{'=' * 120}")

        for probe_type in ["control_probe", "reading_probe"]:
            pt_label = probe_type.replace("_probe", "").title()
            print(f"\n--- {pt_label} Probe (All Layers) ---")
            print(f"{'Dim':>3} {'Name':<20} {'Cat':<10} \u2502 "
                  f"{'t-test p':>10} {'Perm p':>10} \u2502 {'Match':>6}")
            print("\u2500" * 75)

            old_vs = old_data.get("vs_chance", {}).get(probe_type, {})
            for dim_id in sorted(all_results.keys()):
                key = f"{probe_type}_all_layers"
                new_r = all_results[dim_id].get(key)
                old_r = old_vs.get(str(dim_id))
                if new_r is None or old_r is None:
                    continue

                old_p = old_r["p_value"]
                new_p = new_r["p_value"]
                agree = "YES" if (old_p < 0.05) == (new_p < 0.05) else "NO"

                print(f"{dim_id:>3} {all_results[dim_id]['dim_name']:<20} "
                      f"{all_results[dim_id]['category']:<10} \u2502 "
                      f"{old_p:>10.6f} {new_p:>10.4f} \u2502 {agree:>6}")

    # ── Save summary text file ──
    summary_path = os.path.join(STATS_DIR, "permutation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("PERMUTATION TEST SUMMARY\n")
        f.write(f"{N_PERMUTATIONS} permutations, {N_BOOTSTRAP} bootstrap resamples\n")
        f.write("Test statistic: mean projection "
                "(dot product with unit-norm probe)\n")
        f.write(f"Restricted layer cutoff: {RESTRICTED_LAYER_START}\n\n")

        for probe_type in ["control_probe", "reading_probe"]:
            pt_label = probe_type.replace("_probe", "").title()
            for layer_label in ["all_layers", "layers_6plus"]:
                ll_label = ("All Layers" if layer_label == "all_layers"
                            else "Layers 6+")
                key = f"{probe_type}_{layer_label}"
                f.write(f"\n{pt_label} Probe / {ll_label}\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Dim':>3} {'Name':<20} {'Cat':<10} "
                        f"{'Cos':>7} {'Proj':>7} {'p':>8} {'Sig':>4} "
                        f"{'CI_lo':>7} {'CI_hi':>7}\n")

                sorted_dims = sorted(
                    all_results.keys(),
                    key=lambda d: all_results[d].get(key, {}).get(
                        "observed_cosine", 0),
                    reverse=True
                )
                for dim_id in sorted_dims:
                    r = all_results[dim_id].get(key)
                    if r is None:
                        continue
                    f.write(
                        f"{dim_id:>3} "
                        f"{all_results[dim_id]['dim_name']:<20} "
                        f"{all_results[dim_id]['category']:<10} "
                        f"{r['observed_cosine']:>7.4f} "
                        f"{r['observed_projection']:>7.4f} "
                        f"{r['p_value']:>8.4f} {r['sig']:>4} "
                        f"{r['ci_lo']:>7.4f} {r['ci_hi']:>7.4f}\n"
                    )

        # Significance counts
        f.write("\n\nSIGNIFICANCE COUNTS\n")
        f.write("=" * 60 + "\n")
        for probe_type in ["control_probe", "reading_probe"]:
            pt_label = probe_type.replace("_probe", "").title()
            for layer_label in ["all_layers", "layers_6plus"]:
                ll_label = ("All Layers" if layer_label == "all_layers"
                            else "Layers 6+")
                key = f"{probe_type}_{layer_label}"
                n_sig = sum(1 for d in all_results.values()
                            if d.get(key, {}).get("p_value", 1) < 0.05)
                n_total = sum(1 for d in all_results.values()
                              if key in d)
                f.write(f"  {pt_label} / {ll_label}: "
                        f"{n_sig}/{n_total} significant at p < .05\n")

    elapsed = time.time() - t_start
    print(f"\nSummary saved: {summary_path}")
    print(f"Total time: {elapsed:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
