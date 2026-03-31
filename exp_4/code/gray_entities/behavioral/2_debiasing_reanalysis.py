#!/usr/bin/env python3
"""
Experiment 4: Debiasing Reanalysis of Pairwise Data

Reanalysis of pairwise comparison data using two debiasing strategies:

1. Analytical debiasing: R_debiased = (R_AB + (6 - R_BA)) / 2
   Cancels constant position bias by averaging complementary orders.

2. Log-odds: log(P(A>B) / P(B>A)) where P(A>B) = P(1)+P(2), P(B>A) = P(4)+P(5)
   Collapses to binary signal, may be more robust to distributional skew.

Reads existing pairwise_raw_responses.json from the pairwise replication.
Base model only (requires probability distributions in responses).
No GPU needed -- pure CPU analysis.

Output:
    data_dir("gray_entities", "behavioral", tag)/
        debiasing_reanalysis.npz

Usage:
    python 2_debiasing_reanalysis.py --model llama2_13b_base
    python 2_debiasing_reanalysis.py --model llama2_13b_base --include_self
    python 2_debiasing_reanalysis.py --model llama2_13b_base --both

Env: llama2_env
Rachel C. Metzgar / Mar 2026
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir, get_condition_tag,
)
from utils.utils import varimax_rotation, run_pca_varimax, correlate_with_humans
from entities.gray_entities import (
    GRAY_ET_AL_SCORES, CAPACITY_PROMPTS, CAPACITY_NAMES, ENTITY_NAMES,
)


# ========================== DEBIASING FUNCTIONS ========================== #

def compute_debiased_ratings(responses):
    """
    For each (capacity, entity_pair), average the two orders:
    R_debiased = (R_AB + (6 - R_BA)) / 2

    Returns list of debiased response dicts (one per unordered pair per capacity).
    """
    pair_groups = defaultdict(list)
    for resp in responses:
        key = (resp["capacity"],
               frozenset([resp["entity_a"], resp["entity_b"]]))
        pair_groups[key].append(resp)

    debiased = []
    n_both = 0
    n_single = 0
    deviations = []

    for (cap, pair_set), resps in pair_groups.items():
        entities = sorted(pair_set)
        if len(resps) == 2:
            n_both += 1
            if resps[0]["entity_a"] == entities[0]:
                r_ab = resps[0]["expected_rating"]
                r_ba = resps[1]["expected_rating"]
                probs_ab = resps[0]["probs"]
                probs_ba = resps[1]["probs"]
            else:
                r_ab = resps[1]["expected_rating"]
                r_ba = resps[0]["expected_rating"]
                probs_ab = resps[1]["probs"]
                probs_ba = resps[0]["probs"]

            r_debiased = (r_ab + (6 - r_ba)) / 2
            deviations.append(abs(r_ab + r_ba - 6))

            debiased.append({
                "capacity": cap,
                "entity_a": entities[0],
                "entity_b": entities[1],
                "r_ab": r_ab,
                "r_ba": r_ba,
                "debiased_rating": r_debiased,
                "probs_ab": probs_ab,
                "probs_ba": probs_ba,
            })
        else:
            n_single += 1

    print(f"  Pairs with both orders: {n_both}")
    print(f"  Pairs with single order: {n_single}")
    if deviations:
        print(f"  Mean |R_AB + R_BA - 6|: {np.mean(deviations):.3f}")
        print(f"  Mean debiased rating: "
              f"{np.mean([d['debiased_rating'] for d in debiased]):.3f}")
    return debiased


def compute_logodds_ratings(responses):
    """
    For each response, compute log(P(A>B) / P(B>A)):
        P(A>B) = P(1) + P(2)
        P(B>A) = P(4) + P(5)

    Then debias by averaging complementary orders.
    """
    EPS = 1e-8

    pair_groups = defaultdict(list)
    for resp in responses:
        key = (resp["capacity"],
               frozenset([resp["entity_a"], resp["entity_b"]]))
        pair_groups[key].append(resp)

    logodds_data = []
    for (cap, pair_set), resps in pair_groups.items():
        entities = sorted(pair_set)
        if len(resps) == 2:
            if resps[0]["entity_a"] == entities[0]:
                probs_ab, probs_ba = resps[0]["probs"], resps[1]["probs"]
            else:
                probs_ab, probs_ba = resps[1]["probs"], resps[0]["probs"]

            # Log-odds for AB order: positive = A favored
            p_a_ab = probs_ab[0] + probs_ab[1]
            p_b_ab = probs_ab[3] + probs_ab[4]
            lo_ab = np.log((p_a_ab + EPS) / (p_b_ab + EPS))

            # Log-odds for BA order: flip interpretation
            p_b_ba = probs_ba[0] + probs_ba[1]
            p_a_ba = probs_ba[3] + probs_ba[4]
            lo_ba = np.log((p_a_ba + EPS) / (p_b_ba + EPS))

            # Average debiases position effect
            lo_debiased = (lo_ab + lo_ba) / 2

            logodds_data.append({
                "capacity": cap,
                "entity_a": entities[0],
                "entity_b": entities[1],
                "logodds_ab": lo_ab,
                "logodds_ba": lo_ba,
                "logodds_debiased": lo_debiased,
            })

    lo_vals = [d["logodds_debiased"] for d in logodds_data]
    print(f"  Log-odds range: [{min(lo_vals):.3f}, {max(lo_vals):.3f}]")
    print(f"  Log-odds mean: {np.mean(lo_vals):.3f}")
    print(f"  Log-odds std: {np.std(lo_vals):.3f}")
    return logodds_data


def debiased_to_character_means(debiased_data, entity_keys, capacity_keys):
    """Build character means from debiased pairwise ratings."""
    n_cap = len(capacity_keys)
    n_ent = len(entity_keys)
    ent_to_idx = {k: i for i, k in enumerate(entity_keys)}
    scores = [[[] for _ in range(n_ent)] for _ in range(n_cap)]

    for d in debiased_data:
        cap_idx = capacity_keys.index(d["capacity"])
        a_idx = ent_to_idx[d["entity_a"]]
        b_idx = ent_to_idx[d["entity_b"]]
        r = d["debiased_rating"]
        scores[cap_idx][a_idx].append(3 - r)
        scores[cap_idx][b_idx].append(r - 3)

    means = np.zeros((n_cap, n_ent))
    for c in range(n_cap):
        for e in range(n_ent):
            if scores[c][e]:
                means[c, e] = np.mean(scores[c][e])
    return means


def logodds_to_character_means(logodds_data, entity_keys, capacity_keys):
    """Build character means from debiased log-odds."""
    n_cap = len(capacity_keys)
    n_ent = len(entity_keys)
    ent_to_idx = {k: i for i, k in enumerate(entity_keys)}
    scores = [[[] for _ in range(n_ent)] for _ in range(n_cap)]

    for d in logodds_data:
        cap_idx = capacity_keys.index(d["capacity"])
        a_idx = ent_to_idx[d["entity_a"]]
        b_idx = ent_to_idx[d["entity_b"]]
        lo = d["logodds_debiased"]
        # Positive lo = A favored, so A gets +lo, B gets -lo
        scores[cap_idx][a_idx].append(lo)
        scores[cap_idx][b_idx].append(-lo)

    means = np.zeros((n_cap, n_ent))
    for c in range(n_cap):
        for e in range(n_ent):
            if scores[c][e]:
                means[c, e] = np.mean(scores[c][e])
    return means


# ========================== DISPLAY ========================== #

def print_results(label, pca_results, human_corr, entity_keys, capacity_keys):
    """Print detailed results for one analysis method."""
    print(f"\n  === {label} ===")
    ev = pca_results["eigenvalues"]
    evr = pca_results["explained_var_ratio"]
    print(f"  Eigenvalues: {ev[:5].round(3)}")
    print(f"  Factors: {pca_results['n_factors']}, "
          f"explained: {evr[0]*100:.1f}% + {evr[1]*100:.1f}% "
          f"= {sum(evr[:2])*100:.1f}%")

    print(f"\n  Capacity loadings:")
    rl = pca_results["rotated_loadings"]
    for c_idx, cap in enumerate(capacity_keys):
        _, factor = CAPACITY_PROMPTS[cap]
        print(f"    {cap:25s} ({factor})  "
              f"F1={rl[c_idx,0]:+.3f}  F2={rl[c_idx,1]:+.3f}")

    print(f"\n  Entity factor scores (0-1):")
    scores = pca_results["factor_scores_01"]
    for e_idx, ent in enumerate(entity_keys):
        h_e, h_a = GRAY_ET_AL_SCORES[ent]
        print(f"    {ent:15s}  F1={scores[e_idx,0]:.3f}  "
              f"F2={scores[e_idx,1]:.3f}  "
              f"(human: E={h_e:.2f} A={h_a:.2f})")

    print(f"\n  Correlation with human scores:")
    for fi in range(min(2, scores.shape[1])):
        r_e = human_corr[f"f{fi+1}_experience"]
        r_a = human_corr[f"f{fi+1}_agency"]
        sig_e = "*" if r_e["p"] < 0.05 else ""
        sig_a = "*" if r_a["p"] < 0.05 else ""
        print(f"    Factor {fi+1} vs Experience: "
              f"rho={r_e['rho']:+.3f} (p={r_e['p']:.4f}){sig_e}")
        print(f"    Factor {fi+1} vs Agency:     "
              f"rho={r_a['rho']:+.3f} (p={r_a['p']:.4f}){sig_a}")

    # Best alignment
    best_exp = max(
        [(abs(human_corr[f"f{fi+1}_experience"]["rho"]),
          human_corr[f"f{fi+1}_experience"]["rho"],
          human_corr[f"f{fi+1}_experience"]["p"], fi+1)
         for fi in range(min(2, scores.shape[1]))],
        key=lambda x: x[0]
    )
    best_ag = max(
        [(abs(human_corr[f"f{fi+1}_agency"]["rho"]),
          human_corr[f"f{fi+1}_agency"]["rho"],
          human_corr[f"f{fi+1}_agency"]["p"], fi+1)
         for fi in range(min(2, scores.shape[1]))],
        key=lambda x: x[0]
    )
    print(f"\n  Best Experience match: Factor {best_exp[3]} "
          f"(rho={best_exp[1]:+.3f}, p={best_exp[2]:.4f})")
    print(f"  Best Agency match:     Factor {best_ag[3]} "
          f"(rho={best_ag[1]:+.3f}, p={best_ag[2]:.4f})")


# ========================== CONDITION ANALYSIS ========================== #

def analyze_condition(tag, entity_keys, capacity_keys):
    """Run full debiasing reanalysis for one condition."""
    d_dir = data_dir("gray_entities", "behavioral", tag)
    raw_path = os.path.join(str(d_dir), "pairwise_raw_responses.json")

    print(f"\n{'='*70}")
    print(f"  CONDITION: {tag} ({len(entity_keys)} entities)")
    print(f"{'='*70}")

    with open(raw_path) as f:
        responses = json.load(f)
    print(f"\n  Loaded {len(responses)} responses from {raw_path}")

    # ── Original (no debiasing, for comparison) ──
    print(f"\n--- Method: Original (expected rating, no debiasing) ---")
    ent_to_idx = {k: i for i, k in enumerate(entity_keys)}
    orig_scores = [[[] for _ in range(len(entity_keys))]
                   for _ in range(len(capacity_keys))]
    for resp in responses:
        cap_idx = capacity_keys.index(resp["capacity"])
        a_idx = ent_to_idx[resp["entity_a"]]
        b_idx = ent_to_idx[resp["entity_b"]]
        r = resp["expected_rating"]
        orig_scores[cap_idx][a_idx].append(3 - r)
        orig_scores[cap_idx][b_idx].append(r - 3)
    orig_means = np.zeros((len(capacity_keys), len(entity_keys)))
    for c in range(len(capacity_keys)):
        for e in range(len(entity_keys)):
            if orig_scores[c][e]:
                orig_means[c, e] = np.mean(orig_scores[c][e])
    orig_pca = run_pca_varimax(orig_means)
    orig_corr = correlate_with_humans(orig_pca, entity_keys, GRAY_ET_AL_SCORES)
    print_results("Original (expected rating)", orig_pca, orig_corr,
                  entity_keys, capacity_keys)

    # ── Method 1: Analytical debiasing ──
    print(f"\n--- Method 1: Analytical debiasing ---")
    debiased = compute_debiased_ratings(responses)
    debiased_means = debiased_to_character_means(
        debiased, entity_keys, capacity_keys
    )
    debiased_pca = run_pca_varimax(debiased_means)
    debiased_corr = correlate_with_humans(
        debiased_pca, entity_keys, GRAY_ET_AL_SCORES
    )
    print_results("Analytical debiasing", debiased_pca, debiased_corr,
                  entity_keys, capacity_keys)

    # ── Method 2: Log-odds ──
    print(f"\n--- Method 2: Log-odds ---")
    logodds = compute_logodds_ratings(responses)
    logodds_means = logodds_to_character_means(
        logodds, entity_keys, capacity_keys
    )
    logodds_pca = run_pca_varimax(logodds_means)
    logodds_corr = correlate_with_humans(
        logodds_pca, entity_keys, GRAY_ET_AL_SCORES
    )
    print_results("Log-odds", logodds_pca, logodds_corr,
                  entity_keys, capacity_keys)

    # ── Summary comparison ──
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {tag}")
    print(f"{'='*70}")
    print(f"\n  {'Method':<25s} | {'Best Exp rho':>12s} | {'p':>8s} "
          f"| {'Best Ag rho':>12s} | {'p':>8s}")
    print(f"  {'-'*25}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*8}")

    for label, corr in [("Original", orig_corr),
                         ("Debiased", debiased_corr),
                         ("Log-odds", logodds_corr)]:
        best_e = max(
            [(abs(corr[f"f{fi+1}_experience"]["rho"]),
              corr[f"f{fi+1}_experience"]["rho"],
              corr[f"f{fi+1}_experience"]["p"])
             for fi in range(2)], key=lambda x: x[0])
        best_a = max(
            [(abs(corr[f"f{fi+1}_agency"]["rho"]),
              corr[f"f{fi+1}_agency"]["rho"],
              corr[f"f{fi+1}_agency"]["p"])
             for fi in range(2)], key=lambda x: x[0])
        sig_e = "*" if best_e[2] < 0.05 else " "
        sig_a = "*" if best_a[2] < 0.05 else " "
        print(f"  {label:<25s} | {best_e[1]:+.3f}       | "
              f"{best_e[2]:.4f}{sig_e} | "
              f"{best_a[1]:+.3f}       | {best_a[2]:.4f}{sig_a}")

    # Save reanalysis results
    np.savez_compressed(
        os.path.join(str(d_dir), "debiasing_reanalysis.npz"),
        debiased_means=debiased_means,
        logodds_means=logodds_means,
        entity_keys=np.array(entity_keys),
        capacity_keys=np.array(capacity_keys),
    )
    print(f"\n  Saved debiasing_reanalysis.npz to {d_dir}/")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(
        description="Exp 4: Debiasing reanalysis of pairwise data (base only)"
    )
    add_model_argument(parser)
    parser.add_argument(
        "--include_self", action="store_true",
        help="Include 'you_self' entity (default: exclude)"
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Run both with_self and without_self conditions"
    )
    args = parser.parse_args()

    if args.model != "llama2_13b_base":
        parser.error("Debiasing reanalysis requires --model llama2_13b_base "
                      "(needs probability distributions in responses)")

    set_model(args.model)
    print(f"Model: {config.MODEL_LABEL}")

    capacity_keys = CAPACITY_NAMES

    if args.both:
        # Run both conditions
        entity_keys_no_self = [k for k in ENTITY_NAMES if k != "you_self"]
        analyze_condition("without_self", entity_keys_no_self, capacity_keys)
        analyze_condition("with_self", ENTITY_NAMES, capacity_keys)
    else:
        tag = get_condition_tag(args.include_self)
        if args.include_self:
            entity_keys = ENTITY_NAMES
        else:
            entity_keys = [k for k in ENTITY_NAMES if k != "you_self"]
        analyze_condition(tag, entity_keys, capacity_keys)


if __name__ == "__main__":
    main()
