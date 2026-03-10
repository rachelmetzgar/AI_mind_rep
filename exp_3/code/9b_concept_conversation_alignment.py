#!/usr/bin/env python3
"""
Experiment 3, Phase 9b: Concept-Conversation Alignment

Compares standalone concept activation vectors to conversation activations
to test whether concepts are more "present" in human vs AI conversations.

Three sub-approaches:
  A (full prompt-set mean): Average 40 prompts → one concept vector → cosine
  C (concept-contrastive):  Subtract mean of other concepts to isolate unique signal
  D (prompt-level):         Per-prompt alignment — identifies which prompts drive effects

Output per approach:
    results/llama2_13b_chat/{version}/concept_conversation/turn_{turn}/
        approach_{a,c,d}/
            alignment_scores.npz
            stats.csv
            prompt_stats.csv     (approach D only)
            figures/

Usage:
    python 9b_concept_conversation_alignment.py --version balanced_gpt
    python 9b_concept_conversation_alignment.py --version balanced_gpt --approaches a c

Env: llama2_env (CPU, needs scipy/numpy)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    config, set_version, set_model,
    add_version_argument, add_model_argument, add_turn_argument,
    add_variant_argument, set_variant,
    ensure_dir, get_model, DIMENSION_CATEGORIES,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Concept-conversation alignment analysis."
    )
    add_version_argument(parser)
    add_model_argument(parser)
    add_turn_argument(parser)
    parser.add_argument(
        "--approaches", nargs="+", default=["a", "c", "d"],
        choices=["a", "c", "d"],
        help="Which approaches to run (default: all).",
    )
    parser.add_argument(
        "--dim_ids", nargs="+", default=None,
        help="Filter to specific dimension names (e.g., 1_phenomenology).",
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Bootstrap iterations for CIs (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    add_variant_argument(parser)
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_conversation_activations(version, turn, model_name):
    """Load cached conversation activations and metadata."""
    act_dir = os.path.join(
        str(config.RESULTS.root), model_name, version,
        "conversation_activations", f"turn_{turn}",
    )
    npz_path = os.path.join(act_dir, "activations.npz")
    meta_path = os.path.join(act_dir, "metadata.csv")

    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Conversation activations not found: {npz_path}\n"
            f"Run 9a_extract_conversation_activations.py first."
        )

    print(f"Loading conversation activations from {act_dir}...")
    data = np.load(npz_path)
    activations = data["activations"].astype(np.float32)  # (n_convs, n_layers, hidden_dim)
    metadata = pd.read_csv(meta_path)
    print(f"  Shape: {activations.shape}")
    print(f"  Conditions: {metadata['condition'].value_counts().to_dict()}")
    return activations, metadata


def discover_standalone_dims(model_name, dim_filter=None):
    """Discover available standalone concept dimensions."""
    standalone_dir = str(config.RESULTS.concept_activations_standalone)
    dims = []
    for name in sorted(os.listdir(standalone_dir)):
        dim_dir = os.path.join(standalone_dir, name)
        if not os.path.isdir(dim_dir):
            continue
        mean_path = os.path.join(dim_dir, "mean_vectors_per_layer.npz")
        acts_path = os.path.join(dim_dir, "concept_activations.npz")
        if os.path.exists(mean_path):
            if dim_filter and name not in dim_filter:
                continue
            dims.append({
                "name": name,
                "mean_path": mean_path,
                "acts_path": acts_path,
            })
    return dims


def load_concept_mean(mean_path):
    """Load mean concept vector per layer. Returns (n_layers, hidden_dim)."""
    data = np.load(mean_path)
    return data["mean_concept"].astype(np.float32)


def load_concept_prompts(acts_path):
    """Load per-prompt concept activations. Returns (n_prompts, n_layers, hidden_dim)."""
    data = np.load(acts_path)
    return data["activations"].astype(np.float32)


# ============================================================
# ALIGNMENT COMPUTATION
# ============================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm


def compute_alignment_per_conv(concept_vec, conv_acts, layer_start=6):
    """Compute mean cosine similarity between concept vector and each conversation.

    Args:
        concept_vec: (n_layers, hidden_dim) — concept direction
        conv_acts: (n_convs, n_layers, hidden_dim) — conversation activations

    Returns:
        per_conv_alignment: (n_convs,) — mean cosine across layers 6-40
        per_layer_alignment: (n_convs, n_used_layers) — per-layer cosines
    """
    n_convs, n_layers, hidden_dim = conv_acts.shape
    used_layers = list(range(layer_start, n_layers))

    per_layer = np.zeros((n_convs, len(used_layers)), dtype=np.float32)
    for i, layer_idx in enumerate(used_layers):
        cv = concept_vec[layer_idx]
        cv_norm = np.linalg.norm(cv)
        if cv_norm == 0:
            continue
        cv_unit = cv / cv_norm

        # Vectorized cosine for all conversations at this layer
        ca = conv_acts[:, layer_idx, :]  # (n_convs, hidden_dim)
        ca_norms = np.linalg.norm(ca, axis=1, keepdims=True)
        ca_norms = np.maximum(ca_norms, 1e-10)
        cosines = (ca @ cv_unit) / ca_norms.squeeze()
        per_layer[:, i] = cosines

    mean_alignment = per_layer.mean(axis=1)  # (n_convs,)
    return mean_alignment, per_layer


def compute_stats(human_scores, ai_scores, n_bootstrap=1000, rng=None):
    """Compute t-test, effect size, and bootstrap CI for H-vs-A difference."""
    if rng is None:
        rng = np.random.default_rng(42)

    h_mean = np.mean(human_scores)
    a_mean = np.mean(ai_scores)
    diff = h_mean - a_mean

    t_stat, p_val = ttest_ind(human_scores, ai_scores)

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(human_scores, ddof=1) * (len(human_scores) - 1) +
         np.var(ai_scores, ddof=1) * (len(ai_scores) - 1)) /
        (len(human_scores) + len(ai_scores) - 2)
    )
    cohen_d = diff / pooled_std if pooled_std > 0 else 0.0

    # Bootstrap CI on mean difference
    boot_diffs = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        h_boot = rng.choice(human_scores, size=len(human_scores), replace=True)
        a_boot = rng.choice(ai_scores, size=len(ai_scores), replace=True)
        boot_diffs[b] = np.mean(h_boot) - np.mean(a_boot)

    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)

    return {
        "human_mean": h_mean,
        "ai_mean": a_mean,
        "diff": diff,
        "t": t_stat,
        "p": p_val,
        "cohen_d": cohen_d,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_human": len(human_scores),
        "n_ai": len(ai_scores),
    }


def fdr_correction(p_values):
    """Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[sorted_idx[i]] = sorted_p[i]
        else:
            adjusted[sorted_idx[i]] = min(
                sorted_p[i] * n / (i + 1),
                adjusted[sorted_idx[i + 1]],
            )
    return np.minimum(adjusted, 1.0)


# ============================================================
# APPROACH A: Full prompt-set mean
# ============================================================

def run_approach_a(dims, conv_acts, metadata, out_dir, n_bootstrap, rng):
    """Mean concept vector → cosine with each conversation."""
    print(f"\n{'='*70}")
    print("APPROACH A: Full prompt-set mean alignment")
    print(f"{'='*70}")

    ensure_dir(out_dir)
    human_mask = (metadata["condition"] == "human").values
    ai_mask = (metadata["condition"] == "ai").values

    rows = []
    all_scores = {}

    for dim in dims:
        concept_vec = load_concept_mean(dim["mean_path"])
        alignment, per_layer = compute_alignment_per_conv(concept_vec, conv_acts)

        human_scores = alignment[human_mask]
        ai_scores = alignment[ai_mask]
        stats = compute_stats(human_scores, ai_scores, n_bootstrap, rng)
        stats["dimension"] = dim["name"]
        rows.append(stats)
        all_scores[dim["name"]] = {
            "alignment": alignment,
            "per_layer": per_layer,
        }

        sig = ""
        if stats["p"] < 0.001: sig = "***"
        elif stats["p"] < 0.01: sig = "**"
        elif stats["p"] < 0.05: sig = "*"
        print(f"  {dim['name']:<25} H={stats['human_mean']:+.5f}  A={stats['ai_mean']:+.5f}  "
              f"diff={stats['diff']:+.5f}  p={stats['p']:.4f} {sig}  d={stats['cohen_d']:+.3f}")

    # FDR correction
    p_values = [r["p"] for r in rows]
    p_fdr = fdr_correction(p_values)
    for i, r in enumerate(rows):
        r["p_fdr"] = p_fdr[i]

    # Save stats
    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    # Save alignment scores
    np.savez_compressed(
        os.path.join(out_dir, "alignment_scores.npz"),
        **{f"{name}_alignment": data["alignment"] for name, data in all_scores.items()},
        **{f"{name}_per_layer": data["per_layer"] for name, data in all_scores.items()},
        conditions=metadata["condition"].values,
    )

    # Summary
    n_sig = sum(1 for r in rows if r["p"] < 0.05)
    n_sig_fdr = sum(1 for r in rows if r["p_fdr"] < 0.05)
    print(f"\n  Summary: {n_sig}/{len(rows)} significant (p<.05), "
          f"{n_sig_fdr}/{len(rows)} after FDR")

    return stats_df


# ============================================================
# APPROACH C: Concept-contrastive
# ============================================================

def run_approach_c(dims, conv_acts, metadata, out_dir, n_bootstrap, rng):
    """Subtract mean of other concepts to isolate unique signal."""
    print(f"\n{'='*70}")
    print("APPROACH C: Concept-contrastive alignment")
    print(f"{'='*70}")

    ensure_dir(out_dir)
    human_mask = (metadata["condition"] == "human").values
    ai_mask = (metadata["condition"] == "ai").values

    # Load all concept mean vectors
    concept_vecs = {}
    for dim in dims:
        concept_vecs[dim["name"]] = load_concept_mean(dim["mean_path"])

    # Compute grand mean across all concepts
    all_vecs = np.stack(list(concept_vecs.values()))  # (n_concepts, n_layers, hidden_dim)
    grand_mean = all_vecs.mean(axis=0)  # (n_layers, hidden_dim)

    rows = []
    all_scores = {}

    for dim in dims:
        # Contrastive: concept_k - mean(others)
        # = concept_k - (sum_all - concept_k) / (n-1)
        # Simpler: concept_k - grand_mean (which includes concept_k, but with n concepts
        # the difference is small)
        n_concepts = len(concept_vecs)
        other_mean = (grand_mean * n_concepts - concept_vecs[dim["name"]]) / (n_concepts - 1)
        contrastive_vec = concept_vecs[dim["name"]] - other_mean

        # Unit-normalize per layer
        for layer_idx in range(contrastive_vec.shape[0]):
            norm = np.linalg.norm(contrastive_vec[layer_idx])
            if norm > 0:
                contrastive_vec[layer_idx] /= norm

        alignment, per_layer = compute_alignment_per_conv(contrastive_vec, conv_acts)

        human_scores = alignment[human_mask]
        ai_scores = alignment[ai_mask]
        stats = compute_stats(human_scores, ai_scores, n_bootstrap, rng)
        stats["dimension"] = dim["name"]
        rows.append(stats)
        all_scores[dim["name"]] = {
            "alignment": alignment,
            "per_layer": per_layer,
        }

        sig = ""
        if stats["p"] < 0.001: sig = "***"
        elif stats["p"] < 0.01: sig = "**"
        elif stats["p"] < 0.05: sig = "*"
        print(f"  {dim['name']:<25} H={stats['human_mean']:+.5f}  A={stats['ai_mean']:+.5f}  "
              f"diff={stats['diff']:+.5f}  p={stats['p']:.4f} {sig}  d={stats['cohen_d']:+.3f}")

    # FDR correction
    p_values = [r["p"] for r in rows]
    p_fdr = fdr_correction(p_values)
    for i, r in enumerate(rows):
        r["p_fdr"] = p_fdr[i]

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "alignment_scores.npz"),
        **{f"{name}_alignment": data["alignment"] for name, data in all_scores.items()},
        **{f"{name}_per_layer": data["per_layer"] for name, data in all_scores.items()},
        conditions=metadata["condition"].values,
    )

    n_sig = sum(1 for r in rows if r["p"] < 0.05)
    n_sig_fdr = sum(1 for r in rows if r["p_fdr"] < 0.05)
    print(f"\n  Summary: {n_sig}/{len(rows)} significant (p<.05), "
          f"{n_sig_fdr}/{len(rows)} after FDR")

    return stats_df


# ============================================================
# APPROACH D: Per-prompt alignment
# ============================================================

def run_approach_d(dims, conv_acts, metadata, out_dir, n_bootstrap, rng):
    """Per-prompt alignment — identifies which prompts drive effects."""
    print(f"\n{'='*70}")
    print("APPROACH D: Per-prompt alignment")
    print(f"{'='*70}")

    ensure_dir(out_dir)
    human_mask = (metadata["condition"] == "human").values
    ai_mask = (metadata["condition"] == "ai").values
    layer_start = config.ANALYSIS.restricted_layer_start

    all_concept_rows = []
    all_prompt_rows = []

    for dim in dims:
        if not os.path.exists(dim["acts_path"]):
            print(f"  [SKIP] No per-prompt activations: {dim['name']}")
            continue

        prompt_acts = load_concept_prompts(dim["acts_path"])  # (n_prompts, n_layers, hidden_dim)
        n_prompts = prompt_acts.shape[0]

        # Compute per-prompt alignment with each conversation
        prompt_alignments = []  # list of (n_convs,) arrays, one per prompt
        prompt_rows = []

        for p in range(n_prompts):
            prompt_vec = prompt_acts[p]  # (n_layers, hidden_dim)
            alignment, _ = compute_alignment_per_conv(prompt_vec, conv_acts, layer_start)
            prompt_alignments.append(alignment)

            # Per-prompt stats
            h_scores = alignment[human_mask]
            a_scores = alignment[ai_mask]
            t_stat, p_val = ttest_ind(h_scores, a_scores)

            prompt_rows.append({
                "dimension": dim["name"],
                "prompt_idx": p,
                "human_mean": np.mean(h_scores),
                "ai_mean": np.mean(a_scores),
                "diff": np.mean(h_scores) - np.mean(a_scores),
                "t": t_stat,
                "p": p_val,
            })

        all_prompt_rows.extend(prompt_rows)

        # Average across prompts per conversation (averaging in cosine space)
        stacked = np.stack(prompt_alignments)  # (n_prompts, n_convs)
        mean_alignment = stacked.mean(axis=0)  # (n_convs,)

        human_scores = mean_alignment[human_mask]
        ai_scores = mean_alignment[ai_mask]
        stats = compute_stats(human_scores, ai_scores, n_bootstrap, rng)
        stats["dimension"] = dim["name"]

        # Count how many prompts show the effect
        n_prompt_sig = sum(1 for r in prompt_rows if r["p"] < 0.05)
        stats["n_prompts_sig"] = n_prompt_sig
        stats["n_prompts_total"] = n_prompts

        all_concept_rows.append(stats)

        sig = ""
        if stats["p"] < 0.001: sig = "***"
        elif stats["p"] < 0.01: sig = "**"
        elif stats["p"] < 0.05: sig = "*"
        print(f"  {dim['name']:<25} H={stats['human_mean']:+.5f}  A={stats['ai_mean']:+.5f}  "
              f"diff={stats['diff']:+.5f}  p={stats['p']:.4f} {sig}  "
              f"prompts sig: {n_prompt_sig}/{n_prompts}")

    # FDR correction on concept-level
    if all_concept_rows:
        p_values = [r["p"] for r in all_concept_rows]
        p_fdr = fdr_correction(p_values)
        for i, r in enumerate(all_concept_rows):
            r["p_fdr"] = p_fdr[i]

    stats_df = pd.DataFrame(all_concept_rows)
    stats_df.to_csv(os.path.join(out_dir, "stats.csv"), index=False)

    prompt_df = pd.DataFrame(all_prompt_rows)
    prompt_df.to_csv(os.path.join(out_dir, "prompt_stats.csv"), index=False)

    n_sig = sum(1 for r in all_concept_rows if r["p"] < 0.05)
    n_sig_fdr = sum(1 for r in all_concept_rows if r["p_fdr"] < 0.05)
    print(f"\n  Summary: {n_sig}/{len(all_concept_rows)} significant (p<.05), "
          f"{n_sig_fdr}/{len(all_concept_rows)} after FDR")

    return stats_df, prompt_df


# ============================================================
# CROSS-APPROACH SUMMARY
# ============================================================

def make_cross_summary(results, out_dir):
    """Combine stats from all approaches into one comparison table."""
    rows = []
    for approach, df in results.items():
        for _, row in df.iterrows():
            rows.append({
                "approach": approach,
                "dimension": row["dimension"],
                "human_mean": row["human_mean"],
                "ai_mean": row["ai_mean"],
                "diff": row["diff"],
                "p": row["p"],
                "p_fdr": row["p_fdr"],
                "cohen_d": row["cohen_d"],
            })

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "cross_approach_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Cross-approach summary: {summary_path}")

    # Print compact comparison
    print(f"\n{'='*90}")
    print(f"CROSS-APPROACH COMPARISON: Human - AI alignment difference")
    print(f"{'='*90}")
    print(f"{'Dimension':<25} {'A diff':<12} {'A p':<10} {'C diff':<12} {'C p':<10} {'D diff':<12} {'D p':<10}")
    print("-" * 91)

    dim_names = summary_df["dimension"].unique()
    for dim in sorted(dim_names):
        line = f"{dim:<25}"
        for approach in ["A", "C", "D"]:
            sub = summary_df[(summary_df["dimension"] == dim) & (summary_df["approach"] == approach)]
            if len(sub) > 0:
                r = sub.iloc[0]
                sig = ""
                if r["p_fdr"] < 0.001: sig = "***"
                elif r["p_fdr"] < 0.01: sig = "**"
                elif r["p_fdr"] < 0.05: sig = "*"
                line += f" {r['diff']:+.5f}{sig:<4} {r['p']:<10.4f}"
            else:
                line += f" {'N/A':<12} {'N/A':<10}"
        print(line)


# ============================================================
# FIGURES
# ============================================================

def make_figures(stats_df, approach_name, out_dir):
    """Generate alignment figures for one approach."""
    if stats_df.empty:
        print(f"  [SKIP] No data for approach {approach_name}, skipping figures")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping figures")
        return

    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))

    # Bar chart: H-A difference per concept
    stats_sorted = stats_df.sort_values("diff", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = []
    for _, row in stats_sorted.iterrows():
        if row["p_fdr"] < 0.05:
            colors.append("#2196F3")
        elif row["p"] < 0.05:
            colors.append("#90CAF9")
        else:
            colors.append("#BDBDBD")

    ax.bar(range(len(stats_sorted)), stats_sorted["diff"], color=colors)
    ax.set_xticks(range(len(stats_sorted)))
    ax.set_xticklabels(stats_sorted["dimension"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Human - AI mean cosine alignment")
    ax.set_title(f"Approach {approach_name}: Concept-Conversation Alignment (H - A)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#2196F3", label="p_fdr < .05"),
            plt.Rectangle((0, 0), 1, 1, color="#90CAF9", label="p < .05 (uncorrected)"),
            plt.Rectangle((0, 0), 1, 1, color="#BDBDBD", label="n.s."),
        ],
        loc="upper right", fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "alignment_by_concept.png"), dpi=150)
    plt.close()
    print(f"  Saved: {fig_dir}/alignment_by_concept.png")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.variant:
        set_variant(args.variant)
    set_model(args.model)
    set_version(args.version, turn=args.turn)

    model_name = get_model()

    # Load conversation activations
    conv_acts, metadata = load_conversation_activations(
        args.version, args.turn, model_name,
    )

    # Discover standalone concept dimensions
    dims = discover_standalone_dims(model_name, dim_filter=args.dim_ids)
    if not dims:
        print("[ERROR] No standalone concept dimensions found.")
        sys.exit(1)
    print(f"\nFound {len(dims)} concept dimensions")

    # Output root
    variant_suffix = args.variant if args.variant else ""
    out_root = os.path.join(
        str(config.RESULTS.root), model_name, args.version,
        f"concept_conversation{variant_suffix}", f"turn_{args.turn}",
    )

    # Run approaches
    results = {}

    if "a" in args.approaches:
        a_dir = ensure_dir(os.path.join(out_root, "approach_a"))
        a_stats = run_approach_a(dims, conv_acts, metadata, a_dir, args.n_bootstrap, rng)
        make_figures(a_stats, "A", a_dir)
        results["A"] = a_stats

    if "c" in args.approaches:
        c_dir = ensure_dir(os.path.join(out_root, "approach_c"))
        c_stats = run_approach_c(dims, conv_acts, metadata, c_dir, args.n_bootstrap, rng)
        make_figures(c_stats, "C", c_dir)
        results["C"] = c_stats

    if "d" in args.approaches:
        d_dir = ensure_dir(os.path.join(out_root, "approach_d"))
        d_stats, prompt_stats = run_approach_d(
            dims, conv_acts, metadata, d_dir, args.n_bootstrap, rng,
        )
        make_figures(d_stats, "D", d_dir)
        results["D"] = d_stats

    # Cross-approach summary
    if len(results) > 1:
        make_cross_summary(results, out_root)

    print(f"\nConcept-conversation alignment analysis complete.")


if __name__ == "__main__":
    main()
