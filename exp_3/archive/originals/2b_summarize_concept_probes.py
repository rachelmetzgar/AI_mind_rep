#!/usr/bin/env python3
"""
Summarize all concept probe training results: CSV, statistical tests, figures.

Reads from:
    data/concept_probes/{dim_name}/accuracy_summary.pkl
    data/alignment/{dim_name}/{control,reading}_probe/alignment_results.json

Outputs:
    data/concept_probes/summary_all_dimensions.csv       (flat CSV)
    data/concept_probes/summary_stats/statistical_results.json
    data/concept_probes/summary_stats/fig_*.png           (5 figures)

Statistical tests:
    1. Each dimension's mean-diff alignment vs chance (t-test, H0: cos=0)
    2. Each dimension vs shapes (bootstrap on per-layer cosines)
    3. Each dimension vs baseline (bootstrap on per-layer cosines)
    4. Category-level pooled comparisons (Welch's t, Cohen's d)

Key note on Dim 15 (shapes):
    Shapes probes separate ROUND vs ANGULAR (not human vs AI).
    Its alignment with conversational probes is the floor expectation
    for a semantically irrelevant direction.

Usage:
    python summarize_concept_probes.py

Env: llama2_env (or any env with numpy, scipy, matplotlib; no GPU)
Rachel C. Metzgar · Feb 2026
"""

import os
import sys
import json
import pickle
import csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import config


# ========================== CONFIG ========================== #

PROBE_ROOT = str(config.PATHS.concept_probes)
ALIGN_ROOT = str(config.RESULTS.alignment)
OUTPUT_CSV = str(config.RESULTS.root / "probes" / "summary_all_dimensions.csv")
STATS_DIR  = str(config.RESULTS.root / "probes" / "stats")
HIDDEN_DIM = config.INPUT_DIM
N_BOOTSTRAP = config.ANALYSIS.n_bootstrap
RNG = np.random.default_rng(config.ANALYSIS.seed)

CATEGORIES = {
    "Mental":    [1, 2, 3, 4, 5, 6, 7, 16, 17],  # 16=holistic mind (pooled 1-10), 17=attention
    "Physical":  [8, 9, 10],
    "Pragmatic": [11, 12, 13],
    "Baseline":  [0],
    "Bio Ctrl":  [14],
    "Shapes":    [15],
    "SysPrompt": [18],
}

CAT_COLORS = {
    "Mental": "#2196F3", "Physical": "#4CAF50", "Pragmatic": "#FF9800",
    "Baseline": "#9E9E9E", "Bio Ctrl": "#795548", "Shapes": "#E91E63",
    "SysPrompt": "#00BCD4",
}

# Layer cutoff for restricted analysis (exclude early-layer prompt-format confound)
RESTRICTED_LAYER_START = 6


def dim_category(dim_id):
    for cat, ids in CATEGORIES.items():
        if dim_id in ids:
            return cat
    return "Other"


# ========================== DATA LOADING ========================== #

def discover_dimensions():
    """Find all trained concept probe directories."""
    dims = {}
    if not os.path.isdir(PROBE_ROOT):
        return dims
    for name in sorted(os.listdir(PROBE_ROOT)):
        full = os.path.join(PROBE_ROOT, name)
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


def load_accuracy(probe_dir):
    """Load accuracy_summary.pkl, return dict of stats."""
    pkl_path = os.path.join(probe_dir, "accuracy_summary.pkl")
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        summary = pickle.load(f)
    accs = summary.get("acc", [])
    if not accs:
        return None
    accs = np.array(accs)
    n_passing = int(np.sum(accs >= 0.70))
    first_90 = int(np.argmax(accs >= 0.90)) if np.any(accs >= 0.90) else -1
    return {
        "n_layers": len(accs),
        "mean_acc": float(np.mean(accs)),
        "max_acc": float(np.max(accs)),
        "max_acc_layer": int(np.argmax(accs)),
        "min_acc": float(np.min(accs)),
        "n_passing_70": n_passing,
        "first_layer_90": first_90,
        "acc_layer_0": float(accs[0]) if len(accs) > 0 else None,
        "acc_layer_10": float(accs[10]) if len(accs) > 10 else None,
        "acc_layer_20": float(accs[20]) if len(accs) > 20 else None,
        "acc_layer_30": float(accs[30]) if len(accs) > 30 else None,
        "acc_layer_40": float(accs[40]) if len(accs) > 40 else None,
        "per_layer_accs_list": accs.tolist(),
        "per_layer_accs": ",".join(f"{a:.3f}" for a in accs),
    }


def load_alignment(dim_name, probe_type):
    """Load alignment JSON, return summary stats + per-layer values."""
    json_path = os.path.join(
        ALIGN_ROOT, dim_name, probe_type, "alignment_results.json"
    )
    if not os.path.isfile(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)

    result = {}
    per_layer = {}
    for key in ["probe_to_2b", "mean_to_2b", "probe_to_mean"]:
        vals = [v for v in data.get(key, []) if v is not None]
        per_layer[key] = data.get(key, [])
        if vals:
            arr = np.array(vals)
            result[f"{key}_mean"] = float(np.mean(arr))
            result[f"{key}_mean_abs"] = float(np.mean(np.abs(arr)))
            result[f"{key}_max_abs"] = float(np.max(np.abs(arr)))
            result[f"{key}_max_layer"] = int(
                data["layers"][np.argmax(np.abs(arr))]
            )
        else:
            result[f"{key}_mean"] = None
            result[f"{key}_mean_abs"] = None
            result[f"{key}_max_abs"] = None
            result[f"{key}_max_layer"] = None
    return result, per_layer


# ========================== STATISTICAL TESTS ========================== #

def chance_cosine_stats(d=HIDDEN_DIM):
    """Analytical null for cosine between random vectors in R^d."""
    sigma = 1.0 / np.sqrt(d)
    return {
        "mean": 0.0,
        "sigma": sigma,
        "expected_abs": np.sqrt(2 / (np.pi * d)),
        "z_threshold_05": 1.96 * sigma,
        "z_threshold_01": 2.576 * sigma,
        "z_threshold_001": 3.291 * sigma,
    }


def test_vs_chance(per_layer_cosines, d=HIDDEN_DIM):
    """One-sample t-test + bootstrap CI on per-layer cosines vs 0."""
    cosines = np.array([c for c in per_layer_cosines if c is not None])
    if len(cosines) < 3:
        return None
    mean_cos = np.mean(cosines)
    sem = np.std(cosines, ddof=1) / np.sqrt(len(cosines))
    sigma_chance = 1.0 / np.sqrt(d)
    t_stat, p_val = stats.ttest_1samp(cosines, 0)
    z_score = mean_cos / sigma_chance
    boot_means = np.array([
        np.mean(RNG.choice(cosines, size=len(cosines), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "mean": float(mean_cos),
        "sem": float(sem),
        "n_layers": len(cosines),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "z_score": float(z_score),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "sigma_chance": float(sigma_chance),
    }


def test_pairwise(cosines_a, cosines_b, name_a="A", name_b="B"):
    """Bootstrap test: is mean(cosines_a) different from mean(cosines_b)?"""
    a = np.array([c for c in cosines_a if c is not None])
    b = np.array([c for c in cosines_b if c is not None])
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    if min_len < 3:
        return None
    obs_diff = np.mean(a) - np.mean(b)
    boot_diffs = np.array([
        np.mean(RNG.choice(a, size=min_len, replace=True))
        - np.mean(RNG.choice(b, size=min_len, replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    p_val = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    return {
        "dim_a": name_a, "dim_b": name_b,
        "mean_a": float(np.mean(a)), "mean_b": float(np.mean(b)),
        "diff": float(obs_diff), "p_value": float(p_val),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "sig_05": bool(p_val < 0.05),
    }


# ========================== MAIN ========================== #

def main():
    os.makedirs(STATS_DIR, exist_ok=True)

    dims = discover_dimensions()
    if not dims:
        print("[ERROR] No concept probe directories found.")
        return
    print(f"Found {len(dims)} dimensions: {sorted(dims.keys())}")

    chance = chance_cosine_stats()
    sigma = chance["sigma"]
    print(f"\nChance cosine in R^{HIDDEN_DIM}: E=0, σ={sigma:.4f}, 3σ={3*sigma:.4f}")

    # ── Load everything ──
    all_data = {}
    csv_rows = []
    per_layer_align = {}

    for dim_id, dim_name in sorted(dims.items()):
        probe_dir = os.path.join(PROBE_ROOT, dim_name)
        row = {"dim_id": dim_id, "dim_name": dim_name}
        entry = {"dim_id": dim_id, "dim_name": dim_name,
                 "category": dim_category(dim_id)}

        acc_stats = load_accuracy(probe_dir)
        if acc_stats:
            accs_list = acc_stats.pop("per_layer_accs_list")
            per_layer_csv = acc_stats.pop("per_layer_accs")
            row.update(acc_stats)
            row["per_layer_accs"] = per_layer_csv
            entry["per_layer_accs"] = accs_list
            entry["mean_acc"] = acc_stats["mean_acc"]
        else:
            print(f"  [WARN] No accuracy data for {dim_name}")

        per_layer_align[dim_id] = {}
        for probe_type in ["control_probe", "reading_probe"]:
            loaded = load_alignment(dim_name, probe_type)
            if loaded:
                align_summary, pl = loaded
                prefixed = {f"{probe_type}_{k}": v for k, v in align_summary.items()}
                row.update(prefixed)
                per_layer_align[dim_id][probe_type] = pl
                for k, v in pl.items():
                    entry[f"{probe_type}_{k}"] = v

        csv_rows.append(row)
        all_data[dim_id] = entry

    # ── Write CSV ──
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        if "per_layer_accs" in fieldnames:
            fieldnames.remove("per_layer_accs")
            fieldnames.append("per_layer_accs")
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"\n✅ CSV saved to {OUTPUT_CSV}")

    # ── Print readable table ──
    print(f"\n{'='*120}")
    print(f"{'Dim':>4} {'Name':<25} {'Cat':<10} {'Mean':>6} {'Max':>5} {'MaxL':>5} "
          f"{'≥70%':>5} {'1st90':>5} │ "
          f"{'Ctrl M↔2b':>10} {'Read M↔2b':>10} │ "
          f"{'z(ctrl)':>8} {'z(read)':>8}")
    print(f"{'-'*120}")
    for row in csv_rows:
        ctrl_m = row.get("control_probe_mean_to_2b_mean", None)
        read_m = row.get("reading_probe_mean_to_2b_mean", None)
        z_c = ctrl_m / sigma if ctrl_m is not None else None
        z_r = read_m / sigma if read_m is not None else None
        line = (
            f"{row['dim_id']:>4} {row['dim_name']:<25} "
            f"{dim_category(row['dim_id']):<10} "
            f"{row.get('mean_acc', 0):>6.3f} "
            f"{row.get('max_acc', 0):>5.3f} "
            f"{row.get('max_acc_layer', ''):>5} "
            f"{row.get('n_passing_70', ''):>5} "
            f"{row.get('first_layer_90', ''):>5} │ "
        )
        line += f"{ctrl_m:>10.4f} " if ctrl_m is not None else f"{'N/A':>10} "
        line += f"{read_m:>10.4f} │ " if read_m is not None else f"{'N/A':>10} │ "
        line += f"{z_c:>7.1f}σ " if z_c is not None else f"{'':>8} "
        line += f"{z_r:>7.1f}σ" if z_r is not None else f"{'':>8}"
        print(line)

    # ═════════════════════════════════════════════════════════════
    #  STATISTICAL TESTS
    # ═════════════════════════════════════════════════════════════

    # ── Test 1: Each dim vs chance ──
    print(f"\n{'='*100}")
    print(f"TEST 1: MEAN-DIFF ALIGNMENT vs CHANCE (one-sample t-test, H0: cos = 0)")
    print(f"{'='*100}")

    vs_chance_results = {}
    for probe_type in ["control_probe", "reading_probe"]:
        pt_label = probe_type.replace("_probe", "").title()
        print(f"\n--- {pt_label} Probe ---")
        print(f"{'Dim':>3} {'Name':<22} {'Cat':<10} │ {'Mean':>8} {'t':>7} {'p':>10} "
              f"{'z':>6} {'95% CI':>20} {'Sig':>4}")
        print("-" * 100)

        results = {}
        for dim_id in sorted(all_data.keys()):
            pl = per_layer_align.get(dim_id, {}).get(probe_type, {})
            cosines = pl.get("mean_to_2b", [])
            if not cosines:
                continue
            res = test_vs_chance(cosines)
            if res is None:
                continue
            results[dim_id] = res
            sig = ("***" if res["p_value"] < 0.001 else
                   "**"  if res["p_value"] < 0.01  else
                   "*"   if res["p_value"] < 0.05  else "")
            print(f"{dim_id:>3} {all_data[dim_id]['dim_name']:<22} "
                  f"{all_data[dim_id]['category']:<10} │ "
                  f"{res['mean']:>8.4f} {res['t_stat']:>7.2f} {res['p_value']:>10.6f} "
                  f"{res['z_score']:>5.1f}σ "
                  f"[{res['ci_lo']:>7.4f}, {res['ci_hi']:>7.4f}] {sig:>4}")
        vs_chance_results[probe_type] = results

    # ── Test 2: Each dim vs shapes ──
    print(f"\n{'='*100}")
    print(f"TEST 2: PAIRWISE vs SHAPES (dim 15, round↔angular) — bootstrap")
    print(f"{'='*100}")

    vs_shapes_results = {}
    for probe_type in ["control_probe", "reading_probe"]:
        pt_label = probe_type.replace("_probe", "").title()
        print(f"\n--- {pt_label} Probe ---")
        key = "mean_to_2b"
        shapes_cos = per_layer_align.get(15, {}).get(probe_type, {}).get(key, [])
        if not shapes_cos:
            print(f"  [WARN] No shapes data for {probe_type}")
            continue
        print(f"{'Dim':>3} {'Name':<22} {'Cat':<10} │ {'Mean':>7} {'Shapes':>7} "
              f"{'Diff':>7} {'p':>10} {'95% CI diff':>20} {'Sig':>4}")
        print("-" * 100)
        results = {}
        for dim_id in sorted(all_data.keys()):
            if dim_id == 15:
                continue
            dim_cos = per_layer_align.get(dim_id, {}).get(probe_type, {}).get(key, [])
            if not dim_cos:
                continue
            res = test_pairwise(dim_cos, shapes_cos,
                                name_a=all_data[dim_id]["dim_name"],
                                name_b="15_shapes")
            if res is None:
                continue
            results[dim_id] = res
            sig = ("***" if res["p_value"] < 0.001 else
                   "**"  if res["p_value"] < 0.01  else
                   "*"   if res["p_value"] < 0.05  else "")
            print(f"{dim_id:>3} {all_data[dim_id]['dim_name']:<22} "
                  f"{all_data[dim_id]['category']:<10} │ "
                  f"{res['mean_a']:>7.4f} {res['mean_b']:>7.4f} "
                  f"{res['diff']:>7.4f} {res['p_value']:>10.4f} "
                  f"[{res['ci_lo']:>7.4f}, {res['ci_hi']:>7.4f}] {sig:>4}")
        vs_shapes_results[probe_type] = results

    # ── Test 3: Each dim vs baseline ──
    print(f"\n{'='*100}")
    print(f"TEST 3: PAIRWISE vs BASELINE (dim 0, generic entity) — bootstrap")
    print(f"{'='*100}")

    vs_baseline_results = {}
    for probe_type in ["control_probe", "reading_probe"]:
        pt_label = probe_type.replace("_probe", "").title()
        print(f"\n--- {pt_label} Probe ---")
        key = "mean_to_2b"
        baseline_cos = per_layer_align.get(0, {}).get(probe_type, {}).get(key, [])
        if not baseline_cos:
            print(f"  [WARN] No baseline data for {probe_type}")
            continue
        print(f"{'Dim':>3} {'Name':<22} {'Cat':<10} │ {'Mean':>7} {'Bline':>7} "
              f"{'Diff':>7} {'p':>10} {'95% CI diff':>20} {'Sig':>4}")
        print("-" * 100)
        results = {}
        for dim_id in sorted(all_data.keys()):
            if dim_id == 0:
                continue
            dim_cos = per_layer_align.get(dim_id, {}).get(probe_type, {}).get(key, [])
            if not dim_cos:
                continue
            res = test_pairwise(dim_cos, baseline_cos,
                                name_a=all_data[dim_id]["dim_name"],
                                name_b="0_baseline")
            if res is None:
                continue
            results[dim_id] = res
            sig = ("***" if res["p_value"] < 0.001 else
                   "**"  if res["p_value"] < 0.01  else
                   "*"   if res["p_value"] < 0.05  else "")
            print(f"{dim_id:>3} {all_data[dim_id]['dim_name']:<22} "
                  f"{all_data[dim_id]['category']:<10} │ "
                  f"{res['mean_a']:>7.4f} {res['mean_b']:>7.4f} "
                  f"{res['diff']:>7.4f} {res['p_value']:>10.4f} "
                  f"[{res['ci_lo']:>7.4f}, {res['ci_hi']:>7.4f}] {sig:>4}")
        vs_baseline_results[probe_type] = results

    # ── Test 4: Category-level ──
    print(f"\n{'='*100}")
    print(f"TEST 4: CATEGORY COMPARISONS (pooled per-layer cosines, Welch t)")
    print(f"{'='*100}")

    for probe_type in ["control_probe", "reading_probe"]:
        pt_label = probe_type.replace("_probe", "").title()
        print(f"\n--- {pt_label} Probe ---")
        key = "mean_to_2b"

        cat_pooled = {}
        for cat_name, cat_ids in CATEGORIES.items():
            pooled = []
            for did in cat_ids:
                vals = per_layer_align.get(did, {}).get(probe_type, {}).get(key, [])
                pooled.extend([v for v in vals if v is not None])
            if pooled:
                cat_pooled[cat_name] = np.array(pooled)

        if "Mental" not in cat_pooled:
            continue
        mental = cat_pooled["Mental"]
        print(f"\n  Mental (n={len(mental)}, mean={np.mean(mental):.4f}) vs:")
        for other_cat in ["Shapes", "Pragmatic", "Physical", "Baseline", "Bio Ctrl",
                          "SysPrompt"]:
            if other_cat not in cat_pooled:
                continue
            other = cat_pooled[other_cat]
            t, p = stats.ttest_ind(mental, other, equal_var=False)
            pooled_var = (np.var(mental, ddof=1) + np.var(other, ddof=1)) / 2
            d_cohen = (np.mean(mental) - np.mean(other)) / np.sqrt(pooled_var) if pooled_var > 0 else 0
            sig = ("***" if p < 0.001 else "**" if p < 0.01 else
                   "*" if p < 0.05 else "")
            print(f"    {other_cat:<12} (n={len(other):>3}, mean={np.mean(other):.4f}) "
                  f"Δ={np.mean(mental)-np.mean(other):>7.4f} "
                  f"t={t:>6.2f} p={p:.4f} d={d_cohen:.3f} {sig}")

    # ═════════════════════════════════════════════════════════════
    #  FIGURES
    # ═════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print("GENERATING FIGURES")
    print("=" * 100)

    # ── Fig A: Accuracy by layer ──
    fig, ax = plt.subplots(figsize=(12, 5))
    for dim_id in sorted(all_data.keys()):
        d = all_data[dim_id]
        accs = d.get("per_layer_accs", [])
        if not accs:
            continue
        c = CAT_COLORS.get(d["category"], "#666")
        emph = dim_id in [13, 15, 11]
        ax.plot(range(len(accs)), accs, color=c,
                linewidth=2.0 if emph else 0.7,
                alpha=0.9 if emph else 0.25,
                label=d["dim_name"] if emph else None)
    ax.axhline(0.7, color="red", ls="--", alpha=0.4, lw=1)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.3)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Best Val Accuracy", fontsize=11)
    ax.set_title("Concept Probe Classification Accuracy by Layer", fontsize=13)
    ax.set_ylim(0.45, 1.05)
    ax.set_xlim(0, 40)
    patches = [mpatches.Patch(color=c, label=l) for l, c in CAT_COLORS.items()]
    ax.legend(handles=patches + [
        plt.Line2D([], [], color="red", ls="--", alpha=0.4, label="70% threshold")
    ], fontsize=7, loc="lower right", ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "fig_accuracy_by_layer.png"), dpi=200)
    plt.close()

    # ── Fig B: Alignment bars with CIs ──
    for probe_type, probe_label in [("control_probe", "Control"),
                                     ("reading_probe", "Reading")]:
        if probe_type not in vs_chance_results:
            continue
        results = vs_chance_results[probe_type]
        if not results:
            continue

        fig, ax = plt.subplots(figsize=(10, 7))
        ranked = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)
        names = [all_data[did]["dim_name"].replace("_", " ") for did, _ in ranked]
        means = [r["mean"] for _, r in ranked]
        ci_lo = [r["ci_lo"] for _, r in ranked]
        ci_hi = [r["ci_hi"] for _, r in ranked]
        errs = [[m - lo for m, lo in zip(means, ci_lo)],
                [hi - m for m, hi in zip(means, ci_hi)]]
        colors = [CAT_COLORS.get(dim_category(did), "#666") for did, _ in ranked]

        ax.barh(range(len(names)), means, xerr=errs, color=colors,
                edgecolor="white", linewidth=0.5, capsize=3)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(3 * sigma, color="red", ls="--", alpha=0.6, lw=1,
                   label=f"3σ chance ({3*sigma:.4f})")
        shapes_mean = results.get(15, {}).get("mean", 0)
        ax.axvline(shapes_mean, color="#E91E63", ls=":", alpha=0.8, lw=1.5,
                   label=f"Shapes = {shapes_mean:.4f}")

        for i, (did, r) in enumerate(ranked):
            star = ("***" if r["p_value"] < 0.001 else
                    "**"  if r["p_value"] < 0.01  else
                    "*"   if r["p_value"] < 0.05  else "")
            if star:
                ax.text(r["ci_hi"] + 0.001, i, star, va="center",
                        fontsize=8, color="red")

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Cosine Similarity (95% bootstrap CI)", fontsize=10)
        ax.set_title(f"Mean-Diff ↔ Exp 2b {probe_label} Probe\n"
                     f"(shapes = round↔angular, not human↔AI)", fontsize=12)
        ax.legend(fontsize=9, loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(STATS_DIR, f"fig_alignment_{probe_type}.png"),
                    dpi=200)
        plt.close()

    # ── Fig C: Category comparison ──
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_order = ["Mental\n(1-7,16,17)", "Physical\n(8-10)", "Pragmatic\n(11-13)",
                 "SysPrompt\n(18)", "Baseline\n(0)", "Bio Ctrl\n(14)", "Shapes\n(15)"]
    cat_ids_order = [
        CATEGORIES["Mental"], CATEGORIES["Physical"], CATEGORIES["Pragmatic"],
        CATEGORIES["SysPrompt"],
        CATEGORIES["Baseline"], CATEGORIES["Bio Ctrl"], CATEGORIES["Shapes"],
    ]
    x = np.arange(len(cat_order))
    width = 0.35
    for offset, probe_type, color, label in [
        (-width/2, "control_probe", "#1976D2", "↔ Control Probe"),
        (width/2, "reading_probe", "#E64A19", "↔ Reading Probe"),
    ]:
        if probe_type not in vs_chance_results:
            continue
        results = vs_chance_results[probe_type]
        means, stds = [], []
        for ids in cat_ids_order:
            vals = [results[d]["mean"] for d in ids if d in results]
            means.append(np.mean(vals) if vals else 0)
            stds.append(np.std(vals) if len(vals) > 1 else 0)
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=3)
    ax.axhline(3 * sigma, color="red", ls="--", alpha=0.5, label="3σ chance")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=10)
    ax.set_title("Concept Direction Alignment by Category", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_order, fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "fig_category_comparison.png"), dpi=200)
    plt.close()

    # ── Fig D: Convergence artifact ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for dim_id in sorted(all_data.keys()):
        d = all_data[dim_id]
        c = CAT_COLORS.get(d["category"], "#666")
        read_m = d.get("reading_probe_mean_to_2b", [])
        read_p = d.get("reading_probe_probe_to_2b", [])
        if read_m and read_p:
            mm = np.mean([v for v in read_m if v is not None])
            pm = np.mean([v for v in read_p if v is not None])
            ax1.scatter(mm, pm, color=c, s=100, zorder=3,
                        edgecolors="white", linewidth=0.8)
            oy = -15 if dim_id == 16 else 5
            ax1.annotate(str(dim_id), (mm, pm), fontsize=7, ha="center",
                         va="bottom", xytext=(0, oy), textcoords="offset points")

        ctrl_pm = d.get("control_probe_probe_to_mean", [])
        if ctrl_pm and "mean_acc" in d:
            fid = np.mean([v for v in ctrl_pm if v is not None])
            ax2.scatter(d["mean_acc"], fid, color=c, s=100, zorder=3,
                        edgecolors="white", linewidth=0.8)
            ax2.annotate(str(dim_id), (d["mean_acc"], fid), fontsize=7,
                         ha="center", va="bottom", xytext=(0, 5),
                         textcoords="offset points")

    ax1.set_xlabel("Mean-Diff ↔ Reading Probe (raw direction)", fontsize=10)
    ax1.set_ylabel("Trained Probe ↔ Reading Probe", fontsize=10)
    ax1.set_title("Trained probes converge toward reading probe\n"
                  "regardless of raw concept direction", fontsize=11)
    if 16 in all_data:
        rm = all_data[16].get("reading_probe_mean_to_2b", [])
        rp = all_data[16].get("reading_probe_probe_to_2b", [])
        if rm and rp:
            rm_v = [v for v in rm if v is not None]
            rp_v = [v for v in rp if v is not None]
            if rm_v and rp_v:
                ax1.annotate("dim 16 (mind)\nresists convergence",
                             xy=(np.mean(rm_v), np.mean(rp_v)),
                             xytext=(0.013, 0.15), fontsize=8, color="#9C27B0",
                             arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=1.5))

    ax2.set_xlabel("Mean Classification Accuracy", fontsize=10)
    ax2.set_ylabel("Probe ↔ Mean-Diff Fidelity", fontsize=10)
    ax2.set_title("Harder classification → trained probe drifts\n"
                  "further from raw direction", fontsize=11)
    patches = [mpatches.Patch(color=c, label=l) for l, c in CAT_COLORS.items()]
    fig.legend(handles=patches, loc="upper center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "fig_convergence_artifact.png"),
                dpi=200, bbox_inches="tight")
    plt.close()

    # ═════════════════════════════════════════════════════════════
    #  TEST 5: RESTRICTED LAYER ANALYSIS (layers 6+)
    # ═════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print(f"TEST 5: RESTRICTED LAYER ANALYSIS (layers {RESTRICTED_LAYER_START}+, excluding early-layer confound)")
    print(f"{'='*100}")

    vs_chance_restricted = {}
    for probe_type in ["control_probe", "reading_probe"]:
        pt_label = probe_type.replace("_probe", "").title()
        print(f"\n--- {pt_label} Probe (layers {RESTRICTED_LAYER_START}-40 only) ---")
        print(f"{'Dim':>3} {'Name':<22} {'Cat':<10} │ {'Mean':>8} {'t':>7} {'p':>10} "
              f"{'z':>6} {'95% CI':>20} {'Sig':>4}")
        print("-" * 100)

        results = {}
        for dim_id in sorted(all_data.keys()):
            pl = per_layer_align.get(dim_id, {}).get(probe_type, {})
            cosines = pl.get("mean_to_2b", [])
            if not cosines or len(cosines) <= RESTRICTED_LAYER_START:
                continue
            restricted = cosines[RESTRICTED_LAYER_START:]
            res = test_vs_chance(restricted)
            if res is None:
                continue
            results[dim_id] = res
            sig = ("***" if res["p_value"] < 0.001 else
                   "**"  if res["p_value"] < 0.01  else
                   "*"   if res["p_value"] < 0.05  else "")
            print(f"{dim_id:>3} {all_data[dim_id]['dim_name']:<22} "
                  f"{all_data[dim_id]['category']:<10} │ "
                  f"{res['mean']:>8.4f} {res['t_stat']:>7.2f} {res['p_value']:>10.6f} "
                  f"{res['z_score']:>5.1f}σ "
                  f"[{res['ci_lo']:>7.4f}, {res['ci_hi']:>7.4f}] {sig:>4}")
        vs_chance_restricted[probe_type] = results

    # Summary comparison: all layers vs restricted
    print(f"\n--- Comparison: all layers vs layers {RESTRICTED_LAYER_START}+ ---")
    print(f"{'Dim':>3} {'Name':<20} {'Cat':<10} │ {'All mean':>9} {'All p':>10} │ "
          f"{'Restr mean':>10} {'Restr p':>10} │ {'Change':>8}")
    print("-" * 100)
    for dim_id in sorted(all_data.keys()):
        full = vs_chance_results.get("control_probe", {}).get(dim_id)
        rest = vs_chance_restricted.get("control_probe", {}).get(dim_id)
        if full and rest:
            sig_f = "***" if full["p_value"] < 0.001 else "**" if full["p_value"] < 0.01 else "*" if full["p_value"] < 0.05 else "n.s."
            sig_r = "***" if rest["p_value"] < 0.001 else "**" if rest["p_value"] < 0.01 else "*" if rest["p_value"] < 0.05 else "n.s."
            print(f"{dim_id:>3} {all_data[dim_id]['dim_name']:<20} "
                  f"{all_data[dim_id]['category']:<10} │ "
                  f"{full['mean']:>8.4f} {sig_f:>10} │ "
                  f"{rest['mean']:>10.4f} {sig_r:>10} │ "
                  f"{rest['mean']-full['mean']:>+8.4f}")

    # ═════════════════════════════════════════════════════════════
    #  PER-CATEGORY BREAKDOWN FIGURES
    # ═════════════════════════════════════════════════════════════

    print(f"\n{'='*100}")
    print("GENERATING PER-CATEGORY BREAKDOWN FIGURES")
    print("=" * 100)

    breakdown_dir = os.path.join(STATS_DIR, "category_breakdowns")
    os.makedirs(breakdown_dir, exist_ok=True)

    for cat_name, cat_ids in CATEGORIES.items():
        if len(cat_ids) < 2:
            continue  # skip single-member categories

        for probe_type, probe_label in [("control_probe", "Control"),
                                         ("reading_probe", "Reading")]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(cat_ids) * 0.6 + 1)),
                                           gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})

            # Left panel: bar chart of dimensions within this category
            key = "mean_to_2b"
            dim_stats = []
            for did in cat_ids:
                res = vs_chance_results.get(probe_type, {}).get(did)
                if res:
                    dim_stats.append((did, res))

            if not dim_stats:
                plt.close()
                continue

            dim_stats.sort(key=lambda x: x[1]["mean"], reverse=True)
            y_pos = np.arange(len(dim_stats))
            names = [all_data[d]["dim_name"].replace("_", " ") for d, _ in dim_stats]
            means = [r["mean"] for _, r in dim_stats]
            ci_lo = [r["ci_lo"] for _, r in dim_stats]
            ci_hi = [r["ci_hi"] for _, r in dim_stats]
            errs = [[m - lo for m, lo in zip(means, ci_lo)],
                    [hi - m for m, hi in zip(means, ci_hi)]]

            color = CAT_COLORS.get(cat_name, "#666")
            ax1.barh(y_pos, means, xerr=errs, color=color,
                     edgecolor="white", linewidth=0.5, capsize=3, alpha=0.8)
            ax1.axvline(0, color="black", lw=0.4)
            ax1.axvline(3 * sigma, color="red", ls="--", alpha=0.5, lw=0.8)
            shapes_mean_val = vs_chance_results.get(probe_type, {}).get(15, {}).get("mean", 0)
            ax1.axvline(shapes_mean_val, color="#E91E63", ls=":", alpha=0.6, lw=1)

            for i, (did, r) in enumerate(dim_stats):
                star = ("***" if r["p_value"] < 0.001 else
                        "**"  if r["p_value"] < 0.01  else
                        "*"   if r["p_value"] < 0.05  else "")
                if star:
                    ax1.text(r["ci_hi"] + 0.0005, i, star, va="center",
                            fontsize=8, color="#333", fontweight="bold")

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(names, fontsize=9)
            ax1.invert_yaxis()
            ax1.set_xlabel("Mean Cosine Similarity (95% CI)", fontsize=10)
            ax1.set_title(f"{cat_name}: {probe_label} Probe Alignment", fontsize=11)

            # Right panel: layer-by-layer profiles for each dim
            for did, _ in dim_stats:
                vals = per_layer_align.get(did, {}).get(probe_type, {}).get(key, [])
                if vals:
                    ax2.plot(range(len(vals)), vals, lw=1.2, alpha=0.7,
                             label=all_data[did]["dim_name"].replace("_", " "))
            ax2.axhline(0, color="black", lw=0.4)
            ax2.axhline(3 * sigma, color="red", ls="--", alpha=0.4, lw=0.8)
            ax2.axvline(RESTRICTED_LAYER_START, color="gray", ls=":", lw=0.8, alpha=0.5)
            ax2.set_xlabel("Layer", fontsize=10)
            ax2.set_ylabel("Cosine Similarity", fontsize=10)
            ax2.set_title(f"{cat_name}: Layer Profiles ({probe_label})", fontsize=11)
            ax2.legend(fontsize=7, loc="best")
            ax2.set_xlim(0, 40)

            plt.tight_layout()
            fname = f"breakdown_{cat_name.lower().replace(' ', '_')}_{probe_type}.png"
            plt.savefig(os.path.join(breakdown_dir, fname), dpi=200)
            plt.close()
            print(f"  Saved: {fname}")

    # ── Fig E: Restricted vs full comparison ──
    fig, ax = plt.subplots(figsize=(10, 7))
    ctrl_full = vs_chance_results.get("control_probe", {})
    ctrl_rest = vs_chance_restricted.get("control_probe", {})
    for dim_id in sorted(all_data.keys()):
        f = ctrl_full.get(dim_id)
        r = ctrl_rest.get(dim_id)
        if f and r:
            c = CAT_COLORS.get(all_data[dim_id]["category"], "#666")
            ax.scatter(f["mean"], r["mean"], s=80, color=c,
                       edgecolors="white", linewidth=0.7, zorder=5)
            ax.annotate(all_data[dim_id]["dim_name"].split("_")[-1],
                        (f["mean"], r["mean"]), fontsize=6.5,
                        xytext=(4, 3), textcoords="offset points", color="#333")
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, "k--", lw=0.5, alpha=0.3, zorder=0)
    ax.set_xlabel("Mean Cosine (all layers)", fontsize=10)
    ax.set_ylabel(f"Mean Cosine (layers {RESTRICTED_LAYER_START}+ only)", fontsize=10)
    ax.set_title(f"Restricted Layer Analysis: All vs Layers {RESTRICTED_LAYER_START}+\n"
                 f"(points below diagonal = alignment inflated by early layers)", fontsize=11)
    patches = [mpatches.Patch(color=c, label=l) for l, c in CAT_COLORS.items()]
    ax.legend(handles=patches, fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(STATS_DIR, "fig_restricted_vs_full.png"), dpi=200)
    plt.close()

    # ── Save JSON ──
    summary = {
        "chance_stats": chance,
        "categories": {k: v for k, v in CATEGORIES.items()},
        "restricted_layer_start": RESTRICTED_LAYER_START,
        "vs_chance": {
            pt: {str(k): v for k, v in res.items()}
            for pt, res in vs_chance_results.items()
        },
        "vs_chance_restricted": {
            pt: {str(k): v for k, v in res.items()}
            for pt, res in vs_chance_restricted.items()
        },
        "vs_shapes": {
            pt: {str(k): v for k, v in res.items()}
            for pt, res in vs_shapes_results.items()
        },
        "vs_baseline": {
            pt: {str(k): v for k, v in res.items()}
            for pt, res in vs_baseline_results.items()
        },
    }
    json_path = os.path.join(STATS_DIR, "statistical_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ ALL OUTPUTS:")
    print(f"   CSV:    {OUTPUT_CSV}")
    print(f"   Stats:  {json_path}")
    print(f"   Figs:   {STATS_DIR}/fig_*.png")
    print(f"{'='*60}")
    print(f"\nKey:")
    print(f"  Shapes (dim 15) = round↔angular (NOT human↔AI)")
    print(f"  Baseline (dim 0) = generic human↔AI entity")
    print(f"  M↔2b = mean-diff ↔ Exp 2b probe (raw concept direction)")
    print(f"  P↔2b = trained probe ↔ Exp 2b probe (convergence artifact)")


if __name__ == "__main__":
    main()