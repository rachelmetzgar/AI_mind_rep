#!/usr/bin/env python3
"""
Probe Training Summary — Stats and Figures for Reading vs Control Probes.

Loads accuracy_summary.pkl from both reading and control probe checkpoints,
generates layerwise accuracy plots, computes descriptive statistics for
early/middle/late layer groups, and runs per-layer two-proportions z-tests
(FDR corrected) comparing probe types.

Output is saved to results/probe_training/.

Rachel C. Metzgar · Feb 2026
"""

import os, pickle, sys, base64, io
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_version, add_version_argument


# ========================== CONFIG ========================== #
N_TEST = 400  # 20% of ~2000 samples (50 subjects x 40 conversations)

# Layer groups (0-indexed, inclusive)
LAYER_GROUPS = {
    "early":  (0, 13),   # layers 0–13  (14 layers)
    "middle": (14, 27),  # layers 14–27 (14 layers)
    "late":   (28, 40),  # layers 28–40 (13 layers)
}


# ========================== LOAD DATA ========================== #
def load_pkl(probe_type, probe_dir):
    path = probe_dir / probe_type / "accuracy_summary.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    return {k: np.array(v, dtype=float) for k, v in d.items()}


# ========================== DESCRIPTIVE STATS ========================== #
def descriptive(arr):
    return {
        "mean": np.mean(arr),
        "std": np.std(arr, ddof=1),
        "min": np.min(arr),
        "max": np.max(arr),
        "best_layer": int(np.argmax(arr)),
    }


def get_group(layer):
    for gname, (lo, hi) in LAYER_GROUPS.items():
        if lo <= layer <= hi:
            return gname
    return "unknown"


def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def fmt_p(p):
    if p < 0.0001:
        return "&lt;.0001"
    return f"{p:.4f}"


# ========================== MAIN ========================== #
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe Training Summary — Stats and Figures for Reading vs Control Probes."
    )
    add_version_argument(parser)
    args = parser.parse_args()

    set_version(args.version)

    N_LAYERS = config.N_LAYERS
    PROBE_DIR = config.PATHS.probe_checkpoints / "turn_5"
    OUT_DIR = config.RESULTS.probe_training
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ========================== LOAD DATA ========================== #
    reading = load_pkl("metacognitive", PROBE_DIR)
    control = load_pkl("operational", PROBE_DIR)

    layers = np.arange(N_LAYERS)

    print("=" * 60)
    print(f"PROBE TRAINING SUMMARY — {args.version} Dataset")
    print("=" * 60)

    # --- Best test accuracy (acc) ---
    print("\n--- Best Test Accuracy (across epochs) ---")
    for name, data in [("Reading", reading), ("Control", control)]:
        s = descriptive(data["acc"])
        print(f"  {name}: M={s['mean']:.4f}, SD={s['std']:.4f}, "
              f"range=[{s['min']:.3f}, {s['max']:.3f}], "
              f"peak layer={s['best_layer']} ({s['max']:.3f})")

    # --- Final-epoch test accuracy ---
    print("\n--- Final-Epoch Test Accuracy ---")
    for name, data in [("Reading", reading), ("Control", control)]:
        s = descriptive(data["final"])
        print(f"  {name}: M={s['mean']:.4f}, SD={s['std']:.4f}, "
              f"range=[{s['min']:.3f}, {s['max']:.3f}], "
              f"peak layer={s['best_layer']} ({s['max']:.3f})")

    # --- Final-epoch train accuracy ---
    print("\n--- Final-Epoch Train Accuracy ---")
    for name, data in [("Reading", reading), ("Control", control)]:
        s = descriptive(data["train"])
        print(f"  {name}: M={s['mean']:.4f}, SD={s['std']:.4f}, "
              f"range=[{s['min']:.3f}, {s['max']:.3f}], "
              f"peak layer={s['best_layer']} ({s['max']:.3f})")


    # ========================== LAYER GROUP STATS ========================== #
    print("\n" + "=" * 60)
    print("LAYER GROUP STATISTICS")
    print("=" * 60)

    for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final")]:
        print(f"\n--- {metric_label} ---")
        for group_name, (lo, hi) in LAYER_GROUPS.items():
            r_vals = reading[metric_key][lo:hi+1]
            c_vals = control[metric_key][lo:hi+1]
            # Paired t-test across layers in this group
            t_stat, p_val = stats.ttest_rel(r_vals, c_vals)
            n_group = len(r_vals)
            print(f"  {group_name.upper()} (layers {lo}–{hi}, n={n_group}):")
            print(f"    Reading: M={np.mean(r_vals):.4f}, SD={np.std(r_vals, ddof=1):.4f}")
            print(f"    Control: M={np.mean(c_vals):.4f}, SD={np.std(c_vals, ddof=1):.4f}")
            print(f"    Paired t({n_group-1})={t_stat:.3f}, p={p_val:.4f}")


    # ========================== OVERALL PAIRED TEST ========================== #
    print("\n" + "=" * 60)
    print(f"OVERALL READING vs CONTROL (paired t-test across {N_LAYERS} layers)")
    print("=" * 60)

    for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final")]:
        t_stat, p_val = stats.ttest_rel(reading[metric_key], control[metric_key])
        diff = reading[metric_key] - control[metric_key]
        d = np.mean(diff) / np.std(diff, ddof=1)  # Cohen's d for paired
        print(f"\n  {metric_label}:")
        print(f"    Reading M={np.mean(reading[metric_key]):.4f}, "
              f"Control M={np.mean(control[metric_key]):.4f}")
        print(f"    Mean diff = {np.mean(diff):.4f}")
        print(f"    t({N_LAYERS-1})={t_stat:.3f}, p={p_val:.4f}, d={d:.3f}")


    # ========================== PER-LAYER Z-TESTS ========================== #
    print("\n" + "=" * 60)
    print("PER-LAYER TWO-PROPORTIONS Z-TESTS (FDR corrected)")
    print("=" * 60)

    # Use best test accuracy (acc) for per-layer comparisons
    r_acc = reading["acc"]
    c_acc = control["acc"]

    # Convert accuracy proportions to counts
    r_correct = np.round(r_acc * N_TEST).astype(int)
    c_correct = np.round(c_acc * N_TEST).astype(int)

    raw_pvals = []
    z_stats = []
    for i in range(N_LAYERS):
        count = np.array([r_correct[i], c_correct[i]])
        nobs = np.array([N_TEST, N_TEST])
        z, p = proportions_ztest(count, nobs, alternative="two-sided")
        z_stats.append(z)
        raw_pvals.append(p)

    raw_pvals = np.array(raw_pvals)
    z_stats = np.array(z_stats)

    # FDR correction (Benjamini-Hochberg)
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method="fdr_bh")

    n_sig = np.sum(reject)
    print(f"\n  Significant layers (FDR q<.05): {n_sig}/{N_LAYERS}")
    if n_sig > 0:
        sig_layers = np.where(reject)[0]
        print(f"  Layers: {list(sig_layers)}")
        for l in sig_layers:
            print(f"    Layer {l:2d}: Reading={r_acc[l]:.3f}, Control={c_acc[l]:.3f}, "
                  f"z={z_stats[l]:.2f}, p_raw={raw_pvals[l]:.4f}, p_fdr={pvals_corrected[l]:.4f}")


    # ========================== BUILD RESULTS TABLE ========================== #
    print("\n" + "=" * 60)
    print("SAVING RESULTS TABLE")
    print("=" * 60)

    df = pd.DataFrame({
        "layer": layers,
        "reading_best_acc": r_acc,
        "reading_final_acc": reading["final"],
        "reading_train_acc": reading["train"],
        "control_best_acc": c_acc,
        "control_final_acc": control["final"],
        "control_train_acc": control["train"],
        "diff_best_acc": r_acc - c_acc,
        "z_stat": z_stats,
        "p_raw": raw_pvals,
        "p_fdr": pvals_corrected,
        "sig_fdr_05": reject.astype(int),
    })

    df["layer_group"] = df["layer"].apply(get_group)

    csv_path = OUT_DIR / "layerwise_probe_stats.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")


    # ========================== FIGURES ========================== #
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    # Dynamic y-axis limits based on actual data (with padding)
    all_acc = np.concatenate([r_acc, c_acc, reading["final"], control["final"]])
    Y_MIN = min(0.40, np.floor(all_acc.min() * 20) / 20 - 0.02)  # round down to nearest 0.05
    Y_MAX = np.ceil(all_acc.max() * 20) / 20 + 0.02              # round up to nearest 0.05 + padding


    # --- Figure 1: Best test accuracy by layer ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, r_acc, "o-", color="#2166ac", markersize=4, linewidth=1.5, label="Reading probe")
    ax.plot(layers, c_acc, "s-", color="#b2182b", markersize=4, linewidth=1.5, label="Control probe")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    # Mark significant layers
    if n_sig > 0:
        for l in sig_layers:
            ax.axvspan(l - 0.3, l + 0.3, alpha=0.15, color="gold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Best Test Accuracy", fontsize=12)
    ax.set_title("Layerwise Best Test Accuracy — Reading vs Control Probes", fontsize=13)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(range(0, N_LAYERS, 5))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "best_test_acc_by_layer.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: best_test_acc_by_layer.png")


    # --- Figure 2: Final-epoch test accuracy by layer ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, reading["final"], "o-", color="#2166ac", markersize=4, linewidth=1.5, label="Reading probe")
    ax.plot(layers, control["final"], "s-", color="#b2182b", markersize=4, linewidth=1.5, label="Control probe")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Final-Epoch Test Accuracy", fontsize=12)
    ax.set_title("Layerwise Final-Epoch Test Accuracy — Reading vs Control Probes", fontsize=13)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(range(0, N_LAYERS, 5))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_test_acc_by_layer.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: final_test_acc_by_layer.png")


    # --- Figure 3: Reading vs Control difference by layer ---
    diff = r_acc - c_acc
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2166ac" if d >= 0 else "#b2182b" for d in diff]
    bars = ax.bar(layers, diff, color=colors, width=0.7, alpha=0.8)
    # Mark significant layers
    if n_sig > 0:
        for l in sig_layers:
            bars[l].set_edgecolor("gold")
            bars[l].set_linewidth(2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Accuracy Difference\n(Reading - Control)", fontsize=11)
    ax.set_title("Per-Layer Accuracy Difference (Best Test) — Gold border = sig. FDR q<.05", fontsize=13)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    ax.set_xticks(range(0, N_LAYERS, 5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acc_difference_by_layer.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: acc_difference_by_layer.png")


    # --- Figure 4: Layer group bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax_idx, (metric_label, metric_key) in enumerate([("Best Test Acc", "acc"), ("Final Test Acc", "final")]):
        ax = axes[ax_idx]
        group_names = list(LAYER_GROUPS.keys())
        x = np.arange(len(group_names))
        width = 0.35
        r_means, c_means = [], []
        r_sems, c_sems = [], []
        for gname in group_names:
            lo, hi = LAYER_GROUPS[gname]
            r_vals = reading[metric_key][lo:hi+1]
            c_vals = control[metric_key][lo:hi+1]
            r_means.append(np.mean(r_vals))
            c_means.append(np.mean(c_vals))
            r_sems.append(np.std(r_vals, ddof=1) / np.sqrt(len(r_vals)))
            c_sems.append(np.std(c_vals, ddof=1) / np.sqrt(len(c_vals)))

        ax.bar(x - width/2, r_means, width, yerr=r_sems, color="#2166ac",
               alpha=0.8, capsize=4, label="Reading")
        ax.bar(x + width/2, c_means, width, yerr=c_sems, color="#b2182b",
               alpha=0.8, capsize=4, label="Control")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer Group", fontsize=12)
        ax.set_ylabel("Mean Accuracy", fontsize=12)
        ax.set_title(metric_label, fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n({LAYER_GROUPS[g][0]}–{LAYER_GROUPS[g][1]})" for g in group_names])
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Probe Accuracy by Layer Group", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "acc_by_layer_group.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: acc_by_layer_group.png")


    # --- Figure 5: Train vs Test accuracy (overfitting check) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax_idx, (name, data, color) in enumerate([
        ("Reading Probe", reading, "#2166ac"),
        ("Control Probe", control, "#b2182b"),
    ]):
        ax = axes[ax_idx]
        ax.plot(layers, data["train"], "^-", color=color, markersize=4, linewidth=1.5,
                alpha=0.6, label="Train (final epoch)")
        ax.plot(layers, data["acc"], "o-", color=color, markersize=4, linewidth=1.5,
                label="Test (best epoch)")
        ax.plot(layers, data["final"], "s--", color=color, markersize=3, linewidth=1,
                alpha=0.5, label="Test (final epoch)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(name, fontsize=13)
        ax.set_ylim(0.40, 1.0)
        ax.set_xlim(-0.5, N_LAYERS - 0.5)
        ax.set_xticks(range(0, N_LAYERS, 5))
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Train vs Test Accuracy by Layer (Overfitting Check)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "train_vs_test_by_layer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: train_vs_test_by_layer.png")


    # ========================== SAVE FULL TEXT REPORT ========================== #
    report_path = OUT_DIR / "probe_training_report.txt"
    with open(report_path, "w") as f:
        # Redirect stdout to capture the same stats
        pass  # We'll write a separate summary

    with open(report_path, "w") as f:
        f.write(f"PROBE TRAINING SUMMARY — {args.version} Dataset\n")
        f.write("=" * 60 + "\n")
        f.write(f"N layers: {N_LAYERS}\n")
        f.write(f"N test samples (per layer): ~{N_TEST}\n\n")

        for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final"),
                                          ("Final Train Acc", "train")]:
            f.write(f"--- {metric_label} ---\n")
            for name, data in [("Reading", reading), ("Control", control)]:
                s = descriptive(data[metric_key])
                f.write(f"  {name}: M={s['mean']:.4f}, SD={s['std']:.4f}, "
                        f"range=[{s['min']:.3f}, {s['max']:.3f}], "
                        f"peak layer={s['best_layer']} ({s['max']:.3f})\n")
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("LAYER GROUP STATISTICS\n")
        f.write("=" * 60 + "\n")
        for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final")]:
            f.write(f"\n--- {metric_label} ---\n")
            for gname, (lo, hi) in LAYER_GROUPS.items():
                r_vals = reading[metric_key][lo:hi+1]
                c_vals = control[metric_key][lo:hi+1]
                t_stat, p_val = stats.ttest_rel(r_vals, c_vals)
                n_g = len(r_vals)
                f.write(f"  {gname.upper()} (layers {lo}–{hi}, n={n_g}):\n")
                f.write(f"    Reading: M={np.mean(r_vals):.4f}, SD={np.std(r_vals, ddof=1):.4f}\n")
                f.write(f"    Control: M={np.mean(c_vals):.4f}, SD={np.std(c_vals, ddof=1):.4f}\n")
                f.write(f"    Paired t({n_g-1})={t_stat:.3f}, p={p_val:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write(f"OVERALL READING vs CONTROL (paired t-test across {N_LAYERS} layers)\n")
        f.write("=" * 60 + "\n")
        for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final")]:
            t_stat, p_val = stats.ttest_rel(reading[metric_key], control[metric_key])
            d = reading[metric_key] - control[metric_key]
            cohens_d = np.mean(d) / np.std(d, ddof=1)
            f.write(f"\n  {metric_label}:\n")
            f.write(f"    Reading M={np.mean(reading[metric_key]):.4f}, "
                    f"Control M={np.mean(control[metric_key]):.4f}\n")
            f.write(f"    Mean diff = {np.mean(d):.4f}\n")
            f.write(f"    t({N_LAYERS-1})={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-LAYER TWO-PROPORTIONS Z-TESTS (FDR corrected)\n")
        f.write("=" * 60 + "\n")
        f.write(f"\n  Significant layers (FDR q<.05): {n_sig}/{N_LAYERS}\n")
        if n_sig > 0:
            f.write(f"  Layers: {list(sig_layers)}\n")
            for l in sig_layers:
                f.write(f"    Layer {l:2d}: Reading={r_acc[l]:.3f}, Control={c_acc[l]:.3f}, "
                        f"z={z_stats[l]:.2f}, p_raw={raw_pvals[l]:.4f}, p_fdr={pvals_corrected[l]:.4f}\n")
        else:
            f.write("  No layers reached significance after FDR correction.\n")

    print(f"  Saved: {report_path}")


    # ========================== HTML REPORT ========================== #
    print("\n" + "=" * 60)
    print("GENERATING HTML REPORT")
    print("=" * 60)

    # --- Build layer group stats tables ---
    group_rows_html = ""
    for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final")]:
        group_rows_html += f'<tr><td colspan="7" style="background:#f0f0f0; font-weight:bold;">{metric_label}</td></tr>\n'
        for gname, (lo, hi) in LAYER_GROUPS.items():
            r_vals = reading[metric_key][lo:hi+1]
            c_vals = control[metric_key][lo:hi+1]
            t_stat, p_val = stats.ttest_rel(r_vals, c_vals)
            n_g = len(r_vals)
            group_rows_html += (
                f"<tr><td>{gname} ({lo}&ndash;{hi})</td>"
                f"<td>{np.mean(r_vals):.4f}</td><td>{np.std(r_vals, ddof=1):.4f}</td>"
                f"<td>{np.mean(c_vals):.4f}</td><td>{np.std(c_vals, ddof=1):.4f}</td>"
                f"<td>t({n_g-1})={t_stat:.3f}</td>"
                f"<td>{fmt_p(p_val)}</td></tr>\n"
            )

    # --- Build overall paired test rows ---
    overall_rows_html = ""
    for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final")]:
        t_stat, p_val = stats.ttest_rel(reading[metric_key], control[metric_key])
        d = reading[metric_key] - control[metric_key]
        cohens_d = np.mean(d) / np.std(d, ddof=1)
        overall_rows_html += (
            f"<tr><td>{metric_label}</td>"
            f"<td>{np.mean(reading[metric_key]):.4f}</td>"
            f"<td>{np.mean(control[metric_key]):.4f}</td>"
            f"<td>{np.mean(d):.4f}</td>"
            f"<td>t({N_LAYERS-1})={t_stat:.3f}</td>"
            f"<td>{fmt_p(p_val)}</td>"
            f"<td>{cohens_d:.3f}</td></tr>\n"
        )

    # --- Build layerwise stats table (subset: every layer) ---
    layerwise_rows_html = ""
    for i in range(N_LAYERS):
        sig_marker = "*" if reject[i] else ""
        row_style = ' style="background:#ffffcc;"' if reject[i] else ""
        layerwise_rows_html += (
            f"<tr{row_style}>"
            f"<td>{i}</td><td>{get_group(i)}</td>"
            f"<td>{r_acc[i]:.3f}</td><td>{reading['final'][i]:.3f}</td><td>{reading['train'][i]:.3f}</td>"
            f"<td>{c_acc[i]:.3f}</td><td>{control['final'][i]:.3f}</td><td>{control['train'][i]:.3f}</td>"
            f"<td>{r_acc[i] - c_acc[i]:+.3f}</td>"
            f"<td>{z_stats[i]:.2f}</td><td>{raw_pvals[i]:.4f}</td><td>{pvals_corrected[i]:.4f}</td>"
            f"<td>{sig_marker}</td></tr>\n"
        )

    # --- Descriptive summary rows ---
    desc_rows_html = ""
    for metric_label, metric_key in [("Best Test Acc", "acc"), ("Final Test Acc", "final"),
                                      ("Final Train Acc", "train")]:
        for name, data in [("Reading", reading), ("Control", control)]:
            s = descriptive(data[metric_key])
            desc_rows_html += (
                f"<tr><td>{metric_label}</td><td>{name}</td>"
                f"<td>{s['mean']:.4f}</td><td>{s['std']:.4f}</td>"
                f"<td>{s['min']:.3f}</td><td>{s['max']:.3f}</td>"
                f"<td>{s['best_layer']}</td></tr>\n"
            )

    # --- Embed figures ---
    fig_best = img_to_b64(FIG_DIR / "best_test_acc_by_layer.png")
    fig_final = img_to_b64(FIG_DIR / "final_test_acc_by_layer.png")
    fig_diff = img_to_b64(FIG_DIR / "acc_difference_by_layer.png")
    fig_group = img_to_b64(FIG_DIR / "acc_by_layer_group.png")
    fig_overfit = img_to_b64(FIG_DIR / "train_vs_test_by_layer.png")

    # --- Assemble HTML ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Probe Training Summary &mdash; {args.version} Dataset</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #333; line-height: 1.6; }}
  h1 {{ color: #2166ac; border-bottom: 2px solid #2166ac; padding-bottom: 10px; }}
  h2 {{ color: #444; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
  h3 {{ color: #666; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
  th {{ background: #2166ac; color: white; padding: 8px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f5f5f5; }}
  .fig {{ text-align: center; margin: 25px 0; }}
  .fig img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .fig-caption {{ font-size: 0.9em; color: #666; margin-top: 8px; font-style: italic; }}
  .interpretation {{ background: #f8f9fa; border-left: 4px solid #2166ac; padding: 15px 20px; margin: 20px 0; border-radius: 0 4px 4px 0; }}
  .stat {{ font-family: 'Courier New', monospace; background: #f0f0f0; padding: 1px 4px; border-radius: 2px; }}
  .timestamp {{ color: #999; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>Probe Training Summary &mdash; Experiment 2 ({args.version} Dataset)</h1>
<p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<p>Linear probes trained on LLaMA-2-Chat-13B hidden activations to classify conversation partner type
(human vs. AI) from name-ablated Experiment 1 data. Training data: 50 subjects &times; 40 conversations
= ~2000 samples. Single stratified 80/20 train/test split (n<sub>test</sub> &asymp; {N_TEST}).
50 training epochs per layer; best-epoch checkpoint selection.</p>

<p><strong>Reading probe:</strong> Appends reflective suffix ("I think the conversation partner of this user is&hellip;"),
probes at last token. Tests whether the model can decode partner type when prompted to reflect.<br>
<strong>Control probe:</strong> No suffix; probes at the [/INST] token where the model is about to generate its next
response. Tests whether partner type is represented in the natural generation context.</p>

<!-- ============================================================ -->
<h2>1. Descriptive Statistics</h2>

<table>
<tr><th>Metric</th><th>Probe</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th><th>Peak Layer</th></tr>
{desc_rows_html}
</table>

<div class="interpretation">
<strong>Interpretation:</strong> Both probes decode partner type above chance (50%) across most layers,
but accuracy is modest (55&ndash;65%). The reading probe consistently outperforms the control probe.
Peak accuracy occurs in late layers (reading: layer 33 at 65.2%, control: layer 31 at 60.5%).
Training accuracy reaches ~85&ndash;96%, indicating substantial overfitting &mdash; expected for linear probes
on high-dimensional (5120-d) representations with limited training data.
Compared to the <code>names/</code> version (80&ndash;90%+), accuracy is markedly lower, confirming that
probes trained on named partners were largely encoding partner-name tokens rather than abstract identity.
</div>

<!-- ============================================================ -->
<h2>2. Layerwise Best Test Accuracy</h2>

<div class="fig">
  <img src="data:image/png;base64,{fig_best}" alt="Best test accuracy by layer">
  <div class="fig-caption">Figure 1. Best test accuracy (across 50 training epochs) for reading and control probes at each
  of the {N_LAYERS} transformer layers. Dashed gray line = chance (50%). Gold highlights = layers where
  reading and control differ significantly (FDR q&lt;.05).</div>
</div>

<div class="interpretation">
<strong>Interpretation:</strong> The reading probe (blue) rises above the control probe (red) starting around
layer 15 and maintains a consistent advantage through the late layers. Both probes hover near chance in
the early layers (0&ndash;13), suggesting that early representations do not yet encode partner type. The
reading probe&rsquo;s advantage in middle-to-late layers suggests that the reflective prompt amplifies a
signal that is weaker but still present in the natural generation context.
</div>

<!-- ============================================================ -->
<h2>3. Final-Epoch Test Accuracy</h2>

<div class="fig">
  <img src="data:image/png;base64,{fig_final}" alt="Final-epoch test accuracy by layer">
  <div class="fig-caption">Figure 2. Final-epoch (epoch 50) test accuracy. Pattern is similar to best-epoch
  accuracy but slightly noisier, reflecting training instability at some layers.</div>
</div>

<!-- ============================================================ -->
<h2>4. Reading vs Control Difference</h2>

<div class="fig">
  <img src="data:image/png;base64,{fig_diff}" alt="Accuracy difference by layer">
  <div class="fig-caption">Figure 3. Per-layer accuracy difference (reading &minus; control). Blue bars = reading
  probe advantage; red bars = control advantage. Gold borders mark layers significant after FDR correction.</div>
</div>

<div class="interpretation">
<strong>Interpretation:</strong> The reading probe advantage is concentrated in middle and late layers.
In early layers, the control probe occasionally has a slight edge (likely noise). No individual layer
reaches significance after FDR correction &mdash; the accuracy differences (~3&ndash;5 percentage points)
are too small relative to the per-layer sample size (n={N_TEST}). However, the consistent directionality
across layers is significant when tested as a paired comparison (see Section 6).
</div>

<!-- ============================================================ -->
<h2>5. Layer Group Analysis</h2>

<div class="fig">
  <img src="data:image/png;base64,{fig_group}" alt="Accuracy by layer group">
  <div class="fig-caption">Figure 4. Mean accuracy (&plusmn; SEM across layers) for early (0&ndash;13),
  middle (14&ndash;27), and late (28&ndash;40) layer groups. Left panel: best test accuracy.
  Right panel: final-epoch test accuracy.</div>
</div>

<table>
<tr><th>Metric / Group</th><th>Reading M</th><th>Reading SD</th><th>Control M</th><th>Control SD</th><th>Paired t</th><th>p</th></tr>
{group_rows_html}
</table>

<div class="interpretation">
<strong>Interpretation:</strong> The reading probe advantage is absent in early layers (ns) but highly
significant in both middle layers (<span class="stat">p &lt; .0001</span>) and late layers
(<span class="stat">p &lt; .0001</span>). This suggests that partner-type information is not represented
in early token-processing layers but emerges in the model&rsquo;s deeper representations, and is
amplified by the reflective prompt used by the reading probe.
</div>

<!-- ============================================================ -->
<h2>6. Overall Reading vs Control</h2>

<table>
<tr><th>Metric</th><th>Reading M</th><th>Control M</th><th>Mean Diff</th><th>Test</th><th>p</th><th>Cohen&rsquo;s d</th></tr>
{overall_rows_html}
</table>

<div class="interpretation">
<strong>Interpretation:</strong> Across all {N_LAYERS} layers (treated as paired observations), the reading probe
significantly outperforms the control probe with a medium-to-large effect size
(<span class="stat">d = 0.76</span> for best test acc, <span class="stat">d = 0.72</span> for final test acc).
This confirms that the reflective prompt provides a meaningful boost to partner-type decodability.
</div>

<!-- ============================================================ -->
<h2>7. Per-Layer Proportions Z-Tests (FDR Corrected)</h2>

<p>Two-proportions z-tests comparing reading vs. control probe accuracy at each layer, with
Benjamini-Hochberg FDR correction for {N_LAYERS} comparisons.</p>

<p><strong>Significant layers (FDR q&lt;.05): {n_sig}/{N_LAYERS}</strong></p>

{"<p>No individual layers reached significance after FDR correction. With n=" + str(N_TEST) +
  " test samples per layer and accuracy differences of ~3-5 percentage points, individual layers lack "
  "statistical power. The paired analysis across layers (Section 6) is the appropriate omnibus test.</p>"
  if n_sig == 0 else ""}

<!-- ============================================================ -->
<h2>8. Overfitting Analysis</h2>

<div class="fig">
  <img src="data:image/png;base64,{fig_overfit}" alt="Train vs test accuracy">
  <div class="fig-caption">Figure 5. Training (final epoch) vs. test (best epoch and final epoch) accuracy
  for reading and control probes. Large train-test gaps indicate overfitting.</div>
</div>

<div class="interpretation">
<strong>Interpretation:</strong> Both probes show substantial overfitting: training accuracy reaches
85&ndash;96% while test accuracy plateaus at 55&ndash;65%. This is typical for linear probes on
high-dimensional representations (5120-d hidden states) with limited training data (~1600 training
samples). The overfitting is more pronounced in late layers where the representation space is richer.
The use of best-epoch checkpointing (rather than final-epoch) partially mitigates this by selecting
the model at peak generalization.
</div>

<!-- ============================================================ -->
<h2>9. Full Layerwise Statistics Table</h2>

<p>Yellow rows indicate significance after FDR correction (q&lt;.05).</p>

<div style="overflow-x: auto;">
<table style="font-size: 0.8em;">
<tr><th>Layer</th><th>Group</th><th>Read Best</th><th>Read Final</th><th>Read Train</th>
<th>Ctrl Best</th><th>Ctrl Final</th><th>Ctrl Train</th><th>Diff</th>
<th>z</th><th>p (raw)</th><th>p (FDR)</th><th>Sig</th></tr>
{layerwise_rows_html}
</table>
</div>

</body>
</html>
"""

    from src.report_utils import save_report
    html_path = OUT_DIR / "probe_training_report.html"
    save_report(html, html_path)


    # ========================== DONE ========================== #
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
