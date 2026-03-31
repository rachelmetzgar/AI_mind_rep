#!/usr/bin/env python3
"""
Generate bar chart of varimax-rotated capacity loadings sorted by loading value,
colored by original Gray et al. factor (Experience vs Agency).

Reads: data_dir("gray_entities", "behavioral", condition)/pairwise_pca_results.npz
Writes: results_dir("gray_entities", "behavioral", condition)/figures/loadings_sorted_by_value.png + .svg

Usage:
    python behavior/make_loadings_bar_chart.py --model llama2_13b_base
"""
import os, sys, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config, set_model, add_model_argument, data_dir, results_dir
from entities.gray_entities import CAPACITY_PROMPTS


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    add_model_argument(parser)
    parser.add_argument("--condition", default="with_self",
                        choices=["with_self", "without_self"])
    args = parser.parse_args()

    set_model(args.model)
    ddir = data_dir("gray_entities", "behavioral", args.condition)
    rdir = results_dir("gray_entities", "behavioral", args.condition)

    pca = np.load(os.path.join(str(ddir), "pairwise_pca_results.npz"))
    rotated = pca["rotated_loadings"]      # (n_caps, n_factors)
    eigenvalues = pca["eigenvalues"]
    explained = pca["explained_var_ratio"]
    capacity_keys = list(pca["capacity_keys"])

    # Map each capacity to its Gray factor
    gray_factor = {}
    for cap in capacity_keys:
        if cap in CAPACITY_PROMPTS:
            gray_factor[cap] = CAPACITY_PROMPTS[cap][1]  # "E" or "A"
        else:
            gray_factor[cap] = "?"

    COLORS = {"E": "#0d9488", "A": "#d97706", "?": "#94a3b8"}
    LABELS = {"E": "Experience", "A": "Agency", "?": "Unknown"}

    n_factors = min(2, rotated.shape[1])

    for fi in range(n_factors):
        vals = rotated[:, fi]
        sort_idx = np.argsort(vals)  # ascending → lowest at bottom
        sorted_vals = vals[sort_idx]
        sorted_caps = [capacity_keys[i] for i in sort_idx]
        sorted_colors = [COLORS[gray_factor[c]] for c in sorted_caps]

        fig, ax = plt.subplots(figsize=(8, max(5, len(capacity_keys) * 0.35)))
        y = np.arange(len(sorted_caps))
        ax.barh(y, sorted_vals, color=sorted_colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_caps, fontsize=10)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=0.4, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.4, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Varimax-Rotated Loading", fontsize=11)
        ax.set_title(
            f"Factor {fi+1} Loadings (eigenvalue={eigenvalues[fi]:.2f}, "
            f"{explained[fi]*100:.1f}% var) — {config.MODEL_LABEL}",
            fontsize=12
        )

        # Legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=COLORS["E"], label="Experience (Gray)"),
            Patch(facecolor=COLORS["A"], label="Agency (Gray)"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=10)

        plt.tight_layout()
        fig_dir = os.path.join(str(rdir), "figures")
        os.makedirs(fig_dir, exist_ok=True)
        for ext in ("png", "svg"):
            out = os.path.join(fig_dir, f"loadings_F{fi+1}_sorted.{ext}")
            fig.savefig(out, dpi=200, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
