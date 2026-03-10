"""
Generate figures from RSA and behavioral results.
===================================================
Loads rsa_results.json and behavioral_results.json.

1. Layer profile plot (layer_profile_rsa.png)
2. RDM heatmaps at peak layer (rdm_heatmaps.png)
3. Cross-topology consistency scatter (cross_topology_consistency.png)
4. Behavioral accuracy bar chart (behavioral_accuracy.png)
"""

import os, sys, json, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def plot_layer_profile(rsa_results: dict, out_path: str):
    """X=layer, Y=mean RSA. Three lines with SEM error bars and significance markers."""
    per_layer = rsa_results["per_layer"]
    layers = sorted([int(k) for k in per_layer.keys()])

    mean_e = [per_layer[str(l)]["mean_r_epistemic"] for l in layers]
    mean_c = [per_layer[str(l)]["mean_r_communication"] for l in layers]
    mean_p = [per_layer[str(l)]["mean_r_position"] for l in layers]
    sem_e = [per_layer[str(l)]["sem_r_epistemic"] for l in layers]
    sem_c = [per_layer[str(l)]["sem_r_communication"] for l in layers]
    sem_p = [per_layer[str(l)]["sem_r_position"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(layers, mean_e, yerr=sem_e, color="tab:blue", linewidth=2,
                label="Epistemic", capsize=2, capthick=1)
    ax.errorbar(layers, mean_c, yerr=sem_c, color="tab:orange", linewidth=2,
                label="Communication", capsize=2, capthick=1)
    ax.errorbar(layers, mean_p, yerr=sem_p, color="gray", linewidth=2,
                label="Position", capsize=2, capthick=1)

    # Significance markers where epistemic > communication (FDR-corrected)
    sig_layers = [l for l in layers
                  if per_layer[str(l)].get("p_epist_vs_comm_fdr", 1.0) < config.ALPHA]
    if sig_layers:
        y_max = max(max(mean_e), max(mean_c), max(mean_p))
        marker_y = y_max + 0.02
        ax.scatter(sig_layers, [marker_y] * len(sig_layers),
                   marker="*", color="tab:blue", s=80, zorder=5,
                   label=f"Epistemic > Comm (FDR p < {config.ALPHA})")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Spearman r")
    ax.set_title("RSA: Model RDM vs. Candidate RDMs Across Layers")
    ax.legend(loc="upper left")
    ax.set_xlim(-0.5, max(layers) + 0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_rdm_heatmaps(rsa_results: dict, out_path: str):
    """For peak layer, show average model RDM vs ground-truth epistemic RDM for override conditions."""
    peak_layer = rsa_results["peak_layer"]

    # Load the full RDM data to compute average model RDMs per topology
    rdm_path = os.path.join(config.RDMS_DIR, "model_rdms.pkl")
    with open(rdm_path, "rb") as f:
        all_rdms = pickle.load(f)

    topologies = ["chain", "fork", "diamond"]
    n_topo = len(topologies)

    fig, axes = plt.subplots(n_topo, 2, figsize=(8, 3 * n_topo))
    if n_topo == 1:
        axes = axes.reshape(1, -1)

    for row, topo in enumerate(topologies):
        # Collect override narratives for this topology
        override_nids = [nid for nid, data in all_rdms.items()
                         if data["topology"] == topo and "no_override" not in data["condition"]]

        if not override_nids:
            continue

        # Average model RDM at peak layer (upper triangle, 6 values)
        model_uts = np.array([all_rdms[nid]["model_rdm"][peak_layer] for nid in override_nids])
        avg_model_ut = np.mean(model_uts, axis=0)

        # Average epistemic RDM
        epist_uts = np.array([all_rdms[nid]["epistemic_rdm"] for nid in override_nids])
        avg_epist_ut = np.mean(epist_uts, axis=0)

        # Reconstruct 4x4 matrices for plotting
        for col, (ut, title) in enumerate([
            (avg_model_ut, f"{topo.capitalize()} - Model (Layer {peak_layer})"),
            (avg_epist_ut, f"{topo.capitalize()} - Epistemic (ground truth)"),
        ]):
            mat = np.zeros((4, 4))
            idx = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    mat[i, j] = ut[idx]
                    mat[j, i] = ut[idx]
                    idx += 1

            ax = axes[row, col]
            im = ax.imshow(mat, cmap="viridis", vmin=0, aspect="equal")
            ax.set_title(title, fontsize=11)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["A", "B", "C", "D"])
            ax.set_yticklabels(["A", "B", "C", "D"])

            # Add text annotations
            for i in range(4):
                for j in range(4):
                    color = "white" if mat[i, j] > np.max(mat) * 0.5 else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=9, color=color)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cross_topology(rsa_results: dict, out_path: str):
    """Scatter of model RDM correlations from matched cross-topology conditions."""
    cross = rsa_results.get("cross_topology", {})
    if not cross:
        print("  Skipping cross-topology plot: no data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = {"C_tells_D": "tab:blue", "D_only": "tab:red"}
    markers = {"C_tells_D": "o", "D_only": "s"}

    for pair_name, data in cross.items():
        correlations = data["correlations"]
        n = len(correlations)
        x = np.arange(n)
        ax.scatter(x, correlations, color=colors.get(pair_name, "gray"),
                   marker=markers.get(pair_name, "o"), s=60, alpha=0.7,
                   label=f"{pair_name} ({data['description']})")
        ax.axhline(data["mean_r"], color=colors.get(pair_name, "gray"),
                   linestyle="--", alpha=0.5)

    ax.set_xlabel("Matched Pair Index")
    ax.set_ylabel("Spearman r (Model RDM Similarity)")
    ax.set_title("Cross-Topology Consistency:\nSame Epistemic Geometry, Different Communication")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_behavioral_accuracy(out_path: str):
    """Bar chart of behavioral accuracy by condition type."""
    behav_path = os.path.join(config.BEHAVIORAL_DIR, "behavioral_results.json")
    if not os.path.exists(behav_path):
        print("  Skipping behavioral accuracy plot: no behavioral_results.json")
        return

    with open(behav_path) as f:
        behav = json.load(f)

    # Expected structure: by_condition -> {condition: {updated_acc, outdated_acc, overall_acc, n}}
    by_cond = behav.get("by_condition", {})
    if not by_cond:
        print("  Skipping behavioral accuracy plot: no by_condition data")
        return

    conditions = sorted(by_cond.keys())
    updated_accs = []
    outdated_accs = []
    overall_accs = []
    labels = []

    for cond in conditions:
        labels.append(cond.replace("_", "\n"))
        updated_accs.append(by_cond[cond].get("updated_acc", 0))
        outdated_accs.append(by_cond[cond].get("outdated_acc", 0))
        overall_accs.append(by_cond[cond].get("overall_acc", 0))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 6))

    ax.bar(x - width, updated_accs, width, label="Updated agents", color="tab:green", alpha=0.8)
    ax.bar(x, outdated_accs, width, label="Outdated agents", color="tab:red", alpha=0.8)
    ax.bar(x + width, overall_accs, width, label="Overall", color="tab:blue", alpha=0.8)

    ax.set_xlabel("Condition")
    ax.set_ylabel("Accuracy")
    ax.set_title("Behavioral Validation: Belief Attribution Accuracy by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    rsa_path = os.path.join(config.RSA_DIR, "rsa_results.json")
    with open(rsa_path) as f:
        rsa_results = json.load(f)

    print(f"Loaded RSA results (peak layer: {rsa_results['peak_layer']}, "
          f"peak r: {rsa_results['peak_r_epistemic']:.4f})")

    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    print("\nGenerating figures:")

    plot_layer_profile(
        rsa_results,
        os.path.join(config.FIGURES_DIR, "layer_profile_rsa.png")
    )

    plot_rdm_heatmaps(
        rsa_results,
        os.path.join(config.FIGURES_DIR, "rdm_heatmaps.png")
    )

    plot_cross_topology(
        rsa_results,
        os.path.join(config.FIGURES_DIR, "cross_topology_consistency.png")
    )

    plot_behavioral_accuracy(
        os.path.join(config.FIGURES_DIR, "behavioral_accuracy.png")
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
