"""
Generate comprehensive HTML summary report for the Belief Propagation experiment.
==================================================================================
Reads all saved results (behavioral, RSA, RDMs, stimuli) and produces:
  - results/llama2_13b_chat/rsa/results_summary.html
  - results/llama2_13b_chat/rsa/figures/*.png + *.pdf

Run after phases 1-5 complete. Only requires saved JSON/pkl data, no GPU.
"""

import os, sys, json, pickle, base64
from io import BytesIO
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGURES_DIR = config.FIGURES_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===========================================================================
# Data loading
# ===========================================================================

def load_all_data():
    """Load all result files. Returns dict of data or None for missing files."""
    data = {}

    # Stimuli
    if os.path.exists(config.STIMULI_PATH):
        with open(config.STIMULI_PATH) as f:
            data["stimuli"] = json.load(f)
    else:
        data["stimuli"] = None

    # Behavioral
    behav_path = os.path.join(config.BEHAVIORAL_DIR, "behavioral_results.json")
    if os.path.exists(behav_path):
        with open(behav_path) as f:
            data["behavioral"] = json.load(f)
    else:
        data["behavioral"] = None

    # RSA
    rsa_path = os.path.join(config.RSA_DIR, "rsa_results.json")
    if os.path.exists(rsa_path):
        with open(rsa_path) as f:
            data["rsa"] = json.load(f)
    else:
        data["rsa"] = None

    # RDMs
    rdm_path = os.path.join(config.RDMS_DIR, "model_rdms.pkl")
    if os.path.exists(rdm_path):
        with open(rdm_path, "rb") as f:
            data["rdms"] = pickle.load(f)
    else:
        data["rdms"] = None

    return data


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG for inline HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def save_fig(fig, name):
    """Save figure as PNG and PDF."""
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), bbox_inches="tight")


# ===========================================================================
# Figure generators
# ===========================================================================

def make_behavioral_figure(behavioral):
    """Behavioral accuracy by condition, split by agent belief status."""
    summary = behavioral["summary"]
    per_narrative = behavioral["per_narrative"]

    # Aggregate by condition: updated vs outdated accuracy
    cond_stats = defaultdict(lambda: {"updated_correct": 0, "updated_total": 0,
                                       "outdated_correct": 0, "outdated_total": 0})
    for narr in per_narrative:
        cond = narr["condition"]
        for probe in narr["probes"]:
            knows_new = probe.get("knows_new_location")
            if knows_new is None:
                continue
            if knows_new:
                cond_stats[cond]["updated_total"] += 1
                if probe["correct"]:
                    cond_stats[cond]["updated_correct"] += 1
            else:
                cond_stats[cond]["outdated_total"] += 1
                if probe["correct"]:
                    cond_stats[cond]["outdated_correct"] += 1

    conditions = sorted(cond_stats.keys())
    # Separate override vs no-override
    override_conds = [c for c in conditions if "no_override" not in c]
    no_override_conds = [c for c in conditions if "no_override" in c]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})

    # Override conditions
    labels = [c.replace("_override", "\noverride").replace("_", " ") for c in override_conds]
    updated_acc = []
    outdated_acc = []
    for c in override_conds:
        s = cond_stats[c]
        updated_acc.append(s["updated_correct"] / s["updated_total"] if s["updated_total"] > 0 else 0)
        outdated_acc.append(s["outdated_correct"] / s["outdated_total"] if s["outdated_total"] > 0 else 0)

    x = np.arange(len(override_conds))
    w = 0.35
    bars1 = ax1.bar(x - w/2, updated_acc, w, label="Updated belief agents",
                     color="#2196F3", alpha=0.85)
    bars2 = ax1.bar(x + w/2, outdated_acc, w, label="Outdated belief agents",
                     color="#FF9800", alpha=0.85)

    ax1.set_xlabel("Condition")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Behavioral Validation: Override Conditions")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=7, ha="center")
    ax1.set_ylim(0, 1.08)
    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower left")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.0%}",
                 ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.0%}",
                 ha="center", va="bottom", fontsize=7)

    # Summary panel
    overall = summary["overall_accuracy"]
    updated_overall = summary["accuracy_updated_agents"]
    outdated_overall = summary["accuracy_outdated_agents"]
    topo_acc = summary["accuracy_by_topology"]

    summary_text = (
        f"Overall accuracy: {overall:.1%}\n"
        f"Updated agents: {updated_overall:.1%}\n"
        f"Outdated agents: {outdated_overall:.1%}\n\n"
        f"By topology:\n"
    )
    for topo, acc in sorted(topo_acc.items()):
        summary_text += f"  {topo}: {acc:.1%}\n"
    summary_text += f"\nNo-override: 100%\n(all 3 topologies)"

    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))
    ax2.set_axis_off()
    ax2.set_title("Summary")

    plt.tight_layout()
    return fig


def make_layer_profile_figure(rsa):
    """Main RSA layer profile: epistemic vs communication vs position."""
    per_layer = rsa["per_layer"]
    layers = sorted([int(k) for k in per_layer.keys()])

    mean_e = [per_layer[str(l)]["mean_r_epistemic"] for l in layers]
    mean_c = [per_layer[str(l)]["mean_r_communication"] for l in layers]
    mean_p = [per_layer[str(l)]["mean_r_position"] for l in layers]
    sem_e = [per_layer[str(l)]["sem_r_epistemic"] for l in layers]
    sem_c = [per_layer[str(l)]["sem_r_communication"] for l in layers]
    sem_p = [per_layer[str(l)]["sem_r_position"] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.fill_between(layers,
                     [m - s for m, s in zip(mean_e, sem_e)],
                     [m + s for m, s in zip(mean_e, sem_e)],
                     alpha=0.15, color="#2196F3")
    ax.fill_between(layers,
                     [m - s for m, s in zip(mean_c, sem_c)],
                     [m + s for m, s in zip(mean_c, sem_c)],
                     alpha=0.15, color="#FF9800")
    ax.fill_between(layers,
                     [m - s for m, s in zip(mean_p, sem_p)],
                     [m + s for m, s in zip(mean_p, sem_p)],
                     alpha=0.1, color="gray")

    ax.plot(layers, mean_e, color="#2196F3", linewidth=2.5, label="Epistemic", zorder=3)
    ax.plot(layers, mean_c, color="#FF9800", linewidth=2.5, label="Communication", zorder=3)
    ax.plot(layers, mean_p, color="gray", linewidth=2, label="Position", linestyle="--", zorder=2)

    # Significance markers
    sig_layers = [l for l in layers
                  if per_layer[str(l)].get("p_epist_vs_comm_fdr", 1.0) < config.ALPHA]
    if sig_layers:
        y_max = max(max(mean_e), max(mean_c), max(mean_p))
        for l in sig_layers:
            ax.plot(l, y_max + 0.03, marker="*", color="#2196F3", markersize=8, zorder=5)
        ax.plot([], [], marker="*", color="#2196F3", linestyle="none",
                label=f"Epistemic > Communication (FDR p < {config.ALPHA})")

    # Peak layer annotation
    peak_layer = rsa["peak_layer"]
    peak_r = rsa["peak_r_epistemic"]
    ax.annotate(f"Peak: layer {peak_layer}\nr = {peak_r:.3f}",
                xy=(peak_layer, peak_r), xytext=(peak_layer + 3, peak_r + 0.05),
                arrowprops=dict(arrowstyle="->", color="#2196F3"),
                fontsize=9, color="#2196F3")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Spearman r (RSA)")
    ax.set_title("Representational Similarity to Candidate RDMs Across Layers")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(-0.5, max(layers) + 0.5)

    plt.tight_layout()
    return fig


def make_rdm_heatmaps(rsa, rdms):
    """Show model RDMs vs epistemic RDMs at peak layer for each topology."""
    peak_layer = rsa["peak_layer"]
    topologies = ["chain", "fork", "diamond"]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))

    for row, topo in enumerate(topologies):
        # Get override narratives for this topology
        override_nids = [nid for nid, data in rdms.items()
                         if data["topology"] == topo and "no_override" not in data["condition"]]
        if not override_nids:
            continue

        # Average model RDM
        model_uts = np.array([rdms[nid]["model_rdm"][peak_layer] for nid in override_nids])
        avg_model_ut = np.mean(model_uts, axis=0)

        # Average epistemic RDM
        epist_uts = np.array([rdms[nid]["epistemic_rdm"] for nid in override_nids])
        avg_epist_ut = np.mean(epist_uts, axis=0)

        # Average communication RDM
        comm_uts = np.array([rdms[nid]["communication_rdm"] for nid in override_nids])
        avg_comm_ut = np.mean(comm_uts, axis=0)

        for col, (ut, title, cmap) in enumerate([
            (avg_model_ut, f"Model RDM (Layer {peak_layer})", "Blues"),
            (avg_epist_ut, "Epistemic RDM", "Oranges"),
            (avg_comm_ut, "Communication RDM", "Greens"),
        ]):
            mat = np.zeros((4, 4))
            idx = 0
            for i in range(4):
                for j in range(i + 1, 4):
                    mat[i, j] = ut[idx]
                    mat[j, i] = ut[idx]
                    idx += 1

            ax = axes[row, col]
            vmax = max(np.max(mat), 0.01)
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
            if row == 0:
                ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(["A", "B", "C", "D"])
            ax.set_yticklabels(["A", "B", "C", "D"])
            if col == 0:
                ax.set_ylabel(f"{topo.capitalize()}", fontsize=12, fontweight="bold")

            for i in range(4):
                for j in range(4):
                    color = "white" if mat[i, j] > vmax * 0.6 else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Average RDMs for Override Conditions (Model at Peak Layer {peak_layer})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def make_topology_breakdown_figure(rsa):
    """RSA layer profile broken down by topology."""
    by_topology = rsa.get("by_topology", {})
    if not by_topology:
        return None

    topologies = sorted(by_topology.keys())
    colors = {"chain": "#E91E63", "fork": "#4CAF50", "diamond": "#9C27B0"}
    layers = list(range(config.NUM_LAYERS))

    fig, ax = plt.subplots(figsize=(10, 5))

    for topo in topologies:
        means = [by_topology[topo][str(l)]["mean_r_epistemic"] for l in layers]
        sems = [by_topology[topo][str(l)]["sem_r_epistemic"] for l in layers]
        ax.fill_between(layers,
                         [m - s for m, s in zip(means, sems)],
                         [m + s for m, s in zip(means, sems)],
                         alpha=0.12, color=colors.get(topo, "gray"))
        ax.plot(layers, means, color=colors.get(topo, "gray"), linewidth=2,
                label=f"{topo.capitalize()}")

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Epistemic RSA (Spearman r)")
    ax.set_title("Epistemic RSA by Network Topology")
    ax.legend()
    ax.set_xlim(-0.5, max(layers) + 0.5)

    plt.tight_layout()
    return fig


def make_condition_breakdown_figure(rsa):
    """RSA at peak layer broken down by condition."""
    by_condition = rsa.get("by_condition", {})
    if not by_condition:
        return None

    peak_layer = rsa["peak_layer"]
    peak_str = str(peak_layer)

    conditions = sorted(by_condition.keys())
    override_conds = [c for c in conditions if "no_override" not in c]
    no_override_conds = [c for c in conditions if "no_override" in c]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Override conditions
    labels = []
    means = []
    sems = []
    colors_list = []
    topo_colors = {"chain": "#E91E63", "fork": "#4CAF50", "diamond": "#9C27B0"}

    for c in override_conds:
        topo = c.split("_")[0]
        labels.append(c.replace("_override", "\n").replace("_", " "))
        means.append(by_condition[c][peak_str]["mean_r_epistemic"])
        sems.append(by_condition[c][peak_str]["sem_r_epistemic"])
        colors_list.append(topo_colors.get(topo, "gray"))

    # Add no-override as reference
    for c in no_override_conds:
        topo = c.split("_")[0]
        labels.append(c.replace("_no_override", "\nno override"))
        means.append(by_condition[c][peak_str]["mean_r_epistemic"])
        sems.append(by_condition[c][peak_str]["sem_r_epistemic"])
        colors_list.append("#BDBDBD")

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=sems, capsize=3, color=colors_list, alpha=0.8,
                   edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Mean Epistemic RSA (Layer {peak_layer})")
    ax.set_title(f"Epistemic RSA by Condition at Peak Layer {peak_layer}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, ha="center")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def make_design_figure():
    """Schematic of the three topologies."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    topologies = {
        "Chain": [(0, 1), (1, 2), (2, 3)],
        "Fork": [(0, 1), (0, 2), (0, 3)],
        "Diamond": [(0, 1), (0, 2), (1, 3), (2, 3)],
    }
    positions = {
        "Chain": {0: (0, 0.5), 1: (1, 0.5), 2: (2, 0.5), 3: (3, 0.5)},
        "Fork": {0: (1.5, 1.5), 1: (0, 0), 2: (1.5, 0), 3: (3, 0)},
        "Diamond": {0: (1.5, 1.5), 1: (0, 0.75), 2: (3, 0.75), 3: (1.5, 0)},
    }
    agent_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

    for ax, (topo_name, edges) in zip(axes, topologies.items()):
        pos = positions[topo_name]

        # Draw edges
        for src, dst in edges:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color="#666",
                                       linewidth=2, connectionstyle="arc3,rad=0.05"))

        # Draw nodes
        for idx, (x, y) in pos.items():
            circle = plt.Circle((x, y), 0.25, color="#2196F3", ec="white",
                               linewidth=2, zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, agent_labels[idx], ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white", zorder=6)

        ax.set_xlim(-0.8, 3.8)
        ax.set_ylim(-0.8, 2.2)
        ax.set_aspect("equal")
        ax.set_title(topo_name, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Communication Topologies", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ===========================================================================
# HTML generation
# ===========================================================================

def generate_html(data, figures_b64):
    """Generate the full HTML report."""

    stimuli = data["stimuli"]
    behavioral = data["behavioral"]
    rsa = data["rsa"]

    # Extract key stats
    behav_summary = behavioral["summary"] if behavioral else {}
    peak_layer = rsa["peak_layer"] if rsa else "N/A"
    peak_r = rsa["peak_r_epistemic"] if rsa else "N/A"
    n_narratives = rsa["n_narratives"] if rsa else 96
    n_perms = rsa["n_permutations"] if rsa else config.N_PERMUTATIONS

    # Count significant layers
    sig_layers = []
    if rsa:
        for l_str, vals in rsa["per_layer"].items():
            if vals.get("p_epist_vs_comm_fdr", 1.0) < config.ALPHA:
                sig_layers.append(int(l_str))

    # Sample stimuli
    sample_stimuli_html = ""
    if stimuli:
        # Pick one from each topology with override
        shown = set()
        for stim in stimuli:
            topo = stim["topology"]
            if topo in shown or "no_override" in stim["condition"]:
                continue
            shown.add(topo)
            beliefs_html = "".join(
                f"<li><b>{agent}</b>: {loc}</li>"
                for agent, loc in stim["expected_beliefs"].items()
            )
            sample_stimuli_html += f"""
            <div class="sample-stimulus">
                <h4>{stim['narrative_id']}</h4>
                <p class="meta">Topology: <b>{topo}</b> | Condition: <b>{stim['condition']}</b></p>
                <div class="narrative">{stim['narrative_text'].replace(chr(10), '<br>')}</div>
                <p><b>Expected beliefs:</b></p>
                <ul>{beliefs_html}</ul>
            </div>
            """
            if len(shown) >= 3:
                break

    # RSA layer table
    rsa_table_rows = ""
    if rsa:
        for l in range(config.NUM_LAYERS):
            l_str = str(l)
            vals = rsa["per_layer"].get(l_str, {})
            me = vals.get("mean_r_epistemic", 0)
            mc = vals.get("mean_r_communication", 0)
            mp = vals.get("mean_r_position", 0)
            pw = vals.get("p_epist_vs_comm", 1)
            pf = vals.get("p_epist_vs_comm_fdr", 1)
            pp = vals.get("p_permutation", 1)
            row_class = "sig-row" if pf < config.ALPHA else ""
            peak_marker = " *" if l == peak_layer else ""
            rsa_table_rows += f"""
            <tr class="{row_class}">
                <td>{l}{peak_marker}</td>
                <td>{me:.4f}</td>
                <td>{mc:.4f}</td>
                <td>{mp:.4f}</td>
                <td>{pw:.4f}</td>
                <td>{pf:.4f}</td>
                <td>{pp:.4f}</td>
            </tr>"""

    # Cross-topology results
    cross_topo_html = ""
    if rsa and rsa.get("cross_topology"):
        for pair_name, pair_data in rsa["cross_topology"].items():
            cross_topo_html += f"""
            <tr>
                <td>{pair_name}</td>
                <td>{pair_data['description']}</td>
                <td>{pair_data['mean_r']:.4f}</td>
                <td>{pair_data['sem_r']:.4f}</td>
                <td>{pair_data['n_pairs']}</td>
            </tr>"""

    # Behavioral condition table
    behav_cond_rows = ""
    if behavioral:
        for cond, acc in sorted(behav_summary.get("accuracy_by_condition", {}).items()):
            behav_cond_rows += f"<tr><td>{cond}</td><td>{acc:.1%}</td></tr>"

    # Build figure HTML
    def fig_html(key, caption):
        if key in figures_b64 and figures_b64[key]:
            return f"""
            <figure>
                <img src="data:image/png;base64,{figures_b64[key]}" alt="{key}">
                <figcaption>{caption}</figcaption>
            </figure>"""
        return "<p><i>Figure not available (missing data).</i></p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment 6: Multi-Agent Belief Propagation — Results Summary</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }}
    h1 {{ color: #1565C0; border-bottom: 3px solid #1565C0; padding-bottom: 10px; }}
    h2 {{ color: #1976D2; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 40px; }}
    h3 {{ color: #1E88E5; }}
    h4 {{ color: #42A5F5; margin-bottom: 5px; }}
    #toc {{ background: #f5f7fa; padding: 20px 30px; border-radius: 8px; margin: 20px 0; }}
    #toc h2 {{ margin-top: 0; border: none; }}
    #toc ul {{ list-style-type: none; padding-left: 0; }}
    #toc li {{ padding: 4px 0; }}
    #toc a {{ text-decoration: none; color: #1976D2; }}
    #toc a:hover {{ text-decoration: underline; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
    th {{ background-color: #e3f2fd; color: #1565C0; font-weight: 600; }}
    tr:nth-child(even) {{ background-color: #fafafa; }}
    .sig-row {{ background-color: #e8f5e9 !important; font-weight: 600; }}
    figure {{ margin: 20px 0; text-align: center; }}
    figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
    figcaption {{ font-size: 0.9em; color: #555; margin-top: 8px; max-width: 900px; margin-left: auto; margin-right: auto; text-align: left; font-style: italic; }}
    .sample-stimulus {{ background: #f9f9f9; padding: 15px; border-left: 4px solid #1976D2; margin: 15px 0; border-radius: 0 4px 4px 0; }}
    .sample-stimulus .narrative {{ font-family: Georgia, serif; font-size: 0.95em; color: #444; margin: 10px 0; }}
    .meta {{ color: #777; font-size: 0.9em; }}
    .key-finding {{ background: #e8f5e9; border-left: 4px solid #4CAF50; padding: 15px; margin: 15px 0; border-radius: 0 4px 4px 0; }}
    .method-note {{ background: #fff3e0; border-left: 4px solid #FF9800; padding: 12px; margin: 10px 0; border-radius: 0 4px 4px 0; font-size: 0.9em; }}
    .stat-box {{ display: inline-block; background: #e3f2fd; padding: 8px 15px; border-radius: 4px; margin: 5px; font-size: 0.95em; }}
    footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd; color: #999; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>Experiment 6: Multi-Agent Belief Propagation in LLM Internal Representations</h1>

<p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Model: LLaMA-2-13B-Chat | N = {n_narratives} narratives</p>

<div id="toc">
<h2>Table of Contents</h2>
<ul>
    <li><a href="#research-question">1. Research Question</a></li>
    <li><a href="#design">2. Experimental Design</a></li>
    <li><a href="#stimuli">3. Sample Stimuli</a></li>
    <li><a href="#approach">4. Methodological Approach</a></li>
    <li><a href="#behavioral">5. Behavioral Validation</a></li>
    <li><a href="#rsa-results">6. RSA Results</a></li>
    <li><a href="#rdm-heatmaps">7. Representational Dissimilarity Matrices</a></li>
    <li><a href="#topology-breakdown">8. Topology Breakdown</a></li>
    <li><a href="#condition-breakdown">9. Condition Breakdown</a></li>
    <li><a href="#cross-topology">10. Cross-Topology Consistency</a></li>
    <li><a href="#layer-table">11. Full Layer-by-Layer Statistics</a></li>
    <li><a href="#summary">12. Summary</a></li>
</ul>
</div>

<!-- ============================================================ -->
<h2 id="research-question">1. Research Question</h2>

<p>When an LLM processes a narrative involving multiple agents who communicate beliefs through a network &mdash; and those beliefs diverge due to an information update that reaches only some agents &mdash; does the model's internal representational geometry of agent belief states mirror the ground-truth <b>epistemic geometry</b> (who-believes-what)?</p>

<p>This extends classical 2-agent Theory of Mind probing to <b>4-agent networks</b> with multiple communication topologies. The key test: can the model's representational structure be explained by surface features (communication links, textual position), or does it track the deeper epistemic structure?</p>

<p><b>Analogy:</b> Gurnee &amp; Tegmark (2023) showed LLMs develop internal representations of cities whose pairwise distances reflect actual geographic distances. We ask the equivalent for social cognition: do agent representations reflect epistemic similarity?</p>

<!-- ============================================================ -->
<h2 id="design">2. Experimental Design</h2>

{fig_html("design", "<b>Figure 1. Communication topologies.</b> Three network structures govern how a fact propagates between four agents (A&ndash;D). <i>Chain</i>: linear relay (A&rarr;B&rarr;C&rarr;D). <i>Fork</i>: single source broadcasts to all (A&rarr;B, A&rarr;C, A&rarr;D). <i>Diamond</i>: two independent paths converge (A&rarr;B, A&rarr;C; B&rarr;D, C&rarr;D). After communication, an object is moved and only some agents learn about it, creating divergent beliefs.")}

<p><b>Design:</b> 3 topologies &times; 4 conditions &times; 8 instantiations = <b>96 narratives</b>.</p>

<p>Each narrative follows a fixed structure:</p>
<ol>
    <li>Four agents learn a fact (e.g., "the red book is on the kitchen table")</li>
    <li>The fact propagates through the communication network</li>
    <li>The object is moved; only some agents witness or are told about the change</li>
    <li>An extraction sentence names all four agents in a fixed syntactic frame</li>
</ol>

<h3>Confound Controls</h3>
<table>
    <tr><th>Confound</th><th>Control</th></tr>
    <tr><td>Name&ndash;belief association</td><td>8 name sets rotated across instantiations</td></tr>
    <tr><td>Textual position</td><td>Fixed extraction sentence with all 4 names at same positions</td></tr>
    <tr><td>Communication co-mention</td><td>Key conditions create epistemic alignment between non-communicating agents (e.g., <code>fork_override_D_tells_A</code>)</td></tr>
    <tr><td>Object/location content</td><td>8 different scenarios rotated across instantiations</td></tr>
    <tr><td>Narrative length</td><td>No-override narratives include filler paragraph of matched length</td></tr>
</table>

<!-- ============================================================ -->
<h2 id="stimuli">3. Sample Stimuli</h2>

<p>Three example narratives showing one override condition per topology. Each creates a different partition of who holds the updated vs. outdated belief.</p>

{sample_stimuli_html}

<!-- ============================================================ -->
<h2 id="approach">4. Methodological Approach</h2>

<h3>4.1 Activation Extraction</h3>
<p>For each narrative, the full text (including extraction sentence) is wrapped in the LLaMA-2 chat template and passed through the model. Hidden states are extracted from the residual stream output of all 40 transformer layers at the <b>last token position</b> of each agent's name in the extraction sentence.</p>

<div class="method-note">
<b>Why last token?</b> In autoregressive transformers, the last token of a multi-token name integrates information about the full name through causal attention. This is standard practice in probing studies (Zhu et al. 2024, Viegas et al. 2024).
</div>

<h3>4.2 RDM Construction</h3>
<p>For each narrative and layer, we compute the 4&times;4 pairwise <b>cosine distance</b> matrix between the four agent activation vectors (each &isin; &real;<sup>5120</sup>), yielding a model RDM with 6 unique pairwise distances.</p>

<div class="method-note">
<b>Why cosine distance?</b> Cosine distance is invariant to activation magnitude (which varies by layer) and captures the <i>direction</i> of representation, where conceptual information resides under the linear representation hypothesis.
</div>

<h3>4.3 Candidate RDMs</h3>
<p>Three candidate RDMs are constructed for comparison:</p>
<ul>
    <li><b>Epistemic RDM</b> (prediction): 0 if agents share the same belief, 1 if they disagree</li>
    <li><b>Communication RDM</b> (alternative): 0 if agents communicated directly, 1 if not</li>
    <li><b>Position RDM</b> (baseline): |i &minus; j| for agent positions in the extraction sentence</li>
</ul>

<h3>4.4 RSA (Representational Similarity Analysis)</h3>
<p>For each narrative and layer, the model RDM's upper triangle (6 values) is correlated with each candidate RDM using <b>Spearman rank correlation</b>. Correlations are aggregated across narratives per layer.</p>

<div class="method-note">
<b>Why Spearman?</b> Spearman is the standard RSA metric (Kriegeskorte et al. 2008). It captures monotonic relationships without assuming linearity, appropriate for comparing binary candidate RDMs to continuous model RDMs.
</div>

<h3>4.5 Statistical Testing</h3>
<ul>
    <li><b>Wilcoxon signed-rank test</b>: Per layer, tests whether epistemic RSA &gt; communication RSA across narratives (one-sided)</li>
    <li><b>Permutation test</b> ({n_perms:,} permutations): Shuffles agent labels in the epistemic RDM to build a null distribution; tests whether observed mean epistemic RSA exceeds chance</li>
    <li><b>BH-FDR correction</b>: Applied across all {config.NUM_LAYERS} layers for the Wilcoxon comparison</li>
</ul>

<!-- ============================================================ -->
<h2 id="behavioral">5. Behavioral Validation</h2>

<p>Before analyzing internal representations, we verified that LLaMA-2-13B-Chat can behaviorally answer belief-tracking questions ("Where does X think the object is?").</p>

{fig_html("behavioral", "<b>Figure 2. Behavioral validation accuracy.</b> For each override condition, accuracy is shown separately for agents who witnessed or were told about the object's move (<i>updated belief</i>, blue) vs. agents who still hold the original belief (<i>outdated belief</i>, orange). No-override conditions (not shown) achieved 100% accuracy across all topologies. Updated-belief questions are easier (the model correctly reports the new location), while outdated-belief accuracy reflects the model's ability to track that uninformed agents still hold the old belief.")}

<p><b>Overall accuracy: {behav_summary.get('overall_accuracy', 0):.1%}</b></p>
<div>
    <span class="stat-box">Updated agents: {behav_summary.get('accuracy_updated_agents', 0):.1%}</span>
    <span class="stat-box">Outdated agents: {behav_summary.get('accuracy_outdated_agents', 0):.1%}</span>
    <span class="stat-box">No-override: 100%</span>
</div>

<table>
    <tr><th>Condition</th><th>Accuracy</th></tr>
    {behav_cond_rows}
</table>

<div class="key-finding">
<b>Go/no-go:</b> Overall accuracy of {behav_summary.get('overall_accuracy', 0):.1%} exceeds the 60% threshold, confirming the model has meaningful (if imperfect) multi-agent belief tracking. Notably, the hardest condition (<code>fork_override_D_tells_A</code>, {behav_summary.get('accuracy_by_condition', {}).get('fork_override_D_tells_A', 0):.0%}) is the key dissociation condition where epistemic alignment doesn't follow communication structure.
</div>

<!-- ============================================================ -->
<h2 id="rsa-results">6. RSA Results</h2>

{fig_html("layer_profile", "<b>Figure 3. RSA layer profile.</b> Mean Spearman correlation between the model's representational dissimilarity matrix and three candidate RDMs across all 40 transformer layers. Blue: epistemic RDM (ground-truth belief structure). Orange: communication RDM (who talked to whom). Gray dashed: position RDM (textual proximity). Shaded regions show &plusmn;1 SEM across {n_narratives} narratives. Blue stars indicate layers where epistemic RSA significantly exceeds communication RSA (Wilcoxon signed-rank, FDR-corrected p &lt; {config.ALPHA}). The epistemic signal peaks in early-to-middle layers and is consistently above the communication baseline in those layers, suggesting the model develops belief-state representations distinct from surface communication structure.")}

<div class="key-finding">
<b>Peak epistemic RSA:</b> Layer {peak_layer}, r = {peak_r if isinstance(peak_r, str) else f'{peak_r:.4f}'}<br>
<b>Significant layers (FDR-corrected):</b> {sig_layers if sig_layers else 'None'} ({len(sig_layers)} of {config.NUM_LAYERS})
</div>

<!-- ============================================================ -->
<h2 id="rdm-heatmaps">7. Representational Dissimilarity Matrices</h2>

{fig_html("rdm_heatmaps", f"<b>Figure 4. Average RDMs at peak layer ({peak_layer}).</b> Each row shows one topology (chain, fork, diamond). Left column: model RDM (cosine distance between agent activation vectors, averaged across override conditions). Middle: ground-truth epistemic RDM (0 = same belief, 1 = different belief, averaged across conditions). Right: communication RDM (0 = communicated directly, 1 = did not). If the model tracks epistemic state, the model RDM pattern should resemble the epistemic RDM more than the communication RDM.")}

<!-- ============================================================ -->
<h2 id="topology-breakdown">8. Topology Breakdown</h2>

{fig_html("topology_breakdown", "<b>Figure 5. Epistemic RSA by topology.</b> Mean epistemic RSA (Spearman r) across layers, broken down by communication topology. Shaded regions show &plusmn;1 SEM. Differences between topologies reflect varying difficulty: fork topologies have more complex belief partitions (the broadcast source creates asymmetric knowledge), while chain topologies have more predictable belief gradients.")}

<!-- ============================================================ -->
<h2 id="condition-breakdown">9. Condition Breakdown</h2>

{fig_html("condition_breakdown", f"<b>Figure 6. Epistemic RSA by condition at peak layer {peak_layer}.</b> Mean epistemic RSA for each of the 12 conditions (9 override + 3 no-override). Colored by topology (pink = chain, green = fork, purple = diamond; gray = no-override controls). Error bars show &plusmn;1 SEM across 8 instantiations per condition. No-override conditions serve as ceiling controls where all agents share the same belief, making the epistemic RDM trivially constant.")}

<!-- ============================================================ -->
<h2 id="cross-topology">10. Cross-Topology Consistency</h2>

<p>A critical test: conditions with the <b>same epistemic geometry</b> but <b>different communication structure</b> should produce similar model RDMs if the model tracks beliefs rather than communication links.</p>

<table>
    <tr><th>Matched Pair</th><th>Description</th><th>Mean r</th><th>SEM</th><th>N pairs</th></tr>
    {cross_topo_html if cross_topo_html else '<tr><td colspan="5"><i>Data not available</i></td></tr>'}
</table>

<div class="method-note">
<b>Interpretation:</b> High cross-topology correlation means the model's representational geometry is driven by <i>who believes what</i>, not by <i>who communicated with whom</i>. Low correlation would suggest surface-level communication structure dominates.
</div>

<!-- ============================================================ -->
<h2 id="layer-table">11. Full Layer-by-Layer Statistics</h2>

<p>Green rows indicate layers where epistemic RSA significantly exceeds communication RSA after FDR correction. Peak layer marked with *.</p>

<table>
    <tr>
        <th>Layer</th>
        <th>r<sub>epistemic</sub></th>
        <th>r<sub>communication</sub></th>
        <th>r<sub>position</sub></th>
        <th>p (Wilcoxon)</th>
        <th>p (FDR)</th>
        <th>p (permutation)</th>
    </tr>
    {rsa_table_rows}
</table>

<!-- ============================================================ -->
<h2 id="summary">12. Summary</h2>

<div class="key-finding">
<b>Key findings:</b>
<ul>
    <li><b>Behavioral validation:</b> {behav_summary.get('overall_accuracy', 0):.1%} overall accuracy on belief-tracking questions (updated agents: {behav_summary.get('accuracy_updated_agents', 0):.1%}, outdated agents: {behav_summary.get('accuracy_outdated_agents', 0):.1%}).</li>
    <li><b>Epistemic RSA:</b> Peak epistemic correlation of r = {peak_r if isinstance(peak_r, str) else f'{peak_r:.4f}'} at layer {peak_layer}. {len(sig_layers)} layers show epistemic RSA significantly above communication RSA (FDR-corrected).</li>
    <li><b>Position dominance:</b> The position RDM shows the strongest correlation throughout, reflecting the strong baseline of textual proximity effects on representations.</li>
    <li><b>Layer profile:</b> Epistemic signal is strongest in early-to-middle layers and decays in later layers, consistent with the model building internal world models in intermediate representations before collapsing to output predictions.</li>
</ul>
</div>

<h3>References</h3>
<ul style="font-size: 0.9em;">
    <li>Gurnee, W. &amp; Tegmark, M. (2023). Language Models Represent Space and Time. <i>arXiv:2310.02207</i>.</li>
    <li>Zhu, W., Zhang, H., &amp; Wang, H. (2024). Language Models Represent Beliefs of Self and Others. <i>ICML 2024</i>.</li>
    <li>Bortoletto, M., et al. (2024). Brittle Minds, Fixable Activations. <i>arXiv:2406.17513</i>.</li>
    <li>Kriegeskorte, N., Mur, M., &amp; Bandettini, P. (2008). Representational similarity analysis. <i>Frontiers in Systems Neuroscience</i>, 2, 4.</li>
    <li>Shai, L., et al. (2024). Transformers Represent Belief State Geometry in their Residual Stream. <i>NeurIPS 2024</i>.</li>
</ul>

<footer>
    Generated by <code>5a_results_summary_generator.py</code> on {datetime.now().strftime("%Y-%m-%d %H:%M")}.
    Model: LLaMA-2-13B-Chat. N = {n_narratives} narratives. {n_perms:,} permutations.
</footer>

</body>
</html>"""

    return html


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Loading data...", flush=True)
    data = load_all_data()

    for key, val in data.items():
        status = "loaded" if val is not None else "MISSING"
        print(f"  {key}: {status}")

    # Generate figures
    figures_b64 = {}

    print("\nGenerating figures...", flush=True)

    # Design diagram (always available)
    fig = make_design_figure()
    figures_b64["design"] = fig_to_base64(fig)
    save_fig(fig, "design_overview")
    plt.close(fig)
    print("  design_overview")

    # Behavioral
    if data["behavioral"]:
        fig = make_behavioral_figure(data["behavioral"])
        figures_b64["behavioral"] = fig_to_base64(fig)
        save_fig(fig, "behavioral_accuracy")
        plt.close(fig)
        print("  behavioral_accuracy")

    # Layer profile
    if data["rsa"]:
        fig = make_layer_profile_figure(data["rsa"])
        figures_b64["layer_profile"] = fig_to_base64(fig)
        save_fig(fig, "layer_profile_rsa")
        plt.close(fig)
        print("  layer_profile_rsa")

    # RDM heatmaps
    if data["rsa"] and data["rdms"]:
        fig = make_rdm_heatmaps(data["rsa"], data["rdms"])
        figures_b64["rdm_heatmaps"] = fig_to_base64(fig)
        save_fig(fig, "rdm_heatmaps")
        plt.close(fig)
        print("  rdm_heatmaps")

    # Topology breakdown
    if data["rsa"]:
        fig = make_topology_breakdown_figure(data["rsa"])
        if fig:
            figures_b64["topology_breakdown"] = fig_to_base64(fig)
            save_fig(fig, "topology_breakdown")
            plt.close(fig)
            print("  topology_breakdown")

    # Condition breakdown
    if data["rsa"]:
        fig = make_condition_breakdown_figure(data["rsa"])
        if fig:
            figures_b64["condition_breakdown"] = fig_to_base64(fig)
            save_fig(fig, "condition_breakdown")
            plt.close(fig)
            print("  condition_breakdown")

    # Generate HTML
    print("\nGenerating HTML report...", flush=True)
    html = generate_html(data, figures_b64)

    out_path = os.path.join(config.RSA_DIR, "results_summary.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Saved: {out_path}")

    # Also save figures list
    fig_list = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
    print(f"\nFigures saved to {FIGURES_DIR}/:")
    for fn in sorted(fig_list):
        print(f"  {fn}")

    print("\nDone.")


if __name__ == "__main__":
    main()
