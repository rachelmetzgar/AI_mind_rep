"""
RSA Analysis for Belief Propagation Experiment.
================================================
Loads model_rdms.pkl and performs:
  4a. Per-narrative RSA (Spearman correlation per layer per narrative)
  4b. Aggregated RSA with Wilcoxon + permutation tests
  4c. Cross-topology consistency
  4d. Layer profile (mean/SEM)
  4e. Condition-specific breakdowns

Saves a checkpoint after each layer so it can resume if interrupted.
"""

import os, sys, json, pickle
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

CHECKPOINT_PATH = os.path.join(config.RSA_DIR, "rsa_checkpoint.pkl")


def spearman_rsa(model_rdm: np.ndarray, candidate_rdm: np.ndarray) -> float:
    """Compute Spearman correlation between two RDM upper triangles.

    Returns 0.0 if either RDM has zero variance (constant).
    """
    if np.std(model_rdm) < 1e-12 or np.std(candidate_rdm) < 1e-12:
        return 0.0
    r, _ = spearmanr(model_rdm, candidate_rdm)
    return float(r) if np.isfinite(r) else 0.0


def permutation_test(observed_r_values: np.ndarray, model_rdms: list,
                     candidate_rdms: list, n_perms: int = 10000) -> float:
    """Permutation test: is the mean RSA significantly above zero?

    Permutes rows/cols of candidate RDMs to build null distribution.
    """
    n = len(model_rdms)
    observed_mean = np.mean(observed_r_values)

    rng = np.random.default_rng(42)
    null_means = np.zeros(n_perms)

    # Pre-reconstruct all candidate 4x4 matrices once
    cand_mats = []
    for i in range(n):
        cand_ut = candidate_rdms[i]
        cand_mat = np.zeros((4, 4))
        idx = 0
        for r in range(4):
            for c in range(r + 1, 4):
                cand_mat[r, c] = cand_ut[idx]
                cand_mat[c, r] = cand_ut[idx]
                idx += 1
        cand_mats.append(cand_mat)

    # Pre-compute upper triangle indices
    ut_rows, ut_cols = np.triu_indices(4, k=1)

    for perm_i in range(n_perms):
        perm_order = rng.permutation(4)
        perm_rs = np.zeros(n)

        for i in range(n):
            perm_mat = cand_mats[i][np.ix_(perm_order, perm_order)]
            perm_ut = perm_mat[ut_rows, ut_cols]
            perm_rs[i] = spearman_rsa(model_rdms[i], perm_ut)

        null_means[perm_i] = np.mean(perm_rs)

    p_value = float(np.mean(null_means >= observed_mean))
    return p_value


def load_checkpoint():
    """Load checkpoint if it exists. Returns (per_narrative, per_layer, last_completed_layer)."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        print(f"Resuming from checkpoint: layers 0-{ckpt['last_completed_layer']} already done")
        return ckpt["per_narrative"], ckpt["per_layer"], ckpt["last_completed_layer"]
    return None, {}, -1


def save_checkpoint(per_narrative, per_layer, last_completed_layer):
    """Save checkpoint after completing a layer."""
    ckpt = {
        "per_narrative": per_narrative,
        "per_layer": per_layer,
        "last_completed_layer": last_completed_layer,
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(ckpt, f, protocol=4)


def main():
    # Load RDMs
    rdm_path = os.path.join(config.RDMS_DIR, "model_rdms.pkl")
    with open(rdm_path, "rb") as f:
        all_rdms = pickle.load(f)

    print(f"Loaded RDMs for {len(all_rdms)} narratives", flush=True)

    narrative_ids = sorted(all_rdms.keys())
    n_narratives = len(narrative_ids)

    # ================================================================
    # Load checkpoint or start fresh
    # ================================================================
    per_narrative, per_layer, last_completed_layer = load_checkpoint()

    # ================================================================
    # 4a. Per-narrative RSA (fast, recompute fully if no checkpoint)
    # ================================================================
    if per_narrative is None:
        print("\n--- 4a: Per-narrative RSA ---", flush=True)
        per_narrative = {}
        for nid in narrative_ids:
            data = all_rdms[nid]
            per_narrative[nid] = {}
            for layer in range(config.NUM_LAYERS):
                model_rdm = data["model_rdm"][layer]
                per_narrative[nid][layer] = {
                    "r_epistemic": spearman_rsa(model_rdm, data["epistemic_rdm"]),
                    "r_communication": spearman_rsa(model_rdm, data["communication_rdm"]),
                    "r_position": spearman_rsa(model_rdm, data["position_rdm"]),
                }
        print("  Per-narrative RSA complete.", flush=True)

    # ================================================================
    # 4b. Aggregated RSA per layer (with checkpointing)
    # ================================================================
    print(f"--- 4b: Aggregated RSA (starting from layer {last_completed_layer + 1}) ---", flush=True)

    for layer in range(last_completed_layer + 1, config.NUM_LAYERS):
        r_epist = np.array([per_narrative[nid][layer]["r_epistemic"] for nid in narrative_ids])
        r_comm = np.array([per_narrative[nid][layer]["r_communication"] for nid in narrative_ids])
        r_pos = np.array([per_narrative[nid][layer]["r_position"] for nid in narrative_ids])

        # Means and SEMs
        mean_e = float(np.mean(r_epist))
        mean_c = float(np.mean(r_comm))
        mean_p = float(np.mean(r_pos))
        sem_e = float(np.std(r_epist, ddof=1) / np.sqrt(n_narratives))
        sem_c = float(np.std(r_comm, ddof=1) / np.sqrt(n_narratives))
        sem_p = float(np.std(r_pos, ddof=1) / np.sqrt(n_narratives))

        # Wilcoxon signed-rank: epistemic > communication
        diff = r_epist - r_comm
        if np.all(diff == 0):
            p_wilcox = 1.0
        else:
            try:
                _, p_wilcox = wilcoxon(diff, alternative="greater")
                p_wilcox = float(p_wilcox)
            except ValueError:
                p_wilcox = 1.0

        # Permutation test for epistemic RDM significance
        model_rdm_list = [all_rdms[nid]["model_rdm"][layer] for nid in narrative_ids]
        epist_rdm_list = [all_rdms[nid]["epistemic_rdm"] for nid in narrative_ids]
        p_perm = permutation_test(r_epist, model_rdm_list, epist_rdm_list,
                                  n_perms=config.N_PERMUTATIONS)

        per_layer[layer] = {
            "mean_r_epistemic": mean_e,
            "mean_r_communication": mean_c,
            "mean_r_position": mean_p,
            "sem_r_epistemic": sem_e,
            "sem_r_communication": sem_c,
            "sem_r_position": sem_p,
            "p_epist_vs_comm": p_wilcox,
            "p_permutation": p_perm,
        }

        print(f"  Layer {layer:2d}/{config.NUM_LAYERS - 1}: "
              f"r_epist={mean_e:.4f}, r_comm={mean_c:.4f}, "
              f"r_pos={mean_p:.4f}, p_wilcox={p_wilcox:.4f}, p_perm={p_perm:.4f}",
              flush=True)

        # Checkpoint after each layer
        save_checkpoint(per_narrative, per_layer, layer)

    # BH-FDR correction across layers for epistemic > communication
    p_values_wilcox = [per_layer[l]["p_epist_vs_comm"] for l in range(config.NUM_LAYERS)]
    _, p_fdr, _, _ = multipletests(p_values_wilcox, alpha=config.ALPHA, method="fdr_bh")
    for layer in range(config.NUM_LAYERS):
        per_layer[layer]["p_epist_vs_comm_fdr"] = float(p_fdr[layer])

    # Peak layer
    mean_epist_by_layer = [per_layer[l]["mean_r_epistemic"] for l in range(config.NUM_LAYERS)]
    peak_layer = int(np.argmax(mean_epist_by_layer))
    peak_r = float(mean_epist_by_layer[peak_layer])
    print(f"\nPeak layer: {peak_layer} (r_epistemic = {peak_r:.4f})", flush=True)

    # ================================================================
    # 4c. Cross-topology consistency
    # ================================================================
    print("\n--- 4c: Cross-topology consistency ---", flush=True)

    cross_pairs = [
        {
            "name": "C_tells_D",
            "conditions": ["chain_override_C_tells_D", "diamond_override_C_tells_D"],
            "description": "AB=old, CD=new (chain vs diamond)",
        },
        {
            "name": "D_only",
            "conditions": ["chain_override_D", "diamond_override_D_only"],
            "description": "ABC=old, D=new (chain vs diamond)",
        },
    ]

    cross_topology_results = {}
    for pair in cross_pairs:
        cond1, cond2 = pair["conditions"]
        nids_1 = sorted([nid for nid in narrative_ids if all_rdms[nid]["condition"] == cond1])
        nids_2 = sorted([nid for nid in narrative_ids if all_rdms[nid]["condition"] == cond2])

        if not nids_1 or not nids_2:
            print(f"  Skipping {pair['name']}: missing conditions", flush=True)
            continue

        correlations = []
        for n1, n2 in zip(nids_1, nids_2):
            rdm1 = all_rdms[n1]["model_rdm"][peak_layer]
            rdm2 = all_rdms[n2]["model_rdm"][peak_layer]
            r = spearman_rsa(rdm1, rdm2)
            correlations.append(r)

        cross_topology_results[pair["name"]] = {
            "description": pair["description"],
            "conditions": pair["conditions"],
            "n_pairs": len(correlations),
            "mean_r": float(np.mean(correlations)),
            "sem_r": float(np.std(correlations, ddof=1) / np.sqrt(len(correlations))) if len(correlations) > 1 else 0.0,
            "correlations": [float(c) for c in correlations],
        }
        print(f"  {pair['name']}: mean r = {np.mean(correlations):.4f} "
              f"(n={len(correlations)} pairs)", flush=True)

    # ================================================================
    # 4e. Condition-specific breakdowns
    # ================================================================
    print("\n--- 4e: Condition-specific breakdowns ---", flush=True)

    by_condition = defaultdict(lambda: defaultdict(list))
    for nid in narrative_ids:
        condition = all_rdms[nid]["condition"]
        for layer in range(config.NUM_LAYERS):
            by_condition[condition][layer].append(
                per_narrative[nid][layer]["r_epistemic"]
            )

    by_condition_summary = {}
    for cond in sorted(by_condition.keys()):
        by_condition_summary[cond] = {}
        for layer in range(config.NUM_LAYERS):
            vals = by_condition[cond][layer]
            by_condition_summary[cond][layer] = {
                "mean_r_epistemic": float(np.mean(vals)),
                "sem_r_epistemic": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

    by_topology = defaultdict(lambda: defaultdict(list))
    for nid in narrative_ids:
        topo = all_rdms[nid]["topology"]
        for layer in range(config.NUM_LAYERS):
            by_topology[topo][layer].append(
                per_narrative[nid][layer]["r_epistemic"]
            )

    by_topology_summary = {}
    for topo in sorted(by_topology.keys()):
        by_topology_summary[topo] = {}
        for layer in range(config.NUM_LAYERS):
            vals = by_topology[topo][layer]
            by_topology_summary[topo][layer] = {
                "mean_r_epistemic": float(np.mean(vals)),
                "sem_r_epistemic": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

    # Override vs no-override at peak layer
    override_rs = []
    no_override_rs = []
    for nid in narrative_ids:
        cond = all_rdms[nid]["condition"]
        r = per_narrative[nid][peak_layer]["r_epistemic"]
        if "no_override" in cond:
            no_override_rs.append(r)
        else:
            override_rs.append(r)

    print(f"  Override conditions (peak layer {peak_layer}): "
          f"mean r = {np.mean(override_rs):.4f} (n={len(override_rs)})", flush=True)
    print(f"  No-override conditions (peak layer {peak_layer}): "
          f"mean r = {np.mean(no_override_rs):.4f} (n={len(no_override_rs)})", flush=True)

    # ================================================================
    # Save final results
    # ================================================================
    per_layer_str = {str(k): v for k, v in per_layer.items()}
    by_condition_str = {}
    for cond, layers in by_condition_summary.items():
        by_condition_str[cond] = {str(k): v for k, v in layers.items()}
    by_topology_str = {}
    for topo, layers in by_topology_summary.items():
        by_topology_str[topo] = {str(k): v for k, v in layers.items()}

    results = {
        "per_layer": per_layer_str,
        "peak_layer": peak_layer,
        "peak_r_epistemic": peak_r,
        "cross_topology": cross_topology_results,
        "by_condition": by_condition_str,
        "by_topology": by_topology_str,
        "n_narratives": n_narratives,
        "n_layers": config.NUM_LAYERS,
        "n_permutations": config.N_PERMUTATIONS,
        "alpha": config.ALPHA,
    }

    out_path = os.path.join(config.RSA_DIR, "rsa_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved RSA results to {out_path}", flush=True)

    # Clean up checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("Checkpoint removed.", flush=True)


if __name__ == "__main__":
    main()
