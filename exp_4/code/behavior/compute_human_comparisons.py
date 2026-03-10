#!/usr/bin/env python3
"""
Compute comprehensive statistical comparisons between model factor scores
and Gray et al. human factor scores.

For each condition (pairwise/individual × with_self/without_self):
  - Spearman correlations: F1↔Exp, F1↔Ag, F2↔Exp, F2↔Ag
  - Overall: Procrustes analysis (2D shape similarity after optimal rotation)
  - Repeat all excluding fetus + god

Reads:  results/{model}/behavior/{condition}/data/{method}_pca_results.npz
Writes: results/{model}/behavior/{condition}/data/{method}_human_comparisons.json

Usage:
    python behavior/compute_human_comparisons.py --model llama2_13b_base

Env: llama2_env (CPU-only, login node OK)
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.spatial import procrustes

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import config, set_model, add_model_argument, data_dir
from entities.gray_entities import GRAY_ET_AL_SCORES

EXCLUDE_ENTITIES = {"fetus", "god"}
EXCLUDE_ENTITIES_DEAD = {"fetus", "god", "dead_woman"}


def _rv_coeff(X, Y):
    """Compute RV coefficient between two centered matrices."""
    S_xy = X.T @ Y
    S_xx = X.T @ X
    S_yy = Y.T @ Y
    num = np.trace(S_xy @ S_xy.T)
    den = np.sqrt(np.trace(S_xx @ S_xx) * np.trace(S_yy @ S_yy))
    return float(num / den) if den > 0 else 0.0


def compute_correlations(model_scores, human_exp, human_ag, entity_keys,
                         n_perm=10000, rng_seed=42):
    """Compute all factor-dimension Spearman correlations + Procrustes + RV,
    with permutation p-values for Procrustes and RV."""
    n_factors = min(2, model_scores.shape[1])
    n = len(entity_keys)
    results = {}
    rng = np.random.RandomState(rng_seed)

    # Per-factor correlations
    for fi in range(n_factors):
        for dim_name, human_vals in [("experience", human_exp), ("agency", human_ag)]:
            rho, p = spearmanr(model_scores[:, fi], human_vals)
            results[f"f{fi+1}_{dim_name}"] = {"rho": float(rho), "p": float(p)}

    # Combined mindedness: human (E+A) vs each factor, and vs (F1+F2)
    human_combined = human_exp + human_ag
    for fi in range(n_factors):
        rho, p = spearmanr(model_scores[:, fi], human_combined)
        results[f"f{fi+1}_combined"] = {"rho": float(rho), "p": float(p)}
    if n_factors >= 2:
        model_combined = model_scores[:, 0] + model_scores[:, 1]
        rho, p = spearmanr(model_combined, human_combined)
        results["combined_combined"] = {"rho": float(rho), "p": float(p)}

    # Overall: Procrustes + RV on 2D factor space
    if n_factors >= 2:
        model_2d = model_scores[:, :2].copy()
        human_2d = np.column_stack([human_exp, human_ag])

        # Observed Procrustes disparity
        try:
            _, _, obs_disp = procrustes(human_2d, model_2d)
        except Exception as e:
            results["procrustes"] = {"disparity": None, "p": None, "error": str(e)}
            obs_disp = None

        # Observed RV coefficient
        X_c = model_2d - model_2d.mean(axis=0)
        Y_c = human_2d - human_2d.mean(axis=0)
        obs_rv = _rv_coeff(X_c, Y_c)

        # Permutation test: shuffle entity order in model scores
        if obs_disp is not None:
            n_disp_le = 0  # count permutations with disparity <= observed
            n_rv_ge = 0    # count permutations with RV >= observed
            for _ in range(n_perm):
                perm = rng.permutation(n)
                model_perm = model_2d[perm]
                try:
                    _, _, perm_disp = procrustes(human_2d, model_perm)
                    if perm_disp <= obs_disp:
                        n_disp_le += 1
                except Exception:
                    pass
                X_p = model_perm - model_perm.mean(axis=0)
                perm_rv = _rv_coeff(X_p, Y_c)
                if perm_rv >= obs_rv:
                    n_rv_ge += 1

            p_procrustes = (n_disp_le + 1) / (n_perm + 1)
            p_rv = (n_rv_ge + 1) / (n_perm + 1)

            results["procrustes"] = {
                "disparity": float(obs_disp),
                "p": float(p_procrustes),
                "n_perm": n_perm,
            }
            results["rv_coefficient"] = {
                "rv": obs_rv,
                "p": float(p_rv),
                "n_perm": n_perm,
            }
        else:
            results["rv_coefficient"] = {"rv": obs_rv, "p": None}

    results["n_entities"] = len(entity_keys)
    results["entities"] = list(entity_keys)
    return results


def run_condition(model_key, method, condition):
    """Compute comparisons for one method × condition."""
    set_model(model_key)
    ddir = str(data_dir("behavior", condition))

    pca_path = os.path.join(ddir, f"{method}_pca_results.npz")
    if not os.path.exists(pca_path):
        print(f"  Not found: {pca_path}")
        return None

    pca = np.load(pca_path)
    scores_01 = pca["factor_scores_01"]

    if "entity_keys" in pca:
        entity_keys = list(pca["entity_keys"])
    elif "character_keys" in pca:
        entity_keys = list(pca["character_keys"])
    else:
        entity_keys = list(GRAY_ET_AL_SCORES.keys())
        if condition == "without_self":
            entity_keys = [k for k in entity_keys if k != "you_self"]

    human_exp = np.array([GRAY_ET_AL_SCORES[k][0] for k in entity_keys])
    human_ag = np.array([GRAY_ET_AL_SCORES[k][1] for k in entity_keys])

    # ── Full set ──
    full_results = compute_correlations(scores_01, human_exp, human_ag, entity_keys)

    # ── Excluding fetus + god ──
    keep_mask = np.array([k not in EXCLUDE_ENTITIES for k in entity_keys])
    excl_keys = [k for k, m in zip(entity_keys, keep_mask) if m]
    excl_results = compute_correlations(
        scores_01[keep_mask], human_exp[keep_mask], human_ag[keep_mask], excl_keys)

    # ── Excluding fetus + god + dead_woman (drop from full PCA) ──
    keep_mask_dead = np.array([k not in EXCLUDE_ENTITIES_DEAD for k in entity_keys])
    excl_dead_keys = [k for k, m in zip(entity_keys, keep_mask_dead) if m]
    excl_dead_results = compute_correlations(
        scores_01[keep_mask_dead], human_exp[keep_mask_dead],
        human_ag[keep_mask_dead], excl_dead_keys)

    # ── Excl PCA: re-run PCA results (if available) ──
    excl_pca_results = {}
    for excl_name in ["excl_fetus_god", "excl_fetus_god_dead"]:
        excl_pca_path = os.path.join(ddir, f"{method}_pca_results_{excl_name}.npz")
        if os.path.exists(excl_pca_path):
            excl_pca = np.load(excl_pca_path)
            excl_pca_scores = excl_pca["factor_scores_01"]
            excl_pca_keys = list(excl_pca["entity_keys"])
            h_exp_excl_pca = np.array([GRAY_ET_AL_SCORES[k][0] for k in excl_pca_keys])
            h_ag_excl_pca = np.array([GRAY_ET_AL_SCORES[k][1] for k in excl_pca_keys])
            excl_pca_results[excl_name] = compute_correlations(
                excl_pca_scores, h_exp_excl_pca, h_ag_excl_pca, excl_pca_keys)
            print(f"  Computed {excl_name} PCA stats from {excl_pca_path}")

    output = {
        "full": full_results,
        "excl_fetus_god": excl_results,
        "excl_fetus_god_dead": excl_dead_results,
        "method": method,
        "condition": condition,
        "model": model_key,
    }
    for excl_name, res in excl_pca_results.items():
        output[f"{excl_name}_pca"] = res

    out_path = os.path.join(ddir, f"{method}_human_comparisons.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Wrote: {out_path}")

    # Print summary
    for subset_name, res in [("Full", full_results), ("Excl fetus+god", excl_results)]:
        print(f"\n  {subset_name} (n={res['n_entities']}):")
        for key in ["f1_experience", "f2_experience", "f1_agency", "f2_agency"]:
            if key in res:
                r = res[key]
                sig = " *" if r["p"] < 0.05 else ""
                print(f"    {key:20s}  rho={r['rho']:+.3f}  p={r['p']:.4f}{sig}")
        if "procrustes" in res and res["procrustes"].get("disparity") is not None:
            print(f"    {'procrustes':20s}  disparity={res['procrustes']['disparity']:.4f}")
        if "rv_coefficient" in res:
            print(f"    {'rv_coefficient':20s}  RV={res['rv_coefficient']['rv']:.4f}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Compute human comparison stats for Gray replication")
    add_model_argument(parser)
    args = parser.parse_args()

    for condition in ["with_self", "without_self"]:
        for method in ["pairwise", "individual"]:
            print(f"\n{'='*50}")
            print(f"  {method} / {condition}")
            print(f"{'='*50}")
            run_condition(args.model, method, condition)


if __name__ == "__main__":
    main()
