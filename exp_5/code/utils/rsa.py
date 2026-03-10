"""
Experiment 5 — Mental State Attribution RSA
RSA utilities: RDM construction, model RDMs, RSA tests, permutation, FDR.

Rachel C. Metzgar · Mar 2026
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist


# ── RDM computation ──────────────────────────────────────────────────────────

def compute_rdm(activations, metric="correlation"):
    """Compute representational dissimilarity matrix.

    Args:
        activations: (n_items, hidden_dim) array
        metric: "correlation" (1 - Pearson r) or "cosine"

    Returns:
        (n_items, n_items) symmetric distance matrix
    """
    return squareform(pdist(activations, metric=metric))


def lower_triangle(rdm):
    """Extract lower triangle as a vector (excludes diagonal)."""
    n = rdm.shape[0]
    idx = np.tril_indices(n, k=-1)
    return rdm[idx]


# ── Model RDM builders ──────────────────────────────────────────────────────

def _pair_conditions(n_items, n_conds, i_idx, j_idx):
    """Build a binary RDM (0 = similar, 1 = dissimilar).

    i_idx and j_idx index into the 336-row ordering where each item has
    n_conds consecutive rows.  A pair (r, c) gets 0 if both rows belong
    to conditions listed in the similarity set.

    Args:
        n_items: number of items (56)
        n_conds: number of conditions per item (6)
        i_idx: set of condition indices (0-based within item) that form
               the "similar" group
        j_idx: same (i_idx == j_idx for simple models)

    Returns:
        (n_items * n_conds, n_items * n_conds) binary RDM
    """
    n = n_items * n_conds
    rdm = np.ones((n, n), dtype=np.float64)

    for item in range(n_items):
        base = item * n_conds
        for ci in i_idx:
            for cj in j_idx:
                r, c = base + ci, base + cj
                if r != c:
                    rdm[r, c] = 0.0
                    rdm[c, r] = 0.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def _cross_item_condition_rdm(n_items, n_conds, similar_conds):
    """Build RDM where ANY two rows with conditions in similar_conds
    are marked similar, even across items."""
    n = n_items * n_conds
    rdm = np.ones((n, n), dtype=np.float64)
    similar_set = set(similar_conds)

    for r in range(n):
        cond_r = r % n_conds
        if cond_r not in similar_set:
            continue
        for c in range(r + 1, n):
            cond_c = c % n_conds
            if cond_c in similar_set:
                rdm[r, c] = 0.0
                rdm[c, r] = 0.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def build_model_rdms(n_items=56, n_conds=6):
    """Build all theoretical model RDMs.

    Condition indices (within each item's 6 rows):
        0 = mental_state, 1 = dis_mental, 2 = scr_mental,
        3 = action,       4 = dis_action, 5 = scr_action

    Returns:
        dict of name -> (n, n) RDM
    """
    models = {}

    # Model A: Full Attribution — only C1×C1 similar
    models["A"] = _cross_item_condition_rdm(n_items, n_conds, [0])

    # Model B: Mental Verb Presence — C1, C2, C3 all similar
    models["B"] = _cross_item_condition_rdm(n_items, n_conds, [0, 1, 2])

    # Model C: Subject Presence — C1, C4 similar
    models["C"] = _cross_item_condition_rdm(n_items, n_conds, [0, 3])

    # Model D: Item Identity — all 6 conditions of same item similar
    models["D"] = _pair_conditions(n_items, n_conds, [0, 1, 2, 3, 4, 5],
                                   [0, 1, 2, 3, 4, 5])

    # Model E: Mental Verb + Object, Subject-Optional — C1, C2 similar
    models["E"] = _cross_item_condition_rdm(n_items, n_conds, [0, 1])

    # Model F: Grammatical Order — C1, C2, C4, C5 similar
    models["F"] = _cross_item_condition_rdm(n_items, n_conds, [0, 1, 3, 4])

    # Model G: Scrambled Form — C3, C6 similar
    models["G"] = _cross_item_condition_rdm(n_items, n_conds, [2, 5])

    # Model H: Action Verb Presence — C4, C5, C6 similar
    models["H"] = _cross_item_condition_rdm(n_items, n_conds, [3, 4, 5])

    return models


def build_category_rdm(n_items=56, items_per_cat=8):
    """Build category-structure RDM for condition 1 only (56 items).

    Items 0-7 = attention, 8-15 = memory, ..., 48-55 = intention.
    Same-category pairs = 0, different = 1.
    """
    rdm = np.ones((n_items, n_items), dtype=np.float64)
    for cat_start in range(0, n_items, items_per_cat):
        for i in range(cat_start, cat_start + items_per_cat):
            for j in range(cat_start, cat_start + items_per_cat):
                rdm[i, j] = 0.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


# ── RSA tests ────────────────────────────────────────────────────────────────

def simple_rsa(neural_vec, model_vec):
    """Spearman correlation between two RDM vectors."""
    rho, p = spearmanr(neural_vec, model_vec)
    return float(rho), float(p)


def partial_rsa_regression(neural_vec, model_vecs, model_names):
    """Multiple regression of neural RDM on model RDMs.

    Args:
        neural_vec: (n_pairs,) neural RDM lower triangle
        model_vecs: list of (n_pairs,) model RDM lower triangles
        model_names: list of model names (same order)

    Returns:
        dict of model_name -> {"beta": float, "semi_partial_r": float}
    """
    X = np.column_stack(model_vecs)
    y = neural_vec

    # z-score columns for interpretable betas
    X_z = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    y_z = (y - y.mean()) / (y.std() + 1e-12)

    # OLS: beta = (X'X)^-1 X'y
    XtX = X_z.T @ X_z
    Xty = X_z.T @ y_z
    try:
        betas = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        betas = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

    # Semi-partial correlations: for each predictor, remove variance
    # explained by all OTHER predictors, then correlate residual with y
    y_hat = X_z @ betas
    ss_total = np.sum(y_z ** 2)
    results = {}
    for i, name in enumerate(model_names):
        # Reduced model: all predictors except i
        keep = [j for j in range(len(model_names)) if j != i]
        X_reduced = X_z[:, keep]
        try:
            betas_reduced = np.linalg.solve(
                X_reduced.T @ X_reduced, X_reduced.T @ y_z
            )
        except np.linalg.LinAlgError:
            betas_reduced = np.linalg.lstsq(
                X_reduced.T @ X_reduced, X_reduced.T @ y_z, rcond=None
            )[0]
        y_hat_reduced = X_reduced @ betas_reduced
        ss_full = np.sum((y_z - y_hat) ** 2)
        ss_reduced = np.sum((y_z - y_hat_reduced) ** 2)
        delta_r2 = (ss_reduced - ss_full) / ss_total
        semi_partial_r = np.sign(betas[i]) * np.sqrt(max(delta_r2, 0.0))
        results[name] = {
            "beta": float(betas[i]),
            "semi_partial_r": float(semi_partial_r),
        }
    return results


# ── Permutation testing ──────────────────────────────────────────────────────

def _permute_conditions_within_items(n_items, n_conds, rng):
    """Return a permutation array that shuffles condition labels within items.

    Each item's 6 rows get their condition assignments shuffled,
    but item membership is preserved.
    """
    perm = np.arange(n_items * n_conds)
    for item in range(n_items):
        start = item * n_conds
        block = perm[start:start + n_conds].copy()
        rng.shuffle(block)
        perm[start:start + n_conds] = block
    return perm


def permutation_test_simple(neural_rdm, model_rdm, n_perms=10000, seed=42):
    """Permutation test for simple RSA.

    Shuffles condition labels within items, recomputes Spearman rho.

    Returns:
        observed_rho, p_value, null_distribution
    """
    n = neural_rdm.shape[0]
    n_items = n // 6
    n_conds = 6

    neural_vec = lower_triangle(neural_rdm)
    model_vec = lower_triangle(model_rdm)
    observed_rho, _ = simple_rsa(neural_vec, model_vec)

    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_perms)

    for p in range(n_perms):
        perm = _permute_conditions_within_items(n_items, n_conds, rng)
        perm_rdm = neural_rdm[np.ix_(perm, perm)]
        perm_vec = lower_triangle(perm_rdm)
        null_rhos[p], _ = simple_rsa(perm_vec, model_vec)

    p_value = float(np.mean(null_rhos >= observed_rho))
    return observed_rho, p_value, null_rhos


def permutation_test_partial(neural_rdm, model_rdm_dict, hypothesis_key,
                             confound_keys, n_perms=10000, seed=42):
    """Permutation test for partial RSA (multiple regression).

    Shuffles condition labels within items, rebuilds neural RDM,
    re-runs regression, records beta and semi-partial r for each model.

    Returns:
        observed_results: dict from partial_rsa_regression
        p_values: dict of model_name -> p_value (for beta)
        null_betas: dict of model_name -> array of null betas
    """
    n = neural_rdm.shape[0]
    n_items = n // 6
    n_conds = 6

    neural_vec = lower_triangle(neural_rdm)
    all_keys = [hypothesis_key] + list(confound_keys)
    model_vecs = [lower_triangle(model_rdm_dict[k]) for k in all_keys]

    observed = partial_rsa_regression(neural_vec, model_vecs, all_keys)

    rng = np.random.default_rng(seed)
    null_betas = {k: np.empty(n_perms) for k in all_keys}

    for p in range(n_perms):
        perm = _permute_conditions_within_items(n_items, n_conds, rng)
        perm_rdm = neural_rdm[np.ix_(perm, perm)]
        perm_vec = lower_triangle(perm_rdm)
        perm_results = partial_rsa_regression(perm_vec, model_vecs, all_keys)
        for k in all_keys:
            null_betas[k][p] = perm_results[k]["beta"]

    p_values = {}
    for k in all_keys:
        obs_beta = observed[k]["beta"]
        p_values[k] = float(np.mean(np.abs(null_betas[k]) >= abs(obs_beta)))

    return observed, p_values, null_betas


def permutation_test_category(neural_rdm_c1, category_rdm,
                              n_items=56, items_per_cat=8,
                              n_perms=10000, seed=42):
    """Permutation test for within-condition category RSA.

    Shuffles category labels across items (preserving category sizes).

    Returns:
        observed_rho, p_value, null_distribution
    """
    neural_vec = lower_triangle(neural_rdm_c1)
    cat_vec = lower_triangle(category_rdm)
    observed_rho, _ = simple_rsa(neural_vec, cat_vec)

    n_cats = n_items // items_per_cat
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_perms)

    for p in range(n_perms):
        # Shuffle item-to-category assignment
        perm = rng.permutation(n_items)
        perm_rdm = neural_rdm_c1[np.ix_(perm, perm)]
        perm_vec = lower_triangle(perm_rdm)
        null_rhos[p], _ = simple_rsa(perm_vec, cat_vec)

    p_value = float(np.mean(null_rhos >= observed_rho))
    return observed_rho, p_value, null_rhos


# ── FDR correction ───────────────────────────────────────────────────────────

def fdr_correct(p_values):
    """Benjamini-Hochberg FDR correction.

    Args:
        p_values: array-like of p-values

    Returns:
        array of FDR-corrected p-values (same order as input)
    """
    pvals = np.asarray(p_values, dtype=np.float64)
    n = len(pvals)
    if n == 0:
        return pvals

    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    adjusted = np.empty(n)

    for i in range(n):
        rank = i + 1
        adjusted[i] = sorted_p[i] * n / rank

    # Enforce monotonicity (working backwards)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    adjusted = np.minimum(adjusted, 1.0)

    # Unsort
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result
