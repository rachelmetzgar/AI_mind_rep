"""
Experiment 5 — Mental State Attribution RSA
Probe training utilities: LOIO CV, permutation tests, Gram-Schmidt.

Rachel C. Metzgar · Mar 2026
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def train_full_probe(X, y, C=1.0, class_weight="balanced", max_iter=10000):
    """Train logistic regression on all data.

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) binary labels
        C: regularization strength (inverse)
        class_weight: passed to LogisticRegression
        max_iter: maximum solver iterations

    Returns:
        weights: (n_features,) array — learned coefficients
        bias: float — learned intercept
    """
    clf = LogisticRegression(
        C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs"
    )
    clf.fit(X, y)
    return clf.coef_[0], clf.intercept_[0]


def loio_cv(X, y, item_ids, C=1.0, class_weight="balanced", max_iter=10000):
    """Leave-one-item-out cross-validation.

    Each fold holds out all samples belonging to one item (across conditions).

    Args:
        X: (n_samples, n_features) array
        y: (n_samples,) binary labels
        item_ids: (n_samples,) item identifiers for grouping folds
        C: regularization strength
        class_weight: passed to LogisticRegression
        max_iter: maximum solver iterations

    Returns:
        y_true: (n_test_total,) true labels for all held-out samples
        y_pred: (n_test_total,) predicted labels
        y_prob: (n_test_total,) predicted probabilities for class 1
    """
    unique_items = np.unique(item_ids)
    y_true_all = []
    y_pred_all = []
    y_prob_all = []

    for held_out in unique_items:
        test_mask = item_ids == held_out
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Skip if only one class in training set
        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs"
        )
        clf.fit(X_train, y_train)

        y_true_all.append(y_test)
        y_pred_all.append(clf.predict(X_test))
        y_prob_all.append(clf.predict_proba(X_test)[:, 1])

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(y_prob_all),
    )


def loio_accuracy_auc(X, y, item_ids, **kwargs):
    """LOIO CV returning accuracy and AUC.

    Args:
        X, y, item_ids: as in loio_cv
        **kwargs: forwarded to loio_cv

    Returns:
        acc: float — accuracy
        auc: float — ROC AUC (nan if undefined)
    """
    y_true, y_pred, y_prob = loio_cv(X, y, item_ids, **kwargs)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return acc, auc


def permutation_test_loio(X, y, item_ids, n_perms=200, seed=42, **kwargs):
    """Permutation test for LOIO CV.

    Generates a null distribution by shuffling the label vector and
    re-running LOIO CV for each permutation.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) binary labels
        item_ids: (n_samples,) item identifiers
        n_perms: number of permutation iterations
        seed: random seed
        **kwargs: forwarded to loio_accuracy_auc

    Returns:
        observed_acc: float
        observed_auc: float
        p_acc: float — proportion of null >= observed accuracy
        p_auc: float — proportion of null >= observed AUC
    """
    obs_acc, obs_auc = loio_accuracy_auc(X, y, item_ids, **kwargs)

    rng = np.random.default_rng(seed)
    null_accs = np.empty(n_perms)
    null_aucs = np.empty(n_perms)

    for p in range(n_perms):
        y_perm = rng.permutation(y)
        acc, auc = loio_accuracy_auc(X, y_perm, item_ids, **kwargs)
        null_accs[p] = acc
        null_aucs[p] = auc

    p_acc = float(np.mean(null_accs >= obs_acc))
    p_auc = float(np.mean(null_aucs >= obs_auc))
    return obs_acc, obs_auc, p_acc, p_auc


def project_out_directions(w, directions):
    """Gram-Schmidt: project out each direction from w sequentially.

    Args:
        w: (D,) vector to project
        directions: list of (D,) vectors to project out

    Returns:
        w_residual: (D,) residual vector after projecting out all directions
    """
    w_res = w.copy().astype(np.float64)
    for d in directions:
        d = d.astype(np.float64)
        d_norm = d / (np.linalg.norm(d) + 1e-12)
        w_res = w_res - np.dot(w_res, d_norm) * d_norm
    return w_res


def multinomial_loio_cv(X, y, item_ids, C=1.0, max_iter=10000):
    """Multinomial LOIO CV for multi-class classification.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,) integer class labels
        item_ids: (n_samples,) item identifiers

    Returns:
        accuracy: float (nan if no valid folds)
    """
    unique_items = np.unique(item_ids)
    y_true_all = []
    y_pred_all = []

    for held_out in unique_items:
        test_mask = item_ids == held_out
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(
            multi_class="multinomial", solver="lbfgs",
            C=C, max_iter=max_iter,
        )
        clf.fit(X_train, y_train)

        y_true_all.append(y_test)
        y_pred_all.append(clf.predict(X_test))

    if not y_true_all:
        return float("nan")
    return accuracy_score(np.concatenate(y_true_all), np.concatenate(y_pred_all))


def cross_condition_accuracy(X_train, y_train, X_test, y_test,
                             C=1.0, max_iter=10000):
    """Train on one condition, test on another.

    Args:
        X_train, y_train: training data from source condition
        X_test, y_test: test data from target condition
        C: regularization strength
        max_iter: maximum solver iterations

    Returns:
        accuracy: float (nan if training set has < 2 classes)
    """
    if len(np.unique(y_train)) < 2:
        return float("nan")
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        C=C, max_iter=max_iter,
    )
    clf.fit(X_train, y_train)
    return accuracy_score(y_test, clf.predict(X_test))
