#!/usr/bin/env python3
"""
Experiment 4: Shared Utility Functions

Deduplicated functions used across multiple exp_4 scripts:
- RDM computation (cosine distance)
- Human RDM variants (combined, experience, agency)
- RSA (Spearman correlation between RDM upper triangles)
- PCA with varimax rotation
- Character means from pairwise data
- Human correlation computation
- Display helpers (nice_entity, nice_capacity)
- LLaMA-2 chat prompt formatting

Rachel C. Metzgar · Mar 2026
"""

import numpy as np
from scipy.stats import spearmanr


# ============================================================================
# RDM COMPUTATION
# ============================================================================

def compute_rdm_cosine(entity_activations):
    """
    Compute representational dissimilarity matrix (cosine distance)
    at each layer.

    Args:
        entity_activations: (n_entities, n_layers, hidden_dim)

    Returns:
        rdm: (n_layers, n_entities, n_entities) cosine distances
    """
    n_entities, n_layers, hidden_dim = entity_activations.shape
    rdm = np.zeros((n_layers, n_entities, n_entities))

    for layer in range(n_layers):
        vecs = entity_activations[:, layer, :]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        vecs_normed = vecs / norms
        cos_sim = vecs_normed @ vecs_normed.T
        rdm[layer] = 1.0 - cos_sim

    return rdm


# ============================================================================
# HUMAN RDMs
# ============================================================================

def compute_human_rdm_combined(entity_keys, scores_dict):
    """
    Compute human RDM from Gray et al. Experience/Agency scores
    using Euclidean distance in 2D space.
    """
    n = len(entity_keys)
    coords = np.array([scores_dict[k] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
    return rdm


def compute_human_rdm_experience(entity_keys, scores_dict):
    """Compute human RDM using absolute difference in Experience scores only."""
    n = len(entity_keys)
    exp_scores = np.array([scores_dict[k][0] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(exp_scores[i] - exp_scores[j])
    return rdm


def compute_human_rdm_agency(entity_keys, scores_dict):
    """Compute human RDM using absolute difference in Agency scores only."""
    n = len(entity_keys)
    agency_scores = np.array([scores_dict[k][1] for k in entity_keys])
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = abs(agency_scores[i] - agency_scores[j])
    return rdm


def compute_all_human_rdms(entity_keys, scores_dict):
    """Compute all three human RDM variants. Returns dict."""
    return {
        "combined": compute_human_rdm_combined(entity_keys, scores_dict),
        "experience": compute_human_rdm_experience(entity_keys, scores_dict),
        "agency": compute_human_rdm_agency(entity_keys, scores_dict),
    }


# ============================================================================
# RSA
# ============================================================================

def compute_rsa_all_layers(model_rdm, human_rdm, n_entities):
    """
    Spearman correlation between upper triangles of model and human RDMs
    at every layer.

    Returns: list of dicts with {layer, rho, p_value, n_pairs}.
    """
    n_layers = model_rdm.shape[0]
    triu_idx = np.triu_indices(n_entities, k=1)
    human_upper = human_rdm[triu_idx]
    n_pairs = len(human_upper)

    results = []
    for layer in range(n_layers):
        model_upper = model_rdm[layer][triu_idx]
        if np.std(model_upper) < 1e-12:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = spearmanr(model_upper, human_upper)
        results.append({
            "layer": layer,
            "rho": float(rho),
            "p_value": float(p),
            "n_pairs": n_pairs,
        })

    return results


# ============================================================================
# PCA WITH VARIMAX ROTATION
# ============================================================================

def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """
    Varimax rotation of a factor loading matrix.

    Matches Gray et al.'s use of varimax rotation to maximize simple
    structure (each capacity loads highly on one factor).
    """
    n, k = loadings.shape
    rotation = np.eye(k)
    rotated = loadings.copy()

    for iteration in range(max_iter):
        old_rotated = rotated.copy()

        for i in range(k):
            for j in range(i + 1, k):
                x = rotated[:, i]
                y = rotated[:, j]

                u = x ** 2 - y ** 2
                v = 2 * x * y

                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)

                num = D - 2 * A * B / n
                den = C - (A ** 2 - B ** 2) / n

                phi = 0.25 * np.arctan2(num, den)

                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                new_i = rotated[:, i] * cos_phi + rotated[:, j] * sin_phi
                new_j = -rotated[:, i] * sin_phi + rotated[:, j] * cos_phi
                rotated[:, i] = new_i
                rotated[:, j] = new_j

                rot_ij = np.eye(k)
                rot_ij[i, i] = cos_phi
                rot_ij[j, j] = cos_phi
                rot_ij[i, j] = sin_phi
                rot_ij[j, i] = -sin_phi
                rotation = rotation @ rot_ij

        if np.max(np.abs(rotated - old_rotated)) < tol:
            break

    return rotated, rotation


def run_pca_varimax(means):
    """
    PCA with varimax rotation on the capacity-by-entity matrix.

    Following Gray et al. exactly:
    1. Correlations between capacities across characters
    2. PCA on the correlation matrix
    3. Retain factors with eigenvalue > 1
    4. Varimax rotation
    5. Regression-method factor scores
    6. Rescale to 0-1

    Args:
        means: (n_capacities, n_entities) matrix

    Returns dict with rotated loadings, factor scores, explained variance, etc.
    """
    n_cap, n_ent = means.shape

    corr_matrix = np.corrcoef(means)  # (n_cap, n_cap)

    eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_factors = np.sum(eigenvalues > 1.0)
    n_factors = max(n_factors, 2)
    print(f"  Eigenvalues: {eigenvalues[:5]}")
    print(f"  Factors retained (eigenvalue > 1): {n_factors}")

    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])

    total_var = np.sum(eigenvalues)
    explained_var_ratio = eigenvalues[:n_factors] / total_var

    rotated_loadings, rotation_matrix = varimax_rotation(loadings)

    # Factor scores via regression method
    means_std = (means - means.mean(axis=1, keepdims=True))
    stds = means.std(axis=1, keepdims=True)
    stds = np.maximum(stds, 1e-10)
    means_std = means_std / stds

    corr_inv = np.linalg.pinv(corr_matrix)
    score_coefficients = corr_inv @ rotated_loadings

    factor_scores = means_std.T @ score_coefficients  # (n_ent, n_factors)

    # Rescale to 0-1
    factor_scores_01 = np.zeros_like(factor_scores)
    for f in range(n_factors):
        fmin = factor_scores[:, f].min()
        fmax = factor_scores[:, f].max()
        if fmax - fmin > 1e-10:
            factor_scores_01[:, f] = (
                (factor_scores[:, f] - fmin) / (fmax - fmin)
            )

    return {
        "rotated_loadings": rotated_loadings,
        "unrotated_loadings": loadings,
        "factor_scores_raw": factor_scores,
        "factor_scores_01": factor_scores_01,
        "eigenvalues": eigenvalues,
        "explained_var_ratio": explained_var_ratio,
        "n_factors": n_factors,
        "rotation_matrix": rotation_matrix,
        "score_coefficients": score_coefficients,
    }


# ============================================================================
# CHARACTER MEANS (PAIRWISE)
# ============================================================================

def compute_character_means_pairwise(responses, entity_keys, capacity_keys,
                                     rating_key="expected_rating"):
    """
    Compute mean relative rating per entity per capacity from pairwise data.

    For each comparison of entity A vs entity B with rating R (1-5):
        - Entity A gets score: (3 - R)
        - Entity B gets score: (R - 3)

    Args:
        responses: list of response dicts
        entity_keys: list of entity keys
        capacity_keys: list of capacity keys
        rating_key: key in response dict for the rating value
            ("expected_rating" for base, "rating" for chat)
    """
    n_cap = len(capacity_keys)
    n_ent = len(entity_keys)
    ent_to_idx = {k: i for i, k in enumerate(entity_keys)}

    scores = [[[] for _ in range(n_ent)] for _ in range(n_cap)]

    for resp in responses:
        r = resp.get(rating_key)
        if r is None:
            continue
        cap_idx = capacity_keys.index(resp["capacity"])
        a_idx = ent_to_idx[resp["entity_a"]]
        b_idx = ent_to_idx[resp["entity_b"]]

        scores[cap_idx][a_idx].append(3 - r)
        scores[cap_idx][b_idx].append(r - 3)

    means = np.zeros((n_cap, n_ent))
    for c in range(n_cap):
        for e in range(n_ent):
            if scores[c][e]:
                means[c, e] = np.mean(scores[c][e])

    return means


# ============================================================================
# HUMAN CORRELATION
# ============================================================================

def correlate_with_humans(pca_results, entity_keys, scores_dict):
    """
    Compute Spearman correlations between model factor scores and
    human Experience/Agency scores.

    Returns dict: {f1_experience: {rho, p}, f1_agency: {rho, p}, ...}
    """
    human_exp = np.array([scores_dict[k][0] for k in entity_keys])
    human_ag = np.array([scores_dict[k][1] for k in entity_keys])
    scores_01 = pca_results["factor_scores_01"]
    n_factors = min(2, scores_01.shape[1])

    results = {}
    for fi in range(n_factors):
        rho_e, p_e = spearmanr(scores_01[:, fi], human_exp)
        rho_a, p_a = spearmanr(scores_01[:, fi], human_ag)
        results[f"f{fi+1}_experience"] = {"rho": float(rho_e), "p": float(p_e)}
        results[f"f{fi+1}_agency"] = {"rho": float(rho_a), "p": float(p_a)}
    return results


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def nice_entity(name):
    """Pretty-print entity names."""
    lookup = {
        "dead_woman": "Dead woman",
        "frog": "Frog",
        "robot": "Robot (Kismet)",
        "fetus": "Fetus (7 wk)",
        "pvs_patient": "PVS patient",
        "god": "God",
        "dog": "Dog",
        "chimpanzee": "Chimpanzee",
        "baby": "Baby (5 mo)",
        "girl": "Girl (5 yr)",
        "adult_woman": "Adult woman",
        "adult_man": "Adult man",
        "you_self": "You (self)",
    }
    return lookup.get(name, name)


def nice_capacity(name):
    """Pretty-print capacity names."""
    return name.replace("_", " ").replace("emotion recognition", "emotion recog.").title()


# ============================================================================
# LLAMA-2 CHAT PROMPT FORMATTING
# ============================================================================

def llama_v2_prompt(messages, system_prompt=None):
    """Format messages into LLaMA-2-Chat token string.

    If no system message is present in messages, prepends the default
    LLaMA-2 system prompt (or a custom one via system_prompt).
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"

    if system_prompt:
        default_sys = system_prompt
    else:
        default_sys = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Please ensure that your responses are socially unbiased and "
            "positive in nature. If a question does not make any sense, or "
            "is not factually coherent, explain why instead of answering "
            "something not correct. If you don't know the answer to a "
            "question, please don't share false information."
        )

    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": default_sys}] + messages

    # Fold system prompt into first user message
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS
                       + messages[1]["content"],
        }
    ] + messages[2:]

    # Pair up user/assistant turns
    parts = [
        f"{BOS}{B_INST} {prompt['content'].strip()} {E_INST} "
        f"{answer['content'].strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    # Unpaired final user message
    if messages[-1]["role"] == "user":
        parts.append(
            f"{BOS}{B_INST} {messages[-1]['content'].strip()} {E_INST}"
        )

    return "".join(parts)


# ============================================================================
# LLAMA-3 CHAT PROMPT FORMATTING
# ============================================================================

def llama_v3_prompt(messages, system_prompt=None):
    """Format messages into LLaMA-3-Instruct chat template.

    Uses the <|begin_of_text|><|start_header_id|>...<|end_header_id|> format.
    If no system message is present, prepends a default system prompt.
    """
    BOS = "<|begin_of_text|>"
    EOT = "<|eot_id|>"

    def _header(role):
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n"

    if system_prompt is None:
        system_prompt = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe."
        )

    # Insert system message if not already present
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    parts = [BOS]
    for msg in messages:
        parts.append(_header(msg["role"]))
        parts.append(msg["content"].strip())
        parts.append(EOT)

    # Add assistant header to prompt generation
    parts.append(_header("assistant"))

    return "".join(parts)


# ============================================================================
# GEMMA-2 CHAT PROMPT FORMATTING
# ============================================================================

def gemma2_prompt(messages, system_prompt=None):
    """Format messages into Gemma-2 chat template.

    Gemma-2 has no system role. System prompt is folded into the first
    user message (same strategy as LLaMA-2).

    Format:
        <start_of_turn>user
        {content}<end_of_turn>
        <start_of_turn>model
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe."
        )

    # Insert system message if not already present
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    # Fold system prompt into first user message
    sys_content = messages[0]["content"]
    messages = [
        {
            "role": messages[1]["role"],
            "content": sys_content + "\n\n" + messages[1]["content"],
        }
    ] + messages[2:]

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        parts.append(f"<start_of_turn>{role}\n{content}<end_of_turn>\n")

    # Add model header to prompt generation
    parts.append("<start_of_turn>model\n")

    return "".join(parts)


# ============================================================================
# QWEN-2.5 CHAT PROMPT FORMATTING (ChatML)
# ============================================================================

def qwen2_prompt(messages, system_prompt=None):
    """Format messages into Qwen-2.5 ChatML template.

    Format:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {content}<|im_end|>
        <|im_start|>assistant
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe."
        )

    # Insert system message if not already present
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages

    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content'].strip()}<|im_end|>\n")

    # Add assistant header to prompt generation
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# ============================================================================
# QWEN3 CHAT PROMPT FORMATTING (ChatML + /no_think)
# ============================================================================

def qwen3_prompt(messages, system_prompt=None):
    """Format messages into Qwen3 ChatML template with thinking disabled.

    Same ChatML format as Qwen-2.5, but appends /no_think to the system
    message to suppress chain-of-thought <think> blocks that would break
    parse_rating().
    """
    if system_prompt is None:
        system_prompt = (
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe."
        )

    # Append /no_think to disable thinking mode
    system_prompt = system_prompt.rstrip() + "\n\n/no_think"

    # Insert system message if not already present
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_prompt}] + messages
    else:
        # Ensure /no_think is appended even if system message already exists
        if "/no_think" not in messages[0]["content"]:
            messages = list(messages)
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"].rstrip() + "\n\n/no_think",
            }

    parts = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content'].strip()}<|im_end|>\n")

    # Add assistant header to prompt generation
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# ============================================================================
# CHAT PROMPT DISPATCHER
# ============================================================================

def format_chat_prompt(messages, system_prompt=None, family=None):
    """Dispatch to the correct chat template based on model family.

    Args:
        messages: list of {"role": ..., "content": ...} dicts
        system_prompt: optional system prompt override
        family: "llama2" or "llama3". If None, imports config to detect.
    """
    if family is None:
        import sys
        # Import config from the parent package
        config_mod = sys.modules.get("config")
        if config_mod is None:
            raise RuntimeError(
                "Cannot auto-detect model family. "
                "Pass family='llama2' or family='llama3' explicitly."
            )
        family = config_mod.config.MODEL_FAMILY

    if family == "llama2":
        return llama_v2_prompt(messages, system_prompt=system_prompt)
    elif family == "llama3":
        return llama_v3_prompt(messages, system_prompt=system_prompt)
    elif family == "gemma2":
        return gemma2_prompt(messages, system_prompt=system_prompt)
    elif family == "qwen2":
        return qwen2_prompt(messages, system_prompt=system_prompt)
    elif family == "qwen3":
        return qwen3_prompt(messages, system_prompt=system_prompt)
    else:
        raise ValueError(f"Unknown model family: {family}")


# ============================================================================
# CATEGORICAL / BEHAVIORAL RDMs (Concept Geometry)
# ============================================================================

def compute_categorical_rdm(character_keys, type_dict):
    """Binary RDM: same type = 0, different type = 1."""
    n = len(character_keys)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rdm[i, j] = 0.0 if type_dict[character_keys[i]] == type_dict[character_keys[j]] else 1.0
    return rdm


def compute_behavioral_rdm(factor_scores):
    """Euclidean distance in PCA factor space."""
    from scipy.spatial.distance import squareform, pdist
    return squareform(pdist(factor_scores, metric='euclidean'))
