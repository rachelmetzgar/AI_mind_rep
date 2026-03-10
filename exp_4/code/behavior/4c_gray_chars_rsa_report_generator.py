#!/usr/bin/env python3
"""
Gray-with-Characters RSA Report Generator

Uses existing character activations from concept_geometry/rsa/activation/data/
and computes RSA against the behavioral RDM from Gray-with-characters PCA
factor scores. CPU-only — no new activations needed.

Reads from:
    results/{model}/concept_geometry/rsa/activation/data/
        all_character_activations.npz
        rdm_cosine_per_layer.npz
    results/{model}/behavior/gray_characters/data/
        pairwise_pca_results.npz

Writes to:
    results/{model}/behavior/gray_characters/{dataset}/
        gray_chars_rsa_report.html

Usage:
    python behavior/4c_gray_chars_rsa_report_generator.py --model llama2_13b_base
    python behavior/4c_gray_chars_rsa_report_generator.py --model llama2_13b_chat --both
    python behavior/4c_gray_chars_rsa_report_generator.py --model llama2_13b_chat --dataset reduced_dataset

Env: llama2_env (CPU-only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_phase_dir,
)
from concept_geometry.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS, CHARACTER_TYPES,
)
from utils.utils import compute_behavioral_rdm, compute_categorical_rdm
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
)

SECTIONS = [
    {"id": "methods", "label": "1. Methods"},
    {"id": "rsa-profile", "label": "2. RSA Layer Profile"},
    {"id": "results-table", "label": "3. Per-Layer Results"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate Gray-with-characters RSA report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_model(model_key)

    # Load activation RDMs
    act_ddir = data_dir("concept_geometry/rsa", "activation")
    rdm_path = os.path.join(str(act_ddir), "rdm_cosine_per_layer.npz")
    if not os.path.exists(rdm_path):
        print(f"RDM data not found at {rdm_path} — skipping {model_key}")
        return None

    rdm_data = np.load(rdm_path)
    model_rdm = rdm_data["model_rdm"]
    char_keys = list(rdm_data["character_keys"])
    n_chars = len(char_keys)
    n_layers = model_rdm.shape[0]

    # Load Gray-with-characters behavioral PCA
    gray_ddir = data_dir("behavior", "gray_characters")
    gray_pca_path = os.path.join(str(gray_ddir), "pairwise_pca_results.npz")
    if not os.path.exists(gray_pca_path):
        print(f"Gray chars PCA not found at {gray_pca_path} — skipping {model_key}")
        return None

    gray_pca = np.load(gray_pca_path)
    gray_scores = gray_pca["factor_scores_01"]
    gray_char_keys = list(gray_pca["character_keys"])

    # Match character ordering
    if set(gray_char_keys) != set(char_keys):
        print("Warning: character key mismatch between activation and Gray data")
        # Find common characters
        common = [k for k in char_keys if k in gray_char_keys]
        if len(common) < 10:
            print("Too few common characters, skipping")
            return None

    # Compute Gray behavioral RDM
    gray_rdm = compute_behavioral_rdm(gray_scores)

    # Also compute categorical RDM for comparison
    categorical_rdm = compute_categorical_rdm(char_keys, CHARACTER_TYPES)

    triu_idx = np.triu_indices(n_chars, k=1)

    # RSA: Gray behavioral
    gray_upper = gray_rdm[triu_idx]
    cat_upper = categorical_rdm[triu_idx]

    rsa_gray = []
    rsa_categorical = []
    for layer in range(n_layers):
        model_upper = model_rdm[layer][triu_idx]
        if np.std(model_upper) < 1e-12:
            rsa_gray.append({"layer": layer, "rho": float("nan"),
                             "p_value": float("nan")})
            rsa_categorical.append({"layer": layer, "rho": float("nan"),
                                     "p_value": float("nan")})
            continue

        rho_g, p_g = spearmanr(model_upper, gray_upper)
        rsa_gray.append({"layer": layer, "rho": float(rho_g),
                          "p_value": float(p_g)})

        rho_c, p_c = spearmanr(model_upper, cat_upper)
        rsa_categorical.append({"layer": layer, "rho": float(rho_c),
                                 "p_value": float(p_c)})

    # Find peaks
    valid_gray = [r for r in rsa_gray if not np.isnan(r["rho"])]
    valid_cat = [r for r in rsa_categorical if not np.isnan(r["rho"])]
    peak_gray = max(valid_gray, key=lambda r: r["rho"]) if valid_gray else None
    peak_cat = max(valid_cat, key=lambda r: r["rho"]) if valid_cat else None

    figures = {}

    # ── RSA layer profile ──
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(n_layers))
    rhos_gray = [r["rho"] for r in rsa_gray]
    rhos_cat = [r["rho"] for r in rsa_categorical]

    ax.plot(layers, rhos_gray, color="#FF9800", linewidth=2,
            label="Gray Behavioral RDM")
    ax.plot(layers, rhos_cat, color="#4CAF50", linewidth=2,
            label="Categorical RDM")

    if peak_gray:
        ax.axvline(x=peak_gray["layer"], color="#FF9800", linestyle=":",
                   alpha=0.5)
        ax.annotate(f"peak L{peak_gray['layer']}\nrho={peak_gray['rho']:.3f}",
                    (peak_gray["layer"], peak_gray["rho"]),
                    fontsize=8, color="#FF9800",
                    xytext=(5, 10), textcoords="offset points")
    if peak_cat:
        ax.axvline(x=peak_cat["layer"], color="#4CAF50", linestyle=":",
                   alpha=0.5)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"RSA: Gray Capacities Behavioral RDM — {config.MODEL_LABEL}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    figures["rsa_profile"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    fig_num = 0
    html = []
    html.append(build_html_header("Gray Characters RSA Report", config.MODEL_LABEL))

    html.append(f"""
<div class="stat">
<strong>Summary:</strong> RSA between internal character representations
({n_layers} layers, {n_chars} characters) and behavioral RDMs from Gray
capacity ratings.
</div>
""")

    html.append(build_toc(SECTIONS))

    html.append(f"""
<h2 id="methods">1. Methods</h2>
<div class="method">
<strong>Research question:</strong> Do {config.MODEL_LABEL}'s internal character representations
(activations) align with the behavioral folk-psychology structure derived from Gray et al.
pairwise ratings? Specifically, does the geometry of character activations (cosine-distance RDM)
correlate with (a) the behavioral PCA factor-score distances (Gray behavioral RDM) and/or
(b) a binary human-vs-AI categorical RDM?

<p><strong>Data source:</strong> Character activations from "Think about [Name]." prompts
processed through {config.MODEL_LABEL}, extracting last-token residual-stream activations at all
41 layers. Gray behavioral RDM derived from PCA factor scores on 18-capacity pairwise data.
Categorical RDM: same type = 0, different type = 1.</p>

<strong>Procedure:</strong>
<ol>
<li><strong>Model RDM:</strong> Cosine distance (1 − cosine similarity) between all character
    pairs at each of 41 transformer layers, yielding a layer-wise model RDM.</li>
<li><strong>Reference RDMs:</strong> (a) Gray behavioral RDM — Euclidean distance in the
    2D PCA factor space from the behavioral analysis. (b) Categorical RDM — binary matrix
    (0 = same type, 1 = different type).</li>
<li><strong>RSA:</strong> Spearman rank correlation between upper triangles of model RDM
    and each reference RDM, computed at every layer.</li>
<li><strong>Layer profile:</strong> Plot RSA (rho) across layers to identify where
    internal representations best align with behavioral/categorical structure.</li>
</ol>
</div>
""")

    fig_num += 1
    html.append(f'<h2 id="rsa-profile">2. RSA Layer Profile</h2>\n')
    html.append(html_figure(
        figures["rsa_profile"],
        f"RSA layer profile comparing model RDM against Gray behavioral RDM "
        f"(orange) and categorical RDM (green) across {n_layers} layers.",
        fig_num=fig_num,
        alt="RSA profile",
    ))

    if peak_gray:
        html.append(f"""
<div class="stat">
<strong>Gray Behavioral RDM:</strong> Peak at layer {peak_gray['layer']},
rho = {peak_gray['rho']:+.4f}, p = {peak_gray['p_value']:.4f}
</div>
""")
    if peak_cat:
        html.append(f"""
<div class="stat">
<strong>Categorical RDM:</strong> Peak at layer {peak_cat['layer']},
rho = {peak_cat['rho']:+.4f}, p = {peak_cat['p_value']:.4f}
</div>
""")

    # Layer table
    html.append("""
<h2 id="results-table">3. Per-Layer Results</h2>
<table>
<tr><th>Layer</th><th>Gray rho</th><th>Gray p</th>
<th>Categorical rho</th><th>Categorical p</th></tr>
""")
    for layer in range(n_layers):
        rg = rsa_gray[layer]
        rc = rsa_categorical[layer]
        rg_class = ' class="sig"' if rg["p_value"] < 0.05 else ""
        rc_class = ' class="sig"' if rc["p_value"] < 0.05 else ""
        html.append(
            f'<tr><td>{layer}</td>'
            f'<td{rg_class}>{rg["rho"]:+.4f}</td>'
            f'<td>{rg["p_value"]:.4f}</td>'
            f'<td{rc_class}>{rc["rho"]:+.4f}</td>'
            f'<td>{rc["p_value"]:.4f}</td></tr>\n'
        )
    html.append("</table>\n")
    html.append(build_html_footer())

    out_dir = os.path.join(str(results_phase_dir("behavior", "gray_characters")), dataset)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gray_chars_rsa_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html))

    # Save RSA data
    rsa_data_path = os.path.join(str(gray_ddir),
                                  "gray_chars_rsa_results.json")
    with open(rsa_data_path, "w") as f:
        json.dump({
            "gray_behavioral": rsa_gray,
            "categorical": rsa_categorical,
        }, f, indent=2)

    print(f"Report: {out_path}")
    print(f"RSA data: {rsa_data_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gray-with-characters RSA report"
    )
    add_model_argument(parser)
    add_dataset_argument(parser)
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Gray Characters RSA Report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
