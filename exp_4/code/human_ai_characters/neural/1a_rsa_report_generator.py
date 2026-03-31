#!/usr/bin/env python3
"""
Activation RSA Report Generator

Generates an HTML report with full methodology and charts for interpreting
the internal RSA results.

Reads from:
    results/{model}/expanded_mental_concepts/internals/rsa/data/
        all_character_activations.npz
        rdm_cosine_per_layer.npz
        rsa_results.json
        character_prompts.json

Writes to:
    results/{model}/expanded_mental_concepts/internals/rsa/{dataset}/
        activation_rsa_report.html

Usage:
    python expanded_mental_concepts/internals/rsa/activation_rsa_report_generator.py --model llama2_13b_chat
    python expanded_mental_concepts/internals/rsa/activation_rsa_report_generator.py --model llama2_13b_base
    python expanded_mental_concepts/internals/rsa/activation_rsa_report_generator.py --model llama2_13b_chat --dataset reduced_dataset

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

# ── Path setup ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import (
    config, set_model, add_model_argument,
    data_dir, results_dir,
)
from entities.characters import (
    AI_CHARACTERS, HUMAN_CHARACTERS,
    CHARACTER_NAMES, CHARACTER_TYPES,
)
from utils.report_utils import (
    REPORT_CSS, fig_to_b64, build_toc, build_html_header,
    build_html_footer, html_figure, add_dataset_argument,
    expanded_concepts_stimuli_html,
)

SECTIONS = [
    {"id": "research-question", "label": "1. Research Question"},
    {"id": "analysis-approach", "label": "2. Analysis Approach"},
    {"id": "stimuli", "label": "3. Stimuli"},
    {"id": "rsa-by-layer", "label": "4. RSA by Layer"},
    {"id": "rdm-heatmaps", "label": "5. RDM Heatmaps at Peak Layer"},
    {"id": "within-between", "label": "6. Within vs Between Group Distances"},
    {"id": "character-prompts", "label": "7. Character Prompts"},
]


def generate_report(model_key, dataset="full_dataset"):
    """Generate activation RSA report for one model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    set_model(model_key)

    ddir = data_dir("human_ai_characters", "neural/names_only", "rsa_pca")
    rdir = results_dir("human_ai_characters", "neural/names_only", "rsa_pca")

    # ── Load data ──
    rsa_path = os.path.join(str(ddir), "rsa_results.json")
    rdm_path = os.path.join(str(ddir), "rdm_cosine_per_layer.npz")
    prompts_path = os.path.join(str(ddir), "character_prompts.json")

    if not os.path.exists(rsa_path):
        print(f"RSA results not found at {rsa_path} — skipping {model_key}")
        return None

    with open(rsa_path) as f:
        rsa_all = json.load(f)
    rdm_data = np.load(rdm_path)
    with open(prompts_path) as f:
        prompt_metadata = json.load(f)

    model_rdm = rdm_data["model_rdm"]  # (n_layers, n_chars, n_chars)
    categorical_rdm = rdm_data["categorical_rdm"]  # (n_chars, n_chars)
    char_keys = list(rdm_data["character_keys"])
    has_behavioral = "behavioral_rdm" in rdm_data
    if has_behavioral:
        behavioral_rdm = rdm_data["behavioral_rdm"]

    n_chars = len(char_keys)
    n_layers = model_rdm.shape[0]

    figures = {}

    # ── 1. RSA by layer (line plot) ──
    fig, ax = plt.subplots(figsize=(10, 5))
    for variant_name in sorted(rsa_all.keys()):
        rsa_results = rsa_all[variant_name]
        layers = [r["layer"] for r in rsa_results]
        rhos = [r["rho"] for r in rsa_results]
        p_vals = [r["p_value"] for r in rsa_results]

        color = "#E53935" if variant_name == "categorical" else "#1E88E5"
        ax.plot(layers, rhos, color=color, linewidth=2, label=variant_name)

        # Mark significant layers
        sig_layers = [l for l, p in zip(layers, p_vals) if p < 0.05]
        sig_rhos = [r for r, p in zip(rhos, p_vals) if p < 0.05]
        ax.scatter(sig_layers, sig_rhos, color=color, s=15, zorder=3, alpha=0.6)

        # Mark peak
        valid = [(l, r) for l, r, p in zip(layers, rhos, p_vals) if not np.isnan(r)]
        if valid:
            peak_layer, peak_rho = max(valid, key=lambda x: x[1])
            ax.axvline(x=peak_layer, color=color, linestyle=":", alpha=0.4)
            ax.annotate(f"peak L{peak_layer}\n\u03c1={peak_rho:.3f}",
                        (peak_layer, peak_rho), fontsize=8,
                        xytext=(8, 8), textcoords="offset points", color=color)

    ax.axhline(y=0, color="gray", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman \u03c1 (RSA)")
    ax.set_title(f"RSA by Layer — {config.MODEL_LABEL}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    figures["rsa_by_layer"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 2. RDM heatmap at peak layer ──
    cat_results = rsa_all.get("categorical", [])
    valid_cat = [r for r in cat_results if not np.isnan(r["rho"])]
    if valid_cat:
        peak = max(valid_cat, key=lambda r: r["rho"])
        peak_layer = peak["layer"]
    else:
        peak_layer = n_layers // 2

    # Sort characters: AI first, then human (for visual grouping)
    ai_keys = [k for k in char_keys if k in AI_CHARACTERS]
    hu_keys = [k for k in char_keys if k in HUMAN_CHARACTERS]
    sorted_keys = ai_keys + hu_keys
    sort_idx = [char_keys.index(k) for k in sorted_keys]

    rdm_peak = model_rdm[peak_layer]
    rdm_sorted = rdm_peak[np.ix_(sort_idx, sort_idx)]

    fig, axes = plt.subplots(1, 2 if not has_behavioral else 3,
                              figsize=(6 * (3 if has_behavioral else 2), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # Model RDM
    im0 = axes[0].imshow(rdm_sorted, cmap="RdBu_r", aspect="equal")
    axes[0].set_title(f"Model RDM (Layer {peak_layer})")
    sorted_names = [CHARACTER_NAMES[k][:8] for k in sorted_keys]
    axes[0].set_xticks(range(n_chars))
    axes[0].set_xticklabels(sorted_names, rotation=90, fontsize=6)
    axes[0].set_yticks(range(n_chars))
    axes[0].set_yticklabels(sorted_names, fontsize=6)
    # Draw group boundary
    axes[0].axhline(y=len(ai_keys) - 0.5, color="black", linewidth=1)
    axes[0].axvline(x=len(ai_keys) - 0.5, color="black", linewidth=1)
    fig.colorbar(im0, ax=axes[0], shrink=0.7, label="Cosine distance")

    # Categorical RDM
    cat_sorted = categorical_rdm[np.ix_(sort_idx, sort_idx)]
    im1 = axes[1].imshow(cat_sorted, cmap="RdBu_r", aspect="equal")
    axes[1].set_title("Categorical RDM")
    axes[1].set_xticks(range(n_chars))
    axes[1].set_xticklabels(sorted_names, rotation=90, fontsize=6)
    axes[1].set_yticks(range(n_chars))
    axes[1].set_yticklabels(sorted_names, fontsize=6)
    axes[1].axhline(y=len(ai_keys) - 0.5, color="black", linewidth=1)
    axes[1].axvline(x=len(ai_keys) - 0.5, color="black", linewidth=1)
    fig.colorbar(im1, ax=axes[1], shrink=0.7)

    if has_behavioral:
        beh_sorted = behavioral_rdm[np.ix_(sort_idx, sort_idx)]
        im2 = axes[2].imshow(beh_sorted, cmap="RdBu_r", aspect="equal")
        axes[2].set_title("Behavioral RDM")
        axes[2].set_xticks(range(n_chars))
        axes[2].set_xticklabels(sorted_names, rotation=90, fontsize=6)
        axes[2].set_yticks(range(n_chars))
        axes[2].set_yticklabels(sorted_names, fontsize=6)
        axes[2].axhline(y=len(ai_keys) - 0.5, color="black", linewidth=1)
        axes[2].axvline(x=len(ai_keys) - 0.5, color="black", linewidth=1)
        fig.colorbar(im2, ax=axes[2], shrink=0.7)

    plt.tight_layout()
    figures["rdm_heatmaps"] = fig_to_b64(fig)
    plt.close(fig)

    # ── 3. Within-group vs between-group distances at peak ──
    fig, ax = plt.subplots(figsize=(7, 4))
    rdm_peak_flat = rdm_peak
    ai_set = set(k for k in char_keys if k in AI_CHARACTERS)
    hu_set = set(k for k in char_keys if k in HUMAN_CHARACTERS)

    within_ai, within_hu, between = [], [], []
    for i in range(n_chars):
        for j in range(i + 1, n_chars):
            d = rdm_peak_flat[i, j]
            ki, kj = char_keys[i], char_keys[j]
            if ki in ai_set and kj in ai_set:
                within_ai.append(d)
            elif ki in hu_set and kj in hu_set:
                within_hu.append(d)
            else:
                between.append(d)

    positions = [1, 2, 3]
    bp = ax.boxplot([within_ai, within_hu, between], positions=positions,
                    widths=0.5, patch_artist=True)
    colors_bp = ["#FFCDD2", "#BBDEFB", "#E0E0E0"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Within AI", "Within Human", "Between"])
    ax.set_ylabel("Cosine Distance")
    ax.set_title(f"Pairwise Distances at Peak Layer {peak_layer} — {config.MODEL_LABEL}")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    figures["within_between"] = fig_to_b64(fig)
    plt.close(fig)

    # ── Build HTML ──
    html_parts = []
    fig_num = 0

    # Peak info strings for summary
    peak_summaries = []
    for variant_name in sorted(rsa_all.keys()):
        valid = [r for r in rsa_all[variant_name] if not np.isnan(r["rho"])]
        if valid:
            pk = max(valid, key=lambda r: r["rho"])
            peak_summaries.append(
                f"{variant_name}: peak Layer {pk['layer']}, "
                f"\u03c1 = {pk['rho']:+.4f}, p = {pk['p_value']:.4f}"
            )

    html_parts.append(build_html_header("Activation RSA Report", config.MODEL_LABEL))

    html_parts.append(f"""
<div class="stat">
<strong>Summary:</strong> {n_chars} characters ({len(ai_set)} AI, {len(hu_set)} human),
{n_layers} layers. {'<br>'.join(peak_summaries)}
</div>
""")

    html_parts.append(build_toc(SECTIONS))

    html_parts.append(f"""
<h2 id="research-question">1. Research Question</h2>
<p>Does {config.MODEL_LABEL}'s internal representational geometry naturally reflect the
categorical distinction between human and AI characters? Unlike the behavioral PCA
(which measures what the model <em>says</em>), this analysis probes the model's
<em>internal activations</em> — does the model represent human and AI characters differently
at the neural level, even without being asked to compare them?</p>

<h2 id="analysis-approach">2. Analysis Approach</h2>
<div class="method">
<strong>Step-by-step procedure:</strong>
<ol>
<li><strong>Prompts:</strong> Each of {n_chars} characters is presented to the model with a
    minimal prompt: "Think about {{Name}}." This avoids cueing the model about what property
    to evaluate — we want the model's <em>default</em> representation of each character.</li>
<li><strong>Activation extraction:</strong> A single forward pass is run for each prompt.
    At each of the model's {n_layers} layers (embedding + 40 transformer layers), the
    residual-stream activation vector at the <em>last token position</em> is extracted.
    This yields a tensor of shape ({n_chars}, {n_layers}, 5120).</li>
<li><strong>Model RDM:</strong> At each layer, a Representational Dissimilarity Matrix (RDM)
    is computed using <strong>cosine distance</strong> between all {n_chars * (n_chars - 1) // 2}
    character pairs. The RDM is an {n_chars}&times;{n_chars} symmetric matrix where entry (i,j)
    is the cosine distance between character i's and character j's activation vectors. High cosine
    distance = dissimilar representations; low = similar.</li>
<li><strong>Reference RDMs:</strong>
    <ul>
    <li><strong>Categorical RDM:</strong> Binary — 0 for same-type pairs (both AI or both human),
        1 for cross-type pairs. This tests whether the model groups same-type characters together.</li>
    {"<li><strong>Behavioral RDM:</strong> Euclidean distance between characters in the behavioral PCA factor space (from Phase A). This tests whether the model's internal geometry mirrors its explicit behavioral judgments.</li>" if has_behavioral else "<li><strong>Behavioral RDM:</strong> <em>Not available</em> — behavioral PCA has not been run yet for this model.</li>"}
    </ul></li>
<li><strong>RSA (Representational Similarity Analysis):</strong> At each layer, the upper
    triangle of the model RDM is correlated (Spearman rank correlation) with the upper triangle
    of each reference RDM. The resulting \u03c1 value measures how well the model's representational
    geometry matches the reference structure. \u03c1 &gt; 0 means the model's internal distances
    are positively related to the reference — i.e., characters that should be dissimilar
    (cross-type) are farther apart in activation space.</li>
<li><strong>Layer profile:</strong> By computing RSA at every layer, we get a profile showing
    <em>where</em> in the network the human/AI structure emerges, peaks, and fades.</li>
</ol>
</div>

<div class="interpret">
<strong>How to interpret:</strong>
<ul>
<li><strong>RSA \u03c1 &gt; 0:</strong> The model's internal geometry aligns with the reference RDM.
    For the categorical RDM, this means same-type characters are represented more similarly than
    cross-type characters.</li>
<li><strong>Peak layer:</strong> The layer where the human/AI distinction is strongest in the
    model's activations. Early layers capture surface features; middle/late layers capture
    more abstract, semantic features.</li>
<li><strong>RDM heatmaps:</strong> Visual comparison — if the model RDM shows the same block
    structure as the categorical RDM (dark blocks within-group, light blocks between-group),
    the model's representations are categorically organized.</li>
<li><strong>Within vs. between distances:</strong> If between-group distances are systematically
    larger than within-group distances, the model maintains a categorical boundary.</li>
</ul>
</div>
""")

    html_parts.append(expanded_concepts_stimuli_html())

    # RSA by layer
    fig_num += 1
    html_parts.append(f"""
<h2 id="rsa-by-layer">4. RSA by Layer</h2>
<p>Spearman \u03c1 between the model's activation-based RDM and each reference RDM, computed
at every layer. Small dots mark layers with p &lt; .05. The dotted vertical line marks the peak.
A \u03c1 near 0 means no relationship; \u03c1 approaching 1 means the model's representational
geometry closely mirrors the reference structure.</p>
""")
    html_parts.append(html_figure(
        figures["rsa_by_layer"],
        "Spearman rho between model cosine-distance RDM and each reference RDM across transformer layers. Higher values indicate stronger alignment between internal representations and the reference structure.",
        fig_num=fig_num,
        alt="RSA by layer",
    ))

    # RSA table
    html_parts.append("""
<h3>Peak RSA per Reference RDM</h3>
<table>
<tr><th>Reference RDM</th><th>Peak Layer</th><th>Spearman \u03c1</th><th>p-value</th></tr>
""")
    for variant_name in sorted(rsa_all.keys()):
        valid = [r for r in rsa_all[variant_name] if not np.isnan(r["rho"])]
        if valid:
            pk = max(valid, key=lambda r: r["rho"])
            p_class = ' class="sig"' if pk["p_value"] < 0.05 else ""
            html_parts.append(
                f'<tr><td>{variant_name}</td><td>{pk["layer"]}</td>'
                f'<td>{pk["rho"]:+.4f}</td>'
                f'<td{p_class}>{pk["p_value"]:.4f}</td></tr>\n'
            )
    html_parts.append("</table>\n")

    # RDM heatmaps
    fig_num += 1
    html_parts.append(f"""
<h2 id="rdm-heatmaps">5. RDM Heatmaps at Peak Layer ({peak_layer})</h2>
<p>Left: the model's activation-based RDM at the peak layer (cosine distance between character
representations). {"Center" if has_behavioral else "Right"}: the categorical reference RDM
(binary: 0 = same type, 1 = different type).
{" Right: the behavioral reference RDM (Euclidean distance in PCA factor space)." if has_behavioral else ""}
Characters are sorted with AI first (top/left), then human (bottom/right). The black line
marks the group boundary. If the model RDM shows a similar block structure to the reference
RDMs, the internal representations are categorically organized.</p>
""")
    html_parts.append(html_figure(
        figures["rdm_heatmaps"],
        "Representational dissimilarity matrices at the peak RSA layer. Left panels show reference RDMs (categorical and/or behavioral); right panels show the model's cosine-distance RDM.",
        fig_num=fig_num,
        alt="RDM heatmaps",
    ))

    # Within vs between
    fig_num += 1
    html_parts.append(f"""
<h2 id="within-between">6. Within-Group vs. Between-Group Distances</h2>
<p>Distribution of pairwise cosine distances at the peak layer ({peak_layer}), split by
pair type. "Within AI" = both characters are AI; "Within Human" = both are human;
"Between" = one AI, one human. If the model categorically separates human from AI
characters, between-group distances should be systematically larger than within-group
distances.</p>
""")
    html_parts.append(html_figure(
        figures["within_between"],
        "Distribution of pairwise cosine distances within AI/human groups vs between groups at the peak layer. Greater between-group distance relative to within-group distance indicates categorical separation in activation space.",
        fig_num=fig_num,
        alt="Within vs between distances",
    ))
    html_parts.append(f"""<div class="stat">
Within AI: mean = {np.mean(within_ai):.4f} (n={len(within_ai)})<br>
Within Human: mean = {np.mean(within_hu):.4f} (n={len(within_hu)})<br>
Between: mean = {np.mean(between):.4f} (n={len(between)})
</div>
""")

    # Character prompts
    html_parts.append("""
<h2 id="character-prompts">7. Character Prompts</h2>
<p>Exact prompts used for activation extraction. Each character gets a single, minimal
prompt to avoid cueing specific properties.</p>
<table>
<tr><th>Character</th><th>Type</th><th>Prompt</th></tr>
""")
    for pm in prompt_metadata:
        html_parts.append(
            f'<tr><td>{pm["name"]}</td><td>{pm["type"]}</td>'
            f'<td>{pm["prompt"]}</td></tr>\n'
        )
    html_parts.append("</table>\n")

    html_parts.append(build_html_footer())

    out_dir = str(rdir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "activation_rsa_report.html")
    with open(out_path, "w") as f:
        f.write("".join(html_parts))

    print(f"Report: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate activation RSA report with charts"
    )
    add_model_argument(parser)
    add_dataset_argument(parser)
    parser.add_argument("--both", action="store_true",
                        help="Generate for both chat and base models")
    args = parser.parse_args()

    if args.both:
        models = ["llama2_13b_chat", "llama2_13b_base"]
    else:
        models = [args.model]

    for model_key in models:
        print(f"\n{'='*60}")
        print(f"  Generating report: {model_key}")
        print(f"{'='*60}\n")
        generate_report(model_key, dataset=args.dataset)


if __name__ == "__main__":
    main()
