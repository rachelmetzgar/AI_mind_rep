#!/usr/bin/env python3
"""
Experiment 4: Gray Simple — Cross-Model Summary Report

Neural geometry of entity representations using "Think about {entity}" prompts.
Compares RSA and Neural PCA results across all 4 models (LLaMA-2-13B Chat/Base,
LLaMA-3-8B Instruct/Base).

Reads per-model data from:
    results/{model}/gray_simple/internals/with_self/data/
        rsa_results.json
        rdm_cosine_per_layer.npz
        neural_pca_results.npz
        neural_pca_analysis.json

Usage:
    python comparisons/4_gray_simple_summary_generator.py

Env: llama2_env (CPU only, login node OK)
Rachel C. Metzgar · Mar 2026
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# -- Local imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, COMPARISONS_DIR, VALID_MODELS, MODELS, ensure_dir, set_model, data_dir
from utils.report_utils import (
    REPORT_CSS, build_cross_model_header, build_html_footer, build_toc,
    fig_to_b64, html_figure, MODEL_COLORS, MODEL_LABELS, ALL_MODELS,
    gray_entities_stimuli_html, sort_models, GRID_NCOLS, make_model_grid,
    model_row_td, format_p_cell, INSTRUCTION_TUNING_PAIRS, apply_fdr,
    methodology_primer_html, neural_methods_primer_html,
)
from utils.utils import nice_entity
from entities.gray_entities import GRAY_ET_AL_SCORES, ENTITY_NAMES


# -- Style -------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# RSA variant display colors
RSA_VARIANT_STYLES = {
    "combined":   {"ls": "-",  "lw": 2.0, "label": "Combined"},
    "experience": {"ls": "--", "lw": 1.3, "label": "Experience"},
    "agency":     {"ls": ":",  "lw": 1.3, "label": "Agency"},
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_model_data(model_key):
    """Load all gray_simple data for a single model. Returns dict or None."""
    try:
        set_model(model_key)
        ddir = data_dir("gray_entities", "neural", "with_self")

        # RSA results
        rsa_path = ddir / "rsa_results.json"
        if not rsa_path.exists():
            return None
        with open(rsa_path) as f:
            rsa = json.load(f)

        # RDM data
        rdm_path = ddir / "rdm_cosine_per_layer.npz"
        rdm_data = np.load(rdm_path, allow_pickle=True)

        # Neural PCA
        pca_path = ddir / "neural_pca_results.npz"
        pca_data = np.load(pca_path, allow_pickle=True)

        pca_json_path = ddir / "neural_pca_analysis.json"
        with open(pca_json_path) as f:
            pca_json = json.load(f)

        return {
            "rsa": rsa,
            "rdm": rdm_data,
            "pca": pca_data,
            "pca_json": pca_json,
        }
    except Exception as e:
        print(f"  Warning: could not load data for {model_key}: {e}")
        return None


def clean_rho(rho_val):
    """Replace NaN rho values with 0.0 for plotting."""
    if rho_val is None:
        return 0.0
    if isinstance(rho_val, float) and np.isnan(rho_val):
        return 0.0
    return float(rho_val)


def clean_p(p_val):
    """Replace NaN p-values with 1.0."""
    if p_val is None:
        return 1.0
    if isinstance(p_val, float) and np.isnan(p_val):
        return 1.0
    return float(p_val)


def get_peak_rsa(rsa_list):
    """Find the layer with highest rho from a list of RSA result dicts."""
    best = None
    for r in rsa_list:
        rho = clean_rho(r["rho"])
        if best is None or rho > clean_rho(best["rho"]):
            best = r
    return best


def sig_str(p):
    """Return significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report():
    """Build and write the cross-model gray_simple HTML report."""

    print("Loading data for all models...")
    model_data = OrderedDict()
    for mk in ALL_MODELS:
        d = load_model_data(mk)
        if d is not None:
            model_data[mk] = d
            print(f"  Loaded: {mk}")
        else:
            print(f"  Skipped: {mk} (no data)")

    if not model_data:
        print("ERROR: No model data found. Exiting.")
        return

    available = sort_models(list(model_data.keys()))
    fig_num = 0

    # -- Sections for TOC --
    sections = [
        {"id": "overview",          "label": "1. Overview"},
        {"id": "stimuli",           "label": "2. Stimuli"},
        {"id": "methods",           "label": "3. Methods"},
        {"id": "rsa-layerwise",     "label": "4. RSA Layerwise: All Models"},
        {"id": "rsa-overlay",       "label": "5. RSA Overlay: All Variants"},
        {"id": "rsa-peak-table",    "label": "6. Peak RSA Summary Table"},
        {"id": "rdm-peak",          "label": "7. RDM at Peak Layer"},
        {"id": "pca-disparity",     "label": "8. Neural PCA: Procrustes Disparity"},
        {"id": "pca-scatter",       "label": "9. Entity Scatter at Best Layer"},
        {"id": "instruction-tuning","label": "10. Instruction Tuning Effect"},
        {"id": "takeaways",         "label": "11. Key Takeaways"},
    ]

    html = build_cross_model_header("Experiment 4: Gray Simple — Cross-Model Summary")
    html += build_toc(sections)

    # ==================================================================
    # 1. OVERVIEW
    # ==================================================================
    html += '<h2 id="overview">1. Overview</h2>\n'
    html += '<div class="method">\n'
    html += ("<p>This report summarizes the <strong>Gray Simple</strong> branch of "
             "Experiment 4, which examines neural representations of 13 Gray et al. "
             "(2007) entities using minimal &ldquo;Think about {entity}&rdquo; prompts.</p>\n")
    html += "<p>For each model, last-token activations are extracted at every transformer "
    html += "layer, cosine-distance RDMs are computed, and two analyses are performed:</p>\n"
    html += "<ol>\n"
    html += ("<li><strong>RSA:</strong> Spearman correlation between each layer's "
             "model RDM and the human mind perception RDM derived from Gray et al. "
             "factor scores.</li>\n")
    html += ("<li><strong>Neural PCA:</strong> PCA of entity activations at each layer, "
             "Procrustes-rotated to the human 2D (Experience x Agency) space.</li>\n")
    html += "</ol>\n"
    html += (f"<p><strong>Models included:</strong> "
             f"{', '.join(MODEL_LABELS[m] for m in available)} "
             f"({len(available)} of 4).</p>\n")
    html += "</div>\n"

    # ==================================================================
    # 2. STIMULI
    # ==================================================================
    html += gray_entities_stimuli_html(include_capacities=False)

    # ==================================================================
    # 3. METHODS
    # ==================================================================
    html += '<h2 id="methods">3. Methods</h2>\n'
    html += '<div class="method">\n'
    html += "<h3>RSA (Representational Similarity Analysis)</h3>\n"
    html += ("<p>At each transformer layer, a 13x13 cosine-distance RDM is computed "
             "from entity activations. The upper triangle of this model RDM is correlated "
             "(Spearman rho) with the upper triangle of the human mind perception RDM "
             "derived from Gray et al.&rsquo;s Experience and Agency factor scores. "
             "Three human RDM variants are tested: <em>combined</em> (Euclidean distance "
             "in 2D Experience-Agency space), <em>experience-only</em>, and "
             "<em>agency-only</em>. P-values are uncorrected per layer.</p>\n")
    html += "<h3>Neural PCA</h3>\n"
    html += ("<p>PCA is applied to the 13 entity activation vectors at each layer, "
             "retaining the first 2 principal components. These 2D coordinates are then "
             "Procrustes-rotated (translation, rotation, reflection, scaling) to best "
             "match the human 2D layout (Experience, Agency). The Procrustes disparity "
             "quantifies how well the model&rsquo;s entity geometry matches the human "
             "space (lower = more similar).</p>\n")
    html += "</div>\n"

    # ── Interpretation Guide ──
    html += '<h3>Interpretation Guide</h3>\n'
    html += neural_methods_primer_html(
        include_layers=True, include_rdm=True, include_rsa=True,
        include_procrustes=True)
    html += methodology_primer_html(
        include_pca=False, include_spearman=True, include_fdr=True,
        include_prompting=False, include_pairwise=False)

    html += '<div class="method">\n'
    html += '<h4>Why Three RDM Variants?</h4>\n'
    html += (
        '<p>The <strong>combined</strong> RDM (2D Euclidean distance in '
        'Experience&ndash;Agency space) tests whether the model encodes the '
        'full mind perception geometry. The <strong>experience-only</strong> '
        'and <strong>agency-only</strong> variants test each dimension '
        'independently, revealing which aspect of mind perception the model '
        'encodes more strongly. If a model shows significant Agency-only RSA '
        'but not Experience-only, it means the model&rsquo;s entity '
        'representations capture &ldquo;what can it do&rdquo; more than '
        '&ldquo;what does it feel.&rdquo;</p>\n')
    html += '</div>\n'

    # ==================================================================
    # 4. RSA LAYERWISE: BAR CHARTS (colored = FDR significant)
    # ==================================================================
    html += '<h2 id="rsa-layerwise">4. RSA Layerwise: All Models</h2>\n'

    # Apply FDR correction to combined RSA for each model
    for mk in available:
        apply_fdr(model_data[mk]["rsa"]["combined"])

    # Determine shared y-range across all models (combined only for bars)
    all_rhos = []
    for mk in available:
        for r in model_data[mk]["rsa"]["combined"]:
            all_rhos.append(clean_rho(r["rho"]))
    y_min = min(all_rhos) - 0.05
    y_max = max(all_rhos) + 0.1

    positions, ordered, nrows, ncols, _ = make_model_grid(available)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)

    for idx, mk in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        rsa_combined = model_data[mk]["rsa"]["combined"]
        base_color = MODEL_COLORS[mk]

        layers = [r["layer"] for r in rsa_combined]
        rhos = [clean_rho(r["rho"]) for r in rsa_combined]
        qvals = [r.get("q_fdr", 1.0) for r in rsa_combined]

        # Gray bars default, colored if FDR-significant
        colors = [base_color if q < 0.05 else "#cccccc" for q in qvals]
        ax.bar(layers, rhos, color=colors, edgecolor="white", width=0.8)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Spearman ρ")
        ax.set_ylim(y_min, y_max)

        # Annotate peak
        peak = get_peak_rsa(rsa_combined)
        peak_rho = clean_rho(peak["rho"])
        peak_q = peak.get("q_fdr", 1.0)
        n_sig = sum(1 for q in qvals if q < 0.05)
        n_total = len(layers)
        ax.set_title(f"{MODEL_LABELS[mk]}\n{n_sig}/{n_total} layers q < .05, "
                     f"peak layer {peak['layer']} (ρ={peak_rho:.3f})",
                     fontsize=9)
        if peak["layer"] >= 0:
            ax.annotate(
                f"ρ={peak_rho:.3f}\nq={peak_q:.3f}",
                (peak["layer"], peak_rho),
                textcoords="offset points", xytext=(12, 8), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            )

    fig.suptitle("Combined RSA: Model RDM vs Human Mind Perception RDM",
                 fontsize=14)
    fig.tight_layout()
    fig_num += 1
    b64 = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64,
        "Layerwise RSA (combined Experience + Agency) for each model. "
        "Colored bars indicate FDR-significant layers (q &lt; .05); gray bars are "
        "non-significant. Peak layer annotated with ρ and q values.",
        fig_num=fig_num, alt="RSA bar charts"
    )

    # ==================================================================
    # 5. RSA OVERLAY: ALL VARIANTS
    # ==================================================================
    html += '<h2 id="rsa-overlay">5. RSA Overlay: All Variants</h2>\n'

    overlay_variants = [
        ("combined",    "Combined (Experience + Agency)", "Spearman rho (Combined RDM)"),
        ("experience",  "Experience Only",                "Spearman rho (Experience RDM)"),
        ("agency",      "Agency Only",                    "Spearman rho (Agency RDM)"),
    ]

    fig, overlay_axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for vi, (variant, v_title, v_ylabel) in enumerate(overlay_variants):
        ax = overlay_axes[vi]
        for mk in available:
            rsa_v = model_data[mk]["rsa"][variant]
            layers = [r["layer"] for r in rsa_v]
            rhos = [clean_rho(r["rho"]) for r in rsa_v]
            ax.plot(layers, rhos, color=MODEL_COLORS[mk], lw=2,
                    label=MODEL_LABELS[mk], alpha=0.85)

        ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax.set_xlabel("Layer")
        if vi == 0:
            ax.set_ylabel("Spearman rho")
        ax.set_title(v_title)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("RSA Overlay: All Models × All Variants", fontsize=14, y=1.02)
    fig.tight_layout()
    fig_num += 1
    b64 = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64,
        "RSA overlay for all models. Left: Combined (Experience + Agency) RDM. "
        "Center: Experience-only RDM. Right: Agency-only RDM. Each line is one "
        "model, allowing direct comparison of peak layer locations and magnitudes.",
        fig_num=fig_num, alt="RSA overlay all variants"
    )

    # ==================================================================
    # 6. PEAK RSA SUMMARY TABLE (3 side-by-side tables)
    # ==================================================================
    html += '<h2 id="rsa-peak-table">6. Peak RSA Summary Table</h2>\n'

    html += ('<div style="display: flex; gap: 20px; flex-wrap: wrap; '
             'align-items: flex-start;">\n')

    for variant in ["combined", "experience", "agency"]:
        html += '<div style="flex: 1; min-width: 280px;">\n'
        html += f'<h3 style="margin-top: 0;">{variant.title()}</h3>\n'
        html += ('<table style="width: 100%;">\n'
                 '<tr><th>Model</th><th>Peak Layer</th>'
                 '<th>Peak rho</th><th>p</th><th>Sig.</th>'
                 '<th># Sig Layers</th></tr>\n')

        for mk in available:
            rsa = model_data[mk]["rsa"]
            rsa_list = rsa[variant]
            peak = get_peak_rsa(rsa_list)
            peak_layer = peak["layer"]
            peak_rho = clean_rho(peak["rho"])
            peak_p = clean_p(peak["p_value"])
            n_sig = sum(1 for r in rsa_list if clean_p(r["p_value"]) < 0.05)
            sig = sig_str(peak_p)
            sig_class = ' class="sig"' if peak_p < 0.05 else ''

            html += (f'<tr>{model_row_td(mk)}'
                     f'<td>{peak_layer}</td>'
                     f'<td>{peak_rho:.4f}</td>'
                     f'<td>{peak_p:.4f}</td>'
                     f'<td{sig_class}>{sig}</td>'
                     f'<td>{n_sig}</td></tr>\n')

        html += '</table>\n</div>\n'

    html += '</div>\n'

    # ==================================================================
    # 7. RDM AT PEAK LAYER
    # ==================================================================
    html += '<h2 id="rdm-peak">7. RDM at Peak Layer</h2>\n'

    positions, ordered, nrows_rdm, ncols_rdm, human_pos = make_model_grid(available, include_human=True)
    fig, axes = plt.subplots(nrows_rdm, ncols_rdm, figsize=(5 * ncols_rdm, 5 * nrows_rdm), squeeze=False, layout="constrained")
    for ax in axes.flatten():
        ax.set_visible(False)

    # Collect all RDMs to find shared color range
    all_rdm_vals = []
    peak_layers = {}
    for mk in ordered:
        peak = get_peak_rsa(model_data[mk]["rsa"]["combined"])
        peak_layers[mk] = peak["layer"]
        model_rdm = model_data[mk]["rdm"]["model_rdm"]
        all_rdm_vals.append(model_rdm[peak["layer"]])

    # Get human RDM from first available model
    first_mk = ordered[0]
    set_model(first_mk)
    rdm_data = model_data[first_mk]["rdm"]
    human_rdm = rdm_data["human_rdm_combined"]
    entity_keys = list(rdm_data["entity_keys"])
    nice_labels = [nice_entity(e) for e in entity_keys]
    all_rdm_vals.append(human_rdm)

    vmin = min(v.min() for v in all_rdm_vals)
    vmax = max(v.max() for v in all_rdm_vals)

    # Human RDM panel
    ax_human = axes[human_pos[0], human_pos[1]]
    ax_human.set_visible(True)
    im = ax_human.imshow(human_rdm, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax_human.set_xticks(range(len(entity_keys)))
    ax_human.set_xticklabels(nice_labels, rotation=90, fontsize=7)
    ax_human.set_yticks(range(len(entity_keys)))
    ax_human.set_yticklabels(nice_labels, fontsize=7)
    ax_human.set_title("Human\n(Gray et al.)", fontsize=10)

    # Model RDM panels
    visible_axes = [ax_human]
    for idx, mk in enumerate(ordered):
        row, col = positions[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        visible_axes.append(ax)
        layer = peak_layers[mk]
        model_rdm = model_data[mk]["rdm"]["model_rdm"][layer]
        peak_rho = clean_rho(get_peak_rsa(model_data[mk]["rsa"]["combined"])["rho"])

        im = ax.imshow(model_rdm, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(entity_keys)))
        ax.set_xticklabels(nice_labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(entity_keys)))
        ax.set_yticklabels(nice_labels, fontsize=7)
        ax.set_title(f"{MODEL_LABELS[mk]}\nLayer {layer} (rho={peak_rho:.3f})", fontsize=10)

    fig.colorbar(im, ax=visible_axes, shrink=0.7, label="Dissimilarity")
    fig.suptitle("RDMs at Peak RSA Layer", fontsize=14)
    fig_num += 1
    b64 = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64,
        "Representational dissimilarity matrices. Leftmost panel is the human reference "
        "RDM (combined Experience + Agency factor distance). Remaining panels show each "
        "model's cosine-distance RDM at the layer with peak combined RSA.",
        fig_num=fig_num, alt="RDM at peak layer"
    )

    # ==================================================================
    # 8. NEURAL PCA: PROCRUSTES DISPARITY
    # ==================================================================
    html += '<h2 id="pca-disparity">8. Neural PCA: Procrustes Disparity</h2>\n'

    fig, ax = plt.subplots(figsize=(10, 5))
    for mk in available:
        disparity = model_data[mk]["pca"]["procrustes_disparity"]
        layers = np.arange(len(disparity))
        ax.plot(layers, disparity, color=MODEL_COLORS[mk], lw=2,
                label=MODEL_LABELS[mk], alpha=0.85)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Procrustes Disparity")
    ax.set_title("Neural PCA: Procrustes Disparity Across Layers\n"
                 "(Lower = more similar to human 2D layout)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig_num += 1
    b64 = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64,
        "Procrustes disparity between each model's PCA-derived 2D entity layout and "
        "the human Experience x Agency space, across all transformer layers. Lower "
        "disparity indicates better match to human geometry.",
        fig_num=fig_num, alt="Procrustes disparity"
    )

    # ==================================================================
    # 9. ENTITY SCATTER AT BEST LAYER
    # ==================================================================
    html += '<h2 id="pca-scatter">9. Entity Scatter at Best Layer</h2>\n'

    # Get human reference coordinates from Gray et al. scores
    human_exp = np.array([GRAY_ET_AL_SCORES[e][0] for e in entity_keys])
    human_ag = np.array([GRAY_ET_AL_SCORES[e][1] for e in entity_keys])

    # Purple gradient: entity color based on human combined score (dark=high, light=low)
    cmap = plt.cm.Purples
    dists = np.sqrt(human_exp ** 2 + human_ag ** 2)
    d_min, d_max = dists.min(), dists.max()
    d_rng = d_max - d_min if d_max > d_min else 1.0
    entity_colors = [cmap(0.15 + 0.75 * (d - d_min) / d_rng) for d in dists]
    # Build lookup by entity key for model panels
    entity_color_map = {e: entity_colors[i] for i, e in enumerate(entity_keys)}

    positions_sc, ordered_sc, nrows_sc, ncols_sc, human_pos_sc = make_model_grid(available, include_human=True)
    panel_size = 6
    fig, axes = plt.subplots(nrows_sc, ncols_sc, figsize=(panel_size * ncols_sc, panel_size * nrows_sc), squeeze=False)
    for ax in axes.flatten():
        ax.set_visible(False)

    # Human reference panel
    ax_human = axes[human_pos_sc[0], human_pos_sc[1]]
    ax_human.set_visible(True)
    ax_human.scatter(human_ag, human_exp, s=80, c=entity_colors, edgecolor="white",
                     zorder=5, linewidth=0.5)
    for i, ek in enumerate(entity_keys):
        ax_human.annotate(nice_entity(ek), (human_ag[i], human_exp[i]),
                          textcoords="offset points", xytext=(5, 4), fontsize=8.5)
    ax_human.set_xlabel("Agency")
    ax_human.set_ylabel("Experience")
    ax_human.set_title("Human\n(Gray et al., 2007)", fontsize=11)
    ax_human.set_aspect("equal")

    # Model panels: Procrustes-rotated PCA at best layer (same entity colors)
    for idx, mk in enumerate(ordered_sc):
        row, col = positions_sc[idx]
        ax = axes[row, col]
        ax.set_visible(True)
        disparity = model_data[mk]["pca"]["procrustes_disparity"]
        best_layer = int(np.argmin(disparity))
        best_disp = disparity[best_layer]
        coords = model_data[mk]["pca"]["procrustes_coords"][best_layer]  # (13, 2)
        m_colors = [entity_color_map.get(e, "#999999") for e in entity_keys]

        ax.scatter(coords[:, 0], coords[:, 1], s=80, c=m_colors,
                   edgecolor="white", zorder=5, linewidth=0.5)
        for i, ek in enumerate(entity_keys):
            ax.annotate(nice_entity(ek), (coords[i, 0], coords[i, 1]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8.5)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title(f"{MODEL_LABELS[mk]}\nLayer {best_layer} (disp={best_disp:.3f})",
                     fontsize=10)
        ax.set_aspect("equal")

    fig.suptitle("Entity Positions: Human Reference vs Procrustes-Rotated Neural PCA",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig_num += 1
    b64 = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64,
        "Entity scatter plots. Left: human reference (Agency x Experience from Gray "
        "et al.). Remaining panels: model PCA coordinates at the layer with minimum "
        "Procrustes disparity, rotated to best match the human layout.",
        fig_num=fig_num, alt="Entity scatter at best PCA layer"
    )

    # ==================================================================
    # 10. INSTRUCTION TUNING EFFECT
    # ==================================================================
    html += '<h2 id="instruction-tuning">10. Instruction Tuning Effect</h2>\n'

    families = OrderedDict()
    for fam_name, base_key, chat_key in INSTRUCTION_TUNING_PAIRS:
        families[fam_name] = {"base": base_key, "chat": chat_key}

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(families)), 5))

    bar_width = 0.3
    x_positions = []
    x_labels = []
    bars_base = []
    bars_chat = []
    x_idx = 0

    for fam_name, fam_models in families.items():
        base_mk = fam_models["base"]
        chat_mk = fam_models["chat"]

        base_rho = 0.0
        chat_rho = 0.0

        if base_mk in model_data:
            peak = get_peak_rsa(model_data[base_mk]["rsa"]["combined"])
            base_rho = clean_rho(peak["rho"])
        if chat_mk in model_data:
            peak = get_peak_rsa(model_data[chat_mk]["rsa"]["combined"])
            chat_rho = clean_rho(peak["rho"])

        bars_base.append(base_rho)
        bars_chat.append(chat_rho)
        x_positions.append(x_idx)
        x_labels.append(fam_name)
        x_idx += 1

    x_arr = np.array(x_positions, dtype=float)
    base_bars = ax.bar(x_arr - bar_width / 2, bars_base, bar_width,
                       label="Base", color=[MODEL_COLORS.get(families[f]["base"], "#999")
                                            for f in families],
                       edgecolor="white")
    chat_bars = ax.bar(x_arr + bar_width / 2, bars_chat, bar_width,
                       label="Chat/Instruct", color=[MODEL_COLORS.get(families[f]["chat"], "#999")
                                                     for f in families],
                       edgecolor="white")

    # Annotate bar values
    for bar in list(base_bars) + list(chat_bars):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_arr)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Peak Combined RSA (Spearman rho)")
    ax.set_title("Instruction Tuning Effect on Peak RSA")
    ax.legend(loc="upper right")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig_num += 1
    b64 = fig_to_b64(fig)
    plt.close(fig)
    html += html_figure(
        b64,
        "Peak combined RSA (Spearman rho) for base vs chat/instruct models within "
        "each model family. Higher bars indicate better alignment between the model's "
        "entity geometry and human mind perception structure.",
        fig_num=fig_num, alt="Instruction tuning effect"
    )

    # ==================================================================
    # 11. KEY TAKEAWAYS
    # ==================================================================
    html += '<h2 id="takeaways">11. Key Takeaways</h2>\n'
    html += '<div class="interpret">\n'
    html += "<ul>\n"

    # Find highest peak RSA model
    best_model = None
    best_rho = -999
    for mk in available:
        peak = get_peak_rsa(model_data[mk]["rsa"]["combined"])
        rho = clean_rho(peak["rho"])
        if rho > best_rho:
            best_rho = rho
            best_model = mk

    html += (f"<li><strong>Highest peak RSA:</strong> {MODEL_LABELS[best_model]} "
             f"achieves the highest combined RSA (rho = {best_rho:.4f}).</li>\n")

    # Instruction tuning effect
    for fam_name, fam_models in families.items():
        base_mk = fam_models["base"]
        chat_mk = fam_models["chat"]
        if base_mk in model_data and chat_mk in model_data:
            base_peak = clean_rho(get_peak_rsa(model_data[base_mk]["rsa"]["combined"])["rho"])
            chat_peak = clean_rho(get_peak_rsa(model_data[chat_mk]["rsa"]["combined"])["rho"])
            diff = chat_peak - base_peak
            direction = "increases" if diff > 0 else "decreases"
            html += (f"<li><strong>{fam_name}:</strong> Instruction tuning "
                     f"{direction} peak RSA by {abs(diff):.4f} "
                     f"(base: {base_peak:.4f}, chat: {chat_peak:.4f}).</li>\n")

    # Peak layer regions
    peak_layers_summary = []
    for mk in available:
        peak = get_peak_rsa(model_data[mk]["rsa"]["combined"])
        n_layers_total = len(model_data[mk]["rsa"]["combined"])
        frac = peak["layer"] / (n_layers_total - 1)
        peak_layers_summary.append((mk, peak["layer"], n_layers_total, frac))

    if peak_layers_summary:
        fracs = [f for _, _, _, f in peak_layers_summary]
        mean_frac = np.mean(fracs)
        layer_strs = [f"{MODEL_LABELS[mk]}: layer {l}/{n-1} ({f:.0%})"
                      for mk, l, n, f in peak_layers_summary]
        html += (f"<li><strong>Peak layer region:</strong> Average peak at "
                 f"{mean_frac:.0%} of network depth. "
                 f"{'; '.join(layer_strs)}.</li>\n")

    # Procrustes best
    best_pca_model = None
    best_disp = 999
    for mk in available:
        disparity = model_data[mk]["pca"]["procrustes_disparity"]
        min_disp = float(np.min(disparity))
        if min_disp < best_disp:
            best_disp = min_disp
            best_pca_model = mk
            best_pca_layer = int(np.argmin(disparity))

    if best_pca_model is not None:
        html += (f"<li><strong>Best Procrustes match:</strong> {MODEL_LABELS[best_pca_model]} "
                 f"at layer {best_pca_layer} (disparity = {best_disp:.4f}) has entity "
                 f"positions most similar to the human 2D mind perception space.</li>\n")

    # Significant layers count
    for mk in available:
        n_sig = sum(1 for r in model_data[mk]["rsa"]["combined"]
                    if clean_p(r["p_value"]) < 0.05)
        n_total = len(model_data[mk]["rsa"]["combined"])
        html += (f"<li><strong>{MODEL_LABELS[mk]}:</strong> {n_sig}/{n_total} layers "
                 f"with significant combined RSA (p &lt; 0.05).</li>\n")

    html += "</ul>\n"
    html += "</div>\n"

    # -- Footer --
    html += build_html_footer()

    # -- Write --
    out_dir = ensure_dir(COMPARISONS_DIR)
    report_path = out_dir / "gray_simple_summary.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\nReport written to: {report_path}")
    return report_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 4: Gray Simple — Cross-Model Summary")
    print("=" * 60)
    generate_report()
