#!/usr/bin/env python3
"""
Experiment 4: Cross-Model Summary Report

Wraps the 10 publication figures from 1_behavioral_summary_figures_generator.py
into an HTML report with narrative context.

Reads pre-generated PNG files from results/comparisons/figures/.
Run on login node (CPU only, instant).

Usage:
    python comparisons/1a_behavioral_summary_report_generator.py

Env: llama2_env
Rachel C. Metzgar · Mar 2026
"""

import sys
import base64
from pathlib import Path
from datetime import datetime

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, COMPARISONS_DIR, ensure_dir
from utils.report_utils import (
    REPORT_CSS, build_toc, build_html_header, build_html_footer,
    gray_entities_stimuli_html,
)


# ========================== CONFIG ========================== #

FIG_DIR = COMPARISONS_DIR / "figures"

FIGURES = [
    {
        "file": "fig1_scree_plot.png",
        "title": "Eigenvalue Comparison",
        "caption": (
            "Scree plot comparing eigenvalue structure from the model's pairwise "
            "PCA (LLaMA-2-13B base, 13 entities, 18 capacities) against Gray et al.'s "
            "human data. Both show a dominant first factor, with the model's eigenvalue "
            "structure closely paralleling the two-factor solution from human mind "
            "perception research."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig2_loading_comparison.png",
        "title": "Factor Loading Comparison",
        "caption": (
            "Varimax-rotated capacity loadings from the base model's pairwise ratings. "
            "Capacities are colored by their human factor assignment (blue = Experience, "
            "red = Agency). The model's two factors broadly separate Experience and Agency "
            "capacities, mirroring the human factorial structure."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig3_entity_scatter_pairwise.png",
        "title": "Entity Scatter (Pairwise)",
        "caption": (
            "Model factor scores vs human Experience and Agency ratings. Left: Model F2 "
            "vs Human Experience. Right: Model F2 vs Human Agency. The significant "
            "correlation with Experience suggests the model's second factor captures "
            "experiential capacity differences between entities."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig4_mind_space_comparison.png",
        "title": "Mind Perception Space",
        "caption": (
            "Side-by-side 2D entity maps. Left: Human mind perception space from "
            "Gray et al. (2007) with Agency on X and Experience on Y. Right: Model's "
            "factor space from pairwise ratings. Entity positions show qualitative "
            "similarities — high-agency entities (man, woman) cluster together in both "
            "spaces."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig5_rating_heatmap.png",
        "title": "Individual Rating Heatmap",
        "caption": (
            "Heatmap of expected individual Likert ratings (capacities x entities) from "
            "the base model's logit distribution. Ratings near 3.0 indicate uncertainty; "
            "deviations reveal the model's capacity attributions. Factor labels (E/A) "
            "from the human factor assignment."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig6_individual_entity_scatter.png",
        "title": "Individual Ratings vs Human Agency",
        "caption": (
            "Model Factor 1 from individual ratings vs Human Agency scores. Left: "
            "with self-entity (13 entities). Right: without self (12 entities). The "
            "without-self condition shows a stronger correlation, suggesting the self "
            "entity is an outlier in the model's representational space."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig7_rsa_layerwise.png",
        "title": "RSA Across Layers",
        "caption": (
            "Representational Similarity Analysis across transformer layers (chat model). "
            "Each bar shows the Spearman correlation between the model's activation-based "
            "RDM and the human mind perception RDM at that layer. Green bars are "
            "significant (p < .05). The pattern reveals where in the network entity "
            "representations align with human mind perception geometry."
        ),
        "section": "internals",
    },
    {
        "file": "fig8_correlation_summary.png",
        "title": "Model-Human Alignment Summary",
        "caption": (
            "Summary of all model-human Spearman correlations across methods: pairwise "
            "factor scores, individual factor scores, and RSA at peak layer. Green bars "
            "are significant (p < .05). This provides a comprehensive view of where and "
            "how strongly the model's representations align with human mind perception."
        ),
        "section": "summary",
    },
    {
        "file": "fig9_pairwise_heatmap.png",
        "title": "Pairwise Character Means",
        "caption": (
            "Heatmap of pairwise-derived character means (capacities x entities) from "
            "the base model. Each cell shows the mean relative rating for that "
            "entity-capacity pair across all pairwise comparisons. This is the raw data "
            "underlying the PCA in Figures 1-4."
        ),
        "section": "behavioral",
    },
    {
        "file": "fig10_rdm_comparison.png",
        "title": "RDM Comparison",
        "caption": (
            "Side-by-side representational dissimilarity matrices. Left: Human RDM "
            "derived from Gray et al. factor distances. Right: Model RDM from cosine "
            "distance between entity activations at the peak RSA layer. Similar structure "
            "in both matrices underlies the significant RSA correlation."
        ),
        "section": "internals",
    },
]


# ========================== HELPERS ========================== #

def load_fig_b64(fig_name):
    """Load a PNG figure and return base64 string."""
    fig_path = FIG_DIR / fig_name
    if not fig_path.exists():
        return None
    with open(fig_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ========================== REPORT ========================== #

def generate_report():
    """Build HTML report wrapping all comparison figures."""

    sections = [
        {"id": "overview", "label": "Overview"},
        {"id": "stimuli", "label": "Stimuli"},
        {"id": "behavioral", "label": "Behavioral Analysis (Base Model)"},
        {"id": "internals", "label": "Internal Representations (Chat Model)"},
        {"id": "summary", "label": "Cross-Method Summary"},
    ]

    html = build_html_header("Experiment 4: Cross-Model Summary", "LLaMA-2-13B")
    html += build_toc(sections)

    # ── Overview ──
    html += '<h2 id="overview">Overview</h2>\n'
    html += '<div class="method">\n'
    html += "<p>Experiment 4 investigates whether LLMs replicate the two-dimensional "
    html += "structure of human mind perception (Gray et al., 2007). We test this at "
    html += "two levels:</p>\n"
    html += "<ol>\n"
    html += "<li><strong>Behavioral:</strong> PCA of the base model's pairwise and "
    html += "individual ratings of 13 entities on 18 mental capacities. Do the resulting "
    html += "factors mirror human Experience and Agency?</li>\n"
    html += "<li><strong>Internal:</strong> RSA of the chat model's entity activations. "
    html += "Does the internal representational geometry align with the human mind "
    html += "perception RDM?</li>\n"
    html += "</ol>\n"
    html += "<p>This report collects 10 publication-quality figures covering both levels.</p>\n"
    html += "</div>\n"

    # ── Stimuli ──
    html += gray_entities_stimuli_html(include_capacities=True)

    # ── Figures by section ──
    section_labels = {
        "behavioral": "Behavioral Analysis (Base Model)",
        "internals": "Internal Representations (Chat Model)",
        "summary": "Cross-Method Summary",
    }

    fig_num = 1
    current_section = None

    for fig_info in FIGURES:
        sec = fig_info["section"]
        if sec != current_section:
            current_section = sec
            html += f'<h2 id="{sec}">{section_labels[sec]}</h2>\n'

        b64 = load_fig_b64(fig_info["file"])
        if b64 is None:
            html += (f'<div class="warning"><p>Figure {fig_num} '
                     f'({fig_info["file"]}) not found.</p></div>\n')
        else:
            html += '<figure>\n'
            html += (f'  <img src="data:image/png;base64,{b64}" '
                     f'alt="{fig_info["title"]}">\n')
            html += (f'  <figcaption><strong>Figure {fig_num}.</strong> '
                     f'{fig_info["caption"]}</figcaption>\n')
            html += '</figure>\n'

        fig_num += 1

    html += build_html_footer()

    # ── Write ──
    out_dir = ensure_dir(COMPARISONS_DIR)
    report_path = out_dir / "behavioral_summary_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"Report written to: {report_path}")
    return report_path


# ========================== MAIN ========================== #

if __name__ == "__main__":
    generate_report()
