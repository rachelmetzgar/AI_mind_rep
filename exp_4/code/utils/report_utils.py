#!/usr/bin/env python3
"""
Shared HTML Report Utilities

Common CSS, figure encoding, table-of-contents builder, and HTML scaffolding
used by all Experiment 4 report generators.

Rachel C. Metzgar · Mar 2026
"""

import base64
import argparse
from io import BytesIO
from datetime import datetime


# ============================================================================
# SHARED CSS
# ============================================================================

# ============================================================================
# MODEL COLORS AND LABELS (cross-model reports)
# ============================================================================

MODEL_COLORS = {
    "llama2_13b_chat": "#e41a1c",
    "llama2_13b_base": "#377eb8",
    "llama3_8b_instruct": "#ff7f00",
    "llama3_8b_base": "#984ea3",
}

MODEL_LABELS = {
    "llama2_13b_chat": "LLaMA-2-13B Chat",
    "llama2_13b_base": "LLaMA-2-13B Base",
    "llama3_8b_instruct": "LLaMA-3-8B Instruct",
    "llama3_8b_base": "LLaMA-3-8B Base",
}

ALL_MODELS = list(MODEL_LABELS.keys())


REPORT_CSS = """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           max-width: 1100px; margin: 40px auto; padding: 0 20px;
           color: #333; line-height: 1.6; }
    h1 { border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
    h2 { color: #1565C0; margin-top: 40px; }
    h3 { color: #555; }
    img { max-width: 100%; border: 1px solid #eee; border-radius: 4px;
           margin: 10px 0; }
    .stat { background: #f5f5f5; padding: 15px; border-radius: 6px;
             margin: 10px 0; }
    .stat strong { color: #1565C0; }
    .method { background: #f8f9ff; padding: 20px; border-left: 4px solid #2196F3;
              border-radius: 0 6px 6px 0; margin: 15px 0; }
    .method ol { margin: 10px 0; padding-left: 20px; }
    .method li { margin: 6px 0; }
    .interpret { background: #fff8e1; padding: 15px; border-left: 4px solid #FFC107;
                 border-radius: 0 6px 6px 0; margin: 15px 0; }
    table { border-collapse: collapse; margin: 15px 0; }
    th, td { padding: 6px 12px; border: 1px solid #ddd; text-align: right; }
    th { background: #f0f0f0; }
    td:first-child, th:first-child { text-align: left; }
    .sig { color: #E53935; font-weight: bold; }
    .success { background: #e8f5e9; padding: 12px; border-left: 4px solid #4CAF50;
               border-radius: 0 6px 6px 0; margin: 10px 0; }
    .warning { background: #fff3e0; padding: 12px; border-left: 4px solid #FF9800;
               border-radius: 0 6px 6px 0; margin: 10px 0; }
    .match { color: #2E7D32; font-weight: bold; }
    .mismatch { color: #C62828; font-weight: bold; }
    nav.toc { background: #f8f9ff; border: 1px solid #e0e0e0; border-radius: 6px;
              padding: 15px 20px; margin: 15px 0; }
    nav.toc ol { list-style: none; padding: 0; columns: 2; column-gap: 2rem; }
    nav.toc li { padding: 2px 0; font-size: 0.9rem; }
    nav.toc li a { color: #1565C0; text-decoration: none; }
    nav.toc li a:hover { text-decoration: underline; }
    figure { margin: 1.5rem 0; text-align: center; }
    figure img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }
    figcaption { font-size: 0.9rem; color: #555; margin-top: 8px;
                 text-align: left; padding: 0 1rem; line-height: 1.5; }
    figcaption strong { color: #333; }
"""


# ============================================================================
# FIGURE ENCODING
# ============================================================================

def fig_to_b64(fig, dpi=150):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return b64


# ============================================================================
# TABLE OF CONTENTS
# ============================================================================

def build_toc(sections):
    """Build an HTML table-of-contents nav block.

    Args:
        sections: list of {"id": "anchor-id", "label": "Display text"} dicts.

    Returns:
        HTML string for the ToC nav element.
    """
    items = []
    for s in sections:
        items.append(f'<li><a href="#{s["id"]}">{s["label"]}</a></li>')
    return (
        '<nav class="toc">\n'
        '<strong>Contents</strong>\n'
        '<ol>\n' + '\n'.join(items) + '\n</ol>\n'
        '</nav>\n'
    )


# ============================================================================
# HTML HEADER / FOOTER
# ============================================================================

def build_html_header(title, model_label, css=None):
    """Return the opening HTML through <body>, including timestamp.

    Args:
        title: page <title> text.
        model_label: model name shown in the <h1>.
        css: optional extra CSS to append after REPORT_CSS. If None, only
             REPORT_CSS is used.
    """
    full_css = REPORT_CSS
    if css:
        full_css += "\n" + css
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return (
        f'<!DOCTYPE html>\n<html><head>\n'
        f'<meta charset="utf-8">\n'
        f'<title>{title} — {model_label}</title>\n'
        f'<style>{full_css}</style>\n'
        f'</head><body>\n'
        f'<h1>{title} — {model_label}</h1>\n'
        f'<p>Generated: {now}</p>\n'
    )


def build_cross_model_header(title, css=None):
    """Return opening HTML for a cross-model report (no model in title)."""
    full_css = REPORT_CSS
    if css:
        full_css += "\n" + css
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return (
        f'<!DOCTYPE html>\n<html><head>\n'
        f'<meta charset="utf-8">\n'
        f'<title>{title}</title>\n'
        f'<style>{full_css}</style>\n'
        f'</head><body>\n'
        f'<h1>{title}</h1>\n'
        f'<p>Generated: {now}</p>\n'
    )


def build_html_footer():
    """Return the closing </body></html> tags."""
    return '</body></html>'


# ============================================================================
# FIGURE WITH CAPTION
# ============================================================================

def html_figure(b64_data, caption, fig_num=None, alt=""):
    """Wrap a base64-encoded image in <figure>/<figcaption>.

    Args:
        b64_data: base64 PNG string.
        caption: caption text (may contain HTML). If fig_num is provided,
                 the caption is prefixed with "Figure N. ".
        fig_num: optional figure number (int).
        alt: alt text for the image.
    """
    prefix = f"<strong>Figure {fig_num}.</strong> " if fig_num else ""
    return (
        f'<figure>\n'
        f'  <img src="data:image/png;base64,{b64_data}" alt="{alt}">\n'
        f'  <figcaption>{prefix}{caption}</figcaption>\n'
        f'</figure>\n'
    )


# ============================================================================
# STIMULI SECTIONS
# ============================================================================

def gray_entities_stimuli_html(include_capacities=True):
    """Return HTML for a 'Stimuli' section listing the 13 Gray et al. entities.

    Includes entity names, descriptions, activation prompts, and human factor
    scores. Optionally includes the 18 mental capacity survey items.
    """
    from entities.gray_entities import (
        GRAY_ET_AL_SCORES, ENTITY_PROMPTS, ENTITY_NAMES,
        CHARACTER_NAMES, CHARACTER_DESCRIPTIONS, CAPACITY_PROMPTS,
    )

    html = '<h2 id="stimuli">Stimuli</h2>\n'
    html += '<h3>Entities</h3>\n'
    html += ('<p>13 entities from Gray, Gray, &amp; Wegner (2007), spanning '
             'biological organisms, artifacts, and supernatural beings. '
             'Human factor scores (Experience, Agency) estimated from the '
             'original Figure 1.</p>\n')
    html += ('<table>\n'
             '<tr><th>Entity</th><th>Name</th><th>Description</th>'
             '<th>Prompt</th>'
             '<th>Experience</th><th>Agency</th></tr>\n')
    for key in ENTITY_NAMES:
        name = CHARACTER_NAMES.get(key, key)
        desc = CHARACTER_DESCRIPTIONS.get(key, "")
        prompt = ENTITY_PROMPTS.get(key, "")
        exp, ag = GRAY_ET_AL_SCORES[key]
        html += (f'<tr><td>{key}</td><td>{name}</td>'
                 f'<td style="font-size:0.85em">{desc}</td>'
                 f'<td style="font-size:0.85em">{prompt}</td>'
                 f'<td>{exp:.2f}</td><td>{ag:.2f}</td></tr>\n')
    html += '</table>\n'

    if include_capacities:
        html += '<h3>Mental Capacities (18 items)</h3>\n'
        html += ('<p>Pairwise comparison items from Gray et al. Each completes: '
                 '&ldquo;Which character is more capable of &hellip;&rdquo;</p>\n')
        html += ('<table>\n'
                 '<tr><th>Capacity</th><th>Factor</th><th>Question</th></tr>\n')
        for cap, (prompt, factor) in CAPACITY_PROMPTS.items():
            factor_label = "Experience" if factor == "E" else "Agency"
            html += (f'<tr><td>{cap}</td><td>{factor_label}</td>'
                     f'<td style="font-size:0.85em">{prompt}</td></tr>\n')
        html += '</table>\n'

    return html


def characters_stimuli_html(include_capacities=True, characters=None):
    """Return HTML for a 'Stimuli' section listing the 30 AI/human characters.

    Includes character names, types, descriptions, and activation prompts.
    Optionally includes the 18 mental capacity survey items.
    """
    from entities.characters import (
        CHARACTER_INFO, AI_CHARACTERS, HUMAN_CHARACTERS,
        CHARACTER_PROMPTS, ALL_CHARACTERS,
    )
    from entities.gray_entities import CAPACITY_PROMPTS

    char_keys = characters if characters is not None else ALL_CHARACTERS

    html = '<h2 id="stimuli">Stimuli</h2>\n'
    html += '<h3>Characters</h3>\n'

    ai_in = [k for k in char_keys if k in AI_CHARACTERS]
    human_in = [k for k in char_keys if k in HUMAN_CHARACTERS]
    html += (f'<p>{len(char_keys)} characters: {len(ai_in)} AI + '
             f'{len(human_in)} human. AI characters are identified by their '
             'descriptions as artificial systems. Human characters have '
             'naturalistic bios without explicit &ldquo;is a human&rdquo; '
             'labeling.</p>\n')

    html += ('<table>\n'
             '<tr><th>Key</th><th>Name</th><th>Type</th>'
             '<th>Description</th><th>Prompt</th></tr>\n')
    # AI first, then human
    for key in ai_in + human_in:
        info = CHARACTER_INFO[key]
        prompt = CHARACTER_PROMPTS.get(key, "")
        type_label = info["type"].upper()
        html += (f'<tr><td>{key}</td><td>{info["name"]}</td>'
                 f'<td>{type_label}</td>'
                 f'<td style="font-size:0.85em">{info["description"]}</td>'
                 f'<td style="font-size:0.85em">{prompt}</td></tr>\n')
    html += '</table>\n'

    if include_capacities:
        html += '<h3>Mental Capacities (18 items)</h3>\n'
        html += ('<p>Pairwise comparison items from Gray et al. (2007). Each '
                 'completes: &ldquo;Which character is more capable of '
                 '&hellip;&rdquo;</p>\n')
        html += ('<table>\n'
                 '<tr><th>Capacity</th><th>Factor</th><th>Question</th></tr>\n')
        for cap, (prompt, factor) in CAPACITY_PROMPTS.items():
            factor_label = "Experience" if factor == "E" else "Agency"
            html += (f'<tr><td>{cap}</td><td>{factor_label}</td>'
                     f'<td style="font-size:0.85em">{prompt}</td></tr>\n')
        html += '</table>\n'

    return html


def expanded_concepts_stimuli_html(characters=None):
    """Return HTML for a 'Stimuli' section listing concept dimensions + characters.

    Includes concept names, pairwise prompts, expected directions, plus the
    character table.
    """
    from expanded_mental_concepts.concepts import (
        CONCEPT_NAMES, CONCEPT_DIMENSIONS, CONCEPT_KEYS,
        PAIRWISE_PROMPTS, CONCEPT_DIRECTION, N_CONCEPTS,
    )
    from entities.characters import (
        CHARACTER_INFO, AI_CHARACTERS, HUMAN_CHARACTERS,
        CHARACTER_PROMPTS, ALL_CHARACTERS,
    )

    char_keys = characters if characters is not None else ALL_CHARACTERS

    html = '<h2 id="stimuli">Stimuli</h2>\n'

    # Concept dimensions
    html += '<h3>Concept Dimensions</h3>\n'
    html += (f'<p>{N_CONCEPTS} concept dimensions bridging Exp 3 mental '
             'concept vectors into the mind perception space.</p>\n')
    html += ('<table>\n'
             '<tr><th>Dim</th><th>Concept</th><th>Direction</th>'
             '<th>Pairwise Prompt</th></tr>\n')
    for key in CONCEPT_KEYS:
        dim_info = CONCEPT_DIMENSIONS[key]
        dim_id = dim_info["id"]
        name = CONCEPT_NAMES.get(dim_id, key)
        direction = CONCEPT_DIRECTION.get(dim_id, "?")
        prompt = PAIRWISE_PROMPTS.get(dim_id, "")
        html += (f'<tr><td>{dim_id}</td><td>{name}</td>'
                 f'<td>{direction}</td>'
                 f'<td style="font-size:0.85em">{prompt}</td></tr>\n')
    html += '</table>\n'

    # Characters
    ai_in = [k for k in char_keys if k in AI_CHARACTERS]
    human_in = [k for k in char_keys if k in HUMAN_CHARACTERS]
    html += '<h3>Characters</h3>\n'
    html += (f'<p>{len(char_keys)} characters: {len(ai_in)} AI + '
             f'{len(human_in)} human.</p>\n')
    html += ('<table>\n'
             '<tr><th>Key</th><th>Name</th><th>Type</th>'
             '<th>Description</th><th>Prompt</th></tr>\n')
    for key in ai_in + human_in:
        info = CHARACTER_INFO[key]
        prompt = CHARACTER_PROMPTS.get(key, "")
        type_label = info["type"].upper()
        html += (f'<tr><td>{key}</td><td>{info["name"]}</td>'
                 f'<td>{type_label}</td>'
                 f'<td style="font-size:0.85em">{info["description"]}</td>'
                 f'<td style="font-size:0.85em">{prompt}</td></tr>\n')
    html += '</table>\n'

    return html


# ============================================================================
# DATASET ARGUMENT
# ============================================================================

def add_dataset_argument(parser, default="full_dataset"):
    """Add a --dataset argument to an argparse parser.

    Controls whether the report is written into a full_dataset/ or
    reduced_dataset/ subfolder under the normal results directory.
    """
    parser.add_argument(
        "--dataset", type=str, default=default,
        choices=["full_dataset", "reduced_dataset"],
        help=f"Dataset subfolder for output (default: {default})"
    )
