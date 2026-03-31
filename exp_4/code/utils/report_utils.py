#!/usr/bin/env python3
"""
Shared HTML Report Utilities

Common CSS, figure encoding, table-of-contents builder, and HTML scaffolding
used by all Experiment 4 report generators.

Rachel C. Metzgar · Mar 2026
"""

import base64
import argparse
import numpy as np
from io import BytesIO
from datetime import datetime


# ============================================================================
# SHARED CSS
# ============================================================================

# ============================================================================
# MODEL COLORS AND LABELS (cross-model reports)
# ============================================================================

MODEL_COLORS = {
    "llama2_13b_chat": "#5b9bd5",      # lighter blue (chat/instruct)
    "llama2_13b_base": "#8faabc",      # grayish blue (base)
    "llama3_8b_instruct": "#2a5fa5",   # darker blue (chat/instruct)
    "llama3_8b_base": "#5a7080",       # darker grayish blue (base)
    "gemma2_2b_it": "#e74c3c",         # bright red (chat, small)
    "gemma2_2b": "#b07a7a",            # muted rose (base, small)
    "gemma2_9b_it": "#c0392b",         # dark red (chat, large)
    "gemma2_9b": "#8b5e5e",            # muted dark rose (base, large)
    "qwen25_7b_instruct": "#b8860b",   # dark goldenrod (chat)
    "qwen25_7b": "#8b7355",            # muted brown (base)
    "qwen3_8b": "#d4a017",             # goldenrod
}

MODEL_LABELS = {
    "llama2_13b_chat": "LLaMA-2-13B Chat",
    "llama2_13b_base": "LLaMA-2-13B Base",
    "llama3_8b_instruct": "LLaMA-3-8B Instruct",
    "llama3_8b_base": "LLaMA-3-8B Base",
    "gemma2_2b_it": "Gemma-2-2B-IT",
    "gemma2_2b": "Gemma-2-2B Base",
    "gemma2_9b_it": "Gemma-2-9B-IT",
    "gemma2_9b": "Gemma-2-9B Base",
    "qwen25_7b_instruct": "Qwen-2.5-7B-Instruct",
    "qwen25_7b": "Qwen-2.5-7B Base",
    "qwen3_8b": "Qwen3-8B",
}

# Canonical display order: chat/instruct before base within each family
MODEL_ORDER = [
    "llama2_13b_chat",
    "llama2_13b_base",
    "llama3_8b_instruct",
    "llama3_8b_base",
    "gemma2_2b_it",
    "gemma2_2b",
    "gemma2_9b_it",
    "gemma2_9b",
    "qwen25_7b_instruct",
    "qwen25_7b",
    "qwen3_8b",
]

ALL_MODELS = list(MODEL_ORDER)

# Model families: each family = one row in multi-panel figures
MODEL_FAMILIES = [
    ("LLaMA", ["llama2_13b_chat", "llama2_13b_base", "llama3_8b_instruct", "llama3_8b_base"]),
    ("Gemma", ["gemma2_2b_it", "gemma2_2b", "gemma2_9b_it", "gemma2_9b"]),
    ("Qwen",  ["qwen25_7b_instruct", "qwen25_7b", "qwen3_8b"]),
]

# Max family size determines grid columns
GRID_NCOLS = 4


def sort_models(models):
    """Sort a list of model keys by canonical MODEL_ORDER."""
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: order.get(m, 999))


def make_model_grid(available_models, include_human=False):
    """Compute family-grouped grid positions for multi-panel figures.

    Each model family starts a new row.  If *include_human* is True the
    human reference panel occupies (0, 0) and model families start at row 1.

    Args:
        available_models: iterable of model keys that have data.
        include_human: whether to reserve a panel for a human reference.

    Returns:
        positions:      list of (row, col) per model, same order as ordered_models
        ordered_models:  model keys in display order
        nrows:          total grid rows
        ncols:          total grid columns (always GRID_NCOLS)
        human_pos:      (row, col) if include_human else None
    """
    ncols = GRID_NCOLS
    available_set = set(available_models)
    row = 0
    human_pos = None
    if include_human:
        human_pos = (0, 0)
        row = 1

    positions = []
    ordered_models = []
    for _family_name, family_members in MODEL_FAMILIES:
        family_avail = [m for m in family_members if m in available_set]
        if not family_avail:
            continue
        for col_idx, model in enumerate(family_avail):
            positions.append((row, col_idx))
            ordered_models.append(model)
        row += 1

    nrows = max(row, 1)
    return positions, ordered_models, nrows, ncols, human_pos


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
# METHODOLOGY PRIMER SECTIONS
# ============================================================================

def methodology_primer_html(include_pca=True, include_spearman=True,
                            include_fdr=False, include_prompting=True,
                            include_pairwise=True):
    """Return an HTML block explaining core behavioral analysis methods.

    Each subsection can be toggled on/off so generators include only what
    is relevant to their report.
    """
    html = '<div class="method">\n'
    html += '<h3>Methodology Primer</h3>\n'

    if include_pairwise:
        html += '<h4>Pairwise Comparison Paradigm</h4>\n'
        html += (
            '<p>In the pairwise paradigm, entities are compared two at a time '
            'on each mental capacity. The model is asked: '
            '&ldquo;Which is more capable of X: Entity A or Entity B?&rdquo; '
            'and rates on a 1&ndash;5 scale (1&nbsp;=&nbsp;A clearly more, '
            '5&nbsp;=&nbsp;B clearly more, 3&nbsp;=&nbsp;equal). Each pair is '
            'tested in both orders (A&ndash;B and B&ndash;A) to control for '
            'position bias.</p>\n'
            '<p><strong>Relative scores</strong> are computed by transforming '
            'pairwise ratings into per-entity win rates. For each entity&ndash;'
            'capacity combination, the relative score reflects how often and '
            'how strongly that entity was judged &ldquo;more capable&rdquo; '
            'than its comparison partners. These scores are aggregated into an '
            '<em>entity &times; capacity</em> matrix that serves as input to '
            'PCA.</p>\n'
        )

    if include_pca:
        html += '<h4>PCA with Varimax Rotation</h4>\n'
        html += (
            '<p><strong>Principal Component Analysis (PCA)</strong> finds the '
            'axes of maximum variance in multivariate data. Applied to the '
            'entity&ndash;capacity correlation matrix, it reveals latent '
            'dimensions that organize mental capacity judgments.</p>\n'
            '<p><strong>Varimax rotation</strong> is applied after PCA '
            'extraction to achieve &ldquo;simple structure&rdquo;&mdash;each '
            'capacity loads strongly on one factor and weakly on others, '
            'making the factors more interpretable. The rotation does not '
            'change the total variance explained; it only redistributes it '
            'across factors. Gray et al.&nbsp;(2007) used this exact '
            'procedure and found two factors they labeled '
            '<em>Experience</em> (capacity for sensations and feelings) and '
            '<em>Agency</em> (capacity for planning, self-control, and '
            'communication).</p>\n'
            '<p><strong>Kaiser criterion</strong>: Only factors with '
            'eigenvalue&nbsp;&gt;&nbsp;1 are retained. A factor must account '
            'for more variance than a single original variable to be worth '
            'keeping. A minimum of 2 factors is enforced to allow comparison '
            'with the human 2-factor solution.</p>\n'
            '<p><strong>Factor scores</strong> are computed via the regression '
            'method and rescaled to 0&ndash;1, giving each entity a position '
            'in the rotated factor space. These coordinates form the '
            '&ldquo;mind perception space&rdquo; scatter plots.</p>\n'
        )

    if include_spearman:
        html += '<h4>Spearman Rank Correlation (&rho;)</h4>\n'
        html += (
            '<p>Spearman&rsquo;s &rho; measures the strength of a monotonic '
            'relationship between two ranked variables. It ranges from &minus;1 '
            '(perfect inverse ranking) to +1 (identical ranking). It is '
            'preferred over Pearson&rsquo;s <em>r</em> for this data because '
            '(a)&nbsp;mind perception scores may not be normally distributed, '
            '(b)&nbsp;the relationship between model and human scores need '
            'not be linear&mdash;only consistently ordered, and '
            '(c)&nbsp;rank correlation is robust to outliers.</p>\n'
            '<p><strong>Significance stars</strong>: * <em>p</em>&nbsp;&lt;&nbsp;0.05, '
            '** <em>p</em>&nbsp;&lt;&nbsp;0.01, '
            '*** <em>p</em>&nbsp;&lt;&nbsp;0.001.</p>\n'
        )

    if include_fdr:
        html += '<h4>FDR Correction (Benjamini&ndash;Hochberg)</h4>\n'
        html += (
            '<p>When testing many hypotheses simultaneously (e.g., one RSA '
            'test per transformer layer), some will appear significant by '
            'chance. <strong>Benjamini&ndash;Hochberg FDR correction</strong> '
            'controls the expected proportion of false positives among '
            'significant results. A corrected <em>q</em>&nbsp;&lt;&nbsp;0.05 '
            'means at most 5% of the discoveries at that threshold are '
            'expected to be false. In the layerwise RSA plots, only layers '
            'passing FDR correction are shown in color; non-significant '
            'layers are grayed out.</p>\n'
        )

    if include_prompting:
        html += '<h4>Chat vs. Base Model Prompting</h4>\n'
        html += (
            '<p><strong>Chat/instruct models</strong> receive prompts wrapped '
            'in their model-specific chat template (e.g., LLaMA-2 uses '
            '<code>[INST]</code> tags; LLaMA-3 uses '
            '<code>&lt;|start_header_id|&gt;</code> headers; Gemma-2 uses '
            '<code>&lt;start_of_turn&gt;</code>). The model generates a text '
            'response, and the numeric rating is extracted via regex.</p>\n'
            '<p><strong>Base models</strong> receive raw text prompts without '
            'chat formatting. Instead of generating a response, ratings are '
            'extracted from the <strong>next-token logit distribution</strong> '
            'over digit tokens. The expected rating is the probability-'
            'weighted sum: &sum;<sub>d=1..5</sub>&nbsp;d&nbsp;&times;&nbsp;'
            'P(d). This avoids requiring the base model to produce a '
            'formatted reply and provides a continuous rating value.</p>\n'
            '<p><strong>Why two methods?</strong> Chat models are trained to '
            'generate structured responses and reliably output numeric '
            'ratings. Base models lack instruction-following behavior and '
            'often fail to produce parseable responses, so logit-based '
            'extraction is more reliable. Both methods yield ratings on the '
            'same 1&ndash;5 scale.</p>\n'
        )

    html += '</div>\n'
    return html


def neural_methods_primer_html(include_layers=True, include_rdm=True,
                               include_rsa=True, include_procrustes=True):
    """Return an HTML block explaining neural/activation analysis methods.

    Covers transformer layers, last-token extraction, RDMs, RSA, and
    Procrustes alignment.
    """
    html = '<div class="method">\n'
    html += '<h3>Neural Analysis Methods</h3>\n'

    if include_layers:
        html += '<h4>Transformer Layers</h4>\n'
        html += (
            '<p>A transformer model processes input through a stack of '
            'layers (e.g., 40 for LLaMA-2-13B, 32 for LLaMA-3-8B, 26 for '
            'Gemma-2-2B). Each layer transforms the representation. Research '
            'consistently shows that early layers capture surface-level '
            'features (syntax, token identity) while deeper layers encode '
            'more abstract, semantic properties. Layerwise analysis reveals '
            '<em>where</em> in the processing hierarchy a particular '
            'representational structure emerges.</p>\n'
        )
        html += '<h4>Last-Token Activations</h4>\n'
        html += (
            '<p>In autoregressive (causal) transformers, each token can only '
            'attend to preceding tokens. This means the <strong>last token '
            'position</strong> is the only position that has attended to the '
            'entire input sequence. Its hidden-state vector therefore serves '
            'as the model&rsquo;s summary representation of the full prompt. '
            'We extract this vector at every layer, yielding one '
            'high-dimensional vector per entity per layer.</p>\n'
        )

    if include_rdm:
        html += '<h4>Cosine Distance</h4>\n'
        html += (
            '<p>Cosine distance&nbsp;=&nbsp;1&nbsp;&minus;&nbsp;cosine '
            'similarity. It measures the angle between two high-dimensional '
            'vectors, ignoring their magnitudes. Two vectors pointing in the '
            'same direction have distance&nbsp;0; orthogonal vectors have '
            'distance&nbsp;1. Cosine distance is preferred over Euclidean '
            'distance for activation vectors because it is invariant to the '
            'overall scale of activations, which can vary across layers.</p>\n'
        )
        html += '<h4>Representational Dissimilarity Matrix (RDM)</h4>\n'
        html += (
            '<p>An RDM is an <em>N</em>&nbsp;&times;&nbsp;<em>N</em> '
            'symmetric matrix where entry (<em>i</em>,&nbsp;<em>j</em>) is '
            'the cosine distance between entity <em>i</em> and entity '
            '<em>j</em>&rsquo;s activation vectors. The diagonal is always 0 '
            '(each entity is identical to itself). The upper triangle '
            'contains all unique pairwise distances. An RDM captures the '
            '<em>geometry</em> of a set of representations&mdash;which '
            'entities the model treats as similar or different&mdash;without '
            'depending on the absolute position or orientation of the '
            'activation vectors.</p>\n'
        )

    if include_rsa:
        html += '<h4>Representational Similarity Analysis (RSA)</h4>\n'
        html += (
            '<p>RSA compares two RDMs by computing the Spearman rank '
            'correlation between their upper triangles (the unique pairwise '
            'distances). If a model&rsquo;s neural RDM has the same pattern '
            'of relative distances as the human behavioral RDM, it means the '
            'model&rsquo;s internal geometry mirrors human mind perception '
            'judgments.</p>\n'
            '<p><strong>Three human reference RDMs</strong> are tested:</p>\n'
            '<ul>\n'
            '<li><strong>Combined</strong>: Euclidean distance in the 2D '
            'Experience&ndash;Agency space. Tests whether the model encodes '
            'the full mind perception geometry.</li>\n'
            '<li><strong>Experience-only</strong>: 1D distances on the '
            'Experience dimension. Tests whether the model specifically '
            'encodes the capacity for sensations and feelings.</li>\n'
            '<li><strong>Agency-only</strong>: 1D distances on the Agency '
            'dimension. Tests whether the model specifically encodes the '
            'capacity for planning, self-control, and communication.</li>\n'
            '</ul>\n'
            '<p>By comparing all three, we can determine whether the model '
            'encodes both dimensions, or is biased toward one (typically '
            'Agency, since agentic capacities are more prominent in '
            'training data).</p>\n'
            '<p><strong>Peak RSA</strong> refers to the layer with the '
            'highest &rho; value, indicating where the model&rsquo;s '
            'representational geometry best matches the human reference.</p>\n'
        )

    if include_procrustes:
        html += '<h4>Procrustes Alignment</h4>\n'
        html += (
            '<p>Procrustes analysis finds the optimal rotation, reflection, '
            'and uniform scaling to align one set of points onto another. '
            'Applied to neural PCA: we reduce each layer&rsquo;s entity '
            'activations to 2D via PCA, then Procrustes-align them to the '
            'human Experience&ndash;Agency space.</p>\n'
            '<p>The residual <strong>disparity</strong> (sum of squared '
            'distances after optimal alignment, normalized to 0&ndash;1) '
            'quantifies geometric mismatch. A disparity of 0 means the '
            'model&rsquo;s entity layout is a perfect rotated/scaled copy of '
            'the human space; a disparity of 1 means no alignment above '
            'chance. Lower disparity at a given layer indicates that the '
            'model&rsquo;s learned representations spontaneously recapitulate '
            'the human mind perception space.</p>\n'
        )

    html += '</div>\n'
    return html


def expanded_concepts_primer_html():
    """Return an HTML block explaining expanded concepts methodology.

    Covers concept dimensions vs capacities, concept vectors, per-concept
    RSA, and standalone vs contrast alignment.
    """
    html = '<div class="method">\n'
    html += '<h3>Expanded Concepts: Key Terminology</h3>\n'

    html += '<h4>Concept Dimensions vs. Capacities</h4>\n'
    html += (
        '<p><strong>Capacities</strong> are the 18 mental ability items '
        'from Gray et al. (e.g., &ldquo;hunger,&rdquo; '
        '&ldquo;self-control,&rdquo; &ldquo;memory&rdquo;). They measure '
        'behavioral judgments about what an entity <em>can do</em>.</p>\n'
        '<p><strong>Concept dimensions</strong> are 22 broader '
        'psychological constructs from Experiment&nbsp;3 (e.g., '
        '&ldquo;phenomenology,&rdquo; &ldquo;agency,&rdquo; '
        '&ldquo;social cognition,&rdquo; &ldquo;embodiment&rdquo;). '
        'Each is defined by a set of reflective prompts and captures how '
        'the model <em>internally represents</em> an abstract '
        'psychological category. Concept dimensions are more abstract and '
        'theory-driven than Gray&rsquo;s empirical capacity items.</p>\n'
    )

    html += '<h4>Concept Vectors and Contrast Alignment</h4>\n'
    html += (
        '<p>In Experiment&nbsp;3, <strong>concept vectors</strong> were '
        'computed as the mean activation direction induced by prompts '
        'about each psychological concept (e.g., prompts about '
        '&ldquo;phenomenal experience&rdquo;). A <strong>contrast '
        'vector</strong> additionally subtracts AI-context activations '
        'from human-context activations, isolating the direction in '
        'activation space that differentiates how the model represents '
        'that concept for humans versus AI systems.</p>\n'
        '<p><strong>Alignment</strong> measures how well a concept '
        'vector predicts the arrangement of characters in activation '
        'space. High alignment for a mental concept (e.g., '
        '&ldquo;emotions&rdquo;) means that concept&rsquo;s direction '
        'in the model&rsquo;s representation space tracks the '
        'human&ndash;AI distinction among the 28 characters.</p>\n'
    )

    html += '<h4>Per-Concept RSA</h4>\n'
    html += (
        '<p>Instead of using raw high-dimensional activation vectors, '
        'each character&rsquo;s activation is projected onto a single '
        'concept vector, yielding a 1D &ldquo;concept score.&rdquo; An '
        'RDM built from these 1D scores tests whether <em>that specific '
        'concept dimension</em> carries categorical (AI vs. human) '
        'structure. This reveals which psychological constructs the '
        'model uses to differentiate entity types, rather than relying '
        'on the full activation geometry.</p>\n'
    )

    html += '<h4>Standalone vs. Contrast Alignment</h4>\n'
    html += (
        '<p><strong>Standalone alignment</strong> uses the raw concept '
        'vector (mean activation across concept prompts, without '
        'subtracting AI context) to project character representations. '
        'It measures whether the concept direction is informative about '
        'characters in general.</p>\n'
        '<p><strong>Contrast alignment</strong> uses the human-minus-AI '
        'difference vector. It measures whether the <em>human&ndash;AI '
        'differential</em> within each concept tracks the character '
        'distinction. Contrast alignment is more specific: it tests '
        'not just whether the concept direction is active, but whether '
        'it specifically encodes the human&ndash;AI boundary.</p>\n'
    )

    html += '</div>\n'
    return html


# ============================================================================
# DATASET ARGUMENT
# ============================================================================

# ============================================================================
# TABLE FORMATTING HELPERS
# ============================================================================

def model_row_td(model_key):
    """Return a <td> for the first column of a table row identifying a model.

    Includes a colored left border and bold label for visual consistency.
    """
    color = MODEL_COLORS.get(model_key, "#333333")
    label = MODEL_LABELS.get(model_key, model_key)
    return (
        f'<td style="border-left: 4px solid {color}; '
        f'font-weight: 600;">{label}</td>'
    )


def format_p_cell(rho, p, show_rho=True):
    """Return a <td> with consistent significance formatting.

    If show_rho is True, displays rho value with stars.
    If show_rho is False, displays p-value only.
    Significant cells (p < 0.05) get class="sig" (bold red via CSS).
    """
    if rho is None or p is None:
        return "<td>--</td>"
    stars = _sig_stars(p)
    sig_cls = ' class="sig"' if p < 0.05 else ""
    if show_rho:
        return f'<td{sig_cls}>{rho:.3f}{stars}</td>'
    else:
        return f'<td{sig_cls}>{p:.4f}{stars}</td>'


def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ============================================================================
# FDR CORRECTION
# ============================================================================

def apply_fdr(rsa_layer_list):
    """Apply Benjamini-Hochberg FDR correction to a list of RSA layer dicts.

    Adds 'q_fdr' key to each dict. Returns the list (modified in-place).
    """
    from statsmodels.stats.multitest import multipletests
    pvals, valid_idx = [], []
    for i, r in enumerate(rsa_layer_list):
        p = r.get("p_value")
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            pvals.append(p)
            valid_idx.append(i)
    for r in rsa_layer_list:
        r["q_fdr"] = 1.0
    if pvals:
        _, q_corr, _, _ = multipletests(pvals, method="fdr_bh")
        for idx, q in zip(valid_idx, q_corr):
            rsa_layer_list[idx]["q_fdr"] = float(q)
    return rsa_layer_list


# ============================================================================
# INSTRUCTION TUNING PAIRS
# ============================================================================

INSTRUCTION_TUNING_PAIRS = [
    ("LLaMA-2-13B", "llama2_13b_base", "llama2_13b_chat"),
    ("LLaMA-3-8B",  "llama3_8b_base",  "llama3_8b_instruct"),
    ("Gemma-2-2B",  "gemma2_2b",       "gemma2_2b_it"),
    ("Gemma-2-9B",  "gemma2_9b",       "gemma2_9b_it"),
    ("Qwen-2.5-7B", "qwen25_7b",      "qwen25_7b_instruct"),
]


# ============================================================================
# DATASET ARGUMENT
# ============================================================================

def add_dataset_argument(parser, default="full_dataset"):
    """Deprecated: previously controlled full_dataset/reduced_dataset subfolders.

    Kept for CLI compatibility but the argument is ignored. Reports now
    write directly to the results directory without a dataset subfolder.
    """
    parser.add_argument(
        "--dataset", type=str, default=default,
        help="(deprecated, ignored)"
    )
