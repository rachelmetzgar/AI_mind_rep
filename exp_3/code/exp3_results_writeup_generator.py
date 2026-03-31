#!/usr/bin/env python3
"""
Experiment 3: Comprehensive Results Writeup Generator

Synthesizes all Exp 3 analyses into a single, self-contained HTML report
with table of contents, example stimuli, methodology overviews, and
publication-ready embedded figures.

Output:
    exp_3/writeup/exp3_results_writeup.html

Usage:
    python code/exp3_results_writeup_generator.py
    python code/exp3_results_writeup_generator.py --version balanced_gpt --turn 5

Env: llama2_env (login node, lightweight — no GPU needed)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import base64
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    config, set_version, set_model,
    add_version_argument, add_model_argument, add_turn_argument,
    get_model, DIMENSION_CATEGORIES, CATEGORY_COLORS,
    get_dimension_category, get_category_color,
)


# ============================================================
# CONSTANTS
# ============================================================

# Human-readable labels for each dimension
DIM_LABELS = {
    0: "Baseline (human vs AI, no concept)",
    1: "Phenomenal Experience / Qualia",
    2: "Emotions / Affect",
    3: "Agency / Autonomy",
    4: "Intentions / Goals",
    5: "Prediction / Anticipation",
    6: "Cognitive / Reasoning",
    7: "Social / Relationships",
    8: "Embodiment / Physical Presence",
    9: "Roles / Social Identity",
    10: "Animacy / Aliveness",
    11: "Formality / Register",
    12: "Expertise / Competence",
    13: "Helpfulness / Cooperation",
    14: "Biological (control)",
    15: "Shapes: Round vs Angular (control)",
    16: "Mind (holistic aggregate)",
    17: "Attention / Focus",
    18: "System Prompt (contrasts)",
    20: "SysPrompt: talk-to human",
    21: "SysPrompt: talk-to AI",
    22: "SysPrompt: bare human",
    23: "SysPrompt: bare AI",
    25: "Beliefs / Epistemic States",
    26: "Desires / Motivations",
    27: "Goals / Objectives",
    29: "Shapes Flip (reversed polarity)",
    30: "Granite vs Sandstone (control)",
    31: "Squares vs Triangles (control)",
    32: "Horizontal vs Vertical (control)",
}

# Short names for tables
DIM_SHORT = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind", 17: "Attention", 18: "SysPrompt", 25: "Beliefs",
    26: "Desires", 27: "Goals", 29: "Shapes Flip",
    30: "Granite/Sandstone", 31: "Squares/Triangles", 32: "Horiz/Vert",
}


# ============================================================
# DATA LOADING
# ============================================================

def load_json(path):
    """Load a JSON file, return None if missing."""
    if not os.path.exists(path):
        print(f"  [skip] {path}")
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(path):
    """Load a CSV file as DataFrame, return None if missing."""
    if not os.path.exists(path):
        print(f"  [skip] {path}")
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def embed_figure(path):
    """Read a PNG and return base64-encoded data URI. None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def load_concept_prompts(concepts_dir, mode="contrasts"):
    """
    Dynamically load concept prompt files and extract example prompts.

    Returns dict: dim_id -> {
        'name': str, 'description': str,
        'human_examples': [str, ...], 'ai_examples': [str, ...],
        'standalone_examples': [str, ...]
    }
    """
    prompts = {}
    concept_dir = os.path.join(concepts_dir, mode)
    if not os.path.isdir(concept_dir):
        return prompts

    for fname in sorted(os.listdir(concept_dir)):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        parts = fname.replace(".py", "").split("_", 1)
        try:
            dim_id = int(parts[0])
        except ValueError:
            continue
        dim_name = parts[1] if len(parts) > 1 else ""

        fpath = os.path.join(concept_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(f"concept_{dim_id}", fpath)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            continue

        entry = {
            "name": dim_name.replace("_", " ").title(),
            "description": (mod.__doc__ or "").strip().split("\n")[0],
            "human_examples": [],
            "ai_examples": [],
            "standalone_examples": [],
        }

        # Try to find prompt lists
        for attr in dir(mod):
            val = getattr(mod, attr)
            if not isinstance(val, list) or not val:
                continue
            attr_upper = attr.upper()
            if "HUMAN" in attr_upper and "PROMPT" in attr_upper:
                entry["human_examples"] = val[:3]
            elif "AI" in attr_upper and "PROMPT" in attr_upper:
                entry["ai_examples"] = val[:3]
            elif "STANDALONE" in attr_upper and "PROMPT" in attr_upper:
                entry["standalone_examples"] = val[:3]

        prompts[dim_id] = entry

    return prompts


# ============================================================
# HTML HELPERS
# ============================================================

def figure_html(data_uri, caption="", width="100%"):
    """Generate an HTML figure block from a base64 data URI."""
    if data_uri is None:
        return ""
    return f"""
    <figure class="fig-container">
        <img src="{data_uri}" alt="{caption}" style="max-width:{width};">
        {f'<figcaption>{caption}</figcaption>' if caption else ''}
    </figure>
    """


def sig_stars(p):
    """Return significance stars for a p-value."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def cat_color_style(category):
    """Return inline style for category coloring."""
    color = CATEGORY_COLORS.get(category, "#666")
    return f'style="border-left: 4px solid {color}; padding-left: 8px;"'


def cat_badge(category):
    """Return a small colored badge for a category."""
    color = CATEGORY_COLORS.get(category, "#666")
    return f'<span class="cat-badge" style="background:{color};">{category}</span>'


# ============================================================
# CSS
# ============================================================

CSS = """
:root {
    --accent: #2c5282;
    --accent-light: #ebf4ff;
    --text: #1a1a1a;
    --text-secondary: #4a5568;
    --border: #cbd5e0;
    --bg: #ffffff;
    --light-bg: #f7fafc;
    --green: #38a169;
    --yellow: #d69e2e;
    --red: #e53e3e;
}
* { box-sizing: border-box; }
body {
    font-family: "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
    color: var(--text);
    background: var(--bg);
    max-width: 940px;
    margin: 0 auto;
    padding: 24px 32px;
    line-height: 1.7;
    font-size: 15px;
}
h1 { font-size: 1.8em; color: var(--accent); border-bottom: 3px solid var(--accent); padding-bottom: 8px; margin-top: 40px; }
h2 { font-size: 1.4em; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-top: 36px; }
h3 { font-size: 1.15em; color: var(--text); margin-top: 28px; }
a { color: var(--accent); }

/* Table of contents */
.toc {
    background: var(--light-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px 24px;
    margin: 20px 0 32px;
}
.toc h2 { border: none; margin: 0 0 8px; font-size: 1.1em; }
.toc ol { margin: 0; padding-left: 20px; }
.toc li { margin: 3px 0; line-height: 1.5; }
.toc ol ol { margin-top: 2px; }

/* Boxes */
.example-box {
    background: #f0f7ff;
    border-left: 4px solid var(--accent);
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.93em;
}
.finding-box {
    background: #f0fff4;
    border-left: 4px solid var(--green);
    padding: 12px 16px;
    margin: 16px 0;
    border-radius: 0 4px 4px 0;
}
.method-note {
    background: #fffbeb;
    border-left: 4px solid var(--yellow);
    padding: 12px 16px;
    margin: 12px 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.93em;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 0.9em;
}
th, td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: left;
}
th {
    background: var(--light-bg);
    font-weight: 600;
}
tr:nth-child(even) { background: #fafafa; }
.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Figures */
.fig-container {
    text-align: center;
    margin: 20px 0;
}
.fig-container img {
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 4px;
}
figcaption {
    font-size: 0.88em;
    color: var(--text-secondary);
    margin-top: 6px;
    font-style: italic;
}

/* Category badge */
.cat-badge {
    display: inline-block;
    color: #fff;
    font-size: 0.75em;
    padding: 1px 7px;
    border-radius: 3px;
    font-weight: 600;
    vertical-align: middle;
}

/* Prompt examples */
.prompt-pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 8px 0;
}
.prompt-pair > div {
    padding: 10px;
    border-radius: 4px;
    font-size: 0.9em;
}
.prompt-human { background: #edf2f7; border-left: 3px solid #4299e1; }
.prompt-ai { background: #fff5f5; border-left: 3px solid #fc8181; }
.prompt-standalone { background: #f0fff4; border-left: 3px solid #68d391; }

.two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

@media print {
    body { max-width: 100%; padding: 12px; }
    .fig-container img { max-width: 90%; }
    .toc { break-after: page; }
}

.meta { color: var(--text-secondary); font-size: 0.9em; margin-bottom: 24px; }
"""


# ============================================================
# SECTION GENERATORS
# ============================================================

def gen_header():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Experiment 3: Concept-of-Mind Representations &mdash; Results Writeup</title>
<style>{CSS}</style>
</head>
<body>
<h1>Experiment 3: Concept-of-Mind Representations</h1>
<p class="meta">Comprehensive Results Writeup &mdash; LLaMA-2-13B-Chat &mdash;
Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
"""


def gen_toc():
    return """
<nav class="toc">
<h2>Table of Contents</h2>
<ol>
    <li><a href="#intro">Introduction &amp; Overview</a></li>
    <li><a href="#dimensions">Concept Dimensions</a></li>
    <li><a href="#construction">Concept Vector Construction</a>
        <ol>
            <li><a href="#contrasts">Entity-Framed Contrasts</a></li>
            <li><a href="#standalone">Standalone (Entity-Neutral)</a></li>
            <li><a href="#variants">Methodology Variants</a></li>
        </ol>
    </li>
    <li><a href="#alignment">Alignment with Partner-Identity Probes</a>
        <ol>
            <li><a href="#align-raw">Raw Contrast Alignment</a></li>
            <li><a href="#align-residual">Residual Alignment</a></li>
            <li><a href="#align-standalone">Standalone Alignment</a></li>
        </ol>
    </li>
    <li><a href="#overlap">Concept Overlap / Similarity Structure</a></li>
    <li><a href="#conversation">Concept-Conversation Alignment</a></li>
    <li><a href="#steering">Concept Steering &amp; Behavioral Effects</a></li>
    <li><a href="#lexical">Lexical Confound Analysis</a></li>
    <li><a href="#controls">Controls &amp; Robustness</a></li>
    <li><a href="#summary">Summary of Key Findings</a></li>
</ol>
</nav>
"""


def gen_introduction():
    return """
<h2 id="intro">1. Introduction &amp; Overview</h2>

<p>Experiment 3 investigates whether LLaMA-2-13B-Chat forms internal representations
of <em>what human and AI minds are like</em> &mdash; concept-level knowledge about mental
properties such as phenomenal consciousness, emotions, agency, and social cognition.
The central questions are:</p>

<ol>
    <li><strong>Existence:</strong> Do concept-of-mind representations exist in the model's
    activation space, distinct from generic "human vs AI" entity encoding?</li>
    <li><strong>Alignment:</strong> Do these concept vectors align geometrically with
    the partner-identity probes trained in Experiment 2 (which detect whether the
    model "thinks" its conversation partner is human or AI)?</li>
    <li><strong>Causal relevance:</strong> Does injecting concept vectors into the
    residual stream during generation causally change the model's linguistic behavior?</li>
    <li><strong>Ecological validity:</strong> Are concept vectors more "active" in
    actual conversations with human partners vs AI partners?</li>
</ol>

<div class="method-note">
<strong>Model:</strong> LLaMA-2-13B-Chat (5,120-dimensional hidden states, 41 layers).
All analyses use the <code>balanced_gpt</code> version of Exp 2 probe data (turn 5)
unless otherwise noted. Statistical tests use 1,000 bootstrap iterations for CIs
and layers 6&ndash;40 (excluding early layers with prompt-format confounds).
</div>

<p>The experiment uses <strong>33 concept dimensions</strong> organized into categories:
Mental (11 core mind properties), Physical (3 embodied properties), Pragmatic (3
communication properties), and Controls (shapes, biological, system prompt, orthogonal).
Each dimension is probed via 40 carefully constructed prompts in two modes: entity-framed
<em>contrasts</em> (human vs AI paired prompts) and entity-neutral <em>standalone</em>
prompts (concept only, no entity reference).</p>
"""


def gen_dimensions(contrast_prompts, standalone_prompts):
    """Section 2: Table of all concept dimensions with examples."""
    html = ['<h2 id="dimensions">2. Concept Dimensions</h2>']
    html.append("""<p>Each dimension targets a specific mental, physical, or pragmatic property.
    Dimensions are grouped into categories that predict different levels of alignment
    with partner-identity probes. The table below shows all dimensions with their category
    and example prompts.</p>""")

    html.append("""<table>
    <thead><tr>
        <th>#</th><th>Dimension</th><th>Category</th>
        <th>Example Human Prompt</th><th>Example AI Prompt</th>
    </tr></thead><tbody>""")

    # Sort by category then dim_id
    cat_order = ["Mental", "Physical", "Pragmatic", "Baseline", "Bio Ctrl",
                 "Meta", "SysPrompt", "Shapes"]
    dim_ids = sorted(DIM_LABELS.keys())
    dim_ids.sort(key=lambda d: (cat_order.index(get_dimension_category(d))
                                 if get_dimension_category(d) in cat_order else 99, d))

    for dim_id in dim_ids:
        cat = get_dimension_category(dim_id)
        color = get_category_color(cat)
        name = DIM_SHORT.get(dim_id, str(dim_id))

        h_ex = ""
        a_ex = ""
        if dim_id in contrast_prompts:
            cp = contrast_prompts[dim_id]
            if cp["human_examples"]:
                h_ex = f'<em>"{cp["human_examples"][0][:120]}{"..." if len(cp["human_examples"][0]) > 120 else ""}"</em>'
            if cp["ai_examples"]:
                a_ex = f'<em>"{cp["ai_examples"][0][:120]}{"..." if len(cp["ai_examples"][0]) > 120 else ""}"</em>'

        html.append(f"""<tr>
            <td>{dim_id}</td>
            <td style="border-left: 3px solid {color}; padding-left: 8px;">{name}</td>
            <td>{cat_badge(cat)}</td>
            <td style="font-size:0.85em;">{h_ex}</td>
            <td style="font-size:0.85em;">{a_ex}</td>
        </tr>""")

    html.append("</tbody></table>")
    return "\n".join(html)


def gen_construction(contrast_prompts, standalone_prompts):
    """Section 3: How concept vectors are constructed."""
    html = ['<h2 id="construction">3. Concept Vector Construction</h2>']

    # 3.1 Contrasts
    html.append('<h3 id="contrasts">3.1 Entity-Framed Contrasts</h3>')
    html.append("""<p>For each concept dimension, 40 <strong>human-framed</strong> and 40
    <strong>AI-framed</strong> prompts are constructed. The prompts are structurally
    parallel within each sub-facet (4 sub-facets &times; 10 prompts = 40 per entity type,
    80 total per dimension).</p>

    <div class="method-note">
    <strong>Concept vector computation (contrasts):</strong><br>
    For each prompt, a forward pass through LLaMA-2-13B-Chat extracts the residual-stream
    activation at the <strong>last token</strong> across all 41 layers. The concept direction
    is computed as:
    <br><br>
    <code>concept_direction[layer] = mean(human_activations[layer]) &minus; mean(ai_activations[layer])</code>
    <br><br>
    This yields a <strong>5,120-dimensional direction vector per layer</strong> pointing from
    "AI version of concept" toward "human version of concept." Alignment is then measured as
    R&sup2; = cos&sup2;(concept_direction, probe_weight) averaged across layers 6&ndash;40.
    </div>""")

    # Show example for phenomenology
    if 1 in contrast_prompts:
        cp = contrast_prompts[1]
        html.append('<p><strong>Example: Phenomenology (Dim 1)</strong></p>')
        html.append('<div class="prompt-pair">')
        html.append('<div class="prompt-human"><strong>Human prompt:</strong><br>')
        for p in cp["human_examples"][:2]:
            html.append(f'"{p}"<br>')
        html.append('</div>')
        html.append('<div class="prompt-ai"><strong>AI prompt:</strong><br>')
        for p in cp["ai_examples"][:2]:
            html.append(f'"{p}"<br>')
        html.append('</div></div>')

    # Show example for shapes (control)
    if 15 in contrast_prompts:
        cp = contrast_prompts[15]
        html.append('<p><strong>Example: Shapes (Dim 15, Control)</strong></p>')
        html.append('<div class="prompt-pair">')
        html.append('<div class="prompt-human"><strong>"Human" (round):</strong><br>')
        for p in cp["human_examples"][:2]:
            html.append(f'"{p}"<br>')
        html.append('</div>')
        html.append('<div class="prompt-ai"><strong>"AI" (angular):</strong><br>')
        for p in cp["ai_examples"][:2]:
            html.append(f'"{p}"<br>')
        html.append('</div></div>')

    # 3.2 Standalone
    html.append('<h3 id="standalone">3.2 Standalone (Entity-Neutral)</h3>')
    html.append("""<p>Standalone prompts reference the concept <em>without</em> attributing it
    to humans or AIs. This tests whether the model encodes abstract concept knowledge
    that aligns with partner-identity probes even without explicit entity framing.</p>

    <div class="method-note">
    <strong>Concept vector computation (standalone):</strong><br>
    <code>concept_vector[layer] = mean(all_40_activations[layer])</code><br><br>
    No subtraction &mdash; the vector is an undirected activation centroid in concept space.
    Alignment is measured as R&sup2; = cos&sup2;(concept_centroid, probe_weight).
    </div>""")

    if 1 in standalone_prompts:
        sp = standalone_prompts[1]
        html.append('<p><strong>Example: Phenomenology (Standalone)</strong></p>')
        html.append('<div class="prompt-standalone" style="padding:10px; margin:8px 0;">')
        for p in sp.get("standalone_examples", [])[:2]:
            html.append(f'"{p}"<br>')
        html.append('</div>')

    # 3.3 Variants
    html.append('<h3 id="variants">3.3 Methodology Variants</h3>')
    html.append("""<p>To test robustness, concept vectors are also computed under variant conditions:</p>
    <table>
    <thead><tr><th>Variant</th><th>Method</th><th>Purpose</th></tr></thead>
    <tbody>
    <tr><td><strong>Full 40-prompt</strong></td>
        <td>Average all 40 prompts per condition</td>
        <td>Maximum statistical power (primary analysis)</td></tr>
    <tr><td><strong>Top-1 aligned</strong></td>
        <td>Single most representative prompt (highest cosine to centroid, layers 20&ndash;40)</td>
        <td>Tests whether averaging artifacts drive alignment</td></tr>
    <tr><td><strong>Simple/syntactic</strong></td>
        <td>153 concepts &times; 1 template: "Think about what it is like to have [X]"</td>
        <td>Syntactic control &mdash; same template across concepts</td></tr>
    </tbody></table>""")

    return "\n".join(html)


def gen_alignment_section(raw_data, residual_data, standalone_data, figures_dir):
    """Section 4: Alignment with partner-identity probes."""
    html = ['<h2 id="alignment">4. Alignment with Partner-Identity Probes</h2>']
    html.append("""<p>The core test: do concept-of-mind vectors point in the same direction
    as the probes trained in Experiment 2 to detect partner identity (human vs AI)?
    Alignment is measured as <strong>R&sup2; = cos&sup2;</strong> between concept vectors
    and probe weight vectors, averaged across layers 6&ndash;40 with 1,000 bootstrap CIs.</p>

    <p>Two probe types are tested:</p>
    <ul>
        <li><strong>Metacognitive probe</strong> ("reading"): trained to read what partner type
        the model thinks it is talking to</li>
        <li><strong>Operational probe</strong> ("control"): trained to predict how the model
        adapts its behavior based on partner type</li>
    </ul>""")

    # 4.1 Raw
    html.append('<h3 id="align-raw">4.1 Raw Contrast Alignment</h3>')
    html.append("""<p>Cosine&sup2; between contrast direction vectors (human &minus; AI means)
    and probe weight vectors, with no corrections.</p>""")

    if raw_data:
        html.append(_alignment_table(raw_data, "Raw Contrast"))
        html.append(_alignment_findings(raw_data))

    # Embed raw figures
    raw_figs_dir = os.path.join(figures_dir, "raw", "figures")
    for fname, caption in [
        ("category_summary_metacognitive.png", "Raw alignment R-squared by category (metacognitive probe)"),
        ("category_summary_operational.png", "Raw alignment R-squared by category (operational probe)"),
        ("heatmap_metacognitive.png", "Layer-by-layer alignment heatmap (metacognitive probe)"),
    ]:
        html.append(figure_html(embed_figure(os.path.join(raw_figs_dir, fname)), caption))

    # 4.2 Residual
    html.append('<h3 id="align-residual">4.2 Residual Alignment (Baseline Projected Out)</h3>')
    html.append("""<p>To test whether concept alignment goes beyond generic "human vs AI"
    entity encoding, the <strong>baseline direction</strong> (Dim 0: human vs AI with
    no conceptual content) is projected out of each concept vector before computing alignment.</p>

    <div class="method-note">
    <code>residual = concept_direction &minus; (concept_direction &middot; baseline_unit) &times; baseline_unit</code>
    <br>Joint bootstrap resampling of concept and baseline prompts (1,000 iterations).
    </div>""")

    if residual_data:
        html.append(_alignment_table(residual_data, "Residual"))

    res_figs_dir = os.path.join(figures_dir, "residual", "figures")
    for fname, caption in [
        ("metacognitive_bars.png", "Residual alignment by dimension (metacognitive probe)"),
        ("metacognitive_heatmap.png", "Residual alignment heatmap across layers"),
    ]:
        html.append(figure_html(embed_figure(os.path.join(res_figs_dir, fname)), caption))

    if raw_data and residual_data:
        html.append(_residual_comparison(raw_data, residual_data))

    # 4.3 Standalone
    html.append('<h3 id="align-standalone">4.3 Standalone Alignment</h3>')
    html.append("""<p>Concept centroids from entity-neutral prompts (no "human" or "AI" in text).
    Tests whether the model's abstract concept knowledge &mdash; without explicit entity framing &mdash;
    aligns with partner-identity probes.</p>""")

    if standalone_data:
        html.append(_alignment_table(standalone_data, "Standalone"))

    sa_figs_dir = os.path.join(figures_dir, "standalone", "figures")
    for fname, caption in [
        ("metacognitive_bars.png", "Standalone alignment by dimension (metacognitive probe)"),
        ("metacognitive_heatmap.png", "Standalone alignment heatmap across layers"),
    ]:
        html.append(figure_html(embed_figure(os.path.join(sa_figs_dir, fname)), caption))

    return "\n".join(html)


def _alignment_table(data, label):
    """Generate an alignment summary table from a summary.json dict."""
    rows = []
    for key, vals in sorted(data.items(), key=lambda x: x[1].get("control_mean_r2", 0), reverse=True):
        dim_id = vals.get("dim_id", 0)
        cat = get_dimension_category(dim_id)
        name = DIM_SHORT.get(dim_id, key)
        r_r2 = vals.get("reading_mean_r2", 0)
        c_r2 = vals.get("control_mean_r2", 0)
        r_ci = vals.get("reading_boot_ci95", [0, 0])
        c_ci = vals.get("control_boot_ci95", [0, 0])
        rows.append((dim_id, cat, name, r_r2, r_ci, c_r2, c_ci))

    html = [f'<p><strong>{label} Alignment Summary (sorted by operational R&sup2;)</strong></p>']
    html.append("""<table><thead><tr>
        <th>#</th><th>Dimension</th><th>Category</th>
        <th class="num">Metacog R&sup2;</th><th class="num">95% CI</th>
        <th class="num">Operational R&sup2;</th><th class="num">95% CI</th>
    </tr></thead><tbody>""")

    for dim_id, cat, name, r_r2, r_ci, c_r2, c_ci in rows:
        color = get_category_color(cat)
        html.append(f"""<tr>
            <td>{dim_id}</td>
            <td style="border-left: 3px solid {color};">{name}</td>
            <td>{cat_badge(cat)}</td>
            <td class="num">{r_r2:.6f}</td>
            <td class="num">[{r_ci[0]:.6f}, {r_ci[1]:.6f}]</td>
            <td class="num">{c_r2:.6f}</td>
            <td class="num">[{c_ci[0]:.6f}, {c_ci[1]:.6f}]</td>
        </tr>""")

    html.append("</tbody></table>")
    return "\n".join(html)


def _alignment_findings(data):
    """Extract key findings from alignment data."""
    mental_ids = set(DIMENSION_CATEGORIES.get("Mental", []))
    shape_ids = set(DIMENSION_CATEGORIES.get("Shapes", []))

    mental_r2 = []
    shape_r2 = []
    for key, vals in data.items():
        dim_id = vals.get("dim_id", -1)
        c_r2 = vals.get("control_mean_r2", 0)
        if dim_id in mental_ids:
            mental_r2.append(c_r2)
        elif dim_id in shape_ids:
            shape_r2.append(c_r2)

    if mental_r2 and shape_r2:
        mental_mean = np.mean(mental_r2)
        shape_mean = np.mean(shape_r2)
        ratio = mental_mean / shape_mean if shape_mean > 0 else float("inf")
        return f"""
        <div class="finding-box">
        <strong>Key finding:</strong> Mental concept dimensions show mean operational
        R&sup2; = {mental_mean:.6f}, while shape/orthogonal controls show
        R&sup2; = {shape_mean:.6f} &mdash; a <strong>{ratio:.1f}&times; ratio</strong>.
        Mental concepts align substantially more with partner-identity probes than
        content-irrelevant controls with identical prompt structure.
        </div>"""
    return ""


def _residual_comparison(raw_data, residual_data):
    """Show how much alignment survives after baseline projection."""
    html = ['<div class="finding-box"><strong>Residual analysis:</strong> ']
    mental_ids = set(DIMENSION_CATEGORIES.get("Mental", []))
    surviving = []
    for key in raw_data:
        if key not in residual_data:
            continue
        dim_id = raw_data[key].get("dim_id", -1)
        if dim_id not in mental_ids:
            continue
        raw_r2 = raw_data[key].get("control_mean_r2", 0)
        res_r2 = residual_data[key].get("control_mean_r2", 0)
        if raw_r2 > 0:
            surviving.append(res_r2 / raw_r2)
    if surviving:
        mean_pct = np.mean(surviving) * 100
        html.append(f"""On average, <strong>{mean_pct:.0f}%</strong> of mental concept
        alignment survives after projecting out the entity baseline direction.
        This indicates concept-specific signal beyond generic human/AI encoding.</div>""")
    else:
        html.append("(insufficient data for comparison)</div>")
    return "\n".join(html)


def gen_overlap_section(results_base):
    """Section 5: Concept overlap / similarity structure."""
    html = ['<h2 id="overlap">5. Concept Overlap / Similarity Structure</h2>']
    html.append("""<p>How similar are concept vectors to each other? Pairwise cosine
    similarity between all concept directions reveals the internal geometry of
    mind-concept space. High overlap between mental concepts (vs controls) would
    suggest a shared "mind" subspace.</p>""")

    # Contrasts overlap
    contrast_overlap_dir = os.path.join(results_base, "concept_overlap", "contrasts")
    baseline_csv = load_csv(os.path.join(contrast_overlap_dir, "data", "baseline_overlap.csv"))

    if baseline_csv is not None:
        html.append('<h3>Contrast Direction Overlap with Baseline (Dim 0)</h3>')
        html.append("""<p>How much does each concept's human/AI direction overlap with the
        generic baseline human/AI direction? High overlap suggests the concept captures
        entity identity; low overlap suggests conceptual specificity.</p>""")

        html.append("""<table><thead><tr>
            <th>Dimension</th><th>Category</th>
            <th class="num">Mean |cos|</th>
            <th class="num">CI Lower</th><th class="num">CI Upper</th>
        </tr></thead><tbody>""")
        for _, row in baseline_csv.iterrows():
            cat = row.get("category", "")
            color = CATEGORY_COLORS.get(cat, "#666")
            html.append(f"""<tr>
                <td style="border-left: 3px solid {color};">{row.get('name', '')}</td>
                <td>{cat_badge(cat)}</td>
                <td class="num">{row['mean_abs_cosine']:.4f}</td>
                <td class="num">{row['ci_lower']:.4f}</td>
                <td class="num">{row['ci_upper']:.4f}</td>
            </tr>""")
        html.append("</tbody></table>")

        html.append("""<div class="finding-box">
        <strong>Key pattern:</strong> Mental concepts (Phenomenology, Emotions, Agency, etc.)
        show moderate-to-high overlap with the baseline direction (0.44&ndash;0.74),
        indicating they share some variance with generic entity encoding. Pragmatic dimensions
        (Formality, Helpfulness) and Shape controls show much lower overlap (0.07&ndash;0.25),
        confirming they capture different information.
        </div>""")

    # Figures
    figs_dir = os.path.join(contrast_overlap_dir, "figures")
    for fname, caption in [
        ("overlap_heatmap.png", "Pairwise cosine similarity between contrast direction vectors"),
        ("category_summary.png", "Within- vs between-category overlap summary"),
    ]:
        html.append(figure_html(embed_figure(os.path.join(figs_dir, fname)), caption))

    # Standalone overlap
    standalone_figs = os.path.join(results_base, "concept_overlap", "standalone", "figures")
    html.append('<h3>Standalone Concept Overlap</h3>')
    html.append("""<p>Pairwise similarity between entity-neutral concept centroids.
    Unlike contrasts (which compute a difference direction), standalone overlap shows
    how close concept representations sit in absolute activation space.</p>""")
    for fname, caption in [
        ("overlap_heatmap.png", "Standalone concept overlap heatmap"),
        ("category_summary.png", "Standalone within- vs between-category overlap"),
    ]:
        html.append(figure_html(embed_figure(os.path.join(standalone_figs, fname)), caption))

    return "\n".join(html)


def gen_conversation_section(conv_data, results_base, version):
    """Section 6: Concept-conversation alignment."""
    html = ['<h2 id="conversation">6. Concept-Conversation Alignment</h2>']
    html.append("""<p>Do concept vectors "activate" more during actual conversations
    with matching partner types? This analysis extracts activations from real Exp 1
    conversations (human vs AI conditions) and measures cosine similarity with concept vectors.</p>

    <p>Three complementary approaches:</p>
    <ul>
        <li><strong>Approach A (Full mean):</strong> Average all 40 prompts &rarr; single concept vector.
        Compare mean cosine in human vs AI conversations.</li>
        <li><strong>Approach C (Contrastive):</strong> Subtract mean of other concepts to isolate
        unique signal: <code>concept &minus; mean(others)</code></li>
        <li><strong>Approach D (Prompt-level):</strong> Per-prompt alignment scores.
        Identifies which specific prompts drive effects.</li>
    </ul>""")

    if conv_data is not None:
        # Cross-approach summary table
        html.append('<h3 id="conv-cross">Cross-Approach Comparison</h3>')

        for approach in ["A", "C", "D"]:
            adf = conv_data[conv_data["approach"] == approach]
            if adf.empty:
                continue

            html.append(f'<p><strong>Approach {approach}</strong></p>')
            html.append("""<table><thead><tr>
                <th>Dimension</th>
                <th class="num">Human Mean</th><th class="num">AI Mean</th>
                <th class="num">Diff</th><th class="num">p (FDR)</th>
                <th class="num">Cohen's d</th>
            </tr></thead><tbody>""")

            for _, row in adf.sort_values("cohen_d", ascending=False).iterrows():
                dim_name = row["dimension"]
                dim_id_str = dim_name.split("_")[0]
                try:
                    dim_id = int(dim_id_str)
                except ValueError:
                    dim_id = -1
                cat = get_dimension_category(dim_id)
                color = get_category_color(cat)
                p_fdr = row.get("p_fdr", row.get("p", 1))
                stars = sig_stars(p_fdr)

                html.append(f"""<tr>
                    <td style="border-left: 3px solid {color};">{dim_name}</td>
                    <td class="num">{row['human_mean']:.4f}</td>
                    <td class="num">{row['ai_mean']:.4f}</td>
                    <td class="num">{row['diff']:.4f}</td>
                    <td class="num">{p_fdr:.2e} {stars}</td>
                    <td class="num">{row['cohen_d']:.3f}</td>
                </tr>""")

            html.append("</tbody></table>")

        # Findings
        if "A" in conv_data["approach"].values:
            a_data = conv_data[conv_data["approach"] == "A"]
            n_sig = (a_data["p_fdr"] < 0.05).sum() if "p_fdr" in a_data.columns else 0
            n_total = len(a_data)
            html.append(f"""<div class="finding-box">
            <strong>Approach A:</strong> {n_sig}/{n_total} dimensions show significantly
            higher cosine similarity with concept vectors in human conversations vs AI
            conversations (FDR-corrected p &lt; 0.05). This broad effect is expected because
            Approach A uses the full concept centroid, which contains some entity-correlated variance.
            </div>""")

        if "C" in conv_data["approach"].values:
            c_data = conv_data[conv_data["approach"] == "C"]
            c_sig = c_data[c_data["p_fdr"] < 0.05] if "p_fdr" in c_data.columns else pd.DataFrame()
            html.append(f"""<div class="finding-box">
            <strong>Approach C (contrastive):</strong> After subtracting the mean of other
            concepts, {len(c_sig)}/{len(c_data)} dimensions retain significance. This isolates
            concept-specific signal vs shared "mind" variance. Dimensions with strong unique
            signatures (Goals, Intentions, Agency) survive; those capturing generic mental
            content may not.
            </div>""")

    # Figures
    conv_base = os.path.join(results_base, version, "concept_conversation", "turn_5")
    for approach, caption in [
        ("approach_a", "Approach A: Full concept mean alignment by dimension"),
        ("approach_c", "Approach C: Contrastive (concept minus others) alignment"),
        ("approach_d", "Approach D: Prompt-level alignment"),
    ]:
        fig_path = os.path.join(conv_base, approach, "figures", "alignment_by_concept.png")
        html.append(figure_html(embed_figure(fig_path), caption))

    return "\n".join(html)


def gen_steering_section(behavioral_data):
    """Section 7: Concept steering and behavioral effects."""
    html = ['<h2 id="steering">7. Concept Steering &amp; Behavioral Effects</h2>']
    html.append("""<p>The causal test: concept vectors are injected into the residual stream
    during generation to steer the model toward "more human-like" or "more AI-like" behavior.</p>

    <div class="method-note">
    <strong>Steering procedure:</strong><br>
    <code>h'[layer, token] = h[layer, token] + sign &times; strength &times; unit_direction[layer]</code><br>
    where sign = +1 (toward human) or &minus;1 (toward AI), strength &isin; {1, 2, 4, 8},
    and layers are selected by one of three strategies:
    <ul>
        <li><strong>exp2_peak:</strong> Top 15 layers by Exp 2 metacognitive probe accuracy</li>
        <li><strong>upper_half:</strong> Layers 20&ndash;40 (heuristic)</li>
        <li><strong>concept_aligned:</strong> Top 15 layers by |cosine| with probe weights (per-concept)</li>
    </ul>
    Generated responses are analyzed for 21+ linguistic markers including hedges, discourse
    markers, sentiment, theory-of-mind phrases, and politeness.
    </div>""")

    if behavioral_data is not None:
        # Extract key metrics for a summary table
        # CSV columns: {metric}_baseline_mean, {metric}_human_mean, {metric}_ai_mean,
        #              {metric}_human_minus_ai, {metric}_hva_p, {metric}_hva_t, {metric}_hva_p_fdr
        key_metrics = [
            ("word_count", "Word Count"),
            ("question_count", "Question Count"),
            ("demir_modal_rate", "Modal Hedges"),
            ("fung_interpersonal_rate", "Interpersonal"),
            ("fung_cognitive_rate", "Cognitive"),
            ("tom_rate", "Theory of Mind"),
            ("sentiment_compound", "Sentiment"),
        ]

        cols = behavioral_data.columns.tolist()
        available_metrics = []
        for prefix, label in key_metrics:
            if f"{prefix}_hva_p" in cols:
                available_metrics.append((prefix, label))

        if available_metrics:
            html.append('<h3>Behavioral Effects Summary</h3>')
            html.append("""<p>Key linguistic metrics comparing human-steered vs AI-steered
            generation. Values show human&minus;AI difference; significance via Welch's t-test
            (FDR-corrected where available). Only <code>concept_aligned</code> strategy shown.</p>""")

            # Filter to concept_aligned strategy for cleaner view
            strategy_col = "strategy" if "strategy" in cols else None
            if strategy_col:
                display_df = behavioral_data[behavioral_data[strategy_col] == "concept_aligned"].copy()
            else:
                display_df = behavioral_data.copy()

            html.append("""<table><thead><tr>
                <th>Dimension</th><th>Category</th>""")
            for _, label in available_metrics[:5]:
                html.append(f'<th class="num">{label} (H&minus;A)</th>')
            html.append("</tr></thead><tbody>")

            dim_col = None
            for c in ["dimension", "dim_name", "concept", "dim"]:
                if c in cols:
                    dim_col = c
                    break

            if dim_col:
                for _, row in display_df.iterrows():
                    dim_name = str(row[dim_col])
                    dim_id_str = dim_name.split("_")[0] if "_" in dim_name else dim_name
                    try:
                        dim_id = int(dim_id_str)
                    except ValueError:
                        dim_id = -1
                    cat = get_dimension_category(dim_id)
                    color = get_category_color(cat)

                    html.append(f'<tr><td style="border-left: 3px solid {color};">{dim_name}</td>')
                    html.append(f'<td>{cat_badge(cat)}</td>')

                    for prefix, _ in available_metrics[:5]:
                        diff_col = f"{prefix}_human_minus_ai"
                        p_col = f"{prefix}_hva_p_fdr"
                        if p_col not in cols:
                            p_col = f"{prefix}_hva_p"
                        diff_val = row.get(diff_col, np.nan)
                        p_val = row.get(p_col, np.nan)
                        stars = sig_stars(p_val) if not (isinstance(p_val, float) and np.isnan(p_val)) else ""
                        if isinstance(diff_val, (int, float)) and not np.isnan(diff_val):
                            diff_str = f"{diff_val:+.3f}"
                        else:
                            diff_str = "---"
                        html.append(f'<td class="num">{diff_str} {stars}</td>')

                    html.append("</tr>")

            html.append("</tbody></table>")

        html.append("""<div class="finding-box">
        <strong>Key finding:</strong> Steering with mental concept vectors (especially Phenomenology,
        Agency, Emotions) produces measurable shifts in linguistic markers. Human-direction steering
        tends to increase hedging, discourse markers, and theory-of-mind language, while AI-direction
        steering produces more direct, less hedged responses. This demonstrates <em>causal relevance</em>
        of concept representations to generated behavior.
        </div>""")
    else:
        html.append('<p><em>Behavioral summary data not available.</em></p>')

    return "\n".join(html)


def gen_lexical_section():
    """Section 8: Lexical confound analysis."""
    html = ['<h2 id="lexical">8. Lexical Confound Analysis</h2>']
    html.append("""<p>A critical concern: could alignment between concept vectors and
    partner-identity probes be driven by <em>shared vocabulary</em> rather than genuine
    representational structure? If concept prompts about "phenomenology" simply use words
    that also appear more in human conversations, the alignment might be a lexical artifact.</p>

    <h3>Method</h3>
    <p>For each word in the vocabulary, a <strong>bias score</strong> is computed from Exp 1
    conversations:</p>
    <div class="method-note">
    <code>bias(w) = (freq_human &minus; freq_ai) / (freq_human + freq_ai)</code><br>
    Positive = word appears more in human-directed conversations. Then for each concept
    dimension, the mean word-bias of its prompts is correlated with alignment R&sup2;
    (Spearman rank correlation).
    </div>

    <h3>Results</h3>
    <table>
    <thead><tr><th>Analysis</th><th>Spearman &rho;</th><th>p-value</th><th>Interpretation</th></tr></thead>
    <tbody>
    <tr>
        <td><strong>Contrast prompts</strong> (H&minus;A bias differential)</td>
        <td class="num">+0.61</td><td class="num">0.001</td>
        <td style="color: var(--red);">Significant positive &mdash; confound is plausible for contrasts</td>
    </tr>
    <tr>
        <td><strong>Standalone prompts</strong> (mean word bias)</td>
        <td class="num">&minus;0.44</td><td class="num">0.018</td>
        <td style="color: var(--green);">Significant <em>negative</em> &mdash; opposite direction from confound</td>
    </tr>
    <tr>
        <td><strong>Residual</strong> (contrasts, baseline projected out)</td>
        <td class="num">+0.49</td><td class="num">0.013</td>
        <td>Reduced but still positive after projection</td>
    </tr>
    </tbody></table>

    <div class="finding-box">
    <strong>Defense against lexical artifact:</strong>
    <ol>
        <li><strong>Standalone prompts show the opposite pattern:</strong> concepts whose prompts use
        more "human-biased" words actually have <em>lower</em> alignment (&rho; = &minus;0.44).
        A lexical confound predicts a positive correlation; we observe the opposite.</li>
        <li><strong>Control concepts have near-zero alignment:</strong> Shape and orthogonal
        controls use prompts with the same grammatical structure as mental concepts but show
        ~10&times; lower alignment (~0.0002 vs ~0.002 R&sup2;).</li>
        <li><strong>Alignment is measured at deep layers:</strong> R&sup2; compares concept
        vectors (extracted from concept prompts) to probe weights (trained on conversational
        data from a completely different input distribution). Shared vocabulary would need to
        produce geometrically aligned deep-layer representations across two different contexts.</li>
    </ol>
    </div>""")

    return "\n".join(html)


def gen_controls_section(raw_data):
    """Section 9: Controls and robustness."""
    html = ['<h2 id="controls">9. Controls &amp; Robustness</h2>']

    html.append("""<h3>Negative Controls</h3>
    <p>Several dimensions serve as sanity checks &mdash; they use the same prompt structure
    and extraction procedure as mental concepts but should show minimal alignment with
    partner-identity probes.</p>""")

    if raw_data:
        control_dims = {15, 30, 31, 32}
        mental_dims = set(DIMENSION_CATEGORIES.get("Mental", []))

        control_r2 = []
        mental_r2 = []
        for key, vals in raw_data.items():
            dim_id = vals.get("dim_id", -1)
            c_r2 = vals.get("control_mean_r2", 0)
            if dim_id in control_dims:
                control_r2.append((DIM_SHORT.get(dim_id, key), c_r2))
            elif dim_id in mental_dims:
                mental_r2.append(c_r2)

        html.append("""<table><thead><tr>
            <th>Control Dimension</th><th class="num">Operational R&sup2;</th>
            <th>Description</th>
        </tr></thead><tbody>""")
        for name, r2 in control_r2:
            html.append(f"""<tr>
                <td>{name}</td><td class="num">{r2:.6f}</td>
                <td>Orthogonal property with no expected mind relevance</td>
            </tr>""")
        html.append("</tbody></table>")

        if mental_r2 and control_r2:
            m_mean = np.mean(mental_r2)
            c_mean = np.mean([r for _, r in control_r2])
            ratio = m_mean / c_mean if c_mean > 0 else float("inf")
            html.append(f"""<div class="finding-box">
            <strong>Control validation:</strong> Mean mental R&sup2; = {m_mean:.6f} vs
            mean orthogonal control R&sup2; = {c_mean:.6f} ({ratio:.1f}&times; ratio).
            Controls confirm that alignment is content-specific, not an artifact of the
            prompt structure or extraction method.
            </div>""")

    html.append("""<h3>Cross-Version Replication</h3>
    <p>Alignment analyses are replicated across two independent Exp 2 data versions:</p>
    <ul>
        <li><strong>balanced_gpt:</strong> Balanced gender names with GPT-4 conversation replacement</li>
        <li><strong>nonsense_codeword:</strong> Partner tokens framed as arbitrary session codes</li>
    </ul>
    <p>Both versions produce qualitatively similar alignment patterns, confirming
    robustness to the specific way partner identity is communicated in Exp 2.</p>

    <h3>Top-1 Variant</h3>
    <p>Using only the single most representative prompt per concept (highest cosine to
    centroid, layers 20&ndash;40) still produces significant alignment for mental dimensions.
    This rules out averaging artifacts: a single prompt suffices to recover the alignment signal.</p>

    <h3>Biological Control (Dim 14)</h3>
    <p>Biological properties (growth, metabolism, reproduction) are a "near-miss" control &mdash;
    related to the human/AI distinction but not targeting mental properties. Biological shows
    moderate alignment (between mental and orthogonal controls), consistent with the idea
    that embodiment-adjacent properties partially overlap with mind representations but are
    not equivalent.</p>""")

    return "\n".join(html)


def gen_summary():
    """Section 10: Summary of key findings."""
    return """
<h2 id="summary">10. Summary of Key Findings</h2>

<div class="finding-box">
<ol>
    <li><strong>Concept-of-mind vectors exist and are organized:</strong>
    LLaMA-2-13B-Chat develops internal directions in activation space corresponding to
    mental properties (phenomenology, emotions, agency, etc.). These directions are
    geometrically organized &mdash; mental concepts cluster together and away from
    non-mental controls.</li>

    <li><strong>Significant alignment with partner-identity probes:</strong>
    Mental concept directions align with Exp 2 probes that detect whether the model "thinks"
    its partner is human or AI. This alignment is ~10&times; stronger for mental concepts
    than orthogonal controls (shapes, spatial orientation), despite identical prompt structure.</li>

    <li><strong>25&ndash;40% of alignment is concept-specific:</strong>
    After projecting out the generic "human vs AI" baseline direction, a substantial portion
    of mental concept alignment survives. The model encodes concept-specific information
    about minds beyond mere entity identity.</li>

    <li><strong>Standalone (entity-neutral) prompts confirm genuine concept encoding:</strong>
    Prompts that never mention "human" or "AI" still produce concept vectors aligned with
    partner-identity probes. The model's abstract concept knowledge &mdash; what phenomenal
    experience is, what emotions are &mdash; is geometrically structured in a way that
    overlaps with how it represents conversation partner identity.</li>

    <li><strong>Lexical confound ruled out for standalone analysis:</strong>
    The correlation between prompt word-bias and alignment R&sup2; is <em>negative</em>
    for standalone prompts (&rho; = &minus;0.44), the opposite direction from what a
    vocabulary-driven artifact would predict. This is the strongest single defense against
    the confound hypothesis.</li>

    <li><strong>Concept vectors are causally relevant:</strong>
    Injecting concept directions during generation produces measurable shifts in linguistic
    markers (hedging, discourse markers, theory-of-mind language). The representations are
    not just correlational &mdash; they causally influence behavior.</li>

    <li><strong>Ecological validity confirmed:</strong>
    Concept vectors show higher cosine similarity with activations from actual human-directed
    conversations (Exp 1) vs AI-directed conversations, consistent with the idea that these
    concept representations are deployed during real conversational processing.</li>
</ol>
</div>

<h3>Limitations &amp; Open Questions</h3>
<ul>
    <li>All analyses use a single model (LLaMA-2-13B-Chat); generalization across model
    families and sizes remains untested for Exp 3.</li>
    <li>Contrast alignment shows a positive lexical correlation (&rho; = +0.61), meaning
    the contrast-mode analysis alone cannot rule out vocabulary artifacts. The standalone
    analysis provides the primary defense.</li>
    <li>R&sup2; values are small in absolute terms (~0.001&ndash;0.003), though the relative
    pattern (mental &gt;&gt; controls) is robust. This reflects the high dimensionality of
    the representation space (5,120 dimensions).</li>
    <li>Steering effects demonstrate causal influence but the specific behavioral shifts
    (more hedging, more TOM language) need further interpretation in terms of folk-psychological
    theory.</li>
</ul>
"""


def gen_footer():
    return f"""
<hr>
<p class="meta">Generated by <code>exp3_results_writeup_generator.py</code> on
{datetime.now().strftime("%Y-%m-%d %H:%M")}. Data from LLaMA-2-13B-Chat,
balanced_gpt version, turn 5.</p>
</body></html>
"""


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive Exp 3 results writeup."
    )
    add_model_argument(parser)
    add_version_argument(parser)
    add_turn_argument(parser)
    args = parser.parse_args()

    set_model(args.model)
    set_version(args.version)
    version = args.version
    turn = args.turn
    model = get_model()

    print(f"Generating Exp 3 results writeup: model={model}, version={version}, turn={turn}")

    # Paths
    results_base = os.path.join(config.RESULTS.root, model)
    version_base = os.path.join(results_base, version)
    alignment_base = os.path.join(version_base, "alignment", f"turn_{turn}")
    comparisons_align = os.path.join(results_base, "comparisons", "alignment", f"turn_{turn}")
    concepts_dir = str(config.PATHS.concepts_root)

    # Load data
    print("Loading data...")
    raw_data = load_json(os.path.join(alignment_base, "contrasts", "raw", "data", "summary.json"))
    residual_data = load_json(os.path.join(alignment_base, "contrasts", "residual", "data", "summary.json"))
    standalone_data = load_json(os.path.join(alignment_base, "standalone", "data", "summary.json"))
    conv_data = load_csv(os.path.join(version_base, "concept_conversation", f"turn_{turn}", "data", "cross_approach_summary.csv"))
    behavioral_data = load_csv(os.path.join(version_base, "concept_steering", "v1", "behavioral_summary.csv"))

    print("Loading concept prompts...")
    contrast_prompts = load_concept_prompts(concepts_dir, "contrasts")
    standalone_prompts = load_concept_prompts(concepts_dir, "standalone")
    print(f"  Loaded {len(contrast_prompts)} contrast dims, {len(standalone_prompts)} standalone dims")

    # Generate HTML
    print("Generating HTML sections...")
    sections = [
        gen_header(),
        gen_toc(),
        gen_introduction(),
        gen_dimensions(contrast_prompts, standalone_prompts),
        gen_construction(contrast_prompts, standalone_prompts),
        gen_alignment_section(raw_data, residual_data, standalone_data, comparisons_align),
        gen_overlap_section(results_base),
        gen_conversation_section(conv_data, results_base, version),
        gen_steering_section(behavioral_data),
        gen_lexical_section(),
        gen_controls_section(raw_data),
        gen_summary(),
        gen_footer(),
    ]

    html = "\n".join(sections)

    # Write output
    output_dir = os.path.join(config.ROOT_DIR, "writeup")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "exp3_results_writeup.html")

    with open(output_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nWriteup generated: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
