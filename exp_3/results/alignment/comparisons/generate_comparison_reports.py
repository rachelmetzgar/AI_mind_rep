#!/usr/bin/env python3
"""
Generate cross-version alignment comparison reports for all analysis types.

Produces HTML + MD reports for:
  - raw:        contrasts/raw/summary.json
  - residual:   contrasts/residual/summary.json
  - standalone: standalone/summary.json

No external dependencies — uses only Python standard library.

Usage:
    python generate_comparison_reports.py              # all three
    python generate_comparison_reports.py --type raw   # just one

Rachel C. Metzgar · Feb 2026
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIG
# ============================================================================

ALIGNMENT_ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ["labels", "balanced_names", "balanced_gpt", "names",
            "nonsense_codeword", "nonsense_ignore"]

VERSION_LABELS = {
    "labels": "Labels",
    "balanced_names": "Balanced Names",
    "balanced_gpt": "Balanced GPT",
    "names": "Names (orig.)",
    "nonsense_codeword": "Nonsense Codeword",
    "nonsense_ignore": "Nonsense Ignore",
}

VERSION_DESCRIPTIONS = {
    "labels": "Partner identified as 'human' or 'AI' (no names)",
    "balanced_names": "Gender-balanced names (e.g., Alex/Jordan)",
    "balanced_gpt": "Balanced names with GPT-4 replacing 'AI' partner",
    "names": "Original Sam/Casey names (deprecated due to name confound)",
    "nonsense_codeword": "Nonsense codewords replacing identity labels",
    "nonsense_ignore": "Nonsense labels with instruction to ignore them",
}

VERSION_COLORS = {
    "labels": "#E53935",
    "balanced_names": "#1E88E5",
    "balanced_gpt": "#43A047",
    "names": "#FB8C00",
    "nonsense_codeword": "#8E24AA",
    "nonsense_ignore": "#546E7A",
}

# Analysis-specific config
ANALYSIS_TYPES = {
    "raw": {
        "path": "contrasts/raw",
        "title": "Raw Contrast Alignment",
        "description": (
            "Raw cosine alignment between human-AI contrast vectors and probe weight vectors. "
            "No entity baseline subtraction. Concept direction = mean(human) - mean(AI) per layer."
        ),
        "extra_field": None,
    },
    "residual": {
        "path": "contrasts/residual",
        "title": "Residual Alignment (Entity Baseline Projected Out)",
        "description": (
            "Same as raw alignment, but with the entity baseline direction (dim 0: 'this is a human/AI') "
            "projected out of each concept vector before computing alignment. This removes shared "
            "entity-level variance, isolating concept-specific alignment."
        ),
        "extra_field": "entity_overlap",
    },
    "standalone": {
        "path": "standalone",
        "title": "Standalone Concept Alignment",
        "description": (
            "Alignment between standalone mean activation vectors and probe weights. "
            "Standalone prompts describe concepts without entity framing (no 'human' or 'AI' words). "
            "Concept vector = mean activation across all prompts for that concept. "
            "Tests whether alignment is driven by concept content rather than entity labels."
        ),
        "extra_field": None,
    },
}

# Dimension display info — covers both contrasts and standalone numbering
DIM_NAMES = {
    0: "Baseline", 1: "Phenomenology", 2: "Emotions", 3: "Agency",
    4: "Intentions", 5: "Prediction", 6: "Cognitive", 7: "Social",
    8: "Embodiment", 9: "Roles", 10: "Animacy", 11: "Formality",
    12: "Expertise", 13: "Helpfulness", 14: "Biological", 15: "Shapes",
    16: "Mind (holistic)", 17: "Attention", 18: "SysPrompt (labeled)",
    # Standalone-specific
    20: "SysPrompt (talkto human)", 21: "SysPrompt (talkto AI)",
    22: "SysPrompt (bare human)", 23: "SysPrompt (bare AI)",
}

# Standalone has 16_human and 17_ai instead of 16_mind and 17_attention
STANDALONE_DIM_NAMES = dict(DIM_NAMES)
STANDALONE_DIM_NAMES[16] = "Human (concept)"
STANDALONE_DIM_NAMES[17] = "AI (concept)"
STANDALONE_DIM_NAMES[18] = "Attention"

def get_dim_category(dim_id, analysis_type):
    """Get category for a dimension, handling standalone numbering."""
    if analysis_type == "standalone":
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 18],  # 18=attention in standalone
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Control":   [14, 15],
            "Entity":    [16, 17],  # human/ai concept vectors
            "SysPrompt": [20, 21, 22, 23],
        }
    else:
        categories = {
            "Mental":    [1, 2, 3, 4, 5, 6, 7, 17],
            "Physical":  [8, 9, 10],
            "Pragmatic": [11, 12, 13],
            "Control":   [0, 14, 15],
            "SysPrompt": [18],
        }
    for cat, ids in categories.items():
        if dim_id in ids:
            return cat
    return "Other"


CATEGORY_COLORS = {
    "Mental": "#2196F3",
    "Physical": "#4CAF50",
    "Pragmatic": "#FF9800",
    "Control": "#9E9E9E",
    "Entity": "#7B1FA2",
    "SysPrompt": "#00BCD4",
    "Other": "#999",
}

# Exclude dim 16 (mind holistic) only from contrasts analyses
EXCLUDE_DIMS = {
    "raw": {16},
    "residual": {16},
    "standalone": set(),  # standalone 16 = "Human concept", keep it
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_summaries(analysis_type):
    """Load summary.json for each version. Returns dict[version] = summary."""
    subpath = ANALYSIS_TYPES[analysis_type]["path"]
    data = {}
    for v in VERSIONS:
        path = ALIGNMENT_ROOT / v / subpath / "summary.json"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {v}")
            continue
        with open(path) as f:
            data[v] = json.load(f)
    return data


def organize_data(all_data, analysis_type):
    """Organize into structured dict keyed by dim_id."""
    exclude = EXCLUDE_DIMS.get(analysis_type, set())
    dim_names = STANDALONE_DIM_NAMES if analysis_type == "standalone" else DIM_NAMES

    result = {}
    for version, summary in all_data.items():
        for dim_folder, stats in summary.items():
            dim_id = stats["dim_id"]
            if dim_id in exclude:
                continue
            if dim_id not in result:
                cat = get_dim_category(dim_id, analysis_type)
                result[dim_id] = {
                    "name": dim_names.get(dim_id, dim_folder),
                    "folder": dim_folder,
                    "category": cat,
                    "versions": {},
                }
            entry = {
                "reading_r2": stats["reading_mean_r2"],
                "control_r2": stats["control_mean_r2"],
                "reading_ci": stats.get("reading_boot_ci95", [0, 0]),
                "control_ci": stats.get("control_boot_ci95", [0, 0]),
            }
            if "entity_overlap" in stats:
                entry["entity_overlap"] = stats["entity_overlap"]
            result[dim_id]["versions"][version] = entry
    return result


# ============================================================================
# SVG GENERATION
# ============================================================================

def get_cat_order(analysis_type):
    if analysis_type == "standalone":
        return ["Mental", "Physical", "Pragmatic", "Entity", "Control", "SysPrompt", "Other"]
    return ["Mental", "Physical", "Pragmatic", "Control", "SysPrompt", "Other"]


def sort_dims(dim_data, analysis_type):
    cat_order = get_cat_order(analysis_type)
    return sorted(dim_data.keys(),
                  key=lambda d: (cat_order.index(dim_data[d]["category"])
                                 if dim_data[d]["category"] in cat_order else 99, d))


def get_available_versions(dim_data):
    for dd in dim_data.values():
        return [v for v in VERSIONS if v in dd["versions"]]
    return []


def svg_grouped_bar(dim_data, probe_type, global_max, analysis_type,
                    chart_width=750, chart_height=300):
    key = probe_type + "_r2"
    ci_key = probe_type + "_ci"
    sorted_dims = sort_dims(dim_data, analysis_type)
    available_versions = get_available_versions(dim_data)
    n_dims = len(sorted_dims)
    n_av = len(available_versions)

    margin = {"top": 30, "right": 20, "bottom": 110, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]
    group_width = plot_w / max(n_dims, 1)
    bar_width = group_width * 0.8 / max(n_av, 1)
    gap = group_width * 0.1
    y_max = global_max * 1.1

    lines = []
    lines.append(f'<svg viewBox="0 0 {chart_width} {chart_height}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="width:100%;max-width:{chart_width}px;height:auto;font-family:Arial,sans-serif;">')
    lines.append(f'<rect width="{chart_width}" height="{chart_height}" fill="white"/>')
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')

    # Y gridlines
    for i in range(6):
        y_val = y_max * i / 5
        y_pos = plot_h - (y_val / y_max * plot_h)
        lines.append(f'<line x1="0" y1="{y_pos:.1f}" x2="{plot_w}" y2="{y_pos:.1f}" '
                     f'stroke="#eee" stroke-width="1"/>')
        lines.append(f'<text x="-8" y="{y_pos+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#666">{y_val*1000:.2f}</text>')

    lines.append(f'<text x="-50" y="{plot_h/2}" text-anchor="middle" '
                 f'font-size="11" fill="#333" '
                 f'transform="rotate(-90,-50,{plot_h/2})">Mean R\u00b2 (\u00d710\u207b\u00b3)</text>')

    for gi, dim_id in enumerate(sorted_dims):
        dd = dim_data[dim_id]
        x_group = gi * group_width + gap
        for vi, version in enumerate(available_versions):
            if version not in dd["versions"]:
                continue
            vd = dd["versions"][version]
            val = vd[key]
            ci = vd[ci_key]
            x = x_group + vi * bar_width
            bar_h = max(val / y_max * plot_h, 0)
            y = plot_h - bar_h
            color = VERSION_COLORS[version]
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                         f'height="{bar_h:.2f}" fill="{color}" opacity="0.85">'
                         f'<title>{VERSION_LABELS[version]}: {val:.6f}</title></rect>')
            # CI whisker
            cx = x + bar_width / 2
            y_lo = plot_h - max(ci[0] / y_max * plot_h, 0)
            y_hi = plot_h - max(ci[1] / y_max * plot_h, 0)
            lines.append(f'<line x1="{cx:.2f}" y1="{y_lo:.2f}" x2="{cx:.2f}" y2="{y_hi:.2f}" '
                         f'stroke="#333" stroke-width="1"/>')
            lines.append(f'<line x1="{cx-2:.2f}" y1="{y_lo:.2f}" x2="{cx+2:.2f}" y2="{y_lo:.2f}" stroke="#333" stroke-width="1"/>')
            lines.append(f'<line x1="{cx-2:.2f}" y1="{y_hi:.2f}" x2="{cx+2:.2f}" y2="{y_hi:.2f}" stroke="#333" stroke-width="1"/>')

        # X label
        x_center = x_group + (n_av * bar_width) / 2
        cat_color = CATEGORY_COLORS.get(dd["category"], "#333")
        lines.append(f'<text x="{x_center:.2f}" y="{plot_h + 14}" text-anchor="end" '
                     f'font-size="9" fill="{cat_color}" '
                     f'transform="rotate(-45,{x_center:.2f},{plot_h + 14})">{dd["name"]}</text>')

    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333"/>')
    lines.append('</g>')

    # Legend
    leg_y = chart_height - 15
    for i, v in enumerate(available_versions):
        x = margin["left"] + i * 115
        lines.append(f'<rect x="{x}" y="{leg_y-8}" width="10" height="10" fill="{VERSION_COLORS[v]}"/>')
        lines.append(f'<text x="{x+14}" y="{leg_y}" font-size="9" fill="#333">{VERSION_LABELS[v]}</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def svg_heatmap(dim_data, probe_type, global_max, analysis_type,
                chart_width=750, row_height=26):
    key = probe_type + "_r2"
    sorted_dims = sort_dims(dim_data, analysis_type)
    available_versions = get_available_versions(dim_data)
    n_dims = len(sorted_dims)
    n_vers = len(available_versions)

    margin = {"top": 65, "right": 80, "bottom": 20, "left": 175}
    cell_w = (chart_width - margin["left"] - margin["right"]) / max(n_vers, 1)
    chart_height = margin["top"] + n_dims * row_height + margin["bottom"]

    lines = []
    lines.append(f'<svg viewBox="0 0 {chart_width} {chart_height}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="width:100%;max-width:{chart_width}px;height:auto;font-family:Arial,sans-serif;">')
    lines.append(f'<rect width="{chart_width}" height="{chart_height}" fill="white"/>')

    for vi, v in enumerate(available_versions):
        x = margin["left"] + vi * cell_w + cell_w / 2
        lines.append(f'<text x="{x:.1f}" y="{margin["top"]-8}" text-anchor="end" '
                     f'font-size="10" fill="#333" '
                     f'transform="rotate(-40,{x:.1f},{margin["top"]-8})">{VERSION_LABELS[v]}</text>')

    prev_cat = None
    for ri, dim_id in enumerate(sorted_dims):
        dd = dim_data[dim_id]
        y = margin["top"] + ri * row_height
        cat = dd["category"]
        if cat != prev_cat and prev_cat is not None:
            lines.append(f'<line x1="{margin["left"]}" y1="{y}" '
                         f'x2="{margin["left"] + n_vers * cell_w}" y2="{y}" '
                         f'stroke="#999" stroke-width="1.5"/>')
        prev_cat = cat
        cat_color = CATEGORY_COLORS.get(cat, "#333")
        lines.append(f'<text x="{margin["left"]-8}" y="{y + row_height/2 + 4}" '
                     f'text-anchor="end" font-size="10" fill="{cat_color}">{dd["name"]}</text>')

        for vi, v in enumerate(available_versions):
            if v not in dd["versions"]:
                continue
            val = dd["versions"][v][key]
            intensity = min(val / global_max, 1.0) if global_max > 0 else 0
            r = int(255 - intensity * 200)
            g = int(255 - intensity * 150)
            x = margin["left"] + vi * cell_w
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" '
                         f'height="{row_height}" fill="rgb({r},{g},255)" '
                         f'stroke="white" stroke-width="1">'
                         f'<title>{VERSION_LABELS[v]} \u2014 {dd["name"]}: R\u00b2={val:.6f}</title></rect>')
            text_color = "#fff" if intensity > 0.5 else "#333"
            lines.append(f'<text x="{x + cell_w/2:.1f}" y="{y + row_height/2 + 4:.1f}" '
                         f'text-anchor="middle" font-size="8" fill="{text_color}">'
                         f'{val*1000:.2f}</text>')

    # Colorbar
    cb_x = margin["left"] + n_vers * cell_w + 15
    cb_h = n_dims * row_height
    cb_w = 15
    for i in range(50):
        frac = i / 49
        r = int(255 - frac * 200)
        g = int(255 - frac * 150)
        y = margin["top"] + cb_h - (frac * cb_h)
        lines.append(f'<rect x="{cb_x}" y="{y:.1f}" width="{cb_w}" height="{cb_h/49+1:.1f}" '
                     f'fill="rgb({r},{g},255)"/>')
    lines.append(f'<text x="{cb_x + cb_w + 4}" y="{margin["top"] + 4}" '
                 f'font-size="8" fill="#333">{global_max*1000:.2f}</text>')
    lines.append(f'<text x="{cb_x + cb_w + 4}" y="{margin["top"] + cb_h + 4}" '
                 f'font-size="8" fill="#333">0.00</text>')
    lines.append(f'<text x="{cb_x + cb_w + 4}" y="{margin["top"] + cb_h/2 + 4}" '
                 f'font-size="8" fill="#666">\u00d710\u207b\u00b3</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


def svg_version_summary(dim_data, analysis_type, chart_width=700, chart_height=300):
    """Bar chart: mean R² across mental dims, per version, reading vs control."""
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    available_versions = get_available_versions(dim_data)

    version_means = {}
    for v in available_versions:
        r_vals = [dim_data[d]["versions"][v]["reading_r2"]
                  for d in mental_dims if v in dim_data[d]["versions"]]
        c_vals = [dim_data[d]["versions"][v]["control_r2"]
                  for d in mental_dims if v in dim_data[d]["versions"]]
        version_means[v] = {
            "reading": sum(r_vals) / len(r_vals) if r_vals else 0,
            "control": sum(c_vals) / len(c_vals) if c_vals else 0,
        }

    gmax = max(max(vm["reading"], vm["control"]) for vm in version_means.values()) * 1.15

    margin = {"top": 30, "right": 20, "bottom": 80, "left": 70}
    plot_w = chart_width - margin["left"] - margin["right"]
    plot_h = chart_height - margin["top"] - margin["bottom"]
    group_w = plot_w / max(len(available_versions), 1)
    bar_w = group_w * 0.35

    lines = []
    lines.append(f'<svg viewBox="0 0 {chart_width} {chart_height}" '
                 f'xmlns="http://www.w3.org/2000/svg" '
                 f'style="width:100%;max-width:{chart_width}px;height:auto;font-family:Arial,sans-serif;">')
    lines.append(f'<rect width="{chart_width}" height="{chart_height}" fill="white"/>')
    lines.append(f'<g transform="translate({margin["left"]},{margin["top"]})">')

    for i in range(6):
        y_val = gmax * i / 5
        y_pos = plot_h - (y_val / gmax * plot_h) if gmax > 0 else plot_h
        lines.append(f'<line x1="0" y1="{y_pos:.1f}" x2="{plot_w}" y2="{y_pos:.1f}" stroke="#eee"/>')
        lines.append(f'<text x="-8" y="{y_pos+4:.1f}" text-anchor="end" font-size="10" fill="#666">'
                     f'{y_val*1000:.2f}</text>')

    lines.append(f'<text x="-50" y="{plot_h/2}" text-anchor="middle" font-size="11" fill="#333" '
                 f'transform="rotate(-90,-50,{plot_h/2})">Mean R\u00b2 (\u00d710\u207b\u00b3)</text>')

    for gi, v in enumerate(available_versions):
        vm = version_means[v]
        x_group = gi * group_w + group_w * 0.15
        h_r = (vm["reading"] / gmax * plot_h) if gmax > 0 else 0
        lines.append(f'<rect x="{x_group:.1f}" y="{plot_h - h_r:.1f}" '
                     f'width="{bar_w:.1f}" height="{h_r:.1f}" fill="#2196F3" opacity="0.8">'
                     f'<title>Reading: {vm["reading"]*1000:.3f} \u00d710\u207b\u00b3</title></rect>')
        x_c = x_group + bar_w + 2
        h_c = (vm["control"] / gmax * plot_h) if gmax > 0 else 0
        lines.append(f'<rect x="{x_c:.1f}" y="{plot_h - h_c:.1f}" '
                     f'width="{bar_w:.1f}" height="{h_c:.1f}" fill="#FF5722" opacity="0.8">'
                     f'<title>Control: {vm["control"]*1000:.3f} \u00d710\u207b\u00b3</title></rect>')
        x_center = x_group + bar_w
        lines.append(f'<text x="{x_center:.1f}" y="{plot_h+14}" text-anchor="end" font-size="10" fill="#333" '
                     f'transform="rotate(-35,{x_center:.1f},{plot_h+14})">{VERSION_LABELS[v]}</text>')

    lines.append(f'<line x1="0" y1="{plot_h}" x2="{plot_w}" y2="{plot_h}" stroke="#333"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_h}" stroke="#333"/>')
    lines.append('</g>')

    ly = chart_height - 12
    lines.append(f'<rect x="{margin["left"]}" y="{ly-8}" width="10" height="10" fill="#2196F3"/>')
    lines.append(f'<text x="{margin["left"]+14}" y="{ly}" font-size="10">Reading Probe</text>')
    lines.append(f'<rect x="{margin["left"]+120}" y="{ly-8}" width="10" height="10" fill="#FF5722"/>')
    lines.append(f'<text x="{margin["left"]+134}" y="{ly}" font-size="10">Control Probe</text>')
    lines.append('</svg>')
    return '\n'.join(lines)


# ============================================================================
# ANALYSIS-SPECIFIC CONTENT
# ============================================================================

ANALYSIS_METHODS = {
    "raw": """
<p><strong>Concept vectors</strong> (same across all versions): For each conceptual dimension, we extracted
the model's internal direction distinguishing human from AI concepts.
Computed as <code>mean(human prompts) - mean(AI prompts)</code> across layers 6&ndash;40 of LLaMA-2-13B-Chat.</p>
<p><strong>Alignment metric</strong>: Cosine similarity (squared = R&sup2;) between each concept direction and
each probe's weight vector, computed per layer then averaged across layers 6&ndash;40 (35 layers).
Layers 0&ndash;5 are excluded: layer 0 (token embeddings) produces content-independent mean activations
that create spurious alignment, and layers 1&ndash;5 have prompt-format confounds with near-zero-norm
contrast vectors that make cosine similarity unstable.
Bootstrap confidence intervals (1,000 iterations) computed by resampling concept prompts.</p>
""",

    "residual": """
<p><strong>Concept vectors</strong>: Same as raw analysis, but with the <strong>entity baseline direction</strong>
(dim 0: "this is a human/AI") <strong>projected out</strong> of each concept vector before computing alignment.
This removes shared entity-level variance, isolating concept-specific content.</p>
<p><strong>Projection</strong>: For each layer, compute <code>v_residual = v - (v &middot; d&#770;) &middot; d&#770;</code>
where <code>d&#770;</code> is the unit entity baseline direction at that layer.</p>
<p><strong>Layer range</strong>: Layers 6&ndash;40 only (35 layers). Layers 0&ndash;5 excluded due to
content-independent embeddings (layer 0) and prompt-format confounds (layers 1&ndash;5).</p>
<p><strong>Bootstrap</strong>: Both the concept vector and the entity baseline are independently resampled
(prompt-level) at each of 1,000 iterations, so CIs reflect joint uncertainty.</p>
<p><strong>Entity overlap</strong>: Mean |cosine| between each concept direction and the entity baseline,
reported per dimension. High overlap means the concept direction contains substantial entity-level information
that is removed by the residual procedure.</p>
""",

    "standalone": """
<p><strong>Standalone concept vectors</strong>: Unlike contrasts, standalone prompts describe concepts
<em>without entity framing</em> (no "human" or "AI" words). For example, "Phenomenal consciousness involves
subjective experience" rather than "Humans have phenomenal consciousness."</p>
<p><strong>Concept vector</strong> = mean activation across all prompts for a given concept (not a difference vector).</p>
<p><strong>Layer range</strong>: Layers 6&ndash;40 only (35 layers). Layer 0 (token embeddings) is particularly
problematic for standalone analysis: the mean embedding converges to the same vector regardless of prompt
content, creating spurious alignment with any probe whose layer-0 weights happen to correlate with
this universal direction. Layers 1&ndash;5 are also excluded for consistency with the contrast analyses.</p>
<p><strong>Key difference from contrasts</strong>: If standalone alignment is high, concept-probe alignment cannot
be explained by shared entity words in concept prompts and conversation probes.</p>
<p><strong>Different dimensions</strong>: Standalone includes <em>16_human</em> and <em>17_ai</em>
(concept vectors for "human" and "AI" as standalone concepts) and sysprompt variants (20-23),
but does not include dim 0 (entity baseline, which requires paired human/AI prompts).</p>
""",
}

ANALYSIS_INTERPRETATIONS = {
    "raw": """
<div class="key-finding">
<h3>Key Finding: Alignment Scales with Identity Cue Specificity</h3>
<p>Versions with explicit name-based identity cues (balanced_gpt, names, balanced_names) show
substantially higher alignment than versions with abstract labels or nonsense cues.
This ordering is consistent across both reading and control probes and across nearly all
conceptual dimensions.</p>
</div>
<div class="interpretation">
<h3>Interpretation</h3>
<p><strong>Name-based versions</strong>: The model's conceptual knowledge about human and AI minds is
geometrically aligned with the partner-identity representations formed during conversation.
This suggests the model recruits general semantic knowledge when adapting to different partners.</p>
<p><strong>Label/nonsense versions</strong>: Near-zero alignment indicates the probes in these versions
may rely on shallow pattern matching (detecting the label/codeword) rather than activating deep
conceptual representations.</p>
<p><strong>Caveat</strong>: This is <em>raw</em> alignment and includes shared entity-level variance.
See the residual analysis for concept-specific alignment after projecting out the entity baseline.</p>
</div>
""",

    "residual": """
<div class="key-finding">
<h3>Key Finding: Concept-Specific Alignment Persists After Entity Baseline Removal</h3>
<p>After projecting out the entity baseline direction (dim 0), alignment is reduced but the
version ordering is preserved. This demonstrates that alignment is not solely driven by
shared "human vs AI" entity features — concept-specific content contributes independently.</p>
</div>
<div class="interpretation">
<h3>Interpretation</h3>
<p><strong>Entity overlap</strong>: Most concept dimensions show high overlap with the entity baseline
(&gt;0.9 mean |cosine|), confirming that concept vectors contain substantial entity-level information.
This is expected — concepts like "phenomenology" and "emotions" are inherently related to the human/AI
distinction.</p>
<p><strong>Residual alignment</strong>: The fact that alignment persists after removing entity-level variance
means the conversational probes capture something beyond "is this a human or AI?" — they encode
concept-specific information about <em>what kind</em> of mind the partner has.</p>
<p><strong>Nonsense versions</strong>: Should show minimal residual alignment, confirming their probes
lack concept-specific structure.</p>
</div>
""",

    "standalone": """
<div class="key-finding">
<h3>Key Finding: Standalone Alignment Rules Out Lexical Overlap</h3>
<p>Standalone concept vectors — which contain no "human" or "AI" words — still align with
conversational probes. This is strong evidence that alignment reflects genuine conceptual structure,
not shared surface-level vocabulary between concept prompts and conversation data.</p>
</div>
<div class="interpretation">
<h3>Interpretation</h3>
<p><strong>Entity concept vectors (16_human, 17_ai)</strong>: These capture the model's standalone concept
of "human" and "AI" without paired contrast. Their alignment with conversational probes
indicates the model's general knowledge about these entities is recruited during conversation.</p>
<p><strong>Sysprompt variants</strong>: Multiple system prompt formulations (talkto, bare) probe how
sensitive alignment is to prompt framing. Consistency across variants suggests robust conceptual structure.</p>
<p><strong>Comparison to raw/residual</strong>: Standalone alignment is typically higher in absolute magnitude
than contrasts alignment, because the mean activation vector captures more variance than the
difference vector. The relative version ordering (name-based > abstract) is the key comparison.</p>
</div>
""",
}


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_html(dim_data, analysis_type):
    ainfo = ANALYSIS_TYPES[analysis_type]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    available_versions = get_available_versions(dim_data)
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    control_dims_list = [d for d in dim_data if dim_data[d]["category"] == "Control"]
    sorted_dims = sort_dims(dim_data, analysis_type)

    # Global max for uniform scale
    all_r2 = []
    for dd in dim_data.values():
        for vd in dd["versions"].values():
            all_r2.extend([vd["reading_r2"], vd["control_r2"]])
            all_r2.extend(vd["reading_ci"])
            all_r2.extend(vd["control_ci"])
    global_max = max(all_r2) * 1.05 if all_r2 else 0.001

    # Generate SVG figures
    reading_bars = svg_grouped_bar(dim_data, "reading", global_max, analysis_type)
    control_bars = svg_grouped_bar(dim_data, "control", global_max, analysis_type)
    reading_heatmap = svg_heatmap(dim_data, "reading", global_max, analysis_type)
    control_heatmap = svg_heatmap(dim_data, "control", global_max, analysis_type)
    summary_bars = svg_version_summary(dim_data, analysis_type)

    gmax_display = f"{global_max*1000:.2f}"

    # Pre-compute table fragments
    version_desc_rows = ''.join(
        "<tr><td><strong>" + VERSION_LABELS[v] + "</strong></td><td>"
        + VERSION_DESCRIPTIONS[v] + "</td></tr>"
        for v in available_versions
    )

    # Summary table
    summary_table_rows = ''
    for v in available_versions:
        r_m = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_m = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        r_c = [dim_data[d]["versions"][v]["reading_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        c_c = [dim_data[d]["versions"][v]["control_r2"] for d in control_dims_list if v in dim_data[d]["versions"]]
        rm = sum(r_m) / len(r_m) * 1000 if r_m else 0
        cm = sum(c_m) / len(c_m) * 1000 if c_m else 0
        rc = sum(r_c) / len(r_c) * 1000 if r_c else 0
        cc = sum(c_c) / len(c_c) * 1000 if c_c else 0
        summary_table_rows += (
            "<tr><td><strong>" + VERSION_LABELS[v] + "</strong></td>"
            + f"<td>{rm:.3f}</td><td>{cm:.3f}</td>"
            + f"<td>{rc:.3f}</td><td>{cc:.3f}</td></tr>\n"
        )

    # Full data table
    data_table_rows = ''
    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        for v in available_versions:
            if v not in dd["versions"]:
                continue
            vd = dd["versions"][v]
            overlap_col = ""
            if "entity_overlap" in vd:
                overlap_col = f"<td>{vd['entity_overlap']:.4f}</td>"
            data_table_rows += (
                f'<tr><td>{dd["name"]}</td><td>{dd["category"]}</td>'
                f'<td>{VERSION_LABELS[v]}</td>'
                f'<td>{vd["reading_r2"]*1000:.3f}</td>'
                f'<td>[{vd["reading_ci"][0]*1000:.3f}, {vd["reading_ci"][1]*1000:.3f}]</td>'
                f'<td>{vd["control_r2"]*1000:.3f}</td>'
                f'<td>[{vd["control_ci"][0]*1000:.3f}, {vd["control_ci"][1]*1000:.3f}]</td>'
                + overlap_col + '</tr>\n'
            )

    overlap_th = ""
    if analysis_type == "residual":
        overlap_th = "<th>Entity Overlap</th>"

    methods_html = ANALYSIS_METHODS[analysis_type]
    interp_html = ANALYSIS_INTERPRETATIONS[analysis_type]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 3: {ainfo["title"]} \u2014 Cross-Version Comparison</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; line-height: 1.6; }}
  h1 {{ border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
  h2 {{ color: #1565C0; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
  h3 {{ color: #333; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 30px; }}
  .section {{ margin: 30px 0; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.85em; }}
  th, td {{ padding: 8px 10px; border: 1px solid #ddd; text-align: center; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .figure-container {{ margin: 20px 0; background: #fafafa; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; }}
  .caption {{ font-size: 0.85em; color: #555; margin-top: 10px; padding: 8px; background: #f5f5f5; border-left: 3px solid #2196F3; }}
  .interpretation {{ background: #FFF8E1; padding: 15px; border-radius: 8px; border-left: 4px solid #FF9800; margin: 20px 0; }}
  .key-finding {{ background: #E8F5E9; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; margin: 15px 0; }}
  .method-box {{ background: #E3F2FD; padding: 15px; border-radius: 8px; margin: 15px 0; }}
  code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Experiment 3: {ainfo["title"]} \u2014 Cross-Version Comparison</h1>
<div class="meta">
    <p>Generated: {timestamp} | Analysis: Phase 2a ({ainfo["title"]})</p>
    <p>Rachel C. Metzgar, Princeton University, Graziano Lab</p>
</div>

<h2>1. Overview</h2>
<div class="section">
<p>{ainfo["description"]}</p>

<div class="method-box">
<h3>Analysis Details</h3>
{methods_html}
<p><strong>Conversational probes</strong> (version-specific):</p>
<table style="width:auto; margin: 10px 0;">
{version_desc_rows}
</table>
<p><strong>Two probe types</strong>: <strong>Reading probe</strong> (classifies partner identity from model hidden states
while reading partner messages) and <strong>Control probe</strong> (captures the model's behavioral adaptation representation).</p>
</div>
</div>

<h2>2. Summary: Mean Alignment by Version (Mental Dimensions)</h2>
<div class="section">
<table>
<tr><th>Version</th>
<th>Reading R\u00b2 (Mental, \u00d710\u207b\u00b3)</th>
<th>Control R\u00b2 (Mental, \u00d710\u207b\u00b3)</th>
<th>Reading R\u00b2 (Control dims, \u00d710\u207b\u00b3)</th>
<th>Control R\u00b2 (Control dims, \u00d710\u207b\u00b3)</th></tr>
{summary_table_rows}
</table>
<div class="caption"><strong>Table 1.</strong> Mean R\u00b2 alignment averaged across mental dimensions and control dimensions,
for each data version. Values \u00d710\u207b\u00b3. Higher = stronger geometric alignment between conceptual and conversational representations.</div>
</div>

<h2>3. Version Summary (Mental Dimensions Only)</h2>
<div class="figure-container">
{summary_bars}
<div class="caption"><strong>Figure 1.</strong> Mean alignment (R\u00b2) averaged across mental dimensions.
Blue = reading probes; orange = control probes.</div>
</div>

<h2>4. Per-Dimension Alignment: Reading Probes</h2>
<div class="figure-container">
{reading_bars}
<div class="caption"><strong>Figure 2.</strong> Per-dimension R\u00b2 alignment with reading probes across all six versions.
Bars = mean R\u00b2 across layers; whiskers = 95% bootstrap CI (1,000 iterations).
Uniform y-axis (max = {gmax_display} \u00d710\u207b\u00b3) across all figures for direct comparison.</div>
</div>

<h2>5. Per-Dimension Alignment: Control Probes</h2>
<div class="figure-container">
{control_bars}
<div class="caption"><strong>Figure 3.</strong> Per-dimension R\u00b2 alignment with control probes across all six versions.
Same layout and scale as Figure 2.</div>
</div>

<h2>6. Heatmap: Reading Probe Alignment</h2>
<div class="figure-container">
{reading_heatmap}
<div class="caption"><strong>Figure 4.</strong> Heatmap of reading probe R\u00b2 across dimensions and versions.
Cell values \u00d710\u207b\u00b3. Uniform color scale (0 to {gmax_display} \u00d710\u207b\u00b3). Hover for exact values.</div>
</div>

<h2>7. Heatmap: Control Probe Alignment</h2>
<div class="figure-container">
{control_heatmap}
<div class="caption"><strong>Figure 5.</strong> Heatmap of control probe R\u00b2 across dimensions and versions.
Same color scale as Figure 4.</div>
</div>

<h2>8. Full Data Table</h2>
<div class="section" style="overflow-x: auto;">
<table>
<tr><th>Dimension</th><th>Category</th><th>Version</th>
<th>Reading R\u00b2 (\u00d710\u207b\u00b3)</th><th>Reading 95% CI</th>
<th>Control R\u00b2 (\u00d710\u207b\u00b3)</th><th>Control 95% CI</th>{overlap_th}</tr>
{data_table_rows}
</table>
<div class="caption"><strong>Table 2.</strong> Complete alignment statistics. R\u00b2 and 95% bootstrap CIs \u00d710\u207b\u00b3.</div>
</div>

<h2>9. Interpretation</h2>
{interp_html}

<h2>10. Methods</h2>
<div class="method-box">
<p><strong>Model</strong>: LLaMA-2-13B-Chat</p>
<p><strong>Probes</strong>: Logistic linear probes (5,120 \u2192 1) from Exp 2, turn 5.</p>
<p><strong>Layer range</strong>: Layers 6\u201340 (35 of 41). Layers 0\u20135 excluded (embedding artifact + prompt-format confound).</p>
<p><strong>Bootstrap</strong>: 1,000 iterations, prompt-level resampling.</p>
<p><strong>Software</strong>: PyTorch, NumPy. Script: <code>exp_3/code/analysis/alignment/2a_alignment_analysis.py</code></p>
</div>

</body>
</html>"""
    return html


# ============================================================================
# MARKDOWN GENERATION
# ============================================================================

def generate_markdown(dim_data, analysis_type):
    ainfo = ANALYSIS_TYPES[analysis_type]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    available_versions = get_available_versions(dim_data)
    mental_dims = [d for d in dim_data if dim_data[d]["category"] == "Mental"]
    sorted_dims = sort_dims(dim_data, analysis_type)

    lines = []
    lines.append(f"# Exp 3: {ainfo['title']} \u2014 Cross-Version Comparison")
    lines.append(f"\n*Generated: {timestamp}*\n")
    lines.append(f"> {ainfo['description']}\n")

    # Summary table
    lines.append("## Summary (Mental Dimensions)\n")
    lines.append("| Version | Reading R\u00b2 (\u00d710\u207b\u00b3) | Control R\u00b2 (\u00d710\u207b\u00b3) | Description |")
    lines.append("|---------|---|---|---|")
    for v in available_versions:
        r_vals = [dim_data[d]["versions"][v]["reading_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        c_vals = [dim_data[d]["versions"][v]["control_r2"] for d in mental_dims if v in dim_data[d]["versions"]]
        rm = sum(r_vals) / len(r_vals) * 1000 if r_vals else 0
        cm = sum(c_vals) / len(c_vals) * 1000 if c_vals else 0
        lines.append(f"| {VERSION_LABELS[v]} | {rm:.3f} | {cm:.3f} | {VERSION_DESCRIPTIONS[v]} |")

    # Reading table
    lines.append("\n## Per-Dimension: Reading Probe R\u00b2 (\u00d710\u207b\u00b3)\n")
    header = "| Dimension | Category | " + " | ".join(VERSION_LABELS[v] for v in available_versions) + " |"
    sep = "|---|---|" + "|".join("---" for _ in available_versions) + "|"
    lines.append(header)
    lines.append(sep)
    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        vals = [f"{dd['versions'][v]['reading_r2']*1000:.3f}" if v in dd["versions"] else "\u2014"
                for v in available_versions]
        lines.append(f"| {dd['name']} | {dd['category']} | " + " | ".join(vals) + " |")

    # Control table
    lines.append("\n## Per-Dimension: Control Probe R\u00b2 (\u00d710\u207b\u00b3)\n")
    lines.append(header)
    lines.append(sep)
    for dim_id in sorted_dims:
        dd = dim_data[dim_id]
        vals = [f"{dd['versions'][v]['control_r2']*1000:.3f}" if v in dd["versions"] else "\u2014"
                for v in available_versions]
        lines.append(f"| {dd['name']} | {dd['category']} | " + " | ".join(vals) + " |")

    # Entity overlap table (residual only)
    if analysis_type == "residual":
        lines.append("\n## Entity Overlap (Mean |cosine| with baseline)\n")
        lines.append("| Dimension | Entity Overlap |")
        lines.append("|---|---|")
        # Entity overlap is the same for all versions (it's a property of the concept vector)
        v0 = available_versions[0]
        for dim_id in sorted_dims:
            dd = dim_data[dim_id]
            if v0 in dd["versions"] and "entity_overlap" in dd["versions"][v0]:
                lines.append(f"| {dd['name']} | {dd['versions'][v0]['entity_overlap']:.4f} |")

    lines.append("\n## Methods\n")
    lines.append(f"- **Analysis**: {ainfo['title']}")
    lines.append("- **Model**: LLaMA-2-13B-Chat")
    lines.append("- **Probes**: Exp 2, turn 5, logistic per-layer (reading + control)")
    lines.append("- **Layer range**: Layers 6\u201340 (35 of 41). Layers 0\u20135 excluded (embedding artifact + prompt-format confound).")
    lines.append("- **Bootstrap**: 1,000 iterations (prompt resampling)")
    lines.append("- **Script**: `exp_3/code/analysis/alignment/2a_alignment_analysis.py`")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate alignment comparison reports")
    parser.add_argument("--type", choices=["raw", "residual", "standalone", "all"],
                        default="all", help="Which analysis type (default: all)")
    args = parser.parse_args()

    types = ["raw", "residual", "standalone"] if args.type == "all" else [args.type]
    out_dir = Path(__file__).resolve().parent

    for atype in types:
        print(f"\n{'='*60}")
        print(f"  Generating {atype} comparison report")
        print(f"{'='*60}")

        all_data = load_all_summaries(atype)
        print(f"  Loaded {len(all_data)} versions")
        if not all_data:
            print("  ERROR: No data, skipping.")
            continue

        dim_data = organize_data(all_data, atype)
        print(f"  {len(dim_data)} dimensions")

        html = generate_html(dim_data, atype)
        html_path = out_dir / f"{atype}_comparison.html"
        with open(html_path, "w") as f:
            f.write(html)
        print(f"  HTML: {html_path}")

        md = generate_markdown(dim_data, atype)
        md_path = out_dir / f"{atype}_comparison.md"
        with open(md_path, "w") as f:
            f.write(md)
        print(f"  MD:   {md_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
