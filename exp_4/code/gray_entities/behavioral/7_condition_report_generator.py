#!/usr/bin/env python3
"""
Generate per-condition HTML reports for Gray replication behavior results.

Produces 4 separate reports:
  - pairwise_with_self.html
  - pairwise_without_self.html
  - individual_with_self.html
  - individual_without_self.html

Each report includes:
  - Scree plot, loadings table, entity factor scores table
  - Side-by-side scatter: Model vs Human (Gray et al.)
  - Axis-aligned scatter: Model (F2→, F1↑) vs Human (Exp→, Ag↑)
  - Human correlation stats
  - Entity colors: teal gradient by Gray E+A score

Reads from:  data_dir("gray_entities", "behavioral", condition)/
Writes to:   results_dir("gray_entities", "behavioral", condition)/

Usage:
    python behavior/make_condition_reports.py --model llama2_13b_base

Env: llama2_env (CPU-only, login node OK)
"""
import os, sys, json, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import config, set_model, add_model_argument, data_dir, results_dir
from entities.gray_entities import CAPACITY_PROMPTS, GRAY_ET_AL_SCORES

# ── Teal gradient by Gray E+A ──
def entity_color(entity_key):
    """Return hex color: darker teal = higher Gray E+A."""
    scores = GRAY_ET_AL_SCORES
    totals = {k: e + a for k, (e, a) in scores.items()}
    mn, mx = min(totals.values()), max(totals.values())
    rng = mx - mn if mx > mn else 1
    t = (totals.get(entity_key, 0) - mn) / rng
    r = int(140 * (1 - t))
    g = int(200 - 123 * t)
    b = int(192 - 128 * t)
    return f"#{r:02x}{g:02x}{b:02x}"

ENTITY_COLORS = {k: entity_color(k) for k in GRAY_ET_AL_SCORES}


def _rescale_01(arr):
    """Rescale array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn) if mx > mn else np.full_like(arr, 0.5)

SHORT_NAMES = {
    "dead_woman": "dead_w", "frog": "frog", "robot": "robot",
    "fetus": "fetus", "pvs_patient": "pvs", "god": "god",
    "dog": "dog", "chimpanzee": "chimp", "baby": "baby",
    "girl": "girl", "adult_woman": "woman", "adult_man": "man",
    "you_self": "self",
}

def make_loadings_bar_svg(capacity_keys, loadings_col, eigenvalue, explained_pct,
                          factor_idx, title_suffix="", width=600, bar_h=18):
    """Generate an inline SVG horizontal bar chart of factor loadings.

    Bars colored by original Gray factor (E=teal, A=amber).
    Sorted by loading value (ascending → lowest at bottom, highest at top).
    """
    COLORS = {"E": "#0d9488", "A": "#d97706", "?": "#94a3b8"}

    # Sort ascending so highest loading is at top visually
    sort_idx = np.argsort(loadings_col)
    sorted_vals = loadings_col[sort_idx]
    sorted_caps = [capacity_keys[i] for i in sort_idx]
    sorted_factors = []
    for c in sorted_caps:
        _, gf = CAPACITY_PROMPTS.get(c, ("", "?"))
        sorted_factors.append(gf)

    n = len(sorted_caps)
    label_w = 120
    plot_w = width - label_w - 20
    margin_top = 30
    height = margin_top + n * (bar_h + 4) + 40

    # Scale: loadings typically -1 to 1
    max_abs = max(abs(sorted_vals.min()), abs(sorted_vals.max()), 0.5)
    max_abs = max(max_abs, 1.0)  # at least -1 to 1

    zero_x = label_w + plot_w / 2  # center = 0
    scale = (plot_w / 2) / max_abs

    lines = []
    lines.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="font-family:inherit">')
    title = f"Factor {factor_idx+1} Loadings (λ={eigenvalue:.2f}, {explained_pct:.1f}% var)"
    if title_suffix:
        title += f" — {title_suffix}"
    lines.append(f'<text x="{width//2}" y="18" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">{title}</text>')

    # Zero line
    lines.append(f'<line x1="{zero_x:.0f}" y1="{margin_top}" x2="{zero_x:.0f}" y2="{margin_top + n*(bar_h+4)}" stroke="#94a3b8" stroke-width="1"/>')
    # ±0.4 guide lines
    for ref in [0.4, -0.4]:
        rx = zero_x + ref * scale
        if label_w < rx < width - 10:
            lines.append(f'<line x1="{rx:.0f}" y1="{margin_top}" x2="{rx:.0f}" y2="{margin_top + n*(bar_h+4)}" stroke="#e2e8f0" stroke-dasharray="4,3"/>')

    for i, (cap, val, gf) in enumerate(zip(sorted_caps, sorted_vals, sorted_factors)):
        y = margin_top + i * (bar_h + 4)
        color = COLORS.get(gf, "#94a3b8")
        bar_x = zero_x if val >= 0 else zero_x + val * scale
        bar_w = abs(val) * scale

        lines.append(f'<text x="{label_w - 4}" y="{y + bar_h - 4}" text-anchor="end" font-size="9" fill="#475569">{cap}</text>')
        lines.append(f'<rect x="{bar_x:.1f}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" fill="{color}" rx="2"/>')
        # Value label
        if bar_w > 25:
            tx = bar_x + bar_w / 2
            lines.append(f'<text x="{tx:.0f}" y="{y + bar_h - 4}" text-anchor="middle" font-size="8" fill="white" font-weight="600">{val:+.2f}</text>')

    # Legend
    ly = margin_top + n * (bar_h + 4) + 12
    lines.append(f'<rect x="{label_w}" y="{ly}" width="12" height="12" fill="#0d9488" rx="2"/>')
    lines.append(f'<text x="{label_w + 16}" y="{ly + 10}" font-size="9" fill="#475569">Experience (Gray)</text>')
    lines.append(f'<rect x="{label_w + 120}" y="{ly}" width="12" height="12" fill="#d97706" rx="2"/>')
    lines.append(f'<text x="{label_w + 136}" y="{ly + 10}" font-size="9" fill="#475569">Agency (Gray)</text>')

    lines.append('</svg>')
    return '\n'.join(lines)


CSS = """
*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:#1e293b;background:#f8fafc;line-height:1.6;max-width:1100px;margin:0 auto;padding:2rem}
h1{font-size:1.8rem;color:#0f172a;border-bottom:3px solid #0d9488;padding-bottom:.5rem;margin-bottom:1rem}
h2{font-size:1.4rem;color:#0f172a;margin:2rem 0 .75rem;border-bottom:2px solid #e2e8f0;padding-bottom:.4rem}
h3{font-size:1.1rem;color:#334155;margin:1.2rem 0 .5rem}
h4{font-size:1rem;color:#475569;margin:1rem 0 .4rem}
p,li{font-size:.93rem;margin-bottom:.5rem}
ul,ol{padding-left:1.5rem}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1.25rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.04)}
table{width:100%;border-collapse:collapse;margin:.75rem 0;font-size:.85rem}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid #e2e8f0}
th{background:#f1f5f9;font-weight:600;color:#334155}
td{color:#475569}
.num{text-align:right;font-variant-numeric:tabular-nums}
.sig{color:#ef4444;font-weight:700}
.tag-e{background:#ccfbf1;color:#0d9488;padding:1px 6px;border-radius:3px;font-size:.78rem;font-weight:600}
.tag-a{background:#fef3c7;color:#d97706;padding:1px 6px;border-radius:3px;font-size:.78rem;font-weight:600}
.charts-row{display:flex;gap:1.5rem;flex-wrap:wrap;justify-content:center}
.chart-wrap{text-align:center}
.small{font-size:.82rem;color:#64748b}
.bold{font-weight:700}
.legend-box{display:inline-flex;gap:1rem;align-items:center;margin:.5rem 0;font-size:.82rem}
.legend-swatch{display:inline-block;width:14px;height:14px;border-radius:2px;vertical-align:middle}
"""


def make_scatter_svg(title, x_label, y_label, entity_keys, x_vals, y_vals,
                     width=320, height=320, x_label_color="#475569",
                     y_label_color="#475569"):
    """Generate an SVG scatter plot with teal-gradient entity dots."""
    plot_size = 240
    margin_left = 50
    margin_top = 28

    lines = []
    lines.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="font-family:inherit">')
    lines.append(f'<text x="{width//2}" y="16" text-anchor="middle" font-size="11" font-weight="600" fill="#334155">{title}</text>')
    lines.append(f'<g transform="translate({margin_left},{margin_top})">')
    # Axes
    lines.append(f'<line x1="0" y1="{plot_size}" x2="{plot_size}" y2="{plot_size}" stroke="#cbd5e1"/>')
    lines.append(f'<line x1="0" y1="0" x2="0" y2="{plot_size}" stroke="#cbd5e1"/>')
    lines.append(f'<text x="{plot_size//2}" y="{plot_size+22}" text-anchor="middle" font-size="10" fill="{x_label_color}">{x_label}</text>')
    lines.append(f'<text x="-16" y="{plot_size//2}" text-anchor="middle" font-size="10" fill="{y_label_color}" transform="rotate(-90,-16,{plot_size//2})">{y_label}</text>')
    # Grid
    for frac in [0.25, 0.5, 0.75]:
        gx = int(frac * plot_size)
        gy = int(frac * plot_size)
        lines.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{plot_size}" stroke="#f1f5f9"/>')
        lines.append(f'<line x1="0" y1="{gy}" x2="{plot_size}" y2="{gy}" stroke="#f1f5f9"/>')
    # 0/1 labels
    lines.append(f'<text x="0" y="{plot_size+12}" font-size="8" fill="#94a3b8">0</text>')
    lines.append(f'<text x="{plot_size-3}" y="{plot_size+12}" font-size="8" fill="#94a3b8">1</text>')
    lines.append(f'<text x="-4" y="5" text-anchor="end" font-size="8" fill="#94a3b8">1</text>')
    # Points
    for i, ek in enumerate(entity_keys):
        cx = x_vals[i] * plot_size
        cy = plot_size - y_vals[i] * plot_size
        color = ENTITY_COLORS.get(ek, "#64748b")
        short = SHORT_NAMES.get(ek, ek[:6])
        # Offset text to avoid overlap
        tx = cx + 5
        ty = cy - 3
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="{color}"/>'
                      f'<text x="{tx:.0f}" y="{ty:.0f}" font-size="7" fill="{color}">{short}</text>')
    lines.append('</g>')
    lines.append('</svg>')
    return '\n'.join(lines)


def make_legend_html():
    """Color legend showing teal gradient meaning."""
    items = sorted(GRAY_ET_AL_SCORES.items(), key=lambda x: x[1][0]+x[1][1])
    low = items[0]
    high = items[-1]
    return f"""
<div class="legend-box">
<span class="legend-swatch" style="background:{ENTITY_COLORS[low[0]]}"></span>
Low E+A ({SHORT_NAMES[low[0]]}: {low[1][0]+low[1][1]:.2f})
&nbsp;&rarr;&nbsp;
<span class="legend-swatch" style="background:{ENTITY_COLORS[high[0]]}"></span>
High E+A ({SHORT_NAMES[high[0]]}: {high[1][0]+high[1][1]:.2f})
&nbsp;&nbsp;(darker teal = higher combined Experience + Agency in Gray et al.)
</div>"""


def generate_report(model_key, method, condition):
    """Generate one condition report.

    method: 'pairwise' or 'individual'
    condition: 'with_self' or 'without_self'
    """
    set_model(model_key)
    ddir = str(data_dir("gray_entities", "behavioral", condition))
    rdir = str(results_dir("gray_entities", "behavioral", condition))

    pca_path = os.path.join(ddir, f"{method}_pca_results.npz")
    corr_path = os.path.join(ddir, f"{method}_human_correlations.json")
    comp_path = os.path.join(ddir, f"{method}_human_comparisons.json")
    consistency_path = os.path.join(ddir, f"{method}_consistency_stats.json")

    if not os.path.exists(pca_path):
        print(f"  Not found: {pca_path}")
        return None

    pca = np.load(pca_path)
    rotated = pca["rotated_loadings"]
    scores_01 = pca["factor_scores_01"]
    eigenvalues = pca["eigenvalues"]
    explained = pca["explained_var_ratio"]

    if "entity_keys" in pca:
        entity_keys = list(pca["entity_keys"])
    elif "character_keys" in pca:
        entity_keys = list(pca["character_keys"])
    else:
        entity_keys = list(GRAY_ET_AL_SCORES.keys())
        if condition == "without_self":
            entity_keys = [k for k in entity_keys if k != "you_self"]

    capacity_keys = list(pca["capacity_keys"])
    n_entities = len(entity_keys)
    n_caps = len(capacity_keys)
    n_retained = int(np.sum(eigenvalues > 1.0))

    human_corr = {}
    if os.path.exists(corr_path):
        with open(corr_path) as f:
            human_corr = json.load(f)

    comparisons = {}
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comparisons = json.load(f)

    consistency = {}
    if os.path.exists(consistency_path):
        with open(consistency_path) as f:
            consistency = json.load(f)

    method_label = method.capitalize()
    self_label = condition.replace("_", " ").title()
    title = f"Gray Replication — {method_label}, {self_label} ({n_entities} entities)"

    html = []
    html.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — {config.MODEL_LABEL}</title>
<style>{CSS}</style>
</head><body>
<h1>{title}</h1>
<p class="small">{config.MODEL_LABEL} &mdash; Replication of Gray, Gray, &amp; Wegner (2007)</p>
""")

    # ── Summary ──
    html.append(f"""
<div class="card">
<strong>Summary:</strong> {n_caps} mental capacities, {n_entities} entities.
<strong>{n_retained}</strong> factor(s) retained (eigenvalue &gt; 1),
explaining <strong>{np.sum(explained[:n_retained])*100:.1f}%</strong> of variance.
</div>
{make_legend_html()}
""")

    # ── Scree ──
    html.append('<h2>1. Scree Plot</h2><div class="card">')
    html.append('<table><tr><th>Component</th><th class="num">Eigenvalue</th>'
                '<th class="num">Variance</th><th class="num">Cumulative</th></tr>')
    cum = 0
    for i in range(min(5, len(eigenvalues))):
        pct = 100 * eigenvalues[i] / np.sum(eigenvalues)
        cum += pct
        marker = " *" if eigenvalues[i] > 1 else ""
        html.append(f'<tr><td>PC{i+1}{marker}</td>'
                     f'<td class="num">{eigenvalues[i]:.2f}</td>'
                     f'<td class="num">{pct:.1f}%</td>'
                     f'<td class="num">{cum:.1f}%</td></tr>')
    html.append('</table>')
    html.append(f'<p class="small">* eigenvalue &gt; 1 (retained). '
                f'Gray et al.: eigenvalues 15.85 and 1.46, explaining 97%.</p>')
    html.append('</div>')

    # ── Loadings ──
    html.append('<h2>2. Capacity Loadings (Varimax-Rotated)</h2><div class="card">')
    n_show = min(n_retained, 4)
    html.append('<table><tr><th>Capacity</th><th>Gray Factor</th>')
    for fi in range(n_show):
        html.append(f'<th class="num">F{fi+1} ({explained[fi]*100:.1f}%)</th>')
    html.append('</tr>')
    for c_idx, cap in enumerate(capacity_keys):
        _, gf = CAPACITY_PROMPTS.get(cap, ("", "?"))
        tag = f'<span class="tag-e">E</span>' if gf == "E" else f'<span class="tag-a">A</span>'
        html.append(f'<tr><td>{cap}</td><td>{tag}</td>')
        for fi in range(n_show):
            v = rotated[c_idx, fi]
            bold = ' class="num bold"' if abs(v) > 0.4 else ' class="num"'
            html.append(f'<td{bold}>{v:+.3f}</td>')
        html.append('</tr>')
    html.append('</table>')
    html.append('<p class="small">Bold = |loading| &gt; 0.4. Compare to Gray: Experience capacities '
                'loaded .67&ndash;.97 on Factor 1, Agency .73&ndash;.97 on Factor 2.</p>')

    # Loading bar charts (inline SVG)
    html.append('<h3>Loading Bar Charts</h3>')
    for fi in range(n_show):
        html.append(make_loadings_bar_svg(
            capacity_keys, rotated[:, fi], eigenvalues[fi],
            explained[fi] * 100, fi, title_suffix="All Entities"))
    html.append('</div>')

    # ── Excl PCA section ──
    excl_pca_path = os.path.join(ddir, f"{method}_pca_results_excl_fetus_god.npz")
    excl_pca_data = None
    if os.path.exists(excl_pca_path):
        excl_pca_data = np.load(excl_pca_path)
        excl_rotated = excl_pca_data["rotated_loadings"]
        excl_scores_01 = excl_pca_data["factor_scores_01"]
        excl_eigenvalues = excl_pca_data["eigenvalues"]
        excl_explained = excl_pca_data["explained_var_ratio"]
        excl_entity_keys = list(excl_pca_data["entity_keys"])
        excl_capacity_keys = list(excl_pca_data["capacity_keys"])
        excl_n_retained = int(np.sum(excl_eigenvalues > 1.0))
        excl_n_show = min(excl_n_retained, 4)

        html.append('<h2>2b. Factor Analysis Excluding Fetus + God</h2>')
        html.append(f"""<div class="card">
<p class="small">PCA re-run from scratch on the {len(excl_entity_keys)}-entity means matrix
(fetus and god removed before computing correlations/eigenvalues).
This can change factor structure, not just drop two rows.</p>""")

        # Scree
        html.append('<h3>Scree</h3>')
        html.append('<table><tr><th>Component</th><th class="num">Eigenvalue</th>'
                     '<th class="num">Variance</th><th class="num">Cumulative</th></tr>')
        cum = 0
        for i in range(min(5, len(excl_eigenvalues))):
            pct = 100 * excl_eigenvalues[i] / np.sum(excl_eigenvalues)
            cum += pct
            marker = " *" if excl_eigenvalues[i] > 1 else ""
            html.append(f'<tr><td>PC{i+1}{marker}</td>'
                         f'<td class="num">{excl_eigenvalues[i]:.2f}</td>'
                         f'<td class="num">{pct:.1f}%</td>'
                         f'<td class="num">{cum:.1f}%</td></tr>')
        html.append('</table>')

        # Loadings table
        html.append('<h3>Capacity Loadings (Varimax-Rotated)</h3>')
        html.append('<table><tr><th>Capacity</th><th>Gray Factor</th>')
        for fi in range(excl_n_show):
            html.append(f'<th class="num">F{fi+1} ({excl_explained[fi]*100:.1f}%)</th>')
        html.append('</tr>')
        for c_idx, cap in enumerate(excl_capacity_keys):
            _, gf = CAPACITY_PROMPTS.get(cap, ("", "?"))
            tag = f'<span class="tag-e">E</span>' if gf == "E" else f'<span class="tag-a">A</span>'
            html.append(f'<tr><td>{cap}</td><td>{tag}</td>')
            for fi in range(excl_n_show):
                v = excl_rotated[c_idx, fi]
                bold = ' class="num bold"' if abs(v) > 0.4 else ' class="num"'
                html.append(f'<td{bold}>{v:+.3f}</td>')
            html.append('</tr>')
        html.append('</table>')

        # Loading bar charts
        html.append('<h3>Loading Bar Charts (Excl Fetus + God)</h3>')
        for fi in range(excl_n_show):
            html.append(make_loadings_bar_svg(
                excl_capacity_keys, excl_rotated[:, fi], excl_eigenvalues[fi],
                excl_explained[fi] * 100, fi, title_suffix="Excl Fetus+God"))
        html.append('</div>')

    # ── Excl PCA (fetus+god+dead_woman) section ──
    excl_dead_pca_path = os.path.join(ddir, f"{method}_pca_results_excl_fetus_god_dead.npz")
    excl_dead_pca_data = None
    if os.path.exists(excl_dead_pca_path):
        excl_dead_pca_data = np.load(excl_dead_pca_path)
        excl_dead_rotated = excl_dead_pca_data["rotated_loadings"]
        excl_dead_scores_01 = excl_dead_pca_data["factor_scores_01"]
        excl_dead_eigenvalues = excl_dead_pca_data["eigenvalues"]
        excl_dead_explained = excl_dead_pca_data["explained_var_ratio"]
        excl_dead_entity_keys = list(excl_dead_pca_data["entity_keys"])
        excl_dead_capacity_keys = list(excl_dead_pca_data["capacity_keys"])
        excl_dead_n_retained = int(np.sum(excl_dead_eigenvalues > 1.0))
        excl_dead_n_show = min(excl_dead_n_retained, 4)

        html.append('<h2>2c. Factor Analysis Excluding Fetus, God, + Dead Woman</h2>')
        html.append(f"""<div class="card">
<p class="small">PCA re-run from scratch on the {len(excl_dead_entity_keys)}-entity means matrix
(fetus, god, and dead_woman removed before computing correlations/eigenvalues).</p>""")

        # Scree
        html.append('<h3>Scree</h3>')
        html.append('<table><tr><th>Component</th><th class="num">Eigenvalue</th>'
                     '<th class="num">Variance</th><th class="num">Cumulative</th></tr>')
        cum = 0
        for i in range(min(5, len(excl_dead_eigenvalues))):
            pct = 100 * excl_dead_eigenvalues[i] / np.sum(excl_dead_eigenvalues)
            cum += pct
            marker = " *" if excl_dead_eigenvalues[i] > 1 else ""
            html.append(f'<tr><td>PC{i+1}{marker}</td>'
                         f'<td class="num">{excl_dead_eigenvalues[i]:.2f}</td>'
                         f'<td class="num">{pct:.1f}%</td>'
                         f'<td class="num">{cum:.1f}%</td></tr>')
        html.append('</table>')

        # Loadings table
        html.append('<h3>Capacity Loadings (Varimax-Rotated)</h3>')
        html.append('<table><tr><th>Capacity</th><th>Gray Factor</th>')
        for fi in range(excl_dead_n_show):
            html.append(f'<th class="num">F{fi+1} ({excl_dead_explained[fi]*100:.1f}%)</th>')
        html.append('</tr>')
        for c_idx, cap in enumerate(excl_dead_capacity_keys):
            _, gf = CAPACITY_PROMPTS.get(cap, ("", "?"))
            tag = f'<span class="tag-e">E</span>' if gf == "E" else f'<span class="tag-a">A</span>'
            html.append(f'<tr><td>{cap}</td><td>{tag}</td>')
            for fi in range(excl_dead_n_show):
                v = excl_dead_rotated[c_idx, fi]
                bold = ' class="num bold"' if abs(v) > 0.4 else ' class="num"'
                html.append(f'<td{bold}>{v:+.3f}</td>')
            html.append('</tr>')
        html.append('</table>')

        # Loading bar charts
        html.append('<h3>Loading Bar Charts (Excl Fetus, God, Dead Woman)</h3>')
        for fi in range(excl_dead_n_show):
            html.append(make_loadings_bar_svg(
                excl_dead_capacity_keys, excl_dead_rotated[:, fi],
                excl_dead_eigenvalues[fi], excl_dead_explained[fi] * 100, fi,
                title_suffix="Excl Fetus+God+Dead"))
        html.append('</div>')

    # ── Correlation with humans ──
    def _render_comparison_table(subset_data, subset_label, n_show):
        """Render one comparison table (full or excl)."""
        parts = []
        parts.append(f'<h3>{subset_label} (n={subset_data.get("n_entities", "?")})</h3>')
        parts.append('<table><tr><th>Comparison</th>'
                     '<th class="num">Statistic</th><th class="num">p-value</th></tr>')
        # Overall: Procrustes + RV
        if "procrustes" in subset_data:
            disp = subset_data["procrustes"].get("disparity")
            p_proc = subset_data["procrustes"].get("p")
            if disp is not None:
                p_str = f'{p_proc:.4f}' if p_proc is not None else "&mdash;"
                sig = ' class="sig"' if p_proc is not None and p_proc < 0.05 else ""
                parts.append(f'<tr><td>Overall: Procrustes disparity</td>'
                             f'<td class="num">{disp:.4f}</td>'
                             f'<td class="num"{sig}>{p_str}</td></tr>')
        if "rv_coefficient" in subset_data:
            rv = subset_data["rv_coefficient"]["rv"]
            p_rv = subset_data["rv_coefficient"].get("p")
            p_str = f'{p_rv:.4f}' if p_rv is not None else "&mdash;"
            sig = ' class="sig"' if p_rv is not None and p_rv < 0.05 else ""
            parts.append(f'<tr><td>Overall: RV coefficient</td>'
                         f'<td class="num">{rv:.4f}</td>'
                         f'<td class="num"{sig}>{p_str}</td></tr>')
        # Per-factor correlations
        for fi in range(n_show):
            for dim, label in [("experience", "Experience"), ("agency", "Agency")]:
                key = f"f{fi+1}_{dim}"
                if key in subset_data:
                    rho = subset_data[key]["rho"]
                    p = subset_data[key]["p"]
                    sig = ' class="sig"' if p < 0.05 else ""
                    parts.append(f'<tr><td>F{fi+1} &harr; {label}</td>'
                                 f'<td class="num">rho={rho:+.3f}</td>'
                                 f'<td class="num"{sig}>{p:.4f}</td></tr>')
        # Combined mindedness
        for fi in range(n_show):
            key = f"f{fi+1}_combined"
            if key in subset_data:
                rho = subset_data[key]["rho"]
                p = subset_data[key]["p"]
                sig = ' class="sig"' if p < 0.05 else ""
                parts.append(f'<tr><td>F{fi+1} &harr; Combined (E+A)</td>'
                             f'<td class="num">rho={rho:+.3f}</td>'
                             f'<td class="num"{sig}>{p:.4f}</td></tr>')
        if "combined_combined" in subset_data:
            rho = subset_data["combined_combined"]["rho"]
            p = subset_data["combined_combined"]["p"]
            sig = ' class="sig"' if p < 0.05 else ""
            parts.append(f'<tr><td>(F1+F2) &harr; Combined (E+A)</td>'
                         f'<td class="num">rho={rho:+.3f}</td>'
                         f'<td class="num"{sig}>{p:.4f}</td></tr>')
        parts.append('</table>')
        return '\n'.join(parts)

    if comparisons:
        html.append('<h2>3. Statistical Comparison with Human Factor Scores</h2>')
        html.append("""<div class="card">
<h3>What each statistic measures</h3>
<table>
<tr><th style="width:25%">Statistic</th><th>Question it answers</th><th style="width:35%">How it is calculated</th></tr>
<tr>
<td><strong>Procrustes disparity</strong></td>
<td>Does the overall 2D shape of entity positions in the model&rsquo;s factor space
match the shape in Gray et al.&rsquo;s human factor space, allowing for rotation,
reflection, and uniform scaling?</td>
<td>Both point clouds (model [F1,&thinsp;F2] and human [Exp,&thinsp;Ag]) are centered and
normalized to unit sum-of-squares. One cloud is then optimally rotated/reflected
to minimize the sum of squared distances to the other
(<code>scipy.spatial.procrustes</code>). The residual sum of squares after alignment
is the <em>disparity</em>: 0&nbsp;=&nbsp;perfect match, 1&nbsp;=&nbsp;no better than
chance. <strong>p-value (permutation):</strong> We ask: &ldquo;Could this good a match
happen by accident?&rdquo; To find out, we randomly shuffle which model point goes with
which human point 10,000 times (breaking the true entity pairings). Each time, we
recompute Procrustes disparity. The p-value is the fraction of these random shuffles
that achieved a disparity as low or lower than the real one. If p&nbsp;&lt;&nbsp;.05,
the match is better than 95% of random pairings &mdash; the structural similarity is
unlikely to be coincidence.</td>
</tr>
<tr>
<td><strong>RV coefficient</strong></td>
<td>How similar are the multivariate covariance patterns of the two 2D point clouds?
(A multivariate generalization of R&sup2;.)</td>
<td>After centering both matrices X (model) and Y (human):
RV&nbsp;=&nbsp;trace(S<sub>XY</sub>&thinsp;S<sub>YX</sub>)&thinsp;/&thinsp;&radic;(trace(S<sub>XX</sub>&sup2;)&thinsp;&middot;&thinsp;trace(S<sub>YY</sub>&sup2;)),
where S<sub>XY</sub>&nbsp;=&nbsp;X&prime;Y. Range [0,&thinsp;1]; higher means more similar.
Think of it as: &ldquo;How much of the variance in one point cloud is shared with
the other?&rdquo;
<strong>p-value (permutation):</strong> Same logic as Procrustes &mdash; shuffle entity
labels 10,000 times, recompute RV each time. p is the fraction of shuffled RVs
that were as high or higher than the real one. A small p means the shared variance
is unlikely to be a coincidence of the point distributions.</td>
</tr>
<tr>
<td><strong>Spearman rho</strong><br>(e.g., F2 &harr; Experience)</td>
<td>Does a single model factor preserve the <em>rank ordering</em> of entities on a
single human dimension? For example, if humans rank adult&thinsp;man &gt; girl &gt; dog
on Experience, does model F2 put them in the same order?</td>
<td>Standard Spearman rank correlation between the model&rsquo;s entity scores on one
factor and the human scores on one dimension. Computed separately for each of the
four (F1,&thinsp;F2)&nbsp;&times;&nbsp;(Experience,&thinsp;Agency) combinations.
<strong>p-value:</strong> exact (from <code>scipy.stats.spearmanr</code>).</td>
</tr>
<tr>
<td><strong>Combined Spearman</strong><br>(e.g., F1 &harr; E+A, or F1+F2 &harr; E+A)</td>
<td>Does the model capture the overall &ldquo;mindedness&rdquo; hierarchy &mdash; the
total amount of mind humans attribute to each entity &mdash; regardless of which
specific factor carries it? This collapses the 2D structure to 1D, losing the
Experience/Agency distinction, but provides a simple check of whether the model
ranks entities by overall mental capacity the way humans do.</td>
<td>Human combined score = Experience + Agency for each entity. Model combined =
F1 + F2. Spearman rank correlation between these sums, or between each individual
factor and the human sum. <strong>p-value:</strong> exact.</td>
</tr>
</table>
</div>
<div class="card">""")
        if "full" in comparisons:
            html.append(_render_comparison_table(comparisons["full"], "All Entities", n_show))
        if "excl_fetus_god" in comparisons:
            html.append(_render_comparison_table(
                comparisons["excl_fetus_god"],
                "Excluding Fetus + God (scores from full PCA)", n_show))
        if "excl_fetus_god_pca" in comparisons:
            html.append(_render_comparison_table(
                comparisons["excl_fetus_god_pca"],
                "Excluding Fetus + God (PCA re-run from scratch)", n_show))
        if "excl_fetus_god_dead" in comparisons:
            html.append(_render_comparison_table(
                comparisons["excl_fetus_god_dead"],
                "Excluding Fetus, God, Dead Woman (scores from full PCA)", n_show))
        if "excl_fetus_god_dead_pca" in comparisons:
            html.append(_render_comparison_table(
                comparisons["excl_fetus_god_dead_pca"],
                "Excluding Fetus, God, Dead Woman (PCA re-run from scratch)", n_show))
        html.append('</div>')
    elif human_corr:
        # Fallback to old format
        html.append('<h2>3. Correlation with Human Factor Scores</h2><div class="card">')
        html.append('<table><tr><th>Model Factor</th><th>vs Human</th>'
                    '<th class="num">Spearman rho</th><th class="num">p-value</th></tr>')
        for fi in range(n_show):
            for dim, label in [("experience", "Experience"), ("agency", "Agency")]:
                key = f"f{fi+1}_{dim}"
                if key in human_corr:
                    rho = human_corr[key]["rho"]
                    p = human_corr[key]["p"]
                    sig = ' class="sig"' if p < 0.05 else ""
                    html.append(f'<tr><td>F{fi+1}</td><td>{label}</td>'
                                f'<td class="num">{rho:+.3f}</td>'
                                f'<td class="num"{sig}>{p:.4f}</td></tr>')
        html.append('</table></div>')

    # ── Entity scores table ──
    html.append('<h2>4. Entity Factor Scores</h2><div class="card">')
    html.append('<table><tr><th>Entity</th>')
    for fi in range(min(n_show, 2)):
        html.append(f'<th class="num">Model F{fi+1}</th>')
    html.append('<th class="num">Human Exp</th><th class="num">Human Ag</th></tr>')
    for i, ek in enumerate(entity_keys):
        h_exp, h_ag = GRAY_ET_AL_SCORES.get(ek, (0, 0))
        color = ENTITY_COLORS.get(ek, "#475569")
        html.append(f'<tr><td style="color:{color};font-weight:600">{ek}</td>')
        for fi in range(min(n_show, 2)):
            html.append(f'<td class="num">{scores_01[i, fi]:.3f}</td>')
        html.append(f'<td class="num">{h_exp:.2f}</td><td class="num">{h_ag:.2f}</td></tr>')
    html.append('</table></div>')

    # ── Scatter plots ──
    html.append('<h2>5. Entity Positions: Model vs Human</h2>')

    # Original orientation: F1=x, F2=y vs Exp=x, Ag=y
    html.append('<h3>5a. Original Axes (Model: F1&rarr;, F2&uarr;)</h3>')
    html.append('<div class="charts-row">')
    f1 = scores_01[:, 0] if scores_01.shape[1] >= 1 else np.zeros(n_entities)
    f2 = scores_01[:, 1] if scores_01.shape[1] >= 2 else np.zeros(n_entities)
    html.append('<div class="chart-wrap">')
    html.append(make_scatter_svg(
        f"Model Factor Space ({method_label})", "F1", "F2",
        entity_keys, f1, f2))
    html.append('</div><div class="chart-wrap">')
    h_exp = np.array([GRAY_ET_AL_SCORES.get(k, (0, 0))[0] for k in entity_keys])
    h_ag = np.array([GRAY_ET_AL_SCORES.get(k, (0, 0))[1] for k in entity_keys])
    html.append(make_scatter_svg(
        "Human Factor Space (Gray et al.)", "Experience", "Agency",
        entity_keys, h_exp, h_ag,
        x_label_color="#0d9488", y_label_color="#d97706"))
    html.append('</div></div>')

    # Axis-aligned: F2=x, F1=y vs Exp=x, Ag=y
    html.append('<h3>5b. Axis-Aligned (Model: F2&rarr;, F1&uarr;)</h3>')
    html.append('<p class="small">Model axes swapped so F1 (Y) aligns with potential Agency '
                'direction and F2 (X) with potential Experience direction.</p>')
    html.append('<div class="charts-row">')
    html.append('<div class="chart-wrap">')
    html.append(make_scatter_svg(
        f"Model (F2&rarr;, F1&uarr;)", "F2", "F1",
        entity_keys, f2, f1))
    html.append('</div><div class="chart-wrap">')
    html.append(make_scatter_svg(
        "Human (Exp&rarr;, Ag&uarr;)", "Experience", "Agency",
        entity_keys, h_exp, h_ag,
        x_label_color="#0d9488", y_label_color="#d97706"))
    html.append('</div></div>')

    # Exclusion scatter plots: drop fetus + god, rescale to [0,1]
    EXCLUDE = {"fetus", "god"}
    keep_mask = np.array([k not in EXCLUDE for k in entity_keys])
    if keep_mask.sum() < n_entities:
        excl_keys = [k for k in entity_keys if k not in EXCLUDE]
        n_excl = len(excl_keys)

        excl_f1 = _rescale_01(f1[keep_mask])
        excl_f2 = _rescale_01(f2[keep_mask])
        excl_h_exp = _rescale_01(h_exp[keep_mask])
        excl_h_ag = _rescale_01(h_ag[keep_mask])

        html.append(f'<h3>5c. Excluding Fetus + God (n={n_excl}), Rescaled to [0,1]</h3>')
        html.append('<p class="small">Factor scores re-normalized to [0,1] within the remaining '
                    f'{n_excl} entities so the full plot area is used.</p>')
        html.append('<div class="charts-row">')
        html.append('<div class="chart-wrap">')
        html.append(make_scatter_svg(
            f"Model excl. fetus+god", "F1", "F2",
            excl_keys, excl_f1, excl_f2))
        html.append('</div><div class="chart-wrap">')
        html.append(make_scatter_svg(
            "Human excl. fetus+god", "Experience", "Agency",
            excl_keys, excl_h_exp, excl_h_ag,
            x_label_color="#0d9488", y_label_color="#d97706"))
        html.append('</div></div>')

        html.append(f'<h3>5d. Excluding Fetus + God, Axis-Aligned (F2&rarr;, F1&uarr;)</h3>')
        html.append('<div class="charts-row">')
        html.append('<div class="chart-wrap">')
        html.append(make_scatter_svg(
            f"Model excl. (F2&rarr;, F1&uarr;)", "F2", "F1",
            excl_keys, excl_f2, excl_f1))
        html.append('</div><div class="chart-wrap">')
        html.append(make_scatter_svg(
            "Human excl. (Exp&rarr;, Ag&uarr;)", "Experience", "Agency",
            excl_keys, excl_h_exp, excl_h_ag,
            x_label_color="#0d9488", y_label_color="#d97706"))
        html.append('</div></div>')

    # ── Excl PCA scatter plots (fetus+god) ──
    if excl_pca_data is not None:
        excl_f1 = excl_scores_01[:, 0] if excl_scores_01.shape[1] >= 1 else np.zeros(len(excl_entity_keys))
        excl_f2 = excl_scores_01[:, 1] if excl_scores_01.shape[1] >= 2 else np.zeros(len(excl_entity_keys))
        excl_h_exp2 = np.array([GRAY_ET_AL_SCORES.get(k, (0,0))[0] for k in excl_entity_keys])
        excl_h_ag2 = np.array([GRAY_ET_AL_SCORES.get(k, (0,0))[1] for k in excl_entity_keys])
        # Rescale human to [0,1] within subset
        excl_h_exp_r = _rescale_01(excl_h_exp2)
        excl_h_ag_r = _rescale_01(excl_h_ag2)

        html.append(f'<h3>5e. Excl PCA (re-run): Model F1&rarr;, F2&uarr;</h3>')
        html.append(f'<p class="small">Factor scores from PCA re-run without fetus+god. '
                     f'Human scores rescaled to [0,1] within remaining {len(excl_entity_keys)} entities.</p>')
        html.append('<div class="charts-row">')
        html.append('<div class="chart-wrap">')
        html.append(make_scatter_svg(
            f"Model (re-run PCA)", "F1", "F2",
            excl_entity_keys, excl_f1, excl_f2))
        html.append('</div><div class="chart-wrap">')
        html.append(make_scatter_svg(
            "Human excl. fetus+god", "Experience", "Agency",
            excl_entity_keys, excl_h_exp_r, excl_h_ag_r,
            x_label_color="#0d9488", y_label_color="#d97706"))
        html.append('</div></div>')

        html.append(f'<h3>5f. Excl PCA Axis-Aligned (F2&rarr;, F1&uarr;)</h3>')
        html.append('<div class="charts-row">')
        html.append('<div class="chart-wrap">')
        html.append(make_scatter_svg(
            f"Model (F2&rarr;, F1&uarr;)", "F2", "F1",
            excl_entity_keys, excl_f2, excl_f1))
        html.append('</div><div class="chart-wrap">')
        html.append(make_scatter_svg(
            "Human (Exp&rarr;, Ag&uarr;)", "Experience", "Agency",
            excl_entity_keys, excl_h_exp_r, excl_h_ag_r,
            x_label_color="#0d9488", y_label_color="#d97706"))
        html.append('</div></div>')

    # ── Excl PCA scatter plots (fetus+god+dead_woman) ──
    if excl_dead_pca_data is not None:
        ed_f1 = excl_dead_scores_01[:, 0] if excl_dead_scores_01.shape[1] >= 1 else np.zeros(len(excl_dead_entity_keys))
        ed_f2 = excl_dead_scores_01[:, 1] if excl_dead_scores_01.shape[1] >= 2 else np.zeros(len(excl_dead_entity_keys))
        ed_h_exp = np.array([GRAY_ET_AL_SCORES.get(k, (0,0))[0] for k in excl_dead_entity_keys])
        ed_h_ag = np.array([GRAY_ET_AL_SCORES.get(k, (0,0))[1] for k in excl_dead_entity_keys])
        ed_h_exp_r = _rescale_01(ed_h_exp)
        ed_h_ag_r = _rescale_01(ed_h_ag)

        html.append(f'<h3>5g. Excl Fetus+God+Dead (re-run PCA): F1&rarr;, F2&uarr;</h3>')
        html.append(f'<p class="small">Factor scores from PCA re-run without fetus, god, dead_woman. '
                     f'Human scores rescaled to [0,1] within remaining {len(excl_dead_entity_keys)} entities.</p>')
        html.append('<div class="charts-row">')
        html.append('<div class="chart-wrap">')
        html.append(make_scatter_svg(
            f"Model (re-run PCA)", "F1", "F2",
            excl_dead_entity_keys, ed_f1, ed_f2))
        html.append('</div><div class="chart-wrap">')
        html.append(make_scatter_svg(
            "Human excl. fetus+god+dead", "Experience", "Agency",
            excl_dead_entity_keys, ed_h_exp_r, ed_h_ag_r,
            x_label_color="#0d9488", y_label_color="#d97706"))
        html.append('</div></div>')

        html.append(f'<h3>5h. Excl Fetus+God+Dead, Axis-Aligned (F2&rarr;, F1&uarr;)</h3>')
        html.append('<div class="charts-row">')
        html.append('<div class="chart-wrap">')
        html.append(make_scatter_svg(
            f"Model (F2&rarr;, F1&uarr;)", "F2", "F1",
            excl_dead_entity_keys, ed_f2, ed_f1))
        html.append('</div><div class="chart-wrap">')
        html.append(make_scatter_svg(
            "Human (Exp&rarr;, Ag&uarr;)", "Experience", "Agency",
            excl_dead_entity_keys, ed_h_exp_r, ed_h_ag_r,
            x_label_color="#0d9488", y_label_color="#d97706"))
        html.append('</div></div>')

    # ── Consistency ──
    if consistency and method == "pairwise":
        html.append('<h2>6. Order Consistency</h2><div class="card">')
        html.append(f'<p>Pairs with both orders: <strong>{consistency.get("n_pairs_both", "?")}</strong>. '
                    f'Consistent: <strong>{consistency.get("n_consistent", "?")}</strong> '
                    f'({consistency.get("pct_consistent", 0):.1f}%). '
                    f'Mean deviation: <strong>{consistency.get("mean_deviation", 0):.3f}</strong>.</p>')
        html.append('</div>')

    # ── Footer ──
    html.append(f"""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #e2e8f0;text-align:center">
<p class="small">Experiment 4 &mdash; {config.MODEL_LABEL} &mdash; Graziano Lab, Princeton University</p>
</div>
</body></html>""")

    out_path = os.path.join(rdir, f"{method}_{condition}.html")
    with open(out_path, "w") as f:
        f.write('\n'.join(html))
    print(f"  Wrote: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate per-condition Gray replication reports")
    add_model_argument(parser)
    args = parser.parse_args()

    for condition in ["with_self", "without_self"]:
        for method in ["pairwise", "individual"]:
            print(f"\n{method} / {condition}:")
            generate_report(args.model, method, condition)


if __name__ == "__main__":
    main()
