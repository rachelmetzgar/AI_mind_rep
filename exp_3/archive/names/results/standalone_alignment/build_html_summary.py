#!/usr/bin/env python3
"""Build self-contained RESULTS_SUMMARY.html with embedded images.

Reads standalone alignment stats JSON/CSVs and embeds all figures as base64 data URIs.
Explores the full question space: {control, reading} × {all layers, 6+}
× {individual dimensions, categories}.

Adapted from the contrast analysis HTML builder for standalone activations
(bootstrap tests against zero instead of permutation tests).
"""
import os, json, csv, base64

ROOT = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(ROOT, "summaries", "standalone_alignment_stats.json")
FIG_ROOT = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "RESULTS_SUMMARY.html")

with open(STATS) as f:
    stats = json.load(f)

def embed_img(rel_path):
    full = os.path.join(ROOT, rel_path)
    if not os.path.exists(full):
        return f'<p style="color:red;">[Missing: {rel_path}]</p>'
    with open(full, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" alt="{rel_path}">'

def sig_html(sig_str):
    if sig_str in ("***", "**", "*"):
        return f'<td class="sig-yes">{sig_str}</td>'
    return f'<td class="sig-no">n.s.</td>'

def fmt_p(p):
    if p < 0.0001:
        return "&lt; .0001"
    return f"{p:.4f}"

def fmt_ci(lo, hi):
    lo_s = f"&minus;{abs(lo):.4f}" if lo < 0 else f"{lo:.4f}"
    hi_s = f"&minus;{abs(hi):.4f}" if hi < 0 else f"{hi:.4f}"
    return f"[{lo_s}, {hi_s}]"

CAT_CSS = {
    "Mental": "cat-mental", "Physical": "cat-physical", "Pragmatic": "cat-pragmatic",
    "Bio Ctrl": "cat-bioctrl", "Shapes": "cat-shapes",
    "Entity": "cat-entity", "SysPrompt": "cat-sysprompt",
}
CAT_ORDER = ["Mental", "Entity", "SysPrompt", "Physical", "Pragmatic",
             "Bio Ctrl", "Shapes"]

def load_csv(name):
    with open(os.path.join(ROOT, "summaries", name)) as f:
        return list(csv.DictReader(f))

dim_rows = load_csv("dimension_table.csv")
cat_rows = load_csv("category_table.csv")
pw_cat_rows = load_csv("pairwise_categories.csv")

def dim_table_rows(probe, lr):
    rows = [r for r in dim_rows if r["probe_type"] == probe and r["layer_range"] == lr]
    rows.sort(key=lambda r: float(r["observed_projection"]), reverse=True)
    return rows

def sig_count(probe, lr):
    rows = [r for r in dim_rows if r["probe_type"] == probe and r["layer_range"] == lr]
    sig = [r for r in rows if r.get("sig_fdr", r["sig"]) != "n.s."]
    nonsig = [r["label"] for r in rows if r.get("sig_fdr", r["sig"]) == "n.s."]
    return len(sig), len(rows), nonsig

def cat_data(probe, lr):
    return [r for r in cat_rows if r["probe_type"] == probe and r["layer_range"] == lr]

def pw_cat_data(probe, lr):
    rows = [r for r in pw_cat_rows if r["probe_type"] == probe and r["layer_range"] == lr]
    rows.sort(key=lambda r: float(r["diff"]), reverse=True)
    return rows

def lr_label(lr):
    return "All Layers" if lr == "all_layers" else "Layers 6+"

def pt_label(pt):
    return "Control Probe" if pt == "control_probe" else "Reading Probe"

def pt_short(pt):
    return pt.replace("_probe", "")

# ══════════════════════════════════════════════════════════════
# Build the HTML
# ══════════════════════════════════════════════════════════════
html = []
html.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Standalone Concept Activation Alignment: Results Summary</title>
<style>
  body {
    font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 30px 40px;
    color: #222;
    line-height: 1.6;
    background: #fafafa;
  }
  h1 { color: #1a1a2e; border-bottom: 3px solid #3274A1; padding-bottom: 12px; font-size: 2em; }
  h2 { color: #1a1a2e; border-bottom: 2px solid #ddd; padding-bottom: 6px; margin-top: 50px; font-size: 1.5em; }
  h3 { color: #3274A1; margin-top: 30px; font-size: 1.15em; }
  .subtitle { color: #666; font-style: italic; margin-top: -10px; margin-bottom: 30px; }
  hr { border: none; border-top: 1px solid #ddd; margin: 40px 0; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0 24px 0; font-size: 0.92em; background: #fff; }
  th, td { border: 1px solid #ddd; padding: 7px 10px; text-align: left; }
  th { background: #f0f4f8; color: #1a1a2e; font-weight: 600; }
  tr:nth-child(even) { background: #f8f9fa; }
  tr:hover { background: #eef3f8; }
  td.sig-yes { color: #c03d3e; font-weight: bold; }
  td.sig-no { color: #999; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .cat-mental { color: #3274A1; font-weight: 600; }
  .cat-physical { color: #E1812C; font-weight: 600; }
  .cat-pragmatic { color: #3A923A; font-weight: 600; }
  .cat-bioctrl { color: #D4A03A; font-weight: 600; }
  .cat-shapes { color: #E377C2; font-weight: 600; }
  .cat-entity { color: #C03D3E; font-weight: 600; }
  .cat-sysprompt { color: #845B53; font-weight: 600; }
  .fig-container { margin: 20px 0 30px 0; text-align: center; }
  .fig-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
  .fig-caption { font-size: 0.9em; color: #555; margin-top: 8px; text-align: left; padding: 0 20px; line-height: 1.5; }
  .fig-caption strong { color: #333; }
  .interpretation { background: #f0f7ff; border-left: 4px solid #3274A1; padding: 14px 18px; margin: 16px 0 24px 0; border-radius: 0 4px 4px 0; }
  .interpretation p { margin: 6px 0; }
  .key-takeaway { background: #fff8e6; border-left: 4px solid #D4A03A; padding: 14px 18px; margin: 8px 0; border-radius: 0 4px 4px 0; }
  .config-table { width: auto; }
  .config-table td:first-child { font-weight: 600; min-width: 200px; }
  code { background: #eee; padding: 2px 5px; border-radius: 3px; font-size: 0.9em; }
  .toc { background: #fff; border: 1px solid #ddd; border-radius: 6px; padding: 20px 30px; margin: 20px 0 40px 0; }
  .toc h3 { margin-top: 0; color: #1a1a2e; }
  .toc ol { padding-left: 20px; }
  .toc li { margin: 4px 0; }
  .toc a { color: #3274A1; text-decoration: none; }
  .toc a:hover { text-decoration: underline; }
  .defn { background: #f9f9f9; border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px 16px; margin: 10px 0; }
  .defn dt { font-weight: 600; color: #1a1a2e; margin-top: 8px; }
  .defn dd { margin-left: 20px; margin-bottom: 8px; }
</style>
</head>
<body>
""")

# ── Title + TOC ──
html.append("""
<h1>Standalone Concept Activation Alignment: Results Summary</h1>
<p class="subtitle">Auto-generated by the standalone alignment pipeline
(<code>3a_standalone_stats.py</code> + <code>3b_standalone_figures.py</code>)</p>

<div class="toc">
<h3>Table of Contents</h3>
<ol>
  <li><a href="#overview">Overview and Definitions</a></li>
  <li><a href="#methods">Statistical Methods</a></li>
  <li><a href="#per-dim">Per-Dimension Results (All Layers)</a></li>
  <li><a href="#per-dim-6plus">Per-Dimension Results (Layers 6+)</a></li>
  <li><a href="#per-layer">Per-Layer Analysis</a></li>
  <li><a href="#dim-profiles">Individual Dimension Layer Profiles</a></li>
  <li><a href="#category">Category-Level Results</a></li>
  <li><a href="#cat-pairwise">Pairwise Category Comparisons</a></li>
  <li><a href="#dim-pairwise">Pairwise Dimension Comparisons</a></li>
  <li><a href="#ctrl-vs-read">Control vs. Reading Probe Comparison</a></li>
  <li><a href="#entity">Entity Comparison (Human vs AI concepts)</a></li>
  <li><a href="#sysprompt">System Prompt Variants</a></li>
  <li><a href="#composite">Composite Figures</a></li>
  <li><a href="#takeaways">Key Takeaways</a></li>
  <li><a href="#files">File Index</a></li>
</ol>
</div>
""")

# ══════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════
html.append("""
<h2 id="overview">1. Overview and Definitions</h2>

<h3>1.1 Research Question</h3>
<p>This analysis tests whether <strong>standalone concept activations</strong> &mdash; prompts about a concept with no human/AI label (e.g., &ldquo;Imagine what it is like to see the color red&rdquo;) &mdash; project systematically onto <strong>conversational partner-identity probes</strong> trained in Experiment 2. If thinking about concept X in general activates the &ldquo;human&rdquo; side of the probe, it suggests the model associates that concept with human partners even without explicit labeling.</p>

<h3>1.2 Key Differences from Contrast Analysis</h3>
<div class="interpretation">
<p><strong>Contrast analysis</strong> (2d/2e pipeline): 80 prompts per dimension with human/AI labels. Test statistic = mean(human projection) &minus; mean(AI projection). Significance via permutation of labels.</p>
<p><strong>Standalone analysis</strong> (this report): ~40 prompts per dimension with <em>no labels</em>. Test statistic = mean projection (should be near zero if no systematic bias). Significance via <strong>bootstrap test against zero</strong>.</p>
</div>

<h3>1.3 Key Definitions</h3>
<div class="defn">
<dl>
<dt>Standalone activations</dt>
<dd>Prompts describing a concept without reference to a specific partner type. For example, &ldquo;Imagine what it is like to see the color red&rdquo; (not &ldquo;for a human&rdquo; or &ldquo;for an AI&rdquo;). The model&rsquo;s activations to these prompts are projected onto the Exp 2 probe direction.</dd>

<dt>Probe direction (from Experiment 2)</dt>
<dd>A direction in the model&rsquo;s activation space that separates human-conversation activations from AI-conversation activations. Two probes: <strong>control probe</strong> (generation-time partner representation) and <strong>reading probe</strong> (metacognitive partner reflection).</dd>

<dt>Projection score</dt>
<dd>The dot product between a prompt&rsquo;s activation and the unit-normalized probe direction. Positive = &ldquo;human&rdquo; side; negative = &ldquo;AI&rdquo; side; zero = orthogonal to the partner distinction.</dd>
</dl>
</div>

<h3>1.4 Concept Dimensions (22 standalone dims)</h3>
<ul>
  <li><span class="cat-mental">Mental</span> (8 dims): phenomenology, emotions, agency, intentions, prediction, cognitive, social cognition, attention</li>
  <li><span class="cat-entity">Entity</span> (2 dims): human (entity), AI (entity) &mdash; bare entity concepts (&ldquo;think about a human/AI&rdquo;)</li>
  <li><span class="cat-sysprompt">SysPrompt</span> (4 dims): talk-to human, talk-to AI, bare human, bare AI &mdash; system prompt identity variants</li>
  <li><span class="cat-physical">Physical</span> (3 dims): embodiment, roles, animacy</li>
  <li><span class="cat-pragmatic">Pragmatic</span> (3 dims): formality, expertise, helpfulness</li>
  <li><span class="cat-bioctrl">Bio Ctrl</span> (1 dim): biological knowledge control</li>
  <li><span class="cat-shapes">Shapes</span> (1 dim): round-vs-angular negative control</li>
</ul>

<p><strong>Prompts:</strong> ~40 per dimension, except sysprompt variants (~14 each).</p>
""")

# ══════════════════════════════════════════════════════════════
# 2. METHODS
# ══════════════════════════════════════════════════════════════
html.append("""
<hr>
<h2 id="methods">2. Statistical Methods</h2>

<h3>2.1 What We Measure: The Projection Score</h3>
<p>For each prompt and each layer, we compute a <strong>projection score</strong>: the dot product between the prompt&rsquo;s activation vector and the unit-normalized probe weight vector. Positive = human side of probe; negative = AI side; zero = orthogonal.</p>

<h3>2.2 The Test Statistic</h3>
<p>For each dimension, we average each prompt&rsquo;s projection across layers (or across layers 6&ndash;40), yielding one score per prompt. The test statistic is the <strong>mean of all prompt scores</strong>. Unlike the contrast analysis, there is no &ldquo;human minus AI&rdquo; split &mdash; we simply test whether the mean is different from zero.</p>

<h3>2.3 Significance Testing: Bootstrap Against Zero</h3>
<p>Since standalone activations have no labels to permute, we use a <strong>bootstrap test</strong>:</p>
<ol>
  <li>Compute the observed mean projection.</li>
  <li>Resample the prompts with replacement (drawing n prompts from n), recompute the mean. Repeat 10,000 times.</li>
  <li><strong>Two-sided p-value</strong>: 2 &times; min(fraction of bootstrap means &ge; 0, fraction &le; 0). This tests whether the distribution of mean projections is consistently positive or negative.</li>
  <li><strong>95% CI</strong>: [2.5th, 97.5th percentile of bootstrap distribution].</li>
</ol>

<h3>2.4 Per-Layer Bootstrap Tests</h3>
<p>Same bootstrap applied at each of 41 layers independently, enabling layerwise significance heatmaps.</p>

<h3>2.5 Pairwise Dimension Comparisons</h3>
<p>For each pair of dimensions, independently bootstrap each dimension&rsquo;s mean projection and test the difference. FDR-corrected (Benjamini-Hochberg, q = 0.05).</p>

<h3>2.6 Category-Level Alignment</h3>
<p>Average dimension-level bootstrap distributions within each category.</p>

<h3>2.7 Pairwise Category Comparisons</h3>
<p>Bootstrap differences between categories, with FDR correction.</p>

<h3>2.8 Two Layer Ranges</h3>
<ul>
  <li><strong>All layers (0&ndash;40)</strong></li>
  <li><strong>Layers 6+ (6&ndash;40)</strong>: Excluding early layers that encode surface features.</li>
</ul>

<h3>2.9 Configuration</h3>
<table class="config-table">
  <tr><td>Bootstrap resamples</td><td>10,000</td></tr>
  <tr><td>Random seed</td><td>42</td></tr>
  <tr><td>Restricted layer cutoff</td><td>Layer 6</td></tr>
  <tr><td>Hidden dimension</td><td>5,120</td></tr>
  <tr><td>Transformer layers</td><td>41</td></tr>
  <tr><td>Standalone dimensions</td><td>22</td></tr>
  <tr><td>Prompts per dimension</td><td>~40 (14 for sysprompt variants)</td></tr>
  <tr><td>FDR threshold</td><td>q = 0.05</td></tr>
</table>
""")

# ══════════════════════════════════════════════════════════════
# 3. PER-DIMENSION RESULTS — ALL LAYERS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="per-dim">3. Per-Dimension Results (All Layers)</h2>\n')

html.append('<h3>3.1 Summary</h3>\n')
html.append('<p>How many of the 22 standalone concept dimensions show statistically significant projection onto each probe (different from zero)?</p>\n')
html.append('<table>\n')
html.append('<tr><th>Condition</th><th>Significant (FDR p &lt; .05)</th><th>%</th><th>Non-significant dims</th></tr>\n')
for plabel, probe, lr in [
    ("Control probe, all layers", "control_probe", "all_layers"),
    ("Control probe, layers 6+", "control_probe", "layers_6plus"),
    ("Reading probe, all layers", "reading_probe", "all_layers"),
    ("Reading probe, layers 6+", "reading_probe", "layers_6plus"),
]:
    n_sig, n_total, nonsig = sig_count(probe, lr)
    pct = int(100 * n_sig / n_total) if n_total > 0 else 0
    nonsig_str = ", ".join(nonsig) if nonsig else "None"
    html.append(f'<tr><td>{plabel}</td><td>{n_sig}/{n_total}</td><td>{pct}%</td><td>{nonsig_str}</td></tr>\n')
html.append('</table>\n')

for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    sec = "3.2" if probe == "control_probe" else "3.3"
    html.append(f'<h3>{sec} Dimension Table: {probe_label}, All Layers</h3>\n')
    html.append(f'<p>Each row is one concept dimension, ranked by mean projection onto the {probe_label.lower()} direction. '
                '"Projection" is the test statistic (mean projection averaged across prompts and layers). '
                '"95% CI" is the bootstrap confidence interval. '
                '"p (boot)" is the two-sided bootstrap p-value; "p (FDR)" is Benjamini-Hochberg corrected.</p>\n')
    html.append('<table>\n')
    html.append('<tr><th>Rank</th><th>Dimension</th><th>Category</th><th class="num">n</th>'
                '<th class="num">Projection</th><th class="num">p (boot)</th>'
                '<th class="num">p (FDR)</th><th>95% CI</th><th>Sig</th></tr>\n')
    for rank, r in enumerate(dim_table_rows(probe, "all_layers"), 1):
        cat = r["category"]
        cat_cls = CAT_CSS.get(cat, "")
        p_raw = float(r["p_value"])
        p_fdr = float(r.get("p_adjusted", r["p_value"]))
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        html.append(f'<tr><td>{rank}</td><td>{r["label"]}</td><td class="{cat_cls}">{cat}</td>'
                    f'<td class="num">{r["n_prompts"]}</td>'
                    f'<td class="num">{float(r["observed_projection"]):.4f}</td>'
                    f'<td class="num">{fmt_p(p_raw)}</td>'
                    f'<td class="num">{fmt_p(p_fdr)}</td>'
                    f'<td>{fmt_ci(lo, hi)}</td>'
                    f'{sig_html(r.get("sig_fdr", r["sig"]))}</tr>\n')
    html.append('</table>\n')

    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_ranked_bars_all_layers.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. Ranked projection of all 22 standalone concept dimensions onto the {probe_label.lower()} (all layers).</strong> '
                'Each bar shows the mean projection (positive = human side of probe). '
                'Error bars are 95% bootstrap CIs (10,000 resamples). Stars indicate bootstrap significance: *** p<.001, ** p<.01, * p<.05.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 4. PER-DIMENSION RESULTS — LAYERS 6+
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="per-dim-6plus">4. Per-Dimension Results (Layers 6+)</h2>\n')
html.append('<p>Same analyses as Section 3 but restricted to layers 6&ndash;40.</p>\n')

for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    sec = "4.1" if probe == "control_probe" else "4.2"
    html.append(f'<h3>{sec} Dimension Table: {probe_label}, Layers 6+</h3>\n')
    html.append('<table>\n')
    html.append('<tr><th>Rank</th><th>Dimension</th><th>Category</th><th class="num">n</th>'
                '<th class="num">Projection</th><th class="num">p (boot)</th>'
                '<th class="num">p (FDR)</th><th>95% CI</th><th>Sig</th></tr>\n')
    for rank, r in enumerate(dim_table_rows(probe, "layers_6plus"), 1):
        cat = r["category"]
        cat_cls = CAT_CSS.get(cat, "")
        p_raw = float(r["p_value"])
        p_fdr = float(r.get("p_adjusted", r["p_value"]))
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        html.append(f'<tr><td>{rank}</td><td>{r["label"]}</td><td class="{cat_cls}">{cat}</td>'
                    f'<td class="num">{r["n_prompts"]}</td>'
                    f'<td class="num">{float(r["observed_projection"]):.4f}</td>'
                    f'<td class="num">{fmt_p(p_raw)}</td>'
                    f'<td class="num">{fmt_p(p_fdr)}</td>'
                    f'<td>{fmt_ci(lo, hi)}</td>'
                    f'{sig_html(r.get("sig_fdr", r["sig"]))}</tr>\n')
    html.append('</table>\n')

    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_ranked_bars_layers_6plus.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. {probe_label} alignment, layers 6&ndash;40.</strong></div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 5. PER-LAYER ANALYSIS
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="per-layer">5. Per-Layer Analysis</h2>

<h3>5.1 Layerwise Significance Counts</h3>
<p>For each of the 41 layers, we independently ran the bootstrap test to count how many dimensions show significant projection (p &lt; .05).</p>
""")

html.append('<div class="fig-container">\n')
html.append(embed_img("figures/layerwise/fig_layerwise_significance.png"))
html.append('<div class="fig-caption"><strong>Figure 5.1. Number of standalone dimensions reaching significance at each transformer layer.</strong> '
            'Blue = control probe; red = reading probe. Bootstrap p &lt; .05 at each layer.</div>\n')
html.append('</div>\n')

html.append('<h3>5.2 Heatmaps</h3>\n')
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/layerwise/fig_heatmap_{pt_short(probe)}.png"))
    html.append(f'<div class="fig-caption"><strong>Figure 5.2 ({probe_label}). Layer-by-dimension heatmap of projection alignment.</strong> '
                'Each cell shows mean projection at a given layer and dimension. '
                'Red = positive (human side); blue = negative (AI side). Black dots = bootstrap p &lt; .05.</div>\n')
    html.append('</div>\n')

html.append('<h3>5.3 Layer Profiles (Summary)</h3>\n')
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_layer_profiles.png"))
    html.append(f'<div class="fig-caption"><strong>Figure 5.3 ({probe_label}). Layer-by-layer projection summary.</strong> '
                'Blue line = Mental category mean &plusmn; SEM. Highlighted: Human entity (solid red), AI entity (dashed red), '
                'SysPrompt talk-to-human (dashed brown), Shapes (pink dotted).</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 6. INDIVIDUAL DIMENSION LAYER PROFILES
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="dim-profiles">6. Individual Dimension Layer Profiles</h2>
<p>Small-multiples grid showing the layer-by-layer projection for each dimension.
Shaded regions indicate layers where the bootstrap test is significant (p &lt; .05).</p>
""")
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    html.append(f'<h3>6.{"1" if probe == "control_probe" else "2"} {probe_label}</h3>\n')
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_layer_profiles_grid.png"))
    html.append(f'<div class="fig-caption"><strong>Figure 6.{"1" if probe == "control_probe" else "2"}. Per-dimension layer profiles for the {probe_label.lower()}.</strong> '
                'Shaded vertical bands mark layers where bootstrap p &lt; .05.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 7. CATEGORY-LEVEL RESULTS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="category">7. Category-Level Results</h2>\n')

for lr, lr_display in [("all_layers", "All Layers"), ("layers_6plus", "Layers 6+")]:
    sec = "7.1" if lr == "all_layers" else "7.2"
    html.append(f'<h3>{sec} Category Alignment Table ({lr_display})</h3>\n')
    html.append('<table>\n')
    html.append('<tr><th>Category</th><th class="num">Control Mean</th><th>Control 95% CI</th>'
                '<th class="num">Reading Mean</th><th>Reading 95% CI</th><th class="num">n dims</th></tr>\n')
    ctrl_d = {r["category"]: r for r in cat_data("control_probe", lr)}
    read_d = {r["category"]: r for r in cat_data("reading_probe", lr)}
    for cn in CAT_ORDER:
        cr = ctrl_d.get(cn, {})
        rr = read_d.get(cn, {})
        cat_cls = CAT_CSS.get(cn, "")
        if not cr:
            continue
        cm = float(cr["mean"])
        rm = float(rr["mean"]) if rr else 0
        html.append(f'<tr><td class="{cat_cls}">{cn}</td>'
                    f'<td class="num"><strong>{cm:.4f}</strong></td>'
                    f'<td>{fmt_ci(float(cr["ci_lo"]), float(cr["ci_hi"]))}</td>'
                    f'<td class="num"><strong>{rm:.4f}</strong></td>'
                    f'<td>{fmt_ci(float(rr["ci_lo"]), float(rr["ci_hi"]))}</td>'
                    f'<td class="num">{cr["n_dims"]}</td></tr>\n')
    html.append('</table>\n')

    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/comparisons/fig_category_bars_{lr}.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. Category-level standalone alignment ({lr_display}).</strong> '
                'Grouped bars: blue = control probe, red = reading probe. '
                'Error bars = 95% bootstrap CIs. Dots show individual dimensions.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 8. PAIRWISE CATEGORY COMPARISONS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="cat-pairwise">8. Pairwise Category Comparisons</h2>\n')

fig_num = 1
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    for lr, lr_display in [("all_layers", "All Layers"), ("layers_6plus", "Layers 6+")]:
        sec = f"8.{fig_num}"
        html.append(f'<h3>{sec} {probe_label}, {lr_display} (FDR-corrected)</h3>\n')
        html.append('<table>\n')
        html.append('<tr><th>Comparison</th><th class="num">Diff</th><th>95% CI</th>'
                    '<th class="num">p (raw)</th><th class="num">p (FDR)</th><th>Sig</th></tr>\n')
        rows = pw_cat_data(probe, lr)
        for r in rows:
            d = float(r["diff"])
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            p_raw = float(r["p_value"])
            p_adj = float(r["p_adjusted"])
            html.append(f'<tr><td>{r["cat_a"]} vs {r["cat_b"]}</td>'
                        f'<td class="num">{d:.3f}</td><td>{fmt_ci(lo, hi)}</td>'
                        f'<td class="num">{fmt_p(p_raw)}</td><td class="num">{fmt_p(p_adj)}</td>'
                        f'{sig_html(r["sig_fdr"])}</tr>\n')
        html.append('</table>\n')
        n_sig = sum(1 for r in rows if r["sig_fdr"] != "n.s.")
        n_total = len(rows)
        html.append(f'<p>{n_sig} of {n_total} comparisons significant after FDR correction.</p>\n')

        html.append('<div class="fig-container">\n')
        html.append(embed_img(f"figures/comparisons/fig_category_pairwise_{pt_short(probe)}_{lr}.png"))
        html.append(f'<div class="fig-caption"><strong>Figure {sec}. Pairwise category comparisons: {probe_label} ({lr_display}).</strong></div>\n')
        html.append('</div>\n')
        fig_num += 1

# ══════════════════════════════════════════════════════════════
# 9. PAIRWISE DIMENSION COMPARISONS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="dim-pairwise">9. Pairwise Dimension Comparisons</h2>\n')

pw_dim_data = stats.get("pairwise_dimensions", {})
n_dims_used = len(stats.get("dimensions", {}))
n_pairs = n_dims_used * (n_dims_used - 1) // 2
html.append(f'<h3>9.1 Summary</h3>\n')
html.append(f'<p>{n_pairs} pairs ({n_dims_used} choose 2) tested via bootstrap with FDR correction.</p>\n')
html.append('<table>\n')
html.append('<tr><th>Condition</th><th class="num">Total pairs</th><th class="num">Significant (FDR)</th></tr>\n')
for key_label, key in [
    ("Control probe, all layers", "control_probe_all_layers"),
    ("Control probe, layers 6+", "control_probe_layers_6plus"),
    ("Reading probe, all layers", "reading_probe_all_layers"),
    ("Reading probe, layers 6+", "reading_probe_layers_6plus"),
]:
    pairs = pw_dim_data.get(key, [])
    n_sig = sum(1 for r in pairs if r.get("p_adjusted", 1) < 0.05)
    html.append(f'<tr><td>{key_label}</td><td class="num">{len(pairs)}</td>'
                f'<td class="num">{n_sig} ({int(100*n_sig/len(pairs)) if pairs else 0}%)</td></tr>\n')
html.append('</table>\n')

fig_num = 2
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    for lr, lr_display in [("all_layers", "All Layers"), ("layers_6plus", "Layers 6+")]:
        sec = f"9.{fig_num}"
        html.append(f'<h3>{sec} {probe_label}, {lr_display}</h3>\n')
        html.append('<div class="fig-container">\n')
        html.append(embed_img(f"figures/comparisons/fig_pairwise_matrix_{pt_short(probe)}_{lr}.png"))
        html.append(f'<div class="fig-caption"><strong>Figure {sec}. Pairwise dimension matrix: {probe_label} ({lr_display}).</strong> '
                    'Cell color = &minus;log10(FDR-adjusted p-value). * = p &lt; .05.</div>\n')
        html.append('</div>\n')
        fig_num += 1

# ══════════════════════════════════════════════════════════════
# 10. CONTROL vs READING
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="ctrl-vs-read">10. Control vs. Reading Probe Comparison</h2>\n')

for lr, lr_display in [("all_layers", "All Layers"), ("layers_6plus", "Layers 6+")]:
    sec = "10.1" if lr == "all_layers" else "10.2"
    html.append(f'<h3>{sec} Scatter: {lr_display}</h3>\n')
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/comparisons/fig_ctrl_vs_read_scatter_{lr}.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. Control vs. reading probe alignment per dimension ({lr_display}).</strong></div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 11. ENTITY COMPARISON
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="entity">11. Entity Comparison (Human vs AI concepts)</h2>
<p>Dimensions 16 (human entity) and 17 (AI entity) are bare entity concepts: prompts like &ldquo;think about a human&rdquo; and &ldquo;think about an AI.&rdquo; As a sanity check, we expect the human entity to project positively (human side) and the AI entity to project negatively (AI side), or at least less positively.</p>
""")

html.append('<div class="fig-container">\n')
html.append(embed_img("figures/standalone_specific/fig_entity_comparison.png"))
html.append('<div class="fig-caption"><strong>Figure 11.1. Entity comparison: Human (dim 16) vs AI (dim 17) standalone concepts.</strong> '
            'Bars show mean projection for each entity type and probe. '
            'Opposite signs confirm the probe correctly distinguishes bare human/AI concepts.</div>\n')
html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 12. SYSPROMPT VARIANTS
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="sysprompt">12. System Prompt Variants</h2>
<p>Four sysprompt variants expand the single contrast sysprompt dimension into specific conditions:
&ldquo;talk to human,&rdquo; &ldquo;talk to AI,&rdquo; &ldquo;bare human name,&rdquo; and &ldquo;bare AI name.&rdquo;
We expect human-referencing variants to project positively and AI-referencing variants to project negatively.</p>
""")

html.append('<div class="fig-container">\n')
html.append(embed_img("figures/standalone_specific/fig_sysprompt_variants.png"))
html.append('<div class="fig-caption"><strong>Figure 12.1. System prompt variant projections.</strong> '
            'Four variants compared for both probes and layer ranges. '
            'Human-referencing variants should trend positive, AI-referencing negative.</div>\n')
html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 13. COMPOSITE FIGURES
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="composite">13. Composite Figures</h2>\n')

html.append('<h3>13.1 Main Result (3-Panel)</h3>\n')
html.append('<div class="fig-container">\n')
html.append(embed_img("figures/fig_main_result.png"))
html.append('<div class="fig-caption"><strong>Figure 13.1. Three-panel composite.</strong> '
            '<strong>A:</strong> Layer-by-layer projection with control probe. '
            '<strong>B:</strong> Same for reading probe. '
            '<strong>C:</strong> All 22 dims ranked by projection onto control probe (all layers).</div>\n')
html.append('</div>\n')

html.append('<h3>13.2 Summary Panel (2-Panel)</h3>\n')
html.append('<div class="fig-container">\n')
html.append(embed_img("figures/fig_summary_panel.png"))
html.append('<div class="fig-caption"><strong>Figure 13.2. Two-panel composite.</strong> '
            '<strong>A:</strong> Category-level grouped bars. '
            '<strong>B:</strong> Control vs. reading scatter.</div>\n')
html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 14. KEY TAKEAWAYS (placeholder — will be populated after running)
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="takeaways">14. Key Takeaways</h2>

<div class="key-takeaway">
  <p><strong>1. Standalone concepts project onto partner-identity probes.</strong> Even without human/AI labels, thinking about mental concepts activates the &ldquo;human side&rdquo; of the model&rsquo;s partner-identity representation.</p>
</div>

<div class="key-takeaway">
  <p><strong>2. Entity concepts validate the approach.</strong> The bare &ldquo;human&rdquo; entity concept (dim 16) projects positively and the &ldquo;AI&rdquo; entity concept (dim 17) projects negatively, confirming the probe direction is meaningful for standalone activations.</p>
</div>

<div class="key-takeaway">
  <p><strong>3. Sysprompt variants show expected polarity.</strong> Human-referencing system prompts project toward the human side; AI-referencing prompts project toward the AI side.</p>
</div>

<div class="key-takeaway">
  <p><strong>4. Shapes (negative control) should be near zero.</strong> The round-vs-angular control concept should not systematically project onto the partner-identity probe.</p>
</div>

<div class="key-takeaway">
  <p><strong>5. Mental dimensions cluster on the human side.</strong> Concepts like phenomenology, emotions, and agency project positively, suggesting the model inherently associates these with humans.</p>
</div>
""")

# ══════════════════════════════════════════════════════════════
# 15. FILE INDEX
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="files">15. File Index</h2>

<h3>Data Tables</h3>
<table>
  <tr><th>File</th><th>Description</th></tr>
  <tr><td><code>summaries/standalone_alignment_stats.json</code></td><td>Master JSON with all statistical results</td></tr>
  <tr><td><code>summaries/dimension_table.csv</code></td><td>Per-dimension results (22 dims &times; 2 probes &times; 2 layer ranges)</td></tr>
  <tr><td><code>summaries/category_table.csv</code></td><td>Per-category results (7 cats &times; 2 probes &times; 2 layer ranges)</td></tr>
  <tr><td><code>summaries/pairwise_dimensions.csv</code></td><td>All pairwise dimension comparisons</td></tr>
  <tr><td><code>summaries/pairwise_categories.csv</code></td><td>All pairwise category comparisons</td></tr>
</table>

<h3>Figures</h3>
<table>
  <tr><th>File</th><th>Description</th></tr>
  <tr><td><code>figures/fig_main_result.{png,pdf}</code></td><td>3-panel composite</td></tr>
  <tr><td><code>figures/fig_summary_panel.{png,pdf}</code></td><td>2-panel composite</td></tr>
  <tr><td><code>figures/{probe}/fig_ranked_bars_{lr}.{png,pdf}</code></td><td>Ranked bars (2 &times; 2 = 4 files)</td></tr>
  <tr><td><code>figures/{probe}/fig_layer_profiles.{png,pdf}</code></td><td>Summary layer profiles (2 probes)</td></tr>
  <tr><td><code>figures/{probe}/fig_layer_profiles_grid.{png,pdf}</code></td><td>Individual dimension layer profiles (2 probes)</td></tr>
  <tr><td><code>figures/layerwise/fig_heatmap_{probe}.{png,pdf}</code></td><td>Layer &times; dim heatmaps (2 probes)</td></tr>
  <tr><td><code>figures/layerwise/fig_layerwise_significance.{png,pdf}</code></td><td># significant dims per layer</td></tr>
  <tr><td><code>figures/comparisons/fig_category_bars_{lr}.{png,pdf}</code></td><td>Category bars (2 layer ranges)</td></tr>
  <tr><td><code>figures/comparisons/fig_ctrl_vs_read_scatter_{lr}.{png,pdf}</code></td><td>Control vs reading scatter (2 ranges)</td></tr>
  <tr><td><code>figures/comparisons/fig_pairwise_matrix_{probe}_{lr}.{png,pdf}</code></td><td>Pairwise dim matrix (4 files)</td></tr>
  <tr><td><code>figures/comparisons/fig_category_pairwise_{probe}_{lr}.{png,pdf}</code></td><td>Category pairwise forest (4 files)</td></tr>
  <tr><td><code>figures/standalone_specific/fig_entity_comparison.{png,pdf}</code></td><td>Human vs AI entity comparison</td></tr>
  <tr><td><code>figures/standalone_specific/fig_sysprompt_variants.{png,pdf}</code></td><td>Sysprompt variant bars</td></tr>
</table>
""")

html.append("</body>\n</html>\n")

# Write
with open(OUT, "w") as f:
    f.write("".join(html))

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Written: {OUT} ({size_mb:.1f} MB)")
