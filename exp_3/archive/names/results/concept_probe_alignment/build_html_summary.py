#!/usr/bin/env python3
"""Build self-contained RESULTS_SUMMARY.html with embedded images.

Reads stats JSON/CSVs and embeds all figures as base64 data URIs.
Explores the full question space: {control, reading} × {all layers, 6+}
× {individual dimensions, categories}.
"""
import os, json, csv, base64

ROOT = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(ROOT, "summaries", "alignment_stats.json")
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

# Category CSS classes
CAT_CSS = {
    "Mental": "cat-mental", "Physical": "cat-physical", "Pragmatic": "cat-pragmatic",
    "Human vs AI (General)": "cat-hvai-gen", "Bio Ctrl": "cat-bioctrl",
    "Shapes": "cat-shapes", "SysPrompt": "cat-sysprompt",
}
CAT_ORDER = ["Mental", "SysPrompt", "Physical", "Pragmatic",
             "Human vs AI (General)", "Bio Ctrl", "Shapes"]

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
<title>Concept-Probe Alignment: Results Summary</title>
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
  .cat-hvai-gen { color: #999; font-weight: 600; }
  .cat-bioctrl { color: #D4A03A; font-weight: 600; }
  .cat-shapes { color: #E377C2; font-weight: 600; }
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
<h1>Concept-Probe Alignment: Results Summary</h1>
<p class="subtitle">Auto-generated by the concept-probe alignment pipeline
(<code>2d_concept_probe_stats.py</code> + <code>2e_concept_probe_figures.py</code>)</p>

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
<p>This analysis tests whether <strong>concept directions</strong> extracted in Experiment 3 align with <strong>conversational partner-identity probes</strong> trained independently in Experiment 2. If the LLM's internal representation of mental concepts (e.g., phenomenology, emotions, agency) overlaps with how it tracks who it is talking to (human vs. AI), this suggests the model encodes these abstract dimensions as part of its partner model.</p>

<h3>1.2 Key Definitions</h3>
<div class="defn">
<dl>
<dt>Model</dt>
<dd>LLaMA-2-13B-Chat, a large language model (LLM) with 41 transformer layers and a hidden dimension of 5,120. Each layer transforms the input into an internal representation (a vector of 5,120 numbers). These internal representations&mdash;called <em>activations</em>&mdash;are what we analyze.</dd>

<dt>Concept direction (from Experiment 3)</dt>
<dd>For each of 18 concept dimensions (e.g., &ldquo;emotions,&rdquo; &ldquo;agency&rdquo;), we created 40 prompts that describe a conversational partner as human (e.g., &ldquo;You are talking to a human who feels joy&rdquo;) and 40 that describe an AI partner. We fed all 80 prompts through the model and recorded the activations at every layer. The <strong>concept direction</strong> is the vector pointing from the average AI-prompt activation to the average human-prompt activation. It captures, at each layer, which direction in the model&rsquo;s internal space separates &ldquo;human with this property&rdquo; from &ldquo;AI with this property.&rdquo;</dd>

<dt>Conversational partner-identity probes (from Experiment 2)</dt>
<dd>In a separate experiment, we trained simple linear classifiers (probes) to predict whether the model was talking to a human or an AI, using only the model&rsquo;s internal activations at each layer. Each probe learned a <strong>weight vector</strong>&mdash;a direction in the 5,120-dimensional space that best separates human-conversation activations from AI-conversation activations. Two types of probes were trained:
  <ul>
    <li><strong>Control probe:</strong> An active, in-context representation of partner identity that the model uses while generating the first token of its response. This captures how the model operationally encodes who it is talking to at the moment of output generation.</li>
    <li><strong>Reading probe:</strong> A metacognitive reflection of partner identity, captured when the model processes a reflective statement about its conversational partner. This captures a more deliberate, introspective representation of partner identity.</li>
  </ul>
</dd>

<dt>Alignment</dt>
<dd>We measure alignment by asking: does the concept direction (e.g., the direction separating &ldquo;human with emotions&rdquo; from &ldquo;AI with emotions&rdquo;) point in a similar direction as the probe&rsquo;s weight vector (the direction separating human-conversation from AI-conversation activations)? High alignment means the model uses overlapping internal representations for both &ldquo;this partner has emotions&rdquo; and &ldquo;this partner is human.&rdquo;</dd>
</dl>
</div>

<h3>1.3 Concept Dimensions</h3>
<p>18 concept dimensions are tested, grouped into 7 categories:</p>
<ul>
  <li><span class="cat-mental">Mental</span> (8 dims): phenomenology, emotions, agency, intentions, prediction, cognitive, social cognition, attention</li>
  <li><span class="cat-sysprompt">SysPrompt</span> (1 dim): system prompt partner identity (labeled human/AI names)</li>
  <li><span class="cat-physical">Physical</span> (3 dims): embodiment, roles, animacy</li>
  <li><span class="cat-pragmatic">Pragmatic</span> (3 dims): formality, expertise, helpfulness</li>
  <li><span class="cat-hvai-gen">Human vs AI (General)</span> (1 dim): generic human-vs-AI entity contrast without specific conceptual framing</li>
  <li><span class="cat-bioctrl">Bio Ctrl</span> (1 dim): biological knowledge control</li>
  <li><span class="cat-shapes">Shapes</span> (1 dim): round-vs-angular negative control (not human-vs-AI; included to verify the method does not produce false positives)</li>
</ul>

<p><strong>Prompts:</strong> 80 per dimension (40 human-perspective, 40 AI-perspective), except dim 18 (system prompt) which has 28 prompts (14 human names, 14 AI names).</p>
""")

# ══════════════════════════════════════════════════════════════
# 2. METHODS (unchanged)
# ══════════════════════════════════════════════════════════════
html.append("""
<hr>
<h2 id="methods">2. Statistical Methods</h2>

<h3>2.1 What We Measure: The Projection Score</h3>
<p>For each prompt and each layer, we compute a single number called a <strong>projection score</strong>. Here is exactly how it works:</p>
<ol>
  <li><strong>Start with a prompt&rsquo;s activation vector.</strong> When the model processes a prompt (e.g., &ldquo;You are talking to a human who experiences qualia&rdquo;), it produces an internal activation vector at each of its 41 layers. Each activation vector is a list of 5,120 numbers that encodes what the model &ldquo;thinks&rdquo; at that point in processing.</li>
  <li><strong>Start with the probe&rsquo;s weight vector.</strong> For the same layer, the Experiment 2 probe learned a weight vector&mdash;also 5,120 numbers&mdash;that points in the direction that best separates human-conversation activations from AI-conversation activations.</li>
  <li><strong>Normalize the probe weight vector</strong> to unit length (divide by its magnitude), so that the result depends only on direction, not scale.</li>
  <li><strong>Compute the dot product</strong> between the prompt&rsquo;s activation vector and the unit-normalized probe weight vector. This dot product is the projection score. It measures how far the prompt&rsquo;s activation lies in the &ldquo;human-vs-AI partner&rdquo; direction. A large positive value means the activation points strongly in the &ldquo;human partner&rdquo; direction of the probe; a value near zero means the activation is orthogonal (unrelated) to the probe direction.</li>
</ol>
<p>This gives us a matrix of projection scores with shape (n_prompts &times; n_layers)&mdash;for example, (80 &times; 41) for a typical dimension.</p>

<h3>2.2 The Test Statistic</h3>
<p>To summarize whether a concept dimension aligns with the probe, we collapse the projection score matrix into a single number:</p>
<ol>
  <li>For each prompt, <strong>average its projection scores across all layers</strong> (or across layers 6&ndash;40 for the restricted analysis). This gives one number per prompt.</li>
  <li>Separate prompts into their two groups: 40 human-perspective prompts and 40 AI-perspective prompts.</li>
  <li>The test statistic is: <strong>mean(human prompt scores) &minus; mean(AI prompt scores)</strong>.</li>
</ol>
<p>If the concept direction aligns with the probe direction, human prompts should project more positively onto the probe than AI prompts, producing a positive test statistic. A value near zero means no alignment.</p>

<h3>2.3 Significance Testing: Permutation Tests</h3>
<p>To determine whether the observed alignment is statistically significant (i.e., unlikely to occur by chance), we use a <strong>permutation test</strong>:</p>
<ol>
  <li>Compute the observed test statistic from the real data.</li>
  <li>Randomly shuffle the human/AI labels across all 80 prompts (so the 40/40 split is preserved but which prompts are called &ldquo;human&rdquo; is random). Recompute the test statistic with the shuffled labels.</li>
  <li>Repeat 10,000 times to build a <strong>null distribution</strong>&mdash;the range of test statistic values you&rsquo;d expect if there were no real difference between human and AI prompts.</li>
  <li>The <strong>p-value</strong> is the fraction of permuted test statistics that are &ge; the observed test statistic. A small p-value (e.g., p &lt; .05) means the real alignment is larger than almost all chance alignments.</li>
</ol>

<h3>2.4 Confidence Intervals: Bootstrap</h3>
<p>To estimate the uncertainty in our alignment measurement, we use a <strong>bootstrap</strong> procedure:</p>
<ol>
  <li>From the 40 human prompts, draw 40 prompts <em>with replacement</em> (some prompts may be drawn multiple times, others not at all). Do the same for the 40 AI prompts.</li>
  <li>Recompute the test statistic with these resampled prompts.</li>
  <li>Repeat 10,000 times.</li>
  <li>The middle 95% of the resampled statistics (the 2.5th to 97.5th percentiles) form the <strong>95% bootstrap confidence interval</strong>.</li>
</ol>

<h3>2.5 Per-Layer Permutation Tests</h3>
<p>The same permutation test is applied independently at each of the 41 layers (without averaging across layers). This reveals <em>where in the network</em> concept-probe alignment emerges.</p>

<h3>2.6 Pairwise Dimension Comparisons</h3>
<p>To test whether one concept dimension is more aligned than another (e.g., &ldquo;Is emotions more aligned than phenomenology?&rdquo;), we independently bootstrap each dimension&rsquo;s alignment and compute the difference. The p-value is the proportion of bootstrap iterations where the difference is &le; 0 (two-sided). With 18 dimensions there are 153 pairs; p-values are corrected using <strong>Benjamini-Hochberg FDR</strong> at q = 0.05.</p>

<h3>2.7 Category-Level Alignment</h3>
<p>Dimensions are grouped by theoretical category (Mental, Physical, etc.). Category alignment is the average of its dimensions&rsquo; bootstrap distributions, giving confidence intervals that properly reflect prompt-level variance.</p>

<h3>2.8 Pairwise Category Comparisons</h3>
<p>Bootstrap differences between category-level alignment distributions, with FDR correction across all 21 category pairs.</p>

<h3>2.9 Two Layer Ranges</h3>
<p>All analyses are run twice:</p>
<ul>
  <li><strong>All layers (0&ndash;40):</strong> The full network.</li>
  <li><strong>Layers 6+ (6&ndash;40):</strong> Excluding the first 6 layers, which primarily encode surface-level token features and prompt formatting rather than semantic content. This restricted analysis confirms that alignment is not driven by early-layer confounds.</li>
</ul>

<h3>2.10 Cosine Similarity (reported alongside projection)</h3>
<p>We also report <strong>cosine similarity</strong> between the concept mean-difference vector and the probe weight vector at each layer. Cosine similarity ranges from &minus;1 to +1 and measures directional agreement regardless of vector magnitude. It is reported for interpretability but not used as the test statistic because the projection-based statistic better supports prompt-level permutation and bootstrap inference.</p>

<h3>2.11 Configuration</h3>
<table class="config-table">
  <tr><td>Permutations</td><td>10,000</td></tr>
  <tr><td>Bootstrap resamples</td><td>10,000</td></tr>
  <tr><td>Random seed</td><td>42</td></tr>
  <tr><td>Restricted layer cutoff</td><td>Layer 6</td></tr>
  <tr><td>Hidden dimension</td><td>5,120</td></tr>
  <tr><td>Transformer layers</td><td>41</td></tr>
  <tr><td>Prompts per dimension</td><td>80 (40H / 40A), except SysPrompt: 28 (14H / 14A)</td></tr>
  <tr><td>FDR threshold</td><td>q = 0.05</td></tr>
</table>
""")

# ══════════════════════════════════════════════════════════════
# 3. PER-DIMENSION RESULTS — ALL LAYERS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="per-dim">3. Per-Dimension Results (All Layers)</h2>\n')

html.append('<h3>3.1 Summary</h3>\n')
html.append('<p>How many of the 18 concept dimensions show statistically significant alignment with each probe?</p>\n')
html.append('<table>\n')
html.append('<tr><th>Condition</th><th>Significant (p &lt; .05)</th><th>%</th><th>Non-significant dims</th></tr>\n')
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

# Dimension tables + ranked bars for BOTH probes × all_layers
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    sec = "3.2" if probe == "control_probe" else "3.3"
    html.append(f'<h3>{sec} Dimension Table: {probe_label}, All Layers</h3>\n')
    html.append(f'<p>Each row is one concept dimension, ranked by alignment with the {probe_label.lower()}. '
                '"Projection" is the test statistic (mean human &minus; mean AI projection score, averaged across all 41 layers). '
                '"95% CI" is the bootstrap confidence interval. '
                '"p (raw)" is the uncorrected permutation p-value; "p (FDR)" is Benjamini-Hochberg corrected across the 18 dimensions. '
                '"Sig" shows the FDR-corrected result: *** p<.001, ** p<.01, * p<.05, n.s. = not significant.</p>\n')
    html.append('<table>\n')
    html.append('<tr><th>Rank</th><th>Dimension</th><th>Category</th><th class="num">Cosine</th><th class="num">Projection</th><th class="num">p (raw)</th><th class="num">p (FDR)</th><th>95% CI</th><th>Sig</th></tr>\n')
    for rank, r in enumerate(dim_table_rows(probe, "all_layers"), 1):
        cat = r["category"]
        cat_cls = CAT_CSS.get(cat, "")
        p_raw = float(r["p_value"])
        p_fdr = float(r.get("p_adjusted", r["p_value"]))
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        html.append(f'<tr><td>{rank}</td><td>{r["label"]}</td><td class="{cat_cls}">{cat}</td>'
                    f'<td class="num">{float(r["observed_cosine"]):.4f}</td>'
                    f'<td class="num">{float(r["observed_projection"]):.4f}</td>'
                    f'<td class="num">{fmt_p(p_raw)}</td>'
                    f'<td class="num">{fmt_p(p_fdr)}</td>'
                    f'<td>{fmt_ci(lo, hi)}</td>'
                    f'{sig_html(r.get("sig_fdr", r["sig"]))}</tr>\n')
    html.append('</table>\n')

    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_ranked_bars_all_layers.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. Ranked alignment of all 18 concept dimensions with the {probe_label.lower()} (all layers).</strong> '
                'Each horizontal bar shows the observed projection score (test statistic: mean human projection minus mean AI projection, averaged across layers). '
                'Error bars are 95% bootstrap CIs (10,000 resamples). Stars indicate permutation test significance (10,000 permutations): *** p<.001, ** p<.01, * p<.05. '
                'Bar color indicates category membership (see legend). Dimensions are ranked from highest to lowest alignment.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 4. PER-DIMENSION RESULTS — LAYERS 6+
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="per-dim-6plus">4. Per-Dimension Results (Layers 6+)</h2>\n')
html.append('<p>Same analyses as Section 3 but restricted to layers 6&ndash;40, excluding early layers that primarily encode prompt formatting.</p>\n')

for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    sec = "4.1" if probe == "control_probe" else "4.2"
    html.append(f'<h3>{sec} Dimension Table: {probe_label}, Layers 6+</h3>\n')
    html.append('<table>\n')
    html.append('<tr><th>Rank</th><th>Dimension</th><th>Category</th><th class="num">Cosine</th><th class="num">Projection</th><th class="num">p (raw)</th><th class="num">p (FDR)</th><th>95% CI</th><th>Sig</th></tr>\n')
    for rank, r in enumerate(dim_table_rows(probe, "layers_6plus"), 1):
        cat = r["category"]
        cat_cls = CAT_CSS.get(cat, "")
        p_raw = float(r["p_value"])
        p_fdr = float(r.get("p_adjusted", r["p_value"]))
        lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
        html.append(f'<tr><td>{rank}</td><td>{r["label"]}</td><td class="{cat_cls}">{cat}</td>'
                    f'<td class="num">{float(r["observed_cosine"]):.4f}</td>'
                    f'<td class="num">{float(r["observed_projection"]):.4f}</td>'
                    f'<td class="num">{fmt_p(p_raw)}</td>'
                    f'<td class="num">{fmt_p(p_fdr)}</td>'
                    f'<td>{fmt_ci(lo, hi)}</td>'
                    f'{sig_html(r.get("sig_fdr", r["sig"]))}</tr>\n')
    html.append('</table>\n')

    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_ranked_bars_layers_6plus.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. {probe_label} alignment, restricted to layers 6&ndash;40.</strong> '
                'Same format as Section 3. Excluding early layers amplifies effects slightly; rankings remain stable.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 5. PER-LAYER ANALYSIS
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="per-layer">5. Per-Layer Analysis</h2>

<h3>5.1 Layerwise Significance Counts</h3>
<p>For each of the 41 layers, we independently ran the permutation test to count how many of the 18 dimensions show significant alignment (p &lt; .05) at that specific layer.</p>
""")

html.append('<div class="fig-container">\n')
html.append(embed_img("figures/layerwise/fig_layerwise_significance.png"))
html.append('<div class="fig-caption"><strong>Figure 5.1. Number of concept dimensions reaching significance at each transformer layer.</strong> '
            'Blue = control probe; red = reading probe. The gray dotted line shows n = 18 (total dims). '
            'The vertical dashed line at layer 6 marks the restricted-analysis boundary. '
            'Significance is determined by per-layer permutation tests (10,000 permutations each).</div>\n')
html.append('</div>\n')

html.append('<h3>5.2 Heatmaps</h3>\n')
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/layerwise/fig_heatmap_{pt_short(probe)}.png"))
    html.append(f'<div class="fig-caption"><strong>Figure 5.2 ({probe_label}). Layer-by-dimension heatmap of cosine alignment.</strong> '
                'Each cell shows cosine similarity between the concept direction and probe weight at a given layer (y-axis) and dimension (x-axis). '
                'Red = positive (direction agrees); blue = negative. Black dots = per-layer permutation p &lt; .05. '
                'White lines separate categories.</div>\n')
    html.append('</div>\n')

html.append('<h3>5.3 Layer Profiles (Summary)</h3>\n')
html.append('<p>These summary plots show the Mental category average (with SEM band) plus a few key comparison dimensions.</p>\n')
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_layer_profiles.png"))
    html.append(f'<div class="fig-caption"><strong>Figure 5.3 ({probe_label}). Layer-by-layer cosine alignment summary.</strong> '
                'Blue line and shaded band = Mental category mean &plusmn; SEM across 8 dimensions. '
                'Highlighted lines: SysPrompt (brown dashed), Human vs AI General (gray dashed), Shapes (pink dotted), Formality (green dotted). '
                'Thin gray lines = all other dimensions. Vertical dashed line at layer 6 marks the L6+ boundary.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 6. INDIVIDUAL DIMENSION LAYER PROFILES
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="dim-profiles">6. Individual Dimension Layer Profiles</h2>
<p>Small-multiples grid showing the layer-by-layer cosine alignment for each individual dimension.
Shaded regions indicate layers where the per-layer permutation test is significant (p &lt; .05).
All subplots share the same y-axis scale for direct comparison.</p>
""")
for probe, probe_label in [("control_probe", "Control Probe"), ("reading_probe", "Reading Probe")]:
    html.append(f'<h3>6.{"1" if probe == "control_probe" else "2"} {probe_label}</h3>\n')
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/{probe}/fig_layer_profiles_grid.png"))
    html.append(f'<div class="fig-caption"><strong>Figure 6.{"1" if probe == "control_probe" else "2"}. Per-dimension layer profiles for the {probe_label.lower()}.</strong> '
                'Each subplot shows one concept dimension&rsquo;s cosine alignment across all 41 layers. '
                'Line color indicates category. Shaded vertical bands mark layers where the per-layer permutation test is significant (p &lt; .05). '
                'Vertical dashed gray line = layer 6. Dimensions are ordered by category (Mental first, then SysPrompt, Physical, Pragmatic, controls).</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 7. CATEGORY-LEVEL RESULTS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="category">7. Category-Level Results</h2>\n')

for lr, lr_display in [("all_layers", "All Layers"), ("layers_6plus", "Layers 6+")]:
    sec = "7.1" if lr == "all_layers" else "7.2"
    html.append(f'<h3>{sec} Category Alignment Table ({lr_display})</h3>\n')
    html.append('<p>Category alignment = average of dimension-level bootstrap distributions within each category. '
                'CIs account for prompt-level variance.</p>\n')
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

    # Category bars figure
    html.append('<div class="fig-container">\n')
    html.append(embed_img(f"figures/comparisons/fig_category_bars_{lr}.png"))
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. Category-level alignment ({lr_display}).</strong> '
                'Grouped bars: blue = control probe (generation representation), red = reading probe (metacognitive reflection). '
                'Error bars = 95% bootstrap CIs from prompt-level resampling. '
                'Small dots on multi-dimension categories show individual dimension values.</div>\n')
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
        html.append('<p>Each row tests whether two categories differ in mean alignment. '
                    '&ldquo;Diff&rdquo; = bootstrap mean difference. '
                    '&ldquo;p (FDR)&rdquo; = Benjamini-Hochberg adjusted p-value.</p>\n')
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

        # Forest plot figure
        html.append('<div class="fig-container">\n')
        html.append(embed_img(f"figures/comparisons/fig_category_pairwise_{pt_short(probe)}_{lr}.png"))
        html.append(f'<div class="fig-caption"><strong>Figure {sec}. Pairwise category comparisons: {probe_label} ({lr_display}).</strong> '
                    'Each bar shows the bootstrap mean difference between two categories, with 95% CI. '
                    'Red = significant after FDR correction; gray = non-significant.</div>\n')
        html.append('</div>\n')
        fig_num += 1

# ══════════════════════════════════════════════════════════════
# 9. PAIRWISE DIMENSION COMPARISONS
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="dim-pairwise">9. Pairwise Dimension Comparisons</h2>\n')

pw_dim_data = stats.get("pairwise_dimensions", {})
html.append('<h3>9.1 Summary</h3>\n')
html.append('<p>153 pairs (18 choose 2) tested via bootstrap with FDR correction.</p>\n')
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
                    'Lower-triangle heatmap; cell color = &minus;log10(FDR-adjusted p-value). '
                    'Darker = more significant difference. * = p &lt; .05 (FDR-corrected).</div>\n')
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
    html.append(f'<div class="fig-caption"><strong>Figure {sec}. Control vs. reading probe alignment per dimension ({lr_display}).</strong> '
                'Each dot is one concept dimension. x-axis = control probe alignment (generation-time representation); '
                'y-axis = reading probe alignment (metacognitive reflection). Color = category. '
                'The diagonal line marks equal alignment. Points below the diagonal = stronger control-probe alignment.</div>\n')
    html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 11. COMPOSITE FIGURES
# ══════════════════════════════════════════════════════════════
html.append('<hr>\n<h2 id="composite">11. Composite Figures</h2>\n')

html.append('<h3>11.1 Main Result (3-Panel)</h3>\n')
html.append('<div class="fig-container">\n')
html.append(embed_img("figures/fig_main_result.png"))
html.append('<div class="fig-caption"><strong>Figure 11.1. Three-panel composite for publication.</strong> '
            '<strong>A (top-left):</strong> Layer-by-layer cosine with control probe. '
            '<strong>B (top-right):</strong> Same for reading probe (same y-axis scale). '
            '<strong>C (bottom):</strong> All 18 dims ranked by projection onto control probe (all layers), with 95% bootstrap CIs and significance stars.</div>\n')
html.append('</div>\n')

html.append('<h3>11.2 Summary Panel (2-Panel)</h3>\n')
html.append('<div class="fig-container">\n')
html.append(embed_img("figures/fig_summary_panel.png"))
html.append('<div class="fig-caption"><strong>Figure 11.2. Two-panel composite for talks.</strong> '
            '<strong>A:</strong> Category-level grouped bars (control vs. reading). '
            '<strong>B:</strong> Control vs. reading alignment scatter per dimension.</div>\n')
html.append('</div>\n')

# ══════════════════════════════════════════════════════════════
# 12. KEY TAKEAWAYS
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="takeaways">12. Key Takeaways</h2>

<div class="key-takeaway">
  <p><strong>1. Concept directions strongly align with conversational probes.</strong> 16 of 18 concept dimensions significantly align with the control probe, and 15 of 18 with the reading probe. The model's representation of &ldquo;what differs between humans and AI&rdquo; across specific conceptual dimensions overlaps substantially with its general partner-identity tracking mechanisms.</p>
</div>

<div class="key-takeaway">
  <p><strong>2. Mental dimensions show the strongest and most consistent alignment.</strong> All 8 mental dimensions are highly significant (p &lt; .0001) for both probes.</p>
</div>

<div class="key-takeaway">
  <p><strong>3. System prompt partner identity aligns as strongly as rich mental descriptions.</strong> The system prompt dimension uses only labeled names yet matches Mental-category alignment, suggesting the model recruits the same representational space for minimal identity cues as for detailed conceptual descriptions.</p>
</div>

<div class="key-takeaway">
  <p><strong>4. The negative controls validate the analysis.</strong> Shapes (round-vs-angular, unrelated to human-vs-AI) is non-significant across all conditions.</p>
</div>

<div class="key-takeaway">
  <p><strong>5. Control probe alignment exceeds reading probe alignment (~2:1 ratio).</strong> Concept directions align more with the in-context partner representation used during response generation than with the model&rsquo;s metacognitive partner reflection.</p>
</div>

<div class="key-takeaway">
  <p><strong>6. Emotions and Intentions are the most strongly aligned mental dimensions.</strong> These lead both overall rankings and pairwise comparisons, suggesting affective and intentional aspects of the human-AI distinction are most tightly coupled to partner tracking.</p>
</div>

<div class="key-takeaway">
  <p><strong>7. Formality is not aligned with partner identity.</strong> Despite being a plausible human-AI distinction, the formality concept direction is orthogonal to both probes.</p>
</div>

<div class="key-takeaway">
  <p><strong>8. The restricted-layer analysis (L6+) amplifies all effects,</strong> confirming alignment is not driven by early-layer prompt-format confounds.</p>
</div>
""")

# ══════════════════════════════════════════════════════════════
# 13. FILE INDEX
# ══════════════════════════════════════════════════════════════
html.append("""<hr>
<h2 id="files">13. File Index</h2>

<h3>Data Tables</h3>
<table>
  <tr><th>File</th><th>Description</th></tr>
  <tr><td><code>summaries/alignment_stats.json</code></td><td>Master JSON with all statistical results</td></tr>
  <tr><td><code>summaries/dimension_table.csv</code></td><td>Per-dimension results (18 dims &times; 2 probes &times; 2 layer ranges)</td></tr>
  <tr><td><code>summaries/category_table.csv</code></td><td>Per-category results (7 cats &times; 2 probes &times; 2 layer ranges)</td></tr>
  <tr><td><code>summaries/pairwise_dimensions.csv</code></td><td>All pairwise dimension comparisons (153 pairs &times; 4 conditions)</td></tr>
  <tr><td><code>summaries/pairwise_categories.csv</code></td><td>All pairwise category comparisons (21 pairs &times; 4 conditions)</td></tr>
</table>

<h3>Figures</h3>
<table>
  <tr><th>File</th><th>Description</th></tr>
  <tr><td><code>figures/fig_main_result.{png,pdf}</code></td><td>3-panel composite for paper</td></tr>
  <tr><td><code>figures/fig_summary_panel.{png,pdf}</code></td><td>2-panel composite for talks</td></tr>
  <tr><td><code>figures/{probe}/fig_ranked_bars_{lr}.{png,pdf}</code></td><td>Ranked bars (2 probes &times; 2 layer ranges = 4 files)</td></tr>
  <tr><td><code>figures/{probe}/fig_layer_profiles.{png,pdf}</code></td><td>Summary layer profiles (2 probes)</td></tr>
  <tr><td><code>figures/{probe}/fig_layer_profiles_grid.{png,pdf}</code></td><td>Individual dimension layer profiles (2 probes)</td></tr>
  <tr><td><code>figures/layerwise/fig_heatmap_{probe}.{png,pdf}</code></td><td>Layer &times; dim heatmaps (2 probes)</td></tr>
  <tr><td><code>figures/layerwise/fig_layerwise_significance.{png,pdf}</code></td><td># significant dims at each layer</td></tr>
  <tr><td><code>figures/comparisons/fig_category_bars_{lr}.{png,pdf}</code></td><td>Category bars (2 layer ranges)</td></tr>
  <tr><td><code>figures/comparisons/fig_ctrl_vs_read_scatter_{lr}.{png,pdf}</code></td><td>Control vs reading scatter (2 layer ranges)</td></tr>
  <tr><td><code>figures/comparisons/fig_pairwise_matrix_{probe}_{lr}.{png,pdf}</code></td><td>Pairwise dim matrix (2 probes &times; 2 ranges = 4)</td></tr>
  <tr><td><code>figures/comparisons/fig_category_pairwise_{probe}_{lr}.{png,pdf}</code></td><td>Category pairwise forest (2 probes &times; 2 ranges = 4)</td></tr>
</table>
""")

html.append("</body>\n</html>\n")

# Write
with open(OUT, "w") as f:
    f.write("".join(html))

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Written: {OUT} ({size_mb:.1f} MB)")
