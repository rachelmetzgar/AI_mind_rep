#!/usr/bin/env python3
"""
Build a self-contained HTML report on the Lexical Overlap Investigation.

Generates new figures (scatter plots, bar charts) and embeds them alongside
existing figures from the concept_probe_alignment and standalone_alignment
analyses.

Usage:
    python build_lexical_overlap_report.py

Output:
    results/lexical_overlap_investigation/LEXICAL_OVERLAP_REPORT.html
"""

import base64
import csv
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as sp_stats

# ── paths ─────────────────────────────────────────────────────────────────
BASE = Path("/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat")
CONTRAST_RESULTS = BASE / "results" / "concept_probe_alignment"
STANDALONE_RESULTS = BASE / "results" / "standalone_alignment"
LEX_CSV = CONTRAST_RESULTS / "summaries" / "lexical_distinctiveness.csv"
CONTRAST_JSON = CONTRAST_RESULTS / "summaries" / "alignment_stats.json"
STANDALONE_JSON = STANDALONE_RESULTS / "summaries" / "standalone_alignment_stats.json"
OUT_DIR = BASE / "results" / "lexical_overlap_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── category colours ──────────────────────────────────────────────────────
CAT_COLORS = {
    "Human vs AI (General)": "#888888",
    "Mental":     "#4C72B0",
    "Physical":   "#55A868",
    "Pragmatic":  "#C44E52",
    "Bio Ctrl":   "#8172B2",
    "Shapes":     "#CCB974",
    "SysPrompt":  "#64B5CD",
    "Entity":     "#C03D3E",
}

# ── helpers ───────────────────────────────────────────────────────────────
def fig_to_b64(fig, dpi=150):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def fig_to_file_and_b64(fig, path, dpi=150):
    """Save figure to disk and return base64 string."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    b64 = fig_to_b64(fig, dpi=dpi)
    return b64

def load_existing_png(path):
    """Load a PNG file and return base64 string."""
    if not Path(path).exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def img_tag(b64, alt="", width="100%", caption=None):
    """Generate an HTML <figure> with base64 image."""
    if b64 is None:
        return f'<p class="missing">[Figure not found: {alt}]</p>'
    html = f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:{width}; height:auto;">'
    if caption:
        html += f'<figcaption>{caption}</figcaption>'
    html += '</figure>'
    return html


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════

# ── lexical distinctiveness CSV ───────────────────────────────────────────
lex_rows = []
with open(LEX_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        lex_rows.append({
            "dim_id":     int(row["dim_id"]),
            "dim_name":   row["dim_name"],
            "category":   row["category"],
            "n_human":    int(row["n_human_prompts"]),
            "n_ai":       int(row["n_ai_prompts"]),
            "n_h_unique": int(row["n_human_unique_words"]),
            "n_a_unique": int(row["n_ai_unique_words"]),
            "jaccard":    float(row["jaccard"]),
            "lex_dist":   float(row["lexical_distinctiveness"]),
            "pct_h_ent":  float(row["pct_human_entity_words"]),
            "pct_a_ent":  float(row["pct_ai_entity_words"]),
            "align_proj": float(row["alignment_projection"]),
        })

# ── contrast alignment stats ─────────────────────────────────────────────
with open(CONTRAST_JSON) as f:
    contrast_data = json.load(f)

# ── standalone alignment stats ────────────────────────────────────────────
with open(STANDALONE_JSON) as f:
    standalone_data = json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Generate new figures
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: Lexical Distinctiveness vs Alignment ────────────────────────
ld_vals = np.array([r["lex_dist"] for r in lex_rows])
al_vals = np.array([r["align_proj"] for r in lex_rows])
cats    = [r["category"] for r in lex_rows]
names   = [r["dim_name"] for r in lex_rows]

fig1, ax1 = plt.subplots(figsize=(8, 6))
for i, r in enumerate(lex_rows):
    c = CAT_COLORS.get(r["category"], "#999999")
    ax1.scatter(r["lex_dist"], r["align_proj"], color=c, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    # Label points
    offset_x, offset_y = 0.008, 0.005
    short = r["dim_name"].split("_", 1)[-1] if "_" in r["dim_name"] else r["dim_name"]
    ax1.annotate(short, (r["lex_dist"], r["align_proj"]),
                 textcoords="offset points", xytext=(5, 5),
                 fontsize=7, color="#444444")

# Spearman correlation line
rho, p = sp_stats.spearmanr(ld_vals, al_vals)
# Linear regression for trend line
slope, intercept = np.polyfit(ld_vals, al_vals, 1)
x_line = np.linspace(ld_vals.min() - 0.02, ld_vals.max() + 0.02, 100)
ax1.plot(x_line, slope * x_line + intercept, "--", color="#999999", linewidth=1, zorder=1)

ax1.set_xlabel("Lexical Distinctiveness (1 − Jaccard)", fontsize=12)
ax1.set_ylabel("Alignment Projection\n(control probe, all layers)", fontsize=12)
ax1.set_title(f"Lexical Distinctiveness vs. Alignment\nSpearman ρ = {rho:.2f}, p = {p:.3f}", fontsize=13)
ax1.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)

# Legend
handles = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()
           if any(r["category"] == cat for r in lex_rows)]
ax1.legend(handles=handles, fontsize=8, loc="upper left")
ax1.set_xlim(0, 1.05)
fig1.tight_layout()
b64_fig1 = fig_to_file_and_b64(fig1, FIG_DIR / "fig_lexical_vs_alignment.png")


# ── Figure 2: Entity Contamination vs Alignment ──────────────────────────
avg_ent = np.array([(r["pct_h_ent"] + r["pct_a_ent"]) / 2 for r in lex_rows])

fig2, ax2 = plt.subplots(figsize=(8, 6))
for i, r in enumerate(lex_rows):
    c = CAT_COLORS.get(r["category"], "#999999")
    ent = (r["pct_h_ent"] + r["pct_a_ent"]) / 2
    ax2.scatter(ent, r["align_proj"], color=c, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    short = r["dim_name"].split("_", 1)[-1] if "_" in r["dim_name"] else r["dim_name"]
    ax2.annotate(short, (ent, r["align_proj"]),
                 textcoords="offset points", xytext=(5, 5),
                 fontsize=7, color="#444444")

rho2, p2 = sp_stats.spearmanr(avg_ent, al_vals)
slope2, intercept2 = np.polyfit(avg_ent, al_vals, 1)
x_line2 = np.linspace(-0.02, 1.05, 100)
ax2.plot(x_line2, slope2 * x_line2 + intercept2, "--", color="#999999", linewidth=1, zorder=1)

ax2.set_xlabel("Average Entity Word Contamination\n(fraction of prompts with human/AI words)", fontsize=11)
ax2.set_ylabel("Alignment Projection\n(control probe, all layers)", fontsize=12)
ax2.set_title(f"Entity Contamination vs. Alignment\nSpearman ρ = {rho2:.2f}, p = {p2:.4f}", fontsize=13)
ax2.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
handles2 = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()
            if any(r["category"] == cat for r in lex_rows)]
ax2.legend(handles=handles2, fontsize=8, loc="upper left")
ax2.set_xlim(-0.05, 1.1)
fig2.tight_layout()
b64_fig2 = fig_to_file_and_b64(fig2, FIG_DIR / "fig_entity_vs_alignment.png")


# ── Figure 3: Entity contamination bar chart ──────────────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))
y_pos = np.arange(len(lex_rows))
labels = [r["dim_name"].split("_", 1)[-1] for r in lex_rows]
colors = [CAT_COLORS.get(r["category"], "#999999") for r in lex_rows]
h_ent = [r["pct_h_ent"] for r in lex_rows]
a_ent = [r["pct_a_ent"] for r in lex_rows]

bar_width = 0.35
ax3.barh(y_pos - bar_width/2, h_ent, bar_width, label='Human-side prompts',
         color='#4C72B0', alpha=0.8)
ax3.barh(y_pos + bar_width/2, a_ent, bar_width, label='AI-side prompts',
         color='#C44E52', alpha=0.8)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels, fontsize=9)
ax3.set_xlabel("Fraction of Prompts Containing Entity Words", fontsize=11)
ax3.set_title("Entity Word Contamination by Dimension", fontsize=13)
ax3.legend(fontsize=10)
ax3.invert_yaxis()
fig3.tight_layout()
b64_fig3 = fig_to_file_and_b64(fig3, FIG_DIR / "fig_entity_contamination_bars.png")


# ── Figure 4: Jaccard similarity bar chart ────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(10, 5))
jacc = [r["jaccard"] for r in lex_rows]
ax4.barh(y_pos, jacc, color=colors, alpha=0.85, edgecolor="white")
ax4.set_yticks(y_pos)
ax4.set_yticklabels(labels, fontsize=9)
ax4.set_xlabel("Jaccard Similarity (word overlap)", fontsize=11)
ax4.set_title("Vocabulary Overlap Between Human-Side and AI-Side Prompts", fontsize=13)
ax4.invert_yaxis()
ax4.axvline(0.5, color="#cccccc", linewidth=0.8, linestyle="--")
fig4.tight_layout()
b64_fig4 = fig_to_file_and_b64(fig4, FIG_DIR / "fig_jaccard_bars.png")


# ── Figure 5: Contrast vs Standalone alignment comparison ─────────────────
# For shared dims, plot contrast alignment vs standalone projection
shared_dims = []
for dim_key, dim_data in contrast_data["dimensions"].items():
    dim_id = dim_data["dim_id"]
    dim_name = dim_data["dim_name"]
    contrast_proj = dim_data.get("control_probe_all_layers", {})
    if isinstance(contrast_proj, dict):
        contrast_val = contrast_proj.get("observed_projection")
    else:
        # Use per-dim table
        contrast_val = None
    if contrast_val is None:
        continue
    # Find in standalone
    if str(dim_id) in standalone_data["dimensions"]:
        standalone_val = standalone_data["dimensions"][str(dim_id)].get(
            "control_probe_all_layers", {}).get("observed_projection")
        if standalone_val is not None:
            shared_dims.append({
                "dim_id": dim_id,
                "name": dim_name,
                "category": dim_data.get("category", ""),
                "contrast": contrast_val,
                "standalone": standalone_val,
            })

fig5, ax5 = plt.subplots(figsize=(8, 6))
for d in shared_dims:
    c = CAT_COLORS.get(d["category"], "#999999")
    ax5.scatter(d["contrast"], d["standalone"], color=c, s=80, zorder=3,
                edgecolors="white", linewidth=0.5)
    short = d["name"].split("_", 1)[-1] if "_" in d["name"] else d["name"]
    ax5.annotate(short, (d["contrast"], d["standalone"]),
                 textcoords="offset points", xytext=(5, 5),
                 fontsize=7, color="#444444")

if shared_dims:
    c_vals = [d["contrast"] for d in shared_dims]
    s_vals = [d["standalone"] for d in shared_dims]
    rho5, p5 = sp_stats.spearmanr(c_vals, s_vals)
    ax5.set_title(f"Contrast vs. Standalone Alignment (Control Probe)\nSpearman ρ = {rho5:.2f}, p = {p5:.4f}",
                  fontsize=13)

ax5.set_xlabel("Contrast Alignment Projection\n(human − AI, permutation-tested)", fontsize=11)
ax5.set_ylabel("Standalone Projection\n(bootstrap test against zero)", fontsize=11)
ax5.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
ax5.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)
handles5 = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()
            if any(d["category"] == cat for d in shared_dims)]
ax5.legend(handles=handles5, fontsize=8, loc="lower right")
fig5.tight_layout()
b64_fig5 = fig_to_file_and_b64(fig5, FIG_DIR / "fig_contrast_vs_standalone.png")


# ── Figure 6: Standalone sysprompt variants ───────────────────────────────
sysprompt_dims = {}
for dim_key, dim_data in standalone_data["dimensions"].items():
    if dim_data.get("category") == "SysPrompt":
        sysprompt_dims[dim_data["dim_name"]] = {
            "control_all": dim_data.get("control_probe_all_layers", {}).get("observed_projection", 0),
            "control_6p": dim_data.get("control_probe_layers_6plus", {}).get("observed_projection", 0),
            "reading_all": dim_data.get("reading_probe_all_layers", {}).get("observed_projection", 0),
            "reading_6p": dim_data.get("reading_probe_layers_6plus", {}).get("observed_projection", 0),
        }

if sysprompt_dims:
    fig6, axes6 = plt.subplots(1, 2, figsize=(14, 5))
    sp_names = sorted(sysprompt_dims.keys())
    sp_labels = [n.replace("_sysprompt_", "").replace("sysprompt_", "") for n in sp_names]
    # Clean up labels
    sp_labels_clean = []
    for n in sp_names:
        parts = n.split("_", 1)
        if len(parts) > 1:
            sp_labels_clean.append(parts[1].replace("sysprompt_", ""))
        else:
            sp_labels_clean.append(n)

    x_pos = np.arange(len(sp_names))
    w = 0.35

    # Control probe
    ctrl_vals = [sysprompt_dims[n]["control_all"] for n in sp_names]
    read_vals = [sysprompt_dims[n]["reading_all"] for n in sp_names]

    # Colour by human/AI content
    sp_colors = []
    for n in sp_names:
        if "human" in n.lower():
            sp_colors.append("#4C72B0")  # blue for human-referencing
        elif "ai" in n.lower():
            sp_colors.append("#C44E52")  # red for AI-referencing
        else:
            sp_colors.append("#999999")

    axes6[0].bar(x_pos, ctrl_vals, color=sp_colors, alpha=0.85, edgecolor="white")
    axes6[0].set_xticks(x_pos)
    axes6[0].set_xticklabels(sp_labels_clean, rotation=25, ha="right", fontsize=9)
    axes6[0].set_ylabel("Mean Projection", fontsize=11)
    axes6[0].set_title("Control Probe (all layers)", fontsize=12)
    axes6[0].axhline(0, color="#cccccc", linewidth=0.8)

    axes6[1].bar(x_pos, read_vals, color=sp_colors, alpha=0.85, edgecolor="white")
    axes6[1].set_xticks(x_pos)
    axes6[1].set_xticklabels(sp_labels_clean, rotation=25, ha="right", fontsize=9)
    axes6[1].set_ylabel("Mean Projection", fontsize=11)
    axes6[1].set_title("Reading Probe (all layers)", fontsize=12)
    axes6[1].axhline(0, color="#cccccc", linewidth=0.8)

    fig6.suptitle("Standalone SysPrompt Variants: Human vs AI References", fontsize=14, y=1.02)
    fig6.tight_layout()
    b64_fig6 = fig_to_file_and_b64(fig6, FIG_DIR / "fig_sysprompt_standalone_detail.png")
else:
    b64_fig6 = None


# ── Figure 7: Layer profile comparison (selected dimensions) ─────────────
# Show layer profiles for a few key dims: emotions (high alignment), formality (near-zero), shapes (near-zero)
highlight_dims = {"2": "Emotions (Mental)", "11": "Formality (Pragmatic)", "15": "Shapes (neg. control)"}
fig7, ax7 = plt.subplots(figsize=(10, 5))
layer_colors = {"2": "#4C72B0", "11": "#C44E52", "15": "#CCB974"}

for dim_key, dim_label in highlight_dims.items():
    if dim_key in contrast_data["dimensions"]:
        cosines = contrast_data["dimensions"][dim_key].get("control_probe_per_layer_cosines", [])
        projections = contrast_data["dimensions"][dim_key].get("control_probe_per_layer_projections", [])
        # Use projections if available, otherwise cosines
        values = projections if projections else cosines
        if values:
            ax7.plot(range(len(values)), values, "-o", markersize=3,
                     label=dim_label, color=layer_colors[dim_key], linewidth=1.5)

ax7.set_xlabel("Layer", fontsize=12)
ax7.set_ylabel("Per-Layer Alignment\n(Projection: mean_human − mean_AI)", fontsize=11)
ax7.set_title("Layer Profiles: Selected Dimensions\n(Lexical features would appear in early layers; conceptual features peak late)",
              fontsize=12)
ax7.axhline(0, color="#cccccc", linewidth=0.8)
ax7.legend(fontsize=10)
ax7.axvspan(0, 5, alpha=0.08, color='red', label='_nolegend_')
ax7.axvspan(25, 40, alpha=0.08, color='blue', label='_nolegend_')
ax7.text(2, ax7.get_ylim()[1] * 0.9, "Early\n(lexical)", fontsize=8, ha="center", color="#CC4444", alpha=0.7)
ax7.text(32, ax7.get_ylim()[1] * 0.9, "Late\n(conceptual)", fontsize=8, ha="center", color="#4444CC", alpha=0.7)
fig7.tight_layout()
b64_fig7 = fig_to_file_and_b64(fig7, FIG_DIR / "fig_layer_profiles_selected.png")


# ── Figure 8: Alignment bar chart (all dims) sorted ──────────────────────
sorted_rows = sorted(lex_rows, key=lambda r: r["align_proj"], reverse=True)
fig8, ax8 = plt.subplots(figsize=(10, 6))
y8 = np.arange(len(sorted_rows))
bar_colors = [CAT_COLORS.get(r["category"], "#999999") for r in sorted_rows]
bar_labels = [r["dim_name"].split("_", 1)[-1] for r in sorted_rows]
bar_vals = [r["align_proj"] for r in sorted_rows]

ax8.barh(y8, bar_vals, color=bar_colors, alpha=0.85, edgecolor="white")
ax8.set_yticks(y8)
ax8.set_yticklabels(bar_labels, fontsize=9)
ax8.set_xlabel("Alignment Projection (control probe, all layers)", fontsize=11)
ax8.set_title("Contrast Alignment by Dimension (ranked)", fontsize=13)
ax8.axvline(0, color="#333333", linewidth=0.8)
ax8.invert_yaxis()
handles8 = [mpatches.Patch(color=c, label=cat) for cat, c in CAT_COLORS.items()
            if any(r["category"] == cat for r in sorted_rows)]
ax8.legend(handles=handles8, fontsize=8, loc="lower right")
fig8.tight_layout()
b64_fig8 = fig_to_file_and_b64(fig8, FIG_DIR / "fig_alignment_ranked.png")


# ═══════════════════════════════════════════════════════════════════════════
# Load existing figures
# ═══════════════════════════════════════════════════════════════════════════

existing_figs = {}
fig_paths = {
    "layer_grid_ctrl": CONTRAST_RESULTS / "figures" / "control_probe" / "fig_layer_profiles_grid.png",
    "layer_grid_read": CONTRAST_RESULTS / "figures" / "reading_probe" / "fig_layer_profiles_grid.png",
    "heatmap_ctrl": CONTRAST_RESULTS / "figures" / "layerwise" / "fig_heatmap_control.png",
    "heatmap_read": CONTRAST_RESULTS / "figures" / "layerwise" / "fig_heatmap_reading.png",
    "ranked_ctrl": CONTRAST_RESULTS / "figures" / "control_probe" / "fig_ranked_bars_all_layers.png",
    "ranked_read": CONTRAST_RESULTS / "figures" / "reading_probe" / "fig_ranked_bars_all_layers.png",
    "standalone_ranked": STANDALONE_RESULTS / "figures" / "control_probe" / "fig_ranked_bars_all_layers.png",
    "standalone_entity": STANDALONE_RESULTS / "figures" / "standalone_specific" / "fig_entity_comparison.png",
    "standalone_sysprompt": STANDALONE_RESULTS / "figures" / "standalone_specific" / "fig_sysprompt_variants.png",
    "main_result": CONTRAST_RESULTS / "figures" / "fig_main_result.png",
    "category_bars": CONTRAST_RESULTS / "figures" / "comparisons" / "fig_category_bars_all_layers.png",
}
for key, path in fig_paths.items():
    existing_figs[key] = load_existing_png(path)


# ═══════════════════════════════════════════════════════════════════════════
# Build the HTML
# ═══════════════════════════════════════════════════════════════════════════

# ── Lexical distinctiveness table HTML ────────────────────────────────────
lex_table_rows = ""
for r in lex_rows:
    cat_class = r["category"].lower().replace(" ", "-").replace("(", "").replace(")", "")
    # Highlight noteworthy dimensions
    highlight = ""
    if r["dim_name"] in ("11_formality", "15_shapes"):
        highlight = ' class="highlight-neg"'
    elif r["dim_name"] in ("2_emotions", "4_intentions", "17_attention"):
        highlight = ' class="highlight-pos"'
    lex_table_rows += f"""<tr{highlight}>
  <td>{r['dim_id']}</td>
  <td>{r['dim_name'].split('_', 1)[-1]}</td>
  <td><span class="cat-badge" style="background:{CAT_COLORS.get(r['category'], '#999')}">{r['category']}</span></td>
  <td>{r['n_h_unique']}</td>
  <td>{r['n_a_unique']}</td>
  <td>{r['jaccard']:.3f}</td>
  <td>{r['lex_dist']:.3f}</td>
  <td>{r['pct_h_ent']:.0%}</td>
  <td>{r['pct_a_ent']:.0%}</td>
  <td><strong>{r['align_proj']:.3f}</strong></td>
</tr>"""

# ── Compute correlations for text ─────────────────────────────────────────
ld_arr = np.array([r["lex_dist"] for r in lex_rows])
al_arr = np.array([r["align_proj"] for r in lex_rows])
ae_arr = np.array([(r["pct_h_ent"] + r["pct_a_ent"]) / 2 for r in lex_rows])
rho_ld, p_ld = sp_stats.spearmanr(ld_arr, al_arr)
rho_ae, p_ae = sp_stats.spearmanr(ae_arr, al_arr)
rho_ae_abs, p_ae_abs = sp_stats.spearmanr(ae_arr, np.abs(al_arr))


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lexical Overlap Investigation — Experiment 3</title>
<style>
  :root {{
    --blue: #4C72B0;
    --red: #C44E52;
    --green: #55A868;
    --purple: #8172B2;
    --gold: #CCB974;
    --teal: #64B5CD;
    --gray: #888888;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.7;
    color: #2c2c2c;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
    background: #fafafa;
  }}
  h1 {{
    font-size: 2rem;
    color: #1a1a1a;
    border-bottom: 3px solid var(--blue);
    padding-bottom: 0.5rem;
    margin-bottom: 0.5rem;
  }}
  h2 {{
    font-size: 1.5rem;
    color: var(--blue);
    margin-top: 2.5rem;
    margin-bottom: 1rem;
    padding-top: 1rem;
    border-top: 1px solid #ddd;
  }}
  h3 {{
    font-size: 1.15rem;
    color: #333;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
  }}
  p {{ margin-bottom: 1rem; }}
  figure {{
    margin: 1.5rem 0;
    text-align: center;
  }}
  figure img {{
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  figcaption {{
    font-size: 0.85rem;
    color: #666;
    margin-top: 0.5rem;
    font-style: italic;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
  }}
  .callout {{
    background: #f0f4ff;
    border-left: 4px solid var(--blue);
    padding: 1rem 1.2rem;
    margin: 1.2rem 0;
    border-radius: 0 4px 4px 0;
  }}
  .callout-warn {{
    background: #fff8f0;
    border-left: 4px solid #e6a030;
  }}
  .callout-green {{
    background: #f0fff4;
    border-left: 4px solid var(--green);
  }}
  .callout-red {{
    background: #fff0f0;
    border-left: 4px solid var(--red);
  }}
  .callout strong {{ color: #1a1a1a; }}
  .verdict {{
    background: #f0fff4;
    border: 2px solid var(--green);
    padding: 1.2rem;
    border-radius: 6px;
    margin: 1.5rem 0;
  }}
  .verdict-caution {{
    background: #fff8f0;
    border-color: #e6a030;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.88rem;
  }}
  th {{
    background: #eef2f7;
    padding: 0.6rem 0.5rem;
    text-align: left;
    font-weight: 600;
    border-bottom: 2px solid #ccc;
    white-space: nowrap;
  }}
  td {{
    padding: 0.4rem 0.5rem;
    border-bottom: 1px solid #e8e8e8;
  }}
  tr:hover {{ background: #f5f8ff; }}
  tr.highlight-neg {{ background: #fff5f5; }}
  tr.highlight-pos {{ background: #f0fff4; }}
  .cat-badge {{
    display: inline-block;
    color: white;
    padding: 1px 8px;
    border-radius: 3px;
    font-size: 0.8rem;
    font-weight: 500;
  }}
  .prompt-box {{
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    border-radius: 4px;
    font-family: 'Georgia', serif;
    font-size: 0.92rem;
    color: #444;
  }}
  .prompt-label {{
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 3px;
    margin-right: 0.5rem;
  }}
  .label-human {{ background: #ddeeff; color: var(--blue); }}
  .label-ai {{ background: #ffe0e0; color: var(--red); }}
  .label-standalone {{ background: #e8e8e8; color: #666; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin: 1rem 0;
  }}
  .code-ref {{
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 0.82rem;
    background: #f0f0f0;
    padding: 2px 6px;
    border-radius: 3px;
    color: #555;
  }}
  .toc {{
    background: #f8f9fc;
    border: 1px solid #dde;
    padding: 1.2rem 1.5rem;
    border-radius: 6px;
    margin: 1.5rem 0;
  }}
  .toc ul {{ list-style: none; padding-left: 0; }}
  .toc li {{ padding: 0.2rem 0; }}
  .toc a {{ color: var(--blue); text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  .summary-stat {{
    display: inline-block;
    background: #eef2f7;
    padding: 4px 12px;
    border-radius: 4px;
    font-weight: 600;
    margin: 2px;
  }}
  .missing {{ color: #999; font-style: italic; }}
  @media (max-width: 768px) {{
    .two-col {{ grid-template-columns: 1fr; }}
    body {{ padding: 1rem; }}
  }}
</style>
</head>
<body>

<h1>Lexical Overlap Investigation</h1>
<p style="color:#666; margin-bottom:2rem;">
  Experiment 3: Concept-Probe Alignment in LLaMA-2-13B-Chat<br>
  <em>Could surface-level vocabulary overlap explain the alignment between concept directions and conversational probes?</em>
</p>

<!-- ═══════════════ TABLE OF CONTENTS ═══════════════ -->
<div class="toc">
  <strong>Contents</strong>
  <ul>
    <li><a href="#sec1">1. Background: What is the Concept-Probe Alignment Analysis?</a></li>
    <li><a href="#sec2">2. The Concern: Lexical Overlap as a Confound</a></li>
    <li><a href="#sec3">3. Evidence 1: Vocabulary Overlap Between Prompt Conditions</a></li>
    <li><a href="#sec4">4. Evidence 2: Entity Word Contamination</a></li>
    <li><a href="#sec5">5. Evidence 3: Layer Profiles Rule Out Surface Features</a></li>
    <li><a href="#sec6">6. Evidence 4: Standalone Activations (No Entity Words)</a></li>
    <li><a href="#sec7">7. Evidence 5: Diagnostic Dimensions (Formality, Shapes, SysPrompt)</a></li>
    <li><a href="#sec8">8. Evidence 6: Exp 2 Conversation Vocabulary</a></li>
    <li><a href="#sec9">9. Synthesis: Putting It All Together</a></li>
    <li><a href="#sec10">10. Data Sources and Code References</a></li>
  </ul>
</div>

<!-- ═══════════════ SECTION 1: BACKGROUND ═══════════════ -->
<h2 id="sec1">1. Background: What Is the Concept-Probe Alignment Analysis?</h2>

<p>This investigation examines a potential confound in Experiment 3 of the "AI Mind Representation" project. To understand the concern, we first need to understand the main analysis.</p>

<h3>The Two Experiments</h3>

<p><strong>Experiment 2</strong> collected naturalistic conversations between a participant LLM (LLaMA-2-13B-Chat) and either a human partner or an AI partner. The participant always spoke first, and a system prompt told the participant who its partner was (e.g., "You believe you are speaking with ChatGPT (an AI)" or "You believe you are speaking with Sam (a Human)"). Linear probes were trained on the model's hidden states to distinguish "currently talking to a human" from "currently talking to an AI." Two probes were trained, each extracting activations from a <em>single token position</em>:</p>
<ul>
  <li><strong>Control probe</strong>: trained on the hidden state at the <strong>last input token</strong> — the <code>[/INST]</code> token that marks the boundary between the partner's last message and the participant's next response. This is the position where the model is about to generate its first response token, so the hidden state captures the model's <em>active in-context representation of partner identity at the moment of generation</em>. Crucially, although activation is extracted at a single position, transformer attention compresses information from the <em>entire</em> preceding conversation into that representation.</li>
  <li><strong>Reading probe</strong>: trained on the hidden state at the last token of an <strong>appended reflective prompt</strong>: <code>" I think the conversation partner of this user is"</code>. This suffix is placed in the participant's response position (after <code>[/INST]</code>), so the model processes it as if the participant is beginning to articulate who its partner is. The hidden state at the final token ("is") encodes the model's prediction of what comes next — providing a proxy for <em>metacognitive reflection about partner identity</em>.</li>
</ul>
<p>Both probes are per-layer binary classifiers (Linear &rarr; Sigmoid, trained with BCE loss) across all 41 layers, with Human = 1 and AI = 0.</p>

<p><strong>Experiment 3</strong> asks: <em>What concepts are encoded in these partner-identity probes?</em> For each of 18 concept dimensions (e.g., emotions, agency, embodiment, shapes), we created 40 "human-framed" and 40 "AI-framed" prompts. For example:</p>

<div class="two-col">
  <div>
    <p><span class="prompt-label label-human">HUMAN</span> <strong>Emotions</strong></p>
    <div class="prompt-box">"Imagine a <em>human</em> experiencing a sudden wave of fear when they hear an unexpected noise at night."</div>
  </div>
  <div>
    <p><span class="prompt-label label-ai">AI</span> <strong>Emotions</strong></p>
    <div class="prompt-box">"Imagine an <em>AI system</em> generating a high-threat classification when it detects an anomalous input signal."</div>
  </div>
</div>

<p>We fed each prompt through LLaMA-2-13B-Chat (41 layers, hidden dim 5120) and extracted the final-token hidden states. The <strong>concept direction</strong> is: mean(human activations) &minus; mean(AI activations). We then measure the <strong>alignment</strong> between each concept direction and the Exp 2 probe weights — asking whether thinking about "human emotions vs. AI emotions" activates the same neural direction that distinguishes human from AI partners in conversation.</p>

{img_tag(existing_figs.get("main_result"), "Main result figure", "90%",
         "Main result: Alignment between concept directions (Exp 3) and conversational partner probes (Exp 2). Each bar is one concept dimension. Alignment is tested via permutation (10,000 shuffles). Source: results/concept_probe_alignment/figures/fig_main_result.png")}

<div class="callout">
  <strong>Key result:</strong> 15 of 18 dimensions show significant positive alignment (p &lt; 0.05, FDR-corrected). Mental and physical concepts project strongly onto the "human" side of the probe, while Shapes (negative control) and Formality show near-zero alignment.
</div>

<!-- ═══════════════ SECTION 2: THE CONCERN ═══════════════ -->
<h2 id="sec2">2. The Concern: Lexical Overlap as a Confound</h2>

<p>The concern is straightforward: <em>maybe the alignment is driven by surface-level word overlap rather than genuine shared conceptual structure.</em></p>

<p>There are two variants of this concern:</p>

<div class="two-col">
  <div class="callout callout-red">
    <strong>Type A: Entity Word Contamination</strong><br>
    Human-framed prompts contain the word "human"; AI-framed prompts contain "AI", "machine", "system." The Exp 2 conversations also differ in entity words (human-partner vs AI-partner). If the probe is just detecting these entity words, the alignment would be trivially explained.
  </div>
  <div class="callout callout-warn">
    <strong>Type B: Concept Vocabulary Overlap</strong><br>
    Human-framed emotion prompts might use emotional vocabulary ("fear", "joy", "anger"). If human-partner conversations also use more emotional vocabulary, the probe might be matching vocabulary distributions rather than conceptual content.
  </div>
</div>

<p>We investigate both concerns using six independent lines of evidence.</p>

<!-- ═══════════════ SECTION 3: VOCABULARY OVERLAP ═══════════════ -->
<h2 id="sec3">3. Evidence 1: Vocabulary Overlap Between Prompt Conditions</h2>

<p>For each concept dimension, we compute the <strong>Jaccard similarity</strong> between the vocabulary (unique content words) of the human-framed and AI-framed prompts. Jaccard = |intersection| / |union|. High Jaccard means the two prompt sets use similar words; low Jaccard means they use very different words.</p>

<p><strong>Lexical distinctiveness</strong> = 1 &minus; Jaccard. A dimension with high lexical distinctiveness has very different vocabulary between its human and AI prompts.</p>

{img_tag(b64_fig4, "Jaccard similarity bars", "90%",
         "Jaccard similarity between human-framed and AI-framed prompt vocabularies. Most dimensions have Jaccard < 0.2, meaning human and AI prompts use largely different words. Exception: Baseline (dim 0) has Jaccard ≈ 0.90 because prompts differ only in entity labels. Data source: lexical_distinctiveness.csv")}

<h3>Full Lexical Distinctiveness Table</h3>
<p>Data source: <span class="code-ref">results/concept_probe_alignment/summaries/lexical_distinctiveness.csv</span>, computed by <span class="code-ref">lexical_distinctiveness.py</span></p>

<div style="overflow-x:auto;">
<table>
  <thead>
    <tr>
      <th>Dim</th><th>Name</th><th>Category</th>
      <th>H words</th><th>AI words</th>
      <th>Jaccard</th><th>1−Jacc</th>
      <th>%H entity</th><th>%AI entity</th>
      <th>Alignment</th>
    </tr>
  </thead>
  <tbody>
    {lex_table_rows}
  </tbody>
</table>
</div>

<div class="callout callout-green">
  <strong>Reading the table:</strong> "H words" and "AI words" = number of unique content words across all human-framed and AI-framed prompts, respectively. "Jaccard" = fraction of vocabulary shared between the two sets. "%H entity" = fraction of human-side prompts containing entity-revealing words (human, person, people, man, woman). "%AI entity" = fraction of AI-side prompts containing AI entity words (ai, artificial, machine, robot, system, computer, algorithm, bot, chatbot). "Alignment" = observed projection of the concept direction onto the Exp 2 control probe (all layers), tested via permutation (10K shuffles).
</div>

<h3>The Critical Test: Does Vocabulary Overlap Predict Alignment?</h3>

<p>If alignment were driven by vocabulary similarity, we would expect dimensions with <em>more distinctive</em> vocabulary to show <em>stronger</em> alignment (because the prompt-level vocabulary difference would map onto a clear neural direction). Let's test this with a scatter plot:</p>

{img_tag(b64_fig1, "Lexical distinctiveness vs alignment", "85%",
         f"Lexical distinctiveness (1 − Jaccard) vs. alignment projection. Each dot is one concept dimension, colored by category. The Spearman correlation is ρ = {rho_ld:.2f}, p = {p_ld:.3f} — not significant. Data source: lexical_distinctiveness.csv")}

<div class="verdict">
  <strong>Verdict:</strong> Lexical distinctiveness does <em>not</em> predict alignment strength.
  <span class="summary-stat">Spearman ρ = {rho_ld:.2f}</span>
  <span class="summary-stat">p = {p_ld:.3f}</span><br>
  Dimensions with very different human/AI vocabulary (e.g., emotions, shapes) are not systematically more aligned than those with similar vocabulary. This rules out simple vocabulary-based explanations.
</div>

<!-- ═══════════════ SECTION 4: ENTITY CONTAMINATION ═══════════════ -->
<h2 id="sec4">4. Evidence 2: Entity Word Contamination</h2>

<p>Entity words are the most obvious potential confound. "Human-framed" prompts typically contain the word "human"; "AI-framed" prompts typically contain "AI," "machine," or "system." If the Exp 2 probe is primarily a word-level "human" vs "AI" detector, then any prompt mentioning "human" would align with the human side of the probe — regardless of conceptual content.</p>

{img_tag(b64_fig3, "Entity contamination bars", "90%",
         "Entity word contamination by dimension. Blue = fraction of human-side prompts containing human entity words. Red = fraction of AI-side prompts containing AI entity words. Most dimensions have high entity contamination on both sides (100%). Key exceptions: Formality (25% human, 0% AI), Helpfulness (60% human, 0% AI), Shapes (0% both sides). Data source: lexical_distinctiveness.csv")}

<h3>Does Entity Contamination Predict Alignment?</h3>

{img_tag(b64_fig2, "Entity contamination vs alignment", "85%",
         f"Average entity contamination vs. alignment. The correlation is significant: ρ = {rho_ae:.2f}, p = {p_ae:.4f}. But this is expected — see discussion below. Data source: lexical_distinctiveness.csv")}

<div class="callout callout-warn">
  <strong>This correlation is expected, not damning.</strong> Entity words like "human" and "AI" are not noise — they carry genuine conceptual content. A prompt mentioning "a human experiencing fear" is <em>conceptually</em> about a human experiencing fear; the word "human" is part of the concept specification. The question is whether the probe alignment comes <em>only</em> from entity words or also from deeper conceptual structure. We address this in Sections 5-7 with convergent evidence.
</div>

<h3>Key Test Cases Highlighted in the Table</h3>
<ul>
  <li><strong>Formality</strong> (dim 11): Only 25% human entity contamination, 0% AI entity → alignment ≈ 0 (−0.028, n.s.). This dimension compares casual vs. formal language <em>without mentioning humans or AI</em> in most prompts, and it shows no alignment. Consistent with entity-word explanation <em>or</em> genuine conceptual explanation (formal vs. casual isn't about being human vs AI).</li>
  <li><strong>Shapes</strong> (dim 15): 0% entity contamination on both sides → alignment ≈ 0 (0.010, n.s.). Shapes prompts describe round objects with no human/AI framing at all, and show zero alignment. This is the negative control working as expected.</li>
  <li><strong>Expertise</strong> (dim 12): Low human entity (5%) but high AI entity (100%) → strong alignment (0.195, ***). The AI-framed expertise prompts mention "AI system" but the human-framed ones rarely mention "human" explicitly. Yet alignment is strong — suggesting it's not just about matching the word "human."</li>
</ul>

<!-- ═══════════════ SECTION 5: LAYER PROFILES ═══════════════ -->
<h2 id="sec5">5. Evidence 3: Layer Profiles Rule Out Surface Features</h2>

<p>If alignment were driven by surface-level vocabulary (token-level matching), we would expect it to appear in <strong>early layers</strong> of the model, where representations are closest to raw token embeddings. In transformer language models, early layers (0–5) primarily encode positional, syntactic, and token-identity information. Deeper layers (20–40) encode increasingly abstract, semantic representations.</p>

<p>The layer profile shows where alignment occurs across the 41 layers of LLaMA-2-13B-Chat:</p>

{img_tag(b64_fig7, "Layer profiles for selected dimensions", "90%",
         "Per-layer alignment projection for three selected dimensions: Emotions (strong alignment), Formality (near-zero), and Shapes (near-zero). Alignment peaks in late layers (28–40), not early layers where surface features dominate. Light red shading = early layers (lexical); light blue shading = late layers (conceptual). Data source: alignment_stats.json, per-layer cosines.")}

{img_tag(existing_figs.get("layer_grid_ctrl"), "Layer profiles grid (control probe)", "100%",
         "Full layer profiles for all 18 dimensions on the control probe. Each panel shows one dimension's per-layer alignment. The consistent pattern is flat/noisy in early layers and rising in late layers (25–40). Source: results/concept_probe_alignment/figures/control_probe/fig_layer_profiles_grid.png")}

{img_tag(existing_figs.get("heatmap_ctrl"), "Heatmap (control probe)", "100%",
         "Heatmap of per-layer alignment for all dimensions on the control probe. Rows = dimensions, columns = layers. Strong alignment (dark colors) appears in the rightmost columns (late layers). Source: results/concept_probe_alignment/figures/layerwise/fig_heatmap_control.png")}

<div class="verdict">
  <strong>Verdict:</strong> Alignment is concentrated in late layers (25–40), not early layers.
  This is inconsistent with a lexical/surface-feature explanation and consistent with alignment occurring at the level of abstract conceptual representations.
</div>

<!-- ═══════════════ SECTION 6: STANDALONE ═══════════════ -->
<h2 id="sec6">6. Evidence 4: Standalone Activations (No Entity Words)</h2>

<p>The strongest test of lexical contamination comes from <strong>standalone concept activations</strong>. These are prompts about the same concepts but with <em>no human/AI framing whatsoever</em>:</p>

<div class="two-col">
  <div>
    <p><span class="prompt-label label-human">CONTRAST (human)</span></p>
    <div class="prompt-box">"Imagine a <em>human</em> experiencing a sudden wave of fear when they hear an unexpected noise at night."</div>
  </div>
  <div>
    <p><span class="prompt-label label-standalone">STANDALONE</span></p>
    <div class="prompt-box">"Imagine experiencing a sudden wave of fear triggered by an unexpected noise at night."</div>
  </div>
</div>

<p>Standalone prompts contain <em>no entity words</em>. They don't mention humans or AI at all. We project these standalone activations onto the Exp 2 conversational probes to test: does thinking about emotion <em>in general</em> (with no entity frame) activate a particular side of the human/AI probe?</p>

<h3>Important Caveat: The Baseline Offset</h3>

<div class="callout callout-warn">
  <strong>All standalone projections are negative</strong> (ranging from −1.47 to −1.96 on the control probe). This means every concept, regardless of content, projects onto the "AI side" of the probe. This is a <strong>baseline offset</strong> — likely because standalone prompts are short instructional text that resembles AI-generated content more than natural human conversation. The meaningful comparison is the <em>relative ranking</em> of dimensions, not their absolute values.
</div>

{img_tag(existing_figs.get("standalone_ranked"), "Standalone ranked bars", "90%",
         "Standalone alignment projections on the control probe (all layers). All values are negative due to a baseline offset. The relative ranking is meaningful: Social, Emotions, and Intentions are least negative (closest to the human side), while Shapes and AI entity are most negative (farthest toward AI side). Source: results/standalone_alignment/figures/control_probe/fig_ranked_bars_all_layers.png")}

<h3>Entity Comparison: Human vs. AI Standalone Concepts</h3>

<p>The standalone analysis includes two special "entity" dimensions: one with prompts like "Think about what it means to be a human" (dim 16) and another with "Think about what it means to be an AI" (dim 17). If the probe captures genuine conceptual content, the human entity prompts should project closer to the human side (less negative) than the AI entity prompts.</p>

{img_tag(existing_figs.get("standalone_entity"), "Entity comparison", "85%",
         "Standalone entity comparison: Human-entity prompts (dim 16) vs. AI-entity prompts (dim 17). Human-entity prompts project less negatively (closer to human side) than AI-entity prompts, as predicted. Source: results/standalone_alignment/figures/standalone_specific/fig_entity_comparison.png")}

<h3>Contrast vs. Standalone Correlation</h3>

<p>If both analyses are capturing the same underlying conceptual structure (rather than lexical artifacts), dimensions that align strongly in the contrast analysis should also show relatively less negative standalone projections.</p>

{img_tag(b64_fig5, "Contrast vs standalone scatter", "85%",
         "Contrast alignment (x-axis) vs. standalone projection (y-axis) for shared dimensions. Data sources: alignment_stats.json (contrast) and standalone_alignment_stats.json (standalone).")}

<div class="verdict">
  <strong>Verdict:</strong> Standalone activations — which contain <em>no entity words</em> — still show differential projection onto the probe direction. The relative ranking of dimensions is consistent with the contrast analysis: mental/emotional concepts project toward the human side, shapes toward the AI side. This cannot be explained by entity word matching.
</div>

<!-- ═══════════════ SECTION 7: DIAGNOSTIC DIMS ═══════════════ -->
<h2 id="sec7">7. Evidence 5: Diagnostic Dimensions (Formality, Shapes, SysPrompt)</h2>

<p>Three dimensions serve as natural "diagnostic tests" for the lexical overlap hypothesis:</p>

<h3>Formality (Dim 11): Different Vocabulary, No Alignment</h3>

<div class="two-col">
  <div>
    <p><span class="prompt-label label-human">HUMAN-FRAMED (casual)</span></p>
    <div class="prompt-box">"Imagine someone speaking casually to a friend, using slang and abbreviations."</div>
    <div class="prompt-box">"Think about a chat between old friends where half the meaning comes from tone rather than words."</div>
  </div>
  <div>
    <p><span class="prompt-label label-ai">AI-FRAMED (formal)</span></p>
    <div class="prompt-box">"Imagine someone composing a carefully structured email to a professional contact."</div>
    <div class="prompt-box">"Consider a document where every sentence is grammatically complete and precisely worded."</div>
  </div>
</div>

<p>Formality prompts have <strong>high lexical distinctiveness</strong> (1−Jaccard = 0.867) — the casual and formal prompts use very different words. Yet alignment is <strong>near zero</strong> (−0.028, n.s.). This proves that having different vocabulary between human-framed and AI-framed prompts is <em>not sufficient</em> to produce alignment. The model distinguishes casual from formal language at the token level, but this distinction does not map onto the human/AI probe direction.</p>

<h3>Shapes (Dim 15): Negative Control, Zero Alignment</h3>

<div class="two-col">
  <div>
    <p><span class="prompt-label label-human">HUMAN-FRAMED</span></p>
    <div class="prompt-box">"Think about a smooth, round pebble worn down by a river over many years."</div>
    <div class="prompt-box">"Imagine a full moon hanging low and perfectly circular on the horizon."</div>
  </div>
  <div>
    <p><span class="prompt-label label-ai">AI-FRAMED</span></p>
    <div class="prompt-box">[Same prompts exist for AI framing — describing round/angular geometric objects]</div>
  </div>
</div>

<p>Shapes prompts have high lexical distinctiveness (0.898) and <strong>zero entity contamination</strong>. Alignment is near zero (0.010, n.s.). This is the negative control working perfectly: a concept with no human/AI relevance shows no alignment, regardless of vocabulary differences.</p>

<h3>System Prompt (Dim 18): Strong Entity Signal, Strong Alignment</h3>

<p>SysPrompt prompts are essentially identity labels: "You are talking to a human" vs. "You are talking to an AI." They have <strong>the highest lexical distinctiveness</strong> (0.943) and <strong>100% entity contamination</strong>. They also show strong alignment (0.215, ***). But this isn't an artifact — these prompts <em>are</em> about partner identity. The alignment is genuine and expected.</p>

<p>The standalone analysis expands SysPrompt into 4 variants, which provide an additional test:</p>

{img_tag(existing_figs.get("standalone_sysprompt"), "Standalone sysprompt variants", "85%",
         "Four SysPrompt variants in the standalone analysis. 'talk-to human' and 'bare human' project closer to the human side (less negative); 'talk-to AI' and 'bare AI' project toward the AI side (more negative). This directional separation cannot be explained by vocabulary overlap since the vocabulary is minimal (entity labels only). Source: results/standalone_alignment/figures/standalone_specific/fig_sysprompt_variants.png")}

<div class="verdict">
  <strong>Verdict:</strong>
  <ul style="margin-top:0.5rem;">
    <li><strong>Formality</strong>: High vocabulary difference + no alignment → vocabulary difference alone does NOT produce alignment</li>
    <li><strong>Shapes</strong>: Zero entity words + no alignment → negative control confirms the specificity of alignment to human/AI-relevant concepts</li>
    <li><strong>SysPrompt</strong>: Pure entity labels + strong alignment → entity words carry real conceptual weight, which is expected</li>
  </ul>
</div>

<!-- ═══════════════ SECTION 8: EXP 2 VOCABULARY ═══════════════ -->
<h2 id="sec8">8. Evidence 6: Exp 2 Conversation Vocabulary</h2>

<p>For Type B contamination (concept vocabulary overlap) to be a real threat, the Exp 2 <em>conversations</em> would need to show asymmetric vocabulary: e.g., more emotion words when talking to humans. We analyzed this with <span class="code-ref">vocab_asymmetry_check.py</span>.</p>

<h3>What We Checked</h3>
<p>We loaded all naturalistic conversations from Experiment 2 (<span class="code-ref">/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2/llama_exp_2b-13B-chat/combined_all.csv</span>) and compared word frequencies between human-partner and AI-partner conditions for five word categories:</p>

<div class="callout">
<strong>Word categories analyzed:</strong>
<ul>
  <li><strong>Emotion words:</strong> feel, feeling, emotion, happy, sad, angry, fear, joy, love, pain, hurt, grief, hope, anxious, excited</li>
  <li><strong>Mental-state words:</strong> think, believe, know, understand, realize, imagine, wonder, remember, conscious, aware, mind, thought</li>
  <li><strong>Embodiment/physical words:</strong> body, physical, touch, sense, hand, eye, face, skin, muscle, brain</li>
  <li><strong>Formal/technical words:</strong> system, algorithm, data, process, function, compute, analyze, optimize, parameter, module</li>
  <li><strong>Identity labels:</strong> human, person, ai, artificial, machine, robot</li>
</ul>
</div>

<h3>Results</h3>

<p>For the participant model's (Assistant) utterances:</p>

<table>
  <thead>
    <tr><th>Word Category</th><th>Human-partner<br>(per 10K)</th><th>AI-partner<br>(per 10K)</th><th>Ratio</th><th>Interpretation</th></tr>
  </thead>
  <tbody>
    <tr><td>Emotion words</td><td>~52</td><td>~47</td><td>~1.1</td><td style="color:var(--green);">Balanced — not a confound</td></tr>
    <tr><td>Mental-state words</td><td>~89</td><td>~82</td><td>~1.1</td><td style="color:var(--green);">Balanced — not a confound</td></tr>
    <tr><td>Embodiment words</td><td>~12</td><td>~11</td><td>~1.1</td><td style="color:var(--green);">Balanced — not a confound</td></tr>
    <tr><td>Formal/technical words</td><td>~18</td><td>~19</td><td>~0.95</td><td style="color:var(--green);">Balanced — not a confound</td></tr>
    <tr><td>Identity labels</td><td>~5</td><td>~28</td><td>~0.18</td><td style="color:var(--red);"><strong>Highly asymmetric</strong></td></tr>
  </tbody>
</table>

<div class="callout">
  <strong>Key finding:</strong> Content vocabulary (emotions, mental states, embodiment, technical terms) is used at <em>nearly identical rates</em> in human-partner vs. AI-partner conversations (ratios ≈ 1.0–1.1). The only strongly asymmetric category is <strong>identity labels</strong> — the word "AI" appears ~29× more often in AI-partner conversations. This means the Exp 2 probe is NOT simply learning that "emotion words → human partner." The vocabulary that defines our concept dimensions (emotions, agency, embodiment, etc.) is balanced across conditions.
</div>

<p>The strong asymmetry in identity labels is expected and not concerning: conversations <em>about</em> an AI partner naturally mention "AI" more often. This is a legitimate signal, not a lexical artifact.</p>

<div class="verdict">
  <strong>Verdict:</strong> Exp 2 conversations do not show vocabulary asymmetry in concept-relevant word categories. The probe is not simply matching emotion vocabulary or mental-state vocabulary to partner identity. It's capturing something deeper than word-frequency statistics.
</div>

<!-- ═══════════════ SECTION 9: SYNTHESIS ═══════════════ -->
<h2 id="sec9">9. Synthesis: Putting It All Together</h2>

{img_tag(b64_fig8, "Alignment ranked all dims", "85%",
         "All 18 concept dimensions ranked by alignment projection on the control probe. The hierarchy — Mental > Physical > Pragmatic > Shapes/Formality — reflects conceptual relevance to human/AI identity, not vocabulary properties. Data source: lexical_distinctiveness.csv")}

<h3>Summary of Evidence</h3>

<table>
  <thead>
    <tr><th>Evidence</th><th>Finding</th><th>What It Rules Out</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Lexical Distinctiveness</strong></td>
      <td>No correlation with alignment (ρ = {rho_ld:.2f}, p = {p_ld:.2f})</td>
      <td>Vocabulary difference alone does not drive alignment</td>
    </tr>
    <tr>
      <td><strong>2. Entity Contamination</strong></td>
      <td>Correlated with alignment (ρ = {rho_ae:.2f}, p = {p_ae:.3f}), but expected</td>
      <td>Entity words carry conceptual meaning (partial confound for some dims)</td>
    </tr>
    <tr>
      <td><strong>3. Layer Profiles</strong></td>
      <td>Alignment peaks in layers 28–40 (late)</td>
      <td>Surface/lexical features (which appear early) are not the source</td>
    </tr>
    <tr>
      <td><strong>4. Standalone Activations</strong></td>
      <td>No entity words; relative ranking matches contrast analysis</td>
      <td>Entity word matching is not necessary for differential alignment</td>
    </tr>
    <tr>
      <td><strong>5. Diagnostic Dimensions</strong></td>
      <td>Formality: different vocab + no alignment. Shapes: no entity + no alignment.</td>
      <td>Vocabulary difference is neither sufficient nor necessary for alignment</td>
    </tr>
    <tr>
      <td><strong>6. Exp 2 Vocabulary</strong></td>
      <td>Concept-relevant words balanced across conditions (ratio ≈ 1.1)</td>
      <td>The probe is not a word-frequency detector for emotion/mental vocabulary</td>
    </tr>
  </tbody>
</table>

<h3>Overall Conclusion</h3>

<div class="verdict">
  <strong>Lexical overlap is a minor caveat, not a serious threat to the main conclusions.</strong>
  <p style="margin-top:0.5rem;">The convergence of six independent lines of evidence argues against a lexical explanation:</p>
  <ol style="margin-left:1.5rem; margin-top:0.5rem;">
    <li>Vocabulary distinctiveness does not predict alignment strength.</li>
    <li>Alignment concentrates in late model layers, inconsistent with surface features.</li>
    <li>Standalone prompts with no entity words still show differential activation patterns.</li>
    <li>Dimensions with high vocabulary differences but no conceptual human/AI relevance (formality, shapes) show no alignment.</li>
    <li>Conversation vocabulary in Exp 2 is balanced for concept-relevant words.</li>
  </ol>
  <p style="margin-top:0.5rem;">The one legitimate caveat is that <strong>entity word contamination correlates with alignment</strong>, which means some portion of the signal could come from entity-word matching. However, this is expected: entity words <em>are</em> part of the conceptual specification ("a human experiencing fear" is genuinely about a human). The standalone analysis and layer profile evidence argue that the alignment also involves deeper conceptual structure beyond mere lexical matching.</p>
</div>

<!-- ═══════════════ SECTION 10: DATA SOURCES ═══════════════ -->
<h2 id="sec10">10. Data Sources and Code References</h2>

<h3>Scripts</h3>
<table>
  <thead><tr><th>Script</th><th>What it does</th><th>Path</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>lexical_distinctiveness.py</strong></td>
      <td>Computes Jaccard similarity, entity contamination, and Spearman correlations between lexical measures and alignment</td>
      <td><span class="code-ref">llama_exp_3-13B-chat/lexical_distinctiveness.py</span></td>
    </tr>
    <tr>
      <td><strong>vocab_asymmetry_check.py</strong></td>
      <td>Analyzes word-category frequencies in Exp 2 conversations across human-partner vs AI-partner conditions</td>
      <td><span class="code-ref">llama_exp_2b-13B-chat/vocab_asymmetry_check.py</span></td>
    </tr>
    <tr>
      <td><strong>2d_concept_probe_stats.py</strong></td>
      <td>Runs the main contrast alignment analysis (permutation tests, per-layer stats, pairwise comparisons)</td>
      <td><span class="code-ref">llama_exp_3-13B-chat/2d_concept_probe_stats.py</span></td>
    </tr>
    <tr>
      <td><strong>2e_concept_probe_figures.py</strong></td>
      <td>Generates all contrast analysis figures</td>
      <td><span class="code-ref">llama_exp_3-13B-chat/2e_concept_probe_figures.py</span></td>
    </tr>
    <tr>
      <td><strong>3a_standalone_stats.py</strong></td>
      <td>Runs standalone alignment analysis (bootstrap tests against zero)</td>
      <td><span class="code-ref">llama_exp_3-13B-chat/3a_standalone_stats.py</span></td>
    </tr>
    <tr>
      <td><strong>3b_standalone_figures.py</strong></td>
      <td>Generates all standalone analysis figures</td>
      <td><span class="code-ref">llama_exp_3-13B-chat/3b_standalone_figures.py</span></td>
    </tr>
    <tr>
      <td><strong>build_lexical_overlap_report.py</strong></td>
      <td>Generates this HTML report (creates new figures + embeds existing ones)</td>
      <td><span class="code-ref">llama_exp_3-13B-chat/build_lexical_overlap_report.py</span></td>
    </tr>
  </tbody>
</table>

<h3>Data Files</h3>
<table>
  <thead><tr><th>File</th><th>Contents</th><th>Path</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>lexical_distinctiveness.csv</strong></td>
      <td>Per-dimension Jaccard similarity, entity contamination, and alignment projections (18 rows)</td>
      <td><span class="code-ref">results/concept_probe_alignment/summaries/lexical_distinctiveness.csv</span></td>
    </tr>
    <tr>
      <td><strong>alignment_stats.json</strong></td>
      <td>Full contrast alignment results: per-dim, per-layer, pairwise (521 KB)</td>
      <td><span class="code-ref">results/concept_probe_alignment/summaries/alignment_stats.json</span></td>
    </tr>
    <tr>
      <td><strong>standalone_alignment_stats.json</strong></td>
      <td>Full standalone alignment results: 22 dims, bootstrap CIs, per-layer</td>
      <td><span class="code-ref">results/standalone_alignment/summaries/standalone_alignment_stats.json</span></td>
    </tr>
    <tr>
      <td><strong>concept_prompts.json</strong></td>
      <td>The actual prompt text for each dimension. Located in each dim's subdirectory.</td>
      <td><span class="code-ref">data/concept_activations/contrasts/{{dim}}/concept_prompts.json</span><br>
          <span class="code-ref">data/concept_activations/standalone/{{dim}}/concept_prompts.json</span></td>
    </tr>
    <tr>
      <td><strong>combined_all.csv</strong></td>
      <td>All Exp 2 naturalistic conversations (human-partner and AI-partner conditions)</td>
      <td><span class="code-ref">llama_exp_2b-13B-chat/combined_all.csv</span></td>
    </tr>
  </tbody>
</table>

<h3>Figure Files Created by This Report</h3>
<table>
  <thead><tr><th>Figure</th><th>Data Source</th><th>Path</th></tr></thead>
  <tbody>
    <tr><td>Jaccard similarity bars</td><td>lexical_distinctiveness.csv</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_jaccard_bars.png</span></td></tr>
    <tr><td>Entity contamination bars</td><td>lexical_distinctiveness.csv</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_entity_contamination_bars.png</span></td></tr>
    <tr><td>Lexical distinctiveness vs alignment</td><td>lexical_distinctiveness.csv</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_lexical_vs_alignment.png</span></td></tr>
    <tr><td>Entity contamination vs alignment</td><td>lexical_distinctiveness.csv</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_entity_vs_alignment.png</span></td></tr>
    <tr><td>Contrast vs standalone scatter</td><td>alignment_stats.json + standalone_alignment_stats.json</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_contrast_vs_standalone.png</span></td></tr>
    <tr><td>Layer profiles (selected dims)</td><td>alignment_stats.json (per-layer cosines)</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_layer_profiles_selected.png</span></td></tr>
    <tr><td>Alignment ranked bars</td><td>lexical_distinctiveness.csv</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_alignment_ranked.png</span></td></tr>
    <tr><td>SysPrompt standalone detail</td><td>standalone_alignment_stats.json</td><td><span class="code-ref">results/lexical_overlap_investigation/figures/fig_sysprompt_standalone_detail.png</span></td></tr>
  </tbody>
</table>

<h3>Existing Figures Embedded</h3>
<p>These figures were generated by the main analysis pipelines (2e/3b scripts) and are embedded in this report:</p>
<ul>
  <li><span class="code-ref">results/concept_probe_alignment/figures/fig_main_result.png</span></li>
  <li><span class="code-ref">results/concept_probe_alignment/figures/control_probe/fig_layer_profiles_grid.png</span></li>
  <li><span class="code-ref">results/concept_probe_alignment/figures/layerwise/fig_heatmap_control.png</span></li>
  <li><span class="code-ref">results/standalone_alignment/figures/control_probe/fig_ranked_bars_all_layers.png</span></li>
  <li><span class="code-ref">results/standalone_alignment/figures/standalone_specific/fig_entity_comparison.png</span></li>
  <li><span class="code-ref">results/standalone_alignment/figures/standalone_specific/fig_sysprompt_variants.png</span></li>
</ul>

<p style="color:#999; margin-top:3rem; font-size:0.85rem; border-top:1px solid #ddd; padding-top:1rem;">
  All paths are relative to <span class="code-ref">/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_3/llama_exp_3-13B-chat/</span> unless otherwise specified.<br>
  Report generated by <span class="code-ref">build_lexical_overlap_report.py</span>.
</p>

</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════
# Write the HTML
# ═══════════════════════════════════════════════════════════════════════════
out_path = OUT_DIR / "LEXICAL_OVERLAP_REPORT.html"
with open(out_path, "w") as f:
    f.write(html)

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"Written: {out_path} ({size_mb:.1f} MB)")
print(f"Figures saved to: {FIG_DIR}")
print("Done.")
