#!/usr/bin/env python3
"""
Generate self-contained HTML report with figures for RSA-by-dimension analysis.
(Base model version.)

For each RSA variant (combined, experience, agency), generates:
  - fig7-style bar chart: 1x2 (without_self | with_self), gray/colored bars
  - fig10-style RDM pair: 1x2 (human RDM | model RDM at peak), per condition
  - Overlay line plot: all 3 variants on one plot, per condition

Saves figures as PNG/PDF to results/figures/ and embeds as base64 in HTML.

Usage:
    python gen_rsa_report.py

Env: llama2_env (CPU only)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import json
import base64
from io import BytesIO
from datetime import datetime
import numpy as np
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from entities.gray_entities import GRAY_ET_AL_SCORES, ENTITY_PROMPTS, ENTITY_NAMES

MODEL_LABEL = "LLaMA-2-13B (Base)"
PROMPT_DESC = (
    "Each entity prompt is tokenized as raw text (no chat template, no "
    "system prompt, no <code>[INST]</code> tags). The model sees the plain "
    "prompt string directly."
)

FIG_DIR = os.path.join("results", "figures")

# ── Style ──
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

C_COMBINED = "#4daf4a"    # green
C_EXPERIENCE = "#2166ac"  # blue
C_AGENCY = "#b2182b"      # red
VARIANT_COLORS = {"combined": C_COMBINED, "experience": C_EXPERIENCE,
                  "agency": C_AGENCY}
VARIANT_LABELS = {"combined": "Combined (2D)", "experience": "Experience",
                  "agency": "Agency"}


def nice_entity(name):
    lookup = {
        "dead_woman": "Dead woman", "frog": "Frog",
        "robot": "Robot (Kismet)", "fetus": "Fetus (7 wk)",
        "pvs_patient": "PVS patient", "god": "God", "dog": "Dog",
        "chimpanzee": "Chimpanzee", "baby": "Baby (5 mo)",
        "girl": "Girl (5 yr)", "adult_woman": "Adult woman",
        "adult_man": "Adult man", "you_self": "You (self)",
    }
    return lookup.get(name, name)


def _isnan(val):
    return val is None or (isinstance(val, float) and np.isnan(val))


def apply_fdr(rsa_results):
    """Add FDR-corrected p-values (Benjamini-Hochberg) to RSA results in-place."""
    pvals, valid_idx = [], []
    for i, r in enumerate(rsa_results):
        if not _isnan(r["p_value"]):
            pvals.append(r["p_value"])
            valid_idx.append(i)
    for r in rsa_results:
        r["p_fdr"] = float("nan")
    if pvals:
        _, p_corr, _, _ = multipletests(pvals, method='fdr_bh')
        for idx, pc in zip(valid_idx, p_corr):
            rsa_results[idx]["p_fdr"] = float(pc)
    return rsa_results


def apply_fdr_to_all(rsa_data):
    """Apply FDR correction to all variants in an rsa_data dict."""
    if rsa_data is None:
        return None
    for vname in ["combined", "experience", "agency"]:
        if vname in rsa_data:
            apply_fdr(rsa_data[vname])
    return rsa_data


# ═══════════════════════════ DATA LOADING ═══════════════════════════

def load_rsa(tag):
    for base in ["data/entity_activations", "results"]:
        path = os.path.join(base, tag, "rsa_results.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def load_rdm(tag):
    path = os.path.join("data", "entity_activations", tag,
                        "rdm_cosine_per_layer.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def find_peak(rsa_results):
    valid = [r for r in rsa_results if not _isnan(r["rho"])]
    if not valid:
        return {"layer": -1, "rho": float("nan"), "p_value": float("nan")}
    return max(valid, key=lambda r: r["rho"])


def get_human_rdm(rdm_data, variant_name):
    key = f"human_rdm_{variant_name}"
    if key in rdm_data:
        return rdm_data[key]
    if variant_name == "combined" and "human_rdm" in rdm_data:
        return rdm_data["human_rdm"]
    return None


# ═══════════════════════════ FIGURE HELPERS ═══════════════════════════

def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    print(f"  Saved {name}.png/.pdf -> {FIG_DIR}")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ═══════════════════ FIGURE TYPE 1: BAR CHART (fig7-style) ═══════════════════

def fig_rsa_bars(rsa_ns, rsa_ws, variant_name):
    color = VARIANT_COLORS[variant_name]
    vlabel = VARIANT_LABELS[variant_name]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, (cond, rsa_results) in enumerate([
        ("without_self", rsa_ns), ("with_self", rsa_ws),
    ]):
        ax = axes[idx]
        if rsa_results is None:
            ax.set_visible(False)
            continue

        layers = [r["layer"] for r in rsa_results]
        rhos = [r["rho"] if not _isnan(r["rho"]) else 0.0
                for r in rsa_results]
        qvals = [r.get("p_fdr", r["p_value"])
                 if not _isnan(r.get("p_fdr", r["p_value"])) else 1.0
                 for r in rsa_results]

        colors = [color if q < 0.05 else "#cccccc" for q in qvals]
        ax.bar(layers, rhos, color=colors, edgecolor="white", width=0.8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Spearman rho (model RDM vs human RDM)")

        n_ent = 13 if cond == "with_self" else 12
        n_sig = sum(1 for q in qvals if q < 0.05)
        ax.set_title(f"{cond.replace('_', ' ').title()} ({n_ent} entities)\n"
                     f"{n_sig}/{len(layers)} layers q < .05 (FDR)")
        ax.set_ylim(-0.35, 0.55)

        peak = find_peak(rsa_results)
        if peak["layer"] >= 0:
            ax.annotate(
                f"peak: rho={peak['rho']:.3f}\nlayer {peak['layer']}",
                (peak["layer"], peak["rho"]),
                textcoords="offset points", xytext=(15, 10), fontsize=8,
                arrowprops=dict(arrowstyle="->", color="gray"))

    fig.suptitle(
        f"RSA: {vlabel} — {MODEL_LABEL}\n"
        f"Cosine-distance RDM vs {vlabel.lower()} human RDM (FDR-corrected)",
        fontsize=13, y=1.06)
    fig.tight_layout()
    return save_fig(fig, f"rsa_bars_{variant_name}")


# ═══════════════════ FIGURE TYPE 2: RDM PAIR (fig10-style) ═══════════════════

def fig_rdm_pair(rdm_data, rsa_results, variant_name, entity_keys, cond):
    vlabel = VARIANT_LABELS[variant_name]
    peak = find_peak(rsa_results)
    if peak["layer"] < 0:
        return None

    human_rdm = get_human_rdm(rdm_data, variant_name)
    if human_rdm is None:
        return None
    model_rdm = rdm_data["model_rdm"]
    peak_layer = peak["layer"]

    nice_labels = [nice_entity(e) for e in entity_keys]
    n = len(entity_keys)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    im0 = ax.imshow(human_rdm, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(nice_labels, fontsize=8)
    ax.set_title(f"Human RDM: {vlabel}\n(Gray et al. factor distance)")
    fig.colorbar(im0, ax=ax, shrink=0.7, label="Dissimilarity")

    ax = axes[1]
    im1 = ax.imshow(model_rdm[peak_layer], cmap="viridis", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(nice_labels, fontsize=8)
    ax.set_title(f"Model RDM (Layer {peak_layer})\n"
                 f"Cosine distance, rho={peak['rho']:.3f}, "
                 f"p={peak['p_value']:.4f}")
    fig.colorbar(im1, ax=ax, shrink=0.7, label="Cosine Distance")

    cond_label = cond.replace("_", " ").title()
    fig.suptitle(f"RDM: {vlabel} — {n} Entities ({cond_label})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return save_fig(fig, f"rdm_{variant_name}_{cond}")


# ═══════════════════ FIGURE TYPE 3: OVERLAY ═══════════════════

def fig_rsa_overlay(rsa_data, cond_label, n_entities):
    fig, ax = plt.subplots(figsize=(10, 4.5))

    for vname in ["combined", "experience", "agency"]:
        rsa = rsa_data[vname]
        layers = [r["layer"] for r in rsa]
        rhos = [r["rho"] if not _isnan(r["rho"]) else 0.0 for r in rsa]
        qvals = [r.get("p_fdr", r["p_value"])
                 if not _isnan(r.get("p_fdr", r["p_value"])) else 1.0
                 for r in rsa]

        color = VARIANT_COLORS[vname]
        ax.plot(layers, rhos, "o-", color=color, label=VARIANT_LABELS[vname],
                lw=2, ms=4, alpha=0.85)

        sig_l = [l for l, q in zip(layers, qvals) if q < 0.05]
        sig_r = [r for r, q in zip(rhos, qvals) if q < 0.05]
        if sig_l:
            ax.scatter(sig_l, sig_r, color=color, s=60, zorder=5,
                       edgecolor="white", linewidth=0.5)

        peak = find_peak(rsa_data[vname])
        if peak["layer"] >= 0:
            ax.annotate(f"L{peak['layer']} ({peak['rho']:+.3f})",
                        (peak["layer"], peak["rho"]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color=color, fontweight="bold")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title(f"RSA by Layer — {MODEL_LABEL}\n"
                 f"{cond_label} ({n_entities} entities). "
                 f"Filled markers = q < .05 (FDR)")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.35, 0.55)
    fig.tight_layout()
    tag = cond_label.lower().replace(" ", "_")
    return save_fig(fig, f"rsa_layerwise_{tag}")


# ═══════════ FIGURE TYPE 4: 4-PANEL RDM OVERVIEW ═══════════

def fig_rdm_overview(rdm_data, rsa_data, cond_label, entity_keys):
    model_rdm = rdm_data["model_rdm"]
    peak_combined = find_peak(rsa_data["combined"])
    peak_layer = peak_combined["layer"]
    nice_labels = [nice_entity(e) for e in entity_keys]
    n = len(entity_keys)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    ax = axes[0]
    im = ax.imshow(model_rdm[peak_layer], cmap="viridis", aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(nice_labels, fontsize=7)
    ax.set_title(f"Model RDM (L{peak_layer})\nCosine distance")
    fig.colorbar(im, ax=ax, shrink=0.7)

    for i, (vname, vlabel) in enumerate([
        ("combined", "Combined\n(Euclidean 2D)"),
        ("experience", "Experience\n(|exp_i - exp_j|)"),
        ("agency", "Agency\n(|ag_i - ag_j|)"),
    ]):
        ax = axes[i + 1]
        human_rdm = get_human_rdm(rdm_data, vname)
        if human_rdm is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(human_rdm, cmap="viridis", aspect="equal")
        ax.set_xticks(range(n))
        ax.set_xticklabels(nice_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(nice_labels, fontsize=7)
        peak_v = find_peak(rsa_data[vname])
        ax.set_title(f"Human: {vlabel}\nrho={peak_v['rho']:+.3f}, "
                     f"p={peak_v['p_value']:.4f}")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"RDM Comparison — {MODEL_LABEL}, {cond_label}",
                 fontsize=14, y=1.04)
    fig.tight_layout()
    tag = cond_label.lower().replace(" ", "_")
    return save_fig(fig, f"rdm_comparison_{tag}")


# ═══════════════════════════ HTML HELPERS ═══════════════════════════

def fmt_rho(val):
    return "NaN" if _isnan(val) else f"{val:+.4f}"

def fmt_p(val):
    return "NaN" if _isnan(val) else f"{val:.4f}"

def rho_class(rho, p_fdr):
    """CSS class based on FDR-corrected p-value."""
    if _isnan(p_fdr): return ""
    if p_fdr < 0.01: return "sig-strong"
    if p_fdr < 0.05: return "sig-mod"
    if p_fdr < 0.10: return "sig-trend"
    return ""

def html_table(rsa_results):
    """Scrollable RSA table with raw and FDR-corrected p-values."""
    rows = []
    rows.append('<div style="max-height:400px;overflow-y:auto;margin:.5rem 0">')
    rows.append('<table>')
    rows.append('<tr><th class="num">Layer</th>'
                '<th class="num">Spearman rho</th>'
                '<th class="num">p (raw)</th>'
                '<th class="num">q (FDR)</th></tr>')
    for r in rsa_results:
        q = r.get("p_fdr", r["p_value"])
        cls = rho_class(r["rho"], q)
        row_cls = f' class="highlight-row"' if cls else ""
        td_cls = f' class="num {cls}"' if cls else ' class="num"'
        rows.append(f'<tr{row_cls}>'
                     f'<td class="num">{r["layer"]}</td>'
                     f'<td{td_cls}>{fmt_rho(r["rho"])}</td>'
                     f'<td class="num">{fmt_p(r["p_value"])}</td>'
                     f'<td{td_cls}>{fmt_p(q)}</td>'
                     f'</tr>')
    rows.append('</table></div>')
    return "\n".join(rows)

def html_fig(b64, caption):
    return (f'<figure>\n'
            f'  <img src="data:image/png;base64,{b64}">\n'
            f'  <figcaption>{caption}</figcaption>\n'
            f'</figure>\n')


# ═══════════════════════════ MAIN ═══════════════════════════

def generate_html():
    rsa_ws = load_rsa("with_self")
    rsa_ns = load_rsa("without_self")
    rdm_ws = load_rdm("with_self")
    rdm_ns = load_rdm("without_self")

    if rsa_ws is None and rsa_ns is None:
        print("ERROR: No rsa_results.json found. "
              "Run 1_extract_entity_representations.py first.")
        return

    # Apply FDR correction (Benjamini-Hochberg) per variant
    print("Applying FDR correction (Benjamini-Hochberg)...")
    apply_fdr_to_all(rsa_ns)
    apply_fdr_to_all(rsa_ws)

    entity_keys_ns = [k for k in ENTITY_NAMES if k != "you_self"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("Generating figures...")
    figs = {}

    for vname in ["combined", "experience", "agency"]:
        rsa_v_ns = rsa_ns[vname] if rsa_ns else None
        rsa_v_ws = rsa_ws[vname] if rsa_ws else None
        figs[f"bars_{vname}"] = fig_rsa_bars(rsa_v_ns, rsa_v_ws, vname)
        if rdm_ns is not None and rsa_v_ns is not None:
            b = fig_rdm_pair(rdm_ns, rsa_v_ns, vname, entity_keys_ns,
                             "without_self")
            if b:
                figs[f"rdm_{vname}_ns"] = b
        if rdm_ws is not None and rsa_v_ws is not None:
            b = fig_rdm_pair(rdm_ws, rsa_v_ws, vname, ENTITY_NAMES,
                             "with_self")
            if b:
                figs[f"rdm_{vname}_ws"] = b

    if rsa_ns is not None:
        figs["overlay_ns"] = fig_rsa_overlay(rsa_ns, "Without Self", 12)
    if rsa_ws is not None:
        figs["overlay_ws"] = fig_rsa_overlay(rsa_ws, "With Self", 13)
    if rdm_ns is not None and rsa_ns is not None:
        figs["rdm_overview_ns"] = fig_rdm_overview(
            rdm_ns, rsa_ns, "Without Self", entity_keys_ns)
    if rdm_ws is not None and rsa_ws is not None:
        figs["rdm_overview_ws"] = fig_rdm_overview(
            rdm_ws, rsa_ws, "With Self", ENTITY_NAMES)

    print("Building HTML report...")

    css = """*{margin:0;padding:0;box-sizing:border-box}
html{scroll-behavior:smooth}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:#1e293b;background:#f8fafc;line-height:1.6;max-width:1100px;margin:0 auto;padding:2rem}
h1{font-size:1.8rem;color:#0f172a;border-bottom:3px solid #0d9488;padding-bottom:.5rem;margin-bottom:1rem}
h2{font-size:1.4rem;color:#0f172a;margin:2rem 0 .75rem;border-bottom:2px solid #e2e8f0;padding-bottom:.4rem}
h3{font-size:1.1rem;color:#334155;margin:1.2rem 0 .5rem}
p,li{font-size:.93rem;margin-bottom:.5rem}
ul,ol{padding-left:1.5rem}
a{color:#0d9488;text-decoration:none}
a:hover{text-decoration:underline}
nav{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1rem 1.5rem;margin-bottom:2rem}
nav ol{list-style:none;padding:0;columns:2;column-gap:2rem}
nav li{padding:2px 0;font-size:.88rem}
nav li a{color:#334155}
nav li a:hover{color:#0d9488}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:1.25rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,.04)}
table{width:100%;border-collapse:collapse;margin:.75rem 0;font-size:.85rem}
th,td{padding:6px 10px;text-align:left;border-bottom:1px solid #e2e8f0}
th{background:#f1f5f9;font-weight:600;color:#334155;position:sticky;top:0}
td{color:#475569}
tr:hover td{background:#f8fafc}
.num{text-align:right;font-variant-numeric:tabular-nums}
.sig-strong{font-weight:700;color:#059669}
.sig-mod{font-weight:600;color:#0d9488}
.sig-trend{color:#6366f1}
.highlight-row td{background:#f0fdfa}
code{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:.83rem}
.small{font-size:.82rem;color:#64748b}
figure{margin:1.5rem 0;text-align:center}
figure img{max-width:100%;border:1px solid #e2e8f0;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.08)}
figcaption{font-size:.9rem;color:#64748b;margin-top:.5rem;text-align:left;padding:0 1rem}
figcaption strong{color:#1e293b}
@media(max-width:700px){body{padding:1rem}nav ol{columns:1}}"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Exp 4: RSA by Dimension — {MODEL_LABEL}</title>
<style>{css}</style>
</head>
<body>

<h1>Exp 4: RSA by Dimension &mdash; {MODEL_LABEL}</h1>
<p class="small">Generated {now}</p>

<nav>
<strong>Contents</strong>
<ol>
<li><a href="#methodology">1. Methodology</a></li>
<li><a href="#combined">2. Combined RSA</a></li>
<li><a href="#experience">3. Experience-Only RSA</a></li>
<li><a href="#agency">4. Agency-Only RSA</a></li>
<li><a href="#overlay">5. Overlay Comparison</a></li>
<li><a href="#comparison">6. Summary Table</a></li>
</ol>
</nav>

<h2 id="methodology">1. Methodology</h2>
<div class="card">
<h3>Prompts</h3>
<p>Each of the 13 Gray et al. (2007) entities is presented with a simple prompt
(e.g., &ldquo;Think about a dog.&rdquo;). {PROMPT_DESC}</p>
<h3>Model RDM</h3>
<p>Last-token residual-stream activations extracted at all 41 layers
(layer 0 = embedding, layers 1&ndash;40 = transformer blocks). Pairwise
<strong>cosine distance</strong> (1 &minus; cosine similarity) gives the model
RDM at each layer.</p>
<h3>Human RDMs</h3>
<ul>
<li><strong>Combined:</strong> Euclidean distance in 2D (Experience, Agency)
space from Gray et al. factor scores.</li>
<li><strong>Experience-only:</strong> |exp<sub>i</sub> &minus; exp<sub>j</sub>|</li>
<li><strong>Agency-only:</strong> |agency<sub>i</sub> &minus; agency<sub>j</sub>|</li>
</ul>
<h3>RSA</h3>
<p>Spearman rank correlation between upper triangles of model RDM and each
human RDM variant, computed at every layer. P-values are corrected for
multiple comparisons across 41 layers using the Benjamini-Hochberg FDR
procedure (q-values). Significance thresholds use FDR-corrected values.</p>
<h3>Entities</h3>
<table>
<tr><th>Entity</th><th>Prompt</th><th class="num">Experience</th><th class="num">Agency</th></tr>
"""
    for key in ENTITY_NAMES:
        exp, ag = GRAY_ET_AL_SCORES[key]
        html += (f'<tr><td>{key}</td><td>{ENTITY_PROMPTS[key]}</td>'
                 f'<td class="num">{exp:.2f}</td>'
                 f'<td class="num">{ag:.2f}</td></tr>\n')
    html += "</table>\n</div>\n\n"

    fig_num = 1
    for vname, vlabel, sid in [
        ("combined", "Combined (Experience + Agency)", "combined"),
        ("experience", "Experience-Only", "experience"),
        ("agency", "Agency-Only", "agency"),
    ]:
        rsa_v_ns = rsa_ns[vname] if rsa_ns else None
        rsa_v_ws = rsa_ws[vname] if rsa_ws else None
        section_num = {"combined": 2, "experience": 3, "agency": 4}[vname]
        html += f'<h2 id="{sid}">{section_num}. {vlabel} RSA</h2>\n'

        if f"bars_{vname}" in figs:
            peak_ns = find_peak(rsa_v_ns) if rsa_v_ns else None
            peak_ws = find_peak(rsa_v_ws) if rsa_v_ws else None
            parts = []
            if peak_ns and peak_ns["layer"] >= 0:
                ns_sig = sum(1 for r in rsa_v_ns
                             if not _isnan(r.get("p_fdr")) and r["p_fdr"] < 0.05)
                parts.append(
                    f"Left: without self (12 entities), {ns_sig}/41 layers "
                    f"FDR-significant, peak at layer {peak_ns['layer']} "
                    f"(rho = {peak_ns['rho']:.3f}, "
                    f"q = {peak_ns.get('p_fdr', peak_ns['p_value']):.4f})")
            if peak_ws and peak_ws["layer"] >= 0:
                ws_sig = sum(1 for r in rsa_v_ws
                             if not _isnan(r.get("p_fdr")) and r["p_fdr"] < 0.05)
                parts.append(
                    f"Right: with self (13 entities), {ws_sig}/41 layers "
                    f"FDR-significant, peak at layer {peak_ws['layer']} "
                    f"(rho = {peak_ws['rho']:.3f}, "
                    f"q = {peak_ws.get('p_fdr', peak_ws['p_value']):.4f})")
            caption = (f"<strong>Figure {fig_num}. {vlabel} RSA across "
                       f"transformer layers.</strong> Spearman rho between the "
                       f"model&rsquo;s entity cosine-distance RDM and the "
                       f"{vlabel.lower()} human RDM at each of 41 layers. "
                       f"Colored bars indicate q &lt; .05 (FDR-corrected). "
                       + " ".join(parts))
            html += html_fig(figs[f"bars_{vname}"], caption)
            fig_num += 1

        for cond_key, cond_label in [("ns", "without_self"),
                                      ("ws", "with_self")]:
            fk = f"rdm_{vname}_{cond_key}"
            if fk not in figs:
                continue
            rsa_v = rsa_v_ns if cond_key == "ns" else rsa_v_ws
            peak = find_peak(rsa_v)
            n_ent = 12 if cond_key == "ns" else 13
            cond_nice = cond_label.replace("_", " ").title()
            caption = (
                f"<strong>Figure {fig_num}. {vlabel} RDM comparison "
                f"({cond_nice}, {n_ent} entities).</strong> "
                f"Left: Human RDM derived from Gray et al. {vlabel.lower()} "
                f"distances. Right: Model RDM at peak RSA layer "
                f"(layer {peak['layer']}, rho = {peak['rho']:.3f}, "
                f"p = {peak['p_value']:.4f}).")
            html += html_fig(figs[fk], caption)
            fig_num += 1

        html += '<div class="card">\n'
        for tag_label, rsa_v in [("Without Self (12 entities)", rsa_v_ns),
                                  ("With Self (13 entities)", rsa_v_ws)]:
            if rsa_v is None:
                continue
            peak = find_peak(rsa_v)
            html += f'<h3>{tag_label}</h3>\n'
            html += (f'<p><strong>Peak:</strong> Layer {peak["layer"]}, '
                     f'rho = {fmt_rho(peak["rho"])}, '
                     f'p = {fmt_p(peak["p_value"])}</p>\n')
            html += html_table(rsa_v) + "\n"
        html += '</div>\n\n'

    html += '<h2 id="overlay">5. Overlay Comparison</h2>\n'
    if "overlay_ns" in figs:
        html += html_fig(
            figs["overlay_ns"],
            f"<strong>Figure {fig_num}. All three RSA variants overlaid "
            f"&mdash; Without Self (12 entities).</strong> "
            f"Green = Combined, Blue = Experience, Red = Agency. "
            f"Filled markers indicate q &lt; .05 (FDR-corrected).")
        fig_num += 1
    if "overlay_ws" in figs:
        html += html_fig(
            figs["overlay_ws"],
            f"<strong>Figure {fig_num}. All three RSA variants overlaid "
            f"&mdash; With Self (13 entities).</strong> "
            f"Green = Combined, Blue = Experience, Red = Agency. "
            f"Filled markers indicate q &lt; .05 (FDR-corrected).")
        fig_num += 1
    if "rdm_overview_ns" in figs:
        html += html_fig(
            figs["rdm_overview_ns"],
            f"<strong>Figure {fig_num}. RDM overview &mdash; Without Self "
            f"(12 entities).</strong> Model RDM at peak combined-RSA layer "
            f"alongside all three human RDM variants.")
        fig_num += 1
    if "rdm_overview_ws" in figs:
        html += html_fig(
            figs["rdm_overview_ws"],
            f"<strong>Figure {fig_num}. RDM overview &mdash; With Self "
            f"(13 entities).</strong> Model RDM at peak combined-RSA layer "
            f"alongside all three human RDM variants.")
        fig_num += 1

    html += '<h2 id="comparison">6. Summary Table</h2>\n'
    html += '<div class="card">\n'
    html += '<p>Peak RSA layer and Spearman rho for each human RDM variant. Significance uses FDR-corrected q-values.</p>\n'
    html += ('<table><tr><th>Variant</th><th>Condition</th>'
             '<th class="num">Peak Layer</th>'
             '<th class="num">Peak rho</th>'
             '<th class="num">p (raw)</th>'
             '<th class="num">q (FDR)</th></tr>\n')
    for tag_label, rsa_data in [("Without Self", rsa_ns),
                                 ("With Self", rsa_ws)]:
        if rsa_data is None:
            continue
        for vn, vl in [("combined", "Combined"),
                        ("experience", "Experience"),
                        ("agency", "Agency")]:
            pk = find_peak(rsa_data[vn])
            q = pk.get("p_fdr", pk["p_value"])
            cls = rho_class(pk["rho"], q)
            td = f' class="num {cls}"' if cls else ' class="num"'
            html += (f'<tr><td>{vl}</td><td>{tag_label}</td>'
                     f'<td class="num">{pk["layer"]}</td>'
                     f'<td{td}>{fmt_rho(pk["rho"])}</td>'
                     f'<td class="num">{fmt_p(pk["p_value"])}</td>'
                     f'<td{td}>{fmt_p(q)}</td></tr>\n')
    html += '</table>\n</div>\n'

    html += ('\n<p class="small" style="margin-top:2rem;text-align:center">'
             'Generated by <code>gen_rsa_report.py</code></p>\n'
             '</body>\n</html>')

    out_path = "rsa_report.html"
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    generate_html()
