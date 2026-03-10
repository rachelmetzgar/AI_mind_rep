#!/usr/bin/env python3
"""
Experiment 3, Phase 9c: Concept-Conversation Alignment Report

Generates HTML/MD report from saved stats CSVs and figures produced by 9b.

Output:
    results/llama2_13b_chat/{version}/concept_conversation/turn_{turn}/
        concept_conversation_report.html
        concept_conversation_report.md

Usage:
    python 9c_concept_conversation_report.py --version balanced_gpt
    python 9c_concept_conversation_report.py --version balanced_gpt --turn 3

Env: llama2_env (login node, lightweight)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import base64
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    config, set_version, set_model,
    add_version_argument, add_model_argument, add_turn_argument,
    add_variant_argument, set_variant, variant_filename, get_variant_suffix, data_subdir,
    get_model, DIMENSION_CATEGORIES, CATEGORY_COLORS,
)


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate concept-conversation alignment report."
    )
    add_version_argument(parser)
    add_model_argument(parser)
    add_turn_argument(parser)
    add_variant_argument(parser)
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_stats(approach_dir):
    """Load stats.csv from an approach directory's data/ subfolder."""
    path = os.path.join(approach_dir, "data", variant_filename("stats", ".csv"))
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except pd.errors.EmptyDataError:
        return None


def load_prompt_stats(approach_dir):
    """Load prompt_stats.csv (approach D only)."""
    path = os.path.join(approach_dir, "data", variant_filename("prompt_stats", ".csv"))
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except pd.errors.EmptyDataError:
        return None


def load_cross_summary(root_dir):
    """Load cross_approach_summary.csv."""
    path = os.path.join(root_dir, "data", variant_filename("cross_approach_summary", ".csv"))
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def embed_png(path):
    """Read a PNG and return base64-encoded data URI for HTML embedding."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


CATEGORY_OVERRIDES = {
    "30_granite_sandstone": "Orthogonal Ctrl",
    "31_squares_triangles": "Orthogonal Ctrl",
}

def get_dim_category(dim_name):
    """Map dimension name to its category."""
    if dim_name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[dim_name]
    try:
        dim_id = int(dim_name.split("_")[0])
    except (ValueError, IndexError):
        return "Unknown"
    for cat, ids in DIMENSION_CATEGORIES.items():
        if dim_id in ids:
            return cat
    return "Unknown"


def sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ============================================================
# HTML GENERATION
# ============================================================

def generate_html(version, turn, root_dir, stats_a, stats_c, stats_d,
                  prompt_stats_d, cross_summary):
    """Build complete HTML report."""
    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html><head>")
    lines.append(f"<title>Concept-Conversation Alignment: {version} turn {turn}</title>")
    lines.append("<style>")
    lines.append("""
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1100px;
               margin: 0 auto; padding: 20px; color: #333; }
        h1 { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 8px; }
        h2 { color: #1976D2; margin-top: 30px; }
        h3 { color: #1E88E5; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }
        th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
        th { background: #E3F2FD; font-weight: 600; }
        td:first-child, th:first-child { text-align: left; }
        tr:nth-child(even) { background: #FAFAFA; }
        .sig { color: #1565C0; font-weight: bold; }
        .sig-fdr { color: #0D47A1; font-weight: bold; }
        .ns { color: #999; }
        .positive { color: #2E7D32; }
        .negative { color: #C62828; }
        .cat-Mental { background: #E3F2FD; }
        .cat-Physical { background: #E8F5E9; }
        .cat-Pragmatic { background: #FFF3E0; }
        .cat-Baseline { background: #F5F5F5; }
        .cat-Bio-Ctrl { background: #EFEBE9; }
        .cat-Shapes { background: #FCE4EC; }
        .cat-SysPrompt { background: #E0F7FA; }
        .cat-Orthogonal-Ctrl { background: #F3E5F5; }
        img { max-width: 100%; margin: 10px 0; }
        .summary-box { background: #F5F5F5; border-left: 4px solid #1565C0;
                       padding: 12px 16px; margin: 15px 0; }
        .legend { font-size: 12px; color: #666; margin: 5px 0; }
    """)
    lines.append("</style></head><body>")

    # Title and intro
    lines.append(f"<h1>Concept-Conversation Alignment</h1>")
    lines.append(f"<p><strong>Version:</strong> {version} &nbsp; "
                 f"<strong>Turn:</strong> {turn} &nbsp; "
                 f"<strong>Model:</strong> llama2_13b_chat</p>")

    lines.append("""
    <h2>Background & Motivation</h2>

    <p>Our earlier alignment analysis (Phase 2a) compared standalone concept vectors to
    <strong>probe weights</strong> — a learned contrastive representation that separates
    human from AI conversations. While informative, the interpretation is indirect:
    "this concept aligns with the direction a classifier uses to distinguish human from AI."</p>

    <p>This analysis asks a more direct question: <strong>"Is this concept more present/active
    in human conversations than AI conversations?"</strong></p>

    <h3>Method</h3>
    <p>We extract the model's internal activations (hidden states at the last token, across all
    41 layers) for each of ~1000 real conversations from Exp 1. We also have standalone concept
    activation vectors — extracted by prompting the model with 40 entity-neutral prompts per
    concept (e.g., "Describe what emotional experience is like" for the <em>emotions</em>
    dimension). For each concept, we compute cosine similarity between the concept vector and
    every conversation's activation vector, then test whether human conversations have higher
    alignment than AI conversations.</p>

    <p>Cosine similarity is computed per layer (layers 6–40, excluding early layers dominated by
    positional/format information), then averaged across layers to produce one alignment score
    per conversation per concept. Statistical testing uses independent t-tests with
    Benjamini-Hochberg FDR correction across concepts, plus bootstrap 95% CIs on the
    mean difference.</p>

    <h3>Three Approaches</h3>
    <p>A concern with this method is that the 40 prompts per concept may encode linguistic
    features beyond the target concept (shared syntax, vocabulary, or style). Three
    sub-approaches address this at different levels:</p>

    <ul>
        <li><strong>Approach A (Full prompt-set mean):</strong> Average all 40 prompt activations
        into a single concept vector, then compute alignment. This is the simplest baseline.
        If all concepts show similar effects, it may indicate a shared linguistic component
        rather than concept-specific signal.</li>

        <li><strong>Approach C (Concept-contrastive):</strong> For each concept, subtract the
        mean of all <em>other</em> concepts' vectors. This isolates what is <em>unique</em>
        to each concept beyond the shared structure. If approach A shows uniform effects but
        approach C shows selective effects, the shared component was driving A, and C reveals
        genuine concept-specific alignment.</li>

        <li><strong>Approach D (Per-prompt alignment):</strong> Instead of averaging prompts
        before computing cosine, compute alignment separately for each of the 40 prompts, then
        average in cosine space. This also reports per-prompt t-test results, identifying which
        specific prompts drive (or fail to drive) the human-vs-AI difference — useful for
        diagnosing whether effects come from a few outlier prompts or are consistent across
        the prompt set.</li>
    </ul>

    <h3>Interpreting the Results</h3>
    <ul>
        <li><strong>If A and D agree but C diverges:</strong> The shared linguistic structure
        across concept prompts dominates approaches A/D. Approach C strips this out, revealing
        which concepts have genuinely unique alignment above and beyond shared features.</li>

        <li><strong>If all three agree:</strong> The concept-specific signal is robust to
        different analysis strategies — strong evidence for genuine concept-level alignment.</li>

        <li><strong>Controls check:</strong> Negative control concepts (shapes, biological)
        should show weaker effects than mental concepts (phenomenology, emotions, agency).
        If controls show equally strong effects, linguistic confounds may dominate.</li>
    </ul>
    """)


    # Overview / Cross-approach comparison
    if cross_summary is not None:
        lines.append("<h2>Cross-Approach Comparison</h2>")
        lines.append("<p>Side-by-side comparison of all approaches. "
                     "Significance stars use FDR-corrected p-values.</p>")
        lines.append(_cross_comparison_table(cross_summary))

    # Per-approach sections
    for label, stats_df, approach_dir in [
        ("A", stats_a, os.path.join(root_dir, "approach_a")),
        ("C", stats_c, os.path.join(root_dir, "approach_c")),
        ("D", stats_d, os.path.join(root_dir, "approach_d")),
    ]:
        if stats_df is None:
            continue
        lines.append(f"<h2>Approach {label}</h2>")
        lines.append(_approach_description(label))
        lines.append(_stats_table(stats_df, label))

        # Embed figure
        fig_path = os.path.join(approach_dir, f"figures{get_variant_suffix()}", "alignment_by_concept.png")
        img_data = embed_png(fig_path)
        if img_data:
            lines.append(f'<img src="{img_data}" alt="Approach {label} alignment">')

    # Prompt-level analysis (approach D)
    if prompt_stats_d is not None and len(prompt_stats_d) > 0:
        lines.append("<h2>Prompt-Level Analysis (Approach D)</h2>")
        lines.append("<p>Per-prompt t-test results. Identifies which of the 40 prompts "
                     "per concept drive the human-vs-AI alignment difference.</p>")
        lines.append(_prompt_analysis_section(prompt_stats_d))

    # Category summary
    if stats_a is not None:
        lines.append("<h2>Category Summary</h2>")
        lines.append(_category_summary(stats_a, "A"))

    # Controls check
    if stats_a is not None:
        lines.append("<h2>Controls Check</h2>")
        lines.append(_controls_check(stats_a))

    lines.append("</body></html>")
    return "\n".join(lines)


def _approach_description(label):
    """Return a detailed description of each approach."""
    descriptions = {
        "A": (
            "<p><strong>Full prompt-set mean:</strong> For each concept, we average the "
            "activations from all 40 prompts into a single concept vector per layer. We then "
            "compute cosine similarity between this mean concept vector and each conversation's "
            "activation at each layer (6–40), averaging across layers to get one alignment "
            "score per conversation.</p>"
            "<p><em>What this tells us:</em> Whether conversations with human partners are, "
            "on average, more similar to the concept's activation pattern than conversations "
            "with AI partners. This is the most straightforward test. However, because "
            "the mean vector includes any linguistic features shared across all 40 prompts "
            "(not just the target concept), a significant result could reflect either genuine "
            "concept presence or shared prompt-level confounds.</p>"
            "<p><em>What to look for:</em> If nearly all concepts show the same effect size, "
            "that's a red flag — it suggests a shared component is driving results rather "
            "than concept-specific signal. Compare to Approach C.</p>"
        ),
        "C": (
            "<p><strong>Concept-contrastive:</strong> For each concept, we subtract the mean "
            "of all <em>other</em> concepts' vectors to produce a contrastive direction. "
            "This vector is then unit-normalized per layer and used for alignment, following "
            "the same cosine → layer-average → t-test pipeline as Approach A.</p>"
            "<p><em>What this tells us:</em> Whether each concept has a <em>unique</em> "
            "relationship with human vs AI conversations, above and beyond what's shared "
            "across all concept prompts. This is the key control for linguistic confounds. "
            "If Approach A shows uniform effects across all concepts but Approach C shows "
            "selective effects, the uniform signal in A was due to shared prompt structure, "
            "and C reveals which concepts genuinely differ.</p>"
            "<p><em>What to look for:</em> Concepts that are significant here have alignment "
            "that cannot be explained by shared features across concept prompts. "
            "The sign matters: positive = uniquely more present in human conversations; "
            "negative = uniquely more present in AI conversations.</p>"
        ),
        "D": (
            "<p><strong>Per-prompt alignment:</strong> Instead of averaging prompts in "
            "activation space (as in A), we compute alignment separately for each of the 40 "
            "prompts, then average the cosine similarities. This averaging in cosine space "
            "(vs. activation space in A) gives slightly different results.</p>"
            "<p><em>What this tells us:</em> Concept-level results should be similar to "
            "Approach A. The key added value is the <strong>per-prompt breakdown</strong>: "
            "we run a t-test for each individual prompt, showing which prompts contribute "
            "most (and least) to the overall effect. This helps diagnose whether effects "
            "are driven by a few outlier prompts or are consistent across the set, and can "
            "identify poorly-performing prompts for future prompt design.</p>"
            "<p><em>What to look for:</em> Concepts where most prompts individually reach "
            "significance have robust, prompt-independent effects. Concepts where only a few "
            "prompts drive the result may be more fragile or confound-dependent.</p>"
        ),
    }
    return descriptions.get(label, "")


def _stats_table(stats_df, approach_label):
    """Generate HTML table for one approach's stats."""
    rows = []
    rows.append("<table>")
    rows.append("<tr><th>Dimension</th><th>Category</th><th>H mean</th><th>A mean</th>"
                "<th>H-A diff</th><th>95% CI</th><th>t</th><th>p</th>"
                "<th>p<sub>FDR</sub></th><th>d</th><th>N (H/A)</th></tr>")

    for _, r in stats_df.sort_values("diff", ascending=False).iterrows():
        cat = get_dim_category(r["dimension"])
        cat_class = f"cat-{cat.replace(' ', '-')}"

        diff_class = "positive" if r["diff"] > 0 else "negative"
        stars = sig_stars(r["p"])
        fdr_stars = sig_stars(r["p_fdr"])

        p_class = "sig-fdr" if r["p_fdr"] < 0.05 else ("sig" if r["p"] < 0.05 else "ns")

        ci_str = f"[{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]"

        rows.append(
            f'<tr class="{cat_class}">'
            f'<td>{r["dimension"]}</td>'
            f'<td>{cat}</td>'
            f'<td>{r["human_mean"]:.5f}</td>'
            f'<td>{r["ai_mean"]:.5f}</td>'
            f'<td class="{diff_class}">{r["diff"]:+.5f}</td>'
            f'<td>{ci_str}</td>'
            f'<td>{r["t"]:.2f}</td>'
            f'<td class="{p_class}">{r["p"]:.4f} {stars}</td>'
            f'<td class="{p_class}">{r["p_fdr"]:.4f} {fdr_stars}</td>'
            f'<td>{r["cohen_d"]:+.3f}</td>'
            f'<td>{int(r["n_human"])}/{int(r["n_ai"])}</td>'
            f'</tr>'
        )

    rows.append("</table>")
    n_sig = (stats_df["p"] < 0.05).sum()
    n_fdr = (stats_df["p_fdr"] < 0.05).sum()
    rows.append(f'<p class="legend">{n_sig}/{len(stats_df)} significant (p&lt;.05), '
                f'{n_fdr}/{len(stats_df)} after FDR correction</p>')
    return "\n".join(rows)


def _cross_comparison_table(cross_df):
    """Generate side-by-side comparison table across approaches."""
    approaches = sorted(cross_df["approach"].unique())
    dims = sorted(cross_df["dimension"].unique())

    rows = []
    rows.append("<table>")

    header = "<tr><th>Dimension</th><th>Category</th>"
    for ap in approaches:
        header += f"<th>{ap} diff</th><th>{ap} p<sub>FDR</sub></th><th>{ap} d</th>"
    header += "</tr>"
    rows.append(header)

    for dim in dims:
        cat = get_dim_category(dim)
        cat_class = f"cat-{cat.replace(' ', '-')}"
        row = f'<tr class="{cat_class}"><td>{dim}</td><td>{cat}</td>'

        for ap in approaches:
            sub = cross_df[(cross_df["dimension"] == dim) & (cross_df["approach"] == ap)]
            if len(sub) > 0:
                r = sub.iloc[0]
                diff_class = "positive" if r["diff"] > 0 else "negative"
                stars = sig_stars(r["p_fdr"])
                p_class = "sig-fdr" if r["p_fdr"] < 0.05 else "ns"
                row += (f'<td class="{diff_class}">{r["diff"]:+.5f}</td>'
                        f'<td class="{p_class}">{r["p_fdr"]:.4f} {stars}</td>'
                        f'<td>{r["cohen_d"]:+.3f}</td>')
            else:
                row += "<td>—</td><td>—</td><td>—</td>"
        row += "</tr>"
        rows.append(row)

    rows.append("</table>")
    return "\n".join(rows)


def _prompt_analysis_section(prompt_df):
    """Summarize per-prompt results for each dimension."""
    rows = []

    for dim_name in sorted(prompt_df["dimension"].unique()):
        sub = prompt_df[prompt_df["dimension"] == dim_name].copy()
        n_total = len(sub)
        n_sig = (sub["p"] < 0.05).sum()
        n_pos = ((sub["diff"] > 0) & (sub["p"] < 0.05)).sum()
        n_neg = ((sub["diff"] < 0) & (sub["p"] < 0.05)).sum()
        mean_diff = sub["diff"].mean()

        rows.append(f"<h3>{dim_name}</h3>")
        rows.append(f'<div class="summary-box">')
        rows.append(f"<strong>{n_sig}/{n_total}</strong> prompts significant (p&lt;.05): "
                    f"<span class='positive'>{n_pos} H&gt;A</span>, "
                    f"<span class='negative'>{n_neg} A&gt;H</span>. "
                    f"Mean diff: {mean_diff:+.5f}</div>")

        # Top 5 best and worst prompts
        sub_sorted = sub.sort_values("diff", ascending=False)
        best = sub_sorted.head(5)
        worst = sub_sorted.tail(5)

        rows.append("<table>")
        rows.append("<tr><th>Prompt</th><th>H mean</th><th>A mean</th>"
                    "<th>H-A diff</th><th>t</th><th>p</th><th>Role</th></tr>")

        for _, r in best.iterrows():
            stars = sig_stars(r["p"])
            p_class = "sig" if r["p"] < 0.05 else "ns"
            rows.append(
                f'<tr><td>prompt_{int(r["prompt_idx"])}</td>'
                f'<td>{r["human_mean"]:.5f}</td><td>{r["ai_mean"]:.5f}</td>'
                f'<td class="positive">{r["diff"]:+.5f}</td>'
                f'<td>{r["t"]:.2f}</td>'
                f'<td class="{p_class}">{r["p"]:.4f} {stars}</td>'
                f'<td>Best (H&gt;A)</td></tr>'
            )
        for _, r in worst.iterrows():
            stars = sig_stars(r["p"])
            p_class = "sig" if r["p"] < 0.05 else "ns"
            rows.append(
                f'<tr><td>prompt_{int(r["prompt_idx"])}</td>'
                f'<td>{r["human_mean"]:.5f}</td><td>{r["ai_mean"]:.5f}</td>'
                f'<td class="negative">{r["diff"]:+.5f}</td>'
                f'<td>{r["t"]:.2f}</td>'
                f'<td class="{p_class}">{r["p"]:.4f} {stars}</td>'
                f'<td>Worst (A&gt;H)</td></tr>'
            )

        rows.append("</table>")

    return "\n".join(rows)


def _category_summary(stats_df, approach_label):
    """Summarize results by dimension category."""
    rows = []
    stats_df = stats_df.copy()
    stats_df["category"] = stats_df["dimension"].apply(get_dim_category)

    rows.append("<table>")
    rows.append("<tr><th>Category</th><th>N dims</th><th>Mean H-A diff</th>"
                "<th>Mean |d|</th><th>Sig (p&lt;.05)</th>"
                "<th>Sig (FDR)</th></tr>")

    for cat in ["Mental", "Physical", "Pragmatic", "Baseline", "Bio Ctrl", "Shapes", "Orthogonal Ctrl", "SysPrompt"]:
        sub = stats_df[stats_df["category"] == cat]
        if len(sub) == 0:
            continue
        cat_class = f"cat-{cat.replace(' ', '-')}"
        n_sig = (sub["p"] < 0.05).sum()
        n_fdr = (sub["p_fdr"] < 0.05).sum()
        mean_d = sub["cohen_d"].abs().mean()
        mean_diff = sub["diff"].mean()
        diff_class = "positive" if mean_diff > 0 else "negative"

        rows.append(
            f'<tr class="{cat_class}">'
            f'<td>{cat}</td><td>{len(sub)}</td>'
            f'<td class="{diff_class}">{mean_diff:+.5f}</td>'
            f'<td>{mean_d:.3f}</td>'
            f'<td>{n_sig}/{len(sub)}</td>'
            f'<td>{n_fdr}/{len(sub)}</td></tr>'
        )

    rows.append("</table>")
    return "\n".join(rows)


def _controls_check(stats_df):
    """Check whether control concepts show weaker effects than mental concepts."""
    stats_df = stats_df.copy()
    stats_df["category"] = stats_df["dimension"].apply(get_dim_category)

    mental = stats_df[stats_df["category"] == "Mental"]
    controls = stats_df[stats_df["category"].isin(["Shapes", "Bio Ctrl", "Orthogonal Ctrl"])]

    rows = []
    rows.append('<div class="summary-box">')

    if len(mental) > 0 and len(controls) > 0:
        mental_abs_d = mental["cohen_d"].abs().mean()
        control_abs_d = controls["cohen_d"].abs().mean()

        if mental_abs_d > control_abs_d:
            rows.append(f"<strong>PASS:</strong> Mental concepts show larger effects "
                        f"(mean |d| = {mental_abs_d:.3f}) than controls "
                        f"(mean |d| = {control_abs_d:.3f}).")
        else:
            rows.append(f"<strong>INCONCLUSIVE:</strong> Control concepts show similar or larger "
                        f"effects (|d| = {control_abs_d:.3f}) compared to mental concepts "
                        f"(|d| = {mental_abs_d:.3f}). May indicate linguistic confounds.")
    else:
        rows.append("Insufficient data for controls check.")

    rows.append("</div>")

    # Detail table
    for label, sub in [("Mental", mental), ("Controls (Shapes, Bio Ctrl, Orthogonal Ctrl)", controls)]:
        if len(sub) == 0:
            continue
        rows.append(f"<h3>{label}</h3>")
        rows.append("<table><tr><th>Dimension</th><th>H-A diff</th><th>d</th>"
                    "<th>p</th><th>p<sub>FDR</sub></th></tr>")
        for _, r in sub.sort_values("diff", ascending=False).iterrows():
            stars = sig_stars(r["p_fdr"])
            rows.append(
                f'<tr><td>{r["dimension"]}</td>'
                f'<td>{r["diff"]:+.5f}</td>'
                f'<td>{r["cohen_d"]:+.3f}</td>'
                f'<td>{r["p"]:.4f}</td>'
                f'<td>{r["p_fdr"]:.4f} {stars}</td></tr>'
            )
        rows.append("</table>")

    return "\n".join(rows)


# ============================================================
# MARKDOWN GENERATION
# ============================================================

def generate_markdown(version, turn, stats_a, stats_c, stats_d,
                      prompt_stats_d, cross_summary):
    """Build Markdown report."""
    lines = []
    lines.append(f"# Concept-Conversation Alignment: {version}, turn {turn}")
    lines.append("")
    lines.append(f"**Model:** llama2_13b_chat")
    lines.append("")

    # Cross-approach summary
    if cross_summary is not None:
        lines.append("## Cross-Approach Comparison")
        lines.append("")
        approaches = sorted(cross_summary["approach"].unique())
        header = "| Dimension | Category |"
        sep = "|---|---|"
        for ap in approaches:
            header += f" {ap} diff | {ap} p_FDR | {ap} d |"
            sep += "---|---|---|"
        lines.append(header)
        lines.append(sep)

        for dim in sorted(cross_summary["dimension"].unique()):
            cat = get_dim_category(dim)
            row = f"| {dim} | {cat} |"
            for ap in approaches:
                sub = cross_summary[(cross_summary["dimension"] == dim) &
                                    (cross_summary["approach"] == ap)]
                if len(sub) > 0:
                    r = sub.iloc[0]
                    stars = sig_stars(r["p_fdr"])
                    row += f" {r['diff']:+.5f} | {r['p_fdr']:.4f} {stars} | {r['cohen_d']:+.3f} |"
                else:
                    row += " — | — | — |"
            lines.append(row)
        lines.append("")

    # Per-approach tables
    for label, stats_df in [("A", stats_a), ("C", stats_c), ("D", stats_d)]:
        if stats_df is None:
            continue
        lines.append(f"## Approach {label}")
        lines.append("")
        lines.append("| Dimension | Category | H-A diff | p | p_FDR | d |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in stats_df.sort_values("diff", ascending=False).iterrows():
            cat = get_dim_category(r["dimension"])
            stars = sig_stars(r["p_fdr"])
            lines.append(
                f"| {r['dimension']} | {cat} | {r['diff']:+.5f} | "
                f"{r['p']:.4f} | {r['p_fdr']:.4f} {stars} | {r['cohen_d']:+.3f} |"
            )
        n_sig = (stats_df["p"] < 0.05).sum()
        n_fdr = (stats_df["p_fdr"] < 0.05).sum()
        lines.append(f"\n{n_sig}/{len(stats_df)} significant (p<.05), "
                     f"{n_fdr}/{len(stats_df)} after FDR")
        lines.append("")

    # Prompt-level summary (D)
    if prompt_stats_d is not None and len(prompt_stats_d) > 0:
        lines.append("## Prompt-Level Analysis (Approach D)")
        lines.append("")
        for dim_name in sorted(prompt_stats_d["dimension"].unique()):
            sub = prompt_stats_d[prompt_stats_d["dimension"] == dim_name]
            n_sig = (sub["p"] < 0.05).sum()
            lines.append(f"- **{dim_name}**: {n_sig}/{len(sub)} prompts significant")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.variant:
        set_variant(args.variant)
    set_model(args.model)
    set_version(args.version, turn=args.turn)
    model_name = get_model()

    root_dir = os.path.join(
        str(config.RESULTS.root), model_name, args.version,
        "concept_conversation", f"turn_{args.turn}",
    )

    if not os.path.isdir(root_dir):
        print(f"[ERROR] Results directory not found: {root_dir}")
        print("Run 9b_concept_conversation_alignment.py first.")
        sys.exit(1)

    # Load data
    stats_a = load_stats(os.path.join(root_dir, "approach_a"))
    stats_c = load_stats(os.path.join(root_dir, "approach_c"))
    stats_d = load_stats(os.path.join(root_dir, "approach_d"))
    prompt_stats_d = load_prompt_stats(os.path.join(root_dir, "approach_d"))
    cross_summary = load_cross_summary(root_dir)

    if all(x is None for x in [stats_a, stats_c, stats_d]):
        print("[ERROR] No stats files found in any approach directory.")
        sys.exit(1)

    print(f"Loaded data from {root_dir}")
    for label, df in [("A", stats_a), ("C", stats_c), ("D", stats_d)]:
        if df is not None:
            print(f"  Approach {label}: {len(df)} dimensions")

    # Generate HTML
    html = generate_html(
        args.version, args.turn, root_dir,
        stats_a, stats_c, stats_d, prompt_stats_d, cross_summary,
    )
    html_path = os.path.join(root_dir, variant_filename("concept_conversation_report", ".html"))
    with open(html_path, "w") as f:
        f.write(html)
    print(f"Saved: {html_path}")

    # Generate Markdown
    md = generate_markdown(
        args.version, args.turn,
        stats_a, stats_c, stats_d, prompt_stats_d, cross_summary,
    )
    md_path = os.path.join(root_dir, variant_filename("concept_conversation_report", ".md"))
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved: {md_path}")

    print("\nReport generation complete.")


if __name__ == "__main__":
    main()
