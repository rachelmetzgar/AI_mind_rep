#!/usr/bin/env python3
"""
Experiment 4: Data Status Report

Scans the filesystem to check which expected data files exist for each
model x branch combination. Generates an HTML report with a color-coded
status table and per-branch detail sections.

Output: results/comparisons/status_report.html

Usage:
    python comparisons/2_status_report_generator.py

Env: llama2_env (or any — no heavy deps)
Rachel C. Metzgar · Mar 2026
"""

import sys
from pathlib import Path

# ── Local imports ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ROOT_DIR, COMPARISONS_DIR, VALID_MODELS, VALID_BRANCHES,
    MODELS, ensure_dir, set_model, data_dir, results_dir,
)
from utils.report_utils import (
    REPORT_CSS, build_cross_model_header, build_html_footer,
    build_toc, MODEL_COLORS, MODEL_LABELS, ALL_MODELS,
)


# ============================================================================
# EXPECTED DATA FILES
# ============================================================================

# Each entry: (relative_path_from_model_root, filename, applicable_models)
# applicable_models: "all", "base_only", "chat_only", or a tuple of model keys

BASE_MODELS = ("llama2_13b_base", "llama3_8b_base")
ALL = "all"
BASE_ONLY = "base_only"

EXPECTED_FILES = {
    "gray_replication": {
        "label": "Gray Replication (Pairwise Behavioral)",
        "description": (
            "Replication of Gray, Gray, &amp; Wegner (2007) pairwise comparison "
            "paradigm with 13 entities and 18 mental capacity items."
        ),
        "files": [
            ("gray_replication/behavior/with_self/data", "pairwise_pca_results.npz", ALL),
            ("gray_replication/behavior/with_self/data", "pairwise_human_correlations.json", ALL),
            ("gray_replication/behavior/with_self/data", "pairwise_consistency_stats.json", ALL),
            ("gray_replication/behavior/with_self/data", "individual_pca_results.npz", ALL),
            ("gray_replication/behavior/with_self/data", "individual_rating_matrix.npz", ALL),
        ],
    },
    "gray_simple": {
        "label": "Gray Simple (Neural Internals)",
        "description": (
            "Simple activation extraction (&ldquo;Think about {entity}&rdquo;) "
            "for the 13 Gray entities. RSA and neural PCA analyses."
        ),
        "files": [
            ("gray_simple/internals/with_self/data", "rsa_results.json", ALL),
            ("gray_simple/internals/with_self/data", "rdm_cosine_per_layer.npz", ALL),
            ("gray_simple/internals/with_self/data", "neural_pca_results.npz", ALL),
            ("gray_simple/internals/with_self/data", "neural_pca_analysis.json", ALL),
        ],
    },
    "human_ai_adaptation": {
        "label": "Human/AI Adaptation (Behavioral)",
        "description": (
            "30 AI/human characters rated on Gray et al. mental capacities "
            "via pairwise comparisons."
        ),
        "files": [
            ("human_ai_adaptation/behavior/data", "pairwise_pca_results.npz", ALL),
            ("human_ai_adaptation/behavior/data", "pairwise_categorical_analysis.json", ALL),
            ("human_ai_adaptation/behavior/data", "pairwise_character_means.npz", ALL),
        ],
    },
    "expanded_mental_concepts": {
        "label": "Expanded Mental Concepts",
        "description": (
            "28 AI/human characters probed with Exp 3 concept dimensions. "
            "Behavioral (PCA), neural (RSA, concept RSA, alignment)."
        ),
        "files": [
            # Behavioral PCA
            ("expanded_mental_concepts/behavior/pca/data", "pairwise_pca_results.npz", ALL),
            ("expanded_mental_concepts/behavior/pca/data", "pairwise_categorical_analysis.json", ALL),
            # Internals: RSA
            ("expanded_mental_concepts/internals/rsa/data", "rsa_results.json", ALL),
            ("expanded_mental_concepts/internals/rsa/data", "rdm_cosine_per_layer.npz", ALL),
            # Internals: concept RSA
            ("expanded_mental_concepts/internals/concept_rsa/data", "cross_concept_rsa_summary.json", ALL),
            # Internals: contrast alignment
            ("expanded_mental_concepts/internals/contrast_alignment/data", "alignment_results.json", ALL),
            # Internals: standalone alignment
            ("expanded_mental_concepts/internals/standalone_alignment/data", "alignment_results.json", ALL),
        ],
    },
}


# ============================================================================
# CHECK LOGIC
# ============================================================================

def is_model_applicable(model, applicability):
    """Return True if this file is expected for the given model."""
    if applicability == ALL:
        return True
    if applicability == BASE_ONLY:
        return model in BASE_MODELS
    if isinstance(applicability, tuple):
        return model in applicability
    return False


def check_file(model, rel_dir, filename):
    """Check whether a specific data file exists for the given model."""
    path = ROOT_DIR / "results" / model / rel_dir / filename
    return path.exists()


def scan_all():
    """Scan all expected files for all models.

    Returns:
        results: dict[branch][model] = list of (filename, rel_dir, status)
            where status is "found", "missing", or "n/a"
        summary: dict with total/found/missing/na counts
    """
    results = {}
    total, found, missing, na = 0, 0, 0, 0

    for branch, info in EXPECTED_FILES.items():
        results[branch] = {}
        for model in ALL_MODELS:
            model_results = []
            for rel_dir, filename, applicability in info["files"]:
                total += 1
                if not is_model_applicable(model, applicability):
                    model_results.append((filename, rel_dir, "n/a"))
                    na += 1
                elif check_file(model, rel_dir, filename):
                    model_results.append((filename, rel_dir, "found"))
                    found += 1
                else:
                    model_results.append((filename, rel_dir, "missing"))
                    missing += 1
            results[branch] = {**results[branch], model: model_results}

    summary = {"total": total, "found": found, "missing": missing, "na": na}
    return results, summary


# ============================================================================
# HTML GENERATION
# ============================================================================

STATUS_CSS = """
    .status-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    .status-table th, .status-table td {
        padding: 8px 12px; border: 1px solid #ddd; text-align: center;
        font-size: 0.9rem;
    }
    .status-table th { background: #f0f0f0; font-weight: 600; }
    .status-table td:first-child { text-align: left; font-family: monospace;
                                    font-size: 0.85rem; }
    .status-table th:first-child { text-align: left; }
    .cell-found { background: #e8f5e9; color: #2E7D32; font-weight: bold; }
    .cell-missing { background: #ffebee; color: #C62828; font-weight: bold; }
    .cell-na { background: #f5f5f5; color: #999; }
    .summary-box { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }
    .summary-card { padding: 15px 25px; border-radius: 8px; text-align: center;
                    min-width: 120px; }
    .summary-card .count { font-size: 2rem; font-weight: bold; display: block; }
    .summary-card .label { font-size: 0.85rem; color: #555; }
    .card-found { background: #e8f5e9; }
    .card-found .count { color: #2E7D32; }
    .card-missing { background: #ffebee; }
    .card-missing .count { color: #C62828; }
    .card-na { background: #f5f5f5; }
    .card-na .count { color: #999; }
    .card-total { background: #e3f2fd; }
    .card-total .count { color: #1565C0; }
    .branch-header { margin-top: 40px; padding: 10px 15px; background: #f8f9ff;
                     border-left: 4px solid #2196F3; border-radius: 0 6px 6px 0; }
    .branch-desc { color: #555; font-size: 0.9rem; margin: 5px 0 15px 0; }
    .path-hint { font-size: 0.8rem; color: #888; font-family: monospace; }
"""


def cell_html(status):
    """Return a table cell for a given status."""
    if status == "found":
        return '<td class="cell-found">&#10003;</td>'
    elif status == "missing":
        return '<td class="cell-missing">&#10007;</td>'
    else:
        return '<td class="cell-na">N/A</td>'


def build_summary_section(summary):
    """Build the summary cards at the top of the report."""
    applicable = summary["total"] - summary["na"]
    if applicable > 0:
        pct = 100.0 * summary["found"] / applicable
    else:
        pct = 0.0

    html = '<h2 id="summary">Summary</h2>\n'
    html += '<div class="summary-box">\n'
    html += (
        f'<div class="summary-card card-total">'
        f'<span class="count">{applicable}</span>'
        f'<span class="label">Expected Files</span></div>\n'
    )
    html += (
        f'<div class="summary-card card-found">'
        f'<span class="count">{summary["found"]}</span>'
        f'<span class="label">Found ({pct:.0f}%)</span></div>\n'
    )
    html += (
        f'<div class="summary-card card-missing">'
        f'<span class="count">{summary["missing"]}</span>'
        f'<span class="label">Missing</span></div>\n'
    )
    html += (
        f'<div class="summary-card card-na">'
        f'<span class="count">{summary["na"]}</span>'
        f'<span class="label">N/A</span></div>\n'
    )
    html += '</div>\n'
    return html


def build_overview_table(results):
    """Build a branch-level overview table: branches x models, with counts."""
    html = '<h2 id="overview">Overview by Branch</h2>\n'
    html += '<table class="status-table">\n'
    html += '<tr><th>Branch</th>'
    for model in ALL_MODELS:
        html += f'<th>{MODEL_LABELS[model]}</th>'
    html += '</tr>\n'

    for branch, info in EXPECTED_FILES.items():
        html += f'<tr><td><a href="#{branch}">{info["label"]}</a></td>'
        for model in ALL_MODELS:
            model_files = results[branch][model]
            n_found = sum(1 for _, _, s in model_files if s == "found")
            n_applicable = sum(1 for _, _, s in model_files if s != "n/a")
            if n_applicable == 0:
                html += '<td class="cell-na">N/A</td>'
            elif n_found == n_applicable:
                html += f'<td class="cell-found">{n_found}/{n_applicable}</td>'
            elif n_found == 0:
                html += f'<td class="cell-missing">0/{n_applicable}</td>'
            else:
                html += (
                    f'<td style="background:#fff3e0; color:#E65100; '
                    f'font-weight:bold;">{n_found}/{n_applicable}</td>'
                )
        html += '</tr>\n'

    html += '</table>\n'
    return html


def build_branch_detail(branch, info, results):
    """Build a detailed per-file table for a single branch."""
    html = f'<div class="branch-header"><h2 id="{branch}">{info["label"]}</h2></div>\n'
    html += f'<p class="branch-desc">{info["description"]}</p>\n'

    html += '<table class="status-table">\n'
    html += '<tr><th>File</th>'
    for model in ALL_MODELS:
        html += f'<th>{MODEL_LABELS[model]}</th>'
    html += '</tr>\n'

    # Group files by directory for readability
    seen_dirs = set()
    for idx, (rel_dir, filename, applicability) in enumerate(info["files"]):
        if rel_dir not in seen_dirs:
            seen_dirs.add(rel_dir)
            n_cols = len(ALL_MODELS) + 1
            html += (
                f'<tr><td colspan="{n_cols}" class="path-hint" '
                f'style="background:#fafafa; border-top:2px solid #ddd;">'
                f'results/{{model}}/{rel_dir}/</td></tr>\n'
            )

        html += f'<tr><td>{filename}</td>'
        for model in ALL_MODELS:
            status = results[branch][model][idx][2]
            html += cell_html(status)
        html += '</tr>\n'

    html += '</table>\n'
    return html


def generate_report(results, summary):
    """Generate the full HTML report."""
    out_path = ensure_dir(COMPARISONS_DIR) / "status_report.html"

    html = build_cross_model_header(
        "Experiment 4 — Data Status Report",
        css=STATUS_CSS,
    )

    # Table of contents
    sections = [
        {"id": "summary", "label": "Summary"},
        {"id": "overview", "label": "Overview by Branch"},
    ]
    for branch, info in EXPECTED_FILES.items():
        sections.append({"id": branch, "label": info["label"]})
    html += build_toc(sections)

    # Summary cards
    html += build_summary_section(summary)

    # Overview table
    html += build_overview_table(results)

    # Per-branch detail sections
    for branch, info in EXPECTED_FILES.items():
        html += build_branch_detail(branch, info, results)

    # Legend
    html += '<h2 id="legend">Legend</h2>\n'
    html += '<table class="status-table" style="width:auto;">\n'
    html += '<tr><td class="cell-found">&#10003;</td><td style="text-align:left;">File exists</td></tr>\n'
    html += '<tr><td class="cell-missing">&#10007;</td><td style="text-align:left;">File missing</td></tr>\n'
    html += '<tr><td class="cell-na">N/A</td><td style="text-align:left;">Not applicable for this model type</td></tr>\n'
    html += '</table>\n'

    html += build_html_footer()

    out_path.write_text(html)
    print(f"Status report written to: {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Scanning data files across all models and branches...")
    results, summary = scan_all()

    applicable = summary["total"] - summary["na"]
    print(f"  Total expected (applicable): {applicable}")
    print(f"  Found: {summary['found']}")
    print(f"  Missing: {summary['missing']}")
    print(f"  N/A: {summary['na']}")

    generate_report(results, summary)


if __name__ == "__main__":
    main()
