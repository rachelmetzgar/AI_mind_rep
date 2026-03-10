#!/usr/bin/env python3
"""
Shared HTML Report Utilities

Common CSS, figure encoding, table-of-contents builder, and HTML scaffolding
used by all Experiment 4 report generators.

Rachel C. Metzgar · Mar 2026
"""

import base64
import argparse
from io import BytesIO
from datetime import datetime


# ============================================================================
# SHARED CSS
# ============================================================================

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
# DATASET ARGUMENT
# ============================================================================

def add_dataset_argument(parser, default="full_dataset"):
    """Add a --dataset argument to an argparse parser.

    Controls whether the report is written into a full_dataset/ or
    reduced_dataset/ subfolder under the normal results directory.
    """
    parser.add_argument(
        "--dataset", type=str, default=default,
        choices=["full_dataset", "reduced_dataset"],
        help=f"Dataset subfolder for output (default: {default})"
    )
