"""
Utility for saving HTML reports with an auto-generated Markdown companion.

Usage:
    from src.report_utils import save_report

    html = "<html>...</html>"
    save_report(html, Path("results/report.html"))
    # -> saves report.html and report.md

The Markdown version strips styling, converts tags to markdown syntax,
and replaces embedded images with placeholders pointing to the HTML.
"""

import re
from pathlib import Path


def _html_to_markdown(html: str) -> str:
    """Convert an HTML report string to readable Markdown."""
    md = html

    # Remove everything inside <head>...</head>
    md = re.sub(r"<head>.*?</head>", "", md, flags=re.DOTALL)

    # Remove <style>...</style> blocks
    md = re.sub(r"<style[^>]*>.*?</style>", "", md, flags=re.DOTALL)

    # Remove <script>...</script> blocks
    md = re.sub(r"<script[^>]*>.*?</script>", "", md, flags=re.DOTALL)

    # Replace embedded images with placeholder
    md = re.sub(
        r'<img[^>]*src="data:image/[^"]*"[^>]*alt="([^"]*)"[^>]*/?>',
        r"\n*[Figure: \1 — see HTML report]*\n",
        md,
    )
    md = re.sub(
        r'<img[^>]*src="data:image/[^"]*"[^>]*/?>',
        "\n*[Figure — see HTML report]*\n",
        md,
    )

    # Headers
    md = re.sub(r"<h1[^>]*>(.*?)</h1>", r"\n# \1\n", md, flags=re.DOTALL)
    md = re.sub(r"<h2[^>]*>(.*?)</h2>", r"\n## \1\n", md, flags=re.DOTALL)
    md = re.sub(r"<h3[^>]*>(.*?)</h3>", r"\n### \1\n", md, flags=re.DOTALL)
    md = re.sub(r"<h4[^>]*>(.*?)</h4>", r"\n#### \1\n", md, flags=re.DOTALL)

    # Bold and italic
    md = re.sub(r"<strong>(.*?)</strong>", r"**\1**", md)
    md = re.sub(r"<b>(.*?)</b>", r"**\1**", md)
    md = re.sub(r"<em>(.*?)</em>", r"*\1*", md)
    md = re.sub(r"<i>(.*?)</i>", r"*\1*", md)

    # Code
    md = re.sub(r"<code>(.*?)</code>", r"`\1`", md)

    # Line breaks
    md = re.sub(r"<br\s*/?>", "\n", md)
    md = re.sub(r"<hr[^>]*/?>", "\n---\n", md)

    # Lists
    md = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1", md, flags=re.DOTALL)

    # Convert simple tables to markdown tables
    md = _convert_tables(md)

    # Paragraphs and divs → newlines
    md = re.sub(r"<p[^>]*>", "\n", md)
    md = re.sub(r"</p>", "\n", md)
    md = re.sub(r"<div[^>]*>", "\n", md)
    md = re.sub(r"</div>", "\n", md)

    # Strip remaining HTML tags
    md = re.sub(r"<[^>]+>", "", md)

    # Decode common HTML entities
    entities = {
        "&mdash;": "—", "&ndash;": "–", "&hellip;": "...",
        "&rsquo;": "'", "&lsquo;": "'", "&rdquo;": '"', "&ldquo;": '"',
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&asymp;": "~", "&times;": "x", "&plusmn;": "+/-",
        "&nbsp;": " ", "&minus;": "-",
        "&#8212;": "—", "&#8211;": "–",
    }
    for entity, char in entities.items():
        md = md.replace(entity, char)
    # Numeric entities
    md = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))), md)

    # Clean up excessive whitespace
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n[ \t]+\n", "\n\n", md)

    return md.strip() + "\n"


def _convert_tables(html: str) -> str:
    """Convert HTML tables to markdown tables."""
    def _table_to_md(match):
        table_html = match.group(0)

        # Extract all rows
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
        if not rows:
            return table_html

        md_rows = []
        for row_html in rows:
            # Extract cells (th or td)
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.DOTALL)
            # Strip inner HTML from cells
            clean_cells = []
            for cell in cells:
                cell = re.sub(r"<[^>]+>", "", cell)
                cell = cell.strip().replace("|", "\\|")
                # Collapse whitespace
                cell = re.sub(r"\s+", " ", cell)
                clean_cells.append(cell)
            if clean_cells:
                md_rows.append("| " + " | ".join(clean_cells) + " |")

        if not md_rows:
            return ""

        # Insert separator after first row (header)
        if len(md_rows) >= 1:
            n_cols = md_rows[0].count("|") - 1
            separator = "| " + " | ".join(["---"] * n_cols) + " |"
            md_rows.insert(1, separator)

        return "\n" + "\n".join(md_rows) + "\n"

    return re.sub(r"<table[^>]*>.*?</table>", _table_to_md, html, flags=re.DOTALL)


def save_report(html: str, html_path, quiet: bool = False) -> Path:
    """Save an HTML report and auto-generate a Markdown companion.

    Parameters
    ----------
    html : str
        The full HTML content.
    html_path : str or Path
        Where to write the .html file. The .md file is saved alongside it
        with the same stem.
    quiet : bool
        If True, suppress print output.

    Returns
    -------
    Path
        The path to the saved .md file.
    """
    html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Save HTML
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Generate and save Markdown
    md_path = html_path.with_suffix(".md")
    md_content = _html_to_markdown(html)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    if not quiet:
        print(f"  Saved: {html_path}")
        print(f"  Saved: {md_path}")

    return md_path
