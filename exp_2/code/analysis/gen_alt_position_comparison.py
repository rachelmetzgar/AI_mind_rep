#!/usr/bin/env python3
"""
Generate an HTML report comparing alternative token position probes across
versions and conversation turns.

Versions: labels, balanced_gpt, nonsense_codeword
Conditions: control_first (BOS), control_random, control_eos (first </s>),
            reading_irrelevant (weather suffix), + baselines
Turns: 1-5

Output: exp_2/results/comparisons/probe_training/alt_position_comparison.html (+.md)
"""

import pickle
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Configuration ────────────────────────────────────────────────────────────

BASE = '/mnt/cup/labs/graziano/rachel/mind_rep/exp_2'
OUTPUT_PATH = f'{BASE}/results/comparisons/probe_training/alt_position_comparison.html'

VERSIONS = ['labels', 'balanced_gpt', 'nonsense_codeword']
VERSION_LABELS = {
    'labels': 'Labels (Primary)',
    'balanced_gpt': 'Balanced GPT',
    'nonsense_codeword': 'Nonsense Codeword (Control)',
}
VERSION_COLORS = {
    'labels': '#1a3399',
    'balanced_gpt': '#cc2233',
    'nonsense_codeword': '#e8961a',
}

TURNS = [1, 2, 3, 4, 5]
N_LAYERS = 41

ALT_CONDITIONS = ['control_first', 'control_random', 'control_eos', 'reading_irrelevant']
ALT_LABELS = {
    'control_first': 'BOS Token (position 0)',
    'control_random': 'Random Mid-Sequence Token',
    'control_eos': 'First </s> (end of 1st exchange)',
    'reading_irrelevant': 'Irrelevant Suffix (weather)',
}
ALT_COLORS = {
    'control_first': '#e74c3c',
    'control_random': '#f39c12',
    'control_eos': '#27ae60',
    'reading_irrelevant': '#8e44ad',
}
ALT_MARKERS = {
    'control_first': 'D',
    'control_random': '^',
    'control_eos': 'o',
    'reading_irrelevant': 's',
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load all alternative probe data.

    Returns dict: data[version][turn][condition] = array(41,)
    Also loads baselines: data[version][turn]['baseline_reading'] and 'baseline_control'.
    """
    data = {}
    missing = []

    for version in VERSIONS:
        data[version] = {}
        for turn in TURNS:
            data[version][turn] = {}

            # Alternative probe directory
            if turn == 5:
                alt_dir = f'{BASE}/data/{version}/probe_checkpoints/alternative'
            else:
                alt_dir = f'{BASE}/data/{version}/probe_checkpoints/alternative_turn_{turn}'

            for cond in ALT_CONDITIONS:
                path = f'{alt_dir}/{cond}/accuracy_summary.pkl'
                try:
                    with open(path, 'rb') as f:
                        d = pickle.load(f)
                    data[version][turn][cond] = np.array(d['acc'])
                except FileNotFoundError:
                    missing.append(f'{version}/turn_{turn}/{cond}')
                    data[version][turn][cond] = None

            # Baselines from standard turn probes
            for ptype, key in [('reading_probe', 'baseline_reading'),
                                ('control_probe', 'baseline_control')]:
                path = f'{BASE}/data/{version}/probe_checkpoints/turn_{turn}/{ptype}/accuracy_summary.pkl'
                try:
                    with open(path, 'rb') as f:
                        d = pickle.load(f)
                    data[version][turn][key] = np.array(d['acc'])
                except FileNotFoundError:
                    data[version][turn][key] = None

    if missing:
        print(f"Warning: {len(missing)} missing conditions:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    return data


def count_available(data):
    """Count how many conditions have data."""
    total = 0
    available = 0
    for v in VERSIONS:
        for t in TURNS:
            for c in ALT_CONDITIONS:
                total += 1
                if data[v][t][c] is not None:
                    available += 1
    return available, total


# ── Plotting ─────────────────────────────────────────────────────────────────

def style_axis(ax, title):
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Best Test Accuracy', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.6)
    ax.set_xlim(0, N_LAYERS - 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.2)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def make_single_version_turn_figure(data, version, turn):
    """One graph: all conditions + baselines for a single version and turn."""
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(N_LAYERS)
    has_any = False

    # Baselines
    for key, label, color, ls in [
        ('baseline_reading', 'Baseline: Reading Probe', '#333333', '-'),
        ('baseline_control', 'Baseline: Control Probe', '#888888', '--'),
    ]:
        acc = data[version][turn].get(key)
        if acc is not None:
            ax.plot(layers, acc, color=color, lw=2.2, ls=ls, alpha=0.85, label=label)
            peak = int(np.argmax(acc))
            ax.plot(peak, acc[peak], marker='*', markersize=11, color=color,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
            has_any = True

    # Alternative conditions
    for cond in ALT_CONDITIONS:
        acc = data[version][turn].get(cond)
        if acc is not None:
            ax.plot(layers, acc, color=ALT_COLORS[cond], lw=2.0, alpha=0.85,
                    label=ALT_LABELS[cond], marker=ALT_MARKERS[cond],
                    markersize=4, markevery=3)
            peak = int(np.argmax(acc))
            ax.plot(peak, acc[peak], marker='*', markersize=12,
                    color=ALT_COLORS[cond], markeredgecolor='black',
                    markeredgewidth=0.5, zorder=5)
            has_any = True

    if not has_any:
        plt.close(fig)
        return None

    style_axis(ax, f'{VERSION_LABELS[version]} — Turn {turn}')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
    fig.tight_layout()
    return fig


def make_cross_version_figure(data, turn, condition):
    """Compare one condition across all 3 versions for a given turn."""
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = np.arange(N_LAYERS)
    has_any = False

    for version in VERSIONS:
        acc = data[version][turn].get(condition)
        if acc is not None:
            ax.plot(layers, acc, color=VERSION_COLORS[version], lw=2.2, alpha=0.85,
                    label=VERSION_LABELS[version])
            peak = int(np.argmax(acc))
            ax.plot(peak, acc[peak], marker='*', markersize=12,
                    color=VERSION_COLORS[version], markeredgecolor='black',
                    markeredgewidth=0.5, zorder=5)
            has_any = True

    if not has_any:
        plt.close(fig)
        return None

    cond_label = ALT_LABELS.get(condition, condition)
    style_axis(ax, f'{cond_label} — Turn {turn}, Cross-Version')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    return fig


def make_turn_progression_figure(data, version, condition):
    """Show how one condition's accuracy changes across turns 1-5 for a version."""
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(N_LAYERS)
    turn_colors = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#d73027']
    has_any = False

    for i, turn in enumerate(TURNS):
        acc = data[version][turn].get(condition)
        if acc is not None:
            ax.plot(layers, acc, color=turn_colors[i], lw=2.0, alpha=0.85,
                    label=f'Turn {turn}')
            peak = int(np.argmax(acc))
            ax.plot(peak, acc[peak], marker='*', markersize=11,
                    color=turn_colors[i], markeredgecolor='black',
                    markeredgewidth=0.5, zorder=5)
            has_any = True

    if not has_any:
        plt.close(fig)
        return None

    cond_label = ALT_LABELS.get(condition, condition)
    style_axis(ax, f'{VERSION_LABELS[version]} — {cond_label} by Turn')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    return fig


def make_summary_grid(data):
    """5 rows (turns) × 3 columns (versions), each panel shows all conditions."""
    fig, axes = plt.subplots(5, 3, figsize=(18, 22), sharey=True)
    layers = np.arange(N_LAYERS)

    for row, turn in enumerate(TURNS):
        for col, version in enumerate(VERSIONS):
            ax = axes[row, col]

            # Baselines
            for key, label, color, ls in [
                ('baseline_reading', 'Reading', '#333333', '-'),
                ('baseline_control', 'Control', '#888888', '--'),
            ]:
                acc = data[version][turn].get(key)
                if acc is not None:
                    ax.plot(layers, acc, color=color, lw=1.5, ls=ls, alpha=0.7,
                            label=label if row == 0 and col == 0 else None)

            # Alt conditions
            for cond in ALT_CONDITIONS:
                acc = data[version][turn].get(cond)
                if acc is not None:
                    short_label = ALT_LABELS[cond].split('(')[0].strip()
                    ax.plot(layers, acc, color=ALT_COLORS[cond], lw=1.5, alpha=0.8,
                            label=short_label if row == 0 and col == 0 else None)
                    peak = int(np.argmax(acc))
                    ax.plot(peak, acc[peak], marker='*', markersize=8,
                            color=ALT_COLORS[cond], markeredgecolor='black',
                            markeredgewidth=0.3, zorder=5)

            ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
            ax.set_xlim(0, N_LAYERS - 1)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.grid(True, alpha=0.15)

            if row == 0:
                ax.set_title(VERSION_LABELS[version], fontsize=11, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'Turn {turn}\nAccuracy', fontsize=10)
            if row == 4:
                ax.set_xlabel('Layer', fontsize=10)

    # Add legend at top
    axes[0, 0].legend(fontsize=7, loc='upper left', framealpha=0.8, ncol=1)

    fig.suptitle('Alternative Token Position Probes — All Turns × Versions',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


# ── HTML assembly ────────────────────────────────────────────────────────────

def build_html(data):
    available, total = count_available(data)
    print(f"  Data available: {available}/{total} conditions")

    css = """
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px; margin: 0 auto; padding: 24px 20px;
        background: #fafafa; color: #222;
      }
      h1 { text-align: center; margin-bottom: 6px; color: #1a1a2e; }
      h2 { color: #1a3399; border-bottom: 2px solid #e0e4ee; padding-bottom: 6px;
           margin-top: 44px; }
      h3 { color: #444; margin-top: 28px; }
      .subtitle { text-align: center; color: #666; font-size: 14px;
                   margin-bottom: 36px; }
      .fig-container { text-align: center; margin: 12px 0 24px 0; }
      .fig-container img { max-width: 100%; border: 1px solid #ddd;
                           border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
      table { border-collapse: collapse; margin: 12px auto; font-size: 13px; }
      th, td { border: 1px solid #ccc; padding: 6px 14px; text-align: center; }
      th { background: #e8ecf4; color: #1a1a2e; }
      tr:nth-child(even) { background: #f5f6fa; }
      .section-sep { border: 0; border-top: 1px solid #ddd; margin: 40px 0; }
      .note { color: #666; font-size: 13px; font-style: italic; margin: 8px 0; }
      .chance-cell { background: #fee2e2; color: #991b1b; }
      .above-cell { background: #dcfce7; color: #166534; }
      .strong-cell { background: #bbf7d0; color: #14532d; }
    </style>
    """

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Alternative Token Position Probes — Cross-Version Comparison</title>
{css}
</head>
<body>
<h1>Alternative Token Position Probes</h1>
<p class="subtitle">Comparing token position probes across 3 versions x 5 turns x 4 alternative conditions<br>
Labels (Primary) &nbsp;|&nbsp; Balanced GPT &nbsp;|&nbsp; Nonsense Codeword (Control)</p>
"""]

    # ── Caveat: causal attention and the </s> confound ──
    parts.append("""
<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:16px 20px; margin:20px 0;">
<h3 style="color:#856404; margin-top:0;">Important: Causal Attention Confound for First &lt;/s&gt; Token</h3>
<p style="font-size:13px; color:#856404;">
The <strong>first &lt;/s&gt; token</strong> achieves perfect accuracy (1.000) at every turn because
LLaMA-2 uses <strong>causal (left-to-right) attention</strong>. The model&rsquo;s representation at
the first &lt;/s&gt; depends only on tokens that precede it &mdash; and those tokens are
<em>identical</em> regardless of how many conversation turns follow.
</p>
<ul style="font-size:13px; color:#856404; margin-bottom:0;">
<li><strong>Turn 1</strong>: No &lt;/s&gt; token exists (only system + user prompt, no assistant response).
The code falls back to the last token (<code>]</code> from <code>[/INST]</code>), making this
condition identical to the baseline control probe.</li>
<li><strong>Turns 2&ndash;5</strong>: The text before the first &lt;/s&gt; is <em>exactly identical</em>
across all turns (same 1028 chars). The first &lt;/s&gt; sits at the same position with the same
causal context. Subsequent turns cannot influence it because information flows only left-to-right.</li>
</ul>
<p style="font-size:13px; color:#856404; margin-bottom:0;">
The constant perfect accuracy is therefore an artifact of probing an invariant position, not evidence
that partner identity is preserved across turns. It remains true that this structural boundary token
is an exceptionally informative position &mdash; but its accuracy does not address the prompt dilution
question.
</p>
</div>
""")

    # ── Section 1: Summary grid ──
    parts.append('<h2>1. Overview Grid (All Turns x Versions)</h2>')
    parts.append('<p class="note">Each panel shows all 6 conditions (2 baselines + 4 alternative) '
                 'for one version at one turn. Stars mark peak layers.</p>')

    fig = make_summary_grid(data)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                 f'alt="Summary grid all turns x versions"></div>')

    # ── Section 2: Summary table ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>2. Summary Table</h2>')
    parts.append('<p class="note">Peak accuracy and layer for each condition, version, and turn. '
                 'Values at chance (&le;0.55) shown in red; above baseline (&gt;0.70) in green.</p>')
    parts.append(make_summary_tables(data))

    # ── Section 3: Per-version detail (all turns) ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>3. Per-Version Detail (All Conditions by Turn)</h2>')
    parts.append('<p class="note">Each graph shows all conditions for one version at one turn.</p>')

    for version in VERSIONS:
        parts.append(f'<h3>{VERSION_LABELS[version]}</h3>')
        for turn in TURNS:
            fig = make_single_version_turn_figure(data, version, turn)
            if fig is not None:
                b64 = fig_to_base64(fig)
                plt.close(fig)
                parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                             f'alt="{version} turn {turn}"></div>')
            else:
                parts.append(f'<p><em>No data available for {VERSION_LABELS[version]} Turn {turn}.</em></p>')

    # ── Section 4: Cross-version comparison per condition ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>4. Cross-Version Comparison by Condition</h2>')
    parts.append('<p class="note">Compare each alternative condition across all 3 versions, '
                 'one graph per turn.</p>')

    for cond in ALT_CONDITIONS:
        parts.append(f'<h3>{ALT_LABELS[cond]}</h3>')
        for turn in TURNS:
            fig = make_cross_version_figure(data, turn, cond)
            if fig is not None:
                b64 = fig_to_base64(fig)
                plt.close(fig)
                parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                             f'alt="{cond} turn {turn} cross-version"></div>')

    # ── Section 5: Turn progression per condition ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>5. Turn Progression by Condition</h2>')
    parts.append('<p class="note">How each condition changes across turns 1-5 '
                 'for each version. Shows the prompt dilution effect.</p>')

    for version in VERSIONS:
        parts.append(f'<h3>{VERSION_LABELS[version]}</h3>')
        for cond in ALT_CONDITIONS:
            fig = make_turn_progression_figure(data, version, cond)
            if fig is not None:
                b64 = fig_to_base64(fig)
                plt.close(fig)
                parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                             f'alt="{version} {cond} turn progression"></div>')

    # ── Footer ──
    parts.append("""
<hr class="section-sep">
<p class="note" style="text-align:center;">
Generated &nbsp;|&nbsp;
Data: exp_2/data/{version}/probe_checkpoints/alternative[_turn_N]/{condition}/accuracy_summary.pkl
</p>
</body></html>""")

    return '\n'.join(parts)


def make_summary_tables(data):
    """HTML tables: one per version, rows = turns, cols = conditions."""
    all_tables = []

    for version in VERSIONS:
        rows = []
        for turn in TURNS:
            cells = [f'<td style="font-weight:bold;">Turn {turn}</td>']

            # Baselines
            for key in ['baseline_reading', 'baseline_control']:
                acc = data[version][turn].get(key)
                if acc is not None:
                    peak = int(np.argmax(acc))
                    cells.append(f'<td>{acc[peak]:.3f} (L{peak})</td>')
                else:
                    cells.append('<td>—</td>')

            # Alt conditions
            for cond in ALT_CONDITIONS:
                acc = data[version][turn].get(cond)
                if acc is not None:
                    peak = int(np.argmax(acc))
                    peak_acc = acc[peak]
                    css_class = ''
                    if peak_acc <= 0.55:
                        css_class = ' class="chance-cell"'
                    elif peak_acc > 0.70:
                        css_class = ' class="above-cell"'
                    if peak_acc >= 0.95:
                        css_class = ' class="strong-cell"'
                    cells.append(f'<td{css_class}>{peak_acc:.3f} (L{peak})</td>')
                else:
                    cells.append('<td>—</td>')

            rows.append('<tr>' + ''.join(cells) + '</tr>')

        table = f"""
        <h4>{VERSION_LABELS[version]}</h4>
        <table>
          <thead>
            <tr>
              <th>Turn</th>
              <th>Baseline Reading</th>
              <th>Baseline Control</th>
              <th>BOS</th>
              <th>Random</th>
              <th>First &lt;/s&gt;</th>
              <th>Weather Suffix</th>
            </tr>
          </thead>
          <tbody>
        {''.join(rows)}
          </tbody>
        </table>
        """
        all_tables.append(table)

    return '\n'.join(all_tables)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_data()

    print("Building HTML...")
    html = build_html(data)

    import sys
    sys.path.insert(0, os.path.join(BASE, 'code'))
    from src.report_utils import save_report

    print(f"Writing to {OUTPUT_PATH}")
    save_report(html, OUTPUT_PATH)

    # Quick summary
    for version in VERSIONS:
        print(f"\n--- {VERSION_LABELS[version]} ---")
        for turn in TURNS:
            parts = [f"  Turn {turn}:"]
            for cond in ALT_CONDITIONS:
                acc = data[version][turn].get(cond)
                if acc is not None:
                    peak = int(np.argmax(acc))
                    parts.append(f"{cond}={acc[peak]:.3f}")
                else:
                    parts.append(f"{cond}=N/A")
            print(' | '.join(parts))

    print(f"\nDone! Output: {OUTPUT_PATH}")
    fsize = os.path.getsize(OUTPUT_PATH)
    print(f"File size: {fsize / 1024:.1f} KB")


if __name__ == "__main__":
    main()
