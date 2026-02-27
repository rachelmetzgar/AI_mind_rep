#!/usr/bin/env python3
"""
Generate an HTML visualization of probe accuracy across layers and conversation turns.
All 6 labeling variants: names, balanced_names, balanced_gpt, labels, nonsense_codeword, nonsense_ignore.

Produces: exp_2/results/comparisons/probe_training/turn_comparison_layerwise_all_variants.html
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

BASE = '/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2'
OUTPUT_DIR = os.path.join(BASE, 'results', 'comparisons', 'probe_training')

VARIANTS = ['names', 'balanced_names', 'balanced_gpt', 'labels', 'nonsense_codeword', 'nonsense_ignore']
VARIANT_LABELS = {
    'names': 'Names',
    'balanced_names': 'Bal. Names',
    'balanced_gpt': 'Bal. GPT',
    'labels': 'Labels',
    'nonsense_codeword': 'Non. Codeword',
    'nonsense_ignore': 'Non. Ignore',
}
VARIANT_COLORS = {
    'names': '#8e24aa',
    'balanced_names': '#2596be',
    'balanced_gpt': '#cc2233',
    'labels': '#1a3399',
    'nonsense_codeword': '#888888',
    'nonsense_ignore': '#e8961a',
}
VARIANT_LS = {
    'names': '-',
    'balanced_names': '--',
    'balanced_gpt': '-.',
    'labels': '-',
    'nonsense_codeword': ':',
    'nonsense_ignore': '--',
}

TURNS = [1, 2, 3, 4, 5]
PROBE_TYPES = ['reading_probe', 'control_probe']
PROBE_LABELS = {'reading_probe': 'Reading Probe', 'control_probe': 'Control Probe'}
N_LAYERS = 41

TURN_COLORS = ['#1a3399', '#2596be', '#45a847', '#e8961a', '#cc2233']

OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'turn_comparison_layerwise_all_variants.html')

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Returns data[variant][turn][probe_type] = array of 41 floats."""
    data = {}
    for v in VARIANTS:
        data[v] = {}
        for t in TURNS:
            data[v][t] = {}
            for p in PROBE_TYPES:
                path = (f'{BASE}/data/{v}/'
                        f'probe_checkpoints/turn_{t}/{p}/accuracy_summary.pkl')
                with open(path, 'rb') as f:
                    d = pickle.load(f)
                data[v][t][p] = np.array(d['acc'])
    return data


# ── Plotting helpers ─────────────────────────────────────────────────────────

def fig_to_base64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded


def style_axis(ax, title, ylabel='Test Accuracy', xlabel='Layer'):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(-0.5, 40.5)
    ax.set_ylim(0.40, 1.02)
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.7, zorder=0)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, which='major')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Section 1: Per-variant layer profiles (reading + control, 5 turns) ──────

def make_variant_figure(data, variant):
    """Two side-by-side plots: reading vs control for one variant."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    layers = np.arange(N_LAYERS)

    for ax_i, ptype in enumerate(PROBE_TYPES):
        ax = axes[ax_i]
        for ti, turn in enumerate(TURNS):
            acc = data[variant][turn][ptype]
            color = TURN_COLORS[ti]
            ax.plot(layers, acc, color=color, lw=2.0, alpha=0.88,
                    label=f'Turn {turn}')
            peak_layer = int(np.argmax(acc))
            peak_val = acc[peak_layer]
            ax.plot(peak_layer, peak_val, marker='*', markersize=12,
                    color=color, markeredgecolor='black', markeredgewidth=0.6,
                    zorder=5)

        style_axis(ax, f'{PROBE_LABELS[ptype]}')
        if ax_i == 0:
            ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
        else:
            ax.set_ylabel('')

    fig.suptitle(f'{VARIANT_LABELS[variant]} — Layer Profiles by Turn',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


# ── Section 2: Summary table per variant ─────────────────────────────────────

def make_summary_table(data, variant):
    """Return HTML table with peak layer, peak acc, mean acc."""
    rows = []
    for turn in TURNS:
        cells = [f'<td style="font-weight:bold;">Turn {turn}</td>']
        for ptype in PROBE_TYPES:
            acc = data[variant][turn][ptype]
            peak_layer = int(np.argmax(acc))
            peak_acc = acc[peak_layer]
            mean_acc = np.mean(acc)
            cells.append(f'<td>{peak_layer}</td>')
            cells.append(f'<td>{peak_acc:.4f}</td>')
            cells.append(f'<td>{mean_acc:.4f}</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    header = """
    <table>
      <thead>
        <tr>
          <th rowspan="2">Turn</th>
          <th colspan="3">Reading Probe</th>
          <th colspan="3">Control Probe</th>
        </tr>
        <tr>
          <th>Peak Layer</th><th>Peak Acc</th><th>Mean Acc</th>
          <th>Peak Layer</th><th>Peak Acc</th><th>Mean Acc</th>
        </tr>
      </thead>
      <tbody>
    """
    footer = "</tbody></table>"
    return header + '\n'.join(rows) + footer


# ── Section 3: Peak layer shift plot ─────────────────────────────────────────

def make_peak_shift_figure(data):
    """Peak layer vs turn for all variants and probe types."""
    fig, ax = plt.subplots(figsize=(9, 6))
    probe_markers = {'reading_probe': 'o', 'control_probe': 's'}
    probe_ls_mod = {'reading_probe': '-', 'control_probe': '--'}

    for v in VARIANTS:
        for ptype in PROBE_TYPES:
            peaks = []
            for turn in TURNS:
                acc = data[v][turn][ptype]
                peaks.append(int(np.argmax(acc)))
            label = f'{VARIANT_LABELS[v]} / {PROBE_LABELS[ptype]}'
            ax.plot(TURNS, peaks,
                    color=VARIANT_COLORS[v],
                    marker=probe_markers[ptype],
                    ls=probe_ls_mod[ptype],
                    lw=2, markersize=8, label=label, alpha=0.85)

    ax.set_xlabel('Conversation Turn', fontsize=12)
    ax.set_ylabel('Peak Layer', fontsize=12)
    ax.set_title('Peak Layer Shift Across Turns', fontsize=14, fontweight='bold')
    ax.set_xticks(TURNS)
    ax.set_ylim(-1, 41)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc='best', ncol=2, framealpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


# ── Section 4: Cross-variant comparison (all turns, overlaid) ────────────────

def make_cross_variant_figure(data):
    """For each probe type, overlay all 6 variants at every turn (5x2 grid)."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 28), sharey=True)
    layers = np.arange(N_LAYERS)

    for col_i, ptype in enumerate(PROBE_TYPES):
        for row_i, turn in enumerate(TURNS):
            ax = axes[row_i][col_i]
            for v in VARIANTS:
                acc = data[v][turn][ptype]
                ax.plot(layers, acc,
                        color=VARIANT_COLORS[v],
                        ls=VARIANT_LS[v],
                        lw=2.2, alpha=0.85,
                        label=VARIANT_LABELS[v])
                peak_layer = int(np.argmax(acc))
                ax.plot(peak_layer, acc[peak_layer], marker='*', markersize=11,
                        color=VARIANT_COLORS[v], markeredgecolor='black',
                        markeredgewidth=0.5, zorder=5)

            style_axis(ax, f'{PROBE_LABELS[ptype]} — Turn {turn}')
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=8, loc='upper left', framealpha=0.85, ncol=2)
            if col_i > 0:
                ax.set_ylabel('')

    fig.suptitle('Cross-Variant Comparison (All Turns)',
                 fontsize=15, fontweight='bold', y=1.005)
    fig.tight_layout()
    return fig


# ── Section 5: Peak accuracy summary heatmap ─────────────────────────────────

def make_peak_acc_heatmap(data):
    """Heatmap: peak accuracy by variant x turn, for reading and control."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_i, ptype in enumerate(PROBE_TYPES):
        mat = np.zeros((len(VARIANTS), len(TURNS)))
        for vi, v in enumerate(VARIANTS):
            for ti, turn in enumerate(TURNS):
                acc = data[v][turn][ptype]
                mat[vi, ti] = np.max(acc)

        im = ax_i  # just for naming
        cax = axes[ax_i].imshow(mat, aspect='auto', cmap='RdYlGn', vmin=0.45, vmax=1.0)
        axes[ax_i].set_xticks(range(len(TURNS)))
        axes[ax_i].set_xticklabels([f'Turn {t}' for t in TURNS], fontsize=10)
        axes[ax_i].set_yticks(range(len(VARIANTS)))
        axes[ax_i].set_yticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=10)
        axes[ax_i].set_title(PROBE_LABELS[ptype], fontsize=13, fontweight='bold')

        # Annotate cells
        for vi in range(len(VARIANTS)):
            for ti in range(len(TURNS)):
                val = mat[vi, ti]
                color = 'white' if val < 0.65 else 'black'
                axes[ax_i].text(ti, vi, f'{val:.3f}', ha='center', va='center',
                               fontsize=9, fontweight='bold', color=color)

        fig.colorbar(cax, ax=axes[ax_i], shrink=0.8, label='Peak Test Accuracy')

    fig.suptitle('Peak Test Accuracy by Variant and Turn',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ── HTML assembly ────────────────────────────────────────────────────────────

def build_html(data):
    css = """
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 24px 20px;
        background: #fafafa;
        color: #222;
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
    </style>
    """

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Probe Accuracy by Layer and Turn — All 6 Variants</title>
{css}
</head>
<body>
<h1>Probe Accuracy by Layer and Conversation Turn</h1>
<p class="subtitle">Experiment 2 Probes &mdash; Llama-2-13B-chat &nbsp;|&nbsp; 6 labeling variants
 &times; 5 turns &times; 2 probe types &nbsp;|&nbsp; 41 layers<br>
 Generated 2026-02-26</p>
"""]

    # ── Section 0: Peak accuracy heatmap ──
    parts.append('<h2>0. Peak Accuracy Summary</h2>')
    parts.append('<p class="note">Peak test accuracy across all layers for each variant and turn. '
                 'Green = high accuracy, red = near chance.</p>')
    fig = make_peak_acc_heatmap(data)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                 f'alt="Peak accuracy heatmap"></div>')

    # ── Section 1 + 2: Per-variant profiles and tables ──
    parts.append('<h2>1. Layer Profiles by Variant</h2>')
    parts.append('<p class="note">Stars mark peak accuracy layer for each turn. '
                 'Dashed gray line = chance (0.5).</p>')

    for v in VARIANTS:
        parts.append(f'<h3>{VARIANT_LABELS[v]}</h3>')
        fig = make_variant_figure(data, v)
        b64 = fig_to_base64(fig)
        plt.close(fig)
        parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                     f'alt="{v} layer profiles"></div>')
        parts.append(make_summary_table(data, v))

    # ── Section 3: Peak layer shift ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>2. Peak Layer Migration Across Turns</h2>')
    parts.append('<p class="note">Solid lines = reading probes, dashed = control probes. '
                 'Shows how the most informative layer shifts as the conversation progresses.</p>')
    fig = make_peak_shift_figure(data)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                 f'alt="Peak layer shift"></div>')

    # ── Section 4: Cross-variant comparison ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>3. Cross-Variant Comparison (All Turns)</h2>')
    parts.append('<p class="note">Overlays all 6 labeling variants at each conversation turn. '
                 'Stars mark peak layers.</p>')
    fig = make_cross_variant_figure(data)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                 f'alt="Cross-variant comparison"></div>')

    # ── Footer ──
    parts.append("""
<hr class="section-sep">
<p class="note" style="text-align:center;">
Generated 2026-02-26 &nbsp;|&nbsp;
Data: exp_2/data/{variant}/probe_checkpoints/turn_{N}/{probe}/accuracy_summary.pkl
</p>
</body></html>""")

    return '\n'.join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    data = load_all_data()

    print("Building HTML...")
    html = build_html(data)

    print(f"Writing to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)

    # Print quick summary
    for v in VARIANTS:
        print(f"\n--- {VARIANT_LABELS[v]} ---")
        for turn in TURNS:
            r_acc = data[v][turn]['reading_probe']
            c_acc = data[v][turn]['control_probe']
            r_peak = int(np.argmax(r_acc))
            c_peak = int(np.argmax(c_acc))
            print(f"  Turn {turn}: reading peak={r_peak} ({r_acc[r_peak]:.4f}), "
                  f"control peak={c_peak} ({c_acc[c_peak]:.4f})")

    print(f"\nDone! Output: {OUTPUT_PATH}")
    fsize = os.path.getsize(OUTPUT_PATH)
    print(f"File size: {fsize / 1024:.1f} KB")


if __name__ == '__main__':
    main()
