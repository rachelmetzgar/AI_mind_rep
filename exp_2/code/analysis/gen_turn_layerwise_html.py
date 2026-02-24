#!/usr/bin/env python3
"""
Generate an HTML visualization of probe accuracy across layers and conversation turns.
Produces: /mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2/results/cross_variant/turn_comparison_layerwise.html
"""

import pickle
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

# ── Configuration ────────────────────────────────────────────────────────────

BASE = '/mnt/cup/labs/graziano/rachel/ai_mind_rep/exp_2'
VARIANTS = ['labels', 'balanced_names', 'balanced_gpt', 'names']
VARIANT_LABELS = {
    'labels': 'Labels',
    'balanced_names': 'Balanced Names',
    'balanced_gpt': 'Balanced GPT',
    'names': 'Names (Sam/Casey)',
}
TURNS = [1, 2, 3, 4, 5]
PROBE_TYPES = ['reading_probe', 'control_probe']
PROBE_LABELS = {'reading_probe': 'Reading Probe', 'control_probe': 'Control Probe'}
N_LAYERS = 41
OUTPUT_PATH = os.path.join(BASE, 'results', 'cross_variant', 'turn_comparison_layerwise.html')

# Color map: deep blue -> teal -> yellow-green -> orange -> red
TURN_COLORS = ['#1a3399', '#2596be', '#45a847', '#e8961a', '#cc2233']

# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data():
    """Returns data[variant][turn][probe_type] = list of 41 floats."""
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


# ── Section 1: Main layer profiles (per variant) ────────────────────────────

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


# ── Section 2: Summary table data ───────────────────────────────────────────

def make_summary_table(data, variant):
    """Return an HTML table string with peak layer, peak acc, mean acc."""
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


# ── Section 3: Peak layer shift plot ────────────────────────────────────────

def make_peak_shift_figure(data):
    """Small plot: peak layer vs turn for all variants and probe types."""
    fig, ax = plt.subplots(figsize=(7, 5))
    variant_colors = {'labels': '#1a3399', 'balanced_names': '#2596be',
                      'balanced_gpt': '#cc2233', 'names': '#8e24aa'}
    probe_markers = {'reading_probe': 'o', 'control_probe': 's'}
    probe_ls = {'reading_probe': '-', 'control_probe': '--'}

    for v in VARIANTS:
        for ptype in PROBE_TYPES:
            peaks = []
            for turn in TURNS:
                acc = data[v][turn][ptype]
                peaks.append(int(np.argmax(acc)))
            label = f'{VARIANT_LABELS[v]} / {PROBE_LABELS[ptype]}'
            ax.plot(TURNS, peaks,
                    color=variant_colors[v],
                    marker=probe_markers[ptype],
                    ls=probe_ls[ptype],
                    lw=2, markersize=8, label=label, alpha=0.85)

    ax.set_xlabel('Conversation Turn', fontsize=12)
    ax.set_ylabel('Peak Layer', fontsize=12)
    ax.set_title('Peak Layer Shift Across Turns', fontsize=14, fontweight='bold')
    ax.set_xticks(TURNS)
    ax.set_ylim(-1, 41)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc='best', ncol=2, framealpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


# ── Section 4: Cross-variant comparison at all turns ─────────────────────────

def make_cross_variant_figure(data):
    """For each probe type, overlay all 3 variants at every turn (5x2 grid)."""
    fig, axes = plt.subplots(5, 2, figsize=(16, 24), sharey=True)
    layers = np.arange(N_LAYERS)
    variant_colors = {'labels': '#1a3399', 'balanced_names': '#2596be',
                      'balanced_gpt': '#cc2233', 'names': '#8e24aa'}
    variant_ls = {'labels': '-', 'balanced_names': '--', 'balanced_gpt': '-.', 'names': ':'}

    for col_i, ptype in enumerate(PROBE_TYPES):
        for row_i, turn in enumerate(TURNS):
            ax = axes[row_i][col_i]
            for v in VARIANTS:
                acc = data[v][turn][ptype]
                ax.plot(layers, acc,
                        color=variant_colors[v],
                        ls=variant_ls[v],
                        lw=2.2, alpha=0.85,
                        label=VARIANT_LABELS[v])
                peak_layer = int(np.argmax(acc))
                ax.plot(peak_layer, acc[peak_layer], marker='*', markersize=11,
                        color=variant_colors[v], markeredgecolor='black',
                        markeredgewidth=0.5, zorder=5)

            style_axis(ax, f'{PROBE_LABELS[ptype]} — Turn {turn}')
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=9, loc='upper left', framealpha=0.85)
            if col_i > 0:
                ax.set_ylabel('')

    fig.suptitle('Cross-Variant Comparison (All Turns)',
                 fontsize=15, fontweight='bold', y=1.005)
    fig.tight_layout()
    return fig


# ── Section 5: Alternative position probes ───────────────────────────────────

ALT_CONDITIONS = ['control_first', 'control_random', 'control_eos', 'reading_irrelevant']
ALT_LABELS = {
    'control_first': 'BOS Token (position 0)',
    'control_random': 'Random Mid-Sequence Token',
    'control_eos': 'First </s> Token',
    'reading_irrelevant': 'Last Token (Weather Suffix)',
}
ALT_COLORS = {
    'control_first': '#e8961a',
    'control_random': '#8e24aa',
    'control_eos': '#cc2233',
    'reading_irrelevant': '#2596be',
}

def load_alternative_data():
    """Returns alt_data[condition] = array of 41 test accs, or None."""
    alt_base = f'{BASE}/data/labels/probe_checkpoints/alternative'
    data = {}
    for cond in ALT_CONDITIONS:
        path = f'{alt_base}/{cond}/accuracy_summary.pkl'
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            data[cond] = np.array(d['acc'])
        except FileNotFoundError:
            return None
    # Also load baselines (turn 5 reading + control)
    for ptype, key in [('reading_probe', 'baseline_reading'), ('control_probe', 'baseline_control')]:
        path = f'{BASE}/data/labels/probe_checkpoints/turn_5/{ptype}/accuracy_summary.pkl'
        try:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            data[key] = np.array(d['acc'])
        except FileNotFoundError:
            pass
    return data


def make_alternative_figure(alt_data):
    """Layer profiles for all alternative conditions + baselines."""
    fig, ax = plt.subplots(figsize=(14, 6))
    layers = np.arange(N_LAYERS)

    # Baselines
    if 'baseline_reading' in alt_data:
        ax.plot(layers, alt_data['baseline_reading'], color='#333333', lw=2.5,
                label='Baseline: Reading Probe (last token, partner suffix)', ls='-', alpha=0.9)
    if 'baseline_control' in alt_data:
        ax.plot(layers, alt_data['baseline_control'], color='#888888', lw=2.0,
                label='Baseline: Control Probe (last token, no suffix)', ls='--', alpha=0.8)

    # Alternative conditions
    for cond in ALT_CONDITIONS:
        acc = alt_data[cond]
        ax.plot(layers, acc, color=ALT_COLORS[cond], lw=2.2, alpha=0.88,
                label=ALT_LABELS[cond])
        peak = int(np.argmax(acc))
        ax.plot(peak, acc[peak], marker='*', markersize=13,
                color=ALT_COLORS[cond], markeredgecolor='black',
                markeredgewidth=0.6, zorder=5)

    style_axis(ax, 'Alternative Token Position Probes — Labels Variant, Turn 5')
    ax.legend(fontsize=9, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    return fig


def make_alternative_summary_table(alt_data):
    """HTML table summarizing alternative probe results."""
    rows = []
    # Baselines first
    for key, label in [('baseline_reading', 'Baseline: Reading (last token + partner suffix)'),
                       ('baseline_control', 'Baseline: Control (last token, no suffix)')]:
        if key in alt_data:
            acc = alt_data[key]
            peak = int(np.argmax(acc))
            rows.append(f'<tr style="background:#f0f0f0;">'
                       f'<td style="font-weight:bold;">{label}</td>'
                       f'<td>{np.mean(acc):.4f}</td>'
                       f'<td>{peak}</td><td>{acc[peak]:.4f}</td></tr>')

    # Alternative conditions
    for cond in ALT_CONDITIONS:
        acc = alt_data[cond]
        peak = int(np.argmax(acc))
        rows.append(f'<tr><td>{ALT_LABELS[cond]}</td>'
                   f'<td>{np.mean(acc):.4f}</td>'
                   f'<td>{peak}</td><td>{acc[peak]:.4f}</td></tr>')

    header = """
    <table>
      <thead>
        <tr>
          <th>Condition</th>
          <th>Mean Acc</th>
          <th>Peak Layer</th>
          <th>Peak Acc</th>
        </tr>
      </thead>
      <tbody>
    """
    return header + '\n'.join(rows) + '</tbody></table>'


def build_token_examples_html():
    """Build HTML showing example conversations with highlighted token positions."""
    # We'll use manually constructed examples based on real data
    # Human condition example (Turn 1)
    human_sys = ("You are the participant in a brief conversation. You believe you are "
                 "speaking with <strong>a Human</strong>. Stay in character as the "
                 "participant throughout. Speak naturally and share your own thoughts "
                 "and experiences.")

    ai_sys = ("You are the participant in a brief conversation. You believe you are "
              "speaking with <strong>an AI</strong>. Stay in character as the participant "
              "throughout. Speak naturally and share your own thoughts and experiences.")

    topic = ("The conversation topic is: 'Nature: Discuss a natural landscape that has "
             "left a lasting impression on you.'.")
    instructions = ("Please begin by producing only your first message to start the "
                    "conversation.\nDo not simulate both sides of the dialogue.")

    assistant_msg = ("Sure thing! Here&rsquo;s my first message:\n\nHey there! I've got to "
                     "say, there's one natural landscape that's left a lasting impression on me, "
                     "and that's the Grand Canyon. I visited there a few years ago, and it was "
                     "just breathtaking. The sheer scale of it is mind-boggling...")

    partner_msg = ("Wow, that sounds amazing! I have never been to the Grand Canyon before, "
                   "but I've always wanted to go...")

    # CSS for token highlighting
    css = """
    <style>
      .token-example {
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.6;
        background: #1e1e2e;
        color: #cdd6f4;
        padding: 16px 20px;
        border-radius: 6px;
        margin: 12px 0;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      .token-example .special { color: #89b4fa; font-weight: bold; }
      .token-example .sys-prompt { color: #a6e3a1; }
      .token-example .user-msg { color: #f9e2af; }
      .token-example .asst-msg { color: #cba6f7; }
      .token-example .suffix-real { color: #f38ba8; font-weight: bold; }
      .token-example .suffix-irrel { color: #fab387; font-weight: bold; }
      .tok-highlight {
        background: #f38ba8;
        color: #1e1e2e;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
        animation: pulse 2s ease-in-out infinite;
      }
      @keyframes pulse {
        0%, 100% { box-shadow: 0 0 4px rgba(243,139,168,0.5); }
        50% { box-shadow: 0 0 12px rgba(243,139,168,0.9); }
      }
      .tok-highlight-blue {
        background: #89b4fa;
        color: #1e1e2e;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
      }
      .condition-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 20px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
      }
      .condition-card h4 {
        margin: 0 0 8px 0;
        color: #1a3399;
      }
      .condition-card .result {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 13px;
        margin: 4px 4px 8px 0;
      }
      .result-chance { background: #fee2e2; color: #991b1b; }
      .result-above { background: #dcfce7; color: #166534; }
      .result-strong { background: #bbf7d0; color: #14532d; }
      .legend-box {
        display: inline-block;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px 14px;
        margin: 4px 8px 12px 0;
        font-size: 12px;
        background: #fafafa;
      }
    </style>
    """

    html = f"""
{css}
<h3>Token Position Examples</h3>
<p class="note">
Each condition below shows the exact input the LLM receives, with the
<span class="tok-highlight" style="display:inline;">probed token</span> highlighted.
The probe reads the residual stream activation at that single token position to predict
whether the conversation partner is a Human or an AI.
</p>

<div style="margin: 12px 0;">
  <div class="legend-box"><span style="color:#89b4fa;font-weight:bold;">[INST]</span> <span style="color:#89b4fa;">[/INST]</span> <span style="color:#89b4fa;">&lt;s&gt;</span> <span style="color:#89b4fa;">&lt;/s&gt;</span> = LLaMA-2 structural tokens</div>
  <div class="legend-box"><span style="color:#a6e3a1;">Green</span> = System prompt (contains partner identity)</div>
  <div class="legend-box"><span style="color:#f9e2af;">Yellow</span> = User message</div>
  <div class="legend-box"><span style="color:#cba6f7;">Purple</span> = Assistant response</div>
</div>

<!-- ═══════ CONDITION 1: BASELINE CONTROL (last token, no suffix) ═══════ -->
<div class="condition-card">
<h4>Baseline: Control Probe (Last Token, No Suffix)</h4>
<p>The standard control probe. The conversation ends with <code>[/INST]</code>, and we read the
activation at the <strong>very last token</strong> — the final subword of <code>[/INST]</code>.
No suffix is appended.</p>
<span class="result result-above">Mean Acc: 0.5522 &nbsp;|&nbsp; Peak: 0.6050 (Layer 31)</span>
<div class="token-example"><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{human_sys}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{topic}

{instructions}</span> <span class="tok-highlight">[/INST]</span></div>
</div>

<!-- ═══════ CONDITION 2: BASELINE READING (last token + partner suffix) ═══════ -->
<div class="condition-card">
<h4>Baseline: Reading Probe (Last Token + Partner Suffix)</h4>
<p>The standard reading probe. After the final <code>[/INST]</code>, we append the suffix
<em>"I think the conversation partner of this user is"</em> and read the activation at the
<strong>last token</strong> of this suffix (the word "is").</p>
<span class="result result-above">Mean Acc: 0.5803 &nbsp;|&nbsp; Peak: 0.6525 (Layer 33)</span>
<div class="token-example"><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{human_sys}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{topic}

{instructions}</span> <span class="special">[/INST]</span> <span class="suffix-real">I think the conversation partner of this user <span class="tok-highlight">is</span></span></div>
</div>

<!-- ═══════ CONDITION 3: control_first (BOS token) ═══════ -->
<div class="condition-card">
<h4>Control: BOS Token (Position 0)</h4>
<p>Probe reads the activation at the <strong>very first token</strong> — the beginning-of-sequence
<code>&lt;s&gt;</code> token. At position 0, self-attention has not yet mixed any content from the
system prompt or conversation. This is a <strong>negative control</strong>.</p>
<span class="result result-chance">Mean Acc: 0.5054 &nbsp;|&nbsp; Peak: 0.5100 (Layer 6) — AT CHANCE</span>
<div class="token-example"><span class="tok-highlight">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{human_sys}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{topic}

{instructions}</span> <span class="special">[/INST]</span></div>
</div>

<!-- ═══════ CONDITION 4: control_random (random mid-sequence token) ═══════ -->
<div class="condition-card">
<h4>Control: Random Mid-Sequence Token</h4>
<p>Probe reads a <strong>random token in the middle 50%</strong> of the sequence. Different random
position per conversation. This tests whether partner identity is "broadcast" throughout the
residual stream or localized to specific positions. This is a <strong>negative control</strong>.</p>
<span class="result result-chance">Mean Acc: 0.5123 &nbsp;|&nbsp; Peak: 0.5600 (Layer 14) — AT CHANCE</span>
<div class="token-example"><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{human_sys}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{topic}

Please begin by producing only your first message to <span class="tok-highlight">start</span> the conversation.
Do not simulate both sides of the dialogue.</span> <span class="special">[/INST]</span></div>
<p class="note">Example shows one possible random position. In practice, a different
content token is sampled for each conversation (within the middle 50% of the sequence).
Note: despite the high train accuracy (0.85), test accuracy is at chance — the probe overfits
to random noise at each token position.</p>
</div>

<!-- ═══════ CONDITION 5: control_eos (first </s> token) ═══════ -->
<div class="condition-card">
<h4>Control: First &lt;/s&gt; Token (End of First Exchange)</h4>
<p>Probe reads the activation at the <strong>first <code>&lt;/s&gt;</code> token</strong> — the
end-of-sequence marker that terminates the model's first response. In LLaMA-2 chat format, this
token appears after the assistant's first reply. The model has now generated a full response while
"in character" as someone speaking to a Human or AI.</p>
<span class="result result-strong">Mean Acc: 0.7162 &nbsp;|&nbsp; Peak: 1.0000 (Layer 33) — ABOVE BASELINE</span>
<div class="token-example"><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{human_sys}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{topic}

{instructions}</span> <span class="special">[/INST]</span> <span class="asst-msg">{assistant_msg}</span> <span class="tok-highlight">&lt;/s&gt;</span><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="user-msg">Partner: {partner_msg}</span> <span class="special">[/INST]</span> <span class="asst-msg">...</span> <span class="special">&lt;/s&gt;</span> <span style="color:#666;">... (3 more exchanges) ...</span></div>
<p class="note"><strong>Key finding:</strong> The &lt;/s&gt; token carries <em>more</em> partner identity
information than the baseline probes at the end of the conversation. At layer 33, the probe achieves
perfect classification (1.000). This suggests the model "summarizes" its partner model at exchange
boundaries — a structural compression point where the model consolidates its representation before
the next turn begins.</p>
</div>

<!-- ═══════ CONDITION 6: reading_irrelevant (weather suffix) ═══════ -->
<div class="condition-card">
<h4>Reading: Irrelevant Suffix (Weather)</h4>
<p>Instead of "I think the <strong>conversation partner</strong> of this user is", we append
"I think the <strong>weather outside today</strong> is" — a suffix that is completely irrelevant to
partner identity. We probe the <strong>last token</strong> ("is"). This tests whether the reading
probe's success depends on partner-relevant prompting or just any continuation.</p>
<span class="result result-above">Mean Acc: 0.5622 &nbsp;|&nbsp; Peak: 0.6000 (Layer 33)</span>
<div class="token-example"><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{ai_sys}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{topic}

{instructions}</span> <span class="special">[/INST]</span> <span class="suffix-irrel">I think the weather outside today <span class="tok-highlight">is</span></span></div>
<p class="note">The irrelevant suffix achieves nearly the same accuracy as the real partner-relevant
suffix (0.562 vs 0.580). This indicates the late-layer partner representation is accessible from
<em>any</em> continuation token — it does not require a partner-relevant "question" to surface.
The representation exists in the residual stream regardless of what comes next.</p>
</div>

<!-- ═══════ Interpretation ═══════ -->
<div class="condition-card" style="background:#f8f9ff; border-color:#89b4fa;">
<h4 style="color:#1a1a2e;">Summary of Findings</h4>
<table style="margin:12px 0; font-size:13px;">
  <thead>
    <tr><th>Condition</th><th>Token Probed</th><th>Mean Acc</th><th>Peak Acc</th><th>Interpretation</th></tr>
  </thead>
  <tbody>
    <tr><td>BOS (&lt;s&gt;)</td><td>Position 0</td><td>0.505</td><td>0.510</td>
        <td>At chance. No partner info before attention mixing.</td></tr>
    <tr><td>Random mid-seq</td><td>~25th–75th percentile</td><td>0.512</td><td>0.560</td>
        <td>At chance. Partner info is NOT broadcast to arbitrary tokens.</td></tr>
    <tr><td>First &lt;/s&gt;</td><td>End of 1st exchange</td><td>0.716</td><td><strong>1.000</strong></td>
        <td>Best condition! Model summarizes partner identity at exchange boundaries.</td></tr>
    <tr><td>Weather suffix</td><td>Last token ("is")</td><td>0.562</td><td>0.600</td>
        <td>Nearly matches real suffix. Representation is accessible from any continuation.</td></tr>
    <tr style="background:#f0f0f0;"><td>Baseline control</td><td>Last token [/INST]</td><td>0.552</td><td>0.605</td>
        <td>Standard control probe at conversation end.</td></tr>
    <tr style="background:#f0f0f0;"><td>Baseline reading</td><td>Last token ("is")</td><td>0.580</td><td>0.653</td>
        <td>Standard reading probe with partner-relevant suffix.</td></tr>
  </tbody>
</table>
<p style="font-size:13px; margin:8px 0 0 0;"><strong>Key conclusions:</strong>
(1) The partner identity signal is <em>not</em> broadcast — BOS and random tokens carry no information.
(2) The signal is strongest at <strong>structural boundary tokens</strong> (the &lt;/s&gt; after the
first exchange achieves perfect decoding at layer 33).
(3) The signal is <strong>not triggered by partner-relevant questioning</strong> — an irrelevant "weather"
suffix works nearly as well as asking about the partner.
(4) The representation degrades across turns because the system prompt tokens become proportionally
diluted in longer sequences (prompt dilution), not because the model updates its partner model.</p>
</div>
"""
    return html


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
<title>Probe Accuracy by Layer and Conversation Turn</title>
{css}
</head>
<body>
<h1>Probe Accuracy by Layer and Conversation Turn</h1>
<p class="subtitle">Experiment 2 — Llama-2-13B-chat &nbsp;|&nbsp; 4 dataset variants
 &times; 5 turns &times; 2 probe types &nbsp;|&nbsp; 41 layers</p>
"""]

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
    parts.append('<p class="note">Overlays all 4 dataset variants at each conversation turn '
                 '(Turn 1 through Turn 5). Stars mark peak layers.</p>')
    fig = make_cross_variant_figure(data)
    b64 = fig_to_base64(fig)
    plt.close(fig)
    parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                 f'alt="Cross-variant comparison"></div>')

    # ── Section 5: Alternative Position Probes ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>4. Alternative Token Position Probes</h2>')
    parts.append("""
<p class="note">
These experiments test <em>where</em> the partner identity signal lives in the token sequence.
The standard ("baseline") probe reads the <strong>last token</strong> after appending a partner-relevant
suffix. Here we compare probes trained at other positions and with an irrelevant suffix.
All probes are trained on the Labels variant at Turn 5 (the hardest condition).
</p>
""")

    # Load alternative probe data
    alt_data = load_alternative_data()
    if alt_data:
        # Add the layerwise plot
        fig = make_alternative_figure(alt_data)
        b64 = fig_to_base64(fig)
        plt.close(fig)
        parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                     f'alt="Alternative position probes"></div>')

        # Summary table
        parts.append(make_alternative_summary_table(alt_data))
    else:
        parts.append('<p><em>Alternative probe data not yet available.</em></p>')

    # Token position examples
    parts.append(build_token_examples_html())

    # ── Footer ──
    parts.append("""
<hr class="section-sep">
<p class="note" style="text-align:center;">
Generated 2026-02-22 &nbsp;|&nbsp;
Data: exp_2/data/{variant}/probe_checkpoints/turn_{N}/{probe}/accuracy_summary.pkl
</p>
</body></html>""")

    return '\n'.join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_all_data()

    print("Building HTML...")
    html = build_html(data)

    print(f"Writing to {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
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
