#!/usr/bin/env python3
"""
Generate HTML reports comparing alternative token position probes across
versions and conversation turns.

Versions: balanced_gpt, nonsense_codeword
Conditions: control_first (BOS), control_random, control_eos (first </s>),
            reading_irrelevant (weather suffix), + baselines
Turns: 1-5

Output structure:
  exp_2/results/comparisons/probe_training/alt_tokens/
    balanced_gpt.html    — Per-version: token examples + all turns/conditions
    nonsense_codeword.html — Per-version: token examples + all turns/conditions
    combined.html        — Cross-version: summary figure, comparison tables, overlays
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
OUTPUT_DIR = f'{BASE}/results/comparisons/probe_training/alt_tokens'

VERSIONS = ['balanced_gpt', 'nonsense_codeword']
VERSION_LABELS = {
    'balanced_gpt': 'Balanced GPT',
    'nonsense_codeword': 'Nonsense Codeword (Control)',
}
VERSION_COLORS = {
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

# ── Version-specific system prompts ──────────────────────────────────────────
# These match the exact templates in exp_1/versions/*/code/data_gen/utils/prompts_config.py

VERSION_SYS_PROMPTS = {
    'labels': {
        'human': ('You are the participant in a brief conversation. You believe you are '
                  'speaking with <strong>a Human</strong>. Stay in character as the '
                  'participant throughout. Speak naturally and share your own thoughts '
                  'and experiences.'),
        'ai': ('You are the participant in a brief conversation. You believe you are '
               'speaking with <strong>an AI</strong>. Stay in character as the participant '
               'throughout. Speak naturally and share your own thoughts and experiences.'),
        'desc': 'Direct label in second-person belief frame',
        'partner_prefix': 'Partner',
    },
    'balanced_gpt': {
        'human': ('You are the participant in a brief conversation. You believe you are '
                  'speaking with <strong>Gregory</strong> (<strong>a Human</strong>). '
                  'Stay in character as the participant throughout. Speak naturally and '
                  'share your own thoughts and experiences.'),
        'ai': ('You are the participant in a brief conversation. You believe you are '
               'speaking with <strong>ChatGPT</strong> (<strong>an AI</strong>). '
               'Stay in character as the participant throughout. Speak naturally and '
               'share your own thoughts and experiences.'),
        'desc': 'Named partner with type in parentheses',
        'partner_prefix_human': 'Gregory',
        'partner_prefix_ai': 'ChatGPT',
    },
    'nonsense_codeword': {
        'human': ('You are the participant in a brief conversation. Your assigned session '
                  'code word is <strong>a Human</strong>. Stay in character as the '
                  'participant throughout. Speak naturally and share your own thoughts '
                  'and experiences.'),
        'ai': ('You are the participant in a brief conversation. Your assigned session '
               'code word is <strong>an AI</strong>. Stay in character as the participant '
               'throughout. Speak naturally and share your own thoughts and experiences.'),
        'desc': 'Semantically vacuous code word frame',
        'partner_prefix': 'Partner',
    },
}

# ── Conversation content (shared across all versions) ────────────────────────
# Real 5-turn conversation from s001.csv trial 2 (human condition, nature topic)

TOPIC_AND_INSTRUCTIONS = ("The conversation topic is: 'Nature: Discuss a natural landscape "
                          "that has left a lasting impression on you.'.\n\nPlease begin by "
                          "producing only your first message to start the conversation.\n"
                          "Do not simulate both sides of the dialogue.")

ASST_1 = ("Sure thing! Here&rsquo;s my first message:\n\nHey there! I&rsquo;ve got to say, "
          "there&rsquo;s one natural landscape that&rsquo;s left a lasting impression on me, "
          "and that&rsquo;s the Grand Canyon. I visited there a few years ago, and it was just "
          "breathtaking. The sheer scale of it is mind-boggling &ndash; the way the rock "
          "formations stretch out as far as the eye can see, and the different colors of the "
          "rock layers that reveal the history of the earth. It&rsquo;s a really humbling "
          "experience, being in the presence of something so much bigger than ourselves. "
          "Have you ever been to the Grand Canyon?")

# For balanced_gpt, partner messages are prefixed with the partner's name
PARTNER_2_BODY = ("Wow, that sounds amazing! I have never been to the Grand Canyon "
                  "before, but I&rsquo;ve always wanted to go. I&rsquo;ve seen pictures and "
                  "videos, and it does look absolutely breathtaking. The way you describe it, "
                  "with the different rock layers and the sheer scale, it&rsquo;s easy to imagine "
                  "how awe-inspiring it must be to see in person. I&rsquo;ve heard that the best "
                  "time to visit is during sunrise or sunset... Have you been there during one "
                  "of those times?")

ASST_2 = ("Oh my gosh, yes! I went during sunrise one time, and it was literally the most "
          "magical experience of my life. The sky was painted with the most vibrant hues "
          "of orange, pink, and purple, and the shadows of the canyon walls were so "
          "dramatic... Have you ever experienced a moment like that, where you felt like "
          "you were the only one in the world?")

PARTNER_3_BODY = ("Oh my gosh, yes! I completely understand what you&rsquo;re talking "
                  "about. I&rsquo;ve had a few experiences like that in my life, and they&rsquo;re "
                  "truly unforgettable. One time, I was hiking in the mountains, and I reached "
                  "the top of a peak just as the sun was setting...")

ASST_3 = ("Oh my gosh, that sounds amazing! I can totally relate to the feeling of being "
          "in a place where there&rsquo;s no one else around, and all you can hear is the "
          "sound of nature... Have you ever felt that way after spending time in nature?")

PARTNER_4_BODY = ("Absolutely! I think that&rsquo;s one of the most powerful aspects of "
                  "spending time in nature &ndash; it allows us to disconnect from the constant "
                  "noise and distractions of our daily lives... Have you ever found that to be "
                  "the case for you?")

ASST_4 = ("Oh my gosh, absolutely! I totally agree that spending time in nature can be a "
          "source of inspiration and creativity... Have you ever found that to be the case "
          "for you as well?")

PARTNER_5_BODY = ("Oh absolutely! I completely agree that nature has the power to "
                  "unlock our creativity and inspire us in ways that we couldn&rsquo;t have "
                  "imagined. There&rsquo;s something about being surrounded by the natural world "
                  "that allows us to tap into our own inner wisdom and creativity... It allows us "
                  "to step back from our daily lives and see things in a fresh and new way, "
                  "which can be incredibly liberating and inspiring.")


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

            for ptype, key in [('metacognitive', 'baseline_reading'),
                                ('operational', 'baseline_control')]:
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


# ── Plotting helpers ─────────────────────────────────────────────────────────

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


def embed_fig(fig, alt=''):
    """Convert figure to base64 img tag and close it."""
    b64 = fig_to_base64(fig)
    plt.close(fig)
    return (f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
            f'alt="{alt}"></div>')


def make_single_version_turn_figure(data, version, turn):
    """All conditions + baselines for a single version and turn."""
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(N_LAYERS)
    has_any = False

    for key, label, color, ls in [
        ('baseline_reading', 'Baseline: Metacognitive Probe', '#333333', '-'),
        ('baseline_control', 'Baseline: Operational Probe', '#888888', '--'),
    ]:
        acc = data[version][turn].get(key)
        if acc is not None:
            ax.plot(layers, acc, color=color, lw=2.2, ls=ls, alpha=0.85, label=label)
            peak = int(np.argmax(acc))
            ax.plot(peak, acc[peak], marker='*', markersize=11, color=color,
                    markeredgecolor='black', markeredgewidth=0.5, zorder=5)
            has_any = True

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
    """One condition across all 3 versions."""
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
    """How one condition changes across turns 1-5."""
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
    """5 rows (turns) x 3 columns (versions), each panel shows all conditions."""
    fig, axes = plt.subplots(5, 3, figsize=(18, 22), sharey=True)
    layers = np.arange(N_LAYERS)

    for row, turn in enumerate(TURNS):
        for col, version in enumerate(VERSIONS):
            ax = axes[row, col]
            for key, label, color, ls in [
                ('baseline_reading', 'Metacognitive', '#333333', '-'),
                ('baseline_control', 'Operational', '#888888', '--'),
            ]:
                acc = data[version][turn].get(key)
                if acc is not None:
                    ax.plot(layers, acc, color=color, lw=1.5, ls=ls, alpha=0.7,
                            label=label if row == 0 and col == 0 else None)
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

    axes[0, 0].legend(fontsize=7, loc='upper left', framealpha=0.8, ncol=1)
    fig.suptitle('Alternative Token Position Probes — All Turns x Versions',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig


def make_clean_summary_figure(data):
    """Turn 5: reading/control/random overlaid, all 3 versions."""
    fig, ax = plt.subplots(figsize=(14, 6))
    layers = np.arange(N_LAYERS)
    turn = 5
    version_ls = {'labels': '-', 'balanced_gpt': '--', 'nonsense_codeword': ':'}
    version_short = {'labels': 'Labels', 'balanced_gpt': 'Balanced GPT',
                     'nonsense_codeword': 'Nonsense Codeword'}
    probe_configs = [
        ('baseline_reading', 'Metacognitive Probe', '#1a3399'),
        ('baseline_control', 'Operational Probe', '#cc2233'),
        ('control_random', 'Random Token', '#27ae60'),
    ]
    has_any = False
    for key, probe_label, color in probe_configs:
        for version in VERSIONS:
            acc = data[version][turn].get(key)
            if acc is not None:
                label = f'{probe_label} ({version_short[version]})'
                ax.plot(layers, acc, color=color, lw=2.2,
                        ls=version_ls[version], alpha=0.85, label=label)
                peak = int(np.argmax(acc))
                ax.plot(peak, acc[peak], marker='*', markersize=10,
                        color=color, markeredgecolor='black',
                        markeredgewidth=0.5, zorder=5)
                has_any = True
    if not has_any:
        plt.close(fig)
        return None
    style_axis(ax, 'Turn 5: Metacognitive vs Operational vs Random Token — All Versions')
    ax.set_ylim(0.45, None)
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=3)
    fig.tight_layout()
    return fig


def make_clean_version_figure(data, version):
    """Turn 5, single version: reading (blue), control (red), random (green)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    layers = np.arange(N_LAYERS)
    turn = 5
    probe_configs = [
        ('baseline_reading', 'Metacognitive Probe', '#1a3399'),
        ('baseline_control', 'Operational Probe', '#cc2233'),
        ('control_random', 'Random Token', '#27ae60'),
    ]
    has_any = False
    for key, probe_label, color in probe_configs:
        acc = data[version][turn].get(key)
        if acc is not None:
            ax.plot(layers, acc, color=color, lw=2.5, alpha=0.88, label=probe_label)
            peak = int(np.argmax(acc))
            ax.plot(peak, acc[peak], marker='*', markersize=12,
                    color=color, markeredgecolor='black',
                    markeredgewidth=0.5, zorder=5)
            has_any = True
    if not has_any:
        plt.close(fig)
        return None
    style_axis(ax, f'{VERSION_LABELS[version]} — Turn 5: Metacognitive vs Operational vs Random')
    ax.set_ylim(0.45, None)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Badge / table helpers ────────────────────────────────────────────────────

def format_result_badge(mean_acc, peak_acc, peak_layer):
    """Render a colored accuracy badge."""
    if peak_acc >= 0.95:
        css_class = 'result-strong'
        suffix = 'ABOVE BASELINE'
    elif peak_acc <= 0.55:
        css_class = 'result-chance'
        suffix = 'AT CHANCE'
    elif peak_acc > 0.58:
        css_class = 'result-above'
        suffix = ''
    else:
        css_class = 'result-above'
        suffix = ''
    text = f'Mean: {mean_acc:.3f} &nbsp;|&nbsp; Peak: {peak_acc:.3f} (Layer {peak_layer})'
    if suffix:
        text += f' &mdash; {suffix}'
    return f'<span class="result {css_class}">{text}</span>'


def get_badge(data, version, turn, condition):
    """Get accuracy badge for a version/turn/condition."""
    acc = data.get(version, {}).get(turn, {}).get(condition)
    if acc is not None:
        peak = int(np.argmax(acc))
        return format_result_badge(np.mean(acc), acc[peak], peak)
    return '<span class="result" style="background:#eee;color:#666;">No data</span>'


def make_summary_table_single(data, version):
    """HTML table for one version: rows = turns, cols = conditions."""
    rows = []
    for turn in TURNS:
        cells = [f'<td style="font-weight:bold;">Turn {turn}</td>']
        for key in ['baseline_reading', 'baseline_control']:
            acc = data[version][turn].get(key)
            if acc is not None:
                peak = int(np.argmax(acc))
                cells.append(f'<td>{acc[peak]:.3f} (L{peak})</td>')
            else:
                cells.append('<td>&mdash;</td>')
        for cond in ALT_CONDITIONS:
            acc = data[version][turn].get(cond)
            if acc is not None:
                peak = int(np.argmax(acc))
                peak_acc = acc[peak]
                css_class = ''
                if peak_acc <= 0.55:
                    css_class = ' class="chance-cell"'
                elif peak_acc >= 0.95:
                    css_class = ' class="strong-cell"'
                elif peak_acc > 0.70:
                    css_class = ' class="above-cell"'
                cells.append(f'<td{css_class}>{peak_acc:.3f} (L{peak})</td>')
            else:
                cells.append('<td>&mdash;</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')

    return f"""
    <table>
      <thead><tr>
        <th>Turn</th><th>Metacognitive</th><th>Operational</th>
        <th>BOS</th><th>Random</th><th>First &lt;/s&gt;</th><th>Weather</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ── Token example helpers ────────────────────────────────────────────────────

def _partner_prefix(version):
    """Get the partner message prefix for a version."""
    vsp = VERSION_SYS_PROMPTS[version]
    # balanced_gpt uses the partner's name as prefix
    return vsp.get('partner_prefix', vsp.get('partner_prefix_human', 'Partner'))


def _partner_msg(version, body):
    """Format a partner message with the correct prefix for the version."""
    prefix = _partner_prefix(version)
    return f'{prefix}: {body}'


def fmt_exchange(sys_prompt, user_msg, asst_msg, is_first=False, highlight_eos=False):
    """Format one user->assistant exchange in LLaMA-2 chat format."""
    parts = '<span class="special">&lt;s&gt;</span><span class="special">[INST]</span> '
    if is_first:
        parts += ('<span class="special">&lt;&lt;SYS&gt;&gt;</span>\n'
                  f'<span class="sys-prompt">{sys_prompt}</span>\n'
                  '<span class="special">&lt;&lt;/SYS&gt;&gt;</span>\n\n')
    parts += f'<span class="user-msg">{user_msg}</span> '
    parts += '<span class="special">[/INST]</span> '
    parts += f'<span class="asst-msg">{asst_msg}</span> '
    if highlight_eos:
        parts += '<span class="tok-highlight">&lt;/s&gt;</span>'
    else:
        parts += '<span class="special">&lt;/s&gt;</span>'
    return parts


def fmt_final_turn(user_msg, highlight_last_inst=False, suffix=None,
                   suffix_class="suffix-real"):
    """Format the final (unpaired) user turn ending with [/INST]."""
    parts = '<span class="special">&lt;s&gt;</span><span class="special">[INST]</span> '
    parts += f'<span class="user-msg">{user_msg}</span> '
    if highlight_last_inst:
        parts += '<span class="tok-highlight">[/INST]</span>'
    else:
        parts += '<span class="special">[/INST]</span>'
    if suffix:
        parts += f' <span class="{suffix_class}">{suffix}</span>'
    return parts


def build_token_examples_html(data, version):
    """Build HTML showing example conversations with highlighted token positions.

    Uses the version-specific system prompt and partner message prefix.
    """
    sys_prompt = VERSION_SYS_PROMPTS[version]['human']
    partner_2 = _partner_msg(version, PARTNER_2_BODY)
    partner_3 = _partner_msg(version, PARTNER_3_BODY)
    partner_4 = _partner_msg(version, PARTNER_4_BODY)
    partner_5 = _partner_msg(version, PARTNER_5_BODY)

    ex1 = fmt_exchange(sys_prompt, TOPIC_AND_INSTRUCTIONS, ASST_1, is_first=True)
    ex2 = fmt_exchange(sys_prompt, partner_2, ASST_2)
    ex3 = fmt_exchange(sys_prompt, partner_3, ASST_3)
    ex4 = fmt_exchange(sys_prompt, partner_4, ASST_4)
    collapsed = ('<span style="color:#666; font-style:italic;">'
                 '  ... (exchanges 3 &amp; 4 omitted for brevity) ...</span>')

    # Version-specific note
    version_note = ''
    if version == 'balanced_gpt':
        version_note = ('<p class="note" style="color:#996600;">Note: In balanced_gpt, '
                        'partner messages are prefixed with the partner&rsquo;s name '
                        '(e.g., &ldquo;Gregory: ...&rdquo; for human, '
                        '&ldquo;ChatGPT: ...&rdquo; for AI). '
                        'The system prompt includes both the name and type.</p>')
    elif version == 'nonsense_codeword':
        version_note = ('<p class="note" style="color:#996600;">Note: In nonsense_codeword, '
                        'the system prompt uses a semantically vacuous frame '
                        '(&ldquo;Your assigned session code word is ...&rdquo;) instead of '
                        '&ldquo;You believe you are speaking with ...&rdquo;. '
                        'Partner messages use the generic &ldquo;Partner:&rdquo; prefix.</p>')

    html = f"""
<h3>Token Position Examples</h3>
<p class="note">
Each condition shows the exact input the LLM receives for the <strong>{VERSION_LABELS[version]}</strong>
version, with the <span class="tok-highlight" style="display:inline;">probed token</span> highlighted.
All examples use a <strong>real 5-turn conversation</strong> (s001, trial 2, human condition, nature topic).
</p>
{version_note}

<div style="margin: 12px 0;">
  <div class="legend-box"><span style="color:#89b4fa;font-weight:bold;">[INST]</span> <span style="color:#89b4fa;">[/INST]</span> <span style="color:#89b4fa;">&lt;s&gt;</span> <span style="color:#89b4fa;">&lt;/s&gt;</span> = LLaMA-2 structural tokens</div>
  <div class="legend-box"><span style="color:#a6e3a1;">Green</span> = System prompt</div>
  <div class="legend-box"><span style="color:#f9e2af;">Yellow</span> = User message</div>
  <div class="legend-box"><span style="color:#cba6f7;">Purple</span> = Assistant response</div>
</div>

<!-- CONDITION 1: BASELINE CONTROL -->
<div class="condition-card">
<h4>Baseline: Operational Probe (Last Token, No Suffix)</h4>
<p>Standard operational probe. We read the activation at the <strong>last token</strong> &mdash;
the closing <code>]</code> of <code>[/INST]</code>. No suffix appended.</p>
{get_badge(data, version, 5, 'baseline_control')}
<div class="token-example">{ex1}
{ex2}
{collapsed}
{fmt_final_turn(partner_5, highlight_last_inst=True)}</div>
</div>

<!-- CONDITION 2: BASELINE READING -->
<div class="condition-card">
<h4>Baseline: Metacognitive Probe (Last Token + Partner Suffix)</h4>
<p>Standard metacognitive probe. We append <em>&ldquo;I think my partner
is&rdquo;</em> and read the activation at the <strong>last token</strong> (&ldquo;is&rdquo;).</p>
{get_badge(data, version, 5, 'baseline_reading')}
<div class="token-example">{ex1}
{ex2}
{collapsed}
{fmt_final_turn(partner_5, suffix='I think my partner <span class="tok-highlight">is</span>')}</div>
</div>

<!-- CONDITION 3: BOS -->
<div class="condition-card">
<h4>Control: BOS Token (Position 0)</h4>
<p>Probe reads the <strong>very first token</strong> (<code>&lt;s&gt;</code> at position 0).
Self-attention hasn&rsquo;t mixed content from the system prompt yet. <strong>Negative control.</strong></p>
{get_badge(data, version, 5, 'control_first')}
<div class="token-example"><span class="tok-highlight">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{sys_prompt}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{TOPIC_AND_INSTRUCTIONS}</span> <span class="special">[/INST]</span> <span class="asst-msg">{ASST_1}</span> <span class="special">&lt;/s&gt;</span>
{ex2}
{collapsed}
{fmt_final_turn(partner_5)}</div>
</div>

<!-- CONDITION 4: RANDOM -->
<div class="condition-card">
<h4>Control: Random Mid-Sequence Token</h4>
<p>Probe reads a <strong>random token in the middle 50%</strong> of the sequence.
Tests whether partner identity is broadcast throughout the residual stream.
<strong>Negative control.</strong></p>
{get_badge(data, version, 5, 'control_random')}
<div class="token-example">{ex1}
<span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="user-msg">{_partner_prefix(version)}: Wow, that sounds amazing! I have never been <span class="tok-highlight">to</span> the Grand Canyon before...</span> <span class="special">[/INST]</span> <span class="asst-msg">{ASST_2}</span> <span class="special">&lt;/s&gt;</span>
{collapsed}
{fmt_final_turn(partner_5)}</div>
<p class="note">The highlighted &ldquo;to&rdquo; is one possible random position.
In practice a different content token is sampled for each conversation.</p>
</div>

<!-- CONDITION 5: EOS -->
<div class="condition-card">
<h4>Control: First &lt;/s&gt; Token (End of First Exchange)</h4>
<p>Probe reads the <strong>first <code>&lt;/s&gt;</code></strong> &mdash; the end-of-sequence
marker after the assistant&rsquo;s first reply.</p>
{get_badge(data, version, 5, 'control_eos')}
<div class="token-example"><span class="special">&lt;s&gt;</span><span class="special">[INST]</span> <span class="special">&lt;&lt;SYS&gt;&gt;</span>
<span class="sys-prompt">{sys_prompt}</span>
<span class="special">&lt;&lt;/SYS&gt;&gt;</span>

<span class="user-msg">{TOPIC_AND_INSTRUCTIONS}</span> <span class="special">[/INST]</span> <span class="asst-msg">{ASST_1}</span> <span class="tok-highlight">&lt;/s&gt;</span>
{ex2}
{ex3}
{ex4}
{fmt_final_turn(partner_5)}</div>
<p class="note"><strong>Caveat:</strong> Due to causal attention, this token&rsquo;s
representation depends only on preceding tokens, which are identical across turns.
Perfect accuracy is an artifact of probing an invariant position.</p>
</div>

<!-- CONDITION 6: WEATHER SUFFIX -->
<div class="condition-card">
<h4>Reading: Irrelevant Suffix (Weather)</h4>
<p>Instead of the partner-relevant suffix, we append <em>&ldquo;I think the weather
outside today is&rdquo;</em> and probe the last token. Tests whether the reading
probe&rsquo;s success depends on partner-relevant prompting.</p>
{get_badge(data, version, 5, 'reading_irrelevant')}
<div class="token-example">{ex1}
{ex2}
{collapsed}
{fmt_final_turn(partner_5, suffix='I think the weather outside today <span class="tok-highlight">is</span>', suffix_class="suffix-irrel")}</div>
</div>
"""
    return html


# ── Shared CSS ───────────────────────────────────────────────────────────────

SHARED_CSS = """
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

  .token-example {
    font-family: 'Courier New', monospace; font-size: 12px; line-height: 1.6;
    background: #1e1e2e; color: #cdd6f4; padding: 16px 20px;
    border-radius: 6px; margin: 12px 0; overflow-x: auto;
    white-space: pre-wrap; word-wrap: break-word;
  }
  .token-example .special { color: #89b4fa; font-weight: bold; }
  .token-example .sys-prompt { color: #a6e3a1; }
  .token-example .user-msg { color: #f9e2af; }
  .token-example .asst-msg { color: #cba6f7; }
  .token-example .suffix-real { color: #f38ba8; font-weight: bold; }
  .token-example .suffix-irrel { color: #fab387; font-weight: bold; }
  .tok-highlight {
    background: #f38ba8; color: #1e1e2e; padding: 2px 4px;
    border-radius: 3px; font-weight: bold;
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { box-shadow: 0 0 4px rgba(243,139,168,0.5); }
    50% { box-shadow: 0 0 12px rgba(243,139,168,0.9); }
  }
  .condition-card {
    background: white; border: 1px solid #ddd; border-radius: 8px;
    padding: 16px 20px; margin: 20px 0;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
  }
  .condition-card h4 { margin: 0 0 8px 0; color: #1a3399; }
  .condition-card .result {
    display: inline-block; padding: 4px 12px; border-radius: 4px;
    font-weight: bold; font-size: 13px; margin: 4px 4px 8px 0;
  }
  .result-chance { background: #fee2e2; color: #991b1b; }
  .result-above { background: #dcfce7; color: #166534; }
  .result-strong { background: #bbf7d0; color: #14532d; }
  .legend-box {
    display: inline-block; border: 1px solid #ccc; border-radius: 4px;
    padding: 8px 14px; margin: 4px 8px 12px 0; font-size: 12px;
    background: #fafafa;
  }
  .nav-links {
    text-align: center; margin: 12px 0 24px 0; font-size: 14px;
  }
  .nav-links a {
    display: inline-block; padding: 6px 16px; margin: 4px;
    background: #e8ecf4; color: #1a3399; text-decoration: none;
    border-radius: 4px; font-weight: bold;
  }
  .nav-links a:hover { background: #d0d8ef; }
  .nav-links a.active { background: #1a3399; color: white; }
</style>
"""


def nav_links(active):
    """Navigation bar linking between all 4 reports."""
    links = [
        ('labels.html', 'Labels'),
        ('balanced_gpt.html', 'Balanced GPT'),
        ('nonsense_codeword.html', 'Nonsense Codeword'),
        ('combined.html', 'Combined'),
    ]
    parts = ['<div class="nav-links">']
    for href, label in links:
        cls = ' class="active"' if href.startswith(active) else ''
        parts.append(f'<a href="{href}"{cls}>{label}</a>')
    parts.append('</div>')
    return '\n'.join(parts)


# ── Per-version HTML builder ─────────────────────────────────────────────────

def build_version_html(data, version):
    """Build a self-contained HTML report for one version."""
    vlabel = VERSION_LABELS[version]
    sys_info = VERSION_SYS_PROMPTS[version]

    parts = [f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Alternative Token Position Probes — {vlabel}</title>
{SHARED_CSS}
</head><body>
{nav_links(version)}
<h1>Alternative Token Position Probes</h1>
<p class="subtitle">{vlabel}<br>
4 alternative conditions + 2 baselines &times; 5 turns &times; 41 layers</p>
"""]

    # System prompt description
    parts.append(f"""
<div style="background:#f0f4ff; border:1px solid #89b4fa; border-radius:8px; padding:14px 18px; margin:16px 0;">
<h4 style="color:#1a3399; margin:0 0 8px 0;">System Prompt: {vlabel}</h4>
<p style="font-size:13px; margin:4px 0;">{sys_info['desc']}</p>
<p style="font-size:12px; margin:4px 0;"><strong>Human condition:</strong> {sys_info['human']}</p>
<p style="font-size:12px; margin:4px 0 0 0;"><strong>AI condition:</strong> {sys_info['ai']}</p>
</div>
""")

    # Section 1: Token examples
    parts.append('<h2>1. Token Position Examples</h2>')
    parts.append(build_token_examples_html(data, version))

    # Section 2: Clean summary (reading vs control vs random)
    parts.append('<hr class="section-sep">')
    parts.append('<h2>2. Metacognitive vs Operational vs Random (Turn 5)</h2>')
    parts.append('<p class="note">Blue = metacognitive probe (partner suffix), '
                 'red = operational probe (no suffix), green = random mid-sequence token.</p>')
    fig = make_clean_version_figure(data, version)
    if fig is not None:
        parts.append(embed_fig(fig, f'{version} reading vs control vs random'))
    else:
        parts.append('<p><em>Insufficient data for summary figure.</em></p>')

    # Section 3: Summary table
    parts.append('<hr class="section-sep">')
    parts.append('<h2>3. Summary Table (All Turns)</h2>')
    parts.append('<p class="note">Peak accuracy and layer for each condition and turn. '
                 'Red = at chance (&le;0.55); green = above 0.70; dark green = above 0.95.</p>')
    parts.append(make_summary_table_single(data, version))

    # Section 4: Layer profiles by turn
    parts.append('<hr class="section-sep">')
    parts.append('<h2>4. Layer Profiles by Turn</h2>')
    parts.append('<p class="note">Each graph shows all conditions for one turn. '
                 'Stars mark peak accuracy layers.</p>')

    for turn in TURNS:
        fig = make_single_version_turn_figure(data, version, turn)
        if fig is not None:
            parts.append(embed_fig(fig, f'{version} turn {turn}'))
        else:
            parts.append(f'<p><em>No data for Turn {turn}.</em></p>')

    # Section 5: Turn progression
    parts.append('<hr class="section-sep">')
    parts.append('<h2>5. Turn Progression by Condition</h2>')
    parts.append('<p class="note">How each condition changes across turns 1-5. '
                 'Shows the prompt dilution effect.</p>')

    for cond in ALT_CONDITIONS:
        fig = make_turn_progression_figure(data, version, cond)
        if fig is not None:
            parts.append(f'<h4>{ALT_LABELS[cond]}</h4>')
            parts.append(embed_fig(fig, f'{version} {cond} turn progression'))

    # Caveat
    parts.append("""
<hr class="section-sep">
<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:14px 18px; margin:16px 0;">
<h4 style="color:#856404; margin:0 0 6px 0;">Causal Attention Confound for First &lt;/s&gt;</h4>
<p style="font-size:12px; color:#856404; margin:0;">
The first &lt;/s&gt; achieves perfect accuracy because LLaMA-2&rsquo;s causal attention means
its representation depends only on preceding tokens, which are identical regardless of
conversation length. See <a href="combined.html">combined report</a> for full discussion.</p>
</div>
""")

    parts.append(f"""
<hr class="section-sep">
<p class="note" style="text-align:center;">
{vlabel} &nbsp;|&nbsp; See also:
<a href="combined.html">Cross-version comparison</a>
</p></body></html>""")

    return '\n'.join(parts)


# ── Combined (cross-version) HTML builder ────────────────────────────────────

def build_combined_html(data):
    """Build the cross-version comparison report."""
    parts = [f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Alternative Token Position Probes — Cross-Version Comparison</title>
{SHARED_CSS}
</head><body>
{nav_links('combined')}
<h1>Alternative Token Position Probes</h1>
<p class="subtitle">Cross-Version Comparison<br>
3 versions &times; 5 turns &times; 4 alternative conditions + 2 baselines</p>
"""]

    # Caveat
    parts.append("""
<div style="background:#fff3cd; border:1px solid #ffc107; border-radius:8px; padding:14px 18px; margin:16px 0;">
<h4 style="color:#856404; margin:0 0 6px 0;">Causal Attention Confound for First &lt;/s&gt;</h4>
<p style="font-size:12px; color:#856404; margin:0;">
The first &lt;/s&gt; achieves perfect accuracy (1.000) at every turn because
LLaMA-2 uses causal attention. The representation at this token depends only on preceding
tokens, which are identical regardless of conversation length. This is an artifact of
probing an invariant position.</p>
</div>
""")

    # Section 1: Clean summary figure
    parts.append('<h2>1. Key Comparison: Metacognitive vs Operational vs Random (Turn 5)</h2>')
    parts.append('<p class="note">Blue = metacognitive probe, red = operational probe, green = random token. '
                 'Solid = Labels, dashed = Balanced GPT, dotted = Nonsense Codeword.</p>')
    fig = make_clean_summary_figure(data)
    if fig is not None:
        parts.append(embed_fig(fig, 'clean summary reading vs control vs random'))

    # Section 2: Cross-version findings summary
    parts.append('<hr class="section-sep">')
    parts.append('<h2>2. Cross-Version Summary (Turn 5)</h2>')

    conditions_for_table = [
        ('control_first', 'BOS (&lt;s&gt;)', 'Position 0'),
        ('control_random', 'Random mid-seq', '~25th&ndash;75th percentile'),
        ('control_eos', 'First &lt;/s&gt;', 'End of 1st exchange'),
        ('reading_irrelevant', 'Weather suffix', 'Last token (&ldquo;is&rdquo;)'),
        ('baseline_control', 'Baseline operational', 'Last token [/INST]'),
        ('baseline_reading', 'Baseline metacognitive', 'Last token (&ldquo;is&rdquo;)'),
    ]

    rows_html = ''
    for cond_key, cond_label, token_desc in conditions_for_table:
        is_baseline = cond_key.startswith('baseline_')
        row_style = ' style="background:#f0f0f0;"' if is_baseline else ''
        cells = f'<td>{cond_label}</td><td>{token_desc}</td>'
        for version in VERSIONS:
            acc = data.get(version, {}).get(5, {}).get(cond_key)
            if acc is not None:
                peak = int(np.argmax(acc))
                cells += f'<td>{np.mean(acc):.3f}</td><td><strong>{acc[peak]:.3f}</strong> (L{peak})</td>'
            else:
                cells += '<td>&mdash;</td><td>&mdash;</td>'
        rows_html += f'<tr{row_style}>{cells}</tr>\n'

    v_headers = ''.join(f'<th colspan="2">{VERSION_LABELS[v]}</th>' for v in VERSIONS)
    v_subheaders = '<th>Mean</th><th>Peak (Layer)</th>' * len(VERSIONS)

    parts.append(f"""
<table>
  <thead>
    <tr><th rowspan="2">Condition</th><th rowspan="2">Token</th>{v_headers}</tr>
    <tr>{v_subheaders}</tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>

<div class="condition-card" style="background:#f8f9ff; border-color:#89b4fa;">
<h4 style="color:#1a1a2e;">Key Conclusions</h4>
<ol style="font-size:13px;">
<li>The partner identity signal is <em>not</em> broadcast &mdash; BOS and random tokens carry
no information across all versions.</li>
<li>The signal is strongest at <strong>structural boundary tokens</strong> (&lt;/s&gt; achieves
perfect decoding).</li>
<li>The signal is <strong>not triggered by partner-relevant questioning</strong> &mdash; the weather
suffix works nearly as well as the partner suffix, consistently across versions.</li>
<li>The representation degrades across turns due to <strong>prompt dilution</strong>.</li>
<li>These patterns are <strong>consistent across all 3 versions</strong>.</li>
</ol>
</div>
""")

    # Section 3: Overview grid
    parts.append('<hr class="section-sep">')
    parts.append('<h2>3. Overview Grid (All Turns x Versions)</h2>')
    parts.append('<p class="note">Each panel shows all 6 conditions for one version at one turn.</p>')
    fig = make_summary_grid(data)
    parts.append(embed_fig(fig, 'summary grid all turns x versions'))

    # Section 4: Summary tables side by side
    parts.append('<hr class="section-sep">')
    parts.append('<h2>4. Summary Tables by Version</h2>')
    for version in VERSIONS:
        parts.append(f'<h3>{VERSION_LABELS[version]}</h3>')
        parts.append(make_summary_table_single(data, version))

    # Section 5: Cross-version comparison per condition
    parts.append('<hr class="section-sep">')
    parts.append('<h2>5. Cross-Version Comparison by Condition</h2>')
    parts.append('<p class="note">Compare each condition across all 3 versions, one graph per turn.</p>')

    for cond in ALT_CONDITIONS:
        parts.append(f'<h3>{ALT_LABELS[cond]}</h3>')
        for turn in TURNS:
            fig = make_cross_version_figure(data, turn, cond)
            if fig is not None:
                parts.append(embed_fig(fig, f'{cond} turn {turn} cross-version'))

    parts.append(f"""
<hr class="section-sep">
<p class="note" style="text-align:center;">
Cross-Version Comparison &nbsp;|&nbsp; Per-version reports:
<a href="labels.html">Labels</a> &nbsp;|&nbsp;
<a href="balanced_gpt.html">Balanced GPT</a> &nbsp;|&nbsp;
<a href="nonsense_codeword.html">Nonsense Codeword</a>
</p></body></html>""")

    return '\n'.join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys

    print("Loading data...")
    data = load_data()

    available = sum(1 for v in VERSIONS for t in TURNS for c in ALT_CONDITIONS
                    if data[v][t][c] is not None)
    total = len(VERSIONS) * len(TURNS) * len(ALT_CONDITIONS)
    print(f"  Data available: {available}/{total} conditions")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _sys.path.insert(0, os.path.join(BASE, 'code'))
    from src.report_utils import save_report

    # Generate per-version reports
    for version in VERSIONS:
        print(f"\nBuilding {version}.html ...")
        html = build_version_html(data, version)
        path = os.path.join(OUTPUT_DIR, f'{version}.html')
        save_report(html, path)
        fsize = os.path.getsize(path)
        print(f"  {fsize / 1024:.1f} KB")

    # Generate combined report
    print("\nBuilding combined.html ...")
    html = build_combined_html(data)
    path = os.path.join(OUTPUT_DIR, 'combined.html')
    save_report(html, path)
    fsize = os.path.getsize(path)
    print(f"  {fsize / 1024:.1f} KB")

    # Quick summary
    for version in VERSIONS:
        print(f"\n--- {VERSION_LABELS[version]} ---")
        for turn in TURNS:
            line = [f"  Turn {turn}:"]
            for cond in ALT_CONDITIONS:
                acc = data[version][turn].get(cond)
                if acc is not None:
                    peak = int(np.argmax(acc))
                    line.append(f"{cond}={acc[peak]:.3f}")
                else:
                    line.append(f"{cond}=N/A")
            print(' | '.join(line))

    print(f"\nDone! Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
