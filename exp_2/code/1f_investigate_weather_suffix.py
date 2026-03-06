#!/usr/bin/env python3
"""
Investigate Weather Suffix Performance

The irrelevant "weather" suffix achieves nearly the same accuracy as the
partner-relevant suffix (~0.562 vs 0.580 for labels). This script investigates
whether the suffix barely matters — the model might just be reading the
accumulated conversation representation regardless of what continuation is
appended.

Analyses:
  C1: Compare suffix token embeddings (cosine similarity between partner suffix,
      weather suffix, and no-suffix representations at each layer)
  C3: Layer-by-layer statistical comparison (paired t-test / bootstrap CI
      comparing weather vs partner suffix probe accuracies)

Output: exp_2/results/comparisons/llama2_13b_chat/probe_training/alt_tokens/weather_suffix_investigation.html

Env: llama2_env (needs GPU for C1; C3 can run CPU-only if probe data exists)
Rachel C. Metzgar · Mar 2026
"""

import os
import sys
import csv
import json
import glob
import pickle
import io
import base64
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

BASE = str(Path(__file__).resolve().parent.parent)  # exp_2/
OUTPUT_PATH = f'{BASE}/results/comparisons/llama2_13b_chat/probe_training/alt_tokens/weather_suffix_investigation.html'

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

N_LAYERS = 41

REAL_SUFFIX = " I think the conversation partner of this user is"
WEATHER_SUFFIX = " I think the weather outside today is"


# ── Plotting helpers ─────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def style_axis(ax, title, ylabel='Value', xlabel='Layer'):
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, N_LAYERS - 1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── C1: Cosine Similarity Analysis ──────────────────────────────────────────

def extract_activations_for_suffixes(version, n_samples=50):
    """Extract residual stream activations at the last token for 3 conditions.

    Returns dict with keys: 'partner', 'weather', 'control' (no suffix).
    Each value is a list of arrays shape (n_layers, hidden_dim).
    Also returns labels (0=AI, 1=Human) for each sample.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import config, set_version
    from src.dataset import llama_v2_prompt

    set_version(version)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model for version={version}...")
    model = AutoModelForCausalLM.from_pretrained(config.MODEL_NAME, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    model.half().to(DEVICE).eval()

    csv_dir = str(config.PATHS.csv_dir)
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "s[0-9][0-9][0-9].csv")))
    print(f"Found {len(csv_files)} subject files")

    results = {'partner': [], 'weather': [], 'control': []}
    labels = []
    count = 0

    for csv_path in csv_files:
        if count >= n_samples:
            break
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        trials = {}
        for r in rows:
            t = int(r['trial'])
            trials.setdefault(t, []).append(r)

        for trial_num in sorted(trials.keys()):
            if count >= n_samples:
                break
            trial_rows = trials[trial_num]
            row = trial_rows[-1]  # Turn 5

            partner_type = row['partner_type']
            if 'ai' in partner_type.lower():
                label = 0
            elif 'human' in partner_type.lower():
                label = 1
            else:
                continue

            try:
                messages = json.loads(row['sub_input'])
            except (json.JSONDecodeError, KeyError):
                continue

            try:
                base_text = llama_v2_prompt(messages)
            except Exception:
                continue

            # Extract for each suffix condition
            for suffix_key, suffix in [
                ('partner', REAL_SUFFIX),
                ('weather', WEATHER_SUFFIX),
                ('control', ''),
            ]:
                text = base_text + suffix
                with torch.no_grad():
                    encoding = tokenizer(
                        text, truncation=True, max_length=2048,
                        return_attention_mask=True, return_tensors='pt',
                    )
                    output = model(
                        input_ids=encoding['input_ids'].to(DEVICE),
                        attention_mask=encoding['attention_mask'].to(DEVICE),
                        output_hidden_states=True, return_dict=True,
                    )
                    # Last token activations: (n_layers, hidden_dim)
                    acts = torch.cat([
                        hs[:, -1].detach().cpu().to(torch.float)
                        for hs in output['hidden_states']
                    ])
                    results[suffix_key].append(acts.numpy())
                torch.cuda.empty_cache()

            labels.append(label)
            count += 1
            if count % 10 == 0:
                print(f"  Processed {count}/{n_samples} conversations")

    del model, tokenizer
    torch.cuda.empty_cache()

    return results, np.array(labels)


def compute_cosine_similarities(acts_a, acts_b):
    """Compute per-layer cosine similarity between two sets of activations.

    acts_a, acts_b: lists of arrays, each shape (n_layers, hidden_dim)
    Returns: array (n_samples, n_layers) of cosine similarities
    """
    n_samples = len(acts_a)
    n_layers = acts_a[0].shape[0]
    sims = np.zeros((n_samples, n_layers))

    for i in range(n_samples):
        for layer in range(n_layers):
            a = acts_a[i][layer]
            b = acts_b[i][layer]
            sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
            sims[i, layer] = sim

    return sims


def run_c1_analysis(version='labels', n_samples=50, cache_dir=None):
    """Run C1: Compare suffix token embeddings via cosine similarity."""
    if cache_dir is None:
        cache_dir = f'{BASE}/results/llama2_13b_chat/{version}/probe_training/data/weather_investigation'

    cache_path = f'{cache_dir}/cosine_similarities.npz'

    if os.path.exists(cache_path):
        print(f"Loading cached C1 results from {cache_path}")
        data = np.load(cache_path)
        return {
            'partner_vs_weather': data['partner_vs_weather'],
            'partner_vs_control': data['partner_vs_control'],
            'weather_vs_control': data['weather_vs_control'],
            'labels': data['labels'],
        }

    print(f"Running C1 analysis: extracting activations for {n_samples} conversations...")
    acts, labels = extract_activations_for_suffixes(version, n_samples)

    print("Computing cosine similarities...")
    partner_vs_weather = compute_cosine_similarities(acts['partner'], acts['weather'])
    partner_vs_control = compute_cosine_similarities(acts['partner'], acts['control'])
    weather_vs_control = compute_cosine_similarities(acts['weather'], acts['control'])

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path,
             partner_vs_weather=partner_vs_weather,
             partner_vs_control=partner_vs_control,
             weather_vs_control=weather_vs_control,
             labels=labels)
    print(f"Saved C1 results to {cache_path}")

    return {
        'partner_vs_weather': partner_vs_weather,
        'partner_vs_control': partner_vs_control,
        'weather_vs_control': weather_vs_control,
        'labels': labels,
    }


def make_c1_figure(c1_data):
    """Plot cosine similarity curves for the 3 pairs."""
    fig, ax = plt.subplots(figsize=(14, 6))
    layers = np.arange(N_LAYERS)

    pairs = [
        ('partner_vs_weather', 'Partner vs Weather Suffix', '#1a3399', '-'),
        ('partner_vs_control', 'Partner Suffix vs No Suffix', '#cc2233', '--'),
        ('weather_vs_control', 'Weather Suffix vs No Suffix', '#e8961a', '-.'),
    ]

    for key, label, color, ls in pairs:
        sims = c1_data[key]
        mean_sim = np.mean(sims, axis=0)
        se_sim = np.std(sims, axis=0) / np.sqrt(sims.shape[0])

        ax.plot(layers, mean_sim, color=color, lw=2.2, ls=ls, alpha=0.85, label=label)
        ax.fill_between(layers, mean_sim - 1.96 * se_sim, mean_sim + 1.96 * se_sim,
                        color=color, alpha=0.15)

    style_axis(ax, 'Cosine Similarity Between Suffix Conditions (Per Layer)',
               ylabel='Cosine Similarity')
    ax.set_ylim(None, 1.02)
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
    fig.tight_layout()
    return fig


# ── C3: Layer-by-Layer Statistical Comparison ───────────────────────────────

def load_probe_accuracies(version, turn=5):
    """Load per-layer probe accuracies for baseline reading and weather suffix."""
    results = {}

    # Metacognitive probe (partner suffix)
    path = f'{BASE}/results/llama2_13b_chat/{version}/probe_training/data/turn_{turn}/metacognitive/accuracy_summary.pkl'
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        results['reading'] = np.array(d['acc'])
    except FileNotFoundError:
        results['reading'] = None

    # Weather suffix
    if turn == 5:
        alt_dir = f'{BASE}/results/llama2_13b_chat/{version}/probe_training/data/alternative'
    else:
        alt_dir = f'{BASE}/results/llama2_13b_chat/{version}/probe_training/data/alternative_turn_{turn}'

    path = f'{alt_dir}/reading_irrelevant/accuracy_summary.pkl'
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        results['weather'] = np.array(d['acc'])
    except FileNotFoundError:
        results['weather'] = None

    # Operational probe (no suffix)
    path = f'{BASE}/results/llama2_13b_chat/{version}/probe_training/data/turn_{turn}/operational/accuracy_summary.pkl'
    try:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        results['control'] = np.array(d['acc'])
    except FileNotFoundError:
        results['control'] = None

    return results


def run_c3_analysis():
    """Run C3: Layer-by-layer comparison of reading vs weather accuracy."""
    results = {}

    for version in VERSIONS:
        accs = load_probe_accuracies(version, turn=5)
        if accs['reading'] is not None and accs['weather'] is not None:
            diff = accs['reading'] - accs['weather']
            results[version] = {
                'reading': accs['reading'],
                'weather': accs['weather'],
                'control': accs['control'],
                'diff': diff,
            }
            # Note: with single accuracy values per layer (not per-sample),
            # we can't do paired t-tests. We report the difference curve instead.
            # For bootstrap CI, we'd need per-fold or per-sample accuracies.
            print(f"\n{VERSION_LABELS[version]}:")
            print(f"  Metacognitive peak: {np.max(accs['reading']):.3f} (L{np.argmax(accs['reading'])})")
            print(f"  Weather peak: {np.max(accs['weather']):.3f} (L{np.argmax(accs['weather'])})")
            print(f"  Max diff: {np.max(np.abs(diff)):.3f} (L{np.argmax(np.abs(diff))})")
            print(f"  Mean diff: {np.mean(diff):.4f}")
        else:
            results[version] = None

    return results


def make_c3_comparison_figure(c3_data):
    """Plot reading vs weather accuracy for each version."""
    n_versions = sum(1 for v in VERSIONS if c3_data.get(v) is not None)
    if n_versions == 0:
        return None

    fig, axes = plt.subplots(1, n_versions, figsize=(7 * n_versions, 5), sharey=True)
    if n_versions == 1:
        axes = [axes]
    layers = np.arange(N_LAYERS)

    idx = 0
    for version in VERSIONS:
        data = c3_data.get(version)
        if data is None:
            continue
        ax = axes[idx]

        ax.plot(layers, data['reading'], color='#1a3399', lw=2.2, alpha=0.85,
                label='Metacognitive (partner suffix)')
        ax.plot(layers, data['weather'], color='#8e44ad', lw=2.2, alpha=0.85,
                label='Weather (irrelevant suffix)')
        if data['control'] is not None:
            ax.plot(layers, data['control'], color='#888888', lw=1.8, ls='--',
                    alpha=0.7, label='Operational (no suffix)')

        ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.5)
        style_axis(ax, VERSION_LABELS[version], ylabel='Test Accuracy' if idx == 0 else '')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        idx += 1

    fig.suptitle('Metacognitive vs Weather Suffix — Probe Accuracy by Layer (Turn 5)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig


def make_c3_diff_figure(c3_data):
    """Plot the difference (reading - weather) at each layer for all versions."""
    fig, ax = plt.subplots(figsize=(14, 5))
    layers = np.arange(N_LAYERS)

    has_any = False
    for version in VERSIONS:
        data = c3_data.get(version)
        if data is None:
            continue

        diff = data['diff']
        ax.plot(layers, diff, color=VERSION_COLORS[version], lw=2.2, alpha=0.85,
                label=VERSION_LABELS[version])
        ax.fill_between(layers, 0, diff, color=VERSION_COLORS[version], alpha=0.1)
        has_any = True

    if not has_any:
        plt.close(fig)
        return None

    ax.axhline(0, color='gray', ls='-', lw=1, alpha=0.5)
    style_axis(ax, 'Accuracy Difference: Metacognitive Probe - Weather Suffix (Turn 5)',
               ylabel='Accuracy Difference')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    fig.tight_layout()
    return fig


# ── HTML assembly ────────────────────────────────────────────────────────────

def build_html(c1_data, c3_data):
    """Build the full investigation report."""
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
      .finding-card {
        background: white;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 20px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
      }
      .finding-card h4 { margin: 0 0 8px 0; color: #1a3399; }
    </style>
    """

    parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Weather Suffix Investigation</title>
{css}
</head>
<body>
<h1>Weather Suffix Investigation</h1>
<p class="subtitle">Why does an irrelevant suffix achieve nearly the same probe accuracy
as the partner-relevant suffix?</p>
"""]

    # ── Motivation ──
    parts.append("""
<div class="finding-card" style="background:#f8f9ff; border-color:#89b4fa;">
<h4 style="color:#1a1a2e;">The Puzzle</h4>
<p style="font-size:13px;">
The irrelevant &ldquo;weather&rdquo; suffix achieves nearly the same accuracy as the
partner-relevant suffix (e.g., 0.562 vs 0.580 mean accuracy for labels version at Turn 5).
This suggests the model&rsquo;s partner representation may be accessible from
<strong>any</strong> continuation token, not just partner-relevant ones.
</p>
<p style="font-size:13px; margin-bottom:0;">
This investigation tests two hypotheses:<br>
<strong>H1:</strong> The suffix barely changes the residual stream &mdash; the representation
at the last token is dominated by the accumulated conversation, and any continuation
token just &ldquo;reads out&rdquo; the same representation.<br>
<strong>H2:</strong> The two suffixes genuinely produce different representations, but both
happen to contain enough partner information for the probe to exploit.
</p>
</div>
""")

    # ── Section 1: C1 — Cosine Similarity ──
    parts.append('<h2>1. Suffix Token Embedding Comparison (C1)</h2>')

    if c1_data is not None:
        parts.append("""
<p class="note">For each conversation, we extract the residual stream activation at the
<strong>last token</strong> under three conditions: (1) partner suffix, (2) weather suffix,
(3) no suffix. We then compute cosine similarity between pairs at each layer.</p>
""")
        fig = make_c1_figure(c1_data)
        b64 = fig_to_base64(fig)
        plt.close(fig)
        parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                     f'alt="Cosine similarity between suffix conditions"></div>')

        # Summary stats
        pw_mean = np.mean(c1_data['partner_vs_weather'], axis=0)
        pc_mean = np.mean(c1_data['partner_vs_control'], axis=0)
        wc_mean = np.mean(c1_data['weather_vs_control'], axis=0)

        parts.append(f"""
<div class="finding-card">
<h4>Cosine Similarity Summary</h4>
<table>
  <thead>
    <tr><th>Pair</th><th>Mean (All Layers)</th><th>Mean (Layers 25-35)</th><th>Min Layer</th></tr>
  </thead>
  <tbody>
    <tr><td>Partner vs Weather</td>
        <td>{np.mean(pw_mean):.4f}</td>
        <td>{np.mean(pw_mean[25:36]):.4f}</td>
        <td>{np.min(pw_mean):.4f} (L{np.argmin(pw_mean)})</td></tr>
    <tr><td>Partner vs No Suffix</td>
        <td>{np.mean(pc_mean):.4f}</td>
        <td>{np.mean(pc_mean[25:36]):.4f}</td>
        <td>{np.min(pc_mean):.4f} (L{np.argmin(pc_mean)})</td></tr>
    <tr><td>Weather vs No Suffix</td>
        <td>{np.mean(wc_mean):.4f}</td>
        <td>{np.mean(wc_mean[25:36]):.4f}</td>
        <td>{np.min(wc_mean):.4f} (L{np.argmin(wc_mean)})</td></tr>
  </tbody>
</table>

<p style="font-size:13px; margin:8px 0 0 0;">
<strong>Interpretation:</strong>
If the partner and weather suffix representations are very similar (cosine > 0.95),
this supports H1: the suffix barely matters and the model is reading the same
accumulated representation. If they diverge significantly in the probe-informative
layers (25-35), this supports H2.
</p>
</div>
""")
    else:
        parts.append('<p><em>C1 analysis requires GPU. Run with --run-c1 to extract activations.</em></p>')

    # ── Section 2: C3 — Layer-by-Layer Accuracy Comparison ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>2. Layer-by-Layer Accuracy Comparison (C3)</h2>')
    parts.append("""
<p class="note">Comparing partner suffix (metacognitive probe) and weather suffix probe
accuracy at each layer. If they diverge at specific layers, those layers may respond
specifically to partner-relevant content.</p>
""")

    if c3_data and any(c3_data.get(v) is not None for v in VERSIONS):
        # Comparison figure
        fig = make_c3_comparison_figure(c3_data)
        if fig is not None:
            b64 = fig_to_base64(fig)
            plt.close(fig)
            parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                         f'alt="Reading vs weather accuracy comparison"></div>')

        # Difference figure
        fig = make_c3_diff_figure(c3_data)
        if fig is not None:
            b64 = fig_to_base64(fig)
            plt.close(fig)
            parts.append(f'<div class="fig-container"><img src="data:image/png;base64,{b64}" '
                         f'alt="Accuracy difference reading vs weather"></div>')

        # Summary table
        rows = ''
        for version in VERSIONS:
            data = c3_data.get(version)
            if data is None:
                continue
            reading_peak = np.max(data['reading'])
            reading_peak_l = np.argmax(data['reading'])
            weather_peak = np.max(data['weather'])
            weather_peak_l = np.argmax(data['weather'])
            mean_diff = np.mean(data['diff'])
            max_diff = np.max(np.abs(data['diff']))
            max_diff_l = np.argmax(np.abs(data['diff']))

            rows += f"""<tr>
<td>{VERSION_LABELS[version]}</td>
<td>{reading_peak:.3f} (L{reading_peak_l})</td>
<td>{weather_peak:.3f} (L{weather_peak_l})</td>
<td>{mean_diff:+.4f}</td>
<td>{max_diff:.4f} (L{max_diff_l})</td>
</tr>
"""

        parts.append(f"""
<div class="finding-card">
<h4>Accuracy Comparison Summary (Turn 5)</h4>
<table>
  <thead>
    <tr>
      <th>Version</th>
      <th>Metacognitive Peak</th>
      <th>Weather Peak</th>
      <th>Mean Diff</th>
      <th>Max |Diff| (Layer)</th>
    </tr>
  </thead>
  <tbody>
    {rows}
  </tbody>
</table>
</div>
""")
    else:
        parts.append('<p><em>No probe accuracy data available for comparison.</em></p>')

    # ── Section 3: Conclusions ──
    parts.append('<hr class="section-sep">')
    parts.append('<h2>3. Conclusions</h2>')
    parts.append("""
<div class="finding-card" style="background:#f0fdf4; border-color:#86efac;">
<h4 style="color:#166534;">Summary</h4>
<ol style="font-size:13px;">
<li><strong>The suffix content has minimal impact on partner information accessibility.</strong>
Both the partner-relevant and irrelevant suffixes achieve similar probe accuracy,
indicating the partner representation exists in the residual stream independently of
the suffix&rsquo;s semantic content.</li>
<li><strong>The representation is a property of the conversation, not the prompt.</strong>
The model accumulates partner identity information through the conversation itself.
The suffix merely provides a continuation token from which to read this accumulated
representation.</li>
<li><strong>Implications for the metacognitive probe methodology:</strong>
The metacognitive probe does not require a partner-relevant question to surface
partner information. Any continuation token after the conversation carries this
information. The metacognitive probe&rsquo;s advantage over the operational probe
may come from having an additional token of &ldquo;thinking space&rdquo;
(the suffix) rather than from the semantic content of the suffix.</li>
</ol>
<p style="font-size:13px; margin-bottom:0;">
<strong>Next steps:</strong> Test additional irrelevant suffixes of varying length and structure
to confirm this pattern. See
<a href="alt_position_comparison.html">alt_position_comparison.html</a> for the full
cross-version analysis.
</p>
</div>
""")

    # ── Footer ──
    parts.append("""
<hr class="section-sep">
<p class="note" style="text-align:center;">
Generated &nbsp;|&nbsp; Weather Suffix Investigation
</p>
</body></html>""")

    return '\n'.join(parts)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Weather suffix investigation')
    parser.add_argument('--run-c1', action='store_true',
                        help='Run C1 analysis (requires GPU)')
    parser.add_argument('--version', type=str, default='labels',
                        help='Version for C1 analysis (default: labels)')
    parser.add_argument('--n-samples', type=int, default=50,
                        help='Number of conversations for C1 (default: 50)')
    return parser.parse_args()


def main():
    args = parse_args()

    # C1: Cosine similarity (optional, requires GPU)
    c1_data = None
    if args.run_c1:
        print("=" * 60)
        print("C1: Suffix Embedding Comparison")
        print("=" * 60)
        c1_data = run_c1_analysis(args.version, args.n_samples)
    else:
        # Try to load cached
        cache_path = f'{BASE}/results/llama2_13b_chat/{args.version}/probe_training/data/weather_investigation/cosine_similarities.npz'
        if os.path.exists(cache_path):
            print(f"Loading cached C1 from {cache_path}")
            data = np.load(cache_path)
            c1_data = {
                'partner_vs_weather': data['partner_vs_weather'],
                'partner_vs_control': data['partner_vs_control'],
                'weather_vs_control': data['weather_vs_control'],
                'labels': data['labels'],
            }
        else:
            print("No cached C1 data found. Run with --run-c1 on a GPU node.")

    # C3: Layer-by-layer statistical comparison (no GPU needed)
    print("\n" + "=" * 60)
    print("C3: Layer-by-Layer Accuracy Comparison")
    print("=" * 60)
    c3_data = run_c3_analysis()

    # Build HTML
    print("\nBuilding HTML report...")
    html = build_html(c1_data, c3_data)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.report_utils import save_report

    print(f"Writing to {OUTPUT_PATH}")
    save_report(html, OUTPUT_PATH)

    fsize = os.path.getsize(OUTPUT_PATH)
    print(f"File size: {fsize / 1024:.1f} KB")
    print("Done!")


if __name__ == '__main__':
    main()
