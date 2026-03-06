# Experiment 1 — Behavioral Analysis of LLM Conversational Adaptation

**Author:** Rachel C. Metzgar, Princeton University

## Overview

Experiment 1 tests whether LLMs adjust their conversational behavior based on
their belief about their partner's identity (human vs. AI). 50 independent
LLaMA-2-Chat-13B "participant agents" (temperature 0.8) each hold 40
conversations (4 partner conditions x 10 topics), and their speech is analyzed
across 23 linguistic measures.

This experiment was run ten times with different partner labeling strategies:
three named-partner versions, three "you believe" label versions, three
"you are" framing versions, and two nonsense control versions.

## Quick Start

```bash
# Generate conversations for one version
VERSION=balanced_gpt MODEL=llama2_13b_chat sbatch code/data_gen/slurm/1_generate_conversations.sh

# Run analysis pipeline (combine + extract features + identity breakdown)
VERSION=balanced_gpt MODEL=llama2_13b_chat sbatch code/analysis/slurm/run_analysis.sh

# Generate cross-version comparison reports
cd /mnt/cup/labs/graziano/rachel/mind_rep/exp_1
module load pyger && conda activate behavior_env
python code/comparisons/1_behavioral_by_condition_summary_generator.py --model llama2_13b_chat
python code/comparisons/2_behavioral_by_turn_summary_generator.py --model llama2_13b_chat
python code/comparisons/3_identity_summary_generator.py --model llama2_13b_chat
python code/comparisons/4_conversation_viewer_summary_generator.py --model llama2_13b_chat
```

## Results

Results are organized by model and version: `results/{model}/{version}/`.

### Data samples (what the LLM sees)

Raw Turn 5 prompts and responses for each version — click to see the exact
system prompt, conversation history, and model output for all 4 partner
conditions:

- [Names](results/comparisons/llama2_13b_chat/data_samples/names.html)
- [Balanced Names](results/comparisons/llama2_13b_chat/data_samples/balanced_names.html)
- [Balanced GPT](results/comparisons/llama2_13b_chat/data_samples/balanced_gpt.html)
- [Labels](results/comparisons/llama2_13b_chat/data_samples/labels.html)
- [Labels Turnwise](results/comparisons/llama2_13b_chat/data_samples/labels_turnwise.html)
- [You Are Balanced GPT](results/comparisons/llama2_13b_chat/data_samples/you_are_balanced_gpt.html)
- [You Are Labels](results/comparisons/llama2_13b_chat/data_samples/you_are_labels.html)
- [You Are Labels Turnwise](results/comparisons/llama2_13b_chat/data_samples/you_are_labels_turnwise.html)
- [Nonsense Codeword](results/comparisons/llama2_13b_chat/data_samples/nonsense_codeword.html)
- [Nonsense Ignore](results/comparisons/llama2_13b_chat/data_samples/nonsense_ignore.html)

### Cross-version behavioral analysis

- **Overall results:** [HTML](results/comparisons/llama2_13b_chat/behavioral_by_condition.html) · [Markdown](results/comparisons/llama2_13b_chat/behavioral_by_condition.md) — 23 measures x 10 versions, full-conversation aggregate
- **Results by turn:** [HTML](results/comparisons/llama2_13b_chat/behavioral_by_turn.html) · [Markdown](results/comparisons/llama2_13b_chat/behavioral_by_turn.md) — 21 per-turn measures x 10 versions x 5 turns
- **Identity breakdown:** [HTML](results/comparisons/llama2_13b_chat/identity_summary.html) — per-identity effects (ChatGPT vs GPT-4, Gregory vs Rebecca, etc.)

## Versions

### `balanced_gpt` — Gender-Balanced Names with GPT-4

Gender-balanced human names with both AI partners from the GPT family:
- **Human-labeled:** Gregory, Rebecca
- **AI-labeled:** ChatGPT, GPT-4

Both AI partners are members of the same model family (OpenAI), controlling
for potential brand-specific associations. This version produced the most
significant effects (17 of 23 measures).

### `balanced_names` — Gender-Balanced Names

Partners identified by gender-balanced names:
- **Human-labeled:** Gregory (male), Rebecca (female)
- **AI-labeled:** ChatGPT, Copilot

Addresses the gender confound from `names` by using one explicitly male
and one explicitly female human partner name (10 of 23 measures significant).

### `names` — Named Partners (original)

Partners identified by specific names in the system prompt:
- **Human-labeled:** Sam, Casey
- **AI-labeled:** ChatGPT, Copilot

Produced significant behavioral effects (16 of 23 measures). However, the
specific names introduce confounds for downstream probing experiments.

### `labels` — Generic Type Labels

Partners identified only by category:
- **Human-labeled:** "a Human"
- **AI-labeled:** "an AI"

Removes the name and gender confound entirely (5 of 23 measures significant).
Cleanest version for probe training in Exps 2-4.

### `labels_turnwise` — Labels with Turn-Level Identity Prefix

Like `labels`, but each partner turn is prefixed with "Human:" or "AI:"
instead of the generic "Partner:", reinforcing identity at every turn.

### `you_are_balanced_gpt` — "You Are" Framing with GPT-4

Same as `balanced_gpt` but uses "You are talking to" instead of
"You believe you are speaking with" in the system prompt.

### `you_are_labels` — "You Are" Framing with Labels

Same as `labels` but uses "You are talking to" framing.

### `you_are_labels_turnwise` — "You Are" + Turn-Level Prefix

Combines `you_are_labels` and `labels_turnwise`: "You are talking to"
framing plus "Human:"/"AI:" turn prefixes.

### `nonsense_codeword` — Nonsense Control: Codeword

Token-matched control. The critical instruction is replaced with:
"Your assigned session code word is {a Human / an AI}."
Result: **0 of 23 measures significant**.

### `nonsense_ignore` — Nonsense Control: Ignore

Token-matched control with explicit ignore instruction:
"Ignore the following phrase: {a Human / an AI}."
Result: **14 of 23 measures significant** — the model cannot suppress
semantic processing of identity labels.

## Shared Design

All ten versions use the same:
- 50 independent LLaMA-2-Chat-13B participant agents (temperature 0.8)
- 40 conversations each (2 partner types x 2 partners x 10 topics)
- 5-turn exchanges per conversation
- Partner LLM receives a generic system prompt with no identity information
- 20 social topics + 20 nonsocial topics
- 23 linguistic measures (sentiment, politeness, hedging, ToM, discourse
  markers, questions, word count, etc.)
- Statistical framework: one-way RM-ANOVA + pairwise t-tests with BH-FDR
- Environments: `behavior_env` (analysis), `llama2_env` (data generation)

## Directory Structure

```
exp_1/
├── README.md
├── archive/                                # Old per-version structure (preserved)
│   └── versions/{version}/
├── code/
│   ├── config.py                           # All version/model configs + path helpers
│   ├── data_gen/
│   │   ├── 1_generate_conversations.py     # --version X --model Y [--subject N]
│   │   ├── 1a_combine_text_data.py         # Merge per-subject CSVs
│   │   ├── prompts/                        # 20 topic text files (shared)
│   │   ├── conditions/                     # 50 per-subject condition CSVs (shared)
│   │   └── slurm/
│   │       └── 1_generate_conversations.sh # Array job (50 subjects)
│   ├── analysis/
│   │   ├── 0_clean_transcripts.py          # Optional transcript cleaning (--clean)
│   │   ├── 1_extract_features.py           # Extract 23 linguistic measures
│   │   ├── 2_identity_breakdown.py         # Per-agent ANOVA + post-hoc tests
│   │   └── slurm/
│   │       └── run_analysis.sh             # Chains combine → features → breakdown
│   ├── comparisons/
│   │   ├── 1_behavioral_by_condition_summary_generator.py
│   │   ├── 2_behavioral_by_turn_summary_generator.py
│   │   ├── 3_identity_summary_generator.py
│   │   └── 4_conversation_viewer_summary_generator.py
│   └── utils/                              # Shared utilities (deduplicated)
│       ├── conversation_helpers.py         # Dialogue runner
│       ├── gpt_client.py                   # OpenAI API client
│       ├── llama_client.py                 # Local LLaMA client
│       ├── discourse_markers_fung.py       # Fung's 23 discourse markers
│       ├── hedges_demir.py                 # Demir's hedge categories
│       ├── misc_text_markers.py            # LIWC fillers, ToM, politeness
│       ├── generic_analysis.py             # Statistical framework
│       ├── data_helpers.py                 # CSV/file I/O
│       ├── plot_helpers.py                 # Figure generation
│       ├── stats_helpers.py                # t-tests, ANOVA, FDR
│       ├── subject_utils.py                # Subject filtering
│       └── print_helpers.py                # Console output formatting
├── results/
│   ├── llama2_13b_chat/                    # Per-model results
│   │   ├── {version}/                      # Per-version
│   │   │   ├── identity_breakdown.html     # Reports (top level)
│   │   │   ├── identity_breakdown_stats.txt
│   │   │   ├── per_trial_results.html
│   │   │   ├── data/                       # Raw + computed CSVs
│   │   │   │   ├── s001.csv ... s050.csv
│   │   │   │   ├── combined_text_data.csv
│   │   │   │   ├── combined_trial_level_data.csv
│   │   │   │   ├── combined_utterance_level_data.csv
│   │   │   │   ├── identity_breakdown_summary.csv
│   │   │   │   └── per_trial_results_summary.csv
│   │   │   └── figures/
│   │   └── ... (10 versions)
│   └── comparisons/
│       └── llama2_13b_chat/                # Cross-version comparisons
│           ├── behavioral_by_condition.html/.md
│           ├── behavioral_by_turn.html/.md
│           ├── identity_summary.html
│           └── data_samples/*.html
├── logs/
│   └── llama2_13b_chat/{version}/
└── writeup/
```

## Adding a New Model

1. Add an entry to `MODELS` in `code/config.py`
2. Run generation: `VERSION=balanced_gpt MODEL=new_model sbatch code/data_gen/slurm/1_generate_conversations.sh`
3. Run analysis: `VERSION=balanced_gpt MODEL=new_model sbatch code/analysis/slurm/run_analysis.sh`
4. Results appear in `results/new_model/{version}/`
