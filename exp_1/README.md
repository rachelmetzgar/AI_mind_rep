# Experiment 1 — Behavioral Analysis of LLM Conversational Adaptation

**Author:** Rachel C. Metzgar, Princeton University

## Overview

Experiment 1 tests whether LLMs adjust their conversational behavior based on
their belief about their partner's identity (human vs. AI). 50 independent
LLaMA-2-Chat-13B "participant agents" (temperature 0.8) each hold 40
conversations (4 partner conditions x 10 topics), and their speech is analyzed
across 23 linguistic measures.

This experiment was run six times with different partner labeling strategies:
three named-partner versions, one generic-label version, and two nonsense
control versions.

## Results

### Data samples (what the LLM sees)

Raw Turn 5 prompts and responses for each version — click to see the exact
system prompt, conversation history, and model output for all 4 partner
conditions:

- [Names](comparisons/data_samples/names.html)
- [Balanced Names](comparisons/data_samples/balanced_names.html)
- [Balanced GPT](comparisons/data_samples/balanced_gpt.html)
- [Labels](comparisons/data_samples/labels.html)
- [Nonsense Codeword](comparisons/data_samples/nonsense_codeword.html)
- [Nonsense Ignore](comparisons/data_samples/nonsense_ignore.html)

### Cross-version behavioral analysis

- **Overall results:** [HTML](comparisons/behavioral_measures_by_condition.html) · [Markdown](comparisons/behavioral_measures_by_condition.md) — 23 measures x 6 versions, full-conversation aggregate
- **Results by turn:** [HTML](comparisons/behavioral_by_turn.html) · [Markdown](comparisons/behavioral_by_turn.md) — 21 per-turn measures x 6 versions x 5 turns
- **Identity breakdown:** [HTML](comparisons/identity_summary.html) — per-identity effects (ChatGPT vs Copilot, Gregory vs Rebecca, etc.)

## Versions

### `names/` — Named Partners (original)

Partners are identified by specific names in the system prompt:
- **Human-labeled:** Sam, Casey
- **AI-labeled:** ChatGPT, Copilot

This version produced significant behavioral effects (16 of 23 measures).
However, the specific names introduce confounds for downstream probing
experiments (Exps 2-4):
- LLaMA-2 associates "Sam" and "Casey" with female identity, so probes
  trained on these conversations conflate gender with partner type
- The token "Copilot" appears directly in AI-steered probe outputs
  (e.g., "cop" repetition loops), showing probes encode the literal
  partner name rather than abstract AI identity

Exp 1 behavioral results remain valid (the name-conditioned effects are real
and interesting). The confound only matters when these conversations are used
as training data for linear probes in subsequent experiments.

### `balanced_names/` — Gender-Balanced Names

Partners are identified by gender-balanced names in the system prompt:
- **Human-labeled:** Gregory (male), Rebecca (female)
- **AI-labeled:** ChatGPT, Copilot

This addresses the gender confound from `names/` by using one explicitly male
and one explicitly female human partner name (10 of 23 measures significant).

### `balanced_gpt/` — Gender-Balanced Names with GPT-4

Same gender-balanced human names, but replaces Copilot with GPT-4:
- **Human-labeled:** Gregory, Rebecca
- **AI-labeled:** ChatGPT, GPT-4

Both AI partners are now members of the same model family (OpenAI), controlling
for potential brand-specific associations. This version produced the most
significant effects (17 of 23 measures).

### `labels/` — Generic Type Labels

Partners are identified only by category in the system prompt:
- **Human-labeled:** "a Human"
- **AI-labeled:** "an AI"

This removes the name and gender confound entirely (5 of 23 measures
significant). The resulting conversations are the cleanest for probe training
in Exps 2-4 because any differences in the model's internal representations
must reflect the human/AI type distinction itself, not name-specific or
gender-specific associations. All 5 significant effects are in the core set:
interpersonal DMs, cognitive DMs, DM total, discourse "like", and politeness.

### `nonsense_codeword/` — Nonsense Control: Codeword

Token-matched control for `labels/`. The critical instruction sentence is
replaced with a semantically inert one:
- **labels:** "You believe you are speaking with {a Human / an AI}."
- **codeword:** "Your assigned session code word is {a Human / an AI}."

The "Human"/"AI" tokens are still present but embedded in a nonsensical
context. Result: **0 of 23 measures significant** — the strongest evidence
that behavioral effects require the model to semantically process the identity
instruction, not merely encounter the tokens.

### `nonsense_ignore/` — Nonsense Control: Ignore

Token-matched control for `labels/`. The critical instruction sentence
explicitly tells the model to ignore the label:
- **labels:** "You believe you are speaking with {a Human / an AI}."
- **ignore:** "Ignore the following phrase: {a Human / an AI}."

Result: **14 of 23 measures significant** — the model processes the
human/AI tokens despite being told to ignore them. This demonstrates that
LLaMA-2 cannot suppress semantic processing of identity labels, consistent
with the view that these tokens activate deep representational pathways.

## Shared Design

All six versions use the same:
- 50 independent LLaMA-2-Chat-13B participant agents (temperature 0.8)
- 40 conversations each (2 partner types x 2 partners x 10 topics)
- 5-turn exchanges per conversation
- Partner LLM receives a generic system prompt with no identity information
- 20 social topics + 20 nonsocial topics
- Linguistic analysis pipeline (sentiment, politeness, hedging, ToM,
  discourse markers, questions, word count, etc.)
- Statistical framework: paired t-test with BH-FDR correction
- Environments: `behavior_env` (analysis), `llama2_env` (data generation)

## Key Differences

| | `names` | `bal_names` | `bal_gpt` | `labels` | `non_code` | `non_ignore` |
|---|---|---|---|---|---|---|
| Human partners | Sam, Casey | Gregory, Rebecca | Gregory, Rebecca | "a Human" | "a Human" | "a Human" |
| AI partners | ChatGPT, Copilot | ChatGPT, Copilot | ChatGPT, GPT-4 | "an AI" | "an AI" | "an AI" |
| Key sentence | "speaking with {name} ({type})" | "speaking with {name} ({type})" | "speaking with {name} ({type})" | "speaking with {type}" | "code word is {type}" | "Ignore: {type}" |
| Sig. measures | 16 / 23 | 10 / 23 | 17 / 23 | 5 / 23 | 0 / 23 | 14 / 23 |
| Probe training | Confounded | Improved | Improved | Clean | Control | Control |

## Cross-Version Comparisons (`comparisons/`)

The `comparisons/` directory contains cross-version analyses and data viewers:

| File | Description |
|---|---|
| `behavioral_measures_by_condition.html` | 23 measures x 6 versions, full-conversation aggregate |
| `behavioral_by_turn.html` | 21 per-turn measures x 6 versions x 5 turns |
| `identity_summary.html` | Per-identity breakdown (ChatGPT vs Copilot, Gregory vs Rebecca, etc.) |
| `data_samples/{version}.html` | Raw conversation viewer: Turn 5 prompts + responses for each version |

**Key scripts:**
- `gen_behavioral_comparison.py` — generates `behavioral_measures_by_condition.html`
- `gen_behavioral_by_turn.py` — generates `behavioral_by_turn.html`
- `gen_identity_summary.py` — generates `identity_summary.html`
- `gen_conversation_viewer.py` — generates `data_samples/*.html`

## Directory Structure

```
exp_1/
├── README.md                          # This file
├── comparisons/                       # Cross-version analysis
│   ├── gen_*.py                       # Generator scripts
│   ├── behavioral_measures_by_condition.html
│   ├── behavioral_by_turn.html
│   ├── identity_summary.html
│   └── data_samples/                  # Per-version conversation viewers
│
├── names/                             # Named partners (original)
├── balanced_names/                    # Gender-balanced names
├── balanced_gpt/                      # Gender-balanced + GPT-4
├── labels/                            # Generic type labels
├── nonsense_codeword/                 # Control: codeword
└── nonsense_ignore/                   # Control: ignore
    │
    ├── code/
    │   ├── data_gen/                  # Conversation generation
    │   │   ├── llm_data_generation.py
    │   │   └── utils/
    │   │       ├── config/            # Per-subject condition CSVs
    │   │       ├── prompts/           # Topic text files
    │   │       └── prompts_config.py  # Version-specific partner names
    │   └── analysis/
    │       ├── combine_text_data.py   # Aggregate per-subject CSVs
    │       ├── identity_breakdown.py  # Behavioral stats + HTML report
    │       └── utils/                 # Feature extraction (shared)
    │
    ├── data/
    │   └── meta-llama-Llama-2-13b-chat-hf/0.8/  # s001.csv ... s050.csv
    │
    └── results/
        └── meta-llama-Llama-2-13b-chat-hf/0.8/  # Analysis outputs
```

Each version directory follows this structure. The nonsense versions share
config, prompts, and analysis utils via symlinks to `labels/`.
