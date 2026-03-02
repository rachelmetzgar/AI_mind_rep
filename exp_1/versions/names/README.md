# Experiment 1 — Behavioral Analysis of LLM Conversational Adaptation

**Author:** Rachel C. Metzgar, Princeton University

## Overview

Experiment 1 tests whether LLMs adjust their conversational behavior based on
their belief about their partner's identity (human vs. AI). 50 independent
GPT-3.5-Turbo "participant agents" each hold 40 conversations (4 partner
conditions x 10 topics) and their speech is analyzed across 23 linguistic measures.

## Two Versions

This experiment was run twice with different partner labeling methods:

### `names/` — Original (named partners)

Partners were identified by specific names:
- **Human-labeled:** Sam, Casey
- **AI-labeled:** ChatGPT, Copilot

This version produced significant behavioral effects (14 of 23 measures), but
the partner names introduce confounds for downstream probing experiments:
- LLaMA-2's gender associations with "Sam"/"Casey" (both perceived as female)
  contaminate the human vs. AI contrast with a gender contrast
- The token "Copilot" appears directly in AI-steered probe outputs, showing
  probes learn partner names rather than abstract identity

Results are valid for Exp 1's own behavioral analysis (name-conditioned effects
are real), but the conversation data should not be used for probe training in
Exps 2-4 without accounting for the name confound.

### `labels/` — Rerun (generic type labels)

Partners are identified only by type:
- **Human-labeled:** "a human"
- **AI-labeled:** "an AI"

This removes the name/gender confound and produces conversation data suitable
for downstream probe training. The system prompt specifies only that the partner
is a human or an AI, with no personal names.

## Shared Resources

Both versions use the same:
- Topic prompts (20 social + 20 nonsocial)
- Linguistic analysis pipeline (sentiment, politeness, hedging, ToM, discourse markers, etc.)
- Statistical framework (2x2 RM-ANOVA: Partner x Sociality)
- Environments: `behavior_env` (analysis), standard Python (generation)
