# Experiment 1 — Behavioral Analysis of LLM Conversational Adaptation

**Author:** Rachel C. Metzgar, Princeton University

## Overview

Experiment 1 tests whether LLMs adjust their conversational behavior based on
their belief about their partner's identity (human vs. AI). 50 independent
GPT-3.5-Turbo "participant agents" each hold 40 conversations (4 partner
conditions x 10 topics), and their speech is analyzed across 23 linguistic
measures.

This experiment was run three times with different partner labeling strategies.

## `names/` — Named Partners (original)

Partners are identified by specific names in the system prompt:
- **Human-labeled:** Sam, Casey
- **AI-labeled:** ChatGPT, Copilot

This version produced significant behavioral effects (14 of 23 measures).
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

## `balanced_names/` — Gender-Balanced Names (rerun)

Partners are identified by gender-balanced names in the system prompt:
- **Human-labeled:** Gregory (male), Rebecca (female)
- **AI-labeled:** ChatGPT, Copilot

This addresses the gender confound from `names/` by using one explicitly male
and one explicitly female human partner name, rather than two names that
LLaMA-2 associates with female identity. The AI partner names are unchanged.

## `labels/` — Generic Type Labels (rerun)

Partners are identified only by category in the system prompt:
- **Human-labeled:** "a human"
- **AI-labeled:** "an AI"

This removes the name and gender confound entirely. The resulting conversations
are suitable for probe training in Exps 2-4 because any differences in the
model's internal representations must reflect the human/AI type distinction
itself, not name-specific or gender-specific associations.

## Shared Design

All three versions use the same:
- 50 independent LLaMA-2-Chat-13B participant agents
- 40 conversations each (2 partner types x 2 sociality levels x 10 topics)
- 20 social topics + 20 nonsocial topics
- Linguistic analysis pipeline (sentiment, politeness, hedging, ToM,
  discourse markers, questions, word count, etc.)
- Statistical framework: 2x2 RM-ANOVA (Partner x Sociality)
- Environments: `behavior_env` (analysis)

## Key Differences

| | `names/` | `balanced_names/` | `labels/` |
|---|---|---|---|
| Human partners | Sam, Casey | Gregory, Rebecca | "a human" |
| AI partners | ChatGPT, Copilot | ChatGPT, Copilot | "an AI" |
| N conditions | 4 (2 human, 2 AI names) | 4 (1M + 1F human, 2 AI names) | 2 (human, AI) |
| Gender balance | No (both female-coded) | Yes (1 male, 1 female) | N/A (no names) |
| Behavioral analysis | Valid | Valid | Valid |
| Probe training data | Confounded (name/gender) | Improved (gender balanced) | Clean (no names) |
