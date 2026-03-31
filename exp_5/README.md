# Experiment 5 — Mental State Attribution RSA

## Overview

Tests whether LLMs maintain a dedicated representational structure for **mental state attributions** — the bound proposition {subject + mental state verb + object} — that is distinct from representations of the component parts in isolation. Run across 11 models spanning 4 families (LLaMA-2-13B, LLaMA-3-8B, Gemma-2-2B/9B, Qwen-2.5-7B, Qwen3-8B; base and instruct variants).

The core claim: if the model has genuine mental state attribution machinery, then sentences like "He believes the story" should produce a representational geometry that is NOT explained by:
- The presence of mental state vocabulary alone
- The syntactic frame alone
- The lexical identity of any individual word

The geometry should ONLY emerge when all three components (subject, mental state verb, object) are bound together in grammatical order.

## Theoretical Motivation

Mental state attribution (also called "mentalizing" or "theory of mind") requires representing a structured proposition: AGENT + MENTAL STATE + CONTENT. This is distinct from:
- Knowing what mental state words mean (lexical knowledge)
- Processing subject-verb-object syntax (syntactic parsing)
- Representing mental states without attributing them to an agent

If the model has internalized mental state attribution as a structured operation, we should find a representational signature that is unique to the full bound attribution and absent when any component is removed or scrambled.

## Stimuli

### Design

56 items x 6 conditions = 336 sentences total.

**Subject** is fixed to "He" throughout (eliminates subject as a variable).

Each item has a **mental state verb** and a matched **action verb** (maximally concrete/physical, no mental state leakage). The same object noun appears across all 6 conditions for a given item.

### The 6 Conditions

For item 1 (verb: "notices", action: "fills", object: "the crack"):

| Condition | Label | Example | What it controls |
|-----------|-------|---------|-----------------|
| 1 | `mental_state` | He notices the crack. | Full attribution: subject + mental verb + object |
| 2 | `dis_mental` | Notice the crack. | Mental verb + object, NO subject |
| 3 | `scr_mental` | The crack to notice. | Same words as cond 2, scrambled order |
| 4 | `action` | He fills the crack. | Subject + action verb + object (same syntactic frame, no mental state) |
| 5 | `dis_action` | Fill the crack. | Action verb + object, NO subject |
| 6 | `scr_action` | The crack to fill. | Same words as cond 5, scrambled order |

#### Open Question:
Not sure the representational space is fully defined and cleaned of confounds. For example, do I need to control for the word "notices" or find a way to use notices in a grammatically correct sentence without referencing mental states? Or the word "he"
or "he notices" but no object? Need to be careful to control for everything without introducing excess conditions that might limit signal detection power.

### Mental State Verb Categories (7 categories x 8 verbs = 56 items)

| Category | Verbs |
|----------|-------|
| **Attention** (8) | notices, observes, watches, sees, detects, examines, inspects, distinguishes |
| **Memory** (8) | remembers, recalls, forgets, recognizes, misremembers, reminisces, retains, recollects |
| **Sensation** (8) | feels, senses, perceives, tastes, smells, hears, touches, experiences |
| **Belief** (8) | believes, knows, assumes, trusts, doubts, thinks, suspects, supposes |
| **Desire** (8) | wants, craves, desires, needs, yearns for, pursues, seeks, prefers |
| **Emotion** (8) | fears, loves, dreads, envies, admires, hates, resents, cherishes |
| **Intention** (8) | contemplates, plans, expects, anticipates, ponders, decides, chooses, considers |

### Action Verb Constraints
- Every action verb is unique (no repeats across the 56 items)
- All action verbs are maximally concrete/physical ("security camera test": you could identify the action from silent video)
- Action verbs should NOT imply mental states (no "decided," "ignored," "avoided," etc.)

### Condition Feature Matrix

| Feature | C1 (mental_state) | C2 (dis_mental) | C3 (scr_mental) | C4 (action) | C5 (dis_action) | C6 (scr_action) |
|---|---|---|---|---|---|---|
| Has subject "He" | yes | no | no | yes | no | no |
| Mental verb present | yes | yes | yes | no | no | no |
| Action verb present | no | no | no | yes | yes | yes |
| Grammatical word order | yes | yes | no | yes | yes | no |
| Scrambled form ("The X to Y") | no | no | yes | no | no | yes |
| Shared object noun (within item) | yes | yes | yes | yes | yes | yes |

## Analysis: 5-Predictor Regression RSA

The core analysis uses a 5-predictor multiple regression RSA on conditions C1-C4 (dropping C5/C6 scrambled action controls). The 5 model RDMs isolate progressively specific aspects of mental state attribution:

| Predictor | Name | Conditions | Tests |
|-----------|------|------------|-------|
| A | Full Attribution | C1 only | Bound {subject + mental verb + object} |
| B | Mental Verb + Object | C1, C2 | Mental verb with object, no subject required |
| C | Mental Verb Presence | C1, C2, C3 | Presence of mental verb in any form |
| D | Verb + Object | C1, C2, C4 | Verb-object binding regardless of verb type |
| E | Subject + Verb + Object | C1, C4 | Full sentence structure regardless of verb type |

**Key question:** Does predictor A capture unique variance (delta-R-squared) beyond what is explained by its component features (B-E)?

Analyses run at three token positions: **verb** (before object is seen), **object** (verb-object binding begins), **period** (full sentence processed). 10,000 permutation tests with FDR correction.

## Reports

- [Cross-model comparison (11 models)](results/comparisons/rsa/5_predictors/5_predictors_cross_model_positional.html)
- Per-model reports at `results/{model}/rsa/reports/5_predictors_positional_report.html`

## Pipeline

| Script | Description | GPU |
|--------|-------------|-----|
| `code/rsa/1_extract_activations.py` | Extract last-token activations (336 sentences) | Yes |
| `code/probes/1_extract_multipos_activations.py` | Extract verb/object/period activations | Yes |
| `code/rsa/5_predictors/1_reduced_1_4_rsa.py` | 5-predictor RSA (last-token) | No |
| `code/rsa/5_predictors/1b_positional_rsa.py` | 5-predictor RSA (verb + object) | No |
| `code/rsa/5_predictors/2b_positional_report.py` | Per-model + cross-model HTML reports | No |

## Directory Structure

```
exp_5/
├── code/
│   ├── config.py
│   ├── stimuli.py
│   ├── rsa/
│   │   ├── 1_extract_activations.py
│   │   ├── 5_predictors/           # 5-predictor RSA analysis
│   │   └── slurm/
│   ├── probes/                     # Multi-position extraction + probe analyses
│   ├── interchange/                # Interchange intervention analyses
│   └── utils/
├── results/
│   ├── {model}/
│   │   ├── activations/data/       # Extracted activations (gitignored)
│   │   ├── rsa/data/5_predictors/  # RSA results (CSV, JSON, NPZ)
│   │   └── rsa/reports/            # Per-model HTML reports
│   └── comparisons/rsa/5_predictors/  # Cross-model comparison report
└── concepts/                       # Stimulus definitions
```

