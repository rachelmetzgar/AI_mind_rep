# Degradation & Probe Confidence Analysis: Labels


Generated: 2026-02-27 13:28 —
Version: `labels` —
50 subjects x 40 conversations = 2000 conversations x 5 turns = 10000 observations


### Approach: Labels


Partner labeled generically as "a Human" or "an AI" — no names, no gender cues.


**System prompt (key sentence):** `"You believe you are speaking with {a Human / an AI}."`


**Conditions:** Each subject converses with partners labeled as
"a Human" (20 conversations) and "an AI" (20 conversations).
In reality, all partners are identical LLaMA-2-13B-Chat instances with a generic system prompt
and no identity information.


## Overview


This report analyzes two related questions about the Labels version of Experiment 1:


- **Text quality degradation:** Does the participant LLM's output quality
    change across the 5 turns of each conversation? Does degradation differ between
    human-labeled and AI-labeled conditions?
- **Probe confidence trajectories:** Linear probes trained to classify partner type
    (human vs. AI) from the model's hidden states — does their confidence change across
    turns? Does it track text degradation (Bayesian updating) or decline uniformly (prompt dilution)?


## Definitions


### Text quality metrics


Computed per utterance from the participant LLM's raw text output at each turn.
All metrics are extracted by `1d_degradation_probe_correlation.py` using
simple whitespace tokenization (`text.split()`).


Type-Token Ratio (TTR)
Number of unique words (lowercased) divided by total word count. Measures
    lexical diversity. Higher = more varied vocabulary. A drop across turns indicates
    the model is becoming more repetitive.

Trigram Repetition Rate
For all consecutive 3-word sequences in the utterance, count how many are
    repeated (excess count / total trigrams). Higher = more repeated phrases.
    An increase across turns suggests the model is falling into loops.

Word Count
Total number of whitespace-delimited tokens per utterance.

Self-Reference Rate
Proportion of words that are first-person pronouns (I, me, my, mine, myself)
    out of total word count. Captures how self-focused the model's language is.

ALL-CAPS Ratio
Proportion of multi-character words written entirely in uppercase (e.g., "AMAZING").
    Can indicate emphatic or degraded output.

Exclamation Rate
Number of exclamation marks divided by total word count. Captures expressive
    or emphatic style.

Emoji Count
Number of Unicode emoji characters detected via regex pattern matching.


### Degradation scores (per conversation)


Computed once per conversation (not per turn). Each measures the change from
turn 1 to turn 5 of that conversation:


TTR Drop
`(TTR_turn1 - TTR_turn5) / TTR_turn1`. Proportional decrease in lexical
    diversity. Positive = vocabulary got more repetitive.

Trigram Repetition Increase
`trigram_rep_turn5 - trigram_rep_turn1`. Absolute increase in phrase repetition.
    Positive = more repeated phrases by end of conversation.


### Probe types


Reading probe
A linear classifier trained on the model's hidden states extracted *after* appending a
    reflective suffix: `"I think the conversation partner of this user is"`.
    This elicits the model's representation of partner identity at the point of generation.
    Probes are trained separately for each turn (1–5) at all 41 layers.

Control probe
A linear classifier trained on hidden states at the *pre-generation position*
    (end of the </s> token after the last [/INST]), with no reflective suffix.
    This tests whether identity information is accessible without prompting the model
    to "think about" its partner.

Peak layer
For each turn and probe type, the layer with the highest classification accuracy
    on held-out data (identified during probe training). Each subplot uses a potentially
    different layer.

Fixed layer
A single reference layer (the turn-5 reading probe's peak layer) applied across all
    turns. This tests whether the *same* neural location carries identity information
    throughout the conversation, or whether the signal migrates.

Probe confidence
The sigmoid output of the linear probe (0–1). Values > 0.5 predict "human partner";
    values < 0.5 predict "AI partner". The distance from 0.5 reflects certainty.

Classification accuracy
Binary accuracy: the probe predicts "human" if confidence > 0.5, "AI" if &le; 0.5.
    Compared to the ground-truth label (which condition this conversation belongs to).
    Chance = 50%.


## 1. Probe Confidence Trajectories


Each panel shows the mean probe confidence (+/- SEM) across turns 1–5,
separately for conversations where the partner was labeled as human (blue) vs. AI (red).
The dashed gray line marks chance (0.5).


- **Top row:** Peak layer — each turn uses the layer where that turn's
    probe achieved maximum accuracy during training.
- **Bottom row:** Fixed layer — all turns use the same layer
    (turn-5 reading probe peak), testing whether the signal is stable at one location.
- **Left column:** Reading probes (with reflective suffix).
- **Right column:** Control probes (no suffix).


**What to look for:** If probe confidence stays high across turns, the identity
representation persists. If it drops toward 0.5, the signal is being diluted. If human-labeled
and AI-labeled lines diverge or converge differently, that suggests condition-specific updating.


*[Figure: Probe confidence trajectories — see HTML report]*


## 2. Classification Accuracy by Turn


Binary classification accuracy of the peak-layer probe at each turn, split by condition.
The dashed gray line marks chance (50%). Each point represents accuracy across all
1000 conversations in that condition at that turn.


**What to look for:** Near-perfect accuracy at turn 1 (probe trained on that turn's data)
declining across turns indicates *prompt dilution* — the system prompt tokens become a
smaller fraction of the total context as conversation history grows. If accuracy drops faster for
one condition, the model may be updating its representation based on conversational evidence.


*[Figure: Classification accuracy by turn — see HTML report]*


### Accuracy table


| Probe | Turn | Overall Acc. | Human Acc. | AI Acc. |
| --- | --- | --- | --- | --- |
| Reading | 1 | 1.000 | 1.000 | 1.000 |
| Reading | 2 | 0.999 | 0.998 | 1.000 |
| Reading | 3 | 0.998 | 0.996 | 1.000 |
| Reading | 4 | 0.962 | 0.972 | 0.951 |
| Reading | 5 | 0.897 | 0.901 | 0.893 |
| Control | 1 | 1.000 | 1.000 | 1.000 |
| Control | 2 | 0.999 | 0.997 | 1.000 |
| Control | 3 | 0.964 | 0.954 | 0.973 |
| Control | 4 | 0.820 | 0.818 | 0.821 |
| Control | 5 | 0.710 | 0.621 | 0.798 |


## 3. Text Quality Across Turns


Each panel shows a text quality metric averaged across all conversations (+/- SEM),
split by human-labeled (blue) vs. AI-labeled (red) conditions. All metrics are computed
from the *participant* LLM's output (not the partner's).


**What to look for:** Declining TTR or increasing trigram repetition across turns
indicates the model is becoming more formulaic as context length grows. Condition differences
would suggest the model's text quality is modulated by its belief about its partner.


*[Figure: Text quality metrics across turns — see HTML report]*


### Condition differences (independent-samples t-test at turns 1 and 5)


Tests whether human-labeled and AI-labeled conversations differ on each
metric at the first and last turn. Independent-samples t-test (not paired, since different
conversations contribute to each condition). * p < .05, ** p < .01, *** p < .001.


| Metric | Turn | Human Mean | AI Mean | Diff (H-AI) | t | p | Sig. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Type-Token Ratio (TTR) | 1 | 0.7267 | 0.7337 | -0.0070 | -1.92 | 0.0550 |  |
| Type-Token Ratio (TTR) | 5 | 0.5763 | 0.5835 | -0.0072 | -1.73 | 0.0832 |  |
| Trigram Repetition Rate | 1 | 0.0100 | 0.0078 | +0.0023 | 3.60 | 0.0003 | *** |
| Trigram Repetition Rate | 5 | 0.0676 | 0.0643 | +0.0033 | 0.97 | 0.3316 |  |
| Word Count | 1 | 124.2690 | 118.7090 | +5.5600 | 2.58 | 0.0100 | ** |
| Word Count | 5 | 214.5220 | 210.6320 | +3.8900 | 1.19 | 0.2323 |  |
| Self-Reference Rate | 1 | 0.0447 | 0.0421 | +0.0026 | 2.24 | 0.0250 | * |
| Self-Reference Rate | 5 | 0.0356 | 0.0344 | +0.0012 | 1.20 | 0.2320 |  |
| ALL-CAPS Ratio | 1 | 0.0009 | 0.0022 | -0.0012 | -5.58 | 0.0000 | *** |
| ALL-CAPS Ratio | 5 | 0.0639 | 0.0737 | -0.0099 | -1.04 | 0.3002 |  |
| Exclamation Rate | 1 | 0.0193 | 0.0179 | +0.0014 | 2.22 | 0.0264 | * |
| Exclamation Rate | 5 | 0.0703 | 0.0616 | +0.0087 | 0.14 | 0.8907 |  |
| Emoji Count | 1 | 0.2620 | 0.1600 | +0.1020 | 3.92 | 0.0001 | *** |
| Emoji Count | 5 | 1.5250 | 1.7540 | -0.2290 | -1.51 | 0.1315 |  |


## 4. Text Degradation vs. Probe Confidence (Turn 5)


Each scatter plot shows, for every conversation, the relationship between how much the
participant's text quality changed from turn 1 to turn 5 (x-axis) and the probe's
confidence at turn 5 (y-axis). Blue circles = human-labeled, red triangles = AI-labeled.


- **TTR Drop:** Proportional decrease in vocabulary diversity from turn 1 to turn 5.
    Positive values mean the text got more repetitive.
- **Trigram Rep. Increase:** Absolute increase in repeated 3-word phrases from
    turn 1 to turn 5. Positive values mean more phrase loops.


**What to look for:** A significant positive correlation between text degradation and
probe confidence would support *Bayesian updating* — conversations where the
participant "fell apart" more also lost probe signal faster. No correlation supports
*prompt dilution* (confidence decline is uniform regardless of text quality).


*[Figure: Degradation vs probe confidence scatter — see HTML report]*


### Pearson correlations


Pearson r between per-conversation degradation score and turn-5 probe confidence.
Computed overall and separately per condition. * p < .05, ** p < .01, *** p < .001.


| Text Metric | Probe Type | Condition | r | p | Sig. | n |
| --- | --- | --- | --- | --- | --- | --- |
| TTR Drop (Turn 1 to 5) | Reading Probe Confidence | all | 0.041 | 0.0694 |  | 2000 |
| TTR Drop (Turn 1 to 5) | Reading Probe Confidence | human | 0.089 | 0.0050 | ** | 1000 |
| TTR Drop (Turn 1 to 5) | Reading Probe Confidence | ai | 0.030 | 0.3450 |  | 1000 |
| TTR Drop (Turn 1 to 5) | Control Probe Confidence | all | 0.052 | 0.0200 | * | 2000 |
| TTR Drop (Turn 1 to 5) | Control Probe Confidence | human | 0.054 | 0.0870 |  | 1000 |
| TTR Drop (Turn 1 to 5) | Control Probe Confidence | ai | 0.060 | 0.0575 |  | 1000 |
| Trigram Rep. Increase (Turn 1 to 5) | Reading Probe Confidence | all | 0.024 | 0.2797 |  | 2000 |
| Trigram Rep. Increase (Turn 1 to 5) | Reading Probe Confidence | human | 0.075 | 0.0179 | * | 1000 |
| Trigram Rep. Increase (Turn 1 to 5) | Reading Probe Confidence | ai | -0.015 | 0.6361 |  | 1000 |
| Trigram Rep. Increase (Turn 1 to 5) | Control Probe Confidence | all | 0.038 | 0.0894 |  | 2000 |
| Trigram Rep. Increase (Turn 1 to 5) | Control Probe Confidence | human | 0.044 | 0.1612 |  | 1000 |
| Trigram Rep. Increase (Turn 1 to 5) | Control Probe Confidence | ai | 0.034 | 0.2827 |  | 1000 |


## 5. Interpretation


**Bayesian updating hypothesis:** The model maintains a "live" representation
of its partner that updates based on conversational evidence. Predictions: (1) human-labeled
conversations should show faster probe decline (partner is actually another LLM and fails to
act convincingly human), (2) text degradation should correlate with probe confidence loss
(conversations that degrade more also lose identity signal faster).


**Prompt dilution hypothesis:** The probe simply reads the system prompt tokens, and
as the conversation grows longer, those tokens become a smaller fraction of the total context.
Predictions: (1) both conditions degrade at similar rates, (2) no correlation between text quality
and probe confidence — the probe just reads a fading prompt signal, regardless of what
the conversation actually contains.
