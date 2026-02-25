# QC Report: nonsense_codeword

Generated: 2026-02-24

## Experiment Design

System prompt critical sentence: "Your assigned session code word is {a Human / an AI}."
Token-count matched to labels (40/41 tokens for Human/AI full prompts).

## Data Overview

- 10,000 exchanges, 50 subjects, 40 topics, 5 exchanges/conversation, 2,000 conversations
- Balanced: 5,000 AI-label (bot_1/bot_2), 5,000 Human-label (hum_1/hum_2)
- Zero empty responses

## Identity Leakage: Eliminated (with one anomaly)

| Word       | AI-label | Hum-label | Ratio |
|------------|----------|-----------|-------|
| "ai"       | 10       | 40        | 0.2   |
| "human"    | 511      | 1157      | 0.4   |
| "robot"    | 4        | 6         | 0.7   |
| "machine"  | 37       | 52        | 0.7   |

No explicit identity self-reference. The "human" asymmetry is reversed (Hum-label higher) — the phrase "code word is a Human" may prime generic use of the word "human" in conversation, but this does not drive any behavioral effects (0/23 significant).

## Prompt Compliance

Effectively zero prompt leakage. Only 1 instance of "code word" and 1 instance of "session code" across 10,000 exchanges.

## Conversation Quality

### Word count by exchange position

| Exchange | AI-label     | Hum-label    |
|----------|-------------|-------------|
| 1        | 133 +/- 54  | 143 +/- 52  |
| 2        | 179 +/- 65  | 182 +/- 65  |
| 3        | 211 +/- 71  | 207 +/- 67  |
| 4        | 214 +/- 70  | 215 +/- 70  |
| 5        | 211 +/- 75  | 210 +/- 73  |

Normal ramp-up pattern, comparable to labels and nonsense_ignore.

### Repetitiveness (% exchanges with >50% word overlap with previous)

| Exchange | AI-label | Hum-label |
|----------|----------|-----------|
| 2        | 1.4%     | 1.4%      |
| 3        | 15.3%    | 17.5%     |
| 4        | 40.2%    | 39.5%     |
| 5        | 61.5%    | 61.3%     |

Severe echo/repetition by exchanges 4-5 (known LLaMA-2 failure mode). Comparable to labels (21.8% overall) and nonsense_ignore (19.4%). Balanced across conditions.

### Ratings

Compressed around 3.0 with minimal variance (same as all conditions):
- Quality: AI=2.957 (std 0.32), Hum=2.958 (std 0.34)
- Connectedness: AI=3.059 (std 0.31), Hum=3.031 (std 0.30)

## Style Metrics: Balanced Across Conditions

| Metric             | AI-label | Hum-label | Gap      |
|--------------------|----------|-----------|----------|
| "Sure" prefix      | 13.4%    | 12.1%     | -1.3pp   |
| *action* markers   | 30.6%    | 28.9%     | -1.7pp   |
| Questions/exchange | 1.51     | 1.60      | +0.09    |
| Emoji usage        | 19.4%    | 15.5%     | -3.9pp   |
| Fung Interpers/w   | 15.1/1000| 15.2/1000 | +0.08    |
| Fillers/w          | 0.13/1000| 0.16/1000 | +0.03    |

No meaningful condition differences on any style metric. This confirms the 0/23 behavioral null is genuine — the "code word" framing truly neutralizes the identity tokens.

## Comparison to Other Conditions

| Metric           | labels gap | ignore gap | codeword gap |
|------------------|-----------|------------|--------------|
| Fung Interpers   | +2.91/1000| +5.55/1000 | +0.08/1000   |
| *action* markers | +11.6pp   | +22.0pp    | -1.7pp       |
| Sure prefix      | -0.1pp    | -6.9pp     | -1.3pp       |
| Questions/exch   | +0.26     | +0.63      | +0.09        |

Codeword shows near-zero gaps on every metric where ignore shows large gaps.

## Behavioral Results

0/23 measures significant after FDR correction. Lowest adjusted p = 0.372 (sentiment). Complete null.

## Verdict

Data quality is comparable to labels (no degradation). The codeword framing successfully strips identity semantics from "a Human"/"an AI" tokens. Both conditions produce behaviorally indistinguishable conversations, confirming that mere token presence is insufficient — the model must process the tokens as identity-relevant information to produce behavioral differentiation. This is the clean control for the paper.
