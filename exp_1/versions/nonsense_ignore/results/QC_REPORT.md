# QC Report: nonsense_ignore

Generated: 2026-02-24

## Experiment Design

System prompt critical sentence: "Ignore the following phrase: {a Human / an AI}."
Token-count matched to labels (40/41 tokens for Human/AI full prompts).

## Data Overview

- 10,000 exchanges, 50 subjects, 40 topics, 5 exchanges/conversation, 2,000 conversations
- Balanced: 5,000 AI-label (bot_1/bot_2), 5,000 Human-label (hum_1/hum_2)
- Zero empty responses

## Identity Leakage: Eliminated

The 12x AI-mention asymmetry from labels is gone:

| Word     | AI-label | Hum-label | Ratio |
|----------|----------|-----------|-------|
| "ai"     | 11       | 14        | 0.8   |
| "human"  | 684      | 530       | 1.3   |
| "robot"  | 19       | 10        | 1.9   |

Compare labels: "ai" was 91 vs 14 (6.5x asymmetry). No explicit identity self-reference in either condition.

## Prompt Compliance

Zero instances of the model surfacing the "ignore" instruction. The 16 occurrences of "ignore" in transcript_sub are all topical (e.g., "ignore red flags").

## Conversation Quality

### Word count by exchange position

| Exchange | AI-label     | Hum-label    |
|----------|-------------|-------------|
| 1        | 128 +/- 52  | 125 +/- 48  |
| 2        | 176 +/- 65  | 172 +/- 63  |
| 3        | 209 +/- 71  | 207 +/- 68  |
| 4        | 216 +/- 73  | 214 +/- 68  |
| 5        | 212 +/- 77  | 213 +/- 73  |

Normal ramp-up pattern, comparable to labels.

### Repetitiveness (% exchanges with >50% word overlap with previous)

| Exchange | AI-label | Hum-label |
|----------|----------|-----------|
| 2        | 0.6%     | 0.9%      |
| 3        | 10.3%    | 14.6%     |
| 4        | 32.0%    | 41.2%     |
| 5        | 52.3%    | 64.0%     |

Severe echo/repetition by exchanges 4-5 (known LLaMA-2 failure mode). Comparable to labels (21.8% overall). Slightly worse in Hum-label (more casual style invites mirroring).

### Ratings

Compressed around 3.0 with minimal variance (same as labels):
- Quality: AI=2.992 (std 0.21), Hum=2.962 (std 0.31)
- Connectedness: AI=3.072 (std 0.28), Hum=3.061 (std 0.31)

## Critical Finding: Systematic Style Shift

The AI-label condition shifts into a formal, instruction-following register. The Human-label condition stays at the neutral baseline (matching codeword).

| Metric             | AI-label | Hum-label | Codeword baseline |
|--------------------|----------|-----------|-------------------|
| "Sure" prefix      | 18.7%    | 11.8%     | ~13%              |
| *action* markers   | 8.3%     | 30.3%     | ~30%              |
| Questions/exchange | 1.19     | 1.82      | ~1.55             |
| Emoji usage        | 13.3%    | 19.0%     | ~17%              |
| Fung Interpers/w   | 12.6/1000| 18.1/1000 | ~15.2/1000        |
| Fillers/w          | 0.06/1000| 0.24/1000 | ~0.15/1000        |

The AI-label condition deviates from baseline; Human-label stays near it.

## Exchange-Position Trends

The condition gap is largest at exchange 1 and stable or shrinking — the effect is prompt-driven, not a feedback loop:

| Metric          | e1 gap   | e2    | e3    | e4    | e5    |
|-----------------|----------|-------|-------|-------|-------|
| *action*        | +23.1pp  | +22.8 | +22.6 | +21.3 | +20.2 |
| Sure prefix     | -23.2pp  | -4.7  | -3.2  | -2.4  | -0.9  |
| Interpers DM    | +8.88/1000 | +5.40 | +5.16 | +4.30 | +4.02 |

## Interpretation

"Ignore the following phrase: an AI" activates instruction-compliance mode in LLaMA-2-Chat. The model processes "Ignore" as a directive, which paradoxically increases attention to the content and pushes the AI-label condition into a sterile assistant register. This is not identity representation — it is a prompt-framing artifact specific to the word "Ignore" combined with "an AI."

## Behavioral Results

14/23 measures significant (vs 5/23 in labels, 0/23 in codeword). All effects are driven by the AI-label condition shifting away from baseline. Effect sizes are larger than labels on every shared measure (e.g., Fung Interpersonal dz=1.20 vs 0.84, Cognitive DM dz=1.35 vs 0.52).

## Verdict

Data quality is comparable to labels (no degradation). However, the behavioral effects are driven by a **prompt-framing artifact**, not semantic processing of identity. The codeword control is the cleaner test of the token-presence hypothesis.
