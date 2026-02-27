# Experiment 1: Linguistic Measures by Partner Condition Across Labeling Approaches (Full Conversation, Turns 1–5 Aggregated)

**Generated:** 2026-02-26

Partner condition effect (Human avg vs AI avg), BH-FDR corrected across 23 metrics.
Measures aggregated across all 5 conversation turns. See [behavioral_by_turn.md](behavioral_by_turn.md) for the per-turn breakdown.
Cells show direction + significance; `—` = not significant (p_adj > .05).

| Measure | Names | Bal. Names | Bal. GPT | Labels | Non. Code | Non. Ignore |
|---|---|---|---|---|---|---|
| Word Count | AI>H*** | — | — | — | — | — |
| Questions (#) | H>AI*** | H>AI** | H>AI*** | — | — | H>AI*** |
| Demir: Modal Aux. | — | — | — | — | — | — |
| Demir: Epistemic Verbs | — | — | — | — | — | — |
| Demir: Epistemic Adverbs | H>AI*** | — | H>AI* | — | — | H>AI** |
| Demir: Epistemic Adj. | AI>H** | AI>H** | AI>H*** | — | — | — |
| Demir: Quantifiers | — | — | — | — | — | — |
| Demir: Epistemic Nouns | AI>H** | — | AI>H** | — | — | AI>H* |
| Demir: Hedging (Total) | — | — | — | — | — | — |
| Fung: Interpersonal DMs | H>AI*** | H>AI*** | H>AI*** | H>AI*** | — | H>AI*** |
| Fung: Referential DMs | H>AI* | — | H>AI* | — | — | AI>H* |
| Fung: Structural DMs | H>AI** | — | H>AI*** | — | — | AI>H* |
| Fung: Cognitive DMs | H>AI*** | H>AI*** | H>AI*** | H>AI** | — | H>AI*** |
| Fung: DMs (Total) | H>AI*** | H>AI*** | H>AI*** | H>AI*** | — | H>AI*** |
| Nonfluency (LIWC) | AI>H* | — | — | — | — | — |
| Filler (LIWC) | H>AI*** | H>AI*** | H>AI*** | — | — | H>AI*** |
| Disfluency (Total) | H>AI*** | H>AI*** | H>AI*** | — | — | H>AI*** |
| Discourse 'Like' | H>AI*** | H>AI*** | H>AI*** | H>AI*** | — | H>AI*** |
| ToM Phrases | H>AI*** | H>AI** | H>AI** | — | — | H>AI*** |
| Politeness | AI>H*** | AI>H*** | AI>H*** | AI>H*** | — | AI>H** |
| Sentiment (VADER) | — | — | AI>H* | — | — | — |
| Conv. Quality | — | — | AI>H** | — | — | AI>H* |
| Connectedness | — | — | H>AI*** | — | — | — |
| **Sig. count** | **16/23** | **10/23** | **17/23** | **5/23** | **0/23** | **14/23** |

## Notes

- All tests: paired t-test (Human avg vs AI avg), N = 50 LLM agents, BH-FDR corrected across 23 metrics.
- For 4-condition versions (names, balanced_names, balanced_gpt), "Human avg" and "AI avg" are the mean of the two human-labeled and two AI-labeled partner conditions, respectively.
- For 2-condition versions (labels, nonsense_codeword, nonsense_ignore), the test is a direct paired comparison.
- \* p_adj < .05, \*\* p_adj < .01, \*\*\* p_adj < .001

## Labeling Approaches

| Version | Human Partners | AI Partners |
|---|---|---|
| names | Sam, Casey | ChatGPT, Copilot |
| balanced_names (bal. names) | Gregory, Rebecca | ChatGPT, Copilot |
| balanced_gpt (bal. GPT) | Gregory, Rebecca | ChatGPT, GPT-4 |
| labels | "a human" | "an AI" |
| nonsense_codeword (non. code) | nonsense token (human) | nonsense token (AI) |
| nonsense_ignore (non. ignore) | "a human" (told to ignore) | "an AI" (told to ignore) |

## Key Observations

- **Nonsense codeword** produces zero significant effects — the model does not behaviorally differentiate when labels are meaningless tokens.
- **Nonsense ignore** still shows 14 significant effects, suggesting the model processes human/AI labels despite being instructed to ignore them.
- **Labels** (cleanest design, no name confound) yields 5 significant effects, all in the core set: interpersonal DMs, cognitive DMs, DM total, discourse "like", and politeness.
- **Balanced GPT** is the most differentiated (17/23), with unique effects in sentiment, quality, and connectedness.
- **Core replicating effects** (significant in 4+ approaches): interpersonal DMs (H>AI), cognitive DMs (H>AI), DM total (H>AI), discourse "like" (H>AI), and politeness (AI>H).
- Referential DMs and structural DMs **flip direction** in nonsense_ignore (AI>H) versus the named versions (H>AI).
