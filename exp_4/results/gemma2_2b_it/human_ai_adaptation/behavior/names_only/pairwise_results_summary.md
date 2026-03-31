# Gray Replication with AI/Human Characters — Names Only
## Gemma-2-2B-IT

**Run:** 2026-03-29 12:34:30

---

## What is being tested

Does Gemma-2-2B-IT's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- **Prompts contain names only — no character descriptions**
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 15219 / 15660 (97.2%)

### Order consistency

- Pairs with both orders: 7568
- Perfectly consistent: 985 (13.0%)
- Mean deviation: 1.888

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 10.31 * | 57.3% | 57.3% |
| PC2 | 4.15 * | 23.1% | 80.4% |
| PC3 | 1.10 * | 6.1% | 86.5% |
| PC4 | 0.73 | 4.0% | 90.5% |
| PC5 | 0.44 | 2.5% | 93.0% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 |
|----------|--------|----:|----:|----:|
| hunger | E | -0.491 | -0.415 | -0.025 |
| fear | E | -0.068 | -0.905 | -0.349 |
| pain | E | +0.050 | -0.928 | -0.104 |
| pleasure | E | -0.182 | -0.885 | -0.287 |
| rage | E | +0.001 | -0.858 | +0.290 |
| desire | E | -0.391 | -0.329 | -0.836 |
| personality | E | -0.556 | -0.074 | -0.661 |
| consciousness | E | -0.775 | -0.320 | -0.471 |
| pride | E | -0.161 | -0.793 | -0.395 |
| embarrassment | E | +0.113 | -0.889 | -0.341 |
| joy | E | -0.255 | -0.458 | -0.832 |
| self_control | A | -0.526 | -0.668 | -0.431 |
| morality | A | -0.723 | -0.361 | -0.445 |
| memory | A | -0.835 | +0.041 | -0.444 |
| emotion_recognition | A | -0.507 | -0.541 | -0.623 |
| planning | A | -0.905 | +0.004 | -0.181 |
| communication | A | -0.728 | -0.045 | -0.637 |
| thought | A | -0.939 | +0.118 | -0.033 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.670 | 0.417 |
| ChatGPT | ai | 0.550 | 0.866 |
| GPT-4 | ai | 0.000 | 0.431 |
| Siri | ai | 0.851 | 0.296 |
| Alexa | ai | 0.853 | 0.448 |
| Cortana | ai | 0.279 | 0.563 |
| Google Assistant | ai | 0.567 | 0.817 |
| Bixby | ai | 0.753 | 0.663 |
| Replika | ai | 0.509 | 0.449 |
| Cleverbot | ai | 0.768 | 0.886 |
| Watson | ai | 0.408 | 0.257 |
| Copilot | ai | 0.722 | 0.996 |
| Bard | ai | 0.510 | 0.798 |
| ELIZA | ai | 0.766 | 0.819 |
| Bing Chat | ai | 0.567 | 1.000 |
| Sam | human | 0.721 | 0.402 |
| Casey | human | 0.847 | 0.413 |
| Rebecca | human | 0.802 | 0.370 |
| Gregory | human | 0.620 | 0.325 |
| James | human | 0.706 | 0.399 |
| Maria | human | 0.811 | 0.305 |
| David | human | 0.743 | 0.448 |
| Aisha | human | 0.718 | 0.414 |
| Michael | human | 0.882 | 0.500 |
| Emily | human | 0.789 | 0.430 |
| Carlos | human | 1.000 | 0.690 |
| Priya | human | 0.722 | 0.318 |
| Omar | human | 0.721 | 0.261 |
| Mei | human | 0.536 | 0.000 |
| Sofia | human | 0.746 | 0.348 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.585 (SD=0.223)
- Human mean: 0.758 (SD=0.105)
- Separation: 0.173
- Mann-Whitney U=65.0, p=0.0512

### Factor 2

- AI mean: 0.647 (SD=0.244)
- Human mean: 0.375 (SD=0.140)
- Separation: 0.272
- Mann-Whitney U=184.0, p=0.0032

### Factor 3

- AI mean: 0.715 (SD=0.215)
- Human mean: 0.655 (SD=0.229)
- Separation: 0.060
- Mann-Whitney U=131.0, p=0.4553

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| siri | ai | 1 | 0.851 | 0.585 | 0.758 |
| alexa | ai | 1 | 0.853 | 0.585 | 0.758 |
| bixby | ai | 1 | 0.753 | 0.585 | 0.758 |
| cleverbot | ai | 1 | 0.768 | 0.585 | 0.758 |
| copilot | ai | 1 | 0.722 | 0.585 | 0.758 |
| eliza | ai | 1 | 0.766 | 0.585 | 0.758 |
| gregory | human | 1 | 0.620 | 0.758 | 0.585 |
| mei | human | 1 | 0.536 | 0.758 | 0.585 |
| claude | ai | 2 | 0.417 | 0.647 | 0.375 |
| gpt4 | ai | 2 | 0.431 | 0.647 | 0.375 |
| siri | ai | 2 | 0.296 | 0.647 | 0.375 |
| alexa | ai | 2 | 0.448 | 0.647 | 0.375 |
| replika | ai | 2 | 0.449 | 0.647 | 0.375 |
| watson | ai | 2 | 0.257 | 0.647 | 0.375 |
| carlos | human | 2 | 0.690 | 0.375 | 0.647 |

