# Gray Replication with AI/Human Characters
## Gemma-2-9B-IT

**Run:** 2026-03-28 15:47:45

---

## What is being tested

Does Gemma-2-9B-IT's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 15660 / 15660 (100.0%)

### Order consistency

- Pairs with both orders: 7830
- Perfectly consistent: 5909 (75.5%)
- Mean deviation: 0.597

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.17 * | 84.3% | 84.3% |
| PC2 | 1.75 * | 9.7% | 94.0% |
| PC3 | 0.63 | 3.5% | 97.5% |
| PC4 | 0.13 | 0.7% | 98.3% |
| PC5 | 0.11 | 0.6% | 98.9% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.923 | +0.360 |
| fear | E | +0.921 | +0.367 |
| pain | E | +0.931 | +0.340 |
| pleasure | E | +0.916 | +0.385 |
| rage | E | +0.851 | -0.052 |
| desire | E | +0.893 | +0.429 |
| personality | E | +0.858 | +0.499 |
| consciousness | E | +0.848 | +0.520 |
| pride | E | +0.894 | +0.436 |
| embarrassment | E | +0.895 | -0.075 |
| joy | E | +0.869 | +0.471 |
| self_control | A | +0.781 | +0.570 |
| morality | A | +0.799 | +0.570 |
| memory | A | +0.044 | +0.928 |
| emotion_recognition | A | +0.831 | +0.517 |
| planning | A | +0.482 | +0.821 |
| communication | A | +0.771 | +0.607 |
| thought | A | +0.420 | +0.874 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.301 | 0.681 |
| ChatGPT | ai | 0.210 | 0.562 |
| GPT-4 | ai | 0.000 | 1.000 |
| Siri | ai | 0.500 | 0.053 |
| Alexa | ai | 0.540 | 0.044 |
| Cortana | ai | 0.356 | 0.436 |
| Google Assistant | ai | 0.253 | 0.446 |
| Bixby | ai | 0.584 | 0.000 |
| Replika | ai | 0.681 | 0.041 |
| Cleverbot | ai | 0.451 | 0.298 |
| Watson | ai | 0.250 | 0.581 |
| Copilot | ai | 0.315 | 0.505 |
| Bard | ai | 0.261 | 0.493 |
| ELIZA | ai | 0.421 | 0.013 |
| Bing Chat | ai | 0.275 | 0.563 |
| Sam | human | 0.900 | 0.561 |
| Casey | human | 0.861 | 0.596 |
| Rebecca | human | 0.856 | 0.636 |
| Gregory | human | 1.000 | 0.425 |
| James | human | 0.898 | 0.566 |
| Maria | human | 0.913 | 0.553 |
| David | human | 0.884 | 0.599 |
| Aisha | human | 0.905 | 0.585 |
| Michael | human | 0.926 | 0.490 |
| Emily | human | 0.911 | 0.549 |
| Carlos | human | 0.991 | 0.432 |
| Priya | human | 0.867 | 0.636 |
| Omar | human | 0.900 | 0.570 |
| Mei | human | 0.909 | 0.527 |
| Sofia | human | 0.899 | 0.524 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.360 (SD=0.166)
- Human mean: 0.908 (SD=0.039)
- Separation: 0.548
- Mann-Whitney U=0.0, p=0.0000

### Factor 2

- AI mean: 0.381 (SD=0.287)
- Human mean: 0.550 (SD=0.061)
- Separation: 0.169
- Mann-Whitney U=66.0, p=0.0564

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| replika | ai | 1 | 0.681 | 0.360 | 0.908 |
| claude | ai | 2 | 0.681 | 0.381 | 0.550 |
| chatgpt | ai | 2 | 0.562 | 0.381 | 0.550 |
| gpt4 | ai | 2 | 1.000 | 0.381 | 0.550 |
| watson | ai | 2 | 0.581 | 0.381 | 0.550 |
| copilot | ai | 2 | 0.505 | 0.381 | 0.550 |
| bard | ai | 2 | 0.493 | 0.381 | 0.550 |
| bing_chat | ai | 2 | 0.563 | 0.381 | 0.550 |
| gregory | human | 2 | 0.425 | 0.550 | 0.381 |
| carlos | human | 2 | 0.432 | 0.550 | 0.381 |

