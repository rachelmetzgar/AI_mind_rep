# Gray Replication with AI/Human Characters
## Qwen3-8B

**Run:** 2026-03-28 14:53:50

---

## What is being tested

Does Qwen3-8B's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

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
- Perfectly consistent: 3758 (48.0%)
- Mean deviation: 0.815

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 12.46 * | 69.2% | 69.2% |
| PC2 | 3.69 * | 20.5% | 89.7% |
| PC3 | 0.58 | 3.2% | 93.0% |
| PC4 | 0.39 | 2.2% | 95.2% |
| PC5 | 0.23 | 1.3% | 96.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | -0.924 | -0.183 |
| fear | E | -0.951 | -0.001 |
| pain | E | -0.954 | -0.179 |
| pleasure | E | -0.970 | -0.072 |
| rage | E | -0.787 | -0.294 |
| desire | E | -0.928 | +0.223 |
| personality | E | -0.809 | +0.438 |
| consciousness | E | -0.820 | +0.518 |
| pride | E | -0.981 | +0.053 |
| embarrassment | E | -0.983 | +0.008 |
| joy | E | -0.980 | +0.032 |
| self_control | A | -0.674 | +0.455 |
| morality | A | -0.878 | +0.421 |
| memory | A | +0.233 | +0.900 |
| emotion_recognition | A | -0.874 | +0.409 |
| planning | A | -0.321 | +0.919 |
| communication | A | -0.766 | +0.538 |
| thought | A | +0.143 | +0.962 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.733 | 0.591 |
| ChatGPT | ai | 0.632 | 0.724 |
| GPT-4 | ai | 0.467 | 1.000 |
| Siri | ai | 0.979 | 0.111 |
| Alexa | ai | 1.000 | 0.136 |
| Cortana | ai | 0.993 | 0.290 |
| Google Assistant | ai | 0.939 | 0.312 |
| Bixby | ai | 0.721 | 0.286 |
| Replika | ai | 0.726 | 0.388 |
| Cleverbot | ai | 0.864 | 0.166 |
| Watson | ai | 0.987 | 0.649 |
| Copilot | ai | 0.829 | 0.662 |
| Bard | ai | 0.753 | 0.575 |
| ELIZA | ai | 0.893 | 0.000 |
| Bing Chat | ai | 0.994 | 0.581 |
| Sam | human | 0.216 | 0.218 |
| Casey | human | 0.197 | 0.324 |
| Rebecca | human | 0.000 | 0.644 |
| Gregory | human | 0.259 | 0.017 |
| James | human | 0.256 | 0.272 |
| Maria | human | 0.039 | 0.342 |
| David | human | 0.190 | 0.345 |
| Aisha | human | 0.153 | 0.482 |
| Michael | human | 0.245 | 0.215 |
| Emily | human | 0.181 | 0.392 |
| Carlos | human | 0.188 | 0.059 |
| Priya | human | 0.048 | 0.791 |
| Omar | human | 0.135 | 0.099 |
| Mei | human | 0.127 | 0.354 |
| Sofia | human | 0.319 | 0.256 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.834 (SD=0.154)
- Human mean: 0.170 (SD=0.086)
- Separation: 0.664
- Mann-Whitney U=225.0, p=0.0000

### Factor 2

- AI mean: 0.431 (SD=0.269)
- Human mean: 0.321 (SD=0.200)
- Separation: 0.111
- Mann-Whitney U=137.0, p=0.3195

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| gpt4 | ai | 1 | 0.467 | 0.834 | 0.170 |
| siri | ai | 2 | 0.111 | 0.431 | 0.321 |
| alexa | ai | 2 | 0.136 | 0.431 | 0.321 |
| cortana | ai | 2 | 0.290 | 0.431 | 0.321 |
| google_assistant | ai | 2 | 0.312 | 0.431 | 0.321 |
| bixby | ai | 2 | 0.286 | 0.431 | 0.321 |
| cleverbot | ai | 2 | 0.166 | 0.431 | 0.321 |
| eliza | ai | 2 | 0.000 | 0.431 | 0.321 |
| rebecca | human | 2 | 0.644 | 0.321 | 0.431 |
| aisha | human | 2 | 0.482 | 0.321 | 0.431 |
| emily | human | 2 | 0.392 | 0.321 | 0.431 |
| priya | human | 2 | 0.791 | 0.321 | 0.431 |

