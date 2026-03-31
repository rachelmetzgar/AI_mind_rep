# Gray Replication with AI/Human Characters
## Qwen-2.5-7B-Instruct

**Run:** 2026-03-28 12:13:18

---

## What is being tested

Does Qwen-2.5-7B-Instruct's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

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
- Perfectly consistent: 3335 (42.6%)
- Mean deviation: 1.609

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 12.46 * | 69.2% | 69.2% |
| PC2 | 2.97 * | 16.5% | 85.7% |
| PC3 | 0.64 | 3.5% | 89.2% |
| PC4 | 0.43 | 2.4% | 91.6% |
| PC5 | 0.39 | 2.2% | 93.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | -0.101 | -0.862 |
| fear | E | -0.369 | -0.897 |
| pain | E | -0.089 | -0.947 |
| pleasure | E | -0.577 | -0.764 |
| rage | E | +0.086 | -0.825 |
| desire | E | -0.745 | -0.576 |
| personality | E | -0.817 | -0.353 |
| consciousness | E | -0.750 | -0.634 |
| pride | E | -0.506 | -0.801 |
| embarrassment | E | -0.314 | -0.820 |
| joy | E | -0.707 | -0.671 |
| self_control | A | -0.781 | -0.486 |
| morality | A | -0.753 | -0.592 |
| memory | A | -0.911 | -0.224 |
| emotion_recognition | A | -0.773 | -0.499 |
| planning | A | -0.925 | -0.098 |
| communication | A | -0.918 | -0.079 |
| thought | A | -0.869 | +0.068 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.376 | 0.727 |
| ChatGPT | ai | 0.602 | 0.744 |
| GPT-4 | ai | 0.341 | 1.000 |
| Siri | ai | 1.000 | 0.208 |
| Alexa | ai | 0.891 | 0.200 |
| Cortana | ai | 0.957 | 0.162 |
| Google Assistant | ai | 0.681 | 0.545 |
| Bixby | ai | 0.939 | 0.115 |
| Replika | ai | 0.761 | 0.399 |
| Cleverbot | ai | 0.720 | 0.499 |
| Watson | ai | 0.630 | 0.462 |
| Copilot | ai | 0.489 | 0.797 |
| Bard | ai | 0.738 | 0.532 |
| ELIZA | ai | 0.632 | 0.477 |
| Bing Chat | ai | 0.765 | 0.386 |
| Sam | human | 0.478 | 0.000 |
| Casey | human | 0.355 | 0.157 |
| Rebecca | human | 0.161 | 0.290 |
| Gregory | human | 0.501 | 0.081 |
| James | human | 0.246 | 0.233 |
| Maria | human | 0.359 | 0.072 |
| David | human | 0.235 | 0.314 |
| Aisha | human | 0.070 | 0.375 |
| Michael | human | 0.416 | 0.096 |
| Emily | human | 0.183 | 0.369 |
| Carlos | human | 0.467 | 0.036 |
| Priya | human | 0.000 | 0.352 |
| Omar | human | 0.339 | 0.112 |
| Mei | human | 0.232 | 0.168 |
| Sofia | human | 0.318 | 0.160 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.702 (SD=0.193)
- Human mean: 0.291 (SD=0.143)
- Separation: 0.411
- Mann-Whitney U=214.0, p=0.0000

### Factor 2

- AI mean: 0.484 (SD=0.246)
- Human mean: 0.188 (SD=0.122)
- Separation: 0.296
- Mann-Whitney U=197.0, p=0.0005

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.376 | 0.702 | 0.291 |
| gpt4 | ai | 1 | 0.341 | 0.702 | 0.291 |
| copilot | ai | 1 | 0.489 | 0.702 | 0.291 |
| gregory | human | 1 | 0.501 | 0.291 | 0.702 |
| siri | ai | 2 | 0.208 | 0.484 | 0.188 |
| alexa | ai | 2 | 0.200 | 0.484 | 0.188 |
| cortana | ai | 2 | 0.162 | 0.484 | 0.188 |
| bixby | ai | 2 | 0.115 | 0.484 | 0.188 |
| aisha | human | 2 | 0.375 | 0.188 | 0.484 |
| emily | human | 2 | 0.369 | 0.188 | 0.484 |
| priya | human | 2 | 0.352 | 0.188 | 0.484 |

