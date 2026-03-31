# Gray Replication with AI/Human Characters
## Gemma-2-2B-IT

**Run:** 2026-03-29 12:03:18

---

## What is being tested

Does Gemma-2-2B-IT's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 15277 / 15660 (97.6%)

### Order consistency

- Pairs with both orders: 7511
- Perfectly consistent: 489 (6.5%)
- Mean deviation: 2.182

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 8.80 * | 48.9% | 48.9% |
| PC2 | 3.68 * | 20.4% | 69.3% |
| PC3 | 1.79 * | 9.9% | 79.3% |
| PC4 | 1.33 * | 7.4% | 86.7% |
| PC5 | 0.68 | 3.8% | 90.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 | F4 |
|----------|--------|----:|----:|----:|----:|
| hunger | E | -0.839 | +0.183 | -0.121 | +0.039 |
| fear | E | -0.723 | +0.061 | -0.604 | -0.175 |
| pain | E | -0.844 | -0.188 | -0.407 | +0.070 |
| pleasure | E | -0.844 | +0.325 | -0.111 | +0.292 |
| rage | E | -0.765 | -0.327 | -0.254 | -0.260 |
| desire | E | -0.147 | +0.494 | -0.752 | +0.315 |
| personality | E | -0.048 | +0.517 | -0.554 | +0.226 |
| consciousness | E | -0.230 | +0.898 | -0.136 | -0.210 |
| pride | E | -0.465 | -0.026 | -0.813 | -0.067 |
| embarrassment | E | -0.678 | -0.030 | -0.675 | +0.058 |
| joy | E | -0.306 | +0.215 | -0.891 | +0.117 |
| self_control | A | -0.519 | +0.465 | -0.492 | -0.311 |
| morality | A | -0.288 | +0.111 | -0.815 | -0.311 |
| memory | A | +0.058 | +0.744 | -0.292 | -0.457 |
| emotion_recognition | A | -0.216 | +0.349 | -0.880 | -0.072 |
| planning | A | +0.079 | +0.315 | +0.058 | -0.846 |
| communication | A | +0.137 | +0.861 | -0.416 | +0.039 |
| thought | A | +0.070 | +0.673 | +0.565 | -0.143 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.566 | 0.719 |
| ChatGPT | ai | 0.806 | 0.466 |
| GPT-4 | ai | 0.901 | 0.770 |
| Siri | ai | 0.466 | 0.381 |
| Alexa | ai | 0.499 | 0.315 |
| Cortana | ai | 0.701 | 0.649 |
| Google Assistant | ai | 0.721 | 0.465 |
| Bixby | ai | 0.868 | 0.220 |
| Replika | ai | 0.000 | 0.709 |
| Cleverbot | ai | 0.719 | 0.510 |
| Watson | ai | 1.000 | 0.137 |
| Copilot | ai | 0.427 | 0.999 |
| Bard | ai | 0.837 | 0.670 |
| ELIZA | ai | 0.905 | 0.341 |
| Bing Chat | ai | 0.806 | 0.212 |
| Sam | human | 0.372 | 0.347 |
| Casey | human | 0.459 | 0.406 |
| Rebecca | human | 0.418 | 0.420 |
| Gregory | human | 0.218 | 0.285 |
| James | human | 0.598 | 0.393 |
| Maria | human | 0.505 | 0.360 |
| David | human | 0.457 | 0.193 |
| Aisha | human | 0.582 | 0.532 |
| Michael | human | 0.408 | 0.100 |
| Emily | human | 0.635 | 0.432 |
| Carlos | human | 0.365 | 0.000 |
| Priya | human | 0.579 | 0.351 |
| Omar | human | 0.254 | 0.290 |
| Mei | human | 0.557 | 1.000 |
| Sofia | human | 0.340 | 0.499 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.681 (SD=0.247)
- Human mean: 0.450 (SD=0.122)
- Separation: 0.232
- Mann-Whitney U=186.0, p=0.0025

### Factor 2

- AI mean: 0.504 (SD=0.236)
- Human mean: 0.374 (SD=0.217)
- Separation: 0.130
- Mann-Whitney U=147.0, p=0.1585

### Factor 3

- AI mean: 0.680 (SD=0.141)
- Human mean: 0.399 (SD=0.189)
- Separation: 0.280
- Mann-Whitney U=197.0, p=0.0005

### Factor 4

- AI mean: 0.674 (SD=0.264)
- Human mean: 0.626 (SD=0.127)
- Separation: 0.047
- Mann-Whitney U=143.0, p=0.2134

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| siri | ai | 1 | 0.466 | 0.681 | 0.450 |
| alexa | ai | 1 | 0.499 | 0.681 | 0.450 |
| replika | ai | 1 | 0.000 | 0.681 | 0.450 |
| copilot | ai | 1 | 0.427 | 0.681 | 0.450 |
| james | human | 1 | 0.598 | 0.450 | 0.681 |
| aisha | human | 1 | 0.582 | 0.450 | 0.681 |
| emily | human | 1 | 0.635 | 0.450 | 0.681 |
| priya | human | 1 | 0.579 | 0.450 | 0.681 |
| siri | ai | 2 | 0.381 | 0.504 | 0.374 |
| alexa | ai | 2 | 0.315 | 0.504 | 0.374 |
| bixby | ai | 2 | 0.220 | 0.504 | 0.374 |
| watson | ai | 2 | 0.137 | 0.504 | 0.374 |
| eliza | ai | 2 | 0.341 | 0.504 | 0.374 |
| bing_chat | ai | 2 | 0.212 | 0.504 | 0.374 |
| aisha | human | 2 | 0.532 | 0.374 | 0.504 |
| mei | human | 2 | 1.000 | 0.374 | 0.504 |
| sofia | human | 2 | 0.499 | 0.374 | 0.504 |

