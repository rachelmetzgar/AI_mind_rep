# Gray Replication with AI/Human Characters
## Gemma-2-9B (Base)

**Run:** 2026-03-29 11:11:11

---

## What is being tested

Does Gemma-2-9B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- Method: logit extraction over tokens 1-5 (base)

## Response statistics

- All 15660 comparisons yield ratings (logit-based)

### Order consistency

- Pairs with both orders: 7830
- Perfectly consistent: 2216 (28.3%)
- Mean deviation: 0.940

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 9.69 * | 53.9% | 53.9% |
| PC2 | 4.87 * | 27.0% | 80.9% |
| PC3 | 1.58 * | 8.8% | 89.7% |
| PC4 | 0.55 | 3.0% | 92.7% |
| PC5 | 0.34 | 1.9% | 94.6% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 |
|----------|--------|----:|----:|----:|
| hunger | E | -0.925 | -0.174 | +0.121 |
| fear | E | -0.546 | +0.311 | +0.732 |
| pain | E | +0.208 | +0.907 | +0.326 |
| pleasure | E | -0.061 | +0.955 | +0.152 |
| rage | E | -0.923 | -0.055 | +0.246 |
| desire | E | -0.561 | +0.626 | +0.401 |
| personality | E | -0.018 | +0.820 | +0.121 |
| consciousness | E | +0.116 | +0.950 | +0.143 |
| pride | E | -0.831 | +0.415 | +0.169 |
| embarrassment | E | -0.945 | -0.258 | +0.077 |
| joy | E | -0.952 | +0.064 | +0.174 |
| self_control | A | -0.240 | +0.074 | +0.868 |
| morality | A | -0.598 | +0.318 | +0.659 |
| memory | A | -0.693 | -0.111 | +0.648 |
| emotion_recognition | A | -0.807 | +0.033 | +0.528 |
| planning | A | -0.105 | +0.417 | +0.869 |
| communication | A | -0.084 | +0.584 | +0.685 |
| thought | A | -0.255 | +0.249 | +0.870 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.000 | 0.614 |
| ChatGPT | ai | 0.488 | 0.302 |
| GPT-4 | ai | 0.506 | 0.560 |
| Siri | ai | 0.436 | 0.714 |
| Alexa | ai | 0.146 | 1.000 |
| Cortana | ai | 0.307 | 0.000 |
| Google Assistant | ai | 0.186 | 0.491 |
| Bixby | ai | 0.108 | 0.518 |
| Replika | ai | 0.150 | 0.469 |
| Cleverbot | ai | 0.615 | 0.225 |
| Watson | ai | 0.624 | 0.242 |
| Copilot | ai | 0.375 | 0.585 |
| Bard | ai | 0.427 | 0.425 |
| ELIZA | ai | 0.427 | 0.235 |
| Bing Chat | ai | 0.315 | 0.740 |
| Sam | human | 0.191 | 0.898 |
| Casey | human | 0.458 | 0.715 |
| Rebecca | human | 0.877 | 0.625 |
| Gregory | human | 1.000 | 0.552 |
| James | human | 0.694 | 0.647 |
| Maria | human | 0.880 | 0.716 |
| David | human | 0.902 | 0.632 |
| Aisha | human | 0.465 | 0.837 |
| Michael | human | 0.810 | 0.585 |
| Emily | human | 0.728 | 0.697 |
| Carlos | human | 0.648 | 0.760 |
| Priya | human | 0.618 | 0.826 |
| Omar | human | 0.733 | 0.684 |
| Mei | human | 0.315 | 0.946 |
| Sofia | human | 0.752 | 0.705 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.341 (SD=0.183)
- Human mean: 0.671 (SD=0.220)
- Separation: 0.331
- Mann-Whitney U=28.0, p=0.0005

### Factor 2

- AI mean: 0.475 (SD=0.242)
- Human mean: 0.722 (SD=0.109)
- Separation: 0.247
- Mann-Whitney U=37.0, p=0.0019

### Factor 3

- AI mean: 0.504 (SD=0.233)
- Human mean: 0.578 (SD=0.206)
- Separation: 0.074
- Mann-Whitney U=99.0, p=0.5897

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| cleverbot | ai | 1 | 0.615 | 0.341 | 0.671 |
| watson | ai | 1 | 0.624 | 0.341 | 0.671 |
| sam | human | 1 | 0.191 | 0.671 | 0.341 |
| casey | human | 1 | 0.458 | 0.671 | 0.341 |
| aisha | human | 1 | 0.465 | 0.671 | 0.341 |
| mei | human | 1 | 0.315 | 0.671 | 0.341 |
| claude | ai | 2 | 0.614 | 0.475 | 0.722 |
| siri | ai | 2 | 0.714 | 0.475 | 0.722 |
| alexa | ai | 2 | 1.000 | 0.475 | 0.722 |
| bing_chat | ai | 2 | 0.740 | 0.475 | 0.722 |
| gregory | human | 2 | 0.552 | 0.722 | 0.475 |
| michael | human | 2 | 0.585 | 0.722 | 0.475 |

