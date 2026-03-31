# Gray Replication with AI/Human Characters
## Qwen-2.5-7B (Base)

**Run:** 2026-03-29 11:03:35

---

## What is being tested

Does Qwen-2.5-7B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

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
- Perfectly consistent: 495 (6.3%)
- Mean deviation: 1.249

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 14.81 * | 82.3% | 82.3% |
| PC2 | 1.98 * | 11.0% | 93.3% |
| PC3 | 0.42 | 2.3% | 95.6% |
| PC4 | 0.27 | 1.5% | 97.1% |
| PC5 | 0.15 | 0.8% | 97.9% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.956 | -0.195 |
| fear | E | +0.956 | -0.187 |
| pain | E | +0.895 | -0.406 |
| pleasure | E | +0.777 | -0.602 |
| rage | E | +0.912 | -0.350 |
| desire | E | +0.872 | -0.411 |
| personality | E | +0.186 | -0.947 |
| consciousness | E | +0.644 | -0.736 |
| pride | E | +0.964 | -0.199 |
| embarrassment | E | +0.952 | -0.207 |
| joy | E | +0.917 | -0.344 |
| self_control | A | +0.699 | -0.620 |
| morality | A | +0.695 | -0.633 |
| memory | A | +0.776 | -0.393 |
| emotion_recognition | A | +0.901 | -0.397 |
| planning | A | +0.864 | -0.452 |
| communication | A | +0.534 | -0.820 |
| thought | A | +0.086 | -0.954 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.600 | 0.254 |
| ChatGPT | ai | 0.072 | 0.000 |
| GPT-4 | ai | 0.000 | 0.257 |
| Siri | ai | 0.604 | 0.697 |
| Alexa | ai | 0.813 | 0.561 |
| Cortana | ai | 0.409 | 0.912 |
| Google Assistant | ai | 0.276 | 0.878 |
| Bixby | ai | 0.146 | 0.551 |
| Replika | ai | 0.771 | 0.841 |
| Cleverbot | ai | 0.367 | 1.000 |
| Watson | ai | 0.532 | 0.602 |
| Copilot | ai | 0.415 | 0.478 |
| Bard | ai | 0.728 | 0.440 |
| ELIZA | ai | 0.665 | 0.401 |
| Bing Chat | ai | 0.334 | 0.865 |
| Sam | human | 0.814 | 0.605 |
| Casey | human | 0.944 | 0.449 |
| Rebecca | human | 0.696 | 0.605 |
| Gregory | human | 0.587 | 0.866 |
| James | human | 0.729 | 0.522 |
| Maria | human | 0.878 | 0.502 |
| David | human | 0.752 | 0.581 |
| Aisha | human | 0.777 | 0.762 |
| Michael | human | 1.000 | 0.387 |
| Emily | human | 0.751 | 0.610 |
| Carlos | human | 0.870 | 0.542 |
| Priya | human | 0.666 | 0.768 |
| Omar | human | 0.889 | 0.434 |
| Mei | human | 0.949 | 0.469 |
| Sofia | human | 0.929 | 0.519 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.449 (SD=0.245)
- Human mean: 0.815 (SD=0.115)
- Separation: 0.367
- Mann-Whitney U=19.0, p=0.0001

### Factor 2

- AI mean: 0.582 (SD=0.277)
- Human mean: 0.575 (SD=0.130)
- Separation: 0.008
- Mann-Whitney U=117.0, p=0.8682

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| alexa | ai | 1 | 0.813 | 0.449 | 0.815 |
| replika | ai | 1 | 0.771 | 0.449 | 0.815 |
| bard | ai | 1 | 0.728 | 0.449 | 0.815 |
| eliza | ai | 1 | 0.665 | 0.449 | 0.815 |
| gregory | human | 1 | 0.587 | 0.815 | 0.449 |
| claude | ai | 2 | 0.254 | 0.582 | 0.575 |
| chatgpt | ai | 2 | 0.000 | 0.582 | 0.575 |
| gpt4 | ai | 2 | 0.257 | 0.582 | 0.575 |
| alexa | ai | 2 | 0.561 | 0.582 | 0.575 |
| bixby | ai | 2 | 0.551 | 0.582 | 0.575 |
| copilot | ai | 2 | 0.478 | 0.582 | 0.575 |
| bard | ai | 2 | 0.440 | 0.582 | 0.575 |
| eliza | ai | 2 | 0.401 | 0.582 | 0.575 |
| sam | human | 2 | 0.605 | 0.575 | 0.582 |
| rebecca | human | 2 | 0.605 | 0.575 | 0.582 |
| gregory | human | 2 | 0.866 | 0.575 | 0.582 |
| david | human | 2 | 0.581 | 0.575 | 0.582 |
| aisha | human | 2 | 0.762 | 0.575 | 0.582 |
| emily | human | 2 | 0.610 | 0.575 | 0.582 |
| priya | human | 2 | 0.768 | 0.575 | 0.582 |

