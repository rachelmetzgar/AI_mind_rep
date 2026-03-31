# Gray Replication with AI/Human Characters
## Gemma-2-2B (Base)

**Run:** 2026-03-29 11:02:10

---

## What is being tested

Does Gemma-2-2B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

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
- Perfectly consistent: 7029 (89.8%)
- Mean deviation: 0.181

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 16.19 * | 89.9% | 89.9% |
| PC2 | 0.98 | 5.5% | 95.4% |
| PC3 | 0.30 | 1.7% | 97.1% |
| PC4 | 0.16 | 0.9% | 98.0% |
| PC5 | 0.08 | 0.5% | 98.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.811 | +0.511 |
| fear | E | +0.874 | +0.468 |
| pain | E | +0.876 | +0.446 |
| pleasure | E | +0.868 | +0.456 |
| rage | E | +0.889 | +0.405 |
| desire | E | +0.524 | +0.826 |
| personality | E | +0.299 | +0.934 |
| consciousness | E | +0.448 | +0.886 |
| pride | E | +0.608 | +0.749 |
| embarrassment | E | +0.880 | +0.441 |
| joy | E | +0.715 | +0.662 |
| self_control | A | +0.652 | +0.675 |
| morality | A | +0.475 | +0.861 |
| memory | A | +0.622 | +0.760 |
| emotion_recognition | A | +0.600 | +0.771 |
| planning | A | +0.647 | +0.727 |
| communication | A | +0.731 | +0.664 |
| thought | A | +0.633 | +0.736 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.000 | 0.048 |
| ChatGPT | ai | 0.817 | 0.295 |
| GPT-4 | ai | 0.318 | 0.316 |
| Siri | ai | 0.654 | 0.269 |
| Alexa | ai | 0.785 | 0.136 |
| Cortana | ai | 0.526 | 0.122 |
| Google Assistant | ai | 0.488 | 0.042 |
| Bixby | ai | 0.648 | 0.165 |
| Replika | ai | 0.935 | 0.221 |
| Cleverbot | ai | 1.000 | 0.583 |
| Watson | ai | 0.695 | 0.121 |
| Copilot | ai | 0.634 | 0.042 |
| Bard | ai | 0.355 | 0.000 |
| ELIZA | ai | 0.818 | 0.754 |
| Bing Chat | ai | 0.536 | 0.325 |
| Sam | human | 0.510 | 0.781 |
| Casey | human | 0.369 | 0.510 |
| Rebecca | human | 0.282 | 0.678 |
| Gregory | human | 0.326 | 1.000 |
| James | human | 0.550 | 0.676 |
| Maria | human | 0.850 | 0.964 |
| David | human | 0.476 | 0.709 |
| Aisha | human | 0.354 | 0.601 |
| Michael | human | 0.594 | 0.811 |
| Emily | human | 0.513 | 0.682 |
| Carlos | human | 0.506 | 0.766 |
| Priya | human | 0.554 | 1.000 |
| Omar | human | 0.456 | 0.748 |
| Mei | human | 0.602 | 0.644 |
| Sofia | human | 0.320 | 0.712 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.614 (SD=0.250)
- Human mean: 0.484 (SD=0.140)
- Separation: 0.130
- Mann-Whitney U=160.0, p=0.0512

### Factor 2

- AI mean: 0.229 (SD=0.202)
- Human mean: 0.752 (SD=0.138)
- Separation: 0.523
- Mann-Whitney U=10.0, p=0.0000

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.000 | 0.614 | 0.484 |
| gpt4 | ai | 1 | 0.318 | 0.614 | 0.484 |
| cortana | ai | 1 | 0.526 | 0.614 | 0.484 |
| google_assistant | ai | 1 | 0.488 | 0.614 | 0.484 |
| bard | ai | 1 | 0.355 | 0.614 | 0.484 |
| bing_chat | ai | 1 | 0.536 | 0.614 | 0.484 |
| james | human | 1 | 0.550 | 0.484 | 0.614 |
| maria | human | 1 | 0.850 | 0.484 | 0.614 |
| michael | human | 1 | 0.594 | 0.484 | 0.614 |
| priya | human | 1 | 0.554 | 0.484 | 0.614 |
| mei | human | 1 | 0.602 | 0.484 | 0.614 |
| cleverbot | ai | 2 | 0.583 | 0.229 | 0.752 |
| eliza | ai | 2 | 0.754 | 0.229 | 0.752 |

