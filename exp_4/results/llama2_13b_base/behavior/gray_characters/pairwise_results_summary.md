# Gray Replication with AI/Human Characters
## LLaMA-2-13B (Base)

**Run:** 2026-03-08 22:40:55

---

## What is being tested

Does LLaMA-2-13B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

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
- Perfectly consistent: 28 (0.4%)
- Mean deviation: 1.104

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 16.45 * | 91.4% | 91.4% |
| PC2 | 0.55 | 3.0% | 94.4% |
| PC3 | 0.34 | 1.9% | 96.3% |
| PC4 | 0.16 | 0.9% | 97.2% |
| PC5 | 0.12 | 0.7% | 97.9% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.900 | +0.397 |
| fear | E | +0.851 | +0.487 |
| pain | E | +0.573 | +0.772 |
| pleasure | E | +0.543 | +0.795 |
| rage | E | +0.716 | +0.674 |
| desire | E | +0.755 | +0.621 |
| personality | E | +0.384 | +0.890 |
| consciousness | E | +0.616 | +0.755 |
| pride | E | +0.688 | +0.696 |
| embarrassment | E | +0.798 | +0.539 |
| joy | E | +0.837 | +0.523 |
| self_control | A | +0.634 | +0.726 |
| morality | A | +0.609 | +0.728 |
| memory | A | +0.759 | +0.615 |
| emotion_recognition | A | +0.754 | +0.619 |
| planning | A | +0.663 | +0.721 |
| communication | A | +0.537 | +0.826 |
| thought | A | +0.671 | +0.658 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.777 | 0.517 |
| ChatGPT | ai | 0.573 | 0.830 |
| GPT-4 | ai | 0.791 | 0.918 |
| Siri | ai | 0.428 | 0.000 |
| Alexa | ai | 0.734 | 0.429 |
| Cortana | ai | 1.000 | 0.318 |
| Google Assistant | ai | 0.415 | 0.325 |
| Bixby | ai | 0.519 | 0.314 |
| Replika | ai | 0.638 | 0.866 |
| Cleverbot | ai | 0.318 | 0.488 |
| Watson | ai | 0.509 | 0.521 |
| Copilot | ai | 0.678 | 0.728 |
| Bard | ai | 0.326 | 0.416 |
| ELIZA | ai | 0.480 | 0.568 |
| Bing Chat | ai | 0.983 | 0.238 |
| Sam | human | 0.144 | 0.614 |
| Casey | human | 0.503 | 0.863 |
| Rebecca | human | 0.727 | 0.938 |
| Gregory | human | 0.000 | 0.625 |
| James | human | 0.310 | 0.899 |
| Maria | human | 0.502 | 0.728 |
| David | human | 0.505 | 0.791 |
| Aisha | human | 0.280 | 0.883 |
| Michael | human | 0.590 | 0.738 |
| Emily | human | 0.620 | 0.887 |
| Carlos | human | 0.456 | 0.729 |
| Priya | human | 0.876 | 1.000 |
| Omar | human | 0.558 | 0.889 |
| Mei | human | 0.424 | 0.636 |
| Sofia | human | 0.561 | 0.719 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.611 (SD=0.208)
- Human mean: 0.470 (SD=0.213)
- Separation: 0.141
- Mann-Whitney U=150.0, p=0.1249

### Factor 2

- AI mean: 0.498 (SD=0.245)
- Human mean: 0.796 (SD=0.118)
- Separation: 0.297
- Mann-Whitney U=35.0, p=0.0014

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| siri | ai | 1 | 0.428 | 0.611 | 0.470 |
| google_assistant | ai | 1 | 0.415 | 0.611 | 0.470 |
| bixby | ai | 1 | 0.519 | 0.611 | 0.470 |
| cleverbot | ai | 1 | 0.318 | 0.611 | 0.470 |
| watson | ai | 1 | 0.509 | 0.611 | 0.470 |
| bard | ai | 1 | 0.326 | 0.611 | 0.470 |
| eliza | ai | 1 | 0.480 | 0.611 | 0.470 |
| rebecca | human | 1 | 0.727 | 0.470 | 0.611 |
| michael | human | 1 | 0.590 | 0.470 | 0.611 |
| emily | human | 1 | 0.620 | 0.470 | 0.611 |
| priya | human | 1 | 0.876 | 0.470 | 0.611 |
| omar | human | 1 | 0.558 | 0.470 | 0.611 |
| sofia | human | 1 | 0.561 | 0.470 | 0.611 |
| chatgpt | ai | 2 | 0.830 | 0.498 | 0.796 |
| gpt4 | ai | 2 | 0.918 | 0.498 | 0.796 |
| replika | ai | 2 | 0.866 | 0.498 | 0.796 |
| copilot | ai | 2 | 0.728 | 0.498 | 0.796 |
| sam | human | 2 | 0.614 | 0.796 | 0.498 |
| gregory | human | 2 | 0.625 | 0.796 | 0.498 |
| mei | human | 2 | 0.636 | 0.796 | 0.498 |

