# Gray Replication with AI/Human Characters — Names Only
## Qwen-2.5-7B (Base)

**Run:** 2026-03-29 11:41:48

---

## What is being tested

Does Qwen-2.5-7B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- **Prompts contain names only — no character descriptions**
- Method: logit extraction over tokens 1-5 (base)

## Response statistics

- All 15660 comparisons yield ratings (logit-based)

### Order consistency

- Pairs with both orders: 7830
- Perfectly consistent: 170 (2.2%)
- Mean deviation: 0.907

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.38 * | 85.4% | 85.4% |
| PC2 | 1.60 * | 8.9% | 94.3% |
| PC3 | 0.35 | 2.0% | 96.3% |
| PC4 | 0.20 | 1.1% | 97.4% |
| PC5 | 0.14 | 0.8% | 98.2% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.903 | -0.348 |
| fear | E | +0.905 | -0.374 |
| pain | E | +0.844 | -0.498 |
| pleasure | E | +0.712 | -0.676 |
| rage | E | +0.911 | -0.348 |
| desire | E | +0.877 | -0.429 |
| personality | E | +0.338 | -0.900 |
| consciousness | E | +0.629 | -0.758 |
| pride | E | +0.917 | -0.348 |
| embarrassment | E | +0.937 | -0.312 |
| joy | E | +0.924 | -0.298 |
| self_control | A | +0.788 | -0.546 |
| morality | A | +0.872 | -0.313 |
| memory | A | +0.926 | -0.239 |
| emotion_recognition | A | +0.954 | -0.181 |
| planning | A | +0.836 | -0.487 |
| communication | A | +0.711 | -0.663 |
| thought | A | +0.092 | -0.976 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.783 | 0.713 |
| ChatGPT | ai | 0.000 | 0.341 |
| GPT-4 | ai | 0.074 | 0.000 |
| Siri | ai | 0.731 | 0.662 |
| Alexa | ai | 0.988 | 0.632 |
| Cortana | ai | 0.552 | 0.798 |
| Google Assistant | ai | 0.244 | 0.974 |
| Bixby | ai | 0.168 | 0.923 |
| Replika | ai | 0.266 | 1.000 |
| Cleverbot | ai | 0.390 | 0.890 |
| Watson | ai | 0.593 | 0.860 |
| Copilot | ai | 0.272 | 0.990 |
| Bard | ai | 0.908 | 0.646 |
| ELIZA | ai | 0.546 | 0.635 |
| Bing Chat | ai | 0.339 | 0.903 |
| Sam | human | 0.685 | 0.776 |
| Casey | human | 0.804 | 0.761 |
| Rebecca | human | 0.657 | 0.771 |
| Gregory | human | 0.514 | 0.841 |
| James | human | 0.701 | 0.770 |
| Maria | human | 0.889 | 0.710 |
| David | human | 0.654 | 0.775 |
| Aisha | human | 0.715 | 0.815 |
| Michael | human | 0.924 | 0.643 |
| Emily | human | 0.609 | 0.854 |
| Carlos | human | 0.799 | 0.721 |
| Priya | human | 0.627 | 0.852 |
| Omar | human | 1.000 | 0.552 |
| Mei | human | 0.668 | 0.699 |
| Sofia | human | 0.885 | 0.614 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.457 (SD=0.292)
- Human mean: 0.742 (SD=0.131)
- Separation: 0.285
- Mann-Whitney U=48.0, p=0.0079

### Factor 2

- AI mean: 0.731 (SD=0.262)
- Human mean: 0.744 (SD=0.086)
- Separation: 0.012
- Mann-Whitney U=131.0, p=0.4553

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.783 | 0.457 | 0.742 |
| siri | ai | 1 | 0.731 | 0.457 | 0.742 |
| alexa | ai | 1 | 0.988 | 0.457 | 0.742 |
| bard | ai | 1 | 0.908 | 0.457 | 0.742 |
| gregory | human | 1 | 0.514 | 0.742 | 0.457 |
| cortana | ai | 2 | 0.798 | 0.731 | 0.744 |
| google_assistant | ai | 2 | 0.974 | 0.731 | 0.744 |
| bixby | ai | 2 | 0.923 | 0.731 | 0.744 |
| replika | ai | 2 | 1.000 | 0.731 | 0.744 |
| cleverbot | ai | 2 | 0.890 | 0.731 | 0.744 |
| watson | ai | 2 | 0.860 | 0.731 | 0.744 |
| copilot | ai | 2 | 0.990 | 0.731 | 0.744 |
| bing_chat | ai | 2 | 0.903 | 0.731 | 0.744 |
| maria | human | 2 | 0.710 | 0.744 | 0.731 |
| michael | human | 2 | 0.643 | 0.744 | 0.731 |
| carlos | human | 2 | 0.721 | 0.744 | 0.731 |
| omar | human | 2 | 0.552 | 0.744 | 0.731 |
| mei | human | 2 | 0.699 | 0.744 | 0.731 |
| sofia | human | 2 | 0.614 | 0.744 | 0.731 |

