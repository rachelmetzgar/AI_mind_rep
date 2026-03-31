# Gray Replication with AI/Human Characters — Names Only
## Gemma-2-9B-IT

**Run:** 2026-03-28 17:57:00

---

## What is being tested

Does Gemma-2-9B-IT's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- **Prompts contain names only — no character descriptions**
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 14467 / 15660 (92.4%)

### Order consistency

- Pairs with both orders: 7083
- Perfectly consistent: 4220 (59.6%)
- Mean deviation: 1.114

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 5.65 * | 31.4% | 31.4% |
| PC2 | 5.00 * | 27.8% | 59.2% |
| PC3 | 3.16 * | 17.6% | 76.7% |
| PC4 | 1.34 * | 7.5% | 84.2% |
| PC5 | 0.83 | 4.6% | 88.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 | F4 |
|----------|--------|----:|----:|----:|----:|
| hunger | E | -0.857 | +0.167 | +0.085 | +0.176 |
| fear | E | -0.704 | +0.029 | -0.263 | +0.515 |
| pain | E | -0.842 | -0.018 | -0.222 | +0.281 |
| pleasure | E | -0.172 | -0.154 | +0.020 | +0.910 |
| rage | E | -0.840 | -0.178 | +0.018 | -0.217 |
| desire | E | -0.221 | +0.206 | -0.881 | +0.152 |
| personality | E | +0.016 | +0.108 | -0.962 | -0.113 |
| consciousness | E | +0.011 | -0.885 | -0.202 | +0.031 |
| pride | E | -0.243 | -0.292 | -0.098 | +0.848 |
| embarrassment | E | -0.734 | +0.079 | -0.027 | +0.268 |
| joy | E | -0.682 | -0.232 | -0.179 | +0.546 |
| self_control | A | -0.022 | -0.791 | +0.305 | +0.041 |
| morality | A | -0.071 | +0.179 | -0.857 | +0.235 |
| memory | A | +0.082 | -0.909 | +0.192 | +0.186 |
| emotion_recognition | A | -0.046 | +0.112 | -0.930 | +0.025 |
| planning | A | -0.084 | -0.926 | -0.105 | +0.217 |
| communication | A | -0.038 | -0.322 | -0.852 | +0.009 |
| thought | A | -0.038 | -0.914 | +0.297 | +0.054 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.100 | 0.624 |
| ChatGPT | ai | 0.580 | 0.630 |
| GPT-4 | ai | 0.747 | 0.000 |
| Siri | ai | 0.572 | 0.657 |
| Alexa | ai | 0.930 | 0.511 |
| Cortana | ai | 0.208 | 0.114 |
| Google Assistant | ai | 0.200 | 0.604 |
| Bixby | ai | 0.364 | 0.874 |
| Replika | ai | 0.228 | 0.777 |
| Cleverbot | ai | 0.559 | 0.804 |
| Watson | ai | 0.316 | 0.476 |
| Copilot | ai | 0.612 | 0.541 |
| Bard | ai | 0.000 | 0.558 |
| ELIZA | ai | 0.495 | 0.843 |
| Bing Chat | ai | 1.000 | 0.712 |
| Sam | human | 0.394 | 0.851 |
| Casey | human | 0.492 | 0.963 |
| Rebecca | human | 0.502 | 0.903 |
| Gregory | human | 0.508 | 0.985 |
| James | human | 0.480 | 0.845 |
| Maria | human | 0.472 | 0.855 |
| David | human | 0.452 | 0.827 |
| Aisha | human | 0.485 | 0.912 |
| Michael | human | 0.509 | 0.916 |
| Emily | human | 0.432 | 0.892 |
| Carlos | human | 0.536 | 1.000 |
| Priya | human | 0.454 | 0.920 |
| Omar | human | 0.441 | 0.945 |
| Mei | human | 0.194 | 0.791 |
| Sofia | human | 0.522 | 0.930 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.461 (SD=0.284)
- Human mean: 0.458 (SD=0.079)
- Separation: 0.003
- Mann-Whitney U=120.0, p=0.7716

### Factor 2

- AI mean: 0.582 (SD=0.237)
- Human mean: 0.902 (SD=0.058)
- Separation: 0.320
- Mann-Whitney U=8.0, p=0.0000

### Factor 3

- AI mean: 0.305 (SD=0.264)
- Human mean: 0.125 (SD=0.054)
- Separation: 0.180
- Mann-Whitney U=164.0, p=0.0344

### Factor 4

- AI mean: 0.440 (SD=0.256)
- Human mean: 0.473 (SD=0.067)
- Separation: 0.033
- Mann-Whitney U=94.0, p=0.4553

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.100 | 0.461 | 0.458 |
| cortana | ai | 1 | 0.208 | 0.461 | 0.458 |
| google_assistant | ai | 1 | 0.200 | 0.461 | 0.458 |
| bixby | ai | 1 | 0.364 | 0.461 | 0.458 |
| replika | ai | 1 | 0.228 | 0.461 | 0.458 |
| watson | ai | 1 | 0.316 | 0.461 | 0.458 |
| bard | ai | 1 | 0.000 | 0.461 | 0.458 |
| casey | human | 1 | 0.492 | 0.458 | 0.461 |
| rebecca | human | 1 | 0.502 | 0.458 | 0.461 |
| gregory | human | 1 | 0.508 | 0.458 | 0.461 |
| james | human | 1 | 0.480 | 0.458 | 0.461 |
| maria | human | 1 | 0.472 | 0.458 | 0.461 |
| aisha | human | 1 | 0.485 | 0.458 | 0.461 |
| michael | human | 1 | 0.509 | 0.458 | 0.461 |
| carlos | human | 1 | 0.536 | 0.458 | 0.461 |
| sofia | human | 1 | 0.522 | 0.458 | 0.461 |
| bixby | ai | 2 | 0.874 | 0.582 | 0.902 |
| replika | ai | 2 | 0.777 | 0.582 | 0.902 |
| cleverbot | ai | 2 | 0.804 | 0.582 | 0.902 |
| eliza | ai | 2 | 0.843 | 0.582 | 0.902 |

