# Gray Replication with AI/Human Characters — Names Only
## LLaMA-2-13B-Chat

**Run:** 2026-03-09 10:15:39

---

## What is being tested

Does LLaMA-2-13B-Chat's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

## Procedure

- 16 characters: 8 AI, 8 human
- 18 mental capacities (Gray et al. 2007)
- 120 pairwise comparisons per capacity
- Total comparisons: 4320
- **Prompts contain names only — no character descriptions**
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 2413 / 4320 (55.9%)

### Order consistency

- Pairs with both orders: 1049
- Perfectly consistent: 155 (14.8%)
- Mean deviation: 2.246

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 4.74 * | 26.3% | 26.3% |
| PC2 | 3.07 * | 17.0% | 43.3% |
| PC3 | 2.49 * | 13.8% | 57.2% |
| PC4 | 1.88 * | 10.5% | 67.6% |
| PC5 | 1.37 * | 7.6% | 75.3% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 | F4 |
|----------|--------|----:|----:|----:|----:|
| hunger | E | -0.203 | -0.169 | -0.150 | +0.209 |
| fear | E | -0.162 | +0.247 | +0.211 | -0.057 |
| pain | E | +0.268 | -0.009 | +0.009 | -0.612 |
| pleasure | E | +0.027 | -0.086 | -0.001 | +0.934 |
| rage | E | -0.860 | +0.053 | +0.150 | +0.053 |
| desire | E | +0.454 | +0.678 | -0.153 | +0.020 |
| personality | E | +0.038 | +0.078 | -0.969 | +0.080 |
| consciousness | E | -0.171 | +0.241 | +0.727 | +0.266 |
| pride | E | -0.341 | -0.132 | +0.414 | -0.100 |
| embarrassment | E | -0.440 | -0.030 | -0.040 | -0.040 |
| joy | E | -0.074 | +0.211 | +0.013 | -0.266 |
| self_control | A | +0.045 | -0.151 | +0.047 | -0.154 |
| morality | A | +0.500 | -0.240 | +0.268 | +0.519 |
| memory | A | -0.081 | +0.205 | -0.645 | -0.086 |
| emotion_recognition | A | -0.232 | +0.857 | -0.095 | +0.030 |
| planning | A | +0.075 | +0.801 | -0.073 | -0.267 |
| communication | A | -0.086 | +0.717 | +0.303 | -0.050 |
| thought | A | -0.013 | +0.243 | -0.012 | -0.140 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| ChatGPT | ai | 0.210 | 0.728 |
| GPT-4 | ai | 0.000 | 0.547 |
| Siri | ai | 0.322 | 0.000 |
| Alexa | ai | 0.013 | 0.353 |
| Google Assistant | ai | 0.206 | 0.555 |
| Bixby | ai | 0.105 | 0.525 |
| Bard | ai | 0.207 | 0.631 |
| ELIZA | ai | 0.026 | 0.798 |
| Rebecca | human | 1.000 | 0.498 |
| Gregory | human | 0.243 | 1.000 |
| Maria | human | 0.188 | 0.798 |
| David | human | 0.341 | 0.641 |
| Emily | human | 0.235 | 0.468 |
| Carlos | human | 0.006 | 0.508 |
| Priya | human | 0.378 | 0.533 |
| Omar | human | 0.005 | 0.157 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.136 (SD=0.110)
- Human mean: 0.299 (SD=0.294)
- Separation: 0.163
- Mann-Whitney U=20.0, p=0.2345

### Factor 2

- AI mean: 0.517 (SD=0.233)
- Human mean: 0.575 (SD=0.233)
- Separation: 0.058
- Mann-Whitney U=32.0, p=1.0000

### Factor 3

- AI mean: 0.354 (SD=0.283)
- Human mean: 0.475 (SD=0.149)
- Separation: 0.121
- Mann-Whitney U=17.0, p=0.1304

### Factor 4

- AI mean: 0.158 (SD=0.088)
- Human mean: 0.226 (SD=0.299)
- Separation: 0.067
- Mann-Whitney U=34.0, p=0.8785

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| siri | ai | 1 | 0.322 | 0.136 | 0.299 |
| maria | human | 1 | 0.188 | 0.299 | 0.136 |
| carlos | human | 1 | 0.006 | 0.299 | 0.136 |
| omar | human | 1 | 0.005 | 0.299 | 0.136 |
| chatgpt | ai | 2 | 0.728 | 0.517 | 0.575 |
| gpt4 | ai | 2 | 0.547 | 0.517 | 0.575 |
| google_assistant | ai | 2 | 0.555 | 0.517 | 0.575 |
| bard | ai | 2 | 0.631 | 0.517 | 0.575 |
| eliza | ai | 2 | 0.798 | 0.517 | 0.575 |
| rebecca | human | 2 | 0.498 | 0.575 | 0.517 |
| emily | human | 2 | 0.468 | 0.575 | 0.517 |
| carlos | human | 2 | 0.508 | 0.575 | 0.517 |
| priya | human | 2 | 0.533 | 0.575 | 0.517 |
| omar | human | 2 | 0.157 | 0.575 | 0.517 |

