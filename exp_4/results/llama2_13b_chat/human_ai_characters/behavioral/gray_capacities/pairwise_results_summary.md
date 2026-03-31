# Gray Replication with AI/Human Characters
## LLaMA-2-13B-Chat

**Run:** 2026-03-09 09:22:18

---

## What is being tested

Does LLaMA-2-13B-Chat's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

## Procedure

- 16 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 120 pairwise comparisons per capacity
- Total comparisons: 4320
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 3008 / 4320 (69.6%)

### Order consistency

- Pairs with both orders: 1390
- Perfectly consistent: 225 (16.2%)
- Mean deviation: 1.861

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 8.56 * | 47.5% | 47.5% |
| PC2 | 1.98 * | 11.0% | 58.5% |
| PC3 | 1.64 * | 9.1% | 67.6% |
| PC4 | 1.27 * | 7.0% | 74.6% |
| PC5 | 1.21 * | 6.7% | 81.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 | F4 |
|----------|--------|----:|----:|----:|----:|
| hunger | E | +0.857 | -0.022 | -0.120 | +0.320 |
| fear | E | +0.404 | +0.173 | -0.375 | +0.189 |
| pain | E | -0.293 | -0.075 | -0.082 | -0.757 |
| pleasure | E | -0.082 | -0.085 | +0.001 | -0.934 |
| rage | E | +0.005 | +0.025 | -0.944 | -0.058 |
| desire | E | +0.809 | -0.282 | -0.284 | +0.325 |
| personality | E | +0.190 | -0.907 | +0.138 | -0.036 |
| consciousness | E | +0.229 | +0.059 | +0.038 | +0.197 |
| pride | E | -0.263 | -0.705 | -0.205 | -0.179 |
| embarrassment | E | +0.750 | -0.104 | +0.032 | +0.427 |
| joy | E | +0.709 | +0.096 | -0.054 | +0.187 |
| self_control | A | +0.562 | +0.245 | +0.035 | +0.248 |
| morality | A | -0.153 | -0.140 | +0.101 | -0.136 |
| memory | A | +0.594 | -0.202 | +0.356 | -0.095 |
| emotion_recognition | A | +0.852 | +0.185 | +0.212 | -0.116 |
| planning | A | +0.768 | -0.072 | +0.095 | -0.142 |
| communication | A | +0.543 | +0.088 | -0.027 | +0.297 |
| thought | A | +0.231 | -0.209 | -0.139 | +0.126 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| ChatGPT | ai | 0.394 | 0.857 |
| GPT-4 | ai | 0.000 | 0.448 |
| Siri | ai | 0.851 | 0.691 |
| Alexa | ai | 1.000 | 0.607 |
| Google Assistant | ai | 0.776 | 0.000 |
| Bixby | ai | 0.776 | 0.306 |
| Bard | ai | 0.508 | 0.721 |
| ELIZA | ai | 0.499 | 0.683 |
| Rebecca | human | 0.201 | 0.743 |
| Gregory | human | 0.293 | 0.271 |
| Maria | human | 0.450 | 0.213 |
| David | human | 0.172 | 0.216 |
| Emily | human | 0.419 | 0.724 |
| Carlos | human | 0.096 | 0.223 |
| Priya | human | 0.278 | 1.000 |
| Omar | human | 0.237 | 0.484 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.600 (SD=0.297)
- Human mean: 0.268 (SD=0.112)
- Separation: 0.332
- Mann-Whitney U=54.0, p=0.0207

### Factor 2

- AI mean: 0.539 (SD=0.259)
- Human mean: 0.484 (SD=0.285)
- Separation: 0.055
- Mann-Whitney U=35.0, p=0.7984

### Factor 3

- AI mean: 0.457 (SD=0.259)
- Human mean: 0.453 (SD=0.055)
- Separation: 0.004
- Mann-Whitney U=27.0, p=0.6454

### Factor 4

- AI mean: 0.509 (SD=0.208)
- Human mean: 0.362 (SD=0.207)
- Separation: 0.147
- Mann-Whitney U=35.0, p=0.7984

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| chatgpt | ai | 1 | 0.394 | 0.600 | 0.268 |
| gpt4 | ai | 1 | 0.000 | 0.600 | 0.268 |
| maria | human | 1 | 0.450 | 0.268 | 0.600 |
| gpt4 | ai | 2 | 0.448 | 0.539 | 0.484 |
| google_assistant | ai | 2 | 0.000 | 0.539 | 0.484 |
| bixby | ai | 2 | 0.306 | 0.539 | 0.484 |
| rebecca | human | 2 | 0.743 | 0.484 | 0.539 |
| emily | human | 2 | 0.724 | 0.484 | 0.539 |
| priya | human | 2 | 1.000 | 0.484 | 0.539 |

