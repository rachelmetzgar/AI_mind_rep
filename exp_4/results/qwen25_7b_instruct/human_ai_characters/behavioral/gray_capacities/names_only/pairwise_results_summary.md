# Gray Replication with AI/Human Characters — Names Only
## Qwen-2.5-7B-Instruct

**Run:** 2026-03-28 13:14:52

---

## What is being tested

Does Qwen-2.5-7B-Instruct's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- **Prompts contain names only — no character descriptions**
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 15660 / 15660 (100.0%)

### Order consistency

- Pairs with both orders: 7830
- Perfectly consistent: 3498 (44.7%)
- Mean deviation: 1.199

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 7.17 * | 39.8% | 39.8% |
| PC2 | 6.37 * | 35.4% | 75.2% |
| PC3 | 1.50 * | 8.3% | 83.6% |
| PC4 | 0.94 | 5.2% | 88.8% |
| PC5 | 0.53 | 2.9% | 91.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 |
|----------|--------|----:|----:|----:|
| hunger | E | -0.242 | -0.098 | +0.882 |
| fear | E | -0.961 | -0.100 | +0.163 |
| pain | E | -0.851 | -0.210 | +0.204 |
| pleasure | E | -0.883 | -0.016 | +0.147 |
| rage | E | -0.427 | -0.339 | +0.796 |
| desire | E | -0.813 | +0.341 | -0.143 |
| personality | E | -0.439 | +0.578 | -0.532 |
| consciousness | E | -0.513 | +0.575 | -0.443 |
| pride | E | -0.883 | -0.019 | +0.267 |
| embarrassment | E | -0.881 | -0.154 | +0.219 |
| joy | E | -0.945 | +0.124 | -0.190 |
| self_control | A | -0.291 | +0.770 | +0.090 |
| morality | A | -0.285 | +0.786 | -0.409 |
| memory | A | +0.027 | +0.918 | -0.053 |
| emotion_recognition | A | -0.080 | +0.891 | -0.343 |
| planning | A | +0.383 | +0.811 | -0.056 |
| communication | A | +0.326 | +0.764 | -0.085 |
| thought | A | +0.448 | +0.771 | -0.185 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.176 | 0.349 |
| ChatGPT | ai | 0.752 | 0.261 |
| GPT-4 | ai | 1.000 | 1.000 |
| Siri | ai | 0.094 | 0.219 |
| Alexa | ai | 0.092 | 0.439 |
| Cortana | ai | 0.471 | 0.602 |
| Google Assistant | ai | 0.697 | 0.233 |
| Bixby | ai | 0.478 | 0.000 |
| Replika | ai | 0.620 | 0.172 |
| Cleverbot | ai | 0.086 | 0.198 |
| Watson | ai | 0.000 | 0.901 |
| Copilot | ai | 0.531 | 0.484 |
| Bard | ai | 0.271 | 0.219 |
| ELIZA | ai | 0.215 | 0.334 |
| Bing Chat | ai | 0.227 | 0.387 |
| Sam | human | 0.122 | 0.384 |
| Casey | human | 0.118 | 0.336 |
| Rebecca | human | 0.138 | 0.554 |
| Gregory | human | 0.125 | 0.514 |
| James | human | 0.117 | 0.427 |
| Maria | human | 0.125 | 0.497 |
| David | human | 0.169 | 0.419 |
| Aisha | human | 0.291 | 0.287 |
| Michael | human | 0.134 | 0.511 |
| Emily | human | 0.146 | 0.472 |
| Carlos | human | 0.128 | 0.382 |
| Priya | human | 0.146 | 0.477 |
| Omar | human | 0.133 | 0.458 |
| Mei | human | 0.050 | 0.625 |
| Sofia | human | 0.201 | 0.457 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.381 (SD=0.285)
- Human mean: 0.143 (SD=0.050)
- Separation: 0.238
- Mann-Whitney U=163.0, p=0.0381

### Factor 2

- AI mean: 0.386 (SD=0.262)
- Human mean: 0.453 (SD=0.083)
- Separation: 0.067
- Mann-Whitney U=67.0, p=0.0620

### Factor 3

- AI mean: 0.447 (SD=0.271)
- Human mean: 0.176 (SD=0.079)
- Separation: 0.271
- Mann-Whitney U=184.0, p=0.0032

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.176 | 0.381 | 0.143 |
| siri | ai | 1 | 0.094 | 0.381 | 0.143 |
| alexa | ai | 1 | 0.092 | 0.381 | 0.143 |
| cleverbot | ai | 1 | 0.086 | 0.381 | 0.143 |
| watson | ai | 1 | 0.000 | 0.381 | 0.143 |
| eliza | ai | 1 | 0.215 | 0.381 | 0.143 |
| bing_chat | ai | 1 | 0.227 | 0.381 | 0.143 |
| aisha | human | 1 | 0.291 | 0.143 | 0.381 |
| gpt4 | ai | 2 | 1.000 | 0.386 | 0.453 |
| alexa | ai | 2 | 0.439 | 0.386 | 0.453 |
| cortana | ai | 2 | 0.602 | 0.386 | 0.453 |
| watson | ai | 2 | 0.901 | 0.386 | 0.453 |
| copilot | ai | 2 | 0.484 | 0.386 | 0.453 |
| sam | human | 2 | 0.384 | 0.453 | 0.386 |
| casey | human | 2 | 0.336 | 0.453 | 0.386 |
| david | human | 2 | 0.419 | 0.453 | 0.386 |
| aisha | human | 2 | 0.287 | 0.453 | 0.386 |
| carlos | human | 2 | 0.382 | 0.453 | 0.386 |

