# Gray Replication with AI/Human Characters — Names Only
## Qwen3-8B

**Run:** 2026-03-28 16:13:53

---

## What is being tested

Does Qwen3-8B's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

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
- Perfectly consistent: 4219 (53.9%)
- Mean deviation: 0.734

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 11.68 * | 64.9% | 64.9% |
| PC2 | 3.79 * | 21.1% | 85.9% |
| PC3 | 0.70 | 3.9% | 89.8% |
| PC4 | 0.60 | 3.3% | 93.2% |
| PC5 | 0.35 | 2.0% | 95.1% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.264 | -0.919 |
| fear | E | -0.314 | -0.843 |
| pain | E | -0.263 | -0.825 |
| pleasure | E | -0.319 | -0.900 |
| rage | E | +0.144 | -0.769 |
| desire | E | -0.758 | -0.560 |
| personality | E | -0.743 | -0.549 |
| consciousness | E | -0.878 | -0.385 |
| pride | E | -0.665 | -0.654 |
| embarrassment | E | -0.347 | -0.819 |
| joy | E | -0.645 | -0.716 |
| self_control | A | -0.801 | -0.359 |
| morality | A | -0.847 | -0.402 |
| memory | A | -0.916 | +0.202 |
| emotion_recognition | A | -0.786 | -0.541 |
| planning | A | -0.959 | -0.079 |
| communication | A | -0.950 | -0.144 |
| thought | A | -0.967 | +0.051 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.697 | 0.349 |
| ChatGPT | ai | 0.275 | 0.756 |
| GPT-4 | ai | 0.000 | 0.723 |
| Siri | ai | 0.750 | 1.000 |
| Alexa | ai | 0.706 | 0.624 |
| Cortana | ai | 0.595 | 0.000 |
| Google Assistant | ai | 0.696 | 0.797 |
| Bixby | ai | 0.759 | 0.456 |
| Replika | ai | 0.529 | 0.273 |
| Cleverbot | ai | 0.871 | 0.736 |
| Watson | ai | 0.737 | 0.274 |
| Copilot | ai | 0.750 | 0.844 |
| Bard | ai | 0.719 | 0.476 |
| ELIZA | ai | 1.000 | 0.917 |
| Bing Chat | ai | 0.953 | 0.993 |
| Sam | human | 0.793 | 0.389 |
| Casey | human | 0.800 | 0.436 |
| Rebecca | human | 0.763 | 0.267 |
| Gregory | human | 0.750 | 0.404 |
| James | human | 0.801 | 0.399 |
| Maria | human | 0.749 | 0.310 |
| David | human | 0.772 | 0.427 |
| Aisha | human | 0.749 | 0.367 |
| Michael | human | 0.801 | 0.379 |
| Emily | human | 0.725 | 0.274 |
| Carlos | human | 0.811 | 0.365 |
| Priya | human | 0.773 | 0.287 |
| Omar | human | 0.892 | 0.274 |
| Mei | human | 0.699 | 0.101 |
| Sofia | human | 0.733 | 0.365 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.669 (SD=0.244)
- Human mean: 0.774 (SD=0.044)
- Separation: 0.105
- Mann-Whitney U=67.0, p=0.0620

### Factor 2

- AI mean: 0.615 (SD=0.288)
- Human mean: 0.336 (SD=0.084)
- Separation: 0.278
- Mann-Whitney U=177.0, p=0.0079

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| siri | ai | 1 | 0.750 | 0.669 | 0.774 |
| bixby | ai | 1 | 0.759 | 0.669 | 0.774 |
| cleverbot | ai | 1 | 0.871 | 0.669 | 0.774 |
| watson | ai | 1 | 0.737 | 0.669 | 0.774 |
| copilot | ai | 1 | 0.750 | 0.669 | 0.774 |
| eliza | ai | 1 | 1.000 | 0.669 | 0.774 |
| bing_chat | ai | 1 | 0.953 | 0.669 | 0.774 |
| mei | human | 1 | 0.699 | 0.774 | 0.669 |
| claude | ai | 2 | 0.349 | 0.615 | 0.336 |
| cortana | ai | 2 | 0.000 | 0.615 | 0.336 |
| bixby | ai | 2 | 0.456 | 0.615 | 0.336 |
| replika | ai | 2 | 0.273 | 0.615 | 0.336 |
| watson | ai | 2 | 0.274 | 0.615 | 0.336 |

