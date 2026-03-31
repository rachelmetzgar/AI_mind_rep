# Gray Replication with AI/Human Characters
## LLaMA-3-8B-Instruct

**Run:** 2026-03-28 12:56:07

---

## What is being tested

Does LLaMA-3-8B-Instruct's folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

## Procedure

- 30 characters: 15 AI, 15 human
- 18 mental capacities (Gray et al. 2007)
- 435 pairwise comparisons per capacity
- Total comparisons: 15660
- Method: text generation + parse rating (chat)

## Response statistics

- Successfully parsed: 15649 / 15660 (99.9%)

### Order consistency

- Pairs with both orders: 7819
- Perfectly consistent: 2448 (31.3%)
- Mean deviation: 1.643

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 8.85 * | 49.2% | 49.2% |
| PC2 | 4.26 * | 23.7% | 72.8% |
| PC3 | 1.83 * | 10.2% | 83.0% |
| PC4 | 0.83 | 4.6% | 87.6% |
| PC5 | 0.46 | 2.6% | 90.2% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 |
|----------|--------|----:|----:|----:|
| hunger | E | -0.928 | -0.158 | +0.003 |
| fear | E | -0.960 | +0.029 | -0.087 |
| pain | E | -0.940 | +0.068 | -0.002 |
| pleasure | E | -0.965 | -0.112 | +0.075 |
| rage | E | -0.676 | +0.128 | -0.393 |
| desire | E | -0.804 | +0.215 | +0.428 |
| personality | E | -0.163 | +0.115 | +0.796 |
| consciousness | E | -0.299 | +0.595 | +0.611 |
| pride | E | -0.950 | -0.053 | +0.001 |
| embarrassment | E | -0.940 | -0.096 | +0.046 |
| joy | E | -0.902 | -0.180 | +0.197 |
| self_control | A | -0.526 | +0.571 | -0.188 |
| morality | A | -0.919 | +0.247 | -0.018 |
| memory | A | +0.283 | +0.835 | +0.088 |
| emotion_recognition | A | -0.173 | +0.101 | +0.810 |
| planning | A | -0.214 | +0.835 | +0.220 |
| communication | A | +0.241 | +0.351 | +0.840 |
| thought | A | +0.250 | +0.872 | +0.291 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.831 | 0.441 |
| ChatGPT | ai | 0.780 | 0.714 |
| GPT-4 | ai | 0.562 | 0.992 |
| Siri | ai | 0.736 | 0.217 |
| Alexa | ai | 0.705 | 0.498 |
| Cortana | ai | 0.452 | 1.000 |
| Google Assistant | ai | 0.739 | 0.922 |
| Bixby | ai | 0.789 | 0.426 |
| Replika | ai | 0.838 | 0.000 |
| Cleverbot | ai | 0.804 | 0.515 |
| Watson | ai | 0.653 | 0.930 |
| Copilot | ai | 0.687 | 0.782 |
| Bard | ai | 0.747 | 0.716 |
| ELIZA | ai | 0.824 | 0.150 |
| Bing Chat | ai | 1.000 | 0.274 |
| Sam | human | 0.000 | 0.586 |
| Casey | human | 0.149 | 0.397 |
| Rebecca | human | 0.364 | 0.570 |
| Gregory | human | 0.245 | 0.364 |
| James | human | 0.270 | 0.569 |
| Maria | human | 0.245 | 0.492 |
| David | human | 0.336 | 0.911 |
| Aisha | human | 0.100 | 0.426 |
| Michael | human | 0.246 | 0.222 |
| Emily | human | 0.325 | 0.240 |
| Carlos | human | 0.298 | 0.349 |
| Priya | human | 0.090 | 0.736 |
| Omar | human | 0.024 | 0.435 |
| Mei | human | 0.254 | 0.084 |
| Sofia | human | 0.126 | 0.502 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.743 (SD=0.123)
- Human mean: 0.205 (SD=0.111)
- Separation: 0.539
- Mann-Whitney U=225.0, p=0.0000

### Factor 2

- AI mean: 0.572 (SD=0.312)
- Human mean: 0.459 (SD=0.199)
- Separation: 0.113
- Mann-Whitney U=138.0, p=0.2998

### Factor 3

- AI mean: 0.487 (SD=0.290)
- Human mean: 0.437 (SD=0.282)
- Separation: 0.051
- Mann-Whitney U=122.0, p=0.7089

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| cortana | ai | 1 | 0.452 | 0.743 | 0.205 |
| claude | ai | 2 | 0.441 | 0.572 | 0.459 |
| siri | ai | 2 | 0.217 | 0.572 | 0.459 |
| alexa | ai | 2 | 0.498 | 0.572 | 0.459 |
| bixby | ai | 2 | 0.426 | 0.572 | 0.459 |
| replika | ai | 2 | 0.000 | 0.572 | 0.459 |
| cleverbot | ai | 2 | 0.515 | 0.572 | 0.459 |
| eliza | ai | 2 | 0.150 | 0.572 | 0.459 |
| bing_chat | ai | 2 | 0.274 | 0.572 | 0.459 |
| sam | human | 2 | 0.586 | 0.459 | 0.572 |
| rebecca | human | 2 | 0.570 | 0.459 | 0.572 |
| james | human | 2 | 0.569 | 0.459 | 0.572 |
| david | human | 2 | 0.911 | 0.459 | 0.572 |
| priya | human | 2 | 0.736 | 0.459 | 0.572 |

