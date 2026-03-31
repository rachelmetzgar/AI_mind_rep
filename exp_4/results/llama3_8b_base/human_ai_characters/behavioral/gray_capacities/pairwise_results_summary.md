# Gray Replication with AI/Human Characters
## LLaMA-3-8B (Base)

**Run:** 2026-03-28 09:23:13

---

## What is being tested

Does LLaMA-3-8B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities? Do the factors separate AI from human characters?

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
- Perfectly consistent: 240 (3.1%)
- Mean deviation: 1.193

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 16.50 * | 91.6% | 91.6% |
| PC2 | 0.72 | 4.0% | 95.6% |
| PC3 | 0.37 | 2.0% | 97.7% |
| PC4 | 0.12 | 0.6% | 98.3% |
| PC5 | 0.10 | 0.5% | 98.8% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.904 | +0.378 |
| fear | E | +0.741 | +0.636 |
| pain | E | +0.760 | +0.632 |
| pleasure | E | +0.717 | +0.689 |
| rage | E | +0.811 | +0.555 |
| desire | E | +0.664 | +0.728 |
| personality | E | +0.475 | +0.819 |
| consciousness | E | +0.510 | +0.846 |
| pride | E | +0.801 | +0.551 |
| embarrassment | E | +0.876 | +0.456 |
| joy | E | +0.874 | +0.464 |
| self_control | A | +0.364 | +0.853 |
| morality | A | +0.529 | +0.837 |
| memory | A | +0.744 | +0.622 |
| emotion_recognition | A | +0.729 | +0.642 |
| planning | A | +0.577 | +0.807 |
| communication | A | +0.657 | +0.713 |
| thought | A | +0.637 | +0.754 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.420 | 0.464 |
| ChatGPT | ai | 0.408 | 0.528 |
| GPT-4 | ai | 1.000 | 0.548 |
| Siri | ai | 0.722 | 0.504 |
| Alexa | ai | 0.846 | 0.564 |
| Cortana | ai | 0.622 | 0.558 |
| Google Assistant | ai | 0.134 | 0.383 |
| Bixby | ai | 0.352 | 0.381 |
| Replika | ai | 0.350 | 1.000 |
| Cleverbot | ai | 0.836 | 0.203 |
| Watson | ai | 0.463 | 0.669 |
| Copilot | ai | 0.000 | 0.493 |
| Bard | ai | 0.476 | 0.529 |
| ELIZA | ai | 0.169 | 0.288 |
| Bing Chat | ai | 0.432 | 0.000 |
| Sam | human | 0.492 | 0.756 |
| Casey | human | 0.427 | 0.757 |
| Rebecca | human | 0.172 | 0.830 |
| Gregory | human | 0.218 | 0.738 |
| James | human | 0.493 | 0.843 |
| Maria | human | 0.501 | 0.747 |
| David | human | 0.559 | 0.870 |
| Aisha | human | 0.825 | 0.606 |
| Michael | human | 0.562 | 0.887 |
| Emily | human | 0.305 | 0.817 |
| Carlos | human | 0.374 | 0.638 |
| Priya | human | 0.686 | 0.808 |
| Omar | human | 0.582 | 0.644 |
| Mei | human | 0.586 | 0.914 |
| Sofia | human | 0.358 | 0.671 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.482 (SD=0.271)
- Human mean: 0.476 (SD=0.167)
- Separation: 0.006
- Mann-Whitney U=106.0, p=0.8035

### Factor 2

- AI mean: 0.474 (SD=0.215)
- Human mean: 0.768 (SD=0.093)
- Separation: 0.294
- Mann-Whitney U=18.0, p=0.0001

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.420 | 0.482 | 0.476 |
| chatgpt | ai | 1 | 0.408 | 0.482 | 0.476 |
| google_assistant | ai | 1 | 0.134 | 0.482 | 0.476 |
| bixby | ai | 1 | 0.352 | 0.482 | 0.476 |
| replika | ai | 1 | 0.350 | 0.482 | 0.476 |
| watson | ai | 1 | 0.463 | 0.482 | 0.476 |
| copilot | ai | 1 | 0.000 | 0.482 | 0.476 |
| bard | ai | 1 | 0.476 | 0.482 | 0.476 |
| eliza | ai | 1 | 0.169 | 0.482 | 0.476 |
| bing_chat | ai | 1 | 0.432 | 0.482 | 0.476 |
| sam | human | 1 | 0.492 | 0.476 | 0.482 |
| james | human | 1 | 0.493 | 0.476 | 0.482 |
| maria | human | 1 | 0.501 | 0.476 | 0.482 |
| david | human | 1 | 0.559 | 0.476 | 0.482 |
| aisha | human | 1 | 0.825 | 0.476 | 0.482 |
| michael | human | 1 | 0.562 | 0.476 | 0.482 |
| priya | human | 1 | 0.686 | 0.476 | 0.482 |
| omar | human | 1 | 0.582 | 0.476 | 0.482 |
| mei | human | 1 | 0.586 | 0.476 | 0.482 |
| replika | ai | 2 | 1.000 | 0.474 | 0.768 |
| watson | ai | 2 | 0.669 | 0.474 | 0.768 |
| aisha | human | 2 | 0.606 | 0.768 | 0.474 |

