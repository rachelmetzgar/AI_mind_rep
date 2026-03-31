# Gray Replication with AI/Human Characters — Names Only
## Gemma-2-9B (Base)

**Run:** 2026-03-29 11:46:31

---

## What is being tested

Does Gemma-2-9B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

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
- Perfectly consistent: 472 (6.0%)
- Mean deviation: 1.332

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.79 * | 76.6% | 76.6% |
| PC2 | 1.95 * | 10.8% | 87.4% |
| PC3 | 0.73 | 4.1% | 91.5% |
| PC4 | 0.40 | 2.2% | 93.7% |
| PC5 | 0.26 | 1.4% | 95.1% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 |
|----------|--------|----:|----:|
| hunger | E | +0.274 | -0.892 |
| fear | E | +0.552 | -0.768 |
| pain | E | +0.929 | -0.284 |
| pleasure | E | +0.920 | -0.325 |
| rage | E | +0.384 | -0.840 |
| desire | E | +0.594 | -0.675 |
| personality | E | +0.859 | -0.276 |
| consciousness | E | +0.887 | -0.330 |
| pride | E | +0.387 | -0.819 |
| embarrassment | E | +0.205 | -0.950 |
| joy | E | +0.270 | -0.897 |
| self_control | A | +0.806 | -0.397 |
| morality | A | +0.695 | -0.631 |
| memory | A | +0.625 | -0.732 |
| emotion_recognition | A | +0.646 | -0.719 |
| planning | A | +0.774 | -0.447 |
| communication | A | +0.933 | -0.293 |
| thought | A | +0.687 | -0.533 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.354 | 0.830 |
| ChatGPT | ai | 0.000 | 0.714 |
| GPT-4 | ai | 1.000 | 0.759 |
| Siri | ai | 0.527 | 0.845 |
| Alexa | ai | 0.328 | 0.777 |
| Cortana | ai | 0.348 | 0.476 |
| Google Assistant | ai | 0.007 | 0.784 |
| Bixby | ai | 0.325 | 0.723 |
| Replika | ai | 0.636 | 0.475 |
| Cleverbot | ai | 0.643 | 1.000 |
| Watson | ai | 0.551 | 0.664 |
| Copilot | ai | 0.406 | 0.646 |
| Bard | ai | 0.277 | 0.883 |
| ELIZA | ai | 0.430 | 0.000 |
| Bing Chat | ai | 0.008 | 0.604 |
| Sam | human | 0.394 | 0.522 |
| Casey | human | 0.339 | 0.608 |
| Rebecca | human | 0.291 | 0.734 |
| Gregory | human | 0.372 | 0.769 |
| James | human | 0.311 | 0.627 |
| Maria | human | 0.309 | 0.686 |
| David | human | 0.336 | 0.663 |
| Aisha | human | 0.299 | 0.932 |
| Michael | human | 0.321 | 0.579 |
| Emily | human | 0.361 | 0.842 |
| Carlos | human | 0.351 | 0.652 |
| Priya | human | 0.305 | 0.736 |
| Omar | human | 0.300 | 0.515 |
| Mei | human | 0.252 | 0.661 |
| Sofia | human | 0.227 | 0.775 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.389 (SD=0.260)
- Human mean: 0.318 (SD=0.042)
- Separation: 0.071
- Mann-Whitney U=148.0, p=0.1466

### Factor 2

- AI mean: 0.679 (SD=0.228)
- Human mean: 0.687 (SD=0.111)
- Separation: 0.008
- Mann-Whitney U=127.0, p=0.5614

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| chatgpt | ai | 1 | 0.000 | 0.389 | 0.318 |
| alexa | ai | 1 | 0.328 | 0.389 | 0.318 |
| cortana | ai | 1 | 0.348 | 0.389 | 0.318 |
| google_assistant | ai | 1 | 0.007 | 0.389 | 0.318 |
| bixby | ai | 1 | 0.325 | 0.389 | 0.318 |
| bard | ai | 1 | 0.277 | 0.389 | 0.318 |
| bing_chat | ai | 1 | 0.008 | 0.389 | 0.318 |
| sam | human | 1 | 0.394 | 0.318 | 0.389 |
| gregory | human | 1 | 0.372 | 0.318 | 0.389 |
| emily | human | 1 | 0.361 | 0.318 | 0.389 |
| claude | ai | 2 | 0.830 | 0.679 | 0.687 |
| chatgpt | ai | 2 | 0.714 | 0.679 | 0.687 |
| gpt4 | ai | 2 | 0.759 | 0.679 | 0.687 |
| siri | ai | 2 | 0.845 | 0.679 | 0.687 |
| alexa | ai | 2 | 0.777 | 0.679 | 0.687 |
| google_assistant | ai | 2 | 0.784 | 0.679 | 0.687 |
| bixby | ai | 2 | 0.723 | 0.679 | 0.687 |
| cleverbot | ai | 2 | 1.000 | 0.679 | 0.687 |
| bard | ai | 2 | 0.883 | 0.679 | 0.687 |
| sam | human | 2 | 0.522 | 0.687 | 0.679 |
| casey | human | 2 | 0.608 | 0.687 | 0.679 |
| james | human | 2 | 0.627 | 0.687 | 0.679 |
| david | human | 2 | 0.663 | 0.687 | 0.679 |
| michael | human | 2 | 0.579 | 0.687 | 0.679 |
| carlos | human | 2 | 0.652 | 0.687 | 0.679 |
| omar | human | 2 | 0.515 | 0.687 | 0.679 |
| mei | human | 2 | 0.661 | 0.687 | 0.679 |

