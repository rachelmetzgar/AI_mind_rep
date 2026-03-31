# Gray Replication with AI/Human Characters — Names Only
## Gemma-2-2B (Base)

**Run:** 2026-03-29 11:42:24

---

## What is being tested

Does Gemma-2-2B (Base)'s folk psychology produce an Experience/Agency factor structure (Gray et al. 2007) when rating 30 AI/human characters on 18 mental capacities, using **names only** (no descriptions)? This tests whether prior knowledge of names alone is sufficient to drive differential mental capacity attributions.

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
- Perfectly consistent: 4775 (61.0%)
- Mean deviation: 0.177

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 13.24 * | 73.5% | 73.5% |
| PC2 | 1.43 * | 7.9% | 81.5% |
| PC3 | 1.09 * | 6.0% | 87.5% |
| PC4 | 0.60 | 3.3% | 90.9% |
| PC5 | 0.51 | 2.8% | 93.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated capacity loadings

| Capacity | Factor | F1 | F2 | F3 |
|----------|--------|----:|----:|----:|
| hunger | E | +0.794 | -0.194 | +0.436 |
| fear | E | +0.745 | +0.343 | +0.479 |
| pain | E | +0.690 | +0.384 | +0.530 |
| pleasure | E | +0.828 | +0.406 | +0.304 |
| rage | E | +0.305 | +0.694 | +0.583 |
| desire | E | +0.373 | +0.278 | +0.809 |
| personality | E | +0.464 | -0.027 | +0.816 |
| consciousness | E | +0.397 | +0.415 | +0.728 |
| pride | E | +0.194 | +0.293 | +0.910 |
| embarrassment | E | +0.657 | +0.297 | +0.637 |
| joy | E | +0.634 | +0.332 | +0.512 |
| self_control | A | +0.162 | +0.876 | +0.182 |
| morality | A | +0.689 | +0.404 | +0.265 |
| memory | A | +0.697 | +0.194 | +0.632 |
| emotion_recognition | A | +0.949 | +0.064 | +0.179 |
| planning | A | +0.755 | +0.441 | +0.332 |
| communication | A | +0.820 | +0.306 | +0.375 |
| thought | A | +0.734 | +0.278 | +0.421 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 1.000 | 0.642 |
| ChatGPT | ai | 0.165 | 0.853 |
| GPT-4 | ai | 0.000 | 0.542 |
| Siri | ai | 0.611 | 0.550 |
| Alexa | ai | 0.214 | 0.490 |
| Cortana | ai | 0.486 | 0.614 |
| Google Assistant | ai | 0.334 | 0.604 |
| Bixby | ai | 0.701 | 0.653 |
| Replika | ai | 0.505 | 0.247 |
| Cleverbot | ai | 0.392 | 0.716 |
| Watson | ai | 0.546 | 0.587 |
| Copilot | ai | 0.605 | 0.798 |
| Bard | ai | 0.468 | 1.000 |
| ELIZA | ai | 0.998 | 0.524 |
| Bing Chat | ai | 0.540 | 0.392 |
| Sam | human | 0.529 | 0.606 |
| Casey | human | 0.431 | 0.374 |
| Rebecca | human | 0.403 | 0.553 |
| Gregory | human | 0.406 | 0.672 |
| James | human | 0.233 | 0.795 |
| Maria | human | 0.379 | 0.728 |
| David | human | 0.384 | 0.773 |
| Aisha | human | 0.207 | 0.205 |
| Michael | human | 0.394 | 0.900 |
| Emily | human | 0.465 | 0.770 |
| Carlos | human | 0.225 | 0.618 |
| Priya | human | 0.352 | 0.454 |
| Omar | human | 0.439 | 0.000 |
| Mei | human | 0.223 | 0.564 |
| Sofia | human | 0.363 | 0.644 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.504 (SD=0.265)
- Human mean: 0.362 (SD=0.094)
- Separation: 0.142
- Mann-Whitney U=160.0, p=0.0512

### Factor 2

- AI mean: 0.614 (SD=0.177)
- Human mean: 0.577 (SD=0.230)
- Separation: 0.037
- Mann-Whitney U=110.0, p=0.9339

### Factor 3

- AI mean: 0.315 (SD=0.272)
- Human mean: 0.202 (SD=0.094)
- Separation: 0.113
- Mann-Whitney U=126.0, p=0.5897

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| chatgpt | ai | 1 | 0.165 | 0.504 | 0.362 |
| gpt4 | ai | 1 | 0.000 | 0.504 | 0.362 |
| alexa | ai | 1 | 0.214 | 0.504 | 0.362 |
| google_assistant | ai | 1 | 0.334 | 0.504 | 0.362 |
| cleverbot | ai | 1 | 0.392 | 0.504 | 0.362 |
| sam | human | 1 | 0.529 | 0.362 | 0.504 |
| emily | human | 1 | 0.465 | 0.362 | 0.504 |
| omar | human | 1 | 0.439 | 0.362 | 0.504 |
| gpt4 | ai | 2 | 0.542 | 0.614 | 0.577 |
| siri | ai | 2 | 0.550 | 0.614 | 0.577 |
| alexa | ai | 2 | 0.490 | 0.614 | 0.577 |
| replika | ai | 2 | 0.247 | 0.614 | 0.577 |
| watson | ai | 2 | 0.587 | 0.614 | 0.577 |
| eliza | ai | 2 | 0.524 | 0.614 | 0.577 |
| bing_chat | ai | 2 | 0.392 | 0.614 | 0.577 |
| sam | human | 2 | 0.606 | 0.577 | 0.614 |
| gregory | human | 2 | 0.672 | 0.577 | 0.614 |
| james | human | 2 | 0.795 | 0.577 | 0.614 |
| maria | human | 2 | 0.728 | 0.577 | 0.614 |
| david | human | 2 | 0.773 | 0.577 | 0.614 |
| michael | human | 2 | 0.900 | 0.577 | 0.614 |
| emily | human | 2 | 0.770 | 0.577 | 0.614 |
| carlos | human | 2 | 0.618 | 0.577 | 0.614 |
| sofia | human | 2 | 0.644 | 0.577 | 0.614 |

