# Concept Geometry, Phase A: Behavioral PCA
## LLaMA-3-8B (Base)

**Run:** 2026-03-28 09:58:06

---

## What is being tested

Does LLaMA-3-8B (Base)'s explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

## Procedure

- 30 characters: 15 AI, 15 human
- 22 concept dimensions
- 435 pairwise comparisons per dimension
- Total comparisons: 19140
- Method: logit extraction over tokens 1-5 (base model)

## Response statistics

- All 19140 comparisons yield ratings (logit-based)

### Order consistency

- Pairs with both orders: 9570
- Perfectly consistent: 18 (0.2%)
- Mean deviation: 1.444

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 20.50 * | 93.2% | 93.2% |
| PC2 | 0.47 | 2.1% | 95.3% |
| PC3 | 0.26 | 1.2% | 96.5% |
| PC4 | 0.20 | 0.9% | 97.4% |
| PC5 | 0.14 | 0.6% | 98.0% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 |
|---------|----:|----:|
| phenomenology | +0.723 | +0.665 |
| emotions | +0.633 | +0.738 |
| agency | +0.744 | +0.601 |
| intentions | +0.751 | +0.641 |
| prediction | +0.584 | +0.794 |
| cognitive | +0.569 | +0.773 |
| social | +0.544 | +0.816 |
| embodiment | +0.860 | +0.486 |
| roles | +0.832 | +0.470 |
| animacy | +0.760 | +0.613 |
| formality | +0.678 | +0.689 |
| expertise | +0.773 | +0.605 |
| helpfulness | +0.773 | +0.599 |
| biological | +0.706 | +0.688 |
| shapes | +0.436 | +0.869 |
| human | +0.738 | +0.621 |
| ai | +0.581 | +0.769 |
| attention | +0.618 | +0.758 |
| mind | +0.658 | +0.717 |
| beliefs | +0.708 | +0.687 |
| desires | +0.859 | +0.498 |
| goals | +0.735 | +0.661 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.573 | 0.350 |
| ChatGPT | ai | 0.282 | 0.512 |
| GPT-4 | ai | 0.351 | 0.760 |
| Siri | ai | 0.541 | 0.590 |
| Alexa | ai | 0.741 | 0.673 |
| Cortana | ai | 0.369 | 0.860 |
| Google Assistant | ai | 0.442 | 0.050 |
| Bixby | ai | 0.385 | 0.681 |
| Replika | ai | 0.609 | 0.625 |
| Cleverbot | ai | 0.167 | 0.477 |
| Watson | ai | 0.505 | 0.532 |
| Copilot | ai | 0.289 | 0.302 |
| Bard | ai | 0.638 | 0.427 |
| ELIZA | ai | 0.000 | 0.202 |
| Bing Chat | ai | 0.317 | 0.000 |
| Sam | human | 0.961 | 0.473 |
| Casey | human | 0.768 | 0.524 |
| Rebecca | human | 0.916 | 0.077 |
| Gregory | human | 0.813 | 0.368 |
| James | human | 1.000 | 0.527 |
| Maria | human | 0.911 | 0.417 |
| David | human | 0.993 | 0.524 |
| Aisha | human | 0.285 | 0.922 |
| Michael | human | 0.963 | 0.611 |
| Emily | human | 0.831 | 0.331 |
| Carlos | human | 0.758 | 0.291 |
| Priya | human | 0.725 | 1.000 |
| Omar | human | 0.807 | 0.424 |
| Mei | human | 0.785 | 0.565 |
| Sofia | human | 0.694 | 0.294 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.414 (SD=0.187)
- Human mean: 0.814 (SD=0.171)
- Separation: 0.400
- Mann-Whitney U=14.0, p=0.0000

### Factor 2

- AI mean: 0.470 (SD=0.242)
- Human mean: 0.490 (SD=0.226)
- Separation: 0.020
- Mann-Whitney U=119.0, p=0.8035

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| alexa | ai | 1 | 0.741 | 0.414 | 0.814 |
| bard | ai | 1 | 0.638 | 0.414 | 0.814 |
| aisha | human | 1 | 0.285 | 0.814 | 0.414 |
| chatgpt | ai | 2 | 0.512 | 0.470 | 0.490 |
| gpt4 | ai | 2 | 0.760 | 0.470 | 0.490 |
| siri | ai | 2 | 0.590 | 0.470 | 0.490 |
| alexa | ai | 2 | 0.673 | 0.470 | 0.490 |
| cortana | ai | 2 | 0.860 | 0.470 | 0.490 |
| bixby | ai | 2 | 0.681 | 0.470 | 0.490 |
| replika | ai | 2 | 0.625 | 0.470 | 0.490 |
| watson | ai | 2 | 0.532 | 0.470 | 0.490 |
| sam | human | 2 | 0.473 | 0.490 | 0.470 |
| rebecca | human | 2 | 0.077 | 0.490 | 0.470 |
| gregory | human | 2 | 0.368 | 0.490 | 0.470 |
| maria | human | 2 | 0.417 | 0.490 | 0.470 |
| emily | human | 2 | 0.331 | 0.490 | 0.470 |
| carlos | human | 2 | 0.291 | 0.490 | 0.470 |
| omar | human | 2 | 0.424 | 0.490 | 0.470 |
| sofia | human | 2 | 0.294 | 0.490 | 0.470 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.033 | +0.033 | +0.066 |
| emotions | -0.031 | +0.031 | +0.062 |
| agency | -0.022 | +0.022 | +0.044 |
| intentions | -0.033 | +0.033 | +0.066 |
| prediction | -0.018 | +0.018 | +0.036 |
| cognitive | -0.021 | +0.021 | +0.041 |
| social | -0.018 | +0.018 | +0.036 |
| embodiment | -0.029 | +0.029 | +0.058 |
| roles | -0.023 | +0.023 | +0.047 |
| animacy | -0.030 | +0.030 | +0.060 |
| formality | -0.025 | +0.025 | +0.049 |
| expertise | -0.035 | +0.035 | +0.069 |
| helpfulness | -0.031 | +0.031 | +0.062 |
| biological | -0.025 | +0.025 | +0.050 |
| shapes | -0.016 | +0.016 | +0.032 |
| human | -0.031 | +0.031 | +0.062 |
| ai | -0.015 | +0.015 | +0.030 |
| attention | -0.028 | +0.028 | +0.056 |
| mind | -0.019 | +0.019 | +0.037 |
| beliefs | -0.028 | +0.028 | +0.055 |
| desires | -0.034 | +0.034 | +0.067 |
| goals | -0.029 | +0.029 | +0.058 |

