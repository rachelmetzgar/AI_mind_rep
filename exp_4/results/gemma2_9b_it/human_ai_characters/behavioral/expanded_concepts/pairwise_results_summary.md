# Concept Geometry, Phase A: Behavioral PCA
## Gemma-2-9B-IT

**Run:** 2026-03-28 16:56:37

---

## What is being tested

Does Gemma-2-9B-IT's explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

## Procedure

- 30 characters: 15 AI, 15 human
- 22 concept dimensions
- 435 pairwise comparisons per dimension
- Total comparisons: 19140
- Method: text generation + parse rating (chat model)

## Response statistics

- Successfully parsed: 18985 / 19140 (99.2%)

### Order consistency

- Pairs with both orders: 9454
- Perfectly consistent: 6985 (73.9%)
- Mean deviation: 0.631

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.98 * | 72.6% | 72.6% |
| PC2 | 3.66 * | 16.7% | 89.3% |
| PC3 | 0.82 | 3.7% | 93.0% |
| PC4 | 0.56 | 2.6% | 95.5% |
| PC5 | 0.31 | 1.4% | 97.0% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 |
|---------|----:|----:|
| phenomenology | +0.967 | -0.242 |
| emotions | +0.967 | -0.230 |
| agency | +0.939 | -0.302 |
| intentions | +0.914 | -0.392 |
| prediction | +0.657 | -0.720 |
| cognitive | +0.344 | -0.877 |
| social | +0.939 | -0.298 |
| embodiment | +0.962 | -0.254 |
| roles | +0.499 | -0.400 |
| animacy | +0.970 | -0.220 |
| formality | -0.435 | -0.705 |
| expertise | +0.363 | -0.841 |
| helpfulness | +0.480 | -0.654 |
| biological | +0.973 | -0.212 |
| shapes | -0.756 | -0.295 |
| human | +0.960 | -0.258 |
| ai | -0.942 | -0.199 |
| attention | +0.097 | -0.927 |
| mind | +0.921 | -0.372 |
| beliefs | +0.914 | -0.370 |
| desires | +0.961 | -0.261 |
| goals | +0.522 | -0.814 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.114 | 0.198 |
| ChatGPT | ai | 0.119 | 0.358 |
| GPT-4 | ai | 0.004 | 0.000 |
| Siri | ai | 0.308 | 0.822 |
| Alexa | ai | 0.298 | 0.751 |
| Cortana | ai | 0.089 | 0.347 |
| Google Assistant | ai | 0.205 | 0.496 |
| Bixby | ai | 0.253 | 0.802 |
| Replika | ai | 0.285 | 0.661 |
| Cleverbot | ai | 0.306 | 0.688 |
| Watson | ai | 0.000 | 0.189 |
| Copilot | ai | 0.102 | 0.356 |
| Bard | ai | 0.155 | 0.399 |
| ELIZA | ai | 0.304 | 1.000 |
| Bing Chat | ai | 0.008 | 0.204 |
| Sam | human | 0.975 | 0.416 |
| Casey | human | 0.994 | 0.438 |
| Rebecca | human | 0.895 | 0.273 |
| Gregory | human | 0.935 | 0.408 |
| James | human | 0.934 | 0.407 |
| Maria | human | 0.909 | 0.308 |
| David | human | 0.839 | 0.239 |
| Aisha | human | 0.932 | 0.428 |
| Michael | human | 0.954 | 0.472 |
| Emily | human | 0.945 | 0.450 |
| Carlos | human | 0.966 | 0.436 |
| Priya | human | 0.831 | 0.257 |
| Omar | human | 0.948 | 0.384 |
| Mei | human | 1.000 | 0.638 |
| Sofia | human | 0.851 | 0.390 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.170 (SD=0.113)
- Human mean: 0.927 (SD=0.051)
- Separation: 0.757
- Mann-Whitney U=0.0, p=0.0000

### Factor 2

- AI mean: 0.485 (SD=0.279)
- Human mean: 0.396 (SD=0.096)
- Separation: 0.089
- Mann-Whitney U=122.0, p=0.7089

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 2 | 0.198 | 0.485 | 0.396 |
| chatgpt | ai | 2 | 0.358 | 0.485 | 0.396 |
| gpt4 | ai | 2 | 0.000 | 0.485 | 0.396 |
| cortana | ai | 2 | 0.347 | 0.485 | 0.396 |
| watson | ai | 2 | 0.189 | 0.485 | 0.396 |
| copilot | ai | 2 | 0.356 | 0.485 | 0.396 |
| bard | ai | 2 | 0.399 | 0.485 | 0.396 |
| bing_chat | ai | 2 | 0.204 | 0.485 | 0.396 |
| michael | human | 2 | 0.472 | 0.396 | 0.485 |
| emily | human | 2 | 0.450 | 0.396 | 0.485 |
| mei | human | 2 | 0.638 | 0.396 | 0.485 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -1.016 | +1.016 | +2.032 |
| emotions | -1.034 | +1.034 | +2.069 |
| agency | -0.931 | +0.931 | +1.862 |
| intentions | -0.964 | +0.964 | +1.929 |
| prediction | -0.454 | +0.454 | +0.908 |
| cognitive | -0.378 | +0.378 | +0.756 |
| social | -0.862 | +0.862 | +1.724 |
| embodiment | -1.034 | +1.034 | +2.069 |
| roles | -0.368 | +0.440 | +0.808 |
| animacy | -1.034 | +1.034 | +2.069 |
| formality | +0.186 | -0.186 | -0.372 |
| expertise | -0.522 | +0.522 | +1.044 |
| helpfulness | -0.279 | +0.279 | +0.559 |
| biological | -1.034 | +1.034 | +2.069 |
| shapes | +0.230 | -0.230 | -0.460 |
| human | -1.034 | +1.034 | +2.069 |
| ai | +0.906 | -0.906 | -1.811 |
| attention | -0.132 | +0.132 | +0.264 |
| mind | -0.892 | +0.892 | +1.784 |
| beliefs | -0.943 | +0.943 | +1.885 |
| desires | -1.017 | +1.017 | +2.034 |
| goals | -0.478 | +0.478 | +0.956 |

