# Concept Geometry, Phase A: Behavioral PCA
## Gemma-2-9B (Base)

**Run:** 2026-03-29 11:14:48

---

## What is being tested

Does Gemma-2-9B (Base)'s explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

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
- Perfectly consistent: 2513 (26.3%)
- Mean deviation: 1.037

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 17.73 * | 80.6% | 80.6% |
| PC2 | 1.13 * | 5.2% | 85.7% |
| PC3 | 0.79 | 3.6% | 89.3% |
| PC4 | 0.51 | 2.3% | 91.6% |
| PC5 | 0.41 | 1.9% | 93.5% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 |
|---------|----:|----:|
| phenomenology | +0.622 | -0.683 |
| emotions | +0.572 | -0.774 |
| agency | +0.744 | -0.587 |
| intentions | +0.714 | -0.635 |
| prediction | +0.802 | -0.535 |
| cognitive | +0.811 | -0.502 |
| social | +0.769 | -0.582 |
| embodiment | +0.682 | -0.587 |
| roles | +0.514 | -0.547 |
| animacy | +0.438 | -0.842 |
| formality | +0.961 | -0.097 |
| expertise | +0.750 | -0.400 |
| helpfulness | +0.790 | -0.511 |
| biological | +0.499 | -0.738 |
| shapes | +0.738 | -0.524 |
| human | +0.121 | -0.920 |
| ai | +0.708 | -0.551 |
| attention | +0.733 | -0.567 |
| mind | +0.609 | -0.726 |
| beliefs | +0.549 | -0.743 |
| desires | +0.653 | -0.695 |
| goals | +0.817 | -0.472 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 1.000 | 0.587 |
| ChatGPT | ai | 0.455 | 0.690 |
| GPT-4 | ai | 0.949 | 1.000 |
| Siri | ai | 0.472 | 0.326 |
| Alexa | ai | 0.550 | 0.261 |
| Cortana | ai | 0.669 | 0.592 |
| Google Assistant | ai | 0.569 | 0.509 |
| Bixby | ai | 0.664 | 0.445 |
| Replika | ai | 0.511 | 0.000 |
| Cleverbot | ai | 0.187 | 0.464 |
| Watson | ai | 0.764 | 0.568 |
| Copilot | ai | 0.418 | 0.647 |
| Bard | ai | 0.614 | 0.370 |
| ELIZA | ai | 0.263 | 0.206 |
| Bing Chat | ai | 0.621 | 0.481 |
| Sam | human | 0.323 | 0.428 |
| Casey | human | 0.180 | 0.662 |
| Rebecca | human | 0.028 | 0.660 |
| Gregory | human | 0.071 | 0.729 |
| James | human | 0.294 | 0.769 |
| Maria | human | 0.000 | 0.474 |
| David | human | 0.197 | 0.519 |
| Aisha | human | 0.131 | 0.289 |
| Michael | human | 0.221 | 0.662 |
| Emily | human | 0.027 | 0.860 |
| Carlos | human | 0.254 | 0.674 |
| Priya | human | 0.238 | 0.413 |
| Omar | human | 0.236 | 0.637 |
| Mei | human | 0.318 | 0.445 |
| Sofia | human | 0.148 | 0.428 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.580 (SD=0.213)
- Human mean: 0.178 (SD=0.103)
- Separation: 0.403
- Mann-Whitney U=214.0, p=0.0000

### Factor 2

- AI mean: 0.476 (SD=0.226)
- Human mean: 0.577 (SD=0.155)
- Separation: 0.100
- Mann-Whitney U=80.0, p=0.1844

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| cleverbot | ai | 1 | 0.187 | 0.580 | 0.178 |
| eliza | ai | 1 | 0.263 | 0.580 | 0.178 |
| claude | ai | 2 | 0.587 | 0.476 | 0.577 |
| chatgpt | ai | 2 | 0.690 | 0.476 | 0.577 |
| gpt4 | ai | 2 | 1.000 | 0.476 | 0.577 |
| cortana | ai | 2 | 0.592 | 0.476 | 0.577 |
| watson | ai | 2 | 0.568 | 0.476 | 0.577 |
| copilot | ai | 2 | 0.647 | 0.476 | 0.577 |
| sam | human | 2 | 0.428 | 0.577 | 0.476 |
| maria | human | 2 | 0.474 | 0.577 | 0.476 |
| david | human | 2 | 0.519 | 0.577 | 0.476 |
| aisha | human | 2 | 0.289 | 0.577 | 0.476 |
| priya | human | 2 | 0.413 | 0.577 | 0.476 |
| mei | human | 2 | 0.445 | 0.577 | 0.476 |
| sofia | human | 2 | 0.428 | 0.577 | 0.476 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | +0.017 | -0.017 | -0.034 |
| emotions | +0.019 | -0.019 | -0.038 |
| agency | +0.023 | -0.023 | -0.045 |
| intentions | +0.022 | -0.022 | -0.045 |
| prediction | +0.024 | -0.024 | -0.047 |
| cognitive | +0.021 | -0.021 | -0.043 |
| social | +0.021 | -0.021 | -0.042 |
| embodiment | +0.030 | -0.030 | -0.061 |
| roles | +0.028 | -0.028 | -0.057 |
| animacy | +0.019 | -0.019 | -0.037 |
| formality | +0.020 | -0.020 | -0.039 |
| expertise | +0.010 | -0.010 | -0.019 |
| helpfulness | +0.022 | -0.022 | -0.044 |
| biological | +0.016 | -0.016 | -0.032 |
| shapes | +0.025 | -0.025 | -0.051 |
| human | +0.007 | -0.007 | -0.013 |
| ai | +0.041 | -0.041 | -0.082 |
| attention | +0.021 | -0.021 | -0.041 |
| mind | +0.025 | -0.025 | -0.049 |
| beliefs | +0.013 | -0.013 | -0.027 |
| desires | +0.019 | -0.019 | -0.038 |
| goals | +0.017 | -0.017 | -0.035 |

