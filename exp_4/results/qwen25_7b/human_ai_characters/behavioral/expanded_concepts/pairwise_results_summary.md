# Concept Geometry, Phase A: Behavioral PCA
## Qwen-2.5-7B (Base)

**Run:** 2026-03-29 11:04:53

---

## What is being tested

Does Qwen-2.5-7B (Base)'s explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

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
- Perfectly consistent: 1648 (17.2%)
- Mean deviation: 1.080

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 16.54 * | 75.2% | 75.2% |
| PC2 | 3.00 * | 13.6% | 88.8% |
| PC3 | 0.94 | 4.3% | 93.1% |
| PC4 | 0.46 | 2.1% | 95.2% |
| PC5 | 0.33 | 1.5% | 96.7% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 |
|---------|----:|----:|
| phenomenology | +0.910 | -0.353 |
| emotions | +0.859 | -0.479 |
| agency | +0.706 | -0.633 |
| intentions | +0.868 | -0.463 |
| prediction | +0.572 | -0.790 |
| cognitive | +0.381 | -0.877 |
| social | +0.520 | -0.802 |
| embodiment | +0.902 | -0.268 |
| roles | -0.010 | -0.893 |
| animacy | +0.983 | -0.104 |
| formality | +0.725 | -0.593 |
| expertise | +0.701 | -0.623 |
| helpfulness | +0.894 | -0.290 |
| biological | +0.964 | -0.013 |
| shapes | +0.414 | -0.491 |
| human | +0.977 | +0.074 |
| ai | -0.100 | -0.917 |
| attention | +0.704 | -0.549 |
| mind | +0.871 | -0.397 |
| beliefs | +0.807 | -0.529 |
| desires | +0.934 | -0.274 |
| goals | +0.864 | -0.429 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.658 | 0.262 |
| ChatGPT | ai | 0.000 | 0.039 |
| GPT-4 | ai | 0.003 | 0.000 |
| Siri | ai | 0.095 | 0.896 |
| Alexa | ai | 0.339 | 0.738 |
| Cortana | ai | 0.272 | 0.754 |
| Google Assistant | ai | 0.045 | 0.836 |
| Bixby | ai | 0.159 | 0.886 |
| Replika | ai | 0.760 | 0.279 |
| Cleverbot | ai | 0.315 | 0.631 |
| Watson | ai | 0.170 | 0.617 |
| Copilot | ai | 0.497 | 0.414 |
| Bard | ai | 0.592 | 0.464 |
| ELIZA | ai | 0.831 | 0.130 |
| Bing Chat | ai | 0.348 | 0.532 |
| Sam | human | 0.860 | 0.618 |
| Casey | human | 0.860 | 0.582 |
| Rebecca | human | 0.556 | 0.871 |
| Gregory | human | 0.530 | 1.000 |
| James | human | 0.735 | 0.728 |
| Maria | human | 0.912 | 0.570 |
| David | human | 0.656 | 0.723 |
| Aisha | human | 0.634 | 0.803 |
| Michael | human | 1.000 | 0.510 |
| Emily | human | 0.723 | 0.734 |
| Carlos | human | 0.856 | 0.643 |
| Priya | human | 0.769 | 0.600 |
| Omar | human | 0.890 | 0.508 |
| Mei | human | 0.818 | 0.675 |
| Sofia | human | 0.804 | 0.635 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.339 (SD=0.265)
- Human mean: 0.774 (SD=0.130)
- Separation: 0.435
- Mann-Whitney U=21.0, p=0.0002

### Factor 2

- AI mean: 0.498 (SD=0.293)
- Human mean: 0.680 (SD=0.130)
- Separation: 0.182
- Mann-Whitney U=78.0, p=0.1585

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.658 | 0.339 | 0.774 |
| replika | ai | 1 | 0.760 | 0.339 | 0.774 |
| bard | ai | 1 | 0.592 | 0.339 | 0.774 |
| eliza | ai | 1 | 0.831 | 0.339 | 0.774 |
| gregory | human | 1 | 0.530 | 0.774 | 0.339 |
| siri | ai | 2 | 0.896 | 0.498 | 0.680 |
| alexa | ai | 2 | 0.738 | 0.498 | 0.680 |
| cortana | ai | 2 | 0.754 | 0.498 | 0.680 |
| google_assistant | ai | 2 | 0.836 | 0.498 | 0.680 |
| bixby | ai | 2 | 0.886 | 0.498 | 0.680 |
| cleverbot | ai | 2 | 0.631 | 0.498 | 0.680 |
| watson | ai | 2 | 0.617 | 0.498 | 0.680 |
| casey | human | 2 | 0.582 | 0.680 | 0.498 |
| maria | human | 2 | 0.570 | 0.680 | 0.498 |
| michael | human | 2 | 0.510 | 0.680 | 0.498 |
| omar | human | 2 | 0.508 | 0.680 | 0.498 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.015 | +0.015 | +0.030 |
| emotions | -0.012 | +0.012 | +0.024 |
| agency | -0.005 | +0.005 | +0.011 |
| intentions | -0.010 | +0.010 | +0.020 |
| prediction | -0.002 | +0.002 | +0.004 |
| cognitive | -0.002 | +0.002 | +0.005 |
| social | -0.004 | +0.004 | +0.008 |
| embodiment | -0.010 | +0.010 | +0.021 |
| roles | +0.005 | -0.005 | -0.010 |
| animacy | -0.018 | +0.018 | +0.036 |
| formality | -0.007 | +0.007 | +0.014 |
| expertise | -0.007 | +0.007 | +0.014 |
| helpfulness | -0.009 | +0.009 | +0.018 |
| biological | -0.022 | +0.022 | +0.044 |
| shapes | +0.002 | -0.002 | -0.005 |
| human | -0.025 | +0.025 | +0.050 |
| ai | +0.013 | -0.013 | -0.026 |
| attention | -0.003 | +0.003 | +0.007 |
| mind | -0.011 | +0.011 | +0.021 |
| beliefs | -0.008 | +0.008 | +0.015 |
| desires | -0.015 | +0.015 | +0.030 |
| goals | -0.009 | +0.009 | +0.017 |

