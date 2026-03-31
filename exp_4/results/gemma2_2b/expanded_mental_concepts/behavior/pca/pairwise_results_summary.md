# Concept Geometry, Phase A: Behavioral PCA
## Gemma-2-2B (Base)

**Run:** 2026-03-29 11:06:44

---

## What is being tested

Does Gemma-2-2B (Base)'s explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

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
- Perfectly consistent: 9408 (98.3%)
- Mean deviation: 0.140

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 19.79 * | 90.0% | 90.0% |
| PC2 | 0.98 | 4.4% | 94.4% |
| PC3 | 0.48 | 2.2% | 96.6% |
| PC4 | 0.21 | 1.0% | 97.5% |
| PC5 | 0.13 | 0.6% | 98.1% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 |
|---------|----:|----:|
| phenomenology | +0.852 | +0.498 |
| emotions | +0.839 | +0.510 |
| agency | +0.828 | +0.520 |
| intentions | +0.921 | +0.368 |
| prediction | +0.581 | +0.793 |
| cognitive | +0.687 | +0.667 |
| social | +0.670 | +0.671 |
| embodiment | +0.878 | +0.448 |
| roles | +0.770 | +0.528 |
| animacy | +0.634 | +0.716 |
| formality | +0.627 | +0.726 |
| expertise | +0.734 | +0.585 |
| helpfulness | +0.487 | +0.848 |
| biological | +0.758 | +0.619 |
| shapes | +0.633 | +0.724 |
| human | +0.758 | +0.578 |
| ai | +0.232 | +0.935 |
| attention | +0.721 | +0.671 |
| mind | +0.838 | +0.518 |
| beliefs | +0.705 | +0.690 |
| desires | +0.831 | +0.543 |
| goals | +0.943 | +0.292 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.913 | 0.000 |
| ChatGPT | ai | 0.473 | 0.498 |
| GPT-4 | ai | 0.439 | 0.329 |
| Siri | ai | 0.275 | 0.544 |
| Alexa | ai | 0.000 | 0.425 |
| Cortana | ai | 0.335 | 0.436 |
| Google Assistant | ai | 0.240 | 0.398 |
| Bixby | ai | 0.166 | 0.521 |
| Replika | ai | 0.318 | 0.656 |
| Cleverbot | ai | 0.490 | 0.610 |
| Watson | ai | 0.293 | 0.423 |
| Copilot | ai | 0.173 | 0.497 |
| Bard | ai | 0.316 | 0.277 |
| ELIZA | ai | 0.705 | 1.000 |
| Bing Chat | ai | 0.490 | 0.333 |
| Sam | human | 0.922 | 0.604 |
| Casey | human | 0.935 | 0.349 |
| Rebecca | human | 0.844 | 0.427 |
| Gregory | human | 0.964 | 0.605 |
| James | human | 0.797 | 0.447 |
| Maria | human | 0.933 | 0.640 |
| David | human | 0.869 | 0.520 |
| Aisha | human | 1.000 | 0.223 |
| Michael | human | 0.840 | 0.505 |
| Emily | human | 0.839 | 0.485 |
| Carlos | human | 0.859 | 0.444 |
| Priya | human | 0.773 | 0.534 |
| Omar | human | 0.907 | 0.564 |
| Mei | human | 0.936 | 0.515 |
| Sofia | human | 0.781 | 0.396 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.375 (SD=0.217)
- Human mean: 0.880 (SD=0.066)
- Separation: 0.505
- Mann-Whitney U=9.0, p=0.0000

### Factor 2

- AI mean: 0.463 (SD=0.208)
- Human mean: 0.484 (SD=0.105)
- Separation: 0.021
- Mann-Whitney U=95.0, p=0.4807

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.913 | 0.375 | 0.880 |
| eliza | ai | 1 | 0.705 | 0.375 | 0.880 |
| chatgpt | ai | 2 | 0.498 | 0.463 | 0.484 |
| siri | ai | 2 | 0.544 | 0.463 | 0.484 |
| bixby | ai | 2 | 0.521 | 0.463 | 0.484 |
| replika | ai | 2 | 0.656 | 0.463 | 0.484 |
| cleverbot | ai | 2 | 0.610 | 0.463 | 0.484 |
| copilot | ai | 2 | 0.497 | 0.463 | 0.484 |
| eliza | ai | 2 | 1.000 | 0.463 | 0.484 |
| casey | human | 2 | 0.349 | 0.484 | 0.463 |
| rebecca | human | 2 | 0.427 | 0.484 | 0.463 |
| james | human | 2 | 0.447 | 0.484 | 0.463 |
| aisha | human | 2 | 0.223 | 0.484 | 0.463 |
| carlos | human | 2 | 0.444 | 0.484 | 0.463 |
| sofia | human | 2 | 0.396 | 0.484 | 0.463 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.012 | +0.012 | +0.024 |
| emotions | -0.012 | +0.012 | +0.024 |
| agency | -0.008 | +0.008 | +0.016 |
| intentions | -0.013 | +0.013 | +0.026 |
| prediction | -0.007 | +0.007 | +0.014 |
| cognitive | -0.007 | +0.007 | +0.015 |
| social | -0.008 | +0.008 | +0.016 |
| embodiment | -0.014 | +0.014 | +0.028 |
| roles | -0.010 | +0.010 | +0.019 |
| animacy | -0.010 | +0.010 | +0.019 |
| formality | -0.006 | +0.006 | +0.013 |
| expertise | -0.007 | +0.007 | +0.014 |
| helpfulness | -0.005 | +0.005 | +0.011 |
| biological | -0.008 | +0.008 | +0.016 |
| shapes | -0.007 | +0.007 | +0.015 |
| human | -0.011 | +0.011 | +0.022 |
| ai | -0.002 | +0.002 | +0.003 |
| attention | -0.009 | +0.009 | +0.017 |
| mind | -0.011 | +0.011 | +0.023 |
| beliefs | -0.010 | +0.010 | +0.021 |
| desires | -0.010 | +0.010 | +0.021 |
| goals | -0.014 | +0.014 | +0.027 |

