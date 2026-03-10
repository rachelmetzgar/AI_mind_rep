# Concept Geometry, Phase A: Behavioral PCA
## LLaMA-2-13B (Base)

**Run:** 2026-03-08 15:33:10

---

## What is being tested

Does LLaMA-2-13B (Base)'s explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

## Procedure

- 30 characters: 15 AI, 15 human
- 21 concept dimensions
- 435 pairwise comparisons per dimension
- Total comparisons: 18270
- Method: logit extraction over tokens 1-5 (base model)

## Response statistics

- All 18270 comparisons yield ratings (logit-based)

### Order consistency

- Pairs with both orders: 9135
- Perfectly consistent: 26 (0.3%)
- Mean deviation: 1.055

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 19.06 * | 90.8% | 90.8% |
| PC2 | 0.64 | 3.0% | 93.8% |
| PC3 | 0.33 | 1.6% | 95.4% |
| PC4 | 0.24 | 1.1% | 96.5% |
| PC5 | 0.18 | 0.9% | 97.4% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 |
|---------|----:|----:|
| phenomenology | +0.799 | +0.580 |
| emotions | +0.821 | +0.528 |
| agency | +0.768 | +0.604 |
| intentions | +0.897 | +0.419 |
| prediction | +0.760 | +0.597 |
| cognitive | +0.759 | +0.625 |
| social | +0.758 | +0.615 |
| embodiment | +0.769 | +0.606 |
| roles | +0.787 | +0.511 |
| animacy | +0.735 | +0.594 |
| formality | +0.541 | +0.788 |
| expertise | +0.686 | +0.698 |
| helpfulness | +0.610 | +0.733 |
| biological | +0.479 | +0.843 |
| shapes | +0.424 | +0.895 |
| human | +0.813 | +0.507 |
| ai | +0.551 | +0.786 |
| attention | +0.759 | +0.542 |
| beliefs | +0.882 | +0.420 |
| desires | +0.772 | +0.589 |
| goals | +0.780 | +0.579 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.902 | 0.536 |
| ChatGPT | ai | 0.479 | 0.533 |
| GPT-4 | ai | 0.744 | 0.610 |
| Siri | ai | 0.000 | 0.393 |
| Alexa | ai | 0.478 | 0.573 |
| Cortana | ai | 0.471 | 0.688 |
| Google Assistant | ai | 0.223 | 0.535 |
| Bixby | ai | 0.166 | 0.631 |
| Replika | ai | 0.794 | 0.655 |
| Cleverbot | ai | 0.566 | 0.525 |
| Watson | ai | 0.430 | 0.225 |
| Copilot | ai | 0.653 | 0.573 |
| Bard | ai | 0.586 | 0.528 |
| ELIZA | ai | 0.736 | 0.668 |
| Bing Chat | ai | 0.574 | 1.000 |
| Sam | human | 0.526 | 0.043 |
| Casey | human | 0.885 | 0.630 |
| Rebecca | human | 1.000 | 0.729 |
| Gregory | human | 0.430 | 0.000 |
| James | human | 0.944 | 0.239 |
| Maria | human | 0.712 | 0.280 |
| David | human | 0.883 | 0.158 |
| Aisha | human | 0.667 | 0.625 |
| Michael | human | 0.764 | 0.385 |
| Emily | human | 0.793 | 0.327 |
| Carlos | human | 0.739 | 0.224 |
| Priya | human | 0.929 | 0.494 |
| Omar | human | 0.856 | 0.255 |
| Mei | human | 0.536 | 0.372 |
| Sofia | human | 0.690 | 0.261 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.520 (SD=0.237)
- Human mean: 0.757 (SD=0.161)
- Separation: 0.237
- Mann-Whitney U=49.0, p=0.0090

### Factor 2

- AI mean: 0.578 (SD=0.159)
- Human mean: 0.335 (SD=0.204)
- Separation: 0.243
- Mann-Whitney U=182.0, p=0.0042

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.902 | 0.520 | 0.757 |
| gpt4 | ai | 1 | 0.744 | 0.520 | 0.757 |
| replika | ai | 1 | 0.794 | 0.520 | 0.757 |
| copilot | ai | 1 | 0.653 | 0.520 | 0.757 |
| eliza | ai | 1 | 0.736 | 0.520 | 0.757 |
| sam | human | 1 | 0.526 | 0.757 | 0.520 |
| gregory | human | 1 | 0.430 | 0.757 | 0.520 |
| mei | human | 1 | 0.536 | 0.757 | 0.520 |
| siri | ai | 2 | 0.393 | 0.578 | 0.335 |
| watson | ai | 2 | 0.225 | 0.578 | 0.335 |
| casey | human | 2 | 0.630 | 0.335 | 0.578 |
| rebecca | human | 2 | 0.729 | 0.335 | 0.578 |
| aisha | human | 2 | 0.625 | 0.335 | 0.578 |
| priya | human | 2 | 0.494 | 0.335 | 0.578 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.001 | +0.001 | +0.003 |
| emotions | -0.006 | +0.006 | +0.012 |
| agency | -0.005 | +0.005 | +0.010 |
| intentions | -0.009 | +0.009 | +0.017 |
| prediction | -0.003 | +0.003 | +0.006 |
| cognitive | -0.001 | +0.001 | +0.001 |
| social | -0.001 | +0.001 | +0.001 |
| embodiment | -0.001 | +0.001 | +0.002 |
| roles | +0.000 | -0.000 | -0.001 |
| animacy | -0.006 | +0.006 | +0.013 |
| formality | +0.003 | -0.003 | -0.007 |
| expertise | +0.004 | -0.004 | -0.008 |
| helpfulness | +0.001 | -0.001 | -0.002 |
| biological | +0.008 | -0.008 | -0.016 |
| shapes | +0.011 | -0.011 | -0.023 |
| human | -0.009 | +0.009 | +0.019 |
| ai | +0.009 | -0.009 | -0.018 |
| attention | -0.007 | +0.007 | +0.013 |
| beliefs | -0.007 | +0.007 | +0.015 |
| desires | +0.002 | -0.002 | -0.004 |
| goals | -0.001 | +0.001 | +0.001 |

