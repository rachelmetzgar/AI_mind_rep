# Concept Geometry, Phase A: Behavioral PCA
## Qwen-2.5-7B-Instruct

**Run:** 2026-03-28 12:14:38

---

## What is being tested

Does Qwen-2.5-7B-Instruct's explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

## Procedure

- 30 characters: 15 AI, 15 human
- 22 concept dimensions
- 435 pairwise comparisons per dimension
- Total comparisons: 19140
- Method: text generation + parse rating (chat model)

## Response statistics

- Successfully parsed: 19140 / 19140 (100.0%)

### Order consistency

- Pairs with both orders: 9570
- Perfectly consistent: 3777 (39.5%)
- Mean deviation: 1.813

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 12.38 * | 56.3% | 56.3% |
| PC2 | 3.76 * | 17.1% | 73.4% |
| PC3 | 1.64 * | 7.5% | 80.9% |
| PC4 | 1.25 * | 5.7% | 86.5% |
| PC5 | 0.64 | 2.9% | 89.5% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 | F3 | F4 |
|---------|----:|----:|----:|----:|
| phenomenology | +0.964 | +0.103 | -0.077 | +0.077 |
| emotions | +0.679 | -0.112 | -0.580 | -0.256 |
| agency | +0.025 | -0.015 | -0.886 | +0.281 |
| intentions | +0.841 | +0.416 | -0.011 | +0.252 |
| prediction | +0.663 | +0.596 | +0.158 | +0.124 |
| cognitive | +0.355 | +0.834 | +0.177 | +0.182 |
| social | +0.841 | +0.405 | +0.017 | +0.108 |
| embodiment | +0.914 | +0.015 | -0.230 | +0.020 |
| roles | +0.009 | +0.419 | +0.029 | +0.797 |
| animacy | +0.954 | +0.177 | -0.174 | +0.110 |
| formality | -0.452 | +0.775 | -0.087 | -0.120 |
| expertise | +0.664 | +0.686 | +0.031 | +0.141 |
| helpfulness | +0.140 | -0.030 | -0.230 | +0.809 |
| biological | +0.960 | +0.175 | -0.131 | +0.085 |
| shapes | +0.382 | -0.246 | -0.616 | -0.350 |
| human | +0.723 | +0.335 | -0.165 | -0.171 |
| ai | -0.897 | -0.051 | -0.183 | -0.083 |
| attention | +0.340 | +0.644 | +0.186 | +0.494 |
| mind | +0.440 | +0.805 | -0.028 | +0.168 |
| beliefs | +0.918 | +0.253 | +0.042 | +0.026 |
| desires | +0.955 | +0.084 | -0.136 | -0.033 |
| goals | +0.672 | +0.488 | +0.001 | +0.433 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.200 | 0.693 |
| ChatGPT | ai | 0.066 | 0.652 |
| GPT-4 | ai | 0.000 | 1.000 |
| Siri | ai | 0.156 | 0.174 |
| Alexa | ai | 0.152 | 0.134 |
| Cortana | ai | 0.232 | 0.224 |
| Google Assistant | ai | 0.054 | 0.498 |
| Bixby | ai | 0.327 | 0.032 |
| Replika | ai | 0.453 | 0.000 |
| Cleverbot | ai | 0.377 | 0.294 |
| Watson | ai | 0.073 | 0.975 |
| Copilot | ai | 0.181 | 0.426 |
| Bard | ai | 0.232 | 0.303 |
| ELIZA | ai | 0.464 | 0.575 |
| Bing Chat | ai | 0.125 | 0.748 |
| Sam | human | 0.918 | 0.367 |
| Casey | human | 0.956 | 0.350 |
| Rebecca | human | 0.834 | 0.864 |
| Gregory | human | 0.907 | 0.408 |
| James | human | 0.866 | 0.564 |
| Maria | human | 0.908 | 0.589 |
| David | human | 0.725 | 0.772 |
| Aisha | human | 0.892 | 0.671 |
| Michael | human | 0.999 | 0.368 |
| Emily | human | 0.969 | 0.479 |
| Carlos | human | 0.873 | 0.474 |
| Priya | human | 0.752 | 0.879 |
| Omar | human | 1.000 | 0.398 |
| Mei | human | 0.927 | 0.579 |
| Sofia | human | 0.706 | 0.563 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.206 (SD=0.139)
- Human mean: 0.882 (SD=0.090)
- Separation: 0.676
- Mann-Whitney U=0.0, p=0.0000

### Factor 2

- AI mean: 0.449 (SD=0.309)
- Human mean: 0.555 (SD=0.170)
- Separation: 0.106
- Mann-Whitney U=86.0, p=0.2808

### Factor 3

- AI mean: 0.473 (SD=0.249)
- Human mean: 0.435 (SD=0.109)
- Separation: 0.039
- Mann-Whitney U=127.0, p=0.5614

### Factor 4

- AI mean: 0.556 (SD=0.260)
- Human mean: 0.619 (SD=0.082)
- Separation: 0.063
- Mann-Whitney U=92.0, p=0.4068

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 2 | 0.693 | 0.449 | 0.555 |
| chatgpt | ai | 2 | 0.652 | 0.449 | 0.555 |
| gpt4 | ai | 2 | 1.000 | 0.449 | 0.555 |
| watson | ai | 2 | 0.975 | 0.449 | 0.555 |
| eliza | ai | 2 | 0.575 | 0.449 | 0.555 |
| bing_chat | ai | 2 | 0.748 | 0.449 | 0.555 |
| sam | human | 2 | 0.367 | 0.555 | 0.449 |
| casey | human | 2 | 0.350 | 0.555 | 0.449 |
| gregory | human | 2 | 0.408 | 0.555 | 0.449 |
| michael | human | 2 | 0.368 | 0.555 | 0.449 |
| emily | human | 2 | 0.479 | 0.555 | 0.449 |
| carlos | human | 2 | 0.474 | 0.555 | 0.449 |
| omar | human | 2 | 0.398 | 0.555 | 0.449 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.799 | +0.799 | +1.598 |
| emotions | -0.261 | +0.261 | +0.522 |
| agency | -0.049 | +0.049 | +0.099 |
| intentions | -0.734 | +0.734 | +1.469 |
| prediction | -0.437 | +0.437 | +0.874 |
| cognitive | -0.320 | +0.320 | +0.639 |
| social | -0.632 | +0.632 | +1.264 |
| embodiment | -0.768 | +0.768 | +1.536 |
| roles | -0.051 | +0.051 | +0.101 |
| animacy | -0.970 | +0.970 | +1.940 |
| formality | +0.189 | -0.189 | -0.377 |
| expertise | -0.575 | +0.575 | +1.149 |
| helpfulness | -0.071 | +0.071 | +0.143 |
| biological | -1.025 | +1.025 | +2.051 |
| shapes | -0.129 | +0.129 | +0.257 |
| human | -0.382 | +0.382 | +0.763 |
| ai | +0.736 | -0.736 | -1.471 |
| attention | -0.176 | +0.176 | +0.352 |
| mind | -0.360 | +0.360 | +0.720 |
| beliefs | -0.643 | +0.643 | +1.285 |
| desires | -0.793 | +0.793 | +1.586 |
| goals | -0.416 | +0.416 | +0.832 |

