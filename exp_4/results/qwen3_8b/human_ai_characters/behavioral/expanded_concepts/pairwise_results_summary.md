# Concept Geometry, Phase A: Behavioral PCA
## Qwen3-8B

**Run:** 2026-03-28 15:04:31

---

## What is being tested

Does Qwen3-8B's explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

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
- Perfectly consistent: 2805 (29.3%)
- Mean deviation: 1.381

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 10.28 * | 46.7% | 46.7% |
| PC2 | 7.80 * | 35.5% | 82.2% |
| PC3 | 1.03 * | 4.7% | 86.9% |
| PC4 | 0.68 | 3.1% | 90.0% |
| PC5 | 0.45 | 2.1% | 92.0% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 | F3 |
|---------|----:|----:|----:|
| phenomenology | -0.185 | -0.932 | -0.150 |
| emotions | -0.207 | -0.944 | -0.114 |
| agency | -0.840 | +0.323 | +0.224 |
| intentions | -0.781 | -0.418 | -0.323 |
| prediction | -0.912 | +0.110 | -0.163 |
| cognitive | -0.945 | +0.030 | -0.165 |
| social | -0.640 | -0.552 | -0.265 |
| embodiment | +0.223 | -0.911 | +0.145 |
| roles | -0.794 | +0.138 | -0.448 |
| animacy | -0.019 | -0.967 | +0.002 |
| formality | -0.801 | +0.468 | +0.178 |
| expertise | -0.923 | -0.076 | +0.141 |
| helpfulness | -0.544 | -0.149 | -0.674 |
| biological | +0.167 | -0.863 | +0.109 |
| shapes | +0.082 | +0.722 | +0.526 |
| human | +0.052 | -0.944 | -0.151 |
| ai | -0.417 | +0.791 | +0.131 |
| attention | -0.847 | -0.056 | -0.375 |
| mind | -0.740 | -0.527 | -0.137 |
| beliefs | -0.738 | -0.585 | -0.143 |
| desires | -0.326 | -0.868 | -0.048 |
| goals | -0.926 | -0.144 | -0.109 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.344 | 0.632 |
| ChatGPT | ai | 0.296 | 0.710 |
| GPT-4 | ai | 0.000 | 0.703 |
| Siri | ai | 0.811 | 0.912 |
| Alexa | ai | 0.833 | 0.969 |
| Cortana | ai | 0.725 | 0.673 |
| Google Assistant | ai | 0.695 | 1.000 |
| Bixby | ai | 0.563 | 0.756 |
| Replika | ai | 0.658 | 0.561 |
| Cleverbot | ai | 0.749 | 0.598 |
| Watson | ai | 0.221 | 0.842 |
| Copilot | ai | 0.403 | 0.717 |
| Bard | ai | 0.579 | 0.666 |
| ELIZA | ai | 1.000 | 0.732 |
| Bing Chat | ai | 0.608 | 0.836 |
| Sam | human | 0.625 | 0.169 |
| Casey | human | 0.620 | 0.118 |
| Rebecca | human | 0.271 | 0.128 |
| Gregory | human | 0.790 | 0.223 |
| James | human | 0.722 | 0.269 |
| Maria | human | 0.586 | 0.151 |
| David | human | 0.536 | 0.205 |
| Aisha | human | 0.628 | 0.047 |
| Michael | human | 0.679 | 0.192 |
| Emily | human | 0.634 | 0.171 |
| Carlos | human | 0.743 | 0.206 |
| Priya | human | 0.231 | 0.000 |
| Omar | human | 0.732 | 0.165 |
| Mei | human | 0.686 | 0.115 |
| Sofia | human | 0.754 | 0.307 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.566 (SD=0.257)
- Human mean: 0.616 (SD=0.158)
- Separation: 0.050
- Mann-Whitney U=103.0, p=0.7089

### Factor 2

- AI mean: 0.754 (SD=0.127)
- Human mean: 0.164 (SD=0.076)
- Separation: 0.589
- Mann-Whitney U=225.0, p=0.0000

### Factor 3

- AI mean: 0.479 (SD=0.311)
- Human mean: 0.434 (SD=0.143)
- Separation: 0.045
- Mann-Whitney U=118.0, p=0.8357

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| siri | ai | 1 | 0.811 | 0.566 | 0.616 |
| alexa | ai | 1 | 0.833 | 0.566 | 0.616 |
| cortana | ai | 1 | 0.725 | 0.566 | 0.616 |
| google_assistant | ai | 1 | 0.695 | 0.566 | 0.616 |
| replika | ai | 1 | 0.658 | 0.566 | 0.616 |
| cleverbot | ai | 1 | 0.749 | 0.566 | 0.616 |
| eliza | ai | 1 | 1.000 | 0.566 | 0.616 |
| bing_chat | ai | 1 | 0.608 | 0.566 | 0.616 |
| rebecca | human | 1 | 0.271 | 0.616 | 0.566 |
| maria | human | 1 | 0.586 | 0.616 | 0.566 |
| david | human | 1 | 0.536 | 0.616 | 0.566 |
| priya | human | 1 | 0.231 | 0.616 | 0.566 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.337 | +0.337 | +0.674 |
| emotions | -0.377 | +0.377 | +0.754 |
| agency | +0.078 | -0.078 | -0.156 |
| intentions | -0.092 | +0.092 | +0.184 |
| prediction | +0.047 | -0.047 | -0.094 |
| cognitive | +0.075 | -0.075 | -0.149 |
| social | -0.097 | +0.097 | +0.193 |
| embodiment | -0.544 | +0.544 | +1.087 |
| roles | +0.034 | -0.034 | -0.069 |
| animacy | -0.475 | +0.475 | +0.949 |
| formality | +0.159 | -0.159 | -0.317 |
| expertise | +0.016 | -0.016 | -0.032 |
| helpfulness | -0.013 | +0.013 | +0.025 |
| biological | -0.467 | +0.467 | +0.933 |
| shapes | +0.147 | -0.147 | -0.294 |
| human | -0.745 | +0.745 | +1.490 |
| ai | +0.543 | -0.543 | -1.085 |
| attention | -0.001 | +0.001 | +0.002 |
| mind | -0.085 | +0.085 | +0.170 |
| beliefs | -0.098 | +0.098 | +0.195 |
| desires | -0.169 | +0.169 | +0.338 |
| goals | -0.031 | +0.031 | +0.062 |

