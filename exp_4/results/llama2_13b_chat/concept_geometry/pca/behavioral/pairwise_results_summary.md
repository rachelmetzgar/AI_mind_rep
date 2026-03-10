# Concept Geometry, Phase A: Behavioral PCA
## LLaMA-2-13B-Chat

**Run:** 2026-03-09 08:43:08

---

## What is being tested

Does LLaMA-2-13B-Chat's explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

## Procedure

- 30 characters: 15 AI, 15 human
- 21 concept dimensions
- 435 pairwise comparisons per dimension
- Total comparisons: 18270
- Method: text generation + parse rating (chat model)

## Response statistics

- Successfully parsed: 15412 / 18270 (84.4%)

### Order consistency

- Pairs with both orders: 7214
- Perfectly consistent: 850 (11.8%)
- Mean deviation: 2.242

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 9.75 * | 46.4% | 46.4% |
| PC2 | 2.71 * | 12.9% | 59.3% |
| PC3 | 1.80 * | 8.6% | 67.9% |
| PC4 | 1.25 * | 5.9% | 73.8% |
| PC5 | 0.99 | 4.7% | 78.5% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 | F3 | F4 |
|---------|----:|----:|----:|----:|
| phenomenology | +0.271 | -0.775 | +0.108 | +0.038 |
| emotions | -0.365 | -0.343 | -0.584 | +0.318 |
| agency | -0.579 | -0.048 | +0.369 | +0.155 |
| intentions | -0.679 | -0.002 | +0.256 | +0.327 |
| prediction | -0.875 | +0.163 | +0.111 | +0.273 |
| cognitive | -0.650 | +0.010 | +0.306 | +0.231 |
| social | -0.772 | +0.268 | -0.140 | +0.290 |
| embodiment | +0.063 | -0.859 | -0.123 | -0.309 |
| roles | +0.009 | -0.170 | +0.790 | +0.195 |
| animacy | -0.596 | +0.645 | -0.197 | +0.327 |
| formality | -0.743 | +0.445 | -0.301 | -0.006 |
| expertise | -0.811 | +0.304 | -0.080 | +0.172 |
| helpfulness | -0.495 | +0.712 | -0.239 | +0.147 |
| biological | -0.127 | +0.240 | +0.111 | +0.669 |
| shapes | -0.798 | +0.225 | -0.257 | -0.116 |
| human | -0.472 | +0.331 | +0.146 | +0.645 |
| ai | +0.228 | -0.383 | +0.399 | -0.640 |
| attention | -0.412 | -0.147 | +0.268 | +0.631 |
| beliefs | -0.882 | +0.307 | -0.083 | +0.164 |
| desires | +0.221 | -0.796 | -0.026 | +0.076 |
| goals | -0.847 | +0.070 | +0.043 | +0.208 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.513 | 0.754 |
| ChatGPT | ai | 0.395 | 0.742 |
| GPT-4 | ai | 1.000 | 0.871 |
| Siri | ai | 0.054 | 0.693 |
| Alexa | ai | 0.113 | 0.595 |
| Cortana | ai | 0.000 | 0.484 |
| Google Assistant | ai | 0.289 | 0.772 |
| Bixby | ai | 0.176 | 0.643 |
| Replika | ai | 0.739 | 0.929 |
| Cleverbot | ai | 0.483 | 0.667 |
| Watson | ai | 0.174 | 0.566 |
| Copilot | ai | 0.505 | 1.000 |
| Bard | ai | 0.108 | 0.524 |
| ELIZA | ai | 0.306 | 0.663 |
| Bing Chat | ai | 0.291 | 0.669 |
| Sam | human | 0.416 | 0.319 |
| Casey | human | 0.472 | 0.468 |
| Rebecca | human | 0.828 | 0.369 |
| Gregory | human | 0.975 | 0.632 |
| James | human | 0.462 | 0.415 |
| Maria | human | 0.541 | 0.232 |
| David | human | 0.478 | 0.533 |
| Aisha | human | 0.524 | 0.000 |
| Michael | human | 0.257 | 0.289 |
| Emily | human | 0.622 | 0.316 |
| Carlos | human | 0.570 | 0.296 |
| Priya | human | 0.696 | 0.168 |
| Omar | human | 0.515 | 0.017 |
| Mei | human | 0.651 | 0.495 |
| Sofia | human | 0.843 | 0.693 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.343 (SD=0.263)
- Human mean: 0.590 (SD=0.179)
- Separation: 0.247
- Mann-Whitney U=46.0, p=0.0062

### Factor 2

- AI mean: 0.705 (SD=0.140)
- Human mean: 0.349 (SD=0.193)
- Separation: 0.355
- Mann-Whitney U=210.0, p=0.0001

### Factor 3

- AI mean: 0.452 (SD=0.183)
- Human mean: 0.449 (SD=0.233)
- Separation: 0.003
- Mann-Whitney U=108.0, p=0.8682

### Factor 4

- AI mean: 0.733 (SD=0.159)
- Human mean: 0.577 (SD=0.278)
- Separation: 0.156
- Mann-Whitney U=149.0, p=0.1354

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.513 | 0.343 | 0.590 |
| gpt4 | ai | 1 | 1.000 | 0.343 | 0.590 |
| replika | ai | 1 | 0.739 | 0.343 | 0.590 |
| cleverbot | ai | 1 | 0.483 | 0.343 | 0.590 |
| copilot | ai | 1 | 0.505 | 0.343 | 0.590 |
| sam | human | 1 | 0.416 | 0.590 | 0.343 |
| james | human | 1 | 0.462 | 0.590 | 0.343 |
| michael | human | 1 | 0.257 | 0.590 | 0.343 |
| cortana | ai | 2 | 0.484 | 0.705 | 0.349 |
| bard | ai | 2 | 0.524 | 0.705 | 0.349 |
| gregory | human | 2 | 0.632 | 0.349 | 0.705 |
| david | human | 2 | 0.533 | 0.349 | 0.705 |
| sofia | human | 2 | 0.693 | 0.349 | 0.705 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.119 | +0.293 | +0.412 |
| emotions | +0.015 | -0.010 | -0.025 |
| agency | +0.056 | -0.093 | -0.150 |
| intentions | +0.109 | -0.123 | -0.232 |
| prediction | +0.187 | -0.178 | -0.365 |
| cognitive | +0.083 | -0.134 | -0.216 |
| social | +0.174 | -0.180 | -0.354 |
| embodiment | -0.191 | +0.538 | +0.729 |
| roles | +0.003 | +0.000 | -0.003 |
| animacy | +0.506 | -0.869 | -1.375 |
| formality | +0.256 | -0.241 | -0.497 |
| expertise | +0.299 | -0.314 | -0.613 |
| helpfulness | +0.310 | -0.313 | -0.623 |
| biological | +0.085 | -0.097 | -0.182 |
| shapes | +0.127 | -0.130 | -0.256 |
| human | +0.208 | -0.418 | -0.627 |
| ai | -0.353 | +0.363 | +0.715 |
| attention | +0.081 | -0.074 | -0.156 |
| beliefs | +0.257 | -0.281 | -0.538 |
| desires | -0.195 | +0.577 | +0.772 |
| goals | +0.174 | -0.172 | -0.346 |

