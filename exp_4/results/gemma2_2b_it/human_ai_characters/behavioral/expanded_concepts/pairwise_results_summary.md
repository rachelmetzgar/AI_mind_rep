# Concept Geometry, Phase A: Behavioral PCA
## Gemma-2-2B-IT

**Run:** 2026-03-29 11:59:09

---

## What is being tested

Does Gemma-2-2B-IT's explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

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
- Perfectly consistent: 683 (7.1%)
- Mean deviation: 2.400

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 9.07 * | 41.2% | 41.2% |
| PC2 | 5.94 * | 27.0% | 68.2% |
| PC3 | 2.09 * | 9.5% | 77.7% |
| PC4 | 1.26 * | 5.7% | 83.4% |
| PC5 | 1.21 * | 5.5% | 88.9% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 | F3 | F4 |
|---------|----:|----:|----:|----:|
| phenomenology | -0.202 | +0.158 | -0.268 | +0.067 |
| emotions | -0.691 | -0.381 | -0.243 | +0.025 |
| agency | -0.873 | +0.164 | -0.266 | +0.158 |
| intentions | -0.912 | +0.258 | -0.047 | +0.048 |
| prediction | -0.127 | +0.926 | -0.080 | +0.019 |
| cognitive | -0.218 | +0.801 | -0.055 | +0.465 |
| social | -0.252 | +0.782 | +0.013 | -0.072 |
| embodiment | -0.683 | +0.076 | -0.403 | -0.131 |
| roles | +0.137 | +0.737 | -0.239 | -0.019 |
| animacy | -0.275 | +0.273 | -0.877 | -0.004 |
| formality | -0.000 | +0.887 | -0.185 | +0.204 |
| expertise | -0.625 | +0.513 | -0.128 | +0.461 |
| helpfulness | -0.328 | +0.707 | -0.088 | -0.371 |
| biological | -0.094 | +0.111 | -0.908 | +0.091 |
| shapes | +0.019 | +0.548 | -0.550 | -0.494 |
| human | -0.703 | -0.245 | -0.531 | -0.175 |
| ai | +0.637 | +0.723 | -0.074 | +0.137 |
| attention | -0.009 | +0.971 | -0.153 | -0.048 |
| mind | -0.349 | +0.335 | -0.574 | +0.550 |
| beliefs | -0.805 | +0.110 | -0.153 | +0.302 |
| desires | -0.785 | -0.158 | -0.275 | -0.141 |
| goals | -0.947 | +0.151 | +0.070 | +0.054 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.000 | 1.000 |
| ChatGPT | ai | 0.707 | 0.471 |
| GPT-4 | ai | 0.552 | 0.702 |
| Siri | ai | 0.538 | 0.798 |
| Alexa | ai | 0.637 | 0.558 |
| Cortana | ai | 0.578 | 0.603 |
| Google Assistant | ai | 0.951 | 0.439 |
| Bixby | ai | 1.000 | 0.229 |
| Replika | ai | 0.750 | 0.416 |
| Cleverbot | ai | 0.959 | 0.266 |
| Watson | ai | 0.337 | 0.754 |
| Copilot | ai | 0.472 | 0.944 |
| Bard | ai | 0.445 | 0.727 |
| ELIZA | ai | 0.974 | 0.268 |
| Bing Chat | ai | 0.902 | 0.318 |
| Sam | human | 0.332 | 0.144 |
| Casey | human | 0.327 | 0.214 |
| Rebecca | human | 0.272 | 0.301 |
| Gregory | human | 0.214 | 0.225 |
| James | human | 0.331 | 0.134 |
| Maria | human | 0.292 | 0.198 |
| David | human | 0.341 | 0.120 |
| Aisha | human | 0.381 | 0.265 |
| Michael | human | 0.358 | 0.000 |
| Emily | human | 0.296 | 0.181 |
| Carlos | human | 0.407 | 0.042 |
| Priya | human | 0.309 | 0.210 |
| Omar | human | 0.195 | 0.263 |
| Mei | human | 0.071 | 0.376 |
| Sofia | human | 0.164 | 0.323 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.653 (SD=0.272)
- Human mean: 0.286 (SD=0.087)
- Separation: 0.368
- Mann-Whitney U=206.0, p=0.0001

### Factor 2

- AI mean: 0.566 (SD=0.241)
- Human mean: 0.200 (SD=0.098)
- Separation: 0.366
- Mann-Whitney U=212.0, p=0.0000

### Factor 3

- AI mean: 0.600 (SD=0.242)
- Human mean: 0.612 (SD=0.111)
- Separation: 0.011
- Mann-Whitney U=105.0, p=0.7716

### Factor 4

- AI mean: 0.581 (SD=0.297)
- Human mean: 0.550 (SD=0.117)
- Separation: 0.031
- Mann-Whitney U=130.0, p=0.4807

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| claude | ai | 1 | 0.000 | 0.653 | 0.286 |
| watson | ai | 1 | 0.337 | 0.653 | 0.286 |
| bard | ai | 1 | 0.445 | 0.653 | 0.286 |
| bixby | ai | 2 | 0.229 | 0.566 | 0.200 |
| cleverbot | ai | 2 | 0.266 | 0.566 | 0.200 |
| eliza | ai | 2 | 0.268 | 0.566 | 0.200 |
| bing_chat | ai | 2 | 0.318 | 0.566 | 0.200 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.009 | +0.009 | +0.018 |
| emotions | -0.246 | +0.246 | +0.492 |
| agency | -0.108 | +0.108 | +0.216 |
| intentions | -0.072 | +0.072 | +0.145 |
| prediction | +0.095 | -0.095 | -0.191 |
| cognitive | +0.130 | -0.130 | -0.260 |
| social | +0.079 | -0.079 | -0.159 |
| embodiment | -0.066 | +0.066 | +0.131 |
| roles | +0.144 | -0.144 | -0.287 |
| animacy | +0.016 | -0.016 | -0.032 |
| formality | +0.244 | -0.244 | -0.487 |
| expertise | -0.002 | +0.002 | +0.005 |
| helpfulness | +0.041 | -0.041 | -0.083 |
| biological | +0.021 | -0.021 | -0.041 |
| shapes | +0.092 | -0.092 | -0.184 |
| human | -0.197 | +0.197 | +0.393 |
| ai | +1.030 | -1.030 | -2.060 |
| attention | +0.141 | -0.141 | -0.283 |
| mind | +0.014 | -0.014 | -0.028 |
| beliefs | -0.080 | +0.080 | +0.161 |
| desires | -0.138 | +0.138 | +0.276 |
| goals | -0.106 | +0.106 | +0.211 |

