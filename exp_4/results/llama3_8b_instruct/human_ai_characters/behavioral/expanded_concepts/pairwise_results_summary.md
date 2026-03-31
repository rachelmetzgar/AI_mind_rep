# Concept Geometry, Phase A: Behavioral PCA
## LLaMA-3-8B-Instruct

**Run:** 2026-03-28 13:08:23

---

## What is being tested

Does LLaMA-3-8B-Instruct's explicit judgments about characters' mental properties separate human characters from AI characters? PCA with varimax rotation reveals the latent factor structure of concept ratings across 28 characters (14 AI, 14 human).

## Procedure

- 30 characters: 15 AI, 15 human
- 22 concept dimensions
- 435 pairwise comparisons per dimension
- Total comparisons: 19140
- Method: text generation + parse rating (chat model)

## Response statistics

- Successfully parsed: 19136 / 19140 (100.0%)

### Order consistency

- Pairs with both orders: 9566
- Perfectly consistent: 2849 (29.8%)
- Mean deviation: 1.971

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 8.07 * | 36.7% | 36.7% |
| PC2 | 5.52 * | 25.1% | 61.8% |
| PC3 | 1.93 * | 8.8% | 70.5% |
| PC4 | 1.58 * | 7.2% | 77.7% |
| PC5 | 1.13 * | 5.1% | 82.9% |

*eigenvalue > 1 (retained)

### Varimax-rotated concept loadings

| Concept | F1 | F2 | F3 | F4 |
|---------|----:|----:|----:|----:|
| phenomenology | -0.234 | -0.138 | +0.730 | -0.501 |
| emotions | +0.033 | -0.371 | +0.681 | -0.471 |
| agency | -0.183 | +0.556 | +0.115 | -0.665 |
| intentions | -0.744 | +0.155 | +0.111 | -0.544 |
| prediction | -0.121 | +0.947 | -0.054 | +0.044 |
| cognitive | -0.679 | +0.640 | +0.064 | +0.064 |
| social | +0.077 | +0.940 | +0.074 | +0.031 |
| embodiment | -0.227 | -0.231 | +0.124 | -0.663 |
| roles | -0.703 | -0.007 | -0.119 | -0.273 |
| animacy | -0.276 | -0.258 | +0.562 | -0.658 |
| formality | -0.535 | +0.362 | +0.171 | +0.523 |
| expertise | -0.876 | -0.010 | +0.128 | -0.158 |
| helpfulness | -0.162 | +0.160 | +0.112 | -0.093 |
| biological | -0.304 | -0.196 | +0.138 | -0.789 |
| shapes | +0.024 | -0.135 | -0.782 | -0.073 |
| human | -0.542 | -0.491 | +0.349 | -0.324 |
| ai | +0.086 | +0.857 | -0.292 | +0.297 |
| attention | -0.298 | +0.827 | +0.073 | +0.163 |
| mind | -0.252 | +0.305 | +0.682 | -0.150 |
| beliefs | -0.634 | -0.064 | +0.610 | -0.108 |
| desires | -0.324 | -0.353 | +0.391 | -0.712 |
| goals | -0.891 | +0.121 | +0.182 | -0.239 |

### Character positions (factor scores, 0-1)

| Character | Type | F1 | F2 |
|-----------|------|----:|----:|
| Claude | ai | 0.559 | 0.545 |
| ChatGPT | ai | 0.304 | 1.000 |
| GPT-4 | ai | 0.299 | 0.915 |
| Siri | ai | 0.874 | 0.334 |
| Alexa | ai | 0.761 | 0.546 |
| Cortana | ai | 0.324 | 0.919 |
| Google Assistant | ai | 0.292 | 0.699 |
| Bixby | ai | 0.802 | 0.470 |
| Replika | ai | 1.000 | 0.720 |
| Cleverbot | ai | 0.761 | 0.595 |
| Watson | ai | 0.192 | 0.725 |
| Copilot | ai | 0.415 | 0.430 |
| Bard | ai | 0.549 | 0.537 |
| ELIZA | ai | 0.997 | 0.810 |
| Bing Chat | ai | 0.531 | 0.186 |
| Sam | human | 0.450 | 0.272 |
| Casey | human | 0.513 | 0.287 |
| Rebecca | human | 0.193 | 0.181 |
| Gregory | human | 0.597 | 0.036 |
| James | human | 0.454 | 0.261 |
| Maria | human | 0.439 | 0.051 |
| David | human | 0.000 | 0.170 |
| Aisha | human | 0.496 | 0.421 |
| Michael | human | 0.588 | 0.110 |
| Emily | human | 0.609 | 0.142 |
| Carlos | human | 0.557 | 0.081 |
| Priya | human | 0.217 | 0.233 |
| Omar | human | 0.491 | 0.217 |
| Mei | human | 0.811 | 0.210 |
| Sofia | human | 0.688 | 0.000 |

## Categorical Alignment (AI vs Human)

### Factor 1

- AI mean: 0.577 (SD=0.263)
- Human mean: 0.474 (SD=0.198)
- Separation: 0.104
- Mann-Whitney U=131.0, p=0.4553

### Factor 2

- AI mean: 0.629 (SD=0.221)
- Human mean: 0.178 (SD=0.108)
- Separation: 0.451
- Mann-Whitney U=217.0, p=0.0000

### Factor 3

- AI mean: 0.393 (SD=0.212)
- Human mean: 0.527 (SD=0.167)
- Separation: 0.134
- Mann-Whitney U=64.0, p=0.0465

### Factor 4

- AI mean: 0.533 (SD=0.232)
- Human mean: 0.351 (SD=0.230)
- Separation: 0.182
- Mann-Whitney U=158.0, p=0.0620

### Anomalies (closer to other group)

| Character | Type | Factor | Score | Own mean | Other mean |
|-----------|------|-------:|------:|---------:|-----------:|
| chatgpt | ai | 1 | 0.304 | 0.577 | 0.474 |
| gpt4 | ai | 1 | 0.299 | 0.577 | 0.474 |
| cortana | ai | 1 | 0.324 | 0.577 | 0.474 |
| google_assistant | ai | 1 | 0.292 | 0.577 | 0.474 |
| watson | ai | 1 | 0.192 | 0.577 | 0.474 |
| copilot | ai | 1 | 0.415 | 0.577 | 0.474 |
| gregory | human | 1 | 0.597 | 0.474 | 0.577 |
| michael | human | 1 | 0.588 | 0.474 | 0.577 |
| emily | human | 1 | 0.609 | 0.474 | 0.577 |
| carlos | human | 1 | 0.557 | 0.474 | 0.577 |
| mei | human | 1 | 0.811 | 0.474 | 0.577 |
| sofia | human | 1 | 0.688 | 0.474 | 0.577 |
| siri | ai | 2 | 0.334 | 0.629 | 0.178 |
| bing_chat | ai | 2 | 0.186 | 0.629 | 0.178 |
| aisha | human | 2 | 0.421 | 0.178 | 0.629 |

## Per-Concept Group Means

| Concept | AI mean | Human mean | Difference |
|---------|--------:|-----------:|-----------:|
| phenomenology | -0.229 | +0.229 | +0.458 |
| emotions | -0.418 | +0.418 | +0.837 |
| agency | +0.036 | -0.036 | -0.071 |
| intentions | -0.125 | +0.125 | +0.251 |
| prediction | +0.380 | -0.380 | -0.761 |
| cognitive | +0.198 | -0.198 | -0.395 |
| social | +0.407 | -0.407 | -0.814 |
| embodiment | -0.253 | +0.253 | +0.506 |
| roles | -0.072 | +0.072 | +0.145 |
| animacy | -0.368 | +0.368 | +0.736 |
| formality | +0.099 | -0.099 | -0.198 |
| expertise | -0.128 | +0.128 | +0.255 |
| helpfulness | +0.026 | -0.026 | -0.053 |
| biological | -0.234 | +0.234 | +0.469 |
| shapes | +0.009 | -0.009 | -0.018 |
| human | -0.183 | +0.183 | +0.366 |
| ai | +0.892 | -0.892 | -1.784 |
| attention | +0.187 | -0.187 | -0.375 |
| mind | -0.061 | +0.061 | +0.122 |
| beliefs | -0.155 | +0.155 | +0.310 |
| desires | -0.470 | +0.470 | +0.940 |
| goals | -0.115 | +0.115 | +0.230 |

