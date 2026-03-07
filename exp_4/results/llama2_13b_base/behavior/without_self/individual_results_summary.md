# Experiment 4, Phase 2c: Individual Likert Ratings
## BASE MODEL (LLaMA-2-13B, no chat fine-tuning)

**Run:** 2026-02-19 16:26:28

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 0.0%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.283
- Mean expected rating: 3.165

### Expected rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 5 | 2.3% |
| 2 | 6 | 2.8% |
| 3 | 16 | 7.4% |
| 4 | 188 | 87.0% |
| 5 | 1 | 0.5% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    2.839 |    3.209 |    3.267 |    3.133 |    3.028 |    2.999 |    3.364 |    3.369 |    3.342 |    3.187 |    3.232 |    3.151 |
| fear     |    2.884 |    3.232 |    3.289 |    3.110 |    3.089 |    3.016 |    3.355 |    3.385 |    3.361 |    3.273 |    3.285 |    3.195 |
| pain     |    2.987 |    3.268 |    3.334 |    3.048 |    3.146 |    3.003 |    3.412 |    3.406 |    3.343 |    3.296 |    3.314 |    3.238 |
| pleasure |    2.951 |    3.264 |    3.401 |    2.999 |    3.020 |    3.019 |    3.394 |    3.357 |    3.315 |    3.203 |    3.183 |    3.079 |
| rage     |    2.834 |    2.916 |    3.088 |    2.887 |    2.992 |    2.951 |    2.974 |    3.170 |    2.994 |    3.026 |    3.058 |    3.033 |
| desire   |    2.903 |    3.134 |    3.249 |    3.072 |    2.989 |    3.120 |    3.207 |    3.212 |    3.236 |    3.197 |    3.252 |    3.172 |
| personal |    2.837 |    3.211 |    3.392 |    3.051 |    2.988 |    3.107 |    3.331 |    3.278 |    3.305 |    3.260 |    3.336 |    3.185 |
| consciou |    2.953 |    3.301 |    3.390 |    3.066 |    3.095 |    3.133 |    3.332 |    3.271 |    3.328 |    3.266 |    3.380 |    3.280 |
| pride    |    3.001 |    3.162 |    3.429 |    3.026 |    3.132 |    3.101 |    3.212 |    3.278 |    3.231 |    3.209 |    3.264 |    3.192 |
| embarras |    2.979 |    3.170 |    3.367 |    3.030 |    3.097 |    3.049 |    3.246 |    3.196 |    3.194 |    3.227 |    3.325 |    3.221 |
| joy      |    2.992 |    3.244 |    3.389 |    3.101 |    3.089 |    3.199 |    3.495 |    3.311 |    3.380 |    3.255 |    3.228 |    3.139 |
| self_con |    2.953 |    3.222 |    3.359 |    3.025 |    2.947 |    3.172 |    3.181 |    3.160 |    3.210 |    3.173 |    3.262 |    3.147 |
| morality |    2.984 |    3.225 |    3.369 |    3.010 |    2.976 |    3.177 |    3.254 |    3.130 |    3.262 |    3.285 |    3.349 |    3.218 |
| memory   |    2.887 |    3.170 |    3.273 |    3.039 |    2.972 |    3.103 |    3.201 |    3.222 |    3.166 |    3.147 |    3.275 |    3.172 |
| emotion_ |    2.846 |    3.124 |    3.292 |    3.059 |    2.937 |    3.109 |    3.243 |    3.150 |    3.186 |    3.142 |    3.194 |    3.027 |
| planning |    2.970 |    3.166 |    3.320 |    2.971 |    2.867 |    3.177 |    3.158 |    3.148 |    3.136 |    3.162 |    3.243 |    3.160 |
| communic |    2.889 |    3.174 |    3.366 |    3.017 |    2.978 |    3.137 |    3.227 |    3.238 |    3.236 |    3.209 |    3.246 |    3.131 |
| thought  |    2.925 |    3.161 |    3.318 |    3.011 |    2.990 |    3.142 |    3.171 |    3.221 |    3.162 |    3.146 |    3.305 |    3.219 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.05 * | 83.6% | 83.6% |
| PC2 | 1.25 * | 6.9% | 90.6% |
| PC3 | 0.78 | 4.3% | 94.9% |
| PC4 | 0.32 | 1.8% | 96.7% |
| PC5 | 0.29 | 1.6% | 98.3% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.358 | -0.906 |
| fear | E | +0.393 | -0.904 |
| pain | E | +0.377 | -0.894 |
| pleasure | E | +0.449 | -0.830 |
| rage | E | +0.435 | -0.622 |
| desire | E | +0.739 | -0.623 |
| personality | E | +0.718 | -0.687 |
| consciousness | E | +0.734 | -0.639 |
| pride | E | +0.679 | -0.609 |
| embarrassment | E | +0.731 | -0.571 |
| joy | E | +0.470 | -0.769 |
| self_control | A | +0.913 | -0.365 |
| morality | A | +0.903 | -0.347 |
| memory | A | +0.791 | -0.575 |
| emotion_recognition | A | +0.708 | -0.608 |
| planning | A | +0.945 | -0.245 |
| communication | A | +0.786 | -0.600 |
| thought | A | +0.870 | -0.432 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.148 | 1.000 | 0.06 | 0.07 |
| frog | 0.583 | 0.499 | 0.25 | 0.14 |
| robot | 1.000 | 0.413 | 0.13 | 0.22 |
| fetus | 0.190 | 0.621 | 0.17 | 0.08 |
| pvs_patient | 0.000 | 0.474 | 0.17 | 0.10 |
| god | 0.723 | 0.973 | 0.20 | 0.80 |
| dog | 0.432 | 0.053 | 0.55 | 0.35 |
| chimpanzee | 0.352 | 0.000 | 0.63 | 0.48 |
| baby | 0.476 | 0.170 | 0.71 | 0.17 |
| girl | 0.587 | 0.430 | 0.84 | 0.62 |
| adult_woman | 0.899 | 0.539 | 0.93 | 0.91 |
| adult_man | 0.638 | 0.621 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.399 (p=0.1985) | rho=+0.713 (p=0.0092) |
| Factor 2 | rho=-0.228 (p=0.4767) | rho=-0.049 (p=0.8799) |

