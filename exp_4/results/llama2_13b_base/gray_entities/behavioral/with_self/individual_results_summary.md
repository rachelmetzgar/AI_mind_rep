# Experiment 4, Phase 2c: Individual Likert Ratings
## BASE MODEL (LLaMA-2-13B, no chat fine-tuning)

**Run:** 2026-02-19 16:27:16

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 234
- Entities: 13
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 0.0%
- P(top rating) >= 0.7: 0.0%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.282
- Mean expected rating: 3.137

### Expected rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 18 | 7.7% |
| 2 | 6 | 2.6% |
| 3 | 18 | 7.7% |
| 4 | 191 | 81.6% |
| 5 | 1 | 0.4% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma | you_self |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    2.839 |    3.209 |    3.267 |    3.133 |    3.028 |    2.999 |    3.364 |    3.369 |    3.342 |    3.187 |    3.232 |    3.151 |    2.815 |
| fear     |    2.884 |    3.232 |    3.289 |    3.110 |    3.089 |    3.016 |    3.355 |    3.385 |    3.361 |    3.273 |    3.285 |    3.195 |    2.591 |
| pain     |    2.987 |    3.268 |    3.334 |    3.048 |    3.146 |    3.003 |    3.412 |    3.406 |    3.343 |    3.296 |    3.314 |    3.238 |    2.729 |
| pleasure |    2.951 |    3.264 |    3.401 |    2.999 |    3.020 |    3.019 |    3.394 |    3.357 |    3.315 |    3.203 |    3.183 |    3.079 |    2.815 |
| rage     |    2.834 |    2.916 |    3.088 |    2.887 |    2.992 |    2.951 |    2.974 |    3.170 |    2.994 |    3.026 |    3.058 |    3.033 |    2.631 |
| desire   |    2.903 |    3.134 |    3.249 |    3.072 |    2.989 |    3.120 |    3.207 |    3.212 |    3.236 |    3.197 |    3.252 |    3.172 |    2.896 |
| personal |    2.837 |    3.211 |    3.392 |    3.051 |    2.988 |    3.107 |    3.331 |    3.278 |    3.305 |    3.260 |    3.336 |    3.185 |    2.855 |
| consciou |    2.953 |    3.301 |    3.390 |    3.066 |    3.095 |    3.133 |    3.332 |    3.271 |    3.328 |    3.266 |    3.380 |    3.280 |    2.684 |
| pride    |    3.001 |    3.162 |    3.429 |    3.026 |    3.132 |    3.101 |    3.212 |    3.278 |    3.231 |    3.209 |    3.264 |    3.192 |    3.018 |
| embarras |    2.979 |    3.170 |    3.367 |    3.030 |    3.097 |    3.049 |    3.246 |    3.196 |    3.194 |    3.227 |    3.325 |    3.221 |    3.007 |
| joy      |    2.992 |    3.244 |    3.389 |    3.101 |    3.089 |    3.199 |    3.495 |    3.311 |    3.380 |    3.255 |    3.228 |    3.139 |    2.916 |
| self_con |    2.953 |    3.222 |    3.359 |    3.025 |    2.947 |    3.172 |    3.181 |    3.160 |    3.210 |    3.173 |    3.262 |    3.147 |    2.841 |
| morality |    2.984 |    3.225 |    3.369 |    3.010 |    2.976 |    3.177 |    3.254 |    3.130 |    3.262 |    3.285 |    3.349 |    3.218 |    2.821 |
| memory   |    2.887 |    3.170 |    3.273 |    3.039 |    2.972 |    3.103 |    3.201 |    3.222 |    3.166 |    3.147 |    3.275 |    3.172 |    2.777 |
| emotion_ |    2.846 |    3.124 |    3.292 |    3.059 |    2.937 |    3.109 |    3.243 |    3.150 |    3.186 |    3.142 |    3.194 |    3.027 |    2.863 |
| planning |    2.970 |    3.166 |    3.320 |    2.971 |    2.867 |    3.177 |    3.158 |    3.148 |    3.136 |    3.162 |    3.243 |    3.160 |    2.782 |
| communic |    2.889 |    3.174 |    3.366 |    3.017 |    2.978 |    3.137 |    3.227 |    3.238 |    3.236 |    3.209 |    3.246 |    3.131 |    2.827 |
| thought  |    2.925 |    3.161 |    3.318 |    3.011 |    2.990 |    3.142 |    3.171 |    3.221 |    3.162 |    3.146 |    3.305 |    3.219 |    2.736 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 15.82 * | 87.9% | 87.9% |
| PC2 | 0.76 | 4.2% | 92.1% |
| PC3 | 0.53 | 3.0% | 95.0% |
| PC4 | 0.39 | 2.2% | 97.2% |
| PC5 | 0.18 | 1.0% | 98.2% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.427 | -0.882 |
| fear | E | +0.486 | -0.848 |
| pain | E | +0.477 | -0.853 |
| pleasure | E | +0.503 | -0.817 |
| rage | E | +0.539 | -0.672 |
| desire | E | +0.740 | -0.632 |
| personality | E | +0.721 | -0.669 |
| consciousness | E | +0.706 | -0.667 |
| pride | E | +0.703 | -0.561 |
| embarrassment | E | +0.748 | -0.522 |
| joy | E | +0.514 | -0.781 |
| self_control | A | +0.878 | -0.449 |
| morality | A | +0.872 | -0.441 |
| memory | A | +0.768 | -0.616 |
| emotion_recognition | A | +0.710 | -0.619 |
| planning | A | +0.902 | -0.375 |
| communication | A | +0.777 | -0.617 |
| thought | A | +0.820 | -0.525 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.161 | 0.822 | 0.06 | 0.07 |
| frog | 0.542 | 0.442 | 0.25 | 0.14 |
| robot | 1.000 | 0.468 | 0.13 | 0.22 |
| fetus | 0.161 | 0.490 | 0.17 | 0.08 |
| pvs_patient | 0.000 | 0.380 | 0.17 | 0.10 |
| god | 0.712 | 0.860 | 0.20 | 0.80 |
| dog | 0.356 | 0.029 | 0.55 | 0.35 |
| chimpanzee | 0.292 | 0.000 | 0.63 | 0.48 |
| baby | 0.419 | 0.150 | 0.71 | 0.17 |
| girl | 0.578 | 0.420 | 0.84 | 0.62 |
| adult_woman | 0.901 | 0.564 | 0.93 | 0.91 |
| adult_man | 0.638 | 0.591 | 0.91 | 0.95 |
| you_self | 0.040 | 1.000 | 0.97 | 1.00 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.151 (p=0.6217) | rho=+0.401 (p=0.1744) |
| Factor 2 | rho=+0.066 (p=0.8303) | rho=+0.319 (p=0.2886) |

