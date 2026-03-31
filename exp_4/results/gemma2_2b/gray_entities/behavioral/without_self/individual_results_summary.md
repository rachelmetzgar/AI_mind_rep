# Experiment 4: Individual Likert Ratings
## Gemma-2-2B (Base)

**Run:** 2026-03-29 11:33:46

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
- Mean max P: 0.278
- Mean expected rating: 3.084

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 21 | 9.7% |
| 2 | 2 | 0.9% |
| 3 | 49 | 22.7% |
| 4 | 141 | 65.3% |
| 5 | 3 | 1.4% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    2.958 |    3.031 |    3.050 |    3.283 |    2.847 |    2.807 |    3.333 |    3.145 |    3.171 |    3.074 |    3.196 |    3.136 |
| fear     |    2.976 |    2.968 |    3.077 |    3.292 |    2.893 |    2.652 |    3.425 |    3.230 |    3.148 |    3.136 |    3.242 |    3.229 |
| pain     |    2.939 |    2.936 |    3.032 |    3.363 |    2.883 |    2.612 |    3.210 |    3.068 |    3.070 |    3.099 |    3.137 |    3.154 |
| pleasure |    2.989 |    2.911 |    3.064 |    3.300 |    2.830 |    2.650 |    3.323 |    3.071 |    3.019 |    3.074 |    3.201 |    3.175 |
| rage     |    2.955 |    2.880 |    2.989 |    3.215 |    2.904 |    2.617 |    3.050 |    3.167 |    3.092 |    3.086 |    3.093 |    3.166 |
| desire   |    2.712 |    2.738 |    3.066 |    3.265 |    2.890 |    2.496 |    3.395 |    3.182 |    3.095 |    3.055 |    3.208 |    3.141 |
| personal |    2.896 |    3.004 |    3.142 |    3.356 |    2.942 |    2.693 |    3.535 |    3.317 |    3.312 |    3.244 |    3.295 |    3.302 |
| consciou |    2.875 |    3.003 |    3.144 |    3.360 |    2.897 |    2.615 |    3.402 |    3.155 |    3.190 |    3.145 |    3.258 |    3.245 |
| pride    |    2.983 |    2.928 |    3.011 |    3.175 |    2.911 |    2.616 |    3.270 |    3.195 |    3.090 |    3.029 |    3.219 |    3.092 |
| embarras |    2.951 |    2.967 |    3.028 |    3.155 |    2.827 |    2.672 |    3.172 |    3.106 |    3.036 |    3.081 |    3.207 |    3.173 |
| joy      |    3.022 |    3.008 |    3.109 |    3.274 |    2.897 |    2.691 |    3.459 |    3.200 |    3.178 |    3.137 |    3.239 |    3.168 |
| self_con |    2.932 |    2.767 |    3.072 |    3.228 |    2.854 |    2.606 |    3.331 |    3.115 |    3.124 |    3.068 |    3.251 |    3.190 |
| morality |    2.913 |    2.922 |    3.117 |    3.337 |    2.932 |    2.643 |    3.405 |    3.207 |    3.196 |    3.122 |    3.303 |    3.219 |
| memory   |    2.987 |    3.026 |    3.088 |    3.168 |    2.844 |    2.831 |    3.425 |    3.234 |    3.120 |    3.093 |    3.223 |    3.203 |
| emotion_ |    2.958 |    3.008 |    3.187 |    3.164 |    2.867 |    2.764 |    3.458 |    3.248 |    3.161 |    3.141 |    3.249 |    3.209 |
| planning |    2.936 |    2.814 |    3.068 |    3.099 |    2.877 |    2.604 |    3.418 |    3.276 |    3.205 |    3.179 |    3.382 |    3.292 |
| communic |    2.977 |    3.023 |    3.163 |    3.281 |    2.875 |    2.743 |    3.455 |    3.230 |    3.193 |    3.188 |    3.299 |    3.263 |
| thought  |    2.932 |    3.037 |    3.121 |    3.184 |    2.851 |    2.800 |    3.431 |    3.243 |    3.175 |    3.113 |    3.262 |    3.225 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 16.74 * | 93.0% | 93.0% |
| PC2 | 0.52 | 2.9% | 95.9% |
| PC3 | 0.26 | 1.4% | 97.4% |
| PC4 | 0.16 | 0.9% | 98.3% |
| PC5 | 0.11 | 0.6% | 98.9% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.755 | +0.597 |
| fear | E | +0.733 | +0.675 |
| pain | E | +0.485 | +0.857 |
| pleasure | E | +0.646 | +0.723 |
| rage | E | +0.390 | +0.888 |
| desire | E | +0.691 | +0.675 |
| personality | E | +0.757 | +0.625 |
| consciousness | E | +0.679 | +0.712 |
| pride | E | +0.677 | +0.694 |
| embarrassment | E | +0.650 | +0.717 |
| joy | E | +0.771 | +0.614 |
| self_control | A | +0.681 | +0.701 |
| morality | A | +0.692 | +0.711 |
| memory | A | +0.895 | +0.424 |
| emotion_recognition | A | +0.878 | +0.462 |
| planning | A | +0.759 | +0.565 |
| communication | A | +0.799 | +0.596 |
| thought | A | +0.882 | +0.460 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.147 | 0.531 | 0.06 | 0.07 |
| frog | 0.329 | 0.359 | 0.25 | 0.14 |
| robot | 0.428 | 0.504 | 0.13 | 0.22 |
| fetus | 0.235 | 1.000 | 0.17 | 0.08 |
| pvs_patient | 0.000 | 0.555 | 0.17 | 0.10 |
| god | 0.193 | 0.000 | 0.20 | 0.80 |
| dog | 1.000 | 0.428 | 0.55 | 0.35 |
| chimpanzee | 0.550 | 0.570 | 0.63 | 0.48 |
| baby | 0.449 | 0.586 | 0.71 | 0.17 |
| girl | 0.341 | 0.645 | 0.84 | 0.62 |
| adult_woman | 0.577 | 0.630 | 0.93 | 0.91 |
| adult_man | 0.438 | 0.702 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.676 (p=0.0158) | rho=+0.545 (p=0.0666) |
| Factor 2 | rho=+0.448 (p=0.1438) | rho=+0.140 (p=0.6646) |

