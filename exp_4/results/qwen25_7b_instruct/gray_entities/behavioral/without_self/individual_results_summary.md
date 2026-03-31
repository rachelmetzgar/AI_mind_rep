# Experiment 4: Individual Likert Ratings
## Qwen-2.5-7B-Instruct

**Run:** 2026-03-28 13:01:09

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 100.0%
- P(top rating) >= 0.7: 88.9%
- P(top rating) >= 0.9: 73.6%
- Mean max P: 0.918
- Mean expected rating: 2.697

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 91 | 42.1% |
| 2 | 16 | 7.4% |
| 3 | 19 | 8.8% |
| 4 | 45 | 20.8% |
| 5 | 45 | 20.8% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    1.000 |    4.848 |    1.000 |    1.000 |    1.000 |    1.000 |    5.000 |    5.000 |    5.000 |    4.999 |    4.987 |    4.995 |
| fear     |    1.000 |    3.974 |    1.000 |    1.000 |    1.000 |    1.000 |    4.546 |    4.996 |    3.183 |    4.637 |    4.008 |    4.027 |
| pain     |    1.000 |    4.011 |    1.000 |    1.000 |    1.000 |    1.000 |    4.999 |    5.000 |    4.160 |    4.995 |    4.893 |    4.961 |
| pleasure |    1.000 |    3.924 |    1.001 |    1.000 |    1.000 |    1.000 |    4.981 |    4.992 |    4.953 |    4.950 |    4.377 |    4.162 |
| rage     |    1.000 |    1.016 |    1.000 |    1.000 |    1.000 |    1.000 |    1.000 |    3.256 |    1.000 |    1.006 |    2.986 |    2.063 |
| desire   |    1.001 |    1.870 |    1.689 |    1.000 |    1.000 |    3.749 |    3.858 |    3.948 |    1.001 |    4.014 |    4.034 |    4.072 |
| personal |    3.273 |    3.164 |    3.992 |    1.000 |    1.000 |    4.985 |    4.926 |    4.955 |    3.833 |    4.777 |    4.050 |    4.051 |
| consciou |    1.000 |    3.446 |    2.335 |    1.000 |    1.000 |    4.997 |    4.320 |    4.984 |    4.986 |    4.999 |    4.798 |    4.887 |
| pride    |    1.000 |    1.777 |    1.000 |    1.000 |    1.000 |    1.005 |    2.644 |    1.078 |    1.000 |    3.946 |    4.029 |    4.136 |
| embarras |    1.000 |    1.002 |    1.000 |    1.000 |    1.000 |    1.000 |    1.206 |    1.005 |    1.000 |    3.860 |    4.160 |    4.637 |
| joy      |    1.000 |    3.808 |    2.703 |    1.000 |    1.000 |    4.975 |    4.994 |    4.972 |    4.981 |    5.000 |    4.725 |    4.341 |
| self_con |    1.070 |    2.256 |    2.288 |    1.000 |    1.000 |    4.619 |    1.555 |    1.826 |    1.000 |    1.945 |    3.686 |    3.991 |
| morality |    1.407 |    1.099 |    1.004 |    1.000 |    1.000 |    4.934 |    1.936 |    1.665 |    1.000 |    2.808 |    4.009 |    4.038 |
| memory   |    1.000 |    3.024 |    2.998 |    1.000 |    1.000 |    4.999 |    3.250 |    3.948 |    1.127 |    3.370 |    3.997 |    3.516 |
| emotion_ |    1.000 |    1.002 |    3.981 |    1.000 |    1.000 |    4.994 |    2.133 |    3.212 |    1.000 |    2.999 |    4.001 |    3.050 |
| planning |    1.000 |    1.351 |    2.012 |    1.000 |    1.000 |    4.999 |    1.000 |    2.164 |    1.000 |    2.022 |    4.000 |    4.000 |
| communic |    1.000 |    1.000 |    3.999 |    1.000 |    1.000 |    4.798 |    3.156 |    3.980 |    1.000 |    4.177 |    4.004 |    4.000 |
| thought  |    1.001 |    3.045 |    2.654 |    1.000 |    1.000 |    5.000 |    3.847 |    4.629 |    1.000 |    4.124 |    4.002 |    4.002 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 10.88 * | 60.4% | 60.4% |
| PC2 | 3.76 * | 20.9% | 81.3% |
| PC3 | 1.48 * | 8.2% | 89.6% |
| PC4 | 0.77 | 4.3% | 93.9% |
| PC5 | 0.50 | 2.8% | 96.7% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | -0.041 | -0.963 |
| fear | E | -0.109 | -0.961 |
| pain | E | -0.103 | -0.955 |
| pleasure | E | -0.031 | -0.987 |
| rage | E | -0.334 | -0.409 |
| desire | E | -0.739 | -0.528 |
| personality | E | -0.696 | -0.576 |
| consciousness | E | -0.599 | -0.734 |
| pride | E | -0.255 | -0.488 |
| embarrassment | E | -0.289 | -0.320 |
| joy | E | -0.581 | -0.766 |
| self_control | A | -0.873 | +0.028 |
| morality | A | -0.820 | -0.043 |
| memory | A | -0.913 | -0.320 |
| emotion_recognition | A | -0.957 | +0.002 |
| planning | A | -0.889 | +0.068 |
| communication | A | -0.903 | -0.208 |
| thought | A | -0.850 | -0.436 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.931 | 0.941 | 0.06 | 0.07 |
| frog | 0.862 | 0.264 | 0.25 | 0.14 |
| robot | 0.521 | 0.933 | 0.13 | 0.22 |
| fetus | 1.000 | 0.999 | 0.17 | 0.08 |
| pvs_patient | 1.000 | 0.999 | 0.17 | 0.10 |
| god | 0.000 | 1.000 | 0.20 | 0.80 |
| dog | 0.699 | 0.023 | 0.55 | 0.35 |
| chimpanzee | 0.529 | 0.000 | 0.63 | 0.48 |
| baby | 0.978 | 0.120 | 0.71 | 0.17 |
| girl | 0.630 | 0.121 | 0.84 | 0.62 |
| adult_woman | 0.468 | 0.355 | 0.93 | 0.91 |
| adult_man | 0.518 | 0.385 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.424 (p=0.1698) | rho=-0.867 (p=0.0003) |
| Factor 2 | rho=-0.564 (p=0.0562) | rho=-0.245 (p=0.4433) |

