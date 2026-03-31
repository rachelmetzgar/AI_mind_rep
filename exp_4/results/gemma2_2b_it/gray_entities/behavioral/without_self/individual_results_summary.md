# Experiment 4: Individual Likert Ratings
## Gemma-2-2B-IT

**Run:** 2026-03-29 12:10:09

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 99.5%
- P(top rating) >= 0.7: 87.5%
- P(top rating) >= 0.9: 70.8%
- Mean max P: 0.907
- Mean expected rating: 2.442

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 71 | 32.9% |
| 2 | 35 | 16.2% |
| 3 | 64 | 29.6% |
| 4 | 35 | 16.2% |
| 5 | 11 | 5.1% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    1.895 |    3.688 |    1.001 |    2.013 |    1.000 |    1.003 |    4.000 |    4.000 |    4.034 |    3.999 |    3.401 |    3.531 |
| fear     |    2.796 |    2.984 |    1.261 |    1.000 |    1.000 |    1.040 |    3.969 |    3.998 |    1.006 |    3.625 |    3.003 |    2.993 |
| pain     |    1.130 |    1.982 |    1.001 |    1.005 |    1.001 |    1.003 |    3.894 |    3.997 |    1.001 |    1.623 |    3.093 |    3.061 |
| pleasure |    2.942 |    2.000 |    1.557 |    1.000 |    1.000 |    1.039 |    4.001 |    3.980 |    3.141 |    3.987 |    3.106 |    3.004 |
| rage     |    1.017 |    1.032 |    1.001 |    1.000 |    1.000 |    1.102 |    1.000 |    1.189 |    1.000 |    1.015 |    2.986 |    3.000 |
| desire   |    3.686 |    2.001 |    1.981 |    1.000 |    1.000 |    4.726 |    2.034 |    2.000 |    1.000 |    3.901 |    3.385 |    3.005 |
| personal |    3.994 |    2.568 |    3.956 |    1.000 |    1.000 |    4.296 |    3.933 |    3.740 |    2.064 |    3.917 |    3.334 |    3.027 |
| consciou |    3.212 |    2.842 |    2.986 |    1.001 |    1.000 |    4.942 |    3.858 |    3.934 |    3.423 |    3.944 |    3.328 |    3.058 |
| pride    |    3.056 |    1.993 |    1.435 |    1.000 |    1.000 |    4.659 |    1.987 |    1.701 |    1.000 |    2.019 |    3.007 |    2.987 |
| embarras |    2.822 |    1.062 |    1.826 |    1.000 |    1.000 |    1.007 |    1.050 |    1.002 |    1.000 |    2.017 |    3.290 |    3.178 |
| joy      |    3.145 |    2.000 |    1.997 |    1.000 |    1.000 |    4.793 |    4.002 |    3.514 |    2.502 |    3.999 |    3.004 |    2.983 |
| self_con |    3.065 |    2.000 |    2.118 |    1.000 |    1.000 |    4.944 |    2.001 |    1.014 |    1.000 |    1.991 |    3.016 |    2.965 |
| morality |    3.531 |    2.000 |    2.705 |    1.000 |    1.000 |    4.685 |    3.129 |    2.828 |    1.001 |    3.746 |    3.007 |    2.998 |
| memory   |    3.050 |    2.027 |    3.092 |    1.000 |    1.000 |    4.997 |    4.002 |    3.936 |    1.267 |    3.130 |    3.005 |    3.002 |
| emotion_ |    3.030 |    1.989 |    3.193 |    1.000 |    1.000 |    4.918 |    2.830 |    2.044 |    1.000 |    2.882 |    3.003 |    2.993 |
| planning |    2.973 |    2.000 |    2.000 |    1.000 |    1.000 |    4.941 |    1.963 |    1.728 |    1.000 |    2.000 |    3.058 |    2.999 |
| communic |    3.216 |    1.995 |    3.877 |    1.000 |    1.000 |    4.920 |    2.345 |    2.045 |    1.000 |    3.024 |    3.045 |    3.005 |
| thought  |    3.095 |    2.006 |    2.761 |    1.000 |    1.000 |    4.987 |    3.572 |    3.461 |    1.000 |    3.025 |    3.003 |    3.001 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 10.59 * | 58.8% | 58.8% |
| PC2 | 3.75 * | 20.9% | 79.7% |
| PC3 | 1.88 * | 10.5% | 90.1% |
| PC4 | 0.65 | 3.6% | 93.8% |
| PC5 | 0.56 | 3.1% | 96.9% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.169 | -0.898 |
| fear | E | -0.216 | -0.888 |
| pain | E | -0.097 | -0.840 |
| pleasure | E | -0.109 | -0.943 |
| rage | E | -0.098 | -0.209 |
| desire | E | -0.897 | -0.030 |
| personality | E | -0.867 | -0.364 |
| consciousness | E | -0.815 | -0.430 |
| pride | E | -0.885 | +0.089 |
| embarrassment | E | -0.245 | -0.132 |
| joy | E | -0.844 | -0.447 |
| self_control | A | -0.888 | +0.247 |
| morality | A | -0.961 | -0.185 |
| memory | A | -0.912 | -0.309 |
| emotion_recognition | A | -0.975 | +0.050 |
| planning | A | -0.915 | +0.157 |
| communication | A | -0.938 | +0.142 |
| thought | A | -0.950 | -0.231 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.454 | 0.642 | 0.06 | 0.07 |
| frog | 0.748 | 0.508 | 0.25 | 0.14 |
| robot | 0.558 | 0.858 | 0.13 | 0.22 |
| fetus | 1.000 | 0.878 | 0.17 | 0.08 |
| pvs_patient | 0.989 | 0.940 | 0.17 | 0.10 |
| god | 0.000 | 1.000 | 0.20 | 0.80 |
| dog | 0.551 | 0.048 | 0.55 | 0.35 |
| chimpanzee | 0.636 | 0.000 | 0.63 | 0.48 |
| baby | 0.920 | 0.480 | 0.71 | 0.17 |
| girl | 0.509 | 0.245 | 0.84 | 0.62 |
| adult_woman | 0.576 | 0.393 | 0.93 | 0.91 |
| adult_man | 0.600 | 0.401 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.039 (p=0.9054) | rho=-0.371 (p=0.2356) |
| Factor 2 | rho=-0.676 (p=0.0158) | rho=-0.427 (p=0.1667) |

