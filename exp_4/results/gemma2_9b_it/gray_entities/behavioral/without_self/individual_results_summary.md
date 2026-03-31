# Experiment 4: Individual Likert Ratings
## Gemma-2-9B-IT

**Run:** 2026-03-28 17:05:55

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 99.1%
- P(top rating) >= 0.7: 92.1%
- P(top rating) >= 0.9: 84.3%
- Mean max P: 0.945
- Mean expected rating: 2.279

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 79 | 36.6% |
| 2 | 37 | 17.1% |
| 3 | 74 | 34.3% |
| 4 | 15 | 6.9% |
| 5 | 11 | 5.1% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    1.000 |    3.985 |    1.000 |    1.000 |    1.001 |    1.003 |    4.752 |    4.917 |    4.956 |    4.992 |    4.281 |    4.103 |
| fear     |    1.000 |    2.450 |    1.000 |    1.000 |    1.000 |    1.134 |    4.004 |    4.014 |    2.003 |    4.052 |    3.051 |    3.003 |
| pain     |    1.000 |    2.077 |    1.000 |    1.001 |    1.001 |    1.627 |    3.868 |    4.021 |    2.986 |    4.346 |    3.006 |    3.007 |
| pleasure |    1.000 |    2.009 |    1.000 |    1.000 |    1.000 |    2.866 |    4.009 |    4.000 |    2.998 |    4.614 |    3.007 |    3.000 |
| rage     |    1.001 |    1.002 |    1.000 |    1.000 |    1.000 |    2.006 |    1.955 |    3.980 |    1.000 |    2.026 |    2.997 |    2.906 |
| desire   |    1.002 |    1.993 |    1.023 |    1.001 |    1.000 |    3.006 |    3.000 |    2.909 |    1.975 |    3.887 |    3.037 |    3.000 |
| personal |    1.109 |    1.996 |    2.176 |    1.022 |    1.000 |    3.061 |    3.659 |    3.182 |    1.984 |    3.141 |    3.271 |    3.000 |
| consciou |    1.000 |    2.702 |    2.024 |    1.013 |    1.000 |    3.974 |    3.024 |    3.073 |    2.000 |    3.022 |    3.028 |    3.001 |
| pride    |    1.001 |    1.001 |    1.001 |    1.000 |    1.000 |    1.942 |    2.004 |    2.036 |    1.000 |    2.978 |    3.010 |    3.001 |
| embarras |    1.000 |    1.000 |    1.000 |    1.000 |    1.000 |    1.022 |    2.000 |    2.006 |    1.000 |    2.995 |    3.014 |    3.009 |
| joy      |    1.001 |    2.006 |    1.537 |    1.000 |    1.000 |    3.235 |    4.803 |    3.980 |    2.966 |    4.991 |    3.005 |    3.000 |
| self_con |    1.024 |    1.967 |    1.005 |    1.000 |    1.000 |    3.649 |    2.001 |    1.997 |    1.000 |    2.000 |    3.000 |    3.000 |
| morality |    2.735 |    1.001 |    1.000 |    1.000 |    1.000 |    3.773 |    2.003 |    1.999 |    1.000 |    2.920 |    3.000 |    3.000 |
| memory   |    1.001 |    2.593 |    2.307 |    1.000 |    1.000 |    4.993 |    3.000 |    3.000 |    1.997 |    2.997 |    3.000 |    3.000 |
| emotion_ |    2.615 |    1.990 |    2.869 |    1.000 |    1.000 |    3.629 |    3.000 |    2.993 |    1.986 |    2.991 |    3.000 |    2.999 |
| planning |    1.001 |    2.000 |    2.026 |    1.000 |    1.000 |    4.210 |    2.007 |    2.743 |    1.000 |    2.012 |    3.003 |    3.000 |
| communic |    1.000 |    1.995 |    2.992 |    1.000 |    1.000 |    3.418 |    3.003 |    3.000 |    1.999 |    2.999 |    3.016 |    3.000 |
| thought  |    1.000 |    2.001 |    2.038 |    1.000 |    1.000 |    4.795 |    2.987 |    2.999 |    1.060 |    2.999 |    3.001 |    3.000 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 12.71 * | 70.6% | 70.6% |
| PC2 | 2.80 * | 15.5% | 86.2% |
| PC3 | 1.05 * | 5.9% | 92.0% |
| PC4 | 0.48 | 2.7% | 94.7% |
| PC5 | 0.40 | 2.2% | 96.9% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | -0.049 | -0.948 |
| fear | E | -0.161 | -0.926 |
| pain | E | -0.195 | -0.954 |
| pleasure | E | -0.407 | -0.866 |
| rage | E | -0.412 | -0.492 |
| desire | E | -0.569 | -0.698 |
| personality | E | -0.687 | -0.654 |
| consciousness | E | -0.864 | -0.462 |
| pride | E | -0.417 | -0.501 |
| embarrassment | E | -0.166 | -0.598 |
| joy | E | -0.463 | -0.841 |
| self_control | A | -0.816 | -0.132 |
| morality | A | -0.627 | -0.027 |
| memory | A | -0.953 | -0.268 |
| emotion_recognition | A | -0.799 | -0.269 |
| planning | A | -0.925 | -0.065 |
| communication | A | -0.815 | -0.457 |
| thought | A | -0.927 | -0.234 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.917 | 1.000 | 0.06 | 0.07 |
| frog | 0.671 | 0.498 | 0.25 | 0.14 |
| robot | 0.582 | 0.855 | 0.13 | 0.22 |
| fetus | 0.999 | 0.868 | 0.17 | 0.08 |
| pvs_patient | 1.000 | 0.869 | 0.17 | 0.10 |
| god | 0.000 | 0.985 | 0.20 | 0.80 |
| dog | 0.599 | 0.028 | 0.55 | 0.35 |
| chimpanzee | 0.614 | 0.098 | 0.63 | 0.48 |
| baby | 0.863 | 0.229 | 0.71 | 0.17 |
| girl | 0.698 | 0.000 | 0.84 | 0.62 |
| adult_woman | 0.599 | 0.464 | 0.93 | 0.91 |
| adult_man | 0.603 | 0.487 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.228 (p=0.4767) | rho=-0.692 (p=0.0126) |
| Factor 2 | rho=-0.739 (p=0.0060) | rho=-0.469 (p=0.1245) |

