# Experiment 4: Individual Likert Ratings
## Qwen3-8B

**Run:** 2026-03-28 15:28:53

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 32.4%
- P(top rating) >= 0.7: 1.4%
- P(top rating) >= 0.9: 0.0%
- Mean max P: 0.481
- Mean expected rating: 1.871

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 137 | 63.4% |
| 2 | 79 | 36.6% |
| 3 | 0 | 0.0% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    1.945 |    2.024 |    1.718 |    1.795 |    1.656 |    1.875 |    1.965 |    1.719 |    1.716 |    2.053 |    1.916 |    1.857 |
| fear     |    2.009 |    1.967 |    1.510 |    1.816 |    1.901 |    2.072 |    1.969 |    2.011 |    1.627 |    1.972 |    2.032 |    2.003 |
| pain     |    1.807 |    1.843 |    1.486 |    1.750 |    1.883 |    2.079 |    1.897 |    1.998 |    1.491 |    1.848 |    1.978 |    2.081 |
| pleasure |    1.868 |    1.856 |    1.487 |    1.706 |    1.952 |    2.052 |    1.903 |    2.144 |    1.468 |    1.918 |    2.165 |    2.193 |
| rage     |    2.033 |    1.895 |    1.518 |    1.808 |    1.851 |    2.198 |    1.955 |    1.874 |    1.616 |    1.877 |    2.006 |    2.140 |
| desire   |    2.028 |    2.000 |    1.564 |    1.732 |    1.853 |    2.113 |    2.099 |    2.122 |    1.647 |    1.975 |    2.121 |    2.151 |
| personal |    1.495 |    1.428 |    1.391 |    1.759 |    1.905 |    2.072 |    1.970 |    1.876 |    1.810 |    1.931 |    1.973 |    1.918 |
| consciou |    1.729 |    1.705 |    1.414 |    1.691 |    1.932 |    1.977 |    1.802 |    1.752 |    1.545 |    1.803 |    1.758 |    1.941 |
| pride    |    2.049 |    2.042 |    1.738 |    1.963 |    1.743 |    1.891 |    2.068 |    1.820 |    1.903 |    2.069 |    1.915 |    1.819 |
| embarras |    2.021 |    2.100 |    1.791 |    1.968 |    1.828 |    1.863 |    2.047 |    1.839 |    1.879 |    2.011 |    1.841 |    1.816 |
| joy      |    1.991 |    1.976 |    1.649 |    1.813 |    1.709 |    1.925 |    2.015 |    1.781 |    1.704 |    2.067 |    1.896 |    1.757 |
| self_con |    1.530 |    1.432 |    1.460 |    1.970 |    1.684 |    2.349 |    1.999 |    2.031 |    1.814 |    1.987 |    1.944 |    1.892 |
| morality |    1.524 |    1.490 |    1.535 |    1.873 |    1.768 |    2.304 |    2.024 |    1.964 |    1.819 |    1.984 |    1.845 |    1.819 |
| memory   |    1.974 |    2.019 |    1.570 |    1.874 |    1.787 |    1.899 |    2.081 |    1.756 |    1.849 |    2.125 |    2.015 |    2.012 |
| emotion_ |    2.083 |    1.999 |    1.531 |    1.698 |    1.899 |    2.279 |    2.012 |    1.996 |    1.609 |    2.048 |    2.253 |    2.259 |
| planning |    1.815 |    1.735 |    1.397 |    1.741 |    1.916 |    2.023 |    2.017 |    1.788 |    1.707 |    2.096 |    1.847 |    2.177 |
| communic |    1.792 |    1.799 |    1.395 |    1.755 |    1.826 |    2.206 |    2.062 |    1.967 |    1.650 |    2.017 |    1.957 |    2.182 |
| thought  |    1.710 |    1.798 |    1.643 |    1.909 |    1.734 |    1.953 |    1.950 |    1.705 |    1.816 |    1.863 |    1.775 |    1.761 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 10.51 * | 58.4% | 58.4% |
| PC2 | 3.88 * | 21.6% | 79.9% |
| PC3 | 2.15 * | 12.0% | 91.9% |
| PC4 | 0.49 | 2.7% | 94.6% |
| PC5 | 0.36 | 2.0% | 96.7% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | -0.333 | -0.888 |
| fear | E | -0.898 | -0.333 |
| pain | E | -0.942 | -0.034 |
| pleasure | E | -0.965 | +0.046 |
| rage | E | -0.887 | -0.237 |
| desire | E | -0.920 | -0.257 |
| personality | E | -0.439 | +0.123 |
| consciousness | E | -0.820 | -0.040 |
| pride | E | -0.023 | -0.982 |
| embarrassment | E | +0.062 | -0.945 |
| joy | E | -0.351 | -0.890 |
| self_control | A | -0.319 | +0.039 |
| morality | A | -0.263 | +0.035 |
| memory | A | -0.463 | -0.782 |
| emotion_recognition | A | -0.938 | -0.218 |
| planning | A | -0.708 | -0.259 |
| communication | A | -0.807 | -0.197 |
| thought | A | -0.005 | -0.533 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.323 | 0.167 | 0.06 | 0.07 |
| frog | 0.429 | 0.000 | 0.25 | 0.14 |
| robot | 0.964 | 0.958 | 0.13 | 0.22 |
| fetus | 0.795 | 0.412 | 0.17 | 0.08 |
| pvs_patient | 0.361 | 1.000 | 0.17 | 0.10 |
| god | 0.225 | 0.584 | 0.20 | 0.80 |
| dog | 0.487 | 0.049 | 0.55 | 0.35 |
| chimpanzee | 0.254 | 0.927 | 0.63 | 0.48 |
| baby | 1.000 | 0.613 | 0.71 | 0.17 |
| girl | 0.473 | 0.017 | 0.84 | 0.62 |
| adult_woman | 0.187 | 0.562 | 0.93 | 0.91 |
| adult_man | 0.000 | 0.785 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.343 (p=0.2747) | rho=-0.552 (p=0.0625) |
| Factor 2 | rho=-0.116 (p=0.7206) | rho=+0.112 (p=0.7292) |

