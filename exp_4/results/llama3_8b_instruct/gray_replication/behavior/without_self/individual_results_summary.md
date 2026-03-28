# Experiment 4: Individual Likert Ratings
## LLaMA-3-8B-Instruct

**Run:** 2026-03-28 09:52:51

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 89.8%
- P(top rating) >= 0.7: 58.3%
- P(top rating) >= 0.9: 26.9%
- Mean max P: 0.745
- Mean expected rating: 2.506

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 72 | 33.3% |
| 2 | 41 | 19.0% |
| 3 | 49 | 22.7% |
| 4 | 39 | 18.1% |
| 5 | 15 | 6.9% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    1.137 |    3.660 |    1.470 |    1.098 |    1.009 |    1.122 |    4.705 |    4.167 |    4.734 |    4.724 |    3.893 |    3.761 |
| fear     |    2.809 |    3.140 |    2.328 |    1.001 |    1.000 |    1.754 |    3.158 |    3.248 |    1.582 |    3.922 |    3.500 |    2.861 |
| pain     |    1.668 |    3.029 |    1.253 |    1.248 |    1.090 |    3.892 |    3.072 |    3.751 |    1.346 |    3.447 |    3.573 |    2.999 |
| pleasure |    1.977 |    3.304 |    1.925 |    1.040 |    1.002 |    4.182 |    4.428 |    3.891 |    2.775 |    3.814 |    3.935 |    3.100 |
| rage     |    1.791 |    1.485 |    1.077 |    1.000 |    1.000 |    1.869 |    1.134 |    2.471 |    1.000 |    1.227 |    2.593 |    1.897 |
| desire   |    3.035 |    2.796 |    3.205 |    1.004 |    1.001 |    4.612 |    3.124 |    2.206 |    1.265 |    4.022 |    3.826 |    2.963 |
| personal |    3.683 |    3.015 |    3.891 |    1.006 |    1.001 |    4.343 |    3.278 |    2.787 |    1.103 |    3.224 |    3.903 |    2.508 |
| consciou |    1.870 |    2.902 |    3.038 |    1.181 |    1.008 |    4.573 |    2.717 |    3.155 |    1.241 |    3.510 |    3.927 |    3.454 |
| pride    |    2.498 |    2.128 |    2.242 |    1.000 |    1.000 |    2.380 |    1.925 |    1.999 |    1.002 |    2.449 |    3.859 |    2.879 |
| embarras |    1.628 |    2.122 |    2.138 |    1.000 |    1.000 |    1.946 |    1.581 |    1.364 |    1.000 |    2.717 |    3.735 |    3.320 |
| joy      |    3.264 |    3.630 |    3.163 |    1.021 |    1.000 |    4.763 |    4.912 |    4.122 |    4.274 |    4.867 |    3.859 |    2.541 |
| self_con |    2.608 |    1.761 |    2.331 |    1.000 |    1.000 |    4.766 |    1.466 |    1.405 |    1.001 |    1.835 |    3.287 |    2.660 |
| morality |    3.591 |    2.674 |    2.971 |    1.000 |    1.000 |    4.831 |    1.967 |    1.816 |    1.002 |    3.196 |    3.976 |    3.560 |
| memory   |    2.879 |    2.501 |    2.656 |    1.002 |    1.000 |    4.900 |    2.203 |    2.430 |    1.017 |    2.631 |    3.572 |    3.132 |
| emotion_ |    2.820 |    1.977 |    3.734 |    1.001 |    1.000 |    4.727 |    2.316 |    2.275 |    1.020 |    2.517 |    3.830 |    2.345 |
| planning |    2.922 |    2.225 |    2.300 |    1.000 |    1.000 |    4.870 |    1.383 |    1.397 |    1.001 |    1.940 |    3.946 |    3.477 |
| communic |    1.876 |    1.807 |    3.927 |    1.000 |    1.000 |    4.712 |    2.304 |    2.045 |    1.024 |    2.752 |    3.940 |    2.874 |
| thought  |    2.728 |    2.199 |    2.505 |    1.006 |    1.000 |    4.817 |    1.940 |    2.133 |    1.018 |    2.237 |    3.774 |    3.177 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 12.25 * | 68.1% | 68.1% |
| PC2 | 2.87 * | 15.9% | 84.0% |
| PC3 | 1.12 * | 6.2% | 90.2% |
| PC4 | 0.75 | 4.2% | 94.4% |
| PC5 | 0.45 | 2.5% | 96.9% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.349 | -0.845 |
| fear | E | -0.157 | -0.714 |
| pain | E | -0.440 | -0.696 |
| pleasure | E | -0.366 | -0.882 |
| rage | E | -0.412 | -0.212 |
| desire | E | -0.831 | -0.450 |
| personality | E | -0.830 | -0.370 |
| consciousness | E | -0.758 | -0.485 |
| pride | E | -0.616 | -0.260 |
| embarrassment | E | -0.460 | -0.220 |
| joy | E | -0.362 | -0.891 |
| self_control | A | -0.967 | -0.013 |
| morality | A | -0.904 | -0.142 |
| memory | A | -0.927 | -0.243 |
| emotion_recognition | A | -0.948 | -0.181 |
| planning | A | -0.905 | +0.007 |
| communication | A | -0.901 | -0.205 |
| thought | A | -0.930 | -0.153 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.540 | 0.767 | 0.06 | 0.07 |
| frog | 0.757 | 0.346 | 0.25 | 0.14 |
| robot | 0.453 | 0.712 | 0.13 | 0.22 |
| fetus | 0.924 | 0.979 | 0.17 | 0.08 |
| pvs_patient | 0.924 | 1.000 | 0.17 | 0.10 |
| god | 0.000 | 0.479 | 0.20 | 0.80 |
| dog | 0.771 | 0.000 | 0.55 | 0.35 |
| chimpanzee | 0.838 | 0.174 | 0.63 | 0.48 |
| baby | 1.000 | 0.342 | 0.71 | 0.17 |
| girl | 0.709 | 0.067 | 0.84 | 0.62 |
| adult_woman | 0.479 | 0.452 | 0.93 | 0.91 |
| adult_man | 0.642 | 0.624 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=+0.060 (p=0.8542) | rho=-0.462 (p=0.1309) |
| Factor 2 | rho=-0.606 (p=0.0368) | rho=-0.448 (p=0.1446) |

