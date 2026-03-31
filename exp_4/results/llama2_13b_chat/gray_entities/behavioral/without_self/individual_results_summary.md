# Experiment 4: Individual Likert Ratings
## LLaMA-2-13B-Chat

**Run:** 2026-03-28 13:04:15

---

## Method

Instead of pairwise comparisons (Gray et al. methodology), each entity is rated individually on each mental capacity using a 1-5 Likert scale. This avoids pairwise position bias entirely.

The rating matrix (capacities x entities) is analyzed with PCA + varimax rotation, same as the pairwise version.

## Response statistics

- Total ratings: 216
- Entities: 12
- Capacities: 18

### Probability concentration

- P(top rating) >= 0.5: 98.1%
- P(top rating) >= 0.7: 52.3%
- P(top rating) >= 0.9: 0.5%
- Mean max P: 0.706
- Mean expected rating: 1.599

### Argmax rating distribution

| Rating | Count | Pct |
|-------:|------:|----:|
| 1 | 216 | 100.0% |
| 2 | 0 | 0.0% |
| 3 | 0 | 0.0% |
| 4 | 0 | 0.0% |
| 5 | 0 | 0.0% |

### Rating matrix (expected ratings)

| Capacity | dead_wom |     frog |    robot |    fetus | pvs_pati |      god |      dog | chimpanz |     baby |     girl | adult_wo | adult_ma |
|----------|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
| hunger   |    1.736 |    1.452 |    1.518 |    1.438 |    1.522 |    1.924 |    1.370 |    1.421 |    1.193 |    1.528 |    2.186 |    1.887 |
| fear     |    1.633 |    1.445 |    1.564 |    1.411 |    1.610 |    1.735 |    1.269 |    1.490 |    1.219 |    1.544 |    2.036 |    1.839 |
| pain     |    1.764 |    1.455 |    1.523 |    1.679 |    1.640 |    1.624 |    1.282 |    1.840 |    1.529 |    1.909 |    1.935 |    2.088 |
| pleasure |    1.872 |    1.654 |    1.811 |    1.817 |    1.833 |    1.722 |    1.472 |    2.151 |    1.895 |    2.053 |    2.098 |    2.078 |
| rage     |    1.805 |    1.577 |    1.692 |    1.646 |    1.768 |    1.711 |    1.570 |    2.057 |    1.715 |    1.955 |    1.945 |    1.955 |
| desire   |    1.405 |    1.355 |    1.565 |    1.435 |    1.458 |    1.636 |    1.297 |    1.357 |    1.192 |    1.340 |    1.819 |    1.705 |
| personal |    1.530 |    1.609 |    1.695 |    1.395 |    1.606 |    1.654 |    1.355 |    1.575 |    1.307 |    1.631 |    2.054 |    1.946 |
| consciou |    1.709 |    1.397 |    1.513 |    1.406 |    1.624 |    1.596 |    1.262 |    1.511 |    1.250 |    1.458 |    1.964 |    1.894 |
| pride    |    1.633 |    1.423 |    1.552 |    1.461 |    1.606 |    1.682 |    1.244 |    1.423 |    1.180 |    1.394 |    2.054 |    1.938 |
| embarras |    1.722 |    1.447 |    1.637 |    1.424 |    1.691 |    1.820 |    1.313 |    1.435 |    1.177 |    1.570 |    2.129 |    1.974 |
| joy      |    1.594 |    1.403 |    1.574 |    1.436 |    1.606 |    1.616 |    1.278 |    1.418 |    1.195 |    1.392 |    2.053 |    1.966 |
| self_con |    1.678 |    1.589 |    1.744 |    1.486 |    1.732 |    1.578 |    1.350 |    1.692 |    1.245 |    1.652 |    2.104 |    2.080 |
| morality |    1.708 |    1.443 |    1.536 |    1.530 |    1.668 |    1.602 |    1.277 |    1.686 |    1.291 |    1.488 |    1.969 |    2.029 |
| memory   |    1.518 |    1.371 |    1.531 |    1.510 |    1.516 |    1.780 |    1.246 |    1.426 |    1.190 |    1.385 |    1.790 |    1.737 |
| emotion_ |    1.641 |    1.366 |    1.631 |    1.422 |    1.520 |    1.679 |    1.231 |    1.421 |    1.198 |    1.363 |    2.011 |    1.846 |
| planning |    1.511 |    1.402 |    1.601 |    1.485 |    1.610 |    1.642 |    1.247 |    1.454 |    1.193 |    1.386 |    1.825 |    1.702 |
| communic |    1.486 |    1.446 |    1.667 |    1.480 |    1.493 |    1.652 |    1.278 |    1.464 |    1.219 |    1.390 |    1.969 |    1.889 |
| thought  |    1.837 |    1.417 |    1.585 |    1.495 |    1.696 |    1.713 |    1.243 |    1.539 |    1.201 |    1.565 |    1.865 |    2.117 |

## PCA Results

### Eigenvalues

| Component | Eigenvalue | Variance | Cumulative |
|----------:|-----------:|---------:|-----------:|
| PC1 | 14.94 * | 83.0% | 83.0% |
| PC2 | 1.97 * | 11.0% | 94.0% |
| PC3 | 0.32 | 1.8% | 95.7% |
| PC4 | 0.25 | 1.4% | 97.1% |
| PC5 | 0.21 | 1.1% | 98.2% |

### Varimax-rotated capacity loadings

| Capacity | Human factor | F1 | F2 |
|----------|:------------:|---:|---:|
| hunger | E | +0.925 | -0.202 |
| fear | E | +0.917 | -0.361 |
| pain | E | +0.428 | -0.868 |
| pleasure | E | +0.173 | -0.957 |
| rage | E | +0.193 | -0.953 |
| desire | E | +0.967 | -0.136 |
| personality | E | +0.845 | -0.401 |
| consciousness | E | +0.869 | -0.453 |
| pride | E | +0.949 | -0.299 |
| embarrassment | E | +0.944 | -0.284 |
| joy | E | +0.936 | -0.317 |
| self_control | A | +0.808 | -0.509 |
| morality | A | +0.790 | -0.566 |
| memory | A | +0.935 | -0.182 |
| emotion_recognition | A | +0.949 | -0.273 |
| planning | A | +0.940 | -0.232 |
| communication | A | +0.936 | -0.264 |
| thought | A | +0.817 | -0.455 |

### Entity positions (0-1 scale)

| Entity | Model F1 | Model F2 | Human Exp | Human Ag |
|--------|--------:|--------:|----------:|---------:|
| dead_woman | 0.527 | 0.444 | 0.06 | 0.07 |
| frog | 0.412 | 0.835 | 0.25 | 0.14 |
| robot | 0.598 | 0.725 | 0.13 | 0.22 |
| fetus | 0.387 | 0.648 | 0.17 | 0.08 |
| pvs_patient | 0.535 | 0.549 | 0.17 | 0.10 |
| god | 0.763 | 0.822 | 0.20 | 0.80 |
| dog | 0.257 | 1.000 | 0.55 | 0.35 |
| chimpanzee | 0.203 | 0.000 | 0.63 | 0.48 |
| baby | 0.000 | 0.494 | 0.71 | 0.17 |
| girl | 0.223 | 0.123 | 0.84 | 0.62 |
| adult_woman | 1.000 | 0.326 | 0.93 | 0.91 |
| adult_man | 0.820 | 0.155 | 0.91 | 0.95 |

## Alignment with human Experience/Agency

| | Human Experience | Human Agency |
|---|---:|---:|
| Factor 1 | rho=-0.018 (p=0.9569) | rho=+0.364 (p=0.2453) |
| Factor 2 | rho=-0.480 (p=0.1144) | rho=-0.329 (p=0.2969) |

