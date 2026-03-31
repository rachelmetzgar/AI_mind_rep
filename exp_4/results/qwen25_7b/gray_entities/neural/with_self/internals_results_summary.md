# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:54:43
**Tag:** `with_self`
**Include self:** True
**Model:** Qwen-2.5-7B (Base)

---

## What is being tested

Does Qwen-2.5-7B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Qwen-2.5-7B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

- **Human RDMs**: Three variants from Gray et al. (2007) factor scores:
  - Combined: Euclidean distance in 2D (Experience, Agency) space
  - Experience-only: |exp_i - exp_j|
  - Agency-only: |agency_i - agency_j|

**Test:** Spearman rank correlation between the upper triangles of the model RDM and each human RDM (78 unique entity pairs). Computed at every layer to track where in the network mind-perception structure emerges.

**Interpretation:** A positive Spearman rho means entities that are far apart in human mind-perception space are also far apart in the model's activation space (and vice versa). The model's entity geometry mirrors human folk psychology.

## Entities

| Entity | Prompt | Experience | Agency |
|--------|--------|------------|--------|
| dead_woman | Think about a dead woman. | 0.06 | 0.07 |
| frog | Think about a frog. | 0.25 | 0.14 |
| robot | Think about a robot. | 0.13 | 0.22 |
| fetus | Think about a seven-week-old human fetus. | 0.17 | 0.08 |
| pvs_patient | Think about a person in a persistent vegetative state. | 0.17 | 0.10 |
| god | Think about God. | 0.20 | 0.80 |
| dog | Think about a dog. | 0.55 | 0.35 |
| chimpanzee | Think about a chimpanzee. | 0.63 | 0.48 |
| baby | Think about a five-month-old baby. | 0.71 | 0.17 |
| girl | Think about a five-year-old girl. | 0.84 | 0.62 |
| adult_woman | Think about an adult woman. | 0.93 | 0.91 |
| adult_man | Think about an adult man. | 0.91 | 0.95 |
| you_self | Think about yourself. | 0.97 | 1.00 |

## RSA by Layer -- Combined

**Peak:** Layer 19, rho = +0.3012, p = 0.0074

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1605 |  0.1604 |
|     2 |      +0.0453 |  0.6937 |
|     3 |      +0.0099 |  0.9318 |
|     4 |      +0.1090 |  0.3422 |
|     5 |      +0.1591 |  0.1642 |
|     6 |      +0.1255 |  0.2736 |
|     7 |      +0.1603 |  0.1610 |
|     8 |      +0.2417 |  0.0330 |
|     9 |      +0.2277 |  0.0450 |
|    10 |      +0.2308 |  0.0420 |
|    11 |      +0.2312 |  0.0417 |
|    12 |      +0.2570 |  0.0231 |
|    13 |      +0.2499 |  0.0274 |
|    14 |      +0.2310 |  0.0418 |
|    15 |      +0.2677 |  0.0178 |
|    16 |      +0.2436 |  0.0316 |
|    17 |      +0.2659 |  0.0186 |
|    18 |      +0.2648 |  0.0191 |
|    19 |      +0.3012 |  0.0074 |
|    20 |      +0.2584 |  0.0224 |
|    21 |      +0.2107 |  0.0641 |
|    22 |      +0.1751 |  0.1252 |
|    23 |      +0.1194 |  0.2979 |
|    24 |      +0.0864 |  0.4522 |
|    25 |      +0.0836 |  0.4667 |
|    26 |      +0.0779 |  0.4976 |
|    27 |      +0.0877 |  0.4453 |
|    28 |      +0.2061 |  0.0702 |

## RSA by Layer -- Experience

**Peak:** Layer 19, rho = +0.1255, p = 0.2736

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0591 |  0.6071 |
|     2 |      -0.0694 |  0.5457 |
|     3 |      -0.0801 |  0.4857 |
|     4 |      -0.0207 |  0.8576 |
|     5 |      +0.0057 |  0.9607 |
|     6 |      -0.0165 |  0.8860 |
|     7 |      +0.0175 |  0.8790 |
|     8 |      +0.0937 |  0.4143 |
|     9 |      +0.0862 |  0.4532 |
|    10 |      +0.0919 |  0.4236 |
|    11 |      +0.0757 |  0.5098 |
|    12 |      +0.0925 |  0.4203 |
|    13 |      +0.0871 |  0.4481 |
|    14 |      +0.0742 |  0.5186 |
|    15 |      +0.1062 |  0.3546 |
|    16 |      +0.0864 |  0.4520 |
|    17 |      +0.1035 |  0.3673 |
|    18 |      +0.1039 |  0.3653 |
|    19 |      +0.1255 |  0.2736 |
|    20 |      +0.0972 |  0.3970 |
|    21 |      +0.0738 |  0.5207 |
|    22 |      +0.0583 |  0.6120 |
|    23 |      +0.0261 |  0.8206 |
|    24 |      +0.0141 |  0.9025 |
|    25 |      +0.0108 |  0.9254 |
|    26 |      -0.0016 |  0.9892 |
|    27 |      -0.0167 |  0.8849 |
|    28 |      +0.1219 |  0.2876 |

## RSA by Layer -- Agency

**Peak:** Layer 1, rho = +0.2763, p = 0.0143

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.2763 |  0.0143 |
|     2 |      +0.1116 |  0.3308 |
|     3 |      +0.0445 |  0.6990 |
|     4 |      +0.1146 |  0.3177 |
|     5 |      +0.1392 |  0.2243 |
|     6 |      +0.1145 |  0.3182 |
|     7 |      +0.1564 |  0.1714 |
|     8 |      +0.2003 |  0.0787 |
|     9 |      +0.1786 |  0.1178 |
|    10 |      +0.1615 |  0.1578 |
|    11 |      +0.1611 |  0.1589 |
|    12 |      +0.1959 |  0.0856 |
|    13 |      +0.1883 |  0.0987 |
|    14 |      +0.1677 |  0.1422 |
|    15 |      +0.2103 |  0.0645 |
|    16 |      +0.1976 |  0.0829 |
|    17 |      +0.2280 |  0.0446 |
|    18 |      +0.2204 |  0.0525 |
|    19 |      +0.2562 |  0.0236 |
|    20 |      +0.2025 |  0.0754 |
|    21 |      +0.1409 |  0.2186 |
|    22 |      +0.1102 |  0.3370 |
|    23 |      +0.0695 |  0.5457 |
|    24 |      +0.0438 |  0.7037 |
|    25 |      +0.0537 |  0.6407 |
|    26 |      +0.0443 |  0.7002 |
|    27 |      +0.0510 |  0.6577 |
|    28 |      +0.1445 |  0.2070 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
