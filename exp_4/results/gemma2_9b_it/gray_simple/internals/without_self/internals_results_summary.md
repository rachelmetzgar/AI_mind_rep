# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-28 12:54:44
**Tag:** `without_self`
**Include self:** False
**Model:** Gemma-2-9B-IT

---

## What is being tested

Does Gemma-2-9B-IT's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Gemma-2-9B-IT in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

- **Human RDMs**: Three variants from Gray et al. (2007) factor scores:
  - Combined: Euclidean distance in 2D (Experience, Agency) space
  - Experience-only: |exp_i - exp_j|
  - Agency-only: |agency_i - agency_j|

**Test:** Spearman rank correlation between the upper triangles of the model RDM and each human RDM (66 unique entity pairs). Computed at every layer to track where in the network mind-perception structure emerges.

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

## RSA by Layer -- Combined

**Peak:** Layer 21, rho = +0.3783, p = 0.0017

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0578 |  0.6446 |
|     2 |      -0.0372 |  0.7667 |
|     3 |      -0.0388 |  0.7571 |
|     4 |      -0.0410 |  0.7437 |
|     5 |      +0.0358 |  0.7751 |
|     6 |      +0.0730 |  0.5602 |
|     7 |      +0.1113 |  0.3735 |
|     8 |      +0.1145 |  0.3600 |
|     9 |      +0.2317 |  0.0612 |
|    10 |      +0.3215 |  0.0085 |
|    11 |      +0.3490 |  0.0041 |
|    12 |      +0.3355 |  0.0059 |
|    13 |      +0.3023 |  0.0136 |
|    14 |      +0.2645 |  0.0318 |
|    15 |      +0.2860 |  0.0199 |
|    16 |      +0.2302 |  0.0629 |
|    17 |      +0.2641 |  0.0321 |
|    18 |      +0.2923 |  0.0172 |
|    19 |      +0.3647 |  0.0026 |
|    20 |      +0.3717 |  0.0021 |
|    21 |      +0.3783 |  0.0017 |
|    22 |      +0.3442 |  0.0047 |
|    23 |      +0.2827 |  0.0215 |
|    24 |      +0.2658 |  0.0310 |
|    25 |      +0.2699 |  0.0284 |
|    26 |      +0.2836 |  0.0210 |
|    27 |      +0.2450 |  0.0474 |
|    28 |      +0.2319 |  0.0610 |
|    29 |      +0.1920 |  0.1224 |
|    30 |      +0.2156 |  0.0821 |
|    31 |      +0.2328 |  0.0600 |
|    32 |      +0.2340 |  0.0586 |
|    33 |      +0.2635 |  0.0325 |
|    34 |      +0.2386 |  0.0537 |
|    35 |      +0.1788 |  0.1508 |
|    36 |      +0.1690 |  0.1749 |
|    37 |      +0.1884 |  0.1298 |
|    38 |      +0.1826 |  0.1422 |
|    39 |      +0.1849 |  0.1371 |
|    40 |      +0.1636 |  0.1893 |
|    41 |      +0.1468 |  0.2395 |
|    42 |      +0.1389 |  0.2659 |

## RSA by Layer -- Experience

**Peak:** Layer 21, rho = +0.2859, p = 0.0200

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1177 |  0.3468 |
|     2 |      -0.0699 |  0.5769 |
|     3 |      -0.0751 |  0.5488 |
|     4 |      -0.0700 |  0.5764 |
|     5 |      -0.0469 |  0.7083 |
|     6 |      -0.0518 |  0.6798 |
|     7 |      -0.0825 |  0.5103 |
|     8 |      -0.0696 |  0.5788 |
|     9 |      +0.0481 |  0.7014 |
|    10 |      +0.0917 |  0.4639 |
|    11 |      +0.1778 |  0.1533 |
|    12 |      +0.1763 |  0.1567 |
|    13 |      +0.1209 |  0.3337 |
|    14 |      +0.1250 |  0.3173 |
|    15 |      +0.1416 |  0.2568 |
|    16 |      +0.1305 |  0.2963 |
|    17 |      +0.1729 |  0.1651 |
|    18 |      +0.2135 |  0.0853 |
|    19 |      +0.2412 |  0.0510 |
|    20 |      +0.2799 |  0.0229 |
|    21 |      +0.2859 |  0.0200 |
|    22 |      +0.2385 |  0.0538 |
|    23 |      +0.1976 |  0.1117 |
|    24 |      +0.2020 |  0.1039 |
|    25 |      +0.1970 |  0.1128 |
|    26 |      +0.1877 |  0.1313 |
|    27 |      +0.1797 |  0.1489 |
|    28 |      +0.1715 |  0.1685 |
|    29 |      +0.1297 |  0.2993 |
|    30 |      +0.1503 |  0.2284 |
|    31 |      +0.1590 |  0.2023 |
|    32 |      +0.1612 |  0.1960 |
|    33 |      +0.1868 |  0.1331 |
|    34 |      +0.1844 |  0.1384 |
|    35 |      +0.1236 |  0.3227 |
|    36 |      +0.1052 |  0.4004 |
|    37 |      +0.1252 |  0.3167 |
|    38 |      +0.1172 |  0.3487 |
|    39 |      +0.1139 |  0.3623 |
|    40 |      +0.1114 |  0.3732 |
|    41 |      +0.1036 |  0.4076 |
|    42 |      +0.0896 |  0.4744 |

## RSA by Layer -- Agency

**Peak:** Layer 21, rho = +0.2240, p = 0.0707

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0938 |  0.4540 |
|     2 |      -0.1443 |  0.2477 |
|     3 |      -0.1409 |  0.2592 |
|     4 |      -0.1649 |  0.1857 |
|     5 |      -0.0930 |  0.4578 |
|     6 |      -0.0370 |  0.7681 |
|     7 |      -0.0046 |  0.9708 |
|     8 |      +0.0029 |  0.9816 |
|     9 |      +0.1030 |  0.4107 |
|    10 |      +0.1899 |  0.1267 |
|    11 |      +0.2073 |  0.0950 |
|    12 |      +0.1999 |  0.1076 |
|    13 |      +0.1751 |  0.1595 |
|    14 |      +0.1408 |  0.2593 |
|    15 |      +0.1847 |  0.1376 |
|    16 |      +0.1227 |  0.3264 |
|    17 |      +0.1486 |  0.2339 |
|    18 |      +0.1442 |  0.2480 |
|    19 |      +0.2154 |  0.0824 |
|    20 |      +0.2109 |  0.0891 |
|    21 |      +0.2240 |  0.0707 |
|    22 |      +0.2131 |  0.0858 |
|    23 |      +0.1498 |  0.2299 |
|    24 |      +0.1310 |  0.2943 |
|    25 |      +0.1363 |  0.2752 |
|    26 |      +0.1672 |  0.1796 |
|    27 |      +0.1087 |  0.3850 |
|    28 |      +0.0920 |  0.4626 |
|    29 |      +0.0750 |  0.5494 |
|    30 |      +0.0740 |  0.5546 |
|    31 |      +0.1036 |  0.4080 |
|    32 |      +0.1086 |  0.3854 |
|    33 |      +0.1179 |  0.3459 |
|    34 |      +0.0913 |  0.4660 |
|    35 |      +0.0411 |  0.7434 |
|    36 |      +0.0467 |  0.7099 |
|    37 |      +0.0683 |  0.5859 |
|    38 |      +0.0747 |  0.5514 |
|    39 |      +0.0756 |  0.5465 |
|    40 |      +0.0557 |  0.6568 |
|    41 |      +0.0384 |  0.7595 |
|    42 |      +0.0330 |  0.7926 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
