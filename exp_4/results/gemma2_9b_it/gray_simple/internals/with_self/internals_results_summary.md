# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-28 12:54:46
**Tag:** `with_self`
**Include self:** True
**Model:** Gemma-2-9B-IT

---

## What is being tested

Does Gemma-2-9B-IT's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Gemma-2-9B-IT in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 21, rho = +0.3921, p = 0.0004

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0448 |  0.6967 |
|     2 |      -0.0228 |  0.8431 |
|     3 |      -0.0213 |  0.8531 |
|     4 |      -0.0054 |  0.9628 |
|     5 |      +0.1520 |  0.1841 |
|     6 |      +0.2015 |  0.0769 |
|     7 |      +0.2153 |  0.0584 |
|     8 |      +0.2209 |  0.0519 |
|     9 |      +0.2793 |  0.0133 |
|    10 |      +0.3492 |  0.0017 |
|    11 |      +0.3721 |  0.0008 |
|    12 |      +0.3630 |  0.0011 |
|    13 |      +0.3407 |  0.0023 |
|    14 |      +0.3181 |  0.0045 |
|    15 |      +0.3315 |  0.0030 |
|    16 |      +0.3004 |  0.0075 |
|    17 |      +0.3260 |  0.0036 |
|    18 |      +0.3385 |  0.0024 |
|    19 |      +0.3839 |  0.0005 |
|    20 |      +0.3911 |  0.0004 |
|    21 |      +0.3921 |  0.0004 |
|    22 |      +0.3666 |  0.0010 |
|    23 |      +0.3297 |  0.0032 |
|    24 |      +0.3212 |  0.0041 |
|    25 |      +0.3176 |  0.0046 |
|    26 |      +0.3208 |  0.0042 |
|    27 |      +0.2800 |  0.0130 |
|    28 |      +0.2576 |  0.0228 |
|    29 |      +0.2369 |  0.0367 |
|    30 |      +0.2354 |  0.0380 |
|    31 |      +0.2268 |  0.0459 |
|    32 |      +0.2242 |  0.0484 |
|    33 |      +0.2354 |  0.0380 |
|    34 |      +0.1940 |  0.0887 |
|    35 |      +0.1736 |  0.1285 |
|    36 |      +0.1530 |  0.1810 |
|    37 |      +0.1657 |  0.1470 |
|    38 |      +0.1512 |  0.1864 |
|    39 |      +0.1592 |  0.1637 |
|    40 |      +0.1453 |  0.2042 |
|    41 |      +0.1287 |  0.2615 |
|    42 |      +0.1338 |  0.2428 |

## RSA by Layer -- Experience

**Peak:** Layer 21, rho = +0.2890, p = 0.0103

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1202 |  0.2947 |
|     2 |      -0.0891 |  0.4381 |
|     3 |      -0.0924 |  0.4210 |
|     4 |      -0.0749 |  0.5146 |
|     5 |      +0.0368 |  0.7493 |
|     6 |      +0.0489 |  0.6704 |
|     7 |      +0.0177 |  0.8778 |
|     8 |      +0.0302 |  0.7929 |
|     9 |      +0.0963 |  0.4019 |
|    10 |      +0.1463 |  0.2013 |
|    11 |      +0.2057 |  0.0707 |
|    12 |      +0.2054 |  0.0712 |
|    13 |      +0.1680 |  0.1415 |
|    14 |      +0.1758 |  0.1237 |
|    15 |      +0.1842 |  0.1065 |
|    16 |      +0.1811 |  0.1125 |
|    17 |      +0.2129 |  0.0612 |
|    18 |      +0.2343 |  0.0390 |
|    19 |      +0.2525 |  0.0258 |
|    20 |      +0.2853 |  0.0113 |
|    21 |      +0.2890 |  0.0103 |
|    22 |      +0.2590 |  0.0220 |
|    23 |      +0.2361 |  0.0374 |
|    24 |      +0.2417 |  0.0330 |
|    25 |      +0.2324 |  0.0406 |
|    26 |      +0.2184 |  0.0548 |
|    27 |      +0.2075 |  0.0683 |
|    28 |      +0.1970 |  0.0838 |
|    29 |      +0.1658 |  0.1469 |
|    30 |      +0.1716 |  0.1331 |
|    31 |      +0.1533 |  0.1803 |
|    32 |      +0.1435 |  0.2102 |
|    33 |      +0.1514 |  0.1856 |
|    34 |      +0.1280 |  0.2639 |
|    35 |      +0.0885 |  0.4411 |
|    36 |      +0.0725 |  0.5283 |
|    37 |      +0.0836 |  0.4670 |
|    38 |      +0.0696 |  0.5451 |
|    39 |      +0.0705 |  0.5394 |
|    40 |      +0.0746 |  0.5161 |
|    41 |      +0.0633 |  0.5818 |
|    42 |      +0.0673 |  0.5582 |

## RSA by Layer -- Agency

**Peak:** Layer 19, rho = +0.3057, p = 0.0065

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0378 |  0.7427 |
|     2 |      -0.0634 |  0.5812 |
|     3 |      -0.0538 |  0.6400 |
|     4 |      -0.0533 |  0.6431 |
|     5 |      +0.0771 |  0.5025 |
|     6 |      +0.1512 |  0.1865 |
|     7 |      +0.1438 |  0.2092 |
|     8 |      +0.1579 |  0.1674 |
|     9 |      +0.2283 |  0.0444 |
|    10 |      +0.2856 |  0.0113 |
|    11 |      +0.3031 |  0.0070 |
|    12 |      +0.2999 |  0.0076 |
|    13 |      +0.2814 |  0.0126 |
|    14 |      +0.2598 |  0.0216 |
|    15 |      +0.2878 |  0.0106 |
|    16 |      +0.2505 |  0.0269 |
|    17 |      +0.2694 |  0.0171 |
|    18 |      +0.2611 |  0.0210 |
|    19 |      +0.3057 |  0.0065 |
|    20 |      +0.2999 |  0.0076 |
|    21 |      +0.3038 |  0.0068 |
|    22 |      +0.2905 |  0.0099 |
|    23 |      +0.2531 |  0.0254 |
|    24 |      +0.2418 |  0.0329 |
|    25 |      +0.2421 |  0.0327 |
|    26 |      +0.2557 |  0.0239 |
|    27 |      +0.1919 |  0.0923 |
|    28 |      +0.1604 |  0.1607 |
|    29 |      +0.1626 |  0.1548 |
|    30 |      +0.1421 |  0.2145 |
|    31 |      +0.1425 |  0.2133 |
|    32 |      +0.1468 |  0.1997 |
|    33 |      +0.1481 |  0.1958 |
|    34 |      +0.1161 |  0.3114 |
|    35 |      +0.1101 |  0.3372 |
|    36 |      +0.0915 |  0.4258 |
|    37 |      +0.1090 |  0.3423 |
|    38 |      +0.1029 |  0.3700 |
|    39 |      +0.1119 |  0.3293 |
|    40 |      +0.0953 |  0.4065 |
|    41 |      +0.0714 |  0.5345 |
|    42 |      +0.0778 |  0.4981 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
