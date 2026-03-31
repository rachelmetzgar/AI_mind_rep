# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-28 12:55:42
**Tag:** `without_self`
**Include self:** False
**Model:** Qwen3-8B

---

## What is being tested

Does Qwen3-8B's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Qwen3-8B in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 18, rho = +0.3574, p = 0.0032

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0112 |  0.9288 |
|     2 |      +0.0066 |  0.9578 |
|     3 |      -0.0100 |  0.9365 |
|     4 |      +0.0183 |  0.8843 |
|     5 |      +0.0416 |  0.7401 |
|     6 |      +0.0578 |  0.6450 |
|     7 |      +0.0493 |  0.6941 |
|     8 |      +0.0618 |  0.6222 |
|     9 |      +0.1263 |  0.3124 |
|    10 |      +0.1474 |  0.2375 |
|    11 |      +0.1334 |  0.2856 |
|    12 |      +0.1714 |  0.1687 |
|    13 |      +0.2311 |  0.0620 |
|    14 |      +0.2302 |  0.0630 |
|    15 |      +0.2261 |  0.0679 |
|    16 |      +0.2248 |  0.0696 |
|    17 |      +0.2863 |  0.0198 |
|    18 |      +0.3574 |  0.0032 |
|    19 |      +0.2668 |  0.0303 |
|    20 |      +0.2712 |  0.0276 |
|    21 |      +0.2912 |  0.0177 |
|    22 |      +0.2913 |  0.0176 |
|    23 |      +0.3041 |  0.0130 |
|    24 |      +0.3278 |  0.0072 |
|    25 |      +0.3313 |  0.0066 |
|    26 |      +0.3124 |  0.0107 |
|    27 |      +0.3421 |  0.0049 |
|    28 |      +0.3339 |  0.0061 |
|    29 |      +0.3344 |  0.0061 |
|    30 |      +0.3357 |  0.0059 |
|    31 |      +0.3184 |  0.0092 |
|    32 |      +0.2944 |  0.0164 |
|    33 |      +0.2979 |  0.0151 |
|    34 |      +0.2827 |  0.0215 |
|    35 |      +0.2956 |  0.0159 |
|    36 |      +0.2368 |  0.0556 |

## RSA by Layer -- Experience

**Peak:** Layer 23, rho = +0.2705, p = 0.0280

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0679 |  0.5878 |
|     2 |      -0.0291 |  0.8165 |
|     3 |      -0.0525 |  0.6758 |
|     4 |      -0.0350 |  0.7804 |
|     5 |      -0.0090 |  0.9427 |
|     6 |      +0.0092 |  0.9414 |
|     7 |      -0.0013 |  0.9915 |
|     8 |      +0.0080 |  0.9493 |
|     9 |      +0.0603 |  0.6306 |
|    10 |      +0.0573 |  0.6479 |
|    11 |      +0.0786 |  0.5303 |
|    12 |      +0.0749 |  0.5500 |
|    13 |      +0.1101 |  0.3790 |
|    14 |      +0.1035 |  0.4082 |
|    15 |      +0.0981 |  0.4332 |
|    16 |      +0.1252 |  0.3165 |
|    17 |      +0.1804 |  0.1473 |
|    18 |      +0.2365 |  0.0559 |
|    19 |      +0.1623 |  0.1929 |
|    20 |      +0.1811 |  0.1457 |
|    21 |      +0.2244 |  0.0701 |
|    22 |      +0.2342 |  0.0584 |
|    23 |      +0.2705 |  0.0280 |
|    24 |      +0.2611 |  0.0342 |
|    25 |      +0.2470 |  0.0456 |
|    26 |      +0.2298 |  0.0635 |
|    27 |      +0.2620 |  0.0336 |
|    28 |      +0.2619 |  0.0337 |
|    29 |      +0.2580 |  0.0365 |
|    30 |      +0.2464 |  0.0461 |
|    31 |      +0.2159 |  0.0817 |
|    32 |      +0.2021 |  0.1037 |
|    33 |      +0.2030 |  0.1022 |
|    34 |      +0.1917 |  0.1231 |
|    35 |      +0.2006 |  0.1064 |
|    36 |      +0.1599 |  0.1998 |

## RSA by Layer -- Agency

**Peak:** Layer 18, rho = +0.2577, p = 0.0367

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0874 |  0.4852 |
|     2 |      +0.0745 |  0.5523 |
|     3 |      +0.0245 |  0.8451 |
|     4 |      -0.0749 |  0.5501 |
|     5 |      -0.0704 |  0.5746 |
|     6 |      -0.0468 |  0.7088 |
|     7 |      -0.0625 |  0.6181 |
|     8 |      -0.0524 |  0.6759 |
|     9 |      -0.0128 |  0.9188 |
|    10 |      +0.0191 |  0.8793 |
|    11 |      -0.0110 |  0.9299 |
|    12 |      +0.0666 |  0.5954 |
|    13 |      +0.1359 |  0.2765 |
|    14 |      +0.1244 |  0.3195 |
|    15 |      +0.1065 |  0.3946 |
|    16 |      +0.0949 |  0.4486 |
|    17 |      +0.1735 |  0.1637 |
|    18 |      +0.2577 |  0.0367 |
|    19 |      +0.1726 |  0.1657 |
|    20 |      +0.1803 |  0.1473 |
|    21 |      +0.1910 |  0.1245 |
|    22 |      +0.1887 |  0.1292 |
|    23 |      +0.1806 |  0.1468 |
|    24 |      +0.2034 |  0.1014 |
|    25 |      +0.2116 |  0.0881 |
|    26 |      +0.1892 |  0.1281 |
|    27 |      +0.2195 |  0.0766 |
|    28 |      +0.2092 |  0.0919 |
|    29 |      +0.2108 |  0.0894 |
|    30 |      +0.2231 |  0.0718 |
|    31 |      +0.2064 |  0.0964 |
|    32 |      +0.1852 |  0.1365 |
|    33 |      +0.1909 |  0.1248 |
|    34 |      +0.1809 |  0.1461 |
|    35 |      +0.1924 |  0.1217 |
|    36 |      +0.1372 |  0.2718 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
