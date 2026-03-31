# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-28 11:57:29
**Tag:** `without_self`
**Include self:** False
**Model:** Qwen-2.5-7B-Instruct

---

## What is being tested

Does Qwen-2.5-7B-Instruct's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Qwen-2.5-7B-Instruct in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 19, rho = +0.1939, p = 0.1187

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0657 |  0.6000 |
|     2 |      -0.0917 |  0.4638 |
|     3 |      -0.1092 |  0.3826 |
|     4 |      -0.0581 |  0.6429 |
|     5 |      -0.0287 |  0.8188 |
|     6 |      +0.0007 |  0.9956 |
|     7 |      +0.0637 |  0.6114 |
|     8 |      +0.0846 |  0.4994 |
|     9 |      +0.1257 |  0.3145 |
|    10 |      +0.1313 |  0.2932 |
|    11 |      +0.1302 |  0.2974 |
|    12 |      +0.1322 |  0.2902 |
|    13 |      +0.1775 |  0.1539 |
|    14 |      +0.1722 |  0.1669 |
|    15 |      +0.1813 |  0.1453 |
|    16 |      +0.1284 |  0.3040 |
|    17 |      +0.1313 |  0.2934 |
|    18 |      +0.1531 |  0.2197 |
|    19 |      +0.1939 |  0.1187 |
|    20 |      +0.1478 |  0.2364 |
|    21 |      +0.1781 |  0.1526 |
|    22 |      +0.1723 |  0.1666 |
|    23 |      +0.1400 |  0.2621 |
|    24 |      +0.1570 |  0.2082 |
|    25 |      +0.1464 |  0.2410 |
|    26 |      +0.1634 |  0.1898 |
|    27 |      +0.1698 |  0.1728 |
|    28 |      +0.1772 |  0.1545 |

## RSA by Layer -- Experience

**Peak:** Layer 19, rho = +0.1721, p = 0.1671

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1025 |  0.4128 |
|     2 |      -0.1320 |  0.2909 |
|     3 |      -0.1510 |  0.2262 |
|     4 |      -0.1240 |  0.3213 |
|     5 |      -0.1131 |  0.3660 |
|     6 |      -0.0808 |  0.5187 |
|     7 |      -0.0589 |  0.6385 |
|     8 |      -0.0197 |  0.8754 |
|     9 |      +0.0075 |  0.9524 |
|    10 |      +0.0388 |  0.7572 |
|    11 |      +0.0499 |  0.6909 |
|    12 |      +0.0624 |  0.6187 |
|    13 |      +0.1203 |  0.3358 |
|    14 |      +0.1223 |  0.3279 |
|    15 |      +0.1425 |  0.2538 |
|    16 |      +0.1380 |  0.2690 |
|    17 |      +0.1503 |  0.2282 |
|    18 |      +0.1633 |  0.1901 |
|    19 |      +0.1721 |  0.1671 |
|    20 |      +0.1272 |  0.3089 |
|    21 |      +0.1457 |  0.2431 |
|    22 |      +0.1313 |  0.2933 |
|    23 |      +0.0871 |  0.4867 |
|    24 |      +0.1047 |  0.4028 |
|    25 |      +0.0854 |  0.4956 |
|    26 |      +0.0813 |  0.5165 |
|    27 |      +0.0815 |  0.5155 |
|    28 |      +0.1390 |  0.2657 |

## RSA by Layer -- Agency

**Peak:** Layer 28, rho = +0.0548, p = 0.6623

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1061 |  0.3964 |
|     2 |      -0.1004 |  0.4226 |
|     3 |      -0.1423 |  0.2545 |
|     4 |      -0.1228 |  0.3259 |
|     5 |      -0.1143 |  0.3607 |
|     6 |      -0.0852 |  0.4965 |
|     7 |      -0.0288 |  0.8185 |
|     8 |      -0.0287 |  0.8192 |
|     9 |      +0.0145 |  0.9078 |
|    10 |      +0.0224 |  0.8583 |
|    11 |      +0.0147 |  0.9070 |
|    12 |      +0.0094 |  0.9405 |
|    13 |      +0.0517 |  0.6801 |
|    14 |      +0.0401 |  0.7493 |
|    15 |      +0.0392 |  0.7548 |
|    16 |      -0.0144 |  0.9085 |
|    17 |      -0.0114 |  0.9275 |
|    18 |      +0.0122 |  0.9226 |
|    19 |      +0.0534 |  0.6702 |
|    20 |      +0.0106 |  0.9329 |
|    21 |      +0.0466 |  0.7103 |
|    22 |      +0.0393 |  0.7540 |
|    23 |      +0.0239 |  0.8490 |
|    24 |      +0.0365 |  0.7711 |
|    25 |      +0.0197 |  0.8749 |
|    26 |      +0.0387 |  0.7579 |
|    27 |      +0.0447 |  0.7218 |
|    28 |      +0.0548 |  0.6623 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
