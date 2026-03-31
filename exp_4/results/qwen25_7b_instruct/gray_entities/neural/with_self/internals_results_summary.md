# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-28 11:57:31
**Tag:** `with_self`
**Include self:** True
**Model:** Qwen-2.5-7B-Instruct

---

## What is being tested

Does Qwen-2.5-7B-Instruct's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Qwen-2.5-7B-Instruct in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 19, rho = +0.2634, p = 0.0198

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0143 |  0.9011 |
|     2 |      -0.0398 |  0.7296 |
|     3 |      -0.0604 |  0.5996 |
|     4 |      +0.0357 |  0.7565 |
|     5 |      +0.0773 |  0.5013 |
|     6 |      +0.0933 |  0.4163 |
|     7 |      +0.1403 |  0.2204 |
|     8 |      +0.1595 |  0.1629 |
|     9 |      +0.2129 |  0.0613 |
|    10 |      +0.2200 |  0.0530 |
|    11 |      +0.1997 |  0.0797 |
|    12 |      +0.2044 |  0.0727 |
|    13 |      +0.2421 |  0.0327 |
|    14 |      +0.2296 |  0.0432 |
|    15 |      +0.2281 |  0.0446 |
|    16 |      +0.1973 |  0.0834 |
|    17 |      +0.1977 |  0.0828 |
|    18 |      +0.2210 |  0.0519 |
|    19 |      +0.2634 |  0.0198 |
|    20 |      +0.2430 |  0.0320 |
|    21 |      +0.2536 |  0.0251 |
|    22 |      +0.2462 |  0.0298 |
|    23 |      +0.2222 |  0.0505 |
|    24 |      +0.2292 |  0.0435 |
|    25 |      +0.2159 |  0.0576 |
|    26 |      +0.2295 |  0.0433 |
|    27 |      +0.2171 |  0.0563 |
|    28 |      +0.2410 |  0.0336 |

## RSA by Layer -- Experience

**Peak:** Layer 19, rho = +0.2178, p = 0.0555

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0923 |  0.4218 |
|     2 |      -0.1202 |  0.2947 |
|     3 |      -0.1262 |  0.2709 |
|     4 |      -0.0528 |  0.6460 |
|     5 |      -0.0367 |  0.7499 |
|     6 |      -0.0288 |  0.8021 |
|     7 |      -0.0059 |  0.9589 |
|     8 |      +0.0265 |  0.8177 |
|     9 |      +0.0678 |  0.5556 |
|    10 |      +0.0915 |  0.4258 |
|    11 |      +0.0852 |  0.4582 |
|    12 |      +0.1040 |  0.3648 |
|    13 |      +0.1499 |  0.1902 |
|    14 |      +0.1448 |  0.2060 |
|    15 |      +0.1558 |  0.1732 |
|    16 |      +0.1651 |  0.1485 |
|    17 |      +0.1769 |  0.1214 |
|    18 |      +0.1967 |  0.0843 |
|    19 |      +0.2178 |  0.0555 |
|    20 |      +0.1934 |  0.0897 |
|    21 |      +0.2002 |  0.0789 |
|    22 |      +0.1888 |  0.0979 |
|    23 |      +0.1590 |  0.1643 |
|    24 |      +0.1735 |  0.1288 |
|    25 |      +0.1602 |  0.1612 |
|    26 |      +0.1564 |  0.1716 |
|    27 |      +0.1384 |  0.2270 |
|    28 |      +0.2050 |  0.0718 |

## RSA by Layer -- Agency

**Peak:** Layer 13, rho = +0.1869, p = 0.1014

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0063 |  0.9560 |
|     2 |      -0.0048 |  0.9666 |
|     3 |      -0.0560 |  0.6263 |
|     4 |      +0.0065 |  0.9552 |
|     5 |      +0.0422 |  0.7137 |
|     6 |      +0.0710 |  0.5365 |
|     7 |      +0.1065 |  0.3534 |
|     8 |      +0.1137 |  0.3216 |
|     9 |      +0.1709 |  0.1347 |
|    10 |      +0.1798 |  0.1152 |
|    11 |      +0.1560 |  0.1726 |
|    12 |      +0.1531 |  0.1807 |
|    13 |      +0.1869 |  0.1014 |
|    14 |      +0.1688 |  0.1396 |
|    15 |      +0.1543 |  0.1775 |
|    16 |      +0.1196 |  0.2970 |
|    17 |      +0.1185 |  0.3016 |
|    18 |      +0.1391 |  0.2247 |
|    19 |      +0.1708 |  0.1350 |
|    20 |      +0.1487 |  0.1939 |
|    21 |      +0.1674 |  0.1429 |
|    22 |      +0.1518 |  0.1845 |
|    23 |      +0.1333 |  0.2446 |
|    24 |      +0.1300 |  0.2566 |
|    25 |      +0.1067 |  0.3526 |
|    26 |      +0.1190 |  0.2996 |
|    27 |      +0.1042 |  0.3639 |
|    28 |      +0.1309 |  0.2535 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
