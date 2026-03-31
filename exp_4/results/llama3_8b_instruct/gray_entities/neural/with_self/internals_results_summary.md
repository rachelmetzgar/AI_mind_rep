# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-26 14:14:25
**Tag:** `with_self`
**Include self:** True
**Model:** LLaMA-3-8B-Instruct

---

## What is being tested

Does LLaMA-3-8B-Instruct's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from LLaMA-3-8B-Instruct in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 10, rho = +0.4493, p = 0.0000

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0139 |  0.9037 |
|     2 |      +0.0180 |  0.8754 |
|     3 |      +0.0537 |  0.6403 |
|     4 |      +0.1477 |  0.1968 |
|     5 |      +0.1446 |  0.2065 |
|     6 |      +0.2136 |  0.0604 |
|     7 |      +0.2485 |  0.0283 |
|     8 |      +0.4314 |  0.0001 |
|     9 |      +0.4287 |  0.0001 |
|    10 |      +0.4493 |  0.0000 |
|    11 |      +0.4354 |  0.0001 |
|    12 |      +0.3867 |  0.0005 |
|    13 |      +0.3776 |  0.0007 |
|    14 |      +0.3209 |  0.0042 |
|    15 |      +0.3201 |  0.0043 |
|    16 |      +0.3226 |  0.0040 |
|    17 |      +0.3003 |  0.0076 |
|    18 |      +0.2737 |  0.0153 |
|    19 |      +0.2570 |  0.0231 |
|    20 |      +0.2594 |  0.0218 |
|    21 |      +0.2638 |  0.0196 |
|    22 |      +0.2211 |  0.0517 |
|    23 |      +0.2051 |  0.0716 |
|    24 |      +0.1952 |  0.0868 |
|    25 |      +0.1884 |  0.0985 |
|    26 |      +0.1740 |  0.1277 |
|    27 |      +0.1615 |  0.1577 |
|    28 |      +0.1557 |  0.1736 |
|    29 |      +0.1555 |  0.1741 |
|    30 |      +0.1441 |  0.2080 |
|    31 |      +0.1186 |  0.3009 |
|    32 |      +0.1287 |  0.2613 |

## RSA by Layer -- Experience

**Peak:** Layer 11, rho = +0.3368, p = 0.0026

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0471 |  0.6825 |
|     2 |      -0.0470 |  0.6827 |
|     3 |      -0.0249 |  0.8290 |
|     4 |      +0.0181 |  0.8752 |
|     5 |      +0.0153 |  0.8945 |
|     6 |      +0.0756 |  0.5109 |
|     7 |      +0.1232 |  0.2825 |
|     8 |      +0.2816 |  0.0125 |
|     9 |      +0.2974 |  0.0082 |
|    10 |      +0.3365 |  0.0026 |
|    11 |      +0.3368 |  0.0026 |
|    12 |      +0.3241 |  0.0038 |
|    13 |      +0.3125 |  0.0053 |
|    14 |      +0.2553 |  0.0241 |
|    15 |      +0.2450 |  0.0306 |
|    16 |      +0.2422 |  0.0326 |
|    17 |      +0.2121 |  0.0623 |
|    18 |      +0.1867 |  0.1016 |
|    19 |      +0.1692 |  0.1387 |
|    20 |      +0.1626 |  0.1550 |
|    21 |      +0.1540 |  0.1782 |
|    22 |      +0.1236 |  0.2811 |
|    23 |      +0.1075 |  0.3491 |
|    24 |      +0.1087 |  0.3434 |
|    25 |      +0.1000 |  0.3838 |
|    26 |      +0.0927 |  0.4196 |
|    27 |      +0.0900 |  0.4334 |
|    28 |      +0.0908 |  0.4291 |
|    29 |      +0.1059 |  0.3561 |
|    30 |      +0.0914 |  0.4261 |
|    31 |      +0.0691 |  0.5477 |
|    32 |      +0.0825 |  0.4726 |

## RSA by Layer -- Agency

**Peak:** Layer 10, rho = +0.3662, p = 0.0010

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0212 |  0.8540 |
|     2 |      -0.0208 |  0.8563 |
|     3 |      +0.0095 |  0.9340 |
|     4 |      +0.1260 |  0.2717 |
|     5 |      +0.1137 |  0.3215 |
|     6 |      +0.1706 |  0.1354 |
|     7 |      +0.1778 |  0.1195 |
|     8 |      +0.3586 |  0.0013 |
|     9 |      +0.3617 |  0.0011 |
|    10 |      +0.3662 |  0.0010 |
|    11 |      +0.3360 |  0.0026 |
|    12 |      +0.2714 |  0.0162 |
|    13 |      +0.2735 |  0.0154 |
|    14 |      +0.2295 |  0.0433 |
|    15 |      +0.2310 |  0.0419 |
|    16 |      +0.2438 |  0.0315 |
|    17 |      +0.2363 |  0.0373 |
|    18 |      +0.2275 |  0.0451 |
|    19 |      +0.2298 |  0.0430 |
|    20 |      +0.2242 |  0.0485 |
|    21 |      +0.2456 |  0.0302 |
|    22 |      +0.2020 |  0.0761 |
|    23 |      +0.1890 |  0.0974 |
|    24 |      +0.1622 |  0.1559 |
|    25 |      +0.1546 |  0.1765 |
|    26 |      +0.1378 |  0.2289 |
|    27 |      +0.1262 |  0.2710 |
|    28 |      +0.1221 |  0.2867 |
|    29 |      +0.1099 |  0.3381 |
|    30 |      +0.1032 |  0.3687 |
|    31 |      +0.0798 |  0.4876 |
|    32 |      +0.0960 |  0.4031 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
