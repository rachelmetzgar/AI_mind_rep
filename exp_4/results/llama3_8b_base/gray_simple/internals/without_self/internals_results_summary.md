# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-26 14:16:57
**Tag:** `without_self`
**Include self:** False
**Model:** LLaMA-3-8B (Base)

---

## What is being tested

Does LLaMA-3-8B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from LLaMA-3-8B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 8, rho = +0.3346, p = 0.0060

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1130 |  0.3664 |
|     2 |      -0.0213 |  0.8651 |
|     3 |      +0.0510 |  0.6840 |
|     4 |      +0.2000 |  0.1074 |
|     5 |      +0.1656 |  0.1839 |
|     6 |      +0.1743 |  0.1615 |
|     7 |      +0.2435 |  0.0488 |
|     8 |      +0.3346 |  0.0060 |
|     9 |      +0.2979 |  0.0151 |
|    10 |      +0.2923 |  0.0172 |
|    11 |      +0.2845 |  0.0206 |
|    12 |      +0.2751 |  0.0254 |
|    13 |      +0.3156 |  0.0099 |
|    14 |      +0.2758 |  0.0250 |
|    15 |      +0.2425 |  0.0498 |
|    16 |      +0.2257 |  0.0685 |
|    17 |      +0.1815 |  0.1448 |
|    18 |      +0.1546 |  0.2151 |
|    19 |      +0.1434 |  0.2506 |
|    20 |      +0.1484 |  0.2345 |
|    21 |      +0.1416 |  0.2568 |
|    22 |      +0.1552 |  0.2134 |
|    23 |      +0.1374 |  0.2712 |
|    24 |      +0.1332 |  0.2862 |
|    25 |      +0.1149 |  0.3584 |
|    26 |      +0.1131 |  0.3660 |
|    27 |      +0.1112 |  0.3742 |
|    28 |      +0.0899 |  0.4726 |
|    29 |      +0.0810 |  0.5181 |
|    30 |      +0.0977 |  0.4353 |
|    31 |      +0.1081 |  0.3876 |
|    32 |      +0.2347 |  0.0579 |

## RSA by Layer -- Experience

**Peak:** Layer 8, rho = +0.1533, p = 0.2192

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1441 |  0.2484 |
|     2 |      -0.0810 |  0.5182 |
|     3 |      -0.0380 |  0.7619 |
|     4 |      +0.0210 |  0.8668 |
|     5 |      +0.0146 |  0.9073 |
|     6 |      +0.0307 |  0.8065 |
|     7 |      +0.0816 |  0.5147 |
|     8 |      +0.1533 |  0.2192 |
|     9 |      +0.1254 |  0.3158 |
|    10 |      +0.1149 |  0.3581 |
|    11 |      +0.0935 |  0.4552 |
|    12 |      +0.1018 |  0.4162 |
|    13 |      +0.1305 |  0.2964 |
|    14 |      +0.1247 |  0.3183 |
|    15 |      +0.0937 |  0.4541 |
|    16 |      +0.0908 |  0.4684 |
|    17 |      +0.0459 |  0.7144 |
|    18 |      +0.0163 |  0.8966 |
|    19 |      +0.0061 |  0.9612 |
|    20 |      +0.0091 |  0.9423 |
|    21 |      +0.0003 |  0.9981 |
|    22 |      +0.0039 |  0.9749 |
|    23 |      -0.0145 |  0.9078 |
|    24 |      -0.0225 |  0.8579 |
|    25 |      -0.0343 |  0.7844 |
|    26 |      -0.0346 |  0.7824 |
|    27 |      -0.0330 |  0.7924 |
|    28 |      -0.0499 |  0.6909 |
|    29 |      -0.0625 |  0.6179 |
|    30 |      -0.0454 |  0.7174 |
|    31 |      -0.0406 |  0.7461 |
|    32 |      +0.0469 |  0.7082 |

## RSA by Layer -- Agency

**Peak:** Layer 8, rho = +0.1837, p = 0.1398

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1664 |  0.1818 |
|     2 |      -0.1154 |  0.3563 |
|     3 |      -0.0386 |  0.7584 |
|     4 |      +0.1034 |  0.4085 |
|     5 |      +0.0447 |  0.7213 |
|     6 |      +0.0383 |  0.7602 |
|     7 |      +0.1092 |  0.3829 |
|     8 |      +0.1837 |  0.1398 |
|     9 |      +0.1498 |  0.2300 |
|    10 |      +0.1453 |  0.2443 |
|    11 |      +0.1546 |  0.2153 |
|    12 |      +0.1312 |  0.2938 |
|    13 |      +0.1755 |  0.1588 |
|    14 |      +0.1136 |  0.3638 |
|    15 |      +0.0829 |  0.5079 |
|    16 |      +0.0599 |  0.6327 |
|    17 |      +0.0172 |  0.8911 |
|    18 |      -0.0015 |  0.9902 |
|    19 |      -0.0038 |  0.9757 |
|    20 |      +0.0007 |  0.9958 |
|    21 |      -0.0014 |  0.9912 |
|    22 |      +0.0003 |  0.9983 |
|    23 |      -0.0082 |  0.9480 |
|    24 |      -0.0008 |  0.9951 |
|    25 |      -0.0195 |  0.8765 |
|    26 |      -0.0151 |  0.9043 |
|    27 |      -0.0220 |  0.8609 |
|    28 |      -0.0315 |  0.8020 |
|    29 |      -0.0335 |  0.7893 |
|    30 |      -0.0110 |  0.9303 |
|    31 |      -0.0084 |  0.9464 |
|    32 |      +0.1091 |  0.3832 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
