# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:54:22
**Tag:** `with_self`
**Include self:** True
**Model:** Gemma-2-9B (Base)

---

## What is being tested

Does Gemma-2-9B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Gemma-2-9B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 18, rho = +0.3603, p = 0.0012

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0258 |  0.8225 |
|     2 |      -0.0122 |  0.9159 |
|     3 |      -0.0152 |  0.8950 |
|     4 |      +0.1560 |  0.1725 |
|     5 |      +0.1174 |  0.3061 |
|     6 |      +0.2307 |  0.0421 |
|     7 |      +0.2543 |  0.0246 |
|     8 |      +0.2685 |  0.0175 |
|     9 |      +0.3007 |  0.0075 |
|    10 |      +0.2918 |  0.0095 |
|    11 |      +0.2512 |  0.0265 |
|    12 |      +0.2982 |  0.0080 |
|    13 |      +0.3103 |  0.0057 |
|    14 |      +0.3305 |  0.0031 |
|    15 |      +0.3453 |  0.0020 |
|    16 |      +0.3276 |  0.0034 |
|    17 |      +0.3519 |  0.0016 |
|    18 |      +0.3603 |  0.0012 |
|    19 |      +0.3055 |  0.0065 |
|    20 |      +0.2683 |  0.0175 |
|    21 |      +0.2473 |  0.0290 |
|    22 |      +0.2217 |  0.0510 |
|    23 |      +0.1968 |  0.0842 |
|    24 |      +0.1764 |  0.1224 |
|    25 |      +0.1806 |  0.1135 |
|    26 |      +0.1689 |  0.1393 |
|    27 |      +0.1498 |  0.1906 |
|    28 |      +0.1590 |  0.1643 |
|    29 |      +0.1486 |  0.1942 |
|    30 |      +0.1516 |  0.1852 |
|    31 |      +0.1612 |  0.1584 |
|    32 |      +0.1496 |  0.1911 |
|    33 |      +0.1372 |  0.2310 |
|    34 |      +0.1182 |  0.3027 |
|    35 |      +0.1222 |  0.2866 |
|    36 |      +0.1312 |  0.2520 |
|    37 |      +0.1169 |  0.3079 |
|    38 |      +0.1028 |  0.3702 |
|    39 |      +0.1173 |  0.3065 |
|    40 |      +0.0880 |  0.4439 |
|    41 |      +0.0676 |  0.5567 |
|    42 |      +0.1200 |  0.2952 |

## RSA by Layer -- Experience

**Peak:** Layer 18, rho = +0.1974, p = 0.0832

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0672 |  0.5588 |
|     2 |      -0.0834 |  0.4677 |
|     3 |      -0.0889 |  0.4389 |
|     4 |      +0.0218 |  0.8497 |
|     5 |      +0.0017 |  0.9883 |
|     6 |      +0.0543 |  0.6365 |
|     7 |      +0.0719 |  0.5314 |
|     8 |      +0.0889 |  0.4389 |
|     9 |      +0.1180 |  0.3037 |
|    10 |      +0.1161 |  0.3115 |
|    11 |      +0.1001 |  0.3834 |
|    12 |      +0.1272 |  0.2671 |
|    13 |      +0.1380 |  0.2282 |
|    14 |      +0.1418 |  0.2157 |
|    15 |      +0.1685 |  0.1404 |
|    16 |      +0.1683 |  0.1408 |
|    17 |      +0.1908 |  0.0943 |
|    18 |      +0.1974 |  0.0832 |
|    19 |      +0.1740 |  0.1276 |
|    20 |      +0.1573 |  0.1691 |
|    21 |      +0.1435 |  0.2100 |
|    22 |      +0.1246 |  0.2771 |
|    23 |      +0.1139 |  0.3206 |
|    24 |      +0.0996 |  0.3856 |
|    25 |      +0.0958 |  0.4042 |
|    26 |      +0.0845 |  0.4621 |
|    27 |      +0.0679 |  0.5548 |
|    28 |      +0.0672 |  0.5588 |
|    29 |      +0.0659 |  0.5664 |
|    30 |      +0.0628 |  0.5848 |
|    31 |      +0.0787 |  0.4936 |
|    32 |      +0.0599 |  0.6026 |
|    33 |      +0.0485 |  0.6730 |
|    34 |      +0.0312 |  0.7862 |
|    35 |      +0.0282 |  0.8064 |
|    36 |      +0.0361 |  0.7535 |
|    37 |      +0.0265 |  0.8180 |
|    38 |      +0.0195 |  0.8655 |
|    39 |      +0.0283 |  0.8057 |
|    40 |      +0.0126 |  0.9126 |
|    41 |      +0.0054 |  0.9628 |
|    42 |      +0.0350 |  0.7609 |

## RSA by Layer -- Agency

**Peak:** Layer 15, rho = +0.2776, p = 0.0139

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.1178 |  0.3042 |
|     2 |      +0.0123 |  0.9148 |
|     3 |      +0.0141 |  0.9025 |
|     4 |      +0.1397 |  0.2224 |
|     5 |      +0.0979 |  0.3937 |
|     6 |      +0.2112 |  0.0635 |
|     7 |      +0.2262 |  0.0465 |
|     8 |      +0.2388 |  0.0352 |
|     9 |      +0.2536 |  0.0250 |
|    10 |      +0.2303 |  0.0425 |
|    11 |      +0.1855 |  0.1040 |
|    12 |      +0.2313 |  0.0416 |
|    13 |      +0.2388 |  0.0352 |
|    14 |      +0.2626 |  0.0202 |
|    15 |      +0.2776 |  0.0139 |
|    16 |      +0.2482 |  0.0284 |
|    17 |      +0.2641 |  0.0195 |
|    18 |      +0.2717 |  0.0161 |
|    19 |      +0.2112 |  0.0634 |
|    20 |      +0.1747 |  0.1262 |
|    21 |      +0.1550 |  0.1754 |
|    22 |      +0.1332 |  0.2450 |
|    23 |      +0.1030 |  0.3697 |
|    24 |      +0.0854 |  0.4572 |
|    25 |      +0.0926 |  0.4201 |
|    26 |      +0.0860 |  0.4540 |
|    27 |      +0.0677 |  0.5559 |
|    28 |      +0.0765 |  0.5058 |
|    29 |      +0.0652 |  0.5705 |
|    30 |      +0.0653 |  0.5702 |
|    31 |      +0.0743 |  0.5182 |
|    32 |      +0.0711 |  0.5359 |
|    33 |      +0.0574 |  0.6179 |
|    34 |      +0.0481 |  0.6761 |
|    35 |      +0.0542 |  0.6372 |
|    36 |      +0.0581 |  0.6134 |
|    37 |      +0.0498 |  0.6650 |
|    38 |      +0.0395 |  0.7315 |
|    39 |      +0.0535 |  0.6419 |
|    40 |      +0.0335 |  0.7710 |
|    41 |      +0.0111 |  0.9235 |
|    42 |      +0.0446 |  0.6985 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
