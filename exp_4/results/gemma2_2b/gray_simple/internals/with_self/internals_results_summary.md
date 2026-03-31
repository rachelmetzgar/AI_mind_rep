# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:53:48
**Tag:** `with_self`
**Include self:** True
**Model:** Gemma-2-2B (Base)

---

## What is being tested

Does Gemma-2-2B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Gemma-2-2B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 7, rho = +0.2363, p = 0.0373

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0222 |  0.8471 |
|     2 |      -0.0422 |  0.7139 |
|     3 |      +0.0225 |  0.8452 |
|     4 |      +0.1629 |  0.1541 |
|     5 |      +0.2267 |  0.0460 |
|     6 |      +0.2328 |  0.0402 |
|     7 |      +0.2363 |  0.0373 |
|     8 |      +0.1945 |  0.0879 |
|     9 |      +0.2267 |  0.0459 |
|    10 |      +0.1590 |  0.1644 |
|    11 |      +0.2152 |  0.0585 |
|    12 |      +0.2102 |  0.0647 |
|    13 |      +0.2135 |  0.0605 |
|    14 |      +0.1981 |  0.0821 |
|    15 |      +0.1802 |  0.1144 |
|    16 |      +0.1716 |  0.1331 |
|    17 |      +0.1640 |  0.1514 |
|    18 |      +0.1559 |  0.1730 |
|    19 |      +0.1673 |  0.1432 |
|    20 |      +0.1631 |  0.1537 |
|    21 |      +0.1509 |  0.1872 |
|    22 |      +0.1491 |  0.1927 |
|    23 |      +0.1312 |  0.2520 |
|    24 |      +0.1115 |  0.3311 |
|    25 |      +0.1154 |  0.3144 |
|    26 |      +0.0727 |  0.5270 |

## RSA by Layer -- Experience

**Peak:** Layer 6, rho = +0.1034, p = 0.3678

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0719 |  0.5314 |
|     2 |      -0.0922 |  0.4222 |
|     3 |      -0.0525 |  0.6481 |
|     4 |      +0.0178 |  0.8770 |
|     5 |      +0.0757 |  0.5101 |
|     6 |      +0.1034 |  0.3678 |
|     7 |      +0.0895 |  0.4357 |
|     8 |      +0.0544 |  0.6363 |
|     9 |      +0.0719 |  0.5318 |
|    10 |      +0.0499 |  0.6641 |
|    11 |      +0.0843 |  0.4631 |
|    12 |      +0.0864 |  0.4518 |
|    13 |      +0.0994 |  0.3865 |
|    14 |      +0.0921 |  0.4227 |
|    15 |      +0.0974 |  0.3962 |
|    16 |      +0.0870 |  0.4487 |
|    17 |      +0.0857 |  0.4556 |
|    18 |      +0.0732 |  0.5244 |
|    19 |      +0.0803 |  0.4847 |
|    20 |      +0.0712 |  0.5356 |
|    21 |      +0.0571 |  0.6195 |
|    22 |      +0.0413 |  0.7197 |
|    23 |      +0.0256 |  0.8242 |
|    24 |      +0.0115 |  0.9205 |
|    25 |      +0.0223 |  0.8462 |
|    26 |      +0.0010 |  0.9927 |

## RSA by Layer -- Agency

**Peak:** Layer 7, rho = +0.2214, p = 0.0514

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0009 |  0.9935 |
|     2 |      -0.0235 |  0.8379 |
|     3 |      +0.0497 |  0.6653 |
|     4 |      +0.1520 |  0.1839 |
|     5 |      +0.2079 |  0.0678 |
|     6 |      +0.1975 |  0.0831 |
|     7 |      +0.2214 |  0.0514 |
|     8 |      +0.1715 |  0.1332 |
|     9 |      +0.2016 |  0.0767 |
|    10 |      +0.1253 |  0.2744 |
|    11 |      +0.1869 |  0.1013 |
|    12 |      +0.1681 |  0.1414 |
|    13 |      +0.1510 |  0.1871 |
|    14 |      +0.1287 |  0.2615 |
|    15 |      +0.0994 |  0.3866 |
|    16 |      +0.0860 |  0.4538 |
|    17 |      +0.0705 |  0.5394 |
|    18 |      +0.0679 |  0.5549 |
|    19 |      +0.0782 |  0.4960 |
|    20 |      +0.0789 |  0.4923 |
|    21 |      +0.0772 |  0.5019 |
|    22 |      +0.0773 |  0.5011 |
|    23 |      +0.0673 |  0.5585 |
|    24 |      +0.0499 |  0.6644 |
|    25 |      +0.0434 |  0.7063 |
|    26 |      +0.0114 |  0.9210 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
