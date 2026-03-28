# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-26 14:14:23
**Tag:** `without_self`
**Include self:** False
**Model:** LLaMA-3-8B-Instruct

---

## What is being tested

Does LLaMA-3-8B-Instruct's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from LLaMA-3-8B-Instruct in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 10, rho = +0.4437, p = 0.0002

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0001 |  0.9996 |
|     2 |      -0.0070 |  0.9553 |
|     3 |      +0.0169 |  0.8927 |
|     4 |      +0.0798 |  0.5242 |
|     5 |      +0.0258 |  0.8372 |
|     6 |      +0.1096 |  0.3811 |
|     7 |      +0.1578 |  0.2057 |
|     8 |      +0.4196 |  0.0005 |
|     9 |      +0.4137 |  0.0006 |
|    10 |      +0.4437 |  0.0002 |
|    11 |      +0.4297 |  0.0003 |
|    12 |      +0.3713 |  0.0021 |
|    13 |      +0.3548 |  0.0035 |
|    14 |      +0.3020 |  0.0137 |
|    15 |      +0.3073 |  0.0121 |
|    16 |      +0.3174 |  0.0094 |
|    17 |      +0.2739 |  0.0260 |
|    18 |      +0.2326 |  0.0602 |
|    19 |      +0.2068 |  0.0958 |
|    20 |      +0.2218 |  0.0735 |
|    21 |      +0.2427 |  0.0496 |
|    22 |      +0.2022 |  0.1035 |
|    23 |      +0.1844 |  0.1382 |
|    24 |      +0.1610 |  0.1967 |
|    25 |      +0.1608 |  0.1971 |
|    26 |      +0.1552 |  0.2134 |
|    27 |      +0.1424 |  0.2541 |
|    28 |      +0.1383 |  0.2680 |
|    29 |      +0.1368 |  0.2733 |
|    30 |      +0.1296 |  0.2996 |
|    31 |      +0.0978 |  0.4349 |
|    32 |      +0.1125 |  0.3683 |

## RSA by Layer -- Experience

**Peak:** Layer 11, rho = +0.3616, p = 0.0029

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0379 |  0.7627 |
|     2 |      -0.0518 |  0.6797 |
|     3 |      -0.0432 |  0.7306 |
|     4 |      -0.0146 |  0.9077 |
|     5 |      -0.0602 |  0.6310 |
|     6 |      +0.0020 |  0.9875 |
|     7 |      +0.0586 |  0.6403 |
|     8 |      +0.2814 |  0.0221 |
|     9 |      +0.3006 |  0.0142 |
|    10 |      +0.3573 |  0.0032 |
|    11 |      +0.3616 |  0.0029 |
|    12 |      +0.3324 |  0.0064 |
|    13 |      +0.3016 |  0.0138 |
|    14 |      +0.2415 |  0.0508 |
|    15 |      +0.2301 |  0.0631 |
|    16 |      +0.2340 |  0.0587 |
|    17 |      +0.1874 |  0.1319 |
|    18 |      +0.1537 |  0.2180 |
|    19 |      +0.1361 |  0.2760 |
|    20 |      +0.1362 |  0.2755 |
|    21 |      +0.1491 |  0.2321 |
|    22 |      +0.1242 |  0.3203 |
|    23 |      +0.1056 |  0.3986 |
|    24 |      +0.0987 |  0.4303 |
|    25 |      +0.0990 |  0.4292 |
|    26 |      +0.1014 |  0.4178 |
|    27 |      +0.1038 |  0.4071 |
|    28 |      +0.1033 |  0.4094 |
|    29 |      +0.1192 |  0.3403 |
|    30 |      +0.1078 |  0.3889 |
|    31 |      +0.0747 |  0.5509 |
|    32 |      +0.0830 |  0.5078 |

## RSA by Layer -- Agency

**Peak:** Layer 10, rho = +0.2925, p = 0.0172

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0927 |  0.4591 |
|     2 |      -0.0971 |  0.4380 |
|     3 |      -0.0783 |  0.5322 |
|     4 |      -0.0166 |  0.8949 |
|     5 |      -0.0765 |  0.5416 |
|     6 |      -0.0045 |  0.9713 |
|     7 |      +0.0111 |  0.9293 |
|     8 |      +0.2769 |  0.0244 |
|     9 |      +0.2839 |  0.0209 |
|    10 |      +0.2925 |  0.0172 |
|    11 |      +0.2568 |  0.0374 |
|    12 |      +0.1917 |  0.1231 |
|    13 |      +0.1980 |  0.1111 |
|    14 |      +0.1626 |  0.1921 |
|    15 |      +0.1742 |  0.1619 |
|    16 |      +0.1957 |  0.1154 |
|    17 |      +0.1721 |  0.1670 |
|    18 |      +0.1540 |  0.2169 |
|    19 |      +0.1427 |  0.2532 |
|    20 |      +0.1487 |  0.2334 |
|    21 |      +0.1837 |  0.1397 |
|    22 |      +0.1406 |  0.2602 |
|    23 |      +0.1298 |  0.2991 |
|    24 |      +0.0904 |  0.4705 |
|    25 |      +0.0908 |  0.4684 |
|    26 |      +0.0770 |  0.5388 |
|    27 |      +0.0615 |  0.6238 |
|    28 |      +0.0598 |  0.6336 |
|    29 |      +0.0463 |  0.7121 |
|    30 |      +0.0437 |  0.7275 |
|    31 |      +0.0199 |  0.8738 |
|    32 |      +0.0486 |  0.6981 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
