# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:53:29
**Tag:** `without_self`
**Include self:** False
**Model:** Gemma-2-2B-IT

---

## What is being tested

Does Gemma-2-2B-IT's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Gemma-2-2B-IT in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 9, rho = +0.4136, p = 0.0006

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1009 |  0.4202 |
|     2 |      -0.0741 |  0.5542 |
|     3 |      -0.0234 |  0.8521 |
|     4 |      +0.0761 |  0.5437 |
|     5 |      +0.0818 |  0.5135 |
|     6 |      +0.1707 |  0.1707 |
|     7 |      +0.2175 |  0.0794 |
|     8 |      +0.3662 |  0.0025 |
|     9 |      +0.4136 |  0.0006 |
|    10 |      +0.2977 |  0.0152 |
|    11 |      +0.3060 |  0.0125 |
|    12 |      +0.2597 |  0.0352 |
|    13 |      +0.3233 |  0.0081 |
|    14 |      +0.3807 |  0.0016 |
|    15 |      +0.2892 |  0.0185 |
|    16 |      +0.2425 |  0.0498 |
|    17 |      +0.2620 |  0.0336 |
|    18 |      +0.2382 |  0.0541 |
|    19 |      +0.2704 |  0.0281 |
|    20 |      +0.2865 |  0.0197 |
|    21 |      +0.3050 |  0.0128 |
|    22 |      +0.2976 |  0.0152 |
|    23 |      +0.3199 |  0.0088 |
|    24 |      +0.3036 |  0.0132 |
|    25 |      +0.2810 |  0.0223 |
|    26 |      +0.3078 |  0.0119 |

## RSA by Layer -- Experience

**Peak:** Layer 14, rho = +0.2956, p = 0.0160

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1341 |  0.2832 |
|     2 |      -0.0905 |  0.4700 |
|     3 |      -0.0762 |  0.5430 |
|     4 |      +0.0087 |  0.9447 |
|     5 |      -0.0361 |  0.7735 |
|     6 |      +0.0585 |  0.6405 |
|     7 |      +0.0791 |  0.5276 |
|     8 |      +0.1558 |  0.2116 |
|     9 |      +0.2725 |  0.0268 |
|    10 |      +0.1555 |  0.2126 |
|    11 |      +0.1956 |  0.1154 |
|    12 |      +0.1972 |  0.1124 |
|    13 |      +0.2594 |  0.0354 |
|    14 |      +0.2956 |  0.0160 |
|    15 |      +0.2488 |  0.0440 |
|    16 |      +0.1917 |  0.1231 |
|    17 |      +0.1979 |  0.1112 |
|    18 |      +0.2064 |  0.0964 |
|    19 |      +0.2413 |  0.0510 |
|    20 |      +0.2487 |  0.0441 |
|    21 |      +0.2612 |  0.0342 |
|    22 |      +0.2649 |  0.0316 |
|    23 |      +0.2893 |  0.0185 |
|    24 |      +0.2786 |  0.0235 |
|    25 |      +0.2607 |  0.0345 |
|    26 |      +0.2873 |  0.0193 |

## RSA by Layer -- Agency

**Peak:** Layer 9, rho = +0.2790, p = 0.0233

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1083 |  0.3866 |
|     2 |      -0.1964 |  0.1141 |
|     3 |      -0.1476 |  0.2370 |
|     4 |      -0.0603 |  0.6307 |
|     5 |      -0.0342 |  0.7849 |
|     6 |      +0.0600 |  0.6323 |
|     7 |      +0.1107 |  0.3760 |
|     8 |      +0.2635 |  0.0326 |
|     9 |      +0.2790 |  0.0233 |
|    10 |      +0.1913 |  0.1240 |
|    11 |      +0.1866 |  0.1335 |
|    12 |      +0.1238 |  0.3222 |
|    13 |      +0.1944 |  0.1178 |
|    14 |      +0.2408 |  0.0514 |
|    15 |      +0.1556 |  0.2122 |
|    16 |      +0.1301 |  0.2978 |
|    17 |      +0.1264 |  0.3118 |
|    18 |      +0.1026 |  0.4121 |
|    19 |      +0.1293 |  0.3006 |
|    20 |      +0.1404 |  0.2607 |
|    21 |      +0.1628 |  0.1914 |
|    22 |      +0.1486 |  0.2338 |
|    23 |      +0.1665 |  0.1816 |
|    24 |      +0.1467 |  0.2399 |
|    25 |      +0.1241 |  0.3206 |
|    26 |      +0.1356 |  0.2778 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
