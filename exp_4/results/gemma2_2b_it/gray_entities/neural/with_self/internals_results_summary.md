# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:53:30
**Tag:** `with_self`
**Include self:** True
**Model:** Gemma-2-2B-IT

---

## What is being tested

Does Gemma-2-2B-IT's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from Gemma-2-2B-IT in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 9, rho = +0.4387, p = 0.0001

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0575 |  0.6172 |
|     2 |      -0.0134 |  0.9070 |
|     3 |      +0.0954 |  0.4063 |
|     4 |      +0.1481 |  0.1957 |
|     5 |      +0.1815 |  0.1118 |
|     6 |      +0.2422 |  0.0326 |
|     7 |      +0.2843 |  0.0117 |
|     8 |      +0.3850 |  0.0005 |
|     9 |      +0.4387 |  0.0001 |
|    10 |      +0.3588 |  0.0013 |
|    11 |      +0.3610 |  0.0012 |
|    12 |      +0.3258 |  0.0036 |
|    13 |      +0.3768 |  0.0007 |
|    14 |      +0.4142 |  0.0002 |
|    15 |      +0.3493 |  0.0017 |
|    16 |      +0.3194 |  0.0044 |
|    17 |      +0.3342 |  0.0028 |
|    18 |      +0.3223 |  0.0040 |
|    19 |      +0.3298 |  0.0032 |
|    20 |      +0.3207 |  0.0042 |
|    21 |      +0.3209 |  0.0042 |
|    22 |      +0.2969 |  0.0083 |
|    23 |      +0.2983 |  0.0080 |
|    24 |      +0.2849 |  0.0115 |
|    25 |      +0.2663 |  0.0184 |
|    26 |      +0.2621 |  0.0204 |

## RSA by Layer -- Experience

**Peak:** Layer 14, rho = +0.3071, p = 0.0062

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1110 |  0.3334 |
|     2 |      -0.0628 |  0.5849 |
|     3 |      +0.0035 |  0.9761 |
|     4 |      +0.0379 |  0.7416 |
|     5 |      +0.0313 |  0.7859 |
|     6 |      +0.0833 |  0.4684 |
|     7 |      +0.1070 |  0.3511 |
|     8 |      +0.1741 |  0.1274 |
|     9 |      +0.2864 |  0.0110 |
|    10 |      +0.2066 |  0.0695 |
|    11 |      +0.2311 |  0.0418 |
|    12 |      +0.2305 |  0.0423 |
|    13 |      +0.2803 |  0.0129 |
|    14 |      +0.3071 |  0.0062 |
|    15 |      +0.2722 |  0.0159 |
|    16 |      +0.2382 |  0.0357 |
|    17 |      +0.2525 |  0.0258 |
|    18 |      +0.2753 |  0.0147 |
|    19 |      +0.2970 |  0.0083 |
|    20 |      +0.2902 |  0.0100 |
|    21 |      +0.2892 |  0.0102 |
|    22 |      +0.2832 |  0.0120 |
|    23 |      +0.2887 |  0.0104 |
|    24 |      +0.2778 |  0.0138 |
|    25 |      +0.2597 |  0.0217 |
|    26 |      +0.2568 |  0.0232 |

## RSA by Layer -- Agency

**Peak:** Layer 9, rho = +0.3678, p = 0.0009

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0353 |  0.7590 |
|     2 |      -0.0781 |  0.4967 |
|     3 |      +0.0393 |  0.7326 |
|     4 |      +0.1019 |  0.3747 |
|     5 |      +0.1458 |  0.2028 |
|     6 |      +0.2153 |  0.0584 |
|     7 |      +0.2627 |  0.0201 |
|     8 |      +0.3529 |  0.0015 |
|     9 |      +0.3678 |  0.0009 |
|    10 |      +0.3078 |  0.0061 |
|    11 |      +0.3005 |  0.0075 |
|    12 |      +0.2544 |  0.0246 |
|    13 |      +0.3062 |  0.0064 |
|    14 |      +0.3345 |  0.0028 |
|    15 |      +0.2788 |  0.0135 |
|    16 |      +0.2587 |  0.0222 |
|    17 |      +0.2473 |  0.0290 |
|    18 |      +0.2225 |  0.0502 |
|    19 |      +0.2158 |  0.0577 |
|    20 |      +0.2011 |  0.0775 |
|    21 |      +0.2004 |  0.0785 |
|    22 |      +0.1672 |  0.1434 |
|    23 |      +0.1674 |  0.1430 |
|    24 |      +0.1512 |  0.1864 |
|    25 |      +0.1362 |  0.2345 |
|    26 |      +0.1224 |  0.2859 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
