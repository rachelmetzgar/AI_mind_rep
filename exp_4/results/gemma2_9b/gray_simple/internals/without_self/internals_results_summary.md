# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:54:20
**Tag:** `without_self`
**Include self:** False
**Model:** Gemma-2-9B (Base)

---

## What is being tested

Does Gemma-2-9B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Gemma-2-9B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 18, rho = +0.2807, p = 0.0225

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0558 |  0.6563 |
|     2 |      -0.0857 |  0.4939 |
|     3 |      -0.1044 |  0.4044 |
|     4 |      +0.0495 |  0.6933 |
|     5 |      +0.0202 |  0.8722 |
|     6 |      +0.1418 |  0.2561 |
|     7 |      +0.1557 |  0.2119 |
|     8 |      +0.1855 |  0.1360 |
|     9 |      +0.2282 |  0.0654 |
|    10 |      +0.2133 |  0.0855 |
|    11 |      +0.1446 |  0.2468 |
|    12 |      +0.2140 |  0.0844 |
|    13 |      +0.2331 |  0.0596 |
|    14 |      +0.2529 |  0.0405 |
|    15 |      +0.2748 |  0.0255 |
|    16 |      +0.2415 |  0.0508 |
|    17 |      +0.2691 |  0.0289 |
|    18 |      +0.2807 |  0.0225 |
|    19 |      +0.2385 |  0.0538 |
|    20 |      +0.2184 |  0.0781 |
|    21 |      +0.1864 |  0.1340 |
|    22 |      +0.1492 |  0.2319 |
|    23 |      +0.1163 |  0.3523 |
|    24 |      +0.0917 |  0.4638 |
|    25 |      +0.0967 |  0.4398 |
|    26 |      +0.0792 |  0.5272 |
|    27 |      +0.0484 |  0.6998 |
|    28 |      +0.0506 |  0.6865 |
|    29 |      +0.0423 |  0.7359 |
|    30 |      +0.0496 |  0.6926 |
|    31 |      +0.0596 |  0.6345 |
|    32 |      +0.0575 |  0.6465 |
|    33 |      +0.0464 |  0.7116 |
|    34 |      +0.0158 |  0.9000 |
|    35 |      +0.0192 |  0.8782 |
|    36 |      +0.0295 |  0.8141 |
|    37 |      +0.0185 |  0.8827 |
|    38 |      -0.0030 |  0.9810 |
|    39 |      +0.0174 |  0.8898 |
|    40 |      -0.0268 |  0.8310 |
|    41 |      -0.0521 |  0.6777 |
|    42 |      +0.0202 |  0.8722 |

## RSA by Layer -- Experience

**Peak:** Layer 18, rho = +0.1270, p = 0.3097

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1338 |  0.2843 |
|     2 |      -0.1262 |  0.3125 |
|     3 |      -0.1448 |  0.2462 |
|     4 |      -0.0554 |  0.6589 |
|     5 |      -0.0569 |  0.6497 |
|     6 |      -0.0173 |  0.8903 |
|     7 |      -0.0151 |  0.9042 |
|     8 |      +0.0118 |  0.9248 |
|     9 |      +0.0577 |  0.6454 |
|    10 |      +0.0502 |  0.6891 |
|    11 |      +0.0158 |  0.8995 |
|    12 |      +0.0507 |  0.6860 |
|    13 |      +0.0660 |  0.5987 |
|    14 |      +0.0657 |  0.6001 |
|    15 |      +0.1136 |  0.3636 |
|    16 |      +0.1029 |  0.4112 |
|    17 |      +0.1212 |  0.3324 |
|    18 |      +0.1270 |  0.3097 |
|    19 |      +0.1040 |  0.4060 |
|    20 |      +0.1095 |  0.3814 |
|    21 |      +0.0902 |  0.4712 |
|    22 |      +0.0668 |  0.5940 |
|    23 |      +0.0436 |  0.7283 |
|    24 |      +0.0270 |  0.8296 |
|    25 |      +0.0306 |  0.8074 |
|    26 |      +0.0119 |  0.9244 |
|    27 |      -0.0206 |  0.8698 |
|    28 |      -0.0225 |  0.8579 |
|    29 |      -0.0220 |  0.8608 |
|    30 |      -0.0201 |  0.8729 |
|    31 |      -0.0038 |  0.9761 |
|    32 |      -0.0130 |  0.9173 |
|    33 |      -0.0246 |  0.8443 |
|    34 |      -0.0494 |  0.6936 |
|    35 |      -0.0524 |  0.6759 |
|    36 |      -0.0430 |  0.7318 |
|    37 |      -0.0434 |  0.7293 |
|    38 |      -0.0577 |  0.6452 |
|    39 |      -0.0455 |  0.7167 |
|    40 |      -0.0679 |  0.5878 |
|    41 |      -0.0815 |  0.5151 |
|    42 |      -0.0373 |  0.7665 |

## RSA by Layer -- Agency

**Peak:** Layer 15, rho = +0.1448, p = 0.2462

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      +0.0140 |  0.9109 |
|     2 |      -0.0971 |  0.4380 |
|     3 |      -0.1208 |  0.3339 |
|     4 |      -0.0295 |  0.8143 |
|     5 |      -0.0685 |  0.5848 |
|     6 |      +0.0629 |  0.6157 |
|     7 |      +0.0698 |  0.5778 |
|     8 |      +0.0938 |  0.4540 |
|     9 |      +0.1131 |  0.3661 |
|    10 |      +0.0908 |  0.4686 |
|    11 |      +0.0228 |  0.8560 |
|    12 |      +0.0982 |  0.4327 |
|    13 |      +0.1085 |  0.3857 |
|    14 |      +0.1289 |  0.3024 |
|    15 |      +0.1448 |  0.2462 |
|    16 |      +0.1040 |  0.4058 |
|    17 |      +0.1283 |  0.3046 |
|    18 |      +0.1403 |  0.2613 |
|    19 |      +0.1102 |  0.3784 |
|    20 |      +0.0901 |  0.4719 |
|    21 |      +0.0553 |  0.6591 |
|    22 |      +0.0206 |  0.8698 |
|    23 |      -0.0125 |  0.9209 |
|    24 |      -0.0325 |  0.7956 |
|    25 |      -0.0302 |  0.8100 |
|    26 |      -0.0396 |  0.7525 |
|    27 |      -0.0592 |  0.6367 |
|    28 |      -0.0631 |  0.6145 |
|    29 |      -0.0732 |  0.5590 |
|    30 |      -0.0681 |  0.5868 |
|    31 |      -0.0596 |  0.6345 |
|    32 |      -0.0539 |  0.6673 |
|    33 |      -0.0675 |  0.5904 |
|    34 |      -0.0833 |  0.5059 |
|    35 |      -0.0807 |  0.5194 |
|    36 |      -0.0743 |  0.5535 |
|    37 |      -0.0839 |  0.5032 |
|    38 |      -0.0995 |  0.4269 |
|    39 |      -0.0800 |  0.5234 |
|    40 |      -0.1141 |  0.3617 |
|    41 |      -0.1374 |  0.2712 |
|    42 |      -0.0856 |  0.4943 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
