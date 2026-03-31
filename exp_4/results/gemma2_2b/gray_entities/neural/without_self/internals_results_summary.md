# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-29 10:53:47
**Tag:** `without_self`
**Include self:** False
**Model:** Gemma-2-2B (Base)

---

## What is being tested

Does Gemma-2-2B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from Gemma-2-2B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 7, rho = +0.1545, p = 0.2156

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1063 |  0.3957 |
|     2 |      -0.1327 |  0.2880 |
|     3 |      -0.0654 |  0.6016 |
|     4 |      +0.0404 |  0.7474 |
|     5 |      +0.1160 |  0.3535 |
|     6 |      +0.1350 |  0.2797 |
|     7 |      +0.1545 |  0.2156 |
|     8 |      +0.0979 |  0.4341 |
|     9 |      +0.1164 |  0.3521 |
|    10 |      +0.0687 |  0.5839 |
|    11 |      +0.1203 |  0.3361 |
|    12 |      +0.1420 |  0.2554 |
|    13 |      +0.1443 |  0.2475 |
|    14 |      +0.1443 |  0.2475 |
|    15 |      +0.1230 |  0.3251 |
|    16 |      +0.1016 |  0.4170 |
|    17 |      +0.0744 |  0.5528 |
|    18 |      +0.0629 |  0.6161 |
|    19 |      +0.0744 |  0.5526 |
|    20 |      +0.0686 |  0.5841 |
|    21 |      +0.0516 |  0.6806 |
|    22 |      +0.0689 |  0.5827 |
|    23 |      +0.0600 |  0.6322 |
|    24 |      +0.0391 |  0.7555 |
|    25 |      +0.0320 |  0.7984 |
|    26 |      -0.0160 |  0.8987 |

## RSA by Layer -- Experience

**Peak:** Layer 6, rho = +0.0505, p = 0.6870

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1496 |  0.2305 |
|     2 |      -0.1682 |  0.1771 |
|     3 |      -0.1092 |  0.3827 |
|     4 |      -0.0782 |  0.5326 |
|     5 |      -0.0093 |  0.9411 |
|     6 |      +0.0505 |  0.6870 |
|     7 |      +0.0429 |  0.7325 |
|     8 |      -0.0016 |  0.9895 |
|     9 |      +0.0027 |  0.9830 |
|    10 |      -0.0144 |  0.9087 |
|    11 |      +0.0236 |  0.8506 |
|    12 |      +0.0259 |  0.8362 |
|    13 |      +0.0384 |  0.7594 |
|    14 |      +0.0395 |  0.7531 |
|    15 |      +0.0494 |  0.6934 |
|    16 |      +0.0311 |  0.8040 |
|    17 |      +0.0106 |  0.9329 |
|    18 |      -0.0033 |  0.9789 |
|    19 |      +0.0061 |  0.9615 |
|    20 |      -0.0015 |  0.9906 |
|    21 |      -0.0193 |  0.8775 |
|    22 |      -0.0244 |  0.8456 |
|    23 |      -0.0295 |  0.8138 |
|    24 |      -0.0433 |  0.7298 |
|    25 |      -0.0411 |  0.7431 |
|    26 |      -0.0579 |  0.6443 |

## RSA by Layer -- Agency

**Peak:** Layer 7, rho = +0.0710, p = 0.5711

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0957 |  0.4446 |
|     2 |      -0.1280 |  0.3057 |
|     3 |      -0.0731 |  0.5597 |
|     4 |      -0.0278 |  0.8247 |
|     5 |      +0.0425 |  0.7347 |
|     6 |      +0.0280 |  0.8236 |
|     7 |      +0.0710 |  0.5711 |
|     8 |      +0.0019 |  0.9882 |
|     9 |      +0.0193 |  0.8780 |
|    10 |      -0.0178 |  0.8872 |
|    11 |      +0.0318 |  0.7998 |
|    12 |      +0.0524 |  0.6759 |
|    13 |      +0.0345 |  0.7833 |
|    14 |      +0.0378 |  0.7634 |
|    15 |      +0.0013 |  0.9919 |
|    16 |      -0.0279 |  0.8240 |
|    17 |      -0.0519 |  0.6792 |
|    18 |      -0.0604 |  0.6298 |
|    19 |      -0.0522 |  0.6770 |
|    20 |      -0.0549 |  0.6616 |
|    21 |      -0.0599 |  0.6326 |
|    22 |      -0.0396 |  0.7524 |
|    23 |      -0.0416 |  0.7399 |
|    24 |      -0.0618 |  0.6220 |
|    25 |      -0.0745 |  0.5520 |
|    26 |      -0.1144 |  0.3606 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
