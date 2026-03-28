# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-03-26 14:16:59
**Tag:** `with_self`
**Include self:** True
**Model:** LLaMA-3-8B (Base)

---

## What is being tested

Does LLaMA-3-8B (Base)'s internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from LLaMA-3-8B (Base) in response to a simple prompt ("Think about {{entity}}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

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

**Peak:** Layer 8, rho = +0.3756, p = 0.0007

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0252 |  0.8265 |
|     2 |      +0.0727 |  0.5269 |
|     3 |      +0.1133 |  0.3235 |
|     4 |      +0.2517 |  0.0262 |
|     5 |      +0.2544 |  0.0246 |
|     6 |      +0.2609 |  0.0211 |
|     7 |      +0.3041 |  0.0068 |
|     8 |      +0.3756 |  0.0007 |
|     9 |      +0.3615 |  0.0011 |
|    10 |      +0.3500 |  0.0017 |
|    11 |      +0.3529 |  0.0015 |
|    12 |      +0.3418 |  0.0022 |
|    13 |      +0.3598 |  0.0012 |
|    14 |      +0.3335 |  0.0028 |
|    15 |      +0.3005 |  0.0075 |
|    16 |      +0.2948 |  0.0088 |
|    17 |      +0.2540 |  0.0248 |
|    18 |      +0.2306 |  0.0423 |
|    19 |      +0.2182 |  0.0549 |
|    20 |      +0.2230 |  0.0497 |
|    21 |      +0.2134 |  0.0606 |
|    22 |      +0.2257 |  0.0469 |
|    23 |      +0.2107 |  0.0641 |
|    24 |      +0.2064 |  0.0698 |
|    25 |      +0.1917 |  0.0926 |
|    26 |      +0.1887 |  0.0980 |
|    27 |      +0.1896 |  0.0963 |
|    28 |      +0.1645 |  0.1502 |
|    29 |      +0.1532 |  0.1806 |
|    30 |      +0.1568 |  0.1703 |
|    31 |      +0.1794 |  0.1161 |
|    32 |      +0.2994 |  0.0078 |

## RSA by Layer -- Experience

**Peak:** Layer 8, rho = +0.1794, p = 0.1161

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0910 |  0.4281 |
|     2 |      -0.0201 |  0.8610 |
|     3 |      +0.0041 |  0.9712 |
|     4 |      +0.0645 |  0.5745 |
|     5 |      +0.0799 |  0.4866 |
|     6 |      +0.0938 |  0.4142 |
|     7 |      +0.1235 |  0.2812 |
|     8 |      +0.1794 |  0.1161 |
|     9 |      +0.1738 |  0.1281 |
|    10 |      +0.1632 |  0.1533 |
|    11 |      +0.1550 |  0.1753 |
|    12 |      +0.1594 |  0.1634 |
|    13 |      +0.1671 |  0.1436 |
|    14 |      +0.1675 |  0.1427 |
|    15 |      +0.1384 |  0.2270 |
|    16 |      +0.1454 |  0.2040 |
|    17 |      +0.1137 |  0.3217 |
|    18 |      +0.0876 |  0.4457 |
|    19 |      +0.0734 |  0.5232 |
|    20 |      +0.0781 |  0.4969 |
|    21 |      +0.0691 |  0.5480 |
|    22 |      +0.0718 |  0.5325 |
|    23 |      +0.0571 |  0.6198 |
|    24 |      +0.0472 |  0.6814 |
|    25 |      +0.0403 |  0.7260 |
|    26 |      +0.0389 |  0.7356 |
|    27 |      +0.0391 |  0.7338 |
|    28 |      +0.0222 |  0.8470 |
|    29 |      +0.0119 |  0.9180 |
|    30 |      +0.0166 |  0.8854 |
|    31 |      +0.0240 |  0.8347 |
|    32 |      +0.1230 |  0.2833 |

## RSA by Layer -- Agency

**Peak:** Layer 8, rho = +0.2963, p = 0.0084

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0264 |  0.8185 |
|     2 |      +0.0412 |  0.7201 |
|     3 |      +0.0808 |  0.4819 |
|     4 |      +0.2228 |  0.0499 |
|     5 |      +0.2055 |  0.0711 |
|     6 |      +0.1988 |  0.0810 |
|     7 |      +0.2462 |  0.0298 |
|     8 |      +0.2963 |  0.0084 |
|     9 |      +0.2784 |  0.0136 |
|    10 |      +0.2685 |  0.0175 |
|    11 |      +0.2805 |  0.0129 |
|    12 |      +0.2598 |  0.0216 |
|    13 |      +0.2802 |  0.0130 |
|    14 |      +0.2310 |  0.0419 |
|    15 |      +0.2003 |  0.0787 |
|    16 |      +0.1818 |  0.1112 |
|    17 |      +0.1406 |  0.2195 |
|    18 |      +0.1208 |  0.2919 |
|    19 |      +0.1181 |  0.3031 |
|    20 |      +0.1165 |  0.3096 |
|    21 |      +0.1121 |  0.3285 |
|    22 |      +0.1201 |  0.2949 |
|    23 |      +0.1124 |  0.3270 |
|    24 |      +0.1159 |  0.3125 |
|    25 |      +0.0953 |  0.4067 |
|    26 |      +0.0984 |  0.3913 |
|    27 |      +0.1008 |  0.3801 |
|    28 |      +0.0826 |  0.4724 |
|    29 |      +0.0742 |  0.5183 |
|    30 |      +0.0803 |  0.4848 |
|    31 |      +0.1022 |  0.3731 |
|    32 |      +0.2007 |  0.0781 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity -- no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
