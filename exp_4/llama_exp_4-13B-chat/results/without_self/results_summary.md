# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-02-18 13:38:43
**Tag:** `without_self`
**Include self:** False

---

## What is being tested

Does LLaMA-2-13B-Chat's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 12 entities, we extract the last-token residual-stream activation from LLaMA-2-13B-Chat in response to a simple prompt ("Think about {entity}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 12x12 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

- **Human RDM**: From Gray et al. (2007) Experience and Agency factor scores, we compute pairwise **Euclidean distance** between all entity pairs in 2D (Experience, Agency) space -> a 12x12 RDM.

**Test:** Spearman rank correlation between the upper triangles of the model RDM and human RDM (66 unique entity pairs). Computed at every layer to track where in the network mind-perception structure emerges.

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

## Results: RSA by Layer

**Peak:** Layer 15, rho = +0.2205, p = 0.0752

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.1065 |  0.3946 |
|     2 |      -0.0887 |  0.4786 |
|     3 |      -0.0463 |  0.7121 |
|     4 |      -0.0273 |  0.8276 |
|     5 |      +0.0317 |  0.8007 |
|     6 |      +0.0577 |  0.6453 |
|     7 |      +0.1300 |  0.2981 |
|     8 |      +0.1796 |  0.1491 |
|     9 |      +0.1610 |  0.1965 |
|    10 |      +0.1608 |  0.1972 |
|    11 |      +0.1581 |  0.2049 |
|    12 |      +0.1820 |  0.1436 |
|    13 |      +0.1605 |  0.1981 |
|    14 |      +0.2193 |  0.0769 |
|    15 |      +0.2205 |  0.0752 |
|    16 |      +0.2185 |  0.0779 |
|    17 |      +0.1877 |  0.1313 |
|    18 |      +0.1782 |  0.1523 |
|    19 |      +0.1754 |  0.1589 |
|    20 |      +0.1694 |  0.1738 |
|    21 |      +0.1607 |  0.1974 |
|    22 |      +0.1629 |  0.1912 |
|    23 |      +0.1559 |  0.2114 |
|    24 |      +0.1541 |  0.2166 |
|    25 |      +0.1508 |  0.2267 |
|    26 |      +0.1542 |  0.2164 |
|    27 |      +0.1496 |  0.2305 |
|    28 |      +0.1457 |  0.2430 |
|    29 |      +0.1441 |  0.2484 |
|    30 |      +0.1454 |  0.2441 |
|    31 |      +0.1435 |  0.2503 |
|    32 |      +0.1444 |  0.2474 |
|    33 |      +0.1438 |  0.2495 |
|    34 |      +0.1478 |  0.2362 |
|    35 |      +0.1537 |  0.2180 |
|    36 |      +0.1489 |  0.2327 |
|    37 |      +0.1414 |  0.2573 |
|    38 |      +0.1460 |  0.2422 |
|    39 |      +0.1380 |  0.2692 |
|    40 |      +0.1468 |  0.2396 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity — no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
