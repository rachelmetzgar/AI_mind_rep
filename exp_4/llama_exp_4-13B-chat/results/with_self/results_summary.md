# Experiment 4, Phase 1: Entity Representation Extraction

**Run:** 2026-02-18 13:39:16
**Tag:** `with_self`
**Include self:** True

---

## What is being tested

Does LLaMA-2-13B-Chat's internal representational geometry over diverse entities mirror the human folk-psychological geometry from Gray, Gray, & Wegner (2007)?

Gray et al. had ~2,400 human participants rate 13 entities (from dead woman to God) on 18 mental capacities. Factor analysis revealed two dimensions: **Experience** (capacity to feel) and **Agency** (capacity to act). Each entity has a position in this 2D space.

We ask: does the model's internal representation of these entities have a similar geometric structure?

## Method: Representational Similarity Analysis (RSA)

RSA compares two distance matrices (Kriegeskorte et al., 2008). Instead of comparing representations directly (which would require them to be in the same space), we compare the *pattern of distances* between entities.

**Inputs:**

- **Model RDM**: For each of 13 entities, we extract the last-token residual-stream activation from LLaMA-2-13B-Chat in response to a simple prompt ("Think about {entity}."). We compute pairwise **cosine distance** (1 - cosine similarity) between all entity pairs -> a 13x13 representational dissimilarity matrix (RDM). Computed separately at each of 41 layers (layer 0 = embedding, layers 1-40 = transformer blocks).

- **Human RDM**: From Gray et al. (2007) Experience and Agency factor scores, we compute pairwise **Euclidean distance** between all entity pairs in 2D (Experience, Agency) space -> a 13x13 RDM.

**Test:** Spearman rank correlation between the upper triangles of the model RDM and human RDM (78 unique entity pairs). Computed at every layer to track where in the network mind-perception structure emerges.

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

## Results: RSA by Layer

**Peak:** Layer 8, rho = +0.2318, p = 0.0411

| Layer | Spearman rho | p-value |
|------:|-------------:|--------:|
|     0 |          nan |     nan |
|     1 |      -0.0550 |  0.6326 |
|     2 |      -0.0155 |  0.8927 |
|     3 |      +0.0237 |  0.8366 |
|     4 |      +0.0358 |  0.7557 |
|     5 |      +0.1072 |  0.3502 |
|     6 |      +0.1254 |  0.2739 |
|     7 |      +0.1788 |  0.1173 |
|     8 |      +0.2318 |  0.0411 |
|     9 |      +0.2247 |  0.0479 |
|    10 |      +0.2237 |  0.0490 |
|    11 |      +0.1676 |  0.1425 |
|    12 |      +0.1821 |  0.1106 |
|    13 |      +0.1648 |  0.1494 |
|    14 |      +0.1764 |  0.1223 |
|    15 |      +0.1719 |  0.1323 |
|    16 |      +0.1702 |  0.1363 |
|    17 |      +0.1589 |  0.1648 |
|    18 |      +0.1541 |  0.1781 |
|    19 |      +0.1495 |  0.1913 |
|    20 |      +0.1467 |  0.2000 |
|    21 |      +0.1411 |  0.2180 |
|    22 |      +0.1413 |  0.2171 |
|    23 |      +0.1395 |  0.2231 |
|    24 |      +0.1382 |  0.2275 |
|    25 |      +0.1357 |  0.2363 |
|    26 |      +0.1391 |  0.2244 |
|    27 |      +0.1361 |  0.2347 |
|    28 |      +0.1325 |  0.2476 |
|    29 |      +0.1281 |  0.2637 |
|    30 |      +0.1329 |  0.2462 |
|    31 |      +0.1324 |  0.2478 |
|    32 |      +0.1328 |  0.2465 |
|    33 |      +0.1309 |  0.2531 |
|    34 |      +0.1339 |  0.2424 |
|    35 |      +0.1364 |  0.2339 |
|    36 |      +0.1352 |  0.2379 |
|    37 |      +0.1294 |  0.2588 |
|    38 |      +0.1323 |  0.2481 |
|    39 |      +0.1296 |  0.2583 |
|    40 |      +0.1228 |  0.2842 |

## Notes

- Factor scores from Gray et al. (2007) were estimated by AI from Figure 1 and should be verified before final analyses.
- One prompt per entity — no averaging across prompt variants. Reliability cannot be assessed with this design; robustness can be checked by rerunning with a different template.
- Layer 0 (embedding) typically produces constant cosine distances (all activations have similar geometry before any transformer processing), yielding NaN correlation.
