# Experiment 4: Entity Mind Perception Geometry

**Author:** Rachel C. Metzgar, Princeton University

---

## Motivation

Experiments 1-3 establish that LLaMA-2-13B-Chat (a) behaviorally adapts to partner identity labels, (b) linearly encodes partner identity in its activations, and (c) may share representational structure between its general semantic knowledge of humans/AIs and its conversational partner-identity signal. But all of these experiments treat partner identity as a **binary**: human vs. AI.

Human folk psychology is far richer. The landmark study by Gray, Gray, & Wegner (2007) — *"Dimensions of Mind Perception"* (Science, 315, 619) — showed that humans perceive minds along **two orthogonal dimensions**:

- **Experience**: The capacity to feel and sense (hunger, fear, pain, pleasure, rage, desire, consciousness, pride, embarrassment, joy). Factor loadings: .67-.97.
- **Agency**: The capacity to plan and act (self-control, morality, memory, emotion recognition, planning, communication, thought). Factor loadings: .73-.97.

~2,400 participants rated 13 diverse entities via pairwise comparisons on 18 mental capacities. Different entities occupy very different positions in this 2D space — God is high Agency but low Experience, babies are high Experience but low Agency, adult humans are high on both, robots are low on both.

### The Core Question

**Does the LLM's internal representational geometry over these entities mirror the human folk-psychological geometry?**

If yes, the model has internalized a structured folk psychology of mind — not just a binary human/AI distinction, but a continuous, multi-dimensional space that places different kinds of minds in the same relative positions as human perception does. We can then ask whether the dimensions organizing this space correspond to the same directions the model uses for conversational adaptation (Experiments 2-3).

---

## Key Theoretical Questions

1. **Does the model have a "mind space" that resembles human mind perception?** Compare the model's representational dissimilarity matrix (RDM) over entities to the human RDM from Gray et al. using representational similarity analysis (RSA).

2. **Is the model's mind space organized along Experience and Agency specifically?** PCA on entity representations — do the top components separate entities the way Experience and Agency do? Or does the model carve up mind-space differently?

3. **Can we find linear directions corresponding to Experience and Agency?** Train linear regressors to predict human Experience/Agency scores from entity activations. At which layers do these dimensions become linearly decodable?

4. **Does the model dissociate Experience and Agency?** Does it place God (high Agency, low Experience) and babies (high Experience, low Agency) in dissociated positions, like humans do? Or does it collapse them into a single "more/less mind" dimension?

5. **Do the Experience/Agency directions align with Experiment 2/3 probe directions?** Compute cosine similarity between the learned Experience/Agency directions and the partner-identity probe weight vectors. If they align, the model's conversational adaptation is organized along the same dimensions humans use to perceive minds.

6. **Where in the network does mind perception structure emerge?** Layer-by-layer RSA — at what depth does the entity geometry start resembling human mind perception?

7. **Extended entity set**: Where does the model place modern AI entities (ChatGPT, Siri, Alexa) and different kinds of agents in its mind space?

---

## Human Data: Gray et al. (2007) Factor Scores

Factor scores estimated from Figure 1 in Gray et al. (2007). The paper published these values only as a scatterplot, not a numerical table. **These values were estimated by AI reading the figure and SHOULD BE VERIFIED** — ideally by digitizing Figure 1 with WebPlotDigitizer or by contacting the authors (Kurt Gray, now at UNC Chapel Hill) for exact values. Scores adjusted to 0-1 scale.

```python
# CAUTION: AI-estimated values from figure. Verify before final analyses.
GRAY_ET_AL_ENTITIES = {
    # entity:          (Experience, Agency)
    "adult_woman":     (0.93, 0.91),
    "adult_man":       (0.91, 0.95),
    "you_self":        (0.97, 1.00),
    "girl_5yo":        (0.84, 0.62),
    "baby_5mo":        (0.71, 0.17),
    "fetus_7wk":       (0.17, 0.08),
    "pvs_patient":     (0.17, 0.10),
    "dead_woman":      (0.06, 0.07),
    "chimpanzee":      (0.63, 0.48),
    "dog":             (0.55, 0.35),
    "frog":            (0.25, 0.14),
    "god":             (0.20, 0.80),
    "robot_kismet":    (0.13, 0.22),
}
```

### Original Character Descriptions (Gray et al., 2007)

Used verbatim in prompts to match the original study:

| Entity | Description |
|--------|-------------|
| Green frog | "The Green Frog can be found throughout eastern North America. This classic 'pond frog' is medium-sized and green or bronze in color." |
| Charlie (dog) | "Charlie is a 3-year-old Springer spaniel and a beloved member of the Graham family." |
| Toby (chimpanzee) | "Toby is a two-year-old wild chimpanzee living at an outdoor laboratory in Uganda." |
| 7-week fetus | "At 7 weeks, a human fetus is almost half an inch long — roughly the size of a raspberry." |
| Nicholas (baby) | "Nicholas is a five-month-old baby." |
| Samantha (girl) | "Samantha is a five-year-old girl who lives with her parents and older sister Jennifer." |
| Sharon (adult woman) | "Sharon Harvey, 38, works at an advertising agency in Chicago." |
| Todd (adult man) | "Todd Billingsly is a thirty-year-old accountant who lives in New York City." |
| Gerald (PVS patient) | "Gerald Schiff has been in a persistent vegetative state (PVS) for the past six months. Although he has severe brain damage — Gerald does not appear to communicate with others or make purposeful movements — his basic bodily functions (such as breathing, sleeping, and circulation) are preserved." |
| Delores (dead woman) | "Delores Gleitman recently passed away at the age of 65." |
| God | "Many people believe that God is the creator of the universe and the ultimate source of knowledge, power, and love. However, please draw upon your own personal beliefs about God." |
| Kismet (robot) | "Kismet is part of a new class of 'sociable' robots that can engage people in natural interaction. To do this, Kismet perceives a variety of natural social signals from sound and sight, and delivers his own signals back to the human partner through gaze direction, facial expression, body posture, and vocal babbles." |

**Note:** "You (yourself)" is the 13th entity in Gray et al. but requires self-referential framing. We include it with adapted prompts.

### The 18 Mental Capacities

**Experience factor** (11 items): hunger (.97), fear (.93), pain (.89), pleasure (.85), rage (.78), desire (.76), personality (.72), consciousness (.71), pride (.71), embarrassment (.70), joy (.67)

**Agency factor** (7 items): self-control (.97), morality (.93), memory (.91), emotion recognition (.83), planning (.82), communication (.74), thought (.73)

---

## Related Literature

- **Malle (2019)** "How Many Dimensions of Mind Perception Really Are There?" — Found 3 dimensions: Affect, Moral/Mental Regulation, Reality Interaction. Published the MMP35 scale (20-item validated instrument). Extended entities: advanced robot, present-day robot, computer, God, alien, zombie, nation of Belgium, city council, Coca-Cola company.
- **Weisman, Dweck, & Markman (2017)** — Found 3 dimensions: Body, Heart, Mind. 40 mental capacities.
- **Takahashi, Ban, & Asada (2016)** — Japanese replication using semantic differential scales. 7 targets including robot, supercomputer, god.

---

## Analysis Pipeline

### Phase 1: Entity Representation Extraction (`1_extract_entity_representations.py`)

For each of the 13 Gray et al. entities:
- Write multiple prompts per entity using the original character descriptions (20 per entity, 260 total)
- Present each prompt to LLaMA-2-13B-Chat
- Extract last-token residual-stream activations across all 41 layers
- Average across prompts per entity to get stable entity representations
- Compute split-half stability for each entity

**Prompt design:** Each entity gets prompts that present the character description and ask the model to consider the entity (similar to Exp 3's concept elicitation approach). Prompts vary surface form to average out prompt-specific variance while keeping entity content constant.

**Output:**
```
data/entity_activations/
    {entity_name}/
        entity_activations.npz      # (n_prompts, n_layers, hidden_dim)
        entity_prompts.json          # prompt metadata
        mean_vector_per_layer.npz    # (n_layers, hidden_dim)
        split_half_stability.json
    entity_rdm_per_layer.npz         # (n_layers, 13, 13) model RDM
```

### Phase 2: Representational Similarity Analysis (`2_rsa_analysis.py`)

- Compute pairwise cosine distances between entity mean representations -> model RDM (13x13) at each layer
- Compute pairwise Euclidean distances in human Experience/Agency space -> human RDM (13x13)
- Spearman correlation between model RDM and human RDM at each layer
- Permutation testing for statistical significance (shuffle entity labels, recompute correlation, 10,000 iterations)
- Layer-by-layer RSA profile: where does mind perception structure emerge?

**Output:**
```
data/rsa_results/
    rsa_per_layer.json               # {layer: {rho, p_value, p_perm}}
    human_rdm.npz                    # ground-truth RDM from Gray et al.
    model_rdm_per_layer.npz          # model RDM at each layer
```

### Phase 3: Dimension Recovery (`3_dimension_recovery.py`)

**3a. PCA Analysis:**
- PCA on the 13 entity activation vectors at each layer
- Project entities into PC space
- Correlate PC1/PC2 with human Experience/Agency scores
- Visualize entity positions in the model's top-2 PC space vs. human Experience/Agency space

**3b. Linear Regression:**
- Train ridge regressors to predict Experience and Agency scores from entity activations
- Leave-one-entity-out cross-validation (13-fold)
- Evaluate prediction accuracy (R^2, Spearman correlation) at each layer
- Extract learned Experience and Agency weight vectors for alignment analysis (Phase 5)

**Output:**
```
data/dimension_recovery/
    pca_results_per_layer.npz        # explained variance, loadings, projections
    pca_human_correlation.json       # {layer: {pc1_exp_r, pc1_agency_r, ...}}
    regression_results.json          # {layer: {exp_r2, agency_r2, exp_rho, agency_rho}}
    experience_weights_per_layer.npz # learned Experience direction per layer
    agency_weights_per_layer.npz     # learned Agency direction per layer
```

### Phase 4: Extended Entities (Optional) (`4_extended_entities.py`)

- Add modern AI entities: ChatGPT, Siri, Alexa, self-driving car, calculator, thermostat
- Extract representations using the same pipeline as Phase 1
- Project into the Experience/Agency space learned in Phase 3
- Where does the model place different kinds of AI?

**Output:**
```
data/extended_entities/
    {entity_name}/
        entity_activations.npz
        entity_prompts.json
        mean_vector_per_layer.npz
    projected_scores.json            # predicted Experience/Agency for each extended entity
```

### Phase 5: Alignment with Conversational Probes (`5_probe_alignment.py`)

The key bridge analysis connecting mind perception geometry to conversational adaptation:

- Load Exp 2 probe weight vectors (control + reading probes)
- Load Exp 3 concept probe weight vectors
- Load Phase 3 Experience/Agency weight vectors
- Compute cosine similarity between:
  - Experience direction <-> Exp 2 control probe weights (per layer)
  - Agency direction <-> Exp 2 control probe weights (per layer)
  - Experience direction <-> Exp 2 reading probe weights (per layer)
  - Agency direction <-> Exp 2 reading probe weights (per layer)
  - Experience/Agency directions <-> Exp 3 concept dimensions (per layer)
- Permutation testing for significance

**Output:**
```
data/probe_alignment/
    exp2_alignment.json              # {layer: {exp_ctrl_cos, agency_ctrl_cos, exp_read_cos, agency_read_cos}}
    exp3_alignment.json              # {layer: {dim: {exp_cos, agency_cos}}}
    permutation_results.json
```

---

## Infrastructure

- **Model:** LLaMA-2-13B-Chat (same as Experiments 2-3)
- **Computing:** Princeton Scotty cluster, SLURM job scheduling
- **Activation extraction:** Forward pass with `output_hidden_states=True`, extract last-token residual stream (same pattern as `exp_3/1_elicit_concept_vectors.py`)
- **Environment:** `llama2_env` (activation extraction, GPU phases), `behavior_env` (analysis, CPU phases)
- **Reuse:** Shares `src/dataset.py` (prompt formatting), probe loading utilities, and analysis patterns from Exp 3

---

## Directory Structure

```
exp_4/
└── llama_exp_4-13B-chat/
    ├── README.md                              # This file
    ├── 1_extract_entity_representations.py    # Phase 1: activation extraction
    ├── 2_rsa_analysis.py                      # Phase 2: RSA
    ├── 3_dimension_recovery.py                # Phase 3: PCA + regression
    ├── 4_extended_entities.py                 # Phase 4: modern AI entities
    ├── 5_probe_alignment.py                   # Phase 5: bridge to Exps 2-3
    ├── entities/                              # Entity definitions + prompts
    │   ├── gray_entities.py                   # 13 Gray et al. entities + factor scores
    │   └── extended_entities.py               # Modern AI entities
    ├── src/                                   # Shared utilities (symlinked or imported from exp_3)
    ├── data/                                  # Output data
    │   ├── entity_activations/
    │   ├── rsa_results/
    │   ├── dimension_recovery/
    │   ├── extended_entities/
    │   └── probe_alignment/
    ├── results/                               # Figures and tables
    └── slurm/                                 # SLURM job scripts
```

---

## Connection to the Overall Narrative

Experiments 1 -> 2 -> 3 -> 4 build a progressive story:

1. **Exp 1 (Behavioral):** The model behaviorally adapts to partner identity labels
2. **Exp 2 (Probes):** This adaptation is mediated by specific linear directions in activation space
3. **Exp 3 (Concepts):** These directions share structure with the model's semantic knowledge about mental properties
4. **Exp 4 (Mind Perception Geometry):** The model's representation of diverse entities mirrors human folk psychology of mind perception, organized along Experience and Agency — and these dimensions correspond to the same directions involved in conversational adaptation

This connects machine social cognition to decades of human social cognition research and tests whether the model has internalized a continuous, structured folk psychology rather than just a binary human/AI switch.

---

## References

Gray, H. M., Gray, K., & Wegner, D. M. (2007). Dimensions of mind perception. *Science*, 315(5812), 619. https://doi.org/10.1126/science.1134475

Malle, B. F. (2019). How many dimensions of mind perception really are there? In *Proceedings of the 41st Annual Meeting of the Cognitive Science Society* (pp. 2268-2274).

Weisman, K., Dweck, C. S., & Markman, E. M. (2017). Rethinking people's conceptions of mental life. *Proceedings of the National Academy of Sciences*, 114(43), 11374-11379.

Takahashi, H., Ban, M., & Asada, M. (2016). Semantic differential scale method can reveal multi-dimensional aspects of mind perception. *Frontiers in Psychology*, 7, 1717.
