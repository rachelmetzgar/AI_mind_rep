# Experiment Specification: Multi-Agent Belief Propagation in LLM Internal Representations

## For Claude Code on Scotty Cluster

---

## 1. Research Question

**Primary question:** When an LLM processes a narrative involving multiple agents who communicate beliefs to each other — and those beliefs diverge due to an information update that reaches only some agents — does the model's internal representational geometry of agent belief states mirror the ground-truth epistemic geometry (who-believes-what), in a way that cannot be explained by surface-level statistical features like syntactic proximity, co-occurrence, or communication structure?

**The analogy:** Gurnee & Tegmark (2023) showed that LLMs develop internal representations of cities whose pairwise distances reflect actual geographic distances — structure that can't come from syntax alone. This experiment asks the equivalent for social cognition: do LLMs develop internal representations of agents' beliefs whose pairwise distances reflect actual epistemic similarity — structure that can't come from syntax alone?

**Why this matters:** Every existing mechanistic interpretability study on ToM (Zhu et al. 2024, Bortoletto et al. 2024, Prakash et al. 2025, Li et al. 2025) is limited to 2 agents with binary true/false belief. Nobody has studied multi-agent belief propagation at the representational level, and nobody has applied RSA/RDM to agent belief states in LLM internals. This is a genuine gap confirmed by comprehensive literature search.

---

## 2. Experimental Design

### 2.1 Stimuli

The stimuli are narratives about 4 agents (A, B, C, D) who:
1. Learn a fact (e.g., "the red book is on the kitchen table")
2. Communicate that fact to each other through a network
3. Experience a world-state change (the object moves) that only some agents witness
4. Optionally hear about the change from the witness

**Three network topologies:**
- **Chain:** A → B → C → D (linear)
- **Fork:** A → B, A → C, A → D (broadcast)
- **Diamond:** A → B, A → C, B → D, C → D (convergent)

**Override conditions per topology:** 3 conditions where the object moves and different agents witness/learn about it, plus 1 no-override control = 4 conditions per topology.

**Instantiations:** 8 per condition (different agent names, objects, locations). Names and scenarios rotate across instantiations.

**Total: 3 topologies × 4 conditions × 8 instantiations = 96 narratives.**

The stimulus generation code is already written: `belief_propagation_stimuli.py` produces a JSON file with all 96 narratives, including ground-truth expected beliefs, epistemic RDMs, comprehension probes, and extraction sentences.

### 2.2 What Makes This Experiment Strong

The key conditions create epistemic geometries that **dissociate from surface features:**

- `fork_override_D_tells_A`: Agents A and D agree (both know new location), agents B and C agree (both believe old location). But A and D never communicated directly about the change in the original topology — D's update to A creates an epistemic alignment that doesn't follow from communication structure.
- `chain_override_C_tells_D` vs `diamond_override_B_tells_D`: Same belief partition (2 agents know new, 2 don't) but different communication structure. If representations track epistemic state rather than communication links, the RDMs should be similar despite different topologies.

---

## 3. Confounds and Controls

### 3.1 Name Confound
**Problem:** Agent names might become associated with belief states if the same name always appears in the same position.
**Control:** 8 different name sets rotate across instantiations. Each structural condition appears with all 8 name sets. No name is consistently paired with a particular belief state.

### 3.2 Position/Recency Confound
**Problem:** The model might represent agents based on where they appeared in the text (early vs. late tokens) rather than their epistemic state.
**Control:** The extraction sentence places all 4 agent names in a fixed syntactic frame ("Now, Alice, Bob, Carol, and Dave are all gathered in the same room.") at the end of every narrative. Position in this sentence is constant across conditions for the same name set.

### 3.3 Communication-Structure Confound
**Problem:** Agents who communicated with each other might cluster in representation space because of syntactic co-mention, not shared beliefs.
**Control:** The `fork_override_D_tells_A` condition creates epistemic agreement between agents who did NOT communicate in the original network. If the model tracks epistemic state, A and D cluster; if it tracks communication links, A clusters with B/C/D (since A told all of them).
**Additional control:** Compare RDMs from `chain_override_C_tells_D` and `diamond_override_C_tells_D` — same epistemic geometry, different communication structure. If RSA correlation is high across these, communication structure isn't driving the geometry.

### 3.4 Object/Location Content Confound
**Problem:** Specific objects or locations might drive representational differences.
**Control:** 8 different scenarios (objects, locations, settings) rotate across instantiations. Any content-specific signal averages out when aggregating within structural conditions.

### 3.5 Narrative Length Confound
**Problem:** Override vs. no-override narratives differ in length (override narratives have extra sentences).
**Control:** No-override narratives include a filler paragraph of comparable length ("Some time passes. Everyone goes about their activities…"). Length is approximately matched.

### 3.6 Baseline RDM Controls
**Analysis control:** Compare the model's RDM to THREE candidate RDMs:
1. **Epistemic RDM** (ground truth): pairwise distance = 0 if agents share belief, 1 if they disagree. THE PREDICTION.
2. **Communication RDM**: pairwise distance = 0 if agents communicated directly, 1 if not. The surface-statistics alternative.
3. **Position RDM**: pairwise distance = |position_i - position_j| in the extraction sentence. The positional baseline.

If the model's RDM correlates with the epistemic RDM significantly better than with communication or position RDMs, that's the key finding.

---

## 4. Pipeline Overview

The experiment has 6 phases, each producing outputs consumed by the next:

```
Phase 0: Stimulus Generation
    → data/stimuli/belief_propagation_stimuli.json

Phase 1: Behavioral Validation (can the model answer belief questions?)
    → data/behavioral/behavioral_results.json

Phase 2: Activation Extraction (hidden states at agent name tokens)
    → data/activations/{narrative_id}_layer{L}.pt

Phase 3: RDM Construction (pairwise distances between agent representations)
    → data/rdms/model_rdms.pkl

Phase 4: RSA Analysis (compare model RDMs to candidate RDMs)
    → data/rsa/rsa_results.json

Phase 5: Figures and Summary
    → figures/*.png, results_summary.txt
```

---

## 5. Cluster Environment and Conventions

### 5.1 Filesystem
- **Project root:** `/mnt/cup/labs/graziano/rachel/mind_rep/exp_6/`
- **Model cache:** Use the existing symlinked HuggingFace cache at `~/.cache/huggingface` (already symlinked to lab filesystem to avoid home dir quota issues).
- **Data:** All outputs go under `data/` within the project root.
- **Code:** All scripts in the project root. Utility functions imported from `src/` if shared.

### 5.2 Conda Environment
- **Use `llama2_env`** (the existing environment for Experiments 2 and 3).
- This environment has: PyTorch, transformers, accelerate, numpy, scipy, scikit-learn, matplotlib, tqdm, etc.
- **CRITICAL:** Always include these lines at the top of SLURM scripts:
  ```bash
  module load pyger
  export PYTHONNOUSERSITE=1
  eval "$(conda shell.bash hook)"
  conda activate llama2_env
  ```

### 5.3 Model
- **Primary model:** `meta-llama/Llama-2-13b-chat-hf`
- This is the same model used in Experiments 1-3. It's already downloaded on the cluster.
- Load with `torch.float16` and `device_map="auto"` for multi-GPU support.
- Use the **chat template** (`llama_v2_prompt` from `src/dataset.py`) to wrap narratives.

### 5.4 SLURM Conventions
- Use `#SBATCH --gres=gpu:1` for single-GPU jobs (Phase 1, 2 with batching).
- Use SLURM array jobs for parallelizable work (Phase 2 activation extraction can be split across narratives).
- Request `--time=04:00:00` initially; adjust based on actual runtime.
- Always include `--output=logs/%j_%a.out --error=logs/%j_%a.err`.

### 5.5 Git
- `.gitignore` excluding `data/`, `logs/`, `*.pt`, `*.pkl`, `__pycache__/`.

---

## 6. Phase-by-Phase Implementation

### Phase 0: Stimulus Generation

**Script:** `0_generate_stimuli.py`

**What it does:**
- Generates all 96 narratives with metadata.

**Output:** `data/stimuli/belief_propagation_stimuli.json` — JSON array of 96 stimulus objects, each containing:
- `narrative_id`, `topology`, `condition`, `names`, `scenario`
- `narrative_text` — the full narrative string
- `expected_beliefs` — dict mapping agent name → believed location
- `epistemic_rdm` — dict mapping agent pair → bool (same belief?)
- `comprehension_probes` — list of {question, correct_answer} dicts
- `extraction_sentence` — the sentence containing all 4 names

**Verification:** Run and confirm 96 stimuli, 12 structural conditions, 7 distinct epistemic geometries.

---

### Phase 1: Behavioral Validation

**Script:** `1_behavioral_validation.py`

**Purpose:** Before investing GPU time on activation extraction, verify the model can answer the belief questions correctly. If behavioral accuracy is at floor, the representational analysis is unmotivated.

**What it does:**
1. Load each narrative into the model as a prompt.
2. For each narrative, ask each of the 4 comprehension probes (e.g., "Where does Alice think the red book is?").
3. Parse the model's free-text response and check if it matches the expected answer.
4. Compute accuracy overall, by topology, by condition, and by whether the agent holds a true or false belief.

**Implementation details:**
- Format: Present the narrative text, then append each question one at a time.
- Use the LLaMA-2-Chat template. The narrative goes in the user message. The question is appended after the narrative.
- Prompt format:
  ```
  [INST] <<SYS>>
  You are a helpful assistant. Answer the question about the story below.
  <</SYS>>

  {narrative_text}

  Question: {comprehension_probe_question}
  Answer with just the location. [/INST]
  ```
- Generate with `max_new_tokens=50`, `temperature=0.0` (greedy decoding).
- Parse response: Check if the correct location string appears in the generated text. Use fuzzy matching (check if the key location phrase like "kitchen table" or "bedroom closet" appears in the response, case-insensitive).

**Output:** `data/behavioral/behavioral_results.json` with:
- Per-narrative, per-agent accuracy
- Aggregated accuracy by condition type (override vs no-override)
- Accuracy for agents with updated beliefs vs. agents with outdated beliefs (the harder case)
- Overall accuracy

**Go/no-go criterion:** If overall accuracy is below 60%, the model struggles with multi-agent belief tracking and the representational analysis may be less informative (but still worth running — the model might have partial representations that don't reach behavior). If accuracy is above 75%, proceed with confidence. Report the number either way.

**SLURM:** Single GPU job, ~1-2 hours for 96 narratives × 4 probes each.

---

### Phase 2: Activation Extraction

**Script:** `2_extract_activations.py`

**Purpose:** Extract hidden-state representations for each agent at their name token position in the extraction sentence, across all layers.

**What it does:**
1. Load the model.
2. For each narrative:
   a. Tokenize the full narrative text (including the extraction sentence at the end).
   b. Run a forward pass, collecting hidden states from all layers.
   c. Identify the token positions of each of the 4 agent names in the extraction sentence.
   d. Extract the hidden state vector at each agent's name token, for each layer.
3. Save the extracted activations.

**Critical implementation details:**

- **Token position identification:** The extraction sentence has the form "Now, Alice, Bob, Carol, and Dave are all gathered in the same room." Tokenize the full narrative and search for the agent name tokens. IMPORTANT: Names may be tokenized into multiple tokens (e.g., "Alice" → ["▁Alice"] or ["▁Al", "ice"]). Use the LAST token of each name, as it integrates information about the full name. Verify tokenization for all 8 name sets before running the full extraction.

- **Layer selection:** Extract from ALL layers (LLaMA-2-13B has 40 transformer layers). Save all layers — the analysis in Phase 4 will test which layers carry the most epistemic information.

- **Hidden state source:** Use the residual stream output of each transformer layer (i.e., `model.model.layers[L]` output, taking `output[0]` which is the hidden states tensor). This matches the convention from Experiments 2-3.

- **Batch processing:** Process narratives one at a time (each narrative is a single sequence). The model should fit on a single A100 in float16.

- **Hook-based extraction:** Use PyTorch forward hooks to collect activations during a single forward pass, rather than running separate passes per layer. Example:
  ```python
  activations = {}
  hooks = []
  for layer_idx, layer in enumerate(model.model.layers):
      def hook_fn(module, input, output, idx=layer_idx):
          activations[idx] = output[0].detach().cpu()
      hooks.append(layer.register_forward_hook(hook_fn))

  with torch.no_grad():
      model(input_ids)

  for h in hooks:
      h.remove()
  ```

- **Output format:** For each narrative, save a dict:
  ```python
  {
      "narrative_id": str,
      "agent_activations": {
          agent_name: {
              layer_idx: tensor of shape [hidden_dim]  # 5120 for LLaMA-2-13B
          }
      },
      "token_positions": {agent_name: int},  # for verification
  }
  ```
  Save as `data/activations/{narrative_id}.pt` using `torch.save()`.

**Verification:**
- Print token positions for the first few narratives to confirm names are correctly located.
- Spot-check that activation tensors have shape [5120] and are not all zeros.
- Confirm 96 output files exist.

**SLURM:** Single GPU, ~2-4 hours for 96 narratives (model load once, iterate).

---

### Phase 3: RDM Construction

**Script:** `3_construct_rdms.py`

**Purpose:** Compute pairwise representational dissimilarity between agents within each narrative, producing a 4×4 RDM per narrative per layer.

**What it does:**
1. Load all activation files.
2. For each narrative and each layer:
   a. Get the 4 agent activation vectors (each shape [5120]).
   b. Compute pairwise cosine distance: `distance(i,j) = 1 - cosine_similarity(v_i, v_j)`.
   c. Store as a 4×4 symmetric matrix (diagonal = 0).
3. Also construct the three candidate RDMs for each narrative:
   a. **Epistemic RDM:** 0 if agents share belief, 1 if they disagree. (From the stimulus metadata.)
   b. **Communication RDM:** 0 if agents communicated directly in the topology, 1 if not. (Derived from topology edges.)
   c. **Position RDM:** |i - j| where i, j are agent positions in the extraction sentence (0, 1, 2, 3).
4. Flatten each 4×4 RDM to its upper triangle (6 values) for RSA.

**Output:** `data/rdms/model_rdms.pkl` — dict with:
```python
{
    narrative_id: {
        "model_rdm": {layer_idx: np.array of shape [6]},  # upper triangle
        "epistemic_rdm": np.array of shape [6],
        "communication_rdm": np.array of shape [6],
        "position_rdm": np.array of shape [6],
        "condition": str,
        "topology": str,
    }
}
```

**No GPU needed.** CPU job, runs in minutes.

---

### Phase 4: RSA Analysis

**Script:** `4_rsa_analysis.py`

**Purpose:** The core analysis. Compare the model's representational geometry to the three candidate geometries using RSA.

**What it does:**

#### 4a. Per-narrative RSA
For each narrative and each layer:
1. Correlate the model's RDM (upper triangle, 6 values) with each candidate RDM using **Spearman rank correlation**.
2. This gives three correlation values per narrative per layer: r_epistemic, r_communication, r_position.

Note: With only 6 datapoints per correlation, individual narrative RSA is noisy. This is expected and motivates the aggregation in 4b.

#### 4b. Aggregated RSA across narratives
For each layer:
1. Average r_epistemic, r_communication, r_position across all 96 narratives (or across subsets by topology/condition).
2. Test whether mean r_epistemic > mean r_communication and mean r_epistemic > mean r_position using paired t-tests (or Wilcoxon signed-rank tests if non-normal).
3. Also compute a permutation test: shuffle the epistemic RDM labels 10,000 times and recompute RSA to get a null distribution. Check if the true r_epistemic exceeds the 95th percentile of the null.

#### 4c. Cross-topology consistency
Compare model RDMs from conditions that have the SAME epistemic geometry but DIFFERENT communication structure:
- `chain_override_C_tells_D` and `diamond_override_C_tells_D` both produce {A:old, B:old, C:new, D:new}
- `chain_override_D` and `diamond_override_D_only` both produce {A:old, B:old, C:old, D:new}

If the model tracks epistemic state, the model RDMs from these matched conditions should correlate highly with each other (cross-topology RSA). If it tracks communication structure, they should diverge.

#### 4d. Layer-by-layer profile
Plot RSA correlation (epistemic, communication, position) as a function of layer. This reveals WHERE in the model epistemic tracking emerges — is it early layers (surface features), middle layers (emerging world model), or late layers (pre-output)?

#### 4e. Condition-specific analysis
Break down RSA by:
- Override vs. no-override (no-override is a ceiling control — all RDMs are identical so RSA is trivially high)
- Number of belief-updated agents (1 vs. 2 vs. 3 — does complexity matter?)
- Topology (chain vs. fork vs. diamond — are some structures easier?)

**Statistical tests:**
- Paired t-tests (or Wilcoxon) comparing r_epistemic vs r_communication across narratives, per layer.
- Permutation tests for overall significance.
- BH-FDR correction across layers.

**Output:** `data/rsa/rsa_results.json` with all correlation values, p-values, and summary statistics.

---

### Phase 5: Figures and Summary

**Script:** `5_figures.py`

**What to produce:**

1. **Layer profile plot** (the main figure): X-axis = layer (0-39), Y-axis = mean RSA correlation. Three lines: epistemic (blue), communication (orange), position (gray). Error bars = SEM across narratives. Significance markers (* / ** / ***) where epistemic > communication.

2. **RDM visualization** (supplementary): For a representative layer (the peak epistemic RSA layer), show the average model RDM alongside the ground-truth epistemic RDM for each condition type. Use heatmaps.

3. **Cross-topology consistency** (supplementary): Scatter plot of model RDM values from matched epistemic conditions across different topologies. High correlation = epistemic tracking; low = surface features.

4. **Behavioral accuracy bar chart**: By condition, by agent belief status (updated vs. outdated).

5. **Design overview diagram**: Schematic showing the three topologies with example belief states. (Use matplotlib or a simple diagram.)

**Output:** `figures/` directory with PNGs.

Also produce `results_summary.txt` with:
- Behavioral accuracy summary
- Peak RSA layer and correlation values
- Statistical test results
- Key finding in 2-3 sentences

---

## 7. File Manifest

After all phases, the directory should contain:

```
exp_6/
├── 0_generate_stimuli.py
├── 1_behavioral_validation.py
├── 2_extract_activations.py
├── 3_construct_rdms.py
├── 4_rsa_analysis.py
├── 5_figures.py
├── config.py                    # Shared constants (model path, data paths, etc.)
├── src/
│   └── utils.py                 # Shared utilities (tokenization helpers, etc.)
├── slurm/
│   ├── phase1_behavioral.sh
│   ├── phase2_extract.sh
│   └── phase4_rsa.sh
├── data/
│   ├── stimuli/
│   │   └── belief_propagation_stimuli.json
│   ├── behavioral/
│   │   └── behavioral_results.json
│   ├── activations/
│   │   └── {narrative_id}.pt    (× 96)
│   ├── rdms/
│   │   └── model_rdms.pkl
│   └── rsa/
│       └── rsa_results.json
├── figures/
│   ├── layer_profile_rsa.png
│   ├── rdm_heatmaps.png
│   ├── cross_topology_consistency.png
│   ├── behavioral_accuracy.png
│   └── design_overview.png
├── results_summary.txt
├── logs/
└── .gitignore
```

---

## 8. Implementation Order and Dependencies

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5
(CPU)       (GPU)       (GPU)       (CPU)       (CPU)       (CPU)
 5 min      1-2 hr      2-4 hr      5 min       10 min      5 min
```

Phase 1 and Phase 2 can technically run in parallel (both need the model loaded), but Phase 1 should be checked first as a sanity gate.

Phase 3, 4, 5 are CPU-only and fast.

---

## 9. Key Methodological Decisions (and Why)

### Why cosine distance for RDMs?
Cosine distance is standard in RSA for high-dimensional neural representations. It's invariant to the overall magnitude of activation vectors (which can vary by layer and position) and focuses on the direction of the representation, which is where concept information lives in the linear representation hypothesis.

### Why Spearman correlation for RSA?
Spearman is the standard in neuroscience RSA (Kriegeskorte et al., 2008). It captures monotonic relationships without assuming linearity, which is appropriate when comparing binary candidate RDMs (0/1) to continuous model RDMs.

### Why residual stream and not attention patterns?
Residual stream representations accumulate information across all previous layers and heads, giving the richest single-point readout of what the model "knows" about each agent at that position. Attention patterns are complementary (and could be a follow-up analysis) but are harder to interpret for RSA because they're relational (token-to-token) rather than per-token vectors.

### Why last token of agent name?
In autoregressive transformers, the last token of a multi-token name integrates information about the full name through causal attention. This is the standard practice in probing studies (Zhu et al. 2024, Viegas et al. 2024).

### Why 8 instantiations per condition?
This gives 8 independent RDM samples per structural condition, enough for meaningful within-condition averaging while keeping the total stimulus count manageable (96 total). More is better — if time permits, increase to 12 or 16.

---

## 10. Potential Issues and Troubleshooting

### Issue: Names tokenized into many subwords
Some name sets might tokenize differently. Before running Phase 2, run a tokenization check on all 96 narratives and print token-level alignment for the extraction sentence. Verify that every agent name can be unambiguously located. If a name tokenizes ambiguously (e.g., overlapping with common words), replace it.

### Issue: Extraction sentence names are too close together
If names in "Now, Alice, Bob, Carol, and Dave are all gathered in the same room" are only 1-2 tokens apart, adjacent agents might share information through short-range attention. This is actually fine for the analysis — position effects are controlled for by the position RDM baseline. But if concerned, extend the extraction sentence to space names further apart.

### Issue: Model VRAM
LLaMA-2-13B in float16 needs ~26GB VRAM. A single A100 (80GB) handles this easily. If using smaller GPUs, use `device_map="auto"` to split across multiple GPUs. The activation extraction hooks work regardless of device placement.

### Issue: Low behavioral accuracy
If the model struggles (< 60% accuracy on belief probes), consider:
1. Using a larger model (LLaMA-2-70B or LLaMA-3-8B-Instruct, which may have better ToM).
2. Simplifying to 3 agents instead of 4.
3. Using chain-of-thought prompting for the behavioral validation (but NOT for activation extraction — you want the model's implicit representations, not its explicit reasoning).

### Issue: Flat RSA results (no layer differentiates)
If r_epistemic ≈ r_communication ≈ r_position across all layers, the model may not be tracking epistemic state internally even if it answers behavioral questions correctly (it could be using surface heuristics). This is a VALID NEGATIVE RESULT and is publishable. Report it honestly.

### Issue: Communication RDM wins over epistemic RDM
If r_communication > r_epistemic, the model is representing "who talked to whom" rather than "who believes what." This is also an interesting finding — it suggests the model has a communication-structure world model but not an epistemic world model.

---

## 11. Extensions (If Time Permits)

### 11a. Causal Intervention
After identifying the peak RSA layer, construct an "epistemic steering vector" by averaging the direction from outdated-belief agents to updated-belief agents. Inject this vector at the peak layer for a neutral agent and test whether the model's behavioral output shifts (i.e., it now reports the new location for an agent who should believe the old location). This provides causal evidence that the epistemic representation is functionally used.

### 11b. Attention Pattern Analysis
For the peak RSA layer, examine whether specific attention heads show "epistemic lookback" patterns — attending selectively to the tokens describing the event each agent was present for. Compare attention patterns for agents with updated vs. outdated beliefs.

### 11c. Scaling Up
Test on LLaMA-3-8B, LLaMA-3-70B, and/or Mistral-7B to see if epistemic RSA improves with model scale (as Bortoletto et al. found for simple 2-agent belief probing).

### 11d. Second-Order Beliefs
Add conditions where Agent A is asked not just "where is the ball?" but "where does Bob think the ball is?" — probing second-order belief representations. This dramatically increases complexity but tests a richer form of ToM.

---

## 12. References

- Gurnee, W. & Tegmark, M. (2023). Language Models Represent Space and Time. arXiv:2310.02207.
- Zhu, W., Zhang, H., & Wang, H. (2024). Language Models Represent Beliefs of Self and Others. ICML 2024.
- Bortoletto, M., et al. (2024). Brittle Minds, Fixable Activations. arXiv:2406.17513.
- Prakash, S., et al. (2025). Language Models use Lookbacks to Track Beliefs. arXiv:2505.14685.
- Li, B., et al. (2025). From Black Boxes to Transparent Minds: GridToM. ICML 2025.
- Shai, L., et al. (2024). Transformers Represent Belief State Geometry in their Residual Stream. NeurIPS 2024.
- Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis. Frontiers in Systems Neuroscience, 2, 4.
- Herrmann, D. & Levinstein, B. (2024). Standards for Belief Representations in LLMs. Minds and Machines.
