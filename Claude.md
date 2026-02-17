AI Mind Representation Project
Rules
	•	ONLY access files within this project directory
	•	Do NOT access, read, or reference any other directories on this system
	•	Do NOT access any directories outside of ai_mind_rep/
	•	When editing scripts, preserve existing file structure and imports
Project Overview
This project investigates whether LLMs form internal representations of their conversational partner's identity (human vs. AI) and whether those representations causally drive behavioral adaptation. It bridges cognitive science, computational linguistics, and mechanistic interpretability.
The core theoretical question: Is there a functional dissociation between where models represent vs. use partner identity information? And does the partner representation have compositional structure (mental-property dimensions) or is it an opaque identity switch?
Model
	•	Primary model: LLaMA-2-13B-Chat (all probing and intervention experiments)
	•	Dataset generation: GPT-3.5-Turbo (synthetic conversations for Exp 4)
	•	Behavioral experiment: GPT-3.5-Turbo participant agents (Exp 1)
	•	GPT judge: GPT-4o-mini (causal evaluation; use --judge gpt flag)
	•	Hidden dimension: 5120 (13B model), 4096 (7B model — legacy Exp 4 only)
Experiments
Experiment 1 — Behavioral Analysis (exp_1/)
	•	Question: Do LLMs adjust conversational behavior based on partner identity labels?
	•	Design: 50 GPT-3.5-Turbo "participant agents" × 40 conversations (4 partner conditions × 2 topic types). All partners are the same LLM — only the label differs.
	•	Partner conditions: Sam (human), Casey (human), ChatGPT (AI), Gemini (AI)
	•	Measures: Word count, question frequency, discourse markers (Fung & Carter 2007), "like" usage (Fuller 2003), hedging (Demir 2018 taxonomy), politeness, Theory of Mind phrases (Wagovich et al. 2024), sentiment (VADER)
	•	Key result: LLMs show robust label-conditioned effects across 14/23 measures. Human comparison (N=23, companion fMRI study) shows largely divergent patterns — only "like" usage converges across species.
	•	Analysis: 2×2 repeated-measures ANOVA (Partner × Sociality). Use effect-specific error terms, not pooled.
	•	Status: Complete ✅
Experiment 2 — Naturalistic Conversation Steering (exp_2/llama_exp_2b-13B-chat/)
Historically called Experiment 2b, now renumbered to Experiment 2.
	•	Question: Is partner identity linearly decodable from LLaMA-2-13B-Chat's internal representations, and does steering along that direction causally shift conversational behavior?
	•	Design: Train linear probes on activations from naturalistic conversations (same structure as Exp 1 — system prompt specifies partner identity, no other cues). Two probe types:
	◦	Reading probes: trained on "I think the partner…" reflection token activations
	◦	Control probes: trained on last-user-message token activations (pre-generation boundary)
	•	Key finding: Control probes sit at functional bottlenecks and steer more effectively per unit strength. Reading probes capture a representationally distinct dimension (formality/politeness vs. interpersonal/conversational style). This supports the functional dissociation hypothesis.
	•	Intervention: Activation addition via forward hooks. Steering vector = probe weight direction, scaled by N.
	◦	Formula: h' = h + N · y · θ (where y = +1 for human, -1 for AI)
	◦	Unit normalize steering vectors before intervention for consistent calibration
	◦	Optimal N=1 for control probes + all_70 strategy (N≥2 causes repetition loops)
	◦	Reading probes need slightly higher N (≈5) for comparable effects
	•	Layer selection: all_70 strategy uses layers 7–40 (strongest effects). Avoid exclude mode — it handicaps reading probes by giving them only their worst layers.
	•	Dose-response: N=1 is optimal. N≥2 causes degeneration (token-loop collapse). Human-steered degenerates before AI-steered (baseline is closer to AI-steered due to RLHF).
	•	Judge: GPT-4o-mini pairwise evaluation with randomized presentation order
	•	Status: Complete ✅. Re-run best dose-response N with --judge gpt for paper results.
Experiment 3 — Concept Alignment / Injection (exp_3/llama_exp_3-13B-chat/)
	•	Question: Does the model's general semantic concept of "human" vs. "AI" share representational structure with the conversational partner-identity signal? Does the partner representation have mental-property structure?
	•	Null hypothesis: Partner representation is an opaque switch (entity-type only)
	•	Alternative hypothesis: Partner representation has compositional mental-property structure
	•	This experiment characterizes representational content, not just existence.
	•	Design:
	1	Concept elicitation: Extract activation vectors for "human" and "AI" concepts across 16 dimensions (mental properties like intentions/emotions/prediction, physical properties, behavioral properties, orthogonal controls like shapes)
	2	Alignment analysis: Cosine similarity between concept probe weights and Exp 2 partner probe weights at each layer
	3	Concept injection: Steer conversations using concept vectors (instead of partner probes) and measure behavioral effects
	4	Cross-prediction: Test whether concept dimensions that align with partner probes also causally steer behavior
	•	16 dimensions: baseline, social, communication, agency, intentions, prediction, experience, emotions, embodiment, consciousness, appearance, formality, warmth, technical, biology, shapes (orthogonal control)
	•	Key early finding: Overall concept vectors are orthogonal to conversational partner probes (max cos ≈ 0.03). But dimension-specific analysis needed — mental-property dimensions may align while physical dimensions don't.
	•	Dose-response: Running V1 sweep with N=1,2,4,8 across dimensions
	•	Status: V1 dose-response in progress. Run best N with --judge gpt for paper results.
Experiment 4 — Viégas/TalkTuner Replication (exp_4/)
Historically called Experiment 2a, now renumbered to Experiment 4.
	•	Question: Can we replicate the TalkTuner methodology for human vs. AI partner classification?
	•	Design: Generate synthetic conversations (GPT-3.5-Turbo) with explicit human/AI user roles, extract activations from LLaMA-2, train probes, run causal interventions. This is the non-naturalistic version (conversations are generated with instructions specifying partner type).
	•	Concern: Probes may learn partner names or linguistic artifacts rather than abstract identity representations (name confound). This motivated Exp 2's naturalistic approach.
	•	Contains: exp_2a-13B-chat/ and exp_2a-7B-base/
	•	Status: Complete ✅ (serves as baseline comparison for Exp 2)
Directory Structure
ai_mind_rep/
├── envs/                          # Conda environment YAMLs
│   ├── behavior_env.yml           # For Exp 1 (GPT API calls, behavioral analysis)
│   └── llama2_env.yml             # For Exps 2-4 (LLaMA, probing, intervention)
├── exp_1/                         # Experiment 1: Behavioral analysis
│   ├── code/                      # Analysis scripts
│   ├── configs/                   # Experiment configuration
│   ├── data/                      # Conversation data
│   ├── results/                   # Statistical results, figures
│   └── README.md
├── exp_2/                         # Experiment 2: Naturalistic steering
│   └── llama_exp_2b-13B-chat/
│       ├── 1_preprocess_dataset.py
│       ├── 2_train_and_read_controlling_probes.py
│       ├── 3_causality_generate.py          # V1 (single-turn) + V2 (multi-turn) intervention
│       ├── 4_causality_judge.py             # GPT-4o-mini pairwise evaluation
│       ├── 5_behavior_analysis.py           # Linguistic analysis (same pipeline as Exp 1)
│       ├── src/                             # Shared modules (probes, dataset, losses, etc.)
│       ├── data/
│       │   ├── probe_checkpoints/           # Trained probe weights
│       │   │   ├── control_probe/
│       │   │   └── reading_probe/
│       │   ├── causality_test_questions/    # Held-out V1 test prompts
│       │   └── intervention_results/        # Generated steered conversations
│       ├── slurm/                           # SLURM job scripts
│       └── README.md
├── exp_3/                         # Experiment 3: Concept alignment/injection
│   └── llama_exp_3-13B-chat/
│       ├── 1_elicit_concept_vectors.py      # Phase 1: concept activation extraction
│       ├── 1b_alignment_analysis.py         # Phase 2: concept ↔ partner probe alignment
│       ├── 1c_layer_profile_analysis.py     # Layer-wise alignment profiles
│       ├── 1d_elicit_sysprompt_vectors.py   # System prompt concept elicitation
│       ├── 1e_sysprompt_alignment.py        # System prompt alignment analysis
│       ├── 2_train_concept_probes.py        # Train probes on concept activations
│       ├── 3_concept_intervention.py        # Concept injection into conversations
│       ├── 3_summarize_alignment.py         # Summarize alignment results
│       ├── 4_behavior_analysis.py           # Linguistic analysis on concept-steered output
│       ├── 5_cross_prediction.py            # Cross-experiment representational analysis
│       ├── concepts/                        # Concept dimension definitions
│       ├── data/
│       └── README.md
├── exp_4/                         # Experiment 4: Viégas/TalkTuner replication
│   ├── exp_2a-13B-chat/
│   └── exp_2a-7B-base/
└── README.md
Key Technical Details
Probing Architecture
	•	LinearProbeClassification: Single linear layer + sigmoid (binary logistic probe)
	•	ProbeClassification: Two-layer MLP with ReLU (for multi-class, not used in current exps)
	•	Input dim: 5120 (13B) or 4096 (7B legacy)
	•	Training: 80/20 train/test split, stratified by label
Intervention Mechanics
	•	Forward hooks via inline TraceDict (avoids baukit dependency on cluster)
	•	Steering: add N × direction to residual stream activations at last token position
	•	Direction source: probe weight vector (.proj[0].weight)
	•	Layers intervened: determined by layer selection strategy (all_70, wide, peak_15, narrow)
	•	Always unit-normalize steering vectors for consistent dose-response calibration
Critical Methodological Notes
	•	N=1 is often optimal (N≥2 causes degeneration into repetition loops for control probes)
	•	Reading probes need higher N (≈5) than control probes (≈1) for comparable effects
	•	Use mean-across-layers metrics over single-layer maxima to avoid overfitting artifacts
	•	exclude layer mode is broken — gives reading probes only their worst layers
	•	Name confounds: probes can learn partner names instead of abstract identity. Exp 2's naturalistic design controls for this; Exp 4's synthetic design is vulnerable.
	•	Mojibake in text data: watch for encoding issues in conversation files
Statistical Methods
	•	2×2 repeated-measures ANOVA (Partner × Sociality) with effect-specific error terms
	•	Do NOT use pooled error terms — compute separate error for each effect
	•	Cross-study comparisons: independent t-tests on subject-level condition effects
Cluster Environment
	•	Princeton cluster (Scotty), SLURM scheduler
	•	Conda environments: behavior_env (Exp 1), llama2_env (Exps 2-4)
	•	LLaMA-2-13B-Chat snapshot at: /jukebox/graziano/rachel/ai_percep_clean/.cache/huggingface/hub/models--meta-llama--Llama-2-13b-chat-hf/
	•	GPU jobs: typically 1 GPU, 48G memory, 6-12 hour time limits
Key Dependencies
	•	PyTorch, transformers (HuggingFace)
	•	baukit (activation extraction — some scripts use inline TraceDict instead)
	•	statsmodels (ANOVA), scipy
	•	openai (GPT API for generation and judging)
	•	VADER (sentiment), custom word lists for linguistic measures

References
	•	Chen, Y. et al. (2024). TalkTuner. arXiv:2406.07882.
	•	Demir, C. (2018). Hedging taxonomy. JLLS 14(4).
	•	Fung, L. & Carter, R. (2007). Discourse markers. Applied Linguistics 28(3).
	•	Fuller, J. (2003). "Like" as a discourse marker. Journal of Sociolinguistics 7(3).
	•	Wagovich, S. et al. (2024). Mental state verbs. Language & Communication.
	•	Hutto, C. & Gilbert, E. (2014). VADER sentiment. ICWSM.
	•	Touvron, H. et al. (2023). LLaMA 2. arXiv:2307.09288.
