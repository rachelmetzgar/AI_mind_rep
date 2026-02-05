# Cross-Experiment Behavioral Comparison: Humans vs LLMs

**Author:** Rachel C. Metzgar  
**Script:** `cross_experiment_comparison.py`  
**Python:** 3.8+  
**Conda Environment:** `behavior_env`

This script compares **label-conditioned conversational behavior** between human participants (Exp1) and LLM "participants" (Exp2), examining how beliefs about partner identity (human vs AI) shape linguistic patterns during naturalistic conversation.

---

## Quick Start

1. **Prepare input data**

Place your data files in the working directory:
```
├── exp_csv_human/          # Per-subject CSVs for human experiment
│   ├── sub-001.csv
│   ├── sub-002.csv
│   └── ...
├── combined_text_data_LLM.csv   # Combined LLM experiment data
└── topics.csv              # Topic → social/nonsocial mapping
```

2. **Run the analysis**
```bash
pyger
conda activate behavior_env
python cross_experiment_comparison.py
```

3. **View results**

Outputs saved to `cross_experiment_results/`:
```
cross_experiment_results/
├── cross_experiment_stats_PER_TRIAL.txt      # Main results (trial-level)
├── cross_experiment_stats_PER_UTTERANCE.txt  # Utterance-level results
├── combined_utterance_level_data.csv         # All utterances with metrics
├── combined_trial_level_data.csv             # Trial-aggregated data
├── per_trial_subject_condition_social.csv    # Subject means by condition × sociality
├── per_trial_subject_condition.csv           # Subject means by condition
├── per_utterance_subject_condition_social.csv
└── per_utterance_subject_condition.csv
```

---

## Configuration

Edit the `CONFIGURATION` section at the top of the script:

```python
# Data loading mode
EXP1_USE_INDIVIDUAL_FILES = True          # True: load per-subject CSVs; False: use combined file
EXP1_SUBJECT_DIR = "exp_csv_human"        # Directory containing per-subject CSVs
EXP1_FILE_PATTERN = "{sub_id}.csv"        # Filename pattern for subject files

# Subject IDs (standardized MRI format)
EXP1_SUBJECT_IDS = [
    "sub-001", "sub-002", "sub-003", ...
]

# Alternative: combined file path (if EXP1_USE_INDIVIDUAL_FILES = False)
EXP1_COMBINED_PATH = "combined_text_data_humans.csv"

# LLM experiment data
EXP2_PATH = "combined_text_data_LLM.csv"

# Output directory
OUTPUT_DIR = "cross_experiment_results"

# Topic classification file
TOPICS_PATH = "topics.csv"
```

---

## Input Data Format

### Per-subject CSVs (Exp1 Human)

Required columns:
| Column | Description |
|--------|-------------|
| `transcript_sub` | Participant's speech text |
| `agent` | Partner identifier (must contain "hum" or "bot") |
| `topic` | Conversation topic name |
| `Quality` | Post-conversation quality rating (1-4) |
| `Connectedness` | Post-conversation connectedness rating (1-4) |

### Combined LLM CSV (Exp2)

Same columns as above, plus:
| Column | Description |
|--------|-------------|
| `subject` | LLM participant ID (e.g., "s001") |

### Topics CSV

| Column | Description |
|--------|-------------|
| `topic` | Topic name (lowercase) |
| `social` | 1 = social topic, 0 = nonsocial topic |

---

## Metrics Computed

### Hedging — Demir (2018) Taxonomy

Based on Hyland's (1998) six-category framework:

| Metric | Description |
|--------|-------------|
| `demir_modal_rate` | Modal auxiliaries (can, could, may, might, should, would) |
| `demir_verb_rate` | Epistemic verbs (44 items: seem, believe, suggest, think, etc.) |
| `demir_adverb_rate` | Epistemic adverbs (47 items: perhaps, probably, generally, etc.) |
| `demir_adjective_rate` | Epistemic adjectives (19 items: possible, likely, potential, etc.) |
| `demir_quantifier_rate` | Quantifiers/determiners (a few, some, most, etc.) |
| `demir_noun_rate` | Epistemic nouns (19 items: assumption, belief, possibility, etc.) |
| `demir_total_rate` | All hedge categories combined |

### Disfluencies — LIWC2007 Categories

| Metric | Description |
|--------|-------------|
| `nonfluency_rate` | er, hm, sigh, uh, um, umm, well |
| `liwc_filler_rate` | blah, i don't know, i mean, oh well, or anything, you know, etc. |
| `disfluency_rate` | Combined nonfluencies + fillers |

### Discourse Markers — Fung & Carter (2007)

| Metric | Description |
|--------|-------------|
| `fung_interpersonal_rate` | you know, well, really, I think, like, okay, yeah, etc. |
| `fung_referential_rate` | because, but, and, or, so, anyway, etc. |
| `fung_structural_rate` | now, first, then, finally, etc. |
| `fung_cognitive_rate` | well, I think, I mean, sort of, kind of, etc. |
| `fung_total_rate` | All 23 markers from Fung & Carter Table 2 |

### Other Linguistic Markers

| Metric | Description |
|--------|-------------|
| `word_count` | Total words per conversation |
| `question_count` | Number of questions (? count) |
| `tom_rate` | Second-person ToM phrases (you think, you believe, you feel, etc.) |
| `politeness_rate` | Net politeness (positive + negative markers − impolite markers) |
| `like_rate` | Discourse marker "like" (D'Arcy, 2007) |
| `sentiment` | VADER compound sentiment score (−1 to +1) |
| `quality` | Self-reported conversation quality (1-4) |
| `connectedness` | Self-reported felt connection (1-4) |

---

## Statistical Analyses

### Within-Experiment: 2×2 Repeated-Measures ANOVA

For each experiment separately:
- **Condition** (Human-label vs AI-label) × **Sociality** (Social vs Nonsocial topics)
- Proper error terms for within-subject design: F(1, N−1)
- Simple effects tests when interaction is significant

### Cross-Experiment: Condition Effect Comparison

- Computes condition effect (Human-label − AI-label) per subject
- Independent t-test comparing effect magnitudes between experiments
- Pattern classification: Similar, Flipped, Different, Both ns

---

## Output Files

### Stats Text Files

`cross_experiment_stats_PER_TRIAL.txt` contains:
1. **Summary table** — Quick overview of all metrics with direction and significance
2. **Within-experiment ANOVAs** — Full ANOVA results for each metric × experiment
3. **Cross-experiment tests** — Interaction tests comparing humans vs LLMs

### CSV Files

| File | Level | Use |
|------|-------|-----|
| `combined_utterance_level_data.csv` | Utterance | Raw data with all computed metrics |
| `combined_trial_level_data.csv` | Trial | Summed within each conversation |
| `per_trial_subject_condition_social.csv` | Subject | For 2×2 ANOVA |
| `per_trial_subject_condition.csv` | Subject | For cross-experiment comparison |

---

## Dependencies

```bash
pip install pandas numpy scipy statsmodels nltk
```

For VADER sentiment analysis:
```python
import nltk
nltk.download('vader_lexicon')
```

---

## Two Analysis Pipelines

The script runs **two parallel pipelines**:

### Pipeline 1: Per-Trial Analysis
1. Sum counts within each conversation (trial)
2. Compute rates at trial level
3. Average across trials to get subject means
4. Run ANOVAs on subject-level data

**Use this for:** Primary analyses reported in papers

### Pipeline 2: Per-Utterance Analysis
1. Compute rates at utterance level
2. Average directly to subject means (no trial-level summing)

**Use this for:** Sensitivity analyses, checking robustness

---

## Subject ID Mapping

The script handles legacy ID formats automatically:

| Legacy Format | Standardized Format |
|---------------|---------------------|
| P08, s08 | sub-001 |
| P12, s12 | sub-002 |
| ... | ... |

Mapping defined in `get_sub_id_map()`.

---

## References

- **Hedging:** Demir, C. (2018). Hedging and academic writing: An analysis of lexical hedges. *Journal of Language and Linguistic Studies*, 14(4), 74-92.
- **Hedging framework:** Hyland, K. (1998). Boosting, hedging, and the negotiation of academic knowledge. *Text*, 18(3), 349-382.
- **Discourse markers:** Fung, L., & Carter, R. (2007). Discourse markers and spoken English. *Applied Linguistics*, 28(3), 410-439.
- **Disfluencies:** Pennebaker, J. W., et al. (2007). *LIWC2007 Manual*.
- **"Like":** D'Arcy, A. (2007). Like and language ideology. *American Speech*, 82(4), 386-419.
- **Sentiment:** Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis. *ICWSM*.

---

## Example Output

```
================================================================================
SUMMARY TABLE: Comparison of Belief-Conditioned Effects in Humans vs LLMs
================================================================================

Measure                        Human (Exp1)         LLM (Exp2)           Cross-Exp p     Pattern     
--------------------------------------------------------------------------------------------------------------
word_count                     Human > AI*          AI > Human*          <.001***        Flipped     
demir_total_rate               ns                   Human > AI*          0.0023**        Different   
disfluency_rate                Human > AI*          Human > AI*          0.1860          Similar     
tom_rate                       AI > Human*          Human > AI*          <.001***        Flipped     
...
```

---