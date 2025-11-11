#!/bin/bash
#SBATCH --job-name=behavior_batch
#SBATCH --output=/jukebox/graziano/rachel/ai_mind_rep/exp_1/logs/run_all_%j.out
#SBATCH --error=/jukebox/graziano/rachel/ai_mind_rep/exp_1/logs/run_all_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

# Predefine PS1 so module/conda scripts don't choke under -u
export PS1=${PS1:-}

set -euo pipefail


# === Activate environment ===
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate behavior_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT
# --------------------------------------------

# --- Environment setup ---
export PROJECT_ROOT="/jukebox/graziano/rachel/ai_mind_rep/exp_1"
export CONFIG_PATH="$PROJECT_ROOT/configs/behavior.json"
export LOG_DIR="$PROJECT_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT/code/analysis${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$LOG_DIR"

# --- Script list (excluding B8) ---
scripts=(
  "combine_text_data.py"
  "qual_connect.py"
  "wordcount.py"
  "questions.py"
  "tom_words.py"
  "empath_tom.py"
  "politeness.py"
  "hedging.py"
  "filler.py"
  "sentiment_vader.py"
  #"sentiment_transformer.py"
  #"semantic_diversity.py"
)

#scripts=(
  #"sentiment_transformer.py"
#)# use llama2_env for this one

# --- Run each safely ---
for script in "${scripts[@]}"; do
  echo "Running $script ..."
  log_file="$LOG_DIR/${script%.py}_$(date +%Y%m%d_%H%M%S).log"
  (
    echo "=============================="
    echo " Script: $script"
    echo " Started: $(date)"
    echo "=============================="
    python "$PROJECT_ROOT/code/analysis/$script" --config "$CONFIG_PATH" &>> "$log_file"
    echo "------------------------------"
    echo " Finished: $(date)"
    echo " Log saved to: $log_file"
    echo "------------------------------"
  ) || echo "⚠️  $script crashed — continuing..."
done

echo "✅ All behavior scripts attempted."
