#!/bin/bash
#SBATCH --job-name=exp4_gchars_l3inst
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/human_ai_adaptation/gray_chars_llama3_instruct_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/human_ai_adaptation/gray_chars_llama3_instruct_%j.err
#SBATCH --gres=gpu:1 --mem=32G --time=4:00:00 --cpus-per-task=4

export PS1=${PS1:-}
set -euo pipefail
module load pyger
export PYTHONNOUSERSITE=1
set +u
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate llama2_env
set -u
trap 'set +u; conda deactivate >/dev/null 2>&1 || true; set -u' EXIT

cd /mnt/cup/labs/graziano/rachel/mind_rep/exp_4/code

python human_ai_adaptation/behavior/1_gray_with_characters.py --model llama3_8b_instruct
