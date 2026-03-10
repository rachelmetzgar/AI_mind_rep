#!/bin/bash
#SBATCH --job-name=exp4_names_only_chat
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/behavior/5_names_only_chat_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/behavior/5_names_only_chat_%j.err
#SBATCH --gres=gpu:1 --mem=64G --time=4:00:00 --cpus-per-task=4

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

python behavior/5_gray_names_only.py --model llama2_13b_chat \
    --exclude claude cortana copilot replika bing_chat cleverbot casey michael aisha sam sofia watson mei james
