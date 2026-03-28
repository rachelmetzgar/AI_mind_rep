#!/bin/bash
#SBATCH --job-name=exp4_arsa_l3base
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/expanded_mental_concepts/activation_rsa_llama3_base_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/expanded_mental_concepts/activation_rsa_llama3_base_%j.err
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

python expanded_mental_concepts/internals/rsa/activation_rsa.py --model llama3_8b_base
