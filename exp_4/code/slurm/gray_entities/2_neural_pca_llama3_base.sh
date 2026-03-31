#!/bin/bash
#SBATCH --job-name=exp4_neural_pca_l3base
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/gray_simple/2_neural_pca_llama3_base_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/gray_simple/2_neural_pca_llama3_base_%j.err
#SBATCH --mem=16G --time=1:00:00 --cpus-per-task=4

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

python gray_entities/neural/2_neural_pca.py --model llama3_8b_base --both
