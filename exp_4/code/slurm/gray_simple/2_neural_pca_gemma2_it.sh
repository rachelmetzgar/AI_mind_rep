#!/bin/bash
#SBATCH --job-name=exp4_neural_pca_gem2
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/gray_simple/2_neural_pca_gemma2_it_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/gray_simple/2_neural_pca_gemma2_it_%j.err
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

python gray_simple/internals/2_neural_pca.py --model gemma2_9b_it --both
