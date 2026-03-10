#!/bin/bash
#SBATCH --job-name=contrast_align_base
#SBATCH --output=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/concept_geometry/contrast_alignment_base_%j.out
#SBATCH --error=/mnt/cup/labs/graziano/rachel/mind_rep/exp_4/logs/concept_geometry/contrast_alignment_base_%j.err
#SBATCH --gres=gpu:1 --mem=64G --time=2:00:00 --cpus-per-task=4

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

python concept_geometry/rsa/contrast_alignment.py --model llama2_13b_base
