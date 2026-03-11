#!/bin/bash
#SBATCH --job-name=exp6_rsa
#SBATCH --output=logs/rsa/%j.out
#SBATCH --error=logs/rsa/%j.err
#SBATCH --mem=16G
#SBATCH --time=04:00:00

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

cd /mnt/cup/labs/graziano/rachel/mind_rep/exp_6
python code/4_rsa_analysis.py
