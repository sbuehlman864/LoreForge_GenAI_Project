#!/bin/bash
#SBATCH --job-name=loreforge_hp
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=loreforge_hp_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=spencerbuehlman2026@u.northwestern.edu

module purge
module load python-miniconda3/4.12.0

export PYTHONUSERBASE=/projects/e32706/jgu2930/.local
export HF_HOME=/projects/e32706/jgu2930/.cache/huggingface
export HF_DATASETS_CACHE=/projects/e32706/jgu2930/.cache/huggingface/datasets

cd /projects/e32706/jgu2930

python finetune_gpt2.py --universe harry_potter --n_epochs 3
