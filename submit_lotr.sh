#!/bin/bash
# SLURM job: fine-tune GPT-2 LoRA adapter for the Lord of the Rings universe
# Submit with: sbatch submit_lotr.sh
# Output log:  loreforge_lotr_<jobid>.log (written to /projects/e32706/jgu2930/)

#SBATCH --job-name=loreforge_lotr
#SBATCH --account=e32706
#SBATCH --partition=gengpu          # GPU partition on Quest
#SBATCH --gres=gpu:a100:1           # 1× NVIDIA A100 (40GB HBM2)
#SBATCH --mem=64G                   # CPU RAM for DataLoader workers + HF cache
#SBATCH --time=24:00:00             # 24-hour wall-clock limit
#SBATCH --output=loreforge_lotr_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=spencerbuehlman2026@u.northwestern.edu

module purge
module load python-miniconda3/4.12.0  # loads Python + pip into PATH

# Redirect pip --user installs and HuggingFace cache to /projects (large quota)
export PYTHONUSERBASE=/projects/e32706/jgu2930/.local
export HF_HOME=/projects/e32706/jgu2930/.cache/huggingface
export HF_DATASETS_CACHE=/projects/e32706/jgu2930/.cache/huggingface/datasets

cd /projects/e32706/jgu2930

# 3 epochs on the full LOTR trilogy corpus (pages 45–1055, jeremyarancio/lotr-book).
# Also builds the FAISS index after fine-tuning (no --skip_rag flag).
python finetune_gpt2.py --universe lotr --n_epochs 3
