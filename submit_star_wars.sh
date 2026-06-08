#!/bin/bash
# SLURM job: fine-tune GPT-2 LoRA adapter for the Star Wars universe
# Submit with: sbatch submit_star_wars.sh
# Output log:  loreforge_sw_<jobid>.log (written to /projects/e32706/jgu2930/)

#SBATCH --job-name=loreforge_sw
#SBATCH --account=e32706
#SBATCH --partition=gengpu          # GPU partition on Quest
#SBATCH --gres=gpu:a100:1           # 1× NVIDIA A100 (40GB HBM2)
#SBATCH --mem=64G                   # CPU RAM allocation (for DataLoader workers + HF cache)
#SBATCH --time=24:00:00             # 24-hour wall-clock limit
#SBATCH --output=loreforge_sw_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=spencerbuehlman2026@u.northwestern.edu

module purge
module load python-miniconda3/4.12.0  # loads Python + pip into PATH

# Redirect pip --user installs and HuggingFace cache to /projects (large quota).
# Without these, everything lands in $HOME which has a small quota on Quest.
export PYTHONUSERBASE=/projects/e32706/jgu2930/.local
export HF_HOME=/projects/e32706/jgu2930/.cache/huggingface
export HF_DATASETS_CACHE=/projects/e32706/jgu2930/.cache/huggingface/datasets

cd /projects/e32706/jgu2930

# --resume: load existing adapter checkpoint if one exists (allows continuing from epoch N)
# --skip_rag: skip FAISS index building (star_wars already has a pre-existing index)
# --n_epochs 2: star_wars corpus is wiki-based (shorter), 2 epochs is sufficient
python finetune_gpt2.py --universe star_wars --n_epochs 2 --resume --skip_rag
