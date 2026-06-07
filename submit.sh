#!/bin/bash
#SBATCH --job-name=loreforge_ddp
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=loreforge_%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@u.northwestern.edu

module purge
module load python-miniconda3/4.12.0

cd /projects/e32706/jgu2930

torchrun --nproc_per_node=4 train_ddp.py
