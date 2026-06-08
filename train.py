# =============================================================================
# train.py — Entry point for the from-scratch LoreForge training pipeline
# =============================================================================
# This script launches the full pipeline defined in loreforge.py:
#   1. Downloads all corpora (Gutenberg, Wikipedia, HP books, LOTR)
#   2. Trains a BPE tokenizer on the combined corpus
#   3. Runs Hyperband HPO via Ray Tune to find the best model config
#   4. Pretrains the LoreForgeTransformer on the Gutenberg corpus
#   5. LoRA fine-tunes the pretrained model per universe
#   6. Builds FAISS RAG indices for each universe
#
# Run on Quest via: sbatch submit.sh
# Note: pretrain_max_docs=2000 limits Gutenberg to 2000 documents for
# faster experimentation; remove the cap for full pretraining.
# =============================================================================

import os

# Redirect HuggingFace and Kaggle caches to the project directory on Quest.
# The home directory has a small quota; /projects has much more space.
os.environ["HF_HOME"] = "/projects/e32706/jgu2930/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/projects/e32706/jgu2930/.cache/huggingface/datasets"
os.environ["KAGGLE_USERNAME"] = "spencerbuehlman864"
os.environ["KAGGLE_KEY"] = "f7b244cca98ae261eeae15558513a0c5"

from loreforge import run_training_pipeline

# n_hyperband_samples=8: run 8 Hyperband trials to find the best config.
# pretrain_max_epochs=2: each trial trains for at most 2 epochs before Hyperband prunes.
# pretrain_max_docs=2000: use a 2000-doc subset of Gutenberg for faster HPO trials.
trained_model = run_training_pipeline(
    universes=["star_wars", "harry_potter", "lotr"],
    n_hyperband_samples=8,
    pretrain_max_epochs=2,
    finetune_epochs=3,
    finetune_lr=1e-4,
    pretrain_max_docs=2000,
)

print("Training pipeline complete.")
