import os

os.environ["HF_HOME"] = "/projects/e32706/jgu2930/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/projects/e32706/jgu2930/.cache/huggingface/datasets"
os.environ["KAGGLE_USERNAME"] = "spencerbuehlman864"
os.environ["KAGGLE_KEY"] = "f7b244cca98ae261eeae15558513a0c5"

from loreforge import run_training_pipeline

trained_model = run_training_pipeline(
    universes=["star_wars", "harry_potter", "lotr"],
    n_hyperband_samples=8,
    pretrain_max_epochs=2,
    finetune_epochs=3,
    finetune_lr=1e-4,
    pretrain_max_docs=2000,
)

print("Training pipeline complete.")
