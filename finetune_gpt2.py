# =============================================================================
# finetune_gpt2.py — SLURM entry point for single-universe GPT-2 LoRA fine-tuning
# =============================================================================
# Designed to be submitted as an independent SLURM job, one per universe.
# Running three jobs in parallel (star_wars, harry_potter, lotr) cuts wall-clock
# time to the slowest single job rather than the sum of all three.
#
# Each job:
#   1. Loads the frozen GPT-2 base model and applies LoRA adapters
#   2. Optionally resumes from an existing adapter checkpoint (--resume)
#   3. Fine-tunes the LoRA adapters on the universe corpus
#   4. Optionally builds a FAISS RAG index for the universe (--skip_rag to bypass)
#
# Usage (on Quest):
#   sbatch submit_star_wars.sh    # --universe star_wars --n_epochs 2 --resume --skip_rag
#   sbatch submit_harry_potter.sh # --universe harry_potter --n_epochs 3
#   sbatch submit_lotr.sh         # --universe lotr --n_epochs 3
# =============================================================================

import argparse
import os
import torch
from transformers import GPT2LMHeadModel

# Must be set BEFORE importing transformers/HuggingFace libraries so the cache
# is redirected to /projects (large quota) instead of $HOME (small quota on Quest)
os.environ["HF_HOME"] = "/projects/e32706/jgu2930/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/projects/e32706/jgu2930/.cache/huggingface/datasets"
os.environ["KAGGLE_USERNAME"] = "spencerbuehlman864"
os.environ["KAGGLE_KEY"] = "f7b244cca98ae261eeae15558513a0c5"

from loreforge_gpt2 import (
    load_gpt2_tokenizer, apply_lora_adapters_gpt2,
    load_lora_adapter, finetune_lora_gpt2, save_lora_adapter,
    build_faiss_index, embed_passages, chunk_documents_for_rag,
    load_corpus_texts,
    CHECKPOINTS_DIR, INDICES_DIR, GPT2_MODEL_NAME,
)

parser = argparse.ArgumentParser()
parser.add_argument("--universe",  required=True, choices=["star_wars", "harry_potter", "lotr"])
parser.add_argument("--n_epochs",  type=int, default=3)
parser.add_argument("--lr",        type=float, default=1e-4)
parser.add_argument("--resume",    action="store_true", help="Load existing adapter before training")
parser.add_argument("--skip_rag",  action="store_true", help="Skip FAISS index building")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Universe: {args.universe} | Epochs: {args.n_epochs} | Resume: {args.resume}")

# Load the GPT-2 base model and wrap attention projections with LoRA adapters.
# apply_lora_adapters_gpt2 freezes all base weights and makes only A/B trainable.
tokenizer = load_gpt2_tokenizer(GPT2_MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME)
model = apply_lora_adapters_gpt2(model)

if args.resume:
    # --resume allows continuing a previously interrupted fine-tuning run.
    # Useful when the SLURM job hit its time limit before completing all epochs.
    adapter_path = CHECKPOINTS_DIR / f"{args.universe}_gpt2_lora.pt"
    if adapter_path.exists():
        print(f"Resuming from existing adapter: {adapter_path}")
        model = load_lora_adapter(model, args.universe, CHECKPOINTS_DIR)
    else:
        print("No existing adapter found, starting from scratch.")

model = finetune_lora_gpt2(
    model, args.universe, tokenizer,
    n_epochs=args.n_epochs,
    lr=args.lr,
    device=device,
    checkpoint_dir=CHECKPOINTS_DIR,
)

print(f"Adapter saved for {args.universe}.")

# Build FAISS RAG index after fine-tuning so inference can retrieve lore passages.
# --skip_rag is used when the index already exists (e.g. for star_wars which had
# a pre-existing index from an earlier run).
if not args.skip_rag:
    index_path = INDICES_DIR / f"{args.universe}_gpt2.faiss"
    if not index_path.exists():
        print(f"Building FAISS index for {args.universe}...")
        texts = load_corpus_texts(args.universe)
        # Chunk → embed → index pipeline: splits corpus into 256-token passages,
        # embeds with all-MiniLM-L6-v2, and saves a flat L2 FAISS index to disk
        passages = chunk_documents_for_rag(texts, tokenizer)
        embeddings = embed_passages(passages)
        build_faiss_index(args.universe, passages, embeddings)
        print(f"FAISS index saved ({len(passages)} passages).")
    else:
        print(f"FAISS index already exists, skipping.")

print("Done.")
