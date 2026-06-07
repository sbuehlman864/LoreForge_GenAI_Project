"""
Single-universe GPT-2 LoRA fine-tuning script.
Designed to be called from a SLURM job, one job per universe.

Usage:
    python finetune_gpt2.py --universe star_wars --n_epochs 2 --resume
    python finetune_gpt2.py --universe harry_potter --n_epochs 3
    python finetune_gpt2.py --universe lotr --n_epochs 3
"""

import argparse
import os
import torch
from transformers import GPT2LMHeadModel

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

tokenizer = load_gpt2_tokenizer(GPT2_MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL_NAME)
model = apply_lora_adapters_gpt2(model)

if args.resume:
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

# Build FAISS index if not already present
if not args.skip_rag:
    index_path = INDICES_DIR / f"{args.universe}_gpt2.faiss"
    if not index_path.exists():
        print(f"Building FAISS index for {args.universe}...")
        texts = load_corpus_texts(args.universe)
        passages = chunk_documents_for_rag(texts, tokenizer)
        embeddings = embed_passages(passages)
        build_faiss_index(args.universe, passages, embeddings)
        print(f"FAISS index saved ({len(passages)} passages).")
    else:
        print(f"FAISS index already exists, skipping.")

print("Done.")
