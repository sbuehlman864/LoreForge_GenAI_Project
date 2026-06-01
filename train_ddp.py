"""
DDP entry point for LoreForge training.
Launch with: torchrun --nproc_per_node=NUM_GPUS train_ddp.py
"""

import os
import json
import gc
import torch
import torch.distributed as dist
import pathlib

os.environ["HF_HOME"] = "/projects/e32706/jgu2930/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/projects/e32706/jgu2930/.cache/huggingface/datasets"
os.environ["KAGGLE_USERNAME"] = "spencerbuehlman864"
os.environ["KAGGLE_KEY"] = "f7b244cca98ae261eeae15558513a0c5"

from loreforge import (
    LoreForgeTransformer, load_tokenizer, apply_lora_adapters,
    load_lora_adapter, pretrain, finetune_lora,
    download_gutenberg_corpus, download_harry_potter_books,
    prepare_pretraining_data, prepare_finetuning_data,
    train_bpe_tokenizer, chunk_documents_for_rag,
    embed_passages, build_faiss_index,
    ROOT_DIR, PROCESSED_DIR, RAW_DIR, INDICES_DIR, CHECKPOINTS_DIR, TOKENIZER_PATH,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

UNIVERSES         = ["star_wars", "harry_potter", "lotr"]
PRETRAIN_MAX_DOCS = 2000
PRETRAIN_EPOCHS   = 5
FINETUNE_EPOCHS   = 5
FINETUNE_LR       = 1e-4

# ── DDP Init ──────────────────────────────────────────────────────────────────

dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

if rank == 0:
    print(f"DDP initialized: {world_size} GPUs")

# ── Load best config ──────────────────────────────────────────────────────────

best_config_path = ROOT_DIR / "best_config.json"
with open(best_config_path) as f:
    best_config = json.load(f)

if rank == 0:
    print(f"Config: {best_config}")

# ── Data prep (rank 0 only, others wait) ─────────────────────────────────────

if rank == 0:
    # Tokenizer
    if not TOKENIZER_PATH.exists():
        print("Training tokenizer...")
        gutenberg = download_gutenberg_corpus()
        all_texts = [row["text"] for row in gutenberg]
        train_bpe_tokenizer(all_texts)
        print("Tokenizer saved.")

    # Pretraining binary
    pretrain_bin = PROCESSED_DIR / "pretrain.bin"
    if not pretrain_bin.exists():
        print(f"Building pretrain binary ({PRETRAIN_MAX_DOCS} docs)...")
        gutenberg = download_gutenberg_corpus()
        tokenizer = load_tokenizer()
        prepare_pretraining_data(gutenberg, tokenizer, max_docs=PRETRAIN_MAX_DOCS)
        print("Pretrain binary saved.")

    # Fine-tuning binaries
    SW_KEYWORDS = {
        "star wars", "jedi", "sith", "lightsaber", "skywalker", "darth",
        "stormtrooper", "death star", "the force", "millennium falcon",
        "rebel alliance", "galactic empire", "clone trooper", "mandalorian",
        "wookiee", "coruscant", "tatooine", "dagobah", "galactic republic",
    }
    LOTR_KEYWORDS = {
        "tolkien", "middle-earth", "lord of the rings", "the hobbit",
        "silmarillion", "frodo", "gandalf", "aragorn", "sauron", "mordor",
        "the shire", "rivendell", "rohan", "gondor", "mirkwood", "isengard",
        "arda", "beleriand", "númenor", "numenor",
    }

    tokenizer = load_tokenizer()
    needs_wikipedia = any(u in UNIVERSES for u in ("star_wars", "lotr"))
    if needs_wikipedia:
        print("Downloading Wikipedia...")
        wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    for u in UNIVERSES:
        finetune_bin = PROCESSED_DIR / f"{u}_finetune.bin"
        out_dir = RAW_DIR / u
        out_dir.mkdir(exist_ok=True)
        corpus_txt = out_dir / "corpus.txt"

        if not corpus_txt.exists() or corpus_txt.stat().st_size == 0:
            if u == "harry_potter":
                hp_dir = download_harry_potter_books()
                text = "\n".join(f.read_text(encoding="utf-8") for f in hp_dir.rglob("*.txt"))
            elif u == "star_wars":
                data = wiki_dataset.filter(
                    lambda row: any(kw in row["title"].lower() for kw in SW_KEYWORDS)
                    or sum(kw in row["text"][:500].lower() for kw in SW_KEYWORDS) >= 2
                )
                text = "\n".join(row["text"] for row in data)
            elif u == "lotr":
                data = wiki_dataset.filter(
                    lambda row: any(kw in row["title"].lower() for kw in LOTR_KEYWORDS)
                    or sum(kw in row["text"][:500].lower() for kw in LOTR_KEYWORDS) >= 2
                )
                text = "\n".join(row["text"] for row in data)
            with open(corpus_txt, "w") as f:
                f.write(text)

        if not finetune_bin.exists():
            print(f"Building fine-tune binary for {u}...")
            prepare_finetuning_data(u, out_dir, tokenizer, finetune_bin)

    # FAISS indices
    for u in UNIVERSES:
        index_path = INDICES_DIR / f"{u}.faiss"
        if not index_path.exists():
            print(f"Building FAISS index for {u}...")
            corpus_txt = RAW_DIR / u / "corpus.txt"
            passages = chunk_documents_for_rag([corpus_txt.read_text(encoding="utf-8")], tokenizer)
            embeddings = embed_passages(passages)
            build_faiss_index(u, passages, embeddings)
            print(f"  {u} FAISS index saved ({len(passages)} passages)")

    gc.collect()
    print("Data prep complete.")

# All ranks wait for rank 0 to finish data prep
dist.barrier()
tokenizer = load_tokenizer()

# ── Pretraining ───────────────────────────────────────────────────────────────

pretrain_checkpoint = CHECKPOINTS_DIR / f"pretrain_epoch{PRETRAIN_EPOCHS}.pt"
pretrain_bin = PROCESSED_DIR / "pretrain.bin"

if pretrain_checkpoint.exists():
    if rank == 0:
        print(f"Pretraining checkpoint exists, skipping...")
    model = LoreForgeTransformer(
        vocab_size=best_config["vocab_size"],
        d_model=best_config["d_model"],
        n_layers=best_config["n_layers"],
        n_heads=best_config["n_heads"],
        context_len=best_config["context_len"],
        dropout=best_config["dropout"],
    )
    model.load_state_dict(torch.load(pretrain_checkpoint, weights_only=True))
else:
    if rank == 0:
        print(f"Starting pretraining ({PRETRAIN_EPOCHS} epochs, {world_size} GPUs)...")
    model = LoreForgeTransformer(
        vocab_size=best_config["vocab_size"],
        d_model=best_config["d_model"],
        n_layers=best_config["n_layers"],
        n_heads=best_config["n_heads"],
        context_len=best_config["context_len"],
        dropout=best_config["dropout"],
    )
    if rank == 0:
        print(f"Model parameters: {model.count_parameters():,}")
    model = pretrain(
        model, pretrain_bin,
        context_len=best_config["context_len"],
        batch_size=best_config["batch_size"],
        n_epochs=PRETRAIN_EPOCHS,
        lr=best_config["lr"],
        warmup_steps=1000,
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR,
        rank=rank,
        world_size=world_size,
    )

if rank == 0:
    print("Pretraining complete.")

dist.barrier()

# ── LoRA Fine-tuning ──────────────────────────────────────────────────────────

for u in UNIVERSES:
    adapter_path = CHECKPOINTS_DIR / f"{u}_lora.pt"
    if adapter_path.exists():
        if rank == 0:
            print(f"[LoRA] {u} adapter already exists, skipping...")
        continue

    if rank == 0:
        print(f"[LoRA] Fine-tuning {u}...")

    fresh_model = LoreForgeTransformer(
        vocab_size=best_config["vocab_size"],
        d_model=best_config["d_model"],
        n_layers=best_config["n_layers"],
        n_heads=best_config["n_heads"],
        context_len=best_config["context_len"],
        dropout=best_config["dropout"],
    )
    state_dict = torch.load(
        CHECKPOINTS_DIR / f"pretrain_epoch{PRETRAIN_EPOCHS}.pt",
        weights_only=True,
    )
    fresh_model.load_state_dict(state_dict)
    fresh_model = apply_lora_adapters(fresh_model)

    finetune_lora(
        fresh_model, u,
        bin_path=PROCESSED_DIR / f"{u}_finetune.bin",
        context_len=best_config["context_len"],
        batch_size=best_config["batch_size"],
        n_epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
        device=device,
        checkpoint_dir=CHECKPOINTS_DIR,
        rank=rank,
        world_size=world_size,
    )

    if rank == 0:
        print(f"[LoRA] {u} adapter saved.")

    dist.barrier()

if rank == 0:
    print("All fine-tuning complete.")

dist.destroy_process_group()
