# =============================================================================
# LoreForge: Multi-Universe Lore-Faithful Story Generation
# =============================================================================
# Main pipeline: data download → preprocessing → tokenizer → pretraining →
# LoRA fine-tuning → RAG index construction → inference
#
# Author: Spencer Lepine
# Course: Generative AI — Northwestern MSAI
# =============================================================================

import os
import re
import json
import math
import time
import shutil
import pathlib
import requests


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.optim import AdamW

import ray.train
from ray import tune
from ray.tune.schedulers import ASHAScheduler


from datasets import load_dataset          # HuggingFace datasets
from tokenizers import Tokenizer           # HuggingFace tokenizers (BPE)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from sentence_transformers import SentenceTransformer
import faiss

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR        = pathlib.Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
INDICES_DIR     = DATA_DIR / "indices"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
TOKENIZER_PATH  = ROOT_DIR / "tokenizer.json"

for _dir in [RAW_DIR, PROCESSED_DIR, INDICES_DIR, CHECKPOINTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Universe registry ──────────────────────────────────────────────────────────

UNIVERSES = {
    "star_wars":   {"control_token": "[STAR_WARS]",    "status": "required"},
    "harry_potter":{"control_token": "[HARRY_POTTER]", "status": "required"},
    "lotr":        {"control_token": "[LOTR]",         "status": "required"},
    "tlou":        {"control_token": "[TLOU]",         "status": "stretch"},
}

# ── RAG config ─────────────────────────────────────────────────────────────────

RAG_CHUNK_TOKENS  = 256
RAG_CHUNK_OVERLAP = 32
RAG_TOP_K         = 5
RAG_EMBED_MODEL   = "all-MiniLM-L6-v2"

# ── Hyperband (Ray Tune) note ──────────────────────────────────────────────────
# Hyperparameters (n_layers, n_heads, d_model, lr, batch_size, etc.) will be
# determined via Hyperband search using Ray Tune. The search space and trial
# function are defined in the Pretraining section below.


# =============================================================================
# 1. DATA DOWNLOAD
# =============================================================================

def download_gutenberg_corpus(split: str = "en") -> object:
    """Download the Project Gutenberg pretraining corpus from HuggingFace.

    Uses the `manu/project_gutenberg` dataset (~70 k public-domain English
    novels). The returned dataset object can be iterated or passed directly
    to the preprocessing functions below.

    Args:
        split: Dataset split to load. Splits are by language; use "en" for English.

    Returns:
        A HuggingFace Dataset object with at minimum a "text" column.
    """
    dataset = load_dataset("manu/project_gutenberg", split=split)
    return dataset


def download_harry_potter_books(dest_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    """Download the Harry Potter books corpus from Kaggle.

    Downloads all seven books as plain .txt files via the Kaggle API.
    Dataset: https://www.kaggle.com/datasets/rupanshukapoor/harry-potter-books
    License: MIT (educational/research use only)

    Setup:
        1. pip install kaggle
        2. Create a Kaggle API token at https://www.kaggle.com/settings → API
        3. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json
        4. chmod 600 ~/.kaggle/kaggle.json

    Args:
        dest_dir: Directory to download and unzip the books into.

    Returns:
        Path to the directory containing the extracted .txt book files.
    """
    import kaggle
    out_path = dest_dir / "harry_potter_books"
    out_path.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "rupanshukapoor/harry-potter-books",
        path=str(out_path),
        unzip=True,
    )
    return out_path


def download_star_wars_corpus(split: str = "train") -> object:
    """Download and filter Star Wars-related articles from the Wikipedia HuggingFace dataset.

    Source: https://huggingface.co/datasets/wikimedia/wikipedia (English, 20231101)
    License: CC BY-SA 3.0

    Role in the pipeline:
        Fine-tuning and RAG — provides encyclopedic lore entries for Star Wars
        characters, planets, factions, and events. Filtered by title and lead
        paragraph to keep only articles that are clearly about Star Wars canon.

    Args:
        split: Dataset split to load (only "train" exists for Wikipedia).

    Returns:
        A HuggingFace Dataset object filtered to Star Wars-related articles,
        with columns: id, url, title, text. Use the "text" column for
        chunking and embedding in build_faiss_index().
    """
    SW_KEYWORDS = {
        "star wars", "jedi", "sith", "lightsaber", "skywalker", "darth",
        "stormtrooper", "death star", "the force", "millennium falcon",
        "rebel alliance", "galactic empire", "clone trooper", "mandalorian",
        "wookiee", "coruscant", "tatooine", "dagobah", "galactic republic",
    }

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)

    def is_sw_article(row):
        title_lower = row["title"].lower()
        if any(kw in title_lower for kw in SW_KEYWORDS):
            return True
        text_lower = row["text"][:500].lower()
        return sum(kw in text_lower for kw in SW_KEYWORDS) >= 2

    sw_dataset = dataset.filter(is_sw_article)
    return sw_dataset


def download_lotr_books() -> object:
    """Download the full Lord of the Rings trilogy text from HuggingFace.

    Source: https://huggingface.co/datasets/jeremyarancio/lotr-book
    Content: Pages 45–1055 of the LOTR trilogy as a single continuous text
             block, with headers and footers stripped.
    License: Unstated — Tolkien's work is copyrighted. Use for educational
             and research purposes only.

    Role in the pipeline:
        FINE-TUNING — teaches the LoRA adapter Tolkien's actual prose style:
        archaic diction, elevated register, elvish names and phrases, and the
        specific narrative rhythm of Middle-earth. This is the style signal
        that makes generated text feel like Tolkien rather than generic fantasy.

    Returns:
        A HuggingFace Dataset object with a single "text" column containing
        the full trilogy as one string. Chunk and tokenize in
        prepare_finetuning_data() before training.
    """
    dataset = load_dataset("jeremyarancio/lotr-book")
    return dataset


def download_lotr_wikipedia(split: str = "train") -> object:
    """Download and filter LOTR-related articles from the Wikipedia HuggingFace dataset.

    Source: https://huggingface.co/datasets/wikimedia/wikipedia (English, 20231101)
    License: CC BY-SA 3.0

    Role in the pipeline:
        RAG — provides structured, encyclopedic lore entries for retrieval at
        inference time. Wikipedia's LOTR coverage includes dedicated articles
        for major characters (Frodo, Gandalf, Aragorn), locations (The Shire,
        Mordor, Rivendell), factions, artifacts, and events. These read like
        lore wiki entries, making them ideal context chunks: the model gets a
        factual grounding passage and generates narrative prose around it.
        This complements the book text (used for fine-tuning style) by
        providing clean, retrievable facts rather than scattered narrative.

    Args:
        split: Dataset split to load (only "train" exists for Wikipedia).

    Returns:
        A HuggingFace Dataset object filtered to LOTR-related articles,
        with columns: id, url, title, text. Use the "text" column for
        chunking and embedding in build_faiss_index().
    """
    LOTR_KEYWORDS = {
        "tolkien", "middle-earth", "lord of the rings", "the hobbit",
        "silmarillion", "frodo", "gandalf", "aragorn", "sauron", "mordor",
        "the shire", "rivendell", "rohan", "gondor", "mirkwood", "isengard",
        "arda", "beleriand", "númenor", "numenor",
    }

    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)

    # Filter by title first (fast), then fall back to text content for edge cases
    def is_lotr_article(row):
        title_lower = row["title"].lower()
        if any(kw in title_lower for kw in LOTR_KEYWORDS):
            return True
        text_lower = row["text"][:500].lower()  # check only the lead paragraph
        return sum(kw in text_lower for kw in LOTR_KEYWORDS) >= 2

    lotr_dataset = dataset.filter(is_lotr_article)
    return lotr_dataset


def download_tlou_corpus(dest_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    """[STUB] Acquire The Last of Us community game scripts (stretch goal).

    Community-sourced transcripts for TLOU Parts I and II, supplemented by
    the TLOU wiki. These do not have a single canonical download source; they
    must be assembled manually.

    Manual steps:
        1. Locate community script transcripts (fan sites, GitHub repos, etc.).
        2. Download the TLOU wiki dump from the relevant fandom/wiki export.
        3. Combine and place all text files under: data/raw/tlou/

    Args:
        dest_dir: Directory to store raw corpus files.

    Returns:
        Expected directory path.

    Raises:
        NotImplementedError: Stretch goal — not required for initial submission.
    """
    expected_path = dest_dir / "tlou"
    raise NotImplementedError(
        "TLOU corpus is a stretch goal. Assemble scripts manually and place under "
        f"{expected_path}"
    )


# =============================================================================
# 2. PREPROCESSING
# =============================================================================

def clean_wiki_markup(text: str) -> str:
    """Strip MediaWiki markup, templates, infoboxes, and metadata from raw text.

    Applied to all FandomCorpus and Tolkien Gateway documents before any
    further processing. Should remove:
        - {{template}} blocks
        - [[wikilinks]] (keeping display text)
        - [external links]
        - HTML tags (<ref>, <gallery>, etc.)
        - Category / File / Image prefixes
        - Infobox table syntax

    Args:
        text: Raw wiki markup string.

    Returns:
        Clean plain-English prose string.
    """
    # Strip {{}} blocks
    while '{{' in text:
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    
    # Drop tags whose content should be removed entirely
    text = re.sub(r'<(ref|gallery|math|score)[^>]*>.*?</\1>', '', text, flags=re.DOTALL)

    # Strip all remaining HTML tags (keep the text between them)
    text = re.sub(r'<[^>]+>', '', text)

    text = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', lambda m: m.group(2), text)  # [[target|display]] → display
    text = re.sub(r'\[\[([^\]]+)\]\]', lambda m: m.group(1), text)              # [[target]] → target

    text = re.sub(r'\[https?://[^\]]*\]', '', text)

    text = re.sub(r'^\s*(\{\||\|\}|\|!?|!)[^\n]*', '', text, flags=re.MULTILINE) # Strip wiki table syntax

    text = re.sub(r"'{2,3}", '', text)          # '''bold''' and ''italic''
    text = re.sub(r'={2,6}[^=\n]+={2,6}', '', text)  # == Headings ==
    text = re.sub(r'^\s*[*#:;]+', '', text, flags=re.MULTILINE)  # bullets and list markers

    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse 3+ blank lines to 2
    text = text.strip()

    return text


def prepare_pretraining_data(
    dataset,
    tokenizer: "Tokenizer",
    out_path: pathlib.Path = PROCESSED_DIR / "pretrain.bin",
) -> pathlib.Path:
    """Tokenize the Gutenberg corpus and write a flat binary token file for pretraining.

    Concatenates all documents with an EOS token between them, encodes with the
    trained BPE tokenizer, and memory-maps the result to disk as a uint16 numpy
    array for efficient streaming during training.

    Args:
        dataset:   HuggingFace Dataset returned by download_gutenberg_corpus().
        tokenizer: Trained BPE Tokenizer (see train_bpe_tokenizer()).
        out_path:  Destination .bin file.

    Returns:
        Path to the written binary token file.
    """
    eos_id = tokenizer.token_to_id("[EOS]")

    # Write incrementally in chunks to avoid loading entire corpus into RAM
    CHUNK_SIZE = 100_000
    buffer = []
    with open(out_path, "wb") as f:
        for row in dataset:
            ids = tokenizer.encode(row["text"]).ids
            buffer.extend(ids)
            buffer.append(eos_id)
            if len(buffer) >= CHUNK_SIZE:
                np.array(buffer, dtype=np.uint16).tofile(f)
                buffer = []
        if buffer:
            np.array(buffer, dtype=np.uint16).tofile(f)

    return out_path



def prepare_finetuning_data(
    universe: str,
    raw_path: pathlib.Path,
    tokenizer: "Tokenizer",
    out_path: pathlib.Path = None,
) -> pathlib.Path:
    """Clean, prepend universe control token, and tokenize a universe corpus for LoRA fine-tuning.

    Cleans wiki markup (if applicable), prepends the universe control token
    defined in UNIVERSES[universe]["control_token"], and writes a binary token
    file analogous to the pretraining file.

    Args:
        universe:  Key from UNIVERSES (e.g. "star_wars").
        raw_path:  Path to the raw dump file or directory.
        tokenizer: Trained BPE Tokenizer.
        out_path:  Destination .bin file. Defaults to data/processed/<universe>_finetune.bin.

    Returns:
        Path to the written binary token file.
    """
    control_token = UNIVERSES[universe]["control_token"]
    control_id = tokenizer.token_to_id(control_token)
    eos_id = tokenizer.token_to_id("[EOS]")

    if raw_path.is_dir():
        texts = [f.read_text(encoding="utf-8") for f in raw_path.glob("*.txt")]
    else:
        texts = [raw_path.read_text(encoding="utf-8")]

    all_tokens = []
    for text in texts:
        text = clean_wiki_markup(text)
        ids = tokenizer.encode(text).ids
        all_tokens.append(control_id)   # prepend control token
        all_tokens.extend(ids)
        all_tokens.append(eos_id)

    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(out_path)

    return out_path



def chunk_documents_for_rag(
    documents: list[str],
    tokenizer: "Tokenizer",
    chunk_size: int = RAG_CHUNK_TOKENS,
    overlap: int = RAG_CHUNK_OVERLAP,
) -> list[str]:
    """Split cleaned documents into overlapping token-bounded passages for RAG indexing.

    Each passage is at most `chunk_size` tokens, with `overlap` tokens of
    context carried over from the previous chunk to avoid splitting mid-thought.

    Args:
        documents:  List of clean plain-text strings (post clean_wiki_markup).
        tokenizer:  Trained BPE Tokenizer used to measure token counts.
        chunk_size: Maximum tokens per passage (default 256).
        overlap:    Token overlap between consecutive passages (default 32).

    Returns:
        Flat list of passage strings ready for embedding.
    """
    chunks = []
    for doc in documents:
        ids = tokenizer.encode(doc).ids
        step = chunk_size - overlap
        for start in range(0, len(ids), step):
            chunk_ids = ids[start : start + chunk_size]
            chunks.append(tokenizer.decode(chunk_ids))
    return chunks



# =============================================================================
# 3. TOKENIZER
# =============================================================================

def train_bpe_tokenizer(
    texts,
    vocab_size: int = 16_000,
    save_path: pathlib.Path = TOKENIZER_PATH,
) -> "Tokenizer":
    """Train a BPE tokenizer on the combined pretraining + fine-tuning corpus.

    Uses HuggingFace `tokenizers` with a Whitespace pre-tokenizer. Adds the
    four universe control tokens as special tokens so they are never split.
    Saves the trained tokenizer to disk.

    Args:
        texts:      Iterable of raw strings (Gutenberg + all universe corpora).
        vocab_size: Target vocabulary size (default 16 000; tune as needed).
        save_path:  Where to write the tokenizer JSON.

    Returns:
        Trained and saved Tokenizer object.
    """
    # Create tokenizer and trainer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Define our UNK and EOS tokens
    special_tokens = ["[UNK]", "[EOS]"] + [v["control_token"] for v in UNIVERSES.values()]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Save and return
    tokenizer.save(str(save_path))
    return tokenizer



def load_tokenizer(path: pathlib.Path = TOKENIZER_PATH) -> "Tokenizer":
    """Load a previously trained BPE tokenizer from disk.

    Args:
        path: Path to the tokenizer JSON saved by train_bpe_tokenizer().

    Returns:
        Tokenizer object ready for encode/decode.
    """
    return Tokenizer.from_file(str(path))


# =============================================================================
# 4. MODEL ARCHITECTURE
# =============================================================================

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an autoregressive mask.

    Standard scaled dot-product attention where each position can only attend
    to itself and earlier positions (upper-triangle mask). Supports optional
    injection of LoRA delta weights on the Q and V projections.
    """

    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float = 0.1):
        """
        Args:
            d_model:     Model (embedding) dimension.
            n_heads:     Number of attention heads. Must divide d_model evenly.
            context_len: Maximum sequence length; used to register the causal mask.
            dropout:     Attention dropout probability.
        """
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False) # projects input to Q,K, and V
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask of upper triangle of -inf so future positions masked out
        mask = torch.triu(torch.full((context_len, context_len), float('-inf')), diagonal=1)
        self.register_buffer("mask", mask)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        B, T, C = x.shape # batch, seq_len, d_model

        # Project to Q,K,V and split
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        def reshape(t):
            return t.view(B,T, self.n_heads, self.head_dim).transpose(1,2)
        
        q, k, v = reshape(q), reshape(k), reshape(v)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5 # sqrt d_k
        attention = (q @ k.transpose(-2,-1)) * scale
        attention = attention + self.mask[:T,:T] # Apply causal mask
        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # Combine heads and project out
        output = (attention @ v).transpose(1,2).contiguous().view(B,T,C)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    """A single decoder-only transformer block: LayerNorm → Attention → LayerNorm → FFN.

    Uses pre-norm (norm before sublayer) following the GPT-2 convention.
    FFN hidden dimension is 4 × d_model.
    """

    def __init__(self, d_model: int, n_heads: int, context_len: int, dropout: float = 0.1):
        """
        Args:
            d_model:     Model dimension.
            n_heads:     Number of attention heads.
            context_len: Sequence length for causal mask.
            dropout:     Dropout probability applied after attention and FFN.
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffnn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(), # Gaussian Error Linear Unit, smoother than ReLu 
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.ffnn(self.ln2(x))
        return x


class LoreForgeTransformer(nn.Module):
    """Decoder-only GPT-style transformer for causal language modeling.

    Embedding → N × TransformerBlock → LayerNorm → LM head (tied weights).
    Target: ~50M parameters with config (6L, 8H, 512D, 2048 ctx).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        context_len: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size:  Size of the BPE vocabulary (+ special tokens).
            d_model:     Embedding dimension.
            n_layers:    Number of stacked TransformerBlocks.
            n_heads:     Attention heads per block.
            context_len: Maximum sequence length (context window).
            dropout:     Dropout probability throughout the model.
        """
        super().__init__()

        self.context_len = context_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, context_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights so token embedding and lm_head share same matrix
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a forward pass and optionally compute cross-entropy loss.

        Args:
            input_ids: Token indices of shape (batch, seq_len).
            targets:   Shifted token indices for loss computation (same shape).
                       If None, only logits are returned.

        Returns:
            (logits, loss) where loss is None if targets is None.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)

        x = self.dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
        return logits, loss

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 5. PRETRAINING
# =============================================================================

class PretrainDataset(Dataset):
    """Memory-mapped dataset over the flat binary token file produced by prepare_pretraining_data.

    Streams (context_len + 1)-token windows from the file without loading the
    full corpus into RAM — essential for large corpora on Quest.
    """

    def __init__(self, bin_path: pathlib.Path, context_len: int):
        """
        Args:
            bin_path:    Path to the .bin token file.
            context_len: Number of tokens per training sample (x = tokens[i:i+ctx],
                         y = tokens[i+1:i+ctx+1]).
        """
        # Memory-map the binary file so only accessed parts loaded to RAM
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.context_len = context_len

    def __len__(self) -> int:
        return len(self.data) - self.context_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = torch.from_numpy(self.data[idx : idx + self.context_len + 1].astype(np.int64))
        x = chunk[:-1] # all tokens except last
        y = chunk[1:] # all tokens except the first (shifted by one)
        return x, y


def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Return a cosine decay scheduler with linear warmup.

    Warmup increases LR from 0 to max_lr over `warmup_steps`, then cosine
    decays to ~0 over the remaining steps.

    Args:
        optimizer:    The AdamW optimizer to wrap.
        warmup_steps: Number of linear warmup steps.
        total_steps:  Total number of training steps.

    Returns:
        A torch.optim.lr_scheduler.LambdaLR scheduler.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: LoreForgeTransformer,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one full pass over the pretraining dataloader and return mean loss.

    Args:
        model:      The transformer model.
        dataloader: Pretraining DataLoader.
        optimizer:  AdamW optimizer.
        scheduler:  LR scheduler (stepped per batch).
        device:     torch.device ("cuda" or "cpu").
        grad_clip:  Gradient norm clipping threshold.

    Returns:
        Mean cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for x,y in dataloader:
        x,y = x.to(device), y.to(device)
    
        optimizer.zero_grad()
        _, loss = model(x,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def pretrain(
    model: LoreForgeTransformer,
    bin_path: pathlib.Path,
    context_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    warmup_steps: int,
    device: torch.device,
    checkpoint_dir: pathlib.Path = CHECKPOINTS_DIR,
) -> LoreForgeTransformer:
    """Full pretraining loop with checkpointing.

    Saves a checkpoint after each epoch to checkpoint_dir/pretrain_epoch{n}.pt.

    Args:
        model:          Initialized LoreForgeTransformer.
        bin_path:       Path to the pretraining token binary.
        context_len:    Sequence length (must match model).
        batch_size:     Samples per gradient step.
        n_epochs:       Number of full passes over the corpus.
        lr:             Peak learning rate for AdamW.
        warmup_steps:   Linear warmup steps.
        device:         Training device.
        checkpoint_dir: Where to write epoch checkpoints.

    Returns:
        Trained model (weights updated in place; also returned for convenience).
    """
    dataset = PretrainDataset(bin_path, context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = n_epochs * len(dataloader)
    scheduler = build_lr_schedule(optimizer, warmup_steps, total_steps)
    model = model.to(device)

    for epoch in range(1, n_epochs  + 1):
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch} loss: {loss:.4f}")
        torch.save(model.state_dict(), checkpoint_dir / f"pretrain_epoch{epoch}.pt")
    return model

def ray_train_wrapper(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LoreForgeTransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        context_len=config["context_len"],
        dropout=config["dropout"],
    ).to(device)

    dataset = PretrainDataset(config["bin_path"], config["context_len"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    total_steps = config["max_epochs"] * len(dataloader)
    optimizer = AdamW(model.parameters(), lr=config["lr"])
    scheduler = build_lr_schedule(optimizer, warmup_steps=100, total_steps=total_steps)

    for epoch in range(config["max_epochs"]):
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        ray.train.report({"loss": loss})

def hyperband_search(
    train_fn,
    config_space: dict,
    n_samples: int = 20,
    max_epochs: int = 10,
) -> dict:
    """[STUB] Run a Hyperband search over pretraining hyperparameters using Ray Tune.

    Ray Tune's ASHAScheduler implements Hyperband-style early stopping. It will
    trial many (lr, batch_size, d_model, n_layers, n_heads, dropout) configs,
    aggressively pruning poor runs and allocating more compute to promising ones.

    Setup:
        pip install ray[tune]

    Expected usage:
        best_config = hyperband_search(
            train_fn=ray_train_wrapper,   # a function(config) → {"loss": float}
            config_space={
                "lr":        tune.loguniform(1e-4, 1e-2),
                "batch_size": tune.choice([32, 64, 128]),
                "d_model":   tune.choice([256, 512]),
                "n_layers":  tune.choice([4, 6, 8]),
                "n_heads":   tune.choice([4, 8]),
                "dropout":   tune.uniform(0.05, 0.2),
            },
        )

    Args:
        train_fn:     A callable that accepts a config dict and reports metrics
                      via ray.train.report({"loss": ...}).
        config_space: Dict of Ray Tune search space objects defining the
                      hyperparameter ranges to explore.
        n_samples:    Number of total trials to run.
        max_epochs:   Maximum epochs any single trial is allowed to run before
                      Hyperband forces early stopping.

    Returns:
        Best config dict found (hyperparameter values that minimized loss).

    Raises:
        NotImplementedError: Until Ray Tune integration is wired up.
    """
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        train_fn,
        param_space=config_space,
        tune_config=tune.TuneConfig(
            num_samples=n_samples,
            scheduler=scheduler,
            max_concurrent_trials=1,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")
    return best.config


# =============================================================================
# 6. LoRA FINE-TUNING
# =============================================================================

class LoRALinear(nn.Module):
    """A linear layer augmented with a low-rank LoRA delta: W' = W + (B @ A) * (alpha / r).

    During fine-tuning only A and B are trained; W is frozen. At inference the
    delta is merged or applied on the fly depending on the selected universe.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        """
        Args:
            linear: The frozen pretrained Linear layer to wrap.
            rank:   LoRA rank r. Lower = fewer trainable params.
            alpha:  LoRA scaling factor. Effective scale = alpha / rank.
        """
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale)


def apply_lora_adapters(
    model: LoreForgeTransformer,
    rank: int = 8,
    alpha: float = 16.0,
) -> LoreForgeTransformer:
    """Replace Q and V projection layers in every attention block with LoRALinear wrappers.

    Freezes all original parameters, then marks only the LoRA A/B matrices as
    trainable. Call this before fine-tuning on a universe corpus.

    Args:
        model: Pretrained LoreForgeTransformer (weights will be frozen).
        rank:  LoRA rank.
        alpha: LoRA alpha scaling.

    Returns:
        The same model object with LoRA wrappers applied in place.
    """
    for param in model.parameters():
        param.requires_grad = False
    
    for block in model.blocks:
        block.attention.qkv = LoRALinear(block.attention.qkv , rank, alpha)
        block.attention.out_proj = LoRALinear(block.attention.out_proj, rank, alpha)
    return model

def finetune_lora(
    model: LoreForgeTransformer,
    universe: str,
    bin_path: pathlib.Path,
    context_len: int,
    batch_size: int,
    n_epochs: int,
    lr: float,
    device: torch.device,
    checkpoint_dir: pathlib.Path = CHECKPOINTS_DIR,
) -> LoreForgeTransformer:
    """Fine-tune the LoRA adapters on a single universe corpus.

    The base model weights stay frozen; only LoRA A/B matrices are updated.
    Saves adapter weights (not full model) after each epoch to
    checkpoint_dir/<universe>_lora_epoch{n}.pt.

    Args:
        model:          LoreForgeTransformer with LoRA adapters already applied.
        universe:       Key from UNIVERSES (used for checkpoint naming).
        bin_path:       Path to the universe fine-tuning token binary.
        context_len:    Sequence length.
        batch_size:     Samples per step.
        n_epochs:       Fine-tuning epochs.
        lr:             AdamW learning rate (typically smaller than pretraining lr).
        device:         Training device.
        checkpoint_dir: Where to save adapter checkpoints.

    Returns:
        Model with fine-tuned LoRA adapters.
    """
    dataset = PretrainDataset(bin_path, context_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = build_lr_schedule(optimizer, warmup_steps=100, total_steps=n_epochs * len(dataloader))

    model.to(device)
    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, dataloader, optimizer, scheduler, device)
        print(f"Epoch {epoch} loss: {loss : .4f}")

        save_lora_adapter(model, universe, checkpoint_dir)


    return model


def save_lora_adapter(
    model: LoreForgeTransformer,
    universe: str,
    path: pathlib.Path = CHECKPOINTS_DIR,
) -> pathlib.Path:
    """Extract and save only the LoRA adapter weights for a given universe.

    Saves a dict of {param_name: tensor} containing only A and B matrices so
    the full model checkpoint does not need to be duplicated per universe.

    Args:
        model:    Model with LoRA adapters applied.
        universe: Universe key — used to name the output file.
        path:     Directory to write <universe>_lora.pt into.

    Returns:
        Path to the saved adapter file.
    """
    adapter_weights = {
        k: v for k, v in model.state_dict().items() if "lora_" in k
    }
    out_path = path / f"{universe}_lora.pt"
    torch.save(adapter_weights, out_path)
    return out_path


def load_lora_adapter(
    model: LoreForgeTransformer,
    universe: str,
    path: pathlib.Path = CHECKPOINTS_DIR,
) -> LoreForgeTransformer:
    """Load saved LoRA adapter weights into a model that already has LoRA wrappers applied.

    Call apply_lora_adapters() on the base model first, then this function to
    restore the universe-specific A/B matrices.

    Args:
        model:    LoreForgeTransformer with LoRA wrappers (A/B initialized but untrained).
        universe: Universe key — used to locate <universe>_lora.pt.
        path:     Directory containing the adapter checkpoint.

    Returns:
        Model with the universe adapter weights loaded.
    """
    out_path = path / f"{universe}_lora.pt"

    adapter_weights = torch.load(out_path, weights_only=True)
    model.load_state_dict(adapter_weights, strict=False)
    return model



# =============================================================================
# 7. RAG
# =============================================================================

def embed_passages(
    passages: list[str],
    embed_model_name: str = RAG_EMBED_MODEL,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    """Embed a list of text passages using a sentence-transformer model.

    Args:
        passages:         List of clean passage strings from chunk_documents_for_rag().
        embed_model_name: SentenceTransformer model name (default all-MiniLM-L6-v2).
        batch_size:       Encoding batch size.
        device:           Device for the embedding model.

    Returns:
        Float32 numpy array of shape (n_passages, embedding_dim).
    """
    model = SentenceTransformer(embed_model_name, device=device)
    result = model.encode(passages, batch_size=batch_size, convert_to_numpy=True)

    return result.astype('float32')


def build_faiss_index(
    universe: str,
    passages: list[str],
    embeddings: np.ndarray,
    index_dir: pathlib.Path = INDICES_DIR,
) -> pathlib.Path:
    """Build and save a FAISS flat L2 index for a universe's passage embeddings.

    Also saves a parallel JSON list of passage strings so retrieved embeddings
    can be mapped back to readable text at inference time.

    Args:
        universe:    Universe key — used to name output files.
        passages:    The raw passage strings (same order as embeddings rows).
        embeddings:  (n_passages, dim) float32 array from embed_passages().
        index_dir:   Directory to write <universe>.faiss and <universe>_passages.json.

    Returns:
        Path to the saved .faiss index file.
    """
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)

    index_path = index_dir / f"{universe}.faiss"
    faiss.write_index(idx, str(index_path))

    passages_path = index_dir / f"{universe}_passages.json"
    with open(passages_path, "w") as f:
        json.dump(passages, f)

    return index_path


def load_faiss_index(
    universe: str,
    index_dir: pathlib.Path = INDICES_DIR,
) -> tuple["faiss.Index", list[str]]:
    """Load a FAISS index and its parallel passage list from disk.

    Args:
        universe:  Universe key.
        index_dir: Directory containing <universe>.faiss and <universe>_passages.json.

    Returns:
        (faiss_index, passages) tuple.
    """
    idx = faiss.read_index(str(index_dir / f"{universe}.faiss"))
    with open(index_dir / f"{universe}_passages.json") as f:
        passages_list = json.load(f)

    return(idx, passages_list)


def retrieve_context(
    query: str,
    universe: str,
    faiss_index: "faiss.Index",
    passages: list[str],
    embed_model_name: str = RAG_EMBED_MODEL,
    k: int = RAG_TOP_K,
) -> list[str]:
    """Embed a query and return the top-k most relevant lore passages.

    Args:
        query:            The user's story prompt (plain text).
        universe:         Universe key (for logging / future per-universe model choice).
        faiss_index:      Loaded FAISS index for the selected universe.
        passages:         Parallel passage list returned by load_faiss_index().
        embed_model_name: Embedding model to use (must match what built the index).
        k:                Number of passages to retrieve.

    Returns:
        List of k passage strings, ordered by relevance (most relevant first).
    """
    embed_query = embed_passages([query], embed_model_name=embed_model_name)
    distances, indices = faiss_index.search(embed_query, k)

    return [passages[i] for i in indices[0]]


# =============================================================================
# 8. INFERENCE
# =============================================================================

def build_generation_prompt(
    user_prompt: str,
    retrieved_passages: list[str],
    universe: str,
) -> str:
    """Assemble the full prompt sent to the model at generation time.

    Format:
        [UNIVERSE_TOKEN]
        --- Lore Context ---
        <passage 1>
        <passage 2>
        ...
        --- Story ---
        <user_prompt>

    Args:
        user_prompt:        The raw prompt entered by the user.
        retrieved_passages: List of lore passages from retrieve_context().
        universe:           Universe key (used to look up the control token).

    Returns:
        Formatted prompt string ready for tokenization and generation.
    """
    
    pass


@torch.no_grad()
def generate_story(
    prompt: str,
    universe: str,
    model: LoreForgeTransformer,
    tokenizer: "Tokenizer",
    faiss_index: "faiss.Index",
    passages: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_k: int = 50,
    device: torch.device = None,
) -> dict:
    """Run the full RAG + generation pipeline for a user prompt and universe.

    Steps:
        1. Retrieve top-k lore passages via retrieve_context().
        2. Build the generation prompt via build_generation_prompt().
        3. Tokenize the prompt.
        4. Autoregressively sample up to max_new_tokens from the model.
        5. Decode and return generated text + retrieved passages.

    Args:
        prompt:         User's story prompt.
        universe:       Selected universe key.
        model:          LoreForgeTransformer with the appropriate LoRA adapter loaded.
        tokenizer:      Trained BPE Tokenizer.
        faiss_index:    Loaded FAISS index for the selected universe.
        passages:       Parallel passage strings for the selected universe.
        max_new_tokens: Maximum tokens to generate beyond the prompt.
        temperature:    Sampling temperature (higher = more creative).
        top_k:          Top-k sampling cutoff (0 = disabled).
        device:         Inference device. Defaults to model's current device.

    Returns:
        Dict with keys:
            "generated_text":      The model's story continuation (decoded string).
            "retrieved_passages":  The lore passages used as context.
            "full_prompt":         The assembled prompt sent to the model.
    """
    pass


def run_training_pipeline(
    universes: list[str],
    n_hyperband_samples: int = 20,
    pretrain_max_epochs: int = 10,
    finetune_epochs: int = 3,
    finetune_lr: float = 1e-4,
) -> LoreForgeTransformer:
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/8] Device: {device}")

    # Download Gutenberg corpus
    print("[2/8] Downloading Gutenberg corpus...")
    gutenberg = download_gutenberg_corpus()
    print(f"      Gutenberg loaded: {len(gutenberg)} documents")

    # Download Wikipedia once if either star_wars or lotr needs it, filter for both
    universe_data = {}
    needs_wikipedia = any(u in universes for u in ("star_wars", "lotr"))
    if needs_wikipedia:
        print("[3/8] Downloading Wikipedia (single pass for Star Wars + LOTR)...")
        wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

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

        if "star_wars" in universes:
            universe_data["star_wars"] = wiki_dataset.filter(
                lambda row: any(kw in row["title"].lower() for kw in SW_KEYWORDS)
                or sum(kw in row["text"][:500].lower() for kw in SW_KEYWORDS) >= 2
            )
            print(f"      Star Wars articles: {len(universe_data['star_wars'])}")
        if "lotr" in universes:
            universe_data["lotr"] = wiki_dataset.filter(
                lambda row: any(kw in row["title"].lower() for kw in LOTR_KEYWORDS)
                or sum(kw in row["text"][:500].lower() for kw in LOTR_KEYWORDS) >= 2
            )
            print(f"      LOTR articles: {len(universe_data['lotr'])}")
    else:
        print("[3/8] Skipping Wikipedia download (not needed for selected universes)")

    if "harry_potter" in universes:
        print("      Downloading Harry Potter books...")
        universe_data["harry_potter"] = download_harry_potter_books()
        print("      Harry Potter books downloaded")

    # Build tokenizer — load from disk if already trained
    print("[4/8] Preparing tokenizer...")
    all_texts = [row["text"] for row in gutenberg]
    for u, data in universe_data.items():
        if u == "harry_potter":
            all_texts += [f.read_text(encoding="utf-8") for f in data.glob("*.txt")]
        else:
            all_texts += [row["text"] for row in data]

    if TOKENIZER_PATH.exists():
        print("      Loading existing tokenizer from disk...")
        tokenizer = load_tokenizer()
    else:
        print("      Training new tokenizer...")
        tokenizer = train_bpe_tokenizer(all_texts)
    print("      Tokenizer ready")

    # Write pretraining binary — skip if already exists
    print("[5/8] Preparing pretraining binary...")
    binary_path = PROCESSED_DIR / "pretrain.bin"
    if not binary_path.exists():
        binary_path = prepare_pretraining_data(gutenberg, tokenizer)
        print(f"      Pretraining binary written to {binary_path}")
    else:
        print("      Pretraining binary already exists, skipping...")

    # Write universe text to disk as .txt files and prepare fine-tuning binaries
    print("[6/8] Preparing fine-tuning binaries...")
    finetune_paths = {}
    for u in universes:
        out_dir = RAW_DIR / u
        out_dir.mkdir(exist_ok=True)
        if u == "harry_potter":
            text = "\n".join(f.read_text(encoding="utf-8") for f in universe_data[u].glob("*.txt"))
        else:
            text = "\n".join(row["text"] for row in universe_data[u])
        with open(out_dir / "corpus.txt", "w") as f:
            f.write(text)

        finetune_bin = PROCESSED_DIR / f"{u}_finetune.bin"
        if not finetune_bin.exists():
            print(f"Preparing fine-tuning binary for {u}...")
            finetune_paths[u] = prepare_finetuning_data(
                universe=u,
                raw_path=RAW_DIR / u,
                tokenizer=tokenizer,
                out_path=finetune_bin,
            )
        else:
            print(f"Fine-tuning binary for {u} already exists, skipping...")
            finetune_paths[u] = finetune_bin

    # Set Hyperband search space
    hyperparam_space = {
        "lr" : tune.loguniform(1e-4, 1e-2),
        "batch_size" : tune.choice([32, 64, 128]),
        "d_model" : tune.choice([256, 512]),
        "n_layers" : tune.choice([4,6,8]),
        "n_heads" : tune.choice([4,8]),
        "dropout" : tune.uniform(0.05, 0.3),
        # fixed — use shorter context for hyperband trials to save memory
        "vocab_size": 16_000,
        "context_len": 512,
        "bin_path": binary_path,
        "max_epochs": pretrain_max_epochs,
    }

    # Free large objects from memory before Ray workers spin up
    import gc
    del gutenberg, universe_data, all_texts
    gc.collect()

    # Find the best hyperparam config with hyperband — skip if already saved
    best_config_path = ROOT_DIR / "best_config.json"
    if best_config_path.exists():
        print("[7/8] Loading existing best config from disk...")
        with open(best_config_path) as f:
            best_config = json.load(f)
        print(f"      Best config: {best_config}")
    else:
        print("[7/8] Running Hyperband search...")
        best_config = hyperband_search(ray_train_wrapper, hyperparam_space, n_hyperband_samples)
        print(f"      Best config: {best_config}")
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print("      Best config saved to best_config.json")

    # Pretrain model
    print(f"[8/8] Pretraining model ({pretrain_max_epochs} epochs)...")
    model = LoreForgeTransformer(best_config["vocab_size"], best_config["d_model"], best_config["n_layers"], best_config["n_heads"], best_config["context_len"], best_config["dropout"])
    print(f"      Model parameters: {model.count_parameters():,}")
    model = pretrain(model, binary_path, best_config["context_len"], best_config["batch_size"], pretrain_max_epochs, best_config["lr"], 1000, device, CHECKPOINTS_DIR)
    print("      Pretraining complete")

    # Fine tune on universes
    for i, u in enumerate(universes):
        print(f"[LoRA {i+1}/{len(universes)}] Fine-tuning {u} adapter...")
        fresh_model = LoreForgeTransformer(
            vocab_size=best_config["vocab_size"],
            d_model=best_config["d_model"],
            n_layers=best_config["n_layers"],
            n_heads=best_config["n_heads"],
            context_len=best_config["context_len"],
            dropout=best_config["dropout"],
        )
        state_dict = torch.load(
            CHECKPOINTS_DIR / f"pretrain_epoch{pretrain_max_epochs}.pt",
            weights_only=True
        )
        fresh_model.load_state_dict(state_dict)
        fresh_model = apply_lora_adapters(fresh_model)
        fresh_model = finetune_lora(fresh_model, u, finetune_paths[u], best_config["context_len"], best_config["batch_size"], finetune_epochs, finetune_lr, device, CHECKPOINTS_DIR)
        print(f"      {u} adapter saved")

    for i, u in enumerate(universes):
        print(f"[RAG {i+1}/{len(universes)}] Building FAISS index for {u}...")
        if u == "harry_potter":
            rag_passages = chunk_documents_for_rag(
                [f.read_text(encoding="utf-8") for f in universe_data[u].glob("*.txt")],
                tokenizer
            )
        else:
            rag_passages = chunk_documents_for_rag(
                [row["text"] for row in universe_data[u]],
                tokenizer
            )
        print(f"      {len(rag_passages)} passages chunked, embedding...")
        embeddings = embed_passages(rag_passages)
        build_faiss_index(u, rag_passages, embeddings)
        print(f"      {u} FAISS index saved")

    print("Training pipeline complete.")
    return fresh_model
