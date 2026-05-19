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
import json
import math
import time
import shutil
import pathlib
import requests

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

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

def download_gutenberg_corpus(split: str = "train") -> object:
    """Download the Project Gutenberg pretraining corpus from HuggingFace.

    Uses the `manu/project_gutenberg` dataset (~70 k public-domain English
    novels). The returned dataset object can be iterated or passed directly
    to the preprocessing functions below.

    Args:
        split: Dataset split to load. Nearly all usable text is in "train".

    Returns:
        A HuggingFace Dataset object with at minimum a "text" column.
    """
    dataset = load_dataset("manu/project_gutenberg", split=split)
    return dataset


def download_fandom_corpus(universe: str, dest_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    """[STUB] Download a FandomCorpus preprocessed wiki dump for a given universe.

    FandomCorpus provides pre-cleaned XML dumps for Wookieepedia (Star Wars)
    and the Harry Potter Wiki. Download the files manually from the URL below,
    place them in dest_dir, and update the return path accordingly.

    Manual steps:
        1. Visit https://datamanagementlab.github.io/fandomCorpus/data.html
        2. Locate the dump for `universe` ("star_wars" → Wookieepedia,
           "harry_potter" → Harry Potter Wiki).
        3. Download the .xml or .json archive and place it at:
               data/raw/<universe>_fandom_dump.<ext>
        4. Call prepare_finetuning_data(universe, ...) to continue.

    Args:
        universe: One of the keys in UNIVERSES ("star_wars", "harry_potter").
        dest_dir: Directory to store the raw download.

    Returns:
        Expected path where the dump should be placed.

    Raises:
        NotImplementedError: Until manual download is complete.
    """
    expected_path = dest_dir / f"{universe}_fandom_dump"
    raise NotImplementedError(
        f"Download the FandomCorpus dump for '{universe}' manually from "
        "https://datamanagementlab.github.io/fandomCorpus/data.html "
        f"and place it at {expected_path}"
    )


def export_tolkien_gateway(dest_dir: pathlib.Path = RAW_DIR) -> pathlib.Path:
    """[STUB] Export the Tolkien Gateway wiki via MediaWiki Special:Export.

    Tolkien Gateway is licensed CC BY-SA 4.0. The full wiki can be exported
    as an XML dump through the MediaWiki export interface.

    Manual steps:
        1. Visit https://tolkiengateway.net/w/index.php?title=Special:Export
        2. Export all pages (or use the bulk export URL — see note below).
        3. Save the resulting XML as: data/raw/lotr_tolkiengateway.xml
        4. Call prepare_finetuning_data("lotr", ...) to continue.

    Bulk export note:
        For a full dump, prefer downloading from the site's database dumps page
        if available, or use a MediaWiki scraper such as `wikiteam3` to crawl
        all pages:
            pip install wikiteam3
            wikiteam3dumpgenerator --api https://tolkiengateway.net/w/api.php

    Args:
        dest_dir: Directory to store the raw XML dump.

    Returns:
        Expected path where the dump should be placed.

    Raises:
        NotImplementedError: Until manual export is complete.
    """
    expected_path = dest_dir / "lotr_tolkiengateway.xml"
    raise NotImplementedError(
        f"Export Tolkien Gateway manually (see docstring) and place the XML at {expected_path}"
    )


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
    pass


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
    pass


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
    if out_path is None:
        out_path = PROCESSED_DIR / f"{universe}_finetune.bin"
    pass


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
    pass


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
    pass


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
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        pass


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
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


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
        pass

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
        pass

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

