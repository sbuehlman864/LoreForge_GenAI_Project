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
