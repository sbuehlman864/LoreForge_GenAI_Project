# LoreForge: Multi-Universe Lore-Faithful Story Generation

LoreForge is a generative AI system that produces lore-faithful short stories set in the Star Wars, Harry Potter, and Lord of the Rings universes. It combines fine-tuned language models with Retrieval-Augmented Generation (RAG) to ground outputs in canon lore.

---

## How to Run

### Model Weights

The pretrained GPT-2 LoRA adapter weights (all three universes) are hosted on Google Drive:

**[Download weights from Google Drive](https://drive.google.com/drive/folders/1Dy5blXhRucwaBuT6Vrq_SADiKKE1Ae8Z?usp=sharing)**

Download all three `.pt` files and place them in `data/checkpoints/`:

```
data/checkpoints/
├── star_wars_gpt2_lora.pt
├── harry_potter_gpt2_lora.pt
└── lotr_gpt2_lora.pt
```

### Prerequisites

```bash
pip install torch transformers sentence-transformers faiss-cpu fastapi uvicorn
```

### 1. Start the inference server

```bash
python server.py
# Server runs at http://localhost:8000
```

### 2. Start the React frontend

```bash
cd gui
npm install
npm run dev
# Open http://localhost:5173
```

The server auto-detects which universes have trained adapters and marks them as available in the UI. Select a universe, enter a prompt, and click Generate.

### Keyboard shortcut

`⌘ + Enter` (Mac) or `Ctrl + Enter` (Windows) submits a prompt.

---

## Project Overview

LoreForge went through two major iterations. The first attempted to train a transformer completely from scratch with Hyperband hyperparameter optimization. Due to Quest HPC constraints, the project pivoted to a pretrained GPT-2 base in the second iteration.

---

## Iteration 1: From-Scratch Transformer

**Files:** [`loreforge.py`](loreforge.py), [`train.py`](train.py), [`submit.sh`](submit.sh)

### Goal

Train a custom GPT-style decoder-only transformer from scratch on the Project Gutenberg corpus, then LoRA fine-tune it per universe for lore-faithful generation.

### Model Architecture — LoreForgeTransformer

A custom decoder-only transformer built in PyTorch, targeting ~50M parameters.

| Hyperparameter | Target Value |
|---|---|
| Layers (`n_layers`) | 6 |
| Attention heads (`n_heads`) | 8 |
| Model dimension (`d_model`) | 512 |
| Context length | 2048 tokens |
| FFN hidden size | 4 × d_model = 2048 |
| Vocabulary size | 16,000 (custom BPE) |

**Components:**
- `CausalSelfAttention` — multi-head scaled dot-product attention with a causal (upper-triangular) mask. Uses PyTorch's `scaled_dot_product_attention` (FlashAttention when available)
- `TransformerBlock` — pre-norm formulation (LayerNorm before each sublayer) with residual connections
- `LoreForgeTransformer` — token + position embeddings → N× TransformerBlock → LayerNorm → LM head with tied weights

### Tokenizer

A custom BPE tokenizer (16k vocab) trained on the combined Gutenberg + universe corpus using HuggingFace `tokenizers`. Universe control tokens (`[STAR_WARS]`, `[HARRY_POTTER]`, `[LOTR]`) are added as special tokens so they are never split by BPE.

### Hyperband HPO (Ray Tune)

Hyperparameters were tuned using Ray Tune's `ASHAScheduler` (Asynchronous Successive Halving — a Hyperband variant). The search space covered:
- Learning rate: log-uniform `[1e-4, 1e-2]`
- Batch size: `{32, 64, 128}`
- `d_model`: `{256, 512}`
- `n_layers`: `{4, 6, 8}`
- `n_heads`: `{4, 8}`
- Dropout: uniform `[0.05, 0.2]`

ASHAScheduler aggressively prunes poor-performing trials early, allocating more compute to promising configurations.

### Pretraining

The model was pretrained on the `manu/project_gutenberg` dataset (~70k public-domain English novels) using a binary memory-mapped dataset (`data/processed/pretrain.bin`). Checkpoints are saved after each epoch as `checkpoints/pretrain_epoch{n}.pt`.

### LoRA Fine-tuning (Scratch Model)

After pretraining, LoRA adapters were applied to the QKV and output projection layers of each attention block (rank=8, alpha=16). Only the LoRA A/B matrices (~300K params) are trained per universe; the base model weights are frozen.

### What Went Wrong

**Hyperband HPO took too long.** Each trial required running pretraining from scratch on the Gutenberg corpus. On Quest, individual A100 GPU jobs had queue wait times of 24+ hours. With 8 Hyperband trials and multiple epochs each, the HPO phase alone consumed the majority of the project timeline.

**Pretraining from scratch was too slow.** Even with the `pretrain_max_docs=2000` limit, full pretraining required compute that couldn't be completed within the project deadline.

**Decision:** Pivot to a pretrained GPT-2 base model that already has English fluency, and only fine-tune LoRA adapters (~hours per universe, not days).

---

## Iteration 2: Pretrained GPT-2

**Files:** [`loreforge_gpt2.py`](loreforge_gpt2.py), [`finetune_gpt2.py`](finetune_gpt2.py), [`submit_star_wars.sh`](submit_star_wars.sh), [`submit_harry_potter.sh`](submit_harry_potter.sh), [`submit_lotr.sh`](submit_lotr.sh), [`server.py`](server.py), [`gui/src/App.jsx`](gui/src/App.jsx)

### Goal

Replace the from-scratch pretraining step with OpenAI's pretrained GPT-2 (117M params), then fine-tune lightweight LoRA adapters per universe. The RAG pipeline and inference architecture remain unchanged.

### GPT-2 Architecture

GPT-2 is a decoder-only transformer pretrained by OpenAI on 40GB of web text (WebText corpus).

| Property | Value |
|---|---|
| Model ID | `gpt2` (HuggingFace) |
| Total parameters | 117M (frozen during fine-tuning) |
| Transformer layers | 12 |
| Attention heads | 12 per layer |
| Hidden size (`d_model`) | 768 |
| FFN hidden size | 3,072 (4 × d_model) |
| Context window | 1,024 tokens (hard limit) |
| Vocabulary | 50,257 BPE tokens |
| Attention projection | `c_attn`: 768→2,304 (QKV fused), `c_proj`: 768→768 |
| Layer implementation | `Conv1D` (weight shape `[in, out]`, opposite of `nn.Linear`) |

**Why the Conv1D distinction matters:** GPT-2's attention layers use `transformers.Conv1D` internally, which stores weights transposed compared to `nn.Linear`. The `LoRALinear` wrapper in `loreforge_gpt2.py` detects which type it is wrapping and handles both shapes correctly.

### LoRA Fine-tuning (GPT-2)

LoRA adapters are applied to `c_attn` and `c_proj` in all 12 transformer blocks:
- Rank: 8
- Alpha: 16 (effective scale = 16/8 = 2.0)
- Trainable parameters: ~300K per universe (vs 117M base weights)
- Training: AdamW, cosine decay with 100-step linear warmup, gradient clipping at 1.0

The base GPT-2 weights are frozen. Only the LoRA A/B matrices are updated.

### SLURM Fine-tuning Jobs

Three independent jobs run in parallel on Quest, one per universe:

| Script | Universe | Epochs | Notes |
|---|---|---|---|
| `submit_star_wars.sh` | star_wars | 2 | `--resume --skip_rag` (index pre-exists) |
| `submit_harry_potter.sh` | harry_potter | 3 | Builds FAISS index after training |
| `submit_lotr.sh` | lotr | 3 | Builds FAISS index after training |

Each job requests 1× A100 GPU, 64GB RAM, 24-hour time limit on the `gengpu` partition.

**Quest environment issues encountered:**
- `finetune_gpt2.py` missing from cluster → copied with `scp`
- Home directory quota exceeded by NVIDIA CUDA packages → redirected via `PYTHONUSERBASE` and `HF_HOME` to `/projects`
- HuggingFace model download cache defaulting to `$HOME` → fixed by setting env vars before Python import

### RAG Pipeline

Retrieval-Augmented Generation grounds each story in canon lore by fetching relevant passages before calling the model.

**Step 1 — Chunking** (`chunk_documents_for_rag`): The raw corpus is tokenized and split into 256-token windows with 32-token overlap to avoid cutting mid-sentence.

**Step 2 — Embedding** (`embed_passages`): Each passage is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2` (sentence-transformers). Semantically similar passages cluster together in this space.

**Step 3 — Indexing** (`build_faiss_index`): All embeddings are stored in a FAISS `IndexFlatL2` (exact nearest-neighbour search). The index and parallel passage strings are saved to `data/indices/{universe}_gpt2.faiss` and `{universe}_gpt2_passages.json`.

**Step 4 — Retrieval** (`retrieve_context`): At inference time, the user prompt is embedded and the top-k=5 nearest passages are returned.

**Step 5 — Prompt Assembly** (`build_generation_prompt`): Retrieved passages are injected into the prompt with an explicit narrative instruction to prevent the model from generating encyclopedia-style output (a known issue when training on wiki/fandom corpora).

### Datasets

#### Pretraining (from-scratch iteration only)

| Dataset | Source | License | Use |
|---|---|---|---|
| `manu/project_gutenberg` | HuggingFace | Public domain | Base model pretraining (~70k novels) |

#### Star Wars

| Dataset | Source | License | Use |
|---|---|---|---|
| `lara-martin/Scifi_TV_Shows` (filtered) | HuggingFace | CC-BY-4.0 | Fine-tuning + RAG — ~270 Star Wars stories scraped from the Star Wars Fandom wiki |

#### Harry Potter

| Dataset | Source | License | Use |
|---|---|---|---|
| `rupanshukapoor/harry-potter-books` | Kaggle | MIT (educational/research only) | Fine-tuning + RAG — Full text of all seven HP books (~2.5 MB) |

#### Lord of the Rings

| Dataset | Source | License | Use |
|---|---|---|---|
| `jeremyarancio/lotr-book` | HuggingFace | Educational/research only | Fine-tuning — Full LOTR trilogy (pages 45–1055) |
| `wikimedia/wikipedia` (filtered) | HuggingFace | CC BY-SA 3.0 | RAG — English Wikipedia filtered to LOTR-related articles |

### Inference Server (`server.py`)

A FastAPI server that:
- Loads models lazily on first request per universe (avoids loading all three at startup)
- Auto-detects the available backend (GPT-2 adapter vs from-scratch adapter)
- Exposes `GET /universes` and `POST /generate` endpoints
- Supports CORS for the React frontend

### React GUI (`gui/src/App.jsx`)

A single-page chat interface built with React + Vite:
- Universe picker with per-universe theme colors
- Chat history with animated loading indicator
- Collapsible lore passages panel showing which canon text was retrieved
- Settings controls: max tokens, temperature, RAG on/off toggle
- `⌘/Ctrl + Enter` keyboard shortcut

### Output Quality Issues

**Problem:** Generated output often resembled encyclopedia articles rather than narrative fiction.

**Root cause:** The fine-tuning corpus (Star Wars wiki articles, Harry Potter Fandom articles) was largely encyclopedic text with citation markers (`[48]`, `[B]`). GPT-2 learned to generate in that style.

**Mitigations applied:**
- Added `repetition_penalty=1.3` and `no_repeat_ngram_size=3` to `model.generate()` to break repetition collapse
- Added an explicit narrative instruction to `build_generation_prompt` telling the model to write as a story, not an article

**Remaining limitation:** The prompt framing helps but cannot fully overcome the style learned from training data. Retraining on narrative fiction datasets (fan fiction, actual book text without wiki markup) would be the proper fix.

---

## File Reference

| File | Purpose |
|---|---|
| `loreforge.py` | Full from-scratch pipeline: data, tokenizer, transformer, Hyperband HPO, LoRA, RAG, inference |
| `train.py` | Entry point that calls `run_training_pipeline()` in `loreforge.py` |
| `loreforge_gpt2.py` | GPT-2 pipeline: LoRA fine-tuning on frozen GPT-2, RAG, inference |
| `finetune_gpt2.py` | SLURM-friendly CLI entry point for per-universe GPT-2 fine-tuning |
| `server.py` | FastAPI inference server serving the React GUI |
| `gui/src/App.jsx` | React single-page chat interface |
| `submit_star_wars.sh` | SLURM job: Star Wars fine-tuning (2 epochs, resume, skip RAG) |
| `submit_harry_potter.sh` | SLURM job: Harry Potter fine-tuning (3 epochs) |
| `submit_lotr.sh` | SLURM job: Lord of the Rings fine-tuning (3 epochs) |
| `submit.sh` | SLURM job: original from-scratch DDP pretraining (4× A100) |

---

## Training Notes

### LoRA Epochs Matter

After 1 epoch, model output was incoherent — universe vocabulary was present but grammar was broken. The LoRA B matrix (initialized to zero) had not converged enough to shift the output distribution coherently. A minimum of 3 epochs is required.

### Adapter Checkpoint Loading

Adapters saved on Quest CUDA GPUs are loaded with `map_location="cpu"` on a CPU-only inference machine. `load_lora_adapter` in `loreforge_gpt2.py` handles this automatically.

### Context Window Truncation

GPT-2 has a hard 1024-token context window. The RAG prompt (lore passages + user prompt) can exceed this. `generate_story` truncates the input to `1024 - max_new_tokens` tokens before calling `model.generate` to prevent an out-of-range position embedding error.

### Conv1D Compatibility

GPT-2 uses `Conv1D` layers (weight shape `[in, out]`) rather than `nn.Linear` (`[out, in]`). The `LoRALinear` wrapper detects which type it is wrapping by comparing the two dimensions and sets `in_features`/`out_features` accordingly.
