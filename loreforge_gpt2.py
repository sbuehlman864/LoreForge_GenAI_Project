# =============================================================================
# LoreForge GPT-2: Multi-Universe Lore-Faithful Story Generation
# =============================================================================
# ITERATION 2 — PRETRAINED GPT-2 BASE MODEL
#
# This file is the production pipeline. It replaces the from-scratch transformer
# in loreforge.py with OpenAI's pretrained GPT-2 (117M parameters), eliminating
# the need for pretraining from scratch entirely.
#
# WHY GPT-2 INSTEAD OF SCRATCH:
#   The from-scratch model required Hyperband HPO + full pretraining on Quest,
#   which could not complete within the course timeline due to GPU queue delays.
#   GPT-2 already has strong English fluency from pretraining on 40GB of web
#   text — we only need to fine-tune the LoRA adapters (~300K params/universe)
#   to steer it toward universe-specific vocabulary and style.
#
# GPT-2 ARCHITECTURE (model ID: "gpt2", HuggingFace transformers):
#   Type:            Decoder-only autoregressive transformer
#   Parameters:      117M total (frozen during fine-tuning)
#   Layers:          12 transformer blocks
#   Attention heads: 12 heads per block
#   Hidden size:     768 (d_model)
#   FFN hidden size: 3072 (4 × d_model)
#   Context window:  1024 tokens (hard limit from positional embedding size)
#   Vocabulary:      50,257 BPE tokens (GPT-2 tokenizer)
#   Attention proj:  c_attn: 768→2304 (QKV fused), c_proj: 768→768
#   Layer type:      Conv1D (weight shape [in, out]) not nn.Linear ([out, in])
#   Pretraining:     OpenAI WebText corpus (~40GB, ~8M web pages)
#
# LoRA is applied to c_attn and c_proj in all 12 blocks (rank=8, alpha=16),
# adding ~300K trainable parameters per universe instead of fine-tuning all 117M.
#
# Author: Spencer Lepine
# Course: Generative AI — Northwestern MSAI
# =============================================================================

import os
import re
import json
import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
import faiss

# ── Paths (shared with loreforge.py) ──────────────────────────────────────────

ROOT_DIR        = pathlib.Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
INDICES_DIR     = DATA_DIR / "indices"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

for _dir in [RAW_DIR, PROCESSED_DIR, INDICES_DIR, CHECKPOINTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Universe registry ──────────────────────────────────────────────────────────

UNIVERSES = {
    "star_wars":    {"control_token": "[STAR_WARS]",    "status": "required"},
    "harry_potter": {"control_token": "[HARRY_POTTER]", "status": "required"},
    "lotr":         {"control_token": "[LOTR]",         "status": "required"},
    "tlou":         {"control_token": "[TLOU]",         "status": "stretch"},
}

# ── RAG config ─────────────────────────────────────────────────────────────────

RAG_CHUNK_TOKENS  = 256
RAG_CHUNK_OVERLAP = 32
RAG_TOP_K         = 5
RAG_EMBED_MODEL   = "all-MiniLM-L6-v2"

# ── GPT-2 config ───────────────────────────────────────────────────────────────

# "gpt2" = smallest variant (117M). Larger variants have more layers/heads/d_model
# but require more GPU memory and take longer to fine-tune.
GPT2_MODEL_NAME = "gpt2"   # options: "gpt2", "gpt2-medium"(345M), "gpt2-large"(762M), "gpt2-xl"(1.5B)
CONTEXT_LEN     = 512      # training sequence length; must be ≤ GPT-2's 1024-token hard limit
BATCH_SIZE      = 8        # per-GPU batch size for fine-tuning on A100 (64G)


# =============================================================================
# 1. TOKENIZER
# =============================================================================

def load_gpt2_tokenizer(model_name: str = GPT2_MODEL_NAME) -> GPT2Tokenizer:
    """Load the GPT-2 BPE tokenizer from HuggingFace.

    Args:
        model_name: GPT-2 variant to load tokenizer for.

    Returns:
        GPT2Tokenizer with pad_token set to eos_token.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # GPT-2 has no dedicated pad token — set it to eos_token so the DataCollator
    # and model.generate() don't error on padded sequences
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# =============================================================================
# 2. DATASET
# =============================================================================

class GPT2FinetuneDataset(Dataset):
    """Dataset that reads from a pre-built binary token file using GPT-2 token ids.

    Reuses the same .bin files built by loreforge.py's prepare_finetuning_data,
    but re-tokenizes from the raw corpus text using the GPT-2 tokenizer since
    the existing bins use the custom BPE vocab.
    """

    def __init__(self, texts: list[str], tokenizer: GPT2Tokenizer, context_len: int = CONTEXT_LEN):
        """
        Args:
            texts:       List of raw corpus strings.
            tokenizer:   GPT2Tokenizer.
            context_len: Sequence length for each training sample.
        """
        self.context_len = context_len
        # Join documents with eos_token as boundary marker, then encode the full concatenation.
        # This mirrors how GPT-2 was pretrained on WebText (documents separated by <|endoftext|>).
        combined = tokenizer.eos_token.join(texts)
        self.token_ids = tokenizer.encode(combined)

    def __len__(self) -> int:
        return max(0, len(self.token_ids) - self.context_len)

    def __getitem__(self, idx: int):
        # Sliding window: x = tokens[i..i+ctx-1], y = tokens[i+1..i+ctx] (next-token targets)
        chunk = torch.tensor(self.token_ids[idx : idx + self.context_len + 1], dtype=torch.long)
        return chunk[:-1], chunk[1:]


def load_corpus_texts(universe: str) -> list[str]:
    """Load raw corpus text for a universe from disk.

    Args:
        universe: Universe key.

    Returns:
        List of text strings from the universe corpus.
    """
    corpus_path = RAW_DIR / universe / "corpus.txt"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}. Run the main pipeline first.")
    return [corpus_path.read_text(encoding="utf-8")]


# =============================================================================
# 3. LoRA
# =============================================================================

class LoRALinear(nn.Module):
    """LoRA wrapper compatible with both nn.Linear and GPT-2's Conv1D.

    W' = W + (B @ A) * (alpha / rank)
    """

    def __init__(self, layer, rank: int = 8, alpha: float = 16.0):
        """
        Args:
            layer: nn.Linear or transformers.Conv1D to wrap.
            rank:  LoRA rank.
            alpha: LoRA scaling factor.
        """
        super().__init__()
        self.layer = layer
        self.scale = alpha / rank  # effective LoRA scaling factor applied to the delta

        # GPT-2 internally uses Conv1D (a 1D convolution with kernel_size=1) for its
        # attention projections. Conv1D stores weights as (in_features, out_features),
        # which is the TRANSPOSE of nn.Linear's (out_features, in_features).
        # We detect which type we're wrapping by checking which dimension is larger.
        w = layer.weight
        if w.shape[0] < w.shape[1]:    # Conv1D: weight is (in, out)
            in_features, out_features = w.shape
        else:                           # nn.Linear: weight is (out, in)
            out_features, in_features = w.shape

        # A: initialized non-zero (Kaiming uniform) to give a gradient signal from step 1
        # B: initialized to zero so the net delta ΔW = B@A starts at zero (no disturbance)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through frozen base layer, then add the low-rank LoRA delta
        return self.layer(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale


def apply_lora_adapters_gpt2(
    model: GPT2LMHeadModel,
    rank: int = 8,
    alpha: float = 16.0,
) -> GPT2LMHeadModel:
    """Freeze all GPT-2 weights and apply LoRA to attention projection layers.

    Targets c_attn (QKV combined) and c_proj (output projection) in every
    transformer block.

    Args:
        model: Pretrained GPT2LMHeadModel.
        rank:  LoRA rank.
        alpha: LoRA scaling factor.

    Returns:
        Model with LoRA adapters applied and base weights frozen.
    """
    # Freeze all 117M GPT-2 base weights — only LoRA A/B matrices will be updated
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to c_attn (QKV combined, 768→2304) and c_proj (output, 768→768)
    # in each of GPT-2's 12 transformer blocks. These are the layers most responsible
    # for what the model "attends to" — steering them is sufficient for style adaptation.
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, rank, alpha)
        block.attn.c_proj = LoRALinear(block.attn.c_proj, rank, alpha)

    return model


def save_lora_adapter(
    model: GPT2LMHeadModel,
    universe: str,
    path: pathlib.Path = CHECKPOINTS_DIR,
) -> pathlib.Path:
    """Save LoRA adapter weights for a universe.

    Args:
        model:    GPT-2 model with LoRA adapters applied.
        universe: Universe key.
        path:     Directory to save into.

    Returns:
        Path to saved adapter file.
    """
    adapter_weights = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    out_path = path / f"{universe}_gpt2_lora.pt"
    torch.save(adapter_weights, out_path)
    return out_path


def load_lora_adapter(
    model: GPT2LMHeadModel,
    universe: str,
    path: pathlib.Path = CHECKPOINTS_DIR,
) -> GPT2LMHeadModel:
    """Load saved LoRA adapter weights into a GPT-2 model.

    Args:
        model:    GPT2LMHeadModel with LoRA wrappers applied.
        universe: Universe key.
        path:     Directory containing the adapter checkpoint.

    Returns:
        Model with adapter weights loaded.
    """
    adapter_weights = torch.load(path / f"{universe}_gpt2_lora.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(adapter_weights, strict=False)
    return model


# =============================================================================
# 4. FINE-TUNING
# =============================================================================

def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def finetune_lora_gpt2(
    model: GPT2LMHeadModel,
    universe: str,
    tokenizer: GPT2Tokenizer,
    context_len: int = CONTEXT_LEN,
    batch_size: int = BATCH_SIZE,
    n_epochs: int = 3,
    lr: float = 1e-4,
    device: torch.device = None,
    checkpoint_dir: pathlib.Path = CHECKPOINTS_DIR,
) -> GPT2LMHeadModel:
    """Fine-tune LoRA adapters on a universe corpus using GPT-2.

    Args:
        model:          GPT2LMHeadModel with LoRA adapters applied.
        universe:       Universe key.
        tokenizer:      GPT2Tokenizer.
        context_len:    Sequence length.
        batch_size:     Samples per step.
        n_epochs:       Fine-tuning epochs.
        lr:             AdamW learning rate.
        device:         Training device.
        checkpoint_dir: Where to save adapter checkpoints.

    Returns:
        Fine-tuned model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = load_corpus_texts(universe)
    dataset = GPT2FinetuneDataset(texts, tokenizer, context_len)
    # num_workers=4 for parallel data loading; pin_memory=True speeds up CPU→GPU transfer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Only optimize parameters with requires_grad=True — the LoRA A/B matrices.
    # The 117M frozen GPT-2 base weights are excluded automatically.
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    total_steps = n_epochs * len(dataloader)
    scheduler = build_lr_schedule(optimizer, warmup_steps=100, total_steps=total_steps)

    model = model.to(device)
    model.train()

    log_interval = 100
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            # GPT2LMHeadModel computes cross-entropy loss internally when labels are passed
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss.backward()
            # Gradient clipping prevents exploding gradients from destabilizing LoRA training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % log_interval == 0:
                avg = total_loss / (step + 1)
                print(f"  [{universe}] Epoch {epoch} Step {step+1}/{len(dataloader)} — loss: {avg:.4f}")

        epoch_loss = total_loss / len(dataloader)
        print(f"  [{universe}] Epoch {epoch} complete — loss: {epoch_loss:.4f}")
        # Save adapter after each epoch so partial progress is never lost
        save_lora_adapter(model, universe, checkpoint_dir)

    return model


# =============================================================================
# 5. RAG (shared with loreforge.py — identical implementations)
# =============================================================================

def embed_passages(
    passages: list[str],
    embed_model_name: str = RAG_EMBED_MODEL,
    batch_size: int = 64,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> np.ndarray:
    model = SentenceTransformer(embed_model_name, device=device)
    result = model.encode(passages, batch_size=batch_size, convert_to_numpy=True)
    return result.astype("float32")


def chunk_documents_for_rag(
    documents: list[str],
    tokenizer: GPT2Tokenizer,
    chunk_size: int = RAG_CHUNK_TOKENS,
    overlap: int = RAG_CHUNK_OVERLAP,
) -> list[str]:
    chunks = []
    for doc in documents:
        ids = tokenizer.encode(doc)
        step = chunk_size - overlap
        for start in range(0, len(ids), step):
            chunk_ids = ids[start : start + chunk_size]
            chunks.append(tokenizer.decode(chunk_ids))
    return chunks


def build_faiss_index(
    universe: str,
    passages: list[str],
    embeddings: np.ndarray,
    index_dir: pathlib.Path = INDICES_DIR,
) -> pathlib.Path:
    idx = faiss.IndexFlatL2(embeddings.shape[1])
    idx.add(embeddings)
    index_path = index_dir / f"{universe}_gpt2.faiss"
    faiss.write_index(idx, str(index_path))
    with open(index_dir / f"{universe}_gpt2_passages.json", "w") as f:
        json.dump(passages, f)
    return index_path


def load_faiss_index(
    universe: str,
    index_dir: pathlib.Path = INDICES_DIR,
) -> tuple:
    gpt2_path = index_dir / f"{universe}_gpt2.faiss"
    fallback_path = index_dir / f"{universe}.faiss"
    if gpt2_path.exists():
        index_path = gpt2_path
        passages_path = index_dir / f"{universe}_gpt2_passages.json"
    else:
        index_path = fallback_path
        passages_path = index_dir / f"{universe}_passages.json"
    idx = faiss.read_index(str(index_path))
    with open(passages_path) as f:
        passages = json.load(f)
    return idx, passages


def retrieve_context(
    query: str,
    universe: str,
    faiss_index,
    passages: list[str],
    embed_model_name: str = RAG_EMBED_MODEL,
    k: int = RAG_TOP_K,
) -> list[str]:
    embed_query = embed_passages([query], embed_model_name=embed_model_name)
    _, indices = faiss_index.search(embed_query, k)
    return [passages[i] for i in indices[0]]


# =============================================================================
# 6. INFERENCE
# =============================================================================

def build_generation_prompt(
    user_prompt: str,
    retrieved_passages: list[str],
    universe: str,
) -> str:
    """Assemble the full generation prompt with universe token and lore context."""
    control_token = UNIVERSES[universe]["control_token"]
    # Only include the lore context block when RAG is enabled; skip it for RAG-off mode
    if retrieved_passages:
        joined_passages = "\n\n".join(retrieved_passages)
        context_block = f"Lore reference:\n{joined_passages}\n\n"
    else:
        context_block = ""
    # The narrative instruction explicitly tells GPT-2 NOT to write an article.
    # This is necessary because the fine-tuning corpus was largely wiki/fandom text,
    # which biases the model toward encyclopedic output without this prompt framing.
    return (
        f"{control_token}\n"
        f"{context_block}"
        f"Write a creative short story set in the {universe.replace('_', ' ')} universe "
        f"based on the following prompt. Write as a narrative with characters, dialogue, "
        f"and vivid description — not as an article or encyclopedia entry.\n\n"
        f"Prompt: {user_prompt}\n\n"
        f"Story:\n"
    )


@torch.no_grad()
def generate_story(
    prompt: str,
    universe: str,
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    faiss_index,
    passages: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_k: int = 50,
    device: torch.device = None,
    use_rag: bool = True,
) -> dict:
    """Run the full RAG + GPT-2 generation pipeline.

    Args:
        prompt:         User story prompt.
        universe:       Selected universe key.
        model:          GPT2LMHeadModel with LoRA adapter loaded.
        tokenizer:      GPT2Tokenizer.
        faiss_index:    FAISS index for the universe.
        passages:       Parallel passage list.
        max_new_tokens: Tokens to generate.
        temperature:    Sampling temperature.
        top_k:          Top-k sampling cutoff.
        device:         Inference device.

    Returns:
        Dict with generated_text, retrieved_passages, full_prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    retrieved = retrieve_context(prompt, universe, faiss_index, passages) if use_rag else []
    full_prompt = build_generation_prompt(prompt, retrieved, universe) if use_rag else prompt

    max_prompt_tokens = 1024 - max_new_tokens
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_prompt_tokens).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,      # scales logit distribution; higher = more random
        top_k=top_k,                  # sample only from the top-k most probable tokens
        do_sample=True,               # stochastic sampling (not greedy argmax)
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,       # penalizes tokens that have already appeared; breaks repetition loops
        no_repeat_ngram_size=3,       # prevents any 3-gram from appearing twice; fixes "anan" collapse
    )

    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "retrieved_passages": retrieved,
        "full_prompt": full_prompt,
    }


# =============================================================================
# 7. FULL PIPELINE
# =============================================================================

def run_gpt2_pipeline(
    universes: list[str] = ["star_wars", "harry_potter", "lotr"],
    model_name: str = GPT2_MODEL_NAME,
    finetune_epochs: int = 3,
    finetune_lr: float = 1e-4,
    context_len: int = CONTEXT_LEN,
    batch_size: int = BATCH_SIZE,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Run the full GPT-2 fine-tuning and RAG pipeline.

    Skips any step where output files already exist on disk.

    Args:
        universes:       List of universe keys to fine-tune.
        model_name:      GPT-2 variant to use.
        finetune_epochs: Epochs per universe.
        finetune_lr:     LoRA fine-tuning learning rate.
        context_len:     Sequence length.
        batch_size:      Per-GPU batch size.

    Returns:
        (model, tokenizer) — last fine-tuned model and GPT-2 tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = load_gpt2_tokenizer(model_name)
    print(f"GPT-2 tokenizer loaded (vocab size: {tokenizer.vocab_size})")

    # Fine-tune LoRA adapters
    for i, universe in enumerate(universes):
        adapter_path = CHECKPOINTS_DIR / f"{universe}_gpt2_lora.pt"
        if adapter_path.exists():
            print(f"[LoRA {i+1}/{len(universes)}] {universe} adapter already exists, skipping...")
            continue

        print(f"[LoRA {i+1}/{len(universes)}] Fine-tuning {universe}...")
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model = apply_lora_adapters_gpt2(model)
        model = finetune_lora_gpt2(
            model, universe, tokenizer,
            context_len=context_len,
            batch_size=batch_size,
            n_epochs=finetune_epochs,
            lr=finetune_lr,
            device=device,
            checkpoint_dir=CHECKPOINTS_DIR,
        )
        print(f"      {universe} adapter saved.")

    # Build FAISS indices
    for i, universe in enumerate(universes):
        index_path = INDICES_DIR / f"{universe}_gpt2.faiss"
        if index_path.exists():
            print(f"[RAG {i+1}/{len(universes)}] {universe} index already exists, skipping...")
            continue

        print(f"[RAG {i+1}/{len(universes)}] Building FAISS index for {universe}...")
        texts = load_corpus_texts(universe)
        passages = chunk_documents_for_rag(texts, tokenizer)
        embeddings = embed_passages(passages)
        build_faiss_index(universe, passages, embeddings)
        print(f"      {universe} FAISS index saved ({len(passages)} passages).")

    # Return last model with last universe adapter loaded for immediate testing
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = apply_lora_adapters_gpt2(model)
    model = load_lora_adapter(model, universes[-1])
    model = model.to(device)
    model.eval()

    print("GPT-2 pipeline complete.")
    return model, tokenizer
