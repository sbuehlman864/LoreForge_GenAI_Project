# LoreForge — GPT-2 Fallback Plan

## Overview

If Quest compute resources are unavailable before the deadline, replace the scratch-trained `LoreForgeTransformer` with a pretrained GPT-2 model from HuggingFace. The LoRA fine-tuning, RAG pipeline, FAISS indices, and GUI remain unchanged — only the base model and tokenizer swap out.

---

## Why This Works

GPT-2 is a decoder-only transformer (the same architecture as `LoreForgeTransformer`) trained on 40GB of web text. It already produces fluent English prose, so we skip pretraining entirely and go straight to LoRA fine-tuning on universe corpora. The course requirement is met as long as we understand the GPT-2 architecture — which we do, since we built the same architecture from scratch.

---

## What Changes

| Component | Current | GPT-2 Fallback |
|---|---|---|
| Base model | `LoreForgeTransformer` (scratch) | `GPT2LMHeadModel` (HuggingFace) |
| Tokenizer | Custom BPE (16k vocab) | GPT-2 BPE tokenizer (50,257 vocab) |
| Pretraining | Required | Skipped |
| LoRA fine-tuning | Same | Same (applied to GPT-2 attention layers) |
| RAG pipeline | Same | Same |
| FAISS indices | Same | Same |
| GUI | Same | Same |

---

## Implementation Steps

### 1. Install HuggingFace Transformers

```bash
pip install transformers
```

### 2. Load GPT-2 and its Tokenizer

Replace the custom tokenizer and model init with:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")  # 117M params
# or "gpt2-medium" (345M), "gpt2-large" (774M), "gpt2-xl" (1.5B)
```

### 3. Apply LoRA Adapters to GPT-2

GPT-2's attention layers are in `model.transformer.h[i].attn`. The existing `LoRALinear` class works as-is — just change the target layers:

```python
def apply_lora_adapters_gpt2(model, rank=8, alpha=16.0):
    for param in model.parameters():
        param.requires_grad = False
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, rank, alpha)
        block.attn.c_proj = LoRALinear(block.attn.c_proj, rank, alpha)
    return model
```

> **Note:** GPT-2 uses `Conv1D` layers (not `nn.Linear`) internally. `Conv1D` weights are transposed relative to `nn.Linear` — `LoRALinear` may need a small adjustment to handle this. See Step 3a below.

#### Step 3a — Conv1D Compatibility Fix

GPT-2's `Conv1D` has shape `(in_features, out_features)` instead of `(out_features, in_features)`. Update `LoRALinear` to detect and handle this:

```python
class LoRALinear(nn.Module):
    def __init__(self, layer, rank=8, alpha=16.0):
        super().__init__()
        self.layer = layer
        self.scale = alpha / rank

        # Handle both nn.Linear and GPT-2's Conv1D
        if hasattr(layer, 'weight'):
            w = layer.weight
            if w.shape[0] < w.shape[1]:  # Conv1D: (in, out)
                in_features, out_features = w.shape
            else:                          # nn.Linear: (out, in)
                out_features, in_features = w.shape

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        return self.layer(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scale
```

### 4. Update the Fine-tuning Loop

The existing `finetune_lora` function works with minor changes — GPT-2 returns a `CausalLMOutputWithCrossAttentions` object, so loss extraction changes slightly:

```python
# Instead of:
_, loss = model(x, y)

# Use:
outputs = model(input_ids=x, labels=y)
loss = outputs.loss
```

### 5. Update generate_story

Replace the manual autoregressive loop with HuggingFace's built-in generation:

```python
@torch.no_grad()
def generate_story_gpt2(prompt, universe, model, tokenizer, faiss_index, passages,
                         max_new_tokens=256, temperature=0.9, top_k=50, device=None):
    if device is None:
        device = next(model.parameters()).device

    retrieved = retrieve_context(prompt, universe, faiss_index, passages)
    full_prompt = build_generation_prompt(prompt, retrieved, universe)

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)

    return {
        "generated_text": generated_text,
        "retrieved_passages": retrieved,
        "full_prompt": full_prompt,
    }
```

### 6. Save and Load LoRA Adapters

`save_lora_adapter` and `load_lora_adapter` work unchanged — they filter by `"lora_"` keys in the state dict, which is model-agnostic.

### 7. RAG Pipeline

No changes needed. `embed_passages`, `build_faiss_index`, `load_faiss_index`, and `retrieve_context` are completely independent of the base model.

---

## Fine-tuning Script

Create `finetune_gpt2.py` to run locally or on Quest:

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from loreforge import (
    apply_lora_adapters_gpt2, finetune_lora_gpt2,
    save_lora_adapter, CHECKPOINTS_DIR, PROCESSED_DIR
)

UNIVERSES      = ["star_wars", "harry_potter", "lotr"]
FINETUNE_EPOCHS = 3
FINETUNE_LR    = 1e-4
CONTEXT_LEN    = 512
BATCH_SIZE     = 8   # GPT-2 is larger — smaller batch than scratch model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

for universe in UNIVERSES:
    print(f"Fine-tuning {universe}...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = apply_lora_adapters_gpt2(model)
    model = finetune_lora_gpt2(
        model, universe,
        bin_path=PROCESSED_DIR / f"{universe}_finetune.bin",
        context_len=CONTEXT_LEN,
        batch_size=BATCH_SIZE,
        n_epochs=FINETUNE_EPOCHS,
        lr=FINETUNE_LR,
        device=device,
    )
    save_lora_adapter(model, universe, CHECKPOINTS_DIR)
    print(f"  {universe} adapter saved.")
```

---

## Timeline if Pivoting

| Task | Notes |
|---|---|
| Install transformers, load GPT-2 | ~30 minutes |
| Update LoRALinear for Conv1D | ~1 hour |
| Update fine-tuning loop | ~1 hour |
| Run fine-tuning (3 universes × 3 epochs) | A few hours on GPU, longer on CPU |
| Update generate_story | ~30 minutes |
| Build and test GUI | Already planned |

---

## Decision Point

Cancel the Quest SLURM job and pivot to GPT-2 if:
- The SLURM job has not started by midday tomorrow
- OR the job starts but hits an unrecoverable error

The fine-tuning binaries (`star_wars_finetune.bin`, `harry_potter_finetune.bin`, `lotr_finetune.bin`) and FAISS indices built on Quest are reusable — no data prep needed if pivoting.
