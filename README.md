# LoreForge: Multi-Universe Lore-Faithful Story Generation

## Datasets

### Pretraining

| Dataset | Source | License | Use |
|---|---|---|---|
| `manu/project_gutenberg` | HuggingFace | Public domain | Base model pretraining on general English prose (~70k novels) |

---

### Star Wars

| Dataset | Source | License | Use |
|---|---|---|---|
| `lara-martin/Scifi_TV_Shows` | HuggingFace | CC-BY-4.0 | Fine-tuning + RAG — ~270 Star Wars stories scraped from the Star Wars Fandom wiki, filtered by keyword. Provides lore prose sentences covering characters, events, and locations. |

---

### Harry Potter

| Dataset | Source | License | Use |
|---|---|---|---|
| `rupanshukapoor/harry-potter-books` | Kaggle | MIT (educational/research only) | Fine-tuning + RAG — Full text of all seven HP books as plain .txt files (~2.5 MB). Teaches narrative style and provides retrievable passages covering characters, spells, locations, and events. |

---

### Lord of the Rings

| Dataset | Source | License | Use |
|---|---|---|---|
| `jeremyarancio/lotr-book` | HuggingFace | Unstated (educational/research only) | Fine-tuning — Full LOTR trilogy text (pages 45–1055). Teaches the LoRA adapter Tolkien's prose style: archaic diction, elevated register, and the narrative rhythm of Middle-earth. |
| `wikimedia/wikipedia` (filtered) | HuggingFace | CC BY-SA 3.0 | RAG — English Wikipedia filtered to LOTR-related articles (characters, locations, factions, artifacts). Encyclopedic structure makes these ideal retrieval chunks for grounding generation in canon facts. |

---

## GPT-2 Architecture

GPT-2 is a decoder-only transformer — the same fundamental architecture as LoreForge's scratch-trained model, but pretrained by OpenAI on 40GB of web text. The `gpt2` variant used here has 117M parameters.

### Key components

**Token + Position Embeddings**
Every input token is looked up in a learned embedding table (`model.transformer.wte`, shape `50257 × 768`) and added to a learned positional embedding (`model.transformer.wpe`, shape `1024 × 768`). This gives the model both the identity of each token and its position in the sequence. GPT-2's context window is hard-capped at 1024 tokens by the size of `wpe`.

**Transformer Blocks (`model.transformer.h`)**
GPT-2 has 12 stacked transformer blocks. Each block contains:

- **Causal Self-Attention** — each token can only attend to tokens that came before it (enforced by a causal mask). This is what makes GPT-2 a language model: it predicts the next token given all previous tokens. The attention mechanism has three projections: `c_attn` (query, key, value combined into one `768 → 2304` projection) and `c_proj` (output projection `768 → 768`). These are the layers that LoRA adapters are applied to.
- **Layer Normalization** — applied before attention and before the feed-forward network (pre-norm formulation), stabilizing training.
- **Feed-Forward Network** — two linear layers with a GELU activation: `768 → 3072 → 768`. This is where most of the model's "knowledge" is thought to be stored.
- **Residual Connections** — both the attention and feed-forward outputs are added back to the block's input, allowing gradients to flow cleanly through deep networks.

**Language Model Head**
After all 12 blocks, a final layer norm is applied and the output is projected back to vocabulary size (`model.lm_head`, shape `768 → 50257`). The resulting logits represent the unnormalized probability of each token in the vocabulary being the next token.

**Autoregressive Generation**
At inference time, GPT-2 generates text token by token. Each new token is appended to the input and fed back into the model to produce the next token. LoreForge uses top-k sampling with temperature to control diversity: logits are divided by `temperature`, all but the top-k are masked to `-inf`, and the next token is sampled from the resulting softmax distribution.

### Why LoRA works on GPT-2

LoRA freezes all of GPT-2's 117M base weights and injects small trainable rank decomposition matrices (`lora_A`, `lora_B`) alongside the attention projections. The effective weight update is `ΔW = lora_B @ lora_A * (alpha / rank)`, adding only ~300K trainable parameters per universe instead of fine-tuning all 117M. The base model's English fluency is preserved while the adapter steers the output distribution toward universe-specific vocabulary and style.

---

## RAG Pipeline

Retrieval-Augmented Generation (RAG) grounds each story generation in canon lore by fetching the most relevant passages from a universe-specific vector database before calling the model. All RAG code lives in `loreforge_gpt2.py` lines 315–450.

### Step 1 — Chunking (`chunk_documents_for_rag`, line 329)

```python
ids = tokenizer.encode(doc)
step = chunk_size - overlap
for start in range(0, len(ids), step):
    chunk_ids = ids[start : start + chunk_size]
    chunks.append(tokenizer.decode(chunk_ids))
```

The raw corpus text for each universe is first tokenized into token IDs, then split into fixed-size windows of `chunk_size=256` tokens with `overlap=32` tokens between consecutive chunks. The overlap prevents a relevant sentence from being cut in half at a chunk boundary. Each chunk is decoded back to a string and stored as a passage. This produces thousands of short, dense lore excerpts per universe.

### Step 2 — Embedding (`embed_passages`, line 318)

```python
model = SentenceTransformer(embed_model_name, device=device)
result = model.encode(passages, batch_size=64, convert_to_numpy=True)
return result.astype("float32")
```

Every passage is converted to a dense 384-dimensional vector using `all-MiniLM-L6-v2` from `sentence-transformers`. This model maps semantically similar text to nearby points in vector space — so "Darth Vader's breathing" and "Vader's respirator" would have similar embeddings even though they share no words. Embeddings are computed in batches of 64 for efficiency and returned as a float32 numpy array.

### Step 3 — Indexing (`build_faiss_index`, line 345)

```python
idx = faiss.IndexFlatL2(embeddings.shape[1])
idx.add(embeddings)
faiss.write_index(idx, str(index_path))
with open(index_dir / f"{universe}_gpt2_passages.json", "w") as f:
    json.dump(passages, f)
```

All passage embeddings are loaded into a FAISS `IndexFlatL2` — a flat index that performs exact nearest-neighbour search using L2 (Euclidean) distance. The index and the original passage strings are saved to disk as `{universe}_gpt2.faiss` and `{universe}_gpt2_passages.json`. The passages JSON is kept in parallel so retrieved embedding indices can be mapped back to readable text.

### Step 4 — Retrieval (`retrieve_context`, line 370)

```python
embed_query = embed_passages([query], embed_model_name=embed_model_name)
_, indices = faiss_index.search(embed_query, k)
return [passages[i] for i in indices[0]]
```

At inference time, the user's prompt is embedded using the same `all-MiniLM-L6-v2` model. `faiss_index.search(embed_query, k)` returns the indices of the `k=3` passages whose embeddings are closest to the query embedding in L2 space. Those passages are retrieved from the parallel passages list and returned as strings.

### Step 5 — Prompt Assembly (`build_generation_prompt`, line 387)

```python
control_token = UNIVERSES[universe]["control_token"]
joined_passages = "\n\n".join(retrieved_passages)
return f"{control_token}\n--- Lore Context ---\n{joined_passages}\n--- Story ---\n{user_prompt}"
```

The retrieved passages are joined and wrapped in a structured prompt. A universe control token (e.g. `[STAR_WARS]`) is prepended to signal to the model which universe adapter is active. The final prompt fed to GPT-2 looks like:

```
[STAR_WARS]
--- Lore Context ---
<passage 1>

<passage 2>

<passage 3>
--- Story ---
<user prompt>
```

GPT-2 then continues this text autoregressively, conditioned on both the lore context and the user's prompt. The retrieved passages are also returned to the GUI and displayed alongside the generated story so the user can see which canon lore was used.

---

## Training Notes

### GPT-2 Fallback

The project pivoted from a scratch-trained transformer to a pretrained GPT-2 base model (`gpt2`, 117M params) due to Quest HPC queue delays. LoRA adapters are fine-tuned per universe on top of the frozen GPT-2 weights. The RAG pipeline, FAISS indices, and GUI are unchanged.

### LoRA Fine-tuning: Epochs Matter

**Observation:** After only 1 epoch of fine-tuning, model output was incoherent — Star Wars vocabulary was present but grammar was broken.

**Root cause:** 1 epoch is insufficient. The LoRA adapter (lora_B initialized to zeros) began shifting the model's output distribution toward the training data but did not converge. The model was stuck between fluent base GPT-2 English and the training corpus distribution, producing neither coherently.

**Confirmed:** Base GPT-2 with no adapter loaded produces fluent English. The degradation is purely a function of insufficient training epochs, not a bug in the LoRA implementation.

**Fix:** Run a minimum of 3 epochs. The parallel SLURM scripts (`submit_star_wars.sh`, `submit_harry_potter.sh`, `submit_lotr.sh`) each request 1 A100 GPU and run independently so all three universes fine-tune simultaneously.

### Conv1D Compatibility

GPT-2 uses `Conv1D` layers internally (weight shape `(in, out)`) rather than `nn.Linear` (weight shape `(out, in)`). The `LoRALinear` wrapper detects which type it is wrapping and sets `in_features`/`out_features` accordingly. The forward pass `x @ lora_A.T @ lora_B.T` is correct for both layer types.

### Adapter Checkpoint Loading

Adapters saved on Quest CUDA GPUs must be loaded with `map_location="cpu"` on a CPU-only inference machine. The `load_lora_adapter` function in `loreforge_gpt2.py` handles this automatically.

### Context Window Truncation

GPT-2 has a hard 1024-token context window. The RAG prompt (lore passages + user prompt) can exceed this. The `generate_story` function truncates the input to `1024 - max_new_tokens` before calling `model.generate` to prevent an out-of-range position embedding error.

---

## Running the GUI

### 1. Start the inference server

```bash
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 server.py
```

### 2. Start the React frontend

```bash
cd gui
npm run dev
```

Open `http://localhost:3000`. The server auto-detects which universes have trained adapters and marks them as available in the UI.
