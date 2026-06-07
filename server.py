"""
LoreForge inference server.
Runs locally (or on Quest via port forwarding) to serve the React GUI.

Start:
    pip install fastapi uvicorn
    python server.py

Endpoints:
    GET  /universes            → list of available universes
    POST /generate             → run inference and return story + context
"""

import json
import pathlib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR        = pathlib.Path(__file__).parent
CHECKPOINTS_DIR = ROOT_DIR / "data" / "checkpoints"
INDICES_DIR     = ROOT_DIR / "data" / "indices"
BEST_CONFIG     = ROOT_DIR / "best_config.json"

UNIVERSES = ["star_wars", "harry_potter", "lotr"]

UNIVERSE_LABELS = {
    "star_wars":    "Star Wars",
    "harry_potter": "Harry Potter",
    "lotr":         "Lord of the Rings",
}

# ── Detect which model backend is available ────────────────────────────────────

def _gpt2_adapter_exists(universe: str) -> bool:
    return (CHECKPOINTS_DIR / f"{universe}_gpt2_lora.pt").exists()

def _scratch_adapter_exists(universe: str) -> bool:
    return (CHECKPOINTS_DIR / f"{universe}_lora.pt").exists()

def _gpt2_index_exists(universe: str) -> bool:
    return (INDICES_DIR / f"{universe}_gpt2.faiss").exists() or (INDICES_DIR / f"{universe}.faiss").exists()

def _scratch_index_exists(universe: str) -> bool:
    return (INDICES_DIR / f"{universe}.faiss").exists()

def _universe_available(universe: str) -> bool:
    gpt2_ready    = _gpt2_adapter_exists(universe) and _gpt2_index_exists(universe)
    scratch_ready = _scratch_adapter_exists(universe) and _scratch_index_exists(universe)
    return gpt2_ready or scratch_ready

def _universe_backend(universe: str) -> str:
    """Return 'gpt2' or 'scratch', preferring whichever is available."""
    if _gpt2_adapter_exists(universe) and _gpt2_index_exists(universe):
        return "gpt2"
    if _scratch_adapter_exists(universe) and _scratch_index_exists(universe):
        return "scratch"
    return "none"

# ── Model cache ───────────────────────────────────────────────────────────────
# Models are loaded lazily on first use to avoid loading all three at startup.
# Each entry: { "model": ..., "tokenizer": ..., "faiss_index": ..., "passages": ... }

_model_cache: dict = {}

def _load_universe(universe: str, device: torch.device):
    if universe in _model_cache:
        return _model_cache[universe]

    backend = _universe_backend(universe)
    if backend == "none":
        raise RuntimeError(f"No trained adapter found for universe '{universe}'")

    if backend == "gpt2":
        from loreforge_gpt2 import (
            load_gpt2_tokenizer, apply_lora_adapters_gpt2,
            load_lora_adapter, load_faiss_index,
        )
        from transformers import GPT2LMHeadModel

        tokenizer = load_gpt2_tokenizer()
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model = apply_lora_adapters_gpt2(model)
        model = load_lora_adapter(model, universe, CHECKPOINTS_DIR)
        model = model.to(device)
        model.eval()
        faiss_index, passages = load_faiss_index(universe)

    else:  # scratch
        from loreforge import (
            LoreForgeTransformer, load_tokenizer, apply_lora_adapters,
            load_lora_adapter, load_faiss_index, TOKENIZER_PATH,
        )
        with open(BEST_CONFIG) as f:
            cfg = json.load(f)

        tokenizer = load_tokenizer(TOKENIZER_PATH)
        pretrain_epochs = cfg["pretrain_epochs"]
        model = LoreForgeTransformer(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            context_len=cfg["context_len"],
            dropout=0.0,
        )
        ckpt_path = CHECKPOINTS_DIR / f"pretrain_epoch{pretrain_epochs}.pt"
        model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
        model = apply_lora_adapters(model)
        model = load_lora_adapter(model, universe, CHECKPOINTS_DIR)
        model = model.to(device)
        model.eval()
        faiss_index, passages = load_faiss_index(universe)

    entry = {
        "model": model,
        "tokenizer": tokenizer,
        "faiss_index": faiss_index,
        "passages": passages,
        "backend": backend,
    }
    _model_cache[universe] = entry
    return entry

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="LoreForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenerateRequest(BaseModel):
    universe: str
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_k: int = 50
    use_rag: bool = True


@app.get("/universes")
def list_universes():
    return [
        {
            "id": u,
            "label": UNIVERSE_LABELS[u],
            "available": _universe_available(u),
            "backend": _universe_backend(u),
        }
        for u in UNIVERSES
    ]


@app.post("/generate")
def generate(req: GenerateRequest):
    if req.universe not in UNIVERSES:
        raise HTTPException(status_code=400, detail=f"Unknown universe '{req.universe}'")

    if not _universe_available(req.universe):
        raise HTTPException(
            status_code=503,
            detail=f"Universe '{req.universe}' has no trained adapter yet. Run the training pipeline first.",
        )

    entry = _load_universe(req.universe, device)

    if entry["backend"] == "gpt2":
        from loreforge_gpt2 import generate_story
    else:
        from loreforge import generate_story

    result = generate_story(
        prompt=req.prompt,
        universe=req.universe,
        model=entry["model"],
        tokenizer=entry["tokenizer"],
        faiss_index=entry["faiss_index"],
        passages=entry["passages"],
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        device=device,
        use_rag=req.use_rag,
    )

    return {
        "universe": req.universe,
        "prompt": req.prompt,
        "generated_text": result["generated_text"],
        "retrieved_passages": result["retrieved_passages"],
        "backend": entry["backend"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
