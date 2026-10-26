"""LLMLingua-2 prompt compressor as a local HTTP service.

Uses Microsoft's LLMLingua-2 model to score token importance and drop the
low-information ones. Works on any prompt type (prose, image, code, etc.) —
unlike regex rules, it's trained on human-labeled compression data.

Model: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (CPU-friendly).
"""

from __future__ import annotations

import os
import platform
from threading import Lock
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_ID = os.environ.get(
    "LLMLINGUA_MODEL", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
)

_compressor = None
_load_lock = Lock()


def _pick_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load():
    global _compressor
    if _compressor is not None:
        return _compressor
    with _load_lock:
        if _compressor is not None:
            return _compressor
        from llmlingua import PromptCompressor

        _compressor = PromptCompressor(
            model_name=MODEL_ID,
            use_llmlingua2=True,
            device_map=_pick_device(),
        )
    return _compressor


class CompressRequest(BaseModel):
    prompt: str
    rate: float = 0.5           # target compression rate (fraction of tokens kept)
    force_tokens: Optional[list[str]] = None  # tokens that must survive


app = FastAPI(title="Prompt Compressor (LLMLingua-2)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only; relax for the Chrome extension
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"ok": True, "device": _pick_device(), "model": MODEL_ID}


@app.post("/api/compress")
def compress(req: CompressRequest):
    text = (req.prompt or "").strip()
    if not text:
        return JSONResponse({"error": "empty prompt"}, status_code=400)

    try:
        comp = _load()
        result = comp.compress_prompt(
            text,
            rate=max(0.1, min(0.95, req.rate)),
            force_tokens=req.force_tokens or ["\n", ".", "!", "?", ","],
        )
        return {
            "compressed": result["compressed_prompt"],
            "origin_tokens": result.get("origin_tokens"),
            "compressed_tokens": result.get("compressed_tokens"),
            "ratio": result.get("ratio"),
            "rate": result.get("rate"),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7870)
