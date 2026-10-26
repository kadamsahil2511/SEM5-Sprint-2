from __future__ import annotations

import json
import os
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from llmlingua import PromptCompressor
from pydantic import BaseModel, Field


app = FastAPI(title="Prompt Compressor")
COMPRESSOR_MODEL = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"


class CompressRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)


def generate(prompt: str, model: str) -> str:
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    payload = json.dumps(body).encode("utf-8")
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urlopen(request, timeout=120) as response:
            return json.load(response)["response"].strip()
    except HTTPError as error:
        raise HTTPException(status_code=502, detail=f"Ollama returned HTTP {error.code}") from error
    except (URLError, TimeoutError) as error:
        raise HTTPException(status_code=503, detail="Ollama is unavailable") from error
    except (KeyError, ValueError, TypeError, UnicodeError) as error:
        raise HTTPException(status_code=502, detail="Invalid response from Ollama") from error


@lru_cache(maxsize=1)
def compressor_model() -> PromptCompressor:
    return PromptCompressor(
        model_name=COMPRESSOR_MODEL,
        use_llmlingua2=True,
        device_map="cpu",
        model_config={
            "trust_remote_code": False,
        },
    )


def compress_prompt(prompt: str) -> str:
    result = compressor_model().compress_prompt_llmlingua2(
        prompt, rate=0.6, force_tokens=["\n", ".", "?"], force_reserve_digit=True
    )
    compressed = result["compressed_prompt"].strip()
    return compressed if compressed and len(compressed) < len(prompt) else prompt


@app.post("/compress")
def compress(body: CompressRequest) -> dict[str, str]:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be blank")

    return {"compressed_prompt": compress_prompt(prompt)}


@app.post("/chat")
def chat(body: CompressRequest) -> dict[str, str]:
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be blank")

    compressed = compress_prompt(prompt)
    answer = generate(compressed, os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx"))
    return {"compressed_prompt": compressed, "answer": answer}
