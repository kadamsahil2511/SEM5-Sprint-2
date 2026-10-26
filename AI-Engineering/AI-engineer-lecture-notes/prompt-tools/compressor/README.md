# Prompt Compressor (LLMLingua-2)

Local HTTP service that compresses any prompt using Microsoft's
**LLMLingua-2** model (xlm-roberta-large fine-tuned for prompt compression).

No regex. Real NLP. Works on prose, image prompts, code, instructions &mdash;
anything &mdash; because the model learned token importance from human labels.

## Install

```bash
source ../../../../.venv/bin/activate  # Install once from the sprint root first.
```

Runs on CPU (slow-ish), CUDA, or Apple Silicon (MPS) &mdash; auto-detected.

## Run

```bash
python app.py
```

Listens on `http://127.0.0.1:7870`.

## API

### `POST /api/compress`

```json
{
  "prompt": "your long prompt...",
  "rate": 0.5,
  "force_tokens": ["\n", ".", ","]
}
```

- `rate` &mdash; fraction of tokens to **keep** (0.5 = keep ~half, drop ~half).
- `force_tokens` &mdash; tokens that must never be dropped (defaults to common punctuation).

Response:

```json
{
  "compressed": "...",
  "origin_tokens": 993,
  "compressed_tokens": 420,
  "ratio": "2.4x",
  "rate": 0.42
}
```

### `GET /api/health`

Returns device + model info.

## Chrome extension integration

The extension in `../extension/` calls this service when **Deep mode** is on
and the server is reachable. If the server is off, it falls back to the
built-in rules.

## Model override

```bash
LLMLINGUA_MODEL=microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank python app.py
```

The `bert-base-multilingual` variant is smaller/faster at the cost of
compression quality.
