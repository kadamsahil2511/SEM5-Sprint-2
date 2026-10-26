# PydanticAI Firecrawl agent

A small CLI agent built with [PydanticAI](https://pydantic.dev/docs/ai/) and one
`search_web` tool. It calls [Firecrawl Search](https://docs.firecrawl.dev/features/search)
for three web results, then uses a local Ollama model to answer with source URLs.

From this directory:

```sh
source ../../.venv/bin/activate
python agent.py "Find the official Pydantic AI documentation"
```

Put `FIRECRAWL_API_KEY=...` in `.env`; `.env.example` shows the format. Do not
overwrite an existing `.env` when copying the example.

The default local model is `gemma4:e2b-mlx`. Set `OLLAMA_MODEL` in `.env` to use
another installed Ollama model. This project uses the sprint's shared `.venv`;
the key in `.env` is ignored by Git.
