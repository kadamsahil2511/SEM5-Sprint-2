# Multi-tool agent

A small LangChain CLI with two tools:

- `web_search` finds current web pages through keyless DuckDuckGo search and returns source links.
- `get_current_weather` gets current conditions by city from [Open-Meteo](https://open-meteo.com/).

The local Ollama model chooses a tool and turns its result into an answer. Rich
renders Markdown responses in the terminal, including visible source URLs. The
chat remembers earlier turns until you exit. No search or weather API key is needed.

From this directory:

```sh
source ../../.venv/bin/activate
python agent.py
```

Ask at the `You:` prompt. Enter `quit` or `exit` to leave; Ctrl-C and Ctrl-D
also work. For a single answer, pass the question on the command line:

```sh
python agent.py "What is the weather in Mumbai, IN?"
python agent.py "Search the web for the official LangChain Python documentation."
```

Start Ollama first and have `gemma4:e2b-mlx` installed. To use another local
model, set `OLLAMA_MODEL` in an optional `.env` file. Web search uses the
unofficial [`ddgs` package](https://github.com/deedy5/ddgs), so availability
depends on its public search backend. Open-Meteo's public API is free for
non-commercial use within its [usage limits](https://open-meteo.com/en/terms).
Weather data: Open-Meteo (CC BY 4.0). Location data: GeoNames.
