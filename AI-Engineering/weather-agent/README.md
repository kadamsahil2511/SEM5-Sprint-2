# Weather agent

A small LangChain agent with one tool for current weather by city. The tool
resolves the city and retrieves current conditions from [Open-Meteo](https://open-meteo.com/).
The local Ollama model turns those conditions into a short answer. No weather
API key, account, or subscription is needed for non-commercial use.

From this directory:

```sh
source ../../.venv/bin/activate
python agent.py "What is the weather in Mumbai, IN?"
```

Start Ollama first and have `gemma4:e2b-mlx` installed. To use another local
model, set `OLLAMA_MODEL` in an optional `.env` file. Open-Meteo's public API is
free for non-commercial use within its [usage limits](https://open-meteo.com/en/terms).
Weather data: Open-Meteo (CC BY 4.0). Location data: GeoNames.
