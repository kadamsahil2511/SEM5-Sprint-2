# Weather agent

A small LangChain agent with one tool for current weather by city. The tool
resolves the city with OpenWeather Geocoding, then calls the One Call API 4.0
current-weather endpoint. It returns Celsius temperatures.

From this directory:

```sh
cp .env.example .env
# Add OPENWEATHER_API_KEY to .env.
source ../../.venv/bin/activate
python agent.py "What is the weather in Mumbai, IN?"
```

Start Ollama first and have `gemma4:e2b-mlx` installed, or set `OLLAMA_MODEL`
in `.env` to another local model. OpenWeather requires a separate
[One Call API 4.0 subscription](https://openweathermap.org/api/one-call-4?collection=one_call_api)
for this endpoint. The key stays in `.env`, which Git ignores.

If geocoding returns HTTP 401, check the key in your OpenWeather account and
confirm the account email. New keys can take up to two hours to activate.
