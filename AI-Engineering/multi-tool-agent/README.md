# Multi-tool agent TUI

A full-screen terminal chat built with Textual and Rich, using a LangChain agent
with two tools:

- `web_search` finds current web pages through keyless DuckDuckGo search and returns source links.
- `get_current_weather` gets current conditions by city from [Open-Meteo](https://open-meteo.com/).

The local Ollama model chooses a tool and turns its result into an answer. Rich
renders Markdown responses in the scrollable chat, including visible source
URLs. Chats are saved locally between runs. No search or weather API key is needed.

From this directory:

```sh
source ../../.venv/bin/activate
python agent.py
```

Enter a question in the input box. The app has these commands:

- `/help` shows the commands.
- `/new` starts a new chat.
- `/sessions` lists saved chats and their IDs.
- `/load ID` opens a saved chat.
- `/quit` or Ctrl-Q exits.

The last selected chat reopens on startup. Sessions are stored in the ignored
`.chat_sessions.json` file in this directory. For a single answer without
opening the TUI, pass the question on the command line:

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
