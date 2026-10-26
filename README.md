# SEM5 Sprint 2

## AI Engineering

| Project | Purpose |
| --- | --- |
| [AI engineer lecture notes](AI-Engineering/AI-engineer-lecture-notes/) | Lecture examples, RAG labs, and prompt tools |
| [LLMs and RAG](AI-Engineering/LLMs-Rag-DG/) | Gemini chat and NLP chatbot experiments |
| [Image Studio](AI-Engineering/image-studio/) | Local image generation API |
| [PDF RAG](AI-Engineering/pdf-rag/) | PDF retrieval and chat with FAISS or Chroma |
| [Sentiment Analysis](AI-Engineering/sentiment-analysis/) | Sentiment analysis web app |
| [Token Maxxing](AI-Engineering/token-maxxing/) | Browser extension and prompt compressor |
| [Multi-Tool Agent](AI-Engineering/multi-tool-agent/) | LangChain agent with keyless web search and current weather tools |

All Python projects use the single `.venv` in this sprint folder:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock
source .venv/bin/activate
```

Run each Python command from its project directory after activation. The
JavaScript projects use their own `package.json` and `npm ci`; Node packages
are separate from the Python environment. Local `.env` files, model caches,
generated indexes, and virtual environments are excluded from Git.
`requirements.txt` lists direct dependencies; `requirements.lock` records
the tested versions, including the spaCy English model.

The prior standalone repositories remain available at
[AI engineer lecture notes](https://github.com/Horizon733/AI-engineer-lecture-notes),
[NLP chatbot](https://github.com/Horizon733/GenAI-NLP-chatbot),
[Image Studio](https://github.com/kadamsahil2511/image-studio), and
[Token Maxxing](https://github.com/kadamsahil2511/token-maxxing).
