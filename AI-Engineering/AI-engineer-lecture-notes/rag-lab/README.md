# RAG Lab — three stages, one PDF

A hands-on tour of the retrieval stack. Same source PDF across all three stages, so you can compare behaviour.

- **Stage 1 — FAISS**: embeddings in raw memory. You see every number.
- **Stage 2 — ChromaDB**: metadata filtering and real query writing.
- **Stage 3 — Qdrant via LangChain**: production-shape code once you've earned the abstraction.

## Setup

```
source ../../../.venv/bin/activate  # Install once from the sprint root first.
cp .env.example .env   # then put your real GROQ_API_KEY in .env
```

Start Qdrant locally (needed only for Stage 3):

```
docker run -p 6333:6333 qdrant/qdrant
```

Drop any multi-page PDF at `data/sample.pdf` — a college handbook, a textbook chapter, a research paper. All three stages read the same file.

## Run order

```
# Stage 1 — FAISS
cd stage1_faiss
python 01_build_index.py
python 02_query.py
python 03_chatbot.py

# Stage 2 — ChromaDB
cd ../stage2_chromadb
python 01_build_index.py
python 02_query_basic.py
python 03_query_with_filters.py
streamlit run 04_chatbot.py

# Stage 3 — Qdrant + LangChain
cd ../stage3_qdrant
python 01_build_index.py
streamlit run 02_chatbot_langchain.py
```

## What each stage teaches

| Stage | DB      | What you touch                | What's hidden              |
| ----- | ------- | ----------------------------- | -------------------------- |
| 1     | FAISS   | numpy arrays, model.encode    | nothing (you did it all)   |
| 2     | Chroma  | query DSL, metadata filters   | embedding call             |
| 3     | Qdrant  | retriever.as_retriever()      | almost everything          |

## Final note

If a demo breaks in Stage 3, you now know enough to debug it in Stage 1 terms. That's the point.
