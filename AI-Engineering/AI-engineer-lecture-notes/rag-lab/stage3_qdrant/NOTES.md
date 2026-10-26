# Stage 3 — Qdrant via LangChain

## Why Qdrant?
Production-grade: Rust-based, fast, strong hybrid search support, cloud-hosted option, scales horizontally. If a team is building a serious retrieval service in 2026, it is probably Qdrant, Weaviate, pgvector, or a hyperscaler index — Qdrant is a representative pick.

## Why LangChain here instead of raw `qdrant-client`?
To show you the framework path you will see in every tutorial, every job, every GitHub repo. Framework code is the shape of production.

## What you learned
By Stage 3 you can read framework code because you built the pieces by hand in Stages 1 and 2. When `RetrievalQA.from_chain_type` surprises you, you can picture the embedding call, the vector lookup, the prompt assembly, the model call — because you wrote each one.

## Where to go next
- Reranking (Day 5) — fetch 20, carefully pick 5
- Hybrid search in Qdrant (dense + sparse / BM25) for keyword-heavy queries
- Wrap this retriever as a **tool** on an agent (Day 5)
