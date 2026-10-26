# Stage 2 — ChromaDB

## Why ChromaDB?
Persistence, metadata, filtering, a real query language. Still local, still a single process — but now there's a database between you and the arrays.

## What you learned
- Real search is **similarity + filters**. The retrieval you'll ship uses a `where` clause almost every time.
- Writing the query is the skill. `$and`, `$or`, `$gte`, `$in`, `where_document.$contains` — these are the knobs.
- Metadata is cheap to add and worth adding. Page numbers, section guesses, word counts become query handles.

## What ChromaDB struggles with at scale
- Horizontal scaling and high write throughput
- Hybrid (dense + sparse) search out of the box
- Multi-tenant isolation and strict auth

That's why Stage 3 moves to **Qdrant** — Rust-based, cluster-ready — and uses **LangChain** on top so you can see what a real production codebase looks like.
