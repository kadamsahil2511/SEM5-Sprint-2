# Stage 1 — FAISS

## Why FAISS?
Fastest, smallest, no server, no abstraction. The "vector DB" stripped to its essentials: a numpy matrix plus a nearest-neighbour routine.

## What you learned
- An **embedding** is a numpy array of floats (384 for `all-MiniLM-L6-v2`).
- **Nearest-neighbour** is literally dot product + sort — `IndexFlatIP` does nothing fancier.
- Normalising vectors turns inner product into cosine similarity.
- The whole "vector DB" at this scale is one `.faiss` file and one pickled list.

## What FAISS does NOT give you
- Metadata filtering (`where page >= 10`)
- A proper query language
- Persistence beyond a file dump
- Concurrent writes
- Auth, backups, horizontal scale

That's why Stage 2 moves to **ChromaDB**: it adds a query language and metadata without yet jumping to a client/server system.
