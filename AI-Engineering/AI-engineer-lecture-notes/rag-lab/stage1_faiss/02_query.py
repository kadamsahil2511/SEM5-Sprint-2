"""Stage 1 / Step 2 — nearest-neighbour search is just dot product + sort."""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

HERE = os.path.dirname(__file__)
INDEX_PATH = os.path.join(HERE, "index.faiss")
CHUNKS_PATH = os.path.join(HERE, "chunks.pkl")

query = "What is LLM?"


def main():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded index with {index.ntotal} vectors and {len(chunks)} chunks.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    q_vec = model.encode([query])
    q_vec = np.asarray(q_vec, dtype="float32")
    faiss.normalize_L2(q_vec)

    k = 4
    scores, idxs = index.search(q_vec, k)
    print(f"\nQuery: {query!r}")
    print(f"Top-{k} results (score = cosine similarity):\n")
    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        c = chunks[idx]
        print(f"#{rank}  score={score:.4f}  page={c['page']}  chunk_id={c['chunk_id']}")
        print(f"     {c['text'][:220]}")
        print()

    # TRY: change k from 4 to 10. observe quality.
    # TRY: compute similarity by hand using numpy.dot and verify FAISS score.
    #      Example:
    #         raw = np.load(...)  # or re-encode the first chunk
    #         print(np.dot(q_vec[0], raw / np.linalg.norm(raw)))
    # TRY: write 3 queries that should NOT match any chunk. does FAISS still
    #      return top-k? why? (hint: nearest-neighbour always returns SOMETHING;
    #      the score tells you how bad the match is — set a threshold).


if __name__ == "__main__":
    main()
