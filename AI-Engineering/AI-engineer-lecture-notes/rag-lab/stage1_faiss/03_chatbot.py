"""Stage 1 / Step 3 — CLI RAG loop. No Streamlit, no magic, retrieval in plain view."""

import os
import sys
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.llm import generate_answer

HERE = os.path.dirname(__file__)
INDEX_PATH = os.path.join(HERE, "index.faiss")
CHUNKS_PATH = os.path.join(HERE, "chunks.pkl")


def main():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Stage 1 RAG chatbot (FAISS). Type 'quit' to exit.")
    print("One-shot QA — no chat history. Retrieval first, generation second.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break

        q_vec = model.encode([q])
        q_vec = np.asarray(q_vec, dtype="float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = index.search(q_vec, 4)

        hits = []
        for score, idx in zip(scores[0], idxs[0]):
            c = chunks[idx]
            hits.append({"text": c["text"], "score": float(score), "page": c["page"], "chunk_id": c["chunk_id"]})

        answer = generate_answer(q, hits)
        print(f"\nBot: {answer}\n")
        print("Used chunks:")
        for i, h in enumerate(hits, start=1):
            print(f"  [Chunk {i}] score={h['score']:.3f}  page={h['page']}  chunk_id={h['chunk_id']}")
        print()


if __name__ == "__main__":
    main()
