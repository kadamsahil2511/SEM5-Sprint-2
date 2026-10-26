"""Stage 1 / Step 1 — see embeddings as numpy arrays and build a FAISS index by hand."""

import os
import sys
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.loader import load_and_chunk

HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(HERE, "..", "data", "sample.pdf")
INDEX_PATH = os.path.join(HERE, "index.faiss")
CHUNKS_PATH = os.path.join(HERE, "chunks.pkl")


def main():
    print(f"Loading PDF from {PDF_PATH} ...")
    chunks = load_and_chunk(PDF_PATH, chunk_size=500, overlap=50)
    print(f"Produced {len(chunks)} chunks.")

    # NOTE: we call the embedding model DIRECTLY, no LangChain wrapper.
    # This is the whole point of Stage 1 — you SEE the model producing numbers.
    print("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    print()
    print("=== What an embedding looks like ===")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Embedding dimension   : {embeddings.shape[1]}")
    print(f"First 10 numbers of chunk 0: {embeddings[0][:10]}")
    print()

    # IndexFlatIP = inner product. On L2-normalized vectors this is exactly cosine similarity.
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    print(f"Built FAISS IndexFlatIP, ntotal = {index.ntotal}")

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved index  -> {INDEX_PATH}")
    print(f"Saved chunks -> {CHUNKS_PATH}")

    # TRY: change the model to 'all-mpnet-base-v2' and re-run. compare dim.
    # TRY: print the cosine similarity between the first two chunks manually.
    #      Hint:  np.dot(embeddings[0], embeddings[1])  (already normalized)


if __name__ == "__main__":
    main()
