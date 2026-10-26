"""Stage 2 / Step 2 — a plain similarity query, with distances and metadata printed."""

import os
import chromadb
from chromadb.utils import embedding_functions

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "chroma_db")


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name="handbook", embedding_function=embed_fn)

    queries = [
        # TRY these in order and watch what happens:
        "attendance policy",                           # a straightforward hit
        "do I have to come to class",                  # semantic — no keyword overlap
        "list all the rules",                          # ambiguous — expect noise
    ]

    for q in queries:
        print(f"\n=== Query: {q!r} ===")
        res = collection.query(query_texts=[q], n_results=4)
        for i, (doc, meta, dist) in enumerate(
            zip(res["documents"][0], res["metadatas"][0], res["distances"][0]), start=1
        ):
            print(f"#{i}  distance={dist:.4f}  meta={meta}")
            print(f"     {doc[:200]}")


if __name__ == "__main__":
    main()
