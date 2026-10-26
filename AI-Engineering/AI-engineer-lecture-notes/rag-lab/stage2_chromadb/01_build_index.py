"""Stage 2 / Step 1 — persistent ChromaDB collection with hand-attached metadata."""

import os
import sys
import re
import chromadb
from chromadb.utils import embedding_functions

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.loader import load_and_chunk

HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(HERE, "..", "data", "sample.pdf")
DB_PATH = os.path.join(HERE, "chroma_db")

ALL_CAPS = re.compile(r"^[A-Z][A-Z0-9 \-&/]{3,}$")


def guess_section(text: str) -> str:
    """Heuristic: last ALL-CAPS line that appears early in the chunk."""
    for line in text.splitlines():
        line = line.strip()
        if ALL_CAPS.match(line):
            return line
    return "unknown"


def main():
    print(f"Loading PDF from {PDF_PATH} ...")
    chunks = load_and_chunk(PDF_PATH, chunk_size=500, overlap=50)
    print(f"Produced {len(chunks)} chunks.")

    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Fresh collection on every build.
    try:
        client.delete_collection("handbook")
    except Exception:
        pass
    collection = client.create_collection(
        name="handbook",
        embedding_function=embed_fn,
    )

    docs, metas, ids = [], [], []
    for c in chunks:
        text = c["text"]
        metas.append({
            "page": int(c["page"]),
            "chunk_id": int(c["chunk_id"]),
            "section_guess": guess_section(text),
            "word_count": len(text.split()),
        })
        docs.append(text)
        ids.append(f"chunk-{c['chunk_id']}")

    # Chroma embeds under the hood using the embedding_function we attached.
    collection.add(documents=docs, metadatas=metas, ids=ids)
    print(f"Indexed {collection.count()} items into collection 'handbook' at {DB_PATH}")

    print("\nFirst 3 stored items:")
    sample = collection.get(ids=ids[:3], include=["documents", "metadatas"])
    for i, (d, m) in enumerate(zip(sample["documents"], sample["metadatas"]), start=1):
        print(f"\n  #{i}  metadata={m}")
        print(f"       text={d[:180]}")


if __name__ == "__main__":
    main()
