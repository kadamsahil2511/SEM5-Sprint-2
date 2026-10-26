"""Stage 3 / Step 1 — index the same PDF into Qdrant through the LangChain abstraction.

Compare this file to stage1_faiss/01_build_index.py. The framework hid
the embedding model call, the vector store, the normalization, the ID
generation. That's abstraction earned, not abstraction skipped.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.loader import load_and_chunk

load_dotenv()

HERE = os.path.dirname(__file__)
PDF_PATH = os.path.join(HERE, "..", "data", "sample.pdf")
COLLECTION = "handbook"


def main():
    print(f"Loading PDF from {PDF_PATH} ...")
    chunks = load_and_chunk(PDF_PATH, chunk_size=500, overlap=50)
    print(f"Produced {len(chunks)} chunks.")

    # Convert plain dicts (shared with Stages 1 & 2) into LangChain Documents in-place.
    documents = [
        Document(
            page_content=c["text"],
            metadata={"page": c["page"], "chunk_id": c["chunk_id"]},
        )
        for c in chunks
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    QdrantVectorStore.from_documents(
        documents,
        embeddings,
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        collection_name=COLLECTION,
        force_recreate=True,
    )
    print(f"indexed {len(documents)} chunks into Qdrant collection {COLLECTION!r}")


if __name__ == "__main__":
    main()
