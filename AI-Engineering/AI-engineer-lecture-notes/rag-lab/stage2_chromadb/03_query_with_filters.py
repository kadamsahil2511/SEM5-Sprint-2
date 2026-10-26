"""Stage 2 / Step 3 — writing real ChromaDB queries: filters, where_document, include."""

import os
import chromadb
from chromadb.utils import embedding_functions

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "chroma_db")


def pretty(label: str, res: dict) -> None:
    print(f"\n=== {label} ===")
    docs = res.get("documents") or [[]]
    metas = res.get("metadatas") or [[]]
    dists = res.get("distances") or [[]]
    for i in range(len(docs[0])):
        meta = metas[0][i] if metas and metas[0] else None
        dist = dists[0][i] if dists and dists[0] else None
        doc = docs[0][i]
        print(f"#{i+1}  distance={dist}  meta={meta}")
        print(f"     {doc[:200] if doc else ''}")


def main():
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(name="handbook", embedding_function=embed_fn)

    # --------------------------------------------------------------
    # Query 1 — basic top-k.
    # Nothing fancy: just "find me the 5 nearest chunks to this text".
    # --------------------------------------------------------------
    res = collection.query(query_texts=["attendance rules"], n_results=5)
    pretty("Q1 — basic top-k", res)

    # --------------------------------------------------------------
    # Query 2 — filter by metadata with a `where` clause.
    # The `where` dict is applied BEFORE similarity ranking: Chroma
    # restricts the candidate pool, then ranks within that pool.
    # Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.
    # --------------------------------------------------------------
    res = collection.query(
        query_texts=["attendance rules"],
        n_results=5,
        where={"page": {"$gte": 10}},
    )
    pretty("Q2 — where page >= 10", res)

    # --------------------------------------------------------------
    # Query 3 — multi-condition filter using $and.
    # Combine conditions with $and / $or. Each clause is still an
    # {field: {op: value}} dict.
    # --------------------------------------------------------------
    res = collection.query(
        query_texts=["exam weightage"],
        n_results=5,
        where={
            "$and": [
                {"page": {"$lte": 20}},
                {"section_guess": {"$ne": "unknown"}},
            ]
        },
    )
    pretty("Q3 — page <= 20 AND section known", res)

    # --------------------------------------------------------------
    # Query 4 — filter on the document TEXT itself with `where_document`.
    # `where` filters on metadata. `where_document` filters on the raw text.
    # Operators: $contains, $not_contains.
    # --------------------------------------------------------------
    res = collection.query(
        query_texts=["placement"],
        n_results=5,
        where_document={"$contains": "percentage"},
    )
    pretty("Q4 — where_document contains 'percentage'", res)

    # --------------------------------------------------------------
    # Query 5 — choose which fields come back with `include`.
    # Default returns everything; pass `include` to trim the payload.
    # Valid values: "documents", "metadatas", "distances", "embeddings".
    # --------------------------------------------------------------
    res = collection.query(
        query_texts=["library hours"],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    pretty("Q5 — include documents + metadatas + distances", res)

    # --------------------------------------------------------------
    # WRITE YOUR OWN:
    # 1. Find chunks about "fees" only from the LAST 30 pages of the PDF.
    #    (Hint: find max page via collection.get() first, then use $gte.)
    # 2. Find chunks that mention "scholarship" AND are from a KNOWN section.
    #    (Combine where_document with where via $and-style thinking.)
    # 3. Return only the chunk text + distance (NO metadata) for any query.
    # --------------------------------------------------------------


if __name__ == "__main__":
    main()
