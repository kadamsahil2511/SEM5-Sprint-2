"""Stage 2 / Step 4 — Streamlit chatbot where sidebar controls become metadata filters."""

import os
import sys
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.llm import generate_answer

HERE = os.path.dirname(__file__)
DB_PATH = os.path.join(HERE, "chroma_db")


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_collection(name="handbook", embedding_function=embed_fn)


@st.cache_data
def collection_summary():
    col = get_collection()
    all_items = col.get(include=["metadatas"])
    metas = all_items["metadatas"] or []
    pages = [m["page"] for m in metas if "page" in m]
    sections = sorted({m.get("section_guess", "unknown") for m in metas})
    min_p = min(pages) if pages else 0
    max_p = max(pages) if pages else 0
    return min_p, max_p, sections


def build_where(page_range, section):
    conds = [
        {"page": {"$gte": int(page_range[0])}},
        {"page": {"$lte": int(page_range[1])}},
    ]
    if section and section != "(any)":
        conds.append({"section_guess": {"$eq": section}})
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


def main():
    st.set_page_config(page_title="Stage 2 — Chroma RAG", layout="wide")
    st.title("Stage 2 — Chroma RAG with metadata filters")

    min_p, max_p, sections = collection_summary()
    with st.sidebar:
        st.header("Filters")
        page_range = st.slider(
            "Page range",
            min_value=int(min_p),
            max_value=int(max_p) if max_p > min_p else int(min_p) + 1,
            value=(int(min_p), int(max_p) if max_p > min_p else int(min_p) + 1),
        )
        section = st.selectbox("Section", options=["(any)"] + sections)
        k = st.slider("Top-k", 1, 10, 4)

    q = st.chat_input("Ask something about your PDF")
    if not q:
        st.caption("Pick filters on the left, then type a question below.")
        return

    collection = get_collection()
    where = build_where(page_range, section)
    res = collection.query(
        query_texts=[q],
        n_results=k,
        where=where,
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    hits = [{"text": d, "page": m.get("page"), "section": m.get("section_guess"), "distance": dist}
            for d, m, dist in zip(docs, metas, dists)]

    if not hits:
        st.warning("No chunks matched the current filters. Loosen them and retry.")
        return

    answer = generate_answer(q, hits)
    st.markdown("### Answer")
    st.write(answer)

    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            st.caption(f"Chunk {i} — page {h['page']} — section {h['section']} — distance {h['distance']:.3f}")
            st.write(h["text"][:400])


if __name__ == "__main__":
    main()
