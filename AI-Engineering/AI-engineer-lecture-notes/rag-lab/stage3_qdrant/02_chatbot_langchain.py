"""Stage 3 / Step 2 — the same RAG chatbot as Stage 1, now ~15 lines of real work.

Count the lines of code. Compare to stage1_faiss/03_chatbot.py.
The framework collapsed ~60 lines into ~15. That's the trade:
less code, more magic. Only use magic you understand.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

COLLECTION = "handbook"


@st.cache_resource
def build_qa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY") or None,
        collection_name=COLLECTION,
    )
    # search_type="similarity" is the default. Swap to "mmr" with one argument
    # change for diversity-aware retrieval (full MMR treatment is Day 7).
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )


def main():
    st.set_page_config(page_title="Stage 3 — Qdrant + LangChain", layout="wide")
    st.title("Stage 3 — Qdrant RAG through LangChain")

    qa = build_qa()
    q = st.chat_input("Ask something about your PDF")
    if not q:
        st.caption("Type a question. Retrieval happens in Qdrant, generation happens on Groq.")
        return

    result = qa.invoke({"query": q})
    st.markdown("### Answer")
    st.write(result["result"])

    sources = result.get("source_documents", [])
    with st.expander(f"Sources ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page", "?")
            st.caption(f"Chunk {i} — page {page}")
            st.write(doc.page_content[:400])


if __name__ == "__main__":
    main()
