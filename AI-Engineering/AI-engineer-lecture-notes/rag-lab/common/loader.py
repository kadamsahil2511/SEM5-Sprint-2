"""Shared PDF loader + chunker. Returns plain dicts so Stage 1 stays framework-free."""

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """Load a PDF, split into chunks, and return list of {text, page, chunk_id} dicts.

    The dicts are intentionally framework-free so Stage 1 (raw FAISS) can use them
    without pulling in a LangChain Document wrapper.
    """
    reader = PdfReader(pdf_path)

    # Keep page -> text so we can propagate page numbers through chunking.
    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_num, text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    chunk_id = 0
    for page_num, text in pages:
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "text": piece,
                "page": page_num,
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    return chunks
