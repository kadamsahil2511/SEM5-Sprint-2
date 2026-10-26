"""Shared answer generator. Groq + llama-3.3-70b with the standard RAG prompt."""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_SYSTEM = (
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know.\" "
    "Cite the chunk number for every fact, like [Chunk 2]."
)


def _format_context(context_chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(context_chunks, start=1):
        text = c.get("text") or c.get("page_content") or ""
        lines.append(f"[Chunk {i}] {text}")
    return "\n\n".join(lines)


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Call Groq with the RAG prompt. context_chunks is a list of {text, metadata} dicts."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context = _format_context(context_chunks)
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    )
    return resp.choices[0].message.content
