"""02 — A 10-sentence search engine, ranked three ways.

Teaching goal: run the SAME query through dot product, Euclidean distance and
cosine similarity on real sentence embeddings, and see where the rankings
agree and where they don't.

Detail: all-MiniLM-L6-v2 normally L2-normalizes its output (its last layer
is a `Normalize` module). With normalized vectors, dot product and cosine give
IDENTICAL rankings and the lesson disappears. So we drop that last layer and
use the RAW mean-pooled embeddings, whose lengths differ from sentence to
sentence. Then flip NORMALIZE to True and watch dot product's ranking
collapse into cosine's.

Run:  python 02_text_search.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

NORMALIZE = False   # <- flip to True and rerun: dot product becomes cosine
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 10
COL = 44

SENTENCES = [
    # food
    "I love samosas",
    "Dosa is the best breakfast",
    "Grandma's biryani needs hours of slow cooking",
    # code
    "Python lists are zero-indexed",
    "I fixed the bug in my JavaScript for-loop",
    "Git merge conflicts ruin my evenings",
    # sports
    "Virat Kohli scored a century at Wankhede",
    "Football practice starts at 6 am sharp",
    "She won the state badminton final",
    # outlier
    "The moon looks orange tonight",
]


def dot_product(a, b) -> float:
    return float(np.sum(a * b))


def euclidean_distance(a, b) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_similarity(a, b) -> float:
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))))


def load_raw_model() -> SentenceTransformer:
    """all-MiniLM-L6-v2 minus its final Normalize layer -> raw, un-normalized vectors."""
    full = SentenceTransformer(MODEL_NAME)
    return SentenceTransformer(modules=[m for m in full if type(m).__name__ != "Normalize"])


def encode(model, texts):
    vecs = model.encode(texts, convert_to_numpy=True)
    if NORMALIZE:
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def rank(query_vec, doc_vecs, metric, higher_is_better):
    scores = [metric(query_vec, d) for d in doc_vecs]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    return [(i, scores[i]) for i in order]


def cell(rank_no, idx, score):
    text = SENTENCES[idx]
    text = text if len(text) <= 28 else text[:27] + "…"
    return f"{rank_no:>2}. {text:<28} {score:>8.3f}"


def show(query, query_vec, doc_vecs):
    rankings = {
        "DOT PRODUCT (high = close)": rank(query_vec, doc_vecs, dot_product, True),
        "EUCLIDEAN (low = close)": rank(query_vec, doc_vecs, euclidean_distance, False),
        "COSINE (high = close)": rank(query_vec, doc_vecs, cosine_similarity, True),
    }
    print(f"\nQuery: {query!r}   (query vector length = {np.linalg.norm(query_vec):.2f})\n")
    print("".join(f"{name:<{COL}}" for name in rankings))
    print("".join(f"{'-' * (COL - 2):<{COL}}" for _ in rankings))
    for r in range(TOP_K):
        row = []
        for ranking in rankings.values():
            idx, score = ranking[r]
            row.append(f"{cell(r + 1, idx, score):<{COL}}")
        firsts = {ranking[r][0] for ranking in rankings.values()}
        print("".join(row) + ("" if len(firsts) == 1 else "  ← disagree"))


print(f"Loading {MODEL_NAME} (first run downloads ~90 MB)...")
model = load_raw_model()
doc_vecs = encode(model, SENTENCES)

print(f"\nNORMALIZE = {NORMALIZE}. Vector lengths of the 10 sentences:")
for s, v in zip(SENTENCES, doc_vecs):
    print(f"  {np.linalg.norm(v):6.3f}  {s}")
print("\nLonger vectors get a head start under dot product. Cosine ignores length.")

while True:
    query = input("\nType a query (or press Enter to quit): ").strip()
    if not query:
        break
    show(query, encode(model, [query])[0], doc_vecs)

print("\nIf we L2-normalize the vectors, dot product becomes equivalent to cosine.")
print("Try changing `NORMALIZE = False` to `True` at the top of this file.")
