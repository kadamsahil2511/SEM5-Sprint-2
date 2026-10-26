"""03 — See metric choice in 2D on real text embeddings.

Teaching goal: paste your own sentences, project their 384-dim embeddings down
to 2D, drop a query in as a red star, and rank the sentences by dot product,
Euclidean distance or cosine similarity. The comparison table at the bottom
highlights in orange every rank where the three metrics pick different
sentences.

Like 02_text_search.py, this uses the model's RAW embeddings (its final
Normalize layer is removed), so vector length matters and dot product really
does behave differently from cosine.

Run:  streamlit run 03_visualizer.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
ORANGE = "#ffd8a8"

DEFAULT_SENTENCES = """I love samosas
Dosa is the best breakfast
Grandma's biryani needs hours of slow cooking
Street-side pani puri in Mumbai is unbeatable
Python lists are zero-indexed
I fixed the bug in my JavaScript for-loop
Git merge conflicts ruin my evenings
Recursion finally clicked for me today
Virat Kohli scored a century at Wankhede
Football practice starts at 6 am sharp
She won the state badminton final
The moon looks orange tonight"""


def dot_product(a, b) -> float:
    return float(np.sum(a * b))


def euclidean_distance(a, b) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_similarity(a, b) -> float:
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))))


METRICS = {
    "dot": (dot_product, True),
    "euclidean": (euclidean_distance, False),
    "cosine": (cosine_similarity, True),
}


@st.cache_resource
def load_raw_model() -> SentenceTransformer:
    full = SentenceTransformer(MODEL_NAME)
    return SentenceTransformer(modules=[m for m in full if type(m).__name__ != "Normalize"])


def rank(query_vec, doc_vecs, metric_name):
    metric, higher_is_better = METRICS[metric_name]
    scores = [metric(query_vec, d) for d in doc_vecs]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=higher_is_better)
    return [(i, scores[i]) for i in order]


def project_2d(doc_vecs, query_vec, method):
    """Returns (doc_points, query_point_or_None)."""
    if method == "PCA":
        pca = PCA(n_components=2).fit(doc_vecs)
        q = pca.transform(query_vec[None, :])[0] if query_vec is not None else None
        return pca.transform(doc_vecs), q
    # t-SNE can't place a new point into an existing map, so the query is fitted with the docs.
    data = doc_vecs if query_vec is None else np.vstack([doc_vecs, query_vec])
    perplexity = max(2, min(30, (len(data) - 1) // 3))
    pts = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(data)
    return (pts, None) if query_vec is None else (pts[:-1], pts[-1])


def short(text, n=20):
    return text if len(text) <= n else text[: n - 1] + "…"


st.set_page_config(page_title="Similarity Visualizer", layout="wide")
st.title("Similarity Metric Visualizer")
st.caption("Same vectors, three rulers: dot product · Euclidean distance · cosine similarity")

state = st.session_state
model = load_raw_model()
left, right = st.columns(2)

with left:
    st.subheader("1. Your sentences")
    raw_text = st.text_area("Paste 10–30 short sentences, one per line", DEFAULT_SENTENCES, height=300)
    if st.button("Encode", type="primary"):
        sentences = [s.strip() for s in raw_text.splitlines() if s.strip()]
        if len(sentences) < 3:
            st.error("Need at least 3 sentences.")
        else:
            if not 10 <= len(sentences) <= 30:
                st.warning(f"You have {len(sentences)} sentences — 10 to 30 works best.")
            state.sentences = sentences
            state.doc_vecs = model.encode(sentences, convert_to_numpy=True)
            state.query = None
            state.query_vec = None
    method = st.selectbox("Dimensionality reduction (384D → 2D)", ["PCA", "t-SNE"])

with right:
    st.subheader("2. Query")
    with st.form("query_form"):
        query = st.text_input("Query", placeholder="e.g. something tasty to eat")
        metric_name = st.radio("Metric", list(METRICS), index=2, horizontal=True)
        submitted = st.form_submit_button("Search")
    if submitted:
        if "doc_vecs" not in state:
            st.warning("Click Encode on the left first.")
        elif query.strip():
            state.query = query.strip()
            state.query_vec = model.encode([state.query], convert_to_numpy=True)[0]

if "doc_vecs" not in state:
    left.info("Click **Encode** to embed your sentences and draw the map.")
    st.stop()

sentences, doc_vecs = state.sentences, state.doc_vecs
query_vec = state.get("query_vec")
ranking = rank(query_vec, doc_vecs, metric_name) if query_vec is not None else None

with left:
    doc_pts, q_pt = project_2d(doc_vecs, query_vec, method)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(doc_pts[:, 0], doc_pts[:, 1], s=60, c="#9c36b5", alpha=0.8)
    for (x, y), s in zip(doc_pts, sentences):
        ax.annotate(short(s), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    if q_pt is not None:
        for idx, _ in ranking[:TOP_K]:
            ax.plot([q_pt[0], doc_pts[idx, 0]], [q_pt[1], doc_pts[idx, 1]], c="red", lw=0.8, alpha=0.4)
        ax.scatter([q_pt[0]], [q_pt[1]], marker="*", s=450, c="red", edgecolors="black", zorder=5)
        ax.annotate("QUERY", q_pt, xytext=(8, -12), textcoords="offset points", color="red", fontweight="bold")
    ax.set_title(f"{method} projection" + (f" — red lines = top {TOP_K} by {metric_name}" if ranking else ""))
    ax.set_xticks([]); ax.set_yticks([])
    st.pyplot(fig)
    plt.close(fig)
    st.caption("2D is a squashed view of 384D: distances on the plot are only approximate. "
               "The ranking uses the full 384-dim vectors.")

with right:
    if ranking is None:
        st.info("Type a query and press **Search**.")
    else:
        direction = "lower = closer" if metric_name == "euclidean" else "higher = closer"
        st.markdown(f"**Top {TOP_K} for** “{state.query}” **by {metric_name}** ({direction})")
        st.dataframe(pd.DataFrame(
            [{"rank": r + 1, "sentence": sentences[i], "score": round(s, 4),
              "vector length": round(float(np.linalg.norm(doc_vecs[i])), 3)}
             for r, (i, s) in enumerate(ranking[:TOP_K])]),
            hide_index=True, width="stretch")
        st.caption(f"Query vector length: {np.linalg.norm(query_vec):.3f}. "
                   "Dot product rewards long vectors; cosine ignores length.")

if ranking is not None:
    st.divider()
    st.subheader("3. All three metrics side by side")
    k = min(TOP_K, len(sentences))
    all_rankings = {name: rank(query_vec, doc_vecs, name) for name in METRICS}
    rows, disagree = [], []
    for r in range(k):
        picks = {name: all_rankings[name][r] for name in METRICS}
        rows.append({"rank": r + 1, **{name: f"{sentences[i]}  ({s:.3f})" for name, (i, s) in picks.items()}})
        disagree.append(len({i for i, _ in picks.values()}) > 1)
    table = pd.DataFrame(rows)
    styled = table.style.apply(
        lambda row: [f"background-color: {ORANGE}" if disagree[row.name] else ""] * len(row), axis=1)
    st.dataframe(styled, hide_index=True, width="stretch")
    n_dis = sum(disagree)
    st.caption(f"Orange rows = the three metrics picked DIFFERENT sentences at that rank "
               f"({n_dis} of {k}). Rank 1 {'differs' if disagree[0] else 'agrees'} across metrics.")

    st.divider()
    st.subheader("4. How each score was calculated")
    pick = st.selectbox("Sentence to compare with the query", range(len(sentences)),
                        index=ranking[0][0], format_func=lambda i: sentences[i])
    q, d = query_vec, doc_vecs[pick]
    n_show = 3
    dims = len(q)
    dot = dot_product(q, d)
    q_len, d_len = float(np.linalg.norm(q)), float(np.linalg.norm(d))

    def vec_tex(v):
        return r"[" + ", ".join(f"{x:.3f}" for x in v[:n_show]) + r", \dots]"

    st.latex(rf"\mathbf{{q}} = {vec_tex(q)} \qquad \mathbf{{d}} = {vec_tex(d)} \qquad ({dims}\text{{ dims}})")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Dot product** (higher = closer)")
        st.latex(r"\mathbf{q}\cdot\mathbf{d} = \sum_{i=1}^{n} q_i d_i")
        terms = " + ".join(f"({q[i]:.3f})({d[i]:.3f})" for i in range(n_show))
        st.latex(rf"= {terms} + \dots")
        st.latex(rf"= {dot:.4f}")
    with c2:
        st.markdown("**Euclidean distance** (lower = closer)")
        st.latex(r"\lVert\mathbf{q}-\mathbf{d}\rVert = \sqrt{\sum_{i=1}^{n} (q_i - d_i)^2}")
        terms = " + ".join(f"({q[i]:.3f} - {d[i]:.3f})^2" for i in range(n_show))
        sq_sum = float(np.sum((q - d) ** 2))
        st.latex(rf"= \sqrt{{{terms} + \dots}}")
        st.latex(rf"= \sqrt{{{sq_sum:.4f}}} = {euclidean_distance(q, d):.4f}")
    with c3:
        st.markdown("**Cosine similarity** (higher = closer)")
        st.latex(r"\cos\theta = \frac{\mathbf{q}\cdot\mathbf{d}}{\lVert\mathbf{q}\rVert\,\lVert\mathbf{d}\rVert}")
        st.latex(rf"= \frac{{{dot:.4f}}}{{{q_len:.3f} \times {d_len:.3f}}}")
        st.latex(rf"= {cosine_similarity(q, d):.4f}")
    st.caption(f"Only the first {n_show} of {dims} terms are written out; the result sums all {dims}. "
               r"Vector length ‖v‖ = √(Σ vᵢ²). Cosine is just the dot product divided by both lengths.")
