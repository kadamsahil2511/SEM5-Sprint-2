# Similarity Metric Visualizer — Day 5

Dot product, Euclidean distance and cosine similarity, written in plain NumPy and
tried on real sentence embeddings. There's no vector DB and no LangChain, so
you work with the math directly.

## Setup

```bash
cd project-similarity
source ../../../.venv/bin/activate  # Install once from the sprint root first.
```

You don't need an API key. The first run of 02 or 03 downloads the
`all-MiniLM-L6-v2` model, which is about 90 MB.

## Run order

```bash
python 01_metrics_explained.py     # pure NumPy, runs instantly
python 02_text_search.py           # type queries at the prompt; press Enter on an empty line to quit
streamlit run 03_visualizer.py
```

## LEARNING GOALS

- **After 01:** you can explain the three metrics with NumPy, not formulas.
- **After 02:** you can predict which metric will rank differently and why.
- **After 03:** you can see metric choice in 2D on real text embeddings.

## Why the "raw" embeddings?

`all-MiniLM-L6-v2` normally ends with a `Normalize` layer, so every vector it
returns has length 1. With length-1 vectors, dot product and cosine give exactly
the same ranking, and the difference between them disappears.

So 02 and 03 remove that last layer and use the **raw** vectors, whose lengths
differ from sentence to sentence. Now dot product can reward long vectors, and
you can watch it happen.

## What to try

1. **In `02_text_search.py`, flip `NORMALIZE` to `True`.** Dot product now agrees
   with cosine. Explain why.
   Hint: cos(θ) = (A · B) / (‖A‖ × ‖B‖). If ‖A‖ = ‖B‖ = 1, then cos(θ) = A · B.
2. **In `03_visualizer.py`, add a sentence that is a LONG paragraph** (4–5 lines
   about anything). **Predict first:** will dot product boost it? Then check the
   "vector length" column.
   Surprise: the long paragraph gets the *shortest* vector (about 2.4, while short
   sentences are about 5–7). The model averages over all the words, and averaging
   many different words cancels them out. So:
   - **Dot product** pushes it DOWN, because a short vector means a small dot product.
   - **Euclidean** pulls it UP to #1 for almost every query. A short vector sits near
     the origin, and the origin is close to everything.
   - **Cosine** ignores length and judges the paragraph on meaning alone, which is fair.

   Lesson: magnitude-sensitive metrics depend on vector length, and vector length
   depends on things like text length that have nothing to do with relevance.
3. **Add a sentence in a different language** (Hindi, Marathi, French...). Which
   metric still finds it? Note that this model was trained mostly on English.

## Files

| File | What it does |
|---|---|
| `01_metrics_explained.py` | Implements the 3 metrics in NumPy, applies them to 4 toy vectors, and prints the full table plus 3 observations. |
| `02_text_search.py` | A 10-sentence search engine that prints the ranking from all three metrics side by side. |
| `03_visualizer.py` | A Streamlit app: your sentences on a 2D map (PCA / t-SNE), the query as a red star, and a top-5 comparison across metrics. |
