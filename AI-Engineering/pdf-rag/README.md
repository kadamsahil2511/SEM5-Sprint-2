# PDF chunk loader

```sh
source ../../.venv/bin/activate
python loader.py path/to/document.pdf
python query.py "What is the main topic of the document?"
python chatbot.py
python baseline_rag.py path/to/document.pdf
```

The command saves `faiss.index` (normalized vector index) and `chunks.pkl`
(ordered chunk records) in this directory. Each record contains `chunk_id`, `text`,
the 1-based `page`, and the absolute `source` path. IDs are stable for the
same PDF contents and chunking settings. Rerunning the command replaces both
output files, so this simple loader holds one PDF at a time.
The loader uses layout-based PDF extraction and collapses extra whitespace
before chunking, so result previews read as normal sentences.

The embedding model is `nomic-ai/nomic-embed-text-v2-moe`. Its first use
downloads model weights and executes the model's Hugging Face code. The loader
uses the model's `passage` prompt. `query.py` uses the same model with its
`query` prompt, normalizes the query vector, and prints up to four matching
chunks in a table with scores and page numbers. The table's short `Chunk`
column is the per-page chunk number; the full ID remains in `chunks.pkl`.
Both scripts run the model on CPU with one
PyTorch thread to avoid the macOS OpenMP crash seen while loading it. Only
load the `.pkl` file if you trust its source, since pickle can execute code.

`chatbot.py` uses the cached Nomic model for retrieval and the locally installed
`gemma4:e2b-mlx` Ollama model for answers. Start Ollama first if needed with
`ollama serve`. It includes the PDF's cover chunk with the top four matches,
uses only `127.0.0.1`, and keeps each question independent.
Download both models before disconnecting from the internet.

`baseline_rag.py` is a separate, one-PDF-at-a-time chatbot. It uses the same
PDF chunks, Nomic embeddings, Ollama model, and answer prompt as `chatbot.py`.
It stores normalized embeddings and chunk metadata in `.chroma/`, retrieves up
to 20 candidates, then uses `cross-encoder/ms-marco-MiniLM-L6-v2` to rerank
them and passes the best four to Ollama. The displayed `rerank_score` is a
model score, not a cosine similarity or probability. Reopening the same PDF
reuses its Chroma index; changed PDF contents get a separate collection. The first run
downloads the cross encoder model, so download it before working offline.

Next RAG step: test answer quality and citations against representative questions.
For scanned PDFs, add OCR before chunking. A generation model/client and OCR
package are not included in this loader's dependencies.
