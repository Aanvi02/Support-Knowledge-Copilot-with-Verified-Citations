import json
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
INDEX = faiss.read_index("data/index.faiss")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def tokenize(text):
    return re.findall(r"\w+", text.lower())


BM25 = BM25Okapi([tokenize(f"{c['title']} {c['heading']} {c['text']}") for c in CHUNKS])


def search(query, k=5, pool=20, rrf_k=60):
    q = np.asarray(MODEL.encode([query], normalize_embeddings=True), dtype="float32")
    _, dense_ids = INDEX.search(q, pool)
    sparse_ids = np.argsort(BM25.get_scores(tokenize(query)))[::-1][:pool]

    # Reciprocal Rank Fusion: dono rankings ko merge karna
    fused = {}
    for ranking in (dense_ids[0], sparse_ids):
        for rank, i in enumerate(ranking):
            fused[int(i)] = fused.get(int(i), 0) + 1 / (rrf_k + rank + 1)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [CHUNKS[i] for i in top]


if __name__ == "__main__":
    for c in search("How do I request parental leave?"):
        print(f"- {c['title']} > {c['heading']}\n  {c['url']}")