import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
texts = [f"{c['title']} - {c['heading']}\n{c['text']}" for c in chunks]

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 512
emb = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
emb = np.asarray(emb, dtype="float32")

index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, "data/index.faiss")
print(f"\nIndex saved: {index.ntotal} vectors -> data/index.faiss")