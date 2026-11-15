from datetime import datetime, timezone
from collections import Counter
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from generate import answer
from retrieve import CHUNKS, search
from verify import verify_claim

app = FastAPI(title="Support Knowledge Copilot")
STATIC = Path(__file__).parent / "static"


class Ask(BaseModel):
    question: str


class Check(BaseModel):
    claim: str


def pack(chunks):
    return [
        {"n": i + 1, "title": s["title"], "heading": s["heading"], "url": s["url"], "text": s["text"]}
        for i, s in enumerate(chunks)
    ]


@app.get("/")
def home():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/classic")
def classic():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {
        "documents": len({c["title"] for c in CHUNKS}),
        "chunks": len(CHUNKS),
        "model": os.environ.get("GROQ_MODEL"),
        "retrieval": "Hybrid search (dense + BM25)",
    }
@app.get("/documents")
def documents():
    counts = Counter(c["title"] for c in CHUNKS)
    urls = {}
    for c in CHUNKS:
        urls.setdefault(c["title"], c["url"])
    index_file = Path("data/index.faiss")
    indexed = (
        datetime.fromtimestamp(index_file.stat().st_mtime, tz=timezone.utc).isoformat()
        if index_file.exists()
        else None
    )
    return [
        {
            "name": title,
            "type": "Handbook page",
            "chunks": n,
            "status": "Indexed",
            "last_indexed": indexed,
            "url": urls[title],
        }
        for title, n in sorted(counts.items())
    ]

@app.post("/ask")
def ask(body: Ask):
    out = answer(body.question)
    claims = []
    for c in out["claims"]:
        r = verify_claim(c["text"], c["sources"], out["sources"])
        claims.append({"text": c["text"], "sources": c["sources"], **r})
    return {"answerable": out["answerable"], "claims": claims, "sources": pack(out["sources"])}


@app.post("/check")
def check(body: Check):
    chunks = search(body.claim, k=5)
    r = verify_claim(body.claim, list(range(1, len(chunks) + 1)), chunks)
    return {"claim": body.claim, **r, "sources": pack(chunks)}