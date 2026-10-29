from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from generate import answer
from retrieve import search
from verify import verify_claim

app = FastAPI(title="Support Knowledge Copilot")


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
    return FileResponse(Path(__file__).parent / "static" / "index.html")


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