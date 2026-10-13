from fastapi import FastAPI
from pydantic import BaseModel

from generate import answer
from verify import verify_claim

app = FastAPI(title="Support Knowledge Copilot")


class Ask(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Ask):
    out = answer(body.question)
    claims = []
    for c in out["claims"]:
        r = verify_claim(c["text"], c["sources"], out["sources"])
        claims.append({"text": c["text"], "sources": c["sources"], **r})
    return {
        "answerable": out["answerable"],
        "claims": claims,
        "sources": [
            {"n": i + 1, "title": s["title"], "heading": s["heading"], "url": s["url"], "text": s["text"][:400]}
            for i, s in enumerate(out["sources"])
        ],
    }