import re

import numpy as np
from sentence_transformers import CrossEncoder

NLI = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
# is model ke output ka order: contradiction, entailment, neutral


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def windows(text, size=3):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 20]
    if not sents:
        return [text[:1000]]
    return [" ".join(sents[i:i + size]) for i in range(max(len(sents) - size + 1, 1))]


def verify_claim(claim, cites, sources):
    valid = [c for c in cites if isinstance(c, int) and 1 <= c <= len(sources)]
    if not valid:
        return {"verdict": "INVALID_CITATION", "entail": 0.0, "contradict": 0.0}
    pairs = [(w, claim) for c in valid for w in windows(sources[c - 1]["text"])]
    probs = softmax(NLI.predict(pairs))
    ent, con = float(probs[:, 1].max()), float(probs[:, 0].max())
    if ent >= 0.5:
        verdict = "VERIFIED"
    elif con >= 0.5:
        verdict = "CONTRADICTED"
    else:
        verdict = "UNSUPPORTED"
    return {"verdict": verdict, "entail": round(ent, 2), "contradict": round(con, 2)}


if __name__ == "__main__":
    from generate import answer

    out = answer("How much paid parental leave do team members get?")
    claims = list(out["claims"])
    # ek jaanbujh ke galat claim, taaki dekh sakein ki system pakadta hai ya nahi
    claims.append({"text": "GitLab team members receive 6 weeks of paid parental leave.", "sources": [1]})
    claims.append({"text": "Team members get a free company car.", "sources": [1]})

    for c in claims:
        r = verify_claim(c["text"], c["sources"], out["sources"])
        print(f"{r['verdict']:<16} entail={r['entail']} contradict={r['contradict']} | {c['text']}")