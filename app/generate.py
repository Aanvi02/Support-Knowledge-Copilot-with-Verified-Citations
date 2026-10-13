import json
import os

from dotenv import load_dotenv
from groq import Groq

from retrieve import search

load_dotenv()
CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ["GROQ_MODEL"]

SYSTEM = """You answer questions using ONLY the numbered sources provided.
Return JSON only, in this exact shape:
{"answerable": true, "claims": [{"text": "one factual sentence", "sources": [1]}]}
Rules:
- Every claim must be supported by the source numbers listed in "sources".
- Never use outside knowledge and never cite a number that was not provided.
- If the sources do not contain the answer, return {"answerable": false, "claims": []}."""


def answer(question, k=5):
    chunks = search(question, k=k)
    sources = "\n\n".join(
        f"[{i + 1}] {c['title']} > {c['heading']}\n{c['text']}" for i, c in enumerate(chunks)
    )
    resp = CLIENT.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return {
        "question": question,
        "answerable": data.get("answerable", False),
        "claims": data.get("claims", []),
        "sources": chunks,
    }


if __name__ == "__main__":
    out = answer("How much paid parental leave do team members get?")
    print("Answerable:", out["answerable"])
    for c in out["claims"]:
        print(f"- {c['text']}  {c['sources']}")
    print("\nSources:")
    for i, s in enumerate(out["sources"], 1):
        print(f"[{i}] {s['title']} > {s['heading']}")