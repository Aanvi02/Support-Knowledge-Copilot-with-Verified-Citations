import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
CLIENT = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ["GROQ_MODEL"]

chunks = json.loads(Path("data/chunks.json").read_text(encoding="utf-8"))
pool = [c for c in chunks if len(c["text"]) > 400]
random.seed(7)
sample = random.sample(pool, 30)

PROMPT = """Write ONE specific question an employee could ask that is answered by the passage below.
The question must make sense on its own (never say 'the passage' or 'this section').
Return only the question."""

items = []
for c in sample:
    r = CLIENT.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"{c['title']} > {c['heading']}\n{c['text']}"},
        ],
    )
    q = r.choices[0].message.content.strip()
    items.append({"question": q, "chunk_id": c["id"], "url": c["url"], "answerable": True})
    print(q)
    time.sleep(2)  # free tier ki rate limit ke liye

for q in [
    "What is the weather in Bangalore today?",
    "Who won the last cricket world cup?",
    "How do I reset my Netflix password?",
    "What is the recipe for chicken biryani?",
    "Which programming language is best for beginners?",
    "What is the capital of Australia?",
]:
    items.append({"question": q, "chunk_id": None, "url": None, "answerable": False})

Path("data/eval_set.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved {len(items)} questions -> data/eval_set.json")