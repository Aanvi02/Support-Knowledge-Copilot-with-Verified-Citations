import json
import sys

sys.path.insert(0, "app")
from retrieve import search

r = json.load(open("evals/results.json", encoding="utf-8"))
lines = []
for x in r:
    bad = [c for c in x["claims"] if c["verdict"] != "VERIFIED"]
    if not bad:
        continue
    chunks = search(x["question"], k=5)
    for c in bad:
        lines.append(f"{c['verdict']} (entail={c['entail']})")
        lines.append(f"Q: {x['question']}")
        lines.append(f"A: {c['text']} {c['sources']}")
        for s in c["sources"]:
            if isinstance(s, int) and 1 <= s <= len(chunks):
                lines.append(f"SOURCE [{s}] {chunks[s - 1]['heading']}: {chunks[s - 1]['text'][:700]}")
        lines.append("-" * 60)

open("evals/errors.txt", "w", encoding="utf-8").write("\n".join(lines))
print(len(lines), "lines saved -> evals/errors.txt")