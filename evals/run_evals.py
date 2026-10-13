import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "app")

ITEMS = json.loads(Path("data/eval_set.json").read_text(encoding="utf-8"))


def retrieval_eval():
    from retrieve import search

    ans = [x for x in ITEMS if x["answerable"]]
    ranks, page_hits = [], 0
    for x in ans:
        res = search(x["question"], k=5)
        ids = [c["id"] for c in res]
        ranks.append(ids.index(x["chunk_id"]) + 1 if x["chunk_id"] in ids else None)
        page_hits += x["url"] in [c["url"] for c in res]
    n = len(ans)
    hits = [r for r in ranks if r]
    print("\n== Retrieval (answerable questions:", n, ") ==")
    print(f"Recall@5 (exact chunk): {len(hits) / n:.1%}")
    print(f"Recall@5 (same page)  : {page_hits / n:.1%}")
    print(f"MRR                   : {sum(1 / r for r in hits) / n:.3f}")


def ask(question):
    from generate import answer

    for wait in (5, 15, 30, 60):
        try:
            return answer(question)
        except Exception as e:
            print(f"  retry in {wait}s ({type(e).__name__})")
            time.sleep(wait)
    return None


def full_eval():
    from verify import verify_claim

    counts = {"VERIFIED": 0, "UNSUPPORTED": 0, "CONTRADICTED": 0, "INVALID_CITATION": 0}
    decision_ok, refused_ok, total_unans, results = 0, 0, 0, []
    for i, x in enumerate(ITEMS, 1):
        print(f"[{i}/{len(ITEMS)}] {x['question']}")
        out = ask(x["question"])
        if out is None:
            continue
        ok = out["answerable"] == x["answerable"]
        decision_ok += ok
        if not x["answerable"]:
            total_unans += 1
            refused_ok += ok
        claims = []
        for c in out["claims"]:
            r = verify_claim(c["text"], c["sources"], out["sources"])
            counts[r["verdict"]] += 1
            claims.append({"text": c["text"], "sources": c["sources"], **r})
        results.append({"question": x["question"], "expected": x["answerable"],
                        "predicted": out["answerable"], "claims": claims})
        time.sleep(5)  # free tier ki rate limit ke liye

    total = sum(counts.values())
    print("\n== End-to-end ==")
    print(f"Answer/refuse decision accuracy: {decision_ok / len(results):.1%}")
    print(f"Unanswerable correctly refused : {refused_ok}/{total_unans}")
    print(f"Claims checked by NLI          : {total}")
    for k, v in counts.items():
        print(f"  {k:<17}: {v} ({v / max(total, 1):.1%})")
    Path("evals/results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Details saved -> evals/results.json")


if __name__ == "__main__":
    retrieval_eval()
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        full_eval()