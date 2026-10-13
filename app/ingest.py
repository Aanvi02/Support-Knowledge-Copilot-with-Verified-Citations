import json
import re
from pathlib import Path

ROOT = Path("data/raw/handbook_repo/content")
SRC = ROOT / "handbook" / "people-group"
OUT = Path("data/chunks.json")
MIN_CHARS = 200
MAX_CHARS = 1500


def split_front_matter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.S)
    if not m:
        return "", text
    t = re.search(r'^title:\s*["\']?(.*?)["\']?\s*$', m.group(1), re.M)
    return (t.group(1) if t else ""), text[m.end():]


def clean(body):
    body = re.sub(r"\{\{[<%].*?[>%]\}\}", "", body, flags=re.S)  # Hugo shortcodes
    return re.sub(r"<!--.*?-->", "", body, flags=re.S)  # HTML comments


def split_sections(body):
    sections, heading, buf, in_code = [], "", [], False
    for line in body.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,4}\s+", line):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading = re.sub(r"^#{1,4}\s+", "", line).strip()
            heading = re.sub(r"\s*\{#.*?\}\s*$", "", heading)
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, t) for h, t in sections if t]


def split_long(text):
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for para in text.split("\n\n"):
        if cur and len(cur) + len(para) > MAX_CHARS:
            parts.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def to_url(path):
    rel = path.relative_to(ROOT).with_suffix("")
    parts = [p for p in rel.parts if p != "_index"]
    return "https://handbook.gitlab.com/" + "/".join(parts) + "/"


def main():
    chunks = []
    for path in sorted(SRC.rglob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        title, body = split_front_matter(raw)
        file_chunks = []
        for heading, text in split_sections(clean(body)):
            for part in split_long(text):
                if file_chunks and len(part) < MIN_CHARS:
                    file_chunks[-1]["text"] += "\n\n" + part
                else:
                    file_chunks.append(
                        {"title": title, "heading": heading, "text": part, "url": to_url(path)}
                    )
        chunks.extend(file_chunks)

    for i, c in enumerate(chunks):
        c["id"] = i
    OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(chunks)} chunks saved to {OUT}")
    if chunks:
        print("\nSample chunk:\n", json.dumps(chunks[len(chunks) // 2], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()