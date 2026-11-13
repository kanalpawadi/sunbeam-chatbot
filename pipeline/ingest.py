"""Chunk the scraped documents and embed them with Google Gemini.

Writes data/knowledge.json (chunks + embeddings). Chunks whose text did not change keep
their previous embedding, so a daily refresh only spends API calls on new or changed text.
The file is rewritten only when the content actually changed, so the scheduled GitHub
Action commits nothing on days the website is unchanged.

Run from the repository root:  python -m pipeline.ingest
"""
import hashlib
import json
from datetime import datetime, timezone

from shared.config import DOCUMENTS_FILE, EMBED_DIM, EMBED_MODEL, KNOWLEDGE_FILE
from shared.gemini import embed_documents

MAX_CHARS = 1400


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_document(doc: dict) -> list[str]:
    """Split on '## ' sections, pack sections up to MAX_CHARS, and prefix every chunk
    with the document title so each chunk is self-explanatory on its own."""
    text = doc["text"]
    header = f"{doc['title']} ({doc['category']})" if doc.get("category") else doc["title"]
    sections = [s.strip() for s in text.split("\n## ") if s.strip()]
    sections = [sections[0]] + [f"## {s}" for s in sections[1:]]

    pieces: list[str] = []
    for sec in sections:
        while len(sec) > MAX_CHARS:  # very long section: split on line boundaries
            cut = sec.rfind("\n", 0, MAX_CHARS)
            cut = cut if cut > MAX_CHARS // 3 else MAX_CHARS
            pieces.append(sec[:cut].strip())
            sec = sec[cut:].strip()
        pieces.append(sec)

    chunks: list[str] = []
    current = ""
    for p in pieces:
        if current and len(current) + len(p) + 2 > MAX_CHARS:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)

    return [c if c.startswith(f"# {doc['title']}") else f"[{header}]\n{c}" for c in chunks]


def main() -> int:
    documents = json.loads(DOCUMENTS_FILE.read_text(encoding="utf-8"))

    previous: dict[str, list[float]] = {}
    old_hash = None
    if KNOWLEDGE_FILE.exists():
        old = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
        old_hash = old.get("content_hash")
        if old.get("embed_model") == EMBED_MODEL and old.get("embed_dim") == EMBED_DIM:
            previous = {c["hash"]: c["embedding"] for c in old.get("chunks", [])}

    chunks = []
    for doc in documents:
        parts = chunk_document(doc)
        for i, text in enumerate(parts):
            chunks.append({
                "id": f"{doc['id']}#{i}",
                "doc_id": doc["id"],
                "title": doc["title"],
                "type": doc["type"],
                "category": doc.get("category", ""),
                "url": doc.get("url", ""),
                "text": text,
                "hash": sha(text),
            })

    content_hash = sha("\n".join(c["hash"] for c in chunks) + EMBED_MODEL + str(EMBED_DIM))
    if content_hash == old_hash:
        print(f"No content changes ({len(chunks)} chunks) - knowledge.json left as is.")
        return 0

    todo = [c for c in chunks if c["hash"] not in previous]
    print(f"{len(chunks)} chunks: {len(chunks) - len(todo)} reused, {len(todo)} to embed with {EMBED_MODEL}")
    if todo:
        vectors = embed_documents([c["text"] for c in todo])
        for c, v in zip(todo, vectors):
            previous[c["hash"]] = v
    for c in chunks:
        c["embedding"] = [round(x, 6) for x in previous[c["hash"]]]

    KNOWLEDGE_FILE.write_text(json.dumps({
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "content_hash": content_hash,
        "document_count": len(documents),
        "chunks": chunks,
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {KNOWLEDGE_FILE.name} with {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
