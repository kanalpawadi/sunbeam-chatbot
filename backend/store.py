"""In-memory ChromaDB index built from data/knowledge.json.

The index is rebuilt automatically when knowledge.json changes on disk (for example after
`git pull` brings in a fresh scrape from the GitHub Action), so no restart is needed.
"""
import json
import threading
import uuid
from dataclasses import dataclass, field

import chromadb

from shared.config import CATALOG_FILE, EMBED_DIM, EMBED_MODEL, KNOWLEDGE_FILE
from shared.gemini import embed_query


@dataclass
class Snapshot:
    collection: chromadb.Collection | None = None
    catalog: list[dict] = field(default_factory=list)
    updated_at: str | None = None
    chunk_count: int = 0
    document_count: int = 0
    mtime: float = -1.0


class KnowledgeStore:
    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._lock = threading.Lock()
        self._snap = Snapshot()

    def _mtime(self) -> float:
        try:
            return KNOWLEDGE_FILE.stat().st_mtime + (CATALOG_FILE.stat().st_mtime if CATALOG_FILE.exists() else 0)
        except FileNotFoundError:
            return -1.0

    def snapshot(self) -> Snapshot:
        mtime = self._mtime()
        if mtime != self._snap.mtime:
            with self._lock:
                if mtime != self._snap.mtime:
                    self._snap = self._load(mtime)
        return self._snap

    def _load(self, mtime: float) -> Snapshot:
        if mtime < 0:
            return Snapshot(mtime=mtime)
        data = json.loads(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
        if data.get("embed_model") != EMBED_MODEL or data.get("embed_dim") != EMBED_DIM:
            raise RuntimeError(
                f"knowledge.json was built with {data.get('embed_model')}/{data.get('embed_dim')} "
                f"but the backend is configured for {EMBED_MODEL}/{EMBED_DIM}. Re-run the pipeline."
            )
        old = self._snap.collection
        col = self._client.create_collection(
            name=f"sunbeam_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "cosine"}
        )
        chunks = data["chunks"]
        for start in range(0, len(chunks), 500):
            batch = chunks[start:start + 500]
            col.add(
                ids=[c["id"] for c in batch],
                embeddings=[c["embedding"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {"title": c["title"], "url": c["url"], "type": c["type"], "category": c["category"]}
                    for c in batch
                ],
            )
        if old is not None:
            try:
                self._client.delete_collection(old.name)
            except Exception:
                pass
        catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8")) if CATALOG_FILE.exists() else []
        print(f"[store] loaded {len(chunks)} chunks, {len(catalog)} courses (updated {data.get('updated_at')})")
        return Snapshot(
            collection=col,
            catalog=catalog,
            updated_at=data.get("updated_at"),
            chunk_count=len(chunks),
            document_count=data.get("document_count", 0),
            mtime=mtime,
        )

    def search(self, query: str, k: int = 6) -> list[dict]:
        snap = self.snapshot()
        if snap.collection is None:
            return []
        res = snap.collection.query(query_embeddings=[embed_query(query)], n_results=k)
        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({**meta, "text": doc, "score": round(1 - dist, 3)})
        return hits


store = KnowledgeStore()
