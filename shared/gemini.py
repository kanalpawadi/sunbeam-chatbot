"""Google Gemini embeddings (online) for documents and queries."""
import math
import os
import re
import time
from collections import deque

from google import genai
from google.genai import errors, types

from .config import EMBED_DIM, EMBED_MODEL, require_env

BATCH_SIZE = 45
# Free tier counts every text as a request: 100 per minute. Stay a little under it.
TEXTS_PER_MINUTE = int(os.getenv("EMBED_TEXTS_PER_MINUTE", "90"))

_client: genai.Client | None = None
_client_key: str | None = None
_sent: deque[tuple[float, int]] = deque()  # (timestamp, texts) in the last minute


def _get_client() -> genai.Client:
    global _client, _client_key
    key = require_env("GOOGLE_API_KEY")
    if _client is None or key != _client_key:
        _client, _client_key = genai.Client(api_key=key), key
    return _client


def _normalize(vec: list[float]) -> list[float]:
    # Gemini only normalizes the full 3072-dim output; truncated vectors must be normalized here.
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _pace(count: int) -> None:
    """Block until sending `count` more texts keeps us under TEXTS_PER_MINUTE."""
    while True:
        now = time.monotonic()
        while _sent and now - _sent[0][0] > 60:
            _sent.popleft()
        used = sum(n for _, n in _sent)
        if used + count <= TEXTS_PER_MINUTE or not _sent:
            _sent.append((now, count))
            return
        wait = 60 - (now - _sent[0][0]) + 0.5
        print(f"  pacing for free-tier quota: waiting {wait:.0f}s")
        time.sleep(wait)


def _retry_delay(exc: Exception) -> float | None:
    """Seconds to wait for a retryable error, or None if retrying won't help."""
    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            m = re.search(r"retry in ([\d.]+)s", str(exc))
            return float(m.group(1)) + 1 if m else 30.0
        if exc.code and exc.code >= 500:
            return 5.0
        return None  # 400/401/403: bad key or request - fail fast
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return 5.0
    return None


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    out: list[list[float]] = []
    client = _get_client()
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(6):
            _pace(len(batch))
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
                )
                out.extend(_normalize(e.values) for e in resp.embeddings)
                break
            except Exception as exc:
                wait = _retry_delay(exc)
                if wait is None or attempt == 5:
                    raise
                print(f"  Gemini {getattr(exc, 'code', type(exc).__name__)}: retrying in {wait:.0f}s")
                time.sleep(wait)
    return out


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    return _embed([text], "RETRIEVAL_QUERY")[0]
