"""Google Gemini embeddings (online) for documents and queries."""
import math
import time
from functools import lru_cache

from google import genai
from google.genai import types

from .config import EMBED_DIM, EMBED_MODEL, require_env

BATCH_SIZE = 50


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    return genai.Client(api_key=require_env("GOOGLE_API_KEY"))


def _normalize(vec: list[float]) -> list[float]:
    # Gemini only normalizes the full 3072-dim output; truncated vectors must be normalized here.
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embed(texts: list[str], task_type: str) -> list[list[float]]:
    out: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(5):
            try:
                resp = _client().models.embed_content(
                    model=EMBED_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=task_type, output_dimensionality=EMBED_DIM
                    ),
                )
                out.extend(_normalize(e.values) for e in resp.embeddings)
                break
            except Exception as exc:  # rate limits on the free tier: back off and retry
                if attempt == 4:
                    raise
                wait = 2 ** attempt * 5
                print(f"  embed retry in {wait}s ({exc.__class__.__name__}: {exc})")
                time.sleep(wait)
    return out


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> list[float]:
    return _embed([text], "RETRIEVAL_QUERY")[0]
