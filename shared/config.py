"""Paths and settings shared by the pipeline and the backend."""
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

CHATBOT_DIR = Path(__file__).resolve().parent.parent  # repository root

# Real environment variables win over values in .env.
load_dotenv(CHATBOT_DIR / ".env")

DATA_DIR = CHATBOT_DIR / "data"
DOCUMENTS_FILE = DATA_DIR / "documents.json"   # cleaned text documents from the scraper
CATALOG_FILE = DATA_DIR / "catalog.json"       # compact course list used by the list_courses tool
KNOWLEDGE_FILE = DATA_DIR / "knowledge.json"   # chunks + Gemini embeddings used by the backend

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        # The key may have been added to .env after the process started.
        value = (dotenv_values(CHATBOT_DIR / ".env").get(name) or "").strip()
        if value:
            os.environ[name] = value
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to .env (local) or to the repository's "
            f"GitHub Actions secrets (CI)."
        )
    return value
