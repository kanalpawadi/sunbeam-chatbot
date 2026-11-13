# Sunbeam Assistant

A RAG chatbot for [Sunbeam Institute](https://sunbeaminfo.in): a React chat UI, a FastAPI +
LangGraph agent, and a daily GitHub Action that scrapes the live website and refreshes the
knowledge base.

```
sunbeaminfo.in / .com ──► pipeline/scrape.py ──► data/documents.json, data/catalog.json
   (public JSON API)       pipeline/ingest.py ──► data/knowledge.json   (Gemini embeddings)
                                                          │
React (frontend/) ◄── streamed answer ── FastAPI (backend/) ◄┘  in-memory ChromaDB
                                         LangGraph agent + Groq LLM
                                         tools: search_sunbeam, list_courses
```

| Part | Tech |
| --- | --- |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS v4, react-markdown |
| Backend | FastAPI, LangGraph / LangChain agent, ChromaDB |
| LLM | Groq `openai/gpt-oss-20b` |
| Embeddings | Google Gemini `gemini-embedding-001` (768 dims) |
| Data refresh | GitHub Actions, daily |

## Setup

```bash
cp .env.example .env            # add GROQ_API_KEY and GOOGLE_API_KEY
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

Keys: Groq at <https://console.groq.com/keys>, Gemini at <https://aistudio.google.com/apikey>.

## Build the knowledge base

```bash
python -m pipeline.scrape       # fetch the current website data
python -m pipeline.ingest       # chunk + embed (only changed chunks are re-embedded)
```

## Run (development)

```bash
uvicorn backend.main:app --reload --port 8000     # from the repo root
cd frontend && npm run dev                         # http://localhost:5173
```

## Run (single server)

```bash
cd frontend && npm run build && cd ..
uvicorn backend.main:app --port 8000              # UI + API on http://localhost:8000
```

## GitHub Actions

- **Refresh Sunbeam knowledge base** (`.github/workflows/refresh-knowledge.yml`): daily at
  06:00 IST, or on demand from the Actions tab. Scrapes the site, re-embeds changed text and
  commits `data/` only when something changed. Needs the repository secret `GOOGLE_API_KEY`
  (Settings → Secrets and variables → Actions).
- **CI** (`.github/workflows/ci.yml`): builds the frontend and import-checks the backend on
  every push and pull request.

The backend reloads `data/knowledge.json` when the file changes, so a `git pull` on the server
is enough to serve fresh data.
