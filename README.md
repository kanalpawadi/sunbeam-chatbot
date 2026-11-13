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

## Deploy (GitHub Pages + Render, both free)

GitHub Pages can only host static files, so the UI goes there and the Python API (which keeps
the API keys secret) runs on Render.

1. **Backend on Render**: sign in at <https://render.com> with GitHub → **New → Blueprint** →
   pick this repository. Render reads `render.yaml`; paste `GROQ_API_KEY` and `GOOGLE_API_KEY`
   when asked. Your API URL looks like `https://sunbeam-chatbot-api.onrender.com`; open
   `/api/info` on it to check it is up.
2. **Frontend on GitHub Pages**: repository **Settings → Pages → Source: GitHub Actions**.
   If Render gave your service a different URL, add it under **Settings → Secrets and variables
   → Actions → Variables** as `API_URL`.
3. Run **Deploy frontend to GitHub Pages** from the Actions tab (it also runs on every push that
   changes `frontend/`). The site is published at `https://<owner>.github.io/sunbeam-chatbot/`.

Render's free plan sleeps after 15 minutes without traffic; the first message after that takes
up to a minute while it wakes, and the UI says so. The daily data refresh commits to `data/`,
which makes Render redeploy with the fresh knowledge base.

## GitHub Actions

- **Refresh Sunbeam knowledge base** (`.github/workflows/refresh-knowledge.yml`): daily at
  06:00 IST, or on demand from the Actions tab. Scrapes the site, re-embeds changed text and
  commits `data/` only when something changed. Needs the repository secret `GOOGLE_API_KEY`
  (Settings → Secrets and variables → Actions).
- **Deploy frontend to GitHub Pages** (`.github/workflows/deploy-pages.yml`): builds the UI
  with the backend URL and publishes it.
- **CI** (`.github/workflows/ci.yml`): builds the frontend and import-checks the backend on
  every push and pull request.

The backend reloads `data/knowledge.json` when the file changes, so a `git pull` on the server
is enough to serve fresh data.
