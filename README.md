# ☀️ Sunbeam Assistant

An AI chatbot for [Sunbeam Institute of Information Technology](https://sunbeaminfo.in) (Pune & Karad).
Ask about courses, syllabus, batch dates, fees, internships, C-DAC PG programmes, branches and
contact details. Answers come from Sunbeam's live website, refreshed automatically every day.

**Live demo:** <https://kanalpawadi.github.io/sunbeam-chatbot/>
&nbsp;·&nbsp; **API:** <https://sunbeam-chatbot-api.onrender.com/api/info>

> The API runs on Render's free plan and sleeps when idle, so the first question after a quiet
> spell can take up to a minute. The app tells you while it wakes up.

## Features

- **Streaming answers** with Markdown tables for fees, batches and comparisons
- **Source links** under every answer, pointing to the exact Sunbeam pages used
- **Always current**: a GitHub Action scrapes the website daily and refreshes the knowledge base
- **Grounded**: the agent answers only from Sunbeam data, says when something isn't listed, and
  politely declines unrelated questions
- **Chat history** saved in the browser, new chat (`Ctrl+K`), stop, retry and copy
- **Light and dark themes**, and a layout made for phones as well as desktops

## How it works

```
sunbeaminfo.in / .com ──► pipeline/scrape.py ──► data/documents.json, data/catalog.json
   (public JSON API)       pipeline/ingest.py ──► data/knowledge.json   (Gemini embeddings)
                                                          │
React (GitHub Pages) ◄── streamed answer ── FastAPI (Render) ◄┘  in-memory ChromaDB
                                            LangGraph agent + Groq LLM
                                            tools: search_sunbeam, list_courses
```

1. **Scrape**: the Sunbeam website loads its content from a public JSON API. The scraper reads
   the same endpoints (courses, batches, fees, syllabus, branches, FAQs, about, contact) and
   turns each record into a clean text document.
2. **Embed**: documents are split into chunks and embedded with Google Gemini. Only chunks whose
   text changed are re-embedded, so daily runs stay within the free tier.
3. **Answer**: a LangGraph agent on Groq decides which tool to call. `search_sunbeam` does a
   vector search in ChromaDB; `list_courses` returns the full catalog with fees and next batch.
   The reply streams to the browser as Server-Sent Events.

| Part | Tech |
| --- | --- |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS v4, react-markdown |
| Backend | FastAPI, LangGraph / LangChain agent, ChromaDB |
| LLM | Groq `openai/gpt-oss-20b` |
| Embeddings | Google Gemini `gemini-embedding-001` (768 dims) |
| Hosting | GitHub Pages (UI) + Render (API) |
| Automation | GitHub Actions: daily data refresh, Pages deploy, CI |

## Project structure

```
backend/      FastAPI app (main.py), LangGraph agent and tools (agent.py), vector store (store.py)
frontend/     React app (src/App.tsx, src/components.tsx, src/api.ts)
pipeline/     scrape.py (website -> documents), ingest.py (chunks -> Gemini embeddings)
shared/       config and Gemini embedding helper used by backend and pipeline
data/         scraped documents, course catalog and knowledge base (updated by CI)
render.yaml   Render blueprint for the API
.github/      workflows: refresh-knowledge, deploy-pages, ci
```

## Run locally

Requirements: Python 3.11, Node 22, a [Groq API key](https://console.groq.com/keys) and a
[Gemini API key](https://aistudio.google.com/apikey).

```bash
cp .env.example .env            # add GROQ_API_KEY and GOOGLE_API_KEY
pip install -r requirements.txt
cd frontend && npm install && cd ..

python -m pipeline.scrape       # fetch the current website data
python -m pipeline.ingest       # chunk + embed

uvicorn backend.main:app --reload --port 8000     # API, from the repo root
cd frontend && npm run dev                         # UI on http://localhost:5173
```

To serve the UI and API from one server, run `npm run build` in `frontend/`, then start uvicorn;
it serves the built app on <http://localhost:8000>.

## Deploy

GitHub Pages only hosts static files, so the UI goes there and the Python API (which keeps the API
keys secret) runs on Render. Both are free.

1. **API on Render**: sign in at <https://render.com> with GitHub → **New → Blueprint** → pick
   this repository. Render reads `render.yaml`; paste `GROQ_API_KEY` and `GOOGLE_API_KEY` when
   asked.
2. **UI on GitHub Pages**: **Settings → Pages → Source: GitHub Actions**, then run **Deploy
   frontend to GitHub Pages** from the Actions tab. If your Render URL differs from
   `https://sunbeam-chatbot-api.onrender.com`, set it as the Actions variable `API_URL`.
3. **Daily refresh**: add `GOOGLE_API_KEY` under **Settings → Secrets and variables → Actions**.

The API only accepts browser requests from the origins in `ALLOWED_ORIGINS` (set in
`render.yaml`).

## GitHub Actions

| Workflow | When | What it does |
| --- | --- | --- |
| Refresh Sunbeam knowledge base | Daily 06:00 IST, or manually | Scrapes the site, re-embeds changed text, commits `data/` if anything changed; Render then redeploys with the fresh data |
| Deploy frontend to GitHub Pages | Push to `frontend/`, or manually | Builds the React app and publishes it |
| CI | Every push and pull request | Builds the frontend, import-checks the backend, smoke-tests the scraper |

## Developer

Developed by **Kanal Pawadi** · [github.com/kanalpawadi](https://github.com/kanalpawadi)

Course data belongs to Sunbeam Institute; always confirm fees and dates with the institute.
