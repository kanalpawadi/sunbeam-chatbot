"""LangGraph agent: Groq-hosted LLM + tools over the Sunbeam knowledge base."""
import json
from datetime import datetime, timedelta, timezone

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from shared.config import GROQ_BASE_URL, GROQ_MODEL, require_env

from .store import store

IST = timezone(timedelta(hours=5, minutes=30))


@tool(response_format="content_and_artifact")
def search_sunbeam(query: str) -> tuple[str, list[dict]]:
    """Search Sunbeam Institute's website data (courses, syllabus, batches, fees, schedules,
    internships, branches, contact details, FAQs, about Sunbeam, C-DAC, testimonials).

    Args:
        query: A focused search query, e.g. "Core Java syllabus" or "Hinjawadi branch address".
    """
    hits = store.search(query, k=6)
    if not hits:
        return "No matching Sunbeam data found.", []
    content = json.dumps(
        [{"title": h["title"], "url": h["url"], "content": h["text"]} for h in hits],
        ensure_ascii=False,
    )
    sources = []
    seen = set()
    for h in hits:
        if h["url"] and (h["title"], h["url"]) not in seen:
            seen.add((h["title"], h["url"]))
            sources.append({"title": h["title"], "url": h["url"], "type": h["type"]})
    return content, sources


@tool(response_format="content_and_artifact")
def list_courses(category: str = "") -> tuple[str, list[dict]]:
    """List every Sunbeam course with category, duration, fees and next batch date.
    Use this for "which courses", "all courses", comparisons, or "courses starting soon".

    Args:
        category: Optional filter matched against the category name, e.g. "internship",
            "certified", "entrance", "PG certificate". Leave empty for all courses.
    """
    catalog = store.snapshot().catalog
    term = category.lower().strip()
    if term:
        catalog = [c for c in catalog if term in c["category"].lower() or term in c["name"].lower()] or catalog
    rows = [
        {k: c[k] for k in ("name", "category", "duration", "fees", "next_batch", "mode", "url") if c.get(k)}
        for c in catalog
    ]
    sources = [{"title": "All Sunbeam courses", "url": "https://sunbeaminfo.in/courses", "type": "page"}]
    return json.dumps(rows, ensure_ascii=False), sources


SYSTEM_PROMPT = """You are "Sunbeam Assistant", the friendly official AI assistant of Sunbeam Institute
of Information Technology (Pune & Karad, India; authorized C-DAC training centre).
Today's date is {today}.

How to answer:
- For ANY question about Sunbeam (courses, syllabus, fees, batches, dates, internships, branches,
  contact, placements, C-DAC, FAQs) call a tool first. Use `list_courses` for course lists and
  overviews, and `search_sunbeam` for details. You may call tools more than once.
- Answer ONLY from tool results. Never invent fees, dates, phone numbers or syllabus topics.
  If the data does not contain the answer, say so briefly and share the contact details
  (search for them) so the user can ask the institute directly.
- Batches whose start date is before today have already started; point that out if relevant.
- Greetings and small talk: reply briefly and warmly, then offer help. No tool needed.
- Unrelated questions (general coding help, news, etc.): politely say you can only help with
  Sunbeam Institute and suggest a Sunbeam-related question.

Style:
- Be concise and well structured. Use Markdown: short headings, bullet lists, **bold** key facts,
  and tables when comparing courses or listing batches/fees.
- Fees are in Indian Rupees (₹).
- Finish detailed answers with a link to the relevant course page, as a Markdown link.
"""


def build_agent():
    llm = ChatOpenAI(
        model=GROQ_MODEL,
        base_url=GROQ_BASE_URL,
        api_key=require_env("GROQ_API_KEY"),
        temperature=0.2,
        streaming=True,
    )
    today = datetime.now(IST).strftime("%d %B %Y")
    return create_agent(
        model=llm,
        tools=[search_sunbeam, list_courses],
        system_prompt=SYSTEM_PROMPT.format(today=today),
    )
