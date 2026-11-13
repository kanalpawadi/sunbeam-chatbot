"""Scrape the current Sunbeam website data.

The Sunbeam site (sunbeaminfo.in / sunbeaminfo.com) is a React app that loads all of its
content from a public JSON API. We read the same endpoints the site's own pages call,
turn every record into a clean text document and write:

  data/documents.json  - one text document per course / branch / FAQ / page
  data/catalog.json    - compact course list (name, category, duration, fees, next batch)

Run from the repository root:  python -m pipeline.scrape
"""
import html
import json
import re
import sys
import warnings
from datetime import datetime, timedelta, timezone

import httpx
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

from shared.config import CATALOG_FILE, DATA_DIR, DOCUMENTS_FILE

API = "https://api.sunbeaminfo.in/api/v1"
DOMAINS = {
    "sunbeaminfo.in": "https://sunbeaminfo.in",    # modular / certified courses, internships, PreCAT
    "sunbeaminfo.com": "https://sunbeaminfo.com",  # C-DAC PG certificate programmes
}
IST = timezone(timedelta(hours=5, minutes=30))

client = httpx.Client(
    timeout=30,
    headers={"User-Agent": "SunbeamChatbotScraper/1.0 (+github-actions)"},
    follow_redirects=True,
)


def get(path: str, domain: str, **params) -> dict | list:
    params["domain"] = domain
    for attempt in range(3):
        try:
            resp = client.get(f"{API}/{path}", params=params)
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success", True):
                raise RuntimeError(body.get("message", "API returned success=false"))
            return body["data"]
        except Exception as exc:
            if attempt == 2:
                raise RuntimeError(f"GET {path} ({domain}) failed: {exc}") from exc
    raise AssertionError("unreachable")


# ---------- text helpers ----------

def clean_html(value: str | None) -> str:
    if not value:
        return ""
    soup = BeautifulSoup(value, "html.parser")
    for li in soup.find_all("li"):
        li.insert_before("\n- ")
    for tag in soup.find_all(["p", "br", "div", "h1", "h2", "h3", "h4", "tr"]):
        tag.insert_after("\n")
    text = html.unescape(soup.get_text(" "))
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fmt_date(value: str | None) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(IST)
        return dt.strftime("%d %b %Y")
    except ValueError:
        return value


def fmt_money(fees: dict | None) -> str:
    if not fees or fees.get("amount") in (None, ""):
        return ""
    cur = fees.get("currency", "INR")
    symbol = "₹" if cur == "INR" else f"{cur} "
    return f"{symbol}{int(fees['amount']):,}"


def bullet_list(items) -> str:
    return "\n".join(f"- {clean_html(str(i))}" for i in items if str(i).strip())


def value_to_text(value, indent: int = 0) -> str:
    """Render arbitrary JSON (used for 'about' settings) as readable text."""
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            if k.startswith("_") or v in (None, "", [], {}):
                continue
            label = re.sub(r"(?<!^)(?=[A-Z])", " ", k).capitalize()
            rendered = value_to_text(v, indent + 1)
            if "\n" in rendered or isinstance(v, (dict, list)):
                lines.append(f"{pad}{label}:\n{rendered}")
            else:
                lines.append(f"{pad}{label}: {rendered.strip()}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {value_to_text(v, indent + 1).strip()}" for v in value)
    return pad + clean_html(str(value))


# ---------- document builders ----------

def course_document(c: dict, domain: str) -> tuple[dict, dict]:
    url = f"{DOMAINS[domain]}/courses/{c['slug']}"
    category = (c.get("category") or {}).get("name", "")
    duration = c.get("durationText") or (
        f"{c['duration']['value']} {c['duration']['unit']}" if c.get("duration") else ""
    )
    batches = [b for b in (c.get("activeBatches") or c.get("batches") or []) if b.get("isActive", True)]
    batches.sort(key=lambda b: b.get("startDate") or "")

    parts = [f"# {c['name']}", f"Course page: {url}"]
    if category:
        parts.append(f"Category: {category}")
    if duration:
        parts.append(f"Duration: {duration}")
    if c.get("shortDescription"):
        parts.append(f"\n## Overview\n{clean_html(c['shortDescription'])}")
    if c.get("description"):
        parts.append(f"\n## Description\n{clean_html(c['description'])}")
    if c.get("prerequisites"):
        parts.append(f"\n## Prerequisites\n{bullet_list(c['prerequisites'])}")
    if c.get("learningOutcomes"):
        parts.append(f"\n## What you will learn / highlights\n{bullet_list(c['learningOutcomes'])}")
    if c.get("syllabus"):
        mods = []
        for i, m in enumerate(c["syllabus"], 1):
            topics = ", ".join(t for t in (m.get("topics") or []) if t)
            mods.append(f"{i}. {m.get('module', 'Module')}" + (f": {topics}" if topics else ""))
        parts.append("\n## Syllabus\n" + "\n".join(mods))
    if c.get("instructors"):
        names = [i.get("name", "") if isinstance(i, dict) else str(i) for i in c["instructors"]]
        parts.append(f"\n## Instructors\n{bullet_list(names)}")
    if batches:
        rows = []
        for b in batches:
            seg = [f"Starts {fmt_date(b.get('startDate'))}"]
            if b.get("endDate"):
                seg.append(f"ends {fmt_date(b['endDate'])}")
            if b.get("timing"):
                seg.append(f"batch/timing: {b['timing']}")
            if b.get("mode"):
                seg.append(f"mode: {b['mode']}")
            if fmt_money(b.get("fees")):
                seg.append(f"fees: {fmt_money(b['fees'])}")
            if b.get("seatsAvailable") is not None:
                seg.append(f"seats: {b['seatsAvailable']}")
            rows.append("- " + ", ".join(seg))
        parts.append("\n## Upcoming batches, schedule and fees\n" + "\n".join(rows))
    else:
        parts.append("\n## Upcoming batches, schedule and fees\nNo upcoming batch is listed on the website right now.")
    if c.get("youtubeUrl"):
        parts.append(f"\nIntro video: {c['youtubeUrl']}")

    doc = {
        "id": f"course:{domain}:{c['slug']}",
        "type": "course",
        "title": c["name"],
        "category": category,
        "url": url,
        "text": "\n".join(parts).strip(),
    }
    next_batch = batches[0] if batches else None
    catalog = {
        "name": c["name"],
        "category": category,
        "duration": duration,
        "url": url,
        "summary": clean_html(c.get("shortDescription"))[:240],
        "fees": fmt_money(next_batch.get("fees")) if next_batch else "",
        "next_batch": fmt_date(next_batch.get("startDate")) if next_batch else "",
        "mode": next_batch.get("mode", "") if next_batch else "",
        "upcoming_batches": len(batches),
    }
    return doc, catalog


def branch_document(b: dict, domain: str) -> dict:
    address = ", ".join(x for x in [b.get("address"), b.get("city"), b.get("state"), b.get("pincode")] if x)
    parts = [f"# {b['name']}", f"Address: {address}"]
    phones = b.get("phone") or []
    if phones:
        parts.append("Phone: " + ", ".join(phones if isinstance(phones, list) else [phones]))
    if b.get("email"):
        parts.append(f"Email: {b['email']}")
    if b.get("timings"):
        parts.append("Office timings:\n" + "\n".join(f"- {t['day']}: {t['time']}" for t in b["timings"]))
    if b.get("facilities"):
        parts.append("Facilities:\n" + bullet_list(b["facilities"]))
    return {
        "id": f"branch:{b['_id']}",
        "type": "branch",
        "title": b["name"],
        "category": "Branches / locations",
        "url": f"{DOMAINS[domain]}/branches",
        "text": "\n".join(parts),
    }


def faq_document(f: dict, domain: str) -> dict:
    return {
        "id": f"faq:{f['_id']}",
        "type": "faq",
        "title": clean_html(f["question"]),
        "category": "FAQ",
        "url": f"{DOMAINS[domain]}/faqs",
        "text": f"FAQ: {clean_html(f['question'])}\nAnswer: {clean_html(f.get('answer'))}",
    }


def settings_documents(s: dict, stats: dict, domain: str) -> list[dict]:
    site = DOMAINS[domain]
    docs = []
    if s.get("aboutUs"):
        docs.append({
            "id": "page:about",
            "type": "page",
            "title": "About Sunbeam Institute",
            "category": "About",
            "url": f"{site}/about",
            "text": "# About Sunbeam Institute of Information Technology\n" + value_to_text(s["aboutUs"]),
        })
    if s.get("aboutCdac"):
        docs.append({
            "id": "page:about-cdac",
            "type": "page",
            "title": "About C-DAC",
            "category": "About",
            "url": f"{site}/about",
            "text": "# About C-DAC (Sunbeam is an authorized C-DAC training centre)\n" + value_to_text(s["aboutCdac"]),
        })
    stat_lines = {
        "studentsTrained": "Students trained",
        "recruitingPartners": "Recruiting partners",
        "placementRate": "Placement rate (%)",
        "yearsOfExcellence": "Years of excellence",
    }
    contact = ["# Contact Sunbeam Institute"]
    if s.get("contactEmail"):
        contact.append(f"Email: {s['contactEmail']}")
    for p in s.get("phones") or []:
        contact.append(f"Phone ({p.get('label', 'phone')}): {p.get('number')}")
    for a in s.get("addresses") or []:
        contact.append(f"Address ({a.get('type', '').replace('_', ' ')}): {a.get('address')}")
    for net, link in (s.get("socialMedia") or {}).items():
        if link:
            contact.append(f"{net.capitalize()}: {link}")
    contact.append(f"Websites: {', '.join(DOMAINS.values())}")
    figures = [f"- {label}: {stats[k]}" for k, label in stat_lines.items() if stats.get(k) is not None]
    if figures:
        contact.append("\n## Key figures\n" + "\n".join(figures))
    docs.append({
        "id": "page:contact",
        "type": "page",
        "title": "Contact & key facts",
        "category": "Contact",
        "url": f"{site}/contact",
        "text": "\n".join(contact),
    })
    return docs


def testimonials_document(items: list[dict], domain: str) -> dict | None:
    lines = []
    for t in items[:40]:
        who = ", ".join(x for x in [t.get("studentName"), t.get("company"), t.get("location")] if x)
        content = clean_html(t.get("content"))
        if content:
            lines.append(f"- {who}: {content[:600]}")
    if not lines:
        return None
    return {
        "id": "page:testimonials",
        "type": "page",
        "title": "Student testimonials",
        "category": "Testimonials",
        "url": f"{DOMAINS[domain]}/testimonials",
        "text": "# Student testimonials and reviews\n" + "\n".join(lines),
    }


# ---------- main ----------

def scrape() -> tuple[list[dict], list[dict]]:
    documents: dict[str, dict] = {}
    catalog: list[dict] = []

    for domain in DOMAINS:
        courses = get("courses", domain, limit=200, isActive="true")
        print(f"[{domain}] {len(courses)} courses")
        for summary in courses:
            detail = get(f"courses/{summary['slug']}", domain)
            doc, entry = course_document(detail, domain)
            documents[doc["id"]] = doc
            catalog.append(entry)

        for b in get("branches", domain):
            if b.get("isActive", True):
                doc = branch_document(b, domain)
                documents.setdefault(doc["id"], doc)

        for f in get("faqs", domain, limit=200):
            if f.get("isActive", True):
                doc = faq_document(f, domain)
                documents.setdefault(doc["id"], doc)

        if domain == "sunbeaminfo.in":
            settings = get("settings/public", domain)
            stats = get("settings/stats", domain)
            for doc in settings_documents(settings, stats, domain):
                documents[doc["id"]] = doc
            t_doc = testimonials_document(get("testimonials", domain, limit=50), domain)
            if t_doc:
                documents[t_doc["id"]] = t_doc

    # Overview of all courses, so "which courses do you offer?" retrieves one complete list.
    by_cat: dict[str, list[str]] = {}
    for e in catalog:
        by_cat.setdefault(e["category"] or "Other", []).append(
            f"- {e['name']} ({e['duration'] or 'duration n/a'})"
            + (f", next batch {e['next_batch']}, fees {e['fees']}" if e["next_batch"] else "")
        )
    overview = "# All courses offered by Sunbeam\n" + "\n".join(
        f"\n## {cat}\n" + "\n".join(items) for cat, items in by_cat.items()
    )
    documents["page:all-courses"] = {
        "id": "page:all-courses",
        "type": "page",
        "title": "All Sunbeam courses",
        "category": "Courses",
        "url": f"{DOMAINS['sunbeaminfo.in']}/courses",
        "text": overview,
    }

    return list(documents.values()), catalog


def main() -> int:
    try:
        documents, catalog = scrape()
    except Exception as exc:
        # Never overwrite good data with a partial scrape.
        print(f"Scrape failed, existing data left untouched: {exc}", file=sys.stderr)
        return 1
    if len(catalog) < 5:
        print(f"Only {len(catalog)} courses found - refusing to overwrite data.", file=sys.stderr)
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents.sort(key=lambda d: d["id"])
    catalog.sort(key=lambda e: (e["category"], e["name"]))
    DOCUMENTS_FILE.write_text(json.dumps(documents, ensure_ascii=False, indent=1), encoding="utf-8")
    CATALOG_FILE.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    counts: dict[str, int] = {}
    for d in documents:
        counts[d["type"]] = counts.get(d["type"], 0) + 1
    print(f"Wrote {len(documents)} documents {counts} and {len(catalog)} catalog entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
