import json
import logging
import os
from math import ceil

import uvicorn

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from config import UNIT_DEFAULTS
from utils import clean_and_extract_json

# ─ Setup ────────────────────────────────────────────────────────────────────────

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
jinja_env = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")))

# In‑memory store
UNIT_STORE: dict[str, dict] = {}


# ─ Routes ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return jinja_env.get_template("login.html").render(request=request)


@app.get("/unit/new", response_class=HTMLResponse)
async def unit_form(request: Request):
    return jinja_env.get_template("unit_form.html").render(request=request)


@app.get("/unit/{unit_id}", response_class=HTMLResponse)
async def show_unit(request: Request, unit_id: str):
    unit = UNIT_STORE.get(unit_id)
    lessons = unit["summary"]
    categories = list(UNIT_DEFAULTS["lesson_mix"].keys())

    # DEBUG log
    logger.info("Unit %s lessons: %s", unit_id,
                [(l['__idx'], l['lesson_type']) for l in lessons])

    return jinja_env.get_template("unit_result.html").render(
        request=request, unit_id=unit_id, lessons=lessons, categories=categories
    )


@app.post("/unit/{unit_id}/lesson/add", response_class=HTMLResponse)
async def add_lesson(request: Request, unit_id: str, category: str = Form(...)):
    store = UNIT_STORE.get(unit_id)
    if not store:
        return HTMLResponse("Unit not found", status_code=404)
    meta = store["meta"]
    # compute next index
    existing = store["summary"] + list(store["details"].values())
    idx = max([l["__idx"] for l in existing], default=-1) + 1

    # strict prompt
    prompt = (
      f"Generate exactly 1 new {category} lesson summary (idx {idx}) "
      f"for Grade {meta['grade']} on topic “{meta['topic']}” in genre “{meta['genre']}”.\n"
      "Output only valid JSON with top-level key \"lessons\":\n"
      "[{ __idx (integer), lesson_type (string), title (string), objective (string) }]."
    )
    resp = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[{"role":"user","content":prompt}],
      temperature=0.2
    )
    raw = resp.choices[0].message.content
    logger.info("Add-lesson raw response: %s", raw)

    try:
        data = clean_and_extract_json(raw)
        lessons = data.get("lessons")
        if not isinstance(lessons, list) or not lessons:
            raise ValueError(f"`lessons` missing or empty in response: {data}")
        lesson = lessons[0]
        lesson["__idx"] = idx
        store["summary"].append(lesson)

        # Render the single-row partial
        return jinja_env.get_template("_lesson_row.html").render(
            unit_id=unit_id, lesson=lesson
        )

    except Exception as e:
        logger.exception("Failed to parse add_lesson response")
        # Show user a minimal error row
        return HTMLResponse(
          content=f"<tr><td colspan='3' class='text-red-500'>Error generating lesson: {e}</td></tr>",
          status_code=200
        )

@app.get("/ping")
async def ping():
    return {"ping": "pong"}


@app.post("/generate-summary")
async def generate_summary(
        topic: str = Form(...),
        grade: int = Form(...),
        genre: str = Form(...)
):
    try:
        # fixed set of categories from your config
        cats = list(UNIT_DEFAULTS["lesson_mix"].keys())
        summary = []

        for i, cat in enumerate(cats):
            prompt = (
                f"Generate 1 {cat} lesson summary for Grade {grade} on topic “{topic}” "
                f"in genre “{genre}”. Output JSON with top‑level key \"lessons\": "
                "[{ __idx (integer), lesson_type (string), title (string), objective (string) }]. "
                "Output only valid JSON."
            )
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            raw = resp.choices[0].message.content
            logger.info("Summary raw for category %s: %s", cat, raw)

            # parse and validate
            data = clean_and_extract_json(raw)
            lessons = data.get("lessons")
            if not isinstance(lessons, list) or not lessons:
                raise ValueError(f"No lessons array in AI response for {cat}: {data}")

            # take the first lesson and enforce our category
            lesson = lessons[0]
            lesson["lesson_type"] = cat
            lesson["__idx"] = i

            summary.append(lesson)

        # store summary and meta for on‑demand adds
        unit_id = str(uuid4())
        UNIT_STORE[unit_id] = {
            "summary": summary,
            "details": {},
            "meta": {"topic": topic, "grade": grade, "genre": genre}
        }

        # redirect to the overview page
        return RedirectResponse(url=f"/unit/{unit_id}", status_code=status.HTTP_302_FOUND)

    except Exception as e:
        logger.exception("generate-summary failed")
        return HTMLResponse(
            content=f"<h1>Generation Error</h1><pre>{e}</pre>",
            status_code=500
        )

@app.get("/unit/{unit_id}/week/{w}/lesson/{idx}", response_class=HTMLResponse)
async def lesson_detail_on_demand(request: Request, unit_id: str, w: int, idx: int):
    store = UNIT_STORE.get(unit_id)
    if not store:
        return HTMLResponse("<h1>Unit not found</h1>", status_code=404)

    # summary and details cache
    summary = store["summary"]
    details = store.setdefault("details", {})

    key = f"lesson{idx}"
    # if not generated yet, create full detail from summary
    if key not in details:
        # find the summary entry
        lesson_summary = next((l for l in summary if l["__idx"] == idx), None)
        if not lesson_summary:
            return HTMLResponse("<h1>Lesson summary not found</h1>", status_code=404)

        # build prompt to expand summary into full lesson
        prompt = (
          f"Expand this lesson summary into a full lesson plan:\n"
          f"{lesson_summary}\n"
          "Include:\n"
          "- All applicable standards\n"
          "- A clear learning objective\n"
          "- Differentiated success_criteria (emerging/proficient/advanced)\n"
          "- At least one ELL strategy\n"
          "- A list of activities\n"
          "- An assessment description\n\n"
          "Output only valid JSON with key \"lesson\" containing these fields."
        )
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2
        )
        raw = resp.choices[0].message.content
        logger.info("Detail raw for lesson %s: %s", idx, raw)
        data = clean_and_extract_json(raw)
        detail = data.get("lesson") or data.get("lessons", [None])[0]
        if not detail:
            return HTMLResponse("<h1>Invalid detail response</h1>", status_code=500)
        # carry over type, idx
        detail["lesson_type"] = lesson_summary["lesson_type"]
        detail["__idx"] = idx
        details[key] = detail

    # render detail via template
    lesson = details[key]
    return jinja_env.get_template("lesson_detail.html").render(
        request=request,
        lesson=lesson,
        unit_id=unit_id,
        idx=idx
    )


@app.post("/unit/{unit_id}/lesson/add", response_class=HTMLResponse)
async def add_lesson(request: Request, unit_id: str, category: str = Form(...)):
    store = UNIT_STORE.get(unit_id)
    if not store: return HTMLResponse("Unit not found", 404)
    meta = store["meta"]
    # next index
    existing = store["summary"] + list(store["details"].values())
    idx = max([l["__idx"] for l in existing], default=-1) + 1

    prompt = (
        f"Generate 1 {category} lesson summary (idx {idx}) for Grade {meta['grade']} "
        f"on topic “{meta['topic']}” in genre “{meta['genre']}”. "
        "Output JSON with __idx, lesson_type, title, objective."
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    lesson = clean_and_extract_json(resp.choices[0].message.content)["lessons"][0]
    lesson["__idx"] = idx
    store["summary"].append(lesson)

    # render just the row fragment
    return jinja_env.get_template("_lesson_row.html").render(
        unit_id=unit_id, lesson=lesson
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
