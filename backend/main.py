# backend/main.py

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader

# new OpenAI client
from openai import OpenAI
from uuid import uuid4

# Simple in‑memory store: unit_id → unit dict
UNIT_STORE: dict[str, dict] = {}
# ─── Setup ─────────────────────────────────────────────────────────────────────

# load .env and API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# instantiate OpenAI client
client = OpenAI(api_key=api_key)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create FastAPI app
app = FastAPI()

# compute absolute paths
BASE_DIR      = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR    = BASE_DIR / "static"

# mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# configure Jinja2
jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

# ─── Helper to render templates ────────────────────────────────────────────────

def render_template(name: str, **context):
    tmpl = jinja_env.get_template(name)
    return HTMLResponse(tmpl.render(**context))

# ─── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return jinja_env.get_template("login.html").render(request=request)

@app.post("/login")
async def login_submit(email: str = Form(...), password: str = Form(...)):
    # placeholder auth
    if email == "teacher@example.com" and password == "password":
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    return RedirectResponse(url="/?error=1", status_code=status.HTTP_302_FOUND)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return jinja_env.get_template("dashboard.html").render(request=request)

@app.get("/unit/new", response_class=HTMLResponse)
async def unit_form(request: Request):
    return jinja_env.get_template("unit_form.html").render(request=request)

@app.post("/generate")
async def generate_unit(
    topic: str = Form(...),
    grade: int = Form(...),
    standard: str = Form(...),
):
    # 1) Build and call the AI prompt as before
    prompt = (
        f"Create a 4-lesson unit plan on “{topic}” for Grade {grade} "
        f"aligned to CCSS standard {standard}. Output only valid JSON…"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    content = resp.choices[0].message.content
    try:
        unit = json.loads(content)
    except:
        unit = {"error": "Invalid JSON", "raw": content}

    # 2) Store it under a new UUID
    unit_id = str(uuid4())
    UNIT_STORE[unit_id] = unit

    # 3) Redirect to the GET results page
    return RedirectResponse(url=f"/unit/{unit_id}", status_code=status.HTTP_302_FOUND)


@app.get("/unit/{unit_id}", response_class=HTMLResponse)
async def show_unit(request: Request, unit_id: str):
    unit = UNIT_STORE.get(unit_id)
    if not unit:
        return HTMLResponse("<h1>Unit not found</h1>", status_code=404)
    return jinja_env.get_template("unit_result.html").render(request=request, unit=unit)
