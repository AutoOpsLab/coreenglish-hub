import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, status, Depends, HTTPException, Security
from fastapi.openapi.models import Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from sqlmodel import Session, select
from starlette.middleware.sessions import SessionMiddleware

from backend.auth import decode_access_token, create_access_token, SECRET_KEY
from backend.security import hash_password, verify_password
from backend.db import init_db, engine, Unit, Lesson, User
from config import UNIT_DEFAULTS
from utils import clean_and_extract_json

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ─ Setup ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # ← add app parameter here
    init_db()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
jinja_env = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")))

oauth2_optional = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)


async def get_current_user_optional(token: str = Security(oauth2_optional)):
    if not token:
        return None
    from auth import decode_access_token
    email = decode_access_token(token)
    if not email:
        return None
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).one_or_none()
    return user


def render_template(template_name: str, **context) -> str:
    """
    Render a Jinja template with the given context,
    automatically injecting `current_user` (which may be None).
    Returns the rendered HTML string.
    """
    # Ensure every template sees a current_user key (None if not provided)
    context.setdefault("current_user", None)
    # Render via Jinja
    return jinja_env.get_template(template_name).render(**context)


async def get_current_user(request: Request) -> User:
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    email = decode_access_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")

    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).one_or_none()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user


# ─ Routes ───────────────────────────────────────────────────────────────────────
@app.post("/signup")
async def signup(form: OAuth2PasswordRequestForm = Depends()):
    with Session(engine) as session:
        exists = session.exec(select(User).where(User.email == form.username)).first()
        if exists:
            raise HTTPException(status_code=400, detail="Email already registered")
        user = User(email=form.username, hashed_password=hash_password(form.password))
        session.add(user)
        session.commit()
        session.refresh(user)
    token = create_access_token(user.email)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return render_template("login.html", request=request)


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == username)).one_or_none()
    if not user or not verify_password(password, user.hashed_password):
        return render_template("login.html", request=request, error="Invalid credentials")

    token = create_access_token(user.email)
    response = RedirectResponse(url="/my-units", status_code=302)
    response.set_cookie("access_token", token, httponly=True, max_age=3600)
    return response


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request, current_user=Depends(get_current_user_optional)):
    return render_template("home.html", request=request, current_user=current_user)


@app.get("/unit/new", response_class=HTMLResponse)
async def unit_form(request: Request, current_user: User = Depends(get_current_user)):
    template = render_template("unit_form.html", request=request, current_user=current_user)
    return HTMLResponse(template)


@app.get("/unit/{unit_id}", response_class=HTMLResponse)
async def show_unit(request: Request, unit_id: str, current_user: User = Depends(get_current_user)):
    # Fetch unit metadata (optional—if you need topic/grade/genre)
    with Session(engine) as session:
        unit = session.get(Unit, unit_id)
        if not unit:
            return HTMLResponse("<h1>Unit not found</h1>", status_code=404)

        # Fetch all summary lessons for this unit, ordered by idx
        lessons = session.exec(
            select(Lesson).where(Lesson.unit_id == unit_id).order_by(Lesson.idx)
        ).all()

    # Convert DB rows into simple dicts for your template
    lessons_data = [
        {
            "__idx": l.idx,
            "lesson_type": l.lesson_type,
            "title": l.title,
            "objective": l.objective
        }
        for l in lessons
    ]

    # Fixed categories list from config
    categories = list(UNIT_DEFAULTS["lesson_mix"].keys())

    template = render_template("unit_result.html", request=request, unit_id=unit_id, lessons=lessons_data,
                               categories=categories, current_user=current_user)
    return HTMLResponse(template)


@app.post("/unit/{unit_id}/lesson/add", response_class=HTMLResponse)
async def add_lesson(request: Request, unit_id: str, category: str = Form(...),
                     current_user=Depends(get_current_user_optional)):
    # 1) Verify the unit exists and belongs to this user (if enforcing ownership)
    with Session(engine) as session:
        unit = session.get(Unit, unit_id)
        if not unit:
            return HTMLResponse("Unit not found", status_code=404)
        # (optional) if unit.owner_id and current_user, enforce match

        # 2) Determine next idx by querying existing lessons
        existing = session.exec(
            select(Lesson.idx).where(Lesson.unit_id == unit_id)
        ).all()
        next_idx = max(existing or [-1]) + 1

    # 3) Generate a new summary lesson via AI
    prompt = (
        f"Generate exactly 1 new {category} lesson summary (idx {next_idx}) "
        f"for Grade {unit.grade} on topic “{unit.topic}” in genre “{unit.genre}”.\n"
        "Output only valid JSON with top-level key \"lessons\":\n"
        "[{ __idx (integer), lesson_type (string), title (string), objective (string) }]."
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    raw = resp.choices[0].message.content
    data = clean_and_extract_json(raw)
    lessons = data.get("lessons")
    if not lessons or not isinstance(lessons, list):
        return HTMLResponse(
            content=f"<tr><td colspan='4' class='text-red-500'>Error: no lessons in AI response</td></tr>",
            status_code=200
        )
    new = lessons[0]
    new["__idx"] = next_idx
    new["lesson_type"] = category  # enforce

    # 4) Persist the new summary lesson into the DB
    with Session(engine) as session:
        db_lesson = Lesson(
            unit_id=unit_id,
            idx=next_idx,
            lesson_type=category,
            title=new["title"],
            objective=new["objective"],
            # detail fields empty until drilled down
            standards="[]",
            success_criteria="{}",
            ell_strategies="[]",
            activities="[]",
            assessment=""
        )
        session.add(db_lesson)
        session.commit()

    # 5) Render and return the row partial via render_template
    html = render_template(
        "_lesson_row.html",
        request=request,
        unit_id=unit_id,
        lesson=new,
        current_user=current_user
    )
    return HTMLResponse(html)


@app.post("/generate-summary")
async def generate_summary(
        topic: str = Form(...),
        grade: int = Form(...),
        genre: str = Form(...),
        current_user: User = Depends(get_current_user)
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

        with Session(engine) as session:
            session.add(Unit(id=unit_id, topic=topic, grade=grade, genre=genre, owner_id=current_user.id))
            # persist summary lessons
            for lesson in summary:
                session.add(Lesson(
                    unit_id=unit_id,
                    idx=lesson["__idx"],
                    lesson_type=lesson["lesson_type"],
                    title=lesson["title"],
                    objective=lesson["objective"],
                    # leave detail fields empty for now
                    standards="[]",
                    success_criteria="{}",
                    ell_strategies="[]",
                    activities="[]",
                    assessment=""
                ))
            session.commit()

        # redirect to the overview page
        return RedirectResponse(url="/my-units", status_code=status.HTTP_302_FOUND)

    except Exception as e:
        logger.exception("generate-summary failed")
        return HTMLResponse(
            content=f"<h1>Generation Error</h1><pre>{e}</pre>",
            status_code=500
        )


@app.get("/unit/{unit_id}/lesson/{idx}", response_class=HTMLResponse)
async def lesson_detail_on_demand(request: Request, unit_id: str, idx: int,
                                  current_user=Depends(get_current_user_optional)):
    # 1) Load the lesson row from the database
    with Session(engine) as session:
        db_lesson = session.exec(
            select(Lesson)
            .where(Lesson.unit_id == unit_id)
            .where(Lesson.idx == idx)
        ).one_or_none()

    if not db_lesson:
        return HTMLResponse("<h1>Lesson not found</h1>", status_code=404)

    # 2) If detail already present, use it
    if db_lesson.standards != "[]" and db_lesson.success_criteria != "{}":
        detail = {
            "__idx": db_lesson.idx,
            "lesson_type": db_lesson.lesson_type,
            "title": db_lesson.title,
            "objective": db_lesson.objective,
            "standards": json.loads(db_lesson.standards),
            "success_criteria": json.loads(db_lesson.success_criteria),
            "ell_strategies": json.loads(db_lesson.ell_strategies),
            "activities": json.loads(db_lesson.activities),
            "assessment": db_lesson.assessment
        }
    else:
        # 3) Build a clean summary dict
        summary_dict = {
            "__idx": db_lesson.idx,
            "lesson_type": db_lesson.lesson_type,
            "title": db_lesson.title,
            "objective": db_lesson.objective
        }

        # 4) Create the prompt using JSON serialization
        prompt = (
            "Expand this lesson summary into a full lesson plan:\n"
            f"{json.dumps(summary_dict)}\n"
            "Include:\n"
            "- All applicable standards\n"
            "- A clear learning objective\n"
            "- Differentiated success_criteria (emerging/proficient/advanced)\n"
            "- At least one ELL strategy\n"
            "- A list of activities\n"
            "- An assessment description with only a string as assessment\n\n"
            "Output only valid JSON with key \"lesson\" containing these fields."
        )

        # 5) Call OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        raw = response.choices[0].message.content
        logger.info("Detail raw for lesson %s: %s", idx, raw)

        # 6) Parse the JSON
        data = clean_and_extract_json(raw)
        detail = data.get("lesson") or (data.get("lessons") or [None])[0]
        if not detail:
            return HTMLResponse("<h1>Invalid detail response</h1>", status_code=500)

        # 7) Ensure all expected keys exist
        detail.setdefault("standards", [])
        detail.setdefault("success_criteria", {"emerging": "", "proficient": "", "advanced": ""})
        detail.setdefault("ell_strategies", [])
        detail.setdefault("activities", [])
        detail.setdefault("assessment", "")

        # 8) Persist detail
        with Session(engine) as session:
            db_lesson.standards = json.dumps(detail["standards"])
            db_lesson.success_criteria = json.dumps(detail["success_criteria"])
            db_lesson.ell_strategies = json.dumps(detail["ell_strategies"])
            db_lesson.activities = json.dumps(detail["activities"])
            db_lesson.assessment = detail["assessment"]
            session.add(db_lesson)
            session.commit()

    # 9) Render template with current_user
    return HTMLResponse(
        render_template(
            "lesson_detail.html",
            request=request,
            lesson=detail,
            unit_id=unit_id,
            idx=idx,
            current_user=current_user
        )
    )


@app.get("/my-units", response_class=HTMLResponse)
async def my_units(request: Request, search: str = "", current_user: User = Depends(get_current_user)):
    with Session(engine) as session:
        stmt = select(Unit).where(Unit.owner_id == current_user.id)
        if search:
            stmt = stmt.where(Unit.topic.contains(search))
        units = session.exec(stmt).all()

    return HTMLResponse(
        render_template(
            "my_units.html",
            request=request,
            units=units,
            search=search,
            current_user=current_user
        )
    )


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("access_token")
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
