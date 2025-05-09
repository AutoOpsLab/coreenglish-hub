import re, json, logging
from openai import OpenAI
from config import UNIT_DEFAULTS

logger = logging.getLogger(__name__)


def build_chunk_prompt(topic: str, grade: int, genre: str, framework: str,
                       mix: dict, start: int, end: int, total: int) -> str:
    return f"""
Generate lessons {start}–{end} of a {total}-lesson unit for Grade {grade}
on topic “{topic}” in genre “{genre}” using {framework}. This batch is lessons {start} to {end}.
Each week must include:
- {mix['reading']} reading lessons
- {mix['writing']} writing lessons
- {mix['grammar']} grammar lessons
- {mix['vocabulary']} vocabulary lessons
- {mix['listening_speaking']} listening/speaking lessons

For each lesson include: standards, objective, differentiated success_criteria, ell_strategies, activities, assessment.
Output only valid JSON with key "lessons": list of {end - start + 1} lessons.
""".strip()


def clean_and_extract_json(raw: str) -> dict:
    # remove JS comments
    no_comments = re.sub(r'^\s*//.*\n?', '', raw, flags=re.MULTILINE)
    # extract JSON object
    start_idx = no_comments.find('{')
    end_idx = no_comments.rfind('}')
    if start_idx < 0 or end_idx < 0:
        raise ValueError("No JSON object found")
    snippet = no_comments[start_idx:end_idx + 1]
    return json.loads(snippet)
