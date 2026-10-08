"""The in-app agent's tools (app/agent/openai_agent.py): thin wrappers around the MCP tools in app/agent/tools.py.

They exist to keep every model call small and the conversation short:
- one-line descriptions (the MCP docstrings are long, written for agents that see nothing else) and strict schemas;
- composite steps: complete_resume = provide_missing_details + create_profile + match_roles, complete_job =
  provide_job_details + create_job, so answering a question costs one tool call instead of three;
- compact results: no empty values, no trace ids, no "next" hints naming tools this agent doesn't have.
A JD can only be given as pasted text here (an attached file goes through the pipeline before the model sees anything),
so the model can never make the server fetch a URL.

Imported only by openai_agent.py, so the Agents SDK is loaded only when the agent runs.
"""
import json
from contextvars import ContextVar

from agents import function_tool
from pydantic import BaseModel

from app import pipeline
from app.agent import formatting, intakes, tools

# The file_ids / jd_ids the tools touched in this turn, so the reply can carry a [context] line for them.
touched: ContextVar = ContextVar("touched_intakes", default=None)

GONE = ("I no longer have that file or job description on this server (they are kept for 30 minutes, and per server): "
        "ask the recruiter to attach it again")
DROP = {"trace_id", "next", "parser"}


class Answer(BaseModel):
    id: str     # an id from missing_items, e.g. "email" or "job_title:1"
    value: str  # exactly what the recruiter said


def compact(value):
    """Drops empty values and fields the model doesn't need, recursively."""
    if isinstance(value, dict):
        out = {k: compact(v) for k, v in value.items() if k not in DROP}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [compact(v) for v in value]
    return value


def out(result: dict) -> str:
    if result.get("ok") is False and str(result.get("error", "")).startswith("no file found"):
        result = {**result, "error": GONE}
    for key in ("file_id", "jd_id"):
        if result.get(key):
            note(result[key])
    return json.dumps(compact(result), separators=(",", ":"), ensure_ascii=False, default=str)


def note(ref) -> None:
    seen = touched.get()
    if seen is not None and ref and ref not in seen:
        seen.append(ref)


def _answers(answers) -> list:
    return [{"id": a.id, "value": a.value} for a in answers or []]


async def get_intake_status(ref: str) -> str:
    intake, err = intakes.lookup(ref)
    if err or intake is None:
        return out({"ok": False, "error": GONE})
    return out(tools._where_it_stands(intake))


async def complete_resume(file_id: str, answers: list[Answer], fill_missing: bool) -> str:
    intake, err = intakes.lookup(file_id)
    if err or intake is None:
        return out({"ok": False, "error": GONE})
    if intake.kind == "jd":
        return out({"ok": False, "error": "that is a job description: use complete_job with its jd_id"})
    exists = intake.profile and intake.profile.get("status") in ("created", "duplicate")
    details = None
    if not exists:
        details = await tools.provide_missing_details(intake.id, _answers(answers))
        if not details.get("ok"):
            return out(details)
        if not details.get("info_complete") and not fill_missing:
            return out({**details, "profile_created": False,
                        "note": "still missing: ask the recruiter for missing_items, then call complete_resume again"})
        created = await tools.create_profile(intake.id, fill_missing=fill_missing)
        if (created.get("profile") or {}).get("status") not in ("created", "duplicate"):
            return out(created)
    matched = await tools.match_roles(intake.id)
    profile = intake.profile or {}
    return out({"ok": True, "file_id": intake.id,
                "candidate": (details or {}).get("candidate") or (formatting.candidate(pipeline.redact(intake.parsed)) if intake.parsed else None),
                "profile_created": profile.get("status") == "created", "profile": formatting.profile_view(profile),
                **{k: matched.get(k) for k in ("roles_status", "roles_message", "top_roles", "screening", "error")}})


async def match_roles(file_id: str) -> str:
    return out(await tools.match_roles(file_id))


async def complete_job(jd_id: str, answers: list[Answer], allow_duplicate: bool) -> str:
    if answers:
        details = await tools.provide_job_details(jd_id, _answers(answers))
        if not details.get("ok") or not details.get("info_complete"):
            return out(details)
    return out(await tools.create_job(jd_id, allow_duplicate=allow_duplicate))


async def process_jd_text(text: str) -> str:
    return out(await tools.process_job_description(text=text))


async def find_candidates_for_job(job: str, top_k: int) -> str:
    return out(await tools.find_candidates_for_job(job, max(1, min(int(top_k or 3), 10))))


async def add_candidates_to_job(job: str, candidate_ids: list[str]) -> str:
    return out(await tools.add_candidates_to_job(job, candidate_ids))


async def list_open_roles(search: str, limit: int) -> str:
    return out(await tools.list_open_roles(search or "", limit or 10))


async def get_intake_summary(days: int) -> str:
    return out(await tools.get_intake_summary(days or 1))


async def list_recent_intakes(limit: int) -> str:
    return out(await tools.list_recent_intakes(limit or 10))


DESCRIPTIONS = {
    get_intake_status: "Where a resume or JD taken in this chat stands: candidate summary (experience, recent roles, "
                       "education, skills), profile, matched roles, or the job. ref = its file_id or jd_id. Read-only.",
    complete_resume: "Finish a resume: store the recruiter's answers for its missing_items (ids from missing_items, values "
                     "exactly as said), then create the Questlight profile and match/screen roles. fill_missing=true only "
                     "if the recruiter said to go ahead without the missing details. WRITES TO QUESTLIGHT.",
    match_roles: "Best matching open roles for a candidate whose profile exists (file_id); a new profile is also put on "
                 "the top 3 at the Screening stage. Retries failed screenings.",
    complete_job: "Finish a JD: store answers for its missing_items (ids from missing_items), then create the job and screen "
                  "the best candidates. allow_duplicate=true only if the recruiter said to create it although a similar "
                  "job is open. WRITES TO QUESTLIGHT.",
    process_jd_text: "Take in a job description the recruiter pasted (the exact text): creates the job and screens "
                     "candidates, or lists what is missing. WRITES TO QUESTLIGHT.",
    find_candidates_for_job: "Rank candidates already in Questlight for an open job (job id or title). top_k 1-10, use 3 "
                             "unless asked. Read-only.",
    add_candidates_to_job: "Put chosen candidates (candidate ids) on a job at the Screening stage. ONLY after the recruiter "
                           "clearly said yes. WRITES TO QUESTLIGHT.",
    list_open_roles: "Count and sample of open roles in Questlight. search filters by title (empty = all); limit 1-30.",
    get_intake_summary: "Counts of resumes and JDs received in the last N days (1-90), from the audit log.",
    list_recent_intakes: "The most recent files received, newest first (limit 1-50).",
}


def build() -> list:
    """The agent's tools, built fresh (cheap) so tests can change the environment between runs."""
    return [function_tool(fn, description_override=text, use_docstring_info=False) for fn, text in DESCRIPTIONS.items()]
