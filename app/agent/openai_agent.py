"""The recruiter agent, inside this app (OpenAI Agents SDK). It replaces the Perfox agent.

How a turn runs (hybrid):
- An attached file goes straight through the fixed pipeline (tools.process_resume: junk check, parse, profile in
  Questlight, matching, screening; a job description goes to the JD intake), with no model deciding anything. Then one
  small call (WRITER, no tools, OPENAI_REPLY_MODEL) writes the reply; if the recruiter also typed something, the chat agent
  answers instead, with the result in its input. The pipeline is never cut short, and if the reply call fails a plain
  template reply is built from the result: the work is done, so the recruiter is told what happened, not "error".
- A typed message goes to the chat agent (OPENAI_MODEL) with ~10 tools (app/agent/agent_tools.py).

History: the page sends the last few messages (text only) and the [context] lines of the files taken in so far. The
conversation itself is not kept here, so it works whichever Vercel instance answers; the files are still kept per
instance, for 30 minutes (app/agent/intakes.py).

Each turn is a trace (chat-<id>) with one span per model call and its tokens (see /trace); the reply carries the same
numbers in "usage". Tracing to OpenAI is off: resumes are personal data.

Settings: OPENAI_API_KEY (required), OPENAI_MODEL (gpt-5.4-mini), OPENAI_REPLY_MODEL (gpt-5.4-nano), OPENAI_BASE_URL
(optional), OPENAI_REASONING_EFFORT (low) and OPENAI_VERBOSITY (low) (empty = the model's default; needed for models
that don't take them), AGENT_MAX_TURNS (6), AGENT_HISTORY_MESSAGES (8).
"""
import asyncio
import json
import os
import secrets
import time

from app.agent import intakes, prompts, tools, uploads
from app.observability import tracing

TURN_BUDGET_S = 270      # Vercel stops a request at 300 s
WRITER_MIN_S = 20        # less time than this left after the pipeline: the template reply, no model call
MESSAGE_MAX = 4000       # characters of the recruiter's message
HISTORY_TEXT_MAX = 2000  # characters of each earlier message
CONTEXT_MAX = 1500


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(_env(name, str(default))), high))
    except ValueError:
        return default


def configured() -> bool:
    return bool(_env("OPENAI_API_KEY"))


def chat_model() -> str:
    return _env("OPENAI_MODEL") or "gpt-5.4-mini"


def reply_model() -> str:
    return _env("OPENAI_REPLY_MODEL") or "gpt-5.4-nano"


# ---------- what the page sends ----------

def clean_history(raw) -> list:
    """The page's last messages: only user/assistant text, each one capped, the most recent AGENT_HISTORY_MESSAGES."""
    try:
        items = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or [])
    except ValueError:
        return []
    keep = [{"role": m["role"], "content": m["content"].strip()[:HISTORY_TEXT_MAX]}
            for m in (items if isinstance(items, list) else [])
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
            and m["content"].strip()]
    limit = _int("AGENT_HISTORY_MESSAGES", 8, 0, 30)
    return keep[-limit:] if limit else []


def context_line(ref):
    """One line about a resume or JD taken in this chat, for the next turns: its id, who/what it is and what is missing."""
    intake = intakes.find(ref)
    if intake is None:
        return None
    if intake.kind == "jd":
        job = intake.job or {}
        bits = [f"jd '{intake.name}'", f"jd_id={intake.id}", f"job={job.get('status') or 'not created'}"]
        if job.get("job_id"):
            bits.append(str(job["job_id"]))
        missing = [i.get("id") for i in job.get("missing_items") or []]
    else:
        profile = intake.profile or {}
        bits = [f"resume '{intake.name}'", f"file_id={intake.id}"]
        if (intake.parsed or {}).get("name"):
            bits.append(f"candidate={intake.parsed['name']}")
        bits.append(f"profile={profile.get('status') or 'not created'}" + (f" {profile['applicantId']}" if profile.get("applicantId") else ""))
        missing = [i.get("id") for i in profile.get("missing_items") or []]
    if missing:
        bits.append("missing=" + ",".join(str(m) for m in missing))
    return "[context] " + ", ".join(bits)


# ---------- the SDK (imported only when a turn runs) ----------

def _client():
    from openai import AsyncOpenAI
    options = {"api_key": _env("OPENAI_API_KEY"), "timeout": 60, "max_retries": 1}
    if _env("OPENAI_BASE_URL"):
        options["base_url"] = _env("OPENAI_BASE_URL")
    return AsyncOpenAI(**options)


def _model_settings(with_tools: bool, max_tokens: int):
    from agents import ModelSettings
    from openai.types.shared import Reasoning
    options = {"store": False, "max_tokens": max_tokens}  # store=False: OpenAI keeps no copy of the conversation
    effort = _env("OPENAI_REASONING_EFFORT", "low")
    if effort:
        options["reasoning"] = Reasoning(effort=effort)
        options["response_include"] = ["reasoning.encrypted_content"]  # needed between calls when nothing is stored
    if _env("OPENAI_VERBOSITY", "low"):
        options["verbosity"] = _env("OPENAI_VERBOSITY", "low")
    if with_tools:
        options["parallel_tool_calls"] = False  # complete_resume then match_roles must not run at the same time
    return ModelSettings(**options)


def _agents():
    """The chat agent (tools) and the writer (no tools), built from the current settings."""
    from agents import Agent, OpenAIResponsesModel, set_tracing_disabled

    from app.agent import agent_tools
    if _env("OPENAI_AGENTS_TRACING") != "1":
        set_tracing_disabled(True)
    client = _client()
    chat = Agent(name="recruiter", instructions=prompts.CHAT, tools=agent_tools.build(),
                 model=OpenAIResponsesModel(chat_model(), client), model_settings=_model_settings(True, 2000))
    writer = Agent(name="writer", instructions=prompts.WRITER,
                   model=OpenAIResponsesModel(reply_model(), client), model_settings=_model_settings(False, 1200))
    return chat, writer


class _Meter:
    """Counts each model call of a turn (time and tokens) and adds it to the trace as an "llm" span."""

    def __init__(self, trace: tracing.Trace):
        self.trace, self.calls, self._open = trace, [], {}

    def hooks(self):
        from agents import RunHooks
        meter = self

        class Hooks(RunHooks):
            async def on_llm_start(self, context, agent, system_prompt, input_items):
                root = meter.trace.spans[0] if meter.trace.spans else None
                span = tracing.Span(meter.trace, f"model call ({agent.name})", "llm", root)
                meter.trace.spans.append(span)
                meter._open[agent.name] = span

            async def on_llm_end(self, context, agent, response):
                span = meter._open.pop(agent.name, None)
                usage = response.usage
                cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
                model = getattr(agent.model, "model", None)
                meter.calls.append({"agent": agent.name, "model": model, "input": usage.input_tokens,
                                    "cached": cached, "output": usage.output_tokens})
                if span:
                    span.duration_ms = (time.perf_counter() - span._t0) * 1000
                    span.set_usage(usage.input_tokens, usage.output_tokens, "openai", model)
                    span.set(cached_tokens=cached)

        return Hooks()

    def usage(self, started: float) -> dict:
        return {"model_calls": len(self.calls), "input_tokens": sum(c["input"] for c in self.calls),
                "cached_tokens": sum(c["cached"] for c in self.calls), "output_tokens": sum(c["output"] for c in self.calls),
                "models": sorted({c["model"] for c in self.calls if c["model"]}), "seconds": round(time.monotonic() - started, 1),
                "trace_id": self.trace.id}


async def _run(agent, items, meter: _Meter, budget: float) -> str:
    from agents import RunConfig, Runner
    result = await asyncio.wait_for(
        Runner.run(agent, items, max_turns=_int("AGENT_MAX_TURNS", 6, 1, 20), hooks=meter.hooks(),
                   run_config=RunConfig(tracing_disabled=True, workflow_name="resume intake")),
        timeout=budget)
    return str(result.final_output or "").strip()


def _problem(exc: BaseException) -> str:
    """A plain message for the recruiter, never the key."""
    import openai
    from agents import MaxTurnsExceeded
    if isinstance(exc, MaxTurnsExceeded):
        return "I stopped after too many steps without finishing: try asking in a simpler way"
    if isinstance(exc, (asyncio.TimeoutError, openai.APITimeoutError)):
        return "the model didn't answer in time: try again"
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key: check OPENAI_API_KEY"
    if isinstance(exc, openai.PermissionDeniedError):
        return "OpenAI refused the request: the key's project may not have access to this model"
    if isinstance(exc, openai.NotFoundError):
        return f"OpenAI doesn't know the model ({chat_model()} / {reply_model()}): check OPENAI_MODEL and OPENAI_REPLY_MODEL"
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI's rate limit or quota was reached: wait a minute, or check the account's billing"
    if isinstance(exc, openai.APIConnectionError):
        return "couldn't reach OpenAI: try again"
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI returned HTTP {exc.status_code}"
    return f"the agent failed ({type(exc).__name__})"


# ---------- the reply when the model can't write it ----------

def template_reply(result: dict) -> str:
    """A plain reply built from an intake result, used when the reply call fails or there is no time left for it."""
    name = result.get("file") or "the file"
    if not result.get("ok"):
        return f"I couldn't take in **{name}**: {result.get('error') or 'unknown error'}."
    if result.get("kind") == "job_description" or "job_status" in result:
        job, status = result.get("job") or {}, result.get("job_status")
        title = job.get("title") or name
        if status == "created":
            lines = [f"Job **{title}** is in Questlight ({result.get('job_id')})."]
            lines += [f"- {c.get('name')} ({c.get('candidate_id')}): {c.get('score')}/100" for c in result.get("top_candidates") or []]
            return "\n".join(lines)
        if status == "not_created":
            wanted = ", ".join(str(i.get("what")) for i in result.get("missing_items") or [])
            return f"I read the job description for **{title}**. To create the job, Questlight still needs: {wanted}."
        return f"**{title}**: {result.get('job_message') or status}."
    decision = result.get("decision")
    if decision in ("junk", "needs_review"):
        verdict = "isn't a resume, so it was set aside" if decision == "junk" else "needs a person to look at it"
        return f"**{name}** {verdict}: {result.get('reason')}."
    who = (result.get("candidate") or {}).get("name") or name
    profile = result.get("profile") or {}
    status = profile.get("status")
    if status == "created":
        lines = [f"Done: **{who}** is in Questlight ({profile.get('candidate_id')})."]
    elif status == "duplicate":
        lines = [f"**{who}** is already in Questlight (same email), so nothing was changed."]
    elif status == "not_created":
        wanted = ", ".join(f"{i.get('what')} ({i.get('for')})" for i in result.get("missing_items") or [])
        return f"I read **{who}**'s resume. To create the profile, Questlight still needs: {wanted}. What are they?"
    else:
        lines = [f"**{who}**: {profile.get('message') or 'the profile was not created'}."]
    lines += [f"- {r.get('title')} ({r.get('job_id')}): {r.get('score')}/100" for r in result.get("top_roles") or []]
    if (result.get("screening") or {}).get("message"):
        lines.append(str(result["screening"]["message"]).capitalize() + ".")
    return "\n".join(lines)


# ---------- one turn ----------

async def chat(message: str, history="", context="", file=None) -> dict:
    """One turn of the page's chat. file is (name, bytes) or None.
    Returns {"ok": True, "reply", "context": [lines], "usage"} or {"ok": False, "error", "usage"}."""
    started = time.monotonic()
    text = str(message or "").strip()[:MESSAGE_MAX]
    if not file and not text:
        return {"ok": False, "error": "nothing to send"}
    if file and uploads.file_problem(*file):
        return {"ok": False, "error": uploads.file_problem(*file)}

    from app.agent import agent_tools
    items = clean_history(history)
    if str(context or "").strip():
        items.append({"role": "developer", "content": str(context).strip()[:CONTEXT_MAX]})
    trace = tracing.Trace("chat-" + secrets.token_hex(6))
    meter = _Meter(trace)
    seen = []
    token = agent_tools.touched.set(seen)
    reply, error = None, None
    try:
        with trace.span("agent turn", kind="chain") as root:
            root.set(channel=tools.CHANNEL, model=chat_model(), with_file=bool(file), history_messages=len(items))
            try:
                chat_agent, writer = _agents()
                if file:
                    name, data = file
                    link = uploads.store_file(name, data)
                    with tracing.span("intake pipeline") as step:   # never cancelled: it writes to Questlight
                        result = await tools.process_resume(link, name)
                        step.set(pipeline_trace=result.get("trace_id"), decision=result.get("decision"))
                    agent_tools.note(result.get("file_id") or result.get("jd_id"))
                    summary = agent_tools.out(dict(result))
                    left = TURN_BUDGET_S - (time.monotonic() - started)
                    if text:
                        items.append({"role": "user", "content": f"{text}\n\n[The attached file {name} was processed. Result: {summary}]"})
                        reply = await _run(chat_agent, items, meter, max(left, 5))
                    elif left >= WRITER_MIN_S:
                        try:
                            reply = await _run(writer, [{"role": "user", "content": f"Intake result for {name}: {summary}"}], meter, left)
                        except Exception as exc:  # the work is done: tell the recruiter what happened anyway
                            root.set(writer_failed=_problem(exc))
                    reply = reply or template_reply(result)
                else:
                    items.append({"role": "user", "content": text})
                    reply = await _run(chat_agent, items, meter, TURN_BUDGET_S)
            except Exception as exc:
                error = _problem(exc)
                root.fail(error)
                print(f"[agent] {type(exc).__name__}: {error}")
    finally:
        agent_tools.touched.reset(token)
        trace.save()
    usage = meter.usage(started)
    if error and not reply:
        return {"ok": False, "error": error, "usage": usage}
    lines = [line for line in (context_line(ref) for ref in seen) if line]
    return {"ok": True, "reply": reply or "Done.", "context": lines, "usage": usage}
