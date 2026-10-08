"""OPT-IN: the agent against the REAL OpenAI API. Skipped unless RUN_LIVE_OPENAI=1. Needs OPENAI_API_KEY (from the
environment or .env). Questlight stays FAKE (the parser and profile creation are stubbed, the job list is hand-made), so
nothing is created anywhere; the cost is a fraction of a rupee.

Prints what each turn cost (model calls, tokens, seconds) so the numbers can be compared with Perfox's.
"""
import json
import os
import sys
from pathlib import Path

if os.getenv("RUN_LIVE_OPENAI") != "1":
    print("skipped: set RUN_LIVE_OPENAI=1 to run the agent against the real OpenAI API")
    sys.exit(0)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from dotenv import dotenv_values  # noqa: E402

live = {k: v for k, v in {**dotenv_values(ROOT / ".env"), **os.environ}.items()
        if k.startswith(("OPENAI_", "AGENT_")) and k != "OPENAI_BASE_URL" and v}
if not live.get("OPENAI_API_KEY"):
    print("OPENAI_API_KEY is not set (environment or .env)")
    sys.exit(1)

import harness  # noqa: E402

harness.isolate(**live)
from harness import RESUMES, check, finish  # noqa: E402

import agent_harness as ah  # noqa: E402
from fixtures.parsed import ROHIT  # noqa: E402

fakes = ah.start({"rohit": ROHIT})
import httpx  # noqa: E402

client = httpx.Client(timeout=200)


def chat(message="", file=None, context=""):
    r = client.post(f"{fakes.server.url}/api/chat", data={"message": message, "history": "[]", "context": context},
                    files={"file": file} if file else None).json()
    print("   ", json.dumps(r.get("usage")), "|", (r.get("reply") or r.get("error") or "")[:160].replace("\n", " "))
    return r


r = chat("how many roles are open right now?")
check("a typed question: answered by the real model (with a tool)", r.get("ok") and r["usage"]["model_calls"] >= 1, r.get("error"))
r = chat("", ("rohit.pdf", (RESUMES / "Rohit_Verma_Python_Backend.pdf").read_bytes(), "application/pdf"))
check("an upload: one model call writes the reply, and it names the candidate",
      r.get("ok") and r["usage"]["model_calls"] == 1 and "Rohit" in r["reply"], r.get("error"))
ctx = (r.get("context") or [""])[0]
r = chat("what are his skills?", context=ctx)
check("a follow-up: answered from the server's copy", r.get("ok") and r["usage"]["model_calls"] >= 1, r.get("error"))
finish()
