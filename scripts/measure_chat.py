"""Measures the chat of a deployment: time, model calls, tokens and cost per turn, over a few scripted conversations.

    python scripts/measure_chat.py https://<deployment> --runs 5 [--password APP_PASSWORD] [--file a_fake_resume.pdf]

Run it against this app and against the old Perfox deployment (same /api/chat): the old one returns no "usage" block, so
only its time is shown; take its tokens from Perfox's own reports. Uploads create (or find a duplicate of) a candidate
profile in Questlight: use FAKE resumes only. Costs are list prices per 1M tokens (USD), converted at --usd-inr.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "tests" / "fixtures" / "resumes" / "Rohit_Verma_Python_Backend.pdf"   # a fake resume
PRICES = {  # model: (input, cached input, output), USD per 1M tokens, from OpenAI's pricing page (Oct 2026)
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02, 1.25),
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5-nano": (0.05, 0.005, 0.40),
}


def turn(client, base, message, file=None, history=None, context=""):
    data = {"message": message, "session_id": "measure", "history": json.dumps(history or []), "context": context}
    files = {"file": (file.name, file.read_bytes(), "application/octet-stream")} if file else None
    started = time.monotonic()
    r = client.post(f"{base}/api/chat", data=data, files=files)
    seconds = time.monotonic() - started
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"ok": False, "error": f"HTTP {r.status_code}"}
    return seconds, body


def cost_inr(usage: dict, usd_inr: float):
    """Approximate: the whole turn priced at its first model's rate (an upload turn uses one model only)."""
    models = usage.get("models") or []
    rates = PRICES.get(models[0]) if models else None
    if not rates:
        return None
    fresh = usage["input_tokens"] - usage["cached_tokens"]
    usd = (fresh * rates[0] + usage["cached_tokens"] * rates[1] + usage["output_tokens"] * rates[2]) / 1e6
    return usd * usd_inr


def conversations(file: Path):
    yield "hello", [("hello", None)]
    yield "open roles", [("how many roles are open?", None)]
    yield "upload", [("", file)]
    yield "upload + follow-up", [("", file), ("what is his education?", None)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base_url")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--password", default="")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE)
    ap.add_argument("--usd-inr", type=float, default=97.0)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    auth = ("measure", args.password) if args.password else None
    rows = {}
    with httpx.Client(timeout=300, auth=auth) as client:
        for name, steps in conversations(args.file):
            for _ in range(args.runs):
                history, context = [], []
                for i, (message, file) in enumerate(steps):
                    seconds, body = turn(client, base, message, file, history, "\n".join(context))
                    usage = body.get("usage") or {}
                    label = f"{name} [{i + 1}/{len(steps)}]"
                    rows.setdefault(label, []).append((seconds, body.get("ok"), usage, cost_inr(usage, args.usd_inr) if usage else None))
                    if not body.get("ok"):
                        print(f"  {label}: {body.get('error')}", file=sys.stderr)
                    history += [{"role": "user", "content": message or f"[attached {file.name}]"},
                                {"role": "assistant", "content": body.get("reply") or ""}]
                    context += body.get("context") or []

    print(f"\n{'turn':32} {'ok':>5} {'median s':>9} {'calls':>6} {'input':>8} {'cached':>7} {'output':>7} {'INR':>7}")
    for label, runs in rows.items():
        ok = sum(1 for r in runs if r[1])
        med = lambda xs: statistics.median(xs) if xs else None  # noqa: E731
        usages = [r[2] for r in runs if r[2]]
        def col(key):
            v = med([u.get(key, 0) for u in usages])
            return f"{v:.0f}" if v is not None else "-"
        inr = med([r[3] for r in runs if r[3] is not None])
        print(f"{label:32} {ok:>2}/{len(runs):<2} {med([r[0] for r in runs]):>9.1f} {col('model_calls'):>6} {col('input_tokens'):>8} "
              f"{col('cached_tokens'):>7} {col('output_tokens'):>7} {(f'{inr:.2f}' if inr is not None else '-'):>7}")


if __name__ == "__main__":
    main()
