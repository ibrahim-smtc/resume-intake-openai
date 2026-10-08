"""A stand-in for OpenAI's Responses API (POST /v1/responses), so the agent tests never reach OpenAI.

Script what the "model" answers, call by call: say("text") replies with a message, call("tool", {args}) asks for a tool call,
fail(429) answers with an HTTP error, slow(5) answers late. Every request the agent sent is kept in mock.requests, so a test can check how many
model calls a turn made, with which tools, and how big they were. Token counts in the answers are the request's size / 4.

    mock = start()                      # after harness.isolate(); sets OPENAI_BASE_URL and OPENAI_API_KEY
    mock.script(call("list_open_roles", {"search": "", "limit": 5}), say("There are 3 open roles."))
"""
import asyncio
import json
import os
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from harness import serve


def say(text: str) -> dict:
    return {"text": text}


def call(name: str, args: dict) -> dict:
    return {"call": name, "args": args}


def fail(status: int, message: str = "mock error") -> dict:
    return {"status": status, "message": message}


def slow(seconds: float, text: str = "late") -> dict:
    return {"sleep": seconds, "text": text}


class Mock:
    def __init__(self):
        self.requests, self.queue, self.url = [], [], None

    def script(self, *steps) -> None:
        self.queue = list(steps)

    def reset(self) -> None:
        self.requests.clear()
        self.queue.clear()

    def tool_calls(self, request: dict) -> list:
        return [i.get("name") for i in request.get("input", []) if isinstance(i, dict) and i.get("type") == "function_call"]


def _response(request: dict, step: dict, n: int) -> dict:
    size = len(json.dumps(request))
    if "call" in step:
        item = {"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{uuid.uuid4().hex[:8]}", "name": step["call"],
                "arguments": json.dumps(step["args"]), "status": "completed"}
    else:
        item = {"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": step.get("text", ""), "annotations": []}]}
    return {"id": f"resp_{n}", "object": "response", "created_at": 0, "model": request.get("model"), "status": "completed",
            "output": [item], "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": size // 4, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": size // 4 + 20}}


def start() -> Mock:
    mock = Mock()
    app = FastAPI()

    @app.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        mock.requests.append(body)
        if request.headers.get("authorization") != "Bearer sk-test-key":
            return JSONResponse({"error": {"message": "Incorrect API key provided", "type": "invalid_request_error"}}, 401)
        step = mock.queue.pop(0) if mock.queue else {"text": "(no scripted answer)"}
        if "sleep" in step:
            await asyncio.sleep(step["sleep"])
        if "status" in step:
            return JSONResponse({"error": {"message": step["message"], "type": "mock"}}, step["status"])
        return _response(body, step, len(mock.requests))

    mock.url = serve(app).url + "/v1"
    os.environ["OPENAI_BASE_URL"] = mock.url
    os.environ["OPENAI_API_KEY"] = "sk-test-key"
    return mock
