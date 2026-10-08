# The recruiter agent (OpenAI Agents SDK)

The chat page's text box is answered by an agent that runs inside this app ([app/agent/openai_agent.py](../app/agent/openai_agent.py)),
built on the OpenAI Agents SDK. It replaces the Perfox agent: no webhook, no external agent platform. The same tools are still
served at `POST /mcp` for any MCP client.

```
 page ─► POST /api/chat ─► openai_agent.chat()
                              │
          a file attached ────┤──► the fixed pipeline (tools.process_resume): junk check, parse, profile, match, screening
                              │      no model decides anything ─► ONE small call writes the reply (writer, no tools)
                              │                                    (if the recruiter also typed something: the chat agent)
          only text ──────────┴──► the chat agent (model + ~10 tools) ─► reply
```

## Why it is built this way

Measured with Perfox: about 80k tokens per conversation, about 5 model calls and ~11 s of model work per turn, because every
call carried a ~1.9k-token prompt and ~4k tokens of tool descriptions. Here:

- **An attached file costs one small call.** The pipeline is plain code; the model only writes the reply (`OPENAI_REPLY_MODEL`).
- **Small calls.** About 2k tokens of instructions + tool definitions per call: one-line tool descriptions
  ([app/agent/agent_tools.py](../app/agent/agent_tools.py)), short prompts ([app/agent/prompts.py](../app/agent/prompts.py)).
- **Fewer calls.** `complete_resume` stores the recruiter's answers, creates the profile and matches roles in one tool call;
  `complete_job` does the same for a JD.
- **Prompt caching.** Instructions and tools come first and never change, so OpenAI's automatic cache applies (cached input is
  billed at 10%).
- **No server-side conversation.** The page sends its last few messages (text only) and the `[context]` lines of the files
  taken in so far. Any Vercel instance can answer, and there are no half-finished tool calls in the history.

## Settings

| Setting | Default | What for |
|---|---|---|
| `OPENAI_API_KEY` | (none) | required; without it the page uploads straight into the pipeline |
| `OPENAI_MODEL` | `gpt-5.4-mini` | answers typed messages, with tools |
| `OPENAI_REPLY_MODEL` | `gpt-5.4-nano` | writes the reply after an attached file was processed |
| `OPENAI_REASONING_EFFORT`, `OPENAI_VERBOSITY` | `low` | empty = the model's default (needed for models that don't take them) |
| `AGENT_MAX_TURNS` | `6` | model calls one typed message may take |
| `AGENT_HISTORY_MESSAGES` | `8` | earlier messages sent with each turn |
| `OPENAI_BASE_URL` | | only for a proxy or a compatible endpoint |

## Privacy

Candidate data (the parsed resume summary, names, contact details) is sent to OpenAI to write replies. Requests are sent with
`store=false`, and the SDK's tracing to OpenAI is off (set `OPENAI_AGENTS_TRACING=1` to turn it on). The resume file itself is
never sent to OpenAI: it goes to Questlight's parser, as before.

## Measuring it

Every turn is a trace (`chat-<id>`) on `/trace`, with one span per model call and its tokens, and `/api/chat` returns the same
numbers in `usage`. To compare with the Perfox deployment:

```
python scripts/measure_chat.py https://<this deployment> --runs 5 --password <APP_PASSWORD>
python scripts/measure_chat.py https://<the Perfox deployment> --runs 5 --password <its APP_PASSWORD>
```

The Perfox deployment returns no `usage`, so only its time is shown; its tokens come from Perfox's own reports. Uploads create
(or find a duplicate of) a profile in Questlight: fake resumes only.

## Known limits

- Files taken in are still kept per Vercel instance for 30 minutes ([app/agent/intakes.py](../app/agent/intakes.py)): a
  follow-up served by another instance can't find the file and says so. A shared store (Redis or a database) would fix it.
- Questlight's parser (7-29 s) is unchanged, so an upload still takes about 15-25 s; the pipeline is the same as before on
  purpose, so the comparison with Perfox is fair.
