# Resume Intake for Questlight

An AI resume intake agent for **Questlight** (a multi-tenant ATS). A recruiter drops in a resume; the app screens out files that
aren't resumes, reads it, creates the candidate's profile in Questlight, shows the best matching open roles and puts the
candidate on the best 3 of them at the Screening stage. If the resume
lacks something Questlight requires (a job title, say), it asks, and the recruiter's answer completes the profile.

It also takes **job descriptions** (PDF, DOCX, TXT, old Word DOC, or text pasted into the chat): the JD is read, the job is
created in Questlight, Questlight's own matching is started, and the best candidates already in Questlight (strong matches
only, up to 3) are put on the new job at the Screening stage. What a JD doesn't say but Questlight needs (the client, the
salary range...) is asked for. Recruiters use it through a chat page, answered by an agent inside this app (OpenAI Agents
SDK); the same code is also served as MCP tools.

AI is used **only to read the resume or the JD** (Questlight's parsers). Everything else (junk and JD rules, the profile and
job mapping, matching, the audit log) is plain, hardcoded code.

```
 resume ─► intake checks ─► "Junk?" ─► parse ─► load into Questlight ─► match open roles ─► screen top 3
              (size, type)   (rules,    (AI)     (profile created,       (own scoring,        (added to each job
                              no AI)             or: what is missing)     no AI)               at Screening, no AI)
                       every step is written to the audit log and traced

 JD ─► intake checks ─► "What is it?" ─► parse JD ─► duplicate? ─► create the job ─► Questlight's matching ─► screen the best
        (PDF, DOCX,      (rules: a JD,    (AI,        (Questlight   (or: what is       (started, as its       (own ranking of the
         TXT, DOC, text)  a resume, junk)  Questlight)  check)        missing, to ask)   own UI does)           candidates, 40+ only)
```

## Layout

```
app/
  main.py            the web app: pages, JSON APIs, and POST /mcp   (Vercel's entrypoint)
  settings.py        every setting, from environment variables / .env
  security.py        who may reach what (password for the pages, bearer token for /mcp)
  pipeline.py        runs the resume steps in order: the BRD workflow
  job_pipeline.py    runs the JD steps in order: a JD in, a Questlight job out, screened with the best candidates
  intake/            the workflow blocks
    documents.py       read text out of a PDF, DOCX, TXT or old Word DOC
    junk.py            "Junk?" rules, and telling a JD from a resume
    jobs.py            read a JD (Questlight's JD parser), build the Questlight job, create it
    parser_questlight.py, parser_perfox.py     the two resume parsers (switch: RESUME_PARSER)
    questlight.py      fit a parsed resume to Questlight's rules, create the profile
    corrections.py     details a recruiter supplies for what a resume lacked
    matching.py        match open roles (and the scoring both directions use)
    candidates.py      the other direction: rank the candidates already in Questlight for a job
  agent/             the recruiter agent and the MCP tools
    openai_agent.py    the chat page's agent (OpenAI Agents SDK): one turn of the chat
    agent_tools.py     the agent's ~10 tools: short wrappers around tools.py
    prompts.py         the agent's instructions
    mcp_app.py         the MCP server and its token-checked /mcp endpoint
    tools.py           the 14 tools (MCP), and what the agent's tools call
    intakes.py         the short-term memory of the files taken in
    uploads.py         files attached in the chat, kept under private links
    formatting.py, downloads.py, tool_guide.py
  observability/     audit.py (the audit log), tracing.py (per-step timings and tokens)
  static/            the pages: index.html (chat), classic.html, log.html, trace.html
tests/               offline test suites + fixtures (python tests/run_all.py)
docs/                openai-agent.md, architecture diagram
scripts/             measure_chat.py: time, tokens and cost per chat turn of a deployment
run.py               run it locally
```

## Run it locally

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then fill in .env (at least QUESTLIGHT_TOKEN)
python run.py                   # http://127.0.0.1:8000
```

Restart after changing `.env`. Pages: `/` the chat page (type `/tools` to see what the assistant can do), `/classic` the original
one-button page, `/log` the audit log, `/trace` per-step timings and tokens.

To let an outside MCP client reach your machine during development, run a Cloudflare quick tunnel (`tools\cloudflared.exe
tunnel --url http://127.0.0.1:8000`). Through a tunnel only `/mcp` is reachable.

## Settings

All of them are in [.env.example](.env.example), with what each means. The ones that matter first:

| Setting | What for |
|---|---|
| `QUESTLIGHT_TOKEN` | Questlight API token (lasts 10 days). Without it no profile is created and no jobs are read |
| `RESUME_PARSER` | `questlight` (default) or `perfox` (temporary stand-in) |
| `OPENAI_API_KEY` | the chat page's agent (without it the page uploads straight into the pipeline); models and limits in [docs/openai-agent.md](docs/openai-agent.md) |
| `MCP_TOKEN` | the bearer token an MCP client must send to `/mcp` (optional) |
| `APP_PASSWORD` | optional password for the pages and APIs. Not set = open to anyone with the URL (fine for testing) |

## Tests

```
python tests/run_all.py                 # all offline suites, about 2 minutes
python tests/test_audit.py              # or one at a time
RUN_LIVE=1 python tests/run_all.py      # also the live checks against Questlight's dev services (test_live.py)
```

The offline tests never read your `.env`, never touch a real service and never write to your audit log: Questlight, OpenAI's
Responses API (tests/openai_mock.py) and Perfox's parser are small mock servers, the parser is faked, and each script uses a
throwaway database. `test_live.py` is opt-in; it reads the real
parser and job list and sends profiles that Questlight must reject, so nothing is created.

## Deploying to Vercel

The app is a FastAPI `app` in `app/main.py`, which Vercel detects on its own ([pyproject.toml](pyproject.toml) names it
explicitly, [vercel.json](vercel.json) sets the function's maximum duration to 300 s, [.vercelignore](.vercelignore) keeps tests,
docs and secrets out of the upload).

1. Put the project in a Git repository and import it in Vercel (or run `vercel` from this folder). No build command is needed.
   Check that `.env` is **not** committed (it is in `.gitignore`).
2. Under *Settings > Environment Variables* add the settings from `.env.example`: at least `QUESTLIGHT_TOKEN`, `RESUME_PARSER`
   and `OPENAI_API_KEY` (`MCP_TOKEN` only if an MCP client will use `/mcp`). `APP_PASSWORD` is optional: without it the pages are open to
   anyone with the URL, which is fine while testing but means anyone can create profiles in Questlight with your token. Set it
   before sharing the link.
3. Deploy, open the URL, and enter the password (any user name).

**What to know before relying on it.** I couldn't deploy from here, so none of this has been run on Vercel itself; the settings follow
Vercel's documentation for FastAPI and Python functions.

- **Files are kept per instance.** The conversation comes from the page with each message, but the files taken in (and their
  upload links) live in the running process, and Vercel may serve a later request from a different instance, or start a new
  one after a quiet spell. Then the agent says it no longer has the file and it has to be attached again. Fine for a demo with
  one user; for real use these stores need a shared store (Redis or Vercel KV), or a host that keeps one long-lived process.
- **The audit log and traces don't persist.** SQLite can only live in `/tmp` there, per instance, and is lost when the instance
  goes away. `/log`, `/trace` and the agent's "what came in" tools will show only what that instance saw. Real use needs a hosted
  database.
- **Uploads are limited to 4 MB** (a Vercel function can't receive more than 4.5 MB), so the limit there is 4 MB by default
  (`MAX_UPLOAD_MB`).
- **A JD is one long step.** Questlight writes the job's summary and screening questions with AI inside its create call, so
  a JD can take a minute or two end to end. That fits the 300 s function limit (a chat turn stops at 270 s).
- **Questlight's token expires every 10 days.** Update `QUESTLIGHT_TOKEN` and redeploy (environment changes only apply to new
  deployments). A service account or API key from the Questlight team would remove this chore.
- **Vercel's own "Deployment Protection"** (if switched on for the project) would block outside MCP clients from calling `/mcp`.
- **Secrets.** Generate fresh values for production: a new `MCP_TOKEN` and an `APP_PASSWORD`; keep `OPENAI_API_KEY` in Vercel's
  settings only. Candidate data goes to OpenAI to write replies (`store=false`, tracing off): see docs/openai-agent.md.

## More

- [docs/openai-agent.md](docs/openai-agent.md): how the agent works, its settings, privacy, and how to measure it.
- Not in this repository (it is public): `sources/` (the BRD and Questlight's own documents) and `docs/notes.md` (internal notes on
  Questlight and Perfox behaviour, open decisions). They stay with the team's local copy.
