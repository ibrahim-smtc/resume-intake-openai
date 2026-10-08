"""Files attached in the page's chat, kept on this server under private links.

The page posts the file to /api/chat (app/main.py) with the message. The file is kept here for 30 minutes under a random
private link ("intake-file://<token>"), so the tools can open it again later in the conversation (app/agent/downloads.py,
load_file()). Nobody else can: the link only means something inside this process.
"""
import re
import secrets
import time
from collections import OrderedDict
from pathlib import Path

from app import settings

SCHEME = "intake-file://"
FILE_TTL_S = 30 * 60
FILE_MAX = 20  # at most 20 files held at once; the oldest goes first
_LINK_RE = re.compile(re.escape(SCHEME) + r"([0-9a-f]{24})")

_files: "OrderedDict[str, tuple]" = OrderedDict()  # token -> (file name, bytes, time stored)


def store_file(name: str, data: bytes) -> str:
    now = time.monotonic()
    for token in [t for t, (_, _, born) in _files.items() if now - born > FILE_TTL_S]:
        del _files[token]
    token = secrets.token_hex(12)
    _files[token] = (name, data, now)
    while len(_files) > FILE_MAX:
        _files.popitem(last=False)
    return SCHEME + token


def is_link(url: str) -> bool:
    return str(url or "").strip().startswith(SCHEME)


def load_file(url: str):
    """(file name, bytes) for a link made by store_file, or None when it is unknown or expired.
    Tolerates punctuation a model may have added around the link."""
    m = _LINK_RE.search(str(url or ""))
    entry = _files.get(m.group(1)) if m else None
    if entry and time.monotonic() - entry[2] <= FILE_TTL_S:
        return entry[0], entry[1]
    return None


def token_of(text):
    """The 24-character upload token inside an intake-file link (or on its own), else None. A reference to a file may be
    the link, or just its token, where a file_id belongs."""
    match = re.search(r"(?<![0-9a-f])([0-9a-f]{24})(?![0-9a-f])", str(text or ""))
    return match.group(1) if match else None


ACCEPTED = (".pdf", ".docx", ".txt", ".doc")  # resumes are PDF or DOCX; a job description may also be TXT or DOC


def file_problem(name: str, data: bytes):
    if Path(name).suffix.lower() not in ACCEPTED:
        return f"{name}: I can only read PDF, DOCX, TXT or DOC files."
    if not data:
        return f"{name} is empty."
    if len(data) > settings.MAX_UPLOAD_BYTES:
        return f"{name} is larger than {settings.MAX_UPLOAD_LABEL}."
    return None
