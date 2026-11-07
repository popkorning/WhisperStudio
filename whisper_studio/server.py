"""Local HTTP API used by the desktop UI (127.0.0.1 only).

Plain http.server instead of FastAPI/uvicorn: those pull in ~400 modules (pydantic, starlette,
anyio), so on a cold disk the window waited seconds
for the server before it could show anything. Everything here is stdlib + our light modules;
transcription runs in the worker process (jobs.py, worker.py).
"""
from __future__ import annotations

import json
import mimetypes
import re
import secrets
import sys
import threading
import traceback
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import export, jobs, llm, models, options, settings, store
from .config import HOME, WEB_DIR

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("image/svg+xml", ".svg")
mimetypes.add_type("audio/ogg", ".opus")
mimetypes.add_type("audio/ogg", ".oga")
mimetypes.add_type("audio/mp4", ".m4a")
mimetypes.add_type("audio/wav", ".wav")


# Any web page open in a browser may send requests to 127.0.0.1. Only our own window may change
# anything: it gets this token as a SameSite=Strict cookie with index.html, and a cross-site
# request never carries it. A Host other than ours (DNS rebinding) is refused outright.
TOKEN = secrets.token_urlsafe(24)
COOKIE = "ws_token"


class HTTPError(Exception):
    def __init__(self, status: int, detail: str = ""):
        super().__init__(detail)
        self.status, self.detail = status, detail


class FileResult:
    def __init__(self, path: Path, ctype: str | None = None):
        self.path, self.ctype = path, ctype


class Text(str):
    """A plain-text answer (JSON is the default)."""


# ----------------------------------------------------------------- routing

ROUTES: list[tuple[str, re.Pattern, object]] = []


def route(method: str, pattern: str):
    rx = re.compile("^" + re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern) + "$")

    def wrap(fn):
        ROUTES.append((method, rx, fn))
        return fn
    return wrap


class Request:
    def __init__(self, handler: Handler, query: dict):
        self.h, self.query = handler, query
        self._body = None

    def arg(self, k: str, default=None):
        v = self.query.get(k)
        return v[0] if v else default

    @property
    def length(self) -> int:
        return int(self.h.headers.get("Content-Length") or 0)

    def json(self) -> dict:
        if self._body is None:
            raw = self.read()
            self._body = json.loads(raw) if raw.strip() else {}
        return self._body

    def read(self) -> bytes:
        self._body = {}
        return self.h.rfile.read(self.length) if self.length else b""

    def stream(self, size: int = 1 << 20):
        """The body in chunks (uploads); raises if the client sends less than it announced."""
        self._body = {}
        left = self.length
        while left > 0:
            chunk = self.h.rfile.read(min(left, size))
            if not chunk:
                self.h.close_connection = True
                raise HTTPError(400, "Файл передан не полностью")
            left -= len(chunk)
            yield chunk

    def drain(self) -> None:
        """Read an unread body: left in the socket it would break the next keep-alive request.
        A big one (a refused upload) is not worth it: the connection is closed instead."""
        if self._body is None and self.length:
            if self.length > 1 << 20:
                self._body = {}
                self.h.close_connection = True
                return
            self.read()


def _opts(o: dict | None):
    return options.make_options(o)


# ----------------------------------------------------------------- status, settings, models

@route("GET", "/api/status")
def status(_):
    dev, ctype = jobs.device()
    w = jobs.worker_state()
    jobs.kick()
    return dict(device=dev or "cpu", compute_type=ctype or "float32", home=str(HOME),
                current=w["current"], queued=jobs.queued_count(),
                worker=dict(running=w["running"], pid=w["pid"]),
                profiles={k: v["title"] for k, v in options.PROFILES.items()}, llm=llm.status())


@route("GET", "/api/settings")
def get_settings(_):
    return settings.get()


@route("PUT", "/api/settings")
def put_settings(r):
    return settings.update(r.json())


@route("GET", "/api/dictionary")
def get_dictionary(_):
    return settings.dictionary()


@route("PUT", "/api/dictionary")
def put_dictionary(r):
    b = r.json()
    return settings.save_dictionary(b, b.get("rev"))


@route("GET", "/api/models")
def list_models(_):
    return models.status()


@route("POST", "/api/models/{key}/install")
def install_model(_, key):
    if key not in models.MODELS:
        raise HTTPError(404, "unknown model")
    models.install_async(key)
    return {"ok": True}


# ----------------------------------------------------------------- docs

@route("GET", "/api/docs")
def docs(_):
    jobs.kick()
    return store.list_docs()


@route("GET", "/api/search")
def search(r):
    # one hit per document (with the count): the list shows documents, not segments
    return store.search(r.arg("q", ""), limit=int(r.arg("limit", 1000)), per_doc=1)


@route("GET", "/api/docs/{doc_id}")
def doc(_, doc_id):
    return store.load(doc_id)


@route("PATCH", "/api/docs/{doc_id}")
def patch_doc(r, doc_id):
    return store.rename(doc_id, r.json().get("title", ""))


@route("DELETE", "/api/docs/{doc_id}")
def delete_doc(_, doc_id):
    store.delete(doc_id)
    return {"ok": True}


@route("POST", "/api/transcribe")
def transcribe(r):
    body = r.json()
    opts = _opts(body.get("options"))
    out = []
    for p in body.get("paths", []):
        d = options.submit(p, opts)
        jobs.submit_process(d["id"])
        out.append(store.summary(d))
    return out


@route("POST", "/api/upload")
def upload(r):
    """One file per request, the raw bytes as the body: ?name=<file name>&options=<json>."""
    name = Path(r.arg("name") or "audio").name
    try:
        opts = _opts(json.loads(r.arg("options") or "{}"))
    except (ValueError, TypeError):  # broken options: the app defaults rather than a lost upload
        opts = _opts({})
    d = store.create(Path(name).stem, name, {})
    dst = store.path(d["id"]) / f"audio{Path(name).suffix.lower() or '.bin'}"
    try:
        with open(dst, "wb") as fh:
            for chunk in r.stream():
                fh.write(chunk)
    except BaseException:
        store.delete(d["id"])
        raise
    d.update(source_path=str(dst), options=asdict(opts))
    store.save(d, snapshot=False)
    jobs.submit_process(d["id"])
    return [store.summary(d)]


@route("POST", "/api/docs/{doc_id}/retranscribe")
def retranscribe(r, doc_id):
    old = store.load(doc_id)
    src = old["source_path"]
    if not Path(src).exists() and old.get("audio_file"):
        src = str(store.path(doc_id) / old["audio_file"])
    d = options.submit(src, _opts(r.json().get("options")), title=old["title"] + " (новая версия)")
    options.reuse_hypotheses(old, d)
    jobs.submit_process(d["id"])
    return store.summary(d)


@route("POST", "/api/docs/{doc_id}/rediarize")
def rediarize(r, doc_id):
    n = r.json().get("num_speakers")
    n = int(n) if n not in (None, "", "auto") and int(n) > 0 else None
    return jobs.post_job(doc_id, "rediarize", num_speakers=n)


def _not_gold(doc_id: str) -> None:
    if store.is_gold(store.load(doc_id)):
        raise HTTPError(409, store.GOLD_AI)


@route("POST", "/api/docs/{doc_id}/proofread")
def proofread(_, doc_id):
    _not_gold(doc_id)
    if not llm.status(0)["available"]:
        raise HTTPError(503, "Локальная LLM не запущена")
    return jobs.post_job(doc_id, "proofread")


@route("POST", "/api/docs/{doc_id}/apply-dictionary")
def apply_dictionary(_, doc_id):
    d, n = store.apply_dictionary(doc_id)
    return {"doc": d, "changed": n}


@route("POST", "/api/docs/{doc_id}/corrections")
def corrections(r, doc_id):
    _not_gold(doc_id)
    body = r.json()
    d, n = store.apply_corrections(doc_id, body.get("corrections") or {}, ai=body.get("ai", True))
    return {"doc": d, "changed": n}


@route("GET", "/api/docs/{doc_id}/audio")
def audio(_, doc_id):
    d = store.load(doc_id)
    f = store.path(doc_id) / (d.get("audio_file") or "")
    if not d.get("audio_file") or not f.exists():
        raise HTTPError(404, "Аудио не найдено")
    return FileResult(f)


@route("GET", "/api/docs/{doc_id}/peaks")
def peaks(_, doc_id):
    f = store.path(doc_id) / "peaks.json"
    return FileResult(f, "application/json") if f.exists() else []


@route("GET", "/api/docs/{doc_id}/export/{fmt}")
def export_doc(r, doc_id, fmt):
    return Text(export.render(store.load(doc_id), fmt, r.arg("timestamps", "true") != "false"))


# ----------------------------------------------------------------- edits

@route("POST", "/api/docs/{doc_id}/segments/{seg_id}/text")
def edit_text(r, doc_id, seg_id):
    return store.edit_text(doc_id, seg_id, r.json().get("text", ""))


@route("POST", "/api/docs/{doc_id}/segments/{seg_id}/verified")
def set_verified(r, doc_id, seg_id):
    return store.set_verified(doc_id, seg_id, bool(r.json().get("on", True)))


@route("POST", "/api/docs/{doc_id}/segments/{seg_id}/speaker")
def set_speaker(r, doc_id, seg_id):
    b = r.json()
    return store.set_speaker(doc_id, seg_id, b.get("speaker"), b.get("name"))


@route("POST", "/api/docs/{doc_id}/segments/{seg_id}/split")
def split(r, doc_id, seg_id):
    b = r.json()
    return store.split(doc_id, seg_id, int(b["word_index"]), b.get("new_speaker"))


@route("POST", "/api/docs/{doc_id}/segments/{seg_id}/merge")
def merge(_, doc_id, seg_id):
    return store.merge_next(doc_id, seg_id)


@route("POST", "/api/docs/{doc_id}/segments/{seg_id}/keep")
def keep(_, doc_id, seg_id):
    return store.clear_flags(doc_id, seg_id)


@route("DELETE", "/api/docs/{doc_id}/segments/{seg_id}")
def delete_segment(_, doc_id, seg_id):
    return store.delete_segment(doc_id, seg_id)


@route("POST", "/api/docs/{doc_id}/flagged/delete")
def delete_flagged(_, doc_id):
    return store.delete_flagged(doc_id)


@route("POST", "/api/docs/{doc_id}/speakers/{spk}")
def rename_speaker(r, doc_id, spk):
    b = r.json()
    return store.rename_speaker(doc_id, spk, b.get("name", ""), b.get("color"))


@route("POST", "/api/docs/{doc_id}/speakers/{spk}/merge")
def merge_speaker(r, doc_id, spk):
    return store.merge_speakers(doc_id, spk, r.json()["into"])


@route("POST", "/api/docs/{doc_id}/replace")
def replace(r, doc_id):
    b = r.json()
    d, n = store.replace_all(doc_id, b["find"], b.get("replace", ""), b.get("whole_word", True))
    return {"doc": d, "count": n}


@route("POST", "/api/docs/{doc_id}/undo")
def undo(_, doc_id):
    return store.undo(doc_id)


# ----------------------------------------------------------------- gold reference (docs/GOLD.md)

@route("POST", "/api/docs/{doc_id}/gold")
def set_gold(r, doc_id):
    return store.summary(store.set_gold(doc_id, bool(r.json().get("on", True))))


def _gold_eval() -> dict:
    st = jobs.gold_state()
    return dict(running=st["running"], result=jobs.gold_result(), history=jobs.gold_history(), error=st["error"])


@route("GET", "/api/gold")
def gold(_):
    keys = ("id", "title", "duration", "segments", "verified", "verified_seconds", "status")
    docs = [{k: s.get(k) for k in keys} for s in store.list_docs() if s.get("gold")]
    ev = _gold_eval()
    return dict(docs=docs, eval=ev["result"], running=ev["running"], error=ev["error"])


@route("GET", "/api/gold/eval")
def gold_eval(_):
    return _gold_eval()


@route("POST", "/api/gold/eval")
def start_gold_eval(r):
    """In its own CPU process, not in the GPU queue (busy for hours, and no GPU is needed)."""
    jobs.start_gold_eval(bool(r.json().get("tune")))  # already running: nothing to start
    return {"running": True}


# ----------------------------------------------------------------- HTTP plumbing

def _log(msg: str) -> None:
    if sys.stderr:  # None under pythonw
        try:
            sys.stderr.write(msg + "\n")
        except Exception:  # noqa: BLE001
            pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive: WebView2 reuses connections
    server_version = "WhisperStudio"

    def log_message(self, *_):  # the default writes to stderr, which pythonw does not have
        pass

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("GET", head=True)

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")

    _sent = False

    def send_response(self, *args, **kw):
        self._sent = True
        super().send_response(*args, **kw)

    def _dispatch(self, method: str, head: bool = False):
        self._sent = False
        url = urlsplit(self.path)
        path = unquote(url.path)
        req = Request(self, parse_qs(url.query))
        try:
            self._guard(method, path)
            if path.startswith("/api/"):
                for m, rx, fn in ROUTES:
                    hit = rx.match(path)
                    if hit and m == method:
                        result = fn(req, **hit.groupdict())
                        break
                else:
                    raise HTTPError(404, "Not Found")
                req.drain()
                if isinstance(result, FileResult):
                    return self._file(result.path, result.ctype, head, cache=False)
                if isinstance(result, Text):
                    return self._send(200, result.encode("utf-8"), "text/plain; charset=utf-8", head=head)
                body = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                return self._send(200, body, "application/json", head=head)
            if method != "GET":
                raise HTTPError(405, "Method Not Allowed")
            return self._static(path, head)
        except (ConnectionError, TimeoutError):
            self.close_connection = True
        except Exception as e:  # noqa: BLE001
            try:
                req.drain()
            except Exception:  # noqa: BLE001 - the client went away mid-body
                self.close_connection = True
            self._error(e, head)

    def _guard(self, method: str, path: str) -> None:
        port = self.server.server_address[1]
        if (self.headers.get("Host") or "").lower() not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            raise HTTPError(403, "Чужой адрес сервера")
        if method not in ("GET", "HEAD") and path.startswith("/api/"):
            jar = self.headers.get("Cookie") or ""
            sent = next((c.split("=", 1)[1] for c in (x.strip() for x in jar.split(";"))
                         if c.startswith(COOKIE + "=")), "")
            if not secrets.compare_digest(sent, TOKEN):
                raise HTTPError(403, "Запрос не из окна приложения: перезапустите приложение")

    def _error(self, e: Exception, head: bool):
        if self._sent:  # headers (and maybe part of a file) are out: just drop the connection
            self.close_connection = True
            _log(f"error after headers: {e!r}")
            return
        if isinstance(e, HTTPError):
            code, detail = e.status, e.detail
        elif isinstance(e, (jobs.Busy, store.Refused, settings.Conflict)):
            code, detail = 409, str(e)
        elif isinstance(e, KeyError):
            code, detail = 404, e.args[0] if e.args else str(e)
        elif isinstance(e, FileNotFoundError):
            code, detail = 404, "Запись не найдена" if str(e).endswith("doc.json'") else str(e)
        elif isinstance(e, (ValueError, json.JSONDecodeError)):
            code, detail = 400, str(e)
        else:
            code, detail = 500, repr(e)
            _log(traceback.format_exc())
        try:
            self._send(code, json.dumps({"detail": str(detail)}, ensure_ascii=False).encode("utf-8"),
                       "application/json", head=head,
                       headers={"Connection": "close"} if self.close_connection else None)
        except (ConnectionError, TimeoutError):
            pass

    def _send(self, code: int, body: bytes, ctype: str, head: bool = False, headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _static(self, path: str, head: bool):
        rel = path.lstrip("/") or "index.html"
        f = (WEB_DIR / rel).resolve()
        if not f.is_relative_to(WEB_DIR.resolve()) or not f.is_file():
            raise HTTPError(404, "Not Found")
        # WebView2 caches static files heuristically; make it revalidate (cheap 304 via ETag),
        # so an updated UI is picked up without a stale index.html / icons.svg.
        self._file(f, None, head, cache=True)

    def _file(self, f: Path, ctype: str | None, head: bool, cache: bool):
        st = f.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        ctype = ctype or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "text/javascript"):
            ctype += "; charset=utf-8"
        base = {"ETag": etag, "Accept-Ranges": "bytes"}
        if cache:
            base["Cache-Control"] = "no-cache"
        if f.name == "index.html":
            base["Set-Cookie"] = f"{COOKIE}={TOKEN}; Path=/; HttpOnly; SameSite=Strict"
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            for k, v in base.items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        size, start, end = st.st_size, 0, st.st_size - 1
        code = 200
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if rng and size:
            a, b = rng.groups()
            if a:
                start, end = int(a), min(int(b), size - 1) if b else size - 1
            elif b:  # the last b bytes
                start = max(0, size - int(b))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            code = 206
            base["Content-Range"] = f"bytes {start}-{end}/{size}"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        for k, v in base.items():
            self.send_header(k, v)
        self.end_headers()
        if head:
            return
        with open(f, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = fh.read(min(left, 256 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # Windows: True would let two servers share a port


PORTS = (47811, 47812, 47813, 47814)


def make_server(port: int | None = None) -> Server:
    """Bound and listening; call serve_forever() (in a thread) to answer.
    No port given: one of PORTS, so the page keeps its origin between launches - WebView2
    keeps localStorage (marks «Верно», the last document, panel state, theme) per origin,
    and a random port forgot all of it every time. Any free port as the last resort."""
    httpd = None
    for p in (port,) if port is not None else (*PORTS, 0):
        try:
            httpd = Server(("127.0.0.1", p), Handler)
            break
        except OSError:
            if port is not None or p == 0:
                raise
    assert httpd is not None
    threading.Thread(target=_startup, daemon=True).start()
    return httpd


def _startup():
    try:
        jobs.ensure_worker()  # jobs left from the last session
        store.list_docs()     # warm the library index before the UI asks for it
    except Exception:  # noqa: BLE001
        _log(traceback.format_exc())


def main(port: int = 8765):
    httpd = make_server(port)
    print(f"http://127.0.0.1:{httpd.server_address[1]}/")
    httpd.serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
