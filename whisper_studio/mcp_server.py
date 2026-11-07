"""MCP server (stdio) so AI agents can transcribe, proofread and edit transcripts.

A small JSON-RPC loop on the standard library instead of the MCP SDK: the SDK pulls in
~700 modules (HTTP transports, pydantic, uvicorn), so the server needed seconds to answer `initialize` on a cold disk. This one imports
about a dozen files of its own.

Jobs go into the shared queue (jobs.py) and run in the worker process, which starts on
demand whether or not the desktop app is open, so this process never loads models.
Transcripts are shared with the app, and options left out fall back to the app's defaults,
so an agent sees exactly the same transcript as the user.

Answers are written for a language model's context: compact text, long transcripts
in pages (from_segment), variants of other models as diffs only.
"""
from __future__ import annotations

import inspect
import json
import sys
import threading
import time
import traceback
import types
import typing
from pathlib import Path

from . import export, jobs, llm, settings, store

NAME, VERSION = "whisper-studio", "2.0"
PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")

INSTRUCTIONS = (
    "Local high-accuracy speech-to-text for the user's recordings (mostly Russian phone calls and "
    "conversations with German terms), shared with the desktop app «Транскрибация».\n"
    "Workflow:\n"
    "1. transcribe_audio(path) first looks for an existing transcript of that file (the user may have "
    "corrected it by hand) and returns it instead of transcribing again; force=true makes a new one. "
    "find_transcripts(name) finds transcripts by file name or title, search_transcripts(text) by content.\n"
    "2. Leave language, model, profile and num_speakers unset unless the user asked otherwise — the "
    "app defaults are tuned for these recordings. Pass recording-specific names in glossary.\n"
    "3. ALWAYS proofread afterwards: proofread_pack(id), fix recognition errors following its "
    "instructions (German words written in Cyrillic, names, mishearings, repetition loops), then send "
    "only changed segments with apply_corrections. Long transcripts come in pages: continue with "
    "from_segment as the answer says. Never paraphrase or translate.\n"
    "4. If speakers look wrong (more than really talk), call rediarize with the right count.\n"
    "5. New names or terms you learned → update_dictionary, so future transcripts get them right.\n"
    "Transcript ids: the full id or its last 6 characters. «неразборчиво»: the models heard different "
    "things there (noise or mumbling) - real speech, but check it against the variants. ⚠ marks may be Whisper "
    "hallucinations on silence or music; don't treat them as facts. undo reverts the last edit.\n"
    "Transcripts marked «эталон» are gold references the owner checked by ear: never proofread or "
    "edit their text (the tools refuse).")


class ToolError(Exception):
    """A readable error for the agent (isError result, no traceback)."""


# ------------------------------------------------------------------ tool registry

TOOLS: dict[str, dict] = {}


def _schema(tp) -> dict:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        return _schema(rest[0]) if len(rest) == 1 else {}
    if tp is str:
        return {"type": "string"}
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if origin is list:
        return {"type": "array", "items": _schema(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object", "additionalProperties": _schema(args[1]) if len(args) > 1 else {}}
    return {}


def tool(**params: str):
    """Register a function as a tool. params: descriptions of its arguments."""
    def wrap(fn):
        hints = typing.get_type_hints(fn)
        props, required = {}, []
        for name, p in inspect.signature(fn).parameters.items():
            if name == "ctx":
                continue
            s = _schema(hints.get(name, str))
            if name in params:
                s["description"] = params[name]
            if p.default is inspect.Parameter.empty:
                required.append(name)
            elif p.default is not None:
                s["default"] = p.default
            props[name] = s
        doc = inspect.cleandoc(fn.__doc__ or "")
        TOOLS[fn.__name__] = dict(fn=fn, spec=dict(
            name=fn.__name__, description=doc,
            inputSchema={"type": "object", "properties": props, "required": required}))
        return fn
    return wrap


def _coerce(props: dict, args: dict) -> dict:
    """Clients sometimes send numbers and booleans as strings ("300", "true")."""
    out = {}
    for k, v in args.items():
        kind = props.get(k, {}).get("type")
        try:
            if kind == "integer" and isinstance(v, (str, float)) and str(v).strip():
                v = int(float(v))
            elif kind == "number" and isinstance(v, str) and v.strip():
                v = float(v)
            elif kind == "boolean" and isinstance(v, str):
                v = v.strip().lower() in ("true", "1", "yes", "да")
            elif kind == "array" and isinstance(v, str):
                v = json.loads(v) if v.strip().startswith("[") else [v]
            elif kind == "object" and isinstance(v, str):
                v = json.loads(v)
        except ValueError:
            raise ToolError(f"{k}: ожидается {kind}, получено {v!r}") from None
        out[k] = v
    return out


class Context:
    """Progress and cancellation of one tools/call."""

    def __init__(self, server: Server, token):
        self.server, self.token, self.cancelled = server, token, False
        self.last = -1.0

    def progress(self, value: float, message: str = "") -> None:
        """MCP wants progress to grow: a new stage at the same value is not sent."""
        value = round(float(value), 3)
        if self.token is not None and value > self.last:
            self.last = value
            self.server.notify("notifications/progress", dict(
                progressToken=self.token, progress=value, total=1.0, message=message or ""))


# ------------------------------------------------------------------ helpers

def _doc_id(ref: str) -> str:
    """Full id, its random tail ("0f1e2d"), or an exact title."""
    ref = (ref or "").strip()
    try:
        if (store.path(ref) / "doc.json").exists():
            return ref
    except ValueError:
        pass
    docs = store.list_docs()
    hits = [d for d in docs if d["id"].endswith(ref)] or [d for d in docs if d.get("title") == ref]
    if len(hits) == 1:
        return hits[0]["id"]
    if not hits:
        raise ToolError(f"Расшифровка «{ref}» не найдена. Найдите id через find_transcripts или list_transcripts.")
    raise ToolError(f"«{ref}» подходит к нескольким расшифровкам: " + ", ".join(d["id"] for d in hits[:8]))


def _load(ref: str) -> dict:
    return store.load(_doc_id(ref))


GOLD_REFUSED = ("Это эталон: текст, выверенный владельцем на слух, — мерило качества распознавания. "
                "ИИ-вычитка и любые правки текста через агента для него отключены (правка агента — не на слух). "
                "Можно читать, менять спикеров и отменять (undo).")


def _writable(ref: str) -> str:
    """Id of a transcript whose text an agent may change: not a gold reference."""
    doc_id = _doc_id(ref)
    if store.is_gold(store.load(doc_id)):
        raise ToolError(GOLD_REFUSED)
    return doc_id


def _clock(t) -> str:
    return export.clock(t or 0)


def _summary_line(d: dict) -> str:
    bits = [d["id"], f"«{d.get('title')}»", _clock(d.get("duration"))]
    if d.get("language"):
        bits.append(d["language"])
    if d.get("speakers"):
        bits.append(f"{d['speakers']} сп.")
    from .models import short_name

    if d.get("model"):
        bits.append(short_name(d["model"]))
    st = d.get("status")
    if st == "done":
        bits.append("исправлена" if d.get("edited") else "готова")
    elif st == "error":
        bits.append(f"ошибка: {d.get('error')}")
    else:
        bits.append(f"{d.get('stage') or st} {round((d.get('progress') or 0) * 100)}%")
    if d.get("gold"):
        bits.append("эталон: текст не менять")
    bits.append(f"создана {(d.get('created') or '')[:16].replace('T', ' ')}")
    if d.get("source"):
        bits.append(f"файл: {d['source']}")
    return " · ".join(bits)


def _flag_note(s: dict) -> str:
    flags = s.get("flags") or []
    out = " ⚠" + ",".join(f for f in flags if f != "disagree") if any(f != "disagree" for f in flags) else ""
    return out + (" (неразборчиво: модели услышали разное)" if "disagree" in flags else "")


def _seg_line(doc: dict, s: dict, mark_uncertain: bool = False) -> str:
    sp = store.speaker_name(doc, s.get("speaker"))
    flag = _flag_note(s)
    text = llm.mark(s) if mark_uncertain else s["text"]
    return f"{s['id']} [{_clock(s['start'])}–{_clock(s['end'])}]{' ' + sp + ':' if sp else ''} {text}{flag}"


def _head(d: dict) -> str:
    from .models import short_name

    sp = ", ".join(f"{k}={v['name']}" for k, v in (d.get("speakers") or {}).items()) or "—"
    return (f"id: {d['id']}\ntitle: {d.get('title')}\nsource: {d.get('source_path')}\n"
            f"language: {d.get('language')} · duration: {_clock(d.get('duration'))} · "
            f"model: {short_name((d.get('meta') or {}).get('model'))} · segments: {len(d['segments'])}\n"
            f"speakers: {sp}\nedited: {bool(d.get('edited'))}\n\n")


def _page(doc: dict, from_segment: str | None, start_time: float | None, end_time: float | None,
          max_chars: int, line) -> tuple[str, str | None]:
    """Segment lines from from_segment / start_time up to max_chars; returns (text, next id)."""
    segs = doc["segments"]
    i0 = 0
    if from_segment:
        i0 = next((i for i, s in enumerate(segs) if s["id"] == from_segment), -1)
        if i0 < 0:
            raise ToolError(f"Реплика {from_segment} не найдена")
    elif start_time is not None:
        i0 = next((i for i, s in enumerate(segs) if s["end"] > start_time), len(segs))
    out, size = [], 0
    for s in segs[i0:]:
        if end_time is not None and s["start"] >= end_time:
            return "\n".join(out), None
        ln = line(s)
        if out and size + len(ln) > max_chars:
            return "\n".join(out), s["id"]
        out.append(ln)
        size += len(ln) + 1
    return "\n".join(out), None


def _wait(ctx: Context | None, doc_id: str, timeout: float | None = None) -> dict:
    t0 = time.monotonic()
    last = None
    while True:
        d = store.load(doc_id)
        if d["status"] in ("done", "error"):
            return d
        if ctx:
            if ctx.cancelled:
                raise ToolError(f"Ожидание отменено; задача {doc_id} продолжается в фоне (get_status).")
            key = (d.get("progress"), d.get("stage"))
            if key != last:
                ctx.progress(d.get("progress") or 0, d.get("stage") or "")
                last = key
        if timeout is not None and time.monotonic() - t0 > timeout:
            return d
        jobs.kick(30)
        time.sleep(1.5)


def _start(path: str, o: dict, reuse: dict | None = None) -> str:
    """Queue a file; `reuse`: an earlier transcript of it whose model hypotheses can be reused."""
    from . import options

    p = Path(path).expanduser()
    if not p.exists():
        raise ToolError(f"Файл не найден: {p}")
    doc = options.submit(p, options.make_options(o))
    if reuse:
        options.reuse_hypotheses(store.load(reuse["id"]), doc)
    jobs.submit_process(doc["id"])
    return doc["id"]


def _options(**kw) -> dict:
    """Only what the agent actually set; the rest comes from the app defaults."""
    o = {k: v for k, v in kw.items() if v not in (None, "")}
    if o.get("num_speakers") == 0:
        o["num_speakers"] = "auto"
    return o


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.samefile(b)
    except OSError:
        return False


def _existing(path: str) -> list[dict]:
    """Transcripts of this very file, best first: done and edited, then newest.
    The same path (or the same file under another path); a file with the same name elsewhere
    only if its size matches the copy kept in the library (a moved recording)."""
    p = Path(path).expanduser()
    want = str(p.resolve()).lower()
    try:
        size = p.stat().st_size
    except OSError:
        size = None
    out = []
    for d in store.list_docs():
        src = d.get("source") or ""
        if not src:
            continue
        if src.lower() == want or _same_file(Path(src), p):
            out.append(d)
            continue
        if size is None or Path(src).name.lower() != p.name.lower() or Path(src).name.lower().startswith("audio."):
            continue
        try:
            doc = store.load(d["id"])
            copy = store.path(d["id"]) / (doc.get("audio_file") or "")
            if doc.get("audio_file") and copy.stat().st_size == size:
                out.append(d)
        except (OSError, ValueError):
            pass
    return sorted(out, key=lambda d: (d["status"] == "done", bool(d.get("edited")), d.get("created") or ""),
                  reverse=True)


def _transcript_text(d: dict, max_chars: int = 20000) -> str:
    body, nxt = _page(d, None, None, None, max_chars, lambda s: _seg_line(d, s))
    more = f"\n… дальше: get_transcript(\"{d['id']}\", from_segment=\"{nxt}\")" if nxt else ""
    return _head(d) + body + more


# ------------------------------------------------------------------ tools: transcription

_OPT = dict(
    language="\"ru\", \"de\", \"en\"… or \"auto\". Leave unset: the app default is tuned for these recordings.",
    model="Model key from list_models. Leave unset.",
    profile="balanced | clean | noisy. Leave unset.",
    num_speakers="Exact speaker count, 0 = detect. Leave unset unless known.",
    glossary="Names and terms of this recording, comma separated (the project dictionary is added automatically).",
    topic="One sentence about the recording (context for recognition).",
    mixed="Russian speech with German words (default on).",
)


@tool(path="Absolute path of the audio/video file.",
      force="Transcribe again even if a transcript of this file exists.",
      wait_seconds="How long to wait for the result (default: until done). On timeout the id is returned; "
                   "continue with get_status.", **_OPT)
def transcribe_audio(path: str, ctx: Context, force: bool = False, wait_seconds: int | None = None,
                     language: str | None = None, model: str | None = None, profile: str | None = None,
                     num_speakers: int | None = None, glossary: str = "", topic: str = "",
                     mixed: bool | None = None) -> str:
    """Transcribe an audio/video file (ogg, mp3, wav, m4a, opus, mp4…) and return the transcript.
    If the file was transcribed before, returns that transcript (with the user's edits) right away
    instead of transcribing again; force=true makes a new one. Long recordings take minutes."""
    if not force:
        for d in _existing(path):
            if d["status"] == "done":
                return ("Эта запись уже расшифрована (возможно, исправлена вручную) — возвращаю её; "
                        "force=true распознает заново.\n\n" + _transcript_text(store.load(d["id"])))
            if d["status"] in ("queued", "processing"):
                doc = _wait(ctx, d["id"], wait_seconds)
                break
        else:
            doc = None
    else:
        doc = None
    if doc is None:
        o = _options(language=language, model=model, profile=profile, num_speakers=num_speakers,
                     glossary=glossary, topic=topic, mixed=mixed)
        prev = next((d for d in _existing(path) if d["status"] == "done"), None) if force else None
        doc = _wait(ctx, _start(path, o, reuse=prev), wait_seconds)
    if doc["status"] == "error":
        raise ToolError(f"Ошибка транскрипции {doc['id']}: {doc.get('error')}")
    if doc["status"] != "done":
        return (f"Ещё идёт: {doc['id']} — {doc.get('stage')} {round((doc.get('progress') or 0) * 100)}%. "
                f"Проверяйте get_status(\"{doc['id']}\").")
    return _transcript_text(doc) + "\n\nДальше: proofread_pack → apply_corrections."


@tool(paths="Absolute paths of the files.", force="Queue even files that already have a transcript.", **_OPT)
def start_transcription(paths: list[str], force: bool = False, language: str | None = None,
                        model: str | None = None, profile: str | None = None, num_speakers: int | None = None,
                        glossary: str = "", topic: str = "", mixed: bool | None = None) -> str:
    """Queue one or many files without waiting (processed one by one on the GPU).
    Files that already have a transcript are not queued again unless force=true.
    Returns one line per file with the transcript id; follow with get_status."""
    o = _options(language=language, model=model, profile=profile, num_speakers=num_speakers,
                 glossary=glossary, topic=topic, mixed=mixed)
    out = []
    for p in paths:
        old = [] if force else _existing(p)
        if old:
            out.append(f"{p} → уже есть: {_summary_line(old[0])}")
            continue
        try:
            out.append(f"{p} → {_start(p, o)} в очереди")
        except ToolError as e:
            out.append(f"{p} → {e}")
    return "\n".join(out)


@tool(transcript_id="Transcript id (or its last 6 characters). Leave empty for the whole queue.")
def get_status(transcript_id: str = "") -> str:
    """Status of a transcript (queued / processing with stage and % / done / error), or of the
    queue and the worker when no id is given."""
    if transcript_id:
        d = _load(transcript_id)
        if d["status"] in ("queued", "processing"):
            jobs.kick()  # restart the worker if it died with jobs left
        return _summary_line(store.summary(d))
    w = jobs.worker_state()
    busy = [d for d in store.list_docs() if d["status"] in ("queued", "processing")]
    head = (f"воркер: {'работает' if w['running'] else 'не запущен'}"
            f"{', сейчас ' + w['current'] if w.get('current') else ''}; в очереди: {len(busy)}")
    if busy:
        jobs.kick()
    return "\n".join([head] + [_summary_line(d) for d in busy])


# ------------------------------------------------------------------ tools: library

@tool(limit="How many (newest first).")
def list_transcripts(limit: int = 30) -> str:
    """Transcripts in the library, newest first: id · title · duration · language · speakers ·
    model · state · created · source file."""
    limit = max(1, limit)
    docs = store.list_docs()
    lines = [_summary_line(d) for d in docs[:limit]]
    if len(docs) > limit:
        lines.append(f"… и ещё {len(docs) - limit} (limit=…)")
    return "\n".join(lines) or "Библиотека пуста"


@tool(query="File name (with or without folder and extension) or part of the title.")
def find_transcripts(query: str) -> str:
    """Transcripts whose source file or title matches query, e.g. a file name. Use it before
    transcribing: the user may already have a (hand-corrected) transcript."""
    q = Path(query).name.lower() or query.lower()
    q = q.rsplit(".", 1)[0] if "." in q else q
    hits = [d for d in store.list_docs()
            if q in (d.get("source") or "").lower() or q in (d.get("title") or "").lower()]
    return "\n".join(_summary_line(d) for d in hits) or f"Ничего не найдено по «{query}». Поиск по тексту: search_transcripts."


@tool(query="Words to find (case-insensitive, ё = е).", transcript_id="Only in this transcript.",
      limit="Maximum number of hits.")
def search_transcripts(query: str, transcript_id: str = "", limit: int = 40) -> str:
    """Full-text search across all transcripts (or one). One line per hit:
    transcript id · title · segment id · time · speaker: text."""
    ids = [_doc_id(transcript_id)] if transcript_id else None
    limit = max(1, limit)
    hits = store.search(query, limit=limit + 1, doc_ids=ids)
    lines = [f"{h['doc_id']} «{h['title']}» " + (f"{h['segment_id']} [{_clock(h['start'])}] {h['speaker']}: {h['text']}"
                                                 if h["segment_id"] else "(совпало название / файл)")
             for h in hits[:limit]]
    if len(hits) > limit:
        lines.append(f"… есть ещё (limit={limit})")
    return "\n".join(lines) or f"«{query}» не найдено"


@tool(transcript_id="Transcript id (or its last 6 characters).",
      format="segments (ids, times, speakers — best for editing) | txt | md | srt | vtt | json",
      from_segment="Start from this segment id (paging).", start_time="Start at this second.",
      end_time="Stop at this second.", max_chars="Page size for format=segments.",
      mark_uncertain="Wrap low-confidence words in ⟨…⟩.", include_words="json: keep word timings.")
def get_transcript(transcript_id: str, format: str = "segments", from_segment: str = "",
                   start_time: float | None = None, end_time: float | None = None, max_chars: int = 30000,
                   mark_uncertain: bool = False, include_words: bool = False) -> str:
    """A transcript exactly as the user sees it in the app (with their edits). format=segments
    comes in pages of max_chars; the answer names the next from_segment."""
    d = _load(transcript_id)
    if d["status"] != "done":
        return _summary_line(store.summary(d))
    if format == "segments":
        head = _head(d) if not from_segment and start_time is None else ""
        body, nxt = _page(d, from_segment or None, start_time, end_time, max_chars,
                          lambda s: _seg_line(d, s, mark_uncertain))
        more = f"\n… дальше: from_segment=\"{nxt}\"" if nxt else ""
        return head + body + more
    if format == "json":
        d = {**d, "meta": {k: v for k, v in (d.get("meta") or {}).items() if "traceback" not in k}}
        if not include_words:
            d["segments"] = [{k: v for k, v in s.items() if k not in ("words", "alts")} for s in d["segments"]]
    if format not in export.FORMATS:
        raise ToolError(f"Формат {format}: нужен один из segments, {', '.join(export.FORMATS)}")
    return export.render(d, format)


# ------------------------------------------------------------------ tools: proofreading and edits

@tool(transcript_id="Transcript id (or its last 6 characters).",
      from_segment="Continue from this segment (the previous page names it).", max_chars="Page size.")
def proofread_pack(transcript_id: str, from_segment: str = "", max_chars: int = 24000) -> str:
    """Instructions + project dictionary + the transcript with uncertain words marked ⟨…⟩ and
    what other models heard differently. Correct it yourself, then call apply_corrections.
    Long transcripts come in pages."""
    d = store.load(_writable(transcript_id))
    if d["status"] != "done":
        raise ToolError(f"Расшифровка ещё не готова: {_summary_line(store.summary(d))}")
    return llm.proofread_pack(d, from_segment or None, max_chars)


@tool(transcript_id="Transcript id (or its last 6 characters).",
      corrections="{segment_id: full corrected text of that segment}, only changed segments.")
def apply_corrections(transcript_id: str, corrections: dict[str, str]) -> str:
    """Apply proofreading. One undo step in the app; word timings are re-mapped; the segments
    get an AI mark (✦) and keep their original text."""
    doc_id = _writable(transcript_id)
    doc = store.load(doc_id)
    known = {s["id"]: s for s in doc["segments"]}
    unknown = [k for k in corrections if k not in known]
    _, n = store.apply_corrections(doc_id, {k: v for k, v in corrections.items() if k in known}, ai=True)
    msg = f"исправлено реплик: {n}"
    if unknown:
        msg += f"; не найдены: {', '.join(unknown)}"
    if n < len(corrections) - len(unknown):
        msg += f"; без изменений: {len(corrections) - len(unknown) - n}"
    return msg


@tool(transcript_id="Transcript id (or its last 6 characters).",
      num_speakers="Exact count, or 0 = detect automatically.")
def rediarize(transcript_id: str, ctx: Context, num_speakers: int = 2) -> str:
    """Recompute speakers of an existing transcript (text and edits are kept; voices are cached,
    so it takes seconds). Returns the transcript with the new speakers."""
    doc_id = _doc_id(transcript_id)
    try:
        jobs.post_job(doc_id, "rediarize", num_speakers=num_speakers or None)
    except jobs.Busy as e:
        raise ToolError(str(e)) from None
    d = _wait(ctx, doc_id)
    if d.get("error"):
        raise ToolError(f"Ошибка: {d['error']}")
    return _transcript_text(d)


@tool(transcript_id="Transcript id (or its last 6 characters).", segment_id="g12…",
      text="New full text of the segment; empty deletes it.")
def edit_segment(transcript_id: str, segment_id: str, text: str) -> str:
    """Replace the text of one segment (word timings are re-mapped). Empty text deletes it."""
    d = store.edit_text(_writable(transcript_id), segment_id, text, by="ai")
    s = next((x for x in d["segments"] if x["id"] == segment_id), None)
    return _seg_line(d, s) if s else f"{segment_id} удалена"


@tool(transcript_id="Transcript id (or its last 6 characters).", find="Text to find.",
      replace="Replacement.", whole_word="Match whole words only.")
def replace_text(transcript_id: str, find: str, replace: str, whole_word: bool = True) -> str:
    """Find & replace across the whole transcript (e.g. fix a misspelled name everywhere)."""
    _, n = store.replace_all(_writable(transcript_id), find, replace, whole_word, by="ai")
    return f"заменено: {n}"


@tool(transcript_id="Transcript id (or its last 6 characters).", speaker_id="S1, S2…", name="Real name.")
def rename_speaker(transcript_id: str, speaker_id: str, name: str) -> str:
    """Give a speaker (S1, S2, …) a real name."""
    d = store.rename_speaker(_doc_id(transcript_id), speaker_id, name)
    return ", ".join(f"{k}={v['name']}" for k, v in d["speakers"].items())


@tool(transcript_id="Transcript id (or its last 6 characters).", segment_id="g12…",
      speaker_id="Existing speaker S1, S2…, or empty to create a new one.")
def set_segment_speaker(transcript_id: str, segment_id: str, speaker_id: str) -> str:
    """Assign a segment to another speaker."""
    d = store.set_speaker(_doc_id(transcript_id), segment_id, speaker_id or None)
    s = next(x for x in d["segments"] if x["id"] == segment_id)
    return _seg_line(d, s)


@tool(transcript_id="Transcript id (or its last 6 characters).")
def undo(transcript_id: str) -> str:
    """Revert the last change of a transcript (the same undo as Ctrl+Z in the app)."""
    store.undo(_doc_id(transcript_id))
    return "отменено"


# ------------------------------------------------------------------ tools: dictionary, export, models

@tool()
def get_dictionary() -> str:
    """Project dictionary: terms (go into the recognition prompt), replacement rules
    (recognized → correct, a trailing * matches any ending) and a context paragraph."""
    return json.dumps(settings.dictionary(), ensure_ascii=False)


@tool(add_terms="Terms and names to add.", add_replacements="{as recognized: correct}; a trailing * "
      "matches any ending.", context="Replace the context paragraph (who talks about what).")
def update_dictionary(add_terms: list[str] | None = None, add_replacements: dict[str, str] | None = None,
                      context: str | None = None) -> str:
    """Add terms and replacement rules to the project dictionary, optionally replace the context
    paragraph. Existing entries are kept. Future transcripts use them automatically."""
    added_t, added_r = [], []

    def change(d):
        added_t[:] = [t for t in add_terms or [] if t not in d["terms"]]
        d["terms"] += added_t
        known = {r["from"].lower() for r in d["replace"]}
        added_r[:] = [{"from": k, "to": v} for k, v in (add_replacements or {}).items()
                      if k.lower() not in known]
        d["replace"] += added_r
        if context is not None:
            d["context"] = context

    d = settings.edit_dictionary(change)
    return (f"добавлено терминов: {len(added_t)}, замен: {len(added_r)}; "
            f"всего терминов {len(d['terms'])}, замен {len(d['replace'])}")


@tool(transcript_id="Transcript id (or its last 6 characters).", format="txt | md | srt | vtt | json",
      output_path="Absolute path of the file to write.")
def export_transcript(transcript_id: str, format: str, output_path: str) -> str:
    """Write the transcript to a file. Returns the path."""
    if format not in export.FORMATS:
        raise ToolError(f"Формат {format}: нужен один из {', '.join(export.FORMATS)}")
    Path(output_path).write_text(export.render(_load(transcript_id), format), "utf-8")
    return output_path


@tool()
def list_models() -> str:
    """Speech models: installed or not, languages, size."""
    from . import models

    return "\n".join(f"{m['key']} · {m['short']} · {'установлена' if m['installed'] else 'нет'} · "
                     f"{','.join(m['langs'])} · {m['size_gb']} ГБ · {m['title']}" for m in models.status())


# ------------------------------------------------------------------ JSON-RPC over stdio

class Server:
    def __init__(self, rx, tx):
        self.rx, self.tx = rx, tx
        self.out_lock = threading.Lock()
        self.calls: dict = {}  # request id -> Context

    def send(self, msg: dict) -> None:
        data = json.dumps(msg, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        with self.out_lock:
            self.tx.write(data)
            self.tx.flush()

    def notify(self, method: str, params: dict) -> None:
        self.send(dict(jsonrpc="2.0", method=method, params=params))

    def reply(self, rid, result=None, error: tuple[int, str] | None = None) -> None:
        msg = dict(jsonrpc="2.0", id=rid)
        if error:
            msg["error"] = dict(code=error[0], message=error[1])
        else:
            msg["result"] = result
        self.send(msg)

    def run(self) -> None:
        while True:
            line = self.rx.readline()
            if not line:
                return
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.reply(None, error=(-32700, "Parse error"))
                continue
            for m in msg if isinstance(msg, list) else [msg]:
                try:
                    self.handle(m)
                except Exception as e:  # noqa: BLE001 - never let one message stop the server
                    log(f"handle failed: {e!r}")
                    if isinstance(m, dict) and "id" in m:
                        self.reply(m["id"], error=(-32603, str(e)))

    def handle(self, m: dict) -> None:
        method, rid, params = m.get("method"), m.get("id"), m.get("params") or {}
        if method is None:  # a response to something we never ask
            return
        if rid is None:  # notification
            if method == "notifications/cancelled":
                ctx = self.calls.get(params.get("requestId"))
                if ctx:
                    ctx.cancelled = True
            return
        if method == "initialize":
            want = params.get("protocolVersion")
            self.reply(rid, dict(
                protocolVersion=want if want in PROTOCOLS else PROTOCOLS[-1],
                capabilities=dict(tools=dict(listChanged=False)),
                serverInfo=dict(name=NAME, version=VERSION), instructions=INSTRUCTIONS))
        elif method == "ping":
            self.reply(rid, {})
        elif method == "tools/list":
            self.reply(rid, dict(tools=[t["spec"] for t in TOOLS.values()]))
        elif method == "tools/call":
            ctx = Context(self, (params.get("_meta") or {}).get("progressToken"))
            self.calls[rid] = ctx
            threading.Thread(target=self.call, args=(rid, params, ctx), daemon=True).start()
        elif method in ("resources/list", "resources/templates/list", "prompts/list"):
            key = {"resources/list": "resources", "resources/templates/list": "resourceTemplates",
                   "prompts/list": "prompts"}[method]
            self.reply(rid, {key: []})
        else:
            self.reply(rid, error=(-32601, f"Method not found: {method}"))

    def call(self, rid, params: dict, ctx: Context) -> None:
        name, args = params.get("name"), params.get("arguments") or {}
        t = TOOLS.get(name)
        try:
            if not t:
                raise ToolError(f"Нет инструмента {name}")
            fn = t["fn"]
            sig = inspect.signature(fn).parameters
            extra = [k for k in args if k not in sig]
            if extra:
                raise ToolError(f"{name}: неизвестные параметры {', '.join(extra)}; есть: "
                                + ", ".join(k for k in sig if k != "ctx"))
            args = _coerce(t["spec"]["inputSchema"]["properties"], args)
            if "ctx" in sig:
                args = {**args, "ctx": ctx}
            t0 = time.monotonic()
            text = fn(**args)
            log(f"{name} {time.monotonic() - t0:.2f}s {len(text)} chars")
            result = dict(content=[dict(type="text", text=text)], isError=False)
        except ToolError as e:
            result = dict(content=[dict(type="text", text=str(e))], isError=True)
        except TypeError as e:
            result = dict(content=[dict(type="text", text=f"{name}: {e}")], isError=True)
        except (KeyError, ValueError, FileNotFoundError, PermissionError) as e:
            msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            result = dict(content=[dict(type="text", text=f"Ошибка: {msg}")], isError=True)
        except Exception as e:  # noqa: BLE001
            log(traceback.format_exc())
            result = dict(content=[dict(type="text", text=f"Ошибка: {e!r}")], isError=True)
        finally:
            self.calls.pop(rid, None)
        if not ctx.cancelled:
            self.reply(rid, result)


def log(msg: str) -> None:
    sys.stderr.write(f"[{NAME}] {msg}\n")
    sys.stderr.flush()


def main():
    rx, tx = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # stdout belongs to the protocol: a stray print() must not break it
    Server(rx, tx).run()


if __name__ == "__main__":
    main()
