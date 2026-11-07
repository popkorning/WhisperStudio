"""User settings and the project dictionary (shared by UI, CLI and MCP).

settings.json   — defaults for new transcriptions and the LLM endpoint for proofreading.
dictionary.json — project terms (go into the Whisper prompt), replacement rules
                  ("как распознано" -> "как правильно") and a context paragraph for the LLM.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager

from .config import HOME

SETTINGS_FILE = HOME / "settings.json"
DICTIONARY_FILE = HOME / "dictionary.json"
_lock = threading.Lock()

DEFAULTS = {
    "defaults": {"language": "ru", "profile": "balanced", "num_speakers": 2, "mixed": True, "model": "auto",
                 "quality": "max"},
    # any OpenAI-compatible server; LM Studio listens here by default
    "llm": {"url": "http://127.0.0.1:1234/v1", "model": ""},
    # queue speed-ups for quality "max" (docs/DECISIONS.md); none of them changes the text.
    # prefetch: read the next models into the OS file cache while one works (prefetch.py);
    # batch: several queued recordings model by model, each model loaded once (pipeline.process_batch);
    # parallel_cpu: the CPU members (ONNX) run while the GPU members do (ensemble.lanes);
    # early_voices: speaker voices (diarization, ~1x real time on the CPU) are computed in a
    # separate process while the recording is recognized (pipeline.Voices);
    # diarize_gpu: speakers on onnxruntime with the voice embeddings on the GPU (diarize_ort,
    # lab H12) instead of sherpa-onnx on the CPU; speakers almost always the same, not the text.
    # Defaults from the queue benchmarks of 2026-11-01 (docs/DECISIONS.md, «12. Скорость»): prefetch on;
    # diarize_gpu on (speakers of all words the same) and
    # early_voices off (its CPU process slows the GPU members down: ru-large-v3 224 s vs 137 s);
    # batch and parallel_cpu off until a second gold reference can tell (they change the GPU's
    # memory state and with it ru-large-v3's fp16 decoding a little)
    "speed": {"prefetch": True, "prefetch_gb": 6.0, "batch": False, "batch_docs": 4, "batch_audio_gb": 1.5,
              "parallel_cpu": False, "early_voices": False, "diarize_gpu": True},
}
_SPEED_ENV = {"prefetch": "WHISPER_STUDIO_PREFETCH", "batch": "WHISPER_STUDIO_BATCH",
              "parallel_cpu": "WHISPER_STUDIO_PARALLEL_CPU", "early_voices": "WHISPER_STUDIO_EARLY_VOICES",
              "diarize_gpu": "WHISPER_STUDIO_DIARIZE_GPU"}


def _read(f, fallback):
    try:
        return json.loads(f.read_text("utf-8"))
    except Exception:  # noqa: BLE001 - missing or broken file -> defaults
        return fallback


class Conflict(Exception):
    """The dictionary changed since the caller read it (another window or an agent wrote it)."""


@contextmanager
def _file_lock(f):
    """One writer at a time across processes (UI server, MCP server, CLI) and threads."""
    with _lock:
        fd = os.open(f.with_name(f.name + ".lock"), os.O_RDWR | os.O_CREAT)
        try:
            try:
                import msvcrt
            except ImportError:  # not Windows
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
                return
            for _ in range(100):
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            else:
                raise TimeoutError(f"{f.name} занят другим процессом")
            try:
                yield
            finally:
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.close(fd)


def _write(f, data):
    tmp = f.with_name(f"{f.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    for attempt in range(40):
        try:
            tmp.replace(f)
            return
        except PermissionError:  # a reader has the file open right now (Windows)
            if attempt == 39:
                raise
            time.sleep(0.05)


def get() -> dict:
    s = _read(SETTINGS_FILE, {})
    return {k: {**v, **(s.get(k) or {})} for k, v in DEFAULTS.items()}


def speed() -> dict:
    """The "speed" section; WHISPER_STUDIO_PREFETCH / _BATCH / _PARALLEL_CPU / _EARLY_VOICES = 1 or 0
    override it."""
    s = dict(get()["speed"])
    for k, env in _SPEED_ENV.items():
        v = os.environ.get(env)
        if v is not None and v.strip() != "":
            s[k] = v.strip().lower() not in ("0", "false", "no", "off")
    return s


def update(patch: dict) -> dict:
    with _file_lock(SETTINGS_FILE):
        s = get()
        for k, v in (patch or {}).items():
            if k in s and isinstance(v, dict):
                s[k].update({kk: vv for kk, vv in v.items() if kk in DEFAULTS[k]})
        _write(SETTINGS_FILE, s)
        return s


# ---------------------------------------------------------------- dictionary

def dictionary() -> dict:
    d = _read(DICTIONARY_FILE, {})
    return dict(terms=[t for t in d.get("terms", []) if str(t).strip()],
                replace=[r for r in d.get("replace", []) if str(r.get("from", "")).strip()],
                context=d.get("context", ""), rev=int(d.get("rev") or 0))


def _save_dictionary(d: dict, cur: int) -> dict:
    clean = dict(
        terms=list(dict.fromkeys(str(t).strip() for t in d.get("terms", []) if str(t).strip())),
        replace=[{"from": str(r["from"]).strip(), "to": str(r.get("to", "")).strip()}
                 for r in d.get("replace", []) if str(r.get("from", "")).strip()],
        context=str(d.get("context", "")).strip(), rev=cur + 1,
    )
    _write(DICTIONARY_FILE, clean)
    return clean


def save_dictionary(d: dict, rev: int | None = None) -> dict:
    """Write the whole dictionary. rev: the version the caller started from; if someone saved
    in between, Conflict is raised instead of silently dropping their entries."""
    with _file_lock(DICTIONARY_FILE):
        cur = dictionary()["rev"]
        if rev is not None and int(rev) != cur:
            raise Conflict("Словарь изменился, пока он был открыт: обновите и сохраните ещё раз")
        return _save_dictionary(d, cur)


def edit_dictionary(change) -> dict:
    """Read, change and write the dictionary as one step: change(d) edits d in place."""
    with _file_lock(DICTIONARY_FILE):
        d = dictionary()
        cur = d["rev"]
        change(d)
        return _save_dictionary(d, cur)


def prompt_terms(extra: str = "", limit: int = 600) -> str:
    """Glossary for the Whisper prompt: per-recording terms first, then the project's.
    The decoder sees at most ~224 prompt tokens, so the list is cut by length."""
    seen, out, size = set(), [], 0
    for t in [*re.split(r"[,\n;]+", extra or ""), *dictionary()["terms"]]:
        t = t.strip()
        if not t or t.lower() in seen:
            continue
        if size + len(t) + 2 > limit:
            break
        seen.add(t.lower())
        out.append(t)
        size += len(t) + 2
    return ", ".join(out)


def _rule(src: str) -> re.Pattern:
    stem = src.endswith("*")
    # "хаупт -банхоф" also matches "хаупт-банхоф" and "хаупт - банхоф", not "хаупт банхоф"
    parts = re.split(r"(\s*-\s*|\s+)", src.rstrip("*").strip())
    body = "".join(r"\s*-\s*" if p.strip() == "-" else r"\s+" if p.isspace() else re.escape(p)
                   for p in parts if p)
    return re.compile(rf"(?<!\w){body}{r'\w*' if stem else ''}(?!\w)", re.I)


def rules() -> list[tuple[re.Pattern, str]]:
    # longest first, so "Frau Кесслер" wins over "Кесслер"
    rs = sorted(dictionary()["replace"], key=lambda r: -len(r["from"]))
    return [(_rule(r["from"]), r["to"]) for r in rs]


def apply_rules(text: str, compiled=None) -> tuple[str, int]:
    n = 0
    for rx, to in compiled if compiled is not None else rules():
        text, k = rx.subn(lambda m, to=to: to, text)
        n += k
    return text, n
