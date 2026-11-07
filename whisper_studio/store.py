"""Library of transcripts: one folder per document with doc.json, audio and peaks.

All edits go through this module (UI, CLI and MCP share it). Every mutation keeps
an undo snapshot. When text is edited, word timings are preserved for unchanged
words and interpolated for new ones, so click-to-listen keeps working.
"""
from __future__ import annotations

import difflib
import json
import msvcrt
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .config import LIBRARY_DIR

PALETTE = ["#0A84FF", "#FF9F0A", "#30D158", "#BF5AF2", "#FF375F", "#64D2FF", "#FFD60A", "#AC8E68"]
_lock = threading.RLock()
UNDO_DEPTH = 50
# Words below this p are worth checking. For "max" transcripts p is the models' agreement minus
# what else speaks against the word (ensemble._doubt): on the gold reference ~9 % of the words fall
# below, about 40 % of them are wrong, and they cover 3/4 or more of the wrong words (docs/DECISIONS.md).
# The UI highlights these and counts them (web/app.js LOW_P).
REVIEW_P = 0.35


class Busy(Exception):
    """The document is being processed (transcription, speakers, AI): edits would be lost."""


class Refused(Exception):
    """The change is not allowed on this document (AI proofreading of a gold reference)."""


# A gold reference (doc["gold"], docs/GOLD.md) is text the owner checked by ear: the measure of
# the ensemble. AI or agent corrections would make it measure the AI instead.
GOLD_AI = "Это эталон: ИИ-вычитка отключена, правки только на слух"


def is_gold(doc: dict) -> bool:
    return doc.get("gold") is not None


_held = threading.local()


@contextmanager
def _doc_lock(doc_id: str):
    """Cross-process lock of one document (UI server, MCP server and worker are separate
    processes): load -> change -> save of one never interleaves with another's."""
    depth = getattr(_held, doc_id, 0)
    if depth:  # re-entrant within a thread
        setattr(_held, doc_id, depth + 1)
        try:
            yield
        finally:
            setattr(_held, doc_id, depth)
        return
    fd = os.open(_dir(doc_id) / "doc.lock", os.O_RDWR | os.O_CREAT)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise Busy("Документ занят другим процессом") from None
                time.sleep(0.02)
        setattr(_held, doc_id, 1)
        try:
            yield
        finally:
            setattr(_held, doc_id, 0)
            os.lseek(fd, 0, 0)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


@contextmanager
def editing(doc_id: str, busy_ok: bool = False):
    """Load a document for a change under both locks; refuses while a job works on it
    (the worker holds its own copy and would write over the edit)."""
    with _lock, _doc_lock(doc_id):
        doc = load(doc_id)
        if not busy_ok and doc.get("status") in ("queued", "processing"):
            raise Busy("Документ ещё обрабатывается — правка будет доступна, когда обработка закончится")
        yield doc


def _dir(doc_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9-]+", doc_id):
        raise ValueError("bad id")
    return LIBRARY_DIR / doc_id


def new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def create(title: str, source: str, options: dict) -> dict:
    doc = dict(
        id=new_id(), title=title, source_path=source, created=datetime.now().isoformat(timespec="seconds"),
        updated=None, status="queued", progress=0.0, stage="В очереди", error=None,
        audio_file=None, duration=None, language=None, meta={}, options=options,
        speakers={}, segments=[], edited=False,
    )
    _dir(doc["id"]).mkdir(parents=True)
    save(doc, snapshot=False)
    return doc


def path(doc_id: str) -> Path:
    return _dir(doc_id)


def load(doc_id: str) -> dict:
    with _lock:
        for attempt in range(40):
            try:
                return json.loads((_dir(doc_id) / "doc.json").read_text("utf-8"))
            except (PermissionError, json.JSONDecodeError):  # file is being replaced right now
                if attempt == 39:
                    raise
                time.sleep(0.05)


def save(doc: dict, snapshot: bool = True) -> dict:
    with _lock:
        d = _dir(doc["id"])
        f = d / "doc.json"
        if snapshot and f.exists():
            hist = d / "history"
            hist.mkdir(exist_ok=True)
            shutil.copy2(f, hist / f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}.json")
            for old in sorted(hist.glob("*.json"))[:-UNDO_DEPTH]:
                old.unlink()
        doc["updated"] = datetime.now().isoformat(timespec="seconds")
        doc["rev"] = int(doc.get("rev") or 0) + 1  # the UI reloads a document edited elsewhere by it
        _write_atomic(f, json.dumps(doc, ensure_ascii=False))
        return doc


def _write_atomic(f: Path, text: str) -> None:
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, "utf-8")
    # Windows refuses the rename while another process (UI poll, MCP) is reading doc.json
    for attempt in range(40):
        try:
            tmp.replace(f)
            break
        except PermissionError:
            if attempt == 39:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def undo(doc_id: str) -> dict:
    with editing(doc_id) as cur:
        hist = sorted((_dir(doc_id) / "history").glob("*.json"))
        if not hist:
            return cur
        last = hist[-1]
        doc = json.loads(last.read_text("utf-8"))
        doc["rev"] = int(cur.get("rev") or 0)
        save(doc, snapshot=False)
        last.unlink()
        return doc


def summary(doc: dict) -> dict:
    keys = ("id", "title", "created", "updated", "status", "progress", "stage", "error",
            "duration", "language", "edited")
    s = {k: doc.get(k) for k in keys}
    s["source"] = doc.get("source_path")
    s["rev"] = doc.get("rev") or 0
    s["model"] = (doc.get("meta") or {}).get("model") or (doc.get("options") or {}).get("model")
    s["quality"] = (doc.get("options") or {}).get("quality")
    s["speakers"] = len(doc.get("speakers") or {})
    s["segments"] = len(doc.get("segments") or [])
    s["low"] = sum(1 for x in doc.get("segments") or []
                   for w in x.get("words") or [] if w.get("p", 1) < REVIEW_P and not w.get("edited"))
    s["preview"] = " ".join(x["text"] for x in doc.get("segments", [])[:3])[:160]
    s["gold"] = is_gold(doc)
    if s["gold"]:
        ok = [x for x in doc.get("segments") or [] if x.get("verified")]
        s["verified"] = len(ok)
        s["verified_seconds"] = round(sum(x["end"] - x["start"] for x in ok), 1)
    return s


# Summaries of all documents, keyed by doc.json's mtime and size. Kept in memory and in
# library/index.json, so listing the library reads one file instead of every doc.json
# (cold reads cost a seek each). Any process may
# rewrite the index; a stale or broken one is simply rebuilt.
INDEX_FILE = LIBRARY_DIR / "index.json"
_index: dict[str, dict] = {}
_index_loaded = False


def _stamp(f: Path) -> list[int] | None:
    try:
        st = f.stat()
    except OSError:
        return None
    return [st.st_mtime_ns, st.st_size]


def list_docs() -> list[dict]:
    global _index_loaded
    with _lock:
        if not _index_loaded:
            try:
                _index.update(json.loads(INDEX_FILE.read_text("utf-8")))
            except Exception:  # noqa: BLE001 - missing or broken: rebuilt below
                pass
            _index_loaded = True
        out, seen, dirty = [], set(), False
        for d in LIBRARY_DIR.iterdir():
            stamp = _stamp(d / "doc.json")
            if stamp is None:
                continue
            seen.add(d.name)
            hit = _index.get(d.name)
            if hit and hit.get("stamp") == stamp and "gold" in hit["summary"]:  # older summaries lack it
                out.append(hit["summary"])
                continue
            try:
                s = summary(load(d.name))
            except Exception:  # noqa: BLE001 - being replaced right now, or a foreign folder
                if hit:
                    out.append(hit["summary"])
                continue
            # the stamp from before the read: a save that lands meanwhile is picked up next time
            _index[d.name] = dict(stamp=stamp, summary=s)
            out.append(s)
            dirty = True
        for k in [k for k in _index if k not in seen]:
            del _index[k]
            dirty = True
        if dirty:
            try:
                _write_atomic(INDEX_FILE, json.dumps(_index, ensure_ascii=False))
            except OSError:  # another process is writing it right now
                pass
    return sorted(out, key=lambda x: x["created"] or "", reverse=True)


def delete(doc_id: str) -> None:
    shutil.rmtree(_dir(doc_id), ignore_errors=True)


def norm_text(t: str) -> str:
    """Lower case, ё = е: what search compares."""
    return t.lower().replace("ё", "е")


_texts: dict[str, tuple[list[int], dict]] = {}


def _doc_cached(doc_id: str) -> dict | None:
    """Full document through a small mtime cache (search reads every document)."""
    stamp = _stamp(_dir(doc_id) / "doc.json")
    hit = _texts.get(doc_id)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        doc = load(doc_id)
    except (OSError, ValueError):
        return None
    _texts[doc_id] = (stamp, doc)
    return doc


def search(query: str, limit: int = 200, doc_ids: list[str] | None = None,
           per_doc: int | None = None) -> list[dict]:
    """Segments containing query (case-insensitive, ё = е), newest documents first.
    per_doc: at most that many hits of one document (the first one carries "count", the
    number of all its hits), so a common word can't crowd older documents out."""
    q = norm_text(query.strip())
    if not q:
        return []
    hits = []
    for s in list_docs():
        if doc_ids and s["id"] not in doc_ids:
            continue
        doc = _doc_cached(s["id"])
        if not doc:
            continue
        name = Path(doc.get("source_path") or "").name
        title_hit = q in norm_text(doc.get("title") or "") or q in norm_text(name)
        mine = []
        for seg in doc["segments"]:
            if q in norm_text(seg["text"]):
                mine.append(dict(doc_id=doc["id"], title=doc["title"], segment_id=seg["id"],
                                 start=seg["start"], speaker=speaker_name(doc, seg.get("speaker")),
                                 text=seg["text"]))
        if mine:
            mine[0]["count"] = len(mine)
            hits += mine[:per_doc] if per_doc else mine
        elif title_hit:
            hits.append(dict(doc_id=doc["id"], title=doc["title"], segment_id=None, start=0.0,
                             speaker="", text="", count=0))
        if len(hits) >= limit:
            return hits[:limit]
    return hits


# ------------------------------------------------------------------ helpers

def speaker_name(doc: dict, sid: str | None) -> str:
    if not sid:
        return ""
    return (doc.get("speakers") or {}).get(sid, {}).get("name", sid)


def finalize(doc: dict, segments: list[dict]) -> dict:
    """Assign ids and speaker table after transcription."""
    for i, s in enumerate(segments):
        s["id"] = f"g{i}"
        s.setdefault("speaker", None)
        s.setdefault("flags", [])
    sids = []
    for s in segments:
        if s["speaker"] and s["speaker"] not in sids:
            sids.append(s["speaker"])
    # number speakers in order of first appearance
    remap = {old: f"S{i + 1}" for i, old in enumerate(sids)}
    for s in segments:
        if s["speaker"]:
            s["speaker"] = remap[s["speaker"]]
    doc["speakers"] = {f"S{i + 1}": dict(name=f"Спикер {i + 1}", color=PALETTE[i % len(PALETTE)])
                       for i in range(len(sids))}
    doc["segments"] = segments
    return doc


def _seg(doc: dict, seg_id: str) -> tuple[int, dict]:
    for i, s in enumerate(doc["segments"]):
        if s["id"] == seg_id:
            return i, s
    raise KeyError(f"Сегмент {seg_id} не найден")


def _next_id(doc: dict) -> str:
    n = 0
    for s in doc["segments"]:
        m = re.fullmatch(r"g(\d+)", s["id"])
        if m:
            n = max(n, int(m.group(1)) + 1)
    return f"g{n}"


def retime_words(old: list[dict], new_text: str, start: float, end: float) -> list[dict]:
    """Map edited text onto old word timings (keep equal words, interpolate the rest)."""
    tokens = new_text.split()
    if not tokens:
        return []
    norm = lambda w: re.sub(r"[^\w]", "", w.lower())  # noqa: E731
    sm = difflib.SequenceMatcher(a=[norm(w["w"]) for w in old], b=[norm(t) for t in tokens], autojunk=False)
    out: list[dict | None] = [None] * len(tokens)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                w = old[i1 + k]
                out[j1 + k] = dict(w=tokens[j1 + k], s=w["s"], e=w["e"], p=w.get("p", 1.0),
                                   **({"spoken": w["spoken"]} if "spoken" in w else {}))
        elif tag == "replace" and i2 > i1:
            # spread the new words over the time the replaced words occupied
            s0, e0 = old[i1]["s"], old[i2 - 1]["e"]
            n = j2 - j1
            for k in range(n):
                out[j1 + k] = dict(w=tokens[j1 + k], s=round(s0 + (e0 - s0) * k / n, 3),
                                   e=round(s0 + (e0 - s0) * (k + 1) / n, 3), p=1.0, edited=True)
    # inserted words: interpolate between known neighbours
    i = 0
    while i < len(out):
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        left = out[i - 1]["e"] if i > 0 else start
        right = out[j]["s"] if j < len(out) else end
        if right < left:
            right = left
        n = j - i
        for k in range(n):
            out[i + k] = dict(w=tokens[i + k], s=round(left + (right - left) * k / n, 3),
                              e=round(left + (right - left) * (k + 1) / n, 3), p=1.0, edited=True)
        i = j
    return out  # type: ignore[return-value]


def rewrite(s: dict, text: str) -> None:
    """New text of a segment, keeping word timings."""
    s["words"] = retime_words(s.get("words") or [], text, s["start"], s["end"])
    s["text"] = " ".join(text.split())


# ------------------------------------------------------------------ edits

def clean_text(text: str) -> str:
    """One line, single spaces, without the ⟨uncertain⟩ marks of proofread_pack."""
    return " ".join(str(text).replace("⟨", "").replace("⟩", "").split())


def _silent(doc: dict, s: dict) -> None:
    """A segment deleted from a gold reference: nothing was said there, which counts as checked."""
    if is_gold(doc):
        doc["gold"].setdefault("silent", []).append([s["start"], s["end"]])


def _mark(s: dict, by: str, new: str | None = None) -> None:
    """Record who changed a segment last (owner, ai, rule, replace) and keep what the models heard."""
    if new is not None and new != s["text"]:
        s.setdefault("orig_text", s["text"])
    s["edited"] = True
    s["by"] = by
    if by == "ai":
        s["ai"] = True


def by_owner(s: dict) -> bool:
    """The owner typed or confirmed this segment's text (older documents: any edit but the AI's)."""
    return bool(s.get("edited")) and s.get("by", "ai" if s.get("ai") else "owner") == "owner"


def edit_text(doc_id: str, seg_id: str, text: str, by: str = "owner") -> dict:
    """In a gold reference the owner's edit means the segment was checked by ear."""
    with editing(doc_id) as doc:
        i, s = _seg(doc, seg_id)
        text = clean_text(text)
        if not text:
            _silent(doc, doc["segments"].pop(i))
        else:
            _mark(s, by, text)
            rewrite(s, text)
            s["flags"] = []
            if is_gold(doc) and by == "owner":
                s["verified"] = True
        doc["edited"] = True
        return save(doc)


def set_verified(doc_id: str, seg_id: str, on: bool) -> dict:
    """The owner listened to a segment and confirms its text (gold reference)."""
    with editing(doc_id) as doc:
        _, s = _seg(doc, seg_id)
        if on:
            s["verified"] = True
        else:
            s.pop("verified", None)
        return save(doc)


def set_gold(doc_id: str, on: bool) -> dict:
    """Mark a document as a gold reference, or back as a normal one (verified marks are kept)."""
    with editing(doc_id) as doc:
        if on and not is_gold(doc):
            doc["gold"] = dict(created=datetime.now().isoformat(timespec="seconds"), manifest=[])
        elif not on:
            doc.pop("gold", None)
        return save(doc)


def set_speaker(doc_id: str, seg_id: str, speaker: str | None, name: str | None = None) -> dict:
    """Reassign a segment. speaker=None with a name creates a new speaker."""
    with editing(doc_id) as doc:
        _, s = _seg(doc, seg_id)
        if not speaker:
            speaker = _new_speaker(doc, name)
        elif speaker not in doc["speakers"]:
            raise KeyError(speaker)
        s["speaker"] = speaker
        s["speaker_locked"] = True  # set by hand: re-diarization keeps it (pipeline._keep_locked)
        doc["edited"] = True
        _drop_unused_speakers(doc)
        return save(doc)


def _new_speaker(doc: dict, name: str | None) -> str:
    n = 1
    while f"S{n}" in doc["speakers"]:
        n += 1
    sid = f"S{n}"
    doc["speakers"][sid] = dict(name=name or f"Спикер {n}", color=PALETTE[(n - 1) % len(PALETTE)])
    return sid


def _drop_unused_speakers(doc: dict) -> None:
    used = {s.get("speaker") for s in doc["segments"]}
    for sid in list(doc["speakers"]):
        if sid not in used:
            del doc["speakers"][sid]


def rename_speaker(doc_id: str, speaker: str, name: str, color: str | None = None) -> dict:
    with editing(doc_id) as doc:
        sp = doc["speakers"][speaker]
        sp["name"] = name.strip() or sp["name"]
        if color:
            sp["color"] = color
        return save(doc)


def merge_speakers(doc_id: str, src: str, dst: str) -> dict:
    with editing(doc_id) as doc:
        for s in doc["segments"]:
            if s.get("speaker") == src:
                s["speaker"] = dst
        _drop_unused_speakers(doc)
        doc["edited"] = True
        return save(doc)


def split(doc_id: str, seg_id: str, word_index: int, new_speaker: str | None = None) -> dict:
    """Split a segment before word_index. new_speaker: id, "new", or None (keep)."""
    with editing(doc_id) as doc:
        i, s = _seg(doc, seg_id)
        ws = s.get("words") or []
        if not 0 < word_index < len(ws):
            return doc
        a, b = ws[:word_index], ws[word_index:]
        if is_gold(doc):  # the owner split it while listening
            s["verified"] = True
        tail = {**s, "id": _next_id(doc), "words": b, "start": b[0].get("s", s["start"]), "end": s["end"],
                "text": " ".join(w["w"] for w in b)}
        s.update(words=a, end=a[-1].get("e", s["end"]), text=" ".join(w["w"] for w in a))
        if new_speaker == "new":
            tail["speaker"] = _new_speaker(doc, None)
        elif new_speaker:
            tail["speaker"] = new_speaker
        if new_speaker:
            tail["speaker_locked"] = True
        doc["segments"].insert(i + 1, tail)
        doc["edited"] = True
        return save(doc)


def merge_next(doc_id: str, seg_id: str) -> dict:
    with editing(doc_id) as doc:
        i, s = _seg(doc, seg_id)
        if i + 1 >= len(doc["segments"]):
            return doc
        n = doc["segments"].pop(i + 1)
        s["words"] = (s.get("words") or []) + (n.get("words") or [])
        s["text"] = (s["text"] + " " + n["text"]).strip()
        s["end"] = n["end"]
        s["flags"] = sorted(set(s.get("flags", [])) | set(n.get("flags", [])))
        if is_gold(doc):
            s["verified"] = True
        _drop_unused_speakers(doc)
        doc["edited"] = True
        return save(doc)


def delete_segment(doc_id: str, seg_id: str) -> dict:
    with editing(doc_id) as doc:
        i, _ = _seg(doc, seg_id)
        _silent(doc, doc["segments"].pop(i))
        _drop_unused_speakers(doc)
        doc["edited"] = True
        return save(doc)


def clear_flags(doc_id: str, seg_id: str) -> dict:
    with editing(doc_id) as doc:
        _, s = _seg(doc, seg_id)
        s["flags"] = []
        return save(doc)


def hallucination(seg: dict) -> bool:
    """Flags of a possible Whisper hallucination (phantom phrase, silence, loop)."""
    return any(f != "disagree" for f in seg.get("flags") or [])


def delete_flagged(doc_id: str) -> dict:
    with editing(doc_id) as doc:
        # "disagree" marks real but unclear speech, not a hallucination: it stays
        doc["segments"] = [s for s in doc["segments"] if not hallucination(s)]
        _drop_unused_speakers(doc)
        doc["edited"] = True
        return save(doc)


def replace_all(doc_id: str, find: str, repl: str, whole_word: bool = True,
                case_sensitive: bool = False, by: str = "replace") -> tuple[dict, int]:
    if not find.strip():
        raise ValueError("Пустой текст для поиска")
    with editing(doc_id) as doc:
        pat = re.escape(find)
        if whole_word:
            pat = rf"(?<!\w){pat}(?!\w)"
        rx = re.compile(pat, 0 if case_sensitive else re.I)
        count = 0
        changed = False
        for s in doc["segments"]:
            new, n = rx.subn(lambda _m: repl, s["text"])
            if n:
                count += n
                _mark(s, by, new)
                rewrite(s, new)
                changed = True
        if changed:
            doc["edited"] = True
            save(doc)
        return doc, count


def apply_rules_to_segments(segments: list[dict], compiled) -> int:
    """Dictionary replacements on raw segments (before they are saved)."""
    from .settings import apply_rules

    count = 0
    for s in segments:
        new, n = apply_rules(s["text"], compiled)
        if n and new != s["text"]:
            count += n
            rewrite(s, new)
    return count


def apply_dictionary(doc_id: str) -> tuple[dict, int]:
    from .settings import rules

    with editing(doc_id) as doc:
        before = [s["text"] for s in doc["segments"]]
        n = apply_rules_to_segments(doc["segments"], rules())
        if n:
            for s, old in zip(doc["segments"], before):
                if s["text"] != old:
                    s.setdefault("orig_text", old)
                    _mark(s, "rule")
            doc["edited"] = True
            save(doc)
        return doc, n


def apply_corrections(doc_id: str, corrections: dict[str, str], ai: bool = True) -> tuple[dict, int]:
    """Replace the text of several segments at once (one undo step). Returns changed count."""
    with editing(doc_id) as doc:
        n = correct_segments(doc, corrections, ai)
        if n:
            doc["edited"] = True
            save(doc)
        return doc, n


def correct_segments(doc: dict, corrections: dict[str, str], ai: bool = True) -> int:
    """In-place variant of apply_corrections; the original text is kept in orig_text."""
    if ai and is_gold(doc):
        raise Refused(GOLD_AI)
    n = 0
    for s in doc["segments"]:
        new = corrections.get(s["id"])
        if new is None:
            continue
        new = clean_text(new)
        if not new or new == s["text"]:
            continue
        _mark(s, "ai" if ai else "owner", new)
        rewrite(s, new)
        n += 1
    return n


def rename(doc_id: str, title: str) -> dict:
    with editing(doc_id, busy_ok=True) as doc:
        doc["title"] = title.strip() or doc["title"]
        return save(doc, snapshot=False)
