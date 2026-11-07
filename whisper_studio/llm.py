"""Proofreading of transcripts by a language model.

Two ways, same prompt:
- in the app: any OpenAI-compatible server (LM Studio, Ollama, llama.cpp) from settings.llm;
- through MCP: the connected agent gets the pack (proofread_pack) and sends corrections back.
"""
from __future__ import annotations

import difflib
import json
import re
import time
import urllib.error
import urllib.request

from . import settings, store

LOW_CONFIDENCE = 0.5

INSTRUCTIONS = """Ты корректор автоматической расшифровки разговора (Whisper). Речь в основном русская,
со вставками немецких слов и фраз. Исправляй ТОЛЬКО ошибки распознавания:
- ослышки, когда по смыслу и звучанию очевидно другое слово;
- немецкие слова, записанные кириллицей или искажённые, пиши по-немецки латиницей (вонунг → Wohnung);
- имена, названия и термины пиши так, как в словаре проекта;
- явные повторы-галлюцинации («так, так, так, так») сокращай до сказанного;
- пунктуацию и регистр.
Нельзя: перефразировать, переводить, сокращать, добавлять смысл, менять порядок реплик,
переносить текст между репликами. Сомневаешься — оставь как есть.
Слова в ⟨угловых скобках⟩ модель расслышала неуверенно — к ним присмотрись особо.
Строки «варианты:» — как ту же реплику услышали другие модели; выбирай лучшее по смыслу и звучанию."""


def invented_digits(seg: dict, new: str) -> bool:
    """A correction whose numbers no recognizer heard: language models "fix" numbers
    into plausible ones, so such a change is refused."""
    digits = lambda t: re.findall(r"\d+", t or "")  # noqa: E731
    want = digits(new)
    return all(want != digits(t) for t in [seg["text"], seg.get("orig_text")] +
               [a.get("text") for a in seg.get("alts") or []])


MAX_CHANGE = 0.3  # a local model's fix may rewrite at most this share of a segment's characters


def refused(seg: dict, new: str) -> str | None:
    """Why a local model's correction is not taken, or None. Small models rewrite fluent text
    into what they expect (docs/DECISIONS.md, H8), so a fix may only use what a recognizer heard
    or the dictionary knows: no new numbers, no new words (an ending fixed is fine), and no
    rewrite of more than MAX_CHANGE of the segment unless it is a member's reading as a whole."""
    if invented_digits(seg, new):
        return "числа"
    alts = [a.get("text") or "" for a in seg.get("alts") or []]
    if _norm_text(new) in {_norm_text(t) for t in alts}:
        return None
    d = settings.dictionary()
    heard = [seg["text"], seg.get("orig_text") or "", *alts, *d["terms"], *(r["to"] for r in d["replace"])]
    known = {_norm(w) for t in heard for w in str(t).split()} - {""}
    for w in {_norm(x) for x in new.split()} - known - {""}:
        if not any(difflib.SequenceMatcher(a=w, b=k).ratio() >= 0.8 for k in known if abs(len(k) - len(w)) <= 3):
            return f"новое слово «{w}»"
    if 1 - difflib.SequenceMatcher(a=seg["text"], b=new, autojunk=False).ratio() > MAX_CHANGE:
        return "слишком большая правка"
    return None


def _norm_text(t: str) -> str:
    return " ".join(x for x in (_norm(w) for w in t.split()) if x)


_status: dict = {"t": -1e9, "key": None, "value": None}


def status(max_age: float = 15.0) -> dict:
    """Is the proofreading LLM server up? Cached: a refused connection to localhost
    takes about 2 s on Windows, and the UI asks on every status poll."""
    cfg = settings.get()["llm"]
    url = (cfg.get("url") or "").rstrip("/")
    key = (url, cfg.get("model"))
    if _status["key"] == key and time.monotonic() - _status["t"] < max_age:
        return _status["value"]
    value = _probe(url, cfg)
    _status.update(t=time.monotonic(), key=key, value=value)
    return value


def _probe(url: str, cfg: dict) -> dict:
    if not url:
        return dict(available=False, model=None, url=url)
    try:
        with urllib.request.urlopen(url + "/models", timeout=1.5) as r:
            ids = [m["id"] for m in json.loads(r.read()).get("data", [])]
    except Exception:  # noqa: BLE001 - server not running
        return dict(available=False, model=None, url=url)
    chat = [i for i in ids if "embed" not in i.lower()]
    model = cfg.get("model") if cfg.get("model") in ids else (chat[0] if chat else None)
    return dict(available=bool(model), model=model, url=url)


def mark(seg: dict) -> str:
    """Segment text with low-confidence words marked ⟨…⟩ (fallback: plain text)."""
    ws = seg.get("words") or []
    if not ws or seg.get("edited"):
        return seg["text"]
    return " ".join(f"⟨{w['w']}⟩" if w.get("p", 1) < LOW_CONFIDENCE else w["w"] for w in ws)


def _norm(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е"))


def variants(seg: dict, max_items: int = 10) -> str:
    """What other models heard differently, as compact replacements of the segment's own
    words: «двадцать пятый»→«двадцать четвёртое» (GigaAM, Qwen3). Alternatives that differ
    only in case or punctuation are dropped; an alternative that shares little with the
    text is given whole. Empty when every model agrees."""
    from .models import short_name

    a = seg["text"].split()
    an = [_norm(w) for w in a]
    found: dict[tuple, list[str]] = {}
    whole: dict[str, list[str]] = {}
    for alt in seg.get("alts") or []:
        b = (alt.get("text") or "").split()
        bn = [_norm(w) for w in b]
        if not b or [x for x in bn if x] == [x for x in an if x]:
            continue
        name = short_name(alt.get("model") or "?")
        sm = difflib.SequenceMatcher(a=an, b=bn, autojunk=False)
        if sm.ratio() < 0.5:
            # a much longer alternative spans the neighbouring segments too: no help here
            if len(b) <= 2 * len(a) + 3:
                k = " ".join(x for x in bn if x)
                whole.setdefault(k, [" ".join(b)])
                if name not in whole[k]:
                    whole[k].append(name)
            continue
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal" or (not any(an[i1:i2]) and not any(bn[j1:j2])):
                continue
            key = (i1, i2, " ".join(x for x in bn[j1:j2] if x))
            if key not in found:
                found[key] = [" ".join(a[i1:i2]), " ".join(b[j1:j2])]
            if name not in found[key]:
                found[key].append(name)
    parts = []
    for (i1, _, _), (src, dst, *names) in sorted(found.items(), key=lambda kv: kv[0][0])[:max_items]:
        if not src:  # an inserted word: anchor it to the word before
            prev = a[i1 - 1] if i1 else ""
            parts.append(f"+«{dst}» после «{prev}» ({', '.join(names)})" if prev else f"+«{dst}» в начале ({', '.join(names)})")
        else:
            parts.append(f"«{src}»→«{dst}» ({', '.join(names)})")
    parts += [f"целиком: «{t}» ({', '.join(n)})" for t, *n in whole.values()]
    return "; ".join(parts)


def _line(doc: dict, s: dict) -> str:
    sp = store.speaker_name(doc, s.get("speaker"))
    flags = s.get("flags") or []
    flag = (" ⚠" + ",".join(f for f in flags if f != "disagree") if any(f != "disagree" for f in flags) else "") \
        + (" (неразборчиво)" if "disagree" in flags else "")
    out = f"{s['id']} [{sp}] {mark(s)}{flag}"
    v = variants(s)
    if v:
        out += f"\n    варианты: {v}"
    return out


def header(compact: bool = False) -> str:
    """Context for the model. compact: fits a small local context window (4k tokens)."""
    d = settings.dictionary()
    parts = []
    if d["context"]:
        parts.append("Контекст записей: " + d["context"])
    if d["terms"]:
        parts.append("Словарь проекта: " + ", ".join(d["terms"][:40 if compact else None]))
    if d["replace"]:
        groups: dict[str, list[str]] = {}
        for r in d["replace"][:25 if compact else None]:
            groups.setdefault(r["to"], []).append(r["from"])
        parts.append("Частые ошибки (правильно ← как распознаётся): " +
                     "; ".join(f"{to} ← {', '.join(fr)}" for to, fr in groups.items()))
    return "\n".join(parts)


def chunks(doc: dict, size: int = 15, overlap: int = 3):
    segs = doc["segments"]
    for i in range(0, len(segs), size):
        yield segs[max(0, i - overlap):i], segs[i:i + size]


def proofread_pack(doc: dict, from_segment: str | None = None, max_chars: int = 24000) -> str:
    """Everything an agent needs to proofread one transcript (MCP), page by page:
    the instructions and the dictionary come with the first page only."""
    if store.is_gold(doc):
        raise store.Refused(store.GOLD_AI)
    segs = doc["segments"]
    i0 = next((i for i, s in enumerate(segs) if s["id"] == from_segment), None) if from_segment else 0
    if i0 is None:
        raise KeyError(f"Реплика {from_segment} не найдена")
    out = ""
    if i0 == 0:
        out = INSTRUCTIONS + "\n\n" + header() + "\n\n"
    out += f"Расшифровка {doc['id']} (id [спикер] текст; реплик {len(segs)}):\n"
    lines, size, nxt = [], len(out), None
    for s in segs[i0:]:
        line = _line(doc, s)
        if lines and size + len(line) > max_chars:
            nxt = s["id"]
            break
        lines.append(line)
        size += len(line) + 1
    out += "\n".join(lines)
    out += ("\n\nОтвет: вызови apply_corrections с объектом {id: исправленный текст} "
            "только для реплик, которые изменились.")
    if nxt:
        out += f" Это не вся запись: потом продолжи с proofread_pack(from_segment=\"{nxt}\")."
    return out


def _chat(prompt: str, cfg: dict, system: str = INSTRUCTIONS) -> str:
    st = status()
    body = dict(model=st["model"], temperature=0.1, messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}],
        # reasoning models think for minutes per chunk; this task needs no reasoning
        reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(cfg["url"].rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""
    except urllib.error.HTTPError as e:  # e.g. the prompt is longer than the loaded context
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"LLM ответила {e.code}: {detail}") from None


def _parse(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in d.items() if isinstance(v, str)}


def proofread_doc(doc: dict, progress=None) -> int:
    """Proofread in place with the configured local model. Returns changed segment count."""
    if store.is_gold(doc):
        raise store.Refused(store.GOLD_AI)
    if not status(0)["available"]:
        raise RuntimeError("Локальная LLM не запущена (LM Studio / Ollama) — см. «Настройки»")
    cfg = settings.get()["llm"]
    head = header(compact=True)
    parts = list(chunks(doc))
    changed, skipped = 0, []
    for i, (ctx, segs) in enumerate(parts):
        if progress:
            progress(i / max(len(parts), 1), "Исправляю ИИ")
        prompt = head
        if ctx:
            prompt += "\n\nПредыдущие реплики (только для контекста, не исправлять):\n" + \
                      "\n".join(f"{s['id']} [{store.speaker_name(doc, s.get('speaker'))}] {s['text']}" for s in ctx)
        prompt += "\n\nРеплики для проверки:\n" + "\n".join(_line(doc, s) for s in segs)
        prompt += "\n\nВерни только JSON-объект {\"id\": \"исправленный текст\"} для изменённых реплик, или {}."
        ids = {s["id"] for s in segs}
        reply = _parse(_chat(prompt, cfg, INSTRUCTIONS))
        fixes = {k: v.replace("⟨", "").replace("⟩", "") for k, v in reply.items() if k in ids}
        by_id = {s["id"]: s for s in segs}
        for k, v in list(fixes.items()):
            why = refused(by_id[k], store.clean_text(v))
            if why:
                skipped.append(dict(id=k, text=v, why=why))
                del fixes[k]
        changed += store.correct_segments(doc, fixes, ai=True)
    doc.setdefault("meta", {})["proofread_refused"] = skipped
    return changed
