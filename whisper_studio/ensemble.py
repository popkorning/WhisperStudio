"""Quality mode "max": several recognizers on the same chunks, merged word by word.

1. One segmentation for everybody: Silero VAD speech regions glued into chunks <= 25 s.
   Every model hears exactly the same chunks, decoded independently (no context carry-over,
   so one bad chunk can't swallow the next 25 seconds).
2. Members run one at a time (one heavy model in memory, the rest unloaded). Raw results
   go to hypotheses.json in the document folder after every model: an interrupted job
   continues where it stopped and the merge can be redone without recognition (recombine).
   Optional (settings speed, docs/DECISIONS.md): the ONNX members on a CPU lane next to the
   GPU ones (lanes), several recordings model by model (run_batch), prefetch.py.
3. Chunk language: Whisper language-ID + script of the multilingual large-v3 text.
   Russian-only models vote on Russian chunks, German-capable ones on German chunks,
   both groups on mixed chunks; nobody is forced to translate.
4. Merge (ROVER): hypotheses are aligned into a confusion network (edit distance on
   normalized words + time). Each slot is decided by weighted votes, weight = model weight x
   the model's word confidence; "nothing here" is a vote too. A Cyrillic spelling of an
   unknown word ("вонунг") is a phonetic vote for a similar German word ("Wohnung").
5. Re-check: merged words that are not Russian (pymorphy dictionary) are looked up in the
   project glossary and in a German hypothesis (de-turbo) at the same time position.

Output has the engine.transcribe format plus seg["alts"] = [{"model", "text"}] (every model's
text for that span, for the UI and LLM proofreading) and seg["lang"]. Word p is the
agreement after merging (low where the models disagree) minus doubt penalties (_doubt: sounds
like a glossary term, not Russian, numbers disputed...); the segment's metrics keep the agreement.
Numbers are written in digits (asr/itn.py) after the vote, in the text and in the alts alike;
the words said stay in w["spoken"].
"""
from __future__ import annotations

import gc
import json
import re
import threading
import time
import traceback
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

import numpy as np

from . import asr, engine, models, prefetch, settings
from .asr import gpu, itn
from .asr import text as T
from .config import SAMPLE_RATE

VERSION = 1
MAX_CHUNK = 25.0
MAX_GAP = 2.0            # a longer pause starts a new chunk
CHUNK_PAD = 0.15
OVERLAP = 1.5            # context heard across a cut inside running speech

# Run order matters: the first member also identifies the language of every chunk.
# lang: "auto" = the model decides (multilingual), "chunk" = told the chunk language,
# "single" = told it unless the chunk mixes languages, "ru"/"de" = fixed;
# only: chunk languages the member is run on and votes for.
MEMBERS: list[dict] = [
    dict(name="large-v3", backend="whisper", model="large-v3", lang="auto", weight=1.0, cost=1.0, cpu_ok=True),
    dict(name="ru-large-v3", backend="whisper", model="ru-large-v3", lang="ru", only=("ru",), weight=1.0,
         cost=1.0),
    dict(name="de-turbo", backend="whisper", model="de-turbo", lang="de", only=("de",), weight=1.0, cost=0.4),
    # Qwen3-ASR left to itself translates Russian into German and, given the (German) glossary
    # as context, recites it on near-silence: told the language, no prompt; free only on mixed chunks
    dict(name="qwen3-asr", backend="qwen", model="qwen3-asr", lang="single", weight=1.0, cost=1.0,
         prompt=False),
    dict(name="gigaam-v3", backend="onnx", model="gigaam-v3", lang="ru", only=("ru",), weight=1.2, cost=0.4),
    # Canary in Russian invents and loops ("2000, 2000, 2000…"): only the German hypothesis
    dict(name="canary-v2", backend="onnx", model="canary-v2", lang="de", only=("de",), weight=0.8, cost=1.0),
    dict(name="t-one", backend="onnx", model="t-one", lang="ru", only=("ru",), weight=0.5, cost=0.2,
         punct=False),
]
# whose spelling (digits, case, punctuation) wins when several models voted for the same word
SURFACE = {
    # Qwen3-ASR often drops punctuation and capitals: its spelling is used last
    "ru": ["ru-large-v3", "large-v3", "gigaam-v3", "qwen3-asr", "t-one"],
    "de": ["large-v3", "de-turbo", "canary-v2", "qwen3-asr"],
}
# temperature 0 only: forced German on Russian speech fails the compression/log-prob checks on
# every segment and re-decodes it up to five times, for text that is only evidence (docs/DECISIONS.md)
DE_CHECK = dict(name="de-check", backend="whisper", model="de-turbo", lang="de", weight=0.0, cost=0.2,
                temperature=(0.0,))

GAP_COST = 0.65          # alignment: a word missing in one hypothesis
GAP_VOTE = 0.75          # "nothing said here" vote, relative to a word vote
CROSS_SCRIPT = 0.85      # share of a Cyrillic vote given to a similar German word
DICT_TERM = 1.0          # extra share of its own vote for a Latin word spelled exactly as a dictionary term
DICT_SIM = 0.6           # ... only over Cyrillic words that are no Russian words or sound this much like it
NEAR = 2.0               # s: a model with no word this close to a slot skipped the passage
NO_GAP: set[str] = set()  # members whose silence is no "nothing said here" vote (lab experiment 12)

MIXED_LANGS = ("ru", "de")  # "mixed" = Russian speech with German words and phrases

Progress = Callable[[float, str], None]


def _cfg(opts) -> dict:
    return (opts.extra or {}).get("ensemble") or {}


def members(opts) -> list[dict]:
    """Installed members for these options (opts.extra["ensemble"] may exclude or re-weight)."""
    cfg = _cfg(opts)
    out = []
    for m in MEMBERS:
        if m["name"] in cfg.get("exclude", ()) or models.find(m["model"]) is None:
            continue
        m = dict(m)
        if m["name"] in cfg.get("weights", {}):
            m["weight"] = float(cfg["weights"][m["name"]])
            m.pop("by_lang", None)
        out.append(m)
    return out


def available(opts) -> bool:
    ms = members(opts)
    return len(ms) >= 2 and any(m["backend"] == "whisper" for m in ms)


# ---------------------------------------------------------------- chunks

def speech_regions(audio: np.ndarray) -> list[tuple[float, float]]:
    """Soft Silero VAD: speech regions in seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    vo = VadOptions(threshold=0.35, min_speech_duration_ms=200, min_silence_duration_ms=300,
                    speech_pad_ms=200, max_speech_duration_s=MAX_CHUNK - 1)
    return [(t["start"] / SAMPLE_RATE, t["end"] / SAMPLE_RATE) for t in get_speech_timestamps(audio, vo)]


def make_chunks(audio: np.ndarray, cuts: list[float] | tuple = (),
                regions: list[tuple[float, float]] | None = None) -> list[dict]:
    """Speech regions glued into chunks <= MAX_CHUNK, split at pauses; `cuts` (language
    switches) force a split at the pause closest to them."""
    if regions is None:
        regions = speech_regions(audio)
    dur = len(audio) / SAMPLE_RATE
    regions = list(regions)
    for c in cuts:
        # no pause near the switch: split the speech at the quietest 100 ms around it
        near = any(abs(regions[k - 1][1] - c) <= 1.0 or abs(regions[k][0] - c) <= 1.0
                   for k in range(1, len(regions)))
        inside = [k for k, (r0, r1) in enumerate(regions) if r0 + 1.0 < c < r1 - 1.0]
        if not near and inside:
            k = inside[0]
            lo, hi = max(regions[k][0] + 0.5, c - 1.5), min(regions[k][1] - 0.5, c + 1.5)
            hop = SAMPLE_RATE // 10
            seg = audio[int(lo * SAMPLE_RATE): int(hi * SAMPLE_RATE)]
            n = len(seg) // hop
            if n >= 2:
                e = np.sqrt((seg[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
                t = lo + (int(np.argmin(e)) + 0.5) * hop / SAMPLE_RATE
                r0, r1 = regions[k]
                regions[k:k + 1] = [(r0, t - 0.05), (t + 0.05, r1)]
    forced = set()
    for c in cuts:
        bounds = [(abs((regions[k - 1][1] + regions[k][0]) / 2 - c), k) for k in range(1, len(regions))]
        if bounds:
            d, k = min(bounds)
            if d <= 3.0:
                forced.add(k)
    groups: list[list[tuple[float, float]]] = []
    cur: list[tuple[float, float]] = []
    for ri, r in enumerate(regions):
        if cur and (r[0] - cur[-1][1] > MAX_GAP or ri in forced):
            groups.append(cur)
            cur = []
        cur.append(r)
        while len(cur) > 1 and cur[-1][1] - cur[0][0] > MAX_CHUNK:
            # cut at the widest pause that still fits (a model that starts mid-phrase
            # tends to skip it), slightly preferring longer chunks
            best, cut = -1e9, 1
            for k in range(1, len(cur)):
                if cur[k - 1][1] - cur[0][0] > MAX_CHUNK:
                    break
                length = cur[k - 1][1] - cur[0][0]
                score = cur[k][0] - cur[k - 1][1] + 0.02 * length - (1.0 if length < 5 else 0.0)
                if score > best:
                    best, cut = score, k
            groups.append(cur[:cut])
            cur = cur[cut:]
    if cur:
        groups.append(cur)
    starts = {regions[k][0] for k in forced}
    chunks = [dict(start=g[0][0], end=g[-1][1], lang_cut=g[0][0] in starts) for g in groups]
    for i, c in enumerate(chunks):
        # each chunk owns the time up to the middle of the pause to its neighbours
        lo = (chunks[i - 1]["end"] + c["start"]) / 2 if i else 0.0
        hi = (c["end"] + chunks[i + 1]["start"]) / 2 if i + 1 < len(chunks) else dur
        c["own"] = [lo, hi]
    for i, c in enumerate(chunks):
        a, b = c["start"], c["end"]
        # cut inside running speech: hear a bit of the neighbour too (models skip words they
        # hear without the beginning of the phrase); words outside "own" are dropped later
        # (not across a language switch: the neighbour's words are in the other language)
        if i and c["start"] - chunks[i - 1]["end"] < 0.5 and not c["lang_cut"]:
            a = c["start"] - OVERLAP
        if i + 1 < len(chunks) and chunks[i + 1]["start"] - c["end"] < 0.5 and not chunks[i + 1]["lang_cut"]:
            b = c["end"] + OVERLAP
        c["start"] = round(max(min(a, c["start"] - CHUNK_PAD), c["own"][0] if a == c["start"] else 0.0, 0.0), 3)
        c["end"] = round(min(max(b, c["end"] + CHUNK_PAD), c["own"][1] if b == c["end"] else dur, dur), 3)
        c["own"] = [round(c["own"][0], 3), round(c["own"][1], 3)]
        c["speech"] = [[round(max(r0, c["start"]), 3), round(min(r1, c["end"]), 3)]
                       for r0, r1 in regions if r1 > c["start"] and r0 < c["end"]]
    return chunks


def level(x: np.ndarray) -> np.ndarray:
    """Slow, gentle leveling for phone recordings: lifts the quiet far
    side towards the near one (about 2:1 above the noise floor), never boosts silence."""
    if len(x) < SAMPLE_RATE // 2:
        return x
    hop = SAMPLE_RATE // 100  # 10 ms
    n = len(x) // hop
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms + 1e-9)
    k = 30  # 300 ms smoothing
    sm = np.convolve(db, np.ones(k) / k, mode="same")
    floor = np.percentile(sm, 15) + 6.0
    speech = sm > floor
    if not speech.any():
        return x
    target = np.percentile(sm[speech], 90)
    gain = np.clip((target - sm) * 0.5, 0.0, 15.0)
    gain[~speech] = 0.0
    gain = np.convolve(gain, np.ones(k) / k, mode="same")
    g = np.interp(np.arange(len(x)), np.arange(n) * hop + hop / 2, 10 ** (gain / 20))
    y = (x * g).astype(np.float32)
    peak = float(np.abs(y).max()) or 1.0
    if peak > 0.97:
        y *= 0.97 / peak
    return y


def _chunk_audio(audio: np.ndarray, c: dict, lev: bool) -> np.ndarray:
    x = audio[int(c["start"] * SAMPLE_RATE): int(c["end"] * SAMPLE_RATE)]
    return level(x) if lev else x


def _langs(det: dict | None, text: str, fixed: str | None,
           allowed: tuple[str, ...] | None = None, parts: list[dict] | None = None) -> tuple[str, list[str]]:
    """Primary language of a chunk and the languages present in it. In mixed mode the
    recording is Russian and German, so a short "угу" taken for English stays Russian."""
    if fixed:
        return fixed, [fixed]
    det = det or {}
    if allowed and "ru" in allowed and "de" in allowed:
        return _langs_mixed(det, text, parts)
    if allowed:
        det = {k: v for k, v in det.items() if k in allowed} or {allowed[0]: 1.0}
    top = max(det, key=det.get) if det else "ru"
    lat = T.latin_share(text) if text.strip() else None
    if lat is None:
        primary = top if det.get(top, 0) >= 0.5 or top in ("ru", "de") else "ru"
    elif lat >= 0.5:
        primary = "de" if top in ("ru", "de", "uk") or det.get("de", 0) >= 0.1 else top
    else:
        primary = "ru" if top in ("ru", "de", "uk", "be", "bg") or det.get(top, 0) < 0.6 else top
    langs = {primary}
    if (lat is not None and 0.12 <= lat < 0.5) or det.get("de", 0) >= 0.3:
        langs.add("de")
    if (lat is not None and 0.5 <= lat < 0.85) or det.get("ru", 0) >= 0.3:
        langs.add("ru")
    for part in parts or []:  # the beginning or the end is in another language
        part = {k: v for k, v in part.items() if not allowed or k in allowed}
        if part:
            top_p = max(part, key=part.get)
            if part[top_p] >= 0.5:
                langs.add(top_p)
    return primary, sorted(langs)


_DE_WORD = re.compile(r"[äöüßÄÖÜ]|^[A-ZÄÖÜ][a-zäöüß]{5,}$")


def _german_words(text: str) -> bool:
    """Latin words that look German (a capitalized noun, umlauts) or are project terms -
    not the English filler large-v3 sprinkles over Russian ("Me", "emailsum")."""
    terms = {T.key(t) for t in settings.dictionary()["terms"] if T.LATIN.search(t)}
    function_words = set()
    start = True  # a capital at a sentence start says nothing ("Thanks for watching!")
    for raw in text.split():
        w = T.PUNCT_TAIL.sub("", raw).strip("«\"(")
        if T.LATIN.search(w) and (T.key(w) in terms or re.search(r"[äöüßÄÖÜ]", w)
                                  or (not start and _DE_WORD.search(w))):
            return True
        start = bool(re.search(r"[.!?…]$", raw.rstrip("»\")")))
        if T.key(w) in _DE_FUNCTION:
            function_words.add(T.key(w))
    return len(function_words) >= 2


_DE_FUNCTION = {"ich", "und", "der", "die", "das", "nicht", "ist", "er", "wir", "auch", "aber", "mit", "für",
                "auf", "dann", "hat", "haben", "wird", "kann", "noch", "nur", "ein", "eine", "wenn", "oder",
                "bitte", "danke", "genau", "jetzt", "schon", "mal", "wie", "was", "bei", "vom", "zum", "den"}


def _langs_mixed(det: dict, text: str, parts: list[dict] | None) -> tuple[str, list[str]]:
    """Russian call with German words and phrases. German only on the detector's word:
    on noise, backchannels and the quiet far side large-v3 hallucinates English, Portuguese
    or Polish ("Thank you for watching", "Obrigado"), and Latin script alone used to turn such
    a chunk German - the Russian models were then not even asked, and German ones filled it
    with "Vielen Dank". Real German speech is detected at 0.9+; a tie means both languages."""
    de, ru = det.get("de", 0.0), det.get("ru", 0.0)
    lat = T.latin_share(text) if text.strip() else 0.0
    if de >= 0.5:
        langs = {"de"} | ({"ru"} if ru >= 0.2 or (T.CYRILLIC.search(text) and lat < 0.85) else set())
        return "de", sorted(langs)
    if ru < 0.5 and de >= 0.2 and de >= ru:
        return "de", ["de", "ru"]
    langs = {"ru"}
    # German words inside Russian speech (Straßenbahn, Einkaufen): German models vote too
    if de >= 0.15 or _german_words(text):
        langs.add("de")
    for part in parts or []:  # the beginning or the end is German
        if part.get("de", 0.0) >= 0.5:
            langs.add("de")
    return "ru", sorted(langs)


def _eligible(m: dict, ch: dict) -> bool:
    only = m.get("only")
    return not only or bool(set(only) & set(ch["langs"]))


def _call_lang(m: dict, ch: dict, fixed: str | None) -> str | None:
    if m["lang"] == "auto":
        return fixed
    if m["lang"] == "chunk":
        return ch["lang"]
    if m["lang"] == "single":
        return fixed or (ch["lang"] if len(ch["langs"]) == 1 else None)
    return m["lang"]


# ---------------------------------------------------------------- running members

def _load_state(cache: Path | None, chunks: list[dict], dur: float, lev: bool) -> dict:
    if cache and cache.exists():
        try:
            st = json.loads(cache.read_text("utf-8"))
            same = len(st.get("chunks", [])) == len(chunks) and all(
                abs(a["start"] - b["start"]) < 0.01 and abs(a["end"] - b["end"]) < 0.01
                for a, b in zip(st["chunks"], chunks))
            if st.get("version") == VERSION and abs(st.get("duration", 0) - dur) < 0.05                     and st.get("level") == lev and same:
                return st
        except Exception:  # noqa: BLE001 - broken cache: recognize again
            pass
    return dict(version=VERSION, duration=round(dur, 3), level=lev, chunks=chunks, hyps={}, errors={},
                timing={})


_st_lock = threading.RLock()  # the CPU lane (parallel_cpu) shares a run's state with the GPU lane


def _save_state(cache: Path | None, st: dict) -> None:
    if not cache:
        return
    with _st_lock:
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False), "utf-8")
        tmp.replace(cache)


def _stale(h: dict, m: dict, c: dict, fixed: str | None) -> bool:
    """A cached answer got with another language than the chunk now asks for (the chunk
    language rules changed): a model told "German" about Russian speech must hear it again."""
    if m["name"] == "de-check" or "lang" not in c:
        return False
    want = _call_lang(m, c, fixed)
    if "called" in h:
        return h["called"] != want
    # caches from before "called" was kept: only members told the chunk's language can be off
    return m["lang"] in ("chunk", "single") and want is not None and h.get("language") not in (None, want)


def _todo(m: dict, st: dict, fixed: str | None, only_chunks: set[int] | None = None) -> list[int]:
    """Chunks this member still has to hear (nothing cached, or cached with another language)."""
    chunks = st["chunks"]
    res = st["hyps"].get(m["name"]) or [None] * len(chunks)
    return [i for i, c in enumerate(chunks)
            if (res[i] is None or _stale(res[i], m, c, fixed)) and (only_chunks is None or i in only_chunks)
            and (m["name"] == "de-check" or _eligible(m, c))]


def _run_member(m: dict, st: dict, audio: np.ndarray, prompt: str | None, fixed: str | None,
                rep: Progress, only_chunks: set[int] | None = None, label: str = "",
                keep: bool = False, backend=None) -> None:
    """Recognize the chunks this member still lacks. keep: leave the model loaded (the next
    run uses the same one: de-turbo, then the German re-check). backend: already loaded by the
    caller (a batch of recordings shares one load), then it is neither loaded nor unloaded here."""
    chunks = st["chunks"]
    with _st_lock:
        res = st["hyps"].setdefault(m["name"], [None] * len(chunks))
    todo = _todo(m, st, fixed, only_chunks)
    if not todo:
        return
    t0 = time.monotonic()
    waited0 = gpu.stats["waited"]
    own = backend is None
    b = asr.make(m) if own else backend
    if not own and hasattr(b, "member"):  # a shared load (de-turbo for the German re-check): this member's decoding
        b.member = m
    label = label or m["name"]
    rep(0.0, label)
    t_load = None if own else t0
    ok = False
    use_prompt = m.get("prompt", m["backend"] == "whisper")
    try:
        if own:
            with prefetch.loading(m["model"]):
                b.load(lambda msg: rep(0.0, msg))
            t_load = time.monotonic()
        # backends that take several chunks per call (ONNX, Qwen3-ASR): only without the
        # previous words as prompt, and never the language detector (it runs per chunk)
        size = getattr(b, "batch", 1)
        batched = size > 1 and not use_prompt and m["name"] != st.get("detector")
        done = 0
        rest = [] if batched else todo
        for group in _batches(todo, chunks, size, getattr(b, "batch_seconds", 0.0),
                              lambda i: _call_lang(m, chunks[i], fixed)) if batched else []:
            xs = [_chunk_audio(audio, chunks[i], st["level"]) for i in group]
            lang = _call_lang(m, chunks[group[0]], fixed)
            try:
                hs = b.recognize_batch(xs, lang, None)
            except Exception as e:  # noqa: BLE001
                if not gpu.is_oom(e):
                    raise
                # a batch didn't fit into video memory: the rest goes one chunk at a time
                # (the loop below reloads the model if even one chunk doesn't fit)
                gc.collect()
                rest = [i for i in todo if res[i] is None]
                break
            for i, h in zip(group, hs, strict=True):
                _store_hyp(res, i, h, chunks[i], lang)
            done += len(group)
            rep(done / len(todo), label)
        for n, i in enumerate(rest):
            c = chunks[i]
            x = _chunk_audio(audio, c, st["level"])
            p = prompt if use_prompt else None
            prev = res[i - 1] if i else None
            if use_prompt and prev and prev.get("text") and c["start"] - chunks[i - 1]["end"] < 3.0:
                # Whisper is steadier with the previous words as context (like condition_on_previous_text)
                tail = prev["text"][-200:]
                tail = tail[tail.find(" ") + 1:] if len(prev["text"]) > 200 else tail
                p = f"{prompt} {tail}" if prompt else tail
            try:
                h = b.recognize(x, _call_lang(m, c, fixed), p)
            except Exception as e:  # noqa: BLE001
                if not gpu.is_oom(e):
                    raise
                # video memory ran out mid-run (another program, or the allocator's cache after
                # a long chunk): reload the model and try this chunk once more; if it fails again,
                # leave the chunk to the other members instead of losing the rest of the recording
                rep(n / len(todo), f"{label}: не хватило видеопамяти, перезагружаю модель")
                b.unload()
                gc.collect()
                gpu.wait(2.5, lambda msg: rep(n / len(todo), msg), limit=300)
                b.load(lambda msg: rep(n / len(todo), msg))
                try:
                    h = b.recognize(x, _call_lang(m, c, fixed), p)
                except Exception as e2:  # noqa: BLE001
                    if not gpu.is_oom(e2):
                        raise
                    st.setdefault("oom_skipped", {}).setdefault(m["name"], []).append(i)
                    continue
            if m["name"] == st.get("detector") and hasattr(b, "detect"):
                h["det"] = b.detect(x)
                h["det_parts"] = b.detect_parts(x)
            _store_hyp(res, i, h, c, _call_lang(m, c, fixed))
            rep((n + 1) / len(todo), label)
        ok = True
    finally:
        device = getattr(b, "device", None)
        if own and not (keep and ok):
            b.unload()
        del b
        gc.collect()
        t1 = time.monotonic()
        with _st_lock:
            _add_timing(st, m["name"], t0, t_load, t1, gpu.stats["waited"] - waited0, len(todo),
                        sum(chunks[i]["end"] - chunks[i]["start"] for i in todo), device)
    with _st_lock:
        st["timing"][m["name"]] = round(st["timing"].get(m["name"], 0) + t1 - t0, 1)


def _store_hyp(res: list, i: int, h: dict, c: dict, called: str | None) -> None:
    h["called"] = called
    for w in h.get("words") or []:
        w["s"] = round(w["s"] + c["start"], 3)
        w["e"] = round(w["e"] + c["start"], 3)
    res[i] = h


def _batches(todo: list[int], chunks: list[dict], size: int, seconds: float,
             lang_of: Callable[[int], str | None]) -> list[list[int]]:
    """Chunks for one call each: the same language, at most `size` chunks and `seconds` of
    audio (0 = no limit); similar lengths together, so little of a batch is padding."""
    by_lang: dict = {}
    for i in todo:
        by_lang.setdefault(lang_of(i), []).append(i)
    out = []
    for idx in by_lang.values():
        idx = sorted(idx, key=lambda i: chunks[i]["end"] - chunks[i]["start"])
        cur, total = [], 0.0
        for i in idx:
            d = chunks[i]["end"] - chunks[i]["start"]
            if cur and (len(cur) >= size or (seconds and total + d > seconds)):
                out.append(cur)
                cur, total = [], 0.0
            cur.append(i)
            total += d
        if cur:
            out.append(cur)
    return out


def _add_timing(st: dict, name: str, t0: float, t_load: float | None, t1: float, waited: float,
                chunks: int, audio_s: float, device: str | None) -> None:
    """st["timing_detail"][name]: where a member's time went (seconds, summed over runs).
    load = from start to the model being ready (wait = part of it spent waiting for video
    memory), run = recognition, audio = seconds of audio heard. st["timing"] keeps the total.
    In a batch the shared load is booked on the first recording that needed it (run_batch)."""
    d = st.setdefault("timing_detail", {}).setdefault(name, dict(load=0.0, wait=0.0, run=0.0, chunks=0,
                                                                  audio=0.0))
    load = (t_load if t_load is not None else t1) - t0
    d["load"] = round(d["load"] + load, 1)
    d["wait"] = round(d["wait"] + waited, 1)
    d["run"] = round(d["run"] + (t1 - t_load if t_load is not None else 0.0), 1)
    d["chunks"] += chunks
    d["audio"] = round(d["audio"] + audio_s, 1)
    if device:
        d["device"] = device


# A run = one track of one recording going through the ensemble:
#   r = begin(audio, opts, progress, cache)   language scan, chunks, cached state
#   run_member(r, 0)                          the detector: also gives every chunk its language
#   run_member(r, k) for the other members    (in any order: no member depends on another one)
#   finish(r)                                 merge, German re-check, merge again -> (segs, meta)
# transcribe() does exactly that; pipeline.process_batch interleaves several runs model by model.

def begin(audio: np.ndarray, opts, progress: Progress | None = None, cache: Path | None = None) -> dict:
    rep = progress or (lambda f, m: None)
    cfg = _cfg(opts)
    lev = bool(cfg.get("level", True))
    fixed = None if (opts.mixed or not opts.language) else opts.language
    allowed = MIXED_LANGS if opts.mixed else None
    dur = len(audio) / SAMPLE_RATE
    mem = members(opts)
    prompt = engine.prompt(opts)[0]
    # the first Whisper with automatic language is the language detector
    det = next((m for m in mem if m["backend"] == "whisper" and m["lang"] == "auto"), None)
    if det:
        mem.remove(det)
        mem.insert(0, det)
    # the German re-check uses de-turbo again: run de-turbo last and keep it loaded for it
    # (one model read less from the disk). The order of the runs does not change the merge.
    run = list(mem)
    reuse = next((m for m in run[1:] if m["model"] == DE_CHECK["model"] and m["backend"] == DE_CHECK["backend"]),
                 None) if not cfg.get("no_de_check") else None
    if reuse:
        run.remove(reuse)
        run.append(reuse)
    # mem: the order of the runs; meta_mem: the members in the order the meta lists them
    r = dict(audio=audio, opts=opts, rep=rep, cache=cache, cfg=cfg, fixed=fixed, allowed=allowed, dur=dur,
             mem=run, meta_mem=mem, reuse=reuse["name"] if reuse else None, prompt=prompt,
             total=sum(m["cost"] for m in run) + 0.3, prog={}, active={})
    prefetch.ahead(upcoming(r, 0))  # the members on the slow disk, while the scan runs on large-v3
    regions = speech_regions(audio)
    scan = None
    scan_time = None
    if det and not fixed:
        scan = _cached_scan(cache, dur)
        if scan is None:
            rep(0.0, "Определяю язык по ходу записи")
            t0, waited0, t_load = time.monotonic(), gpu.stats["waited"], None
            try:
                b = asr.make(det)
                with prefetch.loading(det["model"]):
                    b.load(lambda msg: rep(0.0, msg))  # stays loaded for its own run right after
                t_load = time.monotonic()
                scan = _lang_scan(b, audio, regions, allowed)
            except Exception:  # noqa: BLE001 - without the scan chunks just ignore language switches
                scan = None
            finally:
                # drop our reference (the engine keeps the model for the run that follows):
                # a forgotten one keeps 3.5 GB of video memory busy for the rest of the job
                b = None
                gc.collect()
                scan_time = (t0, t_load, time.monotonic(), gpu.stats["waited"] - waited0)
    chunks = make_chunks(audio, _switches(scan) if scan else (), regions)
    st = _load_state(cache, chunks, dur, lev)
    st["scan"] = scan
    if scan_time:
        # the detector's model is read here (its own run then finds it loaded): load + scan
        t0, t_load, t1, waited = scan_time
        _add_timing(st, "scan", t0, t_load, t1, waited, 0, sum(r1 - r0 for r0, r1 in regions),
                    engine.device()[0])
    if det:
        st["detector"] = det["name"]
    else:
        _assign_langs(st, fixed, allowed)
    r["st"] = st
    return r


def upcoming(r: dict, k: int) -> list[str]:
    """Models this run still loads after member k, in order (what to prefetch)."""
    st = r.get("st")
    known = bool(st and st["chunks"] and "langs" in st["chunks"][0])
    keys = [m["model"] for m in r["mem"][k + 1:] if not known or _todo(m, st, r["fixed"])]
    if not r["cfg"].get("no_de_check"):
        keys.append(DE_CHECK["model"])
    return keys + list(r.get("tail") or ())


def _progress(r: dict, name: str, lane: str = "main") -> Progress:
    """Progress of member `name` inside the run: done members + this one's share; with two
    lanes the stage shows both models."""
    cost = {m["name"]: m["cost"] for m in r["mem"]}

    def rep(f: float, msg: str) -> None:
        r["prog"][name] = f
        r["active"][lane] = msg
        frac = sum(cost.get(n, 0) * x for n, x in list(r["prog"].items())) / r["total"]
        r["rep"](min(frac, 0.99), " | ".join(v for v in list(r["active"].values()) if v))

    return rep


def label(r: dict, k: int) -> str:
    m = r["mem"][k]
    return f"Распознаю речь · модель {k + 1} из {len(r['mem'])}: {m['name']}{r.get('label_suffix', '')}"


def run_member(r: dict, k: int, backend=None, lane: str = "main") -> None:
    """Member k of the run; de-turbo (r["reuse"]) stays loaded for the German re-check."""
    m, st = r["mem"][k], r["st"]
    with _st_lock:
        st["errors"].pop(m["name"], None)
    try:
        _run_member(m, st, r["audio"], r["prompt"], r["fixed"], _progress(r, m["name"], lane),
                    label=label(r, k), keep=m["name"] == r.get("reuse"), backend=backend)
    except Exception as e:  # noqa: BLE001 - one model failing must not sink the job
        member_failed(r, m["name"], e)
    r["prog"][m["name"]] = 1.0
    r["active"].pop(lane, None)
    if k == 0:
        _assign_langs(st, r["fixed"], r["allowed"])
    _save_state(r["cache"], st)


def member_failed(r: dict, name: str, e: BaseException) -> None:
    with _st_lock:
        r["st"]["errors"][name] = f"{type(e).__name__}: {e}"
        r["st"].setdefault("tracebacks", {})[name] = traceback.format_exc()[-3000:]


def on_cpu(m: dict) -> bool:
    """An ONNX member that stays on the CPU: with onnxruntime-gpu one that goes to the GPU
    (asr.onnx_models.prefers_cuda) belongs with the GPU members, not in the CPU thread."""
    if m["backend"] != "onnx":
        return False
    from .asr import onnx_models

    return not onnx_models.prefers_cuda(m)


def lanes(mem: list[dict], parallel: bool) -> list[list[int]]:
    """Member indexes split into lanes that may run at the same time: the ONNX members on the
    CPU don't touch the video memory, so with parallel_cpu they work while the GPU ones do."""
    if not parallel:
        return [list(range(len(mem)))]
    cpu = [k for k, m in enumerate(mem) if on_cpu(m)]
    gpu_ = [k for k in range(len(mem)) if k not in cpu]
    if not cpu or not gpu_:
        return [list(range(len(mem)))]
    return [gpu_, cpu]


def cpu_member(m: dict) -> dict:
    """A CPU-lane member: the ONNX thread pool must not spin (busy-wait) and starve the thread
    feeding the GPU. Same threads, same numbers, so the same text."""
    return dict(m, no_spin=True)


def de_check_need(r: dict) -> set[int]:
    st = r["st"]
    if st["chunks"] and "lang" not in st["chunks"][0]:
        _assign_langs(st, r["fixed"], r["allowed"])
    r["rep"]((r["total"] - 0.3) / r["total"], "Перепроверяю немецкие слова")
    r["segs"] = combine(st, r["opts"])
    need = _de_check_chunks(st, r["segs"])
    if not need or not models.find(DE_CHECK["model"]) or r["cfg"].get("no_de_check"):
        need = set()
    r["need"] = need
    return need


def finish(r: dict, backend=None, unload: bool = True) -> tuple[list[dict], dict]:
    """German re-check of words no Russian dictionary knows, the final merge and the meta."""
    st, opts = r["st"], r["opts"]
    try:
        need = r["need"] if "need" in r else de_check_need(r)
        segs = r["segs"]
        if need:
            done = (r["total"] - 0.3) / r["total"]
            try:
                _run_member(DE_CHECK, st, r["audio"], None, "de",
                            lambda f, msg: r["rep"](done + 0.2 / r["total"] * f, msg), need,
                            label="Перепроверяю немецкие слова" + r.get("label_suffix", ""), backend=backend)
                _save_state(r["cache"], st)
                segs = combine(st, opts)
            except Exception as e:  # noqa: BLE001
                st["errors"]["de-check"] = f"{type(e).__name__}: {e}"
    finally:
        if r.get("reuse") and unload:  # de-turbo kept for the re-check: free the video memory
            engine.unload()
    _save_state(r["cache"], st)
    r["rep"](1.0, "Свожу гипотезы")
    return segs, _meta(st, opts, r["dur"], r["meta_mem"])


def _parallel() -> bool:
    return bool(settings.speed().get("parallel_cpu"))


def run_lanes(groups: list[list], work) -> None:
    """Run work(group) for every lane, the first in this thread, the others in threads."""
    if len(groups) == 1:
        work(groups[0])
        return
    errors: list[BaseException] = []

    def safe(g):
        try:
            work(g)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=safe, args=(g,), name=f"lane-{i}", daemon=True)
               for i, g in enumerate(groups[1:], 1)]
    for t in threads:
        t.start()
    safe(groups[0])
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def transcribe(audio: np.ndarray, opts, progress: Progress | None = None,
               cache: Path | None = None, tail: list | None = None) -> tuple[list[dict], dict]:
    r = begin(audio, opts, progress, cache)
    r["tail"] = tail or []
    if not r["mem"]:
        return finish(r)
    prefetch.ahead(upcoming(r, 0))
    run_member(r, 0)
    rest = list(range(1, len(r["mem"])))
    groups = lanes([r["mem"][k] for k in rest], _parallel())
    if len(groups) > 1:
        for j in groups[1]:
            r["mem"][rest[j]] = cpu_member(r["mem"][rest[j]])

    def work(group: list[int]) -> None:
        lane = "gpu" if len(groups) > 1 and group is groups[0] else ("cpu" if len(groups) > 1 else "main")
        ks = [rest[j] for j in group]
        for n, k in enumerate(ks):
            # prefetch follows the big (GPU) models; the CPU lane's few hundred MB load as they come
            if lane != "cpu":
                later = [r["mem"][x]["model"] for x in ks[n + 1:] if _todo(r["mem"][x], r["st"], r["fixed"])]
                prefetch.ahead(later + upcoming(r, len(r["mem"]) - 1))
            run_member(r, k, lane=lane)

    run_lanes(groups, work)
    prefetch.ahead(upcoming(r, len(r["mem"]) - 1))
    return finish(r)


def run_batch(runs: list[dict], log: Callable[[str], None] | None = None) -> None:
    """Model-major ("batch"): members 1.. of several runs, every one already through begin() and
    run_member(r, 0). Each model is loaded once and hears every run's chunks, then the next model;
    the German re-check goes the same way, then every run is finished: r["result"] = (segs, meta).
    A run that breaks (its recording was deleted meanwhile, a disk error) is dropped with
    r["failed"] = the exception; the others go on."""
    log = log or (lambda msg: None)

    def alive(r: dict) -> bool:
        return not r.get("failed") and (r["cache"] is None or r["cache"].parent.exists())

    def guarded(r: dict, fn) -> None:
        if not alive(r):
            r.setdefault("failed", FileNotFoundError("запись удалена"))
            return
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            r["failed"] = e
            r["failed_tb"] = traceback.format_exc()
            log(f"batch: {r.get('doc_id')} dropped: {type(e).__name__}: {e}")

    def index(r: dict, name: str) -> int | None:
        return next((k for k, m in enumerate(r["mem"]) if k and m["name"] == name), None)

    names: list[str] = []
    backend_of: dict[str, str] = {}
    for r in runs:
        for m in r["mem"][1:]:
            if m["name"] not in names:
                names.append(m["name"])
                backend_of[m["name"]] = m["backend"]
    member_of = {m["name"]: m for r in runs for m in r["mem"][1:]}
    parallel = _parallel()
    cpu = [n for n in names if parallel and on_cpu(member_of[n])]
    gpu_ = [n for n in names if n not in cpu]
    groups = [gpu_, cpu] if cpu and gpu_ else [names]
    reuse = next((r["reuse"] for r in runs if r.get("reuse")), None)
    kept: list = []  # de-turbo left loaded for the German re-check
    tail = []
    for r in runs:
        tail += list(r.get("tail") or ())

    def needed(name: str) -> bool:
        return any(alive(r) and index(r, name) is not None
                   and _todo(r["mem"][index(r, name)], r["st"], r["fixed"]) for r in runs)

    def work(group: list[str]) -> None:
        lane = "main" if len(groups) == 1 else ("gpu" if group is groups[0] else "cpu")
        for pos, name in enumerate(group):
            users = [(r, index(r, name)) for r in runs if alive(r) and index(r, name) is not None]
            need = [(r, k) for r, k in users if _todo(r["mem"][k], r["st"], r["fixed"])]
            needing = {id(r) for r, _ in need}
            for r, k in users:
                if id(r) not in needing:  # nothing to hear: just the bookkeeping, no load
                    guarded(r, lambda r=r, k=k: run_member(r, k, lane=lane))
            if not need:
                continue
            if lane != "cpu":
                nxt = [r0["mem"][index(r0, n)]["model"] for n in group[pos + 1:] if needed(n)
                       for r0 in runs[:1] if index(r0, n) is not None]
                prefetch.ahead(nxt + [DE_CHECK["model"]] + tail)
            r0, k0 = need[0]
            m = r0["mem"][k0]
            b = asr.make(cpu_member(m) if lane == "cpu" else m)
            for r, k in need:
                guarded(r, lambda r=r, k=k: _progress(r, name, lane)(0.0, label(r, k)))
            t0, waited0 = time.monotonic(), gpu.stats["waited"]
            try:
                with prefetch.loading(m["model"]):
                    b.load(lambda msg: [guarded(r, lambda r=r: _progress(r, name, lane)(0.0, msg))
                                        for r, _ in need])
            except Exception as e:  # noqa: BLE001 - like run_member: the member is skipped
                for r, k in need:
                    with _st_lock:  # as _run_member leaves it: an empty answer list and the error
                        r["st"]["hyps"].setdefault(name, [None] * len(r["st"]["chunks"]))
                    member_failed(r, name, e)

                    def skip(r=r):
                        r["prog"][name] = 1.0
                        r["active"].pop(lane, None)
                        _save_state(r["cache"], r["st"])

                    guarded(r, skip)
                log(f"batch: {name} failed to load: {type(e).__name__}: {e}")
                continue
            t1 = time.monotonic()
            with _st_lock:  # the one load, booked on the first recording
                _add_timing(r0["st"], name, t0, t1, t1, gpu.stats["waited"] - waited0, 0, 0.0,
                            getattr(b, "device", None))
            ok = False
            try:
                for r, k in need:
                    guarded(r, lambda r=r, k=k: run_member(r, k, backend=b, lane=lane))
                ok = True
            finally:
                if ok and name == reuse:
                    kept.append(b)
                else:
                    b.unload()
                del b
                gc.collect()
            log(f"batch: {name} load {t1 - t0:.1f}s, {len(need)} recordings {time.monotonic() - t1:.1f}s")

    run_lanes(groups, work)

    live = [r for r in runs if alive(r)]
    for r in live:
        guarded(r, lambda r=r: de_check_need(r))
    need = [r for r in live if alive(r) and r.get("need")]
    b = None
    if need:
        prefetch.ahead(tail)
        b = asr.make(DE_CHECK)
        t0, waited0 = time.monotonic(), gpu.stats["waited"]
        try:
            with prefetch.loading(DE_CHECK["model"]):
                b.load(lambda msg: None)  # finds de-turbo still loaded (kept) when it ran
            t1 = time.monotonic()
            _add_timing(need[0]["st"], DE_CHECK["name"], t0, t1, t1, gpu.stats["waited"] - waited0, 0, 0.0,
                        getattr(b, "device", None))
            log(f"batch: de-check load {t1 - t0:.1f}s")
        except Exception as e:  # noqa: BLE001 - finish() would have stored the same error
            for r in need:
                r["st"]["errors"]["de-check"] = f"{type(e).__name__}: {e}"
                r["need"] = set()
            b = None
    try:
        for r in live:  # the batch frees its models here, not finish()
            guarded(r, lambda r=r: r.__setitem__("result", finish(r, backend=b if r.get("need") else None,
                                                                  unload=False)))
    finally:
        if b is not None:
            b.unload()
            b = None
        elif kept:
            engine.unload()
        kept.clear()
        gc.collect()


SCAN_WIN, SCAN_HOP = 5.0, 2.5


def _cached_scan(cache: Path | None, dur: float) -> list | None:
    try:
        st = json.loads(cache.read_text("utf-8"))
        if st.get("version") == VERSION and abs(st.get("duration", 0) - dur) < 0.05:
            return st.get("scan")
    except Exception:  # noqa: BLE001
        pass
    return None


def _lang_scan(b, audio: np.ndarray, regions: list, allowed: tuple | None) -> list[list]:
    """Whisper language-ID on 5 s windows over the speech: [[centre, lang, prob], ...].
    Language switches become chunk borders, so no model hears two languages at once
    (a chunk that starts in German and goes on in Russian gets translated one way or the other)."""
    out = []
    if not regions:
        return out
    t, end = regions[0][0], regions[-1][1]
    ri = 0
    while t < end:
        a, z = t, t + SCAN_WIN
        sp = sum(max(0.0, min(z, r1) - max(a, r0)) for r0, r1 in regions[ri:] if r0 < z)
        while ri < len(regions) and regions[ri][1] < a:
            ri += 1
        if sp >= 0.5 * SCAN_WIN:
            probs = b.detect(audio[int(a * SAMPLE_RATE): int(z * SAMPLE_RATE)])
            if allowed:
                probs = {k: v for k, v in probs.items() if k in allowed} or {allowed[0]: 0.0}
            top = max(probs, key=probs.get)
            out.append([round(a + SCAN_WIN / 2, 2), top, round(probs[top], 3)])
        t += SCAN_HOP
    return out


def _switches(scan: list) -> list[float]:
    """Times where the language changes and stays changed for at least two windows."""
    langs = [x[1] for x in scan if x[2] >= 0.5]
    times = [x[0] for x in scan if x[2] >= 0.5]
    cuts = []
    i = 1
    while i < len(langs):
        if langs[i] != langs[i - 1] and i + 1 < len(langs) and langs[i + 1] == langs[i]:
            cuts.append(round((times[i - 1] + times[i]) / 2, 2))
        i += 1
    return cuts


def _assign_langs(st: dict, fixed: str | None, allowed: tuple[str, ...] | None = None) -> None:
    det_name = st.get("detector")
    hy = st["hyps"].get(det_name) or [None] * len(st["chunks"])
    for c, h in zip(st["chunks"], hy):
        c["lang"], c["langs"] = _langs((h or {}).get("det"), (h or {}).get("text", ""), fixed, allowed,
                                       (h or {}).get("det_parts"))


def _meta(st: dict, opts, dur: float, mem: list[dict]) -> dict:
    langs = [c.get("lang") for c in st["chunks"]]
    main = max(set(langs), key=langs.count) if langs else opts.language
    return dict(
        model="ensemble", quality="max", mixed=bool(opts.mixed), language=main or opts.language,
        language_probability=round(langs.count(main) / len(langs), 3) if langs else 0.0, duration=dur,
        members=[m["name"] for m in mem if m["name"] in st["hyps"] and m["name"] not in st["errors"]],
        weights={m["name"]: m["weight"] for m in mem}, chunks=len(st["chunks"]),
        chunk_languages={k: langs.count(k) for k in set(langs) if k},
        errors=st["errors"] or None, timing=st["timing"], timing_detail=st.get("timing_detail"),
        level=st["level"],
        prompt=engine.prompt(opts)[0], vad=True, device=engine.device()[0],
    )


def recombine(cache: Path, opts) -> tuple[list[dict], dict]:
    """Merge stored hypotheses again (new weights, new rules) without recognition."""
    st = json.loads(Path(cache).read_text("utf-8"))
    mem = members(opts)
    if st.get("detector"):
        fixed = None if (opts.mixed or not opts.language) else opts.language
        _assign_langs(st, fixed, MIXED_LANGS if opts.mixed else None)
    segs = combine(st, opts)
    return segs, _meta(st, opts, st["duration"], mem)


def merged_words(folder: Path, opts) -> list[dict] | None:
    """The words a merge of the stored hypotheses gives now (no recognition, no GPU), with the
    dictionary replacements the pipeline applies; None without hypotheses (not a "max" transcript)."""
    caches = sorted(Path(folder).glob("hypotheses*.json"))  # one per channel of a stereo call
    if not caches:
        return None
    words = []
    for c in caches:
        segs, _ = recombine(c, opts)
        if opts.dictionary:
            from .store import apply_rules_to_segments

            apply_rules_to_segments(segs, settings.rules())
        words += [w for x in segs for w in x["words"]]
    return sorted(words, key=lambda w: w["s"])


def refresh_doubt(doc: dict, words: list[dict]) -> int:
    """An existing transcript gets the current doubt marks (_doubt): the new word p goes onto the
    words the merge still produces in the same order. Text, times, speakers and edits stay; edited
    words keep theirs. Returns how many words changed."""
    old = [w for x in doc.get("segments") or [] for w in x.get("words") or []]
    a = [T.key(w["w"]) if not w.get("edited") else f"\0{i}" for i, w in enumerate(old)]
    b = [T.key(w["w"]) for w in words]
    n = 0
    for i, j, size in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(size):
            w, p = old[i + k], words[j + k].get("p", 1.0)
            if w.get("p") != p:
                w["p"] = p
                n += 1
    return n


# ---------------------------------------------------------------- merging

def _tokens(h: dict, ch: dict, name: str) -> list[dict]:
    """One hypothesis -> comparable tokens with times (spread over speech if the model has none)."""
    if h.get("words"):
        raw = [dict(w=w["w"], s=w["s"], e=w["e"], p=w.get("p"), timed=True) for w in h["words"]]
    else:
        parts = h.get("text", "").split()
        raw = [dict(w=w, p=None, timed=False) for w in parts]
        _spread(raw, ch["speech"])
    # "2 000" -> "2000": a digit group split by a space is one number
    glued: list[dict] = []
    for t in raw:
        if (glued and re.fullmatch(r"\d{1,3}", glued[-1]["w"]) and re.fullmatch(r"\d{3}(?:\D.*)?", t["w"])):
            glued[-1] = dict(glued[-1], w=glued[-1]["w"] + t["w"], e=t.get("e", glued[-1].get("e")))
            continue
        glued.append(t)
    raw = glued
    toks = []
    prev_end = True
    for t in raw:
        if not T.key(t["w"]):
            if toks and T.tail_punct(t["w"]):  # stand-alone "—", "?" etc.
                toks[-1]["w"] += t["w"].strip()
            continue
        t["first"] = prev_end
        prev_end = bool(re.search(r"[.!?…]$", t["w"].rstrip("»\")")))
        toks.append(t)
    own = ch.get("own")
    if own:  # overlap with the neighbour chunk: only words in our own time
        toks = [t for t in toks if own[0] <= (t["s"] + t["e"]) / 2 < own[1]]
    toks = T.merge_numbers(toks)
    for t in toks:
        t.setdefault("k", T.key(t["w"]))
        t["m"] = name
        t["lat"] = T.is_latin(t["w"])
    return toks


def _spread(raw: list[dict], speech: list[list[float]]) -> None:
    """Text-only hypothesis: place words over the chunk's speech regions by length."""
    if not raw:
        return
    total = sum(e - s for s, e in speech) or 1e-3
    lens = np.array([len(t["w"]) + 1 for t in raw], dtype=float)
    pos = np.concatenate(([0.0], np.cumsum(lens))) / lens.sum() * total

    def at(x):
        for s, e in speech:
            if x <= e - s:
                return s + x
            x -= e - s
        return speech[-1][1]

    for i, t in enumerate(raw):
        t["s"], t["e"] = round(at(pos[i]), 3), round(at(pos[i + 1]), 3)


class _Slot:
    __slots__ = ("toks", "tsum", "tn")

    def __init__(self):
        self.toks: dict[str, dict] = {}
        self.tsum = 0.0
        self.tn = 0.0

    def add(self, t: dict):
        self.toks[t["m"]] = t
        wgt = 1.0 if t["timed"] else 0.25
        self.tsum += wgt * (t["s"] + t["e"]) / 2
        self.tn += wgt

    @property
    def t(self) -> float | None:
        return self.tsum / self.tn if self.tn else None


def _sim(slot: _Slot, t: dict) -> float:
    best = 0.0
    for u in slot.toks.values():
        if u["k"] == t["k"]:
            return 1.0
        if u.get("num") and t.get("num"):
            best = max(best, 0.6)  # two different numbers in the same place: one was misheard
            continue
        if u.get("num") or t.get("num"):
            continue
        best = max(best, T.phon_sim(u["k"], t["k"]))
    return best if best >= 0.6 else 0.0


def _align(slots: list[_Slot], toks: list[dict]) -> list[_Slot]:
    """Needleman-Wunsch of one hypothesis against the network; returns the new slot list."""
    n, m = len(slots), len(toks)
    if not n:
        out = []
        for t in toks:
            s = _Slot()
            s.add(t)
            out.append(s)
        return out
    INF = 1e9
    dp = np.full((n + 1, m + 1), INF)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)
    dp[0, :] = np.arange(m + 1) * GAP_COST
    dp[:, 0] = np.arange(n + 1) * GAP_COST
    bt[0, 1:] = 2
    bt[1:, 0] = 1
    st = [s.t for s in slots]
    for i in range(1, n + 1):
        si, ti = slots[i - 1], st[i - 1]
        for j in range(1, m + 1):
            t = toks[j - 1]
            mid = (t["s"] + t["e"]) / 2
            dt = abs(ti - mid) if ti is not None else 0.0
            if t["timed"]:
                pen = INF if dt > 6.0 else min(max(0.0, dt - 1.0) * 0.25, 0.5)
            else:
                # a text-only model's times are guessed by spreading its words over the speech;
                # where it skipped the other side's reply they drift by many seconds: no cutoff
                pen = min(max(0.0, dt - 3.0) * 0.03, 0.3)
            sub = INF if pen >= INF else dp[i - 1, j - 1] + 1.0 - _sim(si, t) + pen
            dl = dp[i - 1, j] + GAP_COST
            ins = dp[i, j - 1] + GAP_COST
            if sub <= dl and sub <= ins:
                dp[i, j], bt[i, j] = sub, 0
            elif dl <= ins:
                dp[i, j], bt[i, j] = dl, 1
            else:
                dp[i, j], bt[i, j] = ins, 2
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        b = bt[i, j]
        if b == 0:
            ops.append(("m", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif b == 1:
            ops.append(("d", i - 1, None))
            i -= 1
        else:
            ops.append(("i", None, j - 1))
            j -= 1
    ops.reverse()
    out = []
    for op, si, tj in ops:
        if op == "m":
            slots[si].add(toks[tj])
            out.append(slots[si])
        elif op == "d":
            out.append(slots[si])
        else:
            s = _Slot()
            s.add(toks[tj])
            out.append(s)
    return out


def _known(word: str) -> bool | None:
    return T.known_russian(word)


def _compat(scope: tuple | None, latin: bool) -> bool:
    """Can a member with these chunk languages vote on a word in this script at all?"""
    if not scope:
        return True
    return any(x != "ru" for x in scope) if latin else "ru" in scope


LONE_P = 0.5  # own confidence a model needs for words nobody else heard


def _lone_speech(slot: _Slot, win: dict, t0: float, speech: list | None) -> bool:
    if not speech or not any(a - 0.1 <= t0 <= b + 0.1 for a, b in speech):
        return False
    ps = [slot.toks[v]["p"] for v in win["voters"] if slot.toks[v].get("p") is not None]
    return bool(ps) and max(ps) >= LONE_P


def _vote(slot: _Slot, present: dict[str, float], order: list[str],
          scope: dict[str, tuple | None] | None = None, order_lat: list[str] | None = None,
          order_cyr: list[str] | None = None, mids: dict[str, np.ndarray] | None = None,
          speech: list | None = None, term_keys: set[str] | None = None) -> dict | None:
    """Winner of a slot: {"tok", "voters", "score", "p"} or None if "nothing" wins.
    Agreement and "nothing here" only count models that can write the winner's script:
    a Russian-only model's silence says nothing about a German word."""
    scope = scope or {}
    cands: dict[str, dict] = {}
    for name, t in slot.toks.items():
        w = present.get(name, 0.0)
        if w <= 0:
            continue
        p = t["p"] if t.get("p") is not None else 0.7
        c = cands.setdefault(t["k"], dict(score=0.0, base=0.0, voters=[], lat=t["lat"], num=t.get("num")))
        v = w * (0.55 + 0.45 * p)
        c["score"] += v
        c["base"] += v
        c["voters"].append(name)
    if not cands:
        return None
    # Cyrillic spellings of a word Russian doesn't know vote for a similar German word
    lat = [k for k, c in cands.items() if c["lat"] and not c["num"]]
    if lat:
        for k, c in cands.items():
            if c["lat"] or c["num"]:
                continue
            word = slot.toks[c["voters"][0]]["w"]
            if _known(word) is not False:
                continue
            best = max(lat, key=lambda L: T.phon_sim(L, k))
            if T.phon_sim(best, k) >= 0.7:
                cands[best]["score"] += CROSS_SCRIPT * c["base"]
                cands[best].setdefault("phon", []).extend(c["voters"])
    if DICT_TERM and term_keys:
        # a model wrote a dictionary term letter for letter where the others wrote Russian words
        # (only against words that are no Russian words or sound like the term: over «да», «4-го»,
        # «понял» it is large-v3 repeating its prompt)
        cyr = [(x, c) for x, c in cands.items() if not c["lat"]]
        for k in lat:
            if k in term_keys and cyr and all(
                    not c["num"] and (_known(slot.toks[c["voters"][0]]["w"]) is False
                                     or (len(k) >= 5 and T.phon_sim(k, x) >= DICT_SIM))
                    for x, c in cyr):
                cands[k]["score"] += DICT_TERM * cands[k]["base"]
    rank = {n: i for i, n in enumerate(order)}
    k = max(cands, key=lambda k: (cands[k]["score"], -min(rank.get(v, 99) for v in cands[k]["voters"])))
    win = cands[k]
    judges = {n: w for n, w in present.items() if _compat(scope.get(n), win["lat"])}
    others = {n: w for n, w in present.items() if n not in judges}
    silent = dict(judges)
    if others and not any(n in slot.toks for n in others):
        # nobody of the other language heard anything here either: a word one group
        # "translated" into the gap (large-v3 turning a Russian remark into German)
        silent.update(others)
    # A model that said nothing at all around here while others did skipped the passage
    # (Whisper drops whole stretches): it abstains instead of voting "nothing was said".
    # In real silence the others are silent too, so a lone hallucination still loses.
    t0 = slot.t
    abstain = set()
    if mids is not None and t0 is not None:
        heard = [n for n in present if n in slot.toks or (len(mids.get(n, ())) and
                                                         np.abs(mids[n] - t0).min() <= NEAR)]
        if len(heard) >= 2:
            abstain = {n for n in present if n not in heard}
        elif _lone_speech(slot, win, t0, speech):
            # one model heard a stretch all others skipped, but it is speech (VAD) and the model
            # is sure of it: the quiet far side of a call, kept and marked for review
            abstain = {n for n in present if n not in slot.toks}
    gap = GAP_VOTE * sum(w for n, w in silent.items()
                         if n not in slot.toks and n not in abstain and n not in NO_GAP)
    total = sum(w * (0.5 if n in abstain else 1.0) for n, w in judges.items()) or 1.0
    if gap > win["score"]:
        return None
    if order_lat and win["lat"]:
        rank = {n: i for i, n in enumerate(order_lat)}
    elif order_cyr and not win["lat"]:
        rank = {n: i for i, n in enumerate(order_cyr)}
    voters = sorted(win["voters"], key=lambda n: rank.get(n, 99))
    lead = voters[0]
    if win["num"]:  # numbers: digits read better than words ("25-го" over "двадцать пятого")
        lead = next((v for v in voters if re.search(r"\d", slot.toks[v]["w"])), lead)
    return dict(tok=slot.toks[lead], voters=voters, score=win["score"],
                p=round(min(1.0, win["score"] / total), 3), phon=win.get("phon"))


# ---------------------------------------------------------------- doubt
# The word p the user sees (store.REVIEW_P marks it) is the agreement minus what else speaks
# against the winner. Agreement alone caught a fifth of the errors on the owner's gold reference:
# a German term heard as Russian words is agreed on by several models making the same mistake.
# Tuned on the gold reference with chunks left out (docs/DECISIONS.md): few rules, round numbers.
DOUBT_TERM = 0.5      # a Cyrillic word that sounds like a glossary term or its beginning ("анмель")
DOUBT_FOREIGN = 0.5   # not Russian: unknown to the dictionary, or a lone Latin word in Russian speech
DOUBT_NUMBER = 0.2    # the models heard different numbers here
DOUBT_ACOUSTIC = 0.1  # even the surest of its voters was unsure of it (own p < ACOUSTIC_P)
DOUBT_FEW = 0.05      # two models or fewer voted for it
ACOUSTIC_P = 0.5
TERM_SIM = 0.85


class _Terms:
    """Latin glossary terms: their keys (a word spelled as a term is not in doubt) and the sound
    skeletons of the terms and their parts ("Hotel-Anmeldung" -> also "Anmeldung")."""

    def __init__(self, terms: list[str]):
        parts = [x for t in terms for x in [t] + re.split(r"[\s\-]+", t) if x]
        self.keys = {T.key(x) for x in parts}
        # spelled exactly as a term: whole terms and long parts ("Anmeldung"; not "Haus" of "Garten Haus")
        self.exact = {T.key(t) for t in terms} | {k for k in self.keys if len(k) >= 6}
        self.phons = sorted({ph for ph in map(T.phon, parts) if len(ph) >= 5})

    def sounds_like(self, word: str) -> bool:
        """A Cyrillic word that sounds like a term or its beginning ("анмель" ~ Anmeldung,
        "фаррад" ~ Fahrradweg): a German term may have been heard as Russian words here."""
        if not T.CYRILLIC.search(word) or T.number_value(word) is not None:
            return False
        pw = T.phon(word)
        return len(pw) >= 5 and any(SequenceMatcher(None, pw, pt[:len(pw)]).ratio() >= TERM_SIM
                                    for pt in self.phons)


def _unknown(word: str) -> bool:
    """A Cyrillic word the Russian dictionary doesn't know (each part of "такая-то" checked)."""
    parts = [x for x in re.split(r"[-‐]", T.PUNCT_TAIL.sub("", word)) if len(T.key(x)) >= 3]
    return any(T.known_russian(x) is False for x in parts)


def _doubt_slot(win: dict, slot: _Slot, terms: _Terms) -> dict:
    """What speaks against a slot's winner besides the agreement (see _doubt)."""
    t = win["tok"]
    ps = [slot.toks[n]["p"] for n in win["voters"] if slot.toks[n].get("p") is not None]
    num = bool(t.get("num"))
    return dict(
        term=not num and terms.sounds_like(t["w"]),
        unknown=not t["lat"] and not num and _unknown(t["w"]),
        latin=bool(t["lat"]) and not num and t["k"] not in terms.keys,
        number=num and any(u.get("num") and u["k"] != t["k"] for u in slot.toks.values()),
        acoustic=bool(ps) and max(ps) < ACOUSTIC_P,
        few=len(win["voters"]) <= 2,
    )


def _doubt(words: list[dict], lang: str) -> None:
    """Word p = agreement minus penalties (the agreement stays in w["_agree"]). A Latin word
    counts as foreign only alone or in a pair inside Russian speech (a German term the models
    spelled their own way: "Forn Melding"); a whole German or English phrase is judged by the
    agreement alone."""
    run = [0] * len(words)
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and words[j]["_d"]["latin"]:
            j += 1
        for k in range(i, j):
            run[k] = j - i
        i = max(j, i + 1)
    for i, w in enumerate(words):
        d = w["_d"]
        d["foreign"] = d["unknown"] or (d["latin"] and lang != "de" and run[i] <= 2)
        pen = (DOUBT_TERM * d["term"] + DOUBT_FOREIGN * d["foreign"] + DOUBT_NUMBER * d["number"]
               + DOUBT_ACOUSTIC * d["acoustic"] + DOUBT_FEW * d["few"])
        w["_agree"] = w["p"]
        w["p"] = round(max(0.0, w["p"] - pen), 3)


DISAGREE_P = 0.3     # median word agreement below which a segment is marked "disagree"
def _surface(win: dict, slot: _Slot, order: list[str], punct_members: set[str]) -> str:
    t = win["tok"]
    w = t["w"]
    core = T.PUNCT_TAIL.sub("", w) or w
    core = core.lstrip("«\"(")
    # punctuation from the most trusted punctuating model among those who heard this word
    punct = T.tail_punct(w)
    for n in win["voters"]:
        if n in punct_members:
            punct = T.tail_punct(slot.toks[n]["w"])
            break
    punct = re.sub(r"[»\")\]]", "", punct).strip()
    punct = punct[:1] if punct not in ("...", "…") else "…"
    if punct in ("-", "–", "—"):  # dashes between replies, not sentence punctuation
        punct = ""
    return core + punct


def combine(st: dict, opts) -> list[dict]:
    """Stored hypotheses -> merged segments."""
    mem = {m["name"]: m for m in members(opts)}
    punct_members = {n for n, m in mem.items() if m.get("punct", True)}
    terms = _Terms(_glossary(opts))
    digits = not _cfg(opts).get("no_numbers")
    segments: list[dict] = []
    for ci, ch in enumerate(st["chunks"]):
        lang = ch.get("lang") or opts.language or "ru"
        order = SURFACE.get(lang, SURFACE["de" if T.LATIN.search(lang or "") else "ru"])
        order = order + [n for n in mem if n not in order]
        order_lat = SURFACE["de"] + [n for n in mem if n not in SURFACE["de"]]
        order_cyr = SURFACE["ru"] + [n for n in mem if n not in SURFACE["ru"]]
        present = {}
        hyps = {}
        for name, m in mem.items():
            res = st["hyps"].get(name)
            h = res[ci] if res and ci < len(res) else None
            if h is None or not _eligible(m, ch):
                continue
            present[name] = m.get("by_lang", {}).get(lang, m["weight"])
            hyps[name] = _tokens(h, ch, name)
        if not present:
            continue
        scope = {n: mem[n].get("only") for n in present}
        mids = {n: np.array([(t["s"] + t["e"]) / 2 for t in ts]) for n, ts in hyps.items()}
        # backbone: the most trusted model that said something; then the rest in trust order
        names = sorted(hyps, key=lambda n: (not hyps[n], order.index(n) if n in order else 99))
        slots: list[_Slot] = []
        for n in names:
            if hyps[n]:
                slots = _align(slots, hyps[n])
        # a text-only model's times were guessed; where its words joined timed ones they are
        # known now. Without this it seemed to "hear silence" all over a reply it skipped and
        # voted it away (the quiet far side heard by one model only).
        for n, ts in hyps.items():
            if ts and not any(t["timed"] for t in ts):
                at = [s.t for s in slots if n in s.toks and any(u["timed"] for u in s.toks.values())]
                mids[n] = np.array(at) if at else mids[n]
        words = []
        # Word-by-word voting (ROVER) even where the models disagree: taking one model's version
        # of a disputed stretch whole reads better but measured worse (WER +0.5 % on noisy calls);
        # such stretches are marked "disagree" instead (_split).
        wins = [_vote(slot, present, order, scope, order_lat, order_cyr, mids, ch.get("speech"), terms.exact)
                for slot in slots]
        for si, slot in enumerate(slots):
            win = wins[si]
            if win is None:
                continue
            t = win["tok"]
            parts = t.get("parts")
            surf = _surface(win, slot, order, punct_members)
            timed = next((slot.toks[v] for v in win["voters"] + (win["phon"] or []) if slot.toks[v]["timed"]),
                         None)
            src = timed or t
            ws = [surf] if not parts or len(parts) == 1 else surf.split()
            s0, e0 = src["s"], src["e"]
            step = (e0 - s0) / len(ws)
            # someone wrote it in lower case mid-sentence: the capital elsewhere is just a sentence start
            lower = any(not slot.toks[v].get("first") and slot.toks[v]["w"][:1].islower() for v in win["voters"])
            doubt = _doubt_slot(win, slot, terms)
            for k, w in enumerate(ws):
                words.append(dict(w=w, s=round(s0 + k * step, 3), e=round(s0 + (k + 1) * step, 3),
                                  p=win["p"], _slot=si, _first=t.get("first", False), _lat=t["lat"], _lower=lower,
                                  _voters=win["voters"], _timed=timed is not None, _score=win["score"],
                                  _k=t["k"], _d=doubt))
        words = _dedupe(words)
        if not words:
            continue
        _doubt(words, lang)
        _fix_times(words, ch)
        _fix_case(words)
        if digits:  # a number the vote split into its words becomes one word in digits again
            words = itn.words(words)
        segments += _split(words, slots, hyps, ch, lang, order, digits)
    segments = _recheck(segments, st, opts)
    if not _cfg(opts).get("no_terms"):
        _terms_pass(segments, st, opts, mem)
    for s in segments:  # the agreement and the term mark were for the passes above only
        for w in s["words"]:
            w.pop("_agree", None)
            w.pop("term", None)
    return segments


# hypotheses that don't vote but count as evidence for terms: the forced-German re-check, and
# de-turbo let free on every chunk (writes Russian in Cyrillic, German terms in Latin;
# tools/prompt_experiment.py --variant de-mixed)
EVIDENCE_ONLY = (DE_CHECK["name"], "de-mixed")


def _terms_pass(segs: list[dict], st: dict, opts, mem: dict) -> None:
    """German terms every model wrote in Cyrillic, recovered by sound (whisper_studio/terms.py)."""
    from . import terms

    tl = terms.term_list(_glossary(opts), learned=bool(opts.dictionary))
    if not tl:
        return
    words: dict[str, list[dict]] = {}
    for ci, ch in enumerate(st["chunks"]):
        for name, res in st["hyps"].items():
            m = mem.get(name)
            h = res[ci] if res and ci < len(res) else None
            if h is None or (m is None and name not in EVIDENCE_ONLY) or (m is not None and not _eligible(m, ch)):
                continue
            words.setdefault(name, []).extend(_tokens(h, ch, name))
    log: list = []
    focus, loose = _memory_focus(segs, st, opts, tl)
    if focus is not None:
        tl = tl + [t for t in focus.extra_terms if t not in tl]
    st["term_fixes"] = terms.recover(segs, tl, words, log, focus=focus, loose=loose)
    st["term_log"] = log


def _memory_focus(segs: list[dict], st: dict, opts, tl: list[str]):
    """Retrieval memory (H9, whisper_studio/memory.py): per-chunk term subsets from reference
    texts most like this recording's first-pass text. Off unless extra.ensemble.memory (or
    WHISPER_STUDIO_MEMORY_ON=1); without the index or the model: (None, False)."""
    from . import memory, terms

    cfg = _cfg(opts)
    if not memory.enabled(cfg):
        return None, False
    try:
        focus = memory.retrieve(segs, st["chunks"], tl, terms.term_list(_glossary(opts), learned=False),
                                exclude={str(x).lower() for x in cfg.get("memory_exclude") or ()},
                                extras=bool(cfg.get("memory_extras", False)))
    except Exception as e:  # noqa: BLE001 - memory is an extra: never fail the merge over it
        st["memory_error"] = repr(e)
        return None, False
    if focus is None:
        return None, False
    st["memory"] = dict(extra_terms=focus.extra_terms,  # terms and sources only, no passage text
                        chunks=[dict(chunk=r["chunk"], best=r["best"], terms=r["terms"], new=r["new"],
                                     src=[x["src"] for x in r["passages"]]) for r in focus.report])
    return focus, bool(cfg.get("memory_loose", False))


SPLIT_PIECE = True  # "Gartenhaus Haus": a piece of the Latin word before it, written apart by other models


def _piece(u: dict, w: dict) -> bool:
    a, b = u["_k"], w["_k"]
    if not (u["_lat"] and w["_lat"]) or min(len(a), len(b)) < 4 or a == b:
        return False
    short, long_ = sorted((a, b), key=len)
    return long_.endswith(short) or long_.startswith(short)


def _dedupe(words: list[dict]) -> list[dict]:
    """The same word won in two neighbouring slots, voted by different models: alignment
    split it, it was said once. A real repetition ("ты, ты") is heard by the same models."""
    drop = set()
    for i, w in enumerate(words):
        for j in range(max(0, i - 3), i):
            u = words[j]
            if j in drop or set(u["_voters"]) & set(w["_voters"]):
                continue
            if u["_k"] != w["_k"] and not (SPLIT_PIECE and _piece(u, w)):
                continue
            if abs(w["s"] - u["s"]) < 1.0:
                if u["_k"] != w["_k"]:  # a piece: the whole word stays
                    drop.add(i if len(w["_k"]) < len(u["_k"]) else j)
                else:
                    drop.add(i if w["_score"] <= u["_score"] else j)
                break
    return [w for i, w in enumerate(words) if i not in drop]


def _fix_times(words: list[dict], ch: dict) -> None:
    for i, w in enumerate(words):
        if i and w["s"] < words[i - 1]["s"]:
            w["s"] = words[i - 1]["e"] if words[i - 1]["e"] <= w["e"] else words[i - 1]["s"]
        w["s"] = max(w["s"], ch["start"])
        w["e"] = min(max(w["e"], w["s"] + 0.02), ch["end"])


def _fix_case(words: list[dict]) -> None:
    for i, w in enumerate(words):
        prev = words[i - 1]["w"] if i else None
        start = prev is None or bool(re.search(r"[.!?…]$", prev))
        t = w["w"]
        if not t[:1].isalpha():
            continue
        if start:
            w["w"] = t[:1].upper() + t[1:]
        elif w["_first"] and (w["_lower"] or not w["_lat"]) and t[:1].isupper() and not t[1:2].isupper():
            # capital only because that model began a sentence here
            w["w"] = t[:1].lower() + t[1:]


def _split(words: list[dict], slots: list[_Slot], hyps: dict, ch: dict, lang: str,
           order: list[str], digits: bool = False) -> list[dict]:
    """Chunk words -> sentence-like segments with every model's text for the same span."""
    groups: list[list[dict]] = [[]]
    for w in words:
        g = groups[-1]
        if g:
            gap = w["s"] - g[-1]["e"]
            dur = g[-1]["e"] - g[0]["s"]
            end = re.search(r"[.!?…]$", g[-1]["w"]) and not re.fullmatch(r"\d+\.", g[-1]["w"])  # «am 15. März»
            if gap > 1.2 or (end and dur >= 2.0 and len(g) >= 3) or dur > 20:
                groups.append([])
        groups[-1].append(w)
    out = []
    for g in groups:
        if not g:
            continue
        a, b = g[0]["_slot"], g[-1].get("_slot_last", g[-1]["_slot"])
        # take the slots up to the next group's first slot so dropped words aren't lost in alts
        alts = []
        for n in order:
            if n not in hyps:
                continue
            ws = [slots[i].toks[n]["w"] for i in range(a, b + 1) if n in slots[i].toks]
            if ws:  # in digits like the text, or «25» and «двадцать пять» would look like a difference
                alts.append(dict(model=n, text=itn.text(" ".join(ws)) if digits else " ".join(ws)))
        text = " ".join(w["w"] for w in g)
        ps = [w.get("_agree", w["p"]) for w in g]
        seg_lang = "de" if T.latin_share(text) > 0.5 else ("ru" if T.CYRILLIC.search(text) else lang)
        flags = engine.suspicious(text, 0.0, 0.0, 0.0)
        # the models heard different things almost everywhere here: noise, crosstalk or a
        # mumble each of them filled with its own guess - the whole segment is doubtful
        if len(ps) >= 2 and float(np.median(ps)) < DISAGREE_P and "disagree" not in flags:
            flags.append("disagree")
        out.append(dict(
            start=g[0]["s"], end=g[-1]["e"], text=text,
            words=[dict(w=w["w"], s=w["s"], e=w["e"], p=w["p"], _agree=w.get("_agree", w["p"]),
                        **({"spoken": w["spoken"]} if "spoken" in w else {})) for w in g],
            metrics=dict(agreement=round(float(np.mean(ps)), 3), models=len(hyps), chunk_lang=lang),
            flags=flags, alts=alts, lang=seg_lang,
        ))
    return out


# ---------------------------------------------------------------- German re-check

def _candidates(seg: dict) -> list[int]:
    out = []
    for i, w in enumerate(seg["words"]):
        k = T.key(w["w"])
        if len(k) < 4 or not T.CYRILLIC.search(k) or T.number_value(w["w"]) is not None:
            continue
        if T.known_russian(w["w"]) is False:
            out.append(i)
    return out


def _de_check_chunks(st: dict, segs: list[dict]) -> set[int]:
    need = set()
    for s in segs:
        if _candidates(s):
            for ci, c in enumerate(st["chunks"]):
                if c["start"] - 0.01 <= s["start"] <= c["end"]:
                    need.add(ci)
                    break
    return need


def _glossary(opts) -> list[str]:
    terms = [t.strip() for t in re.split(r"[,\n;]+", opts.glossary or "") if t.strip()]
    if opts.dictionary:
        terms += settings.dictionary()["terms"]
    return [t for t in dict.fromkeys(terms) if T.LATIN.search(t)]


def _recheck(segs: list[dict], st: dict, opts) -> list[dict]:
    terms = _glossary(opts)
    de = []
    for h in (st["hyps"].get("de-check") or []):
        de += (h or {}).get("words") or []
    de_words = [w for w in de if T.key(w["w"]) and T.LATIN.search(w["w"])]
    fixes = 0
    for seg in segs:
        idx = _candidates(seg)
        if not idx:
            continue
        ws = seg["words"]
        for i in reversed(idx):
            w = ws[i]
            punct = T.tail_punct(w["w"])
            best, score, span = None, 0.0, 1
            # glossary terms: one word or two (for "штрассен бан" -> "Straßenbahn")
            for term in terms:
                for n in (1, 2):
                    if i + n > len(ws):
                        continue
                    heard = "".join(T.key(x["w"]) for x in ws[i:i + n])
                    sc = T.phon_sim(heard, T.key(term))
                    if sc > score and sc >= 0.8:
                        best, score, span = term, sc, n
            if best is None and de_words:
                near = [g for g in de_words if g["e"] >= w["s"] - 0.6 and g["s"] <= w["e"] + 0.6]
                for j, g in enumerate(near):
                    for cand in (g["w"], g["w"] + (near[j + 1]["w"] if j + 1 < len(near) else "")):
                        c = T.PUNCT_TAIL.sub("", cand)
                        sc = T.phon_sim(T.key(w["w"]), T.key(c))
                        if sc > score and sc >= 0.75:
                            best, score, span = c, sc, 1
            if best is None:
                continue
            ws[i] = dict(w=best + (punct if span == 1 else T.tail_punct(ws[i + span - 1]["w"])),
                         s=w["s"], e=ws[i + span - 1]["e"], p=max(w["p"], 0.5))
            del ws[i + 1:i + span]
            fixes += 1
        fixes += _glossary_pass(ws, terms)
        seg["text"] = " ".join(x["w"] for x in ws)
        if T.latin_share(seg["text"]) > 0.5:
            seg["lang"] = "de"
    for seg in segs:  # segments without Russian-unknown words: only the glossary pass
        if not _candidates(seg):
            n = _glossary_pass(seg["words"], terms)
            if n:
                fixes += n
                seg["text"] = " ".join(x["w"] for x in seg["words"])
    st["de_fixes"] = fixes
    return segs


GLOSS_PAIR = 0.9    # two words that together sound like a term ("штрассен Bahn")
GLOSS_WORD = 0.88   # a Latin misspelling of a term inside Russian speech ("Kesslerr")
GLOSS_PART = 8      # letters: a part of a multi-word term this long is matched on its own


def _glossary_pass(ws: list[dict], terms: list[str]) -> int:
    """Project terms heard as two words ("штрассен Bahn") or misspelled in Latin inside
    Russian speech. Pairs of real words and pairs across punctuation are left alone."""
    if not terms:
        return 0
    keys = {T.key(t): t for t in terms}
    # long parts of multi-word terms are terms too ("Gartenhauss" -> Gartenhaus of "Gartenhaus Sonnenberg");
    # short parts ("Frau" of "Frau Kessler") would catch ordinary words
    for t in terms:
        for part in re.split(r"[\s\-]+", t):
            if len(part) >= GLOSS_PART and T.LATIN.search(part):
                keys.setdefault(T.key(part), part)
    # in German speech a Latin word is a German word ("gleich" is not the term "Gleis")
    german = T.latin_share(" ".join(w["w"] for w in ws)) > 0.5
    fixes, i = 0, 0
    while i < len(ws):
        for n in (2, 1):
            part = ws[i:i + n]
            if len(part) < n or any(T.number_value(x["w"]) is not None for x in part):
                continue
            heard = "".join(T.key(x["w"]) for x in part)
            if len(heard) < 5 or (n == 1 and heard in keys) or any(T.key(x["w"]) in keys for x in part):
                continue
            latin = all(T.LATIN.search(x["w"]) for x in part)
            if n == 2 and heard not in keys:  # an exact split ("Straßen bahn") is always merged
                if T.tail_punct(part[0]["w"]):
                    continue  # "делать, а" is two phrases, not one word
                if latin and german:
                    continue  # "hat man" in German speech: two German words
                if all(T.CYRILLIC.search(x["w"]) and T.known_russian(x["w"]) for x in part):
                    continue  # two real Russian words ("а потом")
            if n == 1 and (not latin or german):
                continue  # Cyrillic words are the main pass's; in German speech a Latin word is German
            need = GLOSS_PAIR if n == 2 else GLOSS_WORD
            best, score = None, 0.0
            for k, term in keys.items():
                sc = T.phon_sim(heard, k)
                if sc >= need and sc > score:
                    best, score = term, sc
            if best is None:
                continue
            ws[i:i + n] = [dict(w=best + T.tail_punct(part[-1]["w"]), s=part[0]["s"], e=part[-1]["e"],
                                p=max(min(x["p"] for x in part), 0.5))]
            fixes += 1
            break
        i += 1
    return fixes
