"""One "gold" recording for the owner to check by ear (docs/GOLD.md): windows of 4-5 minutes
cut from several real recordings, glued into one audio file with silence between them,
queued for transcription with the app defaults (ensemble "max") and marked as a gold document.

    ws-python tools/make_gold.py --dry-run            # which windows would be taken, and why
    ws-python tools/make_gold.py                      # build the audio and queue it
    ... --total 30 --window 4-5 --gap 2.5 --seed 3    # target minutes, window minutes, pause s
    ... --exclude 12-00-00 --exclude 5f4e3d           # skip a source (file name part or doc id)
    ... --no-queue --out DIR                          # build audio + manifest only, don't queue

(ws-python = the venv's python with PYTHONPATH=<repo>.)

Sources: every finished transcript in the library, the newest per source file (an ensemble
one when there is any); gold documents are skipped. A window starts
and ends in a pause of >= 0.8 s between speech regions (the VAD regions of the ensemble
chunks, together with the word times), so no word is cut. Windows are scored by what makes a
reference useful: the models disagree there (low word agreement), German, numbers, names and
glossary terms, dense speech, both speakers; mostly-silent windows and windows that are mostly
hallucinations or "disagree" segments are skipped, windows with edited text are avoided (the
old transcript is compared with the reference later). Then a diverse set is picked greedily,
at most one window per source: each new label (German, trips, numbers, noise, Russian...) is
a bonus, a repeated language a small penalty.

Output: TMP_DIR/gold/gold-<date>.flac (16 kHz mono, lossless: the samples the app decodes
from the originals, each piece times one constant gain so their speech plays equally loud;
--no-level keeps them bit-exact) and .json next to it with the manifest [{doc, source, start, end, at, label}] (start/end in the source, at = offset in the
glued audio). The pause between pieces (2.5 s) is longer than ensemble.MAX_GAP, so no chunk
spans two pieces. The new document gets doc["gold"] = {"created", "manifest"}; it is written
before the job is queued, so the worker (which loads the document when it takes the job)
can't overwrite it.
"""
from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import settings, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import LIBRARY_DIR, SAMPLE_RATE, TMP_DIR  # noqa: E402

OUT_DIR = TMP_DIR / "gold"
MIN_PAUSE = 0.8          # s: a window may start/end only in a pause at least this long
PAD = 0.5                # s of the pause kept before/after the speech of a piece
MIN_DENSITY = 0.4        # speech share of a window below which it is "mostly silence"
MAX_BAD = 0.5            # share of segments flagged (hallucination or disagree) that rules a window out
TARGET_DB = -24.0        # dBFS RMS of the speech of every piece (a constant gain per piece)
MAX_GAIN_DB = 20.0
PEAK = 0.95
DISAGREE_P = 0.3        # = ensemble.DISAGREE_P: older transcripts have no "disagree" flag, the
                         # median word agreement below it says the same (not imported: ensemble loads engine)

# topic words (keys, prefix match): enough of them per minute gives the label
TRIPS = ("вокзал", "билет", "поезд", "автобус", "bahnhof", "gleis", "fahrkarte", "поездк", "дорог", "ехать", "едем")
FAMILY = ("мама", "мамы", "маме", "мамой", "мамочк", "папа", "папы", "папе", "сын", "дочь", "доч", "бабушк",
          "бабул", "дедушк", "брат", "сестр", "ребен", "ребён", "дети", "детей", "детям", "муж", "жена",
          "жены", "жене", "внук", "внуч", "тёт", "тет", "дяд", "родител")


def mmss(t: float) -> str:
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


# ---------------------------------------------------------------- library

def _newest_per_source(exclude: list[str]) -> list[dict]:
    """Finished transcripts, the newest per source file (an ensemble one if any)."""
    best: dict[str, dict] = {}
    for f in sorted(LIBRARY_DIR.glob("*/doc.json")):
        try:
            d = json.loads(f.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if d.get("gold") or d.get("status") != "done" or not d.get("segments"):
            continue
        src = str(d.get("source_path") or "")
        if any(x and (x.lower() in Path(src).name.lower() or d["id"].endswith(x.lower())) for x in exclude):
            continue
        k = src.lower()
        rank = ((d.get("options") or {}).get("quality") == "max", d.get("created") or "")
        if k not in best or rank > best[k][0]:
            best[k] = (rank, d)
    return [d for _, d in best.values()]


def audio_path(d: dict) -> Path | None:
    """The original if it is still there, else the copy kept in the library (same audio)."""
    src = Path(d.get("source_path") or "")
    if src.is_file():
        return src
    if d.get("audio_file") and (store.path(d["id"]) / d["audio_file"]).is_file():
        return store.path(d["id"]) / d["audio_file"]
    return None


def _union(spans: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _terms() -> set[str]:
    d = settings.dictionary()
    words = [*d["terms"], *(r["to"] for r in d["replace"])]
    return {T.key(w) for t in words for w in re.split(r"[\s,/-]+", t) if len(T.key(w)) >= 4}


class Source:
    """One recording: speech regions, words and segments as prefix sums for fast window stats."""

    def __init__(self, d: dict, terms: set[str]):
        self.doc, self.id = d, d["id"]
        self.audio = audio_path(d)
        self.name = Path(d.get("source_path") or "").name
        self.dur = float(d.get("duration") or 0.0)
        segs = sorted(d["segments"], key=lambda s: (s["start"] + s["end"]) / 2)  # bisect by midpoint
        self.segs = segs
        hyp = store.path(self.id) / "hypotheses.json"
        chunks = []
        if hyp.is_file():
            try:
                chunks = json.loads(hyp.read_text("utf-8")).get("chunks") or []
            except (OSError, json.JSONDecodeError):
                chunks = []
        spans = [(float(a), float(b)) for ch in chunks for a, b in ch.get("speech") or []]
        words = [w for s in segs for w in s.get("words") or [] if w.get("s") is not None and w.get("e") is not None]
        # word times too: a word the VAD missed must not be cut either
        spans += [(w["s"], w["e"]) for w in words]
        spans += [(s["start"], s["end"]) for s in segs if not s.get("words")]
        self.regions = _union(spans)

        # German by chunk language (speech seconds of "de" chunks, half for mixed ones)
        de_spans = []
        for ch in chunks:
            langs = ch.get("langs") or [ch.get("lang")]
            wgt = 1.0 if ch.get("lang") == "de" else 0.5 if "de" in langs else 0.0
            if wgt:
                de_spans += [(a, b, wgt) for a, b in ch.get("speech") or []]
        self.de_spans = de_spans

        # words: time, and per-word features as cumulative sums
        rows = []
        for w in words:
            k = T.key(w["w"])
            rows.append(((w["s"] + w["e"]) / 2, (
                1.0,
                float(w.get("p") if w.get("p") is not None else 1.0),
                float((w.get("p") if w.get("p") is not None else 1.0) < store.REVIEW_P),
                float(T.is_latin(w["w"])),
                float(k in terms),
                float(k.startswith(TRIPS)),
                float(k.startswith(FAMILY)),
                float(bool(w.get("edited"))),
            )))
        rows.sort(key=lambda r: r[0])
        self.wt = [t for t, _ in rows]
        feats = [f for _, f in rows]
        self.wc = np.vstack([np.zeros(8), np.cumsum(np.array(feats), axis=0)]) \
            if feats else np.zeros((1, 8))
        # numbers: digits or number words worth checking (not «один», «два» in passing)
        nt = []
        for s in segs:
            ws = [w for w in s.get("words") or [] if w.get("s") is not None and w.get("e") is not None]
            for t in T.merge_numbers([dict(w=w["w"], s=w["s"], e=w["e"]) for w in ws]):
                if t.get("num") and (re.search(r"\d", t["w"]) or len(t.get("parts") or ()) > 1
                                     or float(t["k"][1:] or 0) >= 10):
                    nt.append(t["s"])
        self.nt = sorted(nt)

        # segments by midpoint: flags, edits, speaker seconds
        self.st = [(s["start"] + s["end"]) / 2 for s in segs]
        spk = sorted({s.get("speaker") or "?" for s in segs})
        rows = []
        for s in segs:
            dur = max(0.0, s["end"] - s["start"])
            ps = [w["p"] for w in s.get("words") or [] if w.get("p") is not None]
            bad = (store.hallucination(s) or "disagree" in (s.get("flags") or [])
                   or (len(ps) >= 2 and float(np.median(ps)) < DISAGREE_P))
            rows.append([1.0, float(bad),
                         float(bool(s.get("ai"))), float((s.get("metrics") or {}).get("agreement", 1.0))]
                        + [dur if (s.get("speaker") or "?") == x else 0.0 for x in spk])
        self.sc = np.vstack([np.zeros(4 + len(spk)), np.cumsum(np.array(rows), axis=0)]) if rows \
            else np.zeros((1, 4 + len(spk)))

    # -------------------------------------------------------------- windows

    def cuts(self) -> list[tuple[float, float]]:
        """Possible boundaries: (end of a piece here, start of a piece here) in each long pause."""
        r = self.regions
        if not r:
            return []
        out = [(0.0, max(0.0, r[0][0] - PAD))]  # file start
        for (a0, a1), (b0, b1) in zip(r, r[1:]):
            gap = b0 - a1
            if gap >= MIN_PAUSE:
                pad = min(PAD, gap / 2)
                out.append((a1 + pad, b0 - pad))
        out.append((min(self.dur or r[-1][1] + PAD, r[-1][1] + PAD), self.dur or r[-1][1] + PAD))  # file end
        return out

    def speech(self, a: float, b: float) -> float:
        return sum(max(0.0, min(y, b) - max(x, a)) for x, y in self.regions if y > a and x < b)

    def windows(self, lo: float, hi: float) -> list[dict]:
        cs = self.cuts()
        out = []
        for i in range(len(cs) - 1):
            start = cs[i][1]
            for j in range(i + 1, len(cs)):
                end = cs[j][0]
                if end - start > hi:
                    break
                if end - start >= lo:
                    out.append(self.stats(start, end))
        return out

    def stats(self, a: float, b: float) -> dict:
        i, j = bisect.bisect_left(self.wt, a), bisect.bisect_right(self.wt, b)
        n, ps, low, lat, term, topic, fam, edited = self.wc[j] - self.wc[i]
        si, sj = bisect.bisect_left(self.st, a), bisect.bisect_right(self.st, b)
        sv = self.sc[sj] - self.sc[si]
        nseg, bad, ai, agr = sv[:4]
        spk = sorted(sv[4:], reverse=True)
        minutes = (b - a) / 60
        speech = self.speech(a, b)
        de = sum(max(0.0, min(y, b) - max(x, a)) * w for x, y, w in self.de_spans if y > a and x < b)
        nums = bisect.bisect_right(self.nt, b) - bisect.bisect_left(self.nt, a)
        return dict(
            src=self, start=round(a, 3), end=round(b, 3), len=b - a,
            words=int(n), segs=int(nseg),
            p=ps / n if n else 1.0, low=low / n if n else 0.0,
            lat=lat / n if n else 0.0, de=de / speech if speech else 0.0,
            num_pm=nums / minutes, term_pm=term / minutes, trip_pm=topic / minutes, fam_pm=fam / minutes,
            density=speech / (b - a), bad=bad / nseg if nseg else 0.0,
            agree=agr / nseg if nseg else 1.0,
            spk2=(spk[1] / (spk[0] + spk[1])) if len(spk) > 1 and spk[0] + spk[1] else 0.0,
            ai=int(ai), edited=int(edited),
        )


# ---------------------------------------------------------------- scoring

def tags(w: dict) -> list[str]:
    """Short label words, the language first."""
    if w["lat"] >= 0.4 or w["de"] >= 0.5:
        lang = "немецкий"
    elif w["lat"] >= 0.1 or w["de"] >= 0.15:
        lang = "смешанный"
    else:
        lang = "русский"
    extra = []
    if w["trip_pm"] >= 1.0:
        extra.append(("поездки", w["trip_pm"] / 1.0))
    if w["num_pm"] >= 1.5:
        extra.append(("числа", w["num_pm"] / 1.5))
    if w["low"] >= 0.2:
        extra.append(("шум", w["low"] / 0.2))
    if w["fam_pm"] >= 0.8 and lang != "немецкий":
        extra.append(("семья", w["fam_pm"] / 0.8))
    if w["term_pm"] >= 1.5:
        extra.append(("имена", w["term_pm"] / 1.5))
    return [lang] + [t for t, _ in sorted(extra, key=lambda x: -x[1])[:2]]


def usable(w: dict) -> bool:
    return w["density"] >= MIN_DENSITY and w["bad"] <= MAX_BAD and w["words"] >= 60


def score(w: dict) -> float:
    """How much a checked text of this window would tell (before diversity)."""
    # disagreement counts up to a point: beyond it the window is mostly noise nobody can check
    return (1.5 * min(w["low"], 0.35) + 1.0 * min(1 - w["p"], 0.45)
            + 0.6 * min(max(w["lat"], w["de"]) / 0.3, 1.0)
            + 0.5 * min(w["num_pm"] / 2.0, 1.0) + 0.4 * min(w["term_pm"] / 2.0, 1.0)
            + 0.3 * min(w["spk2"] / 0.25, 1.0) + 0.4 * min(w["density"] / 0.7, 1.0)
            - 1.0 * max(0.0, w["bad"] - 0.15)
            - 0.6 * (w["ai"] > 0) - 0.2 * (w["edited"] > 0))


def choose(sources: list[Source], lo: float, hi: float, total: float, per_source: int,
           rng: random.Random) -> tuple[list[dict], list[str]]:
    notes = []
    cands = []
    for s in sources:
        if s.audio is None:
            notes.append(f"{s.name}: нет аудио, пропущен")
            continue
        ws = [w for w in s.windows(lo, hi) if usable(w)]
        if not ws:
            why = "короче окна" if s.dur < lo else "нет окна с плотной речью без пауз-разрезов"
            notes.append(f"{s.name} ({s.id[-6:]}, {mmss(s.dur)}): {why}")
            continue
        for w in ws:
            w["tags"] = tags(w)
            w["base"] = score(w) + rng.uniform(0.0, 0.1)
        cands += ws
    picked: list[dict] = []
    covered: dict[str, int] = {}
    used: dict[str, int] = {}
    while cands and sum(w["len"] for w in picked) < total:
        def gain(w):
            g = w["base"]
            for t in w["tags"]:
                g += 0.6 if t not in covered else -0.15 * covered[t]
            return g

        best = max(cands, key=gain)
        best["gain"] = gain(best)
        picked.append(best)
        for t in best["tags"]:
            covered[t] = covered.get(t, 0) + 1
        sid = best["src"].id
        used[sid] = used.get(sid, 0) + 1
        cands = [w for w in cands if not (w["src"].id == sid and (
            used[sid] >= per_source or (w["start"] < best["end"] and best["start"] < w["end"])))]
    picked.sort(key=lambda w: (w["src"].doc.get("created") or "", w["start"]))
    return picked, notes


# ---------------------------------------------------------------- audio

def _gain_db(piece: np.ndarray, src: Source, t0: float) -> float:
    """One constant gain per piece: its speech to TARGET_DB, so the owner doesn't ride the volume
    between a loud and a quiet call. Only as far as the peak allows: never clips, never compresses
    (a constant gain changes nothing for the models; the app levels within chunks itself)."""
    t1 = t0 + len(piece) / SAMPLE_RATE
    idx = [np.arange(int((max(a, t0) - t0) * SAMPLE_RATE), int((min(b, t1) - t0) * SAMPLE_RATE))
           for a, b in src.regions if b > t0 and a < t1]
    sp = piece[np.concatenate(idx)] if idx else piece
    if not len(sp):
        return 0.0
    rms = float(np.sqrt(np.mean(sp.astype(np.float64) ** 2))) or 1e-9
    peak = float(np.abs(piece).max()) or 1e-9
    g = min(TARGET_DB - 20 * np.log10(rms), 20 * np.log10(PEAK / peak))
    return float(np.clip(g, -12.0, MAX_GAIN_DB))


def build(picked: list[dict], gap: float, level: bool = True) -> tuple[np.ndarray, list[dict], list[float]]:
    from whisper_studio import audio as A

    cache: dict[Path, np.ndarray] = {}
    parts, manifest, gains, at = [], [], [], 0.0
    silence = np.zeros(int(round(gap * SAMPLE_RATE)), np.float32)
    for k, w in enumerate(picked):
        src: Source = w["src"]
        if src.audio not in cache:
            cache.clear()  # one source at a time: a 15-minute recording is ~60 MB as float32
            cache[src.audio] = A.load(src.audio)
        x = cache[src.audio]
        a, b = int(round(w["start"] * SAMPLE_RATE)), int(round(w["end"] * SAMPLE_RATE))
        piece = x[a:min(b, len(x))].astype(np.float32)
        g = _gain_db(piece, src, a / SAMPLE_RATE) if level else 0.0
        if abs(g) >= 0.5:
            piece *= 10 ** (g / 20)
        gains.append(round(g, 1) if abs(g) >= 0.5 else 0.0)
        if k:
            parts.append(silence)
            at += len(silence) / SAMPLE_RATE
        manifest.append({"doc": src.id, "source": src.name, "start": round(a / SAMPLE_RATE, 3),
                         "end": round((a + len(piece)) / SAMPLE_RATE, 3), "at": round(at, 3),
                         "label": ", ".join(w["tags"])})
        parts.append(piece)
        at += len(piece) / SAMPLE_RATE
    return (np.concatenate(parts) if parts else np.zeros(0, np.float32)), manifest, gains


def save_audio(x: np.ndarray, dst: Path) -> None:
    """16 kHz mono: FLAC/WAV keep the decoded samples exactly; OGG is Opus (smaller, lossy)."""
    import av

    ext = dst.suffix.lower()
    fmt, codec, rate = {".flac": ("flac", "flac", SAMPLE_RATE), ".wav": ("wav", "pcm_s16le", SAMPLE_RATE),
                        ".ogg": ("ogg", "libopus", 48000)}[ext]
    pcm = (np.clip(x, -1.0, 32767 / 32768) * 32768).astype(np.int16)
    tmp = dst.with_name(dst.stem + ".part" + ext)
    with av.open(str(tmp), "w", format=fmt) as out:
        st = out.add_stream(codec, rate=rate, layout="mono")
        if ext == ".ogg":
            st.bit_rate = 64000
        step = SAMPLE_RATE  # 1 s frames
        for i in range(0, len(pcm), step):
            fr = av.AudioFrame.from_ndarray(pcm[i:i + step].reshape(1, -1), format="s16", layout="mono")
            fr.sample_rate = SAMPLE_RATE
            for p in st.encode(fr):
                out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    tmp.replace(dst)


def check(picked: list[dict]) -> None:
    """No word of the source transcript may straddle a cut."""
    for w in picked:
        for s in w["src"].segs:
            for x in s.get("words") or []:
                for t in (w["start"], w["end"]):
                    if x.get("s") is not None and x["s"] < t < x["e"]:
                        raise SystemExit(f"разрез {mmss(t)} в {w['src'].name} режет слово «{x['w']}»")


# ---------------------------------------------------------------- queue

def queue(dst: Path, manifest: list[dict], created: str, title: str) -> dict:
    from whisper_studio import jobs, options

    # language/model/quality from the app defaults (ru, mixed, max); speakers differ per piece
    opts = options.make_options({"num_speakers": "auto"})
    doc = options.submit(dst, opts, title=title)
    # the job is not queued yet, so no worker holds a copy of the document that could overwrite this
    with store.editing(doc["id"], busy_ok=True) as d:
        d["gold"] = {"created": created, "manifest": manifest}
        store.save(d, snapshot=False)
    jobs.submit_process(doc["id"])
    return d


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="only show the chosen windows")
    ap.add_argument("--no-queue", action="store_true", help="build the audio and manifest, don't queue")
    ap.add_argument("--total", type=float, default=30.0, help="target minutes of audio (default 30)")
    ap.add_argument("--window", default="4-5", help="window length in minutes, min-max (default 4-5)")
    ap.add_argument("--gap", type=float, default=2.5, help="seconds of silence between pieces (default 2.5)")
    ap.add_argument("--max-per-source", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="another seed = another set among near-equal windows")
    ap.add_argument("--exclude", action="append", default=[], help="source file name part or doc id")
    ap.add_argument("--format", choices=("flac", "ogg", "wav"), default="flac")
    ap.add_argument("--no-level", action="store_true", help="no per-piece gain: samples as in the originals")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help=f"folder for the audio (default {OUT_DIR})")
    a = ap.parse_args(argv)

    lo, hi = (float(v) * 60 for v in a.window.split("-"))
    if a.gap <= 2.0:
        print("внимание: пауза <= 2 с (ensemble.MAX_GAP) — куски могут попасть в один чанк")
    terms = _terms()
    sources = [Source(d, terms) for d in _newest_per_source(a.exclude)]
    picked, notes = choose(sources, lo, hi, a.total * 60, a.max_per_source, random.Random(a.seed))
    check(picked)

    print(f"Источников: {len(sources)}, выбрано окон: {len(picked)}")
    for w in picked:
        s = w["src"]
        print(f"  {s.id[-6:]}  {s.name:<34} {mmss(w['start']):>6}–{mmss(w['end']):<6} ({mmss(w['len'])})  "
              f"«{', '.join(w['tags'])}»")
        print(f"          слов {w['words']}, p {w['p']:.2f}, сомнит. {w['low']:.0%}, лат. {w['lat']:.0%}, "
              f"de-чанки {w['de']:.0%}, чисел/мин {w['num_pm']:.1f}, термины/мин {w['term_pm']:.1f}, "
              f"поездки/мин {w['trip_pm']:.1f}, речь {w['density']:.0%}, 2-й голос {w['spk2']:.0%}, "
              f"⚠/неразб. {w['bad']:.0%}" + (f", ИИ-правки {w['ai']}" if w["ai"] else "")
              + (f", правлено слов {w['edited']}" if w["edited"] else "") + f"; балл {w['gain']:.2f}")
    speech = sum(w["len"] for w in picked)
    full = speech + a.gap * max(0, len(picked) - 1)
    print(f"Итого: окна {mmss(speech)}, с паузами {mmss(full)} (цель {mmss(a.total * 60)})")
    if speech < a.total * 60 - lo / 2:
        print(f"  меньше цели: не хватает записей длиннее {mmss(lo)} (по одному окну на запись)")
    for n in notes:
        print(f"  пропущено: {n}")
    if a.dry_run or not picked:
        return

    x, manifest, gains = build(picked, a.gap, level=not a.no_level)
    print("Усиление кусков, дБ: " + ", ".join(f"{g:+.1f}" for g in gains))
    created = datetime.now().isoformat(timespec="seconds")
    a.out.mkdir(parents=True, exist_ok=True)
    stem = f"gold-{datetime.now():%Y-%m-%d}"
    k = 1
    while any((a.out / f"{stem}{'' if k == 1 else f'-{k}'}{e}").exists() for e in (f".{a.format}", ".json")):
        k += 1
    dst = a.out / f"{stem}{'' if k == 1 else f'-{k}'}.{a.format}"
    save_audio(x, dst)
    dst.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), "utf-8")
    print(f"Аудио: {dst} ({len(x) / SAMPLE_RATE:.1f} с, {dst.stat().st_size / 1e6:.1f} МБ), манифест рядом (.json)")
    if a.no_queue:
        return
    d = queue(dst, manifest, created, f"Эталон {datetime.now():%Y-%m-%d}")
    print(f"В очереди: {d['id']} «{d['title']}» — после распознавания выверить по docs/GOLD.md")


if __name__ == "__main__":
    main()
