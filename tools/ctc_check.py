"""Check by sound whether a German term fits the audio better than the merge's word (lab
experiment 17, the "nobody wrote it" places of docs/DECISIONS.md). For every short Cyrillic span of the
merge that sounds a bit like a dictionary term, the span's audio goes through GigaAM-v3 CTC
(CPU) and two texts are scored on it: the context with the merge's words and the same context
with the term spelled in Cyrillic the way a Russian speaker says it (a German spelling ->
Cyrillic rule, not the synthesizer's list). The term replaces the span when it explains the
audio better by a margin.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/ctc_check.py --doc <id> --set %WS_REAL_HOME%/tmp/dialog-set/dev   synthetic: terms right, false Latin
    ws-python tools/ctc_check.py --doc a1b2c3                                           gold: merge errors
    ws-python tools/ctc_check.py --library                                              real recordings: every change
    ... --margin 2,4,6        decision margins (log-probability) to try
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import os
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from whisper_studio import audio as A  # noqa: E402
from whisper_studio import ensemble, gold, settings, store, terms  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402

MODEL = "gigaam-v3-ctc"
WIN, HOP = 20.0, 16.0   # s: CTC windows over the recording, the middle of each is kept
FRAME = 0.04            # s per CTC frame (10 ms features, subsampling 4)
PAD = 1.0               # s of audio around a segment (free for the text, see Scorer.score)
FRAMES_PER_CHAR = 1.2   # less audio than this per letter: the segment's times are off
MIN_SIM = 0.5           # sound similarity for a term to be a candidate at all
SUPPORT = 5.0           # the span must explain the audio this much better than nothing
TAILS = ("", "а", "у", "е", "ом", "ов", "ам", "ы", "и", "ой")

_DE = [("tsch", "ч"), ("sch", "ш"), ("ch", "х"), ("ck", "к"), ("ß", "с"), ("äu", "ой"), ("eu", "ой"),
       ("ei", "ай"), ("ie", "и"), ("tz", "ц"), ("ph", "ф"), ("qu", "кв"), ("ä", "е"), ("ö", "ё"), ("ü", "ю"),
       ("z", "ц"), ("w", "в"), ("v", "ф"), ("y", "и"), ("x", "кс"), ("j", "й"), ("c", "к")]
_LAT = dict(zip("abdefghiklmnoprstu", "абдефгхиклмнопрсту"))


def cyrillic(term: str) -> str:
    """How a Russian speaker says a German spelling: Stadt -> штадт, Wohnung -> вонунг."""
    out = []
    for w in re.split(r"[\s\-]+", term.lower()):
        w = re.sub(r"^s([pt])", r"ш\1", w)
        w = re.sub(r"(?<=[aeiouäöü])h(?=[^aeiouäöü]|$)", "", w)
        w = re.sub(r"s(?=[aeiouäöü])", "з", w)
        for a, b in _DE:
            w = w.replace(a, b)
        w = "".join(_LAT.get(c, c) for c in w)
        out.append(w)
    return " ".join(out)


class Scorer:
    """CTC log-probabilities of a recording (GigaAM-v3, CPU) and the score of a text on a time span."""

    def __init__(self, x: np.ndarray):
        import onnx_asr
        import torch

        self.torch = torch
        asr = onnx_asr.load_model(MODEL, providers=["CPUExecutionProvider"]).asr
        self.blank = asr._blank_idx
        self.tok = {c: i for i, c in asr._vocab.items()}
        frames = []
        n = int(WIN * 16000)
        hop = int(HOP * 16000)
        keep0 = int((WIN - HOP) / 2 / FRAME)
        for k, i in enumerate(range(0, max(1, len(x) - (n - hop)), hop)):
            w = x[i:i + n].astype(np.float32)
            lp, ln = asr._encode(*asr._preprocessor(w[None], np.array([len(w)])))
            lp = lp[0, : ln[0]]
            a = 0 if k == 0 else keep0  # window k keeps [k*HOP + margin, (k+1)*HOP + margin)
            b = len(lp) if i + n >= len(x) else keep0 + int(HOP / FRAME)
            frames.append(lp[a:b])
        self.lp = torch.from_numpy(np.concatenate(frames)).log_softmax(-1)
        self.lp_np = self.lp.numpy().astype(np.float64)

    def score(self, text: str, s: float, e: float) -> float:
        """log P(text | audio) on the best stretch of [s, e]: frames before and after the text are
        free, so a window wider than the speech costs nothing and every candidate gets the same
        audio (CTC forward pass, max over start and end frames)."""
        ids = [self.tok[c] for c in text.lower().replace("ё", "е") if c in self.tok]
        a, b = max(0, int(s / FRAME)), min(len(self.lp), int(e / FRAME) + 1)
        if not ids or b - a < len(ids):
            return -1e9
        lp = self.lp_np[a:b]
        ext = np.full(2 * len(ids) + 1, self.blank)
        ext[1::2] = ids
        skip = np.zeros(len(ext), bool)
        skip[3::2] = ext[3::2] != ext[1:-2:2]
        neg = -1e30
        alpha = np.full(len(ext), neg)
        best = neg
        for t in range(len(lp)):
            if t:
                a2 = np.concatenate(([neg], alpha[:-1]))
                a3 = np.where(skip, np.concatenate(([neg, neg], alpha[:-2])), neg)
                alpha = np.logaddexp(np.logaddexp(alpha, a2), a3)
            alpha[0] = np.logaddexp(alpha[0], 0.0)  # the text may start at any frame
            alpha[1] = np.logaddexp(alpha[1], 0.0)
            alpha = alpha + lp[t, ext]
            best = max(best, float(np.logaddexp(alpha[-1], alpha[-2])))  # and end at any frame
        return best


def clean(w: str) -> str:
    return re.sub(r"[^\wё\- ]", "", w.lower()).replace("-", " ")


def candidates(words: list[dict], term_list: list[str]):
    """(i, j, term, sim): a span words[i:j] of 1-2 Cyrillic words that sounds like the term."""
    keys = [(t, T.key(t)) for t in term_list if len(T.key(t)) >= terms.MIN_TERM]
    for i in range(len(words)):
        for j in (i + 1, i + 2):
            span = words[i:j]
            if j > len(words) or any(T.is_latin(w["w"]) or not T.CYRILLIC.search(w["w"]) for w in span):
                continue
            if any(T.key(w["w"]).isdigit() for w in span):
                continue
            heard = [T.key(w["w"]) for w in span]
            best = []
            for t, tk in keys:
                s, _, _ = terms.best_sim(heard, tk)
                if s >= MIN_SIM:
                    best.append((s, t))
            for s, t in sorted(best, reverse=True)[:3]:
                yield i, j, t, s


def runs(words: list[dict]) -> list[tuple[int, int, float, float]]:
    """Stretches of a segment without numbers (GigaAM's alphabet has no digits, and a number's
    words are gone from the merge): (first, end, window start, window end). The window reaches
    over the neighbouring numbers; the free edges of Scorer.score absorb them."""
    out, i = [], 0
    while i < len(words):
        if re.search(r"\d", words[i]["w"]):
            i += 1
            continue
        j = i
        while j < len(words) and not re.search(r"\d", words[j]["w"]):
            j += 1
        t0 = (words[i - 1]["s"] if i else words[i]["s"]) - PAD
        t1 = (words[j]["e"] if j < len(words) else words[j - 1]["e"]) + PAD
        out.append((i, j, t0, t1))
        i = j
    return out


def spell(w: str) -> str:
    return cyrillic(w) if T.is_latin(w) else clean(w)


def check(segs: list[dict], sc: Scorer, term_list: list[str]) -> list[dict]:
    """Every candidate with its score gain, scored on its stretch of the segment (word times inside
    a segment can be rough; the stretch as a whole, with free edges, covers its audio). Overlapping
    candidates are all kept (decided later). i, j index the flat word list of all segments."""
    out = []
    base_i = 0
    for seg in segs:
        words = seg["words"]
        for r0, r1, t0, t1 in runs(words):
            run = words[r0:r1]
            texts = [spell(w["w"]) for w in run]
            orig = sc.score(" ".join(texts), t0, t1)
            if orig < -1e8 or (t1 - t0) / FRAME < FRAMES_PER_CHAR * len(" ".join(texts)):
                continue  # too little audio for the text: the times are off, no verdict
            for i, j, t, s in candidates(run, term_list):
                # the span without its words: if that explains the audio as well, the text and the
                # audio do not match here and no candidate can be judged
                dele = sc.score(" ".join(texts[:i] + texts[j:]), t0, t1)
                best, tail = -1e9, ""
                for tl in TAILS:
                    v = sc.score(" ".join(texts[:i] + [cyrillic(t) + tl] + texts[j:]), t0, t1)
                    if v > best:
                        best, tail = v, tl
                out.append(dict(i=base_i + r0 + i, j=base_i + r0 + j, term=t, sim=round(s, 3),
                                gain=round(best - orig, 2), support=round(orig - dele, 2), tail=tail,
                                span=" ".join(w["w"] for w in run[i:j]), p=min(w.get("p", 1.0) for w in run[i:j])))
        base_i += len(words)
    return out


def apply(words: list[dict], found: list[dict], margin: float, max_p: float,
          support: float = SUPPORT) -> tuple[list[dict], list[dict]]:
    """Replace spans whose best term wins by the margin where the audio supports the span at all
    (the strongest first, no overlaps)."""
    take, used = [], set()
    for c in sorted(found, key=lambda c: -c["gain"]):
        if (c["gain"] < margin or c["p"] > max_p or c.get("support", support) < support
                or used & set(range(c["i"], c["j"]))):
            continue
        take.append(c)
        used |= set(range(c["i"], c["j"]))
    out, k = [], 0
    starts = {c["i"]: c for c in take}
    while k < len(words):
        c = starts.get(k)
        if not c:
            out.append(words[k])
            k += 1
            continue
        span = words[c["i"]:c["j"]]
        tail = c["tail"] if c["tail"] and c["term"][-1].lower() not in "aeiouyäöü" else ""
        punct = re.search(r"[^\w]*$", span[-1]["w"]).group(0)
        out.append(dict(span[0], w=c["term"] + tail + punct, e=span[-1]["e"], term=True, p=0.5))
        k = c["j"]
    return out, take


def merged_segments(d: gold.Doc) -> list[dict]:
    opts = gold.options(d.doc)
    segs = ensemble.combine(json.loads(json.dumps(d.st)), opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    return segs


def audio_of(doc_id: str) -> np.ndarray:
    folder = LIBRARY_DIR / doc_id
    x = A.load(next(p for p in folder.iterdir() if p.stem == "audio"))
    return x[0] if isinstance(x, tuple) else x


def run_doc(doc_id: str, term_list: list[str], cache: Path) -> tuple[gold.Doc, list[dict], list[dict]]:
    d = gold.Doc(doc_id)
    segs = merged_segments(d)
    words = [w for s in segs for w in s["words"]]
    f = cache / f"{d.id[-6:]}.json"
    if f.exists():
        found = json.loads(f.read_text("utf-8"))
    else:
        sc = Scorer(audio_of(d.id))
        found = check(segs, sc, term_list)
        f.write_text(json.dumps(found, ensure_ascii=False), "utf-8")
    return d, words, found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc")
    ap.add_argument("--set", help="synthetic set folder (manifest.json) for --doc")
    ap.add_argument("--library", action="store_true")
    ap.add_argument("--margin", default="2,4,6,8")
    ap.add_argument("--max-p", type=float, default=1.01, help="only words with a doubt below this")
    ap.add_argument("--support", type=float, default=SUPPORT)
    a = ap.parse_args()
    term_list = [t for t in settings.dictionary()["terms"] if any("a" <= c.lower() <= "z" for c in t)]
    cache = REAL_HOME / "tmp/ctc-lab/cache"
    cache.mkdir(parents=True, exist_ok=True)
    margins = [float(m) for m in a.margin.split(",")]
    docs = []
    if a.library:
        docs = [p.name for p in sorted(LIBRARY_DIR.iterdir()) if (p / "hypotheses.json").exists()]
    elif a.doc:
        docs = [a.doc]
    for doc_id in docs:
        try:
            d, words, found = run_doc(doc_id, term_list, cache)
        except gold.NotFound:
            continue
        print(f"== {d.id[-6:]}: слов {len(words)}, кандидатов {len(found)}", flush=True)
        if a.set:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import dialog_eval as DE
            man = json.loads((Path(a.set) / "manifest.json").read_text("utf-8"))
            refs = DE.ref_words(man)
            r0 = DE.score(words, refs, man)
            t0 = r0["cls_n"]["термин"] - r0["cls_e"]["термин"]
            print(f"   как есть: WER {r0['wer']:.2%}, терминов верно {t0}/{r0['cls_n']['термин']}, "
                  f"лишней латиницы {sum(r0['false_latin'].values())}")
            for m in margins:
                new, take = apply(words, found, m, a.max_p, a.support)
                r = DE.score(new, refs, man)
                tr = r["cls_n"]["термин"] - r["cls_e"]["термин"]
                print(f"   порог {m:4g}: замен {len(take):3d}, WER {r['wer']:.2%}, терминов верно {tr}, "
                      f"лишней латиницы {sum(r['false_latin'].values())}")
        elif d.doc.get("gold"):
            d.set_reference()
            base = sum(gold.errors(d.ref[ci], gold.norm([w["w"] for w in ws]))
                       for ci, ws in d.by_chunk(words).items() if ci in d.ref)
            print(f"   как есть: ошибок {base}")
            for m in margins:
                new, take = apply(words, found, m, a.max_p, a.support)
                e = sum(gold.errors(d.ref[ci], gold.norm([w["w"] for w in ws]))
                        for ci, ws in d.by_chunk(new).items() if ci in d.ref)
                print(f"   порог {m:4g}: замен {len(take):3d}, ошибок {e}  "
                      + "; ".join(f"{c['span']}->{c['term']}" for c in take[:8]))
        else:
            for m in margins:
                _, take = apply(words, found, m, a.max_p, a.support)
                print(f"   порог {m:4g}: замен {len(take)}  " + "; ".join(f"{c['span']}->{c['term']}({c['gain']})" for c in take[:12]))


if __name__ == "__main__":
    main()
