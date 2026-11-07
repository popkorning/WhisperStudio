"""Gold reference: how good the ensemble is against text the owner checked by ear (docs/GOLD.md).

Works on stored hypotheses (hypotheses.json in the document folder), so nothing is recognized
again and no GPU is needed. Heavy (numpy, the ensemble): imported only by tools/asr_eval.py and
by the evaluation process the app starts (`python -m whisper_studio gold-eval`, jobs.start_gold_eval).

The reference is the document's current text, but only where the owner listened:
- a gold document (doc["gold"]): a chunk counts only if every reference word inside its own
  time range comes from a segment marked verified, no unverified segment overlaps that range
  and somebody listened there at all (a verified segment, or a segment deleted as "nothing was
  said" - store keeps those spans in doc["gold"]["silent"]);
- any other document (tools/asr_eval.py --ref): the whole document, or the given span, as checked;
- chunks with "???" (not audible) are skipped either way.

Times of the reference words come from the transcript (edited words get interpolated times),
so words are matched to chunks by time; a word on a chunk boundary can land in the neighbour
chunk, which is noise at the level of a few words.

What it measures (the result dict, see evaluate()): WER of every member on the chunks it votes
on, the merge on the same chunks, oracle; the merge with each member left out; errors by kind
(numbers, German/Latin words, names, glossary terms); doubt marks (word p < store.REVIEW_P, p being
ensemble._doubt's score): precision, recall, share of words marked, also at other thresholds; optionally a coordinate search over member weights,
checked on each reference left out of the search, so a gain that is only overfitting shows.
"""
from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import fields
from datetime import datetime

from . import ensemble, jobs, settings, store
from .asr import text as T
from .config import LIBRARY_DIR
from .options import Options

GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
KINDS = ("число", "латиница", "имя", "термин")
DOUBT_CURVE = (0.2, 0.35, 0.5)  # doubt marks at other thresholds too (0.5: the UI's thin underline)
OVERLAP = 0.05  # s: a segment touching a chunk boundary by less does not overlap it


# ---------------------------------------------------------------- words

MIXED_TAIL = re.compile(r"(?<=[a-zäöüß])[а-яё]+$")  # "anmeldungов": a German word with a Russian ending


def norm(words: list[str]) -> list[str]:
    """Comparison keys: case, punctuation and ё ignored, number words = digits («двадцать пять» = 25),
    a Russian case ending on a German word dropped («Wohnungов» = Wohnung)."""
    return norm_groups(words)[0]


def norm_groups(words: list[str]) -> tuple[list[str], list[list[int]]]:
    """norm() and, for every key, the indexes of the words it came from (a number of several words
    is one key; punctuation-only words have none)."""
    toks = T.merge_numbers([dict(w=w, i=i) for i, w in enumerate(words) if T.key(w) or T.is_percent(w)])
    toks = [t for t in toks if t.get("k") or T.key(t["w"])]  # «12 %» = «12%»; a stray «%» is no word
    keys = [MIXED_TAIL.sub("", t.get("k") or T.key(t["w"])) for t in toks]
    return keys, [[x["i"] for x in t.get("parts") or [t]] for t in toks]


def align(ref: list[str], hyp: list[str]) -> list[tuple[str | None, str | None]]:
    """Levenshtein alignment: pairs (ref, hyp); None on one side = deletion / insertion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((ref[i - 1], None))
            i -= 1
        else:
            out.append((None, hyp[j - 1]))
            j -= 1
    return out[::-1]


def errors(ref: list[str], hyp: list[str]) -> int:
    return sum(r != h for r, h in align(ref, hyp))


def ratio(e: int, n: int) -> float | None:
    return round(e / n, 5) if n else None


# ---------------------------------------------------------------- documents

def options(doc: dict, weights: dict | None = None, exclude=()) -> Options:
    o = {k: v for k, v in (doc.get("options") or {}).items() if k in {f.name for f in fields(Options)}}
    opts = Options(**o)
    cfg = dict((opts.extra or {}).get("ensemble") or {})
    if weights:
        cfg["weights"] = {**cfg.get("weights", {}), **weights}
    if exclude:
        cfg["exclude"] = list(cfg.get("exclude", ())) + list(exclude)
    opts.extra = {**(opts.extra or {}), "ensemble": cfg}
    return opts


class NotFound(LookupError):
    pass


def find(doc_id: str) -> str:
    hits = [p.parent.name for p in LIBRARY_DIR.glob(f"*{doc_id}/doc.json")]  # last 6 chars are enough
    if len(hits) != 1:
        raise NotFound(f"документ {doc_id}: найдено {len(hits)}")
    return hits[0]


class Doc:
    def __init__(self, doc_id: str, span: tuple[float, float] | None = None, whole: bool | None = None):
        """whole: the whole document (or span) counts as checked; default: unless it is a gold one."""
        self.doc = store.load(find(doc_id))
        self.id = self.doc["id"]
        self.whole = self.doc.get("gold") is None if whole is None else whole
        self.cache = LIBRARY_DIR / self.id / "hypotheses.json"
        if not self.cache.exists():
            raise NotFound(f"{self.id[-6:]}: нет hypotheses.json (запись распознана не в режиме «макс.»)")
        self.st = json.loads(self.cache.read_text("utf-8"))
        o = options(self.doc)  # chunk languages assigned as ensemble.recombine does
        if self.st.get("detector"):
            fixed = None if (o.mixed or not o.language) else o.language
            ensemble._assign_langs(self.st, fixed, ensemble.MIXED_LANGS if o.mixed else None)
        lo, hi = span or (0.0, float("inf"))
        # chunks that lie inside the checked span; their own time ranges split the words
        self.chunks = [i for i, ch in enumerate(self.st["chunks"])
                       if ch.get("own", [ch["start"], ch["end"]])[0] >= lo - 0.5
                       and ch.get("own", [ch["start"], ch["end"]])[1] <= hi + 0.5]
        self.ref = None

    def own(self, ci: int) -> tuple[float, float]:
        ch = self.st["chunks"][ci]
        return tuple(ch.get("own") or (ch["start"], ch["end"]))

    def by_chunk(self, words: list[dict]) -> dict[int, list[dict]]:
        out = {ci: [] for ci in self.chunks}
        for w in words:
            mid = (w.get("s", 0) + w.get("e", 0)) / 2
            for ci in self.chunks:
                a, b = self.own(ci)
                if a <= mid < b:
                    out[ci].append(w)
                    break
        return out

    def set_reference(self) -> None:
        segs = self.doc["segments"]
        words = []
        for s in segs:
            ok = self.whole or bool(s.get("verified"))
            ws = s.get("words") or [dict(w=x, s=s["start"], e=s["end"]) for x in s["text"].split()]
            words += [dict(w=w["w"], s=w["s"], e=w["e"], first=(k == 0), ok=ok) for k, w in enumerate(ws)]
        self.ref_words = self.by_chunk(words)
        silent = (self.doc.get("gold") or {}).get("silent") or []
        keep = []
        for ci in self.chunks:
            ws = self.ref_words[ci]
            # "???" = the owner couldn't make it out either: such chunks have no reference
            if any("???" in w["w"] for w in ws):
                continue
            if not self.whole:
                a, b = self.own(ci)
                near = [s for s in segs if min(s["end"], b) - max(s["start"], a) > OVERLAP]
                if not all(w["ok"] for w in ws) or not all(s.get("verified") for s in near):
                    continue
                if not ws and not near and not any(min(x[1], b) - max(x[0], a) > OVERLAP for x in silent):
                    continue  # nothing here was listened to
            keep.append(ci)
        self.chunks = keep
        self.ref = {ci: norm([w["w"] for w in self.ref_words[ci]]) for ci in self.chunks}

    def member(self, name: str) -> dict[int, list[str]]:
        """A member's words per chunk (only chunks it was run on and votes for)."""
        m = next((m for m in ensemble.MEMBERS if m["name"] == name), None)
        out = {}
        res = self.st["hyps"].get(name) or []
        for ci in self.chunks:
            ch = self.st["chunks"][ci]
            h = res[ci] if ci < len(res) else None
            if h is None or (m and not ensemble._eligible(m, ch)):
                continue
            out[ci] = [t.get("k") or T.key(t["w"]) for t in ensemble._tokens(h, ch, name)]
        return out

    def merged(self, opts: Options) -> tuple[dict[int, list[str]], dict[int, list[dict]]]:
        st = json.loads(json.dumps(self.st))
        segs = ensemble.combine(st, opts)
        if opts.dictionary:  # as the pipeline does right after recognition
            store.apply_rules_to_segments(segs, settings.rules())
        words = [w for s in segs for w in s["words"]]
        per = self.by_chunk(words)
        return {ci: norm([w["w"] for w in ws]) for ci, ws in per.items()}, per

    def members(self) -> list[str]:
        return [m["name"] for m in ensemble.members(options(self.doc)) if m["name"] in self.st["hyps"]]


def wer(ref: dict, hyp: dict, chunks=None) -> tuple[int, int]:
    chunks = [c for c in (chunks if chunks is not None else hyp) if c in ref]
    return sum(errors(ref[c], hyp.get(c, [])) for c in chunks), sum(len(ref[c]) for c in chunks)


def _names(docs: list[Doc]) -> list[str]:
    order = [m["name"] for m in ensemble.MEMBERS]
    return sorted({n for d in docs for n in d.members()}, key=order.index)


def kind(word: str, first: bool, terms: set[str]) -> str | None:
    k = T.key(word)
    if k in terms:
        return "термин"
    if T.number_value(word) is not None or any(ch.isdigit() for ch in word):
        return "число"
    if T.is_latin(word):
        return "латиница"
    if not first and word[:1].isupper() and T.CYRILLIC.search(word):
        return "имя"
    return None


def merge_errors(docs: list[Doc], weights: dict | None = None, exclude=()) -> tuple[int, int]:
    e = t = 0
    for d in docs:
        hyp, _ = d.merged(options(d.doc, weights, exclude))
        ee, tt = wer(d.ref, hyp, d.chunks)
        e, t = e + ee, t + tt
    return e, t


# ---------------------------------------------------------------- evaluation

def evaluate(docs: list[Doc], tune: bool = False, note: str | None = None) -> dict:
    """The result dict the app shows and stores (HOME/gold/latest.json); wer values are fractions."""
    docs = [d for d in docs if d.chunks]
    total = sum(len(r) for d in docs for r in d.ref.values())
    res = dict(t=datetime.now().isoformat(timespec="seconds"), docs=[d.id for d in docs], words=total,
               chunks=sum(len(d.chunks) for d in docs), merged=None, members=[], oracle=None, ablation=[],
               kinds={k: dict(wer=None, words=0) for k in KINDS}, doubts=None, tune=None, note=note)
    if not total:
        res["note"] = "; ".join(x for x in (note, "проверенных на слух кусков пока нет") if x)
        return res
    names = _names(docs)
    merged = {d.id: d.merged(options(d.doc)) for d in docs}
    e, t = merge_errors(docs)
    res["merged"] = dict(wer=ratio(e, t), errors=e, words=t)

    hyps = {d.id: {n: d.member(n) for n in names} for d in docs}
    for n in names:
        me = mt = se = st_ = 0
        for d in docs:
            h = hyps[d.id][n]
            a, b = wer(d.ref, h)
            c, dd = wer(d.ref, merged[d.id][0], list(h))
            me, mt, se, st_ = me + a, mt + b, se + c, st_ + dd
        res["members"].append(dict(name=n, wer=ratio(me, mt), words=mt, merged_wer=ratio(se, st_)))
    oe = 0
    for d in docs:
        for ci in d.chunks:
            cands = [hyps[d.id][n].get(ci) for n in names] + [merged[d.id][0].get(ci, [])]
            oe += min(errors(d.ref[ci], c) for c in cands if c is not None)
    res["oracle"] = dict(wer=ratio(oe, total))

    for n in names:
        ex, _ = merge_errors(docs, exclude=[n])
        res["ablation"].append(dict(name=n, wer=ratio(ex, t), delta=ratio(ex - e, t)))

    # errors by kind of the reference word and doubt marks
    terms = set()
    for d in docs:
        terms |= {T.key(x) for x in (d.doc.get("options") or {}).get("glossary", "").replace(";", ",").split(",") if T.key(x)}
    by = {k: [0, 0] for k in KINDS}
    marks = []  # (p of a merged word, wrong?)
    for d in docs:
        per = merged[d.id][1]
        for ci in d.chunks:
            raw = d.ref_words.get(ci, [])
            kinds = [kind(w["w"], w["first"], terms) for w in raw]
            ref_keys = d.ref.get(ci, [])
            # merge_numbers may glue reference words: kinds follow the keys only when lengths match
            kinds = kinds if len(kinds) == len(ref_keys) else [("число" if k.startswith("#") else None) for k in ref_keys]
            hw = per.get(ci, [])
            hkeys, groups = norm_groups([w["w"] for w in hw])
            # a number of several words is one key: it is doubtful if any of its words is
            ps = [min(hw[i].get("p", 1.0) for i in g) for g in groups]
            ri = hi = 0
            for r, h in align(ref_keys, hkeys):
                if r is not None:
                    k = kinds[ri]
                    if k:
                        by[k][1] += 1
                        by[k][0] += r != h
                    ri += 1
                if h is not None:
                    marks.append((ps[hi], r != h))
                    hi += 1
    res["kinds"] = {k: dict(wer=ratio(*by[k]), words=by[k][1]) for k in KINDS}
    if marks:
        res["doubts"] = dict(doubt_marks(marks, store.REVIEW_P),
                             curve=[doubt_marks(marks, x) for x in DOUBT_CURVE])
    if tune:
        res["tune"] = tune_weights(docs, names)
    return res


def doubt_marks(marks: list[tuple[float, bool]], threshold: float) -> dict:
    """Words marked as doubtful (p < threshold): their share, how many of them are wrong (precision)
    and how many of the wrong words they cover (recall). Words the merge dropped are not counted."""
    tp = sum(1 for p, bad in marks if p < threshold and bad)
    fp = sum(1 for p, bad in marks if p < threshold and not bad)
    fn = sum(1 for p, bad in marks if p >= threshold and bad)
    return dict(threshold=threshold, marked=ratio(tp + fp, len(marks)), precision=ratio(tp, tp + fp),
                recall=ratio(tp, tp + fn))


def tune_weights(docs: list[Doc], names: list[str]) -> dict:
    base = {m["name"]: m["weight"] for m in ensemble.MEMBERS if m["name"] in names}

    def search(train: list[Doc]) -> dict:
        w = dict(base)
        best = merge_errors(train, w)[0]
        for _ in range(3):
            moved = False
            for n in names:
                for g in GRID:
                    if g == w[n]:
                        continue
                    e = merge_errors(train, {**w, n: g})[0]
                    if e < best:
                        best, w, moved = e, {**w, n: g}, True
            if not moved:
                break
        return w

    e0, t = merge_errors(docs)
    w = search(docs)
    e1, _ = merge_errors(docs, w)
    held = []
    if len(docs) >= 2:  # weights searched without one reference, checked on it
        for d in docs:
            wh = search([x for x in docs if x is not d])
            a, n_ = merge_errors([d])
            b, _ = merge_errors([d], wh)
            held.append(dict(doc=d.id, before=ratio(a, n_), after=ratio(b, n_)))
    return dict(base=base, weights=w, wer_before=ratio(e0, t), wer_after=ratio(e1, t), heldout=held)


def agreement(docs: list[Doc]) -> dict:
    """Without a reference: pairwise WER between members (row against column, common chunks)
    and each member against the merge of the others (leave-one-out). Members that agree with
    each other but not with the rest make the same mistakes and outvote the others."""
    names = _names(docs)
    hyps = {d.id: {n: d.member(n) for n in names} for d in docs}
    pair = {}
    for a in names:
        pair[a] = {}
        for b in names:
            if a == b:
                continue
            e = t = 0
            for d in docs:
                ha, hb = hyps[d.id][a], hyps[d.id][b]
                ee, tt = wer(hb, ha, [c for c in ha if c in hb])
                e, t = e + ee, t + tt
            pair[a][b] = ratio(e, t)
    loo = []
    for n in names:
        e = t = 0
        for d in docs:
            ref, _ = d.merged(options(d.doc, exclude=[n]))
            ee, tt = wer(ref, hyps[d.id][n])
            e, t = e + ee, t + tt
        loo.append(dict(name=n, wer=ratio(e, t), words=t))
    return dict(docs=[d.id for d in docs], names=names, pairwise=pair, loo=loo)


# ---------------------------------------------------------------- text reports (CLI)

def pct(x: float | None) -> str:
    return f"{100 * x:5.1f}%" if x is not None else "   - "


def report_text(res: dict) -> str:
    out = [f"\nЭталон: {len(res['docs'])} док., {res['chunks']} кусков, {res['words']} слов"]
    if res.get("note"):
        out.append(f"Примечание: {res['note']}")
    if not res["merged"]:
        return "\n".join(out)
    out.append(f"Слияние: WER {pct(res['merged']['wer'])}")
    out.append("\nУчастники (на кусках, где голосуют) и слияние на тех же кусках:")
    for m in res["members"]:
        out.append(f"  {m['name']:<12} {pct(m['wer'])}   слияние {pct(m['merged_wer'])}   ({m['words']} слов)")
    out.append(f"  {'оракул':<12} {pct(res['oracle']['wer'])}   (лучший кусок из всех вариантов)")
    out.append("\nБез одного участника (минус = без него лучше):")
    for a in res["ablation"]:
        out.append(f"  без {a['name']:<12} {pct(a['wer'])}   {100 * a['delta']:+.2f} п.")
    out.append("\nОшибки по видам слов эталона:")
    for k, v in res["kinds"].items():
        out.append(f"  {k:<9} {pct(v['wer'])}   ({v['words']} слов)")
    dz = res.get("doubts")
    if dz:
        out.append(f"\nПометки сомнения (p < {dz['threshold']}): помечено {pct(dz['marked'])} слов, "
                   f"точность {pct(dz['precision'])}, полнота {pct(dz['recall'])}")
        for c in dz.get("curve") or []:
            out.append(f"  порог {c['threshold']:<5} помечено {pct(c['marked'])}, точность {pct(c['precision'])}, "
                       f"полнота {pct(c['recall'])}")
    tn = res.get("tune")
    if tn:
        out.append(f"\nВеса, подобранные на всех эталонах: WER {pct(tn['wer_before'])} -> {pct(tn['wer_after'])}")
        for n, w in tn["weights"].items():
            mark = "" if w == tn["base"][n] else f"   (было {tn['base'][n]})"
            out.append(f"  {n:<12} {w}{mark}")
        if tn["heldout"]:
            out.append("\nПроверка на документе, не участвовавшем в подборе:")
            for h in tn["heldout"]:
                out.append(f"  {h['doc'][-6:]}: {pct(h['before'])} -> {pct(h['after'])}")
    return "\n".join(out)


def agreement_text(ag: dict) -> str:
    names = ag["names"]
    out = ["\nПопарное расхождение (WER строки относительно столбца), общие куски:",
           " " * 12 + "".join(f"{n[:10]:>11}" for n in names)]
    for a in names:
        out.append(f"{a[:11]:<12}" + "".join("      -    " if a == b else f"{pct(ag['pairwise'][a][b]):>11}"
                                              for b in names))
    out.append("\nКаждый против слияния остальных (leave-one-out):")
    for x in ag["loo"]:
        out.append(f"  {x['name']:<12} {pct(x['wer'])}  ({x['words']} слов слияния)")
    return "\n".join(out)


# ---------------------------------------------------------------- the app's evaluation process

def gold_docs() -> tuple[list[Doc], list[str]]:
    """Every gold document of the library, with references set; and what was skipped."""
    docs, skipped = [], []
    for s in store.list_docs():
        if not s.get("gold"):
            continue
        if s["status"] != "done":
            skipped.append(f"{s['id'][-6:]}: ещё обрабатывается")
            continue
        try:
            d = Doc(s["id"])
        except NotFound as e:
            skipped.append(str(e))
            continue
        d.set_reference()
        docs.append(d)
    return docs, skipped


def run(tune: bool = False) -> dict:
    docs, skipped = gold_docs()
    note = None
    if not docs and not skipped:
        note = "нет эталонных записей"
    elif skipped:
        note = "пропущены: " + "; ".join(skipped)
    res = evaluate(docs, tune, note)
    jobs.save_gold_result(res)
    return res


def main(argv: list[str] | None = None) -> None:
    """python -m whisper_studio gold-eval [--tune]: started by the app (jobs.start_gold_eval)."""
    argv = sys.argv[1:] if argv is None else argv
    jobs.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    # pythonw started through WMI has no std handles at all
    sys.stdout = sys.stderr = open(jobs.GOLD_LOG, "a", encoding="utf-8", buffering=1)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{os.getpid()}] gold-eval {' '.join(argv)}", flush=True)
    try:
        res = run("--tune" in argv)
        print(f"done: {len(res['docs'])} docs, {res['words']} words, wer {res['merged'] and res['merged']['wer']}")
    except Exception as e:  # noqa: BLE001 - the app shows the error
        traceback.print_exc()
        jobs.gold_failed(f"{type(e).__name__}: {e}")
    finally:
        jobs.gold_finished()
        sys.stdout.flush()
