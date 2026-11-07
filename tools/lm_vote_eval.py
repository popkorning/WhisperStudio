"""H8: a language model as a weak extra vote on disputed slots of the merge (docs/DECISIONS.md).

    ws-python tools/lm_vote_eval.py --ref a1b2c3 --model %WS_REAL_HOME%/models/llm-qwen3-1.7b-base
    ... --masked %WS_REAL_HOME%/models/llm-rumodernbert-base     # pseudo-log-likelihood instead
    ... --oracle-only                                               # only the selection oracle
    ... --terms                                                     # dictionary terms as extra options
    ... --device cuda --dtype float16                               # on the GPU (only when it is free!)

CPU by default (CUDA hidden unless --device cuda). Reads the library and the dictionary, writes
nothing there: scores and results go to --out (default %WS_REAL_HOME%/tmp/agent-llm).
The LM scores are cached per model, so other weights are re-measured without the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--ref", action="append", default=[])
ap.add_argument("--model", help="causal LM folder (transformers)")
ap.add_argument("--masked", help="masked LM folder (pseudo-log-likelihood)")
ap.add_argument("--device", default="cpu")
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--dispute", type=float, default=0.35)
ap.add_argument("--weights", default="0.25,0.5,1,1.5,2,3,5")
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--oracle-only", action="store_true")
ap.add_argument("--terms", action="store_true", help="dictionary terms as extra options")
ap.add_argument("--out", default=str(REAL_HOME / "tmp/agent-llm"))
ap.add_argument("--home", default=str(REAL_HOME), help="where the library is (read only)")
a = ap.parse_args()

if a.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
OUT = Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
os.environ["WHISPER_STUDIO_HOME"] = str(OUT)  # the app's own writes, if any, land here
os.environ.setdefault("HF_HOME", str(Path(a.home) / "hf-cache"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from whisper_studio import config, ensemble, gold, lm_vote, models, settings, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402

# read-only views of the real library, models and dictionary
HOME = Path(a.home)
for mod in (config, store, gold):
    mod.LIBRARY_DIR = HOME / "library"
config.MODELS_DIR = models.MODELS_DIR = HOME / "models"
settings.DICTIONARY_FILE = HOME / "dictionary.json"
settings.SETTINGS_FILE = HOME / "settings.json"
store.INDEX_FILE = HOME / "library" / "index.json"


def merged_wer(d: gold.Doc, opts, ctx=None) -> tuple[int, int, dict]:
    st = json.loads(json.dumps(d.st))
    with (ctx or _null()):
        segs = ensemble.combine(st, opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    per = d.by_chunk([w for s in segs for w in s["words"]])
    hyp = {ci: gold.norm([w["w"] for w in ws]) for ci, ws in per.items()}
    e, n = gold.wer(d.ref, hyp, d.chunks)
    return e, n, hyp


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *x):
        return False


def chunk_of(d: gold.Doc, t: float | None) -> int | None:
    if t is None:
        return None
    for ci, ch in enumerate(d.st["chunks"]):
        a_, b_ = ch.get("own") or (ch["start"], ch["end"])
        if a_ <= t < b_:
            return ci
    return None


def oracle(d: gold.Doc, slots: list[dict], allowed: set[int]) -> dict[int, object]:
    """Greedy word-level oracle: per gold chunk, each slot (in `allowed`) takes the option that
    gives the fewest errors against the reference, others fixed; two sweeps."""
    choice = {i: s["win"] for i, s in enumerate(slots)}
    by = {}
    for i, s in enumerate(slots):
        by.setdefault(s["chunk"], []).append(i)

    def text(i, o):
        if o is None:
            return None
        return o[1] if isinstance(o, tuple) else slots[i]["words"].get(o, o)

    for ci in d.chunks:
        idx = by.get(ci, [])
        ref = d.ref[ci]

        def err():
            return gold.errors(ref, gold.norm([w for i in idx if (w := text(i, choice[i]))]))

        best = err()
        for _ in range(2):
            for i in idx:
                if i not in allowed:
                    continue
                for o in list(slots[i]["options"]) + slots[i].get("terms", []):
                    if o == choice[i]:
                        continue
                    old = choice[i]
                    choice[i] = o
                    e = err()
                    if e < best:
                        best = e
                    else:
                        choice[i] = old
    return {i: o for i, o in choice.items() if o != slots[i]["win"]}


def term_options(slots: list[dict], terms: list[str], thr: float = 0.7) -> dict[int, list]:
    """Dictionary terms that sound like one of the slot's variants (and nobody produced)."""
    out = {}
    keys = [(T.key(t), t) for t in terms]
    for i, s in enumerate(slots):
        if s["chunk"] is None:
            continue
        have = {k for k in s["options"] if k}
        cands = []
        for k in have:
            if len(k) < 4 or T.number_value(k) is not None:
                continue
            for tk, t in keys:
                if tk in have or len(tk) < 4:
                    continue
                if T.phon_sim(k, tk) >= thr and ("term", t) not in cands:
                    cands.append(("term", t))
        if cands:
            out[i] = cands[:3]
    return out


def main() -> None:
    refs = a.ref or ["a1b2c3"]
    log = lambda *x: print(*x, flush=True)  # noqa: E731
    result = dict(refs=refs, model=a.model or a.masked, dispute=a.dispute, rows=[])
    for r in refs:
        d = gold.Doc(r)
        d.set_reference()
        opts = gold.options(d.doc)
        base_e, n, _ = merged_wer(d, opts)
        with lm_vote.recording() as rec:
            ensemble.combine(json.loads(json.dumps(d.st)), opts)
        slots = rec.slots
        for s in slots:
            s["chunk"] = chunk_of(d, s["t"])
        gold_slots = {i for i, s in enumerate(slots) if s["chunk"] in d.chunks}
        dis = [i for i in lm_vote.disputed(slots, a.dispute) if i in gold_slots]
        multi = [i for i in gold_slots if len(slots[i]["options"]) > 1]
        audio = sum((d.own(ci)[1] - d.own(ci)[0]) for ci in d.chunks) / 60
        log(f"{d.id[-6:]}: {len(d.chunks)} кусков, {n} слов, {audio:.1f} мин; слияние {base_e} ошибок "
            f"({100 * base_e / n:.1f}%); слотов {len(gold_slots)}, с вариантами {len(multi)}, спорных {len(dis)}")
        sigs = [s["sig"] for s in slots]
        extra = {}
        if a.terms:
            extra = {i: v for i, v in term_options(slots, settings.dictionary()["terms"]).items() if i in gold_slots}
            for i, v in extra.items():
                slots[i]["terms"] = v
            log(f"  словарных вариантов: {sum(map(len, extra.values()))} в {len(extra)} слотах: "
                + "; ".join(f"{'/'.join(k or '∅' for k in slots[i]['options'])} -> {', '.join(t for _, t in v)}"
                            for i, v in list(extra.items())[:12]))

        def wer_with(post, w):
            e, _, _ = merged_wer(d, opts, lm_vote.voting(post, w, sigs))
            return e

        # selection oracle: all slots with variants, and only the disputed ones
        for name, allowed in (("оракул: все слоты", set(gold_slots)), ("оракул: спорные", set(dis))):
            ch = oracle(d, slots, allowed)
            e = wer_with({i: {o: 1.0} for i, o in ch.items()}, 100.0)
            log(f"  {name}: {e} ошибок ({100 * e / n:.1f}%), поменяно слотов {len(ch)}")
            result["rows"].append(dict(doc=d.id, kind=name, errors=e, words=n, changed=len(ch)))
            if a.oracle_only:
                for i, o in sorted(ch.items()):
                    s = slots[i]
                    log(f"    m={s['margin']:.2f} {' / '.join(f'{s['words'].get(k, '∅') if k else '∅'}:{v:.2f}' for k, v in s['options'].items())}"
                        f"  было {s['win']} -> {o}   «{lm_vote.contexts(slots, i, [])[0][-60:]}»")
        if a.oracle_only or not (a.model or a.masked):
            result["rows"].append(dict(doc=d.id, kind="база", errors=base_e, words=n))
            continue

        tag = Path(a.model or a.masked).name + ("-terms" if a.terms else "")
        cache = OUT / f"scores-{d.id[-6:]}-{tag}-{a.dispute}.json"
        if cache.exists():
            sc = {int(k): v for k, v in json.loads(cache.read_text("utf-8")).items()}
            for v in sc.values():
                v["options"] = [tuple(o) if isinstance(o, list) else o for o in v["options"]]
            log(f"  оценки из кэша {cache.name}")
        else:
            t0 = time.monotonic()
            if a.model:
                scorer = lm_vote.CausalScorer(a.model, a.dtype, a.threads, a.device)
            else:
                scorer = lm_vote.MaskedScorer(a.masked, a.threads)
            t_load = time.monotonic() - t0
            t1 = time.monotonic()
            if a.terms:  # bonus: only the slots that got dictionary terms
                dis = sorted(extra)
            sc = lm_vote.option_scores(scorer, slots, dis, extra, log)
            secs = time.monotonic() - t1
            log(f"  загрузка {t_load:.0f} с, оценка {len(dis)} слотов {secs:.0f} с = {secs / audio:.1f} с на минуту звука")
            result["time"] = dict(load=round(t_load, 1), score=round(secs, 1), per_min=round(secs / audio, 2),
                                  slots=len(dis), audio_min=round(audio, 2))
            cache.write_text(json.dumps({str(k): v for k, v in sc.items()}, ensure_ascii=False), "utf-8")
        # how often the LM's favourite agrees with the oracle choice on disputed slots
        ch = oracle(d, slots, set(dis))
        ws = [float(x) for x in a.weights.split(",")]
        def neighbours(i):
            return {slots[j]["win"] for j in range(max(0, i - 2), min(len(slots), i + 3)) if j != i}

        for mode in ("all", "words", "similar"):
            # words: only slots where the two strongest options are both words, and the LM chooses
            # among the words only ("nothing here" is left to the members: a sum of log-probs
            # always prefers the shorter sentence)
            use = {}
            for i, r in sc.items():
                top2 = sorted(slots[i]["options"], key=slots[i]["options"].get, reverse=True)[:2]
                if mode in ("words", "similar"):
                    if None in top2:
                        continue
                    keep_o = [k for k, o in enumerate(r["options"]) if o is not None]
                if mode == "similar":
                    # real acoustic confusions only: variants that sound like the winner and are not
                    # a neighbour's word shifted into this slot by the alignment
                    wk = slots[i]["win"]
                    keep_o = [k for k in keep_o if r["options"][k] == wk or (
                        isinstance(r["options"][k], str) and r["options"][k] not in neighbours(i)
                        and T.phon_sim(r["options"][k], wk or "") >= 0.5)]
                    if len(keep_o) < 2:
                        continue
                if mode in ("words", "similar"):
                    r = dict(r, options=[r["options"][k] for k in keep_o], lp=[r["lp"][k] for k in keep_o])
                use[i] = r
            for norm in (False, True):
                post = lm_vote.to_posterior(use, norm, a.temp)
                agree = sum(1 for i in post if i in ch and max(post[i], key=post[i].get) == ch[i])
                keep = sum(1 for i in post if i not in ch and max(post[i], key=post[i].get) == slots[i]["win"])
                log(f"  {mode} {'норм.' if norm else 'сумма'}: слотов {len(post)}, LM = оракул на "
                    f"{agree}/{sum(1 for i in post if i in ch)} исправимых, не трогает верное: "
                    f"{keep}/{sum(1 for i in post if i not in ch)}")
                for thr in (0.2, 0.35, 0.6):
                    sub = {i: v for i, v in post.items() if slots[i]["margin"] < thr}
                    es = []
                    for w in ws:
                        e = wer_with(sub, w)
                        es.append(e)
                        result["rows"].append(dict(doc=d.id, mode=mode, norm=norm, thr=thr, w=w, errors=e, words=n))
                    log(f"    margin<{thr:<4} ({len(sub):3} сл.): " + "  ".join(f"w{w}:{e - base_e:+d}" for w, e in zip(ws, es)))
        if a.terms:
            for i, v in extra.items():
                if i in sc:
                    p = lm_vote.to_posterior({i: sc[i]}, False, a.temp)[i]
                    best = max(p, key=p.get)
                    log(f"    слот {'/'.join(k or '∅' for k in slots[i]['options'])}: LM -> {best} "
                        f"(оракул {ch.get(i, slots[i]['win'])})")
        result["rows"].append(dict(doc=d.id, kind="база", errors=base_e, words=n))
    (OUT / f"result-{Path(a.model or a.masked or 'oracle').name}{'-terms' if a.terms else ''}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str), "utf-8")


if __name__ == "__main__":
    main()
