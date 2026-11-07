"""Retrieval memory (H9, docs/DECISIONS.md) on stored hypotheses: the term pass with the full term
list vs per-chunk retrieved subsets. CPU only, nothing is written to the library.

    set WHISPER_STUDIO_MEMORY=%WS_REAL_HOME%\\tmp\\agent-memory\\memory
    ws-python tools/memory_eval.py --ref a1b2c3 --show        # gold: errors per variant + what was retrieved
    ws-python tools/memory_eval.py --all-docs                 # other docs: replacements per variant

Variants: off (today: every chunk against all terms), focus (a: retrieved terms + dictionary core),
loose (b: focus, and retrieved terms count as primed with a sound bonus), each also without the
extra German nouns of the passages (-noextra). A document's own recording is always left out of
the memory (memory_exclude = its source path), so no document is helped by its own text.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from terms_eval import latin_errors  # noqa: E402

from whisper_studio import ensemble, gold, memory, settings, store, terms  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402

VARIANTS = {
    "off": {},
    "focus": dict(memory=True, memory_extras=False),
    "focus+new": dict(memory=True, memory_extras=True),
    "loose": dict(memory=True, memory_extras=False, memory_loose=True),
    "loose+new": dict(memory=True, memory_extras=True, memory_loose=True),
}


AFTER = "__after__"  # rec mark of wiki passages left out by --wiki-before


def hide_after(day: str, wiki: str) -> int:
    """Leave out wiki passages of files changed on or after `day`."""
    from datetime import datetime

    lim = datetime.fromisoformat(day).timestamp()
    idx = memory.index()
    n = 0
    for p in idx.passages:
        f = Path(wiki) / p["src"]
        if p["kind"] == "wiki" and (not f.exists() or f.stat().st_mtime >= lim):
            p["rec"] = AFTER
            n += 1
    return n


def run(d: gold.Doc, extra: dict) -> tuple[dict, list[dict], list, dict]:
    opts = gold.options(d.doc)
    cfg = dict((opts.extra or {}).get("ensemble") or {})
    cfg.update(extra)
    cfg["memory_exclude"] = [str(d.doc.get("source_path") or d.id).lower(), AFTER]
    opts.extra = {**(opts.extra or {}), "ensemble": cfg}
    st = json.loads(json.dumps(d.st))
    segs = ensemble.combine(st, opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    per = d.by_chunk([w for s in segs for w in s["words"]])
    return {ci: gold.norm([w["w"] for w in ws]) for ci, ws in per.items()}, segs, st.get("term_log") or [], st


def show_retrieval(d: gold.Doc) -> None:
    """What the memory found for each chunk (first-pass merge without the term pass as query)."""
    opts = gold.options(d.doc)
    opts.extra = {**(opts.extra or {}), "ensemble": {**((opts.extra or {}).get("ensemble") or {}), "no_terms": True}}
    st = json.loads(json.dumps(d.st))
    segs = ensemble.combine(st, opts)
    tl = terms.term_list(ensemble._glossary(opts), learned=bool(opts.dictionary))
    core = terms.term_list(ensemble._glossary(opts), learned=False)
    f = memory.retrieve(segs, st["chunks"], tl, core, exclude={str(d.doc.get("source_path")).lower(), AFTER})
    if f is None:
        print("нет индекса памяти или модели")
        return
    print(f"\nПамять для {d.id[-6:]}: найдено терминов на кусок (без ядра словаря {len(f.core)})")
    for r in f.report:
        got = f.spans[r["chunk"]][3]
        print(f"\n кусок {r['chunk']}: лучший {r['best']}  терминов {len(got)}  новых: {', '.join(r['new']) or '-'}")
        print(f"   запрос: {r['query'][:160]}")
        for p in r["passages"]:
            print(f"   {p['score']:.3f} {p['src'][:70]}: {p['text'][:110]}")
        print(f"   термины: {', '.join(sorted(t for t in r['terms']))[:400]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--doc", action="append", default=[])
    ap.add_argument("--all-docs", action="store_true")
    ap.add_argument("--variant", action="append", choices=list(VARIANTS), help="по умолчанию все")
    ap.add_argument("--show", action="store_true", help="что нашла память по кускам")
    ap.add_argument("--json", help="записать итог сюда")
    ap.add_argument("--wiki", default=os.environ.get("WS_WIKI", ""),
                    help="notes folder the index was built from (for --wiki-before)")
    ap.add_argument("--wiki-before", help="без файлов вики, изменённых с этого дня (YYYY-MM-DD)")
    ap.add_argument("--k", type=int, help=f"отрывков на кусок (memory.TOP_K = {memory.TOP_K})")
    ap.add_argument("--rec-k", type=int, help=f"отрывков на запись (memory.REC_TOP_K = {memory.REC_TOP_K})")
    ap.add_argument("--margin", type=float, help=f"memory.MARGIN = {memory.MARGIN}")
    ap.add_argument("--min-score", type=float, help=f"memory.MIN_SCORE = {memory.MIN_SCORE}")
    a = ap.parse_args()
    if a.wiki_before:
        if not a.wiki:
            sys.exit("--wiki-before needs --wiki or WS_WIKI")
        print(f"вики после {a.wiki_before}: убрано {hide_after(a.wiki_before, a.wiki)} отрывков")
    for name, v in (("TOP_K", a.k), ("REC_TOP_K", a.rec_k), ("MARGIN", a.margin), ("MIN_SCORE", a.min_score)):
        if v is not None:
            setattr(memory, name, v)
    names = a.variant or list(VARIANTS)
    result: dict = {"gold": {}, "docs": {}}
    for r in a.ref:
        d = gold.Doc(r)
        d.set_reference()
        if a.show:
            show_retrieval(d)
        print(f"\n{d.id[-6:]}:")
        base_log = None
        for v in names:
            hyp, _, log, st = run(d, VARIANTS[v])
            e, t = gold.wer(d.ref, hyp, d.chunks)
            le, ln = latin_errors(d, hyp)
            result["gold"][v] = dict(errors=e, words=t, latin=le, replaced=len(log))
            print(f"  {v:<10} ошибок {e} ({100 * e / t:.1f}%), латиница {le}/{ln}, замен {len(log)}")
            if base_log is None:
                base_log = {(x["t"], x["new"]) for x in log}
                continue
            now = {(x["t"], x["new"]) for x in log}
            for x in log:
                if (x["t"], x["new"]) not in base_log:
                    print(f"     + {x['t']:7.1f}s {x['old']!r} -> {x['new']}  sim {x['sim']}")
            for t_, new in sorted(base_log - now):
                print(f"     - {t_:7.1f}s -> {new}")
    ids = list(a.doc)
    if a.all_docs:
        gids = {s["id"] for s in store.list_docs() if s.get("gold")}
        seen = {}
        for p in sorted(LIBRARY_DIR.glob("*/hypotheses.json")):
            doc = store.load(p.parent.name)
            if doc["id"] in gids or doc.get("status") != "done":
                continue
            seen[doc.get("source_path") or doc["id"]] = doc["id"]  # newest run of a recording
        ids += list(seen.values())
    totals = {v: 0 for v in names}
    for i in ids:
        try:
            d = gold.Doc(i, whole=True)
        except gold.NotFound as e:
            print(e)
            continue
        logs = {}
        for v in names:
            _, segs, log, _ = run(d, VARIANTS[v])
            logs[v] = (log, segs)
            totals[v] += len(log)
        base = {(x["t"], x["new"]): x for x in logs[names[0]][0]}
        print(f"\n{d.id[-6:]}: " + ", ".join(f"{v} {len(logs[v][0])}" for v in names))
        result["docs"][d.id[-6:]] = {v: [(x["t"], x["old"], x["new"]) for x in logs[v][0]] for v in names}
        for v in names[1:]:
            now = {(x["t"], x["new"]): x for x in logs[v][0]}
            segs = logs[v][1]
            for k in sorted(set(base) - set(now)):
                x = base[k]
                print(f"  {v:<10} - {x['t']:7.1f}s {x['old']!r} -> {x['new']}")
            for k in sorted(set(now) - set(base)):
                x = now[k]
                seg = next((s for s in segs if s["start"] - 0.05 <= x["t"] <= s["end"] + 0.05), None)
                print(f"  {v:<10} + {x['t']:7.1f}s {x['old']!r} -> {x['new']}  «{(seg or {}).get('text', '?')[:120]}»")
    if ids:
        print("\nВсего замен: " + ", ".join(f"{v} {n}" for v, n in totals.items()))
    if a.json:
        Path(a.json).write_text(json.dumps(result, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
