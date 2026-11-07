"""Term recovery (whisper_studio/terms.py) on stored hypotheses: what it replaced, and the
gold WER with and without it. CPU only, nothing is written to the library.

    ws-python tools/terms_eval.py --ref a1b2c3                # gold: errors before/after, Latin words
    ws-python tools/terms_eval.py --doc d4e5f6 --doc 2b3c4d   # other docs: every replacement
    ws-python tools/terms_eval.py --all-docs                  # every "max" doc except gold ones
    WHISPER_STUDIO_TERMS=path\\terms-learned.json ...          # another learned list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, settings, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402


def run(d: gold.Doc, on: bool) -> tuple[dict, list[dict], list]:
    opts = gold.options(d.doc)
    cfg = dict((opts.extra or {}).get("ensemble") or {})
    if not on:
        cfg["no_terms"] = True
    opts.extra = {**(opts.extra or {}), "ensemble": cfg}
    st = json.loads(json.dumps(d.st))
    segs = ensemble.combine(st, opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    words = [w for s in segs for w in s["words"]]
    per = d.by_chunk(words)
    return {ci: gold.norm([w["w"] for w in ws]) for ci, ws in per.items()}, segs, st.get("term_log") or []


def latin_errors(d: gold.Doc, hyp: dict) -> tuple[int, int]:
    e = n = 0
    for ci in d.chunks:
        for r, h in gold.align(d.ref[ci], hyp.get(ci, [])):
            if r is not None and T.LATIN.search(r):
                n += 1
                e += r != h
    return e, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", default=[])
    ap.add_argument("--doc", action="append", default=[])
    ap.add_argument("--all-docs", action="store_true")
    ap.add_argument("--context", action="store_true", help="показать реплику вокруг каждой замены")
    a = ap.parse_args()
    for r in a.ref:
        d = gold.Doc(r)
        d.set_reference()
        h0, _, _ = run(d, False)
        h1, _, log = run(d, True)
        e0, t = gold.wer(d.ref, h0, d.chunks)
        e1, _ = gold.wer(d.ref, h1, d.chunks)
        l0, ln = latin_errors(d, h0)
        l1, _ = latin_errors(d, h1)
        print(f"\n{d.id[-6:]}: {t} слов. Ошибок слияния {e0} -> {e1} (WER {100 * e0 / t:.1f}% -> {100 * e1 / t:.1f}%); "
              f"латиница {l0}/{ln} -> {l1}/{ln}; замен {len(log)}")
        for x in log:
            print(f"  [{x['pass_']}] {x['t']:7.1f}s  {x['old']!r:32} -> {x['new']:<22} sim {x['sim']:.2f}  "
                  + " ".join(f"{n}:{s}{'' if r else '*'}" for n, s, r in x["ev"] if s >= 0.6))
        for ci in d.chunks:
            al0 = sum(r != h for r, h in gold.align(d.ref[ci], h0.get(ci, [])))
            al1 = gold.align(d.ref[ci], h1.get(ci, []))
            e = sum(r != h for r, h in al1)
            if e != al0:
                print(f"  кусок {ci}: {al0} -> {e}   " + " | ".join(f"{r}->{h}" for r, h in al1 if r != h))
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
    total = words = 0
    for i in ids:
        try:
            d = gold.Doc(i, whole=True)
        except gold.NotFound as e:
            print(e)
            continue
        _, segs, log = run(d, True)
        n = sum(len(s["words"]) for s in segs)
        total += len(log)
        words += n
        print(f"\n{d.id[-6:]} ({(d.doc.get('duration') or 0) / 60:.0f} мин, {n} слов): замен {len(log)}")
        for x in log:
            print(f"  [{x['pass_']}] {x['t']:7.1f}s  {x['old']!r:32} -> {x['new']:<22} sim {x['sim']:.2f}  "
                  + " ".join(f"{n}:{s}{'' if r else '*'}" for n, s, r in x["ev"] if s >= 0.6))
            if a.context:
                seg = next((s for s in segs if s["start"] - 0.05 <= x["t"] <= s["end"] + 0.05), None)
                print(f"        «{seg['text'] if seg else '?'}»")
    if ids:
        print(f"\nВсего: {total} замен на {words} слов")


if __name__ == "__main__":
    main()
