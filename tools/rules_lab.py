"""Dictionary replacements learned from the synthetic dialogue set (lab experiment 18): what the
recognizers write where a German term was said (tools/dialog_eval.py --dump words.json) becomes
a rule "stem* -> Term", but only for words that are no Russian words, no Latin word of the
dictionary and long enough. The rules are then tried where they were not learned: the held-out
part of the set, the gold, and every real recording (each change is printed).

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/rules_lab.py --learn %WS_REAL_HOME%/tmp/dialog-set/dev/words.json --out rules.json
    ws-python tools/rules_lab.py --rules rules.json --doc <id> --set %WS_REAL_HOME%/tmp/dialog-set/test
    ws-python tools/rules_lab.py --rules rules.json --doc a1b2c3
    ws-python tools/rules_lab.py --rules rules.json --library          (real home)
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, settings, store, terms  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dialog_eval as DE  # noqa: E402

MIN_STEM = 4
WILD_STEM = 6     # a stem with "*" only this long
PROBE = ("а", "у", "е", "ы", "и", "ом", "ой", "ов", "ам", "ами", "ах", "ия", "ие", "ка", "ки", "ить", "ать", "ся", "ный")
SOURCES = ("слияние", "large-v3", "ru-large-v3", "qwen3-asr", "gigaam-v3", "t-one")


def stem(w: str) -> str:
    k = re.sub(r"[^\wё]", "", w.lower().replace("ё", "е"))
    for e in sorted(terms.ENDINGS, key=len, reverse=True):
        if k.endswith(e) and len(k) - len(e) >= MIN_STEM:
            return k[: -len(e)]
    if len(k) > MIN_STEM and k[-1] in "аеиоуыюя":
        return k[:-1]
    return k


def russian_like(st: str) -> bool:
    """Some Russian word starts with this stem (checked on the usual endings)."""
    return any(T.known_russian(st + e) for e in ("",) + PROBE)


def learn(rows: list[dict]) -> list[dict]:
    """Rules from every wrong spelling at a one-word term's place, when the spelling is no Russian
    word: the word itself, and its stem with any ending when the stem is long and no Russian word
    starts with it. A rule seen for two different terms is dropped."""
    term_list = sorted((t for t in settings.dictionary()["terms"] if " " not in t), key=lambda t: -len(T.key(t)))
    lat = {t.lower() for t in settings.dictionary()["terms"]}
    seen = defaultdict(Counter)
    for r in rows:
        if r["cls"] != "термин":
            continue
        term = next((t for t in term_list if r["k"].startswith(T.key(t))), None)
        if not term:
            continue
        for src in SOURCES:
            got = r["got"].get(src)
            if not got or r["right"].get(src) or " " in got.strip():
                continue
            w = got.strip(".,!?«»—").lower().replace("ё", "е")
            if len(w) < MIN_STEM or re.search(r"\d", w) or T.known_russian(w) or w in lat:
                continue
            seen[w][term] += 1
            st = stem(w)
            if len(st) >= WILD_STEM and st != w and not russian_like(st):
                seen[st + "*"][term] += 1
    rules = []
    for s, c in seen.items():
        if len(c) == 1:
            term, n = c.most_common(1)[0]
            rules.append({"from": s, "to": term, "n": n})
    return sorted(rules, key=lambda r: -r["n"])


def merged(d: gold.Doc, extra: list[dict]) -> list[dict]:
    opts = gold.options(d.doc)
    segs = ensemble.combine(json.loads(json.dumps(d.st)), opts)
    comp = settings.rules() + [(settings._rule(r["from"]), r["to"]) for r in extra]
    comp.sort(key=lambda x: -len(x[0].pattern))
    store.apply_rules_to_segments(segs, comp if opts.dictionary or extra else [])
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--learn")
    ap.add_argument("--out")
    ap.add_argument("--rules")
    ap.add_argument("--doc")
    ap.add_argument("--set")
    ap.add_argument("--library", action="store_true")
    a = ap.parse_args()
    if a.learn:
        rules = learn(json.loads(Path(a.learn).read_text("utf-8")))
        Path(a.out).write_text(json.dumps(rules, ensure_ascii=False, indent=1), "utf-8")
        print(f"{len(rules)} правил -> {a.out}")
        for r in rules:
            print(f"  {r['from']:22s} -> {r['to']} ({r['n']})")
        return
    rules = json.loads(Path(a.rules).read_text("utf-8"))
    docs = [a.doc] if a.doc else [p.name for p in sorted(LIBRARY_DIR.iterdir()) if (p / "hypotheses.json").exists()]
    total = 0
    for doc_id in docs:
        try:
            d = gold.Doc(doc_id)
        except gold.NotFound:
            continue
        before = [w for s in merged(d, []) for w in s["words"]]
        after = [w for s in merged(d, rules) for w in s["words"]]
        if a.set:
            man = json.loads((Path(a.set) / "manifest.json").read_text("utf-8"))
            refs = DE.ref_words(man)
            for name, ws in (("без правил", before), ("с правилами", after)):
                r = DE.score(ws, refs, man)
                print(f"{name:12s} WER {r['wer']:.2%}, терминов верно {r['cls_n']['термин'] - r['cls_e']['термин']}"
                      f"/{r['cls_n']['термин']}, лишней латиницы {sum(r['false_latin'].values())}")
        elif d.doc.get("gold"):
            d.set_reference()
            for name, ws in (("без правил", before), ("с правилами", after)):
                e = sum(gold.errors(d.ref[ci], gold.norm([w["w"] for w in x]))
                        for ci, x in d.by_chunk(ws).items() if ci in d.ref)
                print(f"{name:12s} эталон: ошибок {e}")
        a_w, b_w = [w["w"] for w in before], [w["w"] for w in after]
        ch = [(a_w[i1:i2], b_w[j1:j2]) for t, i1, i2, j1, j2 in
              difflib.SequenceMatcher(a=a_w, b=b_w, autojunk=False).get_opcodes() if t != "equal"]
        total += len(ch)
        if not a.set:
            print(f"{d.id[-6:]}: изменений {len(ch)}  " + "; ".join(f"{' '.join(x)} -> {' '.join(y)}" for x, y in ch[:15]))
    print(f"всего изменений {total}")


if __name__ == "__main__":
    main()
