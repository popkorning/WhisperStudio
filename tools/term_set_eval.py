"""How many German terms of the synthetic term set (tools/make_term_audio.py) come out right
(lab experiment 14). Reads a transcript of term-set.wav and the set's manifest: per sentence,
is the term there in Latin (a Russian case ending allowed), which Latin words were written that
are not the term (false terms), and the sentence's word errors.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/term_set_eval.py --doc <id> --set %WS_REAL_HOME%/tmp/term-set
    ... --hyps          also each ensemble member alone (from the document's hypotheses.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402


def keys(text: str) -> list[str]:
    return gold.norm(text.split())


def score(words: list[dict], items: list[dict]) -> dict:
    hit, wrong, err, n = 0, Counter(), 0, 0
    miss = []
    for it in items:
        ws = [w["w"] for w in words if it["start"] - 0.3 <= (w["s"] + w["e"]) / 2 <= it["end"] + 0.3]
        hyp, ref = keys(" ".join(ws)), keys(it["text"])
        e = gold.errors(ref, hyp)
        err, n = err + e, n + len(ref)
        term = keys(it["term"])
        joined = " ".join(hyp)
        if " ".join(term) in joined:
            hit += 1
        else:
            miss.append((it["term"], " ".join(ws)))
        for w in ws:
            k = keys(w)
            if k and T.is_latin(w) and k[0] not in term:
                wrong[w] += 1
    return dict(sentences=len(items), terms_right=hit, wer=round(err / max(n, 1), 4),
                false_latin=sum(wrong.values()), false_examples=wrong.most_common(8), missed=miss)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--hyps", action="store_true")
    ap.add_argument("--show", type=int, default=15, help="missed terms to print")
    a = ap.parse_args()
    items = json.loads((Path(a.set) / "manifest.json").read_text("utf-8"))
    doc = store.load(gold.find(a.doc))
    words = [w for s in doc["segments"] for w in s.get("words") or []]
    r = score(words, items)
    print(f"Слияние: терминов верно {r['terms_right']} из {r['sentences']}, WER фраз {r['wer']:.1%}, "
          f"лишней латиницы {r['false_latin']} {r['false_examples']}")
    for t, heard in r["missed"][:a.show]:
        print(f"  {t:28s} <- {heard}")
    if a.hyps:
        st = json.loads((gold.LIBRARY_DIR / doc["id"] / "hypotheses.json").read_text("utf-8"))
        for name, res in st["hyps"].items():
            ws = [t for ci, h in enumerate(res or []) if h for t in ensemble._tokens(h, st["chunks"][ci], name)]
            m = score(ws, items)
            print(f"  {name:14s} терминов {m['terms_right']:3d}, WER {m['wer']:.1%}, лишней латиницы {m['false_latin']}")


if __name__ == "__main__":
    main()
