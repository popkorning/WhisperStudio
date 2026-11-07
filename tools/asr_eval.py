"""How good is the ensemble, and how much does each member add? Works on stored hypotheses
(hypotheses.json in the document folder), so nothing is recognized again and no GPU is needed.
The evaluation itself is whisper_studio/gold.py (the app runs the same on its gold documents).

Two modes:

    ws-python tools/asr_eval.py                       # no reference: how alike the members are
    ws-python tools/asr_eval.py --ref 9a8b7c          # a transcript checked by hand = the reference
    ... --ref 9a8b7c:0-300 --ref 0f1e2d               # only seconds 0-300 of the first one were checked
    ... --tune                                        # search member weights on the references
    ... --doc 5f4e3d --doc 0f1e2d                     # which documents (default: every "max" one)
    ... --json                                        # the result as JSON (the app's gold/latest.json)

(ws-python = the venv's python with PYTHONPATH=<repo>.)

Without a reference only agreement can be measured: pairwise WER between members and each
member against the merge of the others (leave-one-out). Members that agree with each other but
not with the rest make the same mistakes and outvote the others; that is a hint, not a WER.

A reference counts only where the owner listened (docs/GOLD.md): in a gold document only the
chunks covered by segments marked verified; any other document passed as --ref counts as
checked as a whole (or within the span). Chunks with "???" (not audible) are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import gold  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402


def parse_ref(s: str) -> tuple[str, tuple[float, float] | None]:
    if ":" in s:
        i, span = s.split(":", 1)
        a, b = span.split("-")
        return i, (float(a), float(b))
    return s, None


def load(i: str, span=None) -> gold.Doc:
    try:
        return gold.Doc(i, span)
    except gold.NotFound as e:
        sys.exit(str(e))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", default=[], help="ID[:from-to] транскрипта, проверенного на слух")
    ap.add_argument("--doc", action="append", default=[], help="документы для режима без эталона")
    ap.add_argument("--tune", action="store_true", help="подобрать веса участников по эталонам")
    ap.add_argument("--json", action="store_true", help="вывести результат в JSON")
    a = ap.parse_args()
    if a.ref:
        docs = []
        for r in a.ref:
            d = load(*parse_ref(r))
            d.set_reference()
            docs.append(d)
        res = gold.evaluate(docs, a.tune)
        print(json.dumps(res, ensure_ascii=False, indent=1) if a.json else gold.report_text(res))
        return
    ids = a.doc or [p.parent.name for p in sorted(LIBRARY_DIR.glob("*/hypotheses.json"))]
    seen = {}
    for i in ids:
        d = load(i)
        seen[d.doc.get("source_path") or d.id] = d  # newest run of the same recording wins (ids sort by time)
    docs = list(seen.values())
    ag = gold.agreement(docs)
    if a.json:
        print(json.dumps(ag, ensure_ascii=False, indent=1))
        return
    print("Документы:", ", ".join(f"{d.id[-6:]} ({(d.doc.get('duration') or 0) / 60:.0f} мин)" for d in docs))
    print(gold.agreement_text(ag))


if __name__ == "__main__":
    main()
