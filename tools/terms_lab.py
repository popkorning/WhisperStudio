"""Term recovery settings on two yardsticks at once (lab experiment 14): the synthetic term set
(tools/make_term_audio.py; how many terms come out right, how much Latin is written where no
term was) and the gold (merge errors: a looser setting must not break real speech). CPU only,
works on stored hypotheses: the merge is rebuilt with the current code and the given settings.

Both documents must be in one library: the lab home holds the term set and a junction to the
gold's folder.

Neither yardstick has real speech that only resembles a term: --library shows, for every real
recording with hypotheses, the words the setting changes (run with the real home). Experiment
14 rejected two settings that won on both yardsticks and inserted wrong terms there.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/terms_lab.py --doc 3c4b5a --set %WS_REAL_HOME%/tmp/term-set --ref a1b2c3
    ... --param terms.SIM=0.68 --param terms.MIN_TERM=4     one setting (module.NAME=value)
    ... --grid terms.SIM=0.64,0.68,0.72                     each value in turn
    ws-python tools/terms_lab.py --library --param terms.SIM_STRONG=0.78
"""
from __future__ import annotations

import argparse
import difflib
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, settings, store  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from term_set_eval import score  # noqa: E402


def setp(spec: str) -> tuple[object, str, object]:
    """'terms.SIM=0.68' -> sets whisper_studio.terms.SIM; returns (module, name, old value)."""
    name, val = spec.split("=", 1)
    mod_name, attr = name.rsplit(".", 1)
    mod = importlib.import_module(f"whisper_studio.{mod_name}")
    old = getattr(mod, attr)
    setattr(mod, attr, type(old)(json.loads(val)) if not isinstance(old, str) else val)
    return mod, attr, old


def merged_words(d: gold.Doc) -> list[str]:
    opts = gold.options(d.doc)
    segs = ensemble.combine(json.loads(json.dumps(d.st)), opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    return [w["w"] for s in segs for w in s["words"]]


def library(params: list[str]) -> None:
    """Words the settings change in every recording with stored hypotheses."""
    total = 0
    for p in sorted(LIBRARY_DIR.iterdir()):
        if not (p / "hypotheses.json").exists():
            continue
        try:
            d = gold.Doc(p.name)
        except gold.NotFound:
            continue
        a = merged_words(d)
        undo = [setp(x) for x in params]
        b = merged_words(d)
        for mod, attr, old in undo:
            setattr(mod, attr, old)
        ch = [(a[i1:i2], b[j1:j2]) for t, i1, i2, j1, j2 in
              difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if t != "equal"]
        total += len(ch)
        print(f"{d.id[-6:]}: слов {len(a)}, изменений {len(ch)}", flush=True)
        for x, y in ch:
            print(f"    {' '.join(x):40s} -> {' '.join(y)}")
    print(f"всего изменений {total}")


def measure(term_doc: gold.Doc, items: list[dict], ref: gold.Doc) -> tuple[dict, int]:
    opts = gold.options(term_doc.doc)
    segs = ensemble.combine(json.loads(json.dumps(term_doc.st)), opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    r = score([w for s in segs for w in s["words"]], items)
    hyp, _ = ref.merged(gold.options(ref.doc))
    return r, gold.wer(ref.ref, hyp, ref.chunks)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", help="the transcribed term set")
    ap.add_argument("--set")
    ap.add_argument("--ref", help="gold document")
    ap.add_argument("--library", action="store_true", help="what --param changes in real recordings")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--grid", help="module.NAME=v1,v2,...")
    ap.add_argument("--show", type=int, default=0, help="missed terms to print")
    a = ap.parse_args()
    if a.library:
        return library(a.param)
    if not (a.doc and a.set and a.ref):
        ap.error("--doc, --set and --ref (or --library)")
    items = json.loads((Path(a.set) / "manifest.json").read_text("utf-8"))
    term_doc = gold.Doc(a.doc)
    ref = gold.Doc(a.ref)
    ref.set_reference()
    for p in a.param:
        setp(p)
    runs = [None]
    if a.grid:
        name, vals = a.grid.split("=", 1)
        runs = [f"{name}={v}" for v in vals.split(",")]
    for run in runs:
        undo = setp(run) if run else None
        r, e = measure(term_doc, items, ref)
        print(f"{run or ', '.join(a.param) or 'как есть':34s} терминов {r['terms_right']:3d}/{r['sentences']}"
              f"  лишней латиницы {r['false_latin']:3d}  WER фраз {r['wer']:.1%}  | эталон: ошибок {e}", flush=True)
        for t, heard in r["missed"][:a.show]:
            print(f"    {t:28s} <- {heard}")
        if undo:
            setattr(undo[0], undo[1], undo[2])


if __name__ == "__main__":
    main()
