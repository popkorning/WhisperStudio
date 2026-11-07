"""Does a bigger ensemble make fewer mistakes? Merge errors on the gold for every subset of members
(lab experiment 12). CPU only: works on stored hypotheses (hypotheses.json) plus the hypotheses
of candidate models from tools/member_lab.py (HOME/tmp/member-lab/<doc id>/<name>.json).

For every subset size k it prints the merge errors (best / median / worst subset) and the
"nobody heard it" floor: reference words no member of the subset got right, which no vote,
forest or LLM choosing among the members' words can fix.

    ws-python tools/ensemble_curve.py --ref a1b2c3              # all subsets of all members
    ws-python tools/ensemble_curve.py --ref a1b2c3 --max 4000   # sample subsets when there are many
    ws-python tools/ensemble_curve.py --ref a1b2c3 --only-base  # without lab members
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, models  # noqa: E402
from whisper_studio.config import HOME  # noqa: E402

LAB = HOME / "tmp" / "member-lab"


def lab_members(doc: gold.Doc) -> list[dict]:
    """Candidate members: LAB/members.json (name, lang, only, weight) with hypotheses for this doc."""
    cfg = json.loads((LAB / "members.json").read_text("utf-8")) if (LAB / "members.json").exists() else []
    out = []
    for m in cfg:
        f = LAB / doc.id / f"{m['name']}.json"
        if not f.exists():
            continue
        hyps = json.loads(f.read_text("utf-8"))
        doc.st["hyps"][m["name"]] = [hyps.get(str(i)) for i in range(len(doc.st["chunks"]))]
        out.append(dict(dict(backend="lab", model=m["name"], weight=1.0, cost=1.0), **m))
    return out


def install(extra: list[dict]) -> None:
    """Let the ensemble see the lab members as installed ones."""
    names = {m["name"] for m in extra}
    ensemble.MEMBERS[:] = [m for m in ensemble.MEMBERS if m["name"] not in names] + extra
    find = models.find
    models.find = lambda key, _f=find: Path(".") if key in names else _f(key)


def covered(d: gold.Doc, hyps: dict[str, dict[int, list[str]]], names) -> int:
    """Reference words that no member in `names` got right."""
    miss = 0
    for ci in d.chunks:
        ref = d.ref[ci]
        ok = [False] * len(ref)
        for n in names:
            h = hyps[n].get(ci)
            if h is None:
                continue
            i = 0
            for r, w in gold.align(ref, h):
                if r is not None:
                    if r == w:
                        ok[i] = True
                    i += 1
        miss += ok.count(False)
    return miss


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--max", type=int, default=3000, help="subsets per size at most (sampled)")
    ap.add_argument("--only-base", action="store_true")
    ap.add_argument("--out", help="JSON with every subset")
    ap.add_argument("--each", action="store_true", help="each lab member added to the base alone")
    ap.add_argument("--lab-weight", type=float, help="voting weight of every lab member")
    ap.add_argument("--no-gap", action="store_true", help="lab members vote for words only, not for gaps")
    ap.add_argument("--witness", action="store_true",
                    help="lab members don't vote; their words are only evidence for the term search")
    a = ap.parse_args()
    random.seed(1)
    docs = []
    extra: dict[str, dict] = {}
    for r in a.ref:
        d = gold.Doc(r)
        d.set_reference()
        if not a.only_base:
            for m in lab_members(d):
                if a.lab_weight is not None:
                    m["weight"] = a.lab_weight
                extra[m["name"]] = m
        docs.append(d)
    install(list(extra.values()))
    if a.no_gap:
        ensemble.NO_GAP.update(extra)
    names = sorted({n for d in docs for n in d.members()}, key=[m["name"] for m in ensemble.MEMBERS].index)
    words = sum(len(r) for d in docs for r in d.ref.values())
    print(f"Участники ({len(names)}): {', '.join(names)}; слов эталона {words}")
    hyps = {d.id: {n: d.member(n) for n in names} for d in docs}
    t0 = time.monotonic()
    if a.each:
        base = [n for n in names if n not in extra]

        def score(s):
            off = [n for n in names if n not in s]
            e = sum(gold.wer(d.ref, d.merged(gold.options(d.doc, exclude=off))[0], d.chunks)[0] for d in docs)
            return e, sum(covered(d, hyps[d.id], s) for d in docs)

        e0, f0 = score(base)
        print(f"База ({len(base)}): ошибок слияния {e0}, никто не услышал {f0}")
        if a.witness:
            labs = [n for n in names if n in extra]
            ev0 = ensemble.EVIDENCE_ONLY
            for n in labs + ["all"]:
                ensemble.EVIDENCE_ONLY = ev0 + tuple(labs if n == "all" else [n])
                e, _ = score(base)
                print(f"  свидетель {n:24s} ошибок {e}", flush=True)
            return
        print(f"{'модель':24s} {'своих ошибок':>13s} {'база+она':>9s} {'никто не усл.':>14s}")
        for n in [n for n in names if n in extra]:
            ow = [gold.wer(d.ref, hyps[d.id][n], list(hyps[d.id][n])) for d in docs]
            own, nw = sum(x[0] for x in ow), sum(x[1] for x in ow)
            e, f = score(base + [n])
            print(f"{n:24s} {own:6d} ({own / max(nw, 1):5.1%}) {e:9d} {f:14d}", flush=True)
        e, f = score(names)
        print(f"{'все ' + str(len(names)):24s} {'':13s} {e:9d} {f:14d}")
        return
    rows = []
    for k in range(1, len(names) + 1):
        subs = list(itertools.combinations(names, k))
        if len(subs) > a.max:
            subs = random.sample(subs, a.max)
        for s in subs:
            off = [n for n in names if n not in s]
            e = n_miss = 0
            for d in docs:
                hyp, _ = d.merged(gold.options(d.doc, exclude=off))
                e += gold.wer(d.ref, hyp, d.chunks)[0]
                n_miss += covered(d, hyps[d.id], s)
            rows.append(dict(k=k, members=list(s), errors=e, floor=n_miss))
        ks = [r for r in rows if r["k"] == k]
        es, fs = [r["errors"] for r in ks], [r["floor"] for r in ks]
        best = min(ks, key=lambda r: r["errors"])
        print(f"k={k:2d}  наборов {len(ks):4d}  ошибки: лучший {min(es):3d}  медиана {statistics.median(es):5.1f}"
              f"  худший {max(es):3d} | никто не услышал: лучший {min(fs):3d}  медиана {statistics.median(fs):5.1f}"
              f"  | лучший набор: {', '.join(best['members'])}  ({time.monotonic() - t0:.0f} с)", flush=True)
    print("\nВклад участника (медиана ошибок наборов с ним минус без него, k ≥ 2):")
    for n in names:
        w = [r["errors"] for r in rows if r["k"] >= 2 and n in r["members"]]
        wo = [r["errors"] for r in rows if r["k"] >= 2 and n not in r["members"]]
        if w and wo:
            print(f"  {n:24s} {statistics.median(w) - statistics.median(wo):+6.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, ensure_ascii=False, indent=0), "utf-8")


if __name__ == "__main__":
    main()
