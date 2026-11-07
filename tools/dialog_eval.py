"""Word-by-word score of a transcribed synthetic dialogue set (lab experiment 16,
tools/make_dialog_audio.py): every recognizer alone (from the document's hypotheses.json), the
merge rebuilt with the current code, and the transcript the app saved. Per word class (German
term, number, short reply, plain Russian, German speech) and per scene noise; for every
reference word, which recognizers wrote it right: the ceiling no choice among variants can pass
(nobody right) and the gap the merge leaves (somebody right, the merge wrong); Latin words
written where none was said; speakers (word diarization error).

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/dialog_eval.py --doc <id> --set %WS_REAL_HOME%/tmp/dialog-set/dev
    ... --param terms.SIM=0.68          the merge with a changed setting (module.NAME=value)
    ... --dump words.json               every reference word with what each source wrote there
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, settings, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from terms_lab import setp  # noqa: E402

CLASSES = ("термин", "число", "короткая реплика", "немецкая речь", "обычное")


def ref_words(man: dict) -> dict[str, list[dict]]:
    """Per scene: reference keys in time order, each with the keys that count as right."""
    german = {s["id"] for s in man["scenes"] if any(v.startswith("de") for v in s["voices"].values())}
    out = defaultdict(list)
    for ln in sorted(man["lines"], key=lambda x: x["start"]):
        alts = {}
        for a in ln["alts"]:
            first, second = a.split("|")
            ka, kb = gold.norm([first]), gold.norm([second])
            if len(ka) == len(kb) == 1:
                alts[ka[0]] = kb[0]
        words = ln["text"].split()
        keys, groups = gold.norm_groups(words)
        forms = list(ln.get("forms") or [])
        short = len(keys) <= 2
        for k, g in zip(keys, groups):
            src = " ".join(words[i] for i in g)
            if ln["scene"] in german:
                cls = "немецкая речь"
            elif T.is_latin(src.strip(".,!?«»")):
                cls = "термин"
            elif re.search(r"\d", k):
                cls = "число"
            elif short:
                cls = "короткая реплика"
            else:
                cls = "обычное"
            ok = {k} | ({alts[k]} if k in alts else set())
            form = None  # how the synthesizer said this term: as written or the Russian way
            base = re.sub(r"[^\wäöüß\-]|[а-яё]+$", "", src.lower())
            for fi, (term, how) in enumerate(forms if cls == "термин" else []):
                parts = [x.lower() for x in re.split(r"[\s]+", term)]
                if base in parts or any(base.startswith(x) for x in parts):
                    form = how
                    if base.startswith(parts[-1]):
                        forms.pop(fi)
                    break
            out[ln["scene"]].append(dict(k=k, ok=ok, cls=cls, w=src, spk=ln["voice"], form=form,
                                         line=f"{ln['scene']}-{ln['i']}"))
    return out


def align(ref: list[dict], hyp: list[str]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment of reference words (each with its right keys) and hypothesis keys:
    pairs of indexes, None on one side = deletion / insertion."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        ok = ref[i - 1]["ok"]
        row, prev = d[i], d[i - 1]
        for j in range(1, m + 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (hyp[j - 1] not in ok))
    out = []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (hyp[j - 1] not in ref[i - 1]["ok"]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def by_scene(words: list[dict], man: dict) -> dict[str, list[dict]]:
    out = defaultdict(list)
    # a recovered term of several words ("Berliner Straße") is one word of the merge
    words = [dict(w, w=x) for w in words for x in (w["w"].split() or [w["w"]])]
    for w in sorted(words, key=lambda x: x["s"]):
        mid = (w["s"] + w["e"]) / 2
        for s in man["scenes"]:
            if s["start"] <= mid < s["end"]:
                out[s["id"]].append(w)
                break
    return out


def member_words(st: dict, name: str) -> list[dict]:
    m = next((m for m in ensemble.MEMBERS if m["name"] == name), None)
    out = []
    for ci, h in enumerate(st["hyps"].get(name) or []):
        ch = st["chunks"][ci]
        if h is None or (m and not ensemble._eligible(m, ch)):
            continue
        out += ensemble._tokens(h, ch, name)
    return out


def merged_words(doc: dict, st: dict) -> list[dict]:
    opts = gold.options(doc)
    segs = ensemble.combine(json.loads(json.dumps(st)), opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, settings.rules())
    return [dict(w, spk=s.get("speaker")) for s in segs for w in s["words"]]


def score(words: list[dict], refs: dict[str, list[dict]], man: dict) -> dict:
    """Errors per class, per reference word right/wrong, Latin insertions, aligned speakers."""
    scenes = by_scene(words, man)
    snr = {s["id"]: s["snr"] for s in man["scenes"]}
    cls_n, cls_e, kinds, by_snr = Counter(), Counter(), Counter(), defaultdict(lambda: [0, 0])
    right, got, false_latin, pairs = {}, {}, Counter(), []
    for sid, ref in refs.items():
        hw = scenes.get(sid, [])
        keys, groups = gold.norm_groups([w["w"] for w in hw])
        for i, j in align(ref, keys):
            if i is None:
                kinds["вставка"] += 1
                src = " ".join(hw[x]["w"] for x in groups[j])
                if T.is_latin(src.strip(".,!?«»")):
                    false_latin[src] += 1
                by_snr[snr[sid]][1] += 1
                continue
            r = ref[i]
            cls_n[r["cls"]] += 1
            by_snr[snr[sid]][0] += 1
            key = (sid, i)
            if j is None:
                kinds["пропуск"] += 1
                cls_e[r["cls"]] += 1
                by_snr[snr[sid]][1] += 1
                right[key], got[key] = False, None
                continue
            src = " ".join(hw[x]["w"] for x in groups[j])
            ok = keys[j] in r["ok"]
            right[key], got[key] = ok, src
            if not ok:
                kinds["замена"] += 1
                cls_e[r["cls"]] += 1
                by_snr[snr[sid]][1] += 1
                if T.is_latin(src.strip(".,!?«»")) and r["cls"] != "термин":
                    false_latin[src] += 1
            spk = hw[groups[j][0]].get("spk")
            if spk is not None:
                pairs.append((spk, r["spk"], sid))
    n = sum(cls_n.values())
    e = sum(kinds.values())
    return dict(n=n, errors=e, wer=e / max(n, 1), kinds=kinds, cls_n=cls_n, cls_e=cls_e, by_snr=dict(by_snr),
                right=right, got=got, false_latin=false_latin, pairs=pairs)


def wder(pairs: list[tuple[str, str]]) -> tuple[int, int, dict]:
    """Aligned words with a wrong speaker after the best one-to-one map of labels to voices."""
    c = Counter((a, b) for a, b, *_ in pairs)
    labels, voices = {p[0] for p in pairs}, {p[1] for p in pairs}
    mapping, used = {}, set()
    for (lab, v), _ in sorted(c.items(), key=lambda x: -x[1]):
        if lab not in mapping and v not in used:
            mapping[lab] = v
            used.add(v)
    bad = sum(k for (lab, v), k in c.items() if mapping.get(lab) != v)
    return bad, len(pairs), dict(mapping=mapping, labels=len(labels), voices=len(voices))


def pct(a: int, b: int) -> str:
    return f"{a / b:.1%}" if b else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--dump")
    ap.add_argument("--only-merge", action="store_true", help="skip the members (quick setting sweeps)")
    a = ap.parse_args()
    man = json.loads((Path(a.set) / "manifest.json").read_text("utf-8"))
    refs = ref_words(man)
    d = gold.Doc(a.doc)
    doc, st = d.doc, d.st  # chunk languages assigned as the pipeline does
    for p in a.param:
        setp(p)

    sources = {}
    if not a.only_merge:
        for name in st["hyps"]:
            sources[name] = member_words(st, name)
        sources["приложение"] = [dict(w, spk=s.get("speaker")) for s in doc["segments"] for w in s.get("words") or []]
    sources["слияние"] = merged_words(doc, st)
    res = {name: score(ws, refs, man) for name, ws in sources.items()}

    print(f"{'источник':14s} {'WER':>6s} {'зам':>5s} {'проп':>5s} {'вст':>5s} | "
          + " ".join(f"{c[:8]:>9s}" for c in CLASSES) + " | лишн. лат.")
    for name, r in res.items():
        cells = " ".join(f"{pct(r['cls_e'][c], r['cls_n'][c]):>9s}" for c in CLASSES)
        print(f"{name:14s} {r['wer']:6.1%} {r['kinds']['замена']:5d} {r['kinds']['пропуск']:5d} "
              f"{r['kinds']['вставка']:5d} | {cells} | {sum(r['false_latin'].values())}")
    n_cls = res["слияние"]["cls_n"]
    print("слов: " + ", ".join(f"{c} {n_cls[c]}" for c in CLASSES) + f"; всего {res['слияние']['n']}")

    m = res["слияние"]
    print("по шуму (SNR дБ: WER слияния): " + ", ".join(
        f"{k:g}: {v[1] / max(v[0], 1):.1%}" for k, v in sorted(m["by_snr"].items())))
    if "приложение" in res and res["приложение"]["pairs"]:
        bad, tot, info = wder(res["приложение"]["pairs"])
        print(f"спикеры (приложение): WDER {pct(bad, tot)} ({bad} из {tot}), "
              f"меток {info['labels']} на {info['voices']} голосов, {info['mapping']}")

        by_sc = defaultdict(list)
        for x in res["приложение"]["pairs"]:
            by_sc[x[2]].append(x)
        sc_bad = sum(wder(v)[0] for v in by_sc.values())
        print(f"  по сценам (каждая сцена — свой разговор): WDER {pct(sc_bad, tot)} ({sc_bad} из {tot}); "
              + ", ".join(f"{k} {wder(v)[0]}" for k, v in sorted(by_sc.items()) if wder(v)[0]))
        conf = Counter((x[0], x[1]) for x in res["приложение"]["pairs"])
        print("  метка × голос: " + ", ".join(f"{a}/{b} {k}" for (a, b), k in sorted(conf.items())))
    for name in [x for x in ("large-v3", "слияние") if x in res]:
        by = defaultdict(lambda: [0, 0])
        for (sid, i), ok in res[name]["right"].items():
            f = refs[sid][i].get("form")
            if f:
                by[f][0] += ok
                by[f][1] += 1
        print(f"термины {name} по произношению синтеза: "
              + ", ".join(f"{f} {v[0]}/{v[1]}" for f, v in sorted(by.items())))

    members = [x for x in res if x not in ("приложение", "слияние")]
    if members:
        keys = list(m["right"])
        nobody = [k for k in keys if not any(res[x]["right"].get(k) for x in members)]
        gap = [k for k in keys if not m["right"][k] and any(res[x]["right"].get(k) for x in members)]
        only = Counter()
        for k in keys:
            hits = [x for x in members if res[x]["right"].get(k)]
            if len(hits) == 1:
                only[hits[0]] += 1
        print(f"потолок: никто не написал верно {len(nobody)} из {len(keys)} слов ({pct(len(nobody), len(keys))}); "
              f"слияние ошиблось там, где кто-то был прав: {len(gap)}; вставок у слияния {m['kinds']['вставка']}")
        cls_nobody = Counter(refs[k[0]][k[1]]["cls"] for k in nobody)
        cls_gap = Counter(refs[k[0]][k[1]]["cls"] for k in gap)
        print("  никто: " + ", ".join(f"{c} {cls_nobody[c]}" for c in CLASSES if cls_nobody[c]))
        print("  зазор: " + ", ".join(f"{c} {cls_gap[c]}" for c in CLASSES if cls_gap[c]))
        print("  верно только у одной модели: " + ", ".join(f"{x} {v}" for x, v in only.most_common()))
    print("лишняя латиница слияния: " + ", ".join(f"{w}×{k}" for w, k in m["false_latin"].most_common(12)))

    if a.dump:
        rows = []
        for sid, ref in refs.items():
            for i, r in enumerate(ref):
                rows.append(dict(scene=sid, i=i, k=r["k"], w=r["w"], cls=r["cls"], line=r["line"],
                                 got={x: res[x]["got"].get((sid, i)) for x in res},
                                 right={x: res[x]["right"].get((sid, i)) for x in res}))
        Path(a.dump).write_text(json.dumps(rows, ensure_ascii=False, indent=0), "utf-8")
        print(f"-> {a.dump}")


if __name__ == "__main__":
    main()
