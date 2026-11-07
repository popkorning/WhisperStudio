"""Speaker quality of library documents (read only, CPU): WDER against a labelled reference and
a report that needs no labels (lab, docs/DECISIONS.md «Опыт 11»).

    ws-python tools/diarize_eval.py wder a1b2c3 --ref ref.json [--rerun] [--legacy a,b,t]
    ws-python tools/diarize_eval.py report a1b2c3 d4e5f6 | --two  [--rerun] [--legacy a,b,t]
    ws-python tools/diarize_eval.py make-ref a1b2c3 --fix g15=S2 g42=S2 --out ref.json
    ws-python tools/diarize_eval.py owner a1b2c3 --ref ref.json [--out owner.npz]
    ws-python tools/diarize_eval.py owner-check --two [--profile owner.npz]

Hypothesis: the speakers stored in the document, or with --rerun the speakers the current code
gives from the document's voices.npz (diarize.label, num_speakers from its options; --voices DIR
takes <DIR>/<id>.npz instead, e.g. voices computed anew). --legacy turns single changes of
Опыт 11 off: a = outlier clusters turn by turn, b = words outside every turn and in overlaps
by the segmentation's activity (only voices from onnxruntime have it), t = only in overlaps.
c lives in the voices themselves (WHISPER_STUDIO_DIAR_CONSTRAIN=1 when computing them).

owner: the owner's voice profile (default HOME/voices/owner.npz) from a reference;
owner-check: cosine of every speaker to it and whom diarize.label would name.

WDER (word diarization error rate): reference and hypothesis words are matched by text
(case, punctuation and ё ignored, difflib blocks) and time (midpoints within 1 s), the
hypothesis labels mapped to the reference ones by the best permutation; WDER = wrong / matched.
The reference is a document JSON (its segments' words and speakers): make-ref copies a library
document and gives the listed segments another speaker («only these are wrong»).

Report: word share per speaker, speaker switches per minute, switches inside one Whisper segment
(and among them those not after punctuation: mid-phrase), short replies («понял», «окей»,
«угу» - a Whisper segment of at most 3 such words): how many each speaker got and how many have
the speaker of the words just before (a reply usually answers the other one).
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import os
import re
import sys
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_studio import diarize as D  # noqa: E402

LIBRARY = Path(os.environ.get("WS_LIBRARY", str(REAL_HOME / "library")))
REPLY = {"угу", "ага", "да", "нет", "ну", "мгм", "ммм", "ладно", "хорошо", "понял", "понятно", "ясно",
         "окей", "ок", "так", "конечно", "давай", "все", "всё", "ja", "okay", "ok", "вот"}
END = (".", "?", "!", "…", ",", ";", ":")


def find(doc_id: str) -> Path:
    hits = [p for p in LIBRARY.iterdir() if p.name.endswith(doc_id)]
    if len(hits) != 1:
        raise SystemExit(f"{doc_id}: {len(hits)} documents")
    return hits[0]


def load(doc_id: str) -> tuple[Path, dict]:
    folder = find(doc_id)
    return folder, json.loads((folder / "doc.json").read_text("utf-8"))


def key(w: str) -> str:
    return re.sub(r"[^\w]", "", w.lower().replace("ё", "е"))


# ---------------------------------------------------------------- words

def _seg_words(segments: list[dict]) -> list[dict]:
    """The words of the stored segments with their speaker, one Whisper segment per rejoin piece."""
    out = []
    pieces = D.rejoin(copy.deepcopy(segments))
    spk = [w_spk for s in segments for w_spk in [s.get("speaker")] * len(s.get("words") or [])]
    i = 0
    for si, s in enumerate(pieces):
        for w in s.get("words") or []:
            sp = spk[i] if i < len(spk) else None
            i += 1
            if key(w["w"]):
                out.append(dict(w=w["w"], s=w["s"], e=w["e"], seg=si, spk=sp))
    return out


def stored(doc: dict) -> list[dict]:
    return _seg_words(doc["segments"])


def rerun(doc: dict, voices: Path, legacy: set[str] = frozenset(), num_speakers: int | None = -1
          ) -> tuple[list[dict], dict]:
    """The words with the speakers the current code gives from a voices cache."""
    extra: dict = {}
    turns, temb = D._load_cache(voices, extra)
    n = (doc.get("options") or {}).get("num_speakers") if num_speakers == -1 else num_speakers
    segs = D.rejoin(copy.deepcopy(doc["segments"]))
    out, info = D.label(segs, turns, temb, n, extra, legacy)
    return _seg_words(out), info


# ---------------------------------------------------------------- WDER

def match(ref: list[dict], hyp: list[dict], slack: float = 1.0) -> list[tuple[int, int]]:
    sm = difflib.SequenceMatcher(None, [key(w["w"]) for w in ref], [key(w["w"]) for w in hyp],
                                 autojunk=False)
    out = []
    for a, b, size in sm.get_matching_blocks():
        for k in range(size):
            r, h = ref[a + k], hyp[b + k]
            if abs((r["s"] + r["e"]) - (h["s"] + h["e"])) / 2 <= slack:
                out.append((a + k, b + k))
    return out


def best_map(pairs: list[tuple[str, str]]) -> dict:
    """Hypothesis label -> reference label maximizing agreement (one to one)."""
    from scipy.optimize import linear_sum_assignment

    R = sorted({r for r, _ in pairs if r is not None})
    H = sorted({h for _, h in pairs if h is not None}, key=str)
    if not R or not H:
        return {}
    m = np.zeros((len(H), len(R)))
    for r, h in pairs:
        if r is not None and h is not None:
            m[H.index(h), R.index(r)] += 1
    hi, ri = linear_sum_assignment(-m)
    return {H[i]: R[j] for i, j in zip(hi, ri)}


def wder(ref: list[dict], hyp: list[dict]) -> dict:
    idx = match(ref, hyp)
    pairs = [(ref[a]["spk"], hyp[b]["spk"]) for a, b in idx]
    mp = best_map(pairs)
    wrong = [(a, b) for a, b in idx if mp.get(hyp[b]["spk"]) != ref[a]["spk"]]
    per = {}
    for a, _b in wrong:
        per[ref[a]["spk"]] = per.get(ref[a]["spk"], 0) + 1
    return dict(ref_words=len(ref), matched=len(idx), wrong=len(wrong),
                wder=round(len(wrong) / max(len(idx), 1), 4), map=mp, wrong_by_ref=per,
                wrong_words=[(round(ref[a]["s"], 2), ref[a]["w"], ref[a]["spk"]) for a, _ in wrong])


# ---------------------------------------------------------------- no-label report

def report(words: list[dict], duration: float | None = None) -> dict:
    if not words:
        return dict(words=0)
    cnt: dict = {}
    for w in words:
        cnt[w["spk"]] = cnt.get(w["spk"], 0) + 1
    share = {k: round(v / len(words), 3) for k, v in sorted(cnt.items(), key=lambda x: -x[1])}
    sw = inside = mid = 0
    for a, b in zip(words, words[1:]):
        if a["spk"] != b["spk"]:
            sw += 1
            if a["seg"] == b["seg"]:
                inside += 1
                if not a["w"].rstrip().endswith(END):
                    mid += 1
    # short replies: a Whisper segment of <= 3 reply words
    segs: dict[int, list[dict]] = {}
    for w in words:
        segs.setdefault(w["seg"], []).append(w)
    replies: dict = {}
    same_prev = total = 0
    prev_last = None
    for si in sorted(segs):
        ws = segs[si]
        if len(ws) <= 3 and all(key(w["w"]) in REPLY for w in ws):
            sp = max({w["spk"] for w in ws}, key=lambda k: sum(w["spk"] == k for w in ws))
            replies[sp] = replies.get(sp, 0) + 1
            total += 1
            if prev_last is not None and prev_last == ws[0]["spk"]:
                same_prev += 1
        prev_last = ws[-1]["spk"]
    dur = duration or (words[-1]["e"] - words[0]["s"]) or 1.0
    return dict(words=len(words), share=share, switches=sw, sw_per_min=round(sw / (dur / 60), 1),
                switch_inside_seg=inside, switch_mid_phrase=mid, replies=replies,
                replies_after_same=same_prev, replies_total=total)


# ---------------------------------------------------------------- the owner's voice

OWNER_WORDS = {"понял", "окей"}


def owner_embedding(doc_id: str, ref: Path | None = None, speaker: str | None = None,
                    pure: float = 0.9) -> tuple[np.ndarray, dict]:
    """The owner's voice from a document: the mean turn embedding (voices.npz, duration-weighted)
    of the turns whose words are at least `pure` theirs. Their speaker: given, else the one of the
    reference (or the document) who says OWNER_WORDS most."""
    folder, doc = load(doc_id)
    words = stored(json.loads(ref.read_text("utf-8")) if ref else doc)
    if speaker is None:
        cnt: dict = {}
        for w in words:
            if key(w["w"]) in OWNER_WORDS:
                cnt[w["spk"]] = cnt.get(w["spk"], 0) + 1
        speaker = max(cnt, key=cnt.get)
    turns, temb = D._load_cache(folder / "voices.npz")
    acc, used, secs = 0, 0, 0.0
    for i, (s, e, _k) in enumerate(turns):
        if i not in temb:
            continue
        inside = [w for w in words if w["s"] < e and w["e"] > s]
        if not inside:
            continue
        mine = sum(w["spk"] == speaker for w in inside) / len(inside)
        if mine >= pure:
            acc = acc + temb[i] * (e - s)
            used += 1
            secs += e - s
    if not used:
        raise SystemExit(f"{doc_id}: у {speaker} нет чистых реплик")
    return D._unit(np.asarray(acc, np.float32)), dict(doc=doc_id, speaker=speaker, turns=used,
                                                        seconds=round(secs, 1))


def owner_scores(ids: list[str], profile: dict, legacy=frozenset()) -> list[dict]:
    """Per document: (cosine to the profile, share of words, times the speaker says the owner's name) of
    every final speaker (library voices, current code), and whom diarize.label names."""
    out = []
    for i in ids:
        folder, doc = load(i)
        if not (folder / "voices.npz").exists():
            continue
        extra: dict = {}
        turns, temb = D._load_cache(folder / "voices.npz", extra)
        segs = D.rejoin(copy.deepcopy(doc["segments"]))
        n = (doc.get("options") or {}).get("num_speakers")
        res, info = D.label(segs, turns, temb, n, extra, legacy, profile)
        final, _ = D.refine(turns, temb, n, legacy)
        words = _seg_words(res)
        said = D.says_name(res, profile["name"])
        cnt: dict = {}
        for w in words:
            cnt[w["spk"]] = cnt.get(w["spk"], 0) + 1
        sims = [round(float(v @ profile["emb"]), 3) for v in D.final_centroids(final, temb).values()]
        out.append(dict(id=i, sims=sorted(sims, reverse=True),
                        share={k: round(c / max(len(words), 1), 2) for k, c in cnt.items()}, says_name=said,
                        owner=info.get("owner"), title=doc.get("title", "")[:40]))
    return out


# ---------------------------------------------------------------- CLI

def make_ref(doc_id: str, fixes: list[str], out: Path) -> None:
    _folder, doc = load(doc_id)
    fx = dict(f.split("=", 1) for f in fixes)
    for s in doc["segments"]:
        if s["id"] in fx:
            s["speaker"] = fx.pop(s["id"])
    if fx:
        raise SystemExit(f"нет сегментов: {', '.join(fx)}")
    doc["reference"] = dict(source=doc["id"], fixes=fixes)
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), "utf-8")


def hypothesis(doc_id: str, args) -> tuple[dict, list[dict], dict | None]:
    folder, doc = load(doc_id)
    if not args.rerun:
        return doc, stored(doc), None
    v = Path(args.voices) / f"{doc_id}.npz" if args.voices else folder / "voices.npz"
    if not v.exists():
        raise SystemExit(f"{doc_id}: нет {v}")
    words, info = rerun(doc, v, set(filter(None, args.legacy.split(","))))
    return doc, words, info


def two_speaker_ids() -> list[str]:
    out = []
    for p in sorted(LIBRARY.iterdir()):
        f = p / "doc.json"
        if f.exists():
            d = json.loads(f.read_text("utf-8"))
            if (d.get("options") or {}).get("num_speakers") == 2 and (p / "voices.npz").exists():
                out.append(p.name[-6:])
    return out


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("wder", "report", "make-ref", "owner", "owner-check"))
    ap.add_argument("--speaker", help="owner: their speaker in the reference (default: by OWNER_WORDS)")
    ap.add_argument("--name", default=D.OWNER_NAME)
    ap.add_argument("--profile", help="owner-check: a profile other than HOME/voices/owner.npz")
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--ref")
    ap.add_argument("--fix", nargs="*", default=[])
    ap.add_argument("--out")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--voices", help="folder of <id>.npz voices instead of the library's")
    ap.add_argument("--legacy", default="", help="a,b: switch these changes off")
    ap.add_argument("--two", action="store_true", help="every two-speaker document with voices")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "make-ref":
        make_ref(a.ids[0], a.fix, Path(a.out))
        return
    if a.cmd == "owner":
        emb, meta = owner_embedding(a.ids[0], Path(a.ref) if a.ref else None, a.speaker)
        out = Path(a.out) if a.out else D.owner_path()
        D.save_owner_profile(out, emb, a.name, **meta)
        print(f"{out}: {meta}")
        return
    if a.cmd == "owner-check":
        prof = D.owner_profile(Path(a.profile) if a.profile else None)
        if prof is None:
            raise SystemExit("нет профиля владельца")
        for r in owner_scores(a.ids + (two_speaker_ids() if a.two else []), prof):
            print(r["id"], r["owner"] or "-", r["sims"], r["share"], r["says_name"], r["title"])
        return
    ids = a.ids + (two_speaker_ids() if a.two else [])
    for i in ids:
        doc, words, info = hypothesis(i, a)
        if a.cmd == "wder":
            ref = stored(json.loads(Path(a.ref).read_text("utf-8")))
            r = wder(ref, words)
        else:
            r = report(words, doc.get("duration"))
        r = dict(id=i, **r)
        if info:
            keep = ("raw_speakers", "merged", "speakers", "folded_whole", "by_activity", "owner")
            r["refine"] = {k: v for k, v in info.items() if k in keep}
        print(json.dumps(r, ensure_ascii=False) if a.json else
              f"{i}: " + ", ".join(f"{k}={v}" for k, v in r.items() if k not in ("id", "wrong_words")))
        if a.cmd == "wder" and not a.json:
            for w in r["wrong_words"]:
                print("   ", *w)


if __name__ == "__main__":
    main()
