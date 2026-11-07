"""Punctuation and capitals on the synthetic dialogue set (lab experiment 19): the merge's own
marks (taken from the recognizers' spellings) against a punctuation tagger run on the merged
words (RUPunct/RUPunct_big, CPU). The tagger only labels each word with "mark after it + case",
so the words cannot change; Latin words and numbers keep their own spelling. Scored on the words
that match the reference: F1 per mark after a word (. , ?) and the share of right capitals.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/punct_lab.py --doc <id> --set %WS_REAL_HOME%/tmp/dialog-set/dev
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("HF_HOME", str(REAL_HOME / "hf-cache"))

from whisper_studio import gold  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dialog_eval as DE  # noqa: E402

MODEL = "RUPunct/RUPunct_big"
MARK = {"PERIOD": ".", "COMMA": ",", "QUESTION": "?", "TIRE": " —", "VOSKL": "!", "DVOETOCHIE": ":",
        "PERIODCOMMA": ";", "DEFIS": "-", "MNOGOTOCHIE": "…", "QUESTIONVOSKL": "?!", "O": ""}


def mark_class(w: str) -> str:
    """The mark after a word as one of . , ? or '' (! … = end, — : ; = comma)."""
    tail = re.search(r"[.,?!…:;—\-]*[»\"]*$", w.strip()).group(0)
    if "?" in tail:
        return "?"
    if any(c in tail for c in ".!…"):
        return "."
    if any(c in tail for c in ",:;—"):
        return ","
    return ""


class Tagger:
    def __init__(self):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tk = AutoTokenizer.from_pretrained(MODEL, strip_accents=False, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(MODEL).eval()
        self.labels = self.model.config.id2label

    def __call__(self, words: list[str]) -> list[str]:
        """Words with the tagger's marks and case; the letters of every word stay as they were."""
        bare = [re.sub(r"[^\w\-'%€$]+", "", w) or w for w in words]
        cyr = [bool(T.CYRILLIC.search(b)) and not re.search(r"[a-zA-ZäöüÄÖÜß]", b) for b in bare]
        inp = [b.lower() if c else b for b, c in zip(bare, cyr)]
        out = []
        for a in range(0, len(inp), 200):
            part = inp[a:a + 200]
            enc = self.tk(part, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
            with self.torch.inference_mode():
                pred = self.model(**enc).logits[0].argmax(-1).tolist()
            first = {}
            for k, wid in enumerate(enc.word_ids()):
                if wid is not None and wid not in first:
                    first[wid] = self.labels[pred[k]]
            for k, w in enumerate(part):
                lab = first.get(k, "LOWER_O")
                case, _, mark = lab.partition("_")
                if mark.startswith("TOTAL_"):
                    mark = mark[6:]
                src = bare[a + k]
                if cyr[a + k]:
                    w = w.upper() if lab.startswith("UPPER_TOTAL") else (w.capitalize() if case == "UPPER" else w)
                else:
                    w = src  # Latin terms and numbers keep their own spelling
                out.append(w + MARK.get(mark, ""))
        return out


def tag_segments(words: list[dict], tagger: Tagger, by: str) -> list[dict]:
    """Run the tagger over each segment (or speaker turn), keep everything but the spelling."""
    groups, cur, key = [], [], None
    for w in words:
        k = w.get(by)
        if cur and k != key:
            groups.append(cur)
            cur = []
        cur.append(w)
        key = k
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        new = tagger([w["w"] for w in g])
        out += [dict(w, w=n) for w, n in zip(g, new)]
    return out


def evaluate(words: list[dict], man: dict) -> dict:
    """Marks and capitals on the words that match the reference."""
    lines = sorted(man["lines"], key=lambda x: x["start"])
    ref = []
    for ln in lines:
        ws = ln["text"].split()
        keys, groups = gold.norm_groups(ws)
        for k, g in zip(keys, groups):
            src = ws[g[-1]]
            ref.append(dict(k=k, ok={k}, mark=mark_class(src), cap=ws[g[0]][:1].isupper(),
                            cyr=bool(T.CYRILLIC.search(ws[g[0]])), scene=ln["scene"]))
    hk, hg = gold.norm_groups([w["w"] for w in words])
    tp, fp, fn = Counter(), Counter(), Counter()
    cap_ok = cap_n = 0
    for i, j in DE.align(ref, hk):
        if i is None or j is None or hk[j] not in ref[i]["ok"]:
            continue
        r = ref[i]
        h = mark_class(words[hg[j][-1]]["w"])
        if h == r["mark"]:
            if h:
                tp[h] += 1
        else:
            if h:
                fp[h] += 1
            if r["mark"]:
                fn[r["mark"]] += 1
        if r["cyr"]:
            cap_n += 1
            cap_ok += words[hg[j][0]]["w"][:1].isupper() == r["cap"]
    f1 = {}
    for m in (".", ",", "?"):
        p = tp[m] / max(tp[m] + fp[m], 1)
        rc = tp[m] / max(tp[m] + fn[m], 1)
        f1[m] = 2 * p * rc / max(p + rc, 1e-9)
    return dict(f1=f1, cap=cap_ok / max(cap_n, 1), n=sum(tp.values()) + sum(fn.values()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--set", required=True)
    a = ap.parse_args()
    man = json.loads((Path(a.set) / "manifest.json").read_text("utf-8"))
    d = gold.Doc(a.doc)
    words = DE.merged_words(d.doc, d.st)
    app = [dict(w, seg=k) for k, s in enumerate(d.doc["segments"]) for w in s.get("words") or []]
    tagger = Tagger()
    # the ceiling: the tagger on the reference words themselves, one speaker turn at a time
    ref_words = [dict(w=w, line=f"{ln['scene']}-{ln['i']}") for ln in sorted(man["lines"], key=lambda x: x["start"])
                 for w in re.sub(r"(\S+)\|\S+", r"\1", ln["text"]).split()]
    rows = {
        "эталон, разметчик (потолок)": evaluate(tag_segments(ref_words, tagger, "line"), man),
        "приложение как есть": evaluate(app, man),
        "приложение + разметчик": evaluate(tag_segments(app, tagger, "seg"), man),
        "слияние + разметчик по спикеру": evaluate(tag_segments(words, tagger, "spk"), man),
    }
    print(f"{'вариант':34s} {'F1 .':>6s} {'F1 ,':>6s} {'F1 ?':>6s} {'регистр':>8s}")
    for k, r in rows.items():
        print(f"{k:34s} {r['f1']['.']:6.2f} {r['f1'][',']:6.2f} {r['f1']['?']:6.2f} {r['cap']:8.1%}")


if __name__ == "__main__":
    main()
