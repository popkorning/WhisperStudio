"""Speakers by meaning with a small local LLM (lab experiment 20). The LLM writes nothing: for
every phrase it sees the neighbouring phrases labelled A / B and the phrase itself marked [?],
and the next-token probabilities of "A" and "B" say who speaks it. The same question is asked
with the letters swapped (A <-> B in the whole window), so a liking for one letter cancels out.
A phrase changes speaker only where the model is sure (and, if asked, only short ones). Two
speakers per conversation: a synthetic set's scenes are separate conversations.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-lab-home
    ws-python tools/speaker_llm.py --doc a1b2c3 --model %WS_REAL_HOME%/models-lab/lfm3-1.7b
    ws-python tools/speaker_llm.py --doc <id> --set %WS_REAL_HOME%/tmp/dialog-set/dev --model ...
    ... --nf4          4B models in 4 bits      ... --cache f.json   reuse the scores
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import gold  # noqa: E402

BEFORE, AFTER = 6, 2   # phrases of context
GOLD_FIX = {"g15": "S2", "g42": "S2", "g46": "S2", "g33": "S1", "g38": "S1", "g50": "S1"}  # docs/DECISIONS.md
PROMPT = ("Ниже кусок расшифровки телефонного разговора двух людей, A и B. Реплики идут по порядку. "
          "Одна реплика отмечена [?]: кто её говорит? Подумай, кому отвечают, кто задал вопрос, "
          "кто продолжает свою мысль. Ответь одной буквой: A или B.\n\n{window}\n\nКто говорит реплику [?]?")


class Judge:
    def __init__(self, path: str, nf4: bool):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from proofread_lab import Model

        self.m = Model(path, "cuda", think=False, nf4=nf4)
        tok = self.m.tok
        self.ids = {x: tok.encode(x, add_special_tokens=False)[0] for x in ("A", "B")}

    def p_a(self, window: str) -> float:
        tok, torch = self.m.tok, self.m.torch
        msgs = [{"role": "user", "content": PROMPT.format(window=window)}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                                      enable_thinking=False).to("cuda")
        with torch.inference_mode():
            logits = self.m.m(**enc).logits[0, -1].float()
        la, lb = logits[self.ids["A"]].item(), logits[self.ids["B"]].item()
        return 1 / (1 + math.exp(lb - la))


def window(phr: list[dict], k: int, swap: bool) -> str:
    """Labels A/B by the phrase's current speaker (the first speaker of the conversation is A)."""
    first = phr[0]["spk"]
    rows = []
    for i in range(max(0, k - BEFORE), min(len(phr), k + AFTER + 1)):
        lab = "A" if phr[i]["spk"] == first else "B"
        if swap:
            lab = "B" if lab == "A" else "A"
        rows.append(f"[?]: {phr[i]['text']}" if i == k else f"{lab}: {phr[i]['text']}")
    return "\n".join(rows)


def conversations(doc: dict, man: dict | None) -> list[list[dict]]:
    """Phrases (segments) per conversation: the whole document, or one per synthetic scene."""
    segs = [dict(id=s.get("id"), text=s["text"], spk=s.get("speaker"), s=s["start"], e=s["end"],
                 words=s.get("words") or []) for s in doc["segments"] if s.get("text", "").strip()]
    if not man:
        return [segs]
    out = defaultdict(list)
    for s in segs:
        mid = (s["s"] + s["e"]) / 2
        sc = next((x["id"] for x in man["scenes"] if x["start"] <= mid < x["end"]), None)
        if sc:
            out[sc].append(s)
    return [v for _, v in sorted(out.items())]


def score_all(judge: Judge, convs: list[list[dict]]) -> list[list[float]]:
    """P(the phrase is the conversation's second speaker), averaged over both letter orders."""
    out = []
    for phr in convs:
        first = phr[0]["spk"]
        ps = []
        for k in range(len(phr)):
            pa = judge.p_a(window(phr, k, False))
            pb = 1 - judge.p_a(window(phr, k, True))  # swapped: "A" now means the second speaker
            p_first = (pa + pb) / 2
            ps.append(1 - p_first)
        out.append(ps)
        print(f"  разговор: {len(phr)} фраз, первый {first}", flush=True)
    return out


def relabel(convs, scores, tau: float, max_words: int) -> list[list[str]]:
    """New speaker per phrase: flipped where the model is sure the other one speaks."""
    out = []
    for phr, ps in zip(convs, scores):
        first = phr[0]["spk"]
        labs = [p["spk"] for p in phr]
        other = next((x for x in labs if x != first), "S2" if first != "S2" else "S1")
        new = []
        for p, lab, p2 in zip(phr, labs, ps):
            short = len(p["text"].split()) <= max_words
            want_second = p2 >= tau
            want_first = 1 - p2 >= tau
            if short and lab == first and want_second:
                new.append(other)
            elif short and lab != first and want_first:
                new.append(first)
            else:
                new.append(lab)
        out.append(new)
    return out


def wder_pairs(pairs: list[tuple[str, str]]) -> int:
    c = Counter(pairs)
    mp, used = {}, set()
    for (h, r), _ in sorted(c.items(), key=lambda x: -x[1]):
        if h not in mp and r not in used:
            mp[h] = r
            used.add(r)
    return sum(k for (h, r), k in c.items() if mp.get(h) != r)


def evaluate(convs, labels, man: dict | None) -> tuple[int, int]:
    bad = tot = 0
    for phr, labs in zip(convs, labels):
        pairs = []
        for p, lab in zip(phr, labs):
            for w in p["words"]:
                if man:
                    mid = (w["s"] + w["e"]) / 2
                    ln = next((x for x in man["lines"] if x["start"] <= mid < x["end"]), None)
                    if not ln:
                        continue
                    ref = ln["voice"]
                else:
                    ref = GOLD_FIX.get(p["id"], p["spk"])
                pairs.append((lab, ref))
        bad += wder_pairs(pairs)
        tot += len(pairs)
    return bad, tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--set")
    ap.add_argument("--model")
    ap.add_argument("--nf4", action="store_true")
    ap.add_argument("--cache")
    ap.add_argument("--skip-voices", default="", help="scenes whose two voices are one person (comma list of voice pairs a+b)")
    a = ap.parse_args()
    man = json.loads((Path(a.set) / "manifest.json").read_text("utf-8")) if a.set else None
    d = gold.Doc(a.doc)
    convs = conversations(d.doc, man)
    if man and a.skip_voices:
        bad_pairs = [set(x.split("+")) for x in a.skip_voices.split(",")]
        same = {s["id"] for s in man["scenes"] if set(s["voices"].values()) in bad_pairs}
        keep = []
        for phr in convs:
            mid = (phr[0]["s"] + phr[0]["e"]) / 2
            sc = next(x["id"] for x in man["scenes"] if x["start"] <= mid < x["end"])
            if sc not in same:
                keep.append(phr)
        convs = keep
    if a.cache and Path(a.cache).exists():
        scores = json.loads(Path(a.cache).read_text("utf-8"))
    else:
        judge = Judge(a.model, a.nf4)
        scores = score_all(judge, convs)
        if a.cache:
            Path(a.cache).write_text(json.dumps(scores), "utf-8")
    bad, tot = evaluate(convs, [[p["spk"] for p in phr] for phr in convs], man)
    print(f"как есть: WDER {bad / tot:.2%} ({bad} из {tot})")
    # how well the model alone tells the speakers apart: its own label on every phrase
    alone = [["S2x" if p2 >= 0.5 else "S1x" for p2 in ps] for ps in scores]
    b2, _ = evaluate(convs, alone, man)
    print(f"только LLM (каждая фраза по её ответу): WDER {b2 / tot:.2%} ({b2})")
    for max_words in (3, 6, 1000):
        for tau in (0.7, 0.8, 0.9, 0.95):
            labs = relabel(convs, scores, tau, max_words)
            changed = sum(x != p["spk"] for phr, ls in zip(convs, labs) for p, x in zip(phr, ls))
            b, _ = evaluate(convs, labs, man)
            print(f"  фразы до {max_words:4d} слов, порог {tau:.2f}: изменено {changed:3d}, WDER {b / tot:.2%} ({b})")


if __name__ == "__main__":
    main()
