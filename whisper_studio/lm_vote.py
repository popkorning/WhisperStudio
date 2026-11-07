"""Language-model vote on disputed slots of the ensemble merge (lab hypothesis H8, docs/DECISIONS.md).

Experimental, off by default: nothing in the app imports this module. tools/lm_vote_eval.py
measures it on the gold documents. Result so far (2026-10-31): the vote makes the merge worse
at any weight (Qwen3-0.6B-Base, RuModernBERT PLL); not used.

The idea: where the ASR members disagree (low vote margin in a slot of the confusion network),
a small language model scores the sentence with each variant the members produced (plus
"nothing here") and adds a weak extra vote. It only chooses among the members' variants, it never
writes a word of its own. Optionally project dictionary terms that sound like a variant join
the options (bonus experiment).

How it plugs in without touching ensemble.py: ensemble.combine calls ensemble._vote once per
slot, in an order that does not depend on the votes. `recording()` wraps _vote to collect the
slots with their options and vote scores; `voting(lm)` wraps it again to add the language
model's vote as extra pseudo-members ("lm0", "lm1", ...) whose weight is w x the LM posterior
of the option they stand for. The decision itself stays ensemble._vote's.

Heavy imports (torch, transformers) only inside the scorers.
"""
from __future__ import annotations

import contextlib
import math
import time

import numpy as np

from . import ensemble
from .asr import text as T

DISPUTE = 0.35     # slot is disputed if (best - second) / (best + second) < this
LEFT, RIGHT = 20, 8  # context words around the slot
PREFIX = "Расшифровка телефонного разговора (русская речь, иногда немецкие слова):\n"


# ---------------------------------------------------------------- slots of a merge

def _scores(slot, present: dict, scope: dict | None, mids: dict | None, speech) -> dict:
    """Vote score of every option of a slot: word key -> score, None -> "nothing here".
    A simplified copy of ensemble._vote (cross-script votes left out), only to find disputes."""
    scope = scope or {}
    opts: dict = {}
    for name, t in slot.toks.items():
        w = present.get(name, 0.0)
        if w <= 0:
            continue
        p = t["p"] if t.get("p") is not None else 0.7
        opts[t["k"]] = opts.get(t["k"], 0.0) + w * (0.55 + 0.45 * p)
    if not opts:
        return {}
    top = max(opts, key=opts.get)
    lat = T.is_latin(top)
    judges = {n: w for n, w in present.items() if ensemble._compat(scope.get(n), lat)}
    t0 = slot.t
    abstain = set()
    if mids is not None and t0 is not None:
        heard = [n for n in present if n in slot.toks or (len(mids.get(n, ())) and
                                                         np.abs(mids[n] - t0).min() <= ensemble.NEAR)]
        if len(heard) >= 2:
            abstain = {n for n in present if n not in heard}
    gap = ensemble.GAP_VOTE * sum(w for n, w in judges.items() if n not in slot.toks and n not in abstain)
    if gap > 0:
        opts[None] = gap
    return opts


class _Rec:
    def __init__(self):
        self.slots: list[dict] = []
        self.n = 0


@contextlib.contextmanager
def _patched(fn):
    orig = ensemble._vote
    ensemble._vote = fn
    try:
        yield orig
    finally:
        ensemble._vote = orig


@contextlib.contextmanager
def recording():
    """with recording() as rec: ensemble.combine(...) -> rec.slots, one dict per slot in order:
    t, options {key: score}, words {key: surface word}, win (key or None), margin, sig."""
    rec = _Rec()
    orig = ensemble._vote

    def vote(slot, present, order, scope=None, order_lat=None, order_cyr=None, mids=None, speech=None):
        win = orig(slot, present, order, scope, order_lat, order_cyr, mids, speech)
        sc = _scores(slot, present, scope, mids, speech)
        ranked = sorted(sc.values(), reverse=True)
        margin = (ranked[0] - ranked[1]) / (ranked[0] + ranked[1]) if len(ranked) > 1 else 1.0
        words = {}
        for n, t in slot.toks.items():
            if n in present:
                words.setdefault(t["k"], T.PUNCT_TAIL.sub("", t["w"]).lstrip("«\"(") or t["w"])
        rec.slots.append(dict(t=slot.t, options=sc, words=words, margin=margin,
                              win=win["tok"]["k"] if win else None,
                              surf=win["tok"]["w"] if win else None,
                              sig=tuple(sorted(f"{n}:{t['k']}" for n, t in slot.toks.items()))))
        return win

    with _patched(vote):
        yield rec


@contextlib.contextmanager
def voting(post: dict[int, dict], w: float, sigs: list | None = None):
    """Add the LM vote while ensemble.combine runs. post: slot number -> {option: posterior}
    (option = word key, None = nothing, or ("term", word) for a dictionary term)."""
    orig = ensemble._vote
    state = dict(n=0)

    def vote(slot, present, order, scope=None, order_lat=None, order_cyr=None, mids=None, speech=None):
        n = state["n"]
        state["n"] += 1
        pp = post.get(n)
        if sigs is not None and n < len(sigs):
            assert sigs[n] == tuple(sorted(f"{m}:{t['k']}" for m, t in slot.toks.items())), "slot order changed"
        if not pp or w <= 0:
            return orig(slot, present, order, scope, order_lat, order_cyr, mids, speech)
        present = dict(present)
        mids = dict(mids) if mids is not None else None
        scope = dict(scope or {})
        src = {t["k"]: t for nm, t in slot.toks.items() if nm in present}
        for i, (opt, p) in enumerate(pp.items()):
            if p <= 0.01:
                continue
            name = f"lm{i}"
            present[name] = w * p
            scope[name] = None
            if mids is not None and slot.t is not None:
                mids[name] = np.array([slot.t])  # it "heard" this place: a vote for nothing counts
            if opt is None:
                continue
            if isinstance(opt, tuple):  # a dictionary term nobody produced
                word = opt[1]
                base = next(iter(src.values()))
                t = dict(base, w=word, k=T.key(word), lat=T.is_latin(word), num=None, p=1.0, m=name)
                t.pop("parts", None)
            else:
                t = dict(src[opt], p=1.0, m=name)
            slot.toks[name] = t
        return orig(slot, present, order, scope, order_lat, order_cyr, mids, speech)

    with _patched(vote):
        yield


def disputed(slots: list[dict], thr: float = DISPUTE) -> list[int]:
    return [i for i, s in enumerate(slots) if len(s["options"]) > 1 and s["margin"] < thr]


def contexts(slots: list[dict], i: int, options: list) -> tuple[str, list[str], str]:
    """Left context, the option texts, right context (pass-1 winners around slot i)."""
    def words(rng):
        return [slots[j]["surf"] for j in rng if slots[j]["surf"]]
    left = words(range(max(0, i - 3 * LEFT), i))[-LEFT:]
    right = words(range(i + 1, min(len(slots), i + 1 + 3 * RIGHT)))[:RIGHT]
    s = slots[i]
    texts = []
    for o in options:
        if o is None:
            texts.append("")
        elif isinstance(o, tuple):
            texts.append(o[1])
        else:
            texts.append(s["words"].get(o, o))
    return " ".join(left), texts, " ".join(right)


def posterior(scores: list[float], temp: float = 1.0) -> list[float]:
    m = max(scores)
    e = [math.exp((x - m) / temp) for x in scores]
    z = sum(e)
    return [x / z for x in e]


# ---------------------------------------------------------------- scorers

class CausalScorer:
    """log P(option + right context | prefix + left context) with a causal LM (transformers)."""

    def __init__(self, path: str, dtype: str = "bfloat16", threads: int = 4, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype)).to(device).eval()

    def score(self, left: str, texts: list[str], right: str) -> list[tuple[float, int]]:
        """(sum of log-probs, number of scored tokens) of each option + the right context."""
        torch = self.torch
        ctx = PREFIX + left
        cids = self.tok(ctx, add_special_tokens=False)["input_ids"]
        seqs, starts = [], []
        for t in texts:
            tail = " ".join(x for x in (t, right) if x)
            full = self.tok(ctx + (" " + tail if tail else ""), add_special_tokens=False)["input_ids"]
            # the tokenization of the context may differ at the seam: score from the first difference
            k = 0
            while k < min(len(cids), len(full)) and cids[k] == full[k]:
                k += 1
            seqs.append(full)
            starts.append(max(k, 1))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        try:
            return self._score_cached(seqs, starts, pad)
        except Exception:  # noqa: BLE001 - a model whose cache can't be repeated: the plain way
            pass
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        with torch.inference_mode():
            logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).logits.float()
        lp = torch.log_softmax(logits[:, :-1], dim=-1)
        out = []
        for i, s in enumerate(seqs):
            tgt = torch.tensor(s[1:], device=lp.device)
            tok_lp = lp[i, torch.arange(len(s) - 1, device=lp.device), tgt]
            part = tok_lp[starts[i] - 1:]
            out.append((float(part.sum()), int(part.numel())))
        return out

    def _score_cached(self, seqs: list[list[int]], starts: list[int], pad: int) -> list[tuple[float, int]]:
        """The shared beginning (prefix + left context) is run once, the options on top of its cache."""
        torch = self.torch
        k = min(starts)  # tokens [0, k) are common to all options; token k-1's logits score token k
        k = min(k, *(len(s) - 1 for s in seqs))
        if k < 2:
            raise ValueError("no common prefix")
        head = torch.tensor([seqs[0][:k]], device=self.device)
        with torch.inference_mode():
            o = self.model(input_ids=head, use_cache=True)
            cache = o.past_key_values
            last = o.logits[0, -1].float()
            n = len(seqs)
            cache.batch_repeat_interleave(n)
            L = max(len(s) for s in seqs) - k
            ids = torch.full((n, L), pad, dtype=torch.long)
            att = torch.zeros((n, k + L), dtype=torch.long)
            att[:, :k] = 1
            for i, s in enumerate(seqs):
                ids[i, :len(s) - k] = torch.tensor(s[k:])
                att[i, k:len(s)] = 1
            logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                past_key_values=cache, use_cache=True).logits.float()
        lp0 = torch.log_softmax(last, dim=-1)
        lp = torch.log_softmax(logits, dim=-1)
        out = []
        for i, s in enumerate(seqs):
            vals = []
            for pos in range(starts[i], len(s)):  # log P(s[pos] | s[:pos])
                j = pos - k  # logits row predicting s[pos] is pos-1-k in the continuation
                vals.append(float(lp0[s[pos]]) if j == 0 else float(lp[i, j - 1, s[pos]]))
            out.append((sum(vals), len(vals)))
        return out


class MaskedScorer:
    """Pseudo-log-likelihood with a masked LM (PLL-word-l2r: a token is masked together with the
    rest of its word), over the option and a few context words on each side."""

    def __init__(self, path: str, threads: int = 4, window: int = 3):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.window = window
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForMaskedLM.from_pretrained(path, dtype=torch.float32).eval()

    def score(self, left: str, texts: list[str], right: str) -> list[tuple[float, int]]:
        torch = self.torch
        lw, rw = left.split(), right.split()
        out = []
        for t in texts:
            mid = t.split()
            words = lw + mid + rw
            enc = self.tok(" ".join(words), return_offsets_mapping=True, return_tensors="pt")
            ids, offs = enc["input_ids"][0], enc["offset_mapping"][0].tolist()
            # word index of each token
            spans, pos = [], 0
            text = " ".join(words)
            for w in words:
                a = text.index(w, pos)
                spans.append((a, a + len(w)))
                pos = a + len(w)
            lo = max(0, len(lw) - self.window)
            hi = min(len(words), len(lw) + len(mid) + self.window)
            wid = []
            for a, b in offs:
                k = next((j for j, (x, y) in enumerate(spans) if a < y and b > x), None) if b > a else None
                wid.append(k)
            pick = [i for i, k in enumerate(wid) if k is not None and lo <= k < hi]
            if not pick:
                out.append((0.0, 0))
                continue
            batch = ids.repeat(len(pick), 1)
            for r, i in enumerate(pick):
                same = [j for j in pick if wid[j] == wid[i] and j >= i]
                batch[r, same] = self.tok.mask_token_id
            with torch.inference_mode():
                logits = self.model(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
            lp = torch.log_softmax(logits.float(), dim=-1)
            s = sum(float(lp[r, i, ids[i]]) for r, i in enumerate(pick))
            out.append((s, len(pick)))
        return out


def option_scores(scorer, slots: list[dict], idx: list[int], extra: dict | None = None,
                  log=print) -> dict[int, dict]:
    """Slot number -> {"options": [...], "lp": [(sum, ntok)], "texts": [...]} for the given slots.
    extra: slot number -> list of additional options (("term", word))."""
    res = {}
    t0 = time.monotonic()
    for n, i in enumerate(idx):
        opts = list(slots[i]["options"])
        opts += [o for o in (extra or {}).get(i, []) if o not in opts]
        left, texts, right = contexts(slots, i, opts)
        res[i] = dict(options=opts, texts=texts, lp=scorer.score(left, texts, right), left=left, right=right)
        if log and (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(idx)} slots, {time.monotonic() - t0:.0f} s")
    return res


def to_posterior(sc: dict[int, dict], norm: bool, temp: float = 1.0) -> dict[int, dict]:
    """Option log-probs -> posteriors; norm: per-token mean scaled by the mean token count."""
    post = {}
    for i, r in sc.items():
        lps = r["lp"]
        if norm:
            nbar = sum(n for _, n in lps) / len(lps)
            vals = [s / max(n, 1) * nbar for s, n in lps]
        else:
            vals = [s for s, _ in lps]
        post[i] = dict(zip(r["options"], posterior(vals, temp)))
    return post
