"""Precise word timestamps via wav2vec2 CTC forced alignment (the WhisperX recipe).

Text stays exactly as Whisper produced it; only word start/end times are refined.
Characters the aligner has no letter for (Latin letters in a Russian segment, digits) become
wildcard tokens that match any letter (WhisperX does the same): one per letter, and for a number
about as many as it has letters when said. So the word gets its own time instead of its sound
being taken by the neighbours.
"""
from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE

_cache: dict = {"path": None, "model": None, "proc": None, "dev": None}


def _load(path: Path):
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    if _cache["path"] == str(path):
        if _cache["dev"] == "cuda":
            _cache["model"].to("cuda")
        return _cache["model"], _cache["proc"], _cache["dev"]
    free()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Wav2Vec2Processor.from_pretrained(str(path))
    model = Wav2Vec2ForCTC.from_pretrained(str(path)).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    _cache.update(path=str(path), model=model, proc=proc, dev=dev)
    return model, proc, dev


def park():
    """Keep the aligner in RAM (reloading from disk is slow) but free VRAM for Whisper."""
    if _cache["model"] is not None and _cache["dev"] == "cuda":
        _cache["model"].to("cpu")
        import torch

        torch.cuda.empty_cache()


def free():
    _cache.update(path=None, model=None, proc=None, dev=None)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _viterbi(em: np.ndarray, tokens: list[int], blank: int) -> list[tuple[int, int]] | None:
    """CTC Viterbi forced alignment. Returns (first_frame, last_frame) per token."""
    T = em.shape[0]
    L = len(tokens)
    if L == 0 or T < L:
        return None
    ext = np.full(2 * L + 1, blank, dtype=np.int64)
    ext[1::2] = tokens
    S = len(ext)
    neg = -1e30
    # skip transition allowed into a non-blank that differs from the token two states back
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]
    dp = np.full(S, neg)
    dp[0] = em[0, blank]
    dp[1] = em[0, ext[1]]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = dp
        prev1 = np.concatenate(([neg], dp[:-1]))
        prev2 = np.where(skip_ok, np.concatenate(([neg, neg], dp[:-2])), neg)
        stacked = np.stack([stay, prev1, prev2])
        arg = stacked.argmax(axis=0)
        dp = stacked[arg, np.arange(S)] + em[t, ext]
        back[t] = arg
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[s] <= neg / 2:
        return None
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])
    spans: list[tuple[int, int]] = []
    for k in range(L):
        idx = np.nonzero(path == 2 * k + 1)[0]
        if len(idx) == 0:
            return None
        spans.append((int(idx[0]), int(idx[-1])))
    return spans


def spoken_len(digits: str) -> int:
    """About how many letters a number takes when said: ~7 per non-zero digit, plus «тысяч»."""
    nz = sum(c != "0" for c in digits.lstrip("0"))
    return max(3, 7 * nz + 6 * (len(digits) >= 4) + 7 * (len(digits) >= 7))


def align(segments: list[dict], audio: np.ndarray, model_path: Path,
          progress: Callable[[float, str], None] | None = None) -> int:
    """Refine words[*].s/e in place. Returns number of aligned words."""
    import torch

    model, proc, dev = _load(model_path)
    vocab = {k.lower(): v for k, v in proc.tokenizer.get_vocab().items()}
    blank = proc.tokenizer.pad_token_id
    delim = vocab.get(proc.tokenizer.word_delimiter_token or "|")
    has_yo = "ё" in vocab
    star = -2  # wildcard: an extra emission column, the best letter at each frame
    aligned = 0

    def chars(word: str) -> list[int]:
        w = word.lower()
        if not has_yo:
            w = w.replace("ё", "е")
        out: list[int] = []
        for part in re.findall(r"\d+|\D", w):
            if part.isdigit():  # «32500» is said «тридцать две тысячи пятьсот»: a wildcard per letter
                out += [star] * spoken_len(part)
            elif part in vocab and vocab[part] not in (blank, delim):
                out.append(vocab[part])
            elif part.isalpha():  # a Latin letter for a Russian aligner and the like
                out.append(star)
        return out

    for i, seg in enumerate(segments):
        words = seg.get("words") or []
        if not words:
            continue
        t0 = max(0.0, seg["start"] - 0.15)
        t1 = min(len(audio) / SAMPLE_RATE, seg["end"] + 0.15)
        chunk = audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE * 0.2:
            continue
        tokens, owner = [], []
        for wi, w in enumerate(words):
            ids = chars(w.get("spoken") or w["w"])  # «25-го» was said «двадцать пятого»
            if ids and tokens and delim is not None:
                tokens.append(delim)
                owner.append(-1)
            tokens += ids
            owner += [wi] * len(ids)
        if not tokens:
            continue
        inp = proc(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(dev)
        if dev == "cuda":
            inp = inp.half()
        with torch.inference_mode():
            logits = model(inp).logits[0].float()
        em = torch.log_softmax(logits, dim=-1).cpu().numpy()
        if star in tokens:
            letters = np.delete(em, [blank, *([delim] if delim is not None else [])], axis=1)
            em = np.concatenate([em, letters.max(axis=1, keepdims=True)], axis=1)
            tokens = [em.shape[1] - 1 if t == star else t for t in tokens]
        spans = _viterbi(em, tokens, blank)
        if spans is None:
            continue
        ratio = (len(chunk) / SAMPLE_RATE) / em.shape[0]
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for (a, b), wi in zip(spans, owner):
            if wi < 0:
                continue
            first.setdefault(wi, a)
            last[wi] = b
        for wi, w in enumerate(words):
            if wi in first:
                w["s"] = round(t0 + first[wi] * ratio, 3)
                w["e"] = round(t0 + (last[wi] + 1) * ratio, 3)
                aligned += 1
        # keep unaligned words inside their neighbours, keep order monotonic
        for wi in range(1, len(words)):
            if words[wi]["s"] < words[wi - 1]["s"]:
                words[wi]["s"] = words[wi - 1]["e"]
            if words[wi]["e"] < words[wi]["s"]:
                words[wi]["e"] = words[wi]["s"]
        seg["start"], seg["end"] = words[0]["s"], max(w["e"] for w in words)
        if progress:
            progress((i + 1) / len(segments), "Уточняю таймкоды слов")
    return aligned
