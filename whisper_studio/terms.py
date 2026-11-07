"""Post-merge term recovery: German words inside Russian speech that every model wrote in
Cyrillic («анмельдунге», «шпильплац», «вонунг»), turned back into the project's term
(Anmeldung, Spielplatz, Wohnung). CPU only, no recognition (docs/DECISIONS.md).

How:
1. Terms: the recording's glossary, the project dictionary (compounds also give their parts:
   Hotel-Anmeldung -> Anmeldung) and the learned list of German words from the
   user texts (tools/build_terms.py -> HOME/terms-learned.json).
2. Candidates: spans of 1-3 merged words in Russian speech, with a Russian case ending cut
   off the last word («Wohnungов», «Anmeldungе»), compared with every term by sound
   (asr.text.phon: one skeleton for Cyrillic and German spelling; a shared beginning counts
   extra - a Russian ear keeps the start of a German word and garbles the rest).
3. Evidence: what each model heard at the same time (its own 1-3 words there). A model counts
   only if what it heard is not plain Russian (a word pymorphy doesn't know, or Latin):
   models agreeing on a real Russian word are just the same mishearing.
4. A span of real Russian words needs more evidence than a span with a word Russian doesn't
   know; a term already recovered with strong evidence in this recording needs less
   (one conversation keeps to one topic).

Pure Python (pymorphy3 through asr.text), imported by ensemble.combine.
"""
from __future__ import annotations

import json
import os
import re
from bisect import bisect_left, bisect_right
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from . import settings
from .asr import text as T

# Russian case endings a speaker puts on a German noun ("Wohnungов", "Anmeldungе")
ENDINGS = ("ами", "ями", "ого", "ому", "ах", "ям", "ям", "ов", "ам", "ом", "ой", "ей", "ую", "ые", "ым",
           "а", "у", "е", "и", "ы", "ю")
SIM = 0.72        # a model's words "sound like" the term
SIM_STRONG = 0.85
MIN_TERM = 5      # shorter terms (ICE, Gleis) only on an exact sound match
TIMED_NEAR = 0.6  # s: a timed model's words this close to the span are "at the same place"
UNTIMED_NEAR = 2.5
GLUED = True      # also try a span without a preposition glued onto it («пошпильплац»)
GLUED_COST = 0.03
PREPOSITIONS = ("про", "по", "на", "за")
LOOSE_BONUS = 0.05  # recover(loose=True): a retrieved term's sound counts this much more


def learned_file() -> Path:
    env = os.environ.get("WHISPER_STUDIO_TERMS")
    return Path(env) if env else settings.HOME / "terms-learned.json"


def learned_terms() -> list[str]:
    try:
        return [t for t in json.loads(learned_file().read_text("utf-8")).get("terms", []) if str(t).strip()]
    except Exception:  # noqa: BLE001 - no learned list yet
        return []


def compound_parts(term: str) -> list[str]:
    """Hotel-Anmeldung -> Hotel, Anmeldung. Names of several words
    (Gartenhaus Sonnenberg) are not split: "Sonnenberg" alone is just a word."""
    if " " in term.strip():
        return []
    return [p for p in term.split("-") if len(p) >= 6 and T.LATIN.search(p) and p != term]


def term_list(base: list[str], learned: bool = True) -> list[str]:
    """Glossary + dictionary terms (as ensemble._glossary gives them), their compound parts,
    then the learned list; only Latin ones, each once."""
    out, seen = [], set()
    for t in [*base, *(p for t in base for p in compound_parts(t)), *(learned_terms() if learned else [])]:
        k = T.key(t)
        if k and k not in seen and T.LATIN.search(t):
            seen.add(k)
            out.append(t)
    return out


# ---------------------------------------------------------------- similarity

@lru_cache(maxsize=200000)
def sim(heard: str, term_key: str) -> float:
    """Sound similarity of a heard key (any script) and a term key, 0..1: edit ratio of the
    phonetic skeletons plus a bonus for a common beginning (Jaro-Winkler style)."""
    a, b = T.phon(heard), T.phon(term_key)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(b) < MIN_TERM:  # ICE, Gleis: same length and beginning, one sound off at most
        if len(a) != len(b) or a[0] != b[0]:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()
    if len(a) <= 2:
        return 0.0
    r = len(a) / len(b)
    if r < 0.6 or r > 1.6:
        return 0.0
    s = SequenceMatcher(None, a, b).ratio()
    p = 0
    while p < min(4, len(a), len(b)) and a[p] == b[p]:
        p += 1
    return s + 0.08 * p * (1 - s)


def glued(k: str) -> list[tuple[str, str]]:
    """(rest, preposition) of a key that may start with a preposition the models glued on
    («пошпильплац» = по Spielplatz, «поштрассенбан» = по Straßenbahn)."""
    if not GLUED or not T.CYRILLIC.search(k[:1]):
        return []
    return [(k[len(p):], p) for p in PREPOSITIONS if k.startswith(p) and len(k) - len(p) >= 4]


@lru_cache(maxsize=100000)
def stems(k: str, last: str) -> tuple[tuple[str, str, str], ...]:
    """(key to compare, Russian ending cut off, preposition cut off) for a span (k: its words'
    keys joined, last: the last word's key): as heard, with a case ending cut off a Cyrillic
    last word, and both again without a preposition glued onto the first word."""
    out = []
    for kk, pre in [(k, "")] + glued(k):
        out.append((kk, "", pre))
        if T.CYRILLIC.search(last):
            for e in ENDINGS:
                if last.endswith(e) and len(last) - len(e) >= 3 and len(kk) - len(e) >= 3:
                    out.append((kk[: -len(e)], e, pre))
    return tuple(out)


@lru_cache(maxsize=400000)
def _best(k: str, last: str, term_key: str) -> tuple[float, str, str]:
    # a preposition cut off costs a little: the whole word wins a tie
    return max(((sim(x, term_key) - (GLUED_COST if pre else 0.0), e, pre) for x, e, pre in stems(k, last)),
               default=(0.0, "", ""))


def best_sim(words: list[str], term_key: str) -> tuple[float, str, str]:
    keys = [T.key(w) for w in words]
    return _best("".join(keys), keys[-1], term_key)


def plain_russian(words: list[str]) -> bool:
    """Every word is Cyrillic and known to the Russian dictionary."""
    return all(T.CYRILLIC.search(w) and not T.LATIN.search(w) and T.known_russian(w) for w in words)


# ---------------------------------------------------------------- the pass

class Index:
    """Terms bucketed by the first sound of their skeleton (a Russian ear keeps it)."""

    def __init__(self, terms: list[str]):
        self.terms = terms
        self.by_first: dict[str, list[tuple[str, str]]] = {}
        for t in terms:
            k = T.key(t)
            ph = T.phon(k)
            if ph:
                self.by_first.setdefault(ph[0], []).append((k, t))

    def near(self, heard: str) -> list[tuple[str, str]]:
        out = []
        for k in [heard] + [r for r, _ in glued(T.key(heard))]:
            ph = T.phon(k)
            for x in self.by_first.get(ph[:1], []) if ph else []:
                if x not in out:
                    out.append(x)
        return out


def prefix_sim(words: list[str], term_key: str) -> float:
    """The span sounds like the beginning of a longer term: the merge kept only a piece
    («штрас» of «штрассен бан»), the rest got lost in the vote."""
    a, b = T.phon("".join(T.key(w) for w in words)), T.phon(term_key)
    if len(a) < 4 or len(b) < 2 * MIN_TERM or len(a) / len(b) >= 0.6:
        return 0.0
    return SequenceMatcher(None, a, b[: len(a) + 1]).ratio()


def _spans(toks: list[dict], n_max: int = 3):
    """Runs of 1-3 words that can be one term: no number, no comma or full stop inside,
    no one- or two-letter word glued in («и», «да»)."""
    for i in range(len(toks)):
        for n in range(1, n_max + 1):
            part = toks[i:i + n]
            if len(part) < n:
                break
            words = [x["w"] for x in part]
            if T.number_value(words[-1]) is not None or (n > 1 and T.tail_punct(words[-2])):
                break
            if n > 1 and any(len(T.key(w)) <= 2 for w in words):
                break
            yield i, part


class Members:
    """Every model's 1-3-word runs over the recording (see _spans), sorted by time, with what
    the evidence needs precomputed: keys, plain Russian or not."""

    def __init__(self, members: dict[str, list[dict]] | None):
        self.runs: dict[str, list[tuple]] = {}
        self.starts: dict[str, list[float]] = {}
        for name, ts in (members or {}).items():
            ts = sorted(ts, key=lambda t: t["s"])
            runs = []
            for _, g in _spans(ts):
                ws = [t["w"] for t in g]
                keys = [T.key(w) for w in ws]
                pad_a = TIMED_NEAR if g[0].get("timed", True) else UNTIMED_NEAR
                pad_b = TIMED_NEAR if g[-1].get("timed", True) else UNTIMED_NEAR
                odd = any(T.LATIN.search(w) or T.known_russian(w) is False for w in ws)
                runs.append((g[0]["s"], g[0]["e"] + pad_a, g[-1]["s"] - pad_b, "".join(keys), keys[-1],
                             plain_russian(ws), odd, g))
            self.runs[name] = runs
            self.starts[name] = [r[0] for r in runs]

    def near(self, name: str, s0: float, e0: float):
        """Runs that lie at [s0, e0] (each word within the padding of the span)."""
        runs, st = self.runs[name], self.starts[name]
        lo = bisect_left(st, s0 - UNTIMED_NEAR - 3.0)
        hi = bisect_right(st, e0 + UNTIMED_NEAR)
        for r in runs[lo:hi]:
            if r[1] >= s0 and r[2] <= e0:
                yield r


def _evidence(term_key: str, span: tuple[float, float], members: Members, heard: str,
              right: float = 0.0) -> list[tuple[str, float, bool, bool]]:
    """Per model: best similarity of its own 1-3 words at the span's time to the term, whether
    those words are plain Russian, and whether they are the merged words themselves (heard).
    right: look further after the span (the merge kept only a piece)."""
    out = []
    s0, e0 = span
    ph = T.phon(heard)
    for name in members.runs:
        best, rus, same = 0.0, True, False
        for r in members.near(name, s0, e0 + right):
            sc = _best(r[3], r[4], term_key)[0]
            if sc > best:
                best, rus = sc, r[5]
                same = SequenceMatcher(None, T.phon(r[3]), ph).ratio() >= 0.85
        if best > 0:
            out.append((name, best, rus, same))
    return out


def _support(ev: list) -> list[float]:
    """Models that heard something non-Russian like the term, strongest first. A model that
    wrote just the merged word is no news unless it is close to the term (all of them writing
    «гартен» does not make it Garten). The forced-German re-check (de-check) mostly
    translates: it counts half, and only when it is close."""
    sup = [s for n, s, rus, same in ev
           if not rus and n != "de-check" and s >= SIM and (not same or s >= SIM_STRONG)]
    sup += [s / 2 for n, s, rus, same in ev if n == "de-check" and s >= SIM_STRONG]
    return sorted(sup, reverse=True)


def _possible(s_m: float, partial: bool, span_rus: bool, has_odd: bool, low_p: bool, short: bool) -> bool:
    """_decide could say yes with some evidence."""
    if short:
        return not span_rus
    if partial:
        return s_m >= 0.75 or (has_odd and low_p)
    if span_rus:
        return s_m >= SIM_STRONG
    return s_m >= SIM or (has_odd and s_m >= 0.35)


def _decide(s_m: float, partial: bool, span_rus: bool, has_odd: bool, low_p: bool, ev: list,
            primed: bool, short: bool) -> bool:
    """Is the term there? s_m: the merged words against the term (partial: against its
    beginning); ev: the models' evidence; primed: found with strong evidence earlier in this
    recording."""
    sup = _support(ev)
    n = sum(1 if s >= SIM else 0.5 for s in sup)
    top = sup[0] if sup else 0.0
    if short:  # ICE, Gleis: never over a real Russian word; the exact sound in the merge or a model
        return not span_rus and (s_m >= 1.0 or (s_m >= 0.6 and any(s >= 1.0 for _, s, r, _s in ev if not r)))
    if partial:
        if s_m < 0.75:
            return has_odd and low_p and n >= 3 and s_m >= 0.35  # three models heard the term
        if span_rus:
            return (n >= 2 and top >= SIM_STRONG) or (primed and (n >= 2 or top >= SIM_STRONG))
        return n >= 2 or (primed and n >= 1)
    if span_rus:
        # real Russian words in the merge: the term's own sound and other models' non-Russian word
        if primed:
            return s_m >= SIM_STRONG and n >= 1
        return s_m >= SIM_STRONG and n >= 2 and top >= SIM_STRONG
    if s_m >= 0.9:
        return True  # «Forn Melding», «вонунг»: the merge itself is the term misspelled
    if s_m >= SIM and (n >= 2 or (primed and n >= 1)):
        return True
    # the merge is far off but three models heard the term: where they disagreed (low p), or
    # one of them heard it nearly exactly
    if has_odd and n >= 3 and sum(sup[:3]) / 3 >= 0.75:
        return (low_p and s_m >= 0.35) or (top >= SIM_STRONG and s_m >= 0.45)
    return False


def hotspots(idx: Index, members: Members) -> dict[str, list[tuple[float, float]]]:
    """Where two or more models heard something non-Russian that sounds like a term:
    {term key: [(start, end), ...]}. The merge there may be plain Russian or a lost piece
    («штрас» where the models heard «штрассен Bahn», «штрассенбан»)."""
    hits: dict[str, list[tuple[float, float, str]]] = {}
    for name, runs in members.runs.items():
        if name == "de-check":
            continue
        for r in runs:
            if not r[6]:
                continue  # plain Russian
            g = r[7]
            for tk, _term in idx.near(r[3]):
                if _best(r[3], r[4], tk)[0] >= SIM:
                    hits.setdefault(tk, []).append((g[0]["s"], g[-1]["e"], name))
    out: dict[str, list[tuple[float, float]]] = {}
    for tk, hs in hits.items():
        hs.sort()
        for s, e, name in hs:
            others = {n for s2, e2, n in hs if n != name and s2 <= e + 1.0 and e2 >= s - 1.0}
            if others:
                out.setdefault(tk, []).append((s - 1.0, e + 1.0))
    return out


def _family(tk: str, keys: set[str]) -> bool:
    """tk or its singular/plural (Wohnung / Wohnungen) is among keys."""
    return any(tk == k or ((tk.startswith(k) or k.startswith(tk)) and abs(len(tk) - len(k)) <= 2
                           and min(len(tk), len(k)) >= 6) for k in keys)


def recover(segs: list[dict], terms: list[str], members: dict[str, list[dict]] | None = None,
            log: list | None = None, focus=None, loose: bool = False) -> int:
    """Replace spans of merged words by terms where sound and the models' evidence agree.
    members: every model's timed tokens over the whole recording ({"w", "s", "e", "timed"}).
    Two passes: the second one also trusts the terms the first found ("primed").
    focus: per-time term subset (whisper_studio/memory.Focus: .at(start, end) -> (allowed keys,
    retrieved keys)); a term not allowed at a span is not tried there. loose: a retrieved term
    counts as primed (the owner's texts on this topic name it) and needs a bit less sound."""
    if not terms:
        return 0
    idx = Index(terms)
    mm = Members(members)
    term_keys = {T.key(t) for t in terms}
    rank = {T.key(t): i for i, t in enumerate(terms)}
    primed: set[str] = set()
    hot = hotspots(idx, mm)
    total = 0
    for pass_no in (1, 2):
        found = set()
        for seg in segs:
            ws = seg["words"]
            text = " ".join(w["w"] for w in ws)
            if seg.get("lang") == "de" or T.latin_share(text) > 0.5:
                continue  # German speech: the German models wrote it
            i = 0
            while i < len(ws):
                cands = []
                allowed, got = focus.at(ws[i]["s"], ws[i]["e"]) if focus is not None else (None, frozenset())
                for start, part in _spans(ws[i:i + 3]):
                    if start > 0:
                        break
                    words = [x["w"] for x in part]
                    keys = [T.key(w) for w in words]
                    if any(k in term_keys or x.get("term") for k, x in zip(keys, part)):
                        continue  # already a term
                    latin = all(T.is_latin(w) for w in words)
                    around = ws[max(0, i - 1):i] + ws[i + len(part):i + len(part) + 1]
                    if latin and any(T.is_latin(x["w"]) for x in around):
                        continue  # a Latin word next to Latin words: a German phrase, not a misspelt term
                    has_odd = any(T.LATIN.search(w) or T.known_russian(w) is False for w in words)
                    agree = min(x.get("_agree", x.get("p", 1)) for x in part)
                    low_p = agree < 0.35
                    unsure = has_odd or agree < 0.5
                    s0, e0 = part[0]["s"], part[-1]["e"]
                    hot_here = {tk for tk, sp in hot.items() if (allowed is None or tk in allowed or tk in primed)
                                and any(a <= e0 and b >= s0 for a, b in sp)}
                    if not (unsure or primed or hot_here or (loose and got)):
                        continue
                    span_rus = plain_russian(words)
                    for tk, term in idx.near("".join(keys)):
                        in_talk = _family(tk, primed)
                        if allowed is not None and tk not in allowed and not in_talk:
                            continue  # not about this part of the talk (memory.Focus)
                        retrieved = loose and tk in got
                        is_primed = in_talk or retrieved
                        if not unsure and not is_primed and tk not in hot_here:
                            continue  # sure plain Russian: only where models heard the term, or a term this talk is about
                        short = len(T.phon(tk)) < MIN_TERM
                        s_m, ending, pre = best_sim(words, tk)
                        partial = False
                        if s_m < 0.45 and not short and " " not in term:  # a piece of one word, not of a name
                            s_m, ending, pre, partial = prefix_sim(words, tk), "", "", True
                        if s_m < (0.6 if short else 0.35) or (latin and not short and s_m < SIM_STRONG):
                            continue  # a Latin word inside Russian speech: only a near-exact misspelling
                        if len(words) > 1 and max(best_sim(words[1:], tk)[0], best_sim(words[:-1], tk)[0]) > s_m - 0.05:
                            continue  # a word glued on that adds nothing («Gartenhaus находимся», «все эти …»)
                        span = (part[0]["s"], part[-1]["e"])
                        if not _possible(s_m + (LOOSE_BONUS if retrieved else 0.0), partial, span_rus, has_odd, low_p, short):
                            continue  # no evidence could make it (saves the lookups)
                        ev = _evidence(tk, span, mm, "".join(keys), right=1.2 if partial else 0.0)
                        if not _decide(s_m + (LOOSE_BONUS if retrieved else 0.0), partial, span_rus, has_odd,
                                       low_p, ev, is_primed, short):
                            continue
                        sup = _support(ev)
                        cands.append((s_m + (sum(sup[:3]) / 3 if sup else 0.0), -rank.get(tk, 0),
                                      len(part), term, ending, ev, s_m, pre))
                if not cands:
                    i += 1
                    continue
                best = max(cands)
                # singular over plural: in Russian speech a German noun takes Russian endings on its
                # singular stem («Wohnungов», docs/GOLD.md), «-en» is rarely heard for sure
                for c in sorted(cands, key=lambda c: len(c[3])):
                    k, kb = T.key(c[3]), T.key(best[3])
                    if c[2] == best[2] and kb.startswith(k) and len(kb) - len(k) <= 2 and c[0] >= best[0] - 0.05:
                        best = c
                        break
                _, _, n, term, ending, ev, s_m, pre = best
                part = ws[i:i + n]
                old = " ".join(x["w"] for x in part)
                if len(ending) == 1 and term[-1].lower() in "aeiouyäöü":
                    ending = ""  # «Spielplatzа»: a vowel case ending after a vowel is not kept
                surf = term + (ending if ending and T.CYRILLIC.search(part[-1]["w"]) else "")
                new = [dict(w=surf + T.tail_punct(part[-1]["w"]), s=part[0]["s"], e=part[-1]["e"],
                            p=max(min(x.get("p", 1) for x in part), 0.5), term=True)]
                if pre:  # the glued-on preposition stays a word of its own at the start of the span
                    w0 = part[0]["w"].lstrip("«\"(")
                    cut = part[0]["s"] + min(0.15, 0.25 * (part[0]["e"] - part[0]["s"]))
                    new = [dict(w=w0[:len(pre)], s=part[0]["s"], e=cut, p=new[0]["p"]), dict(new[0], s=cut)]
                    surf = f"{w0[:len(pre)]} {surf}"
                ws[i:i + n] = new
                found.add(T.key(term))
                total += 1
                if log is not None:
                    log.append(dict(pass_=pass_no, old=old, new=surf, sim=round(s_m, 3), t=round(part[0]["s"], 2),
                                    ev=[(nm, round(sc, 2), r) for nm, sc, r, _s in ev]))
                i += 1
            seg["text"] = " ".join(x["w"] for x in ws)
        if pass_no == 1:
            primed = found
            if not primed:
                break
    return total
