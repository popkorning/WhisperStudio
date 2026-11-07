"""Numbers in digits for the final transcript (inverse text normalization).

The merge votes on number keys (text.merge_numbers: «двадцать пятого» and «25-го» are both #25);
here the winning words are written in one style: «25000 евро», «3 марта», «в 2 часа», «19%», «1,5»,
«5–6», «в 7:30», «25-го», «2025 года», «25-го года», German
«zwei Wochen» -> «2 Wochen», dictated digits after a cue word glued («номер 01232»). A number is
one token without spaces inside: the merge, the gold comparison, the proofreading and the editor
split text on spaces, so «25000», not «25 000» (grouping is for display); only millions and
billions keep the word («65 миллионов», «1,5 миллиарда»).

The text is made from the key, so the key stays what it was. Words stay words where a number
word is not a count: idioms («в первую очередь», «первое время», «ни одного», «раз, два»),
fractions («ноль целых пять десятых»), a bare «тысячу»/«тысячи» without a count, and «один/одна…»
alone, which is a pronoun or an article far more often than a count («один из», «один момент»,
«одно и то же», «один на один»): it becomes 1 only before a unit («один евро», «один процент»).
Homonyms («семью» = 7 or a family, «сорока») need a counted noun after them (pymorphy3; without
it they stay words). The spoken words stay in w["spoken"], so the change can be undone.
"""
from __future__ import annotations

import re

from . import text as T

MONTH = re.compile(r"^(январ|феврал|март|апрел|ма[йяю]$|июн|июл|август|сентябр|октябр|ноябр|декабр)")
UNIT = re.compile(r"^(евро|цент|процент|рубл|доллар|франк|фунт|штук|килограмм|кило|грамм|километр|метр|литр|"
                  r"тонн|страниц|лист|пункт|квартал|месяц|недел|год|лет|час|сутк|тысяч|миллион|миллиард|"
                  r"euro|cent|prozent|stunde|jahr|monat|woche|seite)")
FIRST_IDIOM = {"очередь", "время", "делом", "взгляд", "взгляда", "раз", "порах"}  # «в первую очередь»
KEEP_AFTER = {"ни", "раз"}  # «ни одного», «раз, два, три»
FRACTION = {"целых", "целая", "целой", "десятых", "сотых", "тысячных", "десятая", "сотая", "тысячная"}
HOMONYM = {"семью", "сорока"}
SINGLE_THOUSAND = {"тысяча", "тысячу", "тысячей", "тысяче"}
_LEAD = re.compile(r"^[«\"(\[]+")


def _halves(word: str) -> list[str]:
    return [h for h in re.split(r"[-–—]", T._bare(word)) if h]


def _known(word: str) -> bool:
    """A number word this module may write in digits (stem-guessed forms like «пятерка» are not)."""
    hs = _halves(word)
    return bool(hs) and all(h in T.CARDINALS or T.ordinal(h) or h in T._DE_MULT
                            or (T.LATIN.search(h) and T.german_number(h) is not None) for h in hs)


def _counted(word: str) -> bool:
    """«семью детьми»: a noun in the plural follows (pymorphy3)."""
    T.known_russian("да")  # loads the analyzer once
    m = T._morph["m"]
    if m is None or not word:
        return False
    p = m.parse(T.key(word))[0]
    return "NOUN" in p.tag and "plur" in p.tag


def _render(core: str, ws: list[str], pct: bool, before: list[str], nxt: str) -> list[str] | None:
    """The written form of a number (its key without "#", its words, followed by «%»/«процентов»):
    one or two words, None: leave the words."""
    glued = []
    for w in ws:  # «25 -го» -> «25-го»
        if glued and T._SUFFIX.fullmatch(w):
            glued[-1] += w
        else:
            glued.append(w)
    if len(glued) == 1 and re.search(r"\d", glued[0]):
        return None if glued == ws else glued  # already digits
    num = [w for w in ws if not T.is_percent(w)]
    if not num or not all(re.search(r"\d", w) or _known(w) for w in num):
        return None
    nk, prev = T.key(nxt), [T.key(w) for w in before]
    if nk in FRACTION or (prev and prev[-1] in FRACTION) or any(T.key(w) in FRACTION for w in num):
        return None
    for p in reversed(prev):  # «ни одного», «раз, два, три»
        if p in KEEP_AFTER:
            return None
        if T.number_value(p) is None:
            break
    last = _halves(num[-1])[-1] if not re.search(r"\d", num[-1]) else ""
    if len(num) == 1 and last:
        v = T.CARDINALS.get(last)
        if v in T.MULTIPLIERS and len(_halves(num[0])) == 1 and not (last in SINGLE_THOUSAND and UNIT.match(nk)):
            return None  # «эту тысячу», «все тысячи»; but «тысяча евро» = 1000 евро
        if v == 1 and not pct and not UNIT.match(nk):
            return None  # «один из», «одна она», «один момент»
        if last in HOMONYM and not pct and not (UNIT.match(nk) or _counted(nxt)):
            return None
        o = T.ordinal(last)
        if o and o[0] == 1 and (nk in FIRST_IDIOM or T.number_value(nxt) is not None):
            return None  # «в первую очередь», «первые три урока»
        if o and prev and not T.tail_punct(before[-1]) and (prev[-1] in T.CARDINALS or prev[-1].isdigit()):
            return None  # a fraction: «одна вторая», «три пятых»
    text = core if (":" in core or "–" in core) else core.replace(".", ",")
    if T._CURRENCY.match(nk) and re.fullmatch(r"\d+,\d", text):
        text += "0"  # «2,50 евро»
    lead = _LEAD.match(ws[0])
    m = T.PUNCT_TAIL.search(ws[-1])
    head, tail = (lead.group(0) if lead else ""), (m.group(0) if m else "")
    mult = T.CARDINALS.get(last) or T._DE_MULT.get(last)
    if not pct and len(num) > 1 and mult in (1_000_000, 1_000_000_000) and core.isdigit():
        out = [head + T._fmt(int(core) / mult).replace(".", ","), num[-1]]  # «65 миллионов»
        return None if out == ws else out
    o = T.ordinal(last) if last else None
    if o:
        if MONTH.match(nk) or (core.isdigit() and int(core) >= 1000 and nk.startswith("год")):
            pass  # «3 марта», «2025 года»
        else:
            text += "-" + o[1]
    elif last and T.LATIN.search(last) and T.german_number(last) is not None and T._de_cardinal(last) is None:
        if not (T.LATIN.search(nxt) and nxt[:1].isupper()):
            return None  # a German ordinal is written «15.» only before a noun («am 15. März»)
        text += "."
    m = re.search(r"\d(-?[а-я]{1,3})$", num[-1].lower().rstrip(",.!?;:…»\")"))
    if m and not o:  # «две тысячи 25-го»: the member's own ending
        text += m.group(1) if m.group(1).startswith("-") else "-" + m.group(1)
    out = [head + text + ("%" if pct else "") + tail]
    return None if out == ws else out


def _token(ws: list[dict], idx: list[int], w: str) -> dict:
    t = dict(ws[idx[0]], w=w)
    if "e" in ws[idx[-1]]:
        t["e"] = ws[idx[-1]]["e"]
    for k in ("p", "_agree"):
        vals = [ws[i][k] for i in idx if ws[i].get(k) is not None]
        if vals:
            t[k] = min(vals)  # the number is as doubtful as its most doubtful word
    if "_slot" in ws[idx[-1]]:  # the merge's slot of the last word (ensemble._split takes the alts up to it)
        t["_slot_last"] = ws[idx[-1]]["_slot"]
    t["spoken"] = " ".join(ws[i].get("spoken", ws[i]["w"]) for i in idx)
    return t


def words(ws: list[dict]) -> list[dict]:
    """A segment's words with the numbers in digits: a number becomes one word from the start of
    its first word to the end of its last, p = the lowest, the words said in "spoken". Words the
    owner edited are left alone. Idempotent."""
    toks = T.merge_numbers([dict(w=w["w"], i=i) for i, w in enumerate(ws)])
    out: list[dict] = []
    done: set[int] = set()
    k = 0
    while k < len(toks):
        t = toks[k]
        idx = [p["i"] for p in t.get("parts") or [t]]
        pct = bool(t.get("num") and k + 1 < len(toks) and toks[k + 1].get("k") == "%" and not T.tail_punct(t["w"]))
        if pct:  # «19%» is split into «19» and «%» by merge_numbers
            k += 1
            idx += [p["i"] for p in toks[k].get("parts") or [toks[k]]]
        k += 1
        idx = sorted(set(idx) - done)
        if not idx:
            continue
        done.update(idx)
        r = None
        if t.get("num") and not any(ws[i].get("edited") for i in idx):
            prev = [ws[i]["w"] for i in range(max(0, idx[0] - 3), idx[0])]
            nxt = ws[idx[-1] + 1]["w"] if idx[-1] + 1 < len(ws) else ""
            r = _render(t["k"][1:], [ws[i]["w"] for i in idx], pct, prev, nxt)
        if r is None:
            out += [ws[i] for i in idx]
            continue
        out.append(_token(ws, idx if len(r) == 1 else idx[:-1], r[0]))
        if len(r) == 2:
            out.append(ws[idx[-1]])  # the multiplier word keeps its own time
    return out


def text(s: str) -> str:
    """The same for plain text (members' variants, segments without words)."""
    return " ".join(w["w"] for w in words([dict(w=x) for x in s.split()]))


def renumber_doc(doc: dict) -> list[tuple[str, str, str]]:
    """A stored transcript with its numbers in digits (`ws renumber`), from the words it has:
    segments the owner or the AI edited stay as they are. The members' variants follow, so
    proofreading sees no «25» vs «двадцать пять». Returns (segment id, old text, new text)."""
    changes = []
    for s in doc.get("segments") or []:
        if s.get("edited"):
            continue
        for a in s.get("alts") or []:
            a["text"] = text(a.get("text") or "")
        old = s.get("words")
        if old:
            new = words(old)
            if [w["w"] for w in new] == [w["w"] for w in old]:
                continue
            s["words"], t = new, " ".join(w["w"] for w in new)
        else:
            t = text(s["text"])
        if t != s["text"]:
            changes.append((s.get("id"), s["text"], t))
            s["text"] = t
    return changes
