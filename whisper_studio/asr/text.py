"""Text helpers for merging hypotheses: comparison keys, numbers, a rough phonetic skeleton.

Two words are "the same" for voting when their keys match (case, punctuation and ё ignored;
«25-го» and «двадцать пятого» both become #25). The phonetic skeleton compares words across
scripts, so «вонунг» from a Russian model counts as a vote for «Wohnung».
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache

PUNCT_TAIL = re.compile(r"[\s,.!?;:…»\")\]–—-]+$")
LATIN = re.compile(r"[A-Za-zÄÖÜäöüß]")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def key(word: str) -> str:
    w = word.lower().replace("ё", "е")
    return re.sub(r"[^\w]+", "", w)


def tail_punct(word: str) -> str:
    m = PUNCT_TAIL.search(word)
    return m.group(0).strip() if m else ""


def is_latin(word: str) -> bool:
    lat = len(LATIN.findall(word))
    return lat > 0 and lat >= len(CYRILLIC.findall(word))


def latin_share(text: str) -> float:
    lat, cyr = len(LATIN.findall(text)), len(CYRILLIC.findall(text))
    return lat / (lat + cyr) if lat + cyr else 0.0


# ---------------------------------------------------------------- numbers

_NUM_STEMS = sorted({
    "ноль": 0, "нол": 0, "нул": 0,
    "один": 1, "одна": 1, "одно": 1, "одну": 1, "одного": 1, "одной": 1, "одном": 1, "перв": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "втор": 2,
    "три": 3, "трех": 3, "трем": 3, "трет": 3,
    "четыре": 4, "четырех": 4, "четверт": 4,
    "пять": 5, "пяти": 5, "пят": 5,
    "шесть": 6, "шести": 6, "шест": 6,
    "семь": 7, "семи": 7, "седьм": 7,
    "восемь": 8, "восьми": 8, "восьм": 8,
    "девять": 9, "девяти": 9, "девят": 9,
    "десять": 10, "десяти": 10, "десят": 10,
    "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14, "пятнадцат": 15,
    "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19,
    "двадцат": 20, "тридцат": 30, "сорок": 40, "пятьдесят": 50, "пятидесят": 50,
    "шестьдесят": 60, "шестидесят": 60, "семьдесят": 70, "семидесят": 70,
    "восемьдесят": 80, "восьмидесят": 80, "девяносто": 90, "девяност": 90,
    "сто": 100, "ста": 100, "сот": 100, "двест": 200, "двухсот": 200, "трист": 300, "трехсот": 300,
    "четырест": 400, "четырехсот": 400, "пятьсот": 500, "пятисот": 500, "шестьсот": 600,
    "шестисот": 600, "семьсот": 700, "семисот": 700, "восемьсот": 800, "восьмисот": 800,
    "девятьсот": 900, "девятисот": 900,
    "четырем": 4, "четырьм": 4, "пятьюдесят": 50, "шестьюдесят": 60, "семьюдесят": 70, "восемьюдесят": 80,
    "восьмьюдесят": 80, "двумст": 200, "двумяст": 200, "двухст": 200, "тремст": 300, "тремяст": 300,
    "трехст": 300, "четыремст": 400, "четырьмяст": 400, "четырехст": 400, "пятист": 500, "пятьюст": 500,
    "шестист": 600, "шестьюст": 600, "семист": 700, "семьюст": 700, "восьмист": 800, "восемьюст": 800,
    "восьмьюст": 800, "девятист": 900, "девятьюст": 900, "полтор": 1.5, "полутор": 1.5, "двухтысячн": 2000,
    "тысяч": 1000, "миллион": 1_000_000, "миллиард": 1_000_000_000,
}.items(), key=lambda kv: -len(kv[0]))
# words that start like a number but are not ("сотня" is fine, "стол" is not)
_NOT_NUM = re.compile(r"^(сто[йияелрлкпю]|став|стат|ста[нрвл]|сот[ск]|трет[ья]?[ея]?сь|пятн(?:о|а(?!дцат))|"
                      r"перв[оа]?[нц]|втор[гн]|одна[кж]|однак|одно[вр]|десятк|сороко[кн]|нуле)")


_DE_UNITS = {"null": 0, "eins": 1, "ein": 1, "zwei": 2, "zwo": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6,
             "sieben": 7, "acht": 8, "neun": 9}
_DE_TEENS = {"zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
             "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
_DE_TENS = {"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
            "siebzig": 70, "achtzig": 80, "neunzig": 90}
_DE_ORDINAL = {"erst": 1, "dritt": 3, "siebt": 7}


def _de_below_100(w: str) -> int | None:
    if w in _DE_TEENS:
        return _DE_TEENS[w]
    if w in _DE_TENS:
        return _DE_TENS[w]
    if w in _DE_UNITS and w != "ein":
        return _DE_UNITS[w]
    if "und" in w:  # fünfundzwanzig = 5 + 20
        u, t = w.split("und", 1)
        if u in _DE_UNITS and t in _DE_TENS:
            return _DE_UNITS[u] + _DE_TENS[t]
    return None


def _de_cardinal(w: str) -> int | None:
    if not w:
        return None
    for word, mult in (("tausend", 1000), ("hundert", 100)):
        if word in w:
            left, right = w.split(word, 1)
            lv = 1 if left in ("", "ein") else _de_cardinal(left)
            rv = 0 if right == "" else _de_cardinal(right)
            if lv is None or rv is None or (mult == 100 and lv > 99) or rv >= mult:
                return None
            return lv * mult + rv
    return _de_below_100(w)


def german_number(w: str) -> int | None:
    """German number words, cardinal and ordinal: «fünfundzwanzig» -> 25, «tausendzweihundert»
    -> 1200, «zweitausendfünfundzwanzig» -> 2025, «fünfzehnten» -> 15, «dritte» -> 3.
    Articles (ein, eine, einen…) are not numbers."""
    v = _de_cardinal(w)
    if v is not None:
        return v if w != "ein" else None
    if w == "achte":  # "achten", "Achter" are a verb and a noun far more often than the 8th
        return 8
    m = re.fullmatch(r"([a-zäöüß]+t)e[nrsm]?", w)  # ordinal: zweite(n), dritte, zwanzigsten
    if not m:
        return None
    stem = m.group(1)
    if stem in _DE_ORDINAL:
        return _DE_ORDINAL[stem]
    base = stem[:-2] if stem.endswith("st") and len(stem) > 6 else stem[:-1]  # zwanzig-st, zwei-t
    v = _de_cardinal(base)
    return v if v is not None and base not in ("ein", "acht") else None


_DE_MULT = {"tausend": 1000, "million": 1_000_000, "millionen": 1_000_000, "milliarde": 1_000_000_000,
            "milliarden": 1_000_000_000}


def number_value(word: str) -> int | float | None:
    """Value of a single numeric token: «25-го» -> 25, «0,5» -> 0.5, «пятого» -> 5."""
    k = word.lower().replace("ё", "е").strip(",.!?;:…»«\"()")
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)(?:-?[а-я]{1,3})?%?", k)
    if m:
        v = m.group(1).replace(",", ".")
        return float(v) if "." in v else int(v)
    if re.fullmatch(r"[a-zäöüß]+", k):
        return _DE_MULT.get(k) or german_number(k)
    k = re.sub(r"[^а-я]", "", k)
    if not k or _NOT_NUM.match(k):
        return None
    for stem, val in _NUM_STEMS:
        if k.startswith(stem):
            rest = k[len(stem):]
            # allow only inflection endings after the stem
            if len(rest) <= 4 and (not rest or re.fullmatch(r"[аеиоуыьъяюй]?[а-я]{0,3}", rest)):
                if stem in ("ста", "сот") and rest not in ("", "м", "ми", "х") and not (
                        stem == "сот" and rest in _ORD_END):  # «сотый» is a number, «сотня» is not
                    return None
                return val
            return None
    return None


def _lexicon() -> dict[str, int | float]:
    """Every form of the Russian cardinals (ё as е): the words that may be written in digits."""
    d: dict[str, int | float] = {}

    def add(v, words: str):
        d.update(dict.fromkeys(words.split(), v))

    add(0, "ноль нуль ноля нуля нолю нулю нолем нулем")
    add(1, "один одна одно одни одного одной одному одним одном одну одною одних одними")
    add(2, "два две двух двум двумя")
    add(3, "три трех трем тремя")
    add(4, "четыре четырех четырем четырьмя")
    for v, b in ((5, "пят"), (6, "шест"), (9, "девят"), (10, "десят"), (11, "одиннадцат"), (12, "двенадцат"),
                 (13, "тринадцат"), (14, "четырнадцат"), (15, "пятнадцат"), (16, "шестнадцат"), (17, "семнадцат"),
                 (18, "восемнадцат"), (19, "девятнадцат"), (20, "двадцат"), (30, "тридцат")):
        add(v, f"{b}ь {b}и {b}ью")
    add(7, "семь семи семью")
    add(8, "восемь восьми восемью восьмью")
    add(40, "сорок сорока")
    add(90, "девяносто девяноста")
    add(100, "сто ста")
    for v, a, b in ((50, "пять", "пяти"), (60, "шесть", "шести"), (70, "семь", "семи"), (80, "восемь", "восьми")):
        add(v, f"{a}десят {b}десяти {a}юдесятью")
    add(200, "двести двухсот двумстам двумястами двухстах")
    add(300, "триста трехсот тремстам тремястами трехстах")
    add(400, "четыреста четырехсот четыремстам четырьмястами четырехстах")
    for v, a, b in ((500, "пять", "пяти"), (600, "шесть", "шести"), (700, "семь", "семи"),
                    (800, "восемь", "восьми"), (900, "девять", "девяти")):
        add(v, f"{a}сот {b}сот {b}стам {a}юстами {b}стах")
    add(1.5, "полтора полторы полутора")
    add(1000, "тысяча тысячи тысяче тысячу тысячей тысячею тысяч тысячам тысячами тысячах")
    for v, b in ((1_000_000, "миллион"), (1_000_000_000, "миллиард")):
        add(v, " ".join(b + e for e in ("", "а", "у", "ом", "е", "ы", "ов", "ам", "ами", "ах")))
    return d


CARDINALS = _lexicon()
MULTIPLIERS = (1000, 1_000_000, 1_000_000_000)
# a count was lost before these («…10 тысяч» heard as «тысяч»): nothing joins them
_PLURAL_MULT = {"тысяч", "тысячам", "тысячами", "тысячах", "миллионов", "миллионам", "миллионами", "миллионах",
                "миллиардов", "миллиардам", "миллиардами", "миллиардах"}
_ORD_STEMS = {"перв": 1, "втор": 2, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9,
              "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
              "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19,
              "двадцат": 20, "тридцат": 30, "сороков": 40, "пятидесят": 50, "шестидесят": 60, "семидесят": 70,
              "восьмидесят": 80, "девяност": 90, "сот": 100, "двухсот": 200, "трехсот": 300, "четырехсот": 400,
              "пятисот": 500, "шестисот": 600, "семисот": 700, "восьмисот": 800, "девятисот": 900,
              "тысячн": 1000, "двухтысячн": 2000, "миллионн": 1_000_000}
# ending -> what is written after the digits: «пятого» -> 5-го, «третьем» -> 3-м
_ORD_END = {"ый": "й", "ий": "й", "ой": "й", "ая": "я", "яя": "я", "ое": "е", "ее": "е", "ого": "го", "его": "го",
            "ому": "му", "ему": "му", "ым": "м", "им": "м", "ом": "м", "ем": "м", "ую": "ю", "юю": "ю", "ых": "х",
            "их": "х", "ыми": "ми", "ими": "ми", "ые": "е", "ие": "е", "ей": "й"}
_TRET_END = {"ий": "й", "ья": "я", "ье": "е", "ьи": "и", "ьего": "го", "ьему": "му", "ьей": "й", "ьим": "м",
             "ьем": "м", "ью": "ю", "ьих": "х", "ьими": "ми"}


def _bare(word: str) -> str:
    return word.lower().replace("ё", "е").strip(",.!?;:…»«\"()—–-")


def ordinal(word: str) -> tuple[int, str] | None:
    """A Russian ordinal: value and the ending written after the digits («третьего» -> (3, "го"))."""
    k = _bare(word)
    if k.startswith("трет"):
        e = _TRET_END.get(k[4:])
        return (3, e) if e else None
    for n in (3, 2):
        stem, e = k[:-n], _ORD_END.get(k[-n:])
        if e and stem in _ORD_STEMS:
            return _ORD_STEMS[stem], e
    return None


def is_percent(word: str) -> bool:
    k = _bare(word)
    return word.strip() == "%" or bool(re.fullmatch(r"процент[а-я]{0,3}|prozent", k))


# Number tokens a member writes in digits: «25-го», «2,50», «19%», a range «5–6», a time «7:30»
_DIGITS = re.compile(r"(\d+(?:[.,]\d+)?)(-?[а-я]{1,3})?(%)?")
_RANGE_D = re.compile(r"(\d+)(?:-?[а-я]{1,3})?[-–—](\d+)(-?[а-я]{1,3})?(%)?")  # «25-й—26-й»
_TIME_D = re.compile(r"([01]?\d|2[0-3]):([0-5]\d)")
_SUFFIX = re.compile(r"-[а-яё]{1,3}[,.!?;:…]?")
# a run of separate numbers after these is dictated digits («телефон ноль один семь шесть»)
DICTATION_CUES = re.compile(r"^(телефон|номер|индекс|hausnummer|postleitzahl|"
                            r"telefon|nummer|handynummer)")
TIME_PREP = {"в", "к", "до", "после", "около", "с", "со"}
_CURRENCY = re.compile(r"^(евро|рубл|доллар|франк|фунт|euro|€|\$)")
_NOT_TIME_NEXT = re.compile(r"^(евро|цент|рубл|доллар|франк|процент|тысяч|миллион|%)")


def _fmt(v: int | float) -> str:
    """Exact key of a value: «1234567», «2.5» (not 6 significant digits)."""
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{round(v, 9):.15g}"
    return str(v)


def _item(word: str) -> dict | None:
    """One token as a number: {v, kind, card, pct, key?}. kind: d digits, w word, m multiplier word,
    r range, t time; card: a plain cardinal (may be glued into a range, a time or dictated digits);
    pct: written with «%»."""
    raw = word.lower().replace("ё", "е").strip(",.!?;:…»«\"()")
    if not raw:
        return None
    m = _TIME_D.fullmatch(raw)
    if m:
        return dict(kind="t", key=f"{int(m.group(1))}:{m.group(2)}", v=None, card=False, pct=False)
    m = _RANGE_D.fullmatch(raw)
    if m:
        return dict(kind="r", key=f"{int(m.group(1))}–{int(m.group(2))}", v=None, card=False, pct=bool(m.group(4)))
    m = _DIGITS.fullmatch(raw)
    if m:
        s = m.group(1)
        v = float(s.replace(",", ".")) if re.search(r"[.,]", s) else int(s)
        ds = s if isinstance(v, int) and s.startswith("0") else _fmt(v)  # dictated «0123» keeps its zero
        return dict(kind="d", v=v, ds=ds, card=isinstance(v, int) and not m.group(2) and not m.group(3),
                    pct=bool(m.group(3)))
    parts = re.split(r"[-–—]", raw)
    if len(parts) == 2 and all(parts) and not any(c.isdigit() for c in raw):  # «пять-шесть»
        a, b = number_value(parts[0]), number_value(parts[1])
        if isinstance(a, int) and isinstance(b, int) and a < b:
            return dict(kind="r", key=f"{a}–{b}", v=None, card=False, pct=False)
    v = number_value(word)
    if v is None:
        return None
    k = _bare(word)
    if CARDINALS.get(k) in MULTIPLIERS or k in _DE_MULT:
        return dict(kind="m", v=v, card=True, pct=False)
    return dict(kind="w", v=v, card=k in CARDINALS and isinstance(v, int), pct=False, ord=ordinal(k) is not None)


def _joins(prev: float, v: float) -> bool:
    """«двадцать пять», «сто сорок» form one number; «пять шесть», «двадцать ноль» do not."""
    if not v:
        return False
    if prev >= 100:
        return v < 100 and prev % 100 == 0
    if prev >= 20:
        return v < 10 and prev % 10 == 0
    return False


def _join(toks: list[dict]) -> dict:
    t = dict(toks[0])
    if len(toks) > 1:
        t["w"] = " ".join(x["w"] for x in toks)
        t["e"] = toks[-1].get("e", t.get("e"))
        ps = [x["p"] for x in toks if x.get("p") is not None]
        t["p"] = min(ps) if ps else None
        t["parts"] = [y for x in toks for y in (x.get("parts") or [x])]
    return t


def merge_numbers(tokens: list[dict]) -> list[dict]:
    """Glue runs of number words/digits into one token keyed by the value (t["k"] = "#25000").

    Each token: {"w": surface, ...}; merged tokens keep the parts in "parts". Only multiplier
    words multiply («две тысячи» = 2000, «двадцать 1000» stays two numbers), digits never join
    words except before a multiplier («2 тысячи»). «%» and «процент*» are one word with the key
    "%" («19%» = «19 процентов», «19% процентов» = «19%»), so the number itself votes alone.
    Next to each other plain numbers make a range («пять шесть человек» = «5–6»), a time after a
    preposition («в семь тридцать» = «7:30»), a price («два пятьдесят евро» = 2.5) or dictated
    digits after a cue word («номер ноль один семь» = 017)."""
    out: list[dict] = []
    meta: list[dict | None] = []
    run: list[tuple[dict, dict]] = []
    st = dict(total=0, cur=0, mult=float("inf"))

    def flush():
        if not run:
            return
        toks = [t for t, _ in run]
        its = [i for _, i in run]
        t = _join(toks)
        if len(its) == 1 and its[0]["kind"] in "rt":
            key, v = its[0]["key"], None
        elif len(its) == 1 and its[0]["kind"] == "d":
            key, v = its[0]["ds"], its[0]["v"]
        else:
            v = st["total"] + st["cur"]
            v = int(v) if isinstance(v, float) and v.is_integer() else v
            key = _fmt(v)
        t["k"] = f"#{key}"
        t["num"] = True
        out.append(t)
        meta.append(dict(v=v, ds=key, card=all(i["card"] for i in its) and isinstance(v, int),
                         mult=any(i["kind"] == "m" for i in its)))
        run.clear()
        st.update(total=0, cur=0, mult=float("inf"))

    def percent(t: dict) -> None:
        if out and out[-1].get("k") == "%" and not tail_punct(out[-1]["w"]):
            out[-1] = dict(_join([out[-1], t]), k="%")  # «19% процентов»: said once
        else:
            out.append(dict(t, k="%"))
            meta.append(None)

    def joins(it: dict) -> bool:
        last = run[-1][1]
        if it["kind"] in "rt" or last["kind"] in "rt":
            return False
        if last.get("ord"):
            return False  # an ordinal ends a number: «двадцатого третьего» is not 23
        if it["kind"] == "m":
            return last["kind"] != "m" and it["v"] < st["mult"] and (st["cur"] < 1000 or last["kind"] == "d")
        if isinstance(it["v"], float):
            return False
        if last["kind"] == "m":  # «две тысячи двадцать пять»; «10 тысяч 19%», «10 тысяч десяти» are two
            return (it["kind"] == "w" and it["v"] < min(st["mult"], 1000) and all(i["kind"] != "d" for _, i in run)
                    and not (len(run) == 1 and _bare(run[0][0]["w"]) in _PLURAL_MULT))
        if isinstance(last["v"], float) or "d" in (it["kind"], last["kind"]):
            return False  # 1,5 only before a multiplier; digits never join («тридцать две 32500»)
        return _joins(last["v"], it["v"])

    for t in tokens:
        if is_percent(t["w"]):
            flush()
            percent(t)
            continue
        if (_SUFFIX.fullmatch(t["w"]) and len(run) == 1 and run[0][1]["kind"] == "d"
                and not tail_punct(run[0][0]["w"])):  # «25 -го»: the ending split off by a model
            run.append((t, dict(run[0][1], card=False)))
            flush()
            continue
        it = _item(t["w"])
        if it is None:
            flush()
            out.append(t)
            meta.append(None)
            continue
        if run and not joins(it):
            flush()
        if it["pct"]:  # «19%,» -> «19» + «%,»
            i = t["w"].index("%")
            run.append((dict(t, w=t["w"][:i]), it))
            flush()
            percent(dict(t, w=t["w"][i:]))
            continue
        run.append((t, it))
        if it["kind"] == "m":
            st["total"] += (st["cur"] or 1) * it["v"]
            st["cur"], st["mult"] = 0, it["v"]
        elif it["kind"] in "dw":
            st["cur"] += it["v"]
        if tail_punct(t["w"]):  # "3 тысячи, 3 тысячи" are two numbers
            flush()
    flush()
    return _pairs(out, meta)


def _pairs(out: list[dict], meta: list[dict | None]) -> list[dict]:
    """Neighbouring plain numbers that are one thing: dictated digits, a range, a time (merge_numbers)."""
    res: list[dict] = []
    i = 0
    while i < len(out):
        if meta[i] is None or not meta[i]["card"]:
            res.append(out[i])
            i += 1
            continue
        j = i + 1  # a run of plain numbers, up to a sentence end
        while (j < len(out) and meta[j] is not None and meta[j]["card"]
               and not re.search(r"[.!?…]", tail_punct(out[j - 1]["w"]))):
            j += 1
        before = [x for x in out[max(0, i - 3):i] if not x.get("num")]
        cue = any(DICTATION_CUES.match(key(x["w"])) for x in before)
        seq, ms = out[i:j], meta[i:j]
        if cue and len(seq) >= 2 and not any(m["mult"] for m in ms):
            t = _join(seq)
            t["k"] = "#" + "".join(m["ds"] for m in ms)
            res.append(t)
            i = j
            continue
        if len(seq) == 2 and not tail_punct(seq[0]["w"]):
            a, b = ms[0]["v"], ms[1]["v"]
            prev = key(out[i - 1]["w"]) if i and not out[i - 1].get("num") else ""
            nxt = (out[j].get("k") or key(out[j]["w"])) if j < len(out) else ""
            k = None
            if prev in TIME_PREP and a <= 23 and 10 <= b <= 59 and not ms[0]["mult"] and not _NOT_TIME_NEXT.match(nxt):
                k = f"{a}:{b:02d}"
            elif (b > a > 0 and (b - a == 1 or b <= 2 * a) and a < 1000 and not ms[1]["mult"]
                  and not nxt.startswith("год")):  # «за двадцать пять двадцать шесть год» lists two years
                k = f"{a}–{b}"
            elif _CURRENCY.match(nxt) and 10 <= b <= 99 and not ms[0]["mult"]:
                k = _fmt(a + b / 100)  # «два пятьдесят евро» = 2,50
            if k:
                t = _join(seq)
                t["k"] = "#" + k
                res.append(t)
                i = j
                continue
        res.extend(seq)
        i = j
    return res


# ---------------------------------------------------------------- phonetics

_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r",
                 "s", "t", "u", "f", "h", "ts", "ch", "sh", "sh", "", "i", "", "e", "iu", "ia"]))
_DE = [("tsch", "ch"), ("sch", "sh"), ("ch", "h"), ("ck", "k"), ("ß", "s"), ("ä", "e"), ("ö", "o"),
       ("ü", "iu"), ("äu", "oi"), ("eu", "oi"), ("ei", "ai"), ("ie", "i"), ("tz", "ts"), ("z", "ts"),
       ("ph", "f"), ("qu", "kv"), ("w", "v"), ("y", "i"), ("x", "ks"), ("c", "k"), ("j", "i")]
_FOLD = str.maketrans({"z": "s", "v": "f", "b": "p", "d": "t", "g": "k", "j": "i", "y": "i", "e": "e"})


@lru_cache(maxsize=50000)
def phon(word: str) -> str:
    """Rough sound skeleton shared by Cyrillic and German spelling."""
    w = word.lower().replace("ё", "е")
    w = re.sub(r"[^\wäöüß]+", "", w)
    if CYRILLIC.search(w):
        w = "".join(_CYR.get(c, c) for c in w)
    else:
        w = re.sub(r"^s([pt])", r"sh\1", w)  # Spielplatz ~ шпильплац, Stadt ~ штадт
        w = re.sub(r"(?<=[aeiouäöü])h(?=[^aeiouäöü]|$)", "", w)  # silent h: Wohnung
        for a, b in _DE:
            w = w.replace(a, b)
    w = w.translate(_FOLD)
    # what a Russian ear does to German: "st"/"sp" inside compounds sound "шт"/"шп" too
    # (Vorort ~ форорт, Erholung ~ эрхолунг), and eu / ei / ай / ой / ей blur
    w = w.replace("sht", "st").replace("shp", "sp")
    w = re.sub(r"[aeo]i", "ai", w)
    w = re.sub(r"(.)\1+", r"\1", w)  # doubled letters
    return w


@lru_cache(maxsize=200000)
def phon_sim(a: str, b: str) -> float:
    pa, pb = phon(a), phon(b)
    if not pa or not pb:
        return 0.0
    if pa == pb:
        return 1.0
    return SequenceMatcher(None, pa, pb).ratio()


_morph = {"m": None, "tried": False}


def known_russian(word: str) -> bool | None:
    """True if the word is in the Russian dictionary (pymorphy3); None if unavailable."""
    if not _morph["tried"]:
        _morph["tried"] = True
        try:
            import pymorphy3

            _morph["m"] = pymorphy3.MorphAnalyzer()
        except Exception:  # noqa: BLE001
            _morph["m"] = None
    m = _morph["m"]
    if m is None:
        return None
    w = key(word)
    if not w or not CYRILLIC.search(w):
        return None
    # «где-то», «из-за», «во-первых»: the dictionary knows them with the hyphen, not as «гдето»
    h = re.sub(r"[^\w\-]+", "", word.lower().replace("ё", "е")).strip("-")
    if "-" in h and m.word_is_known(h):
        return True
    return bool(m.word_is_known(w))
