"""Learned term list: German words from the owner's own texts, for post-merge term recovery
(whisper_studio/terms.py). Sources (all read-only):

- the project dictionary (dictionary.json terms; compounds also give their parts:
  Hotel-Anmeldung -> Anmeldung),
- a markdown notes folder (--wiki or WS_WIKI; node_modules and tool folders skipped),
- library transcripts: Latin words of segments the owner typed or confirmed (store.by_owner:
  AI proofreading, dictionary rules and find & replace do not count).

Gold documents and every document of the same recording as a gold one are left out, so the
gold reference stays an honest test (docs/DECISIONS.md). --before YYYY-MM-DD also drops wiki
files changed after that day.

    ws-python tools/build_terms.py --out %WS_REAL_HOME%\\tmp\\agent-terms\\terms-learned.json
    ws-python tools/build_terms.py --check a1b2c3     # coverage of a gold doc's Latin words
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import settings, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import HOME, LIBRARY_DIR  # noqa: E402

WIKI = Path(os.environ.get("WS_WIKI", ""))  # a markdown notes folder, optional
SKIP_DIRS = {"node_modules", ".git", ".obsidian", ".venv", "site-packages"}
WORD = re.compile(r"(?<![\w@/.#-])[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]+(?:-[A-ZÄÖÜa-zäöüß][A-Za-zÄÖÜäöüß]+)*(?![\w@/-])")

# German-looking: umlauts, sch/tz/pf, typical noun and adjective endings
DE_END = re.compile(r"(ung|ungen|heit|keit|schaft|ismus|nahme|gabe|stelle|amt|ämter|bericht|anlage|anlagen|"
                    r"haus|platz|straße|strasse|bahn|hof|garten|nummer|meldung|karte|"
                    r"bereich|wesen|stand|zeit|schule|weg|berg|"
                    r"ordnung|gesetz|pflicht|lich|isch|ig|chen|lein|er|en)$")
DE_MARK = re.compile(r"[äöüßÄÖÜ]|sch|tz|pf|ei|eu|ie|ch")
EN_HINT = re.compile(r"(tion|ing|ness|ship|ment|ity|ous|ive|able|ally|ware|ed|ly)$|^(th|wh)|w[aeiou]rk|oo|ck$|"
                     r"^(however|after|better|another|rather|even|answer|between|written|order|super|young|"
                     r"number|faster|helper|runner|manager|filter|kitchen|browser|config|docker|installer)$")
STOP = {
    # English and code words frequent in Markdown notes
    "the", "and", "for", "with", "from", "this", "that", "true", "false", "none", "null", "json", "http",
    "https", "html", "markdown", "python", "agent", "agents", "claude", "codex", "status", "update",
    "created", "updated", "title", "source", "sources", "wiki", "index", "readme", "draft", "final",
    "check", "verify", "query", "edit", "create", "template", "output", "outputs", "input", "file",
    "files", "page", "pages", "link", "links", "date", "type", "name", "tags", "summary", "notes",
    "open", "done", "todo", "data", "test", "user", "server", "client", "token", "model", "models",
    "online", "offline", "email", "mail", "phone", "google", "apple", "android", "windows",
    # German function words: not terms
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "und", "oder",
    "nicht", "mit", "von", "vom", "zum", "zur", "für", "auf", "aus", "bei", "nach", "über", "unter",
    "ist", "sind", "wird", "werden", "wurde", "haben", "hat", "kann", "können", "muss", "müssen", "soll",
    "sie", "wir", "ich", "ihr", "ihre", "ihren", "ihrem", "unser", "unsere", "sehr",
    "bitte", "danke", "mit", "als", "auch", "noch", "nur", "wie",
    "was", "wenn", "dass", "sich", "dieser", "diese", "dieses", "keine", "kein", "alle", "sowie", "bis",
    "durch", "gegen", "ohne", "um", "zu", "im", "am", "an", "in", "es", "er", "so", "da", "hier", "dort",
}
ASCII_DE = [("ae", "ä"), ("oe", "ö"), ("ue", "ü")]


def german_score(w: str) -> float:
    lw = w.lower()
    if lw in STOP or len(lw) < 4 or not T.LATIN.search(w):
        return 0.0
    s = 0.0
    if re.search(r"[äöüß]", lw):
        s += 2
    m = DE_END.search(lw)
    if m:
        s += 0.5 if m.group(1) in ("er", "en", "ig") else 1
    s += 0.5 * min(2, len(DE_MARK.findall(lw)))
    if w[:1].isupper() and not w.isupper():
        s += 0.5  # German nouns are capitalized
    if EN_HINT.search(lw):
        s -= 1.5
    if w.isupper() and len(w) <= 5:
        s -= 1  # acronyms: only from the dictionary
    return s


def wiki_files(root: Path, before: datetime | None):
    for p in root.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if before and datetime.fromtimestamp(p.stat().st_mtime) >= before:
            continue
        yield p


def gold_recordings() -> tuple[set[str], set[str]]:
    ids, paths = set(), set()
    for s in store.list_docs():
        if s.get("gold"):
            d = store.load(s["id"])
            ids.add(d["id"])
            if d.get("source_path"):
                paths.add(str(d["source_path"]).lower())
    return ids, paths


def library_words() -> Counter:
    gids, gpaths = gold_recordings()
    c = Counter()
    for f in sorted(LIBRARY_DIR.glob("*/doc.json")):
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if d.get("id") in gids or d.get("gold") or str(d.get("source_path") or "").lower() in gpaths:
            continue
        for s in d.get("segments") or []:
            if store.by_owner(s):
                for w in WORD.findall(s.get("text", "")):
                    c[w] += 1
    return c


def parts(term: str) -> list[str]:
    """Compound parts worth a term of their own: Hotel-Anmeldung -> Anmeldung."""
    out = []
    if " " in term.strip():
        return out  # names of several words (Gartenhaus Sonnenberg) are not split
    for p in term.split("-"):
        if len(p) >= 6 and T.LATIN.search(p) and p != term:
            out.append(p)
    return out


def build(wiki: Path | None, before: datetime | None, min_count: int, limit: int) -> dict:
    dict_terms = [t for t in settings.dictionary()["terms"] if T.LATIN.search(t)]
    counts: Counter = Counter()
    files: Counter = Counter()
    if wiki and wiki.exists():
        for p in wiki_files(wiki, before):
            try:
                txt = p.read_text("utf-8", errors="ignore")
            except OSError:
                continue
            txt = re.sub(r"```.*?```", " ", txt, flags=re.S)  # code blocks
            txt = re.sub(r"https?://\S+", " ", txt)
            txt = re.sub(r"[_/\\]", " ", txt)  # file names
            seen = set()
            for w in WORD.findall(txt):
                counts[w] += 1
                seen.add(w)
            for w in seen:
                files[w] += 1
    lib = library_words()
    # one spelling per word: the most frequent one; ASCII "ue" spellings fold into umlauts
    forms: dict[str, Counter] = {}
    for src in (counts, lib):
        for w, n in src.items():
            k = w.lower()
            for a, b in ASCII_DE:
                k = k.replace(a, b)
            forms.setdefault(k, Counter())[w] += n
    learned = []
    for c in forms.values():
        n = sum(c.values())
        nf = sum(files.get(x, 0) for x in c)
        # file names write "fuer": the umlaut spelling wins if it occurs at all
        um = [x for x, _ in c.most_common() if re.search(r"[äöüß]", x)]
        caps = [x for x, _ in c.most_common() if x[:1].isupper()]
        w = um[0] if um else (caps[0] if caps else c.most_common(1)[0][0])
        sc = german_score(w)
        if sc < 2 or n < min_count or (nf < 2 and not any(lib.get(x) for x in c)):
            continue
        # nouns only (capitalized nearly always): verbs and adjectives (könnte, möglich, geprüft)
        # sound like too many Russian words and are rarely what a model gets wrong
        if sum(v for x, v in c.items() if x[:1].isupper()) < 0.8 * n or len(w) < 5:
            continue
        learned.append((w, n, nf, sc))
    learned.sort(key=lambda x: (-x[1] * x[3], x[0]))
    learned = learned[:limit]
    terms: list[str] = []
    for t in dict_terms + [p for t in dict_terms for p in parts(t)] + [w for w, *_ in learned]:
        if t.lower() not in {x.lower() for x in terms}:
            terms.append(t)
    return dict(
        built=datetime.now().isoformat(timespec="seconds"),
        sources=dict(dictionary=len(dict_terms), wiki=str(wiki) if wiki else None,
                     wiki_before=before.date().isoformat() if before else None,
                     library_edited_words=sum(lib.values()), learned=len(learned)),
        terms=terms,
        counts={w: n for w, n, *_ in learned},
    )


def coverage(res: dict, doc_id: str) -> None:
    from whisper_studio import gold

    d = gold.Doc(doc_id)
    d.set_reference()
    words = [w["w"] for ci in d.chunks for w in d.ref_words[ci] if T.is_latin(w["w"])]
    keys = {}
    for t in res["terms"]:
        keys.setdefault(T.key(t), t)
        for p in re.split(r"[-\s]+", t):
            keys.setdefault(T.key(p), t)
    hit = 0
    print(f"\nЛатиница эталона {doc_id}: {len(words)} слов")
    for w in words:
        k = gold.MIXED_TAIL.sub("", T.key(w))
        m = keys.get(k) or next((t for kk, t in keys.items() if len(k) >= 6 and (kk.startswith(k) or k.startswith(kk))
                                 and abs(len(kk) - len(k)) <= 2), None)
        hit += m is not None
        print(f"  {w:<20} {'+ ' + m if m else '-'}")
    print(f"покрыто {hit}/{len(words)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki", default=str(WIKI) if str(WIKI) != "." else "")
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--before", help="только файлы вики, изменённые до этой даты (YYYY-MM-DD)")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--out", default=str(HOME / "terms-learned.json"))
    ap.add_argument("--check", action="append", default=[], help="эталон: какие его латинские слова покрыты")
    ap.add_argument("--dry", action="store_true", help="не записывать файл")
    a = ap.parse_args()
    before = datetime.fromisoformat(a.before) if a.before else None
    res = build(None if a.no_wiki or not a.wiki else Path(a.wiki), before, a.min_count, a.limit)
    print(json.dumps(res["sources"], ensure_ascii=False), "terms:", len(res["terms"]))
    if not a.dry:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")
        print("->", a.out)
    for g in a.check:
        coverage(res, g)


if __name__ == "__main__":
    main()
