"""Retrieval memory index (H9, whisper_studio/memory.py, docs/DECISIONS.md): the owner's texts
cut into passages and embedded by multilingual-e5-small on CPU -> HOME/memory/ (or --out).

Sources (all read-only):
- a folder of Markdown notes (--wiki or WS_WIKI): code blocks, links and front matter dropped; machine
  transcripts saved there (paths with "transcripts") left out - not checked by ear, an error loop;
- library segments edited or verified by hand, and (--agree P) segments every word of which has
  agreement p >= P - grouped per document into passages, tagged with the recording (source path)
  so an evaluation can leave a recording's own text out.
Gold documents and every document of the same recording as a gold one are never used (as in
tools/build_terms.py). --before YYYY-MM-DD drops wiki files changed after that day.

    set CUDA_VISIBLE_DEVICES=-1
    ws-python tools/build_memory.py --out %WS_REAL_HOME%\\tmp\\agent-memory\\memory
    ws-python tools/build_memory.py --query "Anmeldung Vorort Rathaus"   # search the index
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_terms as BT  # noqa: E402

from whisper_studio import memory, store  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import LIBRARY_DIR  # noqa: E402

SKIP_PARTS = (".dist-info", "transcripts")  # package metadata; machine transcripts saved into the notes
FRONT = re.compile(r"\A---\n.*?\n---\n", re.S)


def clean_md(txt: str) -> str:
    txt = FRONT.sub("", txt)
    txt = re.sub(r"```.*?```", " ", txt, flags=re.S)
    txt = re.sub(r"https?://\S+", " ", txt)
    txt = re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", r"\1", txt)  # [[wiki link|alias]]
    txt = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", txt)        # [text](link)
    txt = re.sub(r"[`*_>|#]+", " ", txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt


def german_nouns(text: str) -> list[str]:
    """Capitalized German-looking words of a passage (BT.german_score), once each."""
    out = []
    for w in dict.fromkeys(BT.WORD.findall(text)):
        if w[:1].isupper() and not w.isupper() and len(w) >= 5 and BT.german_score(w) >= 2:
            out.append(w)
    return out


def passage(src: str, kind: str, text: str, rec: str | None = None) -> dict:
    return dict(src=src, kind=kind, rec=rec, text=text, latin=sorted(set(memory.latin_keys(text))),
                de=german_nouns(text))


def wiki_passages(root: Path, before: datetime | None) -> list[dict]:
    out = []
    for p in BT.wiki_files(root, before):
        rel = str(p.relative_to(root)).replace("\\", "/")
        if any(x in rel for x in SKIP_PARTS):
            continue
        try:
            txt = clean_md(p.read_text("utf-8", errors="ignore"))
        except OSError:
            continue
        for t in memory.split_passages(txt):
            out.append(passage(rel, "wiki", t))
    return out


def library_passages(agree: float | None) -> list[dict]:
    gids, gpaths = BT.gold_recordings()
    out, seen = [], set()
    for f in sorted(LIBRARY_DIR.glob("*/doc.json")):
        try:
            d = json.loads(f.read_text("utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rec = str(d.get("source_path") or d.get("id")).lower()
        if d.get("id") in gids or d.get("gold") or rec in gpaths:
            continue
        cur: list[str] = []
        kind = "edited"
        parts = []
        for s in d.get("segments") or []:
            ws = s.get("words") or []
            hand = store.by_owner(s) or s.get("verified")
            sure = agree is not None and len(ws) >= 6 and min(w.get("p", 1) for w in ws) >= agree
            if hand or sure:
                cur.append(s["text"])
                kind = "edited" if hand else kind
                continue
            if cur:
                parts.append((kind, " ".join(cur)))
            cur, kind = [], "agree"
        if cur:
            parts.append((kind, " ".join(cur)))
        for kind, txt in parts:
            for t in memory.split_passages(txt):
                if (rec, t) in seen:  # the same recording recognized twice
                    continue
                seen.add((rec, t))
                out.append(passage(d["id"][-6:], kind, t, rec))
    return out


def build(out: Path, wiki: Path | None, before: datetime | None, agree: float | None) -> None:
    t0 = time.monotonic()
    ps = (wiki_passages(wiki, before) if wiki and wiki.exists() else []) + library_passages(agree)
    print(f"{len(ps)} passages ({sum(p['kind'] == 'wiki' for p in ps)} wiki), "
          f"{sum(len(p['text'].split()) for p in ps)} words, {time.monotonic() - t0:.0f} s")
    emb = memory.Embedder()
    t1 = time.monotonic()
    import numpy as np

    vecs = emb.encode([p["text"] for p in ps], "passage", batch=32)
    print(f"embedded in {time.monotonic() - t1:.0f} s")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vecs.astype("float16"))
    (out / "passages.json").write_text(json.dumps(ps, ensure_ascii=False), "utf-8")
    (out / "meta.json").write_text(json.dumps(dict(
        built=datetime.now().isoformat(timespec="seconds"), model=memory.MODEL, passages=len(ps),
        wiki=str(wiki) if wiki else None, wiki_before=before.date().isoformat() if before else None,
        agree=agree, kinds={k: sum(p["kind"] == k for p in ps) for k in ("wiki", "edited", "agree")},
    ), ensure_ascii=False, indent=1), "utf-8")
    print("->", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wiki", default=str(BT.WIKI) if str(BT.WIKI) != "." else "")
    ap.add_argument("--no-wiki", action="store_true")
    ap.add_argument("--before", help="только файлы вики, изменённые до этой даты (YYYY-MM-DD)")
    ap.add_argument("--agree", type=float, default=0.9, help="реплики библиотеки, где у всех слов p >= этого")
    ap.add_argument("--no-agree", action="store_true", help="из библиотеки только правленое руками")
    ap.add_argument("--out", default=str(memory.memory_dir()))
    ap.add_argument("--query", action="append", default=[], help="поиск по готовому индексу")
    a = ap.parse_args()
    if a.query:
        os.environ["WHISPER_STUDIO_MEMORY"] = a.out
        idx = memory.Index(Path(a.out))
        emb = memory.Embedder()
        for q in a.query:
            print(f"\n{q}")
            for sc, i in idx.search(emb.encode([q], "query")[0], 8):
                p = idx.passages[i]
                print(f"  {sc:.3f} {p['src'][:60]:<60} {p['text'][:110]}")
        return
    build(Path(a.out), None if a.no_wiki or not a.wiki else Path(a.wiki),
          datetime.fromisoformat(a.before) if a.before else None, None if a.no_agree else a.agree)


if __name__ == "__main__":
    main()
