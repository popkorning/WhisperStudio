"""Retrieval memory (H9, docs/DECISIONS.md): reference texts as context for a new recording.

Markdown files (tools/build_memory.py --wiki) and library text that can be trusted (segments edited or
verified by hand, segments every model agreed on) are cut into passages of ~120 words and
embedded once by a small multilingual model on CPU (tools/build_memory.py -> HOME/memory/).
For a new recording, the first-pass merged text of every chunk (with its neighbours) is the
query: the passages most like it give the terms this part of the talk is probably about.

What it gives the term pass (whisper_studio/terms.py, `focus`):
- per chunk: the terms allowed there = the dictionary core (glossary, dictionary terms and their
  parts) + the learned terms found in the retrieved passages (no more matching every chunk
  against all learned words);
- the retrieved terms may get looser sound thresholds (`loose`);
- German nouns of the retrieved passages that are not on the learned list (Stadt) as extra
  terms for that chunk only.
And per-chunk prompts for recognition (tools/prompt_experiment.py --variant retrieved).

Disabled by default (extra.ensemble.memory or WHISPER_STUDIO_MEMORY_ON=1 turns it on), and a
no-op without the index or the model. Heavy (numpy, torch, transformers): imported lazily by
ensemble._terms_pass only. Search is a plain numpy dot product: a few thousand passages.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from .asr import text as T
from .config import HOME, MODELS_DIR

MODEL = "emb-multilingual-e5-small"   # intfloat/multilingual-e5-small, MIT, 118M params, 471 MB
MAX_TOKENS = 256
TOP_K = 6                 # passages per chunk query
MIN_SCORE = 0.80          # e5 cosine: below this a passage is not about the same thing
MARGIN = 0.04             # ...nor if it is this far below the chunk's best passage
REC_TOP_K = 12            # passages for the whole recording (mean of the chunk queries)
QUERY_KIND = os.environ.get("WS_MEM_QUERY", "passage")  # symmetric e5 use: both sides "passage: "
NEIGHBOURS = 1            # chunks on each side added to a chunk's query (chunks are ~20 s)
WORD = re.compile(r"[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]+(?:-[A-Za-zÄÖÜäöüß]+)*")
DE_SUFFIX = ("es", "en", "er", "e", "n", "s")


def memory_dir() -> Path:
    env = os.environ.get("WHISPER_STUDIO_MEMORY")
    return Path(env) if env else HOME / "memory"


def model_dir() -> Path:
    return MODELS_DIR / MODEL


def enabled(cfg: dict | None) -> bool:
    if os.environ.get("WHISPER_STUDIO_MEMORY_ON") == "1":
        return True
    return bool((cfg or {}).get("memory"))


# ---------------------------------------------------------------- embedding

class Embedder:
    """multilingual-e5 with plain transformers: mean pooling over the attention mask, L2-normed.
    e5 wants "query: " / "passage: " prefixes. CPU only, 4 threads."""

    def __init__(self, path: Path | None = None, threads: int = 4):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        p = str(path or model_dir())
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModel.from_pretrained(p).eval()

    def encode(self, texts: list[str], kind: str = "passage", batch: int = 16):
        import numpy as np

        torch = self.torch
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # similar lengths: less padding
        with torch.inference_mode():
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                enc = self.tok([f"{kind}: {texts[j]}" for j in idx], padding=True, truncation=True,
                               max_length=MAX_TOKENS, return_tensors="pt")
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                v = torch.nn.functional.normalize(v, dim=-1)
                out += list(zip(idx, v.numpy()))
        out.sort(key=lambda x: x[0])
        return np.stack([v for _, v in out]).astype("float32") if out else np.zeros((0, 384), "float32")


@lru_cache(maxsize=1)
def embedder() -> Embedder:
    return Embedder()


# ---------------------------------------------------------------- passages

def latin_keys(text: str) -> list[str]:
    """Keys of the Latin words of a text, compounds also as their parts."""
    out = []
    for w in WORD.findall(text):
        out.append(T.key(w))
        if "-" in w:
            out += [T.key(p) for p in w.split("-") if len(p) >= 4]
    return out


def split_passages(text: str, size: int = 120, overlap: int = 20) -> list[str]:
    """Paragraphs packed into passages of about `size` words; a longer paragraph is cut into
    windows that overlap by `overlap` words."""
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n|\n(?=#)", text)]
    out, cur = [], []
    for p in paras:
        ws = p.split()
        if not ws:
            continue
        if len(cur) + len(ws) > size and cur:
            out.append(" ".join(cur))
            cur = []
        if len(ws) > size:
            for i in range(0, len(ws), size - overlap):
                out.append(" ".join(ws[i:i + size]))
                if i + size >= len(ws):
                    break
            continue
        cur += ws
    if cur:
        out.append(" ".join(cur))
    return [x for x in out if len(x.split()) >= 5]


class Index:
    """HOME/memory: passages.json ({src, rec, kind, text, latin, de}) + vectors.npy."""

    def __init__(self, folder: Path | None = None):
        import numpy as np

        folder = folder or memory_dir()
        self.folder = folder
        self.meta = json.loads((folder / "meta.json").read_text("utf-8"))
        self.passages = json.loads((folder / "passages.json").read_text("utf-8"))
        self.vecs = np.load(folder / "vectors.npy").astype("float32")

    def search(self, q, k: int = TOP_K, exclude: set[str] | None = None) -> list[tuple[float, int]]:
        """Best passages for a query vector (or a matrix of them: one list per row)."""
        import numpy as np

        sims = self.vecs @ np.asarray(q, dtype="float32").T
        single = sims.ndim == 1
        if single:
            sims = sims[:, None]
        if exclude:
            bad = [i for i, p in enumerate(self.passages) if p.get("rec") and p["rec"] in exclude]
            sims[bad, :] = -1.0
        res = []
        for col in range(sims.shape[1]):
            s = sims[:, col]
            top = np.argpartition(-s, min(k, len(s) - 1))[:k]
            res.append(sorted(((float(s[i]), int(i)) for i in top), reverse=True))
        return res[0] if single else res


@lru_cache(maxsize=2)
def _index(folder: str) -> Index:
    return Index(Path(folder))


def index() -> Index | None:
    f = memory_dir()
    if not (f / "vectors.npy").exists() or not model_dir().exists():
        return None
    return _index(str(f))


# ---------------------------------------------------------------- terms of passages

class TermMap:
    """Latin word key -> the terms of the list of its family: the same word in another German
    form (Anmeldungen / Anmeldung, Bahnhöfe / Bahnhof) or a compound part."""

    def __init__(self, terms: list[str]):
        self.by_root: dict[str, list[str]] = {}
        self.by_key: dict[str, str] = {}
        for t in terms:
            k = T.key(t)
            self.by_key.setdefault(k, t)
            fam = self.by_root.setdefault(root(k), [])
            if t not in fam:
                fam.append(t)

    def find(self, k: str) -> list[str]:
        return self.by_root.get(root(k), [])


def root(k: str) -> str:
    """A German word key without its plural/case ending (at least 5 letters left)."""
    for suf in DE_SUFFIX:
        if k.endswith(suf) and len(k) - len(suf) >= 5:
            return k[: -len(suf)]
    return k


def passage_terms(p: dict, tmap: TermMap) -> set[str]:
    return {t for k in p.get("latin", ()) for t in tmap.find(k)}


class Focus:
    """Per time span: which terms the term pass may use there and which got there by retrieval
    (terms.recover(focus=...)). spans: [(start, end, allowed keys, retrieved keys)]."""

    def __init__(self, spans: list[tuple[float, float, frozenset, frozenset]], core: set[str],
                 extra_terms: list[str], report: list[dict]):
        self.spans = spans
        self.core = frozenset(core)
        self.extra_terms = extra_terms
        self.report = report

    def at(self, s: float, e: float) -> tuple[frozenset, frozenset]:
        mid = (s + e) / 2
        for a, b, allowed, got in self.spans:
            if a <= mid < b:
                return allowed, got
        return self.core, frozenset()


def content_words(text: str) -> str:
    """The words that carry the topic: Russian nouns, adjectives and infinitives in their dictionary
    form, Latin words and words Russian doesn't know (a garbled term: «анмельдунге»). Without
    this the queries find passages that are *spoken* like the call (other transcripts), not passages about the same thing (docs/DECISIONS.md)."""
    T.known_russian("дом")
    m = T._morph["m"]
    if m is None:
        return text
    out = []
    for w in text.split():
        k = T.key(w)
        if not k or k[0].isdigit():
            continue
        if T.LATIN.search(w):
            out.append(w)
            continue
        pa = m.parse(k)[0]
        if pa.tag.POS in ("NOUN", "ADJF", "INFN") and pa.score > 0.2 and len(k) > 3:
            out.append(pa.normal_form)
        elif not m.word_is_known(k):
            out.append(w)
    return " ".join(out)


def chunk_texts(segs: list[dict], chunks: list[dict]) -> list[str]:
    """The merged words that fall into each chunk's own time range."""
    out = [[] for _ in chunks]
    for s in segs:
        for w in s.get("words") or []:
            mid = (w.get("s", 0) + w.get("e", 0)) / 2
            for ci, ch in enumerate(chunks):
                a, b = ch.get("own") or (ch["start"], ch["end"])
                if a <= mid < b:
                    out[ci].append(w["w"])
                    break
    return [" ".join(x) for x in out]


def retrieve(segs: list[dict], chunks: list[dict], tl: list[str], core: list[str],
             exclude: set[str] | None = None, idx: Index | None = None, emb: Embedder | None = None,
             extras: bool = True) -> Focus | None:
    """Focus for a recording from its first-pass merge. tl: every term the pass knows (core +
    learned); core: always allowed. exclude: recordings (source paths, lower case) whose own
    text must not be used (evaluation; a new recording is not in the index anyway)."""
    import numpy as np

    idx = idx or index()
    if idx is None or not chunks:
        return None
    emb = emb or embedder()
    texts = chunk_texts(segs, chunks)
    content = [content_words(t) for t in texts]
    queries = []
    for ci in range(len(chunks)):
        lo, hi = max(0, ci - NEIGHBOURS), min(len(chunks), ci + NEIGHBOURS + 1)
        queries.append(" ".join(t for t in content[lo:hi] if t) or "-")
    q = emb.encode(queries, QUERY_KIND)
    tmap = TermMap(tl)
    core_k = {T.key(t) for t in core}
    # the whole recording: one topic usually ("одна беседа - одна тема")
    rec_q = q.mean(0)
    rec_q /= max(np.linalg.norm(rec_q), 1e-6)
    rec_hits = idx.search(rec_q, REC_TOP_K, exclude)
    rec_terms: dict[str, float] = {}
    for sc, i in rec_hits:
        if sc >= MIN_SCORE:
            for t in passage_terms(idx.passages[i], tmap):
                rec_terms[t] = rec_terms.get(t, 0.0) + 0.5 * sc  # the topic of the whole talk counts half
    hits = idx.search(q, TOP_K, exclude)
    spans, report, extra = [], [], {}
    for ci, ch in enumerate(chunks):
        a, b = ch.get("own") or (ch["start"], ch["end"])
        hs = hits[ci]
        best = hs[0][0] if hs else 0.0
        use = [(sc, i) for sc, i in hs if sc >= MIN_SCORE and sc >= best - MARGIN] if texts[ci].strip() else []
        weight = dict(rec_terms)
        de_count: dict[str, int] = {}
        for sc, i in use:
            p = idx.passages[i]
            for t in passage_terms(p, tmap):
                weight[t] = weight.get(t, 0.0) + sc
            for w in p.get("de", ()):
                if not tmap.find(T.key(w)):
                    de_count[w] = de_count.get(w, 0) + 1
        new = [w for w, n in de_count.items() if n >= 2] if extras else []  # in two passages at least
        for w in new:
            extra[T.key(w)] = w
            weight[w] = de_count[w] * 0.5
        got = set(weight)
        got_k = frozenset({T.key(t) for t in got} | {T.key(w) for w in new})
        spans.append((a - 0.5, b + 0.5, frozenset(core_k | got_k), got_k - core_k))
        report.append(dict(chunk=ci, query=queries[ci][:300], best=round(best, 3),
                           passages=[dict(score=round(sc, 3), src=idx.passages[i]["src"],
                                          text=idx.passages[i]["text"][:160]) for sc, i in use[:3]],
                           terms=sorted(got), new=sorted(new),
                           ranked=[t for t, _ in sorted(weight.items(), key=lambda kv: -kv[1])]))
    return Focus(spans, core_k, sorted(extra.values()), report)


def prompt_terms(focus: Focus, ci: int, first: list[str] | None = None, n: int = 16) -> list[str]:
    """Up to n terms for a chunk's recognition prompt: `first` (what the CPU term pass found
    there), then the retrieved ones, most weighted first (named by more and closer passages);
    one form of each word."""
    out: list[str] = []
    seen = set()
    for t in [*(first or []), *focus.report[ci]["ranked"]]:
        k = root(T.key(t))  # one form of a word: Wohnung, not also Wohnungen
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out[:n]
