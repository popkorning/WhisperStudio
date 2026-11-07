"""GPU experiment (H6, docs/DECISIONS.md): recognition-side biasing with a per-chunk term prompt.
Re-recognizes only the gold chunks of a document with the chosen members and prompts, into a
COPY of hypotheses.json outside the library, then evaluates against the gold.
The library is never written.

    ws-python tools/prompt_experiment.py --ref a1b2c3 --variant whisper-terms --dry   # CPU: prompts only
    ws-python tools/prompt_experiment.py --ref a1b2c3 --variant whisper-terms         # GPU
    ws-python tools/prompt_experiment.py --ref a1b2c3 --variant all --out %WS_REAL_HOME%\\tmp\\agent-terms\\gpu
    ws-python tools/prompt_experiment.py --ref a1b2c3 --eval-only --out ...            # evaluate a finished copy

Variants (members re-run on the gold chunks only; the rest of the cache stays as recognized):
  whisper-tail   large-v3 + ru-large-v3: today's prompt (dictionary list, cut at 600 chars) but the
                 previous words limited so the term list is never cut (faster-whisper keeps only
                 the LAST 223 prompt tokens: today the tail pushes the first ~8 terms out)
  whisper-terms  large-v3 + ru-large-v3: per-chunk prompt - a short Russian sentence with the
                 terms this chunk probably contains (found by the CPU term pass on the existing
                 hypotheses, in this chunk and its neighbours, plus the recording's most frequent
                 ones), then the previous words within the 223-token budget
  qwen-context   qwen3-asr: the same per-chunk terms as its system-prompt context (Qwen3ASR
                 processor `prompt` = "context/hotwords"); no context on chunks with < 2 s of
                 speech (with the whole glossary it recites it on near-silence), and an answer that
                 recites >= 3 prompt terms in a row is dropped for the one without context
  de-mixed       de-turbo with language=None and the chunk prompt on every chunk, stored as
                 hyps["de-mixed"]: evidence only (ensemble.EVIDENCE_ONLY), it writes Russian in
                 Cyrillic and German terms in Latin (e.g. «Hauptbahnhof»)
  all            whisper-terms + qwen-context + de-mixed
  retrieved      large-v3 + ru-large-v3 + qwen3-asr: per-chunk terms from the retrieval memory
                 (H9, whisper_studio/memory.py, docs/DECISIONS.md): what the CPU term pass found
                 in the chunk first, then the terms of the owner's passages most like the chunk's
                 first-pass text; Whisper gets them within RETRIEVED_TOKENS (the rest of the 223 is
                 the previous words), Qwen the first QWEN_TERMS as context (rules as qwen-context).
                 Needs the index: WHISPER_STUDIO_MEMORY=<folder of tools/build_memory.py>

Output: OUT/<variant>/hypotheses.json (+ prompts.json) and the evaluation: merge errors and Latin
errors with and without the term pass, against the cache as it is in the library.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, store, terms  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402
from whisper_studio.config import HOME, LIBRARY_DIR, MODELS_DIR  # noqa: E402

OUT = HOME / "tmp" / "agent-terms" / "gpu"
BUDGET = 223            # faster-whisper: prompt tokens kept (max_length // 2 - 1), counted from the END
TERMS_PER_CHUNK = 12
QWEN_TERMS = 8
MIN_SPEECH_QWEN = 2.0   # s of VAD speech in a chunk before Qwen gets a context
VARIANTS = ("whisper-tail", "whisper-terms", "qwen-context", "de-mixed", "all", "retrieved")
RETRIEVED_TOKENS = 150  # Whisper prompt: the sentence with the terms at most this long
DE_MIXED = dict(name="de-mixed", backend="whisper", model="de-turbo", lang="auto", weight=0.0, cost=0.4)


# ---------------------------------------------------------------- prompts

def tokenizer():
    from tokenizers import Tokenizer

    for m in ("de-turbo", "large-v3", "ru-large-v3"):  # all Whisper v3 models share the tokenizer
        f = MODELS_DIR / m / "tokenizer.json"
        if f.exists():
            return Tokenizer.from_file(str(f))
    return None


def ntok(tok, text: str) -> int:
    return len(tok.encode(" " + text.strip()).ids) if tok else len(text) // 3


def chunk_terms(d: gold.Doc) -> tuple[dict[int, list[str]], list[str]]:
    """Terms the CPU pass finds (recovered or heard by >= 2 models) per chunk, best first, and
    the recording's most frequent ones. Uses only the hypotheses, never the reference."""
    opts = gold.options(d.doc)
    st = json.loads(json.dumps(d.st))
    ensemble.combine(st, opts)
    tl = terms.term_list(ensemble._glossary(opts), learned=bool(opts.dictionary))
    by_key = {T.key(t): t for t in tl}
    mem = {m["name"]: m for m in ensemble.members(opts)}
    words: dict[str, list[dict]] = {}
    for ci, ch in enumerate(st["chunks"]):
        for name, res in st["hyps"].items():
            m = mem.get(name)
            h = res[ci] if res and ci < len(res) else None
            if h is None or (m is None and name not in ensemble.EVIDENCE_ONLY) or (m and not ensemble._eligible(m, ch)):
                continue
            words.setdefault(name, []).extend(ensemble._tokens(h, ch, name))
    hot = terms.hotspots(terms.Index(tl), terms.Members(words))
    found: list[tuple[float, str]] = [(x["t"], x["new"]) for x in st.get("term_log") or []]
    for tk, spans in hot.items():
        found += [((a + b) / 2, by_key.get(tk, tk)) for a, b in spans]
    count: dict[str, int] = {}
    per: dict[int, list[str]] = {}
    for t, term in found:
        term = gold.MIXED_TAIL.sub("", term)  # «Anmeldungе» -> Anmeldung
        count[term] = count.get(term, 0) + 1
        for ci, ch in enumerate(st["chunks"]):
            a, b = ch.get("own") or (ch["start"], ch["end"])
            if a - 5.0 <= t < b + 5.0:  # this chunk and the edges of its neighbours
                per.setdefault(ci, []).append(term)
    top = [t for t, _ in sorted(count.items(), key=lambda kv: -kv[1])][:5]
    glossary = [x.strip() for x in (opts.glossary or "").replace(";", ",").split(",") if x.strip()]
    out = {}
    for ci in range(len(st["chunks"])):
        out[ci] = list(dict.fromkeys(glossary + per.get(ci, []) + top))[:TERMS_PER_CHUNK]
    return out, top


def retrieved_terms(d: gold.Doc, per: dict[int, list[str]], tok) -> dict[int, list[str]]:
    """Per chunk: the CPU pass's terms there, then the memory's, cut to RETRIEVED_TOKENS in the
    Whisper sentence. The query is the first-pass merge (with the term pass), as in the app."""
    from whisper_studio import memory

    opts = gold.options(d.doc)
    st = json.loads(json.dumps(d.st))
    segs = ensemble.combine(st, opts)
    tl = terms.term_list(ensemble._glossary(opts), learned=bool(opts.dictionary))
    core = terms.term_list(ensemble._glossary(opts), learned=False)
    f = memory.retrieve(segs, st["chunks"], tl, core, exclude={str(d.doc.get("source_path")).lower()})
    if f is None:
        sys.exit("нет индекса памяти: WHISPER_STUDIO_MEMORY=<папка tools/build_memory.py>")
    out = {}
    for ci in range(len(st["chunks"])):
        ts = memory.prompt_terms(f, ci, per.get(ci))
        while ts and ntok(tok, "Разговор по-русски с немецкими словами: " + ", ".join(ts) + ".") > RETRIEVED_TOKENS:
            ts = ts[:-1]
        out[ci] = ts
    return out


def whisper_prompt(tok, ts: list[str], tail: str) -> str:
    """A Russian sentence with the chunk's terms (as the speech would write them), then as much
    of the previous words as still fits into the 223 tokens Whisper keeps."""
    head = ("Разговор по-русски с немецкими словами: " + ", ".join(ts) + ".") if ts else ""
    room = BUDGET - ntok(tok, head) - 2
    words = tail.split()
    while words and ntok(tok, " ".join(words)) > room:
        words = words[1:]
    return " ".join(x for x in (head, " ".join(words)) if x)


def tail_prompt(tok, base: str, tail: str) -> str:
    room = BUDGET - ntok(tok, base) - 2
    words = tail.split()
    while words and ntok(tok, " ".join(words)) > room:
        words = words[1:]
    return " ".join(x for x in (base, " ".join(words)) if x)


def recites(text: str, ts: list[str]) -> bool:
    keys = [T.key(w) for w in text.split()]
    tk = {T.key(t) for t in ts}
    run = best = 0
    for k in keys:
        run = run + 1 if k in tk else 0
        best = max(best, run)
    return best >= 3


# ---------------------------------------------------------------- running

def run_member(m: dict, st: dict, audio, chunks: list[int], prompts: dict[int, str | None],
               lang_of, store_as: str | None = None, qwen_terms: dict | None = None) -> None:
    from whisper_studio import asr

    name = store_as or m["name"]
    res = st["hyps"].setdefault(name, [None] * len(st["chunks"]))
    b = asr.make(m)
    b.load(lambda msg: print("  ", msg))
    t0 = time.monotonic()
    try:
        for ci in chunks:
            c = st["chunks"][ci]
            x = ensemble._chunk_audio(audio, c, st["level"])
            lang = lang_of(m, c)
            h = b.recognize(x, lang, prompts.get(ci))
            if qwen_terms is not None and prompts.get(ci) and recites(h.get("text", ""), qwen_terms.get(ci, [])):
                h = b.recognize(x, lang, None)  # recited the context: the answer without it
                h["context_dropped"] = True
            h["called"] = lang
            h["prompt"] = prompts.get(ci)
            for w in h.get("words") or []:
                w["s"] = round(w["s"] + c["start"], 3)
                w["e"] = round(w["e"] + c["start"], 3)
            res[ci] = h
            print(f"  {name} chunk {ci}: {h.get('text', '')[:100]}")
    finally:
        b.unload()
    st["timing"][name + "-experiment"] = round(time.monotonic() - t0, 1)


def experiment(d: gold.Doc, variant: str, out: Path, dry: bool) -> Path:
    tok = tokenizer()
    per, top = chunk_terms(d)
    opts = gold.options(d.doc)
    fixed = None if (opts.mixed or not opts.language) else opts.language
    st = json.loads(json.dumps(d.st))
    chunks = list(d.chunks)
    base = ensemble.engine.prompt(opts)[0] or ""
    prev_text = {}
    for ci in chunks:
        res = st["hyps"].get("large-v3") or []
        prev = res[ci - 1] if ci and ci - 1 < len(res) else None
        near = ci and st["chunks"][ci]["start"] - st["chunks"][ci - 1]["end"] < 3.0
        prev_text[ci] = (prev or {}).get("text", "")[-200:] if near else ""
    plan = []
    if variant in ("whisper-tail",):
        p = {ci: tail_prompt(tok, base, prev_text[ci]) for ci in chunks}
        plan += [(m, p, None, None) for m in ensemble.MEMBERS if m["name"] in ("large-v3", "ru-large-v3")]
    if variant in ("whisper-terms", "all"):
        p = {ci: whisper_prompt(tok, per[ci], prev_text[ci]) for ci in chunks}
        plan += [(m, p, None, None) for m in ensemble.MEMBERS if m["name"] in ("large-v3", "ru-large-v3")]
    if variant in ("qwen-context", "all"):
        qt = {ci: per[ci][:QWEN_TERMS] for ci in chunks}
        sp = {ci: sum(b - a for a, b in st["chunks"][ci].get("speech", [])) for ci in chunks}
        p = {ci: (", ".join(qt[ci]) if qt[ci] and sp[ci] >= MIN_SPEECH_QWEN else None) for ci in chunks}
        plan += [(m, p, None, qt) for m in ensemble.MEMBERS if m["name"] == "qwen3-asr"]
    if variant in ("de-mixed", "all"):
        p = {ci: whisper_prompt(tok, per[ci], "") for ci in chunks}
        plan.append((DE_MIXED, p, "de-mixed", None))
    if variant == "retrieved":
        rt = retrieved_terms(d, per, tok)
        per = {ci: rt.get(ci, []) for ci in per}
        p = {ci: whisper_prompt(tok, rt.get(ci, []), prev_text[ci]) for ci in chunks}
        plan += [(m, p, None, None) for m in ensemble.MEMBERS if m["name"] in ("large-v3", "ru-large-v3")]
        qt = {ci: rt.get(ci, [])[:QWEN_TERMS] for ci in chunks}
        sp = {ci: sum(b - a for a, b in st["chunks"][ci].get("speech", [])) for ci in chunks}
        qp = {ci: (", ".join(qt[ci]) if qt[ci] and sp[ci] >= MIN_SPEECH_QWEN else None) for ci in chunks}
        plan += [(m, qp, None, qt) for m in ensemble.MEMBERS if m["name"] == "qwen3-asr"]
    folder = out / variant
    folder.mkdir(parents=True, exist_ok=True)
    report = {str(ci): {"terms": per[ci], **{(s or m["name"]): p.get(ci) for m, p, s, _ in plan}} for ci in chunks}
    (folder / "prompts.json").write_text(json.dumps(dict(top=top, chunks=report), ensure_ascii=False, indent=1), "utf-8")
    for m, p, s, _ in plan:
        lens = [ntok(tok, x) for x in p.values() if x]
        print(f"{s or m['name']}: {len(chunks)} кусков, prompt токенов max {max(lens, default=0)}")
    if dry:
        print("dry run: подсказки ->", folder / "prompts.json")
        return folder
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        sys.exit("нужна видеокарта: CUDA_VISIBLE_DEVICES=-1")
    from whisper_studio import audio as A

    src = Path(d.doc.get("source_path") or "")
    if not src.exists():
        src = store.path(d.id) / d.doc["audio_file"]
    x = A.load(src)
    for m, p, s, qt in plan:
        lang_of = (lambda m_, c: None) if s == "de-mixed" else (lambda m_, c: ensemble._call_lang(m_, c, fixed))
        run_member(m, st, x, chunks, p, lang_of, s, qt)
    (folder / "hypotheses.json").write_text(json.dumps(st, ensure_ascii=False), "utf-8")
    return folder


# ---------------------------------------------------------------- evaluation

def evaluate(d: gold.Doc, st: dict) -> dict:
    out = {}
    for on in (False, True):
        opts = gold.options(d.doc)
        if not on:
            opts.extra = {**(opts.extra or {}), "ensemble": {**((opts.extra or {}).get("ensemble") or {}), "no_terms": True}}
        s2 = json.loads(json.dumps(st))
        segs = ensemble.combine(s2, opts)
        if opts.dictionary:
            from whisper_studio import settings

            store.apply_rules_to_segments(segs, settings.rules())
        per = d.by_chunk([w for s in segs for w in s["words"]])
        hyp = {ci: gold.norm([w["w"] for w in ws]) for ci, ws in per.items()}
        e, t = gold.wer(d.ref, hyp, d.chunks)
        le = ln = 0
        for ci in d.chunks:
            for r, h in gold.align(d.ref[ci], hyp.get(ci, [])):
                if r is not None and T.LATIN.search(r):
                    ln += 1
                    le += r != h
        out["terms" if on else "plain"] = dict(errors=e, words=t, latin_errors=le, latin=ln)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--variant", choices=VARIANTS, default="whisper-terms")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry", action="store_true", help="только подсказки, без распознавания (CPU)")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    d = gold.Doc(a.ref)
    d.set_reference()
    out = Path(a.out)
    if out.resolve().is_relative_to(LIBRARY_DIR.resolve()):
        sys.exit("копия кэша не может лежать в библиотеке")
    folder = out / a.variant
    if not a.eval_only:
        folder = experiment(d, a.variant, out, a.dry)
        if a.dry:
            return
    st = json.loads((folder / "hypotheses.json").read_text("utf-8"))
    before, after = evaluate(d, d.st), evaluate(d, st)
    print(f"\n{d.id[-6:]} {a.variant}: {before['plain']['words']} слов")
    for k in ("plain", "terms"):
        b, x = before[k], after[k]
        print(f"  {'без поиска терминов' if k == 'plain' else 'с поиском терминов'}: ошибок {b['errors']} -> {x['errors']}, "
              f"латиница {b['latin_errors']}/{b['latin']} -> {x['latin_errors']}/{x['latin']}")
    (folder / "result.json").write_text(json.dumps(dict(before=before, after=after), ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
