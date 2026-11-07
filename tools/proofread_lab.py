"""Local proofreading by a small LLM, measured on the gold (lab experiment 13). CPU/GPU, nothing
in the library changes.

The merged transcript of the gold is rebuilt from its stored hypotheses. Every segment with a
doubtful word (p < --p) goes to the model with: the text, what the other recognizers heard
(alts) and the project terms that sound closest to its doubtful words. The model returns the
corrected segment; llm.refused() keeps only fixes made of those words (no new words, numbers,
or rewrites). Errors on the gold before and after are counted per chunk, as tools/asr_eval does.

    ws-python tools/proofread_lab.py --ref a1b2c3 --model %WS_REAL_HOME%/models-lab/lfm3-1.7b
    ... --dry       no model: how many segments and candidates would be sent
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import ensemble, gold, llm, store, terms  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402

SYSTEM = """Ты исправляешь ошибки распознавания речи в телефонном разговоре по-русски с немецкими
словами. Немецкие слова пишутся латиницей (Stadt, Wohnung, Anmeldung).
Меняй только слова, помеченные ⟨…⟩, и только на слова из списков «другие модели слышали» или
«термины». Если не уверен — оставь как есть. Ничего не перефразируй, не добавляй и не убирай.
Ответь одной строкой: исправленный текст фрагмента целиком, без пояснений."""


def candidates(seg: dict, all_terms: list[str], p_max: float, k: int) -> list[str]:
    """Terms that sound closest to the doubtful words of a segment (1-2 word spans)."""
    ws = seg.get("words") or []
    spans = []
    for i, w in enumerate(ws):
        if w.get("p", 1) < p_max:
            spans += [[ws[i]["w"]]] + ([[ws[i]["w"], ws[i + 1]["w"]]] if i + 1 < len(ws) else []) \
                + ([[ws[i - 1]["w"], ws[i]["w"]]] if i else [])
    for a in seg.get("alts") or []:  # what another model heard that isn't Russian («вонунг», «шпильплац»)
        aw = (a.get("text") or "").split()
        for i, w in enumerate(aw):
            if T.key(w) and not terms.plain_russian([w]):
                spans += [[w]] + ([[w, aw[i + 1]]] if i + 1 < len(aw) else [])
    scored = {}
    for t in all_terms:
        tk = T.key(t)
        s = max((terms.best_sim(sp, tk)[0] for sp in spans), default=0.0)
        if s > scored.get(t, 0):
            scored[t] = s
    return [t for t, s in sorted(scored.items(), key=lambda x: -x[1])[:k] if s >= 0.45]


def prompt(seg: dict, cands: list[str], p_max: float) -> str:
    marked = " ".join(f"⟨{w['w']}⟩" if w.get("p", 1) < p_max else w["w"] for w in seg.get("words") or [])
    alts = "\n".join(f"- {a['text']}" for a in seg.get("alts") or [] if a.get("text"))
    return (f"Фрагмент: {marked}\n\nДругие модели слышали:\n{alts or '-'}\n\n"
            f"Термины: {', '.join(cands) or '-'}\n\nИсправленный фрагмент:")


class Model:
    def __init__(self, path: str, device: str, think: bool = False, nf4: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        dtype = torch.float16 if device == "cuda" else torch.float32
        kw = dict(dtype=dtype, trust_remote_code=True)
        if nf4:  # a 4B model in fp16 doesn't fit 6 GB: load it in 4 bits
            from transformers import BitsAndBytesConfig

            kw.update(device_map="cuda:0", low_cpu_mem_usage=True, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"]))
        try:
            self.m = AutoModelForCausalLM.from_pretrained(path, **kw)
        except (ValueError, KeyError):  # vision-language checkpoints (Qwen3.5): the text part is used
            from transformers import AutoModelForImageTextToText

            self.m = AutoModelForImageTextToText.from_pretrained(path, **kw)
        self.m = (self.m if nf4 else self.m.to(device)).eval()
        self.device = device
        self.think = think

    def __call__(self, user: str) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                           return_dict=True, enable_thinking=self.think).to(self.device)
        with self.torch.inference_mode():
            out = self.m.generate(**ids, max_new_tokens=1500 if self.think else 200, do_sample=False)
        text = self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=False)
        text = text.split("</think>")[-1]  # the answer after the reasoning, if any
        text = re.sub(r"<\|[^|]*\|>|<think>", "", text).strip()
        return text.splitlines()[0].strip().strip("«»\"") if text else ""


def outside(seg: dict, new: str, p_max: float) -> str | None:
    """--strict: a change may only replace doubtful words (and the punctuation around them)."""
    import difflib

    ws = seg.get("words") or []
    old = [T.key(w["w"]) for w in ws]
    sure = [w.get("p", 1) >= p_max for w in ws]
    sm = difflib.SequenceMatcher(a=old, b=[T.key(w) for w in new.split()], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "insert" or any(sure[i1:i2]):
            return "правка вне сомнительных слов"
        if tag == "replace" and j2 - j1 != i2 - i1:
            return "слово заменено не одним словом"
    return None


def errors(d: gold.Doc, segs: list[dict]) -> int:
    words = [w for s in segs for w in s["words"]]
    per = d.by_chunk(words)
    hyp = {ci: gold.norm([w["w"] for w in ws]) for ci, ws in per.items()}
    return gold.wer(d.ref, hyp, d.chunks)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--p", type=float, default=0.5, help="doubtful words: p below this")
    ap.add_argument("--k", type=int, default=8, help="candidate terms per segment")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--think", action="store_true", help="let the model reason first")
    ap.add_argument("--strict", action="store_true", help="only doubtful words may change")
    ap.add_argument("--nf4", action="store_true", help="load in 4 bits (models over ~2.5B on 6 GB)")
    ap.add_argument("--log", help="JSON lines with every prompt and answer")
    a = ap.parse_args()

    d = gold.Doc(a.ref)
    d.set_reference()
    opts = gold.options(d.doc)
    segs = ensemble.combine(json.loads(json.dumps(d.st)), opts)
    if opts.dictionary:
        store.apply_rules_to_segments(segs, gold.settings.rules())
    all_terms = terms.term_list(ensemble._glossary(opts))
    base = errors(d, segs)
    for i, s in enumerate(segs):
        s.setdefault("id", f"g{i}")
    todo = [s for s in segs if any(w.get("p", 1) < a.p for w in s.get("words") or [])]
    print(f"Эталон {d.id[-6:]}: ошибок слияния {base}; фрагментов с сомнением {len(todo)} из {len(segs)}; "
          f"терминов {len(all_terms)}", flush=True)
    if a.dry:
        for s in todo[:5]:
            print(prompt(s, candidates(s, all_terms, a.p, a.k), a.p), "\n---")
        return
    model = Model(a.model, a.device, a.think, a.nf4)
    log = open(a.log, "w", encoding="utf-8") if a.log else None
    changed = refused = 0
    t0 = time.monotonic()
    for s in todo:
        cands = candidates(s, all_terms, a.p, a.k)
        q = prompt(s, cands, a.p)
        ans = store.clean_text(model(q))
        why = None
        if ans and ans != s["text"]:
            why = llm.refused(s, ans) or (outside(s, ans, a.p) if a.strict else None)
            if why is None:
                old = s["text"]
                store.rewrite(s, ans)
                changed += 1
                print(f"  {s['id']}: {old}\n     -> {ans}", flush=True)
            else:
                refused += 1
        if log:
            log.write(json.dumps(dict(id=s["id"], prompt=q, answer=ans, refused=why), ensure_ascii=False) + "\n")
    after = errors(d, segs)
    print(f"\nМодель {Path(a.model).name}: изменено фрагментов {changed}, отклонено {refused}; "
          f"ошибок {base} -> {after}; {time.monotonic() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
