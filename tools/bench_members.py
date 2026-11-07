"""Speed of the ensemble members: model load and recognition timed apart, on the chunks of a
library document, through the same code the app runs (ensemble._run_member). Nothing is
written into the library: the hypotheses go to a scratch state, the report to --json if asked.

    ws-python tools/bench_members.py --doc a1b2c3                      # every member, one by one
    ws-python tools/bench_members.py --doc a1b2c3 --member gigaam-v3 --member t-one
    ... --compare               # word-level difference to the cached hypotheses.json (did it change?)
    ... --wer                   # the document is a reference: member and merge WER, cached vs new
    ... --show 3                # the 3 most different chunks of each member, both texts
    ... --chunks 0-5            # only these chunks (prompts with the previous words then differ at 0)
    ... --onnx-device cuda --onnx-batch 8 --onnx-threads 6 --onnx-split 0 [--onnx-batch-any]
    ... --qwen-batch 4 --qwen-tps 15
    ... --whisper-temps 0 --beam 5     # cheaper Whisper decoding (changes the text: --compare)
    ... --vram                  # free video memory (nvidia-smi) before/after load and after each call
    ... --cpu                   # hide the GPU (CUDA_VISIBLE_DEVICES=-1): ONNX members on the CPU
    ... --threads 4             # torch / OpenMP CPU threads (default 4: leave the worker some)
    ... --json out.json

(ws-python = the venv's python with PYTHONPATH=<repo>; for onnxruntime-gpu
use the overlay venv <venv-ortgpu>\\Scripts\\python.exe the same way.)

Members: large-v3 ru-large-v3 de-turbo qwen3-asr gigaam-v3 canary-v2 t-one de-check. Each runs
on the chunks it has a cached hypothesis for (the same work as in the app); de-check on the
chunks the app re-checked. large-v3 also runs language-ID per chunk as the detector does
(not the language scan over the whole recording).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, help="ID документа (достаточно последних 6 знаков)")
    ap.add_argument("--hyp", default="hypotheses.json", help="файл гипотез в папке документа")
    ap.add_argument("--member", action="append", default=[], help="участник (можно несколько; по умолчанию все)")
    ap.add_argument("--chunks", default="", help="куски: 0-5,8")
    ap.add_argument("--compare", action="store_true", help="сравнить текст с кэшем (доля изменённых слов)")
    ap.add_argument("--wer", action="store_true",
                    help="документ — эталон: WER участника и слияния до и после (новые гипотезы вместо кэша)")
    ap.add_argument("--show", type=int, default=0, help="показать N самых изменённых кусков")
    ap.add_argument("--onnx-device", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--onnx-batch", type=int)
    ap.add_argument("--onnx-batch-any", action="store_true", help="пакеты и для GigaAM/Canary (текст меняется)")
    ap.add_argument("--onnx-threads", type=int)
    ap.add_argument("--onnx-split", choices=("0", "1"))
    ap.add_argument("--qwen-batch", type=int)
    ap.add_argument("--qwen-tps", type=int, help="new tokens per second of audio (cap)")
    ap.add_argument("--whisper-temps", help="температуры Whisper через запятую, например 0 или 0,0.2")
    ap.add_argument("--beam", type=int, help="beam size Whisper")
    ap.add_argument("--vram", action="store_true", help="мерить свободную видеопамять (nvidia-smi)")
    ap.add_argument("--cpu", action="store_true", help="CUDA_VISIBLE_DEVICES=-1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", help="куда записать отчёт (не в библиотеку)")
    return ap.parse_args()


def chunk_set(spec: str, n: int) -> set[int] | None:
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), (int(b) if b else int(a)) + 1))
    return {i for i in out if 0 <= i < n}


def main() -> None:
    a = parse()
    if a.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        a.onnx_device = a.onnx_device or "cpu"  # nvidia-smi still sees the card: don't wait for it
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(a.threads))
    env = dict(WHISPER_STUDIO_ONNX_DEVICE=a.onnx_device, WHISPER_STUDIO_ONNX_BATCH=a.onnx_batch,
               WHISPER_STUDIO_ONNX_THREADS=a.onnx_threads, WHISPER_STUDIO_ONNX_SPLIT=a.onnx_split,
               WHISPER_STUDIO_ONNX_BATCH_ANY="1" if a.onnx_batch_any else None,
               WHISPER_STUDIO_QWEN_BATCH=a.qwen_batch, WHISPER_STUDIO_QWEN_TOKENS_PER_S=a.qwen_tps)
    for k, v in env.items():
        if v is not None:
            os.environ[k] = str(v)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np  # noqa: F401

    from whisper_studio import asr, audio as A, ensemble, gold, models
    from whisper_studio.asr import gpu
    from whisper_studio.config import LIBRARY_DIR, SAMPLE_RATE

    try:
        import torch

        torch.set_num_threads(a.threads)
    except Exception:  # noqa: BLE001
        torch = None

    folder = LIBRARY_DIR / gold.find(a.doc)
    doc = json.loads((folder / "doc.json").read_text("utf-8"))
    st = json.loads((folder / a.hyp).read_text("utf-8"))
    opts = gold.options(doc)
    if opts.stereo_split and a.hyp == "hypotheses.json" and (folder / "hypotheses-1.json").exists():
        sys.exit("стерео-документ: укажите --hyp hypotheses-1.json")
    t0 = time.monotonic()
    x = A.load(folder / doc["audio_file"])
    if opts.highpass:
        x = A.highpass(x)
    print(f"Документ {folder.name}: {len(x) / SAMPLE_RATE / 60:.1f} мин, {len(st['chunks'])} кусков, "
          f"аудио {time.monotonic() - t0:.1f} с")
    fixed = None if (opts.mixed or not opts.language) else opts.language
    prompt = (doc.get("meta") or {}).get("prompt")  # the prompt of that run (the dictionary may have grown)
    only = chunk_set(a.chunks, len(st["chunks"]))

    by_name = {m["name"]: m for m in ensemble.MEMBERS + [ensemble.DE_CHECK]}
    names = a.member or [n for n in by_name if n in st["hyps"]]
    for n in names:
        if n not in by_name:
            sys.exit(f"нет участника {n}; есть: {', '.join(by_name)}")

    # every call of every backend timed (wrapping asr.make, which ensemble._run_member uses)
    calls: list[dict] = []
    new_hyps: dict = {}
    vram: list[dict] = []
    real_make = asr.make

    def sample(what: str) -> None:
        if a.vram:
            vram.append(dict(at=what, free_gb=gpu.free_gb()))

    def make(m):
        b = real_make(m)
        load, rec, recb = b.load, b.recognize, getattr(b, "recognize_batch", None)

        def timed_load(note=None):
            sample("before load")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            load(note)
            sample("after load")

        def timed(fn, xs, *args):
            t = time.monotonic()
            out = fn(xs, *args)
            calls.append({**(getattr(b, "last", None) or {}),  # the backend's own numbers (qwen: tokens)
                          "seconds": round(time.monotonic() - t, 3), "items": len(xs) if isinstance(xs, list) else 1,
                          "audio": round(sum(len(v) for v in (xs if isinstance(xs, list) else [xs])) / SAMPLE_RATE, 2)})
            sample("after call")
            return out

        b.load = timed_load
        b.recognize = lambda xs, *args: timed(rec, xs, *args)
        if recb is not None:
            b.recognize_batch = lambda xs, *args: timed(recb, xs, *args)
        return b

    asr.make = make
    report = dict(doc=folder.name, duration=st["duration"], env={k: v for k, v in env.items() if v is not None},
                  members={})
    for name in names:
        m = dict(by_name[name])
        if m["backend"] == "whisper":
            if a.whisper_temps:
                m["temperature"] = tuple(float(t) for t in a.whisper_temps.split(","))
            if a.beam:
                m["beam_size"] = a.beam
        if models.find(m["model"]) is None:
            print(f"{name}: модель не установлена")
            continue
        cached = st["hyps"].get(name) or []
        todo = {i for i, h in enumerate(cached) if h is not None and (only is None or i in only)}
        if not todo:
            print(f"{name}: нет кусков")
            continue
        bst = dict(chunks=copy.deepcopy(st["chunks"]), hyps={}, errors={}, timing={}, level=st["level"],
                   detector=name if name == st.get("detector") else None)
        calls.clear()
        vram.clear()
        call_fixed = "de" if name == "de-check" else fixed
        call_prompt = None if name == "de-check" else prompt
        t0 = time.monotonic()
        err = None
        try:
            ensemble._run_member(m, bst, x, call_prompt, call_fixed, lambda f, msg: None, only_chunks=todo)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        wall = time.monotonic() - t0
        det = (bst.get("timing_detail") or {}).get(name, {})
        size = sum(f.stat().st_size for f in Path(models.find(m["model"])).rglob("*") if f.is_file()) / 1e9
        r = dict(wall=round(wall, 1), load=det.get("load"), wait=det.get("wait"), run=det.get("run"),
                 device=det.get("device"), chunks=len(todo), audio=det.get("audio"), size_gb=round(size, 2),
                 calls=list(calls), vram=list(vram), error=err)
        if torch is not None and torch.cuda.is_available():
            r["torch_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
        if a.compare and not err:
            r["compare"] = compare(gold, cached, bst["hyps"].get(name) or [], todo, a.show)
        report["members"][name] = r
        if not err:
            new_hyps[name] = (todo, bst["hyps"].get(name) or [])
        print(line(name, r))
        if r.get("compare"):
            c = r["compare"]
            print(f"    отличие от кэша: {c['changed_words']}/{c['words']} слов ({c['rate']:.1%}), "
                  f"изменилось кусков {c['changed_chunks']}/{len(todo)}")
            for s in c.get("worst", []):
                print(f"    #{s['chunk']} ({s['edits']} правок)\n      было:  {s['old']}\n      стало: {s['new']}")
        if calls and "temperatures" in calls[0]:
            fb = sum(1 for c in calls for t in c["temperatures"] if t and t > 0)
            n_seg = sum(len(c["temperatures"]) for c in calls)
            print(f"    Whisper: сегментов {n_seg}, с повтором на температуре > 0: {fb}")
        if calls and "new_tokens" in calls[0]:
            tok = sum(sum(c["new_tokens"]) for c in calls)
            gen = sum(c["seconds"] for c in calls)
            print(f"    Qwen: вызовов {len(calls)}, новых токенов {tok}, {tok / gen:.1f} ток/с, "
                  f"самый долгий вызов {max(c['seconds'] for c in calls):.1f} с")
    if a.wer and new_hyps:
        report["wer"] = wer_change(gold, folder.name, opts, new_hyps)
        w = report["wer"]
        for n, (e0, e1, t) in w["members"].items():
            print(f"WER {n:12s} {e0 / t:6.1%} -> {e1 / t:6.1%}  ({t} слов эталона)")
        e0, e1, t = w["merge"]
        print(f"WER слияния      {e0 / t:6.1%} -> {e1 / t:6.1%}  ({t} слов эталона, кусков {w['chunks']})")
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
        print(f"Отчёт: {a.json}")


def line(name: str, r: dict) -> str:
    if r.get("error"):
        return f"{name:12s} ОШИБКА {r['error']} (за {r['wall']} с)"
    load, run, audio = r["load"] or 0.0, r["run"] or 0.0, r["audio"] or 0.0
    speed = f"{audio / run:5.1f}x" if run else "  -  "
    mbs = f"{r['size_gb'] * 1000 / load:5.0f} МБ/с" if load > 0.5 else "      -  "
    return (f"{name:12s} {r['device'] or '?':4s} загрузка {load:6.1f} с (ждал {r['wait'] or 0:.0f}, {mbs}, "
            f"{r['size_gb']:.2f} ГБ)  распознавание {run:6.1f} с ({speed} реального времени, "
            f"{r['chunks']} кусков, {audio:.0f} с аудио)")


def wer_change(gold, doc_id: str, opts, new_hyps: dict) -> dict:
    """WER against the owner's reference (docs/GOLD.md) with the cached hypotheses and with the
    new ones put in their place (in memory only)."""
    d = gold.Doc(doc_id)
    d.set_reference()
    before = {n: d.member(n) for n in new_hyps}
    merge0, _ = d.merged(opts)
    for n, (todo, new) in new_hyps.items():
        res = list(d.st["hyps"].get(n) or [None] * len(d.st["chunks"]))
        for i in todo:
            res[i] = new[i]
        d.st["hyps"][n] = res
    out = {}
    for n in new_hyps:
        after = d.member(n)
        e0, t = gold.wer(d.ref, before[n])
        e1, _ = gold.wer(d.ref, after, list(before[n]))
        if t:
            out[n] = (e0, e1, t)
    merge1, _ = d.merged(opts)
    e0, t = gold.wer(d.ref, merge0, d.chunks)
    e1, _ = gold.wer(d.ref, merge1, d.chunks)
    return dict(members=out, merge=(e0, e1, max(t, 1)), chunks=len(d.chunks))


def compare(gold, old: list, new: list, todo: set[int], show: int) -> dict:
    words = edits = changed = 0
    rows = []
    for i in sorted(todo):
        a = gold.norm(((old[i] if i < len(old) else None) or {}).get("text", "").split())
        b = gold.norm(((new[i] if i < len(new) else None) or {}).get("text", "").split())
        e = gold.errors(a, b)
        words += len(a)
        edits += e
        changed += e > 0
        if e:
            rows.append(dict(chunk=i, edits=e, old=(old[i] or {}).get("text", "")[:300],
                             new=(new[i] or {}).get("text", "")[:300]))
    rows.sort(key=lambda r: -r["edits"])
    return dict(words=words, changed_words=edits, rate=edits / words if words else 0.0, changed_chunks=changed,
                worst=rows[:show])


if __name__ == "__main__":
    main()
