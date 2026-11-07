"""Run a candidate recognizer on the chunks of a document, as an ensemble member would (lab
experiment 12). The hypotheses go to HOME/tmp/member-lab/<doc id>/<name>.json and the member
to HOME/tmp/member-lab/members.json; tools/ensemble_curve.py then votes with them. Nothing in
the library changes. Models are downloaded into HOME/hf-cache.

    ws-python tools/member_lab.py --doc a1b2c3 --name gigaam-ml-large --onnx gigaam-multilingual-large-ctc --lang ru
    ws-python tools/member_lab.py --doc a1b2c3 --name turbo-ru-dv --whisper dvislobokov/faster-whisper-large-v3-turbo-russian
    ... --only ru          vote only on chunks with this language (default: the --lang if fixed)
    ... --weight 1.0       voting weight
    ... --lang auto        Whisper: let the model pick the language
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whisper_studio import audio as A  # noqa: E402
from whisper_studio import ensemble, gold  # noqa: E402
from whisper_studio.asr.onnx_models import _words  # noqa: E402
from whisper_studio.config import HOME, SAMPLE_RATE  # noqa: E402

LAB = HOME / "tmp" / "member-lab"


def onnx_backend(name: str, device: str):
    import onnx_asr
    import onnxruntime as rt

    so = rt.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
    asr = onnx_asr.load_model(name, sess_options=so, providers=["CPUExecutionProvider"]).with_timestamps()

    def rec(x, lang):
        kw = dict(language=lang or "ru", pnc=True) if "canary" in name else {}
        r = asr.recognize(x, sample_rate=SAMPLE_RATE, **kw)
        words = _words(r.tokens, r.timestamps, r.logprobs, len(x) / SAMPLE_RATE) \
            if r.timestamps and r.tokens and "canary" not in name else None
        return dict(text=(r.text or "").strip(), words=words)
    return rec


def whisper_backend(repo: str, device: str, beam: int = 5):
    from faster_whisper import WhisperModel

    m = WhisperModel(repo, device=device, compute_type="float16" if device == "cuda" else "int8",
                     download_root=str(HOME / "hf-cache" / "ct2"))

    def rec(x, lang):
        segs, _ = m.transcribe(x, language=lang, beam_size=beam, condition_on_previous_text=False,
                               vad_filter=False, word_timestamps=True, temperature=0.0)
        words = [dict(w=w.word.strip(), s=round(w.start, 3), e=round(w.end, 3), p=round(w.probability, 3))
                 for s in segs for w in (s.words or []) if w.word.strip()]
        return dict(text=" ".join(w["w"] for w in words), words=words)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--name", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--onnx", help="onnx-asr model name")
    g.add_argument("--whisper", help="CTranslate2 Whisper: HF repo or local folder")
    ap.add_argument("--lang", default="ru", help="ru | de | auto | chunk")
    ap.add_argument("--only", default=None, help="vote only on chunks with these languages, comma-separated")
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    d = gold.Doc(a.doc)
    src = next(p for p in (gold.LIBRARY_DIR / d.id).iterdir() if p.stem == "audio")
    x = A.load(src)
    x = x[0] if isinstance(x, tuple) else x
    rec = onnx_backend(a.onnx, a.device) if a.onnx else whisper_backend(a.whisper, a.device)
    out, t0 = {}, time.monotonic()
    for ci, ch in enumerate(d.st["chunks"]):
        lang = None if a.lang == "auto" else ch["lang"] if a.lang == "chunk" else a.lang
        h = rec(ensemble._chunk_audio(x, ch, d.st["level"]), lang)
        res = [None]
        ensemble._store_hyp(res, 0, h, ch, lang)
        out[str(ci)] = res[0]
    took = time.monotonic() - t0
    folder = LAB / d.id
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{a.name}.json").write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    cfg_f = LAB / "members.json"
    cfg = json.loads(cfg_f.read_text("utf-8")) if cfg_f.exists() else []
    only = a.only or (a.lang if a.lang in ("ru", "de") else None)
    m = dict(name=a.name, lang=a.lang if a.lang in ("ru", "de", "auto", "chunk") else "ru", weight=a.weight,
             source=a.onnx or a.whisper, seconds=round(took, 1))
    if only:
        m["only"] = only.split(",")
    cfg = [c for c in cfg if c["name"] != a.name] + [m]
    cfg_f.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), "utf-8")
    audio_s = sum(c["end"] - c["start"] for c in d.st["chunks"])
    print(f"{a.name}: {len(out)} кусков, {took:.0f} с на {audio_s:.0f} с звука -> {folder / (a.name + '.json')}")


if __name__ == "__main__":
    main()
