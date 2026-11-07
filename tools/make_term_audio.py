"""Synthetic test recording for German terms in Russian speech (lab experiment 14): Russian
sentences with a project term, voiced by Audio8-TTS (Edge0/Audio8-TTS-Realistic-0.6b) with
voices cloned from a few seconds of real recordings, glued into one 16 kHz file with pauses.
The manifest keeps every sentence's text, term and time, so term recall and false
replacements can be counted after the app transcribes the file.

Audio8 needs transformers < 5: a copy lives in HOME/tmp/lab-pkgs-tts and is put first on the path.

    ws-python tools/make_term_audio.py --out %WS_REAL_HOME%/tmp/term-set --n 120
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

LAB_PKGS = REAL_HOME / "tmp/lab-pkgs-tts"
sys.path.insert(0, str(LAB_PKGS))
os.environ.setdefault("HF_HOME", str(REAL_HOME / "hf-cache"))  # remote code is cached there, not in the user profile
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

MODEL = REAL_HOME / "models-lab/audio8-tts"
TEMPLATES = [
    "Слушай, по {t} надо ещё раз всё проверить до понедельника.",
    "Она прислала письмо, там написано про {t}, посмотри, пожалуйста.",
    "Я думаю, что {t} мы сделаем на следующей неделе.",
    "Короче, с этим {t} у нас опять проблема.",
    "Выпиши мне отдельно, что там было по {t}, хорошо?",
    "Ну да, {t} она уже отправила, я видел.",
]
# (library doc, segment): clean single-speaker stretches with checked text, voices to clone
REFS = [("a1b2c3", "g12"), ("a1b2c3", "g30")]


def ref_voices(tmp: Path) -> list[tuple[str, str]]:
    from whisper_studio import audio as A, gold

    out = []
    for doc_id, seg_id in REFS:
        d = gold.Doc(doc_id)
        s = next(x for x in d.doc["segments"] if x["id"] == seg_id)
        x = A.load(next(p for p in (gold.LIBRARY_DIR / d.id).iterdir() if p.stem == "audio"))
        x = x[0] if isinstance(x, tuple) else x
        f = tmp / f"ref-{doc_id}-{seg_id}.wav"
        import soundfile as sf
        from scipy.signal import resample_poly

        # at the codec's rate: the processor would need torchaudio to resample
        sf.write(f, resample_poly(x[int(s["start"] * 16000): int(s["end"] * 16000)], 441, 160), 44100)
        out.append((str(f), s["text"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModel, AutoProcessor

    from whisper_studio import settings

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    terms = [t for t in settings.dictionary()["terms"] if any("a" <= c.lower() <= "z" for c in t)]
    voices = ref_voices(out) + [(None, None)]
    proc = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(MODEL), trust_remote_code=True, dtype=torch.bfloat16).eval().to("cuda")
    sr = model.config.codec_sample_rate
    pieces, items, t = [], [], 0.0
    gap = np.zeros(int(1.2 * 16000), np.float32)
    for i in range(a.n):
        term = terms[i % len(terms)]
        text = rnd.choice(TEMPLATES).format(t=term)
        ref, ref_text = voices[i % len(voices)]
        kw = dict(reference_audio=[ref], reference_text=[ref_text]) if ref else {}
        inputs = {k: v.to("cuda") for k, v in proc(text=[text], return_tensors="pt", **kw).items()}
        torch.manual_seed(a.seed * 1000 + i)
        with torch.inference_mode():
            g = model.generate(**inputs, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=50,
                               do_sample=True, return_dict_in_generate=True)
            wav, n = model.decode_audio(g.codes)
        y = wav[0, : int(n[0])].float().cpu().numpy()
        y = resample_poly(y, 16000, sr).astype(np.float32)
        y *= 0.9 / max(1e-6, float(np.abs(y).max()))
        pieces += [y, gap]
        items.append(dict(i=i, text=text, term=term, voice=Path(ref).stem if ref else "default",
                          start=round(t, 3), end=round(t + len(y) / 16000, 3)))
        t += (len(y) + len(gap)) / 16000
        print(f"{i:3d} {items[-1]['end'] - items[-1]['start']:5.1f} с  {text}", flush=True)
    sf.write(out / "term-set.wav", np.concatenate(pieces), 16000)
    (out / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=1), "utf-8")
    print(f"{len(items)} фраз, {t / 60:.1f} мин -> {out / 'term-set.wav'}")


if __name__ == "__main__":
    main()
