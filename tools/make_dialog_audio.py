"""Synthetic dialogue recording with a known text (lab experiment 16): scenes written like the
target calls (Russian with German terms, numbers in words, short replies, interruptions), every
line voiced by Audio8-TTS (Edge0/Audio8-TTS-Realistic-0.6b) with a voice cloned from a real
recording, the far side of the call passed through a narrowband Opus codec, noise at the scene's
SNR. The manifest keeps every line's text, speaker and time, so each recognizer, the merge and
the speakers can be scored word by word.

Script format (see %WS_REAL_HOME%/tmp/dialog-set/script-dev.txt):

    === d01 voices=A:gold1,B:gold2 snr=30
    A: Алло, Миша, привет.
    B~: угу                      a line laid over the end of the previous one
    Fahrkarte|фаркарте            either spelling is right; the first is the reference

A German term is voiced either as written or, at random, as a Russian speaker says it
(PRON); a German voice (name starting with "de") always gets the text as written.

Audio8 needs transformers < 5: a copy lives in HOME/tmp/lab-pkgs-tts and is put first on the path.

    ws-python tools/make_dialog_audio.py --script %WS_REAL_HOME%/tmp/dialog-set/script-dev.txt \\
        --out %WS_REAL_HOME%/tmp/dialog-set/dev
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

LAB_PKGS = REAL_HOME / "tmp/lab-pkgs-tts"
sys.path.insert(0, str(LAB_PKGS))
os.environ.setdefault("HF_HOME", str(REAL_HOME / "hf-cache"))  # remote code is cached there, not in the user profile
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

MODEL = REAL_HOME / "models-lab/audio8-tts"
LIBRARY = REAL_HOME / "library"
SR = 16000
# voice name -> (library doc, segment): clean single-speaker stretches of your own library to clone
VOICES = {
    "gold1": ("a1b2c3", "g12"), "gold2": ("a1b2c3", "g4"), "voice3": ("d4e5f6", "g65"),
    "test1": ("d4e5f6", "g58"), "test2": ("0f1e2d", "g29"), "de1": ("0f1e2d", "g4"),
}
FAR = "B"  # this side of the call comes through the phone codec
# how a Russian speaker says the term (used for about half of the occurrences)
PRON = {
    "Stadt": "штадт", "Wohnung": "вонунг", "Straßenbahn": "штрассенбан", "Hauptbahnhof": "хауптбанхоф",
    "Vorort": "форорт", "Rathaus": "ратхаус", "Frau Kessler": "фрау кесслер",
    "Fahrradweg": "фаррадвег", "Gartenhaus Sonnenberg": "гартенхаус зонненберг", "ICE": "и цэ е",
    "Einkaufen": "айнкауфен", "Anmeldung": "анмельдунг", "Erholung": "эрхолунг",
    "Spielplatz": "шпильплац", "Fahrkarte": "фаркарте", "Bahnsteig": "банштайг", "Haltestelle": "хальтештелле",
    "Flughafen": "флюгхафен", "Wochenende": "вохенэнде", "Museum": "музеум", "Termin": "термин",
    "Feierabend": "файерабенд", "Altstadt": "альтштадт", "Fahrplan": "фарплан",
    "Berliner Straße": "берлинер штрассе", "Wetterbericht": "веттербериг", "Seeufer": "зееуфер",
    "Regenschirm": "регенширм", "Aussicht": "аусзихт", "Fahrrad": "фаррад", "Urlaub": "урлауб",
}
TERM_RE = re.compile(r"(?<![\wÀ-ÿ])(" + "|".join(re.escape(t) for t in sorted(PRON, key=len, reverse=True))
                     + r")([а-яё]*)")


def parse(path: Path) -> list[dict]:
    scenes = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("==="):
            head = line[3:].split()
            kv = dict(x.split("=", 1) for x in head[1:])
            scenes.append(dict(id=head[0], voices=dict(v.split(":") for v in kv["voices"].split(",")),
                               snr=float(kv.get("snr", 30)), lines=[]))
            continue
        who, text = line.split(":", 1)
        scenes[-1]["lines"].append(dict(spk=who.rstrip("~"), over=who.endswith("~"), text=text.strip()))
    return scenes


def spoken(text: str, german: bool, rnd: random.Random) -> tuple[str, list[list[str]]]:
    """Text for the synthesizer and which form each term got: alternatives -> the Cyrillic one,
    a term -> as written or as a Russian speaker says it."""
    text = re.sub(r"(\S+)\|(\S+)", lambda m: m.group(2) if not german else m.group(1), text)
    if german:
        return text, []
    forms = []

    def sub(m):
        if rnd.random() < 0.5:
            forms.append([m.group(1), "as-written"])
            return m.group(0)
        forms.append([m.group(1), "ru"])
        return PRON[m.group(1)] + m.group(2)

    return TERM_RE.sub(sub, text), forms


def reference(text: str) -> str:
    return re.sub(r"(\S+)\|\S+", r"\1", text)


def ref_voice(name: str, tmp: Path) -> tuple[str, str]:
    import soundfile as sf
    from scipy.signal import resample_poly

    from whisper_studio import audio as A

    doc_id, seg_id = VOICES[name]
    folder = next(LIBRARY.glob(f"*{doc_id}"))
    doc = json.loads((folder / "doc.json").read_text("utf-8"))
    s = next(x for x in doc["segments"] if x["id"] == seg_id)
    x = A.load(next(p for p in folder.iterdir() if p.stem == "audio"))
    x = x[0] if isinstance(x, tuple) else x
    f = tmp / f"ref-{name}.wav"
    # at the codec's rate: the processor would need torchaudio to resample
    sf.write(f, resample_poly(x[int(s["start"] * SR): int(s["end"] * SR)], 441, 160), 44100)
    return str(f), s["text"]


def opus(y: np.ndarray, kbps: int) -> np.ndarray:
    """Through narrowband Opus (8 kHz) and back to 16 kHz, like the far side of a phone call."""
    import av
    from scipy.signal import resample_poly

    x = resample_poly(y, 1, 2).astype(np.float32)
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        st = out.add_stream("libopus", rate=8000, layout="mono")
        st.bit_rate = kbps * 1000
        n = 160
        for i in range(0, len(x), n):
            fr = av.AudioFrame.from_ndarray(x[None, i:i + n].copy(), format="flt", layout="mono")
            fr.sample_rate = 8000
            fr.pts = i
            for p in st.encode(fr):
                out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    buf.seek(0)
    got = []
    with av.open(buf) as inp:
        rs = av.AudioResampler(format="flt", layout="mono", rate=8000)
        for fr in inp.decode(audio=0):
            for r in rs.resample(fr):
                got.append(r.to_ndarray().reshape(-1))
    z = np.concatenate(got)[: len(x)]
    return resample_poly(z, 2, 1).astype(np.float32)[: len(y)]


def noise(n: int, rnd: np.random.Generator) -> np.ndarray:
    """Pinkish background (1/f), unit RMS."""
    w = rnd.standard_normal(n)
    f = np.fft.rfft(w)
    f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
    x = np.fft.irfft(f, n)
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kbps", type=int, default=12, help="Opus bitrate of the far side")
    ap.add_argument("--scenes", help="only these scene ids (comma-separated), for a try")
    a = ap.parse_args()
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import AutoModel, AutoProcessor

    out = Path(a.out)
    (out / "lines").mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    nrnd = np.random.default_rng(a.seed)
    scenes = parse(Path(a.script))
    if a.scenes:
        keep = set(a.scenes.split(","))
        scenes = [s for s in scenes if s["id"] in keep]
    names = sorted({v for s in scenes for v in s["voices"].values()})
    refs = {n: ref_voice(n, out) for n in names}
    proc = AutoProcessor.from_pretrained(str(MODEL), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(MODEL), trust_remote_code=True, dtype=torch.bfloat16).eval().to("cuda")
    sr = model.config.codec_sample_rate

    def say(text: str, voice: str, seed: int) -> np.ndarray:
        ref, ref_text = refs[voice]
        inputs = {k: v.to("cuda") for k, v in
                  proc(text=[text], return_tensors="pt", reference_audio=[ref], reference_text=[ref_text]).items()}
        torch.manual_seed(seed)
        with torch.inference_mode():
            g = model.generate(**inputs, max_new_tokens=1500, temperature=0.8, top_p=0.95, top_k=50,
                               do_sample=True, return_dict_in_generate=True)
            wav, n = model.decode_audio(g.codes)
        y = wav[0, : int(n[0])].float().cpu().numpy()
        return resample_poly(y, SR, sr).astype(np.float32)

    lines, track, t = [], [], 0.0
    for sc in scenes:
        start = t
        prev_end = t
        parts = []  # (start, audio)
        for k, ln in enumerate(sc["lines"]):
            voice = sc["voices"][ln["spk"]]
            text, forms = spoken(ln["text"], voice.startswith("de"), rnd)
            y = say(text, voice, a.seed * 100000 + len(lines))
            y *= 0.5 / max(1e-6, float(np.abs(y).max())) * (0.8 + 0.4 * rnd.random())
            if ln["spk"] == FAR:
                y = opus(y, a.kbps)
            sf.write(out / "lines" / f"{sc['id']}-{k:02d}.wav", y, SR)
            dur = len(y) / SR
            if ln["over"] and parts:
                s0 = max(parts[-1][0] + 0.2, prev_end - min(0.7, 0.5 * dur))
            else:
                s0 = prev_end + rnd.uniform(0.25, 0.8)
            parts.append((s0, y))
            prev_end = max(prev_end, s0 + dur)
            lines.append(dict(scene=sc["id"], i=k, spk=ln["spk"], voice=voice, over=ln["over"],
                              text=reference(ln["text"]), alts=re.findall(r"\S+\|\S+", ln["text"]),
                              spoken=text, forms=forms, start=round(s0, 3), end=round(s0 + dur, 3)))
            print(f"{sc['id']} {k:2d} {ln['spk']} {dur:5.1f} с  {text}", flush=True)
        end = prev_end + 2.5
        x = np.zeros(int(round((end - start) * SR)), np.float32)
        for s0, y in parts:
            i = int(round((s0 - start) * SR))
            x[i:i + len(y)] += y[: len(x) - i]
        speech = np.sqrt(np.mean(np.concatenate([y for _, y in parts]) ** 2))
        x += noise(len(x), nrnd) * speech / 10 ** (sc["snr"] / 20)
        track.append(x)
        sc_rec = {k: v for k, v in sc.items() if k != "lines"}
        sc_rec.update(start=round(start, 3), end=round(end, 3))
        sc.clear()
        sc.update(sc_rec)
        t = end
    y = np.concatenate(track)
    y *= 0.9 / max(1e-6, float(np.abs(y).max()))
    sf.write(out / "dialog.wav", y, SR)
    man = dict(scenes=scenes, lines=lines, voices={n: list(VOICES[n]) for n in names}, kbps=a.kbps, seed=a.seed)
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), "utf-8")
    words = sum(len(ln["text"].split()) for ln in lines)
    print(f"{len(scenes)} сцен, {len(lines)} реплик, {words} слов, {t / 60:.1f} мин -> {out / 'dialog.wav'}")


if __name__ == "__main__":
    main()
