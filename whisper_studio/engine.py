"""Whisper transcription tuned for accuracy (see the guide, sections 2-7).

Sequential long-form decoding with faster-whisper, beam 5, temperature fallback,
soft VAD only against hallucinations on long pauses, glossary as prompt+hotwords,
and post-hoc flagging (not deleting) of suspected phantom phrases.
"""
from __future__ import annotations

import gc
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import models
from .config import SAMPLE_RATE
from .options import PROFILES, Options, prompt  # noqa: F401 - re-exported


def _cuda_dlls():
    """ctranslate2 on Windows needs cuBLAS DLLs; torch's CUDA wheel ships them.
    Only the folder is registered: importing torch itself costs a minute under Defender."""
    if sys.platform != "win32":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        lib = Path(spec.origin).parent / "lib" if spec and spec.origin else None
        if lib and lib.exists():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # noqa: BLE001
        pass


_cuda_dlls()


def prewarm():
    """Import torch/transformers in the background so the first alignment doesn't wait."""
    def run():
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=run, daemon=True).start()


def device() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"  # guide: fp16 on GPU, int8 hurts large-v3
    except Exception:  # noqa: BLE001
        pass
    return "cpu", "float32"  # guide: float32 is the most accurate on CPU


# --------------------------------------------------------------- profiles

# PROFILES and Options live in options.py (no numpy there, so light processes can use them)

SOFT_VAD = dict(threshold=0.35, min_speech_duration_ms=250,
                min_silence_duration_ms=2000, speech_pad_ms=400)

# Known phantom phrases from subtitle training data (guide, section 6)
BLACKLIST = [re.compile(p, re.I) for p in [
    r"^субтитры (сделал|создавал|подогнал|делал|подготовил)", r"dimatorzok", r"^продолжение следует",
    r"редактор субтитров", r"корректор [а-я]\.", r"субтитры (создавал|делал) ",
    r"спасибо за просмотр\.?$", r"^подписывайтесь на (наш )?канал", r"ставьте лайк",
    r"untertitel (im auftrag|der amara|von )", r"^copyright wdr", r"amara\.org",
    r"^vielen dank (fürs|für's) zuschauen", r"^thanks for watching", r"^subtitles by",
    r"^please subscribe", r"^\[?музыка\]?$", r"^\[?music\]?$", r"^\[?musik\]?$",
    r"ссылк[аеи] в описании", r"в описании (к )?(этому )?видео", r"подписывайтесь", r"^субтитры:? ",
]]


def suspicious(text: str, no_speech: float, logprob: float, compression: float) -> list[str]:
    t = text.strip().lower()
    flags = []
    if any(p.search(t) for p in BLACKLIST):
        flags.append("phantom")
    if no_speech > 0.6 and logprob < -1.0:
        flags.append("no_speech")
    if compression > 2.4:
        flags.append("repetition")
    return flags


Progress = Callable[[float, str], None]

_model_lock = threading.Lock()
_loaded: dict = {"key": None, "model": None, "dev": None}


def _vram_gb(key: str) -> float:
    """Video memory a model needs in fp16 (weights + decoding buffers, beam 5)."""
    return 2.2 if models.MODELS.get(key, {}).get("size_gb", 3.1) < 2 else 3.8


def load_whisper(key: str, note: Callable[[str], None] | None = None, cpu_fallback: bool = True):
    """Load (or reuse) a Whisper model. When another program holds the video memory,
    wait for it (asr.gpu.wait_limit) and then fall back to the CPU instead of failing
    (or raise, if cpu_fallback is False: an ensemble member is better skipped)."""
    from faster_whisper import WhisperModel

    from .asr import gpu

    with _model_lock:
        if _loaded["key"] == key and _loaded["model"] is not None:
            m = _loaded["model"]
            if not m.model.model_is_loaded:
                m.model.load_model()
            return m
        unload()
        path = models.find(key)
        if path is None:
            raise RuntimeError(f"Модель {key} не установлена")
        dev, ctype = device()
        m = None
        if dev == "cuda":
            deadline = time.monotonic() + gpu.wait_limit()
            while m is None:
                if not gpu.wait(_vram_gb(key), note, max(deadline - time.monotonic(), 0)):
                    break
                try:
                    m = WhisperModel(str(path), device=dev, compute_type=ctype)
                except RuntimeError as e:
                    if not gpu.is_oom(e) or time.monotonic() > deadline:
                        raise
                    gc.collect()
                    time.sleep(gpu.POLL)
            if m is None and not cpu_fallback:
                raise RuntimeError("видеопамять занята другой программой")
            if m is None:
                # int8 on the fallback path: float32 large-v3 needs 6 GB of RAM
                dev, ctype = "cpu", "int8"
                if note:
                    note("Видеопамять занята — распознаю на процессоре (медленнее)")
        if m is None:
            m = WhisperModel(str(path), device=dev, compute_type=ctype, cpu_threads=os.cpu_count() or 4)
        _loaded.update(key=key, model=m, dev=dev)
        return m


def park():
    """Move Whisper weights to RAM to free VRAM for the aligner."""
    m = _loaded.get("model")
    if m is not None and _loaded.get("dev") == "cuda":
        try:
            m.model.unload_model(to_cpu=True)
        except Exception:  # noqa: BLE001
            pass


def unload():
    _loaded.update(key=None, model=None, dev=None)
    gc.collect()


def detect_language(audio: np.ndarray) -> tuple[str, float]:
    key = models.whisper_for(None, "large-v3") or models.whisper_for(None)
    if key is None:
        raise RuntimeError("Нет ни одной установленной модели Whisper")
    m = load_whisper(key)
    lang, prob, _ = m.detect_language(audio, vad_filter=True, language_detection_segments=4,
                                      language_detection_threshold=0.6)
    return lang, prob


_prompt = prompt  # kept for older callers


def words_of(seg, offset: float = 0.0) -> list[dict]:
    """faster-whisper words -> [{w, s, e, p}]; sub-word tokens are glued to their word."""
    words: list[dict] = []
    for w in seg.words or []:
        if not w.word.strip():
            continue
        # a token without a leading space continues the previous word ("26" + "-го")
        if words and not w.word[:1].isspace():
            prev = words[-1]
            prev.update(w=prev["w"] + w.word.strip(), e=round(w.end + offset, 3),
                        p=round(min(prev["p"], w.probability), 3))
            continue
        words.append(dict(w=w.word.strip(), s=round(w.start + offset, 3), e=round(w.end + offset, 3),
                          p=round(w.probability, 3)))
    return words


def mixed_model(opts: Options) -> str | None:
    """Russian with German: multilingual large-v3 keeps German phrases German (measured best
    single model); language-forced models translate them into Russian."""
    if opts.mixed and opts.model in ("auto", "", None) and models.find("large-v3"):
        return "large-v3"
    return None


def transcribe(audio: np.ndarray, opts: Options, progress: Progress | None = None,
               offset: float = 0.0) -> tuple[list[dict], dict]:
    """Return segments [{start,end,text,words,metrics}] and info (one Whisper model)."""
    language = opts.language
    key = mixed_model(opts)
    if key:
        language = None
        opts.extra.setdefault("multilingual", True)
    else:
        key = models.whisper_for(opts.language, opts.model)
    if key is None:
        raise RuntimeError("Подходящая модель Whisper не установлена — откройте «Модели».")
    m = load_whisper(key, (lambda msg: progress(0.0, msg)) if progress else None)
    prof = PROFILES.get(opts.profile, PROFILES["balanced"])
    vad = prof["vad"] if opts.vad is None else opts.vad
    condition = prof["condition"] if opts.condition is None else opts.condition
    prompt, hotwords = _prompt(opts)
    # with context on, the prompt carries the glossary; without it hotwords must (guide, s.5)
    if condition:
        hotwords = None

    params = dict(
        language=language, task="transcribe",
        beam_size=opts.beam_size, best_of=5, patience=1.0,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
        condition_on_previous_text=condition, prompt_reset_on_temperature=0.5,
        initial_prompt=prompt, hotwords=hotwords,
        word_timestamps=True, hallucination_silence_threshold=2.0,
        vad_filter=vad, vad_parameters=SOFT_VAD if vad else None,
    )
    params.update(opts.extra)
    seg_iter, info = m.transcribe(audio, **params)
    total = max(info.duration, 1e-6)
    out = []
    for s in seg_iter:
        words = words_of(s, offset)
        text = s.text.strip()
        if not text:
            continue
        out.append(dict(
            start=round(s.start + offset, 3), end=round(s.end + offset, 3), text=text, words=words,
            metrics=dict(avg_logprob=round(s.avg_logprob, 3), no_speech=round(s.no_speech_prob, 3),
                         compression=round(s.compression_ratio, 3), temperature=s.temperature),
            flags=suspicious(text, s.no_speech_prob, s.avg_logprob, s.compression_ratio),
        ))
        if progress:
            progress(min(s.end / total, 1.0), text[:80])
    meta = dict(model=key, language=info.language, language_probability=round(info.language_probability, 3),
                duration=info.duration, vad=vad, condition_on_previous_text=condition,
                prompt=prompt, hotwords=hotwords, device=_loaded.get("dev") or device()[0],
                quality="fast", mixed=bool(opts.mixed))
    return out, meta


def recognize(audio: np.ndarray, opts: Options, progress: Progress | None = None,
              cache: Path | None = None, tail: list | None = None) -> tuple[list[dict], dict]:
    """Speech -> segments in the chosen quality mode.

    "max": every installed ensemble model on the same chunks, merged word by word
    (ensemble.py; raw hypotheses are kept in `cache` for re-merging); "fast": one Whisper.
    tail: models needed after recognition (the aligners), for prefetch.py."""
    if opts.quality == "max":
        from . import ensemble

        if ensemble.available(opts):
            return ensemble.transcribe(audio, opts, progress, cache=cache, tail=tail)
    segs, meta = transcribe(audio, opts, progress)
    if opts.quality == "max":
        meta["ensemble_skipped"] = "установлено меньше двух моделей для режима «Максимум»"
    return segs, meta


def segment_language(seg: dict, default: str | None) -> str | None:
    """Language of one segment for choosing the word aligner: the ensemble marks it,
    otherwise mostly-Latin text inside a Russian recording is German."""
    if seg.get("lang"):
        return seg["lang"]
    text = seg.get("text", "")
    lat = len(re.findall(r"[A-Za-zÄÖÜäöüß]", text))
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    if default in ("ru", None) and lat > cyr:
        return "de"
    if default in ("de", None) and cyr > lat:
        return "ru"
    return default
