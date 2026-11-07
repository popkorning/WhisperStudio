"""Speech recognizers used by the "max" quality mode (ensemble.py).

Every backend has the same shape:

    b = make(member)        # member: an entry of ensemble.MEMBERS
    b.load(note)            # note(msg): progress text while waiting for the GPU etc.
    b.recognize(x, lang, prompt) -> {"text": str, "words": [{w, s, e, p}] | None, "language": str | None}
    b.unload()

x is a 16 kHz float32 chunk (<= 25 s); word times are seconds from the chunk start; "words"
is None for models that only give text (Canary, Qwen3-ASR). p is the model's own word
probability (None if unknown). Only one heavy model is loaded at a time.
"""
from __future__ import annotations


def make(member: dict):
    kind = member["backend"]
    if kind == "whisper":
        from .whisper import WhisperBackend

        return WhisperBackend(member)
    if kind == "onnx":
        from .onnx_models import OnnxBackend

        return OnnxBackend(member)
    if kind == "qwen":
        from .qwen import QwenBackend

        return QwenBackend(member)
    raise ValueError(f"unknown backend {kind}")
