"""Qwen3-ASR 1.7B (native in transformers >= 5) on the GPU in fp16. Text only, no word times.

RTX 2060 (Turing): fp16 tensor cores yes, bf16 no; attention through PyTorch sdpa (no flash
attention on Turing). Environment (measured on the gold reference with tools/bench_members.py,
docs/DECISIONS.md H4):

    WHISPER_STUDIO_QWEN_BATCH         chunks per generate() call (default 4: ×2.4 faster, the same
                                      merge; 8 is no faster); left padding
    WHISPER_STUDIO_QWEN_TOKENS_PER_S  cap of new tokens per second of audio (0 = always 448):
                                      only a repetition loop ever reaches it
"""
from __future__ import annotations

import gc
import os
import time

from .. import models
from ..config import SAMPLE_RATE
from . import gpu

VRAM_GB = 4.2  # 1.7B parameters in fp16 = 3.4 GB + decoding; a 6 GB card with the desktop has ~4.8 free
MAX_NEW = 448
BATCH_SECONDS = 100.0  # audio per batch: the KV cache grows with audio + text tokens of every item


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


class QwenBackend:
    batch_seconds = BATCH_SECONDS
    device = "cuda"

    def __init__(self, member: dict):
        self.member = member
        self.key = member["model"]
        self.model = None
        self.proc = None
        self.batch = max(1, _int("WHISPER_STUDIO_QWEN_BATCH", 4))
        self.tokens_per_s = _int("WHISPER_STUDIO_QWEN_TOKENS_PER_S", 0)
        self.last: dict = {}  # the last call: new tokens, seconds (tools/bench_members.py)

    def load(self, note=None):
        import torch

        AutoProcessor, Qwen3ASRForConditionalGeneration = _classes()

        path = models.find(self.key)
        if path is None:
            raise RuntimeError(f"Модель {self.key} не установлена")
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-ASR нужна видеокарта")
        if not gpu.wait(VRAM_GB, note, torch_cache=True):
            raise RuntimeError("видеопамять так и не освободилась")
        self.proc = AutoProcessor.from_pretrained(str(path))
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            str(path), dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa").to("cuda").eval()

    def unload(self):
        self.model = None
        self.proc = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _max_new(self, xs: list) -> int:
        if self.tokens_per_s <= 0:
            return MAX_NEW
        longest = max(len(x) for x in xs) / SAMPLE_RATE
        return min(MAX_NEW, int(longest * self.tokens_per_s) + 32)

    def _generate(self, xs: list, lang: str | None, prompt: str | None) -> list[dict]:
        import torch

        n_items = len(xs)
        one = n_items == 1
        inputs = self.proc.apply_transcription_request(
            xs[0] if one else list(xs), language=lang if one or lang is None else [lang] * n_items,
            prompt=(prompt or None) if one else None)
        inputs = inputs.to("cuda")
        for k, v in inputs.items():
            if hasattr(v, "is_floating_point") and v.is_floating_point():
                inputs[k] = v.to(torch.float16)
        n = inputs["input_ids"].shape[1]
        t0 = time.monotonic()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self._max_new(xs), do_sample=False,
                                      repetition_penalty=1.05)
        torch.cuda.synchronize()
        pad = self.model.generation_config.pad_token_id
        new = out[:, n:]
        self.last = dict(items=n_items, prompt_tokens=int(n), seconds=round(time.monotonic() - t0, 3),
                         new_tokens=[int((row != pad).sum()) for row in new])
        res = []
        for row in new:
            r = self.proc.decode(row, return_format="parsed")
            if lang:  # forced language: the prefill isn't part of the generated text
                r = dict(language=lang, transcription=self.proc.decode(row, skip_special_tokens=True))
            res.append(dict(text=(r.get("transcription") or "").strip(), words=None,
                            language=_code(r.get("language"))))
        return res

    def recognize(self, x, lang: str | None, prompt: str | None) -> dict:
        return self._generate([x], lang, prompt)[0]

    def recognize_batch(self, xs: list, lang: str | None, prompt: str | None) -> list[dict]:
        """Several chunks of the same language in one generate() call (no prompt)."""
        return self._generate(list(xs), lang, None)


def _classes():
    """The worker imports transformers in a background thread (engine.prewarm); a second
    import racing it sees a half-initialised lazy module ("cannot import name"): wait, retry."""
    for attempt in range(30):
        try:
            from transformers.models.auto.processing_auto import AutoProcessor
            from transformers.models.qwen3_asr.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration

            return AutoProcessor, Qwen3ASRForConditionalGeneration
        except ImportError:
            if attempt == 29:
                raise
            time.sleep(2)


_NAMES = {"russian": "ru", "german": "de", "english": "en", "ukrainian": "uk", "polish": "pl", "turkish": "tr"}


def _code(name: str | None) -> str | None:
    if not name:
        return None
    return _NAMES.get(name.strip().lower(), name.strip().lower()[:2])
