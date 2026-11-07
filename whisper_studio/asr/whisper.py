"""Whisper (faster-whisper / CTranslate2) on one speech chunk."""
from __future__ import annotations

from .. import engine


class WhisperBackend:
    def __init__(self, member: dict):
        self.member = member
        self.key = member["model"]
        self.m = None
        self.last: dict = {}  # the last call: temperature of each segment (tools/bench_members.py)

    @property
    def device(self) -> str | None:
        return engine._loaded.get("dev") if self.m is not None else None

    def load(self, note=None):
        # only the language detector may fall back to the (much slower) CPU; the other
        # members are skipped when the GPU stays busy: the ensemble works without them
        self.m = engine.load_whisper(self.key, note, cpu_fallback=self.member.get("cpu_ok", False))

    def unload(self):
        self.m = None
        engine.unload()

    def detect(self, x) -> dict[str, float]:
        """Language probabilities of the chunk (Whisper's language-ID head)."""
        _, _, probs = self.m.detect_language(x, language_detection_segments=1)
        return {k: round(float(v), 4) for k, v in probs[:8]}

    def detect_parts(self, x, win: float = 6.0) -> list[dict[str, float]]:
        """Language of the first and the last seconds: a chunk can begin in German and go on
        in Russian, and the whole-chunk guess (and the text) then hides the German part."""
        n = int(win * 16000)
        if len(x) < 2 * n:
            return []
        return [self.detect(x[:n]), self.detect(x[-n:])]

    def recognize(self, x, lang: str | None, prompt: str | None) -> dict:
        # a member may override the decoding (tools/bench_members.py tries cheaper settings)
        params = dict(
            language=lang, task="transcribe", beam_size=self.member.get("beam_size", 5), best_of=5, patience=1.0,
            temperature=tuple(self.member.get("temperature", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))),
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
            # each chunk is decoded on its own; the prompt carries the glossary and the previous words
            condition_on_previous_text=False, initial_prompt=prompt,
            word_timestamps=True, vad_filter=False, multilingual=lang is None,
        )
        segs, info = self.m.transcribe(x, **params)
        words, texts, flags, temps = [], [], [], []
        for s in segs:
            temps.append(s.temperature)
            f = engine.suspicious(s.text, s.no_speech_prob, s.avg_logprob, s.compression_ratio)
            if "phantom" in f or "repetition" in f:
                flags += f
                continue
            texts.append(s.text.strip())
            words += engine.words_of(s)
        self.last = dict(temperatures=temps)
        return dict(text=" ".join(t for t in texts if t), words=words, language=info.language,
                    flags=sorted(set(flags)) or None)
