"""Transcription options and job submission, without numpy/torch.

The UI server and the MCP server only create documents and queue them; the worker does
the heavy work. Keeping this module light keeps those processes quick to start
(on a slow disk every imported file costs a disk seek).
"""
from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import settings, store

PROFILES = {
    # Guide "Recommendations": max-accuracy faster-whisper config
    "balanced": dict(title="Универсальный", vad=True, condition=True),
    # Clean monologue / lecture: no VAD, keep context
    "clean": dict(title="Чистая запись, лекция", vad=False, condition=True),
    # Meetings, pauses, noise: soft VAD, no context (glossary moves to hotwords)
    "noisy": dict(title="Совещание, шум, паузы", vad=True, condition=False),
}


@dataclass
class Options:
    language: str | None = None          # "ru", "de", ... or None = detect
    model: str = "auto"                  # registry key or "auto"
    profile: str = "balanced"
    glossary: str = ""                   # names/terms -> initial_prompt + hotwords
    topic: str = ""                      # optional context sentence for the prompt
    beam_size: int = 5
    num_speakers: int | None = None      # None = auto
    diarize: bool = True
    align: bool = True                   # wav2vec2 word timestamps
    stereo_split: bool = False           # telephony: channel = speaker
    highpass: bool = False
    vad: bool | None = None              # override profile
    condition: bool | None = None        # override profile
    mixed: bool = False                  # Russian speech with German words/phrases: never translate
    quality: str = "max"                 # "max": ensemble of models (ensemble.py), "fast": one Whisper
    dictionary: bool = True              # project dictionary: prompt terms + replacements
    extra: dict = field(default_factory=dict)


def make_options(o: dict | None) -> Options:
    """Options from a request; anything left out falls back to the user's defaults,
    so the UI, MCP and CLI produce the same transcript for the same file."""
    allowed = Options.__dataclass_fields__.keys()
    o = {**settings.get()["defaults"], **{k: v for k, v in (o or {}).items() if v is not None}}
    o = {k: v for k, v in o.items() if k in allowed}
    if o.get("language") in ("", "auto"):
        o["language"] = None
    if o.get("num_speakers") in ("", "auto", 0):
        o["num_speakers"] = None
    if o.get("model") in ("", None):
        o["model"] = "auto"
    return Options(**o)


def prompt(opts: Options) -> tuple[str | None, str | None]:
    """Whisper initial_prompt and hotwords: the topic, then this recording's glossary and the
    project terms (key terms at the end; the prompt is a sample of preceding text)."""
    if opts.dictionary:
        terms = settings.prompt_terms(opts.glossary)
    else:
        terms = ", ".join(t.strip() for t in re.split(r"[,\n;]+", opts.glossary) if t.strip())
    if not terms and not opts.topic:
        return None, None
    parts = []
    if opts.topic:
        parts.append(opts.topic.strip().rstrip(".") + ".")
    if terms:
        parts.append(terms + ".")
    return " ".join(parts), (terms or None)


# what recognition depends on; the rest (speakers, alignment) is redone from the hypotheses
_RECOGNITION = ("language", "mixed", "quality", "profile", "beam_size", "highpass", "stereo_split",
                "vad", "condition", "extra")


def reuse_hypotheses(old: dict, new: dict) -> bool:
    """A new version of a "max" transcript recognized with the same settings and the same
    prompt starts from the old one's model hypotheses and speaker embeddings: only what is
    missing is recognized (minutes instead of a quarter of an hour), then merged again."""
    oo, no = dict(old.get("options") or {}), dict(new.get("options") or {})
    if no.get("mixed") and oo.get("mixed"):
        oo.pop("language", None), no.pop("language", None)  # mixed mode decides per chunk
    elif not no.get("language"):  # auto: the old run stored what it detected
        no["language"] = ((old.get("meta") or {}).get("detected_language") or {}).get("language")
    if no.get("quality") != "max" or any(oo.get(k) != no.get(k) for k in _RECOGNITION):
        return False
    if (old.get("meta") or {}).get("prompt") != prompt(Options(**no))[0]:
        return False
    src, dst = store.path(old["id"]), store.path(new["id"])
    copied = False
    for f in src.glob("hypotheses*.json"):
        shutil.copy2(f, dst / f.name)
        copied = True
    for f in src.glob("voices*.npz"):  # speaker embeddings depend on the audio only
        shutil.copy2(f, dst / f.name)
    return copied


def submit(path: str | Path, opts: Options, title: str | None = None) -> dict:
    """Create a queued document for a file (the caller queues the job, see jobs.submit_process)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return store.create(title or p.stem, str(p), asdict(opts))
