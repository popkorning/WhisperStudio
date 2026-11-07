"""Paths and global settings.

Runtime data (models, library of transcripts) lives outside the code folder,
in WHISPER_STUDIO_HOME (default: %LOCALAPPDATA%\\WhisperStudio).
"""
from __future__ import annotations

import os
from pathlib import Path


def _default_home() -> Path:
    env = os.environ.get("WHISPER_STUDIO_HOME")
    if env:
        return Path(env)
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio"


HOME = _default_home()
# WHISPER_STUDIO_MODELS: the real models with a scratch home (benchmarks, tools/queue_bench.py)
MODELS_DIR = Path(os.environ["WHISPER_STUDIO_MODELS"]) if os.environ.get("WHISPER_STUDIO_MODELS") else HOME / "models"
LIBRARY_DIR = HOME / "library"
TMP_DIR = HOME / "tmp"
HF_CACHE = HOME / "hf-cache"
SERVER_FILE = HOME / "server.json"

for _d in (MODELS_DIR, LIBRARY_DIR, TMP_DIR, HF_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# Downloads from Hugging Face go to the runtime folder
os.environ.setdefault("HF_HOME", str(HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

SAMPLE_RATE = 16000
WEB_DIR = Path(__file__).parent / "web"
