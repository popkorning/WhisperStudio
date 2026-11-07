"""Tests never touch the real data folder.

Every test module runs with WHISPER_STUDIO_HOME pointing at a fresh temporary folder: the tests
create documents, rewrite the dictionary and the owner's voice profile, and delete them again.
This package module runs before any test module imports whisper_studio, also under
`python -m unittest discover`. Models are read from the real folder (WS_REAL_HOME, by default
%LOCALAPPDATA%\\WhisperStudio) through WHISPER_STUDIO_MODELS; nothing is written there.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_REAL = Path(os.environ.get("WS_REAL_HOME")
             or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")

if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-test-home-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == _REAL.resolve():
    raise SystemExit("tests: WHISPER_STUDIO_HOME is the real data folder; unset it or point it at a scratch folder")
if not os.environ.get("WHISPER_STUDIO_MODELS") and (_REAL / "models").is_dir():
    os.environ["WHISPER_STUDIO_MODELS"] = str(_REAL / "models")
