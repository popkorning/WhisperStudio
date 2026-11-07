"""`ws transcribe` takes what is not given from the user's defaults, as the UI and MCP do: with
"mixed" off a whole German stretch was recognized as Russian, that is translated (lab
experiment 16, test part). CPU only, no models:

    python -m unittest tests.test_cli -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-cli-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_cli.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

from whisper_studio import cli, settings  # noqa: E402


def opts(*args: str):
    return cli.transcribe_options(cli.parser().parse_args(["transcribe", "x.wav", *args]))


class Defaults(unittest.TestCase):
    def test_user_defaults_apply(self):
        d = settings.get()["defaults"]
        o = opts()
        self.assertEqual(o.mixed, bool(d.get("mixed")))
        self.assertEqual(o.profile, d.get("profile", "balanced"))
        self.assertTrue(o.diarize and o.align)
        self.assertFalse(o.stereo_split or o.highpass)

    def test_flags_win(self):
        o = opts("--no-mixed", "--lang", "de", "--speakers", "3", "--no-diarize", "--stereo", "--beam", "2")
        self.assertFalse(o.mixed)
        self.assertEqual((o.language, o.num_speakers, o.beam_size), ("de", 3, 2))
        self.assertFalse(o.diarize)
        self.assertTrue(o.stereo_split)
        self.assertTrue(opts("--mixed").mixed)
        self.assertIsNone(opts("--lang", "auto").language)


if __name__ == "__main__":
    unittest.main()
