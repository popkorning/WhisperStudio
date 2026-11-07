"""The local proofreading model may only use what a recognizer heard or the dictionary knows.

CPU only, no model (the chat is faked). Run from the repo root:

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-agent-store
    python -m unittest tests.test_llm -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library
from unittest import mock

if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-llm-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_llm.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

from whisper_studio import llm, settings, store  # noqa: E402

SEG = dict(id="g1", text="пришёл кесслер у друзей на 25 евро",
           alts=[dict(model="gigaam-v3", text="пришёл вонунг у друзей на 25 евро"),
                 dict(model="qwen3-asr", text="пришла кес лер у друзей на 25 евро")])


class Refused(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings.save_dictionary(dict(terms=["Wohnung"], replace=[{"from": "вонунг", "to": "Wohnung"}]))

    def test_allowed(self):
        for new in ("пришёл вонунг у друзей на 25 евро",     # a member's word
                    "пришёл Wohnung у друзей на 25 евро",    # the dictionary's term
                    "пришла кесслер у друзей на 25 евро",     # an ending
                    "Пришёл кесслер, у друзей, на 25 евро."):  # case and punctuation
            self.assertIsNone(llm.refused(SEG, new), new)

    def test_refused(self):
        self.assertEqual(llm.refused(SEG, "пришёл кесслер у друзей на 35 евро"), "числа")
        self.assertIn("новое слово", llm.refused(SEG, "пришёл ключ у друзей на 25 евро"))
        self.assertIsNotNone(llm.refused(SEG, "на 25 евро"))  # most of the segment dropped

    def test_proofread_keeps_only_allowed_fixes(self):
        doc = store.create("t", "t.wav", {})
        doc.update(status="done", segments=[dict(SEG, start=0, end=5, words=[]),
                                            dict(SEG, id="g2", start=6, end=9, words=[])])
        reply = json.dumps({"g1": "пришёл Wohnung у друзей на 25 евро",
                            "g2": "пришёл ключ у друзей на 125 евро"})
        with mock.patch.object(llm, "status", return_value={"available": True}), \
                mock.patch.object(llm, "_chat", return_value=reply):
            self.assertEqual(llm.proofread_doc(doc), 1)
        self.assertEqual(doc["segments"][0]["text"], "пришёл Wohnung у друзей на 25 евро")
        self.assertEqual(doc["segments"][1]["text"], SEG["text"])
        self.assertEqual([r["id"] for r in doc["meta"]["proofread_refused"]], ["g2"])


if __name__ == "__main__":
    unittest.main()
