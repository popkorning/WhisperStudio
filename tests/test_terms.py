"""Post-merge term recovery (whisper_studio/terms.py): a preposition the models glued onto a
German term («пошпильплац» = по Spielplatz) is cut off for the sound comparison and kept as a word
of its own; a plain Russian phrase is left alone. CPU only, no models:

    python -m unittest tests.test_terms -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-terms-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_terms.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

from whisper_studio import terms  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402


def words(text: str, t0: float = 0.0, p: float = 0.9, step: float = 0.5) -> list[dict]:
    return [dict(w=w, s=t0 + i * step, e=t0 + i * step + 0.4, p=p) for i, w in enumerate(text.split())]


def seg(text: str, low: str = "") -> dict:
    ws = words(text)
    for w in ws:
        if w["w"].strip(",.") == low:
            w["p"] = 0.2
    return dict(words=ws, text=text, lang="ru")


class Glued(unittest.TestCase):
    def test_stems_cut_the_preposition(self):
        k = T.key("пошпильплац")
        self.assertIn(("шпильплац", "", "по"), terms.stems(k, k))
        self.assertEqual(terms.glued(T.key("пол")), [])  # too short to be a preposition + word

    def test_near_finds_the_term_behind_the_preposition(self):
        idx = terms.Index(["Spielplatz", "Wohnung"])
        self.assertIn("Spielplatz", [t for _, t in idx.near(T.key("пошпильплац"))])

    def test_recover_keeps_the_preposition(self):
        s = seg("Слушай, пошпильплац надо проверить", low="пошпильплац")
        members = {m: words(f"Слушай, по {w} надо проверить")
                   for m, w in (("a", "Шпильплац"), ("b", "Шпильплаз"), ("c", "Шпилплац"))}
        n = terms.recover([s], ["Spielplatz"], members)
        self.assertEqual(n, 1)
        self.assertEqual(s["text"], "Слушай, по Spielplatz надо проверить")
        by = {w["w"]: w for w in s["words"]}
        self.assertTrue(by["Spielplatz"].get("term"))
        self.assertLessEqual(by["по"]["e"], by["Spielplatz"]["s"])

    def test_russian_phrase_untouched(self):
        s = seg("Слушай, поставь надо проверить")
        members = {m: words("Слушай, поставь надо проверить") for m in "abc"}
        self.assertEqual(terms.recover([s], ["Spielplatz", "Stadt"], members), 0)
        self.assertEqual(s["text"], "Слушай, поставь надо проверить")

    def test_hyphenated_russian_words_are_russian(self):
        # «где-то» used to become «гдето», unknown, and lost its votes to «geht» (lab experiment 16)
        for w in ("Где-то", "какие-то", "из-за", "во-первых,"):
            self.assertIs(T.known_russian(w), True, w)
        self.assertIs(T.known_russian("вонунг-то"), False)

    def test_off_switch(self):
        terms.GLUED = False
        try:
            self.assertEqual(terms.glued(T.key("пошпильплац")), [])
        finally:
            terms.GLUED = True


if __name__ == "__main__":
    unittest.main()
