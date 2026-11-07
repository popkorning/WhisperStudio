"""The word vote of the ensemble (whisper_studio/ensemble.py) on hand-made slots. CPU only, no
models:

    python -m unittest tests.test_vote -v
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
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-vote-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_vote.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

from whisper_studio import ensemble as E  # noqa: E402
from whisper_studio.asr import text as T  # noqa: E402

TERMS = E._Terms(["Erholung", "Hauptbahnhof", "Gartenhaus Sonnenberg", "ICE"])


def slot(*words: str) -> tuple[E._Slot, dict, list]:
    s = E._Slot()
    names = [f"m{i}" for i in range(len(words))]
    for m, w in zip(names, words):
        s.add(dict(w=w, s=1.0, e=1.4, p=0.9, timed=True, m=m, k=T.key(w), lat=T.is_latin(w), first=False))
    return s, {m: 1.0 for m in names}, names


def win(*words: str) -> str | None:
    s, present, order = slot(*words)
    r = E._vote(s, present, order, term_keys=TERMS.exact)
    return r and r["tok"]["w"]


class DictionaryTerm(unittest.TestCase):
    def test_term_beats_german_misspelling_and_unknown_cyrillic(self):
        # lab experiment 16: large-v3 and ru-large-v3 wrote Erholung, de-turbo «Statum»,
        # the Russian models a Cyrillic word that gave its votes to «Statum»
        self.assertEqual(win("Erholung", "Erholung", "Statum", "Statum", "аштатум", "аштатум"), "Erholung")

    def test_no_bonus_over_a_russian_word(self):
        # large-v3 repeating its prompt over «да»
        self.assertEqual(win("Hauptbahnhof", "да", "да"), "да")

    def test_no_sound_check_for_short_terms(self):
        # «вот» sounds like ICE, which says nothing for a three-letter word
        self.assertEqual(win("ICE", "вот", "вот"), "вот")

    def test_off_switch(self):
        old, E.DICT_TERM = E.DICT_TERM, 0.0
        try:
            self.assertEqual(win("Erholung", "Erholung", "Statum", "Statum", "аштатум", "аштатум"), "Statum")
        finally:
            E.DICT_TERM = old


class SplitPiece(unittest.TestCase):
    @staticmethod
    def w(word: str, s: float, voters: list[str]) -> dict:
        return dict(w=word, s=s, e=s + 0.3, _k=T.key(word), _lat=T.is_latin(word), _voters=voters, _score=1.0)

    def test_piece_of_the_word_before_is_dropped(self):
        ws = [self.w("Gartenhaus", 1.0, ["a", "b"]), self.w("Haus", 1.3, ["c", "d", "e"]),
              self.w("Sonnenberg", 1.6, ["a", "b", "c"])]
        self.assertEqual([x["w"] for x in E._dedupe(ws)], ["Gartenhaus", "Sonnenberg"])

    def test_same_models_saying_both_is_kept(self):
        ws = [self.w("Gartenhaus", 1.0, ["a", "b"]), self.w("Haus", 1.3, ["a", "c"])]
        self.assertEqual(len(E._dedupe(ws)), 2)


if __name__ == "__main__":
    unittest.main()
