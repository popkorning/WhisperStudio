"""Edits keep who made them and what the models heard; the dictionary survives two writers.

CPU only, stdlib. Run from the repo root:

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-agent-store
    python -m unittest tests.test_store -v

WHISPER_STUDIO_HOME must not be the real home: the tests create documents there.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-store-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_store.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

from whisper_studio import settings, store  # noqa: E402


def make(*texts: str) -> str:
    doc = store.create("test", "test.wav", {})
    segs = []
    for i, t in enumerate(texts):
        ws = [dict(w=w, s=i * 10 + k, e=i * 10 + k + 0.5, p=0.9) for k, w in enumerate(t.split())]
        segs.append(dict(id=f"g{i + 1}", start=i * 10, end=i * 10 + 9, text=t, words=ws, speaker="S1"))
    doc.update(status="done", segments=segs, speakers={"S1": dict(name="Спикер 1", color="#000")})
    store.save(doc, snapshot=False)
    return doc["id"]


def seg(doc_id: str, sid: str) -> dict:
    return next(s for s in store.load(doc_id)["segments"] if s["id"] == sid)


class EditSource(unittest.TestCase):
    def test_owner_edit_keeps_original(self):
        d = make("пришёл кесслер у друзей")
        store.edit_text(d, "g1", "пришёл Wohnung у друзей")
        s = seg(d, "g1")
        self.assertEqual(s["orig_text"], "пришёл кесслер у друзей")
        self.assertEqual(s["by"], "owner")
        self.assertTrue(store.by_owner(s))
        store.edit_text(d, "g1", "пришла Wohnung у друзей")  # the first original stays
        self.assertEqual(seg(d, "g1")["orig_text"], "пришёл кесслер у друзей")

    def test_unchanged_text_has_no_original(self):
        d = make("всё верно")
        store.edit_text(d, "g1", "всё верно")
        self.assertNotIn("orig_text", seg(d, "g1"))
        self.assertTrue(store.by_owner(seg(d, "g1")))

    def test_machine_edits_are_not_the_owners(self):
        d = make("один кесслер", "два кесслер", "три кесслер", "четыре кесслер")
        store.apply_corrections(d, {"g1": "один Wohnung"})                 # AI proofreading
        store.edit_text(d, "g2", "два Wohnung", by="ai")                  # MCP edit_segment
        store.replace_all(d, "три", "3")                                   # find & replace
        self.assertEqual([seg(d, g).get("by") for g in ("g1", "g2", "g3")], ["ai", "ai", "replace"])
        self.assertTrue(all(seg(d, g).get("orig_text") for g in ("g1", "g2", "g3")))
        self.assertFalse(any(store.by_owner(seg(d, g)) for g in ("g1", "g2", "g3", "g4")))
        store.edit_text(d, "g1", "один Wohnung")  # the owner confirmed the AI's text
        self.assertTrue(store.by_owner(seg(d, "g1")))

    def test_old_documents(self):
        self.assertTrue(store.by_owner(dict(edited=True)))
        self.assertFalse(store.by_owner(dict(edited=True, ai=True)))
        self.assertFalse(store.by_owner(dict(text="x")))


class SpeakerLock(unittest.TestCase):
    def test_hand_set_speaker_survives_rediarization(self):
        from whisper_studio import pipeline

        d = make("алло да", "понял хорошо", "значит так", "окей")
        store.set_speaker(d, "g2", None, "Миша")  # by hand: a new speaker S2
        store.rename_speaker(d, "S1", "Frau Kessler")
        doc = store.load(d)
        self.assertTrue(seg(d, "g2")["speaker_locked"])
        old, old_speakers = [dict(s) for s in doc["segments"]], dict(doc["speakers"])
        names = {k: v["name"] for k, v in old_speakers.items()}
        # the voices put everything into one speaker, numbered the other way round
        new = [dict(s, speaker="X") for s in old]
        new[3]["speaker"] = "Y"
        store.finalize(doc, new)
        same = pipeline._match_speakers(old, doc["segments"])
        pipeline._keep_locked(doc, old, old_speakers, same)
        by = {s["text"]: (doc["speakers"][s["speaker"]]["name"], s["speaker"]) for s in doc["segments"]}
        self.assertEqual(by["понял хорошо"][0], "Миша")
        self.assertNotEqual(by["понял хорошо"][1], by["алло да"][1])
        self.assertEqual(same["S1"], by["алло да"][1])
        self.assertEqual(names["S1"], "Frau Kessler")


ADDER = """
import sys
from whisper_studio import settings
for i in range(100):
    settings.edit_dictionary(lambda d: d["terms"].append(f"{sys.argv[1]}{i}"))
"""


class Dictionary(unittest.TestCase):
    def setUp(self):
        settings.DICTIONARY_FILE.unlink(missing_ok=True)

    def test_two_processes_lose_nothing(self):
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
               "WHISPER_STUDIO_HOME": str(settings.DICTIONARY_FILE.parent)}  # the home this module imported
        ps = [subprocess.Popen([sys.executable, "-c", ADDER, tag], env=env) for tag in ("a", "b")]
        self.assertEqual([p.wait(120) for p in ps], [0, 0])
        d = settings.dictionary()
        self.assertEqual(len(d["terms"]), 200)
        self.assertEqual(d["rev"], 200)

    def test_stale_save_is_refused(self):
        base = settings.dictionary()
        settings.edit_dictionary(lambda d: d["terms"].append("Stadt"))   # an agent, meanwhile
        with self.assertRaises(settings.Conflict):
            settings.save_dictionary({**base, "terms": ["Wohnung"]}, base["rev"])
        self.assertEqual(settings.dictionary()["terms"], ["Stadt"])
        settings.save_dictionary({"terms": ["Stadt", "Wohnung"]}, settings.dictionary()["rev"])
        self.assertEqual(settings.dictionary()["terms"], ["Stadt", "Wohnung"])


if __name__ == "__main__":
    unittest.main()
