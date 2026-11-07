"""diarize: refine, word assignment, the owner's voice, rejoin (Опыт 11). Synthetic inputs, no model.

    set CUDA_VISIBLE_DEVICES=-1
    python -m unittest tests.test_diarize -v
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
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-diar-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_diarize.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

import numpy as np  # noqa: E402

from whisper_studio import diarize as D  # noqa: E402
from whisper_studio import diarize_ort as O  # noqa: E402

A = np.array([1.0, 0.0, 0.0], np.float32)
B = np.array([0.0, 1.0, 0.0], np.float32)
NOPROF = dict(emb=np.zeros(3, np.float32), name="x")


def near(v, eps=0.05):
    return D._unit(v + eps * np.array([0.0, 0.0, 1.0], np.float32))


def two_voices():
    """Raw 0 = voice A, raw 1 = voice B (long), raw 2 = B's short replies inside A's talk (A
    before and after, the nearest in time): one of 0.8 s with an embedding, two of 0.4 s without."""
    turns, temb = [], {}
    t = 0.0

    def add(d, k, v):
        nonlocal t
        turns.append((t, t + d, k))
        if v is not None:
            temb[len(turns) - 1] = v
        t += d + 0.1

    for r in range(6):
        add(10.0, 0, near(A))
        add(6.0, 1, near(B))
        if r in (1, 3, 5):
            add(10.0, 0, near(A))
            add(0.8 if r == 1 else 0.4, 2, near(B, 0.1) if r == 1 else None)
            add(10.0, 0, near(A))
    return turns, temb


class Refine(unittest.TestCase):
    def test_small_cluster_goes_whole_to_closest_voice(self):
        turns, temb = two_voices()
        out, info = D.refine(turns, temb, 2)
        b = {k for (_s, _e, k0), (_s2, _e2, k) in zip(turns, out) if k0 == 1}
        self.assertEqual(len(b), 1)
        b = b.pop()
        for (_s, _e, k0), (_s2, _e2, k) in zip(turns, out):
            if k0 == 2:
                self.assertEqual(k, b)  # the 0.4 s turns too, though A is nearer in time
        self.assertEqual(info["folded_whole"], 1)

    def test_legacy_turn_by_turn(self):
        turns, temb = two_voices()
        out, _ = D.refine(turns, temb, 2, legacy={"a"})
        a = next(k for (_s, _e, k0), (_s2, _e2, k) in zip(turns, out) if k0 == 0)
        short = [k for (s, e, k0), (_s2, _e2, k) in zip(turns, out) if k0 == 2 and e - s < 0.6]
        self.assertTrue(short and all(k == a for k in short))  # to the time neighbour, as before

    def test_cluster_of_short_turns_uses_its_joint_voice(self):
        turns, temb = two_voices()
        for i, (s, e, k) in enumerate(turns):  # no embedding of its own in raw 2
            if k == 2:
                temb.pop(i, None)
        out, _ = D.refine(turns, temb, 2)
        a = next(k for (_s, _e, k0), (_s2, _e2, k) in zip(turns, out) if k0 == 0)
        self.assertTrue(all(k == a for (_s, _e, k0), (_s2, _e2, k) in zip(turns, out) if k0 == 2))
        temb[-1 - 2] = near(B, 0.1)  # turn_embeddings: the cluster's short turns together
        out, _ = D.refine(turns, temb, 2)
        self.assertTrue(all(k != a for (_s, _e, k0), (_s2, _e2, k) in zip(turns, out) if k0 == 2))

    def test_short_clusters(self):
        sr = 16000
        turns = [(0.0, 5.0, 0), (5.0, 5.4, 1), (6.0, 6.3, 1), (7.0, 7.2, 2), (8.0, 8.4, 0)]
        sc = D.short_clusters(turns, 10 * sr)
        self.assertEqual(set(sc), {1})  # 0 has a long turn, 2 has 0.2 s < MIN_CLUSTER
        self.assertEqual(sc[1], [(int(5.0 * sr), int(5.4 * sr)), (int(6.0 * sr), int(6.3 * sr))])


def seg(words):
    return dict(start=words[0][1], end=words[-1][2], text=" ".join(w for w, _s, _e in words),
                words=[dict(w=w, s=s, e=e) for w, s, e in words])


class Activity(unittest.TestCase):
    def extra(self, n=200, dt=0.05):
        act = np.zeros((n, 3), np.uint8)
        return dict(act=act, act_t0=np.float64(0.0), act_dt=np.float64(dt))

    def test_word_outside_turns(self):
        turns = [(0.0, 2.0, 0), (2.9, 5.0, 1)]
        ex = self.extra()
        ex["act"][int(2.3 / 0.05):int(2.7 / 0.05), 2] = 200  # raw 2, folded into speaker 0
        segs = [seg([("a", 1.0, 1.5), ("b", 2.3, 2.6), ("c", 3.5, 4.0)])]
        D.assign_words(segs, turns)
        self.assertEqual(segs[0]["words"][1]["spk"], 1)  # nearest turn
        act = D.Activity(ex, {0: 0, 1: 1, 2: 0})
        D.assign_words(segs, turns, act)
        self.assertEqual([w["spk"] for w in segs[0]["words"]], [0, 0, 1])
        self.assertEqual(act.used, 1)

    def test_silence_falls_back_to_nearest_turn(self):
        turns = [(0.0, 2.0, 0), (2.9, 5.0, 1)]
        ex = self.extra()
        ex["act"][:, 0] = 20  # below ACT_MIN
        segs = [seg([("b", 2.5, 2.8)])]
        D.assign_words(segs, turns, D.Activity(ex, {0: 0, 1: 1}))
        self.assertEqual(segs[0]["words"][0]["spk"], 1)

    def test_overlapping_turns(self):
        turns = [(0.0, 5.0, 0), (2.0, 3.0, 1)]
        ex = self.extra()
        ex["act"][:, 0] = 230
        ex["act"][int(2.0 / 0.05):int(3.0 / 0.05), 1] = 60
        segs = [seg([("x", 2.2, 2.6)])]
        D.assign_words(segs, turns)
        self.assertEqual(segs[0]["words"][0]["spk"], 0)
        turns = [(0.0, 5.0, 1), (2.0, 3.0, 0)]  # equal overlap: the lower label won the tie
        ex["act"][:] = 0
        ex["act"][:, 1] = 230
        ex["act"][int(2.0 / 0.05):int(3.0 / 0.05), 0] = 60
        D.assign_words(segs, turns)
        self.assertEqual(segs[0]["words"][0]["spk"], 0)
        D.assign_words(segs, turns, D.Activity(ex, {0: 0, 1: 1}))
        self.assertEqual(segs[0]["words"][0]["spk"], 1)  # the segmentation hears 1 there

    def test_cache_round_trip(self):
        ex = self.extra()
        ex["act"][3, 1] = 7
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "voices.npz"
            D.save_cache_atomic(p, [(0.0, 1.0, 0)], {0: A, -2: B}, ex)
            got: dict = {}
            turns, temb = D._load_cache(p, got)
            self.assertEqual(turns, [(0.0, 1.0, 0)])
            self.assertEqual(sorted(temb), [-2, 0])
            self.assertEqual(int(got["act"][3, 1]), 7)
            D.save_cache_atomic(p, [(0.0, 1.0, 0)], {0: A})  # sherpa: no activity
            got = {}
            D._load_cache(p, got)
            self.assertEqual(got, {})

    def test_ort_activity_arrays(self):
        class Seg:
            rf_shift, rf_size, num_speakers = 270, 991, 3
            mapping = O._powerset(7, 3, 2)

        logp = np.full((50, 7), -30.0, np.float32)
        logp[:, 1] = 0.0  # local speaker 0 alone
        logp[20:, 1], logp[20:, 4] = -30.0, 0.0  # then 0 and 1 together
        loc = O._local_activity(logp, Seg)
        self.assertAlmostEqual(float(loc[0, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(loc[30, 1]), 1.0, places=4)
        self.assertAlmostEqual(float(loc[0, 2]), 0.0, places=4)
        ex: dict = {}
        O._put_activity(ex, loc, Seg)
        self.assertEqual(ex["act"].dtype, np.uint8)
        self.assertEqual(int(ex["act"][0, 0]), 255)
        self.assertAlmostEqual(float(ex["act_dt"]), 270 / 16000)


class Constrain(unittest.TestCase):
    def test_two_locals_of_a_window_split(self):
        rng = np.random.default_rng(0)
        emb = np.stack([near(A, 0.02 * i) for i in range(4)] + [near(B, 0.02 * i) for i in range(4)])
        emb = emb + 0.001 * rng.standard_normal(emb.shape).astype(np.float32)
        pairs = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
        cl = np.array([0, 0, 0, 0, 1, 1, 1, 0])  # window 3: both locals in cluster 0
        out, fixed = O.constrain(pairs, emb, cl)
        self.assertEqual(fixed, 1)
        self.assertEqual(out.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_untouched_without_conflict_or_room(self):
        emb = np.stack([A, B, A, near(A), near(A, 0.1)]).astype(np.float32)
        pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        cl = np.array([0, 1, 0, 0, 0])  # window 1: 3 locals > 2 clusters
        out, fixed = O.constrain(pairs, emb, cl)
        self.assertEqual(fixed, 0)
        self.assertEqual(out.tolist(), cl.tolist())

    def test_default_off(self):
        self.assertFalse(O.CONSTRAIN)


class Owner(unittest.TestCase):
    def prof(self):
        return dict(emb=A, name="Миша")

    def test_owner_speaker(self):
        p = self.prof()
        self.assertEqual(D.owner_speaker({"S1": near(A), "S2": B}, p)[0], "S1")
        self.assertIsNone(D.owner_speaker({"S1": near(A), "S2": near(A, 0.1)}, p))  # no margin
        self.assertIsNone(D.owner_speaker({"S1": D._unit(A + B)}, p))   # alone: OWNER_SOLO
        self.assertEqual(D.owner_speaker({"S1": near(A)}, p)[0], "S1")
        # people call the owner by name, the owner does not
        self.assertIsNone(D.owner_speaker({"S1": near(A), "S2": B}, p, {"S1": 2}))
        self.assertEqual(D.owner_speaker({"S1": near(A), "S2": B}, p, {"S1": 1})[0], "S1")

    def test_label_and_name_owner(self):
        turns = [(0.0, 30.0, 0), (30.0, 60.0, 1), (60.0, 90.0, 0)]
        temb = {0: near(B), 1: near(A), 2: near(B)}
        segs = [seg([("Миша,", 0.5, 1.0), ("слушай", 1.2, 2.0)]), seg([("понял", 33.5, 34.0)]),
                seg([("ладно", 66.5, 67.0)])]
        out, info = D.label(segs, turns, temb, 2, None, (), self.prof())
        self.assertEqual([s["speaker"] for s in out], ["S1", "S2", "S1"])  # order of appearance
        self.assertEqual(info["owner"]["speaker"], "S2")
        doc = dict(segments=out, speakers={"S1": dict(name="Спикер 1"), "S2": dict(name="Спикер 2")})
        D.save_owner_profile(D.owner_path(), A, "Миша", doc="test")
        try:
            self.assertEqual(D.name_owner(doc, info), "S2")
            self.assertEqual(doc["speakers"]["S2"]["name"], "Миша")
            doc["speakers"] = {"S1": dict(name="Миша"), "S2": dict(name="Спикер 2")}  # rediarize kept it
            D.name_owner(doc, info)
            names = (doc["speakers"]["S1"]["name"], doc["speakers"]["S2"]["name"])
            self.assertEqual(names, ("Спикер 1", "Миша"))
            doc["speakers"]["S2"]["name"] = "Сосед"  # the owner's own name for it stays
            self.assertIsNone(D.name_owner(doc, info))
            self.assertEqual(doc["speakers"]["S2"]["name"], "Сосед")
        finally:
            D.owner_path().unlink()

    def test_no_profile(self):
        self.assertIsNone(D.owner_profile(Path(tempfile.gettempdir()) / "no-such-owner.npz"))
        segs = [seg([("a", 0.5, 1.0)])]
        _out, info = D.label(segs, [(0.0, 3.0, 0)], {0: A}, None, None, (), NOPROF)
        self.assertNotIn("owner", info)


class Rejoin(unittest.TestCase):
    def pieces(self, a, b):
        p = dict(id="g0", src=0, start=0.0, end=1.0, text="один два", words=[], speaker="S1", **a)
        s = dict(id="g1", src=0, start=1.0, end=2.0, text="три", words=[], speaker="S2", **b)
        return D.rejoin([p, s])

    def test_orig_text_joins(self):
        (m,) = self.pieces(dict(edited=True, by="owner", orig_text="адин два"), {})
        self.assertEqual(m["text"], "один два три")
        self.assertEqual(m["orig_text"], "адин два три")
        (m,) = self.pieces({}, {})
        self.assertNotIn("orig_text", m)
        self.assertNotIn("by", m)

    def test_owner_only_when_every_piece_is(self):
        from whisper_studio import store

        (m,) = self.pieces(dict(edited=True, by="owner"), dict(edited=True, by="owner"))
        self.assertTrue(store.by_owner(m))
        (m,) = self.pieces(dict(edited=True, by="owner"), {})
        self.assertTrue(m["edited"])
        self.assertFalse(store.by_owner(m))
        (m,) = self.pieces(dict(edited=True, by="owner"), dict(edited=True, by="rule"))
        self.assertEqual(m["by"], "rule")
        (m,) = self.pieces(dict(edited=True, ai=True), dict(edited=True, by="owner"))
        self.assertEqual(m["by"], "ai")
        self.assertFalse(store.by_owner(m))


if __name__ == "__main__":
    unittest.main()
