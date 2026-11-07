"""Queue speed-ups (docs/DECISIONS.md) give the same transcript as the plain path.

CPU only, no real models: the ensemble members are fakes (asr.make is patched) whose answer
depends only on what they hear (the chunk audio, the language they are told, the prompt), like
a real deterministic recognizer. Run from the repo root:

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-agent-pipeline
    set CUDA_VISIBLE_DEVICES=-1
    python -m unittest tests.test_speed -v

WHISPER_STUDIO_HOME must not be the real home: the tests create and delete documents there.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from collections import Counter
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-speed-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_speed.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")
for k in ("WHISPER_STUDIO_PREFETCH", "WHISPER_STUDIO_BATCH", "WHISPER_STUDIO_PARALLEL_CPU",
          "WHISPER_STUDIO_EARLY_VOICES", "WHISPER_STUDIO_DIARIZE_GPU"):
    os.environ.pop(k, None)

import numpy as np  # noqa: E402

from whisper_studio import asr, ensemble, jobs, models, pipeline, prefetch, store, worker  # noqa: E402
from whisper_studio.config import HOME, SAMPLE_RATE, TMP_DIR  # noqa: E402

WORDS = ("да нет может быть погода дорога поезд срок месяц город кино театр письмо ответ "
         "понедельник вторник вокзал музей парк билет вонунг штрассенбан").split()
DE_WORDS = "Wohnung Stadt Bahnhof Fahrkarte Garten Wetter Urlaub Termin".split()
FAKE_MODELS = TMP_DIR / "fake-models"


# ---------------------------------------------------------------- fakes

class Fake:
    loads: Counter = Counter()
    calls: Counter = Counter()
    load_delay = 0.0
    cpu_delay = 0.0          # seconds per chunk, every member (to see the lanes overlap)
    fail_load: set = set()
    on_load = None           # hook(name)
    active = Counter()       # members inside recognize() right now
    overlap = [0]            # times a CPU member ran while a GPU member did

    def __init__(self, member):
        self.member = member
        self.name = member["name"]

    def load(self, note=None):
        Fake.loads[self.name] += 1
        if Fake.on_load:
            Fake.on_load(self.name)
        if self.name in Fake.fail_load:
            raise RuntimeError("видеопамять занята другой программой")
        time.sleep(Fake.load_delay)

    def unload(self):
        pass

    def detect(self, x):
        de = _is_german(x)
        return {"de": 0.93, "ru": 0.05} if de else {"ru": 0.94, "de": 0.03}

    def detect_parts(self, x):
        n = 6 * SAMPLE_RATE
        return [] if len(x) < 2 * n else [self.detect(x[:n]), self.detect(x[-n:])]

    def recognize(self, x, lang, prompt):
        Fake.calls[self.name] += 1
        onnx = self.member["backend"] == "onnx"
        Fake.active["cpu" if onnx else "gpu"] += 1
        try:
            if onnx and Fake.active["gpu"]:
                Fake.overlap[0] += 1
            if Fake.cpu_delay:
                time.sleep(Fake.cpu_delay)
            h = hashlib.sha1(np.round(x[::7], 3).tobytes() + f"{self.name}|{lang}|{prompt}".encode()).digest()
            dur = len(x) / SAMPLE_RATE
            n = max(2, int(dur * 1.5))
            vocab = DE_WORDS if (lang == "de" or (lang is None and _is_german(x))) else WORDS
            words = []
            for i in range(n):
                # mostly the same words for every model (the speech), sometimes the model's own
                own = h[i % len(h)] % 5 == 0
                base = hashlib.sha1(np.round(x[::7], 3).tobytes() + bytes([i])).digest()[0]
                w = vocab[(h[i % len(h)] if own else base) % len(vocab)]
                s = dur * i / n
                words.append(dict(w=w, s=round(s, 3), e=round(s + dur / n * 0.8, 3), p=0.9))
            text = " ".join(w["w"] for w in words)
            no_words = self.member["backend"] == "qwen" or self.name == "canary-v2"
            return dict(text=text, words=None if no_words else words, language=lang)
        finally:
            Fake.active["cpu" if onnx else "gpu"] -= 1

    @classmethod
    def reset(cls):
        cls.loads, cls.calls = Counter(), Counter()
        cls.load_delay = cls.cpu_delay = 0.0
        cls.fail_load, cls.on_load = set(), None
        cls.active, cls.overlap = Counter(), [0]


def _is_german(x) -> bool:
    # German "speech" is the 900 Hz tone, Russian the 300 Hz one (zero crossings per second)
    if len(x) == 0:
        return False
    zc = np.count_nonzero(np.diff(np.signbit(x))) / (len(x) / SAMPLE_RATE)
    return zc > 1200


def _regions(audio):
    """Energy VAD instead of Silero (tones are no speech for it)."""
    hop = SAMPLE_RATE // 50
    n = len(audio) // hop
    on = np.sqrt((audio[: n * hop].reshape(n, hop) ** 2).mean(axis=1)) > 0.02
    out, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start * hop / SAMPLE_RATE, i * hop / SAMPLE_RATE))
            start = None
    if start is not None:
        out.append((start * hop / SAMPLE_RATE, n * hop / SAMPLE_RATE))
    return out


def _wav(path: Path, plan: list[tuple[float, float]]) -> Path:
    """plan: (seconds, Hz) pieces, 0 Hz = silence."""
    parts = []
    for sec, hz in plan:
        t = np.arange(int(sec * SAMPLE_RATE)) / SAMPLE_RATE
        parts.append(0.3 * np.sin(2 * np.pi * hz * t) if hz else np.zeros_like(t))
    pcm = (np.concatenate(parts) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return path


_real_find = models.find


def _find(key):
    if key in {m["model"] for m in ensemble.MEMBERS} | {ensemble.DE_CHECK["model"]}:
        d = FAKE_MODELS / key
        d.mkdir(parents=True, exist_ok=True)
        f = d / "model.bin"
        if not f.exists():
            f.write_bytes(os.urandom(1 << 20))
        return d
    return None  # no aligners, no diarization model


AUDIO = [
    [(1, 0), (6, 300), (0.8, 0), (7, 300), (3, 0), (5, 900), (3, 0), (9, 300), (1, 0)],
    [(0.5, 0), (12, 300), (0.6, 0), (14, 300), (2.5, 0), (4, 300), (1, 0)],
    [(1, 0), (4, 900), (2.5, 0), (8, 300), (0.4, 0), (20, 300), (2.2, 0), (6, 900), (1, 0)],
]


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._patches = [(asr, "make", Fake), (models, "find", _find), (ensemble, "speech_regions", _regions)]
        cls._saved = [(o, n, getattr(o, n)) for o, n, _ in cls._patches]
        for o, n, v in cls._patches:
            setattr(o, n, v)
        cls.src = TMP_DIR / "speed-audio"
        cls.src.mkdir(parents=True, exist_ok=True)
        cls.files = [_wav(cls.src / f"rec{i}.wav", plan) for i, plan in enumerate(AUDIO)]

    @classmethod
    def tearDownClass(cls):
        for o, n, v in cls._saved:
            setattr(o, n, v)

    def setUp(self):
        Fake.reset()
        for k in ("WHISPER_STUDIO_PREFETCH", "WHISPER_STUDIO_BATCH", "WHISPER_STUDIO_PARALLEL_CPU",
          "WHISPER_STUDIO_EARLY_VOICES", "WHISPER_STUDIO_DIARIZE_GPU"):
            os.environ.pop(k, None)
        prefetch.set_enabled(False)
        self.docs = []

    def tearDown(self):
        for d in self.docs:
            store.delete(d)

    def new_doc(self, f: Path, **opts) -> str:
        o = dict(language="ru", mixed=True, quality="max", align=False, diarize=False, num_speakers=2)
        o.update(opts)
        d = pipeline.submit(f, pipeline.make_options(o))
        self.docs.append(d["id"])
        return d["id"]

    def result(self, doc_id: str) -> dict:
        d = store.load(doc_id)
        segs = [(s["start"], s["end"], s["text"], s.get("speaker"), s.get("lang"),
                 [(w["w"], w["s"], w["e"], w.get("p")) for w in s.get("words") or []],
                 [(a["model"], a["text"]) for a in s.get("alts") or []]) for s in d["segments"]]
        hyps = store.path(doc_id) / "hypotheses.json"
        import json

        st = json.loads(hyps.read_text("utf-8"))
        return dict(status=d["status"], error=d.get("error"), segs=segs, hyps=st["hyps"], errors=st["errors"],
                    chunks=st["chunks"], members=d["meta"].get("members"))


# ---------------------------------------------------------------- tests

class BatchTest(Base):
    def sequential(self):
        ids = [self.new_doc(f) for f in self.files]
        for i in ids:
            pipeline.process(i)
        return [self.result(i) for i in ids]

    def test_batch_same_text_fewer_loads(self):
        seq = self.sequential()
        seq_loads = Fake.loads.copy()
        Fake.reset()
        os.environ["WHISPER_STUDIO_BATCH"] = "1"
        ids = [self.new_doc(f) for f in self.files]
        taken = pipeline.process_batch(ids, log=lambda m: None)
        self.assertEqual(taken, ids)
        bat = [self.result(i) for i in ids]
        for a, b in zip(seq, bat):
            self.assertEqual(a["status"], "done")
            self.assertEqual(a, b)
        self.assertTrue(any("de" in c["langs"] for c in seq[0]["chunks"]), "the test needs German chunks")
        for name, n in Fake.loads.items():
            if name == "large-v3":  # the detector: scan + its own run, per recording
                self.assertEqual(n, seq_loads[name])
            else:
                self.assertEqual(n, 1, name)
                self.assertGreaterEqual(seq_loads[name], 2, name)

    def test_batch_parallel_cpu_same_text(self):
        seq = self.sequential()
        Fake.reset()
        os.environ["WHISPER_STUDIO_PARALLEL_CPU"] = "1"
        ids = [self.new_doc(f) for f in self.files]
        pipeline.process_batch(ids)
        self.assertEqual(seq, [self.result(i) for i in ids])

    def test_member_failure_same_as_sequential(self):
        Fake.fail_load = {"gigaam-v3"}
        seq = self.sequential()
        self.assertIn("gigaam-v3", seq[0]["errors"])
        Fake.reset()
        Fake.fail_load = {"gigaam-v3"}
        ids = [self.new_doc(f) for f in self.files]
        pipeline.process_batch(ids)
        self.assertEqual(seq, [self.result(i) for i in ids])

    def test_deleted_recording_mid_batch(self):
        ids = [self.new_doc(f) for f in self.files]

        def on_load(name):
            if name == "qwen3-asr":
                store.delete(ids[1])

        Fake.on_load = on_load
        taken = pipeline.process_batch(ids)
        self.assertEqual(taken, ids)
        self.assertFalse(store.path(ids[1]).exists())
        for i in (ids[0], ids[2]):
            self.assertEqual(store.load(i)["status"], "done")

    def test_audio_budget_leaves_the_rest(self):
        os.environ["WHISPER_STUDIO_BATCH"] = "1"
        from whisper_studio import settings

        real = settings.speed
        settings.speed = lambda: dict(real(), batch_audio_gb=1e-6)
        try:
            ids = [self.new_doc(f) for f in self.files]
            taken = pipeline.process_batch(ids)
        finally:
            settings.speed = real
        self.assertEqual(taken, ids[:1])
        self.assertEqual(store.load(ids[0])["status"], "done")
        self.assertEqual(store.load(ids[1])["status"], "queued")

    def test_resume_after_crash(self):
        """A batch killed after some models: process() of each recording finishes the rest."""
        ids = [self.new_doc(f) for f in self.files]

        class Crash(BaseException):
            pass

        def on_load(name):
            if name == "gigaam-v3":
                raise Crash()

        Fake.on_load = on_load
        with self.assertRaises(Crash):
            pipeline.process_batch(ids)
        Fake.reset()
        for i in ids:
            pipeline.process(i)
        self.assertEqual(Fake.loads["ru-large-v3"], 0)  # cached in hypotheses.json
        got = [self.result(i) for i in ids]
        Fake.reset()
        self.assertEqual(self.sequential(), got)


class ParallelTest(Base):
    def test_parallel_cpu_same_text_and_overlaps(self):
        ensemble.lanes(ensemble.MEMBERS, True)  # imports onnxruntime (seconds) outside the timing
        Fake.cpu_delay = 0.1
        a = self.new_doc(self.files[2])
        t = time.monotonic()
        pipeline.process(a)
        t_seq = time.monotonic() - t
        Fake.reset()
        Fake.cpu_delay = 0.1
        os.environ["WHISPER_STUDIO_PARALLEL_CPU"] = "1"
        b = self.new_doc(self.files[2])
        t = time.monotonic()
        pipeline.process(b)
        t_par = time.monotonic() - t
        self.assertEqual(self.result(a), self.result(b))
        self.assertGreater(Fake.overlap[0], 0)
        self.assertLess(t_par, t_seq)

    def test_onnx_on_cuda_stays_in_gpu_lane(self):
        from whisper_studio.asr import onnx_models

        real = onnx_models.prefers_cuda
        onnx_models.prefers_cuda = lambda m: m["name"] == "gigaam-v3"
        try:
            mem = [m for m in ensemble.MEMBERS]
            gpu_lane, cpu_lane = ensemble.lanes(mem, True)
            names = lambda ks: [mem[k]["name"] for k in ks]  # noqa: E731
            self.assertIn("gigaam-v3", names(gpu_lane))
            self.assertEqual(names(cpu_lane), ["canary-v2", "t-one"])
            self.assertEqual(len(ensemble.lanes(mem, False)), 1)
        finally:
            onnx_models.prefers_cuda = real

    def test_batch_keeps_timing_detail_and_de_turbo_last(self):
        ids = [self.new_doc(f) for f in self.files[:2]]
        pipeline.process_batch(ids)
        d = store.load(ids[0])["meta"]
        self.assertIn("ru-large-v3", d["timing_detail"])
        self.assertGreaterEqual(d["timing_detail"]["ru-large-v3"]["chunks"], 1)
        order = list(ensemble.begin(np.zeros(SAMPLE_RATE, np.float32),
                                    pipeline.make_options(dict(mixed=True)))["mem"])
        self.assertEqual(order[-1]["name"], "de-turbo")

    def test_onnx_no_spin_flag(self):
        self.assertTrue(ensemble.cpu_member(dict(name="t-one"))["no_spin"])


class EarlyVoicesTest(Base):
    """Speaker voices computed ahead in a separate process give the same speakers.
    Uses the real diarization model (%WS_REAL_HOME%/models/diarization, read only; WS_REAL_HOME defaults to %LOCALAPPDATA%\\WhisperStudio) on 45 s
    of a real recording: set WS_TEST_SPEECH to an audio file, or the test is skipped."""

    def test_same_speakers(self):
        speech = os.environ.get("WS_TEST_SPEECH")
        real = REAL_HOME / "models/diarization"
        if not speech or not (real / "embedding.onnx").exists():
            self.skipTest("WS_TEST_SPEECH not set")
        from whisper_studio import audio as A

        x = A.load(Path(speech))[: 45 * SAMPLE_RATE]
        f = self.src / "speech.wav"
        with wave.open(str(f), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
        os.environ["WHISPER_STUDIO_DIAR_THREADS"] = "4"
        models.find = lambda key: real if key == "diarization" else _find(key)
        try:
            a = self.new_doc(f, diarize=True, num_speakers=None)
            pipeline.process(a)
            self.assertTrue(store.load(a)["meta"].get("diarized"))
            os.environ["WHISPER_STUDIO_EARLY_VOICES"] = "1"
            b = self.new_doc(f, diarize=True, num_speakers=None)
            pipeline.process(b)
            self.assertIn(b, pipeline.Voices.ran)
            ra, rb = self.result(a), self.result(b)
            self.assertTrue(ra["segs"] and all(s[3] for s in ra["segs"]))
            self.assertEqual(ra, rb)
            self.assertEqual(store.load(a)["speakers"], store.load(b)["speakers"])
        finally:
            models.find = _find
            os.environ.pop("WHISPER_STUDIO_DIAR_THREADS", None)
            f.unlink(missing_ok=True)


class PrefetchTest(unittest.TestCase):
    def setUp(self):
        prefetch.set_enabled(True)
        prefetch.stop()
        prefetch._done.clear()
        prefetch._pos.clear()
        self.dir = TMP_DIR / "prefetch-test"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.a, self.b = self.dir / "a", self.dir / "b"
        for f in (self.a, self.b):
            (f).mkdir(exist_ok=True)
            (f / "model.bin").write_bytes(os.urandom(20 << 20))

    def tearDown(self):
        prefetch.stop()
        prefetch.set_enabled(False)

    def test_reads_planned_files(self):
        b0 = prefetch.stats["bytes"]
        prefetch.ahead([self.a, self.b])
        self.assertTrue(prefetch.wait_idle(30))
        self.assertGreaterEqual(prefetch.stats["bytes"] - b0, 40 << 20)
        self.assertIn(self.a / "model.bin", prefetch._done)
        # already read: nothing planned again
        prefetch.ahead([self.a])
        self.assertTrue(prefetch.idle())

    def test_pauses_while_loading(self):
        with prefetch.loading(self.b):
            b0 = prefetch.stats["bytes"]
            prefetch.ahead([self.a])
            time.sleep(0.3)
            self.assertEqual(prefetch.stats["bytes"], b0)  # paused
        self.assertTrue(prefetch.wait_idle(30))
        self.assertGreater(prefetch.stats["bytes"], b0)

    def test_budget(self):
        from whisper_studio import settings

        real = settings.speed
        settings.speed = lambda: dict(real(), prefetch_gb=25 / 1024)  # 25 MB: only the first model
        try:
            prefetch._done.clear()
            prefetch._pos.clear()
            prefetch.ahead([self.a, self.b])
            with prefetch._cv:
                planned = [f for lst in prefetch._lanes.values() for f in lst]
            self.assertNotIn(self.b / "model.bin", planned)
        finally:
            settings.speed = real

    def test_follows_settings(self):
        from whisper_studio import settings

        prefetch.set_enabled(None)
        self.assertEqual(prefetch.enabled(), settings.speed()["prefetch"])


class WorkerBatchTest(unittest.TestCase):
    """worker.batch_for picks the process jobs right behind the first one, in queue order."""

    def setUp(self):
        self.made = []

    def tearDown(self):
        for f in self.made:
            f.unlink(missing_ok=True)
        for d in getattr(self, "docs", []):
            store.delete(d)
        os.environ.pop("WHISPER_STUDIO_BATCH", None)

    def job(self, kind, quality="max", started=False):
        d = store.create("t", "x.wav", dict(quality=quality))
        self.docs = getattr(self, "docs", []) + [d["id"]]
        f = jobs.QUEUE_DIR / f"job-{time.time_ns():020d}-{kind}-{d['id']}.json"
        jobs._write(f, dict(kind=kind, doc_id=d["id"], args={}, created=time.time(),
                            **({"started": 1.0, "pid": 1} if started else {})))
        self.made.append(f)
        time.sleep(0.002)
        return d["id"]

    def first(self):
        j = next(j for j in jobs.pending() if not j.get("started"))
        return j

    def test_off(self):
        self.job("process")
        self.job("process")
        self.assertEqual(len(worker.batch_for(self.first())), 1)

    def test_groups_until_other_kind(self):
        os.environ["WHISPER_STUDIO_BATCH"] = "1"
        a = self.job("process")
        b = self.job("process")
        self.job("rediarize")
        self.job("process")
        got = [j["doc_id"] for j in worker.batch_for(self.first())]
        self.assertEqual(got, [a, b])

    def test_fast_quality_not_batched(self):
        os.environ["WHISPER_STUDIO_BATCH"] = "1"
        a = self.job("process")
        self.job("process", quality="fast")
        self.job("process")
        self.assertEqual([j["doc_id"] for j in worker.batch_for(self.first())], [a])

    def test_run_batch_returns_the_rest(self):
        """Jobs left out of the batch (audio budget) go back to the queue unstarted."""
        os.environ["WHISPER_STUDIO_BATCH"] = "1"
        a, b, c = self.job("process"), self.job("process"), self.job("process")
        real = pipeline.process_batch
        pipeline.process_batch = lambda ids, log=None: ids[:2]
        try:
            worker.run_batch(worker.batch_for(self.first()))
        finally:
            pipeline.process_batch = real
        left = {j["doc_id"]: j for j in jobs.pending()}
        self.assertEqual(set(left), {c})
        self.assertFalse(left[c].get("started"))
        self.assertEqual(left[c].get("attempts"), 0)


if __name__ == "__main__":
    unittest.main()
