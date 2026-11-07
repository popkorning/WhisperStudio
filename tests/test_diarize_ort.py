"""diarize_ort (lab H12) computes what sherpa-onnx computes.

CPU only. The model tests use the real diarization model (%WS_REAL_HOME%/models/diarization,
read only; WS_REAL_HOME defaults to %LOCALAPPDATA%\\WhisperStudio) and are skipped
without it; the speaker test also needs WS_TEST_SPEECH (an audio
file with two people talking). Run from the repo root:

    set CUDA_VISIBLE_DEVICES=-1
    python -m unittest tests.test_diarize_ort -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("WHISPER_STUDIO_DIAR_THREADS", "4")
if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-diar-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_diarize_ort.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

import numpy as np  # noqa: E402

from whisper_studio import diarize as D  # noqa: E402
from whisper_studio import diarize_ort as O  # noqa: E402
from whisper_studio.config import SAMPLE_RATE  # noqa: E402

REAL = REAL_HOME / "models/diarization"


class Pure(unittest.TestCase):
    def test_fbank_frames(self):
        # snip_edges=false: one frame per 10 ms, rounded
        for n, want in ((16000, 100), (16079, 100), (16080, 101), (100, 1), (0, 0)):
            self.assertEqual(O.fbank(np.zeros(n, np.float32) + 0.01).shape, (want, 80))

    def test_cluster_threshold(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        emb = np.stack([a, a + 0.01, b, b + 0.02, a + 0.02]).astype(np.float32)
        lab = O._cluster(emb, 0.5, None)
        self.assertEqual(len(set(lab)), 2)
        self.assertEqual(lab[0], lab[1])
        self.assertEqual(lab[0], lab[4])
        self.assertNotEqual(lab[0], lab[2])
        self.assertEqual(len(set(O._cluster(emb, 1.5, None))), 1)

    def test_frames_to_turns_merges_short_gaps(self):
        class Seg:
            rf_shift, rf_size = 270, 991

        f = np.zeros((300, 2), np.int32)
        f[10:100, 0] = 1
        f[110:200, 0] = 1     # gap 10 frames = 0.17 s < 0.5 s: one turn
        f[250:260, 1] = 1     # 0.17 s <= 0.3 s: dropped
        t = O._frames_to_turns(f, Seg)
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0][2], 0)
        self.assertAlmostEqual(t[0][1] - t[0][0], 190 * 270 / 16000, places=3)


@unittest.skipUnless((REAL / "embedding.onnx").exists(), "no diarization model")
class SameAsSherpa(unittest.TestCase):
    def test_embedding(self):
        rng = np.random.default_rng(0)
        t = np.arange(int(3.3 * SAMPLE_RATE)) / SAMPLE_RATE
        x = (0.1 * np.sin(2 * np.pi * 180 * t * (1 + 0.2 * np.sin(3 * t))) + 0.01 * rng.standard_normal(len(t)))
        x = x.astype(np.float32)
        a = D._embed(D._embedder(REAL), x)
        b = O.Embedder(REAL, cuda=False)([O.fbank(x)])[0]
        self.assertGreater(float(a @ (b / np.linalg.norm(b))), 0.9999)

    def test_torch_graph(self):
        # onnx_torch executes embedding.onnx like onnxruntime (fp32, CPU here)
        import torch

        from whisper_studio.onnx_torch import TorchModel

        torch.set_num_threads(2)
        f = np.random.default_rng(1).standard_normal((2, 150, 80)).astype(np.float32)
        a = O.Embedder(REAL, cuda=False).sess.run(None, {"feats": f})[0]
        b = TorchModel(REAL / "embedding.onnx", "cpu", half=False)(f)
        self.assertLess(float(np.abs(a - b).max()), 1e-3)

    def test_speakers(self):
        speech = os.environ.get("WS_TEST_SPEECH")
        if not speech:
            self.skipTest("WS_TEST_SPEECH not set")
        from whisper_studio import audio as A

        x = A.load(Path(speech))[: 90 * SAMPLE_RATE]
        ref = D.diarize(x, REAL)
        new = O.diarize(x, REAL, embedder=O.Embedder(REAL, cuda=False))
        self.assertEqual(len({k for *_, k in ref}), len({k for *_, k in new}))
        # same speaker on (nearly) every 10 ms of speech, after mapping the labels
        from tools.diarize_compare import agreement, frames

        self.assertGreater(agreement(frames(ref, 90), frames(new, 90)), 0.98)

    def test_setting_switches_backend(self):
        speech = os.environ.get("WS_TEST_SPEECH")
        if not speech:
            self.skipTest("WS_TEST_SPEECH not set")
        from whisper_studio import audio as A

        x = A.load(Path(speech))[: 30 * SAMPLE_RATE]
        segs = [dict(start=0.0, end=30.0, text="a b", words=[dict(w="a", s=1.0, e=1.5), dict(w="b", s=20.0, e=20.4)])]
        os.environ.update(WHISPER_STUDIO_DIARIZE_GPU="1", WHISPER_STUDIO_DIAR_DEVICE="cpu")
        try:
            out, info = D.run([dict(s) for s in segs], x, REAL, None)
        finally:
            os.environ.pop("WHISPER_STUDIO_DIARIZE_GPU")
            os.environ.pop("WHISPER_STUDIO_DIAR_DEVICE")
        self.assertEqual(info["voices"]["backend"], "ort")
        self.assertTrue(all(s.get("speaker") for s in out))


if __name__ == "__main__":
    unittest.main()
