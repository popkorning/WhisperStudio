"""Speaker diarization on onnxruntime: the same pipeline as sherpa-onnx's
OfflineSpeakerDiarization (pyannote segmentation 3.0 -> WeSpeaker ResNet293 embeddings ->
complete-linkage clustering), but the embeddings run in batches on the GPU (settings
speed.diarize_gpu, lab H12). sherpa-onnx in our build has no CUDA and holds the GIL for the
whole computation: 40-50% of a "max" job (a 30-min recording ~20 min).

Ported from sherpa-onnx csrc/offline-speaker-diarization-pyannote-impl.h (v2.0.1):
same windows (10 s, shift 1 s), same powerset decoding, same (chunk, local speaker) sample
ranges, same kaldi fbank (80 bins, 25/10 ms, povey, no dither, int16 sample scale, no
snip_edges), same clustering and the same frame-to-segment post-processing. Only the floats
differ a little (another onnxruntime build, GPU kernels), so the clusters are the same in the
vast majority of cases, not bit for bit.

The embedding model runs with torch on the GPU (onnx_torch, fp32) when torch has CUDA, else
with onnxruntime-gpu, else with onnxruntime on the CPU (as fast as sherpa there).

Environment:
    WHISPER_STUDIO_DIAR_DEVICE   auto (default) | cpu | cuda
    WHISPER_STUDIO_DIAR_ENGINE   auto (default) | torch | ort
    WHISPER_STUDIO_DIAR_FP16     0 (default) | 1: torch convolutions in fp16 (1.5x faster,
                                 speakers drift: lab H12)
    WHISPER_STUDIO_DIAR_BATCH    equal-length inputs per GPU call (fp16 4, else 1)
"""
from __future__ import annotations

import gc
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE

VRAM_GB = 1.5        # torch fp32, one 10 s input: ~0.3 GB + cache (TORCH_CACHE_GB) + CUDA context
ARENA_GB = 2.0       # onnxruntime-gpu memory pool
TORCH_CACHE_GB = 0.8  # torch: cached blocks above this go back to the driver
MIN_DURATION_ON, MIN_DURATION_OFF = 0.3, 0.5
WINDOW_SHIFT_RATIO = 0.1

stats: dict = {}     # timing of the last run (benchmarks)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def _threads() -> int:
    env = os.environ.get("WHISPER_STUDIO_DIAR_THREADS")
    return int(env) if env else max(1, (os.cpu_count() or 4) - 1)


# ---------------------------------------------------------------- kaldi fbank (as knf in sherpa)

_FRAME_LEN, _FRAME_SHIFT, _NFFT, _NMEL = 400, 160, 512, 80


def _mel_banks() -> np.ndarray:
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    low, high = 20.0, SAMPLE_RATE / 2 - 400.0
    mlo, mhi = mel(low), mel(high)
    delta = (mhi - mlo) / (_NMEL + 1)
    m = mel(np.arange(_NFFT // 2) * (SAMPLE_RATE / _NFFT))
    left = mlo + np.arange(_NMEL)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (m - left) / (center - left)
    down = (right - m) / (right - center)
    w = np.where((m > left) & (m < right), np.where(m <= center, up, down), 0.0)
    return np.pad(w, ((0, 0), (0, 1))).T.astype(np.float32)  # [257, 80], Nyquist bin unused


_MEL = _mel_banks()
_WINDOW = ((0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_FRAME_LEN) / (_FRAME_LEN - 1))) ** 0.85).astype(np.float32)
_EPS = np.finfo(np.float32).eps


def _num_frames(n: int) -> int:
    return (n + _FRAME_SHIFT // 2) // _FRAME_SHIFT if n > 0 else 0


def fbank(x: np.ndarray) -> np.ndarray:
    """[T, 80] log-mel features of float audio in [-1, 1] (scaled to int16 like sherpa's
    normalize_samples=0), snip_edges=false: frames centred on 10 ms steps, edges reflected."""
    n = len(x)
    nf = _num_frames(n)
    if nf <= 0:
        return np.zeros((0, _NMEL), np.float32)
    idx = (np.arange(nf)[:, None] * _FRAME_SHIFT + (_FRAME_SHIFT // 2 - _FRAME_LEN // 2)
           + np.arange(_FRAME_LEN)[None, :])
    idx = np.where(idx < 0, -idx - 1, idx)
    idx = np.where(idx >= n, 2 * n - 1 - idx, idx)
    idx = np.clip(idx, 0, n - 1)  # inputs shorter than half a frame
    fr = x.astype(np.float32)[idx] * np.float32(32768.0)
    fr -= fr.mean(axis=1, keepdims=True)
    fr[:, 1:] -= np.float32(0.97) * fr[:, :-1]
    fr[:, 0] *= np.float32(1 - 0.97)
    fr *= _WINDOW
    spec = np.fft.rfft(fr, n=_NFFT, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    return np.log(np.maximum(power @ _MEL, _EPS))


# ---------------------------------------------------------------- the embedding model

def _providers_cuda():
    return [("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": int(ARENA_GB * 1024 ** 3),
        # every input has another length: an exhaustive cuDNN search would run for each shape
        "cudnn_conv_algo_search": "HEURISTIC",
    }), "CPUExecutionProvider"]


def _torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _ort_cuda() -> bool:
    """onnxruntime-gpu present; its cuBLAS/cuDNN DLLs are taken from torch's lib (onnx_models)."""
    from .asr import onnx_models

    return onnx_models.cuda_available()


def want_cuda(note=None) -> bool:
    mode = _env("WHISPER_STUDIO_DIAR_DEVICE", "auto")
    if mode == "cpu" or not (_torch_cuda() or _ort_cuda()):
        return False
    from .asr import gpu

    # without the GPU it still works (on the CPU): wait for video memory only a little
    return gpu.wait(VRAM_GB, note, limit=None if mode == "cuda" else 60.0, torch_cache=True)


class Embedder:
    """embedding.onnx (feats [B, T, 80] -> embs [B, 256]) on the GPU when possible:
    torch (fp16 convolutions, onnx_torch) in the main venv, onnxruntime-gpu where installed
    (WHISPER_STUDIO_DIAR_ENGINE = auto | torch | ort), else onnxruntime on the CPU."""

    def __init__(self, model_dir: Path, cuda: bool | None = None, note=None):
        import onnxruntime as rt

        path = Path(model_dir) / "embedding.onnx"
        self.device, self.engine = "cpu", "ort"
        self.sess = self.torch = None
        self.timing = {"fbank": 0.0, "model": 0.0}
        so = rt.SessionOptions()
        so.intra_op_num_threads = _threads()
        engine = _env("WHISPER_STUDIO_DIAR_ENGINE", "auto")
        if want_cuda(note) if cuda is None else cuda:
            if engine in ("auto", "torch") and _torch_cuda():
                try:
                    from .onnx_torch import TorchModel

                    # fp16 is 1.5x faster but moves the clustering: 2b3c4d 87% same words (fp32 100%)
                    self.torch = TorchModel(path, "cuda", half=_env("WHISPER_STUDIO_DIAR_FP16", "0") == "1",
                                            channels_last=True)
                    self.device, self.engine = "cuda", "torch-fp16" if self.torch.half else "torch"
                    import torch

                    self.base = torch.cuda.memory_reserved()  # other models of the job
                except Exception as e:  # noqa: BLE001 - an operator we don't know: onnxruntime
                    self.torch = None
                    if note:
                        note(f"спикеры: torch не подошёл ({type(e).__name__}: {e})")
            if self.torch is None and engine in ("auto", "ort") and _ort_cuda():
                try:
                    self.sess = rt.InferenceSession(str(path), sess_options=so, providers=_providers_cuda())
                    if self.sess.get_providers()[0] == "CUDAExecutionProvider":
                        self.device = "cuda"
                except Exception as e:  # noqa: BLE001 - CUDA libraries missing, no device: the CPU will do
                    if note:
                        note(f"спикеры: видеокарта недоступна ({type(e).__name__}), считаю на процессоре")
                    self.sess = None
        if self.torch is None and self.sess is None:
            self.sess = rt.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        # equal-length inputs per call: only fp16 gains from it (fp32 is as fast one by one, and
        # onnxruntime's arena fragments with many shapes)
        default = "4" if self.torch is not None and self.torch.half else "1"
        self.batch = max(1, int(_env("WHISPER_STUDIO_DIAR_BATCH", default) or 1)) if self.device == "cuda" else 1

    def _forward(self, x: np.ndarray) -> np.ndarray:
        if self.torch is not None:
            return self.torch(x)
        return self.sess.run(None, {"feats": x})[0]

    def __call__(self, feats: list[np.ndarray], progress: Callable[[int, int], None] | None = None
                 ) -> list[np.ndarray]:
        return self.run([len(f) for f in feats], lambda i: feats[i], progress, waves=False)

    def run(self, lengths: list[int], make: Callable[[int], np.ndarray],
            progress: Callable[[int, int], None] | None = None, waves: bool = True) -> list[np.ndarray]:
        """One embedding per input i of lengths[i] frames; make(i) gives its audio (waves) or
        its features. Inputs of one length go into one call: padding would change the
        statistics pooling, so a batch holds equal lengths only. Features are made batch by
        batch: all of them at once would take ~1 GB for a 30-min recording."""
        from .asr import gpu

        out: list[np.ndarray | None] = [None] * len(lengths)
        order = sorted(range(len(lengths)), key=lambda i: lengths[i])
        done = 0
        i = 0
        while i < len(order):
            t = lengths[order[i]]
            j = i
            while j < len(order) and j - i < self.batch and lengths[order[j]] == t:
                j += 1
            ids = order[i:j]
            t0 = time.monotonic()
            x = np.stack([fbank(make(k)) if waves else make(k) for k in ids]).astype(np.float32)
            t1 = time.monotonic()
            try:
                y = self._forward(x)
            except Exception as e:  # noqa: BLE001
                if not gpu.is_oom(e) or len(ids) == 1:
                    raise
                self._free()
                self.batch = 1  # out of video memory: one at a time from now on
                y = np.concatenate([self._forward(x[r:r + 1]) for r in range(len(ids))])
            self.timing["fbank"] += t1 - t0
            self.timing["model"] += time.monotonic() - t1
            self._trim()
            for k, v in zip(ids, y):
                out[k] = v
            done += len(ids)
            i = j
            if progress:
                progress(done, len(lengths))
        return out  # type: ignore[return-value]

    def _trim(self):
        """Every input length leaves differently sized blocks in torch's cache: without this it
        grows to 4-5 GB over a long recording, taking the memory of the other programs."""
        if self.torch is not None:
            import torch

            if torch.cuda.memory_reserved() - self.base > TORCH_CACHE_GB * 1024 ** 3:
                torch.cuda.empty_cache()
                self.base = min(self.base, torch.cuda.memory_reserved())

    def _free(self):
        gc.collect()
        if self.torch is not None:
            import torch

            torch.cuda.empty_cache()

    def close(self):
        had_torch = self.torch is not None
        self.sess = self.torch = None
        gc.collect()
        if had_torch:  # give the video memory back to the next models of the job
            import torch

            torch.cuda.empty_cache()


class Segmenter:
    def __init__(self, model_dir: Path):
        import onnxruntime as rt

        so = rt.SessionOptions()
        so.intra_op_num_threads = _threads()
        self.sess = rt.InferenceSession(str(Path(model_dir) / "segmentation.onnx"), sess_options=so,
                                        providers=["CPUExecutionProvider"])
        m = self.sess.get_modelmeta().custom_metadata_map
        self.window = int(m["window_size"])
        self.shift = max(1, min(self.window, int(WINDOW_SHIFT_RATIO * self.window)))
        self.rf_size = int(m["receptive_field_size"])
        self.rf_shift = int(m["receptive_field_shift"])
        self.num_speakers = int(m["num_speakers"])
        self.mapping = _powerset(int(m["num_classes"]), self.num_speakers, int(m["powerset_max_classes"]))

    def chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        """Powerset log-probs [frames, classes] per 10 s window (shift 1 s, the last padded)."""
        n, w, s = len(audio), self.window, self.shift
        starts = [0] if n <= w else list(range(0, (n - w) // s * s + 1, s))
        if n > w and (n - w) % s > 0:
            starts.append(starts[-1] + s)
        out = []
        for b in range(0, len(starts), 16):
            x = np.zeros((len(starts[b:b + 16]), 1, w), np.float32)
            for r, p in enumerate(starts[b:b + 16]):
                seg = audio[p:p + w]
                x[r, 0, :len(seg)] = seg
            out.extend(self.sess.run(None, {"x": x})[0])
        return out


def _powerset(num_classes: int, num_speakers: int, max_classes: int) -> np.ndarray:
    m = np.zeros((num_classes, num_speakers), np.int32)
    k = 1
    for i in range(1, max_classes + 1):
        if i == 1:
            for j in range(num_speakers):
                m[k, j] = 1
                k += 1
        elif i == 2:
            for j in range(num_speakers):
                for q in range(j + 1, num_speakers):
                    m[k, j] = m[k, q] = 1
                    k += 1
        else:
            raise ValueError(f"powerset_max_classes = {i} is not supported")
    return m


# ---------------------------------------------------------------- the sherpa pipeline

def _f32(v) -> np.float32:
    return np.float32(v)


def _chunk_start_frame(i: int, shift: int, rf_shift: int) -> int:
    # std::lround(static_cast<float>(i) * window_shift / receptive_field_shift)
    v = _f32(_f32(i) * _f32(shift)) / _f32(rf_shift)
    return int(math.floor(float(v) + 0.5))


def _sample_ranges(label: np.ndarray, chunk: int, seg: Segmenter) -> list[tuple[int, int]] | None:
    """Sample ranges of one local speaker in one window (frames where it talks alone);
    None for fewer than 10 such frames."""
    if label.sum() < 10:
        return None
    nf = len(label)
    off = chunk * seg.shift

    def to_sample(k):
        return int(_f32(_f32(_f32(k) / _f32(nf)) * _f32(seg.window)) + _f32(off))

    out = []
    d = np.diff(np.concatenate([[0], label.astype(np.int8), [0]]))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        out.append((to_sample(a), to_sample(b if b < nf else nf - 1)))
    return out


def _cluster(emb: np.ndarray, threshold: float, num_clusters: int | None) -> np.ndarray:
    """FastClustering: complete linkage on cosine dissimilarity; merges below the threshold."""
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(emb)
    if n == 1:
        return np.zeros(1, np.int64)
    m = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = (m @ m.T).astype(np.float64)
    iu = np.triu_indices(n, 1)
    dist = np.maximum(1.0 - sim[iu], 0.0)
    z = linkage(dist, "complete")
    if num_clusters and num_clusters > 0:
        k = num_clusters
    else:
        k = n - int(np.sum(z[:, 2] < threshold))  # fastcluster cutree_cdist: heights < threshold
    return fcluster(z, max(1, k), "maxclust").astype(np.int64) - 1


def _frames_to_turns(final: np.ndarray, seg: Segmenter) -> list[tuple[float, float, int]]:
    """ComputeResult + MergeSegments: frame labels [frames, speakers] -> (start, end, speaker)."""
    scale = _f32(_f32(seg.rf_shift) / _f32(SAMPLE_RATE))
    offset = _f32(0.5 * seg.rf_size / SAMPLE_RATE)
    out = []
    nf = final.shape[0]
    for spk in range(final.shape[1]):
        col = final[:, spk] > 0
        segs = []
        d = np.diff(np.concatenate([[0], col.astype(np.int8), [0]]))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            b = min(b, nf - 1)
            segs.append([float(_f32(a) * scale + offset), float(_f32(b) * scale + offset)])
        merged = []
        for s in segs:  # sorted, never overlapping: merging neighbours left to right is the same
            if merged and merged[-1][1] < s[0] and merged[-1][1] + MIN_DURATION_OFF >= s[0]:
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        out.extend((s, e, spk) for s, e in merged if e - s > MIN_DURATION_ON)
    out.sort(key=lambda t: t[0])
    return out


def diarize(audio: np.ndarray, model_dir: Path, num_speakers: int | None = None,
            threshold: float = 0.5, progress: Callable[[float, str], None] | None = None,
            embedder: Embedder | None = None, extra: dict | None = None,
            constrained: bool | None = None) -> list[tuple[float, float, int]]:
    """extra: gets the per-frame activity of every raw cluster (diarize.Activity);
    constrained: one window's local speakers go to different clusters (default CONSTRAIN)."""
    t0 = time.monotonic()
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    if n <= 0:
        return []
    seg = Segmenter(model_dir)
    logp = seg.chunks(audio)
    stats["segmentation"] = round(time.monotonic() - t0, 2)
    stats["chunks"] = len(logp)
    if len(logp) == 1:
        if progress:
            progress(1.0, "Определяю спикеров")
        return _single(logp[0], n, seg, extra)
    pairs, ranges, lengths = _pairs(logp, n, seg)
    stats["embeddings"] = len(pairs)
    t1 = time.monotonic()

    def wave(i):
        return np.concatenate([audio[a:b] for a, b in ranges[i]])

    own = embedder is None
    ex = embedder or Embedder(model_dir)
    stats["device"] = ex.device
    stats["engine"] = ex.engine
    try:
        cb = (lambda d, t: progress(d / t, "Определяю спикеров")) if progress else None
        embs = ex.run(lengths, wave, cb)
    finally:
        if own:
            ex.close()
    stats["embed"] = round(time.monotonic() - t1, 2)
    stats["embed_fbank"] = round(ex.timing["fbank"], 2)
    return cluster_turns(logp, n, seg, pairs, embs, threshold, num_speakers, extra, constrained)


def _labels(logp: list[np.ndarray], seg) -> list[np.ndarray]:
    return [seg.mapping[c.argmax(axis=1)] for c in logp]


def _last_chunk(n: int, seg) -> bool:
    return math.fmod(n - seg.window, seg.shift) > 0


def _single(logp: np.ndarray, n: int, seg, extra: dict | None) -> list[tuple[float, float, int]]:
    """One window: its local speakers are the speakers."""
    final = seg.mapping[logp.argmax(axis=1)]
    keep = len(final)
    if _last_chunk(n, seg):
        keep = min(n // seg.rf_shift, len(final))
    if extra is not None:
        _put_activity(extra, _local_activity(logp, seg)[:keep], seg)
    return _frames_to_turns(final[:keep], seg)


def _pairs(logp: list[np.ndarray], n: int, seg):
    """(window, local speaker) with >= 10 frames alone, their sample ranges and fbank lengths."""
    pairs, ranges, lengths = [], [], []
    for ci, lab in enumerate(_labels(logp, seg)):
        alone = np.where(lab.sum(axis=1, keepdims=True) < 2, lab, 0)
        for sp in range(seg.num_speakers):
            rng = _sample_ranges(alone[:, sp], ci, seg)
            if rng is None:
                continue
            rng = [(a, min(b, n)) for a, b in rng if min(b, n) - a > 0]
            nf = _num_frames(sum(b - a for a, b in rng))
            if nf <= 0:
                continue
            pairs.append((ci, sp))
            ranges.append(rng)
            lengths.append(nf)
    return pairs, ranges, lengths


# pyannote's assignment step (clustering.py, constrained_argmax): two speakers that one window
# tells apart are different people. sherpa's FastClustering has no such constraint, and on
# phone calls (voices 0.85-0.9 alike) complete linkage sometimes puts both local speakers of a
# window into one cluster. Only such windows are re-solved, the rest stays exactly as sherpa.
# Off (Опыт 11): on our calls the two local speakers that complete linkage keeps together are
# one voice split by the segmentation (cosine median 0.82-0.87, pairs it separates 0.67-0.72);
# forced apart, a1b2c3 did not change and d4e5f6 got some words given to the other.
CONSTRAIN = os.environ.get("WHISPER_STUDIO_DIAR_CONSTRAIN", "0") == "1"


def constrain(pairs: list[tuple[int, int]], emb: np.ndarray, cl: np.ndarray) -> tuple[np.ndarray, int]:
    """Clusters per pair with no two pairs of one window in one cluster: such a window's local
    speakers get the assignment to the cluster centroids with the largest total cosine
    (Hungarian); a window with more local speakers than clusters keeps its labels.
    Returns (labels, windows re-solved)."""
    from scipy.optimize import linear_sum_assignment

    cl = np.asarray(cl).copy()
    ncl = int(cl.max()) + 1 if len(cl) else 0
    if ncl < 2:
        return cl, 0
    m = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cents = np.stack([m[cl == c].mean(axis=0) for c in range(ncl)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    by: dict[int, list[int]] = {}
    for i, (ci, _sp) in enumerate(pairs):
        by.setdefault(ci, []).append(i)
    fixed = 0
    for idx in by.values():
        if len(idx) < 2 or len(set(cl[idx].tolist())) == len(idx) or len(idx) > ncl:
            continue
        r, c = linear_sum_assignment(-(m[idx] @ cents.T))
        cl[np.array(idx)[r]] = c
        fixed += 1
    return cl, fixed


def cluster_turns(logp: list[np.ndarray], n: int, seg, pairs: list[tuple[int, int]],
                  embs: list[np.ndarray], threshold: float = 0.5, num_speakers: int | None = None,
                  extra: dict | None = None, constrained: bool | None = None
                  ) -> list[tuple[float, float, int]]:
    """Clustering and frames -> turns, from the segmentation and the pair embeddings."""
    t2 = time.monotonic()
    labels = _labels(logp, seg)
    nch = len(labels)
    nfr = (seg.window + (nch - 1) * seg.shift) // seg.rf_shift + 1
    starts = [_chunk_start_frame(i, seg.shift, seg.rf_shift) for i in range(nch)]
    count = np.zeros(nfr, np.float32)
    weight = np.zeros(nfr, np.float32)
    for st, lab in zip(starts, labels):
        count[st:st + len(lab)] += lab.sum(axis=1)
        weight[st:st + len(lab)] += 1
    per_frame = (count / (weight + _f32(1e-12)) + _f32(0.5)).astype(np.int32)
    if per_frame.max() == 0:
        return []
    ok = [i for i, e in enumerate(embs) if not np.isnan(e).any()]
    if not ok:
        return []
    pairs = [pairs[i] for i in ok]
    emb = np.stack([embs[i] for i in ok]).astype(np.float32)
    cl = _cluster(emb, threshold, num_speakers)
    if CONSTRAIN if constrained is None else constrained:
        cl, stats["constrained_windows"] = constrain(pairs, emb, cl)
    to_cluster = {p: int(c) for p, c in zip(pairs, cl)}
    ncl = int(cl.max()) + 1

    spk_count = np.zeros((nfr, ncl), np.int32)
    act = np.zeros((nfr, ncl), np.float32) if extra is not None else None
    for ci, (st, lab) in enumerate(zip(starts, labels)):
        rel = np.zeros((len(lab), ncl), np.int32)
        loc = _local_activity(logp[ci], seg) if act is not None else None
        for sp in range(lab.shape[1]):
            c = to_cluster.get((ci, sp))
            if c is not None:
                rel[lab[:, sp] == 1, c] = 1
                if act is not None:
                    act[st:st + len(lab), c] += loc[:, sp]
        spk_count[st:st + len(lab)] += rel
    keep = len(spk_count)
    if _last_chunk(n, seg):
        keep = min(n // seg.rf_shift, len(spk_count) - 1) + 1
    spk_count = spk_count[:keep]
    if act is not None:
        _put_activity(extra, act[:keep] / (weight[:keep, None] + _f32(1e-12)), seg)
    # FinalizeLabels: the k most counted clusters of a frame, k = speakers in that frame
    order = np.argsort(-spk_count, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(ncl)[None, :].repeat(len(order), 0), axis=1)
    final = (rank < per_frame[:len(spk_count), None]).astype(np.int32)
    stats["clustering"] = round(time.monotonic() - t2, 2)
    stats["raw_clusters"] = ncl
    return _frames_to_turns(final, seg)


def _local_activity(logp: np.ndarray, seg) -> np.ndarray:
    """[frames, local speakers]: probability that each local speaker talks (powerset summed)."""
    p = np.exp(logp.astype(np.float32))
    p /= p.sum(axis=1, keepdims=True)
    return p @ seg.mapping.astype(np.float32)


def _put_activity(extra: dict, act: np.ndarray, seg) -> None:
    """diarize.Activity's arrays: frame f is at act_t0 + f * act_dt seconds (as _frames_to_turns)."""
    extra["act"] = np.rint(np.clip(act, 0.0, 1.0) * 255).astype(np.uint8)
    extra["act_t0"] = np.float64(0.5 * seg.rf_size / SAMPLE_RATE)
    extra["act_dt"] = np.float64(seg.rf_shift / SAMPLE_RATE)


def turn_embeddings(audio: np.ndarray, turns: list[tuple[float, float, int]], model_dir: Path,
                    progress: Callable[[float, str], None] | None = None,
                    embedder: Embedder | None = None) -> dict[int, np.ndarray]:
    """diarize.turn_embeddings on the same embedder: the first 10 s of every turn >= 0.6 s, and
    the short turns of a cluster without such a turn together (key -1 - cluster)."""
    from .diarize import MIN_TURN, short_clusters

    idx, cuts = [], []
    for i, (s, e, _) in enumerate(turns):
        if e - s >= MIN_TURN:
            a, b = int(s * SAMPLE_RATE), int(min(e, s + 10) * SAMPLE_RATE)
            b = min(b, len(audio))
            if _num_frames(b - a) > 0:
                idx.append(i)
                cuts.append([(a, b)])
    for k, c in short_clusters(turns, len(audio)).items():
        idx.append(-1 - k)
        cuts.append(c)
    lengths = [_num_frames(sum(b - a for a, b in c)) for c in cuts]
    own = embedder is None
    ex = embedder or Embedder(model_dir)
    try:
        cb = (lambda d, t: progress(d / t, "Сравниваю голоса")) if progress else None
        embs = ex.run(lengths, lambda k: np.concatenate([audio[a:b] for a, b in cuts[k]]), cb)
    finally:
        if own:
            ex.close()
    return {i: (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32) for i, v in zip(idx, embs)}


def voices(audio: np.ndarray, model_dir: Path, progress: Callable[[float, str], None] | None = None,
           note=None, extra: dict | None = None):
    """diarize.voices on onnxruntime: raw turns and one voice embedding per turn; extra gets
    the per-frame activity (diarize.Activity)."""
    def sub(lo, hi):
        return (lambda f, _m: progress(lo + (hi - lo) * f, "Определяю спикеров")) if progress else None

    t0 = time.monotonic()
    stats.clear()
    ex = Embedder(model_dir, note=note)
    stats["load"] = round(time.monotonic() - t0, 2)
    try:
        turns = diarize(audio, model_dir, None, progress=sub(0.0, 0.9), embedder=ex, extra=extra)
        t1 = time.monotonic()
        temb = turn_embeddings(audio, turns, model_dir, progress=sub(0.9, 1.0), embedder=ex)
        stats["turns"] = round(time.monotonic() - t1, 2)
    finally:
        ex.close()
    stats["total"] = round(time.monotonic() - t0, 2)
    return turns, temb
