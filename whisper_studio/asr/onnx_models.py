"""GigaAM v3, T-one, Parakeet TDT v3 and Canary 1B v2 through onnx-asr (onnxruntime).

On the GPU when onnxruntime-gpu is installed (CUDAExecutionProvider) and video memory is free,
otherwise on the CPU. With onnxruntime (CPU only) everything stays as it was. Environment:

    WHISPER_STUDIO_ONNX_DEVICE   auto (default) | cpu | cuda
    WHISPER_STUDIO_ONNX_BATCH    chunks per call for models padding doesn't change (default 8)
    WHISPER_STUDIO_ONNX_BATCH_ANY=1  batch every model (experiments: GigaAM's text changes)
    WHISPER_STUDIO_ONNX_THREADS  CPU threads (default: all logical cores but one)
    WHISPER_STUDIO_ONNX_SPLIT    1 (default): a transducer's small decoder and joint stay on
                                 the CPU (called once per frame, the GPU only adds copies)

In "auto" int8 models (Canary) stay on the CPU: the CUDA provider runs integer MatMul nodes on
the CPU anyway and copies tensors back and forth.
"""
from __future__ import annotations

import gc
import math
import os

from .. import models
from ..config import SAMPLE_RATE
from . import gpu

# Canary has no automatic language detection: it is told the chunk language
CANARY_LANGS = {"bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv",
                "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk"}

# video memory per model: fp32 weights + activations of a batch of 25 s chunks
VRAM_GB = {"gigaam-v3": 1.8, "t-one": 0.8, "canary-v2": 2.0, "parakeet-v3": 2.4}
ARENA_GB = 2.5          # cap of onnxruntime's GPU memory pool per session
# models whose text doesn't depend on the padding of a batch: T-one is a streaming (causal)
# model, zeros after the end change nothing (1.4x faster on the CPU).
# GigaAM's encoder looks at the whole chunk: batched, some words came out different.
PAD_SAFE = {"t-one"}
BATCH_SECONDS = 160.0   # audio per call: padding and attention memory grow with it


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def cuda_available() -> bool:
    """onnxruntime-gpu is installed and its CUDA libraries load (cuDNN 9 and cuBLAS 12 come
    with torch's CUDA wheel; engine registers torch\\lib as a DLL folder)."""
    try:
        import onnxruntime as rt
    except Exception:  # noqa: BLE001
        return False
    if "CUDAExecutionProvider" not in rt.get_available_providers():
        return False
    _preload(rt)
    return True


_preloaded = {"done": False}


def _preload(rt) -> None:
    if _preloaded["done"]:
        return
    _preloaded["done"] = True
    from .. import engine  # noqa: F401 - registers torch\lib (cuDNN, cuBLAS) as a DLL folder

    preload = getattr(rt, "preload_dlls", None)
    if preload is None:
        return
    try:
        # takes torch\lib when torch is built for the same CUDA major as onnxruntime-gpu
        # (torch cu128 + the CUDA 12 build of onnxruntime-gpu: one set of CUDA DLLs in the process)
        preload()
    except Exception:  # noqa: BLE001 - the session will say what is missing
        pass


def _cuda_options() -> dict:
    return {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",  # grow by what is asked, not in doubling steps
        "gpu_mem_limit": int(ARENA_GB * 1024 ** 3),
        # every chunk has another length: an exhaustive cuDNN search would run for each new shape
        "cudnn_conv_algo_search": "HEURISTIC",
    }


def prefers_cuda(member: dict) -> bool:
    """This member will try the GPU (before any wait for video memory). ensemble.lanes keeps such
    a member with the GPU members, out of the parallel CPU thread (parallel_cpu)."""
    mode = _env("WHISPER_STUDIO_ONNX_DEVICE", "auto")
    if mode == "cpu" or not cuda_available():
        return False
    return not (mode == "auto" and models.MODELS.get(member["model"], {}).get("quant"))


class OnnxBackend:
    batch_seconds = BATCH_SECONDS

    def __init__(self, member: dict):
        self.member = member
        self.key = member["model"]
        self.spec = models.MODELS[self.key]
        self.model = None
        self.device = None
        self.batch = 1
        if self.key in PAD_SAFE or _env("WHISPER_STUDIO_ONNX_BATCH_ANY", "0") == "1":
            self.batch = max(1, int(_env("WHISPER_STUDIO_ONNX_BATCH", "8") or 1))

    @property
    def aed(self) -> bool:
        return self.spec["onnx"] == "nemo-conformer-aed"

    def _want_cuda(self, note) -> bool:
        if not prefers_cuda(self.member):
            return False
        mode = _env("WHISPER_STUDIO_ONNX_DEVICE", "auto")
        need = VRAM_GB.get(self.key, 2.0)
        # the ensemble works fine with these on the CPU: wait for the GPU only a little
        return gpu.wait(need, note, limit=None if mode == "cuda" else 60.0, torch_cache=True)

    def load(self, note=None):
        import onnx_asr
        import onnxruntime as rt

        path = models.find(self.key)
        if path is None:
            raise RuntimeError(f"Модель {self.key} не установлена")
        so = rt.SessionOptions()
        threads = int(_env("WHISPER_STUDIO_ONNX_THREADS", "0") or 0) or max(1, (os.cpu_count() or 4) - 1)
        so.intra_op_num_threads = threads
        if self.member.get("no_spin"):  # CPU lane next to the GPU members (ensemble.lanes): no busy-wait
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        if self._want_cuda(note):
            try:
                # features and resampling stay on the CPU, exactly as in the CPU path
                asr = onnx_asr.load_model(self.spec["onnx"], str(path), quantization=self.spec.get("quant"),
                                          sess_options=so,
                                          providers=[("CUDAExecutionProvider", _cuda_options()),
                                                     "CPUExecutionProvider"],
                                          preprocessor_config={"use_numpy_preprocessors": True},
                                          resampler_config={"sess_options": so,
                                                            "providers": ["CPUExecutionProvider"]})
                # onnxruntime itself falls back to the CPU when the CUDA provider can't start
                self.device = _device(asr)
                if self.device == "cuda" and _env("WHISPER_STUDIO_ONNX_SPLIT", "1") != "0":
                    _decoder_to_cpu(asr, path, threads)
                self.model = asr.with_timestamps()
                return
            except Exception as e:  # noqa: BLE001 - CUDA libraries missing, no device: the CPU will do
                if note:
                    note(f"{self.key}: видеокарта недоступна ({type(e).__name__}), считаю на процессоре")
            gc.collect()
        asr = onnx_asr.load_model(self.spec["onnx"], str(path), quantization=self.spec.get("quant"),
                                  sess_options=so, providers=["CPUExecutionProvider"])
        self.model, self.device = asr.with_timestamps(), "cpu"

    def unload(self):
        self.model = None
        gc.collect()

    def _kw(self, lang: str | None) -> dict:
        if self.aed:
            return dict(language=lang if lang in CANARY_LANGS else "ru", pnc=True)
        return {}

    def _result(self, r, n: int, kw: dict) -> dict:
        text = (r.text or "").strip()
        words = None
        if r.timestamps and r.tokens and not self.aed:
            words = _words(r.tokens, r.timestamps, r.logprobs, n / SAMPLE_RATE)
        return dict(text=text, words=words, language=kw.get("language"))

    def recognize(self, x, lang: str | None, prompt: str | None) -> dict:
        kw = self._kw(lang)
        return self._result(self.model.recognize(x, sample_rate=SAMPLE_RATE, **kw), len(x), kw)

    def recognize_batch(self, xs: list, lang: str | None, prompt: str | None) -> list[dict]:
        """Several chunks of the same language in one call (padded to the longest)."""
        kw = self._kw(lang)
        rs = self.model.recognize(list(xs), sample_rate=SAMPLE_RATE, **kw)
        return [self._result(r, len(x), kw) for r, x in zip(rs, xs, strict=True)]


def _sessions(obj) -> dict:
    """onnxruntime sessions inside an onnx-asr model (encoder, decoder, joint...)."""
    import onnxruntime as rt

    inner = getattr(obj, "asr", obj)
    return {k: v for k, v in vars(inner).items() if isinstance(v, rt.InferenceSession)}


def _device(obj) -> str:
    """"cuda" when the model's sessions run on the GPU (the encoder is what matters)."""
    ss = _sessions(obj)
    return "cuda" if any(s.get_providers()[0] == "CUDAExecutionProvider" for s in ss.values()) else "cpu"


def _decoder_to_cpu(obj, path, threads: int) -> None:
    """RNN-T / TDT: the prediction network and the joint run once per audio frame on a single
    vector; on the GPU each call adds two copies and a kernel launch. Encoder stays on the GPU."""
    import onnxruntime as rt

    inner = getattr(obj, "asr", obj)
    for attr, pattern in (("_decoder", "*rnnt_decoder*.onnx"), ("_joiner", "*rnnt_joint*.onnx"),
                          ("_decoder_joint", "decoder_joint-model*.onnx")):
        if not isinstance(getattr(inner, attr, None), rt.InferenceSession):
            continue
        files = sorted(path.glob(pattern))
        if len(files) != 1:
            continue
        so = rt.SessionOptions()
        so.intra_op_num_threads = min(threads, 2)
        setattr(inner, attr, rt.InferenceSession(str(files[0]), sess_options=so,
                                                 providers=["CPUExecutionProvider"]))


def _words(tokens: list[str], times: list[float], logprobs: list[float] | None, dur: float) -> list[dict]:
    """Sub-word tokens with start times -> words. A token starting with a space (or ▁) opens a word."""
    out: list[dict] = []
    cur = None
    for i, tok in enumerate(tokens):
        t = float(times[i]) if i < len(times) else (out[-1]["e"] if out else 0.0)
        lp = logprobs[i] if logprobs and i < len(logprobs) else None
        starts = tok[:1] in (" ", "▁")
        piece = tok.replace("▁", " ").strip()
        if starts or cur is None:
            if cur and cur["w"]:
                out.append(cur)
            cur = dict(w="", s=t, e=t, lps=[])
            if not piece:
                continue
        cur["w"] += piece
        cur["e"] = t
        if lp is not None:
            cur["lps"].append(lp)
    if cur and cur["w"]:
        out.append(cur)
    res = []
    for i, w in enumerate(out):
        nxt = out[i + 1]["s"] if i + 1 < len(out) else dur
        end = min(max(w["e"] + 0.12, w["s"] + 0.08 * len(w["w"])), nxt, dur)
        p = None
        if w["lps"]:
            p = round(min(1.0, math.exp(min(w["lps"]))), 3)
        res.append(dict(w=w["w"], s=round(w["s"], 3), e=round(max(end, w["s"] + 0.05), 3), p=p))
    return res
