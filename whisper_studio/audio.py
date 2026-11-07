"""Audio I/O. Guide rule: 16 kHz mono PCM, no AGC, no silence cutting."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .config import SAMPLE_RATE

# Formats the desktop web view (Chromium/WebView2) plays natively
PLAYABLE = {".ogg", ".oga", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".flac", ".webm", ".mp4"}


def channels(path: str | Path) -> int:
    import av

    with av.open(str(path)) as c:
        s = next((s for s in c.streams if s.type == "audio"), None)
        return s.codec_context.channels if s else 0


def load(path: str | Path, split_stereo: bool = False):
    """Decode any format to float32 16 kHz. Returns mono array or (left, right)."""
    import av

    layout = "stereo" if split_stereo else "mono"
    res = av.AudioResampler(format="s16", layout=layout, rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(path)) as c:
        stream = next(s for s in c.streams if s.type == "audio")
        for frame in c.decode(stream):
            frame.pts = None
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    pcm = (np.concatenate(chunks) if chunks else np.zeros(0, np.int16)).astype(np.float32) / 32768.0
    if split_stereo:
        return pcm[0::2].copy(), pcm[1::2].copy()
    return pcm


def highpass(x: np.ndarray, hz: float = 80.0) -> np.ndarray:
    """Gentle anti-hum filter (guide: HPF 80 Hz helped on noisy recordings)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(2, hz, btype="highpass", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def peaks(x: np.ndarray, per_second: int = 40) -> list[float]:
    step = SAMPLE_RATE // per_second
    n = len(x) // step
    if n == 0:
        return []
    m = np.abs(x[: n * step]).reshape(n, step).max(axis=1)
    top = float(m.max()) or 1.0
    return [round(float(v), 3) for v in m / top]


def make_playable(src: Path, dest_dir: Path) -> Path:
    """Copy the original for playback, or transcode exotic formats to Opus."""
    ext = src.suffix.lower()
    if ext in PLAYABLE:
        dst = dest_dir / f"audio{ext}"
        if src.resolve() != dst.resolve():  # uploads are already stored in place
            shutil.copy2(src, dst)
        return dst
    import av

    dst = dest_dir / "audio.ogg"
    with av.open(str(src)) as inp, av.open(str(dst), "w", format="ogg") as out:
        ost = out.add_stream("libopus", rate=48000)
        ost.bit_rate = 96000
        for frame in inp.decode(audio=0):
            frame.pts = None
            for p in ost.encode(frame):
                out.mux(p)
        for p in ost.encode(None):
            out.mux(p)
    return dst
