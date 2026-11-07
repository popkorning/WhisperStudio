"""Read the next models' files into the OS file cache while the current model works.

On a slow disk, loading ru-large-v3 or Qwen3-ASR is tens of seconds of
pure reading, every recording again (a typical amount of RAM can't keep all ~12 GB of an ensemble
cached). The members run one after another, so while one decodes on the GPU the disk is idle:
a background thread reads the files of the members that come next, and their load then comes
from RAM. Nothing about recognition changes, only where the bytes come from.

    prefetch.ahead(["de-turbo", "qwen3-asr"])   # what comes next, in order (a budget in GB)
    with prefetch.loading("ru-large-v3"):       # a model is being loaded: the thread pauses,
        backend.load()                          # so two readers don't make the HDD seek

Off unless settings.speed()["prefetch"] (or WHISPER_STUDIO_PREFETCH=1). Stdlib only.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

BLOCK = 8 << 20                 # bytes per read; the pause check runs between blocks
SKIP_SUFFIX = {".md", ".txt", ".gitattributes", ".lock"}

_cv = threading.Condition()
_lanes: dict[str, list[Path]] = {}   # lane -> files still to read, in order
_pos: dict[Path, int] = {}           # bytes of a file already read (resumes after a pause)
_done: set[Path] = set()             # read to the end since the model was last loaded
_busy = 0                            # loads in progress (the reader waits)
_thread: threading.Thread | None = None
stats = dict(bytes=0, seconds=0.0, files=0)
_enabled: bool | None = None


def enabled() -> bool:
    global _enabled
    if _enabled is None:
        from . import settings

        _enabled = bool(settings.speed().get("prefetch"))
    return _enabled


def set_enabled(on: bool | None) -> None:
    """Tests and benchmarks; None = read the setting again."""
    global _enabled
    _enabled = on


def _budget() -> int:
    from . import settings

    return int(float(settings.speed().get("prefetch_gb", 6.0)) * (1 << 30))


def files_of(key_or_path) -> list[Path]:
    """Files of an installed model (a registry key or a folder), biggest last."""
    p = key_or_path if isinstance(key_or_path, Path) else None
    if p is None:
        from . import models

        try:
            p = models.find(str(key_or_path))
        except KeyError:
            p = None
    if p is None or not Path(p).exists():
        return []
    p = Path(p)
    fs = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file() and f.suffix not in SKIP_SUFFIX]
    return sorted(fs, key=lambda f: f.stat().st_size)


def ahead(keys: list, lane: str = "main") -> None:
    """Plan the reading of these models (in this order) for one lane of work, replacing its
    previous plan. Files already read are skipped; the plan stops at the byte budget."""
    if not enabled():
        return
    plan, size, seen = [], 0, set()
    budget = _budget()
    for k in keys:
        for f in files_of(k):
            if f in seen or f in _done:
                continue
            seen.add(f)
            try:
                left = f.stat().st_size - _pos.get(f, 0)
            except OSError:
                continue
            if size + left > budget:
                break
            plan.append(f)
            size += left
        else:
            continue
        break
    with _cv:
        _lanes[lane] = plan
        _start()
        _cv.notify_all()


def stop(lane: str | None = None) -> None:
    with _cv:
        if lane is None:
            _lanes.clear()
        else:
            _lanes.pop(lane, None)
        _cv.notify_all()


@contextmanager
def loading(key):
    """Around a model load: the reader pauses (one reader per disk), and the model's files
    count as not read any more (the next time they are needed the cache may have dropped them)."""
    global _busy
    if not enabled():
        yield
        return
    fs = set(files_of(key))
    with _cv:
        _busy += 1
        for lst in _lanes.values():
            lst[:] = [f for f in lst if f not in fs]
    try:
        yield
    finally:
        with _cv:
            _busy -= 1
            _done.difference_update(fs)
            for f in fs:
                _pos.pop(f, None)
            _cv.notify_all()


def idle() -> bool:
    with _cv:
        return not any(_lanes.values())


def wait_idle(timeout: float = 60.0) -> bool:
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if idle() and not _reading.is_set():
            return True
        time.sleep(0.02)
    return False


_reading = threading.Event()


def _start() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_run, name="prefetch", daemon=True)
        _thread.start()


def _next() -> Path | None:
    for lane in sorted(_lanes, key=lambda x: (x not in ("gpu", "main"), x)):
        if _lanes[lane]:
            return _lanes[lane][0]
    return None


def _run() -> None:
    buf = bytearray(BLOCK)
    view = memoryview(buf)
    while True:
        with _cv:
            while _busy or _next() is None:
                _cv.wait(5.0)
            f = _next()
            _reading.set()
        t0 = time.monotonic()
        n0 = _pos.get(f, 0)
        try:
            with open(f, "rb", buffering=0) as fh:
                fh.seek(n0)
                pos = n0
                while True:
                    with _cv:
                        if _busy or f not in [x for lst in _lanes.values() for x in lst[:1]]:
                            break  # a load started or the plan moved on: resume from pos later
                    n = fh.readinto(view)
                    with _cv:
                        if not n:
                            _done.add(f)
                            stats["files"] += 1
                            break
                        pos += n
                        stats["bytes"] += n
                        if f not in [x for lst in _lanes.values() for x in lst[:1]]:
                            break  # loading() took it meanwhile: forget the position
                        _pos[f] = pos
        except OSError:
            with _cv:
                _done.add(f)  # unreadable: don't try again
        finally:
            stats["seconds"] += time.monotonic() - t0
            with _cv:
                if f in _done:
                    for lst in _lanes.values():
                        if f in lst:
                            lst.remove(f)
                _reading.clear()
