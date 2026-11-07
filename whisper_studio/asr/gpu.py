"""Sharing the GPU with other programs: wait for free video memory instead of failing."""
from __future__ import annotations

import os
import subprocess
import time
from typing import Callable

POLL = 15.0


def wait_limit() -> float:
    return float(os.environ.get("WHISPER_STUDIO_GPU_WAIT", "1200"))


def free_gb(torch_cache: bool = False) -> float | None:
    """Free video memory of GPU 0 in GB, None if there is no NVIDIA GPU / driver tool."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        free = int(r.stdout.strip().splitlines()[0]) / 1024
    except Exception:  # noqa: BLE001
        return None
    return free + (_own_cached_gb() if torch_cache else 0.0)


def _own_cached_gb() -> float:
    """Video memory torch keeps cached in this very process after an earlier model: it is
    ours to reuse, but nvidia-smi counts it as taken (a model would wait for itself)."""
    import sys

    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            return (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024 ** 3
    except Exception:  # noqa: BLE001
        pass
    return 0.0


_gave_up = {"at": -1e9}
# seconds spent waiting for video memory in this process (ensemble timing: load = wait + reading)
stats = {"waited": 0.0}


def wait(need_gb: float, note: Callable[[str], None] | None = None, limit: float | None = None,
         torch_cache: bool = False) -> bool:
    """Block until need_gb of video memory is free. False if it never got free within the limit.
    After one timeout, the next models of the same job only give it a minute."""
    limit = wait_limit() if limit is None else limit
    t0 = time.monotonic()
    if t0 - _gave_up["at"] < 1800:
        limit = min(limit, 60.0)
    while True:
        free = free_gb(torch_cache)
        if free is None or free >= need_gb:
            return True
        waited = time.monotonic() - t0
        if waited >= limit:
            _gave_up["at"] = time.monotonic()
            return False
        if note:
            note(f"Жду видеопамять: свободно {free:.1f} ГБ, нужно {need_gb:.1f} ГБ "
                 f"(занята другой программой, {int(waited // 60)} мин)")
        time.sleep(POLL)
        stats["waited"] += POLL


def is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda_error_out_of_memory" in s or "cublas_status_alloc_failed" in s
