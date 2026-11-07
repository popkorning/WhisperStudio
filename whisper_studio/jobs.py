"""Job queue shared by the desktop app, the MCP server and the CLI.

Heavy work (Whisper, alignment, diarization, LLM proofreading) never runs in the UI or MCP
process. They only drop a job file into HOME/queue/ and make sure the worker process is up
(`python -m whisper_studio worker`, see worker.py). One worker per machine (lock file), it
takes jobs oldest first and quits after a short idle time, so RAM and VRAM are given back.

Job files survive restarts: a file is deleted only when its job has finished.
This module must stay light: stdlib + store only, no numpy/torch.
"""
from __future__ import annotations

import json
import msvcrt
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import store
from .config import HOME

QUEUE_DIR = HOME / "queue"
LOCK_FILE = QUEUE_DIR / "worker.lock"     # locked by the running worker (released by the OS on exit)
STATE_FILE = QUEUE_DIR / "worker.json"    # pid and the job in progress
DEVICE_FILE = QUEUE_DIR / "device.json"   # ("cuda", "float16") probed once by a worker
LOG_FILE = QUEUE_DIR / "worker.log"
# OpenBLAS commits ~16 MB per core when numpy loads. The UI and MCP processes
# only touch numpy in passing, so they run it single-threaded; the worker keeps the default.
LIGHT_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
SPAWN_FILE = QUEUE_DIR / "spawn.json"     # a worker being started (cold start takes 10-40 s)
SPAWN_GRACE = 120.0
KINDS = ("process", "rediarize", "proofread")

QUEUE_DIR.mkdir(parents=True, exist_ok=True)


Busy = store.Busy  # the document is still being processed


# ------------------------------------------------------------------ files

def _write(f: Path, data: dict) -> None:
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), "utf-8")
    for attempt in range(40):
        try:
            tmp.replace(f)
            return
        except PermissionError:  # a reader has the file open right now
            if attempt == 39:
                raise
            time.sleep(0.05)


def _read(f: Path) -> dict | None:
    for _ in range(20):
        try:
            return json.loads(f.read_text("utf-8"))
        except FileNotFoundError:
            return None
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
    return None


def files() -> list[Path]:
    """Job files, oldest first (names start with a nanosecond timestamp)."""
    return sorted(QUEUE_DIR.glob("job-*.json"))


def pending() -> list[dict]:
    out = []
    for f in files():
        j = _read(f)
        if j:
            j["file"] = f
            out.append(j)
    return out


def queued_count() -> int:
    """Jobs waiting for the worker (the one in progress is not counted)."""
    return sum(1 for j in pending() if not j.get("started"))


def has_job(doc_id: str) -> bool:
    return any(f.name.endswith(f"-{doc_id}.json") for f in files())


# ------------------------------------------------------------------ enqueue

def enqueue(kind: str, doc_id: str, **args) -> None:
    if kind not in KINDS:
        raise ValueError(kind)
    f = QUEUE_DIR / f"job-{time.time_ns():020d}-{kind}-{doc_id}.json"
    _write(f, dict(kind=kind, doc_id=doc_id, args=args, created=time.time()))
    ensure_worker()


def submit_process(doc_id: str) -> None:
    """Transcribe a document created by pipeline.submit / store.create (status "queued")."""
    enqueue("process", doc_id)


def post_job(doc_id: str, kind: str, **args) -> dict:
    """Queue re-diarization / proofreading of a finished document; returns its summary."""
    with store.editing(doc_id) as d:
        if d["status"] != "done" or has_job(doc_id):
            raise Busy("Документ ещё обрабатывается")
        store.save(d)  # undo point before the change
        d.update(status="processing", stage="В очереди", progress=0.0, error=None)
        store.save(d, snapshot=False)
    enqueue(kind, doc_id, **args)
    return store.summary(d)


# ------------------------------------------------------------------ worker process

def _locked() -> bool:
    """True while a worker holds the lock. The OS drops the lock when the process dies."""
    try:
        fd = os.open(LOCK_FILE, os.O_RDWR | os.O_CREAT)
    except OSError:
        return True
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError:
        return True
    else:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return False
    finally:
        os.close(fd)


def worker_state() -> dict:
    """{"running": bool, "pid": int | None, "current": doc_id | None}"""
    if not _locked():
        return dict(running=False, pid=None, current=None)
    st = _read(STATE_FILE) or {}
    return dict(running=True, pid=st.get("pid"), current=st.get("current"))


def _alive(pid: int | None) -> bool:
    import ctypes

    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, int(pid or 0))  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def _starting() -> bool:
    """A worker was spawned recently (by any process) and hasn't taken the lock yet."""
    sp = _read(SPAWN_FILE) or {}
    age = time.time() - sp.get("t", 0)
    return age < 30 or (age < SPAWN_GRACE and _alive(sp.get("pid")))  # 30 s: no respawn storm


def _python() -> str:
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return str(w if w.exists() else exe)


def _spawn_wmi(cmd: list[str], cwd: str, env: dict) -> int:
    """Start a process through WMI: it is created by the WMI service, outside our job object."""
    import pythoncom
    import win32com.client

    def create() -> int:
        wmi = win32com.client.GetObject(r"winmgmts:root\cimv2")
        startup = wmi.Get("Win32_ProcessStartup").SpawnInstance_()
        startup.ShowWindow = 0
        startup.PriorityClass = 16384  # below normal
        startup.EnvironmentVariables = [f"{k}={v}" for k, v in env.items()]
        proc = wmi.Get("Win32_Process")
        inp = proc.Methods_("Create").InParameters.SpawnInstance_()
        inp.CommandLine = subprocess.list2cmdline(cmd)
        inp.CurrentDirectory = cwd
        inp.ProcessStartupInformation = startup
        res = proc.ExecMethod_("Create", inp)
        if res.ReturnValue != 0:
            raise OSError(f"Win32_Process.Create: {res.ReturnValue}")
        return int(res.ProcessId)

    pythoncom.CoInitialize()
    try:
        return create()  # COM objects are released when create() returns, before CoUninitialize
    finally:
        pythoncom.CoUninitialize()


def _in_job() -> bool:
    import ctypes

    k = ctypes.windll.kernel32
    res = ctypes.c_int()
    ok = k.IsProcessInJob(ctypes.c_void_p(k.GetCurrentProcess()), None, ctypes.byref(res))
    return bool(ok and res.value)


def _log(msg: str) -> None:
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {msg}\n")
    except OSError:
        pass


def _spawn(*extra: str, command: str = "worker", log_file: Path = LOG_FILE) -> int:
    """Start `python -m whisper_studio worker` (or another command), detached, and return its pid.
    It must outlive whoever started it: MCP clients run their servers inside a kill-on-close
    job object, which would take the worker (and a half-done transcription) down with them."""
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ, PYTHONPATH=pkg_root, PYTHONIOENCODING="utf-8")
    for k, v in LIGHT_ENV.items():
        if env.get(k) == v:
            env.pop(k)
    flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
             | subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    if log_file.exists() and log_file.stat().st_size > 2_000_000:
        log_file.replace(log_file.with_suffix(".log.1"))
    cmd = [_python(), "-m", "whisper_studio", command, *extra]
    if _in_job():
        # Breakaway is not enough: the venv launcher and MCP clients nest jobs, and a "broken away"
        # child still ends up in an outer one. A process created by the WMI service is in none.
        try:
            pid = _spawn_wmi(cmd, pkg_root, env)
            _log(f"spawned {command} {pid} through WMI {' '.join(extra)}")
            return pid
        except Exception as e:  # noqa: BLE001
            _log(f"WMI spawn failed ({e!r}), the worker may stop together with its caller")
    with open(log_file, "ab") as log:
        kw = dict(cwd=pkg_root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True)
        try:
            pid = subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kw).pid
        except OSError:  # inside a job that does not allow breakaway
            pid = subprocess.Popen(cmd, creationflags=flags, **kw).pid
    _log(f"spawned {command} {pid} {' '.join(extra)}")
    return pid


_spawn_lock = threading.Lock()


def ensure_worker() -> bool:
    """Start the worker if there is work and none is running. Returns True if it is (now) running.
    Call it after the job file is written: a worker that is about to quit re-checks the queue
    after releasing its lock, so a job can't be left behind."""
    with _spawn_lock:
        if _locked() or _starting():
            return True
        if not files():
            return False
        _write(SPAWN_FILE, dict(pid=_spawn(), t=time.time()))
        return True


_kick_t = 0.0


def kick(every: float = 5.0) -> None:
    """Cheap self-healing for pollers: restart the worker if jobs wait and it died."""
    global _kick_t
    now = time.monotonic()
    if now - _kick_t >= every:
        _kick_t = now
        try:
            ensure_worker()
        except Exception:  # noqa: BLE001 - never break a status poll
            pass


# ------------------------------------------------------------------ device

_device_probe: threading.Thread | None = None


def device(wait: float = 3.0) -> tuple[str | None, str | None]:
    """Compute device as seen by ctranslate2, without importing it here.
    Cached in DEVICE_FILE by the worker; the first time ever a tiny probe process fills it."""
    global _device_probe
    d = _read(DEVICE_FILE)
    if d:
        return d.get("device"), d.get("compute_type")
    if _device_probe is None:
        def probe():
            try:
                pid, t = _spawn("--probe-device"), time.monotonic()
                while _alive(pid) and time.monotonic() - t < 120:
                    time.sleep(0.2)
            except Exception:  # noqa: BLE001
                pass

        _device_probe = threading.Thread(target=probe, daemon=True)
        _device_probe.start()
    _device_probe.join(wait)
    d = _read(DEVICE_FILE) or {}
    return d.get("device"), d.get("compute_type")


# ------------------------------------------------------------------ gold evaluation (gold.py)
# Not a queue job: the GPU queue may be busy for hours and the evaluation needs no GPU. A separate
# CPU process (`python -m whisper_studio gold-eval`), one at a time; GOLD_RUN marks it running.

GOLD_DIR = HOME / "gold"
GOLD_LATEST = GOLD_DIR / "latest.json"
GOLD_HISTORY = GOLD_DIR / "history.jsonl"
GOLD_RUN = GOLD_DIR / "running.json"      # {"pid", "t", "tune"}; the process removes it when done
GOLD_ERROR = GOLD_DIR / "error.json"      # {"t", "error"} of the last failed run
GOLD_LOG = GOLD_DIR / "eval.log"
_gold_lock = threading.Lock()


def _mtime(f: Path) -> float:
    try:
        return f.stat().st_mtime
    except OSError:
        return 0.0


def gold_state() -> dict:
    """{"running": bool, "error": str | None}. The marker of a process that died is cleared."""
    run = _read(GOLD_RUN)
    running = bool(run) and (_alive(run.get("pid")) or time.time() - run.get("t", 0) < 5)
    if run and not running:
        GOLD_RUN.unlink(missing_ok=True)
        t = run.get("t", 0)
        if _mtime(GOLD_LATEST) < t and _mtime(GOLD_ERROR) < t:  # died without a word
            _write(GOLD_ERROR, dict(t=time.time(), error="Оценка прервалась: процесс завершился (см. gold/eval.log)"))
    err = None if running else (_read(GOLD_ERROR) or {}).get("error")
    return dict(running=running, error=err)


def start_gold_eval(tune: bool = False) -> bool:
    """Start the evaluation unless one is running. Returns False if one already was."""
    with _gold_lock:
        if gold_state()["running"]:
            return False
        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        GOLD_ERROR.unlink(missing_ok=True)
        t = time.time()
        _write(GOLD_RUN, dict(pid=None, t=t, tune=tune))  # before the spawn: other processes see it
        try:
            pid = _spawn(*(["--tune"] if tune else []), command="gold-eval", log_file=GOLD_LOG)
        except BaseException:
            GOLD_RUN.unlink(missing_ok=True)
            raise
        if GOLD_RUN.exists():  # not finished already
            _write(GOLD_RUN, dict(pid=pid, t=t, tune=tune))
        return True


def gold_result() -> dict | None:
    return _read(GOLD_LATEST)


def gold_history(limit: int = 50) -> list[dict]:
    """Short records of past evaluations, newest first."""
    try:
        lines = GOLD_HISTORY.read_text("utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


def save_gold_result(res: dict) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    _write(GOLD_LATEST, res)
    if res.get("merged"):
        rec = dict(t=res["t"], wer=res["merged"]["wer"], words=res["words"], docs=res["docs"])
        with open(GOLD_HISTORY, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def gold_failed(error: str) -> None:
    _write(GOLD_ERROR, dict(t=time.time(), error=error))


def gold_finished() -> None:
    GOLD_RUN.unlink(missing_ok=True)
