"""Background worker process: runs queued jobs (see jobs.py) one at a time on the GPU.

Started on demand by the app, the MCP server or the CLI (`python -m whisper_studio worker`).
Only one runs per machine (lock file). Models stay loaded between consecutive jobs and the
process exits after IDLE_EXIT seconds without work, which frees RAM and VRAM.
"""
from __future__ import annotations

import msvcrt
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

from . import jobs, store

IDLE_EXIT = float(os.environ.get("WHISPER_STUDIO_WORKER_IDLE", "120"))
ORPHAN_AGE = timedelta(minutes=10)  # docs left "queued" by older app versions (no job file)
MAX_ATTEMPTS = 2


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{os.getpid()}] {msg}", flush=True)


class Lock:
    def __init__(self):
        self.fd = None

    def acquire(self, wait: float = 5.0) -> bool:
        fd = os.open(jobs.LOCK_FILE, os.O_RDWR | os.O_CREAT)
        t = time.monotonic()
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                self.fd = fd
                return True
            except OSError:  # held by another worker, or probed by jobs._locked() this instant
                if time.monotonic() - t > wait:
                    os.close(fd)
                    return False
                time.sleep(0.1)

    def release(self):
        if self.fd is not None:
            try:
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            finally:
                os.close(self.fd)
                self.fd = None


def _state(current: str | None = None, kind: str | None = None, batch: list[str] | None = None):
    jobs._write(jobs.STATE_FILE, dict(pid=os.getpid(), current=current, kind=kind, batch=batch,
                                      since=datetime.now().isoformat(timespec="seconds")))


def _run(job: dict) -> None:
    from . import pipeline

    kind, doc_id, args = job["kind"], job["doc_id"], job.get("args") or {}
    if kind == "process":
        pipeline.process(doc_id)
    elif kind == "rediarize":
        pipeline.rediarize(doc_id, args.get("num_speakers"))
    elif kind == "proofread":
        pipeline.proofread(doc_id)


def _batchable(doc: dict) -> bool:
    """Same rule as pipeline.batchable (that module is heavy to import here)."""
    return (doc.get("options") or {}).get("quality", "max") == "max"


def batch_for(job: dict) -> list[dict]:
    """With speed.batch on: this "process" job and the ones waiting right behind it, for
    pipeline.process_batch. Queue order is kept: any other job in between ends the batch."""
    from . import settings

    sp = settings.speed()
    if job["kind"] != "process" or not sp.get("batch"):
        return [job]
    try:
        if not _batchable(store.load(job["doc_id"])):
            return [job]
    except Exception:  # noqa: BLE001
        return [job]
    out = [job]
    seen = False
    for j in jobs.pending():
        f = j.pop("file")
        if f == job["file"]:
            seen = True
            continue
        if not seen or j.get("started"):
            continue
        if len(out) >= max(1, int(sp.get("batch_docs", 4))) or j["kind"] != "process":
            break
        try:
            d = store.load(j["doc_id"])
        except Exception:  # noqa: BLE001 - deleted: next_job drops its file
            continue
        if not _batchable(d):
            break
        out.append(j | {"file": f})
    return out


def run_batch(batch: list[dict]) -> None:
    from . import pipeline

    files = {}
    for job in batch:
        f = job.pop("file")
        job.update(started=time.time(), pid=os.getpid(), attempts=job.get("attempts", 0) + 1)
        jobs._write(f, job)
        files[job["doc_id"]] = (f, job)
    ids = list(files)
    _state(ids[0], "process", ids)
    log(f"start batch of {len(ids)}: {' '.join(ids)}")
    t = time.monotonic()
    try:
        taken = pipeline.process_batch(ids, log=log)
    except Exception:  # noqa: BLE001 - like a failed single job: the documents get the error
        tb = traceback.format_exc()
        log(f"batch failed:\n{tb}")
        taken = ids
        for doc_id in ids:
            try:
                d = store.load(doc_id)
                if d["status"] not in ("done", "error"):
                    d.update(status="error", stage="Ошибка", error="Пакетная обработка прервалась")
                    d["meta"]["traceback"] = tb
                    store.save(d, snapshot=False)
            except Exception:  # noqa: BLE001
                pass
    log(f"done batch in {time.monotonic() - t:.1f}s")
    for doc_id, (f, job) in files.items():
        if doc_id in taken:
            f.unlink(missing_ok=True)
        else:  # over the audio budget: back to the queue untouched
            job.pop("started", None)
            job.pop("pid", None)
            job["attempts"] = max(0, job.get("attempts", 1) - 1)
            jobs._write(f, job)


def _interrupted(job: dict) -> bool:
    """A job a previous worker started and never finished (crash, power loss, killed).
    Run it again, unless it already failed that way MAX_ATTEMPTS times."""
    if job.get("attempts", 0) < MAX_ATTEMPTS:
        return True
    try:
        d = store.load(job["doc_id"])
    except Exception:  # noqa: BLE001
        return False
    if job["kind"] == "process":
        d.update(status="error", stage="Ошибка", error="Обработка прервалась (воркер завершился аварийно)")
    else:  # post-processing: the text is intact, just give the document back
        d.update(status="done", progress=1.0, stage="Готово", error="Обработка прервалась")
    store.save(d, snapshot=False)
    return False


def recover() -> None:
    """Adopt documents left unfinished by app versions that kept the queue in memory."""
    known = {j["doc_id"] for j in jobs.pending()}
    old = (datetime.now() - ORPHAN_AGE).isoformat(timespec="seconds")
    for s in reversed(store.list_docs()):
        if s["status"] not in ("queued", "processing") or s["id"] in known or (s["updated"] or "") > old:
            continue
        d = store.load(s["id"])
        if not d.get("speakers"):
            log(f"recover {s['id']}: transcription")
            f = jobs.QUEUE_DIR / f"job-{time.time_ns():020d}-process-{s['id']}.json"
            jobs._write(f, dict(kind="process", doc_id=s["id"], args={}, created=time.time()))
        else:  # interrupted post-processing: text is intact
            d.update(status="done", progress=1.0, stage="Готово")
            store.save(d, snapshot=False)


def next_job() -> dict | None:
    for job in jobs.pending():
        f = job.pop("file")
        if not job.get("started"):
            return job | {"file": f}
        if job.get("pid") == os.getpid():
            continue
        # started by a worker that died
        if _interrupted(job):
            job.pop("started", None)
            return job | {"file": f}
        f.unlink(missing_ok=True)
    return None


def loop(lock: Lock) -> None:
    idle_since = time.monotonic()
    while True:
        job = next_job()
        if job is None:
            if time.monotonic() - idle_since < IDLE_EXIT:
                time.sleep(1.0)
                continue
            # quit: release first, then look again so a job queued meanwhile isn't stranded
            _state()
            lock.release()
            if not jobs.files() or not lock.acquire(wait=0.5):
                return
            idle_since = time.monotonic()
            continue
        try:
            store.load(job["doc_id"])
        except Exception:  # noqa: BLE001 - document was deleted meanwhile
            job["file"].unlink(missing_ok=True)
            continue
        batch = batch_for(job)
        if len(batch) > 1:
            run_batch(batch)
            _state()
            idle_since = time.monotonic()
            continue
        f = job.pop("file")
        job.update(started=time.time(), pid=os.getpid(), attempts=job.get("attempts", 0) + 1)
        jobs._write(f, job)
        _state(job["doc_id"], job["kind"])
        log(f"start {job['kind']} {job['doc_id']} {job.get('args') or ''}")
        t = time.monotonic()
        try:
            _run(job)
            log(f"done {job['doc_id']} in {time.monotonic() - t:.1f}s")
        except Exception:  # noqa: BLE001 - the error is stored in the document
            log(f"failed {job['doc_id']}:\n{traceback.format_exc()}")
        f.unlink(missing_ok=True)
        _state()
        idle_since = time.monotonic()


def probe_device() -> None:
    from . import engine

    dev, ctype = engine.device()
    jobs._write(jobs.DEVICE_FILE, dict(device=dev, compute_type=ctype))


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    # pythonw started through WMI has no std handles at all
    sys.stdout = sys.stderr = open(jobs.LOG_FILE, "a", encoding="utf-8", buffering=1)
    if "--probe-device" in argv:
        probe_device()
        return
    lock = Lock()
    if not lock.acquire():
        return  # another worker is running
    try:
        _state()
        log(f"worker up ({sys.executable}), idle exit {IDLE_EXIT:.0f}s")
        recover()
        if not jobs.files():
            return
        probe_device()
        from . import engine

        engine.prewarm()  # torch/transformers for the aligner, in the background
        loop(lock)
    except Exception:  # noqa: BLE001
        log("crashed:\n" + traceback.format_exc())
        raise
    finally:
        lock.release()
        log("worker exit")
        sys.stdout.flush()
        # Skip interpreter and DLL teardown: unloading CUDA/cuDNN takes ~10 s while holding GBs.
        import ctypes

        ctypes.windll.kernel32.TerminateProcess(ctypes.c_void_p(-1), 0)  # -1: this process
        os._exit(0)


if __name__ == "__main__":
    main()
