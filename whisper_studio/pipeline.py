"""End-to-end processing of one file and post-processing jobs.

These run in the worker process (worker.py); jobs.py queues them."""
from __future__ import annotations

import json
import re
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from . import audio as A
from . import engine, models, prefetch, settings, store
from .config import SAMPLE_RATE
from .options import make_options, submit  # noqa: F401 - re-exported


class Reporter:
    """Throttled progress writer into doc.json (no undo snapshots)."""

    def __init__(self, doc: dict, echo=None):
        self.doc = doc
        self.echo = echo
        self._t = 0.0

    def __call__(self, frac: float, stage: str, force: bool = False):
        self.doc["progress"] = round(max(0.0, min(frac, 1.0)), 3)
        self.doc["stage"] = stage
        if self.echo:
            self.echo(self.doc["progress"], stage)
        now = time.monotonic()
        if force or now - self._t > 0.5:
            self._t = now
            store.save(self.doc, snapshot=False)

    def span(self, lo: float, hi: float, prefix: str = ""):
        return lambda f, msg: self(lo + (hi - lo) * f, prefix or msg)


def _audio_source(doc: dict) -> Path:
    src = Path(doc.get("source_path") or "")
    if src.is_file():
        return src
    if doc.get("audio_file"):
        return store.path(doc["id"]) / doc["audio_file"]
    raise FileNotFoundError("Аудио этой записи не найдено")


def process(doc_id: str, echo=None) -> dict:
    doc = store.load(doc_id)
    rep = Reporter(doc, echo)
    try:
        ctx = _prepare(doc, rep)
        results = []
        for ci, tr in enumerate(ctx["tracks"]):
            span, cache = _track(ctx, ci)
            results.append(engine.recognize(tr, ctx["opts"], span, cache=cache, tail=ctx["tail"]))
        return _finish(ctx, _collect(ctx, results))
    except Exception as e:  # noqa: BLE001
        _failed(doc, e)
        raise


def _prepare(doc: dict, rep: Reporter) -> dict:
    """Audio, playable copy, waveform peaks and the language: everything before recognition."""
    opts = engine.Options(**doc["options"])
    folder = store.path(doc["id"])
    src = Path(doc["source_path"])
    doc["status"] = "processing"
    rep(0.01, "Читаю аудио", force=True)
    play = A.make_playable(src, folder)
    doc["audio_file"] = play.name

    stereo = opts.stereo_split and A.channels(src) >= 2
    tracks = list(A.load(src, split_stereo=True)) if stereo else [A.load(src)]
    if opts.highpass:
        tracks = [A.highpass(t) for t in tracks]
    mono = tracks[0] if len(tracks) == 1 else (tracks[0] + tracks[1]) / 2
    doc["duration"] = round(len(mono) / SAMPLE_RATE, 3)
    (folder / "peaks.json").write_text(str(A.peaks(mono)), "utf-8")
    rep(0.03, "Аудио готово", force=True)

    if not opts.language:
        # Explicit large-v3 + auto supports code-switching within a recording.
        if opts.model == "large-v3" and not opts.mixed:
            opts.extra.setdefault("multilingual", True)
            doc["meta"]["multilingual"] = True
        rep(0.04, "Определяю язык", force=True)
        lang, prob = engine.detect_language(mono)
        opts.language = lang
        doc["meta"]["detected_language"] = dict(language=lang, probability=round(prob, 3))
    doc["language"] = opts.language
    # aligners come after recognition: prefetch.py reads them while the last models work
    langs = [opts.language] + (["de"] if opts.mixed and opts.language != "de" else [])
    tail = [p for p in (models.aligner_for(lang) for lang in langs) if p] if opts.align else []
    # with speed.diarize_gpu the voices take a minute or two on the GPU after recognition:
    # no CPU process next to the models
    sp = settings.speed()
    if (opts.diarize and len(tracks) == 1 and opts.num_speakers != 1 and sp.get("early_voices")
            and not sp.get("diarize_gpu") and not (folder / "voices.npz").exists()):
        dp = models.find("diarization")
        if dp:
            Voices.start(doc["id"], dp)
    return dict(doc=doc, rep=rep, opts=opts, folder=folder, tracks=tracks, mono=mono, n=len(tracks), tail=tail)


def _track(ctx: dict, ci: int):
    """Progress span and hypotheses cache of one track (a stereo call has two)."""
    n, rep, opts = ctx["n"], ctx["rep"], ctx["opts"]
    lo, hi = 0.05 + 0.65 * ci / n, 0.05 + 0.65 * (ci + 1) / n
    label = "Распознаю речь" + (f" (канал {ci + 1})" if n > 1 else "")
    # "max" reports which model is running (and GPU waits); "fast" shows just the label
    span = rep.span(lo, hi) if opts.quality == "max" else rep.span(lo, hi, label)
    cache = ctx["folder"] / ("hypotheses.json" if n == 1 else f"hypotheses-{ci + 1}.json")
    return span, cache


def _collect(ctx: dict, results: list[tuple[list[dict], dict]]) -> list[dict]:
    doc, n = ctx["doc"], ctx["n"]
    segments: list[dict] = []
    meta: dict = {}
    for ci, (segs, meta) in enumerate(results):
        if n > 1:
            for s in segs:
                s["speaker"] = f"S{ci + 1}"
        segments += segs
    segments.sort(key=lambda s: s["start"])
    for i, s in enumerate(segments):
        s["src"] = i  # Whisper segment of origin: lets re-diarization glue pieces back
    doc["meta"].update(meta)
    engine.park()
    return segments


def _finish(ctx: dict, segments: list[dict]) -> dict:
    """Dictionary, word alignment, speakers, and the document is done."""
    doc, rep, opts, n = ctx["doc"], ctx["rep"], ctx["opts"], ctx["n"]
    tracks, mono, folder = ctx["tracks"], ctx["mono"], ctx["folder"]
    from .asr import itn

    itn.renumber_doc(dict(segments=segments))  # numbers in digits; the ensemble did it already
    if opts.dictionary:
        doc["meta"]["dictionary_replacements"] = store.apply_rules_to_segments(segments, settings.rules())

    if opts.align:
        # aligner per segment language: German phrases in a Russian call get the German one
        by_lang: dict = {}
        for s in segments:
            by_lang.setdefault(engine.segment_language(s, opts.language), []).append(s)
        plan = [(lang, models.aligner_for(lang), part) for lang, part in by_lang.items()]
        plan = [p for p in plan if p[1]]
        if plan:
            rep(0.72, "Уточняю таймкоды слов", force=True)
            from . import align

            try:
                total, done = sum(len(p[2]) for p in plan), 0
                for _lang, mp, part in plan:
                    if n > 1:
                        for ci, tr in enumerate(tracks):
                            align.align([s for s in part if s.get("speaker") == f"S{ci + 1}"], tr, mp)
                    else:
                        align.align(part, mono, mp, rep.span(0.72 + 0.12 * done / total,
                                                             0.72 + 0.12 * (done + len(part)) / total))
                    done += len(part)
                doc["meta"]["aligned"] = True
                doc["meta"]["aligners"] = {lang: len(part) for lang, _, part in plan}
            except Exception as e:  # noqa: BLE001 - keep Whisper timings
                doc["meta"]["align_error"] = str(e)
            finally:
                align.park()

    if opts.diarize and n == 1 and opts.num_speakers != 1:
        dp = models.find("diarization")
        if dp:
            from . import diarize as D

            rep(0.85, "Определяю спикеров", force=True)
            Voices.take(doc["id"], rep.span(0.85, 0.97, "Определяю спикеров"))
            segments, info = D.run(segments, mono, dp, opts.num_speakers,
                                   progress=rep.span(0.85, 0.98), cache=folder / "voices.npz")
            doc["meta"]["diarized"] = True
            doc["meta"]["diarization"] = info
        else:
            doc["meta"]["diarize_skipped"] = "модель спикеров не установлена"
    if opts.num_speakers == 1 and n == 1:
        for s in segments:
            s["speaker"] = "S1"

    store.finalize(doc, segments)
    if (doc["meta"].get("diarization") or {}).get("owner"):
        from . import diarize as D

        D.name_owner(doc, doc["meta"]["diarization"])  # the owner's voice (HOME/voices/owner.npz)
    doc["options"] = asdict(opts)
    doc.update(status="done", progress=1.0, stage="Готово", error=None)
    store.save(doc, snapshot=False)
    return doc


def _failed(doc: dict, e: BaseException, tb: str | None = None) -> None:
    doc.update(status="error", error=f"{e}", stage="Ошибка")
    doc["meta"]["traceback"] = tb or traceback.format_exc()
    store.save(doc, snapshot=False)


def batchable(doc: dict) -> bool:
    """A recording process_batch can take: audio in quality "max"."""
    return (doc.get("options") or {}).get("quality", "max") == "max"


def process_batch(doc_ids: list[str], echo=None, log=None) -> list[str]:
    """Several queued recordings model by model ("model-major", settings speed.batch): each
    ensemble model is loaded once from disk and hears the chunks of every recording,
    instead of 7-8 loads (~12 GB, ~3 min) per recording. The text is the same as process()
    gives: every member hears the same chunks with the same language and prompt, only the
    order of the work changes.

    1. one by one: audio, language, scan, chunks and the detector;
    2. ensemble.run_batch: members 1.. model by model over all recordings, then the German re-check;
    3. one by one: merge, dictionary, alignment, speakers.
    The hypotheses cache is saved after every model, so a crash resumes like process() does.
    Audio stays in memory for the whole batch: recordings stop being added once their decoded
    audio passes speed.batch_audio_gb (the rest waits for the next batch).
    Returns the ids taken (done or failed); the others were not touched."""
    from . import ensemble

    log = log or (lambda msg: None)
    budget = float(settings.speed().get("batch_audio_gb", 1.5)) * (1 << 30)
    ctxs, taken, used = [], [], 0
    t0 = time.monotonic()
    for doc_id in doc_ids:
        if ctxs and used >= budget:
            break
        try:
            doc = store.load(doc_id)
        except Exception:  # noqa: BLE001 - deleted meanwhile
            taken.append(doc_id)
            continue
        taken.append(doc_id)
        rep = Reporter(doc, (lambda f, m, i=doc_id: echo(i, f, m)) if echo else None)
        try:
            if not batchable(doc):
                process(doc_id, (lambda f, m, i=doc_id: echo(i, f, m)) if echo else None)
                continue
            ctx = _prepare(doc, rep)
            used += sum(t.nbytes for t in ctx["tracks"]) + (ctx["mono"].nbytes if ctx["n"] > 1 else 0)
            if not ensemble.available(ctx["opts"]):  # not enough models: one Whisper, right away
                results = [engine.recognize(tr, ctx["opts"], _track(ctx, ci)[0])
                           for ci, tr in enumerate(ctx["tracks"])]
                _finish(ctx, _collect(ctx, results))
                continue
            ctx["runs"] = []
            for ci, tr in enumerate(ctx["tracks"]):
                span, cache = _track(ctx, ci)
                r = ensemble.begin(tr, ctx["opts"], span, cache)
                r.update(tail=ctx["tail"], doc_id=doc_id, label_suffix=" · пакетом")
                if r["mem"]:
                    prefetch.ahead(ensemble.upcoming(r, 0))
                    ensemble.run_member(r, 0)
                ctx["runs"].append(r)
            ctxs.append(ctx)
            log(f"batch: prepared {doc_id} ({ctx['doc']['duration'] / 60:.1f} min)")
        except Exception as e:  # noqa: BLE001 - this recording fails, the others go on
            log(f"batch: {doc_id} failed:\n{traceback.format_exc()}")
            _try(lambda: _failed(doc, e))
    runs = [r for c in ctxs for r in c["runs"]]
    for r in runs:
        r["label_suffix"] = f" · пакет из {len(ctxs)} записей"
    log(f"batch: {len(ctxs)} recordings prepared in {time.monotonic() - t0:.1f}s")
    ensemble.run_batch(runs, log)
    log(f"batch: models done at {time.monotonic() - t0:.1f}s")
    for ctx in ctxs:
        doc = ctx["doc"]
        try:
            bad = next((r for r in ctx["runs"] if r.get("failed")), None)
            if bad is not None:
                _failed(doc, bad["failed"], bad.get("failed_tb"))
                continue
            _finish(ctx, _collect(ctx, [r["result"] for r in ctx["runs"]]))
        except Exception as e:  # noqa: BLE001
            log(f"batch: {doc['id']} failed:\n{traceback.format_exc()}")
            _try(lambda: _failed(doc, e))
        finally:
            ctx["tracks"] = ctx["mono"] = None
            for r in ctx["runs"]:
                r["audio"] = None
    log(f"batch: finished in {time.monotonic() - t0:.1f}s")
    return taken


def _try(fn) -> None:
    try:
        fn()
    except Exception:  # noqa: BLE001 - e.g. the document was deleted meanwhile
        pass


class Voices:
    """Speaker voices computed ahead (settings speed.early_voices). Diarization takes ~1x real
    time on the CPU after recognition (half of a job), but it depends on the audio only. So it
    runs in a separate process (`python -m whisper_studio voices`: sherpa-onnx holds the GIL,
    a thread here would stall the GPU members) at idle priority while the models recognize, one
    recording at a time, and writes voices.npz; diarize.run then just reads it (the same numbers:
    the cache keeps them exactly). A recording whose turn comes before its voices started
    computes them itself, as without the setting."""

    _lock = threading.Lock()
    _queue: list[tuple[str, Path]] = []
    _state: dict[str, str] = {}      # doc id -> "queued" | "running" | "done" | "failed"
    _thread: threading.Thread | None = None
    ran: list[str] = []              # computed by the separate process (tests, benchmarks)

    @classmethod
    def start(cls, doc_id: str, model_dir: Path) -> None:
        with cls._lock:
            if cls._state.get(doc_id) in ("queued", "running"):
                return
            cls._queue.append((doc_id, Path(model_dir)))
            cls._state[doc_id] = "queued"
            if cls._thread is None or not cls._thread.is_alive():
                cls._thread = threading.Thread(target=cls._run, name="voices", daemon=True)
                cls._thread.start()

    @classmethod
    def _run(cls) -> None:
        import os
        import subprocess
        import sys

        from . import jobs

        while True:
            with cls._lock:
                if not cls._queue:
                    cls._thread = None
                    return
                doc_id, md = cls._queue.pop(0)
                cls._state[doc_id] = "running"
            pkg_root = str(Path(__file__).resolve().parent.parent)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=pkg_root, PYTHONIOENCODING="utf-8")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)
            t = time.monotonic()
            try:
                with open(jobs.LOG_FILE, "ab") as log:
                    code = subprocess.run([sys.executable, "-m", "whisper_studio", "voices", doc_id, str(md)],
                                          cwd=pkg_root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                          creationflags=flags).returncode
            except OSError:
                code = -1
            ok = code == 0 and (store.path(doc_id) / "voices.npz").exists()
            jobs._log(f"voices {doc_id}: {'done' if ok else f'failed ({code})'} in {time.monotonic() - t:.1f}s")
            with cls._lock:
                cls._state[doc_id] = "done" if ok else "failed"
                if ok:
                    cls.ran.append(doc_id)

    @classmethod
    def take(cls, doc_id: str, progress=None) -> None:
        """Before diarize.run: wait for voices being computed; a recording still waiting in
        line is taken out (diarize.run computes them in this process, as before)."""
        with cls._lock:
            if cls._state.get(doc_id) == "queued":
                cls._queue = [q for q in cls._queue if q[0] != doc_id]
                cls._state.pop(doc_id, None)
                return
        prog = store.path(doc_id) / "voices.progress"
        while True:
            with cls._lock:
                st = cls._state.get(doc_id)
            if st != "running":
                break
            if progress:
                try:
                    f = float(json.loads(prog.read_text("utf-8")).get("f", 0.0))
                except (OSError, ValueError):
                    f = 0.0
                progress(f, "Определяю спикеров")
            time.sleep(0.5)
        with cls._lock:
            cls._state.pop(doc_id, None)


def _job(doc_id: str, stage: str, work, echo=None) -> dict:
    """Run a post-processing step on a finished document (the caller saved an undo point)."""
    doc = store.load(doc_id)
    rep = Reporter(doc, echo)
    try:
        doc["status"] = "processing"
        rep(0.02, stage, force=True)
        work(doc, rep)
        doc.update(status="done", progress=1.0, stage="Готово", error=None)
        doc["edited"] = True
        store.save(doc, snapshot=False)
        return doc
    except Exception as e:  # noqa: BLE001
        doc.update(status="done", progress=1.0, stage="Готово", error=f"{stage}: {e}")
        doc["meta"]["post_traceback"] = traceback.format_exc()
        store.save(doc, snapshot=False)
        raise


def _wkey(w: dict) -> tuple:
    return round(float(w.get("s") or 0), 2), w.get("w")


def _match_speakers(old: list[dict], new: list[dict]) -> dict[str, str]:
    """Old speaker id -> new one: the pairs sharing the most words, one to one."""
    was = {_wkey(w): s.get("speaker") for s in old for w in s.get("words") or []}
    pairs: dict[tuple, int] = {}
    for s in new:
        for w in s.get("words") or []:
            o = was.get(_wkey(w))
            if o and s.get("speaker"):
                pairs[o, s["speaker"]] = pairs.get((o, s["speaker"]), 0) + 1
    out: dict[str, str] = {}
    for (o, n), _ in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if o not in out and n not in out.values():
            out[o] = n
    return out


def _keep_locked(doc: dict, old: list[dict], old_speakers: dict, same: dict[str, str]) -> None:
    """Segments whose speaker was set by hand (speaker_locked) keep that speaker."""
    locked = {_wkey(w): s["speaker"] for s in old if s.get("speaker_locked") and s.get("speaker")
              for w in s.get("words") or []}
    if not locked:
        return
    for s in doc["segments"]:
        ws = s.get("words") or []
        hits = [locked[_wkey(w)] for w in ws if _wkey(w) in locked]
        if not ws or len(hits) * 2 <= len(ws):
            continue
        sid = max(set(hits), key=hits.count)
        if sid not in same:  # a speaker the voices don't know (added by hand): bring it back
            same[sid] = store._new_speaker(doc, (old_speakers.get(sid) or {}).get("name"))
        s["speaker"], s["speaker_locked"] = same[sid], True
    store._drop_unused_speakers(doc)


def rediarize(doc_id: str, num_speakers: int | None, echo=None) -> dict:
    """Re-assign speakers of an existing transcript (text and edits are kept)."""
    from . import diarize as D

    def work(doc, rep):
        dp = models.find("diarization")
        if not dp:
            raise RuntimeError("модель спикеров не установлена")
        cache = store.path(doc["id"]) / "voices.npz"
        mono = None
        if not cache.exists():
            rep(0.05, "Читаю аудио", force=True)
            mono = A.load(_audio_source(doc))
        names = {sid: sp["name"] for sid, sp in (doc.get("speakers") or {}).items()}
        old = [dict(s, words=list(s.get("words") or [])) for s in doc["segments"]]
        old_speakers = dict(doc.get("speakers") or {})
        segs = D.rejoin(doc["segments"])
        if num_speakers == 1:
            for s in segs:
                s["speaker"] = "S1"
            info = dict(speakers=1)
        else:
            segs, info = D.run(segs, mono, dp, num_speakers, cache=cache,
                               progress=rep.span(0.1, 0.98, "Определяю спикеров"))
        store.finalize(doc, segs)
        doc["meta"]["diarization"] = info
        doc["options"]["num_speakers"] = num_speakers
        same = _match_speakers(old, doc["segments"])
        _keep_locked(doc, old, old_speakers, same)
        # a renamed speaker keeps its name if the same voice is found again
        for sid, new_sid in same.items():
            nm = names.get(sid) or ""
            if new_sid in doc["speakers"] and nm and not re.fullmatch(r"Спикер \d+", nm):
                doc["speakers"][new_sid]["name"] = nm
        D.name_owner(doc, info)

    return _job(doc_id, "Определяю спикеров", work, echo)


def proofread(doc_id: str, echo=None) -> dict:
    from . import llm

    def work(doc, rep):
        n = llm.proofread_doc(doc, rep.span(0.05, 0.98, "Исправляю ИИ"))
        doc["meta"]["proofread"] = dict(changed=n, model=llm.status().get("model"),
                                        at=time.strftime("%Y-%m-%d %H:%M"))

    return _job(doc_id, "Исправляю ИИ", work, echo)
