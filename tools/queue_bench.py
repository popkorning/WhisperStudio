"""Time a queue of "max" transcriptions with the speed settings on or off (docs/DECISIONS.md).

Runs in this process on the GPU, like the worker would, in a scratch home (copies of the
recordings' documents are made there; the real library is only read). Start it only while the
real worker is idle: two CUDA processes don't fit into 6 GB.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-bench
    set WHISPER_STUDIO_MODELS=%WS_REAL_HOME%\\models
    ws-python tools/queue_bench.py a1b2c3 d4e5f6 6d7e8f --out base.json
    ws-python tools/queue_bench.py a1b2c3 d4e5f6 6d7e8f --prefetch --early-voices --out pre.json --compare base.json
    ws-python tools/queue_bench.py a1b2c3 d4e5f6 6d7e8f --batch --prefetch --parallel-cpu --early-voices \\
        --out all.json --compare base.json

Inputs: audio files, or ids (last 6 characters are enough) of documents in the real library.
--compare checks that the text is the same as in another run (the speed settings must not
change it). The OS file cache remembers the models read last: for a fair baseline run it twice
and take the second, or compare the member load times in the "timing" lines.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--prefetch", action="store_true")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--parallel-cpu", action="store_true")
    p.add_argument("--early-voices", action="store_true")
    p.add_argument("--out", help="result JSON")
    p.add_argument("--compare", help="result JSON of another run: is the text the same?")
    a = p.parse_args()

    home = Path(os.environ.get("WHISPER_STUDIO_HOME") or "")
    if not home.name or home.resolve() == REAL_HOME.resolve():
        sys.exit("set WHISPER_STUDIO_HOME to a scratch folder (and WHISPER_STUDIO_MODELS to the real models)")
    if not os.environ.get("WHISPER_STUDIO_MODELS"):
        os.environ["WHISPER_STUDIO_MODELS"] = str(REAL_HOME / "models")
    home.mkdir(parents=True, exist_ok=True)
    for f in ("dictionary.json", "settings.json"):  # the same prompt and defaults as the app
        if (REAL_HOME / f).exists() and not (home / f).exists():
            shutil.copy2(REAL_HOME / f, home / f)
    flags = dict(PREFETCH=a.prefetch, BATCH=a.batch, PARALLEL_CPU=a.parallel_cpu, EARLY_VOICES=a.early_voices)
    for k, v in flags.items():
        os.environ[f"WHISPER_STUDIO_{k}"] = "1" if v else "0"

    from whisper_studio import pipeline, prefetch, store

    sources = [_source(x) for x in a.inputs]
    ids = []
    for src in sources:
        d = pipeline.submit(src, pipeline.make_options({}))
        ids.append(d["id"])
    log = lambda m: print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)  # noqa: E731
    t0 = time.monotonic()
    per_doc = {}
    if a.batch:
        pipeline.process_batch(ids, log=log)
    else:
        for i in ids:
            t = time.monotonic()
            pipeline.process(i)
            per_doc[i] = round(time.monotonic() - t, 1)
            log(f"{i} done in {per_doc[i]}s")
    total = time.monotonic() - t0

    res = dict(flags=flags, total_s=round(total, 1), per_doc_s=per_doc, prefetch=dict(prefetch.stats),
               early_voices=list(pipeline.Voices.ran), docs=[])
    for src, i in zip(sources, ids):
        d = store.load(i)
        hy = store.path(i) / "hypotheses.json"
        timing = json.loads(hy.read_text("utf-8")).get("timing") if hy.exists() else None
        res["docs"].append(dict(source=str(src), id=i, status=d["status"], error=d.get("error"),
                                duration_min=round((d.get("duration") or 0) / 60, 1), timing=timing,
                                speakers=len(d.get("speakers") or {}),
                                text=[(s.get("speaker"), s["text"]) for s in d["segments"]]))
    print(f"\ntotal {total:.0f}s ({total / 60:.1f} min) flags {flags}")
    for d in res["docs"]:
        print(f"  {d['id']} {d['duration_min']} min {d['status']} timing {d['timing']}")
    print(f"  prefetch read {res['prefetch']['bytes'] / 2**30:.1f} GB in {res['prefetch']['seconds']:.0f}s; "
          f"voices ahead: {len(res['early_voices'])}")
    if a.compare:
        other = json.loads(Path(a.compare).read_text("utf-8"))
        same = [x["text"] == y["text"] for x, y in zip(res["docs"], other["docs"])]
        print(f"  same text as {a.compare}: {same}; time {other['total_s']}s -> {res['total_s']}s "
              f"({100 * (res['total_s'] / other['total_s'] - 1):+.0f}%)")
        res["same_as_compare"] = same
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), "utf-8")


def _source(x: str) -> Path:
    p = Path(x)
    if p.is_file():
        return p
    lib = REAL_HOME / "library"
    hits = [d for d in lib.iterdir() if d.name.endswith(x)] if lib.exists() else []
    if len(hits) != 1:
        sys.exit(f"{x}: not a file, and {len(hits)} library documents end with it")
    doc = json.loads((hits[0] / "doc.json").read_text("utf-8"))
    src = Path(doc.get("source_path") or "")
    if src.is_file():
        return src
    if doc.get("audio_file") and (hits[0] / doc["audio_file"]).is_file():
        return hits[0] / doc["audio_file"]
    sys.exit(f"{x}: audio not found")


if __name__ == "__main__":
    main()
