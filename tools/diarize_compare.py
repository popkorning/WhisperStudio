"""Speakers from sherpa-onnx vs the onnxruntime backend (diarize_ort, lab H12) on library
documents (read only): the same words, speaker assignment after refine, labels mapped.

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-agent-diar
    ws-python tools/diarize_compare.py a1b2c3 d4e5f6 --out %WS_REAL_HOME%\\tmp\\agent-diar\\cmp
        [--sherpa]          compute sherpa voices anew (timed) instead of the library voices.npz
        [--device cpu|cuda] onnxruntime device (WHISPER_STUDIO_DIAR_DEVICE)
        [--tag gpu]         name of this run's files in --out

Prints per document: time, raw clusters, word agreement with sherpa and with the document's
current speakers (the owner may have fixed some), raw-turn agreement (10 ms frames).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whisper_studio import audio as A  # noqa: E402
from whisper_studio import diarize as D  # noqa: E402
from whisper_studio.config import SAMPLE_RATE  # noqa: E402

LIBRARY = Path(os.environ.get("WS_LIBRARY", str(REAL_HOME / "library")))
MODELS = Path(os.environ.get("WHISPER_STUDIO_MODELS", str(REAL_HOME / "models"))) / "diarization"


def find(doc_id: str) -> Path:
    hits = [p for p in LIBRARY.iterdir() if p.name.endswith(doc_id)]
    if len(hits) != 1:
        raise SystemExit(f"{doc_id}: {len(hits)} documents")
    return hits[0]


def load_audio(folder: Path, doc: dict) -> np.ndarray:
    src = Path(doc.get("source_path") or "")
    if not src.is_file():
        src = folder / doc["audio_file"]
    x = A.load(src)
    if (doc.get("options") or {}).get("highpass"):
        x = A.highpass(x)
    return x


def words_of(segs: list[dict]) -> list[str | None]:
    out = []
    for s in segs:
        out.extend([s.get("speaker")] * len(s.get("words") or []))
    return out


def assign(doc: dict, turns, temb, n) -> tuple[list, dict]:
    segs = D.rejoin(copy.deepcopy(doc["segments"]))
    turns2, info = D.refine(turns, temb, n)
    D.assign_words(segs, turns2)
    return words_of(D.split_by_speaker(segs)), info


def agreement(a: list, b: list, w: list[float] | None = None) -> float:
    """Share of items with the same label after the best one-to-one label mapping."""
    from scipy.optimize import linear_sum_assignment

    la = sorted({x for x in a if x is not None}, key=str)
    lb = sorted({x for x in b if x is not None}, key=str)
    if not la or not lb:
        return float("nan")
    w = w or [1.0] * len(a)
    m = np.zeros((len(la), len(lb)))
    tot = 0.0
    for x, y, ww in zip(a, b, w):
        if x is None or y is None:
            continue
        m[la.index(x), lb.index(y)] += ww
        tot += ww
    r, c = linear_sum_assignment(-m)
    return float(m[r, c].sum() / tot) if tot else float("nan")


def frames(turns, dur: float) -> list:
    lab = np.full(int(dur * 100) + 1, -1)
    multi = np.zeros(len(lab), bool)
    for s, e, k in turns:
        a, b = int(s * 100), int(e * 100)
        multi[a:b] |= lab[a:b] >= 0
        lab[a:b] = k
    lab[multi] = -2
    return [int(v) if v >= 0 else None for v in lab]


class VramPeak:
    """Peak of nvidia-smi memory.used above what was used before (MiB), None without a GPU."""

    def __init__(self):
        import threading

        self.base = self._used()
        self.peak = self.base
        self.done = False
        if self.base is not None:
            threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _used():
        import subprocess

        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            return int(r.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def _run(self):
        while not self.done:
            v = self._used()
            if v is not None:
                self.peak = max(self.peak, v)
            time.sleep(0.5)

    def stop(self):
        self.done = True
        return None if self.base is None else self.peak - self.base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sherpa", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tag", default="ort")
    ap.add_argument("--reuse", action="store_true", help="read this tag's voices from --out if present")
    a = ap.parse_args()
    if a.device:
        os.environ["WHISPER_STUDIO_DIAR_DEVICE"] = a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from whisper_studio import diarize_ort as O

    rows = []
    for doc_id in a.docs:
        folder = find(doc_id)
        doc = json.loads((folder / "doc.json").read_text("utf-8"))
        n = (doc.get("options") or {}).get("num_speakers")
        x = load_audio(folder, doc)
        dur = len(x) / SAMPLE_RATE
        row = dict(doc=doc_id, minutes=round(dur / 60, 1))

        ref_path = out / f"{doc_id}-sherpa.npz"
        if a.sherpa and not ref_path.exists():
            t = time.monotonic()
            st, se = D.voices(x, MODELS, sherpa=True)
            row["sherpa_s"] = round(time.monotonic() - t, 1)
            D.save_cache_atomic(ref_path, st, se)
            (out / f"{doc_id}-sherpa.json").write_text(json.dumps(dict(seconds=row["sherpa_s"])), "utf-8")
        elif a.sherpa:
            row["sherpa_s"] = json.loads((out / f"{doc_id}-sherpa.json").read_text("utf-8"))["seconds"]
        src = ref_path if ref_path.exists() else folder / "voices.npz"
        st, se = D._load_cache(src)  # read only
        row["ref"] = "fresh" if src == ref_path else "library"

        new_path = out / f"{doc_id}-{a.tag}.npz"
        if a.reuse and new_path.exists():
            nt, ne = D._load_cache(new_path)
            row.update(json.loads((out / f"{doc_id}-{a.tag}.json").read_text("utf-8")))
        else:
            vram = VramPeak()
            t = time.monotonic()
            nt, ne = O.voices(x, MODELS)
            row["new_s"] = round(time.monotonic() - t, 1)
            row["stats"] = dict(O.stats, vram_mib=vram.stop())
            D.save_cache_atomic(new_path, nt, ne)
            (out / f"{doc_id}-{a.tag}.json").write_text(json.dumps(dict(new_s=row["new_s"], stats=row["stats"])),
                                                        "utf-8")

        row["raw_ref"] = len({k for *_, k in st})
        row["raw_new"] = len({k for *_, k in nt})
        row["turns_ref"], row["turns_new"] = len(st), len(nt)
        row["raw_frames_agree"] = round(100 * agreement(frames(st, dur), frames(nt, dur)), 2)
        wr, ir = assign(doc, st, se, n)
        wn, inn = assign(doc, nt, ne, n)
        cur = words_of(doc["segments"])
        dw = [w["e"] - w["s"] for s in D.rejoin(copy.deepcopy(doc["segments"])) for w in s.get("words") or []]
        row["words"] = len(wr)
        row["words_agree"] = round(100 * agreement(wr, wn), 2)
        row["time_agree"] = round(100 * agreement(wr, wn, dw), 2)
        row["ref_vs_doc"] = round(100 * agreement(wr, cur), 2)
        row["new_vs_doc"] = round(100 * agreement(wn, cur), 2)
        row["info_ref"], row["info_new"] = ir, inn
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (out / f"summary-{a.tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
