"""Speaker diarization (pyannote segmentation 3.0 + WeSpeaker embeddings via sherpa-onnx)
and assignment of a speaker to every word."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE


def _threads() -> int:
    """All cores but one; WHISPER_STUDIO_DIAR_THREADS overrides (tests on a busy machine)."""
    env = os.environ.get("WHISPER_STUDIO_DIAR_THREADS")
    return int(env) if env else max(1, (os.cpu_count() or 4) - 1)


def diarize(audio: np.ndarray, model_dir: Path, num_speakers: int | None = None,
            threshold: float = 0.5, progress: Callable[[float, str], None] | None = None
            ) -> list[tuple[float, float, int]]:
    import sherpa_onnx

    threads = _threads()
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(model_dir / "segmentation.onnx")),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_dir / "embedding.onnx"), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers else -1, threshold=threshold),
        min_duration_on=0.3, min_duration_off=0.5,
    )
    if not cfg.validate():
        raise RuntimeError("Неверная конфигурация диаризации")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def cb(done: int, total: int) -> int:
        if progress and total:
            progress(done / total, "Определяю спикеров")
        return 0

    res = sd.process(audio.astype(np.float32), callback=cb).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in res]


# Clustering over-splits real conversations: a voice that changes with distance, a cough,
# overlapping speech or a TV become extra "speakers" with a few seconds each. Worse, asking
# sherpa for exactly N clusters keeps such an outlier and merges the two real voices instead.
# So we always cluster freely, then fold outliers and merge voices ourselves.
TINY_SHARE, TINY_SECONDS = 0.03, 5.0     # outliers: re-assigned turn by turn
MIN_SHARE, MIN_SECONDS = 0.06, 15.0      # auto mode: smaller clusters are not a real speaker
# Centroids this similar (cosine) are surely one person. On phone recordings different
# people still reach 0.85-0.9 (same channel and room), so only near-identical ones merge.
SAME_SPEAKER = 0.95


def _embedder(model_dir: Path):
    import sherpa_onnx

    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_dir / "embedding.onnx"), num_threads=_threads())
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def _embed(ex, audio: np.ndarray) -> np.ndarray:
    st = ex.create_stream()
    st.accept_waveform(SAMPLE_RATE, audio.astype(np.float32))
    st.input_finished()
    return _unit(np.asarray(ex.compute(st), dtype=np.float32))


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def turn_embeddings(audio: np.ndarray, turns: list[tuple[float, float, int]], model_dir: Path,
                    progress: Callable[[float, str], None] | None = None) -> dict[int, np.ndarray]:
    """One voice embedding per turn long enough to judge (its first 10 s); a cluster whose turns
    are all shorter gets one of their audio together under -1 - cluster (short_clusters)."""
    ex = _embedder(model_dir)
    out = {}
    for i, (s, e, _) in enumerate(turns):
        if e - s >= MIN_TURN:
            out[i] = _embed(ex, audio[int(s * SAMPLE_RATE):int(min(e, s + 10) * SAMPLE_RATE)])
        if progress and i % 10 == 0:
            progress(i / max(len(turns), 1), "Сравниваю голоса")
    for k, cuts in short_clusters(turns, len(audio)).items():
        out[-1 - k] = _embed(ex, np.concatenate([audio[a:b] for a, b in cuts]))
    return out


MIN_TURN = 0.6        # s: shorter turns get no voice embedding of their own
MIN_CLUSTER = 0.3     # s: ...but a cluster of only such turns gets one of them all, from this much


def short_clusters(turns: list[tuple[float, float, int]], n: int) -> dict[int, list[tuple[int, int]]]:
    """Raw clusters without a turn of MIN_TURN: the sample ranges of their turns (10 s at most)."""
    has = {k for s, e, k in turns if e - s >= MIN_TURN}
    out: dict[int, list[tuple[int, int]]] = {}
    for s, e, k in turns:
        if k in has:
            continue
        cuts = out.setdefault(k, [])
        left = 10 * SAMPLE_RATE - sum(b - a for a, b in cuts)
        a, b = int(s * SAMPLE_RATE), min(int(e * SAMPLE_RATE), n)
        if left > 0 and b > a:
            cuts.append((a, min(b, a + left)))
    return {k: c for k, c in out.items() if sum(b - a for a, b in c) >= MIN_CLUSTER * SAMPLE_RATE}


SPLIT_MAX = 0.9  # voices this similar (centroid cosine) are one person: no forced split


def _split_voices(turns: list[tuple[float, float, int]], temb: dict[int, np.ndarray],
                  target: int) -> list[tuple[float, float, int]] | None:
    """More speakers wanted than the clustering found (a quiet phone side often joins the near
    one): group the turns by their voice embeddings. None if the groups sound alike."""
    from scipy.cluster.hierarchy import fcluster, linkage

    idx = [i for i in range(len(turns)) if i in temb]
    if len(idx) < 2 * target:
        return None
    X = np.array([temb[i] for i in idx])
    lab = fcluster(linkage(X, "average", metric="cosine"), target, "maxclust")
    ks = sorted(set(lab))
    if len(ks) < 2:
        return None
    cents = [_unit(X[lab == k].mean(axis=0)) for k in ks]
    if max(float(a @ b) for n, a in enumerate(cents) for b in cents[n + 1:]) >= SPLIT_MAX:
        return None
    new = {i: 1000 + int(k) for i, k in zip(idx, lab)}
    out = []
    for i, (s, e, _k) in enumerate(turns):
        if i not in new:  # no embedding (too short): the label of the nearest turn that has one
            j = min(idx, key=lambda j: abs(turns[j][0] - s))
            new[i] = new[j]
        out.append((s, e, new[i]))
    return out


def refine(turns: list[tuple[float, float, int]], temb: dict[int, np.ndarray],
           target: int | None = None, legacy=()) -> tuple[list[tuple[float, float, int]], dict]:
    """Fold outliers into real speakers and merge clusters that are the same voice.
    target: exact speaker count wanted, None = decide automatically. temb: turn index -> voice;
    -1 - k: raw cluster k whose turns are all too short for one of their own.
    legacy "a": folded clusters turn by turn (before Опыт 11)."""
    dur: dict[int, float] = {}
    for s, e, k in turns:
        dur[k] = dur.get(k, 0.0) + e - s
    info = dict(raw_speakers=len(dur), mode="fixed" if target else "auto")
    if target and len(dur) < target:
        split = _split_voices(turns, temb, target)
        if split is not None:  # the clustering heard one voice, the voices themselves differ
            turns = split
            dur = {}
            for s, e, k in turns:
                dur[k] = dur.get(k, 0.0) + e - s
            info["split"] = len(dur)
    total = sum(dur.values()) or 1.0

    def centroid(ks):
        # duration-weighted mean of the turn embeddings
        vs = [temb[i] * (e - s) for i, (s, e, k) in enumerate(turns) if k in ks and i in temb]
        return _unit(np.sum(vs, axis=0)) if vs else None

    groups = {k: {k} for k in dur}   # surviving cluster -> raw clusters merged into it
    gdur = dict(dur)

    def fold(small: set[int]):
        for k in small:
            groups.pop(k)
            gdur.pop(k)

    # 1) outliers stop being clusters; their turns are re-assigned at the end
    tiny = {k for k in dur if dur[k] < max(TINY_SHARE * total, TINY_SECONDS)}
    if len(tiny) < len(dur):
        fold(tiny)
    cents = {k: centroid(g) for k, g in groups.items()}

    # 2) merge the most similar pair: down to the target, or while it is clearly one voice
    while len(groups) > (target or 1):
        keys = [k for k in groups if cents[k] is not None]
        pairs = [(float(cents[a] @ cents[b]), a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
        if not pairs:
            break
        best, a, b = max(pairs)
        if not target and best < SAME_SPEAKER:
            break
        keep, drop = (a, b) if gdur[a] >= gdur[b] else (b, a)
        groups[keep] |= groups.pop(drop)
        gdur[keep] += gdur.pop(drop)
        cents.pop(drop)
        cents[keep] = centroid(groups[keep])
    # 3) auto mode: whatever is still small is not a real participant
    if not target:
        small = {k for k in groups if gdur[k] < max(MIN_SHARE * total, MIN_SECONDS)}
        if len(small) < len(groups):
            fold(small)
    info["merged"] = info.get("split", info["raw_speakers"]) - len(groups)

    owner = {raw: k for k, g in groups.items() for raw in g}
    voices = [k for k in groups if cents.get(k) is not None]
    if "a" not in legacy and voices:
        # a folded cluster goes whole to the closest voice: one person's short replies («понял»,
        # «окей») often form such a cluster; turn by turn, the short ones (no embedding of their
        # own) went to the time neighbour - the other person (Опыт 11)
        folded = 0
        for k in dur:
            if k not in owner:
                c = centroid({k})
                if c is None:
                    c = temb.get(-1 - k)  # the cluster's turns are all short: their audio together
                if c is not None:
                    owner[k] = max((float(c @ cents[b]), b) for b in voices)[1]
                    folded += 1
        info["folded_whole"] = folded
    out = []
    for i, (s, e, k) in enumerate(turns):
        k2 = owner.get(k)
        if k2 is None:  # folded: the closest voice, else the speaker of the nearest kept turn
            if i in temb and voices:
                k2 = max((float(temb[i] @ cents[b]), b) for b in voices)[1]
            else:
                k2 = min((min(abs(s - e2), abs(s2 - e)), owner[kk])
                         for s2, e2, kk in turns if kk in owner)[1]
        out.append((s, e, k2))
    info["speakers"] = len({k for _, _, k in out})
    return out, info


def gpu_backend() -> bool:
    """settings speed.diarize_gpu: voices on onnxruntime, embeddings on the GPU (diarize_ort)."""
    from . import settings

    return bool(settings.speed().get("diarize_gpu"))


def voices(audio: np.ndarray, model_dir: Path, progress: Callable[[float, str], None] | None = None,
           sherpa: bool = False, extra: dict | None = None):
    """The heavy part (CPU, ~1x real time): raw turns and one voice embedding per turn.
    Depends on the audio only, not on the text: it can run before recognition (pipeline.Voices).
    extra (onnxruntime backend only) gets the per-frame speaker activity, see Activity."""
    if not sherpa and gpu_backend():
        from . import diarize_ort

        try:
            return diarize_ort.voices(audio, model_dir, progress, extra=extra)
        except Exception as e:  # noqa: BLE001 - e.g. out of video memory: sherpa as before
            import sys

            if extra is not None:
                extra.clear()
            print(f"diarize_ort failed ({type(e).__name__}: {e}), falling back to sherpa-onnx",
                  file=sys.stderr, flush=True)

    def sub(lo, hi):
        return (lambda f, _m: progress(lo + (hi - lo) * f, "Определяю спикеров")) if progress else None

    turns = diarize(audio, model_dir, None, progress=sub(0.0, 0.75))
    temb = turn_embeddings(audio, turns, model_dir, progress=sub(0.75, 1.0))
    return turns, temb


def save_cache_atomic(path: Path, turns, temb, extra: dict | None = None) -> None:
    tmp = path.with_name(path.stem + ".part.npz")  # savez adds ".npz" to other names
    _save_cache(tmp, turns, temb, extra)
    tmp.replace(path)


ACTIVITY_KEYS = ("act", "act_t0", "act_dt")


def _save_cache(path: Path, turns, temb, extra: dict | None = None):
    idx = sorted(temb)
    more = {k: np.asarray(extra[k]) for k in ACTIVITY_KEYS if extra and k in extra}
    np.savez_compressed(path, turns=np.array(turns, dtype=np.float64).reshape(-1, 3),
                        idx=np.array(idx, dtype=np.int64),
                        emb=np.array([temb[i] for i in idx], dtype=np.float32).reshape(len(idx), -1),
                        **more)


def _load_cache(path: Path, extra: dict | None = None):
    z = np.load(path)
    turns = [(float(s), float(e), int(k)) for s, e, k in z["turns"]]
    if extra is not None and all(k in z.files for k in ACTIVITY_KEYS):  # older caches: none
        extra.update({k: z[k] for k in ACTIVITY_KEYS})
    return turns, {int(i): v for i, v in zip(z["idx"], z["emb"])}


def run(segments: list[dict], audio: np.ndarray, model_dir: Path, num_speakers: int | None,
        progress: Callable[[float, str], None] | None = None, cache: Path | None = None,
        ) -> tuple[list[dict], dict]:
    """Diarize, give every word a speaker and split segments at speaker changes.
    The raw clustering and voice embeddings are cached, so re-diarization is instant."""
    backend = None
    extra: dict = {}
    if cache is not None and cache.exists():
        turns, temb = _load_cache(cache, extra)
    else:
        import time

        t0 = time.monotonic()
        turns, temb = voices(audio, model_dir, progress, extra=extra)
        backend = dict(seconds=round(time.monotonic() - t0, 1), backend="sherpa")
        if gpu_backend():
            from . import diarize_ort

            if diarize_ort.stats.get("total") is not None:
                backend.update(backend="ort", device=diarize_ort.stats.get("device"),
                               engine=diarize_ort.stats.get("engine"))
        if cache is not None:
            try:
                _save_cache(cache, turns, temb, extra)
            except Exception:  # noqa: BLE001 - the cache is only an optimisation
                pass
    out, info = label(segments, turns, temb, num_speakers, extra)
    if backend:
        info["voices"] = backend
    return out, info


def label(segments: list[dict], turns: list[tuple[float, float, int]], temb: dict[int, np.ndarray],
          num_speakers: int | None, extra: dict | None = None, legacy=(), profile=None
          ) -> tuple[list[dict], dict]:
    """run() after the voices: refine, words, pieces, speakers numbered in order of appearance
    (as store.finalize numbers them) and the owner's voice recognised (info["owner"]).
    legacy: changes of Опыт 11 switched off ("a" refine, "b" words outside turns and in
    overlaps, "t" only in overlaps); profile: the owner's
    voice (default owner_profile())."""
    final, info = refine(turns, temb, num_speakers, legacy)
    act = None
    if "b" not in legacy and extra and "act" in extra:
        act = Activity(extra, raw_to_final(turns, final))
    assign_words(segments, final, act, overlap="t" not in legacy)
    if act is not None:
        info["by_activity"] = dict(outside=act.used, overlap=act.ties)
    out = split_by_speaker(segments)
    order: dict[str, str] = {}
    for s in out:
        if s.get("speaker") and s["speaker"] not in order:
            order[s["speaker"]] = f"S{len(order) + 1}"
    for s in out:
        if s.get("speaker"):
            s["speaker"] = order[s["speaker"]]
    prof = owner_profile() if profile is None else profile
    if prof is not None:
        cents = {order[f"S{k + 1}"]: v for k, v in final_centroids(final, temb).items()
                 if f"S{k + 1}" in order}
        who = owner_speaker(cents, prof, says_name(out, prof.get("name") or OWNER_NAME))
        if who:
            first = next(i for i, s in enumerate(out) if s.get("speaker") == who[0])
            info["owner"] = dict(speaker=who[0], cos=round(who[1], 3), first=first)
    return out, info


def says_name(segments: list[dict], name: str) -> dict:
    """Speaker -> how many times their words contain the name («Миша», «Миш»): people call the
    owner by it, the owner does not."""
    if not name:
        return {}
    stem = name.lower()[:max(3, len(name) - 1)]
    out: dict = {}
    for s in segments:
        n = sum(re.sub(r"[^\w]", "", w["w"].lower()).startswith(stem) for w in s.get("words") or [])
        if n and s.get("speaker"):
            out[s["speaker"]] = out.get(s["speaker"], 0) + n
    return out


def raw_to_final(raw: list[tuple[float, float, int]], final: list[tuple[float, float, int]]
                 ) -> dict[int, int]:
    """Raw cluster -> the refined speaker that most of its time went to (refine keeps the turns)."""
    t: dict[int, dict[int, float]] = {}
    for (s, e, k), (_s, _e, k2) in zip(raw, final):
        t.setdefault(k, {})
        t[k][k2] = t[k].get(k2, 0.0) + e - s
    return {k: max(v, key=v.get) for k, v in t.items()}


ACT_PAD = 0.1   # s: activity looked at around a word outside every turn
ACT_MIN = 0.15  # mean activity of the best speaker there; below it nobody talks: nearest turn
OVERLAP_TIE = 0.5  # a word in two speakers' turns, the second covering it >= half as much: activity


class Activity:
    """Per-frame speaker activity of the segmentation (diarize_ort: the soft scores of every
    window's local speakers, averaged per raw cluster, uint8 0..255) for the words that fall in
    no turn: turns shorter than 0.3 s are dropped and pauses < 0.5 s glued, so ~7% of the words
    are outside every turn and used to take the nearest one (Опыт 11)."""

    def __init__(self, extra: dict, rawmap: dict[int, int]):
        self.act = np.asarray(extra["act"])
        self.t0, self.dt = float(extra["act_t0"]), float(extra["act_dt"])
        self.rawmap = rawmap
        self.used = self.ties = 0

    def speaker(self, s: float, e: float, among: set[int] | None = None, tie: bool = False) -> int | None:
        """The speaker most active over [s, e] (among these), None if nobody is."""
        a = max(0, int(np.floor((s - ACT_PAD - self.t0) / self.dt)))
        b = min(len(self.act), int(np.ceil((e + ACT_PAD - self.t0) / self.dt)) + 1)
        if b <= a:
            return None
        m = self.act[a:b].astype(np.float32).mean(axis=0) / 255.0
        per: dict[int, float] = {}
        for k, v in enumerate(m):
            if k in self.rawmap and (among is None or self.rawmap[k] in among):
                per[self.rawmap[k]] = per.get(self.rawmap[k], 0.0) + float(v)
        if not per:
            return None
        best = max(per, key=per.get)
        if per[best] < ACT_MIN:
            return None
        if tie:
            self.ties += 1
        else:
            self.used += 1
        return best


# ---------------------------------------------------------------- the owner's voice

OWNER_NAME = ""  # the profile keeps the name (tools/diarize_eval.py owner --name)
OWNER_LABEL = "Я"  # the owner's speaker name when the profile has none
OWNER_MIN = 0.8      # cosine of a final speaker's voice to the owner's profile (Опыт 11)
OWNER_MARGIN = 0.05  # ...and this much closer than the next speaker
OWNER_SOLO = 0.9     # the only speaker of a recording


def owner_path() -> Path:
    from .config import HOME

    return HOME / "voices" / "owner.npz"


def owner_profile(path: Path | None = None) -> dict | None:
    """{"emb": unit vector, "name": str} or None (no profile made: tools/diarize_eval.py owner)."""
    p = path or owner_path()
    try:
        z = np.load(p)
        name = str(z["name"]) if "name" in z.files else OWNER_NAME
        return dict(emb=_unit(np.asarray(z["emb"], np.float32)), name=name)
    except (OSError, KeyError, ValueError):
        return None


def save_owner_profile(path: Path, emb: np.ndarray, name: str = OWNER_NAME, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part.npz")
    np.savez(tmp, emb=_unit(np.asarray(emb, np.float32)), name=np.array(name),
             meta=np.array(json.dumps(meta, ensure_ascii=False)))
    tmp.replace(path)


def final_centroids(turns: list[tuple[float, float, int]], temb: dict[int, np.ndarray]
                    ) -> dict[int, np.ndarray]:
    """Speaker -> duration-weighted mean voice of its turns (turns as refine returns them)."""
    acc: dict[int, np.ndarray] = {}
    for i, (s, e, k) in enumerate(turns):
        if i in temb:
            acc[k] = acc.get(k, 0) + temb[i] * (e - s)
    return {k: _unit(v) for k, v in acc.items()}


def owner_speaker(cents: dict, profile: dict, said: dict | None = None) -> tuple[object, float] | None:
    """The speaker whose voice is the owner's: (speaker, cosine) or None if none is clearly the owner.
    On one channel other people still reach 0.82-0.87 to the profile, the owner's own speakers 0.82-0.93:
    the closest must also stand out by OWNER_MARGIN; alone in a recording (a voice message)
    OWNER_SOLO. said (says_name): a speaker who calls the owner by name twice and more than
    anybody else is not the owner."""
    sims = sorted(((float(v @ profile["emb"]), k) for k, v in cents.items()), key=lambda x: -x[0])
    if not sims or sims[0][0] < (OWNER_MIN if len(sims) > 1 else OWNER_SOLO):
        return None
    if len(sims) > 1 and sims[0][0] - sims[1][0] < OWNER_MARGIN:
        return None
    said = said or {}
    top = said.get(sims[0][1], 0)
    if top >= 2 and top > max((v for k, v in said.items() if k != sims[0][1]), default=0):
        return None
    return sims[0][1], sims[0][0]


def name_owner(doc: dict, info: dict) -> str | None:
    """After store.finalize (pipeline): give the owner's speaker the owner's name, unless the owner named
    that speaker themselves. info["owner"]["first"] is a segment index (finalize keeps the list)."""
    own = info.get("owner")
    segs = doc.get("segments") or []
    if not own or not 0 <= own["first"] < len(segs):
        return None
    sid = segs[own["first"]].get("speaker")
    spk = doc.get("speakers") or {}
    if sid not in spk:
        return None
    name = (owner_profile() or {}).get("name") or OWNER_LABEL
    if not re.fullmatch(r"Спикер \d+", spk[sid].get("name", "")) and spk[sid].get("name") != name:
        return None  # a name the owner gave
    for k, v in spk.items():  # the name moved to another voice (rediarize kept old names)
        if k != sid and v.get("name") == name:
            v["name"] = f"Спикер {k[1:]}" if k[1:].isdigit() else k
    spk[sid]["name"] = name
    return sid


def rejoin(segments: list[dict]) -> list[dict]:
    """Undo speaker splits: glue consecutive pieces of one Whisper segment back together."""
    out: list[dict] = []
    for s in segments:
        p = out[-1] if out else None
        if p is not None:
            if "src" in s or "src" in p:
                same = s.get("src") is not None and s.get("src") == p.get("src")
            else:  # older documents: a piece that ends mid-sentence continues in the next one
                same = (s["start"] - p["end"] <= 0.3 and s["end"] - p["start"] <= 30
                        and not p["text"].rstrip().endswith((".", "?", "!", "…")))
            if same:
                _glue_marks(p, s)
                p["words"] = (p.get("words") or []) + (s.get("words") or [])
                p["text"] = (p["text"] + " " + s["text"]).strip()
                p["end"] = s["end"]
                p["flags"] = sorted(set(p.get("flags") or []) | set(s.get("flags") or []))
                for k in ("edited", "ai"):
                    if s.get(k):
                        p[k] = True
                continue
        out.append({**s, "words": list(s.get("words") or [])})
    return out


def _glue_marks(p: dict, s: dict) -> None:
    """Edit marks of p when s is glued onto it (before the texts are joined): the text the models
    heard joins too, and the whole counts as the owner's only when every piece was."""
    from . import store

    if "orig_text" in p or "orig_text" in s:
        p["orig_text"] = (p.get("orig_text", p["text"]) + " " + s.get("orig_text", s["text"])).strip()
    if not (p.get("edited") or s.get("edited")):
        return
    if store.by_owner(p) and store.by_owner(s):
        p["by"] = "owner"
        return
    other = [x.get("by") for x in (s, p) if x.get("edited") and not store.by_owner(x)]
    # an unedited piece is the models' text: nobody confirmed it
    p["by"] = (other[0] if other and other[0] else None) or (
        "ai" if p.get("ai") or s.get("ai") else "model")


def assign_words(segments: list[dict], turns: list[tuple[float, float, int]],
                 activity: Activity | None = None, overlap: bool = True) -> None:
    """Give each word the speaker with the largest time overlap; outside every turn the one the
    segmentation hears there (activity), else the nearest turn."""
    if not turns:
        return
    starts = np.array([t[0] for t in turns])
    ends = np.array([t[1] for t in turns])
    spk = np.array([t[2] for t in turns])
    for seg in segments:
        for w in seg.get("words") or []:
            ov = np.minimum(ends, w["e"]) - np.maximum(starts, w["s"])
            if ov.max() > 0:
                # sum overlap per speaker
                per = {int(s): float(ov[(spk == s) & (ov > 0)].sum()) for s in np.unique(spk[ov > 0])}
                best = max(per, key=per.get)  # the first of equal ones, as before
                second = sorted(per.values())[-2] if len(per) > 1 else 0.0
                if overlap and activity is not None and second >= OVERLAP_TIE * per[best] > 0:
                    # inside overlapping turns of two speakers: the one the segmentation hears more
                    k = activity.speaker(w["s"], w["e"], among=set(per), tie=True)
                    best = best if k is None else k
                w["spk"] = best
            else:
                k = activity.speaker(w["s"], w["e"]) if activity is not None else None
                if k is None:
                    mid = (w["s"] + w["e"]) / 2
                    d = np.minimum(np.abs(starts - mid), np.abs(ends - mid))
                    k = int(spk[d.argmin()])
                w["spk"] = k
    # smooth single-word flips inside one speaker's run
    for seg in segments:
        ws = seg.get("words") or []
        for i in range(1, len(ws) - 1):
            a, b, c = ws[i - 1].get("spk"), ws[i].get("spk"), ws[i + 1].get("spk")
            if a == c and b != a and ws[i]["e"] - ws[i]["s"] < 0.6:
                ws[i]["spk"] = a


def split_by_speaker(segments: list[dict]) -> list[dict]:
    """Split Whisper segments wherever the word-level speaker changes."""
    out = []
    for seg in segments:
        ws = seg.get("words") or []
        if not ws:
            out.append({**seg, "speaker": seg.get("speaker")})
            continue
        run = [ws[0]]
        for w in ws[1:]:
            if w.get("spk") == run[-1].get("spk"):
                run.append(w)
            else:
                out.append(_piece(seg, run))
                run = [w]
        out.append(_piece(seg, run))
    return out


def _piece(seg: dict, words: list[dict]) -> dict:
    sp = words[0].get("spk")
    return {
        **seg,
        "start": words[0]["s"], "end": words[-1]["e"],
        "text": " ".join(w["w"] for w in words),
        "words": [{k: v for k, v in w.items() if k != "spk"} for w in words],
        "speaker": f"S{sp + 1}" if sp is not None else seg.get("speaker"),
    }


def precompute_doc(doc_id: str, model_dir: str) -> None:
    """`python -m whisper_studio voices <id> <model dir>`: voices.npz of a queued document, in a
    CPU process of its own while the worker recognizes (pipeline.Voices). sherpa-onnx holds the
    GIL for the whole computation, so in a thread of the worker it would stall the GPU work.
    The audio is decoded exactly like pipeline._prepare does it, so the numbers are the same."""
    import time

    from . import audio as A
    from . import store

    doc = store.load(doc_id)
    folder = store.path(doc_id)
    cache = folder / "voices.npz"
    if cache.exists():
        return
    x = A.load(Path(doc["source_path"]))
    if (doc.get("options") or {}).get("highpass"):
        x = A.highpass(x)
    prog = folder / "voices.progress"
    last = [0.0]

    def progress(f: float, _msg: str) -> None:
        if time.monotonic() - last[0] > 2.0:
            last[0] = time.monotonic()
            try:
                prog.write_text(json.dumps(dict(f=round(f, 3))), "utf-8")
            except OSError:
                pass

    extra: dict = {}
    turns, temb = voices(x, Path(model_dir), progress, extra=extra)
    save_cache_atomic(cache, turns, temb, extra)
    prog.unlink(missing_ok=True)
