"""Model registry: what to download, how to convert, where it lives.

Choices follow the accuracy guide:
  * RU  - antony66/whisper-large-v3-russian (fine-tuned large-v3), converted to CTranslate2 fp16
          coriollon/whisper-large-v3-turbo-russian as a faster alternative
  * DE  - primeline/whisper-large-v3-turbo-german, converted to CTranslate2 fp16
  * any - large-v3 / large-v3-turbo
  * wav2vec2 CTC models for precise word timestamps (the WhisperX approach)
  * pyannote segmentation 3.0 + WeSpeaker ResNet293 embeddings (sherpa-onnx) for speakers
  * quality "max" (ensemble.py) adds GigaAM v3 and T-one (Russian), Canary 1B v2 (German
    hypothesis) through onnx-asr on the CPU, and Qwen3-ASR 1.7B through transformers on the GPU.
    Parakeet TDT v3 was tried and dropped: on these recordings it answers in English.
"""
from __future__ import annotations

import shutil
import tarfile
import threading
import urllib.request
from pathlib import Path
from typing import Callable

from .config import MODELS_DIR

Progress = Callable[[float, str], None]

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

MODELS: dict[str, dict] = {
    "ru-large-v3": dict(
        title="Русский · large-v3 (antony66)",
        kind="whisper", langs=["ru"], size_gb=3.1,
        repo="antony66/whisper-large-v3-russian", convert=True,
        note="Дообученная large-v3 под русский. Максимальная точность.",
    ),
    "ru-turbo": dict(
        title="Русский · large-v3-turbo (coriollon)",
        kind="whisper", langs=["ru"], size_gb=1.6,
        repo="coriollon/whisper-large-v3-turbo-russian", subfolder="ct2-int16",
        note="Быстрее в ~3 раза, на шумных записях часто не хуже.",
    ),
    "de-turbo": dict(
        title="Deutsch · large-v3-turbo (primeline)",
        kind="whisper", langs=["de"], size_gb=1.6,
        repo="primeline/whisper-large-v3-turbo-german", convert=True,
        note="Лучшая готовая модель для немецкого (WER 2.6%).",
    ),
    "large-v3": dict(
        title="Все языки · large-v3",
        kind="whisper", langs=["*"], size_gb=3.1,
        repo="Systran/faster-whisper-large-v3",
        note="Базовая large-v3. Лучше всего на чистом аудио.",
    ),
    "large-v3-turbo": dict(
        title="Все языки · large-v3-turbo",
        kind="whisper", langs=["*"], size_gb=1.6,
        repo="deepdml/faster-whisper-large-v3-turbo-ct2",
        note="Быстрая, меньше галлюцинирует на паузах и шуме.",
    ),
    # ---- other recognizers: members of the "max" ensemble (see ensemble.py, asr/*)
    "gigaam-v3": dict(
        title="Русский · GigaAM v3 (Сбер)",
        kind="asr", engine="onnx-asr", onnx="gigaam-v3-e2e-rnnt", langs=["ru"], size_gb=0.9,
        repo="istupakov/gigaam-v3-onnx",
        patterns=["config.json", "v3_e2e_rnnt_encoder.onnx", "v3_e2e_rnnt_decoder.onnx", "v3_e2e_rnnt_joint.onnx",
                  "v3_e2e_rnnt_vocab.txt", "v3_e2e_rnnt.yaml"],
        files=["v3_e2e_rnnt_encoder.onnx", "v3_e2e_rnnt_vocab.txt"],
        note="Лучшая открытая модель для русского, обучена в том числе на звонках. Для режима «Максимум».",
    ),
    "t-one": dict(
        title="Русский · T-one (Т-Банк, телефония)",
        kind="asr", engine="onnx-asr", onnx="t-one-ctc", langs=["ru"], size_gb=0.15,
        repo="t-tech/T-one", patterns=["config.json", "model.onnx"], files=["model.onnx"],
        note="Обучена на телефонных разговорах 8 кГц. Для режима «Максимум».",
    ),
    "canary-v2": dict(
        title="25 языков · Canary 1B v2 (NVIDIA)",
        kind="asr", engine="onnx-asr", onnx="nemo-conformer-aed", quant="int8", langs=["*"], size_gb=1.05,
        repo="istupakov/canary-1b-v2-onnx", patterns=["config.json", "vocab.txt", "*.int8.onnx"],
        files=["encoder-model.int8.onnx", "decoder-model.int8.onnx", "vocab.txt"],
        note="Распознаёт с заданным языком: в режиме «Максимум» — немецкая гипотеза для немецких кусков.",
    ),
    "qwen3-asr": dict(
        title="Много языков · Qwen3-ASR 1.7B",
        kind="asr", engine="qwen", langs=["*"], size_gb=4.1,
        repo="Qwen/Qwen3-ASR-1.7B-hf", files=["model.safetensors", "config.json", "tokenizer.json"],
        note="Сильная LLM-модель распознавания (GPU, ~4,5 ГБ видеопамяти). Для режима «Максимум».",
    ),
    "align-ru": dict(
        title="Таймкоды слов · русский (wav2vec2)",
        kind="align", langs=["ru"], size_gb=1.3,
        repo="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        patterns=["*.json", "pytorch_model.bin"],
    ),
    "align-de": dict(
        title="Таймкоды слов · немецкий (wav2vec2)",
        kind="align", langs=["de"], size_gb=1.3,
        repo="jonatasgrosman/wav2vec2-large-xlsr-53-german",
        patterns=["*.json", "pytorch_model.bin"],
    ),
    "align-en": dict(
        title="Таймкоды слов · английский (wav2vec2)",
        kind="align", langs=["en"], size_gb=1.3,
        repo="jonatasgrosman/wav2vec2-large-xlsr-53-english",
        patterns=["*.json", "pytorch_model.bin"],
    ),
    "diarization": dict(
        title="Спикеры (pyannote 3.0 + WeSpeaker)",
        kind="diar", langs=["*"], size_gb=0.12,
        urls=[
            f"{SHERPA}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
            f"{SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet293_LM.onnx",
        ],
    ),
}

# Short names for lists and the variants of the "max" ensemble (alts carry member names)
SHORT_NAMES = {
    "ensemble": "Ансамбль", "large-v3": "Whisper", "large-v3-turbo": "Whisper Turbo",
    "ru-large-v3": "Whisper RU", "ru-turbo": "Whisper RU Turbo", "de-turbo": "Whisper DE",
    "de-check": "Whisper DE", "gigaam-v3": "GigaAM", "t-one": "T-one", "canary-v2": "Canary",
    "qwen3-asr": "Qwen3",
}


def short_name(key: str | None) -> str:
    key = key or ""
    return SHORT_NAMES.get(key, key)


# Best Whisper model per language ("auto" in the UI)
AUTO_WHISPER = {"ru": "ru-large-v3", "de": "de-turbo"}
FALLBACK_WHISPER = ["large-v3", "large-v3-turbo"]

# Models that are already on disk elsewhere (reuse, don't copy)
_EXISTING = {
    "large-v3": [
        Path.home() / ".cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots",
    ],
}

_install_lock = threading.Lock()
_installing: dict[str, dict] = {}


def model_dir(key: str) -> Path:
    return MODELS_DIR / key


def _is_complete(p: Path, kind: str, files: list[str] | None = None) -> bool:
    if not p.exists():
        return False
    if kind == "whisper":
        return (p / "model.bin").exists() and (p / "tokenizer.json").exists()
    if kind == "align":
        return (p / "pytorch_model.bin").exists() and (p / "vocab.json").exists()
    if kind == "diar":
        return (p / "segmentation.onnx").exists() and (p / "embedding.onnx").exists()
    if kind == "asr":
        return bool(files) and all((p / f).exists() for f in files)
    return False


def find(key: str) -> Path | None:
    """Return the local folder of an installed model, or None."""
    m = MODELS[key]
    p = model_dir(key)
    if _is_complete(p, m["kind"], m.get("files")):
        return p
    for base in _EXISTING.get(key, []):
        if base.exists():
            for snap in sorted(base.iterdir(), reverse=True):
                if _is_complete(snap, m["kind"]):
                    return snap
    return None


def status() -> list[dict]:
    out = []
    for key, m in MODELS.items():
        p = find(key)
        st = _installing.get(key)
        out.append(dict(
            key=key, title=m["title"], short=short_name(key), kind=m["kind"], langs=m["langs"],
            size_gb=m["size_gb"], note=m.get("note", ""),
            installed=p is not None, path=str(p) if p else None,
            installing=bool(st and st.get("running")),
            progress=st.get("progress", 0) if st else 0,
            message=st.get("message", "") if st else "",
            error=st.get("error") if st else None,
        ))
    return out


def whisper_for(language: str | None, choice: str = "auto") -> str | None:
    """Pick the installed Whisper model key for a language."""
    if choice and choice != "auto":
        return choice if find(choice) else None
    if language and language in AUTO_WHISPER and find(AUTO_WHISPER[language]):
        return AUTO_WHISPER[language]
    for k in FALLBACK_WHISPER:
        if find(k):
            return k
    for k, m in MODELS.items():
        if m["kind"] == "whisper" and find(k):
            return k
    return None


def aligner_for(language: str | None) -> Path | None:
    if not language:
        return None
    key = f"align-{language}"
    return find(key) if key in MODELS else None


# ---------------------------------------------------------------- install

def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def _watch(folder: Path, expected: int, report: Progress, msg: str, stop: threading.Event,
           lo: float = 0.0, hi: float = 1.0):
    while not stop.wait(0.7):
        frac = min(_dir_size(folder) / max(expected, 1), 1.0)
        report(lo + (hi - lo) * frac, msg)


def _hf_download(repo: str, dest: Path, patterns: list[str] | None, expected_gb: float,
                 report: Progress, lo=0.0, hi=1.0) -> Path:
    from huggingface_hub import snapshot_download

    stop = threading.Event()
    t = threading.Thread(target=_watch, args=(dest, int(expected_gb * 1e9), report,
                                              f"Скачиваю {repo}", stop, lo, hi), daemon=True)
    t.start()
    try:
        snapshot_download(repo_id=repo, local_dir=str(dest), allow_patterns=patterns)
    finally:
        stop.set()
    return dest


def _url_download(url: str, dest: Path, report: Progress, lo: float, hi: float):
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                report(lo + (hi - lo) * done / total, f"Скачиваю {url.rsplit('/', 1)[-1]}")
    tmp.replace(dest)


def _convert_whisper(src: Path, out: Path, report: Progress):
    """HF Transformers Whisper -> CTranslate2 float16 (guide: max accuracy on GPU)."""
    import ctranslate2
    from transformers import AutoTokenizer

    report(0.85, "Конвертирую в CTranslate2 (float16)…")
    tmp_out = out.with_name(out.name + ".converting")
    shutil.rmtree(tmp_out, ignore_errors=True)
    copy = [f for f in ("preprocessor_config.json",) if (src / f).exists()]
    ctranslate2.converters.TransformersConverter(str(src), copy_files=copy).convert(
        str(tmp_out), quantization="float16", force=True)
    # faster-whisper needs tokenizer.json; fine-tunes often ship only the slow tokenizer files
    if (src / "tokenizer.json").exists():
        shutil.copy2(src / "tokenizer.json", tmp_out / "tokenizer.json")
    else:
        AutoTokenizer.from_pretrained(str(src), use_fast=True).backend_tokenizer.save(
            str(tmp_out / "tokenizer.json"))
    shutil.rmtree(out, ignore_errors=True)
    tmp_out.rename(out)


def install(key: str, report: Progress | None = None) -> Path:
    """Download (and convert if needed) a model. Blocking."""
    m = MODELS[key]
    state = _installing.setdefault(key, {})
    state.update(running=True, progress=0.0, message="Подготовка…", error=None)

    def rep(frac: float, msg: str):
        state.update(progress=round(frac, 3), message=msg)
        if report:
            report(frac, msg)

    try:
        with _install_lock:
            existing = find(key)
            if existing:
                rep(1.0, "Готово")
                return existing
            out = model_dir(key)
            work = MODELS_DIR / f".download-{key}"
            if m["kind"] == "diar":
                work.mkdir(parents=True, exist_ok=True)
                seg_url, emb_url = m["urls"]
                arc = work / "seg.tar.bz2"
                _url_download(seg_url, arc, rep, 0.0, 0.1)
                with tarfile.open(arc) as tf:
                    tf.extractall(work, filter="data")
                seg = next(work.rglob("model.onnx"))
                _url_download(emb_url, work / "embedding.onnx", rep, 0.1, 1.0)
                out.mkdir(parents=True, exist_ok=True)
                shutil.move(str(seg), out / "segmentation.onnx")
                shutil.move(str(work / "embedding.onnx"), out / "embedding.onnx")
            elif m.get("convert"):
                patterns = ["*.json", "*.txt", "*.safetensors", "*.model"]
                _hf_download(m["repo"], work, patterns, m["size_gb"], rep, 0.0, 0.85)
                _convert_whisper(work, out, rep)
            elif m.get("subfolder"):
                sub = m["subfolder"]
                _hf_download(m["repo"], work, [f"{sub}/*"], m["size_gb"], rep, 0.0, 1.0)
                shutil.rmtree(out, ignore_errors=True)
                shutil.move(str(work / sub), out)
            else:
                _hf_download(m["repo"], out, m.get("patterns"), m["size_gb"], rep, 0.0, 1.0)
            shutil.rmtree(work, ignore_errors=True)  # don't leave multi-GB sources behind
            shutil.rmtree(out / ".cache", ignore_errors=True)
            if not _is_complete(out, m["kind"], m.get("files")):
                raise RuntimeError("Модель скачалась не полностью")
            rep(1.0, "Готово")
            return out
    except Exception as e:  # noqa: BLE001
        state.update(error=str(e), message=f"Ошибка: {e}")
        raise
    finally:
        state["running"] = False


def install_async(key: str) -> None:
    if _installing.get(key, {}).get("running"):
        return
    _installing[key] = {"running": True, "progress": 0, "message": "В очереди…"}

    def run():
        try:
            install(key)
        except Exception:  # noqa: BLE001 - error is stored in state
            pass

    threading.Thread(target=run, daemon=True).start()
