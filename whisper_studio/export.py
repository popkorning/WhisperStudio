"""Export a document to txt / md / srt / vtt / json."""
from __future__ import annotations

import json

from .store import speaker_name

FORMATS = ("txt", "md", "srt", "vtt", "json")


def ts(t: float, sep: str = ",", hours: bool = True) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    if not hours:
        return f"{h * 60 + m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def clock(t: float) -> str:
    t = int(t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def blocks(doc: dict) -> list[dict]:
    """Consecutive segments of the same speaker merged into paragraphs."""
    out: list[dict] = []
    for s in doc["segments"]:
        sp = s.get("speaker")
        if out and out[-1]["speaker"] == sp and s["start"] - out[-1]["end"] < 3.0:
            out[-1]["text"] += " " + s["text"]
            out[-1]["end"] = s["end"]
        else:
            out.append(dict(speaker=sp, start=s["start"], end=s["end"], text=s["text"]))
    return out


def render(doc: dict, fmt: str, timestamps: bool = True) -> str:
    fmt = fmt.lower()
    has_spk = bool(doc.get("speakers"))
    if fmt == "json":
        return json.dumps(doc, ensure_ascii=False, indent=2)
    if fmt in ("txt", "md"):
        lines = []
        if fmt == "md":
            lines += [f"# {doc['title']}", ""]
        for b in blocks(doc):
            head = []
            if timestamps:
                head.append(f"[{clock(b['start'])}]")
            if has_spk and b["speaker"]:
                name = speaker_name(doc, b["speaker"])
                head.append(f"**{name}:**" if fmt == "md" else f"{name}:")
            lines.append((" ".join(head) + " " if head else "") + b["text"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"
    if fmt in ("srt", "vtt"):
        out = ["WEBVTT", ""] if fmt == "vtt" else []
        sep = "." if fmt == "vtt" else ","
        for i, s in enumerate(doc["segments"], 1):
            text = s["text"]
            if has_spk and s.get("speaker"):
                text = (f"<v {speaker_name(doc, s['speaker'])}>" if fmt == "vtt"
                        else f"[{speaker_name(doc, s['speaker'])}] ") + text
            if fmt == "srt":
                out.append(str(i))
            out.append(f"{ts(s['start'], sep)} --> {ts(s['end'], sep)}")
            out += [text, ""]
        return "\n".join(out)
    raise ValueError(f"Неизвестный формат: {fmt}")
