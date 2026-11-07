"""Command line interface: python -m whisper_studio <command> ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

RECOMMENDED = ["large-v3", "large-v3-turbo", "ru-large-v3", "de-turbo",
               "align-ru", "align-de", "align-en", "diarization"]


def _bar(frac: float, msg: str):
    n = int(frac * 30)
    sys.stderr.write(f"\r[{'#' * n}{'.' * (30 - n)}] {frac * 100:5.1f}%  {msg[:60]:<60}")
    sys.stderr.flush()


def _via_worker(doc_id: str, echo) -> dict:
    """Queue the job for the shared worker (one GPU user at a time) and follow its progress."""
    import time

    from . import jobs, store

    jobs.submit_process(doc_id)
    while True:
        time.sleep(0.5)
        d = store.load(doc_id)
        if echo and d["status"] in ("queued", "processing"):
            echo(d.get("progress") or 0.0, d.get("stage") or "")
        if d["status"] == "done":
            return d
        if d["status"] == "error":
            raise SystemExit(f"\nОшибка: {d.get('error')}")
        jobs.kick()


def transcribe_options(a):
    """What is not given comes from the user's defaults (mixed Russian + German among them), as in
    the UI: without "mixed" a whole German stretch is recognized as Russian, that is translated."""
    from . import options

    return options.make_options(dict(
        language=a.lang, model=a.model, profile=a.profile, glossary=a.glossary or "", topic=a.topic or "",
        num_speakers=a.speakers, mixed=a.mixed, beam_size=a.beam,
        diarize=False if a.no_diarize else None, align=False if a.no_align else None,
        stereo_split=True if a.stereo else None, highpass=True if a.highpass else None,
    ))


def cmd_transcribe(a):
    from . import export, pipeline

    opts = transcribe_options(a)
    fmts = [f.strip() for f in a.format.split(",") if f.strip()]
    for f in a.files:
        p = Path(f)
        doc = pipeline.submit(p, opts)
        sys.stderr.write(f"{p.name} -> {doc['id']}\n")
        echo = None if a.quiet else _bar
        doc = pipeline.process(doc["id"], echo=echo) if a.here else _via_worker(doc["id"], echo)
        sys.stderr.write("\n")
        out_dir = Path(a.out) if a.out else p.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for fmt in fmts:
            target = out_dir / f"{p.stem}.{fmt}"
            target.write_text(export.render(doc, fmt), "utf-8")
            sys.stderr.write(f"  saved {target}\n")
        if a.print:
            print(export.render(doc, "txt"))


def cmd_models(a):
    from . import models

    if a.action == "list":
        for m in models.status():
            mark = "✓" if m["installed"] else " "
            print(f"[{mark}] {m['key']:<16} {m['size_gb']:>4} GB  {m['title']}")
        return
    keys = RECOMMENDED if a.action == "install-recommended" else a.keys
    for k in keys:
        sys.stderr.write(f"{k}\n")
        models.install(k, _bar)
        sys.stderr.write("\n")


def cmd_list(_a):
    from . import store

    for d in store.list_docs():
        dur = f"{d['duration'] / 60:.1f} мин" if d["duration"] else "—"
        print(f"{d['id']}  {d['status']:<10} {dur:>9}  {d['language'] or '':<3} {d['title']}")


def cmd_show(a):
    from . import export, store

    print(export.render(store.load(a.id), a.format))


def cmd_export(a):
    from . import export, store

    Path(a.out).write_text(export.render(store.load(a.id), a.format), "utf-8")
    print(a.out)


def cmd_search(a):
    from . import export, store

    for h in store.search(a.query):
        print(f"{h['doc_id']}  [{export.clock(h['start'])}] {h['speaker']}: {h['text']}")


def cmd_redoubt(a):
    """Doubt marks of existing "max" transcripts recomputed from their stored hypotheses (CPU, no
    recognition): only word p changes. Gold references only when named."""
    from dataclasses import fields

    from . import ensemble, store
    from .options import Options

    docs = store.list_docs()
    if a.ids:
        docs = [d for d in docs if any(d["id"].endswith(i) for i in a.ids)]
    else:
        docs = [d for d in docs if not d.get("gold")]
    for s in docs:
        if s["status"] != "done":
            continue
        doc = store.load(s["id"])
        o = {k: v for k, v in (doc.get("options") or {}).items() if k in {f.name for f in fields(Options)}}
        words = ensemble.merged_words(store.path(s["id"]), Options(**o))
        if words is None:
            continue
        with store.editing(s["id"]) as doc:
            n = ensemble.refresh_doubt(doc, words)
            if n:
                store.save(doc)
        low = sum(1 for x in doc["segments"] for w in x.get("words") or []
                  if w.get("p", 1) < store.REVIEW_P and not w.get("edited"))
        print(f"{s['id']}  слов с новой уверенностью: {n}, сомнительных: {low}  {s['title']}")


def cmd_renumber(a):
    """Numbers of existing transcripts written in digits from their stored words (CPU, no
    recognition): segments the owner or the AI edited and gold references stay as they are.
    --dry-run prints every change and saves nothing."""
    import json

    from . import store
    from .asr import itn
    from .config import LIBRARY_DIR

    ids = sorted(p.parent.name for p in LIBRARY_DIR.glob("*/doc.json"))  # read only: no index rebuild
    if a.ids:
        ids = [i for i in ids if any(i.endswith(x) for x in a.ids)]
    total = 0
    for i in ids:
        doc = store.load(i)
        if doc.get("status") != "done" or store.is_gold(doc):
            continue
        before = json.dumps(doc["segments"], ensure_ascii=False)
        ch = itn.renumber_doc(doc)
        if a.dry_run:
            for sid, old, new in ch:
                print(f"{i[-6:]} {sid}: {old}\n           -> {new}")
        elif json.dumps(doc["segments"], ensure_ascii=False) != before:  # the variants count too
            with store.editing(i) as cur:
                ch = itn.renumber_doc(cur)  # again on the locked copy: an edit may have landed meanwhile
                store.save(cur)
        total += len(ch)
        if ch:
            print(f"{i}  сегментов с числами цифрами: {len(ch)}  {doc.get('title') or ''}")
    print(f"Итого сегментов: {total}" + (" (ничего не сохранено: --dry-run)" if a.dry_run else ""))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whisper-studio", description="Точная транскрипция аудио (Whisper)")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("ui", help="открыть приложение")
    sub.add_parser("mcp", help="MCP-сервер (stdio) для ИИ-агентов")
    sub.add_parser("worker", help="фоновый обработчик очереди (запускается сам по требованию)")
    sub.add_parser("shortcut", help="создать ярлык «Транскрибация» на рабочем столе")

    t = sub.add_parser("transcribe", help="распознать файлы")
    t.add_argument("files", nargs="+")
    t.add_argument("--lang", help="ru, de, en… или auto (по умолчанию — из настроек)")
    t.add_argument("--model", help="ключ модели из `models list` или auto")
    t.add_argument("--profile", choices=["balanced", "clean", "noisy"])
    t.add_argument("--speakers", type=int, default=None, help="число спикеров (по умолчанию — из настроек)")
    t.add_argument("--mixed", action=argparse.BooleanOptionalAction, default=None,
                   help="русская речь с немецкими словами и фразами (по умолчанию — из настроек)")
    t.add_argument("--glossary", help="имена и термины через запятую")
    t.add_argument("--topic", help="о чём запись (одно предложение)")
    t.add_argument("--beam", type=int)
    t.add_argument("--no-diarize", action="store_true")
    t.add_argument("--no-align", action="store_true")
    t.add_argument("--stereo", action="store_true", help="стерео-звонок: каждый канал — отдельный спикер")
    t.add_argument("--highpass", action="store_true", help="фильтр гула 80 Гц")
    t.add_argument("--format", default="txt", help="txt,md,srt,vtt,json через запятую")
    t.add_argument("--out", help="папка для результатов (по умолчанию рядом с файлом)")
    t.add_argument("--print", action="store_true", help="вывести текст в stdout")
    t.add_argument("--quiet", action="store_true")
    t.add_argument("--here", action="store_true", help="считать в этом процессе, а не в общем воркере")

    m = sub.add_parser("models", help="модели")
    m.add_argument("action", choices=["list", "install", "install-recommended"])
    m.add_argument("keys", nargs="*")

    sub.add_parser("list", help="библиотека транскриптов")
    s = sub.add_parser("show", help="показать транскрипт")
    s.add_argument("id")
    s.add_argument("--format", default="txt")
    e = sub.add_parser("export", help="экспорт транскрипта")
    e.add_argument("id")
    e.add_argument("--format", default="srt")
    e.add_argument("--out", required=True)
    q = sub.add_parser("search", help="поиск по всем транскриптам")
    q.add_argument("query")
    v = sub.add_parser("voices", help="(служебная) голоса спикеров записи заранее, на процессоре")
    v.add_argument("id")
    v.add_argument("model_dir")
    g = sub.add_parser("gold-eval", help="оценить ансамбль по эталонам (запускает приложение; итог в gold/)")
    g.add_argument("--tune", action="store_true", help="подобрать веса участников")
    r = sub.add_parser("redoubt", help="пересчитать пометки сомнения по сохранённым гипотезам (без GPU)")
    r.add_argument("ids", nargs="*", help="ID транскриптов (по умолчанию все, кроме эталонов)")
    n = sub.add_parser("renumber", help="записать числа цифрами в готовых транскриптах (без GPU)")
    n.add_argument("ids", nargs="*", help="ID транскриптов (по умолчанию все; эталоны не трогаются)")
    n.add_argument("--dry-run", action="store_true", help="только показать изменения")
    return p


def main(argv=None):
    p = parser()
    a, rest = p.parse_known_args(argv)
    if a.cmd == "worker":
        from .worker import main as worker

        return worker(rest)
    if rest:
        p.error("unrecognized arguments: " + " ".join(rest))
    if a.cmd in (None, "ui"):
        from .app import main as ui

        return ui()
    if a.cmd == "shortcut":
        from .app import install_shortcut

        return print(install_shortcut())
    if a.cmd == "mcp":
        import os

        from .jobs import LIGHT_ENV

        os.environ.update(LIGHT_ENV)  # the models run in the worker, not here
        from .mcp_server import main as mcp

        return mcp()
    if a.cmd == "voices":
        from .diarize import precompute_doc

        return precompute_doc(a.id, a.model_dir)
    if a.cmd == "gold-eval":
        from .gold import main as gold_eval

        return gold_eval(["--tune"] if a.tune else [])
    {"transcribe": cmd_transcribe, "models": cmd_models, "list": cmd_list, "show": cmd_show,
     "export": cmd_export, "search": cmd_search, "redoubt": cmd_redoubt,
     "renumber": cmd_renumber}[a.cmd](a)


if __name__ == "__main__":
    main()
