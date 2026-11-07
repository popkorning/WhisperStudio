"""Desktop window: local API server + native WebView2 window (pywebview).

The window opens at once with a splash page and switches to the UI when the server is up.
This process stays light: transcription runs in the worker process (jobs.py, worker.py),
so heavy native imports never hold the GIL that the window's message loop needs.
"""
from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
import threading
import time
import urllib.request
import winreg
from ctypes import wintypes as W
from pathlib import Path

from .config import HOME, SERVER_FILE, WEB_DIR

TITLE = "Транскрибация"
APP_ID = "WhisperStudio.Transcription"  # AppUserModelID: taskbar button, icon and pinning
ICON = WEB_DIR / "icon.ico"
WINDOW_FILE = HOME / "window.json"
PYW = Path(__file__).resolve().parent.parent / "Whisper Studio.pyw"
SIZE, MIN_SIZE = (1320, 860), (760, 480)  # logical pixels (DIP); the UI folds panels below 1240
BG_LIGHT, BG_DARK = "#FFFFFF", "#1E1E20"  # = --bg of the UI, so there is no white flash

_AV = "*.ogg;*.oga;*.opus;*.mp3;*.wav;*.m4a;*.aac;*.flac;*.wma;*.amr;*.webm;*.mp4;*.mkv;*.mov;*.3gp"
AUDIO_TYPES = (f"Аудио и видео ({_AV})", "Все файлы (*.*)")

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32.LoadImageW.restype = W.HANDLE
user32.LoadImageW.argtypes = [W.HINSTANCE, W.LPCWSTR, W.UINT, ctypes.c_int, ctypes.c_int, W.UINT]
user32.SendMessageW.argtypes = [W.HWND, W.UINT, W.WPARAM, W.LPARAM]
user32.MonitorFromRect.restype = W.HANDLE
user32.MonitorFromWindow.restype = W.HANDLE
user32.MonitorFromWindow.argtypes = [W.HWND, W.DWORD]
user32.GetMonitorInfoW.argtypes = [W.HANDLE, ctypes.c_void_p]
kernel32.CreateMutexW.restype = W.HANDLE


class _Placement(ctypes.Structure):
    _fields_ = [("length", W.UINT), ("flags", W.UINT), ("showCmd", W.UINT), ("ptMin", W.POINT),
                ("ptMax", W.POINT), ("rcNormal", W.RECT)]


class _MonitorInfo(ctypes.Structure):
    _fields_ = [("cbSize", W.DWORD), ("rcMonitor", W.RECT), ("rcWork", W.RECT), ("dwFlags", W.DWORD)]


# ------------------------------------------------------------------ one instance

_mutex = None


def _find_window() -> int | None:
    found = []

    @ctypes.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)
    def cb(h, _):
        n = user32.GetWindowTextLengthW(h)
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(h, buf, n + 1)
        cls = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(h, cls, 64)
        if buf.value.startswith(TITLE) and cls.value.startswith("WindowsForms") and user32.IsWindowVisible(h):
            found.append(h)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def _bring_to_front(h: int) -> None:
    user32.ShowWindow(h, 9 if user32.IsIconic(h) else 5)  # SW_RESTORE / SW_SHOW
    if not user32.SetForegroundWindow(h):
        # foreground lock: a synthetic Alt press lets the launching process hand over focus
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)
        user32.SetForegroundWindow(h)


def _first_instance() -> bool:
    """False if the app is already running; its window is then brought to the front."""
    global _mutex
    _mutex = kernel32.CreateMutexW(None, False, "Local\\WhisperStudio.UI")
    if ctypes.get_last_error() != 183:  # ERROR_ALREADY_EXISTS
        return True
    for _ in range(150):  # the other instance may still be starting
        h = _find_window()
        if h:
            _bring_to_front(h)
            break
        time.sleep(0.1)
    return False


# ------------------------------------------------------------------ window geometry

def _scale() -> float:
    return user32.GetDpiForSystem() / 96


def _work_area(rect: tuple[int, int, int, int] | None = None) -> tuple[int, int, int, int] | None:
    """Work area (physical px) of the monitor showing rect, or of the primary one; None if off-screen."""
    if rect:
        mon = user32.MonitorFromRect(ctypes.byref(W.RECT(*rect)), 0)  # MONITOR_DEFAULTTONULL
        if not mon:
            return None
    else:
        mon = user32.MonitorFromWindow(None, 1)  # MONITOR_DEFAULTTOPRIMARY
    mi = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    user32.GetMonitorInfoW(mon, ctypes.byref(mi))
    r = mi.rcWork
    return r.left, r.top, r.right, r.bottom


def _geometry() -> dict:
    """Last size/position if it still fits a connected monitor, else a centered window
    that fits the screen at any scale (1320x860 is taller than 1080p at 125 %)."""
    s = _scale()
    try:
        g = json.loads(WINDOW_FILE.read_text("utf-8"))
        rect = tuple(round(v * s) for v in (g["x"], g["y"], g["x"] + g["width"], g["y"] + g["height"]))
        work = _work_area(rect)
    except Exception:  # noqa: BLE001 - first run or broken file
        g, work = {}, None
    if work is None:
        g, work = {}, _work_area()
    wl, wt, wr, wb = (v / s for v in work)
    ww, wh = wr - wl, wb - wt
    w = int(min(g.get("width", SIZE[0]), ww * (1 if g else 0.92)))
    h = int(min(g.get("height", SIZE[1]), wh * (1 if g else 0.92)))
    x = int(min(max(g.get("x", wl + (ww - w) / 2), wl), wr - w))
    y = int(min(max(g.get("y", wt + (wh - h) / 2), wt), wb - h))
    min_size = (min(MIN_SIZE[0], int(ww * 0.9), w), min(MIN_SIZE[1], int(wh * 0.9), h))
    return dict(x=x, y=y, width=w, height=h, min_size=min_size, maximized=bool(g.get("maximized")))


def _save_geometry(hwnd: int) -> None:
    pl = _Placement(length=ctypes.sizeof(_Placement))
    if not user32.GetWindowPlacement(hwnd, ctypes.byref(pl)):
        return
    r = pl.rcNormal  # workspace coordinates: shift by the taskbar if it sits left/top
    mi = _MonitorInfo(cbSize=ctypes.sizeof(_MonitorInfo))
    user32.GetMonitorInfoW(user32.MonitorFromWindow(hwnd, 2), ctypes.byref(mi))
    dx, dy = mi.rcWork.left - mi.rcMonitor.left, mi.rcWork.top - mi.rcMonitor.top
    s = (user32.GetDpiForWindow(hwnd) or 96) / 96
    g = dict(x=round((r.left + dx) / s), y=round((r.top + dy) / s), width=round((r.right - r.left) / s),
             height=round((r.bottom - r.top) / s), maximized=pl.showCmd == 3)
    WINDOW_FILE.write_text(json.dumps(g), "utf-8")


# ------------------------------------------------------------------ icon / taskbar identity

def _set_app_id() -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:  # noqa: BLE001
        pass


def _pythonw() -> Path:
    pyw = Path(sys.executable).with_name("pythonw.exe")
    return pyw if pyw.exists() else Path(sys.executable)


def _dark_title_bar(hwnd: int, dark: bool) -> None:
    """Match the native title bar to the theme (Windows 10 20H1+ / 11)."""
    v = ctypes.c_int(1 if dark else 0)
    for attr in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE, its pre-20H1 number
        if ctypes.windll.dwmapi.DwmSetWindowAttribute(W.HWND(hwnd), attr, ctypes.byref(v), 4) == 0:
            break


def _apply_icon(hwnd: int) -> None:
    """Crisp icons for the current DPI (WinForms picks one frame) and taskbar identity of the
    window itself, so pinning it relaunches the app rather than bare pythonw.exe."""
    dpi = user32.GetDpiForWindow(hwnd) or 96
    for which, metric in ((1, 11), (0, 49)):  # ICON_BIG/SM_CXICON, ICON_SMALL/SM_CXSMICON
        px = user32.GetSystemMetricsForDpi(metric, dpi)
        hicon = user32.LoadImageW(None, str(ICON), 1, px, px, 0x10)  # IMAGE_ICON, LR_LOADFROMFILE
        if hicon:
            user32.SendMessageW(hwnd, 0x80, which, hicon)  # WM_SETICON
    try:
        import pythoncom
        from win32com.propsys import propsys, pscon

        store = propsys.SHGetPropertyStoreForWindow(hwnd, propsys.IID_IPropertyStore)
        for key, value in ((pscon.PKEY_AppUserModel_ID, APP_ID),
                           (pscon.PKEY_AppUserModel_RelaunchCommand, f'"{_pythonw()}" "{PYW}"'),
                           (pscon.PKEY_AppUserModel_RelaunchDisplayNameResource, TITLE),
                           (pscon.PKEY_AppUserModel_RelaunchIconResource, f"{ICON},0")):
            store.SetValue(key, propsys.PROPVARIANTType(value, pythoncom.VT_LPWSTR))
        store.Commit()
    except Exception:  # noqa: BLE001 - cosmetic
        pass


def install_shortcut(path: str | Path | None = None) -> str:
    """Desktop shortcut that starts the app with our icon and AppUserModelID (one taskbar button)."""
    import pythoncom
    from win32com.propsys import propsys, pscon
    from win32com.shell import shell

    if path is None:
        desktop = shell.SHGetFolderPath(0, 0x10, None, 0)  # CSIDL_DESKTOPDIRECTORY
        path = Path(desktop) / f"{TITLE}.lnk"
    path = Path(path)
    link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
                                      shell.IID_IShellLink)
    link.SetPath(str(_pythonw()))
    link.SetArguments(f'"{PYW}"')
    link.SetWorkingDirectory(str(PYW.parent))
    link.SetIconLocation(str(ICON), 0)
    link.SetDescription("Транскрибация — локальная транскрипция аудио")
    props = link.QueryInterface(propsys.IID_IPropertyStore)
    props.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID, pythoncom.VT_LPWSTR))
    props.Commit()
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(path), 0)
    return str(path)


# ------------------------------------------------------------------ server

def serve(failed: list) -> int:
    """Start the API server in a thread; returns its port (0 if it failed, see failed)."""
    try:
        from .server import make_server

        httpd = make_server()
    except BaseException as e:  # noqa: BLE001 - shown on the splash page
        failed.append(e)
        return 0
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1]


def _dark() -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            return winreg.QueryValueEx(k, "AppsUseLightTheme")[0] == 0
    except OSError:
        return False


def _splash() -> str:
    try:
        logo = "data:image/png;base64," + base64.b64encode((WEB_DIR / "icon.png").read_bytes()).decode()
    except OSError:
        logo = ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:#fff;color:#6e6e73;cursor:default;user-select:none;
font:14px "Segoe UI Variable Text","Segoe UI",sans-serif}}
@media (prefers-color-scheme:dark){{html,body{{background:#1c1c1e;color:#98989d}}}}
body{{display:grid;place-items:center}} .c{{display:grid;justify-items:center;gap:18px}}
img{{width:72px;height:72px}}
.bar{{width:120px;height:3px;border-radius:2px;background:rgba(127,127,127,.18);overflow:hidden}}
.bar i{{display:block;width:40%;height:100%;border-radius:2px;background:#0071e3;animation:m 1.1s ease-in-out infinite}}
@keyframes m{{0%{{transform:translateX(-100%)}}100%{{transform:translateX(250%)}}}}
#e{{max-width:520px;text-align:center;white-space:pre-wrap}} #e.err{{color:#c93400}}
</style></head><body><div class="c"><img src="{logo}" alt=""><div class="bar"><i></i></div>
<div id="e">Запускаю…</div></div></body></html>"""


# ------------------------------------------------------------------ js api

class Api:
    """Functions callable from JS as window.pywebview.api.*"""

    def __init__(self):
        self._window = None

    def pick_files(self):
        import webview

        res = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                             file_types=AUDIO_TYPES)
        return list(res or [])

    def save_export(self, doc_id: str, fmt: str, title: str, timestamps: bool = True):
        import webview

        from . import export, store

        res = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename=f"{title}.{fmt}")
        if not res:
            return None
        target = res if isinstance(res, str) else res[0]
        Path(target).write_text(export.render(store.load(doc_id), fmt, timestamps), "utf-8")
        return target

    def reveal_file(self, path: str):
        """Show the file selected in Explorer."""
        import subprocess

        p = Path(path)
        if p.exists():
            subprocess.Popen(["explorer", "/select,", str(p)])
            return True
        return False

    def set_title(self, title: str = ""):
        # keeps the app name first: single-instance lookup finds the window by it
        self._window.set_title(title if title.startswith(TITLE) else f"{TITLE} — {title}" if title else TITLE)

    def set_theme(self, dark: bool):
        """The UI's own theme choice (system / light / dark) for the native title bar."""
        if self._hwnd:
            _dark_title_bar(self._hwnd, bool(dark))

    _hwnd = 0


# ------------------------------------------------------------------ main

def main():
    from .jobs import LIGHT_ENV

    os.environ.update(LIGHT_ENV)
    _set_app_id()
    if not _first_instance():
        return
    # The server (stdlib only) is up before pywebview/.NET, the slow cold import, even starts.
    failed: list = []
    port = serve(failed)
    url = f"http://127.0.0.1:{port}/"

    user32.SetProcessDPIAware()  # what pywebview sets too; done early so geometry is in real pixels
    geo = _geometry()
    import webview

    api = Api()
    win = webview.create_window(TITLE, html=_splash(), js_api=api, x=geo["x"], y=geo["y"],
                                width=geo["width"], height=geo["height"], min_size=geo["min_size"],
                                maximized=geo["maximized"], text_select=True,
                                background_color=BG_DARK if _dark() else BG_LIGHT)
    api._window = win
    hwnd = [0]

    def on_shown():
        try:
            hwnd[0] = api._hwnd = int(win.native.Handle.ToInt64())
            _dark_title_bar(hwnd[0], _dark())
            _apply_icon(hwnd[0])
        except Exception:  # noqa: BLE001
            pass

    def on_closing():
        try:
            if hwnd[0]:
                _save_geometry(hwnd[0])
        except Exception:  # noqa: BLE001
            pass

    win.events.shown += on_shown
    win.events.closing += on_closing

    def boot():
        t = time.monotonic()
        while time.monotonic() - t < 180:
            if failed:
                msg = json.dumps(f"Не удалось запустить сервер:\n{failed[0]!r}")
                win.evaluate_js(f"var e=document.getElementById('e');e.className='err';e.textContent={msg}")
                return
            try:
                urllib.request.urlopen(url + "api/settings", timeout=1)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        SERVER_FILE.write_text(json.dumps({"url": url, "pid": os.getpid()}), "utf-8")
        win.load_url(url)

    try:
        webview.start(boot, private_mode=False, icon=str(ICON))
    finally:
        try:
            if json.loads(SERVER_FILE.read_text("utf-8")).get("pid") == os.getpid():
                SERVER_FILE.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        # Leave right away: waiting for .NET/uvicorn teardown used to keep a windowless process alive.
        os._exit(0)


if __name__ == "__main__":
    main()
