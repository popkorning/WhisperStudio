"""Only the app's own window may change anything through the local HTTP API.

CPU only, stdlib. Run from the repo root:

    set WHISPER_STUDIO_HOME=%TEMP%\\ws-agent-store
    python -m unittest tests.test_server -v
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
REAL_HOME = Path(os.environ.get("WS_REAL_HOME") or Path(os.environ.get("LOCALAPPDATA", Path.home())) / "WhisperStudio")  # the real data folder: models, library

if not os.environ.get("WHISPER_STUDIO_HOME"):
    os.environ["WHISPER_STUDIO_HOME"] = tempfile.mkdtemp(prefix="ws-server-")
if Path(os.environ["WHISPER_STUDIO_HOME"]).resolve() == REAL_HOME.resolve():
    sys.exit("tests/test_server.py: set WHISPER_STUDIO_HOME to a scratch folder, not the real home")

from whisper_studio import server  # noqa: E402


class Guard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = server.Server(("127.0.0.1", 0), server.Handler)  # no worker start: not make_server
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def ask(self, method, path, body=None, host=None, cookie=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        h = {"Host": host or f"127.0.0.1:{self.port}"}
        if cookie:
            h["Cookie"] = cookie
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "text/plain"  # what a cross-site "simple" request may send
        c.request(method, path, data, h)
        r = c.getresponse()
        r.read()
        c.close()
        return r.status, r.getheader("Set-Cookie")

    def test_window_gets_the_token(self):
        code, cookie = self.ask("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn(f"{server.COOKIE}={server.TOKEN}", cookie)
        self.assertIn("SameSite=Strict", cookie)

    def test_reads_are_allowed(self):
        self.assertEqual(self.ask("GET", "/api/dictionary")[0], 200)

    def test_changes_need_the_token(self):
        body = {"terms": ["Stadt"]}
        self.assertEqual(self.ask("PUT", "/api/dictionary", body)[0], 403)
        self.assertEqual(self.ask("PUT", "/api/dictionary", body, cookie=f"{server.COOKIE}=wrong")[0], 403)
        ok = f"theme=dark; {server.COOKIE}={server.TOKEN}"
        self.assertEqual(self.ask("PUT", "/api/dictionary", body, cookie=ok)[0], 200)

    def test_foreign_host_is_refused(self):  # DNS rebinding: evil.example resolving to 127.0.0.1
        self.assertEqual(self.ask("GET", "/api/dictionary", host=f"evil.example:{self.port}")[0], 403)
        self.assertEqual(self.ask("GET", "/", host="evil.example")[0], 403)
        self.assertEqual(self.ask("GET", "/api/dictionary", host=f"localhost:{self.port}")[0], 200)


if __name__ == "__main__":
    unittest.main()
