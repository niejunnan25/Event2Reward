"""Local web server of the annotation page: `event2reward annotate serve PROJECT`.

Serves the page, the project's videos (with byte ranges, so the browser can
seek), and a small JSON API. It binds to 127.0.0.1 and needs only the
standard library.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import shutil
from urllib.parse import unquote, urlparse

from . import project as store
from .schema import LABELS

WEB = Path(__file__).with_name("web")


class Handler(BaseHTTPRequestHandler):
    root: Path

    def log_message(self, *args):
        pass

    def send_json(self, value, status=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/api/project":
            return self.send_json({"name": self.root.name, "videos": store.read_project(self.root)["videos"],
                                   "annotations": store.read_annotations(self.root), "labels": LABELS})
        if path == "/api/export.jsonl":
            rows = store.export(self.root)
            data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="annotations.jsonl"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        base, relative = (self.root, path.lstrip("/")) if path.startswith("/media/") else (WEB, path.lstrip("/") or "index.html")
        target = (base / relative).resolve()
        if not target.is_relative_to(base.resolve()) or not target.is_file():
            return self.send_error(404)
        self.send_file(target)

    def send_file(self, target: Path):
        size = target.stat().st_size
        start, end = 0, size - 1
        ranged = self.headers.get("Range", "").startswith("bytes=")
        if ranged:
            left, _, right = self.headers["Range"][6:].partition("-")
            try:
                start, end = (int(left), min(int(right or end), end)) if left else (max(0, size - int(right)), end)
            except ValueError:
                return self.send_error(416)
            if start > end:
                return self.send_error(416)
        self.send_response(206 if ranged else 200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with target.open("rb") as f:
            f.seek(start)
            try:
                shutil.copyfileobj(_Limited(f, end - start + 1), self.wfile)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_POST(self):
        if urlparse(self.path).path != "/api/annotation":
            return self.send_error(404)
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self.send_json({"error": "cross-site request"}, 403)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            self.send_json(store.save_annotation(self.root, body["uid"], body["annotation"]))
        except (ValueError, KeyError, TypeError) as error:
            self.send_json({"error": str(error)}, 400)


class _Limited:
    def __init__(self, f, n):
        self.f, self.n = f, n

    def read(self, size=-1):
        if self.n <= 0:
            return b""
        data = self.f.read(self.n if size < 0 else min(size, self.n))
        self.n -= len(data)
        return data


def serve(root: str | Path, port: int = 8765, open_browser: bool = True) -> None:
    root = Path(root).resolve()
    store.read_project(root)
    handler = type("ProjectHandler", (Handler,), {"root": root})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Annotating {root}\nOpen {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
