"""
Atlas bridge core — runs *inside* 3ds Max.

Start it from the MAXScript listener (the one that opens by default):

    python.ExecuteFile @"<path-to-repo>\\bridge\\start_bridge.py"

Architecture — the DCC bridge pattern:

    socket thread -> queue -> QTimer on the main thread -> pymxs -> result

The 3ds Max API is **main-thread only**. Calling pymxs from a socket handler
thread corrupts scene state or hard-crashes Max, and the crash lands minutes
later somewhere unrelated, so it is nearly impossible to attribute. Every call
is marshalled onto the main thread through a QTimer, with the socket thread
blocking on an Event so the call still looks synchronous to the caller.

If PySide6 cannot be imported the bridge **refuses to start** rather than
falling back to direct calls: a bridge that runs and crashes the host hours
later is worse than one that will not start.

This module holds only the transport and lifecycle — the parts that own a bound
socket and rarely change. All command handling lives in ``atlas_max_handlers``,
which is hot-reloaded between requests when ATLAS_DEV_RELOAD=1 so handler edits
take effect without restarting the bridge or 3ds Max.
"""

from __future__ import annotations

import importlib
import json
import os
import queue
import socket
import threading
import traceback

try:
    from PySide6 import QtCore
except ImportError as _exc:  # pragma: no cover - depends on host
    raise RuntimeError(
        "Atlas bridge requires PySide6 to marshal calls onto the 3ds Max main "
        "thread. Without it, pymxs calls from the socket thread would crash Max "
        "unpredictably. Refusing to start."
    ) from _exc

try:
    import pymxs  # noqa: F401  (imported for the availability check)
except ImportError as _exc:  # pragma: no cover
    raise RuntimeError(
        "Atlas bridge must run inside 3ds Max (pymxs not importable)."
    ) from _exc

import atlas_max_handlers

# ── Configuration ─────────────────────────────────────────────────────────────

# 9876 is Houdini, 9877 ZBrush, 9878 Marvelous Designer on this machine.
HOST = "127.0.0.1"
PORT = int(os.environ.get("ATLAS_MAX_PORT", "9879"))

# Drain a bounded number of jobs per tick so a flood cannot freeze the UI.
JOBS_PER_TICK = 8
TICK_MS = 20

# Reload the handlers module before each job. On by default: handler edits are
# frequent during development, and the alternative is interrupting whoever is
# using Max to re-run the launcher after every change.
DEV_RELOAD = os.environ.get("ATLAS_DEV_RELOAD", "1") == "1"

ALLOW_MAXSCRIPT = atlas_max_handlers.ALLOW_MAXSCRIPT


class _Job:
    __slots__ = ("payload", "done", "result", "error", "traceback")

    def __init__(self, payload: dict):
        self.payload = payload
        self.done = threading.Event()
        self.result = None
        self.error: str | None = None
        self.traceback: str | None = None


class _Bridge:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.queue: "queue.Queue[_Job]" = queue.Queue()
        self.timer: QtCore.QTimer | None = None
        self.server: socket.socket | None = None
        self.thread: threading.Thread | None = None
        self.running = False

    # -- main thread --
    def _tick(self):
        for _ in range(JOBS_PER_TICK):
            try:
                job = self.queue.get_nowait()
            except queue.Empty:
                return

            try:
                handlers = atlas_max_handlers
                if DEV_RELOAD:
                    # Reload here, on the main thread, immediately before use.
                    # Reloading from the socket thread would race a job already
                    # executing against the previous module object.
                    handlers = importlib.reload(atlas_max_handlers)
                job.result = handlers.run_job(job.payload)
            except Exception as exc:
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
            finally:
                job.done.set()

    # -- socket thread --
    def _serve(self):
        assert self.server is not None
        while self.running:
            try:
                conn, _addr = self.server.accept()
            except OSError:
                break  # listener closed
            threading.Thread(
                target=self._handle_connection, args=(conn,), daemon=True
            ).start()

    def _handle_connection(self, conn: socket.socket):
        # Per-command connections: these bridges are low-throughput, and a
        # dropped socket then cannot wedge the session.
        with conn:
            try:
                conn.settimeout(1800.0)
                buf = b""
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf += chunk
                    if len(buf) > 64 * 1024 * 1024:
                        raise ValueError("request exceeded 64 MB")

                payload = json.loads(buf.partition(b"\n")[0].decode("utf-8"))

                job = _Job(payload)
                self.queue.put(job)

                timeout = float(payload.get("timeout", 120.0))
                if not job.done.wait(timeout):
                    response = {
                        "ok": False,
                        "error": (
                            f"timed out after {timeout}s waiting for the 3ds Max "
                            "main thread. A modal dialog open in Max blocks every "
                            "command behind it — check for one."
                        ),
                    }
                elif job.error:
                    response = {
                        "ok": False,
                        "error": job.error,
                        "traceback": job.traceback,
                    }
                else:
                    response = {"ok": True, "result": job.result}

            except Exception as exc:
                response = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }

            try:
                conn.sendall(json.dumps(response, default=str).encode("utf-8") + b"\n")
            except OSError:
                pass

    # -- lifecycle --
    def start(self):
        if self.running:
            print(f"[atlas] bridge already running on {self.host}:{self.port}")
            return

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Loopback only: this socket accepts code for execution inside the
            # host and must never be reachable off-machine.
            srv.bind((self.host, self.port))
        except OSError as exc:
            srv.close()
            raise RuntimeError(
                f"cannot bind {self.host}:{self.port} ({exc}). Another DCC bridge "
                f"may hold it — 9876 Houdini, 9877 ZBrush, 9878 Marvelous "
                f"Designer. Set ATLAS_MAX_PORT to choose another."
            ) from exc
        srv.listen(8)

        self.server = srv
        self.running = True

        # The QTimer must be created on the main thread, which is why start()
        # has to be called from the listener rather than from a thread.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

        print(f"[atlas] bridge listening on {self.host}:{self.port}")
        print(f"[atlas] raw MaxScript : {'ENABLED' if ALLOW_MAXSCRIPT else 'disabled'}")
        print(f"[atlas] handler reload: {'on' if DEV_RELOAD else 'off'}")

    def stop(self):
        self.running = False
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
            self.server = None
        print("[atlas] bridge stopped")


_BRIDGE = _Bridge(HOST, PORT)


def start():
    """Start the bridge. Must be called from the 3ds Max main thread."""
    _BRIDGE.start()


def stop():
    """Stop the bridge."""
    _BRIDGE.stop()


if __name__ == "__main__":
    start()
