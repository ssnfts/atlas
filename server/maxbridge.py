"""
Client for the 3ds Max bridge.

Speaks newline-delimited JSON over TCP to ``bridge/atlas_max_bridge.py`` running
inside 3ds Max. One connection per command, matching the bridge side.
"""

from __future__ import annotations

import json
import os
import socket
from typing import Any

__all__ = ["MaxBridge", "MaxBridgeError", "MaxNotRunning"]

DEFAULT_HOST = "127.0.0.1"
# 9876 Houdini, 9877 ZBrush, 9878 Marvelous Designer -- see .env.example.
DEFAULT_PORT = int(os.environ.get("ATLAS_MAX_PORT", "9879"))


class MaxBridgeError(RuntimeError):
    """A command reached 3ds Max and failed there."""


class MaxNotRunning(MaxBridgeError):
    """Nothing is listening — Max is closed or the bridge was never started."""


class MaxBridge:
    """Thin synchronous client. Cheap to construct; holds no socket between calls."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port

    # ── transport ────────────────────────────────────────────────────────────

    def _send(self, payload: dict, timeout: float = 120.0) -> Any:
        payload = {**payload, "timeout": timeout}
        blob = json.dumps(payload).encode("utf-8") + b"\n"

        try:
            with socket.create_connection((self.host, self.port), timeout=10.0) as sock:
                sock.settimeout(timeout + 15.0)
                sock.sendall(blob)

                buf = b""
                while b"\n" not in buf:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
        except ConnectionRefusedError as exc:
            raise MaxNotRunning(
                f"Nothing is listening on {self.host}:{self.port}. Start 3ds Max, "
                f"then run in its Python listener:\n"
                f'    import sys; sys.path.append(r"{_bridge_dir()}")\n'
                f"    import atlas_max_bridge; atlas_max_bridge.start()"
            ) from exc
        except socket.timeout as exc:
            raise MaxBridgeError(
                f"timed out talking to 3ds Max after {timeout}s. A modal dialog "
                "open in Max blocks every command behind it."
            ) from exc

        if not buf:
            raise MaxBridgeError("3ds Max closed the connection without replying")

        line, _, _ = buf.partition(b"\n")
        try:
            response = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise MaxBridgeError(f"malformed reply from Max: {line[:400]!r}") from exc

        if not response.get("ok"):
            raise MaxBridgeError(response.get("error", "unknown error in 3ds Max"))
        return response.get("result")

    # ── commands ─────────────────────────────────────────────────────────────

    def ping(self) -> dict:
        """Health check. Returns Max version, scene name and system units."""
        return self._send({"command": "ping"}, timeout=15.0)

    def is_available(self) -> bool:
        try:
            self.ping()
            return True
        except MaxBridgeError:
            return False

    def call(self, func: str, *args: Any, timeout: float = 120.0, **kwargs: Any) -> Any:
        """Invoke a MaxScript function, e.g. ``call("Box", length=10)``."""
        return self._send(
            {
                "command": "call",
                "mode": "call",
                "func": func,
                "args": list(args),
                "kwargs": kwargs,
            },
            timeout=timeout,
        )

    def get(self, path: str, timeout: float = 60.0) -> Any:
        """
        Read a MaxScript global or dotted path, e.g. ``get("units.SystemType")``.

        Separate from :meth:`call` because the mode cannot be inferred: pymxs
        value wrappers report as callable, so reading and invoking must be
        distinguished by the caller, not guessed.
        """
        return self._send(
            {"command": "call", "mode": "get", "func": path}, timeout=timeout
        )

    def set(self, path: str, value: Any, timeout: float = 60.0) -> Any:
        """Write a MaxScript global or dotted path."""
        return self._send(
            {"command": "call", "mode": "set", "func": path, "value": value},
            timeout=timeout,
        )

    def properties(
        self, *, node: str | None = None, cls: str | None = None, timeout: float = 60.0
    ) -> dict:
        """
        Introspect a live object's properties.

        Use this instead of trusting remembered parameter names — V-Ray renames
        sun/sky parameters between releases and a wrong name fails silently.
        """
        return self._send(
            {"command": "properties", "node": node, "class": cls}, timeout=timeout
        )

    def node_get(self, node: str, prop: str, timeout: float = 60.0) -> Any:
        """Read one property from a scene node."""
        return self._send(
            {"command": "node_get", "node": node, "prop": prop}, timeout=timeout
        )

    def node_set(self, node: str, prop: str, value: Any, timeout: float = 60.0) -> Any:
        """
        Write one property on a scene node.

        Raises if the property does not exist on that object, rather than
        letting MaxScript swallow the assignment silently.
        """
        return self._send(
            {"command": "node_set", "node": node, "prop": prop, "value": value},
            timeout=timeout,
        )

    def batch(
        self, steps: list[dict], *, stop_on_error: bool = True, timeout: float = 300.0
    ) -> list[dict]:
        """
        Run several commands inside one main-thread slot.

        Correctness feature as much as a speed one: the steps are not interleaved
        with UI events, so a read-modify-write sequence sees a consistent scene.
        """
        result = self._send(
            {"command": "batch", "steps": steps, "stop_on_error": stop_on_error},
            timeout=timeout,
        )
        return result["steps"]

    def vray_sky_setup(
        self,
        *,
        sun_node: str | None = None,
        sky_class: str = "VRaySky",
        params: dict | None = None,
        timeout: float = 60.0,
    ) -> dict:
        """
        Create a VRaySky, assign it to the environment slot and bind it to a sun.

        Atomic by necessity: a texmap has no name or handle, so unlike a scene
        node it cannot be returned to this process and referenced in a follow-up
        call. The whole sequence runs in one main-thread slot inside Max.
        """
        return self._send(
            {
                "command": "vray_sky_setup",
                "sun_node": sun_node,
                "sky_class": sky_class,
                "params": params or {},
            },
            timeout=timeout,
        )

    def vray_hdri_env(
        self,
        path: str,
        *,
        horizontal_rotation: float = 0.0,
        multiplier: float = 1.0,
        maptype: int = 2,
        timeout: float = 120.0,
    ) -> dict:
        """
        Put an HDRI in the environment slot, rotated to a bearing.

        Atomic host-side: a texmap has no name or handle to round-trip. Check
        ``use_environment_map`` in the reply — assigning the map and enabling it
        are separate operations in Max, and a scene with a perfectly good HDRI
        sitting in an unticked slot renders on the default grey.
        """
        return self._send(
            {
                "command": "vray_hdri_env",
                "path": path,
                "horizontal_rotation": horizontal_rotation,
                "multiplier": multiplier,
                "maptype": maptype,
            },
            timeout=timeout,
        )

    def set_keys(self, node: str, keys: list[dict], *, timeout: float = 180.0) -> dict:
        """
        Key a node's position and heading over frames.

        Each key is ``{"frame": f, "pos": [x, y, z], "heading_deg": h}``; either
        of the two values may be omitted. Check ``moved`` in the reply — a node
        whose controller refuses keys accepts the assignment and keeps its old
        value, which looks like a scene that simply does not animate.
        """
        return self._send(
            {"command": "set_keys", "node": node, "keys": keys}, timeout=timeout
        )

    def animation_range(self, start: int, end: int, *, fps: int | None = None,
                        timeout: float = 60.0) -> dict:
        """Set the scene's animation range, and optionally the frame rate."""
        return self._send(
            {"command": "animation_range", "start": start, "end": end, "fps": fps},
            timeout=timeout,
        )

    def save_scene(self, path: str, *, use_new_file: bool = False,
                   timeout: float = 600.0) -> dict:
        """
        Save the scene to ``path``, verified from disk.

        Raises if the file was not rewritten. ``saveMaxFile`` reports that it
        queued the save rather than that the bytes landed, so the reply carries
        the size and the object count read back after the fact.
        """
        return self._send(
            {"command": "save_scene", "path": path, "use_new_file": use_new_file},
            timeout=timeout,
        )

    def maxscript(self, code: str, timeout: float = 120.0) -> Any:
        """Evaluate raw MaxScript. Disabled unless ATLAS_ALLOW_MAXSCRIPT=1 in Max."""
        return self._send({"command": "maxscript", "code": code}, timeout=timeout)

    def viewport_capture(self, path: str, timeout: float = 60.0) -> dict:
        """Save a viewport grab so a multimodal model can check its own work."""
        return self._send(
            {"command": "viewport_capture", "path": path}, timeout=timeout
        )

    def scene_list(
        self, *, cls: str | None = None, prefix: str | None = None, timeout: float = 60.0
    ) -> dict:
        """List scene nodes, optionally filtered by class or name prefix."""
        return self._send(
            {"command": "scene_list", "class": cls, "prefix": prefix}, timeout=timeout
        )

    def assign_material(
        self,
        nodes: str | list[str],
        *,
        params: dict | None = None,
        name: str | None = None,
        material_class: str = "VRayMtl",
        timeout: float = 120.0,
    ) -> dict:
        """Create a material and assign it to one or more nodes."""
        return self._send(
            {
                "command": "assign_material",
                "nodes": [nodes] if isinstance(nodes, str) else list(nodes),
                "params": params or {},
                "name": name,
                "material_class": material_class,
            },
            timeout=timeout,
        )

    def create_mesh(
        self,
        name: str,
        verts: list[tuple[float, float, float]],
        faces: list[tuple[int, int, int]],
        *,
        uvs: list[tuple[float, float]] | None = None,
        smooth: int = 0,
        wirecolor: tuple[float, float, float] | None = None,
        timeout: float = 300.0,
    ) -> dict:
        """
        Build an Editable_Mesh from vertices and 0-based triangle indices.

        Atomic host-side: assembling a mesh through generic ``call`` would cost
        one main-thread slot per vertex, and a city block is thousands of them.

        Indices stay 0-based on this side. The +1 for MaxScript happens once,
        inside the handler.

        The reply carries the vertex count and bounding box read back out of the
        scene, so a caller can check that what landed is what was sent — see
        :meth:`create_meshes`, which does exactly that.
        """
        return self._send(
            {
                "command": "create_mesh",
                "name": name,
                "verts": [[float(x), float(y), float(z)] for x, y, z in verts],
                "faces": [[int(a), int(b), int(c)] for a, b, c in faces],
                "uvs": [[float(u), float(v)] for u, v in uvs] if uvs else None,
                "smooth": int(smooth),
                "wirecolor": list(wirecolor) if wirecolor else None,
            },
            timeout=timeout,
        )

    def create_meshes(
        self,
        meshes: list[tuple[str, list, list]],
        *,
        chunk: int = 40,
        stop_on_error: bool = False,
        timeout: float = 600.0,
    ) -> list[dict]:
        """
        Create many meshes, batched into one main-thread slot per chunk.

        ``stop_on_error`` defaults to False here, unlike :meth:`batch`: one
        badly-mapped OSM footprint should cost that building, not the other
        thirty-nine in the chunk. Failures come back as entries with ``ok``
        False and are the caller's to count.

        Chunking exists because a single request holds the main thread for its
        whole duration — a thousand buildings in one batch freezes Max's UI for
        long enough to look like a hang.
        """
        results: list[dict] = []
        for start in range(0, len(meshes), chunk):
            steps = []
            for entry in meshes[start:start + chunk]:
                # (name, verts, faces) or (name, verts, faces, uvs).
                name, verts, faces = entry[0], entry[1], entry[2]
                uvs = entry[3] if len(entry) > 3 else None
                step = {
                    "command": "create_mesh",
                    "name": name,
                    "verts": [[float(x), float(y), float(z)] for x, y, z in verts],
                    "faces": [[int(a), int(b), int(c)] for a, b, c in faces],
                }
                if uvs:
                    step["uvs"] = [[float(u), float(v)] for u, v in uvs]
                steps.append(step)
            results += self.batch(steps, stop_on_error=stop_on_error, timeout=timeout)
        return results

    def build_material(
        self,
        graph: dict,
        nodes: list[str],
        *,
        name: str | None = None,
        timeout: float = 300.0,
    ) -> dict:
        """
        Build a procedural texmap graph and assign the resulting material.

        ``graph`` is a ``texturing.Graph.as_dict()``. Atomic host-side because a
        texmap has no name or handle and cannot round-trip to this process.

        Check ``rejected`` in the reply. Texmap parameter names could not be
        verified offline, so a non-empty map is the loud failure that replaces a
        quietly untextured render.
        """
        return self._send(
            {
                "command": "build_material",
                "graph": graph,
                "nodes": [nodes] if isinstance(nodes, str) else list(nodes),
                "name": name,
            },
            timeout=timeout,
        )

    def set_renderer(
        self,
        renderer: str,
        *,
        also_activeshade: bool = True,
        timeout: float = 120.0,
    ) -> dict:
        """
        Set the production renderer. Accepts a prefix, e.g. ``"V_Ray_GPU"``.

        Must be called **after** any resetMaxFile: resetting the scene reverts
        the renderer to the application default.
        """
        return self._send(
            {
                "command": "set_renderer",
                "renderer": renderer,
                "also_activeshade": also_activeshade,
            },
            timeout=timeout,
        )

    def render(
        self,
        path: str,
        *,
        camera: str | None = None,
        width: int = 640,
        height: int = 360,
        expect_renderer: str | None = None,
        timeout: float = 1800.0,
    ) -> dict:
        """
        Render to a file.

        Pass ``expect_renderer`` to make the render fail loudly rather than
        quietly producing an image from the wrong engine.

        A render holds the Max main thread for its whole duration, so every
        other command queues behind it — hence the long default timeout.
        """
        return self._send(
            {
                "command": "render",
                "path": path,
                "camera": camera,
                "width": width,
                "height": height,
                "expect_renderer": expect_renderer,
            },
            timeout=timeout,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def point3(x: float, y: float, z: float) -> dict:
        """Wrap a triple so the bridge rebuilds it as a MaxScript Point3."""
        return {"__point3__": [x, y, z]}

    @staticmethod
    def color(r: float, g: float, b: float) -> dict:
        return {"__color__": [r, g, b]}

    @staticmethod
    def name(value: str) -> dict:
        """A MaxScript #name literal."""
        return {"__name__": value}

    @staticmethod
    def node(name: str) -> dict:
        """Reference an existing scene node by name."""
        return {"__node__": name}


def _bridge_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge")
