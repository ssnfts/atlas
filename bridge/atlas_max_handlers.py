"""
Command handlers for the Atlas bridge. **Runs on the 3ds Max main thread.**

Deliberately separate from ``atlas_max_bridge``: the socket server, queue and
QTimer there are stable and own a bound port, whereas these handlers change
constantly during development. Keeping them apart lets the core hot-reload this
module between requests (``ATLAS_DEV_RELOAD=1``) so handler edits take effect
without restarting the bridge — or 3ds Max.

Nothing here may touch the socket or the queue; it is called with the main
thread already held.
"""

from __future__ import annotations

import os
import re

import pymxs

rt = pymxs.runtime

# Raw MaxScript is arbitrary code execution inside the host. Needed for building
# option structs pymxs cannot express, so it cannot be removed outright — but it
# stays opt-in.
ALLOW_MAXSCRIPT = os.environ.get("ATLAS_ALLOW_MAXSCRIPT", "0") == "1"

PROTOCOL_VERSION = 2


# ── Value coercion ────────────────────────────────────────────────────────────

def coerce(value, depth: int = 0):
    """
    Convert a MaxScript value into something json.dumps will accept.

    Degrades unknown types to str() rather than failing the whole call: a
    partially-readable result beats a serialization error that discards work
    already done on the main thread.
    """
    if depth > 12:
        return "<max depth>"

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    try:
        if value is rt.undefined or value is rt.OK:
            return None
    except Exception:
        pass

    for attrs in (("x", "y", "z"), ("x", "y"), ("r", "g", "b")):
        if all(hasattr(value, a) for a in attrs):
            try:
                return [float(getattr(value, a)) for a in attrs]
            except Exception:
                pass

    # Scene nodes: return a stable identity, never the live wrapper.
    if hasattr(value, "name") and hasattr(value, "handle"):
        try:
            return {
                "__node__": str(value.name),
                "handle": int(value.handle),
                "class": str(rt.classOf(value)),
            }
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        return [coerce(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce(v, depth + 1) for k, v in value.items()}

    try:
        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            return [coerce(v, depth + 1) for v in value]
    except Exception:
        pass

    try:
        return str(value)
    except Exception:
        return repr(value)


def to_mxs(value):
    """Convert a JSON value into a MaxScript value where a mapping exists."""
    if isinstance(value, dict):
        if "__point3__" in value:
            x, y, z = value["__point3__"]
            return rt.Point3(float(x), float(y), float(z))
        if "__color__" in value:
            r, g, b = value["__color__"]
            return rt.Color(float(r), float(g), float(b))
        if "__name__" in value:
            return rt.Name(str(value["__name__"]))
        if "__node__" in value:
            node = rt.getNodeByName(str(value["__node__"]))
            if node is None:
                raise ValueError(f"no scene node named {value['__node__']!r}")
            return node
    if isinstance(value, list):
        return [to_mxs(v) for v in value]
    return value


# ── Path resolution ───────────────────────────────────────────────────────────

_INDEX_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\[(\d+)\]$")


class _IndexedLeaf:
    """Wraps an already-resolved indexed value so get/set/call read uniformly."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _step(target, part: str):
    """
    Resolve one path segment, supporting MaxScript's 1-based array indexing.

    ``objects[3]`` is ordinary MaxScript; without this the whole string is taken
    as one attribute name and fails with a confusing "no attribute 'objects[3]'".
    """
    if part.startswith("_"):
        raise ValueError("refusing to traverse private attribute")

    match = _INDEX_RE.match(part)
    if match is None:
        return getattr(target, part)

    name, index = match.group(1), int(match.group(2))
    collection = getattr(target, name)
    if index < 1:
        raise ValueError("MaxScript arrays are 1-based; index must be >= 1")
    try:
        return collection[index - 1]
    except (IndexError, TypeError) as exc:
        raise IndexError(f"{name}[{index}] is out of range") from exc


def resolve_path(path: str):
    """Walk a dotted MaxScript path, returning (owner, leaf_name)."""
    if not isinstance(path, str) or not path:
        raise ValueError("'func' must be a non-empty string")
    if path.startswith("_"):
        raise ValueError("refusing to access private attribute")

    parts = path.split(".")
    target = rt
    for part in parts[:-1]:
        target = _step(target, part)

    leaf = parts[-1]
    if _INDEX_RE.match(leaf) is not None:
        return _IndexedLeaf(_step(target, leaf)), "value"

    if not hasattr(target, leaf):
        raise AttributeError(f"MaxScript has no '{path}'")
    return target, leaf


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ping(_params: dict) -> dict:
    return {
        "pong": True,
        "protocol": PROTOCOL_VERSION,
        "max_version": coerce(rt.maxVersion()),
        "product": str(rt.productAppID),
        "scene": str(rt.maxFileName) or "<unsaved>",
        "units": str(rt.units.SystemType),
        "object_count": int(rt.objects.count),
        "maxscript_enabled": ALLOW_MAXSCRIPT,
        "dev_reload": os.environ.get("ATLAS_DEV_RELOAD", "0") == "1",
    }


def cmd_call(params: dict):
    """
    Generic dispatch against the MaxScript global namespace.

    ``mode`` is explicit rather than inferred: pymxs wraps every MaxScript
    value, and those wrappers report as callable even when they hold a plain
    value, so auto-detection tried to *invoke* ``#inches`` when asked to read
    ``units.SystemType``.
    """
    path = params.get("func")
    mode = params.get("mode", "call")
    target, leaf = resolve_path(path)

    if mode == "get":
        return coerce(getattr(target, leaf))

    if mode == "set":
        if "value" not in params:
            raise ValueError("mode 'set' requires a 'value'")
        setattr(target, leaf, to_mxs(params["value"]))
        return {"set": path}

    if mode != "call":
        raise ValueError(f"unknown mode {mode!r}; use call, get or set")

    args = [to_mxs(a) for a in params.get("args", [])]
    kwargs = {k: to_mxs(v) for k, v in (params.get("kwargs") or {}).items()}
    return coerce(getattr(target, leaf)(*args, **kwargs))


def _resolve_node(params: dict):
    node_name = params.get("node")
    if not node_name:
        raise ValueError("'node' is required")
    obj = rt.getNodeByName(str(node_name))
    if obj is None:
        raise ValueError(f"no scene node named {node_name!r}")
    return obj


# Node-level properties do not appear in getPropNames, which lists the base
# object's parameters. Without this allowlist, valid writes get rejected.
_NODE_LEVEL_PROPS = {
    "name", "pos", "position", "rotation", "scale", "wirecolor", "parent",
    "target", "ishidden", "isfrozen", "transform", "boxmode", "renderable",
}


def cmd_node_get(params: dict):
    obj = _resolve_node(params)
    prop = params.get("prop")
    if not isinstance(prop, str) or not prop:
        raise ValueError("'prop' must be a non-empty string")
    if prop.lower() in _NODE_LEVEL_PROPS:
        return coerce(getattr(obj, prop))
    return coerce(rt.getProperty(obj, rt.Name(prop)))


def cmd_node_set(params: dict):
    """
    Write one property on a scene node, verifying it exists first.

    MaxScript silently ignores assignment to an unknown property, which is how a
    misremembered V-Ray parameter becomes a default-lit render with no error
    anywhere to explain it.
    """
    obj = _resolve_node(params)
    prop = params.get("prop")
    if not isinstance(prop, str) or not prop:
        raise ValueError("'prop' must be a non-empty string")

    known = {str(n).lower() for n in rt.getPropNames(obj)}
    if prop.lower() not in known and prop.lower() not in _NODE_LEVEL_PROPS:
        available = sorted(str(n) for n in rt.getPropNames(obj))
        raise ValueError(
            f"{rt.classOf(obj)} has no property {prop!r}. "
            f"Available ({len(available)}): {', '.join(available[:40])}"
        )

    value = to_mxs(params.get("value"))
    if prop.lower() in _NODE_LEVEL_PROPS:
        setattr(obj, prop, value)
    else:
        rt.setProperty(obj, rt.Name(prop), value)
    return {"node": str(obj.name), "prop": prop, "set": True}


def cmd_properties(params: dict) -> dict:
    """
    Introspect an object's or class's properties.

    Exists so parameter names are *discovered* from the live host rather than
    recalled — V-Ray renames and adds sun/sky parameters between versions, and a
    wrong name fails silently as a default value.
    """
    target_name = params.get("node")
    class_name = params.get("class")

    if target_name:
        obj = _resolve_node(params)
    elif class_name:
        cls = getattr(rt, str(class_name), None)
        if cls is None:
            raise ValueError(f"unknown class {class_name!r}")
        obj = cls()
    else:
        raise ValueError("pass either 'node' or 'class'")

    names = [str(n) for n in rt.getPropNames(obj)]
    props = {}
    for name in names:
        try:
            props[name] = coerce(rt.getProperty(obj, rt.Name(name)))
        except Exception as exc:
            props[name] = f"<unreadable: {type(exc).__name__}>"

    result = {
        "class": str(rt.classOf(obj)),
        "superclass": str(rt.superClassOf(obj)),
        "property_count": len(names),
        "properties": props,
    }

    # A class probe instantiates a throwaway object; do not leave it behind.
    if class_name and not target_name:
        try:
            rt.delete(obj)
        except Exception:
            pass
    return result


def cmd_scene_list(params: dict) -> dict:
    """List scene nodes, optionally filtered by class or name prefix."""
    want_class = params.get("class")
    prefix = params.get("prefix")

    nodes = []
    for obj in rt.objects:
        cls = str(rt.classOf(obj))
        name = str(obj.name)
        if want_class and cls.lower() != str(want_class).lower():
            continue
        if prefix and not name.startswith(str(prefix)):
            continue
        nodes.append({"name": name, "class": cls, "handle": int(obj.handle)})
    return {"count": len(nodes), "nodes": nodes}


def cmd_vray_sky_setup(params: dict) -> dict:
    """
    Create a VRaySky, put it in the environment slot and bind it to a sun.

    Atomic by necessity. A texmap is **not a scene node**: no name, no handle,
    so unlike a node it cannot be handed back to the client as a stable identity
    and referenced in a follow-up call. The live reference must stay in Max for
    the whole sequence. Materials, modifiers and controllers are the same.
    """
    sun_name = params.get("sun_node")
    sky_class_name = params.get("sky_class", "VRaySky")

    sky_class = getattr(rt, str(sky_class_name), None)
    if sky_class is None:
        raise ValueError(f"unknown sky class {sky_class_name!r}")

    sun = None
    if sun_name:
        sun = rt.getNodeByName(str(sun_name))
        if sun is None:
            raise ValueError(f"no scene node named {sun_name!r}")

    sky = sky_class()
    prop_names = {str(n).lower() for n in rt.getPropNames(sky)}
    applied = {}

    if sun is not None and "sun_node" in prop_names:
        rt.setProperty(sky, rt.Name("sun_node"), sun)
        applied["sun_node"] = str(sun.name)
        if "manual_sun_node" in prop_names:
            rt.setProperty(sky, rt.Name("manual_sun_node"), True)
            applied["manual_sun_node"] = True

    for key, value in (params.get("params") or {}).items():
        if str(key).lower() in prop_names:
            rt.setProperty(sky, rt.Name(str(key)), to_mxs(value))
            applied[str(key)] = coerce(rt.getProperty(sky, rt.Name(str(key))))

    rt.environmentMap = sky
    rt.useEnvironmentMap = True

    return {
        "sky_class": str(rt.classOf(sky)),
        "environment_map_set": True,
        "use_environment_map": bool(rt.useEnvironmentMap),
        "applied": applied,
        "available_params": sorted(str(n) for n in rt.getPropNames(sky)),
    }


def cmd_assign_material(params: dict) -> dict:
    """
    Create a VRayMtl and assign it to one or more nodes.

    Atomic host-side for the same reason as the sky and the renderer: a material
    is not a scene node, so it has no name-or-handle identity that could survive
    a round trip to the client and be referenced in a follow-up call.

    Note that a material's colour parameters (``diffuse``, ``reflection``) do
    **not** appear in ``getPropNames``, which lists only the scripted-plugin
    parameter block. They are plain attributes, so they are set with setattr and
    validated by reading back rather than by a name check.
    """
    nodes = params.get("nodes") or ([params["node"]] if params.get("node") else [])
    if not nodes:
        raise ValueError("pass 'node' or 'nodes'")

    mtl_class_name = params.get("material_class", "VRayMtl")
    mtl_class = getattr(rt, str(mtl_class_name), None)
    if mtl_class is None:
        raise ValueError(f"unknown material class {mtl_class_name!r}")

    mtl = mtl_class()
    if params.get("name"):
        mtl.name = str(params["name"])

    applied = {}
    for key, value in (params.get("params") or {}).items():
        try:
            setattr(mtl, str(key), to_mxs(value))
            applied[str(key)] = coerce(getattr(mtl, str(key)))
        except Exception as exc:
            applied[str(key)] = f"<failed: {type(exc).__name__}: {exc}>"

    assigned = []
    for node_name in nodes:
        node = rt.getNodeByName(str(node_name))
        if node is None:
            raise ValueError(f"no scene node named {node_name!r}")
        node.material = mtl
        assigned.append(str(node.name))

    return {
        "material_class": str(rt.classOf(mtl)),
        "material_name": str(mtl.name),
        "assigned_to": assigned,
        "applied": applied,
    }


def cmd_create_mesh(params: dict) -> dict:
    """
    Build an Editable_Mesh from explicit vertices and faces.

    Atomic host-side, like the sky and the material commands, but for a
    different reason: a mesh *is* a node and could round-trip, yet assembling
    one through generic ``call`` would mean a separate main-thread slot per
    vertex. A single city block is a few thousand vertices, which is a few
    thousand round trips — minutes of stalled UI for something that takes
    milliseconds in one slot.

    Vertices and faces are set with ``setVert``/``setFace`` rather than handed
    to the ``mesh`` constructor as arrays. The constructor form needs MaxScript
    arrays, and whether pymxs marshals a Python list into one is a question
    about this specific pymxs build — the loop needs no such assumption and
    costs nothing extra, because it already runs inside the host.

    **Face indices arriving here are 0-based and are converted once, here.**
    MaxScript is 1-based everywhere. Doing the shift at the boundary rather
    than in the geometry code means it cannot be applied twice, or not at all —
    an off-by-one that yields a building with its faces shuffled rather than an
    error.

    The reply includes the vertex count and the bounding box **read back from
    the scene**, not echoed from the request. That is what lets the caller
    detect the unit trap: send metres into a scene still set to inches and the
    box comes back 39.37x too big, with nothing else to indicate it.
    """
    name = str(params.get("name") or "atlas_mesh")
    verts = params.get("verts") or []
    faces = params.get("faces") or []

    if not verts:
        raise ValueError("'verts' is empty")
    if not faces:
        raise ValueError("'faces' is empty")

    vertex_count = len(verts)
    for index, face in enumerate(faces):
        if len(face) != 3:
            raise ValueError(f"face {index} has {len(face)} indices; meshes are triangles")
        for corner in face:
            if not isinstance(corner, int) or corner < 0 or corner >= vertex_count:
                raise ValueError(
                    f"face {index} references vertex {corner}, outside 0..{vertex_count - 1}. "
                    "Indices must be 0-based; the +1 for MaxScript happens here."
                )

    msh = rt.mesh(numverts=vertex_count, numfaces=len(faces))

    for i, vertex in enumerate(verts):
        x, y, z = vertex
        rt.setVert(msh, i + 1, rt.Point3(float(x), float(y), float(z)))

    for i, face in enumerate(faces):
        a, b, c = face
        rt.setFace(msh, i + 1, rt.Point3(a + 1, b + 1, c + 1))
        # Smoothing group 0 = faceted. A building is flat planes meeting at hard
        # corners; smoothing them averages the normals across the roof edge and
        # gives every block a soft, inflated silhouette in the render.
        rt.setFaceSmoothGroup(msh, i + 1, 0)

    msh.name = name
    if params.get("wirecolor"):
        r, g, b = params["wirecolor"]
        msh.wirecolor = rt.Color(float(r), float(g), float(b))

    rt.update(msh)

    low, high = msh.min, msh.max
    return {
        "node": str(msh.name),
        "handle": int(msh.handle),
        "class": str(rt.classOf(msh)),
        "requested_verts": vertex_count,
        "requested_faces": len(faces),
        # Read back from the scene — the point of the exercise.
        "vertex_count": int(rt.getNumVerts(msh)),
        "face_count": int(rt.getNumFaces(msh)),
        "bbox_min": [float(low.x), float(low.y), float(low.z)],
        "bbox_max": [float(high.x), float(high.y), float(high.z)],
        "units": str(rt.units.SystemType),
    }


def cmd_list_renderers(_params: dict) -> dict:
    """Enumerate installed renderer classes and report which slot holds what."""
    classes = [str(c) for c in rt.RendererClass.classes]
    return {
        "available": classes,
        "current": {
            "production": str(rt.renderers.production),
            "medit": str(rt.renderers.medit),
            "activeShade": str(rt.renderers.activeShade),
        },
    }


def cmd_set_renderer(params: dict) -> dict:
    """
    Set the production (and optionally ActiveShade) renderer.

    Atomic host-side, because a renderer is not a scene node and cannot be
    handed back to the client and reassigned in a second call.

    This matters more than it looks: **resetMaxFile resets the renderer to the
    application default**. A scene-building routine that resets the file will
    silently revert to Arnold, and a V-Ray sun and sky then contribute nothing
    to the render with no error raised anywhere.
    """
    name = params.get("renderer")
    if not name:
        raise ValueError("'renderer' is required")

    available = [str(c) for c in rt.RendererClass.classes]
    match = next((c for c in available if c.lower() == str(name).lower()), None)
    if match is None:
        # Allow a prefix match so callers need not carry the exact build suffix.
        candidates = [c for c in available if c.lower().startswith(str(name).lower())]
        if len(candidates) == 1:
            match = candidates[0]
        elif not candidates:
            raise ValueError(
                f"no renderer matching {name!r}. Available: {', '.join(available)}"
            )
        else:
            raise ValueError(
                f"{name!r} is ambiguous — matches {', '.join(candidates)}"
            )

    cls = getattr(rt, match, None)
    if cls is None:
        raise ValueError(f"renderer class {match!r} is not constructible")

    rt.renderers.production = cls()
    applied = {"production": str(rt.renderers.production)}

    if params.get("also_activeshade", True):
        try:
            rt.renderers.activeShade = cls()
            applied["activeShade"] = str(rt.renderers.activeShade)
        except Exception as exc:
            applied["activeShade_error"] = str(exc)

    if params.get("also_medit", False):
        try:
            rt.renderers.medit = cls()
            applied["medit"] = str(rt.renderers.medit)
        except Exception as exc:
            applied["medit_error"] = str(exc)

    return {"requested": name, "resolved": match, "applied": applied}


def cmd_render(params: dict) -> dict:
    """
    Render the active or a named camera to a file.

    Rendering holds the main thread for its whole duration, so every other
    command queues behind it — callers must pass a generous timeout.
    """
    path = params.get("path")
    if not path:
        raise ValueError("'path' is required")
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Refuse to render under an unexpected renderer. resetMaxFile silently
    # reverts to the application default, and a V-Ray-lit scene rendered in
    # Arnold produces a featureless image with no error to explain it.
    expect = params.get("expect_renderer")
    if expect:
        actual = str(rt.renderers.production)
        if not actual.lower().startswith(str(expect).lower()):
            raise RuntimeError(
                f"production renderer is {actual!r}, expected something starting "
                f"with {expect!r}. A V-Ray sun and sky contribute nothing to an "
                f"Arnold render. Did a resetMaxFile revert it?"
            )

    kwargs = {
        "outputwidth": int(params.get("width", 640)),
        "outputheight": int(params.get("height", 360)),
        "vfb": bool(params.get("vfb", False)),
        "outputFile": path,
    }
    cam_name = params.get("camera")
    if cam_name:
        cam = rt.getNodeByName(str(cam_name))
        if cam is None:
            raise ValueError(f"no camera named {cam_name!r}")
        kwargs["camera"] = cam

    rt.render(**kwargs)

    if not os.path.exists(path):
        raise RuntimeError(f"render reported success but {path} does not exist")
    return {"path": path, "bytes": os.path.getsize(path), **{
        k: v for k, v in kwargs.items() if k != "camera" and k != "outputFile"
    }}


def cmd_viewport_capture(params: dict) -> dict:
    """
    Save a viewport grab to disk.

    Return values cannot tell you whether a result *looks* right. A multimodal
    model checking its own framing and shadow direction catches errors no
    assertion does.
    """
    path = params.get("path")
    if not path:
        raise ValueError("'path' is required")
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    rt.completeRedraw()
    bmp = rt.gw.getViewportDib()
    if bmp is None:
        raise RuntimeError("viewport capture returned nothing")
    rt.setProperty(bmp, rt.Name("filename"), path)
    rt.save(bmp)
    rt.close(bmp)

    if not os.path.exists(path):
        raise RuntimeError(f"viewport capture did not produce {path}")
    return {"path": path, "bytes": os.path.getsize(path)}


def cmd_maxscript(params: dict):
    if not ALLOW_MAXSCRIPT:
        raise PermissionError(
            "Raw MaxScript is disabled. Set ATLAS_ALLOW_MAXSCRIPT=1 in the "
            "environment before launching 3ds Max to enable it."
        )
    code = params.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("'code' must be a non-empty string")
    return coerce(rt.execute(code))


HANDLERS = {
    "ping": cmd_ping,
    "call": cmd_call,
    "node_get": cmd_node_get,
    "node_set": cmd_node_set,
    "properties": cmd_properties,
    "scene_list": cmd_scene_list,
    "vray_sky_setup": cmd_vray_sky_setup,
    "assign_material": cmd_assign_material,
    "create_mesh": cmd_create_mesh,
    "list_renderers": cmd_list_renderers,
    "set_renderer": cmd_set_renderer,
    "render": cmd_render,
    "viewport_capture": cmd_viewport_capture,
    "maxscript": cmd_maxscript,
}


def dispatch(command: str, params: dict):
    handler = HANDLERS.get(command)
    if handler is None:
        raise ValueError(
            f"unknown command {command!r}; known: {', '.join(sorted(HANDLERS))}"
        )
    return handler(params)


def run_job(payload: dict):
    """Execute one request. Called on the main thread."""
    command = payload.get("command")

    if command == "batch":
        # Several calls in one main-thread slot are not interleaved with UI
        # events, so a read-modify-write sequence sees a consistent scene. This
        # is a correctness property, not just a speed optimisation.
        results = []
        for i, step in enumerate(payload.get("steps", [])):
            try:
                results.append({"ok": True, "result": dispatch(step.get("command"), step)})
            except Exception as exc:
                results.append(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}", "step": i}
                )
                if payload.get("stop_on_error", True):
                    break
        return {"steps": results}

    return dispatch(command, payload)
