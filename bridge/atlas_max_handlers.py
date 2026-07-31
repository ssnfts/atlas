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
    parameter block. They are plain attributes reached with setattr.

    **Reading back after setattr does not validate them.** setattr on a pymxs
    wrapper creates an ordinary Python attribute for any name at all, and
    getattr then returns it, so an invented parameter reported itself as
    applied while V-Ray quietly used the default — measured: a probe with
    ``bogus_param_xyz`` came back OK. Existence is therefore checked with
    hasattr *before* assigning, which is the one thing that distinguishes a
    real V-Ray attribute from a typo.
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

    # Names the scripted parameter block does declare. Colour attributes are not
    # among them, hence the hasattr check below rather than only this set.
    block_params = {str(n).lower() for n in rt.getPropNames(mtl)}

    applied = {}
    rejected = {}
    for key, value in (params.get("params") or {}).items():
        name = str(key)
        if not hasattr(mtl, name) and name.lower() not in block_params:
            rejected[name] = (
                f"{rt.classOf(mtl)} has no attribute {name!r}. Discover the real "
                "name from the live material rather than recalling it."
            )
            continue
        try:
            setattr(mtl, name, to_mxs(value))
            applied[name] = coerce(getattr(mtl, name))
        except Exception as exc:
            rejected[name] = f"{type(exc).__name__}: {exc}"

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
        "rejected": rejected,
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

    # Smoothing group 0 = faceted, and it is the right default: a building is
    # flat planes meeting at hard corners, and smoothing them averages the
    # normals across the roof edge, giving every block a soft inflated
    # silhouette. But it is wrong for anything actually curved — a 16-sided
    # wheel shaded faceted reads as an octagon, which is what made the car
    # proxies look like stacked boxes. Callers with curved geometry pass 1.
    smooth_group = int(params.get("smooth") or 0)

    for i, face in enumerate(faces):
        a, b, c = face
        rt.setFace(msh, i + 1, rt.Point3(a + 1, b + 1, c + 1))
        rt.setFaceSmoothGroup(msh, i + 1, smooth_group)

    # Texture coordinates, one per vertex, sharing the face list.
    #
    # Without these a procedural map can only be projected in *world* space,
    # which is fine for a wall of concrete and useless for anything that has to
    # follow a shape: kerb stripes must run along a kerb as it bends, and an
    # edge line must stay parallel to the edge. Both need a coordinate that
    # travels with the surface, which is what a UV is.
    #
    # `buildTVFaces` must come *after* the tverts exist and *before* setTVFace,
    # and it resets the map face list — calling it later wipes what was set.
    uvs = params.get("uvs")
    uv_report: dict = {}
    if uvs:
        if len(uvs) != vertex_count:
            raise ValueError(
                f"'uvs' has {len(uvs)} entries for {vertex_count} vertices; "
                "one texture coordinate per vertex is required"
            )
        rt.setNumTVerts(msh, vertex_count)
        for i, uv in enumerate(uvs):
            u, v = (list(uv) + [0.0])[:2]
            rt.setTVert(msh, i + 1, rt.Point3(float(u), float(v), 0.0))
        rt.buildTVFaces(msh)
        for i, face in enumerate(faces):
            a, b, c = face
            rt.setTVFace(msh, i + 1, rt.Point3(a + 1, b + 1, c + 1))
        # Read back rather than assume: a mesh with tverts set but no TVFaces
        # renders untextured with nothing in any log to say so.
        uv_report = {
            "tvert_count": int(rt.getNumTVerts(msh)),
            "tvface_count": int(rt.meshop.getNumMapFaces(msh, 1)),
        }

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
        **uv_report,
    }


def cmd_set_keys(params: dict) -> dict:
    """
    Key a node's position and Z rotation over a list of frames.

    Until now a sequence from this project was a series of stills: move
    everything, render, repeat. That is restartable and needs nothing stored in
    the scene, but it leaves nothing *in* the scene either — the artist who
    opens the file cannot scrub it, cannot retime it, and cannot hand it to a
    render farm as an animation. Real keys fix that.

    Rotation is a single angle about Z rather than a full matrix on purpose.
    Everything this drives — a car on a circuit, a drone over it — is upright
    and turning in plan, and a scalar heading is a value a human can read back
    and check. A matrix per key is unreadable, and this project has already lost
    a day to an orientation that was wrong in a way nobody could see.

    ``pymxs.animate`` is what makes an assignment create a key instead of just
    setting a value. Without it every write lands on frame 0 and the node sits
    still, with nothing to say why.
    """
    name = str(params.get("node") or "")
    node = rt.getNodeByName(name)
    if node is None:
        raise ValueError(f"no node named {name!r}")

    keys = params.get("keys") or []
    if not keys:
        raise ValueError("'keys' is empty")

    with pymxs.animate(True):
        for key in keys:
            frame = float(key["frame"])
            with pymxs.attime(frame):
                # **Rotation first, then position, and the order is load
                # bearing.** Assigning `.rotation` rotates the node's whole
                # transform, translation included: keying pos=(100,100) and then
                # a 90 degree heading put the node at (-100,100). Measured, not
                # reasoned — pos alone was right, pos with a zero heading was
                # right, and only a non-zero heading moved it. Setting the
                # rotation first and the position after leaves the translation
                # as the last word.
                # A quaternion, composed by the caller, is the preferred form.
                # Max's Euler convention produced a *mirrored* yaw here — a car
                # keyed at heading 90 pointed west — and applied pitch and roll
                # about the world axes rather than the body's, so a 3 degree
                # roll at heading 90 came out as pitch. Both were invisible near
                # heading 0 and wrong everywhere else. Sending an explicit
                # quaternion moves that convention into Python, where it is
                # arithmetic and can be tested without a running 3ds Max.
                quat = key.get("quat")
                if quat is not None:
                    w, qx, qy, qz = quat
                    node.rotation = rt.quat(float(qx), float(qy), float(qz), float(w))
                    position = key.get("pos")
                    if position is not None:
                        x, y, z = position
                        node.pos = rt.Point3(float(x), float(y), float(z))
                    continue

                heading = key.get("heading_deg")
                pitch = key.get("pitch_deg")
                roll = key.get("roll_deg")
                if heading is not None or pitch is not None or roll is not None:
                    # Scene headings are clockwise from +Y (north); a right
                    # handed Z rotation runs the other way, hence the sign.
                    #
                    # Cars are modelled nose along +Y, so pitch is about X (the
                    # lateral axis) and roll about Y (the nose axis). Euler
                    # composition order would matter for large angles; these are
                    # body attitude on a stiff race car and stay under 3 degrees,
                    # where the difference is far below what a frame shows.
                    euler = rt.EulerAngles(
                        float(pitch or 0.0),
                        float(roll or 0.0),
                        -float(heading or 0.0),
                    )
                    node.rotation = rt.eulerToQuat(euler)
                position = key.get("pos")
                if position is not None:
                    x, y, z = position
                    node.pos = rt.Point3(float(x), float(y), float(z))

    # Read the transform back at the first and last key rather than trusting the
    # write: a node with a position controller that refuses keys (a constraint,
    # a locked track) accepts the assignment and keeps its old value.
    first, last = float(keys[0]["frame"]), float(keys[-1]["frame"])
    middle = float(keys[len(keys) // 2]["frame"])

    samples = []
    for f in (first, middle, last):
        with pymxs.attime(f):
            p = node.pos
            samples.append((float(p.x), float(p.y), float(p.z)))

    # Compared across three samples rather than just the ends, because a closed
    # circuit puts the last key back on the first: a car that ran a whole lap
    # has identical start and end positions, and an end-to-end test calls that
    # "did not move". The first version of this check did exactly that and
    # failed a lap that was keyed perfectly well.
    spread = max(
        abs(a[i] - bb[i])
        for a in samples for bb in samples for i in range(3)
    )

    return {
        "node": name,
        "keys": len(keys),
        "frame_range": [first, last],
        "pos_at_first": list(samples[0]),
        "pos_at_middle": list(samples[1]),
        "pos_at_last": list(samples[2]),
        "max_displacement": round(spread, 6),
        "moved": spread > 1e-6,
    }


def cmd_animation_range(params: dict) -> dict:
    """Set the scene's animation range and frame rate."""
    start = int(params.get("start", 0))
    end = int(params.get("end", 100))
    if end <= start:
        raise ValueError(f"animation range end ({end}) must exceed start ({start})")

    if params.get("fps"):
        rt.frameRate = int(params["fps"])
    rt.animationRange = rt.interval(start, end)

    return {
        "start": int(rt.animationRange.start),
        "end": int(rt.animationRange.end),
        "fps": int(rt.frameRate),
    }


def cmd_save_scene(params: dict) -> dict:
    """
    Save the scene, and confirm from disk that it actually happened.

    This exists because a session's work has now been lost twice to 3ds Max
    exiting between builds. Everything Atlas makes is reproducible from source,
    which is the reason that was survivable — but reproducing it costs the
    minutes a rebuild takes, and an artist's manual edits are not reproducible
    at all.

    ``saveMaxFile`` returns true when it queued the save, not when the bytes
    landed, so the reply carries the file's size and modification time read back
    from disk. A save that silently did nothing is otherwise indistinguishable
    from one that worked.
    """
    path = str(params.get("path") or "")
    if not path:
        raise ValueError("'path' is required")

    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    before = os.path.getmtime(path) if os.path.isfile(path) else 0.0
    ok = rt.saveMaxFile(path, useNewFile=bool(params.get("use_new_file", False)))

    if not os.path.isfile(path):
        raise RuntimeError(f"saveMaxFile reported {ok!r} but {path} does not exist")

    after = os.path.getmtime(path)
    if after <= before:
        raise RuntimeError(
            f"{path} was not rewritten (mtime unchanged). The save did not happen."
        )

    return {
        "path": path,
        "bytes": os.path.getsize(path),
        "objects": int(rt.objects.count),
        "scene": str(rt.maxFileName) or "<unsaved>",
    }


def cmd_vray_hdri_env(params: dict) -> dict:
    """
    Put a VRayHDRI in the environment slot, rotated to a given bearing.

    Atomic host-side for the same reason as the sky: a texmap has no name or
    handle, so it cannot be handed back to the client and referenced later.

    ``horizontal_rotation`` is what makes an HDRI usable in a georeferenced
    scene. A captured sky has a sun baked into it at whatever bearing the
    photographer was standing, and dropping it in unrotated puts that sun
    somewhere unrelated to the one this project computed — two suns, in two
    directions, one of them fictional. Rotating the map so its own sun lands on
    the computed azimuth is what reconciles them.

    Everything is read back, because ``environmentMap`` accepts an assignment
    whether or not ``useEnvironmentMap`` is on, and a scene lit by the default
    grey with a perfectly good HDRI sitting unused in the slot looks like a bad
    HDRI rather than an unticked box.
    """
    path = str(params.get("path") or "")
    if not path or not os.path.isfile(path):
        raise ValueError(f"HDRI not found on the host: {path!r}")

    tex = rt.VRayHDRI()
    applied: dict = {}
    rejected: dict = {}

    wanted = {
        "HDRIMapName": path,
        "maptype": int(params.get("maptype", 2)),   # 2 = spherical
        "horizontalRotation": float(params.get("horizontal_rotation", 0.0)),
        "multiplier": float(params.get("multiplier", 1.0)),
    }
    for key, value in wanted.items():
        if not hasattr(tex, key):
            rejected[key] = "VRayHDRI has no such parameter on this build"
            continue
        try:
            setattr(tex, key, value)
            applied[key] = getattr(tex, key)
        except Exception as exc:
            rejected[key] = f"{type(exc).__name__}: {exc}"

    rt.environmentMap = tex
    rt.useEnvironmentMap = True

    return {
        "applied": applied,
        "rejected": rejected,
        # Read back from the scene, not echoed from the request.
        "environment_map_class": str(rt.classOf(rt.environmentMap)),
        "use_environment_map": bool(rt.useEnvironmentMap),
        "map_name_on_host": str(getattr(rt.environmentMap, "HDRIMapName", "")),
    }


def cmd_build_material(params: dict) -> dict:
    """
    Build a procedural texmap graph and assign the material to nodes.

    Atomic host-side for the same reason as the sky: a texmap is not a scene
    node, so it has no name-or-handle identity that could survive a round trip
    to the client. The whole tree has to be built and wired inside Max.

    Nodes are created in dependency order, then inputs are wired by id. The
    client validates the graph is acyclic before sending, so the topological
    walk here cannot loop — but it counts iterations anyway, because a cycle
    that slipped through would hang Max's main thread and lock the application
    rather than raise.

    **Every write is checked with hasattr first.** Texmap parameter names were
    not verifiable offline, so anything unknown lands in ``rejected`` rather
    than being silently swallowed — the same discipline as cmd_assign_material,
    for the same reason. A non-empty ``rejected`` is the loud failure that
    replaces a quietly untextured render.
    """
    spec = params.get("graph") or {}
    nodes_spec = spec.get("nodes") or []
    if not nodes_spec:
        raise ValueError("'graph' has no nodes")

    node_ids = {n["id"] for n in nodes_spec}
    built: dict = {}
    rejected: dict = {}

    # Create every texmap first, without inputs. Wiring afterwards means the
    # creation order does not have to be topological.
    for entry in nodes_spec:
        cls_name = str(entry["class"])
        cls = getattr(rt, cls_name, None)
        if cls is None:
            rejected[entry["id"]] = f"no texmap class {cls_name!r} on this host"
            continue
        try:
            built[entry["id"]] = cls()
        except Exception as exc:
            # Listed by textureMap.classes is not the same as constructible —
            # Wood and fallofftextureMap are both listed and both fail here.
            rejected[entry["id"]] = f"{cls_name} is not constructible: {exc}"

    applied: dict = {}
    for entry in nodes_spec:
        node = built.get(entry["id"])
        if node is None:
            continue
        known = {str(n).lower() for n in rt.getPropNames(node)}

        for key, value in (entry.get("params") or {}).items():
            name = str(key)
            # Carried in the spec for offline bounds checks, not a host param.
            if name in ("bump_multiplier", "hue_shift", "mode"):
                continue
            if name.lower() not in known and not hasattr(node, name):
                rejected[f"{entry['id']}.{name}"] = f"{entry['class']} has no {name!r}"
                continue
            try:
                _set_texmap_param(node, name, value, known)
                applied[f"{entry['id']}.{name}"] = True
            except Exception as exc:
                rejected[f"{entry['id']}.{name}"] = f"{type(exc).__name__}: {exc}"

        for key, ref in (entry.get("inputs") or {}).items():
            name = str(key)
            target = built.get(str(ref))
            if target is None:
                rejected[f"{entry['id']}.{name}"] = f"input node {ref!r} was not built"
                continue
            if name.lower() not in known and not hasattr(node, name):
                rejected[f"{entry['id']}.{name}"] = f"{entry['class']} has no input {name!r}"
                continue
            try:
                _set_texmap_param(node, name, target, known)
                applied[f"{entry['id']}.{name}"] = f"<- {ref}"
            except Exception as exc:
                rejected[f"{entry['id']}.{name}"] = f"{type(exc).__name__}: {exc}"

    # The material itself: base scalar/colour values, then the texmap channels.
    mtl_class = getattr(rt, str(params.get("material_class", "VRayMtl")), None)
    if mtl_class is None:
        raise ValueError(f"unknown material class {params.get('material_class')!r}")
    mtl = mtl_class()
    if params.get("name"):
        mtl.name = str(params["name"])

    mtl_known = {str(n).lower() for n in rt.getPropNames(mtl)}
    for key, value in (spec.get("base_params") or {}).items():
        name = str(key)
        if not hasattr(mtl, name) and name.lower() not in mtl_known:
            rejected[f"mtl.{name}"] = f"VRayMtl has no {name!r}"
            continue
        try:
            setattr(mtl, name, to_mxs(value))
            applied[f"mtl.{name}"] = True
        except Exception as exc:
            rejected[f"mtl.{name}"] = f"{type(exc).__name__}: {exc}"

    # Channel slots. The client emits each map with its `_on` flag, which
    # defaults False — a map set without it is attached and never used.
    for slot, value in (spec.get("slot_writes") or {}).items():
        name = str(slot)
        resolved = value
        if isinstance(value, dict) and "__node_ref__" in value:
            resolved = built.get(str(value["__node_ref__"]))
            if resolved is None:
                rejected[name] = f"node {value['__node_ref__']!r} was not built"
                continue
        if not hasattr(mtl, name) and name.lower() not in mtl_known:
            rejected[name] = f"VRayMtl has no slot {name!r}"
            continue
        try:
            if name.lower() in mtl_known:
                rt.setProperty(mtl, rt.Name(name), to_mxs(resolved))
            else:
                setattr(mtl, name, to_mxs(resolved))
            applied[name] = True
        except Exception as exc:
            rejected[name] = f"{type(exc).__name__}: {exc}"

    assigned = []
    for node_name in params.get("nodes") or []:
        obj = rt.getNodeByName(str(node_name))
        if obj is None:
            rejected[f"node:{node_name}"] = "no such scene node"
            continue
        obj.material = mtl
        assigned.append(str(obj.name))

    return {
        "material_name": str(mtl.name),
        "material_class": str(rt.classOf(mtl)),
        "texmaps_built": len(built),
        "texmaps_requested": len(node_ids),
        "assigned_to": len(assigned),
        "applied": len(applied),
        "rejected": rejected,
    }


def _set_texmap_param(node, name: str, value, known: set) -> None:
    """Write through the parameter block when the name is declared there."""
    if name.lower() in known:
        rt.setProperty(node, rt.Name(name), to_mxs(value))
    else:
        setattr(node, name, to_mxs(value))


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
    "vray_hdri_env": cmd_vray_hdri_env,
    "save_scene": cmd_save_scene,
    "set_keys": cmd_set_keys,
    "animation_range": cmd_animation_range,
    "assign_material": cmd_assign_material,
    "create_mesh": cmd_create_mesh,
    "build_material": cmd_build_material,
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
