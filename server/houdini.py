"""
Checkpointing the Houdini scene, and refusing to believe it happened.

Houdini is not reached the way 3ds Max is. Max has an Atlas socket bridge that
this package opens directly; Houdini is driven through an MCP tool that only the
agent session can call. So there is nothing here to connect with, and this module
does the two parts that *are* Atlas's job: build the code that runs in the host,
and judge the result it sends back.

The judging is the point. ``hou.hipFile.save()`` raising nothing is Houdini's
*claim* that it saved. The size and modification time read back through ``os``
are the filesystem's *evidence*. Those are two different things, and this project
has already lost work twice to a host that exited between builds — so the
generated script gathers the evidence and :func:`parse_save_reply` refuses the
claim when the evidence does not support it.

What a passing checkpoint proves, exactly: a file exists at that path, it was
rewritten just now, it is not empty, and Houdini no longer considers the scene
dirty. What it does **not** prove is that the file is loadable. Only reopening it
proves that, and reopening it would discard the live session, so this module does
not pretend to know.

One rule is enforced before the host is ever called. An Apprentice licence writes
``.hipnc``, and asking this host to save an Apprentice scene to a ``.hip`` path
aborted the MCP connection outright (``WinError 10053``) rather than returning an
error — verified live, the hard way. A path that the licence cannot write is
therefore rejected here, in Python, where a mistake costs an exception instead of
the session.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePath

__all__ = [
    "APPRENTICE_SUFFIX",
    "KNOWN_SUFFIXES",
    "SENTINEL_BEGIN",
    "SENTINEL_END",
    "check_save_path",
    "save_script",
    "align_script",
    "parse_save_reply",
    "parse_align_reply",
]

# Verified live on this host: hou.isApprentice() is True and the open scene is
# ``atlas.hipnc``. The other two suffixes are the documented commercial and
# Indie forms; this module never asserts which of them a non-Apprentice licence
# will accept, it only declines to guess on Apprentice's behalf.
APPRENTICE_SUFFIX = ".hipnc"
KNOWN_SUFFIXES = (".hip", ".hiplc", ".hipnc")

SENTINEL_BEGIN = "ATLAS_HOU_BEGIN"
SENTINEL_END = "ATLAS_HOU_END"


def check_save_path(path: str, *, apprentice: bool) -> str:
    """
    Normalise a save path, or refuse it before the host sees it.

    Returns the path with forward slashes, which is the form ``hou.hipFile.path``
    reports and the form Houdini accepts on Windows.

    Raises ``ValueError`` when the suffix is unknown, or when an Apprentice
    licence is asked for anything but ``.hipnc``. That second case is not
    defensive tidiness: it aborted the MCP connection on this host.
    """
    if not str(path).strip():
        raise ValueError("a save path is required")

    pure = PurePath(str(path))
    suffix = pure.suffix.lower()

    if suffix not in KNOWN_SUFFIXES:
        raise ValueError(
            f"{path!r} does not look like a Houdini scene file: expected one of "
            f"{', '.join(KNOWN_SUFFIXES)}, got {suffix or '<no suffix>'!r}"
        )

    if apprentice and suffix != APPRENTICE_SUFFIX:
        raise ValueError(
            f"an Apprentice licence cannot save {suffix!r}; use "
            f"{APPRENTICE_SUFFIX!r}. Asking the host to do it anyway aborted the "
            f"connection rather than returning an error."
        )

    return str(pure).replace("\\", "/")


def save_script(path: str, *, label: str = "", apprentice: bool = True) -> str:
    """
    Build the in-host Python that saves the scene and reports the evidence.

    ``label`` is carried through into the reply untouched, so a checkpoint can
    say which step of the work it followed. It is embedded as a JSON string
    rather than interpolated raw, so a quote in the label cannot break the
    script.

    The generated code records the modification time *before* saving, so
    :func:`parse_save_reply` can tell a real write from a call that returned
    quietly and did nothing.
    """
    target = check_save_path(path, apprentice=apprentice)

    return f'''
import hou, os, json

_target = {json.dumps(target)}
_label = {json.dumps(label)}

_folder = os.path.dirname(_target)
if _folder and not os.path.isdir(_folder):
    os.makedirs(_folder, exist_ok=True)

_before = os.path.getmtime(_target) if os.path.isfile(_target) else 0.0

_reply = {{"label": _label, "asked_for": _target, "mtime_before": _before}}

try:
    hou.hipFile.save(_target, save_to_recent_files=False)
    _reply["raised"] = None
except Exception as _exc:
    _reply["raised"] = "{{}}: {{}}".format(type(_exc).__name__, _exc)

# Evidence, gathered through os rather than through hou. A save that silently
# did nothing is otherwise indistinguishable from one that worked.
_reply["exists"] = os.path.isfile(_target)
_reply["bytes"] = os.path.getsize(_target) if _reply["exists"] else 0
_reply["mtime_after"] = os.path.getmtime(_target) if _reply["exists"] else 0.0

# What the file is supposed to contain, so a later reader can tell whether this
# checkpoint captured the work or ran before it.
_reply["path_after"] = hou.hipFile.path()
_reply["unsaved_after"] = hou.hipFile.hasUnsavedChanges()
_reply["apprentice"] = hou.isApprentice()
_reply["fps"] = hou.fps()
_reply["frame_range"] = list(hou.playbar.frameRange())
_reply["nodes"] = {{
    _c: len(hou.node("/" + _c).children()) if hou.node("/" + _c) else 0
    for _c in ("obj", "shop", "out", "ch", "vex", "stage", "mat")
}}
_reply["node_paths"] = sorted(
    _n.path() for _c in ("obj", "out", "stage")
    if hou.node("/" + _c) for _n in hou.node("/" + _c).children()
)

print({json.dumps(SENTINEL_BEGIN)} + json.dumps(_reply) + {json.dumps(SENTINEL_END)})
'''.strip()


def align_script(*, fps: int, start: int, end: int) -> str:
    """
    Build the in-host Python that puts Houdini on Atlas's timeline.

    A Houdini scene opens at 1-240. The Atlas edit is 0-1937 at 24 fps, so a
    simulation left at the default range covers the first ten seconds of an
    eighty-second lap and reports no error at all — the exact failure this
    project is built to catch. Both numbers are set and both are read back.

    ``hou.setFps`` and ``hou.playbar.setFrameRange`` were read off this host's
    own docstrings rather than recalled.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if end <= start:
        raise ValueError(f"frame range must advance, got {start} to {end}")

    return f'''
import hou, json

hou.setFps({int(fps)})
hou.playbar.setFrameRange({int(start)}, {int(end)})
hou.playbar.setPlaybackRange({int(start)}, {int(end)})

_reply = {{
    "fps": hou.fps(),
    "frame_range": list(hou.playbar.frameRange()),
    "playback_range": list(hou.playbar.playbackRange()),
}}
print({json.dumps(SENTINEL_BEGIN)} + json.dumps(_reply) + {json.dumps(SENTINEL_END)})
'''.strip()


def _extract(stdout: str) -> dict:
    """
    Pull the reply out of whatever else the host printed around it.

    Houdini writes warnings to stdout freely and the MCP tool wraps the result in
    its own framing, so the payload is delimited rather than assumed to be the
    whole of it.
    """
    text = str(stdout)
    begin = text.find(SENTINEL_BEGIN)
    end = text.find(SENTINEL_END, begin + 1)
    if begin < 0 or end < 0:
        raise RuntimeError(
            "no Atlas reply in the host output — the script did not reach its "
            f"final line. Output was: {text[:400]!r}"
        )
    blob = text[begin + len(SENTINEL_BEGIN):end]

    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass

    # Some transports hand the payload back still JSON-string-escaped. Undoing
    # that is a matter of applying the exact inverse — decoding it *as* a JSON
    # string — not of replacing escape sequences by hand. Doing it by hand turns
    # a legitimately escaped quote inside a label into a syntax error, which is
    # how this branch came to be written in the first place.
    try:
        return json.loads(json.loads(f'"{blob}"'))
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"the host reply was not readable JSON: {blob[:200]!r}"
        ) from exc


def parse_save_reply(stdout: str, *, expect_label: str | None = None) -> dict:
    """
    Judge a checkpoint, and raise unless the evidence supports it.

    Four things have to hold: the host did not raise, the file exists, it was
    rewritten just now, and Houdini no longer thinks the scene is dirty. The
    third is the one that catches a save that quietly did nothing.
    """
    reply = _extract(stdout)

    if reply.get("raised"):
        raise RuntimeError(f"Houdini refused the save: {reply['raised']}")

    if not reply.get("exists"):
        raise RuntimeError(
            f"{reply.get('asked_for')} does not exist after saving to it"
        )

    if reply.get("bytes", 0) <= 0:
        raise RuntimeError(f"{reply.get('asked_for')} was written empty")

    before = float(reply.get("mtime_before") or 0.0)
    after = float(reply.get("mtime_after") or 0.0)
    if after <= before:
        raise RuntimeError(
            f"{reply.get('asked_for')} was not rewritten (mtime unchanged). "
            f"The save did not happen."
        )

    if reply.get("unsaved_after"):
        raise RuntimeError(
            "Houdini still reports unsaved changes after the save — the file on "
            "disk is behind the session."
        )

    if expect_label is not None and reply.get("label") != expect_label:
        raise RuntimeError(
            f"checkpoint label mismatch: asked for {expect_label!r}, host "
            f"reported {reply.get('label')!r}"
        )

    return reply


def parse_align_reply(stdout: str, *, fps: int, start: int, end: int) -> dict:
    """
    Confirm the timeline landed, comparing against what was asked for.

    Setting a frame range can leave the playback range a hair short — this host
    reported an end of 1936.9999999999998 for a range of 1937 — so the playback
    bound is compared with a tolerance while the global range is required exact.
    A ROP that renders the global range is unaffected; one that follows playback
    would silently drop the last frame.
    """
    reply = _extract(stdout)

    if int(reply.get("fps", 0)) != int(fps):
        raise RuntimeError(f"fps is {reply.get('fps')!r}, asked for {fps}")

    got = [float(v) for v in reply.get("frame_range", [])]
    if got != [float(start), float(end)]:
        raise RuntimeError(
            f"frame range is {got}, asked for [{float(start)}, {float(end)}]"
        )

    play = [float(v) for v in reply.get("playback_range", [])]
    if play and (abs(play[0] - start) > 1e-6 or abs(play[1] - end) > 1e-3):
        raise RuntimeError(
            f"playback range is {play}, too far from [{start}, {end}]"
        )

    return reply
