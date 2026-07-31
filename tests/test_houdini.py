"""
Tests for houdini.py — offline only, no Houdini required.

The generators emit Python that runs inside the host, so the strongest test
available here is to *run it* against a stub ``hou`` and feed the result back
through the parser. A generated script that is subtly malformed, or a parser
that disagrees with its own generator, is the silent-wrong-output failure this
project exists to refuse, and neither shows up in a substring assertion.
"""

from __future__ import annotations

import io
import json
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import houdini  # noqa: E402


# ── A stub host ───────────────────────────────────────────────────────────────

class _Node:
    def __init__(self, path, children=()):
        self._path = path
        self._children = list(children)

    def path(self):
        return self._path

    def children(self):
        return list(self._children)


def _stub_hou(*, target_written=True, raises=None, apprentice=True,
              unsaved_after=False, fps=24.0, frame_range=(0.0, 1937.0),
              playback_range=None, scene_path="C:/scenes/atlas.hipnc",
              contents=b"x" * 512):
    """
    Build a fake ``hou`` module.

    ``target_written`` False models the case this module exists to catch: save()
    returns without complaint and nothing reaches the disk.
    """
    hou = types.ModuleType("hou")
    state = {"saved_to": None}

    def save(file_name=None, save_to_recent_files=True):
        state["saved_to"] = file_name
        if raises is not None:
            raise raises
        if target_written:
            Path(file_name).write_bytes(contents)

    hou.hipFile = types.SimpleNamespace(
        save=save,
        path=lambda: scene_path,
        hasUnsavedChanges=lambda: unsaved_after,
    )
    hou.isApprentice = lambda: apprentice
    hou.fps = lambda: fps
    hou.setFps = lambda v, **kw: state.__setitem__("fps", v)
    hou.playbar = types.SimpleNamespace(
        frameRange=lambda: tuple(frame_range),
        playbackRange=lambda: tuple(playback_range or frame_range),
        setFrameRange=lambda a, b: state.__setitem__("range", (a, b)),
        setPlaybackRange=lambda a, b: state.__setitem__("play", (a, b)),
    )

    tree = {
        "/obj": _Node("/obj", [_Node("/obj/smoke_sim"), _Node("/obj/track")]),
        "/out": _Node("/out", [_Node("/out/vdb_cache")]),
    }
    hou.node = lambda p: tree.get(p)
    hou.state = state
    return hou


def _run(script, hou):
    """Execute a generated script against a stub host, returning its stdout."""
    saved = sys.modules.get("hou")
    sys.modules["hou"] = hou
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(script, "<generated>", "exec"), {"__name__": "__atlas__"})
        return buf.getvalue()
    finally:
        if saved is None:
            sys.modules.pop("hou", None)
        else:
            sys.modules["hou"] = saved


# ── check_save_path ───────────────────────────────────────────────────────────

def test_apprentice_may_not_save_hip():
    """The case that aborted the live MCP connection, refused in Python."""
    with pytest.raises(ValueError, match="Apprentice"):
        houdini.check_save_path("C:/scenes/atlas.hip", apprentice=True)


def test_apprentice_may_not_save_hiplc():
    with pytest.raises(ValueError, match="Apprentice"):
        houdini.check_save_path("C:/scenes/atlas.hiplc", apprentice=True)


def test_apprentice_accepts_hipnc():
    got = houdini.check_save_path(r"C:\scenes\atlas.hipnc", apprentice=True)
    assert got == "C:/scenes/atlas.hipnc"


def test_commercial_accepts_hip():
    assert houdini.check_save_path("/s/a.hip", apprentice=False) == "/s/a.hip"


def test_unknown_suffix_refused():
    with pytest.raises(ValueError, match="Houdini scene file"):
        houdini.check_save_path("C:/scenes/atlas.max", apprentice=True)


def test_no_suffix_refused():
    with pytest.raises(ValueError, match="no suffix"):
        houdini.check_save_path("C:/scenes/atlas", apprentice=True)


def test_empty_path_refused():
    with pytest.raises(ValueError, match="required"):
        houdini.check_save_path("   ", apprentice=True)


def test_save_script_refuses_bad_path_before_generating():
    with pytest.raises(ValueError):
        houdini.save_script("C:/scenes/atlas.hip", apprentice=True)


# ── the generated save script, actually executed ──────────────────────────────

def test_save_round_trip(tmp_path):
    """Generate, run against a stub, parse. The three must agree."""
    target = tmp_path / "atlas.hipnc"
    target.write_bytes(b"old")
    old_mtime = target.stat().st_mtime
    os.utime(target, (old_mtime - 10, old_mtime - 10))

    script = houdini.save_script(str(target), label="after track import")
    out = _run(script, _stub_hou(scene_path=str(target).replace("\\", "/")))
    reply = houdini.parse_save_reply(out, expect_label="after track import")

    assert reply["label"] == "after track import"
    assert reply["bytes"] == 512
    assert reply["mtime_after"] > reply["mtime_before"]
    assert reply["nodes"]["obj"] == 2
    assert reply["nodes"]["out"] == 1
    assert reply["node_paths"] == ["/obj/smoke_sim", "/obj/track", "/out/vdb_cache"]
    assert reply["frame_range"] == [0.0, 1937.0]


def test_save_catches_a_write_that_never_happened(tmp_path):
    """save() returns quietly, nothing reaches disk. This is the whole point."""
    target = tmp_path / "atlas.hipnc"
    script = houdini.save_script(str(target))
    out = _run(script, _stub_hou(target_written=False))
    with pytest.raises(RuntimeError, match="does not exist"):
        houdini.parse_save_reply(out)


def test_save_catches_a_stale_file(tmp_path):
    """The file exists but was not rewritten — mtime is the only witness."""
    target = tmp_path / "atlas.hipnc"
    target.write_bytes(b"stale")
    script = houdini.save_script(str(target))
    out = _run(script, _stub_hou(target_written=False))
    with pytest.raises(RuntimeError, match="not rewritten"):
        houdini.parse_save_reply(out)


def test_save_catches_an_empty_write(tmp_path):
    target = tmp_path / "atlas.hipnc"
    script = houdini.save_script(str(target))
    out = _run(script, _stub_hou(contents=b""))
    with pytest.raises(RuntimeError, match="empty"):
        houdini.parse_save_reply(out)


def test_save_reports_a_host_exception(tmp_path):
    target = tmp_path / "atlas.hipnc"
    script = houdini.save_script(str(target))
    out = _run(script, _stub_hou(raises=RuntimeError("permission denied")))
    with pytest.raises(RuntimeError, match="permission denied"):
        houdini.parse_save_reply(out)


def test_save_catches_a_session_still_dirty(tmp_path):
    target = tmp_path / "atlas.hipnc"
    script = houdini.save_script(str(target))
    out = _run(script, _stub_hou(unsaved_after=True))
    with pytest.raises(RuntimeError, match="unsaved changes"):
        houdini.parse_save_reply(out)


def test_save_catches_a_label_mismatch(tmp_path):
    target = tmp_path / "atlas.hipnc"
    script = houdini.save_script(str(target), label="step one")
    out = _run(script, _stub_hou())
    with pytest.raises(RuntimeError, match="label mismatch"):
        houdini.parse_save_reply(out, expect_label="step two")


def test_awkward_label_survives_generation(tmp_path):
    """A quote, a brace and a backslash in the label must not break the script."""
    nasty = 'after "smoke" {sim} C:\\path — 100%'
    target = tmp_path / "atlas.hipnc"
    script = houdini.save_script(str(target), label=nasty)
    out = _run(script, _stub_hou())
    assert houdini.parse_save_reply(out, expect_label=nasty)["label"] == nasty


def test_save_creates_a_missing_folder(tmp_path):
    target = tmp_path / "deep" / "nested" / "atlas.hipnc"
    script = houdini.save_script(str(target))
    out = _run(script, _stub_hou())
    assert houdini.parse_save_reply(out)["bytes"] == 512
    assert target.exists()


# ── the generated align script ────────────────────────────────────────────────

def test_align_round_trip():
    script = houdini.align_script(fps=24, start=0, end=1937)
    out = _run(script, _stub_hou())
    reply = houdini.parse_align_reply(out, fps=24, start=0, end=1937)
    assert reply["frame_range"] == [0.0, 1937.0]


def test_align_tolerates_the_playback_epsilon():
    """This host reported 1936.9999999999998 for a range of 1937."""
    hou = _stub_hou(playback_range=(0.0, 1936.9999999999998))
    out = _run(houdini.align_script(fps=24, start=0, end=1937), hou)
    assert houdini.parse_align_reply(out, fps=24, start=0, end=1937)


def test_align_catches_a_range_that_did_not_take():
    """The default 1-240 left in place is a ten-second sim of an 80-second lap."""
    out = _run(houdini.align_script(fps=24, start=0, end=1937),
               _stub_hou(frame_range=(1.0, 240.0)))
    with pytest.raises(RuntimeError, match="frame range is"):
        houdini.parse_align_reply(out, fps=24, start=0, end=1937)


def test_align_catches_a_wrong_fps():
    out = _run(houdini.align_script(fps=24, start=0, end=1937),
               _stub_hou(fps=25.0))
    with pytest.raises(RuntimeError, match="fps is"):
        houdini.parse_align_reply(out, fps=24, start=0, end=1937)


def test_align_catches_a_drifted_playback_range():
    out = _run(houdini.align_script(fps=24, start=0, end=1937),
               _stub_hou(playback_range=(0.0, 240.0)))
    with pytest.raises(RuntimeError, match="playback range"):
        houdini.parse_align_reply(out, fps=24, start=0, end=1937)


def test_align_refuses_a_backwards_range():
    with pytest.raises(ValueError, match="must advance"):
        houdini.align_script(fps=24, start=1937, end=0)


def test_align_refuses_a_zero_fps():
    with pytest.raises(ValueError, match="fps must be positive"):
        houdini.align_script(fps=0, start=0, end=1937)


# ── reply extraction ──────────────────────────────────────────────────────────

def test_reply_survives_surrounding_host_noise(tmp_path):
    target = tmp_path / "atlas.hipnc"
    out = _run(houdini.save_script(str(target)), _stub_hou())
    noisy = f"Warning: something\n{out}\nCode executed successfully."
    assert houdini.parse_save_reply(noisy)["bytes"] == 512


def test_missing_reply_is_not_silently_a_pass():
    """A script that died before its last line must not read as success."""
    with pytest.raises(RuntimeError, match="did not reach its final line"):
        houdini.parse_save_reply("Traceback (most recent call last): boom")


def test_generated_scripts_are_valid_python(tmp_path):
    for script in (houdini.save_script(str(tmp_path / "a.hipnc"), label='q"{}'),
                   houdini.align_script(fps=24, start=0, end=1937)):
        compile(script, "<generated>", "exec")


def test_save_script_embeds_the_path_as_json(tmp_path):
    """A Windows path must not arrive in the host as escape sequences."""
    script = houdini.save_script(r"C:\scenes\new\atlas.hipnc")
    assert json.dumps("C:/scenes/new/atlas.hipnc") in script
