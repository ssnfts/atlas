"""
Bridge logic tests that do not require 3ds Max.

``atlas_max_bridge`` imports pymxs and PySide6 at module scope and refuses to
load without them — correct behaviour in production, inconvenient for testing.
Both are stubbed here so the parts that are ordinary Python (value coercion,
command dispatch, batch semantics, argument validation) can be exercised in
milliseconds instead of behind a five-minute application launch.

What this deliberately does NOT cover: main-thread marshalling and anything
touching a real scene. Those need a live host and are verified separately.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# ── Stub host modules before importing the bridge ─────────────────────────────

class _FakeSentinel:
    def __init__(self, label: str):
        self.label = label

    def __repr__(self) -> str:
        return self.label


class _FakePoint3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)


class _FakeNode:
    """Stands in for a MaxScript scene node: has both .name and .handle."""

    def __init__(self, name: str, handle: int, cls: str = "Box"):
        self.name = name
        self.handle = handle
        self._cls = cls


class _FakeRuntime:
    def __init__(self):
        self.undefined = _FakeSentinel("undefined")
        self.OK = _FakeSentinel("ok")
        self.Point3 = _FakePoint3
        self._nodes: dict[str, _FakeNode] = {}
        self.calls: list[tuple] = []
        self.scalar_property = 42

    # -- functions the bridge calls --
    def classOf(self, obj):
        return getattr(obj, "_cls", type(obj).__name__)

    def superClassOf(self, obj):
        return "GeometryClass"

    def getNodeByName(self, name):
        return self._nodes.get(name)

    def maxVersion(self):
        return [29000, 1, 0]

    # -- a callable to dispatch to --
    def box(self, width=1.0, height=1.0, length=1.0):
        self.calls.append(("box", width, height, length))
        node = _FakeNode(f"Box{len(self._nodes) + 1:03d}", 100 + len(self._nodes))
        self._nodes[node.name] = node
        return node

    def boom(self):
        raise RuntimeError("deliberate failure")

    @property
    def objects(self):
        return list(self._nodes.values())


def _install_stubs():
    fake_pymxs = types.ModuleType("pymxs")
    fake_pymxs.runtime = _FakeRuntime()
    sys.modules["pymxs"] = fake_pymxs

    if "PySide6" not in sys.modules:
        pyside = types.ModuleType("PySide6")
        qtcore = types.ModuleType("PySide6.QtCore")

        class _QTimer:
            def __init__(self, *_a, **_k):
                self._interval = 0

            def setInterval(self, ms):
                self._interval = ms

            @property
            def timeout(self):
                class _Sig:
                    def connect(self, _fn):
                        return None

                return _Sig()

            def start(self):
                return None

            def stop(self):
                return None

        qtcore.QTimer = _QTimer
        pyside.QtCore = qtcore
        sys.modules["PySide6"] = pyside
        sys.modules["PySide6.QtCore"] = qtcore


_install_stubs()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bridge"))

# Only the handlers module is imported. The bridge core owns a socket and a
# QTimer and is exercised against a live host instead.
import atlas_max_handlers as handlers  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_runtime(monkeypatch):
    """Give every test a clean fake runtime."""
    rt = _FakeRuntime()
    monkeypatch.setattr(handlers, "rt", rt)
    return rt


# ── Value coercion ────────────────────────────────────────────────────────────

def test_coerce_passes_through_primitives():
    for value in (None, True, False, 0, -3, 2.5, "text", ""):
        assert handlers.coerce(value) == value


def test_coerce_undefined_and_ok_become_none(fresh_runtime):
    assert handlers.coerce(fresh_runtime.undefined) is None
    assert handlers.coerce(fresh_runtime.OK) is None


def test_coerce_point3_becomes_list():
    assert handlers.coerce(_FakePoint3(1, 2, 3)) == [1.0, 2.0, 3.0]


def test_coerce_node_returns_identity_not_wrapper():
    """A live node must never cross the wire; only a stable identity does."""
    out = handlers.coerce(_FakeNode("Wall_01", 77, cls="Editable_Poly"))
    assert out == {"__node__": "Wall_01", "handle": 77, "class": "Editable_Poly"}


def test_coerce_nested_structures():
    out = handlers.coerce({"a": [1, _FakePoint3(0, 0, 1)], "b": (2, "x")})
    assert out == {"a": [1, [0.0, 0.0, 1.0]], "b": [2, "x"]}


def test_coerce_degrades_unknown_type_rather_than_raising():
    """A partial result beats losing completed work to a serialization error."""

    class Opaque:
        def __str__(self):
            return "<opaque thing>"

    assert handlers.coerce(Opaque()) == "<opaque thing>"


def test_coerce_is_depth_limited():
    """A self-referencing structure must terminate, not blow the stack."""
    cyclic: list = [1]
    cyclic.append(cyclic)
    assert "<max depth>" in repr(handlers.coerce(cyclic))


# ── JSON -> MaxScript conversion ──────────────────────────────────────────────

def test_to_mxs_builds_point3(fresh_runtime):
    p = handlers.to_mxs({"__point3__": [1, 2, 3]})
    assert isinstance(p, _FakePoint3) and (p.x, p.y, p.z) == (1.0, 2.0, 3.0)


def test_to_mxs_resolves_named_node(fresh_runtime):
    node = _FakeNode("Ground", 5)
    fresh_runtime._nodes["Ground"] = node
    assert handlers.to_mxs({"__node__": "Ground"}) is node


def test_to_mxs_rejects_missing_node(fresh_runtime):
    with pytest.raises(ValueError, match="no scene node"):
        handlers.to_mxs({"__node__": "DoesNotExist"})


# ── Dispatch ──────────────────────────────────────────────────────────────────

def test_call_invokes_function_with_kwargs(fresh_runtime):
    out = handlers.dispatch(
        "call", {"func": "box", "mode": "call", "kwargs": {"width": 4, "height": 2}}
    )
    assert out["__node__"].startswith("Box")
    assert fresh_runtime.calls == [("box", 4, 2, 1.0)]


def test_get_reads_a_value_without_invoking_it(fresh_runtime):
    """
    Regression: mode used to be inferred from callable(), but pymxs value
    wrappers report as callable. Reading units.SystemType therefore tried to
    *invoke* #inches. Read and call must be distinguished by the caller.
    """
    assert handlers.dispatch("call", {"func": "scalar_property", "mode": "get"}) == 42


def test_get_does_not_call_a_callable(fresh_runtime):
    """A callable read with mode 'get' is returned, not invoked."""
    handlers.dispatch("call", {"func": "box", "mode": "get"})
    assert fresh_runtime.calls == [], "mode 'get' must never invoke"


def test_set_writes_property(fresh_runtime):
    handlers.dispatch("call", {"func": "scalar_property", "mode": "set", "value": 7})
    assert fresh_runtime.scalar_property == 7


def test_set_without_value_is_rejected(fresh_runtime):
    with pytest.raises(ValueError, match="requires a 'value'"):
        handlers.dispatch("call", {"func": "scalar_property", "mode": "set"})


def test_unknown_mode_is_rejected(fresh_runtime):
    with pytest.raises(ValueError, match="unknown mode"):
        handlers.dispatch("call", {"func": "scalar_property", "mode": "frobnicate"})


def test_default_mode_is_call(fresh_runtime):
    handlers.dispatch("call", {"func": "box"})
    assert len(fresh_runtime.calls) == 1


def test_call_rejects_private_attribute():
    with pytest.raises(ValueError, match="private"):
        handlers.dispatch("call", {"func": "_nodes", "mode": "get"})


def test_call_rejects_private_attribute_in_dotted_path():
    with pytest.raises(ValueError, match="private"):
        handlers.dispatch("call", {"func": "_secret.thing", "mode": "get"})


def test_call_rejects_empty_func():
    with pytest.raises(ValueError, match="non-empty"):
        handlers.dispatch("call", {"func": ""})


def test_call_reports_unknown_function(fresh_runtime):
    with pytest.raises(AttributeError, match="no 'nope'"):
        handlers.dispatch("call", {"func": "nope"})


def test_unknown_command_lists_known_ones():
    with pytest.raises(ValueError, match="unknown command"):
        handlers.dispatch("frobnicate", {})


# ── Array indexing ────────────────────────────────────────────────────────────

def test_indexed_access_is_one_based(fresh_runtime):
    """MaxScript arrays start at 1, so objects[1] is the first node."""
    fresh_runtime.box()
    fresh_runtime.box()
    first = handlers.dispatch("call", {"func": "objects[1]", "mode": "get"})
    second = handlers.dispatch("call", {"func": "objects[2]", "mode": "get"})
    assert first["__node__"] == "Box001"
    assert second["__node__"] == "Box002"


def test_indexed_access_supports_dotted_suffix(fresh_runtime):
    """objects[1].name must resolve through the index to the property."""
    fresh_runtime.box()
    assert handlers.dispatch("call", {"func": "objects[1].name", "mode": "get"}) == "Box001"


def test_index_zero_is_rejected(fresh_runtime):
    fresh_runtime.box()
    with pytest.raises(ValueError, match="1-based"):
        handlers.dispatch("call", {"func": "objects[0]", "mode": "get"})


def test_index_out_of_range_is_reported(fresh_runtime):
    fresh_runtime.box()
    with pytest.raises(IndexError, match="out of range"):
        handlers.dispatch("call", {"func": "objects[9]", "mode": "get"})


# ── MaxScript gate ────────────────────────────────────────────────────────────

def test_raw_maxscript_is_disabled_by_default(monkeypatch):
    monkeypatch.setattr(handlers, "ALLOW_MAXSCRIPT", False)
    with pytest.raises(PermissionError, match="ATLAS_ALLOW_MAXSCRIPT"):
        handlers.dispatch("maxscript", {"code": "delete objects"})


# ── Batch semantics ───────────────────────────────────────────────────────────

def test_batch_runs_steps_in_order(fresh_runtime):
    out = handlers.run_job(
        {
            "command": "batch",
            "steps": [
                {"command": "call", "mode": "call", "func": "box", "kwargs": {"width": 1}},
                {"command": "call", "mode": "call", "func": "box", "kwargs": {"width": 2}},
            ],
        }
    )
    assert [s["ok"] for s in out["steps"]] == [True, True]
    assert [c[1] for c in fresh_runtime.calls] == [1, 2]


def test_batch_stops_on_error_by_default(fresh_runtime):
    out = handlers.run_job(
        {
            "command": "batch",
            "steps": [
                {"command": "call", "func": "boom"},
                {"command": "call", "func": "box"},
            ],
        }
    )
    assert len(out["steps"]) == 1
    assert out["steps"][0]["ok"] is False
    assert "deliberate failure" in out["steps"][0]["error"]
    assert fresh_runtime.calls == []


def test_batch_can_continue_past_errors(fresh_runtime):
    out = handlers.run_job(
        {
            "command": "batch",
            "stop_on_error": False,
            "steps": [
                {"command": "call", "func": "boom"},
                {"command": "call", "func": "box"},
            ],
        }
    )
    assert [s["ok"] for s in out["steps"]] == [False, True]
    assert len(fresh_runtime.calls) == 1


def test_batch_error_records_step_index(fresh_runtime):
    out = handlers.run_job(
        {
            "command": "batch",
            "stop_on_error": False,
            "steps": [
                {"command": "call", "func": "box"},
                {"command": "call", "func": "boom"},
            ],
        }
    )
    assert out["steps"][1]["step"] == 1


# ── Job object ────────────────────────────────────────────────────────────────

def test_handlers_module_has_no_socket_state():
    """
    The handlers module must stay free of transport state.

    The core hot-reloads it between requests; anything owning a socket, a queue
    or a QTimer would be recreated mid-flight and either leak the bound port or
    silently stop delivering results.
    """
    forbidden = {"socket", "queue", "threading", "QtCore"}
    present = forbidden & set(vars(handlers))
    assert not present, f"handlers must not hold transport state: {present}"
