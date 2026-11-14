"""
Camera orientation tests — no 3ds Max.

A camera created from script has identity rotation and looks straight down its
local -Z, exactly like a VRaySun with no target. Setting a position and a
``target_distance`` does not change that, and nothing errors: the camera simply
points at the ground.

That defect sat in ``build_camera`` unnoticed, because the project's only camera
was ``demo_shadow_test``'s — a straight-down view of a ground plane, the one
case where the broken orientation is also the correct one. It surfaced the first
time a camera was aimed at anything else, as a completely empty frame.

These stub the bridge so the shape of the conversation with Max can be asserted
without launching it: that a target node is created and assigned, and that the
resulting direction is *verified* rather than assumed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from scene import _transform_z_axis, build_camera  # noqa: E402


class _FakeBridge:
    """
    Records what build_camera does, and reports a transform consistent with a
    camera that really is aimed at its target.

    ``aim`` False makes it report an identity transform instead — a camera
    pointing straight down — which is what the host returned before the fix.
    """

    def __init__(self, *, aim: bool = True, existing: set[str] | None = None):
        self.aim = aim
        self.nodes: set[str] = set(existing or ())
        self.sets: list[tuple[str, str, object]] = []
        self.created: list[str] = []
        self.positions: dict[str, tuple] = {}

    def call(self, func, *args, **kwargs):
        if func == "getNodeByName":
            name = args[0]
            return {"__node__": name} if name in self.nodes else None
        # Any other call is a class constructor.
        self.created.append(func)
        handle = f"{func}_{len(self.created):03d}"
        self.nodes.add(handle)
        return {"__node__": handle}

    def node_set(self, node, prop, value, **_kw):
        self.sets.append((node, prop, value))
        if prop == "name":
            self.nodes.discard(node)
            self.nodes.add(value)
        if prop == "pos" and isinstance(value, dict):
            self.positions[node] = tuple(value["__point3__"])
        return {"node": node, "prop": prop, "set": True}

    def node_get(self, node, prop, **_kw):
        if prop != "transform":
            raise AssertionError(f"unexpected read of {prop!r}")
        if not self.aim:
            # Identity: the camera looks straight down -Z.
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
        camera_pos = self.positions.get(node, (0.0, 0.0, 0.0))
        target_pos = self.positions.get(f"{node}_Target", (0.0, 0.0, 0.0))
        back = [c - t for c, t in zip(camera_pos, target_pos)]
        length = sum(v * v for v in back) ** 0.5 or 1.0
        unit = [v / length for v in back]
        return [[1, 0, 0], [0, 1, 0], unit, list(camera_pos)]

    def properties(self, **_kw):
        return {}


def _build(bridge, **kwargs):
    return build_camera(
        bridge,
        position=kwargs.pop("position", (0.0, -950.0, 420.0)),
        target=kwargs.pop("target", (0.0, 0.0, 60.0)),
        camera_name="Atlas_Cam",
        **kwargs,
    )


# ── Target node ───────────────────────────────────────────────────────────────

def test_camera_gets_a_target_node():
    """Without one the rotation stays identity and the camera faces the ground."""
    bridge = _FakeBridge()
    _build(bridge)
    assert "Targetobject" in bridge.created


def test_target_is_assigned_to_the_camera():
    bridge = _FakeBridge()
    _build(bridge)
    assigned = [v for node, prop, v in bridge.sets if prop == "target"]
    assert assigned, "camera was never given a target"
    assert assigned[0] == {"__node__": "Atlas_Cam_Target"}


def test_target_is_placed_at_the_requested_point():
    bridge = _FakeBridge()
    _build(bridge, target=(10.0, 20.0, 30.0))
    assert bridge.positions["Atlas_Cam_Target"] == (10.0, 20.0, 30.0)


def test_camera_is_not_left_untargeted():
    """
    Regression. The original set `targeted` to False and relied on a transform
    it never wrote — so the camera kept identity rotation and looked straight
    down whatever `target` said.
    """
    bridge = _FakeBridge()
    _build(bridge)
    targeted = [v for _n, prop, v in bridge.sets if prop == "targeted"]
    assert targeted == [] or all(v is True for v in targeted)


def test_existing_target_is_reused_not_duplicated():
    """Re-aiming a shot must update in place, not litter the scene."""
    bridge = _FakeBridge(existing={"Atlas_Cam", "Atlas_Cam_Target"})
    _build(bridge)
    assert "Targetobject" not in bridge.created


def test_existing_camera_is_reused():
    bridge = _FakeBridge(existing={"Atlas_Cam", "Atlas_Cam_Target"})
    _build(bridge)
    assert bridge.created == []


# ── Direction verification ────────────────────────────────────────────────────

def test_direction_is_verified_not_assumed():
    result = _build(_FakeBridge(aim=True))
    assert result["applied"].get("direction_verified") is True


def test_a_camera_pointing_the_wrong_way_is_reported():
    """
    The check has to fail when the host disagrees, or it is decoration. This is
    the exact state the bug produced: correct position, identity rotation.
    """
    result = _build(_FakeBridge(aim=False))
    applied = result["applied"]
    assert "direction_verified" not in applied
    assert "direction_error" in applied
    assert "away from its target" in applied["direction_error"]


def test_reported_error_angle_is_the_real_angle():
    """A camera 950 m north of and 360 m above its target is ~69° off nadir."""
    result = _build(_FakeBridge(aim=False), position=(0.0, -950.0, 420.0),
                    target=(0.0, 0.0, 60.0))
    text = result["applied"]["direction_error"]
    degrees = float(text.split("points")[1].split("°")[0])
    assert degrees == pytest.approx(69.2, abs=1.0)


def test_target_distance_is_the_true_distance():
    result = _build(_FakeBridge(), position=(0.0, 0.0, 100.0), target=(0.0, 0.0, 0.0))
    assert result["applied"]["target_distance"] == pytest.approx(100.0)


def test_straight_down_camera_still_verifies():
    """
    The shadow test's view. It worked with the bug and must keep working
    without it — a fix that only helps oblique cameras would break the
    acceptance test.
    """
    result = _build(_FakeBridge(aim=True), position=(0.0, 0.0, 190.0), target=(0.0, 0.0, 0.0))
    assert result["applied"].get("direction_verified") is True


# ── Transform parsing ─────────────────────────────────────────────────────────

def test_transform_z_axis_reads_the_third_row():
    assert _transform_z_axis([[1, 0, 0], [0, 1, 0], [0, 0, 1], [5, 6, 7]]) == (0.0, 0.0, 1.0)


def test_transform_z_axis_rejects_a_shape_it_does_not_understand():
    """
    Returning None rather than guessing. A misread matrix would produce a
    confident direction check against nonsense.
    """
    assert _transform_z_axis("(matrix3 ...)") is None
    assert _transform_z_axis([[1, 0, 0]]) is None
