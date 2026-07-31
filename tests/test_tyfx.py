"""
Tests for tyfx.py — offline only, no 3ds Max required.

All functions that touch the bridge are tested with a mock. The generators
are pure string emitters and are checked against their own output.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import tyfx
from raceanim import CRASH, CrashSpec


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _oval(straight=500.0, radius=45.0, arc_n=40, straight_n=24):
    """A small closed circuit with two straights and two arcs at known radius."""
    pts = []
    half = straight / 2.0
    for i in range(arc_n):
        a = math.pi * i / arc_n
        pts.append((radius * math.cos(a), half + radius * math.sin(a)))
    for i in range(straight_n):
        f = i / straight_n
        pts.append((-radius, half - straight * f))
    for i in range(arc_n):
        a = math.pi + math.pi * i / arc_n
        pts.append((radius * math.cos(a), -half + radius * math.sin(a)))
    for i in range(straight_n):
        f = i / straight_n
        pts.append((radius, -half + straight * f))
    return pts


_SPINE = _oval()


def _mock_bridge(maxscript_return=None):
    b = MagicMock()
    b.maxscript.return_value = maxscript_return
    return b


# ── tyre_smoke stub ───────────────────────────────────────────────────────────

def test_tyre_smoke_returns_complete_false_when_unavailable():
    b = _mock_bridge()
    b.properties.side_effect = Exception("tyFlow not installed")
    result = tyfx.tyre_smoke(b, ["car_01_tyres"])
    assert result["complete"] is False
    assert result["created"] is False


def test_tyre_smoke_complete_false_is_honoured_even_on_success():
    """The stub must still return complete=False after the scaffold call."""
    b = _mock_bridge()
    b.properties.return_value = {"class": "tyFlow", "properties": {}}
    b.call.return_value = {"__node__": "Atlas_TyreSmoke"}
    b.node_set.return_value = None
    b.node_get.return_value = True
    result = tyfx.tyre_smoke(b, ["car_01_tyres"])
    assert result["complete"] is False, "complete must stay False — no events are built"


# ── script_header ─────────────────────────────────────────────────────────────

def test_script_header_starts_with_comment():
    h = tyfx._script_header("Test", {"key": "value"})
    assert h.startswith("-- Atlas generated MaxScript:")


def test_script_header_contains_params():
    h = tyfx._script_header("Smoke", {"wind": "331deg", "end_frame": 2400})
    assert "wind" in h
    assert "331deg" in h
    assert "end_frame" in h


# ── generate_smoke_script ─────────────────────────────────────────────────────

def test_generate_smoke_script_non_empty():
    gate = [True] * 100 + [False] * 100
    script = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_01_tyres", "car_02_tyres"],
        speed_gate_frames=gate,
        wind_bearing_deg=331.0,
        wind_speed_ms=7.34,
        site_z=5.27,
        end_frame=200,
    )
    assert len(script) > 100


def test_generate_smoke_script_starts_with_header():
    gate = [True] * 10
    script = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_01_tyres"],
        speed_gate_frames=gate,
        wind_bearing_deg=0.0,
        wind_speed_ms=5.0,
        site_z=0.0,
        end_frame=10,
    )
    assert script.startswith("-- Atlas generated MaxScript: Tyre Smoke")


def test_generate_smoke_script_uses_verified_tyflow_event_api():
    """Generated smoke uses methods and properties verified in the live host."""
    gate = [False, True, True, False, True]
    script = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_03_tyres"],
        speed_gate_frames=gate,
        wind_bearing_deg=331.0,
        wind_speed_ms=7.34,
        site_z=5.27,
        end_frame=50,
    )
    for keyword in (
        'atlasMakeSmokeEvent tf "Smoke_Birth_01" 1 2',
        'atlasMakeSmokeEvent tf "Smoke_Birth_02" 4 4',
        "ev.setName eventName",
        '.addOperator "Birth" -1',
        '.addOperator "Position Object" -1',
        '.addOperator "Speed" -1',
        '.addOperator "Force" -1',
        '.addOperator "Scale" -1',
        '.addOperator "Time Test" -1',
        '.addOperator "Delete" -1',
        'age.connect death',
        'age.value = 60',
        "birth.birthMode = 1",
        "birth.birthEndEnable = true",
        "position.objectList = emitterNodes",
        "tf.reset_simulation()",
    ):
        assert keyword in script, f"expected '{keyword}' in generated script"
    for obsolete in ("tyBirthFlow", "tyPositionObject", "tySpeed", "tf.Update()"):
        assert obsolete not in script


def test_generate_smoke_script_turns_each_active_gate_range_into_a_birth_event():
    script = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_03_tyres"],
        speed_gate_frames=[False, True, True, False, True, True, True, False],
        wind_bearing_deg=0.0,
        wind_speed_ms=1.0,
        site_z=0.0,
        end_frame=7,
    )
    assert "birth.BirthStart = firstFrame" in script
    assert "birth.BirthEnd = lastFrame" in script
    assert 'atlasMakeSmokeEvent tf "Smoke_Birth_01" 1 2' in script
    assert 'atlasMakeSmokeEvent tf "Smoke_Birth_02" 4 6' in script
    assert script.count('atlasMakeSmokeEvent tf "Smoke_Birth_') == 2


def test_generate_smoke_script_embeds_emitter_names():
    gate = [True] * 10
    script = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_05_tyres", "car_06_tyres"],
        speed_gate_frames=gate,
        wind_bearing_deg=90.0,
        wind_speed_ms=3.0,
        site_z=0.0,
        end_frame=10,
    )
    assert 'car_05_tyres' in script
    assert 'car_06_tyres' in script


def test_generate_smoke_script_wind_direction_is_unit_vector():
    """Wind force direction must be (sin, cos) of bearing+180, not raw bearing."""
    gate = [True] * 10
    script = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_01_tyres"],
        speed_gate_frames=gate,
        wind_bearing_deg=0.0,   # wind from north → blows south → [0, -1, 0]
        wind_speed_ms=10.0,
        site_z=0.0,
        end_frame=10,
    )
    # bearing 0+180=180 → sin(180°)=0, cos(180°)=-1
    assert "0.0" in script       # wx ≈ 0
    assert "-1.0" in script      # wy = -1


def test_generated_event_graph_scripts_scope_their_local_declarations():
    """MaxScript rejects ``local`` declarations at top level."""
    smoke = tyfx.generate_smoke_script(
        "Atlas_TyreSmoke", ["car_01_tyres"],
        speed_gate_frames=[True] * 10,
        wind_bearing_deg=0.0,
        wind_speed_ms=5.0,
        site_z=0.0,
        end_frame=10,
    )
    debris = tyfx.generate_debris_script(
        "Atlas_Debris", ["car_03"],
        contact_frame=10,
        car_speed_ms=50.0,
        yaw_rate_degs=100.0,
        site_z=0.0,
        end_frame=20,
    )
    sparks = tyfx.generate_sparks_script(
        "Atlas_Sparks", ["car_03"],
        contact_frame=10,
        car_speed_ms=50.0,
        site_z=0.0,
        end_frame=20,
    )
    for script in (smoke, debris, sparks):
        assert "\n(\n" in script
        assert script.rstrip().endswith(")")
        assert ".endFrame" not in script


# ── write_smoke_script ────────────────────────────────────────────────────────

def test_write_smoke_script_creates_file(tmp_path):
    result = tyfx.write_smoke_script(
        tmp_path / "tyfx_smoke.ms",
        ["car_01_tyres", "car_02_tyres"],
        _SPINE,
        wind_bearing_deg=331.0,
        wind_speed_ms=7.34,
        site_z=5.27,
        end_frame=200,
    )
    assert Path(result["path"]).is_file()
    assert result["lines"] > 0


def test_write_smoke_script_summary_mentions_threshold(tmp_path):
    result = tyfx.write_smoke_script(
        tmp_path / "smoke.ms",
        ["car_01_tyres"],
        _SPINE,
        speed_threshold_ms=50.0,   # 180 km/h
        end_frame=100,
    )
    assert "180" in result["summary"]   # 50 m/s * 3.6 = 180 km/h


def test_write_smoke_script_params_dict(tmp_path):
    result = tyfx.write_smoke_script(
        tmp_path / "smoke.ms",
        ["car_01_tyres"],
        _SPINE,
        end_frame=100,
    )
    p = result["params"]
    assert "speed_threshold_kmh" in p
    assert "active_frames" in p
    assert "total_frames" in p
    assert p["emitter_count"] == 1


# ── run_smoke_script ──────────────────────────────────────────────────────────

def test_run_smoke_script_raises_if_file_missing():
    b = _mock_bridge()
    with pytest.raises(FileNotFoundError):
        tyfx.run_smoke_script(b, "/nonexistent/tyfx_smoke.ms")


def test_run_smoke_script_raises_on_disabled_flag(tmp_path):
    script = tmp_path / "tyfx_smoke.ms"
    script.write_text("-- test", encoding="utf-8")
    b = _mock_bridge()
    b.maxscript.side_effect = Exception("maxscript disabled in this bridge")
    with pytest.raises(RuntimeError, match="ATLAS_ALLOW_MAXSCRIPT"):
        tyfx.run_smoke_script(b, script)


def test_run_smoke_script_preserves_a_host_compile_error(tmp_path):
    script = tmp_path / "tyfx_smoke.ms"
    script.write_text("local tf = undefined", encoding="utf-8")
    b = _mock_bridge()
    b.maxscript.side_effect = Exception(
        "RuntimeError: MAXScript exception raised. -- Compile error: no local declarations at top level"
    )
    with pytest.raises(Exception, match="Compile error"):
        tyfx.run_smoke_script(b, script)


def test_run_smoke_script_success(tmp_path):
    script = tmp_path / "tyfx_smoke.ms"
    script.write_text("print 'hello'", encoding="utf-8")
    b = _mock_bridge(maxscript_return="hello")
    result = tyfx.run_smoke_script(b, script)
    assert result["executed"] is True
    b.maxscript.assert_called_once()


# ── particle_count ────────────────────────────────────────────────────────────

def test_particle_count_returns_int():
    b = _mock_bridge(maxscript_return=42)
    count = tyfx.particle_count(b, "Atlas_TyreSmoke", 100)
    assert count == 42


def test_particle_count_returns_zero_on_bad_response():
    b = _mock_bridge(maxscript_return=None)
    count = tyfx.particle_count(b, "Atlas_TyreSmoke", 100)
    assert count == 0


def test_particle_count_query_contains_flow_name():
    b = _mock_bridge(maxscript_return=0)
    tyfx.particle_count(b, "MyFlow", 55)
    call_args = b.maxscript.call_args[0][0]
    assert "MyFlow" in call_args
    assert "55" in call_args
    assert call_args.startswith("(\n")
    assert "tf.updateParticles 55" in call_args
    assert "tf.numParticles()" in call_args


# ── contact_frame_for ─────────────────────────────────────────────────────────

def test_contact_frame_for_oval_is_positive():
    """Any crash distance inside the oval returns a positive frame number."""
    spec = CrashSpec(at_distance_m=300.0)
    frame = tyfx.contact_frame_for(_SPINE, spec, fps=24)
    assert frame > 0


def test_contact_frame_for_increases_with_distance():
    """A crash further along the lap happens later."""
    spec_early = CrashSpec(at_distance_m=200.0)
    spec_late = CrashSpec(at_distance_m=600.0)
    f_early = tyfx.contact_frame_for(_SPINE, spec_early, fps=24)
    f_late = tyfx.contact_frame_for(_SPINE, spec_late, fps=24)
    assert f_early < f_late


def test_contact_frame_for_crash_spec_is_integer():
    frame = tyfx.contact_frame_for(_SPINE, CRASH, fps=24)
    assert isinstance(frame, int)


# ── generate_debris_script ────────────────────────────────────────────────────

def test_generate_debris_script_contains_required_operators():
    script = tyfx.generate_debris_script(
        "Atlas_Debris", ["car_03_body"],
        contact_frame=1680,
        car_speed_ms=55.0,
        yaw_rate_degs=152.9,
        site_z=5.27,
        end_frame=2400,
    )
    for keyword in ("tyBirthInstant", "tyPositionObject", "tyShape",
                    "tyPhysX", "Freeze"):
        assert keyword in script, f"expected '{keyword}' in debris script"


def test_generate_debris_script_starts_with_header():
    script = tyfx.generate_debris_script(
        "Atlas_Debris", [],
        contact_frame=100,
        car_speed_ms=50.0,
        yaw_rate_degs=100.0,
        site_z=0.0,
        end_frame=500,
    )
    assert script.startswith("-- Atlas generated MaxScript: Crash Debris")


def test_generate_debris_script_contact_frame_embedded():
    cf = 1742
    script = tyfx.generate_debris_script(
        "Atlas_Debris", ["car_03_body"],
        contact_frame=cf,
        car_speed_ms=55.0,
        yaw_rate_degs=150.0,
        site_z=5.27,
        end_frame=2400,
    )
    assert str(cf) in script


# ── write_debris_script ───────────────────────────────────────────────────────

def test_write_debris_script_creates_file(tmp_path):
    result = tyfx.write_debris_script(
        tmp_path / "tyfx_debris.ms",
        ["car_03_body"],
        _SPINE,
        CRASH,
        site_z=5.27,
        end_frame=2400,
    )
    assert Path(result["path"]).is_file()
    assert result["lines"] > 0
    assert "contact_frame" in result["params"]


# ── run_debris_script ─────────────────────────────────────────────────────────

def test_run_debris_script_raises_if_file_missing():
    with pytest.raises(FileNotFoundError):
        tyfx.run_debris_script(_mock_bridge(), "/no/such/file.ms")


def test_run_debris_script_success(tmp_path):
    p = tmp_path / "debris.ms"
    p.write_text("print 'debris'", encoding="utf-8")
    b = _mock_bridge(maxscript_return="ok")
    result = tyfx.run_debris_script(b, p)
    assert result["executed"] is True


# ── generate_sparks_script ────────────────────────────────────────────────────

def test_generate_sparks_script_contains_verified_operators_and_contact_source():
    script = tyfx.generate_sparks_script(
        "Atlas_Sparks", ["car_03_body"],
        contact_frame=1680,
        car_speed_ms=55.0,
        site_z=5.27,
        end_frame=2400,
        contact_position=(12.5, -4.0),
    )
    for keyword in (
        "Sphere radius:0.02",
        "source.pos = [12.5, -4.0, 5.27]",
        '.addOperator "Birth" -1',
        '.addOperator "Position Object" -1',
        '.addOperator "Speed" -1',
        '.addOperator "Force" -1',
        '.addOperator "Scale" -1',
        '.addOperator "Time Test" -1',
        '.addOperator "Delete" -1',
        "birth.birthMode = 1",
        "force.gravityStrength = -1",
        "speed.magnitude = 5.5",
        "age.value = 8",
        "tf.reset_simulation()",
        "VRayLightMtl",
    ):
        assert keyword in script, f"expected '{keyword}' in sparks script"
    for obsolete in ("tyBirthFlow", "tyPositionObject", "tySpeed", "tf.Update()"):
        assert obsolete not in script


def test_generate_sparks_script_empty_floor_nodes_still_works():
    """Sparks must not fail when floor_nodes is empty — it is a polish layer."""
    script = tyfx.generate_sparks_script(
        "Atlas_Sparks", [],
        contact_frame=1680,
        car_speed_ms=55.0,
        site_z=5.27,
        end_frame=2400,
    )
    assert len(script) > 100
    assert "Sphere radius:0.02" in script


def test_generate_sparks_script_burst_window():
    """Burst should end contact_frame + 10."""
    cf = 1680
    script = tyfx.generate_sparks_script(
        "Atlas_Sparks", [],
        contact_frame=cf,
        car_speed_ms=50.0,
        site_z=0.0,
        end_frame=2400,
    )
    assert str(cf) in script
    assert str(cf + 10) in script


# ── write_sparks_script ───────────────────────────────────────────────────────

def test_write_sparks_script_creates_file_with_nodes(tmp_path):
    result = tyfx.write_sparks_script(
        tmp_path / "tyfx_sparks.ms",
        ["car_03_body"],
        _SPINE,
        CRASH,
        site_z=5.27,
    )
    assert Path(result["path"]).is_file()
    assert result["lines"] > 0


def test_write_sparks_script_empty_floor_nodes(tmp_path):
    """Must write a valid (non-empty) file even with no floor nodes."""
    result = tyfx.write_sparks_script(
        tmp_path / "tyfx_sparks_empty.ms",
        [],
        _SPINE,
        CRASH,
        site_z=5.27,
    )
    assert Path(result["path"]).is_file()
    assert result["lines"] > 0
    assert "contact-point source" in result["summary"]
    assert len(result["params"]["contact_position"]) == 2


# ── run_sparks_script ─────────────────────────────────────────────────────────

def test_run_sparks_script_raises_if_file_missing():
    with pytest.raises(FileNotFoundError):
        tyfx.run_sparks_script(_mock_bridge(), "/no/such/sparks.ms")


def test_run_sparks_script_success(tmp_path):
    p = tmp_path / "sparks.ms"
    p.write_text("print 'sparks'", encoding="utf-8")
    b = _mock_bridge(maxscript_return="ok")
    result = tyfx.run_sparks_script(b, p)
    assert result["executed"] is True
