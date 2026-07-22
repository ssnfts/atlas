"""
MCP server tests — no 3ds Max, no network.

The failure this file exists to catch: FastMCP derives each tool's parameter
schema from ``inspect.signature``, which follows ``__wrapped__``. A decorator
without ``functools.wraps`` leaves the signature reading ``(*args, **kwargs)``,
so the tool is advertised with **no parameters at all**. The server starts, the
tool list is the right length, every name is correct — and the model cannot
call anything. Asserting the tool count would pass throughout.

So these assert on the generated schema itself, and on the wrapper's other
contract: that failures come back as structured results rather than exceptions,
because an exception reaches the model as an opaque protocol error it cannot
reason about.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

pytest.importorskip("fastmcp", reason="MCP server dependency")

import mcp_server  # noqa: E402
from maxbridge import MaxNotRunning  # noqa: E402


@pytest.fixture(scope="module")
def schemas() -> dict:
    """Tool name -> generated JSON schema, straight from a real server."""
    server = mcp_server.build_server()
    tools = asyncio.run(server.list_tools())
    return {t.name: t.parameters for t in tools}


@pytest.fixture(scope="module")
def descriptions() -> dict:
    server = mcp_server.build_server()
    tools = asyncio.run(server.list_tools())
    return {t.name: (t.description or "") for t in tools}


# ── The wraps trap ────────────────────────────────────────────────────────────

def test_a_decorator_without_wraps_fails_loudly_on_this_version():
    """
    Pins what FastMCP 3.x actually does, which is **not** what the project plan
    (written against 2.x) describes.

    The documented trap was that a wraps-less decorator silently produced a
    tool advertising (*args, **kwargs) — server starts, tool list looks right,
    model can call nothing. FastMCP 3.4.4 fixed that upstream: registration now
    raises instead.

    Asserted rather than assumed, so a future release that regresses to the
    silent behaviour fails here. functools.wraps in mcp_server is still
    required either way — without it the server does not start at all.
    """
    from fastmcp import FastMCP

    def unwrapped(fn):
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    server = FastMCP("regression-probe")
    with pytest.raises(ValueError, match=r"\*args"):

        @server.tool
        @unwrapped
        def sample(latitude: float, longitude: float) -> dict:
            """Probe."""
            return {}


def test_no_tool_advertises_args_or_kwargs(schemas):
    """
    Belt and braces behind the upstream check above. Cheap, and it still holds
    if a decorator ever leaks a signature some other way.
    """
    for name, schema in schemas.items():
        properties = set(schema.get("properties", {}))
        assert "args" not in properties, f"{name} exposes *args — wraps is missing"
        assert "kwargs" not in properties, f"{name} exposes **kwargs — wraps is missing"


def test_every_tool_with_parameters_actually_advertises_them(schemas):
    """A tool whose real signature takes arguments must not appear parameterless."""
    import inspect

    for fn in mcp_server.TOOLS:
        real = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]
        advertised = set(schemas[fn.__name__].get("properties", {}))
        assert advertised == {p.name for p in real}, (
            f"{fn.__name__} advertises {advertised}, real signature has "
            f"{[p.name for p in real]}"
        )


def test_decorator_preserves_the_wrapped_signature():
    """Direct check on the mechanism, independent of FastMCP."""
    import inspect

    assert hasattr(mcp_server.atlas_solar_position, "__wrapped__")
    params = list(inspect.signature(mcp_server.atlas_solar_position).parameters)
    assert params == ["latitude", "longitude", "local_time", "timezone_name"]


def test_required_parameters_are_marked_required(schemas):
    required = set(schemas["atlas_solar_position"].get("required", []))
    assert {"latitude", "longitude", "local_time"} <= required
    assert "timezone_name" not in required


def test_types_survive_into_the_schema(schemas):
    props = schemas["atlas_solar_position"]["properties"]
    assert props["latitude"]["type"] == "number"
    assert props["local_time"]["type"] == "string"


def test_literal_becomes_an_enum(schemas):
    """Literal annotations constrain the model's output; plain str does not."""
    prop = schemas["atlas_scene_summary"]["properties"]["filter_class"]
    assert set(prop["enum"]) == {"all", "geometry", "lights", "cameras"}


def test_defaults_are_advertised(schemas):
    props = schemas["atlas_fetch_context"]["properties"]
    assert props["building_radius_m"]["default"] == 500.0
    assert props["include_terrain"]["default"] is True


# ── Tool surface ──────────────────────────────────────────────────────────────

def test_expected_tools_are_registered(schemas):
    assert {
        "atlas_max_ping",
        "atlas_solar_position",
        "atlas_fetch_context",
        "atlas_build_scene",
        "atlas_set_sun",
        "atlas_place_camera",
        "atlas_render",
        "atlas_viewport_capture",
        "atlas_scene_summary",
    } <= set(schemas)


def test_unbuilt_capabilities_are_not_advertised(schemas):
    """
    Photogrammetry does not exist yet. A tool promising it is worse than a
    missing one — the model plans around it and fails late.
    """
    assert "atlas_reconstruct" not in schemas
    assert "atlas_align_recon" not in schemas


def test_every_tool_has_a_description(descriptions):
    for name, text in descriptions.items():
        assert len(text.strip()) > 40, f"{name} has no usable description"


def test_descriptions_say_when_not_to_use_the_tool(descriptions):
    """
    Docstrings are written for the model. The negative case is the part that
    stops it reaching for the expensive tool by default.
    """
    for name in ("atlas_fetch_context", "atlas_render", "atlas_build_scene"):
        text = descriptions[name].lower()
        assert any(k in text for k in ("do not", "rather than", "prefer", "only")), (
            f"{name} never says when not to use it"
        )


def test_max_dependent_tools_say_so(descriptions):
    for name in ("atlas_build_scene", "atlas_set_sun"):
        assert "max" in descriptions[name].lower()


# ── Failures are returned, not raised ─────────────────────────────────────────

def test_missing_max_is_a_structured_result_not_an_exception(monkeypatch):
    """
    An exception reaches the model as an opaque protocol error. A structured
    failure is something it can act on.
    """
    def boom():
        raise MaxNotRunning("nothing is listening on 127.0.0.1:9879")

    monkeypatch.setattr(mcp_server, "_bridge", lambda: type("B", (), {"ping": staticmethod(boom)})())
    result = mcp_server.atlas_max_ping()
    assert result["success"] is False
    assert result["error_type"] == "max_unavailable"
    assert "9879" in result["error"]


def test_max_unavailable_hint_points_at_the_offline_tool(monkeypatch):
    def boom():
        raise MaxNotRunning("down")

    monkeypatch.setattr(mcp_server, "_bridge", lambda: type("B", (), {"ping": staticmethod(boom)})())
    assert "atlas_solar_position" in mcp_server.atlas_max_ping()["hint"]


def test_data_source_failures_are_tagged_separately(monkeypatch):
    """
    Overpass being busy is not the same problem as Max being closed, and the
    model's next move differs: wait and retry versus start an application.
    """
    import osm as osm_module

    monkeypatch.setattr(
        osm_module, "fetch_for_site",
        lambda *a, **k: (_ for _ in ()).throw(osm_module.OverpassError("rate limited")),
    )
    result = mcp_server.atlas_fetch_context(25.08, 55.14)
    assert result["success"] is False
    assert result["error_type"] == "data_source"


def test_bad_input_is_reported_not_raised():
    result = mcp_server.atlas_solar_position(25.08, 55.14, "not a date")
    assert result["success"] is False
    assert "could not read" in result["error"]


def test_success_flag_is_added_to_good_results():
    result = mcp_server.atlas_solar_position(25.2048, 55.2708, "2024-10-12 16:00")
    assert result["success"] is True


# ── Solar tool behaviour (the one that needs neither Max nor network) ─────────

def test_solar_position_matches_the_underlying_module():
    """The tool must not quietly transform what solar.py computed."""
    from datetime import datetime

    from solar import solar_position_utc
    from timeframe import resolve_local_time

    result = mcp_server.atlas_solar_position(25.2048, 55.2708, "2024-10-12 16:00")
    resolved = resolve_local_time(datetime(2024, 10, 12, 16, 0), 25.2048, 55.2708)
    expected = solar_position_utc(resolved.utc, 25.2048, 55.2708)
    assert result["solar"]["azimuth"] == pytest.approx(expected.azimuth, abs=1e-4)
    assert result["solar"]["altitude"] == pytest.approx(expected.altitude, abs=1e-4)


def test_shadow_bearing_is_opposite_the_sun():
    result = mcp_server.atlas_solar_position(25.2048, 55.2708, "2024-10-12 16:00")
    azimuth = result["solar"]["azimuth"]
    assert result["shadow_bearing"] == pytest.approx((azimuth + 180.0) % 360.0, abs=1e-3)


def test_explicit_timezone_is_honoured():
    result = mcp_server.atlas_solar_position(
        41.890209, 12.492231, "2026-07-22 19:30", "Europe/Rome"
    )
    assert result["time"]["timezone"] == "Europe/Rome"
    assert result["solar"]["altitude"] == pytest.approx(10.98, abs=0.05)


def test_timezone_is_resolved_from_coordinates_when_omitted():
    result = mcp_server.atlas_solar_position(41.890209, 12.492231, "2026-07-22 19:30")
    assert result["time"]["timezone"] == "Europe/Rome"


def test_nonexistent_local_time_is_refused():
    """A DST-skipped wall-clock time means the source timestamp is wrong."""
    result = mcp_server.atlas_solar_position(51.5074, -0.1278, "2024-03-31 01:30")
    assert result["success"] is False
    assert "does not exist" in result["error"]


# ── Time parsing ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text",
    ["2024-10-12 16:00", "2024-10-12T16:00", "2024-10-12 16:00:00", "2024-10-12 16", "2024-10-12"],
)
def test_accepted_time_formats(text):
    assert mcp_server._parse_local(text).year == 2024


@pytest.mark.parametrize("text", ["2024-10-12T16:00:00Z", "2024-10-12 16:00+02:00"])
def test_timezone_aware_strings_are_refused(text):
    """
    Two sources of truth for the zone can disagree. The pipeline resolves it
    from the coordinates, so an offset here is rejected rather than reconciled.
    """
    with pytest.raises(ValueError, match="timezone"):
        mcp_server._parse_local(text)


@pytest.mark.parametrize("text", ["", "yesterday", "12/10/2024", "16:00"])
def test_unparseable_times_are_refused(text):
    with pytest.raises(ValueError):
        mcp_server._parse_local(text)
