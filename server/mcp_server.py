"""
Atlas MCP server — the pipeline, exposed to a model.

Wraps the modules that already work. Nothing here computes anything; it
translates between a model's tool call and the functions that do the work, and
it is written around two constraints that are easy to get wrong and hard to
notice afterwards.

**Decorators must carry ``functools.wraps``.** FastMCP builds each tool's JSON
schema from ``inspect.signature``, which follows ``__wrapped__``. Without
``wraps`` the signature reads ``(*args, **kwargs)``.

The consequence depends on the version, and the project plan's note is now out
of date. Under FastMCP 2.x this was silent: the tool was advertised with *no
parameters at all*, the server started, the tool list looked right, and the
model could call nothing. FastMCP 3.4.4 — the version here — rejects it at
registration with "Functions with *args are not supported as tools", so the
failure is loud. ``wraps`` is still required either way; only the failure mode
changed. ``tests/test_mcp_server.py`` pins the current behaviour so a
regression to the silent form is caught.

**Failures are returned, not raised.** An exception inside a tool reaches the
model as an opaque protocol error it cannot reason about. A structured
``{"success": false, "error": ...}`` is something it can act on — retry with a
smaller radius, start 3ds Max, pick a different date. Every tool here is
wrapped so that nothing escapes as an exception.

Deliberately absent: ``atlas_reconstruct`` and ``atlas_align_recon``. The
photogrammetry backend is not built, and a tool that advertises a capability
the server does not have is worse than a missing one — the model will plan
around it and fail late.

Run it with::

    .venv\\Scripts\\python.exe server/mcp_server.py
"""

from __future__ import annotations

import functools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution  # noqa: E402
import materials  # noqa: E402
import osm  # noqa: E402
import terrain  # noqa: E402
import tyfx  # noqa: E402
from frame import SceneFrame  # noqa: E402
from massing import buildings_to_meshes  # noqa: E402
from maxbridge import MaxBridge, MaxBridgeError  # noqa: E402
from scene import (  # noqa: E402
    apply_sky,
    build_camera,
    build_sun,
    configure_units,
    discover_vray,
    link_sky_to_sun,
)
from solar import solar_position_utc, sun_vector  # noqa: E402
from timeframe import resolve_local_time  # noqa: E402
from weather import WeatherError, fetch_observation, sky_from_weather  # noqa: E402

__all__ = ["mcp", "build_server"]

DEFAULT_RENDERER = os.environ.get("ATLAS_RENDERER", "V_Ray_GPU")
OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def _tool_result(fn):
    """
    Turn any exception into a structured failure the model can act on.

    ``functools.wraps`` is load-bearing, not tidiness. FastMCP derives each
    tool's parameter schema from ``inspect.signature``, which follows
    ``__wrapped__``; without it every tool advertises ``(*args, **kwargs)`` and
    the model sees a tool with no parameters. Nothing errors — the server
    starts and the tool list looks correct.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        except MaxBridgeError as exc:
            return {
                "success": False,
                "error": str(exc),
                "error_type": "max_unavailable",
                "hint": (
                    "3ds Max is not running, or the bridge was not started "
                    "inside it. This tool cannot work without a live host; "
                    "atlas_solar_position works without one."
                ),
            }
        except (osm.OverpassError, terrain.TerrainError, WeatherError) as exc:
            return {
                "success": False,
                "error": str(exc),
                "error_type": "data_source",
            }
        except Exception as exc:
            return {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": "internal",
            }
        if isinstance(result, dict) and "success" not in result:
            result = {"success": True, **result}
        return result

    return wrapper


def _bridge() -> MaxBridge:
    return MaxBridge()


# ── Tool implementations ──────────────────────────────────────────────────────
#
# Written as plain functions and registered in build_server(), so the tests can
# import and call them directly without standing up a server.


@_tool_result
def atlas_max_ping() -> dict:
    """
    Check whether 3ds Max is running with the Atlas bridge started.

    Call this first when any other Max-dependent tool fails, to tell "Max is
    closed" apart from "the command was wrong". Returns the Max version, the
    open scene name and the current system units.

    Do not call it before every operation — the other tools already report a
    clear max_unavailable failure of their own.
    """
    return {"bridge": _bridge().ping()}


@_tool_result
def atlas_solar_position(
    latitude: float,
    longitude: float,
    local_time: str,
    timezone_name: str | None = None,
) -> dict:
    """
    Sun azimuth and altitude for a wall-clock time at a location.

    Needs no 3ds Max and no network, so it is the right tool for answering
    "where was the sun at..." and for sanity-checking a shot before building
    anything. Use atlas_build_scene when you actually want geometry.

    Args:
        latitude: degrees, positive north.
        longitude: degrees, positive **east**. A positive value for a western
            longitude is the usual cause of a plausible but wrong answer.
        local_time: naive wall-clock time, "YYYY-MM-DD HH:MM" — what a clock at
            that location reads. Not UTC.
        timezone_name: IANA zone such as "Europe/Rome". Looked up from the
            coordinates when omitted.

    Azimuth is degrees clockwise from true north; altitude is degrees above the
    horizon and is negative when the sun is down. Shadows point at
    (azimuth + 180) mod 360.
    """
    resolved = resolve_local_time(
        _parse_local(local_time), latitude, longitude, timezone_name
    )
    position = solar_position_utc(resolved.utc, latitude, longitude)
    return {
        "solar": position.as_dict(),
        "time": resolved.as_dict(),
        "shadow_bearing": round((position.azimuth + 180.0) % 360.0, 3),
        "sun_vector": [round(c, 6) for c in sun_vector(position.azimuth, position.altitude)],
    }


@_tool_result
def atlas_fetch_context(
    latitude: float,
    longitude: float,
    building_radius_m: float = 500.0,
    terrain_radius_m: float = 1500.0,
    include_terrain: bool = True,
) -> dict:
    """
    Survey the OSM buildings and terrain around a site without touching Max.

    Use it to check coverage before committing to a build: how many buildings
    are mapped, how many heights are real rather than guessed, and how much
    relief the terrain has. A site with 90% defaulted heights will look like a
    city and be mostly invention, which is worth knowing first.

    Do not use it to place geometry — it only reports. atlas_build_scene builds.

    Keep building_radius_m at or below about 1000: Overpass is a free shared
    service and large queries are refused rather than slow.
    """
    frame = SceneFrame(latitude, longitude)
    buildings = osm.fetch_for_site(frame, building_radius_m)
    provenance = attribution.height_provenance(buildings)

    result: dict[str, Any] = {
        "site": {"latitude": latitude, "longitude": longitude},
        "buildings": {
            "count": len(buildings),
            "height_provenance": provenance,
            "tallest": _tallest(buildings),
        },
        "attribution": osm.ATTRIBUTION,
    }

    if include_terrain:
        patch = terrain.fetch_for_site(frame, terrain_radius_m)
        result["terrain"] = {
            "tiles": patch.metadata.get("tiles"),
            "grid": f"{patch.rows}x{patch.cols}",
            "min_m": round(patch.min_elevation(), 2),
            "max_m": round(patch.max_elevation(), 2),
            "relief_m": round(patch.relief(), 2),
        }
        result["terrain_check"] = terrain.looks_like_surface_model(patch, buildings)["note"]

    return result


@_tool_result
def atlas_set_sun(
    latitude: float,
    longitude: float,
    local_time: str,
    timezone_name: str | None = None,
) -> dict:
    """
    Re-time the sun in the scene already open in 3ds Max.

    This is the iteration loop: build the scene once, then move the sun to try
    other times of day without refetching geometry. Reuses the existing sun
    node, so it does not litter the scene.

    Requires a scene that already has geometry — use atlas_build_scene first.
    """
    bridge = _bridge()
    caps = discover_vray(bridge)
    setup = build_sun(
        bridge,
        local_time=_parse_local(local_time),
        latitude=latitude,
        longitude=longitude,
        tz_name=timezone_name,
        caps=caps,
    )
    sky_note = _apply_weather_sky(bridge, setup, latitude, longitude, caps)
    return {"sun": setup.as_dict(), "sky": sky_note}


@_tool_result
def atlas_build_scene(
    latitude: float,
    longitude: float,
    local_time: str,
    timezone_name: str | None = None,
    building_radius_m: float = 500.0,
    terrain_radius_m: float = 1500.0,
    include_terrain: bool = True,
    reset_scene: bool = True,
    assign_materials: bool = True,
) -> dict:
    """
    The headline call: date, time and location in, a lit georeferenced scene out.

    Fetches OSM buildings and Copernicus terrain, extrudes the buildings onto
    the terrain, places a physically-correct V-Ray sun for that moment, derives
    sky turbidity from historical weather, and writes ATTRIBUTION.txt.

    Requires 3ds Max running with the bridge. Takes tens of seconds — it makes
    several network calls and pushes thousands of vertices. Call
    atlas_fetch_context first if you only want to know what is there.

    reset_scene clears the file before building. Leave it True unless you are
    deliberately adding to an existing scene; note that resetting also reverts
    the renderer and units, which this tool re-establishes afterwards.
    """
    bridge = _bridge()
    if reset_scene:
        bridge.call("resetMaxFile", MaxBridge.name("noPrompt"), timeout=180.0)

    units = configure_units(bridge)
    renderer = bridge.set_renderer(DEFAULT_RENDERER)["resolved"]

    frame = SceneFrame(latitude, longitude)
    patch = terrain.fetch_for_site(frame, terrain_radius_m) if include_terrain else None
    buildings = osm.fetch_for_site(frame, building_radius_m)

    ground = patch.elevation_at if patch is not None else None
    meshes, skipped = buildings_to_meshes(buildings, frame, ground)

    push = []
    if patch is not None:
        terrain_mesh = patch.to_mesh(frame, name="atlas_terrain")
        push.append((terrain_mesh.name, terrain_mesh.verts, terrain_mesh.faces))
    push += [(m.name, m.verts, m.faces) for m in meshes]
    pushed = bridge.create_meshes(push, chunk=30, timeout=900.0)
    failed = [p for p in pushed if not p.get("ok")]

    shading = None
    if assign_materials and meshes:
        by_id = {b.osm_id: b for b in buildings}
        pairs = [
            (by_id[m.metadata["osm_id"]], m.name)
            for m in meshes
            if m.metadata.get("osm_id") in by_id
        ]
        shading = materials.assign_materials(bridge, pairs)
        if patch is not None:
            shading["terrain"] = materials.assign_terrain_material(bridge)

    caps = discover_vray(bridge)
    setup = build_sun(
        bridge,
        local_time=_parse_local(local_time),
        latitude=latitude,
        longitude=longitude,
        tz_name=timezone_name,
        caps=caps,
    )
    sky_note = _apply_weather_sky(bridge, setup, latitude, longitude, caps)

    manifest = attribution.manifest_for_build(
        site={
            "location": f"{latitude}, {longitude}",
            "local_time": setup.time.local.isoformat(),
            "buildings": len(buildings),
        },
        buildings=buildings,
        terrain_used=patch is not None,
        terrain_modified=patch is not None,
        weather_used=sky_note.get("weather_available", False),
    )
    OUT_DIR.mkdir(exist_ok=True)
    manifest.write(OUT_DIR / "ATTRIBUTION.txt")

    return {
        "units": units,
        "renderer": renderer,
        "buildings": {"fetched": len(buildings), "built": len(meshes), "skipped": skipped},
        "terrain": patch.as_dict() if patch is not None else None,
        "meshes_pushed": len(pushed) - len(failed),
        "meshes_failed": [f.get("error") for f in failed[:5]],
        "materials": shading,
        "sun": setup.as_dict(),
        "sky": sky_note,
        "attribution_written": str(OUT_DIR / "ATTRIBUTION.txt"),
    }


@_tool_result
def atlas_assign_materials(latitude: float, longitude: float, radius_m: float = 500.0) -> dict:
    """
    Shade the buildings already in the scene from their OSM tags.

    atlas_build_scene does this already — reach for this only to re-shade a
    scene built with assign_materials off, or after editing the presets.

    Materials are shared per material kind, not per building: a thousand
    footprints become fewer than ten VRayMtl instances. The reply reports
    anything the host rejected; an empty rejected map is what makes the applied
    values believable.
    """
    bridge = _bridge()
    frame = SceneFrame(latitude, longitude)
    buildings = osm.fetch_for_site(frame, radius_m)

    from massing import _mesh_name

    live = {n["name"] for n in bridge.scene_list()["nodes"]}
    pairs = [(b, _mesh_name(b)) for b in buildings if _mesh_name(b) in live]
    if not pairs:
        return {
            "success": False,
            "error": "no OSM building nodes from this site are in the scene",
            "error_type": "empty_scene",
            "hint": "run atlas_build_scene for this location first",
        }

    result = materials.assign_materials(bridge, pairs)
    result["terrain"] = materials.assign_terrain_material(bridge)
    return result


@_tool_result
def atlas_place_camera(
    position_x: float,
    position_y: float,
    position_z: float,
    target_x: float = 0.0,
    target_y: float = 0.0,
    target_z: float = 0.0,
    fov_degrees: float = 50.0,
    camera_name: str = "Atlas_Cam",
) -> dict:
    """
    Place and aim a physical camera, in scene metres.

    The frame is +X east, +Y true north, +Z up, with the site at the origin, so
    a camera to the south-west of the site looking at it has negative X and Y.

    The reply reports direction_verified. If that is missing or a
    direction_error appears, the camera is not pointing where it was asked to —
    treat the render as untrustworthy rather than adjusting the position.
    """
    result = build_camera(
        _bridge(),
        position=(position_x, position_y, position_z),
        target=(target_x, target_y, target_z),
        camera_name=camera_name,
        fov_degrees=fov_degrees,
    )
    return {"camera": result}


@_tool_result
def atlas_render(
    output_name: str = "render.png",
    camera_name: str = "Atlas_Cam",
    width: int = 960,
    height: int = 540,
) -> dict:
    """
    Render the named camera to a PNG under out/.

    Slow — a render holds the 3ds Max main thread for its whole duration and
    every other tool call queues behind it. Prefer atlas_viewport_capture for a
    quick look at framing; render when you need the lighting to be real.

    The render asserts the production renderer is V-Ray first, because
    resetMaxFile silently reverts it and a V-Ray sun contributes nothing to an
    Arnold render.
    """
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / Path(output_name).name
    return {
        "render": _bridge().render(
            str(path),
            camera=camera_name,
            width=width,
            height=height,
            expect_renderer="V_Ray",
            timeout=1800.0,
        )
    }


@_tool_result
def atlas_viewport_capture(output_name: str = "viewport.png") -> dict:
    """
    Save a viewport screenshot to out/, for a look at the scene without rendering.

    Cheap compared with atlas_render, and enough to check framing, scale and
    whether geometry landed. It captures the *active* viewport, so it shows
    whatever Max is currently displaying rather than a named camera.

    Note it can return a stale frame if the Max window is minimised or fully
    occluded — if the image does not change between calls, that is why.
    """
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / Path(output_name).name
    return {"capture": _bridge().viewport_capture(str(path))}


@_tool_result
def atlas_scene_summary(
    filter_class: Literal["all", "geometry", "lights", "cameras"] = "all",
) -> dict:
    """
    List what is currently in the 3ds Max scene.

    Use it to confirm a build landed, to find node names before operating on
    them, or to check whether a previous scene is still loaded before building
    over it.
    """
    listing = _bridge().scene_list()
    nodes = listing["nodes"]
    if filter_class != "all":
        wanted = {
            "geometry": ("mesh", "poly", "box", "plane"),
            "lights": ("sun", "light"),
            "cameras": ("camera",),
        }[filter_class]
        nodes = [n for n in nodes if any(w in n["class"].lower() for w in wanted)]

    return {
        "total_in_scene": listing["count"],
        "matched": len(nodes),
        "by_class": _count_classes(nodes),
        "sample": [n["name"] for n in nodes[:25]],
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_local(text: str) -> datetime:
    """
    Parse a wall-clock string, accepting the formats a model actually emits.

    A tz-aware string is rejected rather than silently reinterpreted: this
    pipeline resolves the zone from the coordinates, and accepting an offset
    here would give two sources of truth that can disagree.
    """
    cleaned = str(text).strip().replace("T", " ")
    if cleaned.endswith("Z") or "+" in cleaned[10:]:
        raise ValueError(
            f"{text!r} carries a timezone. Pass local wall-clock time "
            "('YYYY-MM-DD HH:MM') and let the zone be resolved from the "
            "coordinates, or name it in timezone_name."
        )
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d %H", "%Y-%m-%d"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"could not read {text!r} as a local time. Use 'YYYY-MM-DD HH:MM'."
    )


def _apply_weather_sky(bridge, setup, latitude: float, longitude: float, caps) -> dict:
    """
    Derive sky parameters from historical weather, degrading honestly.

    ERA5 lags real time and does not cover every date. When it is unavailable
    the sun is still correct and only the sky tuning is missing, so this reports
    which happened rather than leaving the caller to guess.
    """
    try:
        observation = fetch_observation(setup.time.utc, latitude, longitude)
    except WeatherError as exc:
        link_sky_to_sun(bridge, caps=caps)
        return {"weather_available": False, "reason": str(exc)}

    settings = sky_from_weather(observation)
    applied = apply_sky(bridge, settings.params, caps=caps)
    link_sky_to_sun(bridge, caps=caps)
    return {
        "weather_available": True,
        "conditions": observation.description,
        "cloud_cover_pct": observation.cloud_cover_pct,
        "params": settings.params,
        "rationale": settings.rationale,
        "rejected": applied.get("rejected") if isinstance(applied, dict) else None,
    }


def _tallest(buildings) -> dict | None:
    if not buildings:
        return None
    tallest = max(buildings, key=lambda b: b.height_m)
    return {
        "name": tallest.name,
        "height_m": tallest.height_m,
        "height_source": tallest.height_source,
    }


def _count_classes(nodes) -> dict:
    counts: dict[str, int] = {}
    for node in nodes:
        counts[node["class"]] = counts.get(node["class"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ── Server ────────────────────────────────────────────────────────────────────

@_tool_result
def atlas_tyre_smoke(
    emitter_nodes: list[str],
    spine_json: str,
    *,
    execute: bool = False,
    speed_threshold_kmh: float = 200.0,
    end_frame: int = 2400,
    wind_bearing_deg: float = 331.0,
    wind_speed_ms: float = 7.34,
    site_z: float = 5.27,
) -> dict:
    """
    Generate (and optionally run) the tyFlow tyre-smoke event graph.

    Writes the MaxScript to out/tyfx_smoke.ms and returns a summary so the
    operator can review the file before running it. Set execute=True to also
    send it to 3ds Max — requires ATLAS_ALLOW_MAXSCRIPT=1 in the host
    environment before Max was launched.

    Args:
        emitter_nodes: Names of the rear tyre mesh nodes in the scene
            (e.g. ["car_03_tyres", "car_04_tyres"]).
        spine_json: JSON array of [x, y] pairs — the circuit spine used to
            compute the per-frame speed gate from raceanim.speed_profile.
        execute: When False (default) only writes the script. When True also
            executes it via the bridge (requires ATLAS_ALLOW_MAXSCRIPT=1).
        speed_threshold_kmh: Emit only when car speed exceeds this. Default
            200 km/h captures braking zones and the crash site.
        end_frame: Last frame of the animation range.
        wind_bearing_deg: ERA5 wind bearing in degrees from north (default 331).
        wind_speed_ms: ERA5 wind speed in m/s (default 7.34).
        site_z: Ground elevation in scene metres (Yas Marina = 5.27).
    """
    import json as _json
    spine = [tuple(p) for p in _json.loads(spine_json)]
    path = OUT_DIR / "tyfx_smoke.ms"
    OUT_DIR.mkdir(exist_ok=True)

    result = tyfx.write_smoke_script(
        path, emitter_nodes, spine,
        wind_bearing_deg=wind_bearing_deg,
        wind_speed_ms=wind_speed_ms,
        site_z=site_z,
        end_frame=end_frame,
        speed_threshold_ms=speed_threshold_kmh / 3.6,
    )

    if execute:
        run_result = tyfx.run_smoke_script(_bridge(), path)
        result["executed"] = run_result
    else:
        result["note"] = (
            "Script written but not executed. Review out/tyfx_smoke.ms "
            "then call again with execute=True, or run it manually from "
            "the MaxScript listener."
        )
    return result


@_tool_result
def atlas_crash_debris(
    wing_nodes: list[str],
    spine_json: str,
    *,
    execute: bool = False,
    end_frame: int = 2400,
    site_z: float = 5.27,
    fps: int = 24,
) -> dict:
    """
    Generate (and optionally run) the crash-debris tyFlow event graph.

    Writes the MaxScript to out/tyfx_debris.ms. Debris starts at the contact
    frame derived from raceanim.CRASH (at_distance_m=3650, lap ~0.69), which
    is inside the 11_crash_wide and 12_crash_tight shot windows.

    Args:
        wing_nodes: Names of source car mesh nodes (e.g. ["car_01"]).
            Atlas currently emits the mesh's disconnected components; it does
            not yet perform a Voronoi fracture of the monocoque.
        spine_json: JSON array of [x, y] circuit spine pairs.
        execute: Run via bridge when True (requires ATLAS_ALLOW_MAXSCRIPT=1).
        end_frame: Last frame of the animation range.
        site_z: Ground elevation in scene metres.
        fps: Frames per second (default 24).
    """
    import json as _json
    from raceanim import CRASH
    spine = [tuple(p) for p in _json.loads(spine_json)]
    path = OUT_DIR / "tyfx_debris.ms"
    OUT_DIR.mkdir(exist_ok=True)

    result = tyfx.write_debris_script(
        path, wing_nodes, spine, CRASH,
        site_z=site_z, end_frame=end_frame, fps=fps,
    )

    if execute:
        run_result = tyfx.run_debris_script(_bridge(), path)
        result["executed"] = run_result
    else:
        result["note"] = (
            "Script written but not executed. Review out/tyfx_debris.ms "
            "then call again with execute=True."
        )
    return result


@_tool_result
def atlas_crash_sparks(
    spine_json: str,
    *,
    floor_nodes: list[str] | None = None,
    execute: bool = False,
    end_frame: int = 2400,
    site_z: float = 5.27,
    fps: int = 24,
) -> dict:
    """
    Generate (and optionally run) the crash-sparks tyFlow event graph.

    Titanium skid-block sparks over ~0.4 s from the contact frame. Writes
    the MaxScript to out/tyfx_sparks.ms. Assign VRayLightMtl to the
    particles manually after running for the glow effect.

    Args:
        spine_json: JSON array of [x, y] circuit spine pairs.
        floor_nodes: Names of floor/skid-block mesh nodes. If omitted or
            empty, emits from a point at the crash position.
        execute: Run via bridge when True (requires ATLAS_ALLOW_MAXSCRIPT=1).
        end_frame: Last frame of the animation range.
        site_z: Ground elevation in scene metres.
        fps: Frames per second (default 24).
    """
    import json as _json
    from raceanim import CRASH
    spine = [tuple(p) for p in _json.loads(spine_json)]
    nodes = floor_nodes or []
    path = OUT_DIR / "tyfx_sparks.ms"
    OUT_DIR.mkdir(exist_ok=True)

    result = tyfx.write_sparks_script(
        path, nodes, spine, CRASH,
        site_z=site_z, end_frame=end_frame, fps=fps,
    )

    if execute:
        run_result = tyfx.run_sparks_script(_bridge(), path)
        result["executed"] = run_result
    else:
        result["note"] = (
            "Script written but not executed. Review out/tyfx_sparks.ms "
            "then call again with execute=True. "
            "Assign VRayLightMtl to the particles manually for the glow."
        )
    return result


TOOLS = [
    atlas_max_ping,
    atlas_solar_position,
    atlas_fetch_context,
    atlas_build_scene,
    atlas_set_sun,
    atlas_assign_materials,
    atlas_place_camera,
    atlas_render,
    atlas_viewport_capture,
    atlas_scene_summary,
    atlas_tyre_smoke,
    atlas_crash_debris,
    atlas_crash_sparks,
]


def build_server(name: str = "atlas"):
    """
    Construct the FastMCP server.

    A factory rather than module-level state so tests can build a fresh server
    and inspect the schemas it generates.
    """
    from fastmcp import FastMCP

    server = FastMCP(name)
    for fn in TOOLS:
        server.tool(fn)
    return server


mcp = None


def main() -> None:
    global mcp
    mcp = build_server()
    mcp.run()


if __name__ == "__main__":
    main()
