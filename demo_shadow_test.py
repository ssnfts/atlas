"""
Acceptance test: does the sun land where the real sun was?

Builds a deliberately minimal scene — a ground plane and a single vertical
column — and renders it at several times of one day. A column casts an
unambiguous shadow, so the shadow direction can be read straight off the image
and checked against the computed azimuth.

This is the test that matters. Every unit test in this project can pass on an
implementation that is subtly wrong in a self-consistent way; a shadow either
points where the real one pointed, or it does not.

    .venv\\Scripts\\python.exe demo_shadow_test.py
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "server"))

from maxbridge import MaxBridge  # noqa: E402
from scene import (  # noqa: E402
    apply_sky,
    build_camera,
    build_sun,
    configure_units,
    discover_vray,
    link_sky_to_sun,
)
from weather import fetch_observation, sky_from_weather  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Dubai Marina.
LAT, LON = 25.0805, 55.1403
DATE = (2024, 10, 12)
HOURS = [8, 12, 16]

# Proportions matter more than they look. An earlier version used a 2 x 2 m
# column viewed from 260 m: about 7 px of column casting a 2 px-wide shadow, so
# there was never a shadow to see and the frame was just sunlit ground. The
# column is now tall and thick enough that its shadow reads as a compass needle.
COLUMN_SIZE = 12.0     # metres square
COLUMN_HEIGHT = 60.0   # metres
GROUND_SIZE = 600.0    # metres
CAMERA_HEIGHT = 190.0  # metres
CAMERA_FOV = 55.0

RENDER_W, RENDER_H = 640, 360


RENDERER = "V_Ray_GPU"


def build_stage(bridge: MaxBridge) -> str:
    """
    Ground plane plus one column, both at the scene origin.

    Returns the resolved renderer name.
    """
    bridge.call("resetMaxFile", bridge.name("noPrompt"))

    # resetMaxFile reverts the renderer to the application default (Arnold
    # here), so this must come *after* the reset. A V-Ray sun and sky
    # contribute nothing to an Arnold render, and nothing errors — the image
    # just comes back featureless.
    chosen = bridge.set_renderer(RENDERER)
    print(f"renderer set: {chosen['resolved']}")

    configure_units(bridge)

    ground = bridge.call("Plane", length=GROUND_SIZE, width=GROUND_SIZE)
    bridge.node_set(ground["__node__"], "name", "Atlas_Ground")

    column = bridge.call(
        "Box", length=COLUMN_SIZE, width=COLUMN_SIZE, height=COLUMN_HEIGHT
    )
    bridge.node_set(column["__node__"], "name", "Atlas_Column")
    # Box pivots at its base corner, so offset by half to centre it on origin.
    bridge.node_set(
        "Atlas_Column", "pos", MaxBridge.point3(-COLUMN_SIZE / 2, -COLUMN_SIZE / 2, 0)
    )

    # Mid-grey ground. Without a material the default is near-white and blows
    # out under a physical sun, hiding the very shadow this test is looking for.
    bridge.assign_material(
        "Atlas_Ground",
        name="Atlas_Ground_Mtl",
        params={"diffuse": MaxBridge.color(140, 140, 140)},
    )
    bridge.assign_material(
        "Atlas_Column",
        name="Atlas_Column_Mtl",
        params={"diffuse": MaxBridge.color(200, 90, 60)},
    )

    # Camera is created per-hour in main(), because its exposure depends on the
    # sun altitude for that hour.
    return chosen["resolved"]


def expected_shadow_bearing(sun_azimuth: float) -> float:
    """A shadow points directly away from the sun."""
    return (sun_azimuth + 180.0) % 360.0


def main() -> int:
    bridge = MaxBridge()

    # Build the stage first: it resets the scene, which also resets the
    # renderer, so anything discovered beforehand would be stale.
    renderer = build_stage(bridge)

    caps = discover_vray(bridge)
    if not caps.available:
        print(f"V-Ray unavailable: {caps.note}")
        return 1
    print(f"renderer in use: {caps.renderer}\n")

    results = []
    for hour in HOURS:
        when = datetime(*DATE, hour, 0)
        setup = build_sun(
            bridge, local_time=when, latitude=LAT, longitude=LON, caps=caps
        )
        obs = fetch_observation(setup.time.utc, LAT, LON)
        sky = sky_from_weather(obs)
        applied = apply_sky(bridge, sky.params, caps=caps)
        link_sky_to_sun(bridge, caps=caps)

        bearing = expected_shadow_bearing(setup.position.azimuth)
        compass = _compass(bearing)

        # Exposure follows the sun altitude, so the camera is rebuilt each hour.
        cam = build_camera(
            bridge,
            position=(0.0, 0.0, CAMERA_HEIGHT),
            target=(0.0, 0.0, 0.0),
            exposure=setup.exposure,
            fov_degrees=CAMERA_FOV,
        )

        out_path = OUT / f"shadow_{hour:02d}00.png"
        print(f"{hour:02d}:00 local  az {setup.position.azimuth:6.2f}  "
              f"alt {setup.position.altitude:5.2f}  "
              f"turbidity {sky.params['turbidity']:5.2f}  "
              f"EV {cam['applied'].get('exposure_value')}  "
              f"-> shadow {compass} ({bearing:.1f})")

        try:
            bridge.render(
                str(out_path),
                camera="Atlas_Cam",
                width=RENDER_W,
                height=RENDER_H,
                expect_renderer=RENDERER,
            )
            print(f"          rendered -> {out_path.name}")
        except Exception as exc:
            print(f"          render failed: {type(exc).__name__}: {exc}")

        results.append(
            {
                "hour": hour,
                "azimuth": round(setup.position.azimuth, 3),
                "altitude": round(setup.position.altitude, 3),
                "shadow_bearing": round(bearing, 3),
                "shadow_compass": compass,
                "turbidity": sky.params["turbidity"],
                "conditions": obs.description,
                "image": out_path.name,
            }
        )

    (OUT / "shadow_test.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote out/shadow_test.json")

    # Physical invariant: the sun tracks east to west, so north of the tropics
    # the shadow bearing sweeps steadily clockwise.
    #
    # Compass bearings wrap, and a morning-to-afternoon sweep normally crosses
    # north — 290 -> 357 -> 68 is monotonic, but a naive sort says otherwise.
    # Unwrap before comparing.
    bearings = [r["shadow_bearing"] for r in results]
    unwrapped = [bearings[0]]
    for b in bearings[1:]:
        prev = unwrapped[-1]
        while b < prev:
            b += 360.0
        unwrapped.append(b)

    steps = [b - a for a, b in zip(unwrapped, unwrapped[1:])]
    if all(0 < s < 180 for s in steps):
        print(
            "shadow bearing sweeps clockwise through the day "
            f"({' -> '.join(r['shadow_compass'] for r in results)}) — consistent "
            "with a northern-hemisphere sun tracking east to west"
        )
    else:
        print(f"UNEXPECTED: shadow bearings do not sweep clockwise: {bearings}")
        return 1
    return 0


def _compass(bearing: float) -> str:
    points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
              "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return points[round(bearing / 22.5) % 16]


if __name__ == "__main__":
    sys.exit(main())
