"""
Live verification against a running 3ds Max.

Everything else in this project is tested without Max. This covers only what
genuinely needs the host: the bridge round-trip, main-thread marshalling, and
what V-Ray actually exposes in this build.

Run:  .venv\\Scripts\\python.exe verify_live.py

Prerequisite — start the bridge from the 3ds Max MAXScript listener:

    python.ExecuteFile @"<path-to-repo>\\bridge\\start_bridge.py"
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "server"))

from maxbridge import MaxBridge, MaxBridgeError, MaxNotRunning  # noqa: E402
from scene import build_sun, configure_units, discover_vray  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Dubai Marina, a late-October afternoon — sun should be low in the west.
SITE_LAT, SITE_LON = 25.0805, 55.1403
SHOT_TIME = datetime(2024, 10, 12, 16, 0)

_passes = 0
_failures: list[str] = []


def check(label: str, fn):
    global _passes
    try:
        result = fn()
    except Exception as exc:
        _failures.append(f"{label}: {type(exc).__name__}: {exc}")
        print(f"  FAIL  {label}\n        {type(exc).__name__}: {exc}")
        return None
    _passes += 1
    print(f"  ok    {label}")
    return result


def main() -> int:
    bridge = MaxBridge()

    print(f"\nAtlas live check -> {bridge.host}:{bridge.port}\n")

    # ── 1. Reachability ──────────────────────────────────────────────────────
    print("Bridge")
    try:
        info = bridge.ping()
    except MaxNotRunning as exc:
        print(f"  FAIL  not reachable\n\n{exc}\n")
        return 1
    except MaxBridgeError as exc:
        print(f"  FAIL  {exc}\n")
        return 1

    print(f"  ok    ping — {info.get('product')}, units={info.get('units')}")
    print(f"        scene: {info.get('scene')}")
    print(f"        raw MaxScript: {'enabled' if info.get('maxscript_enabled') else 'disabled'}")

    # ── 2. Round-trip and coercion ───────────────────────────────────────────
    print("\nDispatch")
    check(
        "get reads a value without invoking it",
        lambda: _expect(bridge.get("units.SystemType"), lambda v: v is not None),
    )
    check(
        "call creates and coerces a node",
        lambda: _expect(
            bridge.call("Box", length=10, width=10, height=10),
            lambda v: isinstance(v, dict) and "__node__" in v,
        ),
    )
    check(
        "unknown function raises, not silently passes",
        lambda: _expect_raises(lambda: bridge.call("definitelyNotAFunction")),
    )
    check(
        "private attribute is refused",
        lambda: _expect_raises(lambda: bridge.get("_private")),
    )
    check(
        "unknown node property is refused, not swallowed",
        lambda: _expect_raises(
            lambda: bridge.node_set("Box001", "notARealProperty", 1)
        ),
    )

    # ── 2b. Units ────────────────────────────────────────────────────────────
    print("\nUnits")
    units = check("scene system units set to metres", lambda: configure_units(bridge))
    if units:
        print(f"        {units['system_units_before']} -> {units['system_units_after']}"
              f"{'  (changed)' if units['changed'] else '  (already correct)'}")
    check(
        "system unit really is metres",
        lambda: _expect(
            bridge.get("units.SystemType"),
            lambda v: str(v).lstrip("#").lower() == "meters",
        ),
    )

    # ── 3. Batch atomicity ───────────────────────────────────────────────────
    print("\nBatch")
    steps = check(
        "batch executes in one main-thread slot",
        lambda: bridge.batch(
            [
                {"command": "call", "mode": "call", "func": "Sphere",
                 "kwargs": {"radius": 5}},
                {"command": "call", "mode": "get", "func": "objects.count"},
            ]
        ),
    )
    if steps:
        print(f"        node count after insert: {steps[1].get('result')}")

    # ── 4. Main-thread marshalling under load ────────────────────────────────
    print("\nMain-thread marshalling")

    def hammer():
        t0 = time.time()
        for _ in range(25):
            bridge.get("objects.count")
        return time.time() - t0

    elapsed = check("25 sequential calls without a crash", hammer)
    if elapsed:
        print(f"        {elapsed:.2f}s total, {elapsed/25*1000:.0f} ms/call")

    # ── 5. V-Ray discovery ───────────────────────────────────────────────────
    print("\nV-Ray")
    caps = check("discover sun/sky classes", lambda: discover_vray(bridge))
    if caps:
        print(f"        renderer   : {caps.renderer}")
        print(f"        sun class  : {caps.sun_class} ({len(caps.sun_params)} params)")
        print(f"        sky class  : {caps.sky_class} ({len(caps.sky_params)} params)")
        (OUT / "vray_capabilities.json").write_text(
            json.dumps(caps.as_dict(), indent=2), encoding="utf-8"
        )
        print(f"        written    : out/vray_capabilities.json")

    # ── 6. The headline feature ──────────────────────────────────────────────
    if caps and caps.available:
        print("\nSun placement")
        setup = check(
            "place sun for Dubai Marina, 2024-10-12 16:00 local",
            lambda: build_sun(
                bridge,
                local_time=SHOT_TIME,
                latitude=SITE_LAT,
                longitude=SITE_LON,
                caps=caps,
            ),
        )
        if setup:
            s = setup.position
            print(f"        local      : {setup.time.local.isoformat()} ({setup.time.tz_name})")
            print(f"        UTC        : {setup.time.utc.isoformat()}")
            print(f"        azimuth    : {s.azimuth:.2f}deg  (>180 = west of south)")
            print(f"        altitude   : {s.altitude:.2f}deg")
            print(f"        sun node at: {[round(c) for c in setup.world_position]}")
            print(f"        exposure   : ISO {setup.exposure['iso']:.0f}, "
                  f"f/{setup.exposure['f_number']:.0f}, 1/{setup.exposure['shutter_speed']:.0f}s")
            for w in setup.warnings:
                print(f"        warning    : {w}")

            # Physical sanity: a 16:00 sun must be in the western half of the sky.
            check(
                "sun is west of south at 16:00",
                lambda: _expect(s.azimuth, lambda a: 180.0 < a < 300.0),
            )
            check("sun is above the horizon", lambda: _expect(s.altitude, lambda a: a > 0))
            (OUT / "sun_setup.json").write_text(
                json.dumps(setup.as_dict(), indent=2), encoding="utf-8"
            )

    # ── 7. Viewport capture ──────────────────────────────────────────────────
    print("\nViewport")
    shot = check(
        "capture viewport to disk",
        lambda: bridge.viewport_capture(str(OUT / "viewport.png")),
    )
    if shot:
        print(f"        {shot['path']} ({shot['bytes']:,} bytes)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'-' * 60}")
    if _failures:
        print(f"{_passes} passed, {len(_failures)} FAILED\n")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"{_passes} passed, 0 failed")
    return 0


def _expect(value, predicate):
    if not predicate(value):
        raise AssertionError(f"unexpected value: {value!r}")
    return value


def _expect_raises(fn):
    try:
        result = fn()
    except MaxBridgeError:
        return True
    raise AssertionError(f"expected an error, got {result!r}")


if __name__ == "__main__":
    sys.exit(main())
