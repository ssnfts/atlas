"""
Acceptance test: does OSM massing land in the scene at the right place and size?

Fetches the buildings around a site, extrudes them, pushes them into 3ds Max and
then checks the result **against the host rather than against this process**.
The offline suite already proves the meshes are watertight, correctly wound and
the right volume; what it cannot prove is that they survive the trip into Max at
the right scale, in the right direction, with the indices intact.

Three things are verified here that no offline test can be:

1. **Scale.** Max defaults to inches. Metres into an inch scene come out 39.37x
   too big, with nothing in any log to say so, so the bounding box is read back
   out of the scene and compared with the bounding box computed here.
2. **Orientation.** A building known to be north-east of the site must end up at
   +X and +Y. Get this wrong and every shadow in every render is rotated.
3. **Index integrity.** Vertex and face counts are read back per building, so a
   dropped or shuffled face shows up as a number rather than as a strange
   silhouette three renders later.

    .venv\\Scripts\\python.exe demo_massing.py

Needs 3ds Max running with the bridge started. See the README.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "server"))

import osm  # noqa: E402
from frame import SceneFrame  # noqa: E402
from massing import buildings_to_meshes  # noqa: E402
from maxbridge import MaxBridge, MaxBridgeError  # noqa: E402
from scene import configure_units  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

# Dubai Marina, matching demo_shadow_test.py so both demos build the same site.
LAT, LON = 25.0805, 55.1403
RADIUS_M = 400.0

# Tolerance on the round-trip bounding box, in metres. Generous enough to absorb
# float32 storage in Max, tight enough that an inch/metre mix-up (39.37x) or a
# centimetre mix-up (100x) cannot hide inside it.
BBOX_TOLERANCE_M = 0.05


def main() -> int:
    bridge = MaxBridge()
    try:
        bridge.ping()
    except MaxBridgeError as exc:
        print(f"[atlas] {exc}")
        return 1

    units = configure_units(bridge)
    print(f"[atlas] units: {units}")

    frame = SceneFrame(LAT, LON)
    print(f"[atlas] fetching buildings within {RADIUS_M:.0f} m of {LAT}, {LON}")
    buildings = osm.fetch_for_site(frame, RADIUS_M)
    print(f"[atlas] {len(buildings)} buildings")

    meshes, skipped = buildings_to_meshes(buildings, frame)
    print(f"[atlas] {len(meshes)} meshes, {len(skipped)} skipped")
    for entry in skipped:
        print(f"        skipped {entry['osm_type']} {entry['osm_id']}: {entry['reason']}")

    # What we believe we are sending, computed here, before Max sees any of it.
    expected = {
        mesh.name: {
            "verts": mesh.vertex_count,
            "faces": mesh.face_count,
            "bounds": mesh.bounds(),
        }
        for mesh in meshes
    }

    print(f"[atlas] pushing {len(meshes)} meshes")
    results = bridge.create_meshes(
        [(mesh.name, mesh.verts, mesh.faces) for mesh in meshes]
    )

    failures: list[str] = []
    checked = 0

    for step in results:
        if not step.get("ok"):
            failures.append(f"create failed: {step.get('error')}")
            continue

        result = step["result"]
        name = result["node"]
        want = expected.get(name)
        if want is None:
            failures.append(f"{name}: Max renamed the node — a name collided")
            continue

        checked += 1

        if result["vertex_count"] != want["verts"]:
            failures.append(
                f"{name}: {result['vertex_count']} vertices in Max, sent {want['verts']}"
            )
        if result["face_count"] != want["faces"]:
            failures.append(
                f"{name}: {result['face_count']} faces in Max, sent {want['faces']}"
            )

        (lo, hi) = want["bounds"]
        for axis, got, sent in zip("xyz", result["bbox_min"], lo):
            if abs(got - sent) > BBOX_TOLERANCE_M:
                ratio = got / sent if sent else float("inf")
                failures.append(
                    f"{name}: bbox min {axis} is {got:.3f} in Max, sent {sent:.3f} "
                    f"(x{ratio:.2f} — check scene units)"
                )
        for axis, got, sent in zip("xyz", result["bbox_max"], hi):
            if abs(got - sent) > BBOX_TOLERANCE_M:
                ratio = got / sent if sent else float("inf")
                failures.append(
                    f"{name}: bbox max {axis} is {got:.3f} in Max, sent {sent:.3f} "
                    f"(x{ratio:.2f} — check scene units)"
                )

    # Orientation, checked against the compass rather than against ourselves:
    # the building furthest north in geodetic terms must be furthest +Y in Max.
    northmost = max(buildings, key=lambda b: max(lat for lat, _ in b.outer))
    eastmost = max(buildings, key=lambda b: max(lon for _, lon in b.outer))
    by_name = {step["result"]["node"]: step["result"] for step in results if step.get("ok")}

    for building, axis, index, label in (
        (northmost, "+Y (north)", 1, "northmost"),
        (eastmost, "+X (east)", 0, "eastmost"),
    ):
        mesh = next((m for m in meshes if m.metadata.get("osm_id") == building.osm_id), None)
        if mesh is None or mesh.name not in by_name:
            continue
        in_max = by_name[mesh.name]["bbox_max"][index]
        best = max(r["bbox_max"][index] for r in by_name.values())
        if in_max < best - 1.0:
            failures.append(
                f"the {label} building is not furthest along {axis} in Max — "
                "the scene frame and the host disagree about the compass"
            )
        else:
            print(f"[atlas] {label} building is furthest along {axis} — correct")

    summary = {
        "site": {"lat": LAT, "lon": LON, "radius_m": RADIUS_M},
        "buildings_fetched": len(buildings),
        "meshes_built": len(meshes),
        "meshes_verified": checked,
        "skipped": skipped,
        "failures": failures,
        "attribution": osm.ATTRIBUTION,
    }
    (OUT / "massing_test.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    bridge.viewport_capture(str(OUT / "massing_viewport.png"))
    print(f"[atlas] viewport saved to {OUT / 'massing_viewport.png'}")

    if failures:
        print(f"\n[atlas] {len(failures)} FAILURES")
        for failure in failures[:20]:
            print(f"        {failure}")
        return 1

    print(f"\n[atlas] {checked} buildings verified in the host — scale, "
          f"orientation and index integrity all agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
