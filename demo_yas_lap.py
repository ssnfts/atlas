"""
Render the twelve-shot lap.

Expects the Yas Marina scene already standing in Max (build it with
``demo_yas_marina.py``); this only moves the field, places a camera and renders,
so it costs renders rather than a rebuild.

Each shot is one frame by default. To turn a shot into a sequence, raise
``FRAMES`` — the cars and camera are both functions of lap fraction, so the
whole cut list animates from the same two numbers with nothing keyframed in the
scene. That is also what makes it restartable: a frame is a scene state, not a
position on a timeline, so an interrupted run resumes by rendering the frames
that are missing.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\mabdu\Desktop\atlas")
sys.path.insert(0, str(ROOT / "server"))

import cars                                            # noqa: E402
import raceanim                                        # noqa: E402
import roadway                                         # noqa: E402
from frame import SceneFrame                           # noqa: E402
from materials import PRESETS                          # noqa: E402
from maxbridge import MaxBridge                        # noqa: E402
from scene import build_camera, exposure_for_altitude  # noqa: E402

LAT, LON = 24.46944, 54.60306
SITE_Z = 5.27
SUN_ALT = 7.5557
FRAMES = 1                      # frames per shot; 1 = a still per shot
WIDTH, HEIGHT = 1280, 720

OUT = ROOT / "out" / "lap"
OUT.mkdir(parents=True, exist_ok=True)

b = MaxBridge()
frame = SceneFrame(LAT, LON)

cache = Path(__file__).parent / "out" / "yas_raceway.json"
if cache.exists():
    payload = json.loads(cache.read_text())
    main = roadway.parse_ways(
        payload,
        keep=lambda t: t.get("name", "").lower() == "yas marina circuit"
        and t.get("sport") == "motor")
else:
    main = roadway.fetch_for_site(
        frame, 1500.0,
        keep=lambda t: t.get("name", "").lower() == "yas marina circuit"
        and t.get("sport") == "motor")

spine = roadway.stitch_paths(main, frame)[0].xy
print(f"lap {raceanim.lap_length(spine):.1f} m, {len(raceanim.SHOTS)} shots, "
      f"{FRAMES} frame(s) each")

exposure = exposure_for_altitude(SUN_ALT)


def place_field(lap_fraction: float) -> None:
    """Rebuild the field at a point in the lap.

    Rebuilt rather than transformed because the meshes are baked in world
    coordinates: moving them would mean sending a matrix per node, and the
    bridge deliberately has no matrix marshalling. Sixty small meshes in one
    batched call is cheap next to the render that follows.
    """
    for node in b.scene_list(prefix="car_")["nodes"]:
        b.call("delete", MaxBridge.node(node["name"]))

    places = raceanim.field_positions(spine, lap_fraction, count=20)
    meshes = cars.cars_on_grid(places, z=SITE_Z)
    b.create_meshes([(m.name, m.verts, m.faces) for m in meshes],
                    chunk=20, timeout=900.0)

    b.assign_material([m.name for m in meshes if m.name.endswith("_tyres")],
                      params=PRESETS["tyre"].to_params(), name=PRESETS["tyre"].name)
    b.assign_material([m.name for m in meshes if m.name.endswith("_rims")],
                      params=PRESETS["rim"].to_params(), name=PRESETS["rim"].name)

    bodies: dict[str, list[str]] = {}
    for m in meshes:
        if m.name.endswith(("_tyres", "_rims")):
            continue
        bodies.setdefault(m.metadata["colour"][0], []).append(m.name)
    for label, nodes in bodies.items():
        rgb = next(c for lbl, c in cars.RACING_COLOURS if lbl == label)
        params = PRESETS["car_body"].to_params()
        params["diffuse"] = {"__color__": list(rgb)}
        b.assign_material(nodes, params=params,
                          name="atlas_car_" + label.replace(" ", "_"))


for index, shot in enumerate(raceanim.SHOTS, start=1):
    for f in range(FRAMES):
        t = (shot.lap_from if FRAMES == 1
             else shot.lap_from + (shot.lap_to - shot.lap_from) * f / max(FRAMES - 1, 1))
        out = OUT / f"{shot.name}_{f:03d}.png"
        if out.exists():
            print(f"  {shot.name} f{f:03d}  already rendered, skipping")
            continue

        place_field(t)
        position, target, fov = raceanim.camera_for(shot, spine, t, SITE_Z)
        if position[2] <= SITE_Z:
            raise SystemExit(
                f"{shot.name}: camera at z={position[2]:.2f} is at or below the "
                f"graded site ({SITE_Z:.2f}) — it would render the underside of "
                "the ground sheet"
            )

        cam = build_camera(b, position=position, target=target,
                           camera_name="Atlas_Lap", fov_degrees=fov,
                           exposure=exposure)
        if not cam["applied"].get("direction_verified"):
            print(f"    WARNING {shot.name}: {cam['applied'].get('direction_error')}")

        result = b.render(str(out), camera="Atlas_Lap", width=WIDTH, height=HEIGHT,
                          expect_renderer="V_Ray", timeout=3600.0)

        # A camera inside geometry renders the unlit backs of faces: a frame
        # that is uniformly black, with no error anywhere. It happened once
        # here, to a shot standing 95 m off the circuit that turned out to be
        # inside a building, and it was caught by eye on a contact sheet. File
        # size is the cheapest tell — a flat frame compresses to almost
        # nothing — so the check is done here rather than left to whoever looks
        # at the sequence next.
        flat = result["bytes"] < 40_000
        print(f"  {index:02d} {shot.name:18} {shot.kind:6} lap {t:.3f}  "
              f"{result['bytes'] / 1000:.0f} kB  {shot.note}"
              + ("   <-- FLAT, camera is probably inside geometry" if flat else ""))

print(f"\nwrote {len(list(OUT.glob('*.png')))} frames to {OUT}")
