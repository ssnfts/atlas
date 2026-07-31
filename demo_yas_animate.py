"""
Key the whole lap into the scene: cars, cameras and tyre smoke.

The difference from ``demo_yas_lap.py`` is what is left behind. That script
renders a sequence by moving things between stills, which is restartable and
leaves an empty scene. This one writes real keys, so the file can be scrubbed,
retimed, re-framed and handed to a farm — the thing an artist actually wants
delivered.

Cars are rebuilt **at the origin** for this, not baked into world coordinates.
A baked mesh can only be animated by rebuilding it every frame; a mesh at the
origin animates with two numbers on its transform, which is also the only form
a human can check by reading the track view.
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
import tyfx                                            # noqa: E402
from frame import SceneFrame                           # noqa: E402
from materials import PRESETS                          # noqa: E402
from maxbridge import MaxBridge                        # noqa: E402
from scene import build_camera, exposure_for_altitude  # noqa: E402

LAT, LON = 24.46944, 54.60306
SITE_Z = 5.27
SUN_ALT = 7.5557
FPS = 24
LAP_SECONDS = 100.0                 # screen time for one lap, not a lap time
FIELD = 20

END_FRAME = int(FPS * LAP_SECONDS)
OUT = ROOT / "out"

b = MaxBridge()
frame = SceneFrame(LAT, LON)

payload = json.loads((OUT / "yas_raceway.json").read_text())
main = roadway.parse_ways(
    payload,
    keep=lambda t: t.get("name", "").lower() == "yas marina circuit"
    and t.get("sport") == "motor")
spine = roadway.stitch_paths(main, frame)[0].xy
lap_m = raceanim.lap_length(spine)

print(f"lap {lap_m:.1f} m over {LAP_SECONDS:.0f} s at {FPS} fps "
      f"= {END_FRAME} frames")
print(f"mean speed {lap_m / LAP_SECONDS * 3.6:.0f} km/h "
      f"(constant along the arc — blocking, not a simulation)")
print("range:", b.animation_range(0, END_FRAME, fps=FPS))

# ── cars, rebuilt at the origin so they can be transformed ───────────────────
for node in b.scene_list(prefix="car_")["nodes"]:
    b.call("delete", MaxBridge.node(node["name"]))

origin = [(0.0, 0.0, 0.0)] * FIELD
meshes = cars.cars_on_grid(origin, z=0.0)
b.create_meshes([(m.name, m.verts, m.faces) for m in meshes],
                chunk=20, timeout=900.0)
print(f"built {len(meshes)} car meshes at the origin")

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

# ── key every car round the lap ──────────────────────────────────────────────
# One key every KEY_EVERY frames. Dense enough that the spline through them
# follows the circuit rather than cutting its corners, sparse enough that the
# track view stays readable: at 24 fps a key every 8 frames is a third of a
# second, and no corner here is taken in less than two.
KEY_EVERY = 8
sample_frames = list(range(0, END_FRAME + 1, KEY_EVERY))
if sample_frames[-1] != END_FRAME:
    sample_frames.append(END_FRAME)

for slot in range(FIELD):
    keys = []
    for f in sample_frames:
        lap_fraction = f / END_FRAME
        places = raceanim.field_positions(spine, lap_fraction, count=FIELD)
        x, y, heading = places[slot]
        keys.append({"frame": f, "pos": [x, y, SITE_Z], "heading_deg": heading})

    for suffix in ("", "_tyres", "_rims"):
        node = f"car_{slot + 1:02d}{suffix}"
        result = b.set_keys(node, keys, timeout=300.0)
        if not result["moved"]:
            raise SystemExit(
                f"{node} accepted {result['keys']} keys but did not move — its "
                "transform controller is refusing them"
            )

print(f"keyed {FIELD} cars x 3 parts, {len(sample_frames)} keys each")

# ── cameras: one animated camera per shot ────────────────────────────────────
# Each shot gets its own camera and its own slice of the timeline, so the cut is
# in the scene rather than in a filename convention. A shot that does not move
# still gets two identical keys, which is what makes it obvious in the track
# view that it is locked off by intent rather than by omission.
shot_log = []
for index, shot in enumerate(raceanim.SHOTS, start=1):
    name = f"Cam_{index:02d}_{shot.name}"
    start_f = int(shot.lap_from * END_FRAME)
    end_f = max(int(shot.lap_to * END_FRAME), start_f + 1)

    first_pos, first_tgt, fov = raceanim.camera_for(
        shot, spine, shot.lap_from, SITE_Z)
    build_camera(b, position=first_pos, target=first_tgt, camera_name=name,
                 fov_degrees=fov, exposure=exposure_for_altitude(SUN_ALT))

    cam_keys, tgt_keys = [], []
    steps = max(2, (end_f - start_f) // KEY_EVERY + 1)
    for s in range(steps):
        f = start_f + round((end_f - start_f) * s / (steps - 1))
        t = shot.lap_from + (shot.lap_to - shot.lap_from) * s / (steps - 1)
        pos, tgt, _ = raceanim.camera_for(shot, spine, t, SITE_Z)
        if pos[2] <= SITE_Z:
            raise SystemExit(f"{name} at frame {f}: camera below the graded site")
        cam_keys.append({"frame": f, "pos": list(pos)})
        tgt_keys.append({"frame": f, "pos": list(tgt)})

    b.set_keys(name, cam_keys)
    b.set_keys(f"{name}_Target", tgt_keys)
    shot_log.append({"camera": name, "kind": shot.kind,
                     "frames": [start_f, end_f], "fov": fov, "note": shot.note})
    print(f"  {name:28} {shot.kind:6} f{start_f:4d}-{end_f:<4d} "
          f"{len(cam_keys)} keys  {shot.note}")

(OUT / "shot_list.json").write_text(json.dumps(shot_log, indent=2), encoding="utf-8")

# ── tyFlow: tyre smoke off the rear wheels ───────────────────────────────────
smoke = tyfx.tyre_smoke(b, [f"car_{i + 1:02d}_tyres" for i in range(FIELD)],
                        site_z=SITE_Z, end_frame=END_FRAME)
print("tyFlow:", smoke)

print(f"\nanimation range 0-{END_FRAME} at {FPS} fps. "
      f"Shot list written to {OUT / 'shot_list.json'}.")
