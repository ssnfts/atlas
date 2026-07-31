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
FIELD = 20

# The timeline is the *edit*, not the lap. Shots each declare their screen time
# and are laid end to end; the cars are driven from whatever lap moment the cut
# under the playhead is watching. Deriving the timeline from the lap instead was
# what gave every locked-off shot a single frame.
CUTS = raceanim.build_edit(fps=FPS, lap_seconds=raceanim.lap_time(
    roadway.stitch_paths(
        roadway.parse_ways(
            json.loads((ROOT / 'out' / 'yas_raceway.json').read_text()),
            keep=lambda t: t.get('name', '').lower() == 'yas marina circuit'
            and t.get('sport') == 'motor'),
        SceneFrame(LAT, LON))[0].xy))
END_FRAME = raceanim.edit_length_frames(CUTS)
OUT = ROOT / "out"
SCENE_FILE = OUT / "yas_race.max"

b = MaxBridge()
frame = SceneFrame(LAT, LON)

payload = json.loads((OUT / "yas_raceway.json").read_text())
main = roadway.parse_ways(
    payload,
    keep=lambda t: t.get("name", "").lower() == "yas marina circuit"
    and t.get("sport") == "motor")
spine = roadway.stitch_paths(main, frame)[0].xy
lap_m = raceanim.lap_length(spine)

lap_s = raceanim.lap_time(spine)
speeds = raceanim.speed_profile(spine)
print(f"lap {lap_m:.1f} m, modelled {lap_s:.1f} s "
      f"({min(speeds) * 3.6:.0f}-{max(speeds) * 3.6:.0f} km/h)")
print(f"edit: {len(CUTS)} cuts, {END_FRAME} frames = {END_FRAME / FPS:.1f} s "
      f"at {FPS} fps")
print("range:", b.animation_range(0, END_FRAME, fps=FPS))

# ── cars, rebuilt at the origin so they can be transformed ───────────────────
for node in b.scene_list(prefix="car_")["nodes"]:
    b.call("delete", MaxBridge.node(node["name"]))

origin = [(0.0, 0.0, 0.0)] * FIELD
meshes = cars.cars_on_grid(origin, z=0.0)
b.create_meshes([(m.name, m.verts, m.faces) for m in meshes],
                chunk=20, timeout=900.0)
print(f"built {len(meshes)} car meshes at the origin")
print("checkpoint:", b.save_scene(str(SCENE_FILE)))

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
KEY_EVERY = 4
sample_frames = list(range(0, END_FRAME + 1, KEY_EVERY))
if sample_frames[-1] != END_FRAME:
    sample_frames.append(END_FRAME)


def lap_seconds_at(frame: int) -> float:
    """
    The lap moment at a timeline frame.

    A straight proportion, because the timeline *is* the lap: the field runs
    once round without interruption and the cameras cut around it. An earlier
    version looked this up per cut, which let consecutive shots watch
    non-adjacent parts of the circuit — and the cars then teleported between
    them, sliding 500 m across a single key interval.
    """
    return (frame / END_FRAME) * lap_s


car_keys: dict[int, list[dict]] = {slot: [] for slot in range(FIELD)}
for f in sample_frames:
    places = raceanim.field_at_time(spine, lap_seconds_at(f), count=FIELD)
    for slot, (x, y, heading, roll, pitch) in enumerate(places):
        quat = raceanim.orientation_quat(heading, pitch, roll)
        car_keys[slot].append({"frame": f, "pos": [x, y, SITE_Z],
                               "quat": list(quat)})

for slot in range(FIELD):
    for suffix in ("", "_tyres", "_rims"):
        node = f"car_{slot + 1:02d}{suffix}"
        result = b.set_keys(node, car_keys[slot], timeout=600.0)
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
for index, cut in enumerate(CUTS, start=1):
    shot = cut.shot
    # Plain sequential names. The descriptive part lives in shot_list.json,
    # because a camera list is read in a sequencer where "Shot_07" sorts and
    # scans and "Cam_07_07_long_lens" does not.
    name = f"Shot_{index:02d}"
    start_f, end_f = cut.start_frame, cut.end_frame

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
    shot_log.append({"camera": name, "shot": shot.name, "kind": shot.kind,
                     "frames": [start_f, end_f],
                     "seconds": [round(start_f / FPS, 2), round(end_f / FPS, 2)],
                     "lap": [shot.lap_from, shot.lap_to],
                     "fov": fov, "note": shot.note})
    print(f"  {name:9} {shot.kind:6} f{start_f:4d}-{end_f:<4d} "
          f"{len(cam_keys):3d} keys  {shot.note}")

(OUT / "shot_list.json").write_text(json.dumps(shot_log, indent=2), encoding="utf-8")

# Checkpoint. 3ds Max has exited mid-session four times during this project and
# taken the scene with it each time; the geometry is reproducible from source
# but a rebuild costs minutes, and any manual edit is not reproducible at all.
# save_scene verifies from disk rather than trusting saveMaxFile's return.
print("checkpoint:", b.save_scene(str(SCENE_FILE)))

# ── tyFlow: tyre smoke off the rear wheels ───────────────────────────────────
smoke = tyfx.tyre_smoke(b, [f"car_{i + 1:02d}_tyres" for i in range(FIELD)],
                        site_z=SITE_Z, end_frame=END_FRAME)
print("tyFlow:", smoke)

print(f"\nanimation range 0-{END_FRAME} at {FPS} fps. "
      f"Shot list written to {OUT / 'shot_list.json'}.")
