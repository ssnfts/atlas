"""
Yas Marina Circuit, 1:1, lit for 17:00 local on 26 December 2021.

The circuit is the subject, so it is built from the OSM raceway ways through
roadway.py; the buildings, terrain and sky are context around it.

Date matters here beyond the lighting: the 5.281 km layout this builds is the
*reconfigured* circuit, which first ran at the Abu Dhabi GP on 12 December 2021.
A shot dated two weeks later is on this layout and not the old 5.554 km one.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import median

ROOT = Path(r"C:\Users\mabdu\Desktop\atlas")
sys.path.insert(0, str(ROOT / "server"))

import cars                                                       # noqa: E402
import osm                                                        # noqa: E402
import roadway                                                    # noqa: E402
import terrain                                                    # noqa: E402
import texturing                                                  # noqa: E402
from frame import SceneFrame                                       # noqa: E402
from massing import Mesh, buildings_to_meshes                      # noqa: E402
from materials import PRESETS, group_by_material                   # noqa: E402
from maxbridge import MaxBridge                                    # noqa: E402
from scene import (apply_sky, build_camera, build_sun,             # noqa: E402
                   configure_units, discover_vray, link_sky_to_sun)
from weather import fetch_observation, sky_from_weather            # noqa: E402

# Circuit centroid, so the track sits centred on the world origin.
LAT, LON = 24.46944, 54.60306
SHOT = datetime(2021, 12, 26, 17, 0)

TRACK_W, PIT_W = 15.0, 12.0
BUILDING_RADIUS = 1200.0
TERRAIN_RADIUS = 2200.0

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)

b = MaxBridge()
frame = SceneFrame(LAT, LON)

print("=" * 64)
print(" Yas Marina Circuit — 1:1 — 17:00 local, 26 Dec 2021")
print("=" * 64)

# resetMaxFile reverts renderer to the app default and units to inches; both are
# re-established immediately after, in that order.
b.call("resetMaxFile", MaxBridge.name("noPrompt"), timeout=180.0)
print("units    :", configure_units(b)["system_units_after"])
print("renderer :", b.set_renderer("V_Ray_GPU")["resolved"])

# ── track geometry ────────────────────────────────────────────────────────────
cache = Path(__file__).with_name("yas_raceway.json")
payload = json.loads(cache.read_text()) if cache.exists() else None

def _grab(keep):
    if payload is not None:
        return roadway.parse_ways(payload, keep=keep)
    return roadway.fetch_for_site(frame, 1500.0, keep=keep)

main = _grab(lambda t: t.get("name", "").lower() == "yas marina circuit"
             and t.get("sport") == "motor")
pit = _grab(lambda t: t.get("name", "").lower() == "pit lane"
            and t.get("sport") == "motor")
print(f"\nOSM ways : {len(main)} circuit, {len(pit)} pit lane")

patch = terrain.fetch_for_site(frame, TERRAIN_RADIUS)
print(f"terrain  : {patch.rows}x{patch.cols}, relief {patch.relief():.1f} m "
      f"({patch.min_elevation():.1f}..{patch.max_elevation():.1f} m)")

# The site is graded flat, and that is a measured conclusion rather than a
# convenience. Yas Island is reclaimed land; the circuit's real relief is a
# couple of metres. GLO-30 reads 2.5..16.0 m along the lap, which draped
# vertex-by-vertex implies a 38.9% gradient — twice Eau Rouge. Levelling the
# track alone was not enough: the DEM still rose above the levelled surface at
# 30% of the circuit's vertices, burying it up to 9.7 m deep, which is why the
# first render came out with no track in it.
#
# So the ground is graded to the circuit too. The height is the **median** of the
# DEM along the lap — median, not mean, because the 16 m readings are structures
# and noise and a mean lets them pull the whole site up.
site_profile = []
for x, y in roadway.stitch_paths(main, frame)[0].xy:
    plat, plon, _ = frame.to_geodetic(x, y)
    try:
        site_profile.append(patch.elevation_at(plat, plon))
    except Exception:
        pass
SITE_Z = median(site_profile)
print(f"site     : graded flat at {SITE_Z:.2f} m "
      f"(DEM along lap {min(site_profile):.1f}..{max(site_profile):.1f} m)")

flat = lambda lat, lon: SITE_Z          # noqa: E731 - the graded site plane
track_meshes, skipped = roadway.centerlines_to_meshes(
    main, frame, width=TRACK_W, name_prefix="track",
    ground=flat, lift=0.06)
pit_meshes, _ = roadway.centerlines_to_meshes(
    pit, frame, width=PIT_W, name_prefix="pit",
    ground=flat, lift=0.03)

for m in track_meshes + pit_meshes:
    md = m.metadata
    print(f"  {m.name[:30]:30} {md['centerline_length_m']:8.1f} m  "
          f"{md['surface_area_m2']:8.0f} m2  grade {md['max_gradient_pct']:5.2f}% "
          f"(raw {md['max_gradient_pct_raw']:5.2f}%)  z={m.verts[0][2]:.2f}")
lap = sum(m.metadata["centerline_length_m"] for m in track_meshes)
print(f"  LAP LENGTH {lap:.1f} m vs published 5281 m "
      f"({100*(lap-5281)/5281:+.2f}%)")
if skipped:
    print("  skipped:", skipped)

# ── track furniture: kerbs, painted lines, the grid, and cars on it ───────────
lap = roadway.stitch_paths(main, frame)[0]

kerbs = roadway.kerb_ribbons(lap, track_width=TRACK_W, kerb_width=1.2,
                             frame=frame, ground=flat, lift=0.09)
lines = roadway.edge_lines(lap, track_width=TRACK_W, line_width=0.15,
                           frame=frame, ground=flat, lift=0.075)
boxes, placements = roadway.grid_boxes(lap, start_index=0, slots=20,
                                       track_width=TRACK_W, z=SITE_Z)
grid_cars = cars.cars_on_grid(placements, z=SITE_Z)

print(f"kerbs    : {len(kerbs)} ribbons over "
      f"{sum(len(k.faces) for k in kerbs)} faces (corners only)")
print(f"lines    : {len(lines)} edge lines")
print(f"grid     : {len(boxes)} boxes, {len(grid_cars)} cars "
      f"(generic open-wheel proxies, 2022 regulation dimensions — not F1 models)")

# The placements are written out so a bought car model can be dropped onto the
# same twenty slots these proxies stand on.
(OUT / "grid_placements.json").write_text(
    json.dumps(cars.placements_to_matrices(placements, z=SITE_Z), indent=2),
    encoding="utf-8")

# ── context buildings ─────────────────────────────────────────────────────────
buildings = osm.fetch_for_site(frame, BUILDING_RADIUS)
b_meshes, b_skipped = buildings_to_meshes(buildings, frame, flat)
print(f"\nbuildings: {len(buildings)} fetched, {len(b_meshes)} meshed, "
      f"{len(b_skipped)} skipped")
tagged = sum(1 for x in buildings if x.height_source != "default (no height or level tags)")
print(f"           {tagged}/{len(buildings)} have a real height "
      f"({100*tagged/max(len(buildings),1):.1f}%)")

# ── push everything ───────────────────────────────────────────────────────────
# The ground sheet, graded flat with the circuit. Subdivided rather than two
# giant triangles so the triplanar ground texture and any later displacement
# have vertices to work with.
def flat_sheet(name, half, z, cells=48):
    step = (2.0 * half) / cells
    verts, faces = [], []
    for r in range(cells + 1):
        for c in range(cells + 1):
            verts.append((-half + c * step, -half + r * step, z))
    for r in range(cells):
        for c in range(cells):
            a = r * (cells + 1) + c
            b_, cc, d = a + 1, a + cells + 1, a + cells + 2
            faces += [(a, b_, d), (a, d, cc)]   # CCW seen from above
    m = Mesh(verts=verts, faces=faces, name=name)
    m.metadata.update({"kind": "graded_site", "z": round(z, 3)})
    return m

terrain_mesh = flat_sheet("atlas_terrain", TERRAIN_RADIUS, SITE_Z)
# UVs ride along as a fourth element where a mesh has them; the kerbs need them
# (their stripes are UV-space) and nothing else is harmed by carrying them.
push = [(terrain_mesh.name, terrain_mesh.verts, terrain_mesh.faces)]
push += [(m.name, m.verts, m.faces, m.uvs)
         for m in track_meshes + pit_meshes + kerbs + lines]
push += [(m.name, m.verts, m.faces) for m in boxes + grid_cars]
push += [(m.name, m.verts, m.faces) for m in b_meshes]
results = b.create_meshes(push, chunk=25, timeout=1200.0)
failed = [r for r in results if not r.get("ok")]
print(f"\npushed   : {len(results)-len(failed)}/{len(results)} meshes"
      + (f", {len(failed)} FAILED" if failed else ""))
for f in failed[:3]:
    print("   ", f.get("error"))

# ── sun, sky ──────────────────────────────────────────────────────────────────
caps = discover_vray(b)
sun = build_sun(b, local_time=SHOT, latitude=LAT, longitude=LON, caps=caps)
print(f"\nsun      : az {sun.position.azimuth:.2f}deg  alt "
      f"{sun.position.altitude:.2f}deg  ({sun.time.local.isoformat()})")
for w in sun.warnings:
    print("   warn:", w)

obs = fetch_observation(sun.time.utc, LAT, LON)
print(f"weather  : {obs.description}, {obs.temperature_c}C, "
      f"cloud {obs.cloud_cover_pct}%, wind {obs.wind_speed_ms} m/s")
sky = apply_sky(b, sky_from_weather(obs).params, caps=caps)
if sky["rejected"]:
    print("   sky rejected:", sky["rejected"])
link_sky_to_sun(b, caps=caps)

# ── materials ─────────────────────────────────────────────────────────────────
def build_graph(key, nodes, label):
    graph = texturing.GRAPHS[key]
    res = b.build_material(graph.as_dict(), nodes,
                           name=PRESETS[key].name, timeout=900.0)
    rej = res.get("rejected") or {}
    print(f"  {label:22} {len(nodes):5d} nodes  rejected={len(rej)}"
          + (f"  {list(rej)[:2]}" if rej else ""))
    return rej

def assign_flat(key, nodes, label):
    """A plain VRayMtl, for surfaces whose look is a colour and a gloss rather
    than a pattern — paint and car bodywork."""
    if not nodes:
        return
    res = b.assign_material(nodes, params=PRESETS[key].to_params(),
                            name=PRESETS[key].name)
    rej = res.get("rejected") or {}
    print(f"  {label:22} {len(nodes):5d} nodes  rejected={len(rej)}")


print("\nmaterials:")
track_nodes = [m.name for m in track_meshes + pit_meshes]
build_graph("track_asphalt", track_nodes, "track_asphalt")
build_graph("ground", ["atlas_terrain"], "ground")
if kerbs:
    build_graph("kerb", [m.name for m in kerbs], "kerb")
assign_flat("line_paint", [m.name for m in lines + boxes], "line_paint")

# Cars come back as three meshes each — body, tyres, rims — because rubber,
# machined aluminium and painted carbon are three genuinely different surfaces
# and one material for all three is what makes a proxy look like a toy.
assign_flat("tyre", [m.name for m in grid_cars if m.name.endswith("_tyres")], "tyre")
assign_flat("rim", [m.name for m in grid_cars if m.name.endswith("_rims")], "rim")

# Bodywork: one material per colour, not per car. Colours only — no livery, no
# marks, no team names anywhere in this scene.
bodies: dict[str, list[str]] = {}
for m in grid_cars:
    if m.name.endswith("_tyres") or m.name.endswith("_rims"):
        continue
    label, rgb = m.metadata["colour"]
    bodies.setdefault(label, []).append(m.name)

base = PRESETS["car_body"]
for label, nodes in sorted(bodies.items()):
    rgb = next(c for lbl, c in cars.RACING_COLOURS if lbl == label)
    params = base.to_params()
    params["diffuse"] = {"__color__": list(rgb)}
    res = b.assign_material(nodes, params=params, name=f"atlas_car_{label.replace(' ', '_')}")
    print(f"  {'car ' + label:22} {len(nodes):5d} nodes  "
          f"rejected={len(res.get('rejected') or {})}")

pairs = [(bu, m.name) for bu, m in zip(buildings, b_meshes)]
for key, nodes in group_by_material(pairs).items():
    build_graph(key, nodes, key)

# ── camera & render ───────────────────────────────────────────────────────────
# Sun bears 240deg (WSW) at 7.6deg. The camera stands 2.1 km out on that same
# bearing, so the sun is directly behind it: the circuit is fully front-lit and
# every shadow runs away from the lens rather than toward it. At 7.6deg a 20 m
# grandstand throws a 150 m shadow, which is the whole reason to shoot this hour.
cam = build_camera(b, position=(-1571.0, -884.0, 850.0), target=(248.0, 166.0, 8.0),
                   camera_name="Atlas_YasCam", fov_degrees=55.0,
                   exposure=sun.exposure)
print("\ncamera   : direction_verified =",
      cam["applied"].get("direction_verified", cam["applied"].get("direction_error")))
print("exposure :", sun.exposure)

out = OUT / "yas_marina_1700.png"
r = b.render(str(out), camera="Atlas_YasCam", width=1600, height=900,
             expect_renderer="V_Ray", timeout=3600.0)
print(f"\nrendered : {r['path']}  {r['bytes']} bytes")
