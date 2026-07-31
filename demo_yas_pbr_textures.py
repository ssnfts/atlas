"""Apply the Yas anti-tiling CC0 PBR material pass to the live Max scene.

This is intentionally a scene *updater*, not another site builder. Rebuilding
the track or OSM massing would risk deleting the keyed cars, cameras and tyFlow
flows the animation pass already verified. It only changes the three road/pit
surfaces and the measured ordinary-building set, and saves after every material
milestone.

The pass has a strict no-render rule. It proves the material topology and live
host acceptance structurally; judging its artistic look is deferred until the
user later requests a viewport or render review.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\mabdu\Desktop\atlas")
OUT = ROOT / "out"
sys.path.insert(0, str(ROOT / "server"))

from materials import PRESETS  # noqa: E402
from maxbridge import MaxBridge  # noqa: E402
from texlib import fetch_texture, local_set  # noqa: E402
from texturing import anti_tiling_scanned_graph, validate_graph  # noqa: E402


TRACK_PREFIXES = ("track_", "pit_")
SUPPORTED_GRANDSTAND_NODES = frozenset({
    "osm_w133907079_Marina Grandstand",
    "osm_w133907081_South Grandstand",
    "osm_w133907352_West Grandstand",
    "osm_w168411369_North Grandstand",
    "osm_w188731381_Main Grandstand",
    "osm_w1071647267_Etihad Arena",
})

# These are measured scene facts, deliberately pinned so an OSM rebuild cannot
# quietly turn this narrow material pass into an assignment to the wrong city.
EXPECTED_TRACK_NODES = 3
EXPECTED_OSM_NODES = 379
EXPECTED_ORDINARY_BUILDINGS = 373
EXPECTED_CARS = 20
REQUIRED_MAPS = ("Diffuse", "Rough", "nor_gl")


def require_maps(label: str, texture_set: dict) -> dict:
    """Refuse a partial PBR set before it is allowed near the live scene."""
    missing = [name for name in REQUIRED_MAPS if not texture_set.get(name)]
    if missing:
        raise RuntimeError(f"{label} is missing required PBR maps: {', '.join(missing)}")
    return texture_set


def material_result(label: str, result: dict, expected_nodes: int) -> None:
    """Treat every host rejection or incomplete assignment as a hard failure."""
    rejected = result.get("rejected") or {}
    if rejected:
        raise RuntimeError(f"{label}: V-Ray rejected graph writes: {rejected}")
    if result.get("assigned_to") != expected_nodes:
        raise RuntimeError(
            f"{label}: assigned {result.get('assigned_to')} nodes, expected {expected_nodes}"
        )
    if result.get("texmaps_built") != result.get("texmaps_requested"):
        raise RuntimeError(
            f"{label}: built {result.get('texmaps_built')} texmaps, requested "
            f"{result.get('texmaps_requested')}"
        )
    print(
        f"{label}: {result['assigned_to']} nodes, {result['texmaps_built']} texmaps, "
        "rejected=0"
    )


def verify_car_clearcoats(bridge: MaxBridge, bodies: list[str]) -> None:
    """Confirm generic bodies stay clearcoated paint rather than tiled bitmaps."""
    if len(bodies) != EXPECTED_CARS:
        raise RuntimeError(f"expected {EXPECTED_CARS} car bodies, found {len(bodies)}")

    for name in bodies:
        # Scalar reads deliberately use MaxScript only after it was operator-
        # enabled. There is no bridge path to a geometry node's material pblock.
        # The expression is wrapped in parentheses because MaxScript locals have
        # leaked across listener calls in this host when not scoped.
        material = f'(getNodeByName "{name}").material'
        coat_amount = float(bridge.maxscript(f"({material}.coat_amount)"))
        coat_gloss = float(bridge.maxscript(f"({material}.coat_glossiness)"))
        coat_ior = float(bridge.maxscript(f"({material}.coat_ior)"))
        diffuse_map = str(bridge.maxscript(
            f"(local m = {material}; if m.texmap_diffuse == undefined then \"undefined\" "
            "else (classOf m.texmap_diffuse as string))"
        ))

        if not math.isclose(coat_amount, 0.80, abs_tol=1e-4):
            raise RuntimeError(f"{name}: coat_amount={coat_amount}, expected 0.80")
        if not math.isclose(coat_gloss, 0.96, abs_tol=1e-4):
            raise RuntimeError(f"{name}: coat_glossiness={coat_gloss}, expected 0.96")
        if not math.isclose(coat_ior, 1.52, abs_tol=1e-4):
            raise RuntimeError(f"{name}: coat_ior={coat_ior}, expected 1.52")
        if diffuse_map != "undefined":
            raise RuntimeError(f"{name}: car body gained a diffuse texmap ({diffuse_map})")

    print(f"cars: {len(bodies)} generic clearcoat bodies verified; bitmap body maps=0")


def verify_assignment_boundaries(bridge: MaxBridge) -> None:
    """Prove the narrow material pass did not spill into protected surfaces."""
    track_count = int(bridge.maxscript(
        "(local n = 0; for o in objects where "
        "(o.material != undefined and o.material.name == \"atlas_track_asphalt_antitile\") "
        "do n += 1; n)"
    ))
    building_count = int(bridge.maxscript(
        "(local n = 0; for o in objects where "
        "(o.material != undefined and o.material.name == \"atlas_concrete_antitile\") "
        "do n += 1; n)"
    ))
    if track_count != EXPECTED_TRACK_NODES:
        raise RuntimeError(
            f"track material is assigned to {track_count} nodes, expected {EXPECTED_TRACK_NODES}"
        )
    if building_count != EXPECTED_ORDINARY_BUILDINGS:
        raise RuntimeError(
            f"building material is assigned to {building_count} nodes, "
            f"expected {EXPECTED_ORDINARY_BUILDINGS}"
        )

    overwritten = []
    for name in sorted(SUPPORTED_GRANDSTAND_NODES):
        material_name = str(bridge.maxscript(
            f'(local m = (getNodeByName "{name}").material; '
            'if m == undefined then "undefined" else m.name)'
        ))
        if material_name == "atlas_concrete_antitile":
            overwritten.append(name)
    if overwritten:
        raise RuntimeError(f"protected grandstands were overwritten: {overwritten}")
    print(
        f"assignments: track={track_count}, ordinary_buildings={building_count}, "
        "protected_grandstands_overwritten=0"
    )


def main() -> None:
    # Fetch happens before the bridge is contacted. A network/asset failure
    # cannot leave Max half-updated.
    track_primary = require_maps(
        "Poly Haven asphalt_track",
        fetch_texture("asphalt_track", resolution="4k", maps=REQUIRED_MAPS),
    )
    track_secondary = require_maps("local asphalt_02", local_set("asphalt_02"))
    building_primary = require_maps(
        "Poly Haven concrete_slab_wall_02",
        fetch_texture("concrete_slab_wall_02", resolution="4k", maps=REQUIRED_MAPS),
    )
    building_secondary = require_maps(
        "local concrete_layers_02", local_set("concrete_layers_02"))

    track_graph = anti_tiling_scanned_graph(
        "yas_track_antitile",
        PRESETS["track_asphalt"],
        track_primary,
        track_secondary,
        primary_size_m=2.0,
        secondary_size_m=3.236,
        macro_size_m=37.0,
        normal_multiplier=0.0,
        note="Yas track: dual CC0 asphalt scans, 37m macro breakup, no false bump",
    )
    building_graph = anti_tiling_scanned_graph(
        "yas_buildings_antitile",
        PRESETS["concrete"],
        building_primary,
        building_secondary,
        primary_size_m=2.1,
        secondary_size_m=3.3978,
        macro_size_m=19.0,
        normal_multiplier=0.35,
        note="Yas buildings: dual CC0 concrete scans, 19m macro facade breakup",
    )
    for graph in (track_graph, building_graph):
        problems = validate_graph(graph)
        if problems:
            raise RuntimeError(f"{graph.key}: invalid graph: {problems}")

    bridge = MaxBridge()
    ping = bridge.ping()
    if not ping.get("maxscript_enabled"):
        raise RuntimeError("ATLAS_ALLOW_MAXSCRIPT=1 is required for car-material verification")

    names = {node["name"] for node in bridge.scene_list()["nodes"]}
    track_nodes = sorted(name for name in names if name.startswith(TRACK_PREFIXES))
    osm_nodes = sorted(name for name in names if name.startswith("osm_"))
    ordinary_buildings = [name for name in osm_nodes if name not in SUPPORTED_GRANDSTAND_NODES]
    car_bodies = sorted(
        name for name in names
        if name.startswith("car_") and not name.endswith(("_tyres", "_rims"))
    )

    if len(track_nodes) != EXPECTED_TRACK_NODES:
        raise RuntimeError(f"expected {EXPECTED_TRACK_NODES} track/pit nodes, found {track_nodes}")
    if len(osm_nodes) != EXPECTED_OSM_NODES:
        raise RuntimeError(f"expected {EXPECTED_OSM_NODES} OSM nodes, found {len(osm_nodes)}")
    missing_grandstands = SUPPORTED_GRANDSTAND_NODES - names
    if missing_grandstands:
        raise RuntimeError(f"expected supported nodes are absent: {sorted(missing_grandstands)}")
    if len(ordinary_buildings) != EXPECTED_ORDINARY_BUILDINGS:
        raise RuntimeError(
            f"expected {EXPECTED_ORDINARY_BUILDINGS} ordinary buildings, "
            f"found {len(ordinary_buildings)}"
        )

    # Construct the material with no assignment first. The bridge returns a
    # rejected-property map instead of allowing a typo to silently flatten 373
    # facades. Do not proceed unless that map is empty.
    track_probe = bridge.build_material(
        track_graph.as_dict(), [], name="atlas_track_antitile_probe", timeout=900.0
    )
    material_result("track probe", track_probe, 0)

    track_result = bridge.build_material(
        track_graph.as_dict(), track_nodes, name="atlas_track_asphalt_antitile", timeout=900.0
    )
    material_result("track", track_result, EXPECTED_TRACK_NODES)
    print("checkpoint:", bridge.save_scene(str(OUT / "yas_race_pbr_track.max")))

    building_result = bridge.build_material(
        building_graph.as_dict(), ordinary_buildings,
        name="atlas_concrete_antitile", timeout=900.0,
    )
    material_result("ordinary buildings", building_result, EXPECTED_ORDINARY_BUILDINGS)
    print("checkpoint:", bridge.save_scene(str(OUT / "yas_race_pbr_buildings.max")))

    verify_assignment_boundaries(bridge)
    verify_car_clearcoats(bridge, car_bodies)
    print("checkpoint:", bridge.save_scene(str(OUT / "yas_race_hyperreal_pbr.max")))


if __name__ == "__main__":
    main()
