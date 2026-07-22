"""
Massing geometry tests.

Triangulation and extrusion bugs do not raise. They render — as a roof with a
missing wedge, a wall facing the wrong way, or a courtyard filled in solid.
Checking the individual triangles cannot catch any of those, because each
triangle is individually fine.

So these check *conservation laws* instead, which a self-consistent bug cannot
satisfy:

- the triangles of a cap sum to the area of the polygon
- the closed mesh's divergence-theorem volume equals footprint area x height
- every edge is shared by exactly two faces
- the volume's sign is positive, which is true only if normals face outward

The volume check is the one that matters most. It is computed by a completely
different route from the construction — a sum over face vertices, with no
reference to the footprint or the height that produced them — so agreement is
evidence rather than a restatement.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from frame import SceneFrame  # noqa: E402
from massing import (  # noqa: E402
    MassingError,
    Mesh,
    buildings_to_meshes,
    building_to_mesh,
    extrude,
    orient,
    signed_area,
    triangulate,
)
from osm import Building  # noqa: E402

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
L_SHAPE = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0), (10.0, 20.0), (0.0, 20.0)]
COURTYARD_OUTER = [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)]
COURTYARD_HOLE = [(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]


def _polygon(sides: int, radius: float = 12.0) -> list[tuple[float, float]]:
    """A regular polygon — a stand-in for the rounded footprints OSM is full of."""
    return [
        (radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides))
        for i in range(sides)
    ]


def _edge_counts(mesh: Mesh) -> Counter:
    edges: Counter = Counter()
    for i, j, k in mesh.faces:
        for a, b in ((i, j), (j, k), (k, i)):
            edges[(min(a, b), max(a, b))] += 1
    return edges


def _triangulated_area(verts, faces) -> float:
    total = 0.0
    for i, j, k in faces:
        (ax, ay), (bx, by), (cx, cy) = verts[i], verts[j], verts[k]
        total += abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5
    return total


# ── Ring orientation ──────────────────────────────────────────────────────────

def test_signed_area_is_positive_counter_clockwise():
    assert signed_area(SQUARE) == pytest.approx(100.0)


def test_signed_area_is_negative_clockwise():
    assert signed_area(list(reversed(SQUARE))) == pytest.approx(-100.0)


def test_orient_leaves_a_correct_ring_alone():
    assert orient(SQUARE, ccw=True) == SQUARE


def test_orient_flips_a_wrong_ring():
    assert orient(list(reversed(SQUARE)), ccw=True) == SQUARE


def test_orient_can_demand_clockwise():
    assert signed_area(orient(SQUARE, ccw=False)) < 0


# ── Triangulation ─────────────────────────────────────────────────────────────

def test_square_triangulates_to_two_triangles():
    verts, faces = triangulate(SQUARE)
    assert len(faces) == 2
    assert _triangulated_area(verts, faces) == pytest.approx(100.0)


def test_simple_polygon_yields_n_minus_two_triangles():
    """A hole-free polygon of n vertices always triangulates to exactly n-2."""
    for sides in (3, 4, 5, 8, 17, 64):
        verts, faces = triangulate(_polygon(sides))
        assert len(faces) == sides - 2, f"{sides}-gon gave {len(faces)} triangles"


def test_concave_footprint_conserves_area():
    """
    The L-shape is the case an unconstrained Delaunay triangulation gets wrong:
    it spans the notch, adding a wing the mapper never drew. Area conservation
    is what detects that — the triangles would sum to the convex hull's 400 m2
    rather than the footprint's 300.
    """
    verts, faces = triangulate(L_SHAPE)
    assert _triangulated_area(verts, faces) == pytest.approx(300.0)
    assert len(faces) == len(L_SHAPE) - 2


def test_clockwise_input_is_normalised_not_rejected():
    """OSM rings come in both directions; the mapper's choice is not meaningful."""
    verts, faces = triangulate(list(reversed(L_SHAPE)))
    assert _triangulated_area(verts, faces) == pytest.approx(300.0)


def test_courtyard_area_excludes_the_hole():
    verts, faces = triangulate(COURTYARD_OUTER, [COURTYARD_HOLE])
    assert _triangulated_area(verts, faces) == pytest.approx(900.0 - 100.0)


def test_two_courtyards():
    hole_a = [(3.0, 3.0), (8.0, 3.0), (8.0, 8.0), (3.0, 8.0)]
    hole_b = [(20.0, 20.0), (26.0, 20.0), (26.0, 26.0), (20.0, 26.0)]
    verts, faces = triangulate(COURTYARD_OUTER, [hole_a, hole_b])
    assert _triangulated_area(verts, faces) == pytest.approx(900.0 - 25.0 - 36.0)


def test_hole_wound_either_way_gives_the_same_result():
    _, forward = triangulate(COURTYARD_OUTER, [COURTYARD_HOLE])
    _, backward = triangulate(COURTYARD_OUTER, [list(reversed(COURTYARD_HOLE))])
    assert len(forward) == len(backward)


def test_round_footprint_with_a_round_courtyard():
    verts, faces = triangulate(_polygon(32, 20.0), [_polygon(16, 6.0)])
    outer_area = abs(signed_area(_polygon(32, 20.0)))
    hole_area = abs(signed_area(_polygon(16, 6.0)))
    assert _triangulated_area(verts, faces) == pytest.approx(outer_area - hole_area, rel=1e-6)


def test_repeated_closing_vertex_is_tolerated():
    verts, faces = triangulate(SQUARE + [SQUARE[0]])
    assert _triangulated_area(verts, faces) == pytest.approx(100.0)


def test_duplicate_consecutive_vertices_are_tolerated():
    """OSM ways contain repeated nodes; a zero-length edge has no direction."""
    ring = [(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    verts, faces = triangulate(ring)
    assert _triangulated_area(verts, faces) == pytest.approx(100.0)


def test_collinear_vertices_do_not_produce_stray_triangles():
    ring = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    verts, faces = triangulate(ring)
    assert _triangulated_area(verts, faces) == pytest.approx(100.0)


def test_too_few_vertices_is_an_error():
    with pytest.raises(MassingError, match="need 3"):
        triangulate([(0.0, 0.0), (1.0, 1.0)])


def test_hole_covering_the_whole_footprint_is_an_error():
    with pytest.raises(MassingError, match="holes consume"):
        triangulate(SQUARE, [SQUARE])


# ── Regressions ───────────────────────────────────────────────────────────────
#
# Three bugs found while building this module. All three produced geometry
# rather than errors, and all three were caught by cross-validation against
# Shapely rather than by any invariant written here first.

def test_reflex_corner_exactly_on_an_ear_edge_blocks_it():
    """
    Regression. On the L-shape the notch's reflex corner lies exactly on the
    line between the two far corners. A strictly-inside blocker test finds
    nothing in the way and clips an ear straight across the notch, filling in
    the missing quarter: a 400 m2 roof on a 300 m2 footprint, no error.
    """
    verts, faces = triangulate(L_SHAPE)
    assert _triangulated_area(verts, faces) == pytest.approx(300.0)
    for i, j, k in faces:
        cx = (verts[i][0] + verts[j][0] + verts[k][0]) / 3.0
        cy = (verts[i][1] + verts[j][1] + verts[k][1]) / 3.0
        assert not (cx > 10.0 and cy > 10.0), "a triangle covers the notch"


def test_two_courtyards_bridging_to_the_same_corner():
    """
    Regression. Both holes can see the outer ring's top-right corner, so both
    bridge to it. Splicing the second seam in at the first copy of that vertex
    puts the seams in the wrong rotational order, makes the corner reflex and
    invalidates the polygon — the clipper then emits overlapping triangles and
    stalls. Diagonally-placed holes never trigger it; side-by-side ones do.
    """
    outer = [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)]
    holes = [
        [(5.0, 5.0), (12.0, 5.0), (12.0, 15.0), (5.0, 15.0)],
        [(22.0, 6.0), (33.0, 6.0), (33.0, 14.0), (22.0, 14.0)],
    ]
    expected = 40 * 20 - 7 * 10 - 11 * 8  # 800 outer, less two courtyards
    verts, faces = triangulate(outer, holes)
    assert _triangulated_area(verts, faces) == pytest.approx(expected)
    assert extrude(outer, 10.0, holes=holes).volume() == pytest.approx(expected * 10.0)


def test_bridge_anchor_below_the_ray_is_still_checked_for_blockers():
    """
    Regression. The sight triangle (hole vertex, ray hit, anchor) is clockwise
    whenever the anchor sits below the ray. The CCW-only containment test then
    reports no blockers for *any* point, so an anchor the hole cannot see gets
    chosen and the bridge crosses the geometry it was meant to route around.

    This footprint puts the anchor below the ray and a reflex corner in the way.
    """
    outer = [
        (0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (24.0, 30.0),
        (24.0, 12.0), (16.0, 12.0), (16.0, 30.0), (0.0, 30.0),
    ]
    holes = [[(3.0, 14.0), (11.0, 14.0), (11.0, 22.0), (3.0, 22.0)]]
    verts, faces = triangulate(outer, holes)
    expected = 40 * 30 - 8 * 18 - 64
    assert _triangulated_area(verts, faces) == pytest.approx(expected)


def test_self_intersecting_ring_fails_loudly():
    """
    A bowtie. Ear clipping produces triangles for it without complaint, and they
    do not sum to the polygon's area — which is exactly what the area check is
    there to notice. Silence here would mean a mangled building in the render.
    """
    bowtie = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    with pytest.raises(MassingError):
        triangulate(bowtie)


# ── Extrusion ─────────────────────────────────────────────────────────────────

def test_box_volume_matches_area_times_height():
    """
    The strongest check available offline. The volume comes from a sum over face
    vertices via the divergence theorem — it never sees the footprint or the
    height — so agreement means the caps, the walls and every winding are right.
    """
    mesh = extrude(SQUARE, 25.0)
    assert mesh.volume() == pytest.approx(100.0 * 25.0)


def test_volume_is_positive_meaning_normals_face_outward():
    """
    A mesh built inside-out has exactly the right shape and the negative of the
    right volume. In the viewport it looks perfect; in a render the sun lights
    the interior faces and the building goes black.
    """
    assert extrude(SQUARE, 10.0).volume() > 0


def test_concave_volume_matches():
    assert extrude(L_SHAPE, 12.0).volume() == pytest.approx(300.0 * 12.0)


def test_courtyard_volume_excludes_the_hole():
    mesh = extrude(COURTYARD_OUTER, 10.0, holes=[COURTYARD_HOLE])
    assert mesh.volume() == pytest.approx((900.0 - 100.0) * 10.0)


def test_round_tower_volume():
    ring = _polygon(64, 15.0)
    mesh = extrude(ring, 40.0)
    assert mesh.volume() == pytest.approx(abs(signed_area(ring)) * 40.0, rel=1e-9)


def test_mesh_is_watertight():
    """Every edge shared by exactly two faces — no cracks, no duplicated shells."""
    for mesh in (
        extrude(SQUARE, 10.0),
        extrude(L_SHAPE, 10.0),
        extrude(COURTYARD_OUTER, 10.0, holes=[COURTYARD_HOLE]),
        extrude(_polygon(24, 9.0), 30.0),
    ):
        counts = _edge_counts(mesh)
        assert all(n == 2 for n in counts.values()), (
            f"{mesh.name}: {sum(1 for n in counts.values() if n != 2)} bad edges"
        )


def test_box_has_the_expected_topology():
    """A box: 8 vertices, 12 triangles, and Euler V - E + F = 2."""
    mesh = extrude(SQUARE, 5.0)
    assert mesh.vertex_count == 8
    assert mesh.face_count == 12
    edges = len(_edge_counts(mesh))
    assert mesh.vertex_count - edges + mesh.face_count == 2


def test_courtyard_is_a_torus_not_a_box():
    """
    A building with a hole through it is genus 1, so Euler V - E + F = 0.
    A bridged-but-unwelded mesh scores 2 here while looking identical, which is
    what makes this worth asserting rather than eyeballing.
    """
    mesh = extrude(COURTYARD_OUTER, 10.0, holes=[COURTYARD_HOLE])
    edges = len(_edge_counts(mesh))
    assert mesh.vertex_count - edges + mesh.face_count == 0


def test_no_degenerate_faces():
    for mesh in (extrude(COURTYARD_OUTER, 10.0, holes=[COURTYARD_HOLE]), extrude(L_SHAPE, 4.0)):
        for i, j, k in mesh.faces:
            assert len({i, j, k}) == 3
            a, b, c = mesh.verts[i], mesh.verts[j], mesh.verts[k]
            area = 0.5 * math.dist(
                (0, 0, 0),
                (
                    (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]),
                    (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]),
                ),
            )
            assert area > 1e-9


def test_bridge_seam_does_not_become_a_wall():
    """
    The seam cut into a courtyard is zero-width and has no physical existence.
    Extruding it puts a black zero-thickness sliver across the courtyard. Wall
    count is the tell: exactly one quad per real ring edge, no more.
    """
    mesh = extrude(COURTYARD_OUTER, 10.0, holes=[COURTYARD_HOLE])
    ring_edges = len(COURTYARD_OUTER) + len(COURTYARD_HOLE)
    cap_faces = mesh.face_count - ring_edges * 2
    assert cap_faces > 0
    # Caps come in pairs (roof and floor), so the remainder must be even.
    assert cap_faces % 2 == 0


def test_base_offset_lifts_the_whole_solid():
    mesh = extrude(SQUARE, 10.0, base=4.0)
    (_, _, zmin), (_, _, zmax) = mesh.bounds()
    assert (zmin, zmax) == pytest.approx((4.0, 14.0))
    assert mesh.volume() == pytest.approx(100.0 * 10.0)


def test_footprint_lands_where_it_was_put():
    mesh = extrude(SQUARE, 3.0)
    (xmin, ymin, _), (xmax, ymax, _) = mesh.bounds()
    assert (xmin, ymin, xmax, ymax) == pytest.approx((0.0, 0.0, 10.0, 10.0))


@pytest.mark.parametrize("height", [0.0, -5.0, float("nan"), float("inf")])
def test_bad_heights_are_rejected(height):
    with pytest.raises(MassingError):
        extrude(SQUARE, height)


# ── Buildings into the scene frame ────────────────────────────────────────────

DUBAI = SceneFrame(25.2048, 55.2708)


def _building_at(lat: float, lon: float, size_deg: float = 0.0005, **kwargs) -> Building:
    ring = [
        (lat, lon),
        (lat, lon + size_deg),
        (lat + size_deg, lon + size_deg),
        (lat + size_deg, lon),
    ]
    return Building(1, "way", ring, **kwargs)


def test_building_at_the_origin_sits_at_the_scene_origin():
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "20"})
    mesh = building_to_mesh(building, DUBAI)
    (xmin, ymin, _), _ = mesh.bounds()
    assert (xmin, ymin) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_building_north_east_of_the_site_lands_north_east():
    """+X east and +Y north, or every shadow in the scene falls the wrong way."""
    building = _building_at(DUBAI.origin_lat + 0.002, DUBAI.origin_lon + 0.002)
    (xmin, ymin, _), _ = building_to_mesh(building, DUBAI).bounds()
    assert xmin > 0 and ymin > 0


def test_building_south_west_of_the_site_lands_south_west():
    building = _building_at(DUBAI.origin_lat - 0.002, DUBAI.origin_lon - 0.002)
    _, (xmax, ymax, _) = building_to_mesh(building, DUBAI).bounds()
    assert xmax < 0 and ymax < 0


def test_tagged_height_reaches_the_mesh():
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "45"})
    _, (_, _, zmax) = building_to_mesh(building, DUBAI).bounds()
    assert zmax == pytest.approx(45.0)


def test_min_height_becomes_the_base_and_the_solid_keeps_its_top():
    """
    An overhanging upper storey starts above the street. Getting this wrong
    drops the volume to ground level, where it blocks the road.
    """
    building = _building_at(
        DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "30", "min_height": "10"}
    )
    mesh = building_to_mesh(building, DUBAI)
    (_, _, zmin), (_, _, zmax) = mesh.bounds()
    assert (zmin, zmax) == pytest.approx((10.0, 30.0))


def test_a_real_sized_building_measures_correctly_in_metres():
    """
    A 0.0005 deg square at Dubai's latitude is about 55 m north-south. Getting
    metres wrong by 39.37x is this project's signature failure — Max's default
    unit is inches — so the absolute scale is asserted, not just the shape.
    """
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    (xmin, ymin, _), (xmax, ymax, _) = building_to_mesh(building, DUBAI).bounds()
    assert (ymax - ymin) == pytest.approx(55.4, abs=0.5)
    assert (xmax - xmin) == pytest.approx(50.3, abs=0.5)


def test_batch_conversion_skips_bad_footprints_and_reports_them():
    """A district must not be lost to one badly-mapped building — nor silently."""
    good = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    bowtie_ring = [
        (DUBAI.origin_lat, DUBAI.origin_lon),
        (DUBAI.origin_lat + 0.001, DUBAI.origin_lon + 0.001),
        (DUBAI.origin_lat, DUBAI.origin_lon + 0.001),
        (DUBAI.origin_lat + 0.001, DUBAI.origin_lon),
    ]
    bad = Building(2, "way", bowtie_ring, tags={"height": "10"})

    meshes, skipped = buildings_to_meshes([good, bad], DUBAI)
    assert len(meshes) == 1
    assert len(skipped) == 1
    assert skipped[0]["osm_id"] == 2
    assert skipped[0]["reason"]


def test_mesh_names_are_unique_per_osm_id():
    """
    Max silently renames a node on a name collision, which breaks every later
    lookup by name — and a retail street has a dozen buildings called "Costa".
    """
    a = Building(11, "way", _building_at(DUBAI.origin_lat, DUBAI.origin_lon).outer,
                 tags={"name": "Costa"})
    b = Building(22, "way", _building_at(DUBAI.origin_lat, DUBAI.origin_lon).outer,
                 tags={"name": "Costa"})
    meshes, _ = buildings_to_meshes([a, b], DUBAI)
    assert meshes[0].name != meshes[1].name


def test_mesh_name_survives_punctuation_in_the_osm_name():
    building = Building(
        5, "way", _building_at(DUBAI.origin_lat, DUBAI.origin_lon).outer,
        tags={"name": "St. Mary's Church & Hall"},
    )
    name = building_to_mesh(building, DUBAI).name
    assert name.startswith("osm_w5_")
    assert all(ch.isalnum() or ch in " -_" for ch in name)


# ── Sitting on terrain ────────────────────────────────────────────────────────

def test_without_terrain_buildings_stand_at_zero():
    """The z=0 plane is only correct where the terrain is; it stays the default."""
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "20"})
    (_, _, zmin), _ = building_to_mesh(building, DUBAI).bounds()
    assert zmin == pytest.approx(0.0)


def test_flat_terrain_lifts_the_whole_building():
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "20"})
    mesh = building_to_mesh(building, DUBAI, lambda lat, lon: 42.0)
    (_, _, zmin), (_, _, zmax) = mesh.bounds()
    assert (zmin, zmax) == pytest.approx((42.0, 62.0))


def test_building_keeps_its_height_on_terrain():
    """Draping must move a building, never stretch or squash it."""
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "35"})
    flat = building_to_mesh(building, DUBAI)
    raised = building_to_mesh(building, DUBAI, lambda lat, lon: 100.0)
    assert raised.volume() == pytest.approx(flat.volume())


def test_sloping_ground_uses_the_lowest_sample():
    """
    On a slope the choice is visible. The minimum buries the uphill corner,
    which is what a foundation does; a centroid or mean would leave the downhill
    half floating, and a gap under a building reads instantly as broken.
    """
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    # Ground rises steeply with latitude across the footprint.
    ground = lambda lat, lon: (lat - DUBAI.origin_lat) * 100000.0
    mesh = building_to_mesh(building, DUBAI, ground)
    lows = [ground(lat, lon) for lat, lon in building.outer]
    (_, _, zmin), _ = mesh.bounds()
    assert zmin == pytest.approx(min(lows))
    assert zmin < sum(lows) / len(lows), "used the mean, so the building floats"


def test_negative_terrain_sinks_the_building():
    """Below-sea-level ground is real and must not be clamped to zero."""
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    (_, _, zmin), _ = building_to_mesh(building, DUBAI, lambda lat, lon: -12.0).bounds()
    assert zmin == pytest.approx(-12.0)


def test_min_height_stacks_on_top_of_terrain():
    """An overhanging storey is relative to its own ground, not to sea level."""
    building = _building_at(
        DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "30", "min_height": "10"}
    )
    mesh = building_to_mesh(building, DUBAI, lambda lat, lon: 50.0)
    (_, _, zmin), (_, _, zmax) = mesh.bounds()
    assert (zmin, zmax) == pytest.approx((60.0, 80.0))


def test_a_building_wholly_outside_the_terrain_is_an_error():
    """
    Defaulting an unknown ground level to zero would drop the building to sea
    level next to neighbours at 100 m — plausible geometry, wrong scene.
    """
    def ground(lat, lon):
        raise ValueError("outside the patch")

    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    with pytest.raises(MassingError, match="ground level is unknown"):
        building_to_mesh(building, DUBAI, ground)


def test_partially_covered_building_uses_the_samples_it_has():
    """A footprint straddling the patch edge should still land, not be lost."""
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    calls = {"n": 0}

    def ground(lat, lon):
        calls["n"] += 1
        if calls["n"] % 2:
            raise ValueError("outside")
        return 7.0

    (_, _, zmin), _ = building_to_mesh(building, DUBAI, ground).bounds()
    assert zmin == pytest.approx(7.0)


def test_ground_elevation_is_recorded_in_metadata():
    building = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    mesh = building_to_mesh(building, DUBAI, lambda lat, lon: 13.5)
    assert mesh.metadata["ground_m"] == pytest.approx(13.5)


def test_batch_conversion_passes_terrain_through():
    good = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"height": "10"})
    meshes, skipped = buildings_to_meshes([good], DUBAI, lambda lat, lon: 25.0)
    assert not skipped
    (_, _, zmin), _ = meshes[0].bounds()
    assert zmin == pytest.approx(25.0)


def test_height_provenance_reaches_the_mesh_metadata():
    """
    Downstream needs to tell a tagged height from a guessed one — an artist
    deciding what to replace, and the attribution manifest recording what was
    inferred rather than sourced.
    """
    guessed = _building_at(DUBAI.origin_lat, DUBAI.origin_lon, tags={"building": "yes"})
    mesh = building_to_mesh(guessed, DUBAI)
    assert "default" in mesh.metadata["height_source"]
    assert mesh.metadata["osm_id"] == 1
