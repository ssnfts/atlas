"""
Massing cross-validated against Shapely.

``test_massing.py`` checks the geometry against invariants it derives itself.
That is necessary and not sufficient — a triangulator and a test written by the
same hand can share an assumption and agree with each other perfectly while
both being wrong. The same reasoning is why ``solar.py`` is checked against
pvlib's NREL SPA rather than only against its own definitional identities.

So these compare against Shapely: GEOS underneath, a completely separate
implementation of planar polygon geometry, already a project dependency. Where
its area and containment answers agree with the ear-clipper's, that agreement is
evidence.

Shapely is used here as a *reference*, not as the implementation — see the note
in ``massing.py`` on why its unconstrained Delaunay triangulation cannot build
these roofs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

shapely = pytest.importorskip("shapely", reason="cross-validation reference")

from shapely.geometry import Point, Polygon  # noqa: E402

from massing import extrude, signed_area, triangulate  # noqa: E402


def _polygon(sides: int, radius: float, cx: float = 0.0, cy: float = 0.0):
    return [
        (
            cx + radius * math.cos(2 * math.pi * i / sides),
            cy + radius * math.sin(2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]


# A deliberately awkward set: concave notches, a spiral-ish comb, near-collinear
# runs and a very thin wing — the shapes that break naive ear clipping.
FOOTPRINTS = {
    "square": [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    "L": [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0), (10.0, 20.0), (0.0, 20.0)],
    "U": [
        (0.0, 0.0), (30.0, 0.0), (30.0, 20.0), (20.0, 20.0),
        (20.0, 8.0), (10.0, 8.0), (10.0, 20.0), (0.0, 20.0),
    ],
    "T": [
        (0.0, 0.0), (30.0, 0.0), (30.0, 8.0), (19.0, 8.0),
        (19.0, 25.0), (11.0, 25.0), (11.0, 8.0), (0.0, 8.0),
    ],
    "comb": [
        (0.0, 0.0), (40.0, 0.0), (40.0, 12.0), (34.0, 12.0), (34.0, 4.0),
        (26.0, 4.0), (26.0, 12.0), (18.0, 12.0), (18.0, 4.0),
        (10.0, 4.0), (10.0, 12.0), (0.0, 12.0),
    ],
    "thin_wing": [
        (0.0, 0.0), (25.0, 0.0), (25.0, 0.4), (5.0, 0.4), (5.0, 15.0), (0.0, 15.0)
    ],
    "near_collinear": [
        (0.0, 0.0), (10.0, 0.0), (20.0, 0.001), (30.0, 0.0),
        (30.0, 10.0), (0.0, 10.0),
    ],
    "octagon": _polygon(8, 12.0),
    "circle_64": _polygon(64, 18.0),
    "chevron": [(0.0, 0.0), (10.0, 12.0), (20.0, 0.0), (20.0, 6.0), (10.0, 18.0), (0.0, 6.0)],
}

HOLED = {
    "courtyard": (
        [(0.0, 0.0), (30.0, 0.0), (30.0, 30.0), (0.0, 30.0)],
        [[(10.0, 10.0), (20.0, 10.0), (20.0, 20.0), (10.0, 20.0)]],
    ),
    "two_courtyards": (
        [(0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)],
        [
            [(5.0, 5.0), (12.0, 5.0), (12.0, 15.0), (5.0, 15.0)],
            [(22.0, 6.0), (33.0, 6.0), (33.0, 14.0), (22.0, 14.0)],
        ],
    ),
    "round_atrium": (_polygon(32, 25.0), [_polygon(20, 8.0)]),
    "offcentre_atrium": (_polygon(24, 30.0), [_polygon(12, 5.0, cx=14.0, cy=6.0)]),
    "L_with_courtyard": (
        [(0.0, 0.0), (30.0, 0.0), (30.0, 15.0), (15.0, 15.0), (15.0, 30.0), (0.0, 30.0)],
        [[(4.0, 4.0), (11.0, 4.0), (11.0, 11.0), (4.0, 11.0)]],
    ),
}


def _triangulated_area(verts, faces) -> float:
    total = 0.0
    for i, j, k in faces:
        (ax, ay), (bx, by), (cx, cy) = verts[i], verts[j], verts[k]
        total += abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5
    return total


# ── Area agreement ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_triangulated_area_matches_shapely(name):
    ring = FOOTPRINTS[name]
    verts, faces = triangulate(ring)
    assert _triangulated_area(verts, faces) == pytest.approx(Polygon(ring).area, rel=1e-9)


@pytest.mark.parametrize("name", sorted(HOLED))
def test_holed_area_matches_shapely(name):
    outer, holes = HOLED[name]
    verts, faces = triangulate(outer, holes)
    assert _triangulated_area(verts, faces) == pytest.approx(
        Polygon(outer, holes).area, rel=1e-9
    )


@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_signed_area_magnitude_matches_shapely(name):
    ring = FOOTPRINTS[name]
    assert abs(signed_area(ring)) == pytest.approx(Polygon(ring).area, rel=1e-12)


@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_winding_sign_matches_shapely_orientation(name):
    """
    Our CCW-positive convention must be the same handedness Shapely reports, or
    'outward' means opposite things in the two libraries.
    """
    from shapely.geometry.polygon import orient as shapely_orient

    ring = FOOTPRINTS[name]
    ccw = shapely_orient(Polygon(ring), sign=1.0)
    assert signed_area(list(ccw.exterior.coords)[:-1]) > 0


# ── Extruded volume agreement ─────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_volume_matches_shapely_area_times_height(name):
    """
    The divergence-theorem volume of the closed mesh against an independently
    computed footprint area. Nothing in the mesh construction consults Shapely,
    so this catches a bad cap, a flipped wall or an inverted normal at once.
    """
    ring = FOOTPRINTS[name]
    assert extrude(ring, 13.5).volume() == pytest.approx(Polygon(ring).area * 13.5, rel=1e-9)


@pytest.mark.parametrize("name", sorted(HOLED))
def test_holed_volume_matches_shapely(name):
    outer, holes = HOLED[name]
    expected = Polygon(outer, holes).area * 9.0
    assert extrude(outer, 9.0, holes=holes).volume() == pytest.approx(expected, rel=1e-9)


# ── Triangle-level agreement ──────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_every_triangle_lies_inside_the_footprint(name):
    """
    Area agreement alone is not conclusive: a triangulation could spill outside
    the polygon and lose the same area elsewhere. Shapely checks each triangle's
    centroid for containment, which that cancellation cannot survive.
    """
    ring = FOOTPRINTS[name]
    polygon = Polygon(ring)
    verts, faces = triangulate(ring)
    for i, j, k in faces:
        a, b, c = verts[i], verts[j], verts[k]
        centroid = Point((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
        assert polygon.covers(centroid), f"{name}: triangle {(i, j, k)} escapes the footprint"


@pytest.mark.parametrize("name", sorted(HOLED))
def test_no_triangle_covers_a_courtyard(name):
    """The failure that fills in an atrium and leaves the total area plausible."""
    outer, holes = HOLED[name]
    polygon = Polygon(outer, holes)
    verts, faces = triangulate(outer, holes)
    for i, j, k in faces:
        a, b, c = verts[i], verts[j], verts[k]
        centroid = Point((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
        assert polygon.covers(centroid), f"{name}: triangle {(i, j, k)} is over a hole"


@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_triangles_do_not_overlap_each_other(name):
    """
    Overlap is how the L-shape bug presented: an ear clipped across the notch
    double-covered part of the footprint. Pairwise intersection area must be
    zero — triangles may share edges, never interiors.
    """
    ring = FOOTPRINTS[name]
    verts, faces = triangulate(ring)
    triangles = [Polygon([verts[i], verts[j], verts[k]]) for i, j, k in faces]
    for index, first in enumerate(triangles):
        for second in triangles[index + 1:]:
            assert first.intersection(second).area == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", sorted(HOLED))
def test_holed_triangles_do_not_overlap_each_other(name):
    outer, holes = HOLED[name]
    verts, faces = triangulate(outer, holes)
    triangles = [Polygon([verts[i], verts[j], verts[k]]) for i, j, k in faces]
    for index, first in enumerate(triangles):
        for second in triangles[index + 1:]:
            assert first.intersection(second).area == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", sorted(FOOTPRINTS))
def test_union_of_triangles_reconstructs_the_footprint(name):
    """
    The strongest statement available: reassembling every triangle must give
    back the original polygon, to within floating-point noise. Area, coverage
    and non-overlap all follow from this, and it additionally rules out a hole
    left *between* triangles that the other checks would miss.
    """
    from shapely.ops import unary_union

    ring = FOOTPRINTS[name]
    polygon = Polygon(ring)
    verts, faces = triangulate(ring)
    union = unary_union([Polygon([verts[i], verts[j], verts[k]]) for i, j, k in faces])
    assert union.symmetric_difference(polygon).area == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", sorted(HOLED))
def test_union_reconstructs_a_holed_footprint(name):
    from shapely.ops import unary_union

    outer, holes = HOLED[name]
    polygon = Polygon(outer, holes)
    verts, faces = triangulate(outer, holes)
    union = unary_union([Polygon([verts[i], verts[j], verts[k]]) for i, j, k in faces])
    assert union.symmetric_difference(polygon).area == pytest.approx(0.0, abs=1e-9)


# ── Randomised stress ─────────────────────────────────────────────────────────
#
# Both bugs found while writing this module survived every hand-written case and
# died to a shape nobody thought to draw: a reflex corner sitting exactly on an
# ear's edge, and two courtyards bridging to the same outer corner. Hand-picked
# fixtures test the cases the author already imagined. These do not.
#
# The seed is fixed so a failure is reproducible; the generator is star-shaped
# about a centre, which guarantees a simple ring, and Shapely vets every
# generated case before it is asserted on.

def _star_polygon(rng, sides, min_r, max_r, cx=0.0, cy=0.0):
    """A star-shaped ring: sorted angles with random radii cannot self-intersect."""
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(sides))
    ring = []
    for angle in angles:
        radius = rng.uniform(min_r, max_r)
        ring.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return ring


def _random_cases(count, *, holes_per_case, seed):
    import random

    rng = random.Random(seed)
    cases = []
    attempts = 0
    while len(cases) < count and attempts < count * 60:
        attempts += 1
        outer = _star_polygon(rng, rng.randint(3, 24), 20.0, 40.0)
        holes = []
        for _ in range(holes_per_case):
            angle = rng.uniform(0, 2 * math.pi)
            distance = rng.uniform(0.0, 12.0)
            holes.append(
                _star_polygon(
                    rng,
                    rng.randint(3, 10),
                    1.5,
                    4.0,
                    cx=distance * math.cos(angle),
                    cy=distance * math.sin(angle),
                )
            )
        try:
            candidate = Polygon(outer, holes)
        except Exception:
            continue
        # Only assert on inputs that are genuinely well-formed; a generator that
        # emits an overlapping hole is testing the generator, not the clipper.
        if not candidate.is_valid or candidate.area <= 1.0:
            continue
        if any(Polygon(h).area <= 0.5 for h in holes):
            continue
        cases.append((outer, holes, candidate))
    return cases


@pytest.mark.parametrize("index", range(60))
def test_random_simple_footprint_reconstructs(index):
    from shapely.ops import unary_union

    outer, holes, reference = _random_cases(60, holes_per_case=0, seed=20260722)[index]
    verts, faces = triangulate(outer, holes)
    union = unary_union([Polygon([verts[i], verts[j], verts[k]]) for i, j, k in faces])
    assert union.symmetric_difference(reference).area == pytest.approx(
        0.0, abs=1e-6 * reference.area
    )
    assert len(faces) == len(outer) - 2


@pytest.mark.parametrize("holes_per_case", [1, 2, 3])
def test_random_holed_footprints_reconstruct(holes_per_case):
    """
    Multiple courtyards is where the anchor-ordering bug lived. One hole never
    triggered it; two, placed so both could see the same outer corner, did.
    """
    from shapely.ops import unary_union

    cases = _random_cases(40, holes_per_case=holes_per_case, seed=770 + holes_per_case)
    assert cases, "generator produced no valid cases"

    for outer, holes, reference in cases:
        verts, faces = triangulate(outer, holes)
        union = unary_union([Polygon([verts[i], verts[j], verts[k]]) for i, j, k in faces])
        assert union.symmetric_difference(reference).area == pytest.approx(
            0.0, abs=1e-6 * reference.area
        ), f"reconstruction failed for a {len(outer)}-gon with {len(holes)} holes"


@pytest.mark.parametrize("holes_per_case", [0, 1, 2, 3])
def test_random_extrusions_stay_watertight_and_correctly_wound(holes_per_case):
    """
    Volume sign and edge parity over random input. An inverted normal or a
    cracked seam is invisible in the viewport and ruins a render.
    """
    from collections import Counter

    for outer, holes, reference in _random_cases(
        30, holes_per_case=holes_per_case, seed=991 + holes_per_case
    ):
        mesh = extrude(outer, 7.5, holes=holes)
        assert mesh.volume() == pytest.approx(reference.area * 7.5, rel=1e-6)

        edges: Counter = Counter()
        for i, j, k in mesh.faces:
            for a, b in ((i, j), (j, k), (k, i)):
                edges[(min(a, b), max(a, b))] += 1
        assert all(n == 2 for n in edges.values()), "mesh has a crack or a duplicate face"
