"""
Roadway ribbon tests.

Offsetting a polyline into a strip fails the same silent way massing does: a
ribbon at the wrong width, a strip wound so its faces point at the ground, a
spike stabbing across a hairpin — none of them raise, all of them render. So
these check conservation laws a self-consistent bug cannot satisfy:

- the ribbon's centre-line is exactly as long as the polyline it came from;
- the surface area agrees with ``shapely``'s independently-computed buffer;
- every surviving triangle's normal points +Z, so nothing is inside-out;
- a straight run is exactly ``length * width`` and exactly ``width`` wide.

The ``shapely`` cross-check is the load-bearing one, for the same reason the
volume check is in massing: it arrives at the area by a completely different
route (a polygon buffer, not a sum of the strip's own quads), so agreement is
evidence and not a restatement of the construction.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import roadway  # noqa: E402
from frame import SceneFrame  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def _path(xy, closed=False, tags=None, ids=None):
    return roadway.Path(xy=list(xy), closed=closed, tags=tags or {}, osm_ids=ids or [])


def _all_faces_up(mesh) -> int:
    """Number of faces whose normal does not point strictly +Z."""
    bad = 0
    for i, j, k in mesh.faces:
        ax, ay, _ = mesh.verts[i]
        bx, by, _ = mesh.verts[j]
        cx, cy, _ = mesh.verts[k]
        if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) <= 0:
            bad += 1
    return bad


def _mesh_triangle_area(mesh) -> float:
    total = 0.0
    for i, j, k in mesh.faces:
        ax, ay, _ = mesh.verts[i]
        bx, by, _ = mesh.verts[j]
        cx, cy, _ = mesh.verts[k]
        total += abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2.0
    return total


# ── straight run: the exact case ──────────────────────────────────────────────

def test_straight_ribbon_is_length_times_width():
    p = _path([(0.0, 0.0), (100.0, 0.0)])
    mesh = roadway.ribbon_mesh(p, width=10.0)
    assert mesh.metadata["centerline_length_m"] == pytest.approx(100.0)
    # A straight strip has no corners, so area is exact.
    assert _mesh_triangle_area(mesh) == pytest.approx(1000.0, rel=1e-9)
    assert mesh.metadata["faces_dropped_at_folds"] == 0
    assert _all_faces_up(mesh) == 0


def test_straight_ribbon_is_exactly_width_wide():
    p = _path([(0.0, 0.0), (50.0, 0.0)])
    left, right, _ = roadway.offset_ribbon(p.xy, 8.0)
    for (lx, ly), (rx, ry) in zip(left, right):
        assert math.hypot(lx - rx, ly - ry) == pytest.approx(8.0)


def test_left_is_counter_clockwise_of_travel():
    # Travelling +X (east), the left edge must be to +Y (north).
    left, right, _ = roadway.offset_ribbon([(0.0, 0.0), (10.0, 0.0)], 4.0)
    assert left[0][1] > 0 and right[0][1] < 0


def test_endpoints_of_open_path_are_square():
    left, right, _ = roadway.offset_ribbon([(0.0, 0.0), (10.0, 0.0)], 4.0)
    # Square end cap: the two end vertices share the start's x.
    assert left[0][0] == pytest.approx(0.0)
    assert right[0][0] == pytest.approx(0.0)


# ── corners: length conserved, width kept, nothing inside-out ──────────────────

def test_right_angle_corner_conserves_length():
    p = _path([(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)])
    mesh = roadway.ribbon_mesh(p, width=6.0)
    assert mesh.metadata["centerline_length_m"] == pytest.approx(100.0)
    assert _all_faces_up(mesh) == 0


def test_mitre_widens_the_outer_corner():
    # At a 90 deg bend the mitred cross-section is width / cos(45) = width * sqrt2.
    left, right, _ = roadway.offset_ribbon(
        [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)], 6.0
    )
    # middle cross-section is index 1
    lx, ly = left[1]
    rx, ry = right[1]
    span = math.hypot(lx - rx, ly - ry)
    assert span == pytest.approx(6.0 * math.sqrt(2.0), rel=1e-6)


def test_hairpin_triggers_bevel_without_inverted_faces():
    # A near-180 deg switchback over a short span: the mitre would spike, so the
    # bevel path must fire and no face may end up inside-out.
    p = _path([(0.0, 0.0), (20.0, 0.0), (20.0, 1.0), (0.0, 1.0)])
    mesh = roadway.ribbon_mesh(p, width=8.0)
    assert _all_faces_up(mesh) == 0
    assert mesh.metadata["centerline_length_m"] == pytest.approx(20.0 + 1.0 + 20.0)


# ── closed loop: seamless wrap, no caps ────────────────────────────────────────

def test_closed_loop_has_no_end_caps_and_wraps():
    # A square loop, given closed (last vertex repeats the first).
    sq = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]
    p = _path(sq, closed=True)
    left, right, _ = roadway.offset_ribbon(p.xy, 10.0, closed=True)
    # Duplicate closing vertex dropped: four distinct cross-sections, not five.
    assert len(left) == 4 and len(right) == 4
    mesh = roadway.ribbon_mesh(p, width=10.0)
    assert mesh.metadata["closed"] is True
    assert _all_faces_up(mesh) == 0
    # A 100 m square loop has a 400 m spine.
    assert mesh.metadata["centerline_length_m"] == pytest.approx(400.0)


def test_closed_square_loop_area_matches_annulus():
    # Outer offset square is 120x120, inner is 80x80; ribbon area = 120^2 - 80^2.
    sq = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0), (0.0, 0.0)]
    p = _path(sq, closed=True)
    mesh = roadway.ribbon_mesh(p, width=20.0)
    expected = 120.0 * 120.0 - 80.0 * 80.0
    assert _mesh_triangle_area(mesh) == pytest.approx(expected, rel=1e-6)


# ── ground draping ─────────────────────────────────────────────────────────────

def test_ground_callable_lifts_the_ribbon():
    frame = SceneFrame(24.47, 54.60)
    p = _path([(0.0, 0.0), (10.0, 0.0)])
    mesh = roadway.ribbon_mesh(p, width=4.0, frame=frame,
                               ground=lambda lat, lon: 12.0, lift=0.05)
    assert all(v[2] == pytest.approx(12.05) for v in mesh.verts)


def test_ground_probe_is_called_with_geodetic_not_scene_metres():
    """
    The bug this locks: the probe takes (lat, lon), the ribbon is in metres.

    Calling it with scene metres made every sample raise, a blanket except
    swallowed it, and a 5 km circuit came back flat at `lift` — buried under
    terrain that ranged over 41 m, with nothing raised anywhere. So assert the
    probe actually receives coordinates near the frame origin.
    """
    frame = SceneFrame(24.47, 54.60)
    seen: list[tuple[float, float]] = []

    def probe(lat, lon):
        seen.append((lat, lon))
        return 5.0

    p = _path([(0.0, 0.0), (100.0, 0.0)])
    roadway.ribbon_mesh(p, width=4.0, frame=frame, ground=probe)

    assert seen, "ground probe was never called"
    for lat, lon in seen:
        assert abs(lat - 24.47) < 0.01, f"latitude {lat} is not geodetic"
        assert abs(lon - 54.60) < 0.01, f"longitude {lon} is not geodetic"


def test_ground_without_frame_is_refused():
    p = _path([(0.0, 0.0), (10.0, 0.0)])
    with pytest.raises(roadway.RoadwayError, match="frame"):
        roadway.ribbon_mesh(p, width=4.0, ground=lambda lat, lon: 1.0)


def test_ribbon_entirely_outside_the_patch_raises():
    frame = SceneFrame(24.47, 54.60)

    def outside(lat, lon):
        raise ValueError("outside patch")

    p = _path([(0.0, 0.0), (10.0, 0.0)])
    with pytest.raises(roadway.RoadwayError, match="terrain patch"):
        roadway.ribbon_mesh(p, width=4.0, frame=frame, ground=outside)


def test_partial_ground_coverage_survives_and_is_counted():
    """One vertex off the tile should cost that vertex, not the whole lap."""
    frame = SceneFrame(24.47, 54.60)

    def patchy(lat, lon):
        if lon > 54.6005:
            raise ValueError("outside patch")
        return 7.0

    p = _path([(0.0, 0.0), (200.0, 0.0)])
    mesh = roadway.ribbon_mesh(p, width=4.0, frame=frame, ground=patchy, lift=0.1)
    assert mesh.metadata["ground_samples_missed"] > 0
    assert all(v[2] == pytest.approx(7.1) for v in mesh.verts)


# ── grading: a road follows the trend, not the noise ──────────────────────────

def test_smoothing_flattens_dem_noise_but_keeps_real_relief():
    """
    A sawtooth of DEM noise on top of a genuine long hill.

    Smoothing over a window much longer than the noise and much shorter than the
    hill must remove the first and keep the second — that is the whole claim.
    """
    frame = SceneFrame(24.47, 54.60)
    xy = [(t, 0.0) for t in range(0, 1001, 10)]
    p = _path(xy)

    def noisy(lat, lon):
        x, _, _ = frame.to_scene(lat, lon)
        hill = x * 0.02                      # a real 2% grade over 1 km
        noise = 1.5 if int(round(x / 10)) % 2 else -1.5   # +/-1.5 m sawtooth
        return hill + noise

    rough = roadway.ribbon_mesh(p, width=6.0, frame=frame, ground=noisy)
    smooth = roadway.ribbon_mesh(p, width=6.0, frame=frame, ground=noisy,
                                 smooth_m=120.0)

    assert rough.metadata["max_gradient_pct"] > 25.0     # noise dominates
    assert smooth.metadata["max_gradient_pct"] < 5.0     # noise gone

    # The real hill survives: ~20 m of rise across 1 km.
    zs = [v[2] for v in smooth.verts]
    assert max(zs) - min(zs) == pytest.approx(20.0, rel=0.15)


def test_window_wider_than_the_route_levels_it():
    frame = SceneFrame(24.47, 54.60)
    xy = [(t, 0.0) for t in range(0, 501, 25)]
    p = _path(xy)

    def sloped(lat, lon):
        x, _, _ = frame.to_scene(lat, lon)
        return 4.0 + x * 0.01

    mesh = roadway.ribbon_mesh(p, width=6.0, frame=frame, ground=sloped,
                               smooth_m=10_000.0, lift=0.0)
    zs = [v[2] for v in mesh.verts]
    assert max(zs) - min(zs) == pytest.approx(0.0, abs=1e-9)
    assert mesh.metadata["max_gradient_pct"] == pytest.approx(0.0, abs=1e-9)
    # Levelled to the mean of the sampled profile, not to zero.
    assert zs[0] == pytest.approx(4.0 + 0.01 * 250.0, rel=1e-6)


def test_both_edges_share_one_elevation_per_cross_section():
    """No random cross-track banking: left and right must sit at equal height."""
    frame = SceneFrame(24.47, 54.60)
    xy = [(t, 0.0) for t in range(0, 201, 10)]
    p = _path(xy)

    def bumpy(lat, lon):
        _, y, _ = frame.to_scene(lat, lon)
        return 3.0 + y  # varies purely across the ribbon's width

    mesh = roadway.ribbon_mesh(p, width=20.0, frame=frame, ground=bumpy)
    m = len(mesh.verts) // 2
    for k in range(m):
        assert mesh.verts[k][2] == pytest.approx(mesh.verts[m + k][2])


def test_owners_map_cross_sections_to_spine_vertices():
    left, right, owners = roadway.offset_ribbon(
        [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)], 6.0
    )
    assert len(owners) == len(left) == len(right)
    assert owners == sorted(owners)          # emitted in spine order
    assert set(owners) == {0, 1, 2}


# ── parsing & stitching ────────────────────────────────────────────────────────

def _payload(*ways):
    return {"elements": [{"type": "way", "id": wid, "tags": tags,
                          "geometry": [{"lat": la, "lon": lo} for la, lo in geom]}
                         for wid, tags, geom in ways]}


def test_parse_ways_keeps_and_filters():
    payload = _payload(
        (1, {"highway": "raceway", "name": "A"}, [(0, 0), (0, 1)]),
        (2, {"highway": "raceway", "sport": "karting"}, [(1, 0), (1, 1)]),
    )
    both = roadway.parse_ways(payload)
    assert len(both) == 2
    only_a = roadway.parse_ways(payload, keep=lambda t: t.get("name") == "A")
    assert len(only_a) == 1 and only_a[0].osm_id == 1


def test_parse_drops_ways_with_under_two_points():
    payload = _payload((1, {"highway": "raceway"}, [(0, 0)]))
    assert roadway.parse_ways(payload) == []


def test_stitch_joins_ways_sharing_an_endpoint():
    frame = SceneFrame(24.47, 54.60)
    # Two ways meeting at (24.47, 54.60); second is digitised reversed.
    a = roadway.Centerline(1, "way", [(24.470, 54.600), (24.471, 54.601)], {})
    b = roadway.Centerline(2, "way", [(24.472, 54.602), (24.471, 54.601)], {})
    paths = roadway.stitch_paths([a, b], frame)
    assert len(paths) == 1
    assert len(paths[0].xy) == 3  # shared node not duplicated
    assert set(paths[0].osm_ids) == {1, 2}


def test_stitch_recognises_a_closed_loop():
    frame = SceneFrame(24.47, 54.60)
    loop = roadway.Centerline(
        1, "way",
        [(24.470, 54.600), (24.470, 54.601), (24.471, 54.601),
         (24.471, 54.600), (24.470, 54.600)],
        {},
    )
    paths = roadway.stitch_paths([loop], frame)
    assert len(paths) == 1 and paths[0].closed is True


# ── width tag resolution ───────────────────────────────────────────────────────

@pytest.mark.parametrize("tags,default,expected", [
    ({"width": "12"}, 15.0, 12.0),
    ({"width": "12.5 m"}, 15.0, 12.5),
    ({"est_width": "10"}, 15.0, 10.0),
    ({"width": "12;14"}, 15.0, 12.0),        # range -> first value
    ({"width": "500"}, 15.0, 15.0),          # absurd -> fall back
    ({}, 15.0, 15.0),                         # nothing -> default
])
def test_width_resolution(tags, default, expected):
    cl = roadway.Centerline(1, "way", [(0, 0), (0, 1)], tags)
    assert cl.width_m(default) == pytest.approx(expected)


# ── shapely cross-validation: the independent witness ─────────────────────────

def test_area_matches_shapely_buffer_on_a_wavy_line():
    shapely_geom = pytest.importorskip("shapely.geometry")
    # A gently sinusoidal open path — corners, but none sharp enough to fold.
    xy = [(t, 8.0 * math.sin(t / 40.0)) for t in range(0, 400, 5)]
    p = _path(xy)
    mesh = roadway.ribbon_mesh(p, width=6.0)

    line = shapely_geom.LineString(xy)
    buf = line.buffer(3.0, cap_style=2, join_style=2, mitre_limit=4.0)

    mine = _mesh_triangle_area(mesh)
    assert mine == pytest.approx(buf.area, rel=0.02)
    assert _all_faces_up(mesh) == 0


def test_length_survives_projection_round_trip():
    # Centre-line length must be frame-independent: same polyline, two sites.
    frame = SceneFrame(24.47, 54.60)
    pts = [(24.470, 54.600), (24.471, 54.601), (24.472, 54.600)]
    cl = roadway.Centerline(1, "way", pts, {})
    paths = roadway.stitch_paths([cl], frame)
    length = roadway._polyline_length(paths[0].xy, paths[0].closed)
    # Independent great-circle-ish estimate in local metres.
    approx = 0.0
    for a, b in zip(pts, pts[1:]):
        ax, ay, _ = frame.to_scene(*a)
        bx, by, _ = frame.to_scene(*b)
        approx += math.hypot(bx - ax, by - ay)
    assert length == pytest.approx(approx, rel=1e-9)
