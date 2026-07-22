"""
Scene frame tests — georeferencing correctness.

A frame bug does not crash; it silently places geometry in the wrong spot or
mirrors the compass, and the render still looks like a city. These check the
axis conventions and round-trip accuracy explicitly.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from frame import SceneFrame  # noqa: E402
from scene import exposure_for_altitude  # noqa: E402

DUBAI = SceneFrame(25.2048, 55.2708)
LONDON = SceneFrame(51.5074, -0.1278)
QUITO = SceneFrame(-0.1807, -78.4678)


# ── Axis conventions ──────────────────────────────────────────────────────────

def test_origin_maps_to_world_zero():
    """The site sits at (0,0,0) — the whole point of a local frame."""
    x, y, z = DUBAI.to_scene(DUBAI.origin_lat, DUBAI.origin_lon)
    assert (abs(x), abs(y), abs(z)) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_north_is_positive_y():
    """Increasing latitude must move +Y, matching Max's north offset of 0."""
    _, y, _ = DUBAI.to_scene(DUBAI.origin_lat + 0.01, DUBAI.origin_lon)
    assert y > 0


def test_east_is_positive_x():
    """Increasing longitude must move +X."""
    x, _, _ = DUBAI.to_scene(DUBAI.origin_lat, DUBAI.origin_lon + 0.01)
    assert x > 0


def test_south_and_west_are_negative():
    x, y, _ = DUBAI.to_scene(DUBAI.origin_lat - 0.01, DUBAI.origin_lon - 0.01)
    assert x < 0 and y < 0


def test_up_is_positive_z():
    _, _, z = DUBAI.to_scene(DUBAI.origin_lat, DUBAI.origin_lon, 100.0)
    assert z == pytest.approx(100.0)


def test_axis_conventions_hold_in_southern_hemisphere():
    """Sign conventions must not flip below the equator."""
    _, y, _ = QUITO.to_scene(QUITO.origin_lat + 0.01, QUITO.origin_lon)
    x, _, _ = QUITO.to_scene(QUITO.origin_lat, QUITO.origin_lon + 0.01)
    assert y > 0 and x > 0


# ── Scale ─────────────────────────────────────────────────────────────────────

def test_one_degree_of_latitude_is_about_111km():
    """True everywhere; the standard sanity check on a geodetic conversion."""
    _, y, _ = DUBAI.to_scene(DUBAI.origin_lat + 1.0, DUBAI.origin_lon)
    assert 110_500 < y < 111_700


def test_longitude_scale_shrinks_with_latitude():
    """A degree of longitude is ~111 km at the equator and ~69 km at 51N."""
    x_equator, _, _ = QUITO.to_scene(QUITO.origin_lat, QUITO.origin_lon + 1.0)
    x_london, _, _ = LONDON.to_scene(LONDON.origin_lat, LONDON.origin_lon + 1.0)
    assert 110_000 < x_equator < 112_000
    assert 68_000 < x_london < 70_000
    assert x_london < x_equator


@pytest.mark.parametrize("frame", [DUBAI, LONDON, QUITO], ids=["dubai", "london", "quito"])
@pytest.mark.parametrize(
    "dlat,dlon", [(0.01, 0.0), (0.0, 0.01), (0.01, 0.01), (-0.02, 0.015)]
)
def test_distances_match_wgs84_geodesic(frame, dlat, dlon):
    """
    Cross-check against a true WGS84 geodesic (pyproj's Geod).

    Deliberately *not* checked against a haversine sphere: haversine assumes a
    spherical Earth of mean radius, which overstates the meridian arc near the
    equator by ~0.4% (111,195 m/deg vs the true 110,776 m/deg at 25N). It would
    flag this correct ellipsoidal implementation as broken. Geod solves the real
    geodesic on WGS84 and is the right reference.
    """
    geod = pytest.importorskip("pyproj").Geod(ellps="WGS84")

    lat2 = frame.origin_lat + dlat
    lon2 = frame.origin_lon + dlon
    x, y, _ = frame.to_scene(lat2, lon2)

    _, _, true_dist = geod.inv(frame.origin_lon, frame.origin_lat, lon2, lat2)
    ours = math.hypot(x, y)

    # Sub-0.1% over these distances (~1-3 km).
    assert abs(ours - true_dist) / true_dist < 0.001, (
        f"frame {ours:.2f} m vs WGS84 geodesic {true_dist:.2f} m"
    )


# ── Round trip ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("frame", [DUBAI, LONDON, QUITO], ids=["dubai", "london", "quito"])
@pytest.mark.parametrize("dx,dy", [(0, 0), (500, 500), (-2000, 1500), (5000, -5000)])
def test_round_trip_is_lossless(frame, dx, dy):
    lat, lon, _ = frame.to_geodetic(dx, dy, 0.0)
    x, y, _ = frame.to_scene(lat, lon)
    assert x == pytest.approx(dx, abs=0.01)
    assert y == pytest.approx(dy, abs=0.01)


# ── Bounding box ──────────────────────────────────────────────────────────────

def test_bbox_is_south_west_north_east_order():
    """
    Overpass takes (south, west, north, east) while GeoJSON uses
    (west, south, east, north). Swapping them queries the wrong place silently.
    """
    south, west, north, east = DUBAI.bbox_for_radius(1000.0)
    assert south < DUBAI.origin_lat < north
    assert west < DUBAI.origin_lon < east
    assert south < north and west < east


def test_bbox_radius_is_honoured():
    south, west, north, east = DUBAI.bbox_for_radius(1000.0)
    _, y_north, _ = DUBAI.to_scene(north, DUBAI.origin_lon)
    x_east, _, _ = DUBAI.to_scene(DUBAI.origin_lat, east)
    assert y_north == pytest.approx(1000.0, abs=1.0)
    assert x_east == pytest.approx(1000.0, abs=1.0)


# ── Float precision rationale ─────────────────────────────────────────────────

def test_local_coordinates_stay_small_enough_for_float32():
    """
    The reason for a local origin: at UTM magnitudes float32 spacing approaches
    a metre, which is what causes z-fighting and viewport jitter in Max.
    """
    import struct

    def f32_spacing(v: float) -> float:
        a = struct.unpack("f", struct.pack("f", v))[0]
        b = struct.unpack("f", struct.pack("f", v * (1 + 1e-7)))[0]
        return abs(b - a)

    x, y, _ = DUBAI.to_scene(DUBAI.origin_lat + 0.01, DUBAI.origin_lon + 0.01)
    assert max(abs(x), abs(y)) < 2000.0
    assert f32_spacing(max(abs(x), abs(y))) < 0.001  # sub-millimetre locally
    assert f32_spacing(10_000_000.0) > 0.5           # ...and metre-scale at UTM northings


# ── Exposure ──────────────────────────────────────────────────────────────────

def test_exposure_opens_up_as_sun_drops():
    """Lower sun means less light, so a longer shutter."""
    high = exposure_for_altitude(80.0)["shutter_speed"]
    mid = exposure_for_altitude(30.0)["shutter_speed"]
    low = exposure_for_altitude(8.0)["shutter_speed"]
    assert high > mid > low


def test_exposure_below_horizon_is_flagged_as_a_guess():
    out = exposure_for_altitude(-3.0)
    assert "guess" in out["note"]
    assert out["iso"] > 100.0


def test_exposure_shutter_stays_in_a_usable_range():
    for alt in range(1, 90):
        shutter = exposure_for_altitude(float(alt))["shutter_speed"]
        assert 30.0 <= shutter <= 4000.0
