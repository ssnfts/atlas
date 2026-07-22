"""
Terrain tests — no network, no 3ds Max.

The failure this module is built against is measured, not hypothetical: reading
a point outside a Copernicus tile returns **0.0**, the dataset declares no
nodata value, and 0.0 is a real sea-level elevation. A desert point 40 km inland
read 0.0 from the wrong tile and 113.7 m from the right one, with nothing
anywhere to say which had happened.

So the tile arithmetic and the bounds checks are tested hard, and
``elevation_at`` is required to raise rather than return anything at all outside
its patch.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from frame import SceneFrame  # noqa: E402
from osm import Building  # noqa: E402
from terrain import (  # noqa: E402
    ATTRIBUTION,
    TerrainError,
    TerrainPatch,
    looks_like_surface_model,
    tile_name,
    tile_url,
    tiles_for_bbox,
)


# ── Tile naming ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "lat,lon,expected",
    [
        (25.08, 55.14, "N25_00_E055_00"),     # Dubai Marina
        (25.0, 55.0, "N25_00_E055_00"),       # exactly on the corner
        (25.999, 55.999, "N25_00_E055_00"),   # just inside
        (51.5074, -0.1278, "N51_00_W001_00"), # London, west of Greenwich
        (-33.8688, 151.2093, "S34_00_E151_00"),  # Sydney
        (0.0, 0.0, "N00_00_E000_00"),
    ],
)
def test_tile_names(lat, lon, expected):
    assert tile_name(lat, lon) == expected


def test_southern_and_western_hemispheres_floor_away_from_zero():
    """
    A point at 0.5°S is in tile S01, not S00 — tiles are named for their
    south-west corner. int() truncates toward zero and picks S00, which is the
    tile to the north, and the error is one degree of plausible terrain.
    """
    assert tile_name(-0.5, -0.5) == "S01_00_W001_00"
    assert tile_name(-0.5, 0.5) == "S01_00_E000_00"
    assert tile_name(0.5, -0.5) == "N00_00_W001_00"


def test_london_is_west_not_east():
    """Longitude -0.13 must floor to W001. Getting the sign wrong lands in Kent."""
    assert tile_name(51.5, -0.13).endswith("W001_00")


def test_tile_url_shape():
    url = tile_url(25.08, 55.14)
    assert url.endswith(
        "Copernicus_DSM_COG_10_N25_00_E055_00_DEM/Copernicus_DSM_COG_10_N25_00_E055_00_DEM.tif"
    )


@pytest.mark.parametrize("lat,lon", [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)])
def test_out_of_range_coordinates_are_rejected(lat, lon):
    with pytest.raises(TerrainError):
        tile_name(lat, lon)


# ── Multi-tile coverage ───────────────────────────────────────────────────────

def test_bbox_inside_one_tile_needs_one_tile():
    assert tiles_for_bbox((25.07, 55.13, 25.09, 55.15)) == ["N25_00_E055_00"]


def test_bbox_across_a_latitude_boundary_needs_two_tiles():
    """
    A site near a tile edge spans several tiles. Fetching only the origin's tile
    leaves the rest of the patch at 0.0 — a flat sea-level shelf butted against
    real terrain, which looks like a coastline.
    """
    names = tiles_for_bbox((24.99, 55.13, 25.01, 55.15))
    assert set(names) == {"N24_00_E055_00", "N25_00_E055_00"}


def test_bbox_across_a_corner_needs_four_tiles():
    names = tiles_for_bbox((24.99, 54.99, 25.01, 55.01))
    assert len(names) == 4


def test_inverted_bbox_is_rejected():
    with pytest.raises(TerrainError):
        tiles_for_bbox((25.1, 55.0, 25.0, 55.1))
    with pytest.raises(TerrainError):
        tiles_for_bbox((25.0, 55.1, 25.1, 55.0))


# ── Patch construction ────────────────────────────────────────────────────────

def _flat_patch(value: float = 5.0, rows: int = 4, cols: int = 4) -> TerrainPatch:
    return TerrainPatch(
        elevations=[[value] * cols for _ in range(rows)],
        south=25.0, west=55.0, north=25.01, east=55.01,
    )


def _ramp_patch() -> TerrainPatch:
    """Elevation rises with latitude: 0 m at the south edge, 30 m at the north."""
    return TerrainPatch(
        elevations=[[float(r * 10)] * 4 for r in range(4)],
        south=25.0, west=55.0, north=25.03, east=55.03,
    )


def test_empty_patch_is_rejected():
    with pytest.raises(TerrainError):
        TerrainPatch(elevations=[], south=25.0, west=55.0, north=25.1, east=55.1)


def test_degenerate_bbox_is_rejected():
    with pytest.raises(TerrainError):
        TerrainPatch(elevations=[[1.0]], south=25.0, west=55.0, north=25.0, east=55.1)


def test_grid_dimensions():
    patch = _flat_patch(rows=5, cols=7)
    assert (patch.rows, patch.cols) == (5, 7)


# ── Sampling ──────────────────────────────────────────────────────────────────

def test_flat_patch_samples_flat():
    patch = _flat_patch(7.5)
    assert patch.elevation_at(25.005, 55.005) == pytest.approx(7.5)


def test_row_zero_is_south():
    """
    GeoTIFFs are north-up; this stores south-up so the row index increases with
    latitude and with the scene's +Y. Leaving the raster order in place gives
    terrain that is correct except mirrored north-to-south — which still looks
    like terrain.
    """
    patch = _ramp_patch()
    south_sample = patch.elevation_at(25.0, 55.015)
    north_sample = patch.elevation_at(25.03, 55.015)
    assert south_sample == pytest.approx(0.0)
    assert north_sample == pytest.approx(30.0)
    assert north_sample > south_sample


def test_bilinear_interpolation_is_linear_on_a_ramp():
    patch = _ramp_patch()
    midpoint = patch.elevation_at(25.015, 55.015)
    assert midpoint == pytest.approx(15.0)


def test_corners_return_their_own_values():
    patch = TerrainPatch(
        elevations=[[1.0, 2.0], [3.0, 4.0]],
        south=25.0, west=55.0, north=25.01, east=55.01,
    )
    assert patch.elevation_at(25.0, 55.0) == pytest.approx(1.0)     # SW
    assert patch.elevation_at(25.0, 55.01) == pytest.approx(2.0)    # SE
    assert patch.elevation_at(25.01, 55.0) == pytest.approx(3.0)    # NW
    assert patch.elevation_at(25.01, 55.01) == pytest.approx(4.0)   # NE


def test_east_west_orientation():
    """col 0 is west. Mirroring this flips the terrain against the buildings."""
    patch = TerrainPatch(
        elevations=[[0.0, 100.0], [0.0, 100.0]],
        south=25.0, west=55.0, north=25.01, east=55.01,
    )
    assert patch.elevation_at(25.005, 55.0) == pytest.approx(0.0)
    assert patch.elevation_at(25.005, 55.01) == pytest.approx(100.0)


# ── The zero trap ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "lat,lon",
    [(24.99, 55.005), (25.02, 55.005), (25.005, 54.99), (25.005, 55.02)],
)
def test_sampling_outside_the_patch_raises(lat, lon):
    """
    Measured behaviour of the raster: an out-of-bounds read returns 0.0, and the
    dataset declares no nodata value. Since 0.0 is a real sea-level elevation,
    a silent zero here is a 100 m error that reads as a coastline. It must raise.
    """
    with pytest.raises(TerrainError, match="outside this terrain patch"):
        _flat_patch().elevation_at(lat, lon)


def test_the_error_explains_why_it_refuses_to_guess():
    with pytest.raises(TerrainError, match="sea level"):
        _flat_patch().elevation_at(0.0, 0.0)


def test_contains_matches_what_sampling_allows():
    patch = _flat_patch()
    assert patch.contains(25.005, 55.005)
    assert not patch.contains(24.999, 55.005)
    assert not patch.contains(25.011, 55.005)


def test_zero_elevation_is_a_legitimate_value():
    """Sea level must survive as data, not be treated as missing."""
    patch = _flat_patch(0.0)
    assert patch.elevation_at(25.005, 55.005) == 0.0
    assert patch.min_elevation() == 0.0


# ── Statistics ────────────────────────────────────────────────────────────────

def test_relief_is_the_height_range():
    assert _ramp_patch().relief() == pytest.approx(30.0)


def test_flat_terrain_has_no_relief():
    assert _flat_patch().relief() == pytest.approx(0.0)


def test_mean_elevation():
    assert _ramp_patch().mean_elevation() == pytest.approx(15.0)


def test_negative_elevations_survive():
    """Below-sea-level ground is real — the Dead Sea, Death Valley, dredged marinas."""
    patch = TerrainPatch(
        elevations=[[-12.0, -8.0], [-4.0, 0.0]],
        south=25.0, west=55.0, north=25.01, east=55.01,
    )
    assert patch.min_elevation() == -12.0
    assert patch.elevation_at(25.0, 55.0) == pytest.approx(-12.0)


# ── Meshing ───────────────────────────────────────────────────────────────────

DUBAI = SceneFrame(25.0, 55.0)


def test_mesh_has_two_triangles_per_cell():
    mesh = _flat_patch(rows=4, cols=4).to_mesh(DUBAI)
    assert mesh.vertex_count == 16
    assert mesh.face_count == 2 * 3 * 3


def test_mesh_normals_point_up():
    """
    A terrain sheet wound downward is lit from underneath: black under a V-Ray
    sun, and perfectly normal-looking in the viewport. Checked by the sign of
    each triangle's Z component, since an open sheet has no volume to test.
    """
    mesh = _flat_patch(rows=5, cols=5).to_mesh(DUBAI)
    for i, j, k in mesh.faces:
        a, b, c = mesh.verts[i], mesh.verts[j], mesh.verts[k]
        zcross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        assert zcross > 0, "terrain triangle is wound clockwise from above"


def test_mesh_carries_elevation_into_z():
    mesh = _ramp_patch().to_mesh(DUBAI)
    zs = [v[2] for v in mesh.verts]
    assert min(zs) == pytest.approx(0.0)
    assert max(zs) == pytest.approx(30.0)


def test_mesh_north_vertices_are_positive_y():
    """The mesh must agree with the scene frame's compass, not just its own."""
    mesh = _ramp_patch().to_mesh(SceneFrame(25.015, 55.015))
    highest = max(mesh.verts, key=lambda v: v[2])
    lowest = min(mesh.verts, key=lambda v: v[2])
    assert highest[1] > lowest[1], "the high (northern) edge is not at +Y"


def test_stride_decimates_but_keeps_the_extent():
    """
    Decimation must not shrink the patch. Dropping the last row because it does
    not fall on a stride boundary trims the terrain silently, leaving a gap
    between the terrain edge and the buildings that sit on it.
    """
    patch = _flat_patch(rows=10, cols=10)
    full = patch.to_mesh(DUBAI)
    strided = patch.to_mesh(DUBAI, stride=3)
    assert strided.face_count < full.face_count
    (fx0, fy0, _), (fx1, fy1, _) = full.bounds()
    (sx0, sy0, _), (sx1, sy1, _) = strided.bounds()
    assert (sx0, sy0, sx1, sy1) == pytest.approx((fx0, fy0, fx1, fy1))


def test_stride_must_be_positive():
    with pytest.raises(TerrainError):
        _flat_patch().to_mesh(DUBAI, stride=0)


def test_mesh_carries_attribution():
    """Copernicus requires a verbatim notice; it has to survive into the build."""
    mesh = _flat_patch().to_mesh(DUBAI)
    assert mesh.metadata["attribution"] == ATTRIBUTION
    assert "DLR e.V." in ATTRIBUTION and "Airbus" in ATTRIBUTION


# ── Surface-model check ───────────────────────────────────────────────────────

def _building(lat, lon, height):
    ring = [(lat, lon), (lat, lon + 0.0002), (lat + 0.0002, lon + 0.0002), (lat + 0.0002, lon)]
    return Building(1, "way", ring, tags={"height": str(height)})


def test_bare_earth_dem_is_not_flagged():
    """Measured GLO-30 behaviour: flat ground under a very tall building."""
    patch = _flat_patch(5.0)
    result = looks_like_surface_model(patch, [_building(25.005, 55.005, 828.0)])
    assert result["is_surface_model"] is False
    assert result["checked"] == 1
    assert "bare earth" in result["note"]


def test_a_dem_containing_buildings_is_flagged():
    """
    If the DEM did carry building heights, stacking massing on it would
    double-count every roof. Here the terrain under the tower stands 200 m above
    the patch mean, which is most of the building's own height.
    """
    elevations = [[0.0] * 4 for _ in range(4)]
    elevations[2][2] = 400.0
    elevations[2][1] = 400.0
    elevations[1][2] = 400.0
    elevations[1][1] = 400.0
    patch = TerrainPatch(
        elevations=elevations, south=25.0, west=55.0, north=25.03, east=55.03
    )
    result = looks_like_surface_model(patch, [_building(25.015, 55.015, 300.0)])
    assert result["is_surface_model"] is True
    assert "double-count" in result["note"]


def test_buildings_outside_the_patch_are_skipped_not_fatal():
    patch = _flat_patch()
    result = looks_like_surface_model(patch, [_building(40.0, 10.0, 100.0)])
    assert result["checked"] == 0
    assert result["is_surface_model"] is False
