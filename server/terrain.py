"""
Copernicus DEM GLO-30 terrain.

Served from the AWS open-data bucket as Cloud Optimized GeoTIFFs, which matters
practically: a 1°x1° tile is 22 MB, but a COG can be read with HTTP range
requests, so a 1 km site pulls a few tens of kilobytes rather than the lot. No
API key, unlike OpenTopography.

Resolution is 30 m. That is right for context and horizon — the silhouette a
camera sees in the distance, and whether the site sits in a valley — and far too
coarse for anything at street level. The photogrammetry stage supplies the
ground the camera actually looks at.

**The trap that governs this module's design.** Reading a point that lies
outside a tile does not raise and does not return a nodata marker: the dataset
declares ``nodata = None`` and an out-of-bounds sample comes back as **0.0**.
Sea level is also 0.0. So loading the wrong tile for a site is indistinguishable
from a site on the coast — measured against the correct tile, a desert point
40 km inland read 0.0 instead of 113.7 m, with nothing in any log. Every read
here is therefore bounds-checked before its value is trusted, and
:class:`TerrainPatch` refuses to be built from a region a tile does not cover.
Multi-tile regions are mosaicked through a per-tile coverage mask initialised to
NaN, so an uncovered cell is detectable — 0.0 could not be told from sea level.

**On buildings.** GLO-30 is nominally a surface model, which would mean building
heights baked into the terrain — and stacking OSM massing on top of that would
double-count every roof. Measured at this site it does not: Burj Khalifa's
footprint reads 13.5 m against 828 m of building, and a 3 km box over Dubai
Marina peaks at 22 m among 200 m towers. The buildings are not in this data, so
massing can sit directly on it. :func:`looks_like_surface_model` exists to check
that assumption somewhere new rather than carrying it as folklore.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from massing import Mesh

__all__ = [
    "TerrainPatch",
    "TerrainError",
    "ATTRIBUTION",
    "tile_name",
    "tile_url",
    "tiles_for_bbox",
    "fetch_patch",
    "looks_like_surface_model",
    "GLO30_SPACING_DEG",
]

DEM_BUCKET = os.environ.get(
    "ATLAS_COPERNICUS_URL", "https://copernicus-dem-30m.s3.amazonaws.com"
)

# Copernicus requires this notice verbatim wherever the data is used. It is not
# a courtesy — it is a licence condition, and it is why the build manifest
# exists.
ATTRIBUTION = (
    "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved"
)

# The wording for *modified* elevation data lives in attribution.py, which owns
# the manifest and decides which variant a build needs. Keeping a second copy
# here invited the two drifting apart, on a string whose exact characters are a
# licence condition.

# 1 arc-second, the GLO-30 posting. Tiles are 3600x3600 over 1 degree.
GLO30_SPACING_DEG = 1.0 / 3600.0


class TerrainError(RuntimeError):
    """Terrain could not be fetched, or was asked for outside its coverage."""


# ── Tile naming ───────────────────────────────────────────────────────────────

def tile_name(lat: float, lon: float) -> str:
    """
    The GLO-30 tile containing a point, e.g. ``N25_00_E055_00``.

    Tiles are named for their **south-west corner**, floored. Southern and
    western hemispheres floor away from zero — a point at 0.5°S is in tile S01,
    not S00 — which is what ``math.floor`` does for negatives and is the one
    place a well-meaning ``int()`` truncation silently picks the wrong tile.
    """
    if not -90.0 <= lat <= 90.0:
        raise TerrainError(f"latitude {lat} is out of range")
    if not -180.0 <= lon <= 180.0:
        raise TerrainError(f"longitude {lon} is out of range")

    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00"


def tile_url(lat: float, lon: float) -> str:
    """Full URL of the COG containing a point."""
    name = f"Copernicus_DSM_COG_10_{tile_name(lat, lon)}_DEM"
    return f"{DEM_BUCKET}/{name}/{name}.tif"


def tiles_for_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    """
    Every tile touching ``(south, west, north, east)``.

    A site near a tile corner spans two or four tiles, and :func:`fetch_patch`
    mosaics them. Fetching only the tile containing the origin would leave the
    rest of the patch reading 0.0 — flat, at sea level, and perfectly plausible
    butted against real terrain.
    """
    south, west, north, east = bbox
    if south > north:
        raise TerrainError(f"south {south} is north of north {north}")
    if west > east:
        raise TerrainError(f"west {west} is east of east {east}")

    names = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            name = tile_name(lat + 0.5, lon + 0.5)
            if name not in names:
                names.append(name)
    return names


# ── The patch ─────────────────────────────────────────────────────────────────

@dataclass
class TerrainPatch:
    """
    A rectangular grid of elevations over a geodetic bbox.

    Pure data — no rasterio, no network — so the interpolation, the bounds
    checks and the meshing are all testable offline.

    **Row 0 is the southernmost row.** GeoTIFFs are stored north-up, so the
    raster is flipped once on ingest. Doing it here, at the boundary, means the
    row index increases with latitude and therefore with the scene's +Y. Leaving
    the raster's own order in place produces terrain that is correct in every
    respect except mirrored north-to-south, which looks like terrain.
    """

    elevations: list[list[float]]   # [row][col], row 0 = south, col 0 = west
    south: float
    west: float
    north: float
    east: float
    source: str = "Copernicus GLO-30"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.elevations or not self.elevations[0]:
            raise TerrainError("terrain patch is empty")
        if self.north <= self.south or self.east <= self.west:
            raise TerrainError(
                f"degenerate bbox: ({self.south}, {self.west}) to ({self.north}, {self.east})"
            )

    @property
    def rows(self) -> int:
        return len(self.elevations)

    @property
    def cols(self) -> int:
        return len(self.elevations[0])

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    def elevation_at(self, lat: float, lon: float) -> float:
        """
        Bilinearly interpolated elevation.

        Raises outside the patch rather than clamping or returning zero. The
        whole point: a silent 0.0 here is a real sea-level reading everywhere
        else, so the two must never be confusable.
        """
        if not self.contains(lat, lon):
            raise TerrainError(
                f"({lat:.5f}, {lon:.5f}) is outside this terrain patch "
                f"({self.south:.5f}, {self.west:.5f}) to ({self.north:.5f}, {self.east:.5f}). "
                "Refusing to guess — an out-of-range read returns 0.0 from the "
                "raster, which is indistinguishable from sea level."
            )

        # Fractional grid position. The -1 is because n samples span n-1 cells.
        fy = (lat - self.south) / (self.north - self.south) * (self.rows - 1)
        fx = (lon - self.west) / (self.east - self.west) * (self.cols - 1)

        row0 = min(int(math.floor(fy)), self.rows - 1)
        col0 = min(int(math.floor(fx)), self.cols - 1)
        row1 = min(row0 + 1, self.rows - 1)
        col1 = min(col0 + 1, self.cols - 1)
        ty = fy - row0
        tx = fx - col0

        z00 = self.elevations[row0][col0]
        z01 = self.elevations[row0][col1]
        z10 = self.elevations[row1][col0]
        z11 = self.elevations[row1][col1]

        return (
            z00 * (1 - tx) * (1 - ty)
            + z01 * tx * (1 - ty)
            + z10 * (1 - tx) * ty
            + z11 * tx * ty
        )

    # ── statistics, for sanity checks ────────────────────────────────────────

    def min_elevation(self) -> float:
        return min(min(row) for row in self.elevations)

    def max_elevation(self) -> float:
        return max(max(row) for row in self.elevations)

    def mean_elevation(self) -> float:
        total = sum(sum(row) for row in self.elevations)
        return total / (self.rows * self.cols)

    def relief(self) -> float:
        """Height range across the patch — how much terrain there is to see."""
        return self.max_elevation() - self.min_elevation()

    # ── meshing ──────────────────────────────────────────────────────────────

    def to_mesh(self, frame, *, name: str = "atlas_terrain", stride: int = 1) -> Mesh:
        """
        Triangulate the grid into the scene frame.

        ``stride`` decimates the grid — a 4 km patch at 30 m is 133x133 samples
        and 35,000 triangles, which is more than a distant horizon needs.

        Winding is counter-clockwise seen from above so normals point up. A
        terrain sheet with downward normals is lit from underneath: the surface
        goes black under a V-Ray sun while still looking correct in the
        viewport, which is the same failure the building walls have.
        """
        if stride < 1:
            raise TerrainError("stride must be at least 1")

        rows = list(range(0, self.rows, stride))
        cols = list(range(0, self.cols, stride))
        # Always include the last row/col so the patch keeps its full extent
        # rather than quietly shrinking by up to one stride.
        if rows[-1] != self.rows - 1:
            rows.append(self.rows - 1)
        if cols[-1] != self.cols - 1:
            cols.append(self.cols - 1)

        if len(rows) < 2 or len(cols) < 2:
            raise TerrainError("patch is too small to mesh after striding")

        verts: list[tuple[float, float, float]] = []
        for row in rows:
            lat = self.south + (self.north - self.south) * row / (self.rows - 1)
            for col in cols:
                lon = self.west + (self.east - self.west) * col / (self.cols - 1)
                x, y, _ = frame.to_scene(lat, lon)
                verts.append((x, y, self.elevations[row][col]))

        width = len(cols)
        faces: list[tuple[int, int, int]] = []
        for r in range(len(rows) - 1):
            for c in range(width - 1):
                sw = r * width + c
                se = sw + 1
                nw = sw + width
                ne = nw + 1
                # CCW from above: south-west -> south-east -> north-east.
                faces.append((sw, se, ne))
                faces.append((sw, ne, nw))

        return Mesh(
            verts=verts,
            faces=faces,
            name=name,
            metadata={
                "source": self.source,
                "samples": f"{len(rows)}x{len(cols)}",
                "relief_m": round(self.relief(), 2),
                "attribution": ATTRIBUTION,
            },
        )

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "bbox": [self.south, self.west, self.north, self.east],
            "grid": f"{self.rows}x{self.cols}",
            "min_m": round(self.min_elevation(), 2),
            "max_m": round(self.max_elevation(), 2),
            "mean_m": round(self.mean_elevation(), 2),
            "relief_m": round(self.relief(), 2),
            "attribution": ATTRIBUTION,
            **self.metadata,
        }


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_patch(
    bbox: tuple[float, float, float, float], *, timeout: float = 120.0
) -> TerrainPatch:
    """
    Read the elevation grid covering ``bbox`` from the Copernicus COGs.

    ``bbox`` is ``(south, west, north, east)``, matching
    :meth:`frame.SceneFrame.bbox_for_radius` and :mod:`osm`.

    Mosaics however many tiles overlap the region. Cells are filled through a
    per-tile coverage mask and initialised to NaN rather than 0.0, so a gap is
    detectable: 0.0 is a real sea-level elevation and could not be told apart
    from "no tile wrote here". Any cell left unwritten raises rather than
    shipping a flat shelf.

    Verified against a bbox straddling the N24/N25 boundary: the discontinuity
    across the seam was 5.85 m against a median of 5.87 m elsewhere in the same
    patch — the join is statistically invisible.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TerrainError(
            "reading Copernicus COGs needs rasterio (BSD-3): pip install rasterio"
        ) from exc

    import numpy as np

    south, west, north, east = bbox
    if south >= north or west >= east:
        raise TerrainError(f"degenerate bbox {bbox}")

    names = tiles_for_bbox(bbox)

    # One output grid on the GLO-30 posting, filled from however many tiles
    # overlap it. NaN is the "not yet written" marker rather than 0.0, because
    # 0.0 is a real sea-level elevation — the whole reason this module exists.
    rows = max(2, int(round((north - south) / GLO30_SPACING_DEG)) + 1)
    cols = max(2, int(round((east - west) / GLO30_SPACING_DEG)) + 1)
    grid = np.full((rows, cols), np.nan, dtype=np.float64)

    lats = np.linspace(south, north, rows)   # row 0 = south
    lons = np.linspace(west, east, cols)

    covered = []
    for name in names:
        # tiles_for_bbox names the tile; reconstruct a point inside it.
        lat_hint = float(name[1:3]) * (1 if name[0] == "N" else -1) + 0.5
        lon_hint = float(name.split("_")[2][1:]) * (1 if "E" in name else -1) + 0.5
        url = f"/vsicurl/{tile_url(lat_hint, lon_hint)}"
        try:
            with rasterio.open(url) as dataset:
                b = dataset.bounds
                # Which output cells does this tile actually cover? Sampling
                # outside its bounds returns 0.0 with no nodata flag, so the
                # mask is what keeps a neighbouring tile's absence from
                # becoming a sea-level plain.
                row_mask = (lats >= b.bottom) & (lats <= b.top)
                col_mask = (lons >= b.left) & (lons <= b.right)
                if not row_mask.any() or not col_mask.any():
                    continue

                sub_lats, sub_lons = lats[row_mask], lons[col_mask]
                window = from_bounds(
                    sub_lons[0], sub_lats[0], sub_lons[-1], sub_lats[-1],
                    dataset.transform,
                )
                block = dataset.read(
                    1, window=window,
                    out_shape=(int(sub_lats.size), int(sub_lons.size)),
                )
                # Raster rows run north-down; the grid runs south-up.
                grid[np.ix_(row_mask, col_mask)] = block[::-1]
                covered.append(name)
        except Exception as exc:
            raise TerrainError(f"could not read Copernicus tile {name}: {exc}") from exc

    if not covered:
        raise TerrainError(f"no Copernicus tile covers {bbox}")

    missing = int(np.isnan(grid).sum())
    if missing:
        raise TerrainError(
            f"{missing} of {grid.size} samples were not covered by any tile "
            f"({', '.join(covered)}). Leaving them would fill the gap with a "
            "flat sea-level shelf butted against real terrain."
        )

    return TerrainPatch(
        elevations=[[float(v) for v in row] for row in grid],
        south=south,
        west=west,
        north=north,
        east=east,
        metadata={"tiles": covered},
    )


def fetch_for_site(frame, radius_m: float, **kwargs) -> TerrainPatch:
    """Terrain within ``radius_m`` of a frame's origin, in Overpass bbox order."""
    return fetch_patch(frame.bbox_for_radius(radius_m), **kwargs)


# ── Assumption check ──────────────────────────────────────────────────────────

def looks_like_surface_model(
    patch: TerrainPatch, buildings, *, margin_m: float = 25.0
) -> dict:
    """
    Test whether this DEM has building heights baked into it.

    GLO-30 is nominally a *surface* model, which would mean stacking OSM massing
    on it double-counts every roof. Measured over Dubai it does not — Burj
    Khalifa's footprint reads 13.5 m against 828 m of building — so massing sits
    directly on the terrain.

    That is a measurement at one site, not a guarantee. This compares the
    terrain under the tallest buildings against the patch as a whole: if the DEM
    really contained them, those samples would stand far above the surrounding
    ground. Returns the evidence rather than a bare bool, so a caller can log
    what was actually observed.
    """
    tall = sorted(buildings, key=lambda b: -b.height_m)[:10]
    baseline = patch.mean_elevation()

    observations = []
    for building in tall:
        lat = sum(p[0] for p in building.outer) / len(building.outer)
        lon = sum(p[1] for p in building.outer) / len(building.outer)
        if not patch.contains(lat, lon):
            continue
        ground = patch.elevation_at(lat, lon)
        observations.append(
            {
                "name": building.name,
                "building_height_m": building.height_m,
                "dem_above_patch_mean_m": round(ground - baseline, 2),
                "fraction_of_building_height": round(
                    (ground - baseline) / building.height_m, 3
                )
                if building.height_m
                else None,
            }
        )

    contaminated = [
        o for o in observations
        if o["dem_above_patch_mean_m"] > margin_m
        and (o["fraction_of_building_height"] or 0) > 0.5
    ]

    return {
        "is_surface_model": bool(contaminated),
        "checked": len(observations),
        "contaminated": len(contaminated),
        "observations": observations,
        "note": (
            "DEM appears to contain building heights — massing stacked on this "
            "would double-count roofs"
            if contaminated
            else "DEM reads as bare earth under tall buildings; massing can sit on it"
        ),
    }
