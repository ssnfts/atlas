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

    A site near a tile corner spans two or four tiles. Fetching only the one
    containing the origin leaves the rest of the patch reading 0.0 — flat, at
    sea level, and perfectly plausible next to real terrain.
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

    Every returned value is bounds-checked against the raster before it is
    trusted, because an out-of-tile read comes back as a plausible 0.0.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise TerrainError(
            "reading Copernicus COGs needs rasterio (BSD-3): pip install rasterio"
        ) from exc

    south, west, north, east = bbox
    names = tiles_for_bbox(bbox)
    if len(names) > 1:
        # Mosaicking several tiles is a real case near a tile corner, but it is
        # not implemented yet — and returning the first tile's data for the whole
        # region would fill the rest with 0.0 and look like a coastal plain.
        raise TerrainError(
            f"this region spans {len(names)} DEM tiles ({', '.join(names)}) and "
            "mosaicking is not implemented. Move the site origin away from the "
            "tile boundary or reduce the radius."
        )

    url = f"/vsicurl/{tile_url((south + north) / 2, (west + east) / 2)}"

    try:
        with rasterio.open(url) as dataset:
            bounds = dataset.bounds
            # The load-bearing check. Tile bounds carry a half-pixel offset
            # (25.000138, not 25.0), so a point just inside a whole degree can
            # fall in the neighbouring tile even though floor() says otherwise.
            if not (
                bounds.left <= west and bounds.right >= east
                and bounds.bottom <= south and bounds.top >= north
            ):
                raise TerrainError(
                    f"tile {tile_name((south + north) / 2, (west + east) / 2)} covers "
                    f"{bounds} but the request needs ({south}, {west}) to ({north}, {east}). "
                    "Reading anyway would return 0.0 outside the overlap, which is "
                    "indistinguishable from sea level."
                )

            window = from_bounds(west, south, east, north, dataset.transform)
            array = dataset.read(1, window=window)
    except TerrainError:
        raise
    except Exception as exc:
        raise TerrainError(f"could not read the Copernicus tile: {exc}") from exc

    if array.size == 0:
        raise TerrainError("the requested window contains no DEM samples")

    # Flip north-up raster order to south-up, so the row index increases with
    # latitude and therefore with the scene frame's +Y.
    elevations = [[float(v) for v in row] for row in array[::-1]]

    return TerrainPatch(
        elevations=elevations,
        south=south,
        west=west,
        north=north,
        east=east,
        metadata={"tile": tile_name((south + north) / 2, (west + east) / 2)},
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
