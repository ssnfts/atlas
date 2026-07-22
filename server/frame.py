"""
Scene frame — the contract between real-world coordinates and 3ds Max.

Every geometry source in this project (OSM footprints, DEM terrain, a
photogrammetry mesh) has to land in the same frame or nothing lines up. That
frame is fixed here, once, and everything else is expressed relative to it.

    +X = east          +Y = true north          +Z = up
    units             metres
    origin            the site itself, at world (0, 0, 0)

Two of those are load-bearing enough to spell out:

**+Y = true north** matches 3ds Max's own convention (its north offset of 0
points along +Y), so a sun azimuth measured clockwise from north maps to the
scene without a hidden rotation. Get this wrong and every shadow is mirrored or
rotated by an amount that looks plausible until compared with a real photograph.

**Origin at the site, not at the projection origin.** UTM eastings are around
500,000 m and northings can reach 10,000,000 m. 3ds Max stores vertex positions
as 32-bit floats, which at 10^7 have a spacing of about one metre — so a
building modelled at true UTM coordinates would have its vertices quantised to
roughly a metre, producing z-fighting, jitter under camera motion, and shadow
acne. Centring the site at the origin keeps coordinates in the tens or hundreds,
where float precision is millimetric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["SceneFrame", "WGS84_A", "WGS84_F"]

# WGS84 ellipsoid
WGS84_A = 6378137.0            # semi-major axis, metres
WGS84_F = 1.0 / 298.257223563  # flattening
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)  # first eccentricity squared


@dataclass(frozen=True)
class SceneFrame:
    """
    A local east/north/up tangent plane centred on the site.

    Accurate to well under a centimetre across the few-kilometre extents this
    tool deals with, and unlike UTM it has no zone boundaries to straddle and no
    large coordinate offsets to destroy float precision.
    """

    origin_lat: float
    origin_lon: float
    origin_alt: float = 0.0

    # ── metres per degree at the origin ──────────────────────────────────────

    @property
    def _meters_per_deg_lat(self) -> float:
        lat = math.radians(self.origin_lat)
        return (
            111132.92
            - 559.82 * math.cos(2 * lat)
            + 1.175 * math.cos(4 * lat)
            - 0.0023 * math.cos(6 * lat)
        )

    @property
    def _meters_per_deg_lon(self) -> float:
        lat = math.radians(self.origin_lat)
        return (
            111412.84 * math.cos(lat)
            - 93.5 * math.cos(3 * lat)
            + 0.118 * math.cos(5 * lat)
        )

    # ── conversions ──────────────────────────────────────────────────────────

    def to_scene(self, lat: float, lon: float, alt: float = 0.0) -> tuple[float, float, float]:
        """Geodetic -> scene metres (east, north, up)."""
        return (
            (lon - self.origin_lon) * self._meters_per_deg_lon,
            (lat - self.origin_lat) * self._meters_per_deg_lat,
            alt - self.origin_alt,
        )

    def to_geodetic(self, x: float, y: float, z: float = 0.0) -> tuple[float, float, float]:
        """
        Scene metres -> geodetic. Inverse of :meth:`to_scene`.

        No production caller, and it stays anyway: it is what makes the forward
        projection *provable*. ``test_round_trip_is_lossless`` composes the two
        and asserts the identity across three latitudes, which is the only check
        that would catch a metres-per-degree coefficient that is subtly wrong in
        a self-consistent way. Deleting it as unused would delete the evidence.
        """
        return (
            self.origin_lat + y / self._meters_per_deg_lat,
            self.origin_lon + x / self._meters_per_deg_lon,
            self.origin_alt + z,
        )

    def bbox_for_radius(self, radius_m: float) -> tuple[float, float, float, float]:
        """
        (south, west, north, east) in degrees for a square of ±radius about the site.

        Ordered to match the Overpass API's bbox argument, which is
        (south, west, north, east) — not the (west, south, east, north) that
        GeoJSON and most other APIs use. Mixing the two silently queries the
        wrong region rather than erroring.
        """
        dlat = radius_m / self._meters_per_deg_lat
        dlon = radius_m / self._meters_per_deg_lon
        return (
            self.origin_lat - dlat,
            self.origin_lon - dlon,
            self.origin_lat + dlat,
            self.origin_lon + dlon,
        )
