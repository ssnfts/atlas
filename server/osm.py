"""
OpenStreetMap building footprints via the Overpass API.

Supplies the context massing around a site — the blocks that cast shadows into
the shot and occlude the sky. It is not survey data and does not pretend to be:
OSM building heights are contributed, uneven in coverage and frequently absent
altogether. What this module guarantees is that every height it reports carries
a record of *where it came from*, so a facade at a guessed height is
distinguishable from one at a tagged height downstream.

The data is ODbL. Rendered images are unencumbered; redistributing the geometry
itself can trigger share-alike, which is why :data:`ATTRIBUTION` exists and ends
up in the build manifest.

Network access lives in :func:`fetch_buildings` and nothing else. Parsing is
pure, so the awkward cases — split multipolygon rings, feet-and-inches heights,
European decimal commas — are all testable without touching the network.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

__all__ = [
    "Building",
    "OverpassError",
    "ATTRIBUTION",
    "build_query",
    "query_for_site",
    "fetch_for_site",
    "fetch_buildings",
    "parse_overpass",
    "resolve_height",
    "assemble_rings",
    "METRES_PER_LEVEL",
    "DEFAULT_HEIGHT_M",
]

OVERPASS_URL = os.environ.get(
    "ATLAS_OVERPASS_URL", "https://overpass-api.de/api/interpreter"
)

ATTRIBUTION = "Building data © OpenStreetMap contributors, licensed under ODbL 1.0"

# Storey height when only ``building:levels`` is known. 3.0 m is the usual
# planning rule of thumb across mixed residential/commercial stock — floor to
# floor including the slab, not floor to ceiling.
METRES_PER_LEVEL = 3.0

# Last-resort height for a building tagged with neither a height nor a level
# count. Most of OSM outside well-mapped city centres is in this state, so this
# value is load-bearing for how a skyline looks. Roughly two storeys plus a
# parapet — deliberately low, because under-tall massing reads as missing detail
# whereas over-tall massing throws shadows across the shot that do not exist.
DEFAULT_HEIGHT_M = 8.0


class OverpassError(RuntimeError):
    """Overpass could not be reached, or returned something unusable."""


@dataclass
class Building:
    """
    One building footprint, still in geodetic coordinates.

    Rings are ``(lat, lon)`` pairs and are **not** closed — the last vertex is
    not a repeat of the first, since the closing edge is implied and a duplicated
    vertex produces a zero-length edge that breaks triangulation.

    The three height fields are **derived from the tags, not constructed**. When
    they were ordinary fields it was possible to build a Building carrying
    ``height=45`` whose ``height_m`` was still the 8 m default — the tags said
    one thing, the geometry did another, and nothing reconciled them. Making
    them ``init=False`` means the tags are the single source of truth by
    construction. Override after construction if a caller really must.
    """

    osm_id: int
    osm_type: str                       # "way" or "relation"
    outer: list[tuple[float, float]]
    inners: list[list[tuple[float, float]]] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    height_m: float = field(init=False, default=DEFAULT_HEIGHT_M)
    min_height_m: float = field(init=False, default=0.0)
    height_source: str = field(init=False, default="default")

    def __post_init__(self) -> None:
        self.height_m, self.min_height_m, self.height_source = resolve_height(self.tags)

    @property
    def name(self) -> str | None:
        return self.tags.get("name")

    @property
    def material(self) -> str | None:
        """``building:material`` — a hint for tag-driven material assignment."""
        return self.tags.get("building:material")


# ── Query ─────────────────────────────────────────────────────────────────────

def build_query(
    bbox: tuple[float, float, float, float],
    *,
    timeout_s: int = 180,
    include_parts: bool = False,
) -> str:
    """
    Overpass QL for every building in ``bbox``.

    ``bbox`` is ``(south, west, north, east)`` — the order
    :meth:`frame.SceneFrame.bbox_for_radius` returns, and the order Overpass
    itself expects. GeoJSON, Leaflet and most web APIs use
    ``(west, south, east, north)``.

    The range checks below catch a transposition only when the longitudes
    exceed ±90 and become impossible latitudes. **Inside that band the mistake
    is undetectable here** — a transposed Dubai bbox has "south" = 55.27, a
    perfectly ordinary latitude in northern Russia, and the query succeeds and
    returns nothing. Prefer :func:`query_for_site`, which takes the frame and
    builds the tuple itself, so no caller writes the ordering by hand.

    ``out geom`` is what makes a single request sufficient: it inlines the
    coordinates of every way and of every relation member, so no follow-up
    node lookup is needed.
    """
    south, west, north, east = bbox
    if not (-90.0 <= south <= north <= 90.0):
        raise ValueError(
            f"bad latitude range: south={south}, north={north}. "
            "bbox is (south, west, north, east), not (west, south, east, north)."
        )
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError(f"bad longitude range: west={west}, east={east}")

    area = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    selectors = ['way["building"]', 'relation["building"]']
    if include_parts:
        selectors += ['way["building:part"]', 'relation["building:part"]']

    body = "\n  ".join(f"{sel}({area});" for sel in selectors)
    return f"[out:json][timeout:{timeout_s}];\n(\n  {body}\n);\nout geom;"


def query_for_site(frame, radius_m: float, **kwargs) -> str:
    """
    Overpass QL for a square of ±``radius_m`` about a :class:`~frame.SceneFrame`.

    The preferred entry point, because it removes the one mistake that cannot
    be validated after the fact: the frame produces the bbox in Overpass order
    and it is passed straight through, so there is no tuple for a caller to
    assemble in GeoJSON order by mistake.
    """
    return build_query(frame.bbox_for_radius(radius_m), **kwargs)


def fetch_for_site(frame, radius_m: float, **kwargs) -> list[Building]:
    """Fetch every building within ``radius_m`` of a frame's origin."""
    return fetch_buildings(frame.bbox_for_radius(radius_m), **kwargs)


# HTTP codes that mean "the service is busy", as opposed to "your query is
# wrong". Only these are retried; a 400 will fail identically forever.
_TRANSIENT_CODES = {429, 502, 503, 504}


def fetch_buildings(
    bbox: tuple[float, float, float, float],
    *,
    timeout: float = 180.0,
    include_parts: bool = False,
    retries: int = 2,
    backoff_s: float = 5.0,
) -> list[Building]:
    """
    Run the query and parse the result.

    Overpass is a free, shared, heavily rate-limited service, and transient
    overload is routine rather than exceptional — during one afternoon's work it
    returned 504 twice, and on both occasions the very next attempt succeeded.
    So busy responses are retried with a linear backoff before giving up.

    Retries are deliberately bounded and only cover :data:`_TRANSIENT_CODES`. A
    malformed query returns 400 and would fail identically forever; retrying it
    just turns a clear error into a slow one.

    A failure still raises. "Overpass is busy" must never degrade into an empty
    building list, because an empty list renders as a correct picture of an
    empty field and there is nothing downstream to say otherwise.
    """
    query = build_query(bbox, timeout_s=int(timeout), include_parts=include_parts)
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            OVERPASS_URL,
            data=data,
            headers={"User-Agent": "atlas-scene/0.1 (+https://github.com/ssnfts/atlas)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return parse_overpass(payload, include_parts=include_parts)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in _TRANSIENT_CODES:
                raise OverpassError(f"Overpass returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        except json.JSONDecodeError as exc:
            raise OverpassError("Overpass returned a non-JSON body") from exc

        if attempt < retries:
            _sleep(backoff_s * (attempt + 1))

    code = getattr(last_error, "code", None)
    detail = f"HTTP {code}" if code else str(last_error)
    raise OverpassError(
        f"Overpass is rate-limiting or overloaded ({detail}) after "
        f"{retries + 1} attempts. It is a free shared service — wait and retry, "
        "shrink the radius, or point ATLAS_OVERPASS_URL at your own instance "
        "or a mirror."
    ) from last_error


def _sleep(seconds: float) -> None:
    """Indirection so the retry backoff can be silenced in tests."""
    import time

    time.sleep(seconds)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_overpass(payload: dict, *, include_parts: bool = False) -> list[Building]:
    """
    Overpass JSON -> :class:`Building` list. Pure; no network.

    Elements that cannot yield a usable ring are skipped rather than raising.
    One malformed multipolygon in a thousand-building city block should cost
    that building, not the scene.
    """
    buildings: list[Building] = []

    for element in payload.get("elements") or []:
        kind = element.get("type")
        tags = {str(k): str(v) for k, v in (element.get("tags") or {}).items()}

        if not _is_building(tags, include_parts=include_parts):
            continue

        try:
            if kind == "way":
                parsed = _parse_way(element, tags)
            elif kind == "relation":
                parsed = _parse_relation(element, tags)
            else:
                continue
        except (ValueError, TypeError, KeyError):
            continue

        if parsed is not None:
            buildings.append(parsed)

    return buildings


def _is_building(tags: dict, *, include_parts: bool) -> bool:
    """
    Filter to things that are actually buildings.

    ``building=no`` is an explicit assertion that a footprint is *not* a
    building — commonly used to carve an exception out of a larger area — and
    truthiness alone would happily extrude it.
    """
    value = tags.get("building")
    if value and value.lower() not in ("no", "false"):
        return True
    if include_parts:
        part = tags.get("building:part")
        if part and part.lower() not in ("no", "false"):
            return True
    return False


def _parse_way(element: dict, tags: dict) -> Building | None:
    ring = _ring_from_geometry(element.get("geometry"))
    if ring is None:
        return None

    building = Building(
        osm_id=int(element.get("id", 0)),
        osm_type="way",
        outer=ring,
        tags=tags,
    )
    return building


def _parse_relation(element: dict, tags: dict) -> Building | None:
    """
    A multipolygon relation: outer ring(s) plus holes.

    The trap here is that a role is attached to a **way segment, not a ring**.
    A courtyard block's outer boundary is routinely mapped as four or five
    separate ways that share endpoints, because segments get split wherever a
    street name or surface changes. Treating each member as a finished ring
    yields open polylines that triangulate into shrapnel — so members are
    stitched by matching endpoints first (see :func:`assemble_rings`).

    Where a relation has several disjoint outer rings, the largest is kept and
    the rest dropped: this module's contract is one footprint per Building, and
    a scene with the biggest piece present beats one with a randomly chosen
    piece present.
    """
    outer_segments: list[list[tuple[float, float]]] = []
    inner_segments: list[list[tuple[float, float]]] = []

    for member in element.get("members") or []:
        if member.get("type") != "way":
            continue
        points = _points_from_geometry(member.get("geometry"))
        if len(points) < 2:
            continue
        # An empty role on a multipolygon member conventionally means outer.
        role = str(member.get("role") or "outer").lower()
        if role == "inner":
            inner_segments.append(points)
        elif role in ("outer", ""):
            outer_segments.append(points)

    outer_rings = assemble_rings(outer_segments)
    if not outer_rings:
        return None

    outer = max(outer_rings, key=_abs_ring_area)
    inners = [r for r in assemble_rings(inner_segments) if len(r) >= 3]

    building = Building(
        osm_id=int(element.get("id", 0)),
        osm_type="relation",
        outer=outer,
        inners=inners,
        tags=tags,
    )
    return building


def _points_from_geometry(geometry) -> list[tuple[float, float]]:
    """Overpass ``geometry`` array -> (lat, lon) list, dropping null members."""
    if not geometry:
        return []
    points = []
    for node in geometry:
        # Overpass emits null entries for nodes outside the query bbox on a
        # partially-covered way.
        if not node or node.get("lat") is None or node.get("lon") is None:
            continue
        points.append((float(node["lat"]), float(node["lon"])))
    return points


def _ring_from_geometry(geometry) -> list[tuple[float, float]] | None:
    """A closed way's geometry -> an open ring, or None if it is not usable."""
    points = _points_from_geometry(geometry)
    ring = _open_ring(points)
    return ring if len(ring) >= 3 else None


def _open_ring(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    Drop the repeated closing vertex, and any consecutive duplicates.

    A closed OSM way repeats its first node as its last. Carrying that through
    creates a zero-length edge, and a zero-length edge has no direction — ear
    clipping computes a cross product of zero there and mis-classifies the
    corner as reflex, so the roof cap silently comes out with missing triangles.
    """
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if not cleaned or not _same_point(cleaned[-1], point):
            cleaned.append(point)
    while len(cleaned) >= 2 and _same_point(cleaned[0], cleaned[-1]):
        cleaned.pop()
    return cleaned


def _same_point(a: tuple[float, float], b: tuple[float, float]) -> bool:
    # ~1e-9 degrees is about 0.1 mm; anything closer is the same node.
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def assemble_rings(
    segments: list[list[tuple[float, float]]]
) -> list[list[tuple[float, float]]]:
    """
    Stitch way segments into closed rings by matching endpoints.

    Multipolygon members arrive as unordered, arbitrarily-directed polylines.
    This walks them into rings, reversing segments as needed, and returns only
    those that actually close. Segments that lead nowhere are dropped — an
    unclosed boundary is a mapping error upstream and cannot be extruded.

    Already-closed segments (the common single-way case) pass straight through.
    """
    pool = [list(s) for s in segments if len(s) >= 2]
    rings: list[list[tuple[float, float]]] = []

    while pool:
        chain = pool.pop(0)

        # A segment that is already a closed loop needs no stitching.
        if _same_point(chain[0], chain[-1]):
            ring = _open_ring(chain)
            if len(ring) >= 3:
                rings.append(ring)
            continue

        extended = True
        while extended and not _same_point(chain[0], chain[-1]):
            extended = False
            for i, candidate in enumerate(pool):
                if _same_point(chain[-1], candidate[0]):
                    chain += candidate[1:]
                elif _same_point(chain[-1], candidate[-1]):
                    chain += list(reversed(candidate))[1:]
                elif _same_point(chain[0], candidate[-1]):
                    chain = candidate[:-1] + chain
                elif _same_point(chain[0], candidate[0]):
                    chain = list(reversed(candidate))[:-1] + chain
                else:
                    continue
                pool.pop(i)
                extended = True
                break

        if _same_point(chain[0], chain[-1]):
            ring = _open_ring(chain)
            if len(ring) >= 3:
                rings.append(ring)

    return rings


def _abs_ring_area(ring: list[tuple[float, float]]) -> float:
    """
    Unsigned shoelace area in degrees squared — for *comparing* rings only.

    Fine for picking the largest of several rings at one location. Not a real
    area: a degree of longitude is not a degree of latitude anywhere but the
    equator. Real areas are computed after projection into scene metres.
    """
    total = 0.0
    count = len(ring)
    for i in range(count):
        lat1, lon1 = ring[i]
        lat2, lon2 = ring[(i + 1) % count]
        total += lon1 * lat2 - lon2 * lat1
    return abs(total) * 0.5


# ── Heights ───────────────────────────────────────────────────────────────────

# Accepts "12", "12.5", "12 m", "12m", "40 ft", "40'", "12'6\"" and the European
# decimal comma. OSM's wiki says metres and bare numbers, and the data says
# otherwise often enough to matter.
_HEIGHT_RE = re.compile(
    r"""^\s*
        (?P<value>[-+]?\d+(?:[.,]\d+)?)
        \s*
        (?P<unit>m|metre|metres|meter|meters|ft|feet|foot|'|″|")?
        \s*
        (?:(?P<inches>\d+(?:[.,]\d+)?)\s*(?:''|"|″|in|inch|inches)\s*)?
        $""",
    re.IGNORECASE | re.VERBOSE,
)

_FEET_UNITS = {"ft", "feet", "foot", "'"}
_FEET_PER_METRE = 0.3048


def _parse_length_m(raw: str | None) -> float | None:
    """
    An OSM length tag -> metres, or None if it is not parseable.

    Returns None rather than a guess. A silently-defaulted height is the exact
    failure mode this project keeps hitting: no error, and a skyline that is
    plausible but not the one that was there.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    match = _HEIGHT_RE.match(text)
    if match is None:
        return None

    try:
        value = float(match.group("value").replace(",", "."))
    except ValueError:
        return None

    unit = (match.group("unit") or "").lower()
    inches_raw = match.group("inches")

    if unit in _FEET_UNITS:
        metres = value * _FEET_PER_METRE
        if inches_raw:
            metres += float(inches_raw.replace(",", ".")) * _FEET_PER_METRE / 12.0
        return metres

    # A bare number, or an explicit metric unit. Feet-and-inches without a foot
    # marker ("12 6") is not valid OSM and is not guessed at here.
    if inches_raw:
        return None
    return value


def _parse_levels(raw: str | None) -> float | None:
    """
    ``building:levels`` -> storey count.

    Semicolons appear where a mapper recorded alternatives or a range
    ("3;4"); the first value is taken. Half-levels ("2.5") are real and legal.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if ";" in text:
        text = text.split(";", 1)[0].strip()
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def resolve_height(tags: dict) -> tuple[float, float, str]:
    """
    Tags -> ``(height_m, min_height_m, source)``.

    Precedence follows how much the mapper actually knew:

    1. ``height`` — a measured or published figure.
    2. ``building:levels`` × :data:`METRES_PER_LEVEL` — a count, inflated by a
       constant. Adds ``roof:height`` when tagged, since the level count
       describes the occupied storeys and excludes the roof structure.
    3. :data:`DEFAULT_HEIGHT_M` — nothing was known.

    ``source`` is returned, not logged and discarded, because everything
    downstream wants it: an artist deciding what to replace with real geometry,
    and the manifest recording what was inferred rather than sourced.

    ``min_height`` lifts a volume off the ground. It is what makes an archway
    or an overhanging upper storey read correctly on a detailed building, and
    ignoring it drops those volumes to street level where they block roads.
    """
    height = _parse_length_m(tags.get("height"))
    if height is not None and height > 0:
        source = "height tag"
    else:
        levels = _parse_levels(tags.get("building:levels"))
        if levels is not None and levels > 0:
            height = levels * METRES_PER_LEVEL
            source = f"building:levels={_trim(levels)} x {METRES_PER_LEVEL}m"
            roof = _parse_length_m(tags.get("roof:height"))
            if roof is not None and roof > 0:
                height += roof
                source += f" + roof:height={_trim(roof)}m"
        else:
            height = DEFAULT_HEIGHT_M
            source = "default (no height or level tags)"

    min_height = _parse_length_m(tags.get("min_height"))
    if min_height is None:
        min_level = _parse_levels(tags.get("building:min_level"))
        min_height = min_level * METRES_PER_LEVEL if min_level else 0.0
    min_height = max(0.0, min_height)

    # A volume whose base is above its top is inside-out. Seen in the wild where
    # min_height was tagged in feet and height in metres.
    if min_height >= height:
        min_height = 0.0
        source += " (min_height ignored: above the roof)"

    if not math.isfinite(height) or height <= 0:
        height, source = DEFAULT_HEIGHT_M, "default (unusable height)"

    return round(height, 3), round(min_height, 3), source


def _trim(value: float) -> str:
    """Format a float without a trailing ``.0``, for readable source strings."""
    return f"{value:g}"
