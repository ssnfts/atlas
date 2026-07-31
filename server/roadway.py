"""
Linear OSM ways -> flat ribbon meshes in the scene frame.

Where :mod:`massing` turns closed *polygons* (building footprints) into solids,
this turns open *polylines* (roads, race tracks, runways — anything tagged as a
linear way) into a strip of asphalt laid on the ground. It is the second
geometry primitive the project needs: a great deal of what reads as "a place"
in a render is the road surface, and a race circuit is nothing else.

The failure class is the same one that haunts :mod:`massing`: none of the ways
this can go wrong *raise*. A ribbon offset the wrong distance is still a ribbon.
A strip wound so its normals point down renders as a black hole in the ground
under a low sun and as nothing at all under a high one. A mitre spike at a sharp
corner is a plausible-looking triangle that happens to stab across the track. So
the arithmetic is pinned to invariants a self-consistent bug cannot satisfy:

- **Centre-line length is conserved.** The ribbon's spine must be exactly as
  long as the polyline it was built from. For the Yas Marina circuit that spine
  is 5.28 km, a number known independently of this code, so a length that comes
  out wrong is caught against the real world and not just against itself.
- **Area is conserved.** A ribbon of length L and width W has area ~= L*W, off
  only by the small triangles gained and lost at corners. ``shapely`` computes
  the same buffer independently in the tests; the two areas must agree. A
  doubled width, a self-overlap, or a dropped section all break this.
- **Winding is explicit.** Every triangle is wound counter-clockwise seen from
  above, so its normal points +Z. That is a computed guarantee, not a hope that
  the OSM way happened to be digitised in a convenient direction.

**A road is a graded surface, not a drape.** The elevation is sampled once per
centre-line vertex and then low-passed along arc length by ``smooth_m``, because
a machine laid this surface and it does not reproduce every bump in a terrain
model. Draping Yas Marina straight onto Copernicus GLO-30 produced a **38.9%
gradient** — twice Eau Rouge, on a flat reclaimed island — purely from DEM noise
at 30 m posting being read across spine segments as short as 2 m. A window wider
than the route levels it completely, which for a circuit built on reclaimed land
is the physically right answer; the resulting height is still the DEM's own mean,
not an invented number. ``max_gradient_pct`` and its raw counterpart both land in
the mesh metadata so the choice is visible rather than implied.

Dependency-free on purpose, exactly like :mod:`massing`, :mod:`solar` and
:mod:`frame`. ``shapely`` would shorten the offsetting to one ``buffer`` call,
but a buffer returns a *polygon* — it merges the two sides of a hairpin where
the track doubles back on itself, and it throws away the along-track
parameterisation that a strip of quads keeps for free (and that UVs, kerbs and a
racing line all want later). So the strip is built by hand and ``shapely`` is
kept for the test that proves the strip right.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from frame import SceneFrame
from massing import Mesh

__all__ = [
    "Centerline",
    "Path",
    "RoadwayError",
    "OverpassError",
    "RACEWAY_SELECTORS",
    "DEFAULT_WIDTHS",
    "build_way_query",
    "parse_ways",
    "fetch_ways",
    "stitch_paths",
    "offset_ribbon",
    "ribbon_mesh",
    "centerlines_to_meshes",
]

OVERPASS_URL = os.environ.get(
    "ATLAS_OVERPASS_URL", "https://overpass-api.de/api/interpreter"
)

# Rings/points closer than this in the plan are the same node. 1 mm is far below
# OSM's metre-scale accuracy and far above the float noise of projecting degrees
# to metres — the same threshold massing/osm use.
EPS = 1e-6

# Width fallbacks per kind of way, in metres, when OSM carries no ``width`` tag.
# These are load-bearing for how the surface reads: too wide and a circuit's
# corners merge into a car park, too narrow and the racing line looks like a
# footpath. A modern F1 track runs 12-16 m; 15 m is a fair single value for the
# whole lap. The pit lane (fast lane plus the box side) is narrower.
DEFAULT_WIDTHS = {
    "raceway": 15.0,
    "pit_lane": 12.0,
    "road": 7.0,
    "default": 6.0,
}

# Overpass selectors for the linear features this module understands. Kept
# explicit rather than "every highway", because pulling an entire city's street
# grid is a different, much larger job than laying a single circuit.
RACEWAY_SELECTORS = ('way["highway"="raceway"]',)


class RoadwayError(ValueError):
    """A way could not be turned into sound ribbon geometry."""


class OverpassError(RuntimeError):
    """Overpass could not be reached, or returned something unusable."""


# ── Centre-line model ─────────────────────────────────────────────────────────

@dataclass
class Centerline:
    """
    One linear OSM way, still in geodetic coordinates.

    ``points`` are ``(lat, lon)`` in the order the way was digitised. Unlike a
    building ring, a way is **not** implicitly closed: a cul-de-sac loop repeats
    its first node as its last and an ordinary street does not, and both are
    legal. :attr:`is_closed` reports which, rather than assuming.
    """

    osm_id: int
    osm_type: str  # "way"
    points: list[tuple[float, float]]
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str | None:
        return self.tags.get("name")

    @property
    def is_closed(self) -> bool:
        pts = self.points
        return len(pts) >= 3 and _same(pts[0], pts[-1])

    def width_m(self, default: float) -> float:
        """
        Surface width in metres.

        Prefers an OSM ``width`` (or ``est_width``) tag when present, because a
        mapped width beats any default; falls back to the caller's default
        otherwise. Reuses no clever parser — road widths in OSM are plain metres
        far more consistently than building heights are, but a stray unit or a
        range still falls back rather than guessing wrong.
        """
        for key in ("width", "est_width"):
            raw = self.tags.get(key)
            if raw is None:
                continue
            text = str(raw).strip().replace(",", ".")
            # A range "12;14" or "12-14": take the first value.
            for sep in (";", "-"):
                if sep in text:
                    text = text.split(sep, 1)[0].strip()
            try:
                value = float(text.split()[0]) if text else None
            except (ValueError, IndexError):
                value = None
            if value is not None and 0.5 <= value <= 60.0:
                return value
        return default


@dataclass
class Path:
    """
    A maximal chain of ways stitched end to end.

    A circuit is mapped as several ways that meet at shared nodes — the Yas
    Marina lap is six of them — so the ribbon has to be built from the joined
    spine, not way by way, or every join gets a flat end-cap and a seam. ``xy``
    is already in scene metres; ``closed`` says whether the spine returns to its
    start, which decides whether the ribbon wraps or gets end caps.
    """

    xy: list[tuple[float, float]]
    closed: bool
    tags: dict[str, str] = field(default_factory=dict)
    osm_ids: list[int] = field(default_factory=list)

    @property
    def name(self) -> str | None:
        return self.tags.get("name")


# ── Query & parse ─────────────────────────────────────────────────────────────

def build_way_query(
    bbox: tuple[float, float, float, float],
    selectors: tuple[str, ...] = RACEWAY_SELECTORS,
    *,
    timeout_s: int = 120,
) -> str:
    """
    Overpass QL for every matching way in ``bbox``.

    ``bbox`` is ``(south, west, north, east)`` — Overpass's own order and the
    one :meth:`frame.SceneFrame.bbox_for_radius` returns. The latitude sanity
    check catches a transposition only when it produces an impossible latitude;
    inside the valid band it cannot, which is why callers should route through
    :func:`fetch_for_site` and never assemble the tuple by hand.
    """
    south, west, north, east = bbox
    if not (-90.0 <= south <= north <= 90.0):
        raise ValueError(
            f"bad latitude range: south={south}, north={north}. "
            "bbox is (south, west, north, east)."
        )
    if not (-180.0 <= west <= 180.0 and -180.0 <= east <= 180.0):
        raise ValueError(f"bad longitude range: west={west}, east={east}")

    area = f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}"
    body = "\n  ".join(f"{sel}({area});" for sel in selectors)
    return f"[out:json][timeout:{timeout_s}];\n(\n  {body}\n);\nout geom;"


def parse_ways(payload: dict, *, keep=None) -> list[Centerline]:
    """
    Overpass JSON -> :class:`Centerline` list. Pure; no network.

    ``keep`` is an optional ``tags -> bool`` predicate, so a caller can pull "the
    Grand Prix circuit" rather than "every raceway including the kart track and
    the rallycross layout" without a second query. A way that yields fewer than
    two usable points is dropped rather than raising — one bad way should cost
    that way, not the circuit.
    """
    out: list[Centerline] = []
    for element in payload.get("elements") or []:
        if element.get("type") != "way":
            continue
        tags = {str(k): str(v) for k, v in (element.get("tags") or {}).items()}
        if keep is not None and not keep(tags):
            continue
        points = _points_from_geometry(element.get("geometry"))
        if len(points) < 2:
            continue
        out.append(
            Centerline(
                osm_id=int(element.get("id", 0)),
                osm_type="way",
                points=points,
                tags=tags,
            )
        )
    return out


def _points_from_geometry(geometry) -> list[tuple[float, float]]:
    """Overpass ``geometry`` array -> (lat, lon) list, dropping null members and
    consecutive duplicates (a repeated node makes a zero-length segment whose
    direction is undefined, which the offset maths cannot normalise)."""
    if not geometry:
        return []
    pts: list[tuple[float, float]] = []
    for node in geometry:
        if not node or node.get("lat") is None or node.get("lon") is None:
            continue
        p = (float(node["lat"]), float(node["lon"]))
        if not pts or not _same(pts[-1], p):
            pts.append(p)
    return pts


_TRANSIENT_CODES = {429, 502, 503, 504}


def fetch_ways(
    bbox: tuple[float, float, float, float],
    *,
    selectors: tuple[str, ...] = RACEWAY_SELECTORS,
    keep=None,
    timeout: float = 120.0,
    retries: int = 2,
    backoff_s: float = 5.0,
) -> list[Centerline]:
    """
    Run the way query and parse it, retrying only transient overload.

    Mirrors :func:`osm.fetch_buildings`: a busy Overpass (429/502/503/504) is
    retried with a linear backoff, a malformed query (400) fails immediately
    because it would fail identically forever, and any failure raises rather
    than degrading to an empty list — an empty list renders as a bare field with
    nothing downstream to say the track went missing.
    """
    query = build_way_query(bbox, selectors, timeout_s=int(timeout))
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
            return parse_ways(payload, keep=keep)
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
        f"{retries + 1} attempts."
    )


def fetch_for_site(
    frame: SceneFrame, radius_m: float, **kwargs
) -> list[Centerline]:
    """Fetch every matching way within ``radius_m`` of a frame's origin."""
    return fetch_ways(frame.bbox_for_radius(radius_m), **kwargs)


def _sleep(seconds: float) -> None:
    """Indirection so the retry backoff can be silenced in tests."""
    import time

    time.sleep(seconds)


# ── Stitching ─────────────────────────────────────────────────────────────────

def stitch_paths(
    centerlines: list[Centerline], frame: SceneFrame
) -> list[Path]:
    """
    Join ways that meet at shared endpoints into maximal :class:`Path` chains.

    A route mapped as several ways (the six segments of the Yas lap) has to be
    walked into one spine, reversing ways as needed, before it is offset — offset
    each way alone and every join gets a flat cap and a visible seam across the
    track.

    Endpoint matching is done in **scene metres**, not degrees: a shared node is
    bit-identical in the source, but projecting first keeps the whole pipeline in
    one coordinate space and the 1 mm threshold physically meaningful. Ways that
    connect nowhere come back as their own single-way paths.
    """
    # Each way as a projected polyline.
    pool: list[dict] = []
    for cl in centerlines:
        xy = [frame.to_scene(lat, lon)[:2] for lat, lon in cl.points]
        pool.append({"xy": xy, "tags": cl.tags, "id": cl.osm_id})

    paths: list[Path] = []
    while pool:
        head = pool.pop(0)
        chain = list(head["xy"])
        ids = [head["id"]]
        tags = dict(head["tags"])

        extended = True
        while extended and not _same(chain[0], chain[-1]):
            extended = False
            for i, cand in enumerate(pool):
                cxy = cand["xy"]
                if _same(chain[-1], cxy[0]):
                    chain += cxy[1:]
                elif _same(chain[-1], cxy[-1]):
                    chain += list(reversed(cxy))[1:]
                elif _same(chain[0], cxy[-1]):
                    chain = cxy[:-1] + chain
                elif _same(chain[0], cxy[0]):
                    chain = list(reversed(cxy))[:-1] + chain
                else:
                    continue
                ids.append(cand["id"])
                # Keep a name if any member carries one.
                for k, v in cand["tags"].items():
                    tags.setdefault(k, v)
                pool.pop(i)
                extended = True
                break

        closed = len(chain) >= 4 and _same(chain[0], chain[-1])
        paths.append(Path(xy=chain, closed=closed, tags=tags, osm_ids=ids))

    return paths


# ── Ribbon geometry (pure, scene metres) ──────────────────────────────────────

def offset_ribbon(
    polyline: list[tuple[float, float]],
    width: float,
    *,
    closed: bool = False,
    mitre_limit: float = 4.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Offset a centre-line into left and right edge rows, each in metres.

    "Left" is 90 deg counter-clockwise of the direction of travel. At each vertex
    the two adjacent segment normals are averaged into a **mitre** and scaled by
    ``1 / cos(half-turn)`` so the edge stays a constant ``width/2`` from the
    spine through the bend. Where a bend is sharp enough that the mitre would
    shoot past ``mitre_limit`` half-widths — a hairpin, a switchback — the vertex
    is **bevelled** instead: two edge points are emitted, one square to each
    segment, so the corner is cut off rather than allowed to grow a spike that
    stabs back across the surface. Both rows always gain the same number of
    points, so they stay in lock-step for the strip that consumes them.

    Real circuits sampled at OSM density rarely reach the bevel — the Yas lap's
    sharpest vertex turns 20 deg, a mitre scale of 1.02 — but roads and tighter
    tracks do, and a spike is exactly the silent, plausible-looking corruption
    this project exists to refuse.

    Returns ``(left_row, right_row, owners)``. The rows have one entry per
    emitted cross-section and are cyclic for a closed loop (no duplicate closing
    vertex). ``owners`` maps each cross-section back to the centre-line vertex
    that produced it — a bevelled corner emits two cross-sections from one
    vertex, so the mapping is not the identity, and callers that need a
    per-vertex quantity (an elevation profile, a distance along the lap) cannot
    reconstruct it without this.
    """
    pts = [tuple(p) for p in polyline]
    if closed and len(pts) >= 2 and _same(pts[0], pts[-1]):
        pts = pts[:-1]  # cyclic maths owns the wrap; drop the repeat
    if len(pts) < 2:
        raise RoadwayError("a ribbon needs at least two distinct points")

    n = len(pts)
    half = width / 2.0

    # Left-hand unit normal of each segment j (from pts[j] to pts[j+1]).
    seg_normals: list[tuple[float, float] | None] = []
    seg_count = n if closed else n - 1
    for j in range(seg_count):
        a = pts[j]
        b = pts[(j + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < EPS:
            seg_normals.append(None)
        else:
            seg_normals.append((-dy / length, dx / length))

    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    owners: list[int] = []

    def incoming(i: int):
        return seg_normals[(i - 1) % n] if closed else (seg_normals[i - 1] if i > 0 else None)

    def outgoing(i: int):
        return seg_normals[i % n] if closed else (seg_normals[i] if i < n - 1 else None)

    for i in range(n):
        n_in = incoming(i)
        n_out = outgoing(i)

        # Endpoints of an open path: square end, use the one segment's normal.
        if n_in is None:
            n_in = n_out
        if n_out is None:
            n_out = n_in
        if n_in is None or n_out is None:
            raise RoadwayError("degenerate segment left a vertex with no normal")

        mx, my = n_in[0] + n_out[0], n_in[1] + n_out[1]
        mlen = math.hypot(mx, my)
        px, py = pts[i]

        if mlen < EPS:
            # A 180 deg reversal at one vertex: the averaged normal vanishes.
            # Bevel across it using each segment's own normal.
            left += [(px + n_in[0] * half, py + n_in[1] * half),
                     (px + n_out[0] * half, py + n_out[1] * half)]
            right += [(px - n_in[0] * half, py - n_in[1] * half),
                      (px - n_out[0] * half, py - n_out[1] * half)]
            owners += [i, i]
            continue

        mhx, mhy = mx / mlen, my / mlen
        cos_half = mhx * n_in[0] + mhy * n_in[1]  # = cos(half turn angle)
        scale = 1.0 / cos_half if cos_half > EPS else mitre_limit + 1.0

        if scale <= mitre_limit:
            ox, oy = mhx * half * scale, mhy * half * scale
            left.append((px + ox, py + oy))
            right.append((px - ox, py - oy))
            owners.append(i)
        else:
            # Too sharp to mitre: bevel the corner with a point square to each
            # adjacent segment on both sides, keeping the rows the same length.
            left += [(px + n_in[0] * half, py + n_in[1] * half),
                     (px + n_out[0] * half, py + n_out[1] * half)]
            right += [(px - n_in[0] * half, py - n_in[1] * half),
                      (px - n_out[0] * half, py - n_out[1] * half)]
            owners += [i, i]

    return left, right, owners


def ribbon_mesh(
    path: Path,
    *,
    width: float,
    name: str = "roadway",
    frame: SceneFrame | None = None,
    ground=None,
    lift: float = 0.05,
    smooth_m: float = 0.0,
    mitre_limit: float = 4.0,
) -> Mesh:
    """
    Turn one :class:`Path` into a flat ribbon :class:`Mesh` in scene metres.

    ``ground`` is an optional elevation probe taking **geodetic** coordinates —
    ``ground(lat, lon)``, the signature of :meth:`terrain.TerrainPatch.elevation_at`
    and the one :func:`massing.ground_under` already uses. The ribbon's own
    vertices are in scene metres, so ``frame`` is required alongside it to invert
    them; passing ``ground`` without ``frame`` raises rather than guessing.

    That combination is worth spelling out because getting it wrong is silent and
    total. The first version of this function called ``ground(x, y)`` with scene
    metres straight out of the offset rows. Every call raised — a scene easting
    of -161 is not a latitude — a blanket ``except`` swallowed it, and the whole
    5 km circuit came back pinned flat at ``lift`` while the terrain around it
    ranged over 41 m. Nothing errored. The track was simply buried, and the
    render came out as a city with no circuit in it. ``terrain.elevation_at``
    refuses to guess for exactly this reason, and the old code overrode that
    refusal from the outside.

    So failures are counted now, not swallowed: a vertex outside the patch
    inherits the last good height (a circuit may run a little past the DEM tile,
    and one edge vertex should not sink the lap), but a ribbon where *nothing*
    sampled raises :class:`RoadwayError`, and the miss count lands in
    ``metadata``.

    ``lift`` is a few centimetres so the surface rests *just* above the terrain
    sheet rather than z-fighting it — coincident coplanar faces flicker between
    the two materials pixel by pixel, which reads as a rendering fault rather
    than a modelling one.

    Faces are wound counter-clockwise seen from above, so normals point +Z. The
    returned mesh carries its measured length, width and area in ``metadata`` for
    the caller (and the tests) to check against the world.
    """
    if ground is not None and frame is None:
        raise RoadwayError(
            "ribbon_mesh(ground=...) needs frame= as well: the probe takes "
            "geodetic (lat, lon) but ribbon vertices are scene metres, so the "
            "frame is what inverts them. Without it every sample would miss."
        )
    left, right, owners = offset_ribbon(
        path.xy, width, closed=path.closed, mitre_limit=mitre_limit
    )
    m = len(left)
    if m != len(right) or m < 2:
        raise RoadwayError(f"ribbon rows malformed: {len(left)} vs {len(right)}")

    spine = path.xy[:-1] if (path.closed and len(path.xy) >= 2
                             and _same(path.xy[0], path.xy[-1])) else path.xy

    # Elevation is sampled **once per centre-line vertex**, not once per edge
    # vertex. Sampling the two edges independently lets a metre of DEM noise
    # land on one side and not the other, which banks the surface at random —
    # a road that rolls left and right every few metres for no reason.
    profile, misses = _sample_profile(spine, frame, ground)

    if ground is not None and profile is None:
        raise RoadwayError(
            f"no part of {name} falls inside the terrain patch, so its ground "
            f"level is unknown ({misses} samples, all outside). Widen the "
            "terrain radius rather than defaulting the surface to zero."
        )

    raw_grade = _max_gradient(spine, profile, path.closed) if profile else 0.0
    if profile is not None and smooth_m > 0.0:
        profile = _smooth_profile(spine, profile, smooth_m, path.closed)
    graded = _max_gradient(spine, profile, path.closed) if profile else 0.0

    def z_of(k: int) -> float:
        if profile is None:
            return lift
        return profile[owners[k]] + lift

    # Vertices: left row [0..m-1] then right row [m..2m-1].
    verts: list[tuple[float, float, float]] = []
    for k, (x, y) in enumerate(left):
        verts.append((x, y, z_of(k)))
    for k, (x, y) in enumerate(right):
        verts.append((x, y, z_of(k)))

    # Two triangles per quad. On a bend tighter than the half-width the *inner*
    # edge offsets past itself and folds one triangle back — its normal points
    # down, and a down-facing triangle on the ground reads as a black shard under
    # a low sun. Those folds are dropped here, by the same +Z test the winding
    # invariant is checked with, so the invariant holds by construction rather
    # than by hope. The count is recorded, not swallowed: a ribbon that quietly
    # sheds a third of its faces is a wrong width or a corrupt spine, and the
    # number is how a caller sees that.
    faces: list[tuple[int, int, int]] = []
    dropped = 0
    span = m if path.closed else m - 1
    for k in range(span):
        k1 = (k + 1) % m
        lk, rk = k, m + k
        lk1, rk1 = k1, m + k1
        for tri in ((lk, rk, lk1), (rk, rk1, lk1)):
            if _faces_up(verts, tri):
                faces.append(tri)
            else:
                dropped += 1

    mesh = Mesh(verts=verts, faces=faces, name=name)
    mesh.metadata.update(
        {
            "kind": "roadway",
            "osm_ids": path.osm_ids,
            "closed": path.closed,
            "width_m": round(width, 3),
            "centerline_length_m": round(_polyline_length(path.xy, path.closed), 2),
            "surface_area_m2": round(_ribbon_area(left, right, path.closed), 2),
            "cross_sections": m,
            "faces_dropped_at_folds": dropped,
            "ground_samples_missed": misses,
            "smoothing_window_m": round(smooth_m, 1),
            "max_gradient_pct": round(graded * 100.0, 2),
            "max_gradient_pct_raw": round(raw_grade * 100.0, 2),
        }
    )
    return mesh


def _sample_profile(spine, frame, ground):
    """
    Elevation at each centre-line vertex. Returns ``(profile, misses)``.

    ``profile`` is None when ``ground`` is None (a flat ribbon) or when every
    sample fell outside the patch — the caller distinguishes the two. A vertex
    that misses inherits the previous good height, so a lap that strays a little
    past the DEM tile keeps going; the count comes back so the caller can see it
    happened rather than discovering it in a render.
    """
    if ground is None:
        return None, 0

    profile: list[float | None] = []
    misses = 0
    last_good: float | None = None
    first_good: int | None = None

    for x, y in spine:
        lat, lon, _ = frame.to_geodetic(x, y)
        try:
            elevation = float(ground(lat, lon))
        except Exception:
            misses += 1
            profile.append(last_good)  # None until the first success
            continue
        if first_good is None:
            first_good = len(profile)
        last_good = elevation
        profile.append(elevation)

    if first_good is None:
        return None, misses

    # Vertices before the first success have nothing to inherit forwards, so
    # they take the first good height instead. Left as None they would be a
    # silent hole; set to 0.0 they would read as sea level, which terrain.py
    # refuses to do for exactly this reason.
    head = profile[first_good]
    return [head if z is None else z for z in profile], misses


def _smooth_profile(spine, profile, window_m: float, closed: bool):
    """
    Low-pass the elevation profile along arc length.

    A road is a *graded* surface: it follows the ground's trend, but a machine
    laid it and it does not reproduce every bump in the terrain model. A DEM at
    30 m posting carries noise of a metre or two, and draping a ribbon straight
    onto it produced a 27% gradient on a circuit whose real maximum is nearer 3%
    — steeper than Eau Rouge, on reclaimed flat land.

    Averaging over a window of arc length removes that while preserving real
    relief: a genuinely hilly circuit keeps its hills, because they are far
    longer than the window. Wrapping is cyclic for a closed lap so the
    start/finish line is not a step.
    """
    n = len(profile)
    if n < 3 or window_m <= 0.0:
        return profile

    # Cumulative arc length at each vertex.
    s = [0.0]
    for i in range(n - 1):
        s.append(s[-1] + math.hypot(spine[i + 1][0] - spine[i][0],
                                    spine[i + 1][1] - spine[i][1]))
    total = s[-1]
    if closed and total > 0:
        total += math.hypot(spine[0][0] - spine[-1][0],
                            spine[0][1] - spine[-1][1])

    half = window_m / 2.0
    out: list[float] = []
    for i in range(n):
        acc = 0.0
        count = 0
        for j in range(n):
            d = abs(s[j] - s[i])
            if closed and total > 0:
                d = min(d, total - d)
            if d <= half:
                acc += profile[j]
                count += 1
        out.append(acc / count if count else profile[i])
    return out


def _max_gradient(spine, profile, closed: bool) -> float:
    """Steepest rise over run between adjacent centre-line vertices, as a
    fraction. The physical sanity check on a draped surface: a road that comes
    out at 27% is not a road."""
    if not profile:
        return 0.0
    n = len(spine)
    span = n if closed else n - 1
    worst = 0.0
    for i in range(span):
        j = (i + 1) % n
        run = math.hypot(spine[j][0] - spine[i][0], spine[j][1] - spine[i][1])
        if run < EPS:
            continue
        worst = max(worst, abs(profile[j] - profile[i]) / run)
    return worst


def _faces_up(verts, tri) -> bool:
    """True when triangle ``tri`` winds counter-clockwise seen from above, i.e.
    its normal has a strictly positive Z. The cross product's Z component is
    computed in the XY plane, so a fold-back (inner-corner overlap) or a
    zero-area sliver is rejected."""
    i, j, k = tri
    ax, ay, _ = verts[i]
    bx, by, _ = verts[j]
    cx, cy, _ = verts[k]
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > EPS


def centerlines_to_meshes(
    centerlines: list[Centerline],
    frame: SceneFrame,
    *,
    width: float,
    name_prefix: str = "roadway",
    ground=None,
    lift: float = 0.05,
    smooth_m: float = 0.0,
) -> tuple[list[Mesh], list[dict]]:
    """
    Stitch, offset and mesh a set of ways into ribbon :class:`Mesh` objects.

    One mesh per stitched :class:`Path`. A path that cannot be meshed is skipped
    with a recorded reason rather than sinking the whole set — the same contract
    :func:`massing.buildings_to_meshes` keeps.
    """
    paths = stitch_paths(centerlines, frame)
    meshes: list[Mesh] = []
    skipped: list[dict] = []
    for idx, path in enumerate(paths):
        label = path.name or f"{name_prefix}_{idx}"
        safe = _safe_name(label)
        try:
            meshes.append(
                ribbon_mesh(
                    path,
                    width=width,
                    name=f"{name_prefix}_{safe}_{idx}",
                    frame=frame,
                    ground=ground,
                    lift=lift,
                    smooth_m=smooth_m,
                )
            )
        except RoadwayError as exc:
            skipped.append({"path": label, "osm_ids": path.osm_ids, "reason": str(exc)})
    return meshes, skipped


# ── helpers ───────────────────────────────────────────────────────────────────

def _same(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < EPS and abs(a[1] - b[1]) < EPS


def _polyline_length(xy: list[tuple[float, float]], closed: bool) -> float:
    pts = xy[:-1] if (closed and len(xy) >= 2 and _same(xy[0], xy[-1])) else xy
    n = len(pts)
    span = n if closed else n - 1
    total = 0.0
    for i in range(span):
        a, b = pts[i], pts[(i + 1) % n]
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total


def _ribbon_area(left, right, closed: bool) -> float:
    """Shoelace area of the ribbon's outline: down the left row and back up the
    right. An independent number to check length*width against."""
    outline = list(left) + list(reversed(right))
    n = len(outline)
    total = 0.0
    for i in range(n):
        x1, y1 = outline[i]
        x2, y2 = outline[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def _safe_name(label: str) -> str:
    keep = [c if (c.isalnum() or c in "_-") else "_" for c in str(label)]
    return "".join(keep)[:40] or "way"
