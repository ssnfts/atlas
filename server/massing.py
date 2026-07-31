"""
Building footprints -> extruded meshes in the scene frame.

This is the module most exposed to the project's characteristic failure: it
cannot throw. A triangulation that drops a triangle, a ring wound the wrong way,
a normal pointing inward — none of these raise. They render. A city block with
inverted normals looks like a city block until the sun goes behind it and the
shadows fall out of the wrong faces, and by then the render is on a timeline.

So the arithmetic here is checked against invariants that a self-consistent bug
cannot satisfy:

- **Area is conserved.** The triangles of a roof cap must sum to the area of the
  polygon they came from. Ear clipping that skips a vertex or crosses itself
  fails this immediately, and :func:`triangulate` raises rather than returning
  shrapnel.
- **Winding is explicit.** Outer rings are forced counter-clockwise and holes
  clockwise before anything is built, so "outward" is a computed direction and
  not an assumption about what the mapper happened to draw.
- **Volume is closed.** Every mesh is watertight, so the vertex count, face
  count and Euler characteristic are all predictable and testable.

Deliberately dependency-free, like ``solar.py`` and ``frame.py``. ``shapely`` is
in the requirements and would shorten this, but its triangulation is an
unconstrained Delaunay of the vertex set — for a concave L-shaped footprint that
covers the notch as well, adding a wing to the building that OSM never mapped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from frame import SceneFrame
from osm import Building

__all__ = [
    "Mesh",
    "MassingError",
    "signed_area",
    "orient",
    "triangulate",
    "extrude",
    "ground_under",
    "building_to_mesh",
    "buildings_to_meshes",
]

# Rings closer than this in the plan are treated as the same point. 1 mm is far
# below OSM's positional accuracy (metres at best) and far above the float noise
# introduced by projecting degrees to metres.
EPS = 1e-6

# Area conservation tolerance, as a fraction of the polygon's own area.
AREA_TOLERANCE = 1e-4


class MassingError(ValueError):
    """A footprint could not be turned into sound geometry."""


@dataclass
class Mesh:
    """
    A watertight triangle mesh in scene metres.

    ``faces`` are **0-based** index triples, wound counter-clockwise seen from
    outside the solid. MaxScript arrays are 1-based, so the +1 happens once, at
    the bridge boundary, rather than being carried through this module where it
    would be easy to apply twice or not at all.
    """

    verts: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    name: str = "building"
    metadata: dict = field(default_factory=dict)
    # Texture coordinates, one per vertex, sharing ``faces``. Empty for massing,
    # which is projected in world space and needs none; a roadway fills it,
    # because a kerb stripe has to follow the kerb round a bend and world-space
    # projection cannot do that.
    uvs: list[tuple[float, float]] = field(default_factory=list)

    @property
    def vertex_count(self) -> int:
        return len(self.verts)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Axis-aligned bounds as ``(min, max)`` corners."""
        if not self.verts:
            raise MassingError("empty mesh has no bounds")
        xs, ys, zs = zip(*self.verts)
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def volume(self) -> float:
        """
        Signed volume via the divergence theorem.

        Positive for correct outward winding, negative if the mesh is
        inside-out. That sign is the cheapest available check that normals face
        the way they are supposed to, and it is a property no unit test of the
        individual triangles would catch.
        """
        total = 0.0
        for i, j, k in self.faces:
            ax, ay, az = self.verts[i]
            bx, by, bz = self.verts[j]
            cx, cy, cz = self.verts[k]
            total += (
                ax * (by * cz - bz * cy)
                - ay * (bx * cz - bz * cx)
                + az * (bx * cy - by * cx)
            )
        return total / 6.0


# ── Ring geometry ─────────────────────────────────────────────────────────────

def signed_area(ring: list[tuple[float, float]]) -> float:
    """
    Shoelace area of a planar ring. Positive counter-clockwise.

    The sign is the load-bearing part: it is how ring direction is determined,
    and ring direction is what decides whether a wall faces the street or the
    living room.
    """
    total = 0.0
    count = len(ring)
    for i in range(count):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % count]
        total += x1 * y2 - x2 * y1
    return total * 0.5


def orient(ring: list[tuple[float, float]], *, ccw: bool) -> list[tuple[float, float]]:
    """Return ``ring`` wound the requested way, reversing only if needed."""
    area = signed_area(ring)
    if (area < 0.0) == ccw:
        return list(reversed(ring))
    return list(ring)


def _clean_ring(ring: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Drop consecutive duplicate vertices and any repeated closing vertex."""
    cleaned: list[tuple[float, float]] = []
    for point in ring:
        if not cleaned or _dist2(cleaned[-1], point) > EPS * EPS:
            cleaned.append(point)
    while len(cleaned) >= 2 and _dist2(cleaned[0], cleaned[-1]) <= EPS * EPS:
        cleaned.pop()
    return cleaned


def _dist2(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _cross(
    o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]
) -> float:
    """Z component of (a-o) x (b-o). Positive when o->a->b turns left."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """Strictly-inside test for a counter-clockwise triangle."""
    return (
        _cross(a, b, p) > EPS and _cross(b, c, p) > EPS and _cross(c, a, p) > EPS
    )


def _blocks_ear(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """
    Whether vertex ``p`` prevents triangle (a, b, c) from being clipped as an ear.

    The test is inside-**or-on**, and the boundary case is the whole point. On an
    L-shaped footprint the reflex corner of the notch lies exactly on the line
    between the two far corners, so a strictly-inside test finds nothing in the
    way and happily clips an ear straight across the notch — filling in the
    missing quarter of the building. The result is a plausible rectangular
    block, no error anywhere, and a 400 m2 roof on a 300 m2 footprint.

    Vertices coincident with the triangle's own corners are excluded by
    position, because bridging a hole duplicates two vertices by construction
    and each copy would otherwise report itself as blocking.
    """
    if (
        _dist2(p, a) <= EPS * EPS
        or _dist2(p, b) <= EPS * EPS
        or _dist2(p, c) <= EPS * EPS
    ):
        return False
    return (
        _cross(a, b, p) >= -EPS and _cross(b, c, p) >= -EPS and _cross(c, a, p) >= -EPS
    )


# ── Triangulation ─────────────────────────────────────────────────────────────

def triangulate(
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """
    Ear-clip a polygon, with holes, into counter-clockwise triangles.

    Returns ``(vertices, faces)`` — the vertex list is returned alongside
    because bridging a hole introduces duplicated vertices, so the output
    indices do not address the input rings.

    Raises :class:`MassingError` if the triangles do not account for the
    polygon's area to within :data:`AREA_TOLERANCE`. That check is the whole
    reason this is trustworthy: ear clipping degrades gracefully into
    plausible-looking rubbish on a self-intersecting ring, and OSM contains
    self-intersecting rings.
    """
    outer = _clean_ring(list(outer))
    if len(outer) < 3:
        raise MassingError(f"outer ring has {len(outer)} distinct vertices, need 3")

    outer = orient(outer, ccw=True)
    target_area = signed_area(outer)

    cleaned_holes = []
    for hole in holes or []:
        ring = _clean_ring(list(hole))
        if len(ring) < 3:
            continue
        # Clockwise, so that walking the merged polygon keeps the solid on the
        # left through both the outer boundary and the hole boundary.
        ring = orient(ring, ccw=False)
        cleaned_holes.append(ring)
        target_area -= abs(signed_area(ring))

    if target_area <= 0.0:
        raise MassingError("holes consume the whole footprint")

    polygon = _merge_holes(outer, cleaned_holes)
    faces = _ear_clip(polygon)

    if not faces:
        raise MassingError(f"triangulation produced no faces from {len(polygon)} vertices")

    built = sum(
        abs(_cross(polygon[i], polygon[j], polygon[k])) * 0.5 for i, j, k in faces
    )
    if abs(built - target_area) > max(AREA_TOLERANCE * target_area, EPS):
        raise MassingError(
            f"triangulation lost area: {built:.6f} m2 of triangles for a "
            f"{target_area:.6f} m2 polygon ({len(polygon)} vertices, "
            f"{len(faces)} faces). The ring is probably self-intersecting."
        )

    return _weld(polygon, faces)


def _weld(
    polygon: list[tuple[float, float]], faces: list[tuple[int, int, int]]
) -> tuple[list[tuple[float, float]], list[tuple[int, int, int]]]:
    """
    Collapse coincident vertices to one index and drop degenerate triangles.

    Bridging a hole duplicates two vertices by construction, so without this the
    roof triangles and the walls end up referencing *different* indices for the
    same physical point. The mesh looks closed and is not: Max sees unwelded
    vertices, smoothing breaks along the seam, and V-Ray renders a hairline
    crack that no amount of staring at the footprint explains.

    Collapsing the seam can leave triangles with two identical corners. Those
    have no area and no normal, so they are removed rather than handed to Max.
    """
    canonical: dict[tuple[int, int], int] = {}
    verts: list[tuple[float, float]] = []
    remap: list[int] = []

    for point in polygon:
        key = _key(point)
        if key not in canonical:
            canonical[key] = len(verts)
            verts.append(point)
        remap.append(canonical[key])

    welded: list[tuple[int, int, int]] = []
    for i, j, k in faces:
        a, b, c = remap[i], remap[j], remap[k]
        if a == b or b == c or a == c:
            continue
        welded.append((a, b, c))

    return verts, welded


def _merge_holes(
    outer: list[tuple[float, float]], holes: list[list[tuple[float, float]]]
) -> list[tuple[float, float]]:
    """
    Cut each hole into the outer ring, producing one simple polygon.

    The classic bridge construction: from the hole's rightmost vertex, cast a
    ray east, and join to a vertex of the outer ring that is visible from it.
    The seam is traversed twice, once in each direction, so it has zero width
    and contributes no area.

    Holes are merged rightmost-first. Order matters when one hole's bridge would
    otherwise cross another hole that has not been merged yet.
    """
    if not holes:
        return list(outer)

    polygon = list(outer)
    for hole in sorted(holes, key=lambda h: -max(p[0] for p in h)):
        polygon = _bridge_hole(polygon, hole)
    return polygon


def _bridge_hole(
    outer: list[tuple[float, float]], hole: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    m_idx = max(range(len(hole)), key=lambda i: (hole[i][0], hole[i][1]))
    m = hole[m_idx]

    p_idx = _visible_vertex(outer, m)
    if p_idx is None:
        raise MassingError(
            f"no vertex of the outer ring is visible from hole vertex {m}; "
            "the hole probably lies outside or across the footprint"
        )

    p_idx = _anchor_occurrence(outer, p_idx, m)
    rotated = hole[m_idx:] + hole[:m_idx] + [hole[m_idx]]
    return outer[: p_idx + 1] + rotated + outer[p_idx:]


def _anchor_occurrence(
    polygon: list[tuple[float, float]], p_idx: int, m: tuple[float, float]
) -> int:
    """
    Pick *which copy* of the anchor vertex the new bridge attaches to.

    A vertex that has already been bridged appears in the polygon more than
    once, and two holes anchoring to the same corner is not a rare case — it is
    what happens whenever several courtyards can all see the same corner of the
    block. Splicing the second seam in at the first copy puts the two seams in
    the wrong rotational order around that corner, which turns it reflex and
    quietly invalidates the polygon: the ear clipper then emits a few
    overlapping triangles and stalls, and the roof comes out with a wedge
    missing and a wedge doubled.

    The fix is to insert into the correct angular wedge. Each copy of the
    vertex owns one wedge of the interior, bounded by its own two edges; the
    bridge belongs to the copy whose wedge actually contains its direction.
    """
    anchor = polygon[p_idx]
    occurrences = [i for i, p in enumerate(polygon) if _dist2(p, anchor) <= EPS * EPS]
    if len(occurrences) < 2:
        return p_idx

    direction = (m[0] - anchor[0], m[1] - anchor[1])
    for i in occurrences:
        prev_p = polygon[i - 1]
        next_p = polygon[(i + 1) % len(polygon)]
        u = (prev_p[0] - anchor[0], prev_p[1] - anchor[1])
        v = (next_p[0] - anchor[0], next_p[1] - anchor[1])
        if _in_wedge(direction, v, u):
            return i
    return p_idx


def _in_wedge(
    d: tuple[float, float], v: tuple[float, float], u: tuple[float, float]
) -> bool:
    """
    Whether direction ``d`` lies in the interior wedge swept counter-clockwise
    from ``v`` to ``u``.

    The two branches are not interchangeable: a convex corner spans less than
    180° and needs both half-plane tests, a reflex corner spans more than 180°
    and needs either. Using the convex test on a reflex corner rejects every
    direction in the larger half of the wedge.
    """
    if _cross2(v, u) > 0:
        return _cross2(v, d) >= -EPS and _cross2(d, u) >= -EPS
    return _cross2(v, d) >= -EPS or _cross2(d, u) >= -EPS


def _cross2(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Z component of a x b, for vectors already relative to a shared origin."""
    return a[0] * b[1] - a[1] * b[0]


def _visible_vertex(outer: list[tuple[float, float]], m: tuple[float, float]) -> int | None:
    """
    Index of an outer vertex that the hole vertex ``m`` can see.

    Ray-casts east from ``m`` to the nearest outer edge, takes that edge's
    right-hand endpoint as a candidate, then checks whether any reflex vertex of
    the outer ring lies inside the triangle (m, intersection, candidate). If one
    does it blocks the line of sight, and the blocker closest in angle to the
    ray is used instead.
    """
    count = len(outer)
    best_t = math.inf
    best_edge = None
    hit = None

    for i in range(count):
        a, b = outer[i], outer[(i + 1) % count]
        # Only edges the eastward ray can cross, counted half-open so a ray
        # through a shared vertex is not registered against both edges.
        if (a[1] > m[1]) == (b[1] > m[1]):
            continue
        t = (m[1] - a[1]) / (b[1] - a[1])
        x = a[0] + t * (b[0] - a[0])
        if x < m[0] - EPS:
            continue
        if x - m[0] < best_t:
            best_t = x - m[0]
            best_edge = i
            hit = (x, m[1])

    if best_edge is None:
        return None

    a_idx, b_idx = best_edge, (best_edge + 1) % count
    candidate = a_idx if outer[a_idx][0] > outer[b_idx][0] else b_idx

    # An exact hit on a vertex is already visible; nothing can lie between.
    if _dist2(hit, outer[candidate]) <= EPS * EPS:
        return candidate

    # The sight triangle is (m, hit, candidate), and its winding depends on
    # whether the candidate lies above or below the eastward ray. Ordering it
    # counter-clockwise first is not cosmetic: the containment test below is
    # written for CCW input and silently reports "nothing in the way" for every
    # point when handed a clockwise triangle. That picks an anchor the hole
    # cannot actually see, and the bridge crosses the geometry it was routed
    # around.
    corners = (m, hit, outer[candidate])
    if _cross(*corners) < 0:
        corners = (m, outer[candidate], hit)

    blockers = []
    for i in range(count):
        if i in (a_idx, b_idx):
            continue
        prev_p, cur, next_p = outer[i - 1], outer[i], outer[(i + 1) % count]
        if _cross(prev_p, cur, next_p) >= 0:
            continue  # convex; cannot block
        if _point_in_triangle(cur, *corners):
            blockers.append(i)

    if not blockers:
        return candidate

    # Of the blockers, the one subtending the smallest angle to the ray is
    # visible from m — ties (collinear blockers) broken by proximity.
    def angle_key(i: int) -> tuple[float, float]:
        dx, dy = outer[i][0] - m[0], outer[i][1] - m[1]
        return (abs(math.atan2(dy, dx)), dx * dx + dy * dy)

    return min(blockers, key=angle_key)


def _ear_clip(polygon: list[tuple[float, float]]) -> list[tuple[int, int, int]]:
    """
    Ear clipping over a counter-clockwise simple polygon.

    Returns index triples into ``polygon``. Bails out rather than looping
    forever if no ear can be found; the area check in :func:`triangulate` turns
    that into a loud failure.
    """
    count = len(polygon)
    if count < 3:
        return []

    remaining = list(range(count))
    faces: list[tuple[int, int, int]] = []

    # Each successful clip removes one vertex, so this bound cannot be reached
    # by healthy input — it only stops a pathological ring from spinning.
    guard = count * count + 16

    while len(remaining) > 3 and guard > 0:
        guard -= 1
        clipped = False

        for pos in range(len(remaining)):
            prev_i = remaining[pos - 1]
            cur_i = remaining[pos]
            next_i = remaining[(pos + 1) % len(remaining)]

            a, b, c = polygon[prev_i], polygon[cur_i], polygon[next_i]
            if _cross(a, b, c) <= EPS:
                continue  # reflex or collinear — not an ear

            if any(
                _blocks_ear(polygon[other], a, b, c)
                for other in remaining
                if other not in (prev_i, cur_i, next_i)
            ):
                continue

            faces.append((prev_i, cur_i, next_i))
            remaining.pop(pos)
            clipped = True
            break

        if not clipped:
            break

    if len(remaining) == 3:
        faces.append((remaining[0], remaining[1], remaining[2]))
    return faces


# ── Extrusion ─────────────────────────────────────────────────────────────────

def extrude(
    outer: list[tuple[float, float]],
    height: float,
    *,
    holes: list[list[tuple[float, float]]] | None = None,
    base: float = 0.0,
    name: str = "building",
) -> Mesh:
    """
    Extrude a footprint from ``base`` to ``base + height`` into a closed solid.

    Produces a watertight prism: a roof cap facing +Z, a floor cap facing -Z and
    quad walls, all wound so that normals point out of the solid.

    The floor cap is not an optimisation to skip. V-Ray's GPU engine treats an
    open mesh as single-sided during GI, and a building without a floor leaks
    light up into itself from below — subtle, plausible, and maddening to trace
    back to a missing polygon.
    """
    if not math.isfinite(height) or height <= 0.0:
        raise MassingError(f"height must be positive and finite, got {height!r}")

    plan, cap_faces = triangulate(outer, holes)
    top_z = base + height
    count = len(plan)

    # Vertices: the whole plan at the base, then the whole plan at the top. The
    # index of a top vertex is always its base index + count, which keeps the
    # wall construction below readable.
    verts = [(x, y, base) for x, y in plan] + [(x, y, top_z) for x, y in plan]

    faces: list[tuple[int, int, int]] = []

    # Roof: as triangulated, counter-clockwise from above -> normal +Z.
    faces += [(i + count, j + count, k + count) for i, j, k in cap_faces]
    # Floor: the same triangles reversed -> normal -Z.
    faces += [(k, j, i) for i, j, k in cap_faces]

    # Walls. For an edge a->b the outward direction is to the right of travel,
    # which is why the outer ring is forced counter-clockwise and holes
    # clockwise before we get here: it makes "right of travel" mean "away from
    # the solid" on both.
    for a, b in _boundary_edges(plan, outer, holes):
        faces.append((a, b, b + count))
        faces.append((a, b + count, a + count))

    return Mesh(
        verts=verts,
        faces=faces,
        name=name,
        metadata={"height_m": round(height, 3), "base_m": round(base, 3)},
    )


def _boundary_edges(
    plan: list[tuple[float, float]],
    outer: list[tuple[float, float]],
    holes: list[list[tuple[float, float]]] | None,
) -> list[tuple[int, int]]:
    """
    Wall edges as index pairs into the merged plan.

    Walls are generated from the *original* rings rather than from the merged
    polygon, because the merged polygon contains the bridge seams. A seam is a
    zero-width cut with no physical existence — extruding it puts a
    zero-thickness wall through the middle of the courtyard, which renders as a
    black sliver and defeats every attempt to explain it.
    """
    lookup: dict[tuple[int, int], int] = {}
    for index, point in enumerate(plan):
        lookup.setdefault(_key(point), index)

    edges: list[tuple[int, int]] = []
    rings = [orient(_clean_ring(list(outer)), ccw=True)]
    for hole in holes or []:
        ring = _clean_ring(list(hole))
        if len(ring) >= 3:
            rings.append(orient(ring, ccw=False))

    for ring in rings:
        for i, point in enumerate(ring):
            nxt = ring[(i + 1) % len(ring)]
            a, b = lookup.get(_key(point)), lookup.get(_key(nxt))
            if a is None or b is None or a == b:
                raise MassingError(
                    "a ring vertex is missing from the triangulated plan — "
                    "the merge and the rings have diverged"
                )
            edges.append((a, b))
    return edges


def _key(point: tuple[float, float]) -> tuple[int, int]:
    """Quantise to EPS so a vertex duplicated by bridging maps to one index."""
    return (round(point[0] / EPS), round(point[1] / EPS))


# ── Buildings -> meshes ───────────────────────────────────────────────────────

def ground_under(building: Building, ground) -> float:
    """
    The elevation a building should stand on, given a ``ground(lat, lon)`` probe.

    Takes the **minimum** across the footprint's vertices and its centroid, not
    the centroid alone and not the mean. On a slope those differ by metres, and
    the three failure modes are not equally bad:

    - centroid or mean: the downhill half of the building floats, and a gap
      under a building reads instantly as broken;
    - maximum: the whole building sits on stilts, worse again;
    - minimum: the uphill corner is buried in the hillside, which is what a real
      building's foundation does anyway and is invisible in a render.

    So the minimum is chosen because its failure is the one nobody notices. A
    building sunk into a hill looks like a building on a hill.

    ``ground`` is a callable rather than a terrain patch so this module stays
    independent of :mod:`terrain` — terrain already imports this one, and the
    reverse would be a cycle.
    """
    samples = []
    for lat, lon in building.outer:
        try:
            samples.append(ground(lat, lon))
        except Exception:
            continue  # outside the patch; other vertices may still be inside

    if not samples:
        raise MassingError(
            f"no part of {building.osm_type} {building.osm_id} falls inside the "
            "terrain patch, so its ground level is unknown. Widen the terrain "
            "radius rather than defaulting it to zero."
        )
    return min(samples)


def building_to_mesh(building: Building, frame: SceneFrame, ground=None) -> Mesh:
    """
    Project one :class:`~osm.Building` into the scene frame and extrude it.

    ``ground``, when given, is a callable ``(lat, lon) -> elevation`` used to sit
    the building on terrain instead of on the z=0 plane. Without it every
    building stands at zero, which is correct only where the terrain is.
    """
    outer = [frame.to_scene(lat, lon)[:2] for lat, lon in building.outer]
    holes = [
        [frame.to_scene(lat, lon)[:2] for lat, lon in ring] for ring in building.inners
    ]

    base_elevation = ground_under(building, ground) if ground is not None else 0.0

    height = building.height_m - building.min_height_m
    mesh = extrude(
        outer,
        height,
        holes=holes,
        base=base_elevation + building.min_height_m,
        name=_mesh_name(building),
    )
    mesh.metadata.update(
        {
            "osm_id": building.osm_id,
            "osm_type": building.osm_type,
            "height_source": building.height_source,
            "ground_m": round(base_elevation, 3),
        }
    )
    return mesh


def buildings_to_meshes(
    buildings: list[Building], frame: SceneFrame, ground=None
) -> tuple[list[Mesh], list[dict]]:
    """
    Convert many buildings, returning ``(meshes, skipped)``.

    A city block of a thousand footprints will contain a few that OSM has
    mapped badly. Those are collected into ``skipped`` with the reason attached
    rather than aborting the build — but they are *returned*, not swallowed, so
    a caller that silently loses a third of a district can notice.
    """
    meshes: list[Mesh] = []
    skipped: list[dict] = []

    for building in buildings:
        try:
            meshes.append(building_to_mesh(building, frame, ground))
        except MassingError as exc:
            skipped.append(
                {
                    "osm_id": building.osm_id,
                    "osm_type": building.osm_type,
                    "name": building.name,
                    "reason": str(exc),
                }
            )
    return meshes, skipped


def _mesh_name(building: Building) -> str:
    """
    A scene-node name that is stable, unique and searchable.

    The OSM id is included because names are not unique — a retail street has a
    dozen nodes called "Costa" — and 3ds Max silently renames a collision by
    appending a number, which breaks any later lookup by name.
    """
    stem = building.name or building.tags.get("building") or "building"
    safe = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in stem).strip()
    safe = (safe or "building")[:40]
    return f"osm_{building.osm_type[0]}{building.osm_id}_{safe}"
