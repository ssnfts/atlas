"""
A field of cars running a lap, and the shots that cover it.

Two jobs, kept apart because they fail differently. Putting cars on the circuit
is arithmetic — arc length along a closed spine — and it is either right or
obviously wrong. Choosing where the camera goes is not arithmetic, and the only
way to know a shot works is to look at it.

**There is no keyframe here, and that is deliberate.** 3ds Max has an animation
system and the bridge has no access to it, but a sequence driven from this side
does not need one: every frame is a scene state, so a frame is "move the cars,
move the camera, render". Nothing is stored in Max between frames, so a run can
be stopped, resumed, re-ordered or re-rendered at a different resolution without
any of the state that makes an animated scene file fragile. The cost is a scene
update per frame, which is milliseconds against a render measured in minutes.

The unit of time is the **lap fraction**, 0.0 at the start line and 1.0 back at
it. Real speed varies hugely — a modern car is at 320 km/h on the back straight
and 80 km/h through the slow corners — and this does not model that. Cars
advance at constant arc-length rate, which is honest about being blocking rather
than simulation: it puts the right cars in the right part of the circuit at the
right moment for framing a shot, and it should not be described as a lap time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "Shot",
    "SHOTS",
    "lap_length",
    "point_at",
    "field_positions",
    "camera_for",
]


def lap_length(spine, closed: bool = True) -> float:
    """Total arc length of a centre-line."""
    n = len(spine)
    span = n if closed else n - 1
    return sum(
        math.hypot(spine[(i + 1) % n][0] - spine[i][0],
                   spine[(i + 1) % n][1] - spine[i][1])
        for i in range(span)
    )


def point_at(spine, distance: float, *, closed: bool = True):
    """
    Position and heading at ``distance`` metres along the spine.

    Returns ``(x, y, heading_degrees)`` with heading clockwise from +Y (north),
    the convention the scene frame and the sun azimuth both use, and the one
    :func:`cars.cars_on_grid` expects.
    """
    n = len(spine)
    total = lap_length(spine, closed)
    if total <= 0:
        raise ValueError("degenerate spine")
    d = distance % total if closed else max(0.0, min(distance, total))

    span = n if closed else n - 1
    for i in range(span):
        a = spine[i]
        b = spine[(i + 1) % n]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg <= 1e-9:
            continue
        if d <= seg:
            f = d / seg
            x = a[0] + (b[0] - a[0]) * f
            y = a[1] + (b[1] - a[1]) * f
            heading = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0
            return x, y, heading
        d -= seg

    a, b = spine[-1], spine[0]
    heading = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0
    return spine[-1][0], spine[-1][1], heading


def field_positions(spine, lap_fraction: float, count: int = 20, *,
                    gap_m: float = 18.0, lead_offset_m: float = 0.0,
                    stagger_m: float = 3.2, closed: bool = True):
    """
    Where every car is at a given point in the lap.

    The field is strung out behind the leader at ``gap_m`` intervals and
    alternates side of the racing line by ``stagger_m`` — cars do not run in a
    single file nose to tail, and a train of perfectly aligned cars is the tell
    of a placeholder. Returns ``(x, y, heading)`` per car, leader first.
    """
    total = lap_length(spine, closed)
    lead = lap_fraction * total + lead_offset_m

    out = []
    for i in range(count):
        x, y, heading = point_at(spine, lead - i * gap_m, closed=closed)
        # Offset perpendicular to travel, alternating side.
        side = 1.0 if i % 2 == 0 else -1.0
        a = math.radians(heading)
        nx, ny = math.cos(a), -math.sin(a)   # left of travel
        out.append((x + nx * side * stagger_m, y + ny * side * stagger_m, heading))
    return out


@dataclass(frozen=True)
class Shot:
    """
    One camera setup.

    ``kind`` is documentation rather than behaviour — a drone shot and a locked
    -off shot are both "a camera position per frame" — but it is worth carrying,
    because a cut list that cannot say which shots move is not a cut list.

    ``at`` is called with ``(spine, lap_fraction, site_z)`` and returns
    ``(position, target, fov)``. It is a function rather than a start/end pair
    so a move can be an arc or a rise rather than only a straight line.
    """

    name: str
    kind: str                 # "drone" | "static"
    lap_from: float
    lap_to: float
    at: object = field(repr=False)
    note: str = ""


# ── shot helpers ──────────────────────────────────────────────────────────────

def _lead(spine, lap_fraction):
    return point_at(spine, lap_fraction * lap_length(spine))


def _behind(spine, lap_fraction, back, side, up, site_z, look_ahead=14.0,
            fov=44.0):
    """A chase position: behind the leader, offset and raised."""
    x, y, h = _lead(spine, lap_fraction)
    a = math.radians(h)
    fx, fy = math.sin(a), math.cos(a)          # forward
    lx, ly = math.cos(a), -math.sin(a)         # left
    px = x - fx * back + lx * side
    py = y - fy * back + ly * side
    tx, ty, _ = point_at(spine, lap_fraction * lap_length(spine) + look_ahead)
    return (px, py, site_z + up), (tx, ty, site_z + 0.6), fov


def _trackside(spine, lap_fraction, side, dist, up, site_z, fov=52.0):
    """A locked-off camera beside the circuit, looking at the leader."""
    x, y, h = _lead(spine, lap_fraction)
    a = math.radians(h)
    lx, ly = math.cos(a), -math.sin(a)
    return ((x + lx * side * dist, y + ly * side * dist, site_z + up),
            (x, y, site_z + 0.7), fov)


def _orbit(spine, lap_fraction, radius, up, turns, site_z, fov=40.0):
    """A drone circling the leader as it passes."""
    x, y, _ = _lead(spine, lap_fraction)
    a = 2.0 * math.pi * turns * lap_fraction
    return ((x + math.sin(a) * radius, y + math.cos(a) * radius, site_z + up),
            (x, y, site_z + 0.8), fov)


# ── the cut list ──────────────────────────────────────────────────────────────
#
# Twelve shots covering one lap: seven drone moves and five locked off. Ordered
# as a sequence, so the lap fractions advance and the coverage does not double
# back on itself. Heights are above the graded site, not above sea level -- the
# circuit sits at 5.27 m and a camera at "1.5" would be underground, which is
# exactly the mistake that produced a frame of black on the first hero attempt.

SHOTS: list[Shot] = [
    Shot("01_grid_low", "static", 0.0, 0.0,
         lambda s, t, z: _trackside(s, 0.0, -1.0, 11.0, 1.1, z, fov=40.0),
         "grid, low and wide from the sunlit side"),
    Shot("02_grid_rise", "drone", 0.0, 0.02,
         lambda s, t, z: ((_lead(s, t)[0], _lead(s, t)[1] - 40.0,
                           z + 6.0 + 60.0 * (t / 0.02 if t else 0.0)),
                          (_lead(s, t)[0], _lead(s, t)[1], z), 46.0),
         "rise off the grid as the field launches"),
    Shot("03_t1_static", "static", 0.06, 0.06,
         lambda s, t, z: _trackside(s, t, 1.0, 26.0, 3.2, z, fov=34.0),
         "locked off at the first corner"),
    Shot("04_chase_back", "drone", 0.16, 0.20,
         lambda s, t, z: _behind(s, t, 15.0, 2.5, 2.2, z, 22.0, 40.0),
         "low chase, close behind the leader"),
    Shot("05_skim", "drone", 0.26, 0.29,
         lambda s, t, z: _behind(s, t, 9.0, -6.0, 0.9, z, 26.0, 34.0),
         "skimming the surface alongside"),
    Shot("06_hairpin_orbit", "drone", 0.36, 0.40,
         lambda s, t, z: _orbit(s, t, 48.0, 22.0, 6.0, z, fov=38.0),
         "orbiting the hairpin"),
    # On the circuit's own axis, not off to the side. The first version stood
    # 95 m out to one side with a 17 degree lens and rendered pure black -- at
    # that distance it was inside a building, and a camera inside geometry sees
    # the unlit backs of its faces. A compressed telephoto down a straight is an
    # on-axis shot anyway: the long lens is what stacks the field up, and it
    # only stacks if you are looking along the line they are running.
    Shot("07_long_lens", "static", 0.47, 0.47,
         lambda s, t, z: _behind(s, t, 240.0, 1.5, 2.4, z, 30.0, 13.0),
         "long lens down the straight, field compressed"),
    Shot("08_high_wide", "drone", 0.55, 0.58,
         lambda s, t, z: ((_lead(s, t)[0] - 150.0, _lead(s, t)[1] - 190.0, z + 130.0),
                          (_lead(s, t)[0], _lead(s, t)[1], z), 50.0),
         "high and wide, circuit in context"),
    Shot("09_marina", "static", 0.66, 0.66,
         lambda s, t, z: _trackside(s, t, 1.0, 34.0, 8.0, z, fov=40.0),
         "elevated static over the marina section"),
    Shot("10_low_front", "static", 0.74, 0.74,
         lambda s, t, z: (lambda p: ((p[0] + math.sin(math.radians(p[2])) * 30.0,
                                      p[1] + math.cos(math.radians(p[2])) * 30.0,
                                      z + 0.75),
                                     (p[0], p[1], z + 0.6), 30.0))(_lead(s, t)),
         "head-on, tyre height, cars coming at the lens"),
    Shot("11_pullback", "drone", 0.84, 0.88,
         lambda s, t, z: _behind(s, t, 30.0 + 340.0 * ((t - 0.84) / 0.04),
                                 0.0, 12.0 + 150.0 * ((t - 0.84) / 0.04), z, 30.0, 46.0),
         "pull back and up, revealing the lap"),
    Shot("12_finish", "drone", 0.98, 1.0,
         lambda s, t, z: ((_lead(s, t)[0] + 26.0, _lead(s, t)[1] - 30.0, z + 16.0),
                          (_lead(s, t)[0], _lead(s, t)[1], z + 0.5), 40.0),
         "over the line to finish the lap"),
]


def camera_for(shot: Shot, spine, lap_fraction: float, site_z: float):
    """Resolve a shot to ``(position, target, fov)`` at one moment."""
    return shot.at(spine, lap_fraction, site_z)
