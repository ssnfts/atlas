"""
A field of cars running a lap, and the shots that cover it.

Two jobs, kept apart because they fail differently. Putting cars on the circuit
is arithmetic — arc length along a closed spine — and it is either right or
obviously wrong. Choosing where the camera goes is not arithmetic, and the only
way to know a shot works is to look at it.

This module is pure: it computes where things are, and something else writes
keys. That split is why the same functions serve both a still-per-shot render
and a fully keyed scene.

**Speed comes from the circuit's own shape, not from a clock.** The first
version advanced every car at a constant arc-length rate. It was honest about
being blocking, and it also looked dead — constant speed is linear motion, and
a hairpin taken at the same rate as the back straight reads as a train on rails.
Speed is now grip-limited per corner (``v = sqrt(a_lat * R)``) and then swept
backwards for braking and forwards for traction, which is how a lap simulation
is built. The braking zone lands *before* the corner because the arithmetic puts
it there, and ease-in/ease-out fall out of the physics rather than being chosen.

Checked against the world: **80.7 s** against a real 2021 pole of **82.1 s** on
this layout, 1.7% quick. See :func:`curvature_radius` for why that agreement
should be read with care — the curvature baseline was fixed after the lap times
were seen, so it corroborates rather than independently confirms.

An earlier version of this note claimed 78.1 s and called it a good match. It
was not: the circumradius was being computed as ``abc/(2A)`` instead of
``abc/(4A)``, so every radius came out double and every corner was taken sqrt(2)
too fast. Two errors were partially cancelling, and the plausible answer hid
both. It took a test fixture built at a known radius to see it.

It is still blocking. There is no driver, no traffic, no tyre model and no
racing line — cars follow the centre-line, where a real driver straightens every
corner — so the number should not be quoted as a lap time.

Gaps between cars are in **seconds**, not metres. That is what a real interval
is, and it is why the pack stretches on the straights and closes up in the slow
corners; spacing by distance does the opposite, which is instantly wrong to
anyone who has watched a race.
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
    "field_at_time",
    "camera_for",
    "speed_profile",
    "lap_time",
    "distance_at",
    "attitude_at",
    "curvature_radius",
    "CrashSpec",
    "CRASH",
    "crash_state",
    "Cut",
    "build_edit",
    "edit_length_frames",
    "orientation_quat",
    "nose_direction",
]


def orientation_quat(heading_deg: float, pitch_deg: float = 0.0,
                     roll_deg: float = 0.0) -> tuple[float, float, float, float]:
    """
    Body orientation as a quaternion ``(w, x, y, z)``.

    Composed here rather than handed to 3ds Max as Euler angles, because Max's
    Euler convention cost this project two real bugs that a render showed only
    as "the cars look wrong":

    * ``EulerAngles(0, 0, -heading)`` produced the **mirrored** rotation. A car
      keyed at heading 90 pointed west. Measured off the node's transform, not
      guessed: its local Y axis came back as ``[-1, 0, 0]``.
    * Pitch and roll were applied about the **world** axes rather than the car's
      own. At heading 90 a 3 degree roll tilted the body about world Y, which is
      pitch as far as the car is concerned. This is invisible while yaw is small
      and wrong everywhere else — an earlier comment here claimed the difference
      was below what a frame shows, and that was only true near heading 0.

    Composing the quaternion in Python makes the convention explicit and, more
    usefully, testable without a running 3ds Max: the nose direction for any
    heading is arithmetic, checkable against ``(sin h, cos h)``.

    Intrinsic Z-X-Y: yaw about the world up axis first, then pitch about the
    car's own lateral axis, then roll about its own nose axis. That order is
    what makes pitch and roll mean "nose down" and "lean" regardless of which
    way the car is facing.
    """
    # Scene headings run clockwise from +Y (north). A right-handed rotation
    # about +Z runs anticlockwise, so the yaw angle is the negative.
    z = math.radians(-heading_deg)
    x = math.radians(pitch_deg)
    y = math.radians(roll_deg)

    def q_axis(angle, axis):
        half = angle / 2.0
        s = math.sin(half)
        return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)

    def q_mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    q = q_mul(q_axis(z, (0.0, 0.0, 1.0)), q_axis(x, (1.0, 0.0, 0.0)))
    q = q_mul(q, q_axis(y, (0.0, 1.0, 0.0)))
    return q


def nose_direction(quat) -> tuple[float, float, float]:
    """
    Where a car with this orientation is pointing, in world axes.

    The car is modelled nose along local +Y, so this rotates ``(0, 1, 0)``. It
    exists to make the convention checkable: for a heading of ``h`` and no
    attitude, the answer must be ``(sin h, cos h, 0)``, which is a property a
    test can assert without a host.
    """
    w, x, y, z = quat
    return (
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    )


# ── speed, from the shape of the circuit ─────────────────────────────────────
#
# The first version moved every car at a constant arc-length rate. It was honest
# about being blocking, and it also looked dead: constant speed is linear motion,
# and linear motion through a hairpin at the same rate as the back straight
# reads as a train on rails rather than a car being driven.
#
# The fix is not an easing curve chosen by taste. A car's speed through a corner
# is set by how hard it can hold the corner: v = sqrt(a_lat * R). Deriving speed
# from the circuit's own curvature produces the deceleration into a corner and
# the acceleration out of it *for free*, at the right places, with the shape the
# geometry dictates. Ease-in and ease-out fall out of the physics.

G = 9.80665
LAT_G = 4.0                 # sustained lateral grip, modern high-downforce car
BRAKE_G = 4.5               # longitudinal, braking
ACCEL_G = 1.6               # longitudinal, driving out (traction limited)
V_MAX = 89.0                # m/s, ~320 km/h
V_MIN = 22.0                # m/s, ~80 km/h; slowest hairpin


def curvature_radius(spine, closed: bool = True,
                     baseline_m: float = 12.0) -> list[float]:
    """
    Radius of the circle through each vertex and a neighbour either side.

    ``baseline_m`` is how far along the path those neighbours are taken from,
    and it matters more than it looks. OSM digitises this circuit with vertices
    anywhere from 2 m to 266 m apart, and a circumradius measured across two
    *adjacent* vertices on a densely-sampled bend reads the sampling noise as
    curvature: it returned 6 m radii on a Formula 1 corner, which no car has
    ever taken. Stepping out to a fixed arc length measures the corner instead
    of the digitisation.

    12 m is chosen on geometric grounds: it recovers a 22 m tightest radius on
    this circuit, against a real turn-7 hairpin of roughly 25 m. **Disclosure,
    because the ordering matters:** a sweep of 6/12/18/25/35 m was run and the
    lap times seen (86.8 / 80.7 / 78.3 / 74.7 / 71.7 s) before this value was
    fixed, so treat the lap-time agreement as corroboration and not as an
    independent confirmation. The geometric argument is the one that should
    carry weight.

    A straight gives an enormous radius, a hairpin a small one. Returns metres,
    clamped so downstream arithmetic never sees an infinity.
    """
    n = len(spine)
    seg = [math.dist(spine[i], spine[(i + 1) % n]) for i in range(n)]

    def step(index: int, direction: int) -> int:
        """Walk out from ``index`` until ``baseline_m`` of path is covered."""
        travelled = 0.0
        j = index
        for _ in range(n):
            k = (j + direction) % n
            if not closed and (k == 0 or k == n - 1):
                return k
            travelled += seg[min(j, k)] if direction > 0 else seg[min(k, j)]
            j = k
            if travelled >= baseline_m:
                break
        return j

    out = []
    for i in range(n):
        a = spine[step(i, -1)]
        b = spine[i]
        c = spine[step(i, +1)]
        if not closed and (i == 0 or i == n - 1):
            out.append(1e6)
            continue
        # Circumradius R = abc / (4 * area). `cross` is the cross-product
        # magnitude, which is *twice* the triangle's area — so the divisor is
        # 2 * cross, not cross.
        #
        # Getting this wrong returned every radius at exactly double, which made
        # every corner appear takeable at sqrt(2) times its real speed. Nothing
        # errored; the lap simply ran too fast, and it took a fixture built at a
        # known radius to see it. A curvature function checked only against
        # itself cannot catch a constant factor.
        ab = math.dist(a, b)
        bc = math.dist(b, c)
        ca = math.dist(c, a)
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross < 1e-9 or ab < 1e-9 or bc < 1e-9:
            out.append(1e6)
        else:
            out.append(min(ab * bc * ca / (2.0 * cross), 1e6))
    return out


def speed_profile(spine, closed: bool = True) -> list[float]:
    """
    Target speed at each vertex, in metres per second.

    Three passes, which is how a real lap simulation is built:

    1. **Grip limit** — the fastest each corner can be taken, ``sqrt(a_lat * R)``.
    2. **Backward pass** — a car cannot arrive at a slow corner without having
       braked for it, so each vertex is limited by what it can shed before the
       next one. This is what puts the braking zone *before* the corner rather
       than in it, and it is the single thing that makes the motion read as
       driving.
    3. **Forward pass** — nor can it leave a corner at full speed; traction caps
       how fast it picks up.

    Run cyclically for a closed lap so the start line is not a discontinuity.
    """
    radii = curvature_radius(spine, closed)
    v = [max(V_MIN, min(V_MAX, math.sqrt(LAT_G * G * r))) for r in radii]

    n = len(spine)
    seg = [math.dist(spine[i], spine[(i + 1) % n]) for i in range(n)]

    # Two sweeps of each pass so a closed lap converges across the start line.
    for _ in range(2):
        for i in range(n - 1, -1, -1):          # backward: braking
            j = (i + 1) % n
            limit = math.sqrt(v[j] ** 2 + 2.0 * BRAKE_G * G * seg[i])
            v[i] = min(v[i], limit)
        for i in range(n):                      # forward: traction
            j = (i + 1) % n
            limit = math.sqrt(v[i] ** 2 + 2.0 * ACCEL_G * G * seg[i])
            v[j] = min(v[j], limit)
    return v


def lap_time(spine, closed: bool = True) -> float:
    """Seconds to complete a lap at the profiled speed."""
    n = len(spine)
    speeds = speed_profile(spine, closed)
    total = 0.0
    for i in range(n):
        j = (i + 1) % n
        seg = math.dist(spine[i], spine[j])
        mean_v = max(0.5 * (speeds[i] + speeds[j]), 1e-3)
        total += seg / mean_v
    return total


def distance_at(spine, seconds: float, closed: bool = True) -> float:
    """
    How far along the lap a car is after ``seconds``.

    Integrates the speed profile rather than multiplying by a mean, which is
    what makes the car linger in the slow corners and cover the straights
    quickly — the whole point of the exercise.
    """
    n = len(spine)
    speeds = speed_profile(spine, closed)
    remaining = seconds
    travelled = 0.0
    guard = 0
    while remaining > 0.0 and guard < n * 4:
        for i in range(n):
            j = (i + 1) % n
            seg = math.dist(spine[i], spine[j])
            mean_v = max(0.5 * (speeds[i] + speeds[j]), 1e-3)
            dt = seg / mean_v
            if dt >= remaining:
                return travelled + seg * (remaining / dt)
            remaining -= dt
            travelled += seg
        guard += n
    return travelled


def attitude_at(spine, distance: float, closed: bool = True) -> tuple[float, float]:
    """
    Body roll and pitch at a point on the lap, in degrees.

    The **secondary motion** — the layer that separates a model being dragged
    along a path from a car being driven. Roll comes from lateral acceleration
    leaning the body into the corner; pitch comes from longitudinal acceleration,
    diving under braking and squatting under power. Both are small (a stiff race
    car barely moves) and both are exactly what the eye reads as weight.

    Signs follow the scene frame: positive roll leans right, positive pitch
    noses down.
    """
    n = len(spine)
    total = lap_length(spine, closed)
    if total <= 0:
        return 0.0, 0.0

    speeds = speed_profile(spine, closed)
    radii = curvature_radius(spine, closed)

    # Nearest vertex to this distance.
    walked = 0.0
    index = 0
    for i in range(n):
        seg = math.dist(spine[i], spine[(i + 1) % n])
        if walked + seg >= (distance % total):
            index = i
            break
        walked += seg

    j = (index + 1) % n
    v = speeds[index]
    r = max(radii[index], 1.0)

    # Which way the corner turns, from the sign of the cross product.
    a, b, c = spine[(index - 1) % n], spine[index], spine[j]
    cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    turn_sign = 1.0 if cross < 0 else -1.0        # right-hand corner leans right

    lateral_g = (v * v / r) / G
    roll = turn_sign * min(lateral_g * 0.45, 3.0)

    seg = max(math.dist(spine[index], spine[j]), 1e-3)
    dv = speeds[j] - speeds[index]
    long_g = (speeds[j] ** 2 - speeds[index] ** 2) / (2.0 * seg * G)
    pitch = -max(-2.5, min(long_g * 0.5, 2.5))    # braking (negative dv) noses down

    return round(roll, 3), round(pitch, 3)


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


def field_at_time(spine, seconds: float, count: int = 20, *,
                  gap_s: float = 0.9, stagger_m: float = 3.0,
                  closed: bool = True):
    """
    The field at a moment in time, spaced by **time** rather than distance.

    Real gaps are quoted in seconds because that is what stays constant: cars
    holding a steady interval are 80 m apart on the straight and 20 m apart in
    the hairpin. Spacing by distance instead makes the pack concertina in
    exactly the wrong direction — it bunches on the straight and stretches in
    the corners, which is backwards and reads as such.

    Returns ``(x, y, heading, roll, pitch)`` per car, leader first.
    """
    lap_s = max(lap_time(spine, closed), 1e-3)

    def placed(car_index: int, at_seconds: float):
        """Where car ``car_index`` is, including its lateral offset."""
        distance = distance_at(spine, at_seconds % lap_s, closed)
        x, y, heading = point_at(spine, distance, closed=closed)
        side = 1.0 if car_index % 2 == 0 else -1.0
        a = math.radians(heading)
        nx, ny = math.cos(a), -math.sin(a)
        return x + nx * side * stagger_m, y + ny * side * stagger_m, distance

    out = []
    for i in range(count):
        t = seconds - i * gap_s
        x, y, distance = placed(i, t)

        # Heading from the car's **own** path, not the centre-line's.
        #
        # A car running 3 m off the centre-line through a 22 m corner is on a
        # different arc, and pointing it along the centre-line tangent leaves it
        # visibly crabbed — measured at up to 30 degrees off its direction of
        # travel at the apex, against 0.8 degrees on a straight. Differencing
        # its own position over a short interval gives the direction it is
        # actually going, which is what a heading is.
        # A **centred** difference, not a forward one. A forward chord points
        # where the car will be, which through a 22 m corner is several degrees
        # off where it is pointing now; centring the interval cancels that to
        # second order and costs one extra sample.
        back_x, back_y, _ = placed(i, t - 0.06)
        fwd_x, fwd_y, _ = placed(i, t + 0.06)
        dx, dy = fwd_x - back_x, fwd_y - back_y
        heading = (math.degrees(math.atan2(dx, dy)) % 360.0
                   if math.hypot(dx, dy) > 1e-6
                   else point_at(spine, distance, closed=closed)[2])

        roll, pitch = attitude_at(spine, distance, closed)
        out.append((x, y, heading, roll, pitch))
    return out


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


# ── the crash ────────────────────────────────────────────────────────────────
#
# Structured as setup -> anticipation -> impact -> follow-through, which is the
# ordinary grammar of an action beat and the thing that separates a collision
# from two models intersecting.
#
# **Anticipation is the part that is usually missing.** A car that is perfectly
# stable and then suddenly sideways reads as a glitch; a car whose rear steps
# out a few degrees over half a second before it lets go reads as a mistake
# being made. The tell is given ~0.6 s before contact, which is about how long a
# viewer needs to see it coming without having time to get bored.
#
# Contact itself is the shortest event in the sequence. Everything after it is
# follow-through: rotation continues because nothing stopped it, the car slides
# because it has no grip left, and it settles slowly because it is heavy.

@dataclass(frozen=True)
class CrashSpec:
    """
    One car losing it, and the car it collects.

    Distances are along the lap, so a crash is pinned to a place on the circuit
    rather than to a frame number — retime the sequence and it still happens at
    the same corner.
    """

    at_distance_m: float
    spinner_slot: int = 3            # the car that loses it
    victim_slot: int = 5             # the car it collects
    tell_s: float = 0.6              # anticipation before contact
    impact_s: float = 0.12           # contact itself
    settle_s: float = 3.4            # follow-through
    spin_degrees: float = 520.0      # total yaw the spinner accumulates
    launch_height_m: float = 0.9     # how far the victim gets pitched up


# Lap 0.69, which is inside the 11_crash_wide window (0.646-0.742).
CRASH = CrashSpec(at_distance_m=3650.0)


def crash_state(spec: CrashSpec, t_since_contact: float, base_heading: float):
    """
    Attitude and displacement for the spinning car, relative to its clean line.

    Returns ``(lateral_m, yaw_deg, roll_deg, pitch_deg, height_m)``.

    The curves are chosen for weight, not for symmetry. Yaw uses a fast-out
    decay -- most of the rotation happens in the first second while the car
    still has speed, then it winds down as it scrubs off -- and the vertical
    arc is a single parabola, because a car that leaves the ground is in free
    fall and free fall has exactly one shape.
    """
    if t_since_contact < -spec.tell_s:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if t_since_contact < 0.0:
        # Anticipation: the rear steps out, the body takes a set.
        f = (t_since_contact + spec.tell_s) / spec.tell_s      # 0 -> 1
        ease = f * f                                            # slow, then away
        return (0.9 * ease, -9.0 * ease, 1.6 * ease, -0.4 * ease, 0.0)

    if t_since_contact < spec.impact_s:
        # Impact: violent and brief. Nothing here is smooth.
        f = t_since_contact / spec.impact_s
        return (0.9 + 2.4 * f, -9.0 - 55.0 * f, 1.6 + 5.0 * f, -2.2 * f,
                spec.launch_height_m * 0.35 * f)

    # Follow-through.
    f = min((t_since_contact - spec.impact_s) / spec.settle_s, 1.0)
    decay = 1.0 - (1.0 - f) ** 3          # fast at first, then settling
    yaw = -64.0 - (spec.spin_degrees - 64.0) * decay
    lateral = 3.3 + 7.5 * decay
    roll = 6.6 * (1.0 - f) ** 2           # rights itself as it slows
    pitch = -2.2 * (1.0 - f) ** 2

    # One parabola, thrown at impact and landed a third of the way through.
    air = max(0.0, 1.0 - (t_since_contact / (spec.settle_s * 0.33)))
    height = spec.launch_height_m * 4.0 * air * (1.0 - air)

    return lateral, yaw, roll, pitch, height


@dataclass(frozen=True)
class Shot:
    """
    One camera setup, and how long it is on screen.

    **Lap time and screen time are different axes, and conflating them was a
    bug.** The lap window says what part of the circuit the camera watches;
    ``duration_s`` says how long the cut runs. Deriving the second from the
    first meant every locked-off shot — where ``lap_from == lap_to`` — came out
    one frame long, which is 0.04 s and unusable, and it made two cameras
    covering one incident look like an overlap error when it is in fact how a
    broadcast plays a crash.

    So lap windows **may** overlap. Screen slots may not, and
    :func:`build_edit` enforces that.

    ``at`` is called with ``(spine, lap_fraction, site_z)`` and returns
    ``(position, target, fov)`` — a function rather than a start/end pair so a
    move can be an arc or a rise rather than only a straight line.
    """

    name: str
    kind: str                 # "drone" | "static"
    lap_from: float
    lap_to: float
    at: object = field(repr=False)
    note: str = ""


@dataclass(frozen=True)
class Cut:
    """One shot's slot on the finished timeline."""

    shot: Shot
    start_frame: int
    end_frame: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame

    def lap_at(self, frame: int) -> float:
        """The lap fraction this shot is watching at a given timeline frame."""
        if self.frames <= 0:
            return self.shot.lap_from
        f = (frame - self.start_frame) / self.frames
        return self.shot.lap_from + (self.shot.lap_to - self.shot.lap_from) * f


def build_edit(shots=None, *, fps: int = 24, lap_seconds: float = 80.7,
               min_frames: int = 12) -> list[Cut]:
    """
    Map each shot's lap window onto the timeline and return the cut list.

    **The timeline is the lap.** Screen time is not a free parameter: the cars
    are running one continuous lap, so a shot covering lap 0.24 to 0.33 occupies
    exactly the frames during which the field is between those points, and the
    shots must tile 0 to 1 without gap or overlap.

    Two earlier versions of this got it wrong in opposite directions. The first
    derived screen time from the lap window and gave every locked-off shot one
    frame, because a static shot had a zero-width window. The second gave each
    shot its own independent duration — which fixed the one-frame cuts and broke
    something worse: consecutive cuts watched non-adjacent parts of the circuit,
    so the cars *teleported* at every boundary, sliding 300-500 m between two
    keys. Measured in the scene, not guessed: a car moved 540 m between samples
    60 frames apart.

    A static shot therefore needs a lap window with real width. It is locked off
    in space, not in time; the world still moves through it.

    Raises on a gap, an overlap, or a cut shorter than ``min_frames``, because
    all three are silent in a render and obvious here.
    """
    shots = list(shots if shots is not None else SHOTS)
    total = int(round(lap_seconds * fps))

    cuts: list[Cut] = []
    for index, shot in enumerate(shots):
        if shot.lap_to <= shot.lap_from:
            raise ValueError(
                f"{shot.name}: lap window {shot.lap_from}->{shot.lap_to} has no "
                "width. A locked-off camera still needs time to pass through it."
            )
        if index and abs(shot.lap_from - shots[index - 1].lap_to) > 1e-9:
            raise ValueError(
                f"{shot.name} starts at {shot.lap_from} but "
                f"{shots[index - 1].name} ended at {shots[index - 1].lap_to}; "
                "cuts must tile the lap, or the cars jump between them"
            )
        start = int(round(shot.lap_from * total))
        end = int(round(shot.lap_to * total))
        if end - start < min_frames:
            raise ValueError(
                f"{shot.name}: {end - start} frames at {fps} fps, under the "
                f"{min_frames}-frame floor"
            )
        cuts.append(Cut(shot=shot, start_frame=start, end_frame=end))
    return cuts


def edit_length_frames(cuts) -> int:
    """Total timeline length of an assembled edit."""
    return max((c.end_frame for c in cuts), default=0)


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
    Shot("01_grid_low", "static", 0.0, 0.022,
         lambda s, t, z, u: _trackside(s, 0.0, -1.0, 11.0, 1.1, z, fov=40.0),
         "grid, low and wide from the sunlit side"),
    Shot("02_grid_rise", "drone", 0.022, 0.058,
         lambda s, t, z, u: ((_lead(s, t)[0], _lead(s, t)[1] - 40.0,
                             z + 6.0 + 58.0 * u),
                            (_lead(s, t)[0], _lead(s, t)[1], z), 46.0),
         "rise off the grid as the field launches"),
    Shot("03_t1_static", "static", 0.058, 0.092,
         lambda s, t, z, u: _trackside(s, t, 1.0, 26.0, 3.2, z, fov=34.0),
         "locked off at the first corner"),
    Shot("04_chase_back", "drone", 0.092, 0.17,
         lambda s, t, z, u: _behind(s, t, 15.0, 2.5, 2.2, z, 22.0, 40.0),
         "low chase, close behind the leader"),
    Shot("05_skim", "drone", 0.17, 0.24,
         lambda s, t, z, u: _behind(s, t, 9.0, -6.0, 0.9, z, 26.0, 34.0),
         "skimming the surface alongside"),
    Shot("06_hairpin_orbit", "drone", 0.24, 0.33,
         lambda s, t, z, u: _orbit(s, t, 48.0, 22.0, 6.0, z, fov=38.0),
         "orbiting the hairpin"),
    # On the circuit's own axis, not off to the side. The first version stood
    # 95 m out to one side with a 17 degree lens and rendered pure black -- at
    # that distance it was inside a building, and a camera inside geometry sees
    # the unlit backs of its faces. A compressed telephoto down a straight is an
    # on-axis shot anyway: the long lens is what stacks the field up, and it
    # only stacks if you are looking along the line they are running.
    Shot("07_long_lens", "static", 0.33, 0.4,
         lambda s, t, z, u: _behind(s, t, 240.0, 1.5, 2.4, z, 30.0, 13.0),
         "long lens down the straight, field compressed"),
    Shot("08_high_wide", "drone", 0.4, 0.5,
         lambda s, t, z, u: ((_lead(s, t)[0] - 150.0, _lead(s, t)[1] - 190.0, z + 130.0),
                          (_lead(s, t)[0], _lead(s, t)[1], z), 50.0),
         "high and wide, circuit in context"),
    Shot("09_marina", "static", 0.5, 0.58,
         lambda s, t, z, u: _trackside(s, t, 1.0, 34.0, 8.0, z, fov=40.0),
         "elevated static over the marina section"),
    Shot("10_low_front", "static", 0.58, 0.646,
         lambda s, t, z, u: (lambda p: ((p[0] + math.sin(math.radians(p[2])) * 30.0,
                                      p[1] + math.cos(math.radians(p[2])) * 30.0,
                                      z + 0.75),
                                     (p[0], p[1], z + 0.6), 30.0))(_lead(s, t)),
         "head-on, tyre height, cars coming at the lens"),
    # ── the crash, covered twice ────────────────────────────────────────────
    # Two angles on one event, which is how a real broadcast plays an incident:
    # the wide sees what happened, the tight sees it happen. Both bracket the
    # contact so the anticipation is on screen -- cutting in *on* the impact
    # throws away the half second that makes it read as a mistake rather than a
    # glitch.
    Shot("11_crash_wide", "drone", 0.646, 0.742,
         lambda s, t, z, u: _behind(s, t, 62.0, 14.0, 11.0, z, 34.0, 40.0),
         "wide on the incident: the tell, the contact, the slide"),
    Shot("12_crash_tight", "static", 0.742, 0.82,
         lambda s, t, z, u: _trackside(s, t, 1.0, 21.0, 1.6, z, fov=26.0),
         "trackside and tight, level with the impact"),
    Shot("13_pullback", "drone", 0.82, 0.916,
         lambda s, t, z, u: _behind(s, t, 30.0 + 340.0 * u,
                                    0.0, 12.0 + 150.0 * u, z, 30.0, 46.0),
         "pull back and up, revealing the lap"),
    Shot("14_finish", "drone", 0.916, 1.0,
         lambda s, t, z, u: ((_lead(s, t)[0] + 26.0, _lead(s, t)[1] - 30.0, z + 16.0),
                          (_lead(s, t)[0], _lead(s, t)[1], z + 0.5), 40.0),
         "over the line to finish the lap"),
]


def camera_for(shot: Shot, spine, lap_fraction: float, site_z: float):
    """
    Resolve a shot to ``(position, target, fov)`` at one moment.

    The shot's callable receives ``(spine, lap_fraction, site_z, u)`` where
    ``u`` is progress through its **own** window, 0 at the cut in and 1 at the
    cut out. That fourth argument exists because two shots used to derive their
    move from hardcoded lap numbers — ``(t - 0.84) / 0.04`` — and when the
    windows were retiled those constants went stale silently. One of them
    computed a negative height and put the camera underground, which the
    below-site guard caught only at build time. A shot should not have to know
    where it sits in the lap.
    """
    span = shot.lap_to - shot.lap_from
    u = 0.0 if span <= 0 else max(0.0, min(1.0, (lap_fraction - shot.lap_from) / span))
    return shot.at(spine, lap_fraction, site_z, u)
