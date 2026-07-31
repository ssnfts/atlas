"""
Lap motion tests.

Motion fails the way geometry does here: silently and plausibly. A speed profile
that is subtly wrong still produces cars going round a circuit, and a crash with
no anticipation still produces a car that ends up sideways. So these pin the
properties that separate driving from sliding along a path.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import raceanim as ra  # noqa: E402


def _oval(straight=500.0, radius=45.0, arc_n=40, straight_n=24):
    """
    A closed circuit with two sampled straights and two known-radius ends.

    The straights are **sampled**, not jumped in a single segment. The first
    version of this fixture placed vertices only on the arcs and joined them
    with one long edge, so no vertex ever sat mid-straight — every curvature
    reading came off an arc, and four tests failed against perfectly good code.
    A fixture that cannot express the thing under test fails the code for its
    own reasons.
    """
    pts = []
    half = straight / 2.0
    for i in range(arc_n):                      # top arc: (+r, half) -> (-r, half)
        a = math.pi * i / arc_n
        pts.append((radius * math.cos(a), half + radius * math.sin(a)))
    for i in range(straight_n):                 # left straight, running -y
        f = i / straight_n
        pts.append((-radius, half - straight * f))
    for i in range(arc_n):                      # bottom arc: (-r, -half) -> (+r, -half)
        a = math.pi + math.pi * i / arc_n
        pts.append((radius * math.cos(a), -half + radius * math.sin(a)))
    for i in range(straight_n):                 # right straight, running +y
        f = i / straight_n
        pts.append((radius, -half + straight * f))
    return pts


def test_curvature_matches_a_circuit_of_known_radius():
    """
    Built at a known 45 m radius, so the answer is checkable against something
    other than the function itself. This caught the circumradius being computed
    as abc/(2A) instead of abc/(4A) -- every radius exactly double, every corner
    apparently takeable at sqrt(2) times its real speed, nothing raised.
    """
    spine = _oval(radius=45.0)
    radii = ra.curvature_radius(spine)
    assert max(radii) > 1000.0                          # the sampled straights
    assert min(radii) == pytest.approx(45.0, rel=0.08)  # the arcs


def test_speed_is_lower_in_the_corners_than_on_the_straights():
    """The whole point: constant speed is what this replaced."""
    spine = _oval()
    speeds = ra.speed_profile(spine)
    assert max(speeds) > min(speeds) * 1.5


def test_braking_happens_before_the_corner_not_in_it():
    """
    A car must already be slow when it arrives, which means the vertex before a
    slow one cannot be at full speed. This is the backward pass; without it the
    car brakes instantaneously at the corner entry.
    """
    spine = _oval()
    speeds = ra.speed_profile(spine)
    slowest = speeds.index(min(speeds))
    approach = [speeds[(slowest - k) % len(speeds)] for k in range(1, 6)]
    # Tolerance because the cyclic sweeps converge to within float noise on a
    # constant-radius arc; the property under test is the trend, not equality.
    for earlier, later in zip(approach, approach[1:]):
        assert later >= earlier - 1e-6, "speed should fall into the corner"


def test_speed_never_exceeds_the_grip_limit():
    spine = _oval(radius=80.0)
    speeds = ra.speed_profile(spine)
    radii = ra.curvature_radius(spine)
    for v, r in zip(speeds, radii):
        limit = math.sqrt(ra.LAT_G * ra.G * r)
        assert v <= max(limit, ra.V_MIN) + 1e-6


def test_distance_at_advances_and_wraps():
    spine = _oval()
    total = ra.lap_length(spine)
    t = ra.lap_time(spine)
    assert ra.distance_at(spine, 0.0) == pytest.approx(0.0)
    assert ra.distance_at(spine, t) == pytest.approx(total, rel=0.02)
    assert ra.distance_at(spine, t / 2) < total


def test_lap_time_is_length_over_a_plausible_mean_speed():
    spine = _oval()
    t = ra.lap_time(spine)
    mean = ra.lap_length(spine) / t
    assert min(ra.speed_profile(spine)) <= mean <= max(ra.speed_profile(spine))


def test_field_gaps_are_constant_in_time_not_distance():
    """
    Cars holding a steady interval stretch on the straight and close up in the
    corners. Spacing by distance does the opposite, which is the bug this
    replaced.
    """
    spine = _oval()
    field = ra.field_at_time(spine, 12.0, count=8, gap_s=0.9, stagger_m=0.0)
    gaps = [math.dist(field[i][:2], field[i + 1][:2]) for i in range(7)]
    assert max(gaps) > min(gaps) * 1.2


def test_attitude_leans_into_a_corner_and_is_flat_on_a_straight():
    spine = _oval()
    radii = ra.curvature_radius(spine)
    corner = radii.index(min(radii))
    straight = radii.index(max(radii))
    walk = [0.0]
    for i in range(len(spine) - 1):
        walk.append(walk[-1] + math.dist(spine[i], spine[i + 1]))
    corner_roll, _ = ra.attitude_at(spine, walk[corner])
    straight_roll, _ = ra.attitude_at(spine, walk[straight])
    assert abs(corner_roll) > abs(straight_roll)


# ── the crash ────────────────────────────────────────────────────────────────

def test_nothing_happens_before_the_tell():
    state = ra.crash_state(ra.CRASH, -ra.CRASH.tell_s - 0.1, 0.0)
    assert state == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_anticipation_builds_before_contact():
    """
    The half second that makes an impact read as a mistake rather than a glitch.
    Yaw must already be non-zero, and growing, before t=0.
    """
    early = ra.crash_state(ra.CRASH, -0.45, 0.0)[1]
    late = ra.crash_state(ra.CRASH, -0.10, 0.0)[1]
    assert abs(late) > abs(early) > 0.0


def test_impact_is_the_most_violent_part():
    """Yaw rate during contact must exceed both the tell and the follow-through."""
    def rate(a, b):
        return abs(ra.crash_state(ra.CRASH, b, 0)[1] -
                   ra.crash_state(ra.CRASH, a, 0)[1]) / (b - a)

    assert rate(0.0, ra.CRASH.impact_s) > rate(-0.3, 0.0)
    assert rate(0.0, ra.CRASH.impact_s) > rate(1.0, 1.6)


def test_spin_reaches_the_specified_total():
    yaw = ra.crash_state(ra.CRASH, ra.CRASH.impact_s + ra.CRASH.settle_s, 0.0)[1]
    assert yaw == pytest.approx(-ra.CRASH.spin_degrees, rel=1e-6)


def test_the_car_lands():
    """A parabola, thrown at impact. It must come back down and stay down."""
    peak = max(ra.crash_state(ra.CRASH, t / 20.0, 0.0)[4] for t in range(0, 30))
    assert peak > 0.3
    assert ra.crash_state(ra.CRASH, ra.CRASH.settle_s, 0.0)[4] == pytest.approx(0.0)


def test_body_rights_itself_as_it_slows():
    roll_early = abs(ra.crash_state(ra.CRASH, 0.3, 0.0)[2])
    roll_late = abs(ra.crash_state(ra.CRASH, 3.0, 0.0)[2])
    assert roll_early > roll_late


def test_the_cut_list_covers_the_crash_from_two_angles():
    crash = [s for s in ra.SHOTS if "crash" in s.name]
    assert len(crash) == 2
    for shot in crash:
        assert shot.lap_from < shot.lap_to or shot.kind == "static"


# ── orientation: the convention, pinned ──────────────────────────────────────

def test_nose_points_where_the_heading_says():
    """
    The bug this locks: Max's Euler path produced the *mirrored* yaw. A car
    keyed at heading 90 pointed west. Headings are clockwise from +Y, so the
    nose must land on (sin h, cos h).
    """
    for h in (0.0, 45.0, 90.0, 137.0, 180.0, 198.0, 270.0, 355.0):
        nose = ra.nose_direction(ra.orientation_quat(h))
        assert nose[0] == pytest.approx(math.sin(math.radians(h)), abs=1e-9)
        assert nose[1] == pytest.approx(math.cos(math.radians(h)), abs=1e-9)
        assert nose[2] == pytest.approx(0.0, abs=1e-9)


def test_roll_is_about_the_nose_not_the_world():
    """
    The second bug: pitch and roll were applied about world axes, so a roll at
    heading 90 came out as pitch. Rolling about the nose cannot move the nose,
    at any heading.
    """
    for h in (0.0, 90.0, 198.0, 270.0):
        clean = ra.nose_direction(ra.orientation_quat(h))
        rolled = ra.nose_direction(ra.orientation_quat(h, 0.0, 4.0))
        assert math.dist(clean, rolled) < 1e-9


def test_pitch_lifts_or_drops_the_nose_at_any_heading():
    for h in (0.0, 90.0, 198.0, 270.0):
        up = ra.nose_direction(ra.orientation_quat(h, 5.0, 0.0))
        assert up[2] > 0.05, f"pitch did nothing vertical at heading {h}"


def test_quaternions_stay_unit_length():
    for h in (0.0, 123.0, 359.0):
        q = ra.orientation_quat(h, 2.0, -3.0)
        assert math.sqrt(sum(c * c for c in q)) == pytest.approx(1.0, abs=1e-12)


# ── the edit ─────────────────────────────────────────────────────────────────

def test_every_cut_gets_real_screen_time():
    """
    The sequencer bug: screen time was derived from the lap window, so a
    locked-off shot (lap_from == lap_to) came out one frame long — 0.04 s.
    """
    for cut in ra.build_edit(fps=24):
        assert cut.frames >= 12, f"{cut.shot.name} is only {cut.frames} frames"


def test_cuts_tile_the_timeline_without_gaps_or_overlap():
    cuts = ra.build_edit(fps=24)
    assert cuts[0].start_frame == 0
    for earlier, later in zip(cuts, cuts[1:]):
        assert later.start_frame == earlier.end_frame


def test_the_crash_is_covered_by_two_consecutive_cuts():
    """
    Two angles, adjacent rather than overlapping. An earlier design let the two
    crash cameras share a lap window, which meant consecutive cuts watched the
    same moment — and since the cars run one continuous lap, that made them jump
    backwards at the cut. The wide takes the approach and the contact, the tight
    takes the slide.
    """
    crash = [s for s in ra.SHOTS if "crash" in s.name]
    assert len(crash) == 2
    wide, tight = crash
    assert wide.lap_to == pytest.approx(tight.lap_from)
    crash_lap = ra.CRASH.at_distance_m / 5289.5
    assert wide.lap_from < crash_lap < tight.lap_to, "the incident must be inside the coverage"


def test_the_lap_is_tiled_with_no_gap_or_overlap():
    shots = ra.SHOTS
    assert shots[0].lap_from == pytest.approx(0.0)
    assert shots[-1].lap_to == pytest.approx(1.0)
    for earlier, later in zip(shots, shots[1:]):
        assert later.lap_from == pytest.approx(earlier.lap_to)


def test_a_gap_between_shots_is_refused():
    """A gap means the cars jump; better to fail here than in a render."""
    a = ra.Shot("a", "static", 0.0, 0.3, lambda s, t, z: None)
    b_ = ra.Shot("b", "static", 0.5, 1.0, lambda s, t, z: None)   # gap at 0.3-0.5
    with pytest.raises(ValueError, match="tile the lap"):
        ra.build_edit([a, b_], fps=24)


def test_a_zero_width_lap_window_is_refused():
    """
    A locked-off camera is fixed in space, not in time. Giving it a zero-width
    window is what produced 1-frame cuts.
    """
    bad = ra.Shot("bad", "static", 0.5, 0.5, lambda s, t, z: None)
    with pytest.raises(ValueError, match="no width"):
        ra.build_edit([bad], fps=24)


def test_cut_reports_the_lap_moment_it_is_watching():
    cuts = ra.build_edit(fps=24)
    moving = next(c for c in cuts if c.shot.lap_to > c.shot.lap_from)
    assert moving.lap_at(moving.start_frame) == pytest.approx(moving.shot.lap_from)
    mid = moving.lap_at((moving.start_frame + moving.end_frame) // 2)
    assert moving.shot.lap_from < mid < moving.shot.lap_to
