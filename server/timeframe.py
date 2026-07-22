"""
Wall-clock time + location -> UTC.

Kept separate from ``solar.py`` on purpose. The astronomy is pure and testable;
timezone resolution depends on an external database and on ambiguous real-world
rules. Mixing them would let a timezone failure masquerade as an astronomy bug.

Two failure modes are handled loudly rather than silently, because both produce
a *plausible* wrong answer rather than an obvious one:

1. **Missing tzdata.** Windows ships no IANA database, so ``zoneinfo`` raises
   for every zone name unless the ``tzdata`` package is installed. Left
   unchecked this tends to get "fixed" with a fallback to UTC, which silently
   shifts the sun by up to 12 hours.
2. **DST transitions.** 02:30 on a spring-forward date does not exist; 01:30 on
   a fall-back date happens twice. Python resolves both silently by default.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = ["ResolvedTime", "resolve_local_time", "tz_for_location", "assert_tzdata"]


class TimeframeError(Exception):
    """Raised when a wall-clock time cannot be resolved unambiguously."""


def assert_tzdata() -> None:
    """
    Fail fast if the IANA database is unreachable.

    Called at server start. The cost of this check is microseconds; the cost of
    not having it is a rendered sequence lit for the wrong hemisphere-hour.
    """
    try:
        ZoneInfo("Asia/Dubai")
    except ZoneInfoNotFoundError as exc:  # pragma: no cover - environment-dependent
        raise TimeframeError(
            "No IANA timezone database available. On Windows this means the "
            "'tzdata' package is not installed in this interpreter. Install it "
            "with: pip install tzdata"
        ) from exc


@functools.lru_cache(maxsize=1)
def _finder():
    try:
        from timezonefinder import TimezoneFinder
    except ImportError as exc:  # pragma: no cover
        raise TimeframeError(
            "timezonefinder is not installed; cannot map coordinates to a "
            "timezone. Either install it or pass an explicit IANA zone name."
        ) from exc
    return TimezoneFinder()


def tz_for_location(latitude: float, longitude: float) -> str:
    """
    IANA zone name for a coordinate, e.g. ``Asia/Dubai``.

    Returns the nearest zone for points at sea, where the polygon lookup misses.
    """
    tf = _finder()
    name = tf.timezone_at(lat=latitude, lng=longitude)
    if name is None:
        name = tf.certain_timezone_at(lat=latitude, lng=longitude)
    if name is None:
        raise TimeframeError(
            f"Could not determine a timezone for ({latitude}, {longitude}). "
            "Pass an explicit IANA zone name."
        )
    return name


@dataclass(frozen=True)
class ResolvedTime:
    """A wall-clock time pinned to a real instant."""

    utc: datetime
    local: datetime
    tz_name: str
    utc_offset_hours: float
    is_dst: bool
    note: str | None = None  # set when a DST ambiguity had to be resolved

    def as_dict(self) -> dict:
        return {
            "utc": self.utc.isoformat(),
            "local": self.local.isoformat(),
            "timezone": self.tz_name,
            "utc_offset_hours": self.utc_offset_hours,
            "is_dst": self.is_dst,
            **({"note": self.note} if self.note else {}),
        }


def resolve_local_time(
    local_naive: datetime,
    latitude: float,
    longitude: float,
    tz_name: str | None = None,
    *,
    dst_ambiguous: str = "first",
) -> ResolvedTime:
    """
    Interpret a naive wall-clock datetime as local time at a location.

    Args:
        local_naive: naive datetime, i.e. what a clock on the wall there reads.
        latitude, longitude: used to look up the zone when ``tz_name`` is None.
        tz_name: explicit IANA zone, skipping the coordinate lookup.
        dst_ambiguous: which instant to pick for a repeated wall-clock hour --
            ``"first"`` (still on DST) or ``"second"`` (after falling back).

    Raises:
        TimeframeError: if the wall-clock time does not exist because the clocks
            sprang forward across it. This is deliberately not silently shifted:
            if a shoot log says 02:30 on a spring-forward date, that log is
            wrong and the operator needs to know.
    """
    assert_tzdata()

    if local_naive.tzinfo is not None:
        raise TimeframeError(
            "resolve_local_time expects a naive datetime (wall-clock reading). "
            "Pass tz-aware datetimes straight to solar_position_utc instead."
        )

    zone_name = tz_name or tz_for_location(latitude, longitude)
    try:
        zone = ZoneInfo(zone_name)
    except ZoneInfoNotFoundError as exc:
        raise TimeframeError(f"Unknown timezone: {zone_name!r}") from exc

    note = None
    aware = local_naive.replace(tzinfo=zone)

    # Nonexistent time: clocks sprang forward over this reading. Detect by
    # round-tripping through UTC -- a real time survives, a skipped one does not.
    round_trip = aware.astimezone(timezone.utc).astimezone(zone)
    if round_trip.replace(tzinfo=None) != local_naive:
        raise TimeframeError(
            f"{local_naive.isoformat()} does not exist in {zone_name}: the "
            f"clocks moved forward across it (it resolves to "
            f"{round_trip.replace(tzinfo=None).isoformat()}). Check the source "
            "timestamp."
        )

    # Ambiguous time: this wall-clock reading occurs twice. fold=0 is the first
    # (pre-transition) pass, fold=1 the second.
    first = aware.replace(fold=0)
    second = aware.replace(fold=1)
    if first.utcoffset() != second.utcoffset():
        chosen = first if dst_ambiguous == "first" else second
        note = (
            f"{local_naive.isoformat()} occurs twice in {zone_name} "
            f"(DST fall-back); used the {dst_ambiguous} occurrence "
            f"(UTC{_fmt_offset(chosen.utcoffset())})."
        )
        aware = chosen

    offset = aware.utcoffset() or timedelta(0)
    dst = aware.dst() or timedelta(0)

    return ResolvedTime(
        utc=aware.astimezone(timezone.utc),
        local=aware,
        tz_name=zone_name,
        utc_offset_hours=offset.total_seconds() / 3600.0,
        is_dst=dst.total_seconds() != 0,
        note=note,
    )


def _fmt_offset(delta: timedelta | None) -> str:
    total = int((delta or timedelta(0)).total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"
