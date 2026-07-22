"""
Solar position verification.

Three independent layers of evidence, deliberately not circular:

1. **Definitional invariants** — declination at equinox/solstice is fixed by the
   definition of those events, not by any implementation.
2. **Exact geometry** — altitude at solar noon must equal ``90 - |lat - decl|``.
   Derivable on paper; no algorithm involved.
3. **Independent algorithm** — cross-check against pvlib's NREL SPA, which is a
   different algorithm by different authors. Agreement between NOAA and SPA is
   real evidence; checking NOAA against itself would not be.

Runs with no 3ds Max and no network.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from solar import solar_position_utc, sun_vector  # noqa: E402
from timeframe import TimeframeError, resolve_local_time  # noqa: E402


# Obliquity of the ecliptic in the 2020s. Solstice declination equals this.
OBLIQUITY = 23.4392

SITES = {
    "dubai": (25.2048, 55.2708),
    "london": (51.5074, -0.1278),
    "sydney": (-33.8688, 151.2093),
    "reykjavik": (64.1466, -21.9426),
    "tromso": (69.6492, 18.9553),      # above the Arctic Circle
    "quito": (-0.1807, -78.4678),
    "equator_prime": (0.0, 0.0),
}


# ── Layer 1: definitional invariants ──────────────────────────────────────────

@pytest.mark.parametrize(
    "instant,expected_decl",
    [
        (datetime(2024, 3, 20, 3, 6, tzinfo=timezone.utc), 0.0),
        (datetime(2024, 6, 20, 20, 51, tzinfo=timezone.utc), OBLIQUITY),
        (datetime(2024, 9, 22, 12, 44, tzinfo=timezone.utc), 0.0),
        (datetime(2024, 12, 21, 9, 20, tzinfo=timezone.utc), -OBLIQUITY),
    ],
    ids=["mar_equinox", "jun_solstice", "sep_equinox", "dec_solstice"],
)
def test_declination_at_equinox_and_solstice(instant, expected_decl):
    """Declination is ~0 at equinox and ~+-obliquity at solstice, by definition."""
    pos = solar_position_utc(instant, 0.0, 0.0)
    assert abs(pos.declination - expected_decl) < 0.01


# ── Layer 2: exact geometry ───────────────────────────────────────────────────

def _solar_noon(day_utc: datetime, lat: float, lon: float):
    """Highest sun of the day, found by scan then refined to the minute."""
    best = max(
        (solar_position_utc(day_utc + timedelta(minutes=m), lat, lon) for m in range(1440)),
        key=lambda p: p.altitude_true,
    )
    return best


@pytest.mark.parametrize("site", ["dubai", "london", "sydney", "quito", "reykjavik"])
@pytest.mark.parametrize("month,day", [(6, 21), (12, 21), (3, 20)])
def test_solar_noon_altitude_matches_geometry(site, month, day):
    """
    At solar noon: altitude == 90 - |latitude - declination|.

    Pure spherical geometry, independent of any solar algorithm.
    """
    lat, lon = SITES[site]
    pos = _solar_noon(datetime(2024, month, day, tzinfo=timezone.utc), lat, lon)
    predicted = 90.0 - abs(lat - pos.declination)
    assert abs(pos.altitude_true - predicted) < 0.02, (
        f"{site} {month}/{day}: got {pos.altitude_true:.4f}, predicted {predicted:.4f}"
    )


@pytest.mark.parametrize("site", ["dubai", "london", "reykjavik"])
def test_solar_noon_azimuth_is_south_in_northern_hemisphere(site):
    """Northern sites north of the tropics see the midday sun due south."""
    lat, lon = SITES[site]
    pos = _solar_noon(datetime(2024, 12, 21, tzinfo=timezone.utc), lat, lon)
    assert abs(pos.azimuth - 180.0) < 1.0


def test_solar_noon_azimuth_is_north_in_southern_hemisphere():
    lat, lon = SITES["sydney"]
    pos = _solar_noon(datetime(2024, 6, 21, tzinfo=timezone.utc), lat, lon)
    # Due north is 0/360 -- accept either wrap.
    assert min(pos.azimuth, abs(360.0 - pos.azimuth)) < 1.0


# ── Layer 3: independent algorithm (pvlib NREL SPA) ───────────────────────────

@pytest.mark.parametrize("site", list(SITES))
@pytest.mark.parametrize("hour", [3, 6, 9, 12, 15, 18, 21])
@pytest.mark.parametrize("month", [1, 4, 7, 10])
def test_agrees_with_nrel_spa(site, hour, month):
    """
    Cross-check against pvlib's NREL SPA implementation.

    The comparison is split by quantity, because only one of the two is an
    astronomical claim:

    * **Geometric altitude** is pure astronomy. NOAA and SPA are independent
      algorithms and must agree tightly -- 0.01 deg, which is NOAA's own stated
      accuracy for 1800-2100.
    * **Apparent altitude** additionally embeds a choice of refraction model.
      pvlib clamps refraction to zero below the horizon while the NOAA
      polynomial keeps applying it, so the two diverge by up to ~0.24 deg once
      the sun has set. That divergence is a modelling convention, not an error,
      and it is irrelevant for lighting: there is no direct sun below the
      horizon. It is therefore only asserted where it matters -- above it.
    """
    pvlib = pytest.importorskip("pvlib")
    import pandas as pd

    lat, lon = SITES[site]
    instant = datetime(2024, month, 15, hour, 0, tzinfo=timezone.utc)

    ours = solar_position_utc(instant, lat, lon)
    ref = pvlib.solarposition.spa_python(
        pd.DatetimeIndex([instant]), latitude=lat, longitude=lon
    )
    ref_true = float(ref["elevation"].iloc[0])
    ref_app = float(ref["apparent_elevation"].iloc[0])
    ref_az = float(ref["azimuth"].iloc[0])

    # The astronomical claim -- tight bound, everywhere.
    assert abs(ours.altitude_true - ref_true) < 0.01, (
        f"{site} {month}/15 {hour}h: geometric altitude {ours.altitude_true:.5f} "
        f"vs SPA {ref_true:.5f}"
    )

    # The lighting-relevant claim -- only while the sun is actually up.
    if ref_app > 0.0:
        assert abs(ours.altitude - ref_app) < 0.05, (
            f"{site} {month}/15 {hour}h: apparent altitude {ours.altitude:.4f} "
            f"vs SPA {ref_app:.4f}"
        )

    if ref_app > 5.0:
        delta_az = abs(ours.azimuth - ref_az)
        delta_az = min(delta_az, 360.0 - delta_az)  # wrap at north
        assert delta_az < 0.05, (
            f"{site} {month}/15 {hour}h: azimuth {ours.azimuth:.4f} vs SPA {ref_az:.4f}"
        )


# ── Polar edge cases ──────────────────────────────────────────────────────────

def test_midnight_sun_above_arctic_circle():
    """Tromso (69.65N) has 24-hour daylight at the June solstice."""
    lat, lon = SITES["tromso"]
    day = datetime(2024, 6, 21, tzinfo=timezone.utc)
    altitudes = [
        solar_position_utc(day + timedelta(minutes=m), lat, lon).altitude
        for m in range(0, 1440, 10)
    ]
    assert min(altitudes) > 0.0, f"sun set during polar day; min={min(altitudes):.3f}"


def test_polar_night_above_arctic_circle():
    """Tromso has no sunrise at the December solstice."""
    lat, lon = SITES["tromso"]
    day = datetime(2024, 12, 21, tzinfo=timezone.utc)
    altitudes = [
        solar_position_utc(day + timedelta(minutes=m), lat, lon).altitude
        for m in range(0, 1440, 10)
    ]
    assert max(altitudes) < 0.0, f"sun rose during polar night; max={max(altitudes):.3f}"


def test_no_domain_error_at_the_poles():
    """acos domain clamping holds at exactly +-90 latitude."""
    for lat in (90.0, -90.0):
        pos = solar_position_utc(datetime(2024, 6, 21, 12, tzinfo=timezone.utc), lat, 0.0)
        assert math.isfinite(pos.altitude)
        assert math.isfinite(pos.azimuth)


# ── Refraction ────────────────────────────────────────────────────────────────

def test_refraction_lifts_the_sun_near_the_horizon():
    """
    Apparent altitude exceeds true altitude near the horizon by roughly half a
    degree -- the reason the sun is still visible when geometrically set.
    """
    lat, lon = SITES["london"]
    day = datetime(2024, 3, 20, tzinfo=timezone.utc)
    near_horizon = [
        p
        for m in range(1440)
        if -0.5 < (p := solar_position_utc(day + timedelta(minutes=m), lat, lon)).altitude_true < 0.5
    ]
    assert near_horizon, "no near-horizon samples found"
    for pos in near_horizon:
        lift = pos.altitude - pos.altitude_true
        assert 0.4 < lift < 0.7, f"refraction lift {lift:.4f} deg outside expected band"


def test_refraction_negligible_at_zenith():
    pos = solar_position_utc(datetime(2024, 6, 21, 7, 0, tzinfo=timezone.utc), 23.44, 60.0)
    if pos.altitude_true > 85.0:
        assert abs(pos.altitude - pos.altitude_true) < 0.001


# ── Scene-frame vector ────────────────────────────────────────────────────────

def test_sun_vector_axis_conventions():
    """+Y is true north, +X is east, +Z is up."""
    east = sun_vector(90.0, 0.0)
    assert east[0] == pytest.approx(1.0, abs=1e-9)
    assert east[1] == pytest.approx(0.0, abs=1e-9)

    north = sun_vector(0.0, 0.0)
    assert north[1] == pytest.approx(1.0, abs=1e-9)
    assert north[0] == pytest.approx(0.0, abs=1e-9)

    south = sun_vector(180.0, 0.0)
    assert south[1] == pytest.approx(-1.0, abs=1e-9)

    west = sun_vector(270.0, 0.0)
    assert west[0] == pytest.approx(-1.0, abs=1e-9)

    overhead = sun_vector(123.0, 90.0)
    assert overhead[2] == pytest.approx(1.0, abs=1e-9)


def test_sun_vector_is_unit_length():
    for az in range(0, 360, 17):
        for alt in (-10, 0, 12, 45, 89):
            v = sun_vector(float(az), float(alt))
            assert math.sqrt(sum(c * c for c in v)) == pytest.approx(1.0, abs=1e-12)


# ── Input validation ──────────────────────────────────────────────────────────

def test_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        solar_position_utc(datetime(2024, 6, 21, 12), 0.0, 0.0)


@pytest.mark.parametrize("lat,lon", [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)])
def test_rejects_out_of_range_coordinates(lat, lon):
    with pytest.raises(ValueError):
        solar_position_utc(datetime(2024, 6, 21, 12, tzinfo=timezone.utc), lat, lon)


# ── Timezone resolution ───────────────────────────────────────────────────────

def test_resolves_local_time_via_coordinates():
    """16:00 in Dubai is 12:00 UTC; Dubai has no DST."""
    r = resolve_local_time(datetime(2024, 10, 12, 16, 0), *SITES["dubai"])
    assert r.tz_name == "Asia/Dubai"
    assert r.utc_offset_hours == 4.0
    assert r.utc == datetime(2024, 10, 12, 12, 0, tzinfo=timezone.utc)
    assert r.is_dst is False


def test_british_summer_time_offset():
    """London in July is UTC+1, not UTC."""
    r = resolve_local_time(datetime(2024, 7, 15, 13, 0), *SITES["london"])
    assert r.tz_name == "Europe/London"
    assert r.utc_offset_hours == 1.0
    assert r.is_dst is True


def test_nonexistent_time_is_rejected_not_silently_shifted():
    """01:30 on 2024-03-31 never happened in London -- clocks jumped 01:00 to 02:00."""
    with pytest.raises(TimeframeError, match="does not exist"):
        resolve_local_time(datetime(2024, 3, 31, 1, 30), *SITES["london"])


def test_ambiguous_time_is_flagged_and_resolvable_both_ways():
    """01:30 on 2024-10-27 happens twice in London; both are reachable."""
    first = resolve_local_time(
        datetime(2024, 10, 27, 1, 30), *SITES["london"], dst_ambiguous="first"
    )
    second = resolve_local_time(
        datetime(2024, 10, 27, 1, 30), *SITES["london"], dst_ambiguous="second"
    )
    assert first.utc != second.utc
    assert second.utc - first.utc == timedelta(hours=1)
    assert first.note and "occurs twice" in first.note


def test_timezone_and_solar_compose_correctly():
    """
    End-to-end: a wall-clock time in Dubai yields a sun high in the southwest.

    16:00 local in mid-October is late afternoon, so the sun must be west of
    south (azimuth > 180) and still well above the horizon.
    """
    r = resolve_local_time(datetime(2024, 10, 12, 16, 0), *SITES["dubai"])
    pos = solar_position_utc(r.utc, *SITES["dubai"])
    assert 180.0 < pos.azimuth < 270.0, f"expected SW, got azimuth {pos.azimuth:.2f}"
    assert 10.0 < pos.altitude < 45.0, f"unexpected altitude {pos.altitude:.2f}"
