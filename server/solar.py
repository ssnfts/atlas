"""
Solar position — NOAA Solar Position Algorithm.

The core math here is deliberately dependency-free and operates on UTC only, so
it can be unit-tested against NOAA's published calculator with no network, no
3ds Max and no packages installed. Timezone resolution is a separate layer
(``timeframe.py``) precisely so a missing tzdata package can never silently
corrupt the astronomy.

Reference: NOAA Solar Calculator
https://gml.noaa.gov/grad/solcalc/calcdetails.html

Angle conventions used throughout:
  * azimuth  — degrees clockwise from **true north** (N=0, E=90, S=180, W=270)
  * altitude — degrees above the horizon (negative when the sun is below it)

These match the scene frame documented in ``frame.py``: +Y is true north,
+X is east, +Z is up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone

__all__ = ["SolarPosition", "solar_position_utc", "sun_vector"]


# ── Julian date ───────────────────────────────────────────────────────────────

def _julian_day(dt: datetime) -> float:
    """Julian Day for a timezone-aware UTC datetime, including fractional day."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime passed to _julian_day; UTC-aware required")
    dt = dt.astimezone(timezone.utc)

    year, month = dt.year, dt.month
    day = (
        dt.day
        + (dt.hour + (dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0) / 60.0) / 24.0
    )

    # January and February are treated as months 13/14 of the previous year.
    if month <= 2:
        year -= 1
        month += 12

    a = math.floor(year / 100.0)
    b = 2 - a + math.floor(a / 4.0)  # Gregorian correction

    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )


# ── Refraction ────────────────────────────────────────────────────────────────

def _refraction_correction(true_elevation_deg: float) -> float:
    """
    Atmospheric refraction in degrees, to be *added* to true elevation.

    Refraction lifts the apparent sun by roughly half a degree at the horizon —
    slightly more than the solar disc's own diameter, which is why the sun is
    fully visible at the moment it is geometrically already set.

    Convention note: this follows NOAA and keeps applying refraction below the
    horizon, which is what makes ``is_up`` agree with published sunrise/sunset
    times. Some implementations (pvlib's SPA among them) clamp refraction to
    zero once the sun has set, so apparent altitudes diverge from ours by up to
    ~0.24° *below* the horizon. Above the horizon the two agree to <0.05°, and
    below it there is no direct sunlight to place, so the difference has no
    effect on a render. Verified in ``tests/test_solar.py``.
    """
    e = true_elevation_deg

    if e > 85.0:
        return 0.0

    te = math.tan(math.radians(e))
    if e > 5.0:
        arcsec = 58.1 / te - 0.07 / te**3 + 0.000086 / te**5
    elif e > -0.575:
        # Polynomial fit near the horizon, where 1/tan blows up.
        arcsec = 1735.0 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)))
    else:
        arcsec = -20.772 / te

    return arcsec / 3600.0


def _clamp_acos(x: float) -> float:
    """acos with domain clamping — polar latitudes push the argument past ±1."""
    return math.acos(max(-1.0, min(1.0, x)))


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SolarPosition:
    """Sun position for one instant at one place."""

    azimuth: float            # degrees clockwise from true north
    altitude: float           # degrees above horizon, refraction-corrected
    altitude_true: float      # degrees above horizon, geometric (no refraction)
    declination: float        # degrees
    equation_of_time: float   # minutes
    hour_angle: float         # degrees
    utc: datetime

    @property
    def is_up(self) -> bool:
        """True when the apparent centre of the sun is above the horizon."""
        return self.altitude > 0.0

    def as_dict(self) -> dict:
        return {
            "azimuth": round(self.azimuth, 4),
            "altitude": round(self.altitude, 4),
            "altitude_true": round(self.altitude_true, 4),
            "declination": round(self.declination, 4),
            "equation_of_time_min": round(self.equation_of_time, 4),
            "hour_angle": round(self.hour_angle, 4),
            "utc": self.utc.isoformat(),
            "is_up": self.is_up,
        }


# ── Core ──────────────────────────────────────────────────────────────────────

def solar_position_utc(dt_utc: datetime, latitude: float, longitude: float) -> SolarPosition:
    """
    Sun azimuth/altitude for a UTC instant at a geodetic position.

    Args:
        dt_utc:    timezone-aware datetime. Converted to UTC internally.
        latitude:  degrees, positive north, [-90, 90].
        longitude: degrees, positive **east**, [-180, 180].

    Longitude sign is the usual source of a silently-wrong result: this function
    takes positive-east (ISO 6709 / GeoJSON convention). NOAA's own spreadsheet
    uses positive-west internally, hence the sign flip on the hour-angle term.
    """
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude out of range: {latitude}")
    if not -180.0 <= longitude <= 180.0:
        raise ValueError(f"longitude out of range: {longitude}")
    if dt_utc.tzinfo is None:
        raise ValueError("naive datetime; pass a timezone-aware datetime")

    dt_utc = dt_utc.astimezone(timezone.utc)

    jd = _julian_day(dt_utc)
    t = (jd - 2451545.0) / 36525.0  # Julian centuries since J2000.0

    # Geometric mean longitude and anomaly of the sun.
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    m_rad = math.radians(m)

    # Eccentricity of Earth's orbit.
    e_orbit = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)

    # Equation of centre — the correction from mean to true anomaly.
    c = (
        math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m_rad) * 0.000289
    )

    true_long = l0 + c

    # Apparent longitude — corrected for nutation and aberration.
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Obliquity of the ecliptic, with the nutation correction.
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    eps_rad = math.radians(eps)

    # Declination.
    decl = math.degrees(
        math.asin(math.sin(eps_rad) * math.sin(math.radians(app_long)))
    )

    # Equation of time, in minutes.
    y = math.tan(eps_rad / 2.0) ** 2
    l0_rad = math.radians(l0)
    eot = 4.0 * math.degrees(
        y * math.sin(2 * l0_rad)
        - 2 * e_orbit * math.sin(m_rad)
        + 4 * e_orbit * y * math.sin(m_rad) * math.cos(2 * l0_rad)
        - 0.5 * y * y * math.sin(4 * l0_rad)
        - 1.25 * e_orbit * e_orbit * math.sin(2 * m_rad)
    )

    # True solar time. Working directly in UTC avoids any dependence on the
    # local zone offset here — the zone only matters when the caller converts
    # wall-clock time to UTC, which happens upstream in timeframe.py.
    utc_minutes = (
        dt_utc.hour * 60.0
        + dt_utc.minute
        + (dt_utc.second + dt_utc.microsecond / 1e6) / 60.0
    )
    true_solar_time = (utc_minutes + eot + 4.0 * longitude) % 1440.0

    # Hour angle: negative before local solar noon, positive after.
    hour_angle = true_solar_time / 4.0
    hour_angle = hour_angle - 180.0 if hour_angle >= 0 else hour_angle + 180.0
    if hour_angle < -180.0:
        hour_angle += 360.0

    lat_rad = math.radians(latitude)
    decl_rad = math.radians(decl)
    ha_rad = math.radians(hour_angle)

    # Zenith angle.
    cos_zenith = math.sin(lat_rad) * math.sin(decl_rad) + math.cos(lat_rad) * math.cos(
        decl_rad
    ) * math.cos(ha_rad)
    zenith = _clamp_acos(cos_zenith)

    elevation_true = 90.0 - math.degrees(zenith)
    elevation = elevation_true + _refraction_correction(elevation_true)

    # Azimuth, clockwise from true north.
    sin_zenith = math.sin(zenith)
    if abs(sin_zenith) < 1e-9:
        # Sun exactly at the zenith — azimuth is undefined; 180 is the
        # conventional choice and the visual difference is nil.
        azimuth = 180.0
    else:
        cos_az = (math.sin(lat_rad) * math.cos(zenith) - math.sin(decl_rad)) / (
            math.cos(lat_rad) * sin_zenith
        )
        az = math.degrees(_clamp_acos(cos_az))
        azimuth = (az + 180.0) % 360.0 if hour_angle > 0 else (540.0 - az) % 360.0

    return SolarPosition(
        azimuth=azimuth,
        altitude=elevation,
        altitude_true=elevation_true,
        declination=decl,
        equation_of_time=eot,
        hour_angle=hour_angle,
        utc=dt_utc,
    )


# ── Scene-frame conversion ────────────────────────────────────────────────────

def sun_vector(azimuth: float, altitude: float) -> tuple[float, float, float]:
    """
    Unit vector pointing **from the site toward the sun**, in scene axes.

    Scene frame: +Y = true north, +X = east, +Z = up. A VRaySun placed at
    ``origin + vector * distance`` and targeted at the origin then casts shadows
    in the real-world direction.

    Note this is the direction *to* the sun, not the direction light travels.
    """
    az = math.radians(azimuth)
    alt = math.radians(altitude)
    cos_alt = math.cos(alt)
    return (
        math.sin(az) * cos_alt,  # east
        math.cos(az) * cos_alt,  # north
        math.sin(alt),           # up
    )
