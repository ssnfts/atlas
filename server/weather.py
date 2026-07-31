"""
Historical weather -> V-Ray sky parameters.

Source: Open-Meteo's ERA5 reanalysis archive, hourly, global, back to 1940, no
API key. The **data** is CC BY 4.0 and fine for commercial use with attribution;
the free public **endpoint** is limited to non-commercial use at roughly 10k
calls/day, so production should point ATLAS_OPENMETEO_URL at a self-hosted
instance (the server is AGPLv3) or a commercial tier.

Every V-Ray parameter named here was read from a live V-Ray 7 install via
``scene.discover_vray`` and is recorded in ``out/vray_capabilities.json`` —
none are remembered or inferred. Writes still go through ``node_set``, which
rejects unknown names, so a future V-Ray rename surfaces as a loud error rather
than a silently default-lit render.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime

__all__ = ["Observation", "SkySettings", "fetch_observation", "sky_from_weather",
           "WeatherError", "ATTRIBUTION"]

ARCHIVE_URL = os.environ.get(
    "ATLAS_OPENMETEO_URL", "https://archive-api.open-meteo.com/v1/archive"
)
AIR_QUALITY_URL = os.environ.get(
    "ATLAS_OPENMETEO_AQ_URL", "https://air-quality-api.open-meteo.com/v1/air-quality"
)

ATTRIBUTION = (
    "Weather data by Open-Meteo.com (ERA5 reanalysis; aerosol data from CAMS), "
    "licensed CC BY 4.0"
)

# Hourly variables requested from the reanalysis archive.
#
# 'visibility' is deliberately absent: the archive endpoint accepts it but
# returns all-null with units "undefined" (it is a forecast-only field). Asking
# for it produced a turbidity that silently fell back to a hardcoded default
# while the log claimed it had measured 24 km. Aerosol optical depth from the
# air-quality endpoint is both real and a better physical proxy.
_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "weather_code",
]

_AQ_HOURLY = ["aerosol_optical_depth", "dust", "pm10"]


class WeatherError(RuntimeError):
    """The weather archive could not be reached or returned nothing usable."""


@dataclass
class Observation:
    """One hour of reanalysis weather at a place."""

    time_utc: datetime
    latitude: float
    longitude: float
    temperature_c: float | None = None
    humidity_pct: float | None = None
    dew_point_c: float | None = None
    pressure_hpa: float | None = None
    cloud_cover_pct: float | None = None
    cloud_low_pct: float | None = None
    cloud_mid_pct: float | None = None
    cloud_high_pct: float | None = None
    wind_speed_ms: float | None = None
    wind_direction_deg: float | None = None
    precipitation_mm: float | None = None
    weather_code: int | None = None
    # From the air-quality endpoint; None when unavailable for that date.
    aerosol_optical_depth: float | None = None
    dust_ug_m3: float | None = None
    pm10_ug_m3: float | None = None
    source: str = "open-meteo ERA5"

    @property
    def description(self) -> str:
        """
        Plain-language summary from the WMO weather code.

        The None check is explicit rather than ``self.weather_code or -1``:
        clear sky is code **0**, which is falsy, so the ``or`` form reported the
        single most common condition as "unknown conditions".
        """
        if self.weather_code is None:
            return "unknown conditions"
        return _WMO_CODES.get(self.weather_code, f"WMO code {self.weather_code}")

    def as_dict(self) -> dict:
        out = {
            k: v
            for k, v in self.__dict__.items()
            if v is not None and k != "time_utc"
        }
        out["time_utc"] = self.time_utc.isoformat()
        out["description"] = self.description
        out["attribution"] = ATTRIBUTION
        return out


# Abridged WMO 4677 weather-code table, enough to describe a shot.
_WMO_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def fetch_observation(
    when_utc: datetime, latitude: float, longitude: float, *, timeout: float = 30.0
) -> Observation:
    """
    Fetch the ERA5 hour containing ``when_utc``.

    ERA5 is hourly, so a 16:20 shot resolves to the 16:00 record. The archive
    also lags real time by about five days; a request inside that window returns
    nulls rather than an error, which is reported here as a clear failure
    instead of being passed downstream as "perfectly clear weather".
    """
    day = when_utc.date().isoformat()
    query = urllib.parse.urlencode(
        {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "start_date": day,
            "end_date": day,
            "hourly": ",".join(_HOURLY),
            "timezone": "UTC",
            "wind_speed_unit": "ms",
        }
    )
    url = f"{ARCHIVE_URL}?{query}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise WeatherError(
            f"weather archive returned HTTP {exc.code} for {day} at "
            f"({latitude:.4f}, {longitude:.4f})"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise WeatherError(f"could not reach the weather archive: {exc}") from exc

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        raise WeatherError(f"weather archive returned no hours for {day}")

    target = when_utc.strftime("%Y-%m-%dT%H:00")
    try:
        idx = times.index(target)
    except ValueError:
        idx = min(
            range(len(times)),
            key=lambda i: abs(
                datetime.fromisoformat(times[i]).replace(tzinfo=when_utc.tzinfo)
                - when_utc
            ),
        )

    def pick(key: str):
        series = hourly.get(key)
        if not series or idx >= len(series):
            return None
        return series[idx]

    obs = Observation(
        time_utc=when_utc,
        latitude=latitude,
        longitude=longitude,
        temperature_c=pick("temperature_2m"),
        humidity_pct=pick("relative_humidity_2m"),
        dew_point_c=pick("dew_point_2m"),
        pressure_hpa=pick("surface_pressure"),
        cloud_cover_pct=pick("cloud_cover"),
        cloud_low_pct=pick("cloud_cover_low"),
        cloud_mid_pct=pick("cloud_cover_mid"),
        cloud_high_pct=pick("cloud_cover_high"),
        wind_speed_ms=pick("wind_speed_10m"),
        wind_direction_deg=pick("wind_direction_10m"),
        precipitation_mm=pick("precipitation"),
        weather_code=pick("weather_code"),
    )

    if obs.cloud_cover_pct is None and obs.temperature_c is None:
        raise WeatherError(
            f"weather archive has no data for {when_utc.isoformat()}. ERA5 lags "
            "real time by roughly five days — for a recent date use the forecast "
            "API instead of the archive."
        )

    _enrich_with_aerosols(obs, timeout=timeout)
    return obs


def _enrich_with_aerosols(obs: Observation, *, timeout: float = 30.0) -> None:
    """
    Add aerosol optical depth from the CAMS air-quality endpoint.

    Best-effort: the aerosol archive covers a shorter period than ERA5, so a
    1950s date has weather but no AOD. Failure leaves the fields None and the
    turbidity estimate falls back to humidity and weather code — and says so,
    rather than pretending it measured something.
    """
    day = obs.time_utc.date().isoformat()
    query = urllib.parse.urlencode(
        {
            "latitude": f"{obs.latitude:.4f}",
            "longitude": f"{obs.longitude:.4f}",
            "start_date": day,
            "end_date": day,
            "hourly": ",".join(_AQ_HOURLY),
            "timezone": "UTC",
        }
    )
    try:
        with urllib.request.urlopen(f"{AIR_QUALITY_URL}?{query}", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return

    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return

    target = obs.time_utc.strftime("%Y-%m-%dT%H:00")
    idx = times.index(target) if target in times else None
    if idx is None:
        return

    def pick(key: str):
        series = hourly.get(key)
        if not series or idx >= len(series):
            return None
        return series[idx]

    obs.aerosol_optical_depth = pick("aerosol_optical_depth")
    obs.dust_ug_m3 = pick("dust")
    obs.pm10_ug_m3 = pick("pm10")


# ── Weather -> V-Ray ──────────────────────────────────────────────────────────

@dataclass
class SkySettings:
    """
    V-Ray sun/sky parameter values derived from an observation.

    Keys are real V-Ray 7 parameter names, verified by live introspection.
    """

    params: dict = field(default_factory=dict)
    rationale: dict = field(default_factory=dict)
    observation: Observation | None = None


# WMO codes that imply the air itself is obscured, regardless of aerosol load.
_FOG_CODES = {45, 48}
_PRECIP_CODES = {51, 53, 55, 61, 63, 65, 71, 73, 75, 80, 81, 82, 95, 96, 99}


def _turbidity_from(obs: Observation) -> tuple[float, str]:
    """
    Turbidity — the aerosol/haze load of the atmosphere.

    V-Ray's turbidity is effectively the Linke turbidity factor, roughly 2
    (pristine alpine air) to 20 (heavy smog or dust storm). The physically
    correct driver is **aerosol optical depth**, which the CAMS air-quality
    dataset provides directly, so it is used when available:

        T ~= 2.0 + 12 * AOD

    calibrated against the endpoints — AOD 0.07 (clean maritime London) gives
    ~2.8, AOD 0.25 (dusty Dubai) gives ~5.0.

    When AOD is missing the estimate degrades to humidity and weather code, and
    the returned rationale says which inputs were actually used. An earlier
    version silently defaulted to "24 km visibility" for every location on
    earth while reporting that as a measurement.
    """
    aod = obs.aerosol_optical_depth

    if aod is not None:
        turbidity = 2.0 + 12.0 * aod
        why = f"from aerosol optical depth {aod:.3f}"
        if obs.dust_ug_m3 and obs.dust_ug_m3 > 20:
            why += f" (dust {obs.dust_ug_m3:.0f} ug/m3)"
    else:
        # No aerosol data — fall back to moisture, and be explicit about it.
        humidity = obs.humidity_pct
        if humidity is None:
            turbidity, why = 3.0, "no aerosol or humidity data — V-Ray-ish default"
        elif humidity > 85:
            turbidity, why = 5.5, f"no aerosol data; humid ({humidity:.0f}%) so hazier"
        elif humidity > 65:
            turbidity, why = 4.0, f"no aerosol data; moderate humidity ({humidity:.0f}%)"
        else:
            turbidity, why = 3.0, f"no aerosol data; dry ({humidity:.0f}%)"

    # Fog and precipitation obscure independently of aerosol load.
    code = obs.weather_code
    if code in _FOG_CODES:
        turbidity = max(turbidity, 15.0)
        why += "; fog reported, turbidity raised"
    elif code in _PRECIP_CODES:
        turbidity = max(turbidity, 6.0)
        why += f"; precipitation reported ({_WMO_CODES.get(code, code)})"

    return round(max(2.0, min(turbidity, 20.0)), 2), why


def _water_vapour_from(obs: Observation) -> tuple[float, str]:
    """
    Atmospheric water vapour, in V-Ray's cm-of-precipitable-water units.

    Typical range is about 0.5 (desert/polar) to 5 (humid tropics). Estimated
    from dew point, which tracks absolute moisture far better than relative
    humidity does.
    """
    dew = obs.dew_point_c
    if dew is None:
        return 2.0, "no dew point available — V-Ray default"

    # Roughly exponential in dew point across the meteorological range.
    vapour = 0.6 * (2.0 ** (dew / 12.0))
    vapour = max(0.3, min(6.0, vapour))
    return round(vapour, 2), f"from dew point {dew:.1f}C"


def _clouds_from(obs: Observation) -> tuple[dict, str]:
    """
    Map cloud cover onto V-Ray's procedural cloud parameters.

    Uses the layered cover fractions where available: high cloud is cirrus,
    low and mid drive the main cloud deck's density.
    """
    total = obs.cloud_cover_pct
    if total is None:
        return {"clouds_on": False}, "no cloud data — clouds left off"

    if total < 5:
        return (
            {"clouds_on": False},
            f"cloud cover {total:.0f}% — clear, clouds off",
        )

    high = obs.cloud_high_pct if obs.cloud_high_pct is not None else 0.0
    low_mid = max(
        obs.cloud_low_pct or 0.0,
        obs.cloud_mid_pct or 0.0,
    )
    if low_mid == 0.0:
        low_mid = total

    params = {
        "clouds_on": True,
        # Density scales with the low/mid deck rather than total cover, so a
        # sky of thin cirrus does not render as an overcast lid.
        "cloud_density": round(min(low_mid / 100.0, 1.0), 3),
        "clouds_density_multiplier": round(0.5 + (low_mid / 100.0), 3),
        "cloud_variety": round(0.3 + 0.4 * (1.0 - abs(total - 50.0) / 50.0), 3),
        "cirrus_amount": round(min(high / 100.0, 1.0), 3),
    }

    # Cloud base, from the lifting condensation level. Espy's approximation puts
    # it at roughly 125 m for every degree of spread between the air temperature
    # and the dew point — the height a parcel has to rise before it saturates,
    # which is physically what a cloud base *is*.
    #
    # Worth deriving rather than defaulting: V-Ray ships 1000 m, and cloud base
    # is the single number that decides whether a deck reads as low grey weather
    # or as high summer cumulus. On a humid coast it can be 300 m; over a desert
    # in the afternoon it can be 3 km, and the same density looks like a
    # different day at each.
    base = _cloud_base_m(obs)
    if base is not None:
        params["height"] = round(base, 1)

    # Depth of the deck. Thin where there is little cloud, deeper as cover
    # grows; capped because this is a fair-weather approximation and not a
    # convective model — a real cumulonimbus is 10 km deep and nothing in an
    # hourly cover fraction would tell us we were looking at one.
    params["thickness"] = round(200.0 + 8.0 * min(low_mid, 100.0), 1)

    why = (
        f"total cover {total:.0f}% (low/mid {low_mid:.0f}%, high {high:.0f}%) "
        f"-> density {params['cloud_density']}, cirrus {params['cirrus_amount']}"
    )
    if base is not None:
        why += (
            f"; base {params['height']:.0f} m from LCL "
            f"(T {obs.temperature_c:.1f}C, dew point {obs.dew_point_c:.1f}C)"
        )
    return params, why


def _scale_clouds(params: dict, why: str, scale: float) -> tuple[dict, str]:
    """
    Scale the derived cloud amounts, and say loudly that this is not the weather.

    Only the *amount* parameters move. Cloud base stays where the thermodynamics
    put it, because the height of the condensation level is not an artistic
    decision — a deck at 1075 m with twice the cover is a plausible sky, and the
    same deck moved to 300 m is a different climate.
    """
    scaled = dict(params)
    if scale > 0.0:
        scaled["clouds_on"] = True
    for key, cap in (("cloud_density", 1.0), ("cirrus_amount", 1.0),
                     ("cloud_variety", 1.0), ("clouds_density_multiplier", 3.0)):
        if key in scaled:
            scaled[key] = round(min(scaled[key] * scale, cap), 3)
    if "thickness" in scaled:
        scaled["thickness"] = round(min(scaled["thickness"] * scale, 4000.0), 1)

    return scaled, (
        f"{why}; SCALED x{scale:g} for the shot — this sky no longer matches "
        "the observation and must not be described as the recorded weather"
    )


def _cloud_base_m(obs: Observation) -> float | None:
    """
    Height of the cloud base in metres, from the temperature/dew-point spread.

    Espy's approximation: 125 m per degree Celsius of spread. Crude next to a
    real sounding, and far better than a constant — it is the standard field
    estimate for exactly this, and both inputs are already in the observation.

    Returns None when either input is missing, so the caller leaves V-Ray's own
    default alone rather than substituting a fabricated altitude.
    """
    if obs.temperature_c is None or obs.dew_point_c is None:
        return None
    spread = obs.temperature_c - obs.dew_point_c
    if spread < 0.0:
        # Saturated at the surface: fog, not a cloud deck with a base above it.
        return 0.0
    return max(50.0, min(125.0 * spread, 6000.0))


def sky_from_weather(
    obs: Observation,
    *,
    ground_albedo: float | None = None,
    cloud_scale: float = 1.0,
) -> SkySettings:
    """
    Derive V-Ray sun/sky parameters from an observation.

    Returns values rather than applying them, so they can be inspected, logged
    and overridden. Every key is a verified V-Ray 7 parameter name.

    ``cloud_scale`` multiplies the derived cloud amounts. It defaults to 1.0 —
    the sky the instruments recorded — and exists because "put some clouds in
    it" is a legitimate request that the weather sometimes refuses: the Yas
    Marina reference hour was 9% cover, which is a correct and almost empty sky.

    Anything other than 1.0 is a **deviation from the observation**, and is
    recorded as one in :attr:`SkySettings.rationale` and flagged by
    :attr:`SkySettings.matches_observation`. A scene can be lit for the weather
    that happened or for the weather someone wanted, and the file should say
    which — that is the difference between a reconstruction and a picture.
    """
    params: dict = {}
    rationale: dict = {}

    turbidity, why = _turbidity_from(obs)
    params["turbidity"] = turbidity
    rationale["turbidity"] = why

    vapour, why = _water_vapour_from(obs)
    params["water_vapour"] = vapour
    rationale["water_vapour"] = why

    cloud_params, why = _clouds_from(obs)
    if cloud_scale != 1.0:
        cloud_params, why = _scale_clouds(cloud_params, why, cloud_scale)
    params.update(cloud_params)
    rationale["clouds"] = why

    # Ozone affects sky hue; the default suits most latitudes and there is no
    # ozone column in this dataset, so it is only set when asked for.
    if ground_albedo is not None:
        params["ground_albedo"] = ground_albedo
        rationale["ground_albedo"] = "caller-supplied"

    return SkySettings(params=params, rationale=rationale, observation=obs)
