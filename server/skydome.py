"""
Captured-sky environments, aligned to the scene's own sun.

A procedural ``VRaySky`` gets the sky's *colour* right for a moment and gives
you nothing to look at: no cloud structure, no haze banding near the horizon,
nothing for a wet or polished surface to reflect. A captured HDRI has all of
that, and one problem that matters more here than in most pipelines — it has a
sun baked into it, at whatever bearing the photographer happened to be facing.

Drop such a map into a georeferenced scene unrotated and there are two suns
pointing in two directions, one of them fictional, throwing crossed shadows and
a specular highlight nowhere near the real one. So this module's job is
reconciliation: work out where the captured sun actually was, and rotate the map
until it agrees with the sun this project computed from the date, time and
place.

That is computable rather than eyeballed, because a good HDRI library records
where and when each map was shot. Poly Haven publishes ``coords`` and
``date_taken`` per asset, and :mod:`solar` turns those into an azimuth and
altitude the same way it does for the scene. Choosing a map then stops being a
matter of taste and becomes a search for the smallest altitude error — altitude
being the one a horizontal rotation cannot fix.

Licensing is not incidental. Poly Haven's HDRIs are CC0, which is why they are
used here: every other component in this project was chosen to be safe to use
commercially, and a sky under a non-commercial licence would quietly undo that.
"""

from __future__ import annotations

import json
import math
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone

from solar import solar_position_utc

__all__ = [
    "SkyDome",
    "POLYHAVEN_API",
    "ATTRIBUTION",
    "hdri_sun_position",
    "rotation_for_azimuth",
    "choose_hdri",
    "apply_hdri",
]

POLYHAVEN_API = "https://api.polyhaven.com"

ATTRIBUTION = (
    "Sky HDRI from Poly Haven (polyhaven.com), released under CC0 1.0 "
    "(public domain dedication). No attribution is legally required; it is "
    "recorded because a build should be able to say where every input came from."
)

_UA = {"User-Agent": "atlas-scene/0.1 (+https://github.com/ssnfts/atlas)"}


@dataclass(frozen=True)
class SkyDome:
    """A captured sky, and how it has to be turned to match the scene."""

    name: str
    path: str
    hdri_azimuth: float          # where the sun is in the map, degrees from north
    hdri_altitude: float         # how high, degrees
    scene_azimuth: float
    scene_altitude: float
    horizontal_rotation: float   # degrees to spin the map by
    multiplier: float = 1.0

    @property
    def altitude_error(self) -> float:
        """
        Degrees of altitude mismatch left after rotating.

        The residual that matters: a horizontal rotation moves the captured sun
        around the compass but cannot raise or lower it, so this is the part of
        the disagreement that survives. A degree or two is invisible; ten means
        the map's light is coming from a different time of day than the scene's.
        """
        return abs(self.hdri_altitude - self.scene_altitude)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "hdri_sun": {"azimuth": round(self.hdri_azimuth, 3),
                         "altitude": round(self.hdri_altitude, 3)},
            "scene_sun": {"azimuth": round(self.scene_azimuth, 3),
                          "altitude": round(self.scene_altitude, 3)},
            "horizontal_rotation_deg": round(self.horizontal_rotation, 3),
            "altitude_error_deg": round(self.altitude_error, 3),
            "multiplier": self.multiplier,
            "attribution": ATTRIBUTION,
        }


def hdri_sun_position(asset: str, *, timeout: float = 30.0) -> tuple[float, float]:
    """
    Where the sun was when a Poly Haven HDRI was captured.

    Uses the asset's recorded location and capture time, run through the same
    :func:`solar.solar_position_utc` the scene uses — so the map's sun and the
    scene's sun are measured by one ruler rather than two.
    """
    request = urllib.request.Request(f"{POLYHAVEN_API}/info/{asset}", headers=_UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        info = json.loads(response.read().decode("utf-8"))

    coords = info.get("coords")
    taken = info.get("date_taken")
    if not coords or not taken:
        raise ValueError(
            f"{asset} records no coords/date_taken, so its sun cannot be located. "
            "Pick an asset that does rather than guessing a rotation."
        )
    latitude, longitude = coords
    when = datetime.fromtimestamp(int(taken), tz=timezone.utc)
    position = solar_position_utc(when, float(latitude), float(longitude))
    return position.azimuth, position.altitude


def rotation_for_azimuth(hdri_azimuth: float, scene_azimuth: float) -> float:
    """Degrees to rotate a map so its sun lands on the scene's bearing."""
    return (scene_azimuth - hdri_azimuth) % 360.0


def choose_hdri(candidates, scene_altitude: float, *, timeout: float = 30.0):
    """
    Rank candidate assets by how well their sun height matches the scene's.

    Altitude, not azimuth, because azimuth is free — a horizontal rotation fixes
    it exactly — while altitude is baked in. Returns ``(asset, azimuth,
    altitude)`` tuples, best first. Assets whose metadata cannot be read are
    skipped rather than sinking the search.
    """
    scored = []
    for asset in candidates:
        try:
            azimuth, altitude = hdri_sun_position(asset, timeout=timeout)
        except Exception:
            continue
        scored.append((abs(altitude - scene_altitude), asset, azimuth, altitude))
    scored.sort()
    return [(a, az, alt) for _, a, az, alt in scored]


def apply_hdri(
    bridge,
    path: str,
    *,
    asset: str,
    scene_azimuth: float,
    scene_altitude: float,
    multiplier: float = 1.0,
    hdri_sun: tuple[float, float] | None = None,
) -> SkyDome:
    """
    Load an HDRI into the environment, turned so its sun matches the scene's.

    Raises if the host did not actually enable the environment map. Assigning
    the map and switching it on are two separate things in Max, and a scene
    rendered on the default grey with a perfectly good HDRI sitting in an
    unticked slot looks like a bad HDRI rather than an unticked box.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"HDRI not on disk: {path}")

    if hdri_sun is None:
        hdri_sun = hdri_sun_position(asset)
    hdri_azimuth, hdri_altitude = hdri_sun

    rotation = rotation_for_azimuth(hdri_azimuth, scene_azimuth)
    result = bridge.vray_hdri_env(
        path, horizontal_rotation=rotation, multiplier=multiplier
    )

    if not result.get("use_environment_map"):
        raise RuntimeError(
            "the HDRI was assigned but the environment map is switched off; "
            "the render would use the default background instead"
        )
    if result.get("rejected"):
        raise RuntimeError(f"VRayHDRI rejected parameters: {result['rejected']}")

    return SkyDome(
        name=asset,
        path=path,
        hdri_azimuth=hdri_azimuth,
        hdri_altitude=hdri_altitude,
        scene_azimuth=scene_azimuth,
        scene_altitude=scene_altitude,
        horizontal_rotation=rotation,
        multiplier=multiplier,
    )
