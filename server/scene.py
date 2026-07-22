"""
Scene assembly in 3ds Max — sun, sky and camera exposure.

V-Ray parameter names here are **discovered from the live host**, never
hardcoded. Chaos renames and adds sun/sky parameters between releases (the sky
cloud parameters especially), and MaxScript assignment to a property that does
not exist is silently ignored — producing a default-lit render with no error
anywhere to explain it. :func:`discover_vray` reads what actually exists;
writes go through the bridge's ``node_set``, which refuses unknown names.

Sun placement drives the ``VRaySun`` transform directly rather than going
through the 3ds Max Sun Positioner: Sun Positioner is not reliably honoured by
V-Ray, whereas a transform is deterministic and independently testable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime

from maxbridge import MaxBridge
from solar import SolarPosition, solar_position_utc, sun_vector
from timeframe import ResolvedTime, resolve_local_time

__all__ = [
    "SunSetup",
    "VRayCapabilities",
    "SceneError",
    "discover_vray",
    "configure_units",
    "build_sun",
    "apply_sky",
    "link_sky_to_sun",
    "exposure_for_altitude",
]

# Only the *direction* to a V-Ray sun matters, but placing it far out keeps the
# object clear of scene geometry and out of the way when framing a shot.
SUN_DISTANCE_M = 5000.0


class SceneError(RuntimeError):
    """Scene assembly failed inside 3ds Max."""


# ── Capability discovery ──────────────────────────────────────────────────────

@dataclass
class VRayCapabilities:
    """What this particular V-Ray build actually exposes."""

    available: bool
    sun_class: str | None = None
    sky_class: str | None = None
    sun_params: list[str] = field(default_factory=list)
    sky_params: list[str] = field(default_factory=list)
    renderer: str | None = None
    note: str | None = None

    def has_sun_param(self, name: str) -> bool:
        return name.lower() in {p.lower() for p in self.sun_params}

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "sun_class": self.sun_class,
            "sky_class": self.sky_class,
            "sun_params": self.sun_params,
            "sky_params": self.sky_params,
            "renderer": self.renderer,
            **({"note": self.note} if self.note else {}),
        }


# Candidate class names, most likely first. Probed in order against the live
# host rather than assumed.
_SUN_CLASS_CANDIDATES = ("VRaySun", "VRay_Sun", "VRaySunLight")
_SKY_CLASS_CANDIDATES = ("VRaySky", "VRay_Sky")


def configure_units(bridge: MaxBridge) -> dict:
    """
    Put the scene into metres.

    3ds Max defaults to inches. Every coordinate this project produces is in
    metres, so on a default scene a 30 m building would arrive 30 inches tall
    and the sun would sit 5 km *inches* away — about 127 m. Nothing errors; the
    scene is just silently 39.37x wrong, which reads as "the import is broken".

    Only the **system** unit matters for geometry. Display units are cosmetic
    and left to the artist.
    """
    before = bridge.get("units.SystemType")
    if str(before).lstrip("#").lower() != "meters":
        bridge.set("units.SystemType", MaxBridge.name("meters"))
    after = bridge.get("units.SystemType")
    return {
        "system_units_before": str(before),
        "system_units_after": str(after),
        "changed": str(before) != str(after),
    }


def discover_vray(bridge: MaxBridge) -> VRayCapabilities:
    """
    Introspect the installed V-Ray for its sun and sky parameter names.

    Worth re-running after any V-Ray or 3ds Max upgrade; everything downstream
    consults the result instead of assuming a parameter exists.
    """
    caps = VRayCapabilities(available=False)

    try:
        current = bridge.get("renderers.current")
        caps.renderer = str(current)
    except Exception:
        caps.renderer = None

    for sun_class in _SUN_CLASS_CANDIDATES:
        try:
            info = bridge.properties(cls=sun_class)
        except Exception:
            continue
        caps.sun_class = sun_class
        caps.sun_params = sorted(info.get("properties", {}))
        caps.available = True
        break

    for sky_class in _SKY_CLASS_CANDIDATES:
        try:
            info = bridge.properties(cls=sky_class)
        except Exception:
            continue
        caps.sky_class = sky_class
        caps.sky_params = sorted(info.get("properties", {}))
        break

    if not caps.available:
        caps.note = (
            "No V-Ray sun class found. Either V-Ray is not installed for this "
            "3ds Max version, or it is installed but not active — set V-Ray as "
            "the current renderer and retry."
        )

    return caps


# ── Exposure ──────────────────────────────────────────────────────────────────

def exposure_for_altitude(altitude_deg: float) -> dict:
    """
    A sane physical-camera starting exposure for a given sun altitude.

    Anchored on the 'sunny 16' rule: at f/16, a shutter of 1/ISO is correct for
    a subject in full midday sun. Each halving of illumination costs one stop,
    and illumination falls roughly with the sine of the sun's altitude, so the
    exposure is opened up as the sun drops.

    This is a starting point that keeps a first render readable, not a
    substitute for grading — which is why it is returned rather than silently
    imposed.
    """
    import math

    iso = 100.0
    f_number = 8.0

    if altitude_deg <= 0.0:
        # Below the horizon there is no direct sun; the sky alone is several
        # stops darker. Anchor on civil twilight rather than extrapolating.
        return {
            "iso": 800.0,
            "f_number": 4.0,
            "shutter_speed": 30.0,
            "note": "sun below horizon — sky-only lighting, exposure is a guess",
        }

    # Stops below the midday reference, from the sine falloff.
    sin_alt = max(math.sin(math.radians(altitude_deg)), 1e-3)
    stops_down = -math.log2(sin_alt)

    # Sunny-16 at f/16 -> 1/100s at ISO 100. At f/8 that is two stops more
    # light, so the reference shutter is 1/400s.
    shutter_denom = 400.0 / (2.0**stops_down)
    shutter_denom = max(30.0, min(4000.0, shutter_denom))

    return {
        "iso": iso,
        "f_number": f_number,
        "shutter_speed": round(shutter_denom, 1),
        "exposure_value": round(exposure_value(iso, f_number, 1.0 / shutter_denom), 2),
        "note": f"sunny-16 derived, {stops_down:.1f} stops below midday reference",
    }


def exposure_value(iso: float, f_number: float, shutter_seconds: float) -> float:
    """
    Exposure value for a camera setting.

        EV = log2(N^2 / t) - log2(ISO / 100)

    EV is used to drive the 3ds Max Physical Camera because it is a single
    unambiguous number. The alternative — ISO plus f-number plus shutter —
    depends on ``shutter_unit_type``, which selects between seconds, frames and
    degrees; set the wrong one and a 1/165 s exposure is silently read as 165
    frames.
    """
    import math

    return math.log2((f_number**2) / shutter_seconds) - math.log2(iso / 100.0)


def build_camera(
    bridge: MaxBridge,
    *,
    position: tuple[float, float, float],
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    exposure: dict | None = None,
    camera_name: str = "Atlas_Cam",
    fov_degrees: float = 55.0,
) -> dict:
    """
    Create or update a Physical Camera with a physically-derived exposure.

    A plain ``Freecamera`` has **no exposure control at all**, so a V-Ray sun at
    real-world intensity clips the whole frame to white. That is not a subtle
    error but it looks like one — the render is uniformly bright and nothing in
    any log mentions exposure.
    """
    existing = bridge.call("getNodeByName", camera_name)
    if existing is None:
        # Prefer V-Ray's own camera: it definitely honours V-Ray's exposure
        # model, whereas the native Physical Camera's EV is not reliably applied
        # by V-Ray GPU.
        created = None
        for cls in ("VRayPhysicalCamera", "Physical_Camera"):
            try:
                created = bridge.call(cls)
                break
            except Exception:
                continue
        if not isinstance(created, dict) or "__node__" not in created:
            raise SceneError(f"could not create a physical camera (got {created!r})")
        bridge.node_set(created["__node__"], "name", camera_name)

    bridge.node_set(camera_name, "pos", MaxBridge.point3(*position))
    bridge.node_set(camera_name, "specify_fov", True)
    bridge.node_set(camera_name, "fov", float(fov_degrees))

    # A camera created from script has identity rotation and looks straight
    # down its local -Z, exactly like a VRaySun with no target. Setting only the
    # position and target_distance leaves it pointing at the ground no matter
    # what `target` says, and nothing errors.
    #
    # This went unnoticed for a while because the only camera in the project was
    # demo_shadow_test's, which looks straight down at a ground plane — the one
    # view where the broken orientation is also the correct one.
    #
    # Fixed the same way build_sun fixes the sun: give it a real target node and
    # let Max derive the rotation, then verify the direction rather than trust it.
    target_name = f"{camera_name}_Target"
    if bridge.call("getNodeByName", target_name) is None:
        created_target = bridge.call("Targetobject")
        bridge.node_set(created_target["__node__"], "name", target_name)
    bridge.node_set(target_name, "pos", MaxBridge.point3(*target))
    try:
        bridge.node_set(camera_name, "targeted", True)
    except Exception:
        pass
    bridge.node_set(camera_name, "target", MaxBridge.node(target_name))

    dx, dy, dz = (t - p for t, p in zip(target, position))
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    bridge.node_set(camera_name, "target_distance", max(distance, 0.001))

    applied = {"position": list(position), "fov": fov_degrees,
               "target_distance": round(distance, 3)}

    # Verify the *direction*, not the position. A targeted camera looks along
    # its local -Z, so the transform's Z axis must point from the target back
    # toward the camera.
    if distance > 1e-6:
        back = tuple(-c / distance for c in (dx, dy, dz))
        try:
            z_axis = _transform_z_axis(bridge.node_get(camera_name, "transform"))
            if z_axis is not None:
                dot = sum(a * b for a, b in zip(z_axis, back))
                if dot < 0.999:
                    offset = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
                    applied["direction_error"] = (
                        f"camera points {offset:.2f}° away from its target"
                    )
                else:
                    applied["direction_verified"] = True
        except Exception as exc:  # pragma: no cover - host-shape dependent
            applied["direction_check_failed"] = str(exc)

    if exposure:
        ev = exposure.get("exposure_value")
        if ev is None:
            ev = exposure_value(
                exposure["iso"], exposure["f_number"], 1.0 / exposure["shutter_speed"]
            )
        # Parameter names differ between the V-Ray and native cameras, so set
        # whichever this build actually has rather than assuming.
        for prop, value in (
            ("exposure", 1),            # VRayPhysicalCamera: exposure on
            ("exposure_gain_type", 1),  # native: 1 == "Target" (EV-driven)
            ("exposure_value", float(ev)),
            ("f_number", float(exposure["f_number"])),
        ):
            try:
                bridge.node_set(camera_name, prop, value)
                applied[prop] = value
            except Exception:
                pass  # not present on this camera class

        stored = bridge.node_get(camera_name, "exposure_value")
        if abs(float(stored) - float(ev)) > 0.01:
            applied["exposure_value_clamped_to"] = stored

    return {"camera": camera_name, "applied": applied}


# ── Sun setup ─────────────────────────────────────────────────────────────────

@dataclass
class SunSetup:
    """The outcome of placing the sun for a given moment."""

    position: SolarPosition
    time: ResolvedTime
    vector: tuple[float, float, float]
    sun_node: str
    world_position: tuple[float, float, float]
    exposure: dict
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sun_node": self.sun_node,
            "solar": self.position.as_dict(),
            "time": self.time.as_dict(),
            "sun_vector": [round(c, 6) for c in self.vector],
            "world_position": [round(c, 3) for c in self.world_position],
            "suggested_exposure": self.exposure,
            "warnings": self.warnings,
        }


def apply_sky(
    bridge: MaxBridge,
    params: dict,
    *,
    sun_name: str = "Atlas_Sun",
    caps: VRayCapabilities | None = None,
    verify: bool = True,
) -> dict:
    """
    Write weather-derived parameters onto the V-Ray sun, then read them back.

    Read-back is the point. MaxScript ignores assignment to a property that does
    not exist, and V-Ray silently clamps values outside a parameter's legal
    range — both produce a scene that renders wrong with nothing in any log to
    explain it. Anything that did not land is reported.

    Returns a report of what was applied, rejected (parameter absent on this
    build) and clamped (accepted but stored a different value).
    """
    if caps is None:
        caps = discover_vray(bridge)
    if not caps.available:
        raise SceneError(caps.note or "V-Ray sun class unavailable")

    known = {p.lower(): p for p in caps.sun_params}
    applied: dict = {}
    rejected: dict = {}
    clamped: dict = {}

    for key, value in params.items():
        real_name = known.get(key.lower())
        if real_name is None:
            rejected[key] = f"not a parameter of {caps.sun_class} in this build"
            continue
        try:
            bridge.node_set(sun_name, real_name, value)
        except Exception as exc:
            rejected[key] = str(exc)
            continue

        if not verify:
            applied[real_name] = value
            continue

        stored = bridge.node_get(sun_name, real_name)
        if _roughly_equal(stored, value):
            applied[real_name] = stored
        else:
            clamped[real_name] = {"requested": value, "stored": stored}

    return {
        "sun_node": sun_name,
        "applied": applied,
        "clamped": clamped,
        "rejected": rejected,
    }


_MATRIX_ROW_RE = re.compile(r"\[\s*(-?[\d.eE+]+)\s*,\s*(-?[\d.eE+]+)\s*,\s*(-?[\d.eE+]+)\s*\]")


def _transform_z_axis(matrix) -> tuple[float, float, float] | None:
    """
    Extract the local Z axis from a MaxScript matrix3.

    The bridge coerces a matrix3 to its string form,
    ``(matrix3 [1,0,0] [0,1,0] [0,0,1] [x,y,z])``, so the third bracket group is
    the Z axis and the fourth is the translation.
    """
    if isinstance(matrix, (list, tuple)) and len(matrix) >= 3:
        row = matrix[2]
        if isinstance(row, (list, tuple)) and len(row) == 3:
            return tuple(float(c) for c in row)
        return None

    rows = _MATRIX_ROW_RE.findall(str(matrix))
    if len(rows) < 3:
        return None
    return tuple(float(c) for c in rows[2])


def _roughly_equal(a, b, tol: float = 1e-3) -> bool:
    """Compare a read-back value with what was requested."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    return a == b


def link_sky_to_sun(
    bridge: MaxBridge,
    *,
    sun_name: str = "Atlas_Sun",
    caps: VRayCapabilities | None = None,
    extra_params: dict | None = None,
) -> dict:
    """
    Put a VRaySky in the environment slot and drive it from the Atlas sun.

    With ``sun_node`` set, the sky inherits the sun's turbidity, ozone, water
    vapour and cloud settings, so weather only has to be written once.
    """
    if caps is None:
        caps = discover_vray(bridge)
    if caps.sky_class is None:
        raise SceneError("no VRaySky class found in this V-Ray build")

    # One atomic host-side command. A texmap has no name or handle, so it cannot
    # be handed back to this process and referenced in a follow-up call the way
    # a scene node can — the live reference must stay inside Max.
    return bridge.vray_sky_setup(
        sun_node=sun_name, sky_class=caps.sky_class, params=extra_params
    )


def build_sun(
    bridge: MaxBridge,
    *,
    local_time: datetime,
    latitude: float,
    longitude: float,
    tz_name: str | None = None,
    sun_name: str = "Atlas_Sun",
    caps: VRayCapabilities | None = None,
    target_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SunSetup:
    """
    Place a physically-correct V-Ray sun for a wall-clock moment at a location.

    Args:
        local_time: naive datetime — what a clock at the location reads.
        latitude, longitude: site position; longitude positive east.
        tz_name: explicit IANA zone. Looked up from coordinates when omitted.
        sun_name: scene node name. Reused if it already exists, so re-timing a
            shot updates the sun in place instead of littering the scene.
        target_point: what the sun aims at, normally the site origin.
    """
    resolved = resolve_local_time(local_time, latitude, longitude, tz_name)
    pos = solar_position_utc(resolved.utc, latitude, longitude)
    vec = sun_vector(pos.azimuth, pos.altitude)

    if caps is None:
        caps = discover_vray(bridge)
    if not caps.available:
        raise SceneError(caps.note or "V-Ray sun class unavailable")

    warnings: list[str] = []
    if not pos.is_up:
        warnings.append(
            f"Sun is below the horizon (altitude {pos.altitude:.2f}°). A sun "
            "light contributes nothing here — light from the sky and practicals."
        )
    elif pos.altitude < 5.0:
        warnings.append(
            f"Sun altitude is {pos.altitude:.2f}° — very long shadows and a "
            "strong colour shift. This is golden hour or close to it."
        )
    if resolved.note:
        warnings.append(resolved.note)

    # Reuse an existing sun so re-timing updates in place.
    existing = bridge.call("getNodeByName", sun_name)
    if existing is None:
        created = bridge.call(caps.sun_class)
        if not isinstance(created, dict) or "__node__" not in created:
            raise SceneError(
                f"{caps.sun_class}() did not return a scene node (got {created!r})"
            )
        bridge.node_set(created["__node__"], "name", sun_name)

    # A VRaySun is a **targeted** light: its direction comes from
    # position -> target, not from position alone. Created from script it has no
    # target, so its rotation stays identity and V-Ray points it straight down
    # no matter where the node is placed. Nothing errors, and reading back the
    # position confirms nothing — the sun is simply in the wrong direction.
    target_name = f"{sun_name}_Target"
    if bridge.call("getNodeByName", target_name) is None:
        target = bridge.call("Targetobject")
        bridge.node_set(target["__node__"], "name", target_name)
    bridge.node_set(target_name, "pos", MaxBridge.point3(*target_point))
    bridge.node_set(sun_name, "target", MaxBridge.node(target_name))

    world = tuple(
        t + c * SUN_DISTANCE_M for t, c in zip(target_point, vec)
    )
    bridge.node_set(sun_name, "pos", MaxBridge.point3(*world))

    # Verify the *direction*, not the position. A targeted light looks along its
    # local -Z, so the third row of its transform must point from the target
    # back toward the sun.
    direction_error = None
    try:
        matrix = bridge.node_get(sun_name, "transform")
        z_axis = _transform_z_axis(matrix)
        if z_axis is not None:
            dot = sum(a * b for a, b in zip(z_axis, vec))
            if dot < 0.999:
                direction_error = (
                    f"sun points {math.degrees(math.acos(max(-1.0, min(1.0, dot)))):.2f}° "
                    f"away from the computed direction"
                )
    except Exception as exc:  # pragma: no cover - host-dependent
        direction_error = f"could not verify sun direction: {exc}"

    if direction_error:
        warnings.append(direction_error)

    return SunSetup(
        position=pos,
        time=resolved,
        vector=vec,
        sun_node=sun_name,
        world_position=world,  # type: ignore[arg-type]
        exposure=exposure_for_altitude(pos.altitude),
        warnings=warnings,
    )
