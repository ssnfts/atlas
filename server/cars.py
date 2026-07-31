"""
Generic open-wheel race car proxies.

**These are not Formula 1 cars and must not be presented as any.** A real F1
car's bodywork, its livery and its sponsor marks are the teams' designs and
trademarks, and there is no commercially-clean source for them — which matters
here because every other component in this project was chosen to be safe to use
commercially. What this builds is a *generic single-seater at regulation
dimensions*: the correct silhouette and, more importantly, the correct size and
grid footprint for blocking a shot, judging a camera or reading a shadow.

Dimensions follow the 2022 technical regulations, which is the era the December
2021 date sits at the edge of:

    length      5.63 m      wheelbase   3.60 m
    width       2.00 m      track       1.60 m
    height      0.95 m      tyre        0.72 m dia (18 inch rim)

If a licensed model is bought, drop it in instead: :func:`placements_to_matrices`
returns the position and heading of every grid slot, which is all an importer
needs to put real geometry where these proxies stand.

Built from boxes and cylinders as one mesh per car, in the same scene frame as
everything else: +X east, +Y north, +Z up, metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from massing import Mesh

__all__ = [
    "CarSpec",
    "F1_2022",
    "car_mesh",
    "cars_on_grid",
    "placements_to_matrices",
]


@dataclass(frozen=True)
class CarSpec:
    """Overall dimensions of an open-wheel car, in metres."""

    length: float = 5.63
    width: float = 2.00
    height: float = 0.95
    wheelbase: float = 3.60
    track: float = 1.60
    tyre_diameter: float = 0.72
    tyre_width: float = 0.38
    nose_height: float = 0.28
    body_width: float = 0.90


# The 2022-regulation single-seater. Named for the ruleset, not for any car.
F1_2022 = CarSpec()


def _box(cx, cy, cz, sx, sy, sz):
    """Axis-aligned box as (verts, faces), wound outward."""
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    v = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    f = [
        (0, 3, 2), (0, 2, 1),          # bottom (normal -Z)
        (4, 5, 6), (4, 6, 7),          # top
        (0, 1, 5), (0, 5, 4),          # -Y
        (2, 3, 7), (2, 7, 6),          # +Y
        (1, 2, 6), (1, 6, 5),          # +X
        (3, 0, 4), (3, 4, 7),          # -X
    ]
    return v, f


def _cylinder(cx, cy, cz, radius, width, *, axis="x", segments=24):
    """A wheel: a cylinder whose axis lies along ``axis``."""
    verts, faces = [], []
    half = width / 2.0
    for end, sign in ((0, -1.0), (1, 1.0)):
        for s in range(segments):
            a = 2.0 * math.pi * s / segments
            dy, dz = math.cos(a) * radius, math.sin(a) * radius
            if axis == "x":
                verts.append((cx + sign * half, cy + dy, cz + dz))
            else:
                verts.append((cx + dy, cy + sign * half, cz + dz))
        verts.append((cx + sign * half, cy, cz) if axis == "x"
                     else (cx, cy + sign * half, cz))

    ring = segments + 1
    for s in range(segments):
        s2 = (s + 1) % segments
        a0, a1 = s, s2
        b0, b1 = ring + s, ring + s2
        faces += [(a0, b0, b1), (a0, b1, a1)]        # barrel
        faces.append((a0, a1, segments))              # -end cap
        faces.append((b1, b0, ring + segments))       # +end cap
    return verts, faces


def car_mesh(name: str, spec: CarSpec = F1_2022, *, z: float = 0.0) -> Mesh:
    """
    One car, built nose-forward along +Y and centred on the origin.

    Nose along +Y so a heading of 0 means "pointing north", matching the scene
    frame's convention and the heading returned by
    :func:`roadway.grid_boxes`. Rotating into place is then a single angle
    rather than a matrix nobody can check by eye.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add(v, f):
        base = len(verts)
        verts.extend(v)
        faces.extend((a + base, b + base, c + base) for a, b, c in f)

    hl = spec.length / 2.0
    tyre_r = spec.tyre_diameter / 2.0
    axle_z = z + tyre_r

    # Floor and survival cell, sitting between the axles.
    add(*_box(0.0, 0.0, z + 0.09, spec.body_width * 1.25, spec.wheelbase, 0.10))
    add(*_box(0.0, -0.25, z + 0.34, spec.body_width, 2.30, 0.44))

    # Nose, tapering forward to the front wing.
    add(*_box(0.0, hl - 1.05, z + spec.nose_height, 0.32, 1.70, 0.22))

    # Front and rear wings: the two elements that read hardest in silhouette.
    add(*_box(0.0, hl - 0.14, z + 0.11, spec.width, 0.55, 0.10))
    add(*_box(0.0, -hl + 0.28, z + 0.78, spec.width * 0.82, 0.42, 0.26))
    add(*_box(0.0, -hl + 0.30, z + 0.30, 0.70, 0.70, 0.44))   # gearbox/diffuser

    # Sidepods.
    for side in (-1.0, 1.0):
        add(*_box(side * 0.62, -0.15, z + 0.34, 0.44, 1.85, 0.46))

    # Airbox and roll hoop, then the halo as a single blade.
    add(*_box(0.0, -0.62, z + spec.height - 0.14, 0.36, 0.95, 0.34))
    add(*_box(0.0, 0.18, z + spec.height - 0.06, 0.60, 0.06, 0.16))

    # Wheels, at the corners of the wheelbase and track.
    for fy in (spec.wheelbase / 2.0, -spec.wheelbase / 2.0):
        for sx in (-spec.track / 2.0, spec.track / 2.0):
            add(*_cylinder(sx, fy, axle_z, tyre_r, spec.tyre_width, axis="x"))

    mesh = Mesh(verts=verts, faces=faces, name=name)
    mesh.metadata.update({
        "kind": "car_proxy",
        "spec": "generic open-wheel, 2022 regulation dimensions",
        "licensing": "not a Formula 1 car; no team livery or trademark",
        "length_m": spec.length,
        "width_m": spec.width,
    })
    return mesh


def cars_on_grid(placements, spec: CarSpec = F1_2022, *, z: float = 0.0,
                 name: str = "car") -> list[Mesh]:
    """
    A car on every placement returned by :func:`roadway.grid_boxes`.

    Each placement is ``(x, y, heading_degrees)`` with heading measured
    clockwise from +Y, so the rotation below is the standard clockwise-from-north
    form — the same convention the sun azimuth uses. Getting this backwards puts
    the whole grid facing the wrong way down the straight, which is obvious in a
    render and invisible in a log.
    """
    out: list[Mesh] = []
    for index, (x, y, heading) in enumerate(placements):
        car = car_mesh(f"{name}_{index + 1:02d}", spec, z=0.0)
        a = math.radians(heading)
        sin_a, cos_a = math.sin(a), math.cos(a)
        car.verts = [
            (x + vx * cos_a + vy * sin_a,
             y - vx * sin_a + vy * cos_a,
             z + vz)
            for vx, vy, vz in car.verts
        ]
        car.metadata["grid_slot"] = index + 1
        car.metadata["heading_deg"] = round(heading, 2)
        out.append(car)
    return out


def placements_to_matrices(placements, *, z: float = 0.0) -> list[dict]:
    """
    Grid placements as position + rotation, for importing bought car models.

    The proxies in this module are a stand-in. When a licensed model is
    available this is the handoff: it says where each of the twenty cars stands
    and which way it faces, so the real geometry lands exactly where the boxes
    were verified to be.
    """
    return [
        {
            "slot": index + 1,
            "position_m": [round(x, 4), round(y, 4), round(z, 4)],
            "heading_deg_cw_from_north": round(heading, 3),
        }
        for index, (x, y, heading) in enumerate(placements)
    ]
