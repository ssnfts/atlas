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
    # Front and rear are not the same tyre. The 2022 regulations put 305 mm on
    # the front and 405 mm on the rear, and that 100 mm is one of the most
    # recognisable things about an open-wheel car from behind — building both
    # ends at one width is the sort of detail whose absence reads as "model"
    # without anyone being able to say why.
    tyre_width_front: float = 0.305
    tyre_width_rear: float = 0.405
    rim_diameter: float = 0.4572        # 18 inch
    nose_height: float = 0.28
    body_width: float = 0.90


# The 2022-regulation single-seater. Named for the ruleset, not for any car.
F1_2022 = CarSpec()

# A grid's worth of racing colours, two cars to a colour the way a real field
# pairs team-mates.
#
# **Colours only — no liveries, no marks, no team names.** A flat colour is not
# anyone's intellectual property; a livery is a design and a badge is a
# trademark, and neither appears here or should be added. These are named for
# what they are rather than for who runs them, because a shade of red on a
# generic car is a shade of red, and calling it by a team's name is the first
# step toward implying something about that team.
RACING_COLOURS = [
    ("scarlet",      (168, 22, 24)),
    ("scarlet",      (168, 22, 24)),
    ("gunmetal",     (32, 36, 42)),
    ("gunmetal",     (32, 36, 42)),
    ("deep navy",    (18, 34, 84)),
    ("deep navy",    (18, 34, 84)),
    ("papaya",       (214, 96, 18)),
    ("papaya",       (214, 96, 18)),
    ("racing green", (14, 78, 58)),
    ("racing green", (14, 78, 58)),
    ("french blue",  (24, 92, 168)),
    ("french blue",  (24, 92, 168)),
    ("white",        (208, 208, 212)),
    ("white",        (208, 208, 212)),
    ("sky blue",     (86, 150, 198)),
    ("sky blue",     (86, 150, 198)),
    ("maroon",       (96, 24, 40)),
    ("maroon",       (96, 24, 40)),
    ("slate",        (74, 82, 92)),
    ("slate",        (74, 82, 92)),
]


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


def _revolve(cx, cy, cz, profile, *, segments=48):
    """
    Revolve a ``(radius, offset)`` profile about the wheel's axle (the X axis).

    A tyre is not a cylinder. A cylinder has square corners where the tread
    meets the sidewall, and square corners are exactly where a low sun puts a
    hard specular line — which is what made the first proxies read as blocks
    with rounded ends rather than as tyres. Revolving a profile gives the real
    shape: a crowned tread, a rounded shoulder, a sidewall that tucks back in
    to the rim, and a bead. The shoulder highlight then travels around the
    curve the way it does on a photograph of a car.

    ``profile`` runs from one bead across the tyre to the other, as
    ``(radius_m, x_offset_m)`` pairs. Open at both ends; the caller closes it
    with a rim disc.
    """
    verts: list[tuple[float, float, float]] = []
    for radius, offset in profile:
        for s in range(segments):
            a = 2.0 * math.pi * s / segments
            verts.append((cx + offset,
                          cy + math.cos(a) * radius,
                          cz + math.sin(a) * radius))

    faces: list[tuple[int, int, int]] = []
    for ring in range(len(profile) - 1):
        base_a, base_b = ring * segments, (ring + 1) * segments
        for s in range(segments):
            s2 = (s + 1) % segments
            a0, a1 = base_a + s, base_a + s2
            b0, b1 = base_b + s, base_b + s2
            # Wound so the normal points away from the axle. The first version
            # had these reversed, which was invisible in every dimension check
            # -- widths, diameter and profile were all exact -- and showed up
            # only as a negative signed volume once the wheel was closed with
            # its rim discs. An inside-out tyre renders as a dark hole under a
            # low sun rather than as an error.
            faces += [(a0, b1, b0), (a0, a1, b1)]
    return verts, faces


def _tyre_profile(spec: CarSpec, width: float):
    """
    Half-section of a modern slick, mirrored about the wheel centre.

    The numbers are the shape of an 18-inch F1 tyre: a bead at the rim, a
    sidewall that stands nearly straight, a shoulder that rolls over across
    about 45 mm, and a tread that crowns very slightly toward the middle. The
    crown matters more than it sounds — it is why a tyre catches a band of
    light along its centre rather than a flat sheet across the whole tread.
    """
    rim = spec.rim_diameter / 2.0
    outer = spec.tyre_diameter / 2.0
    shoulder = outer - 0.018          # where the roll-over starts
    half = width / 2.0
    lip = half - 0.045                # tread width before the shoulder

    # The bead is the outermost point at each end, so a rim disc placed at
    # +/- half closes the wheel exactly. An earlier version put the sidewall
    # root 10 mm outboard of the bead, which left a ring where the surface was
    # open — invisible in a render and enough to make the signed-volume check
    # meaningless, since the mesh it was measuring was not closed.
    return [
        (rim,          -half),           # bead, seals against the rim disc
        (rim + 0.030,  -half + 0.006),   # sidewall root
        (shoulder,     -half + 0.020),   # sidewall out to the shoulder
        (outer,        -lip),            # shoulder roll-over
        (outer + 0.004, 0.0),            # crowned tread centre
        (outer,         lip),
        (shoulder,      half - 0.020),
        (rim + 0.030,   half - 0.006),
        (rim,           half),           # bead, outboard
    ]


def _disc(cx, cy, cz, radius, offset, *, segments=48, facing=1.0):
    """A flat disc closing the wheel — the rim face."""
    verts = [(cx + offset, cy, cz)]
    for s in range(segments):
        a = 2.0 * math.pi * s / segments
        verts.append((cx + offset,
                      cy + math.cos(a) * radius,
                      cz + math.sin(a) * radius))
    faces = []
    for s in range(segments):
        s2 = (s + 1) % segments
        if facing >= 0:
            faces.append((0, 1 + s, 1 + s2))
        else:
            faces.append((0, 1 + s2, 1 + s))
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

    mesh = Mesh(verts=verts, faces=faces, name=name)
    mesh.metadata.update({
        "kind": "car_proxy",
        "spec": "generic open-wheel, 2022 regulation dimensions",
        "licensing": "not a Formula 1 car; no team livery or trademark",
        "length_m": spec.length,
        "width_m": spec.width,
    })
    return mesh


def wheel_meshes(name: str, spec: CarSpec = F1_2022, *, z: float = 0.0,
                 segments: int = 48) -> tuple[Mesh, Mesh]:
    """
    All four tyres as one mesh, and all four rims as another.

    Split by material, not by wheel: rubber and machined aluminium are as far
    apart as two surfaces in this scene get — one is near-black and almost
    matte, the other is a bright metal — and keeping them in one object would
    force a single compromise material onto both. Four wheels per mesh rather
    than sixteen objects keeps the scene node count down.
    """
    tyre_v: list = []
    tyre_f: list = []
    rim_v: list = []
    rim_f: list = []

    def add(target_v, target_f, v, f):
        base = len(target_v)
        target_v.extend(v)
        target_f.extend((a + base, b + base, c + base) for a, b, c in f)

    axle_z = z + spec.tyre_diameter / 2.0
    rim_r = spec.rim_diameter / 2.0

    for fy, width in ((spec.wheelbase / 2.0, spec.tyre_width_front),
                      (-spec.wheelbase / 2.0, spec.tyre_width_rear)):
        for sx in (-spec.track / 2.0, spec.track / 2.0):
            profile = _tyre_profile(spec, width)
            add(tyre_v, tyre_f, *_revolve(sx, fy, axle_z, profile, segments=segments))
            # Rim faces, one per side, facing outward.
            for offset, facing in ((-width / 2.0, -1.0),
                                   (width / 2.0, 1.0)):
                add(rim_v, rim_f,
                    *_disc(sx, fy, axle_z, rim_r, offset,
                           segments=segments, facing=facing))

    tyres = Mesh(verts=tyre_v, faces=tyre_f, name=f"{name}_tyres")
    tyres.metadata.update({"kind": "tyres", "segments": segments,
                           "width_front_m": spec.tyre_width_front,
                           "width_rear_m": spec.tyre_width_rear})
    rims = Mesh(verts=rim_v, faces=rim_f, name=f"{name}_rims")
    rims.metadata.update({"kind": "rims", "rim_diameter_m": spec.rim_diameter})
    return tyres, rims


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
        slot = f"{name}_{index + 1:02d}"
        parts = [car_mesh(slot, spec, z=0.0)]
        parts.extend(wheel_meshes(slot, spec, z=0.0))

        a = math.radians(heading)
        sin_a, cos_a = math.sin(a), math.cos(a)
        for part in parts:
            part.verts = [
                (x + vx * cos_a + vy * sin_a,
                 y - vx * sin_a + vy * cos_a,
                 z + vz)
                for vx, vy, vz in part.verts
            ]
            part.metadata["grid_slot"] = index + 1
            part.metadata["heading_deg"] = round(heading, 2)
            part.metadata["colour"] = RACING_COLOURS[index % len(RACING_COLOURS)]
        out.extend(parts)
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
