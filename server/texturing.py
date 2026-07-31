"""
Procedural V-Ray material graphs, as pure data.

``materials.py`` gives every building one flat colour. That reads as plastic:
no surface breakup, no dirt in the crevices, no variation between neighbours.
This module describes richer graphs — a texmap tree feeding several channels of
a VRayMtl — while staying pure data, so the whole thing is checkable offline and
a host-side builder does the construction.

**No external texture files.** Chaos Cosmos assets download on demand and are not
guaranteed present, so every graph here is procedural. A material that renders
grey because a bitmap was missing is exactly the silent failure this project
exists to avoid.

**Every class and slot name below was discovered from the live host**, not
recalled, and the manifest is in ``out/vray_texmap_capabilities.json``. That
matters more here than anywhere else in the codebase: MaxScript ignores a write
to a property that does not exist, so a misremembered texmap slot produces an
untextured render with nothing in any log. The confirmed sets are duplicated
into :data:`TEXMAP_CLASSES` and :data:`TEXMAP_SLOTS` so the tests can reject a
name that was never verified.

**The `_on` flag is the trap.** Every texmap slot has ``texmap_<name>_on`` and
``texmap_<name>_multiplier`` companions, and ``_on`` defaults to **False**. A
graph that sets ``texmap_bump`` and forgets ``texmap_bump_on`` is silently
ignored — the map is attached and unused. :meth:`Graph.slot_writes` always emits
the flag alongside the map, so the two cannot be separated by accident.

Design constraint carried over from ``materials.py``: **materials stay shared per
kind**. 1281 Roman buildings collapse to about seven materials, and that must not
regress into 1281 VRayMtl instances. Per-building variation therefore comes from
``VRayMultiSubTex``, which varies its output *per node* from one material.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import materials

__all__ = [
    "TexNode",
    "Graph",
    "GRAPHS",
    "TEXMAP_CLASSES",
    "TEXMAP_SLOTS",
    "CHANNELS",
    "anti_tiling_scanned_graph",
    "graph_for_building",
    "validate_graph",
    "TexturingError",
]


class TexturingError(ValueError):
    """A graph references something the host does not have, or is malformed."""


# Texmap classes verified **constructible** on the live host (3ds Max 2027 +
# V-Ray GPU 7 update 3) — each one was actually instantiated, not merely read
# from a list.
#
# That distinction cost a real bug. `textureMap.classes` enumerates 23 native
# classes, and four of them cannot be constructed at all: `Wood`,
# `fallofftextureMap`, `NoTexture`. The first version of this module used
# `fallofftextureMap` for the glass Fresnel and `Wood` for timber grain, both
# taken from that list, and both would have failed at build time. Membership in
# the class list is not usability — the real Fresnel map is `Falloff`, and there
# is no constructible wood map on this host at all.
#
# `TextureTiles` is absent entirely, which is why brick banding uses Checker.
TEXMAP_CLASSES = frozenset({
    # native
    "Raytrace", "Checker", "Marble", "Dent", "Mask", "RGB_Tint", "Mix", "Noise",
    "Bitmaptexture", "Reflect_Refract", "Flat_Mirror", "Gradient",
    "CompositeTexturemap", "RGB_Multiply", "Falloff", "output",
    # `ColorCorrection` is the constructible MaxScript alias used by the
    # roughness inversion graph; `Color_Correction` is also reported by the
    # native class manifest.
    "Color_Correction", "ColorCorrection", "MultiTile", "Cellular", "Speckle", "Smoke",
    # V-Ray
    "VRayDirt", "VRayTriplanarTex", "VRayNoiseTex", "VRayBitmap", "VRayColor",
    "VRayNormalMap",
    "VRayCompTex", "VRayEdgesTex", "VRayDistanceTex", "VRayMultiSubTex",
})

# Listed by the host but NOT constructible. Kept explicit so a plausible name
# cannot drift back in — every one of these reads as correct.
NOT_CONSTRUCTIBLE = frozenset({"Wood", "fallofftextureMap", "NoTexture"})

# VRayMtl texmap channel slots, base names only — the `_on` and `_multiplier`
# companions are derived, never written by hand.
TEXMAP_SLOTS = frozenset({
    "texmap_diffuse", "texmap_reflection", "texmap_refraction",
    "texmap_reflectionGlossiness", "texmap_refractionGlossiness",
    "texmap_hilightGlossiness", "texmap_reflectionIOR", "texmap_refractionIOR",
    "texmap_bump", "texmap_displacement", "texmap_opacity", "texmap_metalness",
    "texmap_roughness", "texmap_anisotropy", "texmap_anisotropy_rotation",
    "texmap_self_illumination", "texmap_environment", "texmap_translucent",
    "texmap_coat_amount", "texmap_coat_color", "texmap_coat_glossiness",
    "texmap_coat_bump", "texmap_coat_ior", "texmap_sheen", "texmap_sheen_glossiness",
})

# The channels these graphs actually drive. Deliberately small: each one has to
# earn its render cost, and an unbounded pile of maps on massing geometry is
# slower without looking better.
CHANNELS = ("texmap_diffuse", "texmap_bump", "texmap_reflectionGlossiness")

# Bump strength is capped. V-Ray's bump multiplier is in a unit where 30 is the
# default and large values turn a wall into visual noise that reads as a
# rendering artefact rather than as texture.
MAX_BUMP = 60.0

# Per-building hue jitter, in Color_Correction's degrees of hue rotation. Kept
# small on purpose: enough that neighbouring buildings are not identical, small
# enough that a brick street still reads as one material rather than a fruit
# salad. Beyond about 12 degrees terracotta starts going pink and green.
MAX_HUE_JITTER_DEG = 12.0


@dataclass(frozen=True)
class TexNode:
    """
    One texmap in the graph.

    ``params`` holds scalar and colour values; ``inputs`` holds references to
    other nodes by id. Keeping them apart means a cycle check only has to walk
    ``inputs``, and the host builder knows which values to set directly and
    which to resolve to a live texmap first.
    """

    id: str
    cls: str
    params: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"id": self.id, "class": self.cls, "params": dict(self.params),
                "inputs": dict(self.inputs)}


@dataclass
class Graph:
    """
    A material: a base VRayMtl plus a texmap tree feeding named channels.

    ``base`` reuses ``materials.MaterialSpec`` so the flat colour and the
    procedural graph cannot drift apart — the graph tints and breaks up the
    same colour the fallback uses.
    """

    key: str
    base: materials.MaterialSpec
    nodes: dict = field(default_factory=dict)      # id -> TexNode
    channels: dict = field(default_factory=dict)   # slot -> node id
    note: str = ""

    def add(self, node: TexNode) -> str:
        if node.id in self.nodes:
            raise TexturingError(f"duplicate node id {node.id!r} in graph {self.key!r}")
        self.nodes[node.id] = node
        return node.id

    def bind(self, slot: str, node_id: str) -> None:
        self.channels[slot] = node_id

    def slot_writes(self) -> dict:
        """
        The exact VRayMtl writes this graph implies, `_on` flags included.

        Emitting the flag here rather than leaving it to the caller is the whole
        point: ``_on`` defaults False, so a map written without it is attached
        and silently unused. Separating the two invites exactly that.
        """
        writes: dict[str, object] = {}
        for slot, node_id in self.channels.items():
            writes[slot] = {"__node_ref__": node_id}
            writes[f"{slot}_on"] = True
        return writes

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "material_name": self.base.name,
            "nodes": [n.as_dict() for n in self.nodes.values()],
            "channels": dict(self.channels),
            "base_params": self.base.to_params(),
            "slot_writes": self.slot_writes(),
            "note": self.note,
        }


# ── Validation ────────────────────────────────────────────────────────────────

def validate_graph(graph: Graph) -> list[str]:
    """
    Everything about a graph that is decidable without the host.

    Returns problems rather than raising, so a caller can report all of them at
    once. The class-name and slot-name checks are the valuable ones: those are
    the mistakes MaxScript swallows.
    """
    problems: list[str] = []

    for node in graph.nodes.values():
        if node.cls not in TEXMAP_CLASSES:
            problems.append(
                f"{graph.key}: node {node.id!r} uses texmap class {node.cls!r}, "
                "which was not confirmed present on the host"
            )
        for name, ref in node.inputs.items():
            if ref not in graph.nodes:
                problems.append(
                    f"{graph.key}: node {node.id!r} input {name!r} references "
                    f"unknown node {ref!r}"
                )

    for slot, node_id in graph.channels.items():
        if slot not in TEXMAP_SLOTS:
            problems.append(
                f"{graph.key}: channel {slot!r} is not a confirmed VRayMtl texmap slot"
            )
        if node_id not in graph.nodes:
            problems.append(f"{graph.key}: channel {slot!r} references unknown node {node_id!r}")

    problems += _cycle_problems(graph)

    for node in graph.nodes.values():
        bump = node.params.get("bump_multiplier")
        if bump is not None and not 0.0 <= float(bump) <= MAX_BUMP:
            problems.append(
                f"{graph.key}: node {node.id!r} bump {bump} outside 0..{MAX_BUMP} — "
                "an unbounded bump reads as noise, not surface"
            )
        hue = node.params.get("hue_shift")
        if hue is not None and abs(float(hue)) > MAX_HUE_JITTER_DEG:
            problems.append(
                f"{graph.key}: node {node.id!r} hue jitter {hue} exceeds "
                f"{MAX_HUE_JITTER_DEG} deg — neighbours stop reading as one material"
            )

    return problems


def _cycle_problems(graph: Graph) -> list[str]:
    """
    Depth-first cycle detection.

    A cycle would hang the host builder inside Max's main thread, which locks
    the whole application rather than raising — so it is caught here.
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour = {node_id: WHITE for node_id in graph.nodes}
    problems: list[str] = []

    def walk(node_id: str, trail: list[str]) -> None:
        colour[node_id] = GREY
        for ref in graph.nodes[node_id].inputs.values():
            if ref not in graph.nodes:
                continue
            if colour[ref] == GREY:
                problems.append(
                    f"{graph.key}: cycle {' -> '.join(trail + [node_id, ref])}"
                )
            elif colour[ref] == WHITE:
                walk(ref, trail + [node_id])
        colour[node_id] = BLACK

    for node_id in graph.nodes:
        if colour[node_id] == WHITE:
            walk(node_id, [])
    return problems


# ── Graph construction ────────────────────────────────────────────────────────

def _triplanar(node_id: str, source: str, size_m: float, *,
               randomize: bool = False, frame_offset=None,
               texture_rotation=None) -> TexNode:
    """
    World-space projection at a physical size.

    ``VRayTriplanarTex`` projects along the world axes, so the pattern lands at
    a real-world scale with **no UVs at all** — which is what makes these graphs
    usable on massing that has no texture coordinates. It also removes the
    stretching a planar projection puts on a wall that is not axis-aligned.
    """
    # Parameter names discovered from a live VRayTriplanarTex: the size is
    # `size` and the input is `texture`. `texture_size`/`texmap` read as correct
    # and are both rejected by the host.
    params = {"size": size_m}
    if randomize:
        # These names and their behaviour were read from the live V-Ray 7
        # `VRayTriplanarTex` instance. They are not guessed MaxScript fields.
        params.update({
            "random_texture_offset": True,
            "random_texture_rotation": True,
        })
    if frame_offset is not None:
        params["frame_offset"] = {"__point3__": list(frame_offset)}
    if texture_rotation is not None:
        params["texture_rotation"] = {"__point3__": list(texture_rotation)}
    return TexNode(node_id, "VRayTriplanarTex", params=params,
                   inputs={"texture": source})


def _dirt(node_id: str, source: str, radius_m: float) -> TexNode:
    """
    Occlusion darkening — grime in the crevices and along the ground line.

    The single biggest contributor to massing not looking like plastic: real
    buildings are dirtier where surfaces meet, and a flat diffuse has none of
    that regardless of how good the colour is.
    """
    # `unoccluded_color` and `occluded_color` are *colour* slots; feeding a
    # texmap needs the parallel `texmap_unoccluded_color` slot, which — like
    # every texmap slot on this host — has an `_on` flag defaulting False.
    # Setting the map without it produces plain occlusion and loses the base
    # colour entirely, which reads as a uniformly grey building.
    return TexNode(node_id, "VRayDirt",
                   params={"radius": radius_m, "texmap_unoccluded_color_on": True},
                   inputs={"texmap_unoccluded_color": source})


# Per-building colour variation is NOT implemented, and the removed attempt is
# worth recording so it is not retried the same way.
#
# `VRayMultiSubTex` looked ideal: it selects a sub-texture per node, which would
# have given every building a slightly different tint from one shared material,
# preserving the ~7-materials-for-1281-buildings design. It does not work that
# way when the sub-texture list is empty. Measured on the live host, the whole
# graph collapsed to a uniform grey regardless of the base colour — brick and
# concrete rendered identically at rgb(81,74,70) — because with nothing in the
# list it falls through to `default_color` (a 127.5 grey) rather than to the
# wired `default_texmap`. Under a bright sun that reads as blown-out white.
#
# Removing it restored correct colour immediately: brick rgb(66,31,24) with a
# red-blue spread of 42, concrete neutral at 11, glass bluish at -4.
#
# Doing this properly means populating the sub-texture list with N tinted
# variants and letting `random_by_node_handle` choose between them. That is a
# real feature, not a parameter tweak, and it is not attempted here.


def _masonry_graph(key: str, *, grain_m: float, dirt_m: float, bump: float,
                   hue: float, note: str = "") -> Graph:
    """
    The shared skeleton for concrete, brick, stone, plaster and roof tile.

    All five are "a base colour, broken up by noise at a physical scale, dirtied
    in the crevices, varied slightly per building". Only the numbers differ, and
    keeping them one function makes the differences legible instead of burying
    them in five near-identical blocks.
    """
    graph = Graph(key=key, base=materials.PRESETS[key], note=note)
    r, g, b = graph.base.diffuse

    graph.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    graph.add(TexNode("grain", "Noise",
                      params={"size": grain_m, "levels": 3.0, "phase": 0.0}))
    graph.add(_triplanar("grain_world", "grain", grain_m))
    graph.add(TexNode("tinted", "Mix", params={"mixAmount": 0.28},
                      inputs={"map1": "base", "map2": "grain_world"}))
    graph.add(_dirt("weathered", "tinted", dirt_m))
    graph.bind("texmap_diffuse", "weathered")

    graph.add(TexNode("relief", "Noise",
                      params={"size": grain_m * 0.4, "levels": 4.0,
                              "bump_multiplier": bump}))
    graph.add(_triplanar("relief_world", "relief", grain_m * 0.4))
    graph.bind("texmap_bump", "relief_world")

    # Weathering also dulls reflection where dirt collects, which is what stops
    # a wall reading as uniformly polished.
    graph.add(TexNode("gloss_break", "Mix", params={"mixAmount": 0.35},
                      inputs={"map1": "base", "map2": "relief_world"}))
    graph.bind("texmap_reflectionGlossiness", "gloss_break")
    return graph


def scanned_graph(key: str, spec, texture_set: dict, *, size_m: float,
                  note: str = "") -> Graph:
    """
    A material built from a scanned PBR set rather than from noise.

    Every map goes through a ``VRayTriplanarTex`` at a real-world ``size_m``,
    which is what lets 379 building meshes with no UVs take a photographed
    concrete: the projection is world-space, so the only thing that has to be
    right is the physical size of the texture's repeat. Get that wrong and the
    material is not subtly off — a 4 m concrete panel tiled at 0.4 m reads as
    corduroy.

    Two conversions are load-bearing and neither is obvious:

    **Roughness is inverted.** Poly Haven ships a roughness map; V-Ray's slot is
    *glossiness*, which is 1 - roughness. Wiring one to the other directly makes
    every worn surface a mirror and every polished one matte, which looks like a
    lighting problem rather than a plumbing one. The inversion is done by an
    ``Output`` map with a negative RGB level, because V-Ray has no invert flag
    on the bitmap itself.

    **The normal map needs `VRayNormalMap`, not the bump slot.** A tangent-space
    normal map fed straight into ``texmap_bump`` is interpreted as a height
    field, and its flat blue-violet background then reads as a uniform slope.
    ``VRayNormalMap`` decodes it properly. Poly Haven's ``nor_gl`` is the OpenGL
    convention, which is the one V-Ray expects; ``nor_dx`` has the green channel
    flipped and would invert every dent.
    """
    graph = Graph(key, spec, note=note or f"scanned PBR at {size_m:g} m repeat")

    diffuse = texture_set.get("Diffuse")
    if diffuse:
        graph.add(TexNode("albedo_map", "VRayBitmap",
                          params={"HDRIMapName": diffuse}))
        graph.add(_triplanar("albedo", "albedo_map", size_m))
        graph.add(_dirt("albedo_dirt", "albedo", 0.4))
        graph.bind("texmap_diffuse", "albedo_dirt")

    rough = texture_set.get("Rough")
    if rough:
        graph.add(TexNode("rough_map", "VRayBitmap",
                          params={"HDRIMapName": rough,
                                  "color_space": 0}))     # data, not sRGB
        graph.add(_triplanar("rough_tri", "rough_map", size_m))
        # 1 - roughness. The first attempt used an `Output` map's RGB level and
        # offset, and the host refused both: they live on a sub-object rather
        # than as flat parameters, the same shape as Checker's tiling. This is
        # what the rejected map is for — the graph built either way and the
        # difference would only have shown as every surface being wrong.
        #
        # ColorCorrection's rewireMode 2 is Max's own invert, and it is a plain
        # parameter.
        graph.add(TexNode("gloss", "ColorCorrection",
                          params={"rewireMode": 2},
                          inputs={"map": "rough_tri"}))
        graph.bind("texmap_reflectionGlossiness", "gloss")

    normal = texture_set.get("nor_gl")
    if normal:
        graph.add(TexNode("normal_map", "VRayBitmap",
                          params={"HDRIMapName": normal,
                                  "color_space": 0}))
        graph.add(_triplanar("normal_tri", "normal_map", size_m))
        graph.add(TexNode("normal", "VRayNormalMap",
                          params={"normal_map_on": True,
                                  "normal_map_multiplier": 1.0},
                          inputs={"normal_map": "normal_tri"}))
        graph.bind("texmap_bump", "normal")

    return graph


def anti_tiling_scanned_graph(
    key: str,
    spec,
    primary_set: dict,
    secondary_set: dict,
    *,
    primary_size_m: float,
    macro_size_m: float,
    normal_multiplier: float = 1.0,
    secondary_size_m: float | None = None,
    note: str = "",
) -> Graph:
    """Build a scanned PBR graph which cannot reduce to one tiled bitmap.

    A conventional triplanar projection fixes UV stretching but still repeats
    at a predictable world interval. This graph pairs two independently phased
    scanned sources at incommensurate scales, then chooses between them with a
    third, much larger triplanar noise field. The rule is deliberately encoded
    in a reusable graph rather than left to a one-off scene script: callers
    cannot accidentally return to the tempting one-bitmap setup.

    ``secondary_size_m`` defaults to ``primary_size_m * 1.618``. The ratio is
    intentionally not a convenient integer multiple; common repeating periods
    are pushed far outside a camera-visible facade or track section. Both image
    projections additionally request V-Ray's host-confirmed per-object random
    offset and rotation. ``frame_offset`` and ``texture_rotation`` are
    ``Point3`` values at bridge construction time, not Python lists.

    Asphalt callers pass ``normal_multiplier=0``. A real circuit's wearing
    course has millimetric relief, and a shaded normal/bump pass produced false
    corrugation in the project's low-sun test. Its material variation therefore
    comes from photographed albedo and roughness only.
    """
    if primary_size_m <= 0 or macro_size_m <= 0:
        raise TexturingError("anti-tiling texture scales must be positive")
    if normal_multiplier < 0:
        raise TexturingError("normal_multiplier cannot be negative")

    secondary_size_m = (
        primary_size_m * 1.618 if secondary_size_m is None else secondary_size_m
    )
    if secondary_size_m <= 0:
        raise TexturingError("secondary anti-tiling texture scale must be positive")

    required = ("Diffuse", "Rough")
    missing = [
        f"{label}:{name}"
        for label, texture_set in (("primary", primary_set), ("secondary", secondary_set))
        for name in required
        if not texture_set.get(name)
    ]
    if normal_multiplier:
        missing += [
            f"{label}:nor_gl"
            for label, texture_set in (("primary", primary_set), ("secondary", secondary_set))
            if not texture_set.get("nor_gl")
        ]
    if missing:
        raise TexturingError(
            "anti-tiling PBR graph needs both scanned sources: " + ", ".join(missing)
        )

    graph = Graph(
        key,
        spec,
        note=note or (
            f"anti-tiling scanned PBR: {primary_size_m:g}m / "
            f"{secondary_size_m:g}m, {macro_size_m:g}m macro breakup"
        ),
    )

    def image_pair(node_prefix: str, source_key: str, *, data: bool = False) -> tuple[str, str]:
        bitmap_params = {"color_space": 0} if data else {}
        primary_map = f"primary_{node_prefix}_map"
        secondary_map = f"secondary_{node_prefix}_map"
        primary_tri = f"primary_{node_prefix}_tri"
        secondary_tri = f"secondary_{node_prefix}_tri"
        graph.add(TexNode(primary_map, "VRayBitmap", params={
            "HDRIMapName": primary_set[source_key], **bitmap_params,
        }))
        graph.add(_triplanar(primary_tri, primary_map, primary_size_m, randomize=True))
        graph.add(TexNode(secondary_map, "VRayBitmap", params={
            "HDRIMapName": secondary_set[source_key], **bitmap_params,
        }))
        graph.add(_triplanar(
            secondary_tri,
            secondary_map,
            secondary_size_m,
            randomize=True,
            frame_offset=(17.0, 31.0, 11.0),
            texture_rotation=(0.0, 0.0, 31.0),
        ))
        return primary_tri, secondary_tri

    # One macro field drives every blend so a photographed aggregate cannot be
    # recognised as a repeating cell from either the grandstands or a drone
    # camera. `Mask` is the host-discovered, case-sensitive Mix map slot.
    graph.add(TexNode("macro_noise", "Noise", params={
        "size": macro_size_m,
        "levels": 4.0,
        "phase": 0.0,
    }))
    graph.add(_triplanar(
        "macro_world",
        "macro_noise",
        macro_size_m,
        texture_rotation=(0.0, 0.0, 13.0),
    ))

    primary_albedo, secondary_albedo = image_pair("albedo", "Diffuse")
    graph.add(TexNode("albedo_mix", "Mix", params={"mixAmount": 0.5}, inputs={
        "map1": primary_albedo,
        "map2": secondary_albedo,
        "Mask": "macro_world",
    }))
    graph.add(_dirt("albedo_dirt", "albedo_mix", 0.4))
    graph.bind("texmap_diffuse", "albedo_dirt")

    primary_rough, secondary_rough = image_pair("rough", "Rough", data=True)
    graph.add(TexNode("rough_mix", "Mix", params={"mixAmount": 0.5}, inputs={
        "map1": primary_rough,
        "map2": secondary_rough,
        "Mask": "macro_world",
    }))
    graph.add(TexNode("gloss", "ColorCorrection", params={"rewireMode": 2},
                      inputs={"map": "rough_mix"}))
    graph.bind("texmap_reflectionGlossiness", "gloss")

    if normal_multiplier:
        primary_normal, secondary_normal = image_pair("normal", "nor_gl", data=True)
        graph.add(TexNode("normal_mix", "Mix", params={"mixAmount": 0.5}, inputs={
            "map1": primary_normal,
            "map2": secondary_normal,
            "Mask": "macro_world",
        }))
        graph.add(TexNode("normal", "VRayNormalMap", params={
            "normal_map_on": True,
            "normal_map_multiplier": normal_multiplier,
        }, inputs={"normal_map": "normal_mix"}))
        graph.bind("texmap_bump", "normal")

    return graph


def _build_graphs() -> dict:
    graphs: dict[str, Graph] = {}

    # Sizes are the real-world scale of the visible pattern, in metres.
    graphs["concrete"] = _masonry_graph(
        "concrete", grain_m=0.9, dirt_m=0.35, bump=18.0, hue=5.0,
        note="board-marking scale grain; heavy crevice dirt",
    )
    graphs["brick"] = _masonry_graph(
        "brick", grain_m=0.22, dirt_m=0.25, bump=32.0, hue=9.0,
        note="course-scale breakup; brick bond varies most between buildings",
    )
    graphs["stone"] = _masonry_graph(
        "stone", grain_m=0.55, dirt_m=0.4, bump=26.0, hue=6.0,
        note="ashlar block scale; strongest weathering of the masonry set",
    )
    graphs["plaster"] = _masonry_graph(
        "plaster", grain_m=0.12, dirt_m=0.3, bump=10.0, hue=8.0,
        note="fine render texture; streaks below sills rather than crevice dirt",
    )
    graphs["roof_tile"] = _masonry_graph(
        "roof_tile", grain_m=0.3, dirt_m=0.2, bump=34.0, hue=7.0,
        note="pantile scale; strong relief since roofs are seen at grazing angles",
    )
    # Reachable through the synonym table (asphalt, tar_paper) — a flat roof
    # membrane. Without a graph such a building renders as flat colour among
    # textured neighbours, which reads as a patch of plastic in the street.
    graphs["asphalt"] = _masonry_graph(
        "asphalt", grain_m=0.15, dirt_m=0.25, bump=12.0, hue=3.0,
        note="roofing membrane and paving; little hue variation, it is all bitumen",
    )

    # The racing surface. Not `_masonry_graph`, because the masonry recipe puts
    # a single grain frequency in both diffuse and bump, and on a surface this
    # large that is exactly what reads as sandpaper: at 1:1 a circuit fills the
    # frame, and one fine frequency tiling across 78,000 m2 has no larger
    # structure for the eye to land on.
    #
    # So this is built at two scales. A coarse 6 m noise carries the paving
    # lanes and repair patches — the thing actually visible from a helicopter
    # shot — and a fine 0.18 m noise carries the aggregate, mixed in weakly and
    # driving only a gentle bump. Relief is deliberately low (6.0 against
    # concrete's 18.0): a race track is the flattest surface on the site and
    # a strong bump under a 7 degree sun would texture it like gravel.
    track = Graph("track_asphalt", materials.PRESETS["track_asphalt"],
                  note="two-scale asphalt: 6 m paving patches over 0.18 m "
                       "aggregate; low relief because a circuit is near-flat")
    r, g, b = track.base.diffuse
    track.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    # Coarse structure: paving passes, patches, sun-bleached areas.
    track.add(TexNode("patches", "Noise", params={"size": 6.0, "levels": 3.0}))
    track.add(_triplanar("patches_world", "patches", 6.0))
    track.add(TexNode("patchy", "Mix", params={"mixAmount": 0.22},
                      inputs={"map1": "base", "map2": "patches_world"}))
    # Fine structure: the aggregate itself, weak so it never dominates.
    #
    # 0.18 m was too fine and too strong. Seen from a camera standing on the
    # grid it stopped reading as asphalt and started reading as loose gravel —
    # the "sandpaper" failure. A racing surface is a *fine-graded* wearing
    # course: at any distance a person can stand, the individual stones are
    # below the eye's resolution and what remains is a smooth dark sheet with
    # slow tonal drift. 0.45 m at a lower mix keeps the drift and drops the grit.
    track.add(TexNode("aggregate", "Noise", params={"size": 0.45, "levels": 2.0}))
    track.add(_triplanar("aggregate_world", "aggregate", 0.45))
    track.add(TexNode("surfaced", "Mix", params={"mixAmount": 0.06},
                      inputs={"map1": "patchy", "map2": "aggregate_world"}))
    # Grime collects at the edges of the ribbon and against kerbs, not mid-track
    # where the cars sweep it away; a wide dirt radius approximates that.
    track.add(_dirt("weathered", "surfaced", 0.8))
    track.bind("texmap_diffuse", "weathered")
    # **No bump at all**, and that is the measured answer rather than a
    # preference. A racing surface is a fine-graded wearing course whose relief
    # is one to two millimetres — at 1:1 that is well under a pixel from any
    # camera a person could stand at. Meanwhile this scene's sun sits at 7.5
    # degrees, and at grazing incidence *any* procedural bump becomes dramatic:
    # every lit micro-ridge catches the warm low sun and every trough falls to
    # blue sky fill, so the track rendered as tan-and-navy corrugation.
    #
    # Tried in order, each re-rendered and looked at: 0.18 m at 6.0 (gravel),
    # 0.45 m at 2.0 (still corrugated), none (correct). The lesson is that bump
    # strength cannot be judged against a midday reference and then reused at
    # golden hour — the sun angle is part of the material's tuning.
    # Glossiness is deliberately left as the base spec's flat 0.62 rather than
    # driven by a map. Wiring `patches_world` into it looked right in theory —
    # the racing line is polished and the off-line surface is not — and was
    # badly wrong on the host: glossiness is a 0..1 *scalar*, a Noise is mostly
    # bright, and the bright half went to near-mirror. From a camera on the grid
    # the track turned into tan camouflage, because every mirror patch was
    # reflecting the sand around the circuit.
    #
    # This is the same failure as the VRayMultiSubTex one recorded above: a map
    # feeding a scalar slot whose legal range it does not respect. A gloss break
    # needs a map remapped into a narrow band around the base value, not a raw
    # noise, and that is a real change rather than a rewiring.
    graphs["track_asphalt"] = track

    # Kerb. The only graph here that works in **UV space** rather than world
    # space: a triplanar projection is fixed to the world axes, so its stripes
    # would stay pointing north while the kerb curves away underneath them. The
    # ribbon's UVs run in metres along the kerb, and Checker's natural period of
    # 1.0 UV puts the boundary every 0.5 m — the real band width — while a pinned
    # v keeps it striped instead of chequered.
    kerb = Graph("kerb", materials.PRESETS["kerb"],
                 note="UV-space Checker: 0.5 m red/white bands that follow the "
                      "kerb round a corner, which world-space projection cannot do")
    kerb.add(TexNode("bands", "Checker",
                     params={"Soften": 0.02,
                             "color1": {"__color__": [178, 34, 34]},   # red band
                             "color2": {"__color__": [222, 222, 220]}}))  # white
    kerb.add(_dirt("weathered", "bands", 0.4))
    kerb.bind("texmap_diffuse", "weathered")
    # A kerb is a ramped casting, not a flat sticker: the same banding drives a
    # strong bump so the ribs catch the low sun across their edges.
    kerb.add(TexNode("ribs", "Checker",
                     params={"Soften": 0.05, "bump_multiplier": 40.0}))
    kerb.bind("texmap_bump", "ribs")
    graphs["kerb"] = kerb

    # Grandstand seating. The tell is *row pitch*: a seating deck is a few
    # thousand small units on a raked plane at a very regular ~0.8 m spacing,
    # and that regularity is what the eye reads as seating rather than as a wall.
    # So the dominant map is a Checker at seat pitch, not a noise — noise at any
    # scale reads as dirty concrete, which is exactly the failure being fixed.
    #
    # A second, much coarser noise breaks up the block colour so that a 200 m
    # stand is not one flat rectangle of blue, and the dirt pass darkens the
    # gaps between tiers.
    stand = Graph("grandstand", materials.PRESETS["grandstand"],
                  note="seat-row banding at 0.8 m pitch; the regular spacing is "
                       "what distinguishes seating from a blank facade")
    r, g, b = stand.base.diffuse
    stand.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    stand.add(TexNode("rows", "Checker", params={"Soften": 0.15}))
    stand.add(_triplanar("rows_world", "rows", 0.8))
    stand.add(TexNode("banded", "Mix", params={"mixAmount": 0.30},
                      inputs={"map1": "base", "map2": "rows_world"}))
    stand.add(TexNode("blocks", "Noise", params={"size": 12.0, "levels": 2.0}))
    stand.add(_triplanar("blocks_world", "blocks", 12.0))
    stand.add(TexNode("varied", "Mix", params={"mixAmount": 0.14},
                      inputs={"map1": "banded", "map2": "blocks_world"}))
    stand.add(_dirt("weathered", "varied", 0.5))
    stand.bind("texmap_diffuse", "weathered")
    stand.add(TexNode("relief", "Checker",
                      params={"Soften": 0.1, "bump_multiplier": 22.0}))
    stand.add(_triplanar("relief_world", "relief", 0.8))
    stand.bind("texmap_bump", "relief_world")
    graphs["grandstand"] = stand

    # Wood. The obvious choice is the `Wood` map, and it is in the host's class
    # list — but it cannot be constructed on this build, so it is unusable.
    # `Marble` is the next best fit: its veining is directional and stretches
    # into a passable plank grain under an anisotropic triplanar size. Not as
    # good as a real wood map, and said so rather than implied.
    wood = Graph("wood", materials.PRESETS["wood"],
                 note="Marble veining as plank grain — the Wood map is listed "
                      "by the host but is not constructible on this build")
    r, g, b = wood.base.diffuse
    wood.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    wood.add(TexNode("grain", "Marble", params={"size": 0.05, "vein_width": 0.02}))
    wood.add(_triplanar("grain_world", "grain", 0.05))
    wood.add(TexNode("tinted", "Mix", params={"mixAmount": 0.4},
                     inputs={"map1": "base", "map2": "grain_world"}))
    wood.add(_dirt("weathered", "tinted", 0.2))
    wood.bind("texmap_diffuse", "weathered")
    wood.add(TexNode("relief", "Marble",
                     params={"size": 0.02, "bump_multiplier": 24.0}))
    wood.add(_triplanar("relief_world", "relief", 0.02))
    wood.bind("texmap_bump", "relief_world")
    graphs["wood"] = wood

    # Glass: the Fresnel falloff is what makes it read as glass at all — a flat
    # reflection is the classic tell of a fake curtain wall. Storey banding via
    # Checker, since TextureTiles was probed and is absent.
    glass = Graph("glass", materials.PRESETS["glass"],
                  note="Fresnel falloff plus storey banding; opaque, not refractive")
    r, g, b = glass.base.diffuse
    glass.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    # Checker's tiling lives on a `coords` object rather than as direct
    # parameters; the triplanar wrapper already sets the physical scale, so the
    # only thing worth setting here is the edge softness.
    glass.add(TexNode("mullions", "Checker", params={"Soften": 0.05}))
    glass.add(_triplanar("mullions_world", "mullions", 1.5))
    glass.add(TexNode("banded", "Mix", params={"mixAmount": 0.18},
                      inputs={"map1": "base", "map2": "mullions_world"}))
    glass.bind("texmap_diffuse", "banded")
    glass.add(TexNode("fresnel", "Falloff", params={"type": 2}))  # 2 = Fresnel
    glass.bind("texmap_reflectionGlossiness", "fresnel")
    glass.add(TexNode("panel_relief", "Checker",
                      params={"Soften": 0.02, "bump_multiplier": 8.0}))
    glass.add(_triplanar("panel_world", "panel_relief", 1.5))
    glass.bind("texmap_bump", "panel_world")
    graphs["glass"] = glass

    # Metal: metalness is on the base spec; the graph supplies streaking, which
    # is what distinguishes a warehouse roof from a mirror.
    metal = Graph("metal", materials.PRESETS["metal"], note="brushed streaking, light dirt")
    r, g, b = metal.base.diffuse
    metal.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    metal.add(TexNode("streak", "Noise", params={"size": 0.05, "levels": 2.0}))
    metal.add(_triplanar("streak_world", "streak", 0.05))
    metal.add(TexNode("brushed", "Mix", params={"mixAmount": 0.2},
                      inputs={"map1": "base", "map2": "streak_world"}))
    metal.add(_dirt("weathered", "brushed", 0.15))
    metal.bind("texmap_diffuse", "weathered")
    metal.bind("texmap_reflectionGlossiness", "streak_world")
    metal.add(TexNode("relief", "Noise", params={"size": 0.02, "bump_multiplier": 6.0}))
    metal.add(_triplanar("relief_world", "relief", 0.02))
    metal.bind("texmap_bump", "relief_world")
    graphs["metal"] = metal

    # Ground: coarse, matte, no per-building variation — it is one surface, and
    # VRayMultiSubTex on a single node would do nothing but cost a lookup.
    ground = Graph("ground", materials.PRESETS["ground"],
                   note="coarse terrain breakup; no per-node variation, it is one mesh")
    r, g, b = ground.base.diffuse
    ground.add(TexNode("base", "VRayColor",
                      params={"red": r / 255.0, "green": g / 255.0,
                              "blue": b / 255.0}))
    ground.add(TexNode("coarse", "Noise", params={"size": 8.0, "levels": 4.0}))
    ground.add(_triplanar("coarse_world", "coarse", 8.0))
    ground.add(TexNode("fine", "Speckle", params={"size": 0.6}))
    ground.add(_triplanar("fine_world", "fine", 0.6))
    ground.add(TexNode("mixed", "Mix", params={"mixAmount": 0.4},
                       inputs={"map1": "coarse_world", "map2": "fine_world"}))
    ground.add(TexNode("tinted", "Mix", params={"mixAmount": 0.35},
                       inputs={"map1": "base", "map2": "mixed"}))
    ground.bind("texmap_diffuse", "tinted")
    ground.add(TexNode("relief", "Noise", params={"size": 3.0, "bump_multiplier": 14.0}))
    ground.add(_triplanar("relief_world", "relief", 3.0))
    ground.bind("texmap_bump", "relief_world")
    graphs["ground"] = ground

    return graphs


GRAPHS = _build_graphs()


def graph_for_building(building) -> Graph:
    """
    The graph a building should be shaded with.

    Resolution goes through ``materials.material_name_for``, so the tag
    precedence — and the grouping that collapses a city to a handful of
    materials — stays defined in exactly one place.
    """
    key = materials.material_name_for(getattr(building, "tags", {}) or {})
    return GRAPHS[key]
