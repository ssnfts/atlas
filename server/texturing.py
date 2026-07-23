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
    "graph_for_building",
    "validate_graph",
    "TexturingError",
]


class TexturingError(ValueError):
    """A graph references something the host does not have, or is malformed."""


# Texmap classes confirmed present on the live host (3ds Max 2027 + V-Ray GPU 7
# update 3). `TextureTiles` was probed and is NOT present, which is why brick
# banding is built from Checker rather than a dedicated tile map.
TEXMAP_CLASSES = frozenset({
    # native
    "NoTexture", "Raytrace", "Checker", "Marble", "Wood", "Dent", "Mask",
    "RGB_Tint", "Mix", "Noise", "Bitmaptexture", "Reflect_Refract",
    "Flat_Mirror", "Gradient", "CompositeTexturemap", "RGB_Multiply",
    "fallofftextureMap", "output", "Color_Correction", "MultiTile",
    "Cellular", "Speckle", "Smoke",
    # V-Ray
    "VRayDirt", "VRayTriplanarTex", "VRayNoiseTex", "VRayBitmap", "VRayColor",
    "VRayCompTex", "VRayEdgesTex", "VRayDistanceTex", "VRayMultiSubTex",
})

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

def _triplanar(node_id: str, source: str, size_m: float) -> TexNode:
    """
    World-space projection at a physical size.

    ``VRayTriplanarTex`` projects along the world axes, so the pattern lands at
    a real-world scale with **no UVs at all** — which is what makes these graphs
    usable on massing that has no texture coordinates. It also removes the
    stretching a planar projection puts on a wall that is not axis-aligned.
    """
    return TexNode(node_id, "VRayTriplanarTex",
                   params={"texture_size": size_m}, inputs={"texmap": source})


def _dirt(node_id: str, source: str, radius_m: float) -> TexNode:
    """
    Occlusion darkening — grime in the crevices and along the ground line.

    The single biggest contributor to massing not looking like plastic: real
    buildings are dirtier where surfaces meet, and a flat diffuse has none of
    that regardless of how good the colour is.
    """
    return TexNode(node_id, "VRayDirt",
                   params={"radius": radius_m}, inputs={"unoccluded_color": source})


def _per_building_variation(node_id: str, source: str, hue_deg: float) -> TexNode:
    """
    Per-node colour jitter from a single shared material.

    ``VRayMultiSubTex`` can select by node handle, so one material gives every
    building a slightly different tint. That preserves the shared-material
    design — 1281 buildings, ~7 materials — instead of regressing to one
    instance per building, which is what a naive "vary the colour" change does.
    """
    return TexNode(node_id, "VRayMultiSubTex",
                   params={"mode": "by_node_handle", "hue_shift": hue_deg},
                   inputs={"default_texmap": source})


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

    graph.add(TexNode("base", "VRayColor", params={"color": [r, g, b]}))
    graph.add(TexNode("grain", "Noise",
                      params={"size": grain_m, "levels": 3.0, "phase": 0.0}))
    graph.add(_triplanar("grain_world", "grain", grain_m))
    graph.add(TexNode("tinted", "Mix", params={"mix_amount": 0.28},
                      inputs={"map1": "base", "map2": "grain_world"}))
    graph.add(_per_building_variation("varied", "tinted", hue))
    graph.add(_dirt("weathered", "varied", dirt_m))
    graph.bind("texmap_diffuse", "weathered")

    graph.add(TexNode("relief", "Noise",
                      params={"size": grain_m * 0.4, "levels": 4.0,
                              "bump_multiplier": bump}))
    graph.add(_triplanar("relief_world", "relief", grain_m * 0.4))
    graph.bind("texmap_bump", "relief_world")

    # Weathering also dulls reflection where dirt collects, which is what stops
    # a wall reading as uniformly polished.
    graph.add(TexNode("gloss_break", "Mix", params={"mix_amount": 0.35},
                      inputs={"map1": "base", "map2": "relief_world"}))
    graph.bind("texmap_reflectionGlossiness", "gloss_break")
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

    # Wood: Marble is the wrong grain, Wood is the right one and is present.
    wood = Graph("wood", materials.PRESETS["wood"], note="plank-scale grain")
    r, g, b = wood.base.diffuse
    wood.add(TexNode("base", "VRayColor", params={"color": [r, g, b]}))
    wood.add(TexNode("grain", "Wood", params={"grain_size": 0.05, "levels": 3.0}))
    wood.add(_triplanar("grain_world", "grain", 0.05))
    wood.add(TexNode("tinted", "Mix", params={"mix_amount": 0.4},
                     inputs={"map1": "base", "map2": "grain_world"}))
    wood.add(_per_building_variation("varied", "tinted", 7.0))
    wood.add(_dirt("weathered", "varied", 0.2))
    wood.bind("texmap_diffuse", "weathered")
    wood.add(TexNode("relief", "Wood", params={"grain_size": 0.02, "bump_multiplier": 24.0}))
    wood.add(_triplanar("relief_world", "relief", 0.02))
    wood.bind("texmap_bump", "relief_world")
    graphs["wood"] = wood

    # Glass: the Fresnel falloff is what makes it read as glass at all — a flat
    # reflection is the classic tell of a fake curtain wall. Storey banding via
    # Checker, since TextureTiles was probed and is absent.
    glass = Graph("glass", materials.PRESETS["glass"],
                  note="Fresnel falloff plus storey banding; opaque, not refractive")
    r, g, b = glass.base.diffuse
    glass.add(TexNode("base", "VRayColor", params={"color": [r, g, b]}))
    glass.add(TexNode("mullions", "Checker",
                      params={"u_tiling": 1.0, "v_tiling": 1.0, "soften": 0.05}))
    glass.add(_triplanar("mullions_world", "mullions", 1.5))
    glass.add(TexNode("banded", "Mix", params={"mix_amount": 0.18},
                      inputs={"map1": "base", "map2": "mullions_world"}))
    glass.add(_per_building_variation("varied", "banded", 4.0))
    glass.bind("texmap_diffuse", "varied")
    glass.add(TexNode("fresnel", "fallofftextureMap", params={"falloff_type": "Fresnel"}))
    glass.bind("texmap_reflectionGlossiness", "fresnel")
    glass.add(TexNode("panel_relief", "Checker",
                      params={"u_tiling": 1.0, "v_tiling": 1.0, "bump_multiplier": 8.0}))
    glass.add(_triplanar("panel_world", "panel_relief", 1.5))
    glass.bind("texmap_bump", "panel_world")
    graphs["glass"] = glass

    # Metal: metalness is on the base spec; the graph supplies streaking, which
    # is what distinguishes a warehouse roof from a mirror.
    metal = Graph("metal", materials.PRESETS["metal"], note="brushed streaking, light dirt")
    r, g, b = metal.base.diffuse
    metal.add(TexNode("base", "VRayColor", params={"color": [r, g, b]}))
    metal.add(TexNode("streak", "Noise", params={"size": 0.05, "levels": 2.0}))
    metal.add(_triplanar("streak_world", "streak", 0.05))
    metal.add(TexNode("brushed", "Mix", params={"mix_amount": 0.2},
                      inputs={"map1": "base", "map2": "streak_world"}))
    metal.add(_per_building_variation("varied", "brushed", 3.0))
    metal.add(_dirt("weathered", "varied", 0.15))
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
    ground.add(TexNode("base", "VRayColor", params={"color": [r, g, b]}))
    ground.add(TexNode("coarse", "Noise", params={"size": 8.0, "levels": 4.0}))
    ground.add(_triplanar("coarse_world", "coarse", 8.0))
    ground.add(TexNode("fine", "Speckle", params={"size": 0.6}))
    ground.add(_triplanar("fine_world", "fine", 0.6))
    ground.add(TexNode("mixed", "Mix", params={"mix_amount": 0.4},
                       inputs={"map1": "coarse_world", "map2": "fine_world"}))
    ground.add(TexNode("tinted", "Mix", params={"mix_amount": 0.35},
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
