"""
OSM tags -> V-Ray materials.

OSM already carries material hints — ``building:material``, ``roof:material``,
``building`` type — so massing does not have to arrive uniformly grey. These are
procedural ``VRayMtl`` definitions with no texture dependencies: Chaos Cosmos
assets download on demand and are not guaranteed present, so they can be an
enhancement but never a requirement.

**Every parameter name here was read from a live VRayMtl**, not recalled. That
matters more for materials than for most things, because until this was fixed
the bridge could not tell a real attribute from a typo: a material's colour
parameters are not in ``getPropNames``, so they are reached with ``setattr``,
and ``setattr`` on a pymxs wrapper accepts any name at all while ``getattr``
reads it straight back. A probe with ``bogus_param_xyz`` reported itself
applied. The handler now checks ``hasattr`` first and returns a ``rejected``
map, so a misspelled parameter is loud instead of silently defaulted.

**Materials are shared, not per-building.** A thousand footprints must not
become a thousand VRayMtl instances — that bloats the scene, slows the viewport
and makes a later look-development pass impossible. :func:`group_by_material`
collapses them to one material per distinct tag, typically fewer than ten.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MaterialSpec",
    "PRESETS",
    "material_name_for",
    "spec_for_building",
    "group_by_material",
    "assign_materials",
    "assign_terrain_material",
    "DEFAULT_MATERIAL",
    "TERRAIN_MATERIAL",
]

DEFAULT_MATERIAL = "concrete"


@dataclass(frozen=True)
class MaterialSpec:
    """
    A procedural VRayMtl, in real V-Ray 7 parameter names.

    Colours are 0-255 RGB, matching MaxScript's Color. Glossiness is V-Ray's
    convention where 1.0 is a mirror and lower is rougher — the opposite of the
    "roughness" convention used by most PBR pipelines, which is a reliable way
    to produce a scene where every surface is wrong in the same direction.
    """

    name: str
    diffuse: tuple[int, int, int]
    reflection: tuple[int, int, int] = (20, 20, 20)
    reflection_glossiness: float = 0.5
    reflection_ior: float = 1.5
    reflection_metalness: float = 0.0
    note: str = ""
    coat_amount: float = 0.0
    coat_color: tuple[int, int, int] = (255, 255, 255)
    coat_glossiness: float = 1.0
    coat_ior: float = 1.6
    coat_darkening: float = 0.0

    def to_params(self) -> dict:
        """Bridge-ready parameter dict, using colour wrappers the handler rebuilds."""
        params = {
            "diffuse": {"__color__": list(self.diffuse)},
            "reflection": {"__color__": list(self.reflection)},
            "reflection_glossiness": self.reflection_glossiness,
            "reflection_ior": self.reflection_ior,
            "reflection_metalness": self.reflection_metalness,
        }
        if self.coat_amount > 0.0:
            params.update({
                "coat_amount": self.coat_amount,
                "coat_color": {"__color__": list(self.coat_color)},
                "coat_glossiness": self.coat_glossiness,
                "coat_ior": self.coat_ior,
                "coat_darkening": self.coat_darkening,
            })
        return params


# Values are plausible mid-range approximations for massing, not measured
# libraries. The point is that a brick terrace and a glass tower read
# differently in a sun study, not that either is a match for a real sample.
PRESETS: dict[str, MaterialSpec] = {
    "concrete": MaterialSpec(
        "atlas_concrete", (142, 140, 134), (24, 24, 24), 0.45, 1.5,
        note="default for untagged massing",
    ),
    "brick": MaterialSpec("atlas_brick", (138, 78, 60), (16, 16, 16), 0.35, 1.5),
    "stone": MaterialSpec("atlas_stone", (152, 146, 133), (20, 20, 20), 0.35, 1.5),
    "plaster": MaterialSpec("atlas_plaster", (203, 197, 186), (12, 12, 12), 0.25, 1.5),
    "wood": MaterialSpec("atlas_wood", (122, 86, 55), (22, 22, 22), 0.45, 1.5),
    "metal": MaterialSpec(
        "atlas_metal", (122, 124, 128), (196, 198, 202), 0.80, 3.0, 1.0,
        note="metalness 1.0 — diffuse is ignored by V-Ray for a full metal",
    ),
    "glass": MaterialSpec(
        "atlas_glass", (18, 22, 26), (78, 84, 90), 0.95, 1.52,
        note=(
            "reflective but opaque. Massing blocks are solid volumes, and a "
            "refractive glass on one would show the building's own interior "
            "faces — slower to render and wrong to look at."
        ),
    ),
    "roof_tile": MaterialSpec("atlas_roof_tile", (128, 68, 52), (16, 16, 16), 0.30, 1.5),
    "asphalt": MaterialSpec("atlas_asphalt", (52, 52, 55), (18, 18, 18), 0.25, 1.5),
    "track_asphalt": MaterialSpec(
        "atlas_track_asphalt", (41, 41, 44), (30, 30, 30), 0.62, 1.5,
        note=(
            "racing surface, not road paving. Darker and markedly smoother "
            "than the `asphalt` preset: a circuit is laid in a fine-graded "
            "wearing course and then polished along the racing line by the "
            "cars themselves, so it holds a low-sun specular sheet that road "
            "asphalt scatters away. That sheet is most of what makes a track "
            "read as a track at grazing light."
        ),
    ),
    "tyre": MaterialSpec(
        "atlas_tyre", (17, 17, 19), (34, 34, 36), 0.36, 1.52,
        note=(
            "racing slick. The mistake with rubber is making it matte black: "
            "a real tyre is very dark but not flat, and carries a broad soft "
            "sheen across the shoulder that is most of what says 'rubber' "
            "rather than 'painted plastic'. Diffuse near 17/255 with a weak, "
            "rough reflection gives that; a glossy black gives a bowling ball "
            "and a matte black gives a hole in the image."
        ),
    ),
    "rim": MaterialSpec(
        "atlas_rim", (58, 60, 64), (96, 98, 102), 0.55, 3.0, 0.55,
        note=(
            "dark anodised wheel face. Metalness was 1.0 with a bright "
            "reflection, and a full mirror at that angle did the only thing a "
            "mirror can: it reflected the desert. The rims rendered as pale "
            "sand-coloured discs, brighter than the tyres around them, which "
            "reads as a hole in the wheel rather than a wheel.\n\n"
            "A real race rim is dark, anodised and largely shadowed inside the "
            "tyre. Metalness 0.55 keeps some metal character while letting the "
            "dark base colour carry, and the rougher 0.55 gloss breaks the "
            "mirror into a sheen."
        ),
    ),
    "car_body": MaterialSpec(
        "atlas_car_body", (28, 30, 36), (58, 60, 64), 0.90, 1.52,
        note=(
            "generic single-seater bodywork — no livery, deliberately.\n\n"
            "**A scanned texture is the wrong tool here and none is used.** "
            "Race bodywork is painted carbon: smooth, and essentially "
            "featureless at any scale a camera sees. Putting a scanned surface "
            "on it would add dirt and weave that are not there. What makes a "
            "car read as a car is the *coat* — a tight, near-mirror specular "
            "that travels along the panel as the camera moves — and that is a "
            "reflection property, not a map.\n\n"
            "So the gloss goes back up to 0.90 with a modest reflection colour "
            "and an ior of 1.52, which is clearcoat lacquer. The earlier "
            "(128,130,136) at 0.88 was not wrong because it was glossy; it was "
            "wrong because a 50% grey reflection re-broadcasts the whole sky. "
            "Dark reflection, high gloss: a highlight rather than a floodlight.\n\n"
            "The second, clean coat is a physical layer rather than a bitmap. "
            "All five scalar `coat_*` names were read from the live V-Ray 7 "
            "material before this preset used them; the coat is 0.80 strength, "
            "neutral, 0.96 gloss and IOR 1.52."
        ),
        coat_amount=0.80,
        coat_glossiness=0.96,
        coat_ior=1.52,
        coat_darkening=0.0,
    ),
    "line_paint": MaterialSpec(
        "atlas_line_paint", (232, 231, 226), (18, 18, 18), 0.30, 1.5,
        note=(
            "track edge lines and grid boxes. Built as thin geometry rather "
            "than painted into the asphalt map: a line has to stay parallel to "
            "an edge through every corner, and the world-space triplanar the "
            "other graphs use cannot hold that. Geometry follows the curve for "
            "free and needs no texmap parameter that could not be verified "
            "offline."
        ),
    ),
    "kerb": MaterialSpec(
        "atlas_kerb", (196, 196, 196), (30, 30, 30), 0.55, 1.5,
        note=(
            "the red/white banding comes from the texture graph, not from this "
            "colour — a kerb is two colours and a MaterialSpec holds one. The "
            "base is the white band; the red is mixed over it in UV space so "
            "the stripes run along the kerb as it bends."
        ),
    ),
    "grandstand": MaterialSpec(
        "atlas_grandstand", (74, 88, 112), (26, 26, 26), 0.40, 1.5,
        note=(
            "banked seating, not the structure holding it up. Reads as the "
            "seats themselves — Yas Marina's are a desaturated blue — because "
            "from any distance a grandstand *is* its seating deck: a raked "
            "plane of thousands of small units whose colour is nothing like "
            "the concrete frame beneath. Rendered as concrete they disappear "
            "into the pit buildings around them."
        ),
    ),
    "ground": MaterialSpec(
        "atlas_ground", (118, 112, 98), (10, 10, 10), 0.20, 1.5,
        note=(
            "the terrain sheet. Nearly matte on purpose — a glossy ground "
            "throws a specular sheet back at the camera at low sun angles, "
            "which is exactly when sun studies are being looked at."
        ),
    ),
}

TERRAIN_MATERIAL = "ground"

# OSM's building:material values, which are free text in practice. British and
# American spellings, plurals and near-synonyms all appear in the same city.
_MATERIAL_SYNONYMS = {
    "concrete": "concrete",
    "reinforced_concrete": "concrete",
    "cement": "concrete",
    "cement_block": "concrete",
    "precast_concrete": "concrete",
    "brick": "brick",
    "bricks": "brick",
    "brickwork": "brick",
    "clay": "brick",
    "stone": "stone",
    "sandstone": "stone",
    "limestone": "stone",
    "granite": "stone",
    "marble": "stone",
    "masonry": "stone",
    "travertine": "stone",
    "plaster": "plaster",
    "render": "plaster",
    "stucco": "plaster",
    "cement_render": "plaster",
    "wood": "wood",
    "timber": "wood",
    "timber_framing": "wood",
    "log": "wood",
    "metal": "metal",
    "steel": "metal",
    "aluminium": "metal",
    "aluminum": "metal",
    "copper": "metal",
    "zinc": "metal",
    "corrugated_iron": "metal",
    "glass": "glass",
    "mirror": "glass",
    "tile": "roof_tile",
    "tiles": "roof_tile",
    "roof_tiles": "roof_tile",
    "slate": "roof_tile",
    "asphalt": "asphalt",
    "tar_paper": "asphalt",
}

# When nothing is tagged, the building's *type* is a weak but real signal. A
# glass office tower and a plastered house are different enough that guessing
# beats defaulting everything to concrete.
_TYPE_DEFAULTS = {
    "house": "plaster",
    "detached": "plaster",
    "residential": "plaster",
    "apartments": "concrete",
    "terrace": "brick",
    "commercial": "glass",
    "office": "glass",
    "retail": "concrete",
    "industrial": "metal",
    "warehouse": "metal",
    "hangar": "metal",
    "shed": "metal",
    "garage": "concrete",
    "garages": "concrete",
    "church": "stone",
    "cathedral": "stone",
    "chapel": "stone",
    "mosque": "stone",
    "temple": "stone",
    "synagogue": "stone",
    "castle": "stone",
    "ruins": "stone",
    "barn": "wood",
    "farm": "wood",
    "hut": "wood",
    "cabin": "wood",
    "grandstand": "grandstand",
    "stadium": "grandstand",
}

# Names that identify a building whose *type* tag is uninformative. At Yas
# Marina three of the five grandstands carry `building=grandstand` and the other
# two are plain `building=yes` — same structure, same seating, tagged by
# different mappers. Matching the name recovers them.
#
# Deliberately narrow: only words that name a building typology, never a brand
# or a place. "Marina Grandstand" matches on `grandstand`; "Yas Island Rotana"
# matches nothing and stays a default.
_NAME_HINTS = (
    ("grandstand", "grandstand"),
    ("tribune", "grandstand"),
)


def material_name_for(tags: dict) -> str:
    """
    Resolve a preset key from a building's tags.

    Precedence follows how much the mapper actually asserted:

    1. ``building:material`` — a statement about this building's fabric.
    2. the ``building`` type — a category with a typical construction.
    3. :data:`DEFAULT_MATERIAL`.

    ``roof:material`` is deliberately *not* consulted. It describes the roof,
    and a massing block has no separate roof surface to assign it to, so using
    it would clad the walls in roof tiles.
    """
    raw = str(tags.get("building:material") or "").strip().lower()
    if raw:
        # Multi-valued tags appear as "brick;concrete"; the first wins.
        first = raw.split(";")[0].strip().replace(" ", "_")
        if first in _MATERIAL_SYNONYMS:
            return _MATERIAL_SYNONYMS[first]

    kind = str(tags.get("building") or "").strip().lower()
    if kind in _TYPE_DEFAULTS:
        return _TYPE_DEFAULTS[kind]

    name = str(tags.get("name") or "").strip().lower()
    if name:
        for needle, preset in _NAME_HINTS:
            if needle in name:
                return preset

    return DEFAULT_MATERIAL


def spec_for_building(building) -> MaterialSpec:
    """The :class:`MaterialSpec` a building should be shaded with."""
    return PRESETS[material_name_for(getattr(building, "tags", {}) or {})]


def group_by_material(pairs) -> dict[str, list[str]]:
    """
    Collapse ``(building, node_name)`` pairs into one group per material.

    The whole point: a thousand footprints become fewer than ten materials.
    Assigning per building would create a thousand VRayMtl instances, which
    bloats the file, slows the viewport and makes any later look-development
    pass impractical.

    Groups come back in a stable order so a rebuild produces the same scene.
    """
    groups: dict[str, list[str]] = {}
    for building, node_name in pairs:
        key = material_name_for(getattr(building, "tags", {}) or {})
        groups.setdefault(key, []).append(node_name)
    return {k: groups[k] for k in sorted(groups)}


def assign_materials(bridge, pairs, *, chunk: int = 200) -> dict:
    """
    Create one material per group and assign it to that group's nodes.

    Chunked because a single command holds the 3ds Max main thread for its
    whole duration, and a few thousand node names in one call freezes the UI
    long enough to look like a hang.

    Reports anything the host *rejected*. That map is the reason this is
    trustworthy: a parameter name V-Ray does not have is now refused rather
    than silently defaulted, so an empty ``rejected`` means the values really
    landed.
    """
    groups = group_by_material(pairs)
    applied: dict[str, dict] = {}
    rejected: dict[str, dict] = {}

    for key, nodes in groups.items():
        spec = PRESETS[key]
        params = spec.to_params()
        for start in range(0, len(nodes), chunk):
            batch = nodes[start:start + chunk]
            result = bridge.assign_material(batch, params=params, name=spec.name)
            applied.setdefault(key, result.get("applied", {}))
            if result.get("rejected"):
                rejected.setdefault(key, {}).update(result["rejected"])

    return {
        "materials": len(groups),
        "nodes": sum(len(v) for v in groups.values()),
        "groups": {k: len(v) for k, v in groups.items()},
        "applied": applied,
        "rejected": rejected,
    }


def assign_terrain_material(bridge, node_name: str = "atlas_terrain") -> dict:
    """
    Shade the terrain sheet.

    Separate from :func:`assign_materials` because terrain has no OSM tags to
    resolve from — but it is not optional. A mesh created through the bridge
    carries whatever wirecolor Max assigned it, so an unshaded terrain renders
    as a saturated slab of arbitrary colour under an otherwise plausible city,
    which reads as a bug in the lighting rather than a missing material.
    """
    spec = PRESETS[TERRAIN_MATERIAL]
    result = bridge.assign_material(node_name, params=spec.to_params(), name=spec.name)
    return {
        "node": node_name,
        "material": spec.name,
        "applied": result.get("applied", {}),
        "rejected": result.get("rejected", {}),
    }
