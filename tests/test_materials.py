"""
Material resolution tests — no 3ds Max.

The two things worth guarding here are cheap to get wrong and expensive to
notice. One material per building instead of one per *kind* produces a scene
that renders identically and is unusable for look development. And every
parameter name must be one the live host actually has, because until the bridge
started checking, a typo was accepted and silently defaulted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from materials import (  # noqa: E402
    DEFAULT_MATERIAL,
    PRESETS,
    assign_materials,
    assign_terrain_material,
    group_by_material,
    material_name_for,
    spec_for_building,
)
from osm import Building  # noqa: E402

RING = [(25.0, 55.0), (25.0, 55.001), (25.001, 55.001), (25.001, 55.0)]


def _building(**tags):
    return Building(1, "way", RING, tags=tags)


# ── Tag resolution ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value,expected",
    [
        ("brick", "brick"), ("bricks", "brick"), ("brickwork", "brick"),
        ("concrete", "concrete"), ("reinforced_concrete", "concrete"),
        ("glass", "glass"), ("steel", "metal"), ("aluminium", "metal"),
        ("aluminum", "metal"), ("sandstone", "stone"), ("marble", "stone"),
        ("timber_framing", "wood"), ("stucco", "plaster"), ("slate", "roof_tile"),
    ],
)
def test_material_synonyms(value, expected):
    """OSM's building:material is free text; spellings and plurals both appear."""
    assert material_name_for({"building:material": value}) == expected


def test_material_matching_is_case_insensitive():
    assert material_name_for({"building:material": "BRICK"}) == "brick"


def test_multivalued_material_takes_the_first():
    assert material_name_for({"building:material": "brick;concrete"}) == "brick"


def test_spaces_are_normalised():
    assert material_name_for({"building:material": "corrugated iron"}) == "metal"


def test_unknown_material_falls_back_to_the_type_then_the_default():
    assert material_name_for({"building:material": "unobtainium"}) == DEFAULT_MATERIAL
    assert material_name_for(
        {"building:material": "unobtainium", "building": "church"}
    ) == "stone"


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("house", "plaster"), ("terrace", "brick"), ("office", "glass"),
        ("warehouse", "metal"), ("cathedral", "stone"), ("barn", "wood"),
    ],
)
def test_building_type_is_a_fallback_signal(kind, expected):
    """Weak but real: a glass office and a plastered house should differ."""
    assert material_name_for({"building": kind}) == expected


def test_explicit_material_beats_the_type_guess():
    assert material_name_for({"building": "office", "building:material": "brick"}) == "brick"


def test_untagged_building_gets_the_default():
    assert material_name_for({}) == DEFAULT_MATERIAL
    assert material_name_for({"building": "yes"}) == DEFAULT_MATERIAL


def test_roof_material_does_not_clad_the_walls():
    """
    A massing block has no separate roof surface, so honouring roof:material
    would wrap the whole building in roof tiles.
    """
    assert material_name_for({"roof:material": "slate"}) == DEFAULT_MATERIAL
    assert material_name_for(
        {"building:material": "brick", "roof:material": "slate"}
    ) == "brick"


def test_spec_for_building_reads_the_tags():
    assert spec_for_building(_building(**{"building:material": "glass"})).name == "atlas_glass"


# ── Preset sanity ─────────────────────────────────────────────────────────────

def test_every_preset_has_a_unique_scene_name():
    names = [s.name for s in PRESETS.values()]
    assert len(names) == len(set(names))


def test_every_synonym_target_exists():
    """A synonym pointing at a missing preset would KeyError at build time."""
    import materials

    for target in set(materials._MATERIAL_SYNONYMS.values()):
        assert target in PRESETS
    for target in set(materials._TYPE_DEFAULTS.values()):
        assert target in PRESETS


def test_parameters_are_names_the_live_host_has():
    """
    Discovered from a live VRayMtl, not recalled. A name V-Ray does not have is
    now rejected by the bridge rather than silently defaulted — but it should
    not be sent in the first place.
    """
    verified = {
        "diffuse", "reflection", "reflection_glossiness",
        "reflection_ior", "reflection_metalness", "coat_amount", "coat_color",
        "coat_glossiness", "coat_ior", "coat_darkening",
    }
    for spec in PRESETS.values():
        assert set(spec.to_params()) <= verified, f"{spec.name} sends an unverified name"


def test_glossiness_is_in_range():
    for spec in PRESETS.values():
        assert 0.0 <= spec.reflection_glossiness <= 1.0
        assert 0.0 <= spec.reflection_metalness <= 1.0


def test_colours_are_valid_bytes():
    for spec in PRESETS.values():
        for channel in (*spec.diffuse, *spec.reflection):
            assert 0 <= channel <= 255


def test_glass_is_reflective_not_refractive():
    """
    Deliberate. A massing block is a solid volume; refractive glass on one shows
    the building's own interior faces — slower and wrong to look at.
    """
    params = PRESETS["glass"].to_params()
    assert "refraction" not in params
    assert PRESETS["glass"].reflection_glossiness > 0.9


def test_metal_is_flagged_as_metal():
    assert PRESETS["metal"].reflection_metalness == 1.0


def test_rougher_materials_are_less_glossy_than_smoother_ones():
    """Ordering is the part a render actually shows."""
    assert PRESETS["plaster"].reflection_glossiness < PRESETS["concrete"].reflection_glossiness
    assert PRESETS["concrete"].reflection_glossiness < PRESETS["glass"].reflection_glossiness


def test_colour_wrappers_are_the_shape_the_bridge_rebuilds():
    params = PRESETS["brick"].to_params()
    assert params["diffuse"] == {"__color__": [138, 78, 60]}


def test_car_body_uses_the_live_verified_clearcoat_controls():
    """Moving bodywork needs scalar coat, not a world-space texture that swims."""
    params = PRESETS["car_body"].to_params()
    assert params["coat_amount"] > 0.0
    assert params["coat_glossiness"] > params["reflection_glossiness"]
    assert params["coat_ior"] == pytest.approx(1.52)
    assert params["coat_color"] == {"__color__": [255, 255, 255]}


# ── Grouping ──────────────────────────────────────────────────────────────────

def _pairs(n, **tags):
    return [(_building(**tags), f"osm_w{i}") for i in range(n)]


def test_buildings_of_one_kind_share_one_material():
    """
    The point of the module. One material per building means a thousand VRayMtl
    instances, which renders the same and is unusable for look development.
    """
    groups = group_by_material(_pairs(500, **{"building:material": "brick"}))
    assert list(groups) == ["brick"]
    assert len(groups["brick"]) == 500


def test_different_kinds_are_separated():
    pairs = (
        _pairs(3, **{"building:material": "brick"})
        + _pairs(2, **{"building:material": "glass"})
        + _pairs(4, **{"building": "warehouse"})
    )
    groups = group_by_material(pairs)
    assert {k: len(v) for k, v in groups.items()} == {"brick": 3, "glass": 2, "metal": 4}


def test_grouping_is_stable_across_runs():
    """A rebuild must produce the same scene, not a reshuffled one."""
    pairs = _pairs(2, **{"building:material": "glass"}) + _pairs(2, **{"building:material": "brick"})
    assert list(group_by_material(pairs)) == list(group_by_material(pairs))
    assert list(group_by_material(pairs)) == ["brick", "glass"]


def test_grouping_an_empty_scene_is_not_an_error():
    assert group_by_material([]) == {}


# ── Assignment ────────────────────────────────────────────────────────────────

class _FakeBridge:
    def __init__(self, reject=None):
        self.calls = []
        self.reject = reject or {}

    def assign_material(self, nodes, *, params=None, name=None, **_kw):
        # Mirrors MaxBridge.assign_material, which wraps a bare string. Calling
        # list() on one shreds it into characters, which is a bug in the stub
        # rather than in the code under test.
        nodes = [nodes] if isinstance(nodes, str) else list(nodes)
        self.calls.append({"nodes": nodes, "params": params, "name": name})
        return {"applied": dict(params or {}), "rejected": dict(self.reject)}


def test_one_assign_call_per_material_not_per_building():
    bridge = _FakeBridge()
    pairs = _pairs(50, **{"building:material": "brick"}) + _pairs(30, **{"building:material": "glass"})
    result = assign_materials(bridge, pairs)
    assert result["materials"] == 2
    assert result["nodes"] == 80
    assert len(bridge.calls) == 2


def test_large_groups_are_chunked_to_keep_the_ui_responsive():
    """A command holds the Max main thread for its whole duration."""
    bridge = _FakeBridge()
    assign_materials(bridge, _pairs(450, **{"building:material": "brick"}), chunk=200)
    assert len(bridge.calls) == 3
    assert [len(c["nodes"]) for c in bridge.calls] == [200, 200, 50]


def test_every_node_is_assigned_exactly_once():
    bridge = _FakeBridge()
    assign_materials(bridge, _pairs(450, **{"building:material": "brick"}), chunk=200)
    seen = [n for call in bridge.calls for n in call["nodes"]]
    assert len(seen) == 450 == len(set(seen))


def test_the_material_name_is_stable_per_kind():
    bridge = _FakeBridge()
    assign_materials(bridge, _pairs(10, **{"building:material": "glass"}))
    assert {c["name"] for c in bridge.calls} == {"atlas_glass"}


def test_rejected_parameters_are_surfaced_not_swallowed():
    """
    An empty rejected map is what makes 'applied' believable. Hiding it would
    restore exactly the silent-default behaviour the bridge fix removed.
    """
    bridge = _FakeBridge(reject={"reflection_glosiness": "no attribute"})
    result = assign_materials(bridge, _pairs(3, **{"building:material": "brick"}))
    assert result["rejected"]["brick"] == {"reflection_glosiness": "no attribute"}


def test_a_clean_assignment_reports_nothing_rejected():
    result = assign_materials(_FakeBridge(), _pairs(3, **{"building:material": "brick"}))
    assert result["rejected"] == {}


# ── Terrain ───────────────────────────────────────────────────────────────────

def test_terrain_gets_shaded_too():
    """
    An unshaded terrain renders as a saturated slab of whatever wirecolor Max
    assigned, under an otherwise plausible city — it reads as a lighting bug.
    """
    bridge = _FakeBridge()
    result = assign_terrain_material(bridge)
    assert result["material"] == "atlas_ground"
    assert bridge.calls[0]["nodes"] == ["atlas_terrain"]


def test_terrain_material_node_is_overridable():
    bridge = _FakeBridge()
    assign_terrain_material(bridge, "other_terrain")
    assert bridge.calls[0]["nodes"] == ["other_terrain"]


def test_ground_is_nearly_matte():
    """
    A glossy ground throws a specular sheet at the camera at low sun angles —
    precisely when a sun study is being looked at.
    """
    assert PRESETS["ground"].reflection_glossiness <= 0.25


def test_terrain_rejections_are_surfaced():
    bridge = _FakeBridge(reject={"nope": "no attribute"})
    assert assign_terrain_material(bridge)["rejected"] == {"nope": "no attribute"}
