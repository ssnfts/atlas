"""
Procedural material graph tests — no 3ds Max.

The highest-value test in this file is the one asserting every texmap class and
every VRayMtl slot name came from the live-host manifest. MaxScript silently
ignores a write to a property that does not exist, so a misremembered name
produces an untextured render with nothing in any log to explain it — the
project's signature failure, and the reason a design agent refused to write this
module until the names had actually been discovered.

The second is the `_on` flag. Every texmap slot has an `_on` companion that
defaults to **False**, so a graph that attaches a map without setting it renders
exactly as if the map were absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import materials  # noqa: E402
from osm import Building  # noqa: E402
from texturing import (  # noqa: E402
    CHANNELS,
    GRAPHS,
    MAX_BUMP,
    MAX_HUE_JITTER_DEG,
    TEXMAP_CLASSES,
    TEXMAP_SLOTS,
    Graph,
    TexNode,
    TexturingError,
    graph_for_building,
    validate_graph,
)

RING = [(25.0, 55.0), (25.0, 55.001), (25.001, 55.001), (25.001, 55.0)]


# ── Names must have been discovered, not recalled ─────────────────────────────

def test_every_texmap_class_was_confirmed_on_the_host():
    """
    The load-bearing test. A class name that is not on the live host fails
    silently in MaxScript — the map is never created and the render is flat.
    """
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            assert node.cls in TEXMAP_CLASSES, (
                f"{graph.key}/{node.id} uses {node.cls!r}, never confirmed present"
            )


def test_every_channel_is_a_confirmed_vraymtl_slot():
    for graph in GRAPHS.values():
        for slot in graph.channels:
            assert slot in TEXMAP_SLOTS, f"{graph.key} writes unconfirmed slot {slot!r}"


def test_listed_but_unconstructible_classes_are_never_used():
    """
    Regression, and the real lesson of this module. `textureMap.classes` lists
    classes that cannot actually be instantiated — `Wood` and
    `fallofftextureMap` among them. The first version of this file took both
    from that list and would have failed at build time: the Fresnel map is
    really `Falloff`, and there is no constructible wood map on this host.

    Membership in a class list is not usability. These were verified by
    constructing each one.
    """
    from texturing import NOT_CONSTRUCTIBLE

    assert NOT_CONSTRUCTIBLE.isdisjoint(TEXMAP_CLASSES)
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            assert node.cls not in NOT_CONSTRUCTIBLE, (
                f"{graph.key}/{node.id} uses {node.cls!r}, which is listed by the "
                "host but cannot be constructed"
            )


def test_glass_fresnel_uses_the_constructible_falloff_name():
    nodes = {n.cls for n in GRAPHS["glass"].nodes.values()}
    assert "Falloff" in nodes
    assert "fallofftextureMap" not in nodes


def test_texture_tiles_is_not_used():
    """
    `TextureTiles` was probed on the host and is absent — brick banding has to
    come from Checker instead. Asserting the absence keeps a plausible-looking
    name from creeping back in.
    """
    assert "TextureTiles" not in TEXMAP_CLASSES
    for graph in GRAPHS.values():
        assert all(n.cls != "TextureTiles" for n in graph.nodes.values())


def test_validation_rejects_an_invented_class():
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("bad", "VRayTotallyRealTex"))
    graph.bind("texmap_diffuse", "bad")
    assert any("not confirmed present" in p for p in validate_graph(graph))


def test_validation_rejects_an_invented_slot():
    """`texmap_colour` is the kind of near-miss that reads as correct."""
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("ok", "Noise"))
    graph.bind("texmap_colour", "ok")
    assert any("not a confirmed VRayMtl texmap slot" in p for p in validate_graph(graph))


# ── The _on flag ──────────────────────────────────────────────────────────────

def test_every_bound_channel_emits_its_on_flag():
    """
    `_on` defaults False. A map attached without it is silently unused, which
    renders identically to having no map at all.
    """
    for graph in GRAPHS.values():
        writes = graph.slot_writes()
        for slot in graph.channels:
            assert writes.get(f"{slot}_on") is True, f"{graph.key}: {slot} has no _on"


def test_slot_writes_reference_real_nodes():
    for graph in GRAPHS.values():
        for slot, value in graph.slot_writes().items():
            if slot.endswith("_on"):
                continue
            assert value["__node_ref__"] in graph.nodes


def test_on_flags_cannot_be_separated_from_the_map():
    """A graph binding one channel must emit exactly one map and one flag."""
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("n", "Noise"))
    graph.bind("texmap_bump", "n")
    assert graph.slot_writes() == {
        "texmap_bump": {"__node_ref__": "n"}, "texmap_bump_on": True
    }


# ── Graph structure ───────────────────────────────────────────────────────────

def test_every_graph_key_is_a_real_material():
    """A graph for a key materials.py does not have could never be selected."""
    assert set(GRAPHS) <= set(materials.PRESETS)


def test_every_material_a_building_can_resolve_to_has_a_graph():
    """
    The one that matters: material_name_for can only ever return a key from the
    synonym and type tables, and every one of those must have a graph or that
    building silently falls back to flat colour while its neighbours are
    textured — visible as a patch of plastic in the middle of a shaded street.
    """
    reachable = set(materials._MATERIAL_SYNONYMS.values())
    reachable |= set(materials._TYPE_DEFAULTS.values())
    reachable.add(materials.DEFAULT_MATERIAL)
    reachable.add(materials.TERRAIN_MATERIAL)
    missing = reachable - set(GRAPHS)
    assert not missing, f"buildings can resolve to {missing} but no graph exists"


def test_graphs_are_valid():
    for graph in GRAPHS.values():
        assert validate_graph(graph) == [], f"{graph.key}: {validate_graph(graph)}"


def test_no_dangling_input_references():
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            for name, ref in node.inputs.items():
                assert ref in graph.nodes, f"{graph.key}/{node.id}.{name} -> {ref!r}"


def test_a_cycle_is_detected():
    """
    A cycle would hang the host builder on Max's main thread, which locks the
    whole application rather than raising.
    """
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("a", "Mix", inputs={"map1": "b"}))
    graph.add(TexNode("b", "Mix", inputs={"map1": "a"}))
    assert any("cycle" in p for p in validate_graph(graph))


def test_a_self_reference_is_a_cycle():
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("a", "Mix", inputs={"map1": "a"}))
    assert any("cycle" in p for p in validate_graph(graph))


def test_duplicate_node_ids_are_refused():
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("a", "Noise"))
    with pytest.raises(TexturingError, match="duplicate"):
        graph.add(TexNode("a", "Noise"))


def test_every_graph_drives_diffuse_and_bump():
    """
    Bump is what sells surface at grazing angles, which is most of a sun study.
    A graph without it is a flat colour with extra steps.
    """
    for graph in GRAPHS.values():
        assert "texmap_diffuse" in graph.channels, f"{graph.key} has no diffuse"
        assert "texmap_bump" in graph.channels, f"{graph.key} has no bump"


def test_channels_stay_within_the_declared_set():
    for graph in GRAPHS.values():
        assert set(graph.channels) <= set(CHANNELS)


# ── World-space projection ────────────────────────────────────────────────────

def test_patterns_are_projected_in_world_space():
    """
    VRayTriplanarTex is what makes these graphs work on meshes with NO UVs —
    which is every building Atlas currently produces. Without it the maps land
    on default coordinates and stretch unpredictably.
    """
    for graph in GRAPHS.values():
        assert any(n.cls == "VRayTriplanarTex" for n in graph.nodes.values()), (
            f"{graph.key} has no world-space projection and needs UVs it does not have"
        )


def test_projection_sizes_are_physical_and_plausible():
    """
    Sizes are metres of real-world pattern. A brick course is ~0.2 m and a
    terrain undulation is ~10 m; anything outside that band is a unit error.
    """
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            if node.cls != "VRayTriplanarTex":
                continue
            size = node.params["size"]
            assert 0.005 <= size <= 20.0, f"{graph.key}/{node.id} size {size} m"


def test_brick_pattern_is_finer_than_concrete():
    """Ordering is what a render shows: brick courses are smaller than board marks."""
    def size(key):
        return min(n.params["size"] for n in GRAPHS[key].nodes.values()
                   if n.cls == "VRayTriplanarTex")

    assert size("brick") < size("concrete")
    assert size("plaster") < size("stone")


# ── Bounded variation ─────────────────────────────────────────────────────────

def test_per_building_variation_preserves_shared_materials():
    """
    The design constraint from materials.py: 1281 buildings collapse to ~7
    materials. Variation comes from VRayMultiSubTex varying per *node*, not
    from one material per building, which would regress the whole scheme.
    """
    for key in ("concrete", "brick", "stone", "plaster", "glass", "metal", "wood"):
        assert any(n.cls == "VRayMultiSubTex" for n in GRAPHS[key].nodes.values()), (
            f"{key} has no per-node variation"
        )


def test_ground_has_no_per_node_variation():
    """The terrain is a single mesh; per-node variation would cost a lookup for nothing."""
    assert not any(n.cls == "VRayMultiSubTex" for n in GRAPHS["ground"].nodes.values())


def test_hue_jitter_is_bounded():
    """
    Beyond ~12 degrees terracotta goes pink and green and a brick street stops
    reading as one material.
    """
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            hue = node.params.get("hue_shift")
            if hue is not None:
                assert abs(hue) <= MAX_HUE_JITTER_DEG


def test_validation_catches_excessive_hue_jitter():
    graph = Graph("probe", materials.PRESETS["brick"])
    graph.add(TexNode("wild", "VRayMultiSubTex", params={"hue_shift": 90.0}))
    assert any("hue jitter" in p for p in validate_graph(graph))


def test_bump_is_bounded():
    """An unbounded bump on massing reads as a rendering artefact, not surface."""
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            bump = node.params.get("bump_multiplier")
            if bump is not None:
                assert 0.0 <= bump <= MAX_BUMP


def test_validation_catches_excessive_bump():
    graph = Graph("probe", materials.PRESETS["concrete"])
    graph.add(TexNode("loud", "Noise", params={"bump_multiplier": 500.0}))
    assert any("bump" in p for p in validate_graph(graph))


def test_roof_tile_has_stronger_relief_than_plaster():
    """Roofs are seen at grazing angles, where relief does the most work."""
    def bump(key):
        return max(n.params.get("bump_multiplier", 0.0) for n in GRAPHS[key].nodes.values())

    assert bump("roof_tile") > bump("plaster")


# ── Weathering ────────────────────────────────────────────────────────────────

def test_masonry_is_weathered():
    """
    VRayDirt in the crevices is the single biggest reason massing stops looking
    like plastic — real buildings are dirtier where surfaces meet.
    """
    for key in ("concrete", "brick", "stone", "plaster", "roof_tile", "wood", "metal"):
        assert any(n.cls == "VRayDirt" for n in GRAPHS[key].nodes.values()), f"{key} is too clean"


def test_glass_uses_fresnel_not_dirt():
    """
    A flat reflection is the classic tell of a fake curtain wall; Fresnel
    falloff is what makes glass read as glass. Crevice dirt on a sealed
    facade would be wrong.
    """
    nodes = GRAPHS["glass"].nodes.values()
    assert any(n.cls == "Falloff" for n in nodes)
    assert not any(n.cls == "VRayDirt" for n in nodes)


def test_dirt_radii_are_physical():
    for graph in GRAPHS.values():
        for node in graph.nodes.values():
            if node.cls == "VRayDirt":
                assert 0.02 <= node.params["radius"] <= 2.0


# ── Integration with the existing material scheme ─────────────────────────────

def test_graph_base_colour_matches_the_flat_preset():
    """
    The graph tints the same colour the fallback uses, so a scene shaded either
    way reads consistently and the two cannot drift.
    """
    for key, graph in GRAPHS.items():
        assert graph.base is materials.PRESETS[key]
        base_node = graph.nodes.get("base")
        if base_node is not None:
            # VRayColor takes 0..1 float channels, not the 0..255 bytes the
            # MaterialSpec carries — a live probe rejected a plain colour list.
            assert base_node.params["red"] * 255.0 == pytest.approx(graph.base.diffuse[0])
            assert base_node.params["blue"] * 255.0 == pytest.approx(graph.base.diffuse[2])


def test_graph_for_building_follows_the_existing_tag_precedence():
    brick = Building(1, "way", RING, tags={"building:material": "brick"})
    office = Building(2, "way", RING, tags={"building": "office"})
    assert graph_for_building(brick).key == "brick"
    assert graph_for_building(office).key == "glass"


def test_untagged_building_gets_the_default_graph():
    assert graph_for_building(Building(1, "way", RING, tags={})).key == materials.DEFAULT_MATERIAL


def test_graph_serialises_for_the_host():
    data = GRAPHS["brick"].as_dict()
    assert data["key"] == "brick"
    assert data["material_name"] == "atlas_brick"
    assert data["nodes"] and data["channels"] and data["slot_writes"]
    assert all("class" in n for n in data["nodes"])
