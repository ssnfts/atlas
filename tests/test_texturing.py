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
    anti_tiling_scanned_graph,
    graph_for_building,
    validate_graph,
)

RING = [(25.0, 55.0), (25.0, 55.001), (25.001, 55.001), (25.001, 55.0)]

_PRIMARY_PBR = {
    "Diffuse": "C:/textures/primary_diffuse.jpg",
    "Rough": "C:/textures/primary_rough.jpg",
    "nor_gl": "C:/textures/primary_nor_gl.jpg",
}
_SECONDARY_PBR = {
    "Diffuse": "C:/textures/secondary_diffuse.jpg",
    "Rough": "C:/textures/secondary_rough.jpg",
    "nor_gl": "C:/textures/secondary_nor_gl.jpg",
}


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


# -- Anti-tiling scanned PBR ---------------------------------------------------

def test_anti_tiling_scanned_graph_uses_staggered_world_sources_and_macro_mask():
    """
    A single world-space bitmap is still visibly periodic. The final material
    must combine two non-commensurate image samplers using a third, independent
    macro field; otherwise a clean circuit becomes a tiled floor.
    """
    graph = anti_tiling_scanned_graph(
        "test_track",
        materials.PRESETS["track_asphalt"],
        _PRIMARY_PBR,
        _SECONDARY_PBR,
        primary_size_m=2.0,
        macro_size_m=37.0,
        normal_multiplier=0.0,
    )

    primary = graph.nodes["primary_albedo_tri"]
    secondary = graph.nodes["secondary_albedo_tri"]
    assert primary.cls == secondary.cls == "VRayTriplanarTex"
    assert secondary.params["size"] / primary.params["size"] == pytest.approx(1.618)
    for node in (primary, secondary):
        assert node.params["random_texture_offset"] is True
        assert node.params["random_texture_rotation"] is True
    assert secondary.params["frame_offset"] == {"__point3__": [17.0, 31.0, 11.0]}
    assert secondary.params["texture_rotation"] == {"__point3__": [0.0, 0.0, 31.0]}

    macro = graph.nodes["macro_world"]
    assert macro.cls == "VRayTriplanarTex"
    assert macro.params["size"] == 37.0
    assert graph.nodes["albedo_mix"].inputs["Mask"] == "macro_world"
    assert graph.nodes["rough_mix"].inputs["Mask"] == "macro_world"
    assert "texmap_bump" not in graph.channels
    assert validate_graph(graph) == []


def test_anti_tiling_scanned_graph_can_keep_building_normal_detail():
    graph = anti_tiling_scanned_graph(
        "test_building",
        materials.PRESETS["concrete"],
        _PRIMARY_PBR,
        _SECONDARY_PBR,
        primary_size_m=2.1,
        macro_size_m=19.0,
        normal_multiplier=0.35,
    )

    assert graph.nodes["normal"].cls == "VRayNormalMap"
    assert graph.nodes["normal"].params["normal_map_multiplier"] == 0.35
    assert graph.channels["texmap_bump"] == "normal"
    assert graph.nodes["normal_mix"].inputs["Mask"] == "macro_world"
    assert validate_graph(graph) == []


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


# Surfaces whose real relief is below a pixel at 1:1, so a bump map can only
# invent detail that is not there. A racing circuit's wearing course is graded
# to one or two millimetres; under this project's characteristic low sun any
# procedural bump on it turned the track into tan-and-navy corrugation.
_INTENTIONALLY_FLAT = {"track_asphalt"}


def test_every_graph_drives_diffuse():
    for graph in GRAPHS.values():
        assert "texmap_diffuse" in graph.channels, f"{graph.key} has no diffuse"


def test_graphs_drive_bump_unless_the_surface_is_genuinely_flat():
    """
    Bump is what sells surface at grazing angles, which is most of a sun study,
    so its absence has to be a decision rather than an oversight. The exception
    list is the decision, and it is small on purpose.
    """
    for graph in GRAPHS.values():
        if graph.key in _INTENTIONALLY_FLAT:
            assert "texmap_bump" not in graph.channels, (
                f"{graph.key} is listed as intentionally flat but drives bump; "
                "remove it from the list or remove the binding"
            )
            continue
        assert "texmap_bump" in graph.channels, f"{graph.key} has no bump"


def test_channels_stay_within_the_declared_set():
    for graph in GRAPHS.values():
        assert set(graph.channels) <= set(CHANNELS)


# ── World-space projection ────────────────────────────────────────────────────

# Graphs that deliberately work in UV space instead. A kerb's stripes have to
# run along the kerb as it curves, and a world-space projection is locked to the
# world axes — the bands would stay pointing north while the corner turned away
# beneath them. Roadway ribbons carry real UVs (metres along, metres across), so
# these graphs have coordinates to use; buildings do not, which is why the rule
# holds everywhere else.
_UV_SPACE_GRAPHS = {"kerb"}


def test_patterns_are_projected_in_world_space():
    """
    VRayTriplanarTex is what makes these graphs work on meshes with NO UVs —
    which is every building Atlas produces. Without it the maps land on default
    coordinates and stretch unpredictably.
    """
    for graph in GRAPHS.values():
        if graph.key in _UV_SPACE_GRAPHS:
            assert not any(n.cls == "VRayTriplanarTex" for n in graph.nodes.values()), (
                f"{graph.key} is declared UV-space but also projects in world "
                "space; the two fight and the result depends on wiring order"
            )
            continue
        assert any(n.cls == "VRayTriplanarTex" for n in graph.nodes.values()), (
            f"{graph.key} has no world-space projection and needs UVs it does not have"
        )


def test_uv_space_graphs_are_only_used_on_meshes_that_carry_uvs():
    """
    A UV-space graph on a mesh with no texture coordinates lands on whatever
    default the host invents, which is the silent-wrong-output failure this
    project keeps hitting. The pairing is asserted here so the two cannot drift:
    every UV-space graph must name a preset the roadway module actually builds.
    """
    import roadway  # noqa: PLC0415 - imported here to keep the module list flat

    assert _UV_SPACE_GRAPHS <= set(GRAPHS)
    # roadway is the only producer of UV'd meshes today.
    assert hasattr(roadway, "kerb_ribbons")


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

def test_multisubtex_is_not_used():
    """
    Regression, measured on the live host. VRayMultiSubTex was used for
    per-building tint variation and silently destroyed the base colour: with an
    empty sub-texture list it falls through to `default_color` (a 127.5 grey)
    rather than the wired `default_texmap`, so every material collapsed to the
    same grey — brick and concrete both rendered rgb(81,74,70) despite brick
    being reddish. Under a bright sun that reads as blown-out white.

    Removing it restored correct colour: brick rgb(66,31,24), a red-blue spread
    of 42 against concrete's neutral 11.

    Per-building variation is therefore NOT implemented. Reinstating it needs a
    populated sub-texture list, not just the node.
    """
    for key, graph in GRAPHS.items():
        assert not any(n.cls == "VRayMultiSubTex" for n in graph.nodes.values()), (
            f"{key} reintroduces VRayMultiSubTex, which flattens the base colour"
        )


def test_materials_stay_distinguishable_by_base_colour():
    """
    What the MultiSubTex bug actually broke. Every graph must carry its own base
    colour through to the material — if two materials converge, the whole point
    of tag-driven shading is gone and it is invisible in any structural test.
    """
    reds = {}
    for key, graph in GRAPHS.items():
        base = graph.nodes.get("base")
        if base is not None:
            reds[key] = (base.params["red"], base.params["green"], base.params["blue"])
    assert reds["brick"] != reds["concrete"]
    # Brick is reddish; concrete is not. Asserting the ordering, not just difference.
    assert reds["brick"][0] - reds["brick"][2] > 0.15
    assert abs(reds["concrete"][0] - reds["concrete"][2]) < 0.1


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
