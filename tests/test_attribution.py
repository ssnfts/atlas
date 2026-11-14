"""
Attribution manifest tests.

Two of the obligations here cannot be fixed after the fact. Copernicus requires
its notice reproduced verbatim, and OpenStreetMap's ODbL puts a share-alike
condition on redistributed geometry that does not apply to a rendered frame.
A manifest that quietly drops one, or claims a source the build never used, is
found by a lawyer rather than by a renderer — so the wording is asserted
character for character.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from attribution import (  # noqa: E402
    COPERNICUS,
    COPERNICUS_MODIFIED,
    OPENSTREETMAP,
    OPEN_METEO,
    Manifest,
    height_provenance,
    manifest_for_build,
)
from osm import Building  # noqa: E402


def _building(height=None, levels=None, roof=None):
    tags = {}
    if height is not None:
        tags["height"] = str(height)
    if levels is not None:
        tags["building:levels"] = str(levels)
    if roof is not None:
        tags["roof:height"] = str(roof)
    ring = [(25.0, 55.0), (25.0, 55.001), (25.001, 55.001), (25.001, 55.0)]
    return Building(1, "way", ring, tags=tags)


# ── Verbatim notices ──────────────────────────────────────────────────────────

def test_copernicus_notice_is_exact():
    """
    Fixed by the licence. Not a style choice — no reflowing, no abbreviating,
    no swapping the ampersand or the dashes.
    """
    assert COPERNICUS.notice == (
        "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
        "provided under COPERNICUS by the European Union and ESA; all rights "
        "reserved"
    )


def test_copernicus_modified_notice_is_exact():
    """The distinct wording required once the elevation data has been altered."""
    assert COPERNICUS_MODIFIED.notice == (
        "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © "
        "Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
        "the European Union and ESA; all rights reserved"
    )


def test_both_copernicus_notices_are_marked_verbatim():
    assert COPERNICUS.verbatim and COPERNICUS_MODIFIED.verbatim


def test_osm_and_openmeteo_are_not_marked_verbatim():
    """Only claim a wording is fixed where it actually is."""
    assert not OPENSTREETMAP.verbatim
    assert not OPEN_METEO.verbatim


def test_verbatim_notice_survives_rendering_unaltered():
    """
    Wrapping may change whitespace and nothing else. Rejoining the rendered
    lines must reproduce the notice exactly.
    """
    rendered = COPERNICUS.render()
    body = rendered.split("verbatim):", 1)[1]
    assert " ".join(body.split()) == COPERNICUS.notice


def test_verbatim_sources_are_reported():
    m = manifest_for_build(terrain_used=True)
    assert [s.name for s in m.requires_verbatim] == [COPERNICUS.name]


# ── Only claim what was used ──────────────────────────────────────────────────

def test_a_build_without_terrain_does_not_claim_copernicus():
    """
    A false notice is its own problem, and it teaches the reader to skip the
    file — which defeats the one notice that genuinely is required.
    """
    text = manifest_for_build(buildings=[_building(height=20)]).render()
    assert "Copernicus" not in text
    assert "OpenStreetMap" in text


def test_a_build_without_buildings_does_not_claim_osm():
    text = manifest_for_build(terrain_used=True).render()
    assert "OpenStreetMap" not in text
    assert "DLR e.V." in text


def test_a_build_without_weather_does_not_claim_open_meteo():
    assert "Open-Meteo" not in manifest_for_build(terrain_used=True).render()


def test_modified_terrain_selects_the_other_wording():
    """
    This pipeline strides and reprojects the grid, so real builds are modified.
    """
    plain = manifest_for_build(terrain_used=True).render()
    modified = manifest_for_build(terrain_used=True, terrain_modified=True).render()
    assert "produced using Copernicus WorldDEM-30" in modified
    assert "produced using Copernicus WorldDEM-30" not in plain


def test_full_build_claims_all_three():
    text = manifest_for_build(
        buildings=[_building(height=20)], terrain_used=True, weather_used=True
    ).render()
    for expected in ("DLR e.V.", "OpenStreetMap", "Open-Meteo"):
        assert expected in text


def test_sources_are_not_duplicated():
    m = Manifest()
    m.add(OPENSTREETMAP)
    m.add(OPENSTREETMAP)
    assert len(m.sources) == 1


def test_empty_manifest_says_so_rather_than_looking_clean():
    """No sources means nothing was fetched — a bug, not a clean bill of health."""
    assert "No data sources recorded" in Manifest().render()


# ── ODbL caveat ───────────────────────────────────────────────────────────────

def test_odbl_distinguishes_rendered_frames_from_redistributed_geometry():
    """
    The whole practical point of the OSM entry: images are a Produced Work and
    carry no share-alike obligation; shipping the meshes is a Derivative
    Database and does.
    """
    caveat = OPENSTREETMAP.caveat
    assert "Produced Work" in caveat
    assert "Derivative Database" in caveat
    assert "ODbL" in OPENSTREETMAP.licence


def test_open_meteo_flags_the_endpoint_versus_the_data():
    """The data is commercially fine; the free endpoint is not. Both are stated."""
    assert "CC BY 4.0" in OPEN_METEO.licence
    assert "non-commercial" in OPEN_METEO.caveat


# ── Height provenance ─────────────────────────────────────────────────────────

def test_tagged_heights_are_not_counted_as_inferred():
    result = height_provenance([_building(height=20), _building(height=31)])
    assert result["inferred"] == 0
    assert result["by_source"]["tagged height"] == 2


def test_level_derived_heights_are_inferred():
    result = height_provenance([_building(levels=4)])
    assert result["inferred"] == 1
    assert "inferred from building:levels" in result["by_source"]


def test_defaulted_heights_are_inferred():
    result = height_provenance([_building()])
    assert result["inferred"] == 1
    assert "default (nothing tagged)" in result["by_source"]


def test_provenance_collapses_specifics_into_kinds():
    """
    height_source carries the level count and roof height. Left uncollapsed,
    a hundred buildings produce a hundred categories and the summary is useless.
    """
    result = height_provenance(
        [_building(levels=2), _building(levels=17), _building(levels=4, roof=3)]
    )
    assert list(result["by_source"]) == ["inferred from building:levels"]
    assert result["by_source"]["inferred from building:levels"] == 3


def test_mixed_provenance_counts_correctly():
    buildings = [_building(height=20), _building(levels=3), _building(), _building()]
    result = height_provenance(buildings)
    assert result["total"] == 4
    assert result["inferred"] == 3
    assert result["by_source"]["default (nothing tagged)"] == 2


def test_provenance_is_ordered_most_common_first():
    buildings = [_building() for _ in range(3)] + [_building(height=10)]
    order = list(height_provenance(buildings)["by_source"])
    assert order[0] == "default (nothing tagged)"


def test_provenance_appears_in_the_rendered_manifest():
    text = manifest_for_build(
        buildings=[_building(height=20), _building(), _building()]
    ).render()
    assert "height provenance" in text
    assert "66.7%) were inferred" in text


def test_no_buildings_means_no_provenance_section():
    assert "provenance" not in manifest_for_build(terrain_used=True).render()


def test_provenance_of_an_empty_list_does_not_divide_by_zero():
    result = height_provenance([])
    assert result == {"total": 0, "by_source": {}, "inferred": 0}


# ── File output ───────────────────────────────────────────────────────────────

def test_write_produces_the_rendered_text(tmp_path):
    m = manifest_for_build(
        site={"location": "25.0805, 55.1403"},
        buildings=[_building(height=20)],
        terrain_used=True,
        terrain_modified=True,
        weather_used=True,
    )
    path = tmp_path / "ATTRIBUTION.txt"
    written = m.write(path)
    assert path.read_text(encoding="utf-8") == written
    assert "DLR e.V." in written


def test_written_file_is_utf8_and_keeps_the_copyright_glyphs(tmp_path):
    """
    The notice contains © and must survive the round trip. Windows defaults to
    cp1252 for text writes, which mangles it.
    """
    path = tmp_path / "ATTRIBUTION.txt"
    manifest_for_build(terrain_used=True).write(path)
    assert "©" in path.read_text(encoding="utf-8")


def test_site_details_are_recorded(tmp_path):
    text = manifest_for_build(
        site={"location": "41.890209, 12.492231", "local_time": "2026-07-22T19:30"},
        terrain_used=True,
    ).render()
    assert "41.890209" in text
    assert "2026-07-22T19:30" in text


def test_manifest_ends_with_a_single_newline():
    text = manifest_for_build(terrain_used=True).render()
    assert text.endswith("\n") and not text.endswith("\n\n")
