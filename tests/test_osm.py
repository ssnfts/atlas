"""
OSM parsing tests — no network.

Everything here is a case that produces a *building* rather than an error when
it goes wrong: a tower parsed at 12 m instead of 12 storeys, a courtyard block
whose hole is missing, a bbox transposed into the ocean. The scene still
renders. That is the point of testing it this hard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from frame import SceneFrame  # noqa: E402
from osm import (  # noqa: E402
    DEFAULT_HEIGHT_M,
    METRES_PER_LEVEL,
    assemble_rings,
    build_query,
    parse_overpass,
    query_for_site,
    resolve_height,
)


# ── Height tags ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("12", 12.0),
        ("12.5", 12.5),
        ("12 m", 12.0),
        ("12m", 12.0),
        ("12 metres", 12.0),
        ("  30  ", 30.0),
    ],
)
def test_metric_heights(raw, expected):
    height, _, source = resolve_height({"height": raw})
    assert height == pytest.approx(expected)
    assert source == "height tag"


def test_european_decimal_comma():
    """OSM contains "12,5" from locales where the comma is the decimal mark."""
    height, _, _ = resolve_height({"height": "12,5"})
    assert height == pytest.approx(12.5)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("40 ft", 12.192),
        ("40ft", 12.192),
        ("40 feet", 12.192),
        ("40'", 12.192),
        ("12'6\"", 3.8100),
    ],
)
def test_imperial_heights_convert_to_metres(raw, expected):
    """
    US mappers tag feet. Reading 40 ft as 40 m makes a two-storey shopfront
    into a thirteen-storey block, and nothing downstream can tell.
    """
    height, _, _ = resolve_height({"height": raw})
    assert height == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("raw", ["", "   ", "tall", "about 12", "12 6", "N/A", "-"])
def test_unparseable_heights_fall_through_to_the_default(raw):
    """A height that cannot be read must not become a silently invented number."""
    height, _, source = resolve_height({"height": raw})
    assert height == DEFAULT_HEIGHT_M
    assert "default" in source


def test_zero_and_negative_heights_are_rejected():
    for raw in ("0", "-5"):
        height, _, source = resolve_height({"height": raw})
        assert height == DEFAULT_HEIGHT_M
        assert "default" in source


# ── Levels fallback ───────────────────────────────────────────────────────────

def test_levels_fallback():
    height, _, source = resolve_height({"building:levels": "4"})
    assert height == pytest.approx(4 * METRES_PER_LEVEL)
    assert "building:levels=4" in source


def test_height_tag_beats_levels():
    """A measured height outranks a storey count multiplied by a guess."""
    height, _, source = resolve_height({"height": "17", "building:levels": "4"})
    assert height == pytest.approx(17.0)
    assert source == "height tag"


def test_half_levels_are_real():
    height, _, _ = resolve_height({"building:levels": "2.5"})
    assert height == pytest.approx(2.5 * METRES_PER_LEVEL)


def test_semicolon_level_alternatives_take_the_first():
    height, _, _ = resolve_height({"building:levels": "3;4"})
    assert height == pytest.approx(3 * METRES_PER_LEVEL)


def test_roof_height_adds_to_the_level_count():
    """
    building:levels counts occupied storeys and excludes the roof structure, so
    a pitched roof has to be added on top or every gabled house is short.
    """
    height, _, source = resolve_height({"building:levels": "2", "roof:height": "3"})
    assert height == pytest.approx(2 * METRES_PER_LEVEL + 3.0)
    assert "roof:height" in source


def test_roof_height_alone_does_not_become_the_building_height():
    height, _, source = resolve_height({"roof:height": "3"})
    assert height == DEFAULT_HEIGHT_M
    assert "default" in source


def test_no_tags_at_all_uses_the_default_and_says_so():
    height, min_height, source = resolve_height({})
    assert (height, min_height) == (DEFAULT_HEIGHT_M, 0.0)
    assert source == "default (no height or level tags)"


# ── min_height ────────────────────────────────────────────────────────────────

def test_min_height_lifts_the_volume():
    height, min_height, _ = resolve_height({"height": "20", "min_height": "5"})
    assert (height, min_height) == (20.0, 5.0)


def test_min_level_is_converted_like_levels():
    _, min_height, _ = resolve_height({"height": "20", "building:min_level": "2"})
    assert min_height == pytest.approx(2 * METRES_PER_LEVEL)


def test_min_height_above_the_roof_is_discarded():
    """
    Seen in the wild where min_height was tagged in feet and height in metres.
    An inside-out volume renders as a hole in the building.
    """
    height, min_height, source = resolve_height({"height": "10", "min_height": "30"})
    assert (height, min_height) == (10.0, 0.0)
    assert "min_height ignored" in source


# ── Query construction ────────────────────────────────────────────────────────

def test_query_uses_overpass_bbox_order():
    """
    Overpass wants (south, west, north, east). GeoJSON and Leaflet want
    (west, south, east, north). Swapping them queries a different continent and
    returns zero buildings, which reads as "nothing mapped here".
    """
    query = build_query((25.20, 55.26, 25.21, 55.28))
    assert "25.2000000,55.2600000,25.2100000,55.2800000" in query


def test_query_asks_for_ways_and_relations_and_inline_geometry():
    query = build_query((0.0, 0.0, 0.1, 0.1))
    assert 'way["building"]' in query
    assert 'relation["building"]' in query
    assert "out geom;" in query


def test_query_excludes_building_parts_unless_asked():
    plain = build_query((0.0, 0.0, 0.1, 0.1))
    parts = build_query((0.0, 0.0, 0.1, 0.1), include_parts=True)
    assert "building:part" not in plain
    assert 'way["building:part"]' in parts


def test_transposed_bbox_is_rejected_when_the_longitude_exceeds_90():
    """A GeoJSON-ordered bbox for Tokyo has 'south' = 139.7, not a latitude."""
    with pytest.raises(ValueError, match="latitude"):
        build_query((139.69, 35.68, 139.71, 35.69))


def test_transposed_bbox_is_undetectable_inside_the_90_degree_band():
    """
    Documents the limit of validation rather than implying one that is absent.

    A transposed Dubai bbox reads as south=55.27, a real latitude in northern
    Russia. Every range check passes, Overpass answers, and the answer is an
    empty building list — indistinguishable from an unmapped site. This is why
    `query_for_site` exists and why nothing in this codebase writes the tuple
    by hand.
    """
    build_query((55.26, 25.20, 55.28, 25.21))  # does not raise, and cannot


def test_inverted_latitudes_are_rejected():
    with pytest.raises(ValueError):
        build_query((25.21, 55.26, 25.20, 55.28))


def test_query_for_site_gets_the_ordering_right_without_the_caller_helping():
    """The whole point: the frame builds the tuple, so nobody can transpose it."""
    frame = SceneFrame(25.2048, 55.2708)
    query = query_for_site(frame, 500.0)
    south, west, north, east = frame.bbox_for_radius(500.0)
    assert south < frame.origin_lat < north
    assert west < frame.origin_lon < east
    assert f"{south:.7f},{west:.7f},{north:.7f},{east:.7f}" in query


# ── Element parsing ───────────────────────────────────────────────────────────

def _way(way_id: int, coords: list[tuple[float, float]], tags: dict) -> dict:
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in coords],
    }


SQUARE = [(0.0, 0.0), (0.0, 0.001), (0.001, 0.001), (0.001, 0.0), (0.0, 0.0)]


def test_way_becomes_a_building():
    buildings = parse_overpass({"elements": [_way(1, SQUARE, {"building": "yes"})]})
    assert len(buildings) == 1
    assert buildings[0].osm_id == 1
    assert buildings[0].osm_type == "way"


def test_closing_vertex_is_dropped():
    """
    A closed OSM way repeats its first node last. Keeping it makes a zero-length
    edge, which has no direction, which makes ear clipping misread the corner.
    """
    building = parse_overpass({"elements": [_way(1, SQUARE, {"building": "yes"})]})[0]
    assert len(building.outer) == 4
    assert building.outer[0] != building.outer[-1]


def test_building_no_is_excluded():
    """`building=no` asserts a footprint is not a building; truthiness extrudes it."""
    payload = {"elements": [_way(1, SQUARE, {"building": "no"})]}
    assert parse_overpass(payload) == []


def test_non_building_elements_are_ignored():
    payload = {"elements": [_way(1, SQUARE, {"highway": "residential"})]}
    assert parse_overpass(payload) == []


def test_building_parts_are_opt_in():
    payload = {"elements": [_way(1, SQUARE, {"building:part": "yes"})]}
    assert parse_overpass(payload) == []
    assert len(parse_overpass(payload, include_parts=True)) == 1


def test_degenerate_way_is_skipped_not_fatal():
    """One bad footprint costs that building, not the whole district."""
    payload = {
        "elements": [
            _way(1, [(0.0, 0.0), (0.0, 0.001)], {"building": "yes"}),  # 2 points
            _way(2, SQUARE, {"building": "yes"}),
        ]
    }
    buildings = parse_overpass(payload)
    assert [b.osm_id for b in buildings] == [2]


def test_null_geometry_nodes_are_dropped():
    """Overpass emits nulls for nodes of a way that leaves the query bbox."""
    element = _way(1, SQUARE, {"building": "yes"})
    element["geometry"].insert(2, {"lat": None, "lon": None})
    assert len(parse_overpass({"elements": [element]})[0].outer) == 4


def test_tags_and_height_survive_parsing():
    payload = {
        "elements": [_way(7, SQUARE, {"building": "yes", "height": "31", "name": "Tower"})]
    }
    building = parse_overpass(payload)[0]
    assert building.height_m == 31.0
    assert building.name == "Tower"
    assert building.height_source == "height tag"


def test_material_tag_is_exposed_for_later_assignment():
    payload = {
        "elements": [_way(1, SQUARE, {"building": "yes", "building:material": "brick"})]
    }
    assert parse_overpass(payload)[0].material == "brick"


# ── Multipolygon relations ────────────────────────────────────────────────────

OUTER = [(0.0, 0.0), (0.0, 0.002), (0.002, 0.002), (0.002, 0.0)]
INNER = [(0.0005, 0.0005), (0.0005, 0.0015), (0.0015, 0.0015), (0.0015, 0.0005)]


def _member(coords: list[tuple[float, float]], role: str) -> dict:
    return {
        "type": "way",
        "ref": 1,
        "role": role,
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in coords],
    }


def test_relation_with_a_courtyard_keeps_the_hole():
    payload = {
        "elements": [
            {
                "type": "relation",
                "id": 99,
                "tags": {"building": "yes", "type": "multipolygon"},
                "members": [
                    _member(OUTER + [OUTER[0]], "outer"),
                    _member(INNER + [INNER[0]], "inner"),
                ],
            }
        ]
    }
    building = parse_overpass(payload)[0]
    assert building.osm_type == "relation"
    assert len(building.outer) == 4
    assert len(building.inners) == 1
    assert len(building.inners[0]) == 4


def test_outer_ring_split_across_several_ways_is_stitched():
    """
    The multipolygon trap: a role belongs to a *way*, not to a ring. A block's
    boundary is routinely several ways split where a street name changes.
    Treating each as a finished ring gives open polylines, not a footprint.
    """
    payload = {
        "elements": [
            {
                "type": "relation",
                "id": 5,
                "tags": {"building": "yes", "type": "multipolygon"},
                "members": [
                    _member([OUTER[0], OUTER[1]], "outer"),
                    _member([OUTER[1], OUTER[2]], "outer"),
                    _member([OUTER[2], OUTER[3]], "outer"),
                    _member([OUTER[3], OUTER[0]], "outer"),
                ],
            }
        ]
    }
    building = parse_overpass(payload)[0]
    assert len(building.outer) == 4


def test_relation_with_no_closable_ring_is_skipped():
    payload = {
        "elements": [
            {
                "type": "relation",
                "id": 6,
                "tags": {"building": "yes"},
                "members": [_member([OUTER[0], OUTER[1]], "outer")],
            }
        ]
    }
    assert parse_overpass(payload) == []


def test_largest_outer_ring_wins_when_several_are_disjoint():
    big = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)]
    small = [(1.0, 1.0), (1.0, 1.001), (1.001, 1.001), (1.001, 1.0)]
    payload = {
        "elements": [
            {
                "type": "relation",
                "id": 8,
                "tags": {"building": "yes"},
                "members": [
                    _member(small + [small[0]], "outer"),
                    _member(big + [big[0]], "outer"),
                ],
            }
        ]
    }
    building = parse_overpass(payload)[0]
    assert building.outer[0][0] == pytest.approx(0.0)


# ── Ring assembly ─────────────────────────────────────────────────────────────

def test_assemble_handles_reversed_segments():
    """Member ways have arbitrary direction; half of them need flipping."""
    segments = [
        [(0.0, 0.0), (0.0, 1.0)],
        [(1.0, 1.0), (0.0, 1.0)],   # reversed
        [(1.0, 1.0), (1.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0)],   # reversed
    ]
    rings = assemble_rings(segments)
    assert len(rings) == 1
    assert len(rings[0]) == 4


def test_assemble_passes_through_an_already_closed_way():
    closed = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    rings = assemble_rings([closed])
    assert len(rings) == 1
    assert len(rings[0]) == 4


def test_assemble_separates_two_independent_rings():
    a = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 0.0)]
    b = [(5.0, 5.0), (5.0, 6.0), (6.0, 6.0), (5.0, 5.0)]
    assert len(assemble_rings([a, b])) == 2


def test_assemble_drops_an_unclosed_chain():
    """An unclosed boundary is a mapping error and cannot be extruded."""
    assert assemble_rings([[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]]) == []


# ── Building helpers ──────────────────────────────────────────────────────────

# ── Transient-failure retries ─────────────────────────────────────────────────
#
# Overpass overload is routine, not exceptional: it returned 504 twice in one
# afternoon and the next attempt succeeded both times. What must never happen is
# a busy service degrading into an empty building list — that renders as a
# correct picture of an empty field.

import json as _json  # noqa: E402
import urllib.error  # noqa: E402

import osm as osm_module  # noqa: E402

BBOX = (25.07, 55.13, 25.09, 55.15)


class _Response:
    def __init__(self, payload):
        self._body = _json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _http_error(code):
    return urllib.error.HTTPError("http://x", code, "busy", {}, None)


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(osm_module, "_sleep", lambda _s: None)


def _payload_with_one_building():
    return {"elements": [_way(1, SQUARE, {"building": "yes"})]}


def test_a_transient_504_is_retried(monkeypatch, no_sleep):
    attempts = []

    def fake_open(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(504)
        return _Response(_payload_with_one_building())

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", fake_open)
    assert len(osm_module.fetch_buildings(BBOX)) == 1
    assert len(attempts) == 2


@pytest.mark.parametrize("code", [429, 502, 503, 504])
def test_all_busy_codes_are_retried(monkeypatch, no_sleep, code):
    attempts = []

    def fake_open(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(code)
        return _Response(_payload_with_one_building())

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", fake_open)
    assert len(osm_module.fetch_buildings(BBOX)) == 1


def test_a_bad_query_is_not_retried(monkeypatch, no_sleep):
    """400 means the query is wrong; retrying makes a clear error a slow one."""
    attempts = []

    def fake_open(request, timeout=None):
        attempts.append(1)
        raise _http_error(400)

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", fake_open)
    with pytest.raises(osm_module.OverpassError, match="HTTP 400"):
        osm_module.fetch_buildings(BBOX)
    assert len(attempts) == 1


def test_retries_are_bounded(monkeypatch, no_sleep):
    attempts = []

    def fake_open(request, timeout=None):
        attempts.append(1)
        raise _http_error(504)

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", fake_open)
    with pytest.raises(osm_module.OverpassError):
        osm_module.fetch_buildings(BBOX, retries=2)
    assert len(attempts) == 3


def test_exhausted_retries_raise_rather_than_return_empty(monkeypatch, no_sleep):
    """
    The whole point. An empty list here is indistinguishable downstream from a
    genuinely unmapped site, and renders as an empty field with no error.
    """
    monkeypatch.setattr(
        osm_module.urllib.request,
        "urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(_http_error(504)),
    )
    with pytest.raises(osm_module.OverpassError, match="after 3 attempts"):
        osm_module.fetch_buildings(BBOX)


def test_network_errors_are_retried_too(monkeypatch, no_sleep):
    attempts = []

    def fake_open(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError("connection reset")
        return _Response(_payload_with_one_building())

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", fake_open)
    assert len(osm_module.fetch_buildings(BBOX)) == 1


def test_backoff_grows_between_attempts(monkeypatch):
    delays = []
    monkeypatch.setattr(osm_module, "_sleep", lambda s: delays.append(s))
    monkeypatch.setattr(
        osm_module.urllib.request,
        "urlopen",
        lambda request, timeout=None: (_ for _ in ()).throw(_http_error(504)),
    )
    with pytest.raises(osm_module.OverpassError):
        osm_module.fetch_buildings(BBOX, retries=3, backoff_s=2.0)
    assert delays == [2.0, 4.0, 6.0]


def test_no_retry_when_disabled(monkeypatch, no_sleep):
    attempts = []

    def fake_open(request, timeout=None):
        attempts.append(1)
        raise _http_error(504)

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", fake_open)
    with pytest.raises(osm_module.OverpassError):
        osm_module.fetch_buildings(BBOX, retries=0)
    assert len(attempts) == 1


