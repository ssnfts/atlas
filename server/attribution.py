"""
Build manifest — what a scene was made from, and what that obliges you to do.

Every data source in this pipeline carries a licence condition, and two of them
are not satisfiable after the fact. Copernicus requires a notice reproduced
**verbatim**; OpenStreetMap's ODbL puts a share-alike obligation on
redistributed *geometry* that does not apply to a rendered image. Getting either
wrong is discovered by a lawyer, not by a renderer, so the manifest is generated
from the sources a build actually touched rather than pasted in by hand.

The other half of its job is provenance, which is a production concern rather
than a legal one. OSM building heights are contributed and wildly uneven: some
are surveyed, most are a storey count multiplied by a constant, and a large
fraction are a flat default because nothing was tagged at all. A render cannot
show the difference. An artist deciding which blocks to replace with real
geometry needs to know, so :func:`height_provenance` counts them.

Nothing here writes to the scene or the network. It reads what the other modules
already recorded — ``Building.height_source``, ``TerrainPatch.source`` — and
turns it into a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "Source",
    "Manifest",
    "COPERNICUS",
    "COPERNICUS_MODIFIED",
    "OPENSTREETMAP",
    "OPEN_METEO",
    "height_provenance",
    "manifest_for_build",
]


@dataclass(frozen=True)
class Source:
    """One data source and the condition attached to using it."""

    name: str
    licence: str
    notice: str
    verbatim: bool = False
    caveat: str | None = None

    def render(self) -> str:
        lines = [f"{self.name}", f"  Licence: {self.licence}"]
        if self.verbatim:
            lines.append("  Notice (must be reproduced verbatim):")
        else:
            lines.append("  Notice:")
        lines += [f"    {line}" for line in _wrap(self.notice)]
        if self.caveat:
            lines.append("  Note:")
            lines += [f"    {line}" for line in _wrap(self.caveat)]
        return "\n".join(lines)


# The Copernicus wording is fixed by the licence. Do not reflow, abbreviate or
# "tidy" it — the requirement is the exact string.
COPERNICUS = Source(
    name="Copernicus DEM GLO-30 (terrain)",
    licence="Free, worldwide, non-exclusive; verbatim notice required",
    notice=(
        "© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
        "provided under COPERNICUS by the European Union and ESA; all rights "
        "reserved"
    ),
    verbatim=True,
)

# The distinct wording required when the elevation data has been altered rather
# than used as delivered. Resampling, striding or reprojecting counts.
COPERNICUS_MODIFIED = Source(
    name="Copernicus DEM GLO-30 (terrain, modified)",
    licence="Free, worldwide, non-exclusive; verbatim notice required",
    notice=(
        "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © "
        "Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
        "the European Union and ESA; all rights reserved"
    ),
    verbatim=True,
)

OPENSTREETMAP = Source(
    name="OpenStreetMap (building footprints and heights)",
    licence="Open Database License (ODbL) 1.0",
    notice="© OpenStreetMap contributors",
    caveat=(
        "Rendered images are a Produced Work and carry no share-alike "
        "obligation. Redistributing the derived geometry itself — the meshes, "
        "the footprints, a scene file containing them — is a Derivative "
        "Database and must be released under ODbL. Deliver frames, not scenes, "
        "unless you intend to license the scene."
    ),
)

OPEN_METEO = Source(
    name="Open-Meteo (ERA5 reanalysis, CAMS aerosols)",
    licence="CC BY 4.0 (data)",
    notice="Weather data by Open-Meteo.com",
    caveat=(
        "The data is CC BY 4.0 and fine commercially. The free public endpoint "
        "is not: it is limited to non-commercial use at roughly 10k calls/day. "
        "For production, self-host (AGPLv3) or use the commercial tier."
    ),
)


@dataclass
class Manifest:
    """The sources one build used, plus what it inferred rather than sourced."""

    site: dict = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add(self, source: Source) -> None:
        """Add a source once. Repeat calls for the same source are ignored."""
        if not any(s.name == source.name for s in self.sources):
            self.sources.append(source)

    @property
    def requires_verbatim(self) -> list[Source]:
        return [s for s in self.sources if s.verbatim]

    def render(self) -> str:
        out = ["ATTRIBUTION", "=" * 60, ""]

        if self.site:
            out.append("Build")
            for key, value in self.site.items():
                out.append(f"  {key}: {value}")
            out.append("")

        if not self.sources:
            # An empty manifest is a bug, not a clean bill of health — a scene
            # with no sources means nothing was fetched.
            out += ["No data sources recorded for this build.", ""]

        out += ["Sources", "-" * 60, ""]
        for source in self.sources:
            out += [source.render(), ""]

        if self.provenance:
            out += ["Building height provenance", "-" * 60, ""]
            total = self.provenance.get("total", 0)
            for label, count in self.provenance.get("by_source", {}).items():
                share = f"{100.0 * count / total:.1f}%" if total else "—"
                out.append(f"  {count:>6}  {share:>6}  {label}")
            inferred = self.provenance.get("inferred", 0)
            if total:
                out += [
                    "",
                    f"  {inferred} of {total} building heights "
                    f"({100.0 * inferred / total:.1f}%) were inferred rather "
                    "than sourced from a height tag.",
                ]
            out.append("")

        if self.notes:
            out += ["Notes", "-" * 60, ""]
            for note in self.notes:
                out += [f"  - {line}" for line in _wrap(note)]
            out.append("")

        return "\n".join(out).rstrip() + "\n"

    def write(self, path) -> str:
        """Write the manifest and return what was written."""
        text = self.render()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return text


def _wrap(text: str, width: int = 72) -> list[str]:
    """
    Wrap to a fixed width without importing textwrap for one call.

    Verbatim notices are wrapped for readability but never altered: wrapping
    changes only whitespace, and the licence requires the wording, not the line
    breaks.
    """
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def height_provenance(buildings) -> dict:
    """
    Count how many building heights were sourced versus guessed.

    Heights arrive with a ``height_source`` string describing exactly how they
    were derived. Those strings carry specifics — the level count, the roof
    height — so they are collapsed to their kind before counting, or a hundred
    buildings produce a hundred unique categories.
    """
    counts: dict[str, int] = {}
    inferred = 0

    for building in buildings:
        source = getattr(building, "height_source", "") or "unknown"
        label = _height_kind(source)
        counts[label] = counts.get(label, 0) + 1
        if label != "tagged height":
            inferred += 1

    ordered = dict(sorted(counts.items(), key=lambda kv: -kv[1]))
    return {"total": len(buildings), "by_source": ordered, "inferred": inferred}


def _height_kind(source: str) -> str:
    if source.startswith("height tag"):
        return "tagged height"
    if source.startswith("building:levels"):
        return "inferred from building:levels"
    if source.startswith("default"):
        return "default (nothing tagged)"
    return source


def manifest_for_build(
    *,
    site: dict | None = None,
    buildings=None,
    terrain_used: bool = False,
    terrain_modified: bool = False,
    weather_used: bool = False,
) -> Manifest:
    """
    Assemble the manifest for one build from what it actually used.

    Flags rather than a fixed blob: a scene built without terrain must not claim
    Copernicus data, both because the notice would be false and because it
    trains the reader to ignore the file.

    ``terrain_modified`` selects the distinct wording Copernicus requires for
    altered data. Resampling counts as altering — this pipeline strides the grid
    and reprojects it into a local tangent plane, so any build that meshes
    terrain should pass True.
    """
    manifest = Manifest(site=dict(site or {}))

    if buildings:
        manifest.add(OPENSTREETMAP)
        manifest.provenance = height_provenance(buildings)

    if terrain_used:
        manifest.add(COPERNICUS_MODIFIED if terrain_modified else COPERNICUS)

    if weather_used:
        manifest.add(OPEN_METEO)

    return manifest
