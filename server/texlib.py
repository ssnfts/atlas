"""
Scanned PBR texture sets, fetched on demand.

Every surface in this project has been procedural until now — V-Ray noise and
checker through a triplanar projection. That was the right first move: it is
resolution-independent, has no dependencies, and it let the material *structure*
be got right before any assets existed. What it cannot do is look photographed.
Real asphalt is not noise; it has aggregate of a particular size, patches where
it was laid on different days, and a roughness that varies with the wear.

So this fetches scanned sets — albedo, roughness, normal, displacement — and the
existing graphs use them where one exists and stay procedural where none does.

**Poly Haven, CC0 only.** The same source and the same reason as the sky in
:mod:`skydome`: everything else in this project was chosen to be safe to use
commercially, and a texture under a non-commercial or attribution-required
licence would quietly undo that for the whole render. CC0 is a public-domain
dedication, so there is nothing to comply with and nothing to track — the
attribution recorded in :data:`ATTRIBUTION` is a courtesy and a provenance
record, not an obligation.

Files land in ``assets/textures/<asset>/`` and are gitignored. They are inputs,
fetched by name, not source — the same treatment as the HDRI and the DEM tiles.

**These need no UVs.** Fed through ``VRayTriplanarTex`` the maps project in
world space, which is what lets 379 untextured building meshes take a scanned
concrete without anyone unwrapping anything. The track, which does carry UVs
from :mod:`roadway`, can use them directly instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

__all__ = [
    "POLYHAVEN_API",
    "ATTRIBUTION",
    "TEXTURE_ROOT",
    "MAP_SLOTS",
    "fetch_texture",
    "local_set",
    "available_sets",
]

POLYHAVEN_API = "https://api.polyhaven.com"

ATTRIBUTION = (
    "Surface textures from Poly Haven (polyhaven.com), released under CC0 1.0 "
    "(public domain dedication). No attribution is legally required; it is "
    "recorded so a build can say where every input came from."
)

TEXTURE_ROOT = Path(__file__).resolve().parent.parent / "assets" / "textures"

# Poly Haven's map name -> what it drives on a VRayMtl.
#
# ``nor_gl`` rather than ``nor_dx``: the two differ in the sign of the green
# channel, and picking the wrong one inverts every bump so that dents read as
# bumps. V-Ray follows the OpenGL convention. This is a silent failure — the
# render is lit plausibly and wrong — which is why it is written down here
# rather than left to whoever wires the graph.
MAP_SLOTS = {
    "Diffuse": "diffuse",
    "Rough": "reflectionGlossiness",   # inverted at the graph, see below
    "nor_gl": "bump",
    "Displacement": "displacement",
    "AO": "ao",
}

_UA = {"User-Agent": "atlas-scene/0.1 (+https://github.com/ssnfts/atlas)"}


def _api(path: str, timeout: float = 40.0):
    request = urllib.request.Request(f"{POLYHAVEN_API}/{path}", headers=_UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def local_set(asset: str) -> dict:
    """Map name -> path, for whatever of ``asset`` is already on disk."""
    folder = TEXTURE_ROOT / asset
    if not folder.is_dir():
        return {}
    found = {}
    for path in folder.iterdir():
        for name in MAP_SLOTS:
            if path.stem.endswith(name):
                found[name] = str(path)
    return found


def available_sets() -> dict:
    """Every texture set already cached, as ``asset -> {map: path}``."""
    if not TEXTURE_ROOT.is_dir():
        return {}
    return {p.name: local_set(p.name) for p in TEXTURE_ROOT.iterdir() if p.is_dir()}


def fetch_texture(
    asset: str,
    *,
    resolution: str = "4k",
    maps: tuple[str, ...] = ("Diffuse", "Rough", "nor_gl"),
    fmt: str = "jpg",
    timeout: float = 300.0,
) -> dict:
    """
    Download a texture set, skipping anything already cached.

    ``fmt`` defaults to jpg rather than exr for a reason worth stating: the exr
    of a single 4K map here runs 25-37 MB against 2-8 MB for the jpg, and for
    albedo and roughness the extra bit depth buys nothing a render will show.
    Displacement is the exception — it is a height field, and 8-bit banding in a
    height field produces visible terraces — so ask for exr explicitly if
    displacement is going to drive real geometry.

    Returns ``{map_name: path}``. Raises if the asset has none of the requested
    maps, rather than returning an empty dict that a caller would read as
    "fetched, nothing to do".
    """
    folder = TEXTURE_ROOT / asset
    folder.mkdir(parents=True, exist_ok=True)

    cached = local_set(asset)
    wanted = [m for m in maps if m not in cached]
    if not wanted:
        return cached

    files = _api(f"files/{asset}", timeout=timeout)
    got = dict(cached)

    for name in wanted:
        entry = files.get(name)
        if not entry:
            continue
        by_res = entry.get(resolution) or entry.get("4k") or entry.get("2k")
        if not by_res:
            continue
        chosen = by_res.get(fmt) or next(iter(by_res.values()))
        url = chosen["url"]
        target = folder / f"{asset}_{resolution}_{name}{Path(url).suffix}"

        request = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            target.write_bytes(response.read())
        got[name] = str(target)

    if not got:
        raise ValueError(
            f"{asset} yielded none of {maps} at {resolution}/{fmt}. Check the "
            "asset name against https://api.polyhaven.com/assets?t=textures."
        )
    return got
