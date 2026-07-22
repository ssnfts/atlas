"""
Photogrammetry ingest — check the capture before spending an hour of GPU on it.

Meshroom will happily run for an hour on a set that could never have
reconstructed, and report success on a result that is a flat smear. The
expensive failures are all decidable up front from the images themselves, so
this module front-loads them:

- **too few images, or too little overlap implied by the count** — every
  surface point needs to appear in at least three;
- **motion blur** — feature matching degrades sharply, and blur is invisible
  in a thumbnail;
- **inconsistent focal length** — a zoom that moved mid-capture splits the
  intrinsics and weakens the solve;
- **missing EXIF focal length** — Meshroom then guesses the camera intrinsics;
- **missing GPS** — the reconstruction still works, but it cannot be
  georeferenced without manual control points, and that is worth knowing
  before the shoot is over rather than after.

Deliberately **not** a pluggable backend interface. The plan sketched one over
meshroom/colmap/realitycapture, but an abstraction with a single implementation
is a liability: it fixes the shape of the seam before there is any evidence
about where the seam should be. Meshroom is the backend. If a second one ever
lands, extract the interface then, from two real cases.

Nothing here reconstructs anything — it prepares, checks, and builds the
command. The reconstruction is :func:`meshroom_command` handed to a subprocess.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FrameInfo",
    "CaptureReport",
    "Problem",
    "ReconError",
    "read_frame_info",
    "inspect_capture",
    "validate_capture",
    "sharpness",
    "extract_frames",
    "meshroom_command",
    "MIN_IMAGES",
    "RECOMMENDED_IMAGES",
    "MAX_IMAGES_16GB",
]

# Below this a solve is very unlikely to converge at all.
MIN_IMAGES = 20
# Below this it may converge but will be sparse and holey.
RECOMMENDED_IMAGES = 60
# The DepthMap stage is the VRAM constraint. On a 16 GB card this is roughly
# where tuning downscale factors replaces capturing better, so it is a warning
# rather than a hard limit.
MAX_IMAGES_16GB = 500

# Variance-of-Laplacian threshold. Scale-dependent and lens-dependent, so it is
# a heuristic for ranking frames against each other rather than an absolute
# verdict — hence "suspect", not "blurred", in the report.
BLUR_THRESHOLD = 100.0


class ReconError(RuntimeError):
    """Capture ingest failed, or a required external tool is missing."""


@dataclass
class FrameInfo:
    """What one source image tells us about itself."""

    path: Path
    width: int = 0
    height: int = 0
    focal_length_mm: float | None = None
    focal_35mm: float | None = None
    camera: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    altitude_m: float | None = None
    captured_at: str | None = None
    sharpness: float | None = None

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1e6

    @property
    def has_gps(self) -> bool:
        return self.latitude is not None and self.longitude is not None


@dataclass
class Problem:
    """One reason a capture may not reconstruct, and what to do about it."""

    severity: str  # "blocker" | "warning"
    code: str
    detail: str
    remedy: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.code}: {self.detail} — {self.remedy}"


@dataclass
class CaptureReport:
    """Everything decidable about a capture without reconstructing it."""

    frames: list[FrameInfo] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def with_gps(self) -> int:
        return sum(1 for f in self.frames if f.has_gps)

    @property
    def focal_lengths(self) -> set:
        return {f.focal_length_mm for f in self.frames if f.focal_length_mm}

    @property
    def cameras(self) -> set:
        return {f.camera for f in self.frames if f.camera}

    @property
    def blockers(self) -> list[Problem]:
        return [p for p in self.problems if p.severity == "blocker"]

    @property
    def can_reconstruct(self) -> bool:
        return self.count > 0 and not self.blockers

    @property
    def can_georeference(self) -> bool:
        """
        Whether EXIF alone can place this in the world.

        A majority with GPS is enough for a coarse similarity transform; a
        handful of tagged frames among hundreds is not, and quietly using them
        would place the site confidently in the wrong spot.
        """
        return self.count > 0 and self.with_gps >= max(3, self.count // 2)

    def gps_centroid(self) -> tuple[float, float] | None:
        """Mean position of the geotagged frames — the site's coarse location."""
        located = [f for f in self.frames if f.has_gps]
        if not located:
            return None
        return (
            sum(f.latitude for f in located) / len(located),
            sum(f.longitude for f in located) / len(located),
        )

    def as_dict(self) -> dict:
        centroid = self.gps_centroid()
        return {
            "frames": self.count,
            "with_gps": self.with_gps,
            "cameras": sorted(self.cameras),
            "focal_lengths_mm": sorted(self.focal_lengths),
            "megapixels": round(
                sum(f.megapixels for f in self.frames) / self.count, 2
            ) if self.count else 0.0,
            "gps_centroid": {"latitude": centroid[0], "longitude": centroid[1]}
            if centroid
            else None,
            "can_reconstruct": self.can_reconstruct,
            "can_georeference": self.can_georeference,
            "problems": [
                {"severity": p.severity, "code": p.code, "detail": p.detail,
                 "remedy": p.remedy}
                for p in self.problems
            ],
        }


# ── EXIF ──────────────────────────────────────────────────────────────────────

def read_frame_info(path, *, measure_sharpness: bool = True) -> FrameInfo:
    """
    Read one image's dimensions, camera intrinsics and GPS.

    EXIF is read from the file as delivered. A frame that has been cropped or
    re-exported by an editor usually has its focal length stripped or made
    inconsistent with the new pixel dimensions, and its GPS removed — which is
    why the capture guidance is to hand over originals.
    """
    from PIL import ExifTags, Image

    path = Path(path)
    info = FrameInfo(path=path)

    try:
        with Image.open(path) as image:
            info.width, info.height = image.size
            exif = image.getexif()

            tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            info.camera = _clean(tags.get("Model"))
            info.captured_at = _clean(tags.get("DateTimeOriginal") or tags.get("DateTime"))

            ifd = exif.get_ifd(ExifTags.IFD.Exif) if hasattr(ExifTags, "IFD") else {}
            exif_tags = {ExifTags.TAGS.get(k, k): v for k, v in (ifd or {}).items()}
            info.focal_length_mm = _as_float(
                exif_tags.get("FocalLength") or tags.get("FocalLength")
            )
            info.focal_35mm = _as_float(
                exif_tags.get("FocalLengthIn35mmFilm")
                or tags.get("FocalLengthIn35mmFilm")
            )
            if info.captured_at is None:
                info.captured_at = _clean(exif_tags.get("DateTimeOriginal"))

            _read_gps(exif, info)
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ReconError(f"could not read {path.name}: {exc}") from exc

    if measure_sharpness:
        try:
            info.sharpness = sharpness(path)
        except Exception:
            info.sharpness = None

    return info


def _read_gps(exif, info: FrameInfo) -> None:
    """Pull latitude/longitude out of the GPS IFD, applying the hemisphere refs."""
    from PIL import ExifTags

    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
    except Exception:
        return
    if not gps:
        return

    named = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps.items()}

    lat = _dms_to_degrees(named.get("GPSLatitude"))
    lon = _dms_to_degrees(named.get("GPSLongitude"))
    if lat is None or lon is None:
        return

    # The hemisphere is a separate tag. Ignoring it puts the southern
    # hemisphere in the northern one and the Americas in Asia — a large,
    # confident, entirely wrong position.
    if str(named.get("GPSLatitudeRef", "N")).upper().startswith("S"):
        lat = -lat
    if str(named.get("GPSLongitudeRef", "E")).upper().startswith("W"):
        lon = -lon

    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return

    info.latitude, info.longitude = lat, lon

    altitude = _as_float(named.get("GPSAltitude"))
    if altitude is not None:
        # Ref 1 means below sea level.
        if str(named.get("GPSAltitudeRef", 0)) in ("1", "b'\\x01'"):
            altitude = -altitude
        info.altitude_m = altitude


def _dms_to_degrees(value) -> float | None:
    """EXIF degrees/minutes/seconds triple -> decimal degrees."""
    if not value or len(value) != 3:
        return None
    try:
        d, m, s = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return d + m / 60.0 + s / 3600.0


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("\x00")
    return text or None


# ── Sharpness ─────────────────────────────────────────────────────────────────

def sharpness(path) -> float:
    """
    Variance of the Laplacian — the standard blur proxy.

    A sharp image has strong high-frequency content and therefore a large
    second-derivative variance; blur suppresses it. The absolute value depends
    on resolution, subject and lens, so it is only meaningful for ranking
    frames from one capture against each other. That is why the report says
    "suspect" rather than "blurred".

    Downsampled first, because full-resolution Laplacians over 500 frames cost
    more than the information is worth.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        grey = image.convert("L")
        grey.thumbnail((1024, 1024))
        array = np.asarray(grey, dtype=np.float64)

    if array.size == 0 or min(array.shape) < 3:
        return 0.0

    # 4-neighbour Laplacian on the interior, avoiding an edge-padding decision
    # that would bias the variance.
    laplacian = (
        array[:-2, 1:-1]
        + array[2:, 1:-1]
        + array[1:-1, :-2]
        + array[1:-1, 2:]
        - 4.0 * array[1:-1, 1:-1]
    )
    return float(laplacian.var())


# ── Validation ────────────────────────────────────────────────────────────────

def validate_capture(report: CaptureReport) -> list[Problem]:
    """
    Everything decidable before reconstruction, as blockers and warnings.

    Blockers mean a run would waste its time; warnings mean it will probably
    work but the result will be worse than it needed to be. The distinction
    matters because the cost of being wrong is an hour of GPU either way.
    """
    problems: list[Problem] = []
    count = report.count

    if count == 0:
        problems.append(Problem(
            "blocker", "no_images", "the capture is empty",
            "point it at a folder of stills or extract frames from video first",
        ))
        return problems

    if count < MIN_IMAGES:
        problems.append(Problem(
            "blocker", "too_few_images",
            f"{count} images; a solve needs every surface point in at least three",
            f"capture at least {MIN_IMAGES}, ideally {RECOMMENDED_IMAGES}+, "
            "orbiting the subject rather than panning across it",
        ))
    elif count < RECOMMENDED_IMAGES:
        problems.append(Problem(
            "warning", "sparse_capture",
            f"{count} images is thin for a full orbit",
            f"{RECOMMENDED_IMAGES}+ gives a denser, less holey mesh",
        ))

    if count > MAX_IMAGES_16GB:
        problems.append(Problem(
            "warning", "large_capture",
            f"{count} images will stress a 16 GB card at the DepthMap stage",
            "cap the count or raise Meshroom's downscale factor",
        ))

    missing_focal = [f for f in report.frames if f.focal_length_mm is None]
    if len(missing_focal) == count:
        problems.append(Problem(
            "blocker", "no_focal_length",
            "no image carries an EXIF focal length",
            "supply originals — cropping or re-exporting strips EXIF, and "
            "without it Meshroom must guess the camera intrinsics",
        ))
    elif missing_focal:
        problems.append(Problem(
            "warning", "partial_focal_length",
            f"{len(missing_focal)} of {count} images have no focal length",
            "those frames will be solved with guessed intrinsics",
        ))

    if len(report.focal_lengths) > 1:
        lengths = ", ".join(f"{v:g}mm" for v in sorted(report.focal_lengths))
        problems.append(Problem(
            "warning", "focal_length_varies",
            f"several focal lengths in one capture ({lengths})",
            "lock the zoom; a varying focal length splits the intrinsics "
            "groups and weakens the solve",
        ))

    if len(report.cameras) > 1:
        problems.append(Problem(
            "warning", "mixed_cameras",
            f"images from {len(report.cameras)} different cameras",
            "Meshroom handles this, but a single body gives a cleaner solve",
        ))

    if report.with_gps == 0:
        problems.append(Problem(
            "warning", "no_gps",
            "no image carries GPS",
            "the mesh will reconstruct but cannot be georeferenced "
            "automatically — it will need manual control points, so measure "
            "two known points on site",
        ))
    elif not report.can_georeference:
        problems.append(Problem(
            "warning", "sparse_gps",
            f"only {report.with_gps} of {count} images carry GPS",
            "too few to average into a reliable origin; treat the position as "
            "a hint rather than a fix",
        ))

    measured = [f for f in report.frames if f.sharpness is not None]
    suspect = [f for f in measured if f.sharpness < BLUR_THRESHOLD]
    if measured and len(suspect) > len(measured) // 4:
        problems.append(Problem(
            "warning", "motion_blur",
            f"{len(suspect)} of {len(measured)} frames look soft",
            "raise the shutter speed or walk more slowly; blur is invisible in "
            "a thumbnail and degrades feature matching badly",
        ))

    return problems


def inspect_capture(paths, *, measure_sharpness: bool = True) -> CaptureReport:
    """Read every frame and validate the set. The whole pre-flight, in one call."""
    frames = []
    for path in paths:
        try:
            frames.append(read_frame_info(path, measure_sharpness=measure_sharpness))
        except (ReconError, FileNotFoundError):
            continue  # unreadable frames are simply not part of the capture
    report = CaptureReport(frames=frames)
    report.problems = validate_capture(report)
    return report


# ── External tools ────────────────────────────────────────────────────────────

def extract_frames(video, out_dir, *, fps: float = 2.0, ffmpeg: str | None = None) -> list[Path]:
    """
    Pull stills out of a video with ffmpeg.

    2 fps at walking pace gives roughly the 60–80% overlap a solve wants.
    Higher rates mostly add near-duplicate frames, which cost reconstruction
    time without adding parallax.

    Video is the weaker input: frames are compressed, often rolling-shuttered,
    and most cameras write GPS once per *clip* rather than per frame, so the
    georeferencing signal is much thinner than with stills.
    """
    binary = ffmpeg or shutil.which("ffmpeg")
    if not binary:
        raise ReconError(
            "ffmpeg is not installed or not on PATH. It is needed to turn video "
            "into frames; shooting stills instead skips this step entirely."
        )

    video = Path(video)
    if not video.exists():
        raise ReconError(f"no such video: {video}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(out_dir / "frame_%05d.jpg")
    command = [
        binary, "-hide_banner", "-loglevel", "error",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-q:v", "2",          # near-max JPEG quality; artefacts cost features
        pattern,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReconError(f"ffmpeg failed: {result.stderr.strip()[:400]}")

    return sorted(out_dir.glob("frame_*.jpg"))


def meshroom_command(
    images_dir,
    output_dir,
    *,
    meshroom_batch: str | None = None,
    cache_dir=None,
    max_dimension: int | None = None,
) -> list[str]:
    """
    Build the ``meshroom_batch`` invocation. Pure — returns argv, runs nothing.

    Separated from execution so the command can be asserted in tests and shown
    to a user before an hour of GPU time is committed to it.
    """
    binary = meshroom_batch or shutil.which("meshroom_batch") or "meshroom_batch"

    command = [
        binary,
        "--input", str(Path(images_dir)),
        "--output", str(Path(output_dir)),
    ]
    if cache_dir:
        command += ["--cache", str(Path(cache_dir))]
    if max_dimension:
        # Caps the DepthMap stage, which is where a 16 GB card runs out first.
        command += ["--paramOverrides", f"DepthMap.downscale={_downscale_for(max_dimension)}"]
    return command


def _downscale_for(max_dimension: int) -> int:
    """Meshroom's downscale is a power-of-two divisor; pick the smallest that fits."""
    for factor in (1, 2, 4, 8, 16):
        if max_dimension / factor <= 1600:
            return factor
    return 16


def run_meshroom(images_dir, output_dir, **kwargs) -> dict:
    """
    Run a reconstruction. Blocks for a long time — tens of minutes to hours.

    Refuses to start if ``meshroom_batch`` is not on PATH, rather than failing
    deep inside a subprocess with an opaque error.
    """
    command = meshroom_command(images_dir, output_dir, **kwargs)
    if not shutil.which(command[0]) and not os.path.exists(command[0]):
        raise ReconError(
            f"{command[0]} not found. Install Meshroom (MPL-2.0, commercial use "
            "permitted) and put its bin directory on PATH."
        )

    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "succeeded": result.returncode == 0,
        "stderr_tail": result.stderr.strip()[-2000:] if result.stderr else "",
        "outputs": sorted(str(p) for p in Path(output_dir).glob("*")) if
        Path(output_dir).exists() else [],
    }
