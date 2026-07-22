"""
Photogrammetry ingest tests — no footage, no ffmpeg, no Meshroom, no GPU.

Every image these use is synthesised in-process with real EXIF, so the suite
carries no binary fixtures and still exercises the actual EXIF path rather than
a mock of it.

The check that matters most here is the GPS hemisphere reference. Latitude and
longitude arrive as unsigned degrees/minutes/seconds with the hemisphere in a
*separate* tag. Ignore it and every southern-hemisphere capture lands in the
north and every American one in Asia — a confident, precise, completely wrong
position, which is this project's signature failure mode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

pytest.importorskip("PIL", reason="Pillow is needed to read EXIF")

import numpy as np  # noqa: E402
from PIL import ExifTags, Image  # noqa: E402

import recon  # noqa: E402
from recon import (  # noqa: E402
    MIN_IMAGES,
    CaptureReport,
    FrameInfo,
    ReconError,
    inspect_capture,
    meshroom_command,
    read_frame_info,
    sharpness,
    validate_capture,
)

# Dubai Marina, as unsigned DMS — the same magnitudes reused in each hemisphere.
LAT_DMS = (25.0, 12.0, 17.28)   # 25.2048
LON_DMS = (55.0, 16.0, 14.88)   # 55.2708


def _write_image(
    path,
    *,
    size=(640, 480),
    focal=24.0,
    camera="TestCam",
    lat_ref=None,
    lon_ref=None,
    noise=False,
):
    if noise:
        rng = np.random.default_rng(7)
        array = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        image = Image.fromarray(array)
    else:
        image = Image.new("RGB", size, (120, 120, 120))

    exif = Image.Exif()
    if camera:
        exif[ExifTags.Base.Model] = camera
    if focal:
        exif[ExifTags.IFD.Exif] = {ExifTags.Base.FocalLength: focal}
    if lat_ref and lon_ref:
        exif[ExifTags.IFD.GPSInfo] = {
            ExifTags.GPS.GPSLatitude: LAT_DMS,
            ExifTags.GPS.GPSLatitudeRef: lat_ref,
            ExifTags.GPS.GPSLongitude: LON_DMS,
            ExifTags.GPS.GPSLongitudeRef: lon_ref,
        }
    image.save(path, exif=exif)
    return Path(path)


# ── EXIF ──────────────────────────────────────────────────────────────────────

def test_reads_dimensions_and_camera(tmp_path):
    info = read_frame_info(_write_image(tmp_path / "a.jpg", size=(800, 600)))
    assert (info.width, info.height) == (800, 600)
    assert info.camera == "TestCam"
    assert info.megapixels == pytest.approx(0.48)


def test_reads_focal_length(tmp_path):
    assert read_frame_info(_write_image(tmp_path / "a.jpg", focal=35.0)).focal_length_mm == 35.0


def test_missing_exif_is_not_fatal(tmp_path):
    """A frame stripped by an editor still has pixels worth reconstructing."""
    Image.new("RGB", (320, 240)).save(tmp_path / "bare.jpg")
    info = read_frame_info(tmp_path / "bare.jpg")
    assert info.focal_length_mm is None
    assert not info.has_gps
    assert info.width == 320


def test_unreadable_file_raises(tmp_path):
    (tmp_path / "not.jpg").write_text("this is not an image")
    with pytest.raises(ReconError):
        read_frame_info(tmp_path / "not.jpg")


# ── The hemisphere trap ───────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "lat_ref,lon_ref,expect_lat,expect_lon",
    [
        ("N", "E", 25.2048, 55.2708),     # Dubai
        ("S", "E", -25.2048, 55.2708),    # Australia
        ("N", "W", 25.2048, -55.2708),    # Atlantic
        ("S", "W", -25.2048, -55.2708),   # Brazil
    ],
)
def test_gps_hemisphere_reference_is_applied(tmp_path, lat_ref, lon_ref, expect_lat, expect_lon):
    """
    EXIF stores unsigned DMS with the hemisphere in a separate tag. Dropping it
    is not a small error: the same numbers place a site in Dubai, Australia,
    the Atlantic or Brazil.
    """
    info = read_frame_info(
        _write_image(tmp_path / "g.jpg", lat_ref=lat_ref, lon_ref=lon_ref)
    )
    assert info.latitude == pytest.approx(expect_lat, abs=1e-4)
    assert info.longitude == pytest.approx(expect_lon, abs=1e-4)


def test_dms_conversion_is_exact():
    assert recon._dms_to_degrees((25.0, 12.0, 17.28)) == pytest.approx(25.2048)
    assert recon._dms_to_degrees((0.0, 30.0, 0.0)) == pytest.approx(0.5)


def test_malformed_dms_is_rejected():
    for bad in (None, (), (1.0, 2.0), ("a", "b", "c")):
        assert recon._dms_to_degrees(bad) is None


def test_out_of_range_gps_is_discarded(tmp_path):
    """Corrupt GPS must not become a confident position."""
    image = Image.new("RGB", (100, 100))
    exif = Image.Exif()
    exif[ExifTags.IFD.GPSInfo] = {
        ExifTags.GPS.GPSLatitude: (200.0, 0.0, 0.0),
        ExifTags.GPS.GPSLatitudeRef: "N",
        ExifTags.GPS.GPSLongitude: (55.0, 0.0, 0.0),
        ExifTags.GPS.GPSLongitudeRef: "E",
    }
    image.save(tmp_path / "bad.jpg", exif=exif)
    assert not read_frame_info(tmp_path / "bad.jpg").has_gps


# ── Sharpness ─────────────────────────────────────────────────────────────────

def test_noise_is_sharper_than_flat(tmp_path):
    noisy = _write_image(tmp_path / "noise.jpg", noise=True)
    flat = _write_image(tmp_path / "flat.jpg")
    assert sharpness(noisy) > sharpness(flat)


def test_blurring_reduces_sharpness(tmp_path):
    """
    The measure has to respond to blur specifically, not just to content — a
    detector that ranks a blurred frame above a sharp one is worse than none.
    """
    from PIL import ImageFilter

    rng = np.random.default_rng(3)
    array = rng.integers(0, 255, (400, 400, 3), dtype=np.uint8)
    sharp_path = tmp_path / "sharp.jpg"
    blur_path = tmp_path / "blur.jpg"
    Image.fromarray(array).save(sharp_path, quality=95)
    Image.fromarray(array).filter(ImageFilter.GaussianBlur(4)).save(blur_path, quality=95)

    assert sharpness(blur_path) < sharpness(sharp_path)


def test_flat_image_has_no_high_frequency_content(tmp_path):
    assert sharpness(_write_image(tmp_path / "flat.jpg")) == pytest.approx(0.0, abs=1e-6)


# ── Report properties ─────────────────────────────────────────────────────────

def _frames(n, *, gps=0, focal=24.0, camera="TestCam"):
    out = []
    for i in range(n):
        info = FrameInfo(path=Path(f"f{i}.jpg"), width=4000, height=3000,
                         focal_length_mm=focal, camera=camera, sharpness=500.0)
        if i < gps:
            info.latitude, info.longitude = 25.2048 + i * 1e-5, 55.2708
        out.append(info)
    return out


def test_gps_centroid_averages_only_located_frames():
    report = CaptureReport(frames=_frames(10, gps=4))
    lat, lon = report.gps_centroid()
    assert lat == pytest.approx(25.20481, abs=1e-4)
    assert lon == pytest.approx(55.2708)


def test_no_gps_means_no_centroid():
    assert CaptureReport(frames=_frames(5)).gps_centroid() is None


def test_georeferencing_needs_a_majority_with_gps():
    """
    A handful of tagged frames among hundreds is not a fix. Averaging them
    anyway would place the site confidently in the wrong spot.
    """
    assert CaptureReport(frames=_frames(100, gps=60)).can_georeference
    assert not CaptureReport(frames=_frames(100, gps=5)).can_georeference


def test_georeferencing_needs_at_least_three_points():
    assert not CaptureReport(frames=_frames(4, gps=2)).can_georeference


# ── Validation ────────────────────────────────────────────────────────────────

def _validated(frames):
    report = CaptureReport(frames=frames)
    report.problems = validate_capture(report)
    return report


def _codes(report):
    return {p.code for p in report.problems}


def test_empty_capture_is_blocked():
    report = _validated([])
    assert "no_images" in _codes(report)
    assert not report.can_reconstruct


def test_too_few_images_is_a_blocker():
    report = _validated(_frames(MIN_IMAGES - 1, gps=10))
    assert "too_few_images" in _codes(report)
    assert not report.can_reconstruct


def test_a_thin_but_workable_capture_warns_without_blocking():
    report = _validated(_frames(30, gps=30))
    assert "sparse_capture" in _codes(report)
    assert report.can_reconstruct


def test_a_good_capture_has_no_blockers():
    report = _validated(_frames(120, gps=120))
    assert report.can_reconstruct
    assert report.blockers == []


def test_missing_focal_length_everywhere_is_a_blocker():
    report = _validated(_frames(80, gps=80, focal=None))
    assert "no_focal_length" in _codes(report)
    assert not report.can_reconstruct


def test_partially_missing_focal_length_only_warns():
    frames = _frames(80, gps=80)
    for f in frames[:5]:
        f.focal_length_mm = None
    report = _validated(frames)
    assert "partial_focal_length" in _codes(report)
    assert report.can_reconstruct


def test_varying_focal_length_warns():
    """A zoom that moved mid-capture splits the intrinsics groups."""
    frames = _frames(80, gps=80)
    for f in frames[40:]:
        f.focal_length_mm = 50.0
    report = _validated(frames)
    assert "focal_length_varies" in _codes(report)
    assert "24mm" in str(report.problems) and "50mm" in str(report.problems)


def test_mixed_cameras_warn():
    frames = _frames(80, gps=80)
    for f in frames[40:]:
        f.camera = "OtherCam"
    assert "mixed_cameras" in _codes(_validated(frames))


def test_no_gps_warns_but_does_not_block():
    """The mesh still reconstructs; it just cannot place itself."""
    report = _validated(_frames(80))
    assert "no_gps" in _codes(report)
    assert report.can_reconstruct
    assert not report.can_georeference


def test_sparse_gps_warns_separately():
    assert "sparse_gps" in _codes(_validated(_frames(80, gps=4)))


def test_widespread_blur_warns():
    frames = _frames(80, gps=80)
    for f in frames[:40]:
        f.sharpness = 10.0
    assert "motion_blur" in _codes(_validated(frames))


def test_a_few_soft_frames_do_not_warn():
    """Some softness is normal; only a substantial fraction is a capture problem."""
    frames = _frames(80, gps=80)
    for f in frames[:5]:
        f.sharpness = 10.0
    assert "motion_blur" not in _codes(_validated(frames))


def test_oversized_capture_warns_about_vram():
    assert "large_capture" in _codes(_validated(_frames(600, gps=600)))


def test_problems_carry_a_remedy():
    """A problem the user cannot act on is just noise."""
    for problem in _validated(_frames(5)).problems:
        assert problem.remedy and len(problem.remedy) > 15


def test_report_serialises():
    data = _validated(_frames(80, gps=80)).as_dict()
    assert data["frames"] == 80
    assert data["can_reconstruct"] is True
    assert data["gps_centroid"]["longitude"] == pytest.approx(55.2708)


# ── End to end over real files ────────────────────────────────────────────────

def test_inspect_capture_reads_a_folder(tmp_path):
    for i in range(3):
        _write_image(tmp_path / f"img{i}.jpg", lat_ref="N", lon_ref="E")
    report = inspect_capture(sorted(tmp_path.glob("*.jpg")))
    assert report.count == 3
    assert report.with_gps == 3
    assert report.focal_lengths == {24.0}


def test_inspect_capture_skips_unreadable_files(tmp_path):
    _write_image(tmp_path / "good.jpg")
    (tmp_path / "bad.jpg").write_text("junk")
    report = inspect_capture(sorted(tmp_path.glob("*.jpg")))
    assert report.count == 1


# ── External tool orchestration ───────────────────────────────────────────────

def test_meshroom_command_is_pure_and_complete(tmp_path):
    command = meshroom_command(tmp_path / "in", tmp_path / "out", meshroom_batch="mb")
    assert command[0] == "mb"
    assert "--input" in command and "--output" in command
    assert str(tmp_path / "in") in command


def test_meshroom_command_accepts_a_cache_dir(tmp_path):
    command = meshroom_command(tmp_path / "i", tmp_path / "o",
                               meshroom_batch="mb", cache_dir=tmp_path / "c")
    assert "--cache" in command


def test_downscale_backs_off_for_large_images():
    assert recon._downscale_for(1600) == 1
    assert recon._downscale_for(3200) == 2
    assert recon._downscale_for(6400) == 4


def test_max_dimension_sets_a_depthmap_override(tmp_path):
    command = meshroom_command(tmp_path / "i", tmp_path / "o",
                               meshroom_batch="mb", max_dimension=6000)
    assert any("DepthMap.downscale" in part for part in command)


def test_extract_frames_without_ffmpeg_says_so(tmp_path, monkeypatch):
    """Fail with the reason and the alternative, not with a FileNotFoundError."""
    monkeypatch.setattr(recon.shutil, "which", lambda _n: None)
    (tmp_path / "clip.mp4").write_bytes(b"\x00")
    with pytest.raises(ReconError, match="ffmpeg is not installed"):
        recon.extract_frames(tmp_path / "clip.mp4", tmp_path / "frames")


def test_extract_frames_checks_the_video_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(recon.shutil, "which", lambda _n: "ffmpeg")
    with pytest.raises(ReconError, match="no such video"):
        recon.extract_frames(tmp_path / "missing.mp4", tmp_path / "frames")


def test_run_meshroom_refuses_early_when_the_binary_is_absent(tmp_path, monkeypatch):
    """Better than failing deep inside a subprocess with an opaque error."""
    monkeypatch.setattr(recon.shutil, "which", lambda _n: None)
    with pytest.raises(ReconError, match="MPL-2.0"):
        recon.run_meshroom(tmp_path / "i", tmp_path / "o")
