"""Tests for vcrunch script."""

# mypy: ignore-errors

import importlib.util
import io
import json
import logging
import os
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "vcrunch.py"
_SPEC = importlib.util.spec_from_file_location("vcrunch", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)


_CONFIG_VARS = [
    "VCRUNCH_INPUTS",
    "VCRUNCH_PATHS_FROM",
    "VCRUNCH_PATTERN",
    "VCRUNCH_MEDIA",
    "VCRUNCH_TARGET_SIZE",
    "VCRUNCH_CONSTANT_QUALITY",
    "VCRUNCH_AUDIO_BITRATE",
    "VCRUNCH_SAFETY_OVERHEAD",
    "VCRUNCH_OUTPUT_DIR",
    "VCRUNCH_MANIFEST_NAME",
    "VCRUNCH_NAME_SUFFIX",
    "VCRUNCH_MOVE_IF_FIT",
    "VCRUNCH_STAGE_DIR",
    "VCRUNCH_LIST_ERRORS",
    "VCRUNCH_SVT_LP",
    "SVT_LP",
    "VCRUNCH_VERBOSE",
]


def set_env(monkeypatch, **overrides: str) -> None:
    for key in _CONFIG_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def test_parse_size():
    assert script.parse_size("1") == 1
    assert script.parse_size("1k") == 1024
    assert script.parse_size("1.5m") == int(1.5 * 1024**2)
    assert script.parse_size("2g") == 2 * 1024**3
    assert script.parse_size("3t") == 3 * 1024**4
    assert script.parse_size("1KiB") == 1024


def test_kbps_to_bps():
    assert script.kbps_to_bps("1k") == 1000
    assert script.kbps_to_bps("1.5m") == 1_500_000
    assert script.kbps_to_bps("500") == 500


def test_ffprobe_json(monkeypatch):
    def fake_run(cmd, check, stdout, stderr):
        assert cmd == ["ffprobe", "file"]
        assert check is True
        assert stdout is script.subprocess.PIPE
        assert stderr is script.subprocess.PIPE

        class Result:
            def __init__(self) -> None:
                self.stdout = b'{"a": 1}'
                self.stderr = b""

        return Result()

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    assert script.ffprobe_json(["ffprobe", "file"]) == {"a": 1}


def test_parse_time_value_fraction():
    assert script._parse_time_value("1/2") == pytest.approx(0.5)
    assert script._parse_time_value("  ") is None
    assert script._parse_time_value(None) is None


def test_collect_frame_timestamps_seconds_fallback(monkeypatch):
    calls = []

    def fake_ffprobe_json(cmd):
        calls.append(cmd)
        if "-show_frames" in cmd:
            return {"frames": []}
        return {
            "packets": [
                {"pts_time": "0"},
                {"pts_time": "1/2"},
                {"dts_time": "1"},
            ]
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    timestamps = script._collect_frame_timestamps_seconds("input.mpg", 0, "v:0")
    assert timestamps == pytest.approx([0.0, 0.5, 1.0])
    assert any("-show_packets" in cmd for cmd in calls)
    assert any("v:0" in cmd for cmd in calls)


def test_dump_streams_data_fallback(monkeypatch, tmp_path):
    metadata = {
        "format": {"format_name": "matroska"},
        "streams": [
            {
                "index": 0,
                "codec_type": "data",
                "codec_name": "dvd_nav_packet",
            }
        ],
    }

    def fake_ffprobe_json(cmd):
        if "-show_packets" in cmd:
            return {
                "packets": [
                    {"pts_time": "0"},
                    {"pts_time": "1"},
                ]
            }
        return metadata

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)

    monkeypatch.setattr(
        script,
        "_export_attachments",
        lambda src, dest_dir, verbose: [],
    )

    result = script._dump_streams_and_metadata(
        str(tmp_path / "input.mkv"), tmp_path, False, naming_stem="input"
    )

    exports = result["exports"]
    assert len(exports) == 1
    entry = exports[0]
    assert entry["stype"] == "d"
    assert entry["mkv_ok"] is False

    data_path = Path(entry["path"])
    assert data_path.name == "legacy_stream_d0.data.unknown.data"
    assert entry.get("muxer") == "data"
    assert not data_path.exists()

    packets_path = data_path.with_suffix(".timing.json")
    assert packets_path.exists()
    with packets_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload == {"packets": [0.0, 1.0]}
    assert entry.get("packet_timestamps_path") == str(packets_path)

    assert Path(entry["path"]).with_suffix(".timing.json").exists()


def test_dump_streams_data_fallback_empty_packets(monkeypatch, tmp_path):
    metadata = {
        "format": {"format_name": "matroska"},
        "streams": [
            {
                "index": 0,
                "codec_type": "data",
                "codec_name": "dvd_nav_packet",
            }
        ],
    }

    def fake_ffprobe_json(cmd):
        if "-show_packets" in cmd:
            return {"packets": []}
        return metadata

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)

    monkeypatch.setattr(
        script,
        "_export_attachments",
        lambda src, dest_dir, verbose: [],
    )

    result = script._dump_streams_and_metadata(
        str(tmp_path / "input.mkv"), tmp_path, False, naming_stem="input"
    )

    exports = result["exports"]
    assert len(exports) == 1
    entry = exports[0]
    packets_path = Path(entry["path"]).with_suffix(".timing.json")
    assert packets_path.exists()
    with packets_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload == {"packets": []}


def test_compute_stream_bitrate_aggregates_packets(monkeypatch):
    calls = []

    def fake_ffprobe_json(cmd):
        calls.append(cmd)
        if "-show_packets" in cmd:
            return {
                "packets": [
                    {
                        "stream_index": "3",
                        "pts_time": "0",
                        "duration_time": "1",
                        "size": "1000",
                    },
                    {
                        "stream_index": "3",
                        "pts_time": "1",
                        "duration_time": "1",
                        "size": "1000",
                    },
                    {
                        "stream_index": "4",
                        "pts_time": "0",
                        "duration_time": "2",
                        "size": "2000",
                    },
                ]
            }
        return {
            "format": {"duration": "4"},
            "streams": [
                {"index": 3, "duration": "3"},
                {"index": 4, "duration": "3.5"},
            ],
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    monkeypatch.setattr(script.shutil, "which", lambda name: "/usr/bin/ffprobe")

    metrics = script._compute_stream_bitrate("clip.mkv", "d:0", stream_index=3)
    assert metrics is not None
    assert metrics["bitrate"] == pytest.approx(8_000.0)
    assert metrics["total_bytes"] == 2000
    assert any("-show_packets" in cmd for cmd in calls)


def test_estimate_other_stream_bytes_uses_packet_probe(monkeypatch):
    stream = {"codec_type": "data", "index": 2}

    def fake_compute(source_path, spec, *, stream_index=None):
        assert source_path == "clip.mkv"
        assert spec == "d:0"
        assert stream_index == 2
        return {"bitrate": 24_000.0, "duration": 10.0, "total_bytes": 30_000}

    monkeypatch.setattr(script, "_compute_stream_bitrate", fake_compute)

    debug_entries: list[script.BudgetDebugEntry] = []
    estimated, entry = script._estimate_other_stream_bytes(
        stream,
        10.0,
        "d",
        source_path="clip.mkv",
        stream_spec="d:0",
        debug_entries=debug_entries,
        debug_source="clip",
    )
    assert estimated == 30_000
    assert entry is not None
    assert entry["method"] == "packet-bytes"


def test_estimate_other_stream_bytes_defaults_when_probe_missing(monkeypatch):
    stream = {"codec_type": "data"}
    monkeypatch.setattr(script, "_compute_stream_bitrate", lambda *args, **kwargs: None)

    debug_entries: list[script.BudgetDebugEntry] = []
    estimated, entry = script._estimate_other_stream_bytes(
        stream,
        2.5,
        "d",
        source_path="clip.mkv",
        stream_spec="d:0",
        debug_entries=debug_entries,
        debug_source="clip",
    )
    assert estimated == int(2.5 * 4000)
    assert entry is not None
    assert entry["method"] == "data-fallback"


def test_estimate_other_stream_bytes_records_debug(monkeypatch):
    stream = {"codec_type": "data", "index": 1}
    entries: list[script.BudgetDebugEntry] = []

    monkeypatch.setattr(
        script,
        "_compute_stream_bitrate",
        lambda *args, **kwargs: {
            "bitrate": 12_000.0,
            "duration": 5.0,
            "total_bytes": 7_500,
        },
    )

    estimated, entry = script._estimate_other_stream_bytes(
        stream,
        5.0,
        "d",
        source_path="clip.mkv",
        stream_spec="d:0",
        debug_entries=entries,
        debug_source="clip",
    )

    assert estimated == 7_500
    assert entries
    entry = entries[0]
    assert entry["source"] == "clip"
    assert entry["bytes"] == 7_500
    assert entry["method"] == "packet-bytes"


def test_video_copy_budget_uses_measured_bytes(monkeypatch, tmp_path, caplog):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video_path = src_dir / "clip.mp4"
    video_path.write_bytes(b"input")

    real_getsize = os.path.getsize

    def fake_getsize(path):
        try:
            candidate = Path(path)
        except TypeError:
            candidate = Path(str(path))
        if candidate == video_path:
            return 50_000_000
        return real_getsize(path)

    monkeypatch.setattr(os.path, "getsize", fake_getsize)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="40M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_VERBOSE="1",
    )

    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": path.endswith(".mp4"), "duration": 10.0},
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 10.0)

    base_stream_infos = [
        {
            "index": 0,
            "stream": {
                "codec_type": "video",
                "codec_name": "h264",
                "index": 0,
                "duration": "10.0",
            },
            "stype": "v",
            "mkv_ok": True,
            "spec": "v:0",
        },
        {
            "index": 1,
            "stream": {
                "codec_type": "audio",
                "codec_name": "aac",
                "index": 1,
                "duration": "10.0",
            },
            "stype": "a",
            "mkv_ok": True,
            "spec": "a:0",
        },
    ]

    monkeypatch.setattr(
        script,
        "_probe_stream_infos_only",
        lambda path: [
            {
                "index": info["index"],
                "stream": dict(info["stream"]),
                "stype": info["stype"],
                "mkv_ok": info["mkv_ok"],
                "spec": info["spec"],
            }
            for info in base_stream_infos
        ],
    )

    def fake_dump(src, dest_dir, verbose, **kwargs):
        dest_dir.mkdir(parents=True, exist_ok=True)
        return {
            "exports": [],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {},
            "stream_infos": [
                {
                    "index": info["index"],
                    "stream": dict(info["stream"]),
                    "stype": info["stype"],
                    "mkv_ok": info["mkv_ok"],
                    "spec": info["spec"],
                }
                for info in base_stream_infos
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)

    def fake_extract(stream):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            return 1_000_000
        if codec_type == "audio":
            return 128_000
        return None

    monkeypatch.setattr(script, "_extract_stream_bitrate", fake_extract)

    compute_calls: list[tuple[str, str, Optional[int]]] = []

    def fake_compute(path, spec, *, stream_index=None):
        compute_calls.append((path, spec, stream_index))
        if spec == "v:0":
            return {"bitrate": 2_000_000.0, "duration": 10.0, "total_bytes": 2_500_000}
        return None

    monkeypatch.setattr(script, "_compute_stream_bitrate", fake_compute)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)
    monkeypatch.setattr(
        script, "_apply_source_timestamps", lambda *args, **kwargs: None
    )

    def fake_run(cmd, **kwargs):
        if not cmd:
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if cmd[0] == "ffmpeg":
            output_path = None
            if "-f" in cmd:
                for idx, token in enumerate(cmd):
                    if token == "-f" and idx + 2 < len(cmd):
                        output_path = Path(cmd[idx + 2])
            if output_path is None and cmd:
                output_path = Path(cmd[-1])
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"ffmpeg")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if cmd[0] == "mkvmerge":
            out_idx = cmd.index("-o")
            output_path = Path(cmd[out_idx + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"muxed")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if cmd[0] == "mkvpropedit":
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    caplog.set_level(logging.INFO)
    script.main()

    record = next(
        (rec for rec in caplog.records if "copied video bytes" in rec.getMessage()),
        None,
    )
    assert record is not None
    assert "2,500,000" in record.getMessage()
    assert any(spec == "v:0" for _, spec, _ in compute_calls)

    budget_header = next(
        (
            rec
            for rec in caplog.records
            if "stream budget for clip.mp4" in rec.getMessage()
        ),
        None,
    )
    assert budget_header is not None
    assert any("v stream v:0" in rec.getMessage() for rec in caplog.records)
    assert any("difference   ->" in rec.getMessage() for rec in caplog.records)
    assert any(
        "bitrate calculation steps" in rec.getMessage() for rec in caplog.records
    )


def test_dump_streams_subtitle_uses_container_data(monkeypatch, tmp_path):
    metadata = {
        "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "tx3g",
                "tags": {"language": "eng"},
            },
        ],
    }

    monkeypatch.setattr(script, "ffprobe_json", lambda cmd: metadata)
    monkeypatch.setattr(
        script,
        "_export_attachments",
        lambda src, dest_dir, verbose: [],
    )

    result = script._dump_streams_and_metadata("clip.mov", tmp_path, verbose=False)

    exports = result["exports"]
    assert len(exports) == 1
    entry = exports[0]
    assert entry["stype"] == "s"
    assert entry["mkv_ok"] is False
    assert entry["path"].endswith(".mov")
    assert entry.get("muxer") == "mov"
    assert entry.get("spec") == "s:0"
    sidecar_path = Path(entry["path"])
    assert sidecar_path.name == "legacy_stream_s2.subtitle.unknown.mov"
    assert not sidecar_path.exists()


def test_packet_sidecar_path_prefers_recorded(tmp_path):
    export_path = tmp_path / "sample.data"
    export_path.write_bytes(b"")
    recorded = tmp_path / "custom.timing.json"
    recorded.write_text("{}", encoding="utf-8")
    export: script.StreamExport = {
        "path": str(export_path),
        "stream": {},
        "stype": "d",
        "mkv_ok": False,
        "packet_timestamps_path": str(recorded),
    }

    result = script._packet_sidecar_path(export, export_path)
    assert result == recorded


def test_packet_sidecar_path_infers_missing_record(tmp_path):
    export_path = tmp_path / "sample.data"
    export_path.write_bytes(b"")
    inferred = export_path.with_suffix(".timing.json")
    inferred.write_text("{}", encoding="utf-8")
    export: script.StreamExport = {
        "path": str(export_path),
        "stream": {},
        "stype": "d",
        "mkv_ok": False,
    }

    result = script._packet_sidecar_path(export, export_path)
    assert result == inferred


def test_packet_sidecar_path_missing_file(tmp_path):
    export_path = tmp_path / "sample.data"
    export_path.write_bytes(b"")
    export: script.StreamExport = {
        "path": str(export_path),
        "stream": {},
        "stype": "d",
        "mkv_ok": False,
    }

    result = script._packet_sidecar_path(export, export_path)
    assert result is None


def test_ffprobe_duration(monkeypatch):
    monkeypatch.setattr(script, "probe_media_info", lambda path: {"duration": 12.34})
    assert script.ffprobe_duration("path") == 12.34

    monkeypatch.setattr(script, "probe_media_info", lambda path: {"duration": None})
    with pytest.raises(ValueError):
        script.ffprobe_duration("path")


def test_get_container_creation_date(monkeypatch):
    expected = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=creation_time,com.apple.quicktime.creationdate",
        "video.mkv",
    ]

    def fake_ffprobe_json(cmd):
        assert cmd == expected
        return {
            "format": {
                "tags": {
                    "creation_time": "2024-09-28T15:42:11.123456Z",
                }
            }
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    assert script.get_container_creation_date("video.mkv") == "2024-09-28T15:42:11Z"


def test_get_container_creation_date_missing(monkeypatch):
    monkeypatch.setattr(
        script,
        "ffprobe_json",
        lambda cmd: {"format": {"tags": {}}},
    )
    assert script.get_container_creation_date("video.mkv") is None


def test_probe_media_info_uses_stream_duration(monkeypatch):
    expected = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "path",
    ]

    def fake_ffprobe_json(cmd):
        assert cmd == expected
        return {
            "format": {"format_name": "matroska"},
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "12.5",
                }
            ],
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    info = script.probe_media_info("path")
    assert info["is_video"] is True
    assert info["duration"] == pytest.approx(12.5)


def test_probe_media_info_zero_duration_is_still(monkeypatch):
    def fake_ffprobe_json(cmd):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "0",
                }
            ]
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    info = script.probe_media_info("photo.jpg")
    assert info["is_video"] is False
    assert info["duration"] is None


def test_probe_media_info_detects_image_container(monkeypatch):
    expected = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "still.png",
    ]

    def fake_ffprobe_json(cmd):
        assert cmd == expected
        return {
            "format": {"format_name": "image2"},
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "1.0",
                }
            ],
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    info = script.probe_media_info("still.png")
    assert info == {"is_video": False, "duration": None}


def test_probe_media_info_attached_picture(monkeypatch):
    def fake_ffprobe_json(cmd):
        return {
            "streams": [
                {
                    "codec_type": "video",
                    "duration": "5",
                    "disposition": {"attached_pic": 1},
                }
            ]
        }

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    info = script.probe_media_info("cover.mkv")
    assert info["is_video"] is False
    assert info["duration"] is None


def test_probe_media_info_failure(monkeypatch):
    err = subprocess.CalledProcessError(1, ["ffprobe"], output=b"", stderr=b"bad")

    def fake_ffprobe_json(cmd):
        raise err

    monkeypatch.setattr(script, "ffprobe_json", fake_ffprobe_json)
    info = script.probe_media_info("bad")
    assert info["is_video"] is False
    assert info["duration"] is None
    assert info["error"] == "bad"


def test_is_valid_media(monkeypatch):
    monkeypatch.setattr(script, "ffprobe_duration", lambda p: 1.0)
    assert script.is_valid_media("file")
    monkeypatch.setattr(
        script, "ffprobe_duration", lambda p: (_ for _ in ()).throw(Exception())
    )
    assert script.is_valid_media("file") is False


def test_has_video_stream(monkeypatch):
    monkeypatch.setattr(script, "probe_media_info", lambda path: {"is_video": True})
    assert script.has_video_stream("path") is True

    monkeypatch.setattr(script, "probe_media_info", lambda path: {"is_video": False})
    assert script.has_video_stream("path") is False


def test_is_video_file(monkeypatch):
    calls: list[str] = []

    def fake_has_video_stream(path: str) -> bool:
        calls.append(path)
        return path.endswith(".custom")

    monkeypatch.setattr(script, "has_video_stream", fake_has_video_stream)

    assert script.is_video_file("/tmp/image.JPG") is False
    assert script.is_video_file("/tmp/video.mp4") is False
    assert script.is_video_file("/tmp/video.custom") is True
    assert script.is_video_file("/tmp/asset.bin") is False
    assert calls == [
        "/tmp/image.JPG",
        "/tmp/video.mp4",
        "/tmp/video.custom",
        "/tmp/asset.bin",
    ]


def test_copy_assets_skips_done(tmp_path):
    src = tmp_path / "asset.bin"
    src.write_text("source-new")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    dest = out_dir / "asset.bin"
    dest.write_text("existing")

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "items": {},
        "probes": {},
    }

    st = src.stat()
    key = script.src_key(str(src.resolve()), st)
    manifest["items"][key] = {
        "type": "asset",
        "src": str(src),
        "output": "asset.bin",
        "status": "done",
        "finished_at": "2024-01-01T00:00:00Z",
    }

    results = script.copy_assets(
        [str(src)],
        str(out_dir),
        manifest=manifest,
        manifest_path=str(manifest_path),
    )

    assert dest.read_text() == "existing"
    assert results == [(str(src), "asset.bin")]


def test_copy_assets_preserves_metadata(tmp_path):
    src = tmp_path / "asset.bin"
    src.write_text("content")
    ts = 1_600_000_000
    os.utime(src, (ts, ts))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    results = script.copy_assets([str(src)], str(out_dir))

    dest = out_dir / "asset.bin"
    assert dest.exists()
    assert results == [(str(src), "asset.bin")]
    assert int(dest.stat().st_mtime) == ts


def test_copy_assets_lowercases_extension(tmp_path):
    src = tmp_path / "asset.DATA"
    src.write_text("payload")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    results = script.copy_assets([str(src)], str(out_dir))

    dest = out_dir / "asset.data"
    assert dest.exists()
    assert not (out_dir / "asset.DATA").exists()
    assert results == [(str(src), "asset.data")]


def test_copy_assets_renames_and_preserves_metadata(tmp_path):
    src = tmp_path / "asset.txt"
    src.write_text("content")
    ts = 1_700_000_000
    os.utime(src, (ts, ts))

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    old_dest = out_dir / "old.txt"
    old_dest.write_text("old")
    os.utime(old_dest, (1_500_000_000, 1_500_000_000))

    manifest_path = tmp_path / "manifest.json"
    manifest = {"items": {}, "probes": {}}
    src_stat = src.stat()
    key = script.src_key(str(src.resolve()), src_stat)
    manifest["items"][key] = {
        "type": "asset",
        "src": str(src),
        "output": "old.txt",
        "status": "done",
        "finished_at": "2024-01-01T00:00:00Z",
    }

    results = script.copy_assets(
        [str(src)],
        str(out_dir),
        rename_map={str(src): "new.txt"},
        manifest=manifest,
        manifest_path=str(manifest_path),
    )

    new_dest = out_dir / "new.txt"
    assert not old_dest.exists()
    assert new_dest.exists()
    assert results == [(str(src), "new.txt")]
    assert int(new_dest.stat().st_mtime) == ts
    assert manifest["items"][key]["output"] == "new.txt"


def test_run_success(monkeypatch, capsys):
    def fake_run(cmd):
        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    script.run(["echo", "hi"])
    err = capsys.readouterr().err
    assert "+ echo hi" in err


def test_run_failure(monkeypatch):
    def fake_run(cmd):
        class R:
            returncode = 1

        return R()

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        script.run(["bad"])
    assert exc.value.code == 1


def test_main_keeps_original_name_when_larger(monkeypatch, tmp_path):
    src = tmp_path / "video.mp4"
    src.write_bytes(b"source-bytes")
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src),
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
    )

    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": True, "duration": 10.0},
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 10.0)
    monkeypatch.setattr(script, "find_start_timecode", lambda path: "00:00:00:00")
    monkeypatch.setattr(script, "get_container_creation_date", lambda path: None)
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)
    monkeypatch.setattr(
        script,
        "_probe_stream_infos_only",
        lambda path: [
            {
                "index": 0,
                "stream": {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "index": 0,
                },
                "stype": "v",
                "mkv_ok": True,
                "spec": "v:0",
            },
            {
                "index": 1,
                "stream": {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "index": 1,
                },
                "stype": "a",
                "mkv_ok": True,
                "spec": "a:0",
            },
        ],
    )

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded-output-is-larger")

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    script.main()

    output_file = out_dir / "video.mp4"
    assert output_file.exists()
    assert not (out_dir / "video.mkv").exists()

    manifest_path = out_dir / script.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    outputs = [rec["output"] for rec in manifest["items"].values()]
    assert outputs == ["video.mp4"]


def test_collect_all_files(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.txt").write_text("b")
    (sub / "c.log").write_text("c")
    (sub / "._ignored.txt").write_text("meta")
    paths = [str(tmp_path), str(tmp_path / "a.txt")]
    result = script.collect_all_files(paths, "*.txt")
    expected = sorted([str(tmp_path / "a.txt"), str(sub / "b.txt")])
    assert result == expected


def test_collect_all_files_skips_dot_underscore(tmp_path):
    (tmp_path / "._video.mp4").write_text("meta")
    (tmp_path / "video.mp4").write_text("data")
    result = script.collect_all_files([str(tmp_path)], None)
    assert str(tmp_path / "video.mp4") in result
    assert str(tmp_path / "._video.mp4") not in result


def test_read_paths_from_file(tmp_path):
    f = tmp_path / "paths.txt"
    f.write_text("a\n\n b \n")
    assert script.read_paths_from(str(f)) == ["a", "b"]


def test_read_paths_from_stdin(monkeypatch):
    monkeypatch.setattr(script.sys, "stdin", io.StringIO("x\ny\n"))
    assert script.read_paths_from("-") == ["x", "y"]


def test_sanitize_base():
    assert script.sanitize_base(".foo") == "foo"
    assert script.sanitize_base("..\\bar") == "_bar"
    assert script.sanitize_base("") == "file"


def test_now_utc_iso_format():
    s = script.now_utc_iso()
    datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def test_load_manifest_new(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "now_utc_iso", lambda: "TS")
    path = tmp_path / "m.json"
    assert script.load_manifest(str(path)) == {
        "version": 1,
        "updated": "TS",
        "items": {},
        "probes": {},
    }


def test_load_manifest_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "now_utc_iso", lambda: "TS")
    path = tmp_path / "m.json"
    path.write_text('{"foo": 1}')
    assert script.load_manifest(str(path)) == {"foo": 1, "items": {}, "probes": {}}


def test_load_manifest_invalid(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "now_utc_iso", lambda: "TS")
    path = tmp_path / "m.json"
    path.write_text("not json")
    assert script.load_manifest(str(path)) == {
        "version": 1,
        "updated": "TS",
        "items": {},
        "probes": {},
    }


def test_save_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(script, "now_utc_iso", lambda: "TS2")
    path = tmp_path / "m.json"
    manifest = {"version": 1, "updated": "old", "items": {}}
    script.save_manifest(manifest, str(path))
    assert manifest["updated"] == "TS2"
    data = json.loads(path.read_text())
    assert data["updated"] == "TS2"


def test_manifest_error_basenames():
    manifest = {
        "items": {
            "a": {"src": "/path/to/foo.mkv", "error": "fail"},
            "b": {"output": "bar.mov", "error": "nope"},
            "c": {"src": "/path/baz.mp4"},
            "d": "not-a-dict",
        }
    }
    result = script.manifest_error_basenames(manifest)
    assert sorted(result) == ["bar.mov", "foo.mkv"]


def test_main_list_errors(monkeypatch, tmp_path, capsys):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stage_dir = tmp_path / "stage"
    manifest = {
        "items": {
            "1": {"src": "/a/foo.mkv", "error": "fail"},
            "2": {"output": "bar.mov", "error": "bad"},
            "3": {"src": "/b/no-error.mkv"},
        }
    }
    (out_dir / script.MANIFEST_NAME).write_text(json.dumps(manifest))

    set_env(
        monkeypatch,
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_LIST_ERRORS="1",
    )

    script.main()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["bar.mov", "foo.mkv"]


def test_src_key():
    st = types.SimpleNamespace(st_size=123, st_mtime=456.7)
    assert script.src_key("/abs", st) == "/abs|123|456"


def test_all_videos_done(monkeypatch):
    manifest = {"items": {"1": {"type": "video", "output": "a.mkv", "status": "done"}}}
    monkeypatch.setattr(script.os.path, "exists", lambda p: True)
    monkeypatch.setattr(script, "is_valid_media", lambda p: True)
    assert script.all_videos_done(manifest, "/out") is True
    manifest["items"]["1"]["status"] = "pending"
    assert script.all_videos_done(manifest, "/out") is False
    assert script.all_videos_done({"items": {}}, "/out") is False


def test_copy_if_fits(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "a.mp4").write_text("a")
    (src_dir / "b.txt").write_text("b")
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="1M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": path.endswith(".mp4"), "duration": 10.0},
    )
    monkeypatch.setattr(
        script,
        "ffprobe_duration",
        lambda p: (_ for _ in ()).throw(Exception("ffprobe")),
    )
    monkeypatch.setattr(
        script, "run", lambda cmd: (_ for _ in ()).throw(Exception("run"))
    )
    script.main()
    assert (out_dir / "a.mp4").exists()
    assert (out_dir / "b.txt").exists()
    assert (src_dir / "a.mp4").exists()
    manifest = json.loads((out_dir / ".job.json").read_text())
    assert len(manifest["items"]) == 1
    rec = next(iter(manifest["items"].values()))
    assert rec["status"] == "done"


def test_move_if_fits(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "a.mp4"
    video.write_text("a")
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="1M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_MOVE_IF_FIT="1",
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": path.endswith(".mp4"), "duration": 10.0},
    )
    script.main()
    assert not video.exists()
    assert (out_dir / "a.mp4").exists()


def test_constant_quality_groups_and_command(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "a.mp4"
    video.write_bytes(b"v" * (2 * 1024 * 1024))
    asset = src_dir / "notes.txt"
    asset.write_text("notes")
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="1M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_CONSTANT_QUALITY="32",
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {
            "is_video": path.endswith(".mp4"),
            "duration": 60.0 if path.endswith(".mp4") else None,
        },
    )

    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 60.0)
    captured_cmds = []

    def fake_run(cmd, env=None, **kwargs):
        captured_cmds.append(cmd)
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
        elif cmd[0] == "mkvmerge":
            out_index = cmd.index("-o")
            output_path = Path(cmd[out_index + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"remuxed")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
            ],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {"title": "Example Title", "comment": "Example"},
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)
    monkeypatch.setattr(
        script,
        "_probe_stream_infos_only",
        lambda path: [
            {
                "index": 0,
                "stream": {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "index": 0,
                },
                "stype": "v",
                "mkv_ok": True,
                "spec": "v:0",
            },
            {
                "index": 1,
                "stream": {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "index": 1,
                },
                "stype": "a",
                "mkv_ok": True,
                "spec": "a:0",
            },
        ],
    )
    monkeypatch.setattr(
        script,
        "_probe_stream_infos_only",
        lambda path: [
            {
                "index": 0,
                "stream": {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "index": 0,
                },
                "stype": "v",
                "mkv_ok": True,
                "spec": "v:0",
            },
            {
                "index": 1,
                "stream": {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "index": 1,
                },
                "stype": "a",
                "mkv_ok": True,
                "spec": "a:0",
            },
        ],
    )

    script.main()

    dirs = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert dirs == []
    video_out = out_dir / "a.mkv"
    asset_out = out_dir / "notes.txt"
    assert video_out.exists()
    assert asset_out.exists()
    manifest_data = json.loads((out_dir / ".job.json").read_text())
    rec = next(iter(manifest_data["items"].values()))
    assert rec["output"] == "a.mkv"
    cmd = next(c for c in captured_cmds if c[0] == "ffmpeg" and "libsvtav1" in c)
    assert "-fflags" in cmd
    ff_idx = cmd.index("-fflags")
    assert cmd[ff_idx + 1] == "+genpts"
    assert "-avoid_negative_ts" in cmd
    ant_idx = cmd.index("-avoid_negative_ts")
    assert cmd[ant_idx + 1] == "make_zero"
    assert "-max_interleave_delta" in cmd
    mid_idx = cmd.index("-max_interleave_delta")
    assert cmd[mid_idx + 1] == "0"
    assert "-fps_mode:v:0" in cmd
    fps_idx = cmd.index("-fps_mode:v:0")
    assert cmd[fps_idx + 1] == "passthrough"
    assert "-crf:v:0" in cmd
    idx = cmd.index("-crf:v:0")
    assert cmd[idx + 1] == "32"
    assert cmd[idx + 2] == "-b:v:0"
    assert cmd[idx + 3] == "0"
    assert "-f" in cmd
    fmt_idx = cmd.index("-f")
    assert cmd[fmt_idx + 1] == "matroska"
    mkv_cmd = next(c for c in captured_cmds if c[0] == "mkvmerge")
    assert "--timestamps" not in mkv_cmd


def test_constant_quality_ignores_fit_short_circuit(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "a.mp4"
    video.write_bytes(b"v" * 1024)
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="10M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_CONSTANT_QUALITY="28",
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": path.endswith(".mp4"), "duration": 60.0},
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 60.0)

    captured_cmds = []

    def fake_run(cmd, env=None, **kwargs):
        captured_cmds.append(cmd)
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
        elif cmd[0] == "mkvmerge":
            out_index = cmd.index("-o")
            Path(cmd[out_index + 1]).write_bytes(b"remuxed")

        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
            ],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {
                "title": "Original Title",
                "creation_time": "2024-09-28T15:42:11Z",
                "comment": "Container comment",
            },
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)

    script.main()

    assert (out_dir / "a.mkv").exists()
    assert any(cmd[0] == "ffmpeg" for cmd in captured_cmds)


def test_mkvmerge_sets_creation_date_and_attachments(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "a.mp4"
    video.write_bytes(b"v" * (2 * 1024 * 1024))
    desired_mtime = datetime(2020, 1, 1, 12, 30, tzinfo=timezone.utc).timestamp()
    os.utime(video, (desired_mtime, desired_mtime))
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="1M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_CONSTANT_QUALITY="32",
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {
            "is_video": path.endswith(".mp4"),
            "duration": 60.0 if path.endswith(".mp4") else None,
        },
    )

    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 60.0)
    monkeypatch.setattr(script, "find_start_timecode", lambda path: "00:00:00:00")
    monkeypatch.setattr(
        script,
        "get_container_creation_date",
        lambda path: "2024-09-28T15:42:11Z",
    )

    captured_mux_cmds: list[list[str]] = []
    captured_edit_cmds: list[list[str]] = []
    created_metadata_files: list[Path] = []

    def fake_run(cmd, env=None, **kwargs):
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if cmd[0] == "mkvmerge":
            captured_mux_cmds.append(cmd)
            out_index = cmd.index("-o")
            output_path = Path(cmd[out_index + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"remuxed")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if cmd[0] == "mkvpropedit":
            captured_edit_cmds.append(cmd)
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        return types.SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        metadata_path = dest / "legacy_metadata.json"
        metadata_path.write_text("{}\n", encoding="utf-8")
        created_metadata_files.append(metadata_path)
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
            ],
            "attachments": [],
            "metadata_path": metadata_path,
            "container_tags": {
                "title": "Original Title",
                "comment": "Container comment",
            },
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)

    script.main()

    output_video = out_dir / "a.mkv"
    assert output_video.exists()
    assert captured_mux_cmds
    mkv_cmd = captured_mux_cmds[0]
    assert mkv_cmd[1] == "-o"
    assert mkv_cmd[3] == "--disable-track-statistics-tags"
    assert "--timestamps" not in mkv_cmd
    assert not captured_edit_cmds
    assert "--date" in mkv_cmd
    date_index = mkv_cmd.index("--date")
    assert mkv_cmd[date_index + 1] == "2024-09-28T15:42:11Z"
    out_index = mkv_cmd.index("-o")
    assert out_index < date_index
    assert "--title" in mkv_cmd
    title_index = mkv_cmd.index("--title")
    assert mkv_cmd[title_index + 1] == "Original Title"
    assert "--global-tags" in mkv_cmd
    tags_index = mkv_cmd.index("--global-tags")
    tags_path = Path(mkv_cmd[tags_index + 1])
    assert tags_path.name.endswith(".container.tags.xml")
    out_stat = output_video.stat()
    assert pytest.approx(out_stat.st_mtime, rel=0, abs=1) == desired_mtime
    assert created_metadata_files
    metadata_file = created_metadata_files[0]
    metadata_attach_indices = [
        idx
        for idx, token in enumerate(mkv_cmd)
        if token == "--attach-file" and mkv_cmd[idx + 1] == str(metadata_file)
    ]
    assert metadata_attach_indices
    attach_index = metadata_attach_indices[0]
    assert mkv_cmd[attach_index - 6] == "--attachment-name"
    assert mkv_cmd[attach_index - 5] == metadata_file.name
    assert mkv_cmd[attach_index - 4] == "--attachment-mime-type"
    assert mkv_cmd[attach_index - 3] == "application/json"
    assert mkv_cmd[attach_index - 2] == "--attachment-description"
    assert mkv_cmd[attach_index - 1] == "Pre-re-encode metadata"


def test_dump_streams_data_sidecar_uses_container(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **_):
        calls.append(cmd)

        class Result:
            returncode = 0

        return Result()

    ffprobe_output = {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "tags": {},
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "data",
                "codec_name": "bin",
            }
        ],
    }

    monkeypatch.setattr(script.subprocess, "run", fake_run)
    monkeypatch.setattr(script, "ffprobe_json", lambda cmd: ffprobe_output)

    result = script._dump_streams_and_metadata("clip.mov", tmp_path, verbose=False)

    data_exports = [exp for exp in result["exports"] if exp["stype"] == "d"]
    assert len(data_exports) == 1
    assert data_exports[0]["path"].endswith(".mov")
    assert Path(data_exports[0]["path"]).name == "legacy_stream_d0.data.unknown.mov"

    assert calls == []


def test_mov_with_data_stream_outputs_mkv(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "clip.mov"
    video.write_bytes(b"v" * (2 * 1024 * 1024))
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="1M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_CONSTANT_QUALITY="32",
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {
            "is_video": path.endswith(".mov"),
            "duration": 60.0 if path.endswith(".mov") else None,
        },
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 60.0)
    monkeypatch.setattr(script, "find_start_timecode", lambda path: "01:02:03:04")
    monkeypatch.setattr(script.shutil, "which", lambda name: None)

    captured_cmds = []

    def fake_run(cmd, env=None, **kwargs):
        captured_cmds.append(cmd)
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
        elif cmd[0] == "mkvmerge":
            out_index = cmd.index("-o")
            output_path = Path(cmd[out_index + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"remuxed")

        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        data_sidecar = dest / "legacy_stream_d2.data.unknown.mov"
        data_sidecar.write_bytes(b"telemetry")
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
                {
                    "path": str(data_sidecar),
                    "stream": {"codec_type": "data", "codec_name": "bin", "index": 2},
                    "stype": "d",
                    "mkv_ok": False,
                },
            ],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {},
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
                {
                    "index": 2,
                    "stream": {"codec_type": "data", "codec_name": "bin", "index": 2},
                    "stype": "d",
                    "mkv_ok": False,
                    "spec": "d:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)

    script.main()

    bundles = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert bundles == []
    output_video = out_dir / "clip.mkv"
    assert output_video.exists()

    ffmpeg_cmds = [c for c in captured_cmds if c[0] == "ffmpeg"]
    assert len(ffmpeg_cmds) == 1
    encode_cmd = ffmpeg_cmds[0]
    assert "-ignore_unknown" in encode_cmd
    assert "-fflags" in encode_cmd
    ff_idx = encode_cmd.index("-fflags")
    assert encode_cmd[ff_idx + 1] == "+genpts"
    assert "-avoid_negative_ts" in encode_cmd
    ant_idx = encode_cmd.index("-avoid_negative_ts")
    assert encode_cmd[ant_idx + 1] == "make_zero"
    assert "-max_interleave_delta" in encode_cmd
    mid_idx = encode_cmd.index("-max_interleave_delta")
    assert encode_cmd[mid_idx + 1] == "0"
    assert "-fps_mode:v:0" in encode_cmd
    assert "0:v:0" in encode_cmd
    assert "0:a:0" in encode_cmd
    assert (
        "-c:v:0" in encode_cmd
        and encode_cmd[encode_cmd.index("-c:v:0") + 1] == "libsvtav1"
    )
    assert (
        "-c:a:0" in encode_cmd
        and encode_cmd[encode_cmd.index("-c:a:0") + 1] == "libopus"
    )
    assert (
        "-ar:a:0" in encode_cmd
        and encode_cmd[encode_cmd.index("-ar:a:0") + 1] == "48000"
    )
    assert "-f" in encode_cmd and encode_cmd[encode_cmd.index("-f") + 1] == "matroska"
    for flag, value in [
        ("-map_metadata", "0"),
        ("-map_metadata:s:v", "0:s:v"),
        ("-map_metadata:s:a", "0:s:a"),
    ]:
        assert flag in encode_cmd
        assert value in encode_cmd
    for flag in ["-map_metadata:s:s", "-map_metadata:s:d", "-map_metadata:s:t"]:
        assert flag not in encode_cmd
    mkv_cmd = next(c for c in captured_cmds if c[0] == "mkvmerge")
    assert "--timestamps" not in mkv_cmd


def test_low_bitrate_audio_stream_copied(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "clip.mp4"
    video.write_bytes(b"v" * (80 * 1024 * 1024))
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="60M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": path.endswith(".mp4"), "duration": 10.0},
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 10.0)

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, env=None, **kwargs):
        captured_cmds.append(cmd)
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
        elif cmd[0] == "mkvmerge":
            out_index = cmd.index("-o")
            Path(cmd[out_index + 1]).write_bytes(b"remuxed")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
            ],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {},
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)
    monkeypatch.setattr(
        script,
        "_probe_stream_infos_only",
        lambda path: [
            {
                "index": 0,
                "stream": {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "index": 0,
                    "duration": "10",
                },
                "stype": "v",
                "mkv_ok": True,
                "spec": "v:0",
            },
            {
                "index": 1,
                "stream": {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "index": 1,
                    "bit_rate": "64000",
                    "duration": "10",
                },
                "stype": "a",
                "mkv_ok": True,
                "spec": "a:0",
            },
        ],
    )

    script.main()

    encode_cmd = next(c for c in captured_cmds if c[0] == "ffmpeg")
    assert (
        "-c:a:0" in encode_cmd and encode_cmd[encode_cmd.index("-c:a:0") + 1] == "copy"
    )
    assert "libopus" not in encode_cmd


def test_low_bitrate_video_stream_copied(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "clip.mp4"
    video.write_bytes(b"v" * (80 * 1024 * 1024))
    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="60M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {"is_video": path.endswith(".mp4"), "duration": 10.0},
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 10.0)

    captured_cmds: list[list[str]] = []

    def fake_run(cmd, env=None, **kwargs):
        captured_cmds.append(cmd)
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
        elif cmd[0] == "mkvmerge":
            out_index = cmd.index("-o")
            Path(cmd[out_index + 1]).write_bytes(b"remuxed")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
            ],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {},
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)
    monkeypatch.setattr(
        script,
        "_probe_stream_infos_only",
        lambda path: [
            {
                "index": 0,
                "stream": {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "index": 0,
                    "bit_rate": "10000000",
                    "duration": "10",
                },
                "stype": "v",
                "mkv_ok": True,
                "spec": "v:0",
            },
            {
                "index": 1,
                "stream": {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "index": 1,
                    "duration": "10",
                },
                "stype": "a",
                "mkv_ok": True,
                "spec": "a:0",
            },
        ],
    )

    script.main()

    encode_cmd = next(c for c in captured_cmds if c[0] == "ffmpeg")
    assert (
        "-c:v:0" in encode_cmd and encode_cmd[encode_cmd.index("-c:v:0") + 1] == "copy"
    )
    assert "libsvtav1" not in encode_cmd


def test_sidecar_files_are_renamed(monkeypatch, tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    video = src_dir / "a.mp4"
    video.write_bytes(b"v" * (2 * 1024 * 1024))
    sidecar1 = src_dir / "a.mp4.srt"
    sidecar1.write_text("subs")
    sidecar2 = src_dir / "a.mp4.nfo"
    sidecar2.write_text("info")
    other_asset = src_dir / "other.txt"
    other_asset.write_text("other")

    out_dir = tmp_path / "out"
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()

    set_env(
        monkeypatch,
        VCRUNCH_INPUTS=str(src_dir),
        VCRUNCH_TARGET_SIZE="1M",
        VCRUNCH_OUTPUT_DIR=str(out_dir),
        VCRUNCH_STAGE_DIR=str(stage_dir),
        VCRUNCH_CONSTANT_QUALITY="30",
    )
    monkeypatch.setattr(
        script,
        "probe_media_info",
        lambda path: {
            "is_video": path.endswith(".mp4"),
            "duration": 60.0 if path.endswith(".mp4") else None,
        },
    )
    monkeypatch.setattr(script, "ffprobe_duration", lambda path: 60.0)

    def fake_run(cmd, env=None, **kwargs):
        if cmd[0] == "ffmpeg":
            for idx, token in enumerate(cmd):
                if token == "-f" and idx + 2 < len(cmd):
                    output_path = Path(cmd[idx + 2])
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"encoded")
        elif cmd[0] == "mkvmerge":
            out_index = cmd.index("-o")
            output_path = Path(cmd[out_index + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"remuxed")

        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(script.subprocess, "run", fake_run)

    def fake_dump(src, dest, verbose, **kwargs):
        dest.mkdir(parents=True, exist_ok=True)
        video_sidecar = dest / "video.stream.h264.mkv"
        video_sidecar.write_bytes(b"origvideo")
        audio_sidecar = dest / "audio.stream.aac.mkv"
        audio_sidecar.write_bytes(b"origaudio")
        return {
            "exports": [
                {
                    "path": str(video_sidecar),
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                },
                {
                    "path": str(audio_sidecar),
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                },
            ],
            "attachments": [],
            "metadata_path": None,
            "container_tags": {},
            "stream_infos": [
                {
                    "index": 0,
                    "stream": {"codec_type": "video", "codec_name": "h264", "index": 0},
                    "stype": "v",
                    "mkv_ok": True,
                    "spec": "v:0",
                },
                {
                    "index": 1,
                    "stream": {"codec_type": "audio", "codec_name": "aac", "index": 1},
                    "stype": "a",
                    "mkv_ok": True,
                    "spec": "a:0",
                },
            ],
        }

    monkeypatch.setattr(script, "_dump_streams_and_metadata", fake_dump)
    monkeypatch.setattr(
        script, "_pick_real_video_stream_index", lambda path: (0, "v:0")
    )
    monkeypatch.setattr(script, "is_valid_media", lambda path: True)

    script.main()

    bundles = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert bundles == []

    assert (out_dir / "a.mkv").exists()
    assert (out_dir / "a.mkv.srt").exists()
    assert (out_dir / "a.mkv.nfo").exists()
    assert (out_dir / "other.txt").exists()
    assert not (out_dir / "a.mp4.srt").exists()
    assert not (out_dir / "a.mp4.nfo").exists()
