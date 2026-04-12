#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a LeRobot v3 dataset into the legacy layout expected by RLinf."""

from __future__ import annotations

import argparse
import json
import shutil as _shutil
import shutil
import subprocess
from pathlib import Path
from typing import Any


LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = (
    "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot v3 dataset to the legacy RLinf-compatible layout."
    )
    parser.add_argument(
        "--src-dataset",
        type=Path,
        required=True,
        help="Path to the source LeRobot v3 dataset.",
    )
    parser.add_argument(
        "--dst-dataset",
        type=Path,
        required=True,
        help="Path where the converted legacy dataset will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional limit for converting only the first N episodes.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["copy", "reencode"],
        default="copy",
        help="How to cut episode videos. 'copy' is faster, 'reencode' is safer.",
    )
    return parser


def _require_pyarrow():
    try:
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyarrow. Install it in the current environment."
        ) from exc
    return pq, pc


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _set_nested(mapping: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def _resolve_source_layout(src_dataset: Path) -> tuple[Path, Path, Path]:
    meta_dir = src_dataset / "meta"
    episodes_dir = meta_dir / "episodes"
    tasks_parquet = meta_dir / "tasks.parquet"
    info_json = meta_dir / "info.json"

    if not src_dataset.exists():
        raise SystemExit(f"Source dataset does not exist: {src_dataset}")
    if not info_json.exists():
        raise SystemExit(f"Missing info.json: {info_json}")
    if not tasks_parquet.exists():
        raise SystemExit(f"Missing tasks.parquet: {tasks_parquet}")
    if not episodes_dir.exists():
        raise SystemExit(f"Missing meta/episodes directory: {episodes_dir}")

    return meta_dir, episodes_dir, tasks_parquet


def _load_episode_rows(episodes_dir: Path) -> list[dict[str, Any]]:
    pq, _ = _require_pyarrow()
    rows: list[dict[str, Any]] = []
    parquet_paths = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise SystemExit(f"No episode parquet files found in {episodes_dir}")

    for parquet_path in parquet_paths:
        table = pq.read_table(parquet_path)
        for row in table.to_pylist():
            rows.append(_json_safe(row))

    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def _load_task_rows(tasks_parquet: Path) -> list[dict[str, Any]]:
    pq, _ = _require_pyarrow()
    table = pq.read_table(tasks_parquet)
    data = table.to_pydict()

    if "task" in data:
        tasks = data["task"]
    elif "__index_level_0__" in data:
        tasks = data["__index_level_0__"]
    else:
        raise SystemExit(
            f"Could not infer task strings from {tasks_parquet}. Columns: {list(data)}"
        )

    return [
        {"task_index": int(task_index), "task": str(task)}
        for task_index, task in zip(data["task_index"], tasks)
    ]


def _build_episodes_jsonl_rows(
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in episode_rows:
        rows.append({k: v for k, v in row.items() if not k.startswith("stats/")})
    return rows


def _build_episodes_stats_jsonl_rows(
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in episode_rows:
        stats: dict[str, Any] = {}
        for key, value in row.items():
            if key.startswith("stats/"):
                _set_nested(stats, key.split("/")[1:], value)
        rows.append({"episode_index": int(row["episode_index"]), "stats": stats})
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def _episode_chunk(info: dict[str, Any], episode_index: int) -> int:
    chunk_size = int(info.get("chunks_size", 1000))
    return episode_index // chunk_size


def _copy_or_rewrite_info(info: dict[str, Any], video_keys: list[str]) -> dict[str, Any]:
    info = dict(info)
    info["data_path"] = LEGACY_DATA_PATH
    info["video_path"] = LEGACY_VIDEO_PATH if video_keys else None
    features = dict(info.get("features", {}))
    if "action" in features and "actions" not in features:
        features["actions"] = dict(features["action"])
    info["features"] = features
    return info


def _convert_episode_parquets(
    src_dataset: Path,
    dst_dataset: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
) -> None:
    pq, pc = _require_pyarrow()
    source_tables: dict[tuple[int, int], Any] = {}

    grouped_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in episode_rows:
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        grouped_rows.setdefault(key, []).append(row)

    for key, rows in grouped_rows.items():
        chunk_index, file_index = key
        src_parquet = src_dataset / f"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
        if not src_parquet.exists():
            raise SystemExit(f"Missing source parquet: {src_parquet}")
        source_tables[key] = pq.read_table(src_parquet)

        for row in rows:
            ep_index = int(row["episode_index"])
            ep_chunk = _episode_chunk(info, ep_index)
            dst_parquet = (
                dst_dataset
                / LEGACY_DATA_PATH.format(
                    episode_chunk=ep_chunk,
                    episode_index=ep_index,
                )
            )
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            filtered = source_tables[key].filter(
                pc.equal(source_tables[key]["episode_index"], ep_index)
            )
            if "action" in filtered.column_names and "actions" not in filtered.column_names:
                filtered = filtered.append_column("actions", filtered["action"])
            # Strip newer Hugging Face feature metadata so older RLinf/LeRobot
            # stacks don't try to deserialize unsupported feature types such as
            # "List" from the parquet schema metadata.
            filtered = filtered.replace_schema_metadata(None)
            pq.write_table(filtered, dst_parquet)


def _ffmpeg_cut(src: Path, dst: Path, start_s: float, end_s: float, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _shutil.which("ffmpeg") is None:
        _pyav_cut(src, dst, start_s, end_s)
        return

    if mode == "copy":
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.6f}",
            "-to",
            f"{end_s:.6f}",
            "-i",
            str(src),
            "-avoid_negative_ts",
            "make_zero",
            "-c",
            "copy",
            str(dst),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return
        mode = "reencode"

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.6f}",
        "-to",
        f"{end_s:.6f}",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            f"ffmpeg failed while cutting {src} -> {dst}\n{result.stderr.strip()}"
        )


def _pyav_cut(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    try:
        import av
        from fractions import Fraction
    except ImportError as exc:
        raise SystemExit(
            "ffmpeg executable was not found, and PyAV is also unavailable. "
            "Install ffmpeg or `pip install av`."
        ) from exc

    in_container = av.open(str(src))
    video_stream = next((s for s in in_container.streams if s.type == "video"), None)
    if video_stream is None:
        in_container.close()
        raise SystemExit(f"No video stream found in {src}")

    out_container = av.open(str(dst), mode="w")
    out_stream = out_container.add_stream("libx264", rate=video_stream.average_rate)
    out_stream.width = video_stream.codec_context.width
    out_stream.height = video_stream.codec_context.height
    out_stream.pix_fmt = "yuv420p"

    out_tb = Fraction(1, int(video_stream.average_rate)) if video_stream.average_rate else Fraction(1, 25)
    pts = 0

    try:
        for frame in in_container.decode(video=0):
            if frame.time is None:
                continue
            if frame.time + 1e-6 < start_s:
                continue
            if frame.time > end_s + 1e-6:
                break

            new_frame = frame.reformat(
                width=out_stream.width,
                height=out_stream.height,
                format="yuv420p",
            )
            new_frame.pts = pts
            new_frame.time_base = out_tb
            pts += 1

            for packet in out_stream.encode(new_frame):
                out_container.mux(packet)

        for packet in out_stream.encode():
            out_container.mux(packet)
    finally:
        out_container.close()
        in_container.close()


def _convert_videos(
    src_dataset: Path,
    dst_dataset: Path,
    info: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    video_keys: list[str],
    video_mode: str,
) -> None:
    for row in episode_rows:
        ep_index = int(row["episode_index"])
        ep_chunk = _episode_chunk(info, ep_index)
        for video_key in video_keys:
            chunk_index = int(row[f"videos/{video_key}/chunk_index"])
            file_index = int(row[f"videos/{video_key}/file_index"])
            start_s = float(row[f"videos/{video_key}/from_timestamp"])
            end_s = float(row[f"videos/{video_key}/to_timestamp"])
            src_video = src_dataset / f"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
            if not src_video.exists():
                raise SystemExit(f"Missing source video: {src_video}")
            dst_video = (
                dst_dataset
                / LEGACY_VIDEO_PATH.format(
                    video_key=video_key,
                    episode_chunk=ep_chunk,
                    episode_index=ep_index,
                )
            )
            _ffmpeg_cut(src_video, dst_video, start_s, end_s, video_mode)


def _copy_stats_json(src_meta: Path, dst_meta: Path) -> None:
    src_stats = src_meta / "stats.json"
    if src_stats.exists():
        shutil.copy2(src_stats, dst_meta / "stats.json")


def main(
    src_dataset: Path,
    dst_dataset: Path,
    overwrite: bool = False,
    max_episodes: int | None = None,
    video_mode: str = "copy",
) -> None:
    src_dataset = src_dataset.expanduser().resolve()
    dst_dataset = dst_dataset.expanduser().resolve()
    src_meta, episodes_dir, tasks_parquet = _resolve_source_layout(src_dataset)

    if dst_dataset.exists():
        if not overwrite:
            raise SystemExit(
                f"Destination already exists: {dst_dataset}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_dataset)

    info = _load_json(src_meta / "info.json")
    episode_rows = _load_episode_rows(episodes_dir)
    if max_episodes is not None:
        episode_rows = episode_rows[:max_episodes]

    if not episode_rows:
        raise SystemExit("No episode metadata rows found to convert.")

    video_keys = [
        key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"
    ]

    dst_meta = dst_dataset / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    _write_json(dst_meta / "info.json", _copy_or_rewrite_info(info, video_keys))
    _copy_stats_json(src_meta, dst_meta)
    _write_jsonl(dst_meta / "tasks.jsonl", _load_task_rows(tasks_parquet))
    _write_jsonl(dst_meta / "episodes.jsonl", _build_episodes_jsonl_rows(episode_rows))
    _write_jsonl(
        dst_meta / "episodes_stats.jsonl",
        _build_episodes_stats_jsonl_rows(episode_rows),
    )

    _convert_episode_parquets(src_dataset, dst_dataset, info, episode_rows)
    if video_keys:
        _convert_videos(src_dataset, dst_dataset, info, episode_rows, video_keys, video_mode)

    print(f"Converted {len(episode_rows)} episodes into legacy layout at {dst_dataset}")


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(
        src_dataset=args.src_dataset,
        dst_dataset=args.dst_dataset,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        video_mode=args.video_mode,
    )
