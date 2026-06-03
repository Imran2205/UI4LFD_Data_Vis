#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Iterable


DEFAULT_VIDEO = Path(
    "demo_recording/Pick_up_the_green_block_and_put_that_in_the_teal_box/"
    "episode_000002/wrist_rgb.mp4"
)

GRIPPER_CHANNEL_INDEX = 7
OBS_KEYS = ("observation.state", "observation.state_raw_v5", "observation.qpos_actual")
ACTION_KEYS = ("action_command", "action_command_raw_v5", "action")
DEFAULT_FORCE_SCALE = 0.1


def import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: cv2. Install it with:\n"
            "  python3 -m pip install --user opencv-contrib-python numpy\n"
        ) from exc

    if not hasattr(cv2, "aruco"):
        raise SystemExit(
            "Your OpenCV build does not include cv2.aruco. Install contrib OpenCV:\n"
            "  python3 -m pip install --user opencv-contrib-python numpy\n"
        )
    return cv2


def finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def first_present(record: dict, keys: Iterable[str]) -> list | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            return value
    return None


def find_jsonl(video_path: Path) -> Path:
    jsonl_files = sorted(video_path.parent.glob("*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"No JSONL file found next to {video_path}")
    if len(jsonl_files) > 1:
        print(f"Using JSONL file: {jsonl_files[0]}")
    return jsonl_files[0]


def action_is_stable(actions: list[float], index: int, stable_steps: int, epsilon: float) -> bool:
    if stable_steps <= 1:
        return True
    if index < stable_steps - 1:
        return False
    current_action = actions[index]
    start = index - stable_steps + 1
    return all(abs(actions[i] - current_action) <= epsilon for i in range(start, index))


def read_gripper_series(
    jsonl_path: Path,
    stable_steps: int,
    action_epsilon: float,
) -> list[dict[str, float]]:
    series: list[dict[str, float]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line, parse_constant=lambda _constant: None)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Could not parse {jsonl_path}:{line_number}: {exc}") from exc

            obs = first_present(record, OBS_KEYS)
            action = first_present(record, ACTION_KEYS)
            if not obs or not action:
                continue
            if len(obs) <= GRIPPER_CHANNEL_INDEX or len(action) <= GRIPPER_CHANNEL_INDEX:
                continue

            obs_value = finite_float(obs[GRIPPER_CHANNEL_INDEX])
            action_value = finite_float(action[GRIPPER_CHANNEL_INDEX])
            if obs_value is None or action_value is None:
                continue

            series.append(
                {
                    "obs": obs_value,
                    "action": action_value,
                    "diff": action_value - obs_value,
                    "force": 0.0,
                }
            )

    if not series:
        raise SystemExit(f"No usable gripper samples found in {jsonl_path}")

    actions = [sample["action"] for sample in series]
    for index, sample in enumerate(series):
        if sample["action"] >= sample["obs"]:
            sample["force"] = 0.0
            continue
        if not action_is_stable(actions, index, stable_steps, action_epsilon):
            sample["force"] = 0.0
            continue
        sample["force"] = sample["obs"] - sample["action"]
    return series


def robust_force_scale(series: list[dict[str, float]]) -> float:
    forces = sorted(sample["force"] for sample in series if sample["force"] > 0)
    if not forces:
        return 1.0
    index = min(len(forces) - 1, max(0, int(0.95 * (len(forces) - 1))))
    return max(forces[index], 1e-6)


def create_aruco_detector(cv2, dictionary_name: str):
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        valid = sorted(name for name in dir(cv2.aruco) if name.startswith("DICT_"))
        raise SystemExit(
            f"Unknown ArUco dictionary {dictionary_name!r}. Examples: {', '.join(valid[:8])}"
        )

    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)

    params = cv2.aruco.DetectorParameters_create()
    return lambda gray: cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def marker_geometry(corners) -> dict[str, int | str]:
    points = corners.reshape(-1, 2)
    return {
        "center_x": int(points[:, 0].mean()),
        "center_y": int(points[:, 1].mean()),
        "left": int(points[:, 0].min()),
        "right": int(points[:, 0].max()),
        "top": int(points[:, 1].min()),
        "bottom": int(points[:, 1].max()),
        "side": "",
    }


def select_marker_geometries(
    corners,
    ids,
    frame_width: int,
    frame_height: int,
    marker_ids: set[int] | None,
    previous_markers: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    if ids is None or len(ids) == 0:
        return previous_markers

    candidates: list[tuple[int, dict[str, int | str]]] = []
    flat_ids = ids.flatten().tolist()
    for marker_id, marker_corners in zip(flat_ids, corners):
        if marker_ids is not None and marker_id not in marker_ids:
            continue
        candidates.append((marker_id, marker_geometry(marker_corners)))

    if not candidates:
        return previous_markers

    if marker_ids is None and len(candidates) > 2:
        image_center = (frame_width / 2.0, frame_height / 2.0)
        candidates.sort(
            key=lambda item: (
                (int(item[1]["center_x"]) - image_center[0]) ** 2
                + (int(item[1]["center_y"]) - image_center[1]) ** 2
            )
        )
    else:
        candidates.sort(key=lambda item: item[0])

    selected_markers = [geometry for _marker_id, geometry in candidates[:2]]
    selected_markers.sort(key=lambda marker: int(marker["center_x"]))
    if selected_markers:
        selected_markers[0]["side"] = "left"
    if len(selected_markers) > 1:
        selected_markers[1]["side"] = "right"
    return selected_markers


def color_for_level(level: float) -> tuple[int, int, int]:
    level = max(0.0, min(1.0, level))
    if level < 0.5:
        ratio = level / 0.5
        return 40, int(210 - 70 * ratio), int(80 + 175 * ratio)
    ratio = (level - 0.5) / 0.5
    return 40, int(140 - 90 * ratio), 255


def draw_text_with_shadow(cv2, frame, text: str, origin: tuple[int, int], scale: float = 0.42):
    x, y = origin
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (245, 245, 245), 1)


def draw_right_aligned_text_with_shadow(
    cv2,
    frame,
    text: str,
    right_x: int,
    baseline_y: int,
    scale: float = 0.36,
):
    text_size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    draw_text_with_shadow(cv2, frame, text, (right_x - text_size[0], baseline_y), scale=scale)


def draw_bullet_bar(
    cv2,
    frame,
    marker: dict[str, int | str],
    sample: dict[str, float],
    force_scale: float,
):
    frame_height, frame_width = frame.shape[:2]
    bar_width = 92
    bar_height = 12
    pad = 8
    box_width = 132
    box_height = 54
    gap = 6

    marker_left = int(marker["left"])
    marker_right = int(marker["right"])
    marker_top = int(marker["top"])
    side = str(marker.get("side", ""))

    if side == "left":
        x = marker_right - box_width
        connector_x = marker_right
    elif side == "right":
        x = marker_left
        connector_x = marker_left
    else:
        x = int(marker["center_x"]) - box_width // 2
        connector_x = int(marker["center_x"])
    x = max(4, min(frame_width - box_width - 4, x))

    y = marker_top - box_height - gap
    y = max(4, min(frame_height - box_height - 4, y))

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (235, 235, 235), 1)
    connector_x = max(x, min(x + box_width, connector_x))
    cv2.line(frame, (connector_x, marker_top), (connector_x, y + box_height), (235, 235, 235), 1)

    level = min(1.0, sample["force"] / force_scale)
    fill_width = int(round(bar_width * level))
    bar_x = x + pad
    bar_y = y + 22
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (65, 65, 65), -1)
    if fill_width > 0:
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            color_for_level(level),
            -1,
        )
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (235, 235, 235), 1)

    draw_text_with_shadow(cv2, frame, f"contact {sample['force']:.4f}", (x + pad, y + 14))
    draw_text_with_shadow(cv2, frame, "0.000", (bar_x, y + 47), scale=0.36)
    draw_right_aligned_text_with_shadow(cv2, frame, "0.100", bar_x + bar_width, y + 47)


def make_output_path(video_path: Path, suffix: str) -> Path:
    return video_path.with_name(f"{video_path.stem}{suffix}{video_path.suffix}")


def render_video(args: argparse.Namespace, video_path: Path, output_path: Path | None = None) -> Path:
    cv2 = import_cv2()

    video_path = video_path.expanduser().resolve()
    jsonl_path = Path(args.jsonl).expanduser().resolve() if args.jsonl else find_jsonl(video_path)
    if output_path is None:
        output_path = Path(args.output).expanduser().resolve() if args.output else make_output_path(video_path, args.suffix)
    else:
        output_path = output_path.expanduser().resolve()

    series = read_gripper_series(jsonl_path, args.stable_steps, args.action_epsilon)
    force_scale = args.max_diff if args.max_diff and args.max_diff > 0 else DEFAULT_FORCE_SCALE
    marker_ids = set(args.marker_ids) if args.marker_ids else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_width <= 0 or frame_height <= 0:
        raise SystemExit(f"Could not read video dimensions: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {output_path}")

    detect_markers = create_aruco_detector(cv2, args.aruco_dictionary)
    previous_markers: list[dict[str, int | str]] = []
    frame_index = 0
    frames_with_marker_centers = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_count > 1:
            sample_index = round(frame_index * (len(series) - 1) / (frame_count - 1))
        else:
            sample_index = min(frame_index, len(series) - 1)
        sample = series[max(0, min(len(series) - 1, sample_index))]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _rejected = detect_markers(gray)

        markers = select_marker_geometries(
            corners,
            ids,
            frame_width,
            frame_height,
            marker_ids,
            previous_markers,
        )
        previous_markers = markers
        if markers:
            frames_with_marker_centers += 1
        for marker in markers:
            draw_bullet_bar(cv2, frame, marker, sample, force_scale)

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    print(f"Wrote {output_path}")
    print(f"Frames processed: {frame_index}")
    print(f"JSONL samples: {len(series)}")
    print(f"Frames with marker-anchored bars: {frames_with_marker_centers}")
    print(f"Samples with nonzero contact force: {sum(1 for sample in series if sample['force'] > 0)}")
    print(f"Force scale: {force_scale:.6f}")
    return output_path


def replace_input_with_overlay(video_path: Path, rendered_path: Path) -> None:
    backup_path = video_path.with_name("wrist_rgb_.mp4")
    if backup_path.exists():
        raise SystemExit(f"Refusing to overwrite existing backup: {backup_path}")
    video_path.replace(backup_path)
    rendered_path.replace(video_path)
    print(f"Renamed source to {backup_path}")
    print(f"Renamed overlay to {video_path}")


def render_and_maybe_replace(args: argparse.Namespace, video_path: Path) -> Path:
    if args.replace_input:
        if args.output:
            raise SystemExit("--replace-input cannot be combined with --output")
        temp_path = video_path.with_name("wrist_rgb.__overlay_tmp__.mp4")
        if temp_path.exists():
            temp_path.unlink()
        rendered_path = render_video(args, video_path, temp_path)
        replace_input_with_overlay(video_path, rendered_path)
        return video_path
    return render_video(args, video_path)


def process_video(args: argparse.Namespace) -> list[Path]:
    if args.batch_glob:
        if args.jsonl:
            raise SystemExit("--jsonl can only be used for a single video")
        video_paths = [Path(path) for path in sorted(glob.glob(args.batch_glob))]
        if not video_paths:
            raise SystemExit(f"No videos matched: {args.batch_glob}")
        if args.replace_input:
            existing_backups = [
                path.with_name("wrist_rgb_.mp4")
                for path in video_paths
                if path.with_name("wrist_rgb_.mp4").exists()
            ]
            if existing_backups:
                raise SystemExit(
                    "Refusing batch replace because backups already exist:\n"
                    + "\n".join(str(path) for path in existing_backups)
                )

        outputs: list[Path] = []
        for index, video_path in enumerate(video_paths, start=1):
            print(f"\n[{index}/{len(video_paths)}] {video_path}")
            outputs.append(render_and_maybe_replace(args, video_path))
        return outputs

    return [render_and_maybe_replace(args, Path(args.video))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay gripper contact-force bars on wrist RGB video using the UI's "
            "gripper action/observation channel."
        )
    )
    parser.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO), help="Input wrist RGB MP4.")
    parser.add_argument("--jsonl", help="Episode JSONL. Defaults to the first JSONL next to the video.")
    parser.add_argument("--output", help="Output MP4 path. Defaults to input stem plus suffix.")
    parser.add_argument("--suffix", default="_gripper_contact_force", help="Output filename suffix.")
    parser.add_argument("--batch-glob", help="Render every video matching this glob pattern.")
    parser.add_argument(
        "--replace-input",
        action="store_true",
        help="Rename source wrist_rgb.mp4 to wrist_rgb_.mp4 and put the overlay at wrist_rgb.mp4.",
    )
    parser.add_argument(
        "--marker-ids",
        nargs="+",
        type=int,
        help="Specific ArUco IDs on the two gripper jaws. Defaults to the two detections nearest image center.",
    )
    parser.add_argument(
        "--aruco-dictionary",
        default="DICT_6X6_250",
        help="OpenCV ArUco dictionary name, for example DICT_6X6_250 or DICT_4X4_50.",
    )
    parser.add_argument(
        "--max-diff",
        type=float,
        help="Difference value that fills the bullet bar. Defaults to fixed 0.1 scale.",
    )
    parser.add_argument(
        "--stable-steps",
        type=int,
        default=3,
        help="Require this many consecutive equal gripper action samples before showing contact force.",
    )
    parser.add_argument(
        "--action-epsilon",
        type=float,
        default=1e-6,
        help="Tolerance for deciding whether gripper action is unchanged across stable steps.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
