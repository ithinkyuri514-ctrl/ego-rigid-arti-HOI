"""Small IO and process helpers shared by rigid reconstruction stages."""

from __future__ import annotations

import csv
import json
import socket
import subprocess
from pathlib import Path
from typing import Any, Iterable


VALID_STAGE_STATUS = {
    "pending",
    "running",
    "completed",
    "failed",
    "needs_revision",
    "skipped",
}


def jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_stage_state(
    state_path: Path,
    stage_name: str,
    status: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    notes: str | None = None,
) -> None:
    if status not in VALID_STAGE_STATUS:
        raise ValueError(f"Invalid stage status: {status}")
    if not state_path.exists():
        return
    state = read_json(state_path)
    matches = [item for item in state.get("stages", []) if item.get("stage") == stage_name]
    if len(matches) != 1:
        raise KeyError(f"Expected one state record for {stage_name}, found {len(matches)}")
    record = matches[0]
    record["status"] = status
    if inputs is not None:
        record["inputs"] = inputs
    if outputs is not None:
        record["outputs"] = outputs
    if notes is not None:
        record["notes"] = notes
    write_json(state_path, state)


def video_metadata(path: Path) -> dict[str, float | int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    result = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    capture.release()
    return result


def validate_matching_videos(rgb_video: Path, mask_video: Path, fps_tolerance: float = 1e-3) -> dict[str, Any]:
    rgb = video_metadata(rgb_video)
    mask = video_metadata(mask_video)
    errors = []
    for key in ("width", "height", "frame_count"):
        if rgb[key] != mask[key]:
            errors.append(f"{key}: rgb={rgb[key]} mask={mask[key]}")
    if abs(float(rgb["fps"]) - float(mask["fps"])) > fps_tolerance:
        errors.append(f"fps: rgb={rgb['fps']} mask={mask['fps']}")
    if errors:
        raise ValueError("RGB/mask video mismatch: " + "; ".join(errors))
    return {"rgb": rgb, "mask": mask}


def run_checked(command: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> None:
    printable = " ".join(subprocess.list2cmdline([item]) for item in command)
    print(f"$ {printable}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def print_server_addresses(host: str, port: int, title: str) -> None:
    print("", flush=True)
    print(f"{title} is ready:", flush=True)
    print(f"  local:   http://127.0.0.1:{port}", flush=True)
    print(f"  local:   http://localhost:{port}", flush=True)
    try:
        addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        addresses = []
    if host not in {"127.0.0.1", "localhost"}:
        for address in addresses:
            if not address.startswith("127."):
                print(f"  network: http://{address}:{port}", flush=True)
    print(
        f"Remote machine: ssh -L {port}:127.0.0.1:{port} <server>",
        flush=True,
    )
    print("", flush=True)
