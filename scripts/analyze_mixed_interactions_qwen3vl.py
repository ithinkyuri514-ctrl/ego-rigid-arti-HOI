#!/usr/bin/env python3
"""Analyze mixed interactions with a Qwen-VL model through the DashScope API."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import update_stage_state, write_json  # noqa: E402


PROMPT = """Analyze this egocentric SELECTED_EYE video. Return strict JSON only.

The video contains both an articulated-object interaction and a rigid-object interaction.
Split the global timeline into semantic event segments, identify every object physically
manipulated by a hand, and classify each manipulated object as articulated or rigid.

Mandatory timeline completeness pass:
- Inspect the entire supplied timeline before producing JSON. Include every distinct hand-driven
  object motion, including short rigid pick/place/move events that occur before or after a longer
  articulated event.
- The detailed articulated completion analysis must not cause later rigid interactions or their
  object records to be omitted. Every event object_id must have exactly one matching object record.

Classification rule:
- "rigid" means the manipulated object moves as one rigid body, such as picking or placing a cup.
- "articulated" means the object has a moving part connected by a joint, even if its main body
  remains stationary. Opening or closing a microwave/oven/cabinet door is articulated and the
  appliance must be labeled articulated, with the door as a moving part and a revolute joint.
- Classify from the physical action, not from whether the appliance body itself moves.

Important geometry contract:
- Event segments are for semantic routing only.
- Every object will be reconstructed from global frame 0, never from an event-local first frame.
- For every manipulated object, provide its tight silhouette bbox in GLOBAL FRAME 0.
- Do not list the hand, counter, wall, background, or an object that is merely nearby.
- Keep one object record per physical object even if it appears in multiple events.
- Use only the labeled global frame indices shown in the input.
- For a rigid event, put start/end on the closest labeled frames bracketing the visible rigid-body
  manipulation. Do not extend it into a following unrelated interaction.

Articulated tracking boundary contract:
- For every articulated event, start_frame and end_frame define the FULL moving-part tracking
  interval, not merely the hand-contact interval and not a coarse semantic action interval.
- start_frame must be the closest labeled frame at or before the last stable initial articulation
  state, before the moving part begins its state transition. Include a pre-motion frame whenever
  one is available.
- end_frame must be the earliest labeled frame where the moving part reaches the intended terminal
  configuration. It is the last motion/tracking frame; do not extend it only to include a later
  confirmation observation.
- Do not end an articulated event because the hand stops touching, starts to release, occludes the
  part, or because the action merely appears almost complete. Track the part until its geometric
  state transition is complete.
- For a closing operation, completion requires visual evidence that the moving part has reached
  the closed configuration: the residual gap/angle has stopped decreasing and a later labeled
  frame confirms that it remains closed. For an opening operation, require the intended open
  configuration followed by a stable confirmation frame. Apply the same state-transition and
  stability rule to revolute, prismatic, and other articulated joints.
- Explicitly compare at least three labeled frames near the apparent end of an articulated
  transition. A nearly closed/open/extended/retracted frame is not terminal if a later frame has
  a smaller residual gap, a more complete angle/displacement, or continued hand-applied motion.
  For closing, any visible residual opening gap means partially_closed rather than closed.
- terminal_reached_frame must be the earliest labeled frame at the final observed articulation
  state and must equal end_frame. confirmation_frame must be a later labeled frame showing the
  same state, but it is evidence only and lies outside the motion/tracking interval.
- The intended terminal configuration is the demonstrated operation's completed state; do not
  assume that every operation reaches a mechanical joint limit.
- If the video or sampled evidence does not show the terminal state being reached and confirmed,
  set terminal_reached=false. Do not hallucinate completion. Extend end_frame through the last
  labeled frame that still contains relevant part motion or terminal-state evidence, and explain
  the uncertainty.
- completion_evidence_frames must include the first labeled frame where the terminal state is
  observed and a distinct later labeled confirmation frame. Evidence of hand release alone is not
  completion evidence.

JSON schema:
{
  "scene_summary": "short description",
  "events": [
    {
      "event_id": "event_000",
      "interaction_class": "articulated|rigid",
      "object_id": "stable object id",
      "action": "open|close|pick|place|move|rotate|touch|unknown",
      "description": "what the hand does",
      "hand_side": "left|right|both|unknown",
      "start_frame": 0,
      "end_frame": 0,
      "evidence_frames": [0],
      "articulation_completion": {
        "initial_state": "open|closed|partially_open|partially_closed|extended|retracted|unknown",
        "target_terminal_state": "open|closed|extended|retracted|other|unknown",
        "terminal_state_observed": "short visual description",
        "terminal_reached": true,
        "first_motion_frame": 0,
        "terminal_reached_frame": 0,
        "confirmation_frame": 0,
        "completion_evidence_frames": [0, 1],
        "boundary_reason": "why start/end cover the state transition through its first terminal frame"
      },
      "confidence": 0.0
    }
  ],
  "objects": [
    {
      "object_id": "microwave|cup|other stable id",
      "name_en": "canonical English name",
      "name_zh": "中文名",
      "object_class": "articulated|rigid",
      "manipulation_evidence_frames": [0],
      "global_frame0": {
        "visible": true,
        "bbox_2d_norm_1000": [0, 0, 1000, 1000],
        "occlusion_level": "none|low|medium|high",
        "bbox_reason": "what pixels belong to the object"
      },
      "articulation": {
        "root_part": "body or null",
        "moving_parts": ["door"],
        "joint_type": "revolute|prismatic|fixed|unknown"
      },
      "confidence": 0.0
    }
  ],
  "ignored_objects": [{"name_en": "object", "reason": "not hand-manipulated"}],
  "uncertainty_notes": ["short note"]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional result JSON path. Explicit outputs do not update pipeline_state.json.",
    )
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable containing the DashScope API key.",
    )
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--api-image-quality", type=int, default=92)
    parser.add_argument("--max-keyframes", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=3000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--reuse-raw",
        action="store_true",
        help="Validate and package the existing mixed_interactions.raw.txt without rerunning Qwen3-VL.",
    )
    return parser.parse_args()


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Qwen response contains no JSON object")
    return json.loads(cleaned[start : end + 1])


def sample_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count <= maximum:
        return list(range(frame_count))
    return sorted(set(round(i * (frame_count - 1) / (maximum - 1)) for i in range(maximum)))


def image_data_url(path: Path, quality: int) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stream = io.BytesIO()
        rgb.save(stream, format="JPEG", quality=quality, subsampling=0, optimize=True)
    encoded = base64.b64encode(stream.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def call_dashscope(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DashScope API returned HTTP {error.code}: {detail}") from error
    choices = result.get("choices") or []
    if not choices:
        raise ValueError(f"DashScope response has no choices: {result}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"DashScope response has no text content: {result}")
    return content.strip()


def validate(parsed: dict, frame_count: int, labeled_frames: list[int]) -> None:
    events = parsed.get("events")
    objects = parsed.get("objects")
    if not isinstance(events, list) or not events:
        raise ValueError("No events returned")
    if not isinstance(objects, list) or not objects:
        raise ValueError("No manipulated objects returned")
    object_ids = set()
    object_classes = {}
    for target in objects:
        object_id = target.get("object_id")
        if not isinstance(object_id, str) or not object_id or object_id in object_ids:
            raise ValueError(f"Invalid or duplicate object_id: {object_id!r}")
        object_ids.add(object_id)
        if target.get("object_class") not in {"articulated", "rigid"}:
            raise ValueError(f"Invalid object class for {object_id}: {target.get('object_class')}")
        object_classes[object_id] = target["object_class"]
        frame0 = target.get("global_frame0") or {}
        box = frame0.get("bbox_2d_norm_1000")
        if frame0.get("visible") is not True or not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Missing global frame-0 bbox for {object_id}")
        x0, y0, x1, y1 = [float(value) for value in box]
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError(f"Invalid frame-0 bbox for {object_id}: {box}")
    labeled = set(labeled_frames)
    for event in events:
        start, end = event.get("start_frame"), event.get("end_frame")
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start <= end < frame_count):
            raise ValueError(f"Invalid event range: {start}..{end}")
        if start not in labeled or end not in labeled:
            raise ValueError(f"Event boundaries must use labeled frames: {start}..{end}")
        if event.get("object_id") not in object_ids:
            raise ValueError(f"Event references unknown object: {event.get('object_id')}")
        object_id = event["object_id"]
        expected_class = object_classes[object_id]
        reported_class = event.get("interaction_class")
        if reported_class != expected_class:
            parsed.setdefault("validation_corrections", []).append(
                {
                    "event_id": event.get("event_id"),
                    "object_id": object_id,
                    "field": "interaction_class",
                    "reported": reported_class,
                    "corrected": expected_class,
                    "reason": "Event class must match the referenced object's object_class.",
                }
            )
            event["interaction_class"] = expected_class
        if expected_class == "articulated":
            completion = event.get("articulation_completion")
            if not isinstance(completion, dict):
                raise ValueError(
                    f"Articulated event {event.get('event_id')} is missing articulation_completion"
                )
            terminal_reached = completion.get("terminal_reached")
            if not isinstance(terminal_reached, bool):
                raise ValueError(
                    f"Articulated event {event.get('event_id')} has invalid terminal_reached"
                )
            first_motion = completion.get("first_motion_frame")
            confirmation = completion.get("confirmation_frame")
            if not isinstance(first_motion, int) or first_motion not in labeled:
                raise ValueError(
                    f"Articulated event {event.get('event_id')} has invalid first_motion_frame"
                )
            if not isinstance(confirmation, int) or confirmation not in labeled:
                raise ValueError(
                    f"Articulated event {event.get('event_id')} has invalid confirmation_frame"
                )
            if not (start <= first_motion <= end < confirmation < frame_count):
                raise ValueError(
                    f"Articulated event {event.get('event_id')} must keep its later confirmation "
                    f"outside the tracking interval {start}..{end}"
                )
            evidence = completion.get("completion_evidence_frames")
            if not isinstance(evidence, list) or any(
                not isinstance(frame, int) or frame not in labeled for frame in evidence
            ):
                raise ValueError(
                    f"Articulated event {event.get('event_id')} has invalid completion evidence"
                )
            terminal_frame = completion.get("terminal_reached_frame")
            if terminal_reached:
                if not isinstance(terminal_frame, int) or terminal_frame not in labeled:
                    raise ValueError(
                        f"Completed articulated event {event.get('event_id')} has no valid "
                        "terminal_reached_frame"
                    )
                if not (first_motion <= terminal_frame == end < confirmation):
                    raise ValueError(
                        f"Completed articulated event {event.get('event_id')} must end on its "
                        "first terminal frame and cite a distinct later confirmation frame"
                    )
                if terminal_frame not in evidence or confirmation not in evidence:
                    raise ValueError(
                        f"Completed articulated event {event.get('event_id')} must cite both "
                        "terminal and confirmation frames"
                    )
    classes = {event["interaction_class"] for event in events}
    if not {"articulated", "rigid"}.issubset(classes):
        raise ValueError(f"Expected articulated and rigid event classes, got {sorted(classes)}")


def main() -> int:
    args = parse_args()
    workspace = args.workspace.resolve()
    frame_dir = workspace / "outputs/00_rgb_frames/right_rgb_png"
    timeline_path = workspace / "outputs/00_rgb_frames/timeline.csv"
    camera_path = workspace / "outputs/00_rgb_frames/camera.json"
    output = (
        args.output.resolve()
        if args.output is not None
        else workspace / "outputs/01_vlm/mixed_interactions.json"
    )
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(frame_dir)
    with timeline_path.open(newline="", encoding="utf-8") as stream:
        timeline = list(csv.DictReader(stream))
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    selected_eye = str(camera.get("selected_eye", "right")).lower()
    coordinate_frame = f"frame0_{selected_eye}_camera"
    if len(timeline) != len(frames):
        raise ValueError(f"Timeline/frame mismatch: {len(timeline)} vs {len(frames)}")
    indices = sample_indices(len(frames), args.max_keyframes)
    if 0 not in indices:
        indices.insert(0, 0)

    prompt = PROMPT.replace("SELECTED_EYE", f"{selected_eye.upper()}-EYE")
    content = [{"type": "text", "text": prompt}]
    for index in indices:
        timestamp = float(timeline[index]["rgb_timestamp_s"])
        content.extend(
            [
                {"type": "text", "text": f"Global frame {index}, t={timestamp:.6f}s:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(frames[index], args.api_image_quality),
                    },
                },
            ]
        )
    messages = [{"role": "user", "content": content}]
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = output.with_suffix(".raw.txt")
    if args.reuse_raw:
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        response = raw_path.read_text(encoding="utf-8").strip()
        print(f"Reusing Qwen3-VL response from {raw_path}", flush=True)
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"DashScope API key is missing. Export {args.api_key_env} before running this stage."
            )
        print(f"Calling DashScope model {args.model} at {args.base_url}", flush=True)
        response = call_dashscope(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            messages=messages,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout_s=args.request_timeout_s,
        )
        raw_path.write_text(response + "\n", encoding="utf-8")
    parsed = extract_json(response)
    validate(parsed, len(frames), indices)
    for target in parsed["objects"]:
        target["global_frame0"].update(
            {
                "frame_index": 0,
                "timestamp_s": float(timeline[0]["rgb_timestamp_s"]),
                "frame_path": str(frames[0].resolve()),
            }
        )
    result = {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "dashscope_openai_compatible",
            "base_url": args.base_url,
            "model": args.model,
            "camera": selected_eye,
            "frame_count": len(frames),
            "sampled_global_frames": indices,
            "modeling_global_frame": 0,
            "coordinate_frame": coordinate_frame,
        },
        "vlm_result": parsed,
    }
    write_json(output, result)
    if args.output is None:
        update_stage_state(
            workspace / "pipeline_state.json",
            "01_vlm_mixed_interactions",
            "completed",
            inputs=[str(frame_dir), str(timeline_path)],
            outputs=[str(output), str(raw_path)],
            notes=f"Identified {len(parsed['events'])} events and {len(parsed['objects'])} manipulated objects; all mesh prompts are global {selected_eye}-eye frame 0.",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
