#!/usr/bin/env python3
import argparse
import base64
import io
import json
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw


DEFAULT_VLM_JSON = (
    "/code/3DVideo_2026-07-01-19-24-06-667_spatialmp4_export/"
    "qwen3vl8b_reconstruction_targets_right_keyframes.json"
)
DEFAULT_SAM2_ROOT = "/code/sam2"
DEFAULT_SAM2_CHECKPOINT = (
    "/code/ArtHOI-4D-Reconstruction/third_party/sam2/checkpoints/"
    "sam2.1_hiera_large.pt"
)
DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use VLM-selected keyframes and boxes as SAM2 interactive prompts."
    )
    parser.add_argument("--vlm-json", default=DEFAULT_VLM_JSON)
    parser.add_argument(
        "--output-dir",
        default="/code/vlm_sam2_recon/outputs/sam2_masks",
        help="Directory for masks, overlays, and metadata.",
    )
    parser.add_argument("--sam2-root", default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--default-target", default="target_phone")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run non-interactive batch segmentation using VLM boxes and optional clicks JSON.",
    )
    parser.add_argument(
        "--targets",
        default="target_laptop,target_phone",
        help="Comma-separated target object ids to segment.",
    )
    parser.add_argument(
        "--clicks-json",
        default=None,
        help=(
            "Optional JSON with positive/negative click prompts per target. "
            "Coordinates are pixel x,y by default."
        ),
    )
    parser.add_argument(
        "--clicks-normalized-1000",
        action="store_true",
        help="Interpret clicks JSON points as normalized 0..1000 coordinates.",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Ask SAM2 for multiple masks and keep the highest score.",
    )
    parser.add_argument(
        "--no-box",
        action="store_true",
        help="Ignore VLM bbox and use clicks only. Useful for manual correction.",
    )
    parser.add_argument(
        "--write-click-template",
        action="store_true",
        help="Write a click template JSON next to outputs and exit before SAM2 inference.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def import_sam2(sam2_root: Path):
    sys.path.insert(0, str(sam2_root))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return build_sam2, SAM2ImagePredictor


def resolve_image_path(vlm_data: dict, target: dict) -> Path:
    selected = selected_frame(target)
    frame_path = selected.get("frame_path")
    if frame_path:
        return Path(frame_path).resolve()

    frame_file = selected.get("frame_file")
    if frame_file:
        frame_dir = vlm_data.get("metadata", {}).get("frame_dir")
        if frame_dir:
            return (Path(frame_dir) / frame_file).resolve()

    frame_index = selected.get("frame_index")
    frame_files = vlm_data.get("metadata", {}).get("frame_files") or []
    frame_dir = vlm_data.get("metadata", {}).get("frame_dir")
    if isinstance(frame_index, int) and frame_dir and frame_index < len(frame_files):
        return (Path(frame_dir) / frame_files[frame_index]).resolve()

    raise ValueError(f"Cannot resolve selected keyframe image path for {target.get('object_id')}")


def selected_frame(target: dict) -> dict:
    """Return the image-prompt record from legacy or mixed-pipeline VLM output."""
    return target.get("selected_keyframe") or target.get("global_frame0") or {}


def normalized_box_to_pixels(box, width: int, height: int):
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = np.clip(x1 / 1000.0 * width, 0, width - 1)
    x2 = np.clip(x2 / 1000.0 * width, 0, width - 1)
    y1 = np.clip(y1 / 1000.0 * height, 0, height - 1)
    y2 = np.clip(y2 / 1000.0 * height, 0, height - 1)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def points_to_pixels(points, width: int, height: int, normalized_1000: bool):
    if not points:
        return []
    out = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Point must be [x, y], got {point!r}")
        x, y = float(point[0]), float(point[1])
        if normalized_1000:
            x = x / 1000.0 * width
            y = y / 1000.0 * height
        out.append([np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)])
    return out


def target_lookup(vlm_data: dict):
    result = vlm_data.get("vlm_result", {})
    targets = result.get("target_objects") or result.get("objects") or []
    return {target.get("object_id"): target for target in targets if target.get("object_id")}


def make_click_template(vlm_data: dict, target_ids: list[str], output_dir: Path) -> Path:
    lookup = target_lookup(vlm_data)
    template = {
        "_comment": (
            "Add positive_points inside the object and negative_points outside it. "
            "Coordinates are pixel [x, y] unless you run with --clicks-normalized-1000."
        ),
        "targets": {},
    }
    for target_id in target_ids:
        target = lookup.get(target_id)
        if not target:
            continue
        selected = selected_frame(target)
        template["targets"][target_id] = {
            "name_zh": target.get("name_zh"),
            "name_en": target.get("name_en"),
            "frame_index": selected.get("frame_index"),
            "frame_file": selected.get("frame_file"),
            "frame_path": selected.get("frame_path"),
            "positive_points": [],
            "negative_points": [],
            "use_box": True,
        }
    path = output_dir / "sam2_clicks_template.json"
    save_json(path, template)
    return path


def load_clicks(path: Path | None):
    if path is None:
        return {}
    data = load_json(path)
    if "targets" in data:
        return data["targets"]
    return data


def select_best_mask(masks: np.ndarray, scores: np.ndarray, logits: np.ndarray):
    if scores is None or len(scores) == 0:
        idx = 0
    else:
        idx = int(np.argmax(scores))
    return masks[idx], float(scores[idx]) if scores is not None else None, logits[idx]


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def save_mask_outputs(
    image: Image.Image,
    mask: np.ndarray,
    output_prefix: Path,
    color=(30, 144, 255),
) -> dict:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = mask_to_uint8(mask)
    mask_path = output_prefix.with_suffix(".mask.png")
    npy_path = output_prefix.with_suffix(".mask.npy")
    overlay_path = output_prefix.with_suffix(".overlay.png")

    Image.fromarray(mask_u8).save(mask_path)
    np.save(npy_path, mask.astype(bool))

    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    overlay = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8)
    alpha = 0.45
    overlay[mask] = (
        (1 - alpha) * overlay[mask].astype(np.float32) + alpha * color_arr.astype(np.float32)
    ).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)

    return {
        "mask_png": str(mask_path),
        "mask_npy": str(npy_path),
        "overlay_png": str(overlay_path),
        "mask_area_pixels": int(mask.sum()),
    }


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_overlay_image(
    image: Image.Image,
    mask: np.ndarray,
    color=(30, 144, 255),
    alpha=0.45,
) -> Image.Image:
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    overlay = rgb.copy()
    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask] = (
        (1 - alpha) * overlay[mask].astype(np.float32) + alpha * color_arr.astype(np.float32)
    ).astype(np.uint8)
    return Image.fromarray(overlay)


def save_prompt_overlay(
    image: Image.Image,
    box,
    pos_points,
    neg_points,
    output_prefix: Path,
) -> str:
    prompt_path = output_prefix.with_suffix(".prompt.png")
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if box is not None:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 210, 0), width=5)
    for x, y in pos_points:
        r = 10
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(20, 220, 80), outline=(0, 0, 0), width=2)
    for x, y in neg_points:
        r = 10
        draw.line([x - r, y - r, x + r, y + r], fill=(255, 40, 40), width=5)
        draw.line([x - r, y + r, x + r, y - r], fill=(255, 40, 40), width=5)
    canvas.save(prompt_path)
    return str(prompt_path)


def build_predictor(args):
    sam2_root = Path(args.sam2_root).resolve()
    checkpoint = Path(args.sam2_checkpoint).resolve()
    if not sam2_root.is_dir():
        raise FileNotFoundError(f"SAM2 root not found: {sam2_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    build_sam2, SAM2ImagePredictor = import_sam2(sam2_root)
    model = build_sam2(args.sam2_config, str(checkpoint), device=args.device)
    return SAM2ImagePredictor(model)


def prepare_prompt(target: dict, click_data: dict, image_size, args):
    width, height = image_size
    selected = selected_frame(target)
    box = None
    use_box = bool(click_data.get("use_box", True))
    if not args.no_box and use_box:
        box = normalized_box_to_pixels(selected.get("bbox_2d_norm_1000"), width, height)

    pos_points = points_to_pixels(
        click_data.get("positive_points", []),
        width,
        height,
        args.clicks_normalized_1000,
    )
    neg_points = points_to_pixels(
        click_data.get("negative_points", []),
        width,
        height,
        args.clicks_normalized_1000,
    )
    point_coords = None
    point_labels = None
    if pos_points or neg_points:
        point_coords = np.array(pos_points + neg_points, dtype=np.float32)
        point_labels = np.array([1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
    return box, pos_points, neg_points, point_coords, point_labels


def predict_target_mask(
    predictor,
    image: Image.Image,
    target: dict,
    click_data: dict,
    args,
):
    width, height = image.size
    box, pos_points, neg_points, point_coords, point_labels = prepare_prompt(
        target, click_data, (width, height), args
    )
    if box is None and point_coords is None:
        raise ValueError("Need at least one prompt: enable box or add positive/negative points.")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
        predictor.set_image(np.array(image))
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=args.multimask_output,
            return_logits=False,
            normalize_coords=True,
        )

    mask, score, _ = select_best_mask(masks, scores, logits)
    return mask.astype(bool), score, box, pos_points, neg_points


def run_target(
    predictor,
    vlm_data: dict,
    target: dict,
    click_data: dict,
    output_dir: Path,
    args,
) -> dict:
    image_path = resolve_image_path(vlm_data, target)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    selected = selected_frame(target)

    mask, score, box, pos_points, neg_points = predict_target_mask(
        predictor, image, target, click_data, args
    )
    target_id = target["object_id"]
    frame_index = selected.get("frame_index", "unknown")
    output_prefix = output_dir / target_id / f"{target_id}_frame_{frame_index}"
    output_paths = save_mask_outputs(image, mask, output_prefix)
    prompt_overlay = save_prompt_overlay(image, box, pos_points, neg_points, output_prefix)

    return {
        "target_id": target_id,
        "name_zh": target.get("name_zh"),
        "name_en": target.get("name_en"),
        "object_class_for_reconstruction": target.get("object_class_for_reconstruction") or target.get("object_class"),
        "image_path": str(image_path),
        "frame_index": selected.get("frame_index"),
        "frame_file": selected.get("frame_file") or image_path.name,
        "image_size": {"width": width, "height": height},
        "sam2_score": score,
        "box_xyxy_pixels": box.tolist() if box is not None else None,
        "positive_points_pixels": pos_points,
        "negative_points_pixels": neg_points,
        "prompt_overlay_png": prompt_overlay,
        **output_paths,
    }


def load_run_context(args):
    vlm_json = Path(args.vlm_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vlm_data = load_json(vlm_json)
    target_ids = [item.strip() for item in args.targets.split(",") if item.strip()]
    lookup = target_lookup(vlm_data)
    missing = [target_id for target_id in target_ids if target_id not in lookup]
    if missing:
        raise KeyError(f"Targets not found in VLM JSON: {missing}")
    return vlm_json, output_dir, vlm_data, target_ids, lookup


def run_batch(args) -> None:
    vlm_json, output_dir, vlm_data, target_ids, lookup = load_run_context(args)
    if args.write_click_template:
        path = make_click_template(vlm_data, target_ids, output_dir)
        print(f"Wrote click template: {path}")
        return

    clicks = load_clicks(Path(args.clicks_json).resolve() if args.clicks_json else None)
    predictor = build_predictor(args)

    results = []
    for target_id in target_ids:
        print(f"Segmenting {target_id}...")
        results.append(
            run_target(
                predictor=predictor,
                vlm_data=vlm_data,
                target=lookup[target_id],
                click_data=clicks.get(target_id, {}),
                output_dir=output_dir,
                args=args,
            )
        )

    summary = {
        "vlm_json": str(vlm_json),
        "sam2_root": str(Path(args.sam2_root).resolve()),
        "sam2_config": args.sam2_config,
        "sam2_checkpoint": str(Path(args.sam2_checkpoint).resolve()),
        "targets": results,
    }
    summary_path = output_dir / "sam2_mask_summary.json"
    save_json(summary_path, summary)
    print(f"Wrote summary: {summary_path}")


def target_public_metadata(vlm_data: dict, target: dict) -> dict:
    image_path = resolve_image_path(vlm_data, target)
    image = Image.open(image_path)
    selected = selected_frame(target)
    return {
        "target_id": target.get("object_id"),
        "name_zh": target.get("name_zh"),
        "name_en": target.get("name_en"),
        "object_class_for_reconstruction": target.get("object_class_for_reconstruction") or target.get("object_class"),
        "frame_index": selected.get("frame_index"),
        "frame_file": selected.get("frame_file") or image_path.name,
        "image_path": str(image_path),
        "image_width": image.size[0],
        "image_height": image.size[1],
        "bbox_2d_norm_1000": selected.get("bbox_2d_norm_1000"),
        "observed_state": target.get("observed_state"),
        "selection_reason": selected.get("selection_reason"),
    }


def interactive_html(targets_json: str, default_target: str, default_use_box: bool = True) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLM + SAM2 Interactive Mask</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f5f2;
      color: #1d1d1b;
    }}
    header {{
      padding: 14px 18px;
      background: #202322;
      color: white;
      display: flex;
      gap: 16px;
      align-items: center;
      flex-wrap: wrap;
    }}
    main {{
      padding: 16px;
      display: grid;
      grid-template-columns: minmax(480px, 1fr) minmax(360px, 0.78fr);
      gap: 16px;
      align-items: start;
    }}
    .bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    select, button, label {{
      font: inherit;
    }}
    select, button {{
      border: 1px solid #aaa;
      background: white;
      color: #1d1d1b;
      border-radius: 6px;
      padding: 7px 10px;
    }}
    button.active {{
      background: #1f6feb;
      border-color: #1f6feb;
      color: white;
    }}
    button.danger {{
      border-color: #c44949;
      color: #9b1c1c;
    }}
    button.primary {{
      background: #167a45;
      border-color: #167a45;
      color: white;
    }}
    .panel {{
      background: white;
      border: 1px solid #d2d2cc;
      border-radius: 8px;
      padding: 12px;
    }}
    #canvasWrap {{
      overflow: auto;
      border: 1px solid #d8d8d0;
      background: #111;
      max-height: calc(100vh - 190px);
    }}
    canvas {{
      display: block;
      max-width: 100%;
      height: auto;
      cursor: crosshair;
    }}
    #overlayImg {{
      width: 100%;
      max-height: calc(100vh - 230px);
      object-fit: contain;
      background: #111;
      border: 1px solid #d8d8d0;
    }}
    .meta {{
      font-size: 13px;
      color: #454541;
      line-height: 1.45;
      margin-top: 8px;
      white-space: pre-wrap;
    }}
    .points {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
      font-size: 13px;
    }}
    .list {{
      background: #f7f7f3;
      border: 1px solid #deded6;
      border-radius: 6px;
      padding: 8px;
      min-height: 58px;
      max-height: 120px;
      overflow: auto;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <strong>VLM + SAM2 Interactive Mask</strong>
    <div class="bar">
      <select id="targetSelect"></select>
      <button id="positiveBtn" class="active">Positive</button>
      <button id="negativeBtn">Negative</button>
      <button id="undoBtn">Undo</button>
      <button id="clearBtn" class="danger">Clear</button>
      <label><input type="checkbox" id="useBox" {"checked" if default_use_box else ""} /> VLM box</label>
      <label><input type="checkbox" id="multiMask" /> multi-mask</label>
      <button id="runBtn" class="primary">Run SAM2</button>
      <button id="saveBtn">Save mask</button>
    </div>
  </header>
  <main>
    <section class="panel">
      <div id="canvasWrap"><canvas id="imageCanvas"></canvas></div>
      <div class="points">
        <div>
          <strong>Positive</strong>
          <div id="posList" class="list"></div>
        </div>
        <div>
          <strong>Negative</strong>
          <div id="negList" class="list"></div>
        </div>
      </div>
      <div id="meta" class="meta"></div>
    </section>
    <section class="panel">
      <img id="overlayImg" alt="SAM2 overlay will appear here" />
      <div id="status" class="meta">Ready.</div>
    </section>
  </main>
  <script>
    const TARGETS = {targets_json};
    const DEFAULT_TARGET = {json.dumps(default_target)};
    const canvas = document.getElementById("imageCanvas");
    const ctx = canvas.getContext("2d");
    const targetSelect = document.getElementById("targetSelect");
    const statusEl = document.getElementById("status");
    const metaEl = document.getElementById("meta");
    const overlayImg = document.getElementById("overlayImg");
    const useBox = document.getElementById("useBox");
    const multiMask = document.getElementById("multiMask");
    let mode = "positive";
    let image = new Image();
    let currentTarget = null;
    let points = {{}};

    function targetLabel(t) {{
      return `${{t.target_id}} | ${{t.name_en}} | ${{t.object_class_for_reconstruction}} | frame ${{t.frame_index}}`;
    }}

    for (const t of TARGETS) {{
      const option = document.createElement("option");
      option.value = t.target_id;
      option.textContent = targetLabel(t);
      targetSelect.appendChild(option);
      points[t.target_id] = {{positive: [], negative: []}};
    }}
    targetSelect.value = TARGETS.some(t => t.target_id === DEFAULT_TARGET) ? DEFAULT_TARGET : TARGETS[0].target_id;

    function setMode(next) {{
      mode = next;
      document.getElementById("positiveBtn").classList.toggle("active", mode === "positive");
      document.getElementById("negativeBtn").classList.toggle("active", mode === "negative");
    }}

    function bboxPixels(t) {{
      const b = t.bbox_2d_norm_1000;
      if (!b) return null;
      return [
        b[0] / 1000 * t.image_width,
        b[1] / 1000 * t.image_height,
        b[2] / 1000 * t.image_width,
        b[3] / 1000 * t.image_height,
      ];
    }}

    function draw() {{
      if (!currentTarget || !image.complete) return;
      canvas.width = image.naturalWidth;
      canvas.height = image.naturalHeight;
      ctx.drawImage(image, 0, 0);
      if (useBox.checked) {{
        const b = bboxPixels(currentTarget);
        if (b) {{
          ctx.strokeStyle = "rgb(255,210,0)";
          ctx.lineWidth = 5;
          ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        }}
      }}
      const p = points[currentTarget.target_id];
      ctx.lineWidth = 4;
      for (const pt of p.positive) {{
        ctx.beginPath();
        ctx.fillStyle = "rgb(20,220,80)";
        ctx.strokeStyle = "black";
        ctx.arc(pt[0], pt[1], 12, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }}
      for (const pt of p.negative) {{
        ctx.strokeStyle = "rgb(255,40,40)";
        ctx.beginPath();
        ctx.moveTo(pt[0] - 12, pt[1] - 12);
        ctx.lineTo(pt[0] + 12, pt[1] + 12);
        ctx.moveTo(pt[0] - 12, pt[1] + 12);
        ctx.lineTo(pt[0] + 12, pt[1] - 12);
        ctx.stroke();
      }}
      renderPointLists();
    }}

    function renderPointLists() {{
      if (!currentTarget) return;
      const p = points[currentTarget.target_id];
      document.getElementById("posList").textContent = p.positive.map(x => `[${{x[0]}}, ${{x[1]}}]`).join("\\n");
      document.getElementById("negList").textContent = p.negative.map(x => `[${{x[0]}}, ${{x[1]}}]`).join("\\n");
    }}

    function loadTarget(targetId) {{
      currentTarget = TARGETS.find(t => t.target_id === targetId);
      overlayImg.removeAttribute("src");
      metaEl.textContent = [
        `target: ${{targetLabel(currentTarget)}}`,
        `state: ${{currentTarget.observed_state || "unknown"}}`,
        `image: ${{currentTarget.image_path}}`,
        `selection: ${{currentTarget.selection_reason || ""}}`
      ].join("\\n");
      image = new Image();
      image.onload = draw;
      image.src = `/image/${{currentTarget.target_id}}?t=${{Date.now()}}`;
      statusEl.textContent = "Ready.";
    }}

    async function runPredict(save) {{
      if (!currentTarget) return;
      statusEl.textContent = save ? "Running SAM2 and saving..." : "Running SAM2...";
      const p = points[currentTarget.target_id];
      const resp = await fetch("/api/predict", {{
        method: "POST",
        headers: {{"Content-Type": "application/json"}},
        body: JSON.stringify({{
          target_id: currentTarget.target_id,
          positive_points: p.positive,
          negative_points: p.negative,
          use_box: useBox.checked,
          multimask_output: multiMask.checked,
          save: save
        }})
      }});
      const data = await resp.json();
      if (!resp.ok) {{
        statusEl.textContent = data.error || "SAM2 failed.";
        return;
      }}
      overlayImg.src = data.overlay_data_url;
      let lines = [
        `score: ${{data.sam2_score}}`,
        `area: ${{data.mask_area_pixels}} pixels`
      ];
      if (data.saved) {{
        lines.push(`saved mask: ${{data.mask_png}}`);
        lines.push(`saved overlay: ${{data.overlay_png}}`);
      }}
      statusEl.textContent = lines.join("\\n");
    }}

    canvas.addEventListener("click", (ev) => {{
      if (!currentTarget) return;
      const rect = canvas.getBoundingClientRect();
      const x = Math.round((ev.clientX - rect.left) / rect.width * canvas.width);
      const y = Math.round((ev.clientY - rect.top) / rect.height * canvas.height);
      points[currentTarget.target_id][mode].push([x, y]);
      draw();
    }});
    targetSelect.addEventListener("change", () => loadTarget(targetSelect.value));
    useBox.addEventListener("change", draw);
    document.getElementById("positiveBtn").addEventListener("click", () => setMode("positive"));
    document.getElementById("negativeBtn").addEventListener("click", () => setMode("negative"));
    document.getElementById("undoBtn").addEventListener("click", () => {{
      const p = points[currentTarget.target_id][mode];
      p.pop();
      draw();
    }});
    document.getElementById("clearBtn").addEventListener("click", () => {{
      points[currentTarget.target_id] = {{positive: [], negative: []}};
      draw();
    }});
    document.getElementById("runBtn").addEventListener("click", () => runPredict(false));
    document.getElementById("saveBtn").addEventListener("click", () => runPredict(true));
    loadTarget(targetSelect.value);
  </script>
</body>
</html>"""


def run_server(args) -> None:
    from flask import Flask, jsonify, request, send_file

    vlm_json, output_dir, vlm_data, target_ids, lookup = load_run_context(args)
    if args.write_click_template:
        path = make_click_template(vlm_data, target_ids, output_dir)
        print(f"Wrote click template: {path}")
        return

    targets = [target_public_metadata(vlm_data, lookup[target_id]) for target_id in target_ids]
    target_meta = {item["target_id"]: item for item in targets}
    print("Loading SAM2 predictor for interactive server...")
    predictor = build_predictor(args)
    print("SAM2 predictor loaded.")

    app = Flask(__name__)

    @app.get("/")
    def index():
        return interactive_html(
            json.dumps(targets, ensure_ascii=False),
            args.default_target,
            default_use_box=not args.no_box,
        )

    @app.get("/image/<target_id>")
    def image(target_id):
        if target_id not in target_meta:
            return jsonify({"error": f"unknown target: {target_id}"}), 404
        return send_file(target_meta[target_id]["image_path"])

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(force=True)
        target_id = payload.get("target_id")
        if target_id not in lookup:
            return jsonify({"error": f"unknown target: {target_id}"}), 404

        local_args = argparse.Namespace(**vars(args))
        local_args.multimask_output = bool(payload.get("multimask_output", args.multimask_output))
        click_data = {
            "positive_points": payload.get("positive_points") or [],
            "negative_points": payload.get("negative_points") or [],
            "use_box": bool(payload.get("use_box", True)),
        }
        try:
            target = lookup[target_id]
            image_path = resolve_image_path(vlm_data, target)
            image_obj = Image.open(image_path).convert("RGB")
            mask, score, box, pos_points, neg_points = predict_target_mask(
                predictor, image_obj, target, click_data, local_args
            )
            overlay = make_overlay_image(image_obj, mask)
            selected = selected_frame(target)
            frame_index = selected.get("frame_index", "unknown")
            output_prefix = output_dir / target_id / f"{target_id}_frame_{frame_index}_interactive"
            saved = bool(payload.get("save", False))
            output_paths = {}
            prompt_overlay = None
            if saved:
                output_paths = save_mask_outputs(image_obj, mask, output_prefix)
                prompt_overlay = save_prompt_overlay(image_obj, box, pos_points, neg_points, output_prefix)
                summary_path = output_dir / "sam2_interactive_mask_summary.json"
                summary = load_json(summary_path) if summary_path.exists() else {
                    "vlm_json": str(vlm_json),
                    "sam2_root": str(Path(args.sam2_root).resolve()),
                    "sam2_config": args.sam2_config,
                    "sam2_checkpoint": str(Path(args.sam2_checkpoint).resolve()),
                    "targets": {},
                }
                summary["targets"][target_id] = {
                    "target_id": target_id,
                    "name_zh": target.get("name_zh"),
                    "name_en": target.get("name_en"),
                    "object_class_for_reconstruction": target.get("object_class_for_reconstruction") or target.get("object_class"),
                    "image_path": str(image_path),
                    "frame_index": frame_index,
                    "sam2_score": score,
                    "box_xyxy_pixels": box.tolist() if box is not None else None,
                    "positive_points_pixels": pos_points,
                    "negative_points_pixels": neg_points,
                    "prompt_overlay_png": prompt_overlay,
                    **output_paths,
                }
                save_json(summary_path, summary)
            return jsonify({
                "target_id": target_id,
                "sam2_score": score,
                "mask_area_pixels": int(mask.sum()),
                "overlay_data_url": image_to_data_url(overlay),
                "mask_data_url": image_to_data_url(Image.fromarray(mask_to_uint8(mask))),
                "saved": saved,
                "prompt_overlay_png": prompt_overlay,
                **output_paths,
            })
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    host_ips = []
    try:
        host_ips = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")]
    except Exception:
        host_ips = []
    print("", flush=True)
    print("Interactive SAM2 server is ready:", flush=True)
    print(f"  local:   http://127.0.0.1:{args.port}", flush=True)
    print(f"  local:   http://localhost:{args.port}", flush=True)
    for ip in host_ips:
        print(f"  network: http://{ip}:{args.port}", flush=True)
    print("If this is a remote machine, forward the port, e.g. ssh -L 7862:127.0.0.1:7862 <server>", flush=True)
    print("", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


def main() -> None:
    args = parse_args()
    if args.batch:
        run_batch(args)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
