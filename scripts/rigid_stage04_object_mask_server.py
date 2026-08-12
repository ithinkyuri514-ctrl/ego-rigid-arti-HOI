#!/usr/bin/env python3
"""Browser point-prompt UI used by the interactive Stage 04 entry point."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    print_server_addresses,
    read_json,
    update_stage_state,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.sam2_video import (  # noqa: E402
    add_points_or_box,
    build_video_predictor,
    mask_sequence_qc,
    overlay_mask,
    propagate_bidirectional,
    save_propagation_outputs,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SAM2 Object Masks</title>
  <style>
    :root { color-scheme:light; font-family:system-ui,sans-serif; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f2f3f1; color:#20221f; }
    header { display:flex; gap:8px; flex-wrap:wrap; align-items:center; padding:10px 14px; background:#202522; color:#fff; }
    button,input { font:inherit; }
    button { border:1px solid #969b96; background:#fff; padding:7px 10px; border-radius:6px; cursor:pointer; }
    button.active { background:#1769aa; color:#fff; border-color:#1769aa; }
    button.primary { background:#147a46; color:#fff; border-color:#147a46; }
    button.api { background:#8a4f12; color:#fff; border-color:#8a4f12; }
    button:disabled { opacity:.5; cursor:not-allowed; }
    main { display:grid; grid-template-columns:minmax(560px,1fr) minmax(360px,520px); gap:12px; padding:12px; }
    section { background:#fff; border:1px solid #d1d4cf; border-radius:7px; padding:10px; }
    #canvasWrap { overflow:auto; max-height:calc(100vh - 178px); background:#111; }
    canvas { display:block; max-width:100%; height:auto; cursor:crosshair; }
    #preview { width:100%; max-height:66vh; object-fit:contain; background:#111; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
    .grow { flex:1; min-width:220px; }
    #status,#points { white-space:pre-wrap; font:13px/1.45 ui-monospace,monospace; background:#f5f6f3; padding:8px; border-radius:5px; min-height:44px; }
    @media(max-width:980px){ main{grid-template-columns:1fr;} }
  </style>
</head>
<body>
<header>
  <strong>Stage 04 · 物体 Mask</strong>
  <button id="pos" class="active">正点</button><button id="neg">负点</button>
  <button id="undo">撤销</button><button id="clear">清空本帧</button>
  <label><input id="fillHoles" type="checkbox" checked> 填内部孔洞</label>
  <button id="previewBtn">预览本帧</button>
  <button id="propagate" class="primary">传播并保存全视频</button>
  <button id="hunyuan" class="api" disabled>运行混元3D</button>
</header>
<main>
  <section>
    <div class="row">
      <label>帧 <input id="slider" class="grow" type="range" min="0" value="0"></label>
      <input id="frameNumber" type="number" min="0" value="0" style="width:76px">
      <span id="frameLabel"></span>
    </div>
    <div id="canvasWrap"><canvas id="canvas"></canvas></div>
    <div id="points"></div>
  </section>
  <section>
    <img id="preview" alt="SAM2 mask preview">
    <h3>状态</h3><div id="status">正在读取元数据...</div>
  </section>
</main>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const slider=document.getElementById('slider'),frameNumber=document.getElementById('frameNumber');
const statusEl=document.getElementById('status'),pointsEl=document.getElementById('points'),previewEl=document.getElementById('preview');
let meta=null,img=new Image(),mode='positive',store={},hasResults=false;
function key(){return String(slider.value);}
function current(){if(!store[key()])store[key()]={positive_points:[],negative_points:[]};return store[key()];}
function setMode(v){mode=v;document.getElementById('pos').classList.toggle('active',v==='positive');document.getElementById('neg').classList.toggle('active',v==='negative');}
function draw(){if(!img.complete)return;canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);const p=current();
  for(const q of p.positive_points){ctx.beginPath();ctx.fillStyle='#20d56b';ctx.strokeStyle='#000';ctx.lineWidth=3;ctx.arc(q[0],q[1],11,0,Math.PI*2);ctx.fill();ctx.stroke();}
  for(const q of p.negative_points){ctx.strokeStyle='#ff3c38';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(q[0]-10,q[1]-10);ctx.lineTo(q[0]+10,q[1]+10);ctx.moveTo(q[0]-10,q[1]+10);ctx.lineTo(q[0]+10,q[1]-10);ctx.stroke();}
  const prompted=Object.keys(store).filter(k=>store[k].positive_points.length).map(Number).sort((a,b)=>a-b);
  pointsEl.textContent=`frame ${key()}\npositive: ${JSON.stringify(p.positive_points)}\nnegative: ${JSON.stringify(p.negative_points)}\nconditioned frames: ${JSON.stringify(prompted)}`;
}
function loadFrame(v){const n=Math.max(0,Math.min(meta.frame_count-1,Number(v)));slider.value=n;frameNumber.value=n;document.getElementById('frameLabel').textContent=`/ ${meta.frame_count-1}`;img=new Image();img.onload=draw;img.src=`/frame/${n}?t=${Date.now()}`;if(hasResults)previewEl.src=`/result/${n}?t=${Date.now()}`;else previewEl.removeAttribute('src');}
async function post(path,payload){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data;}
canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect(),x=Math.round((e.clientX-r.left)/r.width*canvas.width),y=Math.round((e.clientY-r.top)/r.height*canvas.height);current()[mode+'_points'].push([x,y]);draw();});
slider.addEventListener('input',()=>loadFrame(slider.value));frameNumber.addEventListener('change',()=>loadFrame(frameNumber.value));
document.getElementById('pos').onclick=()=>setMode('positive');document.getElementById('neg').onclick=()=>setMode('negative');
document.getElementById('undo').onclick=()=>{current()[mode+'_points'].pop();draw();};document.getElementById('clear').onclick=()=>{store[key()]={positive_points:[],negative_points:[]};draw();};
document.getElementById('previewBtn').onclick=async()=>{try{statusEl.textContent='SAM2 本帧推理中...';const p=current();const data=await post('/api/preview',{frame_index:Number(slider.value),fill_holes:document.getElementById('fillHoles').checked,...p});previewEl.src=data.overlay_data_url;statusEl.textContent=`已记录条件帧 ${data.frame_index}\n${data.object_id} area=${data.area_pixels}`;}catch(e){statusEl.textContent=e.message;}};
document.getElementById('propagate').onclick=async()=>{try{statusEl.textContent='正在双向传播并保存，请查看终端进度...';const data=await post('/api/propagate',{fill_holes:document.getElementById('fillHoles').checked});hasResults=true;document.getElementById('hunyuan').disabled=false;previewEl.src=`/result/${slider.value}?t=${Date.now()}`;statusEl.textContent=`完成 ${data.frame_count} 帧\nQC passed=${data.qc.passed}\nmask: ${data.prompt_mask}\nmanifest: ${data.manifest}`;}catch(e){statusEl.textContent=e.message;}};
document.getElementById('hunyuan').onclick=async()=>{if(!confirm('将归档旧 mesh，并使用新 mask 调用混元3D API。继续？'))return;try{statusEl.textContent='混元3D API 运行中，请查看终端日志...';const data=await post('/api/hunyuan',{});statusEl.textContent=`混元3D 完成\nmesh: ${data.mesh_path}\narchive: ${data.archive||'none'}`;}catch(e){statusEl.textContent=e.message;}};
fetch('/api/meta').then(r=>r.json()).then(x=>{meta=x;slider.max=x.frame_count-1;frameNumber.max=x.frame_count-1;hasResults=x.has_results;document.getElementById('hunyuan').disabled=(!hasResults)||(!x.hunyuan_enabled);if(!x.hunyuan_enabled)document.getElementById('hunyuan').style.display='none';statusEl.textContent=`${x.object_id}\n${x.frame_count} frames, ${x.width}x${x.height}, ${x.fps} FPS\nsource: ${x.frame_source||''}`;loadFrame(0);});
</script></body></html>"""


def image_data_url(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=91)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def compatible_stage_name(workspace: Path, *candidates: str) -> str:
    """Return the one candidate present in this workspace's pipeline state."""
    state_path = workspace / "pipeline_state.json"
    if not state_path.is_file():
        return candidates[0]
    state = read_json(state_path)
    available = {str(item.get("stage")) for item in state.get("stages", [])}
    matches = [name for name in candidates if name in available]
    if len(matches) != 1:
        raise KeyError(
            f"Expected one compatible stage record from {list(candidates)}, found {matches}"
        )
    return matches[0]


def update_compatible_stage_state(
    workspace: Path, candidates: tuple[str, ...], status: str, **kwargs
) -> str:
    stage_name = compatible_stage_name(workspace, *candidates)
    update_stage_state(workspace / "pipeline_state.json", stage_name, status, **kwargs)
    return stage_name


def try_update_compatible_stage_state(
    workspace: Path, candidates: tuple[str, ...], status: str, **kwargs
) -> str | None:
    try:
        return update_compatible_stage_state(workspace, candidates, status, **kwargs)
    except KeyError as exc:
        print(f"[object-mask-ui] pipeline_state warning: {exc}", flush=True)
        return None


def mirror_mixed_stage04_if_requested(workspace: Path, output_dir: Path, target: dict) -> None:
    if not bool(target.get("_mirror_mixed_legacy_masks", False)):
        return
    object_id = str(target.get("object_id") or "target_rigid_object")
    prompt_manifest_path = output_dir / "mesh_prompt_frame0/prompt_manifest.json"
    if not prompt_manifest_path.is_file():
        return
    prompt = read_json(prompt_manifest_path)
    compat_root = workspace / "outputs/02_sam2_frame0_masks"
    summary_path = compat_root / "sam2_frame0_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        objects = [item for item in summary.get("objects", []) if item.get("object_id") != object_id]
    else:
        summary = {
            "stage": "04_sam2_object_masks_compat_frame0_summary",
            "status": "completed",
            "camera": "right",
            "coordinate_frame": "global_frame0_right_image",
            "frame_index": 0,
            "source": "Compatibility mirror of interactive Stage 04 object masks.",
            "objects": [],
        }
        objects = []
    objects.append(
        {
            "object_id": object_id,
            "name_en": target.get("name_en"),
            "name_zh": target.get("name_zh"),
            "object_class": target.get("object_class") or target.get("object_class_for_reconstruction"),
            "frame_index": 0,
            "rgb": prompt["rgb"],
            "mask": prompt["mask"],
            "mask_source": "interactive_stage04_on_diffueraser",
        }
    )
    summary["objects"] = sorted(objects, key=lambda item: item["object_id"])
    write_json(summary_path, summary)

    source_mask_dir = output_dir / "objects" / object_id
    compat_mask_dir = compat_root / "propagated/objects" / object_id
    if source_mask_dir.is_dir():
        compat_mask_dir.parent.mkdir(parents=True, exist_ok=True)
        if compat_mask_dir.exists():
            shutil.rmtree(compat_mask_dir)
        shutil.copytree(source_mask_dir, compat_mask_dir)
    print(f"[object-mask-ui] wrote mixed compatibility masks for {object_id}: {compat_root}", flush=True)


class ObjectMaskService:
    def __init__(
        self,
        args,
        *,
        workspace: Path,
        output_dir: Path,
        target: dict,
        sam2_frames: list[Path],
        display_frames: list[Path],
    ) -> None:
        self.args = args
        self.workspace = workspace
        self.output_dir = output_dir
        self.target = target
        self.object_id = str(target.get("object_id") or "target_rigid_object")
        self.sam2_frames = sam2_frames
        self.display_frames = display_frames
        constraint_dir = target.get("constraint_mask_dir")
        self.constraint_mask_dir = Path(constraint_dir).resolve() if constraint_dir else None
        self.constraint_mask_paths: list[Path] = []
        if self.constraint_mask_dir is not None:
            self.constraint_mask_paths = sorted(self.constraint_mask_dir.glob("*.png"))
            if len(self.constraint_mask_paths) != len(display_frames):
                raise ValueError(
                    "Constraint-mask/frame mismatch: "
                    f"{len(self.constraint_mask_paths)} vs {len(display_frames)} "
                    f"in {self.constraint_mask_dir}"
                )
        with Image.open(display_frames[0]) as image:
            self.width, self.height = image.size
        self.predictor = None
        self.state = None
        self.prompts: dict[int, dict] = {}
        self.conditioning_frames: set[int] = set()
        self.results_ready = (
            (self.output_dir / "propagation_manifest.json").is_file()
            and (self.output_dir / "mesh_prompt_frame0/object_mask.png").is_file()
            and (self.output_dir / "mesh_prompt_frame0/rgb_no_hand.png").is_file()
        )
        self.lock = threading.Lock()
        self.hunyuan_lock = threading.Lock()
        self.hunyuan_enabled = bool(getattr(args, "enable_hunyuan", True))

    def load_model(self) -> None:
        print("Loading SAM2 video predictor...", flush=True)
        self.predictor = build_video_predictor(
            self.args.sam2_root,
            self.args.sam2_config,
            self.args.sam2_checkpoint,
            self.args.device,
        )
        self.state = self.predictor.init_state(
            str(self.sam2_frames[0].parent),
            offload_video_to_cpu=self.args.offload_video_to_cpu,
            offload_state_to_cpu=self.args.offload_state_to_cpu,
        )
        print("SAM2 video predictor loaded.", flush=True)

    def meta(self) -> dict:
        return {
            "object_id": self.object_id,
            "frame_count": len(self.display_frames),
            "width": self.width,
            "height": self.height,
            "fps": self.args.fps,
            "has_results": self.results_ready,
            "output_dir": str(self.output_dir),
            "frame_source": str(self.display_frames[0].parent),
            "hunyuan_enabled": self.hunyuan_enabled,
        }

    @staticmethod
    def postprocess(mask: np.ndarray, fill_holes: bool) -> np.ndarray:
        binary = np.asarray(mask, dtype=bool)
        return np.asarray(binary_fill_holes(binary), dtype=bool) if fill_holes else binary

    def apply_constraint(self, frame_index: int, mask: np.ndarray) -> np.ndarray:
        if not self.constraint_mask_paths:
            return np.asarray(mask, dtype=bool)
        constraint = np.asarray(
            Image.open(self.constraint_mask_paths[frame_index]).convert("L"), dtype=np.uint8
        ) > 127
        if constraint.shape != mask.shape:
            raise ValueError(
                f"Constraint/SAM2 mask mismatch at frame {frame_index}: "
                f"{constraint.shape} vs {mask.shape}"
            )
        return np.asarray(mask, dtype=bool) & constraint

    def preview(self, payload: dict) -> dict:
        frame_index = int(payload["frame_index"])
        if not 0 <= frame_index < len(self.display_frames):
            raise IndexError(frame_index)
        positive = payload.get("positive_points") or []
        negative = payload.get("negative_points") or []
        if not positive:
            raise ValueError("每个条件帧至少需要一个人工正点")
        with self.lock:
            masks = add_points_or_box(
                self.predictor,
                self.state,
                frame_index=frame_index,
                object_id=self.object_id,
                positive_points=positive,
                negative_points=negative,
            )
        mask = self.postprocess(masks[self.object_id], bool(payload.get("fill_holes", True)))
        mask = self.apply_constraint(frame_index, mask)
        self.prompts[frame_index] = {
            "object_id": self.object_id,
            "frame_index": frame_index,
            "positive_points": positive,
            "negative_points": negative,
        }
        self.conditioning_frames.add(frame_index)
        source = Image.open(self.display_frames[frame_index]).convert("RGB")
        canvas = overlay_mask(source, mask)
        draw = ImageDraw.Draw(canvas)
        for x, y in positive:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(20, 230, 95), outline=(0, 0, 0), width=2)
        for x, y in negative:
            draw.line((x - 8, y - 8, x + 8, y + 8), fill=(255, 30, 30), width=4)
            draw.line((x - 8, y + 8, x + 8, y - 8), fill=(255, 30, 30), width=4)
        return {
            "frame_index": frame_index,
            "object_id": self.object_id,
            "area_pixels": int(mask.sum()),
            "overlay_data_url": image_data_url(canvas),
        }

    def propagate(self, payload: dict) -> dict:
        if not self.conditioning_frames:
            raise ValueError("请先在至少一帧上点击正点并预览")
        fill_holes = bool(payload.get("fill_holes", True))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = self.output_dir / "object_prompts.json"
        write_json(
            prompt_path,
            {
                "policy": "Human-selected positive/negative points; no VLM box; no convex hull",
                "fill_internal_holes": fill_holes,
                "prompts": [self.prompts[key] for key in sorted(self.prompts)],
            },
        )
        with self.lock:
            results = propagate_bidirectional(
                self.predictor,
                self.state,
                sorted(self.conditioning_frames),
            )
        for frame_index, masks in results.items():
            for object_id, mask in list(masks.items()):
                processed = self.postprocess(mask, True) if fill_holes else np.asarray(mask, dtype=bool)
                masks[object_id] = self.apply_constraint(frame_index, processed)
        summary = save_propagation_outputs(
            results,
            self.display_frames,
            self.output_dir,
            fps=self.args.fps,
            video_stem="object_mask",
            save_overlays=bool(getattr(self.args, "save_overlays", False)),
        )
        summary.update(
            {
                "artifact_kind": self.target.get("artifact_kind", "whole_object_mask"),
                "target": {
                    "object_id": self.object_id,
                    "parent_object_id": self.target.get("parent_object_id"),
                    "part_id": self.target.get("part_id"),
                    "name_en": self.target.get("name_en"),
                    "name_zh": self.target.get("name_zh"),
                    "object_class": self.target.get("object_class")
                    or self.target.get("object_class_for_reconstruction"),
                },
                "camera": self.target.get("camera", "right"),
                "coordinate_frame": self.target.get(
                    "image_coordinate_frame", "global_frame0_right_image"
                ),
                "source_policy": self.target.get(
                    "source_policy",
                    "Human point prompts on the hand-removed tracking video.",
                ),
                "mask_constraint": {
                    "applied": bool(self.constraint_mask_paths),
                    "operation": "per_frame_intersection",
                    "mask_dir": str(self.constraint_mask_dir) if self.constraint_mask_dir else None,
                },
            }
        )
        write_json(self.output_dir / "propagation_manifest.json", summary)
        qc = mask_sequence_qc(summary, max_area_ratio=self.args.max_area_ratio, allow_empty=False)
        write_json(self.output_dir / "object_mask_qc.json", qc)
        self.results_ready = True

        prompt_dir = self.output_dir / "mesh_prompt_frame0"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.display_frames[0], prompt_dir / "rgb_no_hand.png")
        shutil.copy2(self.output_dir / "combined/000000.png", prompt_dir / "object_mask.png")
        write_json(
            prompt_dir / "prompt_manifest.json",
            {
                "object_id": self.object_id,
                "name_en": self.target.get("name_en"),
                "name_zh": self.target.get("name_zh"),
                "object_class": self.target.get("object_class") or self.target.get("object_class_for_reconstruction"),
                "frame_index": 0,
                "rgb": str(prompt_dir / "rgb_no_hand.png"),
                "mask": str(prompt_dir / "object_mask.png"),
                "source": "Human-confirmed point-prompted SAM2 propagation",
                "convex_hull_applied": False,
                "fill_internal_holes": fill_holes,
            },
        )
        if not bool(self.target.get("_skip_pipeline_state_updates", False)):
            object_mask_stage = compatible_stage_name(
                self.workspace,
                "04_sam2_object_masks",
                "04_sam2_object_and_part_masks",
            )
            object_mask_status = "completed" if qc["passed"] else "needs_revision"
            if object_mask_stage == "04_sam2_object_and_part_masks" and qc["passed"]:
                object_mask_status = "running"
            update_stage_state(
                self.workspace / "pipeline_state.json",
                object_mask_stage,
                object_mask_status,
                inputs=[str(self.display_frames[0].parent), str(prompt_path)],
                outputs=[str(self.output_dir)],
                notes=(
                    f"Human-point SAM2 propagation saved {summary['frame_count']} masks for "
                    f"{self.object_id}; QC passed={qc['passed']}. Articulated Stage 04 remains "
                    "running until whole, body and door masks are all accepted."
                ),
            )
            try_update_compatible_stage_state(
                self.workspace,
                ("05_hunyuan_mesh", "05_sam3d_frame0_reconstruction", "03_sam3d_frame0_reconstruction"),
                "pending",
                inputs=[str(prompt_dir / "rgb_no_hand.png"), str(prompt_dir / "object_mask.png")],
                outputs=[],
                notes="Interactive Stage 04 object mask changed; regenerate frame-0 object mesh.",
            )
            try_update_compatible_stage_state(
                self.workspace,
                ("07_frame0_mesh_alignment", "08_frame0_whole_part_icp_alignment", "07_frame0_multi_object_alignment"),
                "pending",
                inputs=[],
                outputs=[],
                notes="Interactive object mask changed; mesh alignment is stale.",
            )
        mirror_mixed_stage04_if_requested(self.workspace, self.output_dir, self.target)
        return {
            "frame_count": summary["frame_count"],
            "manifest": str(self.output_dir / "propagation_manifest.json"),
            "prompt_mask": str(prompt_dir / "object_mask.png"),
            "qc": qc,
        }

    def run_hunyuan(self) -> dict:
        if not self.hunyuan_enabled:
            raise RuntimeError("This server was started with Hunyuan disabled for the mixed SAM3D pipeline.")
        prompt_mask = self.output_dir / "mesh_prompt_frame0/object_mask.png"
        prompt_rgb = self.output_dir / "mesh_prompt_frame0/rgb_no_hand.png"
        if not self.results_ready:
            raise RuntimeError("请先传播并确认全视频 mask")
        if not prompt_mask.is_file() or not prompt_rgb.is_file():
            raise FileNotFoundError("请先传播并保存全视频 mask")
        with self.hunyuan_lock:
            output_dir = self.workspace / "outputs/05_hunyuan_mesh"
            archive = None
            if output_dir.exists():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive = self.workspace / f"scratch/05_hunyuan_mesh_before_interactive_{stamp}"
                shutil.copytree(output_dir, archive)
                print(f"Archived previous Hunyuan output: {archive}", flush=True)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/run_rigid_hunyuan_from_mask.py"),
                "--workspace",
                str(self.workspace),
                "--rgb",
                str(prompt_rgb),
                "--mask",
                str(prompt_mask),
                "--target-id",
                self.object_id,
                "--overwrite",
            ]
            print("$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            record = read_json(output_dir / "hunyuan3d_run.json")
            update_compatible_stage_state(
                self.workspace,
                ("07_frame0_mesh_alignment", "08_frame0_whole_part_icp_alignment"),
                "pending",
                inputs=[],
                outputs=[],
                notes="New interactive-mask Hunyuan mesh is ready; rerun metric mesh alignment.",
            )
            return {
                "mesh_path": record["mesh_path"],
                "archive": str(archive) if archive else None,
                "run_record": str(output_dir / "hunyuan3d_run.json"),
            }


def handler_factory(service: ObjectMaskService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args) -> None:
            print(f"[object-mask-ui] {fmt % args}", flush=True)

        def send_bytes(self, content: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, payload: dict, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/meta":
                self.send_json(service.meta())
                return
            if path.startswith("/frame/"):
                try:
                    index = int(path.rsplit("/", 1)[-1])
                    frame = service.display_frames[index]
                    self.send_bytes(frame.read_bytes(), mimetypes.guess_type(frame.name)[0] or "image/png")
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 404)
                return
            if path.startswith("/result/"):
                try:
                    index = int(path.rsplit("/", 1)[-1])
                    overlay = service.output_dir / f"overlays/{index:06d}.jpg"
                    self.send_bytes(overlay.read_bytes(), "image/jpeg")
                except Exception as exc:
                    self.send_json({"error": str(exc)}, 404)
                return
            self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/preview":
                    self.send_json(service.preview(payload))
                elif path == "/api/propagate":
                    self.send_json(service.propagate(payload))
                elif path == "/api/hunyuan":
                    self.send_json(service.run_hunyuan())
                else:
                    self.send_json({"error": "not found"}, 404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

    return Handler


def run_server(
    args,
    *,
    workspace: Path,
    output_dir: Path,
    target: dict,
    sam2_frames: list[Path],
    display_frames: list[Path],
) -> int:
    service = ObjectMaskService(
        args,
        workspace=workspace,
        output_dir=output_dir,
        target=target,
        sam2_frames=sam2_frames,
        display_frames=display_frames,
    )
    print(json.dumps(service.meta(), ensure_ascii=False, indent=2), flush=True)
    service.load_model()
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(service))
    print_server_addresses(args.host, args.port, "Interactive Stage 04 object-mask server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping object-mask server.", flush=True)
    finally:
        server.server_close()
    return 0
