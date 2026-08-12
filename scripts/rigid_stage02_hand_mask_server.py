#!/usr/bin/env python3
"""Interactive point-prompted SAM2 service for full-video hand masks."""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_sam2_recon.rigid_pipeline.common import (  # noqa: E402
    print_server_addresses,
    update_stage_state,
    write_json,
)
from vlm_sam2_recon.rigid_pipeline.sam2_video import (  # noqa: E402
    DEFAULT_SAM2_CHECKPOINT,
    DEFAULT_SAM2_CONFIG,
    DEFAULT_SAM2_ROOT,
    add_points_or_box,
    build_video_predictor,
    list_display_frames,
    overlay_mask,
    mask_sequence_qc,
    propagate_bidirectional,
    save_propagation_outputs,
)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Hand Mask Point Prompts</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { margin:0; background:#f3f4f2; color:#20211f; }
    header { display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding:10px 14px; background:#202522; color:#fff; }
    button,input { font:inherit; }
    button { border:1px solid #9da19d; background:#fff; padding:7px 10px; border-radius:6px; cursor:pointer; }
    button.active { background:#1769aa; color:#fff; border-color:#1769aa; }
    button.primary { background:#147a46; color:#fff; border-color:#147a46; }
    main { display:grid; grid-template-columns:minmax(540px,1fr) minmax(300px,420px); gap:14px; padding:14px; }
    section { background:#fff; border:1px solid #d4d6d2; border-radius:7px; padding:10px; }
    #wrap { overflow:auto; max-height:calc(100vh - 180px); background:#111; }
    canvas { display:block; max-width:100%; height:auto; cursor:crosshair; }
    #preview { width:100%; max-height:65vh; object-fit:contain; background:#111; }
    .row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
    .grow { flex:1; min-width:220px; }
    #status,#points { white-space:pre-wrap; font:13px/1.45 ui-monospace,monospace; background:#f5f6f3; padding:8px; border-radius:5px; min-height:42px; }
    @media(max-width:950px){ main{grid-template-columns:1fr;} }
  </style>
</head>
<body>
<header>
  <strong>交互式手部 Mask</strong>
  <label>手 ID <input id="objectId" value="hand" size="12"></label>
  <button id="pos" class="active">正点</button><button id="neg">负点</button>
  <button id="undo">撤销</button><button id="clear">清空本帧</button>
  <button id="previewBtn">预览本帧</button><button id="propagate" class="primary">传播并保存全视频</button>
</header>
<main>
  <section>
    <div class="row">
      <label>帧 <input id="slider" class="grow" type="range" min="0" value="0"></label>
      <input id="frameNumber" type="number" min="0" value="0" style="width:72px">
      <span id="frameLabel"></span>
    </div>
    <div id="wrap"><canvas id="canvas"></canvas></div>
    <div id="points"></div>
  </section>
  <section>
    <img id="preview" alt="SAM2 preview">
    <h3>状态</h3><div id="status">正在读取元数据...</div>
  </section>
</main>
<script>
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const slider=document.getElementById('slider'),frameNumber=document.getElementById('frameNumber');
const statusEl=document.getElementById('status'),pointsEl=document.getElementById('points');
let meta=null,img=new Image(),mode='positive',store={};
function key(){return document.getElementById('objectId').value.trim()+':'+slider.value;}
function current(){if(!store[key()])store[key()]={positive_points:[],negative_points:[]};return store[key()];}
function setMode(v){mode=v;document.getElementById('pos').classList.toggle('active',v==='positive');document.getElementById('neg').classList.toggle('active',v==='negative');}
function draw(){if(!img.complete)return;canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.drawImage(img,0,0);const p=current();
  for(const q of p.positive_points){ctx.beginPath();ctx.fillStyle='#20d56b';ctx.strokeStyle='#000';ctx.lineWidth=3;ctx.arc(q[0],q[1],11,0,Math.PI*2);ctx.fill();ctx.stroke();}
  for(const q of p.negative_points){ctx.strokeStyle='#ff3c38';ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(q[0]-10,q[1]-10);ctx.lineTo(q[0]+10,q[1]+10);ctx.moveTo(q[0]-10,q[1]+10);ctx.lineTo(q[0]+10,q[1]-10);ctx.stroke();}
  pointsEl.textContent=`${key()}\npositive: ${JSON.stringify(p.positive_points)}\nnegative: ${JSON.stringify(p.negative_points)}`;
}
function loadFrame(v){const n=Math.max(0,Math.min(meta.frame_count-1,Number(v)));slider.value=n;frameNumber.value=n;document.getElementById('frameLabel').textContent=`/ ${meta.frame_count-1}`;img=new Image();img.onload=draw;img.src=`/frame/${n}?t=${Date.now()}`;document.getElementById('preview').removeAttribute('src');}
async function post(path,payload){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data;}
canvas.addEventListener('click',e=>{const r=canvas.getBoundingClientRect(),x=Math.round((e.clientX-r.left)/r.width*canvas.width),y=Math.round((e.clientY-r.top)/r.height*canvas.height);current()[mode+'_points'].push([x,y]);draw();});
slider.addEventListener('input',()=>loadFrame(slider.value));frameNumber.addEventListener('change',()=>loadFrame(frameNumber.value));document.getElementById('objectId').addEventListener('change',draw);
document.getElementById('pos').onclick=()=>setMode('positive');document.getElementById('neg').onclick=()=>setMode('negative');
document.getElementById('undo').onclick=()=>{current()[mode+'_points'].pop();draw();};document.getElementById('clear').onclick=()=>{store[key()]={positive_points:[],negative_points:[]};draw();};
document.getElementById('previewBtn').onclick=async()=>{try{statusEl.textContent='SAM2 本帧推理中...';const p=current();const data=await post('/api/preview',{frame_index:Number(slider.value),object_id:document.getElementById('objectId').value.trim(),...p});document.getElementById('preview').src=data.overlay_data_url;statusEl.textContent=`已记录条件帧 ${data.frame_index}\n${data.object_id} area=${data.area_pixels}`;}catch(e){statusEl.textContent=e.message;}};
document.getElementById('propagate').onclick=async()=>{try{statusEl.textContent='正在双向传播并写出所有帧，请查看终端进度...';const data=await post('/api/propagate',{});statusEl.textContent=`完成 ${data.frame_count} 帧\nmask video: ${data.mask_video}\nmanifest: ${data.manifest}`;}catch(e){statusEl.textContent=e.message;}};
fetch('/api/meta').then(r=>r.json()).then(x=>{meta=x;slider.max=x.frame_count-1;frameNumber.max=x.frame_count-1;statusEl.textContent=`${x.frame_count} frames, ${x.width}x${x.height}, ${x.fps} FPS\n请至少选一个正点，然后点“预览本帧”。`;loadFrame(0);});
</script></body></html>"""


def parse_args() -> argparse.Namespace:
    workspace = PROJECT_ROOT / "run_rigid_20260715_215524"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--sam2-frame-dir", type=Path, default=None, help="JPEG directory passed to SAM2.")
    parser.add_argument("--display-frame-dir", type=Path, default=None, help="PNG/JPEG directory shown in browser.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-config", default=DEFAULT_SAM2_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_SAM2_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-state-to-cpu", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check", action="store_true", help="Validate paths and exit without loading SAM2 or opening a port.")
    return parser.parse_args()


def image_data_url(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class HandMaskService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.workspace = args.workspace.resolve()
        self.sam2_frame_dir = (args.sam2_frame_dir or self.workspace / "outputs/00_rgb_frames/sam2_jpeg").resolve()
        self.display_frame_dir = (args.display_frame_dir or self.workspace / "outputs/00_rgb_frames/right_rgb_png").resolve()
        self.output_dir = (args.output_dir or self.workspace / "outputs/02_hand_masks").resolve()
        self.display_frames = list_display_frames(self.display_frame_dir)
        sam2_frames = list_display_frames(self.sam2_frame_dir)
        if len(sam2_frames) != len(self.display_frames):
            raise ValueError(f"SAM2/display frame mismatch: {len(sam2_frames)} vs {len(self.display_frames)}")
        with Image.open(self.display_frames[0]) as image:
            self.width, self.height = image.size
        self.predictor = None
        self.state = None
        self.prompts: dict[str, dict] = {}
        self.conditioning_frames: set[int] = set()
        self.lock = threading.Lock()

    def load_model(self) -> None:
        print("Loading SAM2 video predictor...", flush=True)
        self.predictor = build_video_predictor(
            self.args.sam2_root,
            self.args.sam2_config,
            self.args.sam2_checkpoint,
            self.args.device,
        )
        self.state = self.predictor.init_state(
            str(self.sam2_frame_dir),
            offload_video_to_cpu=self.args.offload_video_to_cpu,
            offload_state_to_cpu=self.args.offload_state_to_cpu,
        )
        print("SAM2 video predictor loaded.", flush=True)

    def meta(self) -> dict:
        return {
            "frame_count": len(self.display_frames),
            "width": self.width,
            "height": self.height,
            "fps": self.args.fps,
            "output_dir": str(self.output_dir),
        }

    def preview(self, payload: dict) -> dict:
        frame_index = int(payload["frame_index"])
        object_id = str(payload.get("object_id") or "hand").strip()
        if not 0 <= frame_index < len(self.display_frames):
            raise IndexError(frame_index)
        positive = payload.get("positive_points") or []
        negative = payload.get("negative_points") or []
        if not positive:
            raise ValueError("Hand mask requires at least one human-selected positive point")
        with self.lock:
            masks = add_points_or_box(
                self.predictor,
                self.state,
                frame_index=frame_index,
                object_id=object_id,
                positive_points=positive,
                negative_points=negative,
            )
        mask = masks[object_id]
        prompt_key = f"{object_id}:{frame_index}"
        self.prompts[prompt_key] = {
            "object_id": object_id,
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
            "object_id": object_id,
            "area_pixels": int(mask.sum()),
            "overlay_data_url": image_data_url(canvas),
        }

    def propagate(self) -> dict:
        if not self.conditioning_frames:
            raise ValueError("No human point prompts saved. Preview at least one prompted frame first.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.output_dir / "hand_prompts.json",
            {
                "policy": "Human-selected positive/negative points; no automatic hand box",
                "prompts": list(self.prompts.values()),
            },
        )
        with self.lock:
            results = propagate_bidirectional(
                self.predictor,
                self.state,
                sorted(self.conditioning_frames),
            )
        summary = save_propagation_outputs(
            results,
            self.display_frames,
            self.output_dir,
            fps=self.args.fps,
            video_stem="hand_mask",
        )
        qc = mask_sequence_qc(summary, max_area_ratio=4.5, allow_empty=True)
        write_json(self.output_dir / "hand_mask_qc.json", qc)
        pipeline_state_warning = None
        try:
            update_stage_state(
                self.workspace / "pipeline_state.json",
                "02_hand_masks",
                "completed" if qc["passed"] else "needs_revision",
                inputs=[str(self.display_frame_dir), str(self.output_dir / "hand_prompts.json")],
                outputs=[str(self.output_dir)],
                notes=(
                    f"Human-point SAM2 propagation passed QC for {summary['frame_count']} frames."
                    if qc["passed"]
                    else f"Hand-mask propagation needs revision: empty={qc['empty_frames']} jumps={qc['abrupt_area_change_frames']}"
                ),
            )
        except KeyError as exc:
            pipeline_state_warning = str(exc)
            print(f"[hand-mask-ui] pipeline_state warning: {exc}", flush=True)
        return {
            "frame_count": summary["frame_count"],
            "mask_video": summary["mask_video"],
            "manifest": str(self.output_dir / "propagation_manifest.json"),
            "qc": qc,
            "pipeline_state_warning": pipeline_state_warning,
        }


def handler_factory(service: HandMaskService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[hand-mask-ui] {fmt % args}", flush=True)

        def send_bytes(self, content: bytes, content_type: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def send_json(self, payload: dict, status=200):
            self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
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
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if path == "/api/preview":
                    self.send_json(service.preview(payload))
                elif path == "/api/propagate":
                    self.send_json(service.propagate())
                else:
                    self.send_json({"error": "not found"}, 404)
            except Exception as exc:
                traceback.print_exc()
                self.send_json({"error": str(exc)}, 500)

    return Handler


def main() -> int:
    args = parse_args()
    service = HandMaskService(args)
    print(json.dumps(service.meta(), ensure_ascii=False, indent=2))
    if args.check:
        print("Stage 02 path check passed; SAM2 was not loaded and no server was started.")
        return 0
    service.load_model()
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(service))
    print_server_addresses(args.host, args.port, "Interactive hand-mask server")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping hand-mask server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
