#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--iou-output', type=Path, required=True)
    p.add_argument('--track-output', type=Path, required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--base-mesh', type=Path, default=None)
    p.add_argument('--joint-json', type=Path, required=True)
    p.add_argument('--poses', type=Path, required=True)
    p.add_argument('--start-frame', type=int, default=6)
    p.add_argument('--end-frame', type=int, default=15)
    p.add_argument('--port', type=int, default=8098)
    return p.parse_args()


def quat_wxyz(matrix):
    q_xyzw = Rotation.from_matrix(np.asarray(matrix)[:3, :3]).as_quat()
    return np.asarray([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)


def transform_vertices(vertices, origin, axis, angle):
    rot = Rotation.from_rotvec(axis * float(angle)).as_matrix()
    return ((vertices - origin) @ rot.T + origin).astype(np.float32)


def read_joint(path):
    payload = json.loads(path.read_text())
    joints = payload.get('joints', [])
    joint = payload if 'origin_C0' in payload else next(x for x in joints if x.get('name') == 'joint_15_14')
    origin = np.asarray(joint.get('origin_C0', joint.get('origin_xyz')), dtype=np.float32)
    axis = np.asarray(joint.get('axis_C0', joint.get('axis_xyz')), dtype=np.float32)
    axis /= np.linalg.norm(axis)
    return origin, axis


def main():
    args = args_parser()
    iou_data = json.loads((args.iou_output / 'articulate_iou_optimization.json').read_text())
    records = {int(x['frame_index']): x for x in iou_data['frames']}
    camera = json.loads((args.workspace / 'outputs/00_rgb_frames/camera.json').read_text())
    poses = np.load(args.poses)['T_C0_from_Ct']
    points = np.load(args.track_output / 'upper_left_track_3d_C0.npy')[:, 0]
    mesh = trimesh.load(args.mesh, force='mesh', process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    base_path = args.base_mesh or args.mesh.with_name('part_15.obj')
    base_mesh = trimesh.load(base_path, force='mesh', process=False)
    base_vertices = np.asarray(base_mesh.vertices, dtype=np.float32)
    base_faces = np.asarray(base_mesh.faces, dtype=np.uint32)
    origin, axis = read_joint(args.joint_json)
    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    server.scene.set_up_direction('+z')
    server.scene.add_grid('/grid', width=2.0, height=2.0, cell_size=0.05, plane='xy')
    server.scene.add_mesh_simple('/laptop/base', base_vertices, base_faces, color=(150, 155, 165), opacity=0.72, side='double')
    server.scene.add_line_segments('/joint/axis', points=np.asarray([[origin - axis * 0.22, origin + axis * 0.22]]), colors=(255, 40, 40), line_width=5)
    server.scene.add_icosphere('/joint/origin', radius=0.012, color=(255, 40, 40), position=origin, subdivisions=2)
    server.scene.add_label('/joint/label', text='hinge axis', position=origin + np.array([0, -0.03, 0], dtype=np.float32), font_size_mode='screen', font_screen_scale=0.8)
    initial_handles = {}
    optimized_handles = {}
    point_handles = {}
    camera_handles = {}
    reference = points[args.start_frame]
    for frame in range(args.start_frame, args.end_frame + 1):
        record = records[frame]
        initial_vertices = transform_vertices(vertices, origin, axis, np.deg2rad(record['lifted_initial_angle_deg']))
        optimized_vertices = transform_vertices(vertices, origin, axis, np.deg2rad(record['optimized_angle_deg']))
        initial_handles[frame] = server.scene.add_mesh_simple(f'/frames/{frame:06d}/initial', initial_vertices, faces, color=(90, 210, 110), opacity=0.28, side='double')
        optimized_handles[frame] = server.scene.add_mesh_simple(f'/frames/{frame:06d}/optimized', optimized_vertices, faces, color=(55, 130, 255), opacity=0.9, side='double')
        point = points[frame] if np.isfinite(points[frame]).all() else reference
        point_handles[frame] = server.scene.add_icosphere(f'/frames/{frame:06d}/lift_point', radius=0.014, color=(255, 230, 0), position=point, subdivisions=2)
        camera_handles[frame] = server.scene.add_frame(f'/frames/{frame:06d}/camera', axes_length=0.06, axes_radius=0.003, position=poses[frame][:3, 3], wxyz=quat_wxyz(poses[frame]))
        initial_handles[frame].visible = frame == args.start_frame
        optimized_handles[frame].visible = frame == args.start_frame
        point_handles[frame].visible = frame == args.start_frame
        camera_handles[frame].visible = frame == args.start_frame
    track_points = np.asarray(points[args.start_frame:args.end_frame + 1], dtype=np.float32)
    track_segments = np.stack([track_points[:-1], track_points[1:]], axis=1)
    server.scene.add_line_segments('/track/path', points=track_segments, colors=(255, 230, 0), line_width=3)
    frame_slider = server.gui.add_slider('frame', min=args.start_frame, max=args.end_frame, step=1, initial_value=args.start_frame)
    show_initial = server.gui.add_checkbox('show lift initial', initial_value=True)
    show_optimized = server.gui.add_checkbox('show IoU optimized', initial_value=True)
    show_track = server.gui.add_checkbox('show 3D lift point', initial_value=True)
    status = server.gui.add_markdown('')

    def update(frame):
        for value, handle in initial_handles.items():
            handle.visible = value == frame and bool(show_initial.value)
        for value, handle in optimized_handles.items():
            handle.visible = value == frame and bool(show_optimized.value)
        for value, handle in point_handles.items():
            handle.visible = value == frame and bool(show_track.value)
        for value, handle in camera_handles.items():
            handle.visible = value == frame
        record = records[frame]
        status.content = f"**frame {frame}**  \n3D initial: `{record['lifted_initial_angle_deg']:.2f}°`  \nIoU optimized: `{record['optimized_angle_deg']:.2f}°`  \nIoU: `{record['initial_mesh_iou']:.3f} → {record['optimized_mesh_iou']:.3f}`"

    @frame_slider.on_update
    def _(_event):
        update(int(frame_slider.value))

    @show_initial.on_update
    def _(_event):
        update(int(frame_slider.value))

    @show_optimized.on_update
    def _(_event):
        update(int(frame_slider.value))

    @show_track.on_update
    def _(_event):
        update(int(frame_slider.value))

    update(args.start_frame)
    print(f'viser server running at http://localhost:{args.port}', flush=True)
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
