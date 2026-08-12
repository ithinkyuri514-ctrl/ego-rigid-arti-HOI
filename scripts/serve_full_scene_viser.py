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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--iou-output', type=Path, required=True)
    parser.add_argument('--mouse-mesh', type=Path, required=True)
    parser.add_argument('--mouse-poses', type=Path, required=True)
    parser.add_argument('--base-mesh', type=Path, required=True)
    parser.add_argument('--lid-mesh', type=Path, required=True)
    parser.add_argument('--joint-json', type=Path, required=True)
    parser.add_argument('--poses', type=Path, required=True)
    parser.add_argument('--hand-manifest', type=Path, default=None)
    parser.add_argument('--optimized-hand-dir', type=Path, default=None)
    parser.add_argument('--port', type=int, default=8098)
    parser.add_argument('--initial-frame', type=int, default=6)
    return parser.parse_args()


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        meshes = [item for item in mesh.geometry.values() if isinstance(item, trimesh.Trimesh)]
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'Unsupported mesh: {path}')
    return mesh


def transform_vertices(vertices, origin, axis, angle):
    rotation = Rotation.from_rotvec(axis * float(angle)).as_matrix()
    return ((vertices - origin) @ rotation.T + origin).astype(np.float32)


def apply_transform(vertices, transform):
    return (vertices @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def read_joint(path):
    joint = json.loads(path.read_text())['joints'][0]
    origin = np.asarray(joint.get('origin_C0', joint.get('origin_xyz')), dtype=np.float64)
    axis = np.asarray(joint.get('axis_C0', joint.get('axis_xyz')), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    return origin, axis


def mesh_handle(scene, name, mesh, color, opacity=0.9):
    return scene.add_mesh_simple(
        name,
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        side='double',
    )


def main():
    args = parse_args()
    workspace = args.workspace.resolve()
    iou_payload = json.loads((args.iou_output / 'articulate_iou_optimization.json').read_text())
    iou_records = {int(item['frame_index']): item for item in iou_payload['frames']}
    mouse_poses = np.load(args.mouse_poses).astype(np.float64)
    base = load_mesh(args.base_mesh)
    lid = load_mesh(args.lid_mesh)
    mouse = load_mesh(args.mouse_mesh)
    origin, axis = read_joint(args.joint_json)
    hand_manifest_path = args.hand_manifest or workspace / 'outputs/09_egoforce/dynamic_manifest.json'
    hand_frames = json.loads(hand_manifest_path.read_text())['frames']
    optimized_hand_dir = args.optimized_hand_dir or workspace / 'outputs/11_hand_object_optimization_dashscope/final/optimized_C0'
    frame_count = min(len(mouse_poses), len(hand_frames))
    initial_frame = int(np.clip(args.initial_frame, 0, frame_count - 1))

    server = viser.ViserServer(host='0.0.0.0', port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction('-y')

    c0_handle = scene.add_frame('/C0', axes_length=0.07, axes_radius=0.002)
    c0_handle.visible = False
    hinge_axis_handle = scene.add_line_segments(
        '/debug/hinge_axis',
        np.asarray([[origin - axis * 0.22, origin + axis * 0.22]], dtype=np.float32),
        colors=(255, 60, 60),
        line_width=4,
    )
    hinge_origin_handle = scene.add_icosphere(
        '/debug/hinge_origin', radius=0.009, color=(255, 60, 60), position=origin, subdivisions=2
    )
    hinge_axis_handle.visible = False
    hinge_origin_handle.visible = False

    base_handle = mesh_handle(scene, '/objects/laptop/base', base, (145, 155, 175), 0.82)
    lid_handle = mesh_handle(scene, '/objects/laptop/lid', lid, (55, 130, 255), 0.92)
    mouse_handle = mesh_handle(scene, '/objects/mouse', mouse, (190, 125, 55), 0.94)

    right_hand = load_mesh(Path(hand_frames[initial_frame]['right_hand_C0']))
    left_hand = load_mesh(Path(hand_frames[initial_frame]['left_hand_C0']))
    right_hand_handle = mesh_handle(scene, '/hands/right', right_hand, (245, 170, 100), 0.9)
    left_hand_handle = mesh_handle(scene, '/hands/left', left_hand, (120, 220, 145), 0.86)

    track_points = np.load(
        workspace / 'outputs/10_articulate_tracking_vda_track0/link_14/upper_left_track_3d_C0.npy'
    )[:, 0]
    finite_segments = np.isfinite(track_points[:-1]).all(axis=1) & np.isfinite(track_points[1:]).all(axis=1)
    track_segments = np.stack([track_points[:-1][finite_segments], track_points[1:][finite_segments]], axis=1).astype(np.float32)
    track_path_handle = scene.add_line_segments(
        '/debug/lid_track_path', track_segments, colors=(255, 220, 0), line_width=3
    )
    initial_track = track_points[initial_frame]
    if not np.isfinite(initial_track).all():
        initial_track = track_points[np.flatnonzero(np.isfinite(track_points).all(axis=1))[0]]
    track_point_handle = scene.add_icosphere(
        '/debug/lid_track_point', radius=0.01, color=(255, 220, 0), position=initial_track, subdivisions=2
    )
    track_path_handle.visible = False
    track_point_handle.visible = False

    frame_slider = gui.add_slider('Frame', min=0, max=frame_count - 1, step=1, initial_value=initial_frame)
    previous_button = gui.add_button('Previous')
    next_button = gui.add_button('Next')
    play = gui.add_checkbox('Play', initial_value=False)
    fps = gui.add_slider('FPS', min=1.0, max=15.0, step=1.0, initial_value=5.0)
    show_laptop = gui.add_checkbox('Laptop', initial_value=True)
    show_mouse = gui.add_checkbox('Mouse', initial_value=True)
    show_right = gui.add_checkbox('Right hand', initial_value=True)
    show_left = gui.add_checkbox('Left hand', initial_value=True)
    show_coordinates = gui.add_checkbox('C0 coordinates', initial_value=False)
    show_hinge = gui.add_checkbox('Hinge axis', initial_value=False)
    show_track = gui.add_checkbox('Lid track', initial_value=False)
    status = gui.add_markdown('')
    current = int(frame_slider.value)

    center = np.vstack([
        np.asarray(base.vertices),
        np.asarray(lid.vertices),
        np.asarray(mouse.vertices),
        np.asarray(right_hand.vertices),
        np.asarray(left_hand.vertices),
    ]).mean(axis=0)
    bounds = np.vstack([base.bounds, lid.bounds, mouse.bounds, right_hand.bounds, left_hand.bounds])
    extent = max(float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0))), 0.35)

    @server.on_client_connect
    def _on_connect(client):
        client.camera.position = center + extent * np.asarray([0.30, -0.32, -0.85])
        client.camera.look_at = center
        client.camera.up_direction = np.asarray([0.0, -1.0, 0.0])

    def load_hand(frame, side):
        optimized = optimized_hand_dir / f'frame_{frame:06d}' / f'{side}_hand_adaptive_contact_C0.obj'
        if optimized.exists():
            return load_mesh(optimized)
        return load_mesh(Path(hand_frames[frame][f'{side}_hand_C0']))

    def update(frame):
        nonlocal current
        current = int(np.clip(frame, 0, frame_count - 1))
        record = iou_records.get(current)
        if record is None:
            record = iou_records[min(iou_records, key=lambda index: abs(index - current))]
        lid_handle.vertices = transform_vertices(
            np.asarray(lid.vertices), origin, axis, np.deg2rad(record['optimized_angle_deg'])
        )
        mouse_handle.vertices = apply_transform(np.asarray(mouse.vertices), mouse_poses[current])
        right_hand_handle.vertices = np.asarray(load_hand(current, 'right').vertices, dtype=np.float32)
        left_hand_handle.vertices = np.asarray(load_hand(current, 'left').vertices, dtype=np.float32)
        if np.isfinite(track_points[current]).all():
            track_point_handle.position = track_points[current]
        base_handle.visible = lid_handle.visible = bool(show_laptop.value)
        mouse_handle.visible = bool(show_mouse.value)
        right_hand_handle.visible = bool(show_right.value)
        left_hand_handle.visible = bool(show_left.value)
        c0_handle.visible = bool(show_coordinates.value)
        hinge_axis_handle.visible = hinge_origin_handle.visible = bool(show_hinge.value)
        track_path_handle.visible = track_point_handle.visible = bool(show_track.value)
        status.content = (
            f"**Frame {current}** | lid `{record['optimized_angle_deg']:.2f}°` "
            f"| contact `{record.get('status', '')}`"
        )

    @frame_slider.on_update
    def _(_event):
        update(int(frame_slider.value))

    @previous_button.on_click
    def _(_event):
        frame_slider.value = (current - 1) % frame_count
        update(int(frame_slider.value))

    @next_button.on_click
    def _(_event):
        frame_slider.value = (current + 1) % frame_count
        update(int(frame_slider.value))

    @play.on_update
    def _(_event):
        pass

    for control in [show_laptop, show_mouse, show_right, show_left, show_coordinates, show_hinge, show_track]:
        @control.on_update
        def _(_event, control=control):
            update(current)

    update(current)
    print(f'full scene viser running at http://localhost:{args.port}', flush=True)
    while True:
        if play.value:
            time.sleep(1.0 / max(float(fps.value), 1e-6))
            frame_slider.value = (current + 1) % frame_count
            update(int(frame_slider.value))
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
