#!/usr/bin/env python3
"""Visualize the depth40 microwave, bottle, and hands in the common C0 frame."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--body-mesh', type=Path, required=True)
    p.add_argument('--door-mesh', type=Path, required=True)
    p.add_argument('--joint-json', type=Path, required=True)
    p.add_argument('--door-angles', type=Path, required=True)
    p.add_argument('--door-contact-manifest', type=Path, required=True)
    p.add_argument('--bottle-mesh', type=Path, required=True)
    p.add_argument('--bottle-poses', type=Path, required=True)
    p.add_argument('--bottle-contact-manifest', type=Path, required=True)
    p.add_argument('--hand-manifest', type=Path, default=None)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8102)
    p.add_argument('--initial-frame', type=int, default=8)
    p.add_argument('--fps', type=float, default=4.1666666667)
    return p.parse_args()


def load_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])
    return mesh


def mesh_handle(scene, name, mesh, color, opacity):
    return scene.add_mesh_simple(
        name,
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
        color=color,
        opacity=opacity,
        side='double',
    )


def wxyz(rotation_matrix):
    quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return np.asarray(
        [quaternion_xyzw[3], quaternion_xyzw[0], quaternion_xyzw[1], quaternion_xyzw[2]],
        dtype=np.float32,
    )


def build_hand_cache(hand_frames, frame_count):
    vertices_by_side = {}
    faces_by_side = {}
    source_frames_by_side = {}
    for side in ('left', 'right'):
        vertices = []
        source_frames = []
        previous_vertices = None
        previous_source_frame = None
        faces = None
        for frame in range(frame_count):
            hand_path = hand_frames[frame].get(f'{side}_hand_C0')
            if hand_path is not None and Path(hand_path).exists():
                mesh = load_mesh(Path(hand_path))
                current_vertices = np.asarray(mesh.vertices, dtype=np.float32)
                current_faces = np.asarray(mesh.faces, dtype=np.uint32)
                if faces is None:
                    faces = current_faces
                elif current_faces.shape != faces.shape or not np.array_equal(current_faces, faces):
                    raise ValueError(f'{side} hand topology changes at frame {frame}')
                previous_vertices = current_vertices
                previous_source_frame = frame
            vertices.append(previous_vertices)
            source_frames.append(previous_source_frame)
        if previous_vertices is None or faces is None:
            raise ValueError(f'No {side} hand mesh is available')
        first_available = next(index for index, value in enumerate(vertices) if value is not None)
        for frame in range(first_available):
            vertices[frame] = vertices[first_available]
            source_frames[frame] = first_available
        vertices_by_side[side] = vertices
        faces_by_side[side] = faces
        source_frames_by_side[side] = source_frames
    return vertices_by_side, faces_by_side, source_frames_by_side


def main():
    args = parse_args()
    workspace = args.workspace.resolve()
    body = load_mesh(args.body_mesh.resolve())
    door = load_mesh(args.door_mesh.resolve())
    bottle = load_mesh(args.bottle_mesh.resolve())
    bottle_poses = np.load(args.bottle_poses.resolve()).astype(np.float64)
    door_angles = np.load(args.door_angles.resolve()).astype(np.float64)
    joint = json.loads(args.joint_json.read_text())['joints'][0]
    origin = np.asarray(joint.get('origin_C0', joint.get('origin_xyz')), dtype=np.float64)
    axis = np.asarray(joint.get('axis_C0', joint.get('axis_xyz')), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    door_contact = json.loads(args.door_contact_manifest.read_text())
    bottle_contact = json.loads(args.bottle_contact_manifest.read_text())
    hand_manifest_path = args.hand_manifest or workspace / 'outputs/09_egoforce/dynamic_manifest.json'
    hand_frames = json.loads(hand_manifest_path.read_text())['frames']
    frame_count = min(len(bottle_poses), len(door_angles), len(hand_frames))
    initial_frame = int(np.clip(args.initial_frame, 0, frame_count - 1))
    hand_vertices, hand_faces, hand_source_frames = build_hand_cache(hand_frames, frame_count)

    server = viser.ViserServer(host=args.host, port=args.port)
    scene, gui = server.scene, server.gui
    scene.set_up_direction('-y')
    c0_handle = scene.add_frame('/C0', axes_length=0.08, axes_radius=0.002)
    c0_handle.visible = False

    body_handle = mesh_handle(scene, '/objects/microwave/body', body, (170, 175, 185), 0.26)
    door_handle = mesh_handle(scene, '/objects/microwave/door', door, (65, 130, 235), 0.62)
    bottle_handle = mesh_handle(scene, '/objects/bottle', bottle, (235, 135, 45), 0.96)
    right_hand = trimesh.Trimesh(
        vertices=hand_vertices['right'][initial_frame], faces=hand_faces['right'], process=False
    )
    left_hand = trimesh.Trimesh(
        vertices=hand_vertices['left'][initial_frame], faces=hand_faces['left'], process=False
    )
    right_hand_handle = mesh_handle(scene, '/hands/right', right_hand, (245, 170, 100), 0.9)
    left_hand_handle = mesh_handle(scene, '/hands/left', left_hand, (120, 220, 145), 0.86)

    hinge_axis_handle = scene.add_line_segments(
        '/debug/hinge_axis',
        np.asarray([[origin - axis * 0.25, origin + axis * 0.25]], dtype=np.float32),
        colors=(255, 50, 50),
        line_width=4,
    )
    hinge_origin_handle = scene.add_icosphere(
        '/debug/hinge_origin', radius=0.01, color=(255, 50, 50), position=origin, subdivisions=2
    )
    bottle_point = np.asarray(bottle_contact['contact_surface']['point_C0'], dtype=np.float64)
    bottle_normal = np.asarray(bottle_contact['contact_surface']['normal_C0_toward_object'], dtype=np.float64)
    bottle_contact_handle = scene.add_icosphere(
        '/debug/bottle_contact_point', radius=0.009, color=(255, 230, 30), position=bottle_point, subdivisions=2
    )
    bottle_normal_handle = scene.add_line_segments(
        '/debug/bottle_contact_normal',
        np.asarray([[bottle_point, bottle_point + bottle_normal * 0.10]], dtype=np.float32),
        colors=(255, 230, 30),
        line_width=4,
    )
    for handle in [hinge_axis_handle, hinge_origin_handle, bottle_contact_handle, bottle_normal_handle]:
        handle.visible = False

    frame_slider = gui.add_slider('Frame', min=0, max=frame_count - 1, step=1, initial_value=initial_frame)
    previous_button = gui.add_button('Previous')
    next_button = gui.add_button('Next')
    play = gui.add_checkbox('Play', initial_value=False)
    fps = gui.add_slider('FPS', min=1.0, max=15.0, step=0.5, initial_value=float(args.fps))
    show_body = gui.add_checkbox('Microwave body', initial_value=True)
    show_door = gui.add_checkbox('Microwave door', initial_value=True)
    show_bottle = gui.add_checkbox('Bottle', initial_value=True)
    show_right = gui.add_checkbox('Right hand', initial_value=True)
    show_left = gui.add_checkbox('Left hand', initial_value=True)
    show_coordinates = gui.add_checkbox('C0 coordinates', initial_value=False)
    show_hinge = gui.add_checkbox('Door/body contact debug', initial_value=False)
    show_bottle_contact = gui.add_checkbox('Bottle contact debug', initial_value=False)
    status = gui.add_markdown('')
    current = initial_frame

    center = np.vstack([body.vertices, door.vertices, bottle.vertices]).mean(axis=0)
    bounds = np.vstack([body.bounds, door.bounds, bottle.bounds])
    extent = max(float(np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0))), 0.6)

    @server.on_client_connect
    def _on_connect(client):
        client.camera.position = center + extent * np.asarray([0.60, -0.35, -1.10])
        client.camera.look_at = center
        client.camera.up_direction = np.asarray([0.0, -1.0, 0.0])

    def update(frame):
        nonlocal current
        current = int(np.clip(frame, 0, frame_count - 1))
        door_rotation = Rotation.from_rotvec(axis * float(door_angles[current])).as_matrix()
        door_handle.wxyz = wxyz(door_rotation)
        door_handle.position = (origin - door_rotation @ origin).astype(np.float32)
        bottle_pose = bottle_poses[current]
        bottle_handle.wxyz = wxyz(bottle_pose[:3, :3])
        bottle_handle.position = bottle_pose[:3, 3].astype(np.float32)
        right_hand_handle.vertices = hand_vertices['right'][current]
        left_hand_handle.vertices = hand_vertices['left'][current]
        body_handle.visible = bool(show_body.value)
        door_handle.visible = bool(show_door.value)
        bottle_handle.visible = bool(show_bottle.value)
        right_hand_handle.visible = bool(show_right.value)
        left_hand_handle.visible = bool(show_left.value)
        c0_handle.visible = bool(show_coordinates.value)
        hinge_axis_handle.visible = hinge_origin_handle.visible = bool(show_hinge.value)
        bottle_contact_handle.visible = bottle_normal_handle.visible = bool(show_bottle_contact.value)
        door_record = door_contact['frames'][current]
        bottle_state = 'moving' if 4 <= current <= 17 else 'fixed'
        fallback_sides = [
            f"{side}=frame {hand_source_frames[side][current]}"
            for side in ('left', 'right')
            if hand_source_frames[side][current] != current
        ]
        fallback_status = '' if not fallback_sides else f" | hand fallback `{', '.join(fallback_sides)}`"
        status.content = (
            f"**Frame {current}** | door `{door_record['output_angle_deg']:.2f}°` "
            f"| door/body clearance `{door_record['door_body_clearance_m'] * 1000:.2f} mm` "
            f"| bottle `{bottle_state}`{fallback_status}"
        )

    @frame_slider.on_update
    def _(_event):
        update(int(frame_slider.value))

    @previous_button.on_click
    def _(_event):
        frame_slider.value = (current - 1) % frame_count

    @next_button.on_click
    def _(_event):
        frame_slider.value = (current + 1) % frame_count

    for control in [show_body, show_door, show_bottle, show_right, show_left, show_coordinates, show_hinge, show_bottle_contact]:
        @control.on_update
        def _(_event, control=control):
            update(current)

    update(current)
    print(f'microwave+bottle full scene viser running at http://localhost:{args.port}', flush=True)
    while True:
        if play.value:
            time.sleep(1.0 / max(float(fps.value), 1e-6))
            frame_slider.value = (current + 1) % frame_count
        else:
            time.sleep(0.05)


if __name__ == '__main__':
    main()
