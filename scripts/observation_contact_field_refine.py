#!/usr/bin/env python3
"""Observation-driven contact-field refinement for a tracked articulated part."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import diags, eye, coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hand-manifest',type=Path,required=True); p.add_argument('--part-mesh',type=Path,required=True)
    p.add_argument('--joint-json',type=Path,required=True); p.add_argument('--angle-json',type=Path,required=True)
    p.add_argument('--hand-mask-dir',type=Path,required=True); p.add_argument('--object-mask-dir',type=Path,required=True)
    p.add_argument('--depth-dir',type=Path,required=True); p.add_argument('--poses',type=Path,required=True)
    p.add_argument('--max-local-mm',type=float,default=5.0); p.add_argument('--contact-gap-mm',type=float,default=1.0)
    p.add_argument('--max-pairs',type=int,default=32)
    return p.parse_args()

def load_mesh(path):
    value=trimesh.load(path,force='mesh',process=False)
    if isinstance(value,trimesh.Scene): value=trimesh.util.concatenate(tuple(value.geometry.values()))
    return value

def transform(points,matrix): return points @ matrix[:3,:3].T + matrix[:3,3]

def read_joint(path):
    joint=json.loads(path.read_text())['joints'][0]
    origin=np.asarray(joint['origin_C0'],dtype=np.float64); axis=np.asarray(joint['axis_C0'],dtype=np.float64)
    return origin,axis/np.linalg.norm(axis)

def rotate_about(points,origin,axis,angle):
    return transform(points,trimesh.transformations.rotation_matrix(angle,axis,origin)[:3,:])

def graph_laplacian(faces,count):
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]),axis=0)
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    rows=np.concatenate((edges[:,0],edges[:,1])); cols=np.concatenate((edges[:,1],edges[:,0]))
    adjacency=coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(count,count)).tocsr()
    return diags(np.asarray(adjacency.sum(axis=1)).ravel())-adjacency

def make_sdf(object_mesh,pitch=0.0025,padding=0.04):
    voxel=object_mesh.voxelized(pitch=pitch).fill(); pad=max(3,int(np.ceil(padding/pitch)))
    occupancy=np.pad(np.asarray(voxel.matrix,dtype=bool),pad)
    outside=distance_transform_edt(~occupancy); inside=distance_transform_edt(occupancy)
    sdf=np.where(occupancy,-(inside-.5),outside-.5).astype(np.float32)*pitch
    origin=np.asarray(voxel.transform[:3,3],dtype=np.float64)-pad*pitch
    return sdf,origin,pitch

def query_sdf(sdf,origin,pitch,points):
    xyz=np.rint((points-origin)/pitch).astype(np.int64); valid=np.all((xyz>=0)&(xyz<np.asarray(sdf.shape)),axis=1)
    result=np.full(len(points),float(sdf.max()),dtype=np.float64)
    result[valid]=sdf[xyz[valid,0],xyz[valid,1],xyz[valid,2]]
    return result

def query_gradient(gradient, origin, pitch, points):
    xyz=np.rint((points-origin)/pitch).astype(np.int64); xyz=np.clip(xyz, 0, np.asarray(gradient.shape[:3])-1)
    values=gradient[xyz[:,0],xyz[:,1],xyz[:,2]]
    return values/np.maximum(np.linalg.norm(values,axis=1,keepdims=True),1e-8)

def main():
    a=parse_args(); ws=a.workspace.resolve(); out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True); opt=out/'optimized_C0'; opt.mkdir(exist_ok=True)
    manifest=json.loads(a.hand_manifest.read_text()); camera=json.loads((ws/'outputs/00_rgb_frames/camera.json').read_text()); intr=camera['rgb_intrinsics_right']; poses=np.load(a.poses)['T_C0_from_Ct']
    origin,axis=read_joint(a.joint_json); angle_data=json.loads(a.angle_json.read_text())['frames']; angle_by_frame={int(x['frame_index']):np.deg2rad(float(x['optimized_angle_deg'])) for x in angle_data}
    part=load_mesh(a.part_mesh); part_samples,_=trimesh.sample.sample_surface_even(part,5000); smooth=None; records=[]
    for entry in manifest['frames']:
        frame=int(entry['frame'])
        with np.load(Path(entry['geometry_C0_npz'])) as data:
            hand=data['hand_vertices'][1].astype(np.float64); faces=data['right_hand_faces'].astype(np.int64); arm=data['arm_vertices'][1].astype(np.float64); arm_faces=data['arm_faces'].astype(np.int64)
        if smooth is None: smooth=graph_laplacian(faces,len(hand))
        angle=angle_by_frame.get(frame,0.0); lid=rotate_about(part_samples,origin,axis,angle); dynamic_part=trimesh.Trimesh(rotate_about(np.asarray(part.vertices),origin,axis,angle),np.asarray(part.faces),process=False); sdf,sdf_origin,sdf_pitch=make_sdf(dynamic_part); sdf_gradient=np.stack(np.gradient(sdf,sdf_pitch),axis=-1); tree=cKDTree(lid)
        hand_ct=transform(hand,np.linalg.inv(poses[frame])); z=hand_ct[:,2]; u=intr['fx']*hand_ct[:,0]/np.maximum(z,1e-8)+intr['cx']; v=intr['fy']*hand_ct[:,1]/np.maximum(z,1e-8)+intr['cy']; uv=np.rint(np.column_stack((u,v))).astype(np.int64)
        hand_mask=np.asarray(Image.open(a.hand_mask_dir/f'{frame:06d}.png'))>127; object_mask=np.asarray(Image.open(a.object_mask_dir/f'{frame:06d}.png'))>127; overlap=cv2.dilate((hand_mask&object_mask).astype(np.uint8),np.ones((9,9),np.uint8))>0; depth=np.load(a.depth_dir/f'{frame:06d}.npy')
        inside=(z>.05)&(uv[:,0]>=0)&(uv[:,0]<depth.shape[1])&(uv[:,1]>=0)&(uv[:,1]<depth.shape[0]); observed=np.zeros(len(hand),dtype=bool); observed[inside]=overlap[uv[inside,1],uv[inside,0]]; candidate=np.flatnonzero(observed)
        distances=np.full(len(hand),np.inf); targets=np.zeros_like(hand); signed=query_sdf(sdf,sdf_origin,sdf_pitch,hand)
        if len(candidate):
            distances[candidate],nearest=tree.query(hand[candidate],k=1); targets[candidate]=lid[nearest]; candidate=candidate[(distances[candidate]<.035)&(signed[candidate]>-.001)]
            if len(candidate)>a.max_pairs: candidate=candidate[np.argsort(distances[candidate])[:a.max_pairs]]
        desired=np.zeros_like(hand); confidence=np.zeros(len(hand))
        if len(candidate):
            delta=targets[candidate]-hand[candidate]; norms=np.linalg.norm(delta,axis=1,keepdims=True); desired[candidate]=delta*np.maximum(norms-a.contact_gap_mm/1000.,0.)/np.maximum(norms,1e-8); confidence[candidate]=np.exp(-distances[candidate]/.012)
        penetrating=np.flatnonzero(signed<-.001)
        if len(penetrating):
            pen_gradient = query_gradient(sdf_gradient, sdf_origin, sdf_pitch, hand[penetrating])
            desired[penetrating] = pen_gradient * (np.abs(signed[penetrating,None]) + .001)
            confidence[penetrating] = np.maximum(confidence[penetrating], 2.0)
        global_shift=np.median(desired[candidate],axis=0) if len(candidate) else np.zeros(3); global_shift*=min(1.,.004/max(np.linalg.norm(global_shift),1e-8)); target=desired-global_shift; weights=confidence*20.; system=diags(weights)+1.5*smooth+.5*eye(len(hand),format='csr'); local=np.column_stack([spsolve(system,weights*target[:,i]) for i in range(3)]); norms=np.linalg.norm(local,axis=1,keepdims=True); local*=np.minimum(1.,(a.max_local_mm/1000.)/np.maximum(norms,1e-8)); result=hand+global_shift+local
        frame_dir=opt/f'frame_{frame:06d}'; frame_dir.mkdir(exist_ok=True); trimesh.Trimesh(result,faces,process=False).export(frame_dir/'right_hand_adaptive_contact_C0.obj'); trimesh.Trimesh(arm+global_shift,arm_faces,process=False).export(frame_dir/'right_arm_adaptive_contact_C0.obj'); np.savez_compressed(frame_dir/'observation_contact_field_C0.npz',hand_vertices=result.astype(np.float32),hand_faces=faces,arm_vertices=(arm+global_shift).astype(np.float32),arm_faces=arm_faces,contact_vertex_indices=candidate,contact_target_points_C0=targets[candidate].astype(np.float32),signed_distance_after_m=query_sdf(sdf,sdf_origin,sdf_pitch,result).astype(np.float32))
        after=query_sdf(sdf,sdf_origin,sdf_pitch,result); records.append({'frame':frame,'candidate_count':int(len(candidate)),'global_shift_mm':(global_shift*1000).tolist(),'global_shift_norm_mm':float(np.linalg.norm(global_shift)*1000),'local_offset_max_mm':float(np.linalg.norm(local,axis=1).max()*1000),'penetrating_before':int((signed<-.001).sum()),'penetrating_after':int((after<-.001).sum()),'median_contact_distance_mm':float(np.median(distances[candidate])*1000) if len(candidate) else None})
    (out/'contact_field_summary.json').write_text(json.dumps({'method':'observed_contact_field_plus_egoaero_residual','object':'laptop_lid','frames':records},indent=2)+'\n'); print(json.dumps({'output':str(out),'frames':len(records),'active_frames':sum(x['candidate_count']>0 for x in records),'max_local_offset_mm':max(x['local_offset_max_mm'] for x in records)},indent=2))
if __name__=='__main__': main()
