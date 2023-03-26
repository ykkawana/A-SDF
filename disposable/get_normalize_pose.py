# %%
import json
import pickle
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import pycocotools.mask
import copy
from tqdm import tqdm
from collections import defaultdict
import os
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/3detr/third_party/milutils')
import milutils
import numpy as np
import trimesh
from tqdm import tqdm
from skimage.transform import rescale, resize
from scipy.ndimage import median_filter

sys.path.insert(0, '/home/mil/kawana/workspace/3detr')
from datasets import augments


def get_pcd_from_depth(depth, K, bbox=None):
    rows, cols = depth.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    if bbox is not None:
        c = c[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        r = r[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        depth = depth[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    # c = c.astype(np.float64)
    # r = r.astype(np.float64)
    return depth_cam_to_pcd(depth, c, r, fx, fy, cx, cy)

def depth_cam_to_pcd(depth, c, r, fx, fy, cx, cy):
# def depth_cam_to_pcd(depth: np.float64, c: np.int64, r: np.int64, fx: np.float64, fy: np.float64, cx: np.float64, cy: np.float64):
    z = depth
    x = z * (c - cx) / fx
    y = z * (r - cy) / fy
    return np.dstack((x, y, z)).reshape(-1, 3)

def surface_normal(points, sH, sW):
    # These lookups denote y,x offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    d = 2
#     lookups = {0:(-d,0),1:(-d,d),2:(0,d),3:(d,d),4:(d,0),5:(d,-d),6:(0,-d),7:(-d,-d)}

    lookups = {0:(0,-d),1:(d,-d),2:(d,0),3:(d,d),4:(0,d),5:(-d,d),6:(-d,0),7:(-d,-d)}

    surface_normals = np.zeros((sH,sW,3))
    for i in range(sH):
        for j in range(sW):
            min_diff = None
            point1 = points[i,j,:3]
             # We choose the normal calculated from the two points that are
             # closest to the anchor points.  This helps to prevent using large
             # depth disparities at surface borders in the normal calculation.
            for k in range(8):
                try:
                    point2 = points[i+lookups[k][0],j+lookups[k][1],:3]
                    point3 = points[i+lookups[(k+2)%8][0],j+lookups[(k+2)%8][1],:3]
                    diff = np.linalg.norm(point2 - point1) + np.linalg.norm(point3 - point1)
                    if min_diff is None or diff < min_diff:
                        normal = normalize(np.cross(point2-point1,point3-point1))
                        min_diff = diff
                except IndexError:
                    continue
            surface_normals[i,j,:3] = normal
    return surface_normals

def normalize(v):
    return v/np.linalg.norm(v)


def depth2world(depth_map, intrinsic_param):

    # Get world coords
    H, W = depth_map.shape

    WS = np.repeat(np.linspace(1 / (2 * W), 1 - 1 / (2 * W), W).reshape([1, -1]), H, axis=0)
    HS = np.repeat(np.linspace(1 / (2 * H), 1 - 1 / (2 * H), H).reshape([-1, 1]), W, axis=1)

    pixel_coords = np.stack([WS*W, HS*H, np.ones(depth_map.shape)], 2)
    pixel_coords = pixel_coords.reshape(-1, 3).T
    depth_map = depth_map.reshape(-1,1).T
    
    cam_coords = np.linalg.inv(intrinsic_param)@(pixel_coords)
    cam_coords *= depth_map
    
    cam_coords = np.vstack([cam_coords, np.ones((1,cam_coords.shape[1]))])
    # world_coords = np.linalg.inv(extrinsic_param)@cam_coords
    world_coords = cam_coords
    
    world_coords = world_coords.T
    return world_coords

def depth_to_surface_normal_opencv_projection(depth, intrinsics, scale=0.25):
    depth_map = depth.copy()
    H, W = depth.shape
    sH, sW = int(scale*H), int(scale*W)
    depth_map[depth < 0.0001] = 50.0

    # Each 'pixel' containing the 3D point in camera coords
    depth_in_world = depth2world(depth_map, intrinsics)[:,:3].reshape(H,W,3)
    surface_normals = surface_normal(depth_in_world[::int(1/scale),::int(1/scale),:], sH, sW)
    surface_normals = resize(surface_normals, (H, W), anti_aliasing=True)
    return surface_normals


def get_normal(depth, intrinsic_param):
    H, W = depth.shape
    # depth[depth>1] = 0 
    depth = median_filter(depth, size=15)

    surface_normals = depth_to_surface_normal_opencv_projection(depth, intrinsic_param)

    return surface_normals



cat_num = dict(
    dishwasher=1,
    trashcan=1,
    safe=1,
    oven=2,
    storagefurniture=1,
    table=1,
    microwave=1,
    refrigerator=2,
    washingmachine=1,
    box=1,
)

pad = np.eye(4)
pad[:3, :3] = np.eye(3) / (1-0.1)

debug = False

# for split in ['train']:
for split in ['test']:
    if split == 'train':
        split_ = 'train'
    else:
        split_ = 'val'

    train_path = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/derivatives/3detr/splits/oc0.5_tr0.25_str0.95-0-objs4-train20000test4000/{split_}.json'
    with open(train_path) as f:
        paths = json.load(f)

    part_trimesh_path = '/home/mil/kawana/workspace/3detr/artifacts_unagi0/outputs/sapien/2022-09-14-01-41-47-ac484333/packed_trimesh.pkl'
    with open(part_trimesh_path, 'rb') as f:
        part_trimeshes = pickle.load(f)

    part_augmat_path = f'/home/mil/kawana/workspace/3detr/artifacts_unagi0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/derivatives/3detr/part_mesh_augmat-all_side_length_one.pkl'

    with open(part_augmat_path, 'rb') as f:
        part_augmats = pickle.load(f)
    visibility_path = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/visibility.pkl'
    with open(visibility_path, 'rb') as f:
        visibility_pkl = pickle.load(f)

    part_visibility_root = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/part_visibility'
    rendering_root = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/renderings'

    shape_sapien_path = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed.pkl'
    with open(shape_sapien_path, 'rb') as f:
        shape_sapien_pkl = pickle.load(f)

    rendering_direction_dirpath = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/rendering_directions'

    with open(f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_intermediate_{split}_fixed.pkl', 'rb') as f:
        all_objs = pickle.load(f)

    all_objs2 = defaultdict(lambda: {})

    obj_cache = {}

    for path in tqdm(paths):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        # prismatic_found = False
        # for bidx, box in enumerate(pkl['boxes']):
        #     prismatic_found = box['type'] == 1
        #     if prismatic_found:
        #         break
        # if not prismatic_found:
        #     continue

        tmp = os.path.basename(path).split('.')[0].split('-')
        rendering_id = int(tmp[-1])
        scene_id = '-'.join(tmp[:-1])

        with open(os.path.join(part_visibility_root, scene_id, f'{rendering_id:04d}.pkl'), 'rb') as f:
            part_visibility_pkl = pickle.load(f)

        assert len(pkl['boxes']) == len(part_visibility_pkl['labels']) - part_visibility_pkl['obj_offset']
        masks = {}
        for bidx, box in enumerate(pkl['boxes']):
            mask = pycocotools.mask.decode(part_visibility_pkl['masks'][f'part-{bidx}'])
            if box['obj_id_in_scene'] not in masks:
                masks[box['obj_id_in_scene']] = mask
            else:
                masks[box['obj_id_in_scene']] |= mask

        masks_rle = {}
        masks_bbox = {}
        for oidx in masks:
            if np.any(masks[oidx]):
                masks_rle[oidx] = pycocotools.mask.encode(np.asfortranarray(masks[oidx]))
                yy, xx = np.where(masks[oidx])
                masks_bbox[oidx] = (np.min(yy), np.min(xx), np.max(yy)+1, np.max(xx)+1)

        visible_obj_ids = visibility_pkl[(scene_id, rendering_id)]


        if debug:
            color_path = os.path.join(rendering_root, 'color', scene_id, f'Image{rendering_id:04d}.png')
            RGB_image = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)

        depth_path = os.path.join(rendering_root, 'depth16bit_max12', scene_id, f'Image{rendering_id:04d}.png')
        assert os.path.exists(depth_path), depth_path
        depth_image = cv2.imread(depth_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.view(np.float16).astype(np.float32)
        maxm = 12
        depth_image+=0.5
        depth_image*=maxm
        depth_image[depth_image > 12] = 0

        sharpen_depth_th = 0.1
        sharpen_depth_scale = 1

        normal_image = get_normal(depth_image, pkl['camera']['K'])
        normal_image[:, :, 1:] *= -1 

        if debug:
            normal_rgb = np.clip(normal_image * 0.5 + 0.5, 0, 1) * 255
            plt.imshow(normal_rgb.astype(np.uint8))
            plt.show()
            samples_set = []

        depth_image = augments.sharpen_depth(depth_image, th=sharpen_depth_th, scale=sharpen_depth_scale)

        obj_id_cache = []
        for bidx, box in enumerate(pkl['boxes']):
            if box['scene_obj_id'] in obj_id_cache or box['obj_id_in_scene'] not in visible_obj_ids:
                continue
            obj_id_cache.append(box['scene_obj_id'])
            assert box['parent'] == -1
            parent_rot_inv = np.eye(4)
            parent_rot_inv[:3, :3] = box['poses']['frame']['se3_cam'][:3, :3].T


            category = box['category']
            obj = copy.deepcopy(all_objs[category][(scene_id, rendering_id, box['obj_id'], box['obj_id_in_scene'])])
            pcnts = defaultdict(lambda: 0)
            for o in obj:
                pcnts[o['part_cnt']] += 1
            for v in pcnts.values():
                assert v == 1

            meshes = []
            artparts = []
            for bbox in obj:
                mesh = part_trimeshes[bbox['cat_obj_id']][bbox['part_cnt']].copy()
                mesh.apply_transform(bbox['canonical_pose'])
                if bbox['type'] == 0:
                    artparts.append(mesh.copy())
                    pivot = bbox['pivot']
                    axis = bbox['axis']
                    submeshes = []
                    for ang in np.linspace(0, 135/180 * np.pi, 10):
                        tr = trimesh.transformations.rotation_matrix(ang, axis, pivot)
                        submeshes.append(mesh.copy().apply_transform(tr))
                    meshes.extend(submeshes)
                elif bbox['type'] == 1:
                    artparts.append(mesh.copy())
                    axis = bbox['axis']
                    tr = trimesh.transformations.translation_matrix(axis * bbox['max'])
                    mesh.apply_transform(tr)
                    meshes.append(mesh)
                else:
                    meshes.append(mesh)
            all_mesh = trimesh.util.concatenate(meshes)
            length = all_mesh.extents.max()
            scale = trimesh.transformations.scale_matrix(1/length/(1/(1-0.1)))
            all_mesh.apply_transform(scale)
            center = all_mesh.bounds.sum(0)/2
            ct = trimesh.transformations.translation_matrix(-center)
            all_mesh.apply_transform(ct)

            amounts = [] 
            subobj = [o for o in obj if o['scene_obj_id'] == obj[0]['scene_obj_id']]
            subboxes = [b for b in pkl['boxes'] if b['obj_id_in_scene'] == obj[0]['scene_obj_id']]
            # print(len(subboxes), len(subobj))
            for bbox, obj_bbox in zip(subboxes, subobj):
                if obj[0]['scene_obj_id'] != bbox['obj_id_in_scene']:
                    continue
                if bbox['type'] == 0:
                    amounts.append(
                        dict(
                            frame=bbox['poses']['frame']['amount'],
                            max=135/180 * np.pi,
                            max_in_world_scale=135/180 * np.pi,
                            original_max=obj_bbox['max'],
                            frame_in_world_scale=bbox['poses']['frame']['amount'],
                            normalized_frame=bbox['poses']['frame']['amount']/(135/180*np.pi),
                        ))
                elif bbox['type'] == 1:
                    amounts.append(
                        dict(
                            frame=bbox['poses']['frame']['amount'] * scale[0,0],
                            max=obj_bbox['max'] * scale[0,0],
                            max_in_world_scale=obj_bbox['max'],
                            original_max=obj_bbox['max'],
                            frame_in_world_scale=bbox['poses']['frame']['amount'],
                            normalized_frame=bbox['poses']['frame']['amount']/obj_bbox['max'],
                        ))
    #                 print(amounts[-1], scale[0,0])
    #                 assert False
    # #             # print(bbox['poses']['frame']['amount'], bbox['poses']['max']['amount'])
    #             if len(amounts) > 0:
    #                 print(amounts[-1], scale[0,0])
    #         break
    # break

            normalize_transform = ct @ scale @ parent_rot_inv
            normalize_transform2 = ct @ scale

            center_indices = []
            res = 5
            # print('artparts', len(artparts))
            for ap in artparts:
                ap.apply_transform(normalize_transform2)
                bounds = ap.bounds
                xy = bounds.sum(0)[:2] / 2
                z = bounds[1, 2]
                center = np.array([xy[0], xy[1], z])
                center = np.clip((center + 0.45) / 0.9, 0, 1)
                center = (center * res).astype(np.int64)
                center_idx = center[2] + center[1] * res + center[0] * res * res 
                center_indices.append(center_idx)
            sorted_indices = [idx + 1 for _, idx in sorted(zip(center_indices, range(len(artparts))))]
            sorted_indices = [0] + sorted_indices

            for bbox in obj:
                bbox['final_transform'] = ct @ scale @ bbox['canonical_pose']
                if bbox['type'] == 0:
                    normalized_pivot = bbox['pivot'] @ normalize_transform2[:3, :3].T + normalize_transform2[:3, 3]
                    bbox['pivot'] = normalized_pivot
            
            sorted_obj = [obj[i] for i in sorted_indices]
            # print(sorted_indices, len(amounts), obj_bbox['category'])
            sorted_art_amounts = [amounts[i-1] for i in sorted_indices[1:]]
            pkl_boxes = [bbox for bbox in pkl['boxes'] if obj[0]['scene_obj_id'] == bbox['obj_id_in_scene']]
            sorted_pkl_boxes = [pkl_boxes[i] for i in sorted_indices]

            ret = {
                'normalize_transform': normalize_transform,
                'sorted_boxes': sorted_obj,
                'sorted_pkl_boxes': sorted_pkl_boxes,
                'obj_mask_rle': masks_rle[box['obj_id_in_scene']],
                'obj_bbox': masks_bbox[box['obj_id_in_scene']],
                'scale': scale[0, 0],
                'sorted_art_amounts': sorted_art_amounts,
                'sorted_indices': sorted_indices,
            }

            normalized_sample_to_cam_transform = np.linalg.inv(normalize_transform)
            rotsize = np.eye(4)
            rotsize[:3, :3] = normalize_transform[:3, :3]
            scale_mat = trimesh.transformations.scale_matrix(1/ret['scale'])
            normal_transform = scale_mat @ rotsize

            cropped_depth = depth_image[ret['obj_bbox'][0]:ret['obj_bbox'][2], ret['obj_bbox'][1]:ret['obj_bbox'][3]]
            mask0 = cropped_depth.reshape(-1) != 0
            mask = pycocotools.mask.decode(ret['obj_mask_rle'])

            cropped_mask = mask[ret['obj_bbox'][0]:ret['obj_bbox'][2], ret['obj_bbox'][1]:ret['obj_bbox'][3]].astype(np.bool)
            cropped_mask &= mask0.reshape(cropped_mask.shape)

            cropped_normal = normal_image[ret['obj_bbox'][0]:ret['obj_bbox'][2], ret['obj_bbox'][1]:ret['obj_bbox'][3]]
            normal = cropped_normal.reshape(-1, 3)

            pcd = get_pcd_from_depth(depth_image, pkl['camera']['K'], ret['obj_bbox'])
            pcd[:, 1:] *= -1

            pcd = pcd[cropped_mask.reshape(-1)]
            normal = normal[cropped_mask.reshape(-1)]

            pcd = trimesh.transform_points(pcd, normalize_transform)
            normal = trimesh.transform_points(normal, normal_transform)
            normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)

            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd)
            _, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=3.0)
            pcd = pcd[ind]
            normal = normal[ind]



            ret2 = dict( 
                cam_to_normalized_sample_transform=normalize_transform,
                points=pcd,
                normal=normal,
                normalized_sample_to_cam_transform=normalized_sample_to_cam_transform,
                obj_id_in_scene=box['obj_id_in_scene'],
                is_trained_art=(scene_id, bbox['obj_id'], bbox['scene_obj_id']) in shape_sapien_pkl['shapes'][category],
                scale=ret['scale'],
                category=category,
                # bbox=ret['obj_bbox'],
                # cropped_mask_rle=pycocotools.mask.encode(cropped_mask.astype(np.uint8)),
                # normal_transform=normal_transform,
            )

            selected_bboxes = [b for b in pkl['boxes'] if box['obj_id_in_scene'] == b['obj_id_in_scene']]
            assert len(ret['sorted_boxes']) == len(selected_bboxes)
            # amounts = []
            # mas = []
            # for bbox, bbox2 in zip(ret['boxes'], selected_bboxes):

            #     if bbox2['type'] == 1:
            #         amount = bbox2['poses']['frame']['amount']
            #         scale = ret['scale']
            #         ma = bbox2['poses']['frame']['max']
            #     elif bbox2['type'] == 0:
            #         amount = bbox2['poses']['frame']['amount']
            #         ma = bbox2['poses']['frame']['max']
            #     else:
            #         amount = 0
            #         ma = 0
            #     amounts.append(amount)
            #     mas.append(ma)

            # print(ret2['is_trained_art'])
            # if ret2['is_trained_art']:
            #     ret2['sorted_indices'] = shape_sapien_pkl['shapes'][category][(scene_id, bbox['obj_id'], bbox['scene_obj_id'])]['sorted_indices']
            #     ret2['sorted_amount'] = [amounts[i] for i in ret2['sorted_indices']]
            #     ret2['sorted_max_amount'] = [mas[i] for i in ret2['sorted_indices']]

            ret.update(ret2)
            all_objs2[(scene_id, rendering_id)][box['obj_id_in_scene']] = ret
            

            if debug:
                meshes2 = []
                for bbox, bbox2 in zip(obj, selected_bboxes):
                    mesh = part_trimeshes[bbox['cat_obj_id']][bbox['part_cnt']].copy()

                    tr = bbox['final_transform']
                    mesh.apply_transform(tr)
                    if bbox2['type'] == 1:
                        amount = bbox2['poses']['frame']['amount']
                        scale = ret['scale']
                        amount = amount * scale
                        axis = bbox['axis']
                        tr = trimesh.transformations.translation_matrix(axis * amount)
                    elif bbox2['type'] == 0:
                        amount = bbox2['poses']['frame']['amount']
                        axis = bbox['axis']
                        pivot = bbox['pivot']
                        tr = trimesh.transformations.rotation_matrix(amount, axis, pivot)
                    else:
                        tr = np.eye(4)
                    mesh.apply_transform(tr)
                    meshes2.append(mesh) 
                meshes2 = trimesh.util.concatenate(meshes2)



                cropped_color = RGB_image[ret['obj_bbox'][0]:ret['obj_bbox'][2], ret['obj_bbox'][1]:ret['obj_bbox'][3]]
                rgb = cropped_color.reshape(-1, 3)
                rgb = rgb[cropped_mask.reshape(-1)]

                eta = 0.025

                pos = pcd + normal * eta
                neg = pcd - normal * eta 

                eta_vec = np.ones((pos.shape[0], 1)) * eta
                part = np.zeros((pos.shape[0], 1))
                pos = np.hstack([pos, eta_vec, part])
                neg = np.hstack([neg, -eta_vec, part])
                

                samples = meshes2.sample(5000)
                perm = np.random.permutation(pcd.shape[0])
                pcd = pcd[perm[:5000]]
                fig = milutils.visualizer.get_pcd_plot([pcd, samples], marker_sizes=1)
                fig.show()

                # fig = milutils.visualizer.get_pcd_plot([pos, neg], marker_sizes=1)
                # fig.show()


                samples_set.append(trimesh.transform_points(samples, normalized_sample_to_cam_transform))

        if debug:
            all_pcd = get_pcd_from_depth(depth_image, pkl['camera']['K'])
            all_pcd[:, 1:] *= -1
            perm = np.random.permutation(all_pcd.shape[0])
            all_pcd = all_pcd[perm[:20000]]
            fig = milutils.visualizer.get_pcd_plot([all_pcd, *samples_set], marker_sizes=1)
            fig.show()
            break
        # break
    with open(f'./asdf_sapien_partial_shapes_gt_{split}.pkl', 'wb') as f:
        all_objs2 = dict(all_objs2)
        pickle.dump(all_objs2, f)
    if debug:
        break
# %%