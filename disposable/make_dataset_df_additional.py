# %%
import json
import pickle
from tqdm import tqdm
from collections import defaultdict
import os
import numpy as np
import trimesh

split = 'train' # 'test' or 'train'
# split = 'test' # 'test' or 'train'
if split == 'train':
    split_ = 'train'
else:
    split_ = 'val'
train_path = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/derivatives/3detr/splits/oc0.5_tr0.25_str0.95-0-objs4-train20000test4000/{split_}.json'
part_trimesh_path = '/home/mil/kawana/workspace/3detr/artifacts_unagi0/outputs/sapien/2022-09-14-01-41-47-ac484333/packed_trimesh.pkl'
part_augmat_path = f'/home/mil/kawana/workspace/3detr/artifacts_unagi0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/derivatives/3detr/part_mesh_augmat-all_side_length_one.pkl'
rendering_direction_dirpath = f'/home/mil/kawana/workspace/3detr/artifacts_utsubo0/outputs/sapien_procart/2022-09-14-03-06-04-95214e44-ready-scene4/{split}/rendering_directions'
part_trimeshes = pickle.load(open(part_trimesh_path, 'rb'))
part_augmats = pickle.load(open(part_augmat_path, 'rb'))

with open(train_path) as f:
    paths = json.load(f)

objs = defaultdict(lambda: defaultdict(lambda: {}))

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
cat_num_additional = dict(
    storagefurniture=2,
    trashcan=2,
    microwave=2,
    box=4,
    refrigerator=1,
    table=2,
)

# %%
skip_intermediate = True
if not skip_intermediate:
    all_meshes = []
    all_objs = defaultdict(lambda: defaultdict(lambda: []))
    pad = np.eye(4)
    pad[:3, :3] = np.eye(3) / (1-0.1)
    for pidx, path in tqdm(enumerate(paths)):
        with open(path, 'rb') as f:
            pkl = pickle.load(f)
        meshes = []
        for i, bbox in enumerate(pkl['boxes']):
            # if bbox['category'] != 'oven':
            #     continue
            scene_id = pkl['data_tracing']['str_id']
            rendering_id = pkl['data_tracing']['rendering_id']
            if bbox['parent'] == -1:
                parent_rot_inv = np.eye(4)
                parent_rot_inv[:3, :3] = bbox['poses']['frame']['se3_cam'][:3, :3].T

                dir_path = os.path.join(rendering_direction_dirpath, pkl['data_tracing']['str_id'], 'direction.pkl')
                with open(dir_path, 'rb') as f:
                    direction = pickle.load(f)
                obj_id_in_scene = bbox['obj_id_in_scene']
                original_scale = direction['scene_direction']['directions'][obj_id_in_scene]['aug']['scale_and_y0']
                original_scale = np.abs(np.diag(original_scale))[:-1]
                part_cnt = 0
            else:
                part_cnt += 1

            if bbox['type'] in [0, 1]:
                axis = bbox['joint_info']['axis']
                unit_axis = axis / np.linalg.norm(axis)

                se3_min_world = bbox['poses']['min']['se3_world']
                se3_canonical = bbox['se3_canonical']
                scene_pose = se3_min_world @ np.linalg.inv(se3_canonical) @ np.linalg.inv(bbox['poses']['min']['se3_part_transform'])

                camera_pose = pkl['camera']['selected_pose']
                camera_pose_inv = np.linalg.inv(camera_pose)
                unit_axis = unit_axis @ (camera_pose_inv @ scene_pose)[:3, :3].T

            if bbox['type'] == 0:
                anchor = bbox['joint_info']['pivot']
                amat = trimesh.transformations.translation_matrix(anchor)
                fmat = camera_pose_inv @ scene_pose @ amat
                pivot = fmat[:3, 3]


            imp_aug_mat = part_augmats[pkl['data_tracing']['str_id']][i]
            pose  = bbox['poses']['min']['se3_cam']
            size = np.eye(4)
            size[:3, :3] = np.diagflat(bbox['size'])
            canonical_pose = parent_rot_inv @ pose @ size @ pad @ imp_aug_mat

            ret = dict(
                cat_obj_id=bbox['cat_obj_id'],
                category=bbox['category'],
                obj_id=bbox['obj_id'],
                original_scale=original_scale,
                str_id=pkl['data_tracing']['str_id'],
                part_cnt=part_cnt,
                bbox_id=i,
                scene_obj_id=bbox['obj_id_in_scene'],
                type=bbox['type'],
                canonical_pose=canonical_pose,
            )
            if bbox['type'] in [1, 0]:
                unit_axis = unit_axis @ (parent_rot_inv)[:3, :3].T
                ret['axis'] = unit_axis
                ret['raw_max']=bbox['joint_info']['raw_max_as_min_0']
                ret['max']=bbox['joint_info']['max']
            if bbox['type'] == 0:
                pivot = pivot @ (parent_rot_inv)[:3, :3].T
                ret['pivot'] = pivot 
            all_objs[ret['category']][(scene_id, rendering_id, ret['obj_id'], ret['scene_obj_id'])].append(ret)

    # %%
    all_objs_ = dict(all_objs)
    for k, v in all_objs_.items():
        all_objs_[k] = dict(v)

    for k, v in all_objs.items():
        for kk, o in v.items():
            pcnts = defaultdict(lambda: 0)
            for oo in o:
                pcnts[oo['part_cnt']] += 1
            for v in pcnts.values():
                assert v == 1
    # with open(f'../sapien_shapes_intermediate_{split}.pkl', 'rb') as f:
    #     all_objs_2 = pickle.load(f)
    # all_objs_2['oven'] = all_objs_['oven']
    # all_objs_ = all_objs_2

    with open(f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_intermediate_{split}_fixed.pkl', 'wb') as f:
        pickle.dump(all_objs_, f)
# %%
# with open(f'../sapien_shapes_intermediate_{split}.pkl', 'rb') as f:
# with open(f'../sapien_shapes_intermediate_{split}_fixed.pkl', 'rb') as f:
with open(f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_intermediate_{split}_fixed.pkl', 'rb') as f:
    all_objs = pickle.load(f)
all_objs2 = defaultdict(lambda: defaultdict(lambda: {}))
for cat, objs in all_objs.items():
    # if cat != 'oven':
    #     continue
    print(cat)
    # if cat != 'oven':
    #     continue
    for obj_global_id, bboxes in tqdm(objs.items()):
        scene_id, rendering_id, obj_id, scene_obj_id = obj_global_id
        obj_sapien_shape_key = (scene_id, obj_id, scene_obj_id)
        if obj_sapien_shape_key in all_objs2[cat]:
            continue
        match = False
        additional = False
        if len(bboxes) == cat_num[cat] + 1:
            match = True
        if cat in cat_num_additional and len(bboxes) == cat_num_additional[cat] + 1:
            additional = True
            match = True
        if not match:
            continue
    #     break
    # break

        meshes = []
        artparts = []
        for bbox in bboxes:
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
        normalize_transform = ct @ scale

        center_indices = []
        res = 5
        for ap in artparts:
            ap.apply_transform(normalize_transform)
            bounds = ap.bounds
            xy = bounds.sum(0)[:2] / 2
            z = bounds[1, 2]
            center = np.array([xy[0], xy[1], z])
            center = np.clip((center + 0.45) / 0.9, 0, 1)
            center = (center * res).astype(np.int64)
            center_idx = center[2] + center[1] * res + center[0] * res * res 
            center_indices.append(center_idx)
        sorted_indices = [idx for _, idx in sorted(zip(center_indices, range(len(artparts))))]
        new_bboxes = [bboxes[0]]
        new_bboxes.extend([bboxes[i+1] for i in sorted_indices])

        # # print('a', artparts)
        # if sorted_indices != list(range(len(artparts))):
        #     print('swap', len(art_partss))
        # artparts = [artparts[i] for i in sorted_indices]
        # for midx, m in enumerate(artparts):
        #     if midx == 0:
        #         m.visual.vertex_colors = np.array([1.0, 0.0, 0.0, 1]) 
        #     else:
        #         m.visual.vertex_colors = np.array([0.0, 1.0, 0.0, 1]) 
        # # print('b', artparts)

        # art_partss.append((artparts, center_indices, obj_global_id))

        for bbox in new_bboxes:
            bbox['final_transform'] = normalize_transform @ bbox['canonical_pose']
            if bbox['type'] == 0:
                normalized_pivot = bbox['pivot'] @ normalize_transform[:3, :3].T + normalize_transform[:3, 3]
                bbox['pivot'] = normalized_pivot
            if bbox['type'] == 1:
                bbox['max'] *= scale[0, 0]

        all_objs2[cat][obj_sapien_shape_key]['bboxes'] = new_bboxes
        all_objs2[cat][obj_sapien_shape_key]['sorted_indices'] = [0] + [i+1 for i in sorted_indices]

        if additional:
            assert len(new_bboxes) == cat_num_additional[cat] + 1
        else:
            assert len(new_bboxes) == cat_num[cat] + 1

    #     if len(art_partss) > 3:
    #         break
    # if len(art_partss) > 3:
    #     break
# %%
all_objs2_ = dict(all_objs2)
for k, v in all_objs2_.items():
    all_objs2_[k] = dict(v)

sapien_shapes = dict(
    paths=dict(
        train_path = train_path,
        part_trimesh_path = part_trimesh_path,
        part_augmat_path = part_augmat_path,
        rendering_direction_dirpath = rendering_direction_dirpath,
    ),
    shapes=all_objs2_,
)

# with open(f'../sapien_shapes_{split}.pkl', 'rb') as f:
#     sapien_shapes2 = pickle.load(f)
# sapien_shapes2['shapes']['oven'] = sapien_shapes['shapes']['oven']
# sapien_shapes = sapien_shapes2
# # %%

with open(f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_{split}_fixed_additional.pkl', 'wb') as f:
    pickle.dump(sapien_shapes, f)
# %%
# all_meshes = []
# for cat, objs in tqdm(all_objs2.items()):
#     for obj_global_id, obj in objs.items():
#         bboxes = obj['bboxes']
#         meshes = []
#         for bbox in bboxes:

#             mesh = part_trimeshes[bbox['cat_obj_id']][bbox['part_cnt']].copy()
#             tr = bbox['final_transform']
#             if bbox['type'] == 1:
#                 print(len(all_meshes))
#                 tr = trimesh.transformations.translation_matrix(bbox['axis'] * bbox['max']) @ tr
#             elif bbox['type'] == 0:
#                 tr = trimesh.transformations.rotation_matrix(bbox['max'], bbox['axis'], bbox['pivot']) @ tr
#             mesh.apply_transform(tr)
#             meshes.append(mesh)
#         all_mesh = trimesh.util.concatenate(meshes)
#         all_meshes.append(all_mesh)
#         # obj['mesh'] = all_mesh
#         # obj['mesh'].export(f'./data/processed/{cat}/{obj_global_id[0]}_{obj_global_id[1]}_{obj_global_id[2]}.obj')
# # %%
