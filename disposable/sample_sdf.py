# %%
import pickle
import trimesh
import numpy as np

from pysdf import SDF
# %%

path = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/sapien_shapes_train.pkl'

with open(path, 'rb') as f:
    pkl = pickle.load(f)

with open(pkl['paths']['part_trimesh_path'], 'rb') as f:
    part_trimeshes = pickle.load(f)

# %%
# https://github.com/marian42/mesh_to_sdf/blob/66036a747e82e7129f6afc74c5325d676a322114/mesh_to_sdf/utils.py#L46
def sample_uniform_points_in_unit_sphere(amount):
    unit_sphere_points = np.random.uniform(-1, 1, size=(amount * 2 + 20, 3))
    unit_sphere_points = unit_sphere_points[np.linalg.norm(unit_sphere_points, axis=1) < 1]

    points_available = unit_sphere_points.shape[0]
    if points_available < amount:
        # This is a fallback for the rare case that too few points are inside the unit sphere
        result = np.zeros((amount, 3))
        result[:points_available, :] = unit_sphere_points
        result[points_available:, :] = sample_uniform_points_in_unit_sphere(amount - points_available)
        return result
    else:
        return unit_sphere_points[:amount, :]


def get_near_surface_points(mesh, surface_sample_count):
    surface_points = mesh.sample(surface_sample_count)
    # surface_points = trimesh.sample.sample_surface_even(mesh, surface_sample_count)
    near = surface_points + np.random.normal(scale=0.0025, size=(surface_sample_count, 3))
    very_near = surface_points + np.random.normal(scale=0.00025, size=(surface_sample_count, 3))

    return near, very_near

 
# sphere_points_num = 15000
# near_surface_points = 235000 // 2
points_num = 8000
sample_num = points_num * 4
ratio = 47/50
near_surface_points_num = int(sample_num * ratio / 2)
sphere_points_num = sample_num - near_surface_points_num * 2
max_const = 135

sdfs_cache = {}
npzs = []
for cat, objs in pkl['shapes'].items():
    if cat != 'refrigerator':
        continue
    meshes = []
    for k, v in objs.items():
        trs = []
        arts = []
        for bbox in v['bboxes']:
            mesh = part_trimeshes[bbox['cat_obj_id']][bbox['part_cnt']].copy()
            tr = bbox['final_transform']
            mesh.apply_transform(tr)

            sdf_key = (bbox['str_id'], bbox['cat_obj_id'], bbox['part_cnt'])
            if sdf_key not in sdfs_cache:
                sdf_mesh = mesh.copy()
                sdfs_cache[sdf_key] = SDF(sdf_mesh.vertices, sdf_mesh.faces)
            art = np.random.rand()
            if bbox['type'] == 1:
                tr2 = trimesh.transformations.translation_matrix(bbox['axis'] * bbox['max'] * art)
                arts.append(art * max_const)
            elif bbox['type'] == 0:
                tr2 = trimesh.transformations.rotation_matrix(max_const/180 * np.pi * art, bbox['axis'], bbox['pivot'])
                arts.append(art * max_const)
            else:
                tr2 = np.eye(4)
            # tr2 = np.eye(4)
            mesh.apply_transform(tr2)
            meshes.append(mesh)
            trs.append(tr2)
        whole_mesh = trimesh.util.concatenate(meshes)
        sphere_points = sample_uniform_points_in_unit_sphere(sphere_points_num)
        near, very_near = get_near_surface_points(whole_mesh, near_surface_points_num)
        all_points = np.concatenate([sphere_points, near, very_near])

        sdfs = []
        for tr, bbox in zip(trs, v['bboxes']):
            inv_points = (all_points - tr[:3, 3][None]) @ tr[:3, :3]
            sdf_key = (bbox['str_id'], bbox['cat_obj_id'], bbox['part_cnt'])
            sdf = sdfs_cache[sdf_key](inv_points)
            sdfs.append(sdf)
        sdfs = np.stack(sdfs, axis=0)
        sdfs = -sdfs
        sdf_idx = sdfs.argmin(0)
        sdf = sdfs.min(0)
        label = sdf_idx + 1
        pos_mask = sdf > 0
        label = np.where(pos_mask, 0, label)
        perm = np.random.permutation(sample_num)
        permed_label = label[perm]
        permed_points = all_points[perm]
        permed_sdf = sdf[perm]

        permed_pos_mask = permed_sdf > 0
        permed_pos_idx = np.where(permed_pos_mask)[0]
        if len(permed_pos_idx) < points_num:
            permed_pos_idx = np.concatenate([permed_pos_idx, np.random.choice(permed_pos_idx, points_num - len(permed_pos_idx))])

        permed_neg_mask = ~permed_pos_mask
        permed_neg_idx = np.where(permed_neg_mask)[0]
        if len(permed_neg_idx) < points_num:
            permed_neg_idx = np.concatenate([permed_neg_idx, np.random.choice(permed_neg_idx, points_num - len(permed_neg_idx))])

        pos_points = permed_points[permed_pos_idx][:points_num]
        pos_label = np.zeros(len(pos_points), dtype=pos_points.dtype)
        pos_sdf = permed_sdf[permed_pos_idx][:points_num]
        pos = np.concatenate([pos_points, pos_sdf[..., None], pos_label[..., None]], axis=1)
        neg_points = permed_points[permed_neg_idx][:points_num]
        neg_label = permed_label[permed_neg_idx][:points_num]
        neg_sdf = permed_sdf[permed_neg_idx][:points_num]
        neg = np.concatenate([neg_points, neg_sdf[..., None], neg_label[..., None]], axis=1)
        npz=dict(
            pos=pos,
            neg=neg,
        )
        npzs.append(npz)
        if len(npzs) > 10:
            break
    if len(npzs) > 10:
        break
# %%
label = neg_label
xyz = neg_points

ls = np.unique(label)
ps = []
for l in ls:
    print('l', l)
    ps.append(xyz[label == l])

fig = milutils.visualizer.get_pcd_plot(ps)
fig.show()

# %%
import sys
sys.path.insert(0, '/home/mil/kawana/workspace/3detr/third_party/milutils')
import milutils

# path = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/data/SdfSamples/shape2motion/refrigerator/0006art00900090.npz'
# npz = np.load(path)
# %%
npz = npzs[5]
for k, v in npz.items():
    print(k, v.shape)
# xyz [-1, 1]
# pos is outside
# neg is inside
key = 'neg'
xyz = npz[key][:, :3]
print(xyz.min(0), xyz.max(0))
print(np.linalg.norm(xyz.max(0) - xyz.min(0)))
dist = npz[key][:, 3]
label = npz[key][:, 4]
print(label.min())
print(label.max())
print(dist.min())
print(dist.max())
perm = np.random.permutation(xyz.shape[0])[:5000]
xyz = xyz[perm]
dist = dist[perm]
label = label[perm]

ls = np.unique(label)
ps = []
for l in ls:
    ps.append(xyz[label == l])

fig = milutils.visualizer.get_pcd_plot(ps)
fig.show()
# %%
