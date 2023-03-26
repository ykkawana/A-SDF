# %%
import os
os.chdir('/home/mil/kawana/workspace/3detr/third_party/A-SDF')
import sys
# import dotenv
# dotenv.load_dotenv()
# PROJECT_ROOT = os.getenv("PROJECT_ROOT")
sys.path.insert(0, '/home/mil/kawana/workspace/3detr')
import pickle
import trimesh
import re
import skimage
import os
import numpy as np
import torch
import datetime
import pytorch3d.loss
from mmcv import Config
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse
from datasets import DATASET_FUNCTIONS
import math
import glob
import tempfile
import wandb
from trainer import get_cfg_args, make_args_parser, dumpcfg
from utils import pc_util
from tqdm import tqdm

# sys.path.insert(0, 'third_party/miltuils')
# import milutils
# Setup seed
pl.seed_everything(0)
# %%

device = 'cuda:2'

# Setup parse args
parser = make_args_parser()
# cfg, args, args_ = get_cfg_args(parser)
argv = [
    '-c',
    # '/home/mil/kawana/workspace/3detr/artifacts_utsubo0/experiments/45ty21fb-2023-02-10-06-14-56/config.py',
    # '/home/mil/kawana/workspace/3detr/artifacts_utsubo0/experiments/45ty21fb-2023-02-11-05-38-21/config.py',
    '/home/mil/kawana/workspace/3detr/configs/iccv2023/proposed_full_1.py'
]
cfg, args, args_ = get_cfg_args(parser, argv)



# Setup output dir
dirname = str(datetime.datetime.now()).split(".")[0].replace(" ", "-").replace(":", "-")
output_dir = f'/home/mil/kawana/workspace/3detr/third_party/A-SDF/artifacts_utsubo0/evals/{dirname}'
os.makedirs(output_dir, exist_ok=True)


# Setup dataset & dataloader
eval_batch_size = cfg['test_cfg'].get('eval_batch_size', args.batchsize_per_gpu)
dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
test_dataset = dataset_builder(
    dataset_config, 
    cfg=cfg,
    split_set="val", 
    root_dir=args.dataset_root_dir, 
    use_color=args.use_color,
    num_points=args.num_points,
    augment=False
)

test_dataset.is_output_mesh = True
test_dataset.is_output_scene_id = True
test_dataset.is_output_rendering_id = True


path = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/asdf_sapien_partial_shapes_gt_test.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)

device = 'cuda:0'
losses = {}

# inference_root = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/artifacts_utsubo0/inference'
inference_root = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/artifacts_utsubo0/inference/2023-03-18-12-36-00'
# %%
for bidx in tqdm(range(len(test_dataset))):
    batch = test_dataset[bidx]
    sidx = batch['scan_idx'].item()
    meshes = []
    for gidx in range(len(batch['output_mesh'])):
        mesh = batch['output_mesh'][gidx]
        center = batch['gt_box_frame_centers'][gidx]
        rotmat = batch['gt_box_frame_rotmat'][gidx]
        size = batch['gt_box_sizes'][gidx]
        size_ = np.eye(4)
        size_[:3, :3] = np.diagflat(size) / (1-0.1)
        rotmat_ = np.eye(4)
        rotmat_[:3, :3] = rotmat
        center_ = np.eye(4)
        center_[:3, 3] = center
        tr = center_ @ rotmat_ @ size_
        mesh.apply_transform(tr)
        meshes.append(mesh)
    gt_meshes = trimesh.util.concatenate(meshes)

    key = (batch['scene_id'], batch['rendering_id'])
    # mesh_path = os.path.join(inference_root, 'meshes', batch['scene_id'], f'{rid:04d}.ply')
    mesh_path = os.path.join(inference_root, 'meshes', batch['scene_id'], f'{batch["rendering_id"]:04d}.ply')
    if not os.path.exists(mesh_path):
        zeros = torch.zeros((1, 1, 3), dtype=gt_points.dtype, device=gt_points.device)
        loss, _ = pytorch3d.loss.chamfer_distance(zeros, gt_points, norm=1)
        losses[key] = loss.cpu().numpy()
        continue
    # pred_meshes = []
    # for okey, obdata in data[key].items():
    #     rid = batch['rendering_id']
    #     obj_path = os.path.join(inference_root, batch['scene_id'], f'{rid:04d}', 'meshes', f'{okey:04d}.obj')
    #     if not os.path.exists(obj_path):
    #         continue
    #     obj = trimesh.load(obj_path)
    #     obj.apply_transform(obdata['normalized_sample_to_cam_transform'])
    #     pred_meshes.append(obj)
    # pred_meshes = trimesh.util.concatenate(pred_meshes)
    pred_points = torch.from_numpy(trimesh.sample.sample_surface_even(pred_meshes, 40000)[0]).float()[None].to(device, non_blocking=True)[:, :20000]
    gt_points = torch.from_numpy(trimesh.sample.sample_surface_even(gt_meshes, 40000)[0]).float()[None].to(device, non_blocking=True)[:, :20000]
    loss, _ = pytorch3d.loss.chamfer_distance(pred_points, gt_points, norm=1)
    losses[key] = loss.cpu().numpy()

mean_loss = np.mean(list(losses.values()))

losses_flatten = [(*key, loss.item()) for key, loss in losses.items()]
lines = []
for l in losses_flatten:
    lines.append(','.join([str(v) for v in l]))
text = '\n'.join(lines)
with open(os.path.join(output_dir, 'chamfer-l1.csv'), 'w') as f:
    f.write(text)

with open(os.path.join(output_dir, 'chamfer-l1-avg.csv'), 'w') as f:
    text = f'chamfer-l1\n{mean_loss}'
    f.write(text)

# %%

#     # meshes.show()
#     sidx = batch['scan_idx'].item()
#     print(sidx)
#     mesh_paths = sorted(glob.glob(os.path.join(output_dir, 'meshes', f'{sidx:08d}', '*.obj')))
#     qidxs = [int(os.path.basename(path).split('.')[0]) for path in mesh_paths]
#     meshes = []
#     for qidx in qidxs:
#         mesh_path = trimesh.load(os.path.join(output_dir, 'meshes', f'{sidx:08d}', f'{qidx:03d}.obj'))
#         mesh = trimesh.load(mesh_path)
#         trimesh.repair.fix_inversion(mesh)
#         with open(os.path.join(output_dir, 'poses', f'{sidx:08d}', f'{qidx:03d}.pkl'), 'rb') as f:
#             pose_pkl = pickle.load(f)
#         center = pose_pkl['frame_center_unnormalized']
#         rotmat = pose_pkl['frame_rotmat']
#         min_rotmat = pose_pkl['min_rotmat']
#         size = pose_pkl['size_unnormalized']
#         size_ = np.eye(4)
#         size_[:3, :3] = np.diagflat(size)#/ (1-0.1)
#         rotmat_ = np.eye(4)
#         rotmat_[:3, :3] = rotmat @ min_rotmat
#         center_ = np.eye(4)
#         center_[:3, 3] = center
#         tr = center_ @ rotmat_ @ size_
#         mesh.apply_transform(tr)
#         mesh.visual.vertex_colors = np.array([255, 0, 0, 255])

#         meshes.append(mesh)

#     meshes = trimesh.util.concatenate(meshes)
#     # meshes2 = trimesh.util.concatenate([meshes, gt_meshes])
#     # meshes2.show()
#     device = 'cuda:0'
#     pred_points = torch.from_numpy(trimesh.sample.sample_surface_even(meshes, 20000)[0]).float()[None].to(device, non_blocking=True)
#     gt_points = torch.from_numpy(trimesh.sample.sample_surface_even(gt_meshes, 20000)[0]).float()[None].to(device, non_blocking=True)
#     loss, _ = pytorch3d.loss.chamfer_distance(pred_points, gt_points, norm=1)
#     losses[sidx] = loss.cpu().numpy()
#     if bidx > 10:
#         break

# ch1 = np.mean(list(losses.values()))
# header = 'sidx,chamfer-l1,mean'
# lines = [header]
# for sidx, ch in losses.items():
#     lines.append(f'{sidx},{ch.item()},{ch1}')
# with open(os.path.join(output_dir, 'evals', 'chamfer-l1.csv'), 'w') as f:
#     f.write('\n'.join(lines))
# # %%

# %%
