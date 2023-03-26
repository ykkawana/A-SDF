# %%
import pytorch_lightning as pl
pl.seed_everything(0)
import glob
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
torch.set_num_threads(1)
import numpy as np
import copy
import random
import pytorch_lightning as pl
seed = 0
pl.seed_everything(seed)
# # Python random
# random.seed(seed)
# # Numpy
# np.random.seed(seed)
# # Pytorch
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms = True


import torch.utils.data as data_utils
from torch.nn import functional as F
import wandb

import signal
import sys
import os
import logging
import math
import json
import time
import numpy as np
import argparse

import dotenv
dotenv.load_dotenv()
os.chdir('/home/mil/kawana/workspace/3detr/third_party/A-SDF')
sys.path.insert(0, '.')
sys.path.insert(0, '../milutils')
import milutils
import asdf
from asdf.utils import *
import asdf.workspace as ws

from asdf.asdf_reconstruct import reconstruct_ttt
from train import get_arg_parser, load_checkpoints, test

import pickle
import numpy as np
# %%

def load_model(args, specs):
    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    test_split_file = specs["TestSplit"]
    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]

    num_epochs = specs["NumEpochs"]
    lr_schedules = get_learning_rate_schedules(specs)
    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    if grad_clip is not None:
        logging.debug("clipping gradients to max norm {}".format(grad_clip))
    do_sup_with_part = specs["TrainWithParts"]
    num_samp_per_scene = specs["SamplesPerScene"]
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    enforce_minmax = True

    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    code_bound = get_spec_with_default(specs, "CodeBound", 0.1)

    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    test_frequency = get_spec_with_default(specs, "TestFrequency", 10)
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )

    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    decoder = arch.Decoder(num_atc_parts=specs["NumAtcParts"], do_sup_with_part=specs["TrainWithParts"]).cuda()

    try:
        lat_vecs, decoder, optimizer_all, start_epoch, loss_log, lr_log, timing_log, start_epoch = load_checkpoints(args.continue_from, ws, args.experiment_directory, None, decoder, None)
    except:
        decoder = torch.nn.DataParallel(decoder)
        lat_vecs, decoder, optimizer_all, start_epoch, loss_log, lr_log, timing_log, start_epoch = load_checkpoints(args.continue_from, ws, args.experiment_directory, None, decoder, None)
    decoder.eval()

    return decoder



arg_parser = get_arg_parser()
asdf.add_common_args(arg_parser)

dirs = sorted(glob.glob("/home/mil/kawana/workspace/3detr/third_party/A-SDF/examples/sapien_submit/*"))

decoders = {}
decoders_use = {}
for d in dirs:
    argv = [
        '-e',
        d,
        '-c',
        '1000',
        '--use_sapien',
    ]
    args = arg_parser.parse_args(argv)

    specs = ws.load_experiment_specifications(args.experiment_directory)
    decoder = load_model(args, specs)
    decoders[os.path.basename(d)] = (decoder, args, specs)
    decoder_use = load_model(args, specs)
    decoders_use[os.path.basename(d)] = (decoder_use, args, specs)

# %%

data_source = specs["DataSource"]
train_split_file = specs["TrainSplit"]
test_split_file = specs["TestSplit"]
arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

num_samp_per_scene = specs["SamplesPerScene"]

train_split = train_split_file
test_split = test_split_file

# sdf_dataset_test = asdf.data.SapienSDFSamples(
#     data_source, train_split, num_samp_per_scene, specs['Class'], load_ram=False, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"], art_per_instance=specs["ArticulationPerInstance"])

sdf_dataset_test = asdf.data.SapienSDFSamples(
    data_source, test_split, num_samp_per_scene, specs['Class'], load_ram=False, articulation=specs["Articulation"], num_atc_parts=specs["NumAtcParts"], art_per_instance=specs["ArticulationPerInstance"], fixed_articulation_type='fixed')
# %%
cat = 'oven'
decoder_test, args, specs = decoders[cat]
decoder_use = decoders_use[cat][0]
decoder_use.load_state_dict(copy.deepcopy(decoder_test.state_dict()))
bidx =  0
all_sdf_data, _ = torch.utils.data.default_collate([sdf_dataset_test[4]])
all_sdf_data[0] = all_sdf_data[0].reshape(2, -1, 5)
err, atc_err, lat_vec, atc_vec = test(args, specs, all_sdf_data, decoder_use)
with torch.no_grad():
    mesh = asdf.mesh.create_mesh(
        decoder_use, lat_vec, None, N=64, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
        return_mesh=True,
    )
mesh.show()
# %%
path = '/home/mil/kawana/workspace/3detr/third_party/A-SDF/asdf_sapien_partial_shapes_gt_test.pkl'
with open(path, 'rb') as f:
    data = pickle.load(f)
# %%
keys = list(data.keys())
sidx = 1
key = keys[sidx]
oidx = 2
okeys = list(data[key].keys())
okey = okeys[oidx]
obdata = data[key][okey]

cat = obdata['category']
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
# %%
eta = 0.025
pos_pts_world = obdata['points'] + obdata['normal'] * 0.025
neg_pts_world = obdata['points'] - obdata['normal'] * 0.025
eta_vec = np.ones((pos_pts_world.shape[0], 1)) * eta

part = np.zeros((pos_pts_world.shape[0], 1))
pos = np.hstack([pos_pts_world, eta_vec, part])
neg = np.hstack([neg_pts_world, -eta_vec, part])
art = torch.zeros(cat_num[cat])
all_sdf_data = [torch.from_numpy(np.vstack([pos, neg])), art[None], torch.zeros_like(art[:1])]
all_sdf_data[0] = all_sdf_data[0].reshape(2, -1, 5)

all_sdf_data2, _ = torch.utils.data.default_collate([sdf_dataset_test[4]])
all_sdf_data2[0] = all_sdf_data2[0].reshape(2, -1, 5)
# %%
print(cat)
decoder_test, args, specs = decoders[cat]
decoder_use = decoders_use[cat][0]
decoder_use.load_state_dict(copy.deepcopy(decoder_test.state_dict()))
bidx =  0

err, atc_err, lat_vec, atc_vec = test(args, specs, all_sdf_data, decoder_use)
with torch.no_grad():
    mesh = asdf.mesh.create_mesh(
        decoder_use, lat_vec, None, N=64, max_batch=int(2 ** 18), atc_vec=atc_vec, do_sup_with_part=specs["TrainWithParts"], specs=specs,
        return_mesh=True,
    )
mesh.show()
# %%

pc = all_sdf_data[0][0, :, :3].cpu().numpy()
fig = milutils.visualizer.get_pcd_plot(pc)
fig.show()
# %%
